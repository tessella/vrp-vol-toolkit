"""
Variance-risk-premium signal construction.

Signals are built strictly point-in-time: sigma_hat (the realized-vol
forecast) is computed using only prices available on or before trade_date,
from a backward-looking window (configurable via lookback_days). This is the
information set you actually have in real time, so the signals can be used
directly in a walk-forward backtest without look-ahead bias.

Three signal flavours are provided:
  VolStrategy      — cash-gamma-weighted realized vol vs the implied surface
  ClassicVRPSignal — textbook scalar RV - IV
  RFSVSignal       — horizon-matched rough-vol (RFSV) forecast vs IV
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm
from datetime import date

from vrp.core.models import PricingModel, BlackScholesModel
from vrp.core.surface import ImpliedVolSurface
from vrp.core.realized_vol import SimpleRealizedVol, RealizedVolEstimator, RFSVPredictor


class VolStrategy:
    """
    Cash-gamma-weighted VRP signal.

    sigma_hat is computed from a backward-looking window of realized vol
    ending on trade_date, weighted by cash gamma at each historical
    observation.  Only information available at decision time is used, so the
    signal is point-in-time and free of look-ahead bias.
    """

    def __init__(self,
                 surface:       ImpliedVolSurface,
                 prices:        pd.Series,
                 trade_date:    date,
                 model:         PricingModel = None,
                 lookback_days: int = 60,
                 rv_estimator:  RealizedVolEstimator = None):
        """
        Parameters
        ----------
        lookback_days : number of trading days of price history to use
                        for the realized vol estimate.  ~60 days ≈ 3
                        calendar months.  Must be >= 2.
        rv_estimator  : RealizedVolEstimator instance. Defaults to
                        SimpleRealizedVol (close-to-close, zero-mean).
                        Swap in YangZhangRV or GarmanKlassRV for
                        OHLC-based estimation.
        """
        self.surface       = surface
        self.prices        = prices
        self.trade_date    = trade_date
        self._model        = model or BlackScholesModel()
        self.lookback_days = lookback_days
        self.rv_estimator  = rv_estimator or SimpleRealizedVol()
        self.results_df    = None

    # ------------------------------------------------------------------
    #  Vectorised batch sigma-hat (backward-looking only)
    # ------------------------------------------------------------------
    def _sigma_hat_batch(
        self,
        strikes:        np.ndarray,
        expiry:         date,
        r:              float,
        sigma_path:     pd.Series,
        sigma_impl_vec: np.ndarray,
    ) -> np.ndarray:
        """
        Compute sigma_hat for ALL strikes in one vectorised pass.

        `sigma_path` and prices here only cover the backward-looking window
        ending on trade_date.  Cash gamma weights still use time-to-expiry
        (tau) from each historical date to the option's expiry — that's
        observable on trade_date.
        """
        T_years = (expiry - self.trade_date).days / 365.0
        if T_years <= 0:
            return np.full(len(strikes), np.nan)

        # Only use prices UP TO trade_date (backward-looking window)
        end_date = str(self.trade_date)
        all_prior = self.prices.loc[:end_date]
        subset = all_prior.iloc[-self.lookback_days:]

        if len(subset) < 2:
            return np.full(len(strikes), np.nan)

        sigma_t_series = sigma_path.reindex(subset.index)

        S_arr   = subset.values
        sig_arr = sigma_t_series.values
        dates   = np.array([
            t.date() if hasattr(t, "date") else t
            for t in subset.index
        ])
        # tau is still time from each historical date to expiry —
        # this is known at trade_date and is not look-ahead
        tau_arr = np.array(
            [(expiry - d).days / 365.0 for d in dates]
        )

        valid_t = (tau_arr > 0) & np.isfinite(sig_arr)
        if not valid_t.any():
            return np.full(len(strikes), np.nan)

        S_v   = S_arr[valid_t]
        tau_v = tau_arr[valid_t]
        sig_v = sig_arr[valid_t]

        K_arr = strikes

        S2d   = S_v[:, None]
        tau2d = tau_v[:, None]
        K2d   = K_arr[None, :]
        s02d  = sigma_impl_vec[None, :]

        sqrt_tau = np.sqrt(tau2d)
        d1 = (np.log(S2d / K2d) + (r + 0.5 * s02d**2) * tau2d) / (s02d * sqrt_tau)

        cash_gamma = (
            norm.pdf(d1) / (s02d * sqrt_tau)
        ) * S2d

        cash_gamma = np.where(cash_gamma > 0, cash_gamma, 0.0)

        sig_v2d = sig_v[:, None] ** 2

        numerator   = (sig_v2d * cash_gamma).sum(axis=0)
        denominator = cash_gamma.sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            sigma_hat = np.sqrt(
                np.where(denominator > 0, numerator / denominator, np.nan)
            )

        sigma_hat = np.where(np.isfinite(sigma_impl_vec), sigma_hat, np.nan)
        return sigma_hat

    # ------------------------------------------------------------------
    #  Public run()
    # ------------------------------------------------------------------
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Same interface as VolStrategy.run().

        Returns
        -------
        sigma_hat_df : DataFrame — [expiry, T_years, strike, sigma_impl, sigma_hat]
        signal_df    : DataFrame — [expiry, T_years, strike, vol_diff]
        """
        expiries  = self.surface.get_expiries(self.trade_date)
        estimator = self.rv_estimator
        rows      = []

        # Backward-looking price window for realized vol
        end_date  = str(self.trade_date)
        all_prior = self.prices.loc[:end_date]
        lookback  = all_prior.iloc[-self.lookback_days:]

        if len(lookback) < 2:
            raise ValueError(
                f"Not enough price history before {self.trade_date}: "
                f"need {self.lookback_days} days, have {len(lookback)}"
            )

        sigma_path = estimator.estimate_path(lookback)

        for expiry in tqdm(expiries, desc="Expiries"):
            T_years = (expiry - self.trade_date).days / 365.0
            r       = self.surface._get_r(self.trade_date, expiry)

            strikes = np.array(self.surface.get_strikes(self.trade_date, expiry))

            sigma_impl_vec = np.array([
                self.surface.get_mid_vol(self.trade_date, expiry, K)
                for K in strikes
            ], dtype=float)

            sigma_hat_vec = self._sigma_hat_batch(
                strikes, expiry, r, sigma_path, sigma_impl_vec
            )

            for K, s_impl, s_hat in zip(strikes, sigma_impl_vec, sigma_hat_vec):
                rows.append({
                    "expiry":     expiry,
                    "T_years":    round(T_years, 4),
                    "strike":     K,
                    "sigma_impl": s_impl,
                    "sigma_hat":  s_hat,
                    "vol_diff":   s_hat - s_impl,
                })

        self.results_df = pd.DataFrame(rows)

        sigma_hat_df = self.results_df[
            ["expiry", "T_years", "strike", "sigma_impl", "sigma_hat"]
        ].copy()

        signal_df = self.results_df[
            ["expiry", "T_years", "strike", "vol_diff"]
        ].copy()

        return sigma_hat_df, signal_df


class ClassicVRPSignal:
    """
    Standard Variance Risk Premium signal: RV - IV at each (expiry, strike).

    vol_diff = sigma_realized - sigma_implied
    Positive → realized vol > implied → option was cheap → BUY
    Negative → realized vol < implied → option was rich  → SELL

    This is the textbook VRP. It uses a single scalar realized vol
    estimate over the backward-looking window, broadcast to all strikes.
    Simpler than the cash-gamma-weighted estimator but a useful parallel
    signal for the ML model.
    """

    def __init__(self,
                 surface:       ImpliedVolSurface,
                 prices:        pd.Series,
                 trade_date:    date,
                 model:         PricingModel = None,
                 lookback_days: int = 60,
                 rv_estimator:  RealizedVolEstimator = None):
        self.surface       = surface
        self.prices        = prices
        self.trade_date    = trade_date
        self._model        = model or BlackScholesModel()
        self.lookback_days = lookback_days
        self.rv_estimator  = rv_estimator or SimpleRealizedVol()
        self.results_df    = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        sigma_hat_df : DataFrame — [expiry, T_years, strike, sigma_impl, sigma_hat]
        signal_df    : DataFrame — [expiry, T_years, strike, vol_diff]
        """
        expiries = self.surface.get_expiries(self.trade_date)
        rows = []

        # Single scalar RV estimate from the backward-looking window
        end_date = str(self.trade_date)
        all_prior = self.prices.loc[:end_date]
        lookback = all_prior.iloc[-self.lookback_days:]

        if len(lookback) < 2:
            raise ValueError(
                f"Not enough price history before {self.trade_date}: "
                f"need {self.lookback_days} days, have {len(lookback)}"
            )

        sigma_realized = self.rv_estimator.estimate(lookback)

        for expiry in tqdm(expiries, desc="Expiries (VRP)"):
            T_years = (expiry - self.trade_date).days / 365.0
            strikes = np.array(self.surface.get_strikes(self.trade_date, expiry))

            for K in strikes:
                sigma_impl = self.surface.get_mid_vol(
                    self.trade_date, expiry, K
                )
                rows.append({
                    "expiry":     expiry,
                    "T_years":    round(T_years, 4),
                    "strike":     K,
                    "sigma_impl": sigma_impl,
                    "sigma_hat":  sigma_realized,
                    "vol_diff":   sigma_realized - sigma_impl,
                })

        self.results_df = pd.DataFrame(rows)

        sigma_hat_df = self.results_df[
            ["expiry", "T_years", "strike", "sigma_impl", "sigma_hat"]
        ].copy()

        signal_df = self.results_df[
            ["expiry", "T_years", "strike", "vol_diff"]
        ].copy()

        return sigma_hat_df, signal_df


class RFSVSignal:
    """
    RFSV-based VRP signal with horizon-matched vol forecasts.

    For each expiry, computes sigma_hat = RFSV_forecast(delta=DTE) so that
    short-dated and long-dated options get different vol predictions.
    Structurally identical to ClassicVRPSignal in output format.
    """

    def __init__(self,
                 surface:       ImpliedVolSurface,
                 prices:        pd.Series,
                 trade_date:    date,
                 model:         PricingModel = None,
                 lookback_days: int = 250,
                 rfsv_H:        float = 0.14,
                 use_ohlc:      bool = False,
                 zumbach:       bool = True,
                 zumbach_lambda: float = 0.3):
        self.surface       = surface
        self.prices        = prices
        self.trade_date    = trade_date
        self._model        = model or BlackScholesModel()
        self.lookback_days = lookback_days
        self.predictor     = RFSVPredictor(
            H=rfsv_H, lookback=lookback_days,
            use_ohlc=use_ohlc, zumbach=zumbach,
            zumbach_lambda=zumbach_lambda,
        )
        self.results_df = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        expiries = self.surface.get_expiries(self.trade_date)
        rows = []

        end_date = str(self.trade_date)
        all_prior = self.prices.loc[:end_date]
        lookback = all_prior.iloc[-self.lookback_days:]

        if len(lookback) < 10:
            raise ValueError(
                f"Not enough price history before {self.trade_date}: "
                f"need {self.lookback_days} days, have {len(lookback)}"
            )

        for expiry in tqdm(expiries, desc="Expiries (RFSV)"):
            T_years = (expiry - self.trade_date).days / 365.0
            if T_years <= 0:
                continue

            dte = (expiry - self.trade_date).days
            delta_trading = max(1, int(dte * 252 / 365))
            sigma_hat = self.predictor.forecast(lookback, delta_trading)

            strikes = np.array(self.surface.get_strikes(self.trade_date, expiry))
            for K in strikes:
                sigma_impl = self.surface.get_mid_vol(
                    self.trade_date, expiry, K
                )
                rows.append({
                    "expiry":     expiry,
                    "T_years":    round(T_years, 4),
                    "strike":     K,
                    "sigma_impl": sigma_impl,
                    "sigma_hat":  sigma_hat,
                    "vol_diff":   sigma_hat - sigma_impl if not np.isnan(sigma_hat) else np.nan,
                })

        self.results_df = pd.DataFrame(rows)

        sigma_hat_df = self.results_df[
            ["expiry", "T_years", "strike", "sigma_impl", "sigma_hat"]
        ].copy()

        signal_df = self.results_df[
            ["expiry", "T_years", "strike", "vol_diff"]
        ].copy()

        return sigma_hat_df, signal_df


def get_trading_signals(signal_df: pd.DataFrame, eps: float = 0.01) -> pd.DataFrame:
    """Split a signal frame into BUY/SELL legs on |vol_diff| > eps.

    vol_diff > 0  → realized > implied → option cheap → BUY
    vol_diff < 0  → realized < implied → option rich  → SELL
    """
    buy  = signal_df[signal_df["vol_diff"] >  eps].copy()
    sell = signal_df[signal_df["vol_diff"] < -eps].copy()
    buy["signal"]  = "BUY"
    sell["signal"] = "SELL"

    signals = (
        pd.concat([buy, sell])
        .sort_values("vol_diff", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Buy signals:  {len(buy)}")
    print(f"Sell signals: {len(sell)}")
    print(f"Total signals (|vol_diff| > {eps:.2%}): {len(signals)} / {len(signal_df)} points")
    return signals
