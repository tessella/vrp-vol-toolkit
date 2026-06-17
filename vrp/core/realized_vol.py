import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import gamma as gamma_func


class RealizedVolEstimator(ABC):
    TRADING_DAYS = 252  # annualise daily log returns: sqrt(252)

    @abstractmethod
    def estimate(self, prices) -> float:
        """Return annualised realized volatility over the full window."""
        pass

    @abstractmethod
    def estimate_path(self, prices) -> pd.Series:
        """Return rolling annualised realized vol, one value per day."""
        pass

    def __call__(self, prices) -> float:
        return self.estimate(prices)


class SimpleRealizedVol(RealizedVolEstimator):
    """
    Close-to-close realized vol estimator.
    Uses mean of squared log returns (zero-mean assumption).

    estimate()      — scalar over the full window
    estimate_path() — expanding window, one sigma(t) per day
    """

    def estimate(self, prices: pd.Series) -> float:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        variance    = (log_returns ** 2).mean() * self.TRADING_DAYS
        return float(np.sqrt(variance))

    def estimate_path(self, prices: pd.Series) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        sigma_path  = np.sqrt(
            (log_returns ** 2).expanding(min_periods=1).mean() * self.TRADING_DAYS
        )
        sigma_path.iloc[0] = np.nan  # first day has no return
        return sigma_path


class CloseToCloseRV(RealizedVolEstimator):
    """
    Close-to-close with drift correction (sample variance, not zero-mean).
    Slightly less biased than SimpleRealizedVol for trending markets.
    """

    def estimate(self, prices: pd.Series) -> float:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        if len(log_returns) < 2:
            return np.nan
        variance = log_returns.var(ddof=1) * self.TRADING_DAYS
        return float(np.sqrt(variance))

    def estimate_path(self, prices: pd.Series) -> pd.Series:
        log_returns = np.log(prices / prices.shift(1))
        sigma_path = np.sqrt(
            log_returns.expanding(min_periods=2).var(ddof=1) * self.TRADING_DAYS
        )
        return sigma_path


class YangZhangRV(RealizedVolEstimator):
    """
    Yang-Zhang (2000) realized volatility estimator.

    Uses Open, High, Low, Close to achieve ~7x efficiency over close-to-close.
    Drift-independent and handles opening jumps.

    Input: DataFrame with columns 'Open', 'High', 'Low', 'Close'
           (or a Series, in which case falls back to close-to-close).
    """

    def estimate(self, prices) -> float:
        if isinstance(prices, pd.Series):
            return CloseToCloseRV().estimate(prices)

        df = self._validate(prices)
        if df is None or len(df) < 3:
            return np.nan

        n = len(df) - 1  # number of returns

        o = np.log(df["Open"].values[1:] / df["Close"].values[:-1])   # overnight
        c = np.log(df["Close"].values[1:] / df["Open"].values[1:])    # close-to-open (intraday close)
        h = np.log(df["High"].values[1:] / df["Open"].values[1:])
        l = np.log(df["Low"].values[1:] / df["Open"].values[1:])

        # Rogers-Satchell
        rs = (h * (h - c) + l * (l - c)).mean()

        # Overnight variance
        sigma_o = o.var(ddof=1)
        # Close-to-open variance
        sigma_c = c.var(ddof=1)

        k = 0.34 / (1.34 + (n + 1) / (n - 1))

        sigma2 = sigma_o + k * sigma_c + (1 - k) * rs
        sigma2 = max(sigma2, 0.0)
        return float(np.sqrt(sigma2 * self.TRADING_DAYS))

    def estimate_path(self, prices) -> pd.Series:
        if isinstance(prices, pd.Series):
            return CloseToCloseRV().estimate_path(prices)

        df = self._validate(prices)
        if df is None:
            return pd.Series(dtype=float)

        results = pd.Series(np.nan, index=df.index)
        for i in range(2, len(df)):
            sub = df.iloc[:i + 1]
            results.iloc[i] = self.estimate(sub)
        return results

    @staticmethod
    def _validate(df):
        required = {"Open", "High", "Low", "Close"}
        if not isinstance(df, pd.DataFrame) or not required.issubset(df.columns):
            return None
        return df.dropna(subset=list(required))


class GarmanKlassRV(RealizedVolEstimator):
    """
    Garman-Klass (1980) realized volatility estimator.

    Uses OHLC data. ~5x efficiency over close-to-close.
    Assumes no drift (zero-mean) and continuous trading.

    Input: DataFrame with 'Open', 'High', 'Low', 'Close'
           (or a Series — falls back to close-to-close).
    """

    def estimate(self, prices) -> float:
        if isinstance(prices, pd.Series):
            return CloseToCloseRV().estimate(prices)

        df = self._validate(prices)
        if df is None or len(df) < 2:
            return np.nan

        h = np.log(df["High"].values / df["Low"].values)
        c = np.log(df["Close"].values / df["Open"].values)

        gk = 0.5 * h ** 2 - (2 * np.log(2) - 1) * c ** 2
        variance = gk.mean() * self.TRADING_DAYS
        return float(np.sqrt(max(variance, 0.0)))

    def estimate_path(self, prices) -> pd.Series:
        if isinstance(prices, pd.Series):
            return CloseToCloseRV().estimate_path(prices)

        df = self._validate(prices)
        if df is None:
            return pd.Series(dtype=float)

        h = np.log(df["High"] / df["Low"])
        c = np.log(df["Close"] / df["Open"])
        gk = 0.5 * h ** 2 - (2 * np.log(2) - 1) * c ** 2

        sigma_path = np.sqrt(
            gk.expanding(min_periods=1).mean() * self.TRADING_DAYS
        )
        return sigma_path

    @staticmethod
    def _validate(df):
        required = {"Open", "High", "Low", "Close"}
        if not isinstance(df, pd.DataFrame) or not required.issubset(df.columns):
            return None
        return df.dropna(subset=list(required))


class ParkinsonRV(RealizedVolEstimator):
    """
    Parkinson (1980) range-based estimator. Uses only High and Low.
    ~5x efficiency over close-to-close. Assumes continuous trading, zero drift.

    Input: DataFrame with 'High', 'Low' (or Series — falls back to close-to-close).
    """

    def estimate(self, prices) -> float:
        if isinstance(prices, pd.Series):
            return CloseToCloseRV().estimate(prices)

        df = self._validate(prices)
        if df is None or len(df) < 1:
            return np.nan

        hl = np.log(df["High"].values / df["Low"].values)
        variance = (hl ** 2).mean() / (4 * np.log(2)) * self.TRADING_DAYS
        return float(np.sqrt(max(variance, 0.0)))

    def estimate_path(self, prices) -> pd.Series:
        if isinstance(prices, pd.Series):
            return CloseToCloseRV().estimate_path(prices)

        df = self._validate(prices)
        if df is None:
            return pd.Series(dtype=float)

        hl = np.log(df["High"] / df["Low"])
        sigma_path = np.sqrt(
            (hl ** 2).expanding(min_periods=1).mean()
            / (4 * np.log(2)) * self.TRADING_DAYS
        )
        return sigma_path

    @staticmethod
    def _validate(df):
        required = {"High", "Low"}
        if not isinstance(df, pd.DataFrame) or not required.issubset(df.columns):
            return None
        return df.dropna(subset=list(required))


class RFSVPredictor(RealizedVolEstimator):
    """
    Rough Fractional Stochastic Volatility predictor.

    Implements the RFSV prediction formula (Gatheral, Jaisson, Rosenbaum 2018):

        E[log sigma^2_{t+delta} | F_t] = cos(H*pi)/pi * delta^{H+1/2}
            * sum_{k=0}^{N} log(sigma^2_{t-k}) / ((k+delta+1/2)(k+1/2)^{H+1/2})

    Optionally includes OHLC-based daily variance (Garman-Klass) and
    Zumbach/QRH correction for trending-market vol boost.
    """

    def __init__(self, H: float = 0.14, lookback: int = 250,
                 default_horizon: int = 21, use_ohlc: bool = False,
                 zumbach: bool = True, zumbach_lambda: float = 0.3):
        self.H = H
        self.lookback = lookback
        self.default_horizon = default_horizon
        self.use_ohlc = use_ohlc
        self.zumbach = zumbach
        self.zumbach_lambda = zumbach_lambda
        self.nu = None  # estimated on-the-fly or via fit()
        self._intraday_rv = None  # optional pre-computed daily RV series

    _NU_MAX = 1.5  # cap vol-of-vol to prevent log-normal correction blow-up
    _SMOOTH_WINDOW = 5  # rolling window for close-to-close variance proxy

    def set_intraday_rv(self, rv_series: pd.Series):
        """Load pre-computed daily realized variance from intraday data.
        Expects a Series indexed by date with daily variance values (NOT annualized)."""
        self._intraday_rv = rv_series

    # ------------------------------------------------------------------
    #  Daily variance proxy
    # ------------------------------------------------------------------
    def _daily_log_variance(self, prices) -> np.ndarray:
        if self._intraday_rv is not None:
            series = prices["Close"] if isinstance(prices, pd.DataFrame) else prices
            idx = series.index
            rv_aligned = self._intraday_rv.reindex(idx).dropna()
            return np.log(np.clip(rv_aligned.values, 1e-20, None))
        if self.use_ohlc and isinstance(prices, pd.DataFrame):
            return self._gk_daily_log_variance(prices)
        series = prices["Close"] if isinstance(prices, pd.DataFrame) else prices
        log_ret = np.log(series / series.shift(1))
        rolling_var = (log_ret ** 2).rolling(
            window=self._SMOOTH_WINDOW, min_periods=1
        ).mean()
        daily_var = np.clip(rolling_var.dropna().values, 1e-20, None)
        return np.log(daily_var)

    @staticmethod
    def _gk_daily_log_variance(ohlc_df) -> np.ndarray:
        h = np.log(ohlc_df["High"].values / ohlc_df["Open"].values)
        l = np.log(ohlc_df["Low"].values / ohlc_df["Open"].values)
        c = np.log(ohlc_df["Close"].values / ohlc_df["Open"].values)
        gk = 0.5 * h ** 2 + 0.5 * l ** 2 - (2 * np.log(2) - 1) * c ** 2
        gk = np.clip(gk, 1e-20, None)
        return np.log(gk)

    # ------------------------------------------------------------------
    #  RFSV kernel
    # ------------------------------------------------------------------
    def _kernel_weights(self, delta: float, N: int) -> np.ndarray:
        k = np.arange(N, dtype=float)
        return 1.0 / ((k + delta + 0.5) * (k + 0.5) ** (self.H + 0.5))

    # ------------------------------------------------------------------
    #  Log-normal correction
    # ------------------------------------------------------------------
    def _log_normal_correction(self, delta: float, nu: float) -> float:
        H = self.H
        c = (np.sin(np.pi * (0.5 - H)) * gamma_func(1.5 - H) ** 2
             / (np.pi * (0.5 - H) * gamma_func(2 - 2 * H)))
        return 2 * nu ** 2 * c * delta ** (2 * H)

    # ------------------------------------------------------------------
    #  Zumbach / QRH correction
    # ------------------------------------------------------------------
    def _zumbach_forecast(self, prices) -> float:
        series = prices["Close"] if isinstance(prices, pd.DataFrame) else prices
        log_ret = np.log(series / series.shift(1)).dropna().values
        N = min(len(log_ret), self.lookback)
        log_ret = log_ret[-N:]

        k = np.arange(N, dtype=float)
        frac_w = (k + 0.5) ** (self.H - 0.5) / gamma_func(self.H + 0.5)
        frac_w = frac_w[::-1]
        frac_w /= frac_w.sum()

        Z = np.sum(frac_w * log_ret)

        a, b, c = 0.043, 0.74, 0.55
        return a * (Z - b) ** 2 + c

    # ------------------------------------------------------------------
    #  Core forecast
    # ------------------------------------------------------------------
    def forecast(self, prices, horizon_days: int) -> float:
        log_var = self._daily_log_variance(prices)
        log_var = log_var[np.isfinite(log_var)]

        if len(log_var) < 5:
            return np.nan

        # nu scaling: model has log σ² = 2νW^H, so ν = std(Δ log_var) / 2
        nu_raw = (self.nu if self.nu is not None
                  else float(np.std(np.diff(log_var))) / 2.0)
        nu = min(nu_raw, self._NU_MAX)
        N = min(len(log_var), self.lookback)
        history = log_var[-N:]

        delta = float(horizon_days)
        weights = self._kernel_weights(delta, N)

        # Normalized weighted prediction (corrects for finite lookback truncation)
        log_var_pred = np.sum(weights * history[::-1]) / np.sum(weights)
        correction = self._log_normal_correction(delta, nu)
        var_forecast = np.exp(log_var_pred + correction)
        sigma_rfsv = np.sqrt(var_forecast * self.TRADING_DAYS)

        if not np.isfinite(sigma_rfsv):
            return np.nan

        if not self.zumbach:
            return float(sigma_rfsv)

        sigma2_qrh = self._zumbach_forecast(prices)
        daily_var = np.exp(log_var)
        long_run_var = np.mean(daily_var[-N:]) * self.TRADING_DAYS
        sigma_qrh = np.sqrt(sigma2_qrh * long_run_var)

        lam = self.zumbach_lambda
        return float((1 - lam) * sigma_rfsv + lam * sigma_qrh)

    # ------------------------------------------------------------------
    #  ABC interface
    # ------------------------------------------------------------------
    def estimate(self, prices) -> float:
        return self.forecast(prices, self.default_horizon)

    def estimate_path(self, prices) -> pd.Series:
        return SimpleRealizedVol().estimate_path(prices)

    # ------------------------------------------------------------------
    #  Forecast path (for backtesting the predictor)
    # ------------------------------------------------------------------
    def forecast_path(self, prices, horizon_days: int,
                      min_history: int = 60) -> pd.Series:
        if isinstance(prices, pd.DataFrame):
            idx = prices.index
        else:
            idx = prices.index

        results = pd.Series(np.nan, index=idx)
        for i in range(min_history, len(idx)):
            subset = prices.iloc[:i + 1]
            results.iloc[i] = self.forecast(subset, horizon_days)
        return results

    # ------------------------------------------------------------------
    #  Hurst estimation
    # ------------------------------------------------------------------
    def fit(self, prices) -> dict:
        log_var = self._daily_log_variance(prices)
        log_var = log_var[np.isfinite(log_var)]

        if len(log_var) < 200:
            return {"H": self.H, "nu": 0.0, "n_obs": len(log_var),
                    "r_squared": 0.0, "warning": "insufficient data"}

        lags = [1, 2, 5, 10, 20, 40, 60, 100]
        lags = [l for l in lags if l < len(log_var) // 2]
        m_q2, m_q1 = [], []
        for delta in lags:
            inc = log_var[delta:] - log_var[:-delta]
            m_q2.append(np.mean(inc ** 2))
            m_q1.append(np.mean(np.abs(inc)))

        log_lags = np.log(np.array(lags, dtype=float))
        log_m2 = np.log(np.array(m_q2))
        slope2, intercept2 = np.polyfit(log_lags, log_m2, 1)
        H_q2 = slope2 / 2.0

        log_m1 = np.log(np.array(m_q1))
        slope1, _ = np.polyfit(log_lags, log_m1, 1)
        H_q1 = slope1

        ss_res = np.sum((log_m2 - (slope2 * log_lags + intercept2)) ** 2)
        ss_tot = np.sum((log_m2 - np.mean(log_m2)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        nu = min(float(np.std(np.diff(log_var))) / 2.0, self._NU_MAX)

        H_est = float(np.clip(H_q2, 0.01, 0.49))
        self.H = H_est
        self.nu = nu

        result = {"H": H_est, "H_q1": float(np.clip(H_q1, 0.01, 0.49)),
                  "nu": nu, "n_obs": len(log_var), "r_squared": r_squared}

        if abs(H_q2 - H_q1) > 0.05:
            result["warning"] = (f"monofractal check: H_q2={H_q2:.3f} vs "
                                 f"H_q1={H_q1:.3f} — process may not be monofractal")
        return result


def estimate_realized_vol(prices, start: str, end: str,
                          estimator: RealizedVolEstimator = None) -> float:
    """
    Estimate annualised realized vol over [start, end] using the given estimator.
    Defaults to SimpleRealizedVol if no estimator is provided.
    """
    if estimator is None:
        estimator = SimpleRealizedVol()

    subset = prices.loc[start:end]
    if isinstance(subset, pd.DataFrame) and subset.empty:
        raise ValueError(f"No price data found between {start} and {end}.")
    if isinstance(subset, pd.Series) and subset.empty:
        raise ValueError(f"No price data found between {start} and {end}.")

    return estimator.estimate(subset)
