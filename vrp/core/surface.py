import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date
import warnings
from scipy.stats import norm
from scipy.interpolate import CubicSpline

from vrp.core.models import BlackScholesModel


def _bs_price_vec(S, K, r, T, sigma, is_call):
    """
    Vectorised Black-Scholes price.
    All arguments are NumPy arrays of the same shape.
    is_call: boolean array (True = call, False = put)
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc = np.exp(-r * T)

    call_price = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
    put_price  = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)

    return np.where(is_call, call_price, put_price)


def _solve_iv_vec(S, K, r, T, V, is_call,
                  lower=1e-4, upper=3.0, n_iter=50, tol=1e-6):
    """
    Vectorised bisection for implied vol.
    Returns array of implied vols; entries where no root exists are np.nan.
    Bounds [1e-4, 3.0], np.nan when f(lower)*f(upper) > 0 (no root in range).
    Note: f_hi is not tracked after initialisation — bisection only needs
    the sign of f_lo to decide which half to keep.
    """
    S, K, r, T, V = (np.asarray(x, dtype=float) for x in (S, K, r, T, V))
    is_call = np.asarray(is_call, dtype=bool)

    lo = np.full_like(S, lower)
    hi = np.full_like(S, upper)

    f_lo = _bs_price_vec(S, K, r, T, lo, is_call) - V
    f_hi = _bs_price_vec(S, K, r, T, hi, is_call) - V

    # Mark np.nan where no root exists in [lower, upper]
    no_root = f_lo * f_hi > 0
    lo[no_root] = np.nan
    hi[no_root] = np.nan

    for _ in range(n_iter):
        mid   = (lo + hi) * 0.5
        f_mid = _bs_price_vec(S, K, r, T, mid, is_call) - V
        # Move lower bound up where f_mid has same sign as f_lo
        same_sign = f_mid * f_lo > 0
        lo   = np.where(same_sign, mid, lo)
        f_lo = np.where(same_sign, f_mid, f_lo)
        hi   = np.where(~same_sign, mid, hi)
        # f_hi not needed — bisection only uses sign of f_lo

        # Early exit once all active intervals are tight enough
        active = ~no_root
        if active.any() and np.nanmax(np.abs(hi[active] - lo[active])) < tol:
            break

    result = (lo + hi) * 0.5
    result[no_root] = np.nan
    return result

class ImpliedVolSurface:
    """
    Build a vol surface: (trade_date, expiry, strike) -> (bid_vol, mid_vol, offer_vol).
    """

    DAY_COUNT        = 365.0
    MIN_OPTION_PRICE = 1e-4
    MIN_T            = 1.0 / 365.0

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)

        # Normalise cp_flag to lowercase ('C'/'P' -> 'c'/'p')
        self.df["cp_flag"] = self.df["cp_flag"].str.strip().str.lower()

        # mid_price may be missing if load_options_data didn't assign it
        if "mid_price" not in self.df.columns:
            self.df["mid_price"] = (self.df["best_bid"] + self.df["best_offer"]) * 0.5

        self.df["impl_df"] = np.nan
        self.df["impl_fw"] = np.nan

        # Diagnostic counters accumulated across both build steps
        self.filter_report = {
            "input_rows":                      len(self.df),
            "groups_skipped_no_calls_or_puts": 0,
            "groups_skipped_no_parity_pair":   0,
            "groups_skipped_bad_impl_df":      0,
            "otm_rows_after_forward_filter":   0,
            "otm_rows_dropped_price_filter":   0,
            "otm_rows_dropped_short_T":        0,
            "solver_skipped":                  0,
            "surface_points_built":            0,
        }

        self._model = BlackScholesModel()
        self.imply_forward_and_df()
        self.surface = self.build_surface()
        self._build_indices()
        self._print_filter_report()

    # ------------------------------------------------------------------
    #  Step 1 — imply forward price and discount factor via put-call parity
    # ------------------------------------------------------------------
    def imply_forward_and_df(self):
        df = self.df
        g = df.groupby(["date", "exdate"])
        call_mask = df.index[df["cp_flag"] == "c"]
        put_mask  = df.index[df["cp_flag"] == "p"]

        groups = g.groups
        for key, group_idx in tqdm(groups.items(), total=len(groups),
                                   desc="Implying forwards & discount factors"):
            see_call = df.loc[
                group_idx.intersection(call_mask), ["strike_price", "mid_price"]
            ].copy()
            see_put = df.loc[
                group_idx.intersection(put_mask), ["strike_price", "mid_price"]
            ].copy()

            if see_call.empty or see_put.empty:
                self.filter_report["groups_skipped_no_calls_or_puts"] += 1
                continue

            see_call.sort_values("strike_price", inplace=True)
            see_call.set_index("strike_price", inplace=True)
            see_put.sort_values("strike_price", inplace=True)
            see_put.set_index("strike_price", inplace=True)

            # Deduplicate strikes — keep the row with the highest mid_price
            if not (see_call.index.is_unique and see_put.index.is_unique):
                see_call = see_call.sort_values("mid_price").loc[
                    ~see_call.index.duplicated(keep="last")
                ]
                see_put = see_put.sort_values("mid_price").loc[
                    ~see_put.index.duplicated(keep="last")
                ]

            common = see_call.index.intersection(see_put.index)
            # put - call: positive above forward, negative below
            xx1 = see_put.loc[common, "mid_price"] - see_call.loc[common, "mid_price"]

            if len(xx1.loc[xx1 > 0]) == 0 or len(xx1.loc[xx1 <= 0]) == 0:
                self.filter_report["groups_skipped_no_parity_pair"] += 1
                continue

            # K_up: lowest strike where put > call (above forward)
            # K_dn: highest strike where call >= put (below forward)
            K_up = xx1.loc[xx1 > 0].index[0]
            K_dn = xx1.loc[xx1 <= 0].index[-1]

            # Put-call parity: P - C = df*(K - F)
            # Using two strikes K_dn and K_up:
            # (P_up - C_up) - (P_dn - C_dn) = df * (K_up - K_dn)
            call_dn = see_call.at[K_dn, "mid_price"]
            put_dn  = see_put.at[K_dn,  "mid_price"]
            call_up = see_call.at[K_up, "mid_price"]
            put_up  = see_put.at[K_up,  "mid_price"]

            impl_df = (put_up + call_dn - put_dn - call_up) / (K_up - K_dn)
            impl_fw = K_up + (call_up - put_up) / impl_df

            # Reject implausible discount factors (must be in (0, 1])
            if not np.isscalar(impl_df) or not (0 < impl_df <= 1.0):
                self.filter_report["groups_skipped_bad_impl_df"] += 1
                continue

            mask = (df["date"] == key[0]) & (df["exdate"] == key[1])
            df.loc[mask, "impl_df"] = impl_df
            df.loc[mask, "impl_fw"] = impl_fw

    # ------------------------------------------------------------------
    #  Step 2 — build the vol surface using vectorised IV solver
    # ------------------------------------------------------------------
    def build_surface(self):
        df = self.df

        # Keep only OTM options (calls above forward, puts below)
        otm = df.loc[
            ((df["cp_flag"] == "c") & (df["strike_price"] >= df["impl_fw"])) |
            ((df["cp_flag"] == "p") & (df["strike_price"] <= df["impl_fw"]))
        ].dropna(subset=["impl_df", "impl_fw"]).reset_index(drop=True)

        self.filter_report["otm_rows_after_forward_filter"] = len(otm)

        # Use self.DAY_COUNT for consistency with _get_r
        otm["T"]  = (otm["exdate"] - otm["date"]).dt.days / self.DAY_COUNT
        otm["r"]  = -np.log(otm["impl_df"]) / otm["T"]
        # S0 = F * df since F = S*e^{rT} => S = F * e^{-rT} = F * impl_df
        otm["S0"] = otm["impl_fw"] * otm["impl_df"]

        # Price filter: all three quotes must be above minimum
        price_mask = (
            (otm["best_bid"]   >= self.MIN_OPTION_PRICE) &
            (otm["best_offer"] >= self.MIN_OPTION_PRICE) &
            (otm["mid_price"]  >= self.MIN_OPTION_PRICE)
        )
        self.filter_report["otm_rows_dropped_price_filter"] = int((~price_mask).sum())
        otm = otm[price_mask]

        # Short-T filter: avoid near-zero expiries causing r -> +/-inf
        t_mask = otm["T"] >= self.MIN_T
        self.filter_report["otm_rows_dropped_short_T"] = int((~t_mask).sum())
        otm = otm[t_mask].reset_index(drop=True)

        # Vectorised IV solver
        is_call = otm["cp_flag"].values == "c"
        S = otm["S0"].values
        K = otm["strike_price"].values
        r = otm["r"].values
        T = otm["T"].values

        bid_vols   = _solve_iv_vec(S, K, r, T, otm["best_bid"].values,   is_call)
        offer_vols = _solve_iv_vec(S, K, r, T, otm["best_offer"].values, is_call)
        mid_vols   = _solve_iv_vec(S, K, r, T, otm["mid_price"].values,  is_call)

        # Skip rows where any vol could not be solved
        valid   = ~(np.isnan(bid_vols) | np.isnan(offer_vols) | np.isnan(mid_vols))
        skipped = int((~valid).sum())

        surface = {}
        for i in np.where(valid)[0]:
            trade_date = otm.at[i, "date"].date()
            expiry     = otm.at[i, "exdate"].date()
            strike     = otm.at[i, "strike_price"]
            surface[(trade_date, expiry, strike)] = (
                float(bid_vols[i]),
                float(mid_vols[i]),
                float(offer_vols[i]),
            )

        self.filter_report["solver_skipped"]       = skipped
        self.filter_report["surface_points_built"] = len(surface)
        return surface

    # ------------------------------------------------------------------
    #  Index building (O(n) once, then O(1) lookups)
    # ------------------------------------------------------------------
    def _build_indices(self, zero_curve: dict = None):
        """
        Build O(1) lookup dicts from the surface and df.

        Parameters
        ----------
        zero_curve : Optional dict {(date, days) → rate_pct} from
                     optionm.zerocd. If provided, _get_r interpolates
                     from this curve instead of using impl_df.
        """
        from collections import defaultdict

        self._dates_sorted = []
        date_exp = defaultdict(set)
        date_exp_strikes = defaultdict(list)

        for (td, exp, K) in self.surface:
            date_exp[td].add(exp)
            date_exp_strikes[(td, exp)].append(K)

        self._dates_sorted = sorted(date_exp.keys())
        self._date_expiries = {td: sorted(exps) for td, exps in date_exp.items()}
        self._date_expiry_strikes = {
            key: sorted(strikes) for key, strikes in date_exp_strikes.items()
        }

        # Forward price index
        self._impl_fw_map = {}
        df = self.df
        valid = df[df["impl_fw"].notna()].drop_duplicates(
            subset=["date", "exdate"], keep="first"
        )
        dates_v = pd.to_datetime(valid["date"]).dt.date.values
        exdates_v = pd.to_datetime(valid["exdate"]).dt.date.values
        impl_fw_v = valid["impl_fw"].values
        for i in range(len(valid)):
            key = (dates_v[i], exdates_v[i])
            self._impl_fw_map[key] = float(impl_fw_v[i])

        # Risk-free rate index
        self._r_map = {}
        if zero_curve:
            import bisect
            from collections import defaultdict

            # Pre-build per-date sorted tenor lists for fast interpolation
            zc_by_date = defaultdict(list)
            for (dt, days) in zero_curve:
                zc_by_date[dt].append(days)
            for dt in zc_by_date:
                zc_by_date[dt].sort()

            for (td, exp) in self._impl_fw_map:
                dte = (exp - td).days
                # Try exact match first
                exact = zero_curve.get((td, dte))
                if exact is not None:
                    self._r_map[(td, exp)] = exact / 100.0
                    continue

                tenors = zc_by_date.get(td)
                if not tenors:
                    continue
                if dte <= tenors[0]:
                    self._r_map[(td, exp)] = zero_curve[(td, tenors[0])] / 100.0
                elif dte >= tenors[-1]:
                    self._r_map[(td, exp)] = zero_curve[(td, tenors[-1])] / 100.0
                else:
                    idx = bisect.bisect_left(tenors, dte)
                    d_lo, d_hi = tenors[idx - 1], tenors[idx]
                    r_lo = zero_curve[(td, d_lo)] / 100.0
                    r_hi = zero_curve[(td, d_hi)] / 100.0
                    alpha = (dte - d_lo) / (d_hi - d_lo)
                    self._r_map[(td, exp)] = r_lo + alpha * (r_hi - r_lo)
        else:
            # Fallback: derive from impl_df if available
            if "impl_df" in df.columns:
                valid_df = df[df["impl_df"].notna()].drop_duplicates(
                    subset=["date", "exdate"], keep="first"
                )
                for i in range(len(valid_df)):
                    d = pd.to_datetime(valid_df.iloc[i]["date"]).date()
                    e = pd.to_datetime(valid_df.iloc[i]["exdate"]).date()
                    impl_df_val = float(valid_df.iloc[i]["impl_df"])
                    T = (e - d).days / self.DAY_COUNT
                    if T > 0 and 0 < impl_df_val <= 1.2:
                        self._r_map[(d, e)] = float(-np.log(impl_df_val) / T)

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    def _print_filter_report(self):
        r = self.filter_report
        print("\n" + "=" * 60)
        print("ImpliedVolSurface — filter / diagnostics report")
        print("=" * 60)
        print(f"  Input rows                          : {r['input_rows']:>8,}")
        print(f"  Groups skipped (no calls or puts)   : {r['groups_skipped_no_calls_or_puts']:>8,}")
        print(f"  Groups skipped (no parity pair)     : {r['groups_skipped_no_parity_pair']:>8,}")
        print(f"  Groups skipped (bad impl_df)        : {r['groups_skipped_bad_impl_df']:>8,}")
        print(f"  OTM rows after forward filter       : {r['otm_rows_after_forward_filter']:>8,}")
        print(f"  OTM rows dropped (price < threshold): {r['otm_rows_dropped_price_filter']:>8,}")
        print(f"  OTM rows dropped (T too short)      : {r['otm_rows_dropped_short_T']:>8,}")
        print(f"  Rows skipped by solver              : {r['solver_skipped']:>8,}")
        print(f"  Surface points built                : {r['surface_points_built']:>8,}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    #  Public accessors
    # ------------------------------------------------------------------
    def get_vol(self, trade_date: date, expiry: date, strike: float):
        return self.surface.get(
            (trade_date, expiry, strike), (np.nan, np.nan, np.nan)
        )

    def get_bid_vol(self, trade_date: date, expiry: date, strike: float) -> float:
        return self.get_vol(trade_date, expiry, strike)[0]

    def get_mid_vol(self, trade_date: date, expiry: date, strike: float) -> float:
        return self.get_vol(trade_date, expiry, strike)[1]

    def get_offer_vol(self, trade_date: date, expiry: date, strike: float) -> float:
        return self.get_vol(trade_date, expiry, strike)[2]

    def get_dates(self):
        return list(self._dates_sorted)

    def get_expiries(self, trade_date: date):
        return list(self._date_expiries.get(trade_date, []))

    def get_strikes(self, trade_date: date, expiry: date):
        return list(self._date_expiry_strikes.get((trade_date, expiry), []))

    def _get_r(self, trade_date: date, expiry: date) -> float:
        r = self._r_map.get((trade_date, expiry))
        if r is not None:
            return r
        warnings.warn(
            f"_get_r: no rate for ({trade_date}, {expiry}) — returning 0.0",
            RuntimeWarning, stacklevel=2,
        )
        return 0.0
    
class InterpolatedVolSurface:
    """
    Wraps an ImpliedVolSurface and adds interpolation for (strike, expiry)
    pairs that are not on the original discrete surface.

    Strike interpolation : cubic spline on total variance w = σ²T vs
                           log-moneyness k = log(K/F), per slice.
    Expiry interpolation : linear interpolation in total variance w across
                           the two nearest expiry slices (calendar-arbitrage-
                           free by construction).

    get_mid_vol(trade_date, expiry, strike) is the single entry point —
    it first checks the original surface and only interpolates if the point
    is missing.
    """

    def __init__(self, surface: ImpliedVolSurface):
        self.surface = surface
        # Cache key includes trade_date so stale entries across dates are not an issue
        self._spline_cache: dict = {}

    @property
    def df(self):
        return self.surface.df

    # ------------------------------------------------------------------
    #  Public interface — drop-in replacement for ImpliedVolSurface
    # ------------------------------------------------------------------

    def get_mid_vol(self, trade_date: date, expiry: date, strike: float) -> float:
        """
        Return mid vol for (trade_date, expiry, strike).
        Uses the original surface point if available, otherwise interpolates.
        """
        # Fast path — point exists on the original surface
        # get_mid_vol returns np.nan (not KeyError) for missing points
        vol = self.surface.get_mid_vol(trade_date, expiry, strike)
        if not np.isnan(vol) and vol > 0:
            return vol

        # Slow path — interpolate
        return self._interpolate(trade_date, expiry, strike)

    def get_bid_vol(self, trade_date: date, expiry: date, strike: float) -> float:
        return self.surface.get_bid_vol(trade_date, expiry, strike)

    def get_offer_vol(self, trade_date: date, expiry: date, strike: float) -> float:
        return self.surface.get_offer_vol(trade_date, expiry, strike)

    def get_dates(self):
        return self.surface.get_dates()

    def get_expiries(self, trade_date: date):
        return self.surface.get_expiries(trade_date)

    def get_strikes(self, trade_date: date, expiry: date):
        return self.surface.get_strikes(trade_date, expiry)

    def _get_r(self, trade_date: date, expiry: date) -> float:
        return self.surface._get_r(trade_date, expiry)

    # ------------------------------------------------------------------
    #  Interpolation
    # ------------------------------------------------------------------

    def _interpolate(self, trade_date: date, expiry: date, strike: float) -> float:
        """
        Two-step interpolation:
          1. Build a spline w(k) for each bracketing expiry slice.
          2. Linearly interpolate w across the two nearest expiries.
        Then convert back to σ = sqrt(w / T).
        """
        expiries = self.surface.get_expiries(trade_date)

        if len(expiries) == 0:
            return np.nan

        T_target = (expiry - trade_date).days / 365.0
        if T_target <= 0:
            return np.nan

        T_all = np.array([(e - trade_date).days / 365.0 for e in expiries])
        idx_above = np.searchsorted(T_all, T_target)

        if idx_above == 0:
            bracket = [expiries[0]]
        elif idx_above >= len(expiries):
            bracket = [expiries[-1]]
        else:
            bracket = [expiries[idx_above - 1], expiries[idx_above]]

        w_values = []
        T_values = []

        for exp in bracket:
            T_slice = (exp - trade_date).days / 365.0
            w = self._eval_spline(trade_date, exp, strike, T_slice)
            if np.isnan(w):
                return np.nan
            w_values.append(w)
            T_values.append(T_slice)

        if len(w_values) == 1:
            w_interp = w_values[0]
            T_interp = T_values[0]
        else:
            T1, T2   = T_values[0], T_values[1]
            w1, w2   = w_values[0], w_values[1]
            alpha    = (T_target - T1) / (T2 - T1)
            w_interp = w1 + alpha * (w2 - w1)
            T_interp = T_target

        if w_interp < 0:
            return np.nan

        return float(np.sqrt(w_interp / T_interp))

    def _eval_spline(self, trade_date: date, expiry: date,
                     strike: float, T_slice: float) -> float:
        """
        Evaluate the total-variance spline w(k) at log-moneyness
        k = log(strike / F) for the given (trade_date, expiry) slice.
        Builds and caches the spline on first call.
        """
        spline, F = self._get_or_build_spline(trade_date, expiry, T_slice)
        if spline is None:
            return np.nan

        k_target = np.log(strike / F)
        k_min, k_max = spline.x[0], spline.x[-1]

        if k_target < k_min or k_target > k_max:
            warnings.warn(
                f"Strike {strike:.1f} is outside spline range "
                f"[{np.exp(k_min)*F:.1f}, {np.exp(k_max)*F:.1f}] — "
                f"flat extrapolation applied.",
                RuntimeWarning, stacklevel=2
            )
            k_target = np.clip(k_target, k_min, k_max)

        w = float(spline(k_target))
        return max(w, 0.0)  # safeguard against tiny negative values from spline wiggle

    def _get_or_build_spline(self, trade_date: date, expiry: date,
                             T_slice: float):
        """
        Build (and cache) a CubicSpline of w = σ²T vs k = log(K/F)
        for the given slice. Returns (spline, F).
        """
        cache_key = (trade_date, expiry)
        if cache_key in self._spline_cache:
            return self._spline_cache[cache_key]

        strikes = self.surface.get_strikes(trade_date, expiry)
        if len(strikes) < 4:
            self._spline_cache[cache_key] = (None, None)
            return None, None

        # Accessing internal df directly — coupled to ImpliedVolSurface internals
        mask = (
            (self.surface.df["date"].dt.date   == trade_date) &
            (self.surface.df["exdate"].dt.date == expiry)
        )
        impl_fw_series = self.surface.df.loc[mask, "impl_fw"].dropna()
        if impl_fw_series.empty:
            self._spline_cache[cache_key] = (None, None)
            return None, None
        F = float(impl_fw_series.iloc[0])

        k_arr = []
        w_arr = []
        for K in strikes:
            vol = self.surface.get_mid_vol(trade_date, expiry, K)
            if np.isnan(vol) or vol <= 0:
                continue
            k_arr.append(np.log(K / F))
            w_arr.append(vol ** 2 * T_slice)

        if len(k_arr) < 4:
            self._spline_cache[cache_key] = (None, None)
            return None, None

        k_arr = np.array(k_arr)
        w_arr = np.array(w_arr)

        sort_idx = np.argsort(k_arr)
        k_arr    = k_arr[sort_idx]
        w_arr    = w_arr[sort_idx]

        # Natural cubic spline — zero second derivative at boundaries
        # prevents wild extrapolation in the wings
        spline = CubicSpline(k_arr, w_arr, bc_type="natural")

        self._spline_cache[cache_key] = (spline, F)
        return spline, F


class OMVolSurface(ImpliedVolSurface):
    """
    Build a vol surface from OptionMetrics data that already contains
    `impl_volatility` and `forward_price` per option row.

    Skips:
      - Bisection IV solving (uses OM's impl_volatility directly)
      - Put-call parity forward/discount-factor derivation (uses fwdprd
        forwards directly, risk-free rate from zero-coupon curve)

    The resulting surface dict and public API are identical to
    ImpliedVolSurface, so InterpolatedVolSurface wraps this transparently.
    """

    def __init__(self, df: pd.DataFrame, zero_curve: dict = None):
        """
        Parameters
        ----------
        df         : Options DataFrame with impl_volatility and forward_price.
        zero_curve : Dict {(date, days) → rate_pct} from optionm.zerocd.
                     If provided, _get_r uses this instead of impl_df.
        """
        self.df = df.copy().reset_index(drop=True)
        self.df["cp_flag"] = self.df["cp_flag"].str.strip().str.lower()

        if "mid_price" not in self.df.columns:
            self.df["mid_price"] = (
                self.df["best_bid"] + self.df["best_offer"]
            ) * 0.5

        self.df["impl_fw"] = np.nan

        self.filter_report = {
            "input_rows":                      len(self.df),
            "groups_skipped_no_calls_or_puts": 0,
            "groups_skipped_no_parity_pair":   0,
            "groups_skipped_bad_impl_df":      0,
            "otm_rows_after_forward_filter":   0,
            "otm_rows_dropped_price_filter":   0,
            "otm_rows_dropped_short_T":        0,
            "solver_skipped":                  0,
            "surface_points_built":            0,
        }
        self._model = BlackScholesModel()

        if "impl_volatility" not in self.df.columns:
            raise ValueError(
                "OMVolSurface requires 'impl_volatility' column. "
                "Use ImpliedVolSurface for raw data without pre-computed IVs."
            )

        # Step 1: Set impl_fw directly from forward_price column
        if "forward_price" in self.df.columns:
            self.df["impl_fw"] = self.df["forward_price"]
            n_fw = self.df["impl_fw"].notna().sum()
            if n_fw == 0:
                raise ValueError(
                    "forward_price column is all NaN. Merge fwdprd "
                    "forwards into the options df before constructing."
                )
        else:
            raise ValueError(
                "OMVolSurface requires 'forward_price' column. "
                "Merge fwdprd forwards into the options df first."
            )

        # Step 2: Build surface from pre-computed IVs
        self.surface = self._build_from_precomputed()
        self._build_indices(zero_curve=zero_curve)
        self._print_filter_report()

    def _build_from_precomputed(self):
        """Build the surface dict using OM's impl_volatility directly."""
        df = self.df
        valid = df.dropna(subset=["impl_volatility", "impl_fw"]).copy()

        # Keep only OTM options (calls above forward, puts below)
        otm_mask = (
            ((valid["cp_flag"] == "c") & (valid["strike_price"] >= valid["impl_fw"])) |
            ((valid["cp_flag"] == "p") & (valid["strike_price"] <= valid["impl_fw"]))
        )
        valid = valid[otm_mask]
        self.filter_report["otm_rows_after_forward_filter"] = len(valid)

        # Price filter
        price_mask = (
            (valid["best_bid"]   >= self.MIN_OPTION_PRICE) &
            (valid["best_offer"] >= self.MIN_OPTION_PRICE) &
            (valid["mid_price"]  >= self.MIN_OPTION_PRICE)
        )
        self.filter_report["otm_rows_dropped_price_filter"] = int((~price_mask).sum())
        valid = valid[price_mask]

        # Short-T filter
        valid["T"] = (valid["exdate"] - valid["date"]).dt.days / self.DAY_COUNT
        t_mask = valid["T"] >= self.MIN_T
        self.filter_report["otm_rows_dropped_short_T"] = int((~t_mask).sum())
        valid = valid[t_mask]

        # IV sanity filter
        iv_mask = (valid["impl_volatility"] > 0) & (valid["impl_volatility"] < 3.0)
        self.filter_report["solver_skipped"] = int((~iv_mask).sum())
        valid = valid[iv_mask]

        # Build surface dict
        dates = pd.to_datetime(valid["date"]).dt.date.values
        expiries = pd.to_datetime(valid["exdate"]).dt.date.values
        strikes = valid["strike_price"].values
        ivs = valid["impl_volatility"].values

        # Use bid/offer IVs if available, otherwise approximate from mid IV
        has_bid_iv = "iv_bid" in valid.columns
        has_offer_iv = "iv_offer" in valid.columns

        surface = {}
        for i in range(len(valid)):
            iv = float(ivs[i])
            if has_bid_iv and has_offer_iv:
                bid_iv = float(valid.iloc[i]["iv_bid"])
                offer_iv = float(valid.iloc[i]["iv_offer"])
                if np.isnan(bid_iv) or bid_iv <= 0:
                    bid_iv = iv * 0.995
                if np.isnan(offer_iv) or offer_iv <= 0:
                    offer_iv = iv * 1.005
            else:
                # Approximate bid/offer IV as +/- 0.5% of mid
                bid_iv = iv * 0.995
                offer_iv = iv * 1.005

            surface[(dates[i], expiries[i], float(strikes[i]))] = (
                bid_iv, iv, offer_iv
            )

        self.filter_report["surface_points_built"] = len(surface)
        return surface