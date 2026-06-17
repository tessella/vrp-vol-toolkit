# vrp-vol-toolkit

This is the core of a larger (private) variance-risk-premium options research
stack: option pricing, implied-volatility surface construction, realized-vol
estimation, and point-in-time VRP signal generation, with a small delta-hedging
engine.

Built on classic results: Black-Scholes
Greeks, put-call-parity forward implication, standard range-based vol estimators,
rough-volatility (RFSV) forecasting, and the well-documented RV−IV signal. The
strategy-selection, portfolio-construction and validation layers of the full
stack are not included.

## Content

| Module | Contents |
|---|---|
| `vrp/core/models.py` | `PricingModel` ABC + `BlackScholesModel`: price, delta, gamma, theta, vega, **vanna, volga**, cash-gamma, numerical implied vol. |
| `vrp/core/surface.py` | `ImpliedVolSurface`: put-call-parity **forward & discount-factor implication** + vectorised **bisection IV solver** + OTM/price/short-T filtering. `InterpolatedVolSurface`: **cubic spline on total variance** vs log-moneyness, with calendar (term) interpolation. `OMVolSurface`: variant for pre-computed IVs (e.g. OptionMetrics). |
| `vrp/core/realized_vol.py` | `Simple` / `CloseToClose` / **Yang-Zhang** / **Garman-Klass** / **Parkinson** estimators + an **RFSV** rough-vol predictor (Gatheral–Jaisson–Rosenbaum) with Hurst estimation and Zumbach correction. |
| `vrp/core/strategy.py` | `VolStrategy` (cash-gamma-weighted RV), `ClassicVRPSignal` (RV−IV), `RFSVSignal` (horizon-matched rough-vol). |
| `vrp/core/hedging.py` | `OptionPosition`, `BlackScholesDeltaHedger`, and a `HedgeAccount` tracking cash, shares, transaction costs and financing. |

## Install

```bash
pip install -e .          # or: pip install -r requirements.txt
```

## API sketch

```python
import pandas as pd
from vrp.core import ImpliedVolSurface, InterpolatedVolSurface

# chain: columns [date, exdate, cp_flag, strike_price, best_bid, best_offer]
surf   = ImpliedVolSurface(chain)
interp = InterpolatedVolSurface(surf)
iv     = interp.get_mid_vol(trade_date, expiry, strike)   # interpolates if off-grid
```

## License

MIT — see [LICENSE](LICENSE).
