"""Core library: pricing models, vol surface, realized-vol estimators,
VRP signals, and delta hedging."""

from vrp.core.models import (
    OptionType,
    PricingModel,
    BlackScholesModel,
)
from vrp.core.surface import (
    ImpliedVolSurface,
    InterpolatedVolSurface,
    OMVolSurface,
)
from vrp.core.realized_vol import (
    RealizedVolEstimator,
    SimpleRealizedVol,
    CloseToCloseRV,
    YangZhangRV,
    GarmanKlassRV,
    ParkinsonRV,
    RFSVPredictor,
    estimate_realized_vol,
)
from vrp.core.strategy import (
    VolStrategy,
    ClassicVRPSignal,
    RFSVSignal,
    get_trading_signals,
)
from vrp.core.hedging import (
    OptionPosition,
    DeltaHedger,
    BlackScholesDeltaHedger,
    HedgeAccount,
    VolNotAvailableError,
)

__all__ = [
    "OptionType", "PricingModel", "BlackScholesModel",
    "ImpliedVolSurface", "InterpolatedVolSurface", "OMVolSurface",
    "RealizedVolEstimator", "SimpleRealizedVol", "CloseToCloseRV",
    "YangZhangRV", "GarmanKlassRV", "ParkinsonRV", "RFSVPredictor",
    "estimate_realized_vol",
    "VolStrategy", "ClassicVRPSignal", "RFSVSignal", "get_trading_signals",
    "OptionPosition", "DeltaHedger", "BlackScholesDeltaHedger",
    "HedgeAccount", "VolNotAvailableError",
]
