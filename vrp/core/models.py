from abc import ABC, abstractmethod
from scipy.optimize import brentq
from scipy.stats import norm
from enum import Enum
import numpy as np

class OptionType(Enum):
    CALL = 'c'
    PUT  = 'p'

class PricingModel(ABC):
    """All pricing models must implement price(), delta(), and gamma()."""

    @abstractmethod
    def price(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType) -> float:
        pass

    @abstractmethod
    def delta(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType) -> float:
        pass

    @abstractmethod
    def gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        pass

    @abstractmethod
    def theta(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType) -> float:
        pass

    @abstractmethod
    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        pass

    @abstractmethod
    def vanna(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        pass

    @abstractmethod
    def volga(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        pass

    def cash_gamma(self, S: float, K: float, T: float, r: float,
                   sigma: float) -> float:
        """Cash gamma: S^2 * d²V/dS², used as weight in the implied vol estimator."""
        return (S ** 2) * self.gamma(S, K, T, r, sigma)

    def implied_vol(self, market_price: float, S: float, K: float, T: float,
                    r: float, option_type: OptionType,
                    lower: float = 1e-6, upper: float = 10.0) -> float:
        """Invert price() numerically to recover implied volatility."""
        try:
            return brentq(
                lambda sigma: self.price(S, K, T, r, sigma, option_type) - market_price,
                lower, upper, xtol=1e-8
            )
        except ValueError:
            return float("nan")

class BlackScholesModel(PricingModel):
    """Black-Scholes model: price, greeks, and implied vol."""

    @staticmethod
    def d1d2(S: float, K: float, T: float, r: float, sigma: float):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def _blackScholesCall(self, S, K, T, r, sigma):
        d1, d2 = self.d1d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def _blackScholesPut(self, S, K, T, r, sigma):
        d1, d2 = self.d1d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def price(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType) -> float:
        if option_type == OptionType.CALL:
            return self._blackScholesCall(S, K, T, r, sigma)
        else:
            return self._blackScholesPut(S, K, T, r, sigma)

    def delta(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType) -> float:
        d1, _ = self.d1d2(S, K, T, r, sigma)
        if option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(self, S: float, K: float, T: float, r: float,
              sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1, _ = self.d1d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def theta(self, S: float, K: float, T: float, r: float, sigma: float,
              option_type: OptionType) -> float:
        """Theta: ∂V/∂t (per year).  Negative for long options (time decay)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1, d2 = self.d1d2(S, K, T, r, sigma)
        common = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == OptionType.CALL:
            return common - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return common + r * K * np.exp(-r * T) * norm.cdf(-d2)

    def vega(self, S: float, K: float, T: float, r: float,
             sigma: float) -> float:
        """Vega: ∂V/∂σ (per unit of σ).  Same for calls and puts."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1, _ = self.d1d2(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1)

    def vanna(self, S: float, K: float, T: float, r: float,
              sigma: float) -> float:
        """Vanna: ∂²V/(∂S∂σ) = ∂Δ/∂σ.  Same for calls and puts."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1, d2 = self.d1d2(S, K, T, r, sigma)
        return -norm.pdf(d1) * d2 / sigma

    def volga(self, S: float, K: float, T: float, r: float,
              sigma: float) -> float:
        """Volga (vomma): ∂²V/∂σ² = ∂vega/∂σ.  Same for calls and puts."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        d1, d2 = self.d1d2(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1) * d1 * d2 / sigma
