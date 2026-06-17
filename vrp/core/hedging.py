from datetime import date
from dataclasses import dataclass, field
from abc import abstractmethod, ABC
from enum import Enum
import numpy as np
import pandas as pd

from vrp.core.models import OptionType, PricingModel
from vrp.core.surface import InterpolatedVolSurface

@dataclass
class OptionPosition:
    """
    A single option contract entered on entry_date.
    qty > 0 = long, qty < 0 = short.
    """
    strike:        float
    expiry:        date
    option_type:   OptionType  # OptionType.CALL or OptionType.PUT
    qty:           float       # number of contracts; positive = long, negative = short
    entry_date:    date        # date the position was entered
    entry_vol:     float       # implied vol at entry
    entry_price:   float       # option price paid/received per unit
    contract_size: float   # standard SPX contract multiplier

    def __post_init__(self):
        if self.qty == 0:
            raise ValueError("qty cannot be zero.")
        if self.entry_vol <= 0:
            raise ValueError("entry_vol must be positive.")
        if self.entry_price < 0:
            raise ValueError("entry_price cannot be negative.")
        
class DeltaHedger(ABC):
    """
    Given a list of positions and market data (S, r, t), returns the
    total delta of the portfolio in share units.
    """

    @abstractmethod
    def aggregate_delta(self,
                        positions: list[OptionPosition],
                        S:         float,
                        t:         date,
                        r:         float,
                        surface:   InterpolatedVolSurface) -> float:
        """
        Return the total signed delta across all positions.
        + = long (net long delta), - = short (net short delta)
        """
        pass

class VolNotAvailableError(Exception):
    """Raised when the vol surface cannot provide a usable vol for a given point."""
    pass


class BlackScholesDeltaHedger(DeltaHedger):
    """
    Uses Black-Scholes delta looked up from the InterpolatedVolSurface.
    NOTE: currently loops over positions — can be vectorised if BlackScholesModel
    is refactored to accept array inputs.
    """

    def __init__(self, model: PricingModel):
        self.model = model

    def aggregate_delta(
        self,
        positions: list[OptionPosition],
        S:         float,
        t:         date,
        r:         float,
        surface:   InterpolatedVolSurface,
    ) -> float:

        total_delta = 0.0

        for pos in positions:
            T = (pos.expiry - t).days / 365.0
            if T <= 0:
                continue

            vol = surface.get_mid_vol(t, pos.expiry, pos.strike)
            if np.isnan(vol) or vol <= 0:
                raise VolNotAvailableError(
                    f"No usable vol for t={t}, expiry={pos.expiry}, strike={pos.strike}"
                )

            delta        = self.model.delta(S, pos.strike, T, r, vol, pos.option_type)
            total_delta += pos.qty * pos.contract_size * delta

        return total_delta
    
class HedgeAccount:
    """
    Tracks the cash account and share position from daily delta-hedging.
    Cash can go negative — this represents borrowing to finance the hedge.

    Parameters
    ----------
    initial_cash      : starting cash balance. Typically the premium received
                        or paid at entry.
    cost_pct_notional : transaction cost as a fraction of notional per trade,
                        e.g. 0.0001 = 0.01% of |Δshares| × S per rebalance.
    financing_rate    : annual rate charged on negative cash balances
                        (borrowing cost), e.g. 0.05 = 5% p.a.
                        Accrues daily as rate / 252 × |cash| when cash < 0.
                        Defaults to 0.0 (no financing cost).
    """

    TRADING_DAYS = 252.0  # financing accrues on trading days only

    def __init__(
        self,
        initial_cash:      float = 0.0,
        cost_pct_notional: float = 0.0001,
        financing_rate:    float = 0.0,
    ):
        self.cash                 = initial_cash
        self.cost_pct             = cost_pct_notional
        self.financing_rate       = financing_rate
        self.shares_held          = 0.0
        self.total_cost_paid      = 0.0   # cumulative transaction costs
        self.total_financing_paid = 0.0   # cumulative financing costs
        self.log: list[dict]      = []

    def reset(self, initial_cash: float = 0.0) -> None:
        """Reset the account to its initial state for reuse across backtests."""
        self.cash                 = initial_cash
        self.shares_held          = 0.0
        self.total_cost_paid      = 0.0
        self.total_financing_paid = 0.0
        self.log                  = []

    def rebalance(self, target_delta: float, S: float, t: date) -> None:
        """
        1. Accrue daily financing cost on any negative cash balance.
        2. Trade shares to reach -target_delta, debit transaction cost.

        Sign convention
        ---------------
        shares_held = -total_portfolio_delta
        Hedges the net delta exposure of the option book.

        Cash goes negative when we buy shares and have insufficient cash —
        this represents a margin/financing balance.
        """
        if S <= 0:
            raise ValueError(f"Spot price must be positive, got S={S}")

        # ── 1. Financing cost on negative cash balance (accrues daily) ─
        financing_cost = 0.0
        if self.cash < 0:
            financing_cost             = abs(self.cash) * self.financing_rate / self.TRADING_DAYS
            self.cash                 -= financing_cost
            self.total_financing_paid += financing_cost

        # ── 2. Trade shares ────────────────────────────────────────────
        target_shares = -target_delta
        delta_shares  = target_shares - self.shares_held  # shares to buy (>0) or sell (<0)

        notional = abs(delta_shares) * S
        cost     = notional * self.cost_pct

        self.cash            -= delta_shares * S  # negative = bought shares
        self.cash            -= cost              # transaction cost always debits cash
        self.shares_held      = target_shares
        self.total_cost_paid += cost

        self.log.append({
            "date":             t,
            "shares_held":      self.shares_held,
            "delta_traded":     delta_shares,
            "notional":         notional,
            "transaction_cost": cost,
            "financing_cost":   financing_cost,
            "cash":             self.cash,
        })

    def total_value(self, S: float) -> float:
        """
        Mark-to-market value of the hedge account.
        Can be negative if borrowing exceeds the share position value.
        """
        return self.cash + self.shares_held * S

    def log_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)