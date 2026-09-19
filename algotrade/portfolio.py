"""Cash, position, and the accounting that turns fills into an equity curve.

Deliberately single-asset. Multi-asset portfolio accounting is a worthwhile
thing to learn and a terrible thing to learn *at the same time* as backtest
methodology, so the workshop holds it out of scope and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import RiskLimitError
from .execution import Fill

__all__ = ["Account"]


@dataclass
class Account:
    """The strategy's view of its own position.

    Attributes
    ----------
    cash:
        Currency balance. Goes negative when you use leverage; that is a loan.
    units:
        Signed position in units of the asset. Negative means short.
    """

    cash: float
    units: float = 0.0
    last_price: float = float("nan")
    fills: list[Fill] = field(default_factory=list)
    total_commission: float = 0.0
    total_slippage: float = 0.0

    @property
    def position_value(self) -> float:
        return self.units * self.last_price

    @property
    def equity(self) -> float:
        """Mark-to-market net worth: cash plus the value of the position."""
        return self.cash + self.position_value

    @property
    def weight(self) -> float:
        """Fraction of equity currently exposed to the asset.

        Above 1.0 means leveraged. Below 0 means net short. This is the number
        your risk limits are actually about.
        """
        eq = self.equity
        return self.position_value / eq if eq != 0 else 0.0

    @property
    def leverage(self) -> float:
        return abs(self.weight)

    def mark(self, price: float) -> None:
        self.last_price = price

    def apply(self, fill: Fill) -> None:
        """Settle a fill: move units, move cash, pay the fee."""
        self.units += fill.qty
        self.cash -= fill.qty * fill.price
        self.cash -= fill.commission
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage_cost
        self.fills.append(fill)

    def target_units(self, target_weight: float, price: float) -> float:
        """Units needed to hold ``target_weight`` of current equity at ``price``."""
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        return target_weight * self.equity / price

    def check_solvent(self) -> None:
        if self.equity <= 0:
            raise RiskLimitError(
                f"account is wiped out (equity={self.equity:.2f}). With leverage this is "
                f"not a theoretical possibility -- it is the modal outcome."
            )

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return (
            f"<Account equity={self.equity:,.2f} cash={self.cash:,.2f} "
            f"units={self.units:.6f} weight={self.weight:+.2f}>"
        )
