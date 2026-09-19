"""Orders, fills, and the cost model that decides what a trade really cost you.

WHY THIS MODULE EXISTS
Students' first backtests assume they can buy any quantity at the printed
close, for free, instantly. Every part of that is false:

* You pay a **fee** to the exchange, per trade, on notional.
* You cross the **spread**: you buy at the ask and sell at the bid, and the
  mid-price you see on the chart is neither.
* You move the market. A large order eats through the book -- **impact**.
* The price moves between deciding and executing -- **latency slippage**.

The defaults here are deliberately *modest* for a retail crypto taker. If your
strategy only works with the costs turned off, it does not work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = ["OrderType", "OrderStatus", "Order", "Fill", "CostModel", "ZERO_COST", "RETAIL_CRYPTO"]


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"    # submitted, waiting for the next bar
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"    # a limit order that was never touched


@dataclass
class Order:
    """An instruction to trade ``qty`` units. Positive buys, negative sells.

    Orders are submitted on the bar where the decision is made and can only
    fill on a **later** bar. That one rule is what makes the backtester's
    timing honest.
    """

    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    tag: str = ""
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: object = None
    expires_after: int | None = None   # bars; None = good till cancelled

    def __post_init__(self) -> None:
        if self.type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("a LIMIT order needs a limit_price")
        if not np.isfinite(self.qty):
            raise ValueError(f"order qty must be finite, got {self.qty}")

    @property
    def side(self) -> str:
        return "buy" if self.qty > 0 else "sell"


@dataclass
class Fill:
    """What actually happened, including what it cost."""

    timestamp: object
    qty: float
    price: float             # all-in execution price, after spread and impact
    reference_price: float   # the price you saw on the chart
    commission: float
    slippage_cost: float     # currency lost to spread + impact vs reference
    tag: str = ""

    @property
    def notional(self) -> float:
        return abs(self.qty * self.price)

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost


@dataclass(frozen=True)
class CostModel:
    """Converts an intended trade into a realistic execution price and fee.

    Parameters
    ----------
    commission_bps:
        Exchange fee in basis points of notional. 10 bps = 0.10%, roughly a
        retail taker fee on a major crypto venue. Makers pay less; assuming
        you will always be the maker is wishful thinking.
    half_spread_bps:
        Half the bid-ask spread. You cross this on entry *and* on exit, so a
        round trip costs the whole spread. Liquid BTC/USD perpetuals sit near
        1 bp; an illiquid altcoin can be 50.
    impact_coef:
        Square-root market impact. Impact in bps is
        ``impact_coef * 100 * sqrt(participation)`` where participation is your
        order size divided by the bar's volume. Set to 0 to ignore impact --
        reasonable only while your size is tiny relative to the market.
    latency_bps:
        Extra adverse drift between your decision and your fill. A crude proxy
        for being slower than everyone else.

    Notes
    -----
    All components push the price **against** you: buys fill higher, sells
    fill lower. A cost model that could help you is a bug.
    """

    commission_bps: float = 10.0
    half_spread_bps: float = 1.0
    impact_coef: float = 0.0
    latency_bps: float = 0.0
    name: str = "custom"

    def fill_price(self, reference_price: float, qty: float, bar_volume: float | None = None) -> float:
        """The price you actually get, given the price you saw."""
        if qty == 0:
            return reference_price
        direction = 1.0 if qty > 0 else -1.0

        adverse_bps = self.half_spread_bps + self.latency_bps
        if self.impact_coef > 0 and bar_volume:
            participation = min(abs(qty) / max(bar_volume, 1e-12), 1.0)
            adverse_bps += self.impact_coef * 100.0 * np.sqrt(participation)

        return float(reference_price * (1.0 + direction * adverse_bps / 10_000.0))

    def commission_on(self, qty: float, price: float) -> float:
        """Fee paid, always positive, always charged."""
        return abs(qty * price) * self.commission_bps / 10_000.0

    def execute(self, reference_price: float, qty: float, bar_volume: float | None, timestamp, tag: str = "") -> Fill:
        """Build the :class:`Fill` for a trade of ``qty`` at ``reference_price``."""
        price = self.fill_price(reference_price, qty, bar_volume)
        commission = self.commission_on(qty, price)
        slippage_cost = abs(qty) * abs(price - reference_price)
        return Fill(
            timestamp=timestamp,
            qty=qty,
            price=price,
            reference_price=reference_price,
            commission=commission,
            slippage_cost=slippage_cost,
            tag=tag,
        )

    def round_trip_bps(self) -> float:
        """Total cost of buying then selling the same notional, in bps.

        The number worth memorising. If your strategy's average winning trade
        is smaller than this, no amount of parameter tuning will save it.
        """
        return 2.0 * (self.commission_bps + self.half_spread_bps + self.latency_bps)

    def describe(self) -> str:
        return (
            f"{self.name}: {self.commission_bps:.1f}bps fee + {self.half_spread_bps:.1f}bps half-spread"
            f"{f' + {self.latency_bps:.1f}bps latency' if self.latency_bps else ''}"
            f"{f' + sqrt-impact({self.impact_coef})' if self.impact_coef else ''}"
            f"  -> round trip {self.round_trip_bps():.1f}bps"
        )


#: The fantasy. Used exactly once, in Day 1, to produce a beautiful lie.
ZERO_COST = CostModel(commission_bps=0.0, half_spread_bps=0.0, name="zero-cost (FICTIONAL)")

#: Plausible defaults for a small retail taker on a liquid crypto pair.
RETAIL_CRYPTO = CostModel(
    commission_bps=10.0, half_spread_bps=1.0, impact_coef=0.1, latency_bps=1.0, name="retail-crypto"
)
