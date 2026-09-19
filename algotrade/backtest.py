"""The event-driven backtester.

WHY EVENT-DRIVEN AND NOT VECTORISED
A vectorised backtest -- ``signal.shift(1) * returns`` -- is fast, fits in one
line, and is how almost every tutorial does it. It is also where look-ahead
bias hides, because nothing in the expression stops ``signal`` from having been
computed with knowledge of the whole series. You have to *remember* to shift,
and to shift by the right amount, and to shift every derived column.

This engine walks bar by bar and hands the strategy a view that cannot reach
past the current bar. The guarantee is structural: there is no shift to forget,
because the future is not in the room.

THE LOOP, PRECISELY
For each bar ``i``:

1. Advance the clock to ``i``. The strategy can now see bars ``0..i``.
2. **Execute orders submitted on bar i-1**, at bar ``i``'s OPEN price, plus
   costs. This is the honest timing: you decided last bar, you trade at the
   next price available.
3. Mark the account to bar ``i``'s CLOSE.
4. Ask the strategy for a decision, given data through bar ``i``'s close.
   Orders it submits are queued for bar ``i+1``.
5. Record equity, weight, position.

Note what step 2 and step 4 mean together: a signal computed from bar ``i``'s
close fills at bar ``i+1``'s open. One full bar of delay, always.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .data import BARS_PER_YEAR, validate_ohlcv
from .execution import CostModel, Fill, Order, OrderStatus, OrderType, RETAIL_CRYPTO
from .portfolio import Account
from .strategy import Strategy
from .view import Clock, MarketView

__all__ = ["BacktestConfig", "BacktestResult", "Backtester", "run_backtest"]

FillTiming = Literal["next_open", "next_close", "this_close"]


@dataclass(frozen=True)
class BacktestConfig:
    """Everything that is not the strategy or the data.

    Parameters
    ----------
    fill_timing:
        ``"next_open"`` (default, honest): decide at bar i's close, fill at
        bar i+1's open.

        ``"next_close"``: fill at bar i+1's close. Also honest, slightly more
        pessimistic about how quickly you can act.

        ``"this_close"``: fill at bar i's *own* close -- the price that was
        used to generate the signal. This is **execution optimism**: you
        cannot simultaneously observe a closing price and trade at it, so any
        result depending on it describes a market that does not exist.

        It is *not* look-ahead bias, and -- this surprises people -- it does
        not reliably inflate returns either. Measured across 40 dataset/seed
        combinations it beat ``next_open`` 22 times, which is a coin flip.
        Genuine look-ahead (see :mod:`algotrade.naive`) wins 100% of the time.
        Unrealistic and inflated are different properties, and only one of them
        is detectable by looking at the equity curve.
    max_leverage:
        Hard cap on ``abs(weight)``. Target weights are clipped, not rejected.
    allow_short:
        If False, negative target weights are clipped to 0.
    """

    initial_cash: float = 10_000.0
    cost_model: CostModel = RETAIL_CRYPTO
    fill_timing: FillTiming = "next_open"
    max_leverage: float = 1.0
    allow_short: bool = True
    bars_per_year: float = float(BARS_PER_YEAR["1h"])
    min_trade_value: float = 10.0     # venues have a minimum notional; dust orders are rejected
    rebalance_tolerance: float = 0.0  # skip rebalances smaller than this much weight
    stop_on_ruin: bool = True

    @property
    def is_optimistic(self) -> bool:
        return self.fill_timing == "this_close"

    @property
    def is_frictionless(self) -> bool:
        return self.cost_model.round_trip_bps() == 0.0


@dataclass
class BacktestResult:
    """Everything the engine observed. Feed it to :mod:`algotrade.report`."""

    equity: pd.Series
    weights: pd.Series
    units: pd.Series
    price: pd.Series
    targets: pd.Series
    fills: list[Fill]
    config: BacktestConfig
    strategy_name: str
    warnings: list[str] = field(default_factory=list)

    @property
    def returns(self) -> pd.Series:
        """Bar-over-bar simple returns of the equity curve."""
        return self.equity.pct_change().fillna(0.0)

    @property
    def n_trades(self) -> int:
        return len(self.fills)

    @property
    def total_costs(self) -> float:
        return sum(f.total_cost for f in self.fills)

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)

    @property
    def total_slippage(self) -> float:
        return sum(f.slippage_cost for f in self.fills)

    def trades_frame(self) -> pd.DataFrame:
        """One row per fill."""
        if not self.fills:
            return pd.DataFrame(
                columns=["timestamp", "qty", "price", "reference_price",
                         "commission", "slippage_cost", "notional", "tag"]
            ).set_index("timestamp")
        rows = [
            {
                "timestamp": f.timestamp, "qty": f.qty, "price": f.price,
                "reference_price": f.reference_price, "commission": f.commission,
                "slippage_cost": f.slippage_cost, "notional": f.notional, "tag": f.tag,
            }
            for f in self.fills
        ]
        return pd.DataFrame(rows).set_index("timestamp")

    def __repr__(self) -> str:  # pragma: no cover - display helper
        total = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        return (
            f"<BacktestResult {self.strategy_name!r} bars={len(self.equity)} "
            f"trades={self.n_trades} total_return={total:+.1%}>"
        )


class Backtester:
    """Runs one strategy over one OHLCV frame."""

    def __init__(self, data: pd.DataFrame, config: BacktestConfig | None = None, *, validate: bool = True):
        if validate:
            validate_ohlcv(data)
        self.data = data
        self.config = config or BacktestConfig()

    def run(self, strategy: Strategy, *, progress: bool = False) -> BacktestResult:
        cfg = self.config
        data = self.data
        n = len(data)
        if n < 2:
            raise ValueError("need at least 2 bars to backtest")

        opens = data["open"].to_numpy(float)
        closes = data["close"].to_numpy(float)
        volumes = data["volume"].to_numpy(float) if "volume" in data else np.full(n, np.nan)
        index = data.index

        clock = Clock(-1)
        view = MarketView(data, clock)
        account = Account(cash=cfg.initial_cash)

        equity = np.empty(n)
        weights = np.empty(n)
        units = np.empty(n)
        # The weight the strategy ASKED for, as distinct from the weight it
        # ended up with. Paper-trading reconciliation compares targets to
        # targets; comparing a target to an achieved weight always "fails",
        # because fills happen at a different price than the decision.
        targets = np.full(n, np.nan)
        pending: list[Order] = []
        warnings: list[str] = []
        ruined_at = None

        strategy.on_start(view)

        for i in range(n):
            # 1. The strategy may now see bars 0..i and no further.
            view._advance(i)

            # 2. Settle orders decided on an earlier bar.
            if pending:
                ref = opens[i] if cfg.fill_timing == "next_open" else closes[i]
                pending = self._settle(pending, ref, volumes[i], index[i], account, i)

            # 3. Mark to this bar's close.
            account.mark(closes[i])

            if cfg.stop_on_ruin and account.equity <= 0 and ruined_at is None:
                ruined_at = index[i]
                warnings.append(
                    f"ACCOUNT WIPED OUT at {ruined_at}: equity reached {account.equity:,.2f}. "
                    f"Everything after this point is meaningless."
                )
                equity[i:] = 0.0
                weights[i:] = 0.0
                units[i:] = account.units
                break

            # 4. Ask the strategy. Anything it returns is queued for the NEXT bar.
            decision = strategy.on_bar(view, account)
            if isinstance(decision, (int, float, np.floating, np.integer)) and not isinstance(decision, bool):
                targets[i] = float(decision)
            new_orders = self._decision_to_orders(decision, account, closes[i], index[i], cfg)

            if cfg.is_optimistic and new_orders:
                # OPTIMISTIC MODE: fill immediately at the close we just used
                # to make the decision. Day 2 measures what this flattery is
                # worth. Anything that cannot fill here (an untouched limit
                # order) still rolls forward like any other pending order.
                new_orders = self._settle(new_orders, closes[i], volumes[i], index[i], account, i)
                account.mark(closes[i])
            pending = pending + new_orders

            # 5. Record.
            equity[i] = account.equity
            weights[i] = account.weight
            units[i] = account.units

            if progress and n >= 20 and i % (n // 20) == 0:
                print(f"  bar {i}/{n} equity={account.equity:,.0f}", flush=True)

        result = BacktestResult(
            equity=pd.Series(equity, index=index, name="equity"),
            weights=pd.Series(weights, index=index, name="weight"),
            units=pd.Series(units, index=index, name="units"),
            price=data["close"].copy(),
            targets=pd.Series(targets, index=index, name="target_weight"),
            fills=account.fills,
            config=cfg,
            strategy_name=strategy.name,
            warnings=warnings,
        )
        strategy.on_finish(result)
        return result

    # -- internals --------------------------------------------------------
    def _decision_to_orders(self, decision, account: Account, price: float, ts, cfg) -> list[Order]:
        if decision is None:
            return []
        if isinstance(decision, Order):
            decision.submitted_at = ts
            return [decision]
        if isinstance(decision, (list, tuple)):
            for o in decision:
                o.submitted_at = ts
            return list(decision)
        if isinstance(decision, (int, float, np.floating, np.integer)):
            return self._weight_to_orders(float(decision), account, price, ts, cfg)
        raise TypeError(
            f"on_bar returned {type(decision).__name__}; expected None, a float target "
            f"weight, an Order, or a list of Orders"
        )

    @staticmethod
    def _weight_to_orders(target_weight: float, account: Account, price: float, ts, cfg) -> list[Order]:
        if not np.isfinite(target_weight):
            raise ValueError(f"target weight must be finite, got {target_weight}")
        if not cfg.allow_short:
            target_weight = max(0.0, target_weight)
        target_weight = float(np.clip(target_weight, -cfg.max_leverage, cfg.max_leverage))

        if cfg.rebalance_tolerance > 0 and abs(target_weight - account.weight) < cfg.rebalance_tolerance:
            # Letting the weight drift a little is almost always cheaper than
            # paying the spread to correct it. This is how real books control
            # turnover.
            return []

        desired_units = account.target_units(target_weight, price)
        delta = desired_units - account.units
        if abs(delta * price) < cfg.min_trade_value:
            return []
        order = Order(qty=delta, type=OrderType.MARKET, tag=f"target_w={target_weight:+.3f}")
        order.submitted_at = ts
        return [order]

    def _settle(self, orders: list[Order], reference: float, volume: float, ts, account: Account, i: int) -> list[Order]:
        """Fill what can fill at ``reference``; return orders still outstanding."""
        still_pending: list[Order] = []
        for order in orders:
            if order.type is OrderType.LIMIT:
                # A buy limit fills only if the bar traded at or below the limit.
                low = self.data["low"].iat[i]
                high = self.data["high"].iat[i]
                touched = (order.qty > 0 and low <= order.limit_price) or (
                    order.qty < 0 and high >= order.limit_price
                )
                if not touched:
                    if order.expires_after is not None:
                        order.expires_after -= 1
                        if order.expires_after <= 0:
                            order.status = OrderStatus.EXPIRED
                            continue
                    still_pending.append(order)
                    continue
                # Optimistic but standard: assume you get your limit price.
                # No queue position, no partial fills. Real limit orders are
                # worse than this.
                ref = order.limit_price
            else:
                ref = reference

            fill = self.config.cost_model.execute(ref, order.qty, volume, ts, tag=order.tag)
            account.apply(fill)
            order.status = OrderStatus.FILLED
        return still_pending


def run_backtest(
    data: pd.DataFrame, strategy: Strategy, config: BacktestConfig | None = None, **kwargs
) -> BacktestResult:
    """Convenience wrapper: ``run_backtest(df, SMACrossover())``."""
    return Backtester(data, config, **kwargs).run(strategy)
