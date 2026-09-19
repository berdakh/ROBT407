"""Paper trading: a simulated broker that cannot reach a real exchange.

SAFETY POSITION OF THIS MODULE
This code has no network access, no API keys, no credential handling and no
order-routing path. It cannot place a real order because there is nothing in it
that knows how. That is deliberate and permanent: the workshop teaches
methodology, and a student following it should never be one uncommented line
away from sending money to an exchange.

WHAT PAPER TRADING IS ACTUALLY FOR
Not for finding out whether a strategy is profitable -- a few weeks of paper
trading has nowhere near the statistical power to answer that. It is for
finding out whether your *system* works:

* Does the strategy still run when a bar arrives late, or twice, or not at all?
* Do you handle a restart without losing your position state?
* Does your live signal match what the backtest said it would be on the same bar?
* How long does a decision actually take?

Those are engineering questions, and paper trading answers them. Profitability
it cannot answer. The :class:`PaperBroker` deliberately logs the reconciliation
between live and backtest signals, because a mismatch there is the single most
common way a strategy that "worked in backtest" fails in production.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestConfig
from .execution import CostModel, Fill, RETAIL_CRYPTO
from .portfolio import Account
from .strategy import Strategy
from .view import MarketView

__all__ = ["PaperBroker", "PaperTradingSession", "SessionLog"]


@dataclass
class SessionLog:
    """One row per bar processed by a paper session."""

    timestamp: str
    bar_index: int
    price: float
    target_weight: float | None
    actual_weight: float
    equity: float
    action: str
    note: str = ""


class PaperBroker:
    """A broker that only ever pretends.

    Holds simulated cash and position, applies the same :class:`CostModel` the
    backtester uses, and refuses -- loudly -- to do anything else.
    """

    #: Class-level flag. Nothing reads it to enable live trading; it exists so
    #: that code and documentation can assert the intent in one place.
    LIVE_TRADING_SUPPORTED = False

    def __init__(self, initial_cash: float = 10_000.0, cost_model: CostModel = RETAIL_CRYPTO):
        self.account = Account(cash=initial_cash)
        self.cost_model = cost_model
        self.fills: list[Fill] = []

    def submit(self, qty: float, price: float, volume: float | None, timestamp) -> Fill | None:
        """'Execute' a trade against simulated cash. No network call happens."""
        if qty == 0 or not np.isfinite(qty):
            return None
        fill = self.cost_model.execute(price, qty, volume, timestamp, tag="paper")
        self.account.apply(fill)
        self.fills.append(fill)
        return fill

    def connect(self, *args, **kwargs):  # pragma: no cover - guard rail
        raise NotImplementedError(
            "PaperBroker cannot connect to anything. This workshop is paper-trading "
            "only by design: there is no live order path, no credential handling and "
            "no exchange client in this package. If you later choose to trade real "
            "money, that is a decision to make deliberately, with your own code, "
            "having first read docs/handbook.html#why-most-strategies-fail."
        )

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"<PaperBroker SIMULATED {self.account!r}>"


class PaperTradingSession:
    """Replays bars one at a time through a strategy, as a live loop would.

    The difference from the backtester is not the maths -- it is the *shape* of
    the loop. Here, bars arrive one by one from an outside source, state must
    survive between them, and nothing about the future exists even in memory.
    Running your strategy this way is how you discover that it quietly depended
    on having the whole array.

    Usage::

        session = PaperTradingSession(strategy, history=df.iloc[:500])
        for ts, bar in df.iloc[500:].iterrows():
            session.on_new_bar(ts, bar)
        session.summary()

    ``history`` seeds the warm-up window so the strategy can act on its first
    live bar instead of waiting out its lookback again.
    """

    def __init__(
        self,
        strategy: Strategy,
        history: pd.DataFrame,
        *,
        initial_cash: float = 10_000.0,
        cost_model: CostModel = RETAIL_CRYPTO,
        max_leverage: float = 1.0,
        log_dir: str | Path | None = None,
    ):
        self.strategy = strategy
        self.broker = PaperBroker(initial_cash, cost_model)
        self.max_leverage = max_leverage
        # A streaming view appends bars in O(1) instead of rebuilding the whole
        # frame each time, which matters once a session runs for thousands of
        # bars -- and matters much more in production than in a notebook.
        self._view = MarketView._streaming(history, capacity=max(4 * len(history), 1024))
        self.logs: list[SessionLog] = []
        self.pending_qty: float | None = None
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.strategy.on_start(self._view)
        self.started_at = datetime.now(timezone.utc).isoformat()

    def on_new_bar(self, timestamp, bar: pd.Series) -> SessionLog:
        """Process one newly-closed bar. This is the whole live loop.

        Order of operations mirrors :class:`~algotrade.backtest.Backtester`
        exactly, which is what makes live and backtest results comparable:
        settle last bar's decision at this bar's open, mark to this close,
        then decide.
        """
        # Append the new bar. The view's clock advances with it.
        self._view._append_bar(timestamp, bar)

        note = ""
        action = "hold"

        # 1. Settle the order decided on the previous bar, at this bar's open.
        if self.pending_qty:
            fill = self.broker.submit(
                self.pending_qty, float(bar["open"]), float(bar.get("volume", np.nan)), timestamp
            )
            if fill:
                action = "buy" if fill.qty > 0 else "sell"
                note = f"filled {fill.qty:+.6f} @ {fill.price:.2f} (ref {fill.reference_price:.2f}, cost {fill.total_cost:.2f})"
            self.pending_qty = None

        # 2. Mark to market.
        self.broker.account.mark(float(bar["close"]))

        # 3. Decide, using only what is visible.
        decision = self.strategy.on_bar(self._view, self.broker.account)
        target = None
        if isinstance(decision, (int, float, np.floating)):
            target = float(np.clip(decision, -self.max_leverage, self.max_leverage))
            desired = self.broker.account.target_units(target, float(bar["close"]))
            delta = desired - self.broker.account.units
            if abs(delta * float(bar["close"])) >= 10.0:
                self.pending_qty = delta
                note = (note + " | " if note else "") + f"queued {delta:+.6f} for next open"

        entry = SessionLog(
            timestamp=str(timestamp),
            bar_index=self._view.i,
            price=float(bar["close"]),
            target_weight=target,
            actual_weight=self.broker.account.weight,
            equity=self.broker.account.equity,
            action=action,
            note=note,
        )
        self.logs.append(entry)
        if self.log_dir:
            with (self.log_dir / "session.jsonl").open("a") as fh:
                fh.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def frame(self) -> pd.DataFrame:
        """The session log as a DataFrame."""
        return pd.DataFrame([asdict(entry) for entry in self.logs])

    def equity_curve(self) -> pd.Series:
        f = self.frame()
        if f.empty:
            return pd.Series(dtype=float)
        return pd.Series(
            f["equity"].to_numpy(), index=pd.to_datetime(f["timestamp"], utc=True), name="equity"
        )

    def reconcile(self, backtest_targets: pd.Series, *, tolerance: float = 1e-6) -> pd.DataFrame:
        """Compare live target weights against a backtest's targets, bar by bar.

        Pass ``BacktestResult.targets`` -- the weight the strategy *asked* for.
        Do not pass ``.weights``: that is the weight it actually ended up
        holding after fills, which differs from the target for entirely
        legitimate reasons (you fill at the next open, not at the decision
        price) and would make every row look like a mismatch.

        A true mismatch is a bug in one of the two paths, and you want to find
        it now rather than after a month of live divergence. Common causes: the
        live loop seeds a different warm-up window, or the strategy carries
        state that a fresh backtest does not reproduce.
        """
        live = self.frame()
        if live.empty:
            return pd.DataFrame()
        live = live.set_index(pd.to_datetime(live["timestamp"], utc=True))
        common = live.index.intersection(backtest_targets.index)
        out = pd.DataFrame(
            {
                "live_target": live.loc[common, "target_weight"],
                "backtest_target": backtest_targets.loc[common],
            }
        )
        out["difference"] = out["live_target"] - out["backtest_target"]
        # Both sides NaN means "strategy returned None on this bar" in both
        # paths, which agrees. One side NaN is a genuine disagreement.
        both_nan = out["live_target"].isna() & out["backtest_target"].isna()
        one_nan = out["live_target"].isna() ^ out["backtest_target"].isna()
        out["mismatch"] = (out["difference"].abs() > tolerance).fillna(False) | one_nan
        out.loc[both_nan, "mismatch"] = False
        return out

    def summary(self) -> str:
        acct = self.broker.account
        eq = self.equity_curve()
        total = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else float("nan")
        return "\n".join(
            [
                "=" * 66,
                "PAPER TRADING SESSION -- SIMULATED, NO REAL ORDERS WERE PLACED",
                "=" * 66,
                f"  strategy        : {self.strategy.name}",
                f"  started         : {self.started_at}",
                f"  bars processed  : {len(self.logs)}",
                f"  fills           : {len(self.broker.fills)}",
                f"  equity          : {acct.equity:,.2f}",
                f"  return          : {total:+.2%}",
                f"  costs paid      : {acct.total_commission + acct.total_slippage:,.2f}",
                f"  final weight    : {acct.weight:+.3f}",
                "=" * 66,
                "  A few weeks of paper trading cannot tell you whether this is",
                "  profitable. It tells you whether the plumbing works. Those are",
                "  different questions and only one of them is answered here.",
                "=" * 66,
            ]
        )
