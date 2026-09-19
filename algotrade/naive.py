"""The vectorised backtest, exactly as every tutorial writes it. Including the bugs.

This module exists so the workshop can *measure* the damage each shortcut does,
rather than asserting it. Nothing here should be used to evaluate a strategy.
It is a teaching exhibit, and the functions say so at runtime.

The classic one-liner::

    returns = prices.pct_change()
    equity = (1 + signal * returns).cumprod()

has three defects, and students routinely ship all three:

1. **No shift.** ``signal`` at bar ``t`` was computed from the close of bar
   ``t``, then applied to the return *of* bar ``t`` -- the return from ``t-1``
   to ``t``, which had already happened. You earned a return you could only
   have captured by trading in the past. This is look-ahead bias and it is
   worth an enormous amount of fake money.
2. **No costs.** Free trading, infinitely often.
3. **No slippage.** Every fill at the printed mid.

:func:`vectorized_backtest` reproduces all three by default, and
:func:`compare_shift_effect` puts a number on defect 1.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

__all__ = ["vectorized_backtest", "compare_shift_effect", "NaiveBacktestWarning"]


class NaiveBacktestWarning(UserWarning):
    """Emitted every time a knowingly-wrong backtest is run."""


def vectorized_backtest(
    prices: pd.Series,
    signal: pd.Series,
    *,
    shift: int = 0,
    cost_bps: float = 0.0,
    quiet: bool = False,
) -> pd.DataFrame:
    """Run the tutorial backtest. Returns a frame with ``position``, ``ret``, ``equity``.

    Parameters
    ----------
    signal:
        Target position per bar, typically in ``{-1, 0, 1}``.
    shift:
        Bars of delay between computing the signal and earning the return.

        ``shift=0`` is the **bug**: the position at bar ``t`` captures bar
        ``t``'s return, which was already determined before you could have
        acted.

        ``shift=1`` is the minimum correct value: decide using bar ``t``'s
        close, earn bar ``t+1``'s return.
    cost_bps:
        Cost in basis points charged on the absolute change in position, i.e.
        **per side**. A full round trip (0 -> 1 -> 0) changes the position twice
        and therefore pays this twice. To model a 24 bps round trip, pass 12.

        Getting this backwards doubles your modelled costs, which is an easy
        way to talk yourself out of a strategy that was merely mediocre.

    Warnings
    --------
    Emits :class:`NaiveBacktestWarning` when ``shift=0`` or ``cost_bps=0``,
    because both produce numbers that are not achievable.
    """
    prices = pd.Series(prices).astype(float)
    signal = pd.Series(signal).reindex(prices.index).astype(float)

    if not quiet:
        problems = []
        if shift < 1:
            problems.append("shift=0 leaks the future into the position (LOOK-AHEAD BIAS)")
        if cost_bps <= 0:
            problems.append("cost_bps=0 assumes free trading")
        if problems:
            warnings.warn(
                "vectorized_backtest is producing an unachievable result: "
                + "; ".join(problems),
                NaiveBacktestWarning,
                stacklevel=2,
            )

    position = signal.shift(shift).fillna(0.0)
    bar_return = prices.pct_change().fillna(0.0)
    gross = position * bar_return
    traded = position.diff().abs().fillna(position.abs())
    cost = traded * cost_bps / 10_000.0
    net = gross - cost

    return pd.DataFrame(
        {
            "price": prices,
            "signal": signal,
            "position": position,
            "bar_return": bar_return,
            "gross_return": gross,
            "cost": cost,
            "net_return": net,
            "equity": (1.0 + net).cumprod(),
        }
    )


def compare_shift_effect(
    prices: pd.Series, signal: pd.Series, *, shifts=(0, 1, 2), cost_bps: float = 0.0
) -> pd.DataFrame:
    """How much of the 'profit' was just look-ahead?

    Runs the same signal at several delays. The gap between ``shift=0`` and
    ``shift=1`` is pure look-ahead bias -- money that existed only because the
    backtest let you trade on information you did not have.

    A signal whose profit vanishes at ``shift=1`` had no predictive content at
    all; it was reading the answer off the same bar.
    """
    rows = {}
    for s in shifts:
        bt = vectorized_backtest(prices, signal, shift=s, cost_bps=cost_bps, quiet=True)
        eq = bt["equity"]
        rets = bt["net_return"]
        rows[f"shift={s}"] = {
            "total_return": eq.iloc[-1] - 1.0,
            "mean_bar_return": rets.mean(),
            "sharpe_per_bar": rets.mean() / rets.std(ddof=1) if rets.std(ddof=1) > 0 else np.nan,
            "max_drawdown": (eq / eq.cummax() - 1).min(),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "delay"
    return out
