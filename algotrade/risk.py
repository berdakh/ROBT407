"""Position sizing and risk control.

THE SEPARATION THIS MODULE ENFORCES
A signal says *which way* to lean. Sizing says *how much*. Beginners fuse the
two -- "go long 100%" -- and then discover that their strategy's risk doubles
whenever volatility doubles, which is precisely when they can least afford it.

Sizing is also where most of the realisable improvement lives. Given a mediocre
signal, good sizing produces a survivable strategy. Given a good signal, bad
sizing produces a margin call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import Strategy

__all__ = [
    "volatility_target_weight", "vol_target_series", "kelly_fraction",
    "fixed_fractional_size", "VolatilityTargeted", "DrawdownGuard",
]


def volatility_target_weight(
    realized_vol: float, target_vol: float, *, max_leverage: float = 1.0, floor_vol: float = 1e-6
) -> float:
    """Scale factor that maps a realised volatility to a target volatility.

    ``weight = target_vol / realized_vol``, capped at ``max_leverage``.

    The cap is not optional. When realised volatility collapses -- a quiet
    market, or a stablecoin right before it depegs -- the uncapped formula
    demands enormous leverage at exactly the wrong moment.
    """
    if not np.isfinite(realized_vol) or realized_vol < floor_vol:
        return 0.0
    return float(min(target_vol / realized_vol, max_leverage))


def vol_target_series(
    returns: pd.Series,
    *,
    target_vol: float = 0.20,
    lookback: int = 24 * 14,
    bars_per_year: float = 365 * 24,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Per-bar volatility-scaling factors, computed causally.

    The ``shift(1)`` is essential: the scale applied to bar ``t`` uses
    volatility estimated through bar ``t-1``. Without it you size today's
    position using today's volatility, which you only learn at today's close.
    """
    vol = returns.rolling(lookback, min_periods=lookback).std(ddof=1) * np.sqrt(bars_per_year)
    scale = (target_vol / vol.shift(1)).clip(upper=max_leverage)
    return scale.fillna(0.0)


def kelly_fraction(mean_return: float, variance: float, *, safety_factor: float = 0.25) -> float:
    """The Kelly bet size, scaled down by ``safety_factor``.

    Full Kelly maximises long-run log wealth *given that you know the true
    mean and variance*. You do not. You have an estimate from a short, noisy
    sample, and Kelly is extremely sensitive to an overestimated mean -- betting
    twice the Kelly fraction has an expected growth rate of zero, and beyond
    that it is negative.

    Practitioners who use Kelly at all use a quarter of it. The default here is
    0.25 for that reason, and you should treat even that as aggressive.
    """
    if variance <= 0:
        return 0.0
    return float(safety_factor * mean_return / variance)


def fixed_fractional_size(
    equity: float, price: float, stop_distance: float, *, risk_per_trade: float = 0.01
) -> float:
    """Units to buy so that hitting the stop loses ``risk_per_trade`` of equity.

    The one sizing rule worth memorising:
    ``units = (equity * risk_fraction) / distance_to_stop``.

    It makes position size fall automatically when your stop is far away --
    that is, when the market is volatile.
    """
    if stop_distance <= 0 or price <= 0:
        return 0.0
    return float(equity * risk_per_trade / stop_distance)


class VolatilityTargeted(Strategy):
    """Wrap a strategy so its *direction* is kept but its *size* targets a fixed volatility.

    The inner strategy still returns -1/0/+1. This wrapper multiplies that by
    a scale factor derived from trailing realised volatility, so the portfolio
    runs at roughly constant risk instead of constant notional.

    Everything is computed from the view, so the wrapper inherits the
    look-ahead guarantee.
    """

    def __init__(
        self,
        inner: Strategy,
        *,
        target_vol: float = 0.20,
        lookback: int = 24 * 14,
        bars_per_year: float = 365 * 24,
        max_leverage: float = 1.0,
    ):
        self.inner = inner
        self.target_vol = target_vol
        self.lookback = lookback
        self.bars_per_year = bars_per_year
        self.max_leverage = max_leverage
        self.warmup = max(inner.warmup, lookback + 1)
        self.name = f"{inner.name} @ {target_vol:.0%} vol"

    def on_start(self, view):
        self.inner.on_start(view)

    def on_bar(self, view, account):
        raw = self.inner.on_bar(view, account)
        if raw is None or not isinstance(raw, (int, float, np.floating)):
            return raw
        if len(view) < self.lookback + 1:
            return 0.0
        closes = view.close.history(self.lookback + 1)
        rets = np.diff(closes) / closes[:-1]
        realized = float(np.std(rets, ddof=1) * np.sqrt(self.bars_per_year))
        scale = volatility_target_weight(realized, self.target_vol, max_leverage=self.max_leverage)
        return float(raw) * scale


class DrawdownGuard(Strategy):
    """Wrap a strategy so it de-risks after a drawdown and re-risks on recovery.

    When the account's drawdown from its own peak exceeds ``max_drawdown``,
    exposure is cut to ``reduced_exposure`` until equity recovers to within
    ``recover_at`` of the peak.

    An honest caveat: this is a stop-loss on the whole portfolio, and
    portfolio stop-losses systematically sell low. They reduce the chance of
    ruin and, on most strategies, also reduce expected return. That trade is
    often worth making -- being alive matters more than being optimal -- but
    it is a trade, not free insurance.
    """

    def __init__(
        self,
        inner: Strategy,
        *,
        max_drawdown: float = 0.20,
        reduced_exposure: float = 0.0,
        recover_at: float = 0.10,
    ):
        self.inner = inner
        self.max_drawdown = abs(max_drawdown)
        self.reduced_exposure = reduced_exposure
        self.recover_at = abs(recover_at)
        self.warmup = inner.warmup
        self.name = f"{inner.name} + DD guard {max_drawdown:.0%}"
        self._peak = -np.inf
        self._halted = False
        self.halt_log: list = []

    def on_start(self, view):
        self._peak = -np.inf
        self._halted = False
        self.halt_log = []
        self.inner.on_start(view)

    def on_bar(self, view, account):
        equity = account.equity
        self._peak = max(self._peak, equity)
        dd = equity / self._peak - 1.0 if self._peak > 0 else 0.0

        if not self._halted and dd <= -self.max_drawdown:
            self._halted = True
            self.halt_log.append((view.now, "halt", dd))
        elif self._halted and dd >= -self.recover_at:
            self._halted = False
            self.halt_log.append((view.now, "resume", dd))

        raw = self.inner.on_bar(view, account)
        if self._halted:
            return self.reduced_exposure
        return raw
