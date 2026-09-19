"""The Strategy base class and a handful of reference strategies.

A strategy is a function of *observable history* to a *desired position*. It is
given a :class:`~algotrade.view.MarketView`, which physically cannot show it
the future, and an :class:`~algotrade.portfolio.Account`, which tells it what
it currently owns.

It returns one of:

* ``None`` -- do nothing, keep the current position.
* a ``float`` -- the target weight: the fraction of current equity to hold in
  the asset. ``1.0`` is fully long, ``0.0`` flat, ``-0.5`` half short.
* an :class:`~algotrade.execution.Order`, or a list of them, for explicit
  control (used in the order-lifecycle notebook).

Returning a *target weight* rather than "buy 3 units" is a deliberate teaching
choice: it separates the signal from the sizing, so Day 4 can change the sizing
without touching any strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .execution import Order, OrderType
from .view import MarketView

__all__ = [
    "Strategy", "BuyAndHold", "SMACrossover", "MomentumStrategy",
    "ZScoreReversion", "RandomStrategy", "FunctionStrategy", "ConstantWeight",
]


class Strategy(ABC):
    """Base class for every strategy in the workshop.

    Subclasses implement :meth:`on_bar`. Anything expensive that depends only
    on parameters belongs in ``__init__``; anything that depends on data must
    happen inside ``on_bar``, using only what the view offers.

    Do **not** precompute indicators over the full price series in ``__init__``
    and index into them by bar number. That is look-ahead bias wearing a
    convincing disguise, and it is why this class never receives the full
    DataFrame in the first place.
    """

    #: Human-readable name used in report cards.
    name: str = "unnamed"

    #: Bars of history the strategy needs before it can act. The engine will
    #: still call ``on_bar`` earlier, but you should return ``None`` until
    #: ``len(view) >= warmup``.
    warmup: int = 0

    def on_start(self, view: MarketView) -> None:
        """Called once before the first bar. Initialise state here."""

    @abstractmethod
    def on_bar(self, view: MarketView, account) -> float | None | Order | list[Order]:
        """Decide what to do, knowing only ``view``."""

    def on_finish(self, result) -> None:
        """Called once after the last bar, with the finished result."""

    def __repr__(self) -> str:  # pragma: no cover - display helper
        return f"<{type(self).__name__} name={self.name!r}>"


class BuyAndHold(Strategy):
    """Buy on the first bar and never trade again.

    The benchmark every strategy must beat. A great many "profitable"
    strategies turn out to be a worse, more expensive way of being long.
    """

    name = "buy & hold"

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def on_bar(self, view, account):
        return self.weight if len(view) == 1 else None


class ConstantWeight(Strategy):
    """Hold a fixed target weight, rebalancing every bar.

    Mostly useful as a turnover demonstration: identical exposure to
    :class:`BuyAndHold`, ruinously more expensive, because rebalancing to a
    constant weight means trading every single bar.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.name = f"constant weight {weight:g}"

    def on_bar(self, view, account):
        return self.weight


class SMACrossover(Strategy):
    """Long when the fast moving average is above the slow one, else flat (or short).

    The canonical first strategy. It is not good. It is here because it is
    simple enough that every bias we later demonstrate is visible in it.
    """

    def __init__(self, fast: int = 24, slow: int = 96, *, allow_short: bool = False):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow = fast, slow
        self.allow_short = allow_short
        self.warmup = slow
        self.name = f"SMA {fast}/{slow}" + (" L/S" if allow_short else " long-only")

    def on_bar(self, view, account):
        if len(view) < self.warmup:
            return None
        closes = view.close.history(self.slow)
        fast_ma = closes[-self.fast:].mean()
        slow_ma = closes.mean()
        if fast_ma > slow_ma:
            return 1.0
        return -1.0 if self.allow_short else 0.0


class MomentumStrategy(Strategy):
    """Long if the trailing ``lookback``-bar return is positive, short if negative."""

    def __init__(self, lookback: int = 48, *, allow_short: bool = True, threshold: float = 0.0):
        self.lookback = lookback
        self.allow_short = allow_short
        self.threshold = threshold
        self.warmup = lookback + 1
        self.name = f"momentum {lookback}b"

    def on_bar(self, view, account):
        if len(view) < self.warmup:
            return None
        closes = view.close.history(self.lookback + 1)
        ret = closes[-1] / closes[0] - 1.0
        if ret > self.threshold:
            return 1.0
        if ret < -self.threshold:
            return -1.0 if self.allow_short else 0.0
        return 0.0


class ZScoreReversion(Strategy):
    """Fade deviations from a trailing mean.

    Short when price is ``entry`` trailing standard deviations above its
    trailing mean, long when it is that far below, flat inside ``exit``.
    Works on genuinely mean-reverting series and loses steadily on trending
    ones -- which is the point of having both synthetic datasets.
    """

    def __init__(self, window: int = 48, entry: float = 1.5, exit: float = 0.3, *, allow_short: bool = True):
        self.window, self.entry, self.exit = window, entry, exit
        self.allow_short = allow_short
        self.warmup = window
        self.name = f"z-reversion {window}b +/-{entry}"

    def on_bar(self, view, account):
        if len(view) < self.warmup:
            return None
        closes = np.log(view.close.history(self.window))
        sd = closes.std(ddof=1)
        if sd <= 0:
            return None
        z = (closes[-1] - closes.mean()) / sd
        if z > self.entry:
            return -1.0 if self.allow_short else 0.0
        if z < -self.entry:
            return 1.0
        if abs(z) < self.exit:
            return 0.0
        return None


class RandomStrategy(Strategy):
    """Flip a coin every ``every`` bars. Trades for no reason whatsoever.

    Essential equipment. Run a few hundred of these and the best one will have
    a Sharpe ratio you would be delighted to publish. That is the data-snooping
    lesson, and you cannot teach it without a control group.
    """

    def __init__(self, seed: int = 0, every: int = 24, p_long: float = 0.5):
        self.seed, self.every, self.p_long = seed, every, p_long
        self._rng = np.random.default_rng(seed)
        self.name = f"random(seed={seed})"

    def on_start(self, view):
        self._rng = np.random.default_rng(self.seed)

    def on_bar(self, view, account):
        if len(view) % self.every != 0:
            return None
        return 1.0 if self._rng.random() < self.p_long else -1.0


class FunctionStrategy(Strategy):
    """Wrap a plain function ``f(view, account) -> weight`` as a Strategy.

    Convenient in notebooks. The function still only ever sees the view, so the
    look-ahead guard applies exactly as it does to a subclass.
    """

    def __init__(self, fn, name: str = "function", warmup: int = 0):
        self.fn, self.name, self.warmup = fn, name, warmup

    def on_bar(self, view, account):
        return self.fn(view, account)
