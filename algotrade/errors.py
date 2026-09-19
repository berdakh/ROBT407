"""Exceptions raised by the algotrade package.

These are deliberately loud. A backtest that silently does the wrong thing is
worse than one that crashes, because the silent one produces a number you will
believe.
"""

from __future__ import annotations


class AlgotradeError(Exception):
    """Base class for every error raised by this package."""


class LookAheadError(AlgotradeError):
    """Raised when strategy code tries to read data it could not have known yet.

    This is the single most important exception in the package. It is raised by
    :class:`algotrade.view.SeriesView` whenever an index reaches past the
    current bar. If you see this, the backtester just stopped you from
    producing a beautiful, worthless equity curve.
    """


class DataIntegrityError(AlgotradeError):
    """Raised when an OHLCV frame violates an invariant we rely on.

    Examples: a non-monotonic timestamp index, duplicated timestamps, a bar
    whose high is below its low, or a negative volume.
    """


class InsufficientHistoryError(AlgotradeError):
    """Raised when a strategy asks for more history than has elapsed."""


class RiskLimitError(AlgotradeError):
    """Raised when an order would breach a hard risk limit."""
