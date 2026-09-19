"""Read-only, time-truncated views over market data.

THE CENTRAL IDEA OF THIS PACKAGE
--------------------------------
A backtest lies to you when your strategy sees data that did not exist yet at
the moment it made a decision. The usual defence is discipline: "remember to
shift your signal by one bar". Discipline fails, especially at 2am the night
before a deadline.

So we do not rely on discipline. The backtester never hands a strategy the full
price series. It hands it a :class:`MarketView`, which exposes each OHLCV
column as a :class:`SeriesView` that physically refuses to return a value from
a bar the strategy has not reached yet. Reaching forward raises
:class:`~algotrade.errors.LookAheadError`.

    >>> import numpy as np
    >>> from algotrade.view import SeriesView, Clock
    >>> clock = Clock()
    >>> s = SeriesView(np.arange(100.0), clock, name="close")
    >>> clock.i = 10          # the engine has advanced to bar 10
    >>> s[-1]                 # most recent observed bar: fine
    10.0
    >>> len(s)                # 11 bars are visible (0..10)
    11
    >>> s[11]                 # tomorrow: refused
    Traceback (most recent call last):
        ...
    algotrade.errors.LookAheadError: ...

HOW HONEST IS THIS?
-------------------
Honest about accidents, not about sabotage. ``SeriesView`` stores the full
array on a private attribute, so a determined student can reach it with
``s._values``. That is fine. The goal is to make look-ahead bias impossible to
commit *by accident* -- via an off-by-one, a forgotten ``.shift(1)``, a
``df['close'].rolling(20).mean()`` computed over the whole history before the
loop starts. Those are the mistakes that actually happen. Deliberately
defeating the guard requires typing an underscore, which is a decision, not a
slip.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .errors import InsufficientHistoryError, LookAheadError

__all__ = ["Clock", "SeriesView", "MarketView"]


class Clock:
    """A mutable 'current bar index', shared by every view of one backtest.

    The engine owns the clock and advances it once per bar. Views hold a
    reference to it, so a view created before the loop starts stays correct
    throughout -- there is no per-bar object churn.
    """

    __slots__ = ("i",)

    def __init__(self, i: int = -1) -> None:
        self.i = i

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Clock(i={self.i})"


class SeriesView:
    """A read-only window over one array, truncated at the clock's current bar.

    Indexing follows normal Python conventions *within the visible window*:

    * ``s[-1]`` is the current bar, ``s[-2]`` the one before it.
    * ``s[0]`` is the first bar of the dataset.
    * ``len(s)`` is the number of bars observed so far, so ``s[len(s) - 1]``
      is the current bar and ``s[len(s)]`` raises
      :class:`~algotrade.errors.LookAheadError`.
    * Slices are clipped to the visible window only when open-ended
      (``s[-20:]``, ``s[:]``). A slice that explicitly names a future stop
      raises, rather than silently returning a short array.

    Every method returns a **copy**, so a strategy cannot mutate the market.
    """

    __slots__ = ("_values", "_clock", "_name")

    def __init__(self, values: np.ndarray, clock: Clock, name: str = "") -> None:
        arr = np.asarray(values, dtype=float)
        arr.flags.writeable = False
        self._values = arr
        self._clock = clock
        self._name = name

    # -- size -------------------------------------------------------------
    @property
    def _limit(self) -> int:
        """Index of the last bar the strategy is allowed to see."""
        return self._clock.i

    def __len__(self) -> int:
        return max(0, self._limit + 1)

    # -- access -----------------------------------------------------------
    def __getitem__(self, key):
        limit = self._limit
        n_visible = limit + 1

        if isinstance(key, slice):
            return self._get_slice(key, limit, n_visible)

        if isinstance(key, (bool, np.bool_)):
            raise TypeError("SeriesView does not support boolean indexing")

        if isinstance(key, (int, np.integer)):
            idx = int(key)
            if idx < 0:
                idx += n_visible
                if idx < 0:
                    raise IndexError(
                        f"{self._name}[{key}] reaches back before the start of "
                        f"the data (only {n_visible} bars have elapsed)"
                    )
            elif idx > limit:
                raise LookAheadError(
                    f"LOOK-AHEAD BLOCKED: {self._name}[{key}] asks for bar {idx}, "
                    f"but the current bar is {limit}. Your strategy cannot know "
                    f"that value yet. Use {self._name}[-1] for the current bar."
                )
            return float(self._values[idx])

        raise TypeError(
            f"SeriesView indices must be integers or slices, not {type(key).__name__}"
        )

    def _get_slice(self, key: slice, limit: int, n_visible: int) -> np.ndarray:
        start, stop, step = key.start, key.stop, key.step

        if stop is not None:
            stop_i = int(stop)
            if stop_i < 0:
                stop_i += n_visible
            if stop_i > n_visible:
                raise LookAheadError(
                    f"LOOK-AHEAD BLOCKED: {self._name}[{start}:{stop}] would read up "
                    f"to bar {stop_i - 1}, but the current bar is {limit}. "
                    f"Use an open-ended slice like {self._name}[-20:] instead."
                )
        else:
            stop_i = n_visible

        if start is None:
            start_i = 0
        else:
            start_i = int(start)
            if start_i < 0:
                start_i = max(0, start_i + n_visible)
            elif start_i > n_visible:
                raise LookAheadError(
                    f"LOOK-AHEAD BLOCKED: {self._name}[{start}:{stop}] starts at bar "
                    f"{start_i}, past the current bar {limit}."
                )

        return self._values[start_i:stop_i:step].copy()

    def __iter__(self) -> Iterator[float]:
        return iter(self._values[: self._limit + 1].copy())

    def __array__(self, dtype=None, copy=None):
        """Support ``np.asarray(view.close)`` -- visible window only."""
        out = self._values[: self._limit + 1].copy()
        return out if dtype is None else out.astype(dtype)

    # -- convenience ------------------------------------------------------
    @property
    def last(self) -> float:
        """Value at the current bar."""
        if self._limit < 0:
            raise InsufficientHistoryError("no bars have elapsed yet")
        return float(self._values[self._limit])

    def history(self, n: int) -> np.ndarray:
        """The last ``n`` observed values as a fresh array (oldest first).

        Raises :class:`~algotrade.errors.InsufficientHistoryError` if fewer
        than ``n`` bars have elapsed, so a strategy cannot quietly compute a
        20-bar average from 3 bars.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        available = self._limit + 1
        if n > available:
            raise InsufficientHistoryError(
                f"{self._name}.history({n}) needs {n} bars but only {available} "
                f"have elapsed. Guard with `if len(view) < {n}: return None`."
            )
        return self._values[self._limit + 1 - n : self._limit + 1].copy()

    def to_series(self, n: int | None = None, index=None) -> pd.Series:
        """Visible window as a pandas Series (optionally the last ``n`` bars)."""
        values = self._values[: self._limit + 1].copy()
        if n is not None:
            values = values[-n:]
        if index is not None:
            index = index[: self._limit + 1]
            if n is not None:
                index = index[-n:]
        return pd.Series(values, index=index, name=self._name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        n = len(self)
        if n == 0:
            return f"<SeriesView {self._name!r} empty>"
        tail = ", ".join(f"{v:g}" for v in self._values[max(0, n - 3) : n])
        return f"<SeriesView {self._name!r} n={n} ... {tail}>"


class MarketView:
    """Everything a strategy is allowed to know at the current bar.

    Exposes ``open``, ``high``, ``low``, ``close``, ``volume`` as
    :class:`SeriesView` objects, plus the current timestamp and bar index.
    Extra columns present in the source frame are available through
    :meth:`column`.
    """

    _OHLCV = ("open", "high", "low", "close", "volume")

    def __init__(self, frame: pd.DataFrame, clock: Clock | None = None) -> None:
        self._clock = clock if clock is not None else Clock()
        self._index = frame.index
        self._columns: dict[str, SeriesView] = {}
        for col in frame.columns:
            key = str(col).lower()
            self._columns[key] = SeriesView(frame[col].to_numpy(dtype=float), self._clock, name=key)
        missing = [c for c in self._OHLCV if c not in self._columns]
        if missing:
            raise ValueError(f"frame is missing required columns: {missing}")
        self._n = len(frame)

    # -- OHLCV accessors --------------------------------------------------
    @property
    def open(self) -> SeriesView:
        return self._columns["open"]

    @property
    def high(self) -> SeriesView:
        return self._columns["high"]

    @property
    def low(self) -> SeriesView:
        return self._columns["low"]

    @property
    def close(self) -> SeriesView:
        return self._columns["close"]

    @property
    def volume(self) -> SeriesView:
        return self._columns["volume"]

    def column(self, name: str) -> SeriesView:
        """A non-OHLCV column (e.g. a precomputed feature) as a SeriesView."""
        key = name.lower()
        if key not in self._columns:
            raise KeyError(f"no column {name!r}; available: {sorted(self._columns)}")
        return self._columns[key]

    def has_column(self, name: str) -> bool:
        return name.lower() in self._columns

    # -- time -------------------------------------------------------------
    @property
    def i(self) -> int:
        """Index of the current bar."""
        return self._clock.i

    @property
    def now(self) -> pd.Timestamp:
        """Timestamp of the current bar's close."""
        if self._clock.i < 0:
            raise InsufficientHistoryError("no bars have elapsed yet")
        return self._index[self._clock.i]

    @property
    def index(self) -> pd.DatetimeIndex:
        """Timestamps observed so far (a copy)."""
        return self._index[: self._clock.i + 1].copy()

    def __len__(self) -> int:
        """Number of bars observed so far."""
        return max(0, self._clock.i + 1)

    @property
    def total_bars(self) -> int:
        """Length of the whole dataset.

        Knowing how long the backtest is does not let you see prices, but it
        *is* information a live strategy would not have. Do not branch on it.
        """
        return self._n

    # -- frames -----------------------------------------------------------
    def frame(self, n: int | None = None) -> pd.DataFrame:
        """Recent visible bars as a pandas DataFrame copy.

        Convenient, but allocating a DataFrame every bar is slow. Prefer
        :meth:`SeriesView.history` inside hot loops.
        """
        stop = self._clock.i + 1
        start = 0 if n is None else max(0, stop - n)
        data = {name: sv._values[start:stop].copy() for name, sv in self._columns.items()}
        return pd.DataFrame(data, index=self._index[start:stop])

    # -- engine-facing ----------------------------------------------------
    def _advance(self, i: int) -> None:
        """Move the clock forward. Called by the engine only."""
        if i < self._clock.i:
            raise ValueError("time does not run backwards")
        self._clock.i = i

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self._clock.i < 0:
            return "<MarketView not started>"
        return f"<MarketView bar {self._clock.i}/{self._n - 1} at {self.now} close={self.close.last:g}>"
