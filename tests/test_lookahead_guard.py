"""THE MOST IMPORTANT TEST FILE IN THIS REPOSITORY.

The workshop's central claim is that look-ahead bias is *structurally
impossible* in this backtester, not merely discouraged. A claim like that is
worthless without a test that tries to break it.

So here we write strategies that deliberately cheat, in every way a student
plausibly might, and assert that each one fails loudly rather than quietly
returning a wonderful equity curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrade.backtest import BacktestConfig, run_backtest
from algotrade.errors import InsufficientHistoryError, LookAheadError
from algotrade.strategy import Strategy
from algotrade.view import Clock, MarketView, SeriesView


# --------------------------------------------------------------------------
# Strategies that cheat. Every one of these must raise.
# --------------------------------------------------------------------------
class PeeksAtNextBar(Strategy):
    """The direct attempt: read the bar after the current one."""

    name = "cheater: view.close[i + 1]"

    def on_bar(self, view, account):
        return 1.0 if view.close[view.i + 1] > view.close[-1] else -1.0


class PeeksViaPositiveIndex(Strategy):
    """Indexes by absolute bar number, one past the end of the visible window."""

    name = "cheater: close[len(view)]"

    def on_bar(self, view, account):
        return float(view.close[len(view)])


class PeeksViaSlice(Strategy):
    """Asks for a slice whose stop lies in the future."""

    name = "cheater: close[0:i+5]"

    def on_bar(self, view, account):
        future = view.close[0 : view.i + 5]
        return float(np.sign(future[-1] - future[-2]))


class PeeksAtFullLength(Strategy):
    """Uses total_bars to index past the present."""

    name = "cheater: close[total_bars - 1]"

    def on_bar(self, view, account):
        return float(view.close[view.total_bars - 1] > view.close[-1])


class AsksForTooMuchHistory(Strategy):
    """Requests more history than has elapsed -- a subtler form of the same bug."""

    name = "cheater: history(10_000)"

    def on_bar(self, view, account):
        return float(view.close.history(10_000).mean())


CHEATERS = [
    PeeksAtNextBar,
    PeeksViaPositiveIndex,
    PeeksViaSlice,
    PeeksAtFullLength,
]


@pytest.mark.parametrize("cheater", CHEATERS, ids=lambda c: c.name)
def test_cheating_strategy_raises_lookahead_error(cheater, small_data):
    """A strategy that reaches into the future must crash, not profit."""
    with pytest.raises(LookAheadError) as excinfo:
        run_backtest(small_data, cheater())
    # The message must name the problem, because a student will read it at 2am.
    assert "LOOK-AHEAD BLOCKED" in str(excinfo.value)


def test_over_long_history_request_raises(small_data):
    with pytest.raises(InsufficientHistoryError):
        run_backtest(small_data, AsksForTooMuchHistory())


# --------------------------------------------------------------------------
# The guard itself, unit by unit.
# --------------------------------------------------------------------------
@pytest.fixture
def view_at_bar_10():
    idx = pd.date_range("2021-01-01", periods=100, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.arange(100.0),
            "high": np.arange(100.0) + 1,
            "low": np.arange(100.0) - 1,
            "close": np.arange(100.0),
            "volume": np.full(100, 10.0),
        },
        index=idx,
    )
    clock = Clock(10)
    return MarketView(frame, clock), clock


def test_current_bar_is_visible(view_at_bar_10):
    view, _ = view_at_bar_10
    assert view.close[-1] == 10.0
    assert view.close.last == 10.0
    assert view.close[10] == 10.0


def test_len_is_bars_elapsed(view_at_bar_10):
    view, _ = view_at_bar_10
    assert len(view) == 11
    assert len(view.close) == 11


def test_one_bar_past_is_blocked(view_at_bar_10):
    view, _ = view_at_bar_10
    with pytest.raises(LookAheadError):
        view.close[11]


def test_far_future_is_blocked(view_at_bar_10):
    view, _ = view_at_bar_10
    with pytest.raises(LookAheadError):
        view.close[99]


def test_negative_index_walks_backwards_only(view_at_bar_10):
    view, _ = view_at_bar_10
    assert view.close[-1] == 10.0
    assert view.close[-11] == 0.0
    with pytest.raises(IndexError):
        view.close[-12]


def test_open_ended_slice_is_clipped_to_present(view_at_bar_10):
    view, _ = view_at_bar_10
    assert list(view.close[:]) == list(range(11))
    assert list(view.close[-3:]) == [8.0, 9.0, 10.0]


def test_explicit_future_slice_raises(view_at_bar_10):
    view, _ = view_at_bar_10
    with pytest.raises(LookAheadError):
        view.close[0:50]


def test_numpy_conversion_sees_only_the_past(view_at_bar_10):
    """np.asarray must not be an escape hatch."""
    view, _ = view_at_bar_10
    arr = np.asarray(view.close)
    assert len(arr) == 11
    assert arr[-1] == 10.0


def test_iteration_sees_only_the_past(view_at_bar_10):
    view, _ = view_at_bar_10
    assert len(list(view.close)) == 11


def test_frame_sees_only_the_past(view_at_bar_10):
    view, _ = view_at_bar_10
    assert len(view.frame()) == 11
    assert len(view.frame(3)) == 3
    assert len(view.index) == 11


def test_view_data_is_immutable(view_at_bar_10):
    """A strategy must not be able to edit the market."""
    view, _ = view_at_bar_10
    arr = np.asarray(view.close)
    arr[0] = -999.0                       # a copy; harmless
    assert view.close[0] == 0.0
    with pytest.raises(ValueError):
        view.close._values[0] = -999.0    # the source array is read-only


def test_history_returns_a_copy(view_at_bar_10):
    view, _ = view_at_bar_10
    h = view.close.history(3)
    h[:] = 0.0
    assert view.close[-1] == 10.0


def test_clock_cannot_run_backwards(view_at_bar_10):
    view, _ = view_at_bar_10
    with pytest.raises(ValueError):
        view._advance(5)


def test_guard_holds_as_the_clock_advances():
    """Sweep every bar and confirm the boundary moves with the clock."""
    clock = Clock(-1)
    series = SeriesView(np.arange(50.0), clock, name="close")
    for i in range(50):
        clock.i = i
        assert series[-1] == float(i)
        assert len(series) == i + 1
        with pytest.raises(LookAheadError):
            series[i + 1]


def test_honest_strategy_runs_clean(small_data):
    """The guard must not obstruct a correctly-written strategy."""

    class Honest(Strategy):
        name = "honest"
        warmup = 20

        def on_bar(self, view, account):
            if len(view) < self.warmup:
                return None
            hist = view.close.history(20)
            return 1.0 if hist[-1] > hist.mean() else 0.0

    result = run_backtest(small_data, Honest(), BacktestConfig())
    assert len(result.equity) == len(small_data)
    assert np.isfinite(result.equity).all()
