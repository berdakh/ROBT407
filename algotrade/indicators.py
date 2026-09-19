"""Technical indicators. Every one of them is strictly causal.

CAUSALITY RULE
    The value of an indicator at time ``t`` depends only on data at times
    ``<= t``. No centred windows, no ``shift(-1)``, no ``interpolate()`` that
    reaches forwards, no ``bfill``.

This is not a style preference. ``pandas.Series.rolling(window, center=True)``
looks harmless in a chart and silently hands your strategy the future.
``tests/test_causality.py`` proves every function here obeys the rule by
mutating future values and checking that past outputs do not move.

A NOTE ON WHAT THESE ARE WORTH
    These are smoothing functions over a noisy series. None of them is magic
    and none of them "predicts" anything. They are useful because they compress
    a price history into a number you can reason about, not because a golden
    cross means anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "rolling_std", "zscore", "rsi", "macd", "bollinger",
    "true_range", "atr", "realized_volatility", "donchian", "rolling_max_drawdown",
    "roc", "cross_above", "cross_below",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over the trailing ``window`` bars."""
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average.

    ``adjust=False`` gives the recursive form used by every charting package,
    and matters here because it is the form a live strategy can compute
    incrementally from one stored state variable.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rolling_std(series: pd.Series, window: int, *, ddof: int = 1) -> pd.Series:
    """Trailing standard deviation."""
    return series.rolling(window, min_periods=window).std(ddof=ddof)


def zscore(series: pd.Series, window: int) -> pd.Series:
    """How many trailing standard deviations the current value sits from its trailing mean.

    The mean and standard deviation are computed over the trailing window
    *including* the current bar. Using the full-sample mean instead -- which is
    what ``(x - x.mean()) / x.std()`` does -- is look-ahead bias, and it is
    probably the single most common way students leak the future into a
    mean-reversion signal.
    """
    mu = sma(series, window)
    sd = rolling_std(series, window)
    return (series - mu) / sd.replace(0.0, np.nan)


def roc(series: pd.Series, periods: int) -> pd.Series:
    """Rate of change over ``periods`` bars: ``x_t / x_{t-n} - 1``."""
    return series.pct_change(periods)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index via Wilder's smoothing, on a 0-100 scale."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": macd_line - signal_line}
    )


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands: trailing mean plus/minus ``n_std`` trailing deviations."""
    mid = sma(series, window)
    sd = rolling_std(series, window)
    return pd.DataFrame(
        {"mid": mid, "upper": mid + n_std * sd, "lower": mid - n_std * sd, "width": 2 * n_std * sd / mid}
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: the bar's range, extended to include any overnight gap."""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (Wilder-smoothed). A volatility measure in price units."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def realized_volatility(
    returns: pd.Series, window: int = 24 * 7, bars_per_year: float = 365 * 24
) -> pd.Series:
    """Annualised trailing volatility of returns.

    Annualising by ``sqrt(bars_per_year)`` assumes returns are independent
    across bars. They are not -- volatility clusters -- so this systematically
    misstates true risk. It is still the standard, and knowing *why* it is
    wrong is more useful than avoiding it.
    """
    return rolling_std(returns, window) * np.sqrt(bars_per_year)


def donchian(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """Donchian channel: the trailing highest high and lowest low.

    Note the shift: the channel at bar ``t`` uses bars ``t-window .. t-1``, so
    "price breaks above the channel" is a statement about the current bar
    exceeding *past* bars. Without the shift the current bar is inside its own
    channel and a breakout can never trigger.
    """
    return pd.DataFrame(
        {
            "upper": high.rolling(window, min_periods=window).max().shift(1),
            "lower": low.rolling(window, min_periods=window).min().shift(1),
        }
    )


def rolling_max_drawdown(equity: pd.Series, window: int) -> pd.Series:
    """Worst peak-to-trough decline within each trailing window."""
    roll_max = equity.rolling(window, min_periods=1).max()
    return (equity / roll_max - 1.0).rolling(window, min_periods=1).min()


def cross_above(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on the bar where ``fast`` crosses from below ``slow`` to above."""
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def cross_below(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """True on the bar where ``fast`` crosses from above ``slow`` to below."""
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))
