"""Loading, validating and reshaping OHLCV market data.

Market data is dirtier than students expect. Timestamps arrive in three
conventions, exchanges go down, bars go missing, and the same file can mean
"bar opened at 09:00" or "bar closed at 09:00" depending on who wrote it.
Every function here states its convention explicitly.

CONVENTION USED THROUGHOUT THIS WORKSHOP
    The index timestamp is the bar's OPEN time, in UTC.
    A bar labelled 09:00 with 1h bars covers [09:00, 10:00).
    Its close price is only known at 10:00.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .errors import DataIntegrityError

__all__ = [
    "OHLCV_COLUMNS",
    "load_ohlcv",
    "validate_ohlcv",
    "resample_ohlcv",
    "simple_returns",
    "log_returns",
    "find_gaps",
    "data_quality_report",
    "train_test_split_time",
    "BARS_PER_YEAR",
]

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

#: Bars per year for common frequencies. Crypto trades 24/7, so unlike equities
#: there is no 252-trading-day fudge: a year really is 365 days of bars.
BARS_PER_YEAR = {
    "1min": 365 * 24 * 60,
    "5min": 365 * 24 * 12,
    "15min": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
    "1w": 52,
}


def load_ohlcv(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp",
    validate: bool = True,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Load an OHLCV CSV into a validated, UTC-indexed DataFrame.

    Parameters
    ----------
    path:
        CSV with a timestamp column plus open/high/low/close/volume.
    validate:
        Run :func:`validate_ohlcv` and raise on any integrity problem. Leave
        this on. The one time you turn it off is the one time the file has a
        duplicated bar.

    Returns
    -------
    DataFrame indexed by UTC bar-open timestamps, columns lower-cased.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Sample data lives in data/synthetic/; "
            f"real data can be downloaded with scripts/fetch_market_data.py"
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    ts_col = timestamp_column.lower()
    if ts_col not in df.columns:
        candidates = [c for c in ("timestamp", "time", "date", "datetime", "open_time") if c in df.columns]
        if not candidates:
            raise DataIntegrityError(
                f"no timestamp column in {path.name}; found columns {list(df.columns)}"
            )
        ts_col = candidates[0]

    index = pd.to_datetime(df[ts_col], utc=True, format="mixed")
    df = df.drop(columns=[ts_col])
    df.index = pd.DatetimeIndex(index, name="timestamp")
    if tz != "UTC":
        df.index = df.index.tz_convert(tz)

    keep = [c for c in OHLCV_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in OHLCV_COLUMNS]
    df = df[keep + extras].astype({c: float for c in keep})

    if validate:
        validate_ohlcv(df)
    return df


def validate_ohlcv(df: pd.DataFrame, *, require_volume: bool = True) -> None:
    """Raise :class:`DataIntegrityError` on anything that would corrupt a backtest.

    Checks, in order of how often they bite in practice:

    1. Required columns present.
    2. Index is a timezone-aware DatetimeIndex (naive timestamps silently
       misalign when you later merge two sources).
    3. Index is strictly increasing (a shuffled file makes every rolling
       window meaningless).
    4. No duplicated timestamps (double-counts a bar's return).
    5. No NaNs in OHLC.
    6. ``low <= min(open, close) <= max(open, close) <= high``.
    7. Prices strictly positive; volume non-negative.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if require_volume is False:
        missing = [c for c in missing if c != "volume"]
    if missing:
        raise DataIntegrityError(f"missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataIntegrityError(f"index must be a DatetimeIndex, got {type(df.index).__name__}")
    if df.index.tz is None:
        raise DataIntegrityError(
            "index is timezone-naive. Localise it to UTC -- naive timestamps are "
            "the reason your data 'mysteriously' shifts by an hour twice a year."
        )
    if not df.index.is_monotonic_increasing:
        raise DataIntegrityError("index is not sorted ascending")
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()[:5]
        raise DataIntegrityError(f"duplicated timestamps, e.g. {list(dupes)}")

    ohlc = [c for c in ("open", "high", "low", "close") if c in df.columns]
    nan_counts = df[ohlc].isna().sum()
    if nan_counts.any():
        raise DataIntegrityError(f"NaNs in OHLC: {nan_counts[nan_counts > 0].to_dict()}")

    if (df[ohlc] <= 0).to_numpy().any():
        raise DataIntegrityError("non-positive prices found")

    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    bad_high = df["high"] < body_high - 1e-9
    bad_low = df["low"] > body_low + 1e-9
    if bad_high.any() or bad_low.any():
        n = int(bad_high.sum() + bad_low.sum())
        first = df.index[(bad_high | bad_low)][0]
        raise DataIntegrityError(
            f"{n} bars violate low <= open/close <= high, first at {first}"
        )

    if "volume" in df.columns and (df["volume"] < 0).any():
        raise DataIntegrityError("negative volume found")


def resample_ohlcv(df: pd.DataFrame, rule: str, *, dropna: bool = True) -> pd.DataFrame:
    """Aggregate to a coarser frequency with the correct per-column rule.

    open=first, high=max, low=min, close=last, volume=sum. Taking ``mean`` of
    the close (which pandas would do by default if you called ``.resample().mean()``)
    invents prices that never traded.

    Only ever resample to a *coarser* frequency. Upsampling fabricates bars.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    for col in df.columns:
        if col not in agg:
            agg[col] = "last"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    if dropna:
        out = out.dropna(subset=[c for c in ("open", "close") if c in out.columns])
    return out


def simple_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Arithmetic returns: ``p_t / p_{t-1} - 1``.

    Use these when you need portfolio arithmetic: the return of a portfolio is
    the weighted average of its constituents' simple returns. That is not true
    of log returns.
    """
    return prices.pct_change(periods)


def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Log returns: ``ln(p_t / p_{t-1})``.

    Use these when you need time arithmetic: log returns add across time, so an
    n-bar return is just a rolling sum. They are also closer to symmetric,
    which matters for statistical tests.

    They are *not* interchangeable with simple returns. Summing log returns and
    then reporting the total as a percentage overstates losses and understates
    gains. Convert with ``expm1`` before showing a human a number.
    """
    return np.log(prices / prices.shift(periods))


def find_gaps(df: pd.DataFrame, expected_freq: str) -> pd.DataFrame:
    """Locate missing bars, assuming a regular ``expected_freq`` grid.

    Returns one row per gap with its start, end and the number of bars missing.
    An exchange outage looks exactly like a flat price if you forward-fill it
    without looking, which then looks like a low-volatility regime to your
    model.
    """
    expected = pd.date_range(df.index[0], df.index[-1], freq=expected_freq, tz=df.index.tz)
    missing = expected.difference(df.index)
    if len(missing) == 0:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_bars"])

    step = pd.Timedelta(expected.freq)
    breaks = np.where(np.diff(missing.values) != step.to_timedelta64())[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(missing) - 1]])
    return pd.DataFrame(
        {
            "gap_start": missing[starts],
            "gap_end": missing[ends],
            "missing_bars": ends - starts + 1,
        }
    )


def data_quality_report(df: pd.DataFrame, expected_freq: str | None = None) -> dict:
    """A dict of the things worth checking before you trust a dataset."""
    close = df["close"]
    rets = simple_returns(close).dropna()
    report = {
        "rows": len(df),
        "start": df.index[0],
        "end": df.index[-1],
        "span_days": (df.index[-1] - df.index[0]).total_seconds() / 86400,
        "timezone": str(df.index.tz),
        "monotonic": bool(df.index.is_monotonic_increasing),
        "duplicated_timestamps": int(df.index.duplicated().sum()),
        "nan_cells": int(df.isna().sum().sum()),
        "zero_volume_bars": int((df["volume"] == 0).sum()) if "volume" in df else None,
        "flat_bars": int((df["high"] == df["low"]).sum()),
        "max_abs_return": float(rets.abs().max()),
        "returns_over_20pct": int((rets.abs() > 0.20).sum()),
    }
    if expected_freq is not None:
        gaps = find_gaps(df, expected_freq)
        report["gap_count"] = len(gaps)
        report["missing_bars"] = int(gaps["missing_bars"].sum()) if len(gaps) else 0
    return report


def train_test_split_time(
    df: pd.DataFrame, split: float | str | pd.Timestamp = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically. Never shuffle time series.

    ``split`` is either a fraction of rows or a timestamp. The test set is
    everything at or after the cut. There is no overlap and no shuffling,
    because a model trained on Thursday to predict Wednesday is not a model.
    """
    if isinstance(split, float):
        cut = df.index[int(len(df) * split)]
    else:
        cut = pd.Timestamp(split, tz=df.index.tz)
    return df.loc[df.index < cut].copy(), df.loc[df.index >= cut].copy()
