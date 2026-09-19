"""Prove every indicator and feature is causal, by trying to break it.

Method: compute the indicator on a series, then corrupt the *future* half of
that series and recompute. If any value in the past half moves, the indicator
read the future. This catches centred windows, backfills and negative shifts
that a code review would miss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrade import indicators as ind
from algotrade.features import make_features
from algotrade.synthetic import generate

SPLIT = 600
N = 1000

INDICATORS = {
    "sma": lambda d: ind.sma(d["close"], 20),
    "ema": lambda d: ind.ema(d["close"], 20),
    "rolling_std": lambda d: ind.rolling_std(d["close"], 20),
    "zscore": lambda d: ind.zscore(d["close"], 20),
    "roc": lambda d: ind.roc(d["close"], 12),
    "rsi": lambda d: ind.rsi(d["close"], 14),
    "macd_line": lambda d: ind.macd(d["close"])["macd"],
    "macd_hist": lambda d: ind.macd(d["close"])["histogram"],
    "bollinger_upper": lambda d: ind.bollinger(d["close"])["upper"],
    "bollinger_width": lambda d: ind.bollinger(d["close"])["width"],
    "true_range": lambda d: ind.true_range(d["high"], d["low"], d["close"]),
    "atr": lambda d: ind.atr(d["high"], d["low"], d["close"]),
    "realized_vol": lambda d: ind.realized_volatility(d["close"].pct_change(), 24),
    "donchian_upper": lambda d: ind.donchian(d["high"], d["low"])["upper"],
    "donchian_lower": lambda d: ind.donchian(d["high"], d["low"])["lower"],
    "cross_above": lambda d: ind.cross_above(ind.sma(d["close"], 5), ind.sma(d["close"], 20)).astype(float),
}


@pytest.fixture(scope="module")
def pair():
    """A dataset and a copy whose future has been violently corrupted."""
    df, _ = generate("gbm", n=N, seed=555)
    corrupted = df.copy()
    corrupted.iloc[SPLIT:] = corrupted.iloc[SPLIT:] * 3.0
    return df, corrupted


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_ignores_the_future(name, pair):
    df, corrupted = pair
    fn = INDICATORS[name]
    a = fn(df).iloc[:SPLIT]
    b = fn(corrupted).iloc[:SPLIT]
    pd.testing.assert_series_equal(
        a, b, check_names=False,
        obj=f"{name} changed in the past when the future was corrupted -- it is not causal",
    )


def test_all_features_are_causal(pair):
    df, corrupted = pair
    a = make_features(df).iloc[:SPLIT]
    b = make_features(corrupted).iloc[:SPLIT]
    moved = [c for c in a.columns if not a[c].equals(b[c])]
    assert not moved, f"non-causal feature columns: {moved}"


def test_centred_window_would_be_caught(pair):
    """Sanity check on the method itself.

    If this test did not fail for a deliberately non-causal function, the
    whole file would be proving nothing.
    """
    df, corrupted = pair
    bad = lambda d: d["close"].rolling(21, center=True).mean()  # noqa: E731
    a, b = bad(df).iloc[:SPLIT], bad(corrupted).iloc[:SPLIT]
    assert not a.equals(b), "the causality probe is not sensitive enough to detect a centred window"


def test_indicators_have_a_nan_warmup(pair):
    """An indicator with no NaN prefix has invented data for its warm-up."""
    df, _ = pair
    assert ind.sma(df["close"], 20).iloc[:19].isna().all()
    assert ind.zscore(df["close"], 20).iloc[:19].isna().all()
    assert ind.donchian(df["high"], df["low"], 20)["upper"].iloc[:20].isna().all()
