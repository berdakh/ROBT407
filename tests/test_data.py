"""Data loading and the integrity checks that stop a corrupt file silently."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrade.data import (
    data_quality_report, find_gaps, load_ohlcv, log_returns, resample_ohlcv,
    simple_returns, train_test_split_time, validate_ohlcv,
)
from algotrade.errors import DataIntegrityError
from algotrade.synthetic import generate


def test_valid_frame_passes(gbm_data):
    validate_ohlcv(gbm_data)


def test_naive_timestamps_rejected(gbm_data):
    bad = gbm_data.copy()
    bad.index = bad.index.tz_localize(None)
    with pytest.raises(DataIntegrityError, match="timezone-naive"):
        validate_ohlcv(bad)


def test_unsorted_index_rejected(gbm_data):
    bad = gbm_data.iloc[::-1]
    with pytest.raises(DataIntegrityError, match="not sorted"):
        validate_ohlcv(bad)


def test_duplicate_timestamps_rejected(gbm_data):
    bad = pd.concat([gbm_data, gbm_data.iloc[[5]]]).sort_index()
    with pytest.raises(DataIntegrityError, match="[Dd]uplicated"):
        validate_ohlcv(bad)


def test_high_below_close_rejected(gbm_data):
    bad = gbm_data.copy()
    bad.iloc[10, bad.columns.get_loc("high")] = bad["low"].iloc[10] * 0.5
    with pytest.raises(DataIntegrityError, match="violate"):
        validate_ohlcv(bad)


def test_negative_volume_rejected(gbm_data):
    bad = gbm_data.copy()
    bad.iloc[3, bad.columns.get_loc("volume")] = -1.0
    with pytest.raises(DataIntegrityError, match="negative volume"):
        validate_ohlcv(bad)


def test_nan_close_rejected(gbm_data):
    bad = gbm_data.copy()
    bad.iloc[7, bad.columns.get_loc("close")] = np.nan
    with pytest.raises(DataIntegrityError, match="NaN"):
        validate_ohlcv(bad)


def test_missing_column_rejected(gbm_data):
    with pytest.raises(DataIntegrityError, match="missing required columns"):
        validate_ohlcv(gbm_data.drop(columns=["volume"]))


def test_roundtrip_csv(tmp_path, gbm_data):
    path = tmp_path / "x.csv"
    gbm_data.to_csv(path)
    back = load_ohlcv(path)
    pd.testing.assert_frame_equal(gbm_data, back[gbm_data.columns], check_freq=False)


def test_missing_file_message_is_helpful(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/synthetic"):
        load_ohlcv(tmp_path / "nope.csv")


def test_resample_uses_correct_aggregations(gbm_data):
    daily = resample_ohlcv(gbm_data, "1D")
    first_day = gbm_data.loc[daily.index[0] : daily.index[0] + pd.Timedelta("23h")]
    assert daily["open"].iloc[0] == first_day["open"].iloc[0]      # first open
    assert daily["close"].iloc[0] == first_day["close"].iloc[-1]    # LAST close
    assert daily["high"].iloc[0] == first_day["high"].max()
    assert daily["low"].iloc[0] == first_day["low"].min()
    assert np.isclose(daily["volume"].iloc[0], first_day["volume"].sum())


def test_resampled_frame_is_still_valid(gbm_data):
    validate_ohlcv(resample_ohlcv(gbm_data, "4h"))


def test_simple_vs_log_returns_agree_for_small_moves(gbm_data):
    s = simple_returns(gbm_data["close"]).dropna()
    l = log_returns(gbm_data["close"]).dropna()
    assert np.allclose(np.expm1(l), s)
    # And that log returns add across time while simple returns do not.
    two_bar_log = log_returns(gbm_data["close"], 2).dropna()
    assert np.allclose(two_bar_log, (l + l.shift(1)).dropna())


def test_find_gaps_detects_a_hole(gbm_data):
    holed = gbm_data.drop(gbm_data.index[100:110])
    gaps = find_gaps(holed, "1h")
    assert len(gaps) == 1
    assert gaps["missing_bars"].iloc[0] == 10


def test_find_gaps_clean_data(gbm_data):
    assert len(find_gaps(gbm_data, "1h")) == 0


def test_quality_report_shape(gbm_data):
    rep = data_quality_report(gbm_data, "1h")
    assert rep["rows"] == len(gbm_data)
    assert rep["duplicated_timestamps"] == 0
    assert rep["gap_count"] == 0


def test_time_split_does_not_overlap(gbm_data):
    train, test = train_test_split_time(gbm_data, 0.7)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(gbm_data)
