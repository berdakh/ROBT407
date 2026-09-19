"""Splits, labels and features: the leakage surface of the ML half."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algotrade.features import check_stationarity, make_features
from algotrade.labeling import fixed_horizon_class, fixed_horizon_return, triple_barrier
from algotrade.validation import (
    assert_no_leakage, describe_splits, expanding_window_splits,
    purged_kfold_splits, walk_forward_splits,
)


# -- splits ----------------------------------------------------------------
def test_walk_forward_is_strictly_chronological():
    splits = walk_forward_splits(5000, train_size=1000, test_size=250)
    for sp in splits:
        assert sp.train.max() < sp.test.min()
    assert_no_leakage(splits)


def test_walk_forward_test_windows_do_not_overlap():
    splits = walk_forward_splits(5000, train_size=1000, test_size=250)
    seen = np.concatenate([sp.test for sp in splits])
    assert len(seen) == len(np.unique(seen))


def test_purge_creates_the_requested_gap():
    splits = walk_forward_splits(5000, train_size=1000, test_size=250, purge=48)
    for sp in splits:
        assert sp.test.min() - sp.train.max() - 1 >= 48


def test_leakage_detector_catches_a_missing_purge():
    splits = walk_forward_splits(5000, train_size=1000, test_size=250, purge=0)
    with pytest.raises(AssertionError, match="purge"):
        assert_no_leakage(splits, label_horizon=24)


def test_leakage_detector_catches_direct_overlap():
    from algotrade.validation import Split

    bad = [Split(train=np.arange(0, 100), test=np.arange(90, 150), fold=0)]
    with pytest.raises(AssertionError, match="BOTH train and test"):
        assert_no_leakage(bad)


def test_leakage_detector_catches_training_on_the_future():
    from algotrade.validation import Split

    bad = [Split(train=np.arange(200, 300), test=np.arange(0, 100), fold=0)]
    with pytest.raises(AssertionError, match="trained on the future"):
        assert_no_leakage(bad)


def test_shuffled_split_would_be_caught():
    """Proves the detector is sensitive to the mistake it exists to catch."""
    from algotrade.validation import Split

    rng = np.random.default_rng(0)
    idx = rng.permutation(1000)
    shuffled = [Split(train=idx[:800], test=idx[800:], fold=0)]
    with pytest.raises(AssertionError):
        assert_no_leakage(shuffled)


def test_expanding_window_grows():
    splits = expanding_window_splits(5000, min_train=1000, test_size=250)
    sizes = [len(sp.train) for sp in splits]
    assert sizes == sorted(sizes)
    assert all(sp.train.min() == 0 for sp in splits)


def test_purged_kfold_covers_everything_once():
    splits = purged_kfold_splits(1000, n_splits=5)
    covered = np.concatenate([sp.test for sp in splits])
    assert sorted(covered) == list(range(1000))


def test_purged_kfold_removes_adjacent_training_data():
    splits = purged_kfold_splits(1000, n_splits=5, purge=20, embargo=20)
    for sp in splits:
        assert not np.intersect1d(sp.train, sp.test).size
        near = np.arange(max(0, sp.test.min() - 20), sp.test.min())
        assert not np.intersect1d(sp.train, near).size


def test_not_enough_data_is_an_error():
    with pytest.raises(ValueError, match="not enough data"):
        walk_forward_splits(100, train_size=90, test_size=50)


def test_describe_splits_shape(gbm_data):
    splits = walk_forward_splits(len(gbm_data), train_size=500, test_size=100)
    table = describe_splits(splits, gbm_data.index)
    assert len(table) == len(splits)
    assert "test_start" in table.columns


# -- labels ----------------------------------------------------------------
def test_forward_label_has_a_nan_tail(gbm_data):
    lab = fixed_horizon_return(gbm_data["close"], horizon=24)
    assert lab.y.tail(24).isna().all()
    assert not lab.y.head(-24).isna().any()
    assert lab.horizon == 24


def test_classification_label_respects_threshold(gbm_data):
    lab = fixed_horizon_class(gbm_data["close"], horizon=24, threshold=0.05)
    assert set(lab.y.dropna().unique()) <= {-1.0, 0.0, 1.0}
    # A large threshold must produce mostly neutral labels.
    assert (lab.y == 0).sum() > (lab.y != 0).sum()


def test_triple_barrier_labels_are_valid(gbm_data):
    lab = triple_barrier(gbm_data["close"], horizon=48, upper=0.02, lower=0.02,
                         high=gbm_data["high"], low=gbm_data["low"])
    assert set(lab.y.dropna().unique()) <= {-1.0, 0.0, 1.0}
    offsets = lab.meta["touch_bar"].to_numpy() - np.arange(len(gbm_data))
    valid = offsets[~np.isnan(offsets)]
    assert valid.min() >= 1, "a barrier cannot be touched on the entry bar itself"
    assert valid.max() <= 48, "a touch cannot happen after the time barrier"


def test_label_alignment_keeps_rows_matched(gbm_data):
    X = make_features(gbm_data)
    lab = fixed_horizon_class(gbm_data["close"], horizon=24)
    Xa, ya = lab.aligned_with(X)
    assert len(Xa) == len(ya)
    assert Xa.index.equals(ya.index)
    assert not Xa.isna().any().any()
    assert not ya.isna().any()


# -- features --------------------------------------------------------------
def test_features_are_finite_after_warmup(gbm_data):
    X = make_features(gbm_data).dropna()
    assert np.isfinite(X.to_numpy()).all()
    assert len(X) > len(gbm_data) * 0.8


def test_stationarity_check_flags_a_drifting_column(gbm_data):
    X = make_features(gbm_data)
    X = X.assign(raw_price=gbm_data["close"])       # deliberately non-stationary
    report = check_stationarity(X)
    assert report.loc["raw_price", "mean_drift"] > report["mean_drift"].median()
