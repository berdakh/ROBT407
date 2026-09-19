"""Time-series-correct validation. The part everyone gets wrong.

WHY ``train_test_split(shuffle=True)`` IS A CATASTROPHE HERE
``sklearn``'s default shuffles. On market data that puts Tuesday in the
training set and Monday in the test set, so your model is interpolating
between known points rather than extrapolating into an unknown future. Accuracy
goes up, profit goes to zero. This is the single most common fatal error in
student ML-for-trading projects.

THREE THINGS MUST BE TRUE OF EVERY SPLIT HERE
1. **Order.** Every training index precedes every test index.
2. **Purging.** If a label at time ``t`` is built from data up to ``t + h``
   (any forward-looking label is), then training samples within ``h`` of the
   test set overlap it and must be dropped.
3. **Embargo.** Serial correlation means samples *just after* the test set
   also leak. Drop a further embargo window.

Purging and embargo are from Lopez de Prado, *Advances in Financial Machine
Learning*, ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "Split", "walk_forward_splits", "expanding_window_splits",
    "purged_kfold_splits", "assert_no_leakage", "describe_splits",
]


@dataclass(frozen=True)
class Split:
    """One train/test fold, as positional indices."""

    train: np.ndarray
    test: np.ndarray
    fold: int = 0

    def __post_init__(self) -> None:
        if len(self.test) == 0:
            raise ValueError(f"fold {self.fold} has an empty test set")

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train), len(self.test)

    def __repr__(self) -> str:  # pragma: no cover - display helper
        tr, te = self.sizes
        tr_range = f"{self.train.min()}-{self.train.max()}" if tr else "empty"
        return f"<Split {self.fold}: train[{tr_range}] n={tr} | test[{self.test.min()}-{self.test.max()}] n={te}>"


def walk_forward_splits(
    n_samples: int,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    purge: int = 0,
    embargo: int = 0,
) -> list[Split]:
    """Rolling-window walk-forward: train on a fixed window, test on what follows.

    This is the closest offline analogue of how a strategy is actually run:
    refit periodically on recent data, trade forward, repeat. The result is a
    series of genuinely out-of-sample periods that can be stitched into one
    continuous out-of-sample equity curve.

    Parameters
    ----------
    purge:
        Samples dropped from the *end* of train, to remove any whose label
        horizon overlaps the test window. Set this to your label horizon.
    embargo:
        Samples dropped from the *start* of the next train window after a test
        window, to break serial correlation leakage.

    Returns
    -------
    A list of :class:`Split`. Call :func:`assert_no_leakage` on it before you
    trust anything downstream.
    """
    step = step or test_size
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")
    if train_size + test_size > n_samples:
        raise ValueError(
            f"train_size ({train_size}) + test_size ({test_size}) exceeds "
            f"n_samples ({n_samples}); there is not enough data for even one fold"
        )

    splits: list[Split] = []
    start = 0
    fold = 0
    while start + train_size + test_size <= n_samples:
        train_end = start + train_size
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        train_idx = np.arange(start, max(start, train_end - purge))
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) and len(test_idx):
            splits.append(Split(train_idx, test_idx, fold))
            fold += 1
        start += step + embargo
    return splits


def expanding_window_splits(
    n_samples: int,
    *,
    min_train: int,
    test_size: int,
    step: int | None = None,
    purge: int = 0,
) -> list[Split]:
    """Anchored walk-forward: training window grows, always starting at sample 0.

    Use when you believe old data stays relevant. Use a rolling window instead
    when you believe the market changes -- which, in crypto, it demonstrably
    does. Neither choice is obviously right; make it deliberately.
    """
    step = step or test_size
    splits: list[Split] = []
    fold = 0
    train_end = min_train
    while train_end + test_size <= n_samples:
        train_idx = np.arange(0, max(0, train_end - purge))
        test_idx = np.arange(train_end, train_end + test_size)
        if len(train_idx):
            splits.append(Split(train_idx, test_idx, fold))
            fold += 1
        train_end += step
    return splits


def purged_kfold_splits(
    n_samples: int, *, n_splits: int = 5, purge: int = 0, embargo: int = 0
) -> list[Split]:
    """K-fold where training data adjacent to the test fold is removed.

    Unlike walk-forward, this *does* train on data that comes after the test
    fold, so it is not a simulation of live trading. It is useful for
    estimating model variance with limited data, and it is still far better
    than plain shuffled K-fold because the purge and embargo remove the
    overlap that makes shuffled folds meaningless.

    If you use this, say so, and do not present the result as an
    out-of-sample equity curve. It isn't one.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    fold_edges = np.linspace(0, n_samples, n_splits + 1).astype(int)
    splits: list[Split] = []
    for k in range(n_splits):
        test_start, test_end = fold_edges[k], fold_edges[k + 1]
        test_idx = np.arange(test_start, test_end)
        left = np.arange(0, max(0, test_start - purge))
        right = np.arange(min(n_samples, test_end + embargo), n_samples)
        train_idx = np.concatenate([left, right])
        if len(train_idx) and len(test_idx):
            splits.append(Split(train_idx, test_idx, k))
    return splits


def assert_no_leakage(
    splits: list[Split], *, label_horizon: int = 0, require_causal: bool = True
) -> None:
    """Raise if any fold leaks. Run this on every split you generate.

    Checks:

    * train and test indices never intersect;
    * with ``require_causal``, no training index sits within ``label_horizon``
      of the test window (which would mean a training label was built from
      test-period data).

    ``require_causal=False`` permits :func:`purged_kfold_splits`, where
    training data legitimately appears after the test fold.
    """
    for sp in splits:
        overlap = np.intersect1d(sp.train, sp.test)
        if len(overlap):
            raise AssertionError(
                f"fold {sp.fold}: {len(overlap)} indices appear in BOTH train and test "
                f"(e.g. {overlap[:5].tolist()}). This is direct leakage."
            )
        if require_causal and len(sp.train):
            latest_train = int(sp.train.max())
            earliest_test = int(sp.test.min())
            if latest_train >= earliest_test:
                raise AssertionError(
                    f"fold {sp.fold}: training index {latest_train} is not before test "
                    f"index {earliest_test}. The model is trained on the future."
                )
            gap = earliest_test - latest_train - 1
            if gap < label_horizon:
                raise AssertionError(
                    f"fold {sp.fold}: only {gap} bars between the last training sample and "
                    f"the first test sample, but labels look {label_horizon} bars ahead. "
                    f"Those training labels were built from test-period prices. "
                    f"Increase purge to at least {label_horizon}."
                )


def describe_splits(splits: list[Split], index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """A table of folds, for eyeballing before you commit to a scheme."""
    rows = []
    for sp in splits:
        row = {
            "fold": sp.fold,
            "train_n": len(sp.train),
            "test_n": len(sp.test),
            "train_start_i": int(sp.train.min()) if len(sp.train) else None,
            "train_end_i": int(sp.train.max()) if len(sp.train) else None,
            "test_start_i": int(sp.test.min()),
            "test_end_i": int(sp.test.max()),
            "gap_bars": int(sp.test.min() - sp.train.max() - 1) if len(sp.train) else None,
        }
        if index is not None:
            row["test_start"] = index[sp.test.min()]
            row["test_end"] = index[sp.test.max()]
        rows.append(row)
    return pd.DataFrame(rows)
