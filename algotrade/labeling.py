"""Turning prices into supervised-learning targets, without reading the future twice.

A label is *supposed* to look forward -- that is what makes it a prediction
target. The danger is forgetting by *how far*, because that horizon determines:

* how much training data must be **purged** around each test fold
  (see :mod:`algotrade.validation`), and
* how many rows at the end of your dataset have no valid label at all.

Every function here returns the label **and** the number of bars it looked
ahead, so the horizon cannot get lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Labels", "fixed_horizon_return", "fixed_horizon_class", "triple_barrier"]


@dataclass
class Labels:
    """A label series plus the forward horizon used to build it."""

    y: pd.Series
    horizon: int
    kind: str
    meta: pd.DataFrame | None = None

    def aligned_with(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Drop rows where either the features or the label are missing.

        Always align features and labels *together*. Dropping NaNs separately
        silently misaligns them by the number of warm-up rows, which produces a
        model that appears to work and is predicting the wrong bar.
        """
        common = X.index.intersection(self.y.index)
        Xa, ya = X.loc[common], self.y.loc[common]
        mask = Xa.notna().all(axis=1) & ya.notna()
        return Xa[mask], ya[mask]


def fixed_horizon_return(close: pd.Series, horizon: int = 24) -> Labels:
    """Forward log return over ``horizon`` bars. A regression target.

    The last ``horizon`` rows are NaN by construction: their future has not
    happened yet. If your label series has no NaN tail, you have a bug.
    """
    fwd = np.log(close.shift(-horizon) / close)
    return Labels(y=fwd.rename(f"fwd_ret_{horizon}"), horizon=horizon, kind="regression")


def fixed_horizon_class(
    close: pd.Series, horizon: int = 24, threshold: float = 0.0
) -> Labels:
    """Sign of the forward return, as -1 / 0 / +1. A classification target.

    ``threshold`` creates a neutral band. A non-zero threshold is usually the
    right choice: it stops the model from learning to predict noise around
    zero, and it should be set to at least your round-trip cost, because a
    move smaller than that is not tradeable even if you predict it perfectly.
    """
    fwd = np.log(close.shift(-horizon) / close)
    y = pd.Series(0.0, index=close.index, dtype=float)
    y[fwd > threshold] = 1.0
    y[fwd < -threshold] = -1.0
    y[fwd.isna()] = np.nan
    return Labels(
        y=y.rename(f"cls_{horizon}"), horizon=horizon, kind="classification",
        meta=pd.DataFrame({"fwd_return": fwd}),
    )


def triple_barrier(
    close: pd.Series,
    *,
    horizon: int = 48,
    upper: float = 0.02,
    lower: float = 0.02,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
) -> Labels:
    """Label by which barrier is hit first: profit target, stop loss, or time.

    Lopez de Prado's triple-barrier method. It answers the question a trader
    actually has -- "if I enter now, do I hit my target or my stop first?" --
    instead of "what is the return exactly 24 bars from now", which no trader
    has ever cared about.

    Returns +1 (upper hit first), -1 (lower hit first), 0 (time ran out).
    ``meta`` carries ``touch_bar``: the bar the label resolved on. That is the
    **true** horizon of each sample and the correct input to purging -- it is
    usually shorter than ``horizon``, but never longer.

    ``high``/``low`` make the barriers intrabar, which is more realistic (a
    stop is hit by a wick, not by a close). Omit them to use closes only.
    """
    c = close.to_numpy(float)
    hi = c if high is None else high.to_numpy(float)
    lo = c if low is None else low.to_numpy(float)
    n = len(c)

    y = np.full(n, np.nan)
    touch = np.full(n, np.nan)
    ret = np.full(n, np.nan)

    for i in range(n):
        end = min(i + horizon, n - 1)
        if end <= i:
            break
        entry = c[i]
        up_px = entry * (1.0 + upper)
        dn_px = entry * (1.0 - lower)
        label, hit = 0.0, end
        for j in range(i + 1, end + 1):
            up_hit = hi[j] >= up_px
            dn_hit = lo[j] <= dn_px
            if up_hit and dn_hit:
                # Both barriers inside one bar. We cannot tell the order from
                # OHLC alone, so we assume the worse outcome. Assuming the
                # better one is a small, systematic, compounding lie.
                label, hit = -1.0, j
                break
            if up_hit:
                label, hit = 1.0, j
                break
            if dn_hit:
                label, hit = -1.0, j
                break
        y[i], touch[i], ret[i] = label, hit, c[hit] / entry - 1.0

    return Labels(
        y=pd.Series(y, index=close.index, name=f"tb_{horizon}"),
        horizon=horizon,
        kind="classification",
        meta=pd.DataFrame(
            {"touch_bar": touch, "realized_return": ret}, index=close.index
        ),
    )
