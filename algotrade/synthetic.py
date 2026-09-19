"""Synthetic price series with known ground truth.

Real market data has one fatal property for teaching: nobody knows the right
answer. If your momentum strategy makes money on BTC, you cannot tell whether
momentum exists or you got lucky.

So the workshop's core exercises run on synthetic data where the true
data-generating process is written down. If a strategy cannot find a signal
that we *put there on purpose*, the strategy is broken. If it finds a signal in
``pure_noise``, the evaluation is broken. Both are useful lessons.

Every generator returns ``(DataFrame, truth)`` where ``truth`` documents the
parameters used, so exercises can be graded against them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["SyntheticTruth", "make_ohlcv", "generate", "DATASETS"]


@dataclass
class SyntheticTruth:
    """What the generator actually did. The answer key."""

    kind: str
    seed: int
    n_bars: int
    bars_per_year: float
    params: dict = field(default_factory=dict)
    notes: str = ""

    def __str__(self) -> str:  # pragma: no cover - display helper
        lines = [f"Synthetic dataset '{self.kind}' (seed={self.seed}, n={self.n_bars})"]
        for k, v in self.params.items():
            lines.append(f"  {k} = {v}")
        if self.notes:
            lines.append(f"  note: {self.notes}")
        return "\n".join(lines)


def _bars_index(n: int, freq: str, start: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq=freq, tz="UTC", name="timestamp")


def make_ohlcv(
    close: np.ndarray,
    *,
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    intrabar_steps: int = 12,
    volume_base: float = 100.0,
) -> pd.DataFrame:
    """Turn a closing-price path into consistent OHLCV bars.

    We simulate a Brownian bridge *inside* each bar, so the high and low are
    real extremes of a path that starts at the previous close and ends at this
    close. Bars built by adding random noise to the close instead can produce
    ``high < close``, which then fails validation -- or worse, doesn't.

    Volume is correlated with absolute return, because in real markets it is.
    """
    n = len(close)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    open_ = prev_close * (1.0 + rng.normal(0.0, 0.0003, n))

    # Bridge from open to close with intrabar noise, then take extremes.
    steps = np.linspace(0.0, 1.0, intrabar_steps + 1)[1:-1]
    bar_scale = np.abs(np.log(close / open_)) + 1e-4
    noise = rng.normal(0.0, 1.0, (n, len(steps))) * bar_scale[:, None] * 0.6
    bridge = (
        np.log(open_)[:, None]
        + (np.log(close) - np.log(open_))[:, None] * steps[None, :]
        + noise * np.sqrt(steps * (1 - steps))[None, :]
    )
    path = np.column_stack([np.log(open_), bridge, np.log(close)])
    high = np.exp(path.max(axis=1))
    low = np.exp(path.min(axis=1))

    abs_ret = np.abs(np.diff(np.log(close), prepend=np.log(close[0])))
    volume = volume_base * np.exp(rng.normal(0, 0.4, n)) * (1.0 + 30.0 * abs_ret)

    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def generate(
    kind: Literal["gbm", "momentum", "mean_reverting", "regime_switching", "pure_noise", "trending_with_crash"] = "gbm",
    *,
    n: int = 8760,
    seed: int = 7,
    freq: str = "1h",
    start: str = "2021-01-01",
    s0: float = 20_000.0,
    **overrides,
) -> tuple[pd.DataFrame, SyntheticTruth]:
    """Generate an OHLCV dataset whose statistical properties we know.

    Kinds
    -----
    ``gbm``
        Geometric Brownian motion. Returns are i.i.d. normal: there is
        **no** exploitable signal. Any strategy that profits here, profits by
        luck. This is the control group.
    ``momentum``
        Returns follow an AR(1) with positive ``phi``: today's return is
        genuinely predictive of tomorrow's. A trend follower *should* work.
    ``mean_reverting``
        Log price is an Ornstein-Uhlenbeck process around a drifting mean.
        A z-score reversion strategy *should* work; a trend follower should
        lose.
    ``regime_switching``
        Alternates between a calm low-vol regime and a turbulent high-vol one
        via a Markov chain. Teaches why a single backtest number hides
        everything.
    ``pure_noise``
        Random walk with zero drift. Used for the data-snooping lesson: run
        200 strategies on it and admire the best one's Sharpe ratio.
    ``trending_with_crash``
        Strong uptrend punctuated by sharp drawdowns. Teaches why max drawdown
        and Sharpe disagree.

    Returns
    -------
    (df, truth) where ``truth`` records the real parameters.
    """
    rng = np.random.default_rng(seed)
    index = _bars_index(n, freq, start)
    from .data import BARS_PER_YEAR

    bpy = float(BARS_PER_YEAR.get(freq, 365 * 24))
    params: dict = {}
    notes = ""

    if kind == "gbm":
        mu = overrides.get("mu_annual", 0.10)
        sigma = overrides.get("sigma_annual", 0.60)
        dt = 1.0 / bpy
        rets = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
        params = {"mu_annual": mu, "sigma_annual": sigma, "autocorrelation": 0.0}
        notes = "i.i.d. returns -- no predictability exists. Any edge found here is overfitting."

    elif kind == "momentum":
        phi = overrides.get("phi", 0.06)
        sigma = overrides.get("sigma_annual", 0.60)
        dt = 1.0 / bpy
        sd = sigma * np.sqrt(dt) * np.sqrt(1 - phi**2)
        shocks = rng.normal(0.0, sd, n)
        rets = np.empty(n)
        rets[0] = shocks[0]
        for t in range(1, n):
            rets[t] = phi * rets[t - 1] + shocks[t]
        rets += (overrides.get("mu_annual", 0.05) - 0.5 * sigma**2) * dt
        params = {"phi": phi, "sigma_annual": sigma, "autocorrelation": phi}
        notes = f"AR(1) returns with phi={phi}. Trend following should earn a positive edge here."

    elif kind == "mean_reverting":
        kappa = overrides.get("kappa", 0.06)
        sigma = overrides.get("sigma_annual", 0.55)
        drift = overrides.get("mu_annual", 0.05) / bpy
        sd = sigma / np.sqrt(bpy)
        log_s = np.empty(n)
        log_s[0] = np.log(s0)
        level = np.log(s0)
        shocks = rng.normal(0.0, sd, n)
        for t in range(1, n):
            level += drift
            log_s[t] = log_s[t - 1] + kappa * (level - log_s[t - 1]) + shocks[t]
        close = np.exp(log_s)
        params = {"kappa": kappa, "sigma_annual": sigma, "half_life_bars": float(np.log(2) / kappa)}
        notes = (f"Ornstein-Uhlenbeck, half-life {np.log(2)/kappa:.0f} bars. A z-score reversion "
                 f"strategy should earn a real edge here; a trend follower should lose. The pull is "
                 f"far stronger than any real market -- it is dialled up so the signal is findable.")
        df = make_ohlcv(close, index=index, rng=rng)
        return df, SyntheticTruth(kind, seed, n, bpy, params, notes)

    elif kind == "regime_switching":
        sig_lo = overrides.get("sigma_calm", 0.30)
        sig_hi = overrides.get("sigma_turbulent", 1.20)
        p_stay = overrides.get("p_stay", 0.995)
        dt = 1.0 / bpy
        state = np.zeros(n, dtype=int)
        for t in range(1, n):
            state[t] = state[t - 1] if rng.random() < p_stay else 1 - state[t - 1]
        sig = np.where(state == 0, sig_lo, sig_hi)
        mu = np.where(state == 0, 0.30, -0.40)
        rets = rng.normal((mu - 0.5 * sig**2) * dt, sig * np.sqrt(dt))
        params = {"sigma_calm": sig_lo, "sigma_turbulent": sig_hi, "p_stay": p_stay,
                  "pct_turbulent": float(state.mean())}
        notes = "Two volatility regimes. A single Sharpe ratio averages them into a meaningless number."

    elif kind == "pure_noise":
        sigma = overrides.get("sigma_annual", 0.65)
        dt = 1.0 / bpy
        rets = rng.normal(-0.5 * sigma**2 * dt, sigma * np.sqrt(dt), n)
        params = {"mu_annual": 0.0, "sigma_annual": sigma, "autocorrelation": 0.0}
        notes = "Driftless random walk. Ground truth: NO strategy has an edge. Used for the data-snooping lesson."

    elif kind == "trending_with_crash":
        sigma = overrides.get("sigma_annual", 0.55)
        mu = overrides.get("mu_annual", 1.20)
        # Crash *rate* is fixed per year, so a short sample is not carpet-bombed.
        n_crashes = overrides.get("n_crashes", max(1, round(2.0 * n / bpy)))
        dt = 1.0 / bpy
        rets = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
        # Crash length scales with the series, not with the calendar, so the
        # shape of the equity curve is the same whether n is 2,000 or 20,000.
        max_len = max(5, n // 25)
        min_len = max(3, max_len // 4)
        spacing = (n - n // 8) // max(1, n_crashes)
        crash_starts = [
            int(n // 12 + k * spacing + rng.integers(0, max(1, spacing // 3)))
            for k in range(n_crashes)
        ]
        for cs in crash_starts:
            length = int(rng.integers(min_len, max_len + 1))
            depth = rng.uniform(0.20, 0.40)
            end = min(n, cs + length)
            if end <= cs:
                continue
            rets[cs:end] += np.log(1 - depth) / (end - cs)
        params = {"sigma_annual": sigma, "mu_annual": mu, "n_crashes": n_crashes,
                  "crash_starts": sorted(int(c) for c in crash_starts),
                  "max_crash_len_bars": max_len}
        notes = "Strong drift with abrupt drawdowns. Sharpe looks fine; max drawdown does not."

    else:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(DATASETS)}")

    close = s0 * np.exp(np.cumsum(rets))
    df = make_ohlcv(close, index=index, rng=rng)
    return df, SyntheticTruth(kind, seed, n, bpy, params, notes)


#: Datasets committed to ``data/synthetic/``, built by ``scripts/build_datasets.py``.
#:
#: The seeds are not arbitrary: each was chosen so the realised path actually
#: *shows* the lesson the generator is meant to teach. That is legitimate for a
#: teaching fixture and dishonest for a backtest -- which is itself the point of
#: Day 2. Picking the seed that makes your equity curve look good is exactly
#: what data snooping is.
DATASETS = {
    "gbm": dict(kind="gbm", n=8760, seed=22),
    "momentum": dict(kind="momentum", n=8760, seed=11),
    "mean_reverting": dict(kind="mean_reverting", n=8760, seed=13),
    "regime_switching": dict(kind="regime_switching", n=8760, seed=17),
    "pure_noise": dict(kind="pure_noise", n=8760, seed=19),
    "trending_with_crash": dict(kind="trending_with_crash", n=17520, seed=12),
}
