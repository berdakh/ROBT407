"""Performance metrics, and honest notes about what each one hides.

READ THIS BEFORE TRUSTING ANY NUMBER BELOW
Every metric here is a *sample statistic*. It has a standard error, and on a
few hundred bars that standard error is large enough to swallow the metric.
A Sharpe ratio of 1.5 measured over six months is not meaningfully different
from a Sharpe ratio of 0.3. :func:`sharpe_standard_error` will tell you how
wide the error bar is; look at it before you get excited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "total_return", "cagr", "annualized_return", "annualized_volatility",
    "sharpe_ratio", "sharpe_standard_error", "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio", "sortino_ratio", "drawdown_series", "max_drawdown",
    "drawdown_duration", "calmar_ratio", "hit_rate", "turnover",
    "value_at_risk", "conditional_value_at_risk", "tail_ratio", "compute_all",
]


def _clean(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()


def total_return(equity: pd.Series) -> float:
    """Cumulative return over the whole sample."""
    if len(equity) < 2 or equity.iloc[0] == 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, bars_per_year: float) -> float:
    """Compound annual growth rate.

    Meaningless on a sample shorter than a year, and actively misleading when
    quoted from one: annualising a good three-month run produces a number that
    has never happened and never will.
    """
    n = len(equity)
    if n < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    years = (n - 1) / bars_per_year
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 else float("nan")


def annualized_return(returns: pd.Series, bars_per_year: float) -> float:
    """Geometric mean return, annualised."""
    r = _clean(returns)
    if r.empty:
        return float("nan")
    growth = float(np.expm1(np.log1p(r).mean() * bars_per_year))
    return growth


def annualized_volatility(returns: pd.Series, bars_per_year: float) -> float:
    """Standard deviation of returns, scaled by ``sqrt(bars_per_year)``.

    The square-root scaling assumes independent returns. Volatility clusters,
    so this understates the risk of a bad month.
    """
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(bars_per_year)) if len(r) > 1 else float("nan")


def sharpe_ratio(returns: pd.Series, bars_per_year: float, risk_free_rate: float = 0.0) -> float:
    """Excess return per unit of volatility, annualised.

    What it does *not* tell you: whether the losses came in one catastrophic
    week or were spread evenly; whether the distribution has a fat left tail;
    whether you would have survived the drawdown emotionally or financially.
    A strategy that sells insurance has a superb Sharpe ratio until it doesn't.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / bars_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(bars_per_year))


def sharpe_standard_error(returns: pd.Series, bars_per_year: float) -> float:
    """Approximate standard error of the annualised Sharpe ratio.

    Uses the Lo (2002) correction for non-normality. Rule of thumb: over ``n``
    bars the standard error is roughly ``sqrt((1 + SR^2/2) / n)`` in per-bar
    units. Quote your Sharpe with this attached, or do not quote it.
    """
    r = _clean(returns)
    n = len(r)
    if n < 3:
        return float("nan")
    sr_bar = sharpe_ratio(r, bars_per_year) / np.sqrt(bars_per_year)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    var = (1 + 0.5 * sr_bar**2 - skew * sr_bar + (kurt - 3) / 4 * sr_bar**2) / n
    return float(np.sqrt(max(var, 0.0)) * np.sqrt(bars_per_year))


def probabilistic_sharpe_ratio(
    returns: pd.Series, bars_per_year: float, benchmark_sharpe: float = 0.0
) -> float:
    """P(true Sharpe > ``benchmark_sharpe``), given the observed sample.

    Bailey & Lopez de Prado. A PSR below ~0.95 means you cannot reject the
    hypothesis that the strategy is no better than the benchmark. Most
    student backtests land around 0.6, which is a polite way of saying
    "no evidence".
    """
    r = _clean(returns)
    n = len(r)
    if n < 3:
        return float("nan")
    sr = sharpe_ratio(r, bars_per_year) / np.sqrt(bars_per_year)
    sr_star = benchmark_sharpe / np.sqrt(bars_per_year)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_star) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series, bars_per_year: float, n_trials: int, trial_sharpe_std: float | None = None
) -> float:
    """PSR against the Sharpe you would expect from the *best of ``n_trials``* random strategies.

    This is the antidote to data snooping. If you tried 200 parameter
    combinations, the winner's Sharpe must clear the bar that the luckiest of
    200 coin-flippers would have cleared. Very often it does not.
    """
    r = _clean(returns)
    if n_trials < 1 or len(r) < 3:
        return float("nan")
    if trial_sharpe_std is None:
        trial_sharpe_std = annualized_volatility(r, bars_per_year) * 0.0 + 0.5
    euler = 0.5772156649
    e_max = trial_sharpe_std * (
        (1 - euler) * stats.norm.ppf(1 - 1 / max(n_trials, 2))
        + euler * stats.norm.ppf(1 - 1 / (max(n_trials, 2) * np.e))
    )
    return probabilistic_sharpe_ratio(r, bars_per_year, benchmark_sharpe=e_max)


def sortino_ratio(returns: pd.Series, bars_per_year: float, target: float = 0.0) -> float:
    """Like Sharpe, but only counts downside deviation as risk."""
    r = _clean(returns)
    downside = r[r < target]
    if len(r) < 2 or len(downside) == 0:
        return float("nan")
    dd = np.sqrt((downside**2).mean())
    return float((r.mean() - target) / dd * np.sqrt(bars_per_year)) if dd > 0 else float("nan")


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional decline from the running peak, at every point in time."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline. Negative by convention.

    The number that decides whether a strategy is *survivable*. A 60% drawdown
    requires a 150% gain to recover, and most people stop trading long before
    they get there.
    """
    if len(equity) < 2:
        return float("nan")
    return float(drawdown_series(equity).min())


def drawdown_duration(equity: pd.Series) -> dict:
    """Longest underwater stretch, in bars, plus the current one."""
    dd = drawdown_series(equity)
    underwater = dd < -1e-12
    longest = current = 0
    for wet in underwater:
        current = current + 1 if wet else 0
        longest = max(longest, current)
    return {"longest_bars": int(longest), "current_bars": int(current)}


def calmar_ratio(equity: pd.Series, bars_per_year: float) -> float:
    """Annualised return divided by max drawdown."""
    mdd = max_drawdown(equity)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(cagr(equity, bars_per_year) / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of bars with a positive return.

    Beware: a high hit rate with negative expectancy is the signature of a
    strategy that wins often and small, then loses once and enormously.
    """
    r = _clean(returns)
    nonzero = r[r != 0]
    return float((nonzero > 0).mean()) if len(nonzero) else float("nan")


def turnover(weights: pd.Series, bars_per_year: float) -> float:
    """Annualised turnover: how many times a year you replace your whole book.

    Multiply by your round-trip cost in bps to get the annual drag your
    strategy must out-earn before it makes you a single unit of currency.
    """
    w = pd.Series(weights).fillna(0.0)
    if len(w) < 2:
        return float("nan")
    traded = w.diff().abs().sum()
    years = (len(w) - 1) / bars_per_year
    return float(traded / years) if years > 0 else float("nan")


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Historical VaR: the loss exceeded ``level`` of the time."""
    r = _clean(returns)
    return float(np.quantile(r, level)) if len(r) else float("nan")


def conditional_value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Mean loss *given* that you are in the worst ``level`` tail.

    More honest than VaR, which tells you a threshold and nothing about how
    much worse things get beyond it.
    """
    r = _clean(returns)
    if r.empty:
        return float("nan")
    var = np.quantile(r, level)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


def tail_ratio(returns: pd.Series) -> float:
    """95th percentile gain divided by the absolute 5th percentile loss."""
    r = _clean(returns)
    if len(r) < 20:
        return float("nan")
    left = abs(np.quantile(r, 0.05))
    return float(np.quantile(r, 0.95) / left) if left > 0 else float("nan")


def compute_all(
    equity: pd.Series,
    weights: pd.Series | None = None,
    *,
    bars_per_year: float,
    n_trades: int = 0,
    risk_free_rate: float = 0.0,
) -> dict:
    """Every metric in this module, as one dict. Used by the report card."""
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = {
        "total_return": total_return(equity),
        "cagr": cagr(equity, bars_per_year),
        "ann_return": annualized_return(returns, bars_per_year),
        "ann_volatility": annualized_volatility(returns, bars_per_year),
        "sharpe": sharpe_ratio(returns, bars_per_year, risk_free_rate),
        "sharpe_stderr": sharpe_standard_error(returns, bars_per_year),
        "psr_vs_zero": probabilistic_sharpe_ratio(returns, bars_per_year, 0.0),
        "sortino": sortino_ratio(returns, bars_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, bars_per_year),
        "hit_rate": hit_rate(returns),
        "var_5pct": value_at_risk(returns),
        "cvar_5pct": conditional_value_at_risk(returns),
        "tail_ratio": tail_ratio(returns),
        "n_bars": len(equity),
        "years": (len(equity) - 1) / bars_per_year,
        "n_trades": n_trades,
    }
    out.update({f"dd_{k}": v for k, v in drawdown_duration(equity).items()})
    if weights is not None:
        out["turnover"] = turnover(weights, bars_per_year)
        out["avg_exposure"] = float(pd.Series(weights).abs().mean())
        out["max_exposure"] = float(pd.Series(weights).abs().max())
    return out
