"""The report card: one format, every strategy, with the caveats attached.

The point of a fixed report card is that you cannot quietly stop reporting the
metric that got worse. Every strategy in this workshop -- the naive one, the
machine-learned one, your capstone -- is printed the same way, including a
WARNINGS section that the harness fills in automatically.

The warnings are the most valuable part. A backtest with a Sharpe of 3 and six
warnings is worth less than one with a Sharpe of 0.4 and none.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import metrics as M
from .backtest import BacktestResult

__all__ = ["ReportCard", "report_card", "compare"]

#: Thresholds that trigger a warning. Tuned to be annoying rather than polite.
SUSPICIOUS_SHARPE = 2.5
MIN_TRADES_FOR_INFERENCE = 30
MIN_YEARS_FOR_CAGR = 1.0
HIGH_TURNOVER = 200.0


@dataclass
class ReportCard:
    """Metrics plus warnings plus benchmark comparison for one backtest."""

    strategy_name: str
    metrics: dict
    warnings: list[str] = field(default_factory=list)
    benchmark: dict | None = None
    benchmark_name: str = "buy & hold"
    cost_summary: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Metrics as a one-column DataFrame (or two, with a benchmark)."""
        data = {self.strategy_name: self.metrics}
        if self.benchmark:
            data[self.benchmark_name] = self.benchmark
        return pd.DataFrame(data)

    def to_markdown(self) -> str:
        return self.summary()

    def summary(self, width: int = 78) -> str:
        m = self.metrics
        lines = ["=" * width, f"REPORT CARD: {self.strategy_name}", "=" * width]

        def row(label, key, fmt="{:+.2%}", bench=True):
            val = m.get(key, float("nan"))
            cell = fmt.format(val) if np.isfinite(val) else "n/a"
            line = f"  {label:<26} {cell:>14}"
            if bench and self.benchmark:
                bval = self.benchmark.get(key, float("nan"))
                bcell = fmt.format(bval) if np.isfinite(bval) else "n/a"
                line += f" {bcell:>14}"
            return line

        header = f"  {'':<26} {self.strategy_name[:14]:>14}"
        if self.benchmark:
            header += f" {self.benchmark_name[:14]:>14}"
        lines.append(header)
        lines.append("  " + "-" * (width - 4))

        lines.append("  RETURN")
        lines.append(row("total return", "total_return"))
        lines.append(row("CAGR", "cagr"))
        lines.append(row("annualised volatility", "ann_volatility", "{:.2%}"))
        lines.append("  RISK-ADJUSTED")
        lines.append(row("Sharpe ratio", "sharpe", "{:+.2f}"))
        sr, se = m.get("sharpe", np.nan), m.get("sharpe_stderr", np.nan)
        if np.isfinite(sr) and np.isfinite(se):
            lines.append(f"  {'  95% CI':<26} {f'[{sr-1.96*se:+.2f}, {sr+1.96*se:+.2f}]':>14}")
        lines.append(row("P(Sharpe > 0)", "psr_vs_zero", "{:.1%}"))
        lines.append(row("Sortino ratio", "sortino", "{:+.2f}"))
        lines.append(row("Calmar ratio", "calmar", "{:+.2f}"))
        lines.append("  DRAWDOWN")
        lines.append(row("max drawdown", "max_drawdown"))
        lines.append(row("longest underwater (bars)", "dd_longest_bars", "{:,.0f}"))
        lines.append(row("CVaR 5%", "cvar_5pct", "{:+.3%}"))
        lines.append("  ACTIVITY")
        lines.append(row("bars", "n_bars", "{:,.0f}"))
        lines.append(row("years of data", "years", "{:.2f}"))
        lines.append(row("trades", "n_trades", "{:,.0f}"))
        lines.append(row("annual turnover", "turnover", "{:,.1f}x"))
        lines.append(row("average exposure", "avg_exposure", "{:.2f}"))

        if self.cost_summary:
            lines.append("  COSTS")
            cs = self.cost_summary
            lines.append(f"  {'model':<26} {cs.get('model', 'n/a'):>14}")
            lines.append(f"  {'round-trip cost':<26} {cs.get('round_trip_bps', 0):>13.1f}bp")
            lines.append(f"  {'total paid':<26} {cs.get('total_costs', 0):>14,.2f}")
            lines.append(f"  {'costs as % of profit':<26} {cs.get('costs_vs_gross', 'n/a'):>14}")

        lines.append("=" * width)
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}) -- read these before believing anything above")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        else:
            lines.append("No automatic warnings triggered.")
            lines.append("    This does NOT mean the strategy works. It means this harness")
            lines.append("    found nothing obviously wrong. The burden of proof is still yours.")
        lines.append("=" * width)
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.summary()

    def _repr_html_(self):  # pragma: no cover - notebook display
        return f"<pre style='font-size:12px;line-height:1.35'>{self.summary()}</pre>"


def report_card(
    result: BacktestResult,
    benchmark: BacktestResult | None = None,
    *,
    n_trials: int = 1,
    is_out_of_sample: bool = False,
    extra_warnings: list[str] | None = None,
) -> ReportCard:
    """Score a backtest and collect every reason to distrust it.

    Parameters
    ----------
    n_trials:
        How many strategy variants you tried before settling on this one.
        Be honest. Every parameter you tuned by looking at the equity curve
        counts. If you tried 40 moving-average pairs, this is 40, not 1.
    is_out_of_sample:
        Set True only if this result comes from data the strategy was never
        fitted, tuned or eyeballed on.
    """
    cfg = result.config
    m = M.compute_all(
        result.equity, result.weights,
        bars_per_year=cfg.bars_per_year, n_trades=result.n_trades,
    )

    gross = result.equity.iloc[-1] - result.equity.iloc[0] + result.total_costs
    m["deflated_sharpe"] = M.deflated_sharpe_ratio(
        result.returns, cfg.bars_per_year, n_trials
    ) if n_trials > 1 else float("nan")

    cost_summary = {
        "model": cfg.cost_model.name[:14],
        "round_trip_bps": cfg.cost_model.round_trip_bps(),
        "total_costs": result.total_costs,
        "costs_vs_gross": f"{result.total_costs / gross:.1%}" if gross > 0 else "n/a (no gross profit)",
    }

    warnings: list[str] = list(result.warnings)

    # -- methodology warnings (the ones that matter) ----------------------
    if cfg.is_frictionless:
        warnings.append(
            "COSTS ARE ZERO. This is not a backtest, it is a simulation of a market "
            "that does not exist. Every number above is an upper bound you will never reach."
        )
    if cfg.is_optimistic:
        warnings.append(
            "EXECUTION OPTIMISM: fill_timing='this_close' fills you at the very price "
            "your signal was computed from, which is not something you can do. Note this "
            "does not necessarily INFLATE the result -- measured, it is roughly a coin "
            "flip -- it makes it unverifiable. Re-run with fill_timing='next_open'."
        )
    if not is_out_of_sample:
        warnings.append(
            "IN-SAMPLE RESULT. The strategy was chosen or tuned using this data, so this "
            "is a measure of fit, not of skill. Only out-of-sample results are evidence."
        )
    if n_trials > 1:
        dsr = m["deflated_sharpe"]
        warnings.append(
            f"MULTIPLE TESTING: you report {n_trials} trials. Deflated Sharpe "
            f"(P that the true Sharpe beats the best-of-{n_trials}-by-luck benchmark) = "
            f"{dsr:.1%}." + ("  That is not significant." if np.isfinite(dsr) and dsr < 0.95 else "")
        )

    # -- statistical-power warnings ---------------------------------------
    if m["n_trades"] < MIN_TRADES_FOR_INFERENCE:
        warnings.append(
            f"ONLY {m['n_trades']} TRADES. With this few, the result is dominated by a "
            f"handful of outcomes. You cannot distinguish skill from luck here."
        )
    if m["years"] < MIN_YEARS_FOR_CAGR:
        warnings.append(
            f"SHORT SAMPLE ({m['years']:.2f} years). The CAGR shown is an extrapolation "
            f"from less than a year and should not be quoted."
        )
    sr, se = m["sharpe"], m["sharpe_stderr"]
    if np.isfinite(sr) and np.isfinite(se) and sr - 1.96 * se <= 0 < sr:
        warnings.append(
            f"SHARPE NOT DISTINGUISHABLE FROM ZERO: {sr:+.2f} +/- {1.96*se:.2f} (95% CI includes 0)."
        )
    if np.isfinite(sr) and sr > SUSPICIOUS_SHARPE:
        warnings.append(
            f"SHARPE OF {sr:.2f} IS IMPLAUSIBLE for a retail strategy. Before celebrating, "
            f"look for look-ahead bias, a survivorship-filtered universe, or costs you forgot. "
            f"Professional funds run at 1-2."
        )

    # -- economics warnings ------------------------------------------------
    to = m.get("turnover", np.nan)
    if np.isfinite(to) and to > HIGH_TURNOVER:
        drag = to * cfg.cost_model.round_trip_bps() / 2 / 10_000
        warnings.append(
            f"HIGH TURNOVER ({to:,.0f}x/year) implies roughly {drag:.1%} of annual cost drag "
            f"at {cfg.cost_model.round_trip_bps():.0f}bps round trip. Your edge must exceed that before anything is left."
        )
    if result.total_costs > 0 and gross > 0 and result.total_costs / gross > 0.5:
        warnings.append(
            f"COSTS CONSUMED {result.total_costs/gross:.0%} OF GROSS PROFIT. The strategy is "
            f"mostly a fee-generation mechanism for the exchange."
        )
    if np.isfinite(m["max_drawdown"]) and m["max_drawdown"] < -0.5:
        warnings.append(
            f"MAX DRAWDOWN {m['max_drawdown']:.0%}. Recovering needs "
            f"{1/(1+m['max_drawdown'])-1:+.0%}. Would you actually have held on?"
        )
    if m.get("max_exposure", 0) > 1.01:
        warnings.append(
            f"LEVERAGE USED (max exposure {m['max_exposure']:.2f}x). Leverage turns a "
            f"survivable drawdown into a margin call."
        )

    bench_metrics = None
    if benchmark is not None:
        bench_metrics = M.compute_all(
            benchmark.equity, benchmark.weights,
            bars_per_year=benchmark.config.bars_per_year, n_trades=benchmark.n_trades,
        )
        if np.isfinite(sr) and np.isfinite(bench_metrics["sharpe"]) and sr < bench_metrics["sharpe"]:
            warnings.append(
                f"UNDERPERFORMS BUY & HOLD on a risk-adjusted basis "
                f"({sr:+.2f} vs {bench_metrics['sharpe']:+.2f} Sharpe). All this machinery "
                f"is a more expensive way of doing nothing."
            )

    if extra_warnings:
        warnings.extend(extra_warnings)

    return ReportCard(
        strategy_name=result.strategy_name,
        metrics=m,
        warnings=warnings,
        benchmark=bench_metrics,
        benchmark_name=benchmark.strategy_name if benchmark else "buy & hold",
        cost_summary=cost_summary,
    )


def compare(results: dict[str, BacktestResult], *, bars_per_year: float | None = None) -> pd.DataFrame:
    """Side-by-side metrics table for several backtests.

    Sorted by Sharpe, because that is how everyone looks at it, and because
    seeing the random strategies sprinkled through the ranking is instructive.
    """
    rows = {}
    for name, res in results.items():
        bpy = bars_per_year or res.config.bars_per_year
        m = M.compute_all(res.equity, res.weights, bars_per_year=bpy, n_trades=res.n_trades)
        rows[name] = {
            "total_return": m["total_return"], "cagr": m["cagr"],
            "ann_vol": m["ann_volatility"], "sharpe": m["sharpe"],
            "sharpe_se": m["sharpe_stderr"], "max_dd": m["max_drawdown"],
            "calmar": m["calmar"], "turnover": m.get("turnover", np.nan),
            "trades": m["n_trades"], "costs": res.total_costs,
        }
    return pd.DataFrame(rows).T.sort_values("sharpe", ascending=False)
