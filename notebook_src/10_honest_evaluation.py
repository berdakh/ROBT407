# %% [markdown]
# # 10 · Honest evaluation — the report you would defend
#
# **Day 3 · ~90 minutes**
#
# You now have every tool needed to evaluate a strategy properly. This notebook
# assembles them into a single repeatable procedure, and — more importantly —
# into a **written conclusion you would be willing to put your name to**.
#
# That last part is the actual skill. Anyone can produce an equity curve. Writing
# three sentences about it that are true, complete and not misleading is harder
# and much rarer.
#
# By the end you will be able to:
#
# 1. Run every strategy through one identical harness.
# 2. Read a report card's warnings and act on them.
# 3. Benchmark against the things that matter, including random strategies.
# 4. Write an honest conclusion, including when the conclusion is "no".

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from algotrade import BacktestConfig, RETAIL_CRYPTO, ZERO_COST, run_backtest
from algotrade import metrics as M
from algotrade.features import make_features
from algotrade.labeling import fixed_horizon_class
from algotrade.report import compare, report_card
from algotrade.risk import VolatilityTargeted
from algotrade.strategy import (
    BuyAndHold, MomentumStrategy, RandomStrategy, SMACrossover, Strategy, ZScoreReversion,
)
from algotrade.validation import walk_forward_splits

BPY = algotrade.BARS_PER_YEAR["1h"]
df = algotrade.load_ohlcv("data/synthetic/momentum.csv")
cfg = BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY)

# %% [markdown]
# ## 1. The same harness for everything
#
# The point of a fixed report card is that **you cannot quietly stop reporting
# the metric that got worse.** Every strategy faces the same questions.

# %%
candidates = {
    "buy & hold": BuyAndHold(),
    "SMA 12/96": SMACrossover(12, 96),
    "momentum 48": MomentumStrategy(48),
    "z-reversion 48": ZScoreReversion(48, 1.5),
    "vol-targeted momentum": VolatilityTargeted(MomentumStrategy(48), target_vol=0.20),
}
# Random strategies are not filler. They are the control group, and seeing where
# they land in the ranking is the most instructive column in the table.
for seed in range(5):
    candidates[f"random #{seed}"] = RandomStrategy(seed=seed, every=48)

results = {name: run_backtest(df, strategy, cfg) for name, strategy in candidates.items()}
table = compare(results, bars_per_year=BPY)
print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

# %% [markdown]
# **Look at where the random strategies rank.** If a real strategy does not sit
# clearly above all five coin-flippers, you have not demonstrated anything — and
# with only five controls, "clearly above" needs to mean a wide margin, because
# the best of five random strategies is already a biased estimate.

# %%
fig, ax = plt.subplots(figsize=(11, 4.2))
for name, res in results.items():
    is_random = name.startswith("random")
    ax.plot(res.equity.index, res.equity / res.equity.iloc[0],
            linewidth=0.9 if is_random else 1.6,
            alpha=0.45 if is_random else 1.0,
            color="#9198a1" if is_random else None,
            label=None if is_random else name)
ax.plot([], [], color="#9198a1", linewidth=0.9, label="random strategies (5)")
ax.axhline(1.0, color="#57606a", linewidth=0.8, linestyle=":")
ax.set_ylabel("growth of 1")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Every candidate, after costs. Grey lines have no edge by construction.",
             loc="left", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Reading a report card
#
# The metrics are the easy part. The warnings are the valuable part.

# %%
card = report_card(
    results["SMA 12/96"],
    benchmark=results["buy & hold"],
    n_trials=22,                 # Day 1's grid search. Be honest.
    is_out_of_sample=False,      # chosen and evaluated on the same data
)
print(card.summary())

# %% [markdown]
# ### What the harness can and cannot check
#
# | check | automated? |
# |---|---|
# | costs were zero | yes |
# | optimistic fill timing | yes |
# | too few trades for inference | yes |
# | Sharpe indistinguishable from zero | yes |
# | Sharpe implausibly high | yes |
# | turnover implies large cost drag | yes |
# | underperforms buy & hold | yes |
# | **was it evaluated out of sample?** | **it takes your word** |
# | **how many things did you try?** | **it takes your word** |
# | look-ahead in precomputed features | no |
# | survivorship in your universe | no |
#
# The two it takes your word for are the two that matter most. There is no way
# around this: **the integrity of a backtest ultimately rests on the honesty of
# the person who ran it.** Tooling can only make dishonesty require a deliberate
# act rather than an oversight.

# %% [markdown]
# ## 3. The metrics that should lead your report
#
# Not total return. In rough order of usefulness:
#
# 1. **Out-of-sample Sharpe with its confidence interval.** A point estimate
#    alone is not a result.
# 2. **Deflated Sharpe**, given an honest trial count.
# 3. **Fold-by-fold consistency** — how many walk-forward folds were positive,
#    and the t-statistic.
# 4. **Max drawdown**, which decides whether you survive.
# 5. **Turnover and the implied cost drag.**
# 6. **Performance relative to buy-and-hold**, risk-adjusted, after costs.
#
# Total return is last, because it is dominated by the sample period and tells
# you almost nothing about whether the effect repeats.

# %%
best = results["vol-targeted momentum"]
sr = M.sharpe_ratio(best.returns, BPY)
se = M.sharpe_standard_error(best.returns, BPY)

print("The way to report a result:\n")
print(f"  Sharpe ratio        {sr:+.2f}  (95% CI [{sr - 1.96 * se:+.2f}, {sr + 1.96 * se:+.2f}])")
print(f"  P(Sharpe > 0)       {M.probabilistic_sharpe_ratio(best.returns, BPY):.1%}")
print(f"  deflated (22 trials){M.deflated_sharpe_ratio(best.returns, BPY, 22):>7.1%}")
print(f"  max drawdown        {M.max_drawdown(best.equity):+.1%}")
print(f"  turnover            {M.turnover(best.weights, BPY):,.0f}x/yr "
      f"-> {M.turnover(best.weights, BPY) * RETAIL_CRYPTO.round_trip_bps() / 2 / 10_000:.1%} annual drag")
print(f"  vs buy & hold       {sr - M.sharpe_ratio(results['buy & hold'].returns, BPY):+.2f} Sharpe")

print("\nThe way NOT to report it:")
print(f"  'This strategy returned {M.total_return(best.equity):+.1%}.'")

# %% [markdown]
# ## 4. Sub-period stability
#
# A single number over the whole sample hides everything. Split it and look.

# %%
def stability_table(result, n_chunks: int = 6) -> pd.DataFrame:
    edges = np.linspace(0, len(result.equity), n_chunks + 1).astype(int)
    rows = []
    for k in range(n_chunks):
        sl = slice(edges[k], edges[k + 1])
        chunk_returns = result.returns.iloc[sl]
        chunk_equity = result.equity.iloc[sl]
        rows.append({
            "period": k + 1,
            "from": result.equity.index[edges[k]].date(),
            "sharpe": M.sharpe_ratio(chunk_returns, BPY),
            "return": chunk_equity.iloc[-1] / chunk_equity.iloc[0] - 1,
            "max_dd": M.max_drawdown(chunk_equity),
        })
    return pd.DataFrame(rows)


stability = stability_table(results["vol-targeted momentum"])
print(stability.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
print(f"\npositive periods: {int((stability['sharpe'] > 0).sum())} of {len(stability)}")
print(f"Sharpe dispersion: {stability['sharpe'].std():.2f}")
print("\nIf the sign flips between periods, the overall number is describing")
print("your sample, not your strategy.")

# %% [markdown]
# ## 5. Writing the conclusion
#
# Here is a template. It is deliberately blunt, and every field is mandatory.
#
# > **Strategy:** [one sentence — what it does and why it might work]
# >
# > **Data:** [asset, frequency, period, source, and what is missing from it]
# >
# > **Method:** [validation scheme, number of folds, purge and embargo]
# >
# > **Trials:** [honest count, including abandoned ideas]
# >
# > **Result, out of sample, after costs:** Sharpe X (95% CI [a, b]),
# > deflated Sharpe p, max drawdown D, turnover T, positive in k of n folds.
# >
# > **Benchmark:** buy-and-hold achieved Sharpe Y over the same period.
# >
# > **Conclusion:** [one sentence, and it is allowed to be "no evidence of an edge"]
# >
# > **What would change my mind:** [the specific result that would falsify this]
#
# That last line is the one most people skip. If you cannot say what would prove
# you wrong, you are not doing analysis.

# %%
# A worked example, generated from the numbers above.
strategy_name = "vol-targeted momentum"
res = results[strategy_name]
bench = results["buy & hold"]
sr, se = M.sharpe_ratio(res.returns, BPY), M.sharpe_standard_error(res.returns, BPY)

print(f"""
Strategy : 48-bar momentum, sized to a 20% annualised volatility target.
           Rationale: this dataset is AR(1) with phi=0.06, so short-horizon
           trend continuation genuinely exists in it.

Data     : synthetic hourly series, {len(df):,} bars, {df.index[0].date()} to {df.index[-1].date()}.
           Generated, not real. No survivorship or liquidity effects, and no
           fat tails beyond what the generator produces -- so this is an
           EASIER problem than a real market.

Method   : single-pass backtest with next-open fills and a 24 bps round trip.
           NOT walk-forward validated. Parameters were not tuned here, but the
           strategy class was chosen knowing the generator.

Trials   : at least 22 (Day 1's grid) plus every strategy compared above.

Result   : Sharpe {sr:+.2f} (95% CI [{sr - 1.96 * se:+.2f}, {sr + 1.96 * se:+.2f}]),
           max drawdown {M.max_drawdown(res.equity):.1%},
           turnover {M.turnover(res.weights, BPY):,.0f}x/yr,
           deflated Sharpe (22 trials) {M.deflated_sharpe_ratio(res.returns, BPY, 22):.1%}.

Benchmark: buy-and-hold achieved Sharpe {M.sharpe_ratio(bench.returns, BPY):+.2f}
           with {M.max_drawdown(bench.equity):.1%} max drawdown and 1 trade.

Conclusion: {"the strategy does not beat buy-and-hold risk-adjusted on this sample."
             if sr <= M.sharpe_ratio(bench.returns, BPY) else
             "the strategy beats buy-and-hold on this sample, in sample, on synthetic data."}
           This is a single in-sample result on generated data. It is not
           evidence that the approach works on a real market.

What would change my mind: a walk-forward run over >= 10 folds on real hourly
           data, positive in at least 8, with a deflated Sharpe above 0.95
           after an honest trial count.
""")

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 10.1 — Report card your own strategy
#
# Take your best strategy so far. Produce its report card with an honest
# `n_trials` and the correct `is_out_of_sample` flag.
#
# For **every** warning it raises, write either a fix or a one-sentence
# justification for why it does not apply. No warning may be left unaddressed.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 10.2 — Beat the random strategies
#
# Generate **50** random strategies on this dataset and record the distribution
# of their Sharpe ratios. Where does yours fall in that distribution — what
# percentile?
#
# If it is not above the 95th percentile, what is the honest conclusion?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 10.3 — A strategy that fails, reported well
#
# Deliberately build a strategy you expect to lose — a trend follower on
# `mean_reverting.csv` is a good choice.
#
# Write its conclusion using the template. The exercise is to produce a report
# that is **useful** despite the negative result: someone reading it should learn
# something and not repeat your work.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 10.4 — Audit a stranger's claim
#
# Someone shows you this:
#
# > *"My ML crypto strategy returned 340% last year with a Sharpe ratio of 2.8.
# > I used a random forest on 30 technical indicators with 5-fold
# > cross-validation and got 61% accuracy."*
#
# List every question you would ask before believing any of it. Aim for at least
# eight, and for each one say what answer would worry you.
#
# (Three of the red flags are visible in that sentence alone.)

# %%
# Your list here.

# %% [markdown]
# ---
# ## What you should take away
#
# - One harness for every strategy means you cannot hide the metric that got worse.
# - Random strategies are a necessary control, not padding.
# - Lead with out-of-sample Sharpe **and its interval**, deflated Sharpe, fold
#   consistency, drawdown and turnover. Total return last.
# - **"No evidence of an edge" is a complete, publishable, valuable result.**
#   Being able to write it is the skill this workshop is actually teaching.
#
# **Day 3 is complete.** Next: **11 · Risk and position sizing** — the half of
# the problem we have been ignoring.
