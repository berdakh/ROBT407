# %% [markdown]
# # 06 · The bias catalogue — every one of them measured
#
# **Day 2 · ~2 hours**
#
# You have now met two biases directly: look-ahead and ignored costs. This
# notebook is the complete catalogue. For each one we do not merely describe it —
# we **construct** it, **measure** it, and show what detecting it looks like.
#
# | bias | one-line definition | measured here |
# |---|---|---|
# | Look-ahead | using information that did not exist yet | §1 |
# | Data snooping | trying many things, reporting the best | §2 |
# | Overfitting | fitting noise instead of signal | §3 |
# | Survivorship | studying only the things that survived | §4 |
# | Ignored costs | assuming trading is free | notebook 05 |
# | Regime dependence | one number averaging two different worlds | §5 |
# | Selection of the sample | choosing the period that works | §6 |

# %%
import itertools
import warnings

from algotrade import BacktestConfig, RETAIL_CRYPTO, ZERO_COST, run_backtest
from algotrade import metrics as M
from algotrade.indicators import sma, zscore
from algotrade.naive import compare_shift_effect, vectorized_backtest
from algotrade.strategy import BuyAndHold, RandomStrategy, SMACrossover

BPY = algotrade.BARS_PER_YEAR["1h"]
noise = algotrade.load_ohlcv("data/synthetic/pure_noise.csv")
trending = algotrade.load_ohlcv("data/synthetic/trending_with_crash.csv")
gbm = algotrade.load_ohlcv("data/synthetic/gbm.csv")

# %% [markdown]
# ## 1. Look-ahead bias — the only one that always wins
#
# **Definition:** your strategy uses information that was not available at the
# moment it decided.
#
# It hides in more places than the obvious one:
#
# - Forgetting to shift a signal (the classic).
# - Normalising with full-sample statistics: `(x - x.mean()) / x.std()` uses the
#   mean of the *entire* series, including the future.
# - `df.fillna(method="bfill")` — backfill literally copies the future backwards.
# - Interpolating a gap, which uses the value on the far side.
# - Choosing a stop-loss level after looking at the chart.
#
# **The diagnostic:** a real edge does not work on data with no signal in it.
# Look-ahead does. Run your method on `pure_noise.csv` — if it still makes
# money, you have a leak, not an edge.

# %%
signal = (sma(trending["close"], 3) > sma(trending["close"], 24)).astype(float)
print("The same signal, evaluated at different delays:\n")
print(compare_shift_effect(trending["close"], signal, shifts=(0, 1, 2, 5)).to_string(
    float_format=lambda v: f"{v:,.4f}"))

# %%
# The full-sample normalisation trap, which looks completely innocent.
close = gbm["close"]

leaky_z = (np.log(close) - np.log(close).mean()) / np.log(close).std()   # uses the WHOLE series
causal_z = zscore(np.log(close), 168)                                    # trailing window only

for label, z in [("full-sample z-score (LEAKY)", leaky_z), ("trailing z-score (causal)", causal_z)]:
    sig = (-np.sign(z)).clip(-1, 1).fillna(0)
    bt = vectorized_backtest(close, sig, shift=1, cost_bps=0.0, quiet=True)
    print(f"  {label:32} total return {bt['equity'].iloc[-1] - 1:>9,.1%}")

print("\nBoth are shifted by one bar. Only one of them knows the future,")
print("and it is the one that computed a mean over data it had not seen yet.")
print("Remember: gbm.csv has NO predictability in it by construction.")

# %% [markdown]
# ## 2. Data snooping — the bias you commit by working hard
#
# **Definition:** you try many strategies, report the best one, and do not adjust
# for how many you tried.
#
# This is the most insidious bias because it feels like diligence. Testing 200
# variants *is* thorough. The problem is purely in the reporting: the best of 200
# random strategies has an impressive Sharpe ratio **by construction**.
#
# Let us put a number on it. We run 200 strategies that trade by **flipping a
# coin**, on data that is a **driftless random walk**. Both sides of that
# sentence guarantee zero edge.

# %%
coin_flips = {}
for seed in range(200):
    res = run_backtest(noise, RandomStrategy(seed=seed, every=24),
                       BacktestConfig(cost_model=ZERO_COST, bars_per_year=BPY))
    coin_flips[seed] = M.sharpe_ratio(res.returns, BPY)

sharpes = pd.Series(coin_flips)
print(f"200 coin-flipping strategies on a provably unpredictable dataset:\n")
print(f"  mean Sharpe        {sharpes.mean():+.3f}   (should be ~0, and is)")
print(f"  worst              {sharpes.min():+.3f}")
print(f"  BEST               {sharpes.max():+.3f}   <- the one you would publish")
print(f"  95th percentile    {sharpes.quantile(0.95):+.3f}")
print(f"  how many beat 1.0  {int((sharpes > 1.0).sum())} of 200")

# %%
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.hist(sharpes, bins=40, alpha=0.75, color="#0969da")
ax.axvline(0, color="#57606a", linestyle=":", linewidth=1.2, label="the truth: zero edge")
ax.axvline(sharpes.max(), color="#cf222e", linewidth=2,
           label=f"best of 200: Sharpe {sharpes.max():+.2f}")
ax.set_xlabel("annualised Sharpe ratio")
ax.set_ylabel("count")
ax.legend(fontsize=9)
ax.set_title("200 random strategies on random data. None of them has an edge.",
             loc="left", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### The correction: the deflated Sharpe ratio
#
# If you tried $N$ strategies, the right question is not "is this Sharpe greater
# than zero?" but **"is it greater than the best I would expect from $N$ tries at
# nothing?"**
#
# `deflated_sharpe_ratio` answers that. It returns the probability that the true
# Sharpe exceeds the best-of-$N$-by-luck benchmark. Below ~0.95 means you have no
# evidence.

# %%
best_seed = int(sharpes.idxmax())
best_run = run_backtest(noise, RandomStrategy(seed=best_seed, every=24),
                        BacktestConfig(cost_model=ZERO_COST, bars_per_year=BPY))

print(f"The luckiest coin-flipper (seed {best_seed}):\n")
print(f"  Sharpe ratio                 {M.sharpe_ratio(best_run.returns, BPY):+.2f}")
print(f"  P(Sharpe > 0), 1 trial       {M.probabilistic_sharpe_ratio(best_run.returns, BPY):.1%}")
for n_trials in (1, 10, 200, 1000):
    dsr = M.deflated_sharpe_ratio(best_run.returns, BPY, n_trials=n_trials)
    verdict = "significant" if dsr > 0.95 else "NOT significant"
    print(f"  deflated, {n_trials:>4} trials      {dsr:>6.1%}   {verdict}")

# %% [markdown]
# **The honest accounting is harder than it looks.** `n_trials` is not just the
# size of your final grid search. It includes:
#
# - every parameter you tried and abandoned,
# - every indicator you tested and dropped,
# - every time you changed the date range because the result looked bad,
# - and, if you read a paper that tested 50 variants and you are using its
#   winner, **their** 50 too.
#
# Most people's honest `n_trials` is in the hundreds. Almost nobody reports it.

# %% [markdown]
# ## 3. Overfitting — when in-sample and out-of-sample disagree
#
# **Definition:** your model fits the noise in the training data, which by
# definition does not repeat.
#
# **The measurement:** split chronologically. Choose parameters using only the
# first part. Evaluate on the second. The gap between the two is the overfit.

# %%
train = trending.iloc[: int(len(trending) * 0.6)]
test = trending.iloc[int(len(trending) * 0.6) :]
print(f"train: {train.index[0].date()} to {train.index[-1].date()} ({len(train):,} bars)")
print(f"test : {test.index[0].date()} to {test.index[-1].date()} ({len(test):,} bars)")

combos = [(f, s) for f, s in itertools.product([3, 6, 12, 24, 48], [24, 48, 96, 192, 384]) if f < s]
cfg = BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY)

rows = []
for fast, slow in combos:
    in_sample = run_backtest(train, SMACrossover(fast, slow), cfg)
    out_sample = run_backtest(test, SMACrossover(fast, slow), cfg)
    rows.append({
        "params": f"{fast}/{slow}",
        "in_sample_sharpe": M.sharpe_ratio(in_sample.returns, BPY),
        "out_sample_sharpe": M.sharpe_ratio(out_sample.returns, BPY),
    })

split = pd.DataFrame(rows).sort_values("in_sample_sharpe", ascending=False)
print(f"\nTop 5 by IN-SAMPLE Sharpe, with what they actually did out of sample:\n")
print(split.head(5).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

correlation = split["in_sample_sharpe"].corr(split["out_sample_sharpe"])
chosen = split.iloc[0]
print(f"\ncorrelation between in-sample and out-of-sample Sharpe: {correlation:+.3f}")
print(f"the parameters you would have chosen  : {chosen['params']}")
print(f"  their in-sample Sharpe               : {chosen['in_sample_sharpe']:+.3f}")
print(f"  their out-of-sample Sharpe           : {chosen['out_sample_sharpe']:+.3f}")
print(f"  the best out-of-sample Sharpe available: {split['out_sample_sharpe'].max():+.3f} "
      f"({split.loc[split['out_sample_sharpe'].idxmax(), 'params']})")

# %%
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(split["in_sample_sharpe"], split["out_sample_sharpe"], s=45, alpha=0.75)
lims = [min(split.min(numeric_only=True)) - 0.3, max(split.max(numeric_only=True)) + 0.3]
ax.plot(lims, lims, "--", color="#57606a", linewidth=1, label="perfect persistence")
ax.axhline(0, color="#cf222e", linewidth=0.9, linestyle=":")
ax.axvline(0, color="#cf222e", linewidth=0.9, linestyle=":")
ax.scatter([chosen["in_sample_sharpe"]], [chosen["out_sample_sharpe"]], s=180,
           facecolors="none", edgecolors="#cf222e", linewidths=2, label="what you would pick")
ax.set_xlabel("in-sample Sharpe")
ax.set_ylabel("out-of-sample Sharpe")
ax.legend(fontsize=9)
ax.set_title(f"If in-sample predicted out-of-sample, these would line up (r={correlation:+.2f})",
             loc="left", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# A near-zero or negative correlation means **in-sample performance carries no
# information about out-of-sample performance.** Which means the entire parameter
# search accomplished nothing except choosing a number to be disappointed by.
#
# This is why Day 3 is about validation rather than about models.

# %% [markdown]
# ## 4. Survivorship bias
#
# **Definition:** you study only the assets that still exist, and quietly exclude
# the ones that died.
#
# The canonical example: backtesting "buy the S&P 500 constituents" using
# *today's* constituent list. Every company that went bankrupt or was delisted is
# missing, so your universe is pre-filtered for success.
#
# **Crypto has it worse.** Thousands of tokens have gone to zero. An exchange
# that delisted them shows no history at all. A "top 100 coins by market cap"
# backtest run on today's top 100 is studying the winners of a lottery and
# concluding that lottery tickets are a good investment.
#
# We cannot demonstrate this with single-asset synthetic data, so here is a
# simulation of the mechanism.

# %%
rng = np.random.default_rng(7)
n_assets, n_bars = 300, 2000

# 300 assets, all with ZERO expected return. Pure noise, no skill anywhere.
paths = np.exp(np.cumsum(rng.normal(-0.5 * 0.02**2, 0.02, (n_assets, n_bars)), axis=1))

# Reality: assets that fall below 10% of their starting value get delisted.
final = paths[:, -1]
survived = final > 0.10

print(f"universe: {n_assets} assets, all with zero true expected return")
print(f"survivors (never fell below -90%): {survived.sum()} ({survived.mean():.0%})")
print(f"\nmean total return, ALL assets         : {final.mean() - 1:+.1%}")
print(f"mean total return, SURVIVORS only     : {final[survived].mean() - 1:+.1%}")
print(f"\nthe survivorship premium you would 'discover': "
      f"{(final[survived].mean() - final.mean()):+.2%} of starting capital")
print("\nThere is no premium. There is only a filter applied after the fact.")

# %% [markdown]
# **How to avoid it:** use a point-in-time universe — the list of assets as it
# stood on each historical date, including the ones that later died. This data is
# expensive and often simply unavailable for crypto, which is a good reason to
# stick to single liquid assets while learning.

# %% [markdown]
# ## 5. Regime dependence
#
# **Definition:** a single performance number averages over market conditions
# that are qualitatively different, hiding the fact that the strategy works in
# one and fails in the other.

# %%
regime = algotrade.load_ohlcv("data/synthetic/regime_switching.csv")
res = run_backtest(regime, SMACrossover(12, 96), BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY))

print(f"overall Sharpe: {M.sharpe_ratio(res.returns, BPY):+.3f}\n")

# Split the sample into quarters and score each one separately.
quarters = np.array_split(np.arange(len(res.equity)), 8)
print(f"{'period':>8} {'Sharpe':>9} {'return':>10} {'realised vol':>14}")
for i, idx in enumerate(quarters):
    chunk = res.returns.iloc[idx]
    eq = res.equity.iloc[idx]
    print(f"{i + 1:>8} {M.sharpe_ratio(chunk, BPY):>+9.2f} "
          f"{eq.iloc[-1] / eq.iloc[0] - 1:>+10.1%} {M.annualized_volatility(chunk, BPY):>14.1%}")

# %% [markdown]
# If those rows disagree with each other — and they will — then the single
# overall Sharpe ratio is not a description of the strategy. It is a description
# of the particular mixture of regimes that happened to occur in your sample.
#
# **Always report the dispersion across sub-periods, not just the average.**

# %% [markdown]
# ## 6. Sample selection
#
# **Definition:** choosing the time period that makes your strategy look good.
# Usually unconscious: you try 2021–2023, it looks bad, you "focus on the recent
# regime" and use 2023–2024 instead.

# %%
windows = {
    "full sample": trending,
    "first half": trending.iloc[: len(trending) // 2],
    "second half": trending.iloc[len(trending) // 2 :],
    "first quarter": trending.iloc[: len(trending) // 4],
    "last quarter": trending.iloc[3 * len(trending) // 4 :],
}

print(f"{'window':>16} {'bars':>8} {'Sharpe':>9} {'total return':>14}")
for label, window in windows.items():
    r = run_backtest(window, SMACrossover(12, 96), BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY))
    print(f"{label:>16} {len(window):>8,} {M.sharpe_ratio(r.returns, BPY):>+9.2f} "
          f"{M.total_return(r.equity):>+14.1%}")

print("\nSame strategy, same asset, same costs. Pick whichever row you like.")

# %% [markdown]
# ## 7. The checklist
#
# Before you believe any backtest — your own most of all:
#
# 1. **Could the strategy have known everything it used?** Run it on pure noise.
#    If it profits, you have a leak.
# 2. **How many things did you try?** Honestly. Include abandoned ideas and other
#    people's searches. Compute the deflated Sharpe.
# 3. **Was it evaluated on data used to choose it?** If yes, it is a measure of
#    fit, not skill.
# 4. **What did the trades cost?** Turnover × round trip ÷ 2.
# 5. **What is in your universe, and what is missing from it?**
# 6. **Does it work in every sub-period, or did one period carry it?**
# 7. **Did you choose the date range before or after seeing results?**
# 8. **Does it beat buy-and-hold** — risk-adjusted, after costs?
#
# `report_card()` automates checks 2, 3, 4, 6 and 8 as warnings. The other three
# require you to be honest with yourself, which is the hard part.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 6.1 — Audit your own strategy
#
# Take your strategy from Exercise 3.2 and run the full checklist against it.
# Write the answer to each of the eight questions. Then compute its deflated
# Sharpe ratio using an honest `n_trials`.
#
# Be specific about `n_trials`. Count everything.

# %%
# Your audit here.

# %% [markdown]
# ### Exercise 6.2 — Build a leak deliberately
#
# Write a strategy with a *subtle* look-ahead bias — not `close[i+1]`, but
# something that would survive a code review. Suggestions:
#
# - Normalise a feature using full-sample statistics.
# - Compute a rolling max with `center=True`.
# - Use `.interpolate()` on a gapped series before backtesting.
#
# Measure how much fake profit it generates, then verify it by running on
# `pure_noise.csv`. Finally, write the one-line fix.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 6.3 — How many trials until anything looks good?
#
# On `gbm.csv` (no edge by construction), how many random strategies do you need
# to test before one of them shows a Sharpe above 1.5?
#
# Run the experiment. Then compute, analytically, the expected maximum of $N$
# draws from the null distribution and compare.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 6.4 — Rank the biases
#
# Rank all seven biases by how much damage you think each would do to a typical
# student project, and justify your top three in a sentence each.
#
# Then — separately — rank them by how *easy they are to commit without
# noticing*. The two rankings are not the same, and the gap between them is where
# you should spend your attention.

# %%
# Your rankings here, as a comment.

# %% [markdown]
# ---
# ## What you should take away
#
# - Look-ahead is the only bias that wins **every time on every dataset**. That
#   universality is how you detect it: test on noise.
# - Data snooping is committed by working hard, and requires an explicit
#   correction (deflated Sharpe) rather than more effort.
# - In-sample performance may carry **zero** information about out-of-sample
#   performance. Measure the correlation; do not assume it.
# - A single performance number hides regime dependence and sample selection.
#
# **Day 2 is complete.** You have watched a +46,986% strategy become a
# +16% one that loses to doing nothing. Everything from here is about building
# things that do not do that.
#
# Next: **07 · Features** — beginning the machine-learning half, with the same
# scepticism.
