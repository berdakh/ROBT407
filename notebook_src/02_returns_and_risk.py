# %% [markdown]
# # 02 · Returns, risk, and why the average is the wrong number
#
# **Day 1 · ~90 minutes**
#
# Prices are not the object of study. **Returns** are. Prices trend, drift and
# live on different scales; returns are roughly comparable across assets and
# across time, which is what lets you do statistics to them.
#
# By the end of this notebook you will be able to:
#
# 1. Choose correctly between simple and log returns, and say why it matters.
# 2. Annualise a return and a volatility, and explain what that assumes.
# 3. Recognise that market returns are **not** normally distributed, and measure
#    how far off they are.
# 4. Compute a Sharpe ratio *with its standard error*, and see that most of them
#    are indistinguishable from zero.

# %% [markdown]
# ## 1. Two kinds of return, and when each is correct
#
# **Simple (arithmetic) return:** $r_t = \frac{P_t}{P_{t-1}} - 1$
#
# **Log return:** $\ell_t = \ln\!\left(\frac{P_t}{P_{t-1}}\right)$
#
# They are nearly equal for small moves, and importantly different for large ones.
#
# | use | which | why |
# |---|---|---|
# | Combining assets in a portfolio | **simple** | portfolio return = weighted mean of simple returns. Not true of log returns. |
# | Adding across time | **log** | an $n$-bar log return is the sum of $n$ one-bar log returns. |
# | Statistical tests, modelling | **log** | closer to symmetric, better behaved. |
# | Reporting a number to a human | **simple** | "I made 8%" means simple. |
#
# The single most common error: summing log returns and reporting the total as a
# percentage. That **overstates losses and understates gains**.

# %%
from algotrade.data import BARS_PER_YEAR, log_returns, simple_returns
from algotrade import metrics as M

df = algotrade.load_ohlcv("data/synthetic/gbm.csv")
close = df["close"]

simple = simple_returns(close)
logret = log_returns(close)

comparison = pd.DataFrame({"simple": simple, "log": logret, "difference": simple - logret}).dropna()
print("For typical hourly moves the two barely differ:")
print(comparison.head().to_string())

print("\nBut the gap grows with the size of the move:")
for move in [0.01, 0.05, 0.10, 0.25, 0.50, -0.25, -0.50]:
    print(f"  simple {move:+6.0%}  ->  log {np.log1p(move):+7.2%}   (difference {np.log1p(move) - move:+.2%})")

# %% [markdown]
# Notice the asymmetry at the bottom: a −50% simple return is a −69% log return.
# This is the same asymmetry as the drawdown arithmetic we will meet on Day 4 —
# **losing half your money requires doubling to recover.**

# %%
# Log returns add across time. Simple returns do not. Verify it:
n = 24
by_summing_logs = logret.rolling(n).sum()
directly = log_returns(close, n)
print(f"log returns add: max discrepancy = {(by_summing_logs - directly).abs().max():.2e}")

by_summing_simple = simple.rolling(n).sum()
directly_simple = simple_returns(close, n)
print(f"simple returns do NOT add: max discrepancy = {(by_summing_simple - directly_simple).abs().max():.4f}")

# %% [markdown]
# ## 2. Annualising: a convenience with an assumption inside it
#
# To compare an hourly strategy with a daily one, we scale both to a yearly
# figure.
#
# - **Return** compounds: $(1 + \bar{r})^{N} - 1$ where $N$ is bars per year.
# - **Volatility** scales with the square root of time: $\sigma_{\text{annual}} = \sigma_{\text{bar}}\sqrt{N}$.
#
# The $\sqrt{N}$ rule assumes returns are **independent across bars**. They are
# not: volatility clusters — calm periods follow calm periods and violent ones
# follow violent ones. So annualised volatility systematically *understates* the
# risk of a bad month. Use it anyway; everyone does; just know that it is wrong
# in a specific direction.
#
# Crypto trades 24/7, so a year is genuinely `365 × 24 = 8,760` hourly bars. No
# 252-trading-day adjustment.

# %%
BPY = BARS_PER_YEAR["1h"]
print(f"bars per year (hourly crypto): {BPY:,}")


print(f"\nannualised volatility : {M.annualized_volatility(simple, BPY):.1%}")
print(f"annualised return     : {M.annualized_return(simple, BPY):.1%}")
print(f"total return          : {M.total_return(close):.1%}")

# %% [markdown]
# ### Check it against ground truth
#
# `gbm.csv` was generated with a known annual volatility. The manifest records it.

# %%
import json
from pathlib import Path

manifest = json.loads(Path("data/synthetic/manifest.json").read_text())
truth = manifest["gbm"]["ground_truth"]
print("what the generator was told to produce:")
for k, v in truth.items():
    print(f"  {k:18} {v}")
print(f"\nwhat we measured: sigma_annual = {M.annualized_volatility(simple, BPY):.4f}")
print(f"                  autocorr(1)  = {simple.dropna().autocorr(1):+.4f}")

# %% [markdown]
# The measured volatility should land within a percent or two of the target. The
# autocorrelation should be near zero — this dataset has **no predictability in
# it at all**, by construction. Remember that; we will come back to it when a
# strategy appears to make money on this file.

# %% [markdown]
# ## 3. Returns are not normal, and the difference is where you get hurt
#
# Almost every textbook formula assumes a normal distribution. Real market
# returns have **fat tails**: extreme moves happen far more often than a normal
# distribution predicts. A "six-sigma event" should happen roughly once in a
# million years. In markets they happen every few years.

# %%
from scipy import stats

r = simple.dropna()
print(f"mean      {r.mean():+.6f}")
print(f"std       {r.std():.6f}")
print(f"skew      {stats.skew(r):+.3f}     (0 for a normal distribution)")
print(f"kurtosis  {stats.kurtosis(r, fisher=False):.3f}     (3 for a normal distribution)")

sigma = r.std()
for k in (3, 4, 5):
    observed = int((r.abs() > k * sigma).sum())
    expected = 2 * stats.norm.sf(k) * len(r)
    print(f"\n|move| > {k} sigma:  observed {observed:>5}   normal predicts {expected:>8.2f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(r, bins=120, density=True, alpha=0.7, label="observed")
x = np.linspace(r.min(), r.max(), 400)
axes[0].plot(x, stats.norm.pdf(x, r.mean(), r.std()), "r--", linewidth=1.5, label="normal fit")
axes[0].set_title("Distribution of hourly returns", loc="left", fontsize=10)
axes[0].legend()

stats.probplot(r, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot: deviation from the line = fat tails", loc="left", fontsize=10)
axes[1].get_lines()[0].set_markersize(2)

fig.tight_layout()
plt.show()

# %% [markdown]
# On the Q-Q plot, points curving away from the straight line at both ends are
# fat tails. Note that this is **synthetic Gaussian data** and it still shows
# some deviation from normality in the extremes, simply from finite sampling.
# Real crypto returns are dramatically worse — kurtosis in the tens is routine.
#
# **Why you should care:** every risk number that assumes normality — value at
# risk, position sizes derived from standard deviations, confidence intervals on
# Sharpe ratios — underestimates how bad the bad days are.

# %% [markdown]
# ## 4. The Sharpe ratio, and its error bar
#
# $$\text{Sharpe} = \frac{\text{mean excess return}}{\text{standard deviation}} \times \sqrt{N}$$
#
# It is the single most quoted number in the industry, and it is almost always
# quoted without the one thing that would let you interpret it: **how uncertain
# it is.**
#
# Roughly, the standard error of a Sharpe ratio estimated over $n$ bars is
# $\sqrt{(1 + \text{SR}^2/2)/n}$ in per-bar units. Over one year of hourly data
# that error bar is large.

# %%
sr = M.sharpe_ratio(simple, BPY)
se = M.sharpe_standard_error(simple, BPY)
print(f"Sharpe ratio : {sr:+.3f}")
print(f"standard err : {se:.3f}")
print(f"95% interval : [{sr - 1.96 * se:+.3f}, {sr + 1.96 * se:+.3f}]")
print(f"P(true Sharpe > 0) = {M.probabilistic_sharpe_ratio(simple, BPY):.1%}")

# %% [markdown]
# ### How much data do you need to detect a real edge?
#
# Suppose a strategy genuinely has a Sharpe ratio of 1.0 — which would be
# excellent. How long must you observe it before you can distinguish it from zero
# at 95% confidence?

# %%
rows = []
for years in [0.25, 0.5, 1, 2, 3, 5, 10]:
    n_bars = years * BPY
    for true_sr in (0.5, 1.0, 2.0):
        se_bar = np.sqrt((1 + true_sr**2 / (2 * BPY)) / n_bars) * np.sqrt(BPY)
        rows.append(
            {"years": years, "true Sharpe": true_sr, "95% half-width": 1.96 * se_bar,
             "detectable?": "yes" if true_sr - 1.96 * se_bar > 0 else "NO"}
        )
power = pd.DataFrame(rows).pivot(index="years", columns="true Sharpe", values="detectable?")
print("Can you distinguish this Sharpe ratio from zero?\n")
print(power.to_string())

# %% [markdown]
# Read that table carefully, because it governs everything that follows. **A
# strategy with a true Sharpe of 0.5 needs years of data before you can tell it
# apart from noise.** And a six-month backtest showing a Sharpe of 2 tells you
# essentially nothing, because the error bar on it is about ±1.4.
#
# This is not pedantry. It is the reason most published backtests do not survive
# contact with live trading.

# %% [markdown]
# ## 5. Drawdown: the number that actually decides whether you survive
#
# The Sharpe ratio treats an upside surprise and a downside surprise as equally
# "risky". Your bank balance does not. **Max drawdown** — the worst peak-to-trough
# decline — is what determines whether you are still trading next year.

# %%
equity = (1 + simple.fillna(0)).cumprod()
dd = M.drawdown_series(equity)

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True, height_ratios=[2, 1])
axes[0].plot(equity.index, equity, linewidth=0.9)
axes[0].plot(equity.index, equity.cummax(), "--", linewidth=0.8, alpha=0.6, label="running peak")
axes[0].set_ylabel("growth of 1")
axes[0].legend()
axes[1].fill_between(dd.index, dd, 0, alpha=0.4, color="crimson")
axes[1].set_ylabel("drawdown")
axes[1].set_title(f"max drawdown {M.max_drawdown(equity):.1%}", loc="left", fontsize=10)
fig.suptitle("Buy and hold on a dataset with no predictability", y=0.99)
fig.tight_layout()
plt.show()

print(f"max drawdown        : {M.max_drawdown(equity):.1%}")
print(f"longest underwater  : {M.drawdown_duration(equity)['longest_bars']:,} bars "
      f"({M.drawdown_duration(equity)['longest_bars'] / 24:.0f} days)")
print(f"gain needed to recover: {1 / (1 + M.max_drawdown(equity)) - 1:+.1%}")

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 2.1 — Where the two return types diverge
#
# Find the largest single-bar move in the dataset. Compute its simple return and
# its log return. Then compute a whole year's return two ways:
#
# 1. `(1 + simple).prod() - 1`
# 2. `np.expm1(log_returns.sum())`
#
# They should agree to floating-point precision. Then compute `log_returns.sum()`
# on its own and report it as a percentage — the wrong way. How big is the error?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 2.2 — Volatility clustering
#
# The $\sqrt{N}$ annualisation assumes independence. Test whether that holds:
#
# 1. Compute `abs(returns)` and its autocorrelation at lags 1 to 48.
# 2. Compare with the autocorrelation of `returns` themselves at the same lags.
#
# You should find that *returns* are nearly uncorrelated but *absolute returns*
# are not. That is volatility clustering. Explain in two sentences why this means
# annualised volatility understates the risk of a bad month.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 2.3 — Regimes hide inside averages
#
# Load `data/synthetic/regime_switching.csv`. It alternates between a calm and a
# turbulent regime (the manifest records the parameters).
#
# 1. Compute the overall annualised volatility.
# 2. Compute a rolling 500-bar annualised volatility and plot it.
# 3. Find the highest and lowest values of that rolling estimate.
#
# Then answer: **in what sense is the overall number a description of this
# dataset?** How would a risk limit set from the overall number behave during the
# turbulent regime?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 2.4 — Your own power calculation
#
# You are about to test a strategy on 6 months of hourly data. Using the
# standard-error formula, compute the smallest true Sharpe ratio you could
# reliably detect at 95% confidence over that sample.
#
# Now: your backtest comes back with a Sharpe of 1.8. Given your answer, what can
# you honestly conclude? Write two sentences you would be willing to defend.

# %%
# Your code here.

# %% [markdown]
# ---
# ## What you should take away
#
# - Log returns add across time; simple returns combine across assets. Using the
#   wrong one misreports your results.
# - Annualising assumes independence, which is false. It understates tail risk.
# - Market returns have fat tails; normal-distribution formulas underestimate
#   bad days.
# - **A Sharpe ratio without an error bar is not a result.** Over realistic
#   sample sizes, most backtested Sharpe ratios cannot be distinguished from zero.
#
# Next: **03 · Your first strategy** — where we build something that looks
# extremely profitable.
