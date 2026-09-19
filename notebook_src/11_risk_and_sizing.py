# %% [markdown]
# # 11 · Risk and position sizing — the other half of the problem
#
# **Day 4 · ~2.5 hours**
#
# Everything so far has been about the **signal**: which way to lean. This
# notebook is about **how much**, and it is where most of the realisable
# improvement actually lives.
#
# Given a mediocre signal, good sizing produces a survivable strategy. Given an
# excellent signal, bad sizing produces a margin call.
#
# By the end you will be able to:
#
# 1. Separate signal from sizing in your own code.
# 2. Implement volatility targeting and verify it hits its target.
# 3. Apply a drawdown limit, and state honestly what it costs you.
# 4. Explain why leverage is not a way to improve a strategy.
#
# ![position sizing](../diagrams/svg/05-risk-and-position-sizing.svg)

# %%
from algotrade import BacktestConfig, RETAIL_CRYPTO, ZERO_COST, run_backtest
from algotrade import metrics as M
from algotrade.risk import (
    DrawdownGuard, VolatilityTargeted, fixed_fractional_size, kelly_fraction,
    vol_target_series, volatility_target_weight,
)
from algotrade.strategy import BuyAndHold, MomentumStrategy, Strategy

BPY = algotrade.BARS_PER_YEAR["1h"]
df = algotrade.load_ohlcv("data/synthetic/trending_with_crash.csv")
cfg = BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY, max_leverage=3.0)

print(f"{len(df):,} bars. This dataset contains deliberate crashes,")
print("which is exactly the condition sizing exists to survive.")

# %% [markdown]
# ## 1. The separation
#
# A strategy returns a **direction** in $[-1, 1]$. A sizing rule multiplies it by
# a **scale**. Keeping these apart means you can change one without touching the
# other — and it means you can ask "is my signal any good?" and "is my sizing any
# good?" as separate questions, which is the only way to get useful answers.
#
# `VolatilityTargeted` and `DrawdownGuard` are wrappers: they take a strategy and
# return a strategy, so they compose.

# %%
base = MomentumStrategy(48)
plain = run_backtest(df, base, cfg)

print(f"unsized momentum 48:")
print(f"  annualised volatility {M.annualized_volatility(plain.returns, BPY):>8.1%}")
print(f"  max drawdown          {M.max_drawdown(plain.equity):>8.1%}")
print(f"  Sharpe                {M.sharpe_ratio(plain.returns, BPY):>+8.2f}")

# %% [markdown]
# ## 2. Volatility targeting
#
# $$w_t = \min\left(\frac{\sigma_{\text{target}}}{\hat{\sigma}_{t-1}},\ L_{\max}\right)$$
#
# Scale the position inversely to recent realised volatility, so the portfolio
# runs at roughly **constant risk** rather than constant notional.
#
# Two details that matter enormously:
#
# 1. **The estimate must be lagged.** $\hat{\sigma}_{t-1}$, not $\hat{\sigma}_t$.
#    Using today's volatility to size today's position is look-ahead.
# 2. **The cap is not optional.** When realised volatility collapses, the
#    uncapped formula demands enormous leverage — precisely when a quiet market
#    is about to stop being quiet.

# %%
print(f"realised vol -> target weight (target 20%, cap 3x):\n")
for realised in (0.10, 0.20, 0.40, 0.80, 1.60, 0.02):
    w = volatility_target_weight(realised, 0.20, max_leverage=3.0)
    note = "  <- CAPPED; uncapped this would be 10x" if realised == 0.02 else ""
    print(f"  realised {realised:>5.0%}  ->  weight {w:>5.2f}{note}")

# %%
targeted = run_backtest(df, VolatilityTargeted(MomentumStrategy(48), target_vol=0.20,
                                               max_leverage=3.0, bars_per_year=BPY), cfg)

print(f"{'':28} {'realised vol':>13} {'max DD':>9} {'Sharpe':>9} {'return':>10}")
for name, res in [("unsized", plain), ("vol-targeted @ 20%", targeted)]:
    print(f"  {name:26} {M.annualized_volatility(res.returns, BPY):>13.1%} "
          f"{M.max_drawdown(res.equity):>9.1%} {M.sharpe_ratio(res.returns, BPY):>+9.2f} "
          f"{M.total_return(res.equity):>+10.1%}")

print(f"\nThe targeted version's realised volatility should be close to 20%.")
print(f"That is the mechanism working: it is a RISK control, not a return booster.")

# %%
returns = df["close"].pct_change().fillna(0)
scale = vol_target_series(returns, target_vol=0.20, lookback=24 * 14,
                          bars_per_year=BPY, max_leverage=3.0)

fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True, height_ratios=[1, 1, 1.2])
axes[0].plot(returns.index, returns.rolling(24 * 14).std() * np.sqrt(BPY), linewidth=0.9, color="#cf222e")
axes[0].axhline(0.20, color="#0969da", linestyle="--", linewidth=1, label="target 20%")
axes[0].set_ylabel("realised vol")
axes[0].legend(fontsize=8)
axes[1].plot(scale.index, scale, linewidth=0.9, color="#0969da")
axes[1].set_ylabel("position scale")
for name, res, colour in [("unsized", plain, "#9198a1"), ("vol-targeted", targeted, "#1a7f37")]:
    axes[2].plot(res.equity.index, res.equity / res.equity.iloc[0], linewidth=1.3,
                 label=name, color=colour)
axes[2].set_ylabel("growth of 1")
axes[2].set_yscale("log")
axes[2].legend(fontsize=8)
fig.suptitle("Volatility rises, exposure falls", y=0.995)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### The honest caveat
#
# Volatility targeting reliably controls **volatility**. It does not reliably
# improve returns, and on a strongly trending market it will systematically cut
# your exposure during exactly the violent rallies you wanted to be in.
#
# It is a trade: less variance, less upside, far lower chance of ruin. Usually
# worth making. Not free.

# %% [markdown]
# ## 3. Drawdown limits
#
# A portfolio-level stop: when equity falls more than $D$ from its peak, cut
# exposure until it recovers.

# %%
guard = DrawdownGuard(MomentumStrategy(48), max_drawdown=0.20, reduced_exposure=0.0, recover_at=0.10)
guarded = run_backtest(df, guard, cfg)

print(f"{'':30} {'max DD':>9} {'Sharpe':>9} {'return':>10} {'vol':>8}")
for name, res in [("unsized", plain), ("vol-targeted", targeted), ("drawdown guard 20%", guarded)]:
    print(f"  {name:28} {M.max_drawdown(res.equity):>9.1%} {M.sharpe_ratio(res.returns, BPY):>+9.2f} "
          f"{M.total_return(res.equity):>+10.1%} {M.annualized_volatility(res.returns, BPY):>8.1%}")

print(f"\nthe guard halted and resumed {len(guard.halt_log)} times:")
for timestamp, action, dd in guard.halt_log[:6]:
    print(f"  {timestamp}  {action:7} at {dd:+.1%}")

# %% [markdown]
# ### The honest caveat, again
#
# > **A portfolio stop-loss systematically sells low.** It reduces the chance of
# > ruin, and on most strategies it also reduces expected return. Look at the
# > `return` column above and check whether that happened here.
#
# That trade is often worth making — being alive next year matters more than
# being optimal this year — but it is a trade, not free insurance, and anyone
# who presents a drawdown limit as pure upside is not telling you the whole story.

# %% [markdown]
# ## 4. Fixed-fractional sizing
#
# The one sizing rule worth memorising:
#
# $$\text{units} = \frac{\text{equity} \times \text{risk fraction}}{\text{distance to stop}}$$
#
# Risk a fixed fraction of capital per trade. Position size then falls
# automatically when your stop has to be far away — that is, when the market is
# volatile.

# %%
equity = 10_000.0
print(f"equity {equity:,.0f}, risking 1% ({equity * 0.01:,.0f}) per trade:\n")
for atr_pct in (0.005, 0.01, 0.02, 0.05, 0.10):
    price = 20_000.0
    stop_distance = price * atr_pct * 2       # a 2-ATR stop
    units = fixed_fractional_size(equity, price, stop_distance, risk_per_trade=0.01)
    print(f"  volatility {atr_pct:>5.1%}  ->  {units:.5f} units "
          f"({units * price / equity:>6.1%} of equity)  loss at stop {units * stop_distance:>7,.0f}")

print("\nEvery row loses exactly 1% of equity if the stop is hit. That is the point.")

# %% [markdown]
# ## 5. Leverage and the Kelly criterion
#
# **Leverage does not improve a strategy.** It scales returns and volatility
# together, leaving the Sharpe ratio unchanged — right up until it does not,
# because a large enough drawdown ends the game permanently.

# %%
print(f"{'leverage':>10} {'return':>11} {'vol':>9} {'Sharpe':>9} {'max DD':>9} {'ruined?':>9}")
for leverage in (0.5, 1.0, 2.0, 3.0, 5.0):
    scaled_cfg = BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY, max_leverage=leverage)

    class Levered(Strategy):
        def __init__(self, inner, multiple):
            self.inner, self.multiple = inner, multiple
            self.warmup, self.name = inner.warmup, f"{multiple}x"

        def on_bar(self, view, account):
            raw = self.inner.on_bar(view, account)
            return None if raw is None else raw * self.multiple

    res = run_backtest(df, Levered(MomentumStrategy(48), leverage), scaled_cfg)
    ruined = "YES" if res.equity.min() <= 0 or M.max_drawdown(res.equity) < -0.95 else "no"
    print(f"{leverage:>10.1f} {M.total_return(res.equity):>+11.1%} "
          f"{M.annualized_volatility(res.returns, BPY):>9.1%} {M.sharpe_ratio(res.returns, BPY):>+9.2f} "
          f"{M.max_drawdown(res.equity):>9.1%} {ruined:>9}")

# %% [markdown]
# Watch the Sharpe column stay roughly flat while the drawdown column gets worse.
# That is leverage in one table: **it buys you nothing risk-adjusted and it
# shortens the distance to zero.**

# %% [markdown]
# ### Kelly, and why nobody uses full Kelly
#
# The Kelly criterion maximises long-run log wealth **given that you know the
# true mean and variance**. You do not. You have a noisy estimate from a short
# sample, and Kelly is extremely sensitive to an overestimated mean.
#
# Betting **twice** the Kelly fraction has an expected growth rate of exactly
# **zero**. Beyond that it is negative. Since your estimate of the mean is
# routinely off by a factor of two, full Kelly is a coin flip on whether you are
# actually betting double Kelly.

# %%
mean_return, variance = 0.0002, 0.0004
full = kelly_fraction(mean_return, variance, safety_factor=1.0)
print(f"estimated mean {mean_return:.4f}, variance {variance:.4f}\n")
print(f"  full Kelly      {full:>6.2f}x")
print(f"  half Kelly      {full * 0.5:>6.2f}x")
print(f"  quarter Kelly   {full * 0.25:>6.2f}x   <- the library default, and still aggressive")

print("\nIf your true mean were half what you estimated, full Kelly becomes")
print(f"double Kelly ({full:.2f}x against a true optimum of {full / 2:.2f}x),")
print("whose expected long-run growth rate is zero.")

# %% [markdown]
# ## 6. Composing the wrappers

# %%
combined = DrawdownGuard(
    VolatilityTargeted(MomentumStrategy(48), target_vol=0.20, max_leverage=2.0, bars_per_year=BPY),
    max_drawdown=0.25,
)
combined_result = run_backtest(df, combined, cfg)

final = {
    "unsized momentum": plain,
    "vol-targeted": targeted,
    "drawdown guard": guarded,
    "vol-target + DD guard": combined_result,
    "buy & hold": run_backtest(df, BuyAndHold(), cfg),
}

from algotrade.report import compare

print(compare(final, bars_per_year=BPY).to_string(float_format=lambda v: f"{v:,.3f}"))

# %%
fig, ax = plt.subplots(figsize=(11, 4.4))
for name, res in final.items():
    ax.plot(res.equity.index, res.equity / res.equity.iloc[0], linewidth=1.3,
            label=f"{name}  (DD {M.max_drawdown(res.equity):.0%})")
ax.set_yscale("log")
ax.set_ylabel("growth of 1 (log scale)")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Same signal throughout. Only the sizing differs.", loc="left", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 11.1 — Verify the target is hit
#
# Sweep `target_vol` over 5%, 10%, 20%, 40%. For each, measure the **realised**
# annualised volatility of the resulting equity curve.
#
# Plot realised against target. It should be close to a 45° line. Where it
# deviates, explain why — consider the leverage cap and the lag in the estimator.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 11.2 — What the drawdown guard costs
#
# Sweep `max_drawdown` over 10%, 15%, 20%, 30%, 50%. Plot realised max drawdown
# and total return against the limit.
#
# Find where the curve bends: the point beyond which tightening the limit costs
# more return than it saves drawdown. Then state which limit **you** would
# choose, and say what you are optimising for.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 11.3 — Sizing beats signal
#
# Take a deliberately mediocre signal — `RandomStrategy` with a slight long bias,
# or `MomentumStrategy` on `gbm.csv`.
#
# Apply your best sizing scheme. Then take a *good* signal with terrible sizing
# (5× leverage, no limits). Compare the two on max drawdown and on whether the
# account survives.
#
# Write one sentence about what this implies for where to spend your effort.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 11.4 — The ruin calculation
#
# Assume a strategy with a true Sharpe ratio of 0.5 and 20% annualised
# volatility. Using a Monte Carlo simulation over 1,000 paths of one year:
#
# 1. What fraction of paths experience a drawdown worse than 20%?
# 2. …worse than 40%?
# 3. Now repeat at 3× leverage. What fraction lose more than 90%?
#
# This is the calculation that should precede any decision to use leverage.

# %%
# Your code here.

# %% [markdown]
# ---
# ## What you should take away
#
# - Keep signal and sizing separate; it lets you debug them separately.
# - Volatility targeting controls volatility reliably. It does **not** reliably
#   improve returns, and it cuts exposure during the rallies you wanted.
# - A drawdown limit reduces ruin and usually reduces return. State both.
# - **Leverage does not improve a strategy.** It leaves Sharpe unchanged and
#   shortens the distance to zero.
# - Nobody sane uses full Kelly, because nobody knows the true mean.
#
# Next: **12 · Paper trading** — running it forward in real time, and finding out
# what your backtest forgot.
