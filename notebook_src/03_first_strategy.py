# %% [markdown]
# # 03 · Your first strategy — and it makes 46,986%
#
# **Day 1 · ~2 hours. This is the payoff notebook for Day 1.**
#
# We are going to build a moving-average crossover strategy the way almost every
# tutorial on the internet builds one: vectorised pandas, a few lines, a grid
# search to pick the best parameters, and a beautiful equity curve.
#
# It will report a return of roughly **+46,986%**.
#
# It is worthless, and tomorrow we will take it apart piece by piece and watch it
# become a **−71% loss**. But today we build it honestly — meaning we build it
# exactly as badly as it is normally built — because you cannot learn to spot
# this failure mode from a description. You have to feel the pull of the number
# first.
#
# By the end of this notebook you will be able to:
#
# 1. Turn a trading idea into a signal series.
# 2. Run the standard vectorised backtest.
# 3. Search a parameter grid and pick the winner.
# 4. State precisely what you have and have not demonstrated. (Spoiler: nothing.)

# %% [markdown]
# ## 1. The idea
#
# **Moving-average crossover.** When a short-window average of price rises above
# a long-window average, the trend is up: be long. When it falls below, be flat.
#
# It is the oldest idea in technical trading. There is a plausible story behind it
# — trends persist because information spreads slowly and people chase — and
# there is decades of academic argument about whether any of it survives costs.
#
# We use it because it is simple enough that every bias we introduce will be
# visible in it.

# %%
from algotrade import metrics as M
from algotrade.indicators import sma

df = algotrade.load_ohlcv("data/synthetic/trending_with_crash.csv")
close = df["close"]
BPY = algotrade.BARS_PER_YEAR["1h"]

FAST, SLOW = 12, 48
fast_ma = sma(close, FAST)
slow_ma = sma(close, SLOW)

signal = (fast_ma > slow_ma).astype(float)   # 1.0 = long, 0.0 = flat

print(f"{len(df):,} hourly bars, {(df.index[-1] - df.index[0]).days} days")
print(f"signal is long {signal.mean():.1%} of the time")
print(f"it changes its mind {int(signal.diff().abs().sum()):,} times")

# %%
window = slice(2000, 3000)
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(close.index[window], close.iloc[window], linewidth=1.0, label="price", color="#57606a")
ax.plot(fast_ma.index[window], fast_ma.iloc[window], linewidth=1.2, label=f"SMA {FAST}")
ax.plot(slow_ma.index[window], slow_ma.iloc[window], linewidth=1.2, label=f"SMA {SLOW}")
ax.fill_between(close.index[window], close.iloc[window].min(), close.iloc[window].max(),
                where=signal.iloc[window] > 0, alpha=0.12, color="green", label="long")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("The signal: long when the fast average is above the slow one", loc="left", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. The standard vectorised backtest
#
# Here it is, the version you will find in a thousand blog posts:
#
# ```python
# returns = prices.pct_change()
# strategy_returns = signal * returns
# equity = (1 + strategy_returns).cumprod()
# ```
#
# Three lines. Fast, elegant, and containing a defect we will not name until
# tomorrow. `algotrade.naive.vectorized_backtest` implements exactly this, and
# warns you every time you call it, because it is a teaching exhibit rather than
# a tool.

# %%
import warnings

from algotrade.naive import NaiveBacktestWarning, vectorized_backtest

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    backtest = vectorized_backtest(close, signal, shift=0, cost_bps=0.0)
    for w in caught:
        print(f"WARNING: {w.message}\n")

equity = backtest["equity"]
print(f"total return    : {equity.iloc[-1] - 1:+,.1%}")
print(f"max drawdown    : {(equity / equity.cummax() - 1).min():.1%}")
print(f"Sharpe (annual) : {M.sharpe_ratio(backtest['net_return'], BPY):+.2f}")

# %% [markdown]
# Already better than buy-and-hold, with a smaller drawdown. But we have not
# optimised anything yet. Let us do what everyone does next.

# %% [markdown]
# ## 3. The grid search
#
# We do not actually know that 12 and 48 are the right windows. So we try a
# reasonable range of both and keep whichever performs best.
#
# This is a completely standard thing to do. It is also, as Day 2 will show, one
# of the three reasons the number below is fiction.

# %%
import itertools

fast_values = [3, 6, 12, 24, 48]
slow_values = [24, 48, 96, 192, 384]
combos = [(f, s) for f, s in itertools.product(fast_values, slow_values) if f < s]

results = []
for fast, slow in combos:
    sig = (sma(close, fast) > sma(close, slow)).astype(float)
    bt = vectorized_backtest(close, sig, shift=0, cost_bps=0.0, quiet=True)
    eq = bt["equity"]
    results.append({
        "fast": fast, "slow": slow,
        "total_return": eq.iloc[-1] - 1,
        "sharpe": M.sharpe_ratio(bt["net_return"], BPY),
        "max_dd": (eq / eq.cummax() - 1).min(),
        "flips": int(sig.diff().abs().sum()),
    })

grid = pd.DataFrame(results).sort_values("total_return", ascending=False)
print(f"tried {len(combos)} parameter combinations\n")
print(grid.head(8).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

# %%
pivot = grid.pivot(index="fast", columns="slow", values="total_return")

fig, ax = plt.subplots(figsize=(7.5, 4))
im = ax.imshow(np.log10(pivot.to_numpy() + 1), cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(pivot.columns)), pivot.columns)
ax.set_yticks(range(len(pivot.index)), pivot.index)
ax.set_xlabel("slow window")
ax.set_ylabel("fast window")
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.to_numpy()[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:,.0f}x" if v > 10 else f"{v:.1f}x", ha="center", va="center", fontsize=8)
ax.set_title("Total return by parameter pair (log colour scale)", loc="left", fontsize=10)
fig.colorbar(im, ax=ax, label="log10(1 + return)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. The winner

# %%
best = grid.iloc[0]
best_fast, best_slow = int(best["fast"]), int(best["slow"])
best_signal = (sma(close, best_fast) > sma(close, best_slow)).astype(float)
best_bt = vectorized_backtest(close, best_signal, shift=0, cost_bps=0.0, quiet=True)
best_equity = best_bt["equity"]

buy_hold = close / close.iloc[0]

print(f"BEST PARAMETERS: SMA {best_fast} / {best_slow}\n")
print(f"  total return      {best_equity.iloc[-1] - 1:>15,.1%}")
print(f"  buy and hold      {buy_hold.iloc[-1] - 1:>15,.1%}")
print(f"  max drawdown      {(best_equity / best_equity.cummax() - 1).min():>15.1%}")
print(f"  Sharpe ratio      {M.sharpe_ratio(best_bt['net_return'], BPY):>15.2f}")
print(f"  trades            {int(best_signal.diff().abs().sum()):>15,}")

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(best_equity.index, best_equity, linewidth=1.3, label=f"SMA {best_fast}/{best_slow}", color="#1a7f37")
ax.plot(buy_hold.index, buy_hold, linewidth=1.1, label="buy & hold", color="#57606a", alpha=0.8)
ax.set_yscale("log")
ax.set_ylabel("growth of 1 (log scale)")
ax.legend()
ax.set_title("The Day 1 result", loc="left", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Stop. What have we actually demonstrated?
#
# Look at that equity curve. It goes up and to the right, almost monotonically,
# through a dataset that contains four crashes of 20–40% each. The drawdown is
# tiny. The Sharpe ratio is enormous.
#
# Before you go any further, write down honest answers to these:
#
# 1. **On what data was this evaluated?** All of it. The same data used to choose
#    the parameters.
# 2. **How many strategies were tried before this one was selected?** 22.
# 3. **What did it cost to place those trades?** Nothing. We set `cost_bps=0`.
# 4. **When exactly did each trade execute, relative to the information that
#    triggered it?** …you do not actually know, do you.
#
# That fourth question is the one that matters, and we deliberately have not
# examined it. We will tomorrow, first thing.
#
# ### The thing to notice about yourself
#
# You are probably a little bit excited by that number. That reaction is the
# subject of this entire workshop. Every person who has lost money to a trading
# algorithm felt exactly this, looked at a chart exactly like that, and did not
# ask question 4.

# %% [markdown]
# ### A test we can run right now
#
# Here is a check that costs nothing and would have saved a great many people a
# great deal of money. Run the **identical procedure** — same grid, same search,
# same selection rule — on `pure_noise.csv`, a dataset generated as a driftless
# random walk with **no predictability in it whatsoever**.
#
# If the method finds a wonderful strategy there too, the method is not finding
# strategies. It is finding noise.

# %%
noise = algotrade.load_ohlcv("data/synthetic/pure_noise.csv")["close"]

noise_results = []
for fast, slow in combos:
    sig = (sma(noise, fast) > sma(noise, slow)).astype(float)
    bt = vectorized_backtest(noise, sig, shift=0, cost_bps=0.0, quiet=True)
    noise_results.append({"fast": fast, "slow": slow, "total_return": bt["equity"].iloc[-1] - 1})

noise_grid = pd.DataFrame(noise_results).sort_values("total_return", ascending=False)
print("Best strategies found on data that is PURE RANDOM WALK by construction:\n")
print(noise_grid.head(5).to_string(index=False, float_format=lambda v: f"{v:,.2%}"))
print(f"\nbuy and hold on the same file: {noise.iloc[-1] / noise.iloc[0] - 1:+.2%}")

# %% [markdown]
# Sit with that for a moment. The ground truth for that file, written down in
# `data/synthetic/manifest.json` before any of this ran, is:
#
# > *"Driftless random walk. Ground truth: NO strategy has an edge."*
#
# And the procedure found one anyway.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 3.1 — Widen the search
#
# Expand the grid to include `fast` values of 2 and 4 and `slow` values of 12 and
# 768. Does the best total return improve? By how much?
#
# Then: does "the best return improved when I searched more combinations" give
# you *more* confidence in the result, or less? Justify your answer in two
# sentences. (There is a right answer.)

# %%
# Your code here.

# %% [markdown]
# ### Exercise 3.2 — Your own strategy
#
# Write a different signal and run it through the same procedure. Ideas:
#
# - **RSI reversion**: long when `rsi(close, 14) < 30`, flat when `> 70`.
# - **Donchian breakout**: long when price exceeds the trailing 20-bar high.
# - **Z-score reversion**: long when `zscore(close, 48) < -1.5`.
#
# All are available in `algotrade.indicators`. Record the best total return you
# can reach. Then run the identical procedure on `pure_noise.csv` and record what
# you get there too. Keep both numbers — you will want them on Day 2.

# %%
from algotrade.indicators import bollinger, donchian, rsi, zscore

# Your code here.

# %% [markdown]
# ### Exercise 3.3 — Sensitivity
#
# Take your best parameter pair and change each window by ±1. Does performance
# change smoothly, or does it fall off a cliff?
#
# A strategy whose result collapses when you change a window from 12 to 13 has
# not found a market pattern. It has found one specific arrangement of one
# specific price history. Write down which of those two you think you have.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 3.4 — Predict tomorrow
#
# Before opening notebook 04, write down your prediction. Tomorrow we will
# re-run this exact strategy with (a) the execution timing made explicit and
# (b) realistic trading costs of 24 basis points per round trip.
#
# Your prediction for the new total return: `________`
#
# Write it down. Actually write it down. Comparing it with the real answer is
# worth more than reading the answer.

# %%
# my_prediction = ...

# %% [markdown]
# ---
# ## What you should take away
#
# - You can build a spectacular-looking backtest in about twenty lines.
# - The same twenty lines produce a spectacular-looking backtest on data that is
#   provably random.
# - Therefore the spectacular-looking backtest is not evidence of anything.
#
# We have not yet said *why* it is wrong. That is tomorrow, and there are three
# separate reasons, which we will measure one at a time.
#
# Next: **04 · The event-driven backtester** — where we build an engine that
# cannot lie to us in the first place.
