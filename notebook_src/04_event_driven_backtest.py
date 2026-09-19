# %% [markdown]
# # 04 · The event-driven backtester — an engine that cannot lie
#
# **Day 2 · ~2 hours**
#
# Yesterday's backtest was three lines of pandas. Today we replace it with a loop,
# which is slower, longer, and the single most important piece of engineering in
# this repository.
#
# Here is why. The vectorised backtest requires you to *remember* to shift your
# signal by the right number of bars, every time, for every derived column.
# Discipline fails — especially at 2am. So instead of relying on it, we build an
# engine in which **the future is not in the room**: the strategy is handed a
# view of the data that is physically truncated at the current bar, and reaching
# past it raises an exception.
#
# By the end of this notebook you will be able to:
#
# 1. Describe the five steps the engine performs on every bar, and why the order matters.
# 2. Write a strategy against the `MarketView` API.
# 3. Try to cheat, and watch it fail loudly.
# 4. Explain the difference between *look-ahead bias* and *execution optimism*.
#
# ![the backtest loop](../diagrams/svg/01-backtest-loop.svg)

# %% [markdown]
# ## 1. The loop
#
# For each bar `i`:
#
# 1. **Advance the clock to bar `i`.** The strategy can now see bars `0..i`, and
#    nothing beyond. This one line is the whole guarantee.
# 2. **Settle orders submitted on bar `i-1`**, filling at bar `i`'s **open**,
#    minus fees and spread.
# 3. **Mark to market** at bar `i`'s close.
# 4. **Ask the strategy** for a decision, given data through bar `i`'s close.
#    Anything it returns is queued for bar `i+1`.
# 5. **Record** equity, weight and position.
#
# Steps 2 and 4 together are the timing rule: **a signal computed from bar `i`'s
# close fills at bar `i+1`'s open.** One full bar of delay, always. There is no
# shift to forget.

# %%
from algotrade import BacktestConfig, MarketView, Strategy, run_backtest
from algotrade.errors import LookAheadError

df = algotrade.load_ohlcv("data/synthetic/trending_with_crash.csv")
BPY = algotrade.BARS_PER_YEAR["1h"]
print(f"{len(df):,} bars loaded")

# %% [markdown]
# ## 2. What a strategy can see
#
# A strategy implements one method:
#
# ```python
# def on_bar(self, view, account) -> float | None:
#     ...
# ```
#
# `view` is a `MarketView`. It exposes `open`, `high`, `low`, `close` and `volume`,
# and every one of them is truncated at the current bar. The return value is a
# **target weight**: the fraction of your equity to hold in the asset. `1.0` is
# fully long, `0.0` flat, `-0.5` half short. Returning `None` means "no change".
#
# Returning a *weight* rather than "buy 3 units" separates the signal from the
# sizing — which is what lets Day 4 change the sizing without touching any
# strategy.

# %%
from algotrade.view import Clock

# Build a view by hand and park the clock at bar 200, to see what is visible.
clock = Clock(200)
view = MarketView(df, clock)

print(f"current bar index : {view.i}")
print(f"current timestamp : {view.now}")
print(f"bars elapsed      : {len(view):,}")
print(f"\nview.close[-1]    : {view.close[-1]:,.2f}    <- the current bar's close")
print(f"view.close[-2]    : {view.close[-2]:,.2f}    <- the bar before")
print(f"view.close[0]     : {view.close[0]:,.2f}    <- the very first bar")
print(f"view.close.history(5): {np.round(view.close.history(5), 2)}")

# %% [markdown]
# ## 3. Now try to cheat
#
# Every one of these is a mistake a student makes. Every one of them raises.

# %%
attempts = {
    "view.close[view.i + 1]": lambda: view.close[view.i + 1],
    "view.close[len(view)]": lambda: view.close[len(view)],
    "view.close[0:view.i + 5]": lambda: view.close[0 : view.i + 5],
    "view.close[view.total_bars - 1]": lambda: view.close[view.total_bars - 1],
}

for description, attempt in attempts.items():
    try:
        attempt()
        print(f"  {description:34} NOT BLOCKED  <-- this would be a serious bug")
    except LookAheadError as exc:
        print(f"  {description:34} blocked")
        print(f"      {str(exc)[:96]}")

# %% [markdown]
# Note that the escape hatches are closed too — `np.asarray(view.close)` returns
# only the visible window, iterating stops at the current bar, and
# `view.frame()` gives you a DataFrame of the past only.

# %%
print(f"len(np.asarray(view.close)) = {len(np.asarray(view.close)):,}  (not {len(df):,})")
print(f"len(list(view.close))       = {len(list(view.close)):,}")
print(f"len(view.frame())           = {len(view.frame()):,}")
print(f"view.frame(3):\n{view.frame(3)[['open', 'close']].to_string()}")

# %% [markdown]
# ### How honest is this guard?
#
# **Honest about accidents, not about sabotage.** The full array is still stored
# on a private attribute, so a determined student can reach it with
# `view.close._values`. That is fine and deliberate.
#
# The goal is to make look-ahead impossible to commit **by accident** — via an
# off-by-one, a forgotten `.shift(1)`, or a rolling mean computed over the whole
# history before the loop starts. Those are the mistakes that actually happen.
# Defeating the guard requires typing an underscore, which is a decision rather
# than a slip.

# %% [markdown]
# ## 4. Writing a strategy
#
# Here is yesterday's moving-average crossover, written for the engine.

# %%
class MovingAverageCross(Strategy):
    """Long when the fast trailing average exceeds the slow one, else flat."""

    def __init__(self, fast: int = 3, slow: int = 24):
        self.fast, self.slow = fast, slow
        self.warmup = slow
        self.name = f"SMA {fast}/{slow}"

    def on_bar(self, view, account):
        # Guard the warm-up. Without this, history() raises because there is
        # not enough data yet -- which is the correct behaviour, but you should
        # handle it rather than crash.
        if len(view) < self.warmup:
            return None

        closes = view.close.history(self.slow)
        fast_ma = closes[-self.fast:].mean()
        slow_ma = closes.mean()
        return 1.0 if fast_ma > slow_ma else 0.0


result = run_backtest(df, MovingAverageCross(3, 24), BacktestConfig(
    cost_model=algotrade.ZERO_COST,      # still zero, so we compare like for like
    bars_per_year=BPY,
))
print(result)

# %% [markdown]
# ## 5. The first number falls
#
# Yesterday this strategy returned **+46,986%**. That was the vectorised backtest
# with `shift=0`. Here is the same strategy, same data, same zero costs, run
# through an engine where the timing is explicit.

# %%
naive_equity = algotrade.naive.vectorized_backtest(
    df["close"], (algotrade.indicators.sma(df["close"], 3) > algotrade.indicators.sma(df["close"], 24)).astype(float),
    shift=0, cost_bps=0.0, quiet=True,
)["equity"]

buy_hold = df["close"] / df["close"].iloc[0]

print(f"  yesterday (vectorised, shift=0)   {naive_equity.iloc[-1] - 1:>14,.1%}")
print(f"  today     (event-driven engine)   {algotrade.metrics.total_return(result.equity):>14,.1%}")
print(f"  buy & hold                        {buy_hold.iloc[-1] - 1:>14,.1%}")

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(naive_equity.index, naive_equity, linewidth=1.3, color="#cf222e",
        label=f"vectorised, shift=0  ({naive_equity.iloc[-1] - 1:+,.0%})")
normalised = result.equity / result.equity.iloc[0]
ax.plot(normalised.index, normalised, linewidth=1.3, color="#1a7f37",
        label=f"event-driven engine  ({algotrade.metrics.total_return(result.equity):+,.0%})")
ax.plot(buy_hold.index, buy_hold, linewidth=1.0, color="#57606a", alpha=0.7,
        label=f"buy & hold  ({buy_hold.iloc[-1] - 1:+,.0%})")
ax.set_yscale("log")
ax.set_ylabel("growth of 1 (log scale)")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Same strategy. Same data. Still zero costs. Only the timing changed.",
             loc="left", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# **A factor of about 115× of the "profit" was look-ahead bias.** Not costs — we
# have not added any yet. Purely the difference between "earn the return of the
# bar you used to decide" and "earn the return of the next bar".
#
# What is left is still a respectable-looking number. Notebook 05 deals with that.

# %% [markdown]
# ## 6. Look-ahead bias vs execution optimism
#
# These get conflated constantly, and they are different. The engine supports
# three `fill_timing` settings:
#
# | setting | meaning | honest? |
# |---|---|---|
# | `next_open` | decide at bar `i` close, fill at bar `i+1` open | yes (default) |
# | `next_close` | decide at bar `i` close, fill at bar `i+1` close | yes, slightly pessimistic |
# | `this_close` | decide at bar `i` close, fill at bar `i` close | **no** |
#
# `this_close` is unrealistic: you cannot simultaneously observe a closing price
# and trade at it. But — and this surprises people — it is **not** look-ahead
# bias, and it does not reliably inflate your returns.

# %%
timings = {}
for timing in ("next_open", "next_close", "this_close"):
    res = run_backtest(df, MovingAverageCross(3, 24), BacktestConfig(
        cost_model=algotrade.ZERO_COST, fill_timing=timing, bars_per_year=BPY,
    ))
    timings[timing] = algotrade.metrics.total_return(res.equity)

for timing, value in timings.items():
    marker = "  <- unrealistic" if timing == "this_close" else ""
    print(f"  {timing:12} {value:>10,.1%}{marker}")

# %% [markdown]
# On this dataset `this_close` happens to do slightly better. On others it does
# worse. Measured across 40 dataset/seed combinations it beat `next_open` **22
# times** — a coin flip.
#
# Compare that with genuine look-ahead, which beat correct indexing in **40 out
# of 40** trials, on every dataset, including ones built with no signal in them
# at all.
#
# > **That asymmetry is the diagnostic.** A result that is merely *unrealistic*
# > is wrong in an unpredictable direction. A result contaminated by *look-ahead*
# > is wrong in one direction, always, on any data. If your "edge" works on pure
# > noise, it is not an edge, it is a leak.
#
# ![the look-ahead trap](../diagrams/svg/02-lookahead-trap.svg)

# %% [markdown]
# ## 7. What the engine records

# %%
print(f"bars          : {len(result.equity):,}")
print(f"trades        : {result.n_trades:,}")
print(f"final equity  : {result.equity.iloc[-1]:,.2f}")
print(f"\nfirst few fills:")
print(result.trades_frame().head(4).to_string())

print("\nNote the target weights the strategy ASKED for, which differ from the")
print("weights it ACHIEVED, because fills happen at the next bar's open:")
comparison = pd.DataFrame({"asked_for": result.targets, "achieved": result.weights}).iloc[100:104]
print(comparison.to_string())

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 4.1 — Break the guard
#
# Write a `Strategy` subclass that tries to read a future price, in a way not
# already shown above. Ideas: negative slice bounds, `np.array(view.close)[-1:]`,
# passing the view to a pandas constructor, stashing the view on `self` during
# `on_start` and reading it later.
#
# Run it. Either it raises `LookAheadError` — in which case describe in a
# sentence *which* line of `algotrade/view.py` stopped you — or it does not, in
# which case **you have found a real bug** and should write a failing test for it
# in `tests/test_lookahead_guard.py`.

# %%
# Your cheating strategy here.

# %% [markdown]
# ### Exercise 4.2 — Port your Day 1 strategy
#
# Take the strategy you wrote in Exercise 3.2 and reimplement it as a `Strategy`
# subclass. Run it through the engine with `cost_model=ZERO_COST` so the only
# thing that changes is the timing.
#
# Record: how much of your Day 1 return survives? Express it as a ratio.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 4.3 — The warm-up guard
#
# Delete the `if len(view) < self.warmup: return None` line from
# `MovingAverageCross` and run it. Read the exception carefully.
#
# Then answer: would it have been *better* for `history()` to silently return
# however many bars it had? Give one argument for and one against, then say which
# you would choose and why.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 4.4 — Instrument the loop
#
# Write a strategy that does nothing but record, on every bar, the values of
# `view.i`, `view.now`, `len(view)`, `account.equity` and `account.weight`.
#
# Run it, then reconstruct by hand exactly which bar each of your fills happened
# on, and verify the one-bar delay holds for every single trade. Do not take the
# engine's word for it.

# %%
# Your code here.

# %% [markdown]
# ---
# ## What you should take away
#
# - The engine's ordering — advance, settle, mark, decide, record — is what makes
#   the timing honest. The strategy is asked **last** and cannot act until the
#   next bar.
# - Look-ahead bias is structurally impossible here, and the tests in
#   `tests/test_lookahead_guard.py` prove it by trying to cheat four different ways.
# - Fixing the timing alone removed ~115× of yesterday's "profit".
# - **Unrealistic and inflated are different properties.** Only inflated results
#   are wrong in a predictable direction — and those are the ones that fool you.
#
# Next: **05 · Costs and slippage** — where the rest of it goes.
