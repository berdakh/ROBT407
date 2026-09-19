# %% [markdown]
# # 01 · Market data: what it is, and how it lies to you first
#
# **Day 1 · ~90 minutes**
#
# Before any strategy, before any model, you have to be able to trust the numbers
# in front of you. Most of the ways a backtest goes wrong start here, in a CSV
# that looked fine.
#
# By the end of this notebook you will be able to:
#
# 1. Read an OHLCV file and say exactly what each number means and *when* it was known.
# 2. Find the four data defects that silently corrupt a backtest: unsorted rows,
#    duplicated bars, missing bars, and impossible prices.
# 3. Resample to a coarser timeframe without inventing prices that never traded.
# 4. Explain why the timestamp convention matters more than it sounds like it does.

# %% [markdown]
# ## 1. What an OHLCV bar actually is
#
# A **bar** summarises every trade in a time window with five numbers:
#
# | field | meaning | when you know it |
# |---|---|---|
# | `open` | first trade price in the window | at the start |
# | `high` | highest trade price | only at the end |
# | `low` | lowest trade price | only at the end |
# | `close` | last trade price | only at the end |
# | `volume` | total quantity traded | only at the end |
#
# That last column is the one that matters. **Four of the five numbers are only
# knowable once the bar is over.** A strategy that uses a bar's `close` to decide
# something, and then acts at a price inside that same bar, is cheating. We will
# come back to this relentlessly.
#
# ### The timestamp convention
#
# A bar labelled `09:00` could mean "the window starting at 09:00" or "the window
# ending at 09:00". Both conventions exist in the wild. Getting it wrong shifts
# your entire dataset by one bar, which is exactly the size of the error that
# makes a useless strategy look brilliant.
#
# **This workshop always labels bars by their OPEN time, in UTC.** A `1h` bar
# labelled `09:00` covers `[09:00, 10:00)`, and its close is known at `10:00`.

# %%
from algotrade.data import (
    BARS_PER_YEAR,
    data_quality_report,
    find_gaps,
    load_ohlcv,
    resample_ohlcv,
    validate_ohlcv,
)

df = load_ohlcv("data/synthetic/gbm.csv")
print(df.head())
print(f"\n{len(df):,} bars from {df.index[0]} to {df.index[-1]}")
print(f"index timezone: {df.index.tz}")

# %% [markdown]
# ### Why the index is timezone-aware
#
# `load_ohlcv` refuses a timezone-naive index. This is not fussiness. Naive
# timestamps compare and merge incorrectly across sources, and if any of your
# data ever touches a market with daylight saving, a naive index silently shifts
# by an hour twice a year — for a few days, in the middle of your sample, in a
# way no summary statistic will reveal.
#
# Crypto trades 24/7 in UTC, which is one of the reasons this workshop uses it:
# no market hours, no holidays, no session boundaries. One less thing to be
# wrong about.

# %%
# Prove the guard works: strip the timezone and watch it refuse.
from algotrade.errors import DataIntegrityError

naive = df.copy()
naive.index = naive.index.tz_localize(None)

try:
    validate_ohlcv(naive)
except DataIntegrityError as exc:
    print("REJECTED:", exc)

# %% [markdown]
# ## 2. The four defects that corrupt a backtest
#
# `validate_ohlcv` checks for all of them. Let us break the data deliberately and
# watch each one get caught — this is the fastest way to remember what to look for.

# %%
import numpy as np
import pandas as pd

breakages = {
    "unsorted rows": lambda d: d.iloc[::-1],
    "duplicated bar": lambda d: pd.concat([d, d.iloc[[5]]]).sort_index(),
    "high below low": lambda d: d.assign(high=d["high"].mask(d.index == d.index[10], d["low"].iloc[10] * 0.5)),
    "negative volume": lambda d: d.assign(volume=d["volume"].mask(d.index == d.index[3], -1.0)),
    "NaN close": lambda d: d.assign(close=d["close"].mask(d.index == d.index[7], np.nan)),
}

for name, break_it in breakages.items():
    try:
        validate_ohlcv(break_it(df))
        print(f"  {name:18} NOT CAUGHT  <-- this would be a bug in the validator")
    except DataIntegrityError as exc:
        print(f"  {name:18} caught: {str(exc)[:70]}")

# %% [markdown]
# ### Why each one matters
#
# - **Unsorted rows** make every rolling window meaningless. `rolling(20).mean()`
#   happily averages twenty rows in whatever order they appear.
# - **A duplicated bar** double-counts one period's return. Compounded over a
#   year of hourly data, a handful of duplicates is a visible fake profit.
# - **`high < low`** means the file is corrupt or your columns are mislabelled.
#   If you do not check, you will "discover" arbitrage.
# - **Negative volume** is a parsing error, usually a sign column misread.
# - **A NaN close** propagates into every indicator downstream and often turns
#   into a silent zero somewhere.

# %% [markdown]
# ## 3. Missing bars: the defect that looks like data
#
# The other four defects announce themselves. A **missing bar** does not — the
# file just has fewer rows than you expected, and if you forward-fill it, the gap
# becomes a stretch of flat price, which your model will read as a period of
# unusually low volatility. Exchanges go down. Feeds drop out. This is normal.

# %%
holed = df.drop(df.index[500:512]).drop(df.index[3000:3003])
print("quality report on the damaged file:")
for key, value in data_quality_report(holed, "1h").items():
    print(f"  {key:24} {value}")

print("\ngaps found:")
print(find_gaps(holed, "1h").to_string(index=False))

# %% [markdown]
# **How to handle a gap** — in order of preference:
#
# 1. **Leave it.** An event-driven backtester that iterates bars simply has fewer
#    bars. Nothing breaks. This is usually right.
# 2. **Forward-fill, and record that you did.** Acceptable for short gaps, but
#    now your "volatility" in that window is artificially zero, and any strategy
#    that trades on low volatility will love it for fake reasons.
# 3. **Drop the surrounding period.** Right when the gap is long enough that you
#    do not trust the prices either side of it.
#
# Never interpolate. Interpolation between two known points uses the later one —
# that is look-ahead bias, introduced during data cleaning, before you have
# written a single line of strategy code.

# %% [markdown]
# ## 4. Resampling: how to aggregate without inventing prices
#
# Each column needs a different rule. `open` takes the first, `close` the last,
# `high` the max, `low` the min, `volume` the sum. Calling `.resample("1D").mean()`
# — the obvious thing — produces a "price" that never traded.

# %%
daily = resample_ohlcv(df, "1D")
four_hour = resample_ohlcv(df, "4h")

print(f"hourly: {len(df):,} bars   4h: {len(four_hour):,}   daily: {len(daily):,}")
print("\nfirst daily bar, reconstructed by hand from the 24 hourly bars:")
first_day = df.iloc[:24]
print(f"  open   {first_day['open'].iloc[0]:>12,.2f}   (first hourly open)")
print(f"  high   {first_day['high'].max():>12,.2f}   (max of 24 highs)")
print(f"  low    {first_day['low'].min():>12,.2f}   (min of 24 lows)")
print(f"  close  {first_day['close'].iloc[-1]:>12,.2f}   (LAST hourly close)")
print(f"  volume {first_day['volume'].sum():>12,.2f}   (sum)")
print("\nwhat resample_ohlcv produced:")
print(daily.iloc[0].to_string())

# %% [markdown]
# ### Only ever resample *down*
#
# Going from hourly to daily throws information away, which is fine. Going from
# daily to hourly **invents** 23 bars out of nothing. If you find yourself
# upsampling market data, stop: whatever you are about to do is not going to work.
#
# ### What timeframe should you use?
#
# Shorter bars mean more observations (good for statistics) and more trades (bad
# for costs, which we will quantify on Day 2). There is no universally correct
# answer, but there is a constraint: **your edge per trade must exceed your cost
# per trade.** A strategy on 1-minute bars needs an enormous edge to survive.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for ax, (label, data) in zip(axes, [("hourly", df), ("4-hourly", four_hour), ("daily", daily)]):
    ax.plot(data.index, data["close"], linewidth=0.8)
    ax.set_ylabel(label)
    ax.set_title(f"{label}: {len(data):,} bars", loc="left", fontsize=10)
fig.suptitle("The same price series at three resolutions", y=0.995)
fig.tight_layout()
plt.show()

# %% [markdown]
# Same data, same price path, three very different-looking charts. Notice how much
# calmer the daily series looks. Nothing changed except how much you are averaging
# away — and one of the reasons daily backtests look better than intraday ones is
# exactly this, plus the fact that they trade 24× less often.

# %% [markdown]
# ---
# ## Exercises
#
# The synthetic datasets have known properties, so you can check your own answers.

# %% [markdown]
# ### Exercise 1.1 — Bar timing
#
# For a `1h` bar labelled `2021-03-05 14:00:00+00:00`:
#
# 1. What time window does it cover?
# 2. At what wall-clock time do you first know its `close`?
# 3. At what wall-clock time do you first know its `high`?
# 4. If your strategy computes a signal from this bar's close, what is the
#    *earliest* timestamp at which you could possibly have traded on it?
#
# Write your answers as a comment. Question 4 is the one that matters, and we will
# turn it into code on Day 2.

# %%
# Your answers here:
# 1.
# 2.
# 3.
# 4.

# %% [markdown]
# ### Exercise 1.2 — Find the damage
#
# The cell below builds a deliberately corrupted file. Without looking at how it
# was corrupted, write code that finds **every** problem and reports it.
#
# Use `validate_ohlcv`, `data_quality_report` and `find_gaps`. Note that
# `validate_ohlcv` raises on the *first* problem it finds, so you will need to
# fix each one before you can see the next.

# %%
def make_corrupted(clean: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bad = clean.iloc[:2000].copy()
    bad = bad.drop(bad.index[rng.integers(100, 1900, 7)])           # missing bars
    bad = pd.concat([bad, bad.iloc[[42, 900]]]).sort_index()        # duplicates
    pos = bad.columns.get_loc("high")
    bad.iloc[311, pos] = bad["low"].iloc[311] * 0.9                 # impossible bar
    bad.iloc[1200, bad.columns.get_loc("volume")] = -5.0            # bad volume
    return bad


corrupted = make_corrupted(df)
print(f"{len(corrupted):,} rows. Find everything wrong with it.")

# Your investigation here.

# %% [markdown]
# ### Exercise 1.3 — Resample and verify
#
# Resample the hourly data to `12h` bars, then **verify your result** by
# reconstructing three randomly chosen bars by hand from the underlying hourly
# data. If your reconstruction disagrees with `resample_ohlcv`, find out which of
# you is wrong before continuing.
#
# Then answer: how many `12h` bars would you expect from 8,760 hourly bars, and
# do you get that number? If not, why not?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 1.4 — A real file (optional, needs network)
#
# Run `python scripts/fetch_market_data.py --days 180` and load the result.
# Then run `data_quality_report` and `find_gaps` on it and compare with the
# synthetic data.
#
# You will find things the synthetic data does not have. Note down what they are —
# that list is why we do not trust generators alone.

# %% [markdown]
# ---
# ## What you should take away
#
# - Four of a bar's five numbers are only known **after** the bar ends. Every
#   look-ahead bug is ultimately a violation of that sentence.
# - Validate before you analyse. A corrupt file produces confident, wrong answers.
# - Missing bars are normal and are the one defect that does not announce itself.
# - Resample down, never up, and never with `.mean()`.
#
# Next: **02 · Returns and risk** — turning prices into the quantities you can
# actually reason about statistically.
