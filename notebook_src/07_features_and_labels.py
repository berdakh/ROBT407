# %% [markdown]
# # 07 · Features and labels — where leakage actually happens
#
# **Day 3 · ~2 hours**
#
# Machine learning on market data fails for two reasons far more often than it
# fails because the model was wrong:
#
# 1. **The features leaked.** Something in the input knew the future.
# 2. **The features were not stationary.** The model learned a relationship that
#    only held in the training period.
#
# Both happen during feature construction, before any model exists. This notebook
# is about getting that right, because no amount of cross-validation rescues a
# feature matrix that already contains the answer.
#
# By the end you will be able to:
#
# 1. Build a causal feature matrix and **prove** it is causal.
# 2. Recognise non-stationary features and explain why they destroy a model.
# 3. Construct labels that remember their forward horizon.
# 4. Align features with labels without silently misaligning them.

# %%
from algotrade import indicators as ind
from algotrade.features import FEATURE_DESCRIPTIONS, check_stationarity, make_features
from algotrade.labeling import fixed_horizon_class, fixed_horizon_return, triple_barrier

df = algotrade.load_ohlcv("data/synthetic/momentum.csv")
BPY = algotrade.BARS_PER_YEAR["1h"]

# This dataset has a KNOWN edge in it -- AR(1) returns with phi = 0.06 -- so a
# model that finds nothing here is broken, and a model that finds a lot is
# probably leaking.
import json
from pathlib import Path

truth = json.loads(Path("data/synthetic/manifest.json").read_text())["momentum"]
print("ground truth for this dataset:")
for k, v in truth["ground_truth"].items():
    print(f"  {k:18} {v}")
print(f"\n  {truth['lesson']}")

# %% [markdown]
# ## 1. The causality rule
#
# > **A feature at time $t$ may use only data from times $\le t$.**
#
# Sounds obvious. Here are the ways it gets violated anyway, all of which look
# completely ordinary in a code review:
#
# ```python
# df["ma"] = df["close"].rolling(20, center=True).mean()   # centred: uses t+10
# df["z"]  = (df["c"] - df["c"].mean()) / df["c"].std()    # full-sample statistics
# df = df.fillna(method="bfill")                            # copies the future backwards
# df["x"] = df["x"].interpolate()                           # uses the value after the gap
# scaler.fit(X)                                             # fitted on train AND test
# ```
#
# The last one is the most common in ML pipelines specifically. `StandardScaler`
# fitted on the whole dataset before splitting leaks test-set statistics into
# training.

# %%
X = make_features(df)
print(f"{X.shape[1]} features, {len(X):,} rows, {int(X.isna().any(axis=1).sum())} warm-up rows\n")
for name, description in FEATURE_DESCRIPTIONS.items():
    print(f"  {name:14} {description}")

# %% [markdown]
# ## 2. Proving causality rather than asserting it
#
# Here is a test you should run on every feature you ever write. **Corrupt the
# future and check that the past does not move.** If it does, the feature is
# reading forward.

# %%
SPLIT = 1500
corrupted = df.copy()
corrupted.iloc[SPLIT:] = corrupted.iloc[SPLIT:] * 3.0    # violently change the future

before = make_features(df).iloc[:SPLIT]
after = make_features(corrupted).iloc[:SPLIT]

moved = [c for c in before.columns if not before[c].equals(after[c])]
print(f"features that changed in the PAST when the FUTURE was corrupted: {moved or 'none'}")
print("\n(This exact check runs in tests/test_causality.py for every indicator.)")

# %%
# And confirm the probe is actually sensitive -- otherwise it proves nothing.
bad_feature = lambda d: d["close"].rolling(21, center=True).mean()
a, b = bad_feature(df).iloc[:SPLIT], bad_feature(corrupted).iloc[:SPLIT]
print(f"a deliberately centred window IS detected as non-causal: {not a.equals(b)}")

# %% [markdown]
# ## 3. Stationarity — the quieter killer
#
# A feature is **stationary** if its statistical properties do not drift over
# time. Raw price is emphatically not: a model trained when BTC was \$8,000 has
# never seen \$60,000 and will extrapolate nonsense.
#
# The failure is silent. You split chronologically (correctly!), the model does
# badly out of sample, and you conclude "the strategy stopped working" — when the
# truth is the feature was never valid.

# %%
X_with_bad = X.assign(
    raw_price=df["close"],                                    # non-stationary: drifts
    raw_volume=df["volume"].cumsum(),                         # worse: monotonic
)
report = check_stationarity(X_with_bad)
print("Features ranked by drift (mean_drift is in units of the feature's own sd):\n")
print(report.head(8).to_string(float_format=lambda v: f"{v:+.3f}"))
print("\nAnything above ~1.0 will not survive a chronological split.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, col, title in zip(
    axes,
    ["raw_price", "ret_24", "zscore_168"],
    ["raw price (NOT stationary)", "24-bar return (stationary)", "z-score (stationary)"],
):
    series = X_with_bad[col].dropna()
    ax.plot(series.index, series, linewidth=0.7)
    ax.set_title(title, loc="left", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
fig.tight_layout()
plt.show()

# %% [markdown]
# The left panel wanders. The other two oscillate around a stable level. Only the
# latter two describe a relationship a model could learn once and reuse.
#
# **How to make a feature stationary:** take a return or a difference, divide by
# a trailing volatility, or convert to a trailing z-score or rank. Never feed a
# raw price level to a model.

# %% [markdown]
# ## 4. Labels: what exactly are you predicting?
#
# A label *must* look forward — that is what makes it a target. The danger is
# forgetting **how far** forward, because that horizon determines how much
# training data you have to purge around each test fold (notebook 09).

# %% [markdown]
# ### Fixed-horizon return — the simple one

# %%
labels = fixed_horizon_return(df["close"], horizon=24)
print(f"kind: {labels.kind},  horizon: {labels.horizon} bars")
print(f"\nThe last {labels.horizon} values are NaN, because their future has not happened:")
print(labels.y.tail(3).to_string())
print(f"\nIf your label series has no NaN tail, you have a bug.")

# %% [markdown]
# ### Fixed-horizon classification — with a threshold that matters
#
# A neutral band is not cosmetic. **Set the threshold to at least your round-trip
# cost**, because a predicted move smaller than that is not tradeable even if you
# predict it perfectly.

# %%
round_trip = algotrade.RETAIL_CRYPTO.round_trip_bps() / 10_000
print(f"round-trip cost = {round_trip:.4f} ({round_trip:.2%})\n")

for threshold in (0.0, round_trip, 0.01, 0.02):
    lab = fixed_horizon_class(df["close"], horizon=24, threshold=threshold)
    counts = lab.y.value_counts(normalize=True).sort_index()
    tradeable = 1 - counts.get(0.0, 0.0)
    print(f"  threshold {threshold:6.2%}:  down {counts.get(-1.0, 0):.1%}  "
          f"flat {counts.get(0.0, 0):.1%}  up {counts.get(1.0, 0):.1%}   "
          f"({tradeable:.0%} of bars tradeable)")

# %% [markdown]
# ### Triple-barrier labelling — the one traders actually want
#
# Fixed-horizon asks "what is the return exactly 24 bars from now", which no
# trader has ever cared about. **Triple barrier** asks the real question: *if I
# enter now, do I hit my profit target or my stop first?*
#
# Three barriers: an upper (take profit), a lower (stop loss), and a vertical one
# (time runs out). The label is whichever is touched first.

# %%
tb = triple_barrier(df["close"], horizon=48, upper=0.02, lower=0.02,
                    high=df["high"], low=df["low"])

print(f"label distribution: {tb.y.value_counts().sort_index().to_dict()}")
print(f"  +1 = profit target hit first")
print(f"  -1 = stop loss hit first")
print(f"   0 = neither, time barrier reached")

offsets = tb.meta["touch_bar"] - np.arange(len(df))
print(f"\nbars until resolution: median {offsets.median():.0f}, max {offsets.max():.0f}")
print("\nThat median is the TRUE horizon of most samples, and it is what you")
print("should purge by -- usually much shorter than the nominal 48.")

# %% [markdown]
# > **An honest note on the implementation.** When both barriers are touched
# > within the same bar, OHLC data cannot tell you which came first. We assume
# > the worse outcome. Assuming the better one is a small, systematic,
# > compounding lie — and it is what most naive implementations do.

# %% [markdown]
# ## 5. Alignment — the bug that produces a working-looking model
#
# Features have NaNs at the **start** (indicator warm-up). Labels have NaNs at
# the **end** (no future yet). If you drop them separately, your features and
# labels shift relative to each other by the length of the warm-up, and your
# model quietly learns to predict the wrong bar.

# %%
labels = fixed_horizon_class(df["close"], horizon=24, threshold=round_trip)

# WRONG -- separate drops.
X_wrong = X.dropna()
y_wrong = labels.y.dropna()
print(f"dropped separately: X has {len(X_wrong):,} rows, y has {len(y_wrong):,}. "
      f"Different lengths, and misaligned by {len(X) - len(X_wrong)} bars.")

# RIGHT -- align, then drop rows where either is missing.
X_ok, y_ok = labels.aligned_with(X)
print(f"aligned together  : {len(X_ok):,} rows, indices match exactly: {X_ok.index.equals(y_ok.index)}")

# %% [markdown]
# `Labels.aligned_with` does this correctly and is the only way you should
# combine the two in this workshop.

# %% [markdown]
# ## 6. Do the features contain the known signal?
#
# `momentum.csv` was built with AR(1) returns, $\phi = 0.06$. If our features are
# any good, `ret_1` should have some relationship with the next bar's return —
# and crucially, that relationship should be **small**.

# %%
future_return = np.log(df["close"].shift(-1) / df["close"])
correlations = X.corrwith(future_return).sort_values(key=abs, ascending=False)

print("Correlation of each feature with the NEXT bar's return:\n")
print(correlations.head(8).to_string(float_format=lambda v: f"{v:+.4f}"))
print(f"\nThe generator's phi was {truth['ground_truth']['phi']}, and ret_1's correlation")
print(f"with the next return is {correlations['ret_1']:+.4f}. Those should be close, and are.")

# %% [markdown]
# **Look at the size of those numbers.** A correlation of 0.06 is a real,
# deliberately-planted, exploitable signal — and it explains 0.4% of the variance.
#
# This is what a genuine edge looks like. If a feature of yours correlates 0.4
# with the future, you have not found a brilliant signal. You have found a leak.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 7.1 — Write a causal feature and prove it
#
# Add three features of your own to a copy of `make_features`. Ideas: distance
# from a rolling maximum, the ratio of up-volume to down-volume, the slope of a
# linear fit over the last $n$ bars.
#
# Then run the corruption test on each. Any that fail, fix.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 7.2 — Break causality on purpose
#
# Write a feature that is 95% causal and 5% leaky — for example, a rolling mean
# where you accidentally use `shift(-1)` on one input.
#
# 1. Confirm the corruption test catches it.
# 2. Measure its correlation with the future return.
# 3. Compare that number with the honest features above.
#
# You should find it is dramatically larger. **That size difference is your
# smoke alarm.** Write down the threshold above which you would investigate.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 7.3 — Choose a label
#
# For a strategy that holds positions for roughly a day (24 hourly bars), compare:
#
# - `fixed_horizon_class(horizon=24, threshold=0)`
# - `fixed_horizon_class(horizon=24, threshold=round_trip)`
# - `triple_barrier(horizon=24, upper=0.02, lower=0.02)`
# - `triple_barrier(horizon=24, upper=0.03, lower=0.01)` (asymmetric)
#
# For each, report the class balance and the median bars-to-resolution. Then
# argue for one of them in three sentences. There is no single right answer, but
# there are wrong ones — and "whichever gives the best accuracy" is the wrongest.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 7.4 — The stationarity audit
#
# Run `check_stationarity` on your full feature matrix using `n_chunks=8`.
# Identify the three most drifting features. For each, propose a transformation
# that would make it stationary, apply it, and verify the drift falls.

# %%
# Your code here.

# %% [markdown]
# ---
# ## What you should take away
#
# - Leakage happens in **feature construction**, before any model exists. No
#   validation scheme repairs a leaky feature matrix.
# - Prove causality by corrupting the future and checking the past. Assert
#   nothing.
# - Non-stationary features fail silently and look like "the strategy stopped
#   working".
# - Labels must remember their horizon; that number drives purging on Day 3.
# - **A real edge is small.** Correlations of 0.05 are exploitable; correlations
#   of 0.4 are bugs.
#
# Next: **08 · ML signals** — where we find out that accuracy has almost nothing
# to do with profit.
