# %% [markdown]
# # 09 · Walk-forward validation — one split is one sample
#
# **Day 3 · ~2 hours**
#
# Notebook 08 ended with an uncomfortable result: a model trained on data with
# **no signal in it** produced a gross Sharpe ratio of 1.8 over its test period.
# A single train/test split gave a confident, wrong answer.
#
# One split is one sample. You would not estimate a mean from one observation,
# and a backtest on one test period is exactly that.
#
# By the end of this notebook you will be able to:
#
# 1. Explain precisely why `train_test_split(shuffle=True)` is catastrophic here.
# 2. Build walk-forward splits with purging and embargo, and prove they do not leak.
# 3. Produce a continuous out-of-sample equity curve by stitching test windows.
# 4. Measure how much performance *varies* across folds, which is usually the
#    real finding.
#
# ![walk-forward vs shuffled CV](../diagrams/svg/03-walk-forward-vs-naive-cv.svg)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from algotrade import BacktestConfig, RETAIL_CRYPTO, ZERO_COST, run_backtest
from algotrade import metrics as M
from algotrade.features import make_features
from algotrade.labeling import fixed_horizon_class
from algotrade.strategy import BuyAndHold, Strategy
from algotrade.validation import (
    assert_no_leakage, describe_splits, expanding_window_splits,
    purged_kfold_splits, walk_forward_splits,
)

BPY = algotrade.BARS_PER_YEAR["1h"]
HORIZON = 1
df = algotrade.load_ohlcv("data/synthetic/momentum.csv")

X_all = make_features(df)
labels = fixed_horizon_class(df["close"], horizon=HORIZON, threshold=0.0)
X, y = labels.aligned_with(X_all)
forward = labels.meta["fwd_return"].reindex(X.index)
print(f"{len(X):,} aligned samples, {X.shape[1]} features")

# %% [markdown]
# ## 1. Why shuffling is catastrophic
#
# `sklearn`'s `train_test_split` shuffles by default. On a time series that puts
# Tuesday in training and Monday in test, so the model **interpolates between
# known points** rather than extrapolating into an unknown future.
#
# Let us measure the damage. Same data, same model, two splitting schemes.

# %%
def fit_and_score(train_idx, test_idx):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=0))
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    return accuracy_score(y.iloc[test_idx], model.predict(X.iloc[test_idx]))


n = len(X)
idx = np.arange(n)

shuffled_train, shuffled_test = train_test_split(idx, test_size=0.3, shuffle=True, random_state=0)
chrono_train, chrono_test = idx[: int(n * 0.7) - HORIZON], idx[int(n * 0.7):]

print(f"shuffled split      accuracy {fit_and_score(shuffled_train, shuffled_test):.2%}")
print(f"chronological split accuracy {fit_and_score(chrono_train, chrono_test):.2%}")
print(f"\nthe shuffled number is higher, and it is worth nothing.")

# %% [markdown]
# ### Why the shuffled number is worthless, precisely
#
# Two mechanisms, and the second is subtler:
#
# 1. **Adjacent bars are nearly identical.** A 168-bar feature at time $t$ and at
#    $t+1$ share 167 of their 168 inputs. Shuffling puts near-duplicates in both
#    sets, so the model is graded on examples it has effectively memorised.
# 2. **The label horizon overlaps.** A label at $t$ is built from prices up to
#    $t+h$. A training sample at $t$ and a test sample at $t+1$ describe
#    overlapping futures.
#
# The fix for (1) is chronological ordering. The fix for (2) is **purging**.
#
# `assert_no_leakage` catches both.

# %%
from algotrade.validation import Split

try:
    assert_no_leakage([Split(train=shuffled_train, test=shuffled_test, fold=0)])
except AssertionError as exc:
    print("shuffled split REJECTED:")
    print(f"  {exc}")

assert_no_leakage([Split(train=chrono_train, test=chrono_test, fold=0)], label_horizon=HORIZON)
print("\nchronological split with purge: accepted")

# %% [markdown]
# ## 2. Walk-forward: how a strategy is actually run
#
# Walk-forward mirrors live operation: fit on a window of recent history, trade
# forward for a period, then refit and repeat. Each test window is genuinely
# unseen at the time it is traded, so stitching them together produces **one
# continuous out-of-sample equity curve**.
#
# Four parameters:
#
# - `train_size` — how much history to fit on.
# - `test_size` — how long to trade before refitting.
# - `purge` — samples dropped from the end of training, ≥ your label horizon.
# - `embargo` — samples skipped after a test window, to break serial correlation.

# %%
TRAIN, TEST, PURGE, EMBARGO = 3000, 500, HORIZON, 24

splits = walk_forward_splits(len(X), train_size=TRAIN, test_size=TEST, purge=PURGE, embargo=EMBARGO)
assert_no_leakage(splits, label_horizon=HORIZON)

print(f"{len(splits)} folds, no leakage detected\n")
print(describe_splits(splits, X.index).head(6).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(11, 3.4))
for sp in splits:
    ax.barh(sp.fold, len(sp.train), left=sp.train.min(), height=0.62, color="#0969da", alpha=0.75)
    ax.barh(sp.fold, len(sp.test), left=sp.test.min(), height=0.62, color="#1a7f37", alpha=0.85)
    ax.barh(sp.fold, PURGE + EMBARGO, left=sp.train.max(), height=0.62, color="#d29922")
ax.set_xlabel("sample index")
ax.set_ylabel("fold")
ax.invert_yaxis()
ax.set_title("Walk-forward folds: blue = train, amber = purge+embargo, green = test",
             loc="left", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Run the walk-forward
#
# Refit at every fold. Predict only on that fold's test window. Collect the
# predictions into one continuous series covering the whole out-of-sample period.

# %%
oos_edge = pd.Series(np.nan, index=X.index, name="edge")
fold_scores = []

for sp in splits:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=0))
    model.fit(X.iloc[sp.train], y.iloc[sp.train])

    classes = list(model.classes_)
    proba = model.predict_proba(X.iloc[sp.test])
    edge = proba[:, classes.index(1.0)] - proba[:, classes.index(-1.0)]
    oos_edge.iloc[sp.test] = edge

    fwd = forward.iloc[sp.test]
    fold_scores.append({
        "fold": sp.fold,
        "test_start": X.index[sp.test.min()].date(),
        "accuracy": accuracy_score(y.iloc[sp.test], model.predict(X.iloc[sp.test])),
        "IC": np.corrcoef(edge, fwd)[0, 1],
        "mean_fwd_when_bullish": fwd[edge > 0].mean(),
    })

folds = pd.DataFrame(fold_scores)
print(folds.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

# %% [markdown]
# ## 4. Consistency across folds, not just the average
#
# Look at the `IC` column. If the effect were real and stable, every fold would
# be positive and of similar size. If it were noise, the folds would disagree.
#
# This dataset has a genuinely planted edge, so we should expect agreement --
# and if we do not get it, our method is too weak to detect a signal we *know*
# is there.

# %%
ic = folds["IC"]
print(f"folds                  : {len(ic)}")
print(f"mean IC                : {ic.mean():+.4f}")
print(f"standard deviation     : {ic.std():.4f}")
print(f"folds with positive IC : {int((ic > 0).sum())} of {len(ic)}")
print(f"worst fold             : {ic.min():+.4f}")
print(f"best fold              : {ic.max():+.4f}")

t_stat = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else np.nan
print(f"\nt-statistic of mean IC : {t_stat:+.2f}   "
      f"({'significant' if abs(t_stat) > 2 else 'NOT significant'} at ~5%)")

# %% [markdown]
# ### The check that makes this meaningful
#
# A t-statistic above 5 sounds impressive until you ask the obvious question:
# **what would this procedure report on data with no edge at all?**
#
# Notebook 08 showed that a single train/test split was fooled — the noise model
# produced a gross Sharpe of 1.8. Here is the same walk-forward procedure applied
# to all three datasets.

# %%
def walk_forward_ic(dataset: str):
    """Mean IC, dispersion and t-statistic from a walk-forward on one dataset."""
    data = algotrade.load_ohlcv(f"data/synthetic/{dataset}.csv")
    feats = make_features(data)
    lab = fixed_horizon_class(data["close"], horizon=HORIZON, threshold=0.0)
    Xd, yd = lab.aligned_with(feats)
    fwd = lab.meta["fwd_return"].reindex(Xd.index)
    folds_ = walk_forward_splits(len(Xd), train_size=TRAIN, test_size=TEST,
                                 purge=PURGE, embargo=EMBARGO)
    values = []
    for sp in folds_:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=0))
        m.fit(Xd.iloc[sp.train], yd.iloc[sp.train])
        cl = list(m.classes_)
        pr = m.predict_proba(Xd.iloc[sp.test])
        e = pr[:, cl.index(1.0)] - pr[:, cl.index(-1.0)]
        values.append(np.corrcoef(e, fwd.iloc[sp.test])[0, 1])
    values = pd.Series(values)
    return {
        "mean IC": values.mean(),
        "sd IC": values.std(),
        "positive": f"{int((values > 0).sum())}/{len(values)}",
        "t-stat": values.mean() / (values.std() / np.sqrt(len(values))),
    }


truth_labels = {
    "momentum": "real edge, phi=0.06",
    "pure_noise": "NO edge by construction",
    "gbm": "NO edge by construction",
}
print(f"{'dataset':>12} {'ground truth':>26} {'mean IC':>10} {'positive':>10} {'t-stat':>8}")
for dataset, description in truth_labels.items():
    stats_row = walk_forward_ic(dataset)
    print(f"{dataset:>12} {description:>26} {stats_row['mean IC']:>+10.4f} "
          f"{stats_row['positive']:>10} {stats_row['t-stat']:>+8.2f}")

# %% [markdown]
# > **This is walk-forward validation earning its keep.**
# >
# > On the dataset with a planted edge it finds it in every single fold, with a
# > t-statistic above 5. On the two datasets built with no edge it reports
# > roughly half the folds positive and a t-statistic near zero — correctly
# > concluding that there is nothing there.
# >
# > Recall that in notebook 08 a *single* train/test split gave the noise model a
# > gross Sharpe of 1.8 and no way to tell it was an accident. The difference
# > between those two outcomes is the entire value of this notebook.
#
# Two rules follow:
#
# 1. **Report the fold-by-fold dispersion and the t-statistic**, not just the
#    mean. "Positive in 5 of 10 folds, t = −1.1" is a complete and honest result.
# 2. **Run your validation on data with no edge.** If it cannot tell the
#    difference, it is not validating anything.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
colors = ["#1a7f37" if v > 0 else "#cf222e" for v in ic]
axes[0].bar(folds["fold"], ic, color=colors, alpha=0.85)
axes[0].axhline(0, color="#57606a", linewidth=1)
axes[0].axhline(ic.mean(), color="#0969da", linestyle="--", linewidth=1.2,
                label=f"mean {ic.mean():+.4f}")
axes[0].set_xlabel("fold")
axes[0].set_ylabel("information coefficient")
axes[0].legend(fontsize=8)
axes[0].set_title("Per-fold IC on momentum.csv (real edge)", loc="left", fontsize=9)

axes[1].bar(folds["fold"], folds["accuracy"] - 0.5, color="#0969da", alpha=0.8)
axes[1].axhline(0, color="#57606a", linewidth=1)
axes[1].set_xlabel("fold")
axes[1].set_ylabel("accuracy - 50%")
axes[1].set_title("Per-fold accuracy edge", loc="left", fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. The stitched out-of-sample equity curve
#
# This is the only performance number in the workshop that counts as evidence:
# every bar of it was produced by a model that had not seen that bar's period.

# %%
class WalkForwardSignal(Strategy):
    def __init__(self, edge: pd.Series, *, min_edge=0.0, name="walk-forward ML"):
        self.edge, self.min_edge, self.name, self.warmup = edge, min_edge, name, 1

    def on_bar(self, view, account):
        value = self.edge.get(view.now)
        if value is None or not np.isfinite(value):
            return 0.0
        if abs(value) < self.min_edge:
            return 0.0
        return float(np.clip(value, -1.0, 1.0))


oos_start = X.index[splits[0].test.min()]
oos_df = df.loc[oos_start:]

runs = {
    "walk-forward, gross": (WalkForwardSignal(oos_edge, min_edge=0.15), ZERO_COST),
    "walk-forward, net": (WalkForwardSignal(oos_edge, min_edge=0.15), RETAIL_CRYPTO),
    "buy & hold, net": (BuyAndHold(), RETAIL_CRYPTO),
}
results = {
    name: run_backtest(oos_df, strategy, BacktestConfig(cost_model=cost, bars_per_year=BPY))
    for name, (strategy, cost) in runs.items()
}

print(f"continuous out-of-sample period: {oos_start.date()} to {oos_df.index[-1].date()} "
      f"({len(oos_df):,} bars)\n")
print(f"{'':24} {'Sharpe':>9} {'return':>10} {'max DD':>9} {'turnover':>10}")
for name, res in results.items():
    print(f"  {name:22} {M.sharpe_ratio(res.returns, BPY):>+9.2f} "
          f"{M.total_return(res.equity):>+10.1%} {M.max_drawdown(res.equity):>9.1%} "
          f"{M.turnover(res.weights, BPY):>10.0f}")

# %%
fig, ax = plt.subplots(figsize=(11, 4.4))
for name, res in results.items():
    ax.plot(res.equity.index, res.equity / res.equity.iloc[0], linewidth=1.3,
            label=f"{name}  ({M.total_return(res.equity):+.1%})")
for sp in splits[1:]:
    ax.axvline(X.index[sp.test.min()], color="#d0d7de", linewidth=0.7, zorder=0)
ax.axhline(1.0, color="#57606a", linewidth=0.8, linestyle=":")
ax.set_ylabel("growth of 1")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Continuous out-of-sample equity. Vertical lines are refits.",
             loc="left", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Rolling vs expanding vs purged K-fold
#
# Three schemes, three different questions:
#
# | scheme | training window | answers |
# |---|---|---|
# | **rolling walk-forward** | fixed length, slides forward | "does it work if I refit on recent data?" |
# | **expanding walk-forward** | grows from the start | "does it work if old data stays relevant?" |
# | **purged K-fold** | everything except the fold and its neighbourhood | "how much does the model vary?" — **not** a trading simulation |
#
# Purged K-fold trains on data that comes *after* the test fold. That is fine for
# estimating model variance with limited data, and it is **not** an out-of-sample
# equity curve. If you present it as one, you are misrepresenting it.

# %%
schemes = {
    "rolling walk-forward": walk_forward_splits(len(X), train_size=TRAIN, test_size=TEST,
                                                purge=PURGE, embargo=EMBARGO),
    "expanding walk-forward": expanding_window_splits(len(X), min_train=TRAIN, test_size=TEST,
                                                      purge=PURGE),
    "purged K-fold (NOT a simulation)": purged_kfold_splits(len(X), n_splits=8, purge=PURGE,
                                                            embargo=EMBARGO),
}

print(f"{'scheme':>34} {'folds':>7} {'mean IC':>10} {'sd IC':>9} {'positive':>10}")
for name, scheme in schemes.items():
    assert_no_leakage(scheme, label_horizon=HORIZON,
                      require_causal="K-fold" not in name)
    ics = []
    for sp in scheme:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=0))
        m.fit(X.iloc[sp.train], y.iloc[sp.train])
        cl = list(m.classes_)
        pr = m.predict_proba(X.iloc[sp.test])
        e = pr[:, cl.index(1.0)] - pr[:, cl.index(-1.0)]
        ics.append(np.corrcoef(e, forward.iloc[sp.test])[0, 1])
    ics = pd.Series(ics)
    print(f"{name:>34} {len(scheme):>7} {ics.mean():>+10.4f} {ics.std():>9.4f} "
          f"{int((ics > 0).sum()):>6} / {len(ics)}")

# %% [markdown]
# Notice that purged K-fold typically reports the most flattering numbers, because
# it gets to train on the future. That is precisely why it must never be
# presented as a performance estimate.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 9.1 — The cost of forgetting to purge
#
# Run the walk-forward with `purge=0` and a label horizon of 24 instead of 1.
# Compare the mean IC with the correctly purged version.
#
# Then explain, in terms of which specific samples overlap, where the difference
# came from.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 9.2 — Refit frequency
#
# Sweep `test_size` over 100, 250, 500, 1000, 2000 with `train_size` fixed.
#
# For each: mean IC, its standard deviation across folds, and the net Sharpe of
# the stitched curve. Is there a best refit frequency? What is the trade-off you
# are navigating? (Consider: more folds means more evidence but less training
# data per fold, and refitting is not free in production either.)

# %%
# Your code here.

# %% [markdown]
# ### Exercise 9.3 — How many folds do you need?
#
# Section 4 showed that 10 folds cleanly separated the real edge (t = +5.18)
# from noise (t = −1.14). Would 3 have?
#
# Re-run the three-dataset comparison with `test_size` chosen to give 3, 5, 10
# and 20 folds. For each, record whether the t-statistic would have led you to
# the right conclusion on all three datasets.
#
# Then answer: what is the minimum number of folds you would accept in your own
# capstone, and why?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 9.4 — Choosing a hyperparameter legitimately
#
# In Exercise 8.2 you chose `min_edge` by looking at test performance, which is
# data snooping. Fix it: use a **nested** scheme in which, within each
# walk-forward fold, you select `min_edge` using only that fold's training data,
# then apply it to the test window.
#
# Compare the result with the value you "chose" in 8.2. Is the honest version
# worse? (It usually is. That gap is the size of the lie.)

# %%
# Your code here.

# %% [markdown]
# ---
# ## What you should take away
#
# - Shuffled splits are catastrophic here for two reasons: adjacent bars are
#   near-duplicates, and label horizons overlap.
# - Walk-forward mirrors live operation and yields one continuous out-of-sample
#   curve — the only performance number in this workshop that is evidence.
# - **Report the dispersion across folds, not just the mean.** "Positive in 7 of
#   12 folds" with a t-statistic near zero is the honest summary of most results.
# - Purged K-fold measures model variance, not trading performance. Do not
#   present it as the latter.
#
# Next: **10 · Honest evaluation** — assembling all of this into a report you
# would be willing to defend.
