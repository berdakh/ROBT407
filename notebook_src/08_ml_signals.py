# %% [markdown]
# # 08 · ML signals — the model finds the edge, and it does not matter
#
# **Day 3 · ~2 hours**
#
# This notebook has an unusual shape. We are going to train a classifier on data
# with a **real, known, deliberately planted edge**, and it is going to find it:
# out-of-sample gross Sharpe ratio of about **+3.2**.
#
# Then we are going to charge 24 basis points per round trip, and the net Sharpe
# ratio will be about **−9.7**.
#
# The machine learning worked. The strategy did not. Those are different
# statements, and confusing them is why so many people conclude either that "ML
# doesn't work in markets" (false) or that a good validation score means they
# have a business (also false).
#
# By the end you will be able to:
#
# 1. Choose a prediction horizon for a reason rather than by trial and error.
# 2. Train a signal model with no leakage and evaluate it honestly.
# 3. Explain why accuracy is nearly useless as a trading metric.
# 4. Compute the break-even cost of an ML strategy — the number that decides
#    everything.

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from algotrade import BacktestConfig, CostModel, RETAIL_CRYPTO, ZERO_COST, run_backtest
from algotrade import metrics as M
from algotrade.features import make_features
from algotrade.labeling import fixed_horizon_class
from algotrade.report import compare
from algotrade.strategy import BuyAndHold, Strategy

BPY = algotrade.BARS_PER_YEAR["1h"]
df = algotrade.load_ohlcv("data/synthetic/momentum.csv")
ROUND_TRIP = RETAIL_CRYPTO.round_trip_bps() / 10_000

# %% [markdown]
# ## 1. Choosing the horizon, for a reason
#
# How many bars ahead should we predict? This is not a hyperparameter to tune —
# tuning it on test performance is data snooping, and we spent all of Day 2 on
# why that is fatal.
#
# Here we have an advantage we will never have again: **we know the
# data-generating process.** `momentum.csv` is AR(1) in returns with
# $\phi = 0.06$, meaning
#
# $$\mathbb{E}[r_{t+1} \mid r_t] = \phi\, r_t$$
#
# The predictable component **decays geometrically**: by $h$ bars ahead only
# $\phi^h$ of it survives, while the noise grows as $\sqrt{h}$. So the signal is
# at lag 1 and essentially nowhere else.

# %%
phi = 0.06
print(f"{'horizon':>8} {'predictable':>12} {'noise':>8} {'signal/noise':>13}")
for h in (1, 2, 3, 6, 12, 24):
    predictable = phi * (1 - phi**h) / (1 - phi)
    noise = np.sqrt(h)
    print(f"{h:>8} {predictable:>12.4f} {noise:>8.2f} {predictable / noise:>13.4f}")

HORIZON = 1
print(f"\nWe predict {HORIZON} bar ahead, because that is where the signal is.")
print("In real markets you do not know this. You choose the horizon from economic")
print("reasoning, or on TRAINING data -- never by checking which one scores best")
print("on your test set.")

# %% [markdown]
# ## 2. Build the dataset without leaking
#
# Three rules, all established in notebook 07:
#
# 1. Features are causal.
# 2. Features and labels are aligned **together**.
# 3. The split is chronological, with a **purge gap** of `HORIZON` bars — because
#    a label at time $t$ is built from prices up to $t + h$, so training samples
#    within $h$ of the test set have already seen it.

# %%
X_all = make_features(df)
labels = fixed_horizon_class(df["close"], horizon=HORIZON, threshold=0.0)
X, y = labels.aligned_with(X_all)

split_at = int(len(X) * 0.7)
X_train, y_train = X.iloc[: split_at - HORIZON], y.iloc[: split_at - HORIZON]
X_test, y_test = X.iloc[split_at:], y.iloc[split_at:]

print(f"train : {len(X_train):,} samples  ({X_train.index[0].date()} to {X_train.index[-1].date()})")
print(f"purge : {HORIZON} sample(s) dropped from the end of training")
print(f"test  : {len(X_test):,} samples  ({X_test.index[0].date()} to {X_test.index[-1].date()})")
print(f"\nclass balance (train): {y_train.value_counts(normalize=True).sort_index().round(3).to_dict()}")

# %% [markdown]
# ### The scaler must be fitted on training data only
#
# `make_pipeline` gets this right: `.fit(X_train)` learns the mean and standard
# deviation of the **training** data, and those same numbers are applied at
# predict time. Calling `StandardScaler().fit_transform(X)` on the whole matrix
# before splitting leaks test statistics into training, and is one of the most
# common leaks in ML-for-finance code.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=0))
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
majority = y_test.value_counts(normalize=True).max()

print(f"training accuracy          {train_acc:.2%}")
print(f"test accuracy              {test_acc:.2%}")
print(f"always-predict-majority    {majority:.2%}   <- the bar to beat")
print(f"\nedge over majority: {test_acc - majority:+.2%}")

# %% [markdown]
# **That is the whole edge: under one percentage point of accuracy.** If you were
# grading this as a classification problem you would call it a failure.
#
# It is not a failure. It is roughly what a real tradeable signal looks like.

# %% [markdown]
# ## 3. Did it actually find the planted signal?
#
# Accuracy is a blunt instrument. The **information coefficient** — the
# correlation between the model's confidence and the actual forward return —
# tells you much more.

# %%
classes = list(model.classes_)
proba = model.predict_proba(X_test)
edge = proba[:, classes.index(1.0)] - proba[:, classes.index(-1.0)]
forward = labels.meta["fwd_return"].reindex(X_test.index)

ic = np.corrcoef(edge, forward)[0, 1]
print(f"information coefficient (out of sample): {ic:+.4f}")
print(f"the generator's phi was                : {phi}")
print(f"\nROC AUC: {roc_auc_score((y_test > 0).astype(int), edge):.4f}  (0.5 = no skill)")

# Does the model's confidence actually sort future returns?
buckets = pd.qcut(pd.Series(edge, index=X_test.index), 5, labels=["most bearish", "2", "3", "4", "most bullish"])
print("\nMean forward return by model-confidence quintile:\n")
print(forward.groupby(buckets, observed=True).agg(["mean", "count"]).to_string(
    float_format=lambda v: f"{v:+.5f}"))

# %% [markdown]
# If that column increases monotonically from "most bearish" to "most bullish",
# the model has learned something real. Note the magnitudes — hundredths of a
# percent. **That is what an edge looks like.**

# %% [markdown]
# ## 4. The strategy
#
# Two design choices that matter more than the model:
#
# 1. **Confidence sizing** — use $P(\text{up}) - P(\text{down})$ as the weight
#    rather than a hard ±1, so the model takes small positions when unsure.
# 2. **A dead zone** — do not trade at all when the edge is below a threshold.
#    This is what stops the model churning capital into fees.

# %%
def model_edge(fitted_model, features: pd.DataFrame) -> pd.Series:
    """Precompute P(up) - P(down) for every row, once.

    Is precomputing the whole series look-ahead bias? **No** -- and it is worth
    being precise about why. Each prediction depends only on that row's
    features, and every feature is causal (notebook 07). Row t's prediction
    would be identical if rows t+1 onwards did not exist. Batching is an
    efficiency detail, not a change of information set.

    What WOULD be look-ahead: fitting the model on all rows (it isn't -- it was
    fitted on the training slice), or fitting the scaler on all rows (it wasn't),
    or using a feature with a centred window (none do).

    This is also the one place the engine's guard cannot help you, because the
    strategy indexes its own DataFrame rather than the view. Precomputed
    features are exactly where leakage sneaks back in.
    """
    classes = list(fitted_model.classes_)
    proba = fitted_model.predict_proba(features)
    return pd.Series(
        proba[:, classes.index(1.0)] - proba[:, classes.index(-1.0)],
        index=features.index, name="edge",
    )


class ModelSignal(Strategy):
    """Trade a precomputed confidence series, looking up only the current bar."""

    def __init__(self, edge: pd.Series, *, min_edge=0.0, confidence_sizing=True, name="ML signal"):
        self.edge = edge
        self.min_edge, self.confidence_sizing, self.name = min_edge, confidence_sizing, name
        self.warmup = 1

    def on_bar(self, view, account):
        value = self.edge.get(view.now)          # view.now is the CURRENT bar
        if value is None or not np.isfinite(value):
            return None
        if abs(value) < self.min_edge:
            return 0.0
        return float(np.clip(value, -1.0, 1.0)) if self.confidence_sizing else float(np.sign(value))


edge_series = model_edge(model, X)
test_df = df.loc[X_test.index[0] :]
variants = {
    "hard +/-1": ModelSignal(edge_series, confidence_sizing=False),
    "confidence sized": ModelSignal(edge_series, confidence_sizing=True),
    "confidence + dead zone 0.05": ModelSignal(edge_series, min_edge=0.05),
    "confidence + dead zone 0.15": ModelSignal(edge_series, min_edge=0.15),
}

# %% [markdown]
# ## 5. Gross: the machine learning worked

# %%
gross_cfg = BacktestConfig(cost_model=ZERO_COST, bars_per_year=BPY)
net_cfg = BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY)

rows = []
for name, strategy in variants.items():
    gross = run_backtest(test_df, strategy, gross_cfg)
    net = run_backtest(test_df, strategy, net_cfg)
    rows.append({
        "variant": name,
        "gross Sharpe": M.sharpe_ratio(gross.returns, BPY),
        "net Sharpe": M.sharpe_ratio(net.returns, BPY),
        "gross return": M.total_return(gross.equity),
        "net return": M.total_return(net.equity),
        "turnover/yr": M.turnover(net.weights, BPY),
        "costs paid": net.total_costs,
    })

bh = run_backtest(test_df, BuyAndHold(), net_cfg)
summary = pd.DataFrame(rows).set_index("variant")
print(summary.to_string(float_format=lambda v: f"{v:,.2f}"))
print(f"\nbuy & hold over the same period: "
      f"Sharpe {M.sharpe_ratio(bh.returns, BPY):+.2f}, return {M.total_return(bh.equity):+.1%}")

# %% [markdown]
# ## 6. Net: and it did not matter
#
# Read the `gross Sharpe` and `net Sharpe` columns side by side.
#
# > A gross Sharpe ratio above 3 is, in principle, superb — better than most
# > professional funds achieve. The net Sharpe is deeply negative. **The edge was
# > real and it was smaller than the cost of harvesting it.**

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for name, strategy in variants.items():
    g = run_backtest(test_df, strategy, gross_cfg)
    n = run_backtest(test_df, strategy, net_cfg)
    axes[0].plot(g.equity.index, g.equity / g.equity.iloc[0], linewidth=1.2, label=name)
    axes[1].plot(n.equity.index, n.equity / n.equity.iloc[0], linewidth=1.2, label=name)

for ax, title in zip(axes, ["GROSS: zero costs (fiction)", "NET: 24 bps round trip (reality)"]):
    ax.axhline(1.0, color="#57606a", linewidth=0.8, linestyle=":")
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
axes[0].legend(fontsize=8)
axes[0].set_ylabel("growth of 1")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. The break-even cost
#
# The number that decides whether an ML strategy is a business: **at what cost
# does it stop working?** Compute this first, before tuning anything.

# %%
best_variant = ModelSignal(edge_series, min_edge=0.15)
print(f"{'round trip':>12} {'net Sharpe':>12} {'net return':>12}")
for per_side in (0, 0.25, 0.5, 1, 2, 5, 12):
    cfg = BacktestConfig(
        cost_model=CostModel(commission_bps=per_side, half_spread_bps=0.0, name=f"{per_side}bps"),
        bars_per_year=BPY,
    )
    res = run_backtest(test_df, best_variant, cfg)
    marker = "  <- retail reality" if per_side == 12 else ""
    print(f"{2 * per_side:>10.1f} bps {M.sharpe_ratio(res.returns, BPY):>+12.2f} "
          f"{M.total_return(res.equity):>+12.1%}{marker}")

# %% [markdown]
# Find the row where the Sharpe crosses zero. That is your break-even cost, and
# it is almost certainly a small fraction of what a retail trader actually pays.
#
# **This is the central economics of quantitative trading.** Firms that profit
# from signals this weak do so because they pay a tiny fraction of retail costs —
# they are market makers earning the spread rather than paying it, they have
# exchange rebates, and they are co-located. The signal is not their advantage.
# The cost structure is.

# %% [markdown]
# ## 8. The control: run it on noise

# %%
noise_df = algotrade.load_ohlcv("data/synthetic/pure_noise.csv")
X_noise_all = make_features(noise_df)
noise_labels = fixed_horizon_class(noise_df["close"], horizon=HORIZON, threshold=0.0)
X_noise, y_noise = noise_labels.aligned_with(X_noise_all)

n_split = int(len(X_noise) * 0.7)
noise_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=0))
noise_model.fit(X_noise.iloc[: n_split - HORIZON], y_noise.iloc[: n_split - HORIZON])

noise_proba = noise_model.predict_proba(X_noise.iloc[n_split:])
noise_classes = list(noise_model.classes_)
noise_edge = noise_proba[:, noise_classes.index(1.0)] - noise_proba[:, noise_classes.index(-1.0)]
noise_forward = noise_labels.meta["fwd_return"].reindex(X_noise.index).iloc[n_split:]
noise_ic = np.corrcoef(noise_edge, noise_forward)[0, 1]

noise_test_df = noise_df.loc[X_noise.index[n_split] :]
noise_edge_series = model_edge(noise_model, X_noise)
noise_gross = run_backtest(noise_test_df, ModelSignal(noise_edge_series, min_edge=0.15), gross_cfg)

print(f"{'':32} {'test acc':>10} {'IC':>9} {'gross Sharpe':>14}")
print(f"  {'momentum.csv (real edge)':30} {test_acc:>10.2%} {ic:>+9.4f} "
      f"{summary.loc['confidence + dead zone 0.15', 'gross Sharpe']:>+14.2f}")
print(f"  {'pure_noise.csv (NO edge)':30} "
      f"{accuracy_score(y_noise.iloc[n_split:], noise_model.predict(X_noise.iloc[n_split:])):>10.2%} "
      f"{noise_ic:>+9.4f} {M.sharpe_ratio(noise_gross.returns, BPY):>+14.2f}")

print("\nThe noise row is your null hypothesis, measured rather than assumed.")

# %% [markdown]
# **Read that table carefully, because it is more uncomfortable than it looks.**
#
# On data built with no predictability whatsoever, the model achieved a *gross
# Sharpe ratio of about 1.8* over the test period. Not far below the real
# dataset's 2.3.
#
# The metric that does separate them is the information coefficient: **+0.033**
# on the real edge versus **negative** on noise. The noise model's confidence is
# *anti*-correlated with the future, and it made money anyway, over this
# particular sample, by luck.
#
# Two conclusions, both important:
#
# 1. **A gross Sharpe ratio on one test period is not evidence.** One split is
#    one sample. This is exactly why notebook 09 exists.
# 2. **Prefer metrics that measure the relationship, not the outcome.** The IC
#    told the truth here and the equity curve did not.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 8.1 — The accuracy trap, constructed
#
# Build two synthetic signals on the test set using `labels.meta["fwd_return"]`:
#
# - **A:** correct 56% of the time, but systematically wrong on the largest moves.
# - **B:** correct 52% of the time, but correct on the largest moves.
#
# Backtest both with zero costs. Confirm A loses and B wins. Then write one
# sentence explaining why reporting accuracy would have selected the wrong model.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 8.2 — The dead zone, and a trap
#
# Sweep `min_edge` from 0 to 0.4 and plot net Sharpe against it. Find the
# maximum.
#
# Now answer honestly: **you just chose a parameter by looking at test-set
# performance.** What have you actually measured? What would you have to do to
# choose `min_edge` legitimately? (Notebook 09 is the answer, so write your own
# first.)

# %%
# Your code here.

# %% [markdown]
# ### Exercise 8.3 — Does a bigger model help?
#
# Compare logistic regression, random forest and gradient boosting. For each
# report: test accuracy, IC, **gross** Sharpe, **net** Sharpe, turnover.
#
# Rank by accuracy, then by net Sharpe. Are the rankings the same? Does the more
# flexible model produce a bigger IC, and if so does any of it survive costs?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 8.4 — Make it profitable, or prove it cannot be
#
# Given the break-even cost you computed in §7, try to construct a *net*
# profitable version. Legitimate levers: a longer holding period, a wider dead
# zone, a rebalance tolerance, position sizing.
#
# If you succeed, state what you changed and check it is not just noise. If you
# fail, write the three-sentence honest conclusion — that is a completely
# acceptable result, and being able to write it is the skill.

# %%
# Your code here.

# %% [markdown]
# ---
# ## What you should take away
#
# - **The ML can work and the strategy still fail.** Gross Sharpe +3.2, net
#   Sharpe −9.7, same model. Report both or you have reported nothing.
# - Accuracy is a poor trading metric; the information coefficient and the
#   confidence-quintile table tell you far more.
# - Choose the prediction horizon from reasoning or training data. Choosing it
#   from test performance is data snooping wearing a lab coat.
# - **Break-even cost is the number that decides everything.** Professional firms
#   trade signals this weak because their cost structure is different from yours,
#   not because their signals are better.
#
# Next: **09 · Walk-forward validation** — because one chronological split is a
# single sample, and we have been treating it as evidence.
