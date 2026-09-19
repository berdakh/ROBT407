# %% [markdown]
# # 13 · Capstone — your own strategy, evaluated honestly
#
# **Day 4 · the rest of the day, plus your own time**
#
# > ### ⚠️ Before you start, read this
# >
# > **This is a methodology exercise, not an investment exercise.** You are being
# > assessed on the quality of your *evaluation*, not on the performance of your
# > strategy.
# >
# > **A capstone that concludes "I found no evidence of an edge" and demonstrates
# > that rigorously will score higher than one reporting a spectacular backtest
# > with an unexamined methodology.** That is not a consolation prize. Finding
# > nothing, and proving you looked properly, is the normal and correct outcome
# > of most honest quantitative research.
# >
# > Paper trading only. Do not connect this to a live-money account. Nothing in
# > this workshop is investment advice, and most retail algorithmic strategies
# > lose money after costs.

# %% [markdown]
# ## What to produce
#
# One notebook, based on this template, containing:
#
# 1. **A hypothesis** — a sentence about market behaviour, and *why* it might
#    hold. "Moving averages cross" is a description of an indicator, not a
#    hypothesis. "Short-horizon trends persist because information diffuses
#    slowly" is a hypothesis.
# 2. **An implementation** as an `algotrade.Strategy` subclass.
# 3. **A walk-forward evaluation** with purging and embargo.
# 4. **A control experiment** on a no-edge dataset.
# 5. **A report card** with every warning addressed.
# 6. **A paper-trading run** with reconciliation against the backtest.
# 7. **An honest assessment**, including why it might fail with real money.
#
# ### How it is assessed
#
# | weight | criterion |
# |---|---|
# | 30% | **Methodological rigour.** No leakage, correct validation, honest trial count. |
# | 25% | **The honest assessment.** Specific, falsifiable, not hedged. |
# | 20% | **Controls and benchmarks.** Compared against noise, random strategies and buy-and-hold. |
# | 15% | **Code quality.** Readable, uses the library, reproducible. |
# | 10% | **Hypothesis quality.** Is there a reason this should work? |
# | 0% | **Strategy performance.** Deliberately zero. |

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from algotrade import BacktestConfig, RETAIL_CRYPTO, ZERO_COST, run_backtest
from algotrade import metrics as M
from algotrade.features import make_features
from algotrade.labeling import fixed_horizon_class, triple_barrier
from algotrade.paper import PaperTradingSession
from algotrade.report import compare, report_card
from algotrade.risk import DrawdownGuard, VolatilityTargeted
from algotrade.strategy import BuyAndHold, RandomStrategy, Strategy
from algotrade.validation import assert_no_leakage, describe_splits, walk_forward_splits

BPY = algotrade.BARS_PER_YEAR["1h"]

# Track this honestly from the very first cell. Every parameter you try, every
# idea you abandon, every dataset you switch away from. It goes into the
# deflated Sharpe at the end and it is the number most people quietly fudge.
N_TRIALS = 0

# %% [markdown]
# ---
# ## 1. Hypothesis
#
# > **Replace this quote block with your own.**
# >
# > **Hypothesis:** [one sentence about market behaviour]
# >
# > **Why it might be true:** [the mechanism — who is on the other side of your
# > trade, and why are they willing to lose? "The backtest looked good" is not a
# > mechanism.]
# >
# > **Why it might be false:** [the most likely reason you are wrong]
# >
# > **What would falsify it:** [a specific, numeric result that would make you
# > abandon this]
#
# Write these before you look at any results. Then do not edit them afterwards —
# if your hypothesis changes, say so explicitly and increment `N_TRIALS`.

# %% [markdown]
# ## 2. Data
#
# Choose your dataset and state what is wrong with it. Every dataset has
# something wrong with it.

# %%
DATASET = "momentum"          # try: momentum, mean_reverting, regime_switching, trending_with_crash
CONTROL = "pure_noise"        # a dataset with NO edge, by construction

df = algotrade.load_ohlcv(f"data/synthetic/{DATASET}.csv")
control_df = algotrade.load_ohlcv(f"data/synthetic/{CONTROL}.csv")

quality = algotrade.data.data_quality_report(df, "1h")
print(f"{DATASET}: {quality['rows']:,} bars, {quality['start'].date()} to {quality['end'].date()}")
print(f"gaps: {quality.get('gap_count', 0)}, duplicates: {quality['duplicated_timestamps']}")

# %% [markdown]
# > **Limitations of this data — replace with your own:**
# >
# > - It is **synthetic**. It has no fat tails beyond what the generator makes,
# >   no liquidity effects, no exchange outages, and no market microstructure.
# >   It is an *easier* problem than a real market.
# > - It is a single asset, so there is no survivorship bias — and no
# >   cross-sectional information either.
# > - [what else?]

# %% [markdown]
# ## 3. Your strategy
#
# Implement it as a `Strategy` subclass. It must use only the `view`, so the
# look-ahead guard applies. If you precompute anything, re-read the warning in
# notebook 08 §4 — that is the one place the guard cannot protect you.

# %%
class MyStrategy(Strategy):
    """REPLACE THIS with your own strategy.

    This placeholder is a short-horizon trend follower with a neutral band.
    It is here so the template runs end to end; it is not a suggestion.
    """

    def __init__(self, lookback: int = 24, threshold: float = 0.002):
        self.lookback = lookback
        self.threshold = threshold
        self.warmup = lookback + 1
        self.name = f"my strategy (lookback={lookback})"

    def on_bar(self, view, account):
        if len(view) < self.warmup:
            return None
        closes = view.close.history(self.lookback + 1)
        trailing_return = closes[-1] / closes[0] - 1.0
        if trailing_return > self.threshold:
            return 1.0
        if trailing_return < -self.threshold:
            return -1.0
        return 0.0


N_TRIALS += 1
strategy_factory = lambda: MyStrategy()          # noqa: E731

# %% [markdown]
# ## 4. First look — and do not stop here
#
# A single in-sample backtest. This is a **sanity check**, not a result. If the
# code runs and the equity curve is not obviously broken, move on quickly.

# %%
cfg = BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY)
first_look = run_backtest(df, strategy_factory(), cfg)
print(first_look)
print(f"\nSharpe {M.sharpe_ratio(first_look.returns, BPY):+.2f}, "
      f"max DD {M.max_drawdown(first_look.equity):.1%}, "
      f"turnover {M.turnover(first_look.weights, BPY):,.0f}x/yr")

# %% [markdown]
# ### The cost budget, before anything else
#
# Turnover × round-trip ÷ 2. If your edge cannot clear this, stop and change the
# strategy rather than tuning it.

# %%
turnover = M.turnover(first_look.weights, BPY)
drag = turnover * RETAIL_CRYPTO.round_trip_bps() / 2 / 10_000
print(f"turnover      {turnover:>10,.0f}x/yr")
print(f"annual drag   {drag:>10.1%}")
print(f"\nYour gross edge must exceed {drag:.1%} a year before you keep anything.")

# %% [markdown]
# ## 5. Walk-forward evaluation
#
# The only result that counts. Purge by at least your label horizon; embargo to
# break serial correlation.

# %%
TRAIN, TEST, PURGE, EMBARGO = 3000, 500, 24, 24

splits = walk_forward_splits(len(df), train_size=TRAIN, test_size=TEST,
                             purge=PURGE, embargo=EMBARGO)
assert_no_leakage(splits, label_horizon=PURGE)
print(f"{len(splits)} folds, leakage check passed\n")
print(describe_splits(splits, df.index).head(4).to_string(index=False))

# %%
# For a rule-based strategy with no fitting, each fold is simply a backtest over
# that fold's test window. If YOUR strategy fits anything -- a model, a
# threshold, a parameter -- fit it inside the loop on sp.train only.
fold_rows = []
oos_pieces = []

for sp in splits:
    window = df.iloc[sp.test.min() : sp.test.max() + 1]
    if len(window) < 2:
        continue
    res = run_backtest(window, strategy_factory(), cfg)
    fold_rows.append({
        "fold": sp.fold,
        "start": window.index[0].date(),
        "sharpe": M.sharpe_ratio(res.returns, BPY),
        "return": M.total_return(res.equity),
        "max_dd": M.max_drawdown(res.equity),
        "trades": res.n_trades,
    })
    oos_pieces.append(res.returns)

folds = pd.DataFrame(fold_rows)
print(folds.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

sharpes = folds["sharpe"]
t_stat = sharpes.mean() / (sharpes.std() / np.sqrt(len(sharpes))) if sharpes.std() > 0 else np.nan
print(f"\nmean fold Sharpe   {sharpes.mean():+.3f}")
print(f"dispersion (sd)    {sharpes.std():.3f}")
print(f"positive folds     {int((sharpes > 0).sum())} of {len(sharpes)}")
print(f"t-statistic        {t_stat:+.2f}  "
      f"({'significant' if abs(t_stat) > 2 else 'NOT significant'} at ~5%)")

# %% [markdown]
# ## 6. The control experiment
#
# Run the **identical procedure** on a dataset with no edge. Whatever it reports
# there is your noise floor, and your real result must clear it convincingly.

# %%
control_rows = []
for sp in splits:
    if sp.test.max() + 1 > len(control_df):
        continue
    window = control_df.iloc[sp.test.min() : sp.test.max() + 1]
    if len(window) < 2:
        continue
    res = run_backtest(window, strategy_factory(), cfg)
    control_rows.append(M.sharpe_ratio(res.returns, BPY))

control_sharpes = pd.Series(control_rows)
print(f"{'':16} {'mean Sharpe':>13} {'positive':>10} {'t-stat':>9}")
for label, series in [(DATASET, sharpes), (f"{CONTROL} (no edge)", control_sharpes)]:
    t = series.mean() / (series.std() / np.sqrt(len(series))) if series.std() > 0 else np.nan
    print(f"  {label:14} {series.mean():>+13.3f} "
          f"{int((series > 0).sum()):>4} / {len(series):<3} {t:>+9.2f}")

# %% [markdown]
# ## 7. Benchmarks
#
# Buy-and-hold, and a set of random strategies as a control group. If your
# strategy does not sit clearly above the random ones, say so.

# %%
benchmarks = {"my strategy": run_backtest(df, strategy_factory(), cfg),
              "buy & hold": run_backtest(df, BuyAndHold(), cfg)}
for seed in range(8):
    benchmarks[f"random #{seed}"] = run_backtest(df, RandomStrategy(seed=seed, every=48), cfg)

table = compare(benchmarks, bars_per_year=BPY)
print(table.to_string(float_format=lambda v: f"{v:,.3f}"))

rank = list(table.index).index("my strategy") + 1
print(f"\nmy strategy ranks {rank} of {len(table)} by Sharpe ratio.")
random_sharpes = table.loc[[i for i in table.index if i.startswith("random")], "sharpe"]
print(f"random strategies: median {random_sharpes.median():+.2f}, best {random_sharpes.max():+.2f}")

# %%
fig, ax = plt.subplots(figsize=(11, 4.2))
for name, res in benchmarks.items():
    is_random = name.startswith("random")
    ax.plot(res.equity.index, res.equity / res.equity.iloc[0],
            linewidth=0.8 if is_random else 1.7,
            alpha=0.4 if is_random else 1.0,
            color="#9198a1" if is_random else None,
            label=None if is_random else name)
ax.plot([], [], color="#9198a1", linewidth=0.8, label="random strategies (8)")
ax.axhline(1.0, color="#57606a", linewidth=0.8, linestyle=":")
ax.set_ylabel("growth of 1")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Your strategy against its controls", loc="left", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Risk sizing
#
# Apply volatility targeting and a drawdown limit. Report what each one costs as
# well as what it saves.

# %%
sized = {
    "raw": run_backtest(df, strategy_factory(), cfg),
    "vol-targeted 20%": run_backtest(
        df, VolatilityTargeted(strategy_factory(), target_vol=0.20, bars_per_year=BPY), cfg),
    "+ drawdown guard 25%": run_backtest(
        df, DrawdownGuard(VolatilityTargeted(strategy_factory(), target_vol=0.20, bars_per_year=BPY),
                          max_drawdown=0.25), cfg),
}
print(compare(sized, bars_per_year=BPY).to_string(float_format=lambda v: f"{v:,.3f}"))

# %% [markdown]
# ## 9. Report card
#
# Every warning must be addressed — fixed, or explained in one sentence.

# %%
card = report_card(
    benchmarks["my strategy"],
    benchmark=benchmarks["buy & hold"],
    n_trials=max(N_TRIALS, 1),
    is_out_of_sample=False,
)
print(card.summary())

# %% [markdown]
# > **Address every warning here.** For each one, write either the fix you made
# > or why it does not apply:
# >
# > 1. [warning] → [fix or justification]
# > 2. …

# %% [markdown]
# ## 10. Paper trading
#
# Run it forward bar by bar and reconcile against the backtest. A mismatch is a
# bug you want to find now.

# %%
WARMUP = 3000
session = PaperTradingSession(strategy_factory(), history=df.iloc[:WARMUP],
                              initial_cash=10_000.0, cost_model=RETAIL_CRYPTO)
for timestamp, bar in df.iloc[WARMUP:].iterrows():
    session.on_new_bar(timestamp, bar)

print(session.summary())

recon = session.reconcile(run_backtest(df, strategy_factory(), cfg).targets)
print(f"\nreconciliation: {len(recon):,} bars compared, "
      f"{int(recon['mismatch'].sum())} mismatches")

# %% [markdown]
# ## 11. The honest assessment
#
# This section is worth more than all the code above it. Fill in every field.
# Vague answers score zero.

# %%
res = benchmarks["my strategy"]
bench = benchmarks["buy & hold"]
sr, se = M.sharpe_ratio(res.returns, BPY), M.sharpe_standard_error(res.returns, BPY)

print(f"""
=========================================================================
CAPSTONE ASSESSMENT
=========================================================================

HYPOTHESIS
  [your hypothesis, unchanged from section 1]

DATA
  {DATASET}.csv, {len(df):,} hourly bars, {df.index[0].date()} to {df.index[-1].date()}.
  SYNTHETIC -- generated, not a real market.
  Known limitations: [yours]

METHOD
  Walk-forward, {len(splits)} folds, train={TRAIN}, test={TEST},
  purge={PURGE}, embargo={EMBARGO}. Costs: {RETAIL_CRYPTO.round_trip_bps():.0f} bps round trip.
  Fills at next bar's open.

TRIALS
  {max(N_TRIALS, 1)} (honest count, including abandoned ideas)

RESULT (walk-forward, after costs)
  mean fold Sharpe  {sharpes.mean():+.3f}   dispersion {sharpes.std():.3f}
  positive in       {int((sharpes > 0).sum())} of {len(sharpes)} folds   t = {t_stat:+.2f}
  full-sample Sharpe{sr:+.2f}  (95% CI [{sr - 1.96 * se:+.2f}, {sr + 1.96 * se:+.2f}])
  deflated Sharpe   {M.deflated_sharpe_ratio(res.returns, BPY, max(N_TRIALS, 2)):.1%}
  max drawdown      {M.max_drawdown(res.equity):+.1%}
  turnover          {M.turnover(res.weights, BPY):,.0f}x/yr -> {drag:.1%} annual drag

CONTROL ({CONTROL}, no edge exists)
  mean fold Sharpe  {control_sharpes.mean():+.3f}   positive in {int((control_sharpes > 0).sum())} of {len(control_sharpes)}

BENCHMARKS
  buy & hold        Sharpe {M.sharpe_ratio(bench.returns, BPY):+.2f}, {M.total_return(bench.equity):+.1%}
  random (median)   Sharpe {random_sharpes.median():+.2f}
  my rank           {rank} of {len(table)}

CONCLUSION
  [ONE sentence. "No evidence of an edge" is a complete and good answer.]

WHY THIS MIGHT FAIL WITH REAL MONEY
  1. Synthetic data has no fat tails, no outages, no liquidity limits.
  2. Costs modelled at {RETAIL_CRYPTO.round_trip_bps():.0f} bps; real costs vary and can be far worse.
  3. No market impact modelled; size would make this worse.
  4. The regime in this sample may not repeat.
  5. [your own -- the most likely specific failure]

WHAT WOULD CHANGE MY MIND
  [a specific, numeric result that would make you abandon or adopt this]

=========================================================================
PAPER TRADING ONLY. NOT INVESTMENT ADVICE.
Most retail algorithmic strategies lose money after costs.
=========================================================================
""")

# %% [markdown]
# ---
# ## Submission checklist
#
# - [ ] Hypothesis states a **mechanism**, not an indicator.
# - [ ] Strategy uses only `view`; no precomputed future.
# - [ ] Walk-forward with purge ≥ label horizon; `assert_no_leakage` passes.
# - [ ] Control experiment on a no-edge dataset, reported alongside.
# - [ ] Compared against buy-and-hold **and** random strategies.
# - [ ] `N_TRIALS` honestly counted; deflated Sharpe reported.
# - [ ] Every report-card warning addressed.
# - [ ] Paper-trading run reconciles against the backtest.
# - [ ] Honest assessment complete, including "what would change my mind".
# - [ ] Notebook runs top to bottom from a clean kernel.
# - [ ] No live-money connection anywhere.
#
# ---
#
# ### A closing note
#
# You began this workshop by building a strategy that appeared to return
# **+46,986%**, and you watched it become **+16%** — losing comfortably to doing
# nothing at all.
#
# The useful skill you have acquired is not building strategies. It is the
# ability to look at a beautiful equity curve, including your own, and ask the
# four questions that dismantle it.
#
# Most honest quantitative research finds nothing. If your capstone finds nothing
# and proves it carefully, you have done the work correctly.
