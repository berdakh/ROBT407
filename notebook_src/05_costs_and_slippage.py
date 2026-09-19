# %% [markdown]
# # 05 · Costs and slippage — where the rest of the profit goes
#
# **Day 2 · ~2 hours. This is the notebook the workshop is built around.**
#
# Day 1 produced **+46,986%**. Notebook 04 fixed the timing and it fell to
# **+406%** — still a number most people would be pleased with.
#
# Today we charge for the trades. It becomes **+16%** — while buy-and-hold,
# which required no code at all, returned **+582%**.
#
# By the end of this notebook you will be able to:
#
# 1. Name the four components of trading cost and say roughly what each is worth.
# 2. Compute the annual cost drag implied by a strategy's turnover.
# 3. Decide, before running a backtest, whether a strategy can possibly survive.
# 4. Read a report card and take its warnings seriously.
#
# ![the order lifecycle](../diagrams/svg/04-order-lifecycle.svg)

# %% [markdown]
# ## 1. What a trade actually costs
#
# You do not buy at the price on the chart. You buy at a worse one, and then you
# pay a fee on top.
#
# | component | typical retail crypto | what it is |
# |---|---|---|
# | **commission** | 10 bps | the exchange's taker fee, on notional, every trade |
# | **half-spread** | 1 bp | you buy at the ask, sell at the bid; the chart shows the mid |
# | **latency** | ~1 bp | the price moves between your decision and your fill |
# | **market impact** | ~0 bps while small | your own order pushes the price against you |
#
# A **round trip** — buy then sell — pays all of these twice. Call it **24 basis
# points**, or 0.24% of the traded notional.
#
# That number sounds small. It is not, and the reason is turnover.

# %%
from algotrade import BacktestConfig, CostModel, RETAIL_CRYPTO, ZERO_COST, run_backtest
from algotrade.strategy import BuyAndHold, Strategy

df = algotrade.load_ohlcv("data/synthetic/trending_with_crash.csv")
BPY = algotrade.BARS_PER_YEAR["1h"]

print(ZERO_COST.describe())
print(RETAIL_CRYPTO.describe())

# %%
# What a single trade looks like under each model.
reference_price = 20_000.0
quantity = 0.5

for model in (ZERO_COST, RETAIL_CRYPTO):
    buy = model.execute(reference_price, quantity, bar_volume=100.0, timestamp="now")
    sell = model.execute(reference_price, -quantity, bar_volume=100.0, timestamp="now")
    print(f"\n{model.name}")
    print(f"  chart price          {reference_price:>12,.2f}")
    print(f"  you buy at           {buy.price:>12,.2f}   (+{buy.price - reference_price:,.2f})")
    print(f"  you sell at          {sell.price:>12,.2f}   ({sell.price - reference_price:,.2f})")
    print(f"  commission each way  {buy.commission:>12,.2f}")
    print(f"  round-trip cost      {buy.total_cost + sell.total_cost:>12,.2f}  "
          f"on {reference_price * quantity:,.0f} of notional")

# %% [markdown]
# ## 2. The arithmetic you should do *before* the backtest
#
# **Annual cost drag ≈ turnover × round-trip cost ÷ 2**
#
# Turnover is how many times a year you replace your whole position. A strategy
# that flips between long and flat 1,295 times over two years has a turnover of
# roughly 650×/year.

# %%
round_trip_bps = RETAIL_CRYPTO.round_trip_bps()
print(f"round trip = {round_trip_bps:.0f} bps\n")
print(f"{'turnover/yr':>12} {'annual drag':>13}   {'you must earn more than this'}")
for turnover in (2, 12, 52, 200, 650, 2000):
    drag = turnover * round_trip_bps / 2 / 10_000
    print(f"{turnover:>12,} {drag:>12.1%}   {'-' * int(min(drag * 60, 60))}")

# %% [markdown]
# Read the bottom rows. At 650 turns a year you are paying roughly **78% of your
# capital annually** in transaction costs. Your strategy does not need a small
# edge to overcome that. It needs an enormous one.
#
# **This calculation takes thirty seconds and would kill most strategy ideas
# before anyone writes a backtest.** Do it first, always.

# %% [markdown]
# ## 3. The collapse
#
# Same strategy. Same data. Same engine. We change only the cost model.

# %%
class MovingAverageCross(Strategy):
    def __init__(self, fast: int = 3, slow: int = 24):
        self.fast, self.slow = fast, slow
        self.warmup = slow
        self.name = f"SMA {fast}/{slow}"

    def on_bar(self, view, account):
        if len(view) < self.warmup:
            return None
        closes = view.close.history(self.slow)
        return 1.0 if closes[-self.fast:].mean() > closes.mean() else 0.0


runs = {}
for label, cost_model in [
    ("no costs (fiction)", ZERO_COST),
    ("commission only (10bps)", CostModel(commission_bps=10.0, half_spread_bps=0.0, name="fee-only")),
    ("realistic retail (24bps round trip)", RETAIL_CRYPTO),
]:
    runs[label] = run_backtest(df, MovingAverageCross(3, 24),
                               BacktestConfig(cost_model=cost_model, bars_per_year=BPY))

runs["buy & hold (24bps)"] = run_backtest(df, BuyAndHold(),
                                          BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY))

print(f"{'':38} {'total return':>14} {'trades':>8} {'costs paid':>12}")
for label, res in runs.items():
    print(f"  {label:36} {algotrade.metrics.total_return(res.equity):>13,.1%} "
          f"{res.n_trades:>8,} {res.total_costs:>12,.0f}")

# %%
fig, ax = plt.subplots(figsize=(11, 5))
colors = {"no costs (fiction)": "#cf222e", "commission only (10bps)": "#d29922",
          "realistic retail (24bps round trip)": "#8250df", "buy & hold (24bps)": "#1a7f37"}
for label, res in runs.items():
    normalised = res.equity / res.equity.iloc[0]
    ax.plot(normalised.index, normalised, linewidth=1.4, label=
            f"{label}  ({algotrade.metrics.total_return(res.equity):+,.0%})", color=colors[label])
ax.axhline(1.0, color="#57606a", linewidth=0.8, linestyle=":")
ax.set_yscale("log")
ax.set_ylabel("growth of 1 (log scale)")
ax.legend(loc="upper left", fontsize=9)
ax.set_title("The only thing that changed is what the trades cost", loc="left", fontsize=11)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. The whole arc, in one table
#
# This is the spine of the workshop. One strategy, four evaluations.

# %%
from algotrade.indicators import sma
from algotrade.naive import vectorized_backtest

signal = (sma(df["close"], 3) > sma(df["close"], 24)).astype(float)

# NOTE the 12, not 24. vectorized_backtest charges cost_bps on each *change* in
# position, so a round trip pays it twice. 12 bps per side == 24 bps round trip,
# which is what RETAIL_CRYPTO charges. Passing 24 here would double the costs
# and produce a far more dramatic -- and wrong -- collapse.
stages = [
    ("Day 1: vectorised, shift=0, no costs",
     vectorized_backtest(df["close"], signal, shift=0, cost_bps=0.0, quiet=True)["equity"].iloc[-1] - 1,
     "look-ahead bias + free trading"),
    ("+ correct the indexing (shift=1)",
     vectorized_backtest(df["close"], signal, shift=1, cost_bps=0.0, quiet=True)["equity"].iloc[-1] - 1,
     "look-ahead removed"),
    ("+ 24 bps round trip (12 per side)",
     vectorized_backtest(df["close"], signal, shift=1, cost_bps=12.0, quiet=True)["equity"].iloc[-1] - 1,
     "realistic retail costs"),
    ("the same, through the event-driven engine",
     algotrade.metrics.total_return(runs["realistic retail (24bps round trip)"].equity),
     "independent implementation, same answer"),
    ("doing absolutely nothing (buy & hold)",
     algotrade.metrics.total_return(runs["buy & hold (24bps)"].equity),
     "the benchmark it must beat"),
]

print(f"{'':46} {'total return':>14}")
print("  " + "-" * 76)
for label, value, note in stages:
    print(f"  {label:44} {value:>14,.1%}   {note}")

# %% [markdown]
# > **+46,986% \u2192 +406% \u2192 +16%.** And buy-and-hold, which required no code
# > at all, returned **+582%**.
#
# Roughly 99% of the apparent profit was look-ahead bias. Transaction costs then
# consumed 96% of what remained. The strategy is not catastrophic \u2014 it ends
# slightly up \u2014 but it underperforms doing nothing by more than 560
# percentage points, while requiring 1,592 trades and constant attention.
#
# Note that the vectorised and event-driven numbers agree to within a couple of
# percentage points. They are independent implementations with different fill
# assumptions (next close vs next open), so exact agreement would be suspicious;
# close agreement is the check that neither has a gross error in it.

# %% [markdown]
# ### How sensitive is this to the cost assumption?
#
# 24 bps round trip is a *liquid-pair, major-exchange* number. On a smaller
# exchange, a thinner pair, or with worse execution, 48 bps is entirely ordinary.
# The strategy's fate depends heavily on which world you are in \u2014 and that is
# itself the finding.

# %%
print(f"{'round trip':>12} {'total return':>14}")
for per_side in (0, 3, 6, 12, 24, 48):
    bt = vectorized_backtest(df["close"], signal, shift=1, cost_bps=per_side, quiet=True)
    print(f"{2 * per_side:>10} bps {bt['equity'].iloc[-1] - 1:>14,.1%}")

print("\nA strategy whose sign flips between 24 and 48 bps of cost is not a")
print("strategy. It is a bet on your broker.")

# %% [markdown]
# ## 5. The report card
#
# Every strategy in this workshop goes through the same harness, which computes
# the same metrics and — more importantly — attaches the same warnings.

# %%
from algotrade.report import report_card

card = report_card(
    runs["realistic retail (24bps round trip)"],
    benchmark=runs["buy & hold (24bps)"],
    n_trials=22,              # be honest: we searched 22 combinations on Day 1
    is_out_of_sample=False,   # and evaluated on the same data we searched
)
print(card.summary())

# %% [markdown]
# The warnings are the valuable part. A backtest with a Sharpe of 3 and six
# warnings is worth less than one with a Sharpe of 0.4 and none.
#
# Note that the harness flags things you did not tell it: that costs consumed
# the profit, that turnover implies a large drag, that the result underperforms
# buy-and-hold. It cannot detect everything — it takes your word for `n_trials`
# and `is_out_of_sample` — which is why those two parameters require you to be
# honest with yourself.

# %% [markdown]
# ## 6. Can *any* version of this strategy survive?
#
# Rather than tuning parameters, ask the structural question: what would the
# costs have to be for this to work? Or equivalently — how slowly would it have
# to trade?

# %%
survival = []
for fast, slow in [(3, 24), (6, 48), (12, 96), (24, 192), (48, 384), (96, 768)]:
    res = run_backtest(df, MovingAverageCross(fast, slow),
                       BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY))
    free = run_backtest(df, MovingAverageCross(fast, slow),
                        BacktestConfig(cost_model=ZERO_COST, bars_per_year=BPY))
    survival.append({
        "params": f"{fast}/{slow}",
        "trades": res.n_trades,
        "turnover/yr": algotrade.metrics.turnover(res.weights, BPY),
        "gross (no costs)": algotrade.metrics.total_return(free.equity),
        "net (24bps)": algotrade.metrics.total_return(res.equity),
        "cost drag": algotrade.metrics.total_return(free.equity) - algotrade.metrics.total_return(res.equity),
    })

table = pd.DataFrame(survival)
print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

# %% [markdown]
# Trading more slowly reduces the drag, as the arithmetic predicts. Whether it
# produces a *profitable* strategy is a different question, and on this dataset
# the honest answer is visible in the table.
#
# Notice also that the fastest variant — the one Day 1's grid search selected as
# the winner — is the one the costs destroy most completely. **The parameter
# search actively selected for the strategy that could least afford to trade**,
# because with costs switched off, trading more was free.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 5.1 — Your strategy's cost budget
#
# Take the strategy from Exercise 3.2. Before running anything:
#
# 1. Count how many times its signal changes.
# 2. Convert that to annual turnover.
# 3. Compute the implied annual cost drag at 24 bps.
# 4. **Write down your predicted net return.**
#
# Then run it through the engine with `RETAIL_CRYPTO` and compare. How close was
# your prediction?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 5.2 — The break-even fee
#
# For the `3/24` strategy, find the commission (in bps) at which it exactly breaks
# even. Use a search over `CostModel(commission_bps=x, half_spread_bps=0)`.
#
# Then look up what a real exchange actually charges a retail taker. Is the
# break-even fee above or below it, and by how much?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 5.3 — Rebalance tolerance
#
# `BacktestConfig` has a `rebalance_tolerance` parameter: skip a rebalance when
# the weight is within that tolerance of the target. Try values of 0, 0.02, 0.05,
# 0.2 on a strategy that returns a continuous weight rather than 0/1.
#
# Plot turnover against net return. Is there an optimum? What does its existence
# tell you about the trade-off between tracking your signal and paying to do so?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 5.4 — Impact
#
# `CostModel` has an `impact_coef` that models square-root market impact:
# cost grows with the square root of your order size relative to the bar's volume.
#
# Run the same strategy with `initial_cash` of 10,000 / 1,000,000 / 100,000,000
# and `impact_coef=0.3`. What happens, and why does size hurt?
#
# Then answer: what does this imply about backtests published by people who have
# never traded size?

# %%
# Your code here.

# %% [markdown]
# ### Exercise 5.5 — The honest write-up
#
# Write three sentences you would be willing to put your name to, describing what
# the SMA 3/24 strategy does on this dataset. No hedging, no "further research
# is needed", no quoting the gross number.

# %%
# Your three sentences here, as a comment.

# %% [markdown]
# ---
# ## What you should take away
#
# - Round-trip cost × turnover ÷ 2 = annual drag. Compute it **before** you
#   backtest; it kills most ideas in thirty seconds.
# - Costs do not reduce a good strategy to a slightly worse one. They frequently
#   invert the sign.
# - A parameter search run without costs actively selects for strategies that
#   trade too much to survive.
# - Buy-and-hold is a real benchmark and it is harder to beat than it looks.
#
# Next: **06 · The bias catalogue** — the other ways your backtest lies, each one
# measured rather than described.
