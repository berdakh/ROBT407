# %% [markdown]
# # 12 · Paper trading — what the backtest forgot
#
# **Day 4 · ~2 hours**
#
# > ### This workshop is paper-trading only, by design.
# >
# > There is no live order path in this package. `PaperBroker.connect()` raises
# > `NotImplementedError` and always will. No credentials are handled anywhere,
# > and no exchange client is imported. You are not one uncommented line away
# > from sending money to an exchange, and that is deliberate.
#
# **What paper trading is for:** finding out whether your *system* works. Does
# the strategy still run when a bar arrives late, twice, or not at all? Does it
# survive a restart? Does the live signal match what the backtest said it would
# be on the same bar?
#
# **What paper trading is not for:** finding out whether the strategy is
# profitable. A few weeks of paper trading has nowhere near the statistical power
# to answer that — as notebook 02's power table showed, you need *years*.
#
# By the end you will be able to:
#
# 1. Run a strategy through a bar-by-bar live loop.
# 2. Reconcile live signals against the backtest and find discrepancies.
# 3. List the failure modes that only appear in live operation.
# 4. Write an honest assessment of why a strategy might fail with real money.

# %%
from algotrade import BacktestConfig, RETAIL_CRYPTO, run_backtest
from algotrade import metrics as M
from algotrade.paper import PaperBroker, PaperTradingSession
from algotrade.risk import VolatilityTargeted
from algotrade.strategy import MomentumStrategy, SMACrossover

BPY = algotrade.BARS_PER_YEAR["1h"]
df = algotrade.load_ohlcv("data/synthetic/momentum.csv")

print(algotrade.DISCLAIMER)

# %%
# The guard rail, demonstrated rather than promised.
try:
    PaperBroker().connect("https://api.some-exchange.com", api_key="...")
except NotImplementedError as exc:
    print("PaperBroker.connect() ->")
    print(f"  {exc}")

# %% [markdown]
# ## 1. The shape of a live loop
#
# The arithmetic is identical to the backtester. The **shape** is not:
#
# | | backtest | live |
# |---|---|---|
# | data | a complete array | one bar at a time, arriving |
# | state | rebuilt from scratch each run | must survive between bars, and restarts |
# | the future | exists in memory, guarded against | does not exist at all |
# | a bug | wrong number in a notebook | wrong position, in the market |
#
# Running your strategy this way is how you discover that it quietly depended on
# having the whole array.

# %%
WARMUP = 2000
strategy = VolatilityTargeted(MomentumStrategy(48), target_vol=0.20,
                              max_leverage=1.0, bars_per_year=BPY)

session = PaperTradingSession(
    strategy,
    history=df.iloc[:WARMUP],        # seed the warm-up so it can act immediately
    initial_cash=10_000.0,
    cost_model=RETAIL_CRYPTO,
)

print(f"seeded with {WARMUP:,} bars of history")
print(f"now streaming {len(df) - WARMUP:,} bars, one at a time\n")

for timestamp, bar in df.iloc[WARMUP:].iterrows():
    session.on_new_bar(timestamp, bar)

print(session.summary())

# %%
log = session.frame()
print(log.head(3).to_string())
print(f"\n...{len(log):,} rows total, {int((log['action'] != 'hold').sum())} fills")

# %% [markdown]
# ## 2. Reconciliation — the check that catches real bugs
#
# Run the **same strategy** over the **same bars** through the backtester, and
# compare the target weights bar by bar. Any mismatch is a bug in one of the two
# paths, and you want to find it now rather than after a month of live
# divergence.
#
# Compare **targets to targets**, not targets to achieved weights. The weight you
# end up holding differs from the weight you asked for, for entirely legitimate
# reasons — you fill at the next bar's open, not at the decision price — so
# comparing those would make every row look like a mismatch.

# %%
backtest = run_backtest(
    df,
    VolatilityTargeted(MomentumStrategy(48), target_vol=0.20, max_leverage=1.0, bars_per_year=BPY),
    BacktestConfig(cost_model=RETAIL_CRYPTO, bars_per_year=BPY),
)

recon = session.reconcile(backtest.targets)
mismatches = int(recon["mismatch"].sum())

print(f"bars compared : {len(recon):,}")
print(f"mismatches    : {mismatches}")
print(f"max difference: {recon['difference'].abs().max():.2e}")

if mismatches == 0:
    print("\nThe live loop and the backtester agree exactly, bar for bar.")
    print("That is the claim 'this is the same logic in a live shape' being")
    print("verified rather than asserted.")
else:
    print(f"\n{mismatches} bars disagree. Investigate before trusting either.")
    print(recon[recon["mismatch"]].head().to_string())

# %% [markdown]
# ### Common causes of a real mismatch
#
# - The live loop seeded a **different warm-up window**, so an indicator's
#   trailing statistics differ.
# - The strategy holds **internal state** (a random seed, a halt flag) that a
#   fresh backtest does not reproduce.
# - Floating-point accumulation differs between vectorised and iterative paths.
# - The live feed **revised** a bar after publishing it. Exchanges do this.

# %% [markdown]
# ## 3. The failure modes that only appear live
#
# Your backtest assumed a clean, complete, immutable series of bars. Production
# gives you none of those things.

# %%
failure_modes = [
    ("A bar arrives late", "Your loop is blocked or you act on stale data.",
     "Timestamp every bar. Refuse to act on anything older than one interval."),
    ("A bar never arrives", "Your indicator windows silently span a gap.",
     "Track expected vs received bars. Halt after N consecutive misses."),
    ("A bar arrives twice", "You double-count a return and trade twice.",
     "Deduplicate by timestamp before processing. Idempotent handlers."),
    ("A bar is revised", "The close you traded on changes after the fact.",
     "Store what you saw at decision time; reconcile against revisions later."),
    ("The process restarts", "You lose your position state and warm-up window.",
     "Persist state every bar. Reload and reconcile against the broker on start."),
    ("The exchange is down", "Orders queue up and all fill at once on recovery.",
     "Cancel unfilled orders on disconnect. Never blind-resubmit."),
    ("Your order is rejected", "You believe you have a position you do not have.",
     "Reconcile your position against the broker's every single bar."),
    ("A partial fill", "You own 0.3 of what you asked for.",
     "Size from ACTUAL position, never intended position."),
    ("The market gaps through your stop", "You exit far below your stop price.",
     "Model stops pessimistically. A stop is not a guaranteed price."),
    ("Your clock drifts", "You act on the wrong bar boundary.",
     "Use the exchange's timestamps, never your own machine's."),
]

for i, (mode, consequence, mitigation) in enumerate(failure_modes, 1):
    print(f"{i:2}. {mode}")
    print(f"    consequence: {consequence}")
    print(f"    mitigation : {mitigation}\n")

# %% [markdown]
# **None of these are visible in a backtest.** Every one of them has ended a real
# trading system. This is what a few weeks of paper trading is genuinely for.

# %% [markdown]
# ## 4. Simulating the failures
#
# Let us actually break the feed and see what happens.

# %%
def stream_with_faults(data, session_factory, *, drop_every=0, duplicate_every=0, warmup=2000):
    """Replay bars while injecting feed faults, and report what the loop did."""
    live = session_factory()
    delivered = 0
    for i, (timestamp, bar) in enumerate(data.iloc[warmup:].iterrows()):
        if drop_every and i % drop_every == 0:
            continue                                   # a bar that never arrives
        live.on_new_bar(timestamp, bar)
        delivered += 1
        if duplicate_every and i % duplicate_every == 0:
            live.on_new_bar(timestamp, bar)            # the same bar, twice
            delivered += 1
    return live, delivered


def make_session():
    return PaperTradingSession(
        VolatilityTargeted(MomentumStrategy(48), target_vol=0.20, max_leverage=1.0, bars_per_year=BPY),
        history=df.iloc[:WARMUP], initial_cash=10_000.0, cost_model=RETAIL_CRYPTO,
    )


scenarios = {
    "clean feed": dict(),
    "1 bar in 50 dropped": dict(drop_every=50),
    "1 bar in 50 duplicated": dict(duplicate_every=50),
    "both faults": dict(drop_every=50, duplicate_every=37),
}

print(f"{'scenario':>26} {'bars':>8} {'fills':>7} {'final equity':>14} {'vs clean':>10}")
clean_equity = None
for name, faults in scenarios.items():
    live, delivered = stream_with_faults(df, make_session, **faults)
    equity = live.broker.account.equity
    if clean_equity is None:
        clean_equity = equity
    print(f"{name:>26} {delivered:>8,} {len(live.broker.fills):>7,} {equity:>14,.2f} "
          f"{equity / clean_equity - 1:>+10.2%}")

# %% [markdown]
# > **A duplicated bar is not a rounding error.** The loop processes it as a
# > genuine new observation: indicators advance, the position is re-evaluated,
# > and a trade may fire. In production you deduplicate by timestamp *before*
# > anything else touches the bar.
#
# Note that the paper session here does **not** deduplicate — that is why the
# numbers move. Exercise 12.2 asks you to add the guard.

# %% [markdown]
# ## 5. What a paper-trading run can and cannot tell you

# %%
paper_returns = session.equity_curve().pct_change().dropna()
n_bars = len(paper_returns)
se_sharpe = np.sqrt(1 / n_bars) * np.sqrt(BPY)

print(f"paper run length   : {n_bars:,} bars ({n_bars / 24:.0f} days)")
print(f"observed Sharpe    : {M.sharpe_ratio(paper_returns, BPY):+.2f}")
print(f"standard error     : {se_sharpe:.2f}")
print(f"95% interval       : [{M.sharpe_ratio(paper_returns, BPY) - 1.96 * se_sharpe:+.2f}, "
      f"{M.sharpe_ratio(paper_returns, BPY) + 1.96 * se_sharpe:+.2f}]")

print(f"\nTo distinguish a true Sharpe of 1.0 from zero at 95% confidence you need")
print(f"roughly {int((1.96 / 1.0) ** 2 * BPY):,} bars = {(1.96 / 1.0) ** 2:.1f} years.")
print(f"\nThis run is {n_bars / BPY:.2f} years long. It answers ENGINEERING questions.")
print("It cannot answer whether the strategy is profitable, and no honest")
print("amount of paper trading in a student project will.")

# %% [markdown]
# ## 6. Before real money — the questions
#
# This workshop does not connect to a live account and will not help you do so.
# If you later choose to trade real money, that is your decision to make
# deliberately. These are the questions to answer **in writing** first:
#
# 1. **What is the economic reason this works?** Who is on the other side of your
#    trade, and why are they willing to lose to you? "The backtest was good" is
#    not an answer.
# 2. **Is the out-of-sample result significant after deflation**, with an honest
#    trial count?
# 3. **Is it positive in most walk-forward folds**, or did one period carry it?
# 4. **What is the break-even cost**, and what do you actually pay?
# 5. **What happens in the worst historical month?** Could you hold through it?
# 6. **What is your maximum acceptable loss**, and what do you do on reaching it?
# 7. **What would falsify this?** What result would make you stop?
# 8. **What can you afford to lose entirely?** That is the maximum you may risk.
#
# If any answer is missing, the answer to "should I trade this" is no.

# %% [markdown]
# ---
# ## Exercises

# %% [markdown]
# ### Exercise 12.1 — Reconcile a deliberately broken strategy
#
# Write a strategy with internal state that is *not* reset in `on_start` — for
# example, a counter initialised in `__init__` and incremented every bar.
#
# Run it live and in backtest, reconcile, and observe the mismatch. Then fix it
# by resetting the state in `on_start` and confirm the mismatch disappears.
#
# This is a real class of bug: state that survives between runs makes your
# backtest unreproducible.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 12.2 — Add the deduplication guard
#
# Subclass `PaperTradingSession` and override `on_new_bar` so that a bar whose
# timestamp has already been processed is **ignored**.
#
# Re-run the fault scenarios from §4. Confirm the duplicate scenario now matches
# the clean feed exactly. Then decide: should a duplicate be silently ignored, or
# logged and alerted? Justify your answer.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 12.3 — Restart safety
#
# Run a session for 1,000 bars. Serialise everything needed to resume — position,
# cash, warm-up window, any strategy state — to JSON. Start a **fresh** session
# from that file and continue for another 1,000 bars.
#
# Compare against an uninterrupted run. They must match exactly. If they do not,
# find what you failed to persist.

# %%
# Your code here.

# %% [markdown]
# ### Exercise 12.4 — Your pre-flight checklist
#
# Write the operational checklist you would run before letting a strategy trade
# unattended for a week. Include monitoring, alerting, kill switches and a
# specific, numeric condition under which you stop it.
#
# Be concrete. "Monitor performance" is not a checklist item; "halt if realised
# drawdown exceeds 15% or if two consecutive bars fail to arrive" is.

# %%
# Your checklist here.

# %% [markdown]
# ---
# ## What you should take away
#
# - Paper trading answers **engineering** questions. It cannot answer whether a
#   strategy is profitable, and claiming otherwise is a statistical error.
# - Reconcile live against backtest bar by bar. A silent divergence is how a
#   working strategy becomes a losing one.
# - Feed faults — late, missing, duplicated and revised bars — are normal, and
#   none of them appear in a backtest.
# - Before real money, answer all eight questions in writing. A missing answer
#   is itself the answer.
#
# Next: **the capstone.** Everything you have learned, applied to a strategy of
# your own, reported honestly.
