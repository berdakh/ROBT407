---
title: Day plan
layout: default
nav_order: 3
---
# Day-by-day plan

Four days, roughly six hours each. Times are a guide; the exercises expand to
fill whatever you give them.

<div class="callout" markdown="1">
**Read before you start:** the [handbook]({{ '/handbook.html' | relative_url }})
sections listed at the top of each day. They are code-free and take about
twenty minutes each.
</div>

![The four-day arc]({{ '/assets/diagrams/06-workshop-map.svg' | relative_url }})

---

## Day 1 — Data, returns, and a strategy that looks spectacular

**Handbook:** §1 (what a market is), §4 (what Sharpe really means)

| time | session | notebook |
|---|---|---|
| 0:00–0:20 | Setup: clone, install, run the tests | — |
| 0:20–1:50 | Market data: OHLCV, timestamps, the four corrupting defects, resampling | `01_market_data` |
| 1:50–2:00 | Break | |
| 2:00–3:30 | Returns and risk: simple vs log, annualising, fat tails, Sharpe error bars | `02_returns_and_risk` |
| 3:30–4:00 | Lunch | |
| 4:00–6:00 | **Your first strategy**: signal, vectorised backtest, grid search | `03_first_strategy` |

### The moment that matters

Day 1 ends with a working strategy reporting **+46,986%**. Students should leave
excited about it. Do not spoil Day 2.

The one hint to plant, in the last fifteen minutes, is the control experiment:
run the identical procedure on `pure_noise.csv`, a dataset generated as a
driftless random walk whose manifest states in advance that no edge exists. It
finds **+2,119%** there too. Let that sit overnight without explaining it.

### Checkpoint
Students can load and validate an OHLCV file, compute and annualise returns,
implement a signal, and run a vectorised backtest. They should also be able to
state, without prompting, that they do not yet know when their trades executed.

---

## Day 2 — Why it was a lie

**Handbook:** §2 (how an order really executes), §3 (what a backtest is not)

| time | session | notebook |
|---|---|---|
| 0:00–0:15 | Recap. Collect Day 1's predictions before revealing anything | — |
| 0:15–2:15 | The event-driven engine; trying to cheat and failing | `04_event_driven_backtest` |
| 2:15–2:30 | Break | |
| 2:30–4:00 | **Costs and slippage: the collapse** | `05_costs_and_slippage` |
| 4:00–4:30 | Lunch | |
| 4:30–6:00 | The bias catalogue, each one measured | `06_bias_catalogue` |

### The moment that matters

The two-stage collapse, run live in front of the room:

<div class="spine">
  <div class="step bad"><div class="value">+46,986%</div><div class="note">Day 1</div></div>
  <div class="step warn"><div class="value">+406%</div><div class="note">timing fixed</div></div>
  <div class="step warn"><div class="value">+16%</div><div class="note">costs added</div></div>
  <div class="step good"><div class="value">+582%</div><div class="note">buy &amp; hold</div></div>
</div>

Then the distinction most courses skip: **look-ahead bias won 40 out of 40
trials** across datasets, including ones built with no signal. Optimistic fill
timing won only **22 of 40** — a coin flip. Unrealistic and inflated are
different properties, and only the inflated kind fools you.

### Checkpoint
Students can write a `Strategy` subclass, explain why look-ahead is structurally
impossible in the engine, compute a cost budget from turnover, and name and
detect all seven biases.

---

## Day 3 — Machine learning, done correctly

**Handbook:** §3 (backtests), §6 (why strategies fail)

| time | session | notebook |
|---|---|---|
| 0:00–2:00 | Features and labels: causality proofs, stationarity, triple barrier | `07_features_and_labels` |
| 2:00–2:15 | Break | |
| 2:15–4:00 | ML signals: gross vs net, why accuracy is the wrong number | `08_ml_signals` |
| 4:00–4:30 | Lunch | |
| 4:30–6:00 | Walk-forward validation; honest evaluation | `09_walk_forward`, `10_honest_evaluation` |

### The moment that matters

Notebook 08 trains a model on data with a **known planted edge** and it finds it:
out-of-sample information coefficient **+0.033**, gross Sharpe **+2.3**. Then
costs take the net Sharpe to **−4.9**.

> The machine learning worked. The strategy did not. Those are different
> statements.

Notebook 09 then does something the single split could not. Walk-forward finds
the planted edge in **10 of 10 folds (t = +5.18)** and correctly reports nothing
on both no-edge datasets (**4 of 10, t = −1.14**). Recall that in notebook 08 a
single split gave the *noise* model a gross Sharpe of 1.8 with no way to tell it
was an accident. That contrast is the entire value of validation.

![Walk-forward vs shuffled CV]({{ '/assets/diagrams/03-walk-forward-vs-naive-cv.svg' | relative_url }})

### Checkpoint
Students can build a causal feature matrix and prove it causal, construct labels
that remember their horizon, run a purged walk-forward, and report a result with
its dispersion and t-statistic rather than a single number.

---

## Day 4 — Risk, shipping, and the capstone

**Handbook:** §5 (drawdown), §7 (who you trade against), §8 (how professionals work)

| time | session | notebook |
|---|---|---|
| 0:00–2:00 | Risk and position sizing | `11_risk_and_sizing` |
| 2:00–2:15 | Break | |
| 2:15–4:00 | Paper trading, reconciliation, failure modes | `12_paper_trading` |
| 4:00–4:30 | Lunch | |
| 4:30–6:00 | Capstone: start it, agree hypotheses | `13_capstone` |

### The moment that matters

The leverage table in notebook 11. Sharpe stays flat across 0.5× to 5× while max
drawdown deteriorates steadily. **Leverage buys nothing risk-adjusted and
shortens the distance to zero.**

And in notebook 12, the statistical power calculation applied to paper trading
itself: a 282-day run gives a Sharpe standard error of 1.14. Paper trading
answers engineering questions. It cannot answer profitability, and no student
project will.

### Checkpoint
Students can size positions to a volatility target, apply a drawdown limit and
state its cost, run a live loop that reconciles against the backtest, and write
an honest assessment including what would change their mind.

---

## The capstone

Assessed on methodology, not performance:

| weight | criterion |
|---|---|
| 30% | Methodological rigour — no leakage, correct validation, honest trial count |
| 25% | The honest assessment — specific, falsifiable, not hedged |
| 20% | Controls and benchmarks — noise, random strategies, buy-and-hold |
| 15% | Code quality |
| 10% | Hypothesis quality — is there a *mechanism*? |
| **0%** | **Strategy performance** |

<div class="callout warn" markdown="1">
**Say this explicitly, more than once.** A capstone concluding *"I found no
evidence of an edge"* and proving it rigorously scores higher than one reporting
a spectacular backtest with an unexamined methodology.

Most honest quantitative research finds nothing. Students who have only been
graded on getting the right answer find this genuinely difficult to believe, so
repeat it.
</div>

---

## Notes for instructors

**Let Day 1 be exciting.** The workshop does not work if students are warned in
advance. They need to feel the pull of the number before they learn to distrust
it. Some will be annoyed on Day 2; that reaction is the lesson taking hold.

**Collect the Day 1 predictions.** Exercise 3.4 asks students to write down what
they think the strategy will return after Day 2's corrections. Reading those
aloud on Day 2 is the single most effective fifteen minutes of the workshop.

**The control experiment is the transferable skill.** If students leave with one
habit, it should be running their method on data with no signal in it. It is
cheap, it requires no statistics, and it catches the bias that matters most.

**Expect resistance to "no result is a result".** It contradicts years of
training. Grade accordingly and say so in advance.

**Timing is tight on Day 3.** If you are running behind, cut notebook 10 to a
30-minute walkthrough rather than dropping notebook 09 — validation is load-
bearing and the report card is not.
