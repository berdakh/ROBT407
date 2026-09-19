---
title: Home
layout: home
nav_order: 1
---

# Algorithmic & AI-Driven Trading
{: .fs-9 }

A four-day, hands-on workshop for engineers and computer scientists. You will
build a trading strategy that appears to return **+46,986%**, then spend a day
taking it apart until it returns **+16%** — and loses to doing nothing.
{: .fs-6 .fw-300 }

[Start with Day 1](plan.html){: .btn .btn-primary .mr-2 }
[Open the notebooks](notebooks.html){: .btn .mr-2 }
[Get the code](https://github.com/berdakh/ROBT407){: .btn }

{: .warning }
> **Educational use only — not investment advice.**
> This workshop teaches a *methodology for evaluating* trading strategies. It
> recommends no strategy, asset or trade. **Paper trading only** — nothing here
> connects to a live-money account and there is no live order path in the code.
> Most retail algorithmic strategies lose money after costs.

---

![The four-day roadmap: data and a first strategy, why it was a lie, machine learning done correctly, then risk and shipping]({{ '/assets/diagrams/06-workshop-map.svg' | relative_url }})

## The central problem

A student who runs a backtest and sees a 300% return has learned nothing except
how to fool themselves.

So the whole workshop is built around one question — **why is your backtest
lying to you?** — and every bias is something you *measure*, not read about.

Here is the spine. One strategy, an SMA 3/24 crossover picked as the best of a
22-combination grid search, evaluated four ways:

<div class="spine">
  <div class="step bad">
    <div class="value">+46,986%</div>
    <div class="how">no shift, no costs</div>
    <div class="note">the Day 1 result</div>
  </div>
  <div class="step warn">
    <div class="value">+406%</div>
    <div class="how">indexing corrected</div>
    <div class="note">look-ahead removed</div>
  </div>
  <div class="step warn">
    <div class="value">+16%</div>
    <div class="how">+ 24 bps round trip</div>
    <div class="note">costs added</div>
  </div>
  <div class="step good">
    <div class="value">+582%</div>
    <div class="how">buy &amp; hold</div>
    <div class="note">doing nothing won</div>
  </div>
</div>

Day 1 ends at the first number. Day 2 produces the rest.

## The notebooks

Thirteen notebooks, in dependency order. **Every one opens in Google Colab with
one click** — no install, no GPU, no account beyond a Google login. The first
cell clones the repository and installs the package for you.

| Day | # | Notebook | What it does | Open |
|:--|:--|:--|:--|:--|
| 1 | 01 | **Market data** | Timestamps, the four corrupting defects, resampling | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/01_market_data.ipynb) |
| 1 | 02 | **Returns and risk** | Log vs simple, fat tails, Sharpe with its error bar | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/02_returns_and_risk.ipynb) |
| 1 | 03 | **Your first strategy** | Builds the +46,986% result — and +2,119% on noise | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/03_first_strategy.ipynb) |
| 2 | 04 | **Event-driven backtest** | The engine that cannot lie; four cheaters, all blocked | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/04_event_driven_backtest.ipynb) |
| 2 | 05 | **Costs and slippage** | The collapse. Turnover × round trip ÷ 2 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/05_costs_and_slippage.ipynb) |
| 2 | 06 | **The bias catalogue** | Seven biases, each constructed and measured | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/06_bias_catalogue.ipynb) |
| 3 | 07 | **Features and labels** | Causality proofs, stationarity, triple-barrier labels | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/07_features_and_labels.ipynb) |
| 3 | 08 | **ML signals** | Gross Sharpe +2.3, net −4.9. The ML worked; the strategy didn't | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/08_ml_signals.ipynb) |
| 3 | 09 | **Walk-forward validation** | 10/10 folds on a real edge, 4/10 on noise | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/09_walk_forward.ipynb) |
| 3 | 10 | **Honest evaluation** | One report card, with random strategies as the control | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/10_honest_evaluation.ipynb) |
| 4 | 11 | **Risk and sizing** | Vol targeting, drawdown limits, why leverage buys nothing | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/11_risk_and_sizing.ipynb) |
| 4 | 12 | **Paper trading** | The live loop, reconciliation, ten failure modes | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/12_paper_trading.ipynb) |
| 4 | 13 | **Capstone** | Your own strategy. Performance is worth **0%** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/13_capstone.ipynb) |
{: .nb-table }

More detail, including what each notebook needs: [the notebooks page](notebooks.html).

## What you will be able to do

By the end of four days, with no paid data, no GPU and no broker account:

1. **Load and reason about market data** without the classic mistakes.
2. **Implement a strategy** and backtest it in a way that does not lie to you.
3. **Name, detect and measure** look-ahead, survivorship, data snooping,
   ignored costs and slippage.
4. **Apply machine learning** with time-series-correct validation, judged on
   Sharpe, drawdown and turnover rather than classification accuracy.
5. **Size positions and manage risk** with volatility targeting and drawdown limits.
6. **Run a strategy in paper trading** and write an honest assessment of why it
   might fail with real money.

## Prerequisites

Comfortable Python — functions, classes, dicts, NumPy and pandas basics. You do
**not** need any finance background.

| You have | You can do |
|:---|:---|
| Any laptop | Everything. The workshop is CPU-only by design |
| No install at all | Everything, in free Google Colab |
| No data subscription | Everything. Sample data is committed to the repository |
| No broker account | Everything. Paper trading is simulated and always will be |

## The four days

| Day | Theme | Notebooks | You end the day able to |
|:---|:---|:---|:---|
| [**1**](plan.html#day-1--data-returns-and-a-strategy-that-looks-spectacular) | Data & a first strategy | 01–03 | Build a backtest that looks spectacular |
| [**2**](plan.html#day-2--why-it-was-a-lie) | Why it was a lie | 04–06 | Explain and measure every bias that produced it |
| [**3**](plan.html#day-3--machine-learning-done-correctly) | ML, done correctly | 07–10 | Validate a model without fooling yourself |
| [**4**](plan.html#day-4--risk-shipping-and-the-capstone) | Risk & shipping | 11–13 | Size positions, paper trade, and report honestly |

Full hour-by-hour schedule and instructor notes: [the day plan](plan.html).

## Set up in two minutes

```bash
git clone https://github.com/berdakh/ROBT407.git
cd ROBT407
pip install -e ".[dev,notebooks]"
pytest                             # 174 tests, no network needed
jupyter notebook notebooks/
```

Sample data is committed, so everything runs offline immediately.
Full instructions, including Colab and Windows: [Setup](guides/setup.html).

## The one piece of engineering that matters

Look-ahead bias is **structurally impossible** in this backtester, not merely
discouraged. The engine never hands a strategy the full price series — it hands
it a view physically truncated at the current bar:

```python
view.close[-1]          # the current bar     fine
view.close[-20:]        # the last 20 bars    fine
view.close[view.i + 1]  # tomorrow            LookAheadError
```

There is no `.shift(1)` to remember, because the future is not in the room.
`tests/test_lookahead_guard.py` proves it with four strategies that deliberately
cheat and must all crash.

![The look-ahead trap: a signal computed from bar t's close, applied to bar t's own return, earns a move that had already finished]({{ '/assets/diagrams/02-lookahead-trap.svg' | relative_url }})

## Background reading: the handbook

The notebooks teach you to build. [**The handbook**](handbook.html) explains what
you are building on, and it stands alone with no code to run: what a market
actually is, how an order really executes, what Sharpe and drawdown really mean,
why most strategies fail, and who you are trading against.

## Reference guides

- [**Setup**](guides/setup.html) — local, Colab, and fully offline.
- [**Bias reference**](guides/biases.html) — one page, every bias, how to detect it.
- [**Tool ecosystem**](guides/tools.html) — pandas, backtrader, vectorbt, zipline,
  QuantConnect, ccxt, Alpaca, scikit-learn, Optuna, MLflow and more, each with
  the problem it solves **and when not to use it**.
- [**Library reference**](guides/library.html) — the `algotrade` package, module by module.

## A note on honesty

Most honest quantitative research finds nothing. This workshop is graded on
methodology, not performance — the capstone assigns **0%** to how well your
strategy did. A submission concluding *"I found no evidence of an edge"* and
proving it rigorously scores higher than a spectacular backtest with an
unexamined method.
