---
layout: default
title: Algorithmic & AI-Driven Trading
---

<link rel="stylesheet" href="{{ '/assets/style.css' | relative_url }}">

# Algorithmic & AI-Driven Trading

**A four-day hands-on workshop for engineers and computer scientists.** Solid
Python assumed; no finance background required.

<div class="callout danger" markdown="1">
### Educational use only — not investment advice

This workshop teaches a **methodology for evaluating trading strategies**. It
recommends no strategy, asset or trade, and makes no performance claims you
should act on.

- **Paper trading only.** Nothing here connects to a live-money account, and the
  package contains no live order path. `PaperBroker.connect()` raises and always
  will.
- **Most retail algorithmic trading strategies lose money after costs.** That is
  the base rate, and it is the reason this workshop exists.
- **A good backtest is weak evidence.** It describes the past, filtered through
  choices you made after seeing that past.
</div>

## The central problem

A student who runs a backtest and sees a 300% return has learned nothing except
how to fool themselves.

So the entire workshop is built around one question — **why is your backtest
lying to you?** — and every bias is something you *measure*, not something you
read about.

Here is the spine of the course. One strategy, an SMA 3/24 crossover, selected
as the best of a 22-combination grid search, evaluated four ways:

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

Day 1 ends with the first number. Day 2 produces the rest. That contrast is the
course.

![The four-day arc]({{ '/assets/diagrams/06-workshop-map.svg' | relative_url }})

## The four days

<div class="days">
  <div class="day" markdown="1">
  <span class="tag">Day 1</span>
  ### Data & a first strategy
  - Market data, timestamps, gaps
  - Returns, volatility, Sharpe error bars
  - The naive backtest that looks spectacular
  </div>
  <div class="day" markdown="1">
  <span class="tag">Day 2</span>
  ### Why it was a lie
  - The event-driven engine
  - Costs, spread, slippage, impact
  - The full bias catalogue, measured
  </div>
  <div class="day" markdown="1">
  <span class="tag">Day 3</span>
  ### ML, done correctly
  - Causal features, honest labels
  - Walk-forward and purged splits
  - Accuracy is not profit
  </div>
  <div class="day" markdown="1">
  <span class="tag">Day 4</span>
  ### Risk & shipping
  - Volatility targeting, drawdown limits
  - Paper trading and reconciliation
  - The capstone
  </div>
</div>

## Start here

| | |
|---|---|
| **[Day-by-day plan]({{ '/plan.html' | relative_url }})** | Hour-by-hour schedule, notebook order, what each session covers |
| **[The handbook]({{ '/handbook.html' | relative_url }})** | Background reading, no code. Market mechanics, how orders execute, what Sharpe and drawdown really mean, why most strategies fail |
| **[Tool ecosystem]({{ '/guides/tools.html' | relative_url }})** | pandas, backtrader, vectorbt, zipline, QuantConnect, ccxt, Alpaca, TA-Lib, scikit-learn, Optuna, MLflow — what each solves and when *not* to use it |
| **[Bias reference]({{ '/guides/biases.html' | relative_url }})** | One-page catalogue: how to detect and avoid each bias |
| **[Library reference]({{ '/guides/library.html' | relative_url }})** | The `algotrade` package, module by module |
| **[Setup]({{ '/guides/setup.html' | relative_url }})** | Local install, Colab, offline use |

## The notebooks

Every notebook runs in Google Colab (badge included) and offline after one
clone. They are generated from reviewable `.py` sources in `notebook_src/`.

| # | Notebook | Day | What it does |
|---|---|---|---|
| 01 | [Market data](https://github.com/berdakh/ROBT407/blob/master/notebooks/01_market_data.ipynb) | 1 | OHLCV, timestamp conventions, the four corrupting defects, resampling |
| 02 | [Returns and risk](https://github.com/berdakh/ROBT407/blob/master/notebooks/02_returns_and_risk.ipynb) | 1 | Simple vs log returns, annualising, fat tails, Sharpe with error bars |
| 03 | [Your first strategy](https://github.com/berdakh/ROBT407/blob/master/notebooks/03_first_strategy.ipynb) | 1 | Builds the +46,986% result — and finds +2,119% on provably random data |
| 04 | [Event-driven backtest](https://github.com/berdakh/ROBT407/blob/master/notebooks/04_event_driven_backtest.ipynb) | 2 | The engine that cannot lie. Four cheating strategies, all blocked |
| 05 | [Costs and slippage](https://github.com/berdakh/ROBT407/blob/master/notebooks/05_costs_and_slippage.ipynb) | 2 | The collapse. Turnover × round trip ÷ 2 |
| 06 | [The bias catalogue](https://github.com/berdakh/ROBT407/blob/master/notebooks/06_bias_catalogue.ipynb) | 2 | Seven biases, each constructed and measured |
| 07 | [Features and labels](https://github.com/berdakh/ROBT407/blob/master/notebooks/07_features_and_labels.ipynb) | 3 | Causality proofs, stationarity, triple-barrier labels |
| 08 | [ML signals](https://github.com/berdakh/ROBT407/blob/master/notebooks/08_ml_signals.ipynb) | 3 | Gross Sharpe +2.3, net Sharpe −4.9. The ML worked; the strategy did not |
| 09 | [Walk-forward validation](https://github.com/berdakh/ROBT407/blob/master/notebooks/09_walk_forward.ipynb) | 3 | 10/10 folds on a real edge, 4/10 on noise. The method that tells them apart |
| 10 | [Honest evaluation](https://github.com/berdakh/ROBT407/blob/master/notebooks/10_honest_evaluation.ipynb) | 3 | One harness, every strategy, every warning |
| 11 | [Risk and sizing](https://github.com/berdakh/ROBT407/blob/master/notebooks/11_risk_and_sizing.ipynb) | 4 | Volatility targeting, drawdown limits, why leverage buys nothing |
| 12 | [Paper trading](https://github.com/berdakh/ROBT407/blob/master/notebooks/12_paper_trading.ipynb) | 4 | The live loop, reconciliation, the ten failure modes |
| 13 | [Capstone](https://github.com/berdakh/ROBT407/blob/master/notebooks/13_capstone.ipynb) | 4 | Your own strategy. 0% of the grade is performance |

## The one piece of engineering that matters

Look-ahead bias is **structurally impossible** in this backtester, not merely
discouraged.

The engine never hands a strategy the full price series. It hands it a
`MarketView` whose columns are physically truncated at the current bar. Reaching
past it raises `LookAheadError`.

```python
view.close[-1]        # the current bar          fine
view.close[-20:]      # the last 20 bars         fine
view.close[view.i+1]  # tomorrow                 LookAheadError
```

There is no `.shift(1)` to remember, because the future is not in the room.
`tests/test_lookahead_guard.py` proves it with four strategies that deliberately
cheat and must all crash.

![The look-ahead trap]({{ '/assets/diagrams/02-lookahead-trap.svg' | relative_url }})

## What you need

A normal laptop. No GPU, no paid data subscription, no broker account.

```bash
git clone https://github.com/berdakh/ROBT407.git
cd ROBT407
pip install -e .
pytest                              # 174 tests, no network required
jupyter notebook notebooks/
```

Sample data is committed, so everything runs offline immediately. Full
instructions on the [setup page]({{ '/guides/setup.html' | relative_url }}).
