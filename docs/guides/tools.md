---
title: Tool ecosystem
layout: default
nav_order: 3
parent: Guides
---
# The tool ecosystem

Every tool below solves a real problem. Every one of them is the wrong choice
for some situations, and those situations are listed too — because "when not to
use it" is the part that never appears in a README.

This workshop deliberately uses **none of them**. `algotrade` is about 2,000
lines you can read in an afternoon, which is the point: you cannot learn what a
backtester is doing wrong if you cannot see what it is doing.

<div class="callout" markdown="1">
**How to read this page.** Everything here is a mature project with real users.
The criticisms are about *fit*, not quality. Pick the tool whose assumptions
match your problem.
</div>

---

## Data manipulation

### pandas
**Solves:** labelled, time-indexed tabular data. Alignment, resampling,
rolling windows, joins. The default and correct starting point.

**Don't use it when:** your data does not fit comfortably in memory (rule of
thumb: over ~5 GB), or you are in a tight per-bar loop — pandas indexing has
enough per-call overhead to dominate an event-driven backtest. `algotrade`'s
engine drops to numpy arrays inside the loop for exactly this reason.

**Watch out for:** `SettingWithCopyWarning` (it is usually telling you something
real), silent index misalignment on arithmetic between series, and
`.resample().mean()` on prices, which invents prices that never traded.

### polars
**Solves:** the same problems as pandas, much faster, with a lazy query engine
and far better memory behaviour. Multi-core by default.

**Don't use it when:** you need an ecosystem integration that assumes pandas —
most plotting libraries, older ML tooling, and a lot of finance-specific
packages. Also when you are learning: the error messages are less familiar and
the Stack Overflow corpus is smaller.

**Verdict:** if you are starting a new data pipeline over millions of rows,
polars is probably the better choice. For a workshop, pandas' ubiquity wins.

### numpy
**Solves:** raw numerical arrays. Everything else is built on it.

**Don't use it when:** you need labelled axes or time alignment — doing that by
hand with integer indices is how off-by-one bugs get into production.

---

## Backtesting frameworks

<div class="callout warn" markdown="1">
**Before adopting any framework, answer one question:** *when does it execute an
order relative to the bar that generated the signal?* If you cannot find the
answer in the documentation within five minutes, you cannot trust its numbers.
</div>

### backtrader
**Solves:** event-driven backtesting with a mature feature set — multiple data
feeds, resampling, live trading adapters, a large body of examples.

**Don't use it when:** you need speed (it is slow on large datasets), or you
want to understand the execution model — the abstraction layers are deep, and
tracing exactly when a fill happens takes real effort.

**Watch out for:** development has been largely dormant for years. Still widely
used, but check issue activity before committing.

### vectorbt
**Solves:** extremely fast vectorised backtesting, built on numba. Sweeping
thousands of parameter combinations in seconds is its core competence.

**Don't use it when:** your strategy has path dependence that does not vectorise
— state machines, position-dependent sizing, complex order types. And be careful
using it for *research*: making it trivial to test 10,000 parameter combinations
makes it trivial to data-snoop at industrial scale. If you use vectorbt, you
**must** deflate your Sharpe by the number of trials, and that number is now
enormous.

**Verdict:** excellent for what it does. The speed is a liability if your
methodology is not already disciplined.

### zipline / zipline-reloaded
**Solves:** the framework behind the original Quantopian. Realistic equity
market simulation with a solid treatment of corporate actions, trading calendars
and slippage.

**Don't use it when:** you are trading crypto (the calendar model assumes market
hours), or you need modern Python — the original is unmaintained; use
`zipline-reloaded` if you want this.

### QuantConnect / LEAN
**Solves:** a full research-to-live platform. Data, backtesting, paper trading,
live deployment, multiple asset classes, cloud compute.

**Don't use it when:** you want to own your stack, work offline, or avoid a
subscription for serious data. Also: the convenience of live deployment being
one click away is a genuine hazard for a beginner, and this workshop's position
is that the gap between "the backtest looked good" and "money is at risk" should
be deliberately wide.

### Writing your own
**Solves:** understanding. You cannot debug an execution model you have never
looked at.

**Don't use it when:** you need multi-asset portfolio accounting, corporate
actions, or realistic microstructure. Those are genuinely hard and the
frameworks have already done them.

**Verdict:** write one once, to learn. Then use a framework — but now you will
be able to read its execution model and know what to check.

---

## Market data

### ccxt
**Solves:** a unified API across 100+ crypto exchanges. Public market data needs
no API key.

**Don't use it when:** you need deep historical data — most exchanges' public
endpoints cap how far back you can go, and rate limits make long histories slow
to collect. Also, "unified" leaks: exchanges differ in timestamp conventions,
pagination and symbol naming in ways that will bite you.

**Watch out for:** rate limits. Be a good citizen of free APIs; add delays.

### Alpaca
**Solves:** commission-free US equities and crypto, with a **genuine paper
trading API** — a real sandbox with real market data and fake money.

**Don't use it when:** you need non-US markets or professional-grade historical
data.

**Note for this workshop:** Alpaca's paper API is the natural next step if you
want to extend notebook 12 beyond simulation. `algotrade` deliberately does not
integrate with it. Wiring that up should be a decision you make explicitly, and
if you do, keep it on the paper endpoint.

### yfinance
**Solves:** free equity and ETF data, trivially. Good for teaching.

**Don't use it when:** you need reliability or point-in-time correctness. It
scrapes an endpoint that is not a supported API, it breaks periodically, and its
adjusted prices are adjusted *as of today* — which means backtesting with them
is a subtle form of look-ahead bias.

### Paid vendors (Polygon, Databento, Kaiko, Refinitiv)
**Solve:** reliability, depth, tick data, point-in-time universes, support.

**Don't use them when:** you are learning. Nothing in this workshop needs paid
data, and the methodology is identical either way.

---

## Indicators

### TA-Lib
**Solves:** 150+ technical indicators, C-backed, fast, battle-tested.

**Don't use it when:** you cannot install it — it requires a C library and the
installation is a well-known source of frustration, particularly on Windows.

### pandas-ta
**Solves:** the same indicator set, pure Python, `pip install` and done.

**Don't use it when:** you are in a tight loop (it is slower), or you need to
know exactly what a function is doing — check the source for shift conventions
before trusting any indicator library. **An indicator that silently uses a
centred window will hand your strategy the future.**

**Watch out for:** maintenance status varies across forks. Check before adopting.

### Writing them yourself
Ten lines each, and you know the shift convention. `algotrade/indicators.py`
does this, and `tests/test_causality.py` proves every one of them is causal by
corrupting the future and checking the past does not move. That test is worth
copying into whatever you build next, whichever library you use.

---

## Machine learning

### scikit-learn
**Solves:** the standard ML toolkit. Consistent API, excellent documentation,
pipelines that prevent scaler leakage.

**Don't use it when:** you need the *time-series-aware* cross-validation this
workshop insists on. `TimeSeriesSplit` exists and is better than `KFold`, but it
does **not** implement purging or embargo. And the default `train_test_split`
shuffles, which is catastrophic on market data.

**Verdict:** use it, but bring your own splitter. `algotrade.validation` is about
150 lines and does purging and embargo properly.

### XGBoost / LightGBM
**Solve:** gradient boosting on tabular data. Strong baselines, fast, handle
missing values natively.

**Don't use them when:** your signal-to-noise ratio is as low as it is in
finance and you have not first established a linear baseline. Flexible models fit
noise enthusiastically. **If your boosted model dramatically beats your logistic
regression, check for leakage before celebrating.**

### PyTorch / TensorFlow
**Solve:** deep learning.

**Don't use them when:** you are predicting next-bar direction from 20 technical
indicators, which is most of what people try. You have a few thousand effective
samples with a correlation of 0.03 to the target. That is not a deep learning
problem; it is a problem where a linear model with good features will win and be
debuggable.

**When they do make sense:** genuinely high-dimensional inputs — order book
snapshots, news text, cross-sectional data across thousands of assets.

### Optuna
**Solves:** efficient hyperparameter search — Bayesian optimisation, pruning,
parallelism.

**Don't use it when:** you have not yet counted your trials. Optuna running 500
trials means `n_trials = 500` in your deflated Sharpe calculation, and it will
comfortably find a configuration that looks excellent on pure noise. It is a
powerful tool for finding the best of many options, which is precisely the
activity that requires the strongest statistical correction.

**Use it correctly:** run it **inside** each walk-forward fold, on training data
only. Never on the test set.

### MLflow
**Solves:** experiment tracking. Parameters, metrics, artefacts, model versions.

**Don't use it when:** the overhead exceeds the benefit — for a single-notebook
project, a CSV of results is fine.

**Why it genuinely matters here:** MLflow counts your experiments for you. The
single hardest part of an honest deflated Sharpe ratio is knowing $N$, and
people systematically undercount because they forget the ideas they abandoned.
A tracking server does not forget.

---

## Statistics

### scipy.stats
Distributions, hypothesis tests, fitting. Use it for skew, kurtosis and
normality tests on returns — and expect returns to fail them.

### statsmodels
Time-series models (ARIMA, GARCH), stationarity tests (ADF, KPSS), regression
with proper inference. Better than scikit-learn when you want **p-values and
confidence intervals** rather than predictions — which, for evaluating whether
an edge is real, is usually what you want.

### arch
GARCH-family volatility models. Worth knowing because volatility is far more
predictable than returns, and volatility forecasting feeds directly into the
position sizing of notebook 11.

---

## A suggested progression

| stage | stack |
|---|---|
| **Learning** | pandas + scikit-learn + your own 2,000-line backtester |
| **Researching** | pandas/polars + a real framework + MLflow + purged walk-forward |
| **Running something** | a framework with live adapters, proper monitoring, an independent risk check |

Do not skip the first row. Every framework in this page makes an execution-timing
assumption, and you will not know which questions to ask it until you have
written your own loop and seen what a one-bar shift does to an equity curve.
