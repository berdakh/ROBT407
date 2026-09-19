---
title: Library reference
layout: default
nav_order: 4
parent: Guides
---
# The `algotrade` library

About 2,000 lines, designed to be read. The notebooks import it rather than
burying logic in cells, so the same code is under test, under review and in
front of students.

**Read `view.py` first.** It is the reason the rest of the design works.

---

## Module map

| module | what it holds |
|---|---|
| `view` | **The look-ahead guard.** `MarketView`, `SeriesView`, `Clock` |
| `data` | Loading, validation, resampling, returns, gap detection |
| `synthetic` | Price series with known ground truth |
| `indicators` | Strictly causal technical indicators |
| `strategy` | The `Strategy` base class and reference strategies |
| `execution` | `Order`, `Fill`, `CostModel` |
| `portfolio` | `Account` — cash and position accounting |
| `backtest` | `Backtester`, `BacktestConfig`, `BacktestResult` |
| `metrics` | Performance statistics, with error bars |
| `report` | `report_card` — the shared evaluation harness |
| `validation` | Walk-forward and purged splits |
| `features` | Causal feature engineering |
| `labeling` | Forward-looking targets that remember their horizon |
| `risk` | Volatility targeting, drawdown control, Kelly |
| `paper` | A simulated broker that cannot reach an exchange |
| `naive` | The tutorial backtest, bugs included, for demonstration only |

---

## `view` — the piece that matters

The backtester never hands a strategy the full price series. It hands it a
`MarketView` whose columns are truncated at the current bar.

```python
view.close[-1]          # current bar               fine
view.close[-20:]        # last 20 bars              fine
view.close.history(20)  # last 20, as an array      fine
view.close[0]           # the first bar             fine

view.close[view.i + 1]  # tomorrow                  LookAheadError
view.close[len(view)]   # one past the present      LookAheadError
view.close[0:view.i+5]  # a slice into the future   LookAheadError
```

The escape hatches are closed too: `np.asarray(view.close)` returns only the
visible window, iteration stops at the current bar, and `view.frame()` gives a
DataFrame of the past only. Every accessor returns a **copy**, so a strategy
cannot mutate the market.

### How honest is the guard?

**Honest about accidents, not about sabotage.** The full array is stored on a
private attribute, so `view.close._values` reaches it. That is deliberate.

The goal is to make look-ahead impossible to commit *by accident* — via an
off-by-one, a forgotten `.shift(1)`, or a rolling mean computed over the whole
history before the loop starts. Those are the mistakes that actually happen.
Defeating the guard requires typing an underscore, which is a decision rather
than a slip.

`tests/test_lookahead_guard.py` proves the guard holds with four strategies that
deliberately cheat and must all crash.

---

## `backtest` — the engine

![The backtest loop]({{ '/assets/diagrams/01-backtest-loop.svg' | relative_url }})

For each bar `i`:

1. Advance the clock to `i` — the strategy can now see bars `0..i`.
2. Settle orders submitted on bar `i-1`, at bar `i`'s **open**, minus costs.
3. Mark to market at bar `i`'s close.
4. Ask the strategy. Anything it returns is queued for bar `i+1`.
5. Record equity, weight, position.

**The timing rule:** a signal computed from bar `i`'s close fills at bar `i+1`'s
open. One full bar of delay, always.

```python
from algotrade import BacktestConfig, RETAIL_CRYPTO, run_backtest

result = run_backtest(df, MyStrategy(), BacktestConfig(
    cost_model=RETAIL_CRYPTO,
    fill_timing="next_open",     # or "next_close"; "this_close" is unrealistic
    max_leverage=1.0,
    bars_per_year=365 * 24,
))
```

`BacktestResult` carries `equity`, `returns`, `weights` (achieved), `targets`
(requested), `units`, `fills` and `trades_frame()`.

> **`targets` vs `weights`.** The target is what the strategy asked for; the
> weight is what it ended up holding after fills at a different price. Reconcile
> targets against targets, or every row looks like a mismatch.

---

## `strategy` — writing one

```python
from algotrade import Strategy

class MyStrategy(Strategy):
    name = "my strategy"
    warmup = 48

    def on_bar(self, view, account):
        if len(view) < self.warmup:
            return None                       # not enough history yet
        closes = view.close.history(self.warmup)
        return 1.0 if closes[-1] > closes.mean() else 0.0
```

Return one of:

- `None` — no change
- a `float` — **target weight**, the fraction of equity to hold (`1.0` long,
  `0.0` flat, `-0.5` half short)
- an `Order` or list of them — for explicit market/limit control

Returning a *weight* rather than "buy 3 units" separates signal from sizing,
which is what lets the risk wrappers change sizing without touching any strategy.

---

## `execution` — what a trade costs

```python
CostModel(
    commission_bps=10.0,    # exchange taker fee
    half_spread_bps=1.0,    # you cross this each way
    latency_bps=1.0,        # adverse drift before your fill
    impact_coef=0.1,        # sqrt impact vs bar volume
)
```

Two presets: `ZERO_COST` (used once, on Day 1, to produce a beautiful lie) and
`RETAIL_CRYPTO` (**24 bps round trip**). Every component pushes the price
against you; a cost model that could help you is a bug.

`cost_model.round_trip_bps()` is the number to memorise.

---

## `report` — the shared harness

```python
card = report_card(result, benchmark=buy_and_hold,
                   n_trials=22, is_out_of_sample=False)
print(card.summary())
```

The metrics are the easy part. The **warnings** are the point, and the harness
fills them in automatically: zero costs, optimistic fills, in-sample evaluation,
multiple testing, too few trades, Sharpe indistinguishable from zero, Sharpe
implausibly high, high turnover, costs consuming profit, leverage used,
underperforming buy-and-hold.

It takes your word for `n_trials` and `is_out_of_sample`. Those are the two that
matter most, and there is no way around it: the integrity of a backtest rests on
the honesty of the person who ran it. Tooling can only make dishonesty require a
deliberate act rather than an oversight.

---

## `validation` — splits that do not leak

![Walk-forward vs shuffled CV]({{ '/assets/diagrams/03-walk-forward-vs-naive-cv.svg' | relative_url }})

```python
splits = walk_forward_splits(len(X), train_size=3000, test_size=500,
                             purge=24, embargo=24)
assert_no_leakage(splits, label_horizon=24)
```

`assert_no_leakage` raises on: train/test intersection, training data that is
not strictly before test data, and an insufficient gap given your label horizon.
Run it on every split you generate.

`purged_kfold_splits` exists for estimating model variance with limited data. It
trains on data *after* the test fold, so it is **not** a trading simulation and
must never be presented as one.

---

## `risk` — sizing

![Position sizing]({{ '/assets/diagrams/05-risk-and-position-sizing.svg' | relative_url }})

```python
VolatilityTargeted(MyStrategy(), target_vol=0.20, max_leverage=1.0)
DrawdownGuard(MyStrategy(), max_drawdown=0.20, reduced_exposure=0.0)
```

Both are wrappers: strategy in, strategy out, so they compose. Both document
what they cost as well as what they save — volatility targeting cuts exposure
during exactly the rallies you wanted, and a portfolio stop-loss systematically
sells low.

---

## `paper` — simulated, permanently

```python
session = PaperTradingSession(strategy, history=df.iloc[:2000])
for ts, bar in df.iloc[2000:].iterrows():
    session.on_new_bar(ts, bar)

session.reconcile(backtest.targets)     # must be zero mismatches
```

`PaperBroker.connect()` raises `NotImplementedError` and always will. There is
no live order path, no credential handling and no exchange client anywhere in
this package.

---

## `naive` — the tutorial backtest, bugs included

```python
vectorized_backtest(prices, signal, shift=0, cost_bps=0.0)
```

A teaching exhibit, not a tool. It warns every time it produces an unachievable
number. `shift=0` is the look-ahead bug; `cost_bps` is charged **per side**, so
a 24 bps round trip is `cost_bps=12`.

`compare_shift_effect` puts a number on how much of your "profit" was
look-ahead.
