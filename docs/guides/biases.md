---
title: Bias reference
layout: default
nav_order: 2
parent: Guides
---
# Bias reference

A one-page catalogue. Each entry: what it is, how it gets in, how to detect it,
how to prevent it.

![The look-ahead trap]({{ '/assets/diagrams/02-lookahead-trap.svg' | relative_url }})

---

## Look-ahead bias

**What:** the strategy used information that did not exist at decision time.

**Why it is the worst one:** it is the only bias that wins **every time, on every
dataset.** Measured across 40 dataset/seed combinations in this workshop,
look-ahead beat correct indexing 40 times out of 40 — including on data
generated with no predictability at all.

**How it gets in:**

| mechanism | looks like |
|---|---|
| Missing shift | `position = signal` instead of `signal.shift(1)` |
| Full-sample normalisation | `(x - x.mean()) / x.std()` |
| Backfilling | `df.fillna(method="bfill")` |
| Interpolation | `df.interpolate()` across a gap |
| Scaler fitted before split | `StandardScaler().fit_transform(X)` |
| Centred windows | `rolling(20, center=True)` |
| Parameters chosen after seeing the chart | a stop-loss at "obviously the right level" |

**Detect:** run your method on a dataset with no signal in it. If it makes money,
you have a leak. Also: an information coefficient above ~0.1, or a Sharpe above
3, is a leak until proven otherwise.

**Prevent:** use an engine that cannot show you the future. In `algotrade`,
`view.close[view.i + 1]` raises `LookAheadError`. There is no shift to forget
because the future is not in the room.

---

## Execution optimism

**What:** assuming fills you could not have got — trading at a closing price you
only learn at the close, or at the mid rather than crossing the spread.

**The subtlety worth knowing:** this is *not* look-ahead bias, and it does **not**
reliably inflate returns. Measured here, optimistic fill timing beat honest
timing 22 times out of 40 — a coin flip.

> Unrealistic and inflated are different properties. A result that is merely
> unrealistic is wrong in an unpredictable direction. A result contaminated by
> look-ahead is wrong in one direction, always. Only the second kind fools you
> systematically.

**Prevent:** fill at the *next* bar's open, and model the spread.

---

## Ignored costs

**What:** backtesting with zero or unrealistically low transaction costs.

**Detect:** compute `turnover × round-trip cost ÷ 2`. If that number is large
relative to your gross return, costs are doing the deciding.

**The arithmetic:**

| turnover | annual drag at 24 bps |
|---:|---:|
| 52×/year | 6% |
| 200×/year | 24% |
| 650×/year | 78% |

**Prevent:** compute the cost budget **before** the backtest. It kills most ideas
in thirty seconds, and it is the highest-value habit in this workshop.

**Note:** a parameter search run without costs actively *selects for* strategies
that trade too much to survive, because with costs off, trading more is free.

---

## Data snooping

**What:** trying many strategies and reporting the best, without adjusting for
how many you tried.

**Why it is insidious:** it feels like diligence. Testing 200 variants *is*
thorough. The problem is entirely in the reporting.

**The measurement:** 200 coin-flipping strategies on a provably unpredictable
dataset produce a best-of Sharpe ratio you would be delighted to publish. That
is not a bug in the strategies; it is what the maximum of 200 draws looks like.

**Detect:** compute the deflated Sharpe ratio with an honest trial count. Below
0.95 means no evidence.

**Counting honestly** — $N$ includes:

- every parameter combination tried and abandoned
- every indicator tested and dropped
- every time you changed the date range because results looked bad
- every Optuna trial
- if you are using a strategy from a paper that tested 50 variants, **their 50 too**

---

## Overfitting

**What:** fitting the noise in the training data, which does not repeat.

**Detect:** split chronologically, choose parameters on the first part, evaluate
on the second. Then compute the **correlation** between in-sample and
out-of-sample performance across your parameter grid. In this workshop's
measurements it is frequently near zero — which means the parameter search
accomplished nothing except choosing a number to be disappointed by.

**Prevent:** walk-forward validation with purging; prefer fewer parameters;
prefer simpler models. And report the fold-by-fold dispersion, not the mean.

---

## Survivorship bias

**What:** studying only the assets that survived.

**Crypto specifically:** thousands of tokens have gone to zero. A "top 100 coins"
backtest run on today's top 100 is studying lottery winners and concluding that
lottery tickets are a good investment.

**The measurement:** take 300 assets with *zero* true expected return, delist
those that fall below −90%, and measure the survivors. A premium appears from
nothing. There is no premium; there is only a filter applied after the fact.

**Prevent:** use a point-in-time universe including everything that later died.
For crypto this data is expensive and often unavailable — which is a good reason
to stick to single liquid assets while learning.

---

## Regime dependence

**What:** one performance number averaging over market conditions that are
qualitatively different.

**Detect:** split your sample into 6–8 sub-periods and score each. In this
workshop a strategy with a respectable overall Sharpe showed per-period values
ranging from **−5.78 to +2.09**.

**Prevent:** always report the dispersion across sub-periods. If the sign flips
between periods, the overall number is describing your sample, not your strategy.

---

## Sample selection

**What:** choosing the time period that makes your strategy look good. Usually
unconscious.

**The measurement:** the same strategy on the same asset with the same costs
produced Sharpe ratios from **−0.16 to +1.68** depending purely on which window
was selected. Pick whichever row you like.

**Prevent:** fix your evaluation period **before** you look at results. If you
change it, say so and increment your trial count.

---

## The checklist

Before believing any backtest — your own most of all:

1. Could the strategy have known everything it used? *(Run it on noise.)*
2. How many things did you try? *(Honestly. Deflate.)*
3. Was it evaluated on data used to choose it?
4. What did the trades cost? *(Turnover × round trip ÷ 2.)*
5. What is in your universe, and what is missing from it?
6. Does it work in every sub-period?
7. Did you choose the date range before or after seeing results?
8. Does it beat buy-and-hold, risk-adjusted, after costs?

`report_card()` automates 2, 3, 4, 6 and 8 as warnings. The rest require honesty,
which is the hard part and the one tooling cannot supply.
