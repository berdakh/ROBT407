---
layout: default
title: The Handbook
---

<link rel="stylesheet" href="{{ '/assets/style.css' | relative_url }}">

# The Handbook

**Background reading. No code.**

This is the conceptual half of the workshop: how markets actually work, what the
numbers everyone quotes really mean, and why most strategies fail. Read the
relevant section before each day, or read it straight through in an evening.

<div class="callout danger" markdown="1">
**Educational use only — not investment advice.** Nothing here recommends a
strategy, asset or trade. Most retail algorithmic strategies lose money after
costs.
</div>

---

## Contents

1. [What a market actually is](#1-what-a-market-actually-is)
2. [How an order really executes](#2-how-an-order-really-executes)
3. [What a backtest is, and what it is not](#3-what-a-backtest-is-and-what-it-is-not)
4. [What Sharpe really means](#4-what-sharpe-really-means)
5. [What drawdown really means](#5-what-drawdown-really-means)
6. [Why most strategies fail](#6-why-most-strategies-fail)
7. [Who you are trading against](#7-who-you-are-trading-against)
8. [How professionals actually work](#8-how-professionals-actually-work)
9. [A glossary](#9-a-glossary)

---

## 1. What a market actually is

A market is a queue.

Strip away the charts and a modern exchange is a **limit order book**: a sorted
list of offers to buy and offers to sell. The highest buy offer is the **bid**;
the lowest sell offer is the **ask**. The gap between them is the **spread**.

Nothing happens until two orders meet. There are two ways to make that happen:

- A **limit order** joins the queue at a price you name. It might never fill.
- A **market order** crosses the spread immediately, taking whatever the queue
  offers. It always fills, at a price you did not choose.

That distinction generates most of the cost structure of trading. If you post a
limit order and wait, you are a **maker** — you provide liquidity, and exchanges
often pay you a small rebate. If you cross the spread, you are a **taker** — you
consume liquidity, and you pay a fee for it.

### The price on the chart does not exist

A candlestick chart shows you the **last traded price**. You cannot trade at it.
At any moment there are two prices — the bid and the ask — and you get the worse
one. The mid-price shown on a chart is an average of two numbers, and it is a
price at which nobody has agreed to do anything.

This is not a technicality. It is the first component of the cost that converts
profitable backtests into unprofitable strategies.

### What a bar hides

An hourly OHLCV bar compresses thousands of individual trades into five numbers.
What it throws away:

- **The path.** A bar that opens at 100 and closes at 100 may have touched 95
  and 110. Your stop-loss cares enormously about this; the bar does not record it.
- **The order.** If a bar's high and low both breach your stop and your target,
  OHLC data cannot tell you which came first. Any backtest that assumes the
  favourable ordering is lying by a small, systematic, compounding amount.
- **Liquidity.** A bar reports total volume, not whether you could have traded
  any meaningful size at any particular price.

### Crypto specifically

This workshop uses crypto for four practical reasons:

1. **Free data.** Public APIs, no subscription, no licence.
2. **24/7.** No market open, no close, no holidays, no overnight gaps — one
   fewer thing to get wrong.
3. **No survivorship bias in a single pair.** BTC/USD has traded continuously;
   you are not studying a pre-filtered universe.
4. **No corporate actions.** No splits, dividends or mergers to adjust for.

What crypto has *more* of: volatility, fat tails, exchange outages, wash trading,
and genuinely dreadful liquidity outside the top few pairs. The methodology
transfers to equities and futures; the specific numbers do not.

---

## 2. How an order really executes

Between "I want to buy" and "I own it" there are six stages, and each one costs
you time or money.

![The order lifecycle]({{ '/assets/diagrams/04-order-lifecycle.svg' | relative_url }})

### The four costs

**Commission.** The exchange's fee, charged on notional, every trade. Retail
crypto takers pay roughly 10 basis points (0.10%). Makers pay less, sometimes
negative. Assuming you will always be the maker is wishful thinking.

**The spread.** You buy at the ask and sell at the bid. Half the spread each way.
On a liquid BTC perpetual this is around 1 basis point. On an illiquid altcoin it
can be 50.

**Slippage.** The price moves between your decision and your fill. You are slower
than the people whose entire business is being fast.

**Market impact.** Your own order moves the price against you. A small retail
order does not; a large one eats through the book. Impact grows roughly with the
**square root** of your size relative to available volume — which is why a
strategy that works with \$10,000 may not work with \$10,000,000.

### The number to memorise

For a small retail taker on a liquid crypto pair, a **round trip costs about 24
basis points.**

That sounds negligible until you multiply by turnover:

| turnover | annual cost drag |
|---:|---:|
| 2×/year | 0.2% |
| 52×/year | 6% |
| 200×/year | 24% |
| 650×/year | 78% |

A strategy trading 650 times a year must earn 78% annually **before costs** just
to break even. Compute this before you write a backtest. It kills most ideas in
thirty seconds, and it is the single most valuable habit in this handbook.

### Why professionals can trade signals you cannot

Firms profit from signals far weaker than anything in this workshop. Their
advantage is usually **not** a better signal. It is:

- They are makers, earning the spread rather than paying it.
- They have fee rebates from volume.
- They are co-located, so their slippage is measured in microseconds.

Their break-even cost is a small fraction of yours. A signal that is worth
2 basis points per trade is a business for them and a slow loss for you. The
signal is the same. The cost structure is not.

---

## 3. What a backtest is, and what it is not

**A backtest is a simulation of a decision procedure applied to historical data.**
That is all it is.

It is *not* a prediction, and it is *not* evidence that a strategy works. It is
weak evidence at best, because of one structural problem: **you chose the
strategy after seeing the data.** Even if you never explicitly optimised, you
know what happened in that period. You know 2022 was bad for crypto. That
knowledge is in your strategy whether you put it there deliberately or not.

### The four questions

Before believing any backtest — yours most of all:

1. **Could the strategy have known everything it used?**
2. **How many things were tried before this one was reported?**
3. **What did the trades cost?**
4. **Was it evaluated on data used to choose it?**

Most published backtests fail at least two.

### Look-ahead bias

Using information that did not exist at the moment of the decision. It is the
most damaging bias because it is the only one that works **every time, on every
dataset**.

That universality is also how you detect it. A real edge does not work on data
with no signal in it. Run your method on a random walk: if it still makes money,
you have a leak, not an edge.

Where it hides, beyond the obvious missing shift:

- Normalising with full-sample statistics — `(x - x.mean()) / x.std()` uses the
  mean of the entire series, including the future.
- Backfilling missing values, which copies the future backwards.
- Interpolating a gap, which uses the value on the far side of it.
- Fitting a scaler on the whole dataset before splitting.
- Choosing a stop-loss level after looking at the chart.

### Data snooping

Trying many strategies and reporting the best without adjusting for how many you
tried. This is the most insidious bias because **it feels like diligence.**
Testing 200 variants *is* thorough. The problem is entirely in the reporting:
the best of 200 random strategies has an impressive Sharpe ratio by construction.

The correction is the **deflated Sharpe ratio**: instead of asking "is this
Sharpe greater than zero?", ask "is it greater than the best I would expect from
$N$ tries at nothing?"

Counting $N$ honestly is harder than it sounds. It includes every parameter you
abandoned, every indicator you dropped, every time you changed the date range
because results looked bad — and, if you are using a strategy from a paper that
tested 50 variants, theirs too.

### Overfitting

Fitting the noise in the training data, which by definition does not repeat.
Detected by splitting chronologically: choose parameters on the first part,
evaluate on the second. In practice the correlation between in-sample and
out-of-sample performance is frequently near **zero**, which means the parameter
search accomplished nothing at all.

### Survivorship bias

Studying only the things that survived. Backtesting "buy the index constituents"
using today's constituent list excludes every company that went bankrupt. Crypto
is worse: thousands of tokens have gone to zero, and a "top 100 coins" backtest
run on today's top 100 is studying lottery winners and concluding that lottery
tickets are a good investment.

The fix is a **point-in-time universe** — the list as it stood on each historical
date, including everything that later died. That data is expensive and often
unavailable for crypto, which is a good reason to stick to single liquid assets
while learning.

### Regime dependence and sample selection

A single performance number averages over market conditions that are
qualitatively different. Always report the dispersion across sub-periods.

And be honest about how you chose the period. "I focused on the recent regime"
after seeing that the earlier one looked bad is sample selection, whether or not
it felt like a decision at the time.

---

## 4. What Sharpe really means

$$\text{Sharpe} = \frac{\text{mean excess return}}{\text{standard deviation}} \times \sqrt{N}$$

Return per unit of volatility, annualised. It is the most quoted number in the
industry and it is almost always quoted without the one thing that makes it
interpretable: **its error bar.**

### The uncertainty is enormous

The standard error of a Sharpe ratio estimated over $n$ bars is roughly
$\sqrt{(1 + \text{SR}^2/2)/n}$ in per-bar units. In practice:

| sample | can you distinguish SR = 0.5 from zero? | SR = 1.0? |
|---|---|---|
| 3 months | no | no |
| 1 year | no | no |
| 3 years | no | yes |
| 10 years | yes | yes |

**A strategy with a genuine Sharpe ratio of 0.5 needs about a decade of data
before you can confidently tell it apart from noise.** A six-month backtest
reporting a Sharpe of 2 has an error bar of roughly ±1.4 and therefore tells you
almost nothing.

This single fact explains most of the gap between backtested and live
performance across the entire industry.

### What Sharpe does not tell you

- **Whether the losses arrived all at once.** A strategy that bleeds slowly and
  a strategy that loses 40% in a week can have identical Sharpe ratios.
- **The shape of the tail.** A strategy that sells insurance has a superb Sharpe
  ratio until the one day it does not.
- **Whether you could have held it.** Sharpe is computed by someone who never
  panics.

### Calibration

| Sharpe (after costs, out of sample) | interpretation |
|---|---|
| < 0 | loses money |
| 0 – 0.5 | indistinguishable from noise at realistic sample sizes |
| 0.5 – 1.0 | respectable; a good long-only equity portfolio |
| 1.0 – 2.0 | genuinely good; professional fund territory |
| 2.0 – 3.0 | exceptional; expect scrutiny |
| > 3.0 | **look for the bug** |

If your backtest reports a Sharpe above 3, the overwhelmingly likely explanation
is look-ahead bias, a survivorship-filtered universe, or costs you forgot. That
is not cynicism; it is the base rate.

### Better metrics to lead with

- **Probabilistic Sharpe ratio** — the probability the true Sharpe exceeds zero.
- **Deflated Sharpe ratio** — the same, against a best-of-$N$-trials benchmark.
- **Fold-by-fold consistency** — how many walk-forward folds were positive, with
  a t-statistic. "Positive in 5 of 10 folds, t = −1.1" is a complete result.

---

## 5. What drawdown really means

**Maximum drawdown** is the worst peak-to-trough decline. It is the number that
decides whether you are still trading next year.

### The arithmetic is not symmetric

| loss | gain needed to recover |
|---:|---:|
| −10% | +11% |
| −25% | +33% |
| −50% | +100% |
| −75% | +300% |
| −90% | +900% |

A 50% drawdown is not "twice as bad" as a 25% one. It is four times harder to
climb out of. And this is arithmetic, before you account for the fact that most
people stop trading long before they reach the bottom.

### Duration matters as much as depth

A 30% drawdown lasting three weeks is survivable. A 30% drawdown lasting two
years is a different experience entirely: you have to keep executing a strategy
that has been wrong for 24 consecutive months, while your own conviction erodes
and anyone whose money you manage asks pointed questions.

Always report **time underwater** alongside depth.

### Drawdown is a psychological limit, not just a financial one

The honest question is not "what is the maximum drawdown I can afford?" It is
**"what is the maximum drawdown I will actually hold through?"** Those are
different numbers, and the second one is smaller than you think.

This is why volatility targeting and drawdown limits exist. They make a strategy
*survivable*, usually at the cost of some return. That trade is normally worth
making — being alive next year matters more than being optimal this year — but
it is a trade, not free insurance, and anyone presenting a drawdown limit as pure
upside is not telling you the whole story.

---

## 6. Why most strategies fail

In rough order of how often each is the actual cause.

### 1. The edge was never there

The backtest was contaminated. Look-ahead bias, data snooping, or a leaky
feature. The strategy never worked; it only appeared to.

### 2. The edge was real and smaller than the costs

This is more common than people expect, and it is the least intuitive failure
mode. A model can genuinely predict the market — statistically significant,
reproducible, out of sample — and still lose money, because the edge is 2 basis
points and the round trip costs 24.

**A real edge and a profitable strategy are different things.** Confusing them
is why so many people conclude that machine learning does not work on markets,
when what actually happened is that it worked and did not matter.

### 3. The edge decayed

Markets adapt. A pattern that persists is a pattern someone is losing money to,
and they eventually stop. Published anomalies decay measurably after publication.
Anything you can find with public data and a laptop has probably been found.

### 4. The regime changed

The strategy worked in a market that no longer exists. Low-volatility strategies
fitted to a calm period. Trend followers fitted to a bull market. A single
backtest over a single regime cannot distinguish "this works" from "this worked
in this particular weather".

### 5. The implementation was wrong

The live system did not do what the backtest did. Different warm-up, unhandled
duplicate bars, a stale feed, state lost on restart. This is why reconciliation
between live and backtest signals is not optional.

### 6. Risk management failed

The edge was real, the costs were survivable, and a drawdown ended the account
before the strategy got paid. Almost always leverage.

### What to do about it

- Compute the cost budget before writing the backtest.
- Validate walk-forward, and run the same procedure on data with no edge.
- Count your trials honestly and deflate.
- Size positions so you survive being wrong for longer than you expect.
- Write down in advance what would make you stop.

---

## 7. Who you are trading against

Every trade has two sides. Being clear about who is on the other side of yours,
and why they are willing to take it, is the discipline that separates a
hypothesis from a chart pattern.

**Market makers.** They quote both sides and earn the spread. They are faster
than you, better capitalised, and they are the counterparty to most of your
trades. They are not trying to predict direction; they are trying to be flat by
the end of the day.

**Statistical arbitrage funds.** Teams of PhDs with better data, lower costs and
co-located hardware. If your signal is a simple function of public price history,
assume they have tested it.

**Hedgers and flow traders.** Miners selling to cover costs, institutions
rebalancing, funds meeting redemptions. These participants are **not trying to
make money on the trade**, which is exactly what makes them a plausible source of
edge — they are willing to pay for immediacy or for risk transfer.

**Other retail traders.** Slower and less informed than the first two groups.
Also the group you are most likely to be in.

### The question this forces

*"Why is someone willing to lose to me?"*

There are only a few legitimate answers:

- **You are supplying liquidity** when it is scarce, and being paid for it.
- **You are taking risk** that someone else needs to shed.
- **You are slower on purpose** — holding positions over horizons the fast
  players cannot afford to occupy.
- **You are looking somewhere nobody bothers with** — a market too small to
  interest a fund.

"My backtest looked good" is not on that list. If you cannot answer the question,
you do not have a hypothesis; you have a curve that went up.

---

## 8. How professionals actually work

Worth knowing, because the popular picture is wrong in specific ways.

**Most of the work is data.** Cleaning, aligning, handling corporate actions,
building point-in-time universes. The modelling is a small fraction of the
effort, and it is the fraction that tutorials spend all their time on.

**Simple models dominate.** Linear models with well-constructed features beat
deep networks on most financial prediction tasks, because the signal-to-noise
ratio is so low that flexible models mostly fit noise. If a neural network beats
your linear baseline by a lot, check for leakage before celebrating.

**Many small edges, not one big one.** A fund does not have *the* strategy. It
has hundreds of weak signals combined, each contributing a little and
diversifying the others.

**Costs are a first-class engineering problem.** Execution research — how to
trade without moving the market — is a whole discipline, and for many strategies
it is worth more than signal research.

**Risk management is not an afterthought.** Position limits, drawdown limits,
exposure limits, and an independent risk function with the authority to switch
things off.

**Most research fails, and that is normal.** Testing an idea and finding nothing
is the standard outcome. Teams are organised around it: hypotheses are recorded
before testing, trials are counted, and negative results are documented so nobody
repeats them.

That last point is the one worth importing into a student project. **"I found no
evidence of an edge" is a complete, professional, valuable result.**

---

## 9. A glossary

**Alpha** — return not explained by exposure to known risk factors. Often used
loosely to mean "the good bit".

**Ask** — the lowest price at which someone will currently sell. You buy here.

**Backtest** — a simulation of a decision procedure on historical data.

**Basis point (bp)** — one hundredth of a percent. 24 bps = 0.24%.

**Bid** — the highest price at which someone will currently buy. You sell here.

**Drawdown** — decline from a running peak. Max drawdown is the worst such
decline.

**Embargo** — samples skipped after a test window in cross-validation, to break
serial-correlation leakage.

**Information coefficient (IC)** — correlation between a prediction and the
realised outcome. Typical real values are 0.02–0.05.

**Look-ahead bias** — using information that did not exist at decision time.

**Maker / taker** — a maker posts a resting order and supplies liquidity; a taker
crosses the spread and consumes it. Takers pay more.

**Purging** — dropping training samples whose label horizon overlaps the test
period.

**Round trip** — buying and then selling. All costs are paid twice.

**Sharpe ratio** — annualised excess return divided by volatility.

**Slippage** — the difference between the price you expected and the price you got.

**Spread** — the gap between bid and ask.

**Survivorship bias** — studying only the assets that still exist.

**Turnover** — how many times a year you replace your position. Multiply by
round-trip cost to get your annual drag.

**Walk-forward** — refit on a window, test on what follows, roll forward. The
closest offline analogue of live operation.

---

<div class="callout warn" markdown="1">
**If you take one thing from this handbook**

Compute `turnover × round-trip cost ÷ 2` before you write the backtest, and run
your method on data with no signal in it before you believe the result.

Those two habits will save you more time than everything else here combined.
</div>

[← Back to the workshop]({{ '/' | relative_url }})
