# Data

## What is here

`synthetic/` — six generated OHLCV datasets, **committed to the repository** so
every notebook runs the moment you clone it. Each one is built by
`algotrade.synthetic.generate` from a documented stochastic process, and
`manifest.json` records the exact parameters used. That file is the answer key:
when an exercise asks "does this strategy have an edge?", the ground truth is
written down.

| file | process | what it is for |
|---|---|---|
| `gbm.csv` | geometric Brownian motion | The control. Returns are i.i.d., so **no** strategy has an edge. Anything that profits here profits by luck. |
| `momentum.csv` | AR(1) returns, φ = 0.06 | A real, known, positive autocorrelation. Trend following *should* work. |
| `mean_reverting.csv` | Ornstein–Uhlenbeck log price, half-life ≈ 12 bars | Reversion *should* work; trend following should lose. |
| `regime_switching.csv` | two-state Markov volatility | Calm and turbulent regimes. One Sharpe ratio averages them into a meaningless number. |
| `pure_noise.csv` | driftless random walk | Ground truth: nothing works. Used for the data-snooping lesson. |
| `trending_with_crash.csv` | strong drift plus abrupt drawdowns | Sharpe looks respectable, max drawdown does not. Two years of hourly bars. |

All are hourly bars, UTC, timestamped at the **bar open**.

Regenerate with `python scripts/build_datasets.py`. Output is deterministic, and
CI checks that the committed files still match the generator.

## The seeds were chosen deliberately

Each dataset's seed was picked so the realised path actually *displays* the
lesson its generator is meant to teach. That is legitimate for a teaching
fixture and completely illegitimate for a backtest — picking the seed that
makes your equity curve look good is precisely the data snooping that Day 2 is
about. The irony is intentional and notebook 06 points at it.

## What is NOT here: real market data

**No real OHLCV snapshot is committed to this repository.** The environment
this workshop was built in had outbound access to crypto exchanges blocked by
network policy, so a genuine snapshot could not be downloaded and verified. The
alternative — generating something realistic-looking and labelling it "BTC" —
would be fabricating data, which is the exact failure mode the workshop spends
four days warning about.

So instead:

* `scripts/fetch_market_data.py` downloads a real snapshot from a public,
  keyless exchange endpoint. Run it once if your network permits:

  ```bash
  python scripts/fetch_market_data.py --symbol BTCUSDT --interval 1h --days 365
  ```

  Output lands in `real/` and is git-ignored.

* Every notebook runs to completion on the synthetic data without it. Real data
  is an enrichment, not a dependency.

**Run it at least once anyway.** Real data has gaps, duplicated bars, exchange
outages that look like flat prices, and single ticks that move 40%. Notebook 01
teaches you to find all of those, and the synthetic data is too clean to
practise on.
