# Algorithmic & AI-Driven Trading — a four-day workshop

**Hands-on workshop for engineers and CS students.** Solid Python assumed, no
finance background required. Everything runs offline on a normal laptop, and in
Google Colab.

📖 **[Workshop site →](https://berdakh.github.io/ROBT407/)** ·
[Handbook](https://berdakh.github.io/ROBT407/handbook.html) ·
[Day plan](https://berdakh.github.io/ROBT407/plan.html) ·
[Tool ecosystem](https://berdakh.github.io/ROBT407/guides/tools.html)

---

> ## ⚠️ Educational use only — not investment advice
>
> This workshop teaches a **methodology for evaluating trading strategies**. It
> recommends no strategy, asset or trade, and makes no performance claims you
> should act on.
>
> - **Paper trading only.** Nothing here connects to a live-money account. There
>   is no live order path in the package; `PaperBroker.connect()` raises
>   `NotImplementedError` and always will.
> - **Most retail algorithmic trading strategies lose money after costs.** That
>   is the base rate, and it is the reason this workshop exists.
> - **A good backtest is weak evidence.** It describes the past, filtered
>   through choices you made after seeing that past.
> - Past performance, simulated or real, does not predict future results.

---

## The idea

A student who runs a backtest and sees a 300% return has learned nothing except
how to fool themselves.

So the whole workshop is built around one question — **why is your backtest
lying to you?** — and every bias is something you *measure*, not something you
read about.

Here is the spine. One strategy (SMA 3/24, the best of a 22-combination grid
search), evaluated four ways:

| | total return | what changed |
|---|---:|---|
| **Day 1** | **+46,986%** | vectorised backtest, no shift, no costs |
| ↓ | **+406%** | indexing corrected — look-ahead removed |
| ↓ | **+16%** | 24 bps round trip added |
| **benchmark** | **+582%** | buy & hold, which required no code at all |

Day 1 ends with the first number. Day 2 produces the rest.

![The four-day arc](diagrams/svg/06-workshop-map.svg)

## Quick start

```bash
git clone https://github.com/berdakh/ROBT407.git
cd ROBT407
pip install -e ".[dev,notebooks]"

pytest                             # 160 tests, no network needed
python tools/run_notebooks.py      # all 13 notebooks execute
jupyter notebook notebooks/
```

Sample data is committed, so everything runs offline immediately. Or click the
**Open in Colab** badge on any notebook — the first cell sets itself up.

## What's here

```
algotrade/        the library the notebooks import (~2,000 readable lines)
notebook_src/     notebook SOURCES (.py) — edit these, never the .ipynb
notebooks/        generated notebooks, with Colab badges
diagrams/src/     diagram sources — SVG is generated from code
data/synthetic/   six committed datasets with known ground truth
docs/             the GitHub Pages site, including the handbook
tests/            160 tests, no network, no market data feed
```

Notebooks and diagrams are **generated from reviewable sources**. CI checks they
stay in sync, so editing a `.ipynb` directly will be overwritten.

## The one piece of engineering that matters

Look-ahead bias is **structurally impossible** here, not merely discouraged.

The engine never hands a strategy the full price series. It hands it a
`MarketView` whose columns are physically truncated at the current bar:

```python
view.close[-1]          # the current bar     fine
view.close[-20:]        # the last 20 bars    fine
view.close[view.i + 1]  # tomorrow            LookAheadError
```

There is no `.shift(1)` to remember, because the future is not in the room.
`tests/test_lookahead_guard.py` proves it with four strategies that deliberately
cheat and must all crash.

## Two findings, measured rather than assumed

**Look-ahead bias won 40 out of 40 trials** across every dataset — including ones
generated with no signal at all. That universality is the diagnostic: a real edge
does not work on pure noise.

**Optimistic fill timing won only 22 of 40** — a coin flip. It is unrealistic but
not systematically inflating. *Unrealistic* and *inflated* are different
properties, and only the second kind fools you. The workshop teaches them
separately for that reason.

## The four days

| day | notebooks | what happens |
|---|---|---|
| **1** | 01–03 | Market data, returns, and a strategy reporting +46,986% |
| **2** | 04–06 | The event-driven engine, costs, and the full bias catalogue |
| **3** | 07–10 | Causal features, ML signals, walk-forward validation, honest evaluation |
| **4** | 11–13 | Risk sizing, paper trading, capstone |

Full schedule on the [day plan](https://berdakh.github.io/ROBT407/plan.html).

## Data

Six synthetic datasets are committed, each generated from a documented
stochastic process with its parameters recorded in `data/synthetic/manifest.json`.
Two of them (`gbm`, `pure_noise`) have **no exploitable signal by construction** —
they are the control group, and running your method on them is the single most
valuable habit in the workshop.

**No real market data is committed.** Outbound access to crypto exchanges was
blocked by network policy in the environment this was built in, so a genuine
snapshot could not be fetched and verified — and generating something
realistic-looking and labelling it "BTC" would be fabricating data, which is the
exact failure mode this workshop spends four days warning about.

`scripts/fetch_market_data.py` downloads a real snapshot from a public, keyless
endpoint. Nothing in the required path depends on it. See
[`data/README.md`](data/README.md).

## Working on it

| change | edit | then run |
|---|---|---|
| a notebook | `notebook_src/*.py` | `python tools/build_notebooks.py` |
| a diagram | `diagrams/src/*.py` | `python scripts/render_diagrams.py` |
| the datasets | `algotrade/synthetic.py` | `python scripts/build_datasets.py` |

```bash
pytest
python tools/build_notebooks.py --check
python scripts/render_diagrams.py --check
python tools/run_notebooks.py
```

## Licence

MIT. See [`LICENSE`](LICENSE).
