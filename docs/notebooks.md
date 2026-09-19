---
title: Notebooks
layout: default
nav_order: 2
---

# The notebooks
{: .no_toc }

Thirteen notebooks in dependency order. Work them in sequence — each builds on
the last, and Day 2 only lands if you have actually felt Day 1.

1. TOC
{:toc}

---

## Open any of them in Colab

No install, no GPU, no paid anything. Click a badge and go.

| Day | # | Notebook | What it does | Runs in | Open |
|:--|:--|:--|:--|:--|:--|
| 1 | 01 | **Market data** | Timestamps, the four corrupting defects, resampling | ~1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/01_market_data.ipynb) |
| 1 | 02 | **Returns and risk** | Log vs simple, fat tails, Sharpe with its error bar | ~1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/02_returns_and_risk.ipynb) |
| 1 | 03 | **Your first strategy** | Builds the +46,986% result — and +2,119% on noise | ~1 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/03_first_strategy.ipynb) |
| 2 | 04 | **Event-driven backtest** | The engine that cannot lie; four cheaters, all blocked | ~2 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/04_event_driven_backtest.ipynb) |
| 2 | 05 | **Costs and slippage** | The collapse. Turnover × round trip ÷ 2 | ~8 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/05_costs_and_slippage.ipynb) |
| 2 | 06 | **The bias catalogue** | Seven biases, each constructed and measured | ~25 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/06_bias_catalogue.ipynb) |
| 3 | 07 | **Features and labels** | Causality proofs, stationarity, triple-barrier labels | ~2 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/07_features_and_labels.ipynb) |
| 3 | 08 | **ML signals** | Gross Sharpe +2.3, net −4.9. The ML worked; the strategy didn't | ~4 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/08_ml_signals.ipynb) |
| 3 | 09 | **Walk-forward validation** | 10/10 folds on a real edge, 4/10 on noise | ~3 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/09_walk_forward.ipynb) |
| 3 | 10 | **Honest evaluation** | One report card, with random strategies as the control | ~4 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/10_honest_evaluation.ipynb) |
| 4 | 11 | **Risk and sizing** | Vol targeting, drawdown limits, why leverage buys nothing | ~7 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/11_risk_and_sizing.ipynb) |
| 4 | 12 | **Paper trading** | The live loop, reconciliation, ten failure modes | ~5 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/12_paper_trading.ipynb) |
| 4 | 13 | **Capstone** | Your own strategy. Performance is worth **0%** | ~5 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/ROBT407/blob/master/notebooks/13_capstone.ipynb) |
{: .nb-table }

Runtimes are for a free CPU Colab runtime, excluding the one-off setup cell.

## What happens when you click

Colab fetches **only the notebook file** from GitHub — not the repository. So the
first code cell in every notebook does the rest:

```python
if "google.colab" in sys.modules:
    # clone the repo so algotrade/ and data/ exist
    # pip install -e .
    # chdir into it
```

It is safe to re-run, and on your own machine it only fixes `sys.path`. That is
why the same notebook works in both places.

{: .note }
> The first run takes about fifteen seconds while it clones and installs.
> Everything after that is fast, and nothing needs the network again — the
> sample data is in the repository.

## No GPU needed, ever

Nothing here trains a deep network. The heaviest model in the workshop is a
logistic regression on eighteen features, which fits in under a second. A free
CPU runtime is the *intended* environment, not a fallback.

{: .warning }
> **Educational use only — not investment advice.** Paper trading only. Nothing
> in these notebooks connects to a live-money account, and
> `PaperBroker.connect()` raises on purpose. Most retail algorithmic strategies
> lose money after costs.

## Running them locally instead

```bash
git clone https://github.com/berdakh/ROBT407.git
cd ROBT407
pip install -e ".[dev,notebooks]"
jupyter notebook notebooks/
```

Start Jupyter from the repository root, not from `notebooks/` — the notebooks
use relative paths like `data/synthetic/gbm.csv`. Full instructions on the
[setup page](guides/setup.html).

## They are generated, not hand-edited

The `.ipynb` files are built from reviewable `.py` sources in `notebook_src/`.
Edit those, then run:

```bash
python tools/build_notebooks.py
```

CI checks the two stay in sync, so an edit made directly to a notebook will be
overwritten. It also executes all thirteen on every push — a workshop notebook
that does not run is worse than no notebook, because the student assumes the
error is theirs.
