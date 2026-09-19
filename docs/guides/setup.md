---
title: Setup
layout: default
nav_order: 1
parent: Guides
---
# Setup

Three ways to run the workshop. All of them work offline once set up, and none
of them need a paid API, a GPU or a broker account.

---

## Option 1 — Google Colab (nothing to install)

Every notebook carries an **Open in Colab** badge. Click it, then run the first
code cell: it clones the repository, installs the package, and changes into the
repository root so the relative data paths resolve.

```
[ Open in Colab ]  ->  run the first cell  ->  Runtime > Run all
```

That first cell is idempotent — re-running it is safe. On a local machine it
does nothing except make sure the repository root is importable.

**Caveats:** Colab sessions expire, and a fresh session re-clones (about 15
seconds). Save your own work to Drive or download the notebook before closing.

---

## Option 2 — Local install (recommended for the full workshop)

```bash
git clone https://github.com/berdakh/ROBT407.git
cd ROBT407

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev,notebooks]"
```

Verify:

```bash
pytest                             # 174 tests, no network required
python tools/run_notebooks.py      # executes all 13 notebooks
jupyter notebook notebooks/
```

If `pytest` passes, everything is working. The tests deliberately require no
network and no market data feed.

**Start Jupyter from the repository root**, not from `notebooks/`. The notebooks
use relative paths like `data/synthetic/gbm.csv`. (The bootstrap cell walks up
the directory tree to find the package, so it will usually recover either way —
but starting in the right place avoids the question.)

---

## Option 3 — Fully offline

After the initial clone, nothing needs the network:

- All six datasets are **committed to the repository** in `data/synthetic/`.
- The tests never open a socket, and `tests/test_offline_and_safety.py` enforces
  that by monkey-patching `socket.connect` to raise.
- No notebook in the required path makes a network call.

The only script that touches the network is `scripts/fetch_market_data.py`,
which is optional. Its parsing and every error path are covered by
`tests/test_fetch_script.py` against a mocked exchange, so even that runs
offline — though whether the live endpoint still returns the assumed shape is
something only a real call can tell you.

---

## Requirements

| | |
|---|---|
| Python | 3.10 or later |
| RAM | 4 GB is plenty |
| Disk | ~60 MB including data |
| GPU | not used anywhere |
| Accounts | none |

Core dependencies: numpy, pandas, scipy, scikit-learn, matplotlib. Notebook
tooling: jupyter, nbformat, jupytext.

---

## Optional: real market data

The committed data is **synthetic**. That is deliberate — it has known ground
truth, so exercises can be graded against the real answer.

To also work with real prices:

```bash
python scripts/fetch_market_data.py --symbol BTCUSDT --interval 1h --days 365
```

This uses a **public, keyless** exchange endpoint. It sends no credentials,
places no orders, and has no code path that could. Output lands in `data/real/`
and is git-ignored.

<div class="callout" markdown="1">
**If it fails, nothing is broken.** Corporate proxies and many cloud sandboxes
block exchange APIs. Every notebook runs to completion on the synthetic data
without it.

**Run it once anyway if you can.** Real data has gaps, duplicated bars, exchange
outages that look like flat prices, and single ticks that move 40%. Notebook 01
teaches you to find all of those, and the synthetic data is too clean to
practise on.
</div>

---

## Working on the repository

Notebooks and diagrams are **generated**. Never edit them directly.

| you want to change | edit | then run |
|---|---|---|
| a notebook | `notebook_src/*.py` | `python tools/build_notebooks.py` |
| a diagram | `diagrams/src/*.py` | `python scripts/render_diagrams.py` |
| the datasets | `algotrade/synthetic.py` | `python scripts/build_datasets.py` |

CI checks that the generated files still match their sources, so an edit to a
`.ipynb` will be overwritten and an out-of-sync commit will fail the build.

```bash
pytest                                       # the library
python tools/build_notebooks.py --check      # notebooks in sync?
python scripts/render_diagrams.py --check    # diagrams in sync?
python tools/run_notebooks.py                # do they all still execute?
```

To see a diagram while working on it:

```bash
pip install cairosvg
python scripts/render_diagrams.py --png /tmp/preview
```

That resolves the CSS custom properties to literal colours and writes light and
dark PNGs, because most rasterisers do not implement `var()`.

---

## How the site is published

`.github/workflows/pages.yml` builds `docs/` with Jekyll and deploys it on every
push to `master`. The workflow passes `enablement: true` to
`actions/configure-pages`, so it switches Pages on by itself the first time it
runs.

If a deploy ever fails at the `configure-pages` step with

```
Get Pages site failed. Please verify that the repository has Pages enabled
and configured to build using GitHub Actions ... Error: Not Found
```

then Pages is off and the workflow could not turn it on — usually a permissions
problem. Set it manually at **Settings → Pages → Source: GitHub Actions** and
re-run the workflow.

---

## Troubleshooting

**`FileNotFoundError: data/synthetic/gbm.csv`** — Jupyter was started somewhere
other than the repository root. Restart it from the root, or re-run the
bootstrap cell.

**`ModuleNotFoundError: algotrade`** — run `pip install -e .` from the repository
root, or re-run the bootstrap cell.

**`LookAheadError`** — working as intended. Your strategy tried to read a bar it
has not reached. The message names the exact index. See
[notebook 04](https://github.com/berdakh/ROBT407/blob/master/notebooks/04_event_driven_backtest.ipynb).

**`InsufficientHistoryError`** — your strategy asked for more history than has
elapsed. Guard with `if len(view) < self.warmup: return None`.

**`NaiveBacktestWarning`** — `vectorized_backtest` is telling you it is producing
an unachievable number. That is its job; it is a teaching exhibit, not a tool.
