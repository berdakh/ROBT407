#!/usr/bin/env python3
"""Build notebooks/*.ipynb from the reviewable sources in notebook_src/*.py.

    python tools/build_notebooks.py           # build
    python tools/build_notebooks.py --check   # fail if the .ipynb files are stale

WHY THE SOURCES ARE .py FILES
Notebook JSON does not diff, does not merge, and cannot be reviewed. The
sources here are jupytext "percent" format: ordinary Python files where
``# %%`` starts a code cell and ``# %% [markdown]`` starts a prose cell. They
read as code, review as code, and are the only thing you should ever edit.

Two things are injected into every notebook so they cannot drift apart:

1. An **Open in Colab** badge, pointing at this notebook's path in the repo.
2. A **bootstrap cell** that actually makes the notebook run on Colab --
   clones the repository, installs the package, and changes into the repo root
   so relative data paths resolve. On a local machine it just fixes sys.path.

Outputs are stripped: a committed notebook has no execution counts and no
stored results, so the diff is the source change and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "notebook_src"
OUT_DIR = ROOT / "notebooks"

GITHUB_REPO = "berdakh/ROBT407"
#: Branch the Colab badges point at. Colab fetches from GitHub, so this must be
#: a ref that exists there -- use the default branch, not a feature branch that
#: will be deleted after merge.
DEFAULT_REF = "master"

BOOTSTRAP = '''# --- Setup: run this cell first -------------------------------------------
# On Google Colab this clones the workshop and installs it. On your own
# machine it only makes sure the repository root is importable. Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/{repo}.git"
CLONE_DIR = Path("/content/algotrade-workshop")

if "google.colab" in sys.modules:
    if not CLONE_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "{ref}", REPO_URL, str(CLONE_DIR)],
            check=True,
        )
    os.chdir(CLONE_DIR)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
    root = CLONE_DIR
else:
    # Walk up from wherever Jupyter was started until we find the package.
    root = Path.cwd()
    while root != root.parent and not (root / "algotrade").is_dir():
        root = root.parent
    if not (root / "algotrade").is_dir():
        raise RuntimeError(
            "Could not find the algotrade package. Start Jupyter from the "
            "repository root, or run: pip install -e ."
        )
    os.chdir(root)

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import algotrade

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 24)
plt.rcParams.update({{"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.3}})

print(f"algotrade {{algotrade.__version__}}  |  working directory: {{Path.cwd()}}")
print(f"synthetic data present: {{(Path.cwd() / 'data' / 'synthetic' / 'gbm.csv').exists()}}")
'''

DISCLAIMER_MD = """> ### ⚠️ Educational use only — not investment advice
>
> This notebook teaches a **methodology for evaluating trading strategies**. It
> does not recommend any strategy, asset or trade, and it reports no
> performance claims you should act on.
>
> - **Paper trading only.** Nothing in this workshop connects to a live-money
>   account. There is no live order path in the code.
> - **Most retail algorithmic strategies lose money after costs.** That is the
>   base rate, and it is the reason this workshop exists.
> - **A good backtest is weak evidence.** It describes the past, filtered
>   through choices you made after seeing that past."""


def _badge_cell(notebook_name: str, ref: str) -> dict:
    url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{ref}/notebooks/{notebook_name}"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": (
            f'<a href="{url}" target="_parent">'
            f'<img src="https://colab.research.google.com/assets/colab-badge.svg" '
            f'alt="Open In Colab"/></a>'
        ).split("\n"),
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.rstrip("\n").split("\n"),
    }


def _markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.rstrip("\n").split("\n")}


def _normalise(nb: dict) -> dict:
    """Strip everything that makes notebook diffs unreadable."""
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5
    for cell in nb["cells"]:
        cell.pop("id", None)
        cell["metadata"] = {}
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        if isinstance(cell["source"], str):
            cell["source"] = cell["source"].split("\n")
    return nb


def build_one(src: Path, ref: str) -> tuple[str, str]:
    """Return (output filename, notebook JSON text)."""
    import jupytext

    nb = jupytext.reads(src.read_text(), fmt="py:percent")
    nb = json.loads(jupytext.writes(nb, fmt="ipynb"))

    out_name = src.stem + ".ipynb"
    injected = [
        _badge_cell(out_name, ref),
        _markdown_cell(DISCLAIMER_MD),
        _code_cell(BOOTSTRAP.format(repo=GITHUB_REPO, ref=ref)),
    ]
    # The source's own first cell is its title; keep it above the boilerplate.
    cells = nb["cells"]
    if cells and cells[0]["cell_type"] == "markdown":
        nb["cells"] = [cells[0]] + injected + cells[1:]
    else:
        nb["cells"] = injected + cells

    nb = _normalise(nb)
    return out_name, json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if outputs are stale")
    ap.add_argument("--ref", default=DEFAULT_REF, help="git ref the Colab badges point at")
    args = ap.parse_args()

    sources = sorted(SRC_DIR.glob("*.py"))
    if not sources:
        print(f"No sources found in {SRC_DIR}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    stale = []
    for src in sources:
        name, text = build_one(src, args.ref)
        dest = OUT_DIR / name
        if args.check:
            if not dest.exists() or dest.read_text() != text:
                stale.append(name)
        else:
            dest.write_text(text)
            print(f"  notebooks/{name}")

    # An orphaned .ipynb whose source was deleted is also out of sync.
    expected = {src.stem + ".ipynb" for src in sources}
    orphans = {p.name for p in OUT_DIR.glob("*.ipynb")} - expected
    if orphans:
        if args.check:
            stale.extend(sorted(orphans))
        else:
            for name in sorted(orphans):
                (OUT_DIR / name).unlink()
                print(f"  removed orphan notebooks/{name}")

    if args.check:
        if stale:
            print("Notebooks are out of sync with notebook_src/: " + ", ".join(sorted(stale)))
            print("Run: python tools/build_notebooks.py")
            return 1
        print(f"All {len(sources)} notebooks are in sync with their sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
