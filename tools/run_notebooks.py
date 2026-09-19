#!/usr/bin/env python3
"""Execute every notebook and fail loudly on the first error.

    python tools/run_notebooks.py               # all of them
    python tools/run_notebooks.py 03 07         # only matching ones

This is the check that matters: a workshop notebook that does not run is worse
than no notebook, because the student assumes the error is theirs.

Execution happens from the repository root with a plain Python kernel, so the
relative data paths the notebooks use resolve exactly as they do for a student
who started Jupyter in the right place.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def run_notebook(path: Path) -> tuple[bool, str, float]:
    """Execute a notebook's code cells in one shared namespace."""
    nb = json.loads(path.read_text())
    namespace: dict = {"__name__": "__main__"}
    started = time.time()

    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(
            line if line.endswith("\n") else line + "\n" for line in cell["source"]
        )
        # The Colab bootstrap shells out to pip; skip it and set up directly.
        if "google.colab" in source:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            sys.path.insert(0, str(ROOT))
            import algotrade

            namespace.update({"plt": plt, "np": np, "pd": pd, "algotrade": algotrade, "Path": Path})
            pd.set_option("display.width", 110)
            continue
        try:
            exec(compile(source, f"{path.name}[cell {index}]", "exec"), namespace)
        except Exception:
            return False, traceback.format_exc(limit=6), time.time() - started

    return True, "", time.time() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*", help="only run notebooks whose name contains these")
    args = ap.parse_args()

    import os

    os.chdir(ROOT)
    import matplotlib

    matplotlib.use("Agg")

    notebooks = sorted(NB_DIR.glob("*.ipynb"))
    if args.filters:
        notebooks = [p for p in notebooks if any(f in p.name for f in args.filters)]
    if not notebooks:
        print("No notebooks matched.", file=sys.stderr)
        return 1

    failures = []
    for path in notebooks:
        ok, error, elapsed = run_notebook(path)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {path.name:<34} {elapsed:6.1f}s")
        if not ok:
            failures.append((path.name, error))

    if failures:
        print(f"\n{len(failures)} notebook(s) failed:\n")
        for name, error in failures:
            print(f"--- {name} " + "-" * (60 - len(name)))
            print(error)
        return 1

    print(f"\nAll {len(notebooks)} notebooks executed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
