#!/usr/bin/env python3
"""Regenerate the committed synthetic datasets in ``data/synthetic/``.

Run from the repository root::

    python scripts/build_datasets.py

Output is deterministic: the same seeds always produce the same bytes, so CI
can check that the committed CSVs still match their source. If this script and
the committed data disagree, one of them changed and you need to know which.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algotrade.data import data_quality_report, validate_ohlcv  # noqa: E402
from algotrade.synthetic import DATASETS, generate  # noqa: E402

OUT = ROOT / "data" / "synthetic"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for name, kwargs in DATASETS.items():
        df, truth = generate(**kwargs)
        validate_ohlcv(df)

        # Round to realistic tick precision: cents for prices, 4dp for volume.
        # Rounding is monotonic, so low <= open/close <= high still holds.
        out = df.copy()
        for col in ("open", "high", "low", "close"):
            out[col] = out[col].round(2)
        out["volume"] = out["volume"].round(4)
        validate_ohlcv(out)
        path = OUT / f"{name}.csv"
        out.to_csv(path)

        report = data_quality_report(df, "1h")
        manifest[name] = {
            "file": path.name,
            "generator": truth.kind,
            "seed": truth.seed,
            "bars": truth.n_bars,
            "bars_per_year": truth.bars_per_year,
            "start": str(report["start"]),
            "end": str(report["end"]),
            "ground_truth": truth.params,
            "lesson": truth.notes,
        }
        print(f"  {name:22} {len(df):>6} bars -> {path.relative_to(ROOT)}")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nWrote manifest with {len(manifest)} datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
