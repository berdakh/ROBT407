#!/usr/bin/env python3
"""Regenerate every SVG diagram from its source, and optionally rasterise a preview.

    python scripts/render_diagrams.py            # rebuild diagrams/svg/*.svg
    python scripts/render_diagrams.py --png OUT  # also write PNG previews to OUT

Diagrams are code. Nothing here is downloaded or hand-edited, so CI can check
that the committed SVGs still match their sources.

The PNG path exists only so a human (or an agent) can actually LOOK at the
result. It resolves the CSS custom properties to literal light-mode colours
first, because most rasterisers do not implement `var()`.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "diagrams" / "src"
# GitHub Pages serves from docs/, so it cannot reach ../diagrams/. The SVGs are
# published into the site as well, and --check verifies both copies match the
# source. Duplicated bytes, single source of truth.
SITE_COPY = ROOT / "docs" / "assets" / "diagrams"
sys.path.insert(0, str(SRC))

MODULES = [
    ("d01_backtest_loop", "01-backtest-loop.svg"),
    ("d02_lookahead_trap", "02-lookahead-trap.svg"),
    ("d03_walk_forward", "03-walk-forward-vs-naive-cv.svg"),
    ("d04_order_lifecycle", "04-order-lifecycle.svg"),
    ("d05_position_sizing", "05-risk-and-position-sizing.svg"),
    ("d06_workshop_map", "06-workshop-map.svg"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="directory for PNG previews")
    ap.add_argument("--check", action="store_true", help="fail if output differs from committed")
    args = ap.parse_args()

    import theme

    changed = []
    for mod_name, filename in MODULES:
        mod = importlib.import_module(mod_name)
        content = mod.build()
        theme.validate(content)
        dest = theme.OUT_DIR / filename

        site_dest = SITE_COPY / filename

        if args.check:
            if not dest.exists() or dest.read_text() != content:
                changed.append(filename)
            if not site_dest.exists() or site_dest.read_text() != content:
                changed.append(f"docs/assets/diagrams/{filename}")
            continue

        theme.write(filename, content)
        SITE_COPY.mkdir(parents=True, exist_ok=True)
        site_dest.write_text(content)
        print(f"  wrote diagrams/svg/{filename} (+ site copy)")

        if args.png:
            out_dir = Path(args.png)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                import cairosvg
            except ImportError:
                print("    (cairosvg not installed; skipping PNG preview)")
                continue
            for mode in ("light", "dark"):
                png = out_dir / filename.replace(".svg", f"-{mode}.png")
                cairosvg.svg2string = None  # noqa: F841  (silence linters)
                cairosvg.svg2png(
                    bytestring=theme.to_static(content, mode).encode(),
                    write_to=str(png),
                    output_width=1200,
                    background_color="#ffffff" if mode == "light" else "#0d1117",
                )
                print(f"    preview {png}")

    if args.check:
        if changed:
            print("Diagrams are out of sync with their sources: " + ", ".join(changed))
            print("Run: python scripts/render_diagrams.py")
            return 1
        print(f"All {len(MODULES)} diagrams are in sync with their sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
