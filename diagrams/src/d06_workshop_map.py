"""Diagram: the four-day arc, and where the spine of the course runs.

Exists so a student can see, on day one, that the collapse on day two is
planned rather than a failure on their part.
"""

from __future__ import annotations

from theme import arrow_defs, box, line, svg_close, svg_open, text, write

W, H = 960, 524


def build() -> str:
    p = svg_open(W, H, "The four-day arc",
                 "Day 1 builds a naive strategy that looks spectacular. Day 2 removes the "
                 "look-ahead bias and adds realistic costs, and the result collapses from "
                 "plus 46,986 percent to minus 71 percent. Days 3 and 4 rebuild honestly "
                 "with machine learning, walk-forward validation, risk sizing and paper "
                 "trading.")
    p += arrow_defs("muted")

    p.append(text(W / 2, 34, "The four-day arc", size=22, weight="650", anchor="middle"))
    p.append(text(W / 2, 57, "The collapse on Day 2 is the lesson, not an accident", size=13,
                  anchor="middle", cls="muted"))

    days = [
        ("DAY 1", "Data & a first strategy", ["market data, gaps, returns",
                                              "the naive backtest",
                                              "a grid search over 22 variants"], "accent"),
        ("DAY 2", "Why it was a lie", ["the event-driven engine",
                                       "costs, spread, slippage",
                                       "the bias catalogue"], "bad"),
        ("DAY 3", "ML, done correctly", ["causal features, honest labels",
                                         "walk-forward validation",
                                         "accuracy is not profit"], "accent"),
        ("DAY 4", "Risk & shipping", ["volatility targeting",
                                      "drawdown limits",
                                      "paper trading, capstone"], "good"),
    ]

    bw, gap = 208, 22
    x0, y0, bh = 46, 90, 136
    for idx, (day, title, bullets, color) in enumerate(days):
        x = x0 + idx * (bw + gap)
        p.append(box(x, y0, bw, bh, fill="surface", stroke=color, width=1.8))
        p.append(text(x + 18, y0 + 28, day, size=12, weight="700", fill=color, mono=True))
        p.append(text(x + 18, y0 + 50, title, size=14, weight="620"))
        for j, b in enumerate(bullets):
            p.append(text(x + 18, y0 + 74 + j * 18, "• " + b, size=11.5, cls="muted"))
        if idx < 3:
            p.append(line(x + bw + 3, y0 + bh / 2, x + bw + gap - 4, y0 + bh / 2,
                          stroke="muted", width=2, marker=True))

    # The spine: the equity number as it collapses
    sy = 268
    p.append(text(46, sy, "THE SPINE — one strategy, measured four ways", size=14, weight="650"))
    p.append(text(46, sy + 20, "SMA 3/24 on trending_with_crash.csv, the best of 22 grid variants",
                  size=11.5, cls="muted", mono=True))

    steps = [
        ("+46,986%", "no shift, no costs", "the Day 1 result", "bad"),
        ("+406%", "indexing corrected", "look-ahead removed", "warn"),
        ("−70.7%", "+ 24 bps round trip", "costs added", "bad"),
        ("+582%", "buy & hold", "doing nothing beat it", "good"),
    ]
    cw2 = (W - 92 - 3 * 16) / 4
    for j, (value, how, note, color) in enumerate(steps):
        x = 46 + j * (cw2 + 16)
        yy = sy + 40
        p.append(box(x, yy, cw2, 96, fill=f"{color}-bg", stroke=color, width=1.6))
        p.append(text(x + cw2 / 2, yy + 40, value, size=25, weight="700", anchor="middle",
                      fill=color, mono=True))
        p.append(text(x + cw2 / 2, yy + 62, how, size=11.5, anchor="middle"))
        p.append(text(x + cw2 / 2, yy + 80, note, size=10.5, anchor="middle", cls="muted"))
        if j < 3:
            p.append(line(x + cw2 + 2, yy + 48, x + cw2 + 13, yy + 48, stroke="muted", width=2, marker=True))

    # Closing statement
    cy = sy + 158
    p.append(box(46, cy, W - 92, 74, fill="surface", stroke="line"))
    p.append(text(W / 2, cy + 30,
                  "A student who runs a backtest and sees 300% has learned nothing except how to fool themselves.",
                  size=13.5, anchor="middle", weight="600"))
    p.append(text(W / 2, cy + 54,
                  "Every bias in this workshop is something you measure, not something you read about.",
                  size=12, anchor="middle", cls="muted"))
    return svg_close(p)


if __name__ == "__main__":
    print(write("06-workshop-map.svg", build()))
