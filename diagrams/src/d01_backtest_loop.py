"""Diagram: what the event-driven backtester does on every single bar.

The point the picture must make: the strategy is consulted LAST, and what it
returns cannot be acted on until the next bar. The clock advance at the top is
what bounds everything it can see.
"""

from __future__ import annotations

from theme import arrow_defs, box, circle, line, path, svg_close, svg_open, text, write

W, H = 940, 560


def build() -> str:
    p = svg_open(W, H, "The backtest loop, one bar at a time",
                 "Five ordered steps the engine performs for each bar: advance the clock, "
                 "settle the previous bar's orders at this bar's open, mark to market at "
                 "the close, ask the strategy for a decision, record. Orders decided on "
                 "bar i can only fill on bar i+1.")
    p += arrow_defs("muted")

    p.append(text(W / 2, 38, "The backtest loop, one bar at a time", size=22, weight="650", anchor="middle"))
    p.append(text(W / 2, 62, "for i in range(n_bars):   # the strategy is asked LAST, and cannot act until i+1",
                  size=13, anchor="middle", cls="muted", mono=True))

    steps = [
        ("1", "Advance the clock to bar i",
         "view can now see bars 0..i and nothing beyond.\nThis single line is the look-ahead guarantee.", "accent"),
        ("2", "Settle orders from bar i-1",
         "Fill at bar i's OPEN, minus fees and spread.\nDecided last bar; traded now.", "good"),
        ("3", "Mark to market at bar i's CLOSE",
         "equity = cash + units x close", "muted"),
        ("4", "Ask the strategy",
         "on_bar(view, account) -> target weight.\nAnything it returns is QUEUED for bar i+1.", "accent"),
        ("5", "Record equity, weight, position",
         "One row of the equity curve.", "muted"),
    ]

    x, y, w, h, gap = 104, 92, 466, 72, 14
    for idx, (num, title, sub, color) in enumerate(steps):
        yy = y + idx * (h + gap)
        p.append(box(x, yy, w, h, fill="surface", stroke="line"))
        p.append(circle(x + 28, yy + h / 2, 15, fill=f"{color}-bg" if color != "muted" else "line"))
        p.append(text(x + 28, yy + h / 2 + 5, num, size=14, weight="700", anchor="middle",
                      fill=color if color != "muted" else "fg"))
        p.append(text(x + 56, yy + 27, title, size=15, weight="600"))
        for j, ln in enumerate(sub.split("\n")):
            p.append(text(x + 56, yy + 46 + j * 15, ln, size=12, cls="muted",
                          mono="->" in ln or "equity" in ln or "on_bar" in ln))
        if idx < len(steps) - 1:
            p.append(line(x + w / 2, yy + h, x + w / 2, yy + h + gap, stroke="muted", width=2, marker=True))

    # Loop-back arrow, routed around the OUTSIDE of the column so it never
    # crosses a step box.
    cx = x + w / 2
    y_bottom = y + 5 * (h + gap) - gap
    rail, r = 60, 14
    p.append(path(
        f"M {cx} {y_bottom} "
        f"L {cx} {y_bottom + 20} "
        f"L {rail + r} {y_bottom + 20} "
        f"Q {rail} {y_bottom + 20} {rail} {y_bottom + 20 - r} "
        f"L {rail} {y - 20 + r} "
        f"Q {rail} {y - 20} {rail + r} {y - 20} "
        f"L {cx - 8} {y - 20}",
        stroke="muted", width=2, dash="6 5", marker=True))
    p.append(text(rail - 8, (y + y_bottom) / 2, "next", size=11.5, cls="muted", anchor="end"))
    p.append(text(rail - 8, (y + y_bottom) / 2 + 15, "bar", size=11.5, cls="muted", anchor="end"))

    # Side panel: the timing rule
    px, py, pw = 606, 92, 296
    p.append(box(px, py, pw, 176, fill="accent-bg", stroke="accent"))
    p.append(text(px + 18, py + 30, "The timing rule", size=15, weight="650", fill="accent"))
    for j, ln in enumerate([
        "A signal computed from bar i's",
        "close fills at bar i+1's open.",
        "",
        "One full bar of delay. Always.",
        "There is no shift to remember",
        "because the future is not in",
        "the room.",
    ]):
        p.append(text(px + 18, py + 56 + j * 17, ln, size=12.5))

    # Side panel: what the strategy sees
    qy = py + 196
    p.append(box(px, qy, pw, 232, fill="surface", stroke="line"))
    p.append(text(px + 18, qy + 28, "What on_bar can see", size=15, weight="650"))

    bar_w, bar_gap = 17, 4
    bx, by = px + 20, qy + 58
    for k in range(12):
        visible = k <= 7
        p.append(box(bx + k * (bar_w + bar_gap), by, bar_w, 30, rx=3,
                     fill="accent-bg" if visible else "bad-bg",
                     stroke="accent" if visible else "bad",
                     width=1.2, dash=None if visible else "3 2"))
    p.append(line(bx + 8 * (bar_w + bar_gap) - 2, by - 8, bx + 8 * (bar_w + bar_gap) - 2, by + 40,
                  stroke="bad", width=2))
    p.append(text(bx, by + 58, "visible: bars 0..i", size=12, fill="accent"))
    p.append(text(bx, by + 76, "blocked: LookAheadError", size=12, fill="bad", mono=True))
    for j, ln in enumerate([
        "view.close[-1]      current bar",
        "view.close[-20:]    last 20 bars",
        "view.close[i + 1]   raises",
    ]):
        p.append(text(px + 18, qy + 158 + j * 18, ln, size=11.5, mono=True,
                      fill="bad" if "raises" in ln else None,
                      cls="" if "raises" in ln else "muted"))

    p.append(text(W / 2, H - 16,
                  "Vectorised backtests skip steps 1 and 2 entirely. That is where look-ahead bias lives.",
                  size=12.5, anchor="middle", cls="muted"))
    return svg_close(p)


if __name__ == "__main__":
    print(write("01-backtest-loop.svg", build()))
