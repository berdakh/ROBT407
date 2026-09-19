"""Diagram: what happens to an order between 'I want to buy' and 'I own it'.

Students assume this is instantaneous and free. Each stage in this diagram
costs money or time, and the sum of them is why a backtest with zero costs is
science fiction.
"""

from __future__ import annotations

from theme import arrow_defs, box, line, svg_close, svg_open, text, write

W, H = 960, 556


def build() -> str:
    p = svg_open(W, H, "The order lifecycle",
                 "Six stages between a strategy's decision and a settled position: decide, "
                 "submit, queue, match, fill, settle. Each stage costs time or money. The "
                 "sum of the costs is why a zero-cost backtest is fiction.")
    p += arrow_defs("muted")

    p.append(text(W / 2, 34, "The order lifecycle", size=22, weight="650", anchor="middle"))
    p.append(text(W / 2, 57, "Everything between “buy” and “bought”, and what each step costs you",
                  size=13, anchor="middle", cls="muted"))

    stages = [
        ("DECIDE", "strategy returns\na target weight", "bar i close", None),
        ("SUBMIT", "order enters\nthe engine", "queued for i+1", "latency\nyou are slower than\nsomeone else"),
        ("QUEUE", "resting in the\norder book", "market or limit", "queue position\nlimit orders may\nnever fill"),
        ("MATCH", "crossed against\nthe other side", "bar i+1 open", "spread\nyou buy at the ask,\nsell at the bid"),
        ("FILL", "you own it,\nmaybe partially", "price != mid", "impact\nsize moves the price\nagainst you"),
        ("SETTLE", "cash and position\nupdated", "fee charged", "commission\ncharged on notional,\nevery single time"),
    ]

    bw, bh, gap = 132, 84, 22
    x0, y0 = 46, 108
    for idx, (name, body, when, cost) in enumerate(stages):
        x = x0 + idx * (bw + gap)
        is_cost = cost is not None
        p.append(box(x, y0, bw, bh, fill="surface", stroke="accent" if not is_cost else "line", width=1.8))
        p.append(text(x + bw / 2, y0 + 24, name, size=13.5, weight="700", anchor="middle", fill="accent"))
        for j, ln in enumerate(body.split("\n")):
            p.append(text(x + bw / 2, y0 + 44 + j * 14, ln, size=11, anchor="middle"))
        p.append(text(x + bw / 2, y0 + bh - 8, when, size=10, anchor="middle", cls="muted", mono=True))

        if idx < len(stages) - 1:
            p.append(line(x + bw + 3, y0 + bh / 2, x + bw + gap - 4, y0 + bh / 2,
                          stroke="muted", width=2, marker=True))

        if cost:
            head, *rest = cost.split("\n")
            cy = y0 + bh + 54
            p.append(line(x + bw / 2, y0 + bh + 4, x + bw / 2, cy - 6, stroke="bad", width=1.6, dash="3 3"))
            p.append(box(x - 4, cy, bw + 8, 66, fill="bad-bg", stroke="bad", width=1.3))
            p.append(text(x + bw / 2, cy + 20, head, size=12, weight="650", anchor="middle", fill="bad"))
            for j, ln in enumerate(rest):
                p.append(text(x + bw / 2, cy + 36 + j * 13, ln, size=10, anchor="middle"))

    # Summing it up
    sy = 330
    p.append(box(46, sy, W - 92, 118, fill="surface", stroke="line"))
    p.append(text(70, sy + 28, "Adding it up: a round trip for a small retail taker on a liquid crypto pair",
                  size=14, weight="650"))
    items = [("exchange fee", "10.0 bps", "each way"),
             ("half the spread", "1.0 bps", "each way"),
             ("latency drift", "1.0 bps", "each way"),
             ("market impact", "~0 bps", "while your size is tiny")]
    for j, (label, val, note) in enumerate(items):
        yy = sy + 52 + j * 16
        p.append(text(70, yy, label, size=11.5, cls="muted"))
        p.append(text(230, yy, val, size=11.5, mono=True, weight="600"))
        p.append(text(300, yy, note, size=11, cls="muted"))
    p.append(line(560, sy + 40, 560, sy + 106, stroke="line", width=1.5))
    p.append(text(590, sy + 62, "round trip = 24 bps", size=17, weight="700", fill="bad", mono=True))
    p.append(text(590, sy + 84, "Trade 200x a year and that is 24% of capital", size=12, cls="muted"))
    p.append(text(590, sy + 101, "paid in costs before you earn anything.", size=12, cls="muted"))

    # The punchline
    py = sy + 140
    p.append(box(46, py, W - 92, 62, fill="warn-bg", stroke="warn"))
    p.append(text(70, py + 26, "The question this diagram exists to make unavoidable", size=13, weight="650", fill="warn"))
    p.append(text(70, py + 46,
                  "Is your average winning trade bigger than 24 basis points? If not, no amount of parameter "
                  "tuning will save the strategy.", size=12))
    return svg_close(p)


if __name__ == "__main__":
    print(write("04-order-lifecycle.svg", build()))
