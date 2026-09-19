"""Diagram: the look-ahead trap, and the measurement that exposes it.

Design intent: the reader must see that the leaky version's position captures
the price move that produced its own signal. Two tracks, same price path, with
the captured segment shaded. The difference is one index.
"""

from __future__ import annotations

import math

from theme import box, circle, line, path, svg_close, svg_open, text, write

W, H = 960, 660


def _path_points(n=6, seed=2.0):
    """A deterministic wiggle that looks like a price path rather than a hill."""
    vals, x = [], 0.0
    for k in range(n):
        x += math.sin(k * 2.3 + seed) * 1.0 + math.sin(k * 0.9 + 1.2) * 0.7 + 0.25
        vals.append(x)
    lo, hi = min(vals), max(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def build() -> str:
    p = svg_open(W, H, "The look-ahead trap",
                 "A position taken on bar t's signal, applied to bar t's own return, earns "
                 "a price move that had already finished happening. Shifting the position "
                 "by one bar removes it. Measured across 40 dataset and seed combinations "
                 "the leaky version won all 40, including on datasets with no signal at all.")

    p.append(text(W / 2, 36, "The look-ahead trap", size=22, weight="650", anchor="middle"))
    p.append(text(W / 2, 59, "One missing index shift. Unlimited imaginary profit.",
                  size=13, anchor="middle", cls="muted"))

    n = 6
    vals = _path_points(n)
    decide = 3                       # the bar whose close produces the signal
    cw = 88
    x0 = 118
    ph = 86
    # The path ends at x0 + (n-1)*cw, leaving the right margin free for
    # captions. Overlapping annotations onto the data is how a diagram stops
    # being readable.
    caption_x = x0 + (n - 1) * cw + 46

    def track(yy, heading, code, color, captured, caption):
        p.append(text(56, yy - 30, heading, size=15, weight="650", fill=color))
        p.append(text(56, yy - 12, code, size=11.5, cls="muted", mono=True))

        pts = [(x0 + k * cw, yy + ph - vals[k] * ph) for k in range(n)]

        # Shade the price move the position actually captures.
        a, b = captured
        p.append(box(pts[a][0], yy - 14, pts[b][0] - pts[a][0], ph + 40, rx=6,
                     fill=f"{color}-bg", stroke=color, width=1.6, dash="5 4"))
        p.append(path(f"M {pts[a][0]:.1f} {pts[a][1]:.1f} L {pts[b][0]:.1f} {pts[b][1]:.1f}",
                      stroke=color, width=3.5))

        # The price path.
        p.append(path("M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts),
                      stroke="muted", width=2, opacity=0.6))
        for k, (px, py) in enumerate(pts):
            p.append(circle(px, py, 4, fill="bg", stroke="muted", width=2))
            label = {decide - 1: "t−1", decide: "t", decide + 1: "t+1"}.get(k)
            p.append(text(px, yy + ph + 34, label or "", size=11.5, anchor="middle",
                          cls="muted", mono=True))

        # The decision marker, placed BELOW the path so it is never clipped.
        dx, dy = pts[decide]
        p.append(circle(dx, dy, 6.5, fill=color))
        p.append(line(dx, dy + 10, dx, yy + ph + 14, stroke=color, width=1.4, dash="3 3"))
        p.append(text(dx, yy + ph + 56, "signal computed", size=11, anchor="middle", fill=color, weight="600"))
        p.append(text(dx, yy + ph + 70, "from this close", size=11, anchor="middle", fill=color))

        p.append(text(caption_x, yy + 34, caption[0], size=12.5, weight="650", fill=color))
        for j, ln in enumerate(caption[1]):
            p.append(text(caption_x, yy + 56 + j * 16, ln, size=11.5, cls="muted"))

    track(118, "WRONG \u2014 shift = 0", "position = signal", "bad", (decide - 1, decide),
          ("captures the move that MADE the signal",
           ["That move had already finished", "before the signal existed.",
            "You cannot trade the past."]))
    track(322, "RIGHT \u2014 shift = 1", "position = signal.shift(1)", "good", (decide, decide + 1),
          ("captures the NEXT move",
           ["The only move you could actually", "have traded on this signal.",
            "Usually far less kind."]))

    # Verdict panel, with text laid out to fit inside it.
    vy = H - 122
    p.append(box(46, vy, W - 92, 100, fill="surface", stroke="line"))
    p.append(text(70, vy + 26, "Measured, not assumed", size=14, weight="650"))

    rows = [
        ("Look-ahead  (shift=0 vs shift=1)", "won 40 / 40", "bad"),
        ("Optimistic fill timing  (this_close)", "won 22 / 40", "warn"),
    ]
    for j, (label, verdict, color) in enumerate(rows):
        yy = vy + 52 + j * 24
        p.append(text(70, yy, label, size=12))
        p.append(text(370, yy, verdict, size=12.5, weight="700", fill=color, mono=True))

    p.append(line(490, vy + 30, 490, vy + 86, stroke="line", width=1.5))
    p.append(text(516, vy + 48, "Look-ahead wins on every dataset — including ones built", size=12))
    p.append(text(516, vy + 66, "with no signal in them at all. That universality is the tell:", size=12))
    p.append(text(516, vy + 84, "a real edge does not work on pure noise.", size=12, weight="600"))
    return svg_close(p)


if __name__ == "__main__":
    print(write("02-lookahead-trap.svg", build()))
