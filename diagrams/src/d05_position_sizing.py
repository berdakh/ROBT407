"""Diagram: position sizing and risk control.

Two panels. Left: the same signal, sized three ways, and what each does to the
distribution of outcomes. Right: the volatility-targeting mechanism and the
drawdown arithmetic that makes it matter.
"""

from __future__ import annotations

import math

from theme import box, line, path, svg_close, svg_open, text, write

W, H = 960, 620


def build() -> str:
    p = svg_open(W, H, "Position sizing and risk control",
                 "A signal says which way to lean; sizing says how much. Volatility "
                 "targeting scales exposure inversely to recent realised volatility so "
                 "the portfolio runs at roughly constant risk. Drawdown arithmetic is "
                 "asymmetric: a 50 percent loss needs a 100 percent gain to recover.")

    p.append(text(W / 2, 34, "Position sizing: the half that beginners skip", size=22, weight="650", anchor="middle"))
    p.append(text(W / 2, 57, "The signal picks a direction. Sizing decides whether you survive being wrong.",
                  size=13, anchor="middle", cls="muted"))

    # ---- Left panel: volatility targeting mechanism ----------------------
    lx, ly, lw = 46, 84, 452
    p.append(box(lx, ly, lw, 296, fill="surface", stroke="line"))
    p.append(text(lx + 20, ly + 28, "Volatility targeting", size=15, weight="650"))
    p.append(text(lx + 20, ly + 48, "weight = target_vol / recent_realised_vol   (capped)",
                  size=11.5, mono=True, cls="muted"))

    # Volatility trace and the resulting exposure trace.
    gx, gy, gw, gh = lx + 34, ly + 76, lw - 76, 74

    def vol_at(t):
        return 0.30 + 0.34 * (1 / (1 + math.exp(-(t - 0.55) * 22))) - 0.10 * math.sin(t * 9)

    pts_v, pts_w = [], []
    for k in range(61):
        t = k / 60
        v = vol_at(t)
        pts_v.append((gx + t * gw, gy + gh - (v - 0.15) / 0.55 * gh))
        w = min(0.20 / v, 1.0)
        pts_w.append((gx + t * gw, gy + gh + 104 - w * 74))

    p.append(line(gx, gy + gh, gx + gw, gy + gh, stroke="line", width=1.2))
    p.append(path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts_v), stroke="bad", width=2.2))
    p.append(text(gx - 6, gy + 6, "realised", size=10.5, anchor="end", fill="bad"))
    p.append(text(gx - 6, gy + 19, "volatility", size=10.5, anchor="end", fill="bad"))

    p.append(line(gx, gy + gh + 104, gx + gw, gy + gh + 104, stroke="line", width=1.2))
    p.append(path("M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts_w), stroke="accent", width=2.2))
    p.append(text(gx - 6, gy + gh + 42, "position", size=10.5, anchor="end", fill="accent"))
    p.append(text(gx - 6, gy + gh + 55, "size", size=10.5, anchor="end", fill="accent"))

    p.append(line(gx + gw * 0.62, gy - 4, gx + gw * 0.62, gy + gh + 112, stroke="warn", width=1.6, dash="4 4"))
    # Annotations sit clear of the traces: the vol label above its line, the
    # exposure label below its own.
    p.append(text(gx + gw * 0.62 - 8, gy - 8, "volatility spikes", size=10.5, fill="warn", anchor="end"))
    p.append(text(gx + gw * 0.62 + 8, gy + gh + 30, "exposure falls", size=10.5, fill="warn"))

    p.append(text(lx + 20, ly + 274,
                  "Constant risk, not constant notional. The cap matters: uncapped,",
                  size=11.5, cls="muted"))
    p.append(text(lx + 20, ly + 289,
                  "a quiet market demands huge leverage at the worst moment.",
                  size=11.5, cls="muted"))

    # ---- Right panel: drawdown arithmetic --------------------------------
    rx, ry, rw = 514, 84, W - 514 - 46
    p.append(box(rx, ry, rw, 296, fill="surface", stroke="line"))
    p.append(text(rx + 20, ry + 28, "Why drawdown is the limit that binds", size=15, weight="650"))
    p.append(text(rx + 20, ry + 48, "Losses and recoveries are not symmetric.", size=11.5, cls="muted"))

    rows = [(10, 11), (25, 33), (50, 100), (75, 300), (90, 900)]
    # The "needs +X%" column is at a FIXED x, not trailing a variable-width
    # bar, so the widest label cannot overflow the panel.
    bx, by = rx + 22, ry + 74
    bar_x, bar_w = bx + 48, 150
    label_x = bar_x + bar_w + 14
    for j, (loss, gain) in enumerate(rows):
        yy = by + j * 38
        p.append(text(bx, yy + 12, f"-{loss}%", size=12, mono=True, fill="bad", weight="600"))
        p.append(box(bar_x, yy, bar_w * loss / 100, 14, rx=3, fill="bad-bg", stroke="bad", width=1))
        capped = min(gain, 100)
        p.append(box(bar_x, yy + 16, bar_w * capped / 100, 14, rx=3,
                     fill="good-bg", stroke="good", width=1))
        if gain > 100:
            p.append(text(bar_x + bar_w + 3, yy + 27, "\u203a", size=13, fill="good", weight="700"))
        p.append(text(label_x, yy + 20, f"+{gain}%", size=11.5, fill="good", mono=True, weight="600"))

    p.append(text(rx + 22, ry + 274, "A 50% drawdown is not twice as bad as a 25% one.",
                  size=11.5, cls="muted"))
    p.append(text(rx + 22, ry + 289, "It is four times harder to climb out of.",
                  size=11.5, cls="muted"))

    # ---- Bottom: the three sizing rules ----------------------------------
    sy = 404
    cards = [
        ("Fixed fraction of equity", "weight = 0.5, always",
         "Simple. Risk swings with the market:\nyour worst days cluster exactly when\nvolatility is highest.", "warn"),
        ("Volatility targeting", "weight = target_vol / realised_vol",
         "Roughly constant risk. The standard\nchoice. Needs a leverage cap or it\nblows up in quiet markets.", "good"),
        ("Fixed fractional (stop-based)", "units = equity x risk% / stop_distance",
         "Risk a fixed % per trade. Position\nshrinks automatically when your stop\nhas to be far away.", "good"),
    ]
    cw = (W - 92 - 2 * 18) / 3
    for j, (title, formula, body, color) in enumerate(cards):
        x = 46 + j * (cw + 18)
        p.append(box(x, sy, cw, 128, fill="surface", stroke=color, width=1.6))
        p.append(text(x + 16, sy + 26, title, size=13, weight="650", fill=color))
        p.append(text(x + 16, sy + 46, formula, size=10.5, mono=True, cls="muted"))
        for k, ln in enumerate(body.split("\n")):
            p.append(text(x + 16, sy + 68 + k * 15, ln, size=11))

    p.append(text(W / 2, H - 28,
                  "Kelly sizing maximises growth if you know the true mean. You do not. Practitioners who use it use a quarter of it.",
                  size=12, anchor="middle", cls="muted"))
    return svg_close(p)


if __name__ == "__main__":
    print(write("05-risk-and-position-sizing.svg", build()))
