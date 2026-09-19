"""Diagram: walk-forward validation versus shuffled cross-validation.

The picture must make one thing obvious: shuffled K-fold puts future bars in
the training set. Every fold. That is why it reports high accuracy and zero
profit.
"""

from __future__ import annotations

from theme import box, svg_close, svg_open, text, write

W, H = 960, 592
CELL, GAPY = 11.6, 30


def build() -> str:
    p = svg_open(W, H, "Walk-forward validation versus shuffled cross-validation",
                 "Shuffled K-fold places training samples after the test fold in time, so "
                 "the model is fitted on the future. Walk-forward keeps every training "
                 "sample before its test window, with a purge gap for the label horizon "
                 "and an embargo for serial correlation.")

    p.append(text(W / 2, 34, "Walk-forward vs shuffled cross-validation", size=22, weight="650", anchor="middle"))
    p.append(text(W / 2, 57, "Same data, same model, opposite meaning", size=13, anchor="middle", cls="muted"))

    n = 58
    x0 = 152

    def strip(y, assign, label, note=None):
        p.append(text(x0 - 14, y + 11, label, size=12.5, anchor="end", weight="550"))
        for k in range(n):
            role = assign(k)
            fill, stroke = {
                "train": ("accent-bg", "accent"),
                "test": ("good-bg", "good"),
                "purge": ("warn-bg", "warn"),
                "drop": ("surface", "line"),
            }[role]
            p.append(box(x0 + k * (CELL + 1.6), y, CELL, 16, rx=2, fill=fill, stroke=stroke, width=1))
        if note:
            p.append(text(x0 + n * (CELL + 1.6) + 12, y + 12, note, size=11, cls="muted"))

    # -- shuffled K-fold ---------------------------------------------------
    ty = 92
    p.append(box(x0 - 132, ty - 24, W - 60, 176, fill="bad-bg", stroke="bad", width=1.5, opacity=0.35))
    p.append(text(x0 - 120, ty - 4, "WRONG   sklearn train_test_split(shuffle=True) / KFold", size=14,
                  weight="650", fill="bad"))

    import random

    rng = random.Random(11)
    shuffled = list(range(n))
    rng.shuffle(shuffled)
    for f in range(3):
        held = set(shuffled[f * 20 : (f + 1) * 20])
        strip(ty + 18 + f * GAPY, lambda k, h=held: "test" if k in h else "train", f"fold {f + 1}")
    p.append(text(x0, ty + 18 + 3 * GAPY + 16,
                  "Training samples sit on BOTH sides of every test sample. The model interpolates "
                  "between known points", size=11.5, cls="muted"))
    p.append(text(x0, ty + 18 + 3 * GAPY + 32,
                  "instead of extrapolating into an unknown future. Accuracy goes up. Profit does not.",
                  size=11.5, cls="muted"))

    # -- walk-forward ------------------------------------------------------
    wy = 316
    p.append(box(x0 - 132, wy - 24, W - 60, 232, fill="good-bg", stroke="good", width=1.5, opacity=0.3))
    p.append(text(x0 - 120, wy - 4, "RIGHT   walk_forward_splits(train_size, test_size, purge, embargo)",
                  size=14, weight="650", fill="good"))

    train_n, test_n, purge, step = 24, 8, 3, 8
    for f in range(4):
        start = f * step
        tr_end = start + train_n
        def role(k, start=start, tr_end=tr_end):
            if k < start:
                return "drop"
            if k < tr_end - purge:
                return "train"
            if k < tr_end:
                return "purge"
            if k < tr_end + test_n:
                return "test"
            return "drop"
        strip(wy + 18 + f * GAPY, role, f"fold {f + 1}")

    ly = wy + 18 + 4 * GAPY + 22
    legend = [("train", "accent-bg", "accent", "train: strictly before the test window"),
              ("purge", "warn-bg", "warn", "purge: dropped, its labels overlap the test period"),
              ("test", "good-bg", "good", "test: genuinely unseen"),
              ("drop", "surface", "line", "not used in this fold")]
    for j, (_, fill, stroke, desc) in enumerate(legend):
        yy = ly + (j // 2) * 20
        xx = x0 + (j % 2) * 350
        p.append(box(xx, yy - 10, 13, 13, rx=2, fill=fill, stroke=stroke, width=1))
        p.append(text(xx + 20, yy + 1, desc, size=11.5, cls="muted"))

    p.append(text(x0, ly + 58,
                  "Stitching the test windows together gives one continuous out-of-sample equity curve.",
                  size=12, weight="550"))
    p.append(text(x0, ly + 76,
                  "That curve is the only performance number in this workshop that counts as evidence.",
                  size=12, cls="muted"))
    return svg_close(p)


if __name__ == "__main__":
    print(write("03-walk-forward-vs-naive-cv.svg", build()))
