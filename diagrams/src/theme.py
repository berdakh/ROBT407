"""Shared SVG helpers and a palette that works in light and dark browsers.

Diagrams are authored here as code and rendered to plain SVG. Nothing is
downloaded and nothing is hand-edited, so a diagram can be reviewed as a diff
and regenerated deterministically.

Colour approach: a single accessible hue ramp plus one warning hue. Every
diagram declares CSS custom properties on the root ``<svg>`` and redefines them
under ``prefers-color-scheme: dark``, so the same file is legible on a white
GitHub page and a dark one.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parents[1] / "svg"

# Light / dark token pairs. Keep these in sync deliberately: the dark values are
# not auto-derived, because auto-derived dark palettes are how you get grey text
# on a grey box.
TOKENS = {
    "bg":        ("#ffffff", "#0d1117"),
    "fg":        ("#1f2328", "#e6edf3"),
    "muted":     ("#57606a", "#9198a1"),
    "line":      ("#d0d7de", "#30363d"),
    "surface":   ("#f6f8fa", "#161b22"),
    "accent":    ("#0969da", "#4493f8"),
    "accent-bg": ("#ddf4ff", "#0c2d6b"),
    "good":      ("#1a7f37", "#3fb950"),
    "good-bg":   ("#dafbe1", "#0f3d1c"),
    "bad":       ("#cf222e", "#f85149"),
    "bad-bg":    ("#ffebe9", "#4a1418"),
    "warn":      ("#9a6700", "#d29922"),
    "warn-bg":   ("#fff8c5", "#3d2c00"),
}

FONT = ("ui-sans-serif, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', "
        "Arial, sans-serif")
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"


def svg_open(width: int, height: int, title: str, desc: str = "") -> list[str]:
    """Start an SVG document with the theme tokens and base styles."""
    light = "\n".join(f"      --{k}: {v[0]};" for k, v in TOKENS.items())
    dark = "\n".join(f"        --{k}: {v[1]};" for k, v in TOKENS.items())
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-labelledby="title desc" font-family="{FONT}">',
        f"  <title id=\"title\">{escape(title)}</title>",
        f"  <desc id=\"desc\">{escape(desc or title)}</desc>",
        "  <style>",
        "    :root {",
        light,
        "    }",
        "    @media (prefers-color-scheme: dark) {",
        "      :root {",
        dark,
        "      }",
        "    }",
        "    .bg { fill: var(--bg); }",
        "    text { fill: var(--fg); }",
        "    .muted { fill: var(--muted); }",
        f"    .mono {{ font-family: {MONO}; }}",
        "    .card { fill: var(--surface); stroke: var(--line); }",
        "    .edge { stroke: var(--line); fill: none; }",
        "  </style>",
        f'  <rect class="bg" width="{width}" height="{height}"/>',
    ]


def svg_close(parts: list[str]) -> str:
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def text(x, y, content, *, size=14, weight="400", anchor="start", cls="", fill=None, mono=False):
    classes = " ".join(filter(None, [cls, "mono" if mono else ""]))
    attrs = [
        f'x="{x}"', f'y="{y}"', f'font-size="{size}"', f'font-weight="{weight}"',
        f'text-anchor="{anchor}"',
    ]
    if classes:
        attrs.append(f'class="{classes}"')
    if fill:
        attrs.append(f'fill="var(--{fill})"')
    return f'  <text {" ".join(attrs)}>{escape(str(content))}</text>'


def box(x, y, w, h, *, rx=8, fill="surface", stroke="line", width=1.5, dash=None, opacity=None):
    attrs = [
        f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'rx="{rx}"',
        f'fill="var(--{fill})"', f'stroke="var(--{stroke})"', f'stroke-width="{width}"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f'  <rect {" ".join(attrs)}/>'


def line(x1, y1, x2, y2, *, stroke="line", width=1.5, dash=None, marker=False, opacity=None):
    attrs = [
        f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
        f'stroke="var(--{stroke})"', f'stroke-width="{width}"', 'stroke-linecap="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if marker:
        attrs.append(f'marker-end="url(#arrow-{stroke})"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f'  <line {" ".join(attrs)}/>'


def path(d, *, stroke="accent", width=2, fill="none", dash=None, marker=False, opacity=None):
    attrs = [f'd="{d}"', f'stroke="var(--{stroke})"', f'stroke-width="{width}"',
             f'fill="{fill if fill == "none" else f"var(--{fill})"}"',
             'stroke-linejoin="round"', 'stroke-linecap="round"']
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    if marker:
        attrs.append(f'marker-end="url(#arrow-{stroke})"')
    if opacity is not None:
        attrs.append(f'opacity="{opacity}"')
    return f'  <path {" ".join(attrs)}/>'


def circle(cx, cy, r, *, fill="accent", stroke=None, width=1.5):
    attrs = [f'cx="{cx}"', f'cy="{cy}"', f'r="{r}"', f'fill="var(--{fill})"']
    if stroke:
        attrs += [f'stroke="var(--{stroke})"', f'stroke-width="{width}"']
    return f'  <circle {" ".join(attrs)}/>'


def arrow_defs(*colors: str) -> list[str]:
    """Arrowhead markers, one per colour token used with ``marker=True``."""
    out = ["  <defs>"]
    for c in colors:
        out += [
            f'    <marker id="arrow-{c}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
            f'      <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--{c})"/>',
            "    </marker>",
        ]
    out.append("  </defs>")
    return out


def validate(content: str) -> None:
    """Catch the SVG mistakes that render as silently-missing graphics.

    Specifically: a ``marker-end="url(#arrow-x)"`` whose marker was never
    defined. Browsers ignore the dangling reference and simply draw no
    arrowhead, so the diagram looks subtly wrong rather than broken, and you
    do not notice until someone asks why the arrows point nowhere.
    """
    import re

    referenced = set(re.findall(r"url\(#(arrow-[\w-]+)\)", content))
    defined = set(re.findall(r'<marker id="(arrow-[\w-]+)"', content))
    missing = referenced - defined
    if missing:
        raise ValueError(
            f"SVG references undefined markers {sorted(missing)}; "
            f"add them to arrow_defs(). Defined: {sorted(defined)}"
        )
    unused = defined - referenced
    if unused:
        raise ValueError(f"SVG defines unused markers {sorted(unused)}; remove them")

    for token in set(re.findall(r"var\(--([\w-]+)\)", content)):
        if token not in TOKENS:
            raise ValueError(f"SVG uses undefined colour token --{token}")


def to_static(content: str, mode: str = "light") -> str:
    """Resolve every ``var(--token)`` to a literal colour.

    Used by ``scripts/render_diagrams.py`` to rasterise a preview, because
    most SVG-to-PNG converters do not implement CSS custom properties. The
    committed SVGs keep the variables so they adapt to the reader's theme.
    """
    idx = 0 if mode == "light" else 1
    out = content
    for token, values in TOKENS.items():
        out = out.replace(f"var(--{token})", values[idx])
    return out


def write(name: str, content: str) -> Path:
    validate(content)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / name
    dest.write_text(content)
    return dest
