#!/usr/bin/env python3
"""Check that every internal link and image reference resolves.

    python tools/check_links.py

Verifies, without touching the network:

* Markdown links to repository files (README, docs/**.md).
* Jekyll `{{ '/path' | relative_url }}` references against docs/.
* Image references from notebooks and docs to diagrams/svg or docs/assets.
* That every notebook has a Colab badge pointing at its own filename.

External http(s) links are listed but not fetched, because CI must work
offline and a link checker that needs the internet fails for reasons that have
nothing to do with the commit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GITHUB_REPO = "berdakh/ROBT407"

MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
JEKYLL_REL = re.compile(r"\{\{\s*'([^']+)'\s*\|\s*relative_url\s*\}\}")
HTML_SRC = re.compile(r'<(?:img|link)[^>]+(?:src|href)="([^"]+)"')


def resolve_docs_path(target: str) -> Path | None:
    """Map a site path like /guides/tools.html to a file in docs/."""
    clean = target.lstrip("/").split("#")[0]
    if not clean:
        return DOCS / "index.md"
    candidate = DOCS / clean
    if candidate.exists():
        return candidate
    if clean.endswith(".html"):
        md = DOCS / (clean[: -len(".html")] + ".md")
        if md.exists():
            return md
    return None


def check_markdown(path: Path, errors: list[str]) -> int:
    text = path.read_text()
    checked = 0

    for target in MD_LINK.findall(text) + HTML_SRC.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
            continue
        if "{{" in target:
            continue                      # handled by the Jekyll pass below
        checked += 1
        resolved = (path.parent / target.split("#")[0]).resolve()
        if not resolved.exists():
            errors.append(f"{path.relative_to(ROOT)}: broken link -> {target}")

    for target in JEKYLL_REL.findall(text):
        checked += 1
        if resolve_docs_path(target) is None:
            errors.append(f"{path.relative_to(ROOT)}: relative_url does not resolve -> {target}")

    return checked


def check_notebooks(errors: list[str]) -> int:
    import json

    checked = 0
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "markdown"
        )

        checked += 1
        if "colab-badge.svg" not in source:
            errors.append(f"notebooks/{path.name}: no Open in Colab badge")
        elif path.name not in source:
            errors.append(f"notebooks/{path.name}: Colab badge points at the wrong notebook")

        for target in MD_LINK.findall(source):
            if target.startswith(("http://", "https://", "#")):
                continue
            checked += 1
            resolved = (path.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                errors.append(f"notebooks/{path.name}: broken reference -> {target}")

    return checked


def check_baseurl(errors: list[str]) -> int:
    """GitHub Pages serves a project site from /<repo>/, not the domain root.

    Without a matching ``baseurl``, Jekyll's ``relative_url`` emits paths like
    ``/handbook.html``, which resolve to ``<user>.github.io/handbook.html`` --
    a 404 for every page, stylesheet and diagram. The source paths all look
    correct, so only a real build reveals it. This check is the cheap guard.
    """
    config = DOCS / "_config.yml"
    if not config.exists():
        errors.append("docs/_config.yml is missing")
        return 1

    text = config.read_text()
    match = re.search(r"^baseurl:\s*[\"']?([^\"'\n]*)", text, re.MULTILINE)
    expected = "/" + GITHUB_REPO.split("/")[1]

    if match is None:
        errors.append(
            f"docs/_config.yml has no baseurl. A project site needs "
            f'baseurl: "{expected}" or every link 404s.'
        )
    elif match.group(1).strip().rstrip("/") != expected:
        errors.append(
            f"docs/_config.yml baseurl is {match.group(1).strip()!r}, "
            f"expected {expected!r} to match the repository name."
        )
    return 1


def main() -> int:
    errors: list[str] = []
    checked = 0
    checked += check_baseurl(errors)

    for path in [ROOT / "README.md", ROOT / "data" / "README.md", ROOT / "data" / "real" / "README.md"]:
        if path.exists():
            checked += check_markdown(path, errors)
    for path in sorted(DOCS.rglob("*.md")):
        checked += check_markdown(path, errors)

    checked += check_notebooks(errors)

    if errors:
        print(f"{len(errors)} broken reference(s):\n")
        for error in errors:
            print(f"  {error}")
        return 1

    print(f"All {checked} internal references resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
