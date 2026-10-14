#!/usr/bin/env python3
r"""Write the PyPI copy of the README, ``docs/pypi/README.md``.

Usage::

    pypi_readme.py [--ref REF] [--check]

PyPI shows a project's description without the repository around it:
relative links and images resolve nowhere, ``<picture>`` and in-page
anchors do not survive, and ``$...$`` math shows raw.  This writes the
PyPI copy of ``README.md`` (``pyproject.toml``'s ``readme``) with
relative links (Markdown and HTML) made absolute GitHub URLs at *REF*
(default ``main``; a release pins its tag, ``prepare_release.py``),
relative images their ``raw.githubusercontent.com`` URLs, each
``<picture>`` its light ``<img>``, in-page anchors links into the
GitHub README, inline math a code span in plain symbols, the
GitHub-only "Contents" line left out, and the sections
:data:`DROPPED_SECTIONS` names left out, as a release's ``README.md``
leaves them out (``prepare_release.py``).  The repository URL comes
from ``pyproject.toml`` (``[project.urls] Repository``).

A LaTeX command the conversion table (:data:`LATEX`) does not know is
an error, so new README math is converted on purpose, not silently.
``--check`` exits 1 when the tracked copy differs from what it would
write: CI's lint job runs it, which keeps the copy a faithful duplicate
of the README.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

#: The repository holding this script (``lab/scripts/release/``).
REPO = Path(__file__).resolve().parents[3]
#: Where the PyPI copy lives.
OUT = REPO / "docs" / "pypi" / "README.md"
#: Sections of ``README.md`` a published copy (PyPI's, a release's)
#: leaves out, by their ``## `` title.
DROPPED_SECTIONS: tuple[str, ...] = ("Acknowledgements",)

#: LaTeX in the README's inline math -> plain symbols (in order).
LATEX: list[tuple[str, str]] = [
    (r"\\t?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2"),
    (r"\\text\{([^{}]*)\}", r"\1"),
    (r"\\times", "×"),
    (r"\\pm", "±"),
    (r"\\Delta", "Δ"),
    (r"\\theta", "θ"),
    (r"\\pi", "π"),
    (r"\\tau", "τ"),
    (r"\\sin", "sin"),
    (r"\\[,; ]", " "),
    (r"_\{([^{}]*)\}", r"_\1"),
    (r"\^2(?![0-9])", "²"),
    (r"\^\{([^{}]*)\}", r"^\1"),
]


def repository_url() -> str:
    """The GitHub URL of the repository, from ``pyproject.toml``."""
    with open(REPO / "pyproject.toml", "rb") as f:
        meta = tomllib.load(f)
    return meta["project"]["urls"]["Repository"].rstrip("/")


def plain_math(tex: str) -> str:
    """*tex* (one inline formula) in plain symbols; raises on the rest."""
    out = tex
    for pattern, repl in LATEX:
        out = re.sub(pattern, repl, out)
    if "\\" in out:
        raise ValueError(f"no plain form for the LaTeX in ${tex}$")
    return out


def drop_sections(
    readme: str, titles: tuple[str, ...] = DROPPED_SECTIONS
) -> str:
    """*readme* without the ``## title`` sections in *titles*.

    Each section goes from its heading to the next ``## `` heading or
    the end, with its entry in the "Contents" line (``[Title](#slug)
    ·``, or a last entry's leading separator).
    """
    text = readme
    for title in titles:
        slug = re.sub(r"[^\w\- ]", "", title).strip().lower()
        slug = slug.replace(" ", "-")
        text = re.sub(
            rf"\n## {re.escape(title)}\n.*?(?=\n## |\Z)",
            "",
            text,
            flags=re.DOTALL,
        )
        entry = rf"\[{re.escape(title)}\]\(#{slug}\)"
        text = re.sub(entry + r" ·\n", "", text)
        text = re.sub(r" ·\n" + entry + r"(?=\n)", "", text)
    return text


def _outside(text: str, fn, spans: bool) -> str:
    """Apply *fn* to the parts of *text* outside fenced code blocks, and
    outside inline code spans too when *spans*."""
    code = r"(```.*?```|`[^`\n]*`)" if spans else r"(```.*?```)"
    parts = re.split(code, text, flags=re.DOTALL)
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))


def pypi_readme(readme: str, url: str, ref: str) -> str:
    """The PyPI copy of *readme* for the repository at *url*, at *ref*."""
    blob, tree = f"{url}/blob/{ref}", f"{url}/tree/{ref}"
    raw = url.replace(
        "https://github.com/", "https://raw.githubusercontent.com/"
    )
    raw = f"{raw}/{ref}"

    # The sections :data:`DROPPED_SECTIONS` names, then the GitHub-only
    # table of contents (PyPI has its own navigation).
    text = re.sub(
        r"\n\*\*Contents\*\* —.*?\n\n",
        "\n",
        drop_sections(readme),
        flags=re.DOTALL,
    )
    # A <picture> becomes its light <img>.
    text = re.sub(
        r"<picture>.*?(<img\b.*?>).*?</picture>",
        lambda m: m.group(1),
        text,
        flags=re.DOTALL,
    )

    def absolute(target: str) -> str:
        if re.match(r"[a-z]+:", target):
            return target
        if target.startswith("#"):
            return f"{blob}/README.md{target}"
        path, _, anchor = target.partition("#")
        base = tree if path.endswith("/") else blob
        return f"{base}/{path}" + (f"#{anchor}" if anchor else "")

    def image(target: str) -> str:
        return target if re.match(r"[a-z]+:", target) else f"{raw}/{target}"

    def links(part: str) -> str:
        part = re.sub(
            r'(<img\b[^>]*?\bsrc=")([^"]+)(")',
            lambda m: m.group(1) + image(m.group(2)) + m.group(3),
            part,
        )
        part = re.sub(
            r"(!\[[^\]]*\])\(([^)\s]+)\)",
            lambda m: f"{m.group(1)}({image(m.group(2))})",
            part,
        )
        part = re.sub(
            r'(<a\b[^>]*?\bhref=")([^"]+)(")',
            lambda m: m.group(1) + absolute(m.group(2)) + m.group(3),
            part,
        )
        return re.sub(
            r"(?<!!)(\[[^\]]*\])\(([^)\s]+)\)",
            lambda m: f"{m.group(1)}({absolute(m.group(2))})",
            part,
        )

    def math(part: str) -> str:
        return re.sub(
            r"(?<![\w$])\$([^$\n]+?)\$(?![\w$])",
            lambda m: f"`{plain_math(m.group(1))}`",
            part,
        )

    # Links first (a link's text may be a code span), then the math.
    return _outside(_outside(text, links, False), math, True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ref", default="main")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = pypi_readme(
        (REPO / "README.md").read_text(), repository_url(), a.ref
    )
    if a.check:
        if not OUT.is_file() or OUT.read_text() != text:
            print(
                f"{OUT.relative_to(REPO)} is stale: run "
                "lab/scripts/release/pypi_readme.py"
            )
            return 1
        print(f"{OUT.relative_to(REPO)} is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)} at {a.ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
