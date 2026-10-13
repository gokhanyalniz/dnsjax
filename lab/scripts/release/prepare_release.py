#!/usr/bin/env python3
r"""Cut a release commit and its tag, or strip a tree for a release build.

Usage::

    prepare_release.py cut VERSION [--date YYYY-MM-DD] [--dry-run]
    prepare_release.py strip [--tree DIR]

A release holds the solver alone, and both of its forms must install
with ``uv``: the PyPI sdist (no ``uv.lock``) and the GitHub zipball
Zenodo archives (with ``uv.lock``; ``.gitattributes`` keeps ``lab/``
and the agent notes out of it).  ``main``'s ``pyproject.toml`` cannot
serve as is: its default ``lab`` dependency group names ``dnsjax-lab``,
a workspace member neither form contains, so ``uv sync`` there would
fail -- while in a full clone that group is what installs the lab with
the solver, which must keep working.  So a release is cut from its own
commit:

``cut`` checks *VERSION* against ``pyproject.toml`` and
``CITATION.cff`` (bump both on ``main`` first), then, in a throwaway
worktree of ``HEAD``: strips the lab from ``pyproject.toml`` (the
``lab`` group and its default, the workspace and its source, the lab's
test path), sets ``date-released`` in ``CITATION.cff`` (*--date*,
default today), pins the PyPI README's links to the tag
(``pypi_readme.py``), relocks (``uv lock``), commits "Release vVERSION"
and tags it ``vVERSION`` (annotated).  The commit is reachable from the
tag alone: ``main`` is untouched, and pushing the tag
(``git push origin vVERSION``, the author's step) is what starts
``.github/workflows/release.yml``.  ``--dry-run`` prints the changes and
leaves no commit or tag.

``strip`` applies the ``pyproject.toml`` part in place, for a CI job or
a release rehearsal that builds from ``main``.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The repository holding this script (``lab/scripts/release/``).
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from pypi_readme import pypi_readme, repository_url  # noqa: E402

#: The ``pyproject.toml`` edits of a release, as ``(what, pattern,
#: replacement)``; each pattern must match exactly once.
STRIP: list[tuple[str, str, str]] = [
    (
        "the lab dependency group",
        r'\n(?:#[^\n]*\n)*lab = \[\n\s*"dnsjax-lab",\n\]\n',
        "\n",
    ),
    (
        "the lab default group",
        r'default-groups = \["dev", "lab"\]',
        'default-groups = ["dev"]',
    ),
    (
        "the workspace",
        r'\[tool\.uv\.workspace\]\nmembers = \["lab"\]\n+',
        "",
    ),
    (
        "the lab's source",
        r"\[tool\.uv\.sources\]\ndnsjax-lab = \{ workspace = true \}\n+",
        "",
    ),
    (
        "the lab's test path",
        r'(?:#[^\n]*\n)*testpaths = \["tests", "lab/tests"\]',
        'testpaths = ["tests"]',
    ),
]


def strip_pyproject(text: str) -> str:
    """*text* (a ``pyproject.toml``) with the lab taken out."""
    for what, pattern, repl in STRIP:
        text, n = re.subn(pattern, repl, text)
        if n != 1:
            raise SystemExit(f"pyproject.toml: {what} not found once ({n})")
    meta = tomllib.loads(text)
    tool = meta.get("tool", {}).get("uv", {})
    leftover = (
        "lab" in meta.get("dependency-groups", {})
        or "workspace" in tool
        or "sources" in tool
        or "dnsjax-lab" in text
    )
    if leftover:
        raise SystemExit(
            "pyproject.toml: the lab is still named after stripping"
        )
    return text


def _git(*args: str, cwd: Path = REPO) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )
    if out.returncode:
        raise SystemExit(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


def _cut(version: str, date: str, dry_run: bool) -> int:
    tag = f"v{version}"
    if _git("status", "--porcelain"):
        raise SystemExit("the working tree is not clean")
    if _git("tag", "--list", tag):
        raise SystemExit(f"tag {tag} exists")
    with open(REPO / "pyproject.toml", "rb") as f:
        project = tomllib.load(f)["project"]["version"]
    cff = re.search(
        r"^version: *(\S+)", (REPO / "CITATION.cff").read_text(), re.M
    )
    if project != version or cff is None or cff.group(1) != version:
        raise SystemExit(
            f"version {version}: pyproject.toml has {project}, CITATION.cff "
            f"{cff.group(1) if cff else 'none'}; bump both on main first"
        )
    with tempfile.TemporaryDirectory(prefix="dnsjax-release-") as tmp:
        tree = Path(tmp) / "tree"
        _git("worktree", "add", "--detach", str(tree), "HEAD")
        try:
            path = tree / "pyproject.toml"
            path.write_text(strip_pyproject(path.read_text()))
            path = tree / "CITATION.cff"
            text = re.sub(r"\ndate-released:[^\n]*", "", path.read_text())
            path.write_text(
                re.sub(
                    r"^(version: *\S+)$",
                    rf"\1\ndate-released: {date}",
                    text,
                    flags=re.M,
                )
            )
            readme = pypi_readme(
                (tree / "README.md").read_text(), repository_url(), tag
            )
            (tree / "docs" / "pypi" / "README.md").write_text(readme)
            subprocess.run(["uv", "lock", "-q"], cwd=tree, check=True)
            print(_git("diff", "--stat", cwd=tree))
            if dry_run:
                print(_git("diff", cwd=tree))
                print("dry run: no commit, no tag")
                return 0
            _git("add", "-A", cwd=tree)
            _git(
                "commit",
                "-q",
                "-m",
                f"Release {tag}",
                "-m",
                "The release's own commit (prepare_release.py): the lab "
                "taken out of pyproject.toml, uv.lock relocked without "
                "it, the citation dated and the PyPI README pinned to "
                "the tag.",
                cwd=tree,
            )
            _git("tag", "-a", tag, "-m", f"dnsjax {version}", cwd=tree)
            commit = _git("rev-parse", "--short", "HEAD", cwd=tree)
        finally:
            _git("worktree", "remove", "--force", str(tree))
    print(f"tagged {tag} at {commit}; check it, then: git push origin {tag}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    cut = sub.add_parser("cut")
    cut.add_argument("version")
    cut.add_argument("--date", default=datetime.date.today().isoformat())
    cut.add_argument("--dry-run", action="store_true")
    strip = sub.add_parser("strip")
    strip.add_argument("--tree", type=Path, default=REPO)
    a = ap.parse_args()
    if a.cmd == "cut":
        return _cut(a.version, a.date, a.dry_run)
    path = a.tree / "pyproject.toml"
    path.write_text(strip_pyproject(path.read_text()))
    print(f"stripped the lab from {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
