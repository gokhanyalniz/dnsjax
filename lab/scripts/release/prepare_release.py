#!/usr/bin/env python3
r"""Cut a release commit and its tag, strip a tree, check artefacts.

Usage::

    prepare_release.py cut VERSION [--date YYYY-MM-DD] [--dry-run]
    prepare_release.py strip [--tree DIR]
    prepare_release.py files SDIST ARCHIVE

A release holds the solver alone, in two forms with the same files: the
PyPI sdist, and the GitHub zipball of the tag, the file a Zenodo
deposit takes (the sdist's ``source-include`` and ``.gitattributes``
keep the lab, the agent notes, the contributor guide, the development
configuration and ``uv.lock`` out of both).  The lab, flows and
features in development, is in neither form: the repository holds it.
Both forms must install with ``uv``.  ``main``'s
``pyproject.toml`` cannot serve as is: its default ``lab`` dependency
group names ``dnsjax-lab``, a workspace member neither form contains,
so ``uv sync`` there would fail -- while in a full clone that group is
what installs the lab with the solver, which must keep working.  So a
release is cut from its own commit:

``cut`` checks *VERSION* against ``pyproject.toml``, ``CITATION.cff``
and ``lab/pyproject.toml`` (the lab keeps the core's version; bump all
three on ``main`` first), then, in a throwaway worktree of ``HEAD``:
strips the lab from ``pyproject.toml`` (the ``lab`` group and its
default, the workspace and its source, the lab's test path), sets
``date-released`` in ``CITATION.cff`` (*--date*, default the current
date), makes ``README.md`` the PyPI copy at the tag
(``pypi_readme.pypi_readme``: the sections a release leaves out
dropped, the pointer to the lab after the opening paragraph, links
absolute, math in plain symbols), removes ``docs/pypi/`` (that copy,
its one file) and points ``pyproject.toml``'s ``readme`` at
``README.md`` (:data:`README`), so the archive holds one README, the
one PyPI shows, and the sdist, which holds the archive's files, takes
its description from it; then points every link the archive's Markdown
makes to a file the archive leaves out (the lab's, the contributor
guide) at GitHub at the tag (:func:`archive_links`), relocks (``uv
lock``), commits "Release vVERSION" and tags it ``vVERSION``
(annotated).  Every release carries ``main``'s DOI unchanged: Zenodo's
concept DOI, which resolves to the latest version deposited, in
``CITATION.cff``'s ``doi:`` and the README's DOI badge
(:data:`_DOI_BADGE`); ``cut`` refuses a tree where the two differ.  The
repository's commit hooks run on that commit as on any other; the
message check's ``uv run`` syncs an environment of the release's lock
inside the worktree, removed with it.  The commit is reachable from the
tag alone: ``main`` is untouched, and pushing the tag (``git push
origin vVERSION``, the author's step) starts
``.github/workflows/release.yml``.  ``--dry-run`` prints the changes
and leaves no commit or tag.

``strip`` applies the ``pyproject.toml`` part in place, for a CI job or
a release rehearsal that builds from ``main``.

``files`` checks an sdist against an archive (a ``git archive`` tar or
the GitHub zipball): the same files, but for the build backend's own
(:data:`SDIST_ONLY`); exit 1, listing them, on any other difference.
"""

from __future__ import annotations

import argparse
import datetime
import io
import posixpath
import re
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

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


#: The ``pyproject.toml`` edit of a release commit alone, as ``(pattern,
#: replacement)``: the ``readme`` is ``README.md``, the PyPI copy there.
#: Not in :data:`STRIP`, which a rehearsal applies to ``main``, whose
#: ``README.md`` is GitHub's.
README: tuple[str, str] = (
    r'(?:#[^\n]*\n)*readme = "docs/pypi/README\.md"\n',
    "# The README rewritten for PyPI (absolute links, plain math): a\n"
    "# release's README.md is that copy (lab/scripts/release/).\n"
    'readme = "README.md"\n',
)


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


#: The files the build backend writes into an sdist, which no archive
#: holds.
SDIST_ONLY = frozenset({"PKG-INFO", "pyproject.toml.orig"})


def _members(path: Path) -> set[str]:
    """The files in *path* (a tar, gzipped or not, or a zip).

    The top directory is stripped when every file sits in one, as in an
    sdist or a GitHub zipball (a bare ``git archive`` has none).
    """
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
    else:
        with tarfile.open(path) as t:
            names = [m.name for m in t.getmembers() if m.isfile()]
    tops = {n.split("/", 1)[0] for n in names}
    if len(tops) == 1 and all("/" in n for n in names):
        names = [n.split("/", 1)[1] for n in names]
    return set(names)


def _files(sdist: Path, archive: Path) -> int:
    in_sdist, in_archive = _members(sdist), _members(archive)
    problems = [
        *(
            f"only in the sdist: {f}"
            for f in in_sdist - in_archive - SDIST_ONLY
        ),
        *(f"only in the archive: {f}" for f in in_archive - in_sdist),
    ]
    if problems:
        print("\n".join(sorted(problems)))
        return 1
    print(f"the sdist and the archive hold the same {len(in_archive)} files")
    return 0


#: A link target in Markdown: inline ``[text](target)`` (an optional
#: title after it), an HTML ``href`` / ``src`` / single-URL ``srcset``,
#: or a reference definition ``[label]: target`` (not a footnote's).
_LINK = re.compile(
    r'\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\)'
    r'|(?:href|src|srcset)="([^"\s]+)"'
    r"|^\s*\[(?!\^)[^\]]+\]:\s*(\S+)",
    re.M,
)
_IMAGE = (".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp")


def archive_links(tree: Path, url: str, ref: str) -> list[str]:
    """Point the archive's links to files it leaves out at GitHub.

    In every Markdown file of the archive of *tree*'s ``HEAD``
    (``git archive``, so after ``.gitattributes``), a relative link
    whose target the archive leaves out -- tracked, but export-ignored,
    as the lab is -- becomes an absolute URL at *ref*: ``blob/`` for a
    file, ``tree/`` for a directory, the raw file for an image.  Edits
    *tree* in place and returns one line per rewritten link.  Raises on
    a link to a path tracked nowhere, and if a link still leaves the
    archive afterwards.
    """
    archive = subprocess.run(
        ["git", "archive", "HEAD"], cwd=tree, capture_output=True, check=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        files = {m.name for m in t.getmembers() if m.isfile()}
    dirs = {
        str(parent)
        for f in files
        for parent in PurePosixPath(f).parents
        if str(parent) != "."
    }
    tracked = set(_git("ls-files", cwd=tree).split("\n"))
    raw = url.replace(
        "https://github.com/", "https://raw.githubusercontent.com/"
    )

    def leaving(md: str, target: str) -> str | None:
        """*target*'s repository path if it leaves the archive."""
        if re.match(r"[a-z][a-z0-9+.-]*:", target) or target[:1] == "#":
            return None
        path = posixpath.normpath(
            posixpath.join(posixpath.dirname(md), target.partition("#")[0])
        )
        if path in files or path in dirs:
            return None
        if path not in tracked and not any(
            t.startswith(path + "/") for t in tracked
        ):
            raise SystemExit(f"{md}: {target} is tracked nowhere")
        return path

    done: list[str] = []
    for md in sorted(f for f in files if f.endswith(".md")):
        text = (tree / md).read_text()

        def absolute(m: re.Match, md: str = md) -> str:
            target = next(g for g in m.groups() if g)
            path = leaving(md, target)
            if path is None:
                return m.group(0)
            anchor = target.partition("#")[2]
            if path.endswith(_IMAGE):
                new = f"{raw}/{ref}/{path}"
            elif path in tracked:
                new = f"{url}/blob/{ref}/{path}"
            else:
                new = f"{url}/tree/{ref}/{path}"
            new += f"#{anchor}" if anchor else ""
            done.append(f"{md}: {target} -> {new}")
            return m.group(0).replace(target, new, 1)

        new_text = _LINK.sub(absolute, text)
        if new_text != text:
            (tree / md).write_text(new_text)
    left = [
        f"{md}: {target}"
        for md in sorted(f for f in files if f.endswith(".md"))
        for m in _LINK.finditer((tree / md).read_text())
        if leaving(md, target := next(g for g in m.groups() if g))
    ]
    if left:
        raise SystemExit("links still leave the archive:\n" + "\n".join(left))
    return done


#: The README's DOI badge line; group 1 is the DOI it names and links.
_DOI_BADGE = re.compile(
    r"^\[!\[DOI\]\(https://zenodo\.org/badge/DOI/(\S+?)\.svg\)\]"
    r"\(https://doi\.org/\1\)$",
    re.M,
)


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
    with open(REPO / "lab" / "pyproject.toml", "rb") as f:
        lab = tomllib.load(f)["project"]["version"]
    citation = (REPO / "CITATION.cff").read_text()
    cff = re.search(r"^version: *(\S+)", citation, re.M)
    cff_version = cff.group(1) if cff else "none"
    if version != project or version != cff_version or version != lab:
        raise SystemExit(
            f"version {version}: pyproject.toml has {project}, CITATION.cff "
            f"{cff_version}, lab/pyproject.toml {lab}; bump all three on "
            "main first"
        )
    doi = re.search(r"^doi: *(\S+)$", citation, re.M)
    badge = _DOI_BADGE.search((REPO / "README.md").read_text())
    if doi is None or badge is None or badge.group(1) != doi.group(1):
        raise SystemExit(
            "CITATION.cff's doi and README.md's DOI badge do not name one "
            "DOI, the concept DOI each release carries"
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
            path = tree / "README.md"
            path.write_text(
                pypi_readme(path.read_text(), repository_url(), tag)
            )
            for line in archive_links(tree, repository_url(), tag):
                print(line)
            # After archive_links, which reads every Markdown file of
            # HEAD's archive: the README is now the only PyPI copy.
            pypi = tree / "docs" / "pypi"
            (pypi / "README.md").unlink()
            if any(pypi.iterdir()):
                raise SystemExit("docs/pypi/ holds more than the README")
            pypi.rmdir()
            path = tree / "pyproject.toml"
            text, n = re.subn(*README, path.read_text())
            if n != 1:
                raise SystemExit(f"pyproject.toml: the readme found {n}x")
            path.write_text(text)
            subprocess.run(["uv", "lock", "-q"], cwd=tree, check=True)
            print(_git("diff", "--stat", cwd=tree))
            if dry_run:
                print(_git("diff", cwd=tree))
                print("dry run: no commit, no tag")
                return 0
            _git("add", "-A", cwd=tree)
            # The commit-msg hook refuses a body line over 72 columns.
            body = textwrap.fill(
                "The release's own commit (prepare_release.py): the lab "
                "taken out of pyproject.toml, uv.lock relocked without "
                "it, the citation dated and the PyPI README pinned to "
                "the tag.",
                width=72,
            )
            _git("commit", "-q", "-m", f"Release {tag}", "-m", body, cwd=tree)
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
    files = sub.add_parser("files")
    files.add_argument("sdist", type=Path)
    files.add_argument("archive", type=Path)
    a = ap.parse_args()
    if a.cmd == "cut":
        return _cut(a.version, a.date, a.dry_run)
    if a.cmd == "files":
        return _files(a.sdist, a.archive)
    path = a.tree / "pyproject.toml"
    path.write_text(strip_pyproject(path.read_text()))
    print(f"stripped the lab from {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
