#!/usr/bin/env python3
r"""Cut a release commit and its tag, strip a tree, or check the artefacts.

Usage::

    prepare_release.py cut VERSION [--date YYYY-MM-DD] [--dry-run]
    prepare_release.py strip [--tree DIR]
    prepare_release.py files SDIST ARCHIVE

A release holds the solver alone, in two forms with the same files: the
PyPI sdist, and the GitHub zipball Zenodo archives, which adds
``uv.lock`` (the sdist's ``source-include`` and ``.gitattributes`` keep
the lab, the agent notes and the development configuration out of
both).  Both must install with ``uv``.  ``main``'s ``pyproject.toml`` cannot
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
(``pypi_readme.py``), points every link the archive's Markdown makes to
a file the archive leaves out (the lab's) at GitHub at the tag
(:func:`archive_links`), relocks (``uv lock``), commits "Release vVERSION"
and tags it ``vVERSION`` (annotated).  The commit is reachable from the
tag alone: ``main`` is untouched, and pushing the tag
(``git push origin vVERSION``, the author's step) is what starts
``.github/workflows/release.yml``.  ``--dry-run`` prints the changes and
leaves no commit or tag.

``strip`` applies the ``pyproject.toml`` part in place, for a CI job or
a release rehearsal that builds from ``main``.

``files`` checks an sdist against an archive (a ``git archive`` tar or
the GitHub zipball): the same files, but for ``uv.lock``, which the
archive alone holds, and the build backend's own (:data:`SDIST_ONLY`);
exit 1, listing them, on any other difference.
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
#: The files the archive holds that the sdist leaves out.
ARCHIVE_ONLY = frozenset({"uv.lock"})


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
        *(
            f"only in the archive: {f}"
            for f in in_archive - in_sdist - ARCHIVE_ONLY
        ),
        *(f"missing from the archive: {f}" for f in ARCHIVE_ONLY - in_archive),
    ]
    if problems:
        print("\n".join(sorted(problems)))
        return 1
    print(
        f"the sdist holds the archive's {len(in_archive & in_sdist)} files; "
        f"the archive adds {', '.join(sorted(ARCHIVE_ONLY))}"
    )
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
            for line in archive_links(tree, repository_url(), tag):
                print(line)
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
