#!/usr/bin/env python3
r"""Cut a release commit and its tag, strip a tree, build or check artefacts.

Usage::

    prepare_release.py cut VERSION [--date YYYY-MM-DD] [--doi DOI]
                       [--dry-run]
    prepare_release.py strip [--tree DIR]
    prepare_release.py files SDIST ARCHIVE
    prepare_release.py asset REF OUT [--prefix PREFIX]

A release holds the solver alone, in two forms with the same files: the
PyPI sdist, and the GitHub zipball of the tag, the file deposited on
Zenodo (the sdist's ``source-include`` and ``.gitattributes`` keep the
lab, the agent notes, the contributor guide, the development
configuration and ``uv.lock`` out of both).  The lab, flows and
features in development, is not shipped with release archives outside
GitHub: of a release's archives, only the with-lab asset (``asset``)
carries it.  Both forms must install with ``uv``.  ``main``'s
``pyproject.toml`` cannot serve as is: its default ``lab`` dependency
group names ``dnsjax-lab``, a workspace member neither form contains,
so ``uv sync`` there would fail -- while in a full clone that group is
what installs the lab with the solver, which must keep working.  So a
release is cut from its own commit:

``cut`` checks *VERSION* against ``pyproject.toml`` and
``CITATION.cff`` (bump both on ``main`` first), then, in a throwaway
worktree of ``HEAD``: strips the lab from ``pyproject.toml`` (the
``lab`` group and its default, the workspace and its source, the lab's
test path), sets ``date-released`` in ``CITATION.cff`` (*--date*,
default the current date), gives ``README.md`` its published form
(``pypi_readme.published_readme``: the sections a release leaves out
dropped, the pointer to the lab after the opening paragraph), pins the
PyPI README's links to the tag (``pypi_readme.py``), points every link
the archive's Markdown makes to a file the archive leaves out (the
lab's, the contributor guide) at GitHub at the tag
(:func:`archive_links`), relocks (``uv lock``), commits "Release
vVERSION" and tags it ``vVERSION`` (annotated).  With *--doi*, the DOI
Zenodo reserved for the release's deposit, it also records that DOI
(:func:`with_doi`) before the README's published form is made, so the
sdist, the wheel's description and the archive all carry it.  The
repository's commit hooks run on that commit as on any other; the
message check's ``uv run`` syncs an environment of the release's lock
inside the worktree, which goes with it.  The commit is reachable from
the tag alone: ``main`` is untouched, and pushing the tag (``git push
origin vVERSION``, the author's step) is what starts
``.github/workflows/release.yml``.  ``--dry-run`` prints the changes
and leaves no commit or tag.

``strip`` applies the ``pyproject.toml`` part in place, for a CI job or
a release rehearsal that builds from ``main``.

``files`` checks an sdist against an archive (a ``git archive`` tar or
the GitHub zipball): the same files, but for the build backend's own
(:data:`SDIST_ONLY`); exit 1, listing them, on any other difference.

``asset`` writes the archive a GitHub release attaches beside the sdist
and the wheel: a gzipped tar of the commit *REF* (the release's parent,
``vVERSION^``, which the release was cut from) under *PREFIX*, holding
every tracked file but the ones written for coding agents
(:func:`agent_file`), so the lab, the contributor guide, ``uv.lock`` and
the development configuration are in it, with ``README.md`` without the
sections a published copy leaves out (``pypi_readme.drop_sections``),
as the release's own README is.  ``.github/workflows/release.yml``
builds it from the tag; the same command rehearses it locally.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
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
from pypi_readme import (  # noqa: E402
    DROPPED_SECTIONS,
    drop_sections,
    published_readme,
    pypi_readme,
    repository_url,
)

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


#: A DOI Zenodo mints (``cut --doi``): its badge service draws the badge.
_ZENODO_DOI = re.compile(r"10\.5281/zenodo\.[0-9]+")
#: The README's sentence naming the release's DOI (:func:`with_doi`).
_RELEASE_DOI = (
    r"\nThis release is archived on Zenodo as \[[^]\n]*\]\([^)\n]*\)\.\n"
)
#: The README's DOI badge line, as :func:`with_doi` writes it.
_DOI_BADGE = re.compile(
    r"^\[!\[DOI\]\(https://zenodo\.org/badge/DOI/[^)\s]+\.svg\)\]"
    r"\(https://doi\.org/[^)\s]+\)\n",
    re.M,
)


def with_doi(readme: str, cff: str, doi: str) -> tuple[str, str]:
    """*readme* and *cff* (``README.md``, ``CITATION.cff``) recording *doi*.

    The citation's ``doi:`` becomes *doi* (put after ``version:`` and
    any ``date-released:``); in the README, the DOI badge becomes
    *doi*'s (a badge already there is replaced, or one is put after the
    licence badge), and a sentence after the first paragraph of
    "License and citation" says this release is archived under *doi*
    (replacing one already there).
    *doi* is the one Zenodo reserved for the release's deposit, so it
    resolves once that deposit is published.  Raises unless *doi* is a
    Zenodo DOI and each place it goes is found exactly once.
    """
    if not _ZENODO_DOI.fullmatch(doi):
        raise SystemExit(f"--doi {doi}: not a Zenodo DOI (10.5281/zenodo.N)")
    url = f"https://doi.org/{doi}"
    badge = f"[![DOI](https://zenodo.org/badge/DOI/{doi}.svg)]({url})\n"
    if _DOI_BADGE.search(readme):
        readme, n = _DOI_BADGE.subn(lambda _: badge, readme)
    else:
        readme, n = re.subn(
            r"^!\[License[^\n]*\n",
            lambda m: m.group(0) + badge,
            readme,
            flags=re.M,
        )
    if n != 1:
        raise SystemExit(f"README.md: the DOI badge's place found {n}x")
    readme = re.sub(_RELEASE_DOI, "", readme)
    readme, n = re.subn(
        r"(\n## License and citation\n\n(?:[^\n]+\n)+)",
        lambda m: (
            m.group(1)
            + f"\nThis release is archived on Zenodo as [{doi}]({url}).\n"
        ),
        readme,
    )
    if n != 1:
        raise SystemExit(f"README.md: 'License and citation' found {n}x")
    cff = re.sub(r"\ndoi:[^\n]*", "", cff)
    cff, n = re.subn(
        r"^((?:version|date-released): *\S+\n)(?!date-released:)",
        lambda m: m.group(1) + f"doi: {doi}\n",
        cff,
        flags=re.M,
    )
    if n != 1:
        raise SystemExit(f"CITATION.cff: the doi's place found {n}x")
    return readme, cff


def agent_file(path: str) -> bool:
    """Whether *path* (a repository path) is written for coding agents.

    The ``CLAUDE.md`` notes anywhere and everything under ``.claude/``:
    the files ``.gitattributes`` keeps out of every release archive
    that the with-lab asset keeps out too.
    """
    return (
        PurePosixPath(path).name == "CLAUDE.md"
        or path.split("/", 1)[0] == ".claude"
    )


def _asset(ref: str, out: Path, prefix: str) -> int:
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    mtime = int(_git("show", "-s", "--format=%ct", commit))
    entries = []
    tree = _git("ls-tree", "-r", "-z", "--full-tree", commit)
    for line in tree.strip("\0").split("\0"):
        meta, path = line.split("\t", 1)
        mode, kind, sha = meta.split()
        if kind != "blob":
            raise SystemExit(f"{path}: a {kind}, not a file")
        if not agent_file(path):
            entries.append((mode, sha, path))
    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=REPO,
        input="".join(f"{sha}\n" for _, sha, _ in entries).encode(),
        capture_output=True,
        check=True,
    ).stdout
    blobs, at = [], 0
    for _ in entries:
        end = batch.index(b"\n", at)
        size = int(batch[at:end].split()[2])
        blobs.append(batch[end + 1 : end + 1 + size])
        at = end + 2 + size
    dirs = sorted(
        {
            str(parent)
            for _, _, path in entries
            for parent in PurePosixPath(path).parents
            if str(parent) != "."
        }
    )

    def info(name: str, mode: int) -> tarfile.TarInfo:
        member = tarfile.TarInfo(prefix + name)
        member.mode, member.mtime = mode, mtime
        member.uname = member.gname = "root"
        return member

    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        gzip.GzipFile(out, "wb", mtime=mtime) as gz,
        tarfile.open(
            fileobj=gz,
            mode="w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": commit},
        ) as tar,
    ):
        if prefix:
            member = info("", 0o755)
            member.type = tarfile.DIRTYPE
            tar.addfile(member)
        for d in dirs:
            member = info(d + "/", 0o755)
            member.type = tarfile.DIRTYPE
            tar.addfile(member)
        for (mode, _, path), blob in zip(entries, blobs, strict=True):
            if path == "README.md":
                text = drop_sections(blob.decode())
                if any(f"\n## {t}\n" in text for t in DROPPED_SECTIONS):
                    raise SystemExit("README.md: a dropped section is left")
                blob = text.encode()
            member = info(path, 0o755 if mode == "100755" else 0o644)
            if mode == "120000":
                member.type, member.linkname = tarfile.SYMTYPE, blob.decode()
                tar.addfile(member)
                continue
            member.size = len(blob)
            tar.addfile(member, io.BytesIO(blob))
    print(
        f"wrote {out}: {len(entries)} files of {ref} ({commit[:12]}), "
        "the agent files left out"
    )
    return 0


def _git(*args: str, cwd: Path = REPO) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )
    if out.returncode:
        raise SystemExit(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


def _cut(version: str, date: str, doi: str | None, dry_run: bool) -> int:
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
            path = tree / "README.md"
            original = path.read_text()
            if doi is not None:
                cff = tree / "CITATION.cff"
                original, text = with_doi(original, cff.read_text(), doi)
                cff.write_text(text)
            path.write_text(published_readme(original, repository_url()))
            readme = pypi_readme(original, repository_url(), tag)
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
            # The commit-msg hook wraps the body at 72 columns.
            doi_part = ", the release's DOI recorded" if doi else ""
            body = textwrap.fill(
                "The release's own commit (prepare_release.py): the lab "
                "taken out of pyproject.toml, uv.lock relocked without "
                f"it, the citation dated{doi_part} and the PyPI README "
                "pinned to the tag.",
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
    cut.add_argument("--doi", default=None)
    cut.add_argument("--dry-run", action="store_true")
    strip = sub.add_parser("strip")
    strip.add_argument("--tree", type=Path, default=REPO)
    files = sub.add_parser("files")
    files.add_argument("sdist", type=Path)
    files.add_argument("archive", type=Path)
    asset = sub.add_parser("asset")
    asset.add_argument("ref")
    asset.add_argument("out", type=Path)
    asset.add_argument("--prefix", default="")
    a = ap.parse_args()
    if a.cmd == "cut":
        return _cut(a.version, a.date, a.doi, a.dry_run)
    if a.cmd == "files":
        return _files(a.sdist, a.archive)
    if a.cmd == "asset":
        return _asset(a.ref, a.out, a.prefix)
    path = a.tree / "pyproject.toml"
    path.write_text(strip_pyproject(path.read_text()))
    print(f"stripped the lab from {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
