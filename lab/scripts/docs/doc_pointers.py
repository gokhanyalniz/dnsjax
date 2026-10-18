"""List what other text points at: titles, numbering and their readers.

Usage::

    uv run python lab/scripts/docs/doc_pointers.py BASE [NEW]
        [--root DIR] [--also PATH ...] [-- FILE ...]
    uv run python lab/scripts/docs/doc_pointers.py --inbound
        [--root DIR] [--also PATH ...] -- FILE ...

The first form compares each FILE (default: every file changed between
BASE and NEW, NEW being a git ref or, when omitted, the working tree)
and prints, per file, every *named item* present in BASE and missing
from NEW: a Markdown heading, a docstring or comment section title (a
line underlined with ``-``, ``~``, ``=`` or ``^``), a bold title opening
a paragraph or list item (``**Title.**``, bold runs around inline code
or math included; bold inside a paragraph is not a title), a comment
banner (``# -- Title --``, dashes on both sides), and a change in
how many numbered list items carry each number (a renumbering).  Each
missing title is followed by up to twelve lines of other files of the
NEW tree that still quote it, matched after dropping markup, case and
line breaks and joining adjacent string literals, and each vanished
heading's slug (GitHub's rules, duplicate suffixes included) by up to
twelve links to it.  Text elsewhere names these items, so each one
printed needs either its references updated or an approval.

``--inbound`` lists, for each FILE, the lines elsewhere that point into
it: the lines naming its module, path or a top-level symbol together
with a word such as "docstring", "Design notes" or "the why", and the
lines quoting one of its titles.  A review of a change to FILE checks
that each still finds what it promises.

The tree scanned for references is every tracked ``.py``, ``.md``,
``.toml``, ``.yml`` and shell file, the agent notes included, plus each
PATH given with ``--also`` (a file, or a directory searched for
``.md``), for notes outside version control.  *DIR* is the repository
to read (default: the one holding this script).  The exit status is 0
either way: the report is input to a review.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

UNDER = re.compile(r"^\s*([-~=^])\1{2,}\s*$")
BOLD = re.compile(
    r"^\s*(?:#:?\s*)?(?:[-*]\s+|\d+[.)]\s+)?"
    r"((?:\*\*[^*\n]+\*\*\s*(?:`[^`\n]*`\s*)?)+)"
)
BANNER = re.compile(
    r"^\s*#\s*[─━\-=~]{2,}\s*([^─━\-=~\s].*?)\s*[─━\-=~]{2,}\s*$"
)
LIST_BOLD = re.compile(r"^\s*(?:#:?\s*)?(?:[-*]|\d+[.)])\s+\*\*")
NUMBERED = re.compile(r"^\s*(?:#:?\s*)?(\d+)[.)]\s+\S")
MDHEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
POINTER_WORDS = re.compile(
    r"docstring|design notes|the why|says why|pattern and failure|"
    r"failure signature|module docstring|see the|manual",
    re.I,
)
TEXT_KINDS = (".py", ".md", ".toml", ".yml", ".yaml", ".sh", ".slurm")
#: Section names every docstring may carry: reported when they go, but
#: never searched for as references.
GENERIC = {
    "design notes",
    "parameters",
    "returns",
    "raises",
    "yields",
    "notes",
    "examples",
    "usage",
    "see also",
    "attributes",
    "contents",
}


def git(root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else ""


def read(root: Path, ref: str | None, path: str) -> str:
    if ref is None:
        try:
            return (root / path).read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError, IsADirectoryError):
            return ""
    return git(root, "show", f"{ref}:{path}")


def norm(text: str) -> str:
    """Lower case, markup and string-literal joins dropped, one space."""
    text = re.sub(r"[\"']\s*\n?\s*#?\s*[\"']", "", text)
    text = re.sub(r"[*`$]", "", text)
    text = re.sub(r"\s*#:?\s+", " ", text)
    return " ".join(text.lower().split())


def slug(text: str) -> str:
    """GitHub's link slug for a heading text."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    out = []
    for ch in text:
        if ch == " ":
            out.append("-")
        elif ch in "-_" or unicodedata.category(ch)[0] in "LN":
            out.append(ch)
    return "".join(out)


def named_items(path: str, src: str) -> tuple[set[str], Counter, list]:
    """Titles, the count of numbered items per number, heading slugs."""
    titles: set[str] = set()
    numbers: Counter = Counter()
    slugs: list[str] = []
    lines = src.splitlines()
    fence = False
    seen: Counter = Counter()
    for i, line in enumerate(lines):
        if path.endswith(".md") and line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if path.endswith(".md"):
            m = MDHEAD.match(line)
            if m:
                titles.add(m.group(2).strip())
                s = slug(m.group(2))
                slugs.append(s if not seen[s] else f"{s}-{seen[s]}")
                seen[s] += 1
                continue
        if (
            UNDER.match(line)
            and i > 0
            and lines[i - 1].strip()
            and not UNDER.match(lines[i - 1])
        ):
            titles.add(lines[i - 1].strip().lstrip("#: ").strip())
        m = BOLD.match(line)
        prev = lines[i - 1].strip().lstrip("#:").strip() if i else ""
        opens = not prev or prev.endswith('"""') or UNDER.match(prev)
        if m and (opens or LIST_BOLD.match(line)):
            titles.add(" ".join(m.group(1).split()))
        m = BANNER.match(line)
        if m and not path.endswith(".md"):
            titles.add(m.group(1).strip())
        m = NUMBERED.match(line)
        if m:
            numbers[m.group(1)] += 1
    return {t for t in titles if len(t) >= 3}, numbers, slugs


def scan_files(root: Path, ref: str | None, also: list[Path]) -> dict:
    """``name -> text`` of the tree scanned for references."""
    if ref is None:
        names = git(root, "ls-files").split()
    else:
        names = git(root, "ls-tree", "-r", "--name-only", ref).split()
    out = {
        n: read(root, ref, n)
        for n in names
        if n.endswith(TEXT_KINDS) or n == ".gitattributes"
    }
    for p in also:
        paths = sorted(p.rglob("*.md")) if p.is_dir() else [p]
        for q in paths:
            with contextlib.suppress(OSError, UnicodeDecodeError):
                out[str(q)] = q.read_text(encoding="utf-8")
    return out


def find_quotes(tree: dict, needle: str) -> list[str]:
    """``path:line`` of each place *needle* occurs, markup aside."""
    key = norm(needle).rstrip(".:")
    if len(key) < 6 or key in GENERIC:
        return []
    hits = []
    for name, text in tree.items():
        lines = text.splitlines()
        flat = norm(text)
        if key not in flat:
            continue
        found = False
        for n, line in enumerate(lines, 1):
            window = norm(" ".join(lines[max(0, n - 2) : n + 1]))
            if key in window and (key[:12] in norm(line) or not found):
                hits.append(f"{name}:{n}")
                found = True
        if not found:
            hits.append(f"{name}:?")
    return hits


def compare(root, base, new, files, also) -> None:
    tree = scan_files(root, new, also)
    for path in files:
        if not path.endswith(TEXT_KINDS):
            continue
        t_old, n_old, s_old = named_items(path, read(root, base, path))
        t_new, n_new, s_new = named_items(path, read(root, new, path))
        gone = sorted(t_old - t_new)
        renum = sorted(
            k for k in set(n_old) | set(n_new) if n_old[k] != n_new[k]
        )
        slugs = sorted(set(s_old) - set(s_new))
        if not (gone or renum or slugs):
            continue
        print(f"== {path}")
        for t in gone:
            refs = [r for r in find_quotes(tree, t) if not r.startswith(path)]
            tag = "REFERENCED" if refs else "unreferenced"
            print(f"   {tag:12s} title {t[:80]!r}")
            for r in refs[:12]:
                print(f"      -> {r}")
        if renum:
            pairs = ", ".join(f"{k}: {n_old[k]}->{n_new[k]}" for k in renum)
            print(f"   numbering changed ({pairs})")
        for s in slugs:
            refs = [
                f"{name}:{n}"
                for name, text in tree.items()
                for n, line in enumerate(text.splitlines(), 1)
                if f"#{s})" in line
            ]
            tag = "REFERENCED" if refs else "unreferenced"
            print(f"   {tag:12s} link #{s}")
            for r in refs[:12]:
                print(f"      -> {r}")


def pointer_patterns(path: str, src: str) -> re.Pattern:
    """The pattern of the ways text elsewhere names *path*.

    Its path, file or module name, a backticked top-level symbol, or
    ``module.symbol``.
    """
    p = Path(path)
    alts = [re.escape(path), re.escape(p.name)]
    if p.suffix == ".py":
        parts = list(p.with_suffix("").parts)
        for top in (("src",), ("lab", "src")):
            if tuple(parts[: len(top)]) == top:
                parts = parts[len(top) :]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        dotted = ".".join(parts)
        short = parts[-1] if parts else ""
        alts += [re.escape(dotted), rf"`{re.escape(short)}[`.]"]
        if len(short) >= 6:
            alts.append(rf"\b{re.escape(short)}\.\w")
        with contextlib.suppress(SyntaxError):
            for n in ast.parse(src).body:
                if (
                    isinstance(
                        n,
                        (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef),
                    )
                    and len(n.name) >= 4
                ):
                    alts.append(rf"`{re.escape(n.name)}[`.(]")
    return re.compile("|".join(alts))


def inbound(root: Path, files: list[str], also: list[Path]) -> None:
    tree = scan_files(root, None, also)
    for path in files:
        src = read(root, None, path)
        named_by = pointer_patterns(path, src)
        titles, _, _ = named_items(path, src)
        keys = [
            k
            for k in (norm(t).rstrip(".:") for t in titles)
            if len(k) >= 14 and len(k.split()) >= 3 and k not in GENERIC
        ]
        print(f"== {path}")
        for name, text in sorted(tree.items()):
            if name == path:
                continue
            lines = text.splitlines()
            for n, line in enumerate(lines, 1):
                ctx = " ".join(lines[max(0, n - 2) : n + 1])
                named = named_by.search(line) and POINTER_WORDS.search(ctx)
                quoted = any(k in norm(ctx) for k in keys)
                if named or quoted:
                    print(f"   {name}:{n}: {line.strip()[:100]}")


def main(argv: list[str]) -> int:
    files: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, files = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", nargs="?")
    ap.add_argument("new", nargs="?")
    ap.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    ap.add_argument("--also", nargs="*", type=Path, default=[])
    ap.add_argument("--inbound", action="store_true")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if a.inbound:
        inbound(root, files, a.also)
        return 0
    if a.base is None:
        ap.error("BASE is required without --inbound")
    if not files:
        rng = [a.base] + ([a.new] if a.new else [])
        files = git(root, "diff", "--name-only", *rng).split()
    compare(root, a.base, a.new, files, a.also)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
