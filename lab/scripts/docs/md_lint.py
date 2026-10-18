"""Check the Markdown lines a change adds against the prose rules.

Usage::

    uv run python lab/scripts/docs/md_lint.py BASE [NEW] [--root DIR]
        [-- FILE.md ...]

For each FILE (default: every ``.md`` changed between BASE and NEW, NEW
being a git ref or, when omitted, the working tree), the lines present
in NEW and not in BASE, outside code fences, are checked for: a width
over 79 columns (table rows, headings, HTML and lines carrying a URL
or link excepted); a spelling ``oxford.py`` would rewrite (its own
rules, read from that script); a word of ``banned_words.TONE`` or
``WORDS`` outside a code span; and an odd number of ``$`` outside code
spans, which is inline math broken across a line (GitHub renders that
raw).  The rules of the docs and the checks of ``check_docstrings.py``
/ ``lint_docs.py`` for Python prose, applied to Markdown.  Prints
``path:line: reason``; the exit status is 1 on any finding.  *DIR* is
the repository to read (default: the one holding this script).
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from banned_words import TONE, WORDS  # noqa: E402

CODE_SPAN = re.compile(r"`[^`\n]*`")
SKIP_WIDTH = re.compile(r"^\s*(\||#|<)|https?://|\]\(")


def oxford_rules():
    """``(PAT, EXEMPT, OTHER)`` as ``oxford.py`` defines them.

    That script rewrites files when run, so its rules are read from its
    source rather than imported.
    """
    tree = ast.parse((HERE / "oxford.py").read_text(encoding="utf-8"))
    env: dict = {"re": re}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name)
            and t.id in ("SUFFIXES", "PAT", "EXEMPT", "OTHER")
            for t in node.targets
        ):
            exec(compile(ast.Module([node], []), "oxford.py", "exec"), env)
    return env["PAT"], env["EXEMPT"], env["OTHER"]


def git(root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else ""


def read(root: Path, ref: str | None, path: str) -> str:
    if ref is None:
        p = root / path
        return p.read_text(encoding="utf-8") if p.exists() else ""
    return git(root, "show", f"{ref}:{path}")


def added_lines(old: str, new: str) -> list[tuple[int, str]]:
    """``(line number, text)`` of the NEW lines outside code fences."""
    a, b = old.splitlines(), new.splitlines()
    added = set()
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        if tag in ("replace", "insert"):
            added.update(range(j1, j2))
    out, fence = [], False
    for j, line in enumerate(b):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence and j in added:
            out.append((j + 1, line))
    return out


def problems(line: str, rules) -> list[str]:
    pat, exempt, other = rules
    why = []
    if len(line) > 79 and not SKIP_WIDTH.search(line):
        why.append(f"{len(line)} columns")
    prose = CODE_SPAN.sub(" ", line)
    for m in pat.finditer(prose):
        if not exempt.match(m.group(0)):
            why.append(f"spelling {m.group(0)!r}")
    for rx, repl in other:
        for m in rx.finditer(prose):
            why.append(f"spelling {m.group(0)!r} (write {repl!r})")
    banned = {m.group(1) for rx in (TONE, WORDS) for m in rx.finditer(prose)}
    why += [f"banned word {w!r}" for w in sorted(banned)]
    if len(re.findall(r"(?<![\\$])\$(?!\$)", prose)) % 2:
        why.append("inline math broken across a line")
    return why


def main(argv: list[str]) -> int:
    files: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, files = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("new", nargs="?")
    ap.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if not files:
        rng = [a.base] + ([a.new] if a.new else [])
        files = [
            f
            for f in git(root, "diff", "--name-only", *rng).split()
            if f.endswith(".md")
        ]
    rules = oxford_rules()
    bad = 0
    for path in files:
        old, new = read(root, a.base, path), read(root, a.new, path)
        for n, line in added_lines(old, new):
            for why in problems(line, rules):
                print(f"{path}:{n}: {why}")
                bad += 1
    print(f"{bad} finding(s)", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
