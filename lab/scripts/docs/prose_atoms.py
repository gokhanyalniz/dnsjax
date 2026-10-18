"""List what a documentation change drops: atoms and qualifiers.

Usage::

    uv run python lab/scripts/docs/prose_atoms.py BASE [NEW]
        [--root DIR] [-- FILE ...]
    uv run python lab/scripts/docs/prose_atoms.py BASE [NEW] --tree

For each FILE (default: every ``.py`` and ``.md`` changed between BASE
and NEW, NEW being a git ref or, when omitted, the working tree), the
prose of the BASE version is compared with the NEW version's: the
docstrings and comments of a ``.py``, the whole text of a ``.md``.
Two reports, for a review of a trim to account for line by line:

- *Atoms whose count fell*: inline math, code spans, Sphinx roles,
  DOIs, URLs, Markdown link targets, quoted titles, author-year
  citations, and numbers (with the unit or word after them) and number
  words.  A span or a title may wrap onto the next line, and a number's
  word may stand on it, but none reaches across a paragraph break.
  Each is tagged ``in-file`` (still in the file, fewer times), ``moved``
  (gone from the file, present in another file's prose; the first few
  are named) or ``GONE`` (in no prose of the NEW tree).
- *Qualifiers lost*: words that carry a condition, a scope or a reason
  (only, never, unless, not, every, except, because, so that, ...),
  counted per paragraph after aligning the BASE and NEW paragraphs; a
  removed paragraph reports all of its qualifiers.

``--tree`` instead lists the atoms present somewhere in the BASE tree's
prose and absent from all of the NEW tree's.  The prose of the agent
notes (``CLAUDE.md``, ``.claude/``) and of the generated PyPI README is
left out.  *DIR* is the repository to read (default: the one holding
this script).  The exit status is 0 either way: the report is input to
a review, not a verdict.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path

#: A character of a span that may wrap: anything but a blank line.
_WRAP = r"(?:[^\n]|\n(?![ \t]*\n))"
#: Inline math: `` `$...$` `` in a docstring (possibly across a line
#: break), ``$...$`` in Markdown.
MATH_PY = re.compile(r"`\$[^$`]+\$`")
MATH_MD = re.compile(r"\$[^$\n]+\$")
ROLE = re.compile(r":[a-z]+:`[^`]+`")
#: Code spans, each possibly across a line break; a run of three
#: backticks (a Markdown fence) opens or closes none.
CODE2 = re.compile(rf"(?<!`)``(?!`)(?:(?!`){_WRAP})+``(?!`)")
CODE1 = re.compile(rf"(?<!`)`(?!`)(?:(?!`){_WRAP})+`(?!`)")
DOI = re.compile(r"\b10\.\d{4,9}/[^\s,;)\]]+")
URL = re.compile(r"https?://[^\s)>\]]+")
LINK = re.compile(r"\]\(([^)\s]+)\)")
#: A quoted title, possibly across a line break.
QUOTED = re.compile(rf"[\"“]([A-Z](?:(?![\"”]){_WRAP}){{3,80}})[\"”]")
CITE = re.compile(
    r"\b[A-Z][a-zà-ÿ]+(?:(?:\s*(?:&|and|–|--|-)\s*[A-Z][a-zà-ÿ]+)+"
    r"|\s+et al\.)?,?\s*\(?(?:19|20)\d{2}[a-z]?\)?"
)
#: A number, with the unit or word after it on its line or the next
#: (never across a paragraph break).
NUMBER = re.compile(
    r"(?<![\w.])[-+~]?\d+(?:[.,]\d+)*(?:e[-+]?\d+)?"
    r"(?:[ \t]*(?:\n[ \t]*)?(?:%|x\b|×|[A-Za-zμ]+))?"
)
NUMBER_WORDS = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|twenty|hundred|thousand|million|half|third|quarter|twice|"
    r"thrice|double|triple|decade|decades|order of magnitude|orders of "
    r"magnitude)\b",
    re.I,
)
QUALIFIERS = re.compile(
    r"\b(?:only|never|always|unless|until|except|not|no|nor|none|"
    r"every|each|all|both|either|neither|at most|at least|exactly|"
    r"because|since|so that|therefore|hence|otherwise|whereas|but|"
    r"if|when|whenever|while|must|cannot|can't|may|deliberately|"
    r"on purpose|by design|first|before|after)\b",
    re.I,
)
SKIP = re.compile(r"(^|/)CLAUDE\.md$|^\.claude/|^docs/pypi/")


def git(root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else ""


def read(root: Path, ref: str | None, path: str) -> str:
    """*path* at *ref* (the working tree when ``None``); ``""`` if none."""
    if ref is None:
        p = root / path
        try:
            return p.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            return ""
    return git(root, "show", f"{ref}:{path}")


def paragraphs(path: str, src: str) -> list[str]:
    """The prose of *src* as paragraphs, in file order."""
    if not path.endswith(".py"):
        return [p for p in re.split(r"\n\s*\n", src) if p.strip()]
    units: list[tuple[int, str]] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                line = node.body[0].lineno
                for k, p in enumerate(re.split(r"\n\s*\n", doc)):
                    if p.strip():
                        units.append((line + k, p))
    block: list[str] = []
    start = last = -2
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != tokenize.COMMENT:
                continue
            row = tok.start[0]
            if row != last + 1 and block:
                units.append((start, " ".join(block)))
                block = []
            if not block:
                start = row
            block.append(tok.string.lstrip("#").strip())
            last = row
    except tokenize.TokenError:
        pass
    if block:
        units.append((start, " ".join(block)))
    return [p for _, p in sorted(units, key=lambda u: u[0])]


def atoms(text: str, markdown: bool) -> Counter:
    """``(kind, atom) -> count``; a span is blanked once counted."""
    found: Counter = Counter()
    for kind, pattern in (
        ("math", MATH_MD if markdown else MATH_PY),
        ("role", ROLE),
        ("code", CODE2),
        ("code", CODE1),
        ("doi", DOI),
        ("url", URL),
        ("link", LINK),
        ("quoted", QUOTED),
        ("cite", CITE),
    ):
        for m in pattern.finditer(text):
            found[(kind, " ".join(m.group(0).split()))] += 1
        text = pattern.sub(" ", text)
    for m in NUMBER.finditer(text):
        tok = " ".join(m.group(0).split())
        if len(tok) > 1:
            found[("number", tok)] += 1
    for m in NUMBER_WORDS.finditer(text):
        found[("number", m.group(0).lower())] += 1
    return found


def qualifiers(text: str) -> Counter:
    return Counter(m.group(0).lower() for m in QUALIFIERS.finditer(text))


def lost_qualifiers(old: list[str], new: list[str]) -> list[str]:
    """Per aligned paragraph, the qualifier words whose count fell."""
    out = []
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        pool = list(range(j1, j2))
        for i in range(i1, i2):
            best, score = None, 0.0
            for j in pool:
                r = difflib.SequenceMatcher(None, old[i], new[j]).ratio()
                if r > score:
                    best, score = j, r
            target = new[best] if best is not None and score > 0.4 else ""
            if best is not None and score > 0.4:
                pool.remove(best)
            q_old, q_new = qualifiers(old[i]), qualifiers(target)
            fell = sorted(w for w in q_old if q_new[w] < q_old[w])
            if fell:
                head = " ".join(old[i].split())[:70]
                state = "reworded" if target else "removed"
                out.append(f"{state}: {', '.join(fell)}  <- {head!r}")
    return out


def tree_prose(root: Path, ref: str | None) -> dict[str, str]:
    if ref is None:
        names = git(root, "ls-files").split()
    else:
        names = git(root, "ls-tree", "-r", "--name-only", ref).split()
    out = {}
    for n in names:
        if n.endswith((".py", ".md")) and not SKIP.search(n):
            out[n] = "\n\n".join(paragraphs(n, read(root, ref, n)))
    return out


def tree_atoms(prose: dict[str, str]) -> dict:
    """``atom -> set of files`` over a whole tree."""
    where: dict = {}
    for n, text in prose.items():
        for key in atoms(text, not n.endswith(".py")):
            where.setdefault(key, set()).add(n)
    return where


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
    ap.add_argument("--tree", action="store_true")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    new_prose = tree_prose(root, a.new)
    new_where = tree_atoms(new_prose)
    if a.tree:
        old_where = tree_atoms(tree_prose(root, a.base))
        gone = sorted(k for k in old_where if k not in new_where)
        for kind, tok in gone:
            src = ", ".join(sorted(old_where[(kind, tok)])[:3])
            print(f"GONE {kind:6s} {tok[:90]}  (was in {src})")
        print(f"{len(gone)} atom(s) gone from the tree", file=sys.stderr)
        return 0
    if not files:
        rng = [a.base] + ([a.new] if a.new else [])
        files = [
            f
            for f in git(root, "diff", "--name-only", *rng).split()
            if f.endswith((".py", ".md")) and not SKIP.search(f)
        ]
    total = 0
    for path in files:
        md = not path.endswith(".py")
        p_old = paragraphs(path, read(root, a.base, path))
        p_new = paragraphs(path, read(root, a.new, path))
        c_old = atoms("\n\n".join(p_old), md)
        c_new = atoms("\n\n".join(p_new), md)
        dropped = sorted(k for k in c_old if c_new[k] < c_old[k])
        quals = lost_qualifiers(p_old, p_new)
        if not dropped and not quals:
            continue
        print(f"== {path}")
        for key in dropped:
            kind, tok = key
            if c_new[key]:
                tag = f"in-file {c_old[key]}->{c_new[key]}"
            elif new_where.get(key, set()) - {path}:
                others = sorted(new_where[key] - {path})
                tag = "moved -> " + ", ".join(others[:3])
            else:
                tag = "GONE"
            print(f"   {kind:6s} {tok[:80]}  [{tag}]")
            total += 1
        for q in quals:
            print(f"   qualifier {q}")
            total += 1
    print(f"{total} item(s) to account for", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
