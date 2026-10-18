"""List the passages of prose that repeat each other across the tree.

Usage::

    uv run python lab/scripts/docs/dup_prose.py [--root DIR] [--words N]
        [--min-shared K] [--only REGEX]

Splits the tracked prose into passages (each docstring paragraph, each
block of consecutive comment lines, each Markdown paragraph outside
code fences; the agent notes and the generated PyPI README left out),
cuts each into overlapping runs of N words (default 8), and prints
every pair of passages sharing at least K runs (default 8), most shared
first: ``[shared, share of the smaller passage] path:line (kind) <->
path:line (kind)``, then the opening words of each.  A run found in
more than twelve passages is boilerplate and not counted.  ``--only``
keeps the pairs with a path matching REGEX.  A pair is a lead for a
review, not a verdict: two sibling functions may each need their own
contract.  *DIR* is the repository to read (default: the one holding
this script).
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

SKIP = re.compile(r"(^|/)CLAUDE\.md$|^\.claude/|^docs/pypi/")


def passages(path: str, src: str, n: int) -> list[tuple[int, str, str]]:
    """``(line, kind, text)`` of each passage of at least *n* words."""
    out = []
    if path.endswith(".py"):
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return []
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Module,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                doc = ast.get_docstring(node, clean=False)
                if not doc:
                    continue
                line = node.body[0].lineno
                for para in re.split(r"\n\s*\n", doc):
                    if len(para.split()) >= n:
                        out.append((line, "doc", para))
                    line += para.count("\n") + 2
        block, start, last = [], 0, -2
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except tokenize.TokenError:
            toks = []
        for tok in toks:
            if tok.type != tokenize.COMMENT:
                continue
            row = tok.start[0]
            if row != last + 1 and block:
                if len(" ".join(block).split()) >= n:
                    out.append((start, "com", " ".join(block)))
                block = []
            if not block:
                start = row
            block.append(tok.string.lstrip("#").strip())
            last = row
        if block and len(" ".join(block).split()) >= n:
            out.append((start, "com", " ".join(block)))
        return out
    para, start, fence = [], 0, False
    for i, line in enumerate(src.splitlines(), 1):
        if line.strip().startswith("```"):
            fence = not fence
        if fence or line.strip().startswith("```"):
            if para and len(" ".join(para).split()) >= n:
                out.append((start, "md", " ".join(para)))
            para = []
            continue
        if not line.strip() or line.startswith("#"):
            if para and len(" ".join(para).split()) >= n:
                out.append((start, "md", " ".join(para)))
            para = []
            continue
        if not para:
            start = i
        para.append(line.strip())
    if para and len(" ".join(para).split()) >= n:
        out.append((start, "md", " ".join(para)))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    ap.add_argument("--words", type=int, default=8)
    ap.add_argument("--min-shared", type=int, default=8)
    ap.add_argument("--only")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    files = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True
    ).stdout.split()
    items = []
    for f in files:
        if SKIP.search(f) or not f.endswith((".py", ".md")):
            continue
        src = (root / f).read_text(encoding="utf-8")
        for line, kind, text in passages(f, src, a.words):
            items.append((f, line, kind, text))
    runs, index = [], defaultdict(set)
    for i, (_, _, _, text) in enumerate(items):
        w = re.findall(r"[a-z0-9_]+", text.lower())
        s = {" ".join(w[j : j + a.words]) for j in range(len(w) - a.words + 1)}
        runs.append(s)
        for r in s:
            index[r].add(i)
    shared: dict[tuple[int, int], int] = defaultdict(int)
    for ids in index.values():
        if len(ids) > 12:
            continue
        ids = sorted(ids)
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                if items[i][0] == items[j][0] and (
                    abs(items[i][1] - items[j][1]) < 3
                ):
                    continue
                shared[(i, j)] += 1
    only = re.compile(a.only) if a.only else None
    rows = []
    for (i, j), k in shared.items():
        if k < a.min_shared:
            continue
        if only and not (only.search(items[i][0]) or only.search(items[j][0])):
            continue
        frac = k / max(1, min(len(runs[i]), len(runs[j])))
        rows.append((k, frac, i, j))
    for k, frac, i, j in sorted(rows, reverse=True):
        fi, li, ki, ti = items[i]
        fj, lj, kj, tj = items[j]
        print(f"[{k:4d}, {frac:4.0%}] {fi}:{li} ({ki})  <->  {fj}:{lj} ({kj})")
        print("    A:", " ".join(ti.split())[:150])
        print("    B:", " ".join(tj.split())[:150])
    print(f"{len(rows)} pair(s) of {len(items)} passages", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
