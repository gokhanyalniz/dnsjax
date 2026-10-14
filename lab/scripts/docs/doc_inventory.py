"""Inventory docstrings and comments across the tracked Python tree.

Usage: python doc_inventory.py [OUT_DIR]

Writes one text dump per file (docstrings + comments with line numbers)
under OUT_DIR (default ``./dump``, in the current directory: run it from
a scratch directory, not the repository), plus a summary of volumes and
missing docstrings.
"""

import ast
import io
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "dump")
OUT.mkdir(parents=True, exist_ok=True)

files = subprocess.run(
    ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True
).stdout.split()

tot_doc = tot_com = tot_lines = 0
missing = []
rows = []
for f in files:
    src = (ROOT / f).read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    doc_lines = set()
    entries = []  # (lineno, kind, name, text)

    def visit(node, prefix="", *, f=f, doc_lines=doc_lines, entries=entries):
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                name = prefix + child.name
                ds = ast.get_docstring(child, clean=False)
                if ds is not None:
                    d = child.body[0]
                    for ln in range(d.lineno, d.end_lineno + 1):
                        doc_lines.add(ln)
                    entries.append((d.lineno, "doc", name, ds))
                else:
                    public = not child.name.startswith("_")
                    parent_public = all(
                        not p.startswith("_") for p in prefix.split(".") if p
                    )
                    if public and parent_public and not f.startswith("tests/"):
                        missing.append((f, child.lineno, name))
                visit(child, name + ".")
            elif isinstance(child, (ast.If, ast.Try, ast.With)):
                visit(child, prefix)

    mds = ast.get_docstring(tree, clean=False)
    if mds is not None:
        d = tree.body[0]
        for ln in range(d.lineno, d.end_lineno + 1):
            doc_lines.add(ln)
        entries.append((d.lineno, "moddoc", "<module>", mds))
    elif lines and not f.endswith("__init__.py"):
        missing.append((f, 1, "<module>"))
    elif lines:
        missing.append((f, 1, "<module __init__>"))
    visit(tree)
    com_lines = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            com_lines.append((tok.start[0], tok.string))
    nd = len(doc_lines)
    nc = len(com_lines)
    tot_doc += nd
    tot_com += nc
    tot_lines += len(lines)
    rows.append((nd + nc, nd, nc, len(lines), f))
    # dump
    with open(OUT / (f.replace("/", "__") + ".txt"), "w") as fh:
        merged = sorted(
            [(ln, "D", name, text) for ln, k, name, text in entries]
            + [(ln, "C", "", text) for ln, text in com_lines]
        )
        for ln, k, name, text in merged:
            if k == "D":
                fh.write(f"\n@@ {ln} DOC {name}\n{text}\n")
            else:
                fh.write(f"{ln}: {text}\n")

rows.sort()
for r in rows:
    print(f"{r[0]:6d} doc={r[1]:5d} com={r[2]:5d} lines={r[3]:5d} {r[4]}")
print("TOTAL doc lines", tot_doc, "comment lines", tot_com, "of", tot_lines)
print("\nMISSING public docstrings (non-test):", len(missing))
for m in missing:
    print("  ", m)
