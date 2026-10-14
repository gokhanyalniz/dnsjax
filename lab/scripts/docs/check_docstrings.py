"""Mechanical docstring checks over the tracked Python tree, bar this
directory.

A. docstrings with a backslash that are not raw (escape corruption);
B. inline math in docstrings not in the `$...$`-in-backticks form, and
   `$$` / other display forms;
C. doubled words, spaced-hyphen dashes, TODO-like markers, dates;
D. banned / tone words.
"""

import ast
import io
import re
import subprocess
import tokenize
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
files = subprocess.run(
    ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True
).stdout.split()
# The checkers quote the patterns they look for: leave them out.
files = [f for f in files if not f.startswith("lab/scripts/docs/")]


def docstring_nodes(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.append(body[0])
    return out


nonraw = []
math_issues = []
doubled = []
markers = []
dates = []
tone = Counter()
tone_where = {}
TONE = re.compile(
    r"\b(flagship|oracle|anchor(?:s|ed|ing)?|hand-written|handwritten|"
    r"honest(?:ly)?|crucial(?:ly)?|notably|importantly|seamless(?:ly)?|"
    r"leverag(?:e|es|ed|ing)|robust(?:ly|ness)?|comprehensive|"
    r"cutting-edge|state-of-the-art|delv(?:e|es|ing)|simply|obviously|"
    r"clearly|trivially|elegant(?:ly)?|powerful|magic(?:al)?|"
    r"essentially|basically|literally|very|really|actually|just)\b",
    re.I,
)
for f in files:
    src = (ROOT / f).read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    for d in docstring_nodes(tree):
        raw_src = "\n".join(lines[d.lineno - 1 : d.end_lineno])
        seg = raw_src[d.col_offset :] if d.lineno == d.end_lineno else raw_src
        seg = seg.lstrip()
        prefix = re.match(r"^([rRbBuUfF]*)(\"\"\"|'''|\"|')", seg)
        pre = prefix.group(1).lower() if prefix else "?"
        body = d.value.value
        if "r" not in pre and "\\" in raw_src:
            nonraw.append((f, d.lineno))
        # math forms
        if re.search(r"\$\$", body):
            math_issues.append((f, d.lineno, "$$ display"))
        # inline $...$ not wrapped in backticks
        for m in re.finditer(r"(?<!`)\$[^$\n`]+\$(?!`)", body):
            math_issues.append((f, d.lineno, "bare $..$: " + m.group(0)[:50]))
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r"\b(\w+)\s+\1\b", line):
            w = m.group(1)
            if w.lower() in {"that", "had"} or w.isdigit():
                continue
            # skip code-ish lines
            doubled.append((f, i, line.strip()[:100]))
        if re.search(r"\b(TODO|FIXME|XXX|HACK)\b", line):
            markers.append((f, i, line.strip()[:100]))
        if re.search(r"\b20(2[4-9])-\d\d-\d\d\b", line):
            dates.append((f, i, line.strip()[:110]))
    # tone words in docstrings and comments only
    for d in docstring_nodes(tree):
        for m in TONE.finditer(d.value.value):
            w = m.group(1).lower()
            tone[w] += 1
            tone_where.setdefault(w, []).append((f, d.lineno))
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            for m in TONE.finditer(tok.string):
                w = m.group(1).lower()
                tone[w] += 1
                tone_where.setdefault(w, []).append((f, tok.start[0]))

print("A. non-raw docstrings containing a backslash:", len(nonraw))
for x in nonraw:
    print("  ", x)
print("\nB. math form issues:", len(math_issues))
for x in math_issues[:80]:
    print("  ", x)
print("\nC1. doubled words:", len(doubled))
for x in doubled:
    print("  ", x)
print("\nC2. TODO-like markers:", len(markers))
for x in markers:
    print("  ", x)
print("\nC3. dated lines:", len(dates))
for x in dates:
    print("  ", x)
print("\nD. tone words:")
for w, n in tone.most_common():
    print(f"  {w:16s} {n:4d}  e.g. {tone_where[w][:3]}")
