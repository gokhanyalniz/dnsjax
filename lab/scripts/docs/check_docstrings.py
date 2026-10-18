"""Mechanical prose checks over the tracked Python tree, bar this directory.

A. docstrings with a backslash that are not raw (escape corruption);
B. inline math in docstrings not in the `$...$`-in-backticks form, and
   `$$` display math;
C. doubled words outside math (each `` `$...$` `` span, which may wrap,
   and each ``.. math::`` block), TODO-like markers and dates, on every
   line; spaced-hyphen dashes (`word - word`, where the tree writes
   ` -- `) in docstrings and comments, outside math, code spans and
   literal blocks, a comment stating an equation (with `=`) left out;
D. tone words (`banned_words.TONE`) in docstrings and comments.
"""

import ast
import io
import re
import subprocess
import sys
import tokenize
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from banned_words import TONE  # noqa: E402

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


#: Inline math, `` `$...$` ``, which may wrap but not across a blank line.
INLINE_MATH = re.compile(r"`\$(?:[^`$\n]|\n(?![ \t]*\n))+\$`")
#: A code span, ``...`` or `...`, on one line.
CODE_SPAN = re.compile(r"``[^`\n]+``|`[^`\n]+`")
#: A hyphen between two words with a space on each side, used as a dash
#: (one-letter words left out, as the symbols of a formula).
SPACED_HYPHEN = re.compile(
    r"(?<![\w\\])[A-Za-z]\w+[,;:.)'\"]? - [A-Za-z]{2,}\b"
)


def _blank(m: re.Match) -> str:
    """*m*'s text as ``~``, its line breaks kept.

    Neither a word nor a space, so the words either side of a blanked
    span do not meet.
    """
    return re.sub(r"[^\n]", "~", m.group(0))


def _blank_blocks(text: str, opener: str) -> str:
    """*text* with the body of each block a line matching *opener* opens.

    The body is every following line indented deeper than the opener,
    and the blank lines among them; the line count is kept.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if not re.search(opener, lines[i]):
            i += 1
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        i += 1
        while i < len(lines) and (
            not lines[i].strip()
            or len(lines[i]) - len(lines[i].lstrip()) > indent
        ):
            lines[i] = ""
            i += 1
    return "\n".join(lines)


def without_math(text: str) -> str:
    """*text* with its math blanked: inline spans and ``.. math::`` blocks."""
    text = INLINE_MATH.sub(_blank, text)
    return _blank_blocks(text, r"^\s*\.\. math::")


def prose(text: str) -> str:
    """*text* (a docstring) without math, code spans and literal blocks."""
    text = CODE_SPAN.sub(_blank, without_math(text))
    return _blank_blocks(text, r"::\s*$")


nonraw = []
math_issues = []
doubled = []
markers = []
dates = []
dashes = []
tone = Counter()
tone_where = {}
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
        for j, line in enumerate(prose(body).split("\n")):
            if SPACED_HYPHEN.search(line):
                dashes.append((f, d.lineno + j, line.strip()[:100]))
    masked = without_math(src).split("\n")
    for i, line in enumerate(lines, 1):
        for m in re.finditer(r"\b(\w+)\s+\1\b", masked[i - 1]):
            w = m.group(1)
            if w.lower() in {"that", "had"} or w.isdigit():
                continue
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
            row, col = tok.start
            text = CODE_SPAN.sub(_blank, masked[row - 1][col:])
            # A comment stating an equation is a formula, not prose.
            if "=" not in text and SPACED_HYPHEN.search(text):
                dashes.append((f, row, tok.string.strip()[:100]))
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
print("\nC4. spaced-hyphen dashes:", len(dashes))
for x in dashes:
    print("  ", x)
print("\nD. tone words:")
for w, n in tone.most_common():
    print(f"  {w:16s} {n:4d}  e.g. {tone_where[w][:3]}")
