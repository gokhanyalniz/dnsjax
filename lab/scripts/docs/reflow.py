"""Reflow one paragraph of a docstring/comment in place.

Usage: python reflow.py FILE START_SUBSTR END_SUBSTR [WIDTH]

The paragraph runs from the line containing START_SUBSTR to the line
containing END_SUBSTR (inclusive).  Keeps the first line's indentation
(and a leading '# ' comment marker), keeps the original separator width
(one or two spaces) between tokens, and never breaks inside ``...`` or
`$...$` spans.
"""

import re
import sys
from pathlib import Path


def tokens(text):
    # (token, separator-before) pairs; separator is 1 or 2 spaces.
    out = []
    for m in re.finditer(r"(\s*)(\S+)", text):
        sep = m.group(1)
        out.append(
            [
                m.group(2),
                2 if sep.endswith("  ") and "\n" not in sep[-2:] else 1,
            ]
        )
    return out


def merge_spans(toks):
    merged = []
    open_lit = open_math = False
    for tok, sep in toks:
        if merged and (open_lit or open_math):
            merged[-1][0] += " " + tok
        else:
            merged.append([tok, sep])
        n_lit = tok.count("``")
        if n_lit % 2:
            open_lit = not open_lit
        n_math = len(re.findall(r"`\$|\$`", tok))
        if n_math % 2:
            open_math = not open_math
    return merged


def main():
    path, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    width = int(sys.argv[4]) if len(sys.argv) > 4 else 79
    lines = Path(path).read_text().split("\n")
    i = next(k for k, line in enumerate(lines) if start in line)
    j = next(k for k in range(i, len(lines)) if end in lines[k])
    first = lines[i]
    indent = re.match(r"\s*(#\s)?", first).group(0)
    body = []
    for line in lines[i : j + 1]:
        assert line.startswith(indent.rstrip()), line
        body.append(
            line[len(indent) :]
            if line.startswith(indent)
            else line.strip().lstrip("#").strip()
        )
    # Rebuild the text with original separators: join lines with a
    # single space unless the line ended a sentence with two spaces
    # convention (approximate: a line ending in '.' gets two).
    text = ""
    for k, b in enumerate(body):
        if k:
            prev = body[k - 1]
            text += "  " if re.search(r"[.!?][)\"']?$", prev) else " "
        text += b
    toks = merge_spans(tokens(text))
    avail = width - len(indent)
    split = []
    for tok, sep in toks:
        if len(tok) > avail and " " in tok:
            parts = tok.split(" ")
            split.append([parts[0], sep])
            split.extend([q, 1] for q in parts[1:])
        else:
            split.append([tok, sep])
    toks = split
    out, cur = [], indent
    for tok, sep in toks:
        cand = cur + tok if cur == indent else cur + " " * sep + tok
        if len(cand) > width and cur != indent:
            out.append(cur)
            cur = indent + tok
        else:
            cur = cand
    out.append(cur)
    lines[i : j + 1] = out
    Path(path).write_text("\n".join(lines))
    print("\n".join(out))


main()
