"""Normalize module-docstring section underlines from '===' to '---'.

Usage: python underline.py FILE [FILE ...]
Only lines inside the module docstring that consist solely of '='
and directly follow a non-empty title line are rewritten, to the
title's length.
"""

import ast
import re
import sys
from pathlib import Path

for path in sys.argv[1:]:
    src = Path(path).read_text()
    tree = ast.parse(src)
    node = tree.body[0] if tree.body else None
    if not (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ):
        print(f"{path}: no module docstring")
        continue
    lo, hi = node.lineno - 1, node.end_lineno  # 0-based slice
    lines = src.split("\n")
    n = 0
    for i in range(lo + 1, hi):
        if re.fullmatch(r"=+", lines[i]) and lines[i - 1].strip():
            lines[i] = "-" * len(lines[i - 1])
            n += 1
    Path(path).write_text("\n".join(lines))
    print(f"{path}: {n} underline(s)")
