"""Flag docstring/comment style issues in the given Python files.

Usage: python lint_docs.py FILE [FILE ...]
Reports: multi-line docstring summaries, filler/banned words in
docstrings and comments, and Sphinx roles split across lines.
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from banned_words import WORDS  # noqa: E402

ROLE = re.compile(r":(?:func|mod|class|meth|data|attr|obj):`[^`\n]*\n[^`]*?`")

for path in sys.argv[1:]:
    src = Path(path).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            first = doc.strip("\n").split("\n")[0].strip()
            lines = doc.strip("\n").split("\n")
            if (
                len(lines) > 1
                and lines[1].strip()
                and not first.endswith((".", ":", "?"))
            ):
                ln = node.body[0].lineno
                print(f"{path}:{ln}: multi-line summary: {first[:60]!r}")
            for m in WORDS.finditer(doc):
                ln = node.body[0].lineno + doc[: m.start()].count("\n")
                print(f"{path}:{ln}: word {m.group(0)!r}")
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            for m in WORDS.finditer(tok.string):
                print(f"{path}:{tok.start[0]}: comment word {m.group(0)!r}")
    for m in ROLE.finditer(src):
        print(f"{path}:{src.count(chr(10), 0, m.start()) + 1}: split role")
