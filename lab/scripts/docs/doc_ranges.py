"""Character ranges of documentation inside a Python source file.

``doc_ranges(src)`` returns sorted ``(start, end, kind)`` offsets into
*src* for every comment, every docstring (module, class, function) and
every string constant inside a ``description=`` / ``help=`` keyword:
the only text ``oxford.py`` rewrites.
"""

import ast
import io
import tokenize


def _line_offsets(src):
    offs = [0]
    for line in src.splitlines(keepends=True):
        offs.append(offs[-1] + len(line))
    return offs


def doc_ranges(src):
    offs = _line_offsets(src)

    def pos(line, col):
        return offs[line - 1] + col

    ranges = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            ranges.append((pos(*tok.start), pos(*tok.end), "comment"))
    tree = ast.parse(src)
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
                c = body[0].value
                ranges.append(
                    (
                        pos(c.lineno, c.col_offset),
                        pos(c.end_lineno, c.end_col_offset),
                        "docstring",
                    )
                )
        if isinstance(node, ast.keyword) and node.arg in (
            "description",
            "help",
        ):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(
                    sub.value, str
                ):
                    ranges.append(
                        (
                            pos(sub.lineno, sub.col_offset),
                            pos(sub.end_lineno, sub.end_col_offset),
                            "kwstring",
                        )
                    )
    return sorted(ranges)
