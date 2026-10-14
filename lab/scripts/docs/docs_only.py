"""Prove a change to the Python tree touches documentation only.

Usage: python docs_only.py <base-ref> [--allow-annotations FILE ...]

For every ``.py`` changed between <base-ref> and the working tree, parse
both versions, blank every docstring and every string constant inside a
``description=`` / ``help=`` keyword, and compare the ASTs (comments are
not in the AST).  Annotation differences are allowed only in the files
named after ``--allow-annotations`` (then blanked too).  Prints one line
per file and exits non-zero on any other difference.
"""

import ast
import difflib
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])  # the repository root
base = sys.argv[1]
allow_ann = set()
if "--allow-annotations" in sys.argv:
    allow_ann = set(sys.argv[sys.argv.index("--allow-annotations") + 1 :])

changed = subprocess.run(
    ["git", "diff", "--name-only", base, "--", "*.py"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
).stdout.split()
untracked = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
).stdout.split()


class Blank(ast.NodeTransformer):
    def __init__(self, strip_ann):
        self.strip_ann = strip_ann
        self.kw_strings = []

    def _blank_doc(self, node):
        # Drop the docstring statement itself, so adding or removing a
        # docstring is as invisible as editing one.
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
            if not body and not isinstance(node, ast.Module):
                body.append(ast.Pass())

    def visit_Module(self, node):
        self._blank_doc(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        self._blank_doc(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        self._blank_doc(node)
        if self.strip_ann:
            node.returns = None
            for a in (
                node.args.args + node.args.kwonlyargs + node.args.posonlyargs
            ):
                a.annotation = None
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_keyword(self, node):
        if node.arg in ("description", "help"):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(
                    sub.value, str
                ):
                    self.kw_strings.append(sub.value)
                    sub.value = ""
        self.generic_visit(node)
        return node


def norm(src, strip_ann):
    b = Blank(strip_ann)
    tree = b.visit(ast.parse(src))
    return ast.dump(tree, include_attributes=False, indent=1), b.kw_strings


bad = 0
for path in changed:
    try:
        old = subprocess.run(
            ["git", "show", f"{base}:{path}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        print(f"NEW  {path} (no base version)")
        continue
    try:
        new = Path(f"{ROOT}/{path}").read_text()
    except FileNotFoundError:
        print(f"GONE {path}")
        bad += 1
        continue
    strip = path in allow_ann
    a, ka = norm(old, strip)
    b, kb = norm(new, strip)
    tag = []
    if ka != kb:
        tag.append("help/description strings changed")
    if strip:
        tag.append("annotations ignored")
    if a == b:
        print(f"OK   {path}" + (f"  [{'; '.join(tag)}]" if tag else ""))
    else:
        bad += 1
        print(f"DIFF {path}")
        for line in list(
            difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="")
        )[:40]:
            print("     " + line)
for path in untracked:
    print(f"UNTRACKED {path}")
sys.exit(1 if bad else 0)
