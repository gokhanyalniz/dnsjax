"""Prove that a change touches prose only, file by file.

Usage::

    uv run python lab/scripts/docs/prose_only.py BASE [NEW]
        [--root DIR] [--allow-new PATH ...] [-- FILE ...]

Compares each FILE (default: every file changed between BASE and NEW)
in the BASE version with the NEW version: a git ref, or the working
tree when NEW is omitted.  *DIR* is the repository to read (default:
the one holding this script), so a copy kept outside the tree can
check any worktree.  One line per file, ``OK`` or ``DIFF`` with the
reasons; the exit status is 1 on any ``DIFF``, on a new or untracked
file not named after ``--allow-new``, or on a changed file of a kind
the check does not know.

What must be unchanged, by kind:

- Python: the AST once docstrings are dropped (as ``docs_only.py``
  compares), and further the token stream without comments, blank-line
  tokens and docstrings; the directive comments (``# noqa``,
  ``# type:``, ``# fmt:``, ``# pragma``, ``# ruff:``, ``# isort:``), in
  order, each with the code on its line, since the formatter and the
  linter act on them; the shebang and coding lines; the strings under a
  ``description=`` or ``help=`` keyword, which ``--help`` prints; and
  the docstring structure read at run time: where the file uses
  ``__doc__.splitlines()[0]``, a non-empty first line ending in a full
  stop or colon; where it uses ``__doc__.split("\\n\\n")[k]``,
  paragraph *k* as in BASE; where it passes ``description=__doc__``, no
  ``%(`` introduced (``argparse`` formats ``%(prog)s`` there).
- TOML: the parsed data (``tomllib``).
- YAML, shell scripts and ``.gitattributes``: every changed line is a
  whole-line comment, and the directive lines (shebang, ``#SBATCH``,
  ``##SBATCH``) are identical and in order.  A comment inside a YAML
  block scalar is script text: review those by eye.
- Notebooks: every code cell (source, outputs, execution count), the
  cell kinds and order, and the metadata; Markdown cells may change.
- Markdown: prose throughout, always ``OK``.

``docs_only.py`` is the quick whole-tree check; this one is
file-scoped and stricter.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
import tomllib
from pathlib import Path

#: Directive comments the tools act on: kept, with their line's code.
DIRECTIVE = re.compile(r"#\s*(noqa|type:|fmt:|pragma|ruff:|isort:)")
CODING = re.compile(r"^[ \t\f]*#.*?coding[:=]")
SHELL_DIRECTIVE = re.compile(r"^(#!|#SBATCH|##SBATCH)")
SPLIT_PARA = re.compile(r"__doc__\.split\(\s*[\"']\\n\\n[\"']\s*\)\[(\d+)\]")
COMMENT_LINE_KINDS = (".yml", ".yaml", ".sh", ".slurm", ".gitattributes")


def git(root: Path, *args: str) -> str:
    """Stdout of a git command in *root* (empty on failure)."""
    r = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else ""


def version(root: Path, ref: str | None, path: str) -> str | None:
    """*path* at *ref*, or in the working tree; ``None`` if absent."""
    if ref is None:
        p = root / path
        return p.read_text(encoding="utf-8") if p.exists() else None
    r = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    return r.stdout if r.returncode == 0 else None


def _docstring_nodes(tree: ast.AST) -> list[ast.Expr]:
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


class _Blank(ast.NodeTransformer):
    """Drop docstrings; collect and blank ``description=``/``help=``."""

    def __init__(self) -> None:
        self.kw_strings: list[str] = []

    def _drop(self, node):
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
            if not body and not isinstance(node, ast.Module):
                body.append(ast.Pass())
        self.generic_visit(node)
        return node

    visit_Module = visit_ClassDef = _drop
    visit_FunctionDef = visit_AsyncFunctionDef = _drop

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


def _tokens(src: str, doc_starts: set[tuple[int, int]]):
    """Code tokens, directive comments and the leading magic lines."""
    code, directives = [], []
    line_code: dict[int, list] = {}
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for tok in toks:
        if tok.type in (tokenize.NL, tokenize.COMMENT, tokenize.ENCODING):
            continue
        if tok.type == tokenize.STRING and tok.start in doc_starts:
            continue
        code.append((tok.type, tok.string))
        line_code.setdefault(tok.start[0], []).append(tok.string)
    for tok in toks:
        if tok.type == tokenize.COMMENT and DIRECTIVE.search(tok.string):
            row = tok.start[0]
            directives.append((tok.string, tuple(line_code.get(row, ()))))
    lines = src.splitlines()[:2]
    magic = [
        ln
        for i, ln in enumerate(lines)
        if (i == 0 and ln.startswith("#!")) or CODING.match(ln)
    ]
    return code, directives, magic


def check_python(old: str, new: str) -> list[str]:
    """Reasons *new* differs from *old* beyond prose (empty: none)."""
    why = []
    try:
        t_old, t_new = ast.parse(old), ast.parse(new)
    except SyntaxError as exc:
        return [f"does not parse: {exc}"]
    d_old = {(d.lineno, d.col_offset) for d in _docstring_nodes(t_old)}
    d_new = {(d.lineno, d.col_offset) for d in _docstring_nodes(t_new)}
    b_old, b_new = _Blank(), _Blank()
    a_old = ast.dump(b_old.visit(ast.parse(old)), include_attributes=False)
    a_new = ast.dump(b_new.visit(ast.parse(new)), include_attributes=False)
    if a_old != a_new:
        why.append("code (AST) changed")
    if b_old.kw_strings != b_new.kw_strings:
        why.append("description=/help= strings changed")
    c_old, dir_old, mag_old = _tokens(old, d_old)
    c_new, dir_new, mag_new = _tokens(new, d_new)
    if c_old != c_new and a_old == a_new:
        why.append("code tokens changed")
    if dir_old != dir_new:
        why.append("directive comments changed")
    if mag_old != mag_new:
        why.append("shebang or coding line changed")
    why += _runtime_docstring(t_old, t_new, new)
    return why


def _runtime_docstring(t_old, t_new, new_src: str) -> list[str]:
    why = []
    doc_old = ast.get_docstring(t_old, clean=False) or ""
    doc_new = ast.get_docstring(t_new, clean=False) or ""
    if "__doc__.splitlines()[0]" in new_src:
        first = doc_new.splitlines()[0].strip() if doc_new else ""
        if not first or not first.endswith((".", ":")):
            why.append("runtime-read first docstring line malformed")
    for m in SPLIT_PARA.finditer(new_src):
        k = int(m.group(1))
        p_old = doc_old.split("\n\n")
        p_new = doc_new.split("\n\n")
        if (p_old[k] if k < len(p_old) else None) != (
            p_new[k] if k < len(p_new) else None
        ):
            why.append(f"runtime-read docstring paragraph [{k}] changed")
    if "description=__doc__" in new_src and (
        doc_new.count("%(") > doc_old.count("%(")
    ):
        why.append("'%(' added to a docstring argparse formats")
    return why


def check_toml(old: str, new: str) -> list[str]:
    try:
        same = tomllib.loads(old) == tomllib.loads(new)
    except tomllib.TOMLDecodeError as exc:
        return [f"does not parse: {exc}"]
    return [] if same else ["TOML data changed"]


def check_comment_lines(old: str, new: str) -> list[str]:
    """Changed lines must be whole-line comments; directives fixed."""
    import difflib

    why = []
    a, b = old.splitlines(), new.splitlines()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, a, b, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        for line in a[i1:i2] + b[j1:j2]:
            s = line.strip()
            if s and not s.startswith("#"):
                why.append(f"non-comment line changed: {s[:60]!r}")
                break
    d_old = [ln for ln in a if SHELL_DIRECTIVE.match(ln)]
    d_new = [ln for ln in b if SHELL_DIRECTIVE.match(ln)]
    if d_old != d_new:
        why.append("directive lines changed")
    return why


def check_notebook(old: str, new: str) -> list[str]:
    try:
        nb_old, nb_new = json.loads(old), json.loads(new)
    except json.JSONDecodeError as exc:
        return [f"does not parse: {exc}"]
    why = []
    if nb_old.get("metadata") != nb_new.get("metadata"):
        why.append("notebook metadata changed")
    c_old, c_new = nb_old.get("cells", []), nb_new.get("cells", [])
    if [c["cell_type"] for c in c_old] != [c["cell_type"] for c in c_new]:
        return why + ["cell kinds or order changed"]
    for co, cn in zip(c_old, c_new, strict=True):
        if co["cell_type"] != "markdown" and co != cn:
            why.append("a code cell changed")
            break
        if co["cell_type"] == "markdown" and (
            {k: v for k, v in co.items() if k != "source"}
            != {k: v for k, v in cn.items() if k != "source"}
        ):
            why.append("a Markdown cell's metadata changed")
            break
    return why


def check(path: str, old: str, new: str) -> list[str] | None:
    """Reasons for a ``DIFF``, ``[]`` for ``OK``, ``None`` if unknown."""
    name = Path(path).name
    if path.endswith(".py"):
        return check_python(old, new)
    if path.endswith(".toml"):
        return check_toml(old, new)
    if path.endswith(".ipynb"):
        return check_notebook(old, new)
    if path.endswith(".md"):
        return []
    if path.endswith(COMMENT_LINE_KINDS) or name == ".gitattributes":
        return check_comment_lines(old, new)
    return None


def main(argv: list[str]) -> int:
    files: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, files = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], add_help=True
    )
    ap.add_argument("base")
    ap.add_argument("new", nargs="?")
    ap.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    ap.add_argument("--allow-new", nargs="*", default=[])
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if not files:
        rng = [a.base] + ([a.new] if a.new else [])
        files = git(root, "diff", "--name-only", *rng).split()
        if a.new is None:
            files += git(
                root, "ls-files", "--others", "--exclude-standard"
            ).split()
    bad = 0
    for path in sorted(set(files)):
        old = version(root, a.base, path)
        new = version(root, a.new, path)
        if new is None:
            print(f"GONE {path}")
            bad += 1
            continue
        if old is None:
            ok = path in a.allow_new
            print(f"{'NEW ' if ok else 'NEW!'} {path}")
            bad += 0 if ok else 1
            continue
        if old == new:
            continue
        why = check(path, old, new)
        if why is None:
            print(f"DIFF {path}  [a kind this check does not know]")
            bad += 1
        elif why:
            print(f"DIFF {path}  [{'; '.join(why)}]")
            bad += 1
        else:
            print(f"OK   {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
