#!/usr/bin/env python3
"""Check that every dnsjax / dnsjax_lab import in the repository resolves.

Usage::

    check_imports.py [REPO]

Maps the dotted name of each ``import`` or ``from ... import`` (a
relative one resolved against the importing module) onto
``src/dnsjax`` or ``lab/src/dnsjax_lab``, and reports a name that is no
module, and a ``from X import Y`` whose ``Y`` is neither a submodule of
``X`` nor a name ``X`` defines at its top level (a definition, an
assignment or an import, those under an ``if`` or a ``try`` included).
That is the check a module move needs before anything runs: the
importers of a moved file, the paths a rewrite missed, a re-export that
got lost.  Nothing is imported, so it is fast and needs no environment;
it is best-effort too, blind to a name made by ``globals()`` or a star
import.  Every tracked or new Python file of *REPO* (default: the
repository holding this script) is read; exit status 1 on any finding.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

#: The repository holding this script (``lab/scripts/refactor/``).
DEFAULT_REPO = Path(__file__).resolve().parents[3]

#: What :func:`module_file` returns for a module that does not exist.
MISSING = Path("<missing>")


def package_roots(repo: Path) -> dict[str, Path]:
    """The top-level package name -> its source directory."""
    return {
        "dnsjax": repo / "src" / "dnsjax",
        "dnsjax_lab": repo / "lab" / "src" / "dnsjax_lab",
    }


def module_file(name: str, roots: dict[str, Path]) -> Path | None:
    """The source file of module *name*.

    ``None`` for a module outside the two packages, :data:`MISSING` for
    one inside them that does not exist.
    """
    parts = name.split(".")
    root = roots.get(parts[0])
    if root is None:
        return None
    path = root.joinpath(*parts[1:])
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    return MISSING


def module_name(path: Path, roots: dict[str, Path]) -> str | None:
    """The dotted name of the module in *path* (``None``: no package)."""
    for top, root in roots.items():
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = [top, *rel.with_suffix("").parts]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)
    return None


def _assigned(target: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}


_names_cache: dict[Path, set[str]] = {}


def defined_names(path: Path) -> set[str]:
    """The names the module in *path* defines at its top level."""
    if path in _names_cache:
        return _names_cache[path]
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        nodes = [node]
        if isinstance(node, (ast.If, ast.Try)):
            nodes = list(ast.walk(node))
        for sub in nodes:
            if isinstance(
                sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                names.add(sub.name)
            elif isinstance(sub, ast.Assign):
                for target in sub.targets:
                    names |= _assigned(target)
            elif isinstance(sub, ast.AnnAssign):
                names |= _assigned(sub.target)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for alias in sub.names:
                    names.add((alias.asname or alias.name).split(".")[0])
    _names_cache[path] = names
    return names


def absolute_name(
    node: ast.ImportFrom, path: Path, roots: dict[str, Path]
) -> str | None:
    """The absolute module name of *node* (``None``: no package)."""
    if not node.level:
        return node.module or ""
    mod = module_name(path, roots)
    if mod is None:
        return None
    base = mod.split(".")
    # A package's own name is its first level; a module's is its parent.
    drop = node.level - 1 if path.name == "__init__.py" else node.level
    base = base[: len(base) - drop] if drop else base
    return ".".join(base + ([node.module] if node.module else []))


def check_file(path: Path, rel: str, roots: dict[str, Path]) -> list[str]:
    """The unresolved imports of the file at *path*."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:
        return [f"{rel}: syntax error: {exc}"]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if module_file(alias.name, roots) is MISSING:
                    out.append(f"{rel}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            name = absolute_name(node, path, roots)
            if name is None:
                out.append(f"{rel}:{node.lineno}: relative import outside")
                continue
            mfile = module_file(name, roots)
            if mfile is None:
                continue
            if mfile is MISSING:
                out.append(f"{rel}:{node.lineno}: from {name} import ...")
                continue
            for alias in node.names:
                sub = module_file(f"{name}.{alias.name}", roots)
                if alias.name == "*" or sub not in (None, MISSING):
                    continue
                if alias.name not in defined_names(mfile):
                    out.append(
                        f"{rel}:{node.lineno}: {name} has no {alias.name}"
                    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("repo", type=Path, nargs="?", default=DEFAULT_REPO)
    repo = ap.parse_args().repo.resolve()
    roots = package_roots(repo)
    files = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "*.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    findings = []
    for rel in files:
        path = repo / rel
        if path.is_file():
            findings += check_file(path, rel, roots)
    for line in findings:
        print(line)
    print(f"{len(files)} files, {len(findings)} unresolved")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
