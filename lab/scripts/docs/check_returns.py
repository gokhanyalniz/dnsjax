"""Check return-tuple arity and documented parameters against the code.

For each function of the tracked Python tree:

- a fixed-arity ``-> tuple[A, B, C]`` against each tuple literal its own
  body returns (``return (x, y, ...)``; nested functions aside);
- a docstring's ``Return``/``Returns``/``->`` before a double-backticked
  tuple, ``(a, b, c)``, against those returns when they share one arity;
- a numpy-style ``Parameters`` (``Args``, ``Arguments``) section's
  ``name:`` lines against the signature: a name it documents that the
  signature lacks and, in a public function (no leading ``_``), a
  parameter but ``self`` / ``cls`` it omits.
"""

import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository root
files = subprocess.run(
    ["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True
).stdout.split()


def tuple_arity(ann):
    if (
        isinstance(ann, ast.Subscript)
        and isinstance(ann.value, ast.Name)
        and ann.value.id == "tuple"
    ):
        sl = ann.slice
        if isinstance(sl, ast.Tuple):
            if any(
                isinstance(e, ast.Constant) and e.value is Ellipsis
                for e in sl.elts
            ):
                return None
            return len(sl.elts)
        return 1
    return None


def own_returns(fn):
    out = []

    def walk(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(
                ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            ):
                continue
            if isinstance(ch, ast.Return) and ch.value is not None:
                out.append(ch)
            walk(ch)

    walk(fn)
    return out


def ret_arity(r):
    v = r.value
    if isinstance(v, ast.Tuple):
        if any(isinstance(e, ast.Starred) for e in v.elts):
            return None
        return len(v.elts)
    return None


DOC_RET = re.compile(r"(?:Returns?|->)\s*``\(([^`]*)\)``", re.S)

for f in files:
    tree = ast.parse((ROOT / f).read_text())
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        rets = [ret_arity(r) for r in own_returns(fn)]
        rets_known = {a for a in rets if a is not None}
        ann = tuple_arity(fn.returns) if fn.returns is not None else None
        if ann is not None and rets_known and rets_known != {ann}:
            print(
                f"{f}:{fn.lineno}: {fn.name}: annotation tuple[{ann}] "
                f"vs returns {sorted(rets_known)}"
            )
        doc = ast.get_docstring(fn) or ""
        for m in DOC_RET.finditer(doc):
            inner = m.group(1)
            # count top-level commas
            depth = 0
            n = 1
            for ch in inner:
                if ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth -= 1
                elif ch == "," and depth == 0:
                    n += 1
            if rets_known and n not in rets_known and len(rets_known) == 1:
                print(
                    f"{f}:{fn.lineno}: {fn.name}: docstring says {n}-tuple "
                    f"({inner[:60]}) vs returns {sorted(rets_known)}"
                )
        # documented params not in signature (numpy style)
        argnames = {
            a.arg
            for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs
        }
        if fn.args.vararg:
            argnames.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            argnames.add(fn.args.kwarg.arg)
        sec = re.search(
            r"\n\s*(?:Parameters|Args|Arguments)\s*\n\s*-{3,}\n(.*?)"
            r"(?:\n\s*\n\s*(?:Returns|Raises|Yields|Notes|Examples|See Also)"
            r"\s*\n\s*-{3,}|\Z)",
            doc,
            re.S,
        )
        if sec:
            # Entry names sit at the section's own indentation (a wrapped
            # description line indented deeper is never a name); an RST
            # escape (``name\_``) reads as the name.
            body = [
                ln.replace("\\_", "_")
                for ln in sec.group(1).splitlines()
                if ln.strip()
            ]
            indent = len(body[0]) - len(body[0].lstrip()) if body else 0
            documented = [
                name.strip("* ")
                for ln in body
                if len(ln) - len(ln.lstrip()) == indent
                for group in re.findall(
                    r"^\s*((?:\**[A-Za-z_]\w*\**\s*,\s*)*"
                    r"\**[A-Za-z_]\w*\**)\s*(?::|$)",
                    ln,
                )
                for name in group.split(",")
            ]
            for d in documented:
                if d not in argnames:
                    print(
                        f"{f}:{fn.lineno}: {fn.name}: documents '{d}' "
                        f"not in signature {sorted(argnames)[:8]}"
                    )
            # signature params the section omits (public functions only)
            if not fn.name.startswith("_"):
                for d in sorted(argnames - set(documented) - {"self", "cls"}):
                    print(
                        f"{f}:{fn.lineno}: {fn.name}: Parameters section "
                        f"omits '{d}'"
                    )
