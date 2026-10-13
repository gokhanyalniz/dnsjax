r"""The core never names the lab (offline, no JAX, a few seconds).

dnsjax releases its core -- ``src/dnsjax``, ``tests/`` and
``scripts/`` -- on its own.  The lab (``lab/``, the ``dnsjax_lab``
package) is a plugin that registers its flows, its parameter fields
and its extension sections through the core's entry-point groups
(:mod:`dnsjax.flows.registry`, :mod:`dnsjax.parameters`,
:mod:`dnsjax.extensions`).  The core works without it only while no
core code names it, which this script pins by parsing every core
Python file (never importing it):

- ``imports``: no ``import`` of ``dnsjax_lab``, and no string starting
  with ``dnsjax_lab`` (what a dynamic import or a ``module:name``
  reference would take).
- ``systems``: no string equal to a flow system the lab registers or
  to one of its console scripts.
- ``fields``: no access to a field the lab adds to a core section:
  ``<x>.<section>.<field>``, a ``("<section>", "<field>")`` pair (a
  tuple, or a call's first two arguments), or a
  ``"[--]<section>.<field>"`` string.
- ``sections``: no use of a parameter section the lab registers: a
  ``"[--]<section>.<field>"`` or ``"[<section>]"`` string, or a
  ``[...]`` / ``.get(...)`` lookup keyed by the section's name.
- ``registry``: the flows the entry points add to the registry are
  exactly the lab's, and the core's own specs are disjoint from them.
- ``self``: each pattern above is found in a source written to break
  it, from the names the lab registers, so a check that silently
  stopped matching fails here rather than passing everything.

Every check but ``imports`` needs the lab's names, read from its entry
points; without ``dnsjax-lab`` installed (a core-only checkout) they
are skipped.  Strings are compared whole, never searched: prose may
point at the lab (a docstring naming ``lab/tests/...``), code may not.
The lab's section and field names alone are too generic to compare
that way (``force``, ``beta``), hence the access patterns of
``fields`` and ``sections``.  This file names ``dnsjax_lab`` and
``dnsjax-lab`` to find the lab, so it is not scanned itself.

Run::

    uv run python tests/test_layering.py
"""

from __future__ import annotations

import ast
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from _live import report

REPO = Path(__file__).resolve().parent.parent
#: The core's Python trees, scanned whole.
CORE_ROOTS = ("src/dnsjax", "tests", "scripts")
#: The lab's import package and distribution.
LAB_PACKAGE = "dnsjax_lab"
LAB_DIST = "dnsjax-lab"


def core_files() -> list[Path]:
    """Every core Python file but this one."""
    this = Path(__file__).resolve()
    return [
        p
        for root in CORE_ROOTS
        for p in sorted((REPO / root).rglob("*.py"))
        if p.resolve() != this
    ]


def lab_names() -> dict | None:
    """What the lab registers, from its entry points (``None``: absent).

    ``systems`` (its flows), ``scripts`` (its console scripts),
    ``fields`` (``section -> names`` it adds to the core sections) and
    ``sections`` (the parameter sections it registers, those whose
    model it defines).
    """
    try:
        eps = distribution(LAB_DIST).entry_points
    except PackageNotFoundError:
        return None
    from dnsjax.flow_spec import FlowSpec
    from dnsjax.flows.registry import FLOWS_ENTRY_POINT_GROUP

    systems: set[str] = set()
    for ep in eps.select(group=FLOWS_ENTRY_POINT_GROUP):
        obj = ep.load()
        specs = (obj,) if isinstance(obj, FlowSpec) else tuple(obj)
        systems |= {s.system for s in specs}
    fields: dict[str, set[str]] = {}
    for ep in eps.select(group="dnsjax.parameters"):
        for section, named in ep.load().items():
            fields.setdefault(section, set()).update(named)
    scripts = {ep.name for ep in eps.select(group="console_scripts")}
    # Importing the registry loads the lab's extension modules.
    from dnsjax.extensions import EXTENSIONS

    sections = {
        name
        for name, ext in EXTENSIONS.items()
        if ext.model.__module__.split(".")[0] == LAB_PACKAGE
    }
    return {
        "systems": systems,
        "scripts": scripts,
        "fields": fields,
        "sections": sections,
    }


def _strings(tree: ast.AST):
    """The string constants of *tree* (f-string parts included)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node


def _const_pair(nodes: list[ast.expr]) -> tuple[str, str] | None:
    if all(
        isinstance(n, ast.Constant) and isinstance(n.value, str) for n in nodes
    ):
        return nodes[0].value, nodes[1].value
    return None


def find_imports(tree: ast.AST) -> list[tuple[int, str]]:
    """``(line, what)`` per import of the lab or string naming it."""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods = [node.module or ""]
        else:
            continue
        hits += [
            (node.lineno, f"import {m}")
            for m in mods
            if m.split(".")[0] == LAB_PACKAGE
        ]
    hits += [
        (n.lineno, repr(n.value[:60]))
        for n in _strings(tree)
        if n.value.startswith(LAB_PACKAGE)
    ]
    return hits


def find_systems(tree: ast.AST, names: dict) -> list[tuple[int, str]]:
    """``(line, what)`` per string equal to a lab system or script."""
    taken = names["systems"] | names["scripts"]
    return [
        (n.lineno, repr(n.value)) for n in _strings(tree) if n.value in taken
    ]


def find_fields(tree: ast.AST, names: dict) -> list[tuple[int, str]]:
    """``(line, what)`` per access to a field the lab adds."""
    fields = names["fields"]
    hits = []
    for node in ast.walk(tree):
        pair = None
        if isinstance(node, ast.Attribute) and isinstance(
            node.value, ast.Attribute
        ):
            pair = (node.value.attr, node.attr)
        elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
            pair = _const_pair(node.elts)
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            pair = _const_pair(node.args[:2])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            section, _, name = node.value.lstrip("-").partition(".")
            pair = (section, name)
        if pair is not None and pair[1] in fields.get(pair[0], ()):
            hits.append((node.lineno, ".".join(pair)))
    return hits


def find_sections(tree: ast.AST, names: dict) -> list[tuple[int, str]]:
    """``(line, what)`` per use of a section the lab registers."""
    sections = names["sections"]
    hits = []
    for node in ast.walk(tree):
        key = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.lstrip("-")
            if text.startswith("[") and text.endswith("]"):
                key = text[1:-1]
            else:
                section, dot, name = text.partition(".")
                if dot and name.isidentifier():
                    key = section
        elif isinstance(node, ast.Subscript):
            key = _const_str(node.slice)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
        ):
            key = _const_str(node.args[0])
        if key in sections:
            hits.append((node.lineno, f"section {key!r}"))
    return hits


def _const_str(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def check_registry(names: dict) -> list[str]:
    """The registry's non-core flows are exactly the lab's."""
    from dnsjax.flows import registry
    from dnsjax.flows.triply_periodic.specs import SPECS as periodic
    from dnsjax.flows.wall_bounded.specs import SPECS as wall

    core = {s.system for s in (*periodic, *wall)}
    added = set(registry.SPECS) - core
    problems = []
    if core & names["systems"]:
        problems.append(f"core specs hold lab flows {core & names['systems']}")
    if added != names["systems"]:
        problems.append(
            f"entry points add {sorted(added)}, the lab registers "
            f"{sorted(names['systems'])}"
        )
    return problems


def check_self(names: dict) -> list[str]:
    """Each pattern is found in a source written to break it."""
    system = sorted(names["systems"])[0]
    script = sorted(names["scripts"])[0] if names["scripts"] else system
    section = sorted(names["fields"])[0]
    field = sorted(names["fields"][section])[0]
    lab_section = sorted(names["sections"])[0]
    cases = {
        "import": (f"import {LAB_PACKAGE}.x", find_imports),
        "from-import": (f"from {LAB_PACKAGE} import x", find_imports),
        "dynamic import": (f"m = '{LAB_PACKAGE}.x:f'", find_imports),
        "system": (f"s = '{system}'", find_systems),
        "script": (f"s = ['{script}']", find_systems),
        "attribute": (f"v = params.{section}.{field}", find_fields),
        "pair": (f"p = ('{section}', '{field}')", find_fields),
        "call": (f"f('{section}', '{field}')", find_fields),
        "dotted": (f"a = ['--{section}.{field}', '1']", find_fields),
        "section flag": (f"a = ['--{lab_section}.x', '1']", find_sections),
        "section header": (f"t = '[{lab_section}]'", find_sections),
        "section lookup": (f"e = EXTENSIONS['{lab_section}']", find_sections),
        "section get": (f"v = dump.get('{lab_section}')", find_sections),
    }
    missed = []
    for case, (source, find) in cases.items():
        tree = ast.parse(source)
        hits = find(tree) if find is find_imports else find(tree, names)
        if not hits:
            missed.append(f"{case} not found in {source!r}")
    return missed


def main() -> int:
    files = core_files()
    trees = [(p, ast.parse(p.read_text(), str(p))) for p in files]
    names = lab_names()
    print(f"{len(files)} core files under {', '.join(CORE_ROOTS)}")
    if names is None:
        print(f"{LAB_DIST} is not installed: the import check only.")

    def scan(find, *extra) -> list[str]:
        return [
            f"{p.relative_to(REPO)}:{line}: {what}"
            for p, tree in trees
            for line, what in find(tree, *extra)
        ]

    checks = {
        "imports": lambda: scan(find_imports),
        "systems": lambda: scan(find_systems, names),
        "fields": lambda: scan(find_fields, names),
        "sections": lambda: scan(find_sections, names),
        "registry": lambda: check_registry(names),
        "self": lambda: check_self(names),
    }
    passed, failures = 0, []
    for name, check in checks.items():
        if names is None and name != "imports":
            print(f"  SKIP  {name}")
            continue
        problems = check()
        if problems:
            print(f"  FAIL  {name}:")
            for problem in problems:
                print(f"    {problem}")
            failures.append((name, f"{len(problems)}: {problems[0]}"))
        else:
            print(f"  PASS  {name}")
            passed += 1
    return report(passed, failures)


if __name__ == "__main__":
    sys.exit(main())
