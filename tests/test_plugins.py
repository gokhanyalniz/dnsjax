#!/usr/bin/env python3
r"""Plugin discovery: the flows, fields and sections a package adds.

A package extends dnsjax through three entry-point groups
(``docs/extending.md``): ``dnsjax.flows`` (read by
:mod:`dnsjax.flows.registry`), ``dnsjax.parameters`` (by
:mod:`dnsjax.parameters`) and ``dnsjax.extensions`` (by
:mod:`dnsjax.extensions`), each once, when its module is imported.  The
repository's lab is such a package, but a release ships without it, so
this script stands in for one.  Each case runs a child whose
``importlib.metadata.entry_points`` returns the case's own entry points
alone (an installed lab's are hidden), naming modules the case writes
to a scratch directory, and imports the core under them.  Offline, no
JAX, a few seconds.

1. ``discovery``: a plugin's flows are listed after the core flows of
   their geometry, in the plugin's order; a field a plugin adds to
   ``[phys]`` is on the composed model, and with the plugin's section,
   on its flow's ``--help`` and among the ``phys.system`` choices.
2. The refusals, each naming its cause:

   - ``flow-load``: a flow plugin whose module fails to import;
   - ``flow-type``: one naming something other than ``FlowSpec``
     objects, and ``flow-object``: one naming no iterable at all;
   - ``flow-twice``: a plugin flow with a core flow's system name, and
     ``flow-twice-plugins``: one two plugins both register;
   - ``flow-geometry``: a plugin flow on a geometry the core lacks;
   - ``flow-field``: a spec listing a field no parameter section
     defines (it would otherwise vanish from every surface);
   - ``field-core``: a field that redefines a core field;
   - ``field-twice``: a field two plugins add;
   - ``field-section``: a field in a section the core does not have;
   - ``field-object`` and ``field-shape``: a parameter plugin naming
     no mapping, or a definition that is not ``(annotation,
     FieldInfo)``;
   - ``section-core``: an extension section named like a core section;
   - ``section-absent``: a resume from a snapshot recording a section
     no installed package registers (a plugin's, in use when it was
     written) refuses, naming the section and ``init.force_resume``.

3. The flow-module hook lookup (:func:`dnsjax.flows.registry.flow_hook`):
   ``hook-none``, a spec naming no ``flow_module``, is refused by
   name; ``hook-masked``, an ``AttributeError`` raised *inside* the
   flow module's import, reaches the caller as itself rather than as a
   "does not export" verdict on the hook.
4. ``spec-labels``, in process: a spec storing more than three
   components without labelling them, a cylindrical one without their
   parity, and labels of the wrong length are refused at construction
   (:class:`dnsjax.flow_spec.FlowSpec`).

Run::

    uv run python tests/test_plugins.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import report  # noqa: E402

#: The child: replace ``entry_points`` before anything imports it, with
#: one returning the case's entry points (argv[1], JSON
#: ``{group: [[name, value], ...]}``), then run the case's body
#: (argv[2]), which prints one JSON line.
CHILD = r"""
import importlib.metadata as md, json, sys
EPS = json.loads(sys.argv[1])
def entry_points(**kwargs):
    group = kwargs.get("group")
    return [
        md.EntryPoint(name=n, value=v, group=group)
        for n, v in EPS.get(group, [])
    ]
md.entry_points = entry_points
exec(sys.argv[2])
"""

#: The plugin modules a case may name, written to its scratch directory
#: (the child's working directory, so first on its ``sys.path``).
MODULES = {
    "plugflows": """
from dnsjax.flow_spec import FieldSpec, FlowSpec

SPECS = (
    FlowSpec(
        system="plug-cartesian",
        geometry="cartesian",
        summary="A plugin flow on a core geometry.",
        fields=(FieldSpec("phys", "plug_knob"),),
    ),
    FlowSpec(
        system="plug-annular",
        geometry="annular",
        summary="A plugin flow on another core geometry.",
        fields=(),
    ),
)
NOT_SPECS = ("plug-cartesian",)
NUMBER = 42
TWICE = (
    FlowSpec(
        system="pipe",
        geometry="cylindrical",
        summary="A plugin flow named like a core one.",
        fields=(),
    ),
)
MISSING = (
    FlowSpec(
        system="plug-missing",
        geometry="cartesian",
        summary="A plugin flow listing a field no section defines.",
        fields=(FieldSpec("phys", "no_such_knob"),),
    ),
)
MASKED = (
    FlowSpec(
        system="plug-masked",
        geometry="cartesian",
        summary="A plugin flow whose module fails at import.",
        fields=(),
        flow_module="plugmasked",
    ),
)
""",
    "plugfields": """
from pydantic import Field

FIELDS = {"phys": {"plug_knob": (float, Field(1.5, description="Knob."))}}
CORE = {"phys": {"re": (float, Field(1.0, description="Clash."))}}
SECTION = {"plugsec": {"level": (int, Field(0, description="Where."))}}
TUPLE = ("phys",)
SHAPE = {"phys": {"knob": 1.5}}
""",
    "plugsections": """
from pydantic import BaseModel

from dnsjax.extensions import ParamExtension, register_extension


class PlugSection(BaseModel):
    level: int = 0


register_extension(
    ParamExtension(
        name="plugsec",
        model=PlugSection,
        relevant=lambda system: system == "plug-cartesian",
        summary="A plugin's section.",
    )
)
""",
    "plugsections_core": """
from pydantic import BaseModel

from dnsjax.extensions import ParamExtension, register_extension


class Phys(BaseModel):
    level: int = 0


register_extension(
    ParamExtension(name="phys", model=Phys, relevant=lambda system: True)
)
""",
    "plugbroken": """
raise ImportError("plugbroken is broken on purpose")
""",
    "plugmasked": """
raise AttributeError("raised inside the flow module import")
""",
    "plugelsewhere": """
from dnsjax.flow_spec import FlowSpec

SPECS = (
    FlowSpec(
        system="plug-elsewhere",
        geometry="plug-geometry",
        summary="A plugin flow on a geometry the core lacks.",
        fields=(),
    ),
)
""",
}

#: The child body of every refusal: import the three loaders, report
#: the first exception.
IMPORT_ALL = """
try:
    import dnsjax.parameters
    import dnsjax.extensions
except Exception as exc:
    print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
else:
    print(json.dumps({"error": None}))
"""

#: The child body of the hook cases: look one hook up, report the
#: exception (its own type and message) or its absence.
HOOK = """
from dnsjax.flows.registry import flow_hook
try:
    flow_hook(SYSTEM, "init_random_state")
except Exception as exc:
    print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
else:
    print(json.dumps({"error": None}))
"""

#: The child body of ``section-absent``: resolve a resume from a
#: metadata-only snapshot that records a plugin's section, with no
#: plugin installed; report the refusal (a ``SystemExit`` message).
RESUME = """
import io, tarfile, tempfile
from pathlib import Path
from dnsjax.bootstrap import resolve_parameters
from dnsjax.param_surface import recorded_params_dump
from dnsjax.parameters import params
from dnsjax.snapshot_meta import META_MEMBER
stored = recorded_params_dump(params) | {"plugsec": {"level": 1}}
payload = json.dumps(
    {"format_version": 6, "system": "plane-couette", "params": stored}
).encode()
path = Path(tempfile.mkdtemp()) / "plugged.tar"
with tarfile.open(path, "w") as tf:
    info = tarfile.TarInfo(META_MEMBER)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))
try:
    resolve_parameters(["--init.snapshot", str(path)], toml_path=False)
except SystemExit as exc:
    print(json.dumps({"error": "SystemExit", "message": str(exc.code)}))
else:
    print(json.dumps({"error": None}))
"""

DISCOVERY = """
import contextlib, io
from dnsjax.flows import registry
from dnsjax import parameters
from dnsjax.bootstrap import resolve_parameters
from dnsjax.extensions import EXTENSIONS
text = io.StringIO()
with contextlib.redirect_stdout(text):
    try:
        resolve_parameters(["--help", "plug-cartesian"])
    except SystemExit:
        pass
print(json.dumps({
    "order": [[s.system, s.geometry] for s in registry.SPECS.values()],
    "knob": parameters.params.phys.plug_knob,
    "sections": sorted(EXTENSIONS),
    "help": text.getvalue(),
}))
"""

#: ``(entry points, child body)`` per case.
FLOWS, FIELDS, SECTIONS = (
    "dnsjax.flows",
    "dnsjax.parameters",
    "dnsjax.extensions",
)
CASES: dict[str, tuple[dict, str]] = {
    "discovery": (
        {
            FLOWS: [["plug", "plugflows:SPECS"]],
            FIELDS: [["plug", "plugfields:FIELDS"]],
            SECTIONS: [["plug", "plugsections"]],
        },
        DISCOVERY,
    ),
    "flow-load": ({FLOWS: [["broken", "plugbroken:SPECS"]]}, IMPORT_ALL),
    "flow-type": ({FLOWS: [["plug", "plugflows:NOT_SPECS"]]}, IMPORT_ALL),
    "flow-object": ({FLOWS: [["plug", "plugflows:NUMBER"]]}, IMPORT_ALL),
    "flow-twice": ({FLOWS: [["plug", "plugflows:TWICE"]]}, IMPORT_ALL),
    "flow-twice-plugins": (
        {FLOWS: [["a", "plugflows:SPECS"], ["b", "plugflows:SPECS"]]},
        IMPORT_ALL,
    ),
    "flow-geometry": ({FLOWS: [["plug", "plugelsewhere:SPECS"]]}, IMPORT_ALL),
    "flow-field": ({FLOWS: [["plug", "plugflows:MISSING"]]}, IMPORT_ALL),
    "field-core": ({FIELDS: [["plug", "plugfields:CORE"]]}, IMPORT_ALL),
    "field-twice": (
        {FIELDS: [["a", "plugfields:FIELDS"], ["b", "plugfields:FIELDS"]]},
        IMPORT_ALL,
    ),
    "field-section": ({FIELDS: [["plug", "plugfields:SECTION"]]}, IMPORT_ALL),
    "field-object": ({FIELDS: [["plug", "plugfields:TUPLE"]]}, IMPORT_ALL),
    "field-shape": ({FIELDS: [["plug", "plugfields:SHAPE"]]}, IMPORT_ALL),
    "section-core": ({SECTIONS: [["plug", "plugsections_core"]]}, IMPORT_ALL),
    "section-absent": ({}, RESUME),
    "hook-none": (
        {FLOWS: [["plug", "plugflows:SPECS"]]},
        HOOK.replace("SYSTEM", "'plug-cartesian'"),
    ),
    "hook-masked": (
        {FLOWS: [["plug", "plugflows:MASKED"]]},
        HOOK.replace("SYSTEM", "'plug-masked'"),
    ),
}

#: The error each refusal raises: its type and a fragment of its text.
REFUSALS: dict[str, tuple[str, str]] = {
    "flow-load": ("RuntimeError", "flow plugin 'broken'"),
    "flow-type": ("TypeError", "not FlowSpec objects"),
    "flow-object": (
        "TypeError",
        "flow plugin 'plug' (plugflows:NUMBER) names [42]",
    ),
    "flow-twice": ("ValueError", "'pipe' is registered twice"),
    "flow-twice-plugins": (
        "ValueError",
        "'plug-cartesian' is registered twice, by flow plugin 'a' and "
        "flow plugin 'b'",
    ),
    "flow-geometry": ("RuntimeError", "names geometry 'plug-geometry'"),
    "flow-field": (
        "ValueError",
        "'plug-missing' spec names phys.no_such_knob, which no parameter",
    ),
    "field-core": (
        "ValueError",
        "parameter plugin 'plug' redefines core field(s) of [phys]: re",
    ),
    "field-twice": (
        "ValueError",
        "phys.plug_knob is added by two parameter plugins ('a' and 'b')",
    ),
    "field-section": ("ValueError", "unknown section(s) ['plugsec']"),
    "field-object": (
        "TypeError",
        "parameter plugin 'plug' (plugfields:TUPLE) names ('phys',)",
    ),
    "field-shape": (
        "TypeError",
        "parameter plugin 'plug' (plugfields:SHAPE) defines phys.knob as 1.5",
    ),
    "section-core": ("RuntimeError", "'phys' clashes with a core"),
    "section-absent": (
        "SystemExit",
        "records the parameter section(s) [plugsec], which nothing",
    ),
    "hook-none": ("ValueError", "'plug-cartesian' names no flow module"),
    "hook-masked": ("AttributeError", "raised inside the flow module import"),
}


def run_case(name: str, timeout: float) -> dict:
    """Run *name*'s child in a scratch directory; its JSON report."""
    eps, body = CASES[name]
    scratch = Path(tempfile.mkdtemp(prefix="dnsjax_plugins_"))
    try:
        for module, source in MODULES.items():
            (scratch / f"{module}.py").write_text(source)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(NO_COLOR="1", COLUMNS="100")
        proc = subprocess.run(
            [sys.executable, "-c", CHILD, json.dumps(eps), body],
            cwd=scratch,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    lines = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
        raise RuntimeError(f"child exited {proc.returncode}: {tail}")
    return json.loads(lines[-1])


def check_discovery(got: dict) -> str | None:
    """Where the plugin's flows, field and section landed."""
    order = [system for system, _ in got["order"]]
    for plug, geometry in (
        ("plug-cartesian", "cartesian"),
        ("plug-annular", "annular"),
    ):
        own = [s for s, g in got["order"] if g == geometry]
        if own[-1] != plug or len(own) < 2:
            return f"{plug} is not after the core {geometry} flows: {order}"
    choices = re.search(r"--phys\.system \{([^}]*)\}", got["help"])
    if choices is None or not {"plug-cartesian", "plug-annular"} <= set(
        choices.group(1).split(",")
    ):
        return "the plugin's flows are not phys.system choices"
    if got["knob"] != 1.5:
        return f"phys.plug_knob is {got['knob']!r}, not its default 1.5"
    if "plugsec" not in got["sections"]:
        return f"the plugin's section is not registered: {got['sections']}"
    for what in ("--phys.plug_knob", "--plugsec.level"):
        if what not in got["help"]:
            return f"--help plug-cartesian does not show {what}"
    return None


def check_spec_labels() -> str | None:
    """``FlowSpec`` refuses unlabelled or mislabelled stored components."""
    from dnsjax.flow_spec import FlowSpec

    def build(**kwargs):
        return FlowSpec(
            system="plug-labels",
            geometry=kwargs.pop("geometry", "cylindrical"),
            summary="A plugin flow storing more than the velocity.",
            fields=(),
            **kwargs,
        )

    four = ("u_z", "u_r", "u_theta", "c")
    parity = (True, False, False, True)
    cases = {
        "no labels": (dict(n_components=4), "labels none"),
        "no parity": (
            dict(n_components=4, components=four),
            "gives no axis parity",
        ),
        "short labels": (
            dict(n_components=4, components=four[:2], component_parity=parity),
            "components has 2 entries for 4 components",
        ),
        "short parity": (
            dict(n_components=4, components=four, component_parity=parity[:3]),
            "component_parity has 3 entries for 4 components",
        ),
    }
    for label, (kwargs, fragment) in cases.items():
        try:
            build(**kwargs)
        except ValueError as exc:
            if fragment not in str(exc):
                return f"{label}: {exc}"
        else:
            return f"{label}: accepted"
    try:
        build(n_components=4, components=four, component_parity=parity)
        build(geometry="cartesian", n_components=4, components=four)
    except ValueError as exc:
        return f"labelled spec refused: {exc}"
    return None


def main() -> int:
    failures: list[tuple[str, str]] = []
    passed = 0
    reason = check_spec_labels()
    print(f"  {'PASS' if reason is None else 'FAIL'}  spec-labels")
    if reason is None:
        passed += 1
    else:
        print(f"        {reason}")
        failures.append(("spec-labels", reason))
    for name in CASES:
        try:
            got = run_case(name, timeout=300.0)
            if name == "discovery":
                reason = check_discovery(got)
            else:
                kind, fragment = REFUSALS[name]
                if got["error"] != kind or fragment not in got["message"]:
                    reason = (
                        f"expected {kind} with {fragment!r}, got "
                        f"{got['error']}: {got.get('message', '')[:120]}"
                    )
                else:
                    reason = None
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
        print(f"  {'PASS' if reason is None else 'FAIL'}  {name}")
        if reason is None:
            passed += 1
        else:
            print(f"        {reason}")
            failures.append((name, reason))
    return report(passed, failures)


if __name__ == "__main__":
    sys.exit(main())
