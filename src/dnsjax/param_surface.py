r"""Per-flow user-facing parameter surfaces (JAX-free).

For a given flow spec this module builds the *surface*: the ordered
list of parameters that apply to that flow -- the global fields
(:data:`dnsjax.flows.registry.GLOBAL_FIELDS`) plus the spec's own --
under their public names, with per-flow defaults, descriptions, and
narrowed choices.  Every user-facing representation derives from it:

- the CLI parser and ``--help`` (a dynamically created
  pydantic-settings model, :func:`build_surface_model` with
  ``settings=True``),
- the ``parameters.toml`` validator (the plain-``BaseModel`` twin,
  ``settings=False`` -- ``BaseSettings.__init__`` would trigger
  CLI/env sources),
- the startup printout (:func:`print_resolved_parameters`),
- the annotated sample TOML (:func:`render_sample_toml`),
- the snapshot-metadata / sidecar params dump (:func:`externalize`).

All models are built with ``extra="forbid"``, so a parameter that does
not apply to the selected flow is rejected uniformly on the CLI and in
TOML; :func:`internalize` maps a validated public-named dump back onto
the internal ``parameters.Parameters`` names (and raises the
deferred-feature message when a deferred field was set).

Deferred fields are *present* in the parse models but hidden from
``--help`` (``CLI_SUPPRESS``), so setting one produces its "not
implemented yet" message instead of a generic unrecognized-argument
error.
"""

import argparse
import json
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic_settings import (
    CLI_SUPPRESS,
    BaseSettings,
    CliSettingsSource,
    SettingsConfigDict,
)

from .flow_spec import UNSET, FlowSpec
from .flows.registry import GLOBAL_FIELDS, systems_by_geometry
from .parameters import Parameters

#: Short section headers shown as ``--help`` group descriptions (the
#: internal model docstrings are reference documentation, far too long
#: for help output).
_SECTION_DOCS: dict[str, str] = {
    "dist": "Device distribution and JAX platform.",
    "phys": "Physical parameters.",
    "geo": "Domain geometry and wall-normal grid.",
    "res": "Resolution.",
    "init": "Initial condition and resume policy.",
    "outs": "Output cadences and snapshot policy.",
    "step": "Time integration.",
    "stop": "Termination criteria.",
    "solver": "Numerical-kernel execution (speed/memory, never results).",
}


@dataclass(frozen=True)
class SurfaceEntry:
    """One field of a flow's surface, ready for model building."""

    section: str
    internal: str
    public: str
    annotation: object
    field_info: object  # pydantic FieldInfo (copied, per-flow adjusted)
    deferred_message: str | None = None


def _narrow_choices(annotation, choices: tuple[str, ...]):
    """Narrow a ``Literal`` annotation to *choices* (keeping ``| None``)."""
    lit = Literal[choices]  # Literal[('a', 'b')] == Literal['a', 'b']
    if get_origin(annotation) is Union and type(None) in get_args(annotation):
        return lit | None
    return lit


def surface_entries(spec: FlowSpec | None) -> list[SurfaceEntry]:
    """The ordered surface of *spec* (``None``: the global fields only).

    Order follows the internal models (sections in ``Parameters``
    declaration order, fields in each section's declaration order), so
    every rendering of a surface lists parameters consistently.
    """
    global_keys = set(GLOBAL_FIELDS)
    entries: list[SurfaceEntry] = []
    for section, sec_field in Parameters.model_fields.items():
        model = sec_field.annotation
        for name, finfo in model.model_fields.items():
            key = (section, name)
            fs = spec.field_map.get(key) if spec is not None else None
            deferred = spec.deferred_map.get(key) if spec is not None else None
            if key not in global_keys and fs is None and deferred is None:
                continue
            info = deepcopy(finfo)
            annotation = finfo.annotation
            if fs is not None:
                if fs.default is not UNSET:
                    info.default = fs.default
                if fs.description is not None:
                    info.description = fs.description
                if fs.choices is not None:
                    annotation = _narrow_choices(annotation, fs.choices)
            if deferred is not None:
                info.description = CLI_SUPPRESS
            entries.append(
                SurfaceEntry(
                    section=section,
                    internal=name,
                    public=fs.public_name if fs is not None else name,
                    annotation=annotation,
                    field_info=info,
                    deferred_message=(
                        deferred.message if deferred is not None else None
                    ),
                )
            )
    return entries


class _SurfaceBase(BaseSettings):
    """Config carrier for the CLI surface models (dotted
    ``--section.field`` flags, no JSON blobs, strictness)."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_avoid_json=True,
        cli_hide_none_type=True,
        cli_prog_name="dnsjax",
        nested_model_default_partial_update=True,
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # CLI and TOML are the only user-facing configuration channels:
        # drop the env/dotenv/secrets sources, which would otherwise map
        # environment variables named like sections (``FORCE``, ``RES``,
        # ``STEP``, ...) onto the surface -- crashing the parse or
        # silently injecting parameters.  The explicit
        # ``CliSettingsSource`` passed by ``make_cli_source`` is
        # prepended by pydantic-settings after this hook.
        return (init_settings,)


def build_surface_model(
    spec: FlowSpec | None,
    *,
    settings: bool,
    extensions: tuple = (),
) -> type[BaseModel]:
    """Create the surface model of *spec* (``None``: global-only).

    ``settings=True`` builds the pydantic-settings CLI variant;
    ``settings=False`` the plain-``BaseModel`` twin used to validate
    ``parameters.toml``.  Both are ``extra="forbid"`` throughout.
    *extensions* appends the given
    :class:`~dnsjax.extensions.ParamExtension` sections (each entry
    point passes the sections it supports -- the production solver its
    flow-relevant built-ins, an analysis CLI its own).
    """
    entries = surface_entries(spec)
    sections: dict[str, dict] = {}
    for e in entries:
        sections.setdefault(e.section, {})[e.public] = (
            e.annotation,
            e.field_info,
        )
    top_fields = {}
    for section, fields in sections.items():
        sec_model = create_model(
            f"Surface_{section}",
            __config__=ConfigDict(extra="forbid"),
            __doc__=_SECTION_DOCS.get(section, section),
            **fields,
        )
        top_fields[section] = (sec_model, Field(default_factory=sec_model))
    for ext in extensions:
        # A thin subclass so the (short) extension summary, not the
        # (long, reference) model docstring, becomes the help header.
        sec_model = create_model(
            f"Surface_{ext.name}",
            __base__=ext.model,
            __doc__=ext.summary or ext.name,
        )
        top_fields[ext.name] = (
            sec_model,
            Field(default_factory=sec_model),
        )
    name = f"DnsjaxSurface_{spec.system if spec else 'global'}"
    if settings:
        return create_model(name, __base__=_SurfaceBase, **top_fields)
    return create_model(
        name, __config__=ConfigDict(extra="forbid"), **top_fields
    )


def split_extensions(
    dump: dict, extensions: tuple
) -> tuple[dict, dict[str, dict]]:
    """Split a surface dump into core sections and extension overlays.

    The parsed surface mixes core parameter sections with extension
    sections; the core half feeds :func:`internalize` /
    ``update_parameters``, the overlays
    :func:`dnsjax.extensions.apply_extension_layer`.
    """
    ext_names = {ext.name for ext in extensions}
    core: dict = {}
    overlays: dict[str, dict] = {}
    for key, value in dump.items():
        if key in ext_names:
            if isinstance(value, dict) and value:
                overlays[key] = value
        else:
            core[key] = value
    return core, overlays


class _SurfaceParser(argparse.ArgumentParser):
    """Root parser: flow-aware unrecognized-argument errors."""

    surface_system: str | None = None

    def error(self, message: str) -> None:  # pragma: no cover - exits
        if self.surface_system is not None and message.startswith(
            "unrecognized arguments"
        ):
            message += (
                f" -- not parameters of flow "
                f"{self.surface_system!r}; see `{self.prog} --help "
                f"{self.surface_system}`"
            )
        super().error(message)


def flow_list_epilog(prog: str = "dnsjax") -> str:
    """The ``--help`` epilog: implemented flows grouped by geometry."""
    lines = ["flows (geometry: systems):"]
    for label, specs in systems_by_geometry().items():
        lines.append(
            f"  {label + ':':<17} " + " ".join(s.system for s in specs)
        )
    lines.append(
        f"\nRun `{prog} --help <system>` for the parameters of one flow."
    )
    return "\n".join(lines)


def make_cli_source(
    model: type[BaseModel],
    *,
    system: str | None = None,
    epilog: str | None = None,
    cli_args: list[str] | None = None,
    prog: str = "dnsjax",
) -> CliSettingsSource:
    """A ``CliSettingsSource`` over a custom root parser.

    The custom parser contributes the flow-list / flow-summary
    *epilog* and rewrites argparse's ``unrecognized arguments`` error
    into a flow-aware message naming *system*.  *prog* is the program
    name shown in usage/help and the error hint (entry points other
    than the ``dnsjax`` console script pass their own).
    """
    parser = _SurfaceParser(
        prog=prog,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.surface_system = system
    return CliSettingsSource(
        model,
        root_parser=parser,
        cli_parse_args=cli_args if cli_args is not None else True,
    )


def internalize(surface_dump: dict, spec: FlowSpec) -> dict:
    """Public-named ``exclude_unset`` dump -> internal-named dict.

    Raises the deferred-feature message when a deferred field was set
    to anything but its (inert) model default -- an explicit default,
    e.g. a scripted ``--init.localized_rolls False`` on a periodic
    flow, is a no-op and passes.  Global / unaliased names pass
    through unchanged.  The result feeds ``Parameters.model_validate``
    and then ``update_parameters`` (the per-layer merge), so
    ``_user_set_fields`` records internal names.
    """
    out: dict[str, dict] = {}
    for section, sec in surface_dump.items():
        if not isinstance(sec, dict):
            continue
        out_sec: dict = {}
        for public, value in sec.items():
            deferred = spec.deferred_map.get((section, public))
            if deferred is not None and value is not None:
                # Deferred fields are unaliased (public == internal).
                model = Parameters.model_fields[section].annotation
                default = model.model_fields[public].get_default(
                    call_default_factory=True
                )
                if value != default:
                    raise ValueError(deferred.message)
            internal = spec.dealias(section, public)
            out_sec[internal if internal is not None else public] = value
        if out_sec:
            out[section] = out_sec
    return out


def externalize(parameters: Parameters, spec: FlowSpec) -> dict:
    """The flow-relevant, public-named, resolved-value params dump.

    The single representation recorded in snapshot metadata / sidecar
    JSON and shown by the startup printout: only the fields on
    *spec*'s surface (deferred fields excluded), under their public
    names, with the post-``update_parameters`` (materialized) values,
    JSON-safe (``model_dump(mode="json")``).
    """
    dump = parameters.model_dump(mode="json")
    out: dict[str, dict] = {}
    for e in surface_entries(spec):
        if e.deferred_message is not None:
            continue
        out.setdefault(e.section, {})[e.public] = dump[e.section][e.internal]
    return out


def recorded_params_dump(parameters: Parameters) -> dict:
    """The ``params`` payload of snapshot metadata / sidecar JSON.

    :func:`externalize` for the resolved system's spec, plus the
    recordable relevant extension sections
    (:func:`dnsjax.extensions.extension_metadata`) as top-level keys
    (e.g. ``force``, ``probes``) -- so resume can restore them and the
    trajectory diff can compare them.  Snapshot readers map it back
    with :func:`dnsjax.flows.registry.internalize_stored` /
    ``stored_value``.
    """
    from .extensions import extension_metadata
    from .flows.registry import spec_for

    system = parameters.phys.system
    dump = externalize(parameters, spec_for(system))
    dump.update(extension_metadata(system))
    return dump


# ── Rendering (startup printout, sample TOML) ────────────────────


def _toml_value(v) -> str:
    """A TOML-compatible scalar rendering of a JSON-safe value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    return json.dumps(str(v))


def _wrap_comment(text: str) -> list[str]:
    return [
        f"# {line}"
        for line in textwrap.wrap(text, width=77, break_long_words=False)
    ]


def render_sample_toml(spec: FlowSpec, extensions: tuple = ()) -> str:
    """An annotated ``parameters.toml`` template for *spec*.

    Every parameter of the flow's surface appears with its 1--2-line
    description and its (per-flow resolved) default value **commented
    out**, so loading the rendered file pins nothing; only
    ``phys.system`` is active.  A scheme-dependent ``geo.grid_type``
    default shows the default-scheme value.  *extensions* appends the
    given :class:`~dnsjax.extensions.ParamExtension` sections (the
    caller passes the flow-relevant ones), rendered the same way.
    """
    lines = [
        f'# dnsjax parameters -- system "{spec.system}" ({spec.summary}).',
        "# Defaults are shown commented out; uncomment a line to set it.",
        "# CLI flags (e.g. --phys.re 2300) override these values.",
    ]
    current_section = None
    for e in surface_entries(spec):
        if e.deferred_message is not None:
            continue
        if e.section != current_section:
            current_section = e.section
            lines += ["", f"[{e.section}]"]
        desc = e.field_info.description
        if desc:
            lines += _wrap_comment(desc)
        default = e.field_info.default
        if e.section == "geo" and e.internal == "grid_type":
            gd = spec.grid_type_default
            default = gd("iterative-cn") if callable(gd) else gd
        if e.section == "phys" and e.internal == "system":
            lines.append(f'system = "{spec.system}"')
        elif default is None:
            lines.append(f"#{e.public} =")
        else:
            lines.append(f"#{e.public} = {_toml_value(default)}")
    for ext in extensions:
        lines += ["", f"[{ext.name}]"]
        if ext.summary:
            lines += _wrap_comment(ext.summary)
        for name, field_info in ext.model.model_fields.items():
            if field_info.description:
                lines += _wrap_comment(field_info.description)
            if field_info.default is None:
                lines.append(f"#{name} =")
            else:
                lines.append(f"#{name} = {_toml_value(field_info.default)}")
    return "\n".join(lines) + "\n"


def print_resolved_parameters(
    parameters: Parameters, spec: FlowSpec, extensions: tuple = ()
) -> None:
    """Print the resolved (relevant-only, public-named) parameters.

    TOML-shaped for readability and copy-paste; unset optional fields
    print as commented ``#name =`` lines.  *extensions* appends the
    given :class:`~dnsjax.extensions.ParamExtension` sections with
    their live resolved values.
    """
    values = externalize(parameters, spec)
    for ext in extensions:
        values[ext.name] = ext.values.model_dump(mode="json")
    lines = [f"Final working parameters ({spec.system}):"]
    for section, sec in values.items():
        lines += ["", f"[{section}]"]
        for public, value in sec.items():
            if value is None:
                lines.append(f"#{public} =")
            else:
                lines.append(f"{public} = {_toml_value(value)}")
    print("\n".join(lines))
