r"""Registry of all flow parameter specs (JAX-free).

Aggregates the per-flow :class:`~dnsjax.flow_spec.FlowSpec` modules of
``flows/*/specs/``, and those of installed plugins, into the canonical
system list (the source of the ``phys.system`` ``Literal`` and the
``--help`` flow list), the system -> spec lookup, the geometry
groupings, and the helpers that map *stored* (public-named,
relevance-filtered) snapshot-metadata parameters back onto the
internal names.

A plugin package registers flows under the ``dnsjax.flows`` entry-point
group; each entry point names a :class:`~dnsjax.flow_spec.FlowSpec` or
an iterable of them, in a JAX-free module that never imports
:mod:`dnsjax.parameters` (the spec rules).  The canonical order lists
the flows by geometry, in :data:`dnsjax.flow_spec.GEOMETRIES` order,
and within a geometry the core flows before the plugins' (by
entry-point name, each in its own order).

Import direction: ``parameters.py -> flows.registry -> flows.*.specs
-> flow_spec.py``; nothing here may import :mod:`dnsjax.parameters`.
"""

from importlib import import_module
from importlib.metadata import entry_points

from ..flow_spec import GEOMETRIES, FlowSpec
from .triply_periodic.specs import SPECS as _PERIODIC_SPECS
from .wall_bounded.specs import SPECS as _WALL_SPECS

#: The entry-point group plugin packages register their flows under.
FLOWS_ENTRY_POINT_GROUP = "dnsjax.flows"


def _plugin_specs() -> list[tuple[FlowSpec, str]]:
    """The specs registered under :data:`FLOWS_ENTRY_POINT_GROUP`.

    Each with the entry point that registered it, for the refusals.
    Raises if an entry point fails to load or names anything but
    specs: a flow that silently went missing would turn a run of it
    into an "unknown system" error far from the cause.
    """
    out: list[tuple[FlowSpec, str]] = []
    found = entry_points(group=FLOWS_ENTRY_POINT_GROUP)
    for ep in sorted(found, key=lambda e: e.name):
        try:
            obj = ep.load()
        except Exception as exc:
            raise RuntimeError(
                f"flow plugin {ep.name!r} ({ep.value}) failed to load: {exc}"
            ) from exc
        if isinstance(obj, FlowSpec):
            specs: tuple = (obj,)
        else:
            try:
                specs = tuple(obj)
            except TypeError:
                specs = (obj,)
        bad = [s for s in specs if not isinstance(s, FlowSpec)]
        if bad:
            raise TypeError(
                f"flow plugin {ep.name!r} ({ep.value}) names {bad!r}, "
                "not FlowSpec objects"
            )
        out.extend((spec, f"flow plugin {ep.name!r}") for spec in specs)
    return out


def _canonical(specs: list[tuple[FlowSpec, str]]) -> dict[str, FlowSpec]:
    """*specs* (each with its source) by system, grouped by geometry.

    The listing order within a geometry is the input order; a system
    registered twice is refused naming both sources.
    """
    out: dict[str, FlowSpec] = {}
    sources: dict[str, str] = {}
    # A stable sort keeps the core flows before the plugins' within
    # each geometry.
    for spec, source in sorted(
        specs, key=lambda pair: GEOMETRIES.index(pair[0].geometry)
    ):
        if spec.system in out:
            raise ValueError(
                f"flow system {spec.system!r} is registered twice, by "
                f"{sources[spec.system]} and {source}"
            )
        out[spec.system] = spec
        sources[spec.system] = source
    return out


#: All flow specs, in the canonical order (drives the ``phys.system``
#: ``Literal``, ``--help``, and every listing).
SPECS: dict[str, FlowSpec] = _canonical(
    [
        *((spec, "the core") for spec in (*_PERIODIC_SPECS, *_WALL_SPECS)),
        *_plugin_specs(),
    ]
)

#: The parameter fields every flow accepts, ``(section, name)`` on the
#: internal ``parameters.Parameters`` models (public name = internal
#: name for all of these).  A flow's full surface is these plus its
#: ``spec.fields``.  A field every flow would list unchanged belongs
#: here, not in the specs: the bare ``--help`` shows these alone.
GLOBAL_FIELDS: tuple[tuple[str, str], ...] = (
    ("dist", "np0"),
    ("dist", "np1"),
    ("dist", "platform"),
    ("phys", "system"),
    ("phys", "oversampling_factor"),
    ("res", "double_precision"),
    ("init", "start_from_laminar"),
    ("init", "snapshot"),
    ("init", "t0"),
    ("init", "it0"),
    ("init", "isnap0"),
    ("init", "force_resume"),
    ("init", "random_field"),
    ("init", "random_amplitude"),
    ("init", "random_smoothness"),
    ("init", "random_seed"),
    # Every family: the triply-periodic spot needs no wall-normal grid,
    # its `y` factor being the same fixed-physical localization the
    # homogeneous directions take (``ic/localized_rolls.py``).
    ("init", "localized_rolls"),
    ("init", "localized_rolls_amplitude"),
    ("init", "localized_rolls_width"),
    ("init", "localized_rolls_wavelength"),
    ("outs", "it_stats"),
    ("outs", "it_steps"),
    ("outs", "it_snapshot"),
    ("outs", "it_corrector"),
    ("outs", "it_error_check"),
    ("outs", "nbuffer"),
    ("outs", "stats_precision"),
    ("outs", "snapshot_pad_width"),
    ("outs", "snapshot_embed_stats"),
    ("outs", "snapshot_save_initial"),
    ("outs", "snapshot_save_final"),
    ("outs", "snapshot_write_mode"),
    ("lowres", "it_lowres"),
    ("step", "scheme"),
    ("step", "dt"),
    ("step", "implicitness"),
    ("step", "corrector_tolerance"),
    ("step", "max_corrector_iterations"),
    ("step", "corrector_iterations"),
    ("step", "adaptive"),
    ("step", "cfl_target"),
    ("step", "dt_min"),
    ("step", "dt_max"),
    ("step", "dt_min_change"),
    ("step", "dt_max_change"),
    ("step", "dt_threshold"),
    ("step", "cfl_cadence"),
    ("stop", "max_sim_time"),
    ("stop", "max_wall_time"),
    ("stop", "check_laminarization"),
    ("stop", "laminarization_threshold"),
    ("solver", "rhs_transform_chunks"),
)


def all_systems() -> tuple[str, ...]:
    """Every registered system name, canonical order."""
    return tuple(SPECS)


def spec_for(system: str) -> FlowSpec:
    """The spec of *system*; raises with the full list on a miss."""
    try:
        return SPECS[system]
    except KeyError:
        raise ValueError(
            f"unknown flow system {system!r}; available: {', '.join(SPECS)}"
        ) from None


def import_flow_module(system: str):
    """The imported flow module of *system* (``FlowSpec.flow_module``).

    Importing it imports JAX and builds the geometry singletons, so
    only once JAX is configured.  A spec naming no module (``None``:
    the flow is declared but not implemented) is refused by name; an
    error raised while the module imports propagates as itself.
    """
    module = spec_for(system).flow_module
    if module is None:
        raise ValueError(
            f"{system!r} names no flow module (FlowSpec.flow_module): "
            "the flow is not yet implemented"
        )
    return import_module(module)


def flow_hook(system: str, name: str):
    """The export *name* of *system*'s flow module, a required hook.

    Imports the module through :func:`import_flow_module` and raises
    naming the hook if it lacks one (the flow-module surface:
    :class:`~dnsjax.flow_spec.FlowSpec`).
    """
    module = import_flow_module(system)
    try:
        return getattr(module, name)
    except AttributeError:
        raise AttributeError(
            f"{module.__name__} ({system!r}) does not export {name}, "
            "part of the flow-module surface (dnsjax.flow_spec.FlowSpec)"
        ) from None


def systems_by_geometry() -> dict[str, list[FlowSpec]]:
    """Specs grouped by ``geometry`` (canonical order kept)."""
    out: dict[str, list[FlowSpec]] = {}
    for spec in SPECS.values():
        out.setdefault(spec.geometry, []).append(spec)
    return out


def _by_geometry(geometry: str) -> list[str]:
    return [s.system for s in SPECS.values() if s.geometry == geometry]


# Geometry groupings (defined here; consumers import them from the
# registry): "which geometry's machinery drives this?" -- the basis
# maps, the resume regrid, the IC builders, the analysis component
# schema.
periodic_systems: list[str] = _by_geometry("triply-periodic")
cartesian_systems: list[str] = _by_geometry("cartesian")
cylindrical_systems: list[str] = _by_geometry("cylindrical")
annular_systems: list[str] = _by_geometry("annular")
walled_systems: list[str] = [
    *cartesian_systems,
    *cylindrical_systems,
    *annular_systems,
]

# A second axis, cutting across the geometry lists:
# ``total_field_systems`` answers "is the state (and so every snapshot)
# the total field rather than a perturbation about a laminar profile?"
# (``FlowSpec.total_field``).  Its complement among the walled systems
# is the transient-growth scope.
total_field_systems: list[str] = [
    s.system for s in SPECS.values() if s.total_field
]


# ── Stored-metadata helpers (public names -> internal names) ─────


def stored_value(meta_params: dict, system: str, section: str, name: str):
    """Read internal field ``section.name`` from a stored params dict.

    Stored snapshot metadata records the *public* names; this
    looks the field up under its public alias (identity for global /
    unaliased fields).  Returns ``None`` when absent.
    """
    spec = spec_for(system)
    sec = meta_params.get(section) or {}
    return sec.get(spec.alias(section, name))


def internalize_stored(
    meta_params: dict, system: str, *, rehydrate: bool = False
) -> dict:
    """Map a stored (public-named) params dict to internal names.

    An unknown key in a core section -- neither global, nor on
    *system*'s surface under its public name, nor **deferred** for it
    -- **raises**.  A snapshot naming a parameter this version does
    not define was written by code that meant something by it, and
    silently dropping it would resume a run whose setup differs from
    the stored one in a way nothing reports.  A *deferred* key is a
    different case and passes through: this version knows the field
    and refuses it for this flow, so a snapshot predating the deferral
    stays loadable and its stored value is judged by
    :func:`dnsjax.parameters.validate_parameters` (inert default:
    silent; any other value: the deferral's message).

    ``solver`` is the one core section exempt (unknown keys there are
    dropped with a note): it is execution-only by design -- it selects
    *how* the numerics run, never the result -- and
    :func:`dnsjax.parameters.read_snapshot_params` discards the whole
    section anyway, but only *after* internalizing it.

    With ``rehydrate=True`` the spec's
    :attr:`~dnsjax.flow_spec.FlowSpec.rehydrate` hook then fills the
    hidden-derived internal keys (e.g. the wedge ``geo.lz``, the
    circular-Couette ``phys.re`` from ``re1``/``re2``), so offline
    consumers see the same
    internal values a live ``update_parameters`` would produce.
    Extension sections (any section outside the core models) are
    passed through unchanged; their relevance/ownership is the
    caller's concern.

    Raises
    ------
    ValueError
        A stored core-section key (outside ``solver``) that *system*
        does not define in this version.
    """
    spec = spec_for(system)
    global_keys = set(GLOBAL_FIELDS)
    out: dict = {}
    for section, sec_dict in meta_params.items():
        if not isinstance(sec_dict, dict):
            out[section] = sec_dict
            continue
        if not any(k[0] == section for k in global_keys) and not any(
            fs.section == section for fs in spec.fields
        ):
            # Not a core section (an extension section): pass through.
            out[section] = dict(sec_dict)
            continue
        internal_sec: dict = {}
        for public, value in sec_dict.items():
            internal = spec.dealias(section, public)
            if internal is None and (section, public) in global_keys:
                internal = public
            if internal is None and (section, public) in spec.deferred_map:
                # Known and deliberately refused for this flow: passed
                # through so a snapshot older than the deferral stays
                # resumable (the docstring; ``externalize`` never
                # records a deferred field, so only those reach here).
                internal = public
            if internal is None:
                if section != "solver":
                    raise ValueError(
                        f"stored {section}.{public} is not a {system!r} "
                        "parameter in this version"
                    )
                print(
                    f"[params] note: ignoring stored {section}.{public} "
                    f"(not a {system!r} parameter in this version)"
                )
                continue
            internal_sec[internal] = value
        out[section] = internal_sec
    if rehydrate and spec.rehydrate is not None:
        spec.rehydrate(out)
    return out
