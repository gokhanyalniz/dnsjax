r"""Registry of all flow parameter specs (JAX-free).

Aggregates the per-flow :class:`~dnsjax.flow_spec.FlowSpec` modules of
``flows/*/specs/`` into the canonical system list (the source of the
``phys.system`` ``Literal`` and the ``--help`` flow list), the
system -> spec lookup, the family groupings historically defined in
:mod:`dnsjax.parameters` (re-exported there), and the helpers that map
*stored* (public-named, relevance-filtered) snapshot-metadata
parameters back onto the internal names.

Import direction: ``parameters.py -> flows.registry -> flows.*.specs
-> flow_spec.py``; nothing here may import :mod:`dnsjax.parameters`.
"""

from ..flow_spec import FlowSpec
from .triply_periodic.specs import SPECS as _PERIODIC_SPECS
from .wall_bounded.specs import SPECS as _WALL_SPECS

#: All flow specs, in the canonical order (drives the ``phys.system``
#: ``Literal``, ``--help``, and every listing).
SPECS: dict[str, FlowSpec] = {
    spec.system: spec for spec in (*_PERIODIC_SPECS, *_WALL_SPECS)
}

#: The parameter fields every flow accepts, ``(section, name)`` on the
#: internal ``parameters.Parameters`` models (public name = internal
#: name for all of these).  A flow's full surface is these plus its
#: ``spec.fields``.
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
    ("step", "scheme"),
    ("step", "dt"),
    ("step", "implicitness"),
    ("step", "corrector_tolerance"),
    ("step", "max_corrector_iterations"),
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


def systems_by_geometry() -> dict[str, list[FlowSpec]]:
    """Specs grouped by ``geometry_label`` (canonical order kept)."""
    out: dict[str, list[FlowSpec]] = {}
    for spec in SPECS.values():
        out.setdefault(spec.geometry_label, []).append(spec)
    return out


def _by_family(family: str) -> list[str]:
    return [s.system for s in SPECS.values() if s.family == family]


# Family groupings, historically defined in ``dnsjax.parameters`` and
# still re-exported there for the many existing importers.
periodic_systems: list[str] = _by_family("triply-periodic")
monochromatic_systems: list[str] = [
    s for s in periodic_systems if s != "decaying-box"
]
cartesian_systems: list[str] = _by_family("cartesian")
cylindrical_systems: list[str] = _by_family("cylindrical")
annular_systems: list[str] = _by_family("annular")
viscoelastic_systems: list[str] = _by_family("annular-viscoelastic")
walled_systems: list[str] = [
    *cartesian_systems,
    *cylindrical_systems,
    *annular_systems,
    *viscoelastic_systems,
]


# ── Stored-metadata helpers (public names -> internal names) ─────


def stored_value(meta_params: dict, system: str, section: str, name: str):
    """Read internal field ``section.name`` from a stored params dict.

    Stored (v4) snapshot metadata records the *public* names; this
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

    Unknown keys -- fields that are neither global nor on *system*'s
    surface under their public name -- are dropped with a note (schema
    drift across versions).  With ``rehydrate=True`` the spec's
    :attr:`~dnsjax.flow_spec.FlowSpec.rehydrate` hook then fills the
    hidden-derived internal keys (e.g. the wedge ``geo.lz``, the
    derived ``phys.re``/``re2``), so offline consumers see the same
    internal values a live ``update_parameters`` would produce.
    Extension sections (e.g. ``force``) are passed through unchanged;
    their relevance/ownership is the caller's concern.
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
            if internal is None:
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
