r"""JAX-free vocabulary for per-flow parameter specs.

A flow's :class:`FlowSpec` declares its user-facing parameter surface:
which shared parameter fields apply to it (on top of the always-present
global fields listed in :mod:`dnsjax.flows.registry`), the public
(user-facing) names of aliased fields, per-flow default overrides,
narrowed choice sets, deferred (not-yet-implemented) fields, and the
flow's derivation / validation hooks -- the per-system parameter math
that used to live inline in ``parameters.update_parameters`` /
``validate_parameters``.

Specs are plain data plus pure-Python hooks.  They import nothing
heavier than the standard library -- no pydantic, no JAX, and never
:mod:`dnsjax.parameters` (the hooks receive the live ``params`` /
``derived_params`` objects as arguments) -- so rendering ``--help`` or
validating a TOML never configures JAX and cannot create an import
cycle.  The concrete specs live in ``flows/*/specs/``; the aggregation
(system list, lookup, stored-metadata helpers) in
:mod:`dnsjax.flows.registry`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final


class _Unset:
    """Sentinel distinguishing "no per-flow default" from ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "UNSET"


#: Sentinel for :attr:`FieldSpec.default`: the flow does not override
#: the field's model default.
UNSET: Final = _Unset()


@dataclass(frozen=True)
class FieldSpec:
    """One relevant parameter field of a flow's surface.

    ``section``/``name`` address the field on the internal
    ``parameters.Parameters`` models; ``public`` is the user-facing
    name shown on every surface (``--help``, ``parameters.toml``, the
    startup printout, snapshot metadata) -- ``None`` means the internal
    name is already the public one.  ``description`` overrides the
    internal model's ``Field(description=...)`` for per-flow phrasing;
    ``default`` overrides the model default (materialized into the
    live ``params`` when no configuration layer sets the field);
    ``choices`` narrows a ``Literal`` field to the values valid for
    this flow.
    """

    section: str
    name: str
    public: str | None = None
    description: str | None = None
    default: Any = UNSET
    choices: tuple[str, ...] | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.section, self.name)

    @property
    def public_name(self) -> str:
        return self.public if self.public is not None else self.name


@dataclass(frozen=True)
class DeferredSpec:
    """A field this flow will support later but rejects for now.

    Setting the field (any configuration layer, or a direct
    assignment caught by ``validate_parameters``) raises with
    *message* instead of the generic not-a-parameter error, so the
    user learns the feature is planned rather than nonsensical.
    """

    section: str
    name: str
    message: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.section, self.name)


@dataclass(frozen=True)
class FlowSpec:
    r"""Complete per-flow parameter declaration.

    ``family`` keys the shared machinery (``"cartesian"``,
    ``"cylindrical"``, ``"annular"``, ``"cylindrical-viscoelastic"``,
    ``"annular-viscoelastic"``, ``"triply-periodic"``); the two
    viscoelastic families are grouped with their geometry *and* with
    each other by :mod:`dnsjax.flows.registry`, along the geometry and
    rheology axes respectively.  ``geometry_label`` groups the
    ``--help`` flow list (each viscoelastic flow displays under its
    geometry).

    Hooks (all optional):

    ``derive(params, derived, user_set)``
        The flow's parameter derivation, run by ``update_parameters``
        after each configuration layer: required-field checks, derived
        control parameters (e.g. the circular-Couette ``re``/``ccf``),
        geometry-forced fields (the azimuthal wedge ``lz = 2*pi/m0``),
        and ``derived_params`` entries.  ``user_set`` is the set of
        ``(section, name)`` keys explicitly provided by any layer.
    ``validate(params, derived)``
        Flow-specific cross-field checks (and startup summaries), run
        once by ``validate_parameters`` after the final layer.
    ``rehydrate(sections)``
        Offline mirror of ``derive`` for JAX-free metadata consumers:
        fills the hidden-derived *internal* keys (e.g. ``geo.lz``,
        the derived ``phys.re``/``re2``) into a nested
        ``{section: {name: value}}`` dict of internal-named parameters
        read back from snapshot metadata.

    ``grid_type_default`` resolves the wall-normal grid when no layer
    sets ``geo.grid_type``: a literal value, or a callable receiving
    ``step.scheme`` (the cylindrical family is scheme-dependent);
    ``None`` for flows without a wall-normal grid.

    ``flow_module`` is the dotted path of the runtime flow module
    exporting the stepping surface (``init_state``, ``get_stats``,
    ``predict_and_fully_correct(_measured)``, ``step_cnab2(_measured)``,
    ``set_dt`` / ``reset_ab2_kappa`` (the adaptive-dt hooks),
    ``get_perturbation_energy``) -- a string, so the spec stays
    JAX-free; consumers import it lazily (the ``__main__`` flow
    dispatch, the transient-growth driver).  ``n_components`` is the
    leading state-axis size (3 velocity components unless the flow
    carries more, e.g. the 9-component viscoelastic state); read by
    the snapshot writer and the analysis component schemas.
    """

    system: str
    family: str
    geometry_label: str
    summary: str
    fields: tuple[FieldSpec, ...]
    deferred: tuple[DeferredSpec, ...] = ()
    grid_type_default: str | Callable[[str], str] | None = None
    derive: Callable[..., None] | None = None
    validate: Callable[..., None] | None = None
    rehydrate: Callable[[dict], None] | None = None
    flow_module: str | None = None
    n_components: int = 3

    #: ``(section, name) -> FieldSpec`` over :attr:`fields` (cached).
    field_map: dict[tuple[str, str], FieldSpec] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    #: ``(section, name) -> DeferredSpec`` over :attr:`deferred`.
    deferred_map: dict[tuple[str, str], DeferredSpec] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        fm = {fs.key: fs for fs in self.fields}
        if len(fm) != len(self.fields):
            raise ValueError(
                f"duplicate field in {self.system!r} spec: "
                f"{[fs.key for fs in self.fields]}"
            )
        overlap = fm.keys() & {d.key for d in self.deferred}
        if overlap:
            raise ValueError(
                f"{self.system!r} spec lists {sorted(overlap)} as both "
                "relevant and deferred"
            )
        object.__setattr__(self, "field_map", fm)
        object.__setattr__(
            self, "deferred_map", {d.key: d for d in self.deferred}
        )

    def default_for(self, section: str, name: str) -> Any:
        """The flow's default override for a field (``UNSET`` if none)."""
        fs = self.field_map.get((section, name))
        return UNSET if fs is None else fs.default

    def choices_for(self, section: str, name: str) -> tuple[str, ...] | None:
        """The flow's narrowed choice set for a field (``None`` if not
        narrowed)."""
        fs = self.field_map.get((section, name))
        return None if fs is None else fs.choices

    def alias(self, section: str, name: str) -> str:
        """Internal field name -> public name (identity fallback)."""
        fs = self.field_map.get((section, name))
        return name if fs is None else fs.public_name

    def dealias(self, section: str, public: str) -> str | None:
        """Public field name -> internal name within *section*.

        Returns ``None`` when no relevant field of this flow carries
        that public name (the caller decides whether that is an error
        or a key to drop).
        """
        for fs in self.fields:
            if fs.section == section and fs.public_name == public:
                return fs.name
        return None
