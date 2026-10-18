r"""JAX-free vocabulary for per-flow parameter specs.

A flow's :class:`FlowSpec` declares its user-facing parameter surface:
which shared parameter fields apply to it (on top of the always-present
global fields listed in :mod:`dnsjax.flows.registry`), the public
(user-facing) names of aliased fields, per-flow default overrides,
narrowed choice sets, deferred (not-yet-implemented) fields, and the
flow's derivation / validation hooks -- the per-system parameter math
that ``parameters.update_parameters`` / ``validate_parameters`` invoke
through the spec.

Specs are plain data plus pure-Python hooks.  They import nothing
heavier than the standard library -- no pydantic, no JAX, and never
:mod:`dnsjax.parameters` (the hooks receive the live ``params`` /
``derived_params`` objects as arguments) -- so rendering ``--help`` or
validating a TOML never configures JAX and cannot create an import
cycle.  The concrete specs live in ``flows/*/specs/``, and in installed
plugin packages, which register theirs under the ``dnsjax.flows``
entry-point group; the aggregation (system list, lookup,
stored-metadata helpers) in :mod:`dnsjax.flows.registry`.
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

#: The geometries a flow runs on (:attr:`FlowSpec.geometry`), in the
#: order the registry lists them.  The solver's machinery, the
#: ``phys.system`` choices and every geometry dispatch know these alone;
#: a flow with a metric of its own overrides one of them.
GEOMETRIES: Final[tuple[str, ...]] = (
    "triply-periodic",
    "cartesian",
    "cylindrical",
    "annular",
)

#: Axis-parity class of the stored cylindrical velocity
#: ``(u_z, u_r, u_theta)``: ``True`` for the `$(-1)^m$` (even) class,
#: ``False`` for `$(-1)^{m+1}$` (odd).  A scalar (the pressure of a
#: reduced snapshot) is even, like ``u_z``.  The default of
#: :attr:`FlowSpec.component_parity`.
VELOCITY_PARITY: Final[tuple[bool, ...]] = (True, False, False)


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
        """The internal ``(section, name)`` address of the field."""
        return (self.section, self.name)

    @property
    def public_name(self) -> str:
        """The user-facing name: *public* when set, else *name*."""
        return self.public if self.public is not None else self.name


@dataclass(frozen=True)
class DeferredSpec:
    """A field this flow knows but refuses: a deferred feature.

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
        """The internal ``(section, name)`` address of the field."""
        return (self.section, self.name)


@dataclass(frozen=True)
class FlowSpec:
    r"""Complete per-flow parameter declaration.

    ``geometry`` names the geometry whose machinery drives the flow,
    one of :data:`GEOMETRIES`: the registry's geometry lists
    (:data:`~dnsjax.flows.registry.cylindrical_systems`, ...) are built
    from it, and it groups the ``--help`` flow list.

    Hooks (all optional):

    ``derive(params, derived, user_set)``
        The flow's parameter derivation, run by ``update_parameters``
        after each configuration layer: required-field checks, derived
        control parameters (e.g. the circular-Couette ``re`` and the
        base-flow coefficients ``ccf_A`` / ``ccf_B``),
        geometry-forced fields (the azimuthal wedge ``lz = 2*pi/m0``),
        and ``derived_params`` entries.  ``user_set`` is the set of
        ``(section, name)`` keys explicitly provided by any layer.
    ``validate(params, derived)``
        Flow-specific cross-field checks (and startup summaries), run
        once by ``validate_parameters`` after the final layer.
    ``rehydrate(sections)``
        Offline mirror of ``derive`` for JAX-free metadata consumers:
        fills the hidden-derived *internal* keys (e.g. ``geo.lz``,
        the circular-Couette ``phys.re`` from ``re1``/``re2``) into a nested
        ``{section: {name: value}}`` dict of internal-named parameters
        read back from snapshot metadata.

    ``viscosity(params) -> float``
        The kinematic viscosity multiplying the velocity Laplacian
        (``derived_params.nu``), from the final parameters; unset, it
        is ``1/re``.  A flow whose stress splits between a solvent and
        another component gives the solvent's share.
    ``pad(padded, params)``
        Raises the natural padded sizes of ``parameters.padded_res``
        (``nx_padded``, ``ny_padded``, ``nz_padded``) before the core's
        mesh and FFT rounding, appending a startup note to
        ``padded.notes`` for each change: for a nonlinear term that the
        oversampling factor alone does not dealias.
    ``control_summary(params) -> str``
        The flow's control parameters in one line, for the
        transient-growth summary; unset, it is ``re=<phys.re>``.

    ``grid_type_default`` resolves the wall-normal grid when no layer
    sets ``geo.grid_type``: a literal value, or a callable receiving
    ``step.scheme`` (the cylindrical family is scheme-dependent);
    ``None`` for flows without a wall-normal grid.

    ``flow_module`` is the dotted path of the runtime flow module -- a
    string, so the spec stays JAX-free; consumers import it lazily (the
    ``__main__`` flow dispatch, the transient-growth driver,
    :mod:`dnsjax.lowres`, :func:`dnsjax.flows.registry.flow_hook`).
    The module exports:

    - the stepping surface: ``flow`` (the flow instance the steppers
      are bound to), ``init_state`` (the ``start_from_laminar`` state),
      ``get_stats``, ``predict_and_fully_correct(_measured)``,
      ``step_cnab2(_measured)`` (a ``_measured`` variant also returns
      the CFL measurements of ``steps.dat``), ``set_dt`` /
      ``reset_ab2_kappa`` (the adaptive-dt hooks) and
      ``get_perturbation_energy`` (the cheap `$E'$` read behind the
      laminarization check);
    - the initial-condition hooks ``init_random_state(amplitude,
      smoothness, wall_smoothness, wall_confinement, seed, mean_flow)``
      and ``init_localized_rolls(amplitude, width, wavelength)``, which
      :mod:`dnsjax.ic.random_field` and :mod:`dnsjax.ic.localized_rolls`
      call (the flow composes the core generators of its geometry with
      what it adds itself, e.g. a total-field flow's laminar state);
    - on the cylindrical and annular geometries, the basis pair
      ``to_solver_basis`` / ``from_solver_basis`` (the geometry's,
      re-exported), which ``__main__`` requires there: a state that
      enters the stepper in the wrong basis steps without error, only
      wrongly.  A Cartesian or triply-periodic state is physical
      throughout and exports no pair; every consumer looks the pair up
      with ``getattr`` and falls back to the identity;
    - on the cylindrical geometry, the core ``cylindrical.fourier``
      singleton, whose parity classes the resume regrid and ``[lowres]``
      read for every cylindrical flow, and the velocity components in
      the leading three slots of the solver state, the carried slots
      after them (``CARRIED_FIELDS``): the pipe's ``build_stepper`` is
      written for that layout, and a flow evolving more components
      brings its own stepping (a flow carrying a conformation tensor does);
    - on a wall-bounded base-flow flow (``total_field`` false),
      ``frozen_profile_flow(profile)``, the flow frozen about a given
      profile, which the transient-growth driver linearizes about.

    Two exports are **optional**.  ``get_driving(state) -> dict[str,
    Array]`` is the wall-shear inference of the mean-mode driving from
    the physical-basis state (the same contract as ``get_stats``),
    exported only by flows that can apply one (``phys.driving`` /
    ``phys.block_mean_spanwise_velocity``): the applied force `$-\Pi$`
    (:mod:`dnsjax.ic.mean_mode`) under the driving columns' keys,
    ``{}`` when no driving knob is on.  It supplies those ``stats.dat``
    column names and the ``t = t0`` row, which has no step behind it;
    every later row carries the value the corrector applied, threaded
    out of the step.  ``__main__`` reads it with ``getattr`` and falls
    back to no column.
    ``CARRIED_FIELDS`` names the solver state's trailing slots past the
    ``n_components`` physical ones (the pipe's spin-quad differences),
    which ``from_solver_basis`` drops and ``outs.snapshot_embed_carry``
    stores as the snapshot's ``carry/`` member; absent, there are none.

    ``n_components`` is the leading state-axis size (3 velocity
    components unless the flow carries more); the snapshot writer and
    loader read it, while the IC builders and the FFT, sharding and
    stepper machinery are component-count-agnostic (the leading state
    axis is replicated).

    The stored components and axes, as the snapshot analysis
    (:func:`dnsjax.analysis.geometry_info`) labels them:

    ``components``
        One label per stored component, in storage order; ``None``
        means the geometry's velocity, ``("u_x", "u_y", "u_z")`` or,
        in the cylindrical and annular geometries,
        ``("u_z", "u_r", "u_theta")``.  A flow storing more than three
        components declares it.
    ``component_parity``
        The axis-parity class of each stored component, in storage
        order (``True``: the `$(-1)^m$` class of `$u_z$`; ``False``:
        `$(-1)^{m+1}$`, of `$u_r$`, `$u_\theta$`), for the pipe's
        parity-aware regrid
        (:func:`dnsjax.snapshot.stored_component_parity`) and radial
        derivatives; ``None`` means :data:`VELOCITY_PARITY`.  A
        cylindrical flow storing more than the velocity declares it.
    ``axis_names``
        The direction named by each stored axis; ``None`` means the
        geometry's, ``("y", "z", "x")`` or ``("r", "theta", "z")``.
    ``analysis_metric``
        The ``"module:attr"`` path of a JAX-free factory
        ``(params) -> metric``, for a cylindrical flow whose axial
        direction carries a scale factor
        `$h(r, \theta) \ne 1$` (:class:`dnsjax.analysis.AxialMetric`);
        the factory returns ``None`` where `$h = 1$`.  A string, like
        ``flow_module``, imported only when a snapshot is analysed.

    ``total_field`` declares what the state -- and so every snapshot --
    holds: ``False`` (the default) for the perturbation `$u'$` about the
    flow's laminar profile, ``True`` for the total field, integrated
    around ``base_flow = 0`` with a mean-mode body force (a
    force-driven flow).  The registry exposes it as
    :data:`dnsjax.flows.registry.total_field_systems`; a total-field
    flow has no ``frozen_profile_flow`` hook and is outside the
    transient-growth and single-mode-perturbation scope.
    """

    system: str
    geometry: str
    summary: str
    fields: tuple[FieldSpec, ...]
    deferred: tuple[DeferredSpec, ...] = ()
    grid_type_default: str | Callable[[str], str] | None = None
    derive: Callable[..., None] | None = None
    validate: Callable[..., None] | None = None
    rehydrate: Callable[[dict], None] | None = None
    viscosity: Callable[[Any], float] | None = None
    pad: Callable[[Any, Any], None] | None = None
    control_summary: Callable[[Any], str] | None = None
    flow_module: str | None = None
    n_components: int = 3
    components: tuple[str, ...] | None = None
    component_parity: tuple[bool, ...] | None = None
    axis_names: tuple[str, str, str] | None = None
    analysis_metric: str | None = None
    total_field: bool = False

    #: ``(section, name) -> FieldSpec`` over :attr:`fields` (cached).
    field_map: dict[tuple[str, str], FieldSpec] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )
    #: ``(section, name) -> DeferredSpec`` over :attr:`deferred`.
    deferred_map: dict[tuple[str, str], DeferredSpec] = field(
        init=False, repr=False, compare=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.geometry not in GEOMETRIES:
            raise ValueError(
                f"{self.system!r} spec names geometry {self.geometry!r}, "
                f"not one of {', '.join(GEOMETRIES)}"
            )
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
        if self.components is None and self.n_components != 3:
            raise ValueError(
                f"{self.system!r} spec stores {self.n_components} "
                "components but labels none (FlowSpec.components)"
            )
        if (
            self.geometry == "cylindrical"
            and self.n_components != 3
            and self.component_parity is None
        ):
            raise ValueError(
                f"{self.system!r} spec stores {self.n_components} "
                "cylindrical components but gives no axis parity "
                "(FlowSpec.component_parity)"
            )
        for name in ("components", "component_parity"):
            value = getattr(self, name)
            if value is not None and len(value) != self.n_components:
                raise ValueError(
                    f"{self.system!r} spec: {name} has {len(value)} "
                    f"entries for {self.n_components} components"
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
        """The flow's narrowed choice set for a field, or ``None``."""
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
