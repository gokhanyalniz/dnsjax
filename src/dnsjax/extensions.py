r"""Extension parameter sections: JAX-free registry and built-ins.

An *extension* owns a parameter section outside
:mod:`dnsjax.parameters`: the section is parsed from the ``dnsjax``
CLI (``--<name>.<field>``) and from ``parameters.toml``
(``[<name>]``), appears in ``--help`` for the flows it applies to,
is validated strictly (an extension section on a flow it does not
apply to is an error, like any other irrelevant parameter), is
recorded in snapshot metadata / sidecar JSON, and can be
trajectory-defining.  Modules and scripts register their sections via
:func:`register_extension`; the resolved values live on the
extension's ``values`` singleton (analogous to the global ``params``).

Built-ins registered here:

- ``probes`` -- the runtime spectral-mode probe stream
  (:mod:`dnsjax.probes` writes ``probes.bin``/``probes.json`` during
  the run); wall-bounded flows.
- ``force`` -- the white-in-time stochastic mode kicks
  (:mod:`dnsjax.forcing` injects them into the stepping loop and logs
  ``forcing.bin``/``forcing.json``); wall-bounded non-viscoelastic
  flows, **trajectory-defining** (kicks alter the dynamics exactly
  like a ``phys`` change).

Their runtime modules import JAX at module scope, so the parameter
models and live singletons live here (JAX-free), importable before
the JAX runtime is configured -- e.g. by ``--help``.

Import direction: this module may import :mod:`dnsjax.flows.registry`
and :mod:`dnsjax.harmonics` but never :mod:`dnsjax.parameters` (the
validate hooks receive the live ``params`` as an argument), so
``parameters -> flows.registry`` and ``bootstrap -> extensions``
stay cycle-free.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from .flows.registry import viscoelastic_systems, walled_systems
from .harmonics import parse_mode_pairs


class ProbesParams(BaseModel):
    r"""Spectral-mode probe stream (state probes), optional.

    Enabled when ``modes`` and ``it_probes`` are both set: every
    ``it_probes`` steps the run records the complex wall-normal
    profiles ``u_hat(y)`` of the listed global spectral modes into a
    binary ``probes.bin`` (+ a ``probes.json`` schema sidecar).
    ``modes`` is an ``"i2,i3;i2,i3;..."`` list of stored-layout
    indices (axis 2 = complex slot, axis 3 = real-FFT slot -- the
    transient-growth CLI ``--modes`` convention); the mean mode
    ``(0,0)`` is allowed (it records the instantaneous mean profile).
    Wall-bounded systems only.  ``it_probes`` trades time resolution
    for disk (a record is ``8 + K*C*ny*2`` values); the buffered
    writer makes any cadence cheap at runtime.  Format, buffering,
    and the reader: the :mod:`dnsjax.probes` and
    :mod:`dnsjax.analysis.response.probes` docstrings.
    """

    model_config = ConfigDict(extra="forbid")

    modes: str | None = Field(
        default=None,
        description=(
            "Spectral modes 'i2,i3;i2,i3;...' whose complex "
            "wall-normal profiles stream to probes.bin every "
            "it_probes steps."
        ),
    )
    it_probes: int | None = Field(
        default=None,
        ge=1,
        description=("Steps between probe records; set together with modes."),
    )


class ForceParams(BaseModel):
    r"""White-in-time stochastic mode forcing (state kicks), optional.

    Enabled when ``modes`` / ``profiles`` / ``amplitude`` /
    ``it_force`` are all set (all-or-none): every ``it_force`` steps
    the main loop adds to each listed spectral mode (plus its
    real-FFT conjugate partner) a random superposition of the stored
    channel profiles -- a sequence of independent state increments
    ("kicks"), the discrete-time realisation of white-in-time
    forcing.  The drawn coefficients stream to
    ``forcing.bin``/``forcing.json``, keeping the run's full forcing
    history available to offline analysis -- e.g. their
    cross-covariance with the probe stream identifies the mode's
    linear operator (:mod:`dnsjax.analysis.response.ssi`).

    Kicks rather than a body-force term: a forcing term inside the
    nonlinear RHS would be AB2-extrapolated under ``cnab2``
    (colouring the noise) or corrector-iterated under
    ``iterative-cn``, and would trace into the jitted steppers; a
    loop-level kick keeps both schemes untouched and makes the
    per-kick response exactly the solver's own propagator.  Full
    conventions (timing relative to probes/snapshots, resume
    continuation, amplitude guidance): the :mod:`dnsjax.forcing`
    module docstring.

    Wall-bounded, non-viscoelastic systems only (the conjugate-
    partner construction currently encodes the 3-component velocity
    bases).  The whole section is **trajectory-defining**: resuming
    with changed forcing starts a new trajectory (like a ``phys``
    change).
    """

    model_config = ConfigDict(extra="forbid")

    # Forced modes should normally also be probed, or the response
    # cannot be identified (a startup note reminds when they are not).
    modes: str | None = Field(
        default=None,
        description=(
            "Spectral modes 'i2,i3;i2,i3;...' to kick (the "
            "probes.modes convention); the (0,0) mean mode is "
            "rejected."
        ),
    )
    # ``(m, C, Ny)`` complex, unit energy norm -- the
    # ``operator_tools.save_modes_npz`` bundle format; any unit-energy
    # profile set works, typically the leading controllability modes.
    # Exact grid match required; regrid offline if needed.
    profiles: str | None = Field(
        default=None,
        description=(
            "npz with per-mode channel profiles profiles_{i2}_{i3} "
            "on this run's wall-normal grid."
        ),
    )
    # Fewer channels = fewer directions identified but less injected
    # energy and a smaller operator to fit.
    n_channels: int | None = Field(
        default=None,
        ge=1,
        description=("Leading stored channels used per mode (default: all)."),
    )
    # Each kick adds ``eps * sum_j w_j profile_j`` with
    # ``w_j ~ CN(0,1)`` i.i.d.  Pick eps in the linear-response window
    # (halving it must leave the identified operator unchanged); the
    # predicted stationary forced energy for planning:
    # ``dnsjax.analysis.response.ssi.predicted_forced_variance``.
    amplitude: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Kick coefficient scale eps; the expected injected "
            "energy is eps^2 per channel per kick."
        ),
    )
    # ``Delta_f = it_force * dt``.  Larger values give cleaner
    # per-kick responses; smaller values more statistics per run time.
    it_force: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Steps between kicks; must be a multiple of the probe "
            "cadence when probing."
        ),
    )
    seed: int = Field(
        default=0,
        description=(
            "Seed of the kick-coefficient PRNG (host-side; a resumed "
            "run skips the recorded draws and continues the stream "
            "exactly)."
        ),
    )


@dataclass
class ParamExtension:
    """One registered extension parameter section.

    ``name`` is the TOML section, the CLI prefix (``--name.field``),
    and the metadata key; ``relevant(system)`` decides whether the
    section appears on a flow's surface (an irrelevant section is a
    strict error, like any other irrelevant parameter);
    ``validate(values, params)`` runs with the global checks after
    the final configuration layer -- unconditionally, so it must
    itself reject a *configured* section on an unsupported system
    (see :func:`validate_extensions`); ``trajectory_defining`` folds
    the section into the resume trajectory comparison;
    ``record_in_metadata`` includes the resolved section in snapshot
    metadata / sidecar params dumps.  ``values`` is the live merged
    singleton (analogous to the global ``params``).
    """

    name: str
    model: type[BaseModel]
    relevant: Callable[[str], bool]
    #: One line for the ``--help`` section header (the model docstring
    #: is reference documentation, too long for help output).
    summary: str = ""
    validate: Callable[[BaseModel, object], None] | None = None
    trajectory_defining: bool = False
    record_in_metadata: bool = True
    values: BaseModel = field(init=False)

    def __post_init__(self) -> None:
        self.values = self.model()


#: Registered extensions by section name (insertion order = surface /
#: help / metadata order).
EXTENSIONS: dict[str, ParamExtension] = {}

#: Section names owned by the core ``Parameters`` model; extensions
#: may not claim them.
_RESERVED_SECTIONS = frozenset(
    {
        "dist",
        "phys",
        "geo",
        "res",
        "init",
        "outs",
        "step",
        "stop",
        "solver",
    }
)


def register_extension(ext: ParamExtension) -> ParamExtension:
    """Register *ext* (idempotent per name; a name clash is an error)."""
    if ext.name in _RESERVED_SECTIONS:
        raise ValueError(
            f"extension section {ext.name!r} clashes with a core "
            "parameter section"
        )
    existing = EXTENSIONS.get(ext.name)
    if existing is not None:
        if existing.model is not ext.model:
            raise ValueError(
                f"extension section {ext.name!r} is already registered "
                f"with a different model ({existing.model.__name__})"
            )
        return existing
    EXTENSIONS[ext.name] = ext
    return ext


def relevant_extensions(system: str) -> dict[str, ParamExtension]:
    """The registered extensions applying to *system*."""
    return {
        name: ext for name, ext in EXTENSIONS.items() if ext.relevant(system)
    }


def apply_extension_layer(overlays: dict[str, dict]) -> None:
    """Merge one configuration layer's extension sections.

    *overlays* maps section name -> ``exclude_unset``-style dict of
    explicitly-set fields; ``None`` values are skipped (mirroring the
    core merge loop: a layer cannot unset a field).
    """
    for name, overlay in overlays.items():
        ext = EXTENSIONS[name]
        for key, value in overlay.items():
            if value is not None:
                setattr(ext.values, key, value)


def validate_extensions(params) -> None:
    """Run every extension's validate hook on the final configuration.

    Dispatched unconditionally -- not gated on ``relevant`` -- so a
    hook must tolerate an irrelevant system itself: return early when
    the section is unconfigured, and *reject* a configured section on
    a system the feature does not support.  (The CLI/toml surfaces
    never expose an irrelevant section, but direct assignment to the
    ``values`` singleton bypasses them, and silently skipping the
    check would let the runtime consume the stale config.)
    """
    for ext in EXTENSIONS.values():
        if ext.validate is not None:
            ext.validate(ext.values, params)


def extension_metadata(system: str) -> dict[str, dict]:
    """The recordable extension sections for *system* (resolved,
    JSON-safe) -- merged into the ``params`` dump of snapshot
    metadata and sidecar JSON
    (:func:`dnsjax.param_surface.recorded_params_dump`)."""
    return {
        name: ext.values.model_dump(mode="json")
        for name, ext in relevant_extensions(system).items()
        if ext.record_in_metadata
    }


def reset_extensions() -> None:
    """Reset every extension's ``values`` to defaults (test helper).

    Mutates the singletons in place -- consumers hold references to
    them (e.g. ``from dnsjax.extensions import force_params``), so
    they are never replaced.
    """
    for ext in EXTENSIONS.values():
        fresh = ext.model()
        for name in ext.model.model_fields:
            setattr(ext.values, name, getattr(fresh, name))


# ── Built-in extensions ──────────────────────────────────────────


def _validate_probes(values: ProbesParams, params) -> None:
    # Both knobs together, wall-bounded only, and every probed index
    # must address a *true* (unpadded) mode -- axis 2 carries
    # ``nz - 1`` complex modes, axis 3 carries ``nx // 2`` real-FFT
    # modes (Nyquist omitted; ``harmonics``).
    if (values.modes is None) != (values.it_probes is None):
        raise ValueError(
            "probes.modes and probes.it_probes must be set together "
            "(one selects the modes, the other the cadence)."
        )
    if values.modes is None:
        return
    if params.step.adaptive:
        # probes.bin records carry t, but every reader reconstructs
        # the uniform sample interval as it_probes * dt from the
        # sidecar (and dt is an append-resume match key), so the
        # stream is a fixed-dt feature.
        raise ValueError(
            "probes.modes: the probe stream requires a fixed time "
            "step (step.adaptive = False); its readers assume the "
            "uniform sample interval it_probes * dt."
        )
    if params.phys.system not in walled_systems:
        raise ValueError(
            "probes.modes: the spectral-mode probe stream supports "
            "wall-bounded systems only (system "
            f"{params.phys.system!r})."
        )
    n2, n3 = params.res.nz - 1, params.res.nx // 2
    for i2, i3 in parse_mode_pairs(values.modes):
        if i2 >= n2 or i3 >= n3:
            raise ValueError(
                f"probes.modes: mode ({i2},{i3}) out of range "
                f"(axis 2 has {n2} modes from res.nz = "
                f"{params.res.nz}, axis 3 has {n3} modes from "
                f"res.nx = {params.res.nx})."
            )


def _validate_force(values: ForceParams, params) -> None:
    # All-or-none knobs, wall-bounded non-viscoelastic only,
    # true-mode indices, no mean-mode kick, and kick/probe sample
    # alignment.
    f_set = {
        "modes": values.modes,
        "profiles": values.profiles,
        "amplitude": values.amplitude,
        "it_force": values.it_force,
    }
    missing = [k for k, v in f_set.items() if v is None]
    if missing and len(missing) < len(f_set):
        raise ValueError(
            "force.modes, force.profiles, force.amplitude and "
            "force.it_force enable the stochastic forcing together; "
            f"missing: {', '.join(missing)}."
        )
    if values.modes is None:
        # The secondary knobs have no effect without the enabling
        # quartet -- and a recorded non-default value would flag a
        # spurious trajectory change on a later resume (the section is
        # trajectory-defining), so reject rather than ignore.
        stray = [
            k
            for k, v in (
                ("n_channels", values.n_channels),
                ("seed", values.seed if values.seed != 0 else None),
            )
            if v is not None
        ]
        if stray:
            raise ValueError(
                f"force.{' / force.'.join(stray)} set without the "
                "enabling force.modes / profiles / amplitude / "
                "it_force quartet (no forcing is configured)."
            )
        return
    if params.step.adaptive:
        # The white-in-time kick statistics and their readers
        # hard-code the uniform kick interval it_force * dt (and dt
        # is an append-resume match key of forcing.json).
        raise ValueError(
            "force.modes: stochastic forcing requires a fixed time "
            "step (step.adaptive = False); the kick interval "
            "it_force * dt must be uniform."
        )
    if (
        params.phys.system not in walled_systems
        or params.phys.system in viscoelastic_systems
    ):
        raise ValueError(
            "force.modes: stochastic forcing supports the "
            "wall-bounded velocity systems only (system "
            f"{params.phys.system!r})."
        )
    n2, n3 = params.res.nz - 1, params.res.nx // 2
    force_pairs = parse_mode_pairs(values.modes)
    for i2, i3 in force_pairs:
        if i2 >= n2 or i3 >= n3:
            raise ValueError(
                f"force.modes: mode ({i2},{i3}) out of range "
                f"(axis 2 has {n2} modes, axis 3 has {n3} modes)."
            )
        if (i2, i3) == (0, 0):
            raise ValueError(
                "force.modes: the (0,0) mean mode cannot be forced "
                "(its coefficient is real, and under bulk-velocity "
                "driving it is constrained)."
            )
    probes = probes_params
    if probes.it_probes is not None:
        if values.it_force % probes.it_probes != 0:
            raise ValueError(
                f"force.it_force ({values.it_force}) must be a "
                f"multiple of probes.it_probes ({probes.it_probes}) "
                "so every kick coincides with a (pre-kick) probe "
                "sample."
            )
        probed = set(parse_mode_pairs(probes.modes))
        unprobed = [m for m in force_pairs if m not in probed]
        if unprobed:
            # A note, not an error: forcing one mode while probing
            # another is a legitimate cross-mode experiment.
            print(
                f"[force] note: forced mode(s) {unprobed} are not in "
                "probes.modes; their own response will not be "
                "recorded."
            )
    else:
        print(
            "[force] note: no probe stream configured (probes.modes); "
            "the forced responses will not be recorded (response "
            "identification, e.g. dnsjax.analysis.response.ssi, "
            "needs them)."
        )


PROBES_EXTENSION = register_extension(
    ParamExtension(
        name="probes",
        model=ProbesParams,
        relevant=lambda system: system in walled_systems,
        summary="Spectral-mode probe stream (response analysis).",
        validate=_validate_probes,
    )
)

FORCE_EXTENSION = register_extension(
    ParamExtension(
        name="force",
        model=ForceParams,
        relevant=lambda system: (
            system in walled_systems and system not in viscoelastic_systems
        ),
        summary=(
            "White-in-time stochastic mode kicks (response analysis; "
            "trajectory-defining)."
        ),
        validate=_validate_force,
        trajectory_defining=True,
    )
)

#: Live merged built-in sections (analogous to the global ``params``).
probes_params: ProbesParams = PROBES_EXTENSION.values
force_params: ForceParams = FORCE_EXTENSION.values
