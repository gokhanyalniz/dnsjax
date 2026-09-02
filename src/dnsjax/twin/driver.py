r"""Twin-run driver: lockstep perturbation-growth (predictability) DNS.

The ``dnsjax-twin`` console script steps **two** states of the same
flow in lockstep -- a reference trajectory `$\mathbf{u}^{(1)}$` loaded
from a snapshot and a perturbed partner
`$\mathbf{u}^{(2)} = \mathbf{u}^{(1)} + \delta$` -- and records online
diagnostics of the difference field
`$\Delta\mathbf{u} = \mathbf{u}^{(2)} - \mathbf{u}^{(1)}$`
(:mod:`dnsjax.twin.diagnostics`), following Egerique-de-la-Concha &
Hwang, *J. Fluid Mech.* **1036**, A52 (2026),
doi:10.1017/jfm.2026.11608.  Cartesian wall-bounded flows only
(plane-Couette / plane-Poiseuille); both states share every
singleton (grid, operators, jitted steppers, ``dt``), so their
difference is purely dynamical.

Baseline cost, before any diagnostic: a twin run is **two solver runs
sharing one program** -- 2 spectral states resident (4 under
``cnab2``, which carries an RHS history per state) and 2 stepper calls
per step, so plan `$\sim\!2\times$` the memory and `$\sim\!2\times$`
the wall time of the same flow at the same resolution.  The timing
line says ``2x steps per t`` for that reason.  What the ``[twin]``
cadences add on top is priced in :class:`TwinParams`.

Launch exactly like the production solver (``.venv/bin/dnsjax-twin
...`` from a scratch directory, under ``mpirun -np N`` only when it is
multi-process); the parameter surface is the flow's own plus the
``[twin]`` extension section below.
``step.adaptive`` and the ``[force]`` section are rejected (uniform
sampling; a kick would have to be applied identically to both states);
``[probes]`` applies to the **reference** state only, while
``stats.dat`` / ``steps.dat`` / ``corrector.dat`` are written for
*both* states (the partner's under the ``_twin`` suffix).

Initial perturbation
--------------------
`$\delta$` is the divergence-free random field of
:func:`dnsjax.ic.random_field.generate_random_state` (device-count
independent, per-global-mode seeded; the mean mode follows this
driver's own ``twin.mean_flow``, **on** by default -- unlike the
shared ``init.random_mean_flow``, which every flow defaults off and
which this driver never reads -- and is then conditioned on the
mean-mode conservation laws, so under a held mean the partner carries
the reference's bulk velocity exactly and its mean profile differs
only in ways the mean-mode dynamics admits), rescaled so
the solver-measure perturbation energy is exactly ``twin.e0``:
`$E'(\delta) = \|\delta\|^2/2 = e_0$` -- the convention of
``snapshot_perturb --perturb.amplitude_energy``.  ``twin.e0 = 0``
requests an exact zero perturbation (``state2 = copy(state1)``), the
bit-identity determinism guard.  The perturbation is applied once, at
the fresh start; a resume can never re-perturb (below).

Trajectory home, fresh start vs paired resume
---------------------------------------------
A twin trajectory lives in its run directory: the streams below, a
``twin.json`` member record (seed, ``e0``, parent snapshot and clock,
git hash, the resolved parameter dump), and paired snapshots
``state{isnap}.tar`` (reference; the standard name, so every existing
tool works on the reference trajectory) + ``state{isnap}_twin.tar``
(partner, written back-to-back with identical ``t``/``it``).  The
start mode is decided by two files -- the partner of
``init.snapshot`` and the run directory's ``twin.json``:

- partner exists **and** ``twin.json`` matches -> **paired resume**:
  both states load (reference clock inherited, wall-normal regrid as
  usual), no re-perturbation, streams append.  A trajectory-defining
  parameter change is a hard error here (a mid-twin trajectory switch
  would disconnect the pair from its own streams) -- start a fresh
  member instead.  ``stop.max_sim_time`` is a horizon counted from
  the *resumed* clock rather than an end time
  (:class:`dnsjax.parameters.Termination`), so a member restarted
  mid-horizon asks for what is left of it, not for the original
  value.
- partner exists, no ``twin.json`` -> error (either a resume in a
  directory missing its member record, or a fresh start pointed at a
  twin run's own output; copy the reference tar out to seed a fresh
  member from it).
- no partner, ``twin.json`` exists -> error (this directory already
  holds a twin trajectory; resume by pointing ``init.snapshot`` at
  its latest ``state*.tar``, or clean the directory to restart).
- neither -> **fresh start** from ``init.snapshot`` (any plain dnsjax
  snapshot): perturb, write ``twin.json``, save the IC pair
  (``outs.snapshot_save_initial``).  Any of :data:`_STREAM_FILES`
  still lying in the directory is an error here rather than something
  to append to -- every writer appends, so a half-cleaned directory
  would splice a new trajectory onto an old one's records.  The
  parent clock is inherited
  (``t0``/``it0`` from the snapshot -- offline analysis reads the
  perturbation time from ``twin.json``); a trajectory-defining
  override starts the reference at ``t = it = 0`` exactly as in
  ``dnsjax.__main__`` (``init.force_resume`` keeps the clock).

Diagnostic streams
------------------
``twin.dat`` -- the difference-field component energies
(:func:`dnsjax.twin.diagnostics.twin_energies`) every
``twin.it_energy`` steps, in the buffered, ``fsync``-ed,
non-finite-guarded ``.dat`` format of ``stats.dat`` (shared
:func:`dnsjax.__main__._flush_stats`; a ``t0`` row at setup, a final
row after the last step).

The three *per-state* solver streams are written for **both** states
at their usual cadences (``outs.it_stats`` / ``it_steps`` /
``it_corrector``): the reference into ``stats.dat`` / ``steps.dat`` /
``corrector.dat`` under the standard names, the partner into
``stats_twin.dat`` / ``steps_twin.dat`` / ``corrector_twin.dat`` --
the ``_twin`` suffix the snapshot pair already uses
(:func:`_partner_path`).  Each pair shares one column set, since the
two states are the same flow: ``get_stats``, the measurement hook and
``get_driving`` return the same keys.  A partner stream costs almost
nothing -- ``c``/``error`` already come back from its own step, and
its ``get_stats`` is the one the partner snapshot embeds, computed
once and shared.  Recording ``steps.dat`` for both is what makes the
loop run the *identical* jitted program on the two states in every
iteration: there is no measured/unmeasured split between them.

The corrector convergence guard and the non-finite exit-3 guard watch
**both** states' errors at the ``outs.it_error_check`` cadence.  Flush
sites, snapshot consistency (one checked flush before each snapshot
*pair*), SIGTERM/SIGINT flushing, and the FATAL / exit-3 semantics all
mirror :mod:`dnsjax.__main__`.

Ensembles
---------
One member = one run directory = one ``dnsjax-twin`` invocation; vary
``twin.seed`` (and parent snapshot) across members.  An unset
``twin.seed`` is drawn from the system entropy pool
(:mod:`dnsjax.seeding`) on a fresh start, so members launched by hand
are independent without being told to be; a paired resume adopts the
seed recorded in ``twin.json`` rather than drawing, which is what keeps
:data:`_TWIN_MATCH_KEYS` satisfiable without re-typing it.
``scripts/ensemble_setup.py`` harvests parent snapshots and builds
member trees (pinning each member's seed in its ``parameters.toml``);
``dnsjax.analysis.twin`` aggregates the ``twin.dat`` streams.
"""

import json
import math
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns

from pydantic import BaseModel, ConfigDict, Field

from ..__main__ import (
    _flush_stats,
    _interpolate_if_needed,
    _write_dat_header,
)
from ..__main__ import (
    _stats_row as _row,
)
from ..bootstrap import (
    configure_jax_runtime,
    resolve_parameters,
    resolve_seed,
)
from ..extensions import (
    ParamExtension,
    force_params,
    probes_params,
    register_extension,
    relevant_extensions,
)
from ..flows.registry import cartesian_systems
from ..param_surface import print_resolved_parameters, recorded_params_dump
from ..parameters import (
    derived_params,
    ns_to_s,
    padded_res,
    params,
    trajectory_defining_changes,
)
from ..seeding import (
    SOURCE_CLI,
    SOURCE_DRAWN,
    SOURCE_SIDECAR,
    seed_note,
)
from ..snapshot_meta import (
    git_hash,
    is_snapshot_file,
    read_snapshot_meta,
    write_sidecar_json,
)

_PROG = "dnsjax-twin"

#: ``twin.json`` schema version.  Bump when the stored *meaning*
#: changes (the probes/forcing sidecar discipline).
TWIN_FORMAT_VERSION: int = 1

#: ``twin.json`` keys that must match for a paired resume to proceed.
#: The cadences are included: a mid-stream cadence change would break
#: the uniform sample grid the offline fits assume.  ``seed`` is
#: matched, not re-read, so a resume with an unset ``twin.seed`` adopts
#: the recorded one first (in :func:`run`); it is ``null`` exactly when
#: ``e0 = 0``, which draws no perturbation and so has no seed.
_TWIN_MATCH_KEYS: tuple[str, ...] = (
    "format_version",
    "system",
    "e0",
    "seed",
    "smoothness",
    "mean_flow",
    "bins",
    "it_energy",
    "it_budget",
    "it_spectra",
    "it_yspectra",
    "it_ybudget",
    "dt",
    "double_precision",
)

#: Value assumed for a :data:`_TWIN_MATCH_KEYS` entry that a
#: ``twin.json`` written before the key existed does not carry -- so
#: adding one does not strand recorded members.  A key absent from
#: here still mismatches when absent from the file, which is the right
#: default: only a key whose *old* behaviour was a fixed, known value
#: can be back-filled.  ``mean_flow`` is one: the partner followed the
#: Cartesian ``init.random_mean_flow``, whose default was on, so a
#: member recorded then resumes unchanged (and one deliberately run
#: mean-free re-states ``--twin.mean_flow False``, which is correct
#: -- it is a real difference in what was perturbed).  No
#: :data:`TWIN_FORMAT_VERSION` bump: no existing key changes meaning.
_TWIN_LEGACY_DEFAULTS: dict[str, object] = {"mean_flow": True}


class TwinParams(BaseModel):
    r"""Twin-run driver section (``dnsjax-twin`` only), optional.

    Enabled when ``e0`` is set: the driver perturbs the loaded
    reference state by a random divergence-free field of exactly that
    perturbation energy (solver measure, `$\|\delta\|^2/2$`) and
    steps both states in lockstep, streaming the difference-field
    component energies to ``twin.dat`` every ``it_energy`` steps.
    ``e0 = 0`` requests an exact zero perturbation (the determinism
    guard).  Cartesian wall-bounded flows, fixed time step; details
    and the resume rules: the :mod:`dnsjax.twin.driver` module
    docstring.

    Cost of the two optional streams -- neither is priced by its
    cadence alone:

    - ``it_budget`` sets the **run's** peak memory, not just its
      per-sample cost.  ``_twin_budget_jit`` is a separate compiled
      program whose transient (~21 physical-component fields live at
      once -- the 9 cached advectors, 9 gradients of the current
      `$\mathbf{c}$`, and `$\mathbf{q}$` -- plus ~6 masked spectral
      states) is the driver's global high-water mark, since the
      device allocator's pool grows to the maximum over every
      program.  Order `$40$` GB at a `$1024\times257\times256$`
      double-precision plane-Poiseuille target.  If it ever binds,
      the two ways to trade transforms for footprint are in
      :mod:`dnsjax.twin.diagnostics`' "Budget terms".
      In *time* it is equally unsubtle: one sample costs
      `$\sim\!0.9$` of a twin step (measured, size-independent over
      `$48^3$`-`$64^3$`), so ``it_budget = 1`` nearly doubles the
      run and `$10$` costs `$\sim\!9\,\%$`.
    - ``it_ybudget`` costs memory in **two** places, and the
      per-sample transient is the one that is easy to miss.
      *Resident*: :class:`dnsjax.twin.pressure.DifferencePressure`
      holds a second factored banded operator the size of
      ``flow.Lk_op`` plus its two homogeneous columns, two real
      `$(N_y, N_{k_z}, N_{k_x})$` fields (its "Cost" section has the
      numbers).  *Transient*:
      :func:`dnsjax.twin.diagnostics._convective_sources` holds 15
      padded physical fields at once -- 6 advector plus one gradient
      set, the two gradient sets being formed one at a time for
      exactly this reason (holding both is 24, *more* than the
      three-bin pass's ~21 despite half the transforms).  Under
      ``twin.rotational_ybudget`` it is 12, and 21 transforms rather
      than 33.  XLA still schedules, so watch it rather than
      assuming either.
      That figure counts the padded **physical** set alone, which is
      not the whole transient: live *spectral* arrays at the same
      point are `$\Delta\mathbf{u}$`, the four signed operands of
      ``_Sources.advective``, `$\hat{\mathcal N}$`, and then
      ``div_n`` and `$\Delta\hat p$`.  A complex
      `$(3, N_y, N_{k_z}, N_{k_x})$` field is `$24 n_x n_y n_z$`
      bytes against a padded physical component's
      `$18 n_x n_y n_z$` (wall-bounded oversamples `$x$`-`$z$` only,
      `$1.5^2$`), so those `$\sim\!6$` are `$\sim\!8$` more
      padded-component equivalents: size a job against `$\sim\!23$`,
      not 15.  That last step is arithmetic off the shapes rather
      than a measurement, which is one more reason to watch the run.
      ``solver.rhs_transform_chunks``
      caps the transform-stage transient inside each
      :func:`dnsjax.fft.chunked_transform` call; it does **not** touch
      the live field count.
    - ``x0_planes`` gates the `$k_x = 0$` plane of **both**
      `$y$`-resolved streams, and is a static flag on
      :func:`~dnsjax.twin.diagnostics.twin_yspectra` and
      :func:`~dnsjax.twin.diagnostics.twin_ybudget` for the same
      reason ``spectra_ref`` is.  It is not a second field pass --
      the plane is a slice of one already taken -- but it is a third
      of the sample's ``psum`` payload and a third of every stored
      record, on top of ``spectra_ref``'s doubling of the first.
      What it costs to leave off is
      :func:`~dnsjax.analysis.twin.bin_energies`.
    - ``spectra_ref`` gates the reference half of **both** spectra
      streams, in compute as well as on disk: it is a static flag on
      :func:`dnsjax.twin.diagnostics.twin_spectra_2d` and
      :func:`~dnsjax.twin.diagnostics.twin_yspectra`, so with it off
      the reference reduction is never traced.  On the `$(k_z, k_x)$`
      stream that saves one field pass and a `$\sim$`1 MB ``psum``
      per sample; on the `$y$`-resolved one it saves a full real
      `$(3, N_y, N_{k_z}, N_{k_x})$` density and one of the sample's
      two collectives -- about half of it.  What it costs is the
      decorrelation ratio.

    ``it_energy`` is the one per-*step* cost at its default of 1, and
    what it costs depends on ``bins``.  **Off** (the default): one
    extra jitted call per step forming ``delta`` and reducing it and
    ``state1``, three full-state passes against the two steps' FFT and
    solve work.  **On**: ``delta`` and ``du1`` are each read four
    times, so both materialise (~2 full-state complex temporaries) --
    **a few percent of a twin step**, measured at 1.3 % for
    plane-Couette `$48^3$` and 5.0 % at `$64^3$` (the rise is the
    working set leaving cache; the step itself is pure FFT/solve work
    either way), and the ``E_d`` vs ``E_dU + E_du1 + E_du2``
    redundancy those percent buy is a deliberate consistency guard,
    not something to trade away.  Either way the default cadence
    needs no tuning, which is just as well: it is the intended
    Lyapunov sampling rate.
    """

    model_config = ConfigDict(extra="forbid")

    e0: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Initial perturbation energy E'(delta) in the solver "
            "measure; 0 = exact zero perturbation; unset = section "
            "unconfigured."
        ),
    )
    # Unset means "draw one" (:mod:`dnsjax.seeding`), but resolved late
    # -- in ``run`` below, once the fresh-start / paired-resume decision
    # is known.  A resume adopts the seed recorded in ``twin.json``
    # instead of drawing, which is what keeps the _TWIN_MATCH_KEYS check
    # satisfiable without the user re-typing it; ``e0 = 0`` draws nothing
    # at all and records ``null``.
    seed: int | None = Field(
        default=None,
        description=(
            "Perturbation RNG seed (device-count independent); vary "
            "per ensemble member. Unset: drawn from the system entropy "
            "pool on a fresh start, and taken from twin.json on a "
            "resume."
        ),
    )
    smoothness: float = Field(
        default=0.4,
        gt=0,
        lt=1,
        description=(
            "Spectral envelope of the random perturbation "
            "(init.random_smoothness convention)."
        ),
    )
    # The partner's own knob rather than the shared
    # ``init.random_mean_flow`` (which the driver does not read at
    # all: ``dnsjax-twin`` never builds a random *initial condition*,
    # it perturbs a loaded one).  Two reasons it cannot be the shared
    # field.  The defaults differ -- a twin pair wants the mean mode
    # perturbed, a plain run defaults it off -- and ``init`` fields
    # are inherited from the parent snapshot, so a twin started from
    # any ordinary run's snapshot would silently pick that run's
    # ``false`` up as an explicitly-set layer.  Conversely, forcing
    # the shared field on here would write ``random_mean_flow: true``
    # into the twin's own paired snapshots, and a later plain resume
    # from the reference tar would inherit it.
    mean_flow: bool = Field(
        default=True,
        description=(
            "Perturb the mean (kx = kz = 0) profile too, conditioned "
            "on its conservation laws (ic/mean_mode.py). On by "
            "default -- unlike init.random_mean_flow, which this "
            "driver does not read: the mean mode is a difference-field "
            "component in its own right (the Delta-U bin). Off gives a "
            "mean-free partner."
        ),
    )
    bins: bool = Field(
        default=False,
        description=(
            "Also record the Delta-U / Delta-u1 / Delta-u2 three-bin "
            "decomposition in twin.dat (the reference paper's "
            "binning); required by it_budget.  Off by default: the "
            "wall-normal-resolved spectra (it_yspectra) supersede it "
            "and it costs a few percent of every step."
        ),
    )
    x0_planes: bool = Field(
        default=False,
        description=(
            "Also store the k_x = 0 plane (the _x0 fields) in "
            "twin_yspectra.bin / twin_ybudget.bin -- the k_z-resolved "
            "spectrum of the streamwise-averaged field, which is what "
            "recovers the Delta-U / Delta-u1 / Delta-u2 three-bin "
            "split (analysis.twin.bin_energies).  Off by default: it "
            "is a third of every record and of the sample's psum, and "
            "the (y, k) marginals plus the always-stored _xz00 mean "
            "mode supersede it.  Off, it is never traced."
        ),
    )
    it_energy: int = Field(
        default=1,
        ge=1,
        description="Steps between twin.dat energy rows.",
    )
    it_budget: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Steps between twin_budget.dat rows (the production/"
            "transport/dissipation terms); unset disables the stream. "
            "Not a pure cadence knob: enabling it raises the run's "
            "peak memory (see the field's docs)."
        ),
    )
    it_spectra: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Steps between (kz, kx) difference-energy spectra "
            "records; unset disables the stream."
        ),
    )
    it_yspectra: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Steps between twin_yspectra.bin records (componentwise "
            "wall-normal-resolved (y, kz) and (y, kx) energy "
            "spectra); unset disables the stream."
        ),
    )
    it_ybudget: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Steps between twin_ybudget.bin records (the same bins' "
            "production / transfer / viscous / pressure densities); "
            "unset disables the stream.  Like it_budget, not a pure "
            "cadence knob: it raises both the run's resident and its "
            "peak memory (see the field's docs)."
        ),
    )
    rotational_ybudget: bool = Field(
        default=False,
        description=(
            "Write twin_ybudget.bin in the rotational form of the "
            "nonlinear term instead of the convective one.  Its two "
            "transfer terms are then exactly redistributive at every "
            "y and its nonlinear term is the solver's own, at the "
            "cost of a split that does not map onto the reference "
            "paper's production/transport terms (see the "
            "twin.diagnostics docstring).  Off by default."
        ),
    )
    spectra_ref: bool = Field(
        default=True,
        description=(
            "Also record the reference state's spectrum with each "
            "spectra / yspectra sample (for decorrelation ratios); "
            "turning it off also skips computing it."
        ),
    )


def _validate_twin(values: TwinParams, params) -> None:
    # Unconfigured (no e0): reject stray secondary knobs rather than
    # silently ignoring them (the [force] discipline), else return.
    if values.e0 is None:
        defaults = TwinParams()
        stray = [
            name
            for name in (
                "seed",
                "smoothness",
                "mean_flow",
                "bins",
                "x0_planes",
                "it_energy",
                "it_budget",
                "it_spectra",
                "it_yspectra",
                "it_ybudget",
                "rotational_ybudget",
                "spectra_ref",
            )
            if getattr(values, name) != getattr(defaults, name)
        ]
        if stray:
            raise ValueError(
                f"twin.{' / twin.'.join(stray)} set without twin.e0 "
                "(no twin run is configured)."
            )
        return
    # Configured: Cartesian wall-bounded only, fixed dt, sane ranges.
    # The range checks repeat the Field constraints because direct
    # assignment to the values singleton bypasses pydantic (the
    # validate_extensions contract).
    if params.phys.system not in cartesian_systems:
        raise ValueError(
            "twin.e0: the twin-run driver supports the Cartesian "
            "wall-bounded flows only (system "
            f"{params.phys.system!r})."
        )
    if params.step.adaptive:
        raise ValueError(
            "twin.e0: the twin driver requires a fixed time step "
            "(step.adaptive = False); the two states share one dt "
            "and the twin.dat readers assume a uniform sample "
            "interval."
        )
    if values.e0 < 0:
        raise ValueError("twin.e0 must be >= 0.")
    if not (0 < values.smoothness < 1):
        raise ValueError("twin.smoothness must lie in (0, 1).")
    for name in (
        "it_energy",
        "it_budget",
        "it_spectra",
        "it_yspectra",
        "it_ybudget",
    ):
        cadence = getattr(values, name)
        if cadence is not None and cadence < 1:
            raise ValueError(f"twin.{name} must be >= 1.")
    if (
        values.it_yspectra is not None or values.it_ybudget is not None
    ) and params.res.nz % 2:
        raise ValueError(
            "twin.it_yspectra / twin.it_ybudget need an even res.nz "
            f"(got {params.res.nz}): the wall-normal-resolved streams "
            "store the k_z axis folded onto |k_z|, and at odd nz the "
            "stored band is asymmetric -- the highest negative mode "
            "has no positive partner to fold onto "
            "(dnsjax.twin.diagnostics._fold_kz).  Even sizes are the "
            "recommended resolutions anyway (dnsjax.fft)."
        )
    if values.it_budget is not None and not values.bins:
        raise ValueError(
            "twin.it_budget needs twin.bins: the three-bin budget is "
            "checked against twin.dat's E_dU / E_du1 / E_du2 "
            "(analysis.twin.series.closure_residuals), which "
            "twin.bins writes.  For the scale-resolved budget set "
            "twin.it_ybudget instead."
        )


TWIN_EXTENSION = register_extension(
    ParamExtension(
        name="twin",
        model=TwinParams,
        relevant=lambda system: system in cartesian_systems,
        summary=("Twin-run perturbation-growth driver (dnsjax-twin only)."),
        validate=_validate_twin,
        record_in_metadata=False,
    )
)

#: Live merged section (analogous to the global ``params``).
twin_params: TwinParams = TWIN_EXTENSION.values


#: Every file a twin run appends to.  A fresh start refuses to touch
#: a directory holding any of them (see :func:`run`), because all
#: three writer families append rather than truncate: a ``.dat``
#: stream because :class:`_ScalarStream` re-writes no header, and a
#: ``.bin`` stream whenever its sidecar happens to match this run.
#: A superset of ``scripts/twin_postprocess.py``'s ``_OUTPUTS``: the
#: streams that script rebuilds, plus the ones it cannot.
#: ``probes.*`` is here too -- reference-only, and written by plain
#: ``dnsjax`` as well, but a twin run appends to it like the rest.
_STREAM_FILES: tuple[str, ...] = (
    "twin.dat",
    "twin_budget.dat",
    "stats.dat",
    "stats_twin.dat",
    "steps.dat",
    "steps_twin.dat",
    "corrector.dat",
    "corrector_twin.dat",
    "twin_spectra.bin",
    "twin_spectra.json",
    "twin_yspectra.bin",
    "twin_yspectra.json",
    "twin_ybudget.bin",
    "twin_ybudget.json",
    "probes.bin",
    "probes.json",
)


def _partner_path(reference: Path) -> Path:
    """The partner snapshot's path for a reference ``state*.tar``."""
    return reference.with_name(f"{reference.stem}_twin{reference.suffix}")


class _ScalarStream:
    """Buffered ``.dat`` column stream (twin-driver instances).

    The per-stream state machine of the :mod:`dnsjax.__main__` inline
    buffers -- an on-device ``(nbuffer, n_cols)`` buffer, a host
    timestamp list and fill index, disk I/O gated on the main device
    with the index reset on **all** ranks (lockstep) -- factored as a
    class because the twin driver runs several streams.  Rows are
    written by the shared :func:`dnsjax.__main__._flush_stats`
    (append + ``fsync`` + post-write non-finite scan) and the header
    by :func:`dnsjax.__main__._write_dat_header`, once and only when
    the file does not exist, so a resume appends.  :meth:`push`
    fill-flushes when the buffer is full;
    both methods return the non-finite diagnostic message (main
    process only) and the caller aborts on it.
    """

    def __init__(self, path, names, *, jnp, sharding) -> None:
        self._jnp = jnp
        self._sharding = sharding
        self.names = list(names)
        self.path = Path(path)
        self._p = params.outs.stats_precision - 1
        val_width = params.outs.stats_precision + 7
        self._col_width = max(
            val_width, max(len(n) for n in ["t"] + self.names)
        )
        self._nbuffer = params.outs.nbuffer
        self._buffer = jnp.zeros(
            (self._nbuffer, len(self.names)), dtype=sharding.float_type
        )
        self._ts: list[float] = []
        self._idx: int = 0
        if sharding.main_device and not self.path.exists():
            _write_dat_header(self.path, ["t"] + self.names, self._col_width)

    def push(self, values, t: float) -> str | None:
        """Buffer one row; flush (checked) when the buffer fills."""
        self._buffer = self._buffer.at[self._idx].set(values)
        self._ts.append(t)
        self._idx += 1
        if self._idx == self._nbuffer:
            return self.flush()
        return None

    def flush(self, check: bool = True) -> str | None:
        """Write the buffered rows durably; reset on all ranks."""
        if self._idx == 0:
            return None
        bad = None
        if self._sharding.main_device:
            bad = _flush_stats(
                self._buffer,
                self._idx,
                self._ts,
                self.path,
                self._p,
                self._col_width,
                self.names if check else None,
            )
        self._ts.clear()
        self._idx = 0
        return bad


def run(wall_time_start: int, seed_source: str | None = None) -> None:
    """Run the twin time-stepping loop (parameters and JAX final).

    Mirrors :func:`dnsjax.__main__.run` with two states; see the
    module docstring for what differs.  *seed_source* is the
    ``seeding.SOURCE_*`` label of the layer that supplied ``twin.seed``
    (``None`` when none did, and the seed is resolved in here).
    """
    import importlib

    import jax
    import numpy as np
    from jax import numpy as jnp

    from ..flows.registry import spec_for
    from ..sharding import sharding

    # --- Flow dispatch (registry-driven, as in __main__) -----------------
    _spec = spec_for(params.phys.system)
    _flow_mod = importlib.import_module(_spec.flow_module)
    get_perturbation_energy = _flow_mod.get_perturbation_energy
    get_stats = _flow_mod.get_stats
    get_driving = getattr(_flow_mod, "get_driving", None)
    predict_and_fully_correct = _flow_mod.predict_and_fully_correct
    predict_and_fully_correct_measured = (
        _flow_mod.predict_and_fully_correct_measured
    )
    step_cnab2 = _flow_mod.step_cnab2
    step_cnab2_measured = _flow_mod.step_cnab2_measured

    # The [twin] validation restricts to the Cartesian flows, whose
    # state is physical components throughout -- no solver-basis
    # crossing anywhere in this driver.
    from . import diagnostics

    twin_energies = diagnostics.twin_energies

    # --- Driver-level configuration guards -------------------------------
    # These are dnsjax-twin requirements beyond the [twin] validate
    # hook (which also runs under a hypothetical non-twin entry
    # point and must not constrain unrelated sections).
    if twin_params.e0 is None:
        raise SystemExit(
            f"{_PROG}: error: configure the [twin] section "
            "(twin.e0 at minimum); see "
            f"`{_PROG} --help {params.phys.system}`."
        )
    if force_params.modes is not None:
        # A configured [force] can only come from the resumed
        # snapshot's embedded params (the section rides the shared
        # surface): stochastic kicks would have to be applied
        # identically to both states to keep the difference purely
        # dynamical, which the driver does not support.
        raise SystemExit(
            f"{_PROG}: error: the [force] section is not supported "
            "by the twin driver (the resumed snapshot records "
            "stochastic forcing); resume its trajectory with plain "
            "dnsjax, or harvest an unforced snapshot."
        )
    if params.init.snapshot is None or not is_snapshot_file(
        params.init.snapshot
    ):
        raise SystemExit(
            f"{_PROG}: error: a twin run starts from a dnsjax "
            "snapshot (--init.snapshot state*.tar); the in-process "
            "start modes are not supported."
        )

    from ..snapshot import (
        load_snapshot,
        read_metadata,
        save_snapshot,
        validate_snapshot_params,
    )

    # --- Reference state -------------------------------------------------
    ref_path = Path(params.init.snapshot)
    validate_snapshot_params(ref_path)
    state1, t_snap, it_snap = load_snapshot(ref_path)
    meta1 = read_metadata(ref_path)
    changes = trajectory_defining_changes(meta1["params"])

    partner = _partner_path(ref_path)
    json_path = Path("twin.json")
    have_partner = partner.exists()
    have_json = json_path.exists()
    old: dict = {}
    if have_json:
        with open(json_path) as f:
            old = json.load(f)

    # --- The perturbation seed (before the branch: the two differ) -------
    # A paired resume never re-perturbs, so it needs no draw -- it needs
    # the seed its pair was *born* with, the one recorded in twin.json.
    # Adopting it is also what makes the _TWIN_MATCH_KEYS check below
    # satisfiable for a member whose seed was drawn rather than typed.
    # A fresh start draws one; ``e0 = 0`` perturbs by exactly nothing and
    # needs no seed at all, recording ``null``.
    if twin_params.seed is None:
        if have_partner and have_json:
            twin_params.seed = old.get("seed")
            seed_source = SOURCE_SIDECAR
        elif not have_partner and not have_json and twin_params.e0 != 0.0:
            # Every rank, outside any main-device gate: the draw is
            # agreed by a collective so the partner is one field, not
            # one per rank.
            twin_params.seed = resolve_seed("twin.seed", "--twin.seed", None)
            seed_source = SOURCE_DRAWN
    if twin_params.seed is not None:
        # ``or SOURCE_CLI``: a seed that reached here already set, with
        # no layer recorded for it, was assigned directly (a test).
        sharding.print(
            seed_note("twin.seed", twin_params.seed, seed_source or SOURCE_CLI)
        )

    # --- Fresh start vs paired resume (see the module docstring) ---------
    resumed_pair: bool = False
    if have_partner and have_json:
        current = _twin_sidecar_stub()
        mismatch = [
            k
            for k in _TWIN_MATCH_KEYS
            if old.get(k, _TWIN_LEGACY_DEFAULTS.get(k)) != current[k]
        ]
        if mismatch:
            raise SystemExit(
                f"{_PROG}: error: this directory's twin.json does "
                "not match the configured run (differs in: "
                f"{', '.join(mismatch)}); a twin trajectory cannot "
                "change these on resume.  Start a fresh member in a "
                "clean directory instead."
            )
        if changes:
            raise SystemExit(
                f"{_PROG}: error: trajectory-defining parameters "
                f"changed on a twin pair resume ({'; '.join(changes)}"
                "); the pair would disconnect from its recorded "
                "streams.  Start a fresh member instead."
            )
        validate_snapshot_params(partner)
        state2, t2_snap, it2_snap = load_snapshot(partner)
        if (t2_snap, it2_snap) != (t_snap, it_snap):
            raise SystemExit(
                f"{_PROG}: error: partner snapshot {partner} is at "
                f"(t={t2_snap}, it={it2_snap}) but the reference is "
                f"at (t={t_snap}, it={it_snap}); the pair is "
                "inconsistent (a crash between the two writes?  "
                "resume from an earlier complete pair)."
            )
        params.init.t0 = t_snap
        params.init.it0 = it_snap
        isnap_start = meta1["isnap"] + 1
        resumed_pair = True
        state1 = _interpolate_if_needed(
            state1, ref_path, read_metadata, sharding, jnp
        )
        state2 = _interpolate_if_needed(
            state2, partner, read_metadata, sharding, jnp
        )
        sharding.print(
            f"Resumed twin pair: t={t_snap:.6e}, it={it_snap} "
            f"({ref_path.name} + {partner.name})"
        )
    elif have_partner:
        raise SystemExit(
            f"{_PROG}: error: {partner} exists but this directory "
            "has no twin.json member record.  To resume that pair, "
            "restore its twin.json; to seed a fresh member from a "
            "twin run's reference snapshot, copy the reference tar "
            "into the parent-snapshot store (without its partner) "
            "and point --init.snapshot at the copy."
        )
    elif have_json:
        raise SystemExit(
            f"{_PROG}: error: this directory already holds a twin "
            "trajectory (twin.json exists) but "
            f"{ref_path.name} has no partner snapshot.  Resume by "
            "pointing --init.snapshot at the latest state*.tar "
            "written here, or clean the directory (twin.json + "
            "streams) to restart the member from its parent."
        )
    else:
        # Fresh start.  Stale streams without the member record mean
        # a corrupt / half-cleaned directory; refuse to append.  All
        # of them, not just ``twin.dat``: every writer here appends
        # (:data:`_STREAM_FILES`), so a half-cleaned directory would
        # otherwise splice a new trajectory onto an old one's records
        # -- silently for the ``.dat`` streams, and for a ``.bin``
        # stream whenever its sidecar matched.
        stale = [name for name in _STREAM_FILES if Path(name).exists()]
        if stale:
            raise SystemExit(
                f"{_PROG}: error: stale {', '.join(stale)} without "
                "twin.json in this directory; move them away."
            )
        if changes and not params.init.force_resume:
            sharding.print(
                "Fresh twin start: trajectory-defining parameters "
                "changed vs the parent snapshot; starting the "
                "reference at t=it=0 (init.force_resume keeps the "
                "parent clock).  Changes: " + "; ".join(changes)
            )
            params.init.t0 = 0.0
            params.init.it0 = 0
        else:
            if changes:
                sharding.print(
                    "Fresh twin start: force_resume set; keeping the "
                    "parent clock despite changes ("
                    + "; ".join(changes)
                    + ")."
                )
            params.init.t0 = t_snap
            params.init.it0 = it_snap
        isnap_start = params.init.isnap0
        state1 = _interpolate_if_needed(
            state1, ref_path, read_metadata, sharding, jnp
        )

        # --- The perturbation (once, here, never on resume) ---
        if twin_params.e0 == 0.0:
            # Exact zero perturbation: the determinism guard.  Copy,
            # not alias -- the steppers donate their inputs.
            state2 = jnp.copy(state1)
            sharding.print(
                "Twin partner: exact copy (twin.e0 = 0; bit-identity mode)."
            )
        else:
            from ..ic.random_field import generate_random_state

            grid_before = derived_params.wall_normal_grid
            # ||delta|| = sqrt(2 e0) makes E'(delta) = e0 already;
            # the explicit rescale below guards the convention
            # against generator changes at zero cost.
            delta = generate_random_state(
                math.sqrt(2.0 * twin_params.e0),
                twin_params.smoothness,
                twin_params.seed,
                # ``twin.mean_flow`` (on by default), not the shared
                # ``init.random_mean_flow`` -- see the field's docs.
                # The partner's (0, 0) perturbation is conditioned on
                # the mean-mode conservation laws
                # (:mod:`dnsjax.ic.mean_mode`), so under a held mean it
                # shares the reference's bulk velocity exactly and the
                # difference field's mean profile is a genuine
                # perturbation direction rather than a bookkeeping
                # artefact.  ``--twin.mean_flow False`` gives the
                # mean-free partner this used to hardcode.
                twin_params.mean_flow,
            )
            if grid_before is not None and not np.allclose(
                np.asarray(grid_before),
                np.asarray(derived_params.wall_normal_grid),
                atol=1e-12,
            ):  # pragma: no cover - same params build the same grid
                raise RuntimeError(
                    "the random-field generator rebuilt a different "
                    "wall-normal grid than the loaded snapshot's; "
                    "refusing to perturb across grids."
                )
            e_delta = float(get_perturbation_energy(delta))
            if not (math.isfinite(e_delta) and e_delta > 0):
                raise SystemExit(
                    f"{_PROG}: error: generated perturbation has "
                    f"E' = {e_delta}; cannot rescale to twin.e0."
                )
            state2 = state1 + delta * math.sqrt(twin_params.e0 / e_delta)
            sharding.print(
                f"Twin partner: reference + random perturbation "
                f"(E' = {twin_params.e0:.3e}, "
                f"seed = {twin_params.seed})."
            )

        if sharding.main_device:
            sidecar = _twin_sidecar_stub()
            sidecar.update(
                parent=str(ref_path),
                parent_t=params.init.t0,
                parent_it=params.init.it0,
                git_hash=git_hash(),
                params=recorded_params_dump(params),
            )
            write_sidecar_json(json_path, sidecar)

    # --- Stopping criteria -----------------------------------------------
    wall_time_stop = (
        jnp.inf
        if params.stop.max_wall_time is None
        else int(params.stop.max_wall_time.total_seconds() / ns_to_s)
    )
    # Relative to the initial condition, as in ``dnsjax.__main__``:
    # the horizon is counted from ``init.t0`` -- the parent's clock on
    # a fresh start, the resumed pair's on a paired resume (so a
    # member restarted mid-horizon gets a fresh horizon from there).
    t_stop = (
        jnp.inf
        if params.stop.max_sim_time is None
        else params.init.t0 + params.stop.max_sim_time
    )
    if params.stop.max_sim_time is not None and params.init.t0 != 0.0:
        sharding.print(
            f"Stopping at t = {t_stop:.6e} (init.t0 + stop.max_sim_time)."
        )

    it: int = params.init.it0
    it0: int = params.init.it0
    t: float = params.init.t0
    isnap: int = isnap_start
    last_saved_it: int | None = None
    dt_first: float = params.step.dt
    wall_time_now: int = perf_counter_ns()
    last_error: float = 0.0

    check_laminarization: bool = params.stop.check_laminarization
    laminarization_threshold: float = params.stop.laminarization_threshold
    laminarized: bool = False
    e_prime_host: float = float("inf")

    it_error_check: int = params.outs.it_error_check
    c_sum = jnp.zeros((), dtype=jnp.int32)
    e1_dev = None
    e2_dev = None
    c1_dev = None
    c2_dev = None

    def _save_pair(s1, s2, t, it, snap_stats, snap_stats2, isnap):
        """Write the state{isnap}.tar / state{isnap}_twin.tar pair.

        Every rank calls ``save_snapshot`` twice in the same order
        (its internal barriers are collective); the caller flushes
        the streams once *before* the pair, never between.  Both
        states' stats arrive as arguments, computed by the caller
        under the same gate that feeds ``stats.dat`` /
        ``stats_twin.dat``, so a snapshot step coinciding with a stats
        step traverses each state once rather than twice.
        """
        width = params.outs.snapshot_pad_width
        name = f"state{isnap:0{width}d}"
        embed = params.outs.snapshot_embed_stats
        save_snapshot(
            s1,
            t,
            it,
            f"{name}.tar",
            stats=(snap_stats if embed else None),
            isnap=isnap,
        )
        save_snapshot(
            s2,
            t,
            it,
            f"{name}_twin.tar",
            stats=(snap_stats2 if embed else None),
            isnap=isnap,
        )
        return isnap + 1

    # --- Warm-ups (JIT outside the benchmark window) ----------------------
    stats = get_stats(state1)
    stats2 = get_stats(state2)
    tvals = twin_energies(state1, state2, bins=twin_params.bins)
    measure_budget: bool = twin_params.it_budget is not None
    if measure_budget:
        twin_budget = diagnostics.twin_budget
        bvals = twin_budget(state1, state2)
    # Both y-resolved streams share it, and both take it statically.
    _yx0 = twin_params.x0_planes
    measure_yspectra: bool = twin_params.it_yspectra is not None
    if measure_yspectra:
        # ``spectra_ref`` static, as for the (k_z, k_x) stream above:
        # with it off the reference density and its psum -- about half
        # this sample -- are never traced.
        _yref = twin_params.spectra_ref

        def twin_yspectra(s1, s2):
            return diagnostics.twin_yspectra(s1, s2, ref=_yref, x0=_yx0)

        yvals = twin_yspectra(state1, state2)
    measure_ybudget: bool = twin_params.it_ybudget is not None
    if measure_ybudget:
        from .pressure import DifferencePressure

        # One factored Poisson operator, held for the run (see the
        # ``twin.it_ybudget`` field docs for what it costs).
        pressure = DifferencePressure(diagnostics.flow, diagnostics.fourier)
        rot = twin_params.rotational_ybudget

        def twin_ybudget(s1, s2):
            return diagnostics.twin_ybudget(
                s1, s2, pressure, rotational=rot, x0=_yx0
            )

        ybvals = twin_ybudget(state1, state2)

    sharding.print(
        f"t = {t:.2f}",
        f"E_d={float(tvals['E_d']):.3e}",
        f"E_ref={float(tvals['E_ref']):.3e}",
        *[f"{x}={y:.3e}" for x, y in stats.items()],
    )

    bad_init = (
        [k for k, v in stats.items() if not math.isfinite(float(v))]
        + [
            f"{k} (twin)"
            for k, v in stats2.items()
            if not math.isfinite(float(v))
        ]
        + [k for k, v in tvals.items() if not math.isfinite(float(v))]
    )
    if measure_budget:
        bad_init += [
            k for k, v in bvals.items() if not math.isfinite(float(v))
        ]
    if bad_init:
        sharding.print(
            f"FATAL: non-finite initial statistic(s) "
            f"{', '.join(bad_init)} at t = {t:.6e}; aborting."
        )
        sys.exit(3)

    if check_laminarization:
        jax.block_until_ready(get_perturbation_energy(state1))

    # --- IC snapshot pair (fresh start only) -----------------------------
    if params.outs.snapshot_save_initial and not resumed_pair:
        isnap = _save_pair(state1, state2, t, it, stats, stats2, isnap)
        last_saved_it = it

    # --- Streams ---------------------------------------------------------
    # Each state's applied mean-mode driving (the last ``stats*.dat``
    # columns): a *step* quantity threaded out of the corrector, so the
    # ``t = t0`` row -- which has no step behind it -- carries the
    # wall-shear inference instead (``get_driving``).  ``get_driving``
    # takes the *physical* view of a state; this driver is
    # Cartesian-only, whose solver basis **is** the physical one (no
    # ``to_solver_basis`` anywhere here), so both states satisfy that
    # as they stand.
    last_drive1 = get_driving(state1) if get_driving is not None else {}
    last_drive2 = get_driving(state2) if get_driving is not None else {}

    stats_stream = None
    stats2_stream = None
    if params.outs.it_stats is not None:
        # One column set for the pair: the same flow, hence the same
        # ``get_stats`` and ``get_driving`` keys.
        stats_names = list(stats.keys()) + list(last_drive1.keys())
        stats_stream = _ScalarStream(
            "stats.dat", stats_names, jnp=jnp, sharding=sharding
        )
        stats_stream.push(_row(stats, last_drive1), t)
        stats2_stream = _ScalarStream(
            "stats_twin.dat", stats_names, jnp=jnp, sharding=sharding
        )
        stats2_stream.push(_row(stats2, last_drive2), t)

    twin_stream = _ScalarStream(
        "twin.dat", tvals.keys(), jnp=jnp, sharding=sharding
    )
    twin_stream.push(jnp.stack(list(tvals.values())), t)

    budget_stream = None
    if measure_budget:
        budget_stream = _ScalarStream(
            "twin_budget.dat", bvals.keys(), jnp=jnp, sharding=sharding
        )
        budget_stream.push(jnp.stack(list(bvals.values())), t)

    # --- Spectra stream (probes-style binary; t0 sample here, in-loop
    # samples before each step, a final sample only when
    # cadence-aligned -- uniform sample times for the reader).  The
    # gate below is ``it % cadence``, not ``(it - it0) % cadence``, so
    # a resume whose ``it0`` is not a multiple of the cadence leaves
    # one short gap at the seam; the readers key on ``t`` and
    # ``analysis.twin.series.uniform_grid`` re-derives the grid, so
    # this costs an off-grid row rather than a wrong one.  Changing the
    # gate would move the sample times of every existing member's
    # resume, which is why it is documented instead. --------------------
    measure_spectra: bool = twin_params.it_spectra is not None
    spectra_bad_t0: str | None = None
    if measure_spectra:
        from .spectra import TwinSpectraStream

        # ``spectra_ref`` is static in the diagnostic (it decides
        # whether the reference reduction is traced at all), so it is
        # bound here once rather than passed per sample.
        _ref = twin_params.spectra_ref

        def twin_spectra_2d(s1, s2):
            return diagnostics.twin_spectra_2d(s1, s2, ref=_ref)

        spectra_stream = TwinSpectraStream(twin_params)
        spectra_bad_t0 = spectra_stream.record(
            twin_spectra_2d(state1, state2), t
        )

    # --- Wall-normal-resolved streams (same discipline) ------------------
    yspectra_bad_t0: str | None = None
    ybudget_bad_t0: str | None = None
    if measure_yspectra or measure_ybudget:
        from .yspectra import TwinYBudgetStream, TwinYSpectraStream

        y_weights = np.asarray(diagnostics.flow.y_weights).tolist()
    if measure_yspectra:
        yspectra_stream = TwinYSpectraStream(twin_params, y_weights)
        yspectra_bad_t0 = yspectra_stream.record(yvals, t)
    if measure_ybudget:
        ybudget_stream = TwinYBudgetStream(twin_params, y_weights)
        ybudget_bad_t0 = ybudget_stream.record(ybvals, t)

    # --- CN/AB2 history priming (both states; donated args copied) -------
    scheme: str = params.step.scheme
    is_cnab2: bool = scheme == "cnab2"
    if is_cnab2:
        _, rhs1, *_ = step_cnab2(jnp.copy(state1), jnp.zeros_like(state1))
        _, rhs2, *_ = step_cnab2(jnp.copy(state2), jnp.zeros_like(state2))

    # --- Steps (CFL) streams: both states --------------------------------
    measure_steps: bool = params.outs.it_steps is not None
    steps_stream = None
    steps2_stream = None
    if measure_steps:
        if is_cnab2:
            *_, meas = step_cnab2_measured(jnp.copy(state1), jnp.copy(rhs1))
        else:
            *_, meas = predict_and_fully_correct_measured(jnp.copy(state1))
        # ``state2`` no longer forces the unmeasured program to compile
        # in iteration 0: on a recording step both states take the
        # measured variant.  Pre-compile it when the first iteration
        # records but not every one does, exactly as
        # :func:`dnsjax.__main__.run` does (``step.adaptive`` is
        # rejected here, so ``outs.it_steps`` is the only consumer of
        # the measured program).  Only iterative-CN needs this: under
        # cnab2 both cnab2 programs are primed above, and iteration 0
        # self-starts with iterative-CN, which is outside the benchmark
        # window (``bench_start`` is set at ``it == it0 + 1``).
        if (
            not is_cnab2
            and params.outs.it_steps != 1
            and it % params.outs.it_steps == 0
        ):
            predict_and_fully_correct(jnp.copy(state1))
        steps_stream = _ScalarStream(
            "steps.dat", meas.keys(), jnp=jnp, sharding=sharding
        )
        steps2_stream = _ScalarStream(
            "steps_twin.dat", meas.keys(), jnp=jnp, sharding=sharding
        )

    # --- Corrector streams (both states) ---------------------------------
    measure_corrector: bool = params.outs.it_corrector is not None
    corr_stream = None
    corr2_stream = None
    if measure_corrector:
        corr_stream = _ScalarStream(
            "corrector.dat", ["c", "error"], jnp=jnp, sharding=sharding
        )
        corr2_stream = _ScalarStream(
            "corrector_twin.dat", ["c", "error"], jnp=jnp, sharding=sharding
        )

    # --- Probe stream (reference state) ----------------------------------
    measure_probes: bool = probes_params.modes is not None
    probe_bad_t0: str | None = None
    if measure_probes:
        from ..extensions.probes import ProbeStream

        probe_stream = ProbeStream(state1)
        probe_bad_t0 = probe_stream.record(state1, t)

    def flush_all_buffers(check: bool = True) -> None:
        """Flush every stream (the ``__main__`` flush contract)."""
        for stream in (
            stats_stream,
            stats2_stream,
            twin_stream,
            budget_stream,
            steps_stream,
            steps2_stream,
            corr_stream,
            corr2_stream,
        ):
            if stream is not None:
                bad = stream.flush(check=check)
                if bad is not None:
                    _abort_non_finite(bad)
        if measure_probes:
            bad = probe_stream.flush(check=check)
            if bad is not None:
                _abort_non_finite(bad)
        if measure_spectra:
            bad = spectra_stream.flush(check=check)
            if bad is not None:
                _abort_non_finite(bad)
        if measure_yspectra:
            bad = yspectra_stream.flush(check=check)
            if bad is not None:
                _abort_non_finite(bad)
        if measure_ybudget:
            bad = ybudget_stream.flush(check=check)
            if bad is not None:
                _abort_non_finite(bad)

    def _abort_non_finite(reason: str) -> None:
        """FATAL / flush-unchecked / exit-3 (the ``__main__`` path)."""
        sharding.print(f"FATAL: {reason}; aborting.")
        flush_all_buffers(check=False)
        sys.exit(3)

    _terminating: bool = False

    def _flush_and_exit(signum: int, frame: object) -> None:
        nonlocal _terminating
        if _terminating:
            return
        _terminating = True
        sharding.print(
            f"Received signal {signum}; flushing buffers and exiting."
        )
        flush_all_buffers(check=False)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _flush_and_exit)
    signal.signal(signal.SIGINT, _flush_and_exit)

    if probe_bad_t0 is not None:
        _abort_non_finite(probe_bad_t0)
    if spectra_bad_t0 is not None:
        _abort_non_finite(spectra_bad_t0)
    if yspectra_bad_t0 is not None:
        _abort_non_finite(yspectra_bad_t0)
    if ybudget_bad_t0 is not None:
        _abort_non_finite(ybudget_bad_t0)

    sharding.print("Started twin timestepping at", datetime.now())

    # --- Main twin loop --------------------------------------------------
    while (
        t < t_stop
        and wall_time_now - wall_time_start < wall_time_stop
        and last_error < params.step.corrector_tolerance
        and not laminarized
    ):
        if it == it0 + 1:
            jax.block_until_ready(state1)
            jax.block_until_ready(state2)
            flush_all_buffers()
            bench_start = perf_counter_ns()
            sharding.print("First iteration over at", datetime.now())

        do_stats = (
            params.outs.it_stats is not None
            and it % params.outs.it_stats == 0
            and it > it0
        )
        do_twin = it % twin_params.it_energy == 0 and it > it0
        do_budget = (
            measure_budget and it % twin_params.it_budget == 0 and it > it0
        )
        do_snapshot = (
            params.outs.it_snapshot is not None
            and it % params.outs.it_snapshot == 0
            and it > it0
        )

        # All cross-state and reference diagnostics are issued before
        # the two step calls: the steppers donate their inputs, and
        # async dispatch orders these reads before the donation.
        if do_stats or do_snapshot:
            stats = get_stats(state1)
            stats2 = get_stats(state2)
        if do_stats:
            for stream, s, drive in (
                (stats_stream, stats, last_drive1),
                (stats2_stream, stats2, last_drive2),
            ):
                bad = stream.push(_row(s, drive), t)
                if bad is not None:
                    _abort_non_finite(bad)
        if do_twin:
            tvals = twin_energies(state1, state2, bins=twin_params.bins)
            bad = twin_stream.push(jnp.stack(list(tvals.values())), t)
            if bad is not None:
                _abort_non_finite(bad)
        if do_budget:
            bvals = twin_budget(state1, state2)
            bad = budget_stream.push(jnp.stack(list(bvals.values())), t)
            if bad is not None:
                _abort_non_finite(bad)
        if measure_spectra and it % twin_params.it_spectra == 0 and it > it0:
            bad = spectra_stream.record(twin_spectra_2d(state1, state2), t)
            if bad is not None:
                _abort_non_finite(bad)
        if measure_yspectra and it % twin_params.it_yspectra == 0 and it > it0:
            bad = yspectra_stream.record(twin_yspectra(state1, state2), t)
            if bad is not None:
                _abort_non_finite(bad)
        if measure_ybudget and it % twin_params.it_ybudget == 0 and it > it0:
            bad = ybudget_stream.record(twin_ybudget(state1, state2), t)
            if bad is not None:
                _abort_non_finite(bad)
        if measure_probes and it % probes_params.it_probes == 0 and it > it0:
            bad = probe_stream.record(state1, t)
            if bad is not None:
                _abort_non_finite(bad)
        if do_snapshot:
            flush_all_buffers()
            isnap = _save_pair(state1, state2, t, it, stats, stats2, isnap)
            last_saved_it = it

        # The two steps: the same jitted stepper twice, each donating
        # its own buffers.  CN/AB2 self-starts with one iterative-CN
        # step (the __main__ convention).  Both states take the *same*
        # variant in the same iteration -- measured on a recording step
        # (``steps.dat`` / ``steps_twin.dat``), plain otherwise -- so
        # the loop never runs different programs on the pair.
        do_record = measure_steps and it % params.outs.it_steps == 0
        if is_cnab2 and it > it0:
            if do_record:
                (
                    state1,
                    rhs1,
                    e1_dev,
                    c1_dev,
                    drive1,
                    meas,
                ) = step_cnab2_measured(state1, rhs1)
                (
                    state2,
                    rhs2,
                    e2_dev,
                    c2_dev,
                    drive2,
                    meas2,
                ) = step_cnab2_measured(state2, rhs2)
            else:
                state1, rhs1, e1_dev, c1_dev, drive1 = step_cnab2(state1, rhs1)
                state2, rhs2, e2_dev, c2_dev, drive2 = step_cnab2(state2, rhs2)
        else:
            if do_record:
                (
                    state1,
                    e1_dev,
                    c1_dev,
                    drive1,
                    meas,
                ) = predict_and_fully_correct_measured(state1)
                (
                    state2,
                    e2_dev,
                    c2_dev,
                    drive2,
                    meas2,
                ) = predict_and_fully_correct_measured(state2)
            else:
                state1, e1_dev, c1_dev, drive1 = predict_and_fully_correct(
                    state1
                )
                state2, e2_dev, c2_dev, drive2 = predict_and_fully_correct(
                    state2
                )
        # Each state's own applied driving belongs to the step just
        # taken, so it is bound here and consumed by the *next*
        # iteration's ``stats.dat`` / ``stats_twin.dat`` row.
        last_drive1 = drive1
        last_drive2 = drive2

        if do_record:
            for stream, m in ((steps_stream, meas), (steps2_stream, meas2)):
                bad = stream.push(jnp.stack(list(m.values())), t)
                if bad is not None:
                    _abort_non_finite(bad)

        if measure_corrector and it % params.outs.it_corrector == 0:
            for stream, c_dev, e_dev in (
                (corr_stream, c1_dev, e1_dev),
                (corr2_stream, c2_dev, e2_dev),
            ):
                bad = stream.push(
                    jnp.stack(
                        [
                            c_dev.astype(sharding.float_type),
                            e_dev.astype(sharding.float_type),
                        ]
                    ),
                    t,
                )
                if bad is not None:
                    _abort_non_finite(bad)

        t += params.step.dt
        it += 1

        c_sum = c_sum + c1_dev + c2_dev

        if (it - it0) % it_error_check == 0:
            # Periodic host sync: both states' corrector errors feed
            # the convergence stop and the non-finite guard.
            err1 = float(e1_dev)
            err2 = float(e2_dev)
            for label, err in (("reference", err1), ("perturbed", err2)):
                if not math.isfinite(err):
                    _abort_non_finite(
                        f"non-finite corrector error ({err}) in the "
                        f"{label} state at t = {t:.6e}, it = {it}"
                    )
            last_error = max(err1, err2)

            if check_laminarization:
                e_prime_host = float(get_perturbation_energy(state1))
                if not math.isfinite(e_prime_host):
                    _abort_non_finite(
                        f"non-finite perturbation energy E' "
                        f"({e_prime_host}) at t = {t:.6e}, it = {it}"
                    )
                laminarized = e_prime_host < laminarization_threshold

        wall_time_now = perf_counter_ns()

    # --- Post-processing -------------------------------------------------
    n_steps: int = it - it0
    if n_steps > 0:
        last_error = max(float(e1_dev), float(e2_dev))
        c_tot = int(c_sum)
    else:
        c_tot = 0

    if last_error > params.step.corrector_tolerance:
        sharding.print(
            f"Corrector failed to converge at t={t}, it={it}, "
            f"with error = {last_error:.3e}."
        )
    if laminarized:
        sharding.print(
            f"Laminarized: E' = {e_prime_host:.3e} < "
            f"{laminarization_threshold:.3e} at t={t}, it={it}."
        )

    sharding.print("Stopped twin timestepping at", datetime.now())

    # Final rows: stats (guarded) and the final twin energies of the
    # post-step pair; both appended regardless of cadence alignment
    # (the t column carries the timestamp), like the stats stream.
    if it > it0:
        stats = get_stats(state1)
        stats2 = get_stats(state2)
        tvals = twin_energies(state1, state2, bins=twin_params.bins)
        if measure_budget:
            bvals = twin_budget(state1, state2)
        bad_final = (
            [k for k, v in stats.items() if not math.isfinite(float(v))]
            + [
                f"{k} (twin)"
                for k, v in stats2.items()
                if not math.isfinite(float(v))
            ]
            + [k for k, v in tvals.items() if not math.isfinite(float(v))]
        )
        if measure_budget:
            bad_final += [
                k for k, v in bvals.items() if not math.isfinite(float(v))
            ]
        if bad_final:
            _abort_non_finite(
                f"non-finite final statistic(s) {', '.join(bad_final)} "
                f"at t = {t:.6e}, it = {it}"
            )
        if stats_stream is not None:
            stats_stream.push(_row(stats, last_drive1), t)
            stats2_stream.push(_row(stats2, last_drive2), t)
        twin_stream.push(jnp.stack(list(tvals.values())), t)
        if measure_budget:
            budget_stream.push(jnp.stack(list(bvals.values())), t)

    if measure_probes and it > it0 and it % probes_params.it_probes == 0:
        bad = probe_stream.record(state1, t)
        if bad is not None:
            _abort_non_finite(bad)

    if measure_spectra and it > it0 and it % twin_params.it_spectra == 0:
        bad = spectra_stream.record(twin_spectra_2d(state1, state2), t)
        if bad is not None:
            _abort_non_finite(bad)

    if measure_yspectra and it > it0 and it % twin_params.it_yspectra == 0:
        bad = yspectra_stream.record(twin_yspectra(state1, state2), t)
        if bad is not None:
            _abort_non_finite(bad)

    if measure_ybudget and it > it0 and it % twin_params.it_ybudget == 0:
        bad = ybudget_stream.record(twin_ybudget(state1, state2), t)
        if bad is not None:
            _abort_non_finite(bad)

    if params.outs.snapshot_save_final and it > it0 and it != last_saved_it:
        flush_all_buffers()
        isnap = _save_pair(state1, state2, t, it, stats, stats2, isnap)
        last_saved_it = it

    wall_time_now = perf_counter_ns()
    alive_time = ns_to_s * (wall_time_now - wall_time_start)
    sharding.print(f"Job has been alive for {alive_time:.2f}s.")
    if it > it0 + 1:
        wall_time = ns_to_s * (wall_time_now - bench_start)
        wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
        c_per_it = c_tot / (2 * n_steps)
        sharding.print(
            f"t = {t:.2f}",
            f"E_d={float(tvals['E_d']):.3e}",
            f"E_ref={float(tvals['E_ref']):.3e}",
            f"c/it/state = {c_per_it:.2f}",
            f"err = {last_error:.3e}",
        )
        sharding.print(
            f"Ran for {wall_time:.2f}s with {sharding.n_devices} "
            f"device(s): {wall_time_per_sim_time:.3e} s/t "
            f"(twin pair, 2x steps per t).",
        )

    flush_all_buffers()


def _twin_sidecar_stub() -> dict:
    r"""The configuration half of ``twin.json``.

    A **superset** of :data:`_TWIN_MATCH_KEYS`: the resume check reads
    only those, and the last three entries below are recorded for
    provenance alone.  They shape a *stream* rather than the
    trajectory, and each is already pinned where it matters -- the
    budget form by ``terms`` in ``twin_ybudget.json``, the reference
    half by ``includes_ref`` in the two spectra sidecars, and the
    `$k_x = 0$` plane by ``suffixes`` in both -- so a mid-stream flip
    is a hard error from
    :class:`~dnsjax.twin._binstream.BinStream` either way.  Matching
    them here would additionally refuse the harmless case where the
    stream is off, while *not* recording them leaves ``twin.json``
    unable to say which budget form a member ran in (which
    ``scripts/twin_postprocess.py`` has to work around).  Additive
    keys only, so no :data:`TWIN_FORMAT_VERSION` bump and no
    :data:`_TWIN_LEGACY_DEFAULTS` entry: nothing reads them for
    control flow, and ``analysis.twin.series`` reads ``twin.json``
    with ``.get``.
    """
    return {
        "format_version": TWIN_FORMAT_VERSION,
        "system": params.phys.system,
        "e0": twin_params.e0,
        "seed": twin_params.seed,
        "smoothness": twin_params.smoothness,
        "mean_flow": twin_params.mean_flow,
        "bins": twin_params.bins,
        "it_energy": twin_params.it_energy,
        "it_budget": twin_params.it_budget,
        "it_spectra": twin_params.it_spectra,
        "it_yspectra": twin_params.it_yspectra,
        "it_ybudget": twin_params.it_ybudget,
        "dt": params.step.dt,
        "double_precision": params.res.double_precision,
        # Recorded, not matched -- see the docstring.
        "rotational_ybudget": twin_params.rotational_ybudget,
        "spectra_ref": twin_params.spectra_ref,
        "x0_planes": twin_params.x0_planes,
    }


def main(argv: list[str] | None = None) -> int:
    """``dnsjax-twin`` console-script entry point.

    The :func:`dnsjax.__main__.main` phases with the twin surface:
    resolve the parameter layers (the ``[twin]`` section registers at
    this module's import and rides the shared per-flow surface),
    configure the distributed JAX runtime, print the banner on the
    main process, run the twin loop.
    """
    wall_time_start = perf_counter_ns()

    setup = resolve_parameters(argv, prog=_PROG)
    print("Alive at", datetime.now(), flush=True, file=sys.stderr)
    main_device = configure_jax_runtime()

    if main_device:
        print("Distribution initialized at", datetime.now(), flush=True)
        print("Code version:", git_hash(), flush=True)
        if setup.snapshot_params_used:
            # The parent snapshot's own provenance (``_metadata_bytes``
            # always records it; the format-6 floor rejects anything
            # older), as in :func:`dnsjax.__main__.main`.
            print(
                "Snapshot was recorded by code version:",
                read_snapshot_meta(setup.snapshot_path)["git_hash"],
                flush=True,
            )
            print(
                f"Inherited parameters embedded in snapshot "
                f"'{setup.snapshot_path}' (except np0/np1/platform/"
                "double_precision); parameters.toml and command-line "
                "arguments override them.",
                flush=True,
            )
        if os.environ.get("DNSJAX_QUIET_STARTUP") != "1":
            print_resolved_parameters(
                params,
                setup.spec,
                tuple(relevant_extensions(setup.system).values()),
            )
        print(
            "Running with the physical-space (x, y, z) resolution:",
            padded_res.nx_padded,
            padded_res.ny_padded
            if padded_res.ny_padded is not None
            else params.res.ny,
            padded_res.nz_padded,
            flush=True,
        )

    run(wall_time_start, setup.seed_layers.get("twin.seed"))

    print("Shutdown at", datetime.now(), flush=True, file=sys.stderr)
    return 0
