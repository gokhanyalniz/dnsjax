#!/usr/bin/env python3
"""Entry point for the dnsjax DNS solver.

Import order
------------
JAX platform and distributed backend must be configured
*before* importing any module that reads ``sharding`` or a
geometry module, because those modules instantiate global
singletons (``params``, ``derived_params``, ``sharding``,
``fourier``) at import time.  This module enforces the
constraint by keeping all setup in :func:`main` (via
:mod:`dnsjax.bootstrap`) and deferring ``import jax`` and all
flow-module imports until after CLI / TOML configuration is
applied.

Execution phases
----------------
1. **Initialisation** (:func:`main`, via :mod:`dnsjax.bootstrap`):
   parse CLI arguments (``--help`` exits here, side-effect
   free), apply the configuration layers, configure the
   distributed JAX runtime, print the final parameter set.
   :func:`main` is both the ``dnsjax`` console-script target
   and the ``python -m dnsjax`` guard target.

2. **Main loop** (:func:`run`):
   initialise velocity (a provided snapshot wins; otherwise an
   in-process random / localized-rolls / laminar IC, with random
   the default), then iterate:

   - Fused predictor + corrector loop
     (:func:`predict_and_fully_correct`); for triply-periodic
     flows the post-step divergence correction + mean-mode
     zeroing is fused into the step itself (``finalize_fn`` in
     :mod:`dnsjax.timestep`)
   - Periodic diagnostic output (:func:`get_stats`)

   The loop terminates when the simulation time, wall-clock
   time, or corrector divergence criterion is reached.  The
   corrector error and iteration counters stay on the device;
   the error is synced to the host only every
   ``outs.it_error_check`` steps so that JAX async dispatch can
   pipeline steps (divergence is detected at most
   ``it_error_check`` steps late).

Diagnostics (``stats.dat``, ``steps.dat``, ``corrector.dat``,
``probes.bin``, ``forcing.bin``)
-------------------------------------------------------------
``get_stats`` output is accumulated on-device in a fixed
``(nbuffer, n_cols)`` buffer (one row every ``it_stats``
steps) and flushed to ``stats.dat`` when the buffer fills, at
shutdown, after the first (JIT-heavy) step, *before* every
snapshot write (so a buffered non-finite diagnostic aborts
before a snapshot of the same broken state is written, and the
``.dat`` files stay consistent with each snapshot), and on a
termination signal
(``flush_all_buffers``, which calls the shared
``_flush_stats``).  Buffering avoids a
host-device sync per sample; each flush is then ``fsync``-ed,
so the rows are on disk immediately once the on-device buffer
is flushed.  ``stats.dat`` (written by the main device,
appended) has a header row of column names (``t`` plus the
``get_stats`` keys) followed by whitespace-aligned rows at
``stats_precision`` significant digits.  The header is
``#``-commented (:func:`_write_dat_header`), so ``numpy.loadtxt``
reads a stream with no extra flags.

``steps.dat`` records the CFL diagnostic every ``it_steps``
steps with the same buffering and file format.  Each row is
measured from the pre-step state `$u^n$` inside the step's
first nonlinear-term evaluation
(:func:`predict_and_fully_correct_measured`; no extra Fourier
transforms, see :mod:`dnsjax.measurements`), so its timestamp
is the time *before* that step.  Column names come from the
measurement dict keys of a warm-up call, which also compiles
the measured program outside the benchmark window (its
outputs are discarded).  Unlike stats, no extra row is
recorded after the final step.  The ``dt`` column always
records the step's live time step.

With ``step.adaptive`` the measured stepper additionally runs
every ``step.cfl_cadence`` steps and the loop host-syncs
``meas["CFL"]`` there: an accepted
:func:`dnsjax.adaptive.propose_dt` proposal switches the live
``dt`` -- the flow module's ``set_dt`` rebuilds the
``dt``-dependent operator/IMM pytree leaves on device (no
stepper recompilation), ``params.step.dt`` is mutated (time
accounting, snapshot embedding, and resume all read it), one
``[adaptive] ...`` line is printed, and the next CN/AB2 step
runs with the ratio-weighted AB2 history
(``reset_ab2_kappa`` after exactly one step).  Semantics and
knobs: the ``TimeStepping`` docstring.

``corrector.dat`` records the corrector diagnostic every
``it_corrector`` steps (same buffering and file format): the
corrector iteration count ``c`` and the final corrector error
``error``, both already returned by every step, timestamped at
the pre-step time.  ``outs.it_error_check`` must not exceed
``it_corrector`` (:func:`dnsjax.parameters.validate_parameters`).

``probes.bin`` records the complex wall-normal profiles of the
``probes.modes`` spectral modes every ``probes.it_probes`` steps
(the ``probes`` extension section; :mod:`dnsjax.extensions`;
wall-bounded only) with the same buffering, but as fixed-size
binary records with a ``probes.json`` schema sidecar -- see the
:mod:`dnsjax.extensions.probes` module docstring for the format, the
append-on-resume rules, and the
:mod:`dnsjax.analysis.response.probes` reader.  The t0 sample is
recorded at setup, in-loop samples before each step, and a final
post-step sample only when cadence-aligned (uniform sample
times).

``forcing.bin`` records the coefficients of the stochastic mode
kicks (the ``force`` extension section) every ``force.it_force``
steps, with a ``forcing.json`` sidecar.  A kick fires at the top of the loop
after the equal-``t`` probe sample and any snapshot (both
pre-kick) and immediately before the step; format, resume
semantics, and the kick construction: the :mod:`dnsjax.extensions.forcing`
module docstring (reader: :mod:`dnsjax.analysis.response.ssi`).

Every diagnostic is guarded against non-finite floats: each
flushed buffer row (stats / steps / corrector) is scanned on the
host, and the host-synced scalars -- the corrector error and
``E'`` at the ``it_error_check`` cadence, plus the initial and
final stats -- are checked directly.  A NaN or inf prints one
``FATAL: non-finite ...`` line naming the quantity, flushes all
streams unchecked (the offending rows are already on disk for
post-mortem), skips the final snapshot (the state is non-finite;
the last snapshot on disk is the post-mortem artifact), and exits
with code **3**.  Detection lags the device by at most the flush /
``it_error_check`` cadence, preserving async dispatch.

Snapshots and resume
--------------------
Snapshots are named ``state{isnap}.tar`` (``isnap`` zero-padded to
``outs.snapshot_pad_width``), where ``isnap`` is a per-run counter
(start ``init.isnap0``) bumped on every write by
:func:`_save_numbered_snapshot`.  By default the IC is saved as
``state00000.tar`` for any non-continuation start and the final
state is saved on termination (both independent of
``outs.it_snapshot``, deduped against a periodic write at the same
``it``); the periodic save runs at the **top** of the loop so it
shares the ``it_stats`` computation.  Each snapshot embeds the
state's ``get_stats`` as ``_dnsjax_stats.json``
(``outs.snapshot_embed_stats``).  Every snapshot write also
flushes the buffered stats / steps / corrector rows to their
``.dat`` files (``flush_all_buffers``), keeping them consistent
with the snapshot.

When ``init.snapshot`` points at a snapshot tar file (an
uncompressed tar wrapping a zarr3 store; see
:mod:`dnsjax.snapshot`), the parameters embedded in its metadata
are merged in as a configuration layer above the code defaults but
below ``parameters.toml`` and the CLI (``read_snapshot_params``;
the JAX-setup fields ``np0``/``np1``/``platform``/
``double_precision`` are not inherited).  The resume is a
*continuation* (inherit ``t`` / ``it`` / ``isnap`` from the
snapshot, do not re-save the IC) only when
:func:`dnsjax.parameters.trajectory_defining_changes` is empty -- no
Physics/Geometry/Resolution parameter was overridden to a value
different from the snapshot's.  Any such change starts a NEW
trajectory (``t = it = isnap = 0``, IC re-saved as
``state00000.tar``) unless ``init.force_resume`` is set.  When
the current wall-normal grid differs from the snapshot's,
``_interpolate_if_needed`` interpolates the state at load time
(see :mod:`dnsjax.fd` for the interpolation methods).

Benchmarking
------------
The first time step is excluded from wall-clock statistics
because it includes JAX's JIT compilation overhead.
"""

import math
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter_ns

from .adaptive import propose_dt
from .bootstrap import configure_jax_runtime, resolve_parameters
from .extensions import force_params, probes_params, relevant_extensions
from .flows.registry import (
    annular_systems,
    cylindrical_systems,
    walled_systems,
)
from .param_surface import print_resolved_parameters
from .parameters import (
    derived_params,
    ns_to_s,
    padded_res,
    params,
    trajectory_defining_changes,
)
from .snapshot_meta import git_hash, read_snapshot_meta

#: Axis-parity class of each **stored physical** state component, as
#: ``True`` for the `$(-1)^m$` (even) class and ``False`` for
#: `$(-1)^{m+1}$` (odd): the order is
#: ``(u_z, u_r, u_theta, c_zz, c_rz, c_theta_z, c_rr, c_theta_theta,
#: c_r_theta)``, truncated to the first 3 for a velocity-only flow.
#: Read by the pipe's parity-aware resume regrid; the annular
#: geometries have no axis and ignore it.
_PARITY_EVEN_STORED = (
    True,  # u_z
    False,  # u_r
    False,  # u_theta
    True,  # c_zz
    False,  # c_rz
    False,  # c_theta_z
    True,  # c_rr
    True,  # c_theta_theta
    True,  # c_r_theta
)


def _flush_stats(buffer, n_valid, ts_buf, file_path, p, col_width, names=None):
    """Write *n_valid* buffered rows to *file_path*, durably.

    The on-device ``buffer`` is the only batching layer: once it fills
    (``nbuffer`` rows) this transfers it to the host and appends the
    rows.  An explicit ``flush`` + ``fsync`` then forces the bytes out of
    the process and OS buffers so each device-buffer flush is immediately
    on disk (and visible to other clients on networked filesystems).
    Shared by every measurement stream (stats, steps, corrector).

    When *names* (the stream's column names) is given, the flushed
    rows are also scanned for non-finite values -- after the write, so
    the offending rows are on disk for post-mortem -- and a diagnostic
    message naming the first offender is returned (``None`` when all
    values are finite).  The caller aborts the run on it (see
    ``_abort_non_finite`` in :func:`run`).
    """
    import numpy as np

    data = np.asarray(buffer[:n_valid])
    with open(file_path, "a") as f:
        for i in range(n_valid):
            t_str = f"{ts_buf[i]:.{p}e}".rjust(col_width)
            stat_strs = " ".join(
                f"{v:.{p}e}".rjust(col_width) for v in data[i]
            )
            f.write(f"{t_str} {stat_strs}\n")
        f.flush()
        os.fsync(f.fileno())

    if names is not None:
        finite = np.isfinite(data)
        if not finite.all():
            i, j = (int(v) for v in np.argwhere(~finite)[0])
            return (
                f"non-finite diagnostic in {Path(file_path).name}: "
                f"{names[j]} = {data[i, j]} at t = {ts_buf[i]:.{p}e}"
            )
    return None


def _stats_row(stats, driving):
    """One ``stats.dat`` row: the sorted stats, then the driving.

    Both dicts come back from ``jit`` in sorted key order, and the
    driving is appended rather than merged so it stays the **last**
    column whatever the stats keys are named.
    """
    import jax.numpy as jnp

    return jnp.stack(
        [jnp.asarray(v) for v in stats.values()]
        + [jnp.asarray(v) for v in driving.values()]
    )


def _write_dat_header(file_path, columns, col_width) -> None:
    """Create *file_path* with the ``#``-commented column header.

    The ``#`` replaces one leading space of the first column's
    padding, so the header stays aligned with the rows
    :func:`_flush_stats` writes while ``numpy.loadtxt`` skips it as a
    comment (its default ``comments="#"``) -- a ``.dat`` stream loads
    with no extra flags.  Shared by every measurement stream, here
    and in the twin driver.
    """
    padded = [
        n.rjust(col_width - 1 if i == 0 else col_width)
        for i, n in enumerate(columns)
    ]
    with open(file_path, "w") as f:
        f.write("#" + " ".join(padded) + "\n")


def _interpolate_if_needed(state, snap_path, read_metadata, sharding, jnp):
    r"""Interpolate snapshot state if the wall-normal grid changed.

    Compares the snapshot's wall-normal grid against the current
    grid (from ``derived_params``).  When they differ -- either
    in number of points or in point locations -- applies the
    optimal interpolation.

    After interpolation, the first corrector iteration projects
    out any `$O(\varepsilon)$` divergence introduced by the
    changed `$\partial/\partial y$` operator.
    """
    import numpy as np

    from .fd import build_interpolation_matrix
    from .flows.registry import stored_value

    meta = read_metadata(Path(snap_path))
    snap_grid = meta.get("wall_normal_grid")
    # Stored params use public names (res.ny is "nr" for the
    # cylindrical/annular flows); look it up via the alias.
    snap_ny = stored_value(meta.get("params", {}), meta["system"], "res", "ny")
    curr_grid = derived_params.wall_normal_grid

    needs_interp = snap_ny != params.res.ny or not np.allclose(
        snap_grid, curr_grid, atol=1e-12
    )

    if not needs_interp:
        return state

    curr_grid_np = np.array(curr_grid)
    old_grid = np.array(snap_grid)

    # Each geometry list includes its viscoelastic member, so the
    # 9-component flows regrid with their own geometry's operators.
    if params.phys.system in cylindrical_systems:
        geometry = "cylindrical"
    elif params.phys.system in annular_systems:
        geometry = "annular"
    else:
        geometry = "cartesian"
    T = build_interpolation_matrix(
        old_grid, curr_grid_np, geometry, params.res.fd_order
    )

    if isinstance(T, tuple):
        # Spectral parity-aware cylindrical interpolation: apply the
        # even / odd radial matrix per azimuthal mode m by component
        # parity.  A component's axis parity is `$(-1)^{m+s}$` with
        # `$s$` its spin weight, i.e. it is set by how many of its
        # indices are radial/azimuthal (each flips sign under the axis
        # reflection).  State layout (component, r, m, kz):
        #
        #   u_z, c_zz, c_rr, c_theta_theta, c_r_theta -> (-1)^m
        #   u_r, u_theta, c_rz, c_theta_z             -> (-1)^{m+1}
        #
        # The 6 tensor slots are present only for a viscoelastic pipe;
        # ``_PARITY_EVEN_STORED`` is indexed by the **stored physical**
        # component order and truncated to the state's own length.
        #
        # The mask is the geometry's own ``Fourier.m_is_even``, which
        # is the parity of the **physical** `$m = m_0 j$` (the wedge
        # factor matters: with an even ``geo.m0`` every physical mode
        # is even) and which spans the **padded** m axis -- the axis
        # the loaded state is actually laid out on.  Re-deriving it
        # here from ``complex_harmonics`` disagreed with both.
        from .geometries.wall_bounded.cylindrical import fourier

        n_comp = state.shape[0]
        if n_comp not in (3, len(_PARITY_EVEN_STORED)):
            raise ValueError(
                "parity-aware radial interpolation is defined for the "
                "3-component velocity state and the 9-component "
                f"viscoelastic state, got {n_comp} components; a flow "
                "with a different layout needs its own parity "
                "assignment here."
            )
        T_even, T_odd = T
        m_even = fourier.m_is_even.astype(bool)
        T_e = jnp.asarray(T_even, dtype=state.dtype)
        T_o = jnp.asarray(T_odd, dtype=state.dtype)
        # Contract with the two (i_new, j_old) matrices and select per
        # mode, rather than materialising an (m, i_new, j_old) stack
        # per component -- that stack is replicated on every device.
        state = jnp.stack(
            [
                jnp.where(
                    m_even,
                    jnp.einsum("ij, jmk -> imk", a_even, state[c]),
                    jnp.einsum("ij, jmk -> imk", a_odd, state[c]),
                )
                for c, (a_even, a_odd) in enumerate(
                    (T_e, T_o) if even else (T_o, T_e)
                    for even in _PARITY_EVEN_STORED[:n_comp]
                )
            ]
        )
    else:
        T_jax = jnp.asarray(T, dtype=state.dtype)
        # state: (3, ny_old, ...) -- the wall-normal axis is axis 1 in
        # every geometry's spectral layout.
        state = jnp.einsum("ij, cjzx -> cizx", T_jax, state)

    # Enforce wall boundary conditions on the *velocity* only (the first
    # 3 components).  A viscoelastic state carries 6 conformation-tensor
    # components after the velocity; their wall BC is ``div(grad c) = 0``
    # (handled by the Hc operator during stepping), not a Dirichlet zero,
    # so they must not be zeroed here.  For the 3-component systems
    # ``[:3]`` is the whole state.
    if geometry == "cylindrical":
        # Single wall at r = 1 (axis handled by parity).
        state = state.at[:3, -1].set(0.0)
    else:
        # Two walls (Cartesian: y = +/-1; annular: r = r1, r2).
        state = state.at[:3, 0].set(0.0)
        state = state.at[:3, -1].set(0.0)

    sharding.print(
        "Interpolated wall-normal grid; first corrector step "
        "will project out any residual divergence."
    )
    return state


def run(wall_time_start: int) -> None:
    """Run the time-stepping loop after parameters and JAX are initialised.

    *wall_time_start* is the ``perf_counter_ns`` timestamp taken at
    process start (:func:`main`) -- the reference for the
    ``stop.max_wall_time`` budget and the shutdown diagnostics.
    """
    import importlib

    import jax
    from jax import numpy as jnp

    from .flows.registry import spec_for
    from .sharding import sharding

    # --- Flow dispatch -------------------------------------------------------
    # Registry-driven: the flow spec names its runtime module
    # (``FlowSpec.flow_module``), so registering a spec is the whole
    # dispatch story -- no per-system branch here.

    _spec = spec_for(params.phys.system)
    if _spec.flow_module is None:
        sharding.print(
            f"System '{params.phys.system}' is not yet implemented."
        )
        sharding.exit(code=1)
    _flow_mod = importlib.import_module(_spec.flow_module)
    get_perturbation_energy = _flow_mod.get_perturbation_energy
    get_stats = _flow_mod.get_stats
    # Optional: the applied mean-mode driving, inferred from a state.
    # Only flows that can apply one export it (see ``flow_spec``); it
    # supplies both the extra ``stats.dat`` column names and the one row
    # that has no step behind it.
    get_driving = getattr(_flow_mod, "get_driving", None)
    init_state = _flow_mod.init_state
    predict_and_fully_correct = _flow_mod.predict_and_fully_correct
    predict_and_fully_correct_measured = (
        _flow_mod.predict_and_fully_correct_measured
    )
    step_cnab2 = _flow_mod.step_cnab2
    step_cnab2_measured = _flow_mod.step_cnab2_measured
    set_dt = _flow_mod.set_dt
    reset_ab2_kappa = _flow_mod.reset_ab2_kappa

    # --- Component-basis boundary --------------------------------------------
    # The cylindrical/annular solvers work in a decoupled basis
    # (``u_± = u_r ± i u_θ``, plus the conformation spin components)
    # that diagonalizes their implicit operators, while everything
    # outside the time stepper -- snapshots, diagnostics, probes,
    # initial conditions, the analysis package -- works in physical
    # components.  These two maps are that boundary; a given state
    # crosses it at most once and never crosses back (the physical
    # form is a *view* built for the consumers and dropped, so no
    # re-encode is needed).  Cartesian and triply-periodic flows have
    # no such basis, hence the identity fallback.
    def _identity_basis(state):
        return state

    # The fallback is correct for a flow with no solver basis, but it
    # is also indistinguishable from a flow that *has* one and forgot
    # to re-export the pair -- which would feed the stepper a physical
    # state with no error, only wrong answers.  The cylindrical and
    # annular geometries always carry one (their viscoelastic members
    # included -- the geometry lists span them), so require it there
    # and let the fallback serve the rest.
    _needs_basis = params.phys.system in (
        *cylindrical_systems,
        *annular_systems,
    )
    if _needs_basis:
        missing = [
            name
            for name in ("to_solver_basis", "from_solver_basis")
            if not hasattr(_flow_mod, name)
        ]
        if missing:
            raise RuntimeError(
                f"{_spec.flow_module} does not export "
                f"{' / '.join(missing)}; a cylindrical, annular or "
                "viscoelastic flow must re-export the pair (see the "
                "existing flow modules) or its state would enter the "
                "stepper in the wrong basis."
            )
    to_solver_basis = getattr(_flow_mod, "to_solver_basis", _identity_basis)
    from_solver_basis = getattr(
        _flow_mod, "from_solver_basis", _identity_basis
    )

    # Both out-crossings run in the hot loop -- the physical view on
    # every stats/snapshot step, the ``E'`` read every
    # ``outs.it_error_check`` (default 10) -- so the map is jitted.
    # Unfused it dispatches one field-sized eager primitive per
    # operation (measured ~5-6x the jitted cost, and as many
    # field-sized transients live alongside ``state``).  Fusing the
    # ``E'`` read with its conversion saves the intermediate outright.
    # The identity is deliberately left bare: jitting it would only
    # add a dispatch, and the loop relies on ``state_phys is state``
    # there for its release to actually free anything.
    _crosses_basis = from_solver_basis is not _identity_basis
    from_solver_basis_jit = (
        jax.jit(from_solver_basis) if _crosses_basis else from_solver_basis
    )
    if _crosses_basis:

        @jax.jit
        def _perturbation_energy_solver(s):
            return get_perturbation_energy(from_solver_basis(s))
    else:
        _perturbation_energy_solver = get_perturbation_energy

    # --- Initial condition ---------------------------------------------------
    from .snapshot_meta import is_snapshot_file

    # Start-mode precedence: a provided snapshot file wins over every
    # in-process mode; then start_from_laminar, then localized_rolls,
    # then random_field (the
    # default).  A *continuation* resume (dnsjax snapshot with unchanged
    # trajectory params) inherits t/it/isnap and does not re-save the IC;
    # every other start is a fresh trajectory: isnap begins at
    # init.isnap0 and the IC is saved as state00000.tar (see the IC-save
    # block below).
    resumed_continuation: bool = False
    isnap_start: int = params.init.isnap0

    if params.init.snapshot is not None and is_snapshot_file(
        params.init.snapshot
    ):
        # single-file (tar-wrapped zarr3) snapshot
        from .snapshot import (
            load_snapshot,
            read_metadata,
            validate_snapshot_params,
        )

        validate_snapshot_params(params.init.snapshot)
        state, t_snap, it_snap = load_snapshot(params.init.snapshot)
        meta = read_metadata(params.init.snapshot)

        # Continuation vs new trajectory: any override of a Physics /
        # Geometry / Resolution parameter to a value different from the
        # snapshot's starts a new trajectory (reset t/it/isnap) unless
        # init.force_resume is set.
        changes = trajectory_defining_changes(meta["params"])
        if changes and not params.init.force_resume:
            sharding.print(
                "Resume: trajectory-defining parameters changed; "
                "starting a NEW trajectory (t=it=isnap=0). Set "
                "init.force_resume=True to continue. Changes: "
                + "; ".join(changes)
            )
            params.init.t0 = 0.0
            params.init.it0 = 0
            isnap_start = 0
        else:
            if changes:
                sharding.print(
                    "Resume: force_resume set; continuing despite "
                    "changes (" + "; ".join(changes) + ")."
                )
            params.init.t0 = t_snap
            params.init.it0 = it_snap
            isnap_start = meta["isnap"] + 1
            resumed_continuation = True
        sharding.print(
            f"Resumed from snapshot: t={params.init.t0:.6e}, "
            f"it={params.init.it0}"
        )

        # --- Wall-normal grid interpolation ---
        if params.phys.system in walled_systems:
            state = _interpolate_if_needed(
                state,
                Path(params.init.snapshot),
                read_metadata,
                sharding,
                jnp,
            )
    elif params.init.snapshot is not None:
        # A path was given but it is not a dnsjax snapshot.  Refusing
        # here is deliberate: falling through to an in-process mode
        # would start a run that computes something the user never
        # asked for (a typo'd path is the common case).
        sharding.print(
            f"init.snapshot ({params.init.snapshot}) is not a dnsjax "
            "snapshot file: expected an uncompressed tar wrapping a "
            "zarr3 store."
        )
        sharding.exit(code=1)
    elif params.init.start_from_laminar:
        # Laminar / closed-form base state (snapshot is None here).
        state = init_state()
    elif params.init.localized_rolls:
        # In-process deterministic localized-rolls ("spot") IC (no
        # snapshot file). The flow dispatch above already built the
        # geometry ``fourier`` singleton this consumes.
        from .ic.localized_rolls import generate_localized_rolls

        state = generate_localized_rolls(
            params.init.localized_rolls_amplitude,
            params.init.localized_rolls_width,
            params.init.localized_rolls_wavelength,
        )
        sharding.print(
            "Started from an in-process localized-rolls IC: "
            f"amplitude={params.init.localized_rolls_amplitude}, "
            f"width={params.init.localized_rolls_width}, "
            f"wavelength={params.init.localized_rolls_wavelength}."
        )
    elif params.init.random_field:
        # In-process random divergence-free IC -- the default start mode
        # (no snapshot file). The flow dispatch above already built the
        # geometry ``fourier`` singleton this consumes.
        from .ic.random_field import generate_random_state

        state = generate_random_state(
            params.init.random_amplitude,
            params.init.random_smoothness,
            params.init.random_seed,
            params.init.random_mean_flow,
        )
        sharding.print(
            "Started from an in-process random IC: "
            f"amplitude={params.init.random_amplitude}, "
            f"smoothness={params.init.random_smoothness}, "
            f"seed={params.init.random_seed}."
        )
    else:
        # No snapshot and all in-process modes disabled.
        sharding.print(
            "Provide an initial condition: no snapshot given and all "
            "in-process init modes are disabled."
        )
        sharding.exit(code=1)

    # --- Stopping criteria ---------------------------------------------------
    wall_time_stop = (
        jnp.inf
        if params.stop.max_wall_time is None
        else int(params.stop.max_wall_time.total_seconds() / ns_to_s)
    )

    t_stop = (
        jnp.inf
        if params.stop.max_sim_time is None
        else params.stop.max_sim_time
    )

    it: int = params.init.it0
    t: float = params.init.t0

    # Snapshot lineage counter: the index of the *next* snapshot file
    # (state{isnap}.tar).  ``last_saved_it`` is the iteration of the most
    # recent write, used to avoid writing the final state twice.
    isnap: int = isnap_start
    last_saved_it: int | None = None

    def _save_numbered_snapshot(state, t, it, snap_stats, isnap):
        """Write state{isnap}.tar (stats embedded per outs.*), return the
        next isnap.  *state* is the physical view -- the on-disk basis
        is physical components (see the component-basis boundary)."""
        from .snapshot import save_snapshot

        width = params.outs.snapshot_pad_width
        save_snapshot(
            state,
            t,
            it,
            f"state{isnap:0{width}d}.tar",
            stats=(snap_stats if params.outs.snapshot_embed_stats else None),
            isnap=isnap,
        )
        return isnap + 1

    dt_first: float = params.step.dt
    wall_time_now: int = perf_counter_ns()
    last_error: float = 0.0

    # Adaptive CFL controller (``step.adaptive``): the loop reads the
    # measured total CFL every ``cfl_cadence`` steps, asks
    # ``propose_dt`` for the next step, and on an accepted change
    # rebuilds the dt-dependent flow leaves on device (``set_dt``) and
    # mutates the live ``params.step.dt`` (time accounting, snapshot
    # embedding, and resume all read it).  ``kappa_pending`` tracks
    # the one step that runs with the AB2 ratio ``dt_new/dt_old``
    # before ``reset_ab2_kappa`` restores 1 (see ``TimeStepping``).
    adaptive: bool = params.step.adaptive
    cfl_cadence: int = params.step.cfl_cadence
    kappa_pending: bool = False

    # Laminarization (relaminarization) termination: stop once the
    # perturbation kinetic energy E' falls below the threshold.  E' is
    # read on the host at the it_error_check cadence (below), so the
    # flag lags the device by up to that many steps -- the same
    # semantics as the corrector-divergence stop.
    check_laminarization: bool = params.stop.check_laminarization
    laminarization_threshold: float = params.stop.laminarization_threshold
    laminarized: bool = False
    e_prime_host: float = float("inf")

    # Corrector counters stay on the device and accumulate lazily;
    # they are synced to the host only every ``it_error_check``
    # steps (error) or at shutdown (totals), so the host can keep
    # enqueueing steps ahead of the device (JAX async dispatch).
    it_error_check: int = params.outs.it_error_check
    c_sum = jnp.zeros((), dtype=jnp.int32)
    c_first = None
    error_dev = None
    c_dev = None

    # Warm-up call so that JIT compilation does not affect benchmarks
    stats = get_stats(state)

    sharding.print(
        f"t = {t:.2f}",
        *[f"{x}={y:.3e}" for x, y in stats.items()],
    )

    bad_init = [k for k, v in stats.items() if not math.isfinite(float(v))]
    if bad_init:
        # A broken initial condition; nothing is buffered yet, so
        # print and exit directly (the in-run guard proper is
        # ``_abort_non_finite`` below).
        sharding.print(
            f"FATAL: non-finite initial statistic(s) "
            f"{', '.join(bad_init)} at t = {t:.6e}; aborting."
        )
        sys.exit(3)

    if check_laminarization:
        # Compile the E' kernel outside the benchmark window; it is
        # read at the it_error_check cadence in the main loop.  The
        # loop reads it through ``_perturbation_energy_solver`` on a
        # *solver-basis* state, so warm that fused kernel -- warming
        # the bare physical one here would leave the conversion half
        # of it to compile inside the timed region.  ``state`` is
        # still physical at this point, so the warm-up value is
        # meaningless and is discarded; only the trace matters.
        jax.block_until_ready(_perturbation_energy_solver(state))

    # Save the initial condition (state00000.tar) unless this run is a
    # continuation of a dnsjax snapshot trajectory (whose IC is already
    # on disk as that snapshot).  ``stats`` here is the IC's stats.
    if params.outs.snapshot_save_initial and not resumed_continuation:
        isnap = _save_numbered_snapshot(state, t, it, stats, isnap)
        last_saved_it = it

    # Applied mean-mode driving (``constant_bulk_velocity`` /
    # ``block_mean_spanwise_velocity``): a *step* quantity, not a state
    # one -- the converged body force the corrector applied, threaded
    # out of the implicit solve because it is not recoverable from the
    # accepted state (its bulk is zero by construction).  It is appended
    # **after** the sorted ``get_stats`` keys, so it is the last
    # column(s), and the row at time ``t`` carries the value applied by
    # the step that *produced* that state.  The ``t = t0`` row has no
    # such step, so it carries the wall-shear inference of the same
    # quantity instead (``get_driving``; the one inferred entry in the
    # column, and exact in the same limit the two agree).
    #
    # Read here, on the **physical** side of the basis boundary below,
    # because that is ``get_driving``'s contract -- the same one
    # ``get_stats`` has.  Every current ``mean_driving`` happens to
    # touch only the axial/streamwise component, which ``to_pm_basis``
    # leaves alone, so the crossing would not show; a key reading
    # `$u_r$` or `$u_\theta$` would read `$u_\pm$` instead, silently.
    driving = get_driving(state) if get_driving is not None else {}
    driving_names = list(driving.keys())
    last_driving = dict(driving)

    # --- Into the solver -----------------------------------------------------
    # Every start mode above (snapshot resume + regrid, ``init_state``,
    # localized rolls, random field) builds the state in physical
    # components, and so did the diagnostics, the IC snapshot and the
    # driving read just taken from it.  This is where it enters the
    # solver, once (see the component-basis boundary above); from here
    # on ``state`` is in the solver basis until a consumer asks for the
    # physical view.
    state = to_solver_basis(state)

    # --- Stats buffer setup ------------------------------------------------
    p = params.outs.stats_precision - 1
    val_width = params.outs.stats_precision + 7

    if params.outs.it_stats is not None:
        stat_names = list(stats.keys()) + driving_names
        n_stat_cols = len(stat_names)
        buffer = jnp.zeros(
            (params.outs.nbuffer, n_stat_cols),
            dtype=sharding.float_type,
        )
        ts_buf: list[float] = []
        py_idx: int = 0
        col_width = max(
            val_width,
            max(len(n) for n in ["t"] + stat_names),
        )
        stats_file = Path("stats.dat")

        if sharding.main_device and not stats_file.exists():
            _write_dat_header(stats_file, ["t"] + stat_names, col_width)

        buffer = buffer.at[py_idx].set(_stats_row(stats, driving))
        ts_buf.append(t)
        py_idx += 1

    # --- CN/AB2 scheme: seed the Adams-Bashforth history ---------------
    # ``step.scheme == "cnab2"`` carries a nonlinear-RHS history across
    # steps (``rhs_prev``): the full ``N^{n-1}`` for triply-periodic, or
    # the self-advection ``N_nl^{n-1} = (u' x omega')^{n-1}`` for
    # wall-bounded (whose base-flow coupling is made implicit; see
    # ``step_cnab2`` in ``timestep.py``).  The priming ``step_cnab2``
    # call (state discarded) both computes that history at ``u^0`` and
    # compiles ``step_cnab2`` outside the benchmark window.  The very
    # first *time step* is then taken with iterative-CN (a self-starting,
    # non-CFL-bound corrector), after which CN/AB2 uses this history (see
    # the loop's ``it > it0`` guard).  ``step_cnab2`` returns ``(state,
    # carry, error, num_c)``: triply-periodic reports ``error = num_c =
    # 0`` (no corrector); wall-bounded reports its FFT-free
    # base-flow-coupling corrector's count / error (with an automatic
    # iterative-CN fallback on non-convergence), so the corrector
    # diagnostic and the convergence-stop apply to it too.
    # (The steppers donate their array arguments, so every
    # pre-loop call that must keep ``state`` / ``rhs_prev`` alive
    # passes ``jnp.copy``-ies; the main loop rebinds and needs none.)
    scheme: str = params.step.scheme
    is_cnab2: bool = scheme == "cnab2"
    if is_cnab2:
        _, rhs_prev, *_ = step_cnab2(jnp.copy(state), jnp.zeros_like(state))

    # --- Steps (CFL) buffer setup --------------------------------------
    # The measured stepper serves two consumers: the ``steps.dat``
    # record (``outs.it_steps``) and the adaptive-dt controller's CFL
    # read (``step.cfl_cadence``); either enables it.  ``steps.dat``
    # itself stays tied to ``outs.it_steps`` alone.
    measure_steps: bool = params.outs.it_steps is not None
    needs_measured: bool = measure_steps or adaptive
    if needs_measured:
        # Warm-up call (all ranks; collective FFTs): provides the
        # measurement names and compiles the measured program
        # outside the benchmark window.  Outputs are discarded; the
        # donated inputs are copies so ``state`` / ``rhs_prev`` stay
        # alive.  The t0 row itself is recorded by the first loop
        # iteration when it0 % it_steps == 0.
        if is_cnab2:
            *_, meas = step_cnab2_measured(jnp.copy(state), jnp.copy(rhs_prev))
        else:
            *_, meas = predict_and_fully_correct_measured(jnp.copy(state))
        first_measured = (
            measure_steps and it % params.outs.it_steps == 0
        ) or (adaptive and it % cfl_cadence == 0)
        every_measured = (measure_steps and params.outs.it_steps == 1) or (
            adaptive and cfl_cadence == 1
        )
        if first_measured and not every_measured:
            # The first loop iteration (excluded from benchmarks)
            # runs the measured variant, so the unmeasured program
            # would otherwise compile on the second iteration,
            # inside the benchmark window.  Pre-compile it here.
            if is_cnab2:
                step_cnab2(jnp.copy(state), jnp.copy(rhs_prev))
            else:
                predict_and_fully_correct(jnp.copy(state))
    if measure_steps:
        steps_names = list(meas.keys())
        steps_buffer = jnp.zeros(
            (params.outs.nbuffer, len(steps_names)),
            dtype=sharding.float_type,
        )
        steps_ts: list[float] = []
        steps_idx: int = 0
        steps_col_width = max(
            val_width,
            max(len(n) for n in ["t"] + steps_names),
        )
        steps_file = Path("steps.dat")

        if sharding.main_device and not steps_file.exists():
            _write_dat_header(steps_file, ["t"] + steps_names, steps_col_width)

    # --- Corrector buffer setup ----------------------------------------
    # Records the corrector iteration count ``c`` and final error every
    # ``it_corrector`` steps; both are already returned by every step
    # (no measured stepper variant needed).  Same buffering / format as
    # the stats and steps streams.
    measure_corrector: bool = params.outs.it_corrector is not None
    if measure_corrector:
        corr_names = ["c", "error"]
        corr_buffer = jnp.zeros(
            (params.outs.nbuffer, len(corr_names)),
            dtype=sharding.float_type,
        )
        corr_ts: list[float] = []
        corr_idx: int = 0
        corr_col_width = max(
            val_width,
            max(len(n) for n in ["t"] + corr_names),
        )
        corr_file = Path("corrector.dat")

        if sharding.main_device and not corr_file.exists():
            _write_dat_header(corr_file, ["t"] + corr_names, corr_col_width)

    # --- Probe buffer setup ----------------------------------------------
    # Spectral-mode probe stream (``probes.modes``/``probes.it_probes``,
    # the ``probes`` extension section): complex wall-normal mode
    # profiles into the binary ``probes.bin`` (see the
    # :mod:`dnsjax.extensions.probes` docstring).  Same buffering state
    # machine as the streams above; the t0 sample is recorded here (the
    # in-loop record skips ``it == it0``).  A non-finite message from
    # this record (possible only at ``nbuffer == 1``) is deferred to
    # ``_abort_non_finite`` below, which is not yet defined here.
    measure_probes: bool = probes_params.modes is not None
    probe_bad_t0: str | None = None
    if measure_probes:
        from .extensions.probes import ProbeStream

        probe_stream = ProbeStream(state)
        probe_bad_t0 = probe_stream.record(state, t)

    # --- Stochastic forcing setup ------------------------------------------
    # White-in-time mode kicks (``force.modes``, the ``force``
    # extension section; see the :mod:`dnsjax.extensions.forcing` docstring for
    # the conventions).  The forcer holds the channel profiles, the
    # rank-identical coefficient PRNG (advanced past any kicks already
    # recorded in an appended ``forcing.bin``), and the buffered
    # coefficient writer flushed with the other streams.
    force_on: bool = force_params.modes is not None
    if force_on:
        from .extensions.forcing import StochasticForcer

        forcer = StochasticForcer(state)

    def flush_all_buffers(check: bool = True) -> None:
        """Flush every buffered diagnostic stream to disk and reset it.

        Called at shutdown, before every snapshot write, after the
        first (JIT-heavy) step, and on a termination signal, so the
        ``.dat`` files stay consistent with the snapshots and survive
        an interruption.  The disk write is main-device only; the
        index / timestamp-list reset runs on all ranks (matching the
        in-loop fill-flush) so the ranks stay in lockstep.  With
        *check* (the default) the flushed rows are scanned for
        non-finite values and the run aborts on a hit; the abort path
        itself re-enters with ``check=False`` to drain the remaining
        streams (each stream is reset before its abort, so the
        re-entry cannot double-write).
        """
        nonlocal py_idx, steps_idx, corr_idx
        if params.outs.it_stats is not None and py_idx > 0:
            bad = None
            if sharding.main_device:
                bad = _flush_stats(
                    buffer,
                    py_idx,
                    ts_buf,
                    stats_file,
                    p,
                    col_width,
                    stat_names if check else None,
                )
            ts_buf.clear()
            py_idx = 0
            if bad is not None:
                _abort_non_finite(bad)
        if measure_steps and steps_idx > 0:
            bad = None
            if sharding.main_device:
                bad = _flush_stats(
                    steps_buffer,
                    steps_idx,
                    steps_ts,
                    steps_file,
                    p,
                    steps_col_width,
                    steps_names if check else None,
                )
            steps_ts.clear()
            steps_idx = 0
            if bad is not None:
                _abort_non_finite(bad)
        if measure_corrector and corr_idx > 0:
            bad = None
            if sharding.main_device:
                bad = _flush_stats(
                    corr_buffer,
                    corr_idx,
                    corr_ts,
                    corr_file,
                    p,
                    corr_col_width,
                    corr_names if check else None,
                )
            corr_ts.clear()
            corr_idx = 0
            if bad is not None:
                _abort_non_finite(bad)
        if measure_probes:
            # ProbeStream gates disk I/O on the main process and
            # resets its buffer on all ranks itself.
            bad = probe_stream.flush(check=check)
            if bad is not None:
                _abort_non_finite(bad)
        if force_on:
            # Kick coefficients are finite by construction (host-drawn
            # Gaussians): flushed for durability, never checked.
            forcer.flush()

    def _abort_non_finite(reason: str) -> None:
        """Abort the run on a non-finite (NaN/inf) diagnostic value.

        Prints one FATAL line, flushes every stream unchecked (the
        offending rows are already on disk from the checked flush that
        found them), and exits with code 3.  Deliberately writes no
        final snapshot -- the state is non-finite; the last snapshot on
        disk is the post-mortem artifact.  Buffer-scan aborts fire on
        the main process only (flushes are main-device-gated) and rely
        on the launcher to tear down the peers, like
        ``sharding.exit``; the scalar-guard aborts (corrector error,
        ``E'``, initial/final stats) fire identically on every rank.
        """
        sharding.print(f"FATAL: {reason}; aborting.")
        flush_all_buffers(check=False)
        sys.exit(3)

    # On a termination signal (e.g. a scheduler wall-time kill or
    # Ctrl-C) flush the buffers before exiting so no diagnostics are
    # lost.  ``_terminating`` guards against a second signal re-entering
    # the handler mid-flush.  SIGKILL cannot be caught.
    _terminating: bool = False

    def _flush_and_exit(signum: int, frame: object) -> None:
        nonlocal _terminating
        if _terminating:
            return
        _terminating = True
        sharding.print(
            f"Received signal {signum}; flushing buffers and exiting."
        )
        # Unchecked: a non-finite-abort inside a signal handler would
        # only obscure the (user- or scheduler-initiated) termination.
        flush_all_buffers(check=False)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _flush_and_exit)
    signal.signal(signal.SIGINT, _flush_and_exit)

    if probe_bad_t0 is not None:
        # Deferred check of the t0 probe record (see the probe setup).
        _abort_non_finite(probe_bad_t0)

    sharding.print("Started timestepping at", datetime.now())

    # --- Main time-stepping loop ---------------------------------------------
    while (
        t < t_stop
        and wall_time_now - wall_time_start < wall_time_stop
        and last_error < params.step.corrector_tolerance
        and not laminarized
    ):
        if it == params.init.it0 + 1:
            # Start the benchmark clock after the first (JIT-heavy)
            # iteration.  Block explicitly: with async dispatch the
            # first step may still be executing here, and it must
            # stay excluded from the wall-clock statistics.
            jax.block_until_ready(state)

            # Flush the diagnostics buffered during the first step (the
            # t0 stats row plus any first CFL / corrector rows) to disk
            # now, before the benchmark clock starts so the flush I/O is
            # excluded from the timing.
            flush_all_buffers()

            bench_start = perf_counter_ns()

            sharding.print("First iteration over at", datetime.now())

        # Physical view of u^n, built at most once per iteration and
        # shared by every full-field consumer below (the component-basis
        # boundary; the probe stream converts its own extracted columns
        # instead, which are far smaller than the field).  ``None`` on
        # an ordinary step, and the identity on the geometries whose
        # solver basis *is* the physical one.
        do_stats = (
            params.outs.it_stats is not None
            and it % params.outs.it_stats == 0
            and it > params.init.it0
        )
        do_snapshot = (
            params.outs.it_snapshot is not None
            and it % params.outs.it_snapshot == 0
            and it > params.init.it0
        )
        state_phys = (
            from_solver_basis_jit(state) if (do_stats or do_snapshot) else None
        )

        # Periodic diagnostic output -> GPU buffer
        if do_stats:
            # ``last_driving`` is the driving applied by the step that
            # produced this state (the previous iteration's), which is
            # what this row's time stamp refers to.
            stats = get_stats(state_phys)
            buffer = buffer.at[py_idx].set(_stats_row(stats, last_driving))
            ts_buf.append(t)
            py_idx += 1

            if py_idx == params.outs.nbuffer:
                bad = None
                if sharding.main_device:
                    bad = _flush_stats(
                        buffer,
                        py_idx,
                        ts_buf,
                        stats_file,
                        p,
                        col_width,
                        stat_names,
                    )
                ts_buf.clear()
                py_idx = 0
                if bad is not None:
                    _abort_non_finite(bad)

        # Spectral-mode probe sample -> GPU buffer (t0 was recorded at
        # setup; a cadence-aligned final sample is recorded after the
        # loop, keeping the time grid uniform).
        if (
            measure_probes
            and it % probes_params.it_probes == 0
            and it > params.init.it0
        ):
            bad = probe_stream.record(state, t)
            if bad is not None:
                _abort_non_finite(bad)

        # Periodic snapshot save (top of loop: same (state, it, t) as the
        # stats row above, so a fresh ``it_stats`` computation is reused;
        # the final state after the last step is covered by the
        # final-snapshot block after the loop).
        if do_snapshot:
            # ``stats`` was just computed above for this same state.
            snap_stats = stats if do_stats else get_stats(state_phys)
            # Flush first (checked; also keeps the .dat files
            # consistent with this snapshot): a buffered non-finite
            # diagnostic must abort before a snapshot of the same
            # broken state is written.
            flush_all_buffers()
            isnap = _save_numbered_snapshot(
                state_phys, t, it, snap_stats, isnap
            )
            last_saved_it = it

        # Release the physical view before the step: it is a
        # field-sized array, and holding it across the step would add
        # to the step's peak allocation for nothing.
        state_phys = None

        # Stochastic forcing kick (``force.modes``): fires after the
        # equal-t probe sample and any snapshot above (both record
        # the pre-kick state) and immediately before the step, so a
        # resumed continuation never double-applies a kick (saved
        # states are pre-kick; the kick belongs to the segment that
        # steps from them) and the coefficient stream continues
        # seamlessly (see :mod:`dnsjax.extensions.forcing`).
        if force_on and it % force_params.it_force == 0:
            state = forcer.kick(state, t)

        # Time step (single JIT scope): iterative Crank-Nicolson
        # corrector, or one CN/AB2 step carrying the nonlinear-RHS
        # history.  The measured variant also records the CFL of the
        # pre-step state u^n, timestamped at the current t; it runs
        # for the steps.dat record and/or the adaptive controller.
        do_record = measure_steps and it % params.outs.it_steps == 0
        do_cfl = adaptive and it % cfl_cadence == 0
        do_measure = do_record or do_cfl
        # CN/AB2 self-start: take the very first step with the robust
        # iterative-CN corrector (it needs no RHS history and is not
        # advective-CFL bound), then switch to CN/AB2 with the
        # ``rhs_prev`` history seeded from ``u^0`` below.
        if is_cnab2 and it > params.init.it0:
            if do_measure:
                (
                    state,
                    rhs_prev,
                    error_dev,
                    c_dev,
                    last_driving,
                    meas,
                ) = step_cnab2_measured(state, rhs_prev)
            else:
                state, rhs_prev, error_dev, c_dev, last_driving = step_cnab2(
                    state, rhs_prev
                )
        elif do_measure:
            (
                state,
                error_dev,
                c_dev,
                last_driving,
                meas,
            ) = predict_and_fully_correct_measured(state)
        else:
            state, error_dev, c_dev, last_driving = predict_and_fully_correct(
                state
            )

        if do_record:
            steps_buffer = steps_buffer.at[steps_idx].set(
                jnp.stack(list(meas.values()))
            )
            steps_ts.append(t)
            steps_idx += 1

            if steps_idx == params.outs.nbuffer:
                bad = None
                if sharding.main_device:
                    bad = _flush_stats(
                        steps_buffer,
                        steps_idx,
                        steps_ts,
                        steps_file,
                        p,
                        steps_col_width,
                        steps_names,
                    )
                steps_ts.clear()
                steps_idx = 0
                if bad is not None:
                    _abort_non_finite(bad)

        # Corrector diagnostic -> GPU buffer: this step's iteration
        # count and final error, timestamped at the pre-step time.
        if measure_corrector and it % params.outs.it_corrector == 0:
            corr_buffer = corr_buffer.at[corr_idx].set(
                jnp.stack(
                    [
                        c_dev.astype(corr_buffer.dtype),
                        error_dev.astype(corr_buffer.dtype),
                    ]
                )
            )
            corr_ts.append(t)
            corr_idx += 1

            if corr_idx == params.outs.nbuffer:
                bad = None
                if sharding.main_device:
                    bad = _flush_stats(
                        corr_buffer,
                        corr_idx,
                        corr_ts,
                        corr_file,
                        p,
                        corr_col_width,
                        corr_names,
                    )
                corr_ts.clear()
                corr_idx = 0
                if bad is not None:
                    _abort_non_finite(bad)

        t += params.step.dt
        it += 1

        # Adaptive-dt controller: consume the CFL measured at the
        # just-completed step's pre-step state (that step is booked
        # at the old dt above) and, on an accepted proposal, switch
        # the live dt -- an on-device leaf rebuild (``set_dt``), no
        # recompilation.  ``reset_ab2_kappa`` restores the AB2 ratio
        # to 1 after exactly one step at the new dt.
        if adaptive:
            changed = False
            if do_cfl:
                cfl_host = float(meas["CFL"])  # host sync
                if not math.isfinite(cfl_host):
                    _abort_non_finite(
                        f"non-finite CFL ({cfl_host}) at "
                        f"t = {t:.6e}, it = {it}"
                    )
                new_dt = propose_dt(
                    cfl_host,
                    params.step.dt,
                    cfl_target=params.step.cfl_target,
                    dt_min=params.step.dt_min,
                    dt_max=params.step.dt_max,
                    dt_min_change=params.step.dt_min_change,
                    dt_max_change=params.step.dt_max_change,
                    dt_threshold=params.step.dt_threshold,
                )
                if new_dt != params.step.dt:
                    set_dt(new_dt)
                    sharding.print(
                        f"[adaptive] t = {t:.6e}, it = {it}: "
                        f"CFL = {cfl_host:.3f}, dt "
                        f"{params.step.dt:.4e} -> {new_dt:.4e}"
                    )
                    params.step.dt = new_dt
                    changed = True
            if kappa_pending and not changed:
                reset_ab2_kappa()
            kappa_pending = changed

        # On-device accumulation (no host sync).
        c_sum = c_sum + c_dev
        if it == params.init.it0 + 1:
            c_first = c_dev

        if (it - params.init.it0) % it_error_check == 0:
            # Periodic host sync for the convergence check.
            last_error = float(error_dev)
            if not math.isfinite(last_error):
                _abort_non_finite(
                    f"non-finite corrector error ({last_error}) at "
                    f"t = {t:.6e}, it = {it}"
                )

            if check_laminarization:
                # Laminarization (relaminarization) trigger: stop once
                # the perturbation kinetic energy E' falls below the
                # threshold.  ``state`` here is the fully updated
                # post-step field (the periodic divergence correction
                # is fused into the step itself), so this is a second
                # state and gets its own single crossing of the
                # component-basis boundary -- the top-of-loop view was
                # of u^n and is long gone.
                #
                # Future feature: a sharper relaminarization signal is
                # the norm of the *complete* RHS (Laplacian included)
                # going to zero -- it vanishes only at a genuine fixed
                # point, not merely at low energy.  The complete RHS is
                # never explicitly formed (the viscous term is implicit
                # in the Helmholtz solve), but the increment-norm proxy
                # ``||u^{n+1} - u^n|| / dt`` is essentially free: by
                # construction of the Crank-Nicolson update it equals
                # the time-average of the complete projected RHS, needs
                # only states already in hand (one extra norm, no
                # operator evaluations, no pressure solve), and goes to
                # zero at the laminar fixed point.  Tracking it would
                # mean keeping the pre-step state for the difference.
                e_prime_host = float(_perturbation_energy_solver(state))
                if not math.isfinite(e_prime_host):
                    # A NaN would otherwise compare False against the
                    # threshold and the run would keep going.
                    _abort_non_finite(
                        f"non-finite perturbation energy E' "
                        f"({e_prime_host}) at t = {t:.6e}, it = {it}"
                    )
                laminarized = e_prime_host < laminarization_threshold

        wall_time_now = perf_counter_ns()

    # --- Post-processing -----------------------------------------------------
    # Single shutdown sync of the device-side corrector counters
    # (this also waits for all in-flight steps to complete).
    n_steps: int = it - params.init.it0
    if n_steps > 0:
        last_error = float(error_dev)
        last_c = int(c_dev)
        c_tot = int(c_sum)
        c_first_int = int(c_first)
    else:
        last_c = 0
        c_tot = 0
        c_first_int = 0

    if last_error > params.step.corrector_tolerance:
        sharding.print(
            f"Corrector failed to converge at t={t}, it={it}, c={last_c}, "
            f"with error = {last_error:.3e}."
        )

    if laminarized:
        sharding.print(
            f"Laminarized: E' = {e_prime_host:.3e} < "
            f"{laminarization_threshold:.3e} at t={t}, it={it}."
        )

    sharding.print("Stopped timestepping at", datetime.now())

    # Out of the solver, once, for the final state: the physical view
    # shared by the final stats and the final snapshot below (the probe
    # stream converts its own columns).
    state_phys = from_solver_basis_jit(state)

    # Final-state stats: computed once when the run stepped, reused by
    # both the final snapshot and the benchmark diagnostic below.
    if it > params.init.it0:
        stats = get_stats(state_phys)
        bad_final = [
            k for k, v in stats.items() if not math.isfinite(float(v))
        ]
        if bad_final:
            # Catches a blow-up in the last steps between error
            # checks, before the final snapshot below could write it.
            _abort_non_finite(
                f"non-finite final statistic(s) {', '.join(bad_final)} "
                f"at t = {t:.6e}, it = {it}"
            )

    # Final probe sample: the loop records *before* each step, so the
    # post-step final state is never sampled there.  Record it only
    # when cadence-aligned (a stop.max_sim_time horizon of a whole
    # number of probe intervals ends exactly on the grid), keeping the
    # sample times uniform for the readers.
    if (
        measure_probes
        and it > params.init.it0
        and it % probes_params.it_probes == 0
    ):
        bad = probe_stream.record(state, t)
        if bad is not None:
            _abort_non_finite(bad)

    # Final snapshot (default-on, independent of it_snapshot); skipped
    # when the final state was just written (a periodic or IC write at
    # this same iteration).
    if (
        params.outs.snapshot_save_final
        and it > params.init.it0
        and it != last_saved_it
    ):
        # Flush first (checked; also keeps the .dat files consistent
        # with this snapshot): buffered non-finite rows abort before
        # the write.
        flush_all_buffers()
        isnap = _save_numbered_snapshot(state_phys, t, it, stats, isnap)
        last_saved_it = it

    wall_time_now = perf_counter_ns()
    alive_time = ns_to_s * (wall_time_now - wall_time_start)
    sharding.print(f"Job has been alive for {alive_time:.2f}s.")
    if it > params.init.it0 + 1:
        wall_time = ns_to_s * (wall_time_now - bench_start)
        wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
        # Nonlinear/FFT evaluations within the benchmark window,
        # excluding the first (JIT-heavy) step.  iterative-cn: 2 per
        # step plus 1 per extra corrector iteration.  cnab2: exactly
        # one FFT per step -- its base-flow-coupling corrector
        # iterations (counted in c) are FFT-free.
        if is_cnab2:
            rhs_tot = n_steps - 1
        else:
            rhs_tot = (c_tot - c_first_int) + 2 * (n_steps - 1)
        wall_time_per_rhs = wall_time / rhs_tot

        # Final diagnostic output (``stats`` was computed above for the
        # final state).
        c_per_it = c_tot / n_steps

        if params.outs.it_stats is not None:
            buffer = buffer.at[py_idx].set(_stats_row(stats, last_driving))
            ts_buf.append(t)
            py_idx += 1

        sharding.print(
            f"t = {t:.2f}",
            *[f"{x}={y:.3e}" for x, y in stats.items()],
            f"c/it = {c_per_it:.2f}",
            f"err = {last_error:.3e}",
        )

        if sharding.n_devices > 1:
            sharding.print(
                f"Ran for {wall_time:.2f}s with {sharding.n_devices} devices,",
                f"{sharding.n_devices * wall_time:.3e} NP x s:",
                f"{wall_time_per_sim_time:.3e} s/t,",
                f"{sharding.n_devices * wall_time_per_sim_time:.3e} NP x s/t,",
                f"{wall_time_per_rhs:.3e} s/rhs,",
                f"{sharding.n_devices * wall_time_per_rhs:.3e} NP x s/rhs.",
            )
        else:
            sharding.print(
                f"Ran for {wall_time:.2f}s with 1 device.",
                f"{wall_time_per_sim_time:.3e} s/t,",
                f"{wall_time_per_rhs:.3e} s/rhs.",
            )

    # Flush any remaining buffered diagnostic rows.
    flush_all_buffers()


def main(argv: list[str] | None = None) -> int:
    """Production entry point (console script / ``python -m dnsjax``).

    Resolves the configuration layers
    (:func:`dnsjax.bootstrap.resolve_parameters`; ``--help`` exits
    here with the full CLI reference and no side effects), configures
    the distributed JAX runtime
    (:func:`dnsjax.bootstrap.configure_jax_runtime`), prints the
    final configuration on the main process, and runs the simulation
    (:func:`run`).  *argv* defaults to ``sys.argv``.
    """
    wall_time_start = perf_counter_ns()

    # Parameters resolve first (fast, pre-JAX): --help / --sample-toml
    # exit in there with clean output (no banner prefix).
    setup = resolve_parameters(argv)
    # Per-rank lifecycle heartbeats bracket the whole process (this one
    # fires before ``configure_jax_runtime`` -- i.e. before the main
    # device is even known -- so it cannot be main-device-gated).  They
    # go to *stderr* so the main rank stays the sole writer of *stdout*:
    # a peer's ``Shutdown at`` on stdout could otherwise be spliced into
    # the middle of the main rank's final summary line by ``mpirun``'s
    # stream merging (the two are separate ranks writing one merged
    # stdout), which no downstream stdout parser can reassemble.  A
    # barrier is *not* usable here -- the buffer-scan non-finite abort
    # exits the main rank only (peers are torn down by the launcher), so
    # a collective would deadlock.
    print("Alive at", datetime.now(), flush=True, file=sys.stderr)
    main_device = configure_jax_runtime()

    if main_device:
        print("Distribution initialized at", datetime.now(), flush=True)
        print("Code version:", git_hash(), flush=True)
        if setup.snapshot_params_used:
            # The initial-condition snapshot's own provenance.
            print(
                "Snapshot was recorded by code version:",
                read_snapshot_meta(setup.snapshot_path)["git_hash"],
                flush=True,
            )
        if setup.snapshot_params_used:
            print(
                f"Inherited parameters embedded in snapshot "
                f"'{setup.snapshot_path}' (except np0/np1/platform/"
                "double_precision); parameters.toml and command-line "
                "arguments override them.",
                flush=True,
            )
        if setup.params_from_disk:
            print(
                "Loaded parameters.toml, "
                "which override the snapshot and default parameters. "
                "Command-line arguments will further override "
                "the loaded parameters.",
                flush=True,
            )
        else:
            print(
                "Loaded the default parameters, "
                "as parameters.toml was not found. "
                "Command-line arguments will further override "
                "the default parameters.",
                flush=True,
            )
        # The full resolved-parameter dump is provenance for a real run
        # but pure repeated noise when a test launches the solver dozens
        # of times (the launching command already carries every argument
        # and a failing test dumps the child output).  The mpirun smoke
        # tests set ``DNSJAX_QUIET_STARTUP=1`` (``tests/_live.run_live``)
        # to skip it; a normal run is unaffected.
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

    run(wall_time_start)

    # Per-rank shutdown heartbeat -- stderr, see the "Alive at" note
    # above (keeps the main rank's stdout summary line un-spliced).
    print("Shutdown at", datetime.now(), flush=True, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
