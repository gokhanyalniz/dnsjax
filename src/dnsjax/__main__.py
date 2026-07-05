#!/usr/bin/env python3
"""Entry point for the dnsjax DNS solver.

Import order
------------
JAX platform and distributed backend must be configured
*before* importing any module that reads ``sharding`` or a
geometry module, because those modules instantiate global
singletons (``params``, ``derived_params``, ``sharding``,
``fourier``) at import time.  This module enforces the
constraint by deferring ``import jax`` and all flow-module
imports until after CLI / TOML configuration is applied.

Execution phases
----------------
1. **Initialisation** (module level, ``__main__`` guard):
   parse CLI arguments, load ``parameters.toml`` if present,
   configure JAX, print the final parameter set.

2. **Main loop** (:func:`main`):
   initialise velocity (a provided snapshot wins; otherwise an
   in-process random / localized-rolls / laminar IC, with random
   the default), then iterate:

   - Fused predictor + corrector loop
     (:func:`predict_and_fully_correct`)
   - Divergence correction + mean-mode zeroing for
     triply-periodic flows (:func:`correct_velocity`)
   - Periodic diagnostic output (:func:`get_stats`)

   The loop terminates when the simulation time, wall-clock
   time, or corrector divergence criterion is reached.  The
   corrector error and iteration counters stay on the device;
   the error is synced to the host only every
   ``outs.it_error_check`` steps so that JAX async dispatch can
   pipeline steps (divergence is detected at most
   ``it_error_check`` steps late).

Diagnostics (``stats.dat``, ``steps.dat``, ``corrector.dat``)
-------------------------------------------------------------
``get_stats`` output is accumulated on-device in a fixed
``(nbuffer, n_cols)`` buffer (one row every ``it_stats``
steps) and flushed to ``stats.dat`` when the buffer fills, at
shutdown, after the first (JIT-heavy) step, after every
snapshot write, and on a termination signal
(``flush_all_buffers``, which calls the shared
``_flush_stats``).  Buffering avoids a
host-device sync per sample; each flush is then ``fsync``-ed,
so the rows are on disk immediately once the on-device buffer
is flushed.  ``stats.dat`` (written by the main device,
appended) has a header row of column names (``t`` plus the
``get_stats`` keys) followed by whitespace-aligned rows at
``stats_precision`` significant digits.

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
recorded after the final step.

``corrector.dat`` records the corrector diagnostic every
``it_corrector`` steps (same buffering and file format): the
corrector iteration count ``c`` and the final corrector error
``error``, both already returned by every step, timestamped at
the pre-step time.  ``outs.it_error_check`` must not exceed
``it_corrector`` (:func:`dnsjax.parameters.validate_parameters`).

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
``state00000.tar``) unless ``init.force_resume`` is set.  Legacy
``.npz`` files go through geometry-specific ``init_state``.  When
the current wall-normal grid differs from the snapshot's,
``_interpolate_if_needed`` interpolates the state at load time
(see :mod:`dnsjax.fd` for the interpolation methods).

Benchmarking
------------
The first time step is excluded from wall-clock statistics
because it includes JAX's JIT compilation overhead.
"""

import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from pprint import pp
from time import perf_counter_ns

from pydantic_settings import CliApp

from .parameters import (
    CLIParameters,
    Parameters,
    annular_systems,
    cylindrical_systems,
    derived_params,
    ns_to_s,
    padded_res,
    params,
    periodic_systems,
    read_parameters,
    read_snapshot_params,
    trajectory_defining_changes,
    update_parameters,
    validate_parameters,
    viscoelastic_systems,
    walled_systems,
)


def _flush_stats(buffer, n_valid, ts_buf, file_path, p, col_width):
    """Write *n_valid* buffered rows to *file_path*, durably.

    The on-device ``buffer`` is the only batching layer: once it fills
    (``nbuffer`` rows) this transfers it to the host and appends the
    rows.  An explicit ``flush`` + ``fsync`` then forces the bytes out of
    the process and OS buffers so each device-buffer flush is immediately
    on disk (and visible to other clients on networked filesystems).
    Shared by every measurement stream (stats, steps, corrector).
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


def _resume_snapshot_path(
    params_cli: Parameters, params_in: Parameters | None
) -> Path | None:
    """Return the snapshot path to resume from (CLI over TOML).

    Inspects only the *explicitly set* ``init.snapshot`` of each layer
    (via ``exclude_unset``), so an unset field never shadows a lower
    layer.  Returns ``None`` when neither layer sets it (a laminar start,
    or the path lives only in the code defaults -- which is ``None``).
    """
    for layer in (params_cli, params_in):
        if layer is None:
            continue
        init = layer.model_dump(exclude_unset=True).get("init") or {}
        snap = init.get("snapshot")
        if snap is not None:
            return Path(snap)
    return None


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
    from .operators import complex_harmonics

    meta = read_metadata(Path(snap_path))
    snap_grid = meta.get("wall_normal_grid")
    snap_ny = meta.get("params", {}).get("res", {}).get("ny")
    curr_grid = derived_params.wall_normal_grid

    if snap_ny is None or curr_grid is None:
        return state

    needs_interp = snap_ny != params.res.ny or (
        snap_grid is not None
        and not np.allclose(snap_grid, curr_grid, atol=1e-12)
    )

    if not needs_interp:
        return state

    curr_grid_np = np.array(curr_grid)
    if snap_grid is not None:
        old_grid = np.array(snap_grid)
    else:
        # Legacy snapshot without grid metadata: such snapshots
        # predate ``geo.axis_gap``, so assume the default grid of
        # their era (cylindrical: the g = 0 half-CGL grid).
        if params.phys.system in cylindrical_systems:
            N_full = 2 * snap_ny
            s = -np.cos(np.arange(N_full) * np.pi / (N_full - 1))
            old_grid = s[snap_ny:]
        elif params.phys.system in annular_systems:
            # CGL of [-1, 1] mapped to [r_inner, r_outer].
            xi = -np.cos(np.arange(snap_ny) * np.pi / (snap_ny - 1))
            mid = 0.5 * (derived_params.r_inner + derived_params.r_outer)
            half = 0.5 * (derived_params.r_outer - derived_params.r_inner)
            old_grid = mid + half * xi
        else:
            old_grid = -np.cos(np.arange(snap_ny) * np.pi / (snap_ny - 1))

    if params.phys.system in cylindrical_systems:
        geometry = "cylindrical"
    elif (
        params.phys.system in annular_systems
        or params.phys.system in viscoelastic_systems
    ):
        geometry = "annular"
    else:
        geometry = "cartesian"
    T = build_interpolation_matrix(
        old_grid, curr_grid_np, geometry, params.res.fd_order
    )

    if isinstance(T, tuple):
        # Spectral parity-aware cylindrical interpolation: apply the
        # even / odd radial matrix per azimuthal mode m by component
        # parity.  State layout (component, r, m, kz): u_z parity
        # (-1)^m; u_+/u_- parity (-1)^{m+1}.
        T_even, T_odd = T
        m_is_even = np.asarray(complex_harmonics(params.res.nz)) % 2 == 0
        T_p = np.where(m_is_even[:, None, None], T_even, T_odd)  # u_z
        T_v = np.where(m_is_even[:, None, None], T_odd, T_even)  # u_+/-
        T_p_jax = jnp.asarray(T_p, dtype=state.dtype)
        T_v_jax = jnp.asarray(T_v, dtype=state.dtype)
        # (m, i_new, j_old) x (j_old, m, k_kz) -> (i_new, m, k_kz)
        s0 = jnp.einsum("mij, jmk -> imk", T_p_jax, state[0])
        s1 = jnp.einsum("mij, jmk -> imk", T_v_jax, state[1])
        s2 = jnp.einsum("mij, jmk -> imk", T_v_jax, state[2])
        state = jnp.stack([s0, s1, s2])
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


def main() -> None:
    """Run the time-stepping loop after parameters and JAX are initialised."""
    import jax
    from jax import numpy as jnp

    from .sharding import sharding

    # --- Flow dispatch -------------------------------------------------------
    if params.phys.system in periodic_systems:
        from .flows.triply_periodic.monochromatic import (
            correct_velocity,
            get_perturbation_energy,
            get_stats,
            init_state,
            predict_and_fully_correct,
            predict_and_fully_correct_measured,
            step_cnab2,
            step_cnab2_measured,
        )
    elif params.phys.system == "plane-couette":
        from .flows.wall_bounded.plane_couette import (
            get_perturbation_energy,
            get_stats,
            init_state,
            predict_and_fully_correct,
            predict_and_fully_correct_measured,
            step_cnab2,
            step_cnab2_measured,
        )
    elif params.phys.system == "plane-poiseuille":
        from .flows.wall_bounded.plane_poiseuille import (
            get_perturbation_energy,
            get_stats,
            init_state,
            predict_and_fully_correct,
            predict_and_fully_correct_measured,
            step_cnab2,
            step_cnab2_measured,
        )
    elif params.phys.system == "pipe":
        from .flows.wall_bounded.pipe import (
            get_perturbation_energy,
            get_stats,
            init_state,
            predict_and_fully_correct,
            predict_and_fully_correct_measured,
            step_cnab2,
            step_cnab2_measured,
        )
    elif params.phys.system == "taylor-couette":
        from .flows.wall_bounded.taylor_couette import (
            get_perturbation_energy,
            get_stats,
            init_state,
            predict_and_fully_correct,
            predict_and_fully_correct_measured,
            step_cnab2,
            step_cnab2_measured,
        )
    elif params.phys.system == "dean":
        from .flows.wall_bounded.dean import (
            get_perturbation_energy,
            get_stats,
            init_state,
            predict_and_fully_correct,
            predict_and_fully_correct_measured,
            step_cnab2,
            step_cnab2_measured,
        )
    elif params.phys.system == "viscoelastic-dean":
        from .flows.wall_bounded.viscoelastic_dean import (
            get_perturbation_energy,
            get_stats,
            init_state,
            predict_and_fully_correct,
            predict_and_fully_correct_measured,
            step_cnab2,
            step_cnab2_measured,
        )
    else:
        sharding.print(
            f"System '{params.phys.system}' is not yet implemented."
        )
        sharding.exit(code=1)

    # --- Initial condition ---------------------------------------------------
    from .snapshot_meta import is_snapshot_file

    # Start-mode precedence: a provided snapshot file (tar resume or
    # legacy .npz) wins over every in-process mode; then
    # start_from_laminar, then localized_rolls, then random_field (the
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
        # Legacy .npz snapshot.  A provided snapshot still wins over every
        # in-process mode (start_from_laminar / localized_rolls /
        # random_field).
        state = init_state(params.init.snapshot)
    elif params.init.start_from_laminar:
        # Laminar / closed-form base state (snapshot is None here).
        state = init_state(params.init.snapshot)
    elif params.init.localized_rolls:
        # In-process deterministic localized-rolls ("spot") IC (no
        # snapshot file). The flow dispatch above already built the
        # geometry ``fourier`` singleton this consumes.
        from .localized_rolls import generate_localized_rolls

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
        from .random_field import generate_random_state

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
        next isnap."""
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

    if check_laminarization:
        # Compile the E' kernel outside the benchmark window; it is
        # read at the it_error_check cadence in the main loop.
        jax.block_until_ready(get_perturbation_energy(state))

    # Save the initial condition (state00000.tar) unless this run is a
    # continuation of a dnsjax snapshot trajectory (whose IC is already
    # on disk as that snapshot).  ``stats`` here is the IC's stats.
    if params.outs.snapshot_save_initial and not resumed_continuation:
        isnap = _save_numbered_snapshot(state, t, it, stats, isnap)
        last_saved_it = it

    # --- Stats buffer setup ------------------------------------------------
    p = params.outs.stats_precision - 1
    val_width = params.outs.stats_precision + 7

    if params.outs.it_stats is not None:
        stat_names = list(stats.keys())
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
            header = " ".join(n.rjust(col_width) for n in ["t"] + stat_names)
            with open(stats_file, "w") as f:
                f.write(header + "\n")

        stat_vals = jnp.stack(list(stats.values()))
        buffer = buffer.at[py_idx].set(stat_vals)
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
        _, rhs_prev, _, _ = step_cnab2(jnp.copy(state), jnp.zeros_like(state))

    # --- Steps (CFL) buffer setup --------------------------------------
    measure_steps: bool = params.outs.it_steps is not None
    if measure_steps:
        # Warm-up call (all ranks; collective FFTs): provides the
        # measurement names and compiles the measured program
        # outside the benchmark window.  Outputs are discarded; the
        # donated inputs are copies so ``state`` / ``rhs_prev`` stay
        # alive.  The t0 row itself is recorded by the first loop
        # iteration when it0 % it_steps == 0.
        if is_cnab2:
            *_, meas = step_cnab2_measured(jnp.copy(state), jnp.copy(rhs_prev))
        else:
            _, _, _, meas = predict_and_fully_correct_measured(jnp.copy(state))
        if params.outs.it_steps > 1 and it % params.outs.it_steps == 0:
            # The first loop iteration (excluded from benchmarks)
            # runs the measured variant, so the unmeasured program
            # would otherwise compile on the second iteration,
            # inside the benchmark window.  Pre-compile it here.
            if is_cnab2:
                step_cnab2(jnp.copy(state), jnp.copy(rhs_prev))
            else:
                predict_and_fully_correct(jnp.copy(state))
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
            header = " ".join(
                n.rjust(steps_col_width) for n in ["t"] + steps_names
            )
            with open(steps_file, "w") as f:
                f.write(header + "\n")

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
            header = " ".join(
                n.rjust(corr_col_width) for n in ["t"] + corr_names
            )
            with open(corr_file, "w") as f:
                f.write(header + "\n")

    def flush_all_buffers() -> None:
        """Flush every buffered diagnostic stream to disk and reset it.

        Called at shutdown, after every snapshot write, after the
        first (JIT-heavy) step, and on a termination signal, so the
        ``.dat`` files stay consistent with the snapshots and survive
        an interruption.  The disk write is main-device only; the
        index / timestamp-list reset runs on all ranks (matching the
        in-loop fill-flush) so the ranks stay in lockstep.
        """
        nonlocal py_idx, steps_idx, corr_idx
        if params.outs.it_stats is not None and py_idx > 0:
            if sharding.main_device:
                _flush_stats(
                    buffer,
                    py_idx,
                    ts_buf,
                    stats_file,
                    p,
                    col_width,
                )
            ts_buf.clear()
            py_idx = 0
        if measure_steps and steps_idx > 0:
            if sharding.main_device:
                _flush_stats(
                    steps_buffer,
                    steps_idx,
                    steps_ts,
                    steps_file,
                    p,
                    steps_col_width,
                )
            steps_ts.clear()
            steps_idx = 0
        if measure_corrector and corr_idx > 0:
            if sharding.main_device:
                _flush_stats(
                    corr_buffer,
                    corr_idx,
                    corr_ts,
                    corr_file,
                    p,
                    corr_col_width,
                )
            corr_ts.clear()
            corr_idx = 0

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
        flush_all_buffers()
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _flush_and_exit)
    signal.signal(signal.SIGINT, _flush_and_exit)

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

        # Periodic diagnostic output -> GPU buffer
        if (
            params.outs.it_stats is not None
            and it % params.outs.it_stats == 0
            and it > params.init.it0
        ):
            stats = get_stats(state)
            stat_vals = jnp.stack(list(stats.values()))
            buffer = buffer.at[py_idx].set(stat_vals)
            ts_buf.append(t)
            py_idx += 1

            if py_idx == params.outs.nbuffer:
                if sharding.main_device:
                    _flush_stats(
                        buffer,
                        py_idx,
                        ts_buf,
                        stats_file,
                        p,
                        col_width,
                    )
                ts_buf.clear()
                py_idx = 0

        # Periodic snapshot save (top of loop: same (state, it, t) as the
        # stats row above, so a fresh ``it_stats`` computation is reused;
        # the final state after the last step is covered by the
        # final-snapshot block after the loop).
        if (
            params.outs.it_snapshot is not None
            and it % params.outs.it_snapshot == 0
            and it > params.init.it0
        ):
            if (
                params.outs.it_stats is not None
                and it % params.outs.it_stats == 0
            ):
                snap_stats = stats  # just computed above for this state
            else:
                snap_stats = get_stats(state)
            isnap = _save_numbered_snapshot(state, t, it, snap_stats, isnap)
            last_saved_it = it
            # Keep the .dat files consistent with this snapshot.
            flush_all_buffers()

        # Time step (single JIT scope): iterative Crank-Nicolson
        # corrector, or one CN/AB2 step carrying the nonlinear-RHS
        # history.  The measured variant also records the CFL of the
        # pre-step state u^n, timestamped at the current t.
        do_measure = measure_steps and it % params.outs.it_steps == 0
        # CN/AB2 self-start: take the very first step with the robust
        # iterative-CN corrector (it needs no RHS history and is not
        # advective-CFL bound), then switch to CN/AB2 with the
        # ``rhs_prev`` history seeded from ``u^0`` below.
        if is_cnab2 and it > params.init.it0:
            if do_measure:
                state, rhs_prev, error_dev, c_dev, meas = step_cnab2_measured(
                    state, rhs_prev
                )
            else:
                state, rhs_prev, error_dev, c_dev = step_cnab2(state, rhs_prev)
        elif do_measure:
            state, error_dev, c_dev, meas = predict_and_fully_correct_measured(
                state
            )
        else:
            state, error_dev, c_dev = predict_and_fully_correct(state)

        if do_measure:
            steps_buffer = steps_buffer.at[steps_idx].set(
                jnp.stack(list(meas.values()))
            )
            steps_ts.append(t)
            steps_idx += 1

            if steps_idx == params.outs.nbuffer:
                if sharding.main_device:
                    _flush_stats(
                        steps_buffer,
                        steps_idx,
                        steps_ts,
                        steps_file,
                        p,
                        steps_col_width,
                    )
                steps_ts.clear()
                steps_idx = 0

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
                if sharding.main_device:
                    _flush_stats(
                        corr_buffer,
                        corr_idx,
                        corr_ts,
                        corr_file,
                        p,
                        corr_col_width,
                    )
                corr_ts.clear()
                corr_idx = 0

        if params.phys.system in periodic_systems:
            # Divergence correction and mean-mode zeroing
            state = correct_velocity(state)

        t += params.step.dt
        it += 1

        # On-device accumulation (no host sync).
        c_sum = c_sum + c_dev
        if it == params.init.it0 + 1:
            c_first = c_dev

        if (it - params.init.it0) % it_error_check == 0:
            # Periodic host sync for the convergence check.
            last_error = float(error_dev)

            if check_laminarization:
                # Laminarization (relaminarization) trigger: stop once
                # the perturbation kinetic energy E' falls below the
                # threshold.  ``state`` here is the fully updated
                # post-step field (post divergence-correction for
                # periodic systems).
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
                e_prime_host = float(get_perturbation_energy(state))
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

    # Final-state stats: computed once when the run stepped, reused by
    # both the final snapshot and the benchmark diagnostic below.
    if it > params.init.it0:
        stats = get_stats(state)

    # Final snapshot (default-on, independent of it_snapshot); skipped
    # when the final state was just written (a periodic or IC write at
    # this same iteration).
    if (
        params.outs.snapshot_save_final
        and it > params.init.it0
        and it != last_saved_it
    ):
        isnap = _save_numbered_snapshot(state, t, it, stats, isnap)
        last_saved_it = it
        # Keep the .dat files consistent with this snapshot.
        flush_all_buffers()

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
            stat_vals = jnp.stack(list(stats.values()))
            buffer = buffer.at[py_idx].set(stat_vals)
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


if __name__ == "__main__":
    print("Alive at", datetime.now(), flush=True)
    wall_time_start = perf_counter_ns()
    params_cli = CliApp.run(CLIParameters)

    params_file = Path("parameters.toml")
    params_in = (
        read_parameters(params_file) if Path.is_file(params_file) else None
    )
    params_from_disk = params_in is not None

    # Configuration layers, lowest priority first.  The snapshot to
    # resume from is whichever ``init.snapshot`` the user set explicitly
    # (CLI over TOML); its embedded parameters form the lowest layer
    # above the code defaults, so a resume inherits the snapshot's
    # configuration unless TOML or the CLI override it.
    snapshot_path = _resume_snapshot_path(params_cli, params_in)
    snapshot_params_used = False
    if snapshot_path is not None:
        snap_params = read_snapshot_params(snapshot_path)
        if snap_params is not None:
            update_parameters(snap_params)
            snapshot_params_used = True

    # Higher-priority layers: parameters.toml, then CLI arguments.
    if params_in is not None:
        update_parameters(params_in)
    update_parameters(params_cli)

    validate_parameters()
    padded_res.set_padded_resolution(params)

    if params.dist.platform == "cpu":
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )
        os.environ["NPROC"] = "1"

    import jax

    jax.config.update("jax_enable_x64", params.res.double_precision)
    jax.config.update("jax_platforms", params.dist.platform)
    jax.distributed.initialize()
    main_device: bool = bool(jax.process_index() == 0)

    if main_device:
        print("Distribution initialized at", datetime.now(), flush=True)
        if snapshot_params_used:
            print(
                f"Inherited parameters embedded in snapshot "
                f"'{snapshot_path}' (except np0/np1/platform/"
                "double_precision); parameters.toml and command-line "
                "arguments override them.",
                flush=True,
            )
        if params_from_disk:
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
        print("Final working parameters:")
        if main_device:
            pp(params.model_dump())

        print(
            "Running with the physical-space (x, y, z) resolution:",
            padded_res.nx_padded,
            padded_res.ny_padded
            if padded_res.ny_padded is not None
            else params.res.ny,
            padded_res.nz_padded,
            flush=True,
        )

    main()

    print("Shutdown at", datetime.now(), flush=True)
    sys.exit(0)
