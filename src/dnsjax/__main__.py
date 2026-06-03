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
   initialise velocity (from laminar or snapshot), then
   iterate:

   - Fused predictor + corrector loop
     (:func:`predict_and_fully_correct`)
   - Divergence correction + mean-mode zeroing for
     triply-periodic flows (:func:`correct_velocity`)
   - Periodic diagnostic output (:func:`get_stats`)

   The loop terminates when the simulation time, wall-clock
   time, or corrector divergence criterion is reached.

Diagnostics (``stats.dat``)
---------------------------
``get_stats`` output is accumulated on-device in a fixed
``(nstats, n_cols)`` buffer (one row every ``it_stats``
steps) and flushed to ``stats.dat`` when the buffer fills or
at shutdown (``_flush_stats``).  Buffering avoids a
host-device sync per sample.  ``stats.dat`` (written by the
main device, appended) has a header row of column names
(``t`` plus the ``get_stats`` keys) followed by
whitespace-aligned rows at ``stats_precision`` significant
digits.

Snapshot resume
---------------
Loading a zarr3 snapshot directory overrides ``params.init.t0``
and ``params.init.it0`` from the snapshot metadata.  Legacy
``.npz`` files go through geometry-specific ``init_state``.
When the current wall-normal grid differs from the snapshot's,
``_interpolate_if_needed`` interpolates the state at load time
(see :mod:`dnsjax.fd` for the interpolation methods).

Benchmarking
------------
The first time step is excluded from wall-clock statistics
because it includes JAX's JIT compilation overhead.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pp
from time import perf_counter_ns

from pydantic_settings import CliApp

from .parameters import (
    CLIParameters,
    cylindrical_systems,
    derived_params,
    ns_to_s,
    padded_res,
    params,
    periodic_systems,
    read_parameters,
    update_parameters,
    walled_systems,
)


def _flush_stats(buffer, n_valid, ts_buf, file_path, p, col_width):
    """Write *n_valid* buffered stats rows to *file_path*."""
    import numpy as np

    data = np.asarray(buffer[:n_valid])
    with open(file_path, "a") as f:
        for i in range(n_valid):
            t_str = f"{ts_buf[i]:.{p}e}".rjust(col_width)
            stat_strs = " ".join(
                f"{v:.{p}e}".rjust(col_width) for v in data[i]
            )
            f.write(f"{t_str} {stat_strs}\n")


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
        # Legacy snapshot without grid: assume default grid.
        if params.phys.system in cylindrical_systems:
            N_full = 2 * snap_ny
            s = -np.cos(np.arange(N_full) * np.pi / (N_full - 1))
            old_grid = s[snap_ny:]
        else:
            old_grid = -np.cos(np.arange(snap_ny) * np.pi / (snap_ny - 1))

    geometry = (
        "cylindrical"
        if params.phys.system in cylindrical_systems
        else "cartesian"
    )
    T = build_interpolation_matrix(old_grid, curr_grid_np, geometry)

    if isinstance(T, tuple):
        # Parity-aware cylindrical interpolation.
        T_even, T_odd = T
        m_vals = np.asarray(complex_harmonics(params.res.nz))
        m_is_even = m_vals % 2 == 0

        # T_p: parity (-1)^m for u_z (component 0)
        # T_v: parity (-1)^{m+1} for u_+, u_- (components 1, 2)
        T_p = np.where(m_is_even[:, None, None], T_even, T_odd)
        T_v = np.where(m_is_even[:, None, None], T_odd, T_even)
        T_p_jax = jnp.asarray(T_p, dtype=state.dtype)
        T_v_jax = jnp.asarray(T_v, dtype=state.dtype)

        # state shape: (3, Nm, Nkz, Nr_old)
        s0 = jnp.einsum("mij, mkj -> mki", T_p_jax, state[0])
        s1 = jnp.einsum("mij, mkj -> mki", T_v_jax, state[1])
        s2 = jnp.einsum("mij, mkj -> mki", T_v_jax, state[2])
        state = jnp.stack([s0, s1, s2])
    else:
        T_jax = jnp.asarray(T, dtype=state.dtype)
        # state shape: (3, kz, kx, ny_old)
        state = jnp.einsum("ij, ...j -> ...i", T_jax, state)

    # Enforce wall boundary conditions.
    if geometry == "cartesian":
        state = state.at[..., 0].set(0.0)
        state = state.at[..., -1].set(0.0)
    else:
        state = state.at[..., -1].set(0.0)

    sharding.print(
        "Interpolated wall-normal grid; first corrector step "
        "will project out any residual divergence."
    )
    return state


def main() -> None:
    """Run the time-stepping loop after parameters and JAX are initialised."""
    from jax import numpy as jnp

    from .sharding import sharding

    # --- Flow dispatch -------------------------------------------------------
    if params.phys.system in periodic_systems:
        from .flows.triply_periodic.monochromatic import (
            correct_velocity,
            get_stats,
            init_state,
            predict_and_fully_correct,
        )
    elif params.phys.system == "plane-couette":
        from .flows.wall_bounded.plane_couette import (
            get_stats,
            init_state,
            predict_and_fully_correct,
        )
    elif params.phys.system == "plane-poiseuille":
        from .flows.wall_bounded.plane_poiseuille import (
            get_stats,
            init_state,
            predict_and_fully_correct,
        )
    elif params.phys.system == "pipe":
        from .flows.wall_bounded.pipe import (
            get_stats,
            init_state,
            predict_and_fully_correct,
        )
    else:
        sharding.print(
            f"System '{params.phys.system}' is not yet implemented."
        )
        sharding.exit(code=1)

    # --- Initial condition ---------------------------------------------------
    if (
        params.init.snapshot is not None
        and Path(params.init.snapshot).is_dir()
    ):
        # zarr3 snapshot (new format)
        from .snapshot import (
            load_snapshot,
            read_metadata,
            validate_snapshot_params,
        )

        validate_snapshot_params(params.init.snapshot)
        state, t_snap, it_snap = load_snapshot(params.init.snapshot)
        params.init.t0 = t_snap
        params.init.it0 = it_snap
        sharding.print(f"Resumed from snapshot: t={t_snap:.6e}, it={it_snap}")

        # --- Wall-normal grid interpolation ---
        if params.phys.system in walled_systems:
            state = _interpolate_if_needed(
                state,
                Path(params.init.snapshot),
                read_metadata,
                sharding,
                jnp,
            )
    else:
        # Legacy .npz or laminar start
        state = init_state(params.init.snapshot)

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

    rhs_tot: int = 0
    c_tot: int = 0
    dt_first: float = params.step.dt
    wall_time_now: int = perf_counter_ns()
    last_error: float = 0.0
    last_c: int = 0

    # Warm-up call so that JIT compilation does not affect benchmarks
    stats = get_stats(state)

    sharding.print(
        f"t = {t:.2f}",
        *[f"{x}={y:.3e}" for x, y in stats.items()],
    )

    # --- Stats buffer setup ------------------------------------------------
    if params.outs.it_stats is not None:
        stat_names = list(stats.keys())
        n_stat_cols = len(stat_names)
        buffer = jnp.zeros(
            (params.outs.nstats, n_stat_cols),
            dtype=sharding.float_type,
        )
        ts_buf: list[float] = []
        py_idx: int = 0
        p = params.outs.stats_precision - 1
        val_width = params.outs.stats_precision + 7
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

    sharding.print("Started timestepping at", datetime.now())

    # --- Main time-stepping loop ---------------------------------------------
    while (
        t < t_stop
        and wall_time_now - wall_time_start < wall_time_stop
        and last_error < params.step.corrector_tolerance
    ):
        if it == params.init.it0 + 1:
            # Start the benchmark clock after the first (JIT-heavy) iteration
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

            if py_idx == params.outs.nstats:
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

        # Fused predictor + all corrector iterations (single JIT scope).
        state, error, c = predict_and_fully_correct(state)

        if params.phys.system in periodic_systems:
            # Divergence correction and mean-mode zeroing
            state = correct_velocity(state)

        t += params.step.dt
        it += 1

        # Periodic snapshot save
        if (
            params.outs.it_snapshot is not None
            and it % params.outs.it_snapshot == 0
            and it > params.init.it0
        ):
            from .snapshot import save_snapshot

            save_snapshot(state, t, it, f"snapshot_it{it:09d}")

        last_error = error
        c_int = int(c)
        last_c = c_int
        c_tot += c_int

        if it > params.init.it0:
            # 2 RHS evals per predict_and_correct + 1 per corrector iteration
            rhs_tot += c_int + 2

        wall_time_now = perf_counter_ns()

    # --- Post-processing -----------------------------------------------------
    if last_error > params.step.corrector_tolerance:
        sharding.print(
            f"Corrector failed to converge at t={t}, it={it}, c={last_c}, "
            f"with error = {last_error:.3e}."
        )

    sharding.print("Stopped timestepping at", datetime.now())

    # Final snapshot (if snapshotting is active and we stepped)
    if params.outs.it_snapshot is not None and it > params.init.it0:
        from .snapshot import save_snapshot

        save_snapshot(state, t, it, f"snapshot_it{it:09d}")

    wall_time_now = perf_counter_ns()
    alive_time = ns_to_s * (wall_time_now - wall_time_start)
    sharding.print(f"Job has been alive for {alive_time:.2f}s.")
    if it > params.init.it0 + 1:
        wall_time = ns_to_s * (wall_time_now - bench_start)
        wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
        wall_time_per_rhs = wall_time / rhs_tot

        # Final diagnostic output
        stats = get_stats(state)
        c_per_it = c_tot / (it - params.init.it0)

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

    # Flush remaining buffered stats
    if (
        params.outs.it_stats is not None
        and py_idx > 0
        and sharding.main_device
    ):
        _flush_stats(
            buffer,
            py_idx,
            ts_buf,
            stats_file,
            p,
            col_width,
        )


if __name__ == "__main__":
    print("Alive at", datetime.now(), flush=True)
    wall_time_start = perf_counter_ns()
    params_cli = CliApp.run(CLIParameters)

    params_file = Path("parameters.toml")
    params_from_disk = False
    if Path.is_file(params_file):
        params_from_disk = True
        params_in = read_parameters(params_file)
        update_parameters(params_in)

    update_parameters(params_cli)
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
        if params_from_disk:
            print(
                "Loaded parameters.toml, "
                "which override the default parameters. "
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
