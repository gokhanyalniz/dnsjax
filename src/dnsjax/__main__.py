#!/usr/bin/env python3
"""Entry point for the dnsjax DNS solver.

Execution proceeds in two phases:

1. **Initialisation** (module level, under ``if __name__ == "__main__"``):
   parse CLI arguments, load ``parameters.toml`` if present, configure
   JAX platform and distributed backend, print the final parameter set.

2. **Main loop** (:func:`main`):
   initialise velocity (from laminar or snapshot), then iterate:

   - Fused predictor + corrector loop
     (:func:`predict_and_fully_correct`)
   - Divergence correction + mean-mode zeroing for
     triply-periodic flows (:func:`correct_velocity`)
   - Periodic diagnostic output (:func:`get_stats`)

   The loop terminates when the simulation time, wall-clock
   time, or corrector divergence criterion is reached.

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
    ns_to_s,
    padded_res,
    params,
    periodic_systems,
    read_parameters,
    update_parameters,
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
