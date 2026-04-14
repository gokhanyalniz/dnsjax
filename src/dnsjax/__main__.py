#!/usr/bin/env python3
"""Entry point for the dnsjax DNS solver.

Execution proceeds in two phases:

1. **Initialisation** (module level, under ``if __name__ == "__main__"``):
   parse CLI arguments, load ``parameters.toml`` if present, configure
   JAX platform and distributed backend, print the final parameter set.

2. **Main loop** (:func:`main`):
   initialise velocity (from laminar or snapshot), then iterate:

   - Euler predictor + Crank-Nicolson corrector (:func:`predict_and_correct`)
   - Additional corrector iterations if needed (:func:`iterate_correction`)
   - Divergence correction + mean-mode zeroing (:func:`correct_velocity`)
   - Periodic diagnostic output (:func:`get_stats`)

   The loop terminates when the simulation time, wall-clock time, or
   corrector divergence criterion is reached.

Benchmarking
------------
The first time step is excluded from wall-clock statistics because it
includes JAX's JIT compilation overhead.  Additionally, the first call
to ``iterate_correction`` (if it occurs on the first step) is excluded
via the ``bench_delta`` accumulator.
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
    padded_res,
    params,
    periodic_systems,
    read_parameters,
    update_parameters,
)


def main() -> None:
    """Run the time-stepping loop after parameters and JAX are initialised."""
    from jax import numpy as jnp

    from .bench import ns_to_s, timers
    from .sharding import sharding

    # --- Flow dispatch -------------------------------------------------------
    if params.phys.system in periodic_systems:
        from .flows.monochromatic import (
            correct_velocity,
            get_stats,
            init_state,
            iterate_correction,
            predict_and_correct,
        )
    elif params.phys.system == "plane-couette":
        from .flows.plane_couette import (
            correct_velocity,
            get_stats,
            init_state,
            iterate_correction,
            predict_and_correct,
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
    bench_delta: int = 0  # accumulated JIT-compilation time to subtract
    corrector_compiled: bool = False
    last_error = 0
    last_c: int = 0
    norm_corrections: dict | None = {}

    # Warm-up call so that JIT compilation does not affect benchmarks
    stats = get_stats(state)

    sharding.print(
        f"t = {t:.2f}",
        *[f"{x}={y:.3e}" for x, y in stats.items()],
    )

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

        # Periodic diagnostic output
        if (
            params.outs.it_stats is not None
            and it % params.outs.it_stats == 0
            and it > params.init.it0
        ):
            stats = get_stats(state)
            c_per_it = c_tot / (it - params.init.it0)

            sharding.print(
                f"t = {t:.2f}",
                *[f"{x}={y:.3e}" for x, y in stats.items()],
                f"c/it = {c_per_it:.2f}",
                f"err = {last_error:.3e}",
                *[f"{x}={y:.3e}" for x, y in norm_corrections.items()]
                if norm_corrections is not None
                else "",
            )

        # Euler predictor + one Crank-Nicolson corrector
        state_prev = state
        state, rhs_no_lapl, error = predict_and_correct(
            state_prev,
        )
        c = 0

        # Additional corrector iterations until convergence
        while (
            error > params.step.corrector_tolerance
            and c < params.step.max_corrector_iterations
        ):
            if not corrector_compiled:
                # Exclude the first corrector JIT compilation from benchmarks
                bench_delta_start = perf_counter_ns()

            state, rhs_no_lapl, error = iterate_correction(
                state_prev,
                state,
                rhs_no_lapl,
            )
            c += 1

            if not corrector_compiled:
                bench_delta_stop = perf_counter_ns()
                bench_delta += bench_delta_stop - bench_delta_start
                rhs_tot -= 1
                corrector_compiled = True

        # Divergence correction and mean-mode zeroing
        state, norm_corrections = correct_velocity(
            state,
        )

        t += params.step.dt
        it += 1
        last_error = error
        last_c = c
        c_tot += c

        if it > params.init.it0:
            # 2 RHS evals per predict_and_correct + 1 per corrector iteration
            rhs_tot += c + 2

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
        wall_time = ns_to_s * (wall_time_now - bench_delta - bench_start)
        wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
        wall_time_per_rhs = wall_time / rhs_tot

        # Final diagnostic output
        stats = get_stats(state)
        c_per_it = c_tot / (it - params.init.it0)

        sharding.print(
            f"t = {t:.2f}",
            *[f"{x}={y:.3e}" for x, y in stats.items()],
            f"c/it = {c_per_it:.2f}",
            f"err = {last_error:.3e}",
            *[f"{x}={y:.3e}" for x, y in norm_corrections.items()]
            if norm_corrections is not None
            else "",
        )

        if params.debug.time_functions and main_device:
            pp(timers, sort_dicts=True)

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
            "Running with the effective resolution:",
            padded_res.nx_padded,
            padded_res.ny_padded,
            padded_res.nz_padded,
            flush=True,
        )

    main()

    print("Shutdown at", datetime.now(), flush=True)
    sys.exit(0)
