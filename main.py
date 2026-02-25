#!/usr/bin/env python3
import os
from datetime import datetime
from pathlib import Path
from pprint import pp

from pydantic_settings import CliApp

from parameters import (
    CLIParameters,
    padded_res,
    params,
    read_parameters,
    update_parameters,
)


def main():

    from time import perf_counter_ns

    from jax import numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    import bench
    from operators import fourier, phys_to_spec
    from sharding import MESH, N_DEVICES, complex_type
    from stats import get_stats
    from timestep import stepper, timestep
    from velocity import get_laminar

    wall_time_stop = (
        jnp.inf
        if params.stop.max_wall_time is None
        else int(params.stop.max_wall_time.total_seconds() / bench.ns_to_s)
    )

    wall_time_start = perf_counter_ns()

    it = params.init.it0
    t = params.init.t0

    t_stop = (
        jnp.inf
        if params.stop.max_sim_time is None
        else params.stop.max_sim_time
    )

    if params.init.start_from_laminar:
        velocity_spec = get_laminar()

    elif params.init.snapshot is not None:
        velocity_phys = jax.device_put(
            jnp.load(params.init.snapshot)["velocity_phys"].astype(
                complex_type
            ),
            NamedSharding(MESH, P(None, "Z", "X", None)),
        )
        velocity_spec = phys_to_spec(velocity_phys, fourier.DEALIAS)

    else:
        main_print("Need to provide an initial condition.")
        return

    # Call once now not to affect benchmarks later
    stats = get_stats(
        velocity_spec,
        fourier.LAPL,
        fourier.DEALIAS,
    )

    # Useful to know the starting stats
    main_print(
        f"t = {t:.2f}",
        *[f"{x}={y:.6e}" for x, y in stats.items()],
    )

    rhs_tot = 0
    dt_first = params.step.dt
    wall_time_now = perf_counter_ns()
    error = 0

    main_print("Started timestepping at", datetime.now())

    while (
        t < t_stop
        and wall_time_now - wall_time_start < wall_time_stop
        and error < params.step.corrector_tolerance
    ):
        if it == params.init.it0 + 1:
            # Ignore the first hit, probably subject to JIT compilation
            bench_start = perf_counter_ns()

            main_print("First iteration over at", datetime.now())

        if (
            params.outs.it_stats is not None
            and it % params.outs.it_stats == 0
            and it != params.init.it0
        ):
            stats = get_stats(
                velocity_spec,
                fourier.LAPL,
                fourier.DEALIAS,
            )
            main_print(
                f"t = {t:.2f}",
                *[f"{x}={y:.6e}" for x, y in stats.items()],
                f"c/it = {rhs_tot / (it - 1 - params.init.it0):.2f}",
            )

        velocity_spec, error, c = timestep(
            velocity_spec,
            fourier.NABLA,
            fourier.INV_LAPL,
            fourier.ZERO_MEAN,
            fourier.DEALIAS,
            stepper.LDT_1,
            stepper.ILDT_2,
        )

        if it > params.init.it0:
            # Ignore the first hit, probably subject to JIT compilation
            rhs_tot += c + 1  # 1 predictor and c corrector steps per timestep

        t += params.step.dt
        it += 1

        wall_time_now = perf_counter_ns()

    if error > params.step.corrector_tolerance:
        main_print(
            f"Corrector failed to converge at t={t}, it={it}, c={c}, "
            f"with error = {error:.3e}."
        )

    main_print("Stopped timestepping at", datetime.now())

    bench_stop = perf_counter_ns()
    wall_time = bench.ns_to_s * (bench_stop - bench_start)
    wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
    wall_time_per_rhs = wall_time / rhs_tot

    # Useful to final stats
    stats = get_stats(
        velocity_spec,
        fourier.LAPL,
        fourier.DEALIAS,
    )
    main_print(
        f"t = {t:.2f}",
        *[f"{x}={y:.6e}" for x, y in stats.items()],
    )

    if params.debug.time_functions and main_device:
        pp(bench.timers)

    if N_DEVICES > 1:
        main_print(
            f"Ran for {wall_time:.2f} s with {N_DEVICES} devices,",
            f"{wall_time_per_sim_time:.3e} s/t,",
            f"{wall_time_per_rhs:.3e} s/rhs,",
            f"{N_DEVICES * wall_time:.3e} NP x s:",
            f"{N_DEVICES * wall_time_per_sim_time:.3e} NP x s/t,",
            f"{N_DEVICES * wall_time_per_rhs:.3e} NP x s/rhs.",
        )
    else:
        main_print(
            f"Ran for {wall_time:.2f} s with 1 device.",
            f"{wall_time_per_sim_time:.3e} s/t,",
            f"{wall_time_per_rhs:.3e} s/rhs.",
        )


if __name__ == "__main__":
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
    # This will be a default in a later release of JAX:
    # jax.config.update("jax_use_simplified_jaxpr_constants", True)
    jax.distributed.initialize()

    rank = jax.process_index()
    main_device = False
    if rank == 0:
        main_device = True

    def main_print(*args, **kwargs):
        if main_device:
            print(*args, **kwargs, flush=True)

    main_print("JAX initialized at", datetime.now())
    if params_from_disk:
        main_print(
            "Loaded parameters.toml, "
            "which override the default parameters. "
            "Command-line arguments will further override "
            "the loaded parameters.",
        )
    else:
        main_print(
            "Loaded the default parameters, "
            "as parameters.toml was not found. "
            "Command-line arguments will further override "
            "the default parameters.",
        )
    main_print("Final working parameters:")
    if main_device:
        pp(params.model_dump())

    main()

    jax.distributed.shutdown()
    main_print("JAX shutdown at", datetime.now())
