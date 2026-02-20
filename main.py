#!/usr/bin/env python3
import os

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["NPROC"] = "1"

from pathlib import Path
from pprint import pp

import jax
from pydantic_settings import CliApp

from parameters import (
    CLIParameters,
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
    from sharding import MESH, N_DEVICES, complex_type
    from stats import get_stats
    from timestep import timestep
    from transform import phys_to_spec_vector
    from velocity import get_laminar

    rank = jax.process_index()
    main_device = False
    if rank == 0:
        main_device = True

    it = params.init.it0
    t = params.init.t0

    t_stop = (
        jnp.inf if params.stop.max_sim_time < 0 else params.stop.max_sim_time
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
        velocity_spec = phys_to_spec_vector(velocity_phys)

    else:
        print("Need to provide an initial condition.")
        return

    # Call once now not to affect benchmarks later
    stats = get_stats(velocity_spec)

    stats_all = []
    rhs_tot = 0
    dt_first = params.step.dt

    while t < t_stop:
        if it == params.init.it0 + 1:
            # Ignore the first hit, probably subject to JIT compilation
            start = perf_counter_ns()

        if it % params.outs.it_stats == 0:
            if it != params.init.it0:  # If so, already called above
                stats = get_stats(velocity_spec)
            if main_device:
                print(
                    f"t = {t:.2f}",
                    *[f"{x}={y:.6e}" for x, y in stats.items()],
                )
                # Need to array this properly later
                stats_all.append(jnp.array([t, *stats.values()]))

        velocity_spec, error, c = timestep(velocity_spec)

        if it == params.init.it0 + 1:
            # Ignore the first hit, probably subject to JIT compilation
            rhs_tot += c + 1  # 1 predictor and c corrector steps per timestep

        if error > params.step.corrector_tolerance:
            if main_device:
                print("Timestep did not converge")
            return

        t += params.step.dt
        it += 1

    stop = perf_counter_ns()
    wall_time = bench.ns_to_s * (stop - start)
    wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
    wall_time_per_rhs = wall_time / rhs_tot

    if main_device:
        if len(stats_all) > 0:
            jnp.savez("stats.npz", stats_all=stats_all)
        if params.debug.time_functions:
            pp(bench.timers)

        if N_DEVICES > 1:
            print(
                f"Ran for {wall_time:.2f} s with {N_DEVICES} devices,",
                f"{wall_time_per_sim_time:.3e} s/t,",
                f"{wall_time_per_rhs:.3e} s/rhs,",
                f"{N_DEVICES * wall_time:.3e} NP x s:",
                f"{N_DEVICES * wall_time_per_sim_time:.3e} NP x s/t,",
                f"{N_DEVICES * wall_time_per_rhs:.3e} NP x s/rhs.",
            )
        else:
            print(
                f"Ran for {wall_time:.2f} s with 1 device.",
                f"{wall_time / wall_time_per_sim_time:.3e} s/t,",
                f"{wall_time_per_rhs:.3e} s/rhs.",
            )

    jax.distributed.shutdown()


if __name__ == "__main__":
    params_cli = CliApp.run(CLIParameters)

    params_file = Path("parameters.toml")
    params_from_disk = False
    if Path.is_file(params_file):
        params_from_disk = True
        params_in = read_parameters(params_file)
        update_parameters(params_in)

    update_parameters(params_cli)

    jax.config.update("jax_enable_x64", params.res.double_precision)
    jax.config.update("jax_platforms", params.dist.platform)
    jax.distributed.initialize()
    rank = jax.process_index()

    if rank == 0:
        if params_from_disk:
            print(
                "Loaded parameters.toml, "
                "which override the default parameters. "
                "Command-line arguments will further override "
                "the loaded parameters."
            )
        else:
            print(
                "Loaded the default parameters, "
                "as parameters.toml was not found. "
                "Command-line arguments will further override "
                "the default parameters."
            )
        print("Final working parameters:")
        pp(params.model_dump())

    main()
