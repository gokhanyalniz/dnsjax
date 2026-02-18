#!/usr/bin/env python3
from pprint import pp
from time import perf_counter_ns

import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import bench
import transform
import velocity
from bench import ns_to_s
from parameters import params
from sharding import MESH, N_DEVICES, RANK
from stats import get_stats
from timestep import timestep

"""
Run the following exports before running this script, 
especially when running on multiple devices,
to make sure libraries don't spawn multiple threads of their own:

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_DYNAMIC="FALSE"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 
"""


def dns():

    it = params.init.it0
    t = params.init.t0

    t_stop = (
        jnp.inf if params.stop.max_sim_time < 0 else params.stop.max_sim_time
    )

    if params.init.start_from_laminar:
        velocity_spec = velocity.get_laminar()

    elif params.init.snapshot is not None:
        velocity_phys = jax.device_put(
            jnp.load(params.init.snapshot)["velocity_phys"].astype(
                jnp.complex128
            ),
            NamedSharding(MESH, P(None, "Z", "X", None)),
        )
        velocity_spec = transform.phys_to_spec_vector(velocity_phys)

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
            if RANK == 0:
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
            if RANK == 0:
                print("Timestep did not converge")
            return

        t += params.step.dt
        it += 1

    stop = perf_counter_ns()
    wall_time = ns_to_s * (stop - start)
    wall_time_per_sim_time = wall_time / (t - dt_first - params.init.t0)
    wall_time_per_rhs = wall_time / rhs_tot

    if RANK == 0:
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


if __name__ == "__main__":
    dns()
    jax.distributed.shutdown()
