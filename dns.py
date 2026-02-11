#!/usr/bin/env python3
from pprint import pp
from time import perf_counter_ns

import jax
from jax import numpy as jnp

import bench
from bench import ns_to_s
from parameters import DT, I_START, STEPTOL, T_START, T_STOP, TIME_FUNCTIONS
from sharding import NP
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

IC_LAMINAR = False
IC_PERIODIC = True


def dns():

    it = I_START
    t = T_START

    t_stop = jnp.inf if T_STOP < 0 else T_STOP

    if IC_LAMINAR:
        # Start from the laminar state
        import transform
        import velocity
        from sharding import MESH, RANK

        velocity_spec = velocity.get_laminar()

    elif IC_PERIODIC:
        # Start from a periodic solution
        from pathlib import Path

        import f90nml
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        import transform
        from sharding import MESH, RANK

        invariants = Path(
            "/mnt/c/Users/gokhan/Seafile/projects/dnsjax/invariants"
        )
        number = "01"
        params = f90nml.read(invariants / number / "parameters.in")
        period = params["invariants"]["periods"][0]
        velocity_phys = jax.device_put(
            jnp.load(invariants / number / "u0.npz")["velocity_phys"].astype(
                jnp.complex128
            ),
            NamedSharding(MESH, P(None, "Z", "X", None)),
        )
        velocity_spec = transform.phys_to_spec_vector(velocity_phys)

        t_stop = period
    else:
        print("Need to provide an initial condition.")
        return

    # Call once now not to affect benchmarks later
    stats = get_stats(velocity_spec)

    stats_all = []
    rhs_tot = 0
    while t < t_stop:
        if it == I_START + 1:
            # Ignore the first hit, probably subject to JIT compilation
            dt_first = DT
            start = perf_counter_ns()

        if it % 100 == 0:
            if it != I_START:  # If so, already called above
                stats = get_stats(velocity_spec)
            if RANK == 0:
                print(
                    f"t = {t:.2f}",
                    *[f"{x}={y:.6e}" for x, y in stats.items()],
                )
                stats_all.append(jnp.array([t, *stats.values()]))

        velocity_spec, error, c = timestep(velocity_spec)

        if it == I_START + 1:
            # Ignore the first hit, probably subject to JIT compilation
            rhs_tot += c + 1  # 1 predictor and c corrector steps per timestep

        if error > STEPTOL:
            if RANK == 0:
                print("Timestep did not converge")
            jax.distributed.shutdown()

        t += DT
        it += 1

    if RANK == 0:
        if IC_PERIODIC:
            stats_all = jnp.array(stats_all)
            jnp.savez(invariants / number / "stats.npz", stats_all=stats_all)
        if TIME_FUNCTIONS:
            pp(bench.timers)

    stop = perf_counter_ns()
    runtime = ns_to_s * (stop - start)
    if RANK == 0:
        print(
            f"Ran for {runtime:.2f} s with {NP} processes,",
            f"{NP * runtime:.3e} NP x s:",
            f"{runtime / (t - dt_first - T_START):.3e} s/t,",
            f"{runtime / rhs_tot:.3e} s/rhs,",
            f"{NP * runtime / (t - dt_first - T_START):.3e} NP x s/t,",
            f"{NP * runtime / rhs_tot:.3e} NP x s/rhs.",
        )


if __name__ == "__main__":
    dns()
    jax.distributed.shutdown()
