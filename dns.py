import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)  # use 64 bit floating point
jax.config.update("jax_platforms", "cpu")  # stick to CPUs for now
jax.distributed.initialize()

import transform
from parameters import DT, I_START, STEPTOL, T_START
from stats import get_stats
from timestep import timestep

# TODO: JIT all the things

""" 
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_DYNAMIC="FALSE"
export OMP_DYNAMIC="FALSE"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 
"""


def dns():

    it = I_START
    t = T_START

    # # Start from the laminar state to test
    # import velocity
    # from sharding import MESH, RANK
    # velocity_spec = velocity.get_laminar()
    # velocity_phys = transform.spec_to_phys_vector(velocity_spec)

    # Start from a periodic solution
    from pathlib import Path

    import f90nml
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from sharding import MESH, RANK

    invariants = Path("/mnt/c/Users/gokhan/Seafile/projects/dnsjax/invariants")
    number = "01"
    params = f90nml.read(invariants / number / "parameters.in")
    period = params["invariants"]["periods"][0]
    velocity_phys = jax.device_put(
        jnp.load(invariants / number / "u0.npz")["velocity_phys"],
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )
    velocity_spec = transform.phys_to_spec_vector(velocity_phys)

    # from jax_array_info import sharding_vis

    # while True:
    stats_all = []
    while t < period:
        stats = get_stats(velocity_spec)
        if RANK == 0 and it % 10 == 0:
            print(f"t = {t:.6f}", *[f"{x}={y:.6e}" for x, y in stats.items()])
        if RANK == 0:
            stats_all.append(jnp.array([t, *stats.values()]))

        velocity_spec, velocity_phys, error, c = timestep(velocity_spec, velocity_phys)

        if error > STEPTOL:
            exit("Timestep did not converge")

        # sharding_vis(velocity_spec)
        # sharding_vis(velocity_phys)
        t += DT
        it += 1

    if RANK == 0:
        stats_all = jnp.array(stats_all)
        jnp.savez(invariants / number / "stats.npz", stats_all=stats_all)

    return


if __name__ == "__main__":
    dns()
