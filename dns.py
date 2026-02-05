import jax

jax.config.update("jax_enable_x64", True)  # use 64 bit floating point
jax.config.update("jax_platforms", "cpu")  # stick to CPUs for now
jax.distributed.initialize()

import transform
import velocity
from parameters import DT, I_START, T_START
from timestep import timestep
from stats import get_stats

# TODO: JIT all the things

def dns():

    it = I_START
    t = T_START

    # Start from the laminar state to test
    velocity_spec = velocity.get_laminar()
    velocity_phys = transform.spec_to_phys_vector(velocity_spec)

    while True:
        stats = get_stats(velocity_spec)
        print(f"t = {t:.6f}", *[f"{x}={y:.6f}" for x, y in stats.items()])

        velocity_spec, velocity_phys = timestep(velocity_spec, velocity_phys)
        t += DT
        it += 1

    return


if __name__ == "__main__":
    dns()
