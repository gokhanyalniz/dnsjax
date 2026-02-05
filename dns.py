import jax

jax.config.update("jax_enable_x64", True)  # use 64 bit floating point
jax.config.update("jax_platforms", "cpu")  # stick to CPUs for now
jax.distributed.initialize()

import transform
import vfield
from parameters import DT, I_START, T_START
from timestep import timestep


def dns():

    it = I_START
    t = T_START

    # Start from the laminar state to test
    velocity_spec = vfield.get_laminar()
    velocity_phys = transform.spec_to_phys_vector(velocity_spec)

    while True:
        norm_phys = vfield.get_inprod_phys(velocity_phys, velocity_phys)
        norm_spec = vfield.get_norm2(velocity_spec)
        print(f"t = {t:.6f}", f"{norm_spec:.3f}", f"{norm_phys:.6f}")

        velocity_spec, velocity_phys = timestep(velocity_spec, velocity_phys)
        t += DT
        it += 1

    return


if __name__ == "__main__":
    dns()
