from dataclasses import dataclass

from jax import numpy as jnp

from operators import (
    phys_to_spec,
    spec_to_phys,
)
from parameters import params
from sharding import sharding


@dataclass
class Force:
    # Physics
    if params.phys.forcing in ["kolmogorov", "waleffe"]:
        if params.res.ny < 4:
            sharding.print(
                f"{params.phys.forcing} cannot work "
                "with less than 4 modes in y."
            )
            sharding.exit(code=1)

        on = True
        amplitude = jnp.pi**2 / (4 * params.phys.re)
        laminar_amplitude = 1

        ic_f = 0  # Forced component
        qf = 1  # Forcing harmonic

        forced_modes = ((ic_f, ic_f), (qf, -qf), (0, 0), (0, 0))

        if params.phys.forcing == "kolmogorov":
            unit_force = jnp.array([-0.5j, 0.5j], dtype=sharding.complex_type)
        elif params.phys.forcing == "waleffe":
            unit_force = jnp.array([0.5, 0.5], dtype=sharding.complex_type)

            sharding.print("Waleffe flow is not yet implemented.")
            sharding.exit(code=1)

    else:
        on = False
        unit_force = None


force = Force()


def get_rhs_no_lapl(
    velocity_spec,
    kvec,
    inv_lapl,
):

    velocity_phys = spec_to_phys(velocity_spec)  # 3 FFTs

    vorticity_phys = spec_to_phys(
        1j * jnp.cross(kvec, velocity_spec, axis=0)
    )  # 3 FFTs

    nonlin = phys_to_spec(
        jnp.cross(velocity_phys, vorticity_phys, axis=0)
    )  # 3 FFTs

    # Poisson problem for pressure
    lapl_pressure = jnp.sum(1j * kvec * nonlin, axis=0)

    # Add pressure gradient
    rhs_no_lapl = nonlin - 1j * kvec * inv_lapl * lapl_pressure

    # Add forcing
    if force.on:
        rhs_no_lapl = rhs_no_lapl.at[force.forced_modes].add(
            force.unit_force * force.amplitude,
            out_sharding=sharding.spec_vector_shard,
        )

    return rhs_no_lapl
