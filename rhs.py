from dataclasses import dataclass

from jax import numpy as jnp

from operators import (
    phys_to_spec,
    spec_to_phys,
)
from parameters import params
from sharding import get_zeros, sharding


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

# Given 3x3 symmetric matrix M, entries M_{ij} will be used
# This is for the nonlinear term.
n_to_symi = jnp.array([0, 0, 0, 1, 1, 2], dtype=int)
n_to_symj = jnp.array([0, 1, 2, 1, 2, 2], dtype=int)
symij_to_n = jnp.zeros((3, 3), dtype=int)

for n in range(6):
    i = n_to_symi[n]
    j = n_to_symj[n]
    symij_to_n = symij_to_n.at[i, j].set(n)
    symij_to_n = symij_to_n.at[j, i].set(n)


def get_nonlin(velocity_spec):

    velocity_phys = spec_to_phys(velocity_spec)  # 3 FFTs

    nonlin_phys = get_zeros(
        shape=(6, *velocity_phys.shape[1:]),
        dtype=velocity_phys.dtype,
        in_sharding=sharding.phys_vector_shard,
        out_sharding=sharding.phys_vector_shard,
    )

    for i in range(6):
        nonlin_phys = nonlin_phys.at[i].set(
            velocity_phys[n_to_symi[i]] * velocity_phys[n_to_symj[i]]
        )

    # Basdevant optimization: https://doi.org/10.1016/0021-9991(83)90064-5
    # Save on one FFT by the tracelessness of the nonlinear term
    # The trace is effectively moved to pressure. If "true" pressure is ever
    # needed, this needs to be taken into account.

    # Compute trace in-place on symij_to_n[2, 2]
    nonlin_phys = nonlin_phys.at[symij_to_n[2, 2]].add(
        nonlin_phys[symij_to_n[0, 0]] + nonlin_phys[symij_to_n[1, 1]]
    )

    # No need to update (2,2), it's not used
    nonlin_phys = nonlin_phys.at[
        (symij_to_n[0, 0], symij_to_n[1, 1]),
    ].subtract(nonlin_phys[symij_to_n[2, 2]] / 3)

    nonlin = get_zeros(
        shape=(6, *velocity_spec.shape[1:]),
        dtype=velocity_spec.dtype,
        in_sharding=sharding.spec_vector_shard,
        out_sharding=sharding.spec_vector_shard,
    )

    nonlin = nonlin.at[:5].set(phys_to_spec(nonlin_phys[:5]))
    # Basdevant: Get the 5th element from tracelessness
    nonlin = nonlin.at[5].set(
        -(nonlin[symij_to_n[0, 0]] + nonlin[symij_to_n[1, 1]])
    )

    return sharding.constrain_spec_vector(nonlin)


def get_rhs_no_lapl(
    velocity_spec,
    kvec,
    inv_lapl,
):

    nonlin = get_nonlin(velocity_spec)

    rhs_no_lapl = get_zeros(
        shape=velocity_spec.shape,
        dtype=velocity_spec.dtype,
        in_sharding=sharding.spec_vector_shard,
        out_sharding=sharding.spec_vector_shard,
    )

    lapl_pressure = get_zeros(
        shape=inv_lapl.shape,
        dtype=velocity_spec.dtype,
        in_sharding=sharding.spec_scalar_shard,
        out_sharding=sharding.spec_scalar_shard,
    )

    for i in range(3):
        # rhs_no_lapl[i] keeps -dj_uiuj at this stage
        rhs_no_lapl = rhs_no_lapl.at[i].set(
            -jnp.sum(
                1j
                * kvec
                * nonlin[
                    (symij_to_n[i, 0], symij_to_n[i, 1], symij_to_n[i, 2]),
                ],
                axis=0,
            )
        )

        lapl_pressure = lapl_pressure.at[...].add(
            1j * kvec[i] * rhs_no_lapl[i]
        )

    # Add pressure gradient
    rhs_no_lapl = rhs_no_lapl.at[...].add(
        -1j * kvec * inv_lapl * lapl_pressure
    )

    # Add forcing
    if force.on:
        rhs_no_lapl = rhs_no_lapl.at[force.forced_modes].add(
            force.unit_force * force.amplitude
        )

    return sharding.constrain_spec_vector(rhs_no_lapl)
