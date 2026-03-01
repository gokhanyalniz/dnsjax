from dataclasses import dataclass
from functools import partial

import jax
from jax import numpy as jnp
from jax.sharding import explicit_axes

from operators import (
    fourier,
    phys_to_spec,
    spec_to_phys,
)
from parameters import params
from sharding import sharding
from velocity import (
    get_zero_scalar,
    get_zero_vector,
)


@dataclass
class Force:
    # Physics
    if params.phys.forcing in ["kolmogorov", "waleffe"]:
        on = True
        amplitude = jnp.pi**2 / (4 * params.phys.Re)

        ic_f = 0  # Forced component
        qf = 1  # Forcing harmonic
        kf = 2 * jnp.pi * qf / params.geo.Ly

        fp = jnp.nonzero(
            (fourier.qx == 0) & (fourier.qy == qf) & (fourier.qz == 0),
            size=1,
        )
        fn = jnp.nonzero(
            (fourier.qx == 0) & (fourier.qy == -qf) & (fourier.qz == 0),
            size=1,
        )
        fp = (int(i[0]) for i in fp)
        fn = (int(i[0]) for i in fn)

        forced_modes = tuple(zip(fp, fn, strict=True))

        if params.phys.forcing == "kolmogorov":
            unit = (-1j * 0.5, 1j * 0.5)
        elif params.phys.forcing == "waleffe":
            unit = (0.5, 0.5)
            jax.distributed.shutdown()
            exit("Waleffe flow is not yet implemented.")
    else:
        on = False

    if on:

        @partial(explicit_axes, axes=sharding.axis_names)
        def get_laminar_state():
            laminar_state = jnp.zeros(
                sharding.spec_shape,
                dtype=sharding.complex_type,
                out_sharding=sharding.scalar_spec_shard,
            )
            return laminar_state

        laminar_state = get_laminar_state(
            in_sharding=sharding.scalar_spec_shard
        )

        laminar_state = laminar_state.at[forced_modes].add(jnp.array(unit))
    else:
        laminar_state = 0


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


# @jit(out_shardings=sharding.spec_shard)
def get_nonlin(velocity_spec, active_modes):

    velocity_phys = spec_to_phys(velocity_spec)  # 3 FFTs

    nonlin_phys = get_zero_vector(
        shape=(6, *velocity_phys.shape[1:]),
        dtype=velocity_phys.dtype,
        in_sharding=sharding.phys_shard,
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

    nonlin = get_zero_vector(
        shape=(6, *velocity_spec.shape[1:]),
        dtype=velocity_spec.dtype,
        in_sharding=sharding.spec_shard,
    )

    nonlin = nonlin.at[:5].set(phys_to_spec(nonlin_phys[:5], active_modes))
    # Basdevant: Get the 5th element from tracelessness
    nonlin = nonlin.at[5].set(
        -(nonlin[symij_to_n[0, 0]] + nonlin[symij_to_n[1, 1]])
    )

    return nonlin


# @jit(out_shardings=sharding.spec_shard)
def get_rhs_no_lapl(
    velocity_spec,
    laminar_state,
    nabla,
    inv_lapl,
    active_modes,
):

    nonlin = get_nonlin(velocity_spec, active_modes)

    rhs_no_lapl = get_zero_vector(
        shape=velocity_spec.shape,
        dtype=velocity_spec.dtype,
        in_sharding=sharding.spec_shard,
    )

    lapl_pressure = get_zero_scalar(
        shape=inv_lapl.shape,
        dtype=velocity_spec.dtype,
        in_sharding=sharding.scalar_spec_shard,
    )

    for i in range(3):
        minus_dj_uiuj = -jnp.sum(
            nabla
            * nonlin[(symij_to_n[i, 0], symij_to_n[i, 1], symij_to_n[i, 2]),],
            axis=0,
        )

        rhs_no_lapl = rhs_no_lapl.at[i].set(minus_dj_uiuj)
        lapl_pressure = lapl_pressure.at[...].add(nabla[i] * minus_dj_uiuj)

    # Add pressure gradient
    rhs_no_lapl = rhs_no_lapl.at[...].add(-nabla * inv_lapl * lapl_pressure)

    # Add forcing
    if force.on:
        rhs_no_lapl = rhs_no_lapl.at[force.ic_f].add(
            laminar_state * force.amplitude
        )

    return rhs_no_lapl
