from dataclasses import dataclass
from functools import partial

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import explicit_axes

from operators import (
    fourier,
    phys_to_spec,
    spec_to_phys,
)
from parameters import padded_res, params
from sharding import sharding


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

        @partial(explicit_axes, axes=("Z", "X"))
        def get_laminar_state():
            laminar_state = jnp.zeros(
                (
                    padded_res.Nz_padded,
                    padded_res.Nx_padded,
                    padded_res.Ny_padded,
                ),
                dtype=sharding.complex_type,
                out_sharding=P("Z", "X", None),
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


@jit
def get_nonlin_phys(velocity_spec):

    velocity_phys = spec_to_phys(velocity_spec)  # 3 FFTs

    @partial(explicit_axes, axes=("Z", "X"))
    def get_empty_nonlin_phys():
        nonlin_phys = jnp.zeros(
            (
                6,
                padded_res.Ny_padded,
                padded_res.Nz_padded,
                padded_res.Nx_padded,
            ),
            dtype=sharding.complex_type,
            out_sharding=P(None, "Z", "X", None),
        )
        return nonlin_phys

    nonlin_phys = get_empty_nonlin_phys(in_sharding=sharding.phys_shard)

    def set_nonlin_phys(i, val):
        return val.at[i, ...].set(
            velocity_phys[n_to_symi[i]] * velocity_phys[n_to_symj[i]]
        )

    nonlin_phys = lax.fori_loop(
        0, 6, set_nonlin_phys, nonlin_phys, unroll=True
    )

    # Basdevant optimization: https://doi.org/10.1016/0021-9991(83)90064-5
    # Save on one FFT by the tracelessness of the nonlinear term
    # The trace is effectively moved to pressure. If "true" pressure is ever
    # needed, this needs to be taken into account.

    trace = jnp.sum(
        nonlin_phys[(symij_to_n[0, 0], symij_to_n[1, 1], symij_to_n[2, 2]),],
        axis=0,
        dtype=sharding.float_type,
    )

    # No need to update (2,2), it's not used
    nonlin_phys = nonlin_phys.at[
        (symij_to_n[0, 0], symij_to_n[1, 1]),
    ].subtract(trace / 3)

    # Pass the whole array to reuse memory
    return jax.lax.with_sharding_constraint(nonlin_phys, sharding.phys_shard)


@jit(donate_argnums=0)
def get_nonlin_spec(nonlin_phys, dealias):

    @partial(explicit_axes, axes=("Z", "X"))
    def get_empty_nonlin_spec():
        nonlin = jnp.zeros(
            (
                6,
                padded_res.Nz_padded,
                padded_res.Nx_padded,
                padded_res.Ny_padded,
            ),
            dtype=sharding.complex_type,
            out_sharding=P(None, "Z", "X", None),
        )
        return nonlin

    nonlin = get_empty_nonlin_spec(in_sharding=sharding.spec_shard)

    nonlin = nonlin.at[:5].set(phys_to_spec(nonlin_phys[:5], dealias))
    # Basdevant: Get the 5th element from tracelessness
    nonlin = nonlin.at[5].set(
        -(nonlin[symij_to_n[0, 0]] + nonlin[symij_to_n[1, 1]])
    )

    return jax.lax.with_sharding_constraint(nonlin, sharding.spec_shard)


@jit
def get_nonlin(velocity_spec, dealias):
    return jax.lax.with_sharding_constraint(
        get_nonlin_spec(get_nonlin_phys(velocity_spec), dealias),
        sharding.spec_shard,
    )


@jit
def get_rhs_no_lapl(
    velocity_spec,
    laminar_state,
    nabla,
    inv_lapl,
    dealias,
):

    nonlin = get_nonlin(velocity_spec, dealias)

    @partial(explicit_axes, axes=("Z", "X"))
    def get_empty_advect():
        advect = jnp.zeros(
            (
                3,
                padded_res.Nz_padded,
                padded_res.Nx_padded,
                padded_res.Ny_padded,
            ),
            dtype=sharding.complex_type,
            out_sharding=P(None, "Z", "X", None),
        )
        return advect

    advect = get_empty_advect(in_sharding=sharding.spec_shard)

    def set_advect(i, val):
        return val.at[i, ...].set(
            -jnp.sum(
                nabla
                * nonlin[
                    (symij_to_n[0, i], symij_to_n[1, i], symij_to_n[2, i]),
                ],
                axis=0,
            )
        )

    advect = lax.fori_loop(0, 3, set_advect, advect, unroll=True)

    rhs_no_lapl = advect - nabla * inv_lapl * jnp.sum(nabla * advect, axis=0)
    if force.on:
        rhs_no_lapl = rhs_no_lapl.at[force.ic_f].add(
            laminar_state * force.amplitude
        )

    return jax.lax.with_sharding_constraint(rhs_no_lapl, sharding.spec_shard)
