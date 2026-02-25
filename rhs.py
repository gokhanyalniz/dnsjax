from dataclasses import dataclass

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bench import timer
from operators import (
    fourier,
    phys_to_spec,
    spec_to_phys,
)
from parameters import padded_res, params
from sharding import MESH, complex_type, float_type


@dataclass
class Force:
    # Physics
    if params.phys.forcing is not None:
        FORCING_AMPLITUDE = jnp.pi**2 / (4 * params.phys.Re)

        IC_F = 0  # Forced component
        QF = 1  # Forcing harmonic
        KF = 2 * jnp.pi * QF / params.geo.Ly

        FP = (
            IC_F,
            *(
                i[0]
                for i in jnp.nonzero(
                    (fourier.QX == 0) & (fourier.QY == QF) & (fourier.QZ == 0)
                )
            ),
        )
        FN = (
            IC_F,
            *(
                i[0]
                for i in jnp.nonzero(
                    (fourier.QX == 0) & (fourier.QY == -QF) & (fourier.QZ == 0)
                )
            ),
        )
        FORCING_MODES = tuple(zip(FP, FN, strict=True))

        if params.phys.forcing == "kolmogorov":
            FORCING_UNIT = jnp.array([-1j, 1j]) * 0.5
        elif params.phys.forcing == "waleffe":
            FORCING_UNIT = jnp.array([1, 1]) * 0.5
            jax.distributed.shutdown()
            exit("Waleffe flow is not yet implemented.")
    else:
        FORCING_MODES = jnp.array([])
        FORCING_UNIT = 0
        FORCING_AMPLITUDE = 1


force = Force()

# Given 3x3 symmetric matrix M, entries M_{ij} will be used
# This is for the nonlinear term.
ISYM = jnp.array([0, 0, 0, 1, 1, 2], dtype=int)
JSYM = jnp.array([0, 1, 2, 1, 2, 2], dtype=int)
NSYM = jnp.zeros((3, 3), dtype=int)

for n in range(6):
    i = ISYM[n]
    j = JSYM[n]
    NSYM = NSYM.at[i, j].set(n)
    NSYM = NSYM.at[j, i].set(n)


@jit
def get_nonlin_phys(velocity_spec):

    velocity_phys = spec_to_phys(velocity_spec)  # 3 FFTs

    nonlin_phys = jax.device_put(
        jnp.zeros(
            (
                6,
                padded_res.NY_PADDED,
                padded_res.NZ_PADDED,
                padded_res.NX_PADDED,
            ),
            dtype=complex_type,
        ),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )

    def set_nonlin_phys(i, val):
        return val.at[i, ...].set(
            velocity_phys[ISYM[i]] * velocity_phys[JSYM[i]]
        )

    nonlin_phys = lax.fori_loop(
        0, 6, set_nonlin_phys, nonlin_phys, unroll=True
    )

    # Basdevant optimization: https://doi.org/10.1016/0021-9991(83)90064-5
    # Save on one FFT by the tracelessness of the nonlinear term
    # The trace is effectively moved to pressure. If "true" pressure is ever
    # needed, this needs to be taken into account.

    trace = jnp.sum(
        nonlin_phys[(NSYM[0, 0], NSYM[1, 1], NSYM[2, 2]),],
        axis=0,
        dtype=float_type,
    )

    # No need to update (2,2), it's not used
    nonlin_phys = nonlin_phys.at[(NSYM[0, 0], NSYM[1, 1]),].subtract(trace / 3)

    # Pass the whole array to reuse memory
    return jax.lax.with_sharding_constraint(
        nonlin_phys, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@jit(donate_argnums=0, static_argnums=1)
def get_nonlin_spec(nonlin_phys, dealias):

    nonlin = jax.device_put(
        jnp.zeros(
            (
                6,
                padded_res.NZ_PADDED,
                padded_res.NX_PADDED,
                padded_res.NY_PADDED,
            ),
            dtype=complex_type,
        ),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )

    nonlin = nonlin.at[:5].set(phys_to_spec(nonlin_phys[:5]), dealias)
    # Basdevant: Get the 5th element from tracelessness
    nonlin = nonlin.at[5].set(-(nonlin[NSYM[0, 0]] + nonlin[NSYM[1, 1]]))

    return jax.lax.with_sharding_constraint(
        nonlin, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@jit(static_argnums=1)
def get_nonlin(velocity_spec, dealias):

    return jax.lax.with_sharding_constraint(
        get_nonlin_spec(get_nonlin_phys(velocity_spec), dealias),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )


@timer("get_rhs_no_lapl")
@jit(static_argnums=(1, 2, 3, 4, 5, 6))
def get_rhs_no_lapl(
    velocity_spec,
    dealias,
    nabla,
    inv_lapl,
    forcing_modes,
    forcing_unit,
    forcing_amplitude,
):

    nonlin = get_nonlin(velocity_spec, dealias)

    advect = jax.device_put(
        jnp.zeros(
            (
                3,
                padded_res.NZ_PADDED,
                padded_res.NX_PADDED,
                padded_res.NY_PADDED,
            ),
            dtype=complex_type,
        ),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )

    def set_advect(i, val):
        return val.at[i, ...].set(
            -jnp.sum(
                nabla * nonlin[(NSYM[0, i], NSYM[1, i], NSYM[2, i]),],
                axis=0,
            )
        )

    advect = lax.fori_loop(0, 3, set_advect, advect, unroll=True)

    rhs_no_lapl = advect - nabla * inv_lapl * jnp.sum(nabla * advect, axis=0)
    if params.phys.forcing is not None:
        rhs_no_lapl = rhs_no_lapl.at[forcing_modes].add(
            forcing_unit * forcing_amplitude
        )

    return jax.lax.with_sharding_constraint(
        rhs_no_lapl, NamedSharding(MESH, P(None, "Z", "X", None))
    )
