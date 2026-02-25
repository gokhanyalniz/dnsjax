from functools import partial

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bench import timer
from parameters import params
from sharding import MESH, complex_type, float_type
from transform import (
    DEALIAS,
    KX,
    KY,
    KZ,
    NX_PADDED,
    NY_PADDED,
    NZ_PADDED,
    QX,
    QY,
    QZ,
    phys_to_spec_vector,
    spec_to_phys_vector,
)

# Physics
if params.phys.forcing is not None:
    AMP = jnp.pi**2 / (4 * params.phys.Re)

    IC_F = 0  # Forced component
    QF = 1  # Forcing harmonic
    KF = 2 * jnp.pi * QF / params.geo.Ly

    FP = (
        IC_F,
        *(i[0] for i in jnp.nonzero((QX == 0) & (QY == QF) & (QZ == 0))),
    )
    FN = (
        IC_F,
        *(i[0] for i in jnp.nonzero((QX == 0) & (QY == -QF) & (QZ == 0))),
    )
    FORCING_MODES = tuple(zip(FP, FN, strict=True))

    if params.phys.forcing == "kolmogorov":
        FORCING_UNIT = jnp.array([-1j, 1j]) * 0.5
    elif params.phys.forcing == "waleffe":
        FORCING_UNIT = jnp.array([1, 1]) * 0.5
        jax.distributed.shutdown()
        exit("The Ry symmetry needed for Waleffe flow is not yet implemented.")
else:
    FORCING_MODES = jnp.array([])
    FORCING_UNIT = 0
    AMP = 1

NABLA = jnp.zeros((3, NZ_PADDED, NX_PADDED, NY_PADDED), dtype=complex_type)

for ix in range(NX_PADDED):
    NABLA = NABLA.at[0, :, ix, :].set(1j * KX[0, ix, 0])
for iy in range(NY_PADDED):
    NABLA = NABLA.at[1, :, :, iy].set(1j * KY[0, 0, iy])
for iz in range(NZ_PADDED):
    NABLA = NABLA.at[2, iz, :, :].set(1j * KZ[iz, 0, 0])

# Zero the dealiased modes to (potentially) save computation
NABLA = DEALIAS * NABLA
LAPL = (-(KX**2) - KY**2 - KZ**2) * DEALIAS
INV_LAPL = jnp.where(LAPL < 0, 1 / LAPL, 0)

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

    velocity_phys = spec_to_phys_vector(velocity_spec)  # 3 FFTs

    nonlin_phys = jax.device_put(
        jnp.zeros((6, NY_PADDED, NZ_PADDED, NX_PADDED), dtype=complex_type),
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


@partial(jit, donate_argnums=0)
def get_nonlin_spec(nonlin_phys):

    nonlin = jax.device_put(
        jnp.zeros((6, NZ_PADDED, NX_PADDED, NY_PADDED), dtype=complex_type),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )

    nonlin = nonlin.at[:5].set(phys_to_spec_vector(nonlin_phys[:5]))
    # Basdevant: Get the 5th element from tracelessness
    nonlin = nonlin.at[5].set(-(nonlin[NSYM[0, 0]] + nonlin[NSYM[1, 1]]))

    return jax.lax.with_sharding_constraint(
        nonlin, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@jit
def get_nonlin(velocity_spec):

    return jax.lax.with_sharding_constraint(
        get_nonlin_spec(get_nonlin_phys(velocity_spec)),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )


@timer("get_rhs_no_lapl")
@jit
def get_rhs_no_lapl(velocity_spec):

    nonlin = get_nonlin(velocity_spec)

    advect = jax.device_put(
        jnp.zeros((3, NZ_PADDED, NX_PADDED, NY_PADDED), dtype=complex_type),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )

    def set_advect(i, val):
        return val.at[i, ...].set(
            -jnp.sum(
                NABLA * nonlin[(NSYM[0, i], NSYM[1, i], NSYM[2, i]),],
                axis=0,
            )
        )

    advect = lax.fori_loop(0, 3, set_advect, advect, unroll=True)

    rhs_no_lapl = advect - NABLA * INV_LAPL * jnp.sum(NABLA * advect, axis=0)
    if params.phys.forcing is not None:
        rhs_no_lapl = rhs_no_lapl.at[FORCING_MODES].add(FORCING_UNIT * AMP)

    return jax.lax.with_sharding_constraint(
        rhs_no_lapl, NamedSharding(MESH, P(None, "Z", "X", None))
    )
