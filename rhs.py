import jax
from jax import jit
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bench import timer
from parameters import FORCING, LY, RE
from sharding import MESH
from transform import (
    DEALIAS,
    KX,
    KY,
    KZ,
    NXX,
    NYY,
    NZZ,
    QX,
    QY,
    QZ,
    phys_to_spec_vector,
    spec_to_phys_vector,
)

# Physics
if FORCING in [1, 2]:
    AMP = jnp.pi**2 / (4 * RE)

    IC_F = 0  # Forced component
    QF = 1  # Forcing harmonic
    KF = 2 * jnp.pi * QF / LY

FP = (0, *(i[0] for i in jnp.nonzero((QX == 0) & (QY == QF) & (QZ == 0))))
FN = (0, *(i[0] for i in jnp.nonzero((QX == 0) & (QY == -QF) & (QZ == 0))))
FORCING_MODES = tuple(zip(FP, FN, strict=True))

if FORCING == 0:
    FORCING_UNIT = 0
elif FORCING == 1:
    FORCING_UNIT = jnp.array([-1j, 1j]) * 0.5
elif FORCING == 2:
    FORCING_UNIT = jnp.array([1, 1]) * 0.5

NABLA = jnp.zeros((3, NZZ, NXX, NYY), dtype=jnp.complex128)

for ix in range(NXX):
    NABLA = NABLA.at[0, :, ix, :].set(1j * KX[0, ix, 0])
for iy in range(NYY):
    NABLA = NABLA.at[1, :, :, iy].set(1j * KY[0, 0, iy])
for iz in range(NZZ):
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
def get_nonlin(velocity_spec):

    velocity_phys = spec_to_phys_vector(velocity_spec)  # 3 FFTs

    # Basdevant optimization: https://doi.org/10.1016/0021-9991(83)90064-5
    # Save on one FFT by the tracelessness of the nonlinear term
    # The trace is effectively moved to pressure. If "true" pressure is ever
    # needed, this needs to be taken into account.
    uu = jnp.stack(
        [velocity_phys[ISYM[n]] * velocity_phys[JSYM[n]] for n in range(6)],
    )

    trace = jnp.sum(
        uu[tuple(NSYM[i, i] for i in range(3)),], axis=0, dtype=jnp.float64
    )

    # No need to update (2,2), it's not used
    uu = uu.at[(NSYM[0, 0], NSYM[1, 1]),].subtract(trace / 3)

    nonlin = phys_to_spec_vector(uu[:5])  # 5 FFTs
    nonlin = jnp.concatenate(
        (nonlin, -(nonlin[NSYM[0, 0]] + nonlin[None, NSYM[1, 1]])), axis=0
    )

    return jax.lax.with_sharding_constraint(
        nonlin, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@timer("get_rhs_no_lapl")
@jit
def get_rhs_no_lapl(velocity_spec):

    nonlin = get_nonlin(velocity_spec)

    advect = -jnp.stack(
        [
            jnp.sum(
                NABLA * nonlin[tuple(NSYM[n, m] for n in range(3)),],
                axis=0,
            )
            for m in range(3)
        ]
    )

    rhs_no_lapl = advect - NABLA * INV_LAPL * jnp.sum(NABLA * advect, axis=0)
    if FORCING != 0:
        rhs_no_lapl = rhs_no_lapl.at[FORCING_MODES].add(FORCING_UNIT * AMP)

    return jax.lax.with_sharding_constraint(
        rhs_no_lapl, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@jit
def get_rhs(velocity_spec):
    rhs = get_rhs_no_lapl(velocity_spec) + (LAPL / RE) * velocity_spec

    return jax.lax.with_sharding_constraint(
        rhs, NamedSharding(MESH, P(None, "Z", "X", None))
    )
