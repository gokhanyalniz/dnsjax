from jax import numpy as jnp

from parameters import FORCING, IC_F, RE
from transform import FORCE, INV_LAPL, KVEC, LAPL, phys_to_spec_scalar

# Given 3x3 symmetric matrix M, entries M_{ij} will be used
ISYM = jnp.array([0, 0, 0, 1, 1, 2], dtype=int)
JSYM = jnp.array([0, 1, 2, 1, 2, 2], dtype=int)
NSYM = jnp.zeros((3, 3), dtype=int)

for n in range(6):
    i = ISYM[n]
    j = JSYM[n]
    NSYM = NSYM.at[i, j].set(n)
    NSYM = NSYM.at[j, i].set(n)


def get_rhs_no_lapl(velocity_phys):
    def get_nonlin_phys(n):
        return velocity_phys[ISYM[n]] * velocity_phys[JSYM[n]]

    def get_nonlin_spec(n):
        return phys_to_spec_scalar(get_nonlin_phys(n))

    # TODO: Implement Basdevant
    advect = jnp.array(
        [
            -sum([1j * KVEC[n] * get_nonlin_spec(NSYM[n, m]) for n in range(3)])
            for m in range(3)
        ]
    )
    div = jnp.sum(KVEC * advect, axis=0)

    rhs_no_lapl = advect + div * INV_LAPL * KVEC
    if FORCING != 0:
        rhs_no_lapl = rhs_no_lapl.at[IC_F].add(FORCE)

    return rhs_no_lapl


def get_rhs(velocity_spec, velocity_phys):
    rhs = get_rhs_no_lapl(velocity_phys) + (LAPL / RE) * velocity_spec

    return rhs
