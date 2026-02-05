from jax import numpy as jnp

from fft import FORCE, INV_LAPL, KVEC, LAPL, phys_to_spec_scalar
from parameters import FORCING, IC_F, ISYM, JSYM, NSYM, RE


def compute_rhs_no_lapl(vfieldx):
    def compute_nonlin_term_phys(n):
        return vfieldx[ISYM[n]] * vfieldx[JSYM[n]]

    def compute_nonlin_term_spec(n):
        return phys_to_spec_scalar(compute_nonlin_term_phys(n))

    # TODO: Implement Basdevant
    advect = jnp.array(
        [
            -sum([1j * KVEC[n] * compute_nonlin_term_spec(NSYM[n, m]) for n in range(3)])
            for m in range(3)
        ]
    )
    div = jnp.sum(KVEC * advect, axis=0)

    rhs_no_lapl = advect + div * INV_LAPL * KVEC
    if FORCING != 0:
        rhs_no_lapl = rhs_no_lapl.at[IC_F].add(FORCE)

    return rhs_no_lapl


def compute_rhs(vfieldx, vfieldk):
    rhs = compute_rhs_no_lapl(vfieldx) + (LAPL / RE) * vfieldk

    return rhs
