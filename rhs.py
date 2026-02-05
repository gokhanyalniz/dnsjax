from jax import numpy as jnp

from fft import FORCE, INV_LAPL, KVEC, LAPL, sx2k
from parameters import FORCING, IC_F, ISYM, JSYM, NSYM, RE


def nonlin_term(vfieldx):
    def rhs_vfieldx(n):
        return vfieldx[ISYM[n]] * vfieldx[JSYM[n]]

    def rhs_vfieldk(n):
        return sx2k(rhs_vfieldx(n))

    # TODO: Implement Basdevant
    advect = jnp.array(
        [
            -sum([1j * KVEC[n] * rhs_vfieldk(NSYM[n, m]) for n in range(3)])
            for m in range(3)
        ]
    )
    div = jnp.sum(KVEC * advect, axis=0)

    fvel_vfieldk = advect + div * INV_LAPL * KVEC
    if FORCING != 0:
        fvel_vfieldk = fvel_vfieldk.at[IC_F].add(FORCE)

    return fvel_vfieldk


def nonlin_plus_lapl(vfieldx, vfieldk):
    fvel_vfieldk = nonlin_term(vfieldx) + (LAPL / RE) * vfieldk

    return fvel_vfieldk
