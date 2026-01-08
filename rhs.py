from jax import numpy as jnp

from fft import FORCE, INV_LAPL, KVEC, LAPL, sx2k
from parameters import ICF, ISYM, JSYM, NSYM, RE


def nonlin_term(vel_vfieldx):
    def rhs_vfieldx(n):
        return vel_vfieldx[ISYM[n]] * vel_vfieldx[JSYM[n]]

    def rhs_vfieldk(n):
        return sx2k(rhs_vfieldx(n))

    advect = jnp.array(
        [
            -sum([1j * KVEC[n] * rhs_vfieldk(NSYM[n, m]) for n in range(3)])
            for m in range(3)
        ]
    )
    div = sum([KVEC[n] * advect[n] for n in range(3)])

    fvel_vfieldk = jnp.array(
        [
            advect[n] + div * INV_LAPL * KVEC[n] + FORCE
            if n == ICF
            else advect[n] + div * INV_LAPL * KVEC[n]
            for n in range(3)
        ]
    )

    return fvel_vfieldk


def nonlin_plus_lapl(vel_vfieldx, vel_vfieldk):
    fvel_vfieldk = nonlin_term(vel_vfieldx) + (LAPL / RE) * vel_vfieldk

    return fvel_vfieldk
