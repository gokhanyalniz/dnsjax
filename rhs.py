import numpy as np

from fft import FORCE, INV_LAPL, KVEC, LAPL, sx2k
from parameters import ICF, ISYM, JSYM, NSYM, RE


def rhs_nonlin_term(vel_vfieldx):
    def rhs_vfieldx(n):
        return vel_vfieldx[ISYM[n]] * vel_vfieldx[JSYM[n]]

    def rhs_vfieldk(n):
        return sx2k(rhs_vfieldx(n))

    advect = [
        -sum([1j * KVEC[n] * rhs_vfieldk[NSYM[n, j]] for n in range(3)])
        for j in range(3)
    ]
    div = sum([KVEC[n] * advect[n] for n in range(3)])

    fvel_vfieldk = np.array(
        [
            advect[n] + div * INV_LAPL * KVEC[n] + FORCE
            if n == ICF
            else advect[n] + div * INV_LAPL * KVEC[n]
            for n in range(3)
        ]
    )

    return fvel_vfieldk


def rhs_all(vel_vfieldx, vel_vfieldk):
    fvel_vfieldk = rhs_nonlin_term(vel_vfieldx) + (LAPL / RE) * vel_vfieldk

    return fvel_vfieldk
