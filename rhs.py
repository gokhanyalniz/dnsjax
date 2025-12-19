import numpy as np

from parameters import *
from fft import *

def rhs_nonlin_term(vel_vfieldx, vel_vfieldk):

    def rhs_vfieldx(n):
        return vel_vfieldx[ISYM[n]] * vel_vfieldx[JSYM[n]]

    def rhs_vfieldk(n):
        return sx2k(rhs_vfieldx(n))
    
    advect = -sum([NABLA[n]*rhs_vfieldk[NSYM[n,:]] for n in range(3)]) # unclear if this will work
    div = sum([KVEC[n]*advect[n] for n in range(3)])
    fvel_vfieldk = np.array([advect[n] + div*INV_LAPL*KVEC[n] for n in range(3)])
    return fvel_vfieldk