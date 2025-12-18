import numpy as np

from parameters import *
import fft

def rhs_nonlin_term(vel_vfieldxx, vel_vfieldk):

    def rhs_vfieldxx(n):
        return vel_vfieldxx[ISYM[n]] * vel_vfieldxx[JSYM[n]]

    def rhs_vfieldkk(n):
        return fft.sx2k(rhs_vfieldxx(n))

    fvel_vfieldk = None
    return fvel_vfieldk