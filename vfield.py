from jax import numpy as jnp

from fft import KX, KY, KZ
from parameters import FORCING, KF

# TODO; Check whether the spectral norms need normalization


def inprod(vfieldk1, vfieldk2):
    # TODO: Broadcast the result
    res = jnp.sum(jnp.conj(vfieldk1) * vfieldk2).real / 2
    return res


def norm2(vfieldk):
    return inprod(vfieldk, vfieldk)


def norm(vfieldk):
    return jnp.sqrt(inprod(vfieldk, vfieldk))


def laminar():
    if FORCING == 0:
        vfieldk = 0
    elif FORCING == 1:
        vfieldk = jnp.where((KX == 0) & (KY == KF) & (KZ == 0), -1j * 0.5, 0)
    elif FORCING == 2:
        vfieldk = jnp.where((KX == 0) & (KY == KF) & (KZ == 0), 0.5, 0)

    return vfieldk
