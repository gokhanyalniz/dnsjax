from jax import numpy as jnp


def inprod(vfieldk1, vfieldk2):
    res = jnp.sum(jnp.conj(vfieldk1) * vfieldk2).real / 2
    return res


def norm2(vfieldk):
    return inprod(vfieldk, vfieldk)


def norm(vfieldk):
    return jnp.sqrt(inprod(vfieldk, vfieldk))
