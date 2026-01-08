from jax import numpy as jnp


def inprod(vfieldk1, vfieldk2):
    # TODO: Broadcast the result
    res = jnp.sum(jnp.conj(vfieldk1) * vfieldk2).real / 2
    return res


def norm2(vfieldk):
    return inprod(vfieldk, vfieldk)


def norm(vfieldk):
    return jnp.sqrt(inprod(vfieldk, vfieldk))
