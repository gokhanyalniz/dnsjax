from jax import numpy as jnp

from .parameters import (
    params,
    periodic_systems,
)
from .sharding import sharding


def get_inprod(vector_spec_1, vector_spec_2, k_metric, ys):
    system = params.phys.system
    if system in periodic_systems:
        # 1/(Lx Ly Lz) \int dx dy dz u1 * u2
        return jnp.sum(
            jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
            dtype=sharding.float_type,
        )
    else:
        raise NotImplementedError


def get_norm2(vector_spec, k_metric, ys):
    return get_inprod(vector_spec, vector_spec, k_metric, ys)


def get_norm(vector_spec, k_metric, ys):
    return jnp.sqrt(get_norm2(vector_spec, k_metric, ys))
