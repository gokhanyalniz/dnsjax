from jax import numpy as jnp

from .operators import (
    integrate_scalar_in_y,
)
from .parameters import (
    cartesian_systems,
    derived_params,
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
    elif system in cartesian_systems:
        return (
            # 1/(Lx Ly Lz) \int dx dy dz u1 * u2
            integrate_scalar_in_y(
                jnp.sum(
                    jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
                    dtype=sharding.float_type,
                    axis=(0, 2, 3),
                ),
                ys,
            )
            / derived_params.ly
        )
    elif system == "pipe":
        # 1/(Lx \pi R^2) \int r dr d\theta dx u1 * u2
        return (
            integrate_scalar_in_y(
                ys
                * 2
                * jnp.sum(
                    jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
                    dtype=sharding.float_type,
                    axis=(0, 2, 3),
                ),
                ys,
            )
            / (derived_params.ly / 2) ** 2
        )
    else:
        raise NotImplementedError


def get_norm2(vector_spec, k_metric, ys):
    return get_inprod(vector_spec, vector_spec, k_metric, ys)


def get_norm(vector_spec, k_metric, ys):
    return jnp.sqrt(get_norm2(vector_spec, k_metric, ys))
