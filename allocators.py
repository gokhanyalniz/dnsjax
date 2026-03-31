from functools import partial

from jax import numpy as jnp
from jax.sharding import explicit_axes

from sharding import sharding


@partial(explicit_axes, axes=sharding.axis_names)
def get_zero_spec_vector(shape, dtype):
    vector = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=sharding.spec_vector_shard,
    )
    return vector


@partial(explicit_axes, axes=sharding.axis_names)
def get_zero_phys_vector(shape, dtype):
    vector = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=sharding.phys_vector_shard,
    )
    return vector


@partial(explicit_axes, axes=sharding.axis_names)
def get_zero_spec_scalar(shape, dtype):
    scalar = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=sharding.spec_scalar_shard,
    )
    return scalar


@partial(explicit_axes, axes=sharding.axis_names)
def get_zero_phys_scalar(shape, dtype):
    scalar = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=sharding.phys_scalar_shard,
    )
    return scalar
