from functools import partial

from jax import jit
from jax import numpy as jnp
from jax.sharding import explicit_axes

from bench import timer
from parameters import params
from sharding import sharding


def get_inprod(vector_spec_1, vector_spec_2):
    return jnp.sum(
        jnp.conj(vector_spec_1) * vector_spec_2,
        dtype=sharding.float_type,
    )


def get_norm2(vector_spec):
    return get_inprod(vector_spec, vector_spec)


def get_norm(vector_spec):
    return jnp.sqrt(get_norm2(vector_spec))


def correct_divergence(velocity_spec, kvec, inv_lapl):
    correction = (
        kvec
        * inv_lapl
        * jnp.sum(
            kvec * velocity_spec,
            axis=0,
        )
    )

    error = get_norm(correction) if params.debug.measure_corrections else None

    velocity_corrected = velocity_spec + correction
    return sharding.constrain_vector(velocity_corrected), error


@timer("velocity/correct_velocity")
@jit(donate_argnums=0, out_shardings=(sharding.vector_shard, None))
def correct_velocity(velocity_spec, kvec, inv_lapl):
    norm_corrections = {}
    if params.debug.correct_divergence:
        velocity_corrected, error = correct_divergence(
            velocity_spec, kvec, inv_lapl
        )
        norm_corrections["div"] = error
    else:
        velocity_corrected = velocity_spec

    if not params.debug.measure_corrections:
        norm_corrections = None

    return sharding.constrain_vector(velocity_corrected), norm_corrections


@partial(explicit_axes, axes=sharding.axis_names)
def get_zero_vector(shape, dtype):
    vector = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=sharding.vector_shard,
    )
    return vector


@partial(explicit_axes, axes=sharding.axis_names)
def get_zero_scalar(shape, dtype):
    scalar = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=sharding.scalar_shard,
    )
    return scalar
