from functools import partial

from jax import jit
from jax import numpy as jnp
from jax.sharding import explicit_axes

from bench import timer
from parameters import params
from sharding import sharding


# @jit
def get_inprod(vector_spec_1, vector_spec_2):
    return jnp.sum(
        jnp.conj(vector_spec_1) * vector_spec_2,
        dtype=sharding.float_type,
    )


# @jit
def get_norm2(vector_spec):
    return get_inprod(vector_spec, vector_spec)


# @jit
def get_norm(vector_spec):
    return jnp.sqrt(get_norm2(vector_spec))


# @jit(
#     donate_argnums=0,
#     out_shardings=(sharding.spec_shard, None),
# )
def correct_divergence(velocity_spec, nabla, inv_lapl):
    correction = (
        -nabla
        * inv_lapl
        * jnp.sum(
            nabla * velocity_spec,
            axis=0,
        )
    )

    error = get_norm(correction) if params.debug.measure_corrections else None

    velocity_corrected = velocity_spec + correction
    return velocity_corrected, error


@timer("velocity/correct_velocity")
@jit(donate_argnums=0, out_shardings=(sharding.spec_shard, None))
def correct_velocity(velocity_spec, nabla, inv_lapl):
    norm_corrections = {}
    if params.debug.correct_divergence:
        velocity_corrected, error = correct_divergence(
            velocity_spec, nabla, inv_lapl
        )
        norm_corrections["div"] = error
    else:
        velocity_corrected = velocity_spec

    if not params.debug.measure_corrections:
        norm_corrections = None

    return velocity_corrected, norm_corrections


# @jit(static_argnums=0, out_shardings=sharding.spec_shard)
@partial(explicit_axes, axes=sharding.axis_names)
def get_zero_vector(shape, dtype):
    vector = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=sharding.vector_shard,
    )
    return vector


# @jit(out_shardings=sharding.scalar_spec_shard)
@partial(explicit_axes, axes=sharding.axis_names)
def get_zero_scalar(shape, dtype):
    scalar = jnp.zeros(
        shape=shape,
        dtype=dtype,
        out_sharding=sharding.scalar_shard,
    )
    return scalar
