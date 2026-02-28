from functools import partial

from jax import jit
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
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


@partial(explicit_axes, axes=("Z", "X"))
def _get_zero_velocity_spec(ndims):
    velocity_spec = jnp.zeros(
        (ndims, *sharding.spec_shape),
        dtype=sharding.complex_type,
        out_sharding=P(None, "Z", "X", None),
    )
    return velocity_spec


# @jit(static_argnums=0, out_shardings=sharding.spec_shard)
def get_zero_velocity_spec(ndims):

    velocity_spec = _get_zero_velocity_spec(
        ndims=ndims, in_sharding=sharding.spec_shard
    )

    return velocity_spec


@partial(explicit_axes, axes=("Z", "X"))
def _get_zero_scalar_spec():
    scalar_spec = jnp.zeros(
        sharding.spec_shape,
        dtype=sharding.complex_type,
        out_sharding=P("Z", "X", None),
    )
    return scalar_spec


# @jit(out_shardings=sharding.scalar_spec_shard)
def get_zero_scalar_spec():

    scalar_spec = _get_zero_scalar_spec(in_sharding=sharding.spec_shard)

    return scalar_spec


@partial(explicit_axes, axes=("Z", "X"))
def _get_zero_velocity_phys(ndims):
    velocity_phys = jnp.zeros(
        (ndims, *sharding.phys_shape),
        dtype=sharding.phys_type,
        out_sharding=P(None, "Z", "X", None),
    )
    return velocity_phys


# @jit(static_argnums=0, out_shardings=sharding.phys_shard)


def get_zero_velocity_phys(ndims):
    return _get_zero_velocity_phys(
        ndims=ndims, in_sharding=sharding.phys_shard
    )
