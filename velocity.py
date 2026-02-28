from functools import partial

import jax
from jax import jit
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import explicit_axes

from parameters import padded_res
from sharding import sharding


@jit
def get_inprod(vector_spec_1, vector_spec_2):
    return jnp.sum(
        jnp.conj(vector_spec_1) * vector_spec_2,
        dtype=sharding.float_type,
    )


@jit
def get_norm2(vector_spec):
    return get_inprod(vector_spec, vector_spec)


@jit
def get_norm(vector_spec):
    return jnp.sqrt(get_norm2(vector_spec))


@jit(donate_argnums=0)
def correct_divergence(velocity_spec, nabla, inv_lapl):
    correction = (
        -nabla
        * inv_lapl
        * jnp.sum(
            nabla * velocity_spec,
            axis=0,
        )
    )

    velocity_corrected = velocity_spec + correction
    return jax.lax.with_sharding_constraint(
        velocity_corrected, sharding.spec_shard
    )


@jit(donate_argnums=0)
def correct_velocity(velocity_spec, nabla, inv_lapl, zero_mean):
    velocity_corrected = (
        correct_divergence(velocity_spec, nabla, inv_lapl) * zero_mean
    )

    return jax.lax.with_sharding_constraint(
        velocity_corrected, sharding.spec_shard
    )


@partial(explicit_axes, axes=("Z", "X"))
def _get_zero_velocity_spec(ndims):
    velocity_spec = jnp.zeros(
        (
            ndims,
            padded_res.Nz_padded,
            padded_res.Nx_padded,
            padded_res.Ny_padded,
        ),
        dtype=sharding.complex_type,
        out_sharding=P(None, "Z", "X", None),
    )
    return velocity_spec


@jit(static_argnums=0)
def get_zero_velocity_spec(ndims):

    velocity_spec = _get_zero_velocity_spec(
        ndims=ndims, in_sharding=sharding.spec_shard
    )

    return jax.lax.with_sharding_constraint(velocity_spec, sharding.spec_shard)


@partial(explicit_axes, axes=("Z", "X"))
def _get_zero_scalar_spec():
    scalar_spec = jnp.zeros(
        (
            padded_res.Nz_padded,
            padded_res.Nx_padded,
            padded_res.Ny_padded,
        ),
        dtype=sharding.complex_type,
        out_sharding=P("Z", "X", None),
    )
    return scalar_spec


@jit
def get_zero_scalar_spec():

    scalar_spec = _get_zero_scalar_spec(in_sharding=sharding.spec_shard)

    return jax.lax.with_sharding_constraint(
        scalar_spec, sharding.scalar_spec_shard
    )


@partial(explicit_axes, axes=("Z", "X"))
def _get_zero_velocity_phys(ndims):
    velocity_phys = jnp.zeros(
        (
            ndims,
            padded_res.Ny_padded,
            padded_res.Nz_padded,
            padded_res.Nx_padded,
        ),
        dtype=sharding.complex_type,
        out_sharding=P(None, "Z", "X", None),
    )
    return velocity_phys


@jit(static_argnums=0)
def get_zero_velocity_phys(ndims):
    return jax.lax.with_sharding_constraint(
        _get_zero_velocity_phys(ndims=ndims, in_sharding=sharding.spec_shard),
        sharding.spec_shard,
    )
