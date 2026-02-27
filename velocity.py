import jax
from jax import jit
from jax import numpy as jnp

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


@jit
def get_zero_velocity():
    velocity_spec = jax.device_put(
        jnp.zeros(
            (
                3,
                padded_res.Nz_padded,
                padded_res.Nx_padded,
                padded_res.Ny_padded,
            ),
            dtype=sharding.complex_type,
        ),
        sharding.spec_shard,
    )

    return jax.lax.with_sharding_constraint(velocity_spec, sharding.spec_shard)


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
