import jax
from jax import jit
from jax import numpy as jnp

from bench import timer
from parameters import padded_res
from rhs import force
from sharding import sharding


@jit
def get_inprod(vector_spec_1, vector_spec_2, dealias):
    return jnp.sum(
        jnp.conj(vector_spec_1) * vector_spec_2,
        dtype=sharding.float_type,
        where=dealias,
    )


@jit
def get_norm2(vector_spec, dealias):
    return get_inprod(vector_spec, vector_spec, dealias)


@timer("get_norm")
@jit
def get_norm(vector_spec, dealias):
    return jnp.sqrt(get_norm2(vector_spec, dealias))


@jit
def get_laminar():
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
    if force.on:
        velocity_spec = velocity_spec.at[force.forced_modes].add(
            jnp.array(force.unit)
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


@timer("correct_velocity")
@jit(donate_argnums=0)
def correct_velocity(velocity_spec, nabla, inv_lapl, zero_mean):
    velocity_corrected = (
        correct_divergence(velocity_spec, nabla, inv_lapl) * zero_mean
    )

    return jax.lax.with_sharding_constraint(
        velocity_corrected, sharding.spec_shard
    )
