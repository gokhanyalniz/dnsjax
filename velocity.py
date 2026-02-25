import jax
from jax import jit
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bench import timer
from parameters import padded_res, params
from rhs import force
from sharding import MESH, complex_type, float_type


@jit
def get_inprod(vector_spec_1, vector_spec_2, dealias):
    return jnp.sum(
        jnp.conj(vector_spec_1) * vector_spec_2,
        dtype=float_type,
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
                padded_res.NZ_PADDED,
                padded_res.NX_PADDED,
                padded_res.NY_PADDED,
            ),
            dtype=complex_type,
        ),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )
    if params.phys.forcing is not None:
        velocity_spec = velocity_spec.at[force.FORCING_MODES].add(
            jnp.array(force.FORCING_UNIT)
        )

    return jax.lax.with_sharding_constraint(
        velocity_spec, NamedSharding(MESH, P(None, "Z", "X", None))
    )


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
        velocity_corrected, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@timer("correct_velocity")
@jit(donate_argnums=0)
def correct_velocity(velocity_spec, nabla, inv_lapl, zero_mean):
    velocity_corrected = (
        correct_divergence(velocity_spec, nabla, inv_lapl) * zero_mean
    )

    return jax.lax.with_sharding_constraint(
        velocity_corrected, NamedSharding(MESH, P(None, "Z", "X", None))
    )
