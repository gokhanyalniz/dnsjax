from functools import partial

import jax
from jax import jit
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bench import timer
from operators import (
    DEALIAS,
    INV_LAPL,
    NABLA,
    NX_PADDED,
    NY_PADDED,
    NZ_PADDED,
    ZERO_MEAN,
)
from parameters import params
from rhs import FORCING_MODES, FORCING_UNIT
from sharding import MESH, complex_type, float_type


@jit
def get_inprod(vector_spec_1, vector_spec_2):
    return jnp.sum(
        jnp.conj(vector_spec_1) * vector_spec_2,
        dtype=float_type,
        where=DEALIAS,
    )


@jit
def get_norm2(vector_spec):
    return get_inprod(vector_spec, vector_spec)


@timer("get_norm")
@jit
def get_norm(vector_spec):
    return jnp.sqrt(get_norm2(vector_spec))


@jit
def get_inprod_phys(vector_phys_1, vector_phys_2):
    return jnp.average(jnp.sum(vector_phys_1 * vector_phys_2, axis=0))


@jit
def get_laminar():
    velocity_spec = jax.device_put(
        jnp.zeros((3, NZ_PADDED, NX_PADDED, NY_PADDED), dtype=complex_type),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )
    if params.phys.forcing is not None:
        velocity_spec = velocity_spec.at[FORCING_MODES].add(FORCING_UNIT)

    return jax.lax.with_sharding_constraint(
        velocity_spec, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@partial(jit, donate_argnums=0)
def correct_divergence(velocity_spec):
    correction = (
        -NABLA
        * INV_LAPL
        * jnp.sum(
            NABLA * velocity_spec,
            axis=0,
        )
    )

    velocity_corrected = velocity_spec + correction
    return jax.lax.with_sharding_constraint(
        velocity_corrected, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@timer("correct_velocity")
@partial(jit, donate_argnums=0)
def correct_velocity(velocity_spec):
    velocity_corrected = correct_divergence(velocity_spec) * ZERO_MEAN

    return jax.lax.with_sharding_constraint(
        velocity_corrected, NamedSharding(MESH, P(None, "Z", "X", None))
    )
