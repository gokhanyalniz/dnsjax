from functools import partial

import jax
from jax import jit
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bench import timer
from parameters import params
from rhs import FORCING_MODES, FORCING_UNIT, INV_LAPL, NABLA
from sharding import MESH
from transform import DEALIAS, NX_PADDED, NY_PADDED, NZ_PADDED, QX, QY, QZ

ZERO_MEAN = jnp.where((QX == 0) & (QY == 0) & (QZ == 0), False, True)


@jit
def get_inprod(vector_spec_1, vector_spec_2):
    return jnp.sum(
        jnp.conj(vector_spec_1) * vector_spec_2,
        dtype=jnp.float64,
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
        jnp.zeros((3, NZ_PADDED, NX_PADDED, NY_PADDED), dtype=jnp.complex128),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )
    if params.phys.forcing in ["kolmogorov", "waleffe"]:
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
