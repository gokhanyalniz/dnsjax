from functools import partial

import jax
from jax import jit
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import FORCING, IC_F, QF
from sharding import MESH
from transform import INV_LAPL, KVEC, NXX, NYY, NZZ, QX, QY, QZ

ZERO_MEAN = jnp.where((QX == 0) & (QY == 0) & (QZ == 0), 0.0, 1.0)


@jit
def get_inprod(vector_spec_1, vector_spec_2):
    return jnp.sum(jnp.conj(vector_spec_1) * vector_spec_2).real


@jit
def get_norm2(vector_spec):
    return get_inprod(vector_spec, vector_spec)


@jit
def get_norm(vector_spec):
    return jnp.sqrt(get_norm2(vector_spec))


@jit
def get_inprod_phys(vector_phys_1, vector_phys_2):
    return jnp.average(jnp.sum(vector_phys_1 * vector_phys_2, axis=0))


@jit
def get_laminar():
    velocity_spec = jax.device_put(
        jnp.zeros((3, NZZ, NXX, NYY), dtype=jnp.complex128),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )
    if FORCING == 1:
        velocity_spec = velocity_spec.at[IC_F].set(
            jnp.where((QX == 0) & (QY == QF) & (QZ == 0), -1j * 0.5, 0)
        )
        velocity_spec = velocity_spec.at[IC_F].set(
            jnp.where((QX == 0) & (QY == -QF) & (QZ == 0), 1j * 0.5, velocity_spec[0])
        )
    elif FORCING == 2:
        velocity_spec = velocity_spec.at[IC_F].set(
            jnp.where((QX == 0) & (QY == QF) & (QZ == 0), 0.5, 0)
        )
        velocity_spec = velocity_spec.at[IC_F].set(
            jnp.where((QX == 0) & (QY == -QF) & (QZ == 0), 0.5, velocity_spec)
        )

    return jax.lax.with_sharding_constraint(
        velocity_spec, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@partial(jit, donate_argnums=0)
def correct_divergence(velocity_spec):
    correction = INV_LAPL * jnp.sum(KVEC * velocity_spec, axis=0)

    velocity_corrected = velocity_spec + correction * KVEC
    return jax.lax.with_sharding_constraint(
        velocity_corrected, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@partial(jit, donate_argnums=0)
def correct_velocity(velocity_spec):
    velocity_corrected = correct_divergence(velocity_spec) * ZERO_MEAN

    return jax.lax.with_sharding_constraint(
        velocity_corrected, NamedSharding(MESH, P(None, "Z", "X", None))
    )
