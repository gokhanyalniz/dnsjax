import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from parameters import FORCING, IC_F, KF
from sharding import MESH
from transform import KX, KY, KZ, NXX, NYY, NZZ

# TODO; Check whether the spectral norms need normalization


def get_inprod(vector_spec_1, vector_spec_2):
    # TODO: Broadcast the result
    res = jnp.sum(jnp.conj(vector_spec_1) * vector_spec_2).real / 2
    return res


def get_norm2(vector_spec):
    return get_inprod(vector_spec, vector_spec)


def get_norm(vector_spec):
    return jnp.sqrt(get_norm2(vector_spec))


def get_laminar():
    velocity_spec = jax.device_put(
        jnp.zeros((3, NZZ, NXX, NYY), dtype=jnp.complex128),
        NamedSharding(MESH, P(None, "Z", "X", None)),
    )
    if FORCING == 1:
        velocity_spec = velocity_spec.at[IC_F].set(
            jnp.where((KX == 0) & (KY == KF) & (KZ == 0), -1j * 0.5, 0)
        )
    elif FORCING == 2:
        velocity_spec = velocity_spec.at[IC_F].set(
            jnp.where((KX == 0) & (KY == KF) & (KZ == 0), 0.5, 0)
        )

    return velocity_spec
