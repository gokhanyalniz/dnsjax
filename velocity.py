from jax import jit
from jax import numpy as jnp

from bench import timer
from parameters import params
from sharding import sharding


def get_inprod(vector_spec_1, vector_spec_2, metric):
    return jnp.sum(
        jnp.conj(vector_spec_1) * metric * vector_spec_2,
        dtype=sharding.float_type,
    )


def get_norm2(vector_spec, metric):
    return get_inprod(vector_spec, vector_spec, metric)


def get_norm(vector_spec, metric):
    return jnp.sqrt(get_norm2(vector_spec, metric))


def correct_divergence(velocity_spec, kvec, inv_lapl, metric):
    correction = (
        kvec
        * inv_lapl
        * jnp.sum(
            kvec * velocity_spec,
            axis=0,
        )
    )

    error = (
        get_norm(correction, metric)
        if params.debug.measure_corrections
        else None
    )

    velocity_corrected = velocity_spec + correction
    return sharding.constrain_spec_vector(velocity_corrected), error


@timer("velocity/correct_velocity")
@jit(donate_argnums=0, out_shardings=(sharding.spec_vector_shard, None))
def correct_velocity(velocity_spec, kvec, inv_lapl, metric):
    norm_corrections = {}
    if params.debug.correct_divergence:
        velocity_corrected, error = correct_divergence(
            velocity_spec, kvec, inv_lapl, metric
        )
        norm_corrections["div"] = error
    else:
        velocity_corrected = velocity_spec

    if not params.debug.measure_corrections:
        norm_corrections = None

    return sharding.constrain_spec_vector(velocity_corrected), norm_corrections
