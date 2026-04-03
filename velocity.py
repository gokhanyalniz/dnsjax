from jax import jit
from jax import numpy as jnp

from bench import timer
from operators import divergence, gradient, integrate_scalar, inverse_laplacian
from parameters import cartesian_systems, params, periodic_systems
from sharding import sharding


def get_inprod(vector_spec_1, vector_spec_2, k_metric, ys):
    if params.phys.system in periodic_systems:
        return jnp.sum(
            jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
            dtype=sharding.float_type,
        )
    elif params.phys.system in cartesian_systems:
        return (
            integrate_scalar(
                jnp.sum(
                    jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
                    dtype=sharding.float_type,
                    axis=(0, 2, 3),
                ),
                ys,
            )
            / params.geo.ly
        )
    else:
        raise NotImplementedError


def get_norm2(vector_spec, k_metric, ys):
    return get_inprod(vector_spec, vector_spec, k_metric, ys)


def get_norm(vector_spec, k_metric, ys):
    return jnp.sqrt(get_norm2(vector_spec, k_metric, ys))


def correct_divergence(velocity_spec, kx, ky, kz, inv_lapl, k_metric, ys):
    correction = -gradient(
        inverse_laplacian(
            divergence(
                velocity_spec,
                kx,
                ky,
                kz,
            ),
            inv_lapl,
        ),
        kx,
        ky,
        kz,
    )

    error = (
        get_norm(correction, k_metric, ys)
        if params.debug.measure_corrections
        else None
    )

    velocity_corrected = velocity_spec + correction
    return velocity_corrected, error


@timer("velocity/correct_velocity")
@jit(donate_argnums=0)
def correct_velocity(velocity_spec, kx, ky, kz, inv_lapl, k_metric, ys):
    norm_corrections = {}
    velocity_corrected = velocity_spec

    if (
        params.debug.correct_divergence
        and params.phys.system in periodic_systems
    ):
        velocity_corrected, error = correct_divergence(
            velocity_corrected, kx, ky, kz, inv_lapl, k_metric, ys
        )
        norm_corrections["div"] = error

    # Set the mean mode to zero, it is passive
    velocity_corrected = velocity_corrected.at[sharding.vector_mean_mode].set(
        0, out_sharding=sharding.spec_vector_shard
    )

    if not params.debug.measure_corrections:
        norm_corrections = None

    return velocity_corrected, norm_corrections
