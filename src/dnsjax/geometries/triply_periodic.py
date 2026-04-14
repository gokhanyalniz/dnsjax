"""Triply-periodic geometry functions and differential operators."""

from jax import Array
from jax import numpy as jnp

from ..sharding import sharding


def get_inprod(
    vector_spec_1: Array, vector_spec_2: Array, k_metric: Array
) -> Array:
    """Volume-averaged L2 inner product ``<u1, u2>`` in spectral space.

    For triply-periodic flows the sum is a direct Parseval sum over all
    Fourier modes.
    """
    return jnp.sum(
        jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
        dtype=sharding.float_type,
    )


def get_norm2(vector_spec: Array, k_metric: Array) -> Array:
    """Squared L2 norm ``||u||^2 = <u, u>``."""
    return get_inprod(vector_spec, vector_spec, k_metric)


def get_norm(vector_spec: Array, k_metric: Array) -> Array:
    """L2 norm ``||u|| = sqrt(<u, u>)``."""
    return jnp.sqrt(get_norm2(vector_spec, k_metric))


def derivative(
    data_spec: Array, kx: Array, ky: Array, kz: Array, axis: int
) -> Array:
    """Spectral derivative: `$i k_{\text{axis}} \, \text{data\_spec}$`."""
    match axis:
        case 0:
            return 1j * kx * data_spec
        case 1:
            return 1j * ky * data_spec
        case 2:
            return 1j * kz * data_spec


def divergence(velocity_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral divergence: `$i k_x u + i k_y v + i k_z w$`."""
    return sum([derivative(velocity_spec[i], kx, ky, kz, i) for i in range(3)])


def curl(velocity_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral curl (vorticity):
    `$i \mathbf{k} \times \mathbf{u}_{\text{spec}}$`.
    """
    return 1j * jnp.array(
        [
            ky * velocity_spec[2] - kz * velocity_spec[1],
            kz * velocity_spec[0] - kx * velocity_spec[2],
            kx * velocity_spec[1] - ky * velocity_spec[0],
        ]
    )


def gradient(data_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral gradient: `$[i k_x, i k_y, i k_z] \, \text{data\_spec}$`."""
    return jnp.array([derivative(data_spec, kx, ky, kz, i) for i in range(3)])


def laplacian(data_spec: Array, lapl_spec: Array) -> Array:
    """Apply the spectral Laplacian (pointwise multiply by `$-k^2$`)."""
    return lapl_spec * data_spec


def inverse_laplacian(data_spec: Array, inv_lapl_spec: Array) -> Array:
    """Apply the inverse spectral Laplacian
    (pointwise multiply by `$-1/k^2$`)."""
    return inv_lapl_spec * data_spec
