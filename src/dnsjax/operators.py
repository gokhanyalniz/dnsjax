"""Shared spectral utilities: FFT wrappers and wavenumber helpers.

Provides wavenumber generation functions (``real_harmonics``,
``complex_harmonics``), vmapped FFT wrappers for 3D transforms,
and the vector cross product.

Geometry-specific ``Fourier`` dataclasses live in the corresponding
geometry modules (``geometries.triply_periodic``).
"""

from jax import Array, jit, vmap
from jax import numpy as jnp

from .fft import _irfft3d, _rfft3d


def real_harmonics(n: int) -> Array:
    """Non-negative integer wavenumbers for a real-FFT axis.

    The Nyquist mode is omitted, leaving `$n / 2$` modes.

    Parameters
    ----------
    n:
        Full mode count along the axis.

    Returns
    -------
    :
        Wavenumber array `$[0, 1, \\dots, n/2 - 1]$`, shape ``(n // 2,)``.
    """
    # Omits the Nyquist mode
    return jnp.arange(0, n // 2, dtype=int)


def complex_harmonics(n: int) -> Array:
    """Full-complex integer wavenumbers with the Nyquist mode omitted.

    Parameters
    ----------
    n:
        Full mode count along the axis.

    Returns
    -------
    :
        `$n - 1$` wavenumbers in FFT order:
        `$[0, 1, \\dots, n/2-1, -n/2+1, \\dots, -1]$`.
    """
    qs = (jnp.arange(n, dtype=int) + n // 2) % n - n // 2
    # Omits the Nyquist mode
    qs_out = jnp.zeros(n - 1, dtype=int)
    qs_out = qs_out.at[: n // 2].set(qs[: n // 2])
    qs_out = qs_out.at[n // 2 :].set(qs[n // 2 + 1 :])
    return qs_out


@jit
@vmap
def phys_to_spec(velocity_phys: Array) -> Array:
    """Forward 3D real FFT, vmapped over velocity components.

    Used for triply-periodic flows, where all three directions are
    Fourier-expanded.

    Parameters
    ----------
    velocity_phys:
        Physical field of shape ``(3, ny_padded, nz_padded, nx_padded)``,
        sharded on the z axis.

    Returns
    -------
    :
        Spectral field of shape ``(3, ny-1, nz-1, nx//2)`` in
        ``[ky, kz, kx]`` layout, sharded on the kx axis.
    """
    return _rfft3d(velocity_phys)


@jit
@vmap
def spec_to_phys(velocity_spec: Array) -> Array:
    """Inverse 3D real FFT, vmapped over velocity components.

    Used for triply-periodic flows.

    Parameters
    ----------
    velocity_spec:
        Spectral field of shape ``(3, ny-1, nz-1, nx//2)`` in
        ``[ky, kz, kx]`` layout, sharded on the kx axis.

    Returns
    -------
    :
        Physical field of shape ``(3, ny_padded, nz_padded, nx_padded)``,
        sharded on the z axis.
    """
    return _irfft3d(velocity_spec)


def cross(vector_1: Array, vector_2: Array) -> Array:
    """Component-wise vector cross product.

    Computes `$\\mathbf{v}_1 \\times \\mathbf{v}_2$` for two
    three-component vector fields.  The leading axis of length 3
    indexes the vector components; remaining axes are broadcast.

    Parameters
    ----------
    vector_1, vector_2:
        Vector fields of shape ``(3, ...)``.

    Returns
    -------
    :
        Cross product of the same shape as the inputs.
    """

    return jnp.array(
        [
            vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1],
            vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2],
            vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0],
        ]
    )
