"""Shared spectral utilities: FFT wrappers and wavenumber helpers.

Provides wavenumber generation functions (``real_harmonics``,
``complex_harmonics``), vmapped FFT wrappers for 3D / 2D transforms,
and the vector cross product.

Geometry-specific ``Fourier`` dataclasses live in the corresponding
geometry modules (``geometries.triply_periodic``,
``geometries.cartesian``).
"""

from jax import Array, jit, vmap
from jax import numpy as jnp

from .fft import _irfft2d, _irfft3d, _rfft2d, _rfft3d


def real_harmonics(n: int) -> Array:
    """Non-negative integer wavenumbers `$[0, 1, \\dots, n/2 - 1]$`."""
    # Omit the Nyquist mode
    return jnp.arange(0, n // 2, dtype=int)


def complex_harmonics(n: int) -> Array:
    """Full-complex integer wavenumbers with the Nyquist mode omitted.

    Returns `$n - 1$` wavenumbers:
    `$[0, 1, \\dots, n/2-1, -n/2+1, \\dots, -1]$`.
    """
    qs = (jnp.arange(n, dtype=int) + n // 2) % n - n // 2
    # Omit the Nyquist mode
    qs_out = jnp.zeros(n - 1, dtype=int)
    qs_out = qs_out.at[: n // 2].set(qs[: n // 2])
    qs_out = qs_out.at[n // 2 :].set(qs[n // 2 + 1 :])
    return qs_out


@jit
@vmap
def phys_to_spec(velocity_phys: Array) -> Array:
    """Forward 3D FFT vmapped over the three velocity components."""
    return _rfft3d(velocity_phys)


@jit
@vmap
def spec_to_phys(velocity_spec: Array) -> Array:
    """Inverse 3D FFT vmapped over the three velocity components."""
    return _irfft3d(velocity_spec)


@jit
@vmap
def phys_to_spec_2d(velocity_phys: Array) -> Array:
    """Forward 2D FFT (x, z only) vmapped over velocity components.

    For wall-bounded flows where the y-direction stays in grid-point
    space.  Returns spectral data in ``(Nkz, Nkx, Ny)`` layout.
    """
    return jnp.transpose(_rfft2d(velocity_phys), (1, 2, 0))


@jit
@vmap
def spec_to_phys_2d(velocity_spec: Array) -> Array:
    """Inverse 2D FFT (x, z only) vmapped over velocity components.

    Accepts spectral data in ``(Nkz, Nkx, Ny)`` layout and returns
    physical data in ``(Ny, nz, nx)`` layout.
    """
    return _irfft2d(jnp.transpose(velocity_spec, (2, 0, 1)))


def cross(vector_1: Array, vector_2: Array) -> Array:
    """Vector cross product `$\\mathbf{v}_1 \\times \\mathbf{v}_2$`
    (component-wise).
    """

    return jnp.array(
        [
            vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1],
            vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2],
            vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0],
        ]
    )
