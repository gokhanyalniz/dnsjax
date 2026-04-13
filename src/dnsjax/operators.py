"""Spectral differential operators and wavenumber grids.

Provides the ``Fourier`` dataclass holding precomputed wavenumber arrays
and the discrete Laplacian, as well as the spectral operators (curl,
divergence, gradient, Laplacian, inverse Laplacian) that act on Fourier
coefficients via ``ik`` multiplication.

The ``phys_to_spec`` / ``spec_to_phys`` wrappers apply the 3D FFT
(from :mod:`dnsjax.fft`) vmapped over the three velocity components.
"""

from dataclasses import dataclass

import jax
from jax import Array, jit, vmap
from jax import numpy as jnp

from .fft import _irfft2d, _irfft3d, _rfft2d, _rfft3d
from .parameters import derived_params, params, periodic_systems
from .sharding import sharding


@dataclass
class Fourier:
    """Wavenumber grids, spectral Laplacian, and energy-integral metric.

    Wavenumber arrays are shaped for broadcasting with spectral fields of
    shape ``(ny[-1], nz-1, nx//2)``:

    - ``kx``: shape ``(1, 1, nx//2)`` -- non-negative wavenumbers (real FFT)
    - ``kz``: shape ``(1, nz-1, 1)`` -- full-complex wavenumbers, Nyquist omitted
    - ``ky``: shape ``(ny-1, 1, 1)`` -- full-complex (periodic) or ``None`` (walled)

    The Nyquist mode is omitted on every axis to avoid aliasing artefacts
    and to simplify the zero-padding / truncation logic in the FFT module.

    ``k_metric`` equals 2 for ``kx > 0`` and 1 for ``kx == 0``, accounting
    for the Hermitian symmetry of the real FFT when summing over modes
    (e.g. in energy integrals: the ``kx > 0`` modes represent both ``+kx``
    and ``-kx``).
    """

    def real_harmonics(n: int) -> Array:
        """Non-negative integer wavenumbers ``[0, 1, ..., n//2 - 1]``."""
        # Omit the Nyquist mode
        return jnp.arange(0, n // 2, dtype=int)

    def complex_harmonics(n: int) -> Array:
        """Full-complex integer wavenumbers with the Nyquist mode omitted.

        Returns ``n - 1`` wavenumbers:
        ``[0, 1, ..., n//2-1, -n//2+1, ..., -1]``.
        """
        qs = (jnp.arange(n, dtype=int) + n // 2) % n - n // 2
        # Omit the Nyquist mode
        qs_out = jnp.zeros(n - 1, dtype=int)
        qs_out = qs_out.at[: n // 2].set(qs[: n // 2])
        qs_out = qs_out.at[n // 2 :].set(qs[n // 2 + 1 :])
        return qs_out

    kx: Array = (
        jax.device_put(
            real_harmonics(params.res.nx).reshape([1, 1, -1]),
            sharding.spec_scalar_shard,
        )
        * 2
        * jnp.pi
        / params.geo.lx
    )
    kz: Array = (
        complex_harmonics(params.res.nz).reshape([1, -1, 1])
        * 2
        * jnp.pi
        / params.geo.lz
    )

    # Accounts for real-FFT Hermitian symmetry in spectral-space sums:
    # kx > 0 modes contribute twice (for +kx and -kx).
    k_metric: Array = jnp.where(kx == 0, 1, 2)

    if params.phys.system in periodic_systems:
        ky: Array | None = (
            complex_harmonics(params.res.ny).reshape([-1, 1, 1])
            * 2
            * jnp.pi
            / derived_params.ly
        )

        lapl: Array = -(kx**2 + ky**2 + kz**2)
        # Safe pointwise inverse; the k=0 mode maps to 0 (pressure is
        # determined only up to a constant there).
        inv_lapl: Array | None = jnp.where(lapl < 0, 1 / lapl, 0)
    else:
        ky = None
        # Wall-bounded: only the horizontal Laplacian k_x^2 + k_z^2;
        # the y-part is handled by finite-difference matrices.
        lapl = -(kx**2 + kz**2)
        inv_lapl = None


fourier: Fourier = Fourier()


@jit
@vmap
def phys_to_spec(velocity_phys: Array) -> Array:
    """Forward 3D FFT vmapped over the three velocity components."""
    velocity_spec = _rfft3d(
        velocity_phys,
    )

    return velocity_spec


@jit
@vmap
def spec_to_phys(velocity_spec: Array) -> Array:
    """Inverse 3D FFT vmapped over the three velocity components."""

    velocity_phys = _irfft3d(
        velocity_spec,
    )

    return velocity_phys


@jit
@vmap
def phys_to_spec_2d(velocity_phys: Array) -> Array:
    """Forward 2D FFT (x, z only) vmapped over the three velocity components.

    For wall-bounded flows where the y-direction stays in grid-point space.
    """
    return _rfft2d(velocity_phys)


@jit
@vmap
def spec_to_phys_2d(velocity_spec: Array) -> Array:
    """Inverse 2D FFT (x, z only) vmapped over the three velocity components.

    For wall-bounded flows where the y-direction stays in grid-point space.
    """
    return _irfft2d(velocity_spec)


def cross(vector_1: Array, vector_2: Array) -> Array:
    """Vector cross product ``vector_1 x vector_2`` (component-wise)."""

    return jnp.array(
        [
            vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1],
            vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2],
            vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0],
        ]
    )


def derivative(data_spec: Array, kx: Array, ky: Array, kz: Array, axis: int) -> Array:
    """Spectral derivative: ``i * k_axis * data_spec``."""
    match axis:
        case 0:
            return 1j * kx * data_spec
        case 1:
            return 1j * ky * data_spec
        case 2:
            return 1j * kz * data_spec


def divergence(velocity_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral divergence: ``i*kx*u + i*ky*v + i*kz*w``."""
    return sum([derivative(velocity_spec[i], kx, ky, kz, i) for i in range(3)])


def curl(velocity_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral curl (vorticity): ``i * k x velocity_spec``."""
    return 1j * jnp.array(
        [
            ky * velocity_spec[2] - kz * velocity_spec[1],
            kz * velocity_spec[0] - kx * velocity_spec[2],
            kx * velocity_spec[1] - ky * velocity_spec[0],
        ]
    )


def gradient(data_spec: Array, kx: Array, ky: Array, kz: Array) -> Array:
    """Spectral gradient: ``[i*kx, i*ky, i*kz] * data_spec``."""
    return jnp.array([derivative(data_spec, kx, ky, kz, i) for i in range(3)])


def laplacian(data_spec: Array, lapl_spec: Array) -> Array:
    """Apply the spectral Laplacian (pointwise multiply by ``-k^2``)."""
    return lapl_spec * data_spec


def inverse_laplacian(data_spec: Array, inv_lapl_spec: Array) -> Array:
    """Apply the inverse spectral Laplacian (pointwise multiply by ``1/(-k^2)``)."""
    return inv_lapl_spec * data_spec


def integrate_scalar_in_y(scalar_data: Array, ys: Array) -> Array:
    """Composite Simpson's rule on a non-uniform grid in *y*.

    Requires an odd number of grid points (even number of sub-intervals).
    Uses the exact quadrature weights for pairs of non-uniform panels.

    Parameters
    ----------
    scalar_data:
        1-D array of function values at the grid points *ys*.
    ys:
        1-D array of grid-point coordinates (length must be odd).
    """

    if len(ys) % 2 == 0:
        sharding.print(
            "Simpson integration is not yet implemented"
            "for even # of grid points."
        )
        sharding.exit(code=1)

    h = jnp.diff(ys)  # shape (N-1,)
    h0 = h[:-1:2]  # left sub-intervals:  h0, h2, h4, ...
    h1 = h[1::2]  # right sub-intervals: h1, h3, h5, ...

    y0 = scalar_data[:-2:2]  # left points
    y1 = scalar_data[1:-1:2]  # mid points
    y2 = scalar_data[2::2]  # right points

    hsum = h0 + h1
    hprod = h0 * h1
    h0divh1 = h0 / h1

    panels = (hsum / 6) * (
        y0 * (2 - 1 / h0divh1) + y1 * (hsum**2 / hprod) + y2 * (2 - h0divh1)
    )
    return jnp.sum(panels)
