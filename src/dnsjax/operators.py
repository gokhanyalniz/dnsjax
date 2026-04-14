"""Spectral differential operators and wavenumber grids.

Provides the ``Fourier`` dataclass holding precomputed wavenumber arrays
and the discrete Laplacian, as well as the spectral operators (curl,
divergence, gradient, Laplacian, inverse Laplacian) that act on Fourier
coefficients via ``ik`` multiplication.

The ``phys_to_spec`` / ``spec_to_phys`` wrappers apply the 3D FFT
(from :mod:`dnsjax.fft`) vmapped over the three velocity components.
"""

from dataclasses import dataclass, field

import jax
from jax import Array, jit, vmap
from jax import numpy as jnp

from .fft import _irfft2d, _irfft3d, _rfft2d, _rfft3d
from .parameters import derived_params, params, periodic_systems
from .sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class Fourier:
    """Wavenumber grids, spectral Laplacian, and energy-integral metric.

    Wavenumber arrays are shaped for broadcasting with spectral fields of
    shape ``(ny[-1], nz-1, nx//2)``:

    - ``kx``: shape ``(1, 1, nx//2)`` -- non-negative wavenumbers (real FFT)
    - ``kz``: shape ``(1, nz-1, 1)`` -- full-complex wavenumbers
    - ``ky``: shape ``(ny-1, 1, 1)`` -- full-complex (periodic) or ``None``


    The Nyquist mode is omitted on every axis to avoid aliasing artefacts
    and to simplify the zero-padding / truncation logic in the FFT module.

    ``k_metric`` equals 2 for `$k_x > 0$` and 1 for `$k_x = 0$`, accounting
    for the Hermitian symmetry of the real FFT when summing over modes
    (e.g. in energy integrals: the `$k_x > 0$` modes represent both `$+k_x$`
    and `$-k_x$`).
    """

    kx: Array = field(init=False)
    kz: Array = field(init=False)
    k_metric: Array = field(init=False)
    ky: Array | None = field(init=False)
    lapl: Array = field(init=False)
    inv_lapl: Array | None = field(init=False)

    @staticmethod
    def real_harmonics(n: int) -> Array:
        """Non-negative integer wavenumbers `$[0, 1, \dots, n/2 - 1]$`."""
        # Omit the Nyquist mode
        return jnp.arange(0, n // 2, dtype=int)

    @staticmethod
    def complex_harmonics(n: int) -> Array:
        """Full-complex integer wavenumbers with the Nyquist mode omitted.

        Returns `$n - 1$` wavenumbers:
        `$[0, 1, \dots, n/2-1, -n/2+1, \dots, -1]$`.
        """
        qs = (jnp.arange(n, dtype=int) + n // 2) % n - n // 2
        # Omit the Nyquist mode
        qs_out = jnp.zeros(n - 1, dtype=int)
        qs_out = qs_out.at[: n // 2].set(qs[: n // 2])
        qs_out = qs_out.at[n // 2 :].set(qs[n // 2 + 1 :])
        return qs_out

    def __post_init__(self) -> None:
        self.kx = (
            jax.device_put(
                self.real_harmonics(params.res.nx).reshape([1, 1, -1]),
                sharding.spec_scalar_shard,
            )
            * 2
            * jnp.pi
            / params.geo.lx
        )
        self.kz = (
            self.complex_harmonics(params.res.nz).reshape([1, -1, 1])
            * 2
            * jnp.pi
            / params.geo.lz
        )

        # Accounts for real-FFT Hermitian symmetry in spectral-space sums:
        # kx > 0 modes contribute twice (for +kx and -kx).
        self.k_metric = jnp.where(self.kx == 0, 1, 2)

        if params.phys.system in periodic_systems:
            self.ky = (
                self.complex_harmonics(params.res.ny).reshape([-1, 1, 1])
                * 2
                * jnp.pi
                / derived_params.ly
            )

            self.lapl = -(self.kx**2 + self.ky**2 + self.kz**2)
            # Safe pointwise inverse; the k=0 mode maps to 0 (pressure is
            # determined only up to a constant there).
            self.inv_lapl = jnp.where(self.lapl < 0, 1 / self.lapl, 0)
        else:
            self.ky = None
            # Wall-bounded: only the horizontal Laplacian k_x^2 + k_z^2;
            # the y-part is handled by finite-difference matrices.
            self.lapl = -(self.kx**2 + self.kz**2)
            self.inv_lapl = None


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
    """Vector cross product `$\mathbf{v}_1 \times \mathbf{v}_2$`
    (component-wise).
    """

    return jnp.array(
        [
            vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1],
            vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2],
            vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0],
        ]
    )
