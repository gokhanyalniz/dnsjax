"""Triply-periodic geometry: Fourier class, differential operators, norms."""

from dataclasses import dataclass, field

import jax
from jax import Array
from jax import numpy as jnp

from ..operators import complex_harmonics, real_harmonics
from ..parameters import derived_params, params
from ..sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class Fourier:
    """Wavenumber grids for the triply-periodic geometry.

    Broadcasting shapes match the spectral layout ``(ky, kz, kx)``:
    - ``kx``: shape ``(1, 1, nx//2)``
    - ``kz``: shape ``(1, nz-1, 1)``
    - ``ky``: shape ``(ny-1, 1, 1)``

    ``k_metric`` equals 2 for `$k_x > 0$` and 1 for `$k_x = 0$`,
    accounting for the Hermitian symmetry of the real FFT.
    """

    kx: Array = field(init=False)
    kz: Array = field(init=False)
    ky: Array = field(init=False)
    k_metric: Array = field(init=False)
    lapl: Array = field(init=False)
    inv_lapl: Array = field(init=False)

    def __post_init__(self) -> None:
        self.kx = (
            jax.device_put(
                real_harmonics(params.res.nx).reshape([1, 1, -1]),
                sharding.spec_scalar_shard,
            )
            * 2
            * jnp.pi
            / params.geo.lx
        )
        self.kz = (
            complex_harmonics(params.res.nz).reshape([1, -1, 1])
            * 2
            * jnp.pi
            / params.geo.lz
        )
        self.ky = (
            complex_harmonics(params.res.ny).reshape([-1, 1, 1])
            * 2
            * jnp.pi
            / derived_params.ly
        )

        self.k_metric = jnp.where(self.kx == 0, 1, 2)
        self.lapl = -(self.kx**2 + self.ky**2 + self.kz**2)
        self.inv_lapl = jnp.where(self.lapl < 0, 1 / self.lapl, 0)


fourier: Fourier = Fourier()


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
