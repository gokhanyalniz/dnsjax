from dataclasses import dataclass

import jax
from jax import jit, vmap
from jax import numpy as jnp

from .fft import _irfft3d, _rfft3d
from .parameters import derived_params, params, periodic_systems
from .sharding import sharding


@dataclass
class Fourier:
    def real_harmonics(n):
        # Omit the Nyquist mode
        return jnp.arange(0, n // 2, dtype=int)

    def complex_harmonics(n):
        qs = (jnp.arange(n, dtype=int) + n // 2) % n - n // 2
        # Omit the Nyquist mode
        qs_out = jnp.zeros(n - 1, dtype=int)
        qs_out = qs_out.at[: n // 2].set(qs[: n // 2])
        qs_out = qs_out.at[n // 2 :].set(qs[n // 2 + 1 :])
        return qs_out

    kx = (
        jax.device_put(
            real_harmonics(params.res.nx).reshape([1, 1, -1]),
            sharding.spec_scalar_shard,
        )
        * 2
        * jnp.pi
        / params.geo.lx
    )
    kz = (
        complex_harmonics(params.res.nz).reshape([1, -1, 1])
        * 2
        * jnp.pi
        / params.geo.lz
    )

    k_metric = jnp.where(kx == 0, 1, 2)

    if params.phys.system in periodic_systems:
        ky = (
            complex_harmonics(params.res.ny).reshape([-1, 1, 1])
            * 2
            * jnp.pi
            / derived_params.ly
        )

        lapl = -(kx**2 + ky**2 + kz**2)
        inv_lapl = jnp.where(lapl < 0, 1 / lapl, 0)
    else:
        ky = None
        lapl = -(kx**2 + kz**2)
        inv_lapl = None


fourier = Fourier()


@jit
@vmap
def phys_to_spec(velocity_phys):
    velocity_spec = _rfft3d(
        velocity_phys,
    )

    return velocity_spec


@jit
@vmap
def spec_to_phys(velocity_spec):

    velocity_phys = _irfft3d(
        velocity_spec,
    )

    return velocity_phys


def cross(vector_1, vector_2):

    return jnp.array(
        [
            vector_1[1] * vector_2[2] - vector_1[2] * vector_2[1],
            vector_1[2] * vector_2[0] - vector_1[0] * vector_2[2],
            vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0],
        ]
    )


def derivative(data_spec, kx, ky, kz, axis):
    match axis:
        case 0:
            return 1j * kx * data_spec
        case 1:
            return 1j * ky * data_spec
        case 2:
            return 1j * kz * data_spec


def divergence(velocity_spec, kx, ky, kz):
    return sum([derivative(velocity_spec[i], kx, ky, kz, i) for i in range(3)])


def curl(velocity_spec, kx, ky, kz):
    return 1j * jnp.array(
        [
            ky * velocity_spec[2] - kz * velocity_spec[1],
            kz * velocity_spec[0] - kx * velocity_spec[2],
            kx * velocity_spec[1] - ky * velocity_spec[0],
        ]
    )


def gradient(data_spec, kx, ky, kz):
    return jnp.array([derivative(data_spec, kx, ky, kz, i) for i in range(3)])


def laplacian(data_spec, lapl_spec):
    return lapl_spec * data_spec


def inverse_laplacian(data_spec, inv_lapl_spec):
    return inv_lapl_spec * data_spec


def integrate_scalar_in_y(scalar_data, ys):
    """
    Composite Simpson's rule for non-uniform grids.
    Requires an odd number of points (even number of intervals).
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
