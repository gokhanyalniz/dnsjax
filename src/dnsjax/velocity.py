"""L2 inner product, squared norm, and norm for velocity fields.

The inner product is the volume-averaged integral
``<u1, u2> = (1/V) int u1* . u2 dV``, computed in spectral space using
Parseval's theorem.  The ``k_metric`` factor accounts for the real-FFT
Hermitian symmetry in x.  For wall-bounded flows the y-integral is
performed via Simpson's rule; for pipe flow the radial weight *r* is
included.
"""

from jax import Array, numpy as jnp

from .operators import (
    integrate_scalar_in_y,
)
from .parameters import (
    cartesian_systems,
    derived_params,
    params,
    periodic_systems,
)
from .sharding import sharding


def get_inprod(
    vector_spec_1: Array,
    vector_spec_2: Array,
    k_metric: Array,
    ys: Array | None,
) -> Array:
    """Volume-averaged L2 inner product ``<u1, u2>`` in spectral space.

    For triply-periodic flows the sum is a direct Parseval sum over all
    Fourier modes.  For Cartesian walled flows the Fourier modes in x
    and z are summed first, then the resulting y-profile is integrated
    with Simpson's rule.  For pipe flow the integrand additionally
    carries the radial weight *r* from the cylindrical measure.

    Parameters
    ----------
    vector_spec_1, vector_spec_2:
        Spectral velocity fields, shape ``(3, *spec_shape)``.
    k_metric:
        Real-FFT symmetry weight (1 for kx=0, 2 for kx>0).
    ys:
        Wall-normal grid points (``None`` for periodic flows).
    """
    system = params.phys.system
    if system in periodic_systems:
        # 1/(Lx Ly Lz) \int dx dy dz u1 * u2
        return jnp.sum(
            jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
            dtype=sharding.float_type,
        )
    elif system in cartesian_systems:
        return (
            # 1/(Lx Ly Lz) \int dx dy dz u1 * u2
            integrate_scalar_in_y(
                jnp.sum(
                    jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
                    dtype=sharding.float_type,
                    axis=(0, 2, 3),
                ),
                ys,
            )
            / derived_params.ly
        )
    elif system == "pipe":
        # 1/(Lx \pi R^2) \int r dr d\theta dx u1 * u2
        return (
            integrate_scalar_in_y(
                ys
                * 2
                * jnp.sum(
                    jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
                    dtype=sharding.float_type,
                    axis=(0, 2, 3),
                ),
                ys,
            )
            / (derived_params.ly / 2) ** 2
        )
    else:
        raise NotImplementedError


def get_norm2(
    vector_spec: Array, k_metric: Array, ys: Array | None
) -> Array:
    """Squared L2 norm ``||u||^2 = <u, u>``."""
    return get_inprod(vector_spec, vector_spec, k_metric, ys)


def get_norm(
    vector_spec: Array, k_metric: Array, ys: Array | None
) -> Array:
    """L2 norm ``||u|| = sqrt(<u, u>)``."""
    return jnp.sqrt(get_norm2(vector_spec, k_metric, ys))
