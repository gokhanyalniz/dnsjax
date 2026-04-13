"""Plane Poiseuille (channel) flow: pressure-driven flow between fixed walls.

The base flow is the parabolic profile ``U(y) = 1 - y^2`` on the
Chebyshev-Gauss-Lobatto grid ``y in [-1, 1]``.

.. note:: This module is a stub.  Only laminar-state diagnostics are
   defined; time-stepping, forcing, and the influence-matrix pressure
   solve are not yet implemented.
"""

from dataclasses import dataclass

from jax import Array
from jax import numpy as jnp

from ..parameters import params
from ..sharding import sharding


@dataclass
class PlaneChannelFlow:
    """Precomputed data for plane Poiseuille flow (stub)."""

    # Chebyshev-Gauss-Lobatto grid on [-1, 1]
    ys: Array = -jnp.cos(
        jnp.arange(params.res.ny, dtype=sharding.float_type)
        * jnp.pi
        / (params.res.ny - 1)
    )

    base_flow: Array = 1 - ys**2  # U(y) = 1 - y^2
    dy_base_flow: Array = -2 * ys  # dU/dy = -2y
    ekin_lam: float = 4 / 15
    input_lam: float = 4 / (3 * params.phys.re)
    dissip_lam: float = 4 / (3 * params.phys.re)
    pres_grad_lam: float = 2 / params.phys.re
    ubulk_lam: float = 2 / 3


flow: PlaneChannelFlow = PlaneChannelFlow()
