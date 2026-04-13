"""Pipe flow: pressure-driven flow in a circular cross-section.

The base flow is the Hagen-Poiseuille profile ``U(r) = 1 - r^2``.  The
radial grid uses the upper half of a Chebyshev-Gauss-Lobatto grid,
mapping ``r in [0, 1]``.

.. note:: This module is a stub.  Only laminar-state diagnostics are
   defined; time-stepping, forcing, and the pressure solve are not yet
   implemented.
"""

from dataclasses import dataclass

from jax import Array
from jax import numpy as jnp

from ..parameters import params
from ..sharding import sharding


@dataclass
class PipeFlow:
    """Precomputed data for pipe flow (stub)."""

    # Radial grid: upper half of a Chebyshev grid on [0, 1]
    ys: Array = -jnp.cos(
        jnp.arange(2 * params.res.ny + 1, dtype=sharding.float_type)
        * jnp.pi
        / (2 * params.res.ny)
    )[-params.res.ny :]

    base_flow: Array = 1 - ys**2  # U(r) = 1 - r^2
    dy_base_flow: Array = -2 * ys  # dU/dr = -2r
    ekin_lam: float = 1 / 6
    input_lam: float = 2 / params.phys.re
    dissip_lam: float = 2 / params.phys.re
    pres_grad_lam: float = 4 / params.phys.re
    ubulk_lam: float = 1 / 2


flow: PipeFlow = PipeFlow()
