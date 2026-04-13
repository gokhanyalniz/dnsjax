"""Plane Couette flow: wall-bounded shear between two moving plates.

The base flow is ``U(y) = y`` on the Chebyshev-Gauss-Lobatto grid
``y in [-1, 1]``, with walls moving at ``+/-1``.

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
class PlaneCouetteFlow:
    """Precomputed data for plane Couette flow (stub)."""

    # Chebyshev-Gauss-Lobatto grid on [-1, 1]
    ys: Array = -jnp.cos(
        jnp.arange(params.res.ny, dtype=sharding.float_type)
        * jnp.pi
        / (params.res.ny - 1)
    )

    base_flow: Array = jnp.copy(ys)  # U(y) = y
    dy_base_flow: Array = jnp.ones(params.res.ny, dtype=sharding.float_type)
    ekin_lam: float = 1 / 6
    input_lam: float = 1 / params.phys.re
    dissip_lam: float = 1 / params.phys.re
    pres_grad_lam: float = 0
    ubulk_lam: float = 0


flow: PlaneCouetteFlow = PlaneCouetteFlow()
