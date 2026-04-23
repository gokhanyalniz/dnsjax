"""Plane Couette flow: wall-bounded shear between two moving plates.

This module defines the ``PlaneCouetteFlow`` dataclass that holds the
plane-Couette-specific base flow.  Geometry-general infrastructure
(CGL grid, FD matrices, IMM operators, Kleiser-Schumann IMM
iteration, predict / correct / norm, banded / dense LU solvers) is
inherited from ``geometries.cartesian.CartesianFlow``.

It also exports the full flow interface consumed by ``__main__``:

- ``predict_and_correct`` / ``iterate_correction`` -- time stepping
- ``init_state`` -- initial state from laminar or snapshot
- ``get_stats`` -- diagnostic statistics

Unlike the triply-periodic interface, no ``correct_velocity`` is
exported: the influence-matrix method enforces `$\\nabla \\cdot
\\mathbf{u} = 0$` and the no-slip wall BCs exactly at every time
step, so no separate divergence projection is required.

Base flow
---------
The laminar base flow is `$U(y) = y$` on `$y \\in [-1, 1]$`, with
the walls moving at `$\\pm 1$`.  Its derived quantities:

- `$dU_x/dy = 1$`
- `$\\nabla \\times \\mathbf{U} = (0, 0, -1)$`
- `$\\mathbf{U} \\times \\nabla \\times \\mathbf{U} = (0, y, 0)$`
"""

from dataclasses import dataclass
from typing import Any

from jax import Array, jit
from jax import numpy as jnp

from ..bench import timer
from ..geometries.cartesian import (
    CartesianFlow,
    build_cartesian_stepper,
    fourier,
    get_norm2,
)
from ..parameters import params
from ..sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class PlaneCouetteFlow(CartesianFlow):
    """Precomputed data for plane Couette flow."""

    def __post_init__(self) -> None:
        """Build CGL grid, base flow, and IMM operators.

        Constructs the Chebyshev-Gauss-Lobatto grid for
        the wall-normal coordinate ``y`` in ``[-1, 1]``,
        the laminar base flow ``U(y) = y`` and its derived
        quantities, FD matrices D1 and D2, and all per-mode
        IMM operators via ``IMMChunker``.
        """
        super().__post_init__()

        base_flow_np = self.ys.copy()
        dy_base_flow_np = jnp.ones(params.res.ny, dtype=sharding.float_type)

        self.base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[0]
            .set(base_flow_np)[:, :, None, None]
        )
        self.curl_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[2]
            .set(-dy_base_flow_np)[:, :, None, None]
        )
        self.nonlin_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[1]
            .set(base_flow_np * dy_base_flow_np)[:, :, None, None]
        )


flow: PlaneCouetteFlow = PlaneCouetteFlow()

predict_and_correct, iterate_correction, init_state = build_cartesian_stepper(
    flow
)


# ── Diagnostic statistics ────────────────────────────────────────────────
#
@jit
def _get_stats_jit(
    state: Array, fourier_: Any, flow_: Any
) -> dict[str, Array]:
    """Compute diagnostic statistics: E'.

    Only the perturbation kinetic energy is currently computed.  The
    commented-out scaffolding below marks where total-energy, input,
    and dissipation analogues (present for monochromatic flows) would
    be wired in once implemented for wall-bounded flows.
    """
    # Perturbation kinetic energy: `$E' = \\|\\mathbf{u}'\\|^2 / 2$`.
    perturbation_energy = get_norm2(state, fourier_.k_metric, flow_.ys) / 2
    # input = get_input(state, fourier_, flow_)
    # dissipation = get_dissipation(state, input, fourier_, flow_)
    # energy = get_energy(perturbation_energy, input, fourier_, flow_)

    stats = {
        # "E": energy,
        # "I": input,
        # "D": dissipation,
        "E'": perturbation_energy,
    }

    return stats


@timer("stats")
def get_stats(state: Array) -> dict[str, Array]:
    """Bench-timed wrapper around ``_get_stats_jit``."""
    return _get_stats_jit(state, fourier, flow)
