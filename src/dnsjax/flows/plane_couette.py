"""Plane Couette flow: wall-bounded shear between two moving plates.

The base flow is `$U(y) = y$` where `$y \\in [-1, 1]$`,
with walls moving at `$\\pm 1$`.
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
    get_perturbation_energy,
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
    """Compute diagnostic statistics: E, I, D, E'."""
    perturbation_energy = get_perturbation_energy(state, fourier_, flow_)
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
    return _get_stats_jit(state, fourier, flow)
