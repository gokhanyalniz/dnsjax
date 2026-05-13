r"""Pipe flow: pressure-driven flow through a circular pipe.

This module defines the ``PipeFlow`` dataclass that holds the
pipe-flow-specific base flow.  Geometry-general infrastructure
(half-CGL grid, parity-reduced FD matrices, IMM operators,
cylindrical IMM iteration, predict / correct / norm, banded /
dense LU solvers) is inherited from
``geometries.cylindrical.CylindricalFlow``.

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``init_state`` -- initial state from laminar or snapshot
- ``get_stats`` -- diagnostic statistics

Base flow
---------
The laminar base flow is
`$\mathbf{U} = (1 - r^2, 0, 0)$` in `$(u_z, u_r, u_\theta)$`
on `$r \in (0, 1]$`, the Hagen-Poiseuille parabolic profile.

- `$dU_z/dr = -2r$`
- `$\nabla \times \mathbf{U} = (0,\; 0,\; 2r)$`
  (only `$\omega_\theta = -dU_z/dr = 2r$`)
- `$\mathbf{U} \times \nabla \times \mathbf{U}
  = (0,\; -2r(1 - r^2),\; 0)$`
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ..bench import timer
from ..geometries.cylindrical import (
    CylindricalFlow,
    Fourier,
    build_cylindrical_stepper,
    fourier,
    get_norm2_cyl,
)
from ..parameters import params
from ..sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class PipeFlow(CylindricalFlow):
    """Precomputed data for pipe flow."""

    def __post_init__(self) -> None:
        r"""Build radial grid, base flow, and IMM operators.

        Delegates the half-CGL grid, parity-reduced FD matrices,
        and per-mode IMM operator setup to
        :meth:`CylindricalFlow.__post_init__`, then defines the
        pipe base flow `$U_z = 1 - r^2$` and its derived
        quantities.
        """
        super().__post_init__()

        rs = self.rs  # (Nr,) on (0, 1]

        # Base flow: U = (1 - r^2, 0, 0) in (uz, ur, utheta).
        Uz = 1.0 - rs**2
        self.base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[0]
            .set(Uz)[:, :, None, None]
        )

        # curl(U) = (0, 0, 2r): omega_theta = -dUz/dr = 2r.
        omega_theta = 2.0 * rs
        self.curl_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[2]
            .set(omega_theta)[:, :, None, None]
        )

        # U x curl(U) = (0, -2r(1-r^2), 0).
        nonlin_r = -2.0 * rs * (1.0 - rs**2)
        self.nonlin_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[1]
            .set(nonlin_r)[:, :, None, None]
        )


flow: PipeFlow = PipeFlow()

(
    predict_and_correct,
    iterate_correction,
    init_state,
    predict_and_fully_correct,
) = build_cylindrical_stepper(flow)


# ── Diagnostic statistics ────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: PipeFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics: `$E'$`, `$dP'/dz$`.

    - Perturbation kinetic energy with the cylindrical norm
      (radial Jacobian `$r$` and `$u_\pm$` half-factor).
    - Perturbation mean pressure gradient from the global
      momentum balance:
      `$dP'/dz = 2\,\partial_r u'_z|_{r=1} / \mathrm{Re}$`.
    """
    perturbation_energy = (
        get_norm2_cyl(state, fourier_.k_metric, flow_.y_weights) / 2
    )

    D1_wall_row = flow_.D1_wall.ravel()
    wall_shear_all = jnp.einsum("j, mzj -> mz", D1_wall_row, state[0])
    dpdz_pert = (
        2
        * jnp.sum(
            jnp.where(fourier_.k2_is_zero[..., 0], wall_shear_all, 0.0)
        ).real
        / params.phys.re
    )

    stats = {
        "E'": perturbation_energy,
        "dPdz'": dpdz_pert,
    }

    return stats


@timer("stats")
def get_stats(state: Array) -> dict[str, Array]:
    """Bench-timed wrapper around ``_get_stats_jit``."""
    return _get_stats_jit(state, fourier, flow)
