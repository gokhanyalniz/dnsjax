"""Plane Poiseuille (channel) flow: pressure-driven flow between plates.

This module defines the ``PlanePoiseuilleFlow`` dataclass that holds
the plane-Poiseuille-specific base flow.  Geometry-general
infrastructure (CGL grid, FD matrices, IMM operators,
Kleiser-Schumann IMM iteration, predict / correct / norm, banded /
dense LU solvers) is inherited from
``geometries.cartesian.CartesianFlow``.

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``init_state`` -- initial state from laminar or snapshot
- ``get_stats`` -- diagnostic statistics

Base flow
---------
The laminar base flow is `$U_s(y) = 1 - y^2$` on `$y \\in [-1, 1]$`,
oriented in the streamwise direction
`$(\cos\theta, 0, \sin\theta)$` where `$\theta$` is the tilt
angle.  Its derived quantities:

- `$dU_s/dy = -2y$`
- `$\nabla \times \mathbf{U} = (-2y\sin\theta, 0, 2y\cos\theta)$`
- `$\mathbf{U} \times \nabla \times \mathbf{U}
  = (0,\; -2y(1-y^2),\; 0)$` (tilt-independent)

Driving
-------
With ``driving = "constant_pressure_gradient"`` (default), the base
flow is maintained by a fixed mean pressure gradient and the
perturbation pressure gradient is a diagnostic output.

With ``driving = "constant_bulk_velocity"``, each IMM iteration
adjusts the mean-mode streamwise velocity to maintain zero
perturbation bulk velocity; the perturbation pressure gradient is
the diagnostic quantity.
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ..bench import timer
from ..geometries.cartesian import (
    CartesianFlow,
    Fourier,
    build_cartesian_stepper,
    fourier,
    get_norm2,
)
from ..parameters import derived_params, params
from ..sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class PlanePoiseuilleFlow(CartesianFlow):
    """Precomputed data for plane Poiseuille flow."""

    def __post_init__(self) -> None:
        r"""Build CGL grid, base flow, and IMM operators.

        Delegates the CGL grid, FD matrices, and per-mode IMM
        operator setup to :meth:`CartesianFlow.__post_init__`,
        which assembles and factorises `$L_k$`, `$H_k$` directly
        on the device.  This method then defines the
        plane-Poiseuille base flow
        `$\mathbf{U} = (1-y^2)(\cos\theta, 0, \sin\theta)$`
        and its derived quantities.
        """
        super().__post_init__()

        Us = 1.0 - self.ys**2  # U_s(y) = 1 - y^2
        dy_Us = -2.0 * self.ys  # dU_s/dy = -2y

        self.base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[0]
            .set(Us * derived_params.cos_tilt)
            .at[2]
            .set(Us * derived_params.sin_tilt)[:, :, None, None]
        )
        # curl(U) = (dy_Us sin θ, 0, -dy_Us cos θ)
        self.curl_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[0]
            .set(dy_Us * derived_params.sin_tilt)
            .at[2]
            .set(-dy_Us * derived_params.cos_tilt)[:, :, None, None]
        )
        # U x curl(U) = (0, -2y(1-y^2), 0) — tilt-independent
        self.nonlin_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[1]
            .set(Us * dy_Us)[:, :, None, None]
        )


flow: PlanePoiseuilleFlow = PlanePoiseuilleFlow()

(
    predict_and_correct,
    iterate_correction,
    init_state,
    predict_and_fully_correct,
) = build_cartesian_stepper(flow)


# ── Diagnostic statistics ────────────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: PlanePoiseuilleFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics: `$E'$`, `$dP'/ds$`.

    - Perturbation kinetic energy
      `$E' = \|\mathbf{u}'\|^2 / 2$`.
    - Perturbation mean streamwise pressure gradient
      `$dP'/ds = (\partial_y u'_s|_{y=1}
      - \partial_y u'_s|_{y=-1}) / (2\,\mathrm{Re})$`
      where `$u'_s = u'_x \cos\theta + u'_z \sin\theta$`
      at the mean mode.
    """
    perturbation_energy = (
        get_norm2(state, fourier_.k_metric, flow_.y_weights) / 2
    )

    # Wall shear of mean-mode streamwise perturbation velocity.
    # D1_bnd rows: [0] = bottom wall (y=-1), [-1] = top wall (y=1).
    u_wall_shear = jnp.einsum("bj, zxj -> zxb", flow_.D1_bnd, state[0])
    w_wall_shear = jnp.einsum("bj, zxj -> zxb", flow_.D1_bnd, state[2])
    us_wall_shear = (
        u_wall_shear * derived_params.cos_tilt
        + w_wall_shear * derived_params.sin_tilt
    )

    # TODO: Measure spanwise wall shear

    # Extract mean mode via masked sum (kx-sharded).
    mean_us_shear = jnp.sum(
        jnp.where(
            fourier_.k2_is_zero,
            us_wall_shear,
            0.0,
        ),
        axis=(0, 1),
    ).real

    # dP'/ds = (dy_us'|_{y=1} - dy_us'|_{y=-1}) / (2*Re)
    dpds_pert = (mean_us_shear[1] - mean_us_shear[0]) / (2 * params.phys.re)

    stats = {
        "E'": perturbation_energy,
        "dPds'": dpds_pert,
    }

    return stats


@timer("stats")
def get_stats(state: Array) -> dict[str, Array]:
    """Bench-timed wrapper around ``_get_stats_jit``."""
    return _get_stats_jit(state, fourier, flow)
