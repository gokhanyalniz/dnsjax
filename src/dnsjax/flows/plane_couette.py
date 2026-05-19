"""Plane Couette flow: wall-bounded shear between two moving plates.

This module defines the ``PlaneCouetteFlow`` dataclass that holds the
plane-Couette-specific base flow.  Geometry-general infrastructure
(CGL grid, FD matrices, IMM operators, Kleiser-Schumann IMM
iteration, predict / correct / norm, banded / dense LU solvers) is
inherited from ``geometries.cartesian.CartesianFlow``.

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
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

Spanwise blocking
-----------------
With ``block_mean_spanwise_velocity = True``, each IMM iteration
zeroes the perturbation bulk velocity in the spanwise direction
`$(-\\sin\\theta, 0, \\cos\\theta)$`, simulating the presence of
sidewalls that prevent net spanwise momentum.  This option is
independent of ``driving`` and uses the same Helmholtz response
as the streamwise constant-bulk-velocity enforcement.
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ..geometries.cartesian import (
    CartesianFlow,
    Fourier,
    build_cartesian_stepper,
    extract_mean_mode,
    fourier,
    get_norm2,
    get_pert_enstrophy,
    integrate_scalar,
)
from ..parameters import derived_params, params
from ..sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class PlaneCouetteFlow(CartesianFlow):
    r"""Precomputed data for plane Couette flow.

    Laminar constants for `$U_s = y$` on `$[-1, 1]$`:

    - `$I_{\mathrm{lam}} = D_{\mathrm{lam}} = 1/Re$`
    - `$E_{\mathrm{lam}} = 1/6$`
    - `$U_{b,\mathrm{lam}} = 0$`
    """

    I_lam: float = 0.0
    D_lam: float = 0.0
    E_lam: float = 1.0 / 6.0
    U_bulk_lam: float = 0.0

    def __post_init__(self) -> None:
        r"""Build CGL grid, base flow, and IMM operators.

        Delegates the CGL grid, FD matrices, and per-mode IMM
        operator setup to :meth:`CartesianFlow.__post_init__`,
        which assembles and factorises `$L_k$`, `$H_k$` directly
        on the device.  This method then defines the
        plane-Couette base flow
        `$\mathbf{U} = y(\cos\theta, 0, \sin\theta)$`
        and its derived quantities.
        """
        super().__post_init__()
        self.I_lam = 1.0 / params.phys.re
        self.D_lam = self.I_lam

        Us = self.ys.copy()  # U_s(y) = y
        dy_Us = jnp.ones(params.res.ny, dtype=sharding.float_type)

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
        # U x curl(U) = (0, y, 0) — tilt-independent
        self.nonlin_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[1]
            .set(Us * dy_Us)[:, :, None, None]
        )


flow: PlaneCouetteFlow = PlaneCouetteFlow()

(
    predict_and_correct,
    iterate_correction,
    init_state,
    predict_and_fully_correct,
) = build_cartesian_stepper(flow)


# ── Diagnostic statistics ────────────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: PlaneCouetteFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics.

    - `$E'$`: perturbation kinetic energy.
    - `$I$`: energy input rate from wall shear:
      `$I = I_{\mathrm{lam}} + (\partial_y u'_s|_{y=1}
      + \partial_y u'_s|_{y=-1}) / (2\,Re)$`.
    - `$D$`: energy dissipation rate.  Since
      `$\nabla^2 U = 0$`, cross-enstrophy vanishes:
      `$D = D_{\mathrm{lam}} + \Omega'/Re$`.
    - `$E$`: total kinetic energy.
    - `$\tau'_{s,b/t}$`, `$\tau'_{n,b/t}$`: perturbation
      wall shear stress `$(\partial_y u'_{s,n}) / Re$` at
      the bottom (`$y=-1$`) and top (`$y=1$`) walls.
    - `$U'_{b,s}$`, `$U'_{b,n}$`: perturbation bulk
      velocity in the streamwise and spanwise directions.
    """
    Re = params.phys.re
    perturbation_energy = (
        get_norm2(state, fourier_.k_metric, flow_.y_weights) / 2
    )

    # ── Mean velocity profiles ─────────────────────────────
    mean_u = extract_mean_mode(state).real  # (3, Ny)
    mean_us = (
        mean_u[0] * derived_params.cos_tilt
        + mean_u[2] * derived_params.sin_tilt
    )  # (Ny,)
    mean_un = (
        -mean_u[0] * derived_params.sin_tilt
        + mean_u[2] * derived_params.cos_tilt
    )  # (Ny,)

    # ── Wall shear & bulk velocity ─────────────────────────
    mean_us_shear = flow_.D1_bnd @ mean_us  # (2,)
    mean_un_shear = flow_.D1_bnd @ mean_un  # (2,)
    U_bulk_s = (
        integrate_scalar(mean_us, flow_.y_weights) / derived_params.volume_fac
    )
    U_bulk_n = (
        integrate_scalar(mean_un, flow_.y_weights) / derived_params.volume_fac
    )

    # ── Energy input rate I ─────────────────────────────────
    # I = I_lam + (dy_us'|{y=1} + dy_us'|{y=-1}) / (2*Re)
    wall_shear_sum = (mean_us_shear[1] + mean_us_shear[0]) / (2 * Re)
    energy_input = flow_.I_lam + wall_shear_sum

    # ── Dissipation D ───────────────────────────────────────
    # nabla^2 U = 0 => cross-enstrophy = 0
    # D = D_lam + Omega'/Re
    pert_enstrophy = get_pert_enstrophy(
        state,
        flow_.D1,
        fourier_.k2,
        fourier_.k_metric,
        flow_.y_weights,
    )
    dissipation = flow_.D_lam + pert_enstrophy / Re

    # ── Total energy E ──────────────────────────────────────
    # E = E_lam + <U . u'> + E'
    base = flow_.base_flow[:, :, 0, 0]  # (3, Ny)
    cross = (
        integrate_scalar(jnp.sum(base * mean_u, axis=0), flow_.y_weights)
        / derived_params.volume_fac
    )
    total_energy = flow_.E_lam + cross + perturbation_energy

    return {
        "E'": perturbation_energy,
        "I": energy_input,
        "D": dissipation,
        "E": total_energy,
        "tau'_s,b": mean_us_shear[0] / Re,
        "tau'_s,t": mean_us_shear[1] / Re,
        "tau'_n,b": mean_un_shear[0] / Re,
        "tau'_n,t": mean_un_shear[1] / Re,
        "Ub'_s": U_bulk_s,
        "Ub'_n": U_bulk_n,
    }


def get_stats(state: Array) -> dict[str, Array]:
    """Wrapper around ``_get_stats_jit``."""
    return _get_stats_jit(state, fourier, flow)
