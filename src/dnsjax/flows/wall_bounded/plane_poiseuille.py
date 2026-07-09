r"""Plane Poiseuille (channel) flow: pressure-driven flow between plates.

This module defines the ``PlanePoiseuilleFlow`` dataclass that holds
the plane-Poiseuille-specific base flow.  Geometry-general
infrastructure (CGL grid, FD matrices, IMM operators,
Kleiser-Schumann IMM iteration, predict / correct / norm, Pallas /
dense LU solvers) is inherited from
``geometries.wall_bounded.cartesian.CartesianFlow``.

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``predict_and_fully_correct_measured`` -- fused step + CFL
  measurements (``steps.dat``)
- ``init_state`` -- initial state from a snapshot or laminar
- ``get_stats`` -- diagnostic statistics

Base flow
---------
The laminar base flow is `$U_s(y) = 1 - y^2$` on `$y \in [-1, 1]$`,
oriented in the streamwise direction
`$(\cos\theta, 0, \sin\theta)$` where `$\theta$` is the tilt
angle.  Its derived quantities:

- `$dU_s/dy = -2y$`
- `$\nabla \times \mathbf{U} = (-2y\sin\theta, 0, 2y\cos\theta)$`
- `$\mathbf{U} \times \nabla \times \mathbf{U}
  = (0,\; -2y(1-y^2),\; 0)$` (tilt-independent)

Moving frame
------------
``phys.u_grid`` defaults to the laminar bulk velocity
`$U_{b,\mathrm{lam}} = 2/3$` for plane-Poiseuille flow, so by default
the run evolves in the frame translating in `$x$` at
`$U_{grid} = 2/3$`: the convective frame term `$+ i k_x U_{grid}
\mathbf{u}'$` is added in the Cartesian ``_get_rhs_core`` / ``_l_bf``
and the CFL diagnostic advects with `$\mathbf{U} -
U_{grid}\hat{\mathbf{x}}$` (see ``parameters.update_parameters`` and
``geometries.wall_bounded._base.pad_base_flow``).  Set
``--phys.u_grid 0`` for the lab frame.

Driving
-------
With ``driving = "constant_pressure_gradient"`` (default), the base
flow is maintained by a fixed mean pressure gradient and the
perturbation pressure gradient is a diagnostic output.

With ``driving = "constant_bulk_velocity"``, each IMM iteration
adjusts the mean-mode streamwise velocity to maintain zero
perturbation bulk velocity; the perturbation pressure gradient is
the diagnostic quantity.

Spanwise blocking
-----------------
With ``block_mean_spanwise_velocity = True``, each IMM iteration
additionally zeroes the perturbation bulk velocity in the spanwise
direction `$(-\sin\theta, 0, \cos\theta)$`, simulating the
presence of sidewalls that prevent net spanwise momentum.  This
option is independent of ``driving`` and uses the same Helmholtz
response as the streamwise constant-bulk-velocity enforcement.
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ...geometries.wall_bounded.cartesian import (
    CartesianFlow,
    Fourier,
    build_cartesian_stepper,
    extract_mean_mode,
    fourier,
    get_norm2,
    get_pert_enstrophy,
    integrate_scalar,
    pad_base_flow,
    tilted_profile_arrays,
)
from ...geometries.wall_bounded.cartesian import (
    frozen_profile_flow as _frozen_flow_copy,
)
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree


@register_dataclass_pytree
@dataclass
class PlanePoiseuilleFlow(CartesianFlow):
    r"""Precomputed data for plane Poiseuille flow.

    Laminar constants for `$U_s = 1 - y^2$` on `$[-1, 1]$`:

    - `$I_{\mathrm{lam}} = D_{\mathrm{lam}} = 4/(3\,Re)$`
    - `$E_{\mathrm{lam}} = 4/15$`
    - `$U_{b,\mathrm{lam}} = 2/3$`
    """

    I_lam: float = 0.0
    D_lam: float = 0.0
    E_lam: float = 4.0 / 15.0
    U_bulk_lam: float = 2.0 / 3.0

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
        self.I_lam = 4.0 / (3.0 * params.phys.re)
        self.D_lam = self.I_lam

        Us = 1.0 - self.ys**2  # U_s(y) = 1 - y^2
        dy_Us = -2.0 * self.ys  # dU_s/dy = -2y
        self.base_flow, self.curl_base_flow = tilted_profile_arrays(Us, dy_Us)
        pad_base_flow(self)


flow: PlanePoiseuilleFlow = PlanePoiseuilleFlow()

(
    predict_and_correct,
    iterate_correction,
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
) = build_cartesian_stepper(flow)


def frozen_profile_flow(us: Array) -> PlanePoiseuilleFlow:
    r"""Flow linearized around an arbitrary streamwise profile.

    Transient-growth hook (:mod:`dnsjax.analysis.transient_growth`):
    given the *total* streamwise profile `$U_s(y)$` on the code grid
    (``flow.ys``, shape ``(Ny,)``), tilt-split it exactly as the
    laminar `$U_s = 1 - y^2$` (:func:`tilted_profile_arrays`),
    differentiate with the flow's FD `$D_1$` for
    `$\nabla\times\mathbf{U}$`, and return a flow copy carrying that
    base flow (all operators shared; see
    :func:`~dnsjax.geometries.wall_bounded._base.frozen_profile_flow`).
    """
    dy_us = flow.D1 @ us
    base, curl = tilted_profile_arrays(us, dy_us)
    return _frozen_flow_copy(flow, base, curl)


# ── Diagnostic statistics ────────────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: PlanePoiseuilleFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics.

    - `$E'$`: perturbation kinetic energy
      `$\|\mathbf{u}'\|^2 / 2$`.
    - `$I$`: energy input rate
      `$\langle (\mathbf{u} + \mathbf{U}) \cdot
      (-\nabla \Pi) \rangle$`.
    - `$D$`: energy dissipation rate
      `$\langle |\nabla \times (\mathbf{u}' +
      \mathbf{U})|^2 \rangle / Re$`.
    - `$E$`: total kinetic energy
      `$\langle |\mathbf{u}' + \mathbf{U}|^2 \rangle / 2$`.
    - `$\tau'_{s,b/t}$`, `$\tau'_{n,b/t}$`: perturbation
      wall shear stress `$(\partial_y u'_{s,n}) / Re$` at
      the bottom (`$y=-1$`) and top (`$y=1$`) walls.
    - `$U'_{b,s}$`, `$U'_{b,n}$`: perturbation bulk
      velocity in the streamwise and spanwise directions.

    All total-field quantities are computed algebraically
    from perturbation norms and laminar constants, without
    constructing `$\mathbf{u}' + \mathbf{U}$`.  For
    `$-\nabla^2 U = 2$` (constant), the cross-enstrophy
    reduces to
    `$\langle \boldsymbol{\omega}_U \cdot
    \boldsymbol{\omega}_{u'} \rangle
    = 2\,U'_{b,s}$`
    where `$U'_{b,s}$` is the perturbation bulk streamwise
    velocity.
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
    # CPG: I = I_lam + (2/Re) * Ub'_s
    # CBV: I = I_lam - U_bulk_lam * dPds'
    dpds_pert = (mean_us_shear[1] - mean_us_shear[0]) / (2 * Re)
    I_cpg = flow_.I_lam + 2 * U_bulk_s / Re
    I_cbv = flow_.I_lam - flow_.U_bulk_lam * dpds_pert
    is_cbv = params.phys.driving == "constant_bulk_velocity"
    energy_input = jnp.where(is_cbv, I_cbv, I_cpg)

    # ── Dissipation D ───────────────────────────────────────
    # D = D_lam + 4 * Ub'_s / Re + Omega'/Re
    pert_enstrophy = get_pert_enstrophy(
        state,
        flow_.D1,
        fourier_.k2,
        fourier_.k_metric,
        flow_.y_weights,
    )
    dissipation = flow_.D_lam + 4 * U_bulk_s / Re + pert_enstrophy / Re

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


@jit
def _get_perturbation_energy_jit(
    state: Array, fourier_: Fourier, flow_: PlanePoiseuilleFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`."""
    return get_norm2(state, fourier_.k_metric, flow_.y_weights) / 2


def get_perturbation_energy(state: Array) -> Array:
    """Perturbation kinetic energy E' (for the laminarization check)."""
    return _get_perturbation_energy_jit(state, fourier, flow)
