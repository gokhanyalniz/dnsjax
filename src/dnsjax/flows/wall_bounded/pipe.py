r"""Pipe flow: pressure-driven flow through a circular pipe.

This module defines the ``PipeFlow`` dataclass that holds the
pipe-flow-specific base flow.  Geometry-general infrastructure
(half-CGL grid, parity-reduced FD matrices, IMM operators,
cylindrical IMM iteration, predict / correct / norm, banded /
dense LU solvers) is inherited from
``geometries.wall_bounded.cylindrical.CylindricalFlow``.

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``predict_and_fully_correct_measured`` -- fused step + CFL
  measurements (``steps.dat``)
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

Moving frame
-----------
``phys.u_grid`` defaults to the laminar bulk velocity
`$U_{b,\mathrm{lam}} = 1/2$` for pipe flow, so by default the run
evolves in the frame translating axially at `$U_{grid} = 1/2$` (see
``parameters.update_parameters`` and
``geometries.wall_bounded._base.pad_base_flow``).  Set
``--phys.u_grid 0`` for the lab frame.
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ...geometries.wall_bounded.cylindrical import (
    CylindricalFlow,
    Fourier,
    build_cylindrical_stepper,
    extract_mean_mode,
    fourier,
    get_norm2_cyl,
    get_pert_enstrophy_cyl,
    integrate_scalar,
    pad_base_flow,
)
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class PipeFlow(CylindricalFlow):
    r"""Precomputed data for pipe flow.

    Laminar constants for `$U_z = 1 - r^2$` on `$(0, 1]$`:

    - `$I_{\mathrm{lam}} = D_{\mathrm{lam}} = 2/Re$`
    - `$E_{\mathrm{lam}} = 1/6$`
    - `$U_{b,\mathrm{lam}} = 1/2$`
    """

    I_lam: float = 0.0
    D_lam: float = 0.0
    E_lam: float = 1.0 / 6.0
    U_bulk_lam: float = 0.5

    def __post_init__(self) -> None:
        r"""Build radial grid, base flow, and IMM operators.

        Delegates the half-CGL grid, parity-reduced FD matrices,
        and per-mode IMM operator setup to
        :meth:`CylindricalFlow.__post_init__`, then defines the
        pipe base flow `$U_z = 1 - r^2$` and its derived
        quantities.
        """
        super().__post_init__()
        self.I_lam = 2.0 / params.phys.re
        self.D_lam = self.I_lam

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
        pad_base_flow(self)


flow: PipeFlow = PipeFlow()

(
    predict_and_correct,
    iterate_correction,
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
) = build_cylindrical_stepper(flow)


# ── Diagnostic statistics ────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: PipeFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics.

    - `$E'$`: perturbation kinetic energy (cylindrical norm
      with radial Jacobian `$r$` and `$u_\pm$` half-factor).
    - `$I$`: energy input rate.
    - `$D$`: energy dissipation rate.  For
      `$-\nabla^2 U = 4$` (constant), the cross-enstrophy
      reduces to `$4\,U'_{b,z}$`.
    - `$E$`: total kinetic energy.
    - `$\tau'_z$`, `$\tau'_\theta$`: perturbation wall
      shear stress `$(\partial_r u'_{z,\theta}) / Re$`
      at the pipe wall (`$r = 1$`).
    - `$U'_{b,z}$`, `$U'_{b,\theta}$`: perturbation bulk
      velocity in the axial and azimuthal directions.

    All total-field quantities are computed algebraically
    from perturbation norms and laminar constants, without
    constructing `$\mathbf{u}' + \mathbf{U}$`.
    """
    Re = params.phys.re
    perturbation_energy = (
        get_norm2_cyl(state, fourier_.k_metric, flow_.y_weights) / 2
    )

    # ── Mean velocity profiles ───────────────────────────────
    mean_state = extract_mean_mode(state)  # (3, Nr)
    mean_uz = mean_state[0].real  # (Nr,)
    mean_uplus = mean_state[1]  # (Nr,), complex
    mean_utheta = mean_uplus.imag  # (Nr,)

    # ── Wall shear & bulk velocity ──────────────────────────
    D1_wall_row = flow_.D1_wall.ravel()
    tau_z = jnp.dot(D1_wall_row, mean_uz)
    tau_theta = jnp.dot(D1_wall_row, mean_utheta)
    U_bulk_z = (
        integrate_scalar(mean_uz, flow_.y_weights) / derived_params.volume_fac
    )
    U_bulk_theta = (
        integrate_scalar(mean_utheta, flow_.y_weights)
        / derived_params.volume_fac
    )

    # ── Energy input rate I ─────────────────────────────────
    # CPG: I = I_lam + (4/Re) * Ub'_z
    # CBV: I = I_lam - U_bulk_lam * dPdz'
    dpdz_pert = 2 * tau_z / Re
    I_cpg = flow_.I_lam + 4 * U_bulk_z / Re
    I_cbv = flow_.I_lam - flow_.U_bulk_lam * dpdz_pert
    is_cbv = params.phys.driving == "constant_bulk_velocity"
    energy_input = jnp.where(is_cbv, I_cbv, I_cpg)

    # ── Dissipation D ───────────────────────────────────────
    # D = D_lam + 8 * Ub'_z / Re + Omega'/Re
    pert_enstrophy = get_pert_enstrophy_cyl(
        state,
        flow_.D1_pos,
        flow_.D1_ghost,
        fourier_.m_is_even,
        flow_.inv_r,
        fourier_.m,
        fourier_.kz2,
        fourier_.k_metric,
        flow_.y_weights,
    )
    dissipation = flow_.D_lam + 8 * U_bulk_z / Re + pert_enstrophy / Re

    # ── Total energy E ──────────────────────────────────────
    # E = E_lam + <U . u'> + E'
    # Only U_z contributes (base flow has no radial/azimuthal).
    base_uz = flow_.base_flow[0, :, 0, 0]  # (Nr,)
    cross = (
        integrate_scalar(base_uz * mean_uz, flow_.y_weights)
        / derived_params.volume_fac
    )
    total_energy = flow_.E_lam + cross + perturbation_energy

    return {
        "E'": perturbation_energy,
        "I": energy_input,
        "D": dissipation,
        "E": total_energy,
        "tau'_z": tau_z / Re,
        "tau'_th": tau_theta / Re,
        "Ub'_z": U_bulk_z,
        "Ub'_th": U_bulk_theta,
    }


def get_stats(state: Array) -> dict[str, Array]:
    """Wrapper around ``_get_stats_jit``."""
    return _get_stats_jit(state, fourier, flow)
