r"""Taylor-Couette flow: shear-driven flow in a rotating annulus.

This module defines the ``TaylorCouetteFlow`` dataclass that holds the
Taylor-Couette-specific base flow.  Geometry-general infrastructure
(radial grid on `$[r_1, r_2]$`, FD matrices, IMM operators, the
`$2 \times 2$` annular IMM iteration, predict / correct / norm, banded /
dense LU solvers) is inherited from
``geometries.wall_bounded.annular.AnnularFlow``.

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``predict_and_fully_correct_measured`` -- fused step + CFL
  measurements (``steps.dat``)
- ``init_state`` -- initial state from laminar or snapshot
- ``get_stats`` -- diagnostic statistics

Like plane-Couette (and unlike the pressure-driven pipe), the flow is
**shear-driven** by the rotating walls; its statistics use the same
energy-budget structure.

Base flow
---------
The laminar base flow is the azimuthal circular-Couette profile
`$\mathbf{U} = (0,\,0,\,U_\theta)$` in `$(u_z, u_r, u_\theta)$` with
`$U_\theta(r) = A_0 r + B_0/r$` on `$r \in [r_1, r_2]$`, where the
coefficients `$A_0$`, `$B_0$` and radii `$r_1, r_2$` follow from the
control parameters `$(\mathrm{Re}_1, \mathrm{Re}_2, \eta)$` (see the
annular branch of ``parameters.update_parameters``).  Its derived
quantities:

- `$\nabla \times \mathbf{U} = (2 A_0,\,0,\,0)$` (uniform axial
  vorticity `$\omega_z = (1/r)\partial_r(r U_\theta) = 2 A_0$`).
- `$\nabla^2 \mathbf{U} = 0$` (circular-Couette is the steady Stokes
  solution), so in the energy budget the cross-dissipation vanishes and
  `$D = D_{\mathrm{lam}} + \Omega'/\mathrm{Re}$`.

Laminar constants
-----------------
Only the differential-rotation part `$B_0/r$` dissipates (the solid-body
part `$A_0 r$` does not), so

.. math::
    I_{\mathrm{lam}} = D_{\mathrm{lam}}
    = \frac{4 B_0^2}{\mathrm{Re}\,r_1^2 r_2^2}, \qquad
    E_{\mathrm{lam}} = \tfrac{1}{2}\langle U_\theta^2 \rangle.

(In the narrow-gap limit `$\eta \to 1$` these reduce to the
plane-Couette values `$I_{\mathrm{lam}} = D_{\mathrm{lam}} = 1/\mathrm{Re}$`.)
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ...geometries.wall_bounded.annular import (
    AnnularFlow,
    Fourier,
    build_annular_stepper,
    extract_mean_mode,
    fourier,
    get_enstrophy_annular,
    get_norm2_annular,
    integrate_scalar,
    pad_base_flow,
)
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class TaylorCouetteFlow(AnnularFlow):
    r"""Precomputed data for Taylor-Couette flow.

    Laminar constants for `$U_\theta = A_0 r + B_0/r$` on
    `$[r_1, r_2]$` are computed in ``__post_init__`` from the
    circular-Couette coefficients and radii on ``derived_params``.
    """

    I_lam: float = 0.0
    D_lam: float = 0.0
    E_lam: float = 0.0

    def __post_init__(self) -> None:
        r"""Build radial grid, base flow, and IMM operators.

        Delegates the grid, FD matrices, and per-mode IMM operator setup
        to :meth:`AnnularFlow.__post_init__`, then defines the
        circular-Couette base flow `$U_\theta = A_0 r + B_0/r$` and its
        derived quantities and laminar constants.
        """
        super().__post_init__()

        A0 = derived_params.ccf_A
        B0 = derived_params.ccf_B
        Re = params.phys.re
        r1 = derived_params.r_inner
        r2 = derived_params.r_outer

        rs = self.rs  # (Nr,) on [r1, r2]
        u_theta = A0 * rs + B0 / rs

        # Base flow: U = (0, 0, U_theta) in (u_z, u_r, u_theta).
        self.base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[2]
            .set(u_theta)[:, :, None, None]
        )

        # curl(U) = (2 A0, 0, 0): uniform axial vorticity omega_z = 2 A0.
        self.curl_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[0]
            .set(2.0 * A0)[:, :, None, None]
        )
        pad_base_flow(self)

        # Laminar constants.  Only the B0/r (differential-rotation)
        # part dissipates: I_lam = D_lam = 4 B0^2 / (Re r1^2 r2^2).
        self.I_lam = 4.0 * B0**2 / (Re * r1**2 * r2**2)
        self.D_lam = self.I_lam
        # E_lam = <U_theta^2> / 2.
        self.E_lam = float(
            integrate_scalar(u_theta**2, self.y_weights)
            / derived_params.volume_fac
            / 2
        )


flow: TaylorCouetteFlow = TaylorCouetteFlow()

(
    predict_and_correct,
    iterate_correction,
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
) = build_annular_stepper(flow)


# ── Diagnostic statistics ────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: TaylorCouetteFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics.

    - `$E'$`: perturbation kinetic energy (annular norm with radial
      Jacobian `$r$` and `$u_\pm$` half-factor).
    - `$I$`: energy input rate from the wall torques.  At the rotating
      walls the perturbation azimuthal traction
      `$\tau'_\theta = \partial_r u'_\theta / \mathrm{Re}$` does work
      against the wall velocity `$U_\theta(r_{\mathrm{wall}})$`:

      .. math::
          I = I_{\mathrm{lam}} + \frac{r_2 U_\theta(r_2)\,
          \partial_r u'_\theta|_{r_2}
          - r_1 U_\theta(r_1)\,\partial_r u'_\theta|_{r_1}}
          {\mathrm{Re}\,\mathrm{volfac}}.

    - `$D$`: energy dissipation rate.  Since `$\nabla^2 U = 0$` the
      cross-dissipation vanishes: `$D = D_{\mathrm{lam}}
      + \Omega'/\mathrm{Re}$`.
    - `$E$`: total kinetic energy `$E_{\mathrm{lam}} + \langle U \cdot
      u' \rangle + E'$` (only `$U_\theta$` contributes to the cross
      term).
    - `$\tau'_{\theta,i/o}$`, `$\tau'_{z,i/o}$`: perturbation wall shear
      stresses at the inner / outer walls.
    - `$U'_{b,\theta}$`, `$U'_{b,z}$`: perturbation bulk azimuthal and
      axial velocities.
    """
    Re = params.phys.re
    volfac = derived_params.volume_fac
    perturbation_energy = (
        get_norm2_annular(state, fourier_.k_metric, flow_.y_weights) / 2
    )

    # ── Mean velocity profiles ───────────────────────────────
    mean_state = extract_mean_mode(state)  # (3, Nr)
    mean_uz = mean_state[0].real  # (Nr,)
    mean_utheta = mean_state[1].imag  # u_theta = Im(u_+), (Nr,)

    # ── Wall shear & bulk velocity ──────────────────────────
    # D1_bnd @ profile -> [inner (r1), outer (r2)] wall-normal deriv.
    tau_theta = flow_.D1_bnd @ mean_utheta  # (2,)
    tau_z = flow_.D1_bnd @ mean_uz  # (2,)
    U_bulk_theta = integrate_scalar(mean_utheta, flow_.y_weights) / volfac
    U_bulk_z = integrate_scalar(mean_uz, flow_.y_weights) / volfac

    # ── Energy input rate I ─────────────────────────────────
    # Wall torque power: r * U_theta(wall) * tau'_theta, with the outer
    # wall contributing +, the inner wall -.
    Uw_in = flow_.base_flow[2, 0, 0, 0]
    Uw_out = flow_.base_flow[2, -1, 0, 0]
    r1, r2 = flow_.rs[0], flow_.rs[-1]
    delta_I = (r2 * Uw_out * tau_theta[1] - r1 * Uw_in * tau_theta[0]) / (
        Re * volfac
    )
    energy_input = flow_.I_lam + delta_I

    # ── Dissipation D ───────────────────────────────────────
    # nabla^2 U = 0 => cross-dissipation = 0; D = D_lam + Omega'/Re.
    pert_enstrophy = get_enstrophy_annular(
        state,
        flow_.D1,
        flow_.inv_r,
        fourier_.m,
        fourier_.kz2,
        fourier_.k_metric,
        flow_.y_weights,
    )
    dissipation = flow_.D_lam + pert_enstrophy / Re

    # ── Total energy E ──────────────────────────────────────
    # E = E_lam + <U . u'> + E'; only U_theta contributes.
    base_utheta = flow_.base_flow[2, :, 0, 0]  # (Nr,)
    cross = (
        integrate_scalar(base_utheta * mean_utheta, flow_.y_weights) / volfac
    )
    total_energy = flow_.E_lam + cross + perturbation_energy

    return {
        "E'": perturbation_energy,
        "I": energy_input,
        "D": dissipation,
        "E": total_energy,
        "tau'_th,i": tau_theta[0] / Re,
        "tau'_th,o": tau_theta[1] / Re,
        "tau'_z,i": tau_z[0] / Re,
        "tau'_z,o": tau_z[1] / Re,
        "Ub'_th": U_bulk_theta,
        "Ub'_z": U_bulk_z,
    }


def get_stats(state: Array) -> dict[str, Array]:
    """Wrapper around ``_get_stats_jit``."""
    return _get_stats_jit(state, fourier, flow)
