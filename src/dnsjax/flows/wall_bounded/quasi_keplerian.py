r"""Quasi-Keplerian Taylor-Couette flow in a rotating annulus.

This module defines the ``QuasiKeplerianFlow`` dataclass, a
Taylor-Couette flow (concentric rotating cylinders, axially periodic)
restricted to the **quasi-Keplerian regime**: co-rotating cylinders
whose laminar circular-Couette profile has angular momentum increasing
outward and angular velocity decreasing outward, so it is linearly
stable by Rayleigh's criterion.  It is the accretion-disk-relevant
regime studied as a hydrodynamic model for the origin of turbulence in
Keplerian disks.

The physics, operators, and base flow are **identical** to
``taylor_couette.TaylorCouetteFlow`` (both are the circular-Couette base
flow `$U_\theta = A_0 r + B_0/r$` on the annular geometry); this module
differs only in the *control-parameter interface* and its documented
conventions.  Geometry-general infrastructure (radial grid, FD matrices,
IMM operators, the `$2 \times 2$` annular IMM iteration, predict /
correct / norm, Pallas / dense LU solvers) is inherited from
``geometries.wall_bounded.annular.AnnularFlow``.

Control parameters and the quasi-Keplerian regime
-------------------------------------------------
Instead of Taylor-Couette's `$(\mathrm{Re}_1, \mathrm{Re}_2)$` the flow
is specified by the **inner Reynolds number** `$\mathrm{Re}_i$`
(``phys.re1``), the **rotation number** `$R_\Omega$` (``phys.r_omega``),
and the radius ratio `$\eta = r_1/r_2$` (``geo.eta``).  Following the
shear/rotation parameterization of Dubrulle et al. (Phys. Fluids 2005),
with `$\mathrm{Re}_{i(o)} = \Omega_{i(o)} r_{i(o)} d / \nu$` on the gap
`$d = r_2 - r_1$`,

.. math::
    R_\Omega = \frac{(1 - \eta)(\mathrm{Re}_i + \mathrm{Re}_o)}
                    {\eta\,\mathrm{Re}_o - \mathrm{Re}_i}, \qquad
    \mathrm{Re}_s = \frac{2}{1 + \eta}
                    \bigl|\eta\,\mathrm{Re}_o - \mathrm{Re}_i\bigr|,

where `$R_\Omega$` (constant along half-lines through the origin of
`$(\mathrm{Re}_o, \mathrm{Re}_i)$` space) measures the mean rotation and
`$\mathrm{Re}_s$` the shear.  The **annular branch of**
``parameters.update_parameters`` requires `$\mathrm{Re}_i > 0$` and
`$R_\Omega < -1$`, and inverts `$R_\Omega$` for the outer Reynolds
number

.. math::
    \mathrm{Re}_o = \mathrm{Re}_i\,
        \frac{1 - \eta + R_\Omega}{\eta R_\Omega - (1 - \eta)},

storing it as ``phys.re2`` so the shared circular-Couette derivation
(``ccf_A``, ``ccf_B``, `$\mathrm{Re} = \mathrm{Re}_i$`) runs unchanged.
The rotation ratio is `$\mu = \Omega_o/\Omega_i = \eta\,\mathrm{Re}_o/
\mathrm{Re}_i$`.

The **quasi-Keplerian regime** is the open half-line
`$-\infty < R_\Omega < -1$`, bounded by the Rayleigh line
`$R_\Omega = -1$` (`$\mathrm{Re}_o = \eta\,\mathrm{Re}_i$`, marginal
centrifugal stability) and the solid-body limit `$R_\Omega \to -\infty$`
(`$\mathrm{Re}_o = \mathrm{Re}_i/\eta$`, `$\mu = 1$`, no shear).  The
local rotation exponent

.. math::
    q(r) = -\frac{\mathrm{d}\ln\Omega}{\mathrm{d}\ln r}
         = \frac{2 B_0}{A_0 r^2 + B_0}

(with `$\Omega(r) = U_\theta/r = A_0 + B_0/r^2$`) is scale-invariant in
`$(A_0, B_0)$` and lies in `$(0, 2)$` across the gap on this half-line:
`$q = 2$` on the Rayleigh line, `$q = 0$` at solid body, and a Keplerian
disk has the constant `$q = 3/2$`.  ``validate_parameters`` prints the
derived `$\mathrm{Re}_o$`, `$\mathrm{Re}_s$`, `$\mu$`, and the
`$[q(r_2), q(r_1)]$` range at startup.

There is **no rotating frame / Coriolis force**: the base flow is the
lab-frame circular-Couette profile and the solver evolves the
perturbation `$\mathbf{u}'$` around it, exactly as in Taylor-Couette.

Nondimensionalization
---------------------
Gap length unit `$d = r_2 - r_1 = 1$` (`$r_1 = \eta/(1-\eta)$`,
`$r_2 = 1/(1-\eta)$`); velocity unit the inner-cylinder surface speed,
so `$U_\theta(r_1) = 1$` and `$U_\theta(r_2) = \mathrm{Re}_o/
\mathrm{Re}_i$`; the viscous term is `$\mathrm{Re}_i^{-1}\nabla^2$`.  The
code time unit is the **advective unit** `$\tau_d = d/(r_1\Omega_i)$`
(so ``step.dt`` is `$\mathrm{d}t/\tau_d$`).  A common alternative
"viscous-unit" convention (cylinder wall speeds equal `$\mathrm{Re}_i$`)
maps to the code fields by multiplying every velocity by
`$\mathrm{Re}_i$` while keeping the same `$t/\tau_d$`.

Azimuthal wedge
--------------
The reduced azimuthal domains
(`$\theta \in [0, 2\pi/m_0)$` with fundamental azimuthal wavenumber
`$m_0$`) are available through ``geo.m0`` (see the ``annular.Fourier``
docstring): `$m_0 > 1$` restricts the simulation to the
`$m_0$`-periodic subspace at `$1/m_0$` the azimuthal cost.

Base flow and laminar constants
-------------------------------
Identical to Taylor-Couette (see that module's docstring): the base
flow is `$\mathbf{U} = (0, 0, U_\theta)$` in `$(u_z, u_r, u_\theta)$`
with uniform axial vorticity `$\omega_z = 2 A_0$` and
`$\nabla^2\mathbf{U} = 0$`, so the energy budget uses
`$I_{\mathrm{lam}} = D_{\mathrm{lam}} = 4 B_0^2/(\mathrm{Re}\,r_1^2
r_2^2)$` and `$E_{\mathrm{lam}} = \tfrac12\langle U_\theta^2\rangle$`.

It exports the flow interface consumed by ``__main__``
(``predict_and_fully_correct`` / ``_measured``, ``step_cnab2`` /
``_measured``, ``init_state``, ``get_stats``,
``get_perturbation_energy``) and the ``frozen_profile_flow`` hook for
``dnsjax.analysis.transient_growth``.
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
from ...geometries.wall_bounded.annular import (
    frozen_profile_flow as _frozen_flow_copy,
)
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class QuasiKeplerianFlow(AnnularFlow):
    r"""Precomputed data for quasi-Keplerian Taylor-Couette flow.

    The circular-Couette base flow `$U_\theta = A_0 r + B_0/r$` on
    `$[r_1, r_2]$` and its laminar constants are built in
    ``__post_init__`` from the coefficients ``derived_params.ccf_A`` /
    ``ccf_B`` (derived from `$(\mathrm{Re}_i, R_\Omega, \eta)$` in the
    annular branch of ``update_parameters``), identically to
    ``taylor_couette.TaylorCouetteFlow``.
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


flow: QuasiKeplerianFlow = QuasiKeplerianFlow()

(
    predict_and_correct,
    iterate_correction,
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_annular_stepper(flow)


def frozen_profile_flow(u_theta: Array) -> QuasiKeplerianFlow:
    r"""Flow linearized around an arbitrary azimuthal profile.

    Transient-growth hook (:mod:`dnsjax.analysis.transient_growth`):
    given the *total* azimuthal profile `$U_\theta(r)$` on the code
    grid (``flow.rs``, shape ``(Nr,)``) in the `$(u_z, u_r, u_\theta)$`
    basis, build `$\mathbf{U} = (0, 0, U_\theta)$` and
    `$\nabla\times\mathbf{U} = (\omega_z, 0, 0)$` with the axial
    vorticity `$\omega_z = (1/r)\,\partial_r(r U_\theta) = dU_\theta/dr
    + U_\theta/r$` (reduces to the laminar uniform `$2 A_0$` for
    `$U_\theta = A_0 r + B_0/r$`).  The derivative uses the flow's FD
    `$D_1$` (annular has two walls, no `$r = 0$` axis, so no parity
    reduction).  Returns a flow copy carrying that base flow (all
    operators shared; see
    :func:`~dnsjax.geometries.wall_bounded._base.frozen_profile_flow`).
    """
    du = flow.D1 @ u_theta
    omega_z = du + flow.inv_r * u_theta
    base = (
        jnp.zeros(
            (3, params.res.ny),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )
        .at[2]
        .set(u_theta)[:, :, None, None]
    )
    curl = (
        jnp.zeros(
            (3, params.res.ny),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )
        .at[0]
        .set(omega_z)[:, :, None, None]
    )
    return _frozen_flow_copy(flow, base, curl)


# ── Diagnostic statistics ────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: QuasiKeplerianFlow
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


@jit
def _get_perturbation_energy_jit(
    state: Array, fourier_: Fourier, flow_: QuasiKeplerianFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`."""
    return get_norm2_annular(state, fourier_.k_metric, flow_.y_weights) / 2


def get_perturbation_energy(state: Array) -> Array:
    """Perturbation kinetic energy E' (for the laminarization check)."""
    return _get_perturbation_energy_jit(state, fourier, flow)
