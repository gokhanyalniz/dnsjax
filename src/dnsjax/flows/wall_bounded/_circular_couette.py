r"""Shared circular-Couette flow machinery (Taylor-Couette family).

Taylor-Couette and quasi-Keplerian are the *same* flow -- the
circular-Couette base profile `$U_\theta = A_0 r + B_0/r$` on the
annular geometry -- differing only in their control-parameter
interfaces (`$(\mathrm{Re}_1, \mathrm{Re}_2, \eta)$` vs
`$(\mathrm{Re}_i, R_\Omega, \eta)$`; the specs in
``flows/wall_bounded/specs/`` own that math and land in the same
``derived_params.ccf_A``/``ccf_B``/radii).  Everything the two flow
modules share therefore lives here once: the flow dataclass (base
flow, curl, laminar constants), the transient-growth
``frozen_profile_flow`` builder, and the jitted diagnostics.

The per-system modules (:mod:`.taylor_couette`,
:mod:`.quasi_keplerian`) remain the user-facing surface -- their
docstrings carry the per-flow conventions -- and each instantiates
its own ``flow`` singleton here-defined class plus the thin
singleton-binding wrappers ``__main__`` and the analysis drivers
consume.
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ...geometries.wall_bounded.annular import (
    AnnularFlow,
    Fourier,
    extract_mean_mode,
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
class CircularCouetteFlow(AnnularFlow):
    r"""Precomputed data for a circular-Couette (Taylor-Couette-family)
    flow.

    Laminar constants for `$U_\theta = A_0 r + B_0/r$` on
    `$[r_1, r_2]$` are computed in ``__post_init__`` from the
    circular-Couette coefficients and radii on ``derived_params``
    (set by the selected system's spec: Taylor-Couette from
    `$(\mathrm{Re}_1, \mathrm{Re}_2, \eta)$`, quasi-Keplerian from
    `$(\mathrm{Re}_i, R_\Omega, \eta)$`).
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

        # Base flow: U = (0, 0, U_theta) in (u_z, u_r, u_theta) --
        # the azimuthal (streamwise) velocity is component 2 (the
        # annular component-order exception; see annular.py).
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


def frozen_profile_flow(
    flow: CircularCouetteFlow, u_theta: Array
) -> CircularCouetteFlow:
    r"""Flow copy linearized around an arbitrary azimuthal profile.

    Transient-growth hook (:mod:`dnsjax.analysis.transient_growth`):
    given the *total* azimuthal profile `$U_\theta(r)$` on the code
    grid (``flow.rs``, shape ``(Nr,)``) in the `$(u_z, u_r, u_\theta)$`
    basis, build `$\mathbf{U} = (0, 0, U_\theta)$` and
    `$\nabla\times\mathbf{U} = (\omega_z, 0, 0)$` with the axial
    vorticity `$\omega_z = (1/r)\,\partial_r(r U_\theta) = dU_\theta/dr
    + U_\theta/r$` (reduces to the laminar uniform `$2 A_0$` for
    `$U_\theta = A_0 r + B_0/r$`).  The derivative uses the flow's FD
    `$D_1$` (annular has two walls, no `$r = 0$` axis, so no parity
    reduction).  Returns a copy of *flow* carrying that base flow (all
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


def _perturbation_energy(
    state: Array, fourier_: Fourier, flow_: CircularCouetteFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`.

    The single definition, shared by :func:`get_stats` (which reports
    it as ``E'``) and the laminarization read
    ``get_perturbation_energy``.
    """
    return get_norm2_annular(state, fourier_.k_metric, flow_.y_weights) / 2


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: CircularCouetteFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics.

    *state* is the **physical** `$(u_z, u_r, u_\theta)$` view of the
    field -- diagnostics sit outside the solver, whose working basis
    is the decoupled `$(u_z, u_+, u_-)$` one (the ``annular.py``
    module docstring).

    - `$E'$`: perturbation kinetic energy (annular norm with radial
      Jacobian `$r$`).
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
    perturbation_energy = _perturbation_energy(state, fourier_, flow_)

    # ── Mean velocity profiles ───────────────────────────────
    mean_state = extract_mean_mode(state)  # (3, Nr)
    mean_uz = mean_state[0].real  # (Nr,)
    mean_utheta = mean_state[2].real  # (Nr,)

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


@jit
def _get_perturbation_energy_jit(
    state: Array, fourier_: Fourier, flow_: CircularCouetteFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`."""
    return _perturbation_energy(state, fourier_, flow_)
