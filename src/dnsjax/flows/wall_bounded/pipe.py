r"""Pipe flow: pressure-driven flow through a circular pipe.

This module defines the ``PipeFlow`` dataclass that holds the
pipe-flow-specific base flow.  Geometry-general infrastructure
(radial CGL grid -- half-CGL under the default ``iterative-cn``
scheme, rigged-CGL under ``cnab2``, selected by
``geo.grid_type``, parity-reduced FD
matrices, IMM operators, cylindrical IMM iteration, predict /
correct / norm, Pallas / dense LU solvers) is inherited from
``geometries.wall_bounded.cylindrical.CylindricalFlow``.

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``predict_and_fully_correct_measured`` -- fused step + CFL
  measurements (``steps.dat``)
- ``init_state`` -- the ``start_from_laminar`` initial state
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
------------
``phys.u_grid`` defaults to the laminar bulk velocity
`$U_{b,\mathrm{lam}} = 1/2$` for pipe flow, so by default the run
evolves in the frame translating axially at `$U_{grid} = 1/2$`: the
convective frame term `$+ i k_z U_{grid} \mathbf{u}'$` is added in
the cylindrical ``_get_rhs_core`` / ``_l_bf`` and the CFL diagnostic
advects with `$\mathbf{U} - U_{grid}\hat{\mathbf{z}}$` (see
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
    from_solver_basis,  # noqa: F401 — re-exported (basis boundary)
    get_norm2_cyl,
    get_pert_enstrophy_cyl,
    integrate_scalar,
    pad_base_flow,
    to_solver_basis,  # noqa: F401 — re-exported (basis boundary)
)
from ...geometries.wall_bounded.cylindrical import (
    frozen_profile_flow as _frozen_flow_copy,
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

        Delegates the radial CGL grid, parity-reduced FD
        matrices, and per-mode IMM operator setup to
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
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_cylindrical_stepper(flow)


def frozen_profile_flow(uz: Array) -> PipeFlow:
    r"""Flow linearized around an arbitrary axial profile.

    Transient-growth hook (:mod:`dnsjax.analysis.transient_growth`):
    given the *total* axial profile `$U_z(r)$` on the code grid
    (``flow.rs``, shape ``(Nr,)``) in the `$(u_z, u_r, u_\theta)$`
    basis, build `$\mathbf{U} = (U_z, 0, 0)$` and
    `$\nabla\times\mathbf{U} = (0, 0, -dU_z/dr)$`.  The radial
    derivative uses the even-parity FD `$D_1$` (mean mode `$m = 0$`,
    parity `$(-1)^m = +1$`: the common `$D_{1,\mathrm{pos}}$` plus the
    near-axis ghost correction, exactly as ``_curl_fn`` differentiates
    `$u_z$`), so no `$r = 0$` grid point is needed.  Returns a flow copy
    carrying that base flow (all operators shared; see
    :func:`~dnsjax.geometries.wall_bounded._base.frozen_profile_flow`).
    """
    g = flow.D1_ghost.shape[0]
    duz = (flow.D1_pos @ uz).at[:g].add(flow.D1_ghost @ uz)
    base = (
        jnp.zeros(
            (3, params.res.ny),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )
        .at[0]
        .set(uz)[:, :, None, None]
    )
    curl = (
        jnp.zeros(
            (3, params.res.ny),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )
        .at[2]
        .set(-duz)[:, :, None, None]
    )
    return _frozen_flow_copy(flow, base, curl)


# ── Diagnostic statistics ────────────────────────────────────────


def _perturbation_energy(
    state: Array, fourier_: Fourier, flow_: PipeFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`.

    The single definition, shared by :func:`get_stats` (which reports
    it as ``E'``) and the laminarization read
    :func:`get_perturbation_energy`.
    """
    return get_norm2_cyl(state, fourier_.k_metric, flow_.y_weights) / 2


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: PipeFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics.

    *state* is the **physical** `$(u_z, u_r, u_\theta)$` view of the
    field -- diagnostics sit outside the solver, whose working basis
    is the decoupled `$(u_z, u_+, u_-)$` one (the ``cylindrical.py``
    module docstring).

    - `$E'$`: perturbation kinetic energy (cylindrical norm
      with radial Jacobian `$r$`).
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
    perturbation_energy = _perturbation_energy(state, fourier_, flow_)

    # ── Mean velocity profiles ───────────────────────────────
    mean_state = extract_mean_mode(state)  # (3, Nr)
    mean_uz = mean_state[0].real  # (Nr,)
    mean_utheta = mean_state[2].real  # (Nr,)

    # ── Wall shear & bulk velocity ──────────────────────────
    D1_wall_row = flow_.D1_wall.ravel()
    tau_z = jnp.dot(D1_wall_row, mean_uz)
    tau_theta = jnp.dot(D1_wall_row, mean_utheta)
    # mean u_z is even in r (m=0, parity (-1)^m), mean u_theta is odd
    # (parity (-1)^{m+1}): integrate each with its parity's weights.
    U_bulk_z = (
        integrate_scalar(mean_uz, flow_.y_weights) / derived_params.volume_fac
    )
    U_bulk_theta = (
        integrate_scalar(mean_utheta, flow_.y_weights_odd)
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
    """Wrapper around ``_get_stats_jit`` (physical-basis *state*)."""
    return _get_stats_jit(state, fourier, flow)


@jit
def _get_perturbation_energy_jit(
    state: Array, fourier_: Fourier, flow_: PipeFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`."""
    return _perturbation_energy(state, fourier_, flow_)


def get_perturbation_energy(state: Array) -> Array:
    r"""Perturbation kinetic energy E' (for the laminarization check).

    Takes the **physical** `$(u_z, u_r, u_\theta)$` view, like
    :func:`get_stats`, which reports the same number as ``E'``.
    """
    return _get_perturbation_energy_jit(state, fourier, flow)
