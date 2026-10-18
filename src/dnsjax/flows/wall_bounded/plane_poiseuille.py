r"""Plane Poiseuille (channel) flow: pressure-driven flow between plates.

``PlanePoiseuilleFlow`` adds the base flow to
:class:`~dnsjax.geometries.wall_bounded.cartesian.CartesianFlow`, which
holds the geometry (wall-normal grid, FD matrices, influence-matrix
operators and iteration, solvers).  The module exports the flow-module
surface of :class:`~dnsjax.flow_spec.FlowSpec`, with the optional
``get_driving`` and the transient-growth hook ``frozen_profile_flow``.

Base flow
---------
`$U_s(y) = 1 - y^2$` on `$y \in [-1, 1]$`, along the streamwise
direction `$(\cos\theta, 0, \sin\theta)$` at tilt angle `$\theta$`:

- `$dU_s/dy = -2y$`
- `$\nabla \times \mathbf{U} = (-2y\sin\theta, 0, 2y\cos\theta)$`
- `$\mathbf{U} \times \nabla \times \mathbf{U}
  = (0,\; -2y(1-y^2),\; 0)$` (tilt-independent)

Moving frame
------------
``phys.u_grid`` defaults to the laminar bulk velocity
`$U_{b,\mathrm{lam}} = 2/3$`, so a run evolves in the frame translating
in `$x$` at that speed (the frame term and the frame-relative CFL:
:func:`~dnsjax.geometries.wall_bounded._base.pad_base_flow`).
``--phys.u_grid 0`` selects the laboratory frame.

Driving
-------
With ``driving = "constant_pressure_gradient"`` (default), a fixed
mean pressure gradient maintains the base flow and the bulk velocity
is free (``Ub'_s`` in ``stats.dat``).  With
``driving = "constant_bulk_velocity"``, each IMM iteration adjusts the
mean-mode streamwise velocity to hold the perturbation bulk velocity
at zero, and the force it applies is the diagnostic (``-dPds'`` in
``stats.dat``).

Spanwise blocking
-----------------
With ``block_mean_spanwise_velocity = True``, each IMM iteration
additionally zeroes the perturbation bulk velocity in the spanwise
direction `$(-\sin\theta, 0, \cos\theta)$`, as sidewalls preventing
net spanwise momentum would, independently of ``driving``
(:func:`~dnsjax.geometries.wall_bounded.cartesian._apply_bulk_corrections`).
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
    mean_driving,
    pad_base_flow,
    tilted_profile_arrays,
)
from ...geometries.wall_bounded.cartesian import (
    frozen_profile_flow as _frozen_flow_copy,
)
from ...ic.localized_rolls import generate_cartesian_rolls
from ...ic.random_field import generate_cartesian
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
        r"""Add the plane-Poiseuille base flow to the geometry's setup.

        :meth:`CartesianFlow.__post_init__` builds the wall-normal grid,
        the FD matrices and the factorized per-mode operators; this
        method adds the base flow
        `$\mathbf{U} = (1-y^2)(\cos\theta, 0, \sin\theta)$` and its
        curl.
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
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_cartesian_stepper(flow)

# No basis pair (FlowSpec): the state is physical `(u, v, w)` under
# either ``res.consistent_imm`` formulation, the default's evolved
# scalars never leaving ``cartesian._imm_iteration_vw``.


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
    dy_us = flow.D1.dense @ us
    base, curl = tilted_profile_arrays(us, dy_us)
    return _frozen_flow_copy(flow, base, curl)


# ── Diagnostic statistics ────────────────────────────────────────────────


def _perturbation_energy(
    state: Array, fourier_: Fourier, flow_: PlanePoiseuilleFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`.

    The single definition, shared by :func:`get_stats` (which reports
    it as ``E'``) and the laminarization read
    :func:`get_perturbation_energy`.
    """
    return get_norm2(state, fourier_.k_metric, flow_.y_weights) / 2


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: PlanePoiseuilleFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics.

    - `$E'$`: perturbation kinetic energy
      `$\|\mathbf{u}'\|^2 / 2$`.
    - `$I$`: energy input rate
      `$\langle (\mathbf{u}' + \mathbf{U}) \cdot
      (-\Pi_\mathrm{tot}\,\hat{\mathbf{s}}) \rangle
      = -\Pi_\mathrm{tot}\,U_{b,\mathrm{tot}}$`, the driving force
      `$-\Pi$` being the negative of the mean pressure gradient
      (:mod:`dnsjax.ic.mean_mode`).
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

    Under ``constant_bulk_velocity`` the driving work is the whole of
    `$I - I_\mathrm{lam}$`.  The **exact** input rate is
    `$-U_{b,\mathrm{lam}}\Pi$`, with `$-\Pi$` the force the corrector
    applied (``stats.dat``'s ``-dPds'``); this state-only function
    cannot see that force, so it estimates `$\Pi$` from the wall shear,
    low by `$\mathrm{bulk}(\bar N_s)$`: zero continuously, a wall-normal
    truncation residual discretely, and convergent.

    Substituting the exact value makes `$dE/dt = I - D$` close **worse**
    (measured: ``tests/test_energy_budget.py``'s
    ``_check_applied_vs_inferred``), because the two-term form omits the
    discrete `$\langle u\cdot N(u)\rangle$`, zero continuously, whose
    dominant part is `$-U_b\,\mathrm{bulk}(\bar N_s)$`; the estimate's
    error is that quantity with the opposite sign, so it absorbs the
    omission.  `$I$` is therefore the budget-consistent input rate, not
    the applied power; the two converge.

    All total-field quantities are computed algebraically from
    perturbation norms and laminar constants, without constructing
    `$\mathbf{u}' + \mathbf{U}$`.  For `$-\nabla^2 U = 2$` (constant),
    the cross-enstrophy reduces to
    `$\langle \boldsymbol{\omega}_U \cdot \boldsymbol{\omega}_{u'}
    \rangle = 2\,U'_{b,s}$`, with `$U'_{b,s}$` the perturbation bulk
    streamwise velocity.
    """
    Re = params.phys.re
    perturbation_energy = _perturbation_energy(state, fourier_, flow_)

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
    # CBV: I = I_lam - U_bulk_lam * Pi, with Pi the perturbation
    # mean pressure gradient dP/ds (the force being -Pi).
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
    """The ``stats.dat`` columns of *state* (``_get_stats_jit``)."""
    return _get_stats_jit(state, fourier, flow)


@jit
def _get_perturbation_energy_jit(
    state: Array, fourier_: Fourier, flow_: PlanePoiseuilleFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`."""
    return _perturbation_energy(state, fourier_, flow_)


def get_perturbation_energy(state: Array) -> Array:
    """Perturbation kinetic energy E' (for the laminarization check)."""
    return _get_perturbation_energy_jit(state, fourier, flow)


@jit
def _get_driving_jit(
    state: Array, flow_: PlanePoiseuilleFlow
) -> dict[str, Array]:
    r"""Wall-shear inference of the applied mean-mode driving."""
    return mean_driving(state, flow_)


def get_driving(state: Array) -> dict[str, Array]:
    """Applied mean-mode driving (:class:`~dnsjax.flow_spec.FlowSpec`)."""
    return _get_driving_jit(state, flow)


# ── Initial conditions ───────────────────────────────────────────


def init_random_state(
    amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
    mean_flow: bool = False,
) -> Array:
    """The random IC: :func:`dnsjax.ic.random_field.generate_cartesian`.

    The ``init_random_state`` hook of the flow-module surface
    (:class:`~dnsjax.flow_spec.FlowSpec`): the Cartesian draw.
    """
    return generate_cartesian(
        amplitude,
        smoothness,
        wall_smoothness,
        wall_confinement,
        seed,
        mean_flow,
    )


def init_localized_rolls(
    amplitude: float, width: float, wavelength: float
) -> Array:
    """The localized rolls of the flow-module surface's IC hooks.

    :func:`dnsjax.ic.localized_rolls.generate_cartesian_rolls`, the
    ``init_localized_rolls`` hook.
    """
    return generate_cartesian_rolls(amplitude, width, wavelength)
