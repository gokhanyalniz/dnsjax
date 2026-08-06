"""Plane Couette flow: wall-bounded shear between two moving plates.

This module defines the ``PlaneCouetteFlow`` dataclass that holds the
plane-Couette-specific base flow.  Geometry-general infrastructure
(CGL grid, FD matrices, IMM operators, Kleiser-Schumann IMM
iteration, predict / correct / norm, Pallas / dense LU solvers) is
inherited from ``geometries.wall_bounded.cartesian.CartesianFlow``.

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``predict_and_fully_correct_measured`` -- fused step + CFL
  measurements (``steps.dat``)
- ``set_dt`` / ``reset_ab2_kappa`` -- adaptive-dt hooks
  (``step.adaptive``; on-device operator rebuild, no recompile)
- ``init_state`` -- the ``start_from_laminar`` initial state
- ``get_stats`` -- diagnostic statistics

The influence-matrix method enforces the no-slip wall BCs and the
*wall-row* divergence exactly at every time step; the interior
discrete divergence is a truncation-level residual unless
``res.consistent_imm`` selects the `$v$`-`$\\omega_y$` formulation,
where it vanishes algebraically.  No post-step projection is fused
into the stepper (the triply-periodic geometry fuses one via
``make_stepper``'s *finalize_fn*; a wall-bounded state-side
projection is unstable -- see the ``cartesian._imm_iteration``
docs).

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
from ...sharding import register_dataclass_pytree, sharding


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
        self.base_flow, self.curl_base_flow = tilted_profile_arrays(Us, dy_Us)
        pad_base_flow(self)


flow: PlaneCouetteFlow = PlaneCouetteFlow()

(
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_cartesian_stepper(flow)

# No ``to_solver_basis``/``from_solver_basis`` here, in either flag
# state: the Cartesian state is physical `(u, v, w)` throughout, and
# ``res.consistent_imm``'s evolved scalars never leave
# ``cartesian._imm_iteration_vw``.  Every consumer of the pair looks
# it up with ``getattr`` and falls back to the identity.


def frozen_profile_flow(us: Array) -> PlaneCouetteFlow:
    r"""Flow linearized around an arbitrary streamwise profile.

    Transient-growth hook (:mod:`dnsjax.analysis.transient_growth`):
    given the *total* streamwise profile `$U_s(y)$` on the code grid
    (``flow.ys``, shape ``(Ny,)``), tilt-split it exactly as the
    laminar `$U_s = y$` (:func:`tilted_profile_arrays`), differentiate
    with the flow's FD `$D_1$` for `$\nabla\times\mathbf{U}$`, and
    return a flow copy carrying that base flow (all operators shared;
    see :func:`~dnsjax.geometries.wall_bounded._base.frozen_profile_flow`).
    """
    dy_us = flow.D1 @ us
    base, curl = tilted_profile_arrays(us, dy_us)
    return _frozen_flow_copy(flow, base, curl)


# ── Diagnostic statistics ────────────────────────────────────────────────


def _perturbation_energy(
    state: Array, fourier_: Fourier, flow_: PlaneCouetteFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`.

    The single definition, shared by :func:`get_stats` (which reports
    it as ``E'``) and the laminarization read
    :func:`get_perturbation_energy`.
    """
    return get_norm2(state, fourier_.k_metric, flow_.y_weights) / 2


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


@jit
def _get_perturbation_energy_jit(
    state: Array, fourier_: Fourier, flow_: PlaneCouetteFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`."""
    return _perturbation_energy(state, fourier_, flow_)


def get_perturbation_energy(state: Array) -> Array:
    """Perturbation kinetic energy E' (for the laminarization check)."""
    return _get_perturbation_energy_jit(state, fourier, flow)
