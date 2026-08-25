r"""Taylor-Couette flow: shear-driven flow in a rotating annulus.

This module binds the shared circular-Couette machinery
(:mod:`._circular_couette`: the ``CircularCouetteFlow`` dataclass,
diagnostics, and the transient-growth hook) to the Taylor-Couette
control parameters `$(\mathrm{Re}_1, \mathrm{Re}_2, \eta)$` (whose
circular-Couette coefficients the ``taylor_couette`` spec derives).
Geometry-general infrastructure (radial grid on `$[r_1, r_2]$`, FD
matrices, IMM operators, the `$2 \times 2$` annular IMM iteration,
predict / correct / norm, Pallas / dense LU solvers) is inherited from
``geometries.wall_bounded.annular.AnnularFlow``.

It exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``predict_and_fully_correct_measured`` -- fused step + CFL
  measurements (``steps.dat``)
- ``init_state`` -- the ``start_from_laminar`` initial state
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
``taylor_couette`` spec in ``flows/wall_bounded/specs/``).  Its derived
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

from jax import Array

from ...geometries.wall_bounded.annular import (
    build_annular_stepper,
    fourier,
    from_solver_basis,  # noqa: F401 — re-exported (basis boundary)
    to_solver_basis,  # noqa: F401 — re-exported (basis boundary)
)
from ._circular_couette import (
    CircularCouetteFlow,
    _get_driving_jit,
    _get_perturbation_energy_jit,
    _get_stats_jit,
)
from ._circular_couette import (
    frozen_profile_flow as _frozen_cc_flow,
)

flow: CircularCouetteFlow = CircularCouetteFlow()

(
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_annular_stepper(flow)


def frozen_profile_flow(u_theta: Array) -> CircularCouetteFlow:
    r"""Flow linearized around an arbitrary azimuthal profile
    `$U_\theta(r)$` (transient-growth hook; see
    :func:`~dnsjax.flows.wall_bounded._circular_couette.frozen_profile_flow`).
    """
    return _frozen_cc_flow(flow, u_theta)


def get_stats(state: Array) -> dict[str, Array]:
    """Shared circular-Couette ``_get_stats_jit`` (physical *state*)."""
    return _get_stats_jit(state, fourier, flow)


def get_driving(state: Array) -> dict[str, Array]:
    r"""Applied mean-mode driving inferred from *state* alone.

    The optional flow-module export ``__main__`` uses for the one
    ``stats.dat`` row with no step behind it (``t = t0``); every other
    row carries the value the corrector actually applied.  Empty unless
    ``phys.block_mean_spanwise_velocity`` is on.
    """
    return _get_driving_jit(state, flow)


def get_perturbation_energy(state: Array) -> Array:
    r"""Perturbation kinetic energy E' (for the laminarization check).

    Takes the **physical** `$(u_z, u_r, u_\theta)$` view, like
    :func:`get_stats`, which reports the same number as ``E'``.
    """
    return _get_perturbation_energy_jit(state, fourier, flow)
