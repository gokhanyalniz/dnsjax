r"""Quasi-Keplerian Taylor-Couette flow in a rotating annulus.

A Taylor-Couette flow (concentric rotating cylinders, axially
periodic) restricted to the **quasi-Keplerian regime**: co-rotating
cylinders whose laminar circular-Couette profile has angular momentum
increasing outward and angular velocity decreasing outward, so it is
linearly stable by Rayleigh's criterion.  It is the
accretion-disk-relevant regime studied as a hydrodynamic model for the
origin of turbulence in Keplerian disks.

The physics, operators, and base flow are **identical** to
:mod:`.taylor_couette` (both are the circular-Couette base flow
`$U_\theta = A_0 r + B_0/r$` on the annular geometry) and both bind
the shared implementation in :mod:`._circular_couette`; this module
differs only in the *control-parameter interface* and its documented
conventions.  Geometry-general infrastructure (radial grid, FD
matrices, IMM operators, the `$2 \times 2$` annular IMM iteration,
predict / correct / norm, Pallas / dense LU solvers) is inherited from
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
`$\mathrm{Re}_s$` the shear.  The ``quasi_keplerian`` spec
(``flows/wall_bounded/specs/``) requires `$\mathrm{Re}_i > 0$` and
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
Identical to Taylor-Couette (see that module's docstring and
:mod:`._circular_couette`): the base flow is
`$\mathbf{U} = (0, 0, U_\theta)$` in `$(u_z, u_r, u_\theta)$` with
uniform axial vorticity `$\omega_z = 2 A_0$` and
`$\nabla^2\mathbf{U} = 0$`, so the energy budget uses
`$I_{\mathrm{lam}} = D_{\mathrm{lam}} = 4 B_0^2/(\mathrm{Re}\,r_1^2
r_2^2)$` and `$E_{\mathrm{lam}} = \tfrac12\langle U_\theta^2\rangle$`.

It exports the flow interface consumed by ``__main__``
(``predict_and_fully_correct`` / ``_measured``, ``step_cnab2`` /
``_measured``, ``init_state``, ``get_stats``,
``get_perturbation_energy``) and the ``frozen_profile_flow`` hook for
``dnsjax.analysis.transient_growth``.
"""

from jax import Array

from ...geometries.wall_bounded.annular import build_annular_stepper, fourier
from ._circular_couette import (
    CircularCouetteFlow,
    _get_perturbation_energy_jit,
    _get_stats_jit,
)
from ._circular_couette import (
    frozen_profile_flow as _frozen_cc_flow,
)

flow: CircularCouetteFlow = CircularCouetteFlow()

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


def frozen_profile_flow(u_theta: Array) -> CircularCouetteFlow:
    r"""Flow linearized around an arbitrary azimuthal profile
    `$U_\theta(r)$` (transient-growth hook; see
    :func:`~dnsjax.flows.wall_bounded._circular_couette.frozen_profile_flow`).
    """
    return _frozen_cc_flow(flow, u_theta)


def get_stats(state: Array) -> dict[str, Array]:
    """Wrapper around the shared circular-Couette ``_get_stats_jit``."""
    return _get_stats_jit(state, fourier, flow)


def get_perturbation_energy(state: Array) -> Array:
    """Perturbation kinetic energy E' (for the laminarization check)."""
    return _get_perturbation_energy_jit(state, fourier, flow)
