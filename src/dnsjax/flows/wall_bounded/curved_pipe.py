r"""Curved pipe: pressure- or flux-driven flow through a toroidal pipe.

A circular pipe of radius `$a$` bent onto a circle of radius `$R_c$`:
constant curvature `$\kappa = a/R_c$` (``geo.curvature``), zero torsion.
The geometry -- coordinates, the carried variable `$w = (h u_s, u_r,
u_\theta)$`, the four identities that keep the influence-matrix pass
the straight pipe's, and the driving -- is derived once in
:mod:`~dnsjax.geometries.wall_bounded.cylindrical_curved`; this module
adds only the flow-level surface.

Total-field formulation
-----------------------
Like ``dean`` and the two viscoelastic flows, and unlike the straight
pipe, this flow time-integrates the **total** velocity: it sets
``base_flow = curl_base_flow = 0`` (in
:class:`~dnsjax.geometries.wall_bounded.cylindrical_curved.CurvedCylindricalFlow`)
so the rotational term evaluates the full `$\mathbf{u} \times
\boldsymbol{\omega}$`, and the mean pressure gradient enters as a
uniform body force.  There is no base flow to expand around because
there is no closed-form one: the curved pipe's laminar state is the
**two-dimensional** Dean-vortex solution, which is numerical.

Consequences for the diagnostics:

- ``init.start_from_laminar`` gives the *straight* pipe's
  `$u_s = 1 - r^2$`, which at `$\kappa > 0$` is **not** a steady state.
  It is a starting point; the 2D Dean state is what the flow relaxes to
  on a viscous timescale, and is reached by running the code.
- `$E'$` is therefore **not** a deviation from a laminar profile but the
  kinetic energy of the `$k_s \neq 0$` (streamwise-varying) modes.  It
  vanishes identically on any 2D or steady state, needs no reference
  profile, and is exactly the transition indicator here, so
  ``stop.check_laminarization`` reads "relaminarised to 2D".

Exports the flow interface consumed by ``__main__``:
``predict_and_fully_correct`` (+ the measured variant), ``init_state``,
``get_stats``, ``get_perturbation_energy``, ``get_driving`` and the
basis pair ``to_solver_basis`` / ``from_solver_basis`` -- which here
carry the metric weight `$h$` as well as the `$u_\pm$` rotation.
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ...geometries.wall_bounded._base import (
    extract_mean_mode,
    from_pm_basis,
    get_inprod,
    integrate_scalar,  # noqa: F401 — re-exported
    to_pm_basis,
)
from ...geometries.wall_bounded._cylindrical_stepping import (
    CARRIED_FIELDS,  # noqa: F401 — re-exported (snapshot carry/ member)
    _curl_fn,
    _get_rhs,
    with_carried,
)
from ...geometries.wall_bounded.cylindrical import (
    Fourier,
    fourier,
)
from ...geometries.wall_bounded.cylindrical_curved import (
    BULK_TARGET,
    FORCE_S,
    KAPPA,
    CurvedCylindricalFlow,
    build_curved_cylindrical_stepper,
)
from ...operators import phys_to_spec_2d, spec_to_phys_2d
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class CurvedPipeFlow(CurvedCylindricalFlow):
    r"""Precomputed data for the curved pipe.

    Everything is inherited: the straight pipe's grid, FD matrices and
    per-mode operators from ``CylindricalFlow``, the metric and the
    adapter members from ``CurvedCylindricalFlow``.
    """


flow: CurvedPipeFlow = CurvedPipeFlow()

(
    _init_state_zero,  # overridden below (the laminar start is not 0)
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_curved_cylindrical_stepper(flow)


# ── Basis boundary (u_pm rotation **and** the metric weight) ──────


def _carried_velocity(state: Array, flow_: CurvedPipeFlow) -> Array:
    r"""Physical `$(u_s, u_r, u_\theta)$` -> the carried state.

    Applies the metric weight `$w_s = h\,u_s$` before the `$u_\pm$`
    rotation.  Pseudo-spectral (one transform round trip on one field),
    so it is the exact inverse of :func:`_physical_velocity` up to the
    collocation error the dealiasing rule already bounds -- an `$m\pm1$`
    Galerkin product would instead lose the outgoing top mode.
    """
    weighted = phys_to_spec_2d(spec_to_phys_2d(state[:1]) * flow_.h_phys)
    return to_pm_basis(jnp.concatenate([weighted, state[1:]]))


def _physical_velocity(state: Array, flow_: CurvedPipeFlow) -> Array:
    r"""Carried state -> physical `$(u_s, u_r, u_\theta)$`.

    Undoes the metric weight, `$u_s = w_s/h$`.  Spectrally `$1/h$`
    couples every azimuthal mode, so the division is done where it is
    pointwise -- one transform round trip on one field, paid once per
    stats or snapshot interval rather than per corrector iteration.
    Reads the three velocity slots only, so a solver state's carried
    slots are dropped.
    """
    physical = from_pm_basis(state)
    unweighted = phys_to_spec_2d(
        spec_to_phys_2d(physical[:1]) * flow_.inv_h_phys
    )
    return jnp.concatenate([unweighted, physical[1:]])


@jit
def _to_solver_basis_jit(
    state: Array, fourier_: Fourier, flow_: CurvedPipeFlow
) -> Array:
    return with_carried(_carried_velocity(state, flow_), fourier_, flow_)


@jit
def _from_solver_basis_jit(state: Array, flow_: CurvedPipeFlow) -> Array:
    return _physical_velocity(state, flow_)


def to_solver_basis(state: Array) -> Array:
    r"""Physical `$(u_s, u_r, u_\theta)$` -> the solver state.

    The carried velocity `$(w_s, w_+, w_-)$` (:func:`_carried_velocity`)
    followed, under the default ``res.consistent_imm``, by the two
    spin-quad differences the pass carries, derived from it
    (:func:`~dnsjax.geometries.wall_bounded._cylindrical_stepping.with_carried`).
    Jitted with the flow as an **argument**: the metric is a global
    array, which a jit closing over it would bake into the program as
    a constant -- accepted on one process, refused at trace time by a
    multi-process run.
    """
    return _to_solver_basis_jit(state, fourier, flow)


def from_solver_basis(state: Array) -> Array:
    r"""Carried state -> physical `$(u_s, u_r, u_\theta)$`.

    :func:`_physical_velocity`, jitted with the flow as an argument
    (see :func:`to_solver_basis`).  Everything downstream (snapshots,
    ``analysis``, the diagnostics below) therefore sees the
    **physical** components, as every other geometry's consumers do.
    """
    return _from_solver_basis_jit(state, flow)


def init_state() -> Array:
    r"""The ``start_from_laminar`` state: `$u_s = 1 - r^2$`.

    The straight pipe's Hagen-Poiseuille profile, in **physical**
    components at the `$(0,0)$` mode (``__main__`` crosses it into the
    carried basis, where it becomes `$w_s = h(1-r^2)$`).  It is the
    `$\kappa = 0$` steady state, not this flow's -- see the module
    docstring.
    """
    profile = 1.0 - flow.rs**2
    u_s = jnp.where(fourier.mean_mask, profile[:, None, None], 0.0)
    zero = jnp.zeros_like(u_s)
    return jnp.stack([u_s, zero, zero]).astype(sharding.complex_type)


# ── Diagnostic statistics ────────────────────────────────────────


def _metric_weight(field_: Array, flow_: CurvedPipeFlow) -> Array:
    r"""`$h\,f$` on a spectral field, exactly.

    `$h = 1 + \kappa\chi$` has three azimuthal harmonics, so this is
    one `$m\pm1$` shift -- which is why every `$rh$`-weighted integral
    below costs no transform: `$\int f\,\bar g\,r h\,dr\,d\theta =
    \langle h f, g\rangle$`.
    """
    return field_ + KAPPA * flow_.chi_mul(field_)


def _metric_norm2(
    field_: Array, fourier_: Fourier, flow_: CurvedPipeFlow
) -> Array:
    r"""Volume-averaged `$\int |f|^2\,r h\,dr\,d\theta\,ds$`.

    The toroidal volume element is `$r h$`, not `$r$`, so every energy
    and dissipation integral picks up the weight -- and since `$h$` is
    real, `$\int|f|^2 h = \langle f, h f\rangle$` is the ordinary
    inner product against the weighted field.  (The torus's *total*
    volume is still `$\pi L_s$`: the `$\kappa$` term integrates out.)
    """
    return get_inprod(
        field_,
        _metric_weight(field_, flow_),
        fourier_.k_metric,
        flow_.y_weights,
    )


def _energy_3d(
    state: Array, fourier_: Fourier, flow_: CurvedPipeFlow
) -> Array:
    r"""Kinetic energy of the streamwise-varying modes.

    `$E' = \tfrac12\langle u, h u\rangle$` restricted to `$k_s \neq 0$`
    -- identically zero on every two-dimensional (and hence on every
    steady) state of this flow, and nonzero exactly when the flow has
    developed streamwise structure.  The single definition shared by
    :func:`get_stats` and the laminarization read
    :func:`get_perturbation_energy`.
    """
    varying = state * (1.0 - flow_.kz0_mask)
    return _metric_norm2(varying, fourier_, flow_) / 2


def _applied_driving(
    carried: Array, fourier_: Fourier, flow_: CurvedPipeFlow
) -> Array:
    r"""The mean-mode forcing `$-\Pi$` a state is driven with.

    *carried* is the state in the **carried** basis
    (:func:`_carried_velocity`), because both terms below are balances
    of the carried variable `$w_s = h\,u_s$`.

    Under a constant pressure gradient the answer is the constant
    `$4/\mathrm{Re}$` the RHS adds, and costs nothing.  Under
    ``constant_bulk_velocity`` the corrector chooses it each step, and
    it cannot be read back from the accepted state -- the correction
    has already removed the deficit it was computed from -- so it is
    **inferred** from the `$(0,0)$` streamwise momentum balance
    instead:

    .. math::
        \partial_t \langle w_s\rangle = \langle N_s\rangle
        + 2\nu\,\partial_r w_s\big|_{r=1} ,

    with `$\langle\cdot\rangle$` the area average over `$r\,dr\,
    d\theta$` (the `$(0,0)$` mode of the viscous term integrates to
    the wall shear exactly, the `$m^2$` and `$k^2$` pieces vanishing
    there).  Holding the flux steady therefore applies

    .. math::
        -\Pi = -\langle N_s\rangle - 2\nu\,\partial_r w_s|_{r=1} ,

    exact when the mean flow is steady and differing by
    `$\partial_t\langle w_s\rangle$` when it is not -- the same
    caveat the straight pipe's wall-shear inference carries.  It is
    **not** that inference: the toroidal mean balance keeps
    `$O(\kappa)$` volume terms that no wall integral captures, which
    is what `$\langle N_s\rangle$` is here for, and why this costs one
    right-hand-side evaluation.

    The wall term is the carried variable's, not the physical shear:
    `$\partial_r w_s = h\,\partial_r u_s + \kappa\cos\theta\,u_s$`
    at the wall is `$h\,\partial_r u_s$`, whose poloidal mean keeps
    the `$(\kappa/2)\,\partial_r(u_{s,1} + u_{s,-1})$` of the
    `$\cos\theta$` harmonic -- the inner/outer shear asymmetry that
    Dean flow makes `$O(1)$`.  Read off `$u_s$` instead, the inference
    runs ~1 % low at `$\kappa = 0.1$` (measured on a steady Dean state
    against the corrector's applied value).
    """
    if FORCE_S != 0.0:
        # A constant pressure gradient applies exactly what the RHS
        # adds; nothing to infer, and no transform to pay for.
        return jnp.asarray(FORCE_S, dtype=sharding.float_type)
    tau = jnp.einsum("j, jmz -> mz", flow_.D1_wall.ravel(), carried[0])
    wall = jnp.sum(jnp.where(fourier_.mean_mask[0], tau, 0.0)).real
    viscous = 2.0 * derived_params.nu * wall
    rhs = _get_rhs(carried, fourier_, flow_)
    mean_rhs = extract_mean_mode(rhs[:1])[0].real
    return -2.0 * jnp.dot(flow_.y_weights, mean_rhs) - viscous


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: CurvedPipeFlow
) -> dict[str, Array]:
    r"""Compute diagnostic statistics.

    *state* is the **physical** `$(u_s, u_r, u_\theta)$` view.  Every
    integral carries the toroidal volume element `$r h$`
    (:func:`_metric_norm2`).

    - `$E'$`: kinetic energy of the `$k_s \neq 0$` modes -- the
      transition indicator (:func:`_energy_3d`).
    - `$E$`: total kinetic energy.
    - `$D$`: dissipation in the **enstrophy** form
      `$\nu\langle|\boldsymbol{\omega}|^2\rangle$`, which the no-slip
      wall and streamwise periodicity make equal to
      `$\nu\langle|\nabla\mathbf{u}|^2\rangle$` in the continuum.
      The physical vorticity comes from the carried state's *straight*
      curl divided by `$h$` (identity 1 of the geometry docstring), so
      it costs one transform round trip and no toroidal gradient
      operator -- which is why this form and not the other.
      Discretely the two differ by the finite-difference
      integration-by-parts residual, and the gap is a usable
      under-resolution diagnostic: measured against ``pipe``'s own
      gradient-form `$D$` at `$\kappa = 0$` on the same stepped state,
      it is `$4.0\times10^{-4}$` relative at `$n_r = 24$`,
      `$3.5\times10^{-5}$` at 40 and `$1.9\times10^{-6}$` at 64
      (``fd_order`` 6).  Every other reported quantity agrees with
      ``pipe`` to ten digits there.
    - `$I$`: energy input `$-\Pi\,U_b$` -- the driving does work only
      against the mass flux, because its `$1/h$` profile and the `$h$`
      of the volume element cancel exactly.
    - `$U_{b,s}$`: bulk streamwise velocity, i.e. the mass flux over the
      cross-section area.
    - `$\tau_s$`: poloidal-mean wall shear `$\nu\,\partial_r u_s$` at
      `$r = 1$`.

    `$I = D$` is the steady-state balance these two are for.
    """
    nu = derived_params.nu
    carried = _carried_velocity(state, flow_)

    # Physical vorticity: Omega = curl_0(w) is (h w_r, h w_th, w_s)'s
    # own curl and equals (h om_r, h om_th, om_s), so one collocation
    # division recovers omega itself.
    omega_h = _curl_fn(from_pm_basis(carried), fourier_, flow_)
    omega = jnp.concatenate(
        [
            omega_h[:1],
            phys_to_spec_2d(spec_to_phys_2d(omega_h[1:]) * flow_.inv_h_phys),
        ]
    )

    bulk_s = flow_.bulk_deficit(carried[0]) + BULK_TARGET
    force = _applied_driving(carried, fourier_, flow_)
    wall_shear = jnp.einsum("j, jmz -> mz", flow_.D1_wall.ravel(), state[0])
    tau_s = (
        nu * jnp.sum(jnp.where(fourier_.mean_mask[0], wall_shear, 0.0)).real
    )

    return {
        "E'": _energy_3d(state, fourier_, flow_),
        "E": _metric_norm2(state, fourier_, flow_) / 2,
        "D": nu * _metric_norm2(omega, fourier_, flow_),
        "I": force * bulk_s,
        "Ub_s": bulk_s,
        "tau_s": tau_s,
    }


def get_stats(state: Array) -> dict[str, Array]:
    """Wrapper around ``_get_stats_jit`` (physical-basis *state*)."""
    return _get_stats_jit(state, fourier, flow)


@jit
def _get_perturbation_energy_jit(
    state: Array, fourier_: Fourier, flow_: CurvedPipeFlow
) -> Array:
    return _energy_3d(state, fourier_, flow_)


def get_perturbation_energy(state: Array) -> Array:
    r"""The `$k_s \neq 0$` energy (for the laminarization check).

    Takes the **physical** view, like :func:`get_stats`, and reports
    the same number as its ``E'`` column.
    """
    return _get_perturbation_energy_jit(state, fourier, flow)


@jit
def _get_driving_jit(
    state: Array, fourier_: Fourier, flow_: CurvedPipeFlow
) -> dict[str, Array]:
    if params.phys.driving != "constant_bulk_velocity":
        return {}
    return {
        flow_.driving_key: _applied_driving(
            _carried_velocity(state, flow_), fourier_, flow_
        )
    }


def get_driving(state: Array) -> dict[str, Array]:
    r"""Applied mean-mode driving, from *state* alone.

    The optional flow-module export ``__main__`` uses for the one
    ``stats.dat`` row with no step behind it (`$t = t_0$`); same key
    and sign as the corrector's own column.  ``{}`` under a constant
    pressure gradient, where the applied force is the known constant
    `$4/\mathrm{Re}$` and no column is written.
    """
    return _get_driving_jit(state, fourier, flow)
