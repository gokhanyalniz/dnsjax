r"""Viscoelastic (sPTT) pipe flow, driven by an axial pressure gradient.

The flow
--------
Pressure-driven flow of an sPTT viscoelastic fluid through a circular
pipe, driven by a uniform axial body force
`$\Pi_z = 4/\mathrm{Re}$` (a constant mean pressure gradient
`$-\partial p/\partial z$`).  The velocity is coupled to a symmetric
conformation tensor `$\mathbf{c}$` via the polymer-stress divergence;
see the module docstring of
:mod:`~dnsjax.geometries.wall_bounded.cylindrical_viscoelastic` for the
governing equations, the 9-component state layout, the axis parity of
the tensor, and the spin diagonalisation of its Laplacian.

Total-field formulation
-----------------------
The **total** velocity is integrated (base flow zero, so the rotational
nonlinear term evaluates the full field), with the axial body force
applied at the mean mode through ``pi_z``.  The reported perturbation
energy `$E'$` is velocity-only: the kinetic energy of the deviation of
`$\mathbf{u}$` from the analytical laminar profile (the
laminarization-check quantity).  ``get_stats`` also reports polymer
diagnostics (mean trace, elastic energy, polymer work).

Laminar state
-------------
``start_from_laminar`` uses the analytical laminar pair: the
Hagen-Poiseuille profile `$W(r) = 1 - r^2$` (the exact balance of
`$\Pi_z$` against the *total* solvent + polymer stress at
`$\epsilon = 0$`, where `$c_{rz} = \mathrm{Wi}\,W'$` makes the polymer
divergence `$\mathrm{Wi}\,\nabla^2 W$` and the two viscosities sum back
to `$1/\mathrm{Re}$`) and the pointwise sPTT-equilibrium conformation
on the **discrete** local shear `$S = D_1 W$`,

.. math::
    c_{rr} = c_{\theta\theta} = 1, \quad
    c_{rz} = \frac{\mathrm{Wi}\,S}{f}, \quad
    c_{zz} = 1 + \frac{2(\mathrm{Wi}\,S)^2}{f^2}, \quad
    f^3 - f^2 = 2\epsilon(\mathrm{Wi}\,S)^2,

with `$c_{r\theta} = c_{\theta z} = 0$`.  For `$\kappa = 0$` the
conformation slice of the full 9-component RHS vanishes to machine
precision at every `$\epsilon$` (the flow is unidirectional, so the
advection and all but one stretching term drop out algebraically); for
`$\kappa > 0$` it is an approximation (the diffusion of the
`$r$`-varying conformation is neglected, and the equilibrium profile
does not satisfy the `$\nabla^2 c = 0$` wall row `$H_c$` imposes).

**The velocity slice closes exactly only at** `$\epsilon = 0$`, where
`$W = 1 - r^2$` is the true profile -- at `$\epsilon > 0$` the polymer
shear-thins (the true balance is
`$S[\beta + (1-\beta)/f] = -\mathrm{Re}\,\Pi_z r/2$`) and that
correction is neglected, as in the annular twin, so the pair is
**not** a steady state there.  It is not small at the shipped
defaults: stepping it at `$\epsilon = 10^{-3}$` drifts at
`$\max|\Delta u|/\Delta t \approx 5\!\cdot\!10^{-4}$` against
`$2\!\cdot\!10^{-13}$` at `$\epsilon = 0$`, and the laminar ledger
`$I = D_s - W_p$` misses by 8 % (the annulus's 15 %).  Consequences:
``start_from_laminar`` is not laminar, and `$E'$` -- which is measured
against *this* reference, hence the ``stop.check_laminarization``
quantity -- has a floor instead of decaying to zero, so that stop
cannot fire at the default `$\epsilon$`.  Fixing it needs a fixed-point
solve for `$W$` (invert `$\tau(S)$` through the cubic and integrate),
which is why the closed form is kept and the defect documented
instead.

``res.consistent_imm``
----------------------
The default reconstruction scheme is supported here, as for every other
wall-bounded flow, and needs nothing viscoelastic: the polymer stress
reaches the velocity solve only as one more source term, which either
influence-matrix formulation projects unchanged.  Guards: the
``viscoelastic-pipe`` entry in ``tests/test_random_smoke.py`` for the
default and ``viscoelastic-pipe-legacy-imm`` for the legacy path (both
nonlinear gates -- the laminar smoke is linear and blind to either
formulation's nonlinear path, and a fixed-resolution entry cannot see
refinement-crossed thresholds).

Moving frame
------------
``phys.u_grid`` defaults to the laminar bulk velocity
`$U_{b,\mathrm{lam}} = 1/2$`, so by default the run evolves in the
frame translating axially at `$U_{grid} = 1/2$`; the convective frame
term is mode-diagonal on all 9 components.  Set ``--phys.u_grid 0`` for
the lab frame.
"""

from dataclasses import dataclass

import numpy as np
from jax import Array, jit
from jax import numpy as jnp

from ...geometries.wall_bounded._base import (
    extract_mean_mode,
    pad_base_flow,
)
from ...geometries.wall_bounded.cylindrical import (
    get_norm2_cyl,
    get_pert_enstrophy_cyl,
    integrate_scalar,
)
from ...geometries.wall_bounded.cylindrical_viscoelastic import (
    Fourier,
    ViscoelasticCylindricalFlow,
    _div_c,
    build_viscoelastic_stepper,
    fourier,
    from_solver_basis,  # noqa: F401 — re-exported (basis boundary)
    get_norm2_conformation,  # noqa: F401 -- available for callers
    parity_d1_even,
    to_solver_basis,  # noqa: F401 — re-exported (basis boundary)
    viscoelastic_laminar_profiles,
)
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class ViscoelasticPipeFlow(ViscoelasticCylindricalFlow):
    r"""Precomputed data for viscoelastic (sPTT) pipe flow.

    Delegates the velocity grid / parity-reduced FD matrices / IMM
    operators (solvent viscosity `$\nu = \beta/\mathrm{Re}$`) and the
    conformation Helmholtz operator to
    :class:`ViscoelasticCylindricalFlow`, then sets the uniform axial
    body force `$\Pi_z = 4/\mathrm{Re}$` and zeros the base flow
    (total-field integration).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        Re = params.phys.re

        # Uniform axial body force Pi_z = 4 / Re, applied at the mean
        # mode by ``ViscoelasticCylindricalFlow.add_mean_body_force``
        # (the shared RHS's driving adapter).  The
        # coefficient is fixed by the epsilon = 0 laminar balance
        # (module docstring): 4/Re makes W = 1 - r^2 the total-stress
        # solution, i.e. unit centreline velocity.
        self.pi_z = jnp.full(
            params.res.ny,
            4.0 / Re,
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )

        # Total-field formulation: no base flow to subtract.
        self.base_flow = jnp.zeros(
            (3, params.res.ny),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )[:, :, None, None]
        self.curl_base_flow = jnp.zeros_like(self.base_flow)
        pad_base_flow(self)


flow: ViscoelasticPipeFlow = ViscoelasticPipeFlow()

(
    _init_state_laminar_zero,  # overridden below
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_viscoelastic_stepper(flow)


def _build_laminar_state() -> Array:
    r"""Spectral 9-component laminar state at the mean mode.

    The analytical laminar `$r$`-profiles (Hagen-Poiseuille velocity +
    sPTT-equilibrium conformation; see
    :func:`~dnsjax.geometries.wall_bounded.cylindrical_viscoelastic.viscoelastic_laminar_profiles`)
    placed at the mean mode `$(m, k_z) = (0, 0)$`.
    """
    rs = np.asarray(flow.rs)
    prof = viscoelastic_laminar_profiles(
        rs,
        parity_d1_even(rs, params.res.fd_order),
        params.phys.wi,
        params.phys.epsilon,
    )
    prof_jax = jnp.asarray(prof, dtype=sharding.complex_type)
    return jnp.where(fourier.mean_mask[None], prof_jax[:, :, None, None], 0.0)


#: Physical-basis laminar reference.  It never enters the solver as
#: itself -- ``init_state`` hands it to ``__main__``, which converts
#: the initial state once -- so the ``state - laminar_state``
#: deviations below are both physical.
_laminar_state: Array = _build_laminar_state()


def _perturbation_energy(
    state: Array,
    laminar_state: Array,
    fourier_: Fourier,
    flow_: ViscoelasticPipeFlow,
) -> Array:
    r"""Velocity-only deviation energy `$E' = \|u -
    U_{\mathrm{lam}}\|^2/2$`.

    The single definition, shared by :func:`get_stats` (which reports
    it as ``E'``) and the laminarization read
    :func:`get_perturbation_energy`.
    """
    return (
        get_norm2_cyl(
            state[:3] - laminar_state[:3],
            fourier_.k_metric,
            flow_.y_weights,
        )
        / 2
    )


def init_state() -> Array:
    """The ``start_from_laminar`` 9-component state.

    Returns the analytical laminar velocity/conformation pair (this
    flow integrates the **total** field).  Snapshot resume and the
    in-process random / localized-rolls modes
    (:mod:`dnsjax.ic.random_field` / :mod:`dnsjax.ic.localized_rolls`)
    are dispatched in ``__main__``; this is called only for the
    laminar start.
    """
    # Copy: the steppers donate their state argument, and the
    # module-level ``_laminar_state`` must survive for the E'
    # deviation in ``get_stats`` / ``get_perturbation_energy``.
    return jnp.copy(_laminar_state)


# ── Diagnostic statistics ────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array,
    laminar_state: Array,
    fourier_: Fourier,
    flow_: ViscoelasticPipeFlow,
) -> dict[str, Array]:
    r"""Total-field diagnostics + polymer quantities.

    *state* and *laminar_state* are the **physical** 9-component view
    `$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` -- diagnostics sit outside the
    solver, whose working basis is the `$u_\pm$`/spin one (the
    ``cylindrical_viscoelastic.py`` module docstring).

    - `$E$`: total kinetic energy; `$E'$`: velocity-only deviation from
      the laminar profile (laminarization quantity).
    - `$I = \langle u_z \Pi_z \rangle$`: body-force input.
    - `$D_s = \nu\langle|\nabla u|^2\rangle$`: solvent dissipation
      (`$\nu = \beta/\mathrm{Re}$`).
    - `$W_p = \tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}
      \langle u\cdot\nabla\cdot c\rangle$`: polymer work.  In steady
      state `$I = D_s - W_p$` (`$W_p < 0$`: the polymer drains the
      velocity field).
    - `$\langle\mathrm{tr}\,c\rangle$`, elastic energy
      `$E_p = \tfrac{1-\beta}{2\,\mathrm{Re}\,\mathrm{Wi}}
      (\langle\mathrm{tr}\,c\rangle - 3)$`.
    - wall shear stresses (solvent) at `$r = 1$`, bulk velocities.

    The bulk integrals are parity-matched: the mean `$u_z$` is even in
    `$r$` (`$m = 0$`, parity `$(-1)^m$`) and the mean `$u_\theta$` odd,
    so each uses the quadrature vector of its own parity -- as in the
    Newtonian pipe's ``get_stats``.
    """
    Re = params.phys.re
    volfac = derived_params.volume_fac
    nu = derived_params.nu
    coef = (1.0 - params.phys.beta) / (Re * params.phys.wi)

    vel = state[:3]
    total_energy = get_norm2_cyl(vel, fourier_.k_metric, flow_.y_weights) / 2
    perturbation_energy = _perturbation_energy(
        state, laminar_state, fourier_, flow_
    )

    # Mean velocity profiles.
    mean_vel = extract_mean_mode(vel)  # (3, Nr)
    mean_uz = mean_vel[0].real
    mean_utheta = mean_vel[2].real

    D1_wall_row = flow_.D1_wall.ravel()
    tau_z = jnp.dot(D1_wall_row, mean_uz)
    tau_theta = jnp.dot(D1_wall_row, mean_utheta)
    U_bulk_z = integrate_scalar(mean_uz, flow_.y_weights) / volfac
    U_bulk_theta = integrate_scalar(mean_utheta, flow_.y_weights_odd) / volfac

    energy_input = (
        integrate_scalar(mean_uz * flow_.pi_z, flow_.y_weights) / volfac
    )
    enstrophy = get_pert_enstrophy_cyl(
        vel,
        flow_.D1_pos,
        flow_.D1_ghost,
        fourier_.m_is_even,
        flow_.inv_r,
        fourier_.m,
        fourier_.kz2,
        fourier_.k_metric,
        flow_.y_weights,
    )
    dissipation = nu * enstrophy

    # Polymer diagnostics (native components throughout).
    div_r, div_th, div_z = _div_c(
        state[6],
        state[7],
        state[8],
        state[4],
        state[5],
        state[3],
        fourier_,
        flow_,
    )
    # <u . div c> as a spectral inner product in (z, r, theta).
    div_rthz = jnp.array([div_z, div_r, div_th])
    polymer_work = coef * (
        integrate_scalar(
            jnp.sum(
                jnp.conj(vel) * fourier_.k_metric * div_rthz, axis=(0, 2, 3)
            ).real,
            flow_.y_weights,
        )
        / volfac
    )

    mean_c = extract_mean_mode(state[3:])  # (6, Nr)
    # tr c = c_zz + c_rr + c_theta_theta.
    trace_profile = (mean_c[0] + mean_c[3] + mean_c[4]).real
    mean_trace = integrate_scalar(trace_profile, flow_.y_weights) / volfac
    elastic_energy = (
        (1.0 - params.phys.beta)
        / (2.0 * Re * params.phys.wi)
        * (mean_trace - 3.0)
    )

    return {
        "E": total_energy,
        "E'": perturbation_energy,
        "I": energy_input,
        "D_s": dissipation,
        "W_p": polymer_work,
        "E_p": elastic_energy,
        "TrC": mean_trace,
        "tau_z": nu * tau_z,
        "tau_th": nu * tau_theta,
        "Ub_z": U_bulk_z,
        "Ub_th": U_bulk_theta,
    }


def get_stats(state: Array) -> dict[str, Array]:
    """Wrapper around ``_get_stats_jit`` (physical-basis *state*)."""
    return _get_stats_jit(state, _laminar_state, fourier, flow)


@jit
def _get_perturbation_energy_jit(
    state: Array,
    laminar_state: Array,
    fourier_: Fourier,
    flow_: ViscoelasticPipeFlow,
) -> Array:
    r"""Velocity-only deviation energy (laminarization check)."""
    return _perturbation_energy(state, laminar_state, fourier_, flow_)


def get_perturbation_energy(state: Array) -> Array:
    r"""Velocity-only perturbation energy E' (laminarization check).

    Takes the **physical** 9-component view, like :func:`get_stats`,
    which reports the same number as ``E'``.
    """
    return _get_perturbation_energy_jit(state, _laminar_state, fourier, flow)
