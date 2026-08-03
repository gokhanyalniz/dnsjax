r"""Viscoelastic (sPTT) pipe flow, driven by an axial pressure gradient.

Summary of assumptions in the port from the annular geometry
------------------------------------------------------------
This flow was derived from the annular sPTT flow
(:mod:`~dnsjax.flows.wall_bounded.viscoelastic_dean`) rather than from
an independent formulation.  The deductions made in that port, in full:

1. The vector-form momentum and conformation equations are carried
   over unchanged.  Both geometries use the same cylindrical
   coordinates `$(z, r, \theta)$`, so every component expression --
   the tensor advection with its basis-rotation (Christoffel) terms,
   the stretching `$(\nabla u)^{\!\top}c + c\nabla u$`, the sPTT
   relaxation `$f = 1 - 3\epsilon + \epsilon\,\mathrm{tr}\,c$`, the
   polymer stress `$\tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}
   \nabla\cdot c$` and the spin-diagonal tensor Laplacian with
   `$m_{\mathrm{eff}} = m + s$` -- transfers verbatim.  The shared
   code is :mod:`~dnsjax.geometries.wall_bounded._viscoelastic_common`
   (the coordinate-level algebra) and
   :mod:`~dnsjax.geometries.wall_bounded._viscoelastic_stepping` (the
   stepping functions, incl. the tensor Laplacian).
2. The driving translates as: the annulus's constant *azimuthal* mean
   pressure gradient (`$\Pi_\theta = (r_1+r_2)/(\mathrm{Re}\,r)$`)
   becomes a constant *axial* one, `$\Pi_z = 4/\mathrm{Re}$` (uniform
   in `$r$`), applied at the mean mode.  So azimuth `$\to$` spanwise
   and axial `$\to$` streamwise, and the flow has no Dean-ness left:
   the only remaining novelty is the viscoelastic coupling.
3. The amplitude keeps the annular normalisation *principle* -- the
   force is fixed so that the `$\epsilon = 0$` laminar profile has unit
   centreline velocity -- which in the pipe gives the
   Hagen-Poiseuille `$W = 1 - r^2$` of
   :mod:`~dnsjax.flows.wall_bounded.pipe`.  Lengths are pipe radii and
   `$\mathrm{Re}$` is built on those two scales, as for the Newtonian
   pipe; `$\mathrm{Re} := \mathrm{Wi}/\mathrm{El}$` is derived, as for
   the annular twin.
4. The total-field formulation is retained: ``base_flow = 0``, the
   force enters at the mean mode, and the reported `$E'$` is the
   velocity-only deviation from the laminar profile.
5. The laminar conformation is the pointwise sPTT equilibrium on the
   discrete shear `$S = D_1 W$` -- with **no** curvature term, against
   the annulus's `$S = D_1 U_\theta - U_\theta/r$` -- and the sheared
   pair moves from `$(c_{r\theta}, c_{\theta\theta})$` to
   `$(c_{rz}, c_{zz})$`.  See "Laminar state" below.
6. Axis regularity of the conformation is enforced by **parity only**,
   `$(-1)^{m+s}$` per spin-`$s$` slot -- the same treatment (and the
   same approximation class) the pipe already applies to `$u_\pm$`.
   The full `$r^{|m+s|}$` vanishing rates are not separately imposed
   on the carried state.  Derivation and the per-component table: the
   :mod:`~dnsjax.geometries.wall_bounded.cylindrical_viscoelastic`
   module docstring.
7. The non-conservative discrete forms of the tensor advection
   (explicit `$u_\theta/r$` terms) and of `$\nabla\cdot c$` (explicit
   `$1/r$` terms) carry over unchanged; every grid point has
   `$r > 0$`.  The conservative-curl requirement of the
   pressure-elimination chain is unaffected -- that lives in
   ``cylindrical._imm_iteration_vw``, which consumes the
   polymer-augmented sources without change.
8. The conformation BC is `$\nabla^2 c = 0$` at the single wall
   `$r = 1$` when `$\kappa > 0$` (the annulus has two such rows); the
   axis needs none.  ``beta`` / ``epsilon`` / ``kappa`` default to the
   shared sPTT reference values, but ``el`` / ``wi`` do **not**: since
   `$\mathrm{Re} := \mathrm{Wi}/\mathrm{El}$` is derived, those two
   *are* the Reynolds number, so the pipe picks its own regime
   (`$\mathrm{Wi} = 20$`, `$\mathrm{El} = 0.02$`, hence
   `$\mathrm{Re} = 1000$` -- the Newtonian pipe's default, and the
   elasto-inertial range a viscoelastic pipe is usually run in) rather
   than inheriting the annulus's inertialess `$\mathrm{El} = 80$`.
9. The wall-normal grid surface is the Newtonian pipe's, not the
   annulus's: ``half-cgl`` under ``iterative-cn`` and ``rigged-cgl``
   under ``cnab2`` (with the same validation that half-CGL's tighter
   axis point is ``iterative-cn``-only), since the constraint is the
   near-axis explicit CFL, which the conformation does not change.
10. ``phys.u_grid`` defaults to the `$\epsilon = 0$` laminar bulk
    `$1/2$`, as for the Newtonian pipe -- meaningful here because
    streamwise *is* the grid direction.  (The annular twin defaults to
    0 because its streamwise direction is azimuthal, not a
    frame-translatable one; that was never a viscoelastic constraint.)
11. ``phys.block_mean_spanwise_velocity`` is **not** offered: the
    cylindrical geometry has no mean-blocking machinery (the Newtonian
    pipe does not offer it either), and the undriven mean here is the
    azimuthal swirl, which would need its own odd-parity treatment.
12. The stored / observed layout is the annulus's unchanged: the same
    9 physical components in the same order, so snapshots, the probe
    stream and the analysis package need no new component schema --
    only the axis parity classes of assumption 6.
13. ``[probes]`` is supported and ``[force]`` is rejected, as for every
    viscoelastic flow; transient growth is out of scope (a total-field
    flow), so no ``frozen_profile_flow`` is exported.

*This summary is scaffolding for the port review and is meant to be
removed once the flow stands on its own; everything below documents the
flow directly.*

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
The reconstruction scheme is supported here, as for every other
wall-bounded flow, and needs nothing viscoelastic: the polymer stress
reaches the velocity solve only as one more source term, which either
influence-matrix scheme projects unchanged.

It is worth recording why this flow briefly did *not* offer it, since
the stated reason was wrong twice over.  The port measured a flag-on
blow-up and attributed it to the polymer passing through the flag-on
scheme's extra source derivative.  In fact the same divergence appears
at `$\beta = 1$`, where the polymer stress is decoupled from the
velocity entirely, and in the *Newtonian* ``pipe`` at the same
`$\mathrm{Re}$` -- so it was never elastic; `$\kappa$` does not move
it and `$\Delta t$` is not a lever either.  What it actually was is a
defect in the **cylindrical** flag-on pass, whose two free wall
differences were lagged to `$t^n$`, closing a growth loop across time
steps.  This flow merely exposed it, by having defaulted to
`$\mathrm{Re} \approx 1$` -- the regime it inherited from the annular
sPTT flow, where `$\nu = \beta/\mathrm{Re}$` is three decades larger
than at its own `$\mathrm{Re} = 1000$`.  That defect is fixed (the
differences now come from the corrector iterate); the measurements,
the controls that ruled out the elastic explanation, and the
resolution dependence that made it look like a `$\mathrm{Re}$`
threshold are all in ``cylindrical._imm_iteration_vw``.

Guard: the ``viscoelastic-pipe-consistent-imm`` entry in
``tests/test_random_smoke.py`` -- with a caveat about what an entry of
that shape can do.  The *laminar* one is a linear gate and was a clean
fixed point at `$\mathrm{Re} = 1$` throughout the period the scheme
was known to diverge nonlinearly; and the old defect's boundary was
crossed by wall-normal **refinement**, which no fixed-resolution entry
can follow.

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
    predict_and_correct,
    iterate_correction,
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


def init_state(snapshot: str | None) -> Array:
    """Initialise the 9-component total-field state.

    ``start_from_laminar`` returns the analytical laminar pair.  Legacy
    ``.npz`` snapshots are not supported for the tensor state (a zarr3
    snapshot resume is handled in ``__main__`` before this is called);
    the random / localized-rolls modes are built in
    :mod:`dnsjax.random_field` / :mod:`dnsjax.localized_rolls`.
    """
    if snapshot is None and params.init.start_from_laminar:
        # Copy: the steppers donate their state argument, and the
        # module-level ``_laminar_state`` must survive for the E'
        # deviation in ``get_stats`` / ``get_perturbation_energy``.
        return jnp.copy(_laminar_state)
    if snapshot is not None:
        raise NotImplementedError(
            "viscoelastic-pipe does not support legacy .npz snapshots; "
            "use a zarr3 (.tar) snapshot or an in-process IC."
        )
    sharding.print("Provide an initial condition.")
    sharding.exit(code=1)


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
