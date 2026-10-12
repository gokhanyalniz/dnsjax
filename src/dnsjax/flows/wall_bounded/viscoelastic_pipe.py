r"""Viscoelastic (sPTT) pipe flow, driven by an axial pressure gradient.

The flow
--------
Pressure-driven flow of an sPTT viscoelastic fluid through a circular
pipe, driven by a uniform axial body force
`$-\Pi_z = 4/\mathrm{Re}$`, i.e. a constant mean pressure gradient
`$\Pi_z = \partial p/\partial z = -4/\mathrm{Re}$`.  The velocity is
coupled to a symmetric conformation tensor `$\mathbf{c}$` via the
polymer-stress divergence; see the module docstring of
:mod:`~dnsjax.geometries.wall_bounded.cylindrical_viscoelastic` for the
governing equations, the 9-component state layout, the axis parity of
the tensor, and the spin diagonalization of its Laplacian.

Total-field formulation
-----------------------
The **total** velocity is integrated (base flow zero, so the rotational
nonlinear term evaluates the full field), with the axial body force
applied at the mean mode through ``force_z``.  The reported perturbation
energy `$E'$` is velocity-only: the kinetic energy of the deviation of
`$\mathbf{u}$` from the analytical laminar profile (the
laminarization-check quantity).  ``get_stats`` also reports polymer
diagnostics (mean trace, elastic energy, polymer work).

Laminar state
-------------
``start_from_laminar`` uses the analytical laminar pair: the
Hagen-Poiseuille profile `$W(r) = 1 - r^2$` (the exact balance of
`$-\Pi_z$` against the *total* solvent + polymer stress at
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
`$S[\beta + (1-\beta)/f] = \mathrm{Re}\,\Pi_z r/2$`) and that correction is
neglected, as in ``viscoelastic-dean``, so the pair is **not** a steady state
there, and the defect is not small at the shipped defaults (Design notes: "The
laminar defect, measured").  Consequences: ``start_from_laminar`` is not
laminar, and `$E'$` -- which is measured against *this* reference, hence the
``stop.check_laminarization`` quantity -- has a floor instead of decaying to
zero, so that stop cannot fire at the default `$\epsilon$`.  Fixing it needs a
fixed-point solve for `$W$` (invert `$\tau(S)$` through the cubic and
integrate), which is why the closed form is kept and the defect documented
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

Design notes
------------
**The laminar defect, measured.**  Stepping the laminar pair at
`$\epsilon = 10^{-3}$` drifts at
`$\max|\Delta u|/\Delta t \approx 5\!\cdot\!10^{-4}$`, against
`$2\!\cdot\!10^{-13}$` at `$\epsilon = 0$`, and the laminar ledger
`$I = D_s - W_p$` misses by 8 % (``viscoelastic-dean``: 15 %).
"""

from dataclasses import dataclass
from math import pi

import numpy as np
from jax import Array, jit
from jax import numpy as jnp

from ...geometries.wall_bounded._base import (
    extract_mean_mode,
    pad_base_flow,
)
from ...geometries.wall_bounded._viscoelastic_common import to_spin_basis
from ...geometries.wall_bounded.cylindrical import (
    CARRIED_FIELDS,  # noqa: F401 — re-exported (snapshot carry/ member)
    get_norm2_cyl,
    get_pert_enstrophy_cyl,
    integrate_scalar,
    with_carried,
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
    viscoelastic_laminar_profiles,
)
from ...harmonics import complex_harmonics, real_harmonics
from ...ic.localized_rolls import generate_cylindrical_rolls
from ...ic.random_field import (
    column_draw,
    hermitian_column,
    normalize_mode,
    real_product,
    scaled_wall_window,
    wall_normal_filter,
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
    body force `$-\Pi_z = 4/\mathrm{Re}$` and zeros the base flow
    (total-field integration).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        Re = params.phys.re

        # Uniform axial body force -Pi_z = 4 / Re, applied at the mean
        # mode by ``ViscoelasticCylindricalFlow.add_mean_body_force``
        # (the shared RHS's driving adapter).  The coefficient is fixed
        # by the epsilon = 0 laminar balance (module docstring): 4/Re
        # makes W = 1 - r^2 the total-stress solution, i.e. unit
        # centreline velocity.
        self.force_z = jnp.full(
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


@jit
def _to_solver_basis_jit(
    state: Array, fourier_: Fourier, flow_: ViscoelasticPipeFlow
) -> Array:
    return with_carried(to_spin_basis(state), fourier_, flow_)


def to_solver_basis(state: Array) -> Array:
    r"""Physical 9-component state -> the solver state.

    The spin map (``_viscoelastic_common.to_spin_basis``), followed --
    under the default ``res.consistent_imm`` -- by the two spin-quad
    differences the velocity pass carries, derived from the velocity.
    ``from_solver_basis`` drops them again.
    """
    return _to_solver_basis_jit(state, fourier, flow)


(
    _init_state_laminar_zero,  # overridden below
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_viscoelastic_stepper(flow)


def _build_laminar_profile() -> Array:
    r"""The laminar state's mean-mode column, ``(9, N_r)``.

    The analytical laminar `$r$`-profiles (Hagen-Poiseuille velocity +
    sPTT-equilibrium conformation; see
    :func:`~dnsjax.geometries.wall_bounded.cylindrical_viscoelastic.viscoelastic_laminar_profiles`)
    -- the only nonzero column of the laminar state.
    """
    rs = np.asarray(flow.rs)
    prof = viscoelastic_laminar_profiles(
        rs,
        parity_d1_even(rs, params.res.fd_order),
        params.phys.wi,
        params.phys.epsilon,
    )
    return sharding.distribute(
        np.asarray(prof, dtype=sharding.complex_type), sharding.no_shard
    )


#: The laminar reference, kept as its mean-mode column rather than the
#: field it spans: the full ``(C, N_r, N_m, N_{k_z})`` array would hold
#: a whole state's worth of device memory for the run to carry
#: ``C N_r`` nonzero values.  :func:`_laminar_field` rebuilds the field
#: where it is needed -- inside the jitted diagnostics, fused into the
#: subtraction -- and ``init_state`` builds it on demand.  Physical
#: basis: ``init_state`` hands it to ``__main__``, which converts the
#: initial state once, so the deviations below are physical too.
_laminar_profile: Array = _build_laminar_profile()


def _laminar_field(profile: Array, fourier_: Fourier) -> Array:
    """The spectral field that is *profile* ``(C, N_r)`` on the mean mode.

    Zero on every other mode, padding included.
    """
    return jnp.where(fourier_.mean_mask[None], profile[:, :, None, None], 0.0)


def _perturbation_energy(
    state: Array,
    laminar_profile: Array,
    fourier_: Fourier,
    flow_: ViscoelasticPipeFlow,
) -> Array:
    r"""Velocity-only kinetic energy of the deviation from laminar.

    `$E' = \|u - U_{\mathrm{lam}}\|^2/2$` over the three velocity components.
    The single definition, shared by :func:`get_stats` (which reports it as
    ``E'``) and the laminarization read :func:`get_perturbation_energy`.
    """
    return (
        get_norm2_cyl(
            state[:3] - _laminar_field(laminar_profile[:3], fourier_),
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
    return _laminar_field(_laminar_profile, fourier)


# ── Diagnostic statistics ────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array,
    laminar_profile: Array,
    fourier_: Fourier,
    flow_: ViscoelasticPipeFlow,
) -> dict[str, Array]:
    r"""Total-field diagnostics + polymer quantities.

    *state* and *laminar_profile* are the **physical** 9-component view
    `$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` -- diagnostics sit outside the
    solver, whose working basis is the `$u_\pm$`/spin one (the
    ``cylindrical_viscoelastic.py`` module docstring).

    - `$E$`: total kinetic energy; `$E'$`: velocity-only deviation from
      the laminar profile (laminarization quantity).
    - `$I = -\langle u_z \Pi_z \rangle$`: body-force input.
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
        state, laminar_profile, fourier_, flow_
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
        integrate_scalar(mean_uz * flow_.force_z, flow_.y_weights) / volfac
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
    """The ``stats.dat`` columns of the physical-basis *state*."""
    return _get_stats_jit(state, _laminar_profile, fourier, flow)


@jit
def _get_perturbation_energy_jit(
    state: Array,
    laminar_profile: Array,
    fourier_: Fourier,
    flow_: ViscoelasticPipeFlow,
) -> Array:
    r"""Velocity-only deviation energy (laminarization check)."""
    return _perturbation_energy(state, laminar_profile, fourier_, flow_)


def get_perturbation_energy(state: Array) -> Array:
    r"""Velocity-only perturbation energy E' (laminarization check).

    Takes the **physical** 9-component view, like :func:`get_stats`,
    which reports the same number as ``E'``.
    """
    return _get_perturbation_energy_jit(state, _laminar_profile, fourier, flow)


# ── Initial conditions ───────────────────────────────────────────


def add_viscoelastic_pipe_laminar(vel_state: Array) -> Array:
    r"""The pipe counterpart of :func:`add_viscoelastic_laminar`.

    Adds the Hagen-Poiseuille laminar velocity to *vel_state* and
    appends the laminar sPTT-equilibrium conformation (both at the mean
    mode), giving the 9-component total-field IC in the physical
    layout.  Used by the localized-rolls IC (a velocity-only
    perturbation); the random IC builds its 9 components directly.
    """
    from jax import numpy as jnp

    from dnsjax.geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
        fourier,
    )
    from dnsjax.geometries.wall_bounded.cylindrical_viscoelastic import (
        viscoelastic_laminar_profiles,
    )

    rs, D1_even, *_ = build_cylindrical_grid(
        params.res.ny,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    prof = viscoelastic_laminar_profiles(
        np.asarray(rs),
        np.asarray(D1_even),
        params.phys.wi,
        params.phys.epsilon,
    )
    prof_jax = jnp.asarray(prof, dtype=vel_state.dtype)
    laminar = jnp.where(
        fourier.mean_mask[None], prof_jax[:, :, None, None], 0.0
    )
    total_vel = vel_state + laminar[:3]
    return jnp.concatenate([total_vel, laminar[3:]])


# Viscoelastic conformation noise: every physical tensor
# component (c_zz, c_rz, c_theta_z, c_rr, c_theta_theta, c_r_theta) is
# the transform of a real field, so its draws use the same
# per-component machinery as the velocity -- with a trailing ``1`` in
# the PRNG key (the ``stream`` argument) to keep the conformation
# stream distinct from the velocity draw at the same mode.


def generate_viscoelastic_pipe(
    amplitude: float,
    conf_amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
) -> Array:
    r"""Random 9-component IC for viscoelastic (sPTT) pipe flow.

    Built per device (no full-array replication): the velocity part is
    the divergence-free pipe draw of :func:`generate_cylindrical` (rows
    ``0:3``); the conformation part (rows ``3:9``) is windowed,
    spectrally-decaying symmetric-tensor noise.  Velocity and
    conformation noise are rescaled to *amplitude* / *conf_amplitude*
    separately, then the analytical laminar pair (Hagen-Poiseuille
    velocity + sPTT-equilibrium conformation) is added at the mean mode
    (total-field IC).  The conformation noise vanishes at the wall and
    at the mean mode (so the laminar wall / mean values are preserved).

    Unlike the annular flow, the inner end is the pipe **axis**, so
    each tensor column also carries the axis-regularity envelope
    `$r^{|m+s|}$` of its spin weight `$s$` -- applied in the spin
    basis, for the same reason the velocity envelope is
    (:func:`generate_cylindrical`): the condition is a cancellation
    *between* physical components, so enveloping them separately would
    leave the high-spin combinations orders too large near the axis.
    Parity follows from the envelope.
    """
    from jax import numpy as jnp

    from dnsjax.geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
        fourier,
        get_norm2_cyl,
    )
    from dnsjax.geometries.wall_bounded.cylindrical_viscoelastic import (
        viscoelastic_laminar_profiles,
    )
    from dnsjax.snapshot import assemble_local_shards

    nx = params.res.nx
    Nr = params.res.ny
    nz = params.res.nz

    rs, D1_even, D1_odd, _, y_weights, _, inv_r = build_cylindrical_grid(
        Nr,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    rs_np = np.asarray(rs)
    inv_r_np = np.asarray(inv_r)
    D1_even_np = np.asarray(D1_even)
    D1_odd_np = np.asarray(D1_odd)
    yw_np = np.asarray(y_weights)
    kz_np = real_harmonics(nx) * (2 * pi / params.geo.lx)  # axial
    # Physical azimuthal wavenumbers m = m0 * harmonic over the wedge.
    m_np = params.geo.m0 * complex_harmonics(nz)

    decay = 1.0 - smoothness
    # Filter in x = r^2 (an axis-regular field is analytic in it), then
    # a wall window; the r^|m+s| envelopes below supply the axis
    # behaviour and the parity, for the velocity and the tensor alike.
    wn_filter = wall_normal_filter(rs_np**2, 1.0 - wall_smoothness)
    window_base = 1.0 - rs_np

    def fill_local(buf, m_start, n_m, kz_start, n_kz):
        for li in range(n_m):
            g2 = m_start + li
            m_val = int(m_np[g2])
            D1_v = D1_even_np if (m_val + 1) % 2 == 0 else D1_odd_np
            # Axis-regularity envelope per spin weight s (u_z and the
            # spin-0 tensor slots at s = 0, u_pm / c_z+- at s = +-1,
            # c_+-+- at s = +-2).
            env = {s: rs_np ** abs(m_val + s) for s in (0, 1, -1, 2, -2)}
            for lj in range(n_kz):
                g3 = kz_start + lj
                kz_val = kz_np[g3]
                envelope = decay ** (abs(kz_val) + abs(m_val))
                window_wall = scaled_wall_window(
                    window_base, np.hypot(kz_val, m_val), wall_confinement
                )
                window_wn = window_wall**2

                # ── Velocity (rows 0:3): the divergence-free pipe draw
                # of ``generate_cylindrical`` (same windows, envelope
                # and per-mode continuity closure). ──
                if g3 == 0:
                    col = hermitian_column(seed, g2, nz, Nr)
                else:
                    col = column_draw(seed, g2, g3, Nr)
                col = real_product(col, wn_filter.T)
                col[0] *= window_wall
                col[1] *= window_wn
                col[2] *= window_wn
                cp = (col[1] + 1j * col[2]) * env[1]
                cm = (col[1] - 1j * col[2]) * env[-1]
                col[1] = (cp + cm) / 2
                col[2] = (cp - cm) / 2j
                col[0] *= env[0]
                if kz_val != 0:
                    div_perp = (
                        real_product(col[1], D1_v.T)
                        + inv_r_np * col[1]
                        + 1j * m_val * inv_r_np * col[2]
                    )
                    col[0] = -div_perp / (1j * kz_val)
                elif m_val != 0:
                    col[2] = (
                        1j
                        * rs_np
                        * (real_product(col[1], D1_v.T) + inv_r_np * col[1])
                        / m_val
                    )
                else:
                    col[1] = 0.0
                col = normalize_mode(col, yw_np, envelope)
                # Mean mode: the (0, 0) conservation laws are only
                # established for the Cartesian flows, so every other
                # flow defers ``init.random_mean_flow``
                # (``dnsjax.ic.mean_mode``, and the per-flow
                # ``DeferredSpec``s).
                if g2 == 0 and g3 == 0:
                    col[:] = 0.0
                buf[0:3, :, li, lj] = col

                # ── Conformation (rows 3:9): windowed, wall-vanishing
                # noise, axis-enveloped in the spin basis; zero at the
                # mean mode (laminar added below). ──
                if g3 == 0:
                    ccol = hermitian_column(
                        seed, g2, nz, Nr, rows=6, stream=(1,)
                    )
                else:
                    ccol = column_draw(seed, g2, g3, Nr, rows=6, stream=(1,))
                ccol = real_product(ccol, wn_filter.T) * window_wn
                # Stored physical order (c_zz, c_rz, c_thz, c_rr,
                # c_thth, c_rth) -> spin combos (the definitions of
                # ``_viscoelastic_common.phys_combos_to_spin``, inlined
                # in NumPy for this host-side per-mode loop) ->
                # envelope by spin weight -> back.
                c_zz, c_rz, c_thz = ccol[0], ccol[1], ccol[2]
                c_rr, c_thth, c_rth = ccol[3], ccol[4], ccol[5]
                c_zp = (c_rz + 1j * c_thz) * env[1]
                c_zm = (c_rz - 1j * c_thz) * env[-1]
                c_pm = (c_rr + c_thth) * env[0]
                c_pp = ((c_rr - c_thth) + 2j * c_rth) * env[2]
                c_mm = ((c_rr - c_thth) - 2j * c_rth) * env[-2]
                d = (c_pp + c_mm) / 2  # = c_rr - c_theta_theta
                ccol = np.stack(
                    [
                        c_zz * env[0],
                        (c_zp + c_zm) / 2,
                        -0.5j * (c_zp - c_zm),
                        c_pm / 2 + d / 2,
                        c_pm / 2 - d / 2,
                        -0.5j * (c_pp - c_mm) / 2,
                    ]
                )
                ccol = normalize_mode(ccol, yw_np, envelope)
                if g2 == 0 and g3 == 0:
                    ccol[:] = 0.0
                buf[3:9, :, li, lj] = ccol

    state = assemble_local_shards(fill_local)

    # Rescale velocity and conformation noise separately.
    vel_norm2 = get_norm2_cyl(state[:3], fourier.k_metric, y_weights)
    conf_norm2 = get_norm2_conformation(state[3:], fourier.k_metric, y_weights)
    state = jnp.concatenate(
        [
            state[:3] * (amplitude / vel_norm2**0.5),
            state[3:] * (conf_amplitude / conf_norm2**0.5),
        ]
    )

    # Add the laminar pair at the mean mode (total-field IC).
    prof = viscoelastic_laminar_profiles(
        rs_np, D1_even_np, params.phys.wi, params.phys.epsilon
    )
    prof_jax = jnp.asarray(prof, dtype=state.dtype)
    laminar = jnp.where(
        fourier.mean_mask[None], prof_jax[:, :, None, None], 0.0
    )
    return state + laminar


def init_random_state(
    amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
    mean_flow: bool = False,
) -> Array:
    """The random IC: :func:`generate_viscoelastic_pipe`.

    Velocity and conformation noise on the laminar pair, the tensor's
    amplitude ``init.random_conformation_amplitude``.  *mean_flow*
    is deferred for this flow, so always false.
    """
    return generate_viscoelastic_pipe(
        amplitude,
        params.init.random_conformation_amplitude,
        smoothness,
        wall_smoothness,
        wall_confinement,
        seed,
    )


def init_localized_rolls(
    amplitude: float, width: float, wavelength: float
) -> Array:
    """The pipe's localized rolls on the laminar pair.

    :func:`dnsjax.ic.localized_rolls.generate_cylindrical_rolls` and
    :func:`add_viscoelastic_pipe_laminar`.
    """
    return add_viscoelastic_pipe_laminar(
        generate_cylindrical_rolls(amplitude, width, wavelength)
    )
