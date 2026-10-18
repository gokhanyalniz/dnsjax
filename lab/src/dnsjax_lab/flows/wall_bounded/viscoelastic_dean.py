r"""Viscoelastic (sPTT) Dean flow between two stationary cylinders.

Force-driven flow of an sPTT viscoelastic fluid in the annular gap
between two **stationary** concentric cylinders, driven by an azimuthal
body force `$-\Pi_\theta = (r_1 + r_2)/(\mathrm{Re}\,r)$` (half-gap
units, `$r_1 = \delta$`, `$r_2 = \delta + 2$`; at mid-gap the force is
the `$2/\mathrm{Re}$` that drives the plane channel of the sPTT system
this flow integrates).  The velocity is coupled to a symmetric
conformation tensor `$\mathbf{c}$` via the polymer-stress divergence:
the governing equations, the 9-component state layout and the spin
diagonalization of the tensor Laplacian in
:mod:`~dnsjax_lab.geometries.wall_bounded._viscoelastic_common`, the
annular half of the geometry in
:mod:`~dnsjax_lab.geometries.wall_bounded.annular_viscoelastic`.

Total-field formulation
-----------------------
As in Newtonian Dean flow and
:mod:`~dnsjax_lab.flows.wall_bounded.viscoelastic_pipe` (whose section
also defines the reported `$E'$` and polymer diagnostics), the
**total** velocity is integrated, here with the azimuthal body force
applied at the mean mode through ``force_theta``.

Laminar state
-------------
``start_from_laminar`` uses the analytical laminar pair: the azimuthal
velocity profile `$U_\theta(r)$` (solving the viscous + body-force
balance, `$C = r_1 + r_2$`) and the pointwise sPTT-equilibrium
conformation on the **discrete** local shear
`$S = D_1 U_\theta - U_\theta/r$`,

.. math::
    c_{rr} = c_{zz} = 1, \quad c_{r\theta} = \frac{\mathrm{Wi}\,S}{f},
    \quad c_{\theta\theta} = 1 + \frac{2(\mathrm{Wi}\,S)^2}{f^2},
    \quad f^3 - f^2 = 2\epsilon(\mathrm{Wi}\,S)^2,

with `$c_{rz} = c_{\theta z} = 0$`.  For `$\kappa = 0$` the
conformation slice of the full 9-component RHS vanishes to machine
precision at every `$\epsilon$` (the curvilinear advection and
stretching terms cancel algebraically); for `$\kappa > 0$` it is an
approximation twice over -- the diffusion of the `$r$`-varying
conformation is neglected, and the equilibrium profile does not satisfy
the `$\nabla^2 c = 0$` wall rows `$H_c$` imposes.

**The velocity slice closes exactly only at** `$\epsilon = 0$`, where
`$f \equiv 1$` and the polymer contributes a constant extra viscosity, so the
Newtonian `$U_\theta$` is also the sPTT one.  At `$\epsilon > 0$` the polymer
shear-thins (the true balance is
`$(1/r^2)\,\partial_r[r^2\tau] - \Pi_\theta = 0$` with
`$\tau = (S/\mathrm{Re})[\beta + (1-\beta)/f]$`) and that correction is
neglected, as in ``viscoelastic-pipe``, so the pair is **not** a steady state
there, and the defect is not small at the shipped defaults (Design notes: "The
laminar defect, measured").  Consequences: ``start_from_laminar`` is not
laminar, and `$E'$` -- which is measured against *this* reference, hence the
``stop.check_laminarization`` quantity -- has a floor of a few percent of `$E$`
instead of decaying to zero.  Fixing it needs a shooting solve for `$U_\theta$`
(invert `$\tau(S)$` through the cubic, integrate, match both no-slip walls),
which is why the closed form is kept and the defect documented instead.

Design notes
------------
**The laminar defect, measured.**  Stepping the laminar pair at
`$\epsilon = 10^{-3}$` drifts at `$\max|\Delta u|/\Delta t \approx 0.3$`,
against `$2\!\cdot\!10^{-13}$` at `$\epsilon = 0$`, and the laminar
ledger `$I = D_s - W_p$` misses by 15 % (``viscoelastic-pipe``: 8 %).
"""

from dataclasses import dataclass
from math import pi

import numpy as np
from jax import Array, jit
from jax import numpy as jnp

from dnsjax.geometries.wall_bounded._base import (
    extract_mean_mode,
    pad_base_flow,
)
from dnsjax.geometries.wall_bounded.annular import (
    DRIVING_KEY_Z,
    get_enstrophy_annular,
    get_norm2_annular,
    integrate_scalar,
)
from dnsjax.harmonics import complex_harmonics, real_harmonics
from dnsjax.ic.localized_rolls import generate_annular_rolls
from dnsjax.ic.random_field import (
    column_draw,
    hermitian_column,
    normalize_mode,
    real_product,
    scaled_wall_window,
    wall_normal_filter,
)
from dnsjax.parameters import derived_params, params
from dnsjax.sharding import register_dataclass_pytree, sharding
from dnsjax_lab.geometries.wall_bounded.annular_viscoelastic import (
    Fourier,
    ViscoelasticAnnularFlow,
    _div_c,
    build_viscoelastic_stepper,
    fourier,
    from_solver_basis,  # noqa: F401 — re-exported (basis boundary)
    get_norm2_conformation,
    to_solver_basis,  # noqa: F401 — re-exported (basis boundary)
    viscoelastic_laminar_profiles,
)


@register_dataclass_pytree
@dataclass
class ViscoelasticDeanFlow(ViscoelasticAnnularFlow):
    r"""Precomputed data for viscoelastic (sPTT) Dean flow.

    Delegates the velocity grid / IMM operators (solvent viscosity
    `$\nu = \beta/\mathrm{Re}$`) and the conformation Helmholtz operator
    to :class:`ViscoelasticAnnularFlow`, then sets the azimuthal body
    force `$-\Pi_\theta = (r_1 + r_2)/(\mathrm{Re}\,r)$` and zeros the base
    flow (total-field integration).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        r1 = derived_params.r_inner
        r2 = derived_params.r_outer
        Re = params.phys.re

        # Azimuthal body force -Pi_theta = (r1 + r2) / (Re r), applied
        # at the mean mode by
        # ``ViscoelasticAnnularFlow.add_mean_body_force`` (the shared
        # RHS's driving adapter).
        self.force_theta = (r1 + r2) / (self.rs * Re)

        # Total-field formulation: no base flow to subtract.
        self.base_flow = jnp.zeros(
            (3, params.res.ny),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )[:, :, None, None]
        self.curl_base_flow = jnp.zeros_like(self.base_flow)
        pad_base_flow(self)


flow: ViscoelasticDeanFlow = ViscoelasticDeanFlow()

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

    The analytical laminar `$r$`-profiles (velocity + sPTT-equilibrium
    conformation; see
    :func:`~dnsjax_lab.geometries.wall_bounded.annular_viscoelastic.viscoelastic_laminar_profiles`)
    -- the only nonzero column of the laminar state.
    """
    prof = viscoelastic_laminar_profiles(
        flow.rs,
        flow.D1.dense,
        derived_params.r_inner,
        derived_params.r_outer,
        params.phys.wi,
        params.phys.epsilon,
    )
    return sharding.distribute(
        np.asarray(prof, dtype=sharding.complex_type), sharding.no_shard
    )


#: The laminar reference as its mean-mode column, in the physical
#: basis (the comment on ``dean._laminar_profile`` says why).
_laminar_profile: Array = _build_laminar_profile()


def _laminar_field(profile: Array, fourier_: Fourier) -> Array:
    """*profile* ``(C, N_r)`` on the mean mode; zero elsewhere, padding too."""
    return jnp.where(fourier_.mean_mask[None], profile[:, :, None, None], 0.0)


def _perturbation_energy(
    state: Array,
    laminar_profile: Array,
    fourier_: Fourier,
    flow_: ViscoelasticDeanFlow,
) -> Array:
    r"""Velocity-only kinetic energy of the deviation from laminar.

    `$E' = \|u - U_{\mathrm{lam}}\|^2/2$` over the three velocity components.
    The single definition, shared by :func:`get_stats` (which reports it as
    ``E'``) and the laminarization read :func:`get_perturbation_energy`.
    """
    return (
        get_norm2_annular(
            state[:3] - _laminar_field(laminar_profile[:3], fourier_),
            fourier_.k_metric,
            flow_.y_weights,
        )
        / 2
    )


def init_state() -> Array:
    """The ``start_from_laminar`` 9-component state.

    The analytical laminar velocity/conformation pair (this flow
    integrates the **total** field), called for the laminar start only
    (:func:`~dnsjax.geometries.wall_bounded._base.init_state`).
    """
    return _laminar_field(_laminar_profile, fourier)


# ── Diagnostic statistics ────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array,
    laminar_profile: Array,
    fourier_: Fourier,
    flow_: ViscoelasticDeanFlow,
) -> dict[str, Array]:
    r"""Total-field diagnostics + polymer quantities.

    *state* and *laminar_profile* are the **physical** 9-component view
    -- diagnostics sit outside the solver, whose working basis is the
    `$u_\pm$`/spin one (the ``_viscoelastic_common.py`` module
    docstring).  The columns of the pipe's ``_get_stats_jit``
    (:mod:`~dnsjax_lab.flows.wall_bounded.viscoelastic_pipe`), with the
    body-force input `$I = -\langle u_\theta \Pi_\theta \rangle$` and
    the solvent wall shear stresses at both walls.
    """
    Re = params.phys.re
    volfac = derived_params.volume_fac
    nu = derived_params.nu
    coef = (1.0 - params.phys.beta) / (Re * params.phys.wi)

    vel = state[:3]
    total_energy = (
        get_norm2_annular(vel, fourier_.k_metric, flow_.y_weights) / 2
    )
    perturbation_energy = _perturbation_energy(
        state, laminar_profile, fourier_, flow_
    )

    # Mean velocity profiles.
    mean_vel = extract_mean_mode(vel)  # (3, Nr)
    mean_uz = mean_vel[0].real
    mean_utheta = mean_vel[2].real

    tau_theta = flow_.D1_bnd @ mean_utheta  # (2,) [inner, outer]
    tau_z = flow_.D1_bnd @ mean_uz
    U_bulk_theta = integrate_scalar(mean_utheta, flow_.y_weights) / volfac
    U_bulk_z = integrate_scalar(mean_uz, flow_.y_weights) / volfac

    energy_input = (
        integrate_scalar(mean_utheta * flow_.force_theta, flow_.y_weights)
        / volfac
    )
    enstrophy = get_enstrophy_annular(
        vel,
        flow_.D1,
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
        "tau_th,i": nu * tau_theta[0],
        "tau_th,o": nu * tau_theta[1],
        "tau_z,i": nu * tau_z[0],
        "tau_z,o": nu * tau_z[1],
        "Ub_th": U_bulk_theta,
        "Ub_z": U_bulk_z,
    }


def get_stats(state: Array) -> dict[str, Array]:
    """The ``stats.dat`` columns of the physical-basis *state*."""
    return _get_stats_jit(state, _laminar_profile, fourier, flow)


@jit
def _get_perturbation_energy_jit(
    state: Array,
    laminar_profile: Array,
    fourier_: Fourier,
    flow_: ViscoelasticDeanFlow,
) -> Array:
    r"""Velocity-only deviation energy (laminarization check)."""
    return _perturbation_energy(state, laminar_profile, fourier_, flow_)


def get_perturbation_energy(state: Array) -> Array:
    r"""Velocity-only perturbation energy E' (laminarization check).

    Takes the **physical** 9-component view, like :func:`get_stats`,
    which reports the same number as ``E'``.
    """
    return _get_perturbation_energy_jit(state, _laminar_profile, fourier, flow)


def get_driving(state: Array) -> dict[str, Array]:
    r"""Applied mean-mode driving: **keys only**, value zero.

    The wall-shear inference that fills the ``t = t0`` row
    (:class:`~dnsjax.flow_spec.FlowSpec`) is Newtonian only
    (:func:`~dnsjax.geometries.wall_bounded.annular.mean_driving`): the
    polymer carries its own axial wall traction, so the Newtonian
    number (viscosity `$1/\mathrm{Re}$`) would be wrong rather than
    merely approximate.  The zero
    placeholder keeps the column present and correctly shaped from the
    first row; every later row carries the value the corrector applied,
    which is exact.  Empty unless ``phys.block_mean_spanwise_velocity``
    is on.
    """
    if not params.phys.block_mean_spanwise_velocity:
        return {}
    return {DRIVING_KEY_Z: jnp.zeros((), dtype=sharding.float_type)}


# ── Initial conditions ───────────────────────────────────────────


def add_viscoelastic_laminar(vel_state: Array) -> Array:
    r"""Turn a velocity perturbation into a viscoelastic Dean total state.

    The annular counterpart of the pipe's
    ``add_viscoelastic_pipe_laminar``
    (:mod:`~dnsjax_lab.flows.wall_bounded.viscoelastic_pipe`), on this
    flow's laminar pair (:func:`init_state`'s).
    """
    laminar = init_state()
    return jnp.concatenate([vel_state + laminar[:3], laminar[3:]])


# Viscoelastic conformation noise: drawn as in the viscoelastic pipe
# (the comment above ``viscoelastic_pipe.generate_viscoelastic_pipe``).


def generate_viscoelastic_dean(
    amplitude: float,
    conf_amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
) -> Array:
    r"""Random 9-component IC for viscoelastic (sPTT) Dean flow.

    Built as the pipe's ``generate_viscoelastic_pipe``
    (:mod:`~dnsjax_lab.flows.wall_bounded.viscoelastic_pipe`), from the
    divergence-free annular draw of :func:`generate_annular` (rows
    ``0:3``) and the annular laminar pair, with conformation noise that
    vanishes at both walls and needs no axis envelope.
    """
    from jax import numpy as jnp

    from dnsjax.geometries.wall_bounded.annular import (
        build_annular_grid,
        fourier,
        get_norm2_annular,
    )
    from dnsjax.snapshot import assemble_local_shards

    nx = params.res.nx
    Nr = params.res.ny
    nz = params.res.nz
    r1 = derived_params.r_inner
    r2 = derived_params.r_outer
    rs, D1, _, y_weights, inv_r = build_annular_grid(
        Nr,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
        r_inner=r1,
        r_outer=r2,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    rs_np = np.asarray(rs)
    inv_r_np = np.asarray(inv_r)
    D1_np = np.asarray(D1)
    yw_np = np.asarray(y_weights)
    kz_np = real_harmonics(nx) * (2 * pi / params.geo.lx)  # axial
    # Physical azimuthal wavenumbers m = m0 * harmonic over the wedge,
    # as in ``generate_annular`` (which says why).
    m_np = params.geo.m0 * complex_harmonics(nz)

    decay = 1.0 - smoothness
    wn_filter = wall_normal_filter(rs_np, 1.0 - wall_smoothness)
    window_base = (rs_np - r1) * (r2 - rs_np)
    window_base = window_base / np.max(window_base)

    def fill_local(buf, m_start, n_m, kz_start, n_kz):
        for li in range(n_m):
            g2 = m_start + li
            m_val = int(m_np[g2])
            for lj in range(n_kz):
                g3 = kz_start + lj
                if g2 == 0 and g3 == 0:
                    # Mean mode: zero in both fields, as in
                    # ``generate_viscoelastic_pipe``.
                    continue
                kz_val = kz_np[g3]
                envelope = decay ** (abs(kz_val) + abs(m_val))
                window_lin = scaled_wall_window(
                    window_base, np.hypot(kz_val, m_val), wall_confinement
                )
                window_wn = window_lin**2

                # Velocity (rows 0:3): divergence-free draw.
                if g3 == 0:
                    vcol = hermitian_column(seed, g2, nz, Nr)
                else:
                    vcol = column_draw(seed, g2, g3, Nr)
                vcol = real_product(vcol, wn_filter.T)
                vcol[0] *= window_lin
                vcol[1] *= window_wn
                vcol[2] *= window_wn
                # Close continuity per mode against the annular
                # divergence, exactly as :func:`generate_annular` does
                # (same operator, same r-diagonal coefficients).
                if kz_val != 0:
                    div_perp = (
                        real_product(vcol[1], D1_np.T)
                        + inv_r_np * vcol[1]
                        + 1j * m_val * inv_r_np * vcol[2]
                    )
                    vcol[0] = -div_perp / (1j * kz_val)
                else:
                    # k_z = 0 (so m != 0): u_z drops out; close through
                    # u_theta.
                    d1_v = real_product(vcol[1], D1_np.T)
                    vcol[2] = 1j * rs_np * (d1_v + inv_r_np * vcol[1]) / m_val
                vcol = normalize_mode(vcol, yw_np, envelope)
                buf[0:3, :, li, lj] = vcol

                # Conformation (rows 3:9): windowed, wall-vanishing noise.
                if g3 == 0:
                    ccol = hermitian_column(
                        seed, g2, nz, Nr, rows=6, stream=(1,)
                    )
                else:
                    ccol = column_draw(seed, g2, g3, Nr, rows=6, stream=(1,))
                ccol = real_product(ccol, wn_filter.T) * window_wn
                ccol = normalize_mode(ccol, yw_np, envelope)
                buf[3:9, :, li, lj] = ccol

    state = assemble_local_shards(fill_local)

    # Rescale velocity and conformation noise separately.
    vel_norm2 = get_norm2_annular(state[:3], fourier.k_metric, y_weights)
    conf_norm2 = get_norm2_conformation(state[3:], fourier.k_metric, y_weights)
    scale_v = amplitude / vel_norm2**0.5
    scale_c = conf_amplitude / conf_norm2**0.5
    state = jnp.concatenate([state[:3] * scale_v, state[3:] * scale_c])

    # Add the laminar pair at the mean mode (total-field IC).
    return state + init_state()


def init_random_state(
    amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
    mean_flow: bool = False,
) -> Array:
    """The random IC: :func:`generate_viscoelastic_dean`.

    Velocity and conformation noise on the laminar pair, the tensor's
    amplitude ``init.random_conformation_amplitude``.  *mean_flow*
    is deferred for this flow, so always false.
    """
    return generate_viscoelastic_dean(
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
    """The annular localized rolls on the laminar pair.

    :func:`dnsjax.ic.localized_rolls.generate_annular_rolls` and
    :func:`add_viscoelastic_laminar`.
    """
    return add_viscoelastic_laminar(
        generate_annular_rolls(amplitude, width, wavelength)
    )
