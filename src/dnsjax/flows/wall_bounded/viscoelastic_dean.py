r"""Viscoelastic (sPTT) Dean flow between two stationary cylinders.

Force-driven flow of an sPTT viscoelastic fluid in the annular gap
between two **stationary** concentric cylinders, driven by an azimuthal
body force `$\Pi_\theta = (r_1 + r_2)/(\mathrm{Re}\,r)$` (an external
reference normalisation, `$r_1 = \delta$`, `$r_2 = \delta + 2$`).  The
velocity is coupled to a symmetric conformation tensor `$\mathbf{c}$` via
the polymer-stress divergence; see the module docstring of
:mod:`~dnsjax.geometries.wall_bounded.annular_viscoelastic` for the
governing equations, the 9-component state layout, and the spin
diagonalisation of the tensor Laplacian.

Total-field formulation
-----------------------
Like Newtonian Dean flow, the **total** velocity is integrated (base
flow zero, so the rotational nonlinear term evaluates the full field),
with the azimuthal body force applied at the mean mode through
``pi_theta``.  The reported perturbation energy `$E'$` is velocity-only:
the kinetic energy of the deviation of `$\mathbf{u}$` from the analytical
laminar profile (the laminarization-check quantity).  ``get_stats`` also
reports polymer diagnostics (mean trace, elastic energy, polymer work).

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

with `$c_{rz} = c_{\theta z} = 0$`.  For `$\kappa = 0$` this is the
**exact** discrete steady state (the curvilinear advection / stretching
terms cancel algebraically at every `$\epsilon$`), so the full
9-component RHS conformation slice vanishes to machine precision; for
`$\kappa > 0$` it is an approximation (the diffusion of the
`$r$`-varying conformation is neglected).
"""

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ...geometries.wall_bounded._base import (
    extract_mean_mode,
    pad_base_flow,
)
from ...geometries.wall_bounded.annular import (
    get_enstrophy_annular,
    get_norm2_annular,
    integrate_scalar,
)
from ...geometries.wall_bounded.annular_viscoelastic import (
    Fourier,
    ViscoelasticAnnularFlow,
    _div_c,
    _spin_to_phys_combos,
    build_viscoelastic_stepper,
    fourier,
    get_norm2_conformation,  # noqa: F401 -- available for callers
    viscoelastic_laminar_profiles,
)
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class ViscoelasticDeanFlow(ViscoelasticAnnularFlow):
    r"""Precomputed data for viscoelastic (sPTT) Dean flow.

    Delegates the velocity grid / IMM operators (solvent viscosity
    `$\nu = \beta/\mathrm{Re}$`) and the conformation Helmholtz operator
    to :class:`ViscoelasticAnnularFlow`, then sets the azimuthal body
    force `$\Pi_\theta = (r_1 + r_2)/(\mathrm{Re}\,r)$` and zeros the base
    flow (total-field integration).
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        r1 = derived_params.r_inner
        r2 = derived_params.r_outer
        Re = params.phys.re

        # Azimuthal body force Pi_theta = (r1 + r2) / (Re r), applied at
        # the mean mode by ``annular_viscoelastic._get_rhs_core``.
        self.pi_theta = (r1 + r2) / (self.rs * Re)

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

    The analytical laminar `$r$`-profiles (velocity + sPTT-equilibrium
    conformation; see
    :func:`~dnsjax.geometries.wall_bounded.annular_viscoelastic.viscoelastic_laminar_profiles`)
    placed at the mean mode `$(m, k_z) = (0, 0)$`.
    """
    prof = viscoelastic_laminar_profiles(
        flow.rs,
        flow.D1,
        derived_params.r_inner,
        derived_params.r_outer,
        params.phys.wi,
        params.phys.epsilon,
    )
    prof_jax = jnp.asarray(prof, dtype=sharding.complex_type)
    return jnp.where(fourier.mean_mask[None], prof_jax[:, :, None, None], 0.0)


_laminar_state: Array = _build_laminar_state()


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
            "viscoelastic-dean does not support legacy .npz snapshots; "
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
    flow_: ViscoelasticDeanFlow,
) -> dict[str, Array]:
    r"""Total-field diagnostics + polymer quantities.

    - `$E$`: total kinetic energy; `$E'$`: velocity-only deviation from
      the laminar profile (laminarization quantity).
    - `$I = \langle u_\theta \Pi_\theta \rangle$`: body-force input.
    - `$D_s = \nu\langle|\nabla u|^2\rangle$`: solvent dissipation
      (`$\nu = \beta/\mathrm{Re}$`).
    - `$W_p = \tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}
      \langle u\cdot\nabla\cdot c\rangle$`: polymer work.
    - `$\langle\mathrm{tr}\,c\rangle$`, elastic energy
      `$E_p = \tfrac{1-\beta}{2\,\mathrm{Re}\,\mathrm{Wi}}
      (\langle\mathrm{tr}\,c\rangle - 3)$`.
    - wall shear stresses (solvent), bulk velocities.
    """
    Re = params.phys.re
    volfac = derived_params.volume_fac
    nu = derived_params.nu
    coef = (1.0 - params.phys.beta) / (Re * params.phys.wi)

    vel = state[:3]
    total_energy = (
        get_norm2_annular(vel, fourier_.k_metric, flow_.y_weights) / 2
    )
    perturbation_energy = (
        get_norm2_annular(
            vel - laminar_state[:3], fourier_.k_metric, flow_.y_weights
        )
        / 2
    )

    # Mean velocity profiles.
    mean_vel = extract_mean_mode(vel)  # (3, Nr)
    mean_uz = mean_vel[0].real
    mean_utheta = mean_vel[1].imag  # u_theta = Im(u_+)

    tau_theta = flow_.D1_bnd @ mean_utheta  # (2,) [inner, outer]
    tau_z = flow_.D1_bnd @ mean_uz
    U_bulk_theta = integrate_scalar(mean_utheta, flow_.y_weights) / volfac
    U_bulk_z = integrate_scalar(mean_uz, flow_.y_weights) / volfac

    energy_input = (
        integrate_scalar(mean_utheta * flow_.pi_theta, flow_.y_weights)
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

    # Polymer diagnostics.
    u_z, u_plus, u_minus = vel[0], vel[1], vel[2]
    u_r = (u_plus + u_minus) / 2
    u_th = -0.5j * (u_plus - u_minus)
    cs = _spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    div_r, div_th, div_z = _div_c(*cs, fourier_, flow_)
    # <u . div c> as a spectral inner product in (z, r, theta).
    u_rthz = jnp.array([u_z, u_r, u_th])
    div_rthz = jnp.array([div_z, div_r, div_th])
    polymer_work = coef * (
        integrate_scalar(
            jnp.sum(
                jnp.conj(u_rthz) * fourier_.k_metric * div_rthz, axis=(0, 2, 3)
            ).real,
            flow_.y_weights,
        )
        / volfac
    )

    mean_c = extract_mean_mode(state[3:])  # (6, Nr)
    trace_profile = (mean_c[3] + mean_c[0]).real  # c_+- + c_zz = tr c
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
    """Wrapper around ``_get_stats_jit``."""
    return _get_stats_jit(state, _laminar_state, fourier, flow)


@jit
def _get_perturbation_energy_jit(
    state: Array,
    laminar_state: Array,
    fourier_: Fourier,
    flow_: ViscoelasticDeanFlow,
) -> Array:
    r"""Velocity-only deviation energy `$E' = \|u -
    U_{\mathrm{lam}}\|^2/2$` (laminarization check)."""
    return (
        get_norm2_annular(
            state[:3] - laminar_state[:3],
            fourier_.k_metric,
            flow_.y_weights,
        )
        / 2
    )


def get_perturbation_energy(state: Array) -> Array:
    """Velocity-only perturbation energy E' (laminarization check)."""
    return _get_perturbation_energy_jit(state, _laminar_state, fourier, flow)
