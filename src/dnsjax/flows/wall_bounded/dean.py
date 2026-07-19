r"""Dean flow: pressure-driven flow between two stationary cylinders.

A "simplified" Dean flow in the annular geometry: flow in the gap between
two **stationary** concentric cylinders, driven by an azimuthally- and
axially-uniform, radius-dependent **azimuthal body force** (a mean
azimuthal pressure gradient)

.. math::
    \vec{\Pi} = \frac{2\eta + 2}{r\,\mathrm{Re}\,(1 - \eta)}\,
    \hat{\boldsymbol{\theta}}, \qquad \eta = r_1/r_2,

with no-slip walls (all velocity components vanish at `$r_1$` and
`$r_2$`).  Geometry-general infrastructure (radial grid on
`$[r_1, r_2]$`, FD matrices, IMM operators, the `$2 \times 2$` annular
IMM iteration, predict / correct / norm, Pallas / dense solvers) is
inherited from :class:`~dnsjax.geometries.wall_bounded.annular.AnnularFlow`.

Total-field formulation
-----------------------
Unlike every other flow in the solver, Dean flow time-integrates the
**total** velocity field, not a perturbation around a base flow.  This is
realised with **no special stepper**: the flow sets
``base_flow = curl_base_flow = 0`` so the rotational-form nonlinear term
(:func:`dnsjax.rhs.get_nonlin`) evaluates the full
`$(\nabla\times\mathbf{u})\times\mathbf{u}$` of the total field, and the
azimuthal body force is supplied through ``AnnularFlow.pi_theta``
(applied at the mean mode inside ``annular._get_rhs_core``).  Because
there is no base flow, the reported perturbation kinetic energy `$E'$`
is the energy of the **deviation** from the analytical laminar Dean
profile, `$\|\mathbf{u} - \mathbf{U}_{\mathrm{lam}}\|^2 / 2$`;
``get_stats`` also reports total quantities (`$E$`, `$I$`, `$D$`, and
the wall stresses).

Like the (shear-driven) Taylor-Couette flow, the "streamwise" direction
is azimuthal; the optional ``block_mean_spanwise_velocity`` zeroes the
mean **axial** velocity (the undriven homogeneous direction), inherited
unchanged from ``AnnularFlow``.

Laminar state
-------------
``start_from_laminar`` uses the closed-form, Reynolds-independent laminar
profile `$U_\theta(r) = -(C/2) r\ln r + \alpha r + \beta/r$` (see
:func:`~dnsjax.geometries.wall_bounded.annular.dean_laminar_u_theta`).
It is the steady solution of the *continuous* equations; on the FD grid
it is preserved up to truncation (the mean-mode centripetal term is
balanced by pressure and `$u_\theta$` by the viscous + forcing balance).

Exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``predict_and_fully_correct_measured`` -- fused step + CFL
  measurements (``steps.dat``)
- ``init_state`` -- initial state from the laminar profile or snapshot
- ``get_stats`` -- diagnostic statistics
"""

from dataclasses import dataclass

from jax import Array, jit, lax
from jax import numpy as jnp

from ...geometries.wall_bounded.annular import (
    AnnularFlow,
    Fourier,
    build_annular_stepper,
    dean_laminar_u_theta,
    extract_mean_mode,
    fourier,
    get_enstrophy_annular,
    get_norm2_annular,
    integrate_scalar,
    pad_base_flow,
)
from ...geometries.wall_bounded.annular import (
    init_state as _base_init_state,
)
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class DeanFlow(AnnularFlow):
    r"""Precomputed data for Dean flow (force-driven, total-field).

    Delegates the radial grid, FD matrices, and per-mode IMM operators to
    :meth:`AnnularFlow.__post_init__`, then sets the azimuthal body force
    `$\Pi_\theta = (2\eta + 2)/(r\,\mathrm{Re}\,(1-\eta))$` on
    ``pi_theta`` and zeros the base flow (the **total** velocity is
    integrated).
    """

    def __post_init__(self) -> None:
        r"""Build grid / operators, then set the azimuthal body force."""
        super().__post_init__()

        eta = params.geo.eta
        Re = params.phys.re

        # Azimuthal body force Pi_theta = (2 eta + 2) / (r Re (1 - eta)),
        # applied at the mean mode by ``annular._get_rhs_core``.
        C = 2.0 * (eta + 1.0) / (1.0 - eta)
        self.pi_theta = C / (self.rs * Re)

        # Total-field formulation: there is no base flow to subtract, so
        # the rotational term computes the full (curl u) x u of the total
        # field.
        self.base_flow = jnp.zeros(
            (3, params.res.ny),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )[:, :, None, None]
        self.curl_base_flow = jnp.zeros_like(self.base_flow)
        pad_base_flow(self)


flow: DeanFlow = DeanFlow()

(
    predict_and_correct,
    iterate_correction,
    _init_state_laminar_zero,  # overridden below (Dean laminar != 0)
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_annular_stepper(flow)


def _build_laminar_state() -> Array:
    r"""Spectral laminar Dean state: `$U_\theta(r)$` at the mean mode.

    Places the analytical azimuthal profile at the mean mode in
    `$(u_z, u_+, u_-)$` form (`$u_\pm = \pm i\,U_\theta$`, `$u_z = 0$`).
    """
    u_theta = dean_laminar_u_theta(flow.rs, params.geo.eta)  # (Nr,) real
    # Broadcast onto the mean mode (m, k_z) = (0, 0) only.
    u_spec = jnp.where(fourier.mean_mask, u_theta[:, None, None], 0.0)
    zeros = jnp.zeros_like(u_spec)
    u_z = lax.complex(zeros, zeros)
    u_plus = lax.complex(zeros, u_spec)  # i * U_theta
    u_minus = lax.complex(zeros, -u_spec)  # -i * U_theta
    return jnp.stack([u_z, u_plus, u_minus])


_laminar_state: Array = _build_laminar_state()


def init_state(snapshot: str | None) -> Array:
    """Initialise the flow state (the total velocity).

    A provided snapshot path (legacy ``.npz``) takes precedence: it is
    delegated to the base ``init_state``.  Otherwise, for
    ``start_from_laminar``, returns the closed-form laminar Dean profile
    (a nonzero total field).  zarr3 snapshot resume is handled in
    ``__main__`` before this is called.
    """
    if snapshot is None and params.init.start_from_laminar:
        # Copy: the steppers donate their state argument, and the
        # module-level ``_laminar_state`` must survive for the E'
        # deviation in ``get_stats`` / ``get_perturbation_energy``.
        return jnp.copy(_laminar_state)
    return _base_init_state(snapshot)


# ── Diagnostic statistics ────────────────────────────────────────


@jit
def _get_stats_jit(
    state: Array,
    laminar_state: Array,
    fourier_: Fourier,
    flow_: DeanFlow,
) -> dict[str, Array]:
    r"""Compute diagnostic statistics (total-field quantities).

    - `$E$`: total kinetic energy (annular norm with radial Jacobian
      `$r$` and `$u_\pm$` half-factor).
    - `$E'$`: perturbation kinetic energy of the deviation from the
      laminar Dean profile,
      `$\|\mathbf{u} - \mathbf{U}_{\mathrm{lam}}\|^2 / 2$` (near zero
      at ``start_from_laminar``; the laminar-smoke error metric and the
      laminarization-check quantity).
    - `$I$`: energy input rate from the body force,
      `$I = \langle u_\theta\,\Pi_\theta \rangle$` (mean-mode only, as
      `$\Pi_\theta$` is azimuthally/axially uniform).
    - `$D$`: total dissipation rate `$\langle |\nabla\mathbf{u}|^2
      \rangle / \mathrm{Re}$`.  At a steady state `$I = D$`.
    - `$\tau_{\theta,i/o}$`, `$\tau_{z,i/o}$`: wall shear stresses at the
      inner / outer walls.
    - `$U_{b,\theta}$`, `$U_{b,z}$`: bulk azimuthal and axial velocities.
    """
    Re = params.phys.re
    volfac = derived_params.volume_fac

    total_energy = (
        get_norm2_annular(state, fourier_.k_metric, flow_.y_weights) / 2
    )
    perturbation_energy = (
        get_norm2_annular(
            state - laminar_state, fourier_.k_metric, flow_.y_weights
        )
        / 2
    )

    # ── Mean velocity profiles ───────────────────────────────
    mean_state = extract_mean_mode(state)  # (3, Nr)
    mean_uz = mean_state[0].real  # (Nr,)
    mean_utheta = mean_state[1].imag  # u_theta = Im(u_+), (Nr,)

    # ── Wall shear & bulk velocity ──────────────────────────
    tau_theta = flow_.D1_bnd @ mean_utheta  # (2,) [inner, outer]
    tau_z = flow_.D1_bnd @ mean_uz  # (2,)
    U_bulk_theta = integrate_scalar(mean_utheta, flow_.y_weights) / volfac
    U_bulk_z = integrate_scalar(mean_uz, flow_.y_weights) / volfac

    # ── Energy input rate I = <u_theta Pi_theta> ─────────────
    energy_input = (
        integrate_scalar(mean_utheta * flow_.pi_theta, flow_.y_weights)
        / volfac
    )

    # ── Dissipation D = <|grad u|^2> / Re ───────────────────
    enstrophy = get_enstrophy_annular(
        state,
        flow_.D1,
        flow_.inv_r,
        fourier_.m,
        fourier_.kz2,
        fourier_.k_metric,
        flow_.y_weights,
    )
    dissipation = enstrophy / Re

    return {
        "E": total_energy,
        "E'": perturbation_energy,
        "I": energy_input,
        "D": dissipation,
        "tau_th,i": tau_theta[0] / Re,
        "tau_th,o": tau_theta[1] / Re,
        "tau_z,i": tau_z[0] / Re,
        "tau_z,o": tau_z[1] / Re,
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
    flow_: DeanFlow,
) -> Array:
    r"""Perturbation kinetic energy of the deviation from laminar.

    `$E' = \|\mathbf{u} - \mathbf{U}_{\mathrm{lam}}\|^2 / 2$` (Dean
    integrates the total field, so the perturbation is the deviation
    from the analytical laminar Dean profile).
    """
    return (
        get_norm2_annular(
            state - laminar_state, fourier_.k_metric, flow_.y_weights
        )
        / 2
    )


def get_perturbation_energy(state: Array) -> Array:
    """Perturbation kinetic energy E' (for the laminarization check)."""
    return _get_perturbation_energy_jit(state, _laminar_state, fourier, flow)
