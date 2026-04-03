"""Triply-periodic flows: Kolmogorov, Waleffe, and decaying-box.

This module defines the ``TriplyPeriodicFlow`` dataclass that holds all
precomputed, flow-specific data: base flow, forcing, time-stepping
coefficients (``ldt_1``, ``ildt_2``), and laminar-state diagnostics.

It also exports the full flow interface consumed by ``__main__``:

- ``predict_and_correct`` / ``iterate_correction`` -- time stepping
- ``get_stats`` -- diagnostic statistics
- ``correct_velocity`` -- divergence correction + mean-mode zeroing
- ``phys_to_spec`` -- forward 3D FFT (re-exported from operators)

Base flow construction
----------------------
The monochromatic base flow `$U(y)$` is defined analytically via a single
Fourier harmonic (`$q_f = 1$`):

- Kolmogorov: `$U = \\sin(2\\pi y/L_y)$` -- coefficient `-0.5j` at mode `$q_f$`
- Waleffe:    `$U = \\cos(2\\pi y/L_y)$` -- coefficient `+0.5` at mode `$q_f$`
- Decaying-box: `$U = 0$`

The base flow is transformed to physical space on the 3/2-oversampled
grid for use in the nonlinear term.  Its curl
(`$-\\partial U_x/\\partial y$` in the z-component) and the
self-interaction `$\\mathbf{U} \\times \\nabla \\times \\mathbf{U}$`
are precomputed once.

Tilt
----
When the forcing direction is tilted by an angle `$\\theta$` away from
the x-axis in the (x, z) plane, the base flow and its derivatives are
rotated: `$U_x \\to U_x \\cos\\theta$`, `$U_z \\to U_x \\sin\\theta$`.

Time-stepping coefficients
--------------------------
For the triply-periodic case the Helmholtz operator is diagonal in
Fourier space, so the implicit solve reduces to pointwise operations:

    `$ldt_1 = \\frac{1}{\\Delta t} + (1-c) \\frac{\\nabla^2}{\\mathrm{Re}}$`
    (explicit part)
    `$ildt_2 = \\left(
    \\frac{1}{\\Delta t} - c \\frac{\\nabla^2}{\\mathrm{Re}}
    \\right)^{-1}$`
    (inverse of implicit part)

The mean mode `$(k_y, k_z, k_x) = (0, 0, 0)$` is zeroed out, since it
is passive (constant shift) for periodic flows.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
from jax import Array, jit, vmap
from jax import numpy as jnp

from ..bench import timer
from ..geometries.triply_periodic import (
    curl,
    divergence,
    fourier,
    get_norm,
    get_norm2,
    gradient,
    inverse_laplacian,
    laplacian,
)
from ..operators import (
    phys_to_spec,
    spec_to_phys,
)
from ..parameters import (
    derived_params,
    monochromatic_systems,
    padded_res,
    params,
)
from ..rhs import get_nonlin
from ..sharding import register_dataclass_pytree, sharding
from ..timestep import make_stepper


@register_dataclass_pytree
@dataclass
class TriplyPeriodicFlow:
    """Precomputed data for triply-periodic flows.

    All class-level attributes are computed eagerly at definition time so
    that the module-level singleton ``flow = TriplyPeriodicFlow()`` is
    fully initialised at import.
    """

    _system: str = field(init=False)
    base_flow: Array = field(init=False)
    dy_base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    nonlin_base_flow: Array = field(init=False)

    qf: int = field(init=False)
    force_amplitude: Array = field(init=False)
    ekin_lam: float = field(init=False)
    input_lam: Array = field(init=False)
    dissip_lam: Array = field(init=False)

    forced_modes: tuple = field(init=False)
    unit_force: Array = field(init=False)

    ldt_1: Array = field(init=False)
    ildt_2: Array = field(init=False)

    def __post_init__(self) -> None:
        self._system = params.phys.system

        # Fourier coefficients of the streamwise base flow U_x(y).
        base_flow_complex = jnp.zeros(
            padded_res.ny_padded // 2 + 1, dtype=sharding.complex_type
        )
        if self._system in monochromatic_systems:
            self.qf = 1  # Forcing harmonic

            # Kolmogorov: sin(qf * 2pi y / Ly) -> -0.5j at +qf
            # Waleffe:    cos(qf * 2pi y / Ly) ->  0.5  at +qf
            if self._system == "kolmogorov":
                base_flow_complex = base_flow_complex.at[self.qf].add(-0.5j)
            elif self._system == "waleffe":
                base_flow_complex = base_flow_complex.at[self.qf].add(0.5)
            else:
                raise NotImplementedError

            # Forcing amplitude that sustains the laminar state:
            # `$F = \\nu k^2 U$`
            self.force_amplitude = jnp.pi**2 / (4 * params.phys.re)
            self.ekin_lam = 1.0 / 4.0
            self.input_lam = jnp.pi**2 / (8 * params.phys.re)
            self.dissip_lam = self.input_lam

        elif self._system == "decaying-box":
            self.ekin_lam = 0.0
            self.input_lam = 0.0
            self.dissip_lam = 0.0
        else:
            raise NotImplementedError

        # dU/dy in Fourier space: spectral derivative = i * ky * U_hat
        dy_base_flow_complex = (
            1j * (2 * jnp.pi / derived_params.ly) * base_flow_complex
        )

        base_flow = jnp.zeros(
            (3, padded_res.ny_padded),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )[:, :, None, None]
        dy_base_flow = jnp.zeros(
            (3, padded_res.ny_padded),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )[:, :, None, None]
        curl_base_flow = jnp.zeros(
            (3, padded_res.ny_padded),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )[:, :, None, None]
        nonlin_base_flow = jnp.zeros(
            (3, padded_res.ny_padded),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )[:, :, None, None]

        # Transform base flow from Fourier to physical (oversampled) space
        base_flow = base_flow.at[0].set(
            jnp.fft.irfft(
                base_flow_complex, n=padded_res.ny_padded, norm="forward"
            )[:, None, None]
        )
        dy_base_flow = dy_base_flow.at[0].set(
            jnp.fft.irfft(
                dy_base_flow_complex, n=padded_res.ny_padded, norm="forward"
            )[:, None, None]
        )
        # curl(U) = (0, 0, -dU_x/dy) for a unidirectional base flow
        curl_base_flow = curl_base_flow.at[2].set(-dy_base_flow[0])
        # U x curl(U) = (0, U_x * dU_x/dy, 0)
        nonlin_base_flow = nonlin_base_flow.at[1].set(
            base_flow[0] * dy_base_flow[0]
        )

        # Apply tilt: rotate (U_x, 0, 0) ->
        # (U_x cos(theta), 0, U_x sin(theta))
        tilt_rad: float = derived_params.tilt_rad
        if jnp.abs(tilt_rad) != 0:
            base_flow = base_flow.at[2].set(jnp.sin(tilt_rad) * base_flow[0])
            base_flow = base_flow.at[0].multiply(jnp.cos(tilt_rad))
            dy_base_flow = dy_base_flow.at[2].set(
                jnp.sin(tilt_rad) * dy_base_flow[0]
            )
            dy_base_flow = dy_base_flow.at[0].multiply(jnp.cos(tilt_rad))
            curl_base_flow = curl_base_flow.at[0].set(
                -jnp.sin(tilt_rad) * curl_base_flow[2]
            )
            curl_base_flow = curl_base_flow.at[2].multiply(jnp.cos(tilt_rad))

        self.base_flow = base_flow
        self.dy_base_flow = dy_base_flow
        self.curl_base_flow = curl_base_flow
        self.nonlin_base_flow = nonlin_base_flow

        # Forced modes
        if self._system in monochromatic_systems:
            if not derived_params.tilt:
                # No tilt: forcing only in u_x at +/- qf in ky
                self.forced_modes = (
                    (0, 0),
                    (self.qf, -self.qf),
                    (0, 0),
                    (0, 0),
                )
            elif derived_params.tilt_90:
                # 90-degree tilt: forcing only in u_z at +/- qf in ky
                self.forced_modes = (
                    (2, 2),
                    (self.qf, -self.qf),
                    (0, 0),
                    (0, 0),
                )
            else:
                # General tilt: forcing in both u_x and u_z
                self.forced_modes = (
                    (0, 0, 2, 2),
                    (self.qf, -self.qf, self.qf, -self.qf),
                    (0, 0, 0, 0),
                    (0, 0, 0, 0),
                )

            # Unit forcing Fourier coefficients
            if not derived_params.tilt or derived_params.tilt_90:
                if self._system == "kolmogorov":
                    self.unit_force = jnp.array(
                        [-0.5j, 0.5j], dtype=sharding.complex_type
                    )
                elif self._system == "waleffe":
                    self.unit_force = jnp.array(
                        [0.5, 0.5], dtype=sharding.complex_type
                    )
            else:
                if self._system == "kolmogorov":
                    self.unit_force = jnp.array(
                        [
                            -0.5j * jnp.cos(tilt_rad),
                            0.5j * jnp.cos(tilt_rad),
                            -0.5j * jnp.sin(tilt_rad),
                            0.5j * jnp.sin(tilt_rad),
                        ],
                        dtype=sharding.complex_type,
                    )
                elif self._system == "waleffe":
                    self.unit_force = jnp.array(
                        [
                            0.5 * jnp.cos(tilt_rad),
                            0.5 * jnp.cos(tilt_rad),
                            -0.5j * jnp.sin(tilt_rad),
                            0.5j * jnp.sin(tilt_rad),
                        ],
                        dtype=sharding.complex_type,
                    )

        # Time-stepping coefficients
        ldt_1 = (
            1 / params.step.dt
            + (1 - params.step.implicitness) * fourier.lapl / params.phys.re
        )
        ildt_2 = 1 / (
            1 / params.step.dt
            - params.step.implicitness * fourier.lapl / params.phys.re
        )

        # Zero the mean modes in timestepper matrices
        self.ldt_1 = ldt_1.at[sharding.scalar_mean_mode].set(
            0, out_sharding=sharding.spec_scalar_shard
        )
        self.ildt_2 = ildt_2.at[sharding.scalar_mean_mode].set(
            0, out_sharding=sharding.spec_scalar_shard
        )


flow: TriplyPeriodicFlow = TriplyPeriodicFlow()


# ── Initialization ────────────────────────────────────────────────────────


def init_state(snapshot: str | None) -> Array:
    """Initialise the flow state (velocity_spec)."""
    if params.init.start_from_laminar:
        return jnp.zeros(
            shape=(3, *sharding.spec_shape),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
    elif snapshot is not None:
        snapshot_arr = jnp.load(snapshot)["velocity_phys"].astype(
            sharding.float_type
        )
        velocity_phys = jax.device_put(
            snapshot_arr,
            sharding.phys_vector_shard,
        )
        velocity_phys = velocity_phys.at[...].subtract(flow.base_flow)
        return phys_to_spec(velocity_phys)
    else:
        sharding.print("Provide an initial condition.")
        sharding.exit(code=1)


# ── Algebraic Helmholtz operations (triply-periodic specific) ────────────


@partial(vmap, in_axes=(0, 0, None, None))
def _predict_component(
    state: Array,
    rhs_no_lapl: Array,
    ldt_1: Array,
    ildt_2: Array,
) -> Array:
    """Euler predictor step (vmapped over velocity components).

    Computes `$u_p = (u^n \\cdot ldt_1 + f^n) \\cdot ildt_2$`
    as a pointwise operation in spectral space, where the Helmholtz
    inversion is algebraic (multiply by ``ildt_2``).
    """
    return (state * ldt_1 + rhs_no_lapl) * ildt_2


@partial(vmap, in_axes=(0, 0, 0, None))
def _correct_component(
    prediction: Array,
    rhs_no_lapl_prev: Array,
    rhs_no_lapl_next: Array,
    ildt_2: Array,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector step (vmapped over velocity components).

    Computes the correction
    `$\\delta = c (f_{\\text{next}} - f_{\\text{prev}}) \\cdot ildt_2$`
    and returns the updated prediction and the correction itself (for
    convergence monitoring).
    """
    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ildt_2
    )
    return prediction + correction, correction


# ── Flow-specific callables for the stepper factory ──────────────────────


def _curl_fn(state: Array, fourier_: Any) -> Array:
    """Spectral curl with wavenumbers bound from ``fourier``."""
    return curl(state, fourier_.kx, fourier_.ky, fourier_.kz)


def _get_rhs(state: Array, fourier_: Any, flow_: Any) -> Array:
    """Divergence-free RHS: nonlinear term + algebraic pressure projection."""
    nonlin = get_nonlin(
        state,
        flow_.base_flow,
        flow_.curl_base_flow,
        flow_.nonlin_base_flow,
        spec_to_phys,
        phys_to_spec,
        lambda s: _curl_fn(s, fourier_),
    )
    # Pressure Poisson: `$\\nabla^2 p = \\nabla \\cdot \\mathbf{NL}$`
    lapl_pressure = divergence(nonlin, fourier_.kx, fourier_.ky, fourier_.kz)
    # Subtract pressure gradient to enforce incompressibility
    rhs_no_lapl = nonlin - gradient(
        inverse_laplacian(lapl_pressure, fourier_.inv_lapl),
        fourier_.kx,
        fourier_.ky,
        fourier_.kz,
    )
    return rhs_no_lapl


def _predict(
    state: Array, rhs_no_lapl: Array, fourier_: Any, flow_: Any
) -> Array:
    """Euler predictor with algebraic Helmholtz inversion."""
    return _predict_component(state, rhs_no_lapl, flow_.ldt_1, flow_.ildt_2)


def _correct(
    state_prev: Array,
    prediction: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Any,
    flow_: Any,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector with algebraic Helmholtz inversion."""
    return _correct_component(prediction, rhs_prev, rhs_next, flow_.ildt_2)


def _norm(correction: Array, fourier_: Any, flow_: Any) -> Array:
    """L2 convergence norm."""
    return get_norm(correction, fourier_.k_metric)


_predict_and_correct_jit, _iterate_correction_jit = make_stepper(
    _get_rhs, _predict, _correct, _norm
)


def predict_and_correct(state: Array) -> tuple[Array, Array, Array]:
    return _predict_and_correct_jit(state, fourier, flow)


def iterate_correction(
    state_prev: Array, prediction: Array, rhs_prev: Array
) -> tuple[Array, Array, Array]:
    return _iterate_correction_jit(
        state_prev, prediction, rhs_prev, fourier, flow
    )


# ── Diagnostic statistics ────────────────────────────────────────────────


def get_perturbation_energy(state: Array, fourier_: Any, flow_: Any) -> Array:
    """Perturbation kinetic energy `$E' = \\|\\mathbf{u}'\\|^2 / 2$`."""
    return get_norm2(state, fourier_.k_metric) / 2


def get_energy(
    perturbation_energy: Array,
    input: Array,
    fourier_: Any,
    flow_: Any,
) -> Array:
    """Total kinetic energy"""
    if params.phys.system in monochromatic_systems:
        return (
            perturbation_energy
            - flow_.ekin_lam
            + input / flow_.force_amplitude
        )
    else:
        return perturbation_energy


def get_enstrophy(
    state: Array, input: Array, fourier_: Any, flow_: Any
) -> Array:
    """Total enstrophy times Re"""
    return (
        jnp.sum(
            -laplacian(jnp.conj(state) * state, fourier_.lapl),
            dtype=sharding.float_type,
        )
        + 2 * input * params.phys.re
        - flow_.input_lam * params.phys.re
    )


def get_dissipation(
    state: Array, input: Array, fourier_: Any, flow_: Any
) -> Array:
    """Total dissipation rate `$D = \\text{enstrophy} / \\mathrm{Re}$`."""
    return get_enstrophy(state, input, fourier_, flow_) / params.phys.re


def get_input(state: Array, fourier_: Any, flow_: Any) -> Array:
    """Power input from the forcing"""
    return (
        jnp.sum(
            jnp.conj(flow_.unit_force * flow_.force_amplitude)
            * state.at[flow_.forced_modes].get(out_sharding=sharding.no_shard),
            dtype=sharding.float_type,
        )
        + flow_.input_lam
    )


@jit
def _get_stats_jit(
    state: Array, fourier_: Any, flow_: Any
) -> dict[str, Array]:
    """Compute diagnostic statistics: E, I, D, E'."""
    perturbation_energy = get_perturbation_energy(state, fourier_, flow_)
    input = get_input(state, fourier_, flow_)
    dissipation = get_dissipation(state, input, fourier_, flow_)
    energy = get_energy(perturbation_energy, input, fourier_, flow_)

    stats = {
        "E": energy,
        "I": input,
        "D": dissipation,
        "E'": perturbation_energy,
    }

    return stats


@timer("stats")
def get_stats(state: Array) -> dict[str, Array]:
    return _get_stats_jit(state, fourier, flow)


# ── Divergence correction ────────────────────────────────────────────────


def correct_divergence(
    state: Array, fourier_: Any, flow_: Any
) -> tuple[Array, Array | None]:
    """Project the velocity onto the divergence-free subspace."""
    correction = -gradient(
        inverse_laplacian(
            divergence(
                state,
                fourier_.kx,
                fourier_.ky,
                fourier_.kz,
            ),
            fourier_.inv_lapl,
        ),
        fourier_.kx,
        fourier_.ky,
        fourier_.kz,
    )

    error = (
        get_norm(correction, fourier_.k_metric)
        if params.debug.measure_corrections
        else None
    )

    velocity_corrected = state + correction
    return velocity_corrected, error


@jit(donate_argnums=0)
def _correct_velocity_jit(
    state: Array, fourier_: Any, flow_: Any
) -> tuple[Array, dict[str, Array | None] | None]:
    norm_corrections = {}
    velocity_corrected = state

    if params.debug.correct_divergence:
        velocity_corrected, error = correct_divergence(
            velocity_corrected, fourier_, flow_
        )
        norm_corrections["div"] = error

    velocity_corrected = velocity_corrected.at[sharding.vector_mean_mode].set(
        0, out_sharding=sharding.spec_vector_shard
    )

    if not params.debug.measure_corrections:
        norm_corrections = None

    return velocity_corrected, norm_corrections


@timer("velocity/correct_velocity")
def correct_velocity(
    state: Array,
) -> tuple[Array, dict[str, Array | None] | None]:
    return _correct_velocity_jit(state, fourier, flow)
