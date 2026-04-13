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
The monochromatic base flow ``U(y)`` is defined analytically via a single
Fourier harmonic (``qf = 1``):

- Kolmogorov: ``U = sin(2*pi*y/Ly)`` -- coefficient ``-0.5j`` at mode ``qf``
- Waleffe:    ``U = cos(2*pi*y/Ly)`` -- coefficient ``+0.5`` at mode ``qf``
- Decaying-box: ``U = 0``

The base flow is transformed to physical space on the 3/2-oversampled
grid for use in the nonlinear term.  Its curl (``-dU/dy`` in the z-component)
and the self-interaction ``U x curl(U)`` are precomputed once.

Tilt
----
When the forcing direction is tilted by an angle ``theta`` away from the
x-axis in the (x, z) plane, the base flow and its derivatives are rotated:
``U_x -> U_x cos(theta)``, ``U_z -> U_x sin(theta)``.

Time-stepping coefficients
--------------------------
For the triply-periodic case the Helmholtz operator is diagonal in
Fourier space, so the implicit solve reduces to pointwise operations:

    ``ldt_1 = 1/dt + (1-c) * lapl / Re``   (explicit part)
    ``ildt_2 = 1 / (1/dt - c * lapl / Re)`` (inverse of implicit part)

The mean mode ``(ky, kz, kx) = (0, 0, 0)`` is zeroed out, since it
is passive (constant shift) for periodic flows.
"""

from dataclasses import dataclass, field
from functools import partial

import jax
from jax import Array, jit, vmap
from jax import numpy as jnp

from ..bench import timer
from ..operators import (
    curl,
    divergence,
    fourier,
    gradient,
    inverse_laplacian,
    laplacian,
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
from ..sharding import sharding
from ..timestep import make_stepper
from ..velocity import get_norm, get_norm2


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

    ys: Array | None = field(init=False, default=None)

    ldt_1: Array = field(init=False)
    ildt_2: Array = field(init=False)

    def __post_init__(self):
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
            # F = nu * k^2 * U
            self.force_amplitude = jnp.pi**2 / (4 * params.phys.re)
            self.ekin_lam = 1.0 / 4.0
            self.input_lam = jnp.pi**2 / (8 * params.phys.re)
            self.dissip_lam = self.input_lam

        elif self._system == "decaying-box":
            self.ekin_lam = 0.0
            self.input_lam = 0.0
        else:
            raise NotImplementedError

        # dU/dy in Fourier space: spectral derivative = i * ky * U_hat
        dy_base_flow_complex = (
            1j * (2 * jnp.pi / derived_params.ly) * base_flow_complex
        )

        base_flow = jnp.zeros(
            (3, padded_res.ny_padded), dtype=sharding.float_type
        )[:, :, None, None]
        dy_base_flow = jnp.zeros(
            (3, padded_res.ny_padded), dtype=sharding.float_type
        )[:, :, None, None]
        curl_base_flow = jnp.zeros(
            (3, padded_res.ny_padded), dtype=sharding.float_type
        )[:, :, None, None]
        nonlin_base_flow = jnp.zeros(
            (3, padded_res.ny_padded), dtype=sharding.float_type
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

        # Apply tilt: rotate (U_x, 0, 0) -> (U_x cos(theta), 0, U_x sin(theta))
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

        self.ys = None

        # Time-stepping coefficients
        ldt_1 = (
            1 / params.step.dt
            + (1 - params.step.implicitness) * fourier.lapl / params.phys.re
        )
        ildt_2 = 1 / (
            1 / params.step.dt
            - params.step.implicitness * fourier.lapl / params.phys.re
        )

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

    Computes ``u_p = (u^n * ldt_1 + f^n) / (1/dt - c*nu*lapl)``
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

    Computes the correction ``delta = c * (f_next - f_prev) * ildt_2``
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


def _curl_fn(state: Array) -> Array:
    """Spectral curl with wavenumbers bound from ``fourier``."""
    return curl(state, fourier.kx, fourier.ky, fourier.kz)


def _get_rhs(state: Array) -> Array:
    """Divergence-free RHS: nonlinear term + algebraic pressure projection."""
    nonlin = get_nonlin(
        state,
        flow.base_flow,
        flow.curl_base_flow,
        flow.nonlin_base_flow,
        spec_to_phys,
        phys_to_spec,
        _curl_fn,
    )
    # Poisson problem for pressure: lapl(p) = div(NL)
    lapl_pressure = divergence(nonlin, fourier.kx, fourier.ky, fourier.kz)
    # Subtract pressure gradient to enforce incompressibility
    rhs_no_lapl = nonlin - gradient(
        inverse_laplacian(lapl_pressure, fourier.inv_lapl),
        fourier.kx,
        fourier.ky,
        fourier.kz,
    )
    return rhs_no_lapl


def _predict(state: Array, rhs_no_lapl: Array) -> Array:
    """Euler predictor with algebraic Helmholtz inversion."""
    return _predict_component(state, rhs_no_lapl, flow.ldt_1, flow.ildt_2)


def _correct(
    state_prev: Array, prediction: Array, rhs_prev: Array, rhs_next: Array
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector with algebraic Helmholtz inversion."""
    return _correct_component(prediction, rhs_prev, rhs_next, flow.ildt_2)


def _norm(correction: Array) -> Array:
    """L2 convergence norm."""
    return get_norm(correction, fourier.k_metric, flow.ys)


predict_and_correct, iterate_correction = make_stepper(
    _get_rhs, _predict, _correct, _norm
)


# ── Diagnostic statistics ────────────────────────────────────────────────


def get_perturbation_energy(state: Array) -> Array:
    """Perturbation kinetic energy ``E' = ||u'||^2 / 2``."""
    return get_norm2(state, fourier.k_metric, flow.ys) / 2


def get_energy(perturbation_energy: Array, input: Array) -> Array:
    """Total kinetic energy ``E = <u_tot, u_tot> / 2``.

    Reconstructed from the perturbation energy and the forcing input via
    ``E = E' - E_lam + I / |F|``.  For decaying-box flows (no forcing),
    ``E = E'``.
    """
    if params.phys.system in monochromatic_systems:
        return (
            perturbation_energy - flow.ekin_lam + input / flow.force_amplitude
        )
    else:
        return perturbation_energy


def get_enstrophy(state: Array, input: Array) -> Array:
    """Total enstrophy times Re: ``D*Re = <grad(u_tot), grad(u_tot)>``.

    Computed from the perturbation field using
    ``D*Re = D'*Re + 2*I*Re - I_lam*Re``, where ``D'*Re`` is the
    perturbation enstrophy ``sum(-lapl * |u'|^2)``.
    """
    return (
        jnp.sum(
            -laplacian(jnp.conj(state) * state, fourier.lapl),
            dtype=sharding.float_type,
        )
        + 2 * input * params.phys.re
        - flow.input_lam * params.phys.re
    )


def get_dissipation(state: Array, input: Array) -> Array:
    """Total dissipation rate ``D = enstrophy / Re``."""
    return get_enstrophy(state, input) / params.phys.re


def get_input(state: Array) -> Array:
    """Power input from the forcing: ``I = <u_tot, F>``."""
    return (
        jnp.sum(
            jnp.conj(flow.unit_force * flow.force_amplitude)
            * state.at[flow.forced_modes].get(out_sharding=sharding.no_shard),
            dtype=sharding.float_type,
        )
        + flow.input_lam
    )


@timer("stats")
@jit
def get_stats(state: Array) -> dict[str, Array]:
    """Compute diagnostic statistics: E, I, D, E'."""
    perturbation_energy = get_perturbation_energy(state)
    input = get_input(state)
    dissipation = get_dissipation(state, input)
    energy = get_energy(perturbation_energy, input)

    stats = {
        "E": energy,
        "I": input,
        "D": dissipation,
        "E'": perturbation_energy,
    }

    return stats


# ── Divergence correction ────────────────────────────────────────────────


def correct_divergence(
    state: Array,
) -> tuple[Array, Array | None]:
    """Project the velocity onto the divergence-free subspace.

    Computes ``u_corrected = u - grad(lapl^{-1}(div(u)))`` and optionally
    returns the L2 norm of the correction.
    """
    correction = -gradient(
        inverse_laplacian(
            divergence(
                state,
                fourier.kx,
                fourier.ky,
                fourier.kz,
            ),
            fourier.inv_lapl,
        ),
        fourier.kx,
        fourier.ky,
        fourier.kz,
    )

    error = (
        get_norm(correction, fourier.k_metric, flow.ys)
        if params.debug.measure_corrections
        else None
    )

    velocity_corrected = state + correction
    return velocity_corrected, error


@timer("velocity/correct_velocity")
@jit(donate_argnums=0)
def correct_velocity(
    state: Array,
) -> tuple[Array, dict[str, Array | None] | None]:
    """Apply divergence correction and zero out the passive mean mode.

    The input buffer is donated (reused for the output).

    Returns
    -------
    velocity_corrected:
        Corrected velocity, shape ``(3, *spec_shape)``.
    norm_corrections:
        Dictionary with the divergence-correction norm (if measuring),
        or ``None`` if diagnostics are disabled.
    """
    norm_corrections = {}
    velocity_corrected = state

    if params.debug.correct_divergence:
        velocity_corrected, error = correct_divergence(velocity_corrected)
        norm_corrections["div"] = error

    # Set the mean mode to zero, it is passive
    velocity_corrected = velocity_corrected.at[sharding.vector_mean_mode].set(
        0, out_sharding=sharding.spec_vector_shard
    )

    if not params.debug.measure_corrections:
        norm_corrections = None

    return velocity_corrected, norm_corrections
