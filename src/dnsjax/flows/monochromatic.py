"""Triply-periodic flows: Kolmogorov, Waleffe, and decaying-box.

This module defines the ``MonochromaticFlow`` dataclass that holds all
precomputed, flow-specific data: base flow, forcing, and laminar-state
diagnostics.  Geometry-general infrastructure (time-stepping coefficients,
solvers, divergence correction) is inherited from
``geometries.triply_periodic.TriplyPeriodicFlow``.

It also exports the full flow interface consumed by ``__main__``:

- ``predict_and_correct`` / ``iterate_correction`` -- time stepping
- ``get_stats`` -- diagnostic statistics
- ``correct_velocity`` -- divergence correction + mean-mode zeroing
- ``phys_to_spec`` -- forward 3D FFT (re-exported from operators)

Base flow construction
----------------------
The monochromatic base flow `$U(y)$` is defined analytically via a single
Fourier harmonic (`$q_f = 1$`):

- Kolmogorov: `$U = \\sin(2\\pi y/L_y)$`
    -- coefficient `-0.5j` at mode `$q_f$`
- Waleffe:    `$U = \\cos(2\\pi y/L_y)$`
    -- coefficient `+0.5` at mode `$q_f$`
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
rotated:
    `$U_x \\to U_x \\cos\\theta$`, `$U_z \\to U_x \\sin\\theta$`.
"""

from dataclasses import dataclass, field
from typing import Any

from jax import Array, jit
from jax import numpy as jnp

from ..bench import timer
from ..geometries.triply_periodic import (
    TriplyPeriodicFlow,
    build_triply_periodic_stepper,
    fourier,
    get_norm2,
    laplacian,
)
from ..operators import phys_to_spec  # noqa: F401 – re-export for __main__
from ..parameters import (
    derived_params,
    monochromatic_systems,
    padded_res,
    params,
)
from ..sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class MonochromaticFlow(TriplyPeriodicFlow):
    """Precomputed data for monochromatic triply-periodic flows.

    All class-level attributes are computed eagerly at definition time so
    that the module-level singleton ``flow = MonochromaticFlow()`` is
    fully initialised at import.
    """

    _system: str = field(init=False)
    dy_base_flow: Array = field(init=False)

    qf: int = field(init=False)
    force_amplitude: Array = field(init=False)
    ekin_lam: float = field(init=False)
    input_lam: Array = field(init=False)
    dissip_lam: Array = field(init=False)

    forced_modes: tuple = field(init=False)
    unit_force: Array = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

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


flow: MonochromaticFlow = MonochromaticFlow()

predict_and_correct, iterate_correction, init_state, correct_velocity = (
    build_triply_periodic_stepper(flow)
)


# ── Diagnostic statistics ────────────────────────────────────────────────


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
    perturbation_energy = get_norm2(state, fourier_.k_metric) / 2
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
