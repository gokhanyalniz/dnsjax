"""Triply-periodic Kolmogorov flow (monochromatic sine forcing).

This module defines the ``MonochromaticFlow`` dataclass that holds all
precomputed, flow-specific data: base flow, forcing, and laminar-state
diagnostics.  Geometry-general infrastructure (time-stepping coefficients,
solvers, divergence correction) is inherited from
``geometries.triply_periodic.TriplyPeriodicFlow``
(via ``geometries.triply_periodic.triply_periodic``).

It also exports the flow interface consumed by ``__main__``:

- ``predict_and_fully_correct`` -- fused predictor + corrector
- ``predict_and_fully_correct_measured`` -- fused step + CFL
  measurements (``steps.dat``)
- ``step_cnab2`` / ``step_cnab2_measured`` -- the CN/AB2 stepping pair
- ``set_dt`` / ``reset_ab2_kappa`` -- adaptive-dt hooks
  (``step.adaptive``; ``ldt_1``/``ildt_2`` recompute, no recompile)
- ``init_state`` -- the ``start_from_laminar`` initial state
- ``get_stats`` -- diagnostic statistics
- ``get_perturbation_energy`` -- the cheap `$E'$` read for the
  laminarization check

The post-step divergence correction + mean-mode zeroing is fused
into every stepper (``_finalize_state`` via ``make_stepper``'s
*finalize_fn*), so the returned state is already projected and no
separate ``correct_velocity`` export exists.

Base flow construction
----------------------
The monochromatic base flow `$U(y)$` is a single Fourier harmonic
(`$q_f = 1$`): the Kolmogorov profile `$U = \\sin(2\\pi y/L_y)$`,
coefficient `-0.5j` at mode `$q_f$`.

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

from jax import Array, jit
from jax import numpy as jnp

from ...geometries.triply_periodic.triply_periodic import (
    Fourier,
    TriplyPeriodicFlow,
    build_triply_periodic_stepper,
    fourier,
    get_norm2,
    ly,
)
from ...operators import phys_to_spec  # noqa: F401 – public re-export
from ...parameters import derived_params, padded_res, params
from ...sharding import register_dataclass_pytree, sharding


@register_dataclass_pytree
@dataclass
class MonochromaticFlow(TriplyPeriodicFlow):
    """Precomputed data for the monochromatic (Kolmogorov) flow.

    All attributes are built by ``__post_init__``, so the module-level
    singleton ``flow = MonochromaticFlow()`` is fully initialised at
    import.
    """

    qf: int = field(init=False)
    force_amplitude: Array = field(init=False)
    ekin_lam: float = field(init=False)
    input_lam: Array = field(init=False)
    dissip_lam: Array = field(init=False)

    forced_modes: tuple[tuple[int, ...], ...] = field(init=False)
    unit_force: Array = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.qf = 1  # Forcing harmonic

        # Fourier coefficients of the streamwise base flow U_x(y):
        # sin(qf * 2pi y / Ly) -> -0.5j at +qf.
        base_flow_complex = (
            jnp.zeros(
                padded_res.ny_padded // 2 + 1, dtype=sharding.complex_type
            )
            .at[self.qf]
            .add(-0.5j)
        )

        # Forcing amplitude that sustains the laminar state:
        # `$F = \\nu k^2 U$`
        self.force_amplitude = jnp.pi**2 / (4 * params.phys.re)
        self.ekin_lam = 1.0 / 4.0
        self.input_lam = jnp.pi**2 / (8 * params.phys.re)
        self.dissip_lam = self.input_lam

        # dU/dy in Fourier space: spectral derivative = i * ky * U_hat
        dy_base_flow_complex = 1j * (2 * jnp.pi / ly) * base_flow_complex

        Us = jnp.fft.irfft(
            base_flow_complex,
            n=padded_res.ny_padded,
            norm="forward",
        )
        dy_Us = jnp.fft.irfft(
            dy_base_flow_complex,
            n=padded_res.ny_padded,
            norm="forward",
        )

        self.base_flow = (
            jnp.zeros(
                (3, padded_res.ny_padded),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[0]
            .set(Us * derived_params.cos_tilt)
            .at[2]
            .set(Us * derived_params.sin_tilt)[:, :, None, None]
        )
        # `$\nabla \times \mathbf{U}
        #     = (\partial_y U_s \sin\theta,\;0,
        #        \;-\partial_y U_s \cos\theta)$`
        self.curl_base_flow = (
            jnp.zeros(
                (3, padded_res.ny_padded),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[0]
            .set(dy_Us * derived_params.sin_tilt)
            .at[2]
            .set(-dy_Us * derived_params.cos_tilt)[:, :, None, None]
        )

        # Forced modes and unit forcing Fourier coefficients: the
        # (+-qf, 0, 0) pair on the x and z components (tilt split).
        self.forced_modes = (
            (0, 0, 2, 2),
            (self.qf, -self.qf, self.qf, -self.qf),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        )
        self.unit_force = jnp.array(
            [
                -0.5j * derived_params.cos_tilt,
                0.5j * derived_params.cos_tilt,
                -0.5j * derived_params.sin_tilt,
                0.5j * derived_params.sin_tilt,
            ],
            dtype=sharding.complex_type,
        )


flow: MonochromaticFlow = MonochromaticFlow()

(
    init_state,
    predict_and_fully_correct,
    predict_and_fully_correct_measured,
    step_cnab2,
    step_cnab2_measured,
    set_dt,
    reset_ab2_kappa,
) = build_triply_periodic_stepper(flow)


# ── Diagnostic statistics ────────────────────────────────────────────────


def get_energy(
    perturbation_energy: Array,
    input: Array,
    fourier_: Fourier,
    flow_: MonochromaticFlow,
) -> Array:
    """Total kinetic energy"""
    return perturbation_energy - flow_.ekin_lam + input / flow_.force_amplitude


def get_enstrophy(
    state: Array,
    input: Array,
    fourier_: Fourier,
    flow_: MonochromaticFlow,
) -> Array:
    r"""Total enstrophy times Re.

    The perturbation part is the Parseval sum
    `$\sum_k k^2 |\hat{u}'_k|^2$` over the full mode set --
    ``get_norm2`` with the `$k^2$`-weighted metric carries the
    Hermitian real-FFT weight (2 for `$k_x > 0$`), matching the other
    energy diagnostics.
    """
    return (
        get_norm2(state, -fourier_.lapl * fourier_.k_metric)
        + 2 * input * params.phys.re
        - flow_.input_lam * params.phys.re
    )


def get_dissipation(
    state: Array,
    input: Array,
    fourier_: Fourier,
    flow_: MonochromaticFlow,
) -> Array:
    """Total dissipation rate `$D = \\text{enstrophy} / \\mathrm{Re}$`."""
    return get_enstrophy(state, input, fourier_, flow_) / params.phys.re


def get_input(
    state: Array, fourier_: Fourier, flow_: MonochromaticFlow
) -> Array:
    """Power input from the forcing"""
    return (
        jnp.sum(
            jnp.conj(flow_.unit_force * flow_.force_amplitude)
            * state.at[flow_.forced_modes].get(out_sharding=sharding.no_shard),
            dtype=sharding.float_type,
        )
        + flow_.input_lam
    )


def _perturbation_energy(state: Array, fourier_: Fourier) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`.

    The single definition, shared by :func:`get_stats` (which reports
    it as ``E'``) and the laminarization read
    :func:`get_perturbation_energy`.
    """
    return get_norm2(state, fourier_.k_metric) / 2


@jit
def _get_stats_jit(
    state: Array, fourier_: Fourier, flow_: MonochromaticFlow
) -> dict[str, Array]:
    """Compute diagnostic statistics: E, I, D, E'."""
    perturbation_energy = _perturbation_energy(state, fourier_)
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


def get_stats(state: Array) -> dict[str, Array]:
    """Wrapper around ``_get_stats_jit``."""
    return _get_stats_jit(state, fourier, flow)


@jit
def _get_perturbation_energy_jit(
    state: Array, fourier_: Fourier, flow_: MonochromaticFlow
) -> Array:
    r"""Perturbation kinetic energy `$E' = \|\mathbf{u}'\|^2 / 2$`."""
    return _perturbation_energy(state, fourier_)


def get_perturbation_energy(state: Array) -> Array:
    """Perturbation kinetic energy E' (for the laminarization check)."""
    return _get_perturbation_energy_jit(state, fourier, flow)
