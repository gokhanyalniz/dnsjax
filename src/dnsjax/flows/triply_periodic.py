"""Triply-periodic flows: Kolmogorov, Waleffe, and decaying-box.

This module defines the ``TriplyPeriodicFlow`` dataclass that holds all
precomputed, flow-specific data: base flow, forcing, time-stepping
coefficients (``ldt_1``, ``ildt_2``), and laminar-state diagnostics.

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

from dataclasses import dataclass

from jax import Array, jit
from jax import numpy as jnp

from ..bench import timer
from ..operators import (
    divergence,
    fourier,
    gradient,
    inverse_laplacian,
    laplacian,
)
from ..parameters import (
    derived_params,
    monochromatic_systems,
    padded_res,
    params,
)
from ..sharding import sharding
from ..velocity import get_norm, get_norm2


@dataclass
class TriplyPeriodicFlow:
    """Precomputed data for triply-periodic (Kolmogorov/Waleffe/decaying) flows.

    All class-level attributes are computed eagerly at definition time so
    that the module-level singleton ``flow = TriplyPeriodicFlow()`` is
    fully initialised at import.
    """

    _system: str = params.phys.system

    # Fourier coefficients of the streamwise base flow U_x(y).
    # This array includes the Nyquist mode (length ny_padded//2 + 1),
    # unlike the stored spectral velocity arrays which omit it.
    base_flow_complex: Array = jnp.zeros(
        padded_res.ny_padded // 2 + 1, dtype=sharding.complex_type
    )
    if _system in monochromatic_systems:
        qf: int = 1  # Forcing harmonic

        # Kolmogorov: sin(qf * 2pi y / Ly) -> -0.5j at +qf
        # Waleffe:    cos(qf * 2pi y / Ly) ->  0.5  at +qf
        if _system == "kolmogorov":
            base_flow_complex = base_flow_complex.at[qf].add(-0.5j)
        elif _system == "waleffe":
            base_flow_complex = base_flow_complex.at[qf].add(0.5)
        else:
            raise NotImplementedError

        # Forcing amplitude that sustains the laminar state: F = nu * k^2 * U
        force_amplitude: Array = jnp.pi**2 / (4 * params.phys.re)
        ekin_lam: float = 1 / 4
        input_lam: Array = jnp.pi**2 / (8 * params.phys.re)
        dissip_lam: Array = input_lam

    elif _system == "decaying-box":
        ekin_lam = 0
        input_lam = 0
    else:
        raise NotImplementedError

    # dU/dy in Fourier space: spectral derivative = i * ky * U_hat
    dy_base_flow_complex: Array = (
        1j * (2 * jnp.pi / derived_params.ly) * base_flow_complex
    )

    # Physical-space base flow and its derived quantities on the
    # 3/2-oversampled grid.  Shape: (3, ny_padded, 1, 1) -- trailing
    # singleton dimensions broadcast with (nz_padded, nx_padded).
    base_flow: Array = jnp.zeros(
        (3, padded_res.ny_padded), dtype=sharding.float_type
    )[:, :, None, None]
    dy_base_flow: Array = jnp.zeros(
        (3, padded_res.ny_padded), dtype=sharding.float_type
    )[:, :, None, None]
    curl_base_flow: Array = jnp.zeros(
        (3, padded_res.ny_padded), dtype=sharding.float_type
    )[:, :, None, None]
    nonlin_base_flow: Array = jnp.zeros(
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

    # Forced modes: indices into the spectral velocity array
    # (velocity_component, ky, kz, kx) at which forcing is applied.
    # The layout depends on whether / how the forcing is tilted.
    if _system in monochromatic_systems:
        if not derived_params.tilt:
            # No tilt: forcing only in u_x at +/- qf in ky
            forced_modes: tuple = ((0, 0), (qf, -qf), (0, 0), (0, 0))
        elif derived_params.tilt_90:
            # 90-degree tilt: forcing only in u_z at +/- qf in ky
            forced_modes = ((2, 2), (qf, -qf), (0, 0), (0, 0))
        else:
            # General tilt: forcing in both u_x and u_z
            forced_modes = (
                (0, 0, 2, 2),
                (qf, -qf, qf, -qf),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            )

        # Unit forcing Fourier coefficients (before amplitude scaling).
        # Kolmogorov: -0.5j / +0.5j  (sine);  Waleffe: 0.5 / 0.5  (cosine)
        if not derived_params.tilt or derived_params.tilt_90:
            if _system == "kolmogorov":
                unit_force: Array = jnp.array(
                    [-0.5j, 0.5j], dtype=sharding.complex_type
                )
            elif _system == "waleffe":
                unit_force = jnp.array([0.5, 0.5], dtype=sharding.complex_type)
        else:
            if _system == "kolmogorov":
                unit_force = jnp.array(
                    [
                        -0.5j * jnp.cos(tilt_rad),
                        0.5j * jnp.cos(tilt_rad),
                        -0.5j * jnp.sin(tilt_rad),
                        0.5j * jnp.sin(tilt_rad),
                    ],
                    dtype=sharding.complex_type,
                )
            elif _system == "waleffe":
                unit_force = jnp.array(
                    [
                        0.5 * jnp.cos(tilt_rad),
                        0.5 * jnp.cos(tilt_rad),
                        -0.5j * jnp.sin(tilt_rad),
                        0.5j * jnp.sin(tilt_rad),
                    ],
                    dtype=sharding.complex_type,
                )

    ys: Array | None = None  # No wall-normal grid for periodic flows

    # Make sure we never use these:
    del base_flow_complex
    del dy_base_flow_complex

    # Time-stepping coefficients (diagonal Helmholtz operator in Fourier
    # space).  ldt_1 encodes the explicit part; ildt_2 is the reciprocal
    # of the implicit part.
    ldt_1: Array = (
        1 / params.step.dt
        + (1 - params.step.implicitness) * fourier.lapl / params.phys.re
    )
    ildt_2: Array = 1 / (
        1 / params.step.dt
        - params.step.implicitness * fourier.lapl / params.phys.re
    )

    # The mean mode is passive (constant velocity shift); zero it out so
    # that the predictor/corrector leaves it untouched.
    ldt_1 = ldt_1.at[sharding.scalar_mean_mode].set(
        0, out_sharding=sharding.spec_scalar_shard
    )
    ildt_2 = ildt_2.at[sharding.scalar_mean_mode].set(
        0, out_sharding=sharding.spec_scalar_shard
    )


flow: TriplyPeriodicFlow = TriplyPeriodicFlow()


def get_perturbation_energy(
    velocity_spec: Array, k_metric: Array, ys: Array | None
) -> Array:
    """Perturbation kinetic energy ``E' = ||u'||^2 / 2``."""
    # E' = <u, u> / 2
    return get_norm2(velocity_spec, k_metric, ys) / 2


def get_energy(perturbation_energy: Array, input: Array) -> Array:
    """Total kinetic energy ``E = <u_tot, u_tot> / 2``.

    Reconstructed from the perturbation energy and the forcing input via
    ``E = E' - E_lam + I / |F|``.  For decaying-box flows (no forcing),
    ``E = E'``.
    """
    # E_tot = <u + U_lam, u + U_lam> / 2
    # = E' - E_lam + I / |F|
    if params.phys.system in monochromatic_systems:
        return (
            perturbation_energy - flow.ekin_lam + input / flow.force_amplitude
        )
    else:
        return perturbation_energy


def get_enstrophy(velocity_spec: Array, input: Array, lapl: Array) -> Array:
    """Total enstrophy times Re: ``D*Re = <grad(u_tot), grad(u_tot)>``.

    Computed from the perturbation field using
    ``D*Re = D'*Re + 2*I*Re - I_lam*Re``, where ``D'*Re`` is the
    perturbation enstrophy ``sum(-lapl * |u'|^2)``.
    """
    # D * Re = <grad(u_tot),grad(u_tot)>
    # = <k^2 * (u + U_lam), u + U_lam>
    # = D' * Re + 2 * I * Re - I_lam * Re
    return (
        jnp.sum(
            -laplacian(jnp.conj(velocity_spec) * velocity_spec, lapl),
            dtype=sharding.float_type,
        )
        + 2 * input * params.phys.re
        - flow.input_lam * params.phys.re
    )


def get_dissipation(velocity_spec: Array, input: Array, lapl: Array) -> Array:
    """Total dissipation rate ``D = enstrophy / Re``."""
    return get_enstrophy(velocity_spec, input, lapl) / params.phys.re


def get_input(velocity_spec: Array) -> Array:
    """Power input from the forcing: ``I = <u_tot, F>``."""
    # I = <u_tot, F> = <u + U_lam, F>
    return (
        jnp.sum(
            jnp.conj(flow.unit_force * flow.force_amplitude)
            * velocity_spec.at[flow.forced_modes].get(
                out_sharding=sharding.no_shard
            ),
            dtype=sharding.float_type,
        )
        + flow.input_lam
    )


@timer("stats")
@jit
def get_stats(
    velocity_spec: Array,
    lapl: Array,
    k_metric: Array,
    ys: Array | None,
) -> dict[str, Array]:
    """Compute diagnostic statistics: E, I, D, E'."""
    perturbation_energy = get_perturbation_energy(velocity_spec, k_metric, ys)
    input = get_input(velocity_spec)
    dissipation = get_dissipation(velocity_spec, input, lapl)
    energy = get_energy(perturbation_energy, input)

    stats = {
        "E": energy,
        "I": input,
        "D": dissipation,
        "E'": perturbation_energy,
    }

    return stats


def correct_divergence(
    velocity_spec: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    inv_lapl: Array,
    k_metric: Array,
    ys: Array | None,
) -> tuple[Array, Array | None]:
    """Project the velocity onto the divergence-free subspace.

    Computes ``u_corrected = u - grad(lapl^{-1}(div(u)))`` and optionally
    returns the L2 norm of the correction.
    """
    correction = -gradient(
        inverse_laplacian(
            divergence(
                velocity_spec,
                kx,
                ky,
                kz,
            ),
            inv_lapl,
        ),
        kx,
        ky,
        kz,
    )

    error = (
        get_norm(correction, k_metric, ys)
        if params.debug.measure_corrections
        else None
    )

    velocity_corrected = velocity_spec + correction
    return velocity_corrected, error


@timer("velocity/correct_velocity")
@jit(donate_argnums=0)
def correct_velocity(
    velocity_spec: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    inv_lapl: Array,
    k_metric: Array,
    ys: Array | None,
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
    velocity_corrected = velocity_spec

    if params.debug.correct_divergence:
        velocity_corrected, error = correct_divergence(
            velocity_corrected, kx, ky, kz, inv_lapl, k_metric, ys
        )
        norm_corrections["div"] = error

    # Set the mean mode to zero, it is passive
    velocity_corrected = velocity_corrected.at[sharding.vector_mean_mode].set(
        0, out_sharding=sharding.spec_vector_shard
    )

    if not params.debug.measure_corrections:
        norm_corrections = None

    return velocity_corrected, norm_corrections
