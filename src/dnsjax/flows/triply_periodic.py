from dataclasses import dataclass

from jax import jit
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
    _system = params.phys.system

    # This includes the Nyquist mode, unlike all other complex arrays
    base_flow_complex = jnp.zeros(
        padded_res.ny_padded // 2 + 1, dtype=sharding.complex_type
    )
    if _system in monochromatic_systems:
        qf = 1  # Forcing harmonic

        if _system == "kolmogorov":
            base_flow_complex = base_flow_complex.at[qf].add(-0.5j)
        elif _system == "waleffe":
            base_flow_complex = base_flow_complex.at[qf].add(0.5)
        else:
            raise NotImplementedError

        force_amplitude = jnp.pi**2 / (4 * params.phys.re)
        ekin_lam = 1 / 4
        input_lam = jnp.pi**2 / (8 * params.phys.re)
        dissip_lam = input_lam

    elif _system == "decaying-box":
        ekin_lam = 0
        input_lam = 0
    else:
        raise NotImplementedError

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
    curl_base_flow = curl_base_flow.at[2].set(-dy_base_flow[0])

    tilt_rad = derived_params.tilt_rad
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

    if _system in monochromatic_systems:
        if not derived_params.tilt:
            forced_modes = ((0, 0), (qf, -qf), (0, 0), (0, 0))
        elif derived_params.tilt_90:
            forced_modes = ((2, 2), (qf, -qf), (0, 0), (0, 0))
        else:
            forced_modes = (
                (0, 0, 2, 2),
                (qf, -qf, qf, -qf),
                (0, 0, 0, 0),
                (0, 0, 0, 0),
            )

        if not derived_params.tilt or derived_params.tilt_90:
            if _system == "kolmogorov":
                unit_force = jnp.array(
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

    ys = None  # We don't need to do any y-based operation

    # Make sure we never use these:
    del base_flow_complex
    del dy_base_flow_complex

    # Time stepping matrices
    ldt_1 = (
        1 / params.step.dt
        + (1 - params.step.implicitness) * fourier.lapl / params.phys.re
    )
    ildt_2 = 1 / (
        1 / params.step.dt
        - params.step.implicitness * fourier.lapl / params.phys.re
    )

    # Set the mean mode to zero, it is passive
    ldt_1 = ldt_1.at[sharding.scalar_mean_mode].set(
        0, out_sharding=sharding.spec_scalar_shard
    )
    ildt_2 = ildt_2.at[sharding.scalar_mean_mode].set(
        0, out_sharding=sharding.spec_scalar_shard
    )


flow = TriplyPeriodicFlow()


def get_perturbation_energy(velocity_spec, k_metric, ys):
    # E' = <u, u> / 2
    return get_norm2(velocity_spec, k_metric, ys) / 2


def get_energy(perturbation_energy, input):
    # E_tot = <u_tot, u_tot> / 2
    # E_tot = <u + U_lam, u + U_lam> / 2
    # = E' - E_lam + I / |F|
    if params.phys.system in monochromatic_systems:
        return (
            perturbation_energy - flow.ekin_lam + input / flow.force_amplitude
        )
    else:
        return perturbation_energy


def get_enstrophy(velocity_spec, input, lapl):
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


def get_dissipation(velocity_spec, input, lapl):
    return get_enstrophy(velocity_spec, input, lapl) / params.phys.re


def get_input(velocity_spec):
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
    velocity_spec,
    lapl,
    k_metric,
    ys,
):
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


def correct_divergence(velocity_spec, kx, ky, kz, inv_lapl, k_metric, ys):
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
def correct_velocity(velocity_spec, kx, ky, kz, inv_lapl, k_metric, ys):
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
