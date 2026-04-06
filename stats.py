from jax import jit
from jax import numpy as jnp

from bench import timer
from operators import laplacian
from parameters import params, periodic_systems
from rhs import flow, force
from sharding import sharding
from velocity import get_norm2


def get_perturbation_energy(velocity_spec, k_metric, ys):
    # E' = <u, u> / 2
    return get_norm2(velocity_spec, k_metric, ys) / 2


def get_energy(perturbation_energy, input):
    # E_tot = <u_tot, u_tot> / 2
    if params.phys.system in periodic_systems:
        if force.on:
            # E_tot = <u + U_lam, u + U_lam> / 2
            # = E' - E_lam + I / |F|
            return (
                perturbation_energy - flow.ekin_lam + input / force.amplitude
            )
        else:
            return perturbation_energy
    else:
        raise NotImplementedError


def get_enstrophy(velocity_spec, input, lapl):
    if params.phys.system in periodic_systems:
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
    else:
        raise NotImplementedError


def get_dissipation(velocity_spec, input, lapl):
    if params.phys.system in periodic_systems:
        return get_enstrophy(velocity_spec, input, lapl) / params.phys.re


def get_input(velocity_spec):
    system = params.phys.system
    if system in periodic_systems:
        if force.on:
            # I = <u_tot, F> = <u + U_lam, F>
            return (
                jnp.sum(
                    jnp.conj(force.unit_force * force.amplitude)
                    * velocity_spec.at[force.forced_modes].get(
                        out_sharding=sharding.no_shard
                    ),
                    dtype=sharding.float_type,
                )
                + flow.input_lam
            )
        else:
            return 0
    else:
        raise NotImplementedError


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
