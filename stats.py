from jax import jit
from jax import numpy as jnp

from bench import timer
from parameters import params
from rhs import force
from sharding import sharding
from velocity import get_norm2

ekin_lam = 1 / 4 if params.phys.forcing in ["kolmogorov", "waleffe"] else 0


@jit
def get_energy(velocity_spec):
    energy = get_norm2(velocity_spec) / 2
    return energy


@jit
def get_perturbation_energy(energy, input):
    if force.on:
        perturbation_energy = energy + ekin_lam - input / force.amplitude
    else:
        perturbation_energy = energy

    return perturbation_energy


@jit
def get_enstrophy(velocity_spec, lapl):
    enstrophy = jnp.sum(
        -lapl * jnp.conj(velocity_spec) * velocity_spec,
        dtype=sharding.float_type,
    )
    return enstrophy


@jit
def get_dissipation(velocity_spec, lapl):
    dissipation = get_enstrophy(velocity_spec, lapl) / params.phys.Re
    return dissipation


@jit
def get_input(velocity_spec, laminar_state):
    if force.on:
        input = jnp.sum(
            jnp.conj(laminar_state * force.amplitude)
            * velocity_spec[force.ic_f],
            dtype=sharding.float_type,
        )
    else:
        input = 0
    return input


def get_stats(
    velocity_spec,
    laminar_state,
    lapl,
):
    energy = get_energy(velocity_spec)
    input = get_input(velocity_spec, laminar_state)
    dissipation = get_dissipation(velocity_spec, lapl)
    perturbation_energy = get_perturbation_energy(energy, input)

    stats = {
        "E": energy,
        "I": input,
        "D": dissipation,
        "E'": perturbation_energy,
    }

    return stats


get_stats = (
    timer("get_stats")(get_stats)
    if params.debug.time_functions
    else jit(get_stats)
)
