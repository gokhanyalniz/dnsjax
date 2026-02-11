from jax import jit
from jax import numpy as jnp

import velocity
from bench import timer
from parameters import FORCING, RE, TIME_FUNCTIONS
from rhs import AMP, FORCING_MODES, FORCING_UNIT, LAPL
from transform import DEALIAS

EKIN_LAM = 1 / 4 if FORCING in [1, 2] else 0


@timer("get_energy")
@jit
def get_energy(velocity_spec):
    energy = velocity.get_norm2(velocity_spec) / 2
    return energy


@jit
def get_perturbation_energy(energy, input):
    if FORCING in [1, 2]:
        perturbation_energy = energy + EKIN_LAM - input / AMP
    else:
        perturbation_energy = energy

    return perturbation_energy


@jit
def get_enstrophy(velocity_spec):
    enstrophy = jnp.sum(
        -LAPL * (jnp.conj(velocity_spec) * velocity_spec),
        dtype=jnp.float64,
        where=DEALIAS,
    )
    return enstrophy


@timer("get_dissipation")
@jit
def get_dissipation(velocity_spec):
    dissipation = get_enstrophy(velocity_spec) / RE
    return dissipation


@timer("get_input")
@jit
def get_input(velocity_spec):
    if FORCING in [1, 2]:
        input = jnp.sum(
            jnp.conj(velocity_spec[FORCING_MODES]) * FORCING_UNIT * AMP,
            dtype=jnp.float64,
        )
    else:
        input = 0
    return input


def get_stats(velocity_spec):
    energy = get_energy(velocity_spec)
    input = get_input(velocity_spec)
    dissipation = get_dissipation(velocity_spec)
    perturbation_energy = get_perturbation_energy(energy, input)

    stats = {
        "E": energy,
        "I": input,
        "D": dissipation,
        "E'": perturbation_energy,
    }

    return stats


get_stats = timer("get_stats")(get_stats) if TIME_FUNCTIONS else jit(get_stats)
