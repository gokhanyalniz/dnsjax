from jax import jit
from jax import numpy as jnp

from bench import timer
from operators import fourier
from parameters import params
from rhs import force
from sharding import float_type
from velocity import get_norm2

EKIN_LAM = 1 / 4 if params.phys.forcing in ["kolmogorov", "waleffe"] else 0


@timer("get_energy")
@jit
def get_energy(velocity_spec):
    energy = get_norm2(velocity_spec) / 2
    return energy


@jit
def get_perturbation_energy(energy, input):
    if params.phys.forcing is not None:
        perturbation_energy = (
            energy + EKIN_LAM - input / force.FORCING_AMPLITUDE
        )
    else:
        perturbation_energy = energy

    return perturbation_energy


@jit
def get_enstrophy(velocity_spec):
    enstrophy = jnp.sum(
        -fourier.LAPL * (jnp.conj(velocity_spec) * velocity_spec),
        dtype=float_type,
        where=fourier.DEALIAS,
    )
    return enstrophy


@timer("get_dissipation")
@jit
def get_dissipation(velocity_spec):
    dissipation = get_enstrophy(velocity_spec) / params.phys.Re
    return dissipation


@timer("get_input")
@jit
def get_input(velocity_spec):
    if params.phys.forcing is not None:
        input = jnp.sum(
            jnp.conj(velocity_spec[force.FORCING_MODES])
            * force.FORCING_UNIT
            * force.FORCING_AMPLITUDE,
            dtype=float_type,
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


get_stats = (
    timer("get_stats")(get_stats)
    if params.debug.time_functions
    else jit(get_stats)
)
