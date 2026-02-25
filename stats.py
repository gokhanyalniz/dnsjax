from functools import partial

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


# @partial(jit, static_argnames=["force"])
@partial(jit)
def _get_perturbation_energy(energy, input, forcing_amplitude):
    if params.phys.forcing is not None:
        perturbation_energy = energy + EKIN_LAM - input / forcing_amplitude
    else:
        perturbation_energy = energy

    return perturbation_energy


def get_perturbation_energy(
    energy, input, forcing_amplitude=force.FORCING_AMPLITUDE
):
    return _get_perturbation_energy(energy, input, forcing_amplitude)


# @partial(jit, static_argnames=["fourier"])
@partial(jit)
def _get_enstrophy(velocity_spec, lapl, dealias):
    enstrophy = jnp.sum(
        -lapl * (jnp.conj(velocity_spec) * velocity_spec),
        dtype=float_type,
        where=dealias,
    )
    return enstrophy


def get_enstrophy(velocity_spec, lapl=fourier.LAPL, dealias=fourier.DEALIAS):
    return _get_enstrophy(velocity_spec, lapl, dealias)


@timer("get_dissipation")
@jit
def get_dissipation(velocity_spec):
    dissipation = get_enstrophy(velocity_spec) / params.phys.Re
    return dissipation


# @partial(jit, static_argnames=["force"])
@partial(jit)
def _get_input(velocity_spec, forcing_modes, forcing_unit, forcing_amplitude):
    if params.phys.forcing is not None:
        input = jnp.sum(
            jnp.conj(velocity_spec[forcing_modes])
            * forcing_unit
            * forcing_amplitude,
            dtype=float_type,
        )
    else:
        input = 0
    return input


@timer("get_input")
def get_input(
    velocity_spec,
    forcing_modes=force.FORCING_MODES,
    forcing_unit=force.FORCING_UNIT,
    forcing_amplitude=force.FORCING_AMPLITUDE,
):
    return _get_input(
        velocity_spec, forcing_modes, forcing_unit, forcing_amplitude
    )


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
