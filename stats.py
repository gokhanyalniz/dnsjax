from jax import jit
from jax import numpy as jnp

from bench import timer
from operators import laplacian
from parameters import params
from rhs import force
from sharding import sharding
from velocity import get_norm2

ekin_lam = 1 / 4 if params.phys.forcing in ["kolmogorov", "waleffe"] else 0


def get_energy(velocity_spec, metric):
    energy = get_norm2(velocity_spec, metric) / 2
    return energy


def get_perturbation_energy(energy, input):
    if force.on:
        perturbation_energy = energy + ekin_lam - input / force.amplitude
    else:
        perturbation_energy = energy

    return perturbation_energy


def get_enstrophy(velocity_spec, lapl):
    enstrophy = jnp.sum(
        -laplacian(jnp.conj(velocity_spec) * velocity_spec, lapl),
        dtype=sharding.float_type,
    )
    return enstrophy


def get_dissipation(velocity_spec, lapl):
    dissipation = get_enstrophy(velocity_spec, lapl) / params.phys.re
    return dissipation


def get_input(velocity_spec):
    if force.on:
        input = jnp.sum(
            jnp.conj(force.unit_force * force.amplitude)
            * velocity_spec.at[force.forced_modes].get(
                out_sharding=sharding.no_shard
            ),
            dtype=sharding.float_type,
        )
    else:
        input = 0
    return input


@timer("stats")
@jit
def get_stats(
    velocity_spec,
    lapl,
    metric,
):
    energy = get_energy(velocity_spec, metric)
    input = get_input(velocity_spec)
    dissipation = get_dissipation(velocity_spec, lapl)
    perturbation_energy = get_perturbation_energy(energy, input)

    stats = {
        "E": energy,
        "I": input,
        "D": dissipation,
        "E'": perturbation_energy,
    }

    return stats
