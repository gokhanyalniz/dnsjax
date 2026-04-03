from jax import jit
from jax import numpy as jnp

from bench import timer
from operators import laplacian
from parameters import params, periodic_systems
from rhs import force
from sharding import sharding
from velocity import get_norm2

ekin_lam = 1 / 4 if params.phys.system in ["kolmogorov", "waleffe"] else 0


def get_energy(velocity_spec, k_metric, ys):
    if params.phys.system in periodic_systems:
        return get_norm2(velocity_spec, k_metric, ys) / 2
    else:
        raise NotImplementedError


def get_perturbation_energy(energy, input):
    if params.phys.system in periodic_systems:
        if force.on:
            return energy + ekin_lam - input / force.amplitude
        else:
            return energy
    else:
        raise NotImplementedError


def get_enstrophy(velocity_spec, lapl):
    if params.phys.system in periodic_systems:
        return jnp.sum(
            -laplacian(jnp.conj(velocity_spec) * velocity_spec, lapl),
            dtype=sharding.float_type,
        )
    else:
        raise NotImplementedError


def get_dissipation(velocity_spec, lapl):
    if params.phys.system in periodic_systems:
        return get_enstrophy(velocity_spec, lapl) / params.phys.re
    else:
        raise NotImplementedError


def get_input(velocity_spec):
    if params.phys.system in periodic_systems:
        if force.on:
            return jnp.sum(
                jnp.conj(force.unit_force * force.amplitude)
                * velocity_spec.at[force.forced_modes].get(
                    out_sharding=sharding.no_shard
                ),
                dtype=sharding.float_type,
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
    energy = get_energy(velocity_spec, k_metric, ys)
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
