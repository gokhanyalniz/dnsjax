from jax import numpy as jnp

import velocity
from parameters import AMP, IC_F, RE
from transform import FORCE, LAPL

EKIN_LAM = 1 / 4

# TODO: Normalize with respect to the laminar values


def get_energy(velocity_spec):
    return velocity.get_norm2(velocity_spec) / 2


def get_perturbation_energy(velocity_spec):
    energy = get_energy(velocity_spec)
    input = get_input(velocity_spec)
    perturbation_energy = energy + EKIN_LAM - input / AMP

    return perturbation_energy


def get_enstrophy(velocity_spec):
    return jnp.sum(-LAPL * jnp.conj(velocity_spec) * velocity_spec).real


def get_dissipation(velocity_spec):
    return get_enstrophy(velocity_spec) / RE


def get_input(velocity_spec):
    return velocity.get_inprod(velocity_spec[IC_F], FORCE)


def get_stats(velocity_spec):
    stats = {}
    stats["E"] = get_energy(velocity_spec)
    stats["I"] = get_input(velocity_spec)
    stats["D"] = get_dissipation(velocity_spec)
    stats["E'"] = get_perturbation_energy(velocity_spec)

    return stats
