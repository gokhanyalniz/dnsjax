from dataclasses import dataclass
from functools import partial

import jax
from jax import jit, vmap

from bench import timer
from operators import fourier
from parameters import params
from rhs import get_rhs_no_lapl
from sharding import sharding
from velocity import correct_velocity, get_norm


@dataclass
class Stepper:
    # Zero the aliased modes to (potentially) save on computations
    ldt_1 = (
        1 / params.step.dt
        + (1 - params.step.implicitness) * fourier.lapl / params.phys.Re
    ) * fourier.dealias
    ildt_2 = (
        1
        / (
            1 / params.step.dt
            - params.step.implicitness * fourier.lapl / params.phys.Re
        )
    ) * fourier.dealias


stepper = Stepper()


@jit(donate_argnums=0)
@partial(vmap, in_axes=(0, 0, None, None))
def get_prediction(velocity_spec, rhs_no_lapl, ldt1, ildt_2):

    prediction = (velocity_spec * ldt1 + rhs_no_lapl) * ildt_2

    return jax.lax.with_sharding_constraint(
        prediction, sharding.scalar_spec_shard
    )


@jit(donate_argnums=(0, 1))
@partial(vmap, in_axes=(0, 0, 0, None))
def get_correction(prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2):

    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ildt_2
    )

    prediction_new = prediction + correction

    return jax.lax.with_sharding_constraint(
        prediction_new, sharding.scalar_spec_shard
    ), jax.lax.with_sharding_constraint(correction, sharding.scalar_spec_shard)


@timer("timestep")
def timestep(
    velocity_spec,
    laminar_state,
    nabla,
    inv_lapl,
    zero_mean,
    dealias,
    ldt1,
    ildt_2,
):

    rhs_no_lapl_prev = get_rhs_no_lapl(
        velocity_spec,
        laminar_state,
        nabla,
        inv_lapl,
        dealias,
    )
    prediction = get_prediction(velocity_spec, rhs_no_lapl_prev, ldt1, ildt_2)

    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        laminar_state,
        nabla,
        inv_lapl,
        dealias,
    )
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction)
    c = 1
    while (
        error > params.step.corrector_tolerance
        and c < params.step.max_corrector_iterations
    ):
        rhs_no_lapl_prev = rhs_no_lapl_next
        rhs_no_lapl_next = get_rhs_no_lapl(
            prediction,
            laminar_state,
            nabla,
            inv_lapl,
            dealias,
        )
        prediction, correction = get_correction(
            prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
        )

        error = get_norm(correction)
        c += 1

    velocity_spec_next = correct_velocity(
        prediction, nabla, inv_lapl, zero_mean
    )

    return (
        jax.lax.with_sharding_constraint(
            velocity_spec_next, sharding.spec_shard
        ),
        error,
        c,
    )
