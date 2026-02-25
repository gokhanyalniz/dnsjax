from dataclasses import dataclass
from functools import partial

import jax
from jax import jit, lax, vmap

from bench import timer
from operators import fourier
from parameters import params
from rhs import get_rhs_no_lapl
from sharding import sharding
from velocity import correct_velocity, get_norm


@dataclass
class Stepper:
    # Zero the aliased modes to (potentially) save on computations
    LDT_1 = (
        1 / params.step.dt
        + (1 - params.step.implicitness) * fourier.LAPL / params.phys.Re
    ) * fourier.DEALIAS
    ILDT_2 = (
        1
        / (
            1 / params.step.dt
            - params.step.implicitness * fourier.LAPL / params.phys.Re
        )
    ) * fourier.DEALIAS


stepper = Stepper()


@timer("get_prediction")
@jit(donate_argnums=0)
@partial(vmap, in_axes=(0, 0, None, None))
def get_prediction(velocity_spec, rhs_no_lapl, ldt1, ildt_2):

    prediction = (velocity_spec * ldt1 + rhs_no_lapl) * ildt_2

    return jax.lax.with_sharding_constraint(
        prediction, sharding.scalar_spec_shard
    )


@timer("get_correction")
@jit(donate_argnums=(0, 1))
@partial(vmap, in_axes=(0, 0, 0, None))
def get_correction(
    prediction_prev, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
):

    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ildt_2
    )

    prediction_next = prediction_prev + correction

    return (
        jax.lax.with_sharding_constraint(
            prediction_next, sharding.scalar_spec_shard
        ),
        jax.lax.with_sharding_constraint(
            correction, sharding.scalar_spec_shard
        ),
    )


@jit
def timestep_iteration_condition(val):
    _, _, error, c, _ = val
    condition = (c < params.step.max_corrector_iterations) & (
        error > params.step.corrector_tolerance
    )
    return condition


def timestep_iterate(val):
    prediction, rhs_no_lapl_prev, _, c, operators = val
    (
        nabla,
        inv_lapl,
        dealias,
        ildt_2,
    ) = operators
    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        nabla,
        inv_lapl,
        dealias,
    )
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )
    error = get_norm(correction, dealias)
    return (
        jax.lax.with_sharding_constraint(prediction, sharding.spec_shard),
        jax.lax.with_sharding_constraint(
            rhs_no_lapl_next, sharding.spec_shard
        ),
        error,
        c + 1,
        operators,
    )


def timestep(
    velocity_spec,
    nabla,
    inv_lapl,
    zero_mean,
    dealias,
    ldt1,
    ildt_2,
):

    rhs_no_lapl_prev = get_rhs_no_lapl(
        velocity_spec,
        nabla,
        inv_lapl,
        dealias,
    )
    prediction = get_prediction(velocity_spec, rhs_no_lapl_prev, ldt1, ildt_2)

    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        nabla,
        inv_lapl,
        dealias,
    )
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )
    error = get_norm(correction, dealias)
    c = 1

    operators = (
        nabla,
        inv_lapl,
        dealias,
        ildt_2,
    )
    init_val = prediction, rhs_no_lapl_next, error, c, operators
    prediction, _, error, c, _ = lax.while_loop(
        timestep_iteration_condition, timestep_iterate, init_val
    )

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


timestep = (
    timer("timestep")(timestep)
    if params.debug.time_functions
    else jit(timestep, donate_argnums=0)
)

timestep_iterate = (
    timer("timestep_iterate")(timestep_iterate)
    if params.debug.time_functions
    else jit(timestep_iterate, donate_argnums=0)
)
