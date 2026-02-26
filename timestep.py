from dataclasses import dataclass
from functools import partial

import jax
from jax import jit, vmap

from bench import timer
from operators import fourier
from parameters import params
from rhs import get_rhs_no_lapl
from sharding import sharding
from velocity import get_norm


@dataclass
class Stepper:
    # Zero the aliased modes to (potentially) save on computations
    ldt_1 = (
        1 / params.step.dt
        + (1 - params.step.implicitness) * fourier.lapl / params.phys.Re
    ) * fourier.active_modes
    ildt_2 = (
        1
        / (
            1 / params.step.dt
            - params.step.implicitness * fourier.lapl / params.phys.Re
        )
    ) * fourier.active_modes


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


def iterate_correction(
    prediction,
    rhs_no_lapl_prev,
    laminar_state,
    nabla,
    inv_lapl,
    active_modes,
    ildt_2,
):
    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        laminar_state,
        nabla,
        inv_lapl,
        active_modes,
    )

    prediction_next, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction)

    return (
        jax.lax.with_sharding_constraint(prediction_next, sharding.spec_shard),
        jax.lax.with_sharding_constraint(
            rhs_no_lapl_next, sharding.spec_shard
        ),
        error,
    )


def predict_and_correct(
    velocity_spec,
    laminar_state,
    nabla,
    inv_lapl,
    active_modes,
    ldt1,
    ildt_2,
):

    rhs_no_lapl_prev = get_rhs_no_lapl(
        velocity_spec,
        laminar_state,
        nabla,
        inv_lapl,
        active_modes,
    )
    prediction = get_prediction(velocity_spec, rhs_no_lapl_prev, ldt1, ildt_2)

    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        laminar_state,
        nabla,
        inv_lapl,
        active_modes,
    )
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction)

    return (
        jax.lax.with_sharding_constraint(prediction, sharding.spec_shard),
        jax.lax.with_sharding_constraint(
            rhs_no_lapl_next, sharding.spec_shard
        ),
        error,
    )


iterate_correction = (
    timer("iterate_correction")(iterate_correction)
    if params.debug.time_functions
    else jit(iterate_correction, donate_argnums=(0, 1))
)


predict_and_correct = (
    timer("predict_and_correct")(predict_and_correct)
    if params.debug.time_functions
    else jit(predict_and_correct, donate_argnums=0)
)
