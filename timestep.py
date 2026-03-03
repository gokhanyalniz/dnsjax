from dataclasses import dataclass
from functools import partial

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
        + (1 - params.step.implicitness) * fourier.lapl / params.phys.re
    ) * fourier.active_modes
    ildt_2 = (
        1
        / (
            1 / params.step.dt
            - params.step.implicitness * fourier.lapl / params.phys.re
        )
    ) * fourier.active_modes


stepper = Stepper()


@partial(vmap, in_axes=(0, 0, None, None))
def get_prediction(velocity_spec, rhs_no_lapl, ldt_1, ildt_2):

    prediction = (velocity_spec * ldt_1 + rhs_no_lapl) * ildt_2

    return sharding.constrain_scalar(prediction)


@partial(vmap, in_axes=(0, 0, 0, None))
def get_correction(prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2):

    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ildt_2
    )

    prediction_new = prediction + correction

    return sharding.constrain_scalar(
        prediction_new
    ), sharding.constrain_scalar(correction)


@timer("timestep/iterate_correction")
@jit(
    donate_argnums=(0, 1),
    out_shardings=(sharding.vector_shard, sharding.vector_shard, None),
)
def iterate_correction(
    prediction,
    rhs_no_lapl_prev,
    unit_force,
    kvec,
    inv_lapl,
    active_modes,
    ildt_2,
):
    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        unit_force,
        kvec,
        inv_lapl,
        active_modes,
    )

    prediction_next, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction)

    return (
        sharding.constrain_vector(prediction_next),
        sharding.constrain_vector(rhs_no_lapl_next),
        error,
    )


@timer("timestep/predict_and_correct")
@jit(
    donate_argnums=0,
    out_shardings=(sharding.vector_shard, sharding.vector_shard, None),
)
def predict_and_correct(
    velocity_spec,
    unit_force,
    kvec,
    inv_lapl,
    active_modes,
    ldt_1,
    ildt_2,
):

    rhs_no_lapl_prev = get_rhs_no_lapl(
        velocity_spec,
        unit_force,
        kvec,
        inv_lapl,
        active_modes,
    )
    prediction = get_prediction(velocity_spec, rhs_no_lapl_prev, ldt_1, ildt_2)

    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        unit_force,
        kvec,
        inv_lapl,
        active_modes,
    )
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction)

    return (
        sharding.constrain_vector(prediction),
        sharding.constrain_vector(rhs_no_lapl_next),
        error,
    )
