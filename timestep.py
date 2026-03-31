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
    ldt_1 = (
        1 / params.step.dt
        + (1 - params.step.implicitness) * fourier.lapl / params.phys.re
    )
    ildt_2 = 1 / (
        1 / params.step.dt
        - params.step.implicitness * fourier.lapl / params.phys.re
    )

    # Set the mean mode to zero, it is passive
    ldt_1 = ldt_1.at[0, 0, 0].set(0)
    ildt_2 = ildt_2.at[0, 0, 0].set(0)


stepper = Stepper()


@partial(vmap, in_axes=(0, 0, None, None))
def get_prediction(velocity_spec, rhs_no_lapl, ldt_1, ildt_2):

    prediction = (velocity_spec * ldt_1 + rhs_no_lapl) * ildt_2

    return sharding.constrain_spec_scalar(prediction)


@partial(vmap, in_axes=(0, 0, 0, None))
def get_correction(prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2):

    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ildt_2
    )

    prediction_new = prediction + correction

    return sharding.constrain_spec_scalar(
        prediction_new
    ), sharding.constrain_spec_scalar(correction)


@timer("timestep/iterate_correction")
@jit(
    donate_argnums=(0, 1),
    out_shardings=(
        sharding.spec_vector_shard,
        sharding.spec_vector_shard,
        None,
    ),
)
def iterate_correction(
    prediction,
    rhs_no_lapl_prev,
    kvec,
    inv_lapl,
    metric,
    ildt_2,
):
    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        kvec,
        inv_lapl,
    )

    prediction_next, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction, metric)

    return (
        sharding.constrain_spec_vector(prediction_next),
        sharding.constrain_spec_vector(rhs_no_lapl_next),
        error,
    )


@timer("timestep/predict_and_correct")
@jit(
    donate_argnums=0,
    out_shardings=(
        sharding.spec_vector_shard,
        sharding.spec_vector_shard,
        None,
    ),
)
def predict_and_correct(
    velocity_spec,
    kvec,
    inv_lapl,
    metric,
    ldt_1,
    ildt_2,
):

    rhs_no_lapl_prev = get_rhs_no_lapl(
        velocity_spec,
        kvec,
        inv_lapl,
    )
    prediction = get_prediction(velocity_spec, rhs_no_lapl_prev, ldt_1, ildt_2)

    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        kvec,
        inv_lapl,
    )
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction, metric)

    return (
        sharding.constrain_spec_vector(prediction),
        sharding.constrain_spec_vector(rhs_no_lapl_next),
        error,
    )
