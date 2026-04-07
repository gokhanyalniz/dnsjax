from functools import partial

from jax import jit, vmap

from .bench import timer
from .parameters import params
from .rhs import get_rhs_no_lapl
from .velocity import get_norm


@partial(vmap, in_axes=(0, 0, None, None))
def get_prediction(velocity_spec, rhs_no_lapl, ldt_1, ildt_2):

    prediction = (velocity_spec * ldt_1 + rhs_no_lapl) * ildt_2

    return prediction


@partial(vmap, in_axes=(0, 0, 0, None))
def get_correction(prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2):

    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ildt_2
    )

    prediction_new = prediction + correction

    return prediction_new, correction


@timer("timestep/iterate_correction")
@jit(donate_argnums=(0, 1))
def iterate_correction(
    prediction,
    rhs_no_lapl_prev,
    kx,
    ky,
    kz,
    inv_lapl,
    k_metric,
    ys,
    ildt_2,
    base_flow,
    curl_base_flow,
    nonlin_base_flow,
):
    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        kx,
        ky,
        kz,
        inv_lapl,
        base_flow,
        curl_base_flow,
        nonlin_base_flow,
    )

    prediction_next, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction, k_metric, ys)

    return prediction_next, rhs_no_lapl_next, error


@timer("timestep/predict_and_correct")
@jit(donate_argnums=0)
def predict_and_correct(
    velocity_spec,
    kx,
    ky,
    kz,
    inv_lapl,
    k_metric,
    ys,
    ldt_1,
    ildt_2,
    base_flow,
    curl_base_flow,
    nonlin_base_flow,
):

    rhs_no_lapl_prev = get_rhs_no_lapl(
        velocity_spec,
        kx,
        ky,
        kz,
        inv_lapl,
        base_flow,
        curl_base_flow,
        nonlin_base_flow,
    )
    prediction = get_prediction(velocity_spec, rhs_no_lapl_prev, ldt_1, ildt_2)

    rhs_no_lapl_next = get_rhs_no_lapl(
        prediction,
        kx,
        ky,
        kz,
        inv_lapl,
        base_flow,
        curl_base_flow,
        nonlin_base_flow,
    )
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next, ildt_2
    )

    error = get_norm(correction, k_metric, ys)

    return prediction, rhs_no_lapl_next, error
