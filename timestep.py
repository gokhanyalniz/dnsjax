from functools import partial

import jax
from jax import jit, lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_array_info import sharding_vis

from parameters import DT, IMPLICITNESS, NCORR, RE, STEPTOL
from rhs import get_rhs_no_lapl
from sharding import MESH
from transform import LAPL
from velocity import correct_velocity, get_norm

LDT_1 = 1 / DT + (1 - IMPLICITNESS) * LAPL / RE
ILDT_2 = 1 / (1 / DT - IMPLICITNESS * LAPL / RE)


@partial(jit, donate_argnums=0)
def get_prediction(velocity_spec):

    rhs_no_lapl = get_rhs_no_lapl(velocity_spec)

    prediction = (velocity_spec * LDT_1 + rhs_no_lapl) * ILDT_2

    return jax.lax.with_sharding_constraint(
        prediction, NamedSharding(MESH, P(None, "Z", "X", None))
    ), jax.lax.with_sharding_constraint(
        rhs_no_lapl, NamedSharding(MESH, P(None, "Z", "X", None))
    )


@partial(jit, donate_argnums=(0, 1))
def get_correction(prediction_prev, rhs_no_lapl_prev):

    rhs_no_lapl_next = get_rhs_no_lapl(prediction_prev)

    correction = IMPLICITNESS * (rhs_no_lapl_next - rhs_no_lapl_prev) * ILDT_2

    prediction_next = prediction_prev + correction

    error = get_norm(correction)

    return (
        jax.lax.with_sharding_constraint(
            prediction_next, NamedSharding(MESH, P(None, "Z", "X", None))
        ),
        jax.lax.with_sharding_constraint(
            rhs_no_lapl_next, NamedSharding(MESH, P(None, "Z", "X", None))
        ),
        error,
    )


@jit
def cond_fun(val):
    _, _, error, c = val
    return (c < NCORR) & (error > STEPTOL)


@partial(jit, donate_argnums=0)
def body_fun(val):
    prediction, rhs_no_lapl, _, c = val
    prediction, rhs_no_lapl, error = get_correction(prediction, rhs_no_lapl)
    return prediction, rhs_no_lapl, error, c + 1


@partial(jit, donate_argnums=0)
def timestep(velocity_spec):

    prediction, rhs_no_lapl = get_prediction(velocity_spec)

    prediction, rhs_no_lapl, error = get_correction(prediction, rhs_no_lapl)
    c = 1

    init_val = prediction, rhs_no_lapl, error, c
    prediction, rhs_no_lapl, error, c = lax.while_loop(cond_fun, body_fun, init_val)

    velocity_spec_next = correct_velocity(prediction)

    return (
        jax.lax.with_sharding_constraint(
            velocity_spec_next, NamedSharding(MESH, P(None, "Z", "X", None))
        ),
        error,
        c,
    )
