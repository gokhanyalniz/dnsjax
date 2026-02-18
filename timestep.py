from functools import partial

import jax
from jax import jit, lax, vmap
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from bench import timer
from parameters import params
from rhs import LAPL, get_rhs_no_lapl
from sharding import MESH
from transform import DEALIAS
from velocity import correct_velocity, get_norm

# Zero the aliased modes to (potentially) save on computations
LDT_1 = (
    1 / params.step.dt + (1 - params.step.implicitness) * LAPL / params.phys.Re
) * DEALIAS
ILDT_2 = (
    1 / (1 / params.step.dt - params.step.implicitness * LAPL / params.phys.Re)
) * DEALIAS


@timer("get_prediction")
@partial(jit, donate_argnums=0)
@vmap
def get_prediction(velocity_spec, rhs_no_lapl):

    prediction = (velocity_spec * LDT_1 + rhs_no_lapl) * ILDT_2

    return jax.lax.with_sharding_constraint(
        prediction, NamedSharding(MESH, P("Z", "X", None))
    )


@timer("get_correction")
@partial(jit, donate_argnums=(0, 1))
@vmap
def get_correction(prediction_prev, rhs_no_lapl_prev, rhs_no_lapl_next):

    correction = (
        params.step.implicitness
        * (rhs_no_lapl_next - rhs_no_lapl_prev)
        * ILDT_2
    )

    prediction_next = prediction_prev + correction

    return (
        jax.lax.with_sharding_constraint(
            prediction_next, NamedSharding(MESH, P("Z", "X", None))
        ),
        jax.lax.with_sharding_constraint(
            correction, NamedSharding(MESH, P("Z", "X", None))
        ),
    )


@jit
def timestep_iteration_condition(val):
    _, _, error, c = val
    return (c < params.step.max_corrector_iterations) & (
        error > params.step.corrector_tolerance
    )


@partial(jit, donate_argnums=0)
def timestep_iterate(val):
    prediction, rhs_no_lapl_prev, _, c = val
    rhs_no_lapl_next = get_rhs_no_lapl(prediction)
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next
    )
    error = get_norm(correction)
    return (
        jax.lax.with_sharding_constraint(
            prediction, NamedSharding(MESH, P(None, "Z", "X", None))
        ),
        jax.lax.with_sharding_constraint(
            rhs_no_lapl_next, NamedSharding(MESH, P(None, "Z", "X", None))
        ),
        error,
        c + 1,
    )


def timestep(velocity_spec):

    rhs_no_lapl_prev = get_rhs_no_lapl(velocity_spec)
    prediction = get_prediction(velocity_spec, rhs_no_lapl_prev)

    rhs_no_lapl_next = get_rhs_no_lapl(prediction)
    prediction, correction = get_correction(
        prediction, rhs_no_lapl_prev, rhs_no_lapl_next
    )
    error = get_norm(correction)
    c = 1

    init_val = prediction, rhs_no_lapl_next, error, c
    prediction, _, error, c = lax.while_loop(
        timestep_iteration_condition, timestep_iterate, init_val
    )

    velocity_spec_next = correct_velocity(prediction)

    return (
        jax.lax.with_sharding_constraint(
            velocity_spec_next, NamedSharding(MESH, P(None, "Z", "X", None))
        ),
        error,
        c,
    )


timestep = (
    timer("timestep")(timestep)
    if params.debug.time_functions
    else jit(timestep, donate_argnums=0)
)
