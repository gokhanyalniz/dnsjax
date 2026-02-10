from functools import partial

import jax
from jax import jit
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax_array_info import sharding_vis

from parameters import DT, IMPLICITNESS, NCORR, RE, STEPTOL
from rhs import get_rhs_no_lapl
from sharding import MESH
from transform import LAPL, spec_to_phys_vector
from velocity import ZERO_MEAN, correct_divergence, get_norm

LDT_1 = 1 / DT + (1 - IMPLICITNESS) * LAPL / RE
ILDT_2 = 1 / (1 / DT - IMPLICITNESS * LAPL / RE)


@partial(jit, donate_argnums=0)
def get_prediction(velocity_spec, velocity_phys):
    rhs_no_lapl = get_rhs_no_lapl(velocity_phys)

    prediction = (velocity_spec * LDT_1 + rhs_no_lapl) * ILDT_2

    return prediction, rhs_no_lapl


@partial(jit, donate_argnums=(0, 1))
def get_correction(prediction_prev, rhs_no_lapl_prev):

    rhs_no_lapl_next = get_rhs_no_lapl(spec_to_phys_vector(prediction_prev))

    correction = IMPLICITNESS * (rhs_no_lapl_next - rhs_no_lapl_prev) * ILDT_2

    prediction_next = prediction_prev + correction

    error = get_norm(correction)

    return prediction_next, rhs_no_lapl_next, error

def timestep(velocity_spec, velocity_phys):

    prediction, rhs_no_lapl = get_prediction(velocity_spec, velocity_phys)

    for c in range(NCORR):
        prediction, rhs_no_lapl, error = get_correction(prediction, rhs_no_lapl)

        if error < STEPTOL:
            velocity_spec_next = correct_divergence(prediction) * ZERO_MEAN

            # TODO: apply a bunch of symmetries

            velocity_phys_next = spec_to_phys_vector(velocity_spec_next)

            break

        elif c == NCORR - 1:
            exit("Timestep did not converge.")

    return jax.lax.with_sharding_constraint(
        velocity_spec_next, NamedSharding(MESH, P(None, "Z", "X", None))
    ), jax.lax.with_sharding_constraint(
        velocity_phys_next, NamedSharding(MESH, P(None, "Z", "X", None))
    )
