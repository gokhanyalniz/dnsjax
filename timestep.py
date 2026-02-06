from jax import numpy as jnp

from parameters import DT, IMPLICITNESS, NCORR, RE, STEPTOL
from rhs import get_rhs_no_lapl
from transform import INV_LAPL, KVEC, LAPL, QX, QY, QZ, spec_to_phys_vector
from velocity import get_norm


def timestep(velocity_spec, velocity_phys):
    rhs_no_lapl_prev = get_rhs_no_lapl(velocity_phys)
    # print("normrhs", get_norm(rhs_no_lapl_prev))

    prediction = (
        velocity_spec * (1 / DT + (1 - IMPLICITNESS) * LAPL / RE) + rhs_no_lapl_prev
    ) / (1 / DT - IMPLICITNESS * LAPL / RE)

    for c in range(NCORR):
        norm_prediction = get_norm(prediction)
        prediction_phys = spec_to_phys_vector(prediction)
        rhs_no_lapl_next = get_rhs_no_lapl(prediction_phys)

        correction = (
            IMPLICITNESS
            * (rhs_no_lapl_next - rhs_no_lapl_prev)
            / (1 / DT - IMPLICITNESS * LAPL / RE)
        )
        prediction += correction

        error = get_norm(correction)
        # print(error, norm_prediction)

        rhs_no_lapl_prev = rhs_no_lapl_next

        if error / norm_prediction < STEPTOL:
            correction_divergence = INV_LAPL * jnp.sum(KVEC * prediction, axis=0)
            prediction += correction_divergence * KVEC

            # Galilean invariance: set mean to 0
            velocity_spec_next = jnp.where(
                (QX == 0) & (QY == 0) & (QZ == 0), 0, prediction
            )

            # TODO: apply a bunch of symmetries

            velocity_phys_next = spec_to_phys_vector(velocity_spec_next)

            break

        elif c == NCORR - 1:
            exit("Timestep did not converge.")

    return velocity_spec_next, velocity_phys_next
