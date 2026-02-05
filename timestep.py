from jax import numpy as jnp

from fft import INV_LAPL, KVEC, KX, KY, KZ, LAPL, vk2x
from parameters import DT, IMPLICITNESS, NCORR, RE, STEPTOL
from rhs import nonlin_term
from vfield import norm


def precorr(vfieldx, vfieldk):
    # TODO: Check the necessity of the Fourier transforms
    nonlinterm_prev = nonlin_term(vfieldx)

    # Prediction: u(n+1)_1 = ((1/dt + (1 - implicitness) L) u(n) + N(n)) /
    #                        (1/dt - implicitness L)
    prefieldk = (
        vfieldk * (1 / DT + (1 - IMPLICITNESS) * LAPL / RE) + nonlinterm_prev
    ) / (1 / DT - IMPLICITNESS * LAPL / RE)

    for c in range(NCORR):
        pred_norm = norm(prefieldk)
        prefieldx = vk2x(prefieldk)
        nonlinterm_next = nonlin_term(prefieldx)

        # Now we have N(n+1)^c in state(:,:,:,1:3)

        corfieldk = (
            IMPLICITNESS
            * (nonlinterm_next - nonlinterm_prev)
            / (1 / DT - IMPLICITNESS * LAPL / RE)
        )
        prefieldk += corfieldk

        error = norm(corfieldk)

        nonlinterm_prev = nonlinterm_next

        if error / pred_norm < STEPTOL:
            # accept step

            dp = INV_LAPL * jnp.sum(KVEC * prefieldk, axis=0)
            prefieldk += dp * KVEC

            # Mode 0 kept 0
            # TODO: Check if needed
            vfieldk_out = jnp.where((KX == 0) & (KY == 0) & (KZ == 0), 0, prefieldk)

            # TODO: apply a bunch of symmetries

            vfieldx_out = vk2x(vfieldk_out)

            break

        elif c == NCORR - 1:
            exit("Timestep did not converge.")

    return vfieldx_out, vfieldk_out
