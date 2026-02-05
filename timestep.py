from jax import numpy as jnp

from fft import INV_LAPL, KVEC, KX, KY, KZ, LAPL, spect_to_phys_vector
from parameters import DT, IMPLICITNESS, NCORR, RE, STEPTOL
from rhs import compute_rhs_no_lapl
from vfield import norm


def timestep(vfieldx, vfieldk):
    # TODO: Check the necessity of the Fourier transforms
    rhs_no_lapl_prev = compute_rhs_no_lapl(vfieldx)

    # Prediction: u(n+1)_1 = ((1/dt + (1 - implicitness) L) u(n) + N(n)) /
    #                        (1/dt - implicitness L)
    prefieldk = (
        vfieldk * (1 / DT + (1 - IMPLICITNESS) * LAPL / RE) + rhs_no_lapl_prev
    ) / (1 / DT - IMPLICITNESS * LAPL / RE)

    for c in range(NCORR):
        pred_norm = norm(prefieldk)
        prefieldx = spect_to_phys_vector(prefieldk)
        nonlinterm_next = compute_rhs_no_lapl(prefieldx)

        # Now we have N(n+1)^c in state(:,:,:,1:3)

        corfieldk = (
            IMPLICITNESS
            * (nonlinterm_next - rhs_no_lapl_prev)
            / (1 / DT - IMPLICITNESS * LAPL / RE)
        )
        prefieldk += corfieldk

        error = norm(corfieldk)

        rhs_no_lapl_prev = nonlinterm_next

        if error / pred_norm < STEPTOL:
            # accept step

            dp = INV_LAPL * jnp.sum(KVEC * prefieldk, axis=0)
            prefieldk += dp * KVEC

            # Mode 0 kept 0
            # TODO: Check if needed
            vfieldk_out = jnp.where((KX == 0) & (KY == 0) & (KZ == 0), 0, prefieldk)

            # TODO: apply a bunch of symmetries

            vfieldx_out = spect_to_phys_vector(vfieldk_out)

            break

        elif c == NCORR - 1:
            exit("Timestep did not converge.")

    return vfieldx_out, vfieldk_out
