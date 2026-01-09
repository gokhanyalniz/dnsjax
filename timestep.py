from jax import numpy as jnp

from fft import LAPL, vk2x
from parameters import IMPLICITNESS, INVTDT, NCORR, RE, STEPTOL
from rhs import nonlin_term
from vfield import norm


def precorr(vel_vfieldx_in, vel_vfieldk_in):
    fvel_vfieldk = nonlin_term(vel_vfieldx_in)

    timestep_nonlinterm_prev = jnp.copy(fvel_vfieldk)

    # add the linear term to the rhs for others's use
    fvel_vfieldk += (LAPL / RE) * vel_vfieldk_in

    # Prediction: u(n+1)_1 = ((1/dt + (1 - implicitness) L) u(n) + N(n)) /
    #                        (1/dt - implicitness L)
    timestep_prefieldk = (
        vel_vfieldk_in * (INVTDT + (1 - IMPLICITNESS) * LAPL / RE)
        + timestep_nonlinterm_prev
    ) / (INVTDT - IMPLICITNESS * LAPL / RE)

    for c in range(NCORR):
        pred_norm = norm(timestep_prefieldk)
        timestep_prefieldx = vk2x(timestep_prefieldk)
        timestep_nonlinterm_next = nonlin_term(timestep_prefieldx)

        # Now we have N(n+1)^c in state(:,:,:,1:3)

        timestep_corfieldk = (
            IMPLICITNESS
            * (timestep_nonlinterm_next - timestep_nonlinterm_prev)
            / (INVTDT - IMPLICITNESS * LAPL / RE)
        )
        timestep_prefieldk += timestep_corfieldk

        error = norm(timestep_corfieldk)

        timestep_nonlinterm_prev = timestep_nonlinterm_next

        if error / pred_norm < STEPTOL:
            # accept step
            vel_vfieldk_out = timestep_prefieldk

            # apply a bunch of symmetries

            vel_vfieldx_out = vk2x(vel_vfieldk_out)

            break

    vel_vfieldx_out = None
    vel_vfieldk_out = None
    fvel_vfieldk = None
    return vel_vfieldx_out, vel_vfieldk_out, fvel_vfieldk
