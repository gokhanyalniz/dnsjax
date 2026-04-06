from dataclasses import dataclass

from jax import numpy as jnp

from operators import (
    cross,
    curl,
    divergence,
    gradient,
    inverse_laplacian,
    phys_to_spec,
    spec_to_phys,
)
from parameters import cartesian_systems, padded_res, params, periodic_systems
from sharding import sharding


@dataclass
class Force:
    # Physics
    if params.phys.system in ["kolmogorov", "waleffe"]:
        if params.res.ny < 4:
            sharding.print(
                f"{params.phys.system} cannot work "
                "with less than 4 modes in y."
            )
            sharding.exit(code=1)

        on = True
        amplitude = jnp.pi**2 / (4 * params.phys.re)

        ic_f = 0  # Forced component
        qf = 1  # Forcing harmonic

        forced_modes = ((ic_f, ic_f), (qf, -qf), (0, 0), (0, 0))

        if params.phys.system == "kolmogorov":
            unit_force = jnp.array([-0.5j, 0.5j], dtype=sharding.complex_type)
        elif params.phys.system == "waleffe":
            unit_force = jnp.array([0.5, 0.5], dtype=sharding.complex_type)

            sharding.print(
                "Waleffe flow needs the Ry symmetry,"
                "which is not yet implemented."
            )
            raise NotImplementedError

    else:
        on = False
        unit_force = None


@dataclass
class Flow:
    _system = params.phys.system
    if _system in periodic_systems:
        # This includes the Nyquist mode, unlike all other complex arrays
        base_flow_complex = jnp.zeros(
            padded_res.ny_padded // 2 + 1, dtype=sharding.complex_type
        )
        if _system in ["kolmogorov", "waleffe"]:
            qf = 1
            if _system == "kolmogorov":
                unit_force = -0.5j
            elif _system == "waleffe":
                unit_force = 0.5
            base_flow_complex = base_flow_complex.at[qf].add(unit_force)

            ekin_lam = 1 / 4
            input_lam = jnp.pi**2 / (8 * params.phys.re)
            dissip_lam = input_lam

        else:
            ekin_lam = 0
            input_lam = 0

        dy_base_flow_complex = (
            1j * (2 * jnp.pi / params.geo.ly) * base_flow_complex
        )
        base_flow = jnp.fft.irfft(
            base_flow_complex, n=padded_res.ny_padded, norm="forward"
        )[:, None, None]
        dy_base_flow = jnp.fft.irfft(
            dy_base_flow_complex, n=padded_res.ny_padded, norm="forward"
        )[:, None, None]
        nonlin_base_flow = base_flow * dy_base_flow
        ys = None  # We don't need to do any y-based operation

        # Make sure we never use these:
        del base_flow_complex
        del dy_base_flow_complex
    elif _system in cartesian_systems:
        ys = -jnp.cos(
            jnp.arange(params.res.ny, dtype=sharding.float_type)
            * jnp.pi
            / (params.res.ny - 1)
        )

        if _system == "plane-couette":
            base_flow = jnp.copy(ys)
            dy_base_flow = jnp.ones(params.res.ny, dtype=sharding.float_type)
            ekin_lam = 1 / 6
            input_lam = 1 / params.phys.re
            dissip_lam = 1 / params.phys.re
            pres_grad_lam = 0
            ubulk_lam = 0
        elif _system == "plane-channel":
            base_flow = 1 - ys**2
            dy_base_flow = -2 * ys
            ekin_lam = 4 / 15
            input_lam = 4 / (3 * params.phys.re)
            dissip_lam = 4 / (3 * params.phys.re)
            pres_grad_lam = 2 / params.phys.re
            ubulk_lam = 2 / 3
        else:
            raise NotImplementedError
    elif _system == "pipe":
        ys = -jnp.cos(
            jnp.arange(2 * params.res.ny + 1, dtype=sharding.float_type)
            * jnp.pi
            / (2 * params.res.ny)
        )[-params.res.ny :]
        base_flow = 1 - ys**2
        dy_base_flow = -2 * ys
        ekin_lam = 1 / 6
        input_lam = 2 / params.phys.re
        dissip_lam = 2 / params.phys.re
        pres_grad_lam = 4 / params.phys.re
        ubulk_lam = 1 / 2

    else:
        raise NotImplementedError


force = Force()
flow = Flow()


def get_nonlin(velocity_spec, kx, ky, kz):

    velocity_phys = spec_to_phys(velocity_spec)  # 3 FFTs

    vorticity_phys = spec_to_phys(curl(velocity_spec, kx, ky, kz))
    nonlin_phys = cross(velocity_phys, vorticity_phys)

    if params.phys.system in periodic_systems:
        # U' \hat{z} x u + U \hat{x} x \omega + U U' \hat{y}

        nonlin_phys = nonlin_phys.at[0].add(
            -flow.dy_base_flow * velocity_phys[1]
        )
        nonlin_phys = nonlin_phys.at[1].add(
            flow.dy_base_flow * velocity_phys[0]
            - flow.base_flow * vorticity_phys[2]
            + flow.nonlin_base_flow
        )
        nonlin_phys = nonlin_phys.at[2].add(flow.base_flow * vorticity_phys[1])

    nonlin = phys_to_spec(nonlin_phys)

    return nonlin


def get_rhs_no_lapl(
    velocity_spec,
    kx,
    ky,
    kz,
    inv_lapl,
):

    nonlin = get_nonlin(velocity_spec, kx, ky, kz)

    # Poisson problem for pressure
    lapl_pressure = divergence(nonlin, kx, ky, kz)

    # Add pressure gradient
    rhs_no_lapl = nonlin - gradient(
        inverse_laplacian(lapl_pressure, inv_lapl), kx, ky, kz
    )

    return rhs_no_lapl
