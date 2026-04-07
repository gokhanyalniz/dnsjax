from .operators import (
    cross,
    curl,
    divergence,
    gradient,
    inverse_laplacian,
    phys_to_spec,
    spec_to_phys,
)


def get_nonlin(
    velocity_spec, kx, ky, kz, base_flow, curl_base_flow, nonlin_base_flow
):

    velocity_phys = spec_to_phys(velocity_spec)  # 3 FFTs

    vorticity_phys = spec_to_phys(curl(velocity_spec, kx, ky, kz))
    nonlin_phys = cross(velocity_phys, vorticity_phys)

    # u x curl{U} + U x \omega + U x curl(U)

    nonlin_phys = nonlin_phys.at[...].add(
        cross(velocity_phys, curl_base_flow)
        + cross(base_flow, vorticity_phys)
        + nonlin_base_flow
    )

    nonlin = phys_to_spec(nonlin_phys)

    return nonlin


def get_rhs_no_lapl(
    velocity_spec,
    kx,
    ky,
    kz,
    inv_lapl,
    base_flow,
    curl_base_flow,
    nonlin_base_flow,
):

    nonlin = get_nonlin(
        velocity_spec, kx, ky, kz, base_flow, curl_base_flow, nonlin_base_flow
    )

    # Poisson problem for pressure
    lapl_pressure = divergence(nonlin, kx, ky, kz)

    # Add pressure gradient
    rhs_no_lapl = nonlin - gradient(
        inverse_laplacian(lapl_pressure, inv_lapl), kx, ky, kz
    )

    return rhs_no_lapl
