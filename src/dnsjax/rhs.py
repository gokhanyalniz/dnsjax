"""Right-hand side of the momentum equation: nonlinear term and pressure projection.

The nonlinear term uses the *rotational form* around a base flow **U**:

    NL = u' x omega' + u' x curl(U) + U x omega' + U x curl(U)

where ``u'`` is the perturbation velocity and ``omega' = curl(u')``.
The four terms arise from expanding ``(u' + U) x curl(u' + U)``.

The pressure is determined algebraically by projecting the nonlinear
term onto the divergence-free subspace:

    rhs = NL - grad( lapl^{-1}( div(NL) ) )

This module is shared across all triply-periodic flows.  For wall-bounded
flows the pressure solve will be replaced by the influence-matrix method.
"""

from jax import Array

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
    velocity_spec: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    base_flow: Array,
    curl_base_flow: Array,
    nonlin_base_flow: Array,
) -> Array:
    """Compute the rotational-form nonlinear term in spectral space.

    Evaluates the four cross-product contributions in physical space on
    the 3/2-oversampled grid (to dealias the quadratic nonlinearity) and
    transforms the result back to spectral space.

    Cost: 3 inverse FFTs (velocity, vorticity components) + 3 forward
    FFTs (nonlinear term components).

    Parameters
    ----------
    velocity_spec:
        Perturbation velocity in spectral space, shape ``(3, *spec_shape)``.
    kx, ky, kz:
        Wavenumber arrays (broadcastable to ``spec_shape``).
    base_flow:
        Base-flow velocity **U** in physical space, shape ``(3, ny_p, 1, 1)``.
    curl_base_flow:
        ``curl(U)`` in physical space, same shape.
    nonlin_base_flow:
        ``U x curl(U)`` in physical space, same shape.

    Returns
    -------
    :
        Nonlinear term in spectral space, shape ``(3, *spec_shape)``.
    """

    velocity_phys = spec_to_phys(velocity_spec)  # 3 inverse FFTs

    vorticity_phys = spec_to_phys(curl(velocity_spec, kx, ky, kz))
    nonlin_phys = cross(velocity_phys, vorticity_phys)

    # u' x curl(U) + U x omega' + U x curl(U)

    nonlin_phys = nonlin_phys.at[...].add(
        cross(velocity_phys, curl_base_flow)
        + cross(base_flow, vorticity_phys)
        + nonlin_base_flow
    )

    nonlin = phys_to_spec(nonlin_phys)  # 3 forward FFTs

    return nonlin


def get_rhs_no_lapl(
    velocity_spec: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    inv_lapl: Array,
    base_flow: Array,
    curl_base_flow: Array,
    nonlin_base_flow: Array,
) -> Array:
    """Compute the momentum RHS with the pressure gradient removed.

    The pressure is determined from the Poisson equation
    ``lapl(p) = div(NL)`` and subtracted:
    ``rhs = NL - grad(lapl^{-1}(div(NL)))``.

    The Laplacian term is *not* included in the returned RHS -- it is
    handled implicitly by the time-stepping scheme (see
    :mod:`dnsjax.timestep`).

    Parameters
    ----------
    velocity_spec:
        Perturbation velocity in spectral space, shape ``(3, *spec_shape)``.
    kx, ky, kz:
        Wavenumber arrays.
    inv_lapl:
        Inverse spectral Laplacian ``1/(-k^2)``, with the k=0 mode set to 0.
    base_flow, curl_base_flow, nonlin_base_flow:
        Base-flow arrays in physical space (see :func:`get_nonlin`).

    Returns
    -------
    :
        Divergence-free RHS (without the Laplacian / viscous term),
        shape ``(3, *spec_shape)``.
    """

    nonlin = get_nonlin(
        velocity_spec, kx, ky, kz, base_flow, curl_base_flow, nonlin_base_flow
    )

    # Poisson problem for pressure: lapl(p) = div(NL)
    lapl_pressure = divergence(nonlin, kx, ky, kz)

    # Subtract pressure gradient to enforce incompressibility
    rhs_no_lapl = nonlin - gradient(
        inverse_laplacian(lapl_pressure, inv_lapl), kx, ky, kz
    )

    return rhs_no_lapl
