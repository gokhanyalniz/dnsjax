"""Shared nonlinear term computation (rotational form around base flow).

The nonlinear term uses the *rotational form* around a base flow **U**:

    $$
    NL = \\mathbf{u}' \\times \\boldsymbol{\\omega}' +
    \\mathbf{u}' \\times \\nabla \\times \\mathbf{U} +
    \\mathbf{U} \\times \\boldsymbol{\\omega}' +
    \\mathbf{U} \\times \\nabla \\times \\mathbf{U}
    $$

where `$\\mathbf{u}'$` is the perturbation velocity and
`$\\boldsymbol{\\omega}' = \\nabla \\times \\mathbf{u}'$`.
The four terms arise from expanding
`$(\\mathbf{u}' + \\mathbf{U}) \\times \\nabla \\times
(\\mathbf{u}' + \\mathbf{U})$`.

The transforms (``spec_to_phys``, ``phys_to_spec``) and the ``curl``
operator are provided as callables so that this module works with both
3D FFTs (triply-periodic flows) and 2D FFTs (wall-bounded flows).

The pressure projection is *not* performed here -- it is
geometry-specific (algebraic in ``geometries.triply_periodic``,
influence-matrix method in ``geometries.cartesian``) and lives
in the corresponding geometry module.
"""

from collections.abc import Callable

from jax import Array
from jax import numpy as jnp

from .operators import cross


def get_nonlin(
    velocity_spec: Array,
    base_flow: Array,
    curl_base_flow: Array,
    nonlin_base_flow: Array,
    spec_to_phys_fn: Callable[[Array], Array],
    phys_to_spec_fn: Callable[[Array], Array],
    curl_fn: Callable[[Array], Array],
) -> Array:
    """Compute the rotational-form nonlinear term in spectral space.

    Evaluates the four cross-product contributions in physical space on
    the dealiased (3/2-oversampled) grid and transforms the result back
    to spectral space.

    Cost: 3 inverse FFTs (velocity, vorticity components) + 3 forward
    FFTs (nonlinear term components).

    Parameters
    ----------
    velocity_spec:
        Perturbation velocity in spectral space, shape ``(3, *spec_shape)``.
    base_flow:
        Base-flow velocity **U** in physical space,
        shape ``(3, ny_phys, 1, 1)`` where ``ny_phys`` is
        ``ny_padded`` (periodic) or ``ny`` (wall-bounded).
    curl_base_flow:
        `$\\nabla \\times \\mathbf{U}$` in physical space, same shape.
    nonlin_base_flow:
        `$\\mathbf{U} \\times \\nabla \\times \\mathbf{U}$` in physical
        space, same shape.
    spec_to_phys_fn:
        Inverse FFT (spectral -> physical), vmapped over components.
    phys_to_spec_fn:
        Forward FFT (physical -> spectral), vmapped over components.
    curl_fn:
        Spectral curl operator ``velocity_spec -> curl_spec``, with
        wavenumbers already bound.

    Returns
    -------
    :
        Nonlinear term in spectral space, shape ``(3, *spec_shape)``.
    """

    # Batch velocity (3) + vorticity (3) into one transform call
    # so that the FFT reshard happens once for all 6 fields.
    vorticity_spec = curl_fn(velocity_spec)
    combined_phys = spec_to_phys_fn(
        jnp.concatenate([velocity_spec, vorticity_spec])
    )
    velocity_phys = combined_phys[:3]
    vorticity_phys = combined_phys[3:]
    nonlin_phys = cross(velocity_phys, vorticity_phys)

    # u' x curl(U) + U x omega' + U x curl(U)

    nonlin_phys = nonlin_phys.at[...].add(
        cross(velocity_phys, curl_base_flow)
        + cross(base_flow, vorticity_phys)
        + nonlin_base_flow
    )

    nonlin = phys_to_spec_fn(nonlin_phys)  # 3 forward FFTs

    return nonlin
