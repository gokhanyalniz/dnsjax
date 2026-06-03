r"""Perturbation nonlinear term (rotational form).

The nonlinear term uses the *rotational form* of the
perturbation Navier--Stokes equation around a base flow
`$\mathbf{U}$`:

    $$
    NL = \mathbf{u}' \times \boldsymbol{\omega}' +
    \mathbf{u}' \times \nabla \times \mathbf{U} +
    \mathbf{U} \times \boldsymbol{\omega}'
    $$

where `$\mathbf{u}'$` is the perturbation velocity and
`$\boldsymbol{\omega}' = \nabla \times \mathbf{u}'$`.
The three terms arise from expanding
`$(\mathbf{u}' + \mathbf{U}) \times \nabla \times
(\mathbf{u}' + \mathbf{U})$` and subtracting the base-flow
self-interaction `$\mathbf{U} \times \nabla \times \mathbf{U}
= \nabla(|\mathbf{U}|^2/2)$`, which is a pure gradient
absorbed by the pressure.

The transforms (``spec_to_phys``, ``phys_to_spec``) and the
``curl`` operator are provided as callables so that this module
works with both 3D FFTs (triply-periodic flows) and 2D FFTs
(wall-bounded flows).

The pressure projection is *not* performed here -- it is
geometry-specific (algebraic in
``geometries.triply_periodic``, influence-matrix method in
``geometries.wall_bounded.cartesian``) and lives
in the corresponding geometry module.
"""

from collections.abc import Callable

from jax import Array
from jax import numpy as jnp


def _fused_nonlinear(
    u: Array,
    omega: Array,
    U: Array,
    curl_U: Array,
) -> Array:
    r"""Compute the three perturbation cross-product terms.

    Fuses `$\mathbf{u}' \times \boldsymbol{\omega}'
    + \mathbf{u}' \times \nabla \times \mathbf{U}
    + \mathbf{U} \times \boldsymbol{\omega}'$`
    into a single ``jnp.array`` expression per output
    component, eliminating intermediate concatenation and
    scatter kernels.

    Parameters
    ----------
    u:
        Perturbation velocity in physical space, ``(3, ...)``.
    omega:
        Perturbation vorticity in physical space, ``(3, ...)``.
    U:
        Base-flow velocity in physical space, ``(3, ny, 1, 1)``.
    curl_U:
        `$\nabla \times \mathbf{U}$`, ``(3, ny, 1, 1)``.

    Returns
    -------
    :
        Nonlinear term in physical space, ``(3, ...)``.
    """
    u0, u1, u2 = u[0], u[1], u[2]
    w0, w1, w2 = omega[0], omega[1], omega[2]
    U0, U1, U2 = U[0], U[1], U[2]
    cU0, cU1, cU2 = curl_U[0], curl_U[1], curl_U[2]

    return jnp.array(
        [
            (u1 * w2 - u2 * w1) + (u1 * cU2 - u2 * cU1) + (U1 * w2 - U2 * w1),
            (u2 * w0 - u0 * w2) + (u2 * cU0 - u0 * cU2) + (U2 * w0 - U0 * w2),
            (u0 * w1 - u1 * w0) + (u0 * cU1 - u1 * cU0) + (U0 * w1 - U1 * w0),
        ]
    )


def get_nonlin(
    velocity_spec: Array,
    base_flow: Array,
    curl_base_flow: Array,
    spec_to_phys_fn: Callable[[Array], Array],
    phys_to_spec_fn: Callable[[Array], Array],
    curl_fn: Callable[[Array], Array],
) -> Array:
    r"""Compute the perturbation nonlinear term in spectral space.

    Evaluates the three cross-product contributions in physical
    space on the dealiased (3/2-oversampled) grid and transforms
    the result back to spectral space.

    Cost: 3 inverse FFTs (velocity, vorticity components) + 3
    forward FFTs (nonlinear term components).

    Parameters
    ----------
    velocity_spec:
        Perturbation velocity in spectral space,
        shape ``(3, *spec_shape)``.
    base_flow:
        Base-flow velocity `$\mathbf{U}$` in physical space,
        shape ``(3, ny_phys, 1, 1)`` where ``ny_phys`` is
        ``ny_padded`` (periodic) or ``ny + ny_y_pad``
        (wall-bounded).
    curl_base_flow:
        `$\nabla \times \mathbf{U}$` in physical space,
        same shape.
    spec_to_phys_fn:
        Inverse FFT (spectral -> physical), vmapped over
        components.
    phys_to_spec_fn:
        Forward FFT (physical -> spectral), vmapped over
        components.
    curl_fn:
        Spectral curl operator
        ``velocity_spec -> curl_spec``, with wavenumbers
        already bound.

    Returns
    -------
    :
        Nonlinear term in spectral space,
        shape ``(3, *spec_shape)``.
    """

    # Batch velocity (3) + vorticity (3) into one transform call
    # so that the FFT reshard happens once for all 6 fields.
    vorticity_spec = curl_fn(velocity_spec)
    combined_phys = spec_to_phys_fn(
        jnp.concatenate([velocity_spec, vorticity_spec])
    )
    velocity_phys = combined_phys[:3]
    vorticity_phys = combined_phys[3:]

    nonlin_phys = _fused_nonlinear(
        velocity_phys,
        vorticity_phys,
        base_flow,
        curl_base_flow,
    )

    return phys_to_spec_fn(nonlin_phys)
