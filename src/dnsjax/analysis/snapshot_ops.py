r"""Geometry-aware derivatives and integrals of snapshot fields.

The differential operators (:func:`derivative`, :func:`gradient`,
:func:`divergence`, :func:`curl`) work on **spectral** fields in the
snapshot-native layout and basis returned by
:func:`dnsjax.analysis.read_state` (``return_spectral=True``):
``(u_z, u_r, u_θ)`` for cylindrical/annular, ``(u_x, u_y, u_z)``
otherwise.  Differentiation is exact -- ``× i k`` along Fourier axes
(using the wavenumbers from ``spectral_coords``) and a finite-difference
``D1`` along the wall-normal/radial grid axis -- and returns spectral
fields.

:func:`integrate` works on **physical** fields and ``physical_coords``
(nonlinear integrands such as ``|u|²`` are formed in physical space;
this avoids Parseval/Nyquist normalisation pitfalls): a uniform
``L / n`` rule along Fourier axes and finite-difference quadrature
(with the radial Jacobian for cylindrical/annular) along the wall-normal
axis.

:func:`to_physical` / :func:`to_spectral` bridge the two spaces.

Directions are named per geometry: ``"x"/"y"/"z"`` (cartesian),
``"r"/"z"/"theta"`` (cylindrical/annular, ``z`` axial), ``"x"/"y"/"z"``
(triply-periodic).  Wall-normal derivatives/integrals require the full
wall-normal grid -- do not subset ``wall_normal_points`` first.
"""

from __future__ import annotations

import numpy as np

from ..fd import (
    build_integration_weights,
    clenshaw_curtis_weights,
    is_cgl_grid,
)
from . import _core
from ._core import to_physical, to_spectral  # re-exported helpers

__all__ = [
    "derivative",
    "gradient",
    "divergence",
    "curl",
    "integrate",
    "to_physical",
    "to_spectral",
]


def _use_xr2_d1(params) -> bool:
    """Did the pipe solver build the ``x = r^2`` radial ``D1``?

    True under either ``res.consistent_imm`` or ``res.pipe_axis_fit``
    (both select the ``x = r^2`` fit; only ``consistent_imm`` also
    composes ``D2``, which the ``D1``-only analysis operators ignore).
    """
    return bool(params.res.get("consistent_imm")) or bool(
        params.res.get("pipe_axis_fit")
    )


# Maps a component label to its radial parity class (pipe only):
# u_z has parity (-1)^m ("uz"); u_r and u_θ have parity (-1)^{m+1}
# ("utheta").  Accepts the long and short spellings.
_PARITY_CLASS = {
    "uz": "uz",
    "u_z": "uz",
    "ur": "utheta",
    "u_r": "utheta",
    "utheta": "utheta",
    "u_theta": "utheta",
    "uth": "utheta",
}


def _resolve_parity(info, ax, cylindrical_parity):
    r"""Radial parity class for a derivative along on-disk *ax*.

    ``None`` for Fourier axes and for the cartesian/annular grid axis
    (a plain ``D1``).  The pipe's radial axis is parity-dependent at the
    axis, so *cylindrical_parity* must name the component being
    differentiated (``"u_z"`` / ``"u_r"`` / ``"u_theta"``).
    """
    if info.kind[ax] != "grid" or info.family != "cylindrical":
        return None
    if cylindrical_parity is None:
        raise ValueError(
            "pipe radial derivatives are parity-dependent at the axis; "
            "pass cylindrical_parity='u_z', 'u_r', or 'u_theta'."
        )
    try:
        return _PARITY_CLASS[str(cylindrical_parity).lower()]
    except KeyError:
        raise ValueError(
            f"cylindrical_parity={cylindrical_parity!r} invalid; use "
            "'u_z', 'u_r', or 'u_theta'."
        ) from None


def derivative(field, direction, params, coords, cylindrical_parity=None):
    r"""First derivative of a spectral component along *direction*.

    *coords* is the ``spectral_coords`` tuple from
    :func:`~dnsjax.analysis.read_state`.  Returns a spectral array.
    *cylindrical_parity* (``"u_z"`` / ``"u_r"`` / ``"u_theta"``) is
    required only for a **pipe** radial (``"r"``) derivative, naming the
    component being differentiated; ignored otherwise.
    """
    info = _core.geometry_info(params)
    ax = info.axis_of(direction)
    parity = _resolve_parity(info, ax, cylindrical_parity)
    return _core.derivative_axis(
        np.asarray(field),
        ax,
        info,
        coords[ax],
        params.res.get("fd_order"),
        parity=parity,
        xr2_d1=_use_xr2_d1(params),
    )


def gradient(component, params, coords, cylindrical_parity=None):
    r"""Gradient of a spectral scalar component.

    Returns a 3-tuple of spectral partials in axis order
    (``∂/∂``\ *axis0*, *axis1*, *axis2*).  *cylindrical_parity* names
    the component (``"u_z"`` / ``"u_r"`` / ``"u_theta"``) and is
    required only for the **pipe** (its radial partial, axis 0, is
    parity-dependent); ignored otherwise.
    """
    info = _core.geometry_info(params)
    arr = np.asarray(component)
    fd = params.res.get("fd_order")
    xr2 = _use_xr2_d1(params)
    out = []
    for ax in range(3):
        parity = _resolve_parity(info, ax, cylindrical_parity)
        out.append(
            _core.derivative_axis(
                arr,
                ax,
                info,
                coords[ax],
                fd,
                parity=parity,
                xr2_d1=xr2,
            )
        )
    return tuple(out)


def divergence(field, params, coords):
    r"""Divergence of a spectral 3-component field (coordinate-aware).

    *field* is ``(u_x, u_y, u_z)`` (cartesian/periodic) or
    ``(u_z, u_r, u_θ)`` (cylindrical/annular).  Returns a spectral
    scalar.  The cylindrical/annular form is dnsjax's discrete
    expansion ``∂u_r/∂r + u_r/r + (im/r) u_θ + i k_z u_z`` (so it
    matches the solver's operator node-for-node, including the pipe's
    parity-reduced radial ``D1``).
    """
    info = _core.geometry_info(params)
    fd = params.res.get("fd_order")

    if info.family in ("cylindrical", "annular"):
        u_z, u_r, u_th = (np.asarray(f) for f in field)
        r = np.asarray(coords[0], dtype=float)
        rinv = _core._broadcast_along(1.0 / r, 0)
        p_ur = "utheta" if info.family == "cylindrical" else None
        d_ur = _core.radial_derivative(
            u_r,
            r,
            fd,
            info,
            parity=p_ur,
            xr2_d1=_use_xr2_d1(params),
        )
        d_th = _core.fourier_derivative(u_th, 1, coords[1])  # im u_θ
        d_z = _core.fourier_derivative(u_z, 2, coords[2])  # i k_z u_z
        return d_ur + rinv * u_r + rinv * d_th + d_z

    # cartesian / triply-periodic: sum_i ∂u_i/∂x_i
    total = None
    for idx, label in enumerate(info.components):
        direction = label.split("_", 1)[1]  # "u_x" -> "x"
        ax = info.axis_of(direction)
        term = _core.derivative_axis(
            np.asarray(field[idx]), ax, info, coords[ax], fd
        )
        total = term if total is None else total + term
    return total


def curl(field, params, coords):
    r"""Curl (vorticity) of a spectral 3-component field.

    Returns the vorticity components in the same order/basis as *field*
    (``(ω_x, ω_y, ω_z)`` or ``(ω_z, ω_r, ω_θ)``).  The
    cylindrical/annular form is dnsjax's discrete expansion
    (``ω_z = ∂u_θ/∂r + u_θ/r - (im/r) u_r`` etc.), with the pipe's
    parity-reduced radial ``D1``.
    """
    info = _core.geometry_info(params)
    fd = params.res.get("fd_order")

    if info.family in ("cylindrical", "annular"):
        u_z, u_r, u_th = (np.asarray(f) for f in field)
        r = np.asarray(coords[0], dtype=float)
        rinv = _core._broadcast_along(1.0 / r, 0)
        cyl = info.family == "cylindrical"

        def d_th(f):  # ∂/∂θ = i m
            return _core.fourier_derivative(f, 1, coords[1])

        def d_z(f):  # ∂/∂z = i k_z
            return _core.fourier_derivative(f, 2, coords[2])

        xr2 = _use_xr2_d1(params)

        def d_r(f, parity):  # ∂/∂r = D1 (parity-reduced for the pipe)
            return _core.radial_derivative(
                f, r, fd, info, parity=parity, xr2_d1=xr2
            )

        p_uz = "uz" if cyl else None
        p_uth = "utheta" if cyl else None
        w_r = rinv * d_th(u_z) - d_z(u_th)
        w_th = d_z(u_r) - d_r(u_z, p_uz)
        w_z = d_r(u_th, p_uth) + rinv * u_th - rinv * d_th(u_r)
        return (w_z, w_r, w_th)

    def d(comp, direction):
        ax = info.axis_of(direction)
        return _core.derivative_axis(
            np.asarray(comp), ax, info, coords[ax], fd
        )

    u_x, u_y, u_z = field
    w_x = d(u_z, "y") - d(u_y, "z")
    w_y = d(u_x, "z") - d(u_z, "x")
    w_z = d(u_y, "x") - d(u_x, "y")
    return (w_x, w_y, w_z)


def _axis_weights(info, ax, coords, params):
    """Quadrature weights for integrating a physical field over *ax*."""
    if info.kind[ax] == "grid":
        grid = np.asarray(coords[ax], dtype=float)
        p = int(params.res.fd_order)
        if info.family == "cylindrical":
            # Parity-agnostic full-disc rule: a *physical* radial
            # profile (at fixed theta) has no definite r-parity, so the
            # solver's per-mode spectral parity weights do not apply
            # here.  Integrate g = f*r on the axis-augmented grid with
            # the axis r=0 as a free node (g(0)=0 for any bounded f);
            # drop the axis node, fold in r_j.
            r_aug = np.concatenate([[0.0], grid])
            return build_integration_weights(r_aug, p)[1:] * grid
        if info.family == "annular":
            # Affine-mapped Clenshaw-Curtis on the CGL grid (matching
            # the solver); composite fd_order for a custom/tanh grid.
            unit = (2.0 * grid - grid[0] - grid[-1]) / (grid[-1] - grid[0])
            if is_cgl_grid(unit):
                half = 0.5 * (grid[-1] - grid[0])
                return half * clenshaw_curtis_weights(len(grid)) * grid
            return build_integration_weights(grid, p) * grid
        # Cartesian: Clenshaw-Curtis on the CGL grid (matching the
        # solver), composite fd_order otherwise.
        if is_cgl_grid(grid):
            return clenshaw_curtis_weights(len(grid))
        return build_integration_weights(grid, p)
    n = info.n[ax]
    return np.full(n, info.length[ax] / n)


def integrate(field, params, coords, directions=None):
    r"""Integrate a physical field over one or more directions.

    *coords* is the ``physical_coords`` tuple.  ``directions=None``
    integrates over all axes (scalar volume integral); a subset returns
    the field reduced over those axes (e.g. integrate the homogeneous
    directions for a wall-normal profile).  A tuple/list *field* is
    integrated component-wise.

    Examples
    --------
    Total kinetic energy ``½∫|u|² dV``::

        e = 0.5 * integrate(sum(np.abs(u) ** 2 for u in physical),
                            params, coords)
    """
    info = _core.geometry_info(params)
    if directions is None:
        axes = [2, 1, 0]
    else:
        # Descending so the original axis indices stay valid as axes
        # drop.
        axes = sorted({info.axis_of(d) for d in directions}, reverse=True)
    # Weights built once and shared across a tuple's components.
    weights = [(ax, _axis_weights(info, ax, coords, params)) for ax in axes]

    def _reduce(f):
        out = np.asarray(f)
        for ax, w in weights:
            out = np.tensordot(w, out, axes=([0], [ax]))
        return out

    if isinstance(field, (tuple, list)):
        return tuple(_reduce(f) for f in field)
    return _reduce(field)
