r"""Geometry-aware derivatives and integrals of snapshot fields.

The differential operators (:func:`derivative`, :func:`gradient`,
:func:`divergence`, :func:`curl`) work on **spectral** fields in the
snapshot-native layout and basis returned by
:func:`dnsjax.analysis.read_state` (``return_spectral=True``):
``(u_z, u_r, u_θ)`` for cylindrical/annular, ``(u_x, u_y, u_z)``
otherwise.  Differentiation is the exact ``× i k`` along Fourier axes
(the wavenumbers from ``spectral_coords``) and a finite-difference ``D1``
along the wall-normal/radial grid axis, and returns spectral fields.

:func:`integrate` works on **physical** fields and ``physical_coords``
(nonlinear integrands such as ``|u|²`` are formed in physical space;
this avoids Parseval/Nyquist normalization pitfalls): a uniform
``L / n`` rule along Fourier axes and finite-difference quadrature
(with the radial Jacobian for cylindrical/annular) along the wall-normal
axis.

:func:`to_physical` / :func:`to_spectral` bridge the two spaces.

Directions are named per geometry: ``"x"/"y"/"z"`` (cartesian,
triply-periodic), ``"r"/"z"/"theta"`` (cylindrical/annular, ``z``
axial), unless the flow names its axes itself
(:attr:`~dnsjax.analysis._core.GeometryInfo.name`).  Wall-normal
derivatives/integrals require the full wall-normal grid: do not subset
``wall_normal_points`` first.

A cylindrical flow bent along its axis carries an axial scale factor
`$h(r, \theta)$` (:attr:`~dnsjax.analysis._core.GeometryInfo.metric`):
scale factors `$(1, r, h)$` along `$(r, \theta, z)$`.
:func:`derivative` and :func:`gradient` stay coordinate partials
(`$\partial_z$`, not the physical `$h^{-1}\partial_z$`), :func:`curl`
and :func:`divergence` take the metric-weighted forms and divide the
weight out, and :func:`integrate` takes the volume element `$r h$`.
"""

from __future__ import annotations

import numpy as np

from ..fd import (
    axis_augmented_weights,
    build_integration_weights,
    clenshaw_curtis_weights,
    is_cgl_grid,
)
from ..flow_spec import VELOCITY_PARITY
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


#: The pipe's velocity labels, accepted for every cylindrical flow
#: beside its own (:func:`_parity_classes`).
_VELOCITY_LABELS = ("u_z", "u_r", "u_theta")


def _parity_classes(info) -> dict[str, str]:
    r"""Component label -> radial parity class, for a pipe snapshot.

    A component's axis parity is `$(-1)^{m+s}$`, `$s$` the number of
    its radial and azimuthal indices (each flips sign under the axis
    reflection): an even count gives the ``"even"`` class `$(-1)^m$`
    and an odd one the ``"odd"`` class `$(-1)^{m+1}$`.  The labels are
    the flow's stored components (:attr:`GeometryInfo.components`, with
    their :attr:`GeometryInfo.component_parity`) and the velocity's,
    each also spelled without underscores (``uz``, ``utheta``), plus
    ``uth``.
    """
    pairs = [
        *zip(_VELOCITY_LABELS, VELOCITY_PARITY, strict=True),
        *zip(info.components, info.component_parity, strict=True),
    ]
    out = {}
    for label, even in pairs:
        out[label] = out[label.replace("_", "")] = "even" if even else "odd"
    out["uth"] = out["u_theta"]
    return out


def _resolve_parity(info, ax, cylindrical_parity):
    r"""Radial parity class for a derivative along on-disk *ax*.

    ``None`` for Fourier axes and for the cartesian/annular grid axis
    (a plain ``D1``).  The pipe's radial axis is parity-dependent at the
    axis, so *cylindrical_parity* must name the component being
    differentiated -- any label of :func:`_parity_classes`.
    """
    if info.kind[ax] != "grid" or info.family != "cylindrical":
        return None
    classes = _parity_classes(info)
    if cylindrical_parity is None:
        raise ValueError(
            "pipe radial derivatives are parity-dependent at the axis; "
            "pass cylindrical_parity naming the component, one of "
            f"{info.components}."
        )
    try:
        return classes[str(cylindrical_parity).lower()]
    except KeyError:
        raise ValueError(
            f"cylindrical_parity={cylindrical_parity!r} invalid; use "
            f"one of {sorted(classes)}."
        ) from None


def derivative(field, direction, params, coords, cylindrical_parity=None):
    r"""First derivative of a spectral component along *direction*.

    *coords* is the ``spectral_coords`` tuple from
    :func:`~dnsjax.analysis.read_state`.  Returns a spectral array.
    *cylindrical_parity* (``"u_z"``, ``"u_r"``, ``"u_theta"`` or another
    stored component's label) is required only for a **pipe** radial
    (``"r"``) derivative, naming the component being differentiated;
    ignored otherwise.
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
    )


def gradient(component, params, coords, cylindrical_parity=None):
    r"""Gradient of a spectral scalar component.

    Returns a 3-tuple of spectral partials in axis order
    (``∂/∂``\ *axis0*, *axis1*, *axis2*).  *cylindrical_parity* names
    the component (as for :func:`derivative`) and is required only for
    the **pipes** (their radial partial, axis 0, is parity-dependent);
    ignored otherwise.
    """
    info = _core.geometry_info(params)
    arr = np.asarray(component)
    fd = params.res.get("fd_order")
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

    With an axial scale factor `$h$` (:attr:`GeometryInfo.metric`)
    and `$\nabla_0\cdot$` that straight expansion,

    .. math::
        h^2\,\nabla\cdot\mathbf{u}
        = h\,\nabla_0\cdot(u_z,\, h u_r,\, h u_\theta) ,

    whose right-hand side is the continuity row of a solver carrying
    the metric, the quantity its corrector drives to zero.  This forms
    that row and divides the `$h^2$` out, every product the metric's
    truncating one (:meth:`AxialMetric.product`), so it vanishes
    exactly where the solver's does.  Dividing
    `$\nabla_0\cdot(u_z, h u_r, h u_\theta)$` by `$h$` alone would
    not: the truncated product with `$h$` drops the top mode's outflow.
    """
    info = _core.geometry_info(params)
    fd = params.res.get("fd_order")

    if info.family in ("cylindrical", "annular"):
        u_z, u_r, u_th = (np.asarray(f) for f in field)
        r = np.asarray(coords[0], dtype=float)
        rinv = _core._broadcast_along(1.0 / r, 0)
        metric = info.metric
        if metric is not None:
            u_r, u_th = (metric.product(f, r, info, 1) for f in (u_r, u_th))
        p_ur = "odd" if info.family == "cylindrical" else None
        d_ur = _core.radial_derivative(
            u_r,
            r,
            fd,
            info,
            parity=p_ur,
        )
        d_th = _core.fourier_derivative(u_th, 1, coords[1])  # im u_θ
        d_z = _core.fourier_derivative(u_z, 2, coords[2])  # i k_z u_z
        div = d_ur + rinv * u_r + rinv * d_th + d_z
        if metric is not None:
            row = metric.product(div, r, info, 1)
            div = metric.product(row, r, info, -2)
        return div

    # cartesian / triply-periodic: sum_i ∂u_i/∂x_i over the leading
    # three stored components, the velocity by contract (a flow storing
    # more labels the rest, which are not velocities).
    total = None
    for idx, label in enumerate(info.components[:3]):
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

    With an axial scale factor `$h$` (:attr:`GeometryInfo.metric`),
    that expansion applied to `$(h u_z, u_r, u_\theta)$` is
    `$(\omega_z, h\,\omega_r, h\,\omega_\theta)$` for the scale
    factors `$(1, r, h)$`.  This forms it and divides the `$h$` out of
    the transverse pair (:meth:`AxialMetric.product`).
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

        def d_r(f, parity):  # ∂/∂r = D1 (parity-reduced for the pipe)
            return _core.radial_derivative(f, r, fd, info, parity=parity)

        p_uz = "even" if cyl else None
        p_uth = "odd" if cyl else None
        metric = info.metric
        if metric is not None:
            u_z = metric.product(u_z, r, info, 1)  # h u_z
        w_r = rinv * d_th(u_z) - d_z(u_th)
        w_th = d_z(u_r) - d_r(u_z, p_uz)
        w_z = d_r(u_th, p_uth) + rinv * u_th - rinv * d_th(u_r)
        if metric is not None:
            w_r, w_th = (metric.product(f, r, info, -1) for f in (w_r, w_th))
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
            # here.  At fd_order, not the solver's positive order: an
            # integral of a physical field needs no definite norm.
            return axis_augmented_weights(grid, p)
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

    An axial scale factor `$h$` (:attr:`GeometryInfo.metric`) rides
    with the axial direction: an integral over `$z$` measures the
    arclength `$h\,dz$`, so the volume element is
    `$r h\,dr\,d\theta\,dz$` while an integral over `$(r, \theta)$`
    alone keeps the metric-free cross-section element
    `$r\,dr\,d\theta$` (that of `$u_z$` is the mass flux).  An
    integral over `$z$` alone is therefore one along the axial line
    through `$(r, \theta)$`, of length `$h L_z$`: divide by that, not
    by `$L_z$`, for the axial mean (or take ``np.mean`` along the
    axis).

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
    # An axial scale factor h(r, theta), constant along the axis
    # (axis 2), multiplies the field before any axis drops whenever
    # that axis is integrated.
    metric = None
    if info.metric is not None and 2 in axes:
        r = np.asarray(coords[0], dtype=float)[:, None, None]
        theta = np.asarray(coords[1], dtype=float)[None, :, None]
        metric = info.metric.scale(r, theta)

    def _reduce(f):
        out = np.asarray(f)
        if metric is not None:
            out = out * metric
        for ax, w in weights:
            out = np.tensordot(w, out, axes=([0], [ax]))
        return out

    if isinstance(field, (tuple, list)):
        return tuple(_reduce(f) for f in field)
    return _reduce(field)
