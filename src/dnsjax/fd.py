r"""Finite-difference and interpolation infrastructure.

Offline precomputation for the influence-matrix method (IMM) and
wall-normal grid interpolation.  All functions run at
initialisation time outside ``@jit``, so Python loops and
concrete-value branching are used directly.

Functions
---------
fornberg_weights:
    Fornberg's (1998) algorithm for FD weights on non-uniform
    grids.
build_diff_matrices:
    Assemble first- and second-derivative matrices D1, D2.
matrix_half_bandwidth:
    Measured half-bandwidth of an assembled operator, ignoring the
    rows a caller overwrites with boundary rows (the banded-storage
    size).
build_integration_weights:
    Composite polynomial quadrature weights on a non-uniform grid
    (``fd_order``-accurate; the general-purpose rule).
clenshaw_curtis_weights:
    Clenshaw-Curtis quadrature weights on a CGL grid (spectral for
    smooth integrands; used by the full-CGL Cartesian and annular
    grids, where it is exact for smooth base/mean profiles).
cgl_radial_quadrature_weights:
    Parity-specific spectral radial quadrature ``(w_even, w_odd)`` for
    a cylindrical half-CGL / rigged-CGL grid (Clenshaw-Curtis with
    weight ``r`` on ``[0,1]``, baking in the ``r=0`` reconstruction;
    positive). The pipe's spectral analogue of the Cartesian/annular
    CC; ``None`` for a non-CGL grid (caller uses the composite rule).
local_grid_spacing:
    Per-node local grid spacing (the CFL advective length scale).
tanh_two_sided_grid:
    Symmetric tanh-stretched wall-normal grid on [-1, 1]
    (clustering at both walls).
tanh_one_sided_grid:
    One-sided tanh-stretched radial grid on (0, 1] (clustering
    at the outer wall, no point at r = 0).
axis_extrapolation_weights:
    Weights to evaluate a radial field at the axis `$r = 0$` (even /
    odd / one-sided); a shared JAX-free leaf.  On a detected radial
    CGL grid the even path is the exact parity-constrained spectral
    fit (`$x = r^2$` Chebyshev), elsewhere the local Fornberg rule.
is_cgl_grid:
    Detect whether a grid is Chebyshev-Gauss-Lobatto.
cgl_axis_gap:
    Detect a cylindrical radial CGL grid's axis gap (0 = half-CGL,
    1 = rigged-CGL, None = neither).
chebyshev_interpolation_matrix:
    CGL-to-CGL interpolation via Chebyshev coefficient
    truncation/extension.
cgl_parity_interpolation_matrices:
    Spectral parity interpolation between cylindrical radial CGL
    grids (half / rigged, any combination); returns
    ``(T_even, T_odd)``.
local_interpolation_matrix:
    General interpolation via local ``fd_order+1``-point Fornberg
    stencils (bounded Lebesgue constant on any monotone grid; the
    fallback for custom / tanh / undetected grids).
build_interpolation_matrix:
    Dispatcher selecting the optimal interpolation method.

Cylindrical wall-normal interpolation on resume takes the spectral
parity path (``cgl_parity_interpolation_matrices``) for the detected
half-CGL / rigged-CGL grids (machine precision); custom / tanh /
undetected grids use the local ``fd_order`` stencil
(``local_interpolation_matrix``).  A *global* barycentric Lagrange
fit **in r** is spectrally optimal only on a true CGL grid and blows
up on the lopsided half-CGL point set (Lebesgue ``1e9``--``1e15``),
so it is not used here; the safe global formulation on these grids is
the parity-constrained fit in `$u = 2 r^2 - 1$` (where the radial CGL
points are Chebyshev-distributed), solved in the Chebyshev basis --
:func:`_spectral_even_axis_weights`, also the *interpolation*
completion's axis-node reconstruction (the *quadrature* completion
keeps the local rule: full exactness and weight positivity are
incompatible there, see :func:`_cgl_completion_matrices`).
"""

from collections.abc import Sequence

import numpy as np
from numpy import ndarray


def fornberg_weights(z: float, x: ndarray, m: int) -> ndarray:
    r"""Compute finite-difference weights via Fornberg's algorithm.

    Fornberg (1998), *SIAM Rev.* **40**, 685--691.

    Parameters
    ----------
    z:
        Evaluation point (the grid point at which the derivative is
        approximated).
    x:
        Stencil node positions, shape ``(n+1,)``.
    m:
        Maximum derivative order.

    Returns
    -------
    :
        Weight matrix of shape ``(n+1, m+1)``.  Column ``d``
        contains the weights for the ``d``-th derivative.
    """
    n = len(x) - 1
    C = np.zeros((n + 1, m + 1))
    C[0, 0] = 1.0
    c1 = 1.0
    c4 = x[0] - z
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    C[i, k] = (
                        c1 * (k * C[i - 1, k - 1] - c5 * C[i - 1, k]) / c2
                    )
                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                C[j, k] = (c4 * C[j, k] - k * C[j, k - 1]) / c3
            C[j, 0] = c4 * C[j, 0] / c3
        c1 = c2
    return C


def build_diff_matrices(
    y: ndarray,
    p: int,
) -> tuple[ndarray, ndarray]:
    r"""Build first- and second-derivative matrices on a
    non-uniform grid.

    D1 uses ``(p+1)``-point stencils, D2 uses ``(p+2)``-point
    stencils, both achieving accuracy order ``p``.  Interior rows
    use centred stencils; near-wall rows use one-sided stencils of
    the same width.

    Parameters
    ----------
    y:
        Grid-point coordinates, shape ``(Ny,)``.
    p:
        Accuracy order.  The stencil parameter ``n = p + 1``.

    Returns
    -------
    D1:
        First-derivative matrix, shape ``(Ny, Ny)``.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    """
    y = np.asarray(y)
    Ny = len(y)
    s1, s2 = p + 1, p + 2  # stencil widths
    h1, h2 = s1 // 2, s2 // 2  # half-widths for centering
    D1 = np.zeros((Ny, Ny))
    D2 = np.zeros((Ny, Ny))

    for i in range(Ny):
        # D1 stencil
        j0 = max(0, min(i - h1, Ny - s1))
        w = fornberg_weights(y[i], y[j0 : j0 + s1], 1)
        D1[i, j0 : j0 + s1] = w[:, 1]

        # D2 stencil
        j0 = max(0, min(i - h2, Ny - s2))
        w = fornberg_weights(y[i], y[j0 : j0 + s2], 2)
        D2[i, j0 : j0 + s2] = w[:, 2]

    return D1, D2


def matrix_half_bandwidth(A: ndarray, skip_rows: Sequence[int] = ()) -> int:
    r"""Measured half-bandwidth of an assembled wall-normal operator.

    Returns `$\max |i - j|$` over the nonzero entries of *A*, ignoring
    the rows in *skip_rows* -- the rows the caller replaces wholesale
    with boundary rows before the operator reaches banded storage
    (:func:`dnsjax.solvers._assemble_banded_operator`), so their own
    stencil width never has to fit.

    This is what sizes `$2p+1$` banded storage.  The default
    ``fd_order``-wide band is *exactly* right for the direct-fit
    `$D_2$` (a `$(p+2)$`-point one-sided row reaches offset `$p$` at
    row 1), but not for a composed operator: `$D_1 D_1$` reaches
    `$p + \lceil p/2 \rceil - 1$` (11 at ``fd_order = 8``).  No
    shipped configuration composes an operator any more -- the route
    that did was retired on 2026-07-26 -- so today every caller
    measures ``fd_order`` back.  Measuring rather than assuming is
    kept because the assumption is the kind that fails silently: an
    under-sized band truncates entries instead of erroring, and the
    same ``p`` is reused to band neighbouring operators
    (e.g. `$D_1 + 1/r$`).

    Parameters
    ----------
    A:
        Assembled real operator, shape ``(N, N)``.
    skip_rows:
        Row indices to ignore; negative indices count from the end.
    """
    N = A.shape[-1]
    skip = {i % N for i in skip_rows}
    half = 0
    for i in range(N):
        if i in skip:
            continue
        nz = np.nonzero(A[i])[0]
        if nz.size:
            half = max(half, abs(int(nz[0]) - i), abs(int(nz[-1]) - i))
    return half


def build_integration_weights(
    y: ndarray, p: int, left_edge: float | None = None
) -> ndarray:
    r"""Composite polynomial quadrature weights on a non-uniform
    grid.

    For each sub-interval `$[y_i, y_{i+1}]$` a local stencil of
    `$p + 1$` points (same width as the D1 stencil in
    :func:`build_diff_matrices`) is used to build a polynomial
    interpolant whose integral over that sub-interval is
    computed exactly via a Vandermonde system.  The stencil is
    normalised to `$[-1, 1]$` before forming the system for
    numerical conditioning.  Per-interval contributions are
    summed to give global weights `$w_j$` satisfying

    .. math::
        \int_{a}^{y_{N}} f(y)\,dy
        \;\approx\; \sum_j w_j\,f(y_j),

    where `$a = y_0$` by default, or *left_edge* when given.

    Composite accuracy is `$O(h^{p+1})$` for smooth
    integrands, consistent with the FD derivative order `$p$`
    from :func:`build_diff_matrices`.

    Parameters
    ----------
    y:
        Grid-point coordinates, shape ``(Ny,)``.
    p:
        Accuracy order.  Uses ``(p+1)``-point stencils.
    left_edge:
        Optional lower integration bound below the first grid
        point (``left_edge <= y[0]``).  The extra interval
        `$[\mathrm{left\_edge}, y_0]$` is covered by the
        first-stencil interpolant, keeping the composite
        order `$p+1$` for integrands that are smooth across
        the edge.  Note: the extrapolated weights oscillate
        and turn **negative** once the gap approaches the
        local spacing -- the cylindrical radial direction
        (with a gap `$[0, r_0]$` at the axis) therefore does
        **not** use this; ``build_cylindrical_grid`` instead
        integrates `$f\,r$` on the axis-*augmented* grid
        `$[0, r_0, \ldots]$` (the axis `$r=0$` is a real,
        interpolated node, so the weights stay positive).

    Returns
    -------
    :
        Weight array of shape ``(Ny,)``.
    """
    y = np.asarray(y)
    Ny = len(y)
    s = p + 1
    h = s // 2
    w = np.zeros(Ny, dtype=y.dtype)

    # (a, b, j0): integrate the interpolant through the stencil
    # y[j0 : j0 + s] over [a, b].
    intervals = [
        (y[i], y[i + 1], max(0, min(i + 1 - h, Ny - s))) for i in range(Ny - 1)
    ]
    if left_edge is not None:
        if left_edge > y[0]:
            raise ValueError(
                f"left_edge={left_edge} must not exceed y[0]={y[0]}"
            )
        intervals.append((left_edge, y[0], 0))

    for a, b, j0 in intervals:
        xs = y[j0 : j0 + s]

        mid = (xs[0] + xs[-1]) / 2
        half = (xs[-1] - xs[0]) / 2
        t = (xs - mid) / half
        a_n = (a - mid) / half
        b_n = (b - mid) / half

        V = np.vander(t, N=s, increasing=True)
        ks = np.arange(s, dtype=y.dtype)
        mu = half * (b_n ** (ks + 1) - a_n ** (ks + 1)) / (ks + 1)

        q = np.linalg.solve(V.T, mu)
        w[j0 : j0 + s] += q

    return w


def clenshaw_curtis_weights(ny: int) -> ndarray:
    r"""Clenshaw-Curtis quadrature weights for a CGL grid.

    For ``ny`` Chebyshev-Gauss-Lobatto points
    `$y_j = -\cos(j\pi / N)$`, `$j = 0, \ldots, N$` with
    `$N = \texttt{ny} - 1$`, returns the weights `$w_j$` such that
    `$\int_{-1}^{1} f(y)\,dy \approx \sum_j w_j f(y_j)$`, exact for
    polynomials of degree `$\le N$` (spectral accuracy for smooth
    integrands).  Works for both odd and even *ny*.

    Used by the full-CGL wall-normal grids (Cartesian) and, after an
    affine map `$[-1,1] \to [r_1, r_2]$` with the Jacobian folded in,
    the annular radial grid; there it is spectral for smooth
    integrands and exact for the smooth base/mean profiles (flow
    rate, bulk-velocity response).  The **cylindrical** radial grid
    (half of a CGL, with the coordinate axis) cannot use it: the
    `$r$`-Jacobian makes the even extension of `$g = f r$` kink at the
    axis, so a single Clenshaw-Curtis vector is spectral for only one
    parity of `$f$` -- that grid uses the axis-augmented
    ``build_integration_weights`` rule instead (correct for both
    parities).

    Parameters
    ----------
    ny:
        Number of CGL grid points (`$N + 1$`).

    Returns
    -------
    :
        Weight array of shape ``(ny,)``.

    References
    ----------
    Trefethen, *Spectral Methods in MATLAB* (2000), ch. 12.
    """
    N = ny - 1
    theta = np.arange(ny, dtype=np.float64) * np.pi / N

    if N % 2 == 0:  # N even (ny odd)
        M = N // 2 - 1
        w_end = 1.0 / (N * N - 1)
    else:  # N odd (ny even)
        M = (N - 1) // 2
        w_end = 1.0 / (N * N)

    if M > 0:
        k = np.arange(1, M + 1, dtype=np.float64)
        cos_terms = np.cos(2 * k[None, :] * theta[:, None])
        coeffs = 2.0 / (4 * k**2 - 1)
        v = 1.0 - np.sum(cos_terms * coeffs[None, :], axis=1)
    else:
        v = np.ones(ny, dtype=np.float64)

    if N % 2 == 0:
        v = v - np.cos(N * theta) / (N * N - 1)

    w = (2.0 / N) * v
    w[0] = w_end
    w[-1] = w_end
    return w


def _spectral_even_axis_weights(r: ndarray) -> ndarray:
    r"""Exact parity-constrained axis weights on a radial CGL grid.

    The unique weights with `$f(0) = \sum_j w_j\,f(r_j)$` exact for
    **every** even polynomial of degree `$\le 2(N_r - 1)$`: an even
    analytic field is a polynomial in `$u = 2 r^2 - 1$`, the radial
    CGL points map to a (near-)CGL point set in `$u$`, and the fit is
    solved in the Chebyshev basis, where it is well-conditioned (the
    monomial / barycentric-in-`$r$` formulations are not; see the
    module docstring).  `$r = 0$` is `$u = -1$`.

    The price of full exactness is the weight 1-norm (the noise
    amplification of the evaluation functional): exactly
    `$2 N_r - 1$` on a rigged grid -- `$u = -1$` is that set's
    deleted CGL node, the worst evaluation point -- and `$O(1)$`
    (`$\le 5.4$` up to `$N_r = 512$`) on a half grid.  At the
    machine-epsilon scale of the data this is far below the
    `$O(h^{2\,\mathrm{order}})$` truncation error of the local rule
    it replaces (in double precision the two meet only at the eps
    floor).  Exactness through all `$N_r$` points pins the weights
    uniquely, so the amplification is intrinsic, not a formulation
    artefact.  Restricted to detected CGL grids: on non-Chebyshev
    (tanh / custom) nodes a full-order fit's weights grow without
    bound, and :func:`axis_extrapolation_weights` keeps the local
    rule there.
    """
    r = np.asarray(r, dtype=np.float64)
    u = 2.0 * r * r - 1.0
    theta = np.arccos(np.clip(u, -1.0, 1.0))
    k = np.arange(len(r))
    v = np.cos(k[None, :] * theta[:, None])  # V[j, k] = T_k(u_j)
    return np.linalg.solve(v.T, (-1.0) ** k)  # T_k(-1) = (-1)^k


def axis_extrapolation_weights(
    r: ndarray, order: int, parity: str | None = "even"
) -> ndarray:
    r"""Weights to evaluate a radial field at the axis `$r = 0$`.

    Returns ``w`` of shape ``(len(r),)`` with
    `$f(0) \approx \sum_j w_j\,f(r_j)$` for a radial grid that
    excludes the axis (the cylindrical ``build_radial_cgl_grid``):

    - ``parity="even"``: the field is smooth and even in `$r$`, so
      a function of `$x = r^2$`.  On a detected radial CGL grid
      (:func:`cgl_axis_gap`) the weights are the exact spectral
      parity-constrained fit (:func:`_spectral_even_axis_weights`,
      exact for even polynomials of degree `$\le 2(N_r - 1)$`;
      *order* is ignored).  On a custom / tanh grid the innermost
      ``order + 1`` points interpolate in `$x$` (Fornberg; exact
      to degree `$\le 2\,\mathrm{order}$`) -- a global fit is
      ill-conditioned off Chebyshev-distributed nodes.
    - ``parity="odd"``: the field vanishes at the axis identically;
      returns zeros (exact).
    - ``parity=None``: one-sided ``order + 1``-point Lagrange
      extrapolation in `$r$` (no symmetry assumption -- the only
      safe choice for parity-free data, e.g. physical-space
      profiles).

    JAX-free shared leaf: used by ``interpolate_to_axis`` (runtime,
    cylindrical) and by :func:`_cgl_completion_matrices` (via the
    spectral helper) to reconstruct the rigged grid's dropped centre
    node.
    """
    if parity not in (None, "even", "odd"):
        raise ValueError(f"unknown parity {parity!r}")
    r = np.asarray(r, dtype=np.float64)
    if parity == "odd":
        return np.zeros(len(r))
    if parity == "even":
        if cgl_axis_gap(r) is not None:
            return _spectral_even_axis_weights(r)
        return _local_even_axis_weights(r, order)
    out = np.zeros(len(r))
    n = min(order + 1, len(r))
    out[:n] = fornberg_weights(0.0, r[:n], 0)[:, 0]
    return out


def _local_even_axis_weights(r: ndarray, order: int) -> ndarray:
    r"""Local even-parity axis weights (Fornberg in `$x = r^2$`).

    The innermost ``order + 1`` points interpolate in `$x$` (exact
    for even polynomials of degree `$\le 2\,\mathrm{order}$`), the
    rest of the weights are zero.  The bounded-stencil counterpart of
    :func:`_spectral_even_axis_weights`: `$O(1)$` weight 1-norm on any
    monotone grid, used where the full-order fit must not be (non-CGL
    grids) or where its sign structure is disqualifying (the
    quadrature completion; see :func:`_cgl_completion_matrices`).
    """
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros(len(r))
    n = min(order + 1, len(r))
    out[:n] = fornberg_weights(0.0, r[:n] ** 2, 0)[:, 0]
    return out


# RETIRED (superseded by the parity-free axis-augmented rule in
# ``build_cylindrical_grid``: integrate ``g = f*r`` on ``[0, *rs]``
# with the axis ``r=0`` as a free node, ``g(0)=0`` for any bounded
# ``f``).  The even-parity (``x = r^2``) rule silently assumed every
# radial integrand is even in ``r`` -- wrong for the odd mean
# ``u_theta`` -- and is removed; see ``build_cylindrical_grid``.


def local_grid_spacing(nodes: ndarray) -> ndarray:
    r"""Per-node local spacing of a 1-D non-uniform grid.

    Returns, for each node, the distance to its nearest
    neighbour:

    .. math::
        \Delta_j = \min(y_j - y_{j-1},\; y_{j+1} - y_j),

    with one-sided values at the ends
    (`$\Delta_0 = y_1 - y_0$`,
    `$\Delta_{N-1} = y_{N-1} - y_{N-2}$`).  Used as the local
    advection length scale of the CFL diagnostic
    (:mod:`dnsjax.measurements`).  Note the one-sided end
    convention also applies to the innermost node of the radial
    CGL grid, whose distance to the (excluded) axis
    `$r = 0$` is not considered.

    Parameters
    ----------
    nodes:
        Strictly increasing grid coordinates, shape ``(N,)``
        with ``N >= 2``.

    Returns
    -------
    :
        Local spacings, shape ``(N,)``.
    """
    nodes = np.asarray(nodes)
    gaps = np.diff(nodes)
    spacing = np.empty_like(nodes)
    spacing[0] = gaps[0]
    spacing[-1] = gaps[-1]
    spacing[1:-1] = np.minimum(gaps[:-1], gaps[1:])
    return spacing


# ── Stretched grid generation ────────────────────────────────


def tanh_two_sided_grid(ny: int, s: float) -> ndarray:
    r"""Symmetric tanh-stretched grid on `$[-1, 1]$`.

    Wall-normal grids for finite-difference methods do not
    benefit from the CGL `$O(1/N^2)$` wall clustering
    (designed for spectral accuracy), yet inherit the
    `$O(N^4)$` second-derivative conditioning and an
    `$O(1/N^2)$` convective CFL limit (Trefethen 2000;
    Weideman & Reddy 2000).  A tanh-stretched grid with
    controlled wall spacing `$\sim O(1/N)$` achieves the
    same order-`$p$` accuracy with `$O(N^2)$` conditioning
    and an `$O(1/N)$` CFL limit.

    .. math::
        y_j = \frac{\tanh\!\bigl(s\,(2j/(N{-}1) - 1)\bigr)}
                   {\tanh(s)},
        \quad j = 0, \ldots, N{-}1.

    Endpoints are exactly `$y = \pm 1$`.  As `$s \to 0$`
    the grid approaches uniform spacing; larger `$s$`
    increases wall clustering.

    Parameters
    ----------
    ny:
        Number of grid points.
    s:
        Stretching parameter (`$s > 0$`).

    Returns
    -------
    :
        Grid array, shape ``(ny,)``, ascending from
        `$-1$` to `$1$`.
    """
    xi = np.linspace(-1.0, 1.0, ny)
    return np.tanh(s * xi) / np.tanh(s)


def tanh_one_sided_grid(nr: int, s: float) -> ndarray:
    r"""One-sided tanh-stretched grid on `$(0, 1]$`.

    Clusters points toward the wall at `$r = 1$` while
    keeping all points strictly positive (no point at
    `$r = 0$`, consistent with the parity-reduced
    cylindrical formulation).

    .. math::
        \xi_j = (j + 1) / N_r, \qquad
        r_j = 1 - \frac{\tanh\!\bigl(s\,(1 - \xi_j)\bigr)}
                       {\tanh(s)},
        \quad j = 0, \ldots, N_r{-}1.

    `$\xi$` spans `$(0, 1]$` (excluding 0), so `$r_0 > 0$`
    and `$r_{N_r - 1} = 1$` exactly.

    Parameters
    ----------
    nr:
        Number of radial grid points.
    s:
        Stretching parameter (`$s > 0$`).

    Returns
    -------
    :
        Grid array, shape ``(nr,)``, ascending from near
        `$0$` to `$1$`.
    """
    xi = np.arange(1, nr + 1, dtype=np.float64) / nr
    return 1.0 - np.tanh(s * (1.0 - xi)) / np.tanh(s)


# ── Grid detection ───────────────────────────────────────────


def is_cgl_grid(y: ndarray) -> bool:
    r"""Detect whether ``y`` is a CGL grid on `$[-1, 1]$`.

    Compares against `$y_j = -\cos(j\pi/(N{-}1))$` for
    `$j = 0, \ldots, N{-}1$` where `$N$` = ``len(y)``.
    """
    y = np.asarray(y)
    N = len(y)
    expected = -np.cos(np.arange(N) * np.pi / (N - 1))
    return bool(np.allclose(y, expected, atol=1e-12))


def cgl_axis_gap(r: ndarray) -> int | None:
    r"""Detect a cylindrical radial CGL grid's axis gap.

    Returns the auxiliary axis gap `$g$` such that ``r`` is the
    positive half of a `$(2 N_r + g)$`-point CGL grid on
    `$[-1, 1]$` (`$N_r$` = ``len(r)``):

    - ``0`` -- half-CGL (even total, staggered `$r_0 = \Delta r/2$`),
    - ``1`` -- rigged-CGL (odd total, axis centre node dropped,
      `$r_0 = \Delta r$`),
    - ``None`` -- neither (a tanh or custom grid).

    Used to dispatch the spectral parity interpolation
    (:func:`cgl_parity_interpolation_matrices`); a ``None`` grid
    falls back to :func:`local_interpolation_matrix`.
    """
    r = np.asarray(r)
    Nr = len(r)
    if Nr < 2:
        return None
    for g in (0, 1):
        N_full = 2 * Nr + g
        s = -np.cos(np.arange(N_full) * np.pi / (N_full - 1))
        if np.allclose(r, s[Nr + g :], atol=1e-12):
            return g
    return None


# ── Interpolation matrices ───────────────────────────────────


def chebyshev_interpolation_matrix(ny_old: int, ny_new: int) -> ndarray:
    r"""CGL-to-CGL interpolation via Chebyshev coefficients.

    Given values `$f(y_j)$` at `$N_{\mathrm{old}}$` CGL points
    `$y_j = -\cos(j\pi/N)$`, `$N = N_{\mathrm{old}} - 1$`:

    1.  **Analysis** -- Chebyshev coefficients via DCT-I:

        .. math::
            \hat{f}_k = \frac{2}{N\,\bar{c}_k}
            \sum_{j=0}^{N} \frac{f_j}{\bar{c}_j}
            \cos\!\Bigl(\frac{k\,j\,\pi}{N}\Bigr),

        where `$\bar{c}_0 = \bar{c}_N = 2$`,
        `$\bar{c}_k = 1$` otherwise.

    2.  **Truncate** to `$K + 1$` coefficients,
        `$K = \min(N_{\mathrm{old}}, N_{\mathrm{new}}) - 1$`.

    3.  **Synthesis** -- evaluate at `$N_{\mathrm{new}}$` CGL
        points, `$M = N_{\mathrm{new}} - 1$`:

        .. math::
            f(y_i^{\,\mathrm{new}}) = \sum_{k=0}^{K}
            \hat{f}_k \cos\!\Bigl(\frac{k\,i\,\pi}{M}\Bigr).

    The composite matrix `$T = S\,A$` of shape
    `$(N_{\mathrm{new}}, N_{\mathrm{old}})$` is the unique
    best polynomial approximation in the Chebyshev sense,
    exact for polynomials of degree `$\le K$`.

    Parameters
    ----------
    ny_old:
        Number of source CGL points.
    ny_new:
        Number of target CGL points.

    Returns
    -------
    :
        Interpolation matrix, shape ``(ny_new, ny_old)``.
    """
    N = ny_old - 1
    M = ny_new - 1
    K = min(ny_old, ny_new) - 1

    k = np.arange(K + 1, dtype=np.float64)
    j = np.arange(ny_old, dtype=np.float64)
    i = np.arange(ny_new, dtype=np.float64)

    # Analysis: A[k, j] = (2/N) * cos(k*j*pi/N) / (cbar_k * cbar_j)
    cbar_j = np.ones(ny_old)
    cbar_j[0] = 2.0
    cbar_j[-1] = 2.0
    cbar_k = np.ones(K + 1)
    cbar_k[0] = 2.0
    if K == N:
        cbar_k[-1] = 2.0
    A = (
        (2.0 / N)
        * np.cos(k[:, None] * j[None, :] * np.pi / N)
        / (cbar_k[:, None] * cbar_j[None, :])
    )

    # Synthesis: S[i, k] = cos(k*i*pi/M)
    S = np.cos(k[None, :] * i[:, None] * np.pi / M)

    return S @ A


def _cgl_completion_matrices(
    nr: int, gap: int, order: int | None = None
) -> tuple[ndarray, ndarray]:
    r"""Even/odd completion matrices for a radial CGL grid.

    ``E_even``, ``E_odd`` of shape ``(2*nr + gap, nr)`` map the
    positive-half values of a half-CGL (``gap=0``) or rigged-CGL
    (``gap=1``) grid to the full CGL grid: the negative half by parity
    `$f(-r) = \sigma f(r)$`, and (rigged only) the dropped axis centre
    node for `$\sigma = +1$` (an even field, smooth in `$r^2$`) or
    `$0$` for `$\sigma = -1$` (an odd field vanishes at the axis).

    The `$\sigma = +1$` axis node has two rules, chosen by *order*:

    - ``order=None`` -- the exact spectral parity-constrained fit
      (:func:`_spectral_even_axis_weights`), making the completion
      exact for every even polynomial the full grid can represent.
      Used by the **interpolation** (machine-precision resume
      regrids; weight signs are immaterial there).
    - an integer -- the local ``order + 1``-point rule
      (:func:`_local_even_axis_weights`).  Used by the
      **quadrature**: an even rule exact to the full degree
      `$2(N_r - 1)$` is *uniquely* pinned by its exactness
      conditions, and on a rigged grid that unique rule has
      alternating near-axis weights (strictly negative entries from
      ``nr = 48`` up) -- so full exactness and the strict positivity
      the energy norm requires are mathematically incompatible
      there.  The local rule keeps `$w_\sigma > 0$`, at an
      `$O(h^{2\,\mathrm{order}})$` axis-node error further damped by
      the axis node's tiny `$\int r\,dr$` moment.
    """
    n_full = 2 * nr + gap
    s = -np.cos(np.arange(n_full) * np.pi / (n_full - 1))
    rs = s[nr + gap :]
    E_even = np.zeros((n_full, nr))
    E_odd = np.zeros((n_full, nr))
    for k in range(nr):
        pos = nr + gap + k  # positive-half index
        neg = n_full - 1 - pos  # its r -> -r mirror
        E_even[pos, k] = 1.0
        E_odd[pos, k] = 1.0
        E_even[neg, k] = 1.0  # sigma = +1
        E_odd[neg, k] = -1.0  # sigma = -1
    if gap == 1:
        E_even[nr, :] = (
            _spectral_even_axis_weights(rs)
            if order is None
            else _local_even_axis_weights(rs, order)
        )
    return E_even, E_odd


def _halfrange_r_moments(s: ndarray) -> ndarray:
    r"""Full-CGL Lagrange moments `$W_i = \int_0^1 L_i(r)\,r\,dr$`.

    A Clenshaw-Curtis rule with weight `$r$` on the half-interval
    `$[0, 1]$`: `$\sum_i W_i P(s_i) = \int_0^1 P(r)\,r\,dr$` exactly
    for `$P$` of degree `$\le \mathrm{len}(s) - 1$`.  Computed stably
    in the Chebyshev basis (``W = A^T m``, `$A$` the CGL analysis
    matrix, `$m_k = \int_0^1 T_k(r)\,r\,dr$`) -- **not** by a
    Vandermonde solve, which is catastrophically ill-conditioned at
    the degrees involved.
    """
    import numpy.polynomial.chebyshev as npc

    n = len(s)
    n_cheb = n - 1
    kk = np.arange(n)
    cbar = np.ones(n)
    cbar[0] = 2.0
    cbar[-1] = 2.0
    # Analysis matrix A: grid values -> Chebyshev coefficients.
    a_mat = (
        (2.0 / n_cheb)
        * np.cos(np.outer(kk, kk) * np.pi / n_cheb)
        / np.outer(cbar, cbar)
    )
    # Moments m_k = int_0^1 T_k(r) r dr (r = T_1; chebmul/chebint),
    # with the (-1)^k from the CGL grid convention s_j = -cos(...):
    # T_k(s_j) = (-1)^k cos(k j pi / N), so ``a_mat`` returns the
    # coefficients up to (-1)^k (this cancels in interpolation but not
    # against the true moments here).
    m = np.empty(n)
    for k in range(n):
        t_k = np.zeros(k + 1)
        t_k[k] = 1.0
        anti = npc.chebint(npc.chebmul([0.0, 1.0], t_k))
        m[k] = ((-1) ** k) * (npc.chebval(1.0, anti) - npc.chebval(0.0, anti))
    return a_mat.T @ m


def cgl_radial_quadrature_weights(
    rs: ndarray, order: int
) -> tuple[ndarray, ndarray] | None:
    r"""Parity-specific spectral radial quadrature weights.

    For a cylindrical radial CGL grid (half-CGL or rigged-CGL,
    detected by :func:`cgl_axis_gap`) returns ``(w_even, w_odd)`` with
    `$\sum_j w_{\sigma,j} f_j \approx \int_0^1 f(r)\,r\,dr$`
    **spectrally** exact for a smooth field of parity `$\sigma$` in
    `$r$`.  It completes the positive-half values to the full CGL grid
    by parity -- with the *local* ``order`` rule for the rigged axis
    node: the fully exact rule is uniquely pinned and provably
    non-positive at large `$N_r$`, see
    :func:`_cgl_completion_matrices` -- and applies the full-grid
    Clenshaw-Curtis rule with weight `$r$` on `$[0, 1]$`
    (:func:`_halfrange_r_moments`):
    `$w_\sigma = E_\sigma^{\mathsf T} W$`.  Both vectors are strictly
    positive (a definite energy norm).

    A *single* vector cannot be spectral for both parities (an even
    and an odd integrand need different completions, and a
    parity-agnostic radial rule needs the unavailable `$f(-r)$` -- an
    ill-conditioned mirror extrapolation).  So the caller integrates
    each diagnostic with the vector matching its **known** parity
    (energy `$|u|^2$`, mean `$u_z$`, dissipation: even; mean
    `$u_\theta$`: odd).  Returns ``None`` for a non-CGL grid (custom /
    tanh), where the caller uses the parity-agnostic axis-augmented
    composite rule instead.
    """
    g = cgl_axis_gap(rs)
    if g is None:
        return None
    nr = len(rs)
    n_full = 2 * nr + g
    s = -np.cos(np.arange(n_full) * np.pi / (n_full - 1))
    w_moments = _halfrange_r_moments(s)
    e_even, e_odd = _cgl_completion_matrices(nr, g, order)
    return e_even.T @ w_moments, e_odd.T @ w_moments


def cgl_parity_interpolation_matrices(
    nr_old: int,
    nr_new: int,
    gap_old: int,
    gap_new: int,
) -> tuple[ndarray, ndarray]:
    r"""Spectral parity interpolation between cylindrical CGL grids.

    Interpolates between two radial CGL grids -- half-CGL
    (``gap = 0``) or rigged-CGL (``gap = 1``), in any source/target
    combination -- at **spectral** accuracy, exploiting that each
    velocity component at azimuthal mode `$m$` has a definite parity
    `$\sigma = \pm 1$` under `$r \to -r$`:

    1.  **Complete** the source positive half to its full CGL grid
        (`$2 N_r^{\mathrm{old}} + g_{\mathrm{old}}$` points): the
        negative half by parity `$f(-r) = \sigma f(r)$`, and (only
        when `$g_{\mathrm{old}} = 1$`) the dropped axis centre node
        by the exact spectral parity-constrained fit
        (:func:`_spectral_even_axis_weights`) for `$\sigma = +1$`
        (an even field is smooth in `$r^2$`) or `$0$` for
        `$\sigma = -1$` (an odd field vanishes at the axis).
    2.  **Interpolate** full CGL `$\to$` full CGL by Chebyshev
        coefficients (:func:`chebyshev_interpolation_matrix`).
    3.  **Restrict** to the target's positive half (drop its centre
        node, if any, and the negative half).

    Each step is linear, so the result is the pair of matrices
    `$T_\sigma = R\,T_{\mathrm{full}}\,E_\sigma$`, shape
    ``(nr_new, nr_old)``.  Parity assignment per component (the
    caller applies `$T_{\mathrm{even}}$`/`$T_{\mathrm{odd}}$`
    per mode): `$u_z$` `$\sigma = (-1)^m$`; `$u_\pm$`
    `$\sigma = (-1)^{m+1}$`.

    Every ingredient is now spectral (the axis reconstruction
    included), so this is machine-precision for a resolved field --
    vastly better than the ``fd_order``
    :func:`local_interpolation_matrix` on these grids.  Used for
    every half/rigged cylindrical resume; ``local`` handles only
    custom / tanh / undetected grids.

    Parameters
    ----------
    nr_old, nr_new:
        Source / target radial point counts.
    gap_old, gap_new:
        Source / target axis gap (0 = half-CGL, 1 = rigged-CGL).

    Returns
    -------
    T_even, T_odd:
        Interpolation matrices for `$\sigma = +1$` / `$\sigma = -1$`
        fields, each shape ``(nr_new, nr_old)``.
    """
    n_new_full = 2 * nr_new + gap_new
    T_full = chebyshev_interpolation_matrix(2 * nr_old + gap_old, n_new_full)

    # Extension E_sigma: full source-CGL values from the positive half
    # (shared with the quadrature).
    E_even, E_odd = _cgl_completion_matrices(nr_old, gap_old)

    # Restriction R: target positive half (drops centre + negative).
    R = np.zeros((nr_new, n_new_full))
    for k in range(nr_new):
        R[k, nr_new + gap_new + k] = 1.0

    return R @ T_full @ E_even, R @ T_full @ E_odd


def local_interpolation_matrix(
    y_old: ndarray, y_new: ndarray, order: int
) -> ndarray:
    r"""Interpolation via local ``order+1``-point Fornberg stencils.

    For each target point a contiguous window of ``order+1`` source
    nodes is chosen -- centred on the target where possible, clamped
    to the grid ends near the boundaries -- and the 0-th Fornberg
    weights (:func:`fornberg_weights`) evaluate the local Lagrange
    interpolant.  Unlike a single global degree-``N-1`` polynomial
    fit, the local stencil has a bounded Lebesgue constant on *any*
    monotone grid (``O(10)`` vs the global fit's ``1e9``--``1e15`` on
    the lopsided radial CGL point set, which blows the field up on
    resume).  Accuracy is ``order``-th order.  It is the fallback for
    custom / tanh / undetected grids; the spectral
    :func:`cgl_parity_interpolation_matrices` handles the detected
    half-CGL / rigged-CGL cylindrical grids.

    Near the axis a target below the innermost source node (e.g. a
    rigged-to-half radial-grid change on resume of a custom grid) is
    extrapolated by the same one-sided window; the short (sub-spacing)
    reach keeps the Lebesgue constant ``O(10)``.  (Detected radial CGL
    grids take the spectral :func:`cgl_parity_interpolation_matrices`
    path, not this fallback.)

    Parameters
    ----------
    y_old:
        Source grid, shape ``(N_old,)``, strictly monotone.
    y_new:
        Target grid, shape ``(N_new,)``.
    order:
        Interpolation order; stencil width is ``order + 1`` (capped
        at ``N_old``).

    Returns
    -------
    :
        Interpolation matrix, shape ``(N_new, N_old)``.
    """
    y_old = np.asarray(y_old, dtype=np.float64)
    y_new = np.asarray(y_new, dtype=np.float64)
    N_old = len(y_old)
    N_new = len(y_new)
    n = min(order + 1, N_old)

    T = np.zeros((N_new, N_old))
    for i in range(N_new):
        x = y_new[i]
        # Contiguous window of n nodes, centred on x, clamped to ends.
        j = int(np.searchsorted(y_old, x))
        lo = min(max(j - n // 2, 0), N_old - n)
        idx = np.arange(lo, lo + n)
        T[i, idx] = fornberg_weights(x, y_old[idx], 0)[:, 0]

    return T


def build_interpolation_matrix(
    y_old: ndarray,
    y_new: ndarray,
    geometry: str,
    order: int = 4,
) -> ndarray | tuple[ndarray, ndarray]:
    r"""Select the optimal interpolation method for the grids.

    - Cartesian with both grids CGL: Chebyshev coefficient
      truncation/extension (spectrally optimal).
    - Cylindrical with both grids detected radial CGL grids
      (half-CGL or rigged-CGL, :func:`cgl_axis_gap`): the spectral
      parity interpolation :func:`cgl_parity_interpolation_matrices`,
      returning an ``(T_even, T_odd)`` **tuple** applied per
      azimuthal mode by the caller (near machine precision, vs
      ``order`` for the local stencil).
    - Annular with both grids CGL on `$[r_1, r_2]$`: Chebyshev
      coefficient truncation/extension after affine mapping to
      `$[-1, 1]$` (spectrally optimal; the Chebyshev matrix is
      domain-independent, so the same path applies under the affine
      map).
    - Otherwise (custom / tanh / undetected grid): local
      ``order``-stencil Fornberg interpolation
      (:func:`local_interpolation_matrix`), whose Lebesgue constant
      stays bounded on any monotone grid.

    Parameters
    ----------
    y_old:
        Source grid, shape ``(N_old,)``.
    y_new:
        Target grid, shape ``(N_new,)``.
    geometry:
        ``"cartesian"``, ``"cylindrical"``, or ``"annular"``.
    order:
        FD order for the local-stencil fallback (stencil width
        ``order + 1``); ignored on the CGL Chebyshev and parity
        paths (fully spectral, including the rigged-grid axis
        reconstruction).

    Returns
    -------
    :
        Interpolation matrix ``(N_new, N_old)``, or -- for a detected
        cylindrical CGL pair -- an ``(T_even, T_odd)`` parity tuple.
    """
    y_old = np.asarray(y_old, dtype=np.float64)
    y_new = np.asarray(y_new, dtype=np.float64)

    if geometry == "cartesian" and is_cgl_grid(y_old) and is_cgl_grid(y_new):
        return chebyshev_interpolation_matrix(len(y_old), len(y_new))

    if geometry == "cylindrical":
        # Spectral parity interpolation when both grids are detected
        # radial CGL grids (half-CGL or rigged-CGL, any combination);
        # returns an (T_even, T_odd) parity pair applied per azimuthal
        # mode by the caller.  A custom / tanh grid (gap None) falls
        # through to the local stencil.
        gap_old = cgl_axis_gap(y_old)
        gap_new = cgl_axis_gap(y_new)
        if gap_old is not None and gap_new is not None:
            return cgl_parity_interpolation_matrices(
                len(y_old), len(y_new), gap_old, gap_new
            )

    if geometry == "annular":
        # The annular grid is a CGL grid affinely mapped to [r1, r2];
        # normalise each grid to [-1, 1] to detect it.  The Chebyshev
        # interpolation matrix depends only on the point counts (it
        # works in coefficient space), so it is invariant under the
        # affine domain map.
        def _to_unit(g: ndarray) -> ndarray:
            return (2.0 * g - g[0] - g[-1]) / (g[-1] - g[0])

        if is_cgl_grid(_to_unit(y_old)) and is_cgl_grid(_to_unit(y_new)):
            return chebyshev_interpolation_matrix(len(y_old), len(y_new))

    return local_interpolation_matrix(y_old, y_new, order)
