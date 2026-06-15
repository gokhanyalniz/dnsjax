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
build_integration_weights:
    Composite polynomial quadrature weights on a non-uniform grid.
is_cgl_grid:
    Detect whether a grid is Chebyshev-Gauss-Lobatto.
is_half_cgl_grid:
    Detect whether a grid is a half-CGL radial grid.
chebyshev_interpolation_matrix:
    CGL-to-CGL interpolation via Chebyshev coefficient
    truncation/extension.
half_cgl_interpolation_matrices:
    Parity-aware half-CGL-to-half-CGL interpolation.
barycentric_interpolation_matrix:
    General interpolation via barycentric Lagrange formula.
build_interpolation_matrix:
    Dispatcher selecting the optimal interpolation method.

"""

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
        the edge.  Used for the cylindrical radial direction,
        where the axis `$r = 0$` is not a grid point yet the
        integral runs over the full disc: without it, every
        radial integral would drop the `$[0, r_0]$` mass and
        carry an `$O(r_0^2) = O(N_r^{-2})$` bias regardless
        of *p*.  (The extrapolation distance `$r_0$` is small
        compared to the stencil width, so conditioning is not
        an issue.)

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
    convention also applies to the first node of the half-CGL
    radial grid, whose distance to the (excluded) axis
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


def is_half_cgl_grid(r: ndarray) -> bool:
    r"""Detect whether ``r`` is a half-CGL grid on `$(0, 1]$`.

    Compares against the positive half of a `$2 N_r$`-point
    CGL grid.
    """
    r = np.asarray(r)
    Nr = len(r)
    N_full = 2 * Nr
    s = -np.cos(np.arange(N_full) * np.pi / (N_full - 1))
    expected = s[Nr:]
    return bool(np.allclose(r, expected, atol=1e-12))


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


def half_cgl_interpolation_matrices(
    nr_old: int, nr_new: int
) -> tuple[ndarray, ndarray]:
    r"""Parity-aware half-CGL-to-half-CGL interpolation.

    Each velocity component at azimuthal mode `$m$` has
    definite parity `$\sigma = \pm 1$` under `$r \to -r$` on
    the auxiliary grid:

    1.  **Extend** `$N_r^{\mathrm{old}}$` half-grid values to
        `$2 N_r^{\mathrm{old}}$` full CGL values via
        `$f(-r_j) = \sigma\,f(r_j)$`.

    2.  **Interpolate** via full CGL Chebyshev:
        `$2 N_r^{\mathrm{old}} \to 2 N_r^{\mathrm{new}}$`.

    3.  **Restrict** to the positive half
        (`$N_r^{\mathrm{new}}$` points).

    The combined matrix
    `$T_\sigma = R\,T_{\mathrm{full}}\,E_\sigma$` has shape
    `$(N_r^{\mathrm{new}}, N_r^{\mathrm{old}})$`.

    Parity assignment per velocity component:

    ========  ======================  =============================
    field     `$m_{\mathrm{eff}}$`    parity `$\sigma$`
    ========  ======================  =============================
    `$u_z$`   `$m$`                   `$(-1)^m$`
    `$u_+$`   `$m + 1$`              `$(-1)^{m+1}$`
    `$u_-$`   `$m - 1$`              `$(-1)^{m+1}$`
    ========  ======================  =============================

    Parameters
    ----------
    nr_old:
        Number of source half-CGL radial points.
    nr_new:
        Number of target half-CGL radial points.

    Returns
    -------
    T_even:
        Interpolation matrix for even-parity fields
        (`$\sigma = +1$`), shape ``(nr_new, nr_old)``.
    T_odd:
        Interpolation matrix for odd-parity fields
        (`$\sigma = -1$`), shape ``(nr_new, nr_old)``.
    """
    # Full CGL interpolation: 2*nr_old -> 2*nr_new
    T_full = chebyshev_interpolation_matrix(2 * nr_old, 2 * nr_new)

    # Extension matrices E_sigma: (2*nr_old, nr_old)
    # Ghost half = sigma * physical half reversed.
    E_even = np.zeros((2 * nr_old, nr_old))
    E_odd = np.zeros((2 * nr_old, nr_old))
    for k in range(nr_old):
        # Physical half (indices nr_old .. 2*nr_old-1)
        E_even[nr_old + k, k] = 1.0
        E_odd[nr_old + k, k] = 1.0
        # Ghost half (indices 0 .. nr_old-1): mirror
        ghost_idx = nr_old - 1 - k
        E_even[ghost_idx, k] = 1.0  # sigma = +1
        E_odd[ghost_idx, k] = -1.0  # sigma = -1

    # Restriction: take positive half (rows nr_new .. 2*nr_new-1)
    R = np.zeros((nr_new, 2 * nr_new))
    for k in range(nr_new):
        R[k, nr_new + k] = 1.0

    T_even = R @ T_full @ E_even
    T_odd = R @ T_full @ E_odd
    return T_even, T_odd


def barycentric_interpolation_matrix(
    y_old: ndarray, y_new: ndarray
) -> ndarray:
    r"""General interpolation via barycentric Lagrange formula.

    Barycentric weights (Berrut & Trefethen, *SIAM Rev.* 2004):

    .. math::
        w_j = \frac{1}{\prod_{k \neq j}(x_j - x_k)}.

    Computed in log-space for numerical stability at large
    `$N$`.  The interpolation matrix entry is

    .. math::
        T_{ij} = L_j(x_i^{\,\mathrm{new}})
        = \frac{w_j / (x_i - x_j)}
               {\sum_k w_k / (x_i - x_k)},

    with the convention `$L_j = \delta_{jk}$` when `$x_i$`
    coincides with `$x_k$` (within tolerance ``1e-14``).

    Parameters
    ----------
    y_old:
        Source grid points, shape ``(N_old,)``.
    y_new:
        Target grid points, shape ``(N_new,)``.

    Returns
    -------
    :
        Interpolation matrix, shape ``(N_new, N_old)``.
    """
    y_old = np.asarray(y_old, dtype=np.float64)
    y_new = np.asarray(y_new, dtype=np.float64)
    N_old = len(y_old)
    N_new = len(y_new)

    # Barycentric weights in log-space
    log_abs = np.zeros(N_old)
    signs = np.ones(N_old)
    for j in range(N_old):
        for k in range(N_old):
            if k == j:
                continue
            d = y_old[j] - y_old[k]
            log_abs[j] += np.log(abs(d))
            if d < 0:
                signs[j] *= -1
    # w_j = signs[j] * exp(-log_abs[j])
    log_abs -= np.mean(log_abs)  # centre for overflow safety
    w = signs * np.exp(-log_abs)

    T = np.zeros((N_new, N_old))
    tol = 1e-14
    for i in range(N_new):
        diffs = y_new[i] - y_old
        exact = np.where(np.abs(diffs) < tol)[0]
        if len(exact) > 0:
            T[i, exact[0]] = 1.0
        else:
            terms = w / diffs
            T[i, :] = terms / terms.sum()

    return T


def build_interpolation_matrix(
    y_old: ndarray,
    y_new: ndarray,
    geometry: str,
) -> ndarray | tuple[ndarray, ndarray]:
    r"""Select the optimal interpolation method for the grids.

    - Cartesian with both grids CGL: Chebyshev coefficient
      truncation/extension (spectrally optimal).
    - Cylindrical with both grids half-CGL: parity-aware
      Chebyshev interpolation (spectrally optimal).
    - Annular with both grids CGL on `$[r_1, r_2]$`: Chebyshev
      coefficient truncation/extension after affine mapping to
      `$[-1, 1]$` (spectrally optimal; the Chebyshev matrix is
      domain-independent, so the same path applies under the affine
      map).
    - Otherwise: barycentric Lagrange interpolation.

    Parameters
    ----------
    y_old:
        Source grid, shape ``(N_old,)``.
    y_new:
        Target grid, shape ``(N_new,)``.
    geometry:
        ``"cartesian"``, ``"cylindrical"``, or ``"annular"``.

    Returns
    -------
    :
        Single ``(N_new, N_old)`` matrix, or a tuple
        ``(T_even, T_odd)`` for parity-aware cylindrical
        interpolation.
    """
    y_old = np.asarray(y_old, dtype=np.float64)
    y_new = np.asarray(y_new, dtype=np.float64)

    if geometry == "cartesian" and is_cgl_grid(y_old) and is_cgl_grid(y_new):
        return chebyshev_interpolation_matrix(len(y_old), len(y_new))

    if (
        geometry == "cylindrical"
        and is_half_cgl_grid(y_old)
        and is_half_cgl_grid(y_new)
    ):
        return half_cgl_interpolation_matrices(len(y_old), len(y_new))

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

    return barycentric_interpolation_matrix(y_old, y_new)
