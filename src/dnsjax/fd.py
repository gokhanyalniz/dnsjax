"""Finite-difference infrastructure for wall-bounded flows.

Offline precomputation for the influence-matrix method (IMM).  All
functions run at initialisation time outside ``@jit``, so Python
loops and concrete-value branching are used directly.

Functions
---------
fornberg_weights:
    Fornberg's (1998) algorithm for FD weights on non-uniform grids.
build_diff_matrices:
    Assemble first- and second-derivative matrices D1, D2.
build_integration_weights:
    Composite polynomial quadrature weights on a non-uniform grid.

"""

from jax import Array
from jax import numpy as jnp


def fornberg_weights(z: Array, x: Array, m: int) -> Array:
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
    C = jnp.zeros((n + 1, m + 1))
    C = C.at[0, 0].set(1.0)
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
                    C = C.at[i, k].set(
                        c1 * (k * C[i - 1, k - 1] - c5 * C[i - 1, k]) / c2
                    )
                C = C.at[i, 0].set(-c1 * c5 * C[i - 1, 0] / c2)
            for k in range(mn, 0, -1):
                C = C.at[j, k].set((c4 * C[j, k] - k * C[j, k - 1]) / c3)
            C = C.at[j, 0].set(c4 * C[j, 0] / c3)
        c1 = c2
    return C


def build_diff_matrices(y: Array, p: int) -> tuple[Array, Array]:
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
    Ny = len(y)
    s1, s2 = p + 1, p + 2  # stencil widths
    h1, h2 = s1 // 2, s2 // 2  # half-widths for centering
    D1 = jnp.zeros((Ny, Ny))
    D2 = jnp.zeros((Ny, Ny))

    for i in range(Ny):
        # D1 stencil
        j0 = max(0, min(i - h1, Ny - s1))
        w = fornberg_weights(y[i], y[j0 : j0 + s1], 1)
        D1 = D1.at[i, j0 : j0 + s1].set(w[:, 1])

        # D2 stencil
        j0 = max(0, min(i - h2, Ny - s2))
        w = fornberg_weights(y[i], y[j0 : j0 + s2], 2)
        D2 = D2.at[i, j0 : j0 + s2].set(w[:, 2])

    return D1, D2


def build_integration_weights(y: Array, p: int) -> Array:
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
        \int_{y_0}^{y_{N}} f(y)\,dy
        \;\approx\; \sum_j w_j\,f(y_j).

    Composite accuracy is `$O(h^{p+1})$` for smooth
    integrands, consistent with the FD derivative order `$p$`
    from :func:`build_diff_matrices`.

    Parameters
    ----------
    y:
        Grid-point coordinates, shape ``(Ny,)``.
    p:
        Accuracy order.  Uses ``(p+1)``-point stencils.

    Returns
    -------
    :
        Weight array of shape ``(Ny,)``.
    """
    Ny = len(y)
    s = p + 1
    h = s // 2
    w = jnp.zeros(Ny, dtype=y.dtype)

    for i in range(Ny - 1):
        j0 = max(0, min(i + 1 - h, Ny - s))
        xs = y[j0 : j0 + s]

        mid = (xs[0] + xs[-1]) / 2
        half = (xs[-1] - xs[0]) / 2
        t = (xs - mid) / half
        a_n = (y[i] - mid) / half
        b_n = (y[i + 1] - mid) / half

        V = jnp.vander(t, N=s, increasing=True)
        ks = jnp.arange(s, dtype=y.dtype)
        mu = half * (b_n ** (ks + 1) - a_n ** (ks + 1)) / (ks + 1)

        q = jnp.linalg.solve(V.T, mu)
        w = w.at[j0 : j0 + s].add(q)

    return w
