"""Finite-difference infrastructure for wall-bounded flows.

Offline precomputation (pure NumPy) for the influence-matrix method
(IMM).  All quantities produced here are purely real, following the
proof in the IMM reference document (Section 3.7).

**Functional Purity Exception:** This module runs completely offline on
the CPU using pure NumPy (rather than JAX) to construct matrices once.
This is a documented exception to the pure-JAX paradigm of the codebase.

Functions
---------
fornberg_weights:
    Fornberg's (1998) algorithm for FD weights on non-uniform grids.
build_diff_matrices:
    Assemble first- and second-derivative matrices D1, D2.

"""

import numpy as np


def fornberg_weights(z: float, x: np.ndarray, m: int) -> np.ndarray:
    """Compute finite-difference weights via Fornberg's algorithm.

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
        Weight matrix of shape ``(n+1, m+1)``.  Column ``d`` contains the
        weights for the ``d``-th derivative.
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
    y: np.ndarray, p: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build first- and second-derivative matrices on a non-uniform grid.

    D1 uses ``(p+1)``-point stencils, D2 uses ``(p+2)``-point stencils,
    both achieving accuracy order ``p``.  Interior rows use centred
    stencils; near-wall rows use one-sided stencils of the same width.

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



