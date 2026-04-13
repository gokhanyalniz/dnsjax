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
build_Lk_neumann:
    Laplacian operator `$D_2 - k^2 I$` with Neumann BCs (for pressure).
build_Hk_dirichlet:
    Helmholtz operators `$H_k$`, `$H_k^-$` with Dirichlet BCs
    (for velocity).
precompute_imm:
    Full offline loop: D1, D2, Lk, Hk, Hk_minus, p1, p2, M_inv for
    every Fourier mode.  Returns a dict of JAX arrays ready for
    vmapped online solves.
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


def build_Lk_neumann(k2: float, D1: np.ndarray, D2: np.ndarray) -> np.ndarray:
    """Build the Laplacian operator `$D_2 - k^2 I$` with Neumann BCs.

    Boundary rows are replaced by the corresponding rows of D1 to
    encode Neumann (`$\partial p/\partial y = \dots$`) conditions
    for the pressure Poisson equation.

    For the mean mode `$k^2 = 0$`, the first row is instead replaced
    by `$(1, 0, \dots, 0)$` to pin `$p_0 = 0$`, since the pure-Neumann
    problem is singular (pressure determined up to a constant).

    Parameters
    ----------
    k2:
        Squared horizontal wavenumber `$k_x^2 + k_z^2$`.
    D1, D2:
        Derivative matrices, shape ``(Ny, Ny)``.

    Returns
    -------
    :
        Modified Laplacian operator, shape ``(Ny, Ny)``.
    """
    Ny = D2.shape[0]
    L = D2 - k2 * np.eye(Ny)

    if k2 == 0.0:
        # Pin p[0] = 0 for the singular mean mode
        L[0, :] = 0.0
        L[0, 0] = 1.0
        L[-1, :] = D1[-1, :]
    else:
        L[0, :] = D1[0, :]  # Neumann at bottom wall
        L[-1, :] = D1[-1, :]  # Neumann at top wall

    return L


def build_Hk_dirichlet(
    k2: float, D2: np.ndarray, dt: float, c: float, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build the Helmholtz operators with Dirichlet BCs for velocity.

    Returns both `$H_k = (1/\Delta t) I - c \nu (D_2 - k^2 I)$` (implicit) and
    `$H_k^- = (1/\Delta t) I + (1-c) \nu (D_2 - k^2 I)$` (explicit).
    Boundary rows are replaced by identity rows to encode no-slip
    conditions `$u|_{\mathrm{wall}} = 0$`.

    Parameters
    ----------
    k2:
        Squared horizontal wavenumber.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    dt:
        Time step.
    c:
        Implicitness parameter (0.5 = Crank-Nicolson).
    nu:
        Kinematic viscosity `$1/\mathrm{Re}$`.

    Returns
    -------
    Hk:
        Implicit Helmholtz operator, shape ``(Ny, Ny)``.
    Hk_minus:
        Explicit Helmholtz operator, shape ``(Ny, Ny)``.
    """
    Ny = D2.shape[0]
    Lk_raw = D2 - k2 * np.eye(Ny)  # without BC modification

    Hk = (1.0 / dt) * np.eye(Ny) - c * nu * Lk_raw
    Hk_minus = (1.0 / dt) * np.eye(Ny) + (1.0 - c) * nu * Lk_raw

    # Dirichlet BCs: u|_wall = 0
    for H in (Hk, Hk_minus):
        H[0, :] = 0.0
        H[0, 0] = 1.0
        H[-1, :] = 0.0
        H[-1, -1] = 1.0

    return Hk, Hk_minus


def precompute_imm(
    y: np.ndarray,
    kx_vals: np.ndarray,
    kz_vals: np.ndarray,
    p: int,
    dt: float,
    c: float,
    nu: float,
) -> dict[str, np.ndarray]:
    """Full offline precomputation for the influence-matrix method.

    Builds D1, D2, and then loops over all Fourier mode pairs
    `$(k_z, k_x)$` to construct the per-mode operators and IMM data.

    The output arrays are indexed ``[i_kz, i_kx, ...]``, matching the
    spectral array layout ``(ny, nz-1, nx//2)`` where ny is the y-grid
    dimension and the Fourier mode indices are the last two axes.

    All returned arrays are real (``float64``), following the proof in
    the IMM reference document.

    Parameters
    ----------
    y:
        Wall-normal grid points, shape ``(Ny,)``.
    kx_vals:
        Streamwise wavenumber values, shape ``(Nkx,)``.
    kz_vals:
        Spanwise wavenumber values, shape ``(Nkz,)``.
    p:
        Finite-difference accuracy order.
    dt:
        Time step.
    c:
        Implicitness parameter.
    nu:
        Kinematic viscosity `$1/\mathrm{Re}$`.

    Returns
    -------
    :
        Dictionary with keys:

        - ``"D1"``, ``"D2"``: derivative matrices, shape ``(Ny, Ny)``
        - ``"Lk"``: Neumann-modified Laplacian, ``(Nkz, Nkx, Ny, Ny)``
        - ``"Hk"``: implicit Helmholtz, ``(Nkz, Nkx, Ny, Ny)``
        - ``"Hk_minus"``: explicit Helmholtz, ``(Nkz, Nkx, Ny, Ny)``
        - ``"p1"``, ``"p2"``: homogeneous pressure solutions,
          ``(Nkz, Nkx, Ny)``
        - ``"M_inv"``: inverted influence matrix, ``(Nkz, Nkx, 2, 2)``
        - ``"k2"``: squared wavenumber per mode, ``(Nkz, Nkx)``
    """
    Ny = len(y)
    Nkz = len(kz_vals)
    Nkx = len(kx_vals)

    D1, D2 = build_diff_matrices(y, p)

    Lk_all = np.zeros((Nkz, Nkx, Ny, Ny))
    Hk_all = np.zeros((Nkz, Nkx, Ny, Ny))
    Hk_minus_all = np.zeros((Nkz, Nkx, Ny, Ny))
    p1_all = np.zeros((Nkz, Nkx, Ny))
    p2_all = np.zeros((Nkz, Nkx, Ny))
    M_inv_all = np.zeros((Nkz, Nkx, 2, 2))
    k2_all = np.zeros((Nkz, Nkx))

    e1 = np.zeros(Ny)
    e1[0] = 1.0
    e2 = np.zeros(Ny)
    e2[-1] = 1.0

    for iz, kz in enumerate(kz_vals):
        for ix, kx in enumerate(kx_vals):
            k2 = float(kx**2 + kz**2)
            k2_all[iz, ix] = k2

            # Laplacian with Neumann BCs (pressure)
            Lk = build_Lk_neumann(k2, D1, D2)
            Lk_all[iz, ix] = Lk

            # Helmholtz with Dirichlet BCs (velocity)
            Hk, Hk_minus = build_Hk_dirichlet(k2, D2, dt, c, nu)
            Hk_all[iz, ix] = Hk
            Hk_minus_all[iz, ix] = Hk_minus

            # Homogeneous pressure solutions and influence matrix
            if k2 == 0.0:
                # Mean mode: no IMM needed (pressure pinned)
                p1_all[iz, ix] = 0.0
                p2_all[iz, ix] = 0.0
                M_inv_all[iz, ix] = 0.0
            else:
                p1 = np.linalg.solve(Lk, e1)
                p2 = np.linalg.solve(Lk, e2)
                p1_all[iz, ix] = p1
                p2_all[iz, ix] = p2

                M = np.array(
                    [
                        [k2 * p1[0], k2 * p2[0]],
                        [k2 * p1[-1], k2 * p2[-1]],
                    ]
                )
                M_inv_all[iz, ix] = np.linalg.inv(M)

    return {
        "D1": D1,
        "D2": D2,
        "Lk": Lk_all,
        "Hk": Hk_all,
        "Hk_minus": Hk_minus_all,
        "p1": p1_all,
        "p2": p2_all,
        "M_inv": M_inv_all,
        "k2": k2_all,
    }
