"""Cartesian geometry: Fourier class, norms, integration, IMM, and solvers.

Provides all geometry-general infrastructure for wall-bounded Cartesian
flows: the ``Fourier`` wavenumber class, the ``CartesianFlow`` base
dataclass (CGL grid, FD matrices, IMM operators), spectral solvers
(influence-matrix method, predictor-corrector time stepping), and
diagnostic helpers (norms, perturbation energy).

Flow-specific modules (e.g. ``flows.plane_couette``) subclass
``CartesianFlow`` to define the base flow, then call
``build_cartesian_stepper`` to obtain ready-to-use time-stepping
functions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.scipy.linalg as sla
import numpy as np
from jax import Array
from jax import numpy as jnp

from ..fd import build_diff_matrices
from ..operators import (
    complex_harmonics,
    phys_to_spec_2d,
    real_harmonics,
    spec_to_phys_2d,
)
from ..parameters import derived_params, params
from ..rhs import get_nonlin
from ..sharding import register_dataclass_pytree, sharding
from ..timestep import make_stepper


@register_dataclass_pytree
@dataclass
class Fourier:
    """Wavenumber grids for the Cartesian wall-bounded geometry.

    Broadcasting shapes match the spectral layout ``(Nkz, Nkx, Ny)``:
    - ``kx``: shape ``(1, nx//2, 1)``
    - ``kz``: shape ``(nz-1, 1, 1)``

    ``k_metric`` equals 2 for `$k_x > 0$` and 1 for `$k_x = 0$`,
    accounting for the Hermitian symmetry of the real FFT.
    """

    kx: Array = field(init=False)
    kz: Array = field(init=False)
    k_metric: Array = field(init=False)
    k2: Array = field(init=False)
    k2_is_zero: Array = field(init=False)

    def __post_init__(self) -> None:
        kx_vals = real_harmonics(params.res.nx) * 2 * jnp.pi / params.geo.lx
        self.kx = jnp.asarray(
            kx_vals.reshape([1, -1, 1]),
            dtype=sharding.float_type,
            out_sharding=sharding.spec_scalar_shard,
        )
        kz_vals = complex_harmonics(params.res.nz) * 2 * jnp.pi / params.geo.lz
        self.kz = jnp.asarray(
            kz_vals.reshape([-1, 1, 1]),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )

        self.k_metric = jnp.where(self.kx == 0, 1, 2).astype(
            sharding.float_type
        )

        self.k2 = self.kx**2 + self.kz**2
        self.k2_is_zero = self.k2 == 0.0


fourier: Fourier = Fourier()


def get_inprod(
    vector_spec_1: Array, vector_spec_2: Array, k_metric: Array, ys: Array
) -> Array:
    """Volume-averaged L2 inner product ``<u1, u2>`` in spectral space.

    For Cartesian walled flows the Fourier modes in x
    and z are summed first, then the resulting y-profile is integrated
    with Simpson's rule.
    """
    return (
        integrate_scalar_in_y(
            jnp.sum(
                jnp.conj(vector_spec_1) * k_metric * vector_spec_2,
                dtype=sharding.float_type,
                axis=(0, 1, 2),
            ),
            ys,
        )
        / derived_params.ly
    )


def get_norm2(vector_spec: Array, k_metric: Array, ys: Array) -> Array:
    """Squared L2 norm ``||u||^2 = <u, u>``."""
    return get_inprod(vector_spec, vector_spec, k_metric, ys)


def get_norm(vector_spec: Array, k_metric: Array, ys: Array) -> Array:
    """L2 norm ``||u|| = sqrt(<u, u>)``."""
    return jnp.sqrt(get_norm2(vector_spec, k_metric, ys))


def integrate_scalar_in_y(scalar_data: Array, ys: Array) -> Array:
    """Composite Simpson's rule on a non-uniform grid in *y*.

    Requires an odd number of grid points (even number of sub-intervals).
    Uses the exact quadrature weights for pairs of non-uniform panels.

    Parameters
    ----------
    scalar_data:
        1-D array of function values at the grid points *ys*.
    ys:
        1-D array of grid-point coordinates (length must be odd).
    """

    if len(ys) % 2 == 0:
        sharding.print(
            "Simpson integration is not yet implemented "
            "for even # of grid points."
        )
        sharding.exit(code=1)

    h = jnp.diff(ys)  # shape (N-1,)
    h0 = h[:-1:2]  # left sub-intervals:  h0, h2, h4, ...
    h1 = h[1::2]  # right sub-intervals: h1, h3, h5, ...

    y0 = scalar_data[:-2:2]  # left points
    y1 = scalar_data[1:-1:2]  # mid points
    y2 = scalar_data[2::2]  # right points

    hsum = h0 + h1
    hprod = h0 * h1
    h0divh1 = h0 / h1

    panels = (hsum / 6) * (
        y0 * (2 - 1 / h0divh1) + y1 * (hsum**2 / hprod) + y2 * (2 - h0divh1)
    )
    return jnp.sum(panels)


@jax.jit
def _lu_solve(lu_pivots: tuple[Array, Array], b: Array) -> Array:
    """Batched LU solve across 2D (k_z, k_x) Fourier modes."""
    lu, piv = lu_pivots
    dtype = jnp.result_type(lu, b)
    lu = lu.astype(dtype)

    def solve_single(lu_piv, vec):
        return sla.lu_solve(lu_piv, vec)

    return jax.vmap(jax.vmap(solve_single))((lu, piv), b)


@register_dataclass_pytree
@dataclass
class DenseJAXSolver:
    """The current mathematically optimal dense LU cache."""

    matrix: Array
    lu: Array = field(init=False)
    piv: Array = field(init=False)

    def __post_init__(self) -> None:
        """Batch LU-factor over all ``(kz, kx)`` modes."""

        @jax.jit
        def batched_lu_factor(A: Array) -> tuple[Array, Array]:
            return jax.vmap(jax.vmap(sla.lu_factor))(A)

        self.lu, self.piv = batched_lu_factor(self.matrix)
        self.matrix = None

    def solve(self, rhs: Array) -> Array:
        """Batched LU solve.

        A leading batch axis (e.g. the 3 velocity components) is
        supported transparently by an extra ``vmap`` that leaves the
        cached LU factors untouched; this lets ``_imm_iteration`` do
        one stack-and-solve instead of three sequential kernel calls.

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(Nkz, Nkx, Ny)`` or
            ``(C, Nkz, Nkx, Ny)`` for a leading batch axis ``C``.

        Returns
        -------
        :
            Solution array, same shape as *rhs*.
        """
        if rhs.ndim == 4:
            return jax.vmap(_lu_solve, in_axes=(None, 0))(
                (self.lu, self.piv), rhs
            )
        return _lu_solve((self.lu, self.piv), rhs)


@jax.jit
def _spike_solve(
    lu: Array,
    piv: Array,
    V: Array,
    W: Array,
    red_lu: Array,
    red_piv: Array,
    rhs: Array,
) -> Array:
    """Solve `$A x = b$` via the SPIKE algorithm, single 3D RHS.

    The banded operator was partitioned at construction into ``P``
    block-rows of size ``m = N_y / P`` with bandwidth ``p``.  The
    spike matrices `$V_i = A_i^{-1} B_i$` and `$W_i = A_i^{-1} C_i$`
    capture the off-block coupling, and a small reduced system of
    size ``2 P p`` resolves the spike weights at block boundaries.

    Stages:

    1. Local solve `$A_i g_i = f_i$` (per-block dense LU solve,
       parallel across blocks).
    2. Build the reduced RHS from the top-`p` and bottom-`p` slices
       of each ``g_i``.
    3. Reduced solve for the spike weights
       `$\\alpha = (\\alpha^T_i, \\alpha^B_i)`.
    4. Reconstruct
       `$x_i = g_i - V_i \\alpha^T_{i+1} - W_i \\alpha^B_{i-1}$`,
       with neighbour weights zero at the matrix endpoints.

    Parameters
    ----------
    lu, piv:
        Per-block dense LU factors and pivots,
        ``(N_{kz}, N_{kx}, P, m, m)`` and ``(N_{kz}, N_{kx}, P, m)``.
    V, W:
        Spike matrices ``(N_{kz}, N_{kx}, P, m, p)``.
    red_lu, red_piv:
        Dense LU of the `$2 P p \\times 2 P p$` reduced system per
        ``(kz, kx)`` mode.
    rhs:
        Right-hand side, shape ``(N_{kz}, N_{kx}, N_y)``.

    Returns
    -------
    :
        Solution `$x$`, same shape and dtype as *rhs*.
    """
    P, m = lu.shape[-3], lu.shape[-2]
    p = V.shape[-1]
    Ny = P * m

    # Stage 1: local solve A_i g_i = f_i in parallel across blocks.
    rhs_blocks = rhs.reshape(rhs.shape[:-1] + (P, m))
    g = jax.vmap(jax.vmap(jax.vmap(sla.lu_solve)))((lu, piv), rhs_blocks)

    # Stage 2: reduced RHS from g top/bottom slices.
    g_top = g[..., :p]
    g_bot = g[..., m - p :]
    b_red_blocks = jnp.stack([g_top, g_bot], axis=-2)
    b_red = b_red_blocks.reshape(b_red_blocks.shape[:-3] + (2 * P * p,))

    # Stage 3: reduced solve.
    alpha = jax.vmap(jax.vmap(sla.lu_solve))((red_lu, red_piv), b_red)

    # Stage 4: extract per-block alpha^T / alpha^B, then shift.
    alpha_blocks = alpha.reshape(alpha.shape[:-1] + (P, 2, p))
    alpha_T = alpha_blocks[..., 0, :]
    alpha_B = alpha_blocks[..., 1, :]
    zeros_p = jnp.zeros_like(alpha_T[..., :1, :])
    alpha_T_next = jnp.concatenate([alpha_T[..., 1:, :], zeros_p], axis=-2)
    alpha_B_prev = jnp.concatenate([zeros_p, alpha_B[..., :-1, :]], axis=-2)

    # Stage 5: x_i = g_i - V_i alpha^T(i+1) - W_i alpha^B(i-1).
    V_contrib = jnp.einsum("...irc,...ic->...ir", V, alpha_T_next)
    W_contrib = jnp.einsum("...irc,...ic->...ir", W, alpha_B_prev)
    x_blocks = g - V_contrib - W_contrib

    return x_blocks.reshape(x_blocks.shape[:-2] + (Ny,))


@register_dataclass_pytree
@dataclass
class PerModeBandedOperator:
    """SPIKE-factored banded operator (band-preserving, GPU-fast).

    The original `$(N_y, N_y)$` banded operator (bandwidth ``p``) is
    partitioned at construction into ``P`` contiguous block-rows of
    size ``m = N_y / P`` (with ``m >= 2 p``) and factored locally via
    :func:`jax.scipy.linalg.lu_factor` (cuSOLVER-batched dense LU on
    the small `$(m, m)$` blocks).  Off-block coupling is captured by
    spike matrices `$V_i = A_i^{-1} B_i$` and `$W_i = A_i^{-1} C_i$`
    plus a small dense reduced system of size `$2 P p$`, also
    LU-factored once.

    No `$(N_y, N_y)$` array is ever materialised: per-mode storage
    is `$O(N_y m + (P p)^2) = O(N_y p)$`.  At solve time the only
    sequential work is the small reduced solve; the dominant
    per-block solve and the spike combination are all batched
    cuBLAS / cuSOLVER calls.

    Attributes
    ----------
    lu:
        Per-block dense LU factors, shape
        ``(N_{kz}, N_{kx}, P, m, m)``.
    piv:
        Per-block pivot indices, shape ``(N_{kz}, N_{kx}, P, m)``.
    V:
        Right-spike matrix, shape ``(N_{kz}, N_{kx}, P, m, p)``.
    W:
        Left-spike matrix, shape ``(N_{kz}, N_{kx}, P, m, p)``.
    red_lu:
        Dense LU of the reduced system, shape
        ``(N_{kz}, N_{kx}, 2 P p, 2 P p)``.
    red_piv:
        Pivots for the reduced LU, shape
        ``(N_{kz}, N_{kx}, 2 P p)``.
    """

    lu: Array
    piv: Array
    V: Array
    W: Array
    red_lu: Array
    red_piv: Array

    def solve(self, rhs: Array) -> Array:
        """Batched SPIKE solve across ``(kz, kx)`` modes.

        A leading batch axis (e.g. the 3 velocity components) is
        supported transparently by an extra ``vmap`` that leaves the
        cached factors untouched, so the same ``lu`` / ``V`` / ``W``
        / reduced LU are reused across all batched RHSs.

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(N_{kz}, N_{kx}, N_y)`` or
            ``(C, N_{kz}, N_{kx}, N_y)`` for a leading batch axis
            ``C``.  May be real or complex; the dtype is preserved.

        Returns
        -------
        :
            Solution array, same shape and dtype as *rhs*.
        """
        if rhs.ndim == 4:
            return jax.vmap(
                _spike_solve,
                in_axes=(None, None, None, None, None, None, 0),
            )(
                self.lu,
                self.piv,
                self.V,
                self.W,
                self.red_lu,
                self.red_piv,
                rhs,
            )
        return _spike_solve(
            self.lu,
            self.piv,
            self.V,
            self.W,
            self.red_lu,
            self.red_piv,
            rhs,
        )


# ── SPIKE block-partitioned operator builders ─────────────────────


def _spike_memory_per_mode(Ny: int, P: int, p: int) -> float:
    r"""Per-mode SPIKE storage: `$N_y^2 / P + 4 P^2 p^2$`."""
    return Ny * Ny / P + 4 * P * P * p * p


def _choose_block_partition(Ny: int, p: int) -> tuple[int, int]:
    r"""Choose SPIKE block count `$P$` and block size `$m$`.

    Picks the divisor `$P \ge 2$` of `$N_y$` (with
    `$m = N_y / P \ge 2 p$`) that minimises total per-mode
    SPIKE storage `$N_y^2 / P + 4 P^2 p^2$` (block LU
    factors plus reduced system).  Falls back to `$P = 1$`
    when `$N_y$` is prime or too small.

    Parameters
    ----------
    Ny:
        Wall-normal grid size.
    p:
        FD order (half-bandwidth of the banded operator).

    Returns
    -------
    P:
        Number of blocks.
    m:
        Block size (``Ny // P``).
    """
    min_m = max(2 * p, 1)
    max_P = max(Ny // min_m, 1)

    best_P, best_cost = 1, float("inf")
    for P_cand in range(2, max_P + 1):
        if Ny % P_cand == 0:
            cost = _spike_memory_per_mode(Ny, P_cand, p)
            if cost < best_cost:
                best_P, best_cost = P_cand, cost

    if best_P == 1:
        return 1, Ny
    return best_P, Ny // best_P


def _build_Lk_blocks_gpu(
    D1: Array,
    D2: Array,
    k2: Array,
    k2_is_zero: Array,
    p: int,
    P: int,
    m: int,
) -> tuple[Array, Array, Array]:
    r"""Build SPIKE block-partitioned `$L_k$` on GPU.

    The Neumann-BC pressure Poisson operator
    `$L_k = D_2 - k^2 I$` (with `$D_1$` wall rows for
    `$k^2 \ne 0$` and a row-0 pin for the `$k^2 = 0$`
    mean mode) is assembled directly into per-block dense
    form for the SPIKE factorisation.  No
    `$(N_y, N_y)$` matrix is materialised.

    Parameters
    ----------
    D1:
        First-derivative matrix, shape ``(Ny, Ny)``.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    k2:
        `$k_x^2 + k_z^2$`, shape ``(Nkz, Nkx, 1)``.
    k2_is_zero:
        Mean-mode boolean mask, same shape as *k2*.
    p:
        FD order (half-bandwidth).
    P:
        Number of SPIKE blocks.
    m:
        Block size (``Ny // P``).

    Returns
    -------
    A_blocks:
        Diagonal blocks, shape
        ``(N_{kz}, N_{kx}, P, m, m)``.
    B_corner:
        Right-coupling corners, shape
        ``(N_{kz}, N_{kx}, P, p, p)``.
        Zero for the last block.
    C_corner:
        Left-coupling corners, shape
        ``(N_{kz}, N_{kx}, P, p, p)``.
        Zero for the first block.
    """
    dtype = D2.dtype
    eye_m = jnp.eye(m, dtype=dtype)

    # Diagonal blocks of D2 (mode-independent).
    A_D2 = jnp.stack(
        [D2[i * m : (i + 1) * m, i * m : (i + 1) * m] for i in range(P)]
    )  # (P, m, m)

    # Lk = D2 - k2 * I, broadcast across (Nkz, Nkx).
    A_blocks = (
        A_D2[None, None] - k2[..., None, None] * eye_m
    )  # (Nkz, Nkx, P, m, m)

    # Row 0 (block 0, local row 0): D1[0,:m] for
    # k2 != 0, pin row [1,0,...,0] for k2 == 0.
    D1_row0 = D1[0, :m]
    pin_row = jnp.zeros(m, dtype=dtype).at[0].set(1.0)
    row0 = jnp.where(k2_is_zero, pin_row, D1_row0)
    A_blocks = A_blocks.at[:, :, 0, 0, :].set(row0)

    # Row Ny-1 (block P-1, local row m-1): D1[-1,:].
    D1_rowN = D1[-1, (P - 1) * m :]
    A_blocks = A_blocks.at[:, :, -1, -1, :].set(D1_rowN[None, None])

    # Coupling corners (mode-independent, from D2).
    B_list = []
    for i in range(P):
        if i < P - 1:
            r0 = (i + 1) * m - p
            c0 = (i + 1) * m
            B_list.append(D2[r0 : r0 + p, c0 : c0 + p])
        else:
            B_list.append(jnp.zeros((p, p), dtype=dtype))
    B_corner = jnp.broadcast_to(
        jnp.stack(B_list)[None, None],
        k2.shape[:2] + (P, p, p),
    )

    C_list = []
    for i in range(P):
        if i > 0:
            r0 = i * m
            c0 = i * m - p
            C_list.append(D2[r0 : r0 + p, c0 : c0 + p])
        else:
            C_list.append(jnp.zeros((p, p), dtype=dtype))
    C_corner = jnp.broadcast_to(
        jnp.stack(C_list)[None, None],
        k2.shape[:2] + (P, p, p),
    )

    return A_blocks, B_corner, C_corner


def _build_Hk_blocks_gpu(
    D2: Array,
    k2: Array,
    dt: float,
    c: float,
    nu: float,
    p: int,
    P: int,
    m: int,
) -> tuple[Array, Array, Array]:
    r"""Build SPIKE block-partitioned `$H_k$` on GPU.

    The implicit Helmholtz operator
    `$H_k = (1/\Delta t) I - c \nu (D_2 - k^2 I)$`
    with identity (Dirichlet) wall rows is assembled
    directly into per-block dense form.  No
    `$(N_y, N_y)$` matrix is materialised.

    Parameters
    ----------
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    k2:
        `$k_x^2 + k_z^2$`, shape ``(Nkz, Nkx, 1)``.
    dt:
        Time step.
    c:
        Implicitness parameter.
    nu:
        Kinematic viscosity `$1/\mathrm{Re}$`.
    p:
        FD order (half-bandwidth).
    P:
        Number of SPIKE blocks.
    m:
        Block size (``Ny // P``).

    Returns
    -------
    A_blocks:
        Diagonal blocks, shape
        ``(N_{kz}, N_{kx}, P, m, m)``.
    B_corner:
        Right-coupling corners, shape
        ``(N_{kz}, N_{kx}, P, p, p)``.
        Zero for the last block.
    C_corner:
        Left-coupling corners, shape
        ``(N_{kz}, N_{kx}, P, p, p)``.
        Zero for the first block.
    """
    dtype = D2.dtype
    eye_m = jnp.eye(m, dtype=dtype)

    A_D2 = jnp.stack(
        [D2[i * m : (i + 1) * m, i * m : (i + 1) * m] for i in range(P)]
    )  # (P, m, m)

    # Hk = (1/dt + c*nu*k2) I - c*nu*D2
    diag_coeff = (1.0 / dt) + c * nu * k2  # (Nkz, Nkx, 1)
    A_blocks = (
        diag_coeff[..., None, None] * eye_m - c * nu * A_D2[None, None]
    )  # (Nkz, Nkx, P, m, m)

    # Dirichlet identity wall rows.
    e0 = jnp.zeros(m, dtype=dtype).at[0].set(1.0)
    eN = jnp.zeros(m, dtype=dtype).at[-1].set(1.0)
    A_blocks = A_blocks.at[:, :, 0, 0, :].set(e0)
    A_blocks = A_blocks.at[:, :, -1, -1, :].set(eN)

    # Coupling corners: -c*nu * D2 sub-blocks.
    c_nu = c * nu
    B_list = []
    for i in range(P):
        if i < P - 1:
            r0 = (i + 1) * m - p
            c0 = (i + 1) * m
            B_list.append(-c_nu * D2[r0 : r0 + p, c0 : c0 + p])
        else:
            B_list.append(jnp.zeros((p, p), dtype=dtype))
    B_corner = jnp.broadcast_to(
        jnp.stack(B_list)[None, None],
        k2.shape[:2] + (P, p, p),
    )

    C_list = []
    for i in range(P):
        if i > 0:
            r0 = i * m
            c0 = i * m - p
            C_list.append(-c_nu * D2[r0 : r0 + p, c0 : c0 + p])
        else:
            C_list.append(jnp.zeros((p, p), dtype=dtype))
    C_corner = jnp.broadcast_to(
        jnp.stack(C_list)[None, None],
        k2.shape[:2] + (P, p, p),
    )

    return A_blocks, B_corner, C_corner


def _build_reduced_matrix(V: Array, W: Array, p: int) -> Array:
    r"""Assemble the SPIKE reduced system from spike tips.

    The reduced matrix has size
    `$2 P p \times 2 P p$` with identity diagonal blocks
    and `$(p \times p)$` off-diagonal couplings from the
    top/bottom `$p$` rows of the spike matrices `$V_i$`
    and `$W_i$`.

    Parameters
    ----------
    V:
        Right-spike matrix,
        ``(N_{kz}, N_{kx}, P, m, p)``.
    W:
        Left-spike matrix,
        ``(N_{kz}, N_{kx}, P, m, p)``.
    p:
        Half-bandwidth (spike tip size).

    Returns
    -------
    :
        Reduced matrix, shape
        ``(N_{kz}, N_{kx}, 2 P p, 2 P p)``.
    """
    P = V.shape[-3]
    m = V.shape[-2]
    n_red = 2 * P * p
    dtype = V.dtype

    V_top = V[..., :p, :]
    V_bot = V[..., m - p :, :]
    W_top = W[..., :p, :]
    W_bot = W[..., m - p :, :]

    # Derive batch dims from V so A_red inherits kx-sharding
    # (jnp.broadcast_to on an unsharded eye would lose sharding
    # under the Explicit mesh).
    ones_kx = jnp.ones_like(V[..., 0, 0, 0])  # (Nkz, Nkx)
    A_red = ones_kx[..., None, None] * jnp.eye(n_red, dtype=dtype)

    for i in range(P):
        rT = 2 * i * p
        rB = (2 * i + 1) * p

        if i < P - 1:
            c_next = 2 * (i + 1) * p
            A_red = A_red.at[..., rT : rT + p, c_next : c_next + p].set(
                V_top[..., i, :, :]
            )
            A_red = A_red.at[..., rB : rB + p, c_next : c_next + p].set(
                V_bot[..., i, :, :]
            )

        if i > 0:
            c_prev = (2 * (i - 1) + 1) * p
            A_red = A_red.at[..., rT : rT + p, c_prev : c_prev + p].set(
                W_top[..., i, :, :]
            )
            A_red = A_red.at[..., rB : rB + p, c_prev : c_prev + p].set(
                W_bot[..., i, :, :]
            )

    return A_red


def _spike_factor(
    A_blocks: Array,
    B_corner: Array,
    C_corner: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    r"""SPIKE factorisation of a block-partitioned banded operator.

    Performs per-block dense LU (cuSOLVER batched), spike
    matrix solves `$V_i = A_i^{-1} B_i$`,
    `$W_i = A_i^{-1} C_i$`, and reduced-system LU — all
    on the GPU with no `$(N_y, N_y)$` array.

    Array sharding is handled eagerly via
    ``out_sharding`` on the allocating calls before
    the JIT'd compute kernels, because the ``Explicit``
    mesh type used by :mod:`sharding` does not support
    resharding inside
    ``jax.lax.with_sharding_constraint``.

    Parameters
    ----------
    A_blocks:
        Diagonal blocks, shape
        ``(N_{kz}, N_{kx}, P, m, m)``,
        kx-sharded.
    B_corner:
        Right-coupling corners, shape
        ``(N_{kz}, N_{kx}, P, p, p)``.
    C_corner:
        Left-coupling corners, shape
        ``(N_{kz}, N_{kx}, P, p, p)``.

    Returns
    -------
    lu, piv:
        Per-block dense LU factors and pivots.
    V, W:
        Spike matrices,
        ``(N_{kz}, N_{kx}, P, m, p)``.
    red_lu, red_piv:
        Dense LU of the reduced system.
    """
    m = A_blocks.shape[-2]
    p = B_corner.shape[-1]
    dtype = A_blocks.dtype

    # Expand p x p corners to full (m, p) RHS, sharded
    # to match the kx-sharded LU factors.
    B_full = jnp.zeros(
        A_blocks.shape[:-1] + (p,),
        dtype=dtype,
        out_sharding=sharding.spec_dy_blocks_shard,
    )
    B_full = B_full.at[..., m - p :, :].set(B_corner)

    C_full = jnp.zeros(
        A_blocks.shape[:-1] + (p,),
        dtype=dtype,
        out_sharding=sharding.spec_dy_blocks_shard,
    )
    C_full = C_full.at[..., :p, :].set(C_corner)

    @jax.jit
    def _do_factor(A, B, C):
        """JIT-compiled SPIKE factorisation kernel."""
        # Per-block dense LU (cuSOLVER batched).
        lu, piv = jax.vmap(jax.vmap(jax.vmap(sla.lu_factor)))(A)

        # Spike solves: V_i = A_i^{-1} B_i,
        #               W_i = A_i^{-1} C_i.
        V = jax.vmap(jax.vmap(jax.vmap(sla.lu_solve)))((lu, piv), B)
        W = jax.vmap(jax.vmap(jax.vmap(sla.lu_solve)))((lu, piv), C)

        # Reduced system assembly and factorisation.
        A_red = _build_reduced_matrix(V, W, p)
        red_lu, red_piv = jax.vmap(jax.vmap(sla.lu_factor))(A_red)

        return lu, piv, V, W, red_lu, red_piv

    return _do_factor(A_blocks, B_full, C_full)


def _build_Lk_dense_gpu(
    D1: Array, D2: Array, k2: Array, k2_is_zero: Array
) -> Array:
    """Build the Neumann-BC Laplacian `$L_k$` in dense form on GPU.

    Used only by the ``"dense"`` solver backend; allocates
    `$(N_{kz}, N_{kx}, N_y, N_y)$`.  No CPU path.

    Parameters / returns follow
    :func:`_build_Lk_blocks_gpu`, but the output is the
    full dense operator.
    """
    Ny = D2.shape[-1]
    eye = jnp.eye(Ny, dtype=D2.dtype)
    # Lk_interior[..., i, j] = D2[i, j] - k2 * delta_{i, j}
    Lk = D2[None, None, :, :] - k2[..., None] * eye

    # Row 0: D1[0, :] for k2 != 0; pin row [1, 0, ..., 0] for k2 == 0.
    # k2_is_zero is (Nkz, Nkx, 1); `jnp.where` broadcasts the (Ny,)
    # branches along the last axis to give (Nkz, Nkx, Ny).
    pin = eye[0, :]  # (Ny,)
    row_0 = jnp.where(k2_is_zero, pin, D1[0, :])
    Lk = Lk.at[..., 0, :].set(row_0)

    # Row -1: D1[-1, :] for all modes.
    Lk = Lk.at[..., -1, :].set(D1[-1, :])

    return Lk


def _build_Hk_dense_gpu(
    D2: Array, k2: Array, dt: float, c: float, nu: float
) -> Array:
    """Build dense `$H_k$` on GPU (dense backend only).

    Returns the implicit operator
    `$H_k = (1/\\Delta t) I - c \\nu (D_2 - k^2 I)$`
    with identity wall rows for no-slip Dirichlet BCs.
    The explicit counterpart `$H_k^-$` is applied matrix-free
    by :func:`_hk_minus_matvec`.
    """
    Ny = D2.shape[-1]
    eye = jnp.eye(Ny, dtype=D2.dtype)
    Lk_raw = D2[None, None, :, :] - k2[..., None] * eye

    Hk = (1.0 / dt) * eye - c * nu * Lk_raw

    # Dirichlet identity rows.
    zero_row = jnp.zeros(Ny, dtype=D2.dtype)
    e_0 = zero_row.at[0].set(1.0)
    e_Nm1 = zero_row.at[-1].set(1.0)
    Hk = Hk.at[..., 0, :].set(e_0).at[..., -1, :].set(e_Nm1)

    return Hk


# ── CartesianFlow base dataclass ─────────────────────────────────────────


@register_dataclass_pytree
@dataclass
class CartesianFlow:
    """Precomputed data for wall-bounded Cartesian flows.

    Subclasses must set ``base_flow``, ``curl_base_flow``, and
    ``nonlin_base_flow`` *after* calling ``super().__post_init__()``,
    which builds the CGL grid (``ys``), finite-difference matrices,
    and all per-mode IMM operators.
    """

    ys: Array = field(init=False)
    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    nonlin_base_flow: Array = field(init=False)
    D1: Array = field(init=False)
    D2: Array = field(init=False)
    D1_bnd: Array = field(init=False)
    D2_bnd: Array = field(init=False)
    Lk_op: DenseJAXSolver | PerModeBandedOperator = field(init=False)
    Hk_op: DenseJAXSolver | PerModeBandedOperator = field(init=False)
    p1: Array = field(init=False)
    p2: Array = field(init=False)
    v1: Array = field(init=False)
    v2: Array = field(init=False)
    q1: Array = field(init=False)
    q2: Array = field(init=False)
    M_inv: Array = field(init=False)

    def __post_init__(self) -> None:
        """Build CGL grid, FD matrices, and IMM operators.

        Constructs the Chebyshev-Gauss-Lobatto grid for
        the wall-normal coordinate `$y$` in `$[-1, 1]$`,
        FD matrices `$D_1$` and `$D_2$`, and all per-mode
        IMM operators directly on the device.  Under the
        banded backend, `$L_k$` and `$H_k$` are assembled
        into per-block dense form and factorised via the
        SPIKE algorithm (:func:`_spike_factor`): per-block
        dense LU on `$(m, m)$` blocks (cuSOLVER batched)
        plus a small reduced system, with no
        `$(N_y, N_y)$` array materialised.  Under the
        dense backend they are built as full
        `$(N_y, N_y)$` blocks via
        :func:`_build_Lk_dense_gpu` /
        :func:`_build_Hk_dense_gpu` and factorised by
        :class:`DenseJAXSolver`.  Homogeneous IMM data
        (``p1..q2``, ``M_inv``) is derived from the GPU
        operator by :meth:`_derive_imm_homogeneous_data`.
        """
        self.ys = -jnp.cos(
            jnp.arange(params.res.ny, dtype=sharding.float_type)
            * jnp.pi
            / (params.res.ny - 1)
        )

        # ``build_diff_matrices`` stays on the CPU: the ``(Ny, Ny)``
        # derivative matrices are tiny, used once for the mean-mode
        # gauge fix and the IMM operator construction, and copied to
        # the GPU immediately below.
        D1, D2 = build_diff_matrices(np.array(self.ys), params.res.fd_order)
        self.D1 = jnp.asarray(jnp.asarray(D1), out_sharding=sharding.no_shard)
        self.D2 = jnp.asarray(jnp.asarray(D2), out_sharding=sharding.no_shard)
        self.D1_bnd = jnp.asarray(
            jnp.asarray(D1[[0, -1], :]),
            out_sharding=sharding.no_shard,
        )
        self.D2_bnd = jnp.asarray(
            jnp.asarray(D2[[0, -1], :]),
            out_sharding=sharding.no_shard,
        )

        Nkz = params.res.nz - 1
        Nkx = params.res.nx // 2
        Ny = params.res.ny

        p = params.res.fd_order
        dt = params.step.dt
        c = params.step.implicitness
        nu = 1.0 / params.phys.re

        if params.solver.backend == "banded":
            # SPIKE block-partitioned factorisation: dense LU
            # on small (m, m) blocks, no (Ny, Ny) array.
            sbs = params.solver.spike_block_size
            if sbs is not None:
                if Ny % sbs != 0 or sbs < 2 * p:
                    sharding.print(
                        f"spike_block_size={sbs} invalid for "
                        f"Ny={Ny}, p={p}; falling back to auto."
                    )
                    P_blk, m_blk = _choose_block_partition(Ny, p)
                else:
                    P_blk, m_blk = Ny // sbs, sbs
            else:
                P_blk, m_blk = _choose_block_partition(Ny, p)
            sharding.print(
                f"SPIKE partition: P={P_blk}, m={m_blk} (Ny={Ny}, p={p})"
            )
            Lk_A, Lk_B, Lk_C = _build_Lk_blocks_gpu(
                self.D1,
                self.D2,
                fourier.k2,
                fourier.k2_is_zero,
                p,
                P_blk,
                m_blk,
            )
            Hk_A, Hk_B, Hk_C = _build_Hk_blocks_gpu(
                self.D2,
                fourier.k2,
                dt,
                c,
                nu,
                p,
                P_blk,
                m_blk,
            )
            self.Lk_op = PerModeBandedOperator(
                *_spike_factor(Lk_A, Lk_B, Lk_C)
            )
            self.Hk_op = PerModeBandedOperator(
                *_spike_factor(Hk_A, Hk_B, Hk_C)
            )
        else:
            # Dense backend: parity/reference path.  Full
            # `(Nkz, Nkx, Ny, Ny)` matrices are built, LU-factored,
            # then discarded — only the factors are kept.
            Lk_dense = _build_Lk_dense_gpu(
                self.D1, self.D2, fourier.k2, fourier.k2_is_zero
            )
            Hk_dense = _build_Hk_dense_gpu(self.D2, fourier.k2, dt, c, nu)
            self.Lk_op = DenseJAXSolver(Lk_dense)
            self.Hk_op = DenseJAXSolver(Hk_dense)

        self._derive_imm_homogeneous_data(Nkz, Nkx, Ny)

    def _derive_imm_homogeneous_data(
        self, Nkz: int, Nkx: int, Ny: int
    ) -> None:
        """Fill ``p1..q2`` and ``M_inv`` on-device from the GPU operator.

        Both backends converge here once :attr:`Lk_op` and :attr:`Hk_op`
        are in place.  Nothing else on the CPU needs to do another LU
        solve — everything below runs against the already-factored
        device operator.

        The mean mode (`$k^2 = 0$`) is handled analytically: ``M`` has a
        zero first column there (`$p_1 \\equiv 1$` is a pressure gauge),
        so the 2x2 inverse is replaced by `$[[0, 0], [0, 1/M_{11}]]$`
        as in the original CPU path.  The ``jnp.where`` around
        ``safe_det`` keeps the regular branch NaN-free before the
        selection happens.
        """
        # Homogeneous pressure solutions `$L_k p_i = e_i$`.
        e1_b = (
            jnp.zeros(
                (Nkz, Nkx, Ny),
                dtype=sharding.float_type,
                out_sharding=sharding.spec_imm_corr_shard,
            )
            .at[..., 0]
            .set(1.0)
        )
        e2_b = (
            jnp.zeros(
                (Nkz, Nkx, Ny),
                dtype=sharding.float_type,
                out_sharding=sharding.spec_imm_corr_shard,
            )
            .at[..., -1]
            .set(1.0)
        )
        self.p1 = self.Lk_op.solve(e1_b)
        self.p2 = self.Lk_op.solve(e2_b)

        # Homogeneous velocity solutions `$v_i = H_k^{-1} (-D_1 p_i)$`
        # with zero Dirichlet BCs (no-slip).
        rhs_v1 = -jnp.einsum("ij, zxj -> zxi", self.D1, self.p1)
        rhs_v2 = -jnp.einsum("ij, zxj -> zxi", self.D1, self.p2)
        rhs_v1 = rhs_v1.at[..., 0].set(0.0).at[..., -1].set(0.0)
        rhs_v2 = rhs_v2.at[..., 0].set(0.0).at[..., -1].set(0.0)
        self.v1 = self.Hk_op.solve(rhs_v1)
        self.v2 = self.Hk_op.solve(rhs_v2)

        # Homogeneous velocity potentials `$q_i = H_k^{-1} p_i$` with
        # zero Dirichlet BCs.
        q_rhs1 = self.p1.at[..., 0].set(0.0).at[..., -1].set(0.0)
        q_rhs2 = self.p2.at[..., 0].set(0.0).at[..., -1].set(0.0)
        self.q1 = self.Hk_op.solve(q_rhs1)
        self.q2 = self.Hk_op.solve(q_rhs2)

        # Influence matrix `$M_{ji} = (D_1 v_i)|_{\\text{wall}_j}$`.
        M00 = jnp.einsum("j, zxj -> zx", self.D1_bnd[0], self.v1)
        M01 = jnp.einsum("j, zxj -> zx", self.D1_bnd[0], self.v2)
        M10 = jnp.einsum("j, zxj -> zx", self.D1_bnd[-1], self.v1)
        M11 = jnp.einsum("j, zxj -> zx", self.D1_bnd[-1], self.v2)

        is_mean = fourier.k2_is_zero[..., 0]
        det = M00 * M11 - M01 * M10
        safe_det = jnp.where(is_mean, 1.0, det)
        inv_00 = jnp.where(is_mean, 0.0, M11 / safe_det)
        inv_01 = jnp.where(is_mean, 0.0, -M01 / safe_det)
        inv_10 = jnp.where(is_mean, 0.0, -M10 / safe_det)
        inv_11 = jnp.where(is_mean, 1.0 / M11, M00 / safe_det)
        self.M_inv = jnp.stack(
            [
                jnp.stack([inv_00, inv_01], axis=-1),
                jnp.stack([inv_10, inv_11], axis=-1),
            ],
            axis=-2,
        )


# ── Spectral transform aliases ───────────────────────────────────────────

phys_to_spec = phys_to_spec_2d
spec_to_phys = spec_to_phys_2d


# ── Solver functions (geometry-general) ──────────────────────────────────


def init_state(snapshot: str | None) -> Array:
    """Initialise the flow state (velocity_spec)."""
    if params.init.start_from_laminar:
        velocity_spec = jnp.zeros(
            shape=(3, *sharding.spec_shape),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
    elif snapshot is not None:
        snapshot_arr = jnp.load(snapshot)["velocity_phys_nonexpanded"].astype(
            sharding.float_type
        )
        velocity_phys = jnp.asarray(
            jnp.asarray(snapshot_arr),
            out_sharding=sharding.phys_vector_shard,
        )
        velocity_spec = phys_to_spec_2d(velocity_phys)

    else:
        sharding.print("Provide an initial condition.")
        sharding.exit(code=1)

    return velocity_spec


def _curl_fn(
    state: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> Array:
    """Spectral curl with 1D FD in y and spectral derivatives in x and z."""
    u, v, w = state[0], state[1], state[2]

    # Stack (u, w) so the two D1 y-derivatives needed for the curl
    # are one batched GEMM rather than two separate kernel launches.
    dy_uw = jnp.einsum("ij, czxj -> czxi", flow_.D1, jnp.stack([u, w]))
    dy_u, dy_w = dy_uw[0], dy_uw[1]

    dx_v = 1j * fourier_.kx * v
    dz_v = 1j * fourier_.kz * v
    dx_w = 1j * fourier_.kx * w
    dz_u = 1j * fourier_.kz * u

    omega_x = dy_w - dz_v
    omega_y = dz_u - dx_w
    omega_z = dx_v - dy_u

    return jnp.array([omega_x, omega_y, omega_z])


def _get_rhs(
    state: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> Array:
    """Evaluate non-linear RHS terms."""
    nonlin = get_nonlin(
        state,
        flow_.base_flow,
        flow_.curl_base_flow,
        flow_.nonlin_base_flow,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier_, flow_),
    )
    return nonlin


def _lk_matvec(
    u: Array,
    D2: Array,
    D1_bnd: Array,
    k2: Array,
    k2_is_zero: Array,
) -> Array:
    """Apply `$L_k u$` for the Neumann-BC pressure Poisson operator.

    Matrix-free evaluation that avoids storing the per-mode
    ``(Nkz, Nkx, Ny, Ny)`` operator.  The interior of the output is
    `$D_2 u - k^2 u$`; the wall rows use `$D_1$` to encode Neumann
    BCs, except for the `$k^2 = 0$` mean mode where row 0 pins
    `$p_0 = 0$` (matching :func:`build_Lk_neumann`).

    Parameters
    ----------
    u:
        Field, shape ``(Nkz, Nkx, Ny)``.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    D1_bnd:
        Boundary rows `$D_1[0,:]$`, `$D_1[-1,:]$`, shape ``(2, Ny)``.
    k2:
        Squared horizontal wavenumber, broadcasting as
        ``(Nkz, Nkx, 1)``.
    k2_is_zero:
        Boolean mask ``k2 == 0``, same shape as *k2*.

    Returns
    -------
    :
        ``Lk @ u`` with the same shape and dtype as *u*.
    """
    D2u = jnp.einsum("ij, zxj -> zxi", D2, u)
    out = D2u - k2 * u
    bot_neumann = jnp.einsum("j, zxj -> zx", D1_bnd[0], u)
    bot = jnp.where(k2_is_zero[..., 0], u[..., 0], bot_neumann)
    top = jnp.einsum("j, zxj -> zx", D1_bnd[-1], u)
    return out.at[..., 0].set(bot).at[..., -1].set(top)


def _hk_minus_matvec(
    u: Array, D2: Array, k2: Array, dt: float, c: float, nu: float
) -> Array:
    """Apply `$H_k^- u$` for the explicit-side Helmholtz operator.

    Matrix-free evaluation of ``flow_.Hk_minus @ u``:
    `$\\tfrac{1}{\\Delta t} u + (1 - c) \\nu (D_2 u - k^2 u)$` in the
    interior, with identity wall rows (`$u|_\\text{wall}$` unchanged).

    Parameters
    ----------
    u:
        Field, shape ``(Nkz, Nkx, Ny)``.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    k2:
        Squared horizontal wavenumber, broadcasting as
        ``(Nkz, Nkx, 1)``.
    dt:
        Time step.
    c:
        Implicitness parameter.
    nu:
        Kinematic viscosity `$1/\\mathrm{Re}$`.

    Returns
    -------
    :
        ``Hk_minus @ u`` with the same shape and dtype as *u*.
    """
    D2u = jnp.einsum("ij, zxj -> zxi", D2, u)
    out = (1.0 / dt) * u + (1.0 - c) * nu * (D2u - k2 * u)
    return out.at[..., 0].set(u[..., 0]).at[..., -1].set(u[..., -1])


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> tuple[Array, Array]:
    """Kleiser-Schumann influence-matrix method.

    The y-momentum equation supplies only the *interior* Poisson
    equation for pressure; the wall BC is determined indirectly by
    enforcing continuity `$\\nabla \\cdot u = 0$` at the walls.

    Six stages:
    1. Build the interior Poisson RHS from divergence of momentum.
    2. Solve Poisson for the particular pressure `$p_P$` with
       arbitrary (zero) Neumann BCs.
    3. Solve Helmholtz for all three particular velocity components
       `$u_{arb}, v_{arb}, w_{arb}$` against `$p_P$` (zero Dirichlet
       BCs).
    4. Compute wall divergence residual `$d_{\\mathrm{wall}} = (D_1
       v_{arb})|_{\\mathrm{wall}}$` (since `$u = w = 0$` at walls).
    5. Apply the influence matrix `$\\alpha = -M^{-1} d_{\\mathrm{wall}}$`.
    6. Assemble the corrected pressure and all three corrected
       velocity components via Helmholtz linearity, with no further
       Helmholtz solves:

       - `$p = p_P + \\alpha_1 p_1 + \\alpha_2 p_2$`
       - `$v = v_{arb} + \\alpha_1 v_1 + \\alpha_2 v_2$`
       - `$u = u_{arb} - i k_x \\Delta q$`
       - `$w = w_{arb} - i k_z \\Delta q$`

       where `$\\Delta q = \\alpha_1 q_1 + \\alpha_2 q_2$` and
       `$q_i = H_k^{-1} p_i$` (precomputed), using the factorisation
       `$u^{(i)} = -i k_x q_i$`, `$w^{(i)} = -i k_z q_i$` (the scalar
       `$-i k_x$`, `$-i k_z$` commute with `$H_k^{-1}$` per mode).
    """
    c = params.step.implicitness
    dt = params.step.dt
    nu = 1.0 / params.phys.re

    u_n, v_n, w_n = velocity_n[0], velocity_n[1], velocity_n[2]
    Nu_n, Nv_n, Nw_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    Nu_j, Nv_j, Nw_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    # Squared horizontal wavenumber, broadcastable to (Nkz, Nkx, Ny).
    k2 = fourier_.k2
    k2_is_zero = fourier_.k2_is_zero

    # Horizontal spectral-derivative factors, reused across every stage.
    ikx = 1j * fourier_.kx
    ikz = 1j * fourier_.kz

    # Batch the three D1 y-derivatives into one GEMM.
    dy_stack = jnp.einsum(
        "ij, czxj -> czxi",
        flow_.D1,
        jnp.stack([v_n, Nv_j, Nv_n]),
    )
    dy_v_n, dy_Nv_j, dy_Nv_n = dy_stack[0], dy_stack[1], dy_stack[2]

    # d_hat^n (discrete divergence at time n; ~0 after first step).
    d_hat_n = ikx * u_n + dy_v_n + ikz * w_n

    # Stage 1: interior pressure Poisson RHS.
    div_Nj = ikx * Nu_j + dy_Nv_j + ikz * Nw_j
    div_Nn = ikx * Nu_n + dy_Nv_n + ikz * Nw_n

    Lk_d = _lk_matvec(d_hat_n, flow_.D2, flow_.D1_bnd, k2, k2_is_zero)

    f_hat = d_hat_n / dt + c * div_Nj + (1 - c) * div_Nn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure with ZERO Neumann BCs.
    f_hat_P = f_hat.at[..., 0].set(0.0).at[..., -1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: Helmholtz solves for all three velocity components
    # against the particular pressure p_P (zero Dirichlet BCs).  The
    # three components share the same :math:`H_k` operator per mode,
    # so the explicit matvec, the wall-row zeroing, and the final
    # solve are all batched over the component axis — one kernel
    # launch each instead of three sequential ones.
    dx_pP = ikx * pP
    dy_pP = jnp.einsum("ij, zxj -> zxi", flow_.D1, pP)
    dz_pP = ikz * pP
    grad_pP = jnp.stack([dx_pP, dy_pP, dz_pP])  # (3, Nkz, Nkx, Ny)

    Hk_minus_stack = jax.vmap(
        _hk_minus_matvec,
        in_axes=(0, None, None, None, None, None),
    )(velocity_n, flow_.D2, k2, dt, c, nu)

    R_stack = Hk_minus_stack - grad_pP + c * nonlin_j + (1 - c) * nonlin_n
    R_stack = R_stack.at[..., 0].set(0.0).at[..., -1].set(0.0)

    arb_stack = flow_.Hk_op.solve(R_stack)
    u_arb, v_arb, w_arb = arb_stack[0], arb_stack[1], arb_stack[2]

    # Stage 4: wall divergence residual. At walls u=w=0 (no-slip),
    # so div u|_wall = D1 v|_wall.
    d_wall = jnp.einsum("bj, zxj -> zxb", flow_.D1_bnd, v_arb)

    # Mean mode (k²=0) bottom-wall residual is a pressure gauge; zero it.
    d_wall = d_wall.at[..., 0].set(
        jnp.where(k2_is_zero[..., 0], 0.0, d_wall[..., 0])
    )

    # Stage 5: influence matrix algebra alpha = -M_inv @ d_wall.
    alpha = -jnp.einsum("zxab, zxb -> zxa", flow_.M_inv, d_wall)
    alpha1 = alpha[..., 0][..., None]
    alpha2 = alpha[..., 1][..., None]

    # Stage 6: corrected pressure and all three velocity components
    # via Helmholtz linearity — no additional Helmholtz solves.
    # p_new = pP + alpha1 * flow_.p1 + alpha2 * flow_.p2
    v_new = v_arb + alpha1 * flow_.v1 + alpha2 * flow_.v2

    # Horizontal corrections factor through the scalar potential Δq,
    # since u^(i) = -ikx q_i and w^(i) = -ikz q_i (the -ikx, -ikz
    # scalar factors commute with Hk linearity per mode).
    q_new = alpha1 * flow_.q1 + alpha2 * flow_.q2
    u_new = u_arb - ikx * q_new
    w_new = w_arb - ikz * q_new

    velocity_new = jnp.array([u_new, v_new, w_new])

    correction = velocity_new - velocity_j

    return velocity_new, correction


def _predict(
    velocity_n: Array,
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> Array:
    """Euler predictor (Willis 2017 j=0) via Kleiser-Schumann IMM."""
    nonlin_n = rhs_no_lapl

    prediction_state, _ = _imm_iteration(
        velocity_n, velocity_n, nonlin_n, nonlin_n, fourier_, flow_
    )
    return prediction_state


def _correct(
    state_prev: Array,
    prediction_state: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector (Willis 2017 j>0) via Kleiser-Schumann IMM."""
    velocity_n = state_prev
    velocity_j = prediction_state

    nonlin_n = rhs_prev
    nonlin_j = rhs_next

    prediction_state_new, correction = _imm_iteration(
        velocity_n, velocity_j, nonlin_n, nonlin_j, fourier_, flow_
    )
    return prediction_state_new, correction


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> Array:
    """L2 convergence norm."""
    return jnp.sqrt(get_norm2(correction, fourier_.k_metric, flow_.ys))


# ── Stepper factory ─────────────────────────────────────────────────────


def build_cartesian_stepper(
    flow: CartesianFlow,
) -> tuple[
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array, Array, Array], tuple[Array, Array, Array]],
    Callable[[str | None], Array],
    Callable[[Array], tuple[Array, Array, Array, Array]],
]:
    """Build time-stepping functions for a Cartesian wall-bounded flow.

    Returns ``(predict_and_correct, iterate_correction,
    init_state_bound, predict_and_fully_correct)`` with the
    ``fourier`` and *flow* singletons already bound.
    """
    (
        _predict_and_correct_jit,
        _iterate_correction_jit,
        _predict_and_fully_correct_jit,
    ) = make_stepper(_get_rhs, _predict, _correct, _norm)

    def predict_and_correct(
        state: Array,
    ) -> tuple[Array, Array, Array]:
        """Predictor-corrector step with bound singletons."""
        return _predict_and_correct_jit(state, fourier, flow)

    def iterate_correction(
        state_prev: Array,
        prediction: Array,
        rhs_prev: Array,
    ) -> tuple[Array, Array, Array]:
        """One corrector iteration with bound singletons."""
        return _iterate_correction_jit(
            state_prev, prediction, rhs_prev, fourier, flow
        )

    def predict_and_fully_correct(
        state: Array,
    ) -> tuple[Array, Array, Array, Array]:
        """Fused predict + corrector loop with bound singletons."""
        return _predict_and_fully_correct_jit(state, fourier, flow)

    def init_state_bound(snapshot: str | None) -> Array:
        """Initialize the flow state."""
        return init_state(snapshot)

    return (
        predict_and_correct,
        iterate_correction,
        init_state_bound,
        predict_and_fully_correct,
    )
