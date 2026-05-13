"""Geometry-independent linear solver infrastructure.

Provides batched dense LU solvers (:class:`DenseJAXSolver`) and the
SPIKE block-partitioned banded solver
(:class:`PerModeBandedOperator`) used by wall-bounded geometries.
Both solver classes support a leading batch axis (e.g. the 3 velocity
components) transparently via an extra ``vmap``.

The SPIKE algorithm (Polizzi & Sameh 2006) partitions a banded
`$(N_y, N_y)$` operator into ``P`` contiguous blocks of size
``m = N_y / P`` (with ``m >= 2p``, where ``p`` is the
half-bandwidth) and factors each block as a dense `$(m, m)$` LU via
cuSOLVER's batched LU.  Spike matrices
`$V_i = A_i^{-1} B_i$`, `$W_i = A_i^{-1} C_i$` capture the
off-block coupling, and a small dense reduced system of size
`$2 P p$` is also LU-factored once.  At solve time, per-block LU
solves, a tiny reduced solve, and a spike reconstruction replace
a full dense solve -- all cuSOLVER-batched.

Utility helpers :func:`_extract_banded_corners` and
:func:`_choose_block_partition` support the block decomposition and
optimal partitioning.
"""

from dataclasses import dataclass, field

import jax
import jax.scipy.linalg as sla
from jax import Array
from jax import numpy as jnp

from .parameters import params
from .sharding import register_dataclass_pytree, sharding

# ── Dense LU solver ───────────────────────────────────────────────


@jax.jit
def _lu_solve(lu_pivots: tuple[Array, Array], b: Array) -> Array:
    """Batched LU solve across 2D `$(k_z, k_x)$` Fourier modes."""
    lu, piv = lu_pivots
    dtype = jnp.result_type(lu, b)
    lu = lu.astype(dtype)

    def solve_single(lu_piv, vec):
        return sla.lu_solve(lu_piv, vec)

    return jax.vmap(jax.vmap(solve_single))((lu, piv), b)


@register_dataclass_pytree
@dataclass
class DenseJAXSolver:
    """Batched dense LU cache for per-mode operators.

    On construction, the input matrix is LU-factored over all
    `$(k_z, k_x)$` modes via cuSOLVER batched LU, then
    discarded.  Only the factors are retained.
    """

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
        supported transparently by an extra ``vmap`` that leaves
        the cached LU factors untouched; this lets
        ``_imm_iteration`` do one stack-and-solve instead of
        three sequential kernel calls.

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(Nkz, Nkx, Ny)`` or
            ``(C, Nkz, Nkx, Ny)`` for a leading batch axis
            ``C``.

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


# ── SPIKE banded solver ──────────────────────────────────────────


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
    r"""Solve `$A x = b$` via the SPIKE algorithm, single 3D RHS.

    The banded operator was partitioned at construction into ``P``
    block-rows of size ``m = N_y / P`` with bandwidth ``p``.  The
    spike matrices `$V_i = A_i^{-1} B_i$` and
    `$W_i = A_i^{-1} C_i$` capture the off-block coupling, and a
    small reduced system of size ``2 P p`` resolves the spike
    weights at block boundaries.

    Stages:

    1. Local solve `$A_i g_i = f_i$` (per-block dense LU solve,
       parallel across blocks).
    2. Build the reduced RHS from the top-`p` and bottom-`p`
       slices of each ``g_i``.
    3. Reduced solve for the spike weights
       `$\\alpha = (\\alpha^T_i, \\alpha^B_i)`.
    4. Reconstruct
       `$x_i = g_i - V_i \\alpha^T_{i+1}
       - W_i \\alpha^B_{i-1}$`,
       with neighbour weights zero at the matrix endpoints.

    Parameters
    ----------
    lu, piv:
        Per-block dense LU factors and pivots,
        ``(N_{kz}, N_{kx}, P, m, m)`` and
        ``(N_{kz}, N_{kx}, P, m)``.
    V, W:
        Spike matrices ``(N_{kz}, N_{kx}, P, m, p)``.
    red_lu, red_piv:
        Dense LU of the `$2 P p \\times 2 P p$` reduced system
        per ``(kz, kx)`` mode.
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

    The original `$(N_y, N_y)$` banded operator (bandwidth ``p``)
    is partitioned at construction into ``P`` contiguous block-rows
    of size ``m = N_y / P`` (with ``m >= 2 p``) and factored
    locally via :func:`jax.scipy.linalg.lu_factor` (cuSOLVER-batched
    dense LU on the small `$(m, m)$` blocks).  Off-block coupling
    is captured by spike matrices
    `$V_i = A_i^{-1} B_i$` and `$W_i = A_i^{-1} C_i$`
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
        Per-block pivot indices, shape
        ``(N_{kz}, N_{kx}, P, m)``.
    V:
        Right-spike matrix, shape
        ``(N_{kz}, N_{kx}, P, m, p)``.
    W:
        Left-spike matrix, shape
        ``(N_{kz}, N_{kx}, P, m, p)``.
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
        supported transparently by an extra ``vmap`` that leaves
        the cached factors untouched, so the same ``lu`` / ``V``
        / ``W`` / reduced LU are reused across all batched RHSs.

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(N_{kz}, N_{kx}, N_y)`` or
            ``(C, N_{kz}, N_{kx}, N_y)`` for a leading batch
            axis ``C``.  May be real or complex; the dtype is
            preserved.

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


# ── SPIKE block partitioning and factorisation ───────────────────


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


def validate_spike_partition(
    N: int, p: int, label: str = "N"
) -> tuple[int, int]:
    r"""Validate and resolve the SPIKE block partition.

    Checks ``params.solver.spike_block_size``; if valid,
    returns ``(N // sbs, sbs)``.  If invalid or ``None``,
    falls back to :func:`_choose_block_partition` and prints
    the chosen partition.

    Parameters
    ----------
    N:
        Wall-normal grid size (``Ny`` or ``Nr``).
    p:
        FD accuracy order (half-bandwidth).
    label:
        Name used in diagnostic messages (e.g. ``"Ny"``).

    Returns
    -------
    P_blk:
        Number of SPIKE blocks.
    m_blk:
        Block size.
    """
    sbs = params.solver.spike_block_size
    if sbs is not None:
        if N % sbs != 0 or sbs < 2 * p:
            sharding.print(
                f"spike_block_size={sbs} invalid for "
                f"{label}={N}, p={p}; falling back to auto."
            )
            P_blk, m_blk = _choose_block_partition(N, p)
        else:
            P_blk, m_blk = N // sbs, sbs
    else:
        P_blk, m_blk = _choose_block_partition(N, p)
    sharding.print(
        f"SPIKE partition: P={P_blk}, m={m_blk} ({label}={N}, p={p})"
    )
    return P_blk, m_blk


def _extract_banded_corners(
    mat: Array, P: int, m: int, p: int, scale: float = 1.0
) -> tuple[Array, Array]:
    r"""Extract off-diagonal coupling corners from a banded matrix.

    Returns the right-coupling ``B`` corners and left-coupling
    ``C`` corners needed by the SPIKE block decomposition.
    ``B[i]`` contains the `$(p, p)$` sub-block coupling block
    ``i`` to block ``i+1`` (zero for the last block); ``C[i]``
    couples block ``i`` to block ``i-1`` (zero for the first).

    Parameters
    ----------
    mat:
        The banded matrix, shape ``(N_y, N_y)``.
    P:
        Number of SPIKE blocks.
    m:
        Block size (``N_y // P``).
    p:
        Half-bandwidth (coupling-corner size).
    scale:
        Multiplicative factor applied to each extracted corner
        (default 1.0).

    Returns
    -------
    B_corner:
        Right-coupling corners, shape ``(P, p, p)``.
    C_corner:
        Left-coupling corners, shape ``(P, p, p)``.
    """
    dtype = mat.dtype
    zero = jnp.zeros((p, p), dtype=dtype)

    B_list: list[Array] = []
    C_list: list[Array] = []
    for i in range(P):
        if i < P - 1:
            r0 = (i + 1) * m - p
            c0 = (i + 1) * m
            B_list.append(scale * mat[r0 : r0 + p, c0 : c0 + p])
        else:
            B_list.append(zero)

        if i > 0:
            r0 = i * m
            c0 = i * m - p
            C_list.append(scale * mat[r0 : r0 + p, c0 : c0 + p])
        else:
            C_list.append(zero)

    return jnp.stack(B_list), jnp.stack(C_list)


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
    `$W_i = A_i^{-1} C_i$`, and reduced-system LU -- all
    on the GPU with no `$(N_y, N_y)$` array.

    Array sharding is handled eagerly via
    ``out_sharding`` on the allocating calls before
    the JIT'd compute kernels.

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
