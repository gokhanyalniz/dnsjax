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

Complex right-hand sides
------------------------
All operators here are **real**; only the RHS may be complex.  To
avoid promoting the (large) factors to complex on every solve --
which `jax.scipy.linalg.lu_solve` would do, tripling the factor
memory traffic and doubling the triangular-solve FLOPs -- a complex
RHS is split into a real array with a trailing re/im axis of
length 2 (`$\\ldots, N_y$` complex `$\\to \\ldots, N_y, 2$` real)
and solved as two real RHS columns, then recombined.  The split
and merge are single fused elementwise passes over the RHS, far
cheaper than the factor-sized conversion they replace.  The
permutations are precomputed at factorisation time so the solve
path (:func:`_permuted_tri_solve`: permutation gather + two
batched :func:`jax.lax.linalg.triangular_solve` calls) needs no
per-call pivot conversion.

Utility helpers :func:`_extract_banded_corners` and
:func:`_choose_block_partition` support the block decomposition and
optimal partitioning.
"""

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.scipy.linalg as sla
from jax import Array, lax
from jax import numpy as jnp
from jax.lax import linalg as lax_linalg

from .parameters import params
from .sharding import register_dataclass_pytree, sharding


def _real_rhs_view(rhs: Array) -> Array:
    r"""Split a complex array into real with a trailing re/im axis.

    A complex array of shape ``(..., N)`` becomes a real array of
    shape ``(..., N, 2)``.  One fused elementwise pass (XLA has
    no zero-copy complex bitcast); the inverse is
    :func:`_complex_from_view`.
    """
    return jnp.stack([rhs.real, rhs.imag], axis=-1)


def _complex_from_view(x: Array) -> Array:
    """Recombine the trailing re/im axis of length 2 into a
    complex array (inverse of :func:`_real_rhs_view`)."""
    return lax.complex(x[..., 0], x[..., 1])


def _permuted_tri_solve(lu: Array, perm: Array, b: Array) -> Array:
    """LU solve from factors and precomputed permutations.

    Row-permutes *b*, then runs two batched triangular solves.
    Equivalent to ``lu_solve`` but takes permutations instead of
    pivots, skipping the per-call pivot conversion.

    Parameters
    ----------
    lu:
        LU factors, ``(..., m, m)``.
    perm:
        Permutation indices, ``(..., m)``.
    b:
        Right-hand side with a trailing column axis,
        ``(..., m, k)``.  Leading batch dims must match *lu*.
    """
    x = jnp.take_along_axis(b, perm[..., None], axis=-2)
    x = lax_linalg.triangular_solve(
        lu, x, left_side=True, lower=True, unit_diagonal=True
    )
    return lax_linalg.triangular_solve(lu, x, left_side=True, lower=False)


# ── Dense LU solver ───────────────────────────────────────────────


@jax.jit
def _lu_solve(lu_perm: tuple[Array, Array], b: Array) -> Array:
    """Batched LU solve across 2D `$(k_z, k_x)$` Fourier modes.

    The factors are real; a complex *b* is solved as two real
    RHS columns via a re/im split (see module docstring), so no
    complex copy of the factors is ever made.
    """
    lu, perm = lu_perm
    if jnp.iscomplexobj(b):
        x = _permuted_tri_solve(lu, perm, _real_rhs_view(b))
        return _complex_from_view(x)
    return _permuted_tri_solve(lu, perm, b[..., None])[..., 0]


@register_dataclass_pytree
@dataclass
class DenseJAXSolver:
    """Batched dense LU cache for per-mode operators.

    On construction, the input matrix is LU-factored over all
    `$(k_z, k_x)$` modes via cuSOLVER batched LU (the input
    buffer is donated, so the factors reuse its memory), then
    discarded.  Pivots are converted to permutations once so the
    solve path needs no per-call pivot conversion.

    Two storage layouts are supported:

    - **Standard** (``lu.ndim == 4``): one operator shared across
      all RHS components, shape ``(Nkz, Nkx, Ny, Ny)``.
    - **Batched operators** (``lu.ndim == 5``): ``C`` distinct
      operators stacked along a leading axis, shape
      ``(C, Nkz, Nkx, Ny, Ny)``.  Each component of a
      ``(C, Nkz, Nkx, Ny)`` RHS is solved with its own operator.
      Use :meth:`from_factors` to construct from pre-factored
      arrays.
    """

    matrix: Array
    lu: Array = field(init=False)
    perm: Array = field(init=False)

    def __post_init__(self) -> None:
        """Batch LU-factor over all ``(kz, kx)`` modes."""

        @partial(jax.jit, donate_argnums=0)
        def batched_lu_factor(A: Array) -> tuple[Array, Array]:
            lu, piv = jax.vmap(jax.vmap(sla.lu_factor))(A)
            perm = lax_linalg.lu_pivots_to_permutation(piv, A.shape[-1])
            return lu, perm

        self.lu, self.perm = batched_lu_factor(self.matrix)
        self.matrix = None

    @classmethod
    def from_factors(cls, lu: Array, perm: Array) -> DenseJAXSolver:
        """Construct from pre-factored LU arrays.

        Bypasses the ``__post_init__`` factorisation, useful for
        building a combined solver from individually factored
        operators (e.g. stacking cylindrical
        ``Hk_plus``, ``Hk_minus``, ``Hk_z``).

        Parameters
        ----------
        lu:
            LU factors, shape ``(Nkz, Nkx, Ny, Ny)`` or
            ``(C, Nkz, Nkx, Ny, Ny)`` for batched operators.
        perm:
            Permutation indices matching *lu*.
        """
        obj = object.__new__(cls)
        obj.matrix = None
        obj.lu = lu
        obj.perm = perm
        return obj

    def solve(self, rhs: Array) -> Array:
        """Batched LU solve.

        Dispatch:

        - ``lu.ndim == 5``: batched operators — ``vmap`` over
          both the operator and RHS leading axis.
        - ``rhs.ndim == 4``: shared operator — ``vmap`` over
          the RHS leading axis only.
        - Otherwise: single operator, single RHS.

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(Nkz, Nkx, Ny)`` or
            ``(C, Nkz, Nkx, Ny)`` for a leading batch axis
            ``C``.  May be real or complex; complex RHS are
            solved in real arithmetic via a re/im split.

        Returns
        -------
        :
            Solution array, same shape and dtype as *rhs*.
        """
        if self.lu.ndim == 5:
            return jax.vmap(_lu_solve)((self.lu, self.perm), rhs)
        if rhs.ndim == 4:
            return jax.vmap(_lu_solve, in_axes=(None, 0))(
                (self.lu, self.perm), rhs
            )
        return _lu_solve((self.lu, self.perm), rhs)


# ── SPIKE banded solver ──────────────────────────────────────────


@jax.jit
def _spike_solve(
    lu: Array,
    perm: Array,
    V: Array,
    W: Array,
    red_lu: Array,
    red_perm: Array,
    rhs: Array,
) -> Array:
    r"""Solve `$A x = b$` via the SPIKE algorithm, single 3D RHS.

    The banded operator was partitioned at construction into ``P``
    block-rows of size ``m = N_y / P`` with bandwidth ``p``.  The
    spike matrices `$V_i = A_i^{-1} B_i$` and
    `$W_i = A_i^{-1} C_i$` capture the off-block coupling, and a
    small reduced system of size ``2 P p`` resolves the spike
    weights at block boundaries.

    A complex *rhs* is split into a real ``(..., N_y, 2)`` view
    and carried through every stage as a trailing RHS-column axis
    ``k``, so all solves and GEMMs run in real arithmetic against
    the real factors (see module docstring).  A real *rhs* takes
    the same path with ``k = 1``.

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
    lu, perm:
        Per-block dense LU factors and permutations,
        ``(N_{kz}, N_{kx}, P, m, m)`` and
        ``(N_{kz}, N_{kx}, P, m)``.
    V, W:
        Spike matrices ``(N_{kz}, N_{kx}, P, m, p)``.
    red_lu, red_perm:
        Dense LU of the `$2 P p \\times 2 P p$` reduced system
        per ``(kz, kx)`` mode, with its permutations.
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

    is_complex = jnp.iscomplexobj(rhs)
    b = _real_rhs_view(rhs) if is_complex else rhs[..., None]
    k = b.shape[-1]

    # Stage 1: local solve A_i g_i = f_i in parallel across blocks.
    b_blocks = b.reshape(b.shape[:-2] + (P, m, k))
    g = _permuted_tri_solve(lu, perm, b_blocks)

    # Stage 2: reduced RHS from g top/bottom slices.
    g_top = g[..., :p, :]
    g_bot = g[..., m - p :, :]
    b_red_blocks = jnp.stack([g_top, g_bot], axis=-3)
    b_red = b_red_blocks.reshape(b_red_blocks.shape[:-4] + (2 * P * p, k))

    # Stage 3: reduced solve.
    alpha = _permuted_tri_solve(red_lu, red_perm, b_red)

    # Stage 4: extract per-block alpha^T / alpha^B, then shift.
    alpha_blocks = alpha.reshape(alpha.shape[:-2] + (P, 2, p, k))
    alpha_T = alpha_blocks[..., 0, :, :]
    alpha_B = alpha_blocks[..., 1, :, :]
    zeros_p = jnp.zeros_like(alpha_T[..., :1, :, :])
    alpha_T_next = jnp.concatenate([alpha_T[..., 1:, :, :], zeros_p], axis=-3)
    alpha_B_prev = jnp.concatenate([zeros_p, alpha_B[..., :-1, :, :]], axis=-3)

    # Stage 5: x_i = g_i - V_i alpha^T(i+1) - W_i alpha^B(i-1).
    V_contrib = jnp.einsum("...irc,...ick->...irk", V, alpha_T_next)
    W_contrib = jnp.einsum("...irc,...ick->...irk", W, alpha_B_prev)
    x_blocks = g - V_contrib - W_contrib

    x = x_blocks.reshape(x_blocks.shape[:-3] + (Ny, k))
    if is_complex:
        return _complex_from_view(x)
    return x[..., 0]


@jax.jit
def _spike_solve_bt(
    lu: Array,
    perm: Array,
    V: Array,
    W: Array,
    red_diag_lu: Array,
    red_diag_perm: Array,
    red_mod_super: Array,
    red_sub: Array,
    rhs: Array,
) -> Array:
    r"""SPIKE solve with block-Thomas reduced system.

    Identical to :func:`_spike_solve` except the reduced
    solve (Stage 3) uses a pre-factored block-tridiagonal
    system via :func:`_block_thomas_solve` instead of a
    dense ``lu_solve``.  Complex RHS handling is the same
    (real re/im split with a trailing column axis ``k``).

    Parameters
    ----------
    lu, perm:
        Per-block dense LU factors and permutations.
    V, W:
        Spike matrices.
    red_diag_lu, red_diag_perm:
        Block-Thomas diagonal LU factors and permutations,
        ``(N_{kz}, N_{kx}, P, 2p, 2p)`` and
        ``(N_{kz}, N_{kx}, P, 2p)``.
    red_mod_super:
        Modified super-diagonal blocks,
        ``(N_{kz}, N_{kx}, P-1, 2p, 2p)``.
    red_sub:
        Original sub-diagonal blocks,
        ``(N_{kz}, N_{kx}, P-1, 2p, 2p)``.
    rhs:
        Right-hand side, shape ``(N_{kz}, N_{kx}, N_y)``.

    Returns
    -------
    :
        Solution, same shape and dtype as *rhs*.
    """
    P, m = lu.shape[-3], lu.shape[-2]
    p = V.shape[-1]
    Ny = P * m

    is_complex = jnp.iscomplexobj(rhs)
    b = _real_rhs_view(rhs) if is_complex else rhs[..., None]
    k = b.shape[-1]

    # Stage 1: local solve (identical to _spike_solve).
    b_blocks = b.reshape(b.shape[:-2] + (P, m, k))
    g = _permuted_tri_solve(lu, perm, b_blocks)

    # Stage 2: reduced RHS.
    g_top = g[..., :p, :]
    g_bot = g[..., m - p :, :]
    b_red = jnp.concatenate([g_top, g_bot], axis=-2)

    # Stage 3: block-Thomas reduced solve.
    alpha_blocks = _block_thomas_solve(
        red_diag_lu, red_diag_perm, red_mod_super, red_sub, b_red
    )

    # Stage 4: extract and shift.
    alpha_T = alpha_blocks[..., :p, :]
    alpha_B = alpha_blocks[..., p:, :]
    zeros_p = jnp.zeros_like(alpha_T[..., :1, :, :])
    alpha_T_next = jnp.concatenate([alpha_T[..., 1:, :, :], zeros_p], axis=-3)
    alpha_B_prev = jnp.concatenate([zeros_p, alpha_B[..., :-1, :, :]], axis=-3)

    # Stage 5: reconstruction.
    V_contrib = jnp.einsum("...irc,...ick->...irk", V, alpha_T_next)
    W_contrib = jnp.einsum("...irc,...ick->...irk", W, alpha_B_prev)
    x_blocks = g - V_contrib - W_contrib

    x = x_blocks.reshape(x_blocks.shape[:-3] + (Ny, k))
    if is_complex:
        return _complex_from_view(x)
    return x[..., 0]


@register_dataclass_pytree
@dataclass
class PerModeBandedOperator:
    r"""SPIKE-factored banded operator (band-preserving, GPU-fast).

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

    Two reduced-system storage layouts are available:

    - **Dense** (``red_lu is not None``): full
      `$2Pp \times 2Pp$` LU (the default: one batched
      solve, no sequential scan rounds).
    - **Block-Thomas** (``red_bt_diag_lu is not None``):
      block-tridiagonal factorisation storing `$P$` blocks
      of `$(2p, 2p)$` LU plus `$P{-}1$` modified
      super-diagonal and sub-diagonal blocks.
      Memory scales as `$O(P p^2)$` instead of
      `$O(P^2 p^2)$`, at the cost of `$2(P-1)$`
      sequential ``lax.scan`` steps per solve.

    Two operator storage layouts are supported:

    - **Standard** (``lu.ndim == 5``): one operator shared
      across all RHS components.
    - **Batched** (``lu.ndim == 6``): ``C`` distinct
      operators along a leading axis.

    Attributes
    ----------
    lu:
        Per-block dense LU factors, shape
        ``(N_{kz}, N_{kx}, P, m, m)`` or
        ``(C, N_{kz}, N_{kx}, P, m, m)``.
    perm:
        Per-block permutation indices (precomputed from the
        LU pivots, so the solve path skips the per-call
        pivot conversion).
    V, W:
        Spike matrices, ``(N_{kz}, N_{kx}, P, m, p)``.
    red_lu:
        Dense LU of the reduced system (``None`` when
        block-Thomas is active).
    red_perm:
        Permutations for the dense reduced LU.
    red_bt_diag_lu:
        Block-Thomas diagonal LU factors,
        ``(..., P, 2p, 2p)`` (``None`` when dense is
        active).
    red_bt_diag_perm:
        Block-Thomas diagonal permutations.
    red_bt_mod_super:
        Modified super-diagonal blocks
        `$D_i^{-1} U_i$`, ``(..., P-1, 2p, 2p)``.
    red_bt_sub:
        Original sub-diagonal blocks,
        ``(..., P-1, 2p, 2p)``.
    """

    lu: Array
    perm: Array
    V: Array
    W: Array
    red_lu: Array | None = None
    red_perm: Array | None = None
    red_bt_diag_lu: Array | None = None
    red_bt_diag_perm: Array | None = None
    red_bt_mod_super: Array | None = None
    red_bt_sub: Array | None = None

    @property
    def _use_bt(self) -> bool:
        return self.red_bt_diag_lu is not None

    def solve(self, rhs: Array) -> Array:
        """Batched SPIKE solve across ``(kz, kx)`` modes.

        Dispatch:

        - ``lu.ndim == 6``: batched operators — ``vmap`` over
          both the operator and RHS leading axis.
        - ``rhs.ndim == 4``: shared operator — ``vmap`` over
          the RHS leading axis only.
        - Otherwise: single operator, single RHS.

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(N_{kz}, N_{kx}, N_y)`` or
            ``(C, N_{kz}, N_{kx}, N_y)`` for a leading batch
            axis ``C``.  May be real or complex; the dtype is
            preserved (complex RHS are solved in real
            arithmetic via a re/im split).

        Returns
        -------
        :
            Solution array, same shape and dtype as *rhs*.
        """
        solve_fn = _spike_solve_bt if self._use_bt else _spike_solve

        if self._use_bt:
            args = (
                self.lu,
                self.perm,
                self.V,
                self.W,
                self.red_bt_diag_lu,
                self.red_bt_diag_perm,
                self.red_bt_mod_super,
                self.red_bt_sub,
                rhs,
            )
            none_axes = (None,) * 8
        else:
            args = (
                self.lu,
                self.perm,
                self.V,
                self.W,
                self.red_lu,
                self.red_perm,
                rhs,
            )
            none_axes = (None,) * 6

        if self.lu.ndim == 6:
            return jax.vmap(solve_fn)(*args)
        if rhs.ndim == 4:
            return jax.vmap(
                solve_fn,
                in_axes=(*none_axes, 0),
            )(*args)
        return solve_fn(*args)


# ── SPIKE block partitioning and factorisation ───────────────────


def _spike_memory_per_mode(
    Ny: int, P: int, p: int, block_thomas: bool = False
) -> float:
    r"""Per-mode SPIKE storage estimate.

    Block LU factors dominate at `$N_y^2 / P$`.  The reduced-system
    cost is `$(3P{-}2) \cdot 4 p^2$` for block-Thomas or
    `$4 P^2 p^2$` for the dense reduced path.
    """
    block_cost = Ny * Ny / P
    if block_thomas:
        return block_cost + (3 * P - 2) * 4 * p * p
    return block_cost + 4 * P * P * p * p


def _choose_block_partition(
    Ny: int, p: int, block_thomas: bool = False
) -> tuple[int, int]:
    r"""Choose SPIKE block count `$P$` and block size `$m$`.

    Picks the divisor `$P \ge 2$` of `$N_y$` (with
    `$m = N_y / P \ge 2 p$`) that minimises total per-mode
    SPIKE storage.  When *block_thomas* is active the
    reduced-system cost is `$O(P p^2)$` instead of
    `$O(P^2 p^2)$`, favouring more (smaller) blocks.
    Falls back to `$P = 1$` when `$N_y$` is prime or too
    small.

    Parameters
    ----------
    Ny:
        Wall-normal grid size.
    p:
        FD order (half-bandwidth of the banded operator).
    block_thomas:
        Use the block-Thomas reduced-system cost model.

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
            cost = _spike_memory_per_mode(Ny, P_cand, p, block_thomas)
            if cost < best_cost:
                best_P, best_cost = P_cand, cost

    if best_P == 1:
        return 1, Ny
    return best_P, Ny // best_P


def validate_spike_partition(
    N: int, p: int, label: str = "N", block_thomas: bool = False
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
    block_thomas:
        Use the block-Thomas cost model when choosing
        the partition automatically.

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
            P_blk, m_blk = _choose_block_partition(N, p, block_thomas)
        else:
            P_blk, m_blk = N // sbs, sbs
    else:
        P_blk, m_blk = _choose_block_partition(N, p, block_thomas)
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


# ── Block-Thomas solver for block-tridiagonal systems ────────


def _build_reduced_blocks(
    V: Array, W: Array, p: int
) -> tuple[Array, Array, Array]:
    r"""Extract the block-tridiagonal structure of the reduced
    system as separate ``(2p, 2p)`` blocks.

    The reduced system (size `$2Pp$`) is block-tridiagonal
    when viewed as `$P$` blocks of `$2p \times 2p$`:

    - Diagonal blocks `$D_i = I_{2p}$`.
    - Super-diagonal `$U_i$`: coupling from block `$i$` to
      `$i{+}1$` (from spike `$V$` tips).
    - Sub-diagonal `$L_i$`: coupling from block `$i$` to
      `$i{-}1$` (from spike `$W$` tips).

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
    diag:
        Diagonal blocks, ``(..., P, 2p, 2p)``.
    super_diag:
        Super-diagonal blocks, ``(..., P-1, 2p, 2p)``.
    sub_diag:
        Sub-diagonal blocks, ``(..., P-1, 2p, 2p)``.
    """
    P = V.shape[-3]
    m = V.shape[-2]
    bp = 2 * p
    dtype = V.dtype
    batch = V.shape[:-3]

    V_top = V[..., :p, :]
    V_bot = V[..., m - p :, :]
    W_top = W[..., :p, :]
    W_bot = W[..., m - p :, :]

    ones = jnp.ones_like(V[..., 0, 0, 0])
    eye_bp = jnp.eye(bp, dtype=dtype)
    diag = ones[..., None, None, None] * eye_bp
    diag = jnp.broadcast_to(diag, batch + (P, bp, bp))
    diag = jnp.array(diag)

    if P > 1:
        # Build super/sub as (2p, 2p) blocks via .at[] indexing,
        # inheriting sharding from V/W.
        sup_shape = batch + (P - 1, bp, bp)
        sup = jnp.zeros_like(V[..., :1, :1, :1]).squeeze(axis=(-3, -2, -1))[
            ..., None, None, None
        ] * jnp.zeros((1, bp, bp), dtype=dtype)
        sup = jnp.broadcast_to(sup, sup_shape)
        sup = jnp.array(sup)

        # Super: [[V_top[i], 0], [V_bot[i], 0]] for i=0..P-2.
        sup = sup.at[..., :p, :p].set(V_top[..., : P - 1, :, :])
        sup = sup.at[..., p:, :p].set(V_bot[..., : P - 1, :, :])

        sub = jnp.zeros_like(sup)
        # Sub: [[0, W_top[i]], [0, W_bot[i]]] for i=1..P-1.
        sub = sub.at[..., :p, p:].set(W_top[..., 1:, :, :])
        sub = sub.at[..., p:, p:].set(W_bot[..., 1:, :, :])
    else:
        sub = jnp.zeros(batch + (0, bp, bp), dtype=dtype)
        sup = jnp.zeros(batch + (0, bp, bp), dtype=dtype)

    return diag, sup, sub


def _block_thomas_factor(
    diag: Array,
    super_diag: Array,
    sub_diag: Array,
) -> tuple[Array, Array, Array, Array]:
    r"""Block-Thomas LU factorisation of a block-tridiagonal
    system via ``lax.scan``.

    Forward elimination modifies each diagonal block and
    computes the modified super-diagonal (``D_i^{-1} U_i``).

    Parameters
    ----------
    diag:
        Diagonal blocks, ``(..., P, b, b)``.
    super_diag:
        Super-diagonal blocks, ``(..., P-1, b, b)``.
    sub_diag:
        Sub-diagonal blocks, ``(..., P-1, b, b)``.

    Returns
    -------
    diag_lu:
        LU factors of modified diagonal blocks,
        ``(..., P, b, b)``.
    diag_piv:
        Pivots for the diagonal LU, ``(..., P, b)``.
    mod_super:
        Modified super-diagonal blocks
        ``D_i^{-1} U_i``, ``(..., P-1, b, b)``.
    sub_diag:
        Original sub-diagonal blocks (passed through
        unchanged for the solve phase).
    """
    P = diag.shape[-3]

    if P == 1:
        lu, piv = sla.lu_factor(diag[..., 0, :, :])
        return (
            lu[..., None, :, :],
            piv[..., None, :],
            super_diag,
            sub_diag,
        )

    # Initialise with block 0.
    lu0, piv0 = sla.lu_factor(diag[..., 0, :, :])

    scan_diag = jnp.moveaxis(diag[..., 1:, :, :], -3, 0)
    scan_sub = jnp.moveaxis(sub_diag, -3, 0)
    # The scan output stores the PREVIOUS block's modified super.

    def fwd_step_v2(carry, xs):
        prev_lu, prev_piv, cur_super = carry
        d_next, sub_i, next_super = xs
        mod_s = sla.lu_solve((prev_lu, prev_piv), cur_super)
        d_next = d_next - sub_i @ mod_s
        lu_next, piv_next = sla.lu_factor(d_next)
        return (lu_next, piv_next, next_super), (
            prev_lu,
            prev_piv,
            mod_s,
        )

    # Pad super_diag with a dummy for the last step's
    # "next_super" (never used; the last block has no super).
    dummy_super = jnp.zeros_like(super_diag[..., :1, :, :])
    super_padded = jnp.concatenate(
        [super_diag[..., 1:, :, :], dummy_super], axis=-3
    )
    scan_super = jnp.moveaxis(super_padded, -3, 0)

    init_carry = (lu0, piv0, super_diag[..., 0, :, :])
    final_carry, scan_out = lax.scan(
        fwd_step_v2, init_carry, (scan_diag, scan_sub, scan_super)
    )
    prev_lus, prev_pivs, mod_supers = scan_out

    # Assemble: block 0 outputs are in init, blocks 1..P-2
    # in scan_out, block P-1 in final_carry.
    last_lu, last_piv, _ = final_carry

    # Compute the last modified super (from block P-2 to P-1).
    # It was computed inside the last scan step and is the last
    # element of mod_supers. But we also need the very last
    # block's modified super = lu_solve of final carry's super.
    # Actually the last scan step (processing block P-1) outputs
    # mod_supers[-1] which is D_{P-2}^{-1} U_{P-2}. The super
    # for block P-1 doesn't exist (last block). So mod_supers
    # has P-1 entries: the modified supers for blocks 0..P-2.

    all_lu = jnp.concatenate(
        [
            prev_lus[0][..., None, :, :],
            prev_lus[1:],
            last_lu[..., None, :, :],
        ],
        axis=-3,
    )
    all_piv = jnp.concatenate(
        [
            prev_pivs[0][..., None, :],
            prev_pivs[1:],
            last_piv[..., None, :],
        ],
        axis=-2,
    )

    # mod_supers from scan: these are blocks 0..P-2.
    mod_super = jnp.moveaxis(mod_supers, 0, -3)

    return all_lu, all_piv, mod_super, sub_diag


def _block_thomas_solve(
    diag_lu: Array,
    diag_perm: Array,
    mod_super: Array,
    sub_diag: Array,
    rhs: Array,
) -> Array:
    r"""Block-Thomas solve for a pre-factored block-tridiagonal
    system via ``lax.scan``.

    The right-hand side carries a trailing column axis ``k``
    (``k = 2`` for the real view of a complex RHS, ``k = 1``
    for a real RHS), so all block solves and GEMMs stay in
    real arithmetic.

    Parameters
    ----------
    diag_lu:
        LU of modified diagonal blocks, ``(..., P, b, b)``.
    diag_perm:
        Permutations, ``(..., P, b)``.
    mod_super:
        Modified super-diagonal ``D_i^{-1} U_i``,
        ``(..., P-1, b, b)``.
    sub_diag:
        Original sub-diagonal blocks, ``(..., P-1, b, b)``.
    rhs:
        Right-hand side, ``(..., P, b, k)``.

    Returns
    -------
    :
        Solution, ``(..., P, b, k)``.
    """
    P = diag_lu.shape[-3]

    if P == 1:
        x = _permuted_tri_solve(
            diag_lu[..., 0, :, :],
            diag_perm[..., 0, :],
            rhs[..., 0, :, :],
        )
        return x[..., None, :, :]

    # Forward substitution: y_i = D_i^{-1} (b_i - L_i y_{i-1})
    def fwd_sub(y_prev, xs):
        lu_i, perm_i, sub_i, b_i = xs
        r_i = b_i - jnp.einsum("...ij,...jk->...ik", sub_i, y_prev)
        y_i = _permuted_tri_solve(lu_i, perm_i, r_i)
        return y_i, y_i

    y0 = _permuted_tri_solve(
        diag_lu[..., 0, :, :],
        diag_perm[..., 0, :],
        rhs[..., 0, :, :],
    )

    scan_lu = jnp.moveaxis(diag_lu[..., 1:, :, :], -3, 0)
    scan_perm = jnp.moveaxis(diag_perm[..., 1:, :], -2, 0)
    scan_sub = jnp.moveaxis(sub_diag, -3, 0)
    scan_rhs = jnp.moveaxis(rhs[..., 1:, :, :], -3, 0)

    _, ys_rest = lax.scan(
        fwd_sub, y0, (scan_lu, scan_perm, scan_sub, scan_rhs)
    )
    y_all = jnp.concatenate(
        [y0[..., None, :, :], jnp.moveaxis(ys_rest, 0, -3)], axis=-3
    )

    # Backward substitution: x_i = y_i - U_i' x_{i+1}
    def bwd_sub(x_next, xs):
        y_i, u_i = xs
        x_i = y_i - jnp.einsum("...ij,...jk->...ik", u_i, x_next)
        return x_i, x_i

    x_last = y_all[..., -1, :, :]
    # Reverse: process blocks P-2 down to 0.
    scan_y_rev = jnp.moveaxis(y_all[..., :-1, :, :], -3, 0)
    scan_y_rev = scan_y_rev[::-1]
    scan_u_rev = jnp.moveaxis(mod_super, -3, 0)[::-1]

    _, xs_rev = lax.scan(bwd_sub, x_last, (scan_y_rev, scan_u_rev))
    xs_fwd = jnp.moveaxis(xs_rev[::-1], 0, -3)
    x_all = jnp.concatenate([xs_fwd, x_last[..., None, :, :]], axis=-3)

    return x_all


def _spike_factor(
    A_blocks: Array,
    B_corner: Array,
    C_corner: Array,
    block_thomas: bool = False,
) -> PerModeBandedOperator:
    r"""SPIKE factorisation of a block-partitioned banded operator.

    Performs per-block dense LU (cuSOLVER batched), spike
    matrix solves `$V_i = A_i^{-1} B_i$`,
    `$W_i = A_i^{-1} C_i$`, and reduced-system factorisation
    -- all on the GPU with no `$(N_y, N_y)$` array.  The input
    block arrays are donated (their buffers are reused for the
    factors), so callers must not use them afterwards.

    When *block_thomas* is ``True``, the reduced system is
    stored and solved in block-tridiagonal form
    (`$O(P p^2)$` memory) via :func:`_block_thomas_factor`.
    When ``False`` (default), the dense
    `$2Pp \times 2Pp$` LU is used.

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
    block_thomas:
        Use block-Thomas reduced-system factorisation.

    Returns
    -------
    :
        A :class:`PerModeBandedOperator` with either dense
        or block-Thomas reduced-system factors.
    """
    m = A_blocks.shape[-2]
    p = B_corner.shape[-1]
    dtype = A_blocks.dtype

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

    if block_thomas:

        @partial(jax.jit, donate_argnums=(0, 1, 2))
        def _do_factor_bt(A, B, C):
            lu, piv = jax.vmap(jax.vmap(jax.vmap(sla.lu_factor)))(A)
            V = jax.vmap(jax.vmap(jax.vmap(sla.lu_solve)))((lu, piv), B)
            W = jax.vmap(jax.vmap(jax.vmap(sla.lu_solve)))((lu, piv), C)
            perm = lax_linalg.lu_pivots_to_permutation(piv, A.shape[-1])
            diag, sup, sub = _build_reduced_blocks(V, W, p)
            bt_lu, bt_piv, bt_sup, bt_sub = jax.vmap(
                jax.vmap(_block_thomas_factor)
            )(diag, sup, sub)
            bt_perm = lax_linalg.lu_pivots_to_permutation(
                bt_piv, bt_lu.shape[-1]
            )
            return lu, perm, V, W, bt_lu, bt_perm, bt_sup, bt_sub

        (
            lu,
            perm,
            V,
            W,
            bt_lu,
            bt_perm,
            bt_sup,
            bt_sub,
        ) = _do_factor_bt(A_blocks, B_full, C_full)
        return PerModeBandedOperator(
            lu=lu,
            perm=perm,
            V=V,
            W=W,
            red_bt_diag_lu=bt_lu,
            red_bt_diag_perm=bt_perm,
            red_bt_mod_super=bt_sup,
            red_bt_sub=bt_sub,
        )

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def _do_factor(A, B, C):
        lu, piv = jax.vmap(jax.vmap(jax.vmap(sla.lu_factor)))(A)
        V = jax.vmap(jax.vmap(jax.vmap(sla.lu_solve)))((lu, piv), B)
        W = jax.vmap(jax.vmap(jax.vmap(sla.lu_solve)))((lu, piv), C)
        perm = lax_linalg.lu_pivots_to_permutation(piv, A.shape[-1])
        A_red = _build_reduced_matrix(V, W, p)
        red_lu, red_piv = jax.vmap(jax.vmap(sla.lu_factor))(A_red)
        red_perm = lax_linalg.lu_pivots_to_permutation(
            red_piv, A_red.shape[-1]
        )
        return lu, perm, V, W, red_lu, red_perm

    lu, perm, V, W, red_lu, red_perm = _do_factor(A_blocks, B_full, C_full)
    return PerModeBandedOperator(
        lu=lu, perm=perm, V=V, W=W, red_lu=red_lu, red_perm=red_perm
    )


def _stack_banded_operators(
    *ops: PerModeBandedOperator,
) -> PerModeBandedOperator:
    """Stack multiple :class:`PerModeBandedOperator` instances
    along a leading component axis.

    All operators must use the same reduced-system backend
    (all block-Thomas or all dense).
    """

    def _maybe_stack(arrs):
        if arrs[0] is None:
            return None
        return jnp.stack(arrs)

    return PerModeBandedOperator(
        lu=jnp.stack([o.lu for o in ops]),
        perm=jnp.stack([o.perm for o in ops]),
        V=jnp.stack([o.V for o in ops]),
        W=jnp.stack([o.W for o in ops]),
        red_lu=_maybe_stack([o.red_lu for o in ops]),
        red_perm=_maybe_stack([o.red_perm for o in ops]),
        red_bt_diag_lu=_maybe_stack([o.red_bt_diag_lu for o in ops]),
        red_bt_diag_perm=_maybe_stack([o.red_bt_diag_perm for o in ops]),
        red_bt_mod_super=_maybe_stack([o.red_bt_mod_super for o in ops]),
        red_bt_sub=_maybe_stack([o.red_bt_sub for o in ops]),
    )
