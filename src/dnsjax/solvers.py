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

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.scipy.linalg as sla
from jax import Array, lax
from jax import numpy as jnp
from jax.lax import linalg as lax_linalg
from jax.typing import DTypeLike

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

    def solve(self, rhs: Array, component_axis: int = 0) -> Array:
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
            Right-hand side, **mode-inner** shape ``(Ny, Nkz, Nkx)``
            or ``(C, Ny, Nkz, Nkx)`` for a leading batch axis ``C``
            (the velocity field's native layout).  May be real or
            complex; complex RHS are solved in real arithmetic via a
            re/im split.

        Returns
        -------
        :
            Solution array, same shape and dtype as *rhs*.

        Notes
        -----
        The dense LU is inherently mode-outer (``Ny`` on the matrix
        axes), so the mode-inner field is moved to ``(Nkz, Nkx, Ny)``
        on entry and back on exit -- a pure axis permutation, relocated
        here from the old call-site transpose so the Pallas backend can
        drop it (see :func:`_banded_mode_solve`).  *component_axis*
        selects the batched RHS axis: ``0`` (default, ``(C, Ny, ...)``)
        or ``1`` (``(Ny, C, ...)``, the IMM Hk construction's y-leading
        layout); either way one ``moveaxis`` brings ``Ny`` to the block
        axis (this backend is mode-outer regardless -- the Pallas
        backend is the one that stays transpose-free).
        """
        if component_axis == 1:  # y-leading (Ny, C, Nkz, Nkx) stacked RHS
            b = jnp.moveaxis(rhs, 0, -1)  # -> (C, Nkz, Nkx, Ny)
        else:
            b = jnp.moveaxis(rhs, -3, -1)  # mode-inner -> (.., Nkz, Nkx, Ny)
        if self.lu.ndim == 5:
            x = jax.vmap(_lu_solve)((self.lu, self.perm), b)
        elif b.ndim == 4:
            x = jax.vmap(_lu_solve, in_axes=(None, 0))((self.lu, self.perm), b)
        else:
            x = _lu_solve((self.lu, self.perm), b)
        if component_axis == 1:
            return jnp.moveaxis(x, -1, 0)  # -> (Ny, C, Nkz, Nkx)
        return jnp.moveaxis(x, -1, -3)  # -> mode-inner (.., Ny, Nkz, Nkx)


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

    def solve(self, rhs: Array, component_axis: int = 0) -> Array:
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
            Right-hand side, **mode-inner** shape ``(N_y, N_{kz},
            N_{kx})`` or ``(C, N_y, N_{kz}, N_{kx})`` for a leading
            batch axis ``C`` (the velocity field's native layout).
            May be real or complex; the dtype is preserved (complex
            RHS are solved in real arithmetic via a re/im split).

        Returns
        -------
        :
            Solution array, same shape and dtype as *rhs*.

        Notes
        -----
        SPIKE is inherently mode-outer (``N_y`` on the block axes), so
        the mode-inner field is moved to ``(N_{kz}, N_{kx}, N_y)`` on
        entry and back on exit -- a pure axis permutation, relocated
        here from the old call-site transpose so the Pallas backend can
        drop it (see :func:`_banded_mode_solve`).  *component_axis*
        selects the batched RHS axis (``0`` default ``(C, Ny, ...)`` /
        ``1`` y-leading ``(Ny, C, ...)`` for the IMM Hk construction);
        SPIKE is mode-outer regardless, so either way one ``moveaxis``
        brings ``Ny`` to the block axis.
        """
        if component_axis == 1:  # y-leading (Ny, C, Nkz, Nkx) stacked RHS
            b = jnp.moveaxis(rhs, 0, -1)  # -> (C, Nkz, Nkx, Ny)
        else:
            b = jnp.moveaxis(rhs, -3, -1)  # mode-inner -> (.., Nkz, Nkx, Ny)
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
                b,
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
                b,
            )
            none_axes = (None,) * 6

        if self.lu.ndim == 6:
            x = jax.vmap(solve_fn)(*args)
        elif b.ndim == 4:
            x = jax.vmap(
                solve_fn,
                in_axes=(*none_axes, 0),
            )(*args)
        else:
            x = solve_fn(*args)
        if component_axis == 1:
            return jnp.moveaxis(x, -1, 0)  # -> (Ny, C, Nkz, Nkx)
        return jnp.moveaxis(x, -1, -3)  # -> mode-inner (.., Ny, Nkz, Nkx)


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


# ── Pallas per-mode banded solver ────────────────────────────────
#
# The implicit step is one independent banded system per Fourier mode
# (size N_y, half-bandwidth p).  The throughput-optimal GPU strategy is
# the standard one-program-per-mode sequential banded sweep: across-mode
# parallelism fills the GPU while each program walks N_y in fast memory.
# JAX/XLA cannot express a per-lane sequential loop (``vmap(scan)``
# collapses to a single N_y-deep batched scan), so the sweep is written
# as a Pallas (Triton) kernel: two ``fori_loop`` passes that read the
# band one scalar at a time by index (``ref[0, 0, i, d]``) and carry the
# sliding window in registers, so no whole-band block is ever loaded
# (Triton supports neither a non-power-of-two block load nor value
# slicing / reversal / scan ``xs``).  The same banded math runs in pure
# JAX as the CPU path / oracle (``_banded_solve_batched``).
#
# Factors are stored banded: ``L`` carries the ``p`` strict sub-diagonals
# of the unit-lower factor (``L[i, i-p+d]``, ``d = 0..p-1``); ``U`` carries
# the diagonal + ``p`` super-diagonals (``U[i, i+d]``, ``d = 0..p``).  The
# operator band uses ``A[i, i-p+d]``, ``d = 0..2p`` (``d = p`` the
# diagonal), out-of-range entries zero.

# Pallas mode-tile per program: ``(bm0, bm1)`` Fourier modes along the
# ``(k_z, k_x)`` axes, read from ``params.solver.pallas_block_m0`` /
# ``pallas_block_m1`` (must be powers of two -- Triton block loads).  Each
# program runs the sequential banded sweep vectorised across its tile, so
# ``bm0 * bm1 * k`` SIMD lanes are filled instead of just ``k``.
#
# Default ``(2, 32)`` is the H100 tuning.  Partial boundary tiles (when the
# tile does not divide the mode plane) are avoided by padding the plane up
# to whole tiles inside :func:`_pallas_banded_solve` -- the masked
# partial-tile path miscompiles on real Triton (it corrupts even full-tile
# programs, nondeterministically, for nontrivial kernels), so the kernel only
# ever runs the correct full-tile path.  The tuning:
#   * ``bm1 = 32`` is one warp wide along the **contiguous innermost** mode
#     axis (``k_x``), so a warp's band load fully coalesces (256 B).  A
#     smaller ``bm1`` splits the warp across the strided ``k_z`` axis
#     (>= 2 transactions per load).
#   * ``bm0 = 2`` gives 4 warps per program (``k = 2``) to hide the
#     dependent-sweep latency (the recurrence has little per-warp ILP).
# A parameter-independent optimum does not exist: total parallelism is
# fixed at ``Nkz * Nkx * k`` threads, so a typical DNS (~1e3-1e4 modes) is
# mode-count-occupancy-limited on an H100 regardless of tile size.  Tuning
# rule: keep ``bm1 >= 32`` (coalescing) and the program count
# ``cdiv(Nkz, bm0) * cdiv(Nkx, bm1)`` at least a few x the SM count; shrink
# the tile for small mode counts, grow it for very large DNS.  Profile
# (Nsight) to finalise -- see the ``gpu-validation-pallas-banded`` plan.


def _banded_lu_factor_single(a_band: Array) -> tuple[Array, Array]:
    r"""No-pivot banded LU of one operator in banded storage.

    Doolittle factorisation `$A = L U$` with `$L$` unit-lower
    (``p`` sub-diagonals) and `$U$` upper (``p`` super-diagonals +
    diagonal); no fill-in because there is no pivoting.  One-time
    (setup); ``vmap`` over modes via :func:`_banded_factor`.

    Parameters
    ----------
    a_band:
        Operator band, shape ``(N, 2p+1)`` with
        ``a_band[i, d] = A[i, i-p+d]``.

    Returns
    -------
    L:
        Strict-lower factor band, ``(N, p)``.
    U:
        Upper factor band (diagonal first), ``(N, p+1)``.
    """
    N = a_band.shape[0]
    p = (a_band.shape[1] - 1) // 2
    dtype = a_band.dtype
    Lband0 = jnp.zeros((N, p), dtype)
    Upad0 = jnp.zeros((N + p, p + 1), dtype)  # Upad[k+p] holds U row k

    def body(i, carry):
        Lband, Upad = carry
        Arow = lax.dynamic_slice(a_band, (i, 0), (1, 2 * p + 1))[0]
        Uwin = lax.dynamic_slice(Upad, (i, 0), (p, p + 1))  # U[i-p..i-1]

        # L row: Lrow[d] = L[i, i-p+d] (j = i-p+d), d = 0..p-1.
        Lrow = jnp.zeros(p, dtype)
        for d in range(p):
            j_valid = (i - p + d) >= 0
            s = Arow[d]
            for e in range(d):  # U[i-p+e, j] sits at offset d-e in [1, d]
                s = s - Lrow[e] * Uwin[e, d - e]
            denom = Uwin[d, 0]
            safe = jnp.where(j_valid, denom, 1.0)
            Lrow = Lrow.at[d].set(jnp.where(j_valid, s / safe, 0.0))

        # U row: Urow[d] = U[i, i+d] (j = i+d), d = 0..p.
        Urow = jnp.zeros(p + 1, dtype)
        for d in range(p + 1):
            s = Arow[p + d]
            for e in range(p):  # contributes only when offset p+d-e <= p
                off = p + d - e
                u = Uwin[e, jnp.clip(off, 0, p)]
                s = s - jnp.where(off <= p, Lrow[e] * u, 0.0)
            Urow = Urow.at[d].set(s)

        Lband = lax.dynamic_update_slice(Lband, Lrow[None], (i, 0))
        Upad = lax.dynamic_update_slice(Upad, Urow[None], (i + p, 0))
        return (Lband, Upad)

    Lband, Upad = lax.fori_loop(0, N, body, (Lband0, Upad0))
    return Lband, Upad[p:]


def _banded_factor(a_band: Array) -> tuple[Array, Array]:
    """Batched :func:`_banded_lu_factor_single` over ``(Nkz, Nkx)``."""
    return jax.vmap(jax.vmap(_banded_lu_factor_single))(a_band)


def _banded_solve_batched(L: Array, U: Array, b: Array, p: int) -> Array:
    r"""Banded forward/back substitution, arbitrary leading batch.

    Solves `$L U x = b$` from banded factors.  Sequential along the
    `$N_y$` axis (axis ``-2``); vectorised over the leading batch dims
    and the trailing RHS-column axis ``k``.  Used directly as the CPU
    solve path and inside the Pallas kernel on each mode tile.

    Parameters
    ----------
    L:
        Strict-lower factor band, ``(..., N, p)``.
    U:
        Upper factor band (diagonal first), ``(..., N, p+1)``.
    b:
        Right-hand side, ``(..., N, k)`` (real; complex RHS are
        carried as ``k = 2`` real columns by the caller).
    p:
        Half-bandwidth.
    """
    p_w = p  # sliding-window depth

    # Forward: L y = b (unit lower).  The carry window holds the last p
    # solved values ``[y[i-p], ..., y[i-1]]``; deriving it via
    # ``zeros_like`` makes it inherit the RHS sharding (a plain
    # ``jnp.zeros`` would be replicated and mismatch under the Explicit
    # mesh -- and inside the Pallas kernel).
    L_scan = jnp.moveaxis(L, -2, 0)  # (N, ..., p)
    b_scan = jnp.moveaxis(b, -2, 0)  # (N, ..., k)
    win0 = jnp.zeros_like(b[..., :p_w, :])

    def fwd(window, xs):
        Li, bi = xs
        yi = bi - jnp.einsum("...d,...dk->...k", Li, window)
        window = jnp.concatenate(
            [window[..., 1:, :], yi[..., None, :]], axis=-2
        )
        return window, yi

    _, ys = lax.scan(fwd, win0, (L_scan, b_scan))
    y = jnp.moveaxis(ys, 0, -2)  # (..., N, k)

    # Back: U x = y (upper band).  Process i = N-1..0; the window holds
    # ``[x[i+1], ..., x[i+p]]`` (zero past the last row).
    U_scan = jnp.moveaxis(U, -2, 0)[::-1]  # (N, ..., p+1), reversed in i
    y_scan = jnp.moveaxis(y, -2, 0)[::-1]
    winx0 = jnp.zeros_like(y[..., :p_w, :])

    def back(window, xs):
        Ui, yi = xs
        s = yi - jnp.einsum("...d,...dk->...k", Ui[..., 1:], window)
        xi = s / Ui[..., 0:1]
        window = jnp.concatenate(
            [xi[..., None, :], window[..., :-1, :]], axis=-2
        )
        return window, xi

    _, xs = lax.scan(back, winx0, (U_scan, y_scan))
    return jnp.moveaxis(xs[::-1], 0, -2)  # (..., N, k)


def _banded_from_dense(A: Array, p: int) -> Array:
    r"""Extract banded storage from a (banded) dense operator.

    ``A`` is ``(..., N, N)`` with half-bandwidth `$\le p$`; returns
    ``(..., N, 2p+1)`` with ``band[..., i, d] = A[..., i, i-p+d]``
    (``d = p`` the diagonal, out-of-range entries zero).  Used to build
    per-mode banded operators from the shared base operator without ever
    forming an ``(N, N)`` per mode.
    """
    N = A.shape[-1]
    cols = []
    for d in range(2 * p + 1):
        off = d - p
        diag = jnp.zeros(A.shape[:-1], A.dtype)  # (..., N)
        dvals = jnp.diagonal(A, offset=off, axis1=-2, axis2=-1)
        if off >= 0:
            diag = diag.at[..., : N - off].set(dvals)
        else:
            diag = diag.at[..., -off:].set(dvals)
        cols.append(diag)
    return jnp.stack(cols, axis=-1)


def _banded_diag_column(p: int, dtype: DTypeLike) -> Array:
    r"""One-hot band column at the diagonal (band offset 0).

    Returns a ``(2p+1,)`` vector that is ``1`` at index ``p`` (the
    diagonal slot ``A[i, i]``) and ``0`` elsewhere.  Doubles as the
    identity row in banded storage, and -- scaled by a per-mode
    diagonal and added to an operator band -- applies a diagonal shift
    without a scatter, so the result inherits the operand's mode
    sharding (a ``jnp.zeros`` + ``.at[].set`` would replicate under the
    Explicit mesh).  See :func:`_assemble_banded_operator`.
    """
    return jnp.zeros(2 * p + 1, dtype).at[p].set(1.0)


def _banded_wall_row(dense_row: Array, i: int, p: int) -> Array:
    r"""Banded form of a full matrix row at a (static) boundary row.

    Given the ``i``-th row ``dense_row`` of an `$(N, N)$` operator
    (e.g. a `$D_1$` Neumann wall row) and its static index ``i``,
    returns the ``(2p+1,)`` band slice
    ``band[d] = dense_row[i - p + d]`` (``d = 0..2p``, out-of-range
    entries zero) -- the same convention as :func:`_banded_from_dense`.
    The gather is a static Python loop (``i`` and ``N`` are static):
    the inner wall ``i = 0`` lands in band columns ``p..2p``, the outer
    wall ``i = N-1`` in columns ``0..p``.
    """
    N = dense_row.shape[0]
    cols = []
    for d in range(2 * p + 1):
        j = i - p + d
        if 0 <= j < N:
            cols.append(dense_row[j])
        else:
            cols.append(jnp.zeros((), dense_row.dtype))
    return jnp.stack(cols)


def _assemble_banded_operator(
    base_band: Array,
    scale: float,
    diag: Array,
    walls: list[tuple[int, Array]],
) -> Array:
    r"""Assemble a per-mode banded operator from a shared base band.

    Computes ``scale * base_band + diag[..., None] * e`` (with ``e``
    the diagonal one-hot, :func:`_banded_diag_column`), then overwrites
    each boundary row ``(idx, row)`` in *walls* via
    ``band.at[..., idx, :].set(row)``.  The diagonal shift uses the
    one-hot trick so the result inherits the per-mode sharding of
    *diag* (see :func:`_banded_diag_column`).  Shared by the
    wall-bounded geometries' ``_build_{Lk,Hk}_band_gpu`` builders.

    The caller pre-broadcasts *base_band* (``(..., N, 2p+1)``; e.g. a
    parity-selected band may carry a leading mode axis) and *diag*
    (``(..., N)`` per radial point, or a row-constant ``(..., 1)``) to
    the operator's mode layout; *walls* rows may be mode-dependent
    (e.g. a mean-mode identity pin via ``jnp.where``).

    Parameters
    ----------
    base_band:
        Base operator in banded storage, ``(..., N, 2p+1)``.
    scale:
        Scalar multiplying *base_band* (``1`` for `$L_k$`,
        `$-c\nu$` for `$H_k$`).
    diag:
        Per-mode diagonal shift, broadcast over the band axis.
    walls:
        ``(row_index, band_row)`` overrides for the boundary rows
        (one per wall; ``band_row`` is ``(..., 2p+1)``).
    """
    p = (base_band.shape[-1] - 1) // 2
    e = _banded_diag_column(p, base_band.dtype)
    band = scale * base_band + diag[..., None] * e
    for idx, row in walls:
        band = band.at[..., idx, :].set(row)
    return band


def _banded_matvec(a_band: Array, x: Array) -> Array:
    """Banded matrix-vector product ``A @ x`` in banded storage.

    ``a_band`` is ``(..., N, 2p+1)``, ``x`` is ``(..., N, k)``;
    returns ``(..., N, k)``.  Used by the setup-time stability check.
    """
    p = (a_band.shape[-1] - 1) // 2
    N = a_band.shape[-2]
    pad = [(0, 0)] * (x.ndim - 2) + [(p, p), (0, 0)]
    xp = jnp.pad(x, pad)
    y = jnp.zeros_like(x)
    for d in range(2 * p + 1):
        y = y + a_band[..., d][..., None] * lax.slice_in_dim(
            xp, d, d + N, axis=-2
        )
    return y


def _pallas_banded_solve(
    L: Array, U: Array, b: Array, p: int, interpret: bool = False
) -> Array:
    r"""Pallas (Triton) per-mode banded solve, mode-tiled and coalesced.

    Solves `$L U x = b$` for every `$(k_z, k_x)$` Fourier mode.  The grid
    covers the mode plane in ``(bm0, bm1)`` tiles (one Pallas program per
    tile); each program runs the sequential banded sweep **vectorised
    across its mode tile**, filling ``bm0 * bm1 * k`` SIMD lanes instead of
    just ``k`` (one mode's ``k`` re/im columns).  ``interpret`` runs the
    same kernel in pure JAX on CPU (correctness; the GPU path uses
    ``interpret=False``).

    Arrays are **mode-inner**: the ``(k_z, k_x)`` axes are the two trailing
    (innermost-contiguous) axes, so the per-step band loads
    ``l_ref[i, d]`` -- a ``(bm0, bm1)`` tile at dynamic row ``i`` and
    static band offset ``d`` -- coalesce along the contiguous ``k_x`` axis.
    The sweep is two ``fori_loop`` passes (forward `$L y = b$`, back
    `$U x = y$`) reading the band **by index** (no whole-band load) and
    keeping the ``p``-deep sliding window in the loop carry (registers).
    Triton lowers neither a non-power-of-two block load, nor a value-array
    slice / reversal, nor a scan over ``xs``, so the host-side
    :func:`_banded_solve_batched` (whole-band loads, ``[1:]`` window
    slices, reversed scan) cannot run on it.  ``bm0``/``bm1`` are powers of
    two so the tile loads are legal, and the mode plane is **padded up to
    whole ``(bm0, bm1)`` tiles** so no boundary tile is ever partial -- a
    masked partial-tile band load miscompiles on real Triton (see below).

    The ``U`` **diagonal is pre-inverted** (stored as `$1/U_{ii}$`), so the
    backward pass multiplies instead of dividing (`$N$` divisions ->
    `$N$` multiplies per mode -- divisions are far costlier on GPU).

    Each ``y[i]`` is stashed in the output ref between the passes.
    Triton-Pallas has **no scratch / shared memory** ("scratch memory not
    implemented in the Triton backend"), so ``y`` cannot move to fast
    storage; it must round-trip a GMEM ref.  The mode-inner layout makes
    that round-trip coalesced (and L2-hot -- written then re-read soon, in
    reverse).  A ``pltriton.debug_barrier()`` is emitted between the two
    passes on the real-GPU path (it has no CPU lowering) as a defensive
    multi-warp fence -- so that, when a tile spans more than one warp, every
    forward store is globally visible before the backward reads (the GPU
    diagnostic found the round-trip coherent even without it).

    **Partial-tile masking (why the plane is padded).**  When the tile does
    not divide the mode plane (``Nkz % bm0`` or ``Nkx % bm1`` nonzero -- the
    common case, since the real-FFT axis ``Nkx = nx/2 + 1`` is rarely a
    multiple of ``bm1``), the boundary tile is partial and Triton masks the
    loads.  That **masked partial-tile path miscompiles on real Triton**: in
    a grid with a partial boundary tile it corrupts results *across the grid*
    -- **even full-tile programs** -- **nondeterministically**
    (warp-scheduling dependent), for any nontrivial kernel (the masked
    double-index band loads ``l_ref[i, d]`` / ``u_ref[i, d]`` *and* even a
    single-index window-carry sweep), while interpret mode and the CUDA
    lowering both accept it.  ``scripts/pallas_tiling_diagnostic.py``
    localised it: only trivial copy/round-trip kernels survive a partial
    plane; full tiles (no partial boundary anywhere) are always correct at
    ``(2, 32)``.  The fix is to **pad the mode plane up to whole tiles**
    (below) so the kernel only ever runs the correct full-tile path; the
    padded modes get zero
    ``L``/``U``/``b`` and -- because the ``U`` diagonal is pre-inverted, so
    the backward sweep multiplies, never divides -- solve to a clean zero
    (no NaN) and are cropped off the result.  The sequential `$N$`-loop
    itself is an intrinsic recurrence (no Triton-lowerable parallel scan);
    the only parallelism is across modes, which the tiling + grid maximise.

    Parameters
    ----------
    L, U:
        Mode-inner banded factors, ``(N, p, Nkz, Nkx)`` (strict lower,
        ``L[i, d] = L_{i, i-p+d}``) / ``(N, p+1, Nkz, Nkx)`` (diagonal
        first and **reciprocated**: ``U[i, 0] = 1/U_{i,i}``,
        ``U[i, d] = U_{i, i+d}`` for ``d >= 1``).
    b:
        Mode-inner right-hand side, ``(N, k, Nkz, Nkx)`` (real).
    p:
        Half-bandwidth.
    interpret:
        Run the kernel in Pallas interpret mode (CPU).
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    N, _, Nkz, Nkx = L.shape
    k = b.shape[1]
    bm0 = params.solver.pallas_block_m0
    bm1 = params.solver.pallas_block_m1

    # Pad the mode plane up to whole ``(bm0, bm1)`` tiles so no boundary
    # tile is partial (a masked partial-tile band load miscompiles on real
    # Triton -- see the docstring).  Zero-fill is NaN-safe: padded modes
    # solve to zero (the backward sweep multiplies by the pre-inverted
    # diagonal, never divides).  The result is cropped back to the original
    # ``(Nkz, Nkx)``.  A no-op when the plane already tiles evenly.
    Nkz_pad = ((Nkz + bm0 - 1) // bm0) * bm0
    Nkx_pad = ((Nkx + bm1 - 1) // bm1) * bm1
    if (Nkz_pad, Nkx_pad) != (Nkz, Nkx):
        mode_pad = [(0, 0), (0, 0), (0, Nkz_pad - Nkz), (0, Nkx_pad - Nkx)]
        L = jnp.pad(L, mode_pad)
        U = jnp.pad(U, mode_pad)
        b = jnp.pad(b, mode_pad)

    def kernel(l_ref, u_ref, b_ref, x_ref):
        # Window row of zeros, vectorised over the (bm0, bm1) mode tile.
        zero = jnp.zeros((k, bm0, bm1), b.dtype)

        # Forward: L y = b (unit lower).  window[d] = y[i-p+d] (zero
        # before row 0); y[i] is stashed in x_ref for the backward pass.
        def fwd(i, window):
            yi = b_ref[i]  # (k, bm0, bm1)
            for d in range(p):
                yi = yi - l_ref[i, d][None] * window[d]
            x_ref[i] = yi
            return window[1:] + (yi,)

        lax.fori_loop(0, N, fwd, (zero,) * p)

        # The forward pass stashes each ``y[i]`` in the output GMEM ref;
        # the backward pass reads it back.  When a tile spans >1 warp the
        # forward store and backward load need not share a thread->element
        # layout, so this barrier makes all forward stores globally visible
        # first (lowers to ``__syncthreads()``; no CPU lowering, GPU-only).
        # Defensive multi-warp fence: the GPU diagnostic found the round-trip
        # coherent even without it (the ``bm1 > 1`` miscompile was the masked
        # partial-tile band load, fixed by padding the plane, not a visibility
        # bug), but it is kept as a cheap correctness guard.  See the
        # docstring.
        if not interpret:
            pltriton.debug_barrier()

        # Back: U x = y (upper).  Walk i = N-1..0; window[d] = x[i+1+d]
        # (zero past the last row), read y[i] back from x_ref.  The
        # diagonal is pre-inverted, so the divide becomes a multiply.
        def back(t, window):
            i = N - 1 - t
            s = x_ref[i]  # y[i]
            for d in range(p):
                s = s - u_ref[i, d + 1][None] * window[d]
            xi = s * u_ref[i, 0][None]
            x_ref[i] = xi
            return (xi,) + window[:-1]

        lax.fori_loop(0, N, back, (zero,) * p)

    grid = (Nkz_pad // bm0, Nkx_pad // bm1)

    def idx(i, j):
        return (0, 0, i, j)

    out = pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=[
            pl.BlockSpec((N, p, bm0, bm1), idx),
            pl.BlockSpec((N, p + 1, bm0, bm1), idx),
            pl.BlockSpec((N, k, bm0, bm1), idx),
        ],
        out_specs=pl.BlockSpec((N, k, bm0, bm1), idx),
        out_shape=jax.ShapeDtypeStruct((N, k, Nkz_pad, Nkx_pad), b.dtype),
        # Deferred: ``input_output_aliases={2: 0}`` (alias ``b`` -> output)
        # is correct here (the forward pass writes y[i] only after reading
        # b[i], and no b[j] is re-read once overwritten) and lowers +
        # interprets cleanly, dropping one output allocation.  Left out on
        # purpose: the saving is allocation-only (this kernel is bound by
        # memory *bandwidth*, unchanged by aliasing), it adds a donation
        # constraint across the vmapped ``.solve`` paths and the deferred
        # multi-device GPU sharding, and XLA buffer assignment + the
        # stepper's ``donate_argnums`` likely already reuse the dead ``b``.
        # Revisit if GPU profiling shows a peak-memory win.
        # Force Triton: JAX's default Pallas GPU backend (Mosaic GPU,
        # via jax_pallas_use_mosaic_gpu=True) rejects f64 in its TMA
        # gmem->smem copy ("unsupported TMA dtype f64").  Triton (this
        # kernel's intended backend, also the ROCm path) has no such
        # limit.
        compiler_params=pltriton.CompilerParams(
            num_warps=params.solver.pallas_num_warps,
            num_stages=params.solver.pallas_num_stages,
        ),
        interpret=interpret,
    )(L, U, b)
    # Crop the padded modes off (no-op when the plane tiled evenly).
    return out[:, :, :Nkz, :Nkx]


def _banded_mode_solve(L: Array, U: Array, rhs: Array) -> Array:
    r"""Solve one mode-inner banded operator over all ``(Nkz, Nkx)`` modes.

    ``L``/``U`` are the **mode-inner** factors stored by
    :class:`PerModeBandedPallasOperator` (``(N, p, Nkz, Nkx)`` /
    ``(N, p+1, Nkz, Nkx)``, ``U`` diagonal reciprocated).  ``rhs`` is the
    **mode-inner** spectral field ``(N, Nkz, Nkx)`` of the public
    ``.solve`` contract -- the velocity field's native layout
    (``sharding.spec_scalar_shard``).  A complex RHS is split into
    ``k = 2`` real columns (the factors are real), solved, and recombined.

    On GPU the re/im split is the **only** layout touch: stacking the real
    and imaginary parts on a new axis 1 lands directly in the kernel's
    ``(N, k, Nkz, Nkx)`` layout (no transpose), and the recombine reads the
    two columns back.  Because the contract is mode-inner the hot path
    feeds this with no transpose at all -- it previously round-tripped
    ``(N,Nkz,Nkx) <-> (Nkz,Nkx,N)`` around every ``.solve`` (a round-trip
    XLA did not fuse away, ~half this memory-bound solve's HBM traffic).

    On CPU the factors *and* RHS are moved back to mode-outer (and the
    ``U`` diagonal un-inverted) for the standard
    :func:`_banded_solve_batched` (``N_y`` on matrix axis -2); the CPU path
    is the oracle / fallback, not the performance target, so it absorbs the
    transpose internally.
    """
    p = L.shape[1]
    is_complex = jnp.iscomplexobj(rhs)
    if jax.default_backend() == "gpu":
        if is_complex:
            # (N, Nkz, Nkx) complex -> (N, k, Nkz, Nkx) real, re/im on
            # axis 1: already the kernel layout, no transpose.
            b = jnp.stack([rhs.real, rhs.imag], axis=1)
        else:
            b = rhs[:, None]  # (N, 1, Nkz, Nkx)
        x = _pallas_banded_solve(L, U, b, p)  # (N, k, Nkz, Nkx)
        return lax.complex(x[:, 0], x[:, 1]) if is_complex else x[:, 0]
    # CPU fallback: standard mode-outer banded sweep (RHS and factors
    # moved to (Nkz, Nkx, N, .) internally).
    Lo = jnp.moveaxis(L, (0, 1), (-2, -1))  # (Nkz, Nkx, N, p)
    Uo = jnp.moveaxis(U, (0, 1), (-2, -1))  # (Nkz, Nkx, N, p+1)
    Uo = Uo.at[..., 0].set(1.0 / Uo[..., 0])  # un-invert the diagonal
    rhs_o = jnp.moveaxis(rhs, 0, -1)  # (N, Nkz, Nkx) -> (Nkz, Nkx, N)
    b = _real_rhs_view(rhs_o) if is_complex else rhs_o[..., None]
    x = _banded_solve_batched(Lo, Uo, b, p)  # (Nkz, Nkx, N, k)
    x = _complex_from_view(x) if is_complex else x[..., 0]  # (Nkz,Nkx,N)
    return jnp.moveaxis(x, -1, 0)  # -> (N, Nkz, Nkx)


@register_dataclass_pytree
@dataclass
class PerModeBandedPallasOperator:
    r"""Banded operator solved per-mode via a Pallas sweep.

    Holds the no-pivot banded LU factors (real) in **mode-inner** storage
    (the ``(k_z, k_x)`` mode axes trailing/innermost, for coalesced GPU
    loads) with the ``U`` diagonal **reciprocated**; the solve is the
    mode-tiled banded substitution (Pallas/Triton on GPU, pure-JAX
    mode-outer sweep on CPU).  Build via :meth:`from_banded_factors` from
    the standard mode-outer factors of :func:`_banded_factor`.  Same
    public ``.solve`` contract (mode-outer ``(Nkz, Nkx, N)`` spectral
    field) and component-axis dispatch as :class:`PerModeBandedOperator`.

    Attributes
    ----------
    L:
        Strict-lower factor band, mode-inner,
        ``(N, p, Nkz, Nkx)`` or ``(C, N, p, Nkz, Nkx)``.
    U:
        Upper factor band (reciprocated diagonal first), mode-inner,
        ``(N, p+1, Nkz, Nkx)`` or ``(C, ...)``.
    """

    L: Array
    U: Array

    @classmethod
    def from_banded_factors(
        cls, L: Array, U: Array
    ) -> PerModeBandedPallasOperator:
        r"""Build from standard mode-outer banded factors.

        Transposes the factors from the mode-outer layout produced by
        :func:`_banded_factor` (``(Nkz, Nkx, N, p)`` /
        ``(..., N, p+1)``) to the mode-inner storage solved here
        (``(N, p, Nkz, Nkx)`` / ``(N, p+1, Nkz, Nkx)``), and
        reciprocates the ``U`` diagonal so the GPU backward sweep
        multiplies instead of dividing (see :func:`_pallas_banded_solve`).
        Mirrors :meth:`DenseJAXSolver.from_factors`.
        """
        Li = jnp.moveaxis(L, (-2, -1), (0, 1))  # (N, p, Nkz, Nkx)
        Ui = jnp.moveaxis(U, (-2, -1), (0, 1))  # (N, p+1, Nkz, Nkx)
        Ui = Ui.at[:, 0].set(1.0 / Ui[:, 0])  # reciprocate diagonal slot
        return cls(L=Li, U=Ui)

    def solve(self, rhs: Array, component_axis: int = 0) -> Array:
        """Batched banded solve across ``(kz, kx)`` modes.

        ``rhs`` is **mode-inner** -- ``(N, Nkz, Nkx)`` or
        ``(C, N, Nkz, Nkx)`` for a leading batch axis ``C`` -- the
        velocity field's native spectral layout
        (``sharding.spec_scalar_shard`` / ``spec_vector_shard``); the
        result has the same shape and dtype.  This is the layout the
        Pallas kernel stores its factors in, so the GPU path feeds the
        kernel with no transpose (see :func:`_banded_mode_solve`).

        Dispatch mirrors :meth:`PerModeBandedOperator.solve`:
        ``L.ndim == 5`` (batched/stacked operators) vmaps over both
        the operator and RHS leading axis; ``rhs.ndim == 4`` (shared
        operator) vmaps over the RHS leading axis; otherwise a single
        operator / single RHS.

        *component_axis* is the position of that batched RHS axis:
        ``0`` (default, ``(C, N, ...)``) or ``1`` (``(N, C, ...)``,
        the y-leading layout the IMM's Hk construction uses so the
        matvecs stay transpose-free -- see ``apply_y_matrix``).  The
        Pallas kernel is per-mode, so *component_axis* only picks the
        ``vmap`` axis; the kernel itself never transposes and the
        output preserves the input layout.

        Deferred optimization (stacked ``component_axis=1`` path).  For
        the stacked Hk solve (``L.ndim == 5``), ``out_axes=1`` makes XLA
        emit one output-repositioning transpose
        `$(C, N, \ldots) \to (N, C, \ldots)$` on the complex-
        reconstructed result (seen on the H100 optimized HLO as a
        ``c128[N, C, 1, Nkz, Nkx]`` ``dimensions={1,0,2,3,4}`` transpose
        under ``vmap()/complex``).  This is the **only** IMM transpose
        the y-leading contract leaves, and it is a net win: feeding
        ``R_stack`` y-leading removes the three larger ``D1``/``D2``
        matvec transposes at the cost of this one (measured as a
        corrector-step 252 -> 177 optimized-HLO transpose drop), and
        ``component_axis=0`` would merely move it back to the input side
        *and* reintroduce those matvec transposes.  Eliminating it
        entirely means folding the 3-component stack into the kernel's
        `$(k_z, k_x)$` batch so there is no ``vmap`` over components at
        all -- a solve-kernel refactor worth ~2% of the step, deferred
        as low-ROI and cross-cutting.
        """
        ca = component_axis
        if self.L.ndim == 5:
            return jax.vmap(
                _banded_mode_solve, in_axes=(0, 0, ca), out_axes=ca
            )(self.L, self.U, rhs)
        if rhs.ndim == 4:
            return jax.vmap(
                _banded_mode_solve, in_axes=(None, None, ca), out_axes=ca
            )(self.L, self.U, rhs)
        return _banded_mode_solve(self.L, self.U, rhs)


def _stack_pallas_operators(
    *ops: PerModeBandedPallasOperator,
) -> PerModeBandedPallasOperator:
    """Stack Pallas banded operators along a leading component axis."""
    return PerModeBandedPallasOperator(
        L=jnp.stack([o.L for o in ops]),
        U=jnp.stack([o.U for o in ops]),
    )


def _banded_residual(a_band: Array, L: Array, U: Array) -> float:
    """Max relative residual ``||A x - b|| / ||b||`` of the no-pivot
    banded solve (``b = 1``), as a host float for the pivot decision."""
    p = L.shape[-1]
    # ``ones_like`` a slice of the (mode-sharded) operator so the test
    # RHS inherits its sharding -- a plain ``jnp.ones`` is replicated and
    # mismatches the factors under the Explicit mesh.
    b = jnp.ones_like(a_band[..., :1])  # (..., N, 1), inherits sharding
    x = _banded_solve_batched(L, U, b, p)
    r = _banded_matvec(a_band, x) - b
    num = jnp.max(jnp.abs(r))
    den = jnp.max(jnp.abs(b))
    return float(num / den)


def _decide_pallas_or_spike(
    a_bands: list[Array],
    force_pivoting: bool,
    label: str,
    make_spike: Callable[[], PerModeBandedOperator],
) -> PerModeBandedPallasOperator | PerModeBandedOperator:
    r"""Choose the no-pivot Pallas solver or the pivoted SPIKE fallback.

    For one operator group (``a_bands`` has one banded operator, or
    several that must share a backend so a stacked operator stays
    homogeneous):

    - ``force_pivoting`` -> pivoted SPIKE (``make_spike``).
    - else factor no-pivot, measure the max relative solve residual;
      if it exceeds ``params.solver.pallas_stability_tol``, fall back to
      SPIKE.
    - else build the (possibly stacked) Pallas operator.

    A diagnostic line is printed at setup for the group either way.

    Parameters
    ----------
    a_bands:
        Banded operators ``(Nkz, Nkx, N, 2p+1)`` sharing a backend.
    force_pivoting:
        Skip the no-pivot path and use SPIKE.
    label:
        Operator-group name for the diagnostic (e.g. ``"Hk"``).
    make_spike:
        Builds the pivoted SPIKE operator for this group (stacked if
        ``len(a_bands) > 1``).
    """
    if force_pivoting:
        sharding.print(f"[pallas] {label}: forced pivoting -> SPIKE solver")
        return make_spike()

    factors = [_banded_factor(A) for A in a_bands]
    resid = max(
        _banded_residual(A, L, U)
        for A, (L, U) in zip(a_bands, factors, strict=True)
    )

    # ``not (resid <= tol)`` also catches a non-finite residual from a
    # no-pivot breakdown (``nan > tol`` would be False).
    tol = params.solver.pallas_stability_tol
    if not (resid <= tol):
        sharding.print(
            f"[pallas] {label}: no-pivot residual {resid:.2e} > "
            f"{tol:.0e} -> pivoted SPIKE fallback"
        )
        return make_spike()

    sharding.print(
        f"[pallas] {label}: no-pivot banded LU (residual {resid:.2e})"
    )
    ops = [
        PerModeBandedPallasOperator.from_banded_factors(L, U)
        for (L, U) in factors
    ]
    return ops[0] if len(ops) == 1 else _stack_pallas_operators(*ops)
