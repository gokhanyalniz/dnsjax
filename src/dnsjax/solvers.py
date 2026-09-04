r"""Geometry-independent linear solver infrastructure.

Provides the two solver backends used by wall-bounded geometries: the
Pallas per-mode banded production solver
(:class:`PerModeBandedPallasOperator`, ``solver.backend = "pallas"``,
the default) and the batched dense LU reference solver
(:class:`DenseJAXSolver`, ``"dense"`` -- full `$(N_y, N_y)$` pivoted
factors per Fourier mode, kept for mathematical readability and as
the regression oracle for the banded path).  Both solver classes
support a leading batch axis (e.g. the 3 velocity components)
transparently via an extra ``vmap``.

The banded path factors each per-mode operator with a **no-pivot**
banded LU (:func:`_banded_lu_factor_single`); pivoting is never
needed for the diagonally-dominant Helmholtz/Poisson-like operators
solved here.  :func:`_build_pallas_operator` verifies this once at
setup (solve-residual probe + LU element-growth check) and hard-errors
with an actionable message on a genuinely unstable factorisation
instead of proceeding silently.

Complex right-hand sides
------------------------
All operators here are **real**; only the RHS may be complex.  To
avoid promoting the (large) factors to complex on every solve --
which `jax.scipy.linalg.lu_solve` would do, tripling the factor
memory traffic and doubling the triangular-solve FLOPs -- a complex
RHS is split into a real array with a trailing re/im axis of
length 2 (`$\ldots, N_y$` complex `$\to \ldots, N_y, 2$` real)
and solved as two real RHS columns, then recombined.  The split
and merge are single fused elementwise passes over the RHS, far
cheaper than the factor-sized conversion they replace.  The
permutations are precomputed at factorisation time so the solve
path (:func:`_permuted_tri_solve`: permutation gather + two
batched :func:`jax.lax.linalg.triangular_solve` calls) needs no
per-call pivot conversion.

The Pallas backend's solve is a ``shard_map``-local region (mirroring
the :mod:`dnsjax.fft` pipeline): the per-mode systems are independent,
so each device runs the kernel/sweep on its local mode-plane block
with zero communication, and all tile pad/crop bookkeeping happens on
local arrays where no Explicit-mesh sharding rules apply.  See
:meth:`PerModeBandedPallasOperator.solve`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.scipy.linalg as sla
from jax import Array, lax, shard_map
from jax import numpy as jnp
from jax.lax import linalg as lax_linalg
from jax.sharding import PartitionSpec as P
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
        on entry and back on exit -- a pure axis permutation, done here
        rather than at the call sites so the Pallas backend can
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
    so the kernel only ever runs the correct full-tile path; the padded
    modes get zero
    ``L``/``U``/``b`` and -- because the ``U`` diagonal is pre-inverted, so
    the backward sweep multiplies, never divides -- solve to a clean zero
    (no NaN) and are cropped off the result.  The **factors are padded
    once at construction** (:meth:`PerModeBandedPallasOperator.
    from_banded_factors`), not per call: ``Nkz = nz - 1`` is odd, so the
    plane virtually never tiles evenly, and a per-call ``jnp.pad`` of the
    factors would re-copy them (holding a transient duplicate) on every
    solve of every step.  Only the RHS is padded here per call (with a
    fallback factor pad for direct callers passing true-plane factors).
    The sequential `$N$`-loop
    itself is an intrinsic recurrence (no Triton-lowerable parallel scan);
    the only parallelism is across modes, which the tiling + grid maximise.

    Parameters
    ----------
    L, U:
        Mode-inner banded factors, ``(N, p, Nkz*, Nkx*)`` (strict lower,
        ``L[i, d] = L_{i, i-p+d}``) / ``(N, p+1, Nkz*, Nkx*)`` (diagonal
        first and **reciprocated**: ``U[i, 0] = 1/U_{i,i}``,
        ``U[i, d] = U_{i, i+d}`` for ``d >= 1``).  The mode plane is
        normally already padded to whole tiles (the stored form); a
        true-plane factor pair is padded here as a fallback.
    b:
        Mode-inner right-hand side, ``(N, k, Nkz, Nkx)`` (real), at the
        **true** mode plane (its trailing dims set the crop of the
        result).
    p:
        Half-bandwidth.
    interpret:
        Run the kernel in Pallas interpret mode (CPU).
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    N = L.shape[0]
    k = b.shape[1]
    Nkz, Nkx = b.shape[2], b.shape[3]
    bm0 = params.solver.pallas_block_m0
    bm1 = params.solver.pallas_block_m1

    # Whole-``(bm0, bm1)``-tile mode plane, so no boundary tile is
    # partial (a masked partial-tile band load miscompiles on real
    # Triton -- see the docstring).  Zero-fill is NaN-safe: padded modes
    # solve to zero (the backward sweep multiplies by the pre-inverted
    # diagonal, never divides).  The **stored factors are already padded
    # to this plane at construction** (``from_banded_factors``), so only
    # the per-call RHS is padded here; factors from a direct caller that
    # are still at the true plane take the same pad as a fallback.  The
    # kernel plane is the whole-tile roundup of the **larger** of the
    # RHS's true plane and the stored factor plane: factors padded at
    # construction under a *different* (larger) tile than the runtime
    # one are grown, never shrunk (a negative ``jnp.pad`` raises) --
    # their extra rows are valid zero-solving padded modes either way.
    # The result is cropped back to the RHS's ``(Nkz, Nkx)``.
    Nkz_need = max(Nkz, L.shape[2])
    Nkx_need = max(Nkx, L.shape[3])
    Nkz_pad = ((Nkz_need + bm0 - 1) // bm0) * bm0
    Nkx_pad = ((Nkx_need + bm1 - 1) // bm1) * bm1
    if (L.shape[2], L.shape[3]) != (Nkz_pad, Nkx_pad):
        fac_pad = [
            (0, 0),
            (0, 0),
            (0, Nkz_pad - L.shape[2]),
            (0, Nkx_pad - L.shape[3]),
        ]
        L = jnp.pad(L, fac_pad)
        U = jnp.pad(U, fac_pad)
    if (Nkz_pad, Nkx_pad) != (Nkz, Nkx):
        b = jnp.pad(
            b, [(0, 0), (0, 0), (0, Nkz_pad - Nkz), (0, Nkx_pad - Nkx)]
        )

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
        # limit. ``pltriton.CompilerParams`` selects Triton.
        compiler_params=pltriton.CompilerParams(),
        interpret=interpret,
    )(L, U, b)
    # Crop the padded modes off (no-op when the plane tiled evenly).
    return out[:, :, :Nkz, :Nkx]


# Test-only override: force :func:`_banded_mode_solve` onto the Pallas
# kernel branch while tracing on a CPU-only box (the branch condition is
# trace-time Python).  Lets the CPU test suite *lower* the
# ``shard_map(pallas_call)`` composition for cuda -- the composition
# whose trace-time failures (e.g. the ``check_vma`` out-shape rule) are
# otherwise reachable only on a real GPU, because the CPU branch never
# calls ``pallas_call``.  See ``test_pallas_cuda_lowering_sharded_solve``.
_force_kernel_path: bool = False


def _kernel_path() -> bool:
    r"""Whether this run solves through the Pallas kernel.

    The single predicate behind both the storage choice
    (:meth:`PerModeBandedPallasOperator.from_banded_factors`) and the
    solve dispatch (:func:`_banded_mode_solve`), so the two can never
    disagree about what the stored factors mean.  Host-side Python
    either way -- the backend is fixed once ``bootstrap`` has run.

    A test flipping :data:`_force_kernel_path` must therefore flip it
    **before building the operator**, not merely before solving.
    """
    return _force_kernel_path or jax.default_backend() == "gpu"


def _banded_mode_solve(L: Array, U: Array, rhs: Array) -> Array:
    r"""Solve one mode-inner banded operator over a ``(Nkz, Nkx)`` block.

    **Local body** of the ``shard_map`` region in
    :meth:`PerModeBandedPallasOperator.solve`: every argument is a
    device-local block (on one device, local = global).  ``L``/``U``
    are the **mode-inner** factors (``(N, p, nkz*, nkx*)`` /
    ``(N, p+1, nkz*, nkx*)``) in whichever of the two per-backend forms
    :meth:`~PerModeBandedPallasOperator.from_banded_factors` stored --
    ``U`` diagonal reciprocated and the plane tile-padded per shard on
    the kernel path, plain diagonal at the true plane on CPU; the
    branch below and the storage share :func:`_kernel_path`, so they
    always agree.  ``rhs`` is the **mode-inner** spectral block
    ``(N, nkz, nkx)`` at the true local plane.  A complex RHS is split
    into ``k = 2`` real columns (the factors are real), solved, and
    recombined.

    On GPU the re/im split is the **only** layout touch: stacking the real
    and imaginary parts on a new axis 1 lands directly in the kernel's
    ``(N, k, Nkz, Nkx)`` layout (no transpose), and the recombine reads the
    two columns back.  Because the contract is mode-inner the hot path
    feeds this with no transpose at all; a mode-outer contract would
    round-trip ``(N,Nkz,Nkx) <-> (Nkz,Nkx,N)`` around every ``.solve``
    instead (a round-trip XLA does not fuse away, ~half this
    memory-bound solve's HBM traffic).

    *The split-real hoist: measured, and rejected.*  That split and
    recombine are **mandatory** per solve -- JAX has no zero-copy
    complex<->real bitcast, the f64 Triton kernel cannot ingest
    ``c128``, and the CPU sweep runs on real columns too
    (:func:`_real_rhs_view`) -- and some of them look redundant
    *between* consumers: ``Hk_op.solve`` recombines its result to
    complex, the caller only indexes or linearly combines it, and
    ``Lk_op.solve`` splits it straight back apart.  XLA does **not**
    simplify them away (optimized CPU HLO: one ``.solve`` emits one
    ``complex`` and one ``real``/``imag`` pair; one ``_imm_iteration``
    emits 12/6/6 Cartesian, 15/6/6 annular, 23/8/8 pipe).

    Carrying the field split-real across that chain nevertheless
    **loses**.  ``pallas_solve_profile.py`` Part A2 times the real
    ``Hk.solve -> map -> Lk.solve`` chain both ways, fidelity-gated,
    factors as jit arguments (CPU, one device):

    ==============  =========  =========
    geometry        96-ish      `$128^3$`
    ==============  =========  =========
    plane-couette    -0.6 %      +9.1 %
    taylor-couette  -11.0 %     -10.0 %
    pipe             -5.6 %     -14.7 %
    ==============  =========  =========

    (Positive = hoisting is faster.)  All six are **isolated**-chain
    figures, and this module's own layout history (below) is precisely
    that such figures rank options backwards -- so the five negatives
    are no better as proof than the one positive is as refutation, and
    the table decides nothing on its own.

    What decides it is the measurement one level up: Part A's fused
    ``full - sweep`` is already ``<= 0`` on CPU, i.e. the split and
    recombine cost nothing measurable *inside the step*, so there is no
    time there for a hoist to win back.  The A2 table is recorded
    because it is what was run and because its sign does not contradict
    that; it is not the reason for the decision.  The mechanism, if one
    is wanted, is the layout finding's: pre-materialising a
    representation that suits the two solves constrains layout
    assignment across everything around them.

    The two arms agree to ~1e-15, which is the bar: the hoist changes
    only the representation a value is carried in, and XLA is free to
    contract differently around a differently-consumed sweep output.

    On CPU the factors *and* RHS are moved to mode-outer for the
    standard :func:`_banded_solve_batched` (``N_y`` on matrix axis -2).
    There is no crop and no un-inversion: the CPU build already stores
    the plain diagonal at the true plane
    (:meth:`~PerModeBandedPallasOperator.from_banded_factors`).

    **The two permutations stay, and the layout is shared with the
    kernel, because that is what measures fastest -- on CPU too.**  The
    source reads as though the sweep should prefer its own storage, and
    three CPU-native layouts were tried end to end (plane-Couette
    ``64 x 96 x 64``, 29 steady steps, one device, every variant
    bit-identical):

    ===========================  =============  ==============
    stored layout                isolated solve  full step
    ===========================  =============  ==============
    mode-inner (this one)         1.00x           0.44 s
    mode-outer ``(Nkz,Nkx,N,p)``  0.73x           0.70 s
    N-first ``(N,Nkz,Nkx,p)``     0.52x           ~0.9-1.3 s
    ===========================  =============  ==============

    The ranking **inverts**: the more the stored layout is tailored to
    the sweep in isolation, the slower the step gets, monotonically.
    The factors are jit *arguments* (the stepper takes ``flow`` as one),
    so their stored layout constrains layout assignment across the whole
    step; pre-materialising a solve-optimal arrangement wins the solve
    and loses more elsewhere.  Setup compile degrades with it too
    (27 s -> 53 s total wall for mode-outer).

    So: do not re-derive this from a standalone ``jit`` of one solve, or
    from any isolated solve timing -- both rank the options backwards.
    Only an end-to-end step measurement decides it.
    """
    p = L.shape[1]
    is_complex = jnp.iscomplexobj(rhs)
    if _kernel_path():
        if is_complex:
            # (N, Nkz, Nkx) complex -> (N, k, Nkz, Nkx) real, re/im on
            # axis 1: already the kernel layout, no transpose.
            b = jnp.stack([rhs.real, rhs.imag], axis=1)
        else:
            b = rhs[:, None]  # (N, 1, Nkz, Nkx)
        x = _pallas_banded_solve(L, U, b, p)  # (N, k, Nkz, Nkx)
        return lax.complex(x[:, 0], x[:, 1]) if is_complex else x[:, 0]
    # CPU: standard mode-outer banded sweep (RHS and factors moved to
    # (Nkz, Nkx, N, .) internally).  The CPU build stores the plain
    # diagonal at the true plane, so neither the tile crop nor the
    # diagonal un-inversion the kernel storage would need is here --
    # both were measured, and removing them is where the CPU path's
    # win came from (see ``from_banded_factors``).
    Lo = jnp.moveaxis(L, (0, 1), (-2, -1))  # (Nkz, Nkx, N, p)
    Uo = jnp.moveaxis(U, (0, 1), (-2, -1))  # (Nkz, Nkx, N, p+1)
    rhs_o = jnp.moveaxis(rhs, 0, -1)  # (N, Nkz, Nkx) -> (Nkz, Nkx, N)
    b = _real_rhs_view(rhs_o) if is_complex else rhs_o[..., None]
    x = _banded_solve_batched(Lo, Uo, b, p)  # (Nkz, Nkx, N, k)
    x = _complex_from_view(x) if is_complex else x[..., 0]  # (Nkz,Nkx,N)
    return jnp.moveaxis(x, -1, 0)  # -> (N, Nkz, Nkx)


# Tile-padding diagnostics already reported (one entry per distinct
# local-plane/tile geometry): every operator built for a run shares the
# same mode plane, so Lk/Hk/Hc would otherwise repeat the same note.
_tile_pad_reported: set[tuple[int, int, int, int]] = set()


@register_dataclass_pytree
@dataclass
class PerModeBandedPallasOperator:
    r"""Banded operator solved per-mode via a Pallas sweep.

    Holds the no-pivot banded LU factors (real) in **mode-inner** storage
    (the ``(k_z, k_x)`` mode axes trailing/innermost, for coalesced GPU
    loads); the solve is the mode-tiled banded substitution
    (Pallas/Triton on GPU, pure-JAX mode-outer sweep on CPU).  Build via
    :meth:`from_banded_factors` from the standard mode-outer factors of
    :func:`_banded_factor`.  The public ``.solve`` contract takes the
    **mode-inner** ``(N, Nkz, Nkx)`` spectral field, the velocity's
    native layout -- see :meth:`solve` for the component-axis dispatch.

    **The layout is shared by both backends; two transforms on top of
    it are per-backend** (:func:`_kernel_path` decides, and is the
    single predicate behind both the storage and the sweep, so the two
    cannot disagree):

    ==================  ==================  ==================
    stored form         kernel path (GPU)   CPU run
    ==================  ==================  ==================
    ``U`` diagonal      reciprocated        plain
    mode plane          whole-tile padded   the true plane
    ==================  ==================  ==================

    A CPU run never reaches the kernel, so both kernel transforms would
    only be undone again on every solve.  The two backends therefore
    share this class, the layout, the factorisation and the ``.solve``
    skeleton, and differ in exactly those two rows plus the local solve
    body.

    Attributes
    ----------
    L:
        Strict-lower factor band, mode-inner,
        ``(N, p, Nkz*, Nkx*)`` or ``(C, N, p, Nkz*, Nkx*)``.
    U:
        Upper factor band, mode-inner, ``(N, p+1, Nkz*, Nkx*)`` or
        ``(C, ...)``; diagonal first, **reciprocated on the kernel path
        only** (plain on CPU -- see the table above).

    ``Nkz* x Nkx*`` is the stored mode plane: on the kernel path the
    true plane rounded up to whole Pallas tiles **per device shard** at
    construction (zero-filled padded modes; see
    :meth:`from_banded_factors`) -- slightly larger persistent storage
    in exchange for no per-solve factor pad/copy -- and on CPU the true
    plane itself (``Nkz* x Nkx* == Nkz x Nkx``).  Either way ``.solve``
    takes and returns the **true** mode plane and runs as a
    ``shard_map``-local region.
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
        (``(N, p, Nkz, Nkx)`` / ``(N, p+1, Nkz, Nkx)``).  Mirrors
        :meth:`DenseJAXSolver.from_factors`.

        **The layout is shared by both backends; two transforms on top
        of it are not.**  Only when the run will actually reach the
        kernel (:func:`_kernel_path`) does this additionally reciprocate
        the ``U`` diagonal -- so the GPU backward sweep multiplies
        instead of dividing, see :func:`_pallas_banded_solve` -- and
        **pre-pad the mode plane up to whole ``(pallas_block_m0,
        pallas_block_m1)`` tiles** with zeros (zero factor rows solve
        padded modes to a clean zero; the reciprocated diagonal makes
        the backward sweep multiply, never divide).  A CPU run needs
        neither: its sweep divides by the diagonal directly and its
        grid is the true plane, so it stored a reciprocal it had to
        un-invert and a pad it had to crop, on every solve.  Dropping
        both is worth **+2.3 % of the step at plane-Couette
        ``64 x 96 x 64``, +4.0 % at plane-Couette ``128^3``, and
        +1.5 % on the pipe at ``128^3``**, and the padded factor memory
        goes with it.

        The ``128^3`` figure is reproduced **three independent ways**:
        an interleaved one-process A/B (+4.04 %), an independent re-run
        of it (+3.94 %), and a tree-swap between the commits themselves
        -- ``cf4db73`` against ``34aea41``, one fixed harness pointed at
        each ``src`` in turn, 20 *chained* steps, 6 pairs with the first
        discarded: **+4.01 % mean, 3.6-4.7 % range**, no monotone
        settling, and the ratio holding at 4.02 % in a pair where both
        arms ran 12 % slow.  ``num_c`` is **0** in all twelve tree-swap
        runs; chaining does not raise it for this flow, so the restart
        and chained harnesses agree to 0.03 pp.

        **This supersedes the 1.4 / 9.9 / 22 % first recorded here**,
        which came from a prototype, not from these two commits (the
        commit message says so: "reproduces the *prototype's* margin").
        Measured under one harness the real ``cf4db73`` is 1279 ms/step
        and the real ``34aea41`` 1227; the recorded pair is 2745 / 2130
        -- the *before* arm inflated ``2.15x`` against the *after*
        arm's ``1.74x``.  A uniform machine slowdown scales both and
        preserves the ratio; this is asymmetric, in the direction that
        manufactures a gain.  ``cf4db73`` names the likely cause
        itself: every earlier CPU-native prototype replaced this
        method's ``shard_map`` with a bare ``moveaxis`` or an
        ``optimization_barrier``, losing that region -- which makes a
        baseline slow for a reason unrelated to the storage split.
        Its "right call, wrong reason" correction therefore did not go
        far enough: the number needed correcting too.

        The win is **step-level, not solve-level**.  The same
        interleaved A/B on ``Hk_op.solve`` alone gives ``-0.3 %`` at
        plane-Couette ``128^3``: the crop and the un-invert cost
        essentially nothing inside the sweep.  What the padded,
        reciprocated storage costs is everything *around* it -- larger
        factor arrays for XLA to place and move across a step that
        takes them as jit **arguments**.  Same lesson as the layout
        table in :func:`_banded_mode_solve`, running the other way: an
        isolated solve times this change at zero.

        (The arms agree to ``4e-17``-``2e-15`` relative per step,
        growing with the solve count -- machine epsilon rather than
        exactly, since un-inverting round-trips the diagonal through
        ``1/(1/d)``.  Do not assert exact equality across the two.)

        Why the *layout* nevertheless stays shared (three CPU-native
        layouts measured slower end to end, monotonically in how
        solve-optimal they are): :func:`_banded_mode_solve`.

        Padding here is a memory-for-memory (and time) trade: the
        persistent factors grow by the tile-roundup fraction (typically
        one ``k_z`` row -- ``Nkz = nz - 1`` is odd -- and up to
        ``bm1 - 1`` ``k_x`` columns), but no per-solve ``jnp.pad`` of
        the factors is needed: padding at solve time would re-copy them
        into a transient duplicate on every solve of every step, so
        paying once here shrinks both the step's HBM traffic and its
        transient peak.

        The padding is **per device shard** (a ``shard_map`` region,
        like the FFT pipeline): the whole-tile requirement applies to
        each device's *local* mode plane -- the plane its kernel grid
        covers -- so each local block is padded to
        ``(ceil(nkz_loc / bm0) * bm0, ceil(nkx_loc / bm1) * bm1)``,
        entirely locally (no communication, no Explicit-mesh sharding
        rules involved).  Local plane sizes are uniform across devices
        (``sharding.nz_spec`` / ``nx_spec`` are divisibility-padded to
        the mesh), so the stored global plane is well-formed:
        ``np0 * nkz_loc_pad x np1 * nkx_loc_pad`` -- the **sum of
        local roundups**, not the global roundup.  On one device
        local = global and this reduces to the plain whole-tile pad.
        Any nonzero round-up is reported once at startup (main
        process), since the padded modes cost solve work and memory.

        That ``shard_map`` is also load-bearing as a **compilation
        barrier** between the no-pivot factorisation and the
        reciprocate-and-pad above, independently of the pad it carries:
        without it the two fuse into one graph and XLA's CPU algebraic
        simplifier reports a circular simplification loop while
        building ``H_k``, turning a seconds-long setup into a
        minutes-long one.  Keep any future **kernel-storage** layout
        work inside it.

        *That is a kernel-path statement, and the CPU branch is
        measured not to need it.*  Returning early leaves only the two
        ``moveaxis`` -- neither the reciprocal scatter nor the pad the
        simplifier chokes on -- so there is nothing left for a barrier
        to separate.  Checked where the two *can* fuse at all: the
        jitted ``set_dt`` rebuild, the one place
        :func:`_factor_pallas_operator` runs inside a ``jit`` (the
        setup build cannot fuse -- :func:`_build_pallas_operator`
        host-syncs the factors for its residual/growth check before
        packing).  Its first, compiling call takes **0.95-1.5 s** at
        plane-Couette ``64 x 96 x 64`` and **1.9-2.6 s** on the pipe at
        ``128^3``; reinstating a ``shard_map`` barrier on the CPU
        branch lands inside that same spread (three interleaved
        repeats per arm, one process each, orders alternated).  No
        configuration reproduced the pathology without it.
        """
        Li = jnp.moveaxis(L, (-2, -1), (0, 1))  # (N, p, Nkz, Nkx)
        Ui = jnp.moveaxis(U, (-2, -1), (0, 1))  # (N, p+1, Nkz, Nkx)
        if not _kernel_path():
            # CPU storage: the shared layout, plain diagonal, true
            # plane.  The sweep divides by the diagonal and its grid is
            # the true plane, so both kernel transforms below would only
            # be undone again on every solve.
            return cls(L=Li, U=Ui)
        Ui = Ui.at[:, 0].set(1.0 / Ui[:, 0])  # reciprocate diagonal slot
        bm0 = params.solver.pallas_block_m0
        bm1 = params.solver.pallas_block_m1

        # Report the whole-tile round-up once per distinct geometry
        # (host-side; the pad itself happens per shard below and per
        # solve for the RHS, both inside traced regions).
        nkz_loc = Li.shape[2] // sharding.np0
        nkx_loc = Li.shape[3] // sharding.np1
        pad_kz = -nkz_loc % bm0
        pad_kx = -nkx_loc % bm1
        if pad_kz or pad_kx:
            key = (nkz_loc, nkx_loc, bm0, bm1)
            if key not in _tile_pad_reported:
                _tile_pad_reported.add(key)
                sharding.print(
                    f"Pallas mode plane: local ({nkz_loc} x "
                    f"{nkx_loc}) modes padded to ({nkz_loc + pad_kz} "
                    f"x {nkx_loc + pad_kx}) for whole ({bm0} x {bm1}) "
                    "tiles."
                )

        def _pad_local(L_l: Array, U_l: Array) -> tuple[Array, Array]:
            pad_kz = -L_l.shape[2] % bm0
            pad_kx = -L_l.shape[3] % bm1
            if not (pad_kz or pad_kx):
                return L_l, U_l
            pad = [(0, 0), (0, 0), (0, pad_kz), (0, pad_kx)]
            return jnp.pad(L_l, pad), jnp.pad(U_l, pad)

        spec = P(None, None, sharding.a0, sharding.a1)
        Li, Ui = shard_map(
            _pad_local,
            mesh=sharding.mesh,
            in_specs=(spec, spec),
            out_specs=(spec, spec),
        )(Li, Ui)
        return cls(L=Li, U=Ui)

    def solve(self, rhs: Array, component_axis: int = 0) -> Array:
        r"""Batched banded solve across ``(kz, kx)`` modes.

        ``rhs`` is **mode-inner** -- ``(N, Nkz, Nkx)`` or
        ``(C, N, Nkz, Nkx)`` for a leading batch axis ``C`` -- the
        velocity field's native spectral layout
        (``sharding.spec_scalar_shard`` / ``spec_vector_shard``); the
        result has the same shape and dtype.  This is the layout the
        Pallas kernel stores its factors in, so the GPU path feeds the
        kernel with no transpose (see :func:`_banded_mode_solve`).

        Dispatch: ``L.ndim == 5`` (batched/stacked operators) vmaps over both
        the operator and RHS leading axis; ``rhs.ndim == 4`` (shared
        operator) vmaps over the RHS leading axis; otherwise a single
        operator / single RHS.

        The whole solve is one ``shard_map`` region (mirroring the FFT
        pipeline): the per-mode systems are independent, so each device
        runs the kernel (GPU) or sweep (CPU) on its **local** mode-plane
        block with zero communication.  The component-``vmap`` dispatch
        and all tile pad/crop bookkeeping happen *inside* the body on
        local arrays, where no Explicit-mesh sharding rules apply --
        this is what makes the per-shard factor pre-padding and the
        true-plane crop legal on sharded mode axes, and what wires the
        multi-device-GPU Pallas path (real multi-GPU execution pending
        cluster validation).

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

        def _local(L_l: Array, U_l: Array, rhs_l: Array) -> Array:
            if L_l.ndim == 5:
                return jax.vmap(
                    _banded_mode_solve, in_axes=(0, 0, ca), out_axes=ca
                )(L_l, U_l, rhs_l)
            if rhs_l.ndim == 4:
                return jax.vmap(
                    _banded_mode_solve, in_axes=(None, None, ca), out_axes=ca
                )(L_l, U_l, rhs_l)
            return _banded_mode_solve(L_l, U_l, rhs_l)

        # kz/kx are the two trailing axes of every operand rank and of
        # the result, for either component_axis.
        fac_spec = P(*(None,) * (self.L.ndim - 2), sharding.a0, sharding.a1)
        rhs_spec = P(*(None,) * (rhs.ndim - 2), sharding.a0, sharding.a1)
        # ``check_vma=False``: under the default varying-mesh-axes
        # checking, ``pl.pallas_call``'s ``ShapeDtypeStruct`` out-shape
        # inside a shard_map must carry a ``manual_axis_type``
        # annotation, or tracing raises -- a GPU-only failure (the CPU
        # branch never reaches ``pallas_call``), first hit on the real
        # cluster.  The body is communication-free (independent
        # per-mode solves on local blocks), so the check guards
        # nothing here, and disabling it keeps ``_pallas_banded_solve``
        # callable both inside this region and standalone (where no
        # mesh axes exist to annotate).  Regression guard:
        # ``test_pallas_cuda_lowering_sharded_solve`` (forces the
        # kernel branch and lowers this region for cuda on CPU).
        return shard_map(
            _local,
            mesh=sharding.mesh,
            in_specs=(fac_spec, fac_spec, rhs_spec),
            out_specs=rhs_spec,
            check_vma=False,
        )(self.L, self.U, rhs)


def _stack_pallas_operators(
    *ops: PerModeBandedPallasOperator,
) -> PerModeBandedPallasOperator:
    """Stack Pallas banded operators along a leading component axis."""
    return PerModeBandedPallasOperator(
        L=jnp.stack([o.L for o in ops]),
        U=jnp.stack([o.U for o in ops]),
    )


def _pack_banded_factors(
    factors: list[tuple[Array, Array]],
) -> PerModeBandedPallasOperator:
    """Assemble factored band pairs into one Pallas operator.

    One pair becomes a plain operator; several are stacked along a
    leading component axis (:func:`_stack_pallas_operators`).  Shared
    tail of the checked :func:`_build_pallas_operator` and the
    unchecked :func:`_factor_pallas_operator`.
    """
    ops = [
        PerModeBandedPallasOperator.from_banded_factors(L, U)
        for (L, U) in factors
    ]
    return ops[0] if len(ops) == 1 else _stack_pallas_operators(*ops)


def _factor_pallas_operator(
    a_bands: list[Array],
) -> PerModeBandedPallasOperator:
    r"""Factor one operator group for the Pallas backend, unchecked.

    The jittable counterpart of :func:`_build_pallas_operator`: the
    same no-pivot banded LU (:func:`_banded_factor`) and operator
    assembly, with **no** setup-time residual/growth verification (no
    host syncs, no raise) -- so it can run inside ``jit``, e.g. the
    adaptive-``dt`` operator rebuild (the flow builders'
    ``set_dt``).

    Skipping the check is sound only when an equivalent operator was
    already verified by the checked build: the adaptive setup runs
    :func:`_build_pallas_operator` on the same band structure at
    ``step.dt_max``, where the Helmholtz diagonal
    `$1/\Delta t + c\,\nu\,k^2$` is least dominant, so the no-pivot
    element growth of every rebuild at ``dt <= dt_max`` is bounded by
    the verified case.
    """
    return _pack_banded_factors([_banded_factor(A) for A in a_bands])


def _banded_residual(a_band: Array, L: Array, U: Array) -> float:
    """Max relative residual ``||A x - b|| / ||b||`` of the no-pivot
    banded solve (``b = 1``), as a host float for the stability check."""
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


# Element-growth bound for the setup-time stability check in
# :func:`_build_pallas_operator`.  It discriminates genuine no-pivot LU
# instability (element growth, explosive when it occurs) from mere
# ill-conditioning of the operator itself (solve residual above
# tolerance with `$O(1)$` growth, e.g. the near-singular Poisson `$L_k$`
# at the smallest nonzero `$k^2$` of a huge box), where pivoting would
# change nothing fundamental.  The diagonally-dominant operators solved
# here keep growth `$O(1)$`; ``1e3`` is orders of magnitude above what
# any supported configuration produces.
_NO_PIVOT_GROWTH_TOL: float = 1e3


def _build_pallas_operator(
    a_bands: list[Array],
    label: str,
) -> PerModeBandedPallasOperator:
    r"""Factor one operator group for the Pallas backend, with checks.

    For one operator group (``a_bands`` has one banded operator, or
    several that are stacked into one homogeneous operator): factor
    each with the no-pivot banded LU, then verify the group once at
    setup:

    - LU element growth ``max|U| / max|A|`` above
      :data:`_NO_PIVOT_GROWTH_TOL`, or any non-finite factor or
      residual, means the no-pivot factorisation itself is unstable:
      hard ``RuntimeError`` (use ``solver.backend = "dense"``, the
      pivoted reference, or revisit the wall-normal grid).
    - A solve residual above ``params.solver.pallas_stability_tol``
      with benign growth indicates an ill-conditioned operator, not an
      unstable factorisation; pivoting would not help, so the build
      proceeds after printing a notice.

    A ``[pallas] {label}: ...`` line with the measured residual and
    growth is printed at setup for the group either way.

    Parameters
    ----------
    a_bands:
        Banded operators ``(Nkz, Nkx, N, 2p+1)`` forming the group.
    label:
        Operator-group name for the diagnostic (e.g. ``"Hk"``).
    """
    factors = [_banded_factor(A) for A in a_bands]
    resid = max(
        _banded_residual(A, L, U)
        for A, (L, U) in zip(a_bands, factors, strict=True)
    )
    growth = max(
        float(jnp.max(jnp.abs(U)) / jnp.max(jnp.abs(A)))
        for A, (_, U) in zip(a_bands, factors, strict=True)
    )

    # ``not (growth <= tol)`` also catches non-finite growth from a
    # no-pivot breakdown (``nan > tol`` would be False).
    if not (growth <= _NO_PIVOT_GROWTH_TOL) or not math.isfinite(resid):
        raise RuntimeError(
            f"[pallas] {label}: no-pivot banded LU is unstable "
            f"(element growth {growth:.1e}, residual {resid:.2e}): "
            "use solver.backend='dense' (the pivoted reference) or "
            "revisit the wall-normal grid/resolution."
        )
    tol = params.solver.pallas_stability_tol
    if resid > tol:
        sharding.print(
            f"[pallas] {label}: residual {resid:.2e} > tol {tol:.0e}, "
            f"growth {growth:.1e} benign -> ill-conditioned operator, "
            "not LU instability; proceeding (raise "
            "solver.pallas_stability_tol to silence)"
        )
    else:
        sharding.print(
            f"[pallas] {label}: no-pivot banded LU "
            f"(residual {resid:.2e}, growth {growth:.1e})"
        )
    return _pack_banded_factors(factors)
