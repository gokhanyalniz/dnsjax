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
from functools import partial

import jax
import jax.scipy.linalg as sla
import numpy as np
from jax import Array, lax
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
        self.kx = jax.device_put(
            jnp.asarray(kx_vals, dtype=sharding.float_type).reshape(
                [1, -1, 1]
            ),
            sharding.spec_scalar_shard,
        )
        kz_vals = complex_harmonics(params.res.nz) * 2 * jnp.pi / params.geo.lz
        self.kz = jax.device_put(
            jnp.asarray(kz_vals, dtype=sharding.float_type).reshape(
                [-1, 1, 1]
            ),
            sharding.no_shard,
        )

        self.k_metric = jax.device_put(
            jnp.where(self.kx == 0, 1, 2).astype(sharding.float_type),
            sharding.spec_scalar_shard,
        )

        self.k2 = jax.device_put(
            self.kx**2 + self.kz**2,
            sharding.spec_scalar_shard,
        )
        self.k2_is_zero = jax.device_put(
            self.k2 == 0.0,
            sharding.spec_scalar_shard,
        )


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


def _banded_solve_device(
    ab: Array, ipiv: Array, b: Array, kl: int, ku: int
) -> Array:
    """Device-side solve of `$A x = b$` from LAPACK-packed LU factors.

    Equivalent to ``dgbtrs`` for ``trans='N'``.  Forward elimination
    applies the partial-pivoting row swaps stored in ``ipiv`` and
    subtracts the ``kl`` sub-diagonal contributions from the next
    ``kl`` entries of ``b``.  Back substitution divides by the
    diagonal and subtracts the ``kuu = kl + ku`` super-diagonal
    contributions from the preceding rows.  Two ``lax.scan``s keep
    the implementation JIT- and vmap-friendly.

    Padding strategy: ``b`` is padded with ``kl`` zero entries after
    (for forward elim) and ``kuu`` zero entries before (for back
    sub) so that the per-step ``dynamic_slice_in_dim`` updates are
    always of fixed width.  Pad-region writes are harmless because
    the padding is stripped at the end.

    Parameters
    ----------
    ab:
        Packed LU factors for one mode, shape ``(2*kl + ku + 1,
        Ny)``.
    ipiv:
        Pivot indices for one mode, shape ``(Ny,)``, int32, 0-based.
    b:
        Right-hand side for one mode, shape ``(Ny,)``, real.
    kl, ku:
        Band widths (static).

    Returns
    -------
    :
        Solution ``x`` with the same shape and dtype as ``b``.
    """
    Ny = ab.shape[-1]
    kuu = kl + ku

    # Hoist the sub- and super-diagonal slabs out of the scan bodies:
    # each column j then reduces to a single `lower[:, j]` / `upper[:, j]`
    # index op instead of `ab[:, j]` plus an inner `dynamic_slice_in_dim`.
    lower = ab[kl + ku + 1 :, :]  # shape (kl, Ny), sub-diagonal L entries
    upper = ab[:kuu, :]  # shape (kuu, Ny), super-diagonal U entries
    diag = ab[kl + ku, :]  # shape (Ny,), U pivots

    # Forward elimination with row padding at the end.
    b = jnp.concatenate([b, jnp.zeros(kl, dtype=b.dtype)], axis=-1)

    def fwd_step(b_carry: Array, j: Array) -> tuple[Array, None]:
        pj = ipiv[j]
        bj = b_carry[j]
        bpj = b_carry[pj]
        b_carry = b_carry.at[j].set(bpj)
        b_carry = b_carry.at[pj].set(bj)
        lams = lower[:, j]
        block = lax.dynamic_slice_in_dim(b_carry, j + 1, kl, axis=-1)
        block = block - lams * bpj
        b_carry = lax.dynamic_update_slice_in_dim(
            b_carry, block, j + 1, axis=-1
        )
        return b_carry, None

    b, _ = lax.scan(fwd_step, b, jnp.arange(Ny - 1))
    b = b[:Ny]

    # Back substitution with row padding at the front.
    b = jnp.concatenate([jnp.zeros(kuu, dtype=b.dtype), b], axis=-1)

    def bwd_step(b_carry: Array, j_rev: Array) -> tuple[Array, None]:
        j = Ny - 1 - j_rev
        j_padded = j + kuu
        xj = b_carry[j_padded] / diag[j]
        b_carry = b_carry.at[j_padded].set(xj)
        ups = upper[:, j]
        block = lax.dynamic_slice_in_dim(b_carry, j_padded - kuu, kuu, axis=-1)
        block = block - ups * xj
        b_carry = lax.dynamic_update_slice_in_dim(
            b_carry, block, j_padded - kuu, axis=-1
        )
        return b_carry, None

    b, _ = lax.scan(bwd_step, b, jnp.arange(Ny))
    return b[kuu:]


@register_dataclass_pytree
@dataclass
class PerModeBandedOperator:
    """Per-mode banded LU cache in LAPACK ``ab`` packing.

    Stores the partial-pivoted LU factors produced on the GPU by
    :func:`_banded_lu_factor_batched` (one matrix per Fourier mode
    pair ``(kz, kx)``) alongside the pivot indices.  The bandwidth
    is derived from the ab-row count under the invariant
    ``ab.shape[-2] == 3*p + 1`` with ``kl == ku == p``.

    ``solve`` forwards the RHS (real or complex) directly to
    :func:`_banded_solve_device`.  Because the scan body only uses
    real-factor × RHS multiplications and RHS / real divisions, a
    complex RHS flows through under JAX's mixed-type arithmetic —
    no real/imag split is needed.  The real-precision LU factors
    retain their half-memory advantage over complex-typed factors.
    """

    ab: Array
    ipiv: Array

    def solve(self, rhs: Array) -> Array:
        """Batched banded solve across ``(kz, kx)`` modes.

        A leading batch axis (e.g. the 3 velocity components) is
        supported transparently by an extra ``vmap`` that leaves the
        packed LU factors untouched, so the same ``ab`` / ``ipiv``
        are reused across all batched RHSs.

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(Nkz, Nkx, Ny)`` or
            ``(C, Nkz, Nkx, Ny)`` for a leading batch axis ``C``.
            May be real or complex; the dtype is preserved.

        Returns
        -------
        :
            Solution array, same shape and dtype as *rhs*.
        """
        p = (self.ab.shape[-2] - 1) // 3
        kl = ku = p

        def solve_one(ab: Array, ipiv: Array, b: Array) -> Array:
            return _banded_solve_device(ab, ipiv, b, kl, ku)

        per_mode = jax.vmap(jax.vmap(solve_one))
        if rhs.ndim == 4:
            return jax.vmap(per_mode, in_axes=(None, None, 0))(
                self.ab, self.ipiv, rhs
            )
        return per_mode(self.ab, self.ipiv, rhs)


# ── GPU IMM operator builders (direct ab packing, no dense matrix) ──


def _ab_index_map(Ny: int, p: int) -> tuple[Array, Array, Array, Array, Array]:
    """Precompute the `(r, j) -> dense i` index map for ab packing.

    With ``kl = ku = p``, the LAPACK banded ``ab`` format stores
    entry `$A_{i,j}$` at position ``ab[kl + ku + i - j, j]``.  For a
    given ab position ``(r, j)``, the corresponding dense row is
    ``i = r + j - (kl + ku)``.  This helper builds that inverse
    map (shape ``(3p + 1, Ny)``) plus the `$i$`-range mask and its
    clipped form for safe advanced indexing.

    Returns
    -------
    r_idx:
        Column vector of ab row indices, shape ``(3p + 1, 1)``.
    j_idx:
        Row vector of dense column indices, shape ``(1, Ny)``.
    i_idx:
        Dense row indices `$i = r + j - (k_l + k_u)$`, shape
        ``(3p + 1, Ny)``.  Can be negative or `$\\ge N_y$`; use
        ``in_range`` to mask.
    in_range:
        Boolean mask ``(i_idx >= 0) & (i_idx < Ny)``.
    i_clip:
        ``i_idx`` clipped to ``[0, Ny - 1]``, safe to feed into
        advanced indexing into a ``(Ny, Ny)`` matrix.
    """
    kl = ku = p
    ab_rows = 2 * kl + ku + 1
    r_idx = jnp.arange(ab_rows)[:, None]
    j_idx = jnp.arange(Ny)[None, :]
    i_idx = r_idx + j_idx - (kl + ku)
    in_range = (i_idx >= 0) & (i_idx < Ny)
    i_clip = jnp.clip(i_idx, 0, Ny - 1)
    return r_idx, j_idx, i_idx, in_range, i_clip


def _build_Lk_ab_gpu(
    D1: Array, D2: Array, k2: Array, k2_is_zero: Array, p: int
) -> Array:
    """Build `$L_k$` directly in LAPACK ab-packed form on the GPU.

    `$L_k$` is the Neumann-modified Laplacian
    `$D_2 - k^2 I$` with the wall rows replaced by `$D_1$` rows to
    encode Neumann boundary conditions, except that the `$k^2 = 0$`
    mean mode pins `$p_0 = 0$` by setting row 0 to
    `$(1, 0, \\dots, 0)$` (the pure-Neumann problem is singular).

    No dense `$(N_y, N_y)$` or `$(N_{kz}, N_{kx}, N_y, N_y)$`
    intermediate is materialised: each ``ab`` entry is computed
    directly from `$D_1$`, `$D_2$`, and `$k^2$` via a precomputed
    `$(r, j) \\to i$` index map.

    Parameters
    ----------
    D1:
        First-derivative matrix, shape ``(Ny, Ny)``.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    k2:
        Squared horizontal wavenumber, shape ``(Nkz, Nkx, 1)``.
    k2_is_zero:
        Mean-mode boolean mask, shape ``(Nkz, Nkx, 1)``.
    p:
        Finite-difference order (``kl = ku = p``).

    Returns
    -------
    :
        LAPACK ab-packed `$L_k$`, shape ``(Nkz, Nkx, 3p + 1, Ny)``,
        kx-sharded (axis 1).
    """
    Ny = D2.shape[-1]
    kl = ku = p
    r_idx, j_idx, i_idx, in_range, i_clip = _ab_index_map(Ny, p)

    # Interior contribution: Lk[i, j] = D2[i, j] - k2 * delta_{i, j}.
    j_bcast = jnp.broadcast_to(j_idx, i_idx.shape)
    D2_ab = jnp.where(in_range, D2[i_clip, j_bcast], 0.0)
    is_main_diag = (r_idx == kl + ku).astype(D2.dtype)

    # Shape plan:
    # - D2_ab:        (3p+1, Ny)       -> [None, None, :, :]
    # - k2:           (Nkz, Nkx, 1)    -> [..., None] => (Nkz, Nkx, 1, 1)
    # - is_main_diag: (3p+1, 1)        -> [None, None, :, :]
    ab_interior = (
        D2_ab[None, None, :, :]
        - k2[..., None] * is_main_diag[None, None, :, :]
    )

    # Boundary-row overwrites: dense row 0 and Ny-1 are not D2-based.
    is_row_0 = i_idx == 0  # (3p+1, Ny)
    is_row_Nm1 = i_idx == (Ny - 1)  # (3p+1, Ny)

    D1_row_0 = D1[0, j_bcast]  # (3p+1, Ny) — D1[0, :] broadcast over rows
    D1_row_Nm1 = D1[-1, j_bcast]
    pin_row = (j_bcast == 0).astype(D2.dtype)  # (3p+1, Ny)

    row0_val = jnp.where(
        k2_is_zero[..., None],
        pin_row[None, None, :, :],
        D1_row_0[None, None, :, :],
    )

    ab = jnp.where(is_row_0[None, None, :, :], row0_val, ab_interior)
    ab = jnp.where(
        is_row_Nm1[None, None, :, :],
        D1_row_Nm1[None, None, :, :],
        ab,
    )

    return jax.lax.with_sharding_constraint(ab, sharding.spec_dy_op_shard)


def _build_Hk_ab_gpu(
    D2: Array, k2: Array, dt: float, c: float, nu: float, p: int
) -> Array:
    """Build `$H_k$` directly in LAPACK ab-packed form on the GPU.

    `$H_k = (1/\\Delta t) I - c \\nu (D_2 - k^2 I)$` with identity
    boundary rows (no-slip Dirichlet).  No dense intermediate.

    Parameters
    ----------
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    k2:
        Squared horizontal wavenumber, shape ``(Nkz, Nkx, 1)``.
    dt:
        Time step.
    c:
        Implicitness parameter (0.5 = Crank-Nicolson).
    nu:
        Kinematic viscosity `$1/\\mathrm{Re}$`.
    p:
        Finite-difference order (``kl = ku = p``).

    Returns
    -------
    :
        LAPACK ab-packed `$H_k$`, shape ``(Nkz, Nkx, 3p + 1, Ny)``,
        kx-sharded (axis 1).
    """
    Ny = D2.shape[-1]
    kl = ku = p
    r_idx, j_idx, i_idx, in_range, i_clip = _ab_index_map(Ny, p)

    # Interior: Hk[i, j] = (1/dt + c*nu*k2) delta_{i,j} - c*nu*D2[i, j].
    j_bcast = jnp.broadcast_to(j_idx, i_idx.shape)
    D2_ab = jnp.where(in_range, D2[i_clip, j_bcast], 0.0)
    is_main_diag = (r_idx == kl + ku).astype(D2.dtype)

    diag_scalar = (1.0 / dt) + c * nu * k2  # (Nkz, Nkx, 1)
    ab_interior = (
        diag_scalar[..., None] * is_main_diag[None, None, :, :]
        - c * nu * D2_ab[None, None, :, :]
    )

    # Boundary-row overwrites: identity rows.  Ab entry at dense
    # (i, j) is stored at (r, j) with r = kl+ku+i-j; the only
    # nonzero entries for identity rows are the main diagonal.
    is_row_0 = i_idx == 0
    is_row_Nm1 = i_idx == (Ny - 1)
    identity_row_0 = (j_bcast == 0).astype(D2.dtype)
    identity_row_Nm1 = (j_bcast == (Ny - 1)).astype(D2.dtype)

    ab = jnp.where(
        is_row_0[None, None, :, :],
        identity_row_0[None, None, :, :],
        ab_interior,
    )
    ab = jnp.where(
        is_row_Nm1[None, None, :, :],
        identity_row_Nm1[None, None, :, :],
        ab,
    )

    return jax.lax.with_sharding_constraint(ab, sharding.spec_dy_op_shard)


def _build_Lk_dense_gpu(
    D1: Array, D2: Array, k2: Array, k2_is_zero: Array
) -> Array:
    """Build the Neumann-BC Laplacian `$L_k$` in dense form on GPU.

    Used only by the ``"dense"`` solver backend; allocates
    `$(N_{kz}, N_{kx}, N_y, N_y)$`.  No CPU path.

    Parameters / returns follow :func:`_build_Lk_ab_gpu`, but the
    output is the full dense operator.
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

    return jax.lax.with_sharding_constraint(Lk, sharding.spec_dy_op_shard)


def _build_Hk_dense_gpu(
    D2: Array, k2: Array, dt: float, c: float, nu: float
) -> tuple[Array, Array]:
    """Build dense `$H_k$` and `$H_k^-$` on GPU (dense backend only).

    Returns both the implicit operator
    `$H_k = (1/\\Delta t) I - c \\nu (D_2 - k^2 I)$` and the
    explicit one `$H_k^- = (1/\\Delta t) I + (1-c) \\nu (D_2 - k^2 I)$`,
    each with identity wall rows for no-slip Dirichlet BCs.
    """
    Ny = D2.shape[-1]
    eye = jnp.eye(Ny, dtype=D2.dtype)
    Lk_raw = D2[None, None, :, :] - k2[..., None] * eye

    Hk = (1.0 / dt) * eye - c * nu * Lk_raw
    Hk_minus = (1.0 / dt) * eye + (1.0 - c) * nu * Lk_raw

    # Dirichlet identity rows.
    zero_row = jnp.zeros(Ny, dtype=D2.dtype)
    e_0 = zero_row.at[0].set(1.0)
    e_Nm1 = zero_row.at[-1].set(1.0)
    Hk = Hk.at[..., 0, :].set(e_0).at[..., -1, :].set(e_Nm1)
    Hk_minus = Hk_minus.at[..., 0, :].set(e_0).at[..., -1, :].set(e_Nm1)

    Hk = jax.lax.with_sharding_constraint(Hk, sharding.spec_dy_op_shard)
    Hk_minus = jax.lax.with_sharding_constraint(
        Hk_minus, sharding.spec_dy_op_shard
    )
    return Hk, Hk_minus


# ── GPU banded LU factorisation (LAPACK-compatible) ──────────────────


def _banded_lu_factor_mode(ab: Array, kl: int, ku: int) -> tuple[Array, Array]:
    """Single-mode banded LU with partial pivoting (LAPACK-compatible).

    Equivalent to LAPACK ``dgbtrf`` on one `$N_y \\times N_y$`
    matrix in LAPACK ab packing.  The factorisation runs as a
    ``lax.scan`` of length `$N_y$`; at each column *j* a small
    `$(k_l + 1, k_l + k_u + 1)$` dense *window* is extracted from a
    `$k_l + k_u + 1$`-wide dynamic slice of ``ab``, partial-pivoted,
    row-swapped, and Gauss-eliminated, then written back.

    The window corresponds to dense rows `$[j, j + k_l]$` and dense
    columns `$[j, j + k_l + k_u]$` — exactly the entries touched by
    one step of ``dgbtrf``.  Because the LAPACK packing stores
    dense row *i* along the diagonal ``ab[kl + ku + i - j, j]``,
    the window is a parallelogram inside ``ab_window``, and is
    reshaped into a rectangle via the compile-time index map
    ``W[r, c] = ab_window[kl + ku - c + r, c]``.

    To keep the dynamic slice width fixed near the right edge of
    ``ab``, the input is zero-padded by ``kl + ku`` columns on the
    right before the scan and trimmed back afterwards.  Those
    padding columns can absorb harmless writes from the last
    ``kl + ku`` steps because dense rows beyond `$N_y - 1$` are
    zero by construction, so the pivot stays on the true diagonal
    and the elimination is a no-op there.

    Parameters
    ----------
    ab:
        Input banded matrix in LAPACK ab packing, shape
        ``(2 kl + ku + 1, Ny)``.
    kl, ku:
        Lower and upper bandwidths (``kl == ku == p`` in our use).

    Returns
    -------
    ab:
        LU factors in LAPACK ab packing (same shape and layout as
        the input), with ``L`` below and ``U`` on and above the
        diagonal.
    ipiv:
        0-based pivot indices, shape ``(Ny,)`` int32.  At step
        ``j``, dense row ``j`` was swapped with dense row
        ``ipiv[j]``; ``ipiv[j] >= j`` always.
    """
    Ny = ab.shape[-1]
    window_cols = kl + ku + 1
    window_rows = kl + 1
    r_win = jnp.arange(window_rows)[:, None]
    c_win = jnp.arange(window_cols)[None, :]
    win_ab_row = kl + ku - c_win + r_win
    win_c = jnp.broadcast_to(c_win, win_ab_row.shape)

    ab = jnp.pad(ab, [(0, 0), (0, kl + ku)])
    ipiv = jnp.zeros(Ny, dtype=jnp.int32)

    def step(
        carry: tuple[Array, Array], j: Array
    ) -> tuple[tuple[Array, Array], None]:
        ab, ipiv = carry
        ab_win = lax.dynamic_slice_in_dim(ab, j, window_cols, axis=-1)
        # Rectangular view of the (kl+1) x (kl+ku+1) parallelogram.
        W = ab_win[win_ab_row, win_c]

        # Partial pivot on column 0 of W (dense column j).
        r_best = jnp.argmax(jnp.abs(W[:, 0])).astype(jnp.int32)
        ipiv = ipiv.at[j].set(j.astype(jnp.int32) + r_best)

        # Swap dense rows j and j + r_best inside the window.
        row_0 = W[0]
        row_best = W[r_best]
        W = W.at[0].set(row_best).at[r_best].set(row_0)

        # Multipliers, then eliminate the sub-diagonal of column 0
        # across the remaining window columns.
        pivot = W[0, 0]
        mult = W[1:, 0] / pivot
        W = W.at[1:, 0].set(mult)
        W = W.at[1:, 1:].add(-mult[:, None] * W[0, 1:][None, :])

        ab_win = ab_win.at[win_ab_row, win_c].set(W)
        ab = lax.dynamic_update_slice_in_dim(ab, ab_win, j, axis=-1)
        return (ab, ipiv), None

    (ab, ipiv), _ = lax.scan(step, (ab, ipiv), jnp.arange(Ny))
    return ab[:, :Ny], ipiv


@partial(jax.jit, static_argnames=("kl", "ku"))
def _banded_lu_factor_batched(
    ab: Array, kl: int, ku: int
) -> tuple[Array, Array]:
    """Batch :func:`_banded_lu_factor_mode` over `$(N_{kz}, N_{kx})$`.

    ``jax.vmap(jax.vmap(...))`` preserves the kx-sharding of the
    input ab on axis 1, so each device factorises only its local
    modes.
    """
    factor_one = lambda a: _banded_lu_factor_mode(a, kl, ku)  # noqa: E731
    return jax.vmap(jax.vmap(factor_one))(ab)


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
    # ``Lk`` and ``Hk_minus`` are only populated under
    # ``params.solver.backend == "dense"`` for the matrix-free matvecs
    # at :func:`_imm_iteration`.  Under ``"banded"`` they stay ``None``
    # and the matvecs are computed on the fly from ``D2``.
    Lk: Array | None = field(init=False, default=None)
    Hk_minus: Array | None = field(init=False, default=None)
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
        the wall-normal coordinate ``y`` in ``[-1, 1]``, FD
        matrices ``D1`` and ``D2``, and all per-mode IMM
        operators directly on the device.  No CPU LU or chunked
        host->device copy is involved: under the banded backend,
        ``Lk`` and ``Hk`` are assembled straight into LAPACK
        ab-packed form via :func:`_build_Lk_ab_gpu` /
        :func:`_build_Hk_ab_gpu` and factorised by
        :func:`_banded_lu_factor_batched`; under the dense backend
        they are built as full `$(N_y, N_y)$` blocks via
        :func:`_build_Lk_dense_gpu` / :func:`_build_Hk_dense_gpu`
        and factorised by :class:`DenseJAXSolver`.  Homogeneous
        IMM data (``p1..q2``, ``M_inv``) is derived from the GPU
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
        self.D1 = jax.device_put(jnp.asarray(D1), sharding.no_shard)
        self.D2 = jax.device_put(jnp.asarray(D2), sharding.no_shard)
        self.D1_bnd = jax.device_put(
            jnp.asarray(D1[[0, -1], :]), sharding.no_shard
        )
        self.D2_bnd = jax.device_put(
            jnp.asarray(D2[[0, -1], :]), sharding.no_shard
        )

        Nkz = params.res.nz - 1
        Nkx = params.res.nx // 2
        Ny = params.res.ny

        p = params.res.fd_order
        dt = params.step.dt
        c = params.step.implicitness
        nu = 1.0 / params.phys.re

        if params.solver.backend == "banded":
            # Build Lk, Hk directly in LAPACK ab packing (no dense
            # `(Nkz, Nkx, Ny, Ny)` intermediate anywhere) and factorise
            # with the pure-JAX banded LU scan.
            Lk_ab = _build_Lk_ab_gpu(
                self.D1, self.D2, fourier.k2, fourier.k2_is_zero, p
            )
            Hk_ab = _build_Hk_ab_gpu(self.D2, fourier.k2, dt, c, nu, p)
            Lk_ab, Lk_piv = _banded_lu_factor_batched(Lk_ab, p, p)
            Hk_ab, Hk_piv = _banded_lu_factor_batched(Hk_ab, p, p)
            self.Lk_op = PerModeBandedOperator(ab=Lk_ab, ipiv=Lk_piv)
            self.Hk_op = PerModeBandedOperator(ab=Hk_ab, ipiv=Hk_piv)
        else:
            # Dense backend: parity/reference path.  The full
            # `(Nkz, Nkx, Ny, Ny)` matrices only ever materialise
            # under this branch.
            Lk_dense = _build_Lk_dense_gpu(
                self.D1, self.D2, fourier.k2, fourier.k2_is_zero
            )
            Hk_dense, Hk_minus_dense = _build_Hk_dense_gpu(
                self.D2, fourier.k2, dt, c, nu
            )
            self.Lk = Lk_dense
            self.Hk_minus = Hk_minus_dense
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
        velocity_phys = jax.device_put(
            snapshot_arr, sharding.phys_vector_shard
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


def _matvec_4d(op: Array, x: Array) -> Array:
    """Apply operator of shape (Nkz, Nkx, Ny, Ny)
    to field of shape (Nkz, Nkx, Ny).
    """
    return jnp.einsum("zxij, zxj -> zxi", op, x)


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

    # d_hat^n (discrete divergence at time n; ~0 after first step).
    dy_v_n = jnp.einsum("ij, zxj -> zxi", flow_.D1, v_n)
    d_hat_n = ikx * u_n + dy_v_n + ikz * w_n

    # Stage 1: interior pressure Poisson RHS.
    dy_Nv_j = jnp.einsum("ij, zxj -> zxi", flow_.D1, Nv_j)
    dy_Nv_n = jnp.einsum("ij, zxj -> zxi", flow_.D1, Nv_n)
    div_Nj = ikx * Nu_j + dy_Nv_j + ikz * Nw_j
    div_Nn = ikx * Nu_n + dy_Nv_n + ikz * Nw_n

    if params.solver.backend == "banded":
        Lk_d = _lk_matvec(d_hat_n, flow_.D2, flow_.D1_bnd, k2, k2_is_zero)
    else:
        Lk_d = _matvec_4d(flow_.Lk, d_hat_n)

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

    if params.solver.backend == "banded":
        Hk_minus_stack = jax.vmap(
            _hk_minus_matvec,
            in_axes=(0, None, None, None, None, None),
        )(velocity_n, flow_.D2, k2, dt, c, nu)
    else:
        Hk_minus_stack = jax.vmap(_matvec_4d, in_axes=(None, 0))(
            flow_.Hk_minus, velocity_n
        )

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
]:
    """Build time-stepping functions for a Cartesian wall-bounded flow.

    Returns ``(predict_and_correct, iterate_correction, init_state_bound)``
    with the ``fourier`` and *flow* singletons already bound.
    """
    _predict_and_correct_jit, _iterate_correction_jit = make_stepper(
        _get_rhs, _predict, _correct, _norm
    )

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

    def init_state_bound(snapshot: str | None) -> Array:
        """Initialize the flow state."""
        return init_state(snapshot)

    return predict_and_correct, iterate_correction, init_state_bound
