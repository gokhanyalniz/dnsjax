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

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.scipy.linalg as sla
import numpy as np
from jax import Array, jit, lax
from jax import numpy as jnp
from scipy.linalg.lapack import get_lapack_funcs

from ..bench import timer
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

    kx_global: np.ndarray = field(init=False)
    kx: Array = field(init=False)
    kz_global: np.ndarray = field(init=False)
    kz: Array = field(init=False)
    k_metric: Array = field(init=False)
    k2: Array = field(init=False)
    k2_is_zero: Array = field(init=False)

    def __post_init__(self) -> None:
        self.kx_global = np.array(
            real_harmonics(params.res.nx) * 2 * jnp.pi / params.geo.lx
        )
        self.kx = jax.device_put(
            jnp.array(self.kx_global).reshape([1, -1, 1]),
            sharding.spec_scalar_shard,
        )
        self.kz_global = np.array(
            complex_harmonics(params.res.nz) * 2 * jnp.pi / params.geo.lz
        )
        self.kz = jax.device_put(
            jnp.array(self.kz_global).reshape([-1, 1, 1]),
            sharding.no_shard,
        )

        self.k_metric = jax.device_put(
            jnp.where(self.kx == 0, 1, 2),
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


class IMMChunker:
    """Chunked IMM precomputation for ``make_array_from_callback``.

    Lazily computes and caches per-shard IMM operator blocks
    so that only the local ``(kz, kx)`` slice is built on each
    device.
    """

    def __init__(
        self,
        ys_arr: np.ndarray,
        kx_global: np.ndarray,
        kz_global: np.ndarray,
        p: int,
        dt: float,
        c: float,
        nu: float,
        D1_arr: np.ndarray,
        D2_arr: np.ndarray,
        backend: str = "banded",
    ) -> None:
        """
        Parameters
        ----------
        ys_arr:
            Wall-normal grid points, shape ``(Ny,)``.
        kx_global:
            All streamwise wavenumbers, shape ``(Nkx,)``.
        kz_global:
            All spanwise wavenumbers, shape ``(Nkz,)``.
        p:
            Finite-difference accuracy order.
        dt:
            Time step.
        c:
            Implicitness parameter (0.5 for CN).
        nu:
            Kinematic viscosity ``1/Re``.
        D1_arr:
            First-derivative matrix, shape ``(Ny, Ny)``.
        D2_arr:
            Second-derivative matrix, shape ``(Ny, Ny)``.
        backend:
            Either ``"banded"`` (default) or ``"dense"``.  Controls
            whether ``precompute_imm`` emits LAPACK-packed banded LU
            factors or full dense ``Lk``, ``Hk``, ``Hk_minus``.
        """
        self.ys_arr = ys_arr
        self.kx_global = kx_global
        self.kz_global = kz_global
        self.p = p
        self.dt = dt
        self.c = c
        self.nu = nu
        self.D1_arr = D1_arr
        self.D2_arr = D2_arr
        self.backend = backend
        self.cache: dict[tuple, dict[str, np.ndarray]] = {}

    def get_chunk(self, indices: tuple[slice, ...], key: str) -> np.ndarray:
        """Return one shard of the precomputed operator *key*.

        Called by ``jax.make_array_from_callback``.
        *indices* contains per-axis slices identifying the
        shard.  Results are cached so repeated calls for
        different keys reuse the same precomputation.

        Parameters
        ----------
        indices:
            Per-axis shard slices from JAX.
            ``indices[0]`` slices axis 0 (kz),
            ``indices[1]`` slices axis 1 (kx).
        key:
            Operator name (e.g. ``"Lk_ab"``, ``"Hk_piv"``,
            ``"p1"``, ``"M_inv"``; under ``backend="dense"``
            also ``"Lk"``, ``"Hk"``, ``"Hk_minus"``).
        """
        slice_kz, slice_kx = indices[0], indices[1]
        cache_key = (
            slice_kz.start,
            slice_kz.stop,
            slice_kx.start,
            slice_kx.stop,
        )
        if cache_key not in self.cache:
            self.cache[cache_key] = precompute_imm(
                y=self.ys_arr,
                kx_vals=self.kx_global[slice_kx],
                kz_vals=self.kz_global[slice_kz],
                p=self.p,
                dt=self.dt,
                c=self.c,
                nu=self.nu,
                D1=self.D1_arr,
                D2=self.D2_arr,
                backend=self.backend,
            )
        return self.cache[cache_key][key]


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

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(Nkz, Nkx, Ny)``.

        Returns
        -------
        :
            Solution array, same shape as *rhs*.
        """
        return _lu_solve((self.lu, self.piv), rhs)


def _pack_banded(A: np.ndarray, kl: int, ku: int) -> np.ndarray:
    """Pack a dense `$(N_y, N_y)$` matrix into LAPACK ``ab`` format.

    Output shape ``(2*kl + ku + 1, Ny)``, suitable for ``dgbtrf``.
    The leading ``kl`` rows are reserved for fill-in during the
    LU factorisation and are left zero on input.  Within the
    remaining ``kl + ku + 1`` rows, entry `$A_{ij}$` is stored at
    ``ab[kl + ku + i - j, j]`` for
    `$\\max(0, j-k_u) \\le i \\le \\min(N_y-1, j+k_l)$`.

    Parameters
    ----------
    A:
        Dense banded matrix, shape ``(Ny, Ny)``.
    kl, ku:
        Lower and upper bandwidths.

    Returns
    -------
    :
        LAPACK-ab-packed matrix, shape ``(2*kl + ku + 1, Ny)``.
    """
    Ny = A.shape[0]
    ab = np.zeros((2 * kl + ku + 1, Ny), dtype=A.dtype)
    for j in range(Ny):
        i_lo = max(0, j - ku)
        i_hi = min(Ny - 1, j + kl)
        for i in range(i_lo, i_hi + 1):
            ab[kl + ku + i - j, j] = A[i, j]
    return ab


def _banded_lu_numpy(
    A: np.ndarray, kl: int, ku: int
) -> tuple[np.ndarray, np.ndarray]:
    """Factor a dense banded matrix via LAPACK ``dgbtrf`` (CPU-only).

    Partial pivoting is applied.  SciPy's LAPACK wrapper returns
    0-based pivot indices (despite Fortran convention), which is what
    our device-side scan expects.

    Parameters
    ----------
    A:
        Dense banded matrix, shape ``(Ny, Ny)``.
    kl, ku:
        Lower and upper bandwidths.

    Returns
    -------
    ab:
        Packed LU factors, shape ``(2*kl + ku + 1, Ny)``.  `L` and
        `U` share the packed storage; the first ``kl`` rows hold
        fill-in produced by pivoting.
    ipiv:
        Pivot indices, shape ``(Ny,)``, int32, 0-based.  At step
        ``j``, row ``j`` was swapped with row ``ipiv[j]``.
    """
    ab = _pack_banded(A, kl, ku)
    (gbtrf,) = get_lapack_funcs(("gbtrf",), (ab,))
    lu, piv, info = gbtrf(ab, kl, ku, overwrite_ab=True)
    if info != 0:
        raise ValueError(f"dgbtrf failed with info={info}")
    # SciPy's LAPACK wrapper returns 0-based pivot indices already
    # (despite the Fortran-level routine using 1-based indexing), so
    # no conversion is needed.
    return lu, piv.astype(np.int32)


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

    # Forward elimination with row padding at the end.
    b = jnp.concatenate([b, jnp.zeros(kl, dtype=b.dtype)], axis=-1)

    def fwd_step(b_carry: Array, j: Array) -> tuple[Array, None]:
        pj = ipiv[j]
        bj = b_carry[j]
        bpj = b_carry[pj]
        b_carry = b_carry.at[j].set(bpj)
        b_carry = b_carry.at[pj].set(bj)
        # Sub-diagonal L entries for column j live at
        # ab[kl + ku + 1 : kl + ku + 1 + kl, j]
        lams = lax.dynamic_slice_in_dim(ab[:, j], kl + ku + 1, kl, axis=-1)
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
        diag = ab[kl + ku, j]
        xj = b_carry[j_padded] / diag
        b_carry = b_carry.at[j_padded].set(xj)
        # Super-diagonal U entries for column j span
        # ab[0 : kuu, j], in order of increasing i in [j-kuu, j-1].
        ups = lax.dynamic_slice_in_dim(ab[:, j], 0, kuu, axis=-1)
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

    Stores the partial-pivoted LU factors produced on the CPU by
    ``dgbtrf`` (one matrix per Fourier mode pair ``(kz, kx)``) on
    device, alongside the pivot indices.  The bandwidth is derived
    from the ab-row count under the invariant
    ``ab.shape[-2] == 3*p + 1`` with ``kl == ku == p``.

    The ``solve`` method splits complex RHS into real/imag halves,
    runs two real banded solves, and recombines — this keeps the
    LU factors in real float precision (half the memory of complex).
    """

    ab: Array
    ipiv: Array

    def solve(self, rhs: Array) -> Array:
        """Batched banded solve across ``(kz, kx)`` modes.

        Parameters
        ----------
        rhs:
            Right-hand side, shape ``(Nkz, Nkx, Ny)``.  May be
            real or complex.

        Returns
        -------
        :
            Solution array, same shape and dtype as *rhs*.
        """
        p = (self.ab.shape[-2] - 1) // 3
        kl = ku = p

        def solve_one(ab: Array, ipiv: Array, b: Array) -> Array:
            return _banded_solve_device(ab, ipiv, b, kl, ku)

        batched = jax.vmap(jax.vmap(solve_one))
        if jnp.iscomplexobj(rhs):
            xr = batched(self.ab, self.ipiv, rhs.real)
            xi = batched(self.ab, self.ipiv, rhs.imag)
            return xr + 1j * xi
        return batched(self.ab, self.ipiv, rhs)


def build_Lk_neumann(k2: float, D1: np.ndarray, D2: np.ndarray) -> np.ndarray:
    """Build the Laplacian operator `$D_2 - k^2 I$` with Neumann BCs.

    Boundary rows are replaced by the corresponding rows of D1 to
    encode Neumann (`$\\partial p/\\partial y = \\dots$`) conditions
    for the pressure Poisson equation.

    For the mean mode `$k^2 = 0$`, the first row is instead replaced
    by `$(1, 0, \\dots, 0)$` to pin `$p_0 = 0$`, since the pure-Neumann
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

    Returns both `$H_k = (1/\\Delta t) I - c \\nu (D_2 - k^2 I)$` (implicit)
    and `$H_k^- = (1/\\Delta t) I + (1-c) \\nu (D_2 - k^2 I)$` (explicit).
    Boundary rows are replaced by identity rows to encode no-slip
    conditions `$u|_{\\mathrm{wall}} = 0$`.

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
        Kinematic viscosity `$1/\\mathrm{Re}$`.

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
    D1: np.ndarray | None = None,
    D2: np.ndarray | None = None,
    backend: str = "banded",
) -> dict[str, np.ndarray]:
    """Full offline precomputation for the influence-matrix method.

    Builds D1, D2 (if not provided), and then loops over all Fourier mode
    pairs `$(k_z, k_x)$` to construct the per-mode operators and IMM data.

    Output layout: ``[i_kz, i_kx, ...]``, matching the spectral array
    layout ``(Nkz, Nkx, Ny)``.  All returned arrays are real
    (``float64``), following the proof in the IMM reference document.

    The ``backend`` switch controls how the per-mode `$L_k$` and
    `$H_k$` operators are exposed:

    - ``"banded"`` (default): factored once on the CPU via LAPACK
      ``dgbtrf`` with `$k_l = k_u = p$`, returned as LAPACK-packed
      ``ab`` factors of shape ``(3p + 1, Ny)`` plus 0-based pivots.
      Memory use is `$O(N_y)$` per mode rather than `$O(N_y^2)$`.
    - ``"dense"``: returns the full `$N_y \\times N_y$` `$L_k$`,
      `$H_k$`, and explicit `$H_k^-$` matrices (legacy, for parity
      checks against the banded path).

    Parameters
    ----------
    y:
        Wall-normal grid points, shape ``(Ny,)``.
    kx_vals:
        Streamwise wavenumber values, shape ``(Nkx,)``.
    kz_vals:
        Spanwise wavenumber values, shape ``(Nkz,)``.
    p:
        Finite-difference accuracy order.  Doubles as the
        half-bandwidth of `$L_k$` and `$H_k$` (``kl = ku = p``).
    dt:
        Time step.
    c:
        Implicitness parameter.
    nu:
        Kinematic viscosity `$1/\\mathrm{Re}$`.
    D1:
        Optional precomputed first-derivative matrix.
    D2:
        Optional precomputed second-derivative matrix.
    backend:
        ``"banded"`` (default) or ``"dense"``.

    Returns
    -------
    :
        Dictionary with keys (always present):

        - ``"D1"``, ``"D2"``: derivative matrices, ``(Ny, Ny)``.
        - ``"p1"``, ``"p2"``: homogeneous pressure solutions,
          ``(Nkz, Nkx, Ny)``.
        - ``"v1"``, ``"v2"``: homogeneous wall-normal velocity
          solutions, ``(Nkz, Nkx, Ny)``.
        - ``"q1"``, ``"q2"``: homogeneous velocity potentials
          `$q_i = H_k^{-1} p_i$`, ``(Nkz, Nkx, Ny)``.
          The horizontal homogeneous responses are
          `$u^{(i)} = -i k_x q_i$`, `$w^{(i)} = -i k_z q_i$`.
        - ``"M_inv"``: inverted influence matrix,
          ``(Nkz, Nkx, 2, 2)``.

        Backend-specific keys.  Under ``backend="banded"``:

        - ``"Lk_ab"``: LAPACK-packed `$L_k$` LU factors,
          ``(Nkz, Nkx, 3p + 1, Ny)``.
        - ``"Lk_piv"``: `$L_k$` pivot indices (0-based),
          ``(Nkz, Nkx, Ny)`` int32.
        - ``"Hk_ab"``, ``"Hk_piv"``: same for `$H_k$`.

        Under ``backend="dense"``:

        - ``"Lk"``: Neumann-modified Laplacian, ``(Nkz, Nkx, Ny, Ny)``.
        - ``"Hk"``: implicit Helmholtz, ``(Nkz, Nkx, Ny, Ny)``.
        - ``"Hk_minus"``: explicit Helmholtz,
          ``(Nkz, Nkx, Ny, Ny)``.
    """
    if backend not in {"banded", "dense"}:
        raise ValueError(
            f"backend must be 'banded' or 'dense', got {backend!r}"
        )

    Ny = len(y)
    Nkz = len(kz_vals)
    Nkx = len(kx_vals)

    if D1 is None or D2 is None:
        from ..fd import build_diff_matrices

        D1, D2 = build_diff_matrices(y, p)

    # IMM homogeneous data — needed for both backends.
    p1_all = np.zeros((Nkz, Nkx, Ny))
    p2_all = np.zeros((Nkz, Nkx, Ny))
    v1_all = np.zeros((Nkz, Nkx, Ny))
    v2_all = np.zeros((Nkz, Nkx, Ny))
    q1_all = np.zeros((Nkz, Nkx, Ny))
    q2_all = np.zeros((Nkz, Nkx, Ny))
    M_inv_all = np.zeros((Nkz, Nkx, 2, 2))

    # Backend-specific operator storage.
    if backend == "banded":
        kl = ku = p
        ab_rows = 2 * kl + ku + 1  # = 3p + 1
        Lk_ab_all = np.zeros((Nkz, Nkx, ab_rows, Ny))
        Hk_ab_all = np.zeros((Nkz, Nkx, ab_rows, Ny))
        Lk_piv_all = np.zeros((Nkz, Nkx, Ny), dtype=np.int32)
        Hk_piv_all = np.zeros((Nkz, Nkx, Ny), dtype=np.int32)
    else:
        Lk_all = np.zeros((Nkz, Nkx, Ny, Ny))
        Hk_all = np.zeros((Nkz, Nkx, Ny, Ny))
        Hk_minus_all = np.zeros((Nkz, Nkx, Ny, Ny))

    e1 = np.zeros(Ny)
    e1[0] = 1.0
    e2 = np.zeros(Ny)
    e2[-1] = 1.0

    for iz, kz in enumerate(kz_vals):
        for ix, kx in enumerate(kx_vals):
            k2 = float(kx**2 + kz**2)

            # Laplacian with Neumann BCs (pressure)
            Lk = build_Lk_neumann(k2, D1, D2)

            # Helmholtz with Dirichlet BCs (velocity)
            Hk, Hk_minus = build_Hk_dirichlet(k2, D2, dt, c, nu)

            # Homogeneous pressure solutions `$L_k p_i = e_i$`.
            p1 = np.linalg.solve(Lk, e1)
            p2 = np.linalg.solve(Lk, e2)
            p1_all[iz, ix] = p1
            p2_all[iz, ix] = p2

            # Homogeneous velocity solutions `$v_i = H_k^{-1}
            # (-D_1 p_i)$` with zero Dirichlet BCs (no-slip).
            rhs1 = -D1 @ p1
            rhs2 = -D1 @ p2
            rhs1[0] = 0.0
            rhs1[-1] = 0.0
            rhs2[0] = 0.0
            rhs2[-1] = 0.0
            v1 = np.linalg.solve(Hk, rhs1)
            v2 = np.linalg.solve(Hk, rhs2)
            v1_all[iz, ix] = v1
            v2_all[iz, ix] = v2

            # Homogeneous velocity potentials `$q_i = H_k^{-1} p_i$`
            # with zero Dirichlet BCs. The horizontal homogeneous
            # responses factor as `$u^{(i)} = -i k_x q_i$` and
            # `$w^{(i)} = -i k_z q_i$` because the scalar `$-i k_x$`,
            # `$-i k_z$` commute with `$H_k^{-1}$` per mode.
            q_rhs1 = p1.copy()
            q_rhs2 = p2.copy()
            q_rhs1[0] = 0.0
            q_rhs1[-1] = 0.0
            q_rhs2[0] = 0.0
            q_rhs2[-1] = 0.0
            q1 = np.linalg.solve(Hk, q_rhs1)
            q2 = np.linalg.solve(Hk, q_rhs2)
            q1_all[iz, ix] = q1
            q2_all[iz, ix] = q2

            # Influence matrix `$M_{ji} = (D_1 v_i)|_{\\text{wall}_j}$`:
            # wall divergence produced by adding `$p_i$` to the pressure.
            M = np.zeros((2, 2))
            M[0, 0] = D1[0, :] @ v1
            M[0, 1] = D1[0, :] @ v2
            M[1, 0] = D1[-1, :] @ v1
            M[1, 1] = D1[-1, :] @ v2

            if k2 == 0.0:
                # Mean mode: `$p_1 \\equiv 1$` (constant, gauge freedom),
                # so column 0 of M is zero and M is singular. Zero the
                # `$p_1$` contribution and use only the top-wall
                # residual via `$p_2$`.
                M_inv_all[iz, ix] = np.array(
                    [[0.0, 0.0], [0.0, 1.0 / M[1, 1]]]
                )
            else:
                M_inv_all[iz, ix] = np.linalg.inv(M)

            # Operator caches: banded LU factors or dense copies.
            if backend == "banded":
                Lk_ab, Lk_piv = _banded_lu_numpy(Lk, kl, ku)
                Hk_ab, Hk_piv = _banded_lu_numpy(Hk, kl, ku)
                Lk_ab_all[iz, ix] = Lk_ab
                Lk_piv_all[iz, ix] = Lk_piv
                Hk_ab_all[iz, ix] = Hk_ab
                Hk_piv_all[iz, ix] = Hk_piv
            else:
                Lk_all[iz, ix] = Lk
                Hk_all[iz, ix] = Hk
                Hk_minus_all[iz, ix] = Hk_minus

    out: dict[str, np.ndarray] = {
        "D1": D1,
        "D2": D2,
        "p1": p1_all,
        "p2": p2_all,
        "v1": v1_all,
        "v2": v2_all,
        "q1": q1_all,
        "q2": q2_all,
        "M_inv": M_inv_all,
    }
    if backend == "banded":
        out["Lk_ab"] = Lk_ab_all
        out["Lk_piv"] = Lk_piv_all
        out["Hk_ab"] = Hk_ab_all
        out["Hk_piv"] = Hk_piv_all
    else:
        out["Lk"] = Lk_all
        out["Hk"] = Hk_all
        out["Hk_minus"] = Hk_minus_all
    return out


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
        the wall-normal coordinate ``y`` in ``[-1, 1]``,
        FD matrices D1 and D2, and all per-mode
        IMM operators via ``IMMChunker``.
        """
        self.ys = -jnp.cos(
            jnp.arange(params.res.ny, dtype=sharding.float_type)
            * jnp.pi
            / (params.res.ny - 1)
        )

        D1, D2 = build_diff_matrices(np.array(self.ys), params.res.fd_order)
        self.D1 = jax.device_put(jnp.array(D1), sharding.no_shard)
        self.D2 = jax.device_put(jnp.array(D2), sharding.no_shard)
        self.D1_bnd = jax.device_put(
            jnp.array(D1[[0, -1], :]), sharding.no_shard
        )
        self.D2_bnd = jax.device_put(
            jnp.array(D2[[0, -1], :]), sharding.no_shard
        )

        kx_global = fourier.kx_global
        kz_global = fourier.kz_global
        Nkz = len(kz_global)
        Nkx = len(kx_global)
        Ny = params.res.ny

        chunker = IMMChunker(
            ys_arr=np.array(self.ys),
            kx_global=kx_global,
            kz_global=kz_global,
            p=params.res.fd_order,
            dt=params.step.dt,
            c=params.step.implicitness,
            nu=1.0 / params.phys.re,
            D1_arr=D1,
            D2_arr=D2,
            backend=params.solver.backend,
        )

        # Homogeneous IMM data — always needed.
        self.p1 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "p1"),
        )
        self.p2 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "p2"),
        )
        self.v1 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "v1"),
        )
        self.v2 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "v2"),
        )
        self.q1 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "q1"),
        )
        self.q2 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "q2"),
        )
        self.M_inv = jax.make_array_from_callback(
            (Nkz, Nkx, 2, 2),
            sharding.spec_dy_op_shard,
            lambda idx: chunker.get_chunk(idx, "M_inv"),
        )

        # Operator caches: LAPACK-packed banded LU or legacy dense.
        if params.solver.backend == "banded":
            p = params.res.fd_order
            ab_rows = 3 * p + 1  # = 2*kl + ku + 1 with kl = ku = p
            Lk_ab = jax.make_array_from_callback(
                (Nkz, Nkx, ab_rows, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Lk_ab"),
            )
            Lk_piv = jax.make_array_from_callback(
                (Nkz, Nkx, Ny),
                sharding.spec_imm_corr_shard,
                lambda idx: chunker.get_chunk(idx, "Lk_piv"),
            )
            Hk_ab = jax.make_array_from_callback(
                (Nkz, Nkx, ab_rows, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Hk_ab"),
            )
            Hk_piv = jax.make_array_from_callback(
                (Nkz, Nkx, Ny),
                sharding.spec_imm_corr_shard,
                lambda idx: chunker.get_chunk(idx, "Hk_piv"),
            )
            self.Lk_op = PerModeBandedOperator(ab=Lk_ab, ipiv=Lk_piv)
            self.Hk_op = PerModeBandedOperator(ab=Hk_ab, ipiv=Hk_piv)
        else:
            self.Lk = jax.make_array_from_callback(
                (Nkz, Nkx, Ny, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Lk"),
            )
            Hk = jax.make_array_from_callback(
                (Nkz, Nkx, Ny, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Hk"),
            )
            self.Hk_minus = jax.make_array_from_callback(
                (Nkz, Nkx, Ny, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Hk_minus"),
            )
            self.Lk_op = DenseJAXSolver(self.Lk)
            self.Hk_op = DenseJAXSolver(Hk)


# ── Spectral transform aliases ───────────────────────────────────────────

phys_to_spec = phys_to_spec_2d
spec_to_phys = spec_to_phys_2d


# ── Solver functions (geometry-general) ──────────────────────────────────


def init_state(snapshot: str | None, flow: CartesianFlow) -> Array:
    """Initialize the flow state (velocity_spec)."""
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
        # velocity_phys = velocity_phys.at[...].subtract(flow.base_flow)
        velocity_spec = phys_to_spec_2d(velocity_phys)
        velocity_phys = spec_to_phys_2d(velocity_spec)

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

    dy_u = jnp.einsum("ij, zxj -> zxi", flow_.D1, u)
    dy_w = jnp.einsum("ij, zxj -> zxi", flow_.D1, w)

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

    # d_hat^n (discrete divergence at time n; ~0 after first step).
    dy_v_n = jnp.einsum("ij, zxj -> zxi", flow_.D1, v_n)
    d_hat_n = 1j * fourier_.kx * u_n + dy_v_n + 1j * fourier_.kz * w_n

    # Stage 1: interior pressure Poisson RHS.
    dy_Nv_j = jnp.einsum("ij, zxj -> zxi", flow_.D1, Nv_j)
    dy_Nv_n = jnp.einsum("ij, zxj -> zxi", flow_.D1, Nv_n)
    div_Nj = 1j * fourier_.kx * Nu_j + dy_Nv_j + 1j * fourier_.kz * Nw_j
    div_Nn = 1j * fourier_.kx * Nu_n + dy_Nv_n + 1j * fourier_.kz * Nw_n

    if params.solver.backend == "banded":
        Lk_d = _lk_matvec(d_hat_n, flow_.D2, flow_.D1_bnd, k2, k2_is_zero)
    else:
        Lk_d = _matvec_4d(flow_.Lk, d_hat_n)

    f_hat = d_hat_n / dt + c * div_Nj + (1 - c) * div_Nn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure with ZERO Neumann BCs.
    f_hat_P = f_hat.at[..., 0].set(0.0).at[..., -1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: Helmholtz solves for all three velocity components
    # against the particular pressure p_P (zero Dirichlet BCs).
    dx_pP = 1j * fourier_.kx * pP
    dy_pP = jnp.einsum("ij, zxj -> zxi", flow_.D1, pP)
    dz_pP = 1j * fourier_.kz * pP

    if params.solver.backend == "banded":
        Hku_n = _hk_minus_matvec(u_n, flow_.D2, k2, dt, c, nu)
        Hkv_n = _hk_minus_matvec(v_n, flow_.D2, k2, dt, c, nu)
        Hkw_n = _hk_minus_matvec(w_n, flow_.D2, k2, dt, c, nu)
    else:
        Hku_n = _matvec_4d(flow_.Hk_minus, u_n)
        Hkv_n = _matvec_4d(flow_.Hk_minus, v_n)
        Hkw_n = _matvec_4d(flow_.Hk_minus, w_n)

    Ru = Hku_n - dx_pP + c * Nu_j + (1 - c) * Nu_n
    Rv = Hkv_n - dy_pP + c * Nv_j + (1 - c) * Nv_n
    Rw = Hkw_n - dz_pP + c * Nw_j + (1 - c) * Nw_n

    Ru = Ru.at[..., 0].set(0.0).at[..., -1].set(0.0)
    Rv = Rv.at[..., 0].set(0.0).at[..., -1].set(0.0)
    Rw = Rw.at[..., 0].set(0.0).at[..., -1].set(0.0)

    u_arb = flow_.Hk_op.solve(Ru)
    v_arb = flow_.Hk_op.solve(Rv)
    w_arb = flow_.Hk_op.solve(Rw)

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
    u_new = u_arb - 1j * fourier_.kx * q_new
    w_new = w_arb - 1j * fourier_.kz * q_new

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
) -> tuple:
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
        """Initialize the flow state with bound flow singleton."""
        return init_state(snapshot, flow)

    return predict_and_correct, iterate_correction, init_state_bound
