"""Cartesian geometry: Fourier class, norms, integration, IMM."""

from dataclasses import dataclass, field

import jax
import jax.scipy.linalg as sla
import numpy as np
from jax import Array, lax
from jax import numpy as jnp
from scipy.linalg.lapack import get_lapack_funcs

from ..operators import complex_harmonics, real_harmonics
from ..parameters import derived_params, params
from ..sharding import register_dataclass_pytree, sharding


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
