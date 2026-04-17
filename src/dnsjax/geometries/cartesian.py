"""Cartesian geometry: Fourier class, norms, integration, IMM."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.scipy.linalg as sla
import numpy as np
from jax import Array
from jax import numpy as jnp

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

    ``lapl`` is the horizontal Laplacian `$-(k_x^2 + k_z^2)$`; the
    y-part is handled by finite-difference matrices.
    """

    kx_global: np.ndarray = field(init=False)
    kx: Array = field(init=False)
    kz_global: np.ndarray = field(init=False)
    kz: Array = field(init=False)
    k_metric: Array = field(init=False)

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
        self.kz = jnp.array(self.kz_global).reshape([-1, 1, 1])

        self.k_metric = jnp.where(self.kx == 0, 1, 2)


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
            "Simpson integration is not yet implemented"
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
            Operator name (e.g. ``"Lk"``, ``"Hk"``).
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

    def __post_init__(self):
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


@register_dataclass_pytree
@dataclass
class LineaxBandedSolver:
    """The Lineax sparse operator path."""

    matrix: Array
    lower_band: int
    upper_band: int
    operator: Any = field(init=False)

    def __post_init__(self):
        raise NotImplementedError(
            "Lineax banded packing is pending extraction implementation!"
        )

    def solve(self, rhs: Array) -> Array:
        raise NotImplementedError


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
    D1: np.ndarray | None = None,
    D2: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Full offline precomputation for the influence-matrix method.

    Builds D1, D2 (if not provided), and then loops over all Fourier mode
    pairs `$(k_z, k_x)$` to construct the per-mode operators and IMM data.

    The output arrays are indexed ``[i_kz, i_kx, ...]``, matching the
    spectral array layout ``(Nkz, Nkx, Ny)``.

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
    D1:
        Optional precomputed first-derivative matrix.
    D2:
        Optional precomputed second-derivative matrix.

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
        - ``"v1"``, ``"v2"``: homogeneous velocity solutions,
          ``(Nkz, Nkx, Ny)``
        - ``"M_inv"``: inverted influence matrix, ``(Nkz, Nkx, 2, 2)``
    """
    Ny = len(y)
    Nkz = len(kz_vals)
    Nkx = len(kx_vals)

    if D1 is None or D2 is None:
        from ..fd import build_diff_matrices

        D1, D2 = build_diff_matrices(y, p)

    Lk_all = np.zeros((Nkz, Nkx, Ny, Ny))
    Hk_all = np.zeros((Nkz, Nkx, Ny, Ny))
    Hk_minus_all = np.zeros((Nkz, Nkx, Ny, Ny))
    p1_all = np.zeros((Nkz, Nkx, Ny))
    p2_all = np.zeros((Nkz, Nkx, Ny))
    v1_all = np.zeros((Nkz, Nkx, Ny))
    v2_all = np.zeros((Nkz, Nkx, Ny))
    M_inv_all = np.zeros((Nkz, Nkx, 2, 2))

    e1 = np.zeros(Ny)
    e1[0] = 1.0
    e2 = np.zeros(Ny)
    e2[-1] = 1.0

    for iz, kz in enumerate(kz_vals):
        for ix, kx in enumerate(kx_vals):
            k2 = float(kx**2 + kz**2)

            # Laplacian with Neumann BCs (pressure)
            Lk = build_Lk_neumann(k2, D1, D2)
            Lk_all[iz, ix] = Lk

            # Helmholtz with Dirichlet BCs (velocity)
            Hk, Hk_minus = build_Hk_dirichlet(k2, D2, dt, c, nu)
            Hk_all[iz, ix] = Hk
            Hk_minus_all[iz, ix] = Hk_minus

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

            # Influence matrix `$M_{ji} = (D_1 v_i)|_{\text{wall}_j}$`:
            # wall divergence produced by adding `$p_i$` to the pressure.
            M = np.zeros((2, 2))
            M[0, 0] = D1[0, :] @ v1
            M[0, 1] = D1[0, :] @ v2
            M[1, 0] = D1[-1, :] @ v1
            M[1, 1] = D1[-1, :] @ v2

            if k2 == 0.0:
                # Mean mode: `$p_1 \equiv 1$` (constant, gauge freedom),
                # so column 0 of M is zero and M is singular. Zero the
                # `$p_1$` contribution and use only the top-wall
                # residual via `$p_2$`.
                M_inv_all[iz, ix] = np.array(
                    [[0.0, 0.0], [0.0, 1.0 / M[1, 1]]]
                )
            else:
                M_inv_all[iz, ix] = np.linalg.inv(M)

    return {
        "D1": D1,
        "D2": D2,
        "Lk": Lk_all,
        "Hk": Hk_all,
        "Hk_minus": Hk_minus_all,
        "p1": p1_all,
        "p2": p2_all,
        "v1": v1_all,
        "v2": v2_all,
        "M_inv": M_inv_all,
    }
