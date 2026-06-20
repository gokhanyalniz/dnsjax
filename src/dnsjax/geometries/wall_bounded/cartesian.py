"""Cartesian geometry: Fourier class, norms, integration, IMM, and solvers.

Provides all geometry-general infrastructure for wall-bounded Cartesian
flows: the ``Fourier`` wavenumber class, the ``CartesianFlow`` base
dataclass (CGL grid, FD matrices, IMM operators), spectral solvers
(influence-matrix method, predictor-corrector time stepping), and
diagnostic helpers (norms, perturbation energy).

Flow-specific modules (e.g. ``flows.wall_bounded.plane_couette``) subclass
``CartesianFlow`` to define the base flow, then call
``build_cartesian_stepper`` to obtain ready-to-use time-stepping
functions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ...fd import (
    build_diff_matrices,
    build_integration_weights,
    local_grid_spacing,
    tanh_two_sided_grid,
)
from ...measurements import get_cfl
from ...operators import (
    complex_harmonics,
    pad_harmonics,
    phys_to_spec_2d,
    real_harmonics,
    spec_to_phys_2d,
)
from ...parameters import derived_params, params
from ...rhs import get_nonlin
from ...sharding import register_dataclass_pytree, sharding
from ...solvers import (
    DenseJAXSolver,
    PerModeBandedOperator,
    _extract_banded_corners,
    _spike_factor,
    validate_spike_partition,
)
from ._base import (
    apply_y_matrix,
    build_wall_bounded_stepper,
    extract_mean_mode,
    get_inprod,  # noqa: F401 — re-exported
    get_norm,  # noqa: F401 — re-exported
    get_norm2,
    get_pert_enstrophy,  # noqa: F401 — re-exported
    init_state,  # noqa: F401 — re-exported
    integrate_scalar,
    pad_base_flow,  # noqa: F401 — re-exported
    phys_to_spec,  # noqa: F401 — re-exported
    spec_to_phys,  # noqa: F401 — re-exported
)


@register_dataclass_pytree
@dataclass
class Fourier:
    r"""Wavenumber grids for the Cartesian wall-bounded geometry.

    Broadcasting shapes match the spectral layout
    ``(ny, nz_spec, nx_spec)``.  When ``nz_spec > nz - 1``
    or ``nx_spec > nx // 2`` (spectral padding for 2D
    divisibility), the trailing padding entries carry nonzero
    beyond-resolution placeholder wavenumbers (see
    ``pad_harmonics`` in :mod:`dnsjax.operators`): every
    per-mode operator assembled at a padding slot is then
    regular, and the fields there are identically zero (the
    forward FFT re-zeroes the padding slots on every
    evaluation), so zero RHS gives zero solution and the
    padding modes need no special-casing.

    Attributes
    ----------
    kx:
        Streamwise wavenumber, shape ``(1, 1, nx_spec)``,
        sharded on `$k_x$` (``np1``).
    kz:
        Spanwise wavenumber, shape ``(1, nz_spec, 1)``,
        sharded on `$k_z$` (``np0``).
    k_metric:
        Hermitian-symmetry weight: 2 for `$k_x > 0$`,
        1 for `$k_x = 0$` (padding columns get 2 — inert,
        they only ever weight zero fields).
    k2:
        Squared horizontal wavenumber
        `$k_x^2 + k_z^2$`.
    mean_mask:
        Boolean mask that is ``True`` only at the mean mode
        `$(k_z, k_x) = (0, 0)$` (global index ``(0, 0)``;
        padding modes are appended at the end).  The mean
        mode is the only `$k^2 = 0$` mode, so this single
        mask serves the operator pin row, the
        influence-matrix mean branch, and all mean-mode
        physics (projections and bulk-velocity writes).
    """

    kx: Array = field(init=False)
    kz: Array = field(init=False)
    k_metric: Array = field(init=False)
    k2: Array = field(init=False)
    mean_mask: Array = field(init=False)

    def __post_init__(self) -> None:
        kx_vals = (
            pad_harmonics(
                real_harmonics(params.res.nx),
                params.res.nx,
                sharding.nx_spec_pad,
            )
            * 2
            * jnp.pi
            / params.geo.lx
        )
        self.kx = jax.device_put(
            kx_vals.reshape([1, 1, -1]).astype(sharding.float_type),
            P(None, None, sharding.a1),
        )

        kz_vals = (
            pad_harmonics(
                complex_harmonics(params.res.nz),
                params.res.nz,
                sharding.nz_spec_pad,
            )
            * 2
            * jnp.pi
            / params.geo.lz
        )
        self.kz = jax.device_put(
            kz_vals.reshape([1, -1, 1]).astype(sharding.float_type),
            P(None, sharding.a0, None),
        )

        self.k_metric = jnp.where(self.kx == 0, 1, 2).astype(
            sharding.float_type
        )

        self.k2 = self.kx**2 + self.kz**2

        # One-hot at the mean mode (kz, kx) = (0, 0): the true
        # modes precede the padding, so it is global index (0, 0).
        # The mean mode is the only k^2 = 0 mode (padding slots
        # carry nonzero placeholder wavenumbers).
        e_kx = (
            jnp.zeros(kx_vals.shape[0], dtype=sharding.float_type)
            .at[0]
            .set(1.0)
        )
        e_kz = (
            jnp.zeros(kz_vals.shape[0], dtype=sharding.float_type)
            .at[0]
            .set(1.0)
        )
        self.mean_mask = (
            jax.device_put(
                e_kz.reshape([1, -1, 1]), P(None, sharding.a0, None)
            )
            * jax.device_put(
                e_kx.reshape([1, 1, -1]), P(None, None, sharding.a1)
            )
        ) == 1.0


fourier: Fourier = Fourier()


# Backward-compatible alias (used by tests/test_integration.py).
integrate_scalar_in_y = integrate_scalar


def clenshaw_curtis_weights(ny: int) -> Array:
    r"""Clenshaw-Curtis quadrature weights for a CGL grid.

    For ``ny`` Chebyshev-Gauss-Lobatto points
    `$y_j = -\cos(j\pi / N)$`, `$j = 0, \ldots, N$` with
    `$N = \texttt{ny} - 1$`, returns the exact quadrature
    weights `$w_j$` such that

    .. math::
        \int_{-1}^{1} f(y)\,dy
        \;\approx\; \sum_{j=0}^{N} w_j\,f(y_j).

    The rule is exact for polynomials of degree `$\le N$`
    (spectral accuracy for smooth integrands).  Works for
    both odd and even *ny*.

    Parameters
    ----------
    ny:
        Number of CGL grid points (`$N + 1$`).

    Returns
    -------
    :
        Weight array of shape ``(ny,)`` with dtype
        ``sharding.float_type``.

    References
    ----------
    Trefethen, *Spectral Methods in MATLAB* (2000), ch. 12.
    """
    N = ny - 1
    theta = jnp.arange(ny, dtype=sharding.float_type) * jnp.pi / N

    if N % 2 == 0:  # N even (ny odd)
        M = N // 2 - 1
        w_end = 1.0 / (N * N - 1)
    else:  # N odd (ny even)
        M = (N - 1) // 2
        w_end = 1.0 / (N * N)

    if M > 0:
        k = jnp.arange(1, M + 1, dtype=sharding.float_type)
        cos_terms = jnp.cos(2 * k[None, :] * theta[:, None])
        coeffs = 2.0 / (4 * k**2 - 1)
        v = 1.0 - jnp.sum(cos_terms * coeffs[None, :], axis=1)
    else:
        v = jnp.ones(ny, dtype=sharding.float_type)

    if N % 2 == 0:
        v = v - jnp.cos(N * theta) / (N * N - 1)

    w = (2.0 / N) * v
    w = w.at[0].set(w_end)
    w = w.at[-1].set(w_end)

    return w


def build_cartesian_grid(
    ny: int,
    fd_order: int,
    wall_grid: str | None = None,
    grid_type: str | None = None,
    grid_stretch: float = 1.5,
) -> tuple[Array, Array, Array, Array]:
    r"""Build the Cartesian wall-normal grid, FD matrices, and
    quadrature weights.

    Grid selection (precedence):

    1. *wall_grid*: load from file.
    2. *grid_type*: ``"tanh"`` for symmetric tanh stretching,
       ``"cgl"`` for default CGL.
    3. Default: CGL grid.

    Parameters
    ----------
    ny:
        Number of wall-normal grid points.
    fd_order:
        Finite-difference stencil half-bandwidth.
    wall_grid:
        Optional path to a custom wall-normal grid file.
        File format: one coordinate per line in
        wall-to-interior order (first line = top wall
        `$y = 1$`, last line = bottom wall `$y = -1$`).
        The code reverses to ascending order internally.
        Custom grids use composite polynomial integration
        weights (order-*p* accuracy matching the FD stencil)
        instead of Clenshaw-Curtis.
    grid_type:
        Named grid type (``"cgl"`` or ``"tanh"``).
    grid_stretch:
        Stretching parameter for ``grid_type="tanh"``.

    Returns
    -------
    ys:
        Wall-normal grid on `$[-1, 1]$`, shape ``(ny,)``.
    D1:
        First-derivative FD matrix, shape ``(ny, ny)``.
    D2:
        Second-derivative FD matrix, shape ``(ny, ny)``.
    y_weights:
        Quadrature weights, shape ``(ny,)``.
    """
    if wall_grid is not None:
        grid_raw = np.loadtxt(wall_grid, dtype=np.float64)
        if len(grid_raw) != ny:
            raise ValueError(
                f"Wall grid file has {len(grid_raw)} points, expected ny={ny}"
            )
        grid = grid_raw[::-1].copy()
        if not np.isclose(grid[0], -1.0) or not np.isclose(grid[-1], 1.0):
            raise ValueError(
                "Cartesian wall grid must span [-1, 1]"
                f" (got [{grid[0]}, {grid[-1]}])"
            )
        ys = jnp.asarray(grid, dtype=sharding.float_type)
        w = build_integration_weights(grid, fd_order)
        y_weights = jnp.asarray(w, dtype=sharding.float_type)
    elif grid_type == "tanh":
        grid = tanh_two_sided_grid(ny, grid_stretch)
        ys = jnp.asarray(grid, dtype=sharding.float_type)
        w = build_integration_weights(grid, fd_order)
        y_weights = jnp.asarray(w, dtype=sharding.float_type)
    else:
        ys = -jnp.cos(
            jnp.arange(ny, dtype=sharding.float_type) * jnp.pi / (ny - 1)
        )
        y_weights = clenshaw_curtis_weights(ny)

    D1, D2 = build_diff_matrices(ys, fd_order)
    return ys, D1, D2, y_weights


# ── SPIKE block-partitioned operator builders ─────────────────────


def _build_Lk_blocks_gpu(
    D1: Array,
    D2: Array,
    k2: Array,
    mean_mask: Array,
    p: int,
    P: int,
    m: int,
) -> tuple[Array, Array, Array]:
    r"""Build SPIKE block-partitioned `$L_k$` on GPU.

    The Neumann-BC pressure Poisson operator
    `$L_k = D_2 - k^2 I$` (with `$D_1$` wall rows for
    `$k^2 \ne 0$` and a top-wall pin for the mean mode,
    the only `$k^2 = 0$` system) is assembled directly
    into per-block dense form for the SPIKE
    factorisation.  No `$(N_y, N_y)$` matrix is
    materialised.

    Parameters
    ----------
    D1:
        First-derivative matrix, shape ``(Ny, Ny)``.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    k2:
        `$k_x^2 + k_z^2$`, shape ``(Nkz, Nkx, 1)``.
    mean_mask:
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

    # Row 0 (block 0, local row 0): D1[0,:m] for all modes.
    D1_row0 = D1[0, :m]
    A_blocks = A_blocks.at[:, :, 0, 0, :].set(D1_row0[None, None])

    # Row Ny-1 (block P-1, local row m-1): D1[-1,:] for
    # all modes, pin row [...,0,1] at the mean mode.
    D1_rowN = D1[-1, (P - 1) * m :]
    pin_row = jnp.zeros(m, dtype=dtype).at[-1].set(1.0)
    rowN = jnp.where(mean_mask, pin_row, D1_rowN)
    A_blocks = A_blocks.at[:, :, -1, -1, :].set(rowN)

    # Coupling corners (mode-independent, from D2).
    B_raw, C_raw = _extract_banded_corners(D2, P, m, p)
    batch = k2.shape[:2] + (P, p, p)
    B_corner = jnp.broadcast_to(B_raw[None, None], batch)
    C_corner = jnp.broadcast_to(C_raw[None, None], batch)

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
    B_raw, C_raw = _extract_banded_corners(D2, P, m, p, scale=-c * nu)
    batch = k2.shape[:2] + (P, p, p)
    B_corner = jnp.broadcast_to(B_raw[None, None], batch)
    C_corner = jnp.broadcast_to(C_raw[None, None], batch)

    return A_blocks, B_corner, C_corner


def _build_Lk_dense_gpu(
    D1: Array, D2: Array, k2: Array, mean_mask: Array
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

    # Row 0: D1[0, :] for all modes (Neumann).
    Lk = Lk.at[..., 0, :].set(D1[0, :])

    # Row -1: D1[-1, :] for all modes; pin row [0, ..., 0, 1]
    # at the mean mode.  mean_mask is (Nkz, Nkx, 1); `jnp.where`
    # broadcasts the (Ny,) branches to (Nkz, Nkx, Ny).
    pin = eye[-1, :]  # (Ny,)
    row_N = jnp.where(mean_mask, pin, D1[-1, :])
    Lk = Lk.at[..., -1, :].set(row_N)

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

    Subclasses must set ``base_flow`` and ``curl_base_flow``
    *after* calling ``super().__post_init__()``, which builds
    the CGL grid
    (``ys``), Clenshaw-Curtis quadrature weights
    (``y_weights``), finite-difference matrices, and all
    per-mode IMM operators.

    Attributes
    ----------
    D1:
        First-derivative FD matrix, shape ``(Ny, Ny)``.
    D2:
        Second-derivative FD matrix, shape ``(Ny, Ny)``.
    D1_bnd:
        Boundary rows `$D_1[0,:],\\; D_1[-1,:]$`,
        shape ``(2, Ny)``.
    D2_bnd:
        Boundary rows `$D_2[0,:],\\; D_2[-1,:]$`,
        shape ``(2, Ny)``.
    """

    ys: Array = field(init=False)
    y_weights: Array = field(init=False)
    cfl_inv_spacing: Array = field(init=False)
    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    base_flow_padded: Array = field(init=False)
    curl_base_flow_padded: Array = field(init=False)
    D1: Array = field(init=False)
    D2: Array = field(init=False)
    D1_bnd: Array = field(init=False)
    D2_bnd: Array = field(init=False)
    Lk_op: DenseJAXSolver | PerModeBandedOperator = field(init=False)
    Hk_op: DenseJAXSolver | PerModeBandedOperator = field(init=False)
    v1: Array = field(init=False)
    v2: Array = field(init=False)
    q1: Array = field(init=False)
    q2: Array = field(init=False)
    M_inv: Array = field(init=False)
    h_bulk_response: Array = field(init=False)
    H_bulk_inv: Array = field(init=False)

    def __post_init__(self) -> None:
        """Build CGL grid, quadrature weights, FD matrices,
        and IMM operators.

        Constructs the Chebyshev-Gauss-Lobatto grid for
        the wall-normal coordinate `$y$` in `$[-1, 1]$`
        and precomputes Clenshaw-Curtis quadrature weights
        (``y_weights``) for spectral-accuracy integration
        in the wall-normal direction, then builds
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
        self.ys, D1, D2, self.y_weights = build_cartesian_grid(
            params.res.ny,
            params.res.fd_order,
            params.geo.wall_grid,
            params.geo.grid_type,
            params.geo.grid_stretch,
        )

        derived_params.wall_normal_grid = [
            float(v) for v in np.asarray(self.ys)
        ]

        # Inverse local advection length scales for the CFL
        # diagnostic (:func:`dnsjax.measurements.get_cfl`),
        # per component (u, v, w), zero in the ny_y_pad rows.
        # Uniform directions use the spectral-resolution
        # spacing `$\Delta = L/n$`; switch to
        # ``padded_res.nx_padded`` / ``nz_padded`` for the
        # dealiased-grid convention.
        inv_sp = np.zeros(
            (3, params.res.ny + sharding.ny_y_pad),
            dtype=sharding.float_type,
        )
        inv_sp[0, : params.res.ny] = params.res.nx / params.geo.lx
        inv_sp[1, : params.res.ny] = 1.0 / local_grid_spacing(
            np.asarray(self.ys)
        )
        inv_sp[2, : params.res.ny] = params.res.nz / params.geo.lz
        self.cfl_inv_spacing = jax.device_put(
            inv_sp[:, :, None, None], sharding.no_shard
        )

        self.D1 = jax.device_put(D1, sharding.no_shard)
        self.D2 = jax.device_put(D2, sharding.no_shard)
        self.D1_bnd = jax.device_put(D1[[0, -1], :], sharding.no_shard)
        self.D2_bnd = jax.device_put(D2[[0, -1], :], sharding.no_shard)

        Nkz = sharding.nz_spec
        Nkx = sharding.nx_spec
        Ny = params.res.ny

        p = params.res.fd_order
        dt = params.step.dt
        c = params.step.implicitness
        nu = 1.0 / params.phys.re

        # Solver-internal wavenumber arrays: (Nkz, Nkx, 1).
        k2_s = fourier.k2[0, ..., None]
        mean_s = fourier.mean_mask[0, ..., None]

        if params.solver.backend == "banded":
            bt = params.solver.block_thomas
            P_blk, m_blk = validate_spike_partition(Ny, p, "Ny", bt)
            # Build and factor one operator at a time, dropping
            # the block arrays immediately (their buffers are
            # donated into the factorisation), so the setup peak
            # never holds two unfactored operators at once.
            Lk_A, Lk_B, Lk_C = _build_Lk_blocks_gpu(
                self.D1,
                self.D2,
                k2_s,
                mean_s,
                p,
                P_blk,
                m_blk,
            )
            self.Lk_op = _spike_factor(Lk_A, Lk_B, Lk_C, bt)
            del Lk_A, Lk_B, Lk_C
            Hk_A, Hk_B, Hk_C = _build_Hk_blocks_gpu(
                self.D2,
                k2_s,
                dt,
                c,
                nu,
                p,
                P_blk,
                m_blk,
            )
            self.Hk_op = _spike_factor(Hk_A, Hk_B, Hk_C, bt)
            del Hk_A, Hk_B, Hk_C
        else:
            # Dense backend: parity/reference path.  Full
            # `(Nkz, Nkx, Ny, Ny)` matrices are built, LU-factored
            # (donated, so the factors reuse their buffers), then
            # dropped — only the factors are kept.
            Lk_dense = _build_Lk_dense_gpu(self.D1, self.D2, k2_s, mean_s)
            self.Lk_op = DenseJAXSolver(Lk_dense)
            del Lk_dense
            Hk_dense = _build_Hk_dense_gpu(self.D2, k2_s, dt, c, nu)
            self.Hk_op = DenseJAXSolver(Hk_dense)
            del Hk_dense

        self._derive_imm_homogeneous_data(Nkz, Nkx, Ny)
        self._precompute_bulk_response(Nkz, Nkx, Ny)

    def _derive_imm_homogeneous_data(
        self, Nkz: int, Nkx: int, Ny: int
    ) -> None:
        r"""Fill ``v1``, ``v2``, ``q1``, ``q2``, and ``M_inv``
        from the factored GPU operator.

        Both backends converge here once :attr:`Lk_op` and
        :attr:`Hk_op` are in place.  Nothing else on the CPU
        needs to do another LU solve -- everything below runs
        against the already-factored device operator.

        In Schur-complement notation, the arrays
        ``p1, p2, v1, v2, q1, q2`` are the columns of
        `$A_{II}^{-1}\,A_{IB}$` (the interior-to-boundary
        coupling through the factored interior operator), and
        ``M_inv`` is `$S^{-1}$` where `$S$` is the `$2 \times
        2$` Schur complement (influence / capacitance matrix).
        See :func:`_imm_iteration` for the full context.  The
        homogeneous pressures ``p1``, ``p2`` are needed only
        within this derivation (the IMM never assembles the
        pressure), so they are not stored on the dataclass.

        The mean mode (the only `$k^2 = 0$` system) is
        handled analytically: ``M`` has a zero second column
        there (`$p_2 \equiv 1$` is a pressure gauge), so the
        `$2 \times 2$` inverse is replaced by
        `$[[1/M_{00}, 0], [0, 0]]$`.  The ``jnp.where``
        around ``safe_det`` keeps the regular branch NaN-free
        before the selection happens.  Padding modes take the
        regular branch (their placeholder `$k^2 \ne 0$`
        systems are as well-posed as physical ones); the
        values are inert, multiplied only by the exactly-zero
        wall residuals of zero fields.

        After ``M_inv`` is built, ``v1`` and ``v2`` are zeroed
        at the mean mode so the IMM velocity correction produces
        zero there (continuity forces `$v \equiv 0$` at
        `$k^2 = 0$`).  The zeroing must follow the ``M_inv``
        computation, which uses the original ``v1`` to evaluate
        `$1/M_{00}$`.
        """
        # All solver work uses (Nkz, Nkx, Ny) layout; results
        # are transposed to field layout (Ny, Nkz, Nkx) at the end.
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
        p1_s = self.Lk_op.solve(e1_b)
        p2_s = self.Lk_op.solve(e2_b)

        rhs_v1 = -jnp.einsum("ij, zxj -> zxi", self.D1, p1_s)
        rhs_v2 = -jnp.einsum("ij, zxj -> zxi", self.D1, p2_s)
        rhs_v1 = rhs_v1.at[..., 0].set(0.0).at[..., -1].set(0.0)
        rhs_v2 = rhs_v2.at[..., 0].set(0.0).at[..., -1].set(0.0)
        v1_s = self.Hk_op.solve(rhs_v1)
        v2_s = self.Hk_op.solve(rhs_v2)

        q_rhs1 = p1_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
        q_rhs2 = p2_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
        q1_s = self.Hk_op.solve(q_rhs1)
        q2_s = self.Hk_op.solve(q_rhs2)

        # Influence matrix `$M_{ji} = (D_1 v_i)|_{\\text{wall}_j}$`.
        M00 = jnp.einsum("j, zxj -> zx", self.D1_bnd[0], v1_s)
        M01 = jnp.einsum("j, zxj -> zx", self.D1_bnd[0], v2_s)
        M10 = jnp.einsum("j, zxj -> zx", self.D1_bnd[-1], v1_s)
        M11 = jnp.einsum("j, zxj -> zx", self.D1_bnd[-1], v2_s)

        is_mean = fourier.mean_mask[0]
        det = M00 * M11 - M01 * M10
        safe_det = jnp.where(is_mean, 1.0, det)
        inv_00 = jnp.where(is_mean, 1.0 / M00, M11 / safe_det)
        inv_01 = jnp.where(is_mean, 0.0, -M01 / safe_det)
        inv_10 = jnp.where(is_mean, 0.0, -M10 / safe_det)
        inv_11 = jnp.where(is_mean, 0.0, M00 / safe_det)
        self.M_inv = jnp.stack(
            [
                jnp.stack([inv_00, inv_01], axis=-1),
                jnp.stack([inv_10, inv_11], axis=-1),
            ],
            axis=-2,
        )

        # Transpose to field layout (Ny, Nkz, Nkx).
        self.v1 = v1_s.transpose(2, 0, 1)
        self.v2 = v2_s.transpose(2, 0, 1)
        self.q1 = q1_s.transpose(2, 0, 1)
        self.q2 = q2_s.transpose(2, 0, 1)

        # Zero homogeneous wall-normal velocity at the mean mode.
        self.v1 = jnp.where(fourier.mean_mask, 0.0, self.v1)
        self.v2 = jnp.where(fourier.mean_mask, 0.0, self.v2)

    def _precompute_bulk_response(self, Nkz: int, Nkx: int, Ny: int) -> None:
        r"""Precompute the Helmholtz response for mean-mode
        velocity enforcement.

        Solves `$H_k\,h = \mathbf{1}$` (unit uniform RHS,
        zero Dirichlet wall BCs) at the mean mode
        `$(k_x, k_z) = (0, 0)$`.  The response `$h(y)$` is
        the velocity profile produced by a unit mean pressure
        gradient over one implicit time step.  Its bulk
        `$H = \int_{-1}^{1} h\,dy / 2$` gives the scaling
        needed to zero a perturbation bulk velocity component:

        .. math::
            G = -\frac{U_{b,\mathrm{pert}}}{H}, \qquad
            \bar{u}' \;\leftarrow\; \bar{u}' + G\,h

        which is equivalent to adding a uniform forcing `$G$`
        to the mean-mode Helmholtz RHS before solving.

        Used by both ``constant_bulk_velocity`` (streamwise)
        and ``block_mean_spanwise_velocity`` (spanwise);
        the `$H_k$` operator at the mean mode is the same for
        all horizontal velocity components, so a single
        response `$h$` serves both directions.
        """
        if (
            params.phys.driving != "constant_bulk_velocity"
            and not params.phys.block_mean_spanwise_velocity
        ):
            self.h_bulk_response = jnp.zeros(
                Ny,
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            self.H_bulk_inv = jnp.zeros((), dtype=sharding.float_type)
            return

        # Unit uniform RHS at the mean mode only (``mean_mask``;
        # all other modes, padding included, get zero RHS), zero
        # wall BCs.  Solver-internal layout (Nkz, Nkx, Ny).
        ones_vec = (
            jnp.ones(Ny, dtype=sharding.float_type)
            .at[0]
            .set(0.0)
            .at[-1]
            .set(0.0)
        )
        rhs = jnp.where(fourier.mean_mask[0, ..., None], ones_vec, 0.0)

        h_full = self.Hk_op.solve(rhs)

        self.h_bulk_response = jax.device_put(
            extract_mean_mode(h_full.transpose(2, 0, 1)[None])[0],
            sharding.no_shard,
        )
        H_bulk = jnp.dot(self.y_weights, self.h_bulk_response) / 2
        self.H_bulk_inv = 1.0 / H_bulk


# ── Solver functions ─────────────────────────────────────────────────────


def _curl_fn(
    state: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> Array:
    """Spectral curl with 1D FD in y and spectral derivatives in x and z."""
    u, v, w = state[0], state[1], state[2]

    # Stack (u, w) so the two D1 y-derivatives needed for the curl
    # are one batched GEMM rather than two separate kernel launches.
    dy_uw = apply_y_matrix(flow_.D1, jnp.stack([u, w]))
    dy_u, dy_w = dy_uw[0], dy_uw[1]

    dx_v = 1j * fourier_.kx * v
    dz_v = 1j * fourier_.kz * v
    dx_w = 1j * fourier_.kx * w
    dz_u = 1j * fourier_.kz * u

    omega_x = dy_w - dz_v
    omega_y = dz_u - dx_w
    omega_z = dx_v - dy_u

    return jnp.array([omega_x, omega_y, omega_z])


# Per-direction CFL column names, matching the physical-space
# component order (u, v, w) = (x, y, z).
CFL_NAMES: tuple[str, str, str] = ("CFL_x", "CFL_y", "CFL_z")


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
    measure_fn: Callable[[Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    """Evaluate non-linear RHS terms (optionally measured)."""
    return get_nonlin(
        state,
        flow_.base_flow_padded,
        flow_.curl_base_flow_padded,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier_, flow_),
        measure_fn,
    )


def _get_rhs(
    state: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> Array:
    """Evaluate non-linear RHS terms."""
    return _get_rhs_core(state, fourier_, flow_, None)


def _get_rhs_measured(
    state: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate non-linear RHS terms + CFL measurements."""

    def _measure(u_phys: Array, omega_phys: Array) -> dict[str, Array]:
        return get_cfl(
            u_phys,
            flow_.base_flow_padded,
            flow_.cfl_inv_spacing,
            CFL_NAMES,
        )

    return _get_rhs_core(state, fourier_, flow_, _measure)


def _lk_matvec(
    u: Array,
    flow_: CartesianFlow,
    fourier_: Fourier,
) -> Array:
    r"""Apply `$L_k u$` for the Neumann-BC pressure Poisson operator.

    Matrix-free evaluation that avoids storing the per-mode
    ``(Nkz, Nkx, Ny, Ny)`` operator.  The interior of the
    output is `$D_2 u - k^2 u$`; the wall rows use `$D_1$`
    to encode Neumann BCs, except for the mean mode (the
    only `$k^2 = 0$` mode) where the top-wall row pins
    `$p_{N_y-1} = 0$` (matching
    :func:`_build_Lk_dense_gpu`).

    Parameters
    ----------
    u:
        Field, shape ``(Ny, Nkz, Nkx)``.
    flow\_:
        Cartesian flow data (uses ``D2``, ``D1_bnd``).
    fourier\_:
        Wavenumber grids (uses ``k2``, ``mean_mask``).
    """
    D2u = apply_y_matrix(flow_.D2, u)
    out = D2u - fourier_.k2 * u
    bot = jnp.einsum("j, jzx -> zx", flow_.D1_bnd[0], u)
    top_neumann = jnp.einsum("j, jzx -> zx", flow_.D1_bnd[-1], u)
    top = jnp.where(fourier_.mean_mask[0], u[-1], top_neumann)
    return out.at[0].set(bot).at[-1].set(top)


def _hk_minus_matvec(
    u: Array,
    flow_: CartesianFlow,
    fourier_: Fourier,
) -> Array:
    r"""Apply `$H_k^- u$` for the explicit-side Helmholtz
    operator.

    Matrix-free evaluation of `$H_k^- u$`:
    `$\tfrac{1}{\Delta t} u + (1 - c) \nu (D_2 u - k^2 u)$`
    in the interior, with identity wall rows
    (`$u|_\text{wall}$` unchanged).

    Parameters
    ----------
    u:
        Field, shape ``(Ny, Nkz, Nkx)``.
    flow\_:
        Cartesian flow data (uses ``D2``).
    fourier\_:
        Wavenumber grids (uses ``k2``).
    """
    dt = params.step.dt
    c = params.step.implicitness
    nu = 1.0 / params.phys.re
    D2u = apply_y_matrix(flow_.D2, u)
    out = (1.0 / dt) * u + (1.0 - c) * nu * (D2u - fourier_.k2 * u)
    return out.at[0].set(u[0]).at[-1].set(u[-1])


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> tuple[Array, Array]:
    r"""Kleiser-Schumann influence-matrix method.

    The y-momentum equation supplies only the *interior* Poisson
    equation for pressure; the wall BC is determined indirectly by
    enforcing continuity `$\nabla \cdot u = 0$` at the walls.

    Nine stages (six core IMM stages, then three mean-mode projections):

    1. Build the interior Poisson RHS from divergence of momentum.
    2. Solve Poisson for the particular pressure `$p_P$` with
       arbitrary (zero) Neumann BCs.
    3. Solve Helmholtz for all three particular velocity components
       `$u_{arb}, v_{arb}, w_{arb}$` against `$p_P$` (zero
       Dirichlet BCs).
    4. Compute wall divergence residual
       `$d_{\mathrm{wall}} = (D_1 v_{arb})|_{\mathrm{wall}}$`
       (since `$u = w = 0$` at walls).
    5. Apply the influence matrix
       `$\alpha = -M^{-1} d_{\mathrm{wall}}$`.
    6. Assemble the corrected pressure and all three corrected
       velocity components via Helmholtz linearity, with no
       further Helmholtz solves:

       - `$p = p_P + \alpha_1 p_1 + \alpha_2 p_2$`
       - `$v = v_{arb} + \alpha_1 v_1 + \alpha_2 v_2$`
       - `$u = u_{arb} - i k_x \Delta q$`
       - `$w = w_{arb} - i k_z \Delta q$`

       where `$\Delta q = \alpha_1 q_1 + \alpha_2 q_2$` and
       `$q_i = H_k^{-1} p_i$` (precomputed), using the
       factorisation `$u^{(i)} = -i k_x q_i$`,
       `$w^{(i)} = -i k_z q_i$` (the scalar `$-i k_x$`,
       `$-i k_z$` commute with `$H_k^{-1}$` per mode).
    7. Zero the mean-mode wall-normal velocity `$v$`.
       Continuity `$\partial v / \partial y = 0$` plus
       no-slip at both walls forces `$v \equiv 0$` there;
       the projection prevents accumulation of numerical
       noise from the Helmholtz RHS.
    8. *(optional)* If ``constant_bulk_velocity``, zero the
       mean-mode perturbation bulk velocity in the streamwise
       direction `$(\cos\theta, 0, \sin\theta)$`.
    9. *(optional)* If ``block_mean_spanwise_velocity``, zero
       the mean-mode perturbation bulk velocity in the
       spanwise direction `$(-\sin\theta, 0, \cos\theta)$`.

    Steps 7--9 are orthogonal projections and do not
    interfere; all mean-mode projections and writes go
    through ``mean_mask``.  Padding modes need no writes:
    their fields are identically zero (the forward FFT
    re-zeroes the padding slots on every evaluation), their
    placeholder-wavenumber operators are regular, and the
    IMM corrections vanish there.

    Mathematical equivalence
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The IMM is a **Schur-complement (capacitance-matrix)
    reduction**.  The coupled pressure--velocity system has a
    `$2 \times 2$` block structure with interior unknowns
    (`$I$`) and boundary unknowns (`$B$`).  The influence
    matrix `$M$` is the Schur complement
    `$S = A_{BB} - A_{BI}\,A_{II}^{-1}\,A_{IB}$`; the
    homogeneous data (``p1, p2, v1, v2, q1, q2``) are the
    columns of `$A_{II}^{-1}\,A_{IB}$`.  The correction
    (stage 6) is a **rank-2 low-rank update** to the particular
    solution -- the same algebraic structure as the **Woodbury
    matrix identity** applied to boundary conditions.  The
    bulk-velocity correction (step 8) is a **rank-1
    Sherman--Morrison update**.  Cylindrical: same structure
    with a `$1 \times 1$` Schur complement (one wall at
    `$r = 1$`).
    """
    c = params.step.implicitness
    dt = params.step.dt
    nu = 1.0 / params.phys.re

    u_n, v_n, w_n = velocity_n[0], velocity_n[1], velocity_n[2]
    Nu_n, Nv_n, Nw_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    Nu_j, Nv_j, Nw_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    mean_mask = fourier_.mean_mask

    # Horizontal spectral-derivative factors, reused across every stage.
    ikx = 1j * fourier_.kx
    ikz = 1j * fourier_.kz

    # Batch the three D1 y-derivatives into one GEMM.
    dy_stack = apply_y_matrix(flow_.D1, jnp.stack([v_n, Nv_j, Nv_n]))
    dy_v_n, dy_Nv_j, dy_Nv_n = dy_stack[0], dy_stack[1], dy_stack[2]

    # d_hat^n (discrete divergence at time n; ~0 after first step).
    d_hat_n = ikx * u_n + dy_v_n + ikz * w_n

    # Stage 1: interior pressure Poisson RHS.
    div_Nj = ikx * Nu_j + dy_Nv_j + ikz * Nw_j
    div_Nn = ikx * Nu_n + dy_Nv_n + ikz * Nw_n

    Lk_d = _lk_matvec(d_hat_n, flow_, fourier_)

    f_hat = d_hat_n / dt + c * div_Nj + (1 - c) * div_Nn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure with ZERO Neumann BCs.
    f_hat_P = f_hat.at[0].set(0.0).at[-1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P.transpose(1, 2, 0)).transpose(2, 0, 1)

    # Stage 3: Helmholtz solves for all three velocity components
    # against the particular pressure p_P (zero Dirichlet BCs).  The
    # three components share the same :math:`H_k` operator per mode,
    # so the explicit matvec, the wall-row zeroing, and the final
    # solve are all batched over the component axis — one kernel
    # launch each instead of three sequential ones.
    dx_pP = ikx * pP
    dy_pP = apply_y_matrix(flow_.D1, pP)
    dz_pP = ikz * pP
    grad_pP = jnp.stack([dx_pP, dy_pP, dz_pP])  # (3, Ny, Nkz, Nkx)

    Hk_minus_stack = jax.vmap(
        _hk_minus_matvec,
        in_axes=(0, None, None),
    )(velocity_n, flow_, fourier_)

    R_stack = Hk_minus_stack - grad_pP + c * nonlin_j + (1 - c) * nonlin_n
    R_stack = R_stack.at[:, 0].set(0.0).at[:, -1].set(0.0)

    # Zero v-component RHS at the mean mode so the Helmholtz
    # solve itself returns v = 0 there.
    R_stack = R_stack.at[1].set(jnp.where(mean_mask, 0.0, R_stack[1]))

    arb_stack = flow_.Hk_op.solve(R_stack.transpose(0, 2, 3, 1)).transpose(
        0, 3, 1, 2
    )
    u_arb, v_arb, w_arb = arb_stack[0], arb_stack[1], arb_stack[2]

    # Stage 4: wall divergence residual. At walls u=w=0 (no-slip),
    # so div u|_wall = D1 v|_wall.
    d_wall = jnp.einsum("bj, jzx -> zxb", flow_.D1_bnd, v_arb)

    # Mean-mode top-wall residual is a pressure gauge; zero it.
    d_wall = d_wall.at[..., 1].set(
        jnp.where(mean_mask[0], 0.0, d_wall[..., 1])
    )

    # Stage 5: influence matrix algebra alpha = -M_inv @ d_wall.
    alpha = -jnp.einsum("zxab, zxb -> zxa", flow_.M_inv, d_wall)
    alpha1 = alpha[..., 0][None]
    alpha2 = alpha[..., 1][None]

    # Stage 6: corrected velocity components via Helmholtz
    # linearity — no additional Helmholtz solves.  The corrected
    # pressure (pP + alpha1 p1 + alpha2 p2) is never assembled:
    # only velocity is stepped.
    v_new = v_arb + alpha1 * flow_.v1 + alpha2 * flow_.v2

    # Stage 7: zero mean-mode wall-normal velocity.
    v_new = jnp.where(mean_mask, 0.0, v_new)

    # Horizontal corrections factor through the scalar potential Δq,
    # since u^(i) = -ikx q_i and w^(i) = -ikz q_i (the -ikx, -ikz
    # scalar factors commute with Hk linearity per mode).
    q_new = alpha1 * flow_.q1 + alpha2 * flow_.q2
    u_new = u_arb - ikx * q_new
    w_new = w_arb - ikz * q_new

    if (
        params.phys.driving == "constant_bulk_velocity"
        or params.phys.block_mean_spanwise_velocity
    ):
        # Extract mean-mode velocity profiles once (shared by
        # both streamwise and spanwise corrections).
        mean_uw = extract_mean_mode(jnp.stack([u_new, w_new])).real
        mean_u, mean_w = mean_uw[0], mean_uw[1]

        u_corr = 0.0
        w_corr = 0.0

        if params.phys.driving == "constant_bulk_velocity":
            mean_us = (
                mean_u * derived_params.cos_tilt
                + mean_w * derived_params.sin_tilt
            )
            bulk_us = jnp.dot(flow_.y_weights, mean_us) / 2
            G_s = -bulk_us * flow_.H_bulk_inv * flow_.h_bulk_response
            u_corr = u_corr + G_s * derived_params.cos_tilt
            w_corr = w_corr + G_s * derived_params.sin_tilt

        if params.phys.block_mean_spanwise_velocity:
            # The streamwise correction (cos θ, sin θ) is
            # orthogonal to the spanwise direction
            # (-sin θ, cos θ), so mean_u / mean_w from
            # before the correction give the correct
            # spanwise projection.
            mean_un = (
                -mean_u * derived_params.sin_tilt
                + mean_w * derived_params.cos_tilt
            )
            bulk_un = jnp.dot(flow_.y_weights, mean_un) / 2
            G_n = -bulk_un * flow_.H_bulk_inv * flow_.h_bulk_response
            u_corr = u_corr - G_n * derived_params.sin_tilt
            w_corr = w_corr + G_n * derived_params.cos_tilt

        u_new = u_new + jnp.where(mean_mask, u_corr[:, None, None], 0.0)
        w_new = w_new + jnp.where(mean_mask, w_corr[:, None, None], 0.0)

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
    return jnp.sqrt(get_norm2(correction, fourier_.k_metric, flow_.y_weights))


# ── Stepper factory ─────────────────────────────────────────────────────


def build_cartesian_stepper(
    flow: CartesianFlow,
) -> tuple[
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array, Array, Array], tuple[Array, Array, Array]],
    Callable[[str | None], Array],
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array], tuple[Array, Array, Array, dict[str, Array]]],
]:
    """Build time-stepping functions for a Cartesian
    wall-bounded flow.

    Returns ``(predict_and_correct, iterate_correction,
    init_state_bound, predict_and_fully_correct,
    predict_and_fully_correct_measured)`` with the
    ``fourier`` and *flow* singletons already bound.
    """
    return build_wall_bounded_stepper(
        _get_rhs, _predict, _correct, _norm, fourier, flow, _get_rhs_measured
    )
