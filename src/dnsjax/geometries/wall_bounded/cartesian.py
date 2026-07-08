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
    clenshaw_curtis_weights,
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
    PerModeBandedPallasOperator,
    _assemble_banded_operator,
    _banded_diag_column,
    _banded_from_dense,
    _banded_wall_row,
    _build_pallas_operator,
)
from ._base import (
    apply_y_matrix,
    base_flow_coupling,
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


# ``clenshaw_curtis_weights`` now lives in ``dnsjax.fd`` (JAX-free,
# beside ``build_integration_weights``) so the full-CGL Cartesian and
# annular grids share it; re-exported above for callers/tests that
# import it from this module.


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
        y_weights = jnp.asarray(
            clenshaw_curtis_weights(ny), dtype=sharding.float_type
        )

    D1, D2 = build_diff_matrices(ys, fd_order)
    return ys, D1, D2, y_weights


# ── Pallas-backend banded operator builders ───────────────────────


def _build_Lk_band_gpu(
    D1: Array,
    D2: Array,
    k2: Array,
    mean_mask: Array,
    p: int,
) -> Array:
    r"""Build `$L_k$` in banded storage for the Pallas backend.

    Same Neumann-BC pressure Poisson operator
    `$L_k = D_2 - k^2 I$` as :func:`_build_Lk_dense_gpu`,
    but assembled directly in banded
    layout ``(Nkz, Nkx, Ny, 2p+1)``
    (``band[..., i, d] = L_k[..., i, i-p+d]``) from the base band
    ``_banded_from_dense(D2, p)``, with no ``(Ny, Ny)`` per mode.  The
    `$-k^2$` shift is constant across rows; Neumann `$D_1$` rows sit at
    **both** walls (rows 0 and ``Ny-1``), with a mean-mode identity pin
    at the outer wall (the only `$k^2 = 0$` system).

    Parameters
    ----------
    D1, D2:
        First/second-derivative matrices, ``(Ny, Ny)``.
    k2:
        `$k_x^2 + k_z^2$`, ``(Nkz, Nkx, 1)``.
    mean_mask:
        Mean-mode boolean mask, same shape as *k2*.
    p:
        FD order (half-bandwidth).
    """
    Ny = D2.shape[-1]
    band_D2 = _banded_from_dense(D2, p)  # (Ny, 2p+1)
    diag = -k2  # (Nkz, Nkx, 1), constant across rows
    inner = _banded_wall_row(D1[0], 0, p)  # Neumann, inner wall
    neumann_outer = _banded_wall_row(D1[-1], Ny - 1, p)  # Neumann, outer
    outer = jnp.where(
        mean_mask, _banded_diag_column(p, band_D2.dtype), neumann_outer
    )  # (Nkz, Nkx, 2p+1)
    return _assemble_banded_operator(
        band_D2, 1.0, diag, [(0, inner), (Ny - 1, outer)]
    )


def _build_Hk_band_gpu(
    D2: Array,
    k2: Array,
    dt: float,
    c: float,
    nu: float,
    p: int,
) -> Array:
    r"""Build `$H_k$` in banded storage for the Pallas backend.

    Banded analogue of :func:`_build_Hk_dense_gpu`, laid out as
    ``(Nkz, Nkx, Ny, 2p+1)``:
    `$H_k = (1/\Delta t) I - c \nu (D_2 - k^2 I)$` with Dirichlet
    no-slip identity rows at **both** walls.  The single `$H_k$` is
    shared by all three velocity components.
    """
    Ny = D2.shape[-1]
    band_D2 = _banded_from_dense(D2, p)
    diag = 1.0 / dt + c * nu * k2  # (Nkz, Nkx, 1)
    eN = _banded_diag_column(p, band_D2.dtype)  # identity wall row
    return _assemble_banded_operator(
        band_D2, -c * nu, diag, [(0, eN), (Ny - 1, eN)]
    )


def _build_Lk_dense_gpu(
    D1: Array, D2: Array, k2: Array, mean_mask: Array
) -> Array:
    """Build the Neumann-BC Laplacian `$L_k$` in dense form on GPU.

    Used only by the ``"dense"`` solver backend; allocates
    `$(N_{kz}, N_{kx}, N_y, N_y)$`.  No CPU path.

    Parameters follow :func:`_build_Lk_band_gpu` (sans ``p``);
    the output is the full dense operator.
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

_WallBoundedOp = DenseJAXSolver | PerModeBandedPallasOperator


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
    base_flow_adv_padded: Array = field(init=False)
    D1: Array = field(init=False)
    D2: Array = field(init=False)
    D1_bnd: Array = field(init=False)
    D2_bnd: Array = field(init=False)
    Lk_op: _WallBoundedOp = field(init=False)
    Hk_op: _WallBoundedOp = field(init=False)
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
        default pallas backend, `$L_k$` and `$H_k$` are
        assembled directly in banded storage
        (:func:`_build_Lk_band_gpu` /
        :func:`_build_Hk_band_gpu`) and factored by the
        setup-checked no-pivot banded LU
        (:func:`solvers._build_pallas_operator`), with no
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

        if params.solver.backend == "pallas":
            # Pallas backend: one-program-per-mode banded sweep.
            # Operators are assembled directly in banded storage (no
            # (Ny, Ny) per mode) and factored by the setup-checked
            # no-pivot banded LU (_build_pallas_operator).  Lk and the
            # single shared Hk are each one operator group; build one
            # at a time so the setup peak never holds two unfactored
            # operators at once.
            Lk_band = _build_Lk_band_gpu(self.D1, self.D2, k2_s, mean_s, p)
            self.Lk_op = _build_pallas_operator([Lk_band], "Lk")
            del Lk_band

            Hk_band = _build_Hk_band_gpu(self.D2, k2_s, dt, c, nu, p)
            self.Hk_op = _build_pallas_operator([Hk_band], "Hk")
            del Hk_band
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
        # This run-once setup stays in the mode-outer (Nkz, Nkx, Ny)
        # layout: the influence-matrix einsums below operate on it and
        # the results are transposed to field layout (Ny, Nkz, Nkx) at
        # the end.  ``.solve`` now takes a mode-inner field, so each
        # setup solve is wrapped (transpose in, transpose out) to keep
        # this layout.  FUTURE: rebuild this setup natively mode-inner to
        # drop the wrappers -- the hot path already is; here it only
        # relocates a one-time transpose, so it is deferred.
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
        p1_s = self.Lk_op.solve(e1_b.transpose(2, 0, 1)).transpose(1, 2, 0)
        p2_s = self.Lk_op.solve(e2_b.transpose(2, 0, 1)).transpose(1, 2, 0)

        rhs_v1 = -jnp.einsum("ij, zxj -> zxi", self.D1, p1_s)
        rhs_v2 = -jnp.einsum("ij, zxj -> zxi", self.D1, p2_s)
        rhs_v1 = rhs_v1.at[..., 0].set(0.0).at[..., -1].set(0.0)
        rhs_v2 = rhs_v2.at[..., 0].set(0.0).at[..., -1].set(0.0)
        v1_s = self.Hk_op.solve(rhs_v1.transpose(2, 0, 1)).transpose(1, 2, 0)
        v2_s = self.Hk_op.solve(rhs_v2.transpose(2, 0, 1)).transpose(1, 2, 0)

        q_rhs1 = p1_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
        q_rhs2 = p2_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
        q1_s = self.Hk_op.solve(q_rhs1.transpose(2, 0, 1)).transpose(1, 2, 0)
        q2_s = self.Hk_op.solve(q_rhs2.transpose(2, 0, 1)).transpose(1, 2, 0)

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

        # Mode-outer setup RHS; wrap the mode-inner ``.solve`` (run once;
        # see ``_derive_imm_homogeneous_data`` for the FUTURE note).
        h_full = self.Hk_op.solve(rhs.transpose(2, 0, 1)).transpose(1, 2, 0)

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

    # Stack (u, w) y-leading (N_y, 2, ...) so the two D1 y-derivatives
    # for the curl are one batched GEMM that contracts the leading
    # wall-normal axis transpose-free; unstack back to 3-d.
    dy_uw = apply_y_matrix(
        flow_.D1, jnp.stack([u, w], axis=1), component_axis=1
    )
    dy_u, dy_w = dy_uw[:, 0], dy_uw[:, 1]

    dx_v = 1j * fourier_.kx * v
    dz_v = 1j * fourier_.kz * v
    dx_w = 1j * fourier_.kx * w
    dz_u = 1j * fourier_.kz * u

    omega_x = dy_w - dz_v
    omega_y = dz_u - dx_w
    omega_z = dx_v - dy_u

    return jnp.array([omega_x, omega_y, omega_z])


def _l_bf(
    state: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> Array:
    r"""Linear base-flow coupling `$\mathbf{u}' \times \nabla\times
    \mathbf{U} + \mathbf{U} \times \boldsymbol{\omega}'$` (FFT-free).

    The two base-flow terms of the rotational nonlinear form (see
    :mod:`dnsjax.rhs`), evaluated entirely in spectral space: the base
    flow `$\mathbf{U}$` and its curl are `$y$`-only profiles
    (``flow.base_flow`` / ``flow.curl_base_flow``, shape
    ``(3, Ny, 1, 1)``), so multiplying by them is a
    wall-normal-pointwise multiply that does not mix the
    `$(k_x, k_z)$` modes -- no Fourier transform (and, since they are
    `$k_x = k_z = 0$`, no dealiasing either, so this equals the
    corresponding physical-space terms inside :func:`get_nonlin`
    exactly).  `$\boldsymbol{\omega}' = \nabla\times\mathbf{u}'$`
    reuses :func:`_curl_fn` (also FFT-free).

    The CN/AB2 scheme (``step.scheme == "cnab2"``) advances this
    *linear* term implicitly (Crank-Nicolson) while the pure
    self-advection `$\mathbf{u}' \times \boldsymbol{\omega}' =
    \text{get\_rhs} - L_{bf}$` stays explicit (Adams-Bashforth) --
    the base-flow coupling carries the stiff wall-normal derivative
    `$U\,\partial_y u'$` that would otherwise impose a `$1/N^2$`
    time-step limit on the wall-clustered grid.  See the
    ``step_cnab2`` docstring in :mod:`dnsjax.timestep`.

    In a moving frame (``derived_params.u_grid``) the convective
    frame term `$+ i k_x U_{grid} \mathbf{u}'$` -- the same
    expression ``_get_rhs_core`` adds -- is included here, so CN/AB2
    integrates it implicitly and the explicit split stays the pure
    self-advection.

    With ``params.step.implicit_mean_coupling`` (default on) the
    *instantaneous mean-flow* coupling `$L_{mf} = \mathbf{u}' \times
    \nabla\times\bar{\mathbf{u}}' + \bar{\mathbf{u}}' \times
    \boldsymbol{\omega}'$` is folded in by adding the mean profiles
    onto the base-flow profiles (the coupling is linear in the
    profile pair).  `$\bar{\mathbf{u}}' =$` ``extract_mean_mode(u')``
    is a ``psum`` (FFT-free), and `$\nabla\times\bar{\mathbf{u}}' =
    \overline{\boldsymbol{\omega}}'$` because the curl is linear and
    mode-diagonal -- no extra derivative needed.  See the
    ``TimeStepping`` docstring in :mod:`dnsjax.parameters` for the
    split-consistency argument.
    """
    omega = _curl_fn(state, fourier_, flow_)
    # base_flow / curl_base_flow are (3, Ny, 1, 1); broadcast over
    # (k_z, k_x).
    base = flow_.base_flow
    curl_base = flow_.curl_base_flow
    if params.step.implicit_mean_coupling:
        base = base + extract_mean_mode(state)[:, :, None, None]
        curl_base = curl_base + extract_mean_mode(omega)[:, :, None, None]
    l_bf = base_flow_coupling(state, omega, base, curl_base)
    # Moving frame: the convective-form frame term (the same
    # expression ``_get_rhs_core`` adds) belongs to the linear
    # coupling, so CN/AB2 integrates it implicitly and the explicit
    # split stays the pure self-advection.
    u_grid = derived_params.u_grid
    if u_grid == 0:
        return l_bf
    return l_bf + (1j * u_grid) * fourier_.kx * state


# Per-direction CFL column names, matching the physical-space
# component order (u, v, w) = (x, y, z).
CFL_NAMES: tuple[str, str, str] = ("CFL_x", "CFL_y", "CFL_z")


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
    measure_fn: Callable[[Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    r"""Evaluate non-linear RHS terms (optionally measured).

    In a moving frame (``derived_params.u_grid`` `$= U_{grid} \ne
    0$`) the convective-form frame term `$+ i k_x U_{grid}
    \mathbf{u}'$` is added spectrally -- mode-diagonal and
    divergence-free, so the pressure projection is untouched (see
    :func:`~dnsjax.geometries.wall_bounded._base.pad_base_flow`).
    """
    rhs = get_nonlin(
        state,
        flow_.base_flow_padded,
        flow_.curl_base_flow_padded,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier_, flow_),
        measure_fn,
    )
    u_grid = derived_params.u_grid
    if u_grid == 0:
        return rhs
    frame = (1j * u_grid) * fourier_.kx * state
    if measure_fn is None:
        return rhs + frame
    nonlin, measurements = rhs
    return nonlin + frame, measurements


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
            flow_.base_flow_adv_padded,
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

    # Batch the three D1 y-derivatives into one GEMM, stacked y-leading
    # (N_y, 3, ...) so the contraction is transpose-free; unstack to 3-d.
    dy_stack = apply_y_matrix(
        flow_.D1, jnp.stack([v_n, Nv_j, Nv_n], axis=1), component_axis=1
    )
    dy_v_n, dy_Nv_j, dy_Nv_n = dy_stack[:, 0], dy_stack[:, 1], dy_stack[:, 2]

    # d_hat^n (discrete divergence at time n; ~0 after first step).
    d_hat_n = ikx * u_n + dy_v_n + ikz * w_n

    # Stage 1: interior pressure Poisson RHS.
    div_Nj = ikx * Nu_j + dy_Nv_j + ikz * Nw_j
    div_Nn = ikx * Nu_n + dy_Nv_n + ikz * Nw_n

    Lk_d = _lk_matvec(d_hat_n, flow_, fourier_)

    f_hat = d_hat_n / dt + c * div_Nj + (1 - c) * div_Nn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure with ZERO Neumann BCs.
    f_hat_P = f_hat.at[0].set(0.0).at[-1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: Helmholtz solves for all three velocity components
    # against the particular pressure p_P (zero Dirichlet BCs).  The
    # three components share the same :math:`H_k` operator per mode,
    # so the explicit matvec, the wall-row zeroing, and the final
    # solve are all batched over the component axis — one kernel
    # launch each instead of three sequential ones.
    #
    # This Hk path stays **component-leading** (unlike the y-leading
    # curl/divergence matvecs above): it has a single D2 GEMM (the
    # vmapped _hk_minus_matvec), and velocity_n / nonlin_j / nonlin_n
    # all arrive component-leading, so a y-leading conversion would add
    # three transposes to remove the one matvec's two -- a net loss.
    # (Cylindrical/annular convert theirs -- several batched matvecs to
    # amortise; see those modules.)
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

    arb_stack = flow_.Hk_op.solve(R_stack)
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
    Callable[[Array, Array], tuple[Array, Array, Array, Array]],
    Callable[
        [Array, Array], tuple[Array, Array, Array, Array, dict[str, Array]]
    ],
]:
    """Build time-stepping functions for a Cartesian
    wall-bounded flow.

    Returns ``(predict_and_correct, iterate_correction,
    init_state_bound, predict_and_fully_correct,
    predict_and_fully_correct_measured, step_cnab2,
    step_cnab2_measured)`` with the ``fourier`` and *flow*
    singletons already bound.  ``_l_bf`` (the FFT-free base-flow
    coupling) is passed so the CN/AB2 scheme treats it implicitly.
    """
    return build_wall_bounded_stepper(
        _get_rhs,
        _predict,
        _correct,
        _norm,
        fourier,
        flow,
        _get_rhs_measured,
        _l_bf,
    )
