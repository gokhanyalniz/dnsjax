"""Unit tests for the SPIKE solver in :mod:`dnsjax.geometries.cartesian`.

The tests cover four concerns:

1. SPIKE factorisation + solve round-trip for a random banded matrix
   with both real and complex RHS (multi-block ``P >= 2`` partition).
2. Parity between ``PerModeBandedOperator`` (SPIKE) and
   ``DenseJAXSolver`` for `$L_k$` and `$H_k$` at the module-level
   Fourier modes, including the `$k^2 = 0$` mean-mode pin.
3. ``_lk_matvec`` matches a reference `$L_k u$` built from
   `$D_1$`/`$D_2$`.
4. ``_hk_minus_matvec`` matches a reference `$H_k^- u$` built from
   `$D_2$`.

Run as a script via ``uv run python tests/test_banded_solver.py``.
"""

from __future__ import annotations

# Enable float64 *before* any JAX array is created; otherwise JAX
# silently downcasts to float32 and the comparisons below fail.
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Mutate global ``params`` before importing any dnsjax module that
# captures values from it (``sharding.Sharding`` does so at class
# definition time).  Ny = 16 allows a genuine multi-block SPIKE
# partition (P = 2, m = 8) at fd_order = 4.
from dnsjax.parameters import params  # noqa: E402

params.phys.system = "plane-couette"
params.res.nx = 4
params.res.ny = 16
params.res.nz = 4
params.res.fd_order = 4
params.res.double_precision = True

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.fd import build_diff_matrices  # noqa: E402
from dnsjax.geometries.cartesian import (  # noqa: E402
    _build_Hk_blocks_gpu,
    _build_Hk_dense_gpu,
    _build_Lk_blocks_gpu,
    _build_Lk_dense_gpu,
    _hk_minus_matvec,
    _lk_matvec,
    fourier,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedOperator,
    _choose_block_partition,
    _spike_factor,
)

# ── helpers ──────────────────────────────────────────────────────────


def _make_random_banded(Ny: int, p: int, seed: int) -> np.ndarray:
    """Random banded matrix (half-bandwidth *p*), well-conditioned."""
    rng = np.random.default_rng(seed)
    A = np.zeros((Ny, Ny))
    for i in range(Ny):
        j_lo = max(0, i - p)
        j_hi = min(Ny, i + p + 1)
        A[i, j_lo:j_hi] = rng.standard_normal(j_hi - j_lo)
    A += 10.0 * np.eye(Ny)
    return A


def _extract_spike_blocks(
    A: np.ndarray, P: int, m: int, p: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract diagonal blocks and coupling corners from a banded matrix."""
    A_blocks = np.zeros((P, m, m))
    B_corner = np.zeros((P, p, p))
    C_corner = np.zeros((P, p, p))
    for i in range(P):
        r0 = i * m
        A_blocks[i] = A[r0 : r0 + m, r0 : r0 + m]
        if i < P - 1:
            B_corner[i] = A[r0 + m - p : r0 + m, r0 + m : r0 + m + p]
        if i > 0:
            C_corner[i] = A[r0 : r0 + p, r0 - p : r0]
    return A_blocks, B_corner, C_corner


def _build_Lk_reference(
    k2_val: float, D1: np.ndarray, D2: np.ndarray
) -> np.ndarray:
    r"""Build single-mode `$L_k$` (Neumann-BC Laplacian) in NumPy."""
    Ny = D2.shape[0]
    Lk = D2.copy() - k2_val * np.eye(Ny)
    if k2_val == 0.0:
        Lk[0, :] = 0.0
        Lk[0, 0] = 1.0
    else:
        Lk[0, :] = D1[0, :]
    Lk[-1, :] = D1[-1, :]
    return Lk


def _build_Hk_minus_reference(
    k2_val: float,
    D2: np.ndarray,
    dt: float,
    c: float,
    nu: float,
) -> np.ndarray:
    r"""Build single-mode `$H_k^-$` (explicit Helmholtz) in NumPy."""
    Ny = D2.shape[0]
    eye = np.eye(Ny)
    Hk_minus = (1.0 / dt) * eye + (1.0 - c) * nu * (D2 - k2_val * eye)
    Hk_minus[0, :] = eye[0, :]
    Hk_minus[-1, :] = eye[-1, :]
    return Hk_minus


# ── tests ────────────────────────────────────────────────────────────


def test_spike_solve_random() -> None:
    """SPIKE solve matches ``np.linalg.solve`` (real + complex RHS)."""
    Ny, p = 32, 4
    P, m = 4, 8
    A = _make_random_banded(Ny, p, seed=0)
    A_blk, B_crn, C_crn = _extract_spike_blocks(A, P, m, p)

    Nkz = params.res.nz - 1
    Nkx = params.res.nx // 2
    A_j = jnp.asarray(
        jnp.tile(jnp.asarray(A_blk)[None, None], (Nkz, Nkx, 1, 1, 1)),
        out_sharding=sharding.spec_dy_blocks_shard,
    )
    B_j = jnp.tile(jnp.asarray(B_crn)[None, None], (Nkz, Nkx, 1, 1, 1))
    C_j = jnp.tile(jnp.asarray(C_crn)[None, None], (Nkz, Nkx, 1, 1, 1))

    op = PerModeBandedOperator(*_spike_factor(A_j, B_j, C_j))

    rng = np.random.default_rng(10)

    # Real RHS.
    b = rng.standard_normal(Ny)
    rhs = jnp.asarray(
        jnp.tile(jnp.asarray(b)[None, None, :], (Nkz, Nkx, 1)),
        out_sharding=sharding.spec_imm_corr_shard,
    )
    x = np.asarray(op.solve(rhs))[0, 0]
    assert_allclose(x, np.linalg.solve(A, b), atol=1e-10, rtol=1e-10)

    # Complex RHS.
    b_c = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
    rhs_c = jnp.asarray(
        jnp.tile(jnp.asarray(b_c)[None, None, :], (Nkz, Nkx, 1)),
        out_sharding=sharding.spec_imm_corr_shard,
    )
    x_c = np.asarray(op.solve(rhs_c))[0, 0]
    assert_allclose(x_c, np.linalg.solve(A, b_c), atol=1e-10, rtol=1e-10)


def test_spike_vs_dense_on_imm_operators() -> None:
    """``PerModeBandedOperator`` matches ``DenseJAXSolver`` on Lk/Hk."""
    Ny = params.res.ny
    p = params.res.fd_order
    y = -np.cos(np.arange(Ny) * np.pi / (Ny - 1))
    D1, D2 = build_diff_matrices(y, p)
    D1_j, D2_j = jnp.asarray(D1), jnp.asarray(D2)

    dt, c, nu = 0.01, 0.5, 1.0 / 1000.0
    P_opt, m_opt = _choose_block_partition(Ny, p)

    # SPIKE path.
    Lk_A, Lk_B, Lk_C = _build_Lk_blocks_gpu(
        D1_j, D2_j, fourier.k2, fourier.k2_is_zero, p, P_opt, m_opt
    )
    Lk_banded = PerModeBandedOperator(*_spike_factor(Lk_A, Lk_B, Lk_C))

    Hk_A, Hk_B, Hk_C = _build_Hk_blocks_gpu(
        D2_j, fourier.k2, dt, c, nu, p, P_opt, m_opt
    )
    Hk_banded = PerModeBandedOperator(*_spike_factor(Hk_A, Hk_B, Hk_C))

    # Dense path (reference).
    Lk_dense = DenseJAXSolver(
        _build_Lk_dense_gpu(D1_j, D2_j, fourier.k2, fourier.k2_is_zero)
    )
    Hk_dense = DenseJAXSolver(_build_Hk_dense_gpu(D2_j, fourier.k2, dt, c, nu))

    # Solve same complex RHS with both backends.
    Nkz, Nkx = int(fourier.k2.shape[0]), int(fourier.k2.shape[1])
    rng = np.random.default_rng(20)
    b = rng.standard_normal((Nkz, Nkx, Ny)) + 1j * rng.standard_normal(
        (Nkz, Nkx, Ny)
    )
    rhs = jnp.asarray(
        jnp.asarray(b), out_sharding=sharding.spec_imm_corr_shard
    )

    x_b = np.asarray(Lk_banded.solve(rhs))
    x_d = np.asarray(Lk_dense.solve(rhs))
    assert_allclose(x_b, x_d, atol=1e-9, rtol=1e-9)

    x_b = np.asarray(Hk_banded.solve(rhs))
    x_d = np.asarray(Hk_dense.solve(rhs))
    assert_allclose(x_b, x_d, atol=1e-9, rtol=1e-9)


def test_lk_matvec_matches_reference() -> None:
    r"""``_lk_matvec`` matches reference `$L_k u$` from D1/D2."""
    Ny, p = 17, 4
    y = -np.cos(np.arange(Ny) * np.pi / (Ny - 1))
    D1, D2 = build_diff_matrices(y, p)
    D2_j = jnp.asarray(D2)
    D1_bnd_j = jnp.asarray(D1[[0, -1], :])

    rng = np.random.default_rng(30)
    for kz, kx in [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]:
        k2_val = kx**2 + kz**2
        Lk = _build_Lk_reference(k2_val, D1, D2)
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Lk @ u

        u_j = jnp.asarray(u)[None, None, :]
        k2 = jnp.asarray([[[k2_val]]])
        k2_is_zero = k2 == 0.0
        got = np.asarray(_lk_matvec(u_j, D2_j, D1_bnd_j, k2, k2_is_zero))[0, 0]
        assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_hk_minus_matvec_matches_reference() -> None:
    r"""``_hk_minus_matvec`` matches reference `$H_k^- u$` from D2."""
    Ny, p = 17, 4
    dt, c, nu = 0.01, 0.5, 1.0 / 1000.0
    y = -np.cos(np.arange(Ny) * np.pi / (Ny - 1))
    _, D2 = build_diff_matrices(y, p)
    D2_j = jnp.asarray(D2)

    rng = np.random.default_rng(40)
    for kz, kx in [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]:
        k2_val = kx**2 + kz**2
        Hk_minus = _build_Hk_minus_reference(k2_val, D2, dt, c, nu)
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Hk_minus @ u

        u_j = jnp.asarray(u)[None, None, :]
        k2 = jnp.asarray([[[k2_val]]])
        got = np.asarray(_hk_minus_matvec(u_j, D2_j, k2, dt, c, nu))[0, 0]
        assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    test_spike_solve_random()
    test_spike_vs_dense_on_imm_operators()
    test_lk_matvec_matches_reference()
    test_hk_minus_matvec_matches_reference()
    print("All banded-solver tests passed.")
