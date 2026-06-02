"""Unit tests for the Cartesian geometry operators.

Tests cover:

1. SPIKE vs dense parity for `$L_k$` and `$H_k$` on Cartesian
   Fourier modes.
2. ``_lk_matvec`` matches a NumPy reference on CGL and custom grids.
3. ``_hk_minus_matvec`` matches a NumPy reference.

Run as a script via ``uv run python tests/test_cartesian.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

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
from dnsjax.geometries.wall_bounded.cartesian import (  # noqa: E402
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
    _choose_block_partition,
    _spike_factor,
)

# ── helpers ──────────────────────────────────────────────────────────


def _build_Lk_reference(
    k2_val: float, D1: np.ndarray, D2: np.ndarray
) -> np.ndarray:
    r"""Build single-mode `$L_k$` (Neumann-BC Laplacian) in NumPy."""
    Ny = D2.shape[0]
    Lk = D2.copy() - k2_val * np.eye(Ny)
    Lk[0, :] = D1[0, :]
    if k2_val == 0.0:
        Lk[-1, :] = 0.0
        Lk[-1, -1] = 1.0
    else:
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


def _perturbed_cgl_grid(Ny: int, seed: int = 42) -> np.ndarray:
    """Monotonically increasing non-uniform grid on [-1, 1]."""
    rng = np.random.default_rng(seed)
    pts = np.sort(rng.uniform(-1, 1, Ny))
    pts[0] = -1.0
    pts[-1] = 1.0
    return pts


# ── tests ────────────────────────────────────────────────────────────


def test_spike_vs_dense_on_cartesian_operators() -> None:
    """``PerModeBandedOperator`` matches ``DenseJAXSolver`` on Lk/Hk."""
    Ny = params.res.ny
    p = params.res.fd_order
    y = -jnp.cos(jnp.arange(Ny) * jnp.pi / (Ny - 1))
    D1, D2 = build_diff_matrices(y, p)

    dt, c, nu = 0.01, 0.5, 1.0 / 1000.0
    P_opt, m_opt = _choose_block_partition(Ny, p)

    # SPIKE path.
    Lk_A, Lk_B, Lk_C = _build_Lk_blocks_gpu(
        D1, D2, fourier.k2, fourier.k2_is_zero, p, P_opt, m_opt
    )
    Lk_banded = _spike_factor(Lk_A, Lk_B, Lk_C)

    Hk_A, Hk_B, Hk_C = _build_Hk_blocks_gpu(
        D2, fourier.k2, dt, c, nu, p, P_opt, m_opt
    )
    Hk_banded = _spike_factor(Hk_A, Hk_B, Hk_C)

    # Dense path (reference).
    Lk_dense = DenseJAXSolver(
        _build_Lk_dense_gpu(D1, D2, fourier.k2, fourier.k2_is_zero)
    )
    Hk_dense = DenseJAXSolver(_build_Hk_dense_gpu(D2, fourier.k2, dt, c, nu))

    # Solve same complex RHS with both backends.
    Nkz, Nkx = int(fourier.k2.shape[0]), int(fourier.k2.shape[1])
    rng = np.random.default_rng(20)
    b = rng.standard_normal((Nkz, Nkx, Ny)) + 1j * rng.standard_normal(
        (Nkz, Nkx, Ny)
    )
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_imm_corr_shard)

    x_b = np.asarray(Lk_banded.solve(rhs))
    x_d = np.asarray(Lk_dense.solve(rhs))
    assert_allclose(x_b, x_d, atol=1e-9, rtol=1e-9)

    x_b = np.asarray(Hk_banded.solve(rhs))
    x_d = np.asarray(Hk_dense.solve(rhs))
    assert_allclose(x_b, x_d, atol=1e-9, rtol=1e-9)


def test_lk_matvec_matches_reference() -> None:
    r"""``_lk_matvec`` matches reference `$L_k u$` from D1/D2."""
    Ny, p = 17, 4
    y = -jnp.cos(jnp.arange(Ny) * jnp.pi / (Ny - 1))
    D1, D2 = build_diff_matrices(y, p)

    flow_ = SimpleNamespace(D2=D2, D1_bnd=D1[[0, -1], :])

    rng = np.random.default_rng(30)
    for kz, kx in [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]:
        k2_val = kx**2 + kz**2
        Lk = _build_Lk_reference(k2_val, np.asarray(D1), np.asarray(D2))
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Lk @ u

        u_j = jnp.asarray(u)[None, None, :]
        fourier_ = SimpleNamespace(
            k2=jnp.asarray([[[k2_val]]]),
            k2_is_zero=jnp.asarray([[[k2_val]]]) == 0.0,
        )
        got = np.asarray(_lk_matvec(u_j, flow_, fourier_))[0, 0]
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"kz={kz}, kx={kx}",
        )


def test_hk_minus_matvec_matches_reference() -> None:
    r"""``_hk_minus_matvec`` matches reference `$H_k^- u$` from D2."""
    Ny, p = 17, 4
    dt = params.step.dt
    c = params.step.implicitness
    nu = 1.0 / params.phys.re
    y = -jnp.cos(jnp.arange(Ny) * jnp.pi / (Ny - 1))
    _, D2 = build_diff_matrices(y, p)

    flow_ = SimpleNamespace(D2=D2)

    rng = np.random.default_rng(40)
    for kz, kx in [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]:
        k2_val = kx**2 + kz**2
        Hk_minus = _build_Hk_minus_reference(k2_val, np.asarray(D2), dt, c, nu)
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Hk_minus @ u

        u_j = jnp.asarray(u)[None, None, :]
        fourier_ = SimpleNamespace(k2=jnp.asarray([[[k2_val]]]))
        got = np.asarray(_hk_minus_matvec(u_j, flow_, fourier_))[0, 0]
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"kz={kz}, kx={kx}",
        )


def test_lk_matvec_on_custom_grid() -> None:
    r"""``_lk_matvec`` matches reference on a non-CGL grid."""
    Ny, p = 17, 4
    y_np = _perturbed_cgl_grid(Ny)
    y = jnp.asarray(y_np, dtype=jnp.float64)
    D1, D2 = build_diff_matrices(y, p)

    flow_ = SimpleNamespace(D2=D2, D1_bnd=D1[[0, -1], :])

    rng = np.random.default_rng(50)
    for kz, kx in [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]:
        k2_val = kx**2 + kz**2
        Lk = _build_Lk_reference(k2_val, np.asarray(D1), np.asarray(D2))
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Lk @ u

        u_j = jnp.asarray(u)[None, None, :]
        fourier_ = SimpleNamespace(
            k2=jnp.asarray([[[k2_val]]]),
            k2_is_zero=jnp.asarray([[[k2_val]]]) == 0.0,
        )
        got = np.asarray(_lk_matvec(u_j, flow_, fourier_))[0, 0]
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"kz={kz}, kx={kx}",
        )


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
