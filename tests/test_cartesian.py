"""Unit tests for the Cartesian geometry operators.

Tests cover:

1. SPIKE vs dense parity for `$L_k$` and `$H_k$` on Cartesian
   Fourier modes.
2. ``_lk_matvec`` matches a NumPy reference on CGL and custom grids.
3. ``_hk_minus_matvec`` matches a NumPy reference.
4. ``get_norm2`` matches a manual Parseval/quadrature sum.

Run as a script via ``uv run python tests/test_cartesian.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

from dnsjax.parameters import derived_params, params  # noqa: E402

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
from dnsjax.geometries.wall_bounded import get_norm2  # noqa: E402
from dnsjax.geometries.wall_bounded.cartesian import (  # noqa: E402
    _build_Hk_band_gpu,
    _build_Hk_blocks_gpu,
    _build_Hk_dense_gpu,
    _build_Lk_band_gpu,
    _build_Lk_blocks_gpu,
    _build_Lk_dense_gpu,
    _hk_minus_matvec,
    _lk_matvec,
    build_cartesian_grid,
    fourier,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _banded_factor,
    _banded_from_dense,
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

    # Solver-internal (Nkz, Nkx, 1) from field-layout (1, Nkz, Nkx).
    k2_s = fourier.k2[0, ..., None]
    mean_s = fourier.mean_mask[0, ..., None]

    # SPIKE path.
    Lk_A, Lk_B, Lk_C = _build_Lk_blocks_gpu(
        D1, D2, k2_s, mean_s, p, P_opt, m_opt
    )
    Lk_banded = _spike_factor(Lk_A, Lk_B, Lk_C)

    Hk_A, Hk_B, Hk_C = _build_Hk_blocks_gpu(
        D2, k2_s, dt, c, nu, p, P_opt, m_opt
    )
    Hk_banded = _spike_factor(Hk_A, Hk_B, Hk_C)

    # Dense path (reference).
    Lk_dense = DenseJAXSolver(_build_Lk_dense_gpu(D1, D2, k2_s, mean_s))
    Hk_dense = DenseJAXSolver(_build_Hk_dense_gpu(D2, k2_s, dt, c, nu))

    # Solve same complex RHS with both backends.
    Nkz, Nkx = int(fourier.k2.shape[1]), int(fourier.k2.shape[2])
    rng = np.random.default_rng(20)
    b = rng.standard_normal((Ny, Nkz, Nkx)) + 1j * rng.standard_normal(
        (Ny, Nkz, Nkx)
    )
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_scalar_shard)

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

        u_j = jnp.asarray(u)[:, None, None]
        fourier_ = SimpleNamespace(
            k2=jnp.asarray([[[k2_val]]]),
            mean_mask=jnp.asarray([[[k2_val]]]) == 0.0,
        )
        got = np.asarray(_lk_matvec(u_j, flow_, fourier_))[:, 0, 0]
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

        u_j = jnp.asarray(u)[:, None, None]
        fourier_ = SimpleNamespace(k2=jnp.asarray([[[k2_val]]]))
        got = np.asarray(_hk_minus_matvec(u_j, flow_, fourier_))[:, 0, 0]
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

        u_j = jnp.asarray(u)[:, None, None]
        fourier_ = SimpleNamespace(
            k2=jnp.asarray([[[k2_val]]]),
            mean_mask=jnp.asarray([[[k2_val]]]) == 0.0,
        )
        got = np.asarray(_lk_matvec(u_j, flow_, fourier_))[:, 0, 0]
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"kz={kz}, kx={kx}",
        )


def test_get_norm2_cartesian() -> None:
    r"""``get_norm2`` matches a manual Parseval/quadrature sum.

    Pins the spectral contraction: sum `$|u|^2$` over components and the
    two periodic wavenumber axes with the real-FFT multiplicity
    ``k_metric``, integrate over `$y$` with the CGL quadrature weights,
    and normalise by ``derived_params.volume_fac``.
    """
    Ny = params.res.ny
    Nkz = params.res.nz - 1
    Nkx = params.res.nx // 2

    _, _, _, y_weights = build_cartesian_grid(Ny, params.res.fd_order)

    rng = np.random.default_rng(80)
    s_shape = (3, Ny, Nkz, Nkx)
    state_np = rng.standard_normal(s_shape) + 1j * rng.standard_normal(s_shape)
    state = jnp.asarray(state_np)

    k_metric = fourier.k_metric
    got = float(get_norm2(state, k_metric, y_weights))

    # Manual reference mirroring ``get_inprod`` in _base.py.
    k_metric_np = np.asarray(k_metric)
    y_w_np = np.asarray(y_weights)
    integrand = (np.abs(state_np) ** 2 * k_metric_np).sum(axis=(0, 2, 3))
    ref = float(y_w_np @ integrand) / derived_params.volume_fac

    assert_allclose(got, ref, atol=1e-12, err_msg="get_norm2 (cartesian)")


def test_pallas_vs_dense_on_cartesian_operators() -> None:
    r"""``PerModeBandedPallasOperator`` matches ``DenseJAXSolver``.

    Validates the Pallas band assembly (``_build_{Lk,Hk}_band_gpu``):
    the banded operator equals ``banded(dense)`` exactly, and the
    no-pivot banded sweep (CPU pure-JAX path) reproduces the dense
    solve on a complex RHS.
    """
    Ny = params.res.ny
    p = params.res.fd_order
    y = -jnp.cos(jnp.arange(Ny) * jnp.pi / (Ny - 1))
    D1, D2 = build_diff_matrices(y, p)

    dt, c, nu = 0.01, 0.5, 1.0 / 1000.0
    k2_s = fourier.k2[0, ..., None]
    mean_s = fourier.mean_mask[0, ..., None]

    Lk_band = _build_Lk_band_gpu(D1, D2, k2_s, mean_s, p)
    Hk_band = _build_Hk_band_gpu(D2, k2_s, dt, c, nu, p)
    Lk_full = _build_Lk_dense_gpu(D1, D2, k2_s, mean_s)
    Hk_full = _build_Hk_dense_gpu(D2, k2_s, dt, c, nu)

    # Band assembly equals banded(dense).
    to_band = jax.vmap(jax.vmap(lambda A: _banded_from_dense(A, p)))
    assert_allclose(
        np.asarray(Lk_band), np.asarray(to_band(Lk_full)), atol=1e-12
    )
    assert_allclose(
        np.asarray(Hk_band), np.asarray(to_band(Hk_full)), atol=1e-12
    )

    # No-pivot banded solve reproduces the dense solve.
    Lk_pallas = PerModeBandedPallasOperator.from_banded_factors(
        *_banded_factor(Lk_band)
    )
    Hk_pallas = PerModeBandedPallasOperator.from_banded_factors(
        *_banded_factor(Hk_band)
    )
    Lk_dense = DenseJAXSolver(Lk_full)
    Hk_dense = DenseJAXSolver(Hk_full)

    Nkz, Nkx = int(fourier.k2.shape[1]), int(fourier.k2.shape[2])
    rng = np.random.default_rng(21)
    b = rng.standard_normal((Ny, Nkz, Nkx)) + 1j * rng.standard_normal(
        (Ny, Nkz, Nkx)
    )
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_scalar_shard)

    assert_allclose(
        np.asarray(Lk_pallas.solve(rhs)),
        np.asarray(Lk_dense.solve(rhs)),
        atol=1e-9,
        rtol=1e-9,
    )
    assert_allclose(
        np.asarray(Hk_pallas.solve(rhs)),
        np.asarray(Hk_dense.solve(rhs)),
        atol=1e-9,
        rtol=1e-9,
    )


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
