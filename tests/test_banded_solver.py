"""Unit tests for the banded LU helpers in :mod:`dnsjax.geometries.cartesian`.

The tests cover four concerns:

1. Round-trip accuracy of ``_banded_lu_numpy`` +
   ``_banded_solve_device`` against ``np.linalg.solve`` for a random
   banded matrix (real RHS).
2. ``PerModeBandedOperator`` on a complex RHS using the split
   real/imag path.
3. Parity between ``PerModeBandedOperator`` and ``DenseJAXSolver``
   for `$L_k$` and `$H_k$` at representative modes including the
   `$k^2 = 0$` mean-mode pin.
4. ``_lk_matvec`` and ``_hk_minus_matvec`` match the dense
   ``build_Lk_neumann @ u`` / ``build_Hk_dirichlet[1] @ u``.

Run as a script via ``uv run python tests/test_banded_solver.py``.
"""

from __future__ import annotations

# Enable float64 *before* any JAX array is created; otherwise JAX
# silently downcasts to float32 and the comparisons below fail.
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Mutate global ``params`` before importing any dnsjax module that
# captures values from it (``sharding.Sharding`` does so at class
# definition time).  Small resolutions keep the module-level
# ``PlaneCouetteFlow()`` singleton cheap.
from dnsjax.parameters import params  # noqa: E402

params.phys.system = "plane-couette"
params.res.nx = 4
params.res.ny = 17
params.res.nz = 4
params.res.fd_order = 4
params.res.double_precision = True

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.fd import build_diff_matrices  # noqa: E402
from dnsjax.geometries.cartesian import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedOperator,
    _banded_lu_numpy,
    _banded_solve_device,
    build_Hk_dirichlet,
    build_Lk_neumann,
)


def _make_random_banded(Ny: int, kl: int, ku: int, seed: int) -> np.ndarray:
    """Random banded matrix with a strong diagonal (well-conditioned)."""
    rng = np.random.default_rng(seed)
    A = np.zeros((Ny, Ny))
    for i in range(Ny):
        j_lo = max(0, i - kl)
        j_hi = min(Ny, i + ku + 1)
        A[i, j_lo:j_hi] = rng.standard_normal(j_hi - j_lo)
    # Diagonal shift avoids pathological conditioning.
    A += 10.0 * np.eye(Ny)
    return A


def test_banded_solve_random_real() -> None:
    """``_banded_solve_device`` matches ``np.linalg.solve`` (real RHS)."""
    Ny, kl, ku = 16, 4, 4
    A = _make_random_banded(Ny, kl, ku, seed=0)
    rng = np.random.default_rng(10)
    b = rng.standard_normal(Ny)

    ab, ipiv = _banded_lu_numpy(A, kl, ku)
    x = np.asarray(
        _banded_solve_device(
            jnp.asarray(ab), jnp.asarray(ipiv), jnp.asarray(b), kl, ku
        )
    )
    x_ref = np.linalg.solve(A, b)
    assert_allclose(x, x_ref, atol=1e-10, rtol=1e-10)


def test_banded_operator_complex_rhs() -> None:
    """Complex RHS via the split path in ``PerModeBandedOperator``."""
    Ny, kl, ku = 16, 4, 4
    A = _make_random_banded(Ny, kl, ku, seed=1)
    rng = np.random.default_rng(11)
    b = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)

    ab, ipiv = _banded_lu_numpy(A, kl, ku)
    op = PerModeBandedOperator(
        ab=jnp.asarray(ab)[None, None],
        ipiv=jnp.asarray(ipiv)[None, None],
    )
    rhs = jnp.asarray(b)[None, None, :]
    x = np.asarray(op.solve(rhs))[0, 0]
    x_ref = np.linalg.solve(A, b)
    assert_allclose(x, x_ref, atol=1e-10, rtol=1e-10)


def test_banded_vs_dense_on_imm_operators() -> None:
    """``PerModeBandedOperator`` matches ``DenseJAXSolver`` on Lk/Hk."""
    Ny = 17
    p = 4
    kl = ku = p
    y = -np.cos(np.arange(Ny) * np.pi / (Ny - 1))
    D1, D2 = build_diff_matrices(y, p)

    dt, c, nu = 0.01, 0.5, 1.0 / 1000.0
    modes = [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]

    rng = np.random.default_rng(20)
    for kz, kx in modes:
        k2 = kx**2 + kz**2
        Lk = build_Lk_neumann(k2, D1, D2)
        Hk, _ = build_Hk_dirichlet(k2, D2, dt, c, nu)

        # Banded path.
        Lk_ab, Lk_piv = _banded_lu_numpy(Lk, kl, ku)
        Hk_ab, Hk_piv = _banded_lu_numpy(Hk, kl, ku)
        Lk_op = PerModeBandedOperator(
            ab=jnp.asarray(Lk_ab)[None, None],
            ipiv=jnp.asarray(Lk_piv)[None, None],
        )
        Hk_op = PerModeBandedOperator(
            ab=jnp.asarray(Hk_ab)[None, None],
            ipiv=jnp.asarray(Hk_piv)[None, None],
        )

        # Dense path (reference).
        Lk_dense = DenseJAXSolver(jnp.asarray(Lk)[None, None])
        Hk_dense = DenseJAXSolver(jnp.asarray(Hk)[None, None])

        b = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        rhs = jnp.asarray(b)[None, None, :]

        x_banded = np.asarray(Lk_op.solve(rhs))[0, 0]
        x_dense = np.asarray(Lk_dense.solve(rhs))[0, 0]
        assert_allclose(x_banded, x_dense, atol=1e-9, rtol=1e-9)

        x_banded = np.asarray(Hk_op.solve(rhs))[0, 0]
        x_dense = np.asarray(Hk_dense.solve(rhs))[0, 0]
        assert_allclose(x_banded, x_dense, atol=1e-9, rtol=1e-9)


def test_lk_matvec_matches_dense() -> None:
    """``_lk_matvec`` matches ``build_Lk_neumann(...) @ u``."""
    from dnsjax.flows.plane_couette import _lk_matvec

    Ny, p = 17, 4
    y = -np.cos(np.arange(Ny) * np.pi / (Ny - 1))
    D1, D2 = build_diff_matrices(y, p)
    D1_bnd = D1[[0, -1], :]
    D2_j = jnp.asarray(D2)
    D1_bnd_j = jnp.asarray(D1_bnd)

    rng = np.random.default_rng(30)
    modes = [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]
    for kz, kx in modes:
        k2_val = kx**2 + kz**2
        Lk = build_Lk_neumann(k2_val, D1, D2)
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Lk @ u

        u_j = jnp.asarray(u)[None, None, :]
        k2 = jnp.asarray([[[k2_val]]])
        k2_is_zero = k2 == 0.0
        got = np.asarray(
            _lk_matvec(u_j, D2_j, D1_bnd_j, k2, k2_is_zero)
        )[0, 0]
        assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_hk_minus_matvec_matches_dense() -> None:
    """``_hk_minus_matvec`` matches ``build_Hk_dirichlet[1] @ u``."""
    from dnsjax.flows.plane_couette import _hk_minus_matvec

    Ny, p = 17, 4
    dt, c, nu = 0.01, 0.5, 1.0 / 1000.0
    y = -np.cos(np.arange(Ny) * np.pi / (Ny - 1))
    _, D2 = build_diff_matrices(y, p)
    D2_j = jnp.asarray(D2)

    rng = np.random.default_rng(40)
    modes = [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]
    for kz, kx in modes:
        k2_val = kx**2 + kz**2
        _, Hk_minus = build_Hk_dirichlet(k2_val, D2, dt, c, nu)
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Hk_minus @ u

        u_j = jnp.asarray(u)[None, None, :]
        k2 = jnp.asarray([[[k2_val]]])
        got = np.asarray(_hk_minus_matvec(u_j, D2_j, k2, dt, c, nu))[0, 0]
        assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    test_banded_solve_random_real()
    test_banded_operator_complex_rhs()
    test_banded_vs_dense_on_imm_operators()
    test_lk_matvec_matches_dense()
    test_hk_minus_matvec_matches_dense()
    print("All banded-solver tests passed.")
