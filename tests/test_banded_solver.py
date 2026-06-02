"""Unit tests for the geometry-independent SPIKE solver.

Tests that SPIKE factorisation + solve round-trips correctly
for a random banded matrix with both real and complex RHS
(multi-block ``P >= 2`` partition).

Geometry-specific operator tests live in ``test_cartesian.py``
and ``test_cylindrical.py``.

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

from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import _spike_factor  # noqa: E402

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


# ── tests ────────────────────────────────────────────────────────────


def _run_spike_solve(
    A: np.ndarray,
    P: int,
    m: int,
    p: int,
    block_thomas: bool,
    seed_rhs: int = 10,
) -> None:
    """Check SPIKE solve matches np.linalg.solve (both paths)."""
    Ny = P * m
    A_blk, B_crn, C_crn = _extract_spike_blocks(A, P, m, p)

    Nkz = params.res.nz - 1
    Nkx = params.res.nx // 2
    A_j = jax.device_put(
        jnp.tile(jnp.asarray(A_blk)[None, None], (Nkz, Nkx, 1, 1, 1)),
        sharding.spec_dy_blocks_shard,
    )
    B_j = jnp.tile(jnp.asarray(B_crn)[None, None], (Nkz, Nkx, 1, 1, 1))
    C_j = jnp.tile(jnp.asarray(C_crn)[None, None], (Nkz, Nkx, 1, 1, 1))

    op = _spike_factor(A_j, B_j, C_j, block_thomas=block_thomas)

    rng = np.random.default_rng(seed_rhs)
    b = rng.standard_normal(Ny)
    rhs = jax.device_put(
        jnp.tile(jnp.asarray(b)[None, None, :], (Nkz, Nkx, 1)),
        sharding.spec_imm_corr_shard,
    )
    x = np.asarray(op.solve(rhs))[0, 0]
    assert_allclose(x, np.linalg.solve(A, b), atol=1e-10, rtol=1e-10)

    b_c = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
    rhs_c = jax.device_put(
        jnp.tile(jnp.asarray(b_c)[None, None, :], (Nkz, Nkx, 1)),
        sharding.spec_imm_corr_shard,
    )
    x_c = np.asarray(op.solve(rhs_c))[0, 0]
    assert_allclose(x_c, np.linalg.solve(A, b_c), atol=1e-10, rtol=1e-10)


def test_spike_solve_random() -> None:
    """SPIKE solve (dense reduced) matches np.linalg.solve."""
    Ny, p = 32, 4
    P, m = 4, 8
    A = _make_random_banded(Ny, p, seed=0)
    _run_spike_solve(A, P, m, p, block_thomas=False)


def test_spike_solve_block_thomas() -> None:
    """SPIKE solve (block-Thomas reduced) matches np.linalg.solve."""
    Ny, p = 32, 4
    P, m = 4, 8
    A = _make_random_banded(Ny, p, seed=0)
    _run_spike_solve(A, P, m, p, block_thomas=True)


def test_spike_solve_block_thomas_p2() -> None:
    """Block-Thomas with minimum P=2."""
    Ny, p = 16, 4
    P, m = 2, 8
    A = _make_random_banded(Ny, p, seed=42)
    _run_spike_solve(A, P, m, p, block_thomas=True)


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
