"""Multi-device (sharded) Pallas banded-solve tests (offline).

Runs on 4 forced host CPU devices with a ``(np0, np1) = (2, 2)``
Explicit mesh (the ``test_snapshot.py`` offline pattern: ``params``
mutated before importing ``sharding``).  Covers the class the
single-device suite (``test_banded_solver.py``) structurally cannot:

1. **Per-shard factor pre-padding**: on the kernel path
   ``from_banded_factors`` pads each device's *local* mode-plane block
   to whole ``(bm0, bm1)`` tiles inside a ``shard_map``, so the stored
   global plane is the **sum of local roundups** (not the global
   roundup) -- asserted on a plane whose local blocks do not tile.  A
   CPU run stores the true plane instead (it never launches the kernel
   grid); both storages are asserted here.
2. **shard_map-local ``.solve``** on sharded mode axes: dense-oracle
   parity for real + complex RHS, single + stacked operators, and both
   ``component_axis`` layouts, plus the result sharding matching the
   RHS sharding.  This is the exact path that raised
   ``ShardingTypeError`` when the tile padding was attempted on the
   global view (slicing a sharded mode axis back to the true plane
   needs collectives; the shard_map body makes all pad/crop
   bookkeeping local).

On CPU the solve takes the pure-JAX local sweep (the oracle path);
kernel numerics are pinned single-device by the interpret tests.  The
``mpirun``-based guards for the same class are the ``*-mpi-pad``
entries of ``test_random_smoke.py`` and ``test_laminar_smoke.py``
``--np`` variants.  A cuda-lowering guard of the *sharded kernel*
path cannot be built on a CPU box: the body's backend dispatch
(``jax.default_backend()``) correctly traces the CPU sweep here even
when lowering for ``cuda`` (verified -- the AOT lowering itself
succeeds on the mesh), so real multi-GPU kernel execution is the
cluster-validation item.

Run as a script via
``uv run python tests/test_banded_solver_sharded.py``.
"""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(line_buffering=True)

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

# Mutate global ``params`` before importing any dnsjax module that
# captures values from it (``sharding.Sharding`` does so at class
# definition time).  nz = 8 keeps nz_padded = 12 divisible by np1 = 2.
from dnsjax.parameters import params  # noqa: E402

params.phys.system = "plane-couette"
params.res.nx = 8
params.res.ny = 16
params.res.nz = 8
params.res.fd_order = 4
params.res.double_precision = True
params.dist.np0 = 2
params.dist.np1 = 2

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.solvers import (  # noqa: E402
    PerModeBandedPallasOperator,
    _banded_factor,
    _banded_from_dense,
    _stack_pallas_operators,
)

# Test mode plane: local blocks (3, 2) per device -> global (6, 4).
# With the default (bm0, bm1) = (2, 32) tile neither local axis tiles
# evenly, so both are per-shard padded: local (4, 32) -> global (8, 64).
NKZ, NKX = 6, 4
NY, FD_P = 20, 4

_BAND_SPEC = P("np0", "np1", None, None)  # mode-outer band / factors
_RHS_SPEC = P(None, "np0", "np1")  # mode-inner spectral field


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


def _sharded_pallas_op(A: np.ndarray, p: int) -> PerModeBandedPallasOperator:
    """Build a Pallas operator on the sharded (NKZ, NKX) mode plane."""
    band = _banded_from_dense(
        jnp.tile(jnp.asarray(A)[None, None], (NKZ, NKX, 1, 1)), p
    )
    band = jax.device_put(band, _BAND_SPEC)
    L, U = _banded_factor(band)
    return PerModeBandedPallasOperator.from_banded_factors(L, U)


def _tiled_rhs(vec: np.ndarray) -> jnp.ndarray:
    """Tile a length-NY vector over the sharded mode plane."""
    return jax.device_put(
        jnp.tile(jnp.asarray(vec)[:, None, None], (1, NKZ, NKX)), _RHS_SPEC
    )


def test_per_shard_padded_factor_plane() -> None:
    """On the kernel path the stored factor plane is the sum of
    per-shard tile roundups; on the CPU one it is the true plane.

    The roundup is per *shard*, not global -- each device's kernel grid
    covers its own local block.  A CPU run never launches that grid, so
    it stores the true plane instead (``_kernel_path``); this asserts
    both, since only the sharded mesh can tell a sum-of-local roundup
    from a global one.
    """
    import dnsjax.solvers as solvers_mod

    bm0 = params.solver.pallas_block_m0
    bm1 = params.solver.pallas_block_m1
    nkz_loc, nkx_loc = NKZ // 2, NKX // 2
    assert nkz_loc % bm0 != 0 or nkx_loc % bm1 != 0  # locals must not tile

    solvers_mod._force_kernel_path = True
    try:
        op_gpu = _sharded_pallas_op(
            _make_random_banded(NY, FD_P, seed=0), FD_P
        )
    finally:
        solvers_mod._force_kernel_path = False

    def _roundup(n: int, b: int) -> int:
        return -(-n // b) * b

    expect = (2 * _roundup(nkz_loc, bm0), 2 * _roundup(nkx_loc, bm1))
    assert op_gpu.L.shape[2:] == expect, (op_gpu.L.shape, expect)
    assert op_gpu.U.shape[2:] == expect, (op_gpu.U.shape, expect)

    op = _sharded_pallas_op(_make_random_banded(NY, FD_P, seed=0), FD_P)
    assert op.L.shape[2:] == (NKZ, NKX), op.L.shape
    assert op.U.shape[2:] == (NKZ, NKX), op.U.shape


def test_sharded_solve_matches_dense() -> None:
    """Sharded ``.solve`` matches ``np.linalg.solve``; result keeps the
    RHS sharding.  Real + complex RHS, single operator."""
    A = _make_random_banded(NY, FD_P, seed=1)
    op = _sharded_pallas_op(A, FD_P)
    rng = np.random.default_rng(2)

    for b in (
        rng.standard_normal(NY),
        rng.standard_normal(NY) + 1j * rng.standard_normal(NY),
    ):
        rhs = _tiled_rhs(b)
        x = op.solve(rhs)
        assert x.shape == (NY, NKZ, NKX)
        assert x.sharding.spec == rhs.sharding.spec, x.sharding
        ref = np.linalg.solve(A, b)
        x_np = np.asarray(x)
        # Every mode holds the same system; check the four shard
        # corners (each adjacent to a per-shard padding boundary).
        for i in (0, NKZ // 2 - 1, NKZ // 2, NKZ - 1):
            for j in (0, NKX // 2 - 1, NKX // 2, NKX - 1):
                assert_allclose(
                    x_np[:, i, j],
                    ref,
                    atol=1e-9,
                    rtol=1e-9,
                    err_msg=f"mode ({i}, {j})",
                )


def test_sharded_stacked_component_axes() -> None:
    """Stacked operators on the sharded plane solve each component with
    its own factors, for both ``component_axis`` layouts."""
    A0 = _make_random_banded(NY, FD_P, seed=3)
    A1 = _make_random_banded(NY, FD_P, seed=4)
    op = _stack_pallas_operators(
        _sharded_pallas_op(A0, FD_P), _sharded_pallas_op(A1, FD_P)
    )
    rng = np.random.default_rng(5)
    b0 = rng.standard_normal(NY) + 1j * rng.standard_normal(NY)
    b1 = rng.standard_normal(NY) + 1j * rng.standard_normal(NY)
    ref0 = np.linalg.solve(A0, b0)
    ref1 = np.linalg.solve(A1, b1)

    # component_axis = 0: (C, NY, NKZ, NKX).
    rhs0 = jnp.stack([_tiled_rhs(b0), _tiled_rhs(b1)])
    x0 = np.asarray(op.solve(rhs0, component_axis=0))
    assert_allclose(x0[0, :, -1, -1], ref0, atol=1e-9, rtol=1e-9)
    assert_allclose(x0[1, :, -1, -1], ref1, atol=1e-9, rtol=1e-9)

    # component_axis = 1: (NY, C, NKZ, NKX) — the IMM's y-leading Hk
    # layout.
    rhs1 = jnp.stack([_tiled_rhs(b0), _tiled_rhs(b1)], axis=1)
    x1 = np.asarray(op.solve(rhs1, component_axis=1))
    assert_allclose(x1[:, 0, 0, 0], ref0, atol=1e-9, rtol=1e-9)
    assert_allclose(x1[:, 1, 0, 0], ref1, atol=1e-9, rtol=1e-9)


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(
        "Sharded Pallas banded-solve tests: offline, 4 forced CPU "
        "devices simulating a (2, 2) mesh (the device banner above is "
        "CPU by design; real multi-GPU kernel execution is the "
        "cluster-validation item -- see the module docstring).",
        flush=True,
    )
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
