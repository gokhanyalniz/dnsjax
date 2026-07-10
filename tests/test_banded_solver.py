"""Unit tests for the geometry-independent linear solvers.

Pallas banded backend (``PerModeBandedPallasOperator``):

1. Pure-JAX banded-sweep path and Pallas interpret-mode parity vs
   dense over an ``fd_order`` sweep, real + complex RHS, single +
   stacked operators (built via ``from_banded_factors`` into the
   mode-inner layout).
2. The interpret-parity test sweeps two mode-plane sizes so the
   ``(bm0, bm1)`` tile does *not* divide the plane, exercising the
   kernel's pad-to-whole-tiles path (pad/crop checked against the
   CPU sweep).
3. ``test_pallas_cuda_lowering``: compile-only guard (no GPU needed)
   -- lowers the mode-tiled kernel for ``cuda`` and asserts a Triton
   custom call, catching the f64-TMA / non-power-of-two /
   value-slice lowering regressions and the padded-plane lowering at
   IR generation.
4. ``_build_pallas_operator``: a healthy operator builds and solves;
   a no-pivot breakdown or genuine element growth hard-errors; an
   above-tolerance residual with benign growth prints the
   ill-conditioning notice and proceeds.

The interpret/lowering tests call ``_pallas_banded_solve`` directly
with local, uncommitted factors (``_mode_inner_factors``; in
production the kernel runs only inside the ``.solve`` shard_map
region, where arrays are local by construction) and clear the
Explicit mesh (``jax.set_mesh(None)``): the kernel's indexed ref
stores discharge to a sharding-checked ``dynamic_update_slice`` only
in interpret mode.  Real-GPU execution and perf (tile tuning) are
deferred to the ``gpu-validation-pallas-banded`` plan, not this
suite.  Multi-device solve coverage (per-shard factor padding on a
(2, 2) mesh) lives in ``test_banded_solver_sharded.py``.

Geometry-specific operator tests live in ``test_cartesian.py``,
``test_cylindrical.py``, and ``test_annular.py``.

Run as a script via ``uv run python tests/test_banded_solver.py``.
"""

from __future__ import annotations

# Select the JAX backend from --dist.platform (default cpu) and enable
# float64 *before* importing any dnsjax module that captures the platform
# (``sharding.Sharding`` does so at class-definition time) or creating any
# JAX array (float64 avoids a silent float32 downcast that fails the
# comparisons below).  ``--dist.platform cuda`` then executes the real
# Pallas kernels on a GPU.
from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
from dnsjax.parameters import (
    params,
)

configure_jax_platform(platform_from_argv())

# Mutate global ``params`` before importing any dnsjax module that
# captures values from it.
params.phys.system = "plane-couette"
params.res.nx = 4
params.res.ny = 16
params.res.nz = 4
params.res.fd_order = 4
params.res.double_precision = True

import contextlib  # noqa: E402
import io  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    PerModeBandedPallasOperator,
    _banded_factor,
    _banded_from_dense,
    _banded_solve_batched,
    _build_pallas_operator,
    _pallas_banded_solve,
    _stack_pallas_operators,
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


# ── Pallas banded backend tests (CPU: pure-JAX path + interpret) ──────


def _tile_modes(arr: np.ndarray, Nkz: int, Nkx: int) -> jnp.ndarray:
    """Tile a single operator/RHS over the ``(Nkz, Nkx)`` mode axes."""
    return jnp.tile(jnp.asarray(arr)[None, None], (Nkz, Nkx) + (1,) * arr.ndim)


def _pallas_op_from_dense(
    A: np.ndarray, p: int, Nkz: int, Nkx: int
) -> PerModeBandedPallasOperator:
    """Build a Pallas banded operator from one dense banded matrix."""
    band = _banded_from_dense(_tile_modes(A, Nkz, Nkx), p)
    L, U = _banded_factor(band)
    return PerModeBandedPallasOperator.from_banded_factors(L, U)


def _mode_inner_factors(Lo: jnp.ndarray, Uo: jnp.ndarray):
    """Kernel-layout (mode-inner, reciprocated-diagonal) factors from
    the mode-outer ``_banded_factor`` output, as plain local arrays.

    Bypasses ``from_banded_factors``, whose shard_map commits the
    result to the Explicit mesh: the direct ``_pallas_banded_solve``
    calls below unit-test the *local* kernel function with local,
    uncommitted inputs (in production the kernel only ever runs inside
    the ``.solve`` shard_map region).  Leaving the factors unpadded
    also exercises the kernel's internal pad-to-whole-tiles fallback.
    """
    Li = jnp.moveaxis(Lo, (-2, -1), (0, 1))
    Ui = jnp.moveaxis(Uo, (-2, -1), (0, 1))
    return Li, Ui.at[:, 0].set(1.0 / Ui[:, 0])


def test_pallas_banded_matches_dense() -> None:
    """Pallas banded operator (CPU path) matches ``np.linalg.solve``
    across a sweep of half-bandwidth ``p``, real and complex RHS."""
    Nkz, Nkx = params.res.nz - 1, params.res.nx // 2
    for p in (2, 4, 6):
        Ny = 5 * p
        A = _make_random_banded(Ny, p, seed=p)
        op = _pallas_op_from_dense(A, p, Nkz, Nkx)
        rng = np.random.default_rng(100 + p)

        b = rng.standard_normal(Ny)
        rhs = jnp.tile(jnp.asarray(b)[:, None, None], (1, Nkz, Nkx))
        x = np.asarray(op.solve(rhs))[:, 0, 0]
        assert_allclose(x, np.linalg.solve(A, b), atol=1e-9, rtol=1e-9)

        bc = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        rhs_c = jnp.tile(jnp.asarray(bc)[:, None, None], (1, Nkz, Nkx))
        x_c = np.asarray(op.solve(rhs_c))[:, 0, 0]
        assert_allclose(x_c, np.linalg.solve(A, bc), atol=1e-9, rtol=1e-9)


def test_pallas_factors_prepadded_to_tiles() -> None:
    """``from_banded_factors`` stores whole-tile factors; ``.solve``
    keeps the true-plane contract.

    The stored mode plane must be rounded up to the ``(bm0, bm1)``
    Pallas tile at construction (so no per-solve factor pad/copy
    remains -- see the ``from_banded_factors`` docstring), while
    ``.solve`` still takes and returns the true (non-tiling) plane and
    matches the dense oracle on its last true mode (adjacent to the
    padding).
    """
    Nkz, Nkx = params.res.nz - 1, params.res.nx // 2  # (3, 2): no tile
    bm0 = params.solver.pallas_block_m0
    bm1 = params.solver.pallas_block_m1
    assert Nkz % bm0 != 0 or Nkx % bm1 != 0  # plane must not tile
    p = 4
    Ny = 5 * p
    A = _make_random_banded(Ny, p, seed=11)
    op = _pallas_op_from_dense(A, p, Nkz, Nkx)
    assert op.L.shape[2] % bm0 == 0 and op.L.shape[3] % bm1 == 0
    assert op.U.shape[2] % bm0 == 0 and op.U.shape[3] % bm1 == 0
    assert op.L.shape[2] >= Nkz and op.L.shape[3] >= Nkx

    b = np.random.default_rng(12).standard_normal(Ny)
    rhs = jnp.tile(jnp.asarray(b)[:, None, None], (1, Nkz, Nkx))
    x = np.asarray(op.solve(rhs))
    assert x.shape == (Ny, Nkz, Nkx)
    assert_allclose(x[:, -1, -1], np.linalg.solve(A, b), atol=1e-9, rtol=1e-9)


def test_pallas_interpret_matches_cpu_path() -> None:
    """The Pallas kernel (interpret mode) reproduces the pure-JAX
    banded sweep used on CPU.

    Swept over half-bandwidth ``p``, wall-normal size ``Ny``, real
    (``k=1``) / complex-as-real (``k=2``) RHS, and two mode-plane sizes --
    exercising the mode-tiled forward/back ``fori_loop`` sweep (band
    offsets, the reciprocated-diagonal vs super-band split-by-index, the
    y stash) and, crucially, the **pad-to-full-tiles** path: the
    ``(bm0, bm1) = (2, 32)`` tile (now the default, set explicitly here)
    divides neither ``(3, 2)`` nor ``(5, 40)``, so the kernel zero-pads
    the mode plane up to whole tiles, solves, and crops -- the pad/crop
    result is checked against the CPU sweep.  Factors are converted to the
    kernel's mode-inner layout (``from_banded_factors``); RHS/solution are
    moved between mode-outer and mode-inner around the kernel.

    Run with the global Explicit mesh cleared: the kernel's indexed ref
    stores discharge (interpret only) to ``dynamic_update_slice``, which
    rejects the ``{Explicit}`` vs ``{}`` sharding pair under that mesh.
    The real Triton path lowers the store natively (no discharge), so
    this is an interpret-mode artifact; the mesh is a trivial 1-device
    mesh here, so clearing it leaves the numerics unchanged."""
    rng = np.random.default_rng(3)
    orig_mesh = sharding.mesh
    orig_bm = (params.solver.pallas_block_m0, params.solver.pallas_block_m1)
    params.solver.pallas_block_m0 = 2
    params.solver.pallas_block_m1 = 32
    jax.set_mesh(None)
    try:
        mode_dims = [(params.res.nz - 1, params.res.nx // 2), (5, 40)]
        for Nkz, Nkx in mode_dims:
            for p in (2, 3, 4, 6):
                for Ny in (16, 20):
                    A = _make_random_banded(Ny, p, seed=7 + p)
                    Lo, Uo = _banded_factor(
                        _banded_from_dense(_tile_modes(A, Nkz, Nkx), p)
                    )
                    Li, Ui = _mode_inner_factors(Lo, Uo)
                    for k in (1, 2):
                        b = jnp.asarray(rng.standard_normal((Nkz, Nkx, Ny, k)))
                        x_cpu = _banded_solve_batched(Lo, Uo, b, p)
                        bi = jnp.moveaxis(b, (0, 1), (-2, -1))
                        x_pl = _pallas_banded_solve(
                            Li, Ui, bi, p, interpret=True
                        )
                        x_pl = jnp.moveaxis(x_pl, (0, 1), (-2, -1))
                        assert_allclose(
                            np.asarray(x_pl),
                            np.asarray(x_cpu),
                            atol=1e-12,
                            rtol=0,
                        )
    finally:
        jax.set_mesh(orig_mesh)
        params.solver.pallas_block_m0 = orig_bm[0]
        params.solver.pallas_block_m1 = orig_bm[1]


def test_pallas_cuda_lowering() -> None:
    """The Triton kernel lowers for the GPU target (compile-only).

    Lowers ``_pallas_banded_solve(interpret=False)`` for ``cuda`` and
    asserts a Triton custom call is produced.  This is the compile stage
    that rejected the earlier kernel designs (f64 TMA, non-power-of-two
    block loads, value slices / reversal / scan ``xs``); the mode-tiled
    ``fori_loop`` kernel passes it.  Triton IR generation needs no GPU,
    so this runs on the CPU dev box -- a regression guard for the exact
    class of errors hit on the cluster.  ``Ny`` and ``p`` are
    deliberately non-powers-of-two (the band axes), and the
    ``(bm0, bm1) = (2, 32)`` tile (now the default, set explicitly here)
    does not divide ``(Nkz, Nkx)``, so the kernel zero-pads the plane up to
    whole tiles; this guards the padded-plane lowering.  The factors are in
    the kernel's mode-inner layout; the mesh is cleared as in the interpret
    test."""
    Nkz, Nkx = params.res.nz - 1, params.res.nx // 2
    p, Ny, k = 3, 17, 2  # both non-power-of-two
    orig_mesh = sharding.mesh
    orig_bm = (params.solver.pallas_block_m0, params.solver.pallas_block_m1)
    params.solver.pallas_block_m0 = 2
    params.solver.pallas_block_m1 = 32
    jax.set_mesh(None)
    try:
        A = _make_random_banded(Ny, p, seed=1)
        Li, Ui = _mode_inner_factors(
            *_banded_factor(_banded_from_dense(_tile_modes(A, Nkz, Nkx), p))
        )
        b = jnp.zeros((Ny, k, Nkz, Nkx))  # mode-inner RHS

        def solve(L: jnp.ndarray, U: jnp.ndarray, b: jnp.ndarray):
            return _pallas_banded_solve(L, U, b, p, interpret=False)

        lowered = (
            jax.jit(solve).trace(Li, Ui, b).lower(lowering_platforms=("cuda",))
        )
        assert "triton" in lowered.as_text().lower()
    finally:
        jax.set_mesh(orig_mesh)
        params.solver.pallas_block_m0 = orig_bm[0]
        params.solver.pallas_block_m1 = orig_bm[1]


def test_pallas_cuda_lowering_sharded_solve() -> None:
    """The full ``.solve`` shard_map region lowers for cuda
    (compile-only, no GPU).

    Forces the Pallas-kernel branch (``solvers._force_kernel_path``) so
    tracing on a CPU box reaches ``pallas_call`` *inside* the ``.solve``
    ``shard_map`` -- the composition ``test_pallas_cuda_lowering`` above
    cannot cover (it calls the kernel standalone), and where trace-time
    rules differ: under shard_map's default ``check_vma=True`` the
    kernel's ``ShapeDtypeStruct`` out-shape must carry a
    ``manual_axis_type`` or tracing raises.  That failure mode crashed
    every real-GPU pallas run at flow construction while the whole CPU
    suite stayed green (the CPU branch never calls ``pallas_call``);
    ``.solve`` now opts out via ``check_vma=False`` (the region is
    communication-free).  This guards the composition end to end --
    stored pre-padded factors, complex re/im split, live Explicit mesh.
    """
    import dnsjax.solvers as solvers_mod

    Nkz, Nkx = params.res.nz - 1, params.res.nx // 2
    p, Ny = 4, 16
    A = _make_random_banded(Ny, p, seed=21)
    op = _pallas_op_from_dense(A, p, Nkz, Nkx)
    rng = np.random.default_rng(22)
    bc = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
    rhs = jnp.tile(jnp.asarray(bc)[:, None, None], (1, Nkz, Nkx))

    solvers_mod._force_kernel_path = True
    try:
        lowered = (
            jax.jit(lambda r: op.solve(r))
            .trace(rhs)
            .lower(lowering_platforms=("cuda",))
        )
        assert "triton" in lowered.as_text().lower()
    finally:
        solvers_mod._force_kernel_path = False


def test_pallas_stacked_operators() -> None:
    """Stacked multi-component Pallas operator solves each component
    with its own factors."""
    Nkz, Nkx = params.res.nz - 1, params.res.nx // 2
    p, Ny = 4, 16
    As = [_make_random_banded(Ny, p, seed=s) for s in (1, 2, 3)]
    stk = _stack_pallas_operators(
        *[_pallas_op_from_dense(A, p, Nkz, Nkx) for A in As]
    )
    rng = np.random.default_rng(11)
    bs = [rng.standard_normal(Ny) for _ in range(3)]
    rhs = jnp.stack(
        [jnp.tile(jnp.asarray(b)[:, None, None], (1, Nkz, Nkx)) for b in bs]
    )  # (3, Ny, Nkz, Nkx)
    X = np.asarray(stk.solve(rhs))
    for c in range(3):
        assert_allclose(
            X[c, :, 0, 0], np.linalg.solve(As[c], bs[c]), atol=1e-9, rtol=1e-9
        )


def test_build_pallas_operator_checks() -> None:
    """``_build_pallas_operator``: a healthy operator builds a working
    Pallas operator; a no-pivot breakdown (zero leading pivot) and
    genuine element growth (tiny leading pivot) hard-error; an
    above-tolerance residual with benign growth prints the
    ill-conditioning notice and still builds."""
    Nkz, Nkx = params.res.nz - 1, params.res.nx // 2
    p, Ny = 4, 16
    A = _make_random_banded(Ny, p, seed=0)
    band = _banded_from_dense(_tile_modes(A, Nkz, Nkx), p)
    rng = np.random.default_rng(5)
    b = rng.standard_normal(Ny)
    rhs = jnp.tile(jnp.asarray(b)[:, None, None], (1, Nkz, Nkx))
    ref = np.linalg.solve(A, b)

    op = _build_pallas_operator([band], "auto")
    assert isinstance(op, PerModeBandedPallasOperator)
    assert_allclose(np.asarray(op.solve(rhs))[:, 0, 0], ref, atol=1e-9)

    # Zero leading pivot: the no-pivot LU breaks down (non-finite
    # factors) -> hard error.
    A_bad = _make_random_banded(Ny, p, seed=0)
    A_bad[0, 0] = 0.0
    band_bad = _banded_from_dense(_tile_modes(A_bad, Nkz, Nkx), p)
    try:
        _build_pallas_operator([band_bad], "breakdown")
    except RuntimeError as err:
        assert "unstable" in str(err)
    else:
        raise AssertionError("zero-pivot breakdown did not raise")

    # Tiny leading pivot: finite factors, but the elimination
    # multiplies through 1/A[0,0] -> explosive element growth.
    A_grow = _make_random_banded(Ny, p, seed=1)
    A_grow[0, 0] = 1e-12
    band_grow = _banded_from_dense(_tile_modes(A_grow, Nkz, Nkx), p)
    try:
        _build_pallas_operator([band_grow], "growth")
    except RuntimeError as err:
        assert "unstable" in str(err)
    else:
        raise AssertionError("element growth did not raise")

    # Residual above tolerance with benign growth (tolerance squeezed
    # below any attainable float64 residual): the ill-conditioning
    # notice is printed and a working Pallas operator is still built.
    tol0 = params.solver.pallas_stability_tol
    params.solver.pallas_stability_tol = 1e-300
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            op_cond = _build_pallas_operator([band], "cond")
    finally:
        params.solver.pallas_stability_tol = tol0
    assert isinstance(op_cond, PerModeBandedPallasOperator)
    assert "ill-conditioned" in buf.getvalue()
    assert_allclose(np.asarray(op_cond.solve(rhs))[:, 0, 0], ref, atol=1e-9)


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
