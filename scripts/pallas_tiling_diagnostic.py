"""GPU diagnostic: bisect the mode-tiled Pallas banded-kernel miscompile.

``_pallas_banded_solve`` (``solvers.py``) was correct in Pallas interpret
mode and lowered cleanly for ``cuda``, yet returned wrong/NaN results on a
**real GPU** for ``pallas_block_m1 > 1`` -- even with the cross-pass
``debug_barrier`` in place. This script isolated the cause: the **masked
partial-tile path** miscompiles on real Triton.  In a grid with a partial
boundary tile it corrupts results **across the grid (even full-tile
programs)**, **nondeterministically** (warp-scheduling dependent), for any
nontrivial kernel -- masked double-index band loads (``l_ref[i, d]``) *and*
a single-index window-carry sweep (``window_carry``).  Only trivial
copy/round-trip kernels survive; full tiles (no partial boundary anywhere)
are always correct at ``(2, 32)``.

The kernel now **pads the mode plane up to whole tiles** so it only ever
runs the correct full-tile path.  This script remains the regression
confirmation: the ``full`` probe (the real solve, which pads) passes on
partial planes, while the un-padded construct micro-kernels still XFAIL on
partial planes -- the underlying Triton bug they isolate (see the verdict
legend at the end of the run).

It runs a ladder of minimal ``pallas_call(interpret=False)`` probes -- each
exercising one construct the real kernel uses, plus alternative
formulations -- against a NumPy reference, swept over tile size, warp count
and partial-vs-full mode tiles. The first failing probe names the broken
construct; a passing alternative names the fix.

Run **on a GPU** (single device, no mpirun)::

    .venv/bin/python scripts/pallas_tiling_diagnostic.py        # full sweep
    .venv/bin/python scripts/pallas_tiling_diagnostic.py --quick

On a GPU-less box it falls back to a **lowering-only** check
(``lower(lowering_platforms=("cuda",))``) so the probes can be validated
before they reach the cluster. **Paste the full stdout back** for
diagnosis.

Probes (ladder; ``b`` is per-mode-scaled so cross-lane contamination is
visible):
  * ``copy_fori``    -- ``x_ref[i] = b_ref[i]`` in a ``fori_loop``
                        (indexed tile load+store, dynamic index).
  * ``copy_static``  -- same, unrolled Python loop (isolates the dynamic
                        index).
  * ``copy_slice``   -- ``x_ref[i, :, :, :] = b_ref[i, :, :, :]``
                        (explicit-slice store alternative).
  * ``bcast_none``   -- ``l_ref[i, 0][None] * b_ref[i]`` (the kernel's
                        ``[None]`` broadcast of ``(bm0, bm1)`` over ``k``).
  * ``bcast_auto``   -- ``l_ref[i, 0] * b_ref[i]`` (rank-aligned auto
                        broadcast alternative).
  * ``window_carry`` -- a ``fori_loop`` carrying a length-``p`` tuple of
                        ``(k, bm0, bm1)`` tiles (the sliding window +
                        ``window[1:] + (yi,)`` shift), constant coeffs.
  * ``roundtrip_nobar`` / ``roundtrip_bar`` -- forward writes ``b`` to the
                        output ref, backward reads it back and adds a
                        constant (GMEM read-after-write across passes),
                        without / with the ``debug_barrier``.
  * ``forward_only`` -- the real forward sweep ``L y = b`` -> ``y``
                        (band reads + stores + window combined).
  * ``full``         -- the real ``_pallas_banded_solve`` (set via
                        ``params.solver.pallas_*``).
  * ``full_vmap_shared`` -- the real solve **vmapped over an RHS
                        component axis with unmapped (shared) factor
                        refs**, the Cartesian shared-``Hk`` dispatch
                        (``in_axes=(None, None, 0)``): the one
                        production ``pallas_call`` batching pattern no
                        other probe or geometry exercises on a padded
                        partial plane.
"""

from __future__ import annotations

import argparse

from dnsjax.parameters import (
    configure_jax_platform,
    params,
    platform_from_argv,
)

# Select the JAX backend from --dist.platform (default cpu) BEFORE
# importing any dnsjax module that captures it (sharding / solvers), so
# ``... --dist.platform cuda`` executes the probes on a real GPU and the
# device banner is unambiguous.  It is parsed early here (before this
# script's own argparse) and re-declared in ``main`` for --help.
configure_jax_platform(platform_from_argv())

# Mutate the global params singleton before importing any dnsjax module
# that captures it.  A small wall-bounded system is enough -- the probes
# build their own arrays.
params.phys.system = "plane-couette"
params.res.nx = 4
params.res.ny = 16
params.res.nz = 4
params.res.fd_order = 4
params.res.double_precision = True

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import triton as pltriton  # noqa: E402

from dnsjax.solvers import (  # noqa: E402
    _banded_factor,
    _banded_from_dense,
    _pallas_banded_solve,
)

F64 = jnp.float64
TOL = 1e-9


# ── reference helpers (NumPy) ────────────────────────────────────────


def _make_banded(N: int, p: int, seed: int) -> np.ndarray:
    """Well-conditioned random banded matrix, half-bandwidth ``p``."""
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N))
    for i in range(N):
        lo, hi = max(0, i - p), min(N, i + p + 1)
        A[i, lo:hi] = rng.standard_normal(hi - lo)
    return A + 10.0 * np.eye(N)


def _lu_nopiv(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """No-pivot LU (unit lower ``L``, upper ``U``)."""
    N = A.shape[0]
    L, U = np.eye(N), A.astype(float).copy()
    for c in range(N):
        for r in range(c + 1, N):
            f = U[r, c] / U[c, c]
            L[r, c] = f
            U[r, :] -= f * U[c, :]
    return L, U


def _fwd_subst(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Forward substitution for a unit-lower ``L`` (``b`` is ``(N, k)``)."""
    N = L.shape[0]
    y = np.array(b, float).copy()
    for i in range(N):
        for j in range(i):
            y[i] -= L[i, j] * y[j]
    return y


def _build_inputs(N, p, k, Nkz, Nkx, seed=0):
    """Build the per-mode-scaled mode-inner inputs + the dense ``A``.

    ``L``/``U`` are mode-independent (one banded operator tiled over all
    modes); ``b[..., kz, kx] = base_b * scale[kz, kx]`` with a distinct
    per-mode ``scale`` so a lane reading another lane's data shows up.
    """
    A = _make_banded(N, p, seed)
    rng = np.random.default_rng(seed + 1)
    base_b = rng.standard_normal((N, k))
    scale = 1.0 + 0.001 * np.arange(Nkz * Nkx).reshape(Nkz, Nkx)
    b_np = base_b[:, :, None, None] * scale[None, None, :, :]
    band = _banded_from_dense(
        jnp.tile(jnp.asarray(A)[None, None], (Nkz, Nkx, 1, 1)), p
    )
    # Kernel-layout factors as **plain local arrays** -- the uncommitted
    # equivalent of ``from_banded_factors``, whose shard_map commits its
    # result to the Explicit mesh: interpret mode's indexed-store
    # discharge rejects committed operands whenever the grid needs
    # internal padding (partial planes), and this script probes the
    # *local* kernel by design (single device, kernel-local arrays --
    # the unit tests' ``_mode_inner_factors`` pattern).  ``L``/``U`` are
    # at the **true** mode plane: the micro-probes' partial tiles must
    # come from the probe grid itself (that is what they demonstrate),
    # and their NumPy references broadcast against the true-plane ``b``
    # (construction-padded factors here raised shape errors in the
    # ``bcast_*`` references on partial planes).  ``Lp``/``Up`` add the
    # whole-``(bm0, bm1)``-tile zero pad -- the stored-factor form the
    # real solve keeps -- for the ``full`` probe.
    Lo, Uo = _banded_factor(band)
    Li = jnp.moveaxis(Lo, (-2, -1), (0, 1))
    Ui = jnp.moveaxis(Uo, (-2, -1), (0, 1))
    Ui = Ui.at[:, 0].set(1.0 / Ui[:, 0])
    bm0 = params.solver.pallas_block_m0
    bm1 = params.solver.pallas_block_m1
    fac_pad = [(0, 0), (0, 0), (0, -Nkz % bm0), (0, -Nkx % bm1)]
    return {
        "A": A,
        "base_b": base_b,
        "scale": scale,
        "b": jnp.asarray(b_np),
        "L": Li,
        "U": Ui,
        "Lp": jnp.pad(Li, fac_pad),
        "Up": jnp.pad(Ui, fac_pad),
    }


def _scaled(base: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``base`` (N, k) broadcast to (N, k, Nkz, Nkx) by per-mode scale."""
    return base[:, :, None, None] * scale[None, None, :, :]


# ── Pallas runner ────────────────────────────────────────────────────


def _pallas_run(kernel, inputs, out_shape, bm0, bm1, num_warps, interpret):
    """``pallas_call`` over the mode plane with the real kernel's layout."""
    Nkz, Nkx = inputs[0].shape[-2], inputs[0].shape[-1]
    grid = (pl.cdiv(Nkz, bm0), pl.cdiv(Nkx, bm1))

    def idx(i, j):
        return (0, 0, i, j)

    in_specs = [
        pl.BlockSpec((a.shape[0], a.shape[1], bm0, bm1), idx) for a in inputs
    ]
    out_specs = pl.BlockSpec((out_shape[0], out_shape[1], bm0, bm1), idx)
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=jax.ShapeDtypeStruct(out_shape, F64),
        compiler_params=pltriton.CompilerParams(num_warps=num_warps),
        interpret=interpret,
    )(*inputs)


# ── probe construction ───────────────────────────────────────────────


def _build_probe(name, dims, bm0, bm1, num_warps, interpret=False):
    """Return ``(fn, inputs, ref_np)`` for one probe at one config.

    ``fn(*inputs)`` runs the probe's Pallas kernel (so the same callable
    serves both execute and ``lower``); ``ref_np`` is the NumPy truth.
    """
    N, p, k, Nkz, Nkx = dims
    if name in ("full", "full_vmap_shared"):
        # Set the probe's tile *before* building the inputs:
        # ``from_banded_factors`` pads the stored factors to the
        # ``params`` tile, so setting it only afterwards left the
        # factors padded for the *previous* config's tile (a smaller
        # runtime tile then drove the solve's fallback factor pad
        # negative -- the ``full (1, 1)`` partial-plane crash on the
        # cluster; the solve now also grows such factors instead of
        # raising, see ``_pallas_banded_solve``).
        params.solver.pallas_block_m0 = bm0
        params.solver.pallas_block_m1 = bm1
        params.solver.pallas_num_warps = num_warps
    d = _build_inputs(N, p, k, Nkz, Nkx)
    b, L = d["b"], d["L"]
    out_shape = (N, k, Nkz, Nkx)

    def runner(kernel, inputs):
        return lambda *ins: _pallas_run(
            kernel, ins, out_shape, bm0, bm1, num_warps, interpret
        ), inputs

    if name in ("copy_fori", "copy_static", "copy_slice"):
        if name == "copy_static":

            def kernel(b_ref, x_ref):
                for i in range(N):
                    x_ref[i] = b_ref[i]
        elif name == "copy_slice":

            def kernel(b_ref, x_ref):
                def body(i, c):
                    x_ref[i, :, :, :] = b_ref[i, :, :, :]
                    return c

                lax.fori_loop(0, N, body, 0)
        else:

            def kernel(b_ref, x_ref):
                def body(i, c):
                    x_ref[i] = b_ref[i]
                    return c

                lax.fori_loop(0, N, body, 0)

        fn, inputs = runner(kernel, (b,))
        return fn, inputs, np.asarray(b)

    if name in ("bcast_none", "bcast_auto"):
        none = name == "bcast_none"

        def kernel(l_ref, b_ref, x_ref):
            def body(i, c):
                li = l_ref[i, 0][None] if none else l_ref[i, 0]
                x_ref[i] = li * b_ref[i]
                return c

            lax.fori_loop(0, N, body, 0)

        fn, inputs = runner(kernel, (L, b))
        ref = np.asarray(L)[:, 0][:, None, :, :] * np.asarray(b)
        return fn, inputs, ref

    if name == "window_carry":
        coeff = [0.5 ** (dd + 1) for dd in range(p)]

        def kernel(b_ref, x_ref):
            zero = jnp.zeros((k, bm0, bm1), F64)

            def body(i, window):
                yi = b_ref[i]
                for dd in range(p):
                    yi = yi + coeff[dd] * window[dd]
                x_ref[i] = yi
                return window[1:] + (yi,)

            lax.fori_loop(0, N, body, (zero,) * p)

        # NumPy reference: same recurrence on the single-mode base RHS.
        base = d["base_b"]
        y = np.zeros_like(base)
        win = [np.zeros((k,)) for _ in range(p)]
        for i in range(N):
            yi = base[i].copy()
            for dd in range(p):
                yi = yi + coeff[dd] * win[dd]
            y[i] = yi
            win = win[1:] + [yi]
        fn, inputs = runner(kernel, (b,))
        return fn, inputs, _scaled(y, d["scale"])

    if name in ("roundtrip_nobar", "roundtrip_bar"):
        bar = name == "roundtrip_bar"

        def kernel(b_ref, x_ref):
            def fwd(i, c):
                x_ref[i] = b_ref[i]
                return c

            lax.fori_loop(0, N, fwd, 0)
            if bar and not interpret:  # no CPU lowering for the barrier
                pltriton.debug_barrier()

            def back(t, c):
                i = N - 1 - t
                x_ref[i] = x_ref[i] + 1.0
                return c

            lax.fori_loop(0, N, back, 0)

        fn, inputs = runner(kernel, (b,))
        return fn, inputs, np.asarray(b) + 1.0

    if name == "forward_only":

        def kernel(l_ref, b_ref, x_ref):
            zero = jnp.zeros((k, bm0, bm1), F64)

            def fwd(i, window):
                yi = b_ref[i]
                for dd in range(p):
                    yi = yi - l_ref[i, dd][None] * window[dd]
                x_ref[i] = yi
                return window[1:] + (yi,)

            lax.fori_loop(0, N, fwd, (zero,) * p)

        Ld, _ = _lu_nopiv(d["A"])
        y = _fwd_subst(Ld, d["base_b"])
        fn, inputs = runner(kernel, (L, b))
        return fn, inputs, _scaled(y, d["scale"])

    if name == "full":
        # The params tile was set before ``_build_inputs`` (above), so
        # the stored padded factors match this config; pass them (the
        # production form) rather than the true-plane slices.
        def fn(L_, U_, b_):
            return _pallas_banded_solve(L_, U_, b_, p, interpret=interpret)

        x = np.linalg.solve(d["A"], d["base_b"])
        return fn, (d["Lp"], d["Up"], b), _scaled(x, d["scale"])

    if name == "full_vmap_shared":
        # The Cartesian shared-``Hk`` dispatch: ONE operator vmapped
        # over the RHS component axis (``.solve``'s
        # ``in_axes=(None, None, 0)`` branch) -- the only production
        # pattern that batches ``pallas_call`` with **unmapped** factor
        # refs.  Stacked geometries (pipe/annular) map the factors too,
        # so on a padded partial plane this composition runs nowhere
        # else on real Triton; it is the prime suspect for a
        # Cartesian-only cross-backend divergence at ``nx = 34``
        # (``Nkx = 17 -> 32``).
        def fn(L_, U_, b3_):
            return jax.vmap(
                lambda bb: _pallas_banded_solve(
                    L_, U_, bb, p, interpret=interpret
                )
            )(b3_)

        b3 = jnp.stack([b, 2.0 * b, -0.5 * b])
        x = _scaled(np.linalg.solve(d["A"], d["base_b"]), d["scale"])
        ref = np.stack([x, 2.0 * x, -0.5 * x])
        return fn, (d["Lp"], d["Up"], b3), ref

    raise ValueError(f"unknown probe {name}")


# ── evaluation + reporting ───────────────────────────────────────────


def _evaluate(got: np.ndarray, ref: np.ndarray):
    """``(ok, max_err, n_nan, n_inf, argmax_loc)``."""
    nn, ni = int(np.isnan(got).sum()), int(np.isinf(got).sum())
    diff = np.abs(got - ref)
    big = np.where(np.isnan(diff), np.inf, diff)
    loc = np.unravel_index(int(np.argmax(big)), got.shape)
    finite = np.where(np.isnan(diff), 0.0, diff)
    max_err = float(finite.max()) if got.size else 0.0
    ok = nn == 0 and ni == 0 and max_err < TOL
    return ok, max_err, nn, ni, tuple(int(x) for x in loc)


PROBES = [
    "copy_fori",
    "copy_static",
    "copy_slice",
    "bcast_none",
    "bcast_auto",
    "window_carry",
    "roundtrip_nobar",
    "roundtrip_bar",
    "forward_only",
    "full",
    "full_vmap_shared",
]

# Probes that must pass at every config: the real solve (`full` and its
# shared-operator vmap dispatch `full_vmap_shared`, both of which pad
# the plane to whole tiles) and the trivial baselines (copy/round-trip
# survive partial tiles -- lane-local, no cross-lane mixing).
MUST_PASS = frozenset(
    {
        "copy_fori",
        "copy_static",
        "copy_slice",
        "roundtrip_nobar",
        "roundtrip_bar",
        "full",
        "full_vmap_shared",
    }
)
# Probes that run the raw Triton partial-tile bug WITHOUT padding, so they
# are *expected* to fail (XFAIL) on `partial` planes -- nondeterministically,
# and corrupting even full-tile programs in a grid with a partial boundary.
# They must still PASS on `full` planes (no partial boundary anywhere).
BUG_DEMO = frozenset(
    {"bcast_none", "bcast_auto", "window_carry", "forward_only"}
)


def _expected_fail(probe: str, mlabel: str) -> bool:
    """A bug-demo probe on a partial plane is an expected XFAIL (the raw
    Triton partial-tile miscompile, which the real solve pads around).  Any
    other failure -- a must-pass probe, or a bug-demo probe on a full plane
    -- is a real regression."""
    return probe in BUG_DEMO and mlabel == "partial"


# (bm0, bm1, num_warps)
TILES = [
    (1, 1, None),
    (1, 2, None),
    (1, 2, 1),
    (1, 32, None),
    (1, 32, 1),
    (2, 32, None),
    (2, 32, 1),
    (2, 32, 2),
]
TILES_QUICK = [(1, 1, None), (1, 32, None), (1, 32, 1), (2, 32, None)]
# (label, Nkz, Nkx): full tiles vs partial boundary tiles.
MODES = [("full", 4, 64), ("partial", 5, 40)]
MODES_QUICK = [("partial", 5, 40)]
N_FIXED, P_FIXED, K_FIXED = 17, 4, 2


def _print_env() -> None:
    import jaxlib

    print("=" * 72)
    print("Pallas tiling diagnostic")
    print(f"  jax     {jax.__version__}")
    print(f"  jaxlib  {jaxlib.__version__}")
    try:
        import triton

        print(f"  triton  {triton.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"  triton  (import failed: {e})")
    print(f"  devices {jax.devices()}")
    for dv in jax.devices():
        kind = getattr(dv, "device_kind", "?")
        print(f"    - {dv} kind={kind}")
    print(f"  default_backend  {jax.default_backend()}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="short sweep")
    ap.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend (default cpu; already applied at import via "
        "platform_from_argv).  Use cuda to execute the probes on a GPU.",
    )
    ap.add_argument(
        "--interpret",
        action="store_true",
        help="run probes in Pallas interpret mode on CPU (validates the "
        "harness/references; does NOT reproduce the GPU miscompile)",
    )
    args = ap.parse_args()

    _print_env()
    gpu = jax.default_backend() == "gpu"
    interpret = args.interpret
    execute = gpu or interpret
    if interpret:
        print(
            "INTERPRET mode -> probes run in pure-JAX interpret on CPU. "
            "Validates the harness + references only; it CANNOT reproduce "
            "the real-Triton miscompile (expect all PASS).\n"
        )
    elif gpu:
        print("GPU backend -> EXECUTING probes on real Triton.\n")
    else:
        print(
            "No GPU backend -> LOWERING-CHECK only (lower for cuda, no "
            "execute). Run on the cluster for the real diagnostic.\n"
        )

    # Clear the Explicit mesh: the indexed-ref stores otherwise hit a
    # sharding-checked discharge (see test_banded_solver.py); single
    # device, numerics unchanged.
    jax.set_mesh(None)

    tiles = TILES_QUICK if args.quick else TILES
    modes = MODES_QUICK if args.quick else MODES
    xfails: dict[str, int] = {pr: 0 for pr in PROBES}  # expected (partial)
    regs: dict[str, int] = {pr: 0 for pr in PROBES}  # unexpected regressions
    runs: dict[str, int] = {pr: 0 for pr in PROBES}
    dumped = False  # dump arrays only on the first *regression*
    first_reg = None
    results = {}  # (probe, bm0, bm1, nw, mlabel) -> ok

    for probe in PROBES:
        for mlabel, Nkz, Nkx in modes:
            for bm0, bm1, nw in tiles:
                dims = (N_FIXED, P_FIXED, K_FIXED, Nkz, Nkx)
                tag = (
                    f"{probe:16s} bm=({bm0},{bm1}) nw={str(nw):4s} "
                    f"{mlabel:7s} Nkz={Nkz} Nkx={Nkx}"
                )
                try:
                    fn, inputs, ref = _build_probe(
                        probe, dims, bm0, bm1, nw, interpret
                    )
                except Exception as e:  # noqa: BLE001
                    # A probe that cannot even be constructed is a
                    # harness regression regardless of expectations.
                    print(f"{tag} -> BUILD_ERROR: {type(e).__name__}: {e}")
                    regs[probe] += 1
                    if first_reg is None:
                        first_reg = tag
                    continue
                try:
                    if execute:
                        got = np.asarray(jax.jit(fn)(*inputs))
                        ok, me, nn, ni, loc = _evaluate(got, ref)
                        results[(probe, bm0, bm1, nw, mlabel)] = ok
                        runs[probe] += 1
                        if ok:
                            print(f"{tag} -> PASS")
                            continue
                        detail = (
                            f"max_err={me:.2e} nan={nn} inf={ni} "
                            f"@ (n,k,kz,kx)={loc}"
                        )
                        if _expected_fail(probe, mlabel):
                            # Expected: the raw Triton partial-tile bug.
                            xfails[probe] += 1
                            print(f"{tag} -> XFAIL {detail}")
                        else:
                            regs[probe] += 1
                            if first_reg is None:
                                first_reg = tag
                            print(f"{tag} -> NUMERIC {detail}")
                            if not dumped:
                                _dump(probe, got, ref)
                                dumped = True
                    else:
                        jax.jit(fn).trace(*inputs).lower(
                            lowering_platforms=("cuda",)
                        )
                        print(f"{tag} -> LOWERS_OK")
                except Exception as e:  # noqa: BLE001
                    kind = "RUNTIME_ERROR" if execute else "LOWERING_ERROR"
                    msg = str(e).replace("\n", " ")[:160]
                    # A crash is an acceptable manifestation of the raw
                    # partial-tile bug for a bug-demo probe (XFAIL);
                    # anywhere else it is a regression -- errors must
                    # not silently drop out of the verdict tally.
                    if execute and _expected_fail(probe, mlabel):
                        xfails[probe] += 1
                        print(
                            f"{tag} -> XFAIL({kind}): "
                            f"{type(e).__name__}: {msg}"
                        )
                    else:
                        regs[probe] += 1
                        if first_reg is None:
                            first_reg = tag
                        print(f"{tag} -> {kind}: {type(e).__name__}: {msg}")

    if execute:
        _summary(xfails, regs, runs, first_reg, results, modes, tiles)


def _dump(probe: str, got: np.ndarray, ref: np.ndarray) -> None:
    """Full got-vs-ref for two modes of the first regression."""
    print(f"\n  --- first-failure dump [{probe}] (component k=0) ---")
    Nkz, Nkx = got.shape[-2], got.shape[-1]

    def fmt(v: np.ndarray) -> str:
        return np.array2string(v, precision=4, max_line_width=66)

    for kz, kx in [(0, 0), (Nkz - 1, Nkx - 1)]:
        print(f"   mode (kz={kz}, kx={kx}):")
        print(f"     got = {fmt(got[:, 0, kz, kx])}")
        print(f"     ref = {fmt(ref[:, 0, kz, kx])}")
    print()


def _summary(xfails, regs, runs, first_reg, results, modes, tiles) -> None:
    print("\n" + "=" * 72)
    n_reg = sum(regs.values())
    if n_reg == 0:
        print("VERDICT: FIX CONFIRMED")
    else:
        print(f"VERDICT: REGRESSION ({n_reg})")
    print(
        "  must-pass (full + copy/round-trip) green everywhere; bug-demo "
        "(bcast_*/"
    )
    print(
        "  window_carry/forward_only) XFAIL on partial planes = the raw Triton"
    )
    print("  partial-tile bug the real solve pads around (not a regression).")

    print("\nSUMMARY (probe: reg / xfail / runs)")
    for pr in PROBES:
        flag = "   <-- REGRESSION" if regs[pr] else ""
        print(
            f"  {pr:16s} {regs[pr]} reg / {xfails[pr]} xfail / "
            f"{runs[pr]}{flag}"
        )
    if first_reg is None:
        print("\nno regressions -- the kernel fix holds (full passes all).")
    else:
        print(f"\nfirst regression: {first_reg}")

    # num_warps=1 discriminator: for each probe/mode, does forcing one
    # warp turn a failing tiled config into a pass?  (Informative for the
    # XFAIL bug-demo probes -- the partial-tile bug is warp-dependent.)
    print("\nnum_warps=1 discriminator (PASS@nw1 vs result@nwNone):")
    for pr in PROBES:
        for mlabel, _, _ in modes:
            for bm0, bm1 in [(1, 32), (2, 32)]:
                a = results.get((pr, bm0, bm1, None, mlabel))
                b = results.get((pr, bm0, bm1, 1, mlabel))
                if a is None or b is None:
                    continue
                if (not a) and b:
                    print(
                        f"  {pr:16s} {mlabel:7s} bm=({bm0},{bm1}): "
                        "FAIL@auto -> PASS@nw1  (cross-warp implicated)"
                    )
                elif (not a) and (not b):
                    print(
                        f"  {pr:16s} {mlabel:7s} bm=({bm0},{bm1}): "
                        "FAIL@auto and FAIL@nw1 (not warp-count alone)"
                    )

    # Interpretation legend.  The real solve (`full`, calling
    # `_pallas_banded_solve`) pads the mode plane to whole tiles, so it is
    # correct on partial planes; the construct micro-kernels below it do
    # NOT pad, so they still expose the raw Triton bug they isolate.
    print("\nInterpretation (pad-to-full-tiles fix in _pallas_banded_solve):")
    print(
        "  * 'full' PASS on partial planes -> dnsjax solve correct "
        "(pads to full tiles)."
    )
    print(
        "  * bug-demo XFAIL on partial -> raw Triton partial-tile bug "
        "(corrupts even"
    )
    print(
        "    full-tile programs, nondeterministically); these micro-kernels "
        "do not pad."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
