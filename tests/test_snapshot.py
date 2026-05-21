"""Round-trip tests for the zarr3 snapshot module.

Exercises the host (non-GDS) I/O path:

- save -> load equality for ``wb_native``, ``wb_y_major`` and
  ``periodic_native`` layouts;
- **np-agnostic** resume: save at one device count, load at a
  different one (clean global per-component files);
- ``load_y_slice`` matches the corresponding y-plane for a
  ``y_major`` snapshot and raises otherwise;
- ``serial`` write mode produces a byte-identical snapshot
  (true cross-process ordering needs MPI multi-process and is not
  exercised here -- single-process serial reduces to one write).

Each (system, device count) needs its own process because the
geometry/sharding singletons are captured at import time, and
multiple CPU devices are obtained via
``--xla_force_host_platform_device_count``.  The GDS path needs a
GPU + kvikIO and is not unit-tested here; it shares the same slab
offsets/layout as the host path.

Run as a script::

    uv run python tests/test_snapshot.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

# ── configuration ────────────────────────────────────────────────────

NX, NY, NZ = 8, 8, 8  # nx // 2 = 4 (shardable by 1, 2, 4)
SEED = 1234
T_SAVE, IT_SAVE = 1.5, 7

# (name, system, layout, write_mode, save_np, load_np)
CASES: list[tuple[str, str, str, str, int, int]] = [
    ("wb_native", "plane-couette", "native", "concurrent", 1, 1),
    ("wb_y_major", "plane-couette", "y_major", "concurrent", 1, 1),
    ("periodic", "kolmogorov", "native", "concurrent", 1, 1),
    ("wb_y_major np 1->2", "plane-couette", "y_major", "concurrent", 1, 2),
    ("periodic np 2->1", "kolmogorov", "native", "concurrent", 2, 1),
    ("wb_y_major serial", "plane-couette", "y_major", "serial", 1, 1),
    ("periodic serial", "kolmogorov", "native", "serial", 1, 1),
]


# ── worker (runs in its own process) ─────────────────────────────────


def _make_reference(np_mod, shape):
    """Deterministic complex128 field of *shape* from ``SEED``."""
    rng = np_mod.random.default_rng(SEED)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag).astype(np_mod.complex128)


def _worker(
    action: str,
    system: str,
    layout: str,
    write_mode: str,
    npv: int,
    d: str,
):
    """Set up singletons for *npv* CPU devices, then save or load."""
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={npv}"

    import numpy as np

    from dnsjax.parameters import padded_res, params

    params.phys.system = system
    params.res.nx = NX
    params.res.ny = NY
    params.res.nz = NZ
    params.res.double_precision = True
    params.dist.np = npv
    params.dist.platform = "cpu"
    params.outs.snapshot_layout = layout
    params.outs.snapshot_write_mode = write_mode
    padded_res.set_padded_resolution(params)

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    from jax.sharding import NamedSharding

    from dnsjax import snapshot
    from dnsjax.sharding import sharding

    shape = (3, *sharding.spec_shape)
    reference = _make_reference(np, shape)
    vshard = NamedSharding(sharding.mesh, sharding.spec_vector_shard)

    if action == "save":
        state = jax.device_put(reference, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        return

    # action == "load"
    snapshot.validate_snapshot_params(d)
    state, t, it = snapshot.load_snapshot(d)
    got = np.asarray(state)

    assert got.shape == reference.shape, (got.shape, reference.shape)
    assert np.array_equal(got, reference), "loaded state mismatch"
    assert t == T_SAVE, (t, T_SAVE)
    assert it == IT_SAVE, (it, IT_SAVE)

    is_y_major = system not in ("kolmogorov",) and layout == "y_major"
    if is_y_major:
        for y in (0, NY // 2, NY - 1):
            sl = np.asarray(snapshot.load_y_slice(d, y))
            assert np.array_equal(sl, reference[:, :, :, y]), (
                f"y_slice mismatch at y={y}"
            )
    else:
        raised = False
        try:
            snapshot.load_y_slice(d, 0)
        except ValueError:
            raised = True
        assert raised, "load_y_slice should reject non-y_major"

    print("worker-load-ok", flush=True)


# ── orchestrator ─────────────────────────────────────────────────────


def _run_worker(
    action: str,
    system: str,
    layout: str,
    write_mode: str,
    npv: int,
    d: str,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--action",
        action,
        "--system",
        system,
        "--layout",
        layout,
        "--write-mode",
        write_mode,
        "--np",
        str(npv),
        "--dir",
        d,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def _fail(name: str, stage: str, res: subprocess.CompletedProcess) -> None:
    print(f"  FAIL  {name}: {stage} exit {res.returncode}")
    print(res.stdout[-2000:] if res.stdout else "(no stdout)")
    print(res.stderr[-2000:] if res.stderr else "(no stderr)")


def run_case(
    name: str,
    system: str,
    layout: str,
    write_mode: str,
    save_np: int,
    load_np: int,
) -> bool:
    """Save then load a snapshot in separate processes."""
    with tempfile.TemporaryDirectory() as tmp:
        snap_dir = os.path.join(tmp, "snap")
        r_save = _run_worker(
            "save", system, layout, write_mode, save_np, snap_dir
        )
        if r_save.returncode != 0:
            _fail(name, "save", r_save)
            return False
        r_load = _run_worker(
            "load", system, layout, write_mode, load_np, snap_dir
        )
        if r_load.returncode != 0:
            _fail(name, "load", r_load)
            return False
    print(f"  PASS  {name}")
    return True


# ── main ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot tests")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--action", choices=["save", "load"])
    parser.add_argument("--system")
    parser.add_argument("--layout")
    parser.add_argument("--write-mode")
    parser.add_argument("--np", type=int)
    parser.add_argument("--dir")
    args = parser.parse_args()

    if args.worker:
        _worker(
            args.action,
            args.system,
            args.layout,
            args.write_mode,
            args.np,
            args.dir,
        )
        sys.exit(0)

    passed = failed = 0
    for case in CASES:
        if run_case(*case):
            passed += 1
        else:
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
