"""Round-trip tests for the single-file (tar) snapshot module.

Exercises the host (non-GDS) I/O path:

- save -> load equality for ``walled`` and
  ``periodic`` layouts;
- **np-agnostic** resume: save at one ``(np0, np1)``
  configuration, load at a different one (clean global
  per-component chunks with true modes only);
- the snapshot is a single uncompressed tar wrapping a zarr3 store,
  readable with **standard tools and no dnsjax**: the metadata
  member parses as plain JSON (stdlib ``tarfile`` + ``json``), and
  after ``tar xf`` the ``state/`` store reads back via TensorStore
  with exactly the stored data;
- ``load_y_slice`` matches the corresponding y-plane for a
  ``walled`` snapshot and raises otherwise;
- ``serial`` write mode produces a valid snapshot (true
  cross-process ordering needs MPI multi-process and is not
  exercised here -- single-process serial reduces to one write);
- 2D mesh round-trips: save and load with ``np0 > 1``, including
  padding-mode stripping and re-padding;
- the ``isnap`` lineage index round-trips through the metadata, and the
  optional ``_dnsjax_stats.json`` member is written when stats are
  supplied and omitted otherwise.

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
ISNAP_SAVE = 7  # snapshot-lineage index round-tripped via metadata
STATS_SAVE = {"D": -3.5, "E": 1.25}  # embedded _dnsjax_stats.json values

# (name, system, layout, write_mode, save_np, load_np,
#  save_np0, load_np0)
CASES: list[tuple[str, str, str, str, int, int, int, int]] = [
    # ── 1D (np0=1) ──
    ("walled", "plane-couette", "walled", "concurrent", 1, 1, 1, 1),
    ("periodic", "kolmogorov", "periodic", "concurrent", 1, 1, 1, 1),
    (
        "walled np 1->2",
        "plane-couette",
        "walled",
        "concurrent",
        1,
        2,
        1,
        1,
    ),
    ("periodic np 2->1", "kolmogorov", "periodic", "concurrent", 2, 1, 1, 1),
    (
        "walled serial",
        "plane-couette",
        "walled",
        "serial",
        1,
        1,
        1,
        1,
    ),
    ("periodic serial", "kolmogorov", "periodic", "serial", 1, 1, 1, 1),
    # ── other wall-bounded geometries (same `walled` on-disk path) ──
    ("pipe", "pipe", "walled", "concurrent", 1, 1, 1, 1),
    ("pipe np 1->2", "pipe", "walled", "concurrent", 1, 2, 1, 1),
    (
        "taylor-couette",
        "taylor-couette",
        "walled",
        "concurrent",
        1,
        1,
        1,
        1,
    ),
    (
        "taylor-couette np 1->2",
        "taylor-couette",
        "walled",
        "concurrent",
        1,
        2,
        1,
        1,
    ),
    # ── viscoelastic (9-component state: 3 velocity + 6 tensor) ──
    (
        "viscoelastic-dean",
        "viscoelastic-dean",
        "walled",
        "concurrent",
        1,
        1,
        1,
        1,
    ),
    (
        "viscoelastic-dean np 1->2",
        "viscoelastic-dean",
        "walled",
        "concurrent",
        1,
        2,
        1,
        1,
    ),
    # ── 2D (np0 > 1) ──
    ("walled 2D", "plane-couette", "walled", "concurrent", 4, 4, 2, 2),
    ("periodic 2D", "kolmogorov", "periodic", "concurrent", 4, 4, 2, 2),
    # ── cross-mesh resume ──
    (
        "periodic 1D->2D",
        "kolmogorov",
        "periodic",
        "concurrent",
        1,
        4,
        1,
        2,
    ),
    (
        "walled 1D->2D",
        "plane-couette",
        "walled",
        "concurrent",
        1,
        4,
        1,
        2,
    ),
    (
        "walled 2D->1D",
        "plane-couette",
        "walled",
        "concurrent",
        4,
        4,
        2,
        1,
    ),
]

# Periodic systems (must match dnsjax.parameters.periodic_systems).
_PERIODIC = {"kolmogorov", "waleffe", "decaying-box"}

# Viscoelastic systems carry 9 state components (3 velocity + 6
# symmetric conformation-tensor); must match ``snapshot._n_components``.
_VISCOELASTIC = {"viscoelastic-dean"}


def _n_comp(system: str) -> int:
    """Number of stacked state components for *system* (3 or 9)."""
    return 9 if system in _VISCOELASTIC else 3


# ── worker (runs in its own process) ─────────────────────────────────


def _make_reference(np_mod, shape):
    """Deterministic complex128 field of *shape* from ``SEED``."""
    rng = np_mod.random.default_rng(SEED)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag).astype(np_mod.complex128)


def _true_shape(system: str, ny: int) -> tuple[int, ...]:
    """True (unpadded) spectral vector shape."""
    kz, kx = NZ - 1, NX // 2
    nc = _n_comp(system)
    if system in _PERIODIC:
        return (nc, ny - 1, kz, kx)
    return (nc, ny, kz, kx)


def _embed_true_into_padded(true_arr, padded_shape, system):
    """Embed a true-shaped array into a zero-padded array."""
    import numpy as np

    out = np.zeros(padded_shape, dtype=true_arr.dtype)
    kz, kx = NZ - 1, NX // 2
    if system in _PERIODIC:
        out[:, :, :kz, :kx] = true_arr
    else:
        out[:, :, :kz, :kx] = true_arr
    return out


def _on_disk_from_true(ref_true, system):
    """Map a true-shaped reference into the on-disk ``(3, A, kx, B)``
    layout that the zarr3 chunks store (see ``snapshot._extract_*``).

    ``walled`` stores ``comp[y, kx, kz]`` (last two axes swapped);
    ``periodic`` stores ``comp[kz, kx, ky]`` (native ``(ky, kz, kx)``).
    """
    import numpy as np

    if system in _PERIODIC:
        # (3, ky, kz, kx) -> (3, kz, kx, ky)
        return np.transpose(ref_true, (0, 2, 3, 1)).copy()
    # (3, y, kz, kx) -> (3, y, kx, kz)
    return np.transpose(ref_true, (0, 1, 3, 2)).copy()


def _read_state_with_tensorstore(state_dir):
    """Read the extracted zarr3 ``state/`` store without dnsjax."""
    import numpy as np
    import tensorstore as ts

    arr = (
        ts.open(
            {
                "driver": "zarr3",
                "kvstore": {"driver": "file", "path": state_dir},
            }
        )
        .result()
        .read()
        .result()
    )
    return np.asarray(arr)


def _check_standard_tools(d, ref_true, system):
    """The snapshot is a single tar readable with stdlib + zarr3.

    Validates the user-facing guarantee: ``_dnsjax_meta.json`` parses
    with the standard library alone, and ``tar xf`` + a zarr3 reader
    (TensorStore) recovers exactly the stored component data.
    """
    import json
    import os
    import tarfile
    import tempfile

    import numpy as np

    assert os.path.isfile(d), f"snapshot must be a single file: {d}"
    assert tarfile.is_tarfile(d), "snapshot must be a valid tar archive"

    with tarfile.open(d, "r") as tf:
        names = set(tf.getnames())
        expected = {"_dnsjax_meta.json", "state/zarr.json"} | {
            f"state/c/{i}/0/0/0" for i in range(_n_comp(system))
        }
        assert expected <= names, (sorted(names), sorted(expected))
        # stdlib-only metadata read (no dnsjax).
        meta = json.loads(tf.extractfile("_dnsjax_meta.json").read())
        assert meta["format_version"] == 3, meta["format_version"]

    with tempfile.TemporaryDirectory() as ex:
        with tarfile.open(d, "r") as tf:
            tf.extractall(ex, filter="data")
        on_disk = _read_state_with_tensorstore(os.path.join(ex, "state"))

    assert list(on_disk.shape) == meta["on_disk_shape"], (
        on_disk.shape,
        meta["on_disk_shape"],
    )
    expected_on_disk = _on_disk_from_true(ref_true, system)
    assert np.array_equal(on_disk, expected_on_disk), (
        "TensorStore read of the extracted zarr3 store does not match "
        "the stored data"
    )


def _worker(
    action: str,
    system: str,
    layout: str,
    write_mode: str,
    npv: int,
    np0: int,
    d: str,
    ny_override: int | None = None,
):
    """Set up singletons for *npv* CPU devices, then save or load."""
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={npv}"

    import numpy as np

    from dnsjax.parameters import padded_res, params

    params.phys.system = system
    if system == "taylor-couette":
        # Taylor-Couette needs inner/outer Reynolds numbers and a radius
        # ratio set before the geometry singletons build; the snapshot
        # round-trip itself is geometry-independent, so fixed values
        # suffice (save and load workers reconstruct identically).
        params.phys.re1 = 100.0
        params.phys.re2 = 0.0
        params.geo.eta = 0.5
    params.res.nx = NX
    params.res.ny = ny_override if ny_override is not None else NY
    params.res.nz = NZ
    params.res.double_precision = True
    params.dist.np0 = np0
    params.dist.np1 = npv // np0
    params.dist.platform = "cpu"
    params.outs.snapshot_write_mode = write_mode
    padded_res.set_padded_resolution(params)

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    from jax.sharding import NamedSharding

    from dnsjax import snapshot
    from dnsjax.sharding import sharding

    padded_shape = (_n_comp(system), *sharding.spec_shape)
    ny = ny_override if ny_override is not None else NY
    tshape = _true_shape(system, ny)
    vshard = NamedSharding(sharding.mesh, sharding.spec_vector_shard)

    if action == "save":
        ref_true = _make_reference(np, tshape)
        state_np = _embed_true_into_padded(ref_true, padded_shape, system)
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        return

    if action == "save_stats":
        import jax.numpy as jnp

        ref_true = _make_reference(np, tshape)
        state_np = _embed_true_into_padded(ref_true, padded_shape, system)
        state = jax.device_put(state_np, vshard)
        # Embedded stats are (replicated) device scalars in production;
        # exercise the float() host conversion in ``save_snapshot``.
        stats = {k: jnp.asarray(v) for k, v in STATS_SAVE.items()}
        snapshot.save_snapshot(
            state, T_SAVE, IT_SAVE, d, stats=stats, isnap=ISNAP_SAVE
        )
        # A second snapshot with no stats: the member must be omitted and
        # ``isnap`` defaults to 0.
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d + ".nostats")
        print("worker-save-stats-ok", flush=True)
        return

    if action == "save_poly":
        from dnsjax.parameters import derived_params

        y = np.asarray(
            -np.cos(np.arange(ny, dtype=np.float64) * np.pi / (ny - 1))
        )
        derived_params.wall_normal_grid = y.tolist()
        poly = 1 - y**2
        state_np = np.zeros(padded_shape, dtype=np.complex128)
        state_np[0, :, 0, 0] = poly
        state_np[1, :, 0, 0] = poly * 0.5
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        return

    if action == "load_interp":
        from dnsjax.__main__ import _interpolate_if_needed
        from dnsjax.parameters import derived_params

        ny = params.res.ny
        y_curr = -np.cos(np.arange(ny, dtype=np.float64) * np.pi / (ny - 1))
        derived_params.wall_normal_grid = y_curr.tolist()

        snapshot.validate_snapshot_params(d)
        state, t, it = snapshot.load_snapshot(d)
        state = _interpolate_if_needed(
            state,
            os.path.join(d),
            snapshot.read_metadata,
            sharding,
            jax.numpy,
        )
        got = np.asarray(state)
        expected_poly = 1 - y_curr**2
        np.testing.assert_allclose(
            got[0, :, 0, 0].real, expected_poly, atol=1e-10
        )
        np.testing.assert_allclose(
            got[1, :, 0, 0].real, expected_poly * 0.5, atol=1e-10
        )
        assert got[0, 0, 0, 0].real == 0.0
        assert got[0, -1, 0, 0].real == 0.0
        assert t == T_SAVE
        print("worker-load-interp-ok", flush=True)
        return

    # action == "load"
    ref_true = _make_reference(np, tshape)
    reference = _embed_true_into_padded(ref_true, padded_shape, system)
    snapshot.validate_snapshot_params(d)
    state, t, it = snapshot.load_snapshot(d)
    got = np.asarray(state)

    assert got.shape == reference.shape, (got.shape, reference.shape)
    assert np.array_equal(got, reference), "loaded state mismatch"
    assert t == T_SAVE, (t, T_SAVE)
    assert it == IT_SAVE, (it, IT_SAVE)

    # Single uncompressed tar, readable with standard tools / no dnsjax.
    _check_standard_tools(d, ref_true, system)

    if system not in _PERIODIC:
        for yi in (0, ny // 2, ny - 1):
            sl = np.asarray(snapshot.load_y_slice(d, yi))
            assert np.array_equal(sl, ref_true[:, yi]), (
                f"y_slice mismatch at y={yi}"
            )
    else:
        raised = False
        try:
            snapshot.load_y_slice(d, 0)
        except ValueError:
            raised = True
        assert raised, "load_y_slice should reject periodic"

    print("worker-load-ok", flush=True)


# ── orchestrator ─────────────────────────────────────────────────────


def _run_worker(
    action: str,
    system: str,
    layout: str,
    write_mode: str,
    npv: int,
    d: str,
    ny: int | None = None,
    np0: int = 1,
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
        "--np0",
        str(np0),
        "--dir",
        d,
    ]
    if ny is not None:
        cmd.extend(["--ny", str(ny)])
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
    save_np0: int = 1,
    load_np0: int = 1,
) -> bool:
    """Save then load a snapshot in separate processes."""
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save",
            system,
            layout,
            write_mode,
            save_np,
            snap_path,
            np0=save_np0,
        )
        if r_save.returncode != 0:
            _fail(name, "save", r_save)
            return False
        r_load = _run_worker(
            "load",
            system,
            layout,
            write_mode,
            load_np,
            snap_path,
            np0=load_np0,
        )
        if r_load.returncode != 0:
            _fail(name, "load", r_load)
            return False
    print(f"  PASS  {name}")
    return True


def run_stats_isnap_case() -> bool:
    """``save_snapshot`` embeds ``_dnsjax_stats.json`` + ``isnap``, and
    omits the stats member (with ``isnap`` defaulting to 0) when no stats
    are supplied.  Verified with the standard library alone (no dnsjax)."""
    import json
    import tarfile

    name = "isnap + embedded stats"
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap.tar")
        r = _run_worker(
            "save_stats", "plane-couette", "walled", "concurrent", 1, snap
        )
        if r.returncode != 0:
            _fail(name, "save_stats", r)
            return False

        with tarfile.open(snap, "r") as tf:
            names = set(tf.getnames())
            assert "_dnsjax_stats.json" in names, sorted(names)
            meta = json.loads(tf.extractfile("_dnsjax_meta.json").read())
            stats = json.loads(tf.extractfile("_dnsjax_stats.json").read())
        assert meta["isnap"] == ISNAP_SAVE, meta["isnap"]
        assert stats == STATS_SAVE, stats

        with tarfile.open(snap + ".nostats", "r") as tf:
            names = set(tf.getnames())
            assert "_dnsjax_stats.json" not in names, sorted(names)
            meta = json.loads(tf.extractfile("_dnsjax_meta.json").read())
        assert meta["isnap"] == 0, meta["isnap"]
    print(f"  PASS  {name}")
    return True


def run_ny_mismatch_case(
    save_np: int = 1,
    save_np0: int = 1,
    load_np: int = 1,
    load_np0: int = 1,
    label: str = "",
) -> bool:
    """Save at ny=8, load at ny=16 with interpolation."""
    name = f"wb ny 8->16 interp{label}"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save_poly",
            "plane-couette",
            "walled",
            "concurrent",
            save_np,
            snap_path,
            ny=8,
            np0=save_np0,
        )
        if r_save.returncode != 0:
            _fail(name, "save_poly", r_save)
            return False
        r_load = _run_worker(
            "load_interp",
            "plane-couette",
            "walled",
            "concurrent",
            load_np,
            snap_path,
            ny=16,
            np0=load_np0,
        )
        if r_load.returncode != 0:
            _fail(name, "load_interp", r_load)
            return False
    print(f"  PASS  {name}")
    return True


# ── main ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot tests")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--action",
        choices=["save", "load", "save_poly", "load_interp", "save_stats"],
    )
    parser.add_argument("--system")
    parser.add_argument("--layout")
    parser.add_argument("--write-mode")
    parser.add_argument("--np", type=int)
    parser.add_argument("--np0", type=int, default=1)
    parser.add_argument("--dir")
    parser.add_argument("--ny", type=int, default=None)
    args = parser.parse_args()

    if args.worker:
        _worker(
            args.action,
            args.system,
            args.layout,
            args.write_mode,
            args.np,
            args.np0,
            args.dir,
            ny_override=args.ny,
        )
        sys.exit(0)

    passed = failed = 0
    for case in CASES:
        if run_case(*case):
            passed += 1
        else:
            failed += 1

    # ny-mismatch interpolation tests
    if run_ny_mismatch_case():
        passed += 1
    else:
        failed += 1

    if run_ny_mismatch_case(
        save_np=4, save_np0=2, load_np=4, load_np0=2, label=" 2D"
    ):
        passed += 1
    else:
        failed += 1

    # isnap metadata + optional embedded-stats member
    if run_stats_isnap_case():
        passed += 1
    else:
        failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
