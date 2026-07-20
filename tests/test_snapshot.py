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
- wall-normal regrid on load: the Cartesian case interpolates a
  polynomial profile ny 8 -> 16, and the **pipe** case additionally
  pins the alias-aware stored-``nr`` lookup (the metadata stores
  public names) plus the spectral parity interpolation across a
  rigged-CGL -> half-CGL radial-grid change;
- a viscoelastic ``nr``-mismatch load assembles the 9-component
  state at the snapshot's radial count (``_n_components``-driven
  assembly shape);
- ``serial`` write mode produces a valid snapshot (true
  cross-process ordering needs MPI multi-process and is not
  exercised here -- single-process serial reduces to one write);
- 2D mesh round-trips: save and load with ``np0 > 1``, including
  padding-mode stripping and re-padding;
- a **viscoelastic-dean** case (single-device and ``np 1 -> 2``)
  exercises the 9-component chunk count -- the test helpers derive
  the component count from the system via ``_n_comp``, mirroring the
  metadata-driven ``snapshot._n_components``;
- the ``isnap`` lineage index round-trips through the metadata, and the
  optional ``_dnsjax_stats.json`` member is written when stats are
  supplied and omitted otherwise.

Each (system, device count) needs its own process because the
geometry/sharding singletons are captured at import time, and
multiple CPU devices are obtained via
``--xla_force_host_platform_device_count``.  A case whose save and
load np-config is identical runs save + reload in **one** worker
process (one JAX init); the cross-np cases keep the separate
cold-process load, so the fresh-process read path stays covered for
both families.  The GDS path needs a GPU + kvikIO and is not
unit-tested here; it shares the same span generator (offsets and
coalescing tiers) as the host path.

Run as a script::

    uv run python tests/test_snapshot.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from _live import run_live

sys.stdout.reconfigure(line_buffering=True)

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
    # ── other wall-bounded geometries (same `walled` on-disk path;
    # taylor-couette needs no cases of its own -- the layout is
    # geometry-independent, the public-name alias metadata is pinned
    # by the pipe regrid case, and the component-count machinery by
    # the viscoelastic cases) ──
    ("pipe", "pipe", "walled", "concurrent", 1, 1, 1, 1),
    ("pipe np 1->2", "pipe", "walled", "concurrent", 1, 2, 1, 1),
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
_PERIODIC = {"kolmogorov"}

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
        assert meta["format_version"] == 6, meta["format_version"]
        # Provenance key present and non-empty (value depends on the
        # checkout, so only its presence is pinned).
        assert meta.get("git_hash"), meta

    with tempfile.TemporaryDirectory() as ex:
        with tarfile.open(d, "r") as tf:
            tf.extractall(ex, filter="data")
        on_disk = _read_state_with_tensorstore(os.path.join(ex, "state"))

    assert list(on_disk.shape) == meta["native_shape"], (
        on_disk.shape,
        meta["native_shape"],
    )
    # The on-disk bytes ARE the native (solver) spectral layout at
    # true mode counts -- no transpose between memory and disk.
    assert np.array_equal(on_disk, ref_true), (
        "TensorStore read of the extracted zarr3 store does not match "
        "the native-layout reference"
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

    if action in ("save", "roundtrip"):
        ref_true = _make_reference(np, tshape)
        state_np = _embed_true_into_padded(ref_true, padded_shape, system)
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        if action == "save":
            return
        # "roundtrip": same-np case -- fall through to the load checks
        # in this same process (one JAX init instead of two; the
        # cold-process load path stays covered by the cross-np cases).

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

    if action == "save_poly_pipe":
        from dnsjax.parameters import derived_params

        # Rigged-CGL radial grid (axis gap 1) at this worker's nr, with
        # parity-definite mean-mode profiles: u_z even in r, u_r/u_th odd.
        n_full = 2 * ny + 1
        s = -np.cos(np.arange(n_full, dtype=np.float64) * np.pi / (n_full - 1))
        rs = s[ny + 1 :]
        derived_params.wall_normal_grid = rs.tolist()
        state_np = np.zeros(padded_shape, dtype=np.complex128)
        state_np[0, :, 0, 0] = 1.0 - rs**2
        state_np[1, :, 0, 0] = 0.5 * rs * (1.0 - rs**2)
        state_np[2, :, 0, 0] = 0.25 * rs * (1.0 - rs**2)
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        return

    if action == "load_interp_pipe":
        from dnsjax.__main__ import _interpolate_if_needed
        from dnsjax.parameters import derived_params

        # Half-CGL radial grid (axis gap 0) at this worker's nr: both
        # the point count and the grid family differ from the save.
        n_full = 2 * ny
        s = -np.cos(np.arange(n_full, dtype=np.float64) * np.pi / (n_full - 1))
        rs = s[ny:]
        derived_params.wall_normal_grid = rs.tolist()

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
        # v4 metadata stores the radial count under its public name
        # ("nr"); a non-alias-aware lookup would skip the regrid and
        # leave the old-count state (the shape assert below).  The
        # spectral parity interpolation is near machine precision for
        # these low-degree parity-definite profiles.
        assert got.shape[1] == ny, (got.shape, ny)
        np.testing.assert_allclose(
            got[0, :, 0, 0].real, 1.0 - rs**2, atol=1e-10
        )
        np.testing.assert_allclose(
            got[1, :, 0, 0].real, 0.5 * rs * (1.0 - rs**2), atol=1e-10
        )
        np.testing.assert_allclose(
            got[2, :, 0, 0].real, 0.25 * rs * (1.0 - rs**2), atol=1e-10
        )
        assert t == T_SAVE
        print("worker-load-interp-pipe-ok", flush=True)
        return

    if action == "load_shape":
        # ny-mismatch load only: the assembled global state must carry
        # the flow's component count (9 for viscoelastic-dean) at the
        # *snapshot's* radial count; interpolation is a separate step.
        snapshot.validate_snapshot_params(d)
        state, t, it = snapshot.load_snapshot(d)
        assert state.shape[:2] == (_n_comp(system), NY), state.shape
        print("worker-load-shape-ok", flush=True)
        return

    # action == "load" (or the "roundtrip" fall-through)
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
    return run_live(cmd, timeout=300)


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
    """Save then load a snapshot (one process when the np config is
    unchanged, separate save/load processes across np configs)."""
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        if (save_np, save_np0) == (load_np, load_np0):
            r = _run_worker(
                "roundtrip",
                system,
                layout,
                write_mode,
                save_np,
                snap_path,
                np0=save_np0,
            )
            if r.returncode != 0:
                _fail(name, "roundtrip", r)
                return False
        else:
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


def run_pipe_regrid_case() -> bool:
    """Pipe nr 8 (rigged-CGL) -> 10 (half-CGL) regrid on load.

    Pins the alias-aware stored-``nr`` lookup (public-named v4
    metadata) and the spectral parity interpolation across a radial
    grid-family change."""
    name = "pipe nr 8->10 regrid (rigged->half)"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save_poly_pipe",
            "pipe",
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=8,
        )
        if r_save.returncode != 0:
            _fail(name, "save_poly_pipe", r_save)
            return False
        r_load = _run_worker(
            "load_interp_pipe",
            "pipe",
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=10,
        )
        if r_load.returncode != 0:
            _fail(name, "load_interp_pipe", r_load)
            return False
    print(f"  PASS  {name}")
    return True


def run_ve_ny_mismatch_case() -> bool:
    """Viscoelastic nr-mismatch load: the assembled state must carry 9
    components at the snapshot's radial count."""
    name = "viscoelastic nr 8->16 load shape"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save", "viscoelastic-dean", "walled", "concurrent", 1, snap_path
        )
        if r_save.returncode != 0:
            _fail(name, "save", r_save)
            return False
        r_load = _run_worker(
            "load_shape",
            "viscoelastic-dean",
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=16,
        )
        if r_load.returncode != 0:
            _fail(name, "load_shape", r_load)
            return False
    print(f"  PASS  {name}")
    return True


# ── main ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot tests")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--action",
        choices=[
            "save",
            "load",
            "roundtrip",
            "save_poly",
            "load_interp",
            "save_poly_pipe",
            "load_interp_pipe",
            "load_shape",
            "save_stats",
        ],
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

    print(
        "Snapshot round-trip tests: offline, forced CPU device(s) "
        "(multiple devices via --xla_force_host_platform_device_count; "
        "the GDS/GPU I/O path is not unit-tested here).",
        flush=True,
    )

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

    # Pipe regrid (public-named nr + parity interpolation) and the
    # viscoelastic 9-component ny-mismatch assembly.
    if run_pipe_regrid_case():
        passed += 1
    else:
        failed += 1

    if run_ve_ny_mismatch_case():
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
