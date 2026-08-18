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
- ``serial`` write mode produces a valid snapshot whose stored
  component data is identical to the ``concurrent`` one (the
  archives are not byte-identical as *files*: the metadata records
  the write mode).  True cross-process ordering needs MPI
  multi-process and is not exercised here -- single-process serial
  reduces to one write;
- a truncated archive fails **loudly**, and in words that name the
  file and the cause, rather than loading the zeros of the
  sparse-reserved skeleton;
- a save that dies mid-write leaves the *previous* snapshot intact
  and the final name untouched (the archive is built under a
  ``.partial`` name and renamed), with the partial file left behind
  as evidence;
- an archive whose component chunks do not match the ``native_shape``
  describing them is refused -- every read position is computed from
  that shape alone, so a disagreement would return a neighbouring
  component's bytes as state without an error;
- 2D mesh round-trips: save and load with ``np0 > 1``, including
  padding-mode stripping and re-padding;
- the **I/O-layout reshard** the multi-device paths run (save
  reshards onto contiguous leading-axis slabs at the true mode
  counts, load reshards back): every mesh family is covered --
  ``np1``-only, ``np0``-only and 2D -- because the slab a device
  owns is decided by its *flat* mesh position, and each engine call
  is checked for the SPMD ``Involuntary full rematerialization``
  warning, which is what a reshard routed through both mesh axes at
  once would print;
- both mode-padding trims that reshard performs, through a real
  writer and a real reader: the `$k_x$` one needs an ``nx`` whose
  true mode count does not divide ``np1``, and the `$k_z$` one comes
  free with every ``np0``-only row (``nz - 1`` is odd, so no even
  ``np0`` divides it).  An ``ny`` smaller than the device count
  gives a device whose slab is entirely padding;
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

from _live import report, run_live

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
    (
        # The cylindrical 9-component state: same component count, but
        # its family routes through the pipe's regrid/basis paths.
        "viscoelastic-pipe",
        "viscoelastic-pipe",
        "walled",
        "concurrent",
        1,
        1,
        1,
        1,
    ),
    (
        "viscoelastic-pipe np 1->2",
        "viscoelastic-pipe",
        "walled",
        "concurrent",
        1,
        2,
        1,
        1,
    ),
    (
        # The only 9-component *save* on a multi-device mesh: the I/O
        # layout carries the component axis whole and the engines loop
        # over it, so a component-count slip would land every
        # component but the first at the wrong file offset.
        "viscoelastic-dean 2D",
        "viscoelastic-dean",
        "walled",
        "concurrent",
        4,
        4,
        2,
        2,
    ),
    # ── 2D (np0 > 1) ──
    ("walled 2D", "plane-couette", "walled", "concurrent", 4, 4, 2, 2),
    ("periodic 2D", "kolmogorov", "periodic", "concurrent", 4, 4, 2, 2),
    # ── np0-only meshes (np1 = 1).  The I/O layout splits the
    # leading axis across *every* device by flat mesh position, so
    # an np0-only mesh takes a different branch of ``_io_spec``
    # (a single mesh axis, and the one the solver layout does not
    # put on `$k_x$`) than the np1-only and 2D rows above. ──
    ("walled np0-only", "plane-couette", "walled", "concurrent", 2, 2, 2, 2),
    (
        # Periodic A = ny - 1 = 7 over 2 devices: the leading axis is
        # zero-padded to 8, so this is A-padding on an np0-only mesh.
        "periodic np0-only",
        "kolmogorov",
        "periodic",
        "concurrent",
        2,
        2,
        2,
        2,
    ),
    (
        # Save on (2, 1), load on (1, 2): the slab a device owns is
        # unchanged (flat position), but every other axis' padding and
        # sharding differ between the two meshes.
        "walled np0-only -> np1-only",
        "plane-couette",
        "walled",
        "concurrent",
        2,
        2,
        2,
        1,
    ),
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

# Cases that override the resolution to reach a padding trim or a
# slab shape the grids above cannot produce.  Same runner, keyword
# form (``run_case(**case)``) so the tuple rows stay unchanged.
#
# ``nx = 6`` gives ``kx_true = 3``, which no device count above 1
# divides: ``nx_spec`` pads to 4, so this is the only row that
# exercises the `$k_x$` trim in ``_to_io_layout_core`` (and its
# re-pad on load).  ``nx = 8`` has ``kx_true = 4``, divisible by
# every mesh used above.  Left untrimmed, each of these would write
# one ``(a, kz)`` row at a time.
#
# ``ny = 5`` over 4 devices gives ``_a_local = 2`` and hence slabs
# ``[0, 2) [2, 4) [4, 5) []``: a clipped slab *and* a device whose
# slab is entirely padding, both through a real writer and reader.
SHAPE_CASES: list[dict] = [
    {
        "name": "walled kx trim (nx 6, 2D)",
        "system": "plane-couette",
        "layout": "walled",
        "write_mode": "concurrent",
        "save_np": 4,
        "load_np": 4,
        "save_np0": 2,
        "load_np0": 2,
        "nx": 6,
    },
    {
        "name": "walled kx trim on load (nx 6, np 1->4)",
        "system": "plane-couette",
        "layout": "walled",
        "write_mode": "concurrent",
        "save_np": 1,
        "load_np": 4,
        "save_np0": 1,
        "load_np0": 1,
        "nx": 6,
    },
    {
        "name": "walled empty + clipped slabs (ny 5, np 4)",
        "system": "plane-couette",
        "layout": "walled",
        "write_mode": "concurrent",
        "save_np": 4,
        "load_np": 4,
        "save_np0": 1,
        "load_np0": 1,
        "ny": 5,
    },
]

# Periodic systems (must match dnsjax.flows.registry.periodic_systems).
_PERIODIC = {"kolmogorov"}

# Viscoelastic systems carry 9 state components (3 velocity + 6
# symmetric conformation-tensor); must match ``snapshot._n_components``.
_VISCOELASTIC = {"viscoelastic-dean", "viscoelastic-pipe"}


def _n_comp(system: str) -> int:
    """Number of stacked state components for *system* (3 or 9)."""
    return 9 if system in _VISCOELASTIC else 3


def _ve_pipe_profiles(rs):
    """Parity-definite mean-mode profiles, one per stored component.

    Stored physical order ``(u_z, u_r, u_theta, c_zz, c_rz, c_theta_z,
    c_rr, c_theta_theta, c_r_theta)``.  At `m = 0` the `(-1)^m` class is
    *even* in `r` and the `(-1)^{m+1}` class *odd*, so each entry below
    is a low-degree polynomial of its component's own parity -- which
    the spectral parity interpolation reproduces to machine precision
    on the new grid, but only if the component is regridded with the
    matrix its class calls for.  The amplitudes are all distinct so a
    swapped pair cannot cancel out.

    The velocity entries vanish at `r = 1` (the regrid re-imposes
    no-slip on ``[:3]``); the conformation entries deliberately do not
    -- their wall condition is ``grad^2 c = 0``, and a regrid that
    zeroed them would show up here.
    """
    import numpy as np

    rs = np.asarray(rs, dtype=np.float64)
    even, odd = 1.0 - rs**2, rs * (1.0 - rs**2)
    return [
        even,  # u_z         even
        0.5 * odd,  # u_r         odd
        0.25 * odd,  # u_theta     odd
        1.0 + 2.0 * rs**2,  # c_zz        even, nonzero at the wall
        -1.5 * rs,  # c_rz        odd,  nonzero at the wall
        0.3 * odd,  # c_theta_z   odd
        1.0 + 0.1 * rs**2,  # c_rr        even
        1.0 - 0.2 * rs**2,  # c_theta_th  even
        0.4 * even,  # c_r_theta   even
    ]


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


def _check_slab_placement(state, snapshot, sharding) -> None:
    """The I/O layout must place the slab this module's math assumes.

    Every byte range a device writes comes from
    ``_a_ranges(_shard_device_index(shard), ...)`` -- the device's
    *flat* mesh position -- while the placement is chosen by
    ``P(None, (a0, a1), None, None)``.  The two agree only because a
    tuple of mesh axes splits the array axis in row-major (a0-major)
    order, which is a JAX convention this module reads but cannot
    enforce.  A silent change of it would hand every device another
    one's slab, so pin it against the real sharding.

    The mode axes are pinned here too.  Each writer sends
    ``vec[comp][:na]`` in a single call, which is that device's file
    range only if the buffer carries the **true** mode counts; one
    that kept the solver layout's divisibility padding would satisfy
    the placement check below and write the padding modes into the
    file.  On a single-device mesh this also pins the short-circuit in
    ``_to_io_layout``, which returns the solver array untouched on the
    grounds that ``np0 = np1 = 1`` pads nothing.
    """
    import numpy as np

    io = snapshot._to_io_layout(state)
    a_true, ndev = state.shape[1], snapshot._n_devices()
    a_local = snapshot._a_local(a_true, ndev)
    kz_true, kx_true = snapshot._kz_true(), snapshot._kx_true()
    assert io.shape[2:] == (kz_true, kx_true), (
        f"I/O layout carries mode axes {io.shape[2:]}, not the true "
        f"counts {(kz_true, kx_true)}: the writers would send padding "
        "modes, or fragment"
    )
    assert snapshot._io_local_shape(a_true)[2:] == (kz_true, kx_true), (
        snapshot._io_local_shape(a_true),
        (kz_true, kx_true),
    )
    for shard in io.addressable_shards:
        flat = snapshot._shard_device_index(shard)
        want = (flat * a_local, (flat + 1) * a_local)
        got = shard.index[1]
        assert (got.start or 0, got.stop or io.shape[1]) == want, (
            f"device at flat mesh position {flat} holds A rows {got}, "
            f"but the writer sends it {want}"
        )
        # And the payload: the reshard must have moved the rows, not
        # merely relabelled them.
        a_start, na = snapshot._a_ranges(flat, a_true, ndev)
        local = np.asarray(shard.data)
        ref = np.asarray(state)[:, a_start : a_start + na, :kz_true, :kx_true]
        assert np.array_equal(local[:, :na], ref), (
            f"slab content mismatch at flat mesh position {flat}"
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
    nx_override: int | None = None,
):
    """Set up singletons for *npv* CPU devices, then save or load."""
    if nx_override is not None:
        global NX
        NX = nx_override
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
        _check_slab_placement(state, snapshot, sharding)
        # The state must survive its own save: nothing in the write
        # path may donate or mutate the caller's array (``__main__``
        # keeps stepping the state it just snapshotted).
        assert np.array_equal(np.asarray(state), state_np), (
            "save_snapshot mutated the caller's state"
        )
        # A completed save renames its scratch file into place.
        assert not os.path.exists(d + ".partial"), (
            "a completed save left its .partial file behind"
        )
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

    if action == "save_fail":
        # A save that dies mid-write must not take the previous
        # snapshot with it.  The archive is laid out full-length with
        # zero-filled component regions before anything is written, so
        # a half-finished save under the final name would be a valid
        # tar that loads without complaint and is blank where the
        # writes did not reach -- zeros being a legal state, nothing
        # downstream could tell.
        ref_true = _make_reference(np, tshape)
        state_np = _embed_true_into_padded(ref_true, padded_shape, system)
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        with open(d, "rb") as f:
            good = f.read()

        def _boom(*_args, **_kwargs):
            raise OSError("simulated mid-write failure")

        snapshot._write_chunks_host = _boom
        try:
            snapshot.save_snapshot(state, T_SAVE + 1.0, IT_SAVE + 1, d)
        except OSError:
            pass
        else:
            raise AssertionError("the broken writer did not raise")

        with open(d, "rb") as f:
            assert f.read() == good, (
                "a failed save replaced the previous snapshot"
            )
        leftovers = [
            n for n in os.listdir(os.path.dirname(d)) if n.endswith(".partial")
        ]
        assert leftovers, "a failed save left no .partial to diagnose"
        print("worker-save-fail-ok", flush=True)
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

    if action == "save_poly_ve_pipe":
        from dnsjax.parameters import derived_params

        # Rigged-CGL radial grid (axis gap 1); 9 parity-definite
        # mean-mode profiles (see ``_ve_pipe_profiles``).
        n_full = 2 * ny + 1
        s = -np.cos(np.arange(n_full, dtype=np.float64) * np.pi / (n_full - 1))
        rs = s[ny + 1 :]
        derived_params.wall_normal_grid = rs.tolist()
        state_np = np.zeros(padded_shape, dtype=np.complex128)
        for c, prof in enumerate(_ve_pipe_profiles(rs)):
            state_np[c, :, 0, 0] = prof
        state = jax.device_put(state_np, vshard)
        snapshot.save_snapshot(state, T_SAVE, IT_SAVE, d)
        return

    if action == "load_interp_ve_pipe":
        from dnsjax.__main__ import _interpolate_if_needed
        from dnsjax.parameters import derived_params

        # Half-CGL radial grid (axis gap 0): both the point count and
        # the grid family differ from the save.
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
        assert got.shape[:2] == (9, ny), (got.shape, ny)
        want = _ve_pipe_profiles(rs)
        for c, prof in enumerate(want):
            ref = prof.copy()
            if c < 3:
                ref[-1] = 0.0  # the regrid re-imposes no-slip
            np.testing.assert_allclose(
                got[c, :, 0, 0].real, ref, atol=1e-10, err_msg=f"component {c}"
            )
        # The conformation block is *not* wall-zeroed.
        assert abs(got[3, -1, 0, 0].real - want[3][-1]) < 1e-10
        assert t == T_SAVE
        print("worker-load-interp-ve-pipe-ok", flush=True)
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
    nx: int | None = None,
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
    if nx is not None:
        cmd.extend(["--nx", str(nx)])
    return run_live(cmd, timeout=300)


def _fail(name: str, stage: str, res: subprocess.CompletedProcess) -> str:
    """Print the failure detail; return the one-line summary reason."""
    reason = f"{stage} exit {res.returncode}"
    print(f"  FAIL  {name}: {reason}")
    print(res.stdout[-2000:] if res.stdout else "(no stdout)")
    print(res.stderr[-2000:] if res.stderr else "(no stderr)")
    return reason


# XLA's SPMD partitioner prints this when asked to relocate two mesh
# axes at once and gives up, replicating the whole array on every
# device instead of exchanging it -- ndev x the traffic and ndev x the
# peak memory.  It is a warning, not an error, so only a log check
# catches a reshard that stops being routed one mesh axis at a time.
_REMAT = "Involuntary full rematerialization"


def _check(
    name: str, stage: str, res: subprocess.CompletedProcess
) -> str | None:
    """``None`` when the worker is clean, else its failure reason."""
    if res.returncode != 0:
        return _fail(name, stage, res)
    if _REMAT in (res.stdout or "") + (res.stderr or ""):
        reason = f"{stage} SPMD rematerialization"
        print(f"  FAIL  {name}: {reason}")
        return reason
    return None


def run_case(
    name: str,
    system: str,
    layout: str,
    write_mode: str,
    save_np: int,
    load_np: int,
    save_np0: int = 1,
    load_np0: int = 1,
    nx: int | None = None,
    ny: int | None = None,
) -> str | None:
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
                nx=nx,
                ny=ny,
            )
            if reason := _check(name, "roundtrip", r):
                return reason
        else:
            r_save = _run_worker(
                "save",
                system,
                layout,
                write_mode,
                save_np,
                snap_path,
                np0=save_np0,
                nx=nx,
                ny=ny,
            )
            if reason := _check(name, "save", r_save):
                return reason
            r_load = _run_worker(
                "load",
                system,
                layout,
                write_mode,
                load_np,
                snap_path,
                np0=load_np0,
                nx=nx,
                ny=ny,
            )
            if reason := _check(name, "load", r_load):
                return reason
    print(f"  PASS  {name}")
    return None


def run_io_layout_case() -> str | None:
    """Check the I/O-layout slab arithmetic exhaustively, in-process.

    ``_a_ranges`` and ``_a_offset`` are pure, so the branch the
    round-trip cases cannot reach is cheapest to check directly: a
    device whose slab is **entirely padding** (``n_rows == 0``), which
    needs ``a_true <= (ndev - 1) * ceil(a_true / ndev)`` -- 5 rows over
    4 devices, never produced by the ``ny = 8`` grids used above.

    The invariants are the ones the format depends on: the slabs tile
    ``[0, a_true)`` exactly, and written at the offsets ``_a_offset``
    hands them they reassemble the component chunk byte for byte --
    each from a **single** C-contiguous transfer, which is the property
    the trimmed I/O layout exists to provide and the one the writers
    assume when they send ``vec[comp][:na]`` in one call.
    """
    import numpy as np

    from dnsjax.snapshot import _a_local, _a_offset, _a_ranges

    for a_true in (1, 5, 7, 8, 13, 193):
        for ndev in (1, 2, 4, 8):
            covered: list[int] = []
            for flat in range(ndev):
                a_start, na = _a_ranges(flat, a_true, ndev)
                covered.extend(range(a_start, a_start + na))
            if covered != list(range(a_true)):
                return (
                    f"slabs do not tile a_true={a_true} ndev={ndev}: "
                    f"{covered[:12]}..."
                )

    # The engines write ``off * itemsize`` bytes into the file, so
    # replay that: fill each device's local buffer with the chunk
    # values it should own, write its slab at its byte offset, and
    # require the result to be the reference chunk.  This is the layer
    # the element-index check above skips.
    dtype = np.dtype("complex128")
    for a_true, ndev, kz_true, kx_true in (
        (7, 4, 5, 3),  # a clipped last slab
        (5, 4, 15, 4),  # a device whose slab is entirely padding
        (13, 1, 255, 128),  # single device, production-shaped modes
    ):
        chunk = (
            np.arange(a_true * kz_true * kx_true, dtype=np.float64)
            .astype(dtype)
            .reshape(a_true, kz_true, kx_true)
        )
        file_bytes = bytearray(chunk.nbytes)
        for flat in range(ndev):
            a_start, na = _a_ranges(flat, a_true, ndev)
            local = np.zeros(
                (_a_local(a_true, ndev), kz_true, kx_true), dtype=dtype
            )
            local[:na] = chunk[a_start : a_start + na]
            span = local[:na]
            if not span.flags.c_contiguous:
                return f"slab {flat} is not contiguous ({local.shape})"
            start = _a_offset(a_start, kz_true, kx_true) * dtype.itemsize
            file_bytes[start : start + span.nbytes] = span.tobytes()
        if bytes(file_bytes) != chunk.tobytes():
            return (
                "replayed slab bytes differ from the reference chunk "
                f"for a_true={a_true} ndev={ndev} modes "
                f"({kz_true}, {kx_true})"
            )
    return None


def run_gds_detection_case() -> str | None:
    """``_gds_available`` must answer for the *shipped* kvikIO API.

    This check spent its life returning ``False`` on a cluster that
    had kvikIO installed, because ``kvikio.defaults`` is a submodule
    rather than an attribute of the package -- a failure invisible
    from the outside, since "no GDS" is also the correct answer on a
    node without the driver.  Stub kvikIO and drive every branch: the
    submodule bind, the cupy and nvidia-fs gates, each ``CompatMode``
    the enum can hold (a bare truth test misreads ``AUTO``), and a
    ``get`` that answers with something other than that enum -- which
    must take the "unreadable" path rather than raise out of the
    check.
    """
    import enum
    import sys
    import types

    from dnsjax import snapshot

    class CompatMode(enum.Enum):
        OFF = 0
        ON = 1
        AUTO = 2

    ABSENT = object()  # "kvikIO is not installed", not a compat mode

    def install(value) -> None:
        """Put a fake kvikIO answering ``get("compat_mode")`` on path.

        Deliberately faithful about the trap: the parent module gets
        **no** ``defaults`` attribute, exactly like the installed
        package before its submodule is imported, so reaching for
        ``kvikio.defaults`` through the package raises
        ``AttributeError`` here as it did on the cluster.
        """
        kv = types.ModuleType("kvikio")
        kvd = types.ModuleType("kvikio.defaults")
        kvd.get = lambda name: {"compat_mode": value}[name]
        sys.modules["kvikio"], sys.modules["kvikio.defaults"] = kv, kvd

    watched = ("kvikio", "kvikio.defaults", "cupy")
    real = {k: sys.modules.get(k) for k in watched}
    nvfs = snapshot._NVFS_STATS
    with tempfile.TemporaryDirectory() as tmp:
        present = os.path.join(tmp, "stats")
        open(present, "w").close()
        absent = os.path.join(tmp, "no-such-driver")
        # (label, compat mode, cupy, driver, expected)
        cases = [
            ("kvikio absent", ABSENT, True, present, False),
            ("cupy absent", CompatMode.AUTO, False, present, False),
            ("driver absent", CompatMode.AUTO, True, absent, False),
            ("AUTO", CompatMode.AUTO, True, present, True),
            ("OFF", CompatMode.OFF, True, present, True),
            ("ON  -> compat", CompatMode.ON, True, present, False),
            # Not a CompatMode: ``type(mode).ON`` raises, and the
            # check must fall back to the host path.  This is why the
            # enum comparison sits *inside* the guarded block.
            ("non-enum mode", True, True, present, False),
        ]
        try:
            for label, mode, cupy, driver, expected in cases:
                if mode is ABSENT:
                    # ``None`` in sys.modules makes the import raise.
                    sys.modules["kvikio"] = None
                    sys.modules.pop("kvikio.defaults", None)
                else:
                    install(mode)
                # A real cupy (a GPU box) must not decide the outcome
                # either way, so stub both states explicitly.
                sys.modules["cupy"] = (
                    types.ModuleType("cupy") if cupy else None
                )
                snapshot._NVFS_STATS = __import__("pathlib").Path(driver)
                snapshot._gds_available.cache_clear()
                got = snapshot._gds_available()
                if got is not expected:
                    print(f"  FAIL  GDS detection: {label} -> {got}")
                    return f"GDS detection wrong for {label}"
        finally:
            snapshot._NVFS_STATS = nvfs
            snapshot._gds_available.cache_clear()
            for k, v in real.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    print("  PASS  GDS detection branches")
    return None


def run_truncation_case() -> str | None:
    """A truncated archive must fail loudly, not load silently.

    The component data regions are sparse-*reserved* (zero-filled) by
    the skeleton writer, so a short archive has zeros exactly where a
    state should be: the one failure mode that would resume a run
    from a partly blank field without a word.  Cut the last
    component's data in half and require the load to raise.
    """
    name = "truncated archive rejected"
    with tempfile.TemporaryDirectory() as tmp:
        snap = os.path.join(tmp, "snap.tar")
        r = _run_worker(
            "save", "plane-couette", "walled", "concurrent", 1, snap
        )
        if reason := _check(name, "save", r):
            return reason
        # Halfway into the final component: every tar header lies
        # before it, so this is a data-region cut, not a header one.
        comp_nbytes = NY * (NZ - 1) * (NX // 2) * 16
        with open(snap, "r+b") as f:
            f.truncate(os.path.getsize(snap) - 1024 - comp_nbytes // 2)
        r_load = _run_worker(
            "load", "plane-couette", "walled", "concurrent", 1, snap
        )
        if r_load.returncode == 0:
            print(f"  FAIL  {name}: truncated archive loaded silently")
            return "truncated archive loaded silently"
        # ...and it must say so.  Untranslated this is a bare
        # ``ReadError: unexpected end of data`` naming neither the
        # file nor the reason, which on a resume reads as a dnsjax
        # bug rather than a damaged checkpoint.
        out = (r_load.stdout or "") + (r_load.stderr or "")
        if "truncated or corrupt" not in out:
            print(f"  FAIL  {name}: rejected, but not intelligibly")
            return "truncated archive rejected without naming the cause"
    print(f"  PASS  {name}")
    return None


def run_meta_chunk_consistency_case() -> str | None:
    """The chunks must match the ``native_shape`` describing them.

    Every read position is computed arithmetically from
    ``native_shape``; nothing consults the member it lands in.  So a
    snapshot whose metadata outgrew its chunks reads the *next*
    component's bytes as state -- well-formed complex numbers, no
    error.  Built by hand here (stdlib only, no JAX, no subprocess):
    a matching pair must be accepted, and each way of disagreeing
    must be refused.
    """
    import io
    import json
    import tarfile

    name = "chunks match the metadata"
    shape = [3, 8, 7, 4]  # (components, A, kz, kx)
    chunk = 8 * 7 * 4 * 16  # complex128

    def build(tar_path: str, declared, n_chunks: int, chunk_bytes: int):
        meta = json.dumps(
            {
                "format_version": 6,
                "native_shape": declared,
                "dtype": "complex128",
            }
        ).encode()
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo("_dnsjax_meta.json")
            info.size = len(meta)
            tf.addfile(info, io.BytesIO(meta))
            for c in range(n_chunks):
                info = tarfile.TarInfo(f"state/c/{c}/0/0/0")
                info.size = chunk_bytes
                tf.addfile(info, io.BytesIO(b"\x00" * chunk_bytes))

    from dnsjax.snapshot_meta import (
        SnapshotArchiveError,
        snapshot_component_offsets,
    )

    # (label, declared native_shape, chunk count, chunk size, accept?)
    cases = [
        ("consistent", shape, 3, chunk, True),
        ("chunks too small", [3, 9, 7, 4], 3, chunk, False),
        ("chunks too large", [3, 7, 7, 4], 3, chunk, False),
        ("too few chunks", shape, 2, chunk, False),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for label, declared, n_chunks, chunk_bytes, accept in cases:
            p = os.path.join(tmp, "s.tar")
            build(p, declared, n_chunks, chunk_bytes)
            try:
                snapshot_component_offsets(p)
            except SnapshotArchiveError as exc:
                if accept:
                    print(f"  FAIL  {name}: {label} rejected -- {exc}")
                    return f"a consistent archive was rejected ({label})"
            else:
                if not accept:
                    print(f"  FAIL  {name}: {label} accepted")
                    return f"an inconsistent archive was accepted ({label})"
    print(f"  PASS  {name}")
    return None


def run_atomic_write_case() -> str | None:
    """A save that fails mid-write must not destroy the good snapshot."""
    name = "failed save keeps the previous snapshot"
    with tempfile.TemporaryDirectory() as tmp:
        r = _run_worker(
            "save_fail",
            "plane-couette",
            "walled",
            "concurrent",
            1,
            os.path.join(tmp, "snap.tar"),
        )
        if reason := _check(name, "save_fail", r):
            return reason
        if "worker-save-fail-ok" not in (r.stdout or ""):
            print(f"  FAIL  {name}: worker did not confirm")
            return "save_fail worker did not confirm"
    print(f"  PASS  {name}")
    return None


def run_write_mode_identity_case() -> str | None:
    """``serial`` and ``concurrent`` must store the same component data.

    The two modes differ only in *when* each process writes, so the
    stored field cannot depend on the choice.  Only the component
    members are compared: the metadata member records
    ``snapshot_write_mode`` itself (and "concurrent" and "serial" are
    not even the same length), so the archives are legitimately not
    byte-identical as files.
    """
    import tarfile

    name = "serial == concurrent component data"
    comps = [f"state/c/{i}/0/0/0" for i in range(3)]
    with tempfile.TemporaryDirectory() as tmp:
        data: dict[str, list[bytes]] = {}
        for mode in ("concurrent", "serial"):
            snap = os.path.join(tmp, f"{mode}.tar")
            r = _run_worker("save", "plane-couette", "walled", mode, 1, snap)
            if reason := _check(name, f"save {mode}", r):
                return reason
            with tarfile.open(snap, "r") as tf:
                data[mode] = [tf.extractfile(c).read() for c in comps]
        if data["concurrent"] != data["serial"]:
            print(f"  FAIL  {name}: component data differs")
            return "serial and concurrent component data differ"
    print(f"  PASS  {name}")
    return None


def run_stats_isnap_case() -> str | None:
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
        if reason := _check(name, "save_stats", r):
            return reason

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
    return None


def run_ny_mismatch_case(
    save_np: int = 1,
    save_np0: int = 1,
    load_np: int = 1,
    load_np0: int = 1,
    label: str = "",
    save_ny: int = 8,
) -> str | None:
    """Save at *save_ny*, load at ny=16 with interpolation.

    A ``save_ny`` that the load mesh does not divide (7 over 4
    devices) is the one path where the leading axis is zero-padded
    to a divisible length *and* the assembly runs at the snapshot's
    row count rather than the current one -- the reader has to take
    ``a_true`` from the metadata, pad to the load mesh, and strip
    again after resharding back.
    """
    name = f"wb ny {save_ny}->16 interp{label}"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save_poly",
            "plane-couette",
            "walled",
            "concurrent",
            save_np,
            snap_path,
            ny=save_ny,
            np0=save_np0,
        )
        if reason := _check(name, "save_poly", r_save):
            return reason
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
        if reason := _check(name, "load_interp", r_load):
            return reason
    print(f"  PASS  {name}")
    return None


def run_pipe_regrid_case() -> str | None:
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
        if reason := _check(name, "save_poly_pipe", r_save):
            return reason
        r_load = _run_worker(
            "load_interp_pipe",
            "pipe",
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=10,
        )
        if reason := _check(name, "load_interp_pipe", r_load):
            return reason
    print(f"  PASS  {name}")
    return None


def run_ve_pipe_regrid_case() -> str | None:
    """Viscoelastic-pipe nr 8 (rigged-CGL) -> 10 (half-CGL) regrid.

    The only exercise of the **9-component** parity-aware radial
    interpolation: the cylindrical branch of
    ``__main__._interpolate_if_needed`` assigns a parity class per
    stored component, and the six conformation slots are only reachable
    through a viscoelastic flow in the cylindrical family."""
    name = "viscoelastic-pipe nr 8->10 regrid (rigged->half)"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save_poly_ve_pipe",
            "viscoelastic-pipe",
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=8,
        )
        if reason := _check(name, "save_poly_ve_pipe", r_save):
            return reason
        r_load = _run_worker(
            "load_interp_ve_pipe",
            "viscoelastic-pipe",
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=10,
        )
        if reason := _check(name, "load_interp_ve_pipe", r_load):
            return reason
    print(f"  PASS  {name}")
    return None


def run_ve_ny_mismatch_case() -> str | None:
    """Viscoelastic nr-mismatch load: the assembled state must carry 9
    components at the snapshot's radial count."""
    name = "viscoelastic nr 8->16 load shape"
    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.tar")
        r_save = _run_worker(
            "save", "viscoelastic-dean", "walled", "concurrent", 1, snap_path
        )
        if reason := _check(name, "save", r_save):
            return reason
        r_load = _run_worker(
            "load_shape",
            "viscoelastic-dean",
            "walled",
            "concurrent",
            1,
            snap_path,
            ny=16,
        )
        if reason := _check(name, "load_shape", r_load):
            return reason
    print(f"  PASS  {name}")
    return None


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
            "save_poly_ve_pipe",
            "load_interp_ve_pipe",
            "load_shape",
            "save_stats",
            "save_fail",
        ],
    )
    parser.add_argument("--system")
    parser.add_argument("--layout")
    parser.add_argument("--write-mode")
    parser.add_argument("--np", type=int)
    parser.add_argument("--np0", type=int, default=1)
    parser.add_argument("--dir")
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
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
            nx_override=args.nx,
        )
        sys.exit(0)

    print(
        "Snapshot round-trip tests: offline, forced CPU device(s) "
        "(multiple devices via --xla_force_host_platform_device_count; "
        "the GDS/GPU I/O path is not unit-tested here).",
        flush=True,
    )

    # Each case returns None when it passes, else its one-line reason;
    # ``report`` repeats the failures after the counts (see _live).
    results: list[tuple[str, str | None]] = [
        (case[0], run_case(*case)) for case in CASES
    ]
    results.extend((case["name"], run_case(**case)) for case in SHAPE_CASES)

    # ny-mismatch interpolation tests
    results.append(("wb ny 8->16 interp", run_ny_mismatch_case()))
    results.append(
        (
            "wb ny 8->16 interp 2D",
            run_ny_mismatch_case(
                save_np=4, save_np0=2, load_np=4, load_np0=2, label=" 2D"
            ),
        )
    )
    results.append(
        (
            "wb ny 7->16 interp 2D",
            run_ny_mismatch_case(
                save_np=1,
                save_np0=1,
                load_np=4,
                load_np0=2,
                label=" 2D",
                save_ny=7,
            ),
        )
    )

    # Pipe regrid (public-named nr + parity interpolation) and the
    # viscoelastic 9-component ny-mismatch assembly.
    results.append(("pipe nr 8->10 regrid", run_pipe_regrid_case()))
    results.append(
        ("viscoelastic nr 8->16 load shape", run_ve_ny_mismatch_case())
    )
    results.append(
        (
            "viscoelastic-pipe nr 8->10 regrid",
            run_ve_pipe_regrid_case(),
        )
    )

    # isnap metadata + optional embedded-stats member
    results.append(("isnap + embedded stats", run_stats_isnap_case()))

    # Which I/O engine gets picked, and the archive-level guarantees:
    # a short archive is rejected, and the two write modes agree.
    results.append(("GDS detection branches", run_gds_detection_case()))
    results.append(("truncated archive rejected", run_truncation_case()))
    results.append(
        ("chunks match the metadata", run_meta_chunk_consistency_case())
    )
    results.append(
        (
            "failed save keeps the previous snapshot",
            run_atomic_write_case(),
        )
    )
    results.append(
        (
            "serial == concurrent component data",
            run_write_mode_identity_case(),
        )
    )

    # The I/O-layout slab arithmetic, over shapes the round-trip cases
    # above cannot reach (see run_io_layout_case).
    results.append(("I/O layout slabs", run_io_layout_case()))

    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))
