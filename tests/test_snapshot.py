"""Round-trip tests for the single-file (tar) snapshot module.

Exercises the host (non-GDS) I/O path:

- save -> load equality for ``walled`` and ``periodic`` layouts;
- **np-agnostic** resume: save at one ``(np0, np1)`` configuration,
  load at a different one (why it works: the ``snapshot`` docstring);
- the snapshot is a single uncompressed tar wrapping a zarr3 store,
  readable with **standard tools and no dnsjax**: the metadata
  member parses as plain JSON (stdlib ``tarfile`` + ``json``), and
  after ``tar xf`` the ``state/`` store reads back via TensorStore
  with exactly the stored data;
- wall-normal regrid on load: the Cartesian case interpolates a polynomial
  profile ny 8 -> 16 (9 -> 16 from a grid recorded in float32 arithmetic, still
  by the spectral route), and the **pipe** case additionally pins the
  alias-aware stored-``nr`` lookup (the metadata stores public names) plus the
  spectral parity interpolation across a rigged-CGL -> half-CGL radial-grid
  change;
- the Fourier-resolution regrid at load (``MODE_CASES``: ``nx``, ``nz``
  and the periodic ``ny``): a mode the snapshot kept comes back
  bit-identical at its own wavenumber, an added one zero, a dropped
  one gone, the padding slots zero (``_check_mode_regrid``);
- ``serial`` write mode produces a valid snapshot whose stored
  component data is identical to the ``concurrent`` one; true
  cross-process ordering needs MPI multi-process and is not exercised
  here (single-process serial reduces to one write);
- a truncated archive fails **loudly**, and in words that name the
  file and the cause, rather than loading the zeros of the
  sparse-reserved skeleton; a gzip'd copy is refused, in words that
  name the compression;
- a save that dies mid-write leaves the *previous* snapshot intact
  and the final name untouched (the archive is built under a
  ``.partial`` name and renamed), with the partial file left behind
  as evidence;
- an archive whose component chunks do not match the ``native_shape``
  describing them is refused (why: ``snapshot_meta._check_chunks_match_meta``);
- 2D mesh round-trips: save and load with ``np0 > 1``, including
  padding-mode stripping and re-padding;
- the **I/O-layout reshard** of the multi-device paths (the
  ``snapshot`` docstring): every mesh family is covered --
  ``np1``-only, ``np0``-only and 2D -- because the slab a device owns
  is decided by its *flat* mesh position, and each engine call is
  checked for the SPMD ``Involuntary full rematerialization`` warning,
  which a reshard routed through both mesh axes at once would print;
- both mode-padding trims that reshard performs, through a real
  writer and a real reader: the `$k_x$` one needs an ``nx`` whose
  true mode count does not divide ``np1``, and the `$k_z$` one comes
  free with every ``np0``-only row (``nz - 1`` is odd, so no even
  ``np0`` divides it).  ``ny = 5`` over four devices gives a clipped
  slab and a device whose slab is entirely padding;
- the slab arithmetic itself, in-process over more shapes: the slabs
  tile the leading axis and reassemble each chunk byte for byte;
- the ``isnap`` lineage index round-trips through the metadata, and the
  optional ``_dnsjax_stats.json`` member is written when stats are
  supplied and omitted otherwise;
- the ``git_hash`` provenance agrees with ``git describe``, and git is
  spawned with ``posix_spawn`` (never CPython's ``vfork`` path).

Every save and load runs in a worker process (why: the machinery's
docstring).  A case whose save and load np-config is identical runs
save + reload in **one** worker process (one JAX init); the cross-np
cases keep the separate cold-process load, so the fresh-process read
path stays covered for both families.  The GDS transfers need a GPU +
kvikIO and are not tested here (their detection is, against a
simulated kvikIO); they share the slab arithmetic (``_a_ranges``,
``_a_offset``) with the host path.

The machinery and the workers: ``tests/_snapshot.py``.  The lab's
flows have cases of their own, ``lab/tests/test_snapshot.py``: the
stored component count beyond the velocity, and its parity regrid.

Run as a script::

    uv run python tests/test_snapshot.py
"""

from __future__ import annotations

import sys

from _live import report
from _snapshot import (
    run_atomic_write_case,
    run_case,
    run_compressed_case,
    run_gds_detection_case,
    run_git_hash_case,
    run_io_layout_case,
    run_meta_chunk_consistency_case,
    run_mode_case,
    run_ny_mismatch_case,
    run_parity_regrid_case,
    run_stats_isnap_case,
    run_truncation_case,
    run_write_mode_identity_case,
)

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
    # geometry-independent, and the public-name alias metadata is
    # pinned by the pipe regrid case) ──
    ("pipe", "pipe", "walled", "concurrent", 1, 1, 1, 1),
    ("pipe np 1->2", "pipe", "walled", "concurrent", 1, 2, 1, 1),
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

# Fourier-resolution regrid on resume: (nx, ny, nz) at save and at
# load.  ``nx`` is the real-FFT half axis (append / prefix-slice),
# ``nz`` the wrap-ordered complex one (insert / drop at the high-|k|
# end from *both* blocks) -- the two need different arithmetic, so
# the rows move each one alone at least once.  The 2D-mesh rows
# are the ones that matter: the resize happens inside the I/O-layout
# reshard, where only a (2, 2) mesh exercises both legs.
MODE_CASES: list[dict] = [
    {
        "name": "wb kx up (nx 8->16)",
        "system": "plane-couette",
        "save_res": (8, 9, 8),
        "load_res": (16, 9, 8),
    },
    {
        "name": "wb kx down (nx 16->8)",
        "system": "plane-couette",
        "save_res": (16, 9, 8),
        "load_res": (8, 9, 8),
    },
    {
        "name": "wb kz up (nz 8->16)",
        "system": "plane-couette",
        "save_res": (8, 9, 8),
        "load_res": (8, 9, 16),
    },
    {
        "name": "wb kz down (nz 16->8)",
        "system": "plane-couette",
        "save_res": (8, 9, 16),
        "load_res": (8, 9, 8),
    },
    {
        # The one row whose resize is not a doubling: the stored count
        # goes 5 -> 9, so both wrap-order blocks grow by an amount the
        # halving/doubling rows never produce.
        "name": "wb kz non-doubling (nz 6->10)",
        "system": "plane-couette",
        "save_res": (8, 9, 6),
        "load_res": (8, 9, 10),
    },
    {
        "name": "wb both up, 2D mesh (nx/nz 8->16)",
        "system": "plane-couette",
        "save_res": (8, 9, 8),
        "load_res": (16, 9, 16),
        "save_np": 4,
        "save_np0": 2,
        "load_np": 4,
        "load_np0": 2,
    },
    {
        "name": "wb both down, 2D mesh (nx/nz 16->8)",
        "system": "plane-couette",
        "save_res": (16, 9, 16),
        "load_res": (8, 9, 8),
        "save_np": 4,
        "save_np0": 2,
        "load_np": 4,
        "load_np0": 2,
    },
    {
        "name": "wb mixed + np change (nx up, nz down, np 1->4)",
        "system": "plane-couette",
        "save_res": (8, 9, 16),
        "load_res": (16, 9, 8),
        "save_np": 1,
        "save_np0": 1,
        "load_np": 4,
        "load_np0": 2,
    },
    {
        "name": "pipe kz/kx regrid (aliased names)",
        "system": "pipe",
        "save_res": (8, 9, 8),
        "load_res": (16, 9, 16),
    },
    {
        "name": "periodic ky up (ny 8->16)",
        "system": "kolmogorov",
        "save_res": (8, 8, 8),
        "load_res": (8, 16, 8),
    },
    {
        "name": "periodic ky down (ny 16->8)",
        "system": "kolmogorov",
        "save_res": (8, 16, 8),
        "load_res": (8, 8, 8),
    },
    {
        "name": "periodic all three, 2D mesh",
        "system": "kolmogorov",
        "save_res": (8, 8, 8),
        "load_res": (16, 16, 16),
        "save_np": 4,
        "save_np0": 2,
        "load_np": 4,
        "load_np0": 2,
    },
]


if __name__ == "__main__":
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
    results.extend(
        (case["name"], run_mode_case(**case)) for case in MODE_CASES
    )

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
    results.append(
        (
            "wb ny 9->16 interp f32-grid",
            run_ny_mismatch_case(
                label=" f32-grid", save_ny=9, stored_float32=True
            ),
        )
    )

    # Pipe regrid (public-named nr + parity interpolation).
    results.append(("pipe nr 8->10 regrid", run_parity_regrid_case("pipe")))

    # isnap metadata + optional embedded-stats member
    results.append(("isnap + embedded stats", run_stats_isnap_case()))

    # The I/O engine choice and the archive-level guarantees.
    results.append(("GDS detection branches", run_gds_detection_case()))
    results.append(("truncated archive rejected", run_truncation_case()))
    results.append(("compressed archive rejected", run_compressed_case()))
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

    # The I/O-layout slab arithmetic (run_io_layout_case).
    results.append(("I/O layout slabs", run_io_layout_case()))

    # The provenance every snapshot records, and how git is spawned.
    results.append(("git_hash via posix_spawn", run_git_hash_case()))

    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))
