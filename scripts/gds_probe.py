"""Cluster diagnostic: is the snapshot GDS path live, and is it starved?

Two open audit items, both of which need real hardware and a real
filesystem to answer:

**Item 6 -- is the GDS path even taken?**  ``snapshot._gds_available``
compares ``kvikio.defaults.compat_mode()`` against the ``CompatMode``
enum rather than truth-testing it, because ``AUTO`` (the default) is
*available* while a bare ``bool(AUTO)`` reads as "compat".  Whether
that fix was load-bearing or inert depends on what this cluster's
kvikIO actually returns.  Part A answers it, replicating the check
step by step, and goes further: it diffs the ``nvidia-fs`` kernel
counters across a real transfer, which is the only way to tell an
engaged GDS path from a silent POSIX fallback.

**Item 7 -- queue depth.**  ``_write_chunks_gds`` / ``_read_chunks_gds``
issue one **blocking** ``CuFile.write`` / ``.read`` per span.  kvikIO
splits a single call across its thread pool, but the next span is not
submitted until the previous one has drained, so the achieved queue
depth is 1 -- and the module docstring's worst case is ~1e5 spans of
512 B per device per snapshot (a ``256 x 193 x 256`` wall-bounded run
at ``np0 = 1, np1 = 4``).  ``pwrite`` / ``pread`` return an
``IOFuture``; keeping D of them in flight is a small change.  Whether
it is worth making is a latency-vs-bandwidth question about this
storage stack.

Part A  environment + the item-6 verdict (kvikIO, cupy, nvidia-fs
        counters, the target filesystem, ``KVIKIO_*``).
Part B  the span-pattern benchmark: the real ownership pattern of a
        sharded snapshot (``nspans`` spans of ``span_bytes``, every
        ``stride``-th block -- ``stride`` is ``np1``) written and read
        by each engine.  Engines: ``blocking`` (what ships),
        ``pwrite:D`` (the proposed fix at queue depth D), ``contig``
        (one whole-buffer call -- the bandwidth ceiling this layout
        cannot reach), ``host`` (device->host copy per span + POSIX
        pwrite, i.e. the no-GDS engine the module docstring says may
        *beat* GDS on ``np1 > 1`` meshes) and ``posix`` (POSIX from an
        already-host buffer -- the storage-only reference).  Every
        read is verified against the written pattern, so a fast wrong
        answer cannot pass.
Part C  ``--end-to-end``: the same comparison through the real
        ``save_snapshot`` / ``load_snapshot`` at a real resolution and
        mesh, with the ``pwrite`` prototype driven against the very
        same archive (the prototype here is what the fix would be).

Run on a GPU node, from a directory on the **scratch / parallel
filesystem** the runs actually write to (``--outdir``)::

    .venv/bin/python scripts/gds_probe.py --outdir /scratch/$USER/gdsprobe

    # add the real snapshot path (single process, all visible GPUs)
    .venv/bin/python scripts/gds_probe.py --outdir /scratch/$USER/gdsprobe \
        --end-to-end --dist.platform cuda --np1 4 \
        --ny 193 --nx 256 --nz 256

Part A alone is cheap and needs no GPU allocation beyond a node with
the driver::

    .venv/bin/python scripts/gds_probe.py --env-only

``--cpu-smoke`` runs Part B's POSIX engines only, so the harness can
be validated on a box with no GPU, no cupy and no kvikIO.  Paste the
full stdout back.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

MB = 1 << 20
GB = 1 << 30

# Span sizes that bracket the real range: 512 B is the docstring's
# worst case (np1 = 4 at nx = 256), 1 MiB is a np1 = 1 mesh.
DEFAULT_SPANS = "512,4096,65536,1048576"
DEFAULT_DEPTHS = "4,16,64,256"

NVFS_STATS = Path("/proc/driver/nvidia-fs/stats")
NVFS_VERSION = Path("/proc/driver/nvidia-fs/version")


# ── Part A: environment and the item-6 verdict ───────────────────────


def _read_proc(path: Path, limit: int = 40) -> list[str]:
    try:
        return path.read_text().splitlines()[:limit]
    except OSError:
        return []


def _nvfs_counters() -> dict[str, int]:
    """Parse the integer counters of ``/proc/driver/nvidia-fs/stats``."""
    out: dict[str, int] = {}
    for line in _read_proc(NVFS_STATS, limit=200):
        parts = line.replace("=", " ").split()
        key = parts[0] if parts else ""
        for tok in parts[1:]:
            if tok.isdigit():
                out[f"{key}:{len(out)}"] = int(tok)
    return out


def _mount_of(path: Path) -> tuple[str, str, str]:
    """``(device, fstype, options)`` of the mount *path* lives on."""
    best = ("?", "?", "?")
    best_len = -1
    target = str(path.resolve())
    try:
        lines = Path("/proc/mounts").read_text().splitlines()
    except OSError:
        return best
    for line in lines:
        f = line.split()
        if len(f) < 4:
            continue
        mnt = f[1]
        if (target == mnt or target.startswith(mnt.rstrip("/") + "/")) and (
            len(mnt) > best_len
        ):
            best_len = len(mnt)
            best = (f[0], f[2], f[3])
    return best


def _defaults_dump(kvikio) -> None:
    """Print every readable ``kvikio.defaults`` property."""
    d = kvikio.defaults
    names = sorted(n for n in dir(d) if not n.startswith("_"))
    for n in names:
        try:
            val = getattr(d, n)
            if callable(val):
                val = val()
        except Exception as exc:  # noqa: BLE001
            val = f"<{type(exc).__name__}: {exc}>"
        print(f"    defaults.{n:28} {val!r}")


def _part_a(outdir: Path) -> dict:
    """Environment report; returns what Part B needs to know."""
    import platform
    import sys

    print("=" * 72)
    print("PART A -- environment and the item-6 verdict")
    print("=" * 72)
    print(f"  host      {platform.node()}")
    print(f"  python    {sys.version.split()[0]}")
    dev, fstype, opts = _mount_of(outdir)
    print(f"  outdir    {outdir}")
    print(f"    mount   {dev}  type={fstype}")
    print(f"    options {opts}")
    print(
        "  (GDS needs a supported filesystem -- ext4/xfs with "
        "nvidia-fs,\n   Lustre, GPFS, WekaFS, BeeGFS.  Anything else "
        "falls back to\n   the compat path, which is POSIX with extra "
        "steps.)"
    )
    for var in sorted(os.environ):
        if var.startswith("KVIKIO") or var.startswith("CUFILE"):
            print(f"  env       {var}={os.environ[var]}")

    ver = _read_proc(NVFS_VERSION, limit=5)
    print(f"  nvidia-fs {'; '.join(ver) if ver else 'NOT LOADED'}")
    print(f"  nvidia-fs stats readable: {NVFS_STATS.exists()}")

    have = {"kvikio": False, "cupy": False}
    try:
        import kvikio

        have["kvikio"] = True
        print(f"  kvikio    {getattr(kvikio, '__version__', '?')}")
    except ImportError as exc:
        print(f"  kvikio    NOT IMPORTABLE ({exc})")
        kvikio = None
    try:
        import cupy

        have["cupy"] = True
        print(
            f"  cupy      {cupy.__version__}  "
            f"cuda runtime {cupy.cuda.runtime.runtimeGetVersion()}"
        )
    except ImportError as exc:
        print(f"  cupy      NOT IMPORTABLE ({exc})")

    print("\n  ITEM 6 -- the _gds_available check, step by step")
    if kvikio is None:
        print(
            "    kvikio absent -> _gds_available() is False, the host "
            "path is used\n    and the enum fix is unreachable here."
        )
        return have
    try:
        mode = kvikio.defaults.compat_mode()
    except AttributeError as exc:
        print(
            f"    compat_mode() missing ({exc}) -> the AttributeError "
            "branch fires,\n    host path."
        )
        return have
    on = getattr(type(mode), "ON", None)
    available = mode is not on if on is not None else not mode
    print(f"    compat_mode()        {mode!r}")
    print(f"    type                 {type(mode).__name__}")
    print(f"    isinstance(_, bool)  {isinstance(mode, bool)}")
    print(f"    type(mode).ON        {on!r}")
    print(
        f"    bool(mode)           {bool(mode)}   <- what a bare truth "
        "test would have seen"
    )
    print(f"    _gds_available()     {available}")
    if on is None:
        print(
            "    VERDICT: compat_mode() is not an enum here; the fix is "
            "inert but harmless."
        )
    elif bool(mode) and available:
        print(
            "    VERDICT: the enum fix is LOAD-BEARING -- a bare truth "
            "test would have\n             demoted every run to the host "
            "path, so GDS was dead before it."
        )
    else:
        print(
            "    VERDICT: the fix changes nothing for this mode, but it "
            "is the correct\n             comparison (AUTO would have "
            "been misread)."
        )
    print("\n  kvikio.defaults:")
    _defaults_dump(kvikio)
    return have


def _gds_engaged(path: Path, nbytes: int = 1 << 20) -> None:
    """Write once through kvikIO and diff the nvidia-fs counters.

    The only local evidence that separates a real GDS transfer from a
    compat-mode POSIX write dressed up as one.
    """
    print("\n  GDS engagement probe (nvidia-fs counter diff)")
    try:
        import cupy as cp
        import kvikio
    except ImportError as exc:
        print(f"    skipped ({exc})")
        return
    before = _nvfs_counters()
    buf = cp.ones(nbytes // 8, dtype=cp.float64)
    probe = path / "_gds_engagement_probe.bin"
    try:
        with kvikio.CuFile(str(probe), "w") as f:
            f.write(buf)
        with kvikio.CuFile(str(probe), "r") as f:
            f.read(buf)
    except Exception as exc:  # noqa: BLE001
        print(f"    transfer failed: {type(exc).__name__}: {exc}")
        return
    finally:
        probe.unlink(missing_ok=True)
    after = _nvfs_counters()
    moved = {
        k: (before.get(k, 0), v)
        for k, v in after.items()
        if v != before.get(k, 0)
    }
    if not before and not after:
        print(
            "    no nvidia-fs stats -> cannot distinguish; treat the "
            "Part B numbers\n    as the evidence instead."
        )
    elif moved:
        print(f"    {len(moved)} counters moved -> GDS is ENGAGED")
        for k, (b, a) in list(moved.items())[:12]:
            print(f"      {k:32} {b} -> {a}")
    else:
        print(
            "    no counter moved -> the transfer went through the "
            "COMPAT (POSIX) path"
        )


# ── Part B: the span-pattern benchmark ───────────────────────────────


def _plan(span_bytes: int, total_bytes: int, max_spans: int, stride: int):
    """``(nspans, file_bytes, offsets)`` for one span pattern."""
    nspans = max(1, min(max_spans, total_bytes // span_bytes))
    offsets = [i * span_bytes * stride for i in range(nspans)]
    return nspans, offsets[-1] + span_bytes, offsets


def _fill_host(nspans: int, span_bytes: int):
    """A deterministic host buffer of ``nspans`` distinct spans."""
    import numpy as np

    a = np.arange(nspans * (span_bytes // 8), dtype=np.int64)
    return a


def _fsync(path: Path) -> None:
    """Force the write out -- timed, like the POSIX engines' fsync."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _drop_cache(path: Path) -> None:
    """Best-effort page-cache eviction (a no-op on some filesystems)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _set_kvikio_threads(n: int) -> str:
    """Set the kvikIO thread-pool size across API generations."""
    import kvikio

    d = kvikio.defaults
    for attempt in ("set_num_threads", "num_threads_reset"):
        fn = getattr(d, attempt, None)
        if callable(fn):
            try:
                fn(n)
                return attempt
            except Exception:  # noqa: BLE001, S110
                pass
    setter = getattr(d, "set", None)
    if callable(setter):
        try:
            setter({"num_threads": n})
            return "set"
        except Exception:  # noqa: BLE001, S110
            pass
    return "FAILED (set KVIKIO_NTHREADS in the environment instead)"


def _run_write(engine, path, dev, host, offsets, span_elems, depth):
    """One timed write pass; returns seconds."""
    span_bytes = span_elems * 8
    t0 = time.perf_counter()
    if engine == "posix":
        fd = os.open(path, os.O_WRONLY | os.O_CREAT)
        try:
            mv = memoryview(host).cast("B")
            for i, off in enumerate(offsets):
                os.pwrite(fd, mv[i * span_bytes : (i + 1) * span_bytes], off)
            os.fsync(fd)
        finally:
            os.close(fd)
    elif engine == "buffered":
        # `_write_chunks_host` verbatim minus the staging: one buffered
        # `r+b` handle, `seek` then `write` per span.  Paired with
        # `posix` (same bytes via `os.pwrite`) it prices the Python
        # buffered-I/O layer alone -- a `seek` on a BufferedRandom
        # flushes, and can read back the block it lands in.
        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(fd, "r+b") as f:
            mv = memoryview(host).cast("B")
            for i, off in enumerate(offsets):
                f.seek(off)
                f.write(mv[i * span_bytes : (i + 1) * span_bytes])
            f.flush()
            os.fsync(f.fileno())
    elif engine == "host":
        import cupy as cp

        fd = os.open(path, os.O_WRONLY | os.O_CREAT)
        try:
            for i, off in enumerate(offsets):
                chunk = cp.asnumpy(dev[i * span_elems : (i + 1) * span_elems])
                os.pwrite(fd, memoryview(chunk).cast("B"), off)
            os.fsync(fd)
        finally:
            os.close(fd)
    else:
        import kvikio

        with kvikio.CuFile(str(path), "w") as f:
            if engine == "contig":
                f.write(dev)
            elif engine == "blocking":
                for i, off in enumerate(offsets):
                    f.write(
                        dev[i * span_elems : (i + 1) * span_elems],
                        file_offset=off,
                    )
            else:  # pwrite:D
                futures = []
                for i, off in enumerate(offsets):
                    futures.append(
                        f.pwrite(
                            dev[i * span_elems : (i + 1) * span_elems],
                            file_offset=off,
                        )
                    )
                    if len(futures) >= depth:
                        for fu in futures:
                            fu.get()
                        futures.clear()
                for fu in futures:
                    fu.get()
        # Timed, so every engine is compared at the same durability:
        # the POSIX ones fsync inside their own block above.
        _fsync(path)
    return time.perf_counter() - t0


def _run_read(engine, path, dev, host, offsets, span_elems, depth):
    """One timed read pass into a zeroed buffer; returns (sec, ok)."""
    import numpy as np

    span_bytes = span_elems * 8
    if engine in ("posix", "buffered", "host"):
        got = np.zeros_like(host)
    t0 = time.perf_counter()
    if engine == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            mv = memoryview(got).cast("B")
            for i, off in enumerate(offsets):
                mv[i * span_bytes : (i + 1) * span_bytes] = os.pread(
                    fd, span_bytes, off
                )
        finally:
            os.close(fd)
    elif engine == "buffered":
        with open(path, "rb") as f:
            mv = memoryview(got).cast("B")
            for i, off in enumerate(offsets):
                f.seek(off)
                mv[i * span_bytes : (i + 1) * span_bytes] = f.read(span_bytes)
    elif engine == "host":
        import cupy as cp

        fd = os.open(path, os.O_RDONLY)
        try:
            for i, off in enumerate(offsets):
                chunk = np.frombuffer(
                    os.pread(fd, span_bytes, off), dtype=np.int64
                )
                dev[i * span_elems : (i + 1) * span_elems] = cp.asarray(chunk)
        finally:
            os.close(fd)
    else:
        import kvikio

        dev.fill(0)
        with kvikio.CuFile(str(path), "r") as f:
            if engine == "contig":
                f.read(dev)
            elif engine == "blocking":
                for i, off in enumerate(offsets):
                    f.read(
                        dev[i * span_elems : (i + 1) * span_elems],
                        file_offset=off,
                    )
            else:
                futures = []
                for i, off in enumerate(offsets):
                    futures.append(
                        f.pread(
                            dev[i * span_elems : (i + 1) * span_elems],
                            file_offset=off,
                        )
                    )
                    if len(futures) >= depth:
                        for fu in futures:
                            fu.get()
                        futures.clear()
                for fu in futures:
                    fu.get()
    sec = time.perf_counter() - t0
    if engine in ("posix", "buffered"):
        ok = bool(np.array_equal(got, host))
    else:
        import cupy as cp

        ok = bool(cp.asnumpy(dev == cp.asarray(host)).all())
    return sec, ok


def _part_b(outdir: Path, args, have: dict) -> None:
    print("\n" + "=" * 72)
    print("PART B -- the span pattern, engine by engine (item 7)")
    print("=" * 72)
    print("  A device's snapshot bytes are `nspans` spans of `span` bytes,")
    print("  every `stride`-th block of the file (stride = np1): the")
    print("  sharded k_x axis is the file's fastest-varying one, so a")
    print("  device owns a np1-th of every row.  See snapshot.py's")
    print("  'I/O granularity' section.")
    engines = ["posix", "buffered"]
    if have.get("cupy"):
        engines.append("host")
    if have.get("kvikio") and have.get("cupy"):
        engines += (
            ["blocking"]
            + [
                f"pwrite:{d}"
                for d in (int(v) for v in args.depths.split(",") if v.strip())
            ]
            + ["contig"]
        )
    print(f"  engines: {', '.join(engines)}")
    path = outdir / "gds_probe_span.bin"
    for span_bytes in (int(v) for v in args.spans.split(",") if v.strip()):
        if span_bytes % 8:
            raise SystemExit("span bytes must be a multiple of 8")
        for stride in (int(v) for v in args.strides.split(",") if v.strip()):
            nspans, file_bytes, offsets = _plan(
                span_bytes, args.total_mb * MB, args.max_spans, stride
            )
            vol = nspans * span_bytes
            print(
                f"\n  span {span_bytes:>8} B   stride {stride}   "
                f"{nspans} spans   {vol / MB:.1f} MiB   "
                f"file {file_bytes / MB:.1f} MiB"
            )
            host = _fill_host(nspans, span_bytes)
            dev = None
            if have.get("cupy"):
                import cupy as cp

                dev = cp.asarray(host)
            span_elems = span_bytes // 8
            base_w = base_r = None
            print(
                f"    {'engine':12} {'write':>10} {'GB/s':>8} "
                f"{'us/span':>9}   {'read':>10} {'GB/s':>8} "
                f"{'us/span':>9}  ok"
            )
            for eng in engines:
                depth = int(eng.split(":")[1]) if ":" in eng else 1
                if eng == "contig" and stride != 1:
                    continue  # not a layout this pattern can produce
                offs = [0] if eng == "contig" else offsets
                try:
                    tw = min(
                        _run_write(
                            eng, path, dev, host, offs, span_elems, depth
                        )
                        for _ in range(args.reps)
                    )
                    _drop_cache(path)
                    tr, ok = _run_read(
                        eng, path, dev, host, offs, span_elems, depth
                    )
                    for _ in range(args.reps - 1):
                        _drop_cache(path)
                        tr = min(
                            tr,
                            _run_read(
                                eng,
                                path,
                                dev,
                                host,
                                offs,
                                span_elems,
                                depth,
                            )[0],
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"    {eng:12} FAILED {type(exc).__name__}: {exc}")
                    continue
                if eng == "blocking":
                    base_w, base_r = tw, tr
                print(
                    f"    {eng:12} {tw * 1e3:9.2f}ms "
                    f"{vol / tw / GB:8.2f} {tw / nspans * 1e6:9.2f}   "
                    f"{tr * 1e3:9.2f}ms {vol / tr / GB:8.2f} "
                    f"{tr / nspans * 1e6:9.2f}  {'OK' if ok else 'BAD'}"
                )
            if base_w:
                print(
                    f"    (blocking = the shipped engine: "
                    f"{base_w * 1e3:.2f} ms write / "
                    f"{base_r * 1e3:.2f} ms read is what item 7 would "
                    f"improve on)"
                )
            del host
            if dev is not None:
                del dev
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
    path.unlink(missing_ok=True)


# ── Part C: the real snapshot path ───────────────────────────────────


def _write_chunks_gds_async(
    state, tar_path, comp_offsets, comp_shape, itemsize, depth: int
):
    """``snapshot._write_chunks_gds`` with ``pwrite`` futures.

    Byte-for-byte the same spans and offsets as the shipped writer --
    only the submission discipline differs, so the pair is a clean A/B
    of queue depth.  This *is* the item-7 patch, kept here until the
    measurement says whether to land it.
    """
    import cupy as cp
    import kvikio

    from dnsjax.snapshot import (
        _device_ranges,
        _n_components,
        _require_dense,
        _shard_device_index,
        _spans,
    )

    _, kz_true, kx_true = comp_shape
    for shard in state.addressable_shards:
        flat_idx = _shard_device_index(shard)
        kz_start, nkz, kx_start, nkx = _device_ranges(flat_idx)
        if nkz == 0 or nkx == 0:
            continue
        cp_vec = cp.from_dlpack(shard.data)
        _require_dense(cp_vec)
        with cp_vec.device, kvikio.CuFile(str(tar_path), "r+") as f:
            futures = []
            for comp in range(_n_components()):
                base = comp_offsets[comp]
                for idx, off, _ in _spans(
                    cp_vec.shape[1:],
                    kz_true,
                    kx_true,
                    kz_start,
                    nkz,
                    kx_start,
                    nkx,
                ):
                    futures.append(
                        f.pwrite(
                            cp_vec[comp][idx],
                            file_offset=base + off * itemsize,
                        )
                    )
                    if len(futures) >= depth:
                        for fu in futures:
                            fu.get()
                        futures.clear()
            for fu in futures:
                fu.get()


def _part_c(args, outdir: Path) -> None:
    """Time the real snapshot writer against the pwrite prototype."""
    import jax
    import numpy as np

    from dnsjax import snapshot as snap
    from dnsjax.parameters import params
    from dnsjax.snapshot_meta import snapshot_component_offsets

    print("\n" + "=" * 72)
    print("PART C -- the real snapshot path")
    print("=" * 72)
    m = _import_flow(args.system)
    from dnsjax.random_field import generate_random_state

    to_solver = getattr(m, "to_solver_basis", None)
    state = generate_random_state(
        params.init.random_amplitude,
        params.init.random_smoothness,
        params.init.random_seed,
        params.init.random_mean_flow,
    )
    if to_solver is not None:
        state = to_solver(state)
    jax.block_until_ready(state)
    path = outdir / "gds_probe_snapshot.tar"
    nbytes = int(np.prod(state.shape)) * state.dtype.itemsize
    print(
        f"  system {args.system}  state {tuple(state.shape)} "
        f"{state.dtype}  {nbytes / MB:.1f} MiB"
    )
    print(
        f"  mesh np0={params.dist.np0} np1={params.dist.np1}  "
        f"devices {len(jax.devices())}"
    )

    t0 = time.perf_counter()
    snap.save_snapshot(state, 0.0, 0, path)
    t_save = time.perf_counter() - t0
    print(
        f"  save_snapshot (skeleton + write)  {t_save * 1e3:9.2f} ms"
        f"   {nbytes / t_save / GB:6.2f} GB/s"
    )

    t0 = time.perf_counter()
    back, _t, _it = snap.load_snapshot(path)
    jax.block_until_ready(back)
    t_load = time.perf_counter() - t0
    same = bool(np.array_equal(np.asarray(back), np.asarray(state)))
    print(
        f"  load_snapshot                     {t_load * 1e3:9.2f} ms"
        f"   {nbytes / t_load / GB:6.2f} GB/s   round-trip "
        f"{'OK' if same else 'MISMATCH'}"
    )

    if not snap._gds_available():
        print(
            "  GDS path not active -> the engine A/B below needs it; "
            "Part B\n  still measured the pattern."
        )
        path.unlink(missing_ok=True)
        return

    comp_shape = snap._true_spec_shape()
    itemsize = np.dtype(snap._np_dtype(snap._zarr3_dtype_name())).itemsize
    offsets = snapshot_component_offsets(path)
    depths = [int(v) for v in args.depths.split(",") if v.strip()]
    print(f"  {'engine':14} {'write':>11}  {'GB/s':>7}")
    t0 = time.perf_counter()
    snap._write_chunks_gds(state, path, offsets, comp_shape, itemsize)
    t_block = time.perf_counter() - t0
    print(
        f"  {'blocking':14} {t_block * 1e3:9.2f}ms  "
        f"{nbytes / t_block / GB:7.2f}"
    )
    for d in depths:
        t0 = time.perf_counter()
        _write_chunks_gds_async(state, path, offsets, comp_shape, itemsize, d)
        t = time.perf_counter() - t0
        print(
            f"  {'pwrite:' + str(d):14} {t * 1e3:9.2f}ms  "
            f"{nbytes / t / GB:7.2f}   x{t_block / t:.2f}"
        )
    back, _t, _it = snap.load_snapshot(path)
    ok = bool(np.array_equal(np.asarray(back), np.asarray(state)))
    print(
        f"  after the prototype writes, the archive still reads back "
        f"{'OK' if ok else 'WRONG'}"
    )
    path.unlink(missing_ok=True)


def _import_flow(system: str):
    if system == "plane-couette":
        from dnsjax.flows.wall_bounded import plane_couette as m
    elif system == "plane-poiseuille":
        from dnsjax.flows.wall_bounded import plane_poiseuille as m
    elif system == "pipe":
        from dnsjax.flows.wall_bounded import pipe as m
    else:
        raise SystemExit(f"unsupported system: {system}")
    return m


# ── main ─────────────────────────────────────────────────────────────


def _configure(args) -> None:
    """Params + backend for Part C (before any sharding import)."""
    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    params.dist.np0 = args.np0
    params.dist.np1 = args.np1
    configure_jax_platform(args.platform)
    params.phys.system = args.system
    params.phys.re = 400.0
    params.res.nx = args.nx
    params.res.ny = args.ny
    params.res.nz = args.nz
    params.res.double_precision = True
    update_parameters(Parameters())
    padded_res.set_padded_resolution(params)
    validate_parameters()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outdir",
        default=".",
        help="scratch directory on the filesystem the runs write to",
    )
    ap.add_argument(
        "--env-only",
        action="store_true",
        help="Part A only (no I/O benchmark)",
    )
    ap.add_argument(
        "--end-to-end",
        action="store_true",
        help="also run Part C (needs JAX + a GPU)",
    )
    ap.add_argument(
        "--end-to-end-only",
        action="store_true",
        help="Part C only -- skip the span sweep, which is the "
        "expensive part and does not change between meshes",
    )
    ap.add_argument(
        "--spans",
        default=DEFAULT_SPANS,
        help="span sizes in bytes, comma-separated",
    )
    ap.add_argument(
        "--strides",
        default="1,4",
        help="file stride in spans (= np1), comma-separated",
    )
    ap.add_argument(
        "--depths",
        default=DEFAULT_DEPTHS,
        help="pwrite/pread queue depths, comma-separated",
    )
    ap.add_argument(
        "--total-mb",
        type=int,
        default=256,
        help="volume per timed pass (capped by --max-spans)",
    )
    ap.add_argument(
        "--max-spans",
        type=int,
        default=200_000,
        help="cap on spans per pass (small spans get slow)",
    )
    ap.add_argument(
        "--reps",
        type=int,
        default=3,
        help="timed repeats per engine (the min is reported)",
    )
    ap.add_argument(
        "--kvikio-threads",
        type=int,
        default=0,
        help="kvikIO thread-pool size (0 = leave the default).  Queue "
        "depth without threads may not help; KVIKIO_NTHREADS in the "
        "environment does the same thing.",
    )
    ap.add_argument(
        "--system",
        default="plane-couette",
        help="Part C flow: plane-couette/plane-poiseuille/pipe",
    )
    ap.add_argument("--ny", type=int, default=193)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nz", type=int, default=256)
    ap.add_argument("--np0", type=int, default=1)
    ap.add_argument("--np1", type=int, default=1)
    ap.add_argument(
        "--dist.platform",
        dest="platform",
        default="cuda",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend for Part C (default cuda)",
    )
    ap.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="GPU-less self-check: Part B's POSIX engine only, tiny "
        "volume (validates the harness, not the storage)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.cpu_smoke:
        args.spans, args.strides, args.total_mb = "512,65536", "1,4", 4
        args.reps, args.max_spans = 2, 2000
        # Part C too, on CPU at a tiny resolution: it stops at the
        # engine A/B (no GDS), but everything before that -- the params
        # layering, the real save/load and the round-trip check -- is
        # exercised.
        args.end_to_end, args.platform = True, "cpu"
        args.ny, args.nx, args.nz = 17, 8, 8

    have = _part_a(outdir)
    if have.get("kvikio") and args.kvikio_threads:
        how = _set_kvikio_threads(args.kvikio_threads)
        print(f"\n  kvikio num_threads -> {args.kvikio_threads} via {how}")
    if have.get("kvikio") and have.get("cupy") and not args.cpu_smoke:
        _gds_engaged(outdir)
    if args.env_only:
        return
    if not args.end_to_end_only:
        _part_b(outdir, args, have)
    if args.end_to_end or args.end_to_end_only:
        _configure(args)
        _part_c(args, outdir)
    if args.cpu_smoke:
        print("\n--cpu-smoke PASS: harness runs end-to-end.")


if __name__ == "__main__":
    main()
