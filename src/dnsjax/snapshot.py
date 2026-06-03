"""Snapshot save/load for simulation checkpointing.

Stores the spectral perturbation velocity in zarr3 format as
**three combined per-component files** (one zarr3 chunk per
velocity component), each a clean global array with the `$k_z$`
and `$k_x$` axes de-interleaved across devices.  Only **true**
(unpadded) spectral modes are stored; zero-padded dummy modes
added for 2D mesh divisibility are stripped on save and
re-introduced on load.  Because each file holds the full mode
range, a snapshot can be resumed at **any** ``(np0, np1)``
configuration (np-agnostic).

On-disk layout
--------------
All layouts store each component as ``D = (A, kx_true, B)``
using the true (unpadded) mode counts.  The wall-bounded layout
is user-selectable (``params.outs.snapshot_layout``); periodic
flows always use a single native layout.

=================  ==================  ==========================
layout             D = (A, kx, B)      notes
=================  ==================  ==========================
``wb_native``      ``(kz, kx, y)``     contiguous slab writes
``wb_y_major``     ``(y,  kx, kz)``    y slowest -> fast y-reads
``periodic_native`` ``(kz, kx, ky)``   (no wall-normal grid axis)
=================  ==================  ==========================

Memory
------
GPU memory is never doubled.  The field is streamed to disk one
slab at a time; a full-array transpose is never materialised.

**Definitions** (per device, beyond the resident state;
multiply by ``itemsize`` -- 16 for complex128, 8 for complex64
-- for bytes):

- *slab*: ``N_x / (2·np1) × len(B)`` complex elements, where
  ``len(B)`` is ``N_y`` (``wb_native``), ``N_z - 1``
  (``wb_y_major``), or ``N_y - 1`` (``periodic_native``).
- *component*: one velocity component on one device.
- *shard*: all three components on one device =
  ``3 × component``.

**Extra memory per device by I/O engine:**

======================  =========  ==============
engine                  GPU extra  host extra
======================  =========  ==============
GDS (write and read)    one slab   --
Host + cupy (w and r)   one slab   one slab
Host, no cupy (w and r) --         one shard
======================  =========  ==============

I/O engine
----------
Data is written and read with **raw offset I/O** on both backends
(TensorStore writes at chunk granularity, so per-device sub-range
writes to a shared chunk would race / read-modify-write).
TensorStore is used only to create the zarr3 metadata.

- **GDS** (NVIDIA GPUDirect Storage): when ``kvikio`` and ``cupy``
  are available, slabs move directly between GPU memory and disk.
- **Host + cupy** (NVIDIA GPU, no GDS): slabs are extracted /
  placed on GPU via cupy, transferred one slab at a time with
  ``cupy.asnumpy`` (write) or ``cupy.ndarray.set`` (read).
- **Host, no cupy** (CPU runs, non-NVIDIA GPUs): the full shard
  is copied between device and host at once with ``np.asarray``
  (write) or ``jax.device_put`` (read).

Write modes (``params.outs.snapshot_write_mode``):

- ``"concurrent"`` (default): all processes write their disjoint
  byte ranges at once.  Fast, and safe on POSIX/parallel
  filesystems.
- ``"serial"``: rank-ordered (token-passing) writes, one process
  at a time, for filesystems such as NFS where concurrent writes
  can corrupt data.  No effect for single-process runs.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.sharding import NamedSharding

from .parameters import derived_params, params, periodic_systems
from .sharding import sharding


class SnapshotMismatchError(Exception):
    """Snapshot parameters conflict with current config."""


# ── Runtime detection ─────────────────────────────────────


def _gds_available() -> bool:
    """True when kvikIO + GDS can transfer GPU buffers."""
    try:
        import kvikio

        return not kvikio.defaults.compat_mode()
    except ImportError, AttributeError:
        return False


def _is_periodic() -> bool:
    return params.phys.system in periodic_systems


def _kz_true() -> int:
    """True `$k_z$` mode count (unpadded)."""
    return params.res.nz - 1


def _kx_true() -> int:
    """True `$k_x$` mode count (unpadded)."""
    return params.res.nx // 2


def _true_spec_shape() -> tuple[int, ...]:
    """Unpadded spectral shape (what goes on disk)."""
    if _is_periodic():
        return (params.res.ny - 1, _kz_true(), _kx_true())
    return (_kz_true(), _kx_true(), params.res.ny)


def _device_ranges(
    flat_idx: int,
) -> tuple[int, int, int, int]:
    r"""True `$k_z$` and `$k_x$` ranges for a device.

    Parameters
    ----------
    flat_idx:
        Flat index in ``mesh.devices.flat`` (row-major,
        ``i0 * np1 + i1``).

    Returns
    -------
    kz_start:
        First true `$k_z$` mode owned by the device.
    local_kz_true:
        Number of true `$k_z$` modes (may be fewer than
        the padded count on the last np0-block device).
    kx_start:
        First true `$k_x$` mode owned by the device.
    local_kx_true:
        Number of true `$k_x$` modes.
    """
    i0 = flat_idx // sharding.np1
    i1 = flat_idx % sharding.np1
    local_kz_pad = sharding.nz_spec // sharding.np0
    kz_start = i0 * local_kz_pad
    kz_true = _kz_true()
    local_kz_true = max(0, min(kz_start + local_kz_pad, kz_true) - kz_start)
    local_kx_pad = sharding.nx_spec // sharding.np1
    kx_start = i1 * local_kx_pad
    kx_true = _kx_true()
    local_kx_true = max(0, min(kx_start + local_kx_pad, kx_true) - kx_start)
    return kz_start, local_kz_true, kx_start, local_kx_true


def _strip_padding(comp, local_kz_true: int, local_kx_true: int):
    """Slice padding modes off a local component array."""
    if _is_periodic():
        return comp[:, :local_kz_true, :local_kx_true]
    return comp[:local_kz_true, :local_kx_true, :]


# ── Layout descriptors ────────────────────────────────────
#
# A slab is a contiguous ``(local_kx, b_size)`` block in on-disk
# axis order ``(kx, B)``.  ``extract`` pulls slab ``i`` out of a
# native component shard; ``place`` writes a slab back into a
# native component buffer (the inverse).  Both work on either
# numpy or cupy arrays (``xp`` selects the array module for
# ``ascontiguousarray``).


def _extract_wb_native(comp, i, xp):
    """native ``(kz, kx, y)`` -> slab ``(kx, y)`` at ``kz = i``."""
    return xp.ascontiguousarray(comp[i])


def _place_wb_native(comp, i, slab):
    comp[i] = slab


def _extract_wb_y_major(comp, i, xp):
    """native ``(kz, kx, y)`` -> slab ``(kx, kz)`` at ``y = i``."""
    return xp.ascontiguousarray(comp[:, :, i].T)


def _place_wb_y_major(comp, i, slab):
    comp[:, :, i] = slab.T


def _extract_periodic_native(comp, i, xp):
    """native ``(ky, kz, kx)`` -> slab ``(kx, ky)`` at ``kz = i``."""
    return xp.ascontiguousarray(comp[:, i, :].T)


def _place_periodic_native(comp, i, slab):
    comp[:, i, :] = slab.T


_LAYOUT_FNS: dict[str, tuple[Callable, Callable]] = {
    "wb_native": (_extract_wb_native, _place_wb_native),
    "wb_y_major": (_extract_wb_y_major, _place_wb_y_major),
    "periodic_native": (
        _extract_periodic_native,
        _place_periodic_native,
    ),
}


class _Layout(NamedTuple):
    """On-disk layout descriptor for one component file."""

    name: str
    a_size: int  # number of slabs (outer on-disk axis)
    b_size: int  # inner on-disk axis length
    kx_global: int  # full (unsharded) kx extent
    extract: Callable  # (comp, i, xp) -> slab
    place: Callable  # (comp, i, slab) -> None


def _layout() -> _Layout:
    """Layout to write, from geometry + ``snapshot_layout``.

    All dimensions use **true** (unpadded) mode counts so that
    on-disk snapshots never contain dummy padding modes.
    """
    kx = _kx_true()
    kz = _kz_true()
    if _is_periodic():
        ky = params.res.ny - 1
        return _Layout(
            "periodic_native",
            kz,
            ky,
            kx,
            _extract_periodic_native,
            _place_periodic_native,
        )
    ny = params.res.ny
    if params.outs.snapshot_layout == "native":
        return _Layout(
            "wb_native",
            kz,
            ny,
            kx,
            _extract_wb_native,
            _place_wb_native,
        )
    return _Layout(
        "wb_y_major",
        ny,
        kz,
        kx,
        _extract_wb_y_major,
        _place_wb_y_major,
    )


def _layout_from_meta(meta: dict) -> _Layout:
    """Reconstruct the layout recorded in snapshot metadata."""
    name = meta["layout"]
    _, a_size, kx_global, b_size = meta["on_disk_shape"]
    extract, place = _LAYOUT_FNS[name]
    return _Layout(name, a_size, b_size, kx_global, extract, place)


# ── Geometry / shape helpers ──────────────────────────────


def _padded_local_shape() -> tuple[int, ...]:
    """Padded per-device vector shape for the current mesh.

    Used for allocating load buffers.  Each axis that is
    sharded (``kz`` by ``np0``, ``kx`` by ``np1``) is divided
    by its mesh axis size; unsharded axes keep their full
    (potentially padded) extent.
    """
    local_kz = sharding.nz_spec // sharding.np0
    local_kx = sharding.nx_spec // sharding.np1
    if _is_periodic():
        return (3, params.res.ny - 1, local_kz, local_kx)
    return (3, local_kz, local_kx, params.res.ny)


def _padded_local_shape_snap_ny(snap_ny: int) -> tuple[int, ...]:
    """Padded per-device shape with the *snapshot's* ``ny``.

    Used when loading a wall-bounded snapshot whose ``ny``
    differs from the current run.  ``kz`` and ``kx`` use the
    current mesh padding; ``ny`` comes from the snapshot.
    """
    local_kz = sharding.nz_spec // sharding.np0
    local_kx = sharding.nx_spec // sharding.np1
    if _is_periodic():
        return (3, snap_ny - 1, local_kz, local_kx)
    return (3, local_kz, local_kx, snap_ny)


def _shard_device_index(shard) -> int:
    """Mesh position of a shard's device."""
    devices = list(sharding.mesh.devices.flat)
    return devices.index(shard.device)


def _mesh_device_index(device) -> int:
    """Mesh position of a JAX device."""
    devices = list(sharding.mesh.devices.flat)
    return devices.index(device)


def _zarr3_dtype_name() -> str:
    """Sharding complex type as a zarr3 data-type name."""
    if sharding.complex_type == jnp.complex128:
        return "complex128"
    return "complex64"


def _np_dtype(name: str) -> np.dtype:
    """Numpy dtype from a zarr3 data-type name."""
    return np.dtype(name)


def _chunk_file(store_path: Path, component: int) -> Path:
    """Zarr3 chunk file for one velocity component (single chunk
    per component => chunk grid index ``(comp, 0, 0, 0)``)."""
    return store_path / "c" / str(component) / "0" / "0" / "0"


def _slab_offset(layout: _Layout, i: int, kx_start: int) -> int:
    """Element offset of slab ``i`` for a device whose kx block
    starts at ``kx_start`` in the combined component file."""
    return (i * layout.kx_global + kx_start) * layout.b_size


def _place_into_padded(comp_buf, li: int, slab, nkx: int) -> None:
    r"""Place a true-sized slab into a padded component buffer.

    ``slab`` has shape ``(nkx, b_size)`` (true modes only).
    ``comp_buf`` has the padded local shape; ``li`` is the local
    `$k_z$` slab index.
    """
    if _is_periodic():
        comp_buf[:, li, :nkx] = slab.T
    else:
        comp_buf[li, :nkx, :] = slab


# ── Barrier ───────────────────────────────────────────────


def _barrier(tag: str) -> None:
    """Global device barrier (no-op for single process)."""
    if jax.process_count() > 1:
        from jax.experimental.multihost_utils import (
            sync_global_devices,
        )

        sync_global_devices(tag)


# ── Zarr3 store creation + chunk pre-sizing ───────────────


def _create_store(
    store_path: Path,
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    dtype: str,
) -> None:
    """Create a zarr3 store (metadata only) with TensorStore."""
    import tensorstore as ts

    spec = {
        "driver": "zarr3",
        "kvstore": {
            "driver": "file",
            "path": str(store_path),
        },
        "metadata": {
            "shape": list(shape),
            "data_type": dtype,
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": list(chunk_shape),
                },
            },
            "codecs": [
                {
                    "name": "bytes",
                    "configuration": {"endian": "little"},
                },
            ],
            "fill_value": [0, 0],
        },
    }
    ts.open(spec, create=True, delete_existing=True).result()


def _presize_files(store_path: Path, layout: _Layout, itemsize: int) -> None:
    """Create and size the 3 chunk files so that every device can
    safely write its disjoint byte ranges (incl. multi-host)."""
    nbytes = layout.a_size * layout.kx_global * layout.b_size * itemsize
    for comp in range(3):
        chunk_path = _chunk_file(store_path, comp)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_path, "wb") as f:
            f.truncate(nbytes)


# ── Snapshot metadata ─────────────────────────────────────


def _write_metadata(path: Path, t: float, it: int, layout: _Layout) -> None:
    """Write ``_dnsjax_meta.json`` (process 0 only)."""
    meta = {
        "format_version": 2,
        "t": t,
        "it": it,
        "geometry": ("triply_periodic" if _is_periodic() else "wall_bounded"),
        "system": params.phys.system,
        "layout": layout.name,
        "on_disk_shape": [
            3,
            layout.a_size,
            layout.kx_global,
            layout.b_size,
        ],
        "native_shape": [3, *_true_spec_shape()],
        "dtype": _zarr3_dtype_name(),
        "n_devices": sharding.n_devices,
        "wall_normal_grid": derived_params.wall_normal_grid,
        "params": params.model_dump(mode="json"),
    }
    with open(path / "_dnsjax_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def read_metadata(path: Path) -> dict:
    """Read ``_dnsjax_meta.json``."""
    with open(path / "_dnsjax_meta.json") as f:
        return json.load(f)


# ── GDS I/O ───────────────────────────────────────────────


def _write_chunks_gds(
    state: Array, store_path: Path, layout: _Layout, itemsize: int
) -> None:
    """Stream each local shard to disk slab-by-slab via kvikIO."""
    import cupy as cp
    import kvikio

    is_y_major = layout.name == "wb_y_major"
    for shard in state.addressable_shards:
        flat_idx = _shard_device_index(shard)
        kz_start, nkz, kx_start, nkx = _device_ranges(flat_idx)
        if nkz == 0 or nkx == 0:
            continue
        cp_vec = cp.from_dlpack(shard.data)
        with cp_vec.device:
            for comp in range(3):
                comp_true = _strip_padding(cp_vec[comp], nkz, nkx)
                chunk_path = _chunk_file(store_path, comp)
                with kvikio.CuFile(str(chunk_path), "r+") as f:
                    if is_y_major:
                        for i in range(layout.a_size):
                            slab = layout.extract(comp_true, i, cp)
                            for lkx in range(nkx):
                                row = cp.ascontiguousarray(slab[lkx])
                                off = (
                                    i * layout.kx_global + kx_start + lkx
                                ) * layout.b_size + kz_start
                                f.write(row, file_offset=off * itemsize)
                    else:
                        for li in range(nkz):
                            slab = layout.extract(comp_true, li, cp)
                            off = _slab_offset(layout, kz_start + li, kx_start)
                            f.write(slab, file_offset=off * itemsize)


def _read_chunks_gds(
    store_path: Path,
    layout: _Layout,
    dtype: np.dtype,
    local_shape: tuple[int, ...] | None = None,
) -> list[Array]:
    r"""Read each device's `$k_z$`/`$k_x$` sub-range via kvikIO
    into a padded vector shard (np-agnostic)."""
    import cupy as cp
    import kvikio

    itemsize = dtype.itemsize
    if local_shape is None:
        local_shape = _padded_local_shape()
    is_y_major = layout.name == "wb_y_major"
    per_device: list[Array] = []
    for local_idx, device in enumerate(jax.local_devices()):
        flat_idx = _mesh_device_index(device)
        kz_start, nkz, kx_start, nkx = _device_ranges(flat_idx)
        with cp.cuda.Device(local_idx):
            vec = cp.zeros(local_shape, dtype=dtype)
            if nkz > 0 and nkx > 0:
                for comp in range(3):
                    comp_buf = vec[comp]
                    chunk_path = _chunk_file(store_path, comp)
                    with kvikio.CuFile(str(chunk_path), "r") as f:
                        if is_y_major:
                            row_gpu = cp.empty(nkz, dtype=dtype)
                            for i in range(layout.a_size):
                                for lkx in range(nkx):
                                    off = (
                                        i * layout.kx_global + kx_start + lkx
                                    ) * layout.b_size + kz_start
                                    f.read(
                                        row_gpu,
                                        file_offset=off * itemsize,
                                    )
                                    comp_buf[:nkz, lkx, i] = row_gpu
                        else:
                            slab = cp.empty((nkx, layout.b_size), dtype=dtype)
                            for li in range(nkz):
                                off = _slab_offset(
                                    layout, kz_start + li, kx_start
                                )
                                f.read(slab, file_offset=off * itemsize)
                                _place_into_padded(comp_buf, li, slab, nkx)
            per_device.append(jnp.from_dlpack(vec))
    return per_device


# ── Host I/O ──────────────────────────────────────────────


def _write_serialized(
    write_fn: Callable,
    state: Array,
    store_path: Path,
    layout: _Layout,
    itemsize: int,
) -> None:
    """Rank-ordered (token-passing) write across processes.

    Process ``r`` writes its shards only on its turn, so no two
    processes hold a chunk file open for writing at the same time.
    This is safe on filesystems such as NFS where concurrent
    disjoint-range writes can corrupt data: each process opens,
    writes and *closes* every chunk file within its turn, so the
    next process sees flushed bytes (close-to-open consistency).

    All processes call the same ordered sequence of barrier tags;
    only the write itself is gated on ``process_index``, so the
    collectives stay matched.  For a single process this reduces to
    one write and a no-op barrier (identical to ``concurrent``).
    """
    me = jax.process_index()
    for r in range(jax.process_count()):
        if me == r:
            write_fn(state, store_path, layout, itemsize)
        _barrier(f"snapshot_serial_{r}")


def _write_chunks_host(
    state: Array, store_path: Path, layout: _Layout, itemsize: int
) -> None:
    """Stream each local shard to disk slab-by-slab via host I/O.

    When cupy is available (NVIDIA GPU platforms), slabs are
    extracted on GPU and transferred one at a time via
    ``cupy.asnumpy`` (extra memory: one slab on GPU, one slab on
    host).  Otherwise (CPU runs, non-NVIDIA GPUs), the full shard
    is copied with ``np.asarray`` and slabs are extracted on the
    host (extra host memory: one shard per device).
    """
    is_y_major = layout.name == "wb_y_major"
    try:
        import cupy as cp
    except ImportError:
        cp = None
    for shard in state.addressable_shards:
        flat_idx = _shard_device_index(shard)
        kz_start, nkz, kx_start, nkx = _device_ranges(flat_idx)
        if nkz == 0 or nkx == 0:
            continue
        if cp is not None:
            try:
                vec = cp.from_dlpack(shard.data)
                xp = cp
            except Exception:
                cp = None
        if cp is None:
            vec = np.asarray(shard.data)
            xp = np
        for comp in range(3):
            comp_true = _strip_padding(vec[comp], nkz, nkx)
            chunk_path = _chunk_file(store_path, comp)
            with open(chunk_path, "r+b") as f:
                if is_y_major:
                    for i in range(layout.a_size):
                        slab = layout.extract(comp_true, i, xp)
                        for lkx in range(nkx):
                            row = xp.ascontiguousarray(slab[lkx])
                            off = (
                                i * layout.kx_global + kx_start + lkx
                            ) * layout.b_size + kz_start
                            f.seek(off * itemsize)
                            if cp is not None:
                                f.write(cp.asnumpy(row))
                            else:
                                f.write(row)
                else:
                    for li in range(nkz):
                        slab = layout.extract(comp_true, li, xp)
                        off = _slab_offset(layout, kz_start + li, kx_start)
                        f.seek(off * itemsize)
                        if cp is not None:
                            f.write(cp.asnumpy(slab))
                        else:
                            f.write(slab)


def _read_chunks_host(
    store_path: Path,
    layout: _Layout,
    dtype: np.dtype,
    local_shape: tuple[int, ...] | None = None,
) -> list[Array]:
    r"""Read each device's `$k_z$`/`$k_x$` sub-range via host I/O
    into a padded vector shard (np-agnostic).

    When cupy is available (NVIDIA GPU platforms), the output
    buffer and a reusable slab buffer are allocated on GPU; each
    slab is read from disk to host, copied to the GPU buffer via
    ``cupy.ndarray.set``, and placed into the output (extra
    memory: one slab on GPU, one slab on host).  Otherwise (CPU
    runs, non-NVIDIA GPUs), the output is assembled on the host
    and transferred at the end via ``jax.device_put`` (extra host
    memory: one shard per device).
    """
    itemsize = dtype.itemsize
    if local_shape is None:
        local_shape = _padded_local_shape()
    is_y_major = layout.name == "wb_y_major"
    try:
        import cupy as cp
    except ImportError:
        cp = None
    per_device: list[Array] = []
    for local_idx, device in enumerate(jax.local_devices()):
        flat_idx = _mesh_device_index(device)
        kz_start, nkz, kx_start, nkx = _device_ranges(flat_idx)
        if cp is not None:
            try:
                with cp.cuda.Device(local_idx):
                    vec = cp.zeros(local_shape, dtype=dtype)
                    if nkz > 0 and nkx > 0:
                        if is_y_major:
                            row_gpu = cp.empty(nkz, dtype=dtype)
                            for comp in range(3):
                                comp_buf = vec[comp]
                                chunk_path = _chunk_file(store_path, comp)
                                with open(chunk_path, "rb") as f:
                                    for i in range(layout.a_size):
                                        for lkx in range(nkx):
                                            off = (
                                                i * layout.kx_global
                                                + kx_start
                                                + lkx
                                            ) * layout.b_size + kz_start
                                            f.seek(off * itemsize)
                                            raw = f.read(nkz * itemsize)
                                            row_gpu.set(
                                                np.frombuffer(raw, dtype=dtype)
                                            )
                                            comp_buf[:nkz, lkx, i] = row_gpu
                        else:
                            slab_bytes = nkx * layout.b_size * itemsize
                            slab_gpu = cp.empty(
                                (nkx, layout.b_size), dtype=dtype
                            )
                            for comp in range(3):
                                comp_buf = vec[comp]
                                chunk_path = _chunk_file(store_path, comp)
                                with open(chunk_path, "rb") as f:
                                    for li in range(nkz):
                                        off = _slab_offset(
                                            layout,
                                            kz_start + li,
                                            kx_start,
                                        )
                                        f.seek(off * itemsize)
                                        raw = f.read(slab_bytes)
                                        slab_gpu.set(
                                            np.frombuffer(
                                                raw, dtype=dtype
                                            ).reshape(nkx, layout.b_size)
                                        )
                                        _place_into_padded(
                                            comp_buf,
                                            li,
                                            slab_gpu,
                                            nkx,
                                        )
                    per_device.append(jnp.from_dlpack(vec))
                continue
            except Exception:
                cp = None
        vec = np.zeros(local_shape, dtype=dtype)
        if nkz > 0 and nkx > 0:
            if is_y_major:
                for comp in range(3):
                    comp_buf = vec[comp]
                    chunk_path = _chunk_file(store_path, comp)
                    with open(chunk_path, "rb") as f:
                        for i in range(layout.a_size):
                            for lkx in range(nkx):
                                off = (
                                    i * layout.kx_global + kx_start + lkx
                                ) * layout.b_size + kz_start
                                f.seek(off * itemsize)
                                raw = f.read(nkz * itemsize)
                                row = np.frombuffer(raw, dtype=dtype)
                                comp_buf[:nkz, lkx, i] = row
            else:
                slab_bytes = nkx * layout.b_size * itemsize
                for comp in range(3):
                    comp_buf = vec[comp]
                    chunk_path = _chunk_file(store_path, comp)
                    with open(chunk_path, "rb") as f:
                        for li in range(nkz):
                            off = _slab_offset(layout, kz_start + li, kx_start)
                            f.seek(off * itemsize)
                            raw = f.read(slab_bytes)
                            slab = np.frombuffer(raw, dtype=dtype).reshape(
                                nkx, layout.b_size
                            )
                            _place_into_padded(comp_buf, li, slab, nkx)
        per_device.append(jax.device_put(vec, device))
    return per_device


# ── Public API ────────────────────────────────────────────


def save_snapshot(state: Array, t: float, it: int, path: str | Path) -> None:
    r"""Save the spectral state to a zarr3 snapshot.

    The field is streamed to three combined per-component files
    (clean global arrays, `$k_x$` de-interleaved) without ever
    materialising a full-array transpose.

    Parameters
    ----------
    state:
        Spectral perturbation velocity, shape ``(3, *spec_shape)``,
        complex dtype.
    t:
        Current simulation time.
    it:
        Current iteration count.
    path:
        Directory for the zarr3 store.
    """
    path = Path(path)
    layout = _layout()
    dtype_name = _zarr3_dtype_name()
    itemsize = _np_dtype(dtype_name).itemsize
    on_disk = (3, layout.a_size, layout.kx_global, layout.b_size)

    store_path = path / "state"
    if sharding.main_device:
        path.mkdir(parents=True, exist_ok=True)
        _create_store(store_path, on_disk, (1, *on_disk[1:]), dtype_name)
        _presize_files(store_path, layout, itemsize)
    _barrier("snapshot_create")

    use_gds = _gds_available()
    if use_gds:
        sharding.print("Snapshot: using GDS path")
    write_fn = _write_chunks_gds if use_gds else _write_chunks_host

    if params.outs.snapshot_write_mode == "serial":
        _write_serialized(write_fn, state, store_path, layout, itemsize)
    else:
        write_fn(state, store_path, layout, itemsize)
    _barrier("snapshot_write")

    if sharding.main_device:
        _write_metadata(path, t, it, layout)
    _barrier("snapshot_done")

    sharding.print(f"Snapshot saved to {path}")


def load_snapshot(
    path: str | Path,
) -> tuple[Array, float, int]:
    r"""Load a spectral state from a zarr3 snapshot.

    Each current device reads its own `$k_z$`/`$k_x$` sub-range,
    so a snapshot can be resumed at any ``(np0, np1)``
    configuration.  No full-array inverse transpose is performed.

    Parameters
    ----------
    path:
        Directory of the zarr3 store.

    Returns
    -------
    state:
        Spectral perturbation velocity, shape ``(3, *spec_shape)``,
        correctly sharded.
    t:
        Simulation time at snapshot.
    it:
        Iteration count at snapshot.
    """
    path = Path(path)
    meta = read_metadata(path)
    layout = _layout_from_meta(meta)
    dtype = _np_dtype(meta["dtype"])

    # Detect ny mismatch (wall-bounded only).
    snap_native = tuple(meta["native_shape"])
    curr_true = (3, *_true_spec_shape())
    ny_mismatch = snap_native != curr_true

    if ny_mismatch:
        snap_ny = meta["params"]["res"]["ny"]
        local_shape: tuple[int, ...] | None = _padded_local_shape_snap_ny(
            snap_ny
        )
        if _is_periodic():
            assembly_shape = (
                3,
                snap_ny - 1,
                sharding.nz_spec,
                sharding.nx_spec,
            )
        else:
            assembly_shape = (
                3,
                sharding.nz_spec,
                sharding.nx_spec,
                snap_ny,
            )
    else:
        local_shape = None
        assembly_shape = (3, *sharding.spec_shape)

    store_path = path / "state"
    if _gds_available():
        sharding.print("Snapshot: using GDS path")
        per_device = _read_chunks_gds(store_path, layout, dtype, local_shape)
    else:
        per_device = _read_chunks_host(store_path, layout, dtype, local_shape)

    state = jax.make_array_from_single_device_arrays(
        assembly_shape,
        NamedSharding(sharding.mesh, sharding.spec_vector_shard),
        per_device,
    )
    return state, meta["t"], meta["it"]


def load_y_slice(path: str | Path, y_index: int) -> Array:
    r"""Read a single wall-normal coordinate from a ``y_major``
    snapshot without loading the full array.

    With the `$y$`-slowest layout, a y-slice of each component is
    one contiguous byte range in its chunk file, readable with a
    single seek + read per component.

    Parameters
    ----------
    path:
        Directory of the zarr3 store.
    y_index:
        Wall-normal grid-point index.

    Returns
    -------
    :
        Complex array of shape ``(3, N_{k_z}, N_{k_x})``.

    Raises
    ------
    ValueError
        Unless the snapshot uses the ``y_major`` layout.
    """
    path = Path(path)
    meta = read_metadata(path)

    if meta["layout"] != "wb_y_major":
        raise ValueError(
            "Partial y-reads require a 'y_major' wall-bounded snapshot."
        )

    _, _, kx_global, b_size = meta["on_disk_shape"]
    dtype = _np_dtype(meta["dtype"])
    itemsize = dtype.itemsize
    plane = kx_global * b_size  # one component's y-slice, (kx, kz)
    offset = y_index * plane * itemsize
    nbytes = plane * itemsize
    store_path = path / "state"

    comps: list = []
    if _gds_available():
        import cupy as cp
        import kvikio

        for comp in range(3):
            buf = cp.empty((kx_global, b_size), dtype=dtype)
            chunk_path = _chunk_file(store_path, comp)
            with kvikio.CuFile(str(chunk_path), "r") as f:
                f.read(buf, file_offset=offset)
            comps.append(buf.T)  # (kz, kx)
        return jnp.from_dlpack(cp.stack(comps))

    for comp in range(3):
        chunk_path = _chunk_file(store_path, comp)
        with open(chunk_path, "rb") as f:
            f.seek(offset)
            raw = f.read(nbytes)
        arr = np.frombuffer(raw, dtype=dtype).reshape(kx_global, b_size)
        comps.append(arr.T)  # (kz, kx)
    return jnp.asarray(np.stack(comps))


def validate_snapshot_params(
    path: str | Path,
) -> None:
    r"""Check that snapshot metadata matches current parameters.

    Raises :class:`SnapshotMismatchError` on critical mismatches
    (resolution, precision, flow system, or a streamwise extent
    that the current device count cannot evenly shard).  Prints
    warnings for non-critical differences and an info line when
    the device count differs (resume is np-agnostic).

    Parameters
    ----------
    path:
        Directory of the zarr3 store.
    """
    meta = read_metadata(Path(path))
    snap_params = meta.get("params", {})
    current = params.model_dump(mode="json")

    # Critical: must match exactly
    critical = {
        ("res", "nx"): "x resolution",
        ("res", "nz"): "z resolution",
        ("res", "double_precision"): "precision",
        ("phys", "system"): "flow system",
    }
    for (section, key), label in critical.items():
        snap_val = snap_params.get(section, {}).get(key)
        curr_val = current.get(section, {}).get(key)
        if snap_val is not None and snap_val != curr_val:
            raise SnapshotMismatchError(
                f"{label}: snapshot {key}={snap_val}, current {key}={curr_val}"
            )

    native = meta.get("native_shape")
    expected = [3, *_true_spec_shape()]
    if native is not None and list(native) != expected:
        if _is_periodic():
            raise SnapshotMismatchError(
                f"Shape: snapshot {native}, expected {expected}"
            )
        # Wall-bounded: allow ny mismatch (last axis).
        if list(native)[:3] != expected[:3]:
            raise SnapshotMismatchError(
                f"Shape (non-ny axes): snapshot "
                f"{native[:3]}, expected {expected[:3]}"
            )

    # ny mismatch: info (wall-normal interpolation will handle it)
    snap_ny = snap_params.get("res", {}).get("ny")
    curr_ny = current.get("res", {}).get("ny")
    if snap_ny is not None and snap_ny != curr_ny:
        sharding.print(
            f"Info: ny changed: {snap_ny} -> {curr_ny} "
            f"(will interpolate wall-normal grid)"
        )

    # Warnings
    warn_fields = {
        ("phys", "re"): "Reynolds number",
        ("step", "dt"): "time step",
        ("step", "implicitness"): "implicitness",
        ("geo", "lx"): "Lx",
        ("geo", "lz"): "Lz",
        ("geo", "tilt_degree"): "tilt angle",
        ("res", "fd_order"): "FD stencil order",
        ("solver", "backend"): "solver backend",
    }
    for (section, key), label in warn_fields.items():
        snap_val = snap_params.get(section, {}).get(key)
        curr_val = current.get(section, {}).get(key)
        if (
            snap_val is not None
            and curr_val is not None
            and snap_val != curr_val
        ):
            sharding.print(
                f"Warning: {label} changed: {snap_val} -> {curr_val}"
            )

    snap_np = meta.get("n_devices")
    if snap_np is not None and snap_np != sharding.n_devices:
        sharding.print(
            f"Info: device count {snap_np} -> {sharding.n_devices} "
            f"(np0={sharding.np0}, np1={sharding.np1}; "
            f"np-agnostic resume)"
        )
