"""Snapshot save/load for simulation checkpointing.

Stores the spectral perturbation velocity in zarr3 format as
**three combined per-component files** (one zarr3 chunk per
velocity component), each a clean global array with the `$k_x$`
axis de-interleaved across devices.  Because each file holds the
full `$k_x$` range, a snapshot can be resumed at **any** device
count (np-agnostic): on load, every device reads its own `$k_x$`
sub-range.

On-disk layout
--------------
All layouts store each component as ``D = (A, kx_global, B)``.
The wall-bounded layout is user-selectable
(``params.outs.snapshot_layout``); periodic flows always use a
single native layout.

================  ==================  ===========================
layout            D = (A, kx, B)      notes
================  ==================  ===========================
``wb_native``     ``(kz, kx, y)``     zero-copy slab writes
``wb_y_major``    ``(y,  kx, kz)``    y slowest -> fast y-reads
``periodic_native`` ``(kz, kx, ky)``  (no wall-normal grid axis)
================  ==================  ===========================

Memory
------
GPU memory is never doubled.  The field is streamed to disk one
``(local_kx, len(B))`` slab at a time; a full-array transpose is
never materialised.

**Definitions** (per device, beyond the resident state;
multiply by ``itemsize`` -- 16 for complex128, 8 for complex64
-- for bytes):

- *slab*: ``N_x / (2·np) × len(B)`` complex elements, where
  ``len(B)`` is ``N_y`` (``wb_native``), ``N_z - 1``
  (``wb_y_major``), or ``N_y - 1`` (``periodic_native``).
- *component*: one velocity component on one device =
  ``(N_z - 1) × N_x / (2·np) × N_y`` (wall-bounded) or
  ``(N_y - 1) × (N_z - 1) × N_x / (2·np)`` (periodic).
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
    """Layout to write, from geometry + ``snapshot_layout``."""
    spec = sharding.spec_shape
    if _is_periodic():
        # spec = (ky, kz, kx)
        return _Layout(
            "periodic_native",
            spec[1],
            spec[0],
            spec[2],
            _extract_periodic_native,
            _place_periodic_native,
        )
    # spec = (kz, kx, y)
    if params.outs.snapshot_layout == "native":
        return _Layout(
            "wb_native",
            spec[0],
            spec[2],
            spec[1],
            _extract_wb_native,
            _place_wb_native,
        )
    return _Layout(
        "wb_y_major",
        spec[2],
        spec[0],
        spec[1],
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


def _kx_axis_in_component() -> int:
    """kx axis within a single-component native array."""
    return 2 if _is_periodic() else 1


def _native_local_shape(local_kx: int) -> tuple[int, ...]:
    """Native vector-shard shape for a device owning ``local_kx``
    streamwise modes (the result of a load)."""
    spec = list(sharding.spec_shape)
    spec[_kx_axis_in_component()] = local_kx
    return (3, *spec)


def _native_local_shape_from_meta(
    meta: dict, local_kx: int
) -> tuple[int, ...]:
    """Native vector-shard shape using the **snapshot's** shape.

    Used when loading a snapshot whose wall-normal resolution
    (``ny``) differs from the current run.
    """
    native = meta["native_shape"]
    spec = list(native[1:])  # drop leading 3
    ax = 2 if meta.get("geometry") == "triply_periodic" else 1
    spec[ax] = local_kx
    return (3, *spec)


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
        "native_shape": [3, *sharding.spec_shape],
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

    local_kx = layout.kx_global // sharding.n_devices
    for shard in state.addressable_shards:
        cp_vec = cp.from_dlpack(shard.data)
        kx_start = _shard_device_index(shard) * local_kx
        with cp_vec.device:
            for comp in range(3):
                comp_arr = cp_vec[comp]
                chunk_path = _chunk_file(store_path, comp)
                with kvikio.CuFile(str(chunk_path), "r+") as f:
                    for i in range(layout.a_size):
                        slab = layout.extract(comp_arr, i, cp)
                        off = _slab_offset(layout, i, kx_start)
                        f.write(slab, file_offset=off * itemsize)


def _read_chunks_gds(
    store_path: Path,
    layout: _Layout,
    dtype: np.dtype,
    local_shape: tuple[int, ...] | None = None,
) -> list[Array]:
    """Read each device's kx sub-range via kvikIO into a native
    vector shard (np-agnostic)."""
    import cupy as cp
    import kvikio

    local_kx = layout.kx_global // sharding.n_devices
    itemsize = dtype.itemsize
    if local_shape is None:
        local_shape = _native_local_shape(local_kx)
    per_device: list[Array] = []
    for local_idx, device in enumerate(jax.local_devices()):
        kx_start = _mesh_device_index(device) * local_kx
        with cp.cuda.Device(local_idx):
            vec = cp.empty(local_shape, dtype=dtype)
            slab = cp.empty((local_kx, layout.b_size), dtype=dtype)
            for comp in range(3):
                comp_buf = vec[comp]
                chunk_path = _chunk_file(store_path, comp)
                with kvikio.CuFile(str(chunk_path), "r") as f:
                    for i in range(layout.a_size):
                        off = _slab_offset(layout, i, kx_start)
                        f.read(slab, file_offset=off * itemsize)
                        layout.place(comp_buf, i, slab)
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
    local_kx = layout.kx_global // sharding.n_devices
    try:
        import cupy as cp
    except ImportError:
        cp = None
    for shard in state.addressable_shards:
        kx_start = _shard_device_index(shard) * local_kx
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
            comp_arr = vec[comp]
            chunk_path = _chunk_file(store_path, comp)
            with open(chunk_path, "r+b") as f:
                for i in range(layout.a_size):
                    slab = layout.extract(comp_arr, i, xp)
                    off = _slab_offset(layout, i, kx_start)
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
    """Read each device's kx sub-range via host I/O into a native
    vector shard (np-agnostic).

    When cupy is available (NVIDIA GPU platforms), the output
    buffer and a reusable slab buffer are allocated on GPU; each
    slab is read from disk to host, copied to the GPU buffer via
    ``cupy.ndarray.set``, and placed into the output (extra
    memory: one slab on GPU, one slab on host).  Otherwise (CPU
    runs, non-NVIDIA GPUs), the output is assembled on the host
    and transferred at the end via ``jax.device_put`` (extra host
    memory: one shard per device).
    """
    local_kx = layout.kx_global // sharding.n_devices
    itemsize = dtype.itemsize
    slab_bytes = local_kx * layout.b_size * itemsize
    if local_shape is None:
        local_shape = _native_local_shape(local_kx)
    try:
        import cupy as cp
    except ImportError:
        cp = None
    per_device: list[Array] = []
    for local_idx, device in enumerate(jax.local_devices()):
        kx_start = _mesh_device_index(device) * local_kx
        if cp is not None:
            try:
                with cp.cuda.Device(local_idx):
                    vec = cp.empty(local_shape, dtype=dtype)
                    slab_gpu = cp.empty((local_kx, layout.b_size), dtype=dtype)
                    for comp in range(3):
                        comp_buf = vec[comp]
                        chunk_path = _chunk_file(store_path, comp)
                        with open(chunk_path, "rb") as f:
                            for i in range(layout.a_size):
                                off = _slab_offset(layout, i, kx_start)
                                f.seek(off * itemsize)
                                raw = f.read(slab_bytes)
                                slab_gpu.set(
                                    np.frombuffer(raw, dtype=dtype).reshape(
                                        local_kx, layout.b_size
                                    )
                                )
                                layout.place(comp_buf, i, slab_gpu)
                    per_device.append(jnp.from_dlpack(vec))
                continue
            except Exception:
                cp = None
        vec = np.empty(local_shape, dtype=dtype)
        for comp in range(3):
            comp_buf = vec[comp]
            chunk_path = _chunk_file(store_path, comp)
            with open(chunk_path, "rb") as f:
                for i in range(layout.a_size):
                    off = _slab_offset(layout, i, kx_start)
                    f.seek(off * itemsize)
                    raw = f.read(slab_bytes)
                    slab = np.frombuffer(raw, dtype=dtype).reshape(
                        local_kx, layout.b_size
                    )
                    layout.place(comp_buf, i, slab)
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

    Each current device reads its own `$k_x$` sub-range, so a
    snapshot can be resumed at any device count.  No full-array
    inverse transpose is performed.

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

    # When the snapshot's ny differs from the current run,
    # allocate read buffers at the snapshot's ny.
    snap_native = tuple(meta["native_shape"])
    curr_native = (3, *sharding.spec_shape)
    local_kx = layout.kx_global // sharding.n_devices
    if snap_native != curr_native:
        local_shape: tuple[int, ...] | None = _native_local_shape_from_meta(
            meta, local_kx
        )
    else:
        local_shape = None  # default path

    store_path = path / "state"
    if _gds_available():
        sharding.print("Snapshot: using GDS path")
        per_device = _read_chunks_gds(store_path, layout, dtype, local_shape)
    else:
        per_device = _read_chunks_host(store_path, layout, dtype, local_shape)

    state = jax.make_array_from_single_device_arrays(
        snap_native,
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
    expected = [3, *sharding.spec_shape]
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

    kx_global = params.res.nx // 2
    if kx_global % sharding.n_devices != 0:
        raise SnapshotMismatchError(
            f"kx extent {kx_global} is not divisible by the current "
            f"device count np={sharding.n_devices}"
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
            f"(np-agnostic resume)"
        )
