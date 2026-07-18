"""Snapshot save/load for simulation checkpointing.

A snapshot is a **single uncompressed tar archive** wrapping a
zarr3 store plus a JSON metadata member::

    snapshot.tar                 (uncompressed tar; format_version 4)
      _dnsjax_meta.json          plain JSON metadata
      _dnsjax_stats.json         plain JSON stats (optional)
      state/zarr.json            zarr3 array metadata
      state/c/0/0/0/0            component 0 chunk: raw LE complex
      state/c/1/0/0/0            component 1 chunk
      state/c/2/0/0/0            component 2 chunk

The spectral perturbation velocity is stored as **three combined
per-component zarr3 chunks** (one chunk per velocity component),
each a clean global array with the `$k_z$` and `$k_x$` axes
de-interleaved across devices.  Only **true** (unpadded) spectral
modes are stored; zero-valued padding modes added for 2D mesh
divisibility are stripped on save and re-introduced on load.
Because each chunk holds the full mode range, a snapshot can be
resumed at **any** ``(np0, np1)`` configuration (np-agnostic).

Because the tar is uncompressed and each chunk is stored
contiguously, the archive is readable with standard tools and no
dnsjax: ``tar xf snapshot.tar`` yields a directory whose ``state/``
is a valid zarr3 store (open with zarr-python or TensorStore), with
``_dnsjax_meta.json`` describing the axis interpretation.  Worst
case (no zarr library) each chunk is raw little-endian complex with
shape/dtype from ``state/zarr.json`` -- ``numpy.fromfile`` plus a
reshape.

On-disk layout
--------------
All layouts store each component as ``D = (A, kx_true, B)``
using the true (unpadded) mode counts.

==============  ==================  ==========================
layout          D = (A, kx, B)      notes
==============  ==================  ==========================
``walled``      ``(y,  kx, kz)``    wall-bounded (y slowest)
``periodic``    ``(kz, kx, ky)``    triply-periodic
==============  ==================  ==========================

Memory
------
GPU memory is never doubled.  The field is streamed to disk one
slab at a time; a full-array transpose is never materialised.

**Definitions** (per device, beyond the resident state;
multiply by ``itemsize`` -- 16 for complex128, 8 for complex64
-- for bytes):

- *slab*: ``N_x / (2·np1) × len(B)`` complex elements, where
  ``len(B)`` is ``N_z - 1`` (``walled``) or ``N_y - 1``
  (``periodic``).
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
writes to a shared chunk would race / read-modify-write).  Process
0 lays out the whole archive once -- the tar headers plus
sparse-reserved (zero-filled) component data regions -- and every
device then writes its disjoint byte ranges directly into the one
file at ``component_offset + within_component_offset``.  The
component base offsets are the tar members' ``offset_data`` (read
back via ``tarfile``; 512-aligned, so GDS alignment is preserved).
TensorStore is used only to generate the ``zarr.json`` bytes (in a
throwaway temporary directory); compression is never used (it would
break random-access streaming).

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

Metadata and versioning
-----------------------
The ``_dnsjax_meta.json`` member embeds ``t``, ``it``, ``isnap`` (the
snapshot-lineage index this file was written with), the on-disk
``layout`` name, the global (true, unpadded) shapes,
``wall_normal_grid`` (the wall-normal grid points as a float array
for wall-bounded flows, ``None`` for periodic), and the
flow-relevant, public-named, resolved parameter dump
(:func:`dnsjax.param_surface.recorded_params_dump`) for resume
validation.  It is read with the standard library (no JAX) via
:mod:`dnsjax.snapshot_meta`, shared with
:func:`dnsjax.parameters.read_snapshot_params`.

When stats are supplied, an optional ``_dnsjax_stats.json`` member
holds the state's physical diagnostics (the ``get_stats`` dict as
``{name: value}``); readers that do not need it simply ignore the
extra member.

The on-disk format is ``format_version: 4``; snapshots older than 4
(internal-named full parameter dumps) are rejected at read
(:func:`dnsjax.snapshot_meta.read_snapshot_meta`), never translated.
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
from .snapshot_meta import (
    git_hash,
    read_snapshot_meta,
    snapshot_component_offsets,
)


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
    return (params.res.ny, _kz_true(), _kx_true())


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
    return comp[:, :local_kz_true, :local_kx_true]


# ── Layout descriptors ────────────────────────────────────
#
# A slab is a contiguous ``(local_kx, b_size)`` block in on-disk
# axis order ``(kx, B)``.  ``extract`` pulls slab ``i`` out of a
# native component shard; ``place`` writes a slab back into a
# native component buffer (the inverse).  Both work on either
# numpy or cupy arrays (``xp`` selects the array module for
# ``ascontiguousarray``).


def _extract_walled(comp, i, xp):
    """native ``(y, kz, kx)`` -> slab ``(kx, kz)`` at ``y = i``."""
    return xp.ascontiguousarray(comp[i].T)


def _place_walled(comp, i, slab):
    comp[i] = slab.T


def _extract_periodic(comp, i, xp):
    """native ``(ky, kz, kx)`` -> slab ``(kx, ky)`` at ``kz = i``."""
    return xp.ascontiguousarray(comp[:, i, :].T)


def _place_periodic(comp, i, slab):
    comp[:, i, :] = slab.T


_LAYOUT_FNS: dict[str, tuple[Callable, Callable]] = {
    "walled": (_extract_walled, _place_walled),
    "periodic": (
        _extract_periodic,
        _place_periodic,
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
    """Layout to write from the current geometry.

    All dimensions use **true** (unpadded) mode counts so that
    on-disk snapshots never contain padding modes.
    """
    kx = _kx_true()
    kz = _kz_true()
    if _is_periodic():
        ky = params.res.ny - 1
        return _Layout(
            "periodic",
            kz,
            ky,
            kx,
            _extract_periodic,
            _place_periodic,
        )
    return _Layout(
        "walled",
        params.res.ny,
        kz,
        kx,
        _extract_walled,
        _place_walled,
    )


def _layout_from_meta(meta: dict) -> _Layout:
    """Reconstruct the layout recorded in snapshot metadata."""
    name = meta["layout"]
    _, a_size, kx_global, b_size = meta["on_disk_shape"]
    extract, place = _LAYOUT_FNS[name]
    return _Layout(name, a_size, b_size, kx_global, extract, place)


# ── Geometry / shape helpers ──────────────────────────────


def _n_components() -> int:
    """Number of stacked state components for the current system.

    The flow spec's ``n_components`` (3 velocity components unless
    the flow carries more, e.g. the 9-component viscoelastic state).
    ``validate_snapshot_params`` enforces the ``phys.system`` match,
    so a resumed snapshot's component count always equals this -- the
    single source of truth for both save and load.
    """
    from .flows.registry import spec_for

    return spec_for(params.phys.system).n_components


def _padded_local_shape() -> tuple[int, ...]:
    """Padded per-device vector shape for the current mesh.

    Used for allocating load buffers.  Each axis that is
    sharded (``kz`` by ``np0``, ``kx`` by ``np1``) is divided
    by its mesh axis size; unsharded axes keep their full
    (potentially padded) extent.
    """
    local_kz = sharding.nz_spec // sharding.np0
    local_kx = sharding.nx_spec // sharding.np1
    nc = _n_components()
    if _is_periodic():
        return (nc, params.res.ny - 1, local_kz, local_kx)
    return (nc, params.res.ny, local_kz, local_kx)


def _padded_local_shape_snap_ny(snap_ny: int) -> tuple[int, ...]:
    """Padded per-device shape with the *snapshot's* ``ny``.

    Used when loading a wall-bounded snapshot whose ``ny``
    differs from the current run.  ``kz`` and ``kx`` use the
    current mesh padding; ``ny`` comes from the snapshot.
    """
    local_kz = sharding.nz_spec // sharding.np0
    local_kx = sharding.nx_spec // sharding.np1
    nc = _n_components()
    if _is_periodic():
        return (nc, snap_ny - 1, local_kz, local_kx)
    return (nc, snap_ny, local_kz, local_kx)


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


def _slab_offset(layout: _Layout, i: int, kx_start: int) -> int:
    """Element offset of slab ``i`` for a device whose kx block
    starts at ``kx_start`` in the combined component file."""
    return (i * layout.kx_global + kx_start) * layout.b_size


def _place_into_padded(comp_buf, li: int, slab, nkx: int) -> None:
    r"""Place a true-sized slab into a padded component buffer.

    ``slab`` has shape ``(nkx, b_size)`` (true modes only).
    ``comp_buf`` has the padded local shape; ``li`` is the local
    `$k_z$` slab index.  Only used for ``periodic``.
    """
    comp_buf[:, li, :nkx] = slab.T


# ── In-memory per-device assembly ─────────────────────────


def assemble_local_shards(
    fill_local: Callable[[np.ndarray, int, int, int, int], None],
    *,
    dtype: np.dtype | None = None,
) -> Array:
    r"""Assemble a sharded spectral state from per-device-generated shards.

    The generator counterpart of :func:`load_snapshot`'s per-device read:
    each local device's padded shard is allocated zero-filled (shape
    :func:`_padded_local_shape`) and handed to
    ``fill_local(buf, kz_start, nkz, kx_start, nkx)``, which fills
    ``buf[:, :, :nkz, :nkx]`` with that device's **true** modes -- the
    global axis-2 (`$k_z$` / `$m$`, ``np0``) range
    ``[kz_start, kz_start + nkz)`` and the global axis-3 (`$k_x$` /
    `$k_{z,\mathrm{ax}}$`, ``np1``) range ``[kx_start, kx_start + nkx)``;
    the trailing padding modes stay zero.  Shards are placed onto
    ``sharding.spec_vector_shard`` with
    ``jax.make_array_from_single_device_arrays`` -- np-agnostic, and **no
    full array is ever materialised** on any device (so in-process random /
    rolls ICs match dnsjax's per-device construction idiom).

    Parameters
    ----------
    fill_local:
        Callback filling one device's local buffer in place.
    dtype:
        Buffer dtype; defaults to the configured complex type.
    """
    if dtype is None:
        dtype = _np_dtype(_zarr3_dtype_name())
    local_shape = _padded_local_shape()
    per_device: list[Array] = []
    for device in jax.local_devices():
        flat_idx = _mesh_device_index(device)
        kz_start, nkz, kx_start, nkx = _device_ranges(flat_idx)
        buf = np.zeros(local_shape, dtype=dtype)
        fill_local(buf, kz_start, nkz, kx_start, nkx)
        per_device.append(jax.device_put(buf, device))
    return jax.make_array_from_single_device_arrays(
        (_n_components(), *sharding.spec_shape),
        NamedSharding(sharding.mesh, sharding.spec_vector_shard),
        per_device,
    )


# ── Barrier ───────────────────────────────────────────────


def _barrier(tag: str) -> None:
    """Global device barrier (no-op for single process)."""
    if jax.process_count() > 1:
        from jax.experimental.multihost_utils import (
            sync_global_devices,
        )

        sync_global_devices(tag)


# ── Zarr3 metadata + tar skeleton ─────────────────────────


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


def _zarr_json_bytes(
    on_disk: tuple[int, ...], chunk_shape: tuple[int, ...], dtype: str
) -> bytes:
    """Generate the zarr3 ``zarr.json`` bytes for the component store.

    TensorStore only writes metadata to a *directory* kvstore, so the
    store is created in a throwaway temporary directory and its
    ``zarr.json`` is read back.  This keeps the embedded zarr3 metadata
    spec-valid without hand-rolling the schema.
    """
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_zarr_"))
    try:
        _create_store(tmp / "state", on_disk, chunk_shape, dtype)
        return (tmp / "state" / "zarr.json").read_bytes()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _tar_header(name: str, size: int) -> bytes:
    """Deterministic 512-byte (or PAX multi-block) tar header.

    ``mtime``/ownership are pinned to ``0`` so the skeleton is
    byte-reproducible (the ``serial`` vs ``concurrent`` write modes must
    yield identical archives).  PAX format keeps the ``size`` field
    exact for components larger than 8 GiB.
    """
    import tarfile

    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    return info.tobuf(format=tarfile.PAX_FORMAT)


def _write_tar_skeleton(
    tar_path: Path,
    layout: _Layout,
    itemsize: int,
    meta_bytes: bytes,
    zarr_bytes: bytes,
    stats_bytes: bytes | None = None,
) -> None:
    """Lay out the whole uncompressed tar (process 0 only).

    Small members (metadata, the optional stats JSON, ``zarr.json``) are
    written in full; the three component members get a correct header
    followed by a sparse-reserved, zero-filled data region padded to the
    512-byte block boundary.  The archive ends with the two zero blocks
    tar expects.  After this the file is full-length, so every device can
    safely write its disjoint byte ranges into the component regions.
    """
    comp_nbytes = layout.a_size * layout.kx_global * layout.b_size * itemsize
    comp_padded = comp_nbytes + (-comp_nbytes) % 512
    members = [("_dnsjax_meta.json", meta_bytes)]
    if stats_bytes is not None:
        members.append(("_dnsjax_stats.json", stats_bytes))
    members.append(("state/zarr.json", zarr_bytes))
    with open(tar_path, "wb") as f:
        for name, data in members:
            f.write(_tar_header(name, len(data)))
            f.write(data)
            f.write(b"\x00" * ((-len(data)) % 512))
        for comp in range(_n_components()):
            f.write(_tar_header(f"state/c/{comp}/0/0/0", comp_nbytes))
            # Sparse-reserve the (zeroed) data + block padding.
            f.seek(comp_padded - 1, 1)
            f.write(b"\x00")
        f.write(b"\x00" * 1024)  # end-of-archive marker


# ── Snapshot metadata ─────────────────────────────────────


def _metadata_bytes(
    t: float, it: int, layout: _Layout, isnap: int = 0
) -> bytes:
    """Serialise the ``_dnsjax_meta.json`` member content.

    ``git_hash`` records the code revision that wrote the snapshot
    (provenance only -- never read back on load).  Additive keys like
    it need no ``format_version`` bump: readers use targeted lookups
    and ignore unknown keys.  Version 4 records ``params`` as the
    flow-relevant, **public-named**, resolved dump plus the relevant
    extension sections (e.g. ``force``, ``probes``;
    :func:`dnsjax.param_surface.recorded_params_dump`); readers map it
    back via :func:`dnsjax.flows.registry.internalize_stored` /
    ``stored_value``, and pre-4 snapshots (internal-named full dumps)
    are rejected at :func:`dnsjax.snapshot_meta.read_snapshot_meta`.
    """
    from .param_surface import recorded_params_dump

    meta = {
        "format_version": 4,
        "git_hash": git_hash(),
        "t": t,
        "it": it,
        "isnap": isnap,
        "geometry": ("triply_periodic" if _is_periodic() else "wall_bounded"),
        "system": params.phys.system,
        "layout": layout.name,
        "on_disk_shape": [
            _n_components(),
            layout.a_size,
            layout.kx_global,
            layout.b_size,
        ],
        "native_shape": [_n_components(), *_true_spec_shape()],
        "dtype": _zarr3_dtype_name(),
        "n_devices": sharding.n_devices,
        "wall_normal_grid": derived_params.wall_normal_grid,
        "params": recorded_params_dump(params),
    }
    return json.dumps(meta, indent=2, default=str).encode("utf-8")


def read_metadata(path: Path) -> dict:
    """Read the ``_dnsjax_meta.json`` member of a snapshot tar."""
    return read_snapshot_meta(path)


def _stats_json_bytes(stats: dict) -> bytes:
    """Serialise a ``get_stats`` dict for the ``_dnsjax_stats.json``
    member, converting the (replicated) device scalars to host floats."""
    return json.dumps(
        {k: float(v) for k, v in stats.items()}, indent=2
    ).encode("utf-8")


# ── GDS I/O ───────────────────────────────────────────────


def _write_chunks_gds(
    state: Array,
    tar_path: Path,
    comp_offsets: dict[int, int],
    layout: _Layout,
    itemsize: int,
) -> None:
    """Stream each local shard to the tar slab-by-slab via kvikIO.

    Each component's chunk lives at ``comp_offsets[comp]`` inside the
    single archive; all writes are at that base plus the in-component
    byte offset.
    """
    import cupy as cp
    import kvikio

    is_walled = layout.name == "walled"
    for shard in state.addressable_shards:
        flat_idx = _shard_device_index(shard)
        kz_start, nkz, kx_start, nkx = _device_ranges(flat_idx)
        if nkz == 0 or nkx == 0:
            continue
        cp_vec = cp.from_dlpack(shard.data)
        with cp_vec.device:
            for comp in range(_n_components()):
                comp_true = _strip_padding(cp_vec[comp], nkz, nkx)
                base = comp_offsets[comp]
                with kvikio.CuFile(str(tar_path), "r+") as f:
                    if is_walled:
                        for i in range(layout.a_size):
                            slab = layout.extract(comp_true, i, cp)
                            for lkx in range(nkx):
                                row = cp.ascontiguousarray(slab[lkx])
                                off = (
                                    i * layout.kx_global + kx_start + lkx
                                ) * layout.b_size + kz_start
                                f.write(row, file_offset=base + off * itemsize)
                    else:
                        for li in range(nkz):
                            slab = layout.extract(comp_true, li, cp)
                            off = _slab_offset(layout, kz_start + li, kx_start)
                            f.write(slab, file_offset=base + off * itemsize)


def _read_chunks_gds(
    tar_path: Path,
    comp_offsets: dict[int, int],
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
    is_walled = layout.name == "walled"
    per_device: list[Array] = []
    for local_idx, device in enumerate(jax.local_devices()):
        flat_idx = _mesh_device_index(device)
        kz_start, nkz, kx_start, nkx = _device_ranges(flat_idx)
        with cp.cuda.Device(local_idx):
            vec = cp.zeros(local_shape, dtype=dtype)
            if nkz > 0 and nkx > 0:
                for comp in range(_n_components()):
                    comp_buf = vec[comp]
                    base = comp_offsets[comp]
                    with kvikio.CuFile(str(tar_path), "r") as f:
                        if is_walled:
                            row_gpu = cp.empty(nkz, dtype=dtype)
                            for i in range(layout.a_size):
                                for lkx in range(nkx):
                                    off = (
                                        i * layout.kx_global + kx_start + lkx
                                    ) * layout.b_size + kz_start
                                    f.read(
                                        row_gpu,
                                        file_offset=base + off * itemsize,
                                    )
                                    comp_buf[i, :nkz, lkx] = row_gpu
                        else:
                            slab = cp.empty((nkx, layout.b_size), dtype=dtype)
                            for li in range(nkz):
                                off = _slab_offset(
                                    layout, kz_start + li, kx_start
                                )
                                f.read(slab, file_offset=base + off * itemsize)
                                _place_into_padded(comp_buf, li, slab, nkx)
            per_device.append(jnp.from_dlpack(vec))
    return per_device


# ── Host I/O ──────────────────────────────────────────────


def _write_serialized(
    write_fn: Callable,
    state: Array,
    tar_path: Path,
    comp_offsets: dict[int, int],
    layout: _Layout,
    itemsize: int,
) -> None:
    """Rank-ordered (token-passing) write across processes.

    Process ``r`` writes its shards only on its turn, so no two
    processes hold the archive open for writing at the same time.
    This is safe on filesystems such as NFS where concurrent
    disjoint-range writes can corrupt data: each process opens,
    writes and *closes* the file within its turn, so the next process
    sees flushed bytes (close-to-open consistency).

    All processes call the same ordered sequence of barrier tags;
    only the write itself is gated on ``process_index``, so the
    collectives stay matched.  For a single process this reduces to
    one write and a no-op barrier (identical to ``concurrent``).
    """
    me = jax.process_index()
    for r in range(jax.process_count()):
        if me == r:
            write_fn(state, tar_path, comp_offsets, layout, itemsize)
        _barrier(f"snapshot_serial_{r}")


def _write_chunks_host(
    state: Array,
    tar_path: Path,
    comp_offsets: dict[int, int],
    layout: _Layout,
    itemsize: int,
) -> None:
    """Stream each local shard to the tar slab-by-slab via host I/O.

    When cupy is available (NVIDIA GPU platforms), slabs are
    extracted on GPU and transferred one at a time via
    ``cupy.asnumpy`` (extra memory: one slab on GPU, one slab on
    host).  Otherwise (CPU runs, non-NVIDIA GPUs), the full shard
    is copied with ``np.asarray`` and slabs are extracted on the
    host (extra host memory: one shard per device).
    """
    is_walled = layout.name == "walled"
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
        with open(tar_path, "r+b") as f:
            for comp in range(_n_components()):
                comp_true = _strip_padding(vec[comp], nkz, nkx)
                base = comp_offsets[comp]
                if is_walled:
                    for i in range(layout.a_size):
                        slab = layout.extract(comp_true, i, xp)
                        for lkx in range(nkx):
                            row = xp.ascontiguousarray(slab[lkx])
                            off = (
                                i * layout.kx_global + kx_start + lkx
                            ) * layout.b_size + kz_start
                            f.seek(base + off * itemsize)
                            if cp is not None:
                                f.write(cp.asnumpy(row))
                            else:
                                f.write(row)
                else:
                    for li in range(nkz):
                        slab = layout.extract(comp_true, li, xp)
                        off = _slab_offset(layout, kz_start + li, kx_start)
                        f.seek(base + off * itemsize)
                        if cp is not None:
                            f.write(cp.asnumpy(slab))
                        else:
                            f.write(slab)


def _read_chunks_host(
    tar_path: Path,
    comp_offsets: dict[int, int],
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
    is_walled = layout.name == "walled"
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
                        with open(tar_path, "rb") as f:
                            if is_walled:
                                row_gpu = cp.empty(nkz, dtype=dtype)
                                for comp in range(_n_components()):
                                    comp_buf = vec[comp]
                                    base = comp_offsets[comp]
                                    for i in range(layout.a_size):
                                        for lkx in range(nkx):
                                            off = (
                                                i * layout.kx_global
                                                + kx_start
                                                + lkx
                                            ) * layout.b_size + kz_start
                                            f.seek(base + off * itemsize)
                                            raw = f.read(nkz * itemsize)
                                            row_gpu.set(
                                                np.frombuffer(raw, dtype=dtype)
                                            )
                                            comp_buf[i, :nkz, lkx] = row_gpu
                            else:
                                slab_bytes = nkx * layout.b_size * itemsize
                                slab_gpu = cp.empty(
                                    (nkx, layout.b_size), dtype=dtype
                                )
                                for comp in range(_n_components()):
                                    comp_buf = vec[comp]
                                    base = comp_offsets[comp]
                                    for li in range(nkz):
                                        off = _slab_offset(
                                            layout,
                                            kz_start + li,
                                            kx_start,
                                        )
                                        f.seek(base + off * itemsize)
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
            with open(tar_path, "rb") as f:
                if is_walled:
                    for comp in range(_n_components()):
                        comp_buf = vec[comp]
                        base = comp_offsets[comp]
                        for i in range(layout.a_size):
                            for lkx in range(nkx):
                                off = (
                                    i * layout.kx_global + kx_start + lkx
                                ) * layout.b_size + kz_start
                                f.seek(base + off * itemsize)
                                raw = f.read(nkz * itemsize)
                                row = np.frombuffer(raw, dtype=dtype)
                                comp_buf[i, :nkz, lkx] = row
                else:
                    slab_bytes = nkx * layout.b_size * itemsize
                    for comp in range(_n_components()):
                        comp_buf = vec[comp]
                        base = comp_offsets[comp]
                        for li in range(nkz):
                            off = _slab_offset(layout, kz_start + li, kx_start)
                            f.seek(base + off * itemsize)
                            raw = f.read(slab_bytes)
                            slab = np.frombuffer(raw, dtype=dtype).reshape(
                                nkx, layout.b_size
                            )
                            _place_into_padded(comp_buf, li, slab, nkx)
        per_device.append(jax.device_put(vec, device))
    return per_device


# ── Public API ────────────────────────────────────────────


def save_snapshot(
    state: Array,
    t: float,
    it: int,
    path: str | Path,
    *,
    stats: dict | None = None,
    isnap: int = 0,
) -> None:
    r"""Save the spectral state to a single-file snapshot.

    The field is streamed to the per-component zarr3 chunks (one per
    state component: 3, or 9 for the viscoelastic tensor state) inside
    one uncompressed tar (clean global arrays, `$k_x$`
    de-interleaved) without ever materialising a full-array transpose.
    Process 0 lays out the whole archive first; every device then
    writes its disjoint byte ranges into the reserved chunk regions.

    Parameters
    ----------
    state:
        Spectral state, shape ``(n_components, *spec_shape)``, complex
        dtype: the perturbation velocity for the base-flow systems,
        the **total** field for the force-driven dean /
        viscoelastic-dean systems (the latter 9 components --
        velocity + conformation spins).
    t:
        Current simulation time.
    it:
        Current iteration count.
    path:
        Output path for the snapshot tar file.
    stats:
        Optional ``get_stats`` dict embedded as the
        ``_dnsjax_stats.json`` member (``None`` omits the member).
    isnap:
        Snapshot-lineage index recorded in the metadata.
    """
    path = Path(path)
    layout = _layout()
    dtype_name = _zarr3_dtype_name()
    itemsize = _np_dtype(dtype_name).itemsize
    on_disk = (_n_components(), layout.a_size, layout.kx_global, layout.b_size)

    if sharding.main_device:
        path.parent.mkdir(parents=True, exist_ok=True)
        zarr_bytes = _zarr_json_bytes(on_disk, (1, *on_disk[1:]), dtype_name)
        meta_bytes = _metadata_bytes(t, it, layout, isnap)
        stats_bytes = None if stats is None else _stats_json_bytes(stats)
        _write_tar_skeleton(
            path, layout, itemsize, meta_bytes, zarr_bytes, stats_bytes
        )
    _barrier("snapshot_create")

    comp_offsets = snapshot_component_offsets(path)

    use_gds = _gds_available()
    if use_gds:
        sharding.print("Snapshot: using GDS path")
    write_fn = _write_chunks_gds if use_gds else _write_chunks_host

    if params.outs.snapshot_write_mode == "serial":
        _write_serialized(
            write_fn, state, path, comp_offsets, layout, itemsize
        )
    else:
        write_fn(state, path, comp_offsets, layout, itemsize)
    _barrier("snapshot_write")


def load_snapshot(
    path: str | Path,
) -> tuple[Array, float, int]:
    r"""Load a spectral state from a single-file snapshot.

    Each current device reads its own `$k_z$`/`$k_x$` sub-range,
    so a snapshot can be resumed at any ``(np0, np1)``
    configuration.  No full-array inverse transpose is performed.

    Parameters
    ----------
    path:
        Path to the snapshot tar file.

    Returns
    -------
    state:
        Spectral state, shape ``(n_components, *spec_shape)`` (the
        perturbation velocity, or the 9-component viscoelastic total
        field), correctly sharded.
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
    curr_true = (_n_components(), *_true_spec_shape())
    ny_mismatch = snap_native != curr_true

    if ny_mismatch:
        # Stored (v4) params use public names (res.ny is "nr" for the
        # cylindrical/annular flows); look it up via the alias.
        from .flows.registry import stored_value

        snap_ny = stored_value(meta["params"], meta["system"], "res", "ny")
        local_shape: tuple[int, ...] | None = _padded_local_shape_snap_ny(
            snap_ny
        )
        if _is_periodic():
            assembly_shape = (
                _n_components(),
                snap_ny - 1,
                sharding.nz_spec,
                sharding.nx_spec,
            )
        else:
            assembly_shape = (
                _n_components(),
                snap_ny,
                sharding.nz_spec,
                sharding.nx_spec,
            )
    else:
        local_shape = None
        assembly_shape = (_n_components(), *sharding.spec_shape)

    comp_offsets = snapshot_component_offsets(path)
    if _gds_available():
        sharding.print("Snapshot: using GDS path")
        per_device = _read_chunks_gds(
            path, comp_offsets, layout, dtype, local_shape
        )
    else:
        per_device = _read_chunks_host(
            path, comp_offsets, layout, dtype, local_shape
        )

    state = jax.make_array_from_single_device_arrays(
        assembly_shape,
        NamedSharding(sharding.mesh, sharding.spec_vector_shard),
        per_device,
    )
    return state, meta["t"], meta["it"]


def validate_snapshot_params(
    path: str | Path,
) -> None:
    r"""Check that snapshot metadata matches current parameters.

    Raises :class:`SnapshotMismatchError` on critical mismatches
    (resolution, precision, flow system, or a streamwise extent
    that the current device count cannot evenly shard).  Prints
    warnings for non-critical differences and an info line when
    the device count differs (resume is np-agnostic).  Stored (v4)
    metadata records the *public* field names; comparisons run in
    internal space (:func:`dnsjax.flows.registry.internalize_stored`)
    and messages name the public alias.

    Parameters
    ----------
    path:
        Path to the snapshot tar file.
    """
    from .flows.registry import internalize_stored, spec_for

    meta = read_metadata(Path(path))
    stored = meta.get("params", {})
    system = meta.get("system") or stored.get("phys", {}).get("system")
    spec = spec_for(system or params.phys.system)
    snap_params = internalize_stored(stored, spec.system)
    current = params.model_dump(mode="json")

    def _public(section: str, key: str) -> str:
        return spec.alias(section, key)

    # Critical: must match exactly.  The resolution labels are
    # axis-neutral: the message names the flow's public field (e.g.
    # internal ``res.nx`` is the axial ``nz`` on the annular flows).
    critical = {
        ("res", "nx"): "resolution",
        ("res", "nz"): "resolution",
        ("res", "double_precision"): "precision",
        ("phys", "system"): "flow system",
    }
    for (section, key), label in critical.items():
        snap_val = snap_params.get(section, {}).get(key)
        curr_val = current.get(section, {}).get(key)
        if snap_val is not None and snap_val != curr_val:
            name = _public(section, key)
            raise SnapshotMismatchError(
                f"{label}: snapshot {name}={snap_val}, "
                f"current {name}={curr_val}"
            )

    native = meta.get("native_shape")
    expected = [_n_components(), *_true_spec_shape()]
    if native is not None and list(native) != expected:
        if _is_periodic():
            raise SnapshotMismatchError(
                f"Shape: snapshot {native}, expected {expected}"
            )
        # Wall-bounded: allow ny mismatch (axis 1).
        non_ny_snap = [native[0], *native[2:]]
        non_ny_exp = [expected[0], *expected[2:]]
        if non_ny_snap != non_ny_exp:
            raise SnapshotMismatchError(
                f"Shape (non-ny axes): snapshot "
                f"{non_ny_snap}, expected {non_ny_exp}"
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
        ("geo", "lx"): "domain extent",
        ("geo", "lz"): "domain extent",
        ("geo", "tilt_degree"): "tilt angle",
        ("res", "fd_order"): "FD accuracy order",
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
                f"Warning: {label} {_public(section, key)} changed: "
                f"{snap_val} -> {curr_val}"
            )

    snap_np = meta.get("n_devices")
    if snap_np is not None and snap_np != sharding.n_devices:
        sharding.print(
            f"Info: device count {snap_np} -> {sharding.n_devices} "
            f"(np0={sharding.np0}, np1={sharding.np1}; "
            f"np-agnostic resume)"
        )
