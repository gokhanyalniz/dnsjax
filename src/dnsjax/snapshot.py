r"""Snapshot save/load for simulation checkpointing.

A snapshot is a **single uncompressed tar archive** wrapping a
zarr3 store plus a JSON metadata member::

    snapshot.tar                 (uncompressed tar; format_version 6)
      _dnsjax_meta.json          plain JSON metadata
      _dnsjax_stats.json         plain JSON stats (optional)
      state/zarr.json            zarr3 array metadata
      state/c/0/0/0/0            component 0 chunk: raw LE complex
      state/c/1/0/0/0            component 1 chunk
      state/c/2/0/0/0            component 2 chunk

The spectral state is stored as **per-component zarr3 chunks** (one
chunk per state component), each **byte-identical to the solver's
in-memory spectral layout** at true (unpadded) mode counts -- no
transpose exists between memory and disk.  Zero-valued padding modes
added for 2D mesh divisibility are stripped on save and re-introduced
on load.  Because each chunk holds the full mode range, a snapshot
can be resumed at **any** ``(np0, np1)`` configuration (np-agnostic).

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
Each component is stored exactly as the solver holds it (the
:mod:`dnsjax.sharding` spectral layout) at the true mode counts:

===============  =======================  ================
family           component shape          axes
===============  =======================  ================
wall-bounded     ``(ny, nz-1, nx//2)``    ``(y,  kz, kx)``
triply-periodic  ``(ny-1, nz-1, nx//2)``  ``(ky, kz, kx)``
===============  =======================  ================

`$k_x$` (the real-FFT axis, Nyquist omitted) is innermost; the
wall-normal / `$k_y$` axis is outermost.

I/O granularity and memory
--------------------------
No transpose is ever performed, and every transfer is one
C-contiguous **slab**: a device's whole range of the chunk's slowest
axis, for one component.  Which pieces of the file a device holds is
decided by **which axis its shard is cut along**.  The chunk is the
C-ordered global array ``(A, kz, kx)`` (``A`` = the wall-normal /
`$k_y$` axis), so its slowest-varying axis is ``A`` and its fastest
is `$k_x$`.

The solver layout shards the two *fastest* axes (`$k_z$` by ``np0``,
`$k_x$` by ``np1``), which is the worst possible cut for this file:
with ``np1 > 1`` every ``kx_true``-element row is divided between the
``np1`` devices, so a device's pieces are its own ``local_kx`` block
-- 512 B at ``256 x 193 x 256, np1 = 4``, and ``3 x 193 x 255`` of
them per device.  **Measured on BeeGFS: 90 s to save a 288 MiB
state** (151.7 µs per transfer), against 1.7 s for the same state at
``np1 = 1``.

So the state is resharded once per save onto an **I/O layout**
(:func:`_io_spec`) that cuts the *slowest* axis instead: each device
takes a contiguous slab of ``A`` and holds `$k_z$`/`$k_x$` whole, and
its bytes become one contiguous range per component --
``n_components * ndev`` transfers of tens of MiB instead of ~1e5 of
512 B.  Reads take the mirror path (read the slabs, reshard back).
The cost is one exchange per save/load -- **one jitted program**,
routed one mesh axis at a time (:func:`_to_io_layout` says what each
of those buys, in measured milliseconds) -- plus a transient second
copy of the state, distributed one local shard per device.  A
single-device mesh already *is* the I/O layout and pays neither.

That count is ``n_components * ndev`` on **every** mesh, not only an
unpadded one, because the reshard also trims the solver layout's
divisibility padding: the I/O buffer carries the *true* mode counts,
so it **is** the device's byte range rather than a padded superset of
it (:func:`_to_io_layout_core`, :func:`_io_local_shape`).  Without
that trim one padded mode in a *middle* axis breaks the contiguity of
the whole component -- and `$n_z - 1$` being odd forces exactly that
on any even ``np0``: one transfer per ``A`` row, 49 of 0.5 MiB rather
than one of 25 MiB at ``256 x 193 x 256`` on four devices.  A padded
`$k_x$` falls further, to one transfer per ``(a, k_z)`` row: measured
at ``ny = 49, nz = 16, nx = 6`` on four devices, 2205 writes of 48 B
against 12 of 8.8 kiB.  Neither needs an exotic grid -- ``np1`` just
happens to divide ``nx // 2`` for the usual power-of-two ``nx``.

This is **not** the pre-v5 layout change.  The store is still one
fixed np-independent global array in the native order (that is what
makes resume np-agnostic), and the on-disk bytes are unchanged: only
*which device writes which byte range* moves.  Storing a sharded axis
outermost instead is what the pre-v5 layouts did, and that cost a
transpose of every slab on every save and load.

**Extra memory per device by I/O engine** (beyond the resident
state):

=======================  ===============  =====================
engine                   GPU extra        host extra
=======================  ===============  =====================
GDS (write and read)     one local shard  --
Host + cupy (w and r)    one local shard  one slab (a component
                                          of a shard)
Host, no cupy (w and r)  one local shard  one shard
=======================  ===============  =====================

The "one local shard" column is the reshard's transient second copy,
which every engine pays; it is absent on a single-device mesh.  The
padding trim inside the reshard is free of it in the usual case --
it replaces the leading-axis pad's buffer rather than adding one --
and costs a second such transient only when the device count divides
``A`` exactly *and* a mode axis is padded.

I/O engine
----------
Data is written and read with **raw offset I/O** on both backends
(TensorStore writes at chunk granularity, so per-device sub-range
writes to a shared chunk would race / read-modify-write).  Process
0 lays out the whole archive once -- the tar headers plus
sparse-reserved (zero-filled) component data regions -- and every
device then writes its disjoint byte ranges directly into the one
file at ``component_offset + within_component_offset``.  That
layout is built under a ``.partial`` name and renamed once every
write has landed (:data:`_PARTIAL_SUFFIX`), because until then the
file is a valid archive full of zeros.  The
component base offsets are the tar members' ``offset_data`` (read
back via ``tarfile``, so 512-aligned -- a tar block -- but not
generally 4096-aligned, which is what cuFile's direct path prefers.
That costs a bounce buffer for the head and tail of a transfer, so
it was a real penalty when a transfer was 512 B and is a rounding
error now that the I/O layout makes it tens of MiB).
TensorStore is used only to generate the ``zarr.json`` bytes (in a
throwaway temporary directory); compression is never used (it would
break random-access streaming).

- **GDS** (NVIDIA GPUDirect Storage): when ``kvikio`` and ``cupy``
  are available, slabs move directly between GPU memory and disk.
- **Host + cupy** (NVIDIA GPU, no GDS): slabs are transferred one
  at a time with ``cupy.asnumpy`` (write) or ``cupy.ndarray.set``
  (read).
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
snapshot-lineage index this file was written with), the global (true,
unpadded) ``native_shape`` -- which **is** the on-disk array shape,
the per-component chunk shape being ``native_shape[1:]`` --
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

The on-disk format is ``format_version: 6``: the solver's native
spectral layout, physical components in every family
(cylindrical/annular `$u_r$`, `$u_\theta$` and the physical
conformation tensor; the solver's decoupled `$u_\pm$` / spin working
basis is converted by the caller, so these functions stay
basis-agnostic), and the public-named parameter dump.  Older
snapshots are rejected at read
(:func:`dnsjax.snapshot_meta.read_snapshot_meta`), never translated.
"""

import json
import math
from collections.abc import Callable
from functools import cache, partial
from pathlib import Path

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .flows.registry import periodic_systems
from .parameters import derived_params, params
from .sharding import sharding
from .snapshot_meta import (
    git_hash,
    read_snapshot_meta,
    snapshot_component_offsets,
)


class SnapshotMismatchError(Exception):
    """Snapshot parameters conflict with current config."""


#: Suffix of the file a snapshot is built in before it is renamed
#: into place.  The archive is laid out full-length with
#: **zero-filled** component regions and only then filled in, so an
#: interrupted save under the final name would leave a structurally
#: valid tar that loads without complaint and is blank wherever the
#: writes did not reach -- the one corruption a run cannot detect,
#: because zeros are a legal state.  Building under this suffix and
#: renaming (``os.replace``, atomic within a filesystem) means the
#: final name only ever appears on a complete archive; a killed job
#: leaves a ``.partial`` beside the last good snapshot instead of
#: replacing it.  Costs one metadata operation per save.
_PARTIAL_SUFFIX = ".partial"


# ── Runtime detection ─────────────────────────────────────


#: The GDS kernel driver's procfs entry.  It ships with the GDS
#: package and is required for any real GPUDirect transfer.
_NVFS_STATS = Path("/proc/driver/nvidia-fs/stats")


@cache
def _gds_available() -> bool:
    """True when kvikIO + GDS can transfer GPU buffers.

    Four things must hold, and most have bitten this path before:

    - **kvikIO imports.**  ``kvikio.defaults`` is a *submodule*, not
      an attribute of the package, so ``import kvikio`` alone does not
      make ``kvikio.defaults`` resolve on every version -- which is
      how this check spent its life raising ``AttributeError`` and
      taking the "unusable" branch on a cluster that had kvikIO
      installed.  Bind the submodule itself
      (``import kvikio.defaults as ...``, a ``sys.modules`` lookup)
      rather than reaching for it through the package.
    - **The nvidia-fs driver is loaded.**  Without it kvikIO still
      imports and still accepts every call; it just services them
      through its *compat* shim (POSIX I/O on a thread pool), which is
      not GDS and which measured 1.5-6x *slower* than this module's
      own host path below 1 MiB spans.  ``AUTO`` -- the default compat
      mode -- is exactly the case that would otherwise be mistaken for
      "GDS is on".
    - **Compat mode is not explicitly ``ON``.**
      ``defaults.get("compat_mode")`` returns a ``CompatMode`` enum
      (``OFF`` / ``ON`` / ``AUTO``), not a bool; a bare truth test
      would read ``AUTO`` as "compat" and demote every run.  Compare
      against the enum member.
    - **cupy imports.**  Both GDS engines wrap their device buffers
      with it, so without cupy this path cannot run at all -- and
      answering "GDS" would turn the first snapshot of the run into
      an ``ImportError`` instead of a fallback.

    Cached: the answer cannot change within a process, and the
    rejection line would otherwise print on every single snapshot.
    """
    try:
        import cupy  # noqa: F401  the GDS engines' buffer wrapper
        import kvikio.defaults as kvikio_defaults
    except ImportError:
        return False
    if not _NVFS_STATS.exists():
        sharding.print(
            "Snapshot: kvikIO is installed but nvidia-fs is not loaded, "
            "so it would run its compat shim rather than GDS; using the "
            "host path."
        )
        return False
    try:
        mode = kvikio_defaults.get("compat_mode")
        available = mode is not type(mode).ON
    except (AttributeError, KeyError, TypeError) as exc:  # API moved
        sharding.print(
            f"Snapshot: kvikIO present but its compat-mode setting is "
            f"unreadable ({exc}); using the host path."
        )
        return False
    if not available:
        sharding.print(
            f"Snapshot: kvikIO compat mode is {mode!r}; using the host "
            "path (set KVIKIO_COMPAT_MODE=off or =auto for GDS)"
        )
    return available


def _require_dense(vec) -> None:
    """Reject a non-C-contiguous device view before it is written.

    The slab each device transfers is a leading-axis prefix slice,
    contiguous only if the array it slices is dense row-major.  kvikIO
    reads
    ``__cuda_array_interface__`` and cupy's dlpack import honours
    whatever strides it is handed, so a non-default XLA layout would
    transfer the *wrong bytes* with no error at all -- silent
    corruption of a snapshot, which is the one artifact a run cannot
    reconstruct.  Cheap to check, and it converts that class into a
    crash.
    """
    if not vec.flags.c_contiguous:
        raise ValueError(
            "snapshot I/O requires a dense row-major device buffer, "
            f"got strides {vec.strides} for shape {vec.shape}; the "
            "slab offsets assume C-contiguous storage."
        )


def _cuda_ordinal(device) -> int:
    """CUDA ordinal backing a JAX device.

    ``jax.local_devices()`` order is not guaranteed to be the CUDA
    ordinal order, and the read engines allocate cupy buffers whose
    byte ranges were computed from the *JAX* device -- so the two must
    be tied together explicitly.  ``local_hardware_id`` is that tie;
    fall back to ``id`` on backends that do not expose it.
    """
    ordinal = getattr(device, "local_hardware_id", None)
    return int(ordinal if ordinal is not None else device.id)


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


# ── The I/O layout and its slabs ──────────────────────────


def _n_devices() -> int:
    """Devices in the mesh (``np0 * np1``)."""
    return sharding.np0 * sharding.np1


def _io_spec():
    r"""Partition spec of the snapshot **I/O layout**.

    The stored chunk is the C-ordered global array
    ``(A, kz_true, kx_true)``, so its *slowest*-varying axis is the
    leading one (the wall-normal `$y$` / `$k_y$` axis, unsharded in
    the solver layout).  Splitting **that** axis across every device
    -- and leaving `$k_z$` and `$k_x$` whole, at the true mode counts
    (:func:`_to_io_layout_core` trims the padding) -- makes each
    device's bytes one contiguous file range per component.

    The solver layout shards the two *fastest* axes instead (`$k_z$`
    by ``np0``, `$k_x$` by ``np1``), which is why a device's bytes
    are fragmented there: with ``np1 > 1`` every ``kx_true``-element
    file row is divided between the ``np1`` devices, leaving pieces
    of ``local_kx`` elements. :func:`_to_io_layout` therefore reshards
    once per save (and :func:`load_snapshot` reshards back after the
    read); the measured reason is in the module docstring.
    """
    axes = tuple(a for a in (sharding.a0, sharding.a1) if a is not None)
    # ``(a0, a1)`` splits the axis a0-major, so the slab index is the
    # row-major flat mesh index ``i0 * np1 + i1`` that
    # :func:`_shard_device_index` reports.
    return P(None, axes or None, None, None)


def _a_local(a_true: int, ndev: int) -> int:
    """Padded per-device extent of the leading axis (ceil division).

    ``a_true`` is rarely divisible by the device count (wall-normal
    grids are odd), and a :class:`NamedSharding` cannot split an axis
    unevenly -- so the array is zero-padded to ``_a_local * ndev``
    before resharding and the trailing rows are simply never written
    or read (:func:`_a_ranges` clamps to ``a_true``).
    """
    return -(-a_true // ndev)


def _a_ranges(flat_idx: int, a_true: int, ndev: int) -> tuple[int, int]:
    """``(a_start, n_rows)`` of the leading-axis slab of a device.

    The ranges tile ``[0, a_true)`` exactly.  ``n_rows`` is 0 for a
    device whose slab is entirely padding, which happens whenever
    ``a_true <= (ndev - 1) * _a_local`` -- e.g. 5 rows over 4 devices.
    """
    a_local = _a_local(a_true, ndev)
    a_start = flat_idx * a_local
    return a_start, max(0, min(a_start + a_local, a_true) - a_start)


def _io_local_shape(a_true: int) -> tuple[int, ...]:
    r"""Per-device buffer shape in the I/O layout.

    The leading axis is this device's (padded) slab; `$k_z$` and
    `$k_x$` are whole and at the **true** mode counts, the solver
    layout's divisibility padding having been trimmed off inside the
    reshard (:func:`_to_io_layout_core`).  So the buffer *is* the
    device's byte range in the file: one contiguous slab per
    component, whatever the mesh pads.

    The mode counts are the *current* run's, which on a read are the
    stored ones -- ``validate_snapshot_params`` rejects a resolution
    mismatch before the reader gets here, and the offsets it reads at
    come from the stored ``native_shape``.
    """
    return (
        _n_components(),
        _a_local(a_true, _n_devices()),
        _kz_true(),
        _kx_true(),
    )


def _a_offset(a_start: int, kz_true: int, kx_true: int) -> int:
    r"""First element index of a leading-axis slab in a component chunk.

    The chunk is the C-ordered ``(A, kz_true, kx_true)`` global array
    and a device owns whole `$(k_z, k_x)$` planes at the true mode
    counts, so its slab is the single contiguous element range
    ``[off, off + na * kz_true * kx_true)`` starting here -- the same
    range in the file and in the local buffer, in the same order, with
    no transpose and no fragmentation.  The one place the arithmetic
    is written down; ``native_shape``, the shape it is derived from, is
    cross-checked against the chunks in
    :func:`dnsjax.snapshot_meta.snapshot_component_offsets`.
    """
    return a_start * kz_true * kx_true


def _trim_axis(state: Array, axis: int, n_true: int) -> Array:
    """Slice *axis* down to its true mode count (no-op when unpadded)."""
    if state.shape[axis] == n_true:
        return state
    idx = [slice(None)] * state.ndim
    idx[axis] = slice(0, n_true)
    return state[tuple(idx)]


def _pad_axis(state: Array, axis: int, n_padded: int) -> Array:
    """Zero-pad *axis* up to *n_padded* (no-op when already there)."""
    if state.shape[axis] == n_padded:
        return state
    pad = [(0, 0)] * state.ndim
    pad[axis] = (0, n_padded - state.shape[axis])
    return jnp.pad(state, pad)


def _via_mid(
    state: Array, target, mid_fn: Callable[[Array], Array] | None = None
) -> Array:
    r"""Reshard *state* to *target*, moving one mesh axis per step.

    **Trace-time only** -- both callers wrap it in :func:`jax.jit`,
    for the reason in :func:`_to_io_layout`.

    Going straight between the solver layout ``P(-, -, a0, a1)`` and
    the I/O layout ``P(-, (a0, a1), -, -)`` relocates **both** mesh
    axes onto a different array axis at once, and XLA's SPMD
    partitioner cannot express that as an exchange.  It says so and
    takes its "last resort": *replicate* the whole array on every
    device, then slice.  That is ``ndev`` times the traffic and
    ``ndev`` times the peak memory of the exchange it replaces --
    it prints an "Involuntary full rematerialization" warning at
    ``np0 > 1, np1 > 1``.

    Routing through ``P(-, a0, -, a1)`` makes each leg a single-mesh-
    axis move, which SPMD does express as an all-to-all.  On a 1D mesh
    one of the two legs is the identity (the intermediate spec equals
    the source or the target), so this costs nothing there.

    *mid_fn* is applied **on** the intermediate layout, the one point
    in either direction where `$k_z$` is local; the callers use it to
    trim and restore the mode padding (:func:`_to_io_layout_core`).
    """
    mid = P(None, sharding.a0, None, sharding.a1)
    state = jax.sharding.reshard(state, NamedSharding(sharding.mesh, mid))
    if mid_fn is not None:
        state = mid_fn(state)
    return jax.sharding.reshard(state, NamedSharding(sharding.mesh, target))


@jax.jit
def _to_io_layout_core(state: Array) -> Array:
    r"""Pad the leading axis, reshard, and trim to the true modes.

    **One** jitted program, and *where* each trim sits is what makes it
    free.  The solver layout pads `$k_z$` to a multiple of ``np0`` and
    `$k_x$` to a multiple of ``np1``, neither of which the file has;
    trimming them here rather than in the writers is what leaves every
    device with one contiguous slab per component on **every** mesh
    (see the module docstring).  A sharded axis cannot be sliced, so
    each trim goes where its axis is local -- `$k_z$` on the mid
    layout, `$k_x$` on the I/O layout -- which is also immediately
    before the next collective, so the trimmed array *replaces* the
    untrimmed one in the liveness window instead of adding to it.

    A trim is a no-op exactly when its mesh axis is 1
    (``round_up_padded(n, 1) == n``), i.e. exactly when the leg it
    follows is the identity: a grid that pads nothing compiles to the
    reshard alone.
    """
    ndev = _n_devices()
    state = _pad_axis(state, 1, _a_local(state.shape[1], ndev) * ndev)
    state = _via_mid(state, _io_spec(), lambda x: _trim_axis(x, 2, _kz_true()))
    return _trim_axis(state, 3, _kx_true())


@partial(jax.jit, static_argnums=(1,))
def _from_io_layout_core(state: Array, a_true: int) -> Array:
    """Restore the mode padding, reshard back, drop the slab padding.

    The exact mirror of :func:`_to_io_layout_core`, and for the same
    reasons: each mode axis is re-padded where it is local (`$k_x$` on
    the I/O layout, `$k_z$` on the mid one), and the leading-axis slice
    has to follow the reshard because that axis is sharded until then
    -- slicing a sharded axis is the ``ShardingTypeError`` trap.
    """
    state = _pad_axis(state, 3, sharding.nx_spec)
    state = _via_mid(
        state,
        sharding.spec_vector_shard,
        lambda x: _pad_axis(x, 2, sharding.nz_spec),
    )
    return state if state.shape[1] == a_true else state[:, :a_true]


def _to_io_layout(state: Array) -> Array:
    r"""Reshard *state* onto the I/O layout, padding the leading axis.

    One exchange per save, against ~1e5 serialized 512 B writes per
    device without it.  Costs a transient second copy of the state,
    distributed -- one local shard per device.  A single-device mesh
    already *is* the I/O layout down to the padding (``np0 = np1 = 1``
    pads no axis, so the trims and the slab pad in
    :func:`_to_io_layout_core` are all no-ops), so it reshards nothing
    and pays neither.

    **The exchange must be one jitted program.**  Expressed eagerly --
    a :func:`jax.device_put` per leg -- the runtime redistributes
    piece by piece instead of emitting a collective, and the same
    216 MiB took **230 ms** on an H100 node whose fabric moves a
    72 MiB shard device-to-device in 0.32 ms (223 GB/s).  Inside
    ``jit`` the identical two moves take **0.68 ms**: a 338x
    difference, and the whole reason a multi-device snapshot is now
    write-bound rather than reshard-bound.
    """
    if _n_devices() == 1:
        return state
    return _to_io_layout_core(state)


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


# ── In-memory per-device assembly ─────────────────────────


def assemble_local_shards(
    fill_local: Callable[[np.ndarray, int, int, int, int], None],
    *,
    dtype: np.dtype | None = None,
) -> Array:
    r"""Assemble a sharded spectral state from per-device-generated shards.

    Builds a state directly in the **solver** layout, for the
    in-process IC generators.  (:func:`load_snapshot` assembles on the
    I/O layout instead and reshards -- its shards come from the file,
    whose axis order is fixed, whereas a generator can simply fill
    whichever block it is asked for.)  Each local device's padded
    shard is allocated zero-filled (shape
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
    comp_shape: tuple[int, ...],
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
    comp_nbytes = math.prod(comp_shape) * itemsize
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


def _metadata_bytes(t: float, it: int, isnap: int = 0) -> bytes:
    """Serialise the ``_dnsjax_meta.json`` member content.

    ``git_hash`` records the code revision that wrote the snapshot
    (provenance only -- never read back on load).  Additive keys like
    it need no ``format_version`` bump: readers use targeted lookups
    and ignore unknown keys.  Format 6 stores physical components
    (cylindrical/annular `$u_r$`, `$u_\theta$`, the physical
    conformation tensor) in the solver's native spectral layout, so
    ``native_shape[1:]`` **is** the on-disk per-component chunk
    shape; ``params`` is the flow-relevant, **public-named**,
    resolved dump plus the relevant extension sections (e.g.
    ``force``, ``probes``;
    :func:`dnsjax.param_surface.recorded_params_dump`), mapped back
    by readers via :func:`dnsjax.flows.registry.internalize_stored` /
    ``stored_value``.  Pre-6 snapshots are rejected at
    :func:`dnsjax.snapshot_meta.read_snapshot_meta`.
    """
    from .param_surface import recorded_params_dump

    meta = {
        "format_version": 6,
        "git_hash": git_hash(),
        "t": t,
        "it": it,
        "isnap": isnap,
        "geometry": ("triply_periodic" if _is_periodic() else "wall_bounded"),
        "system": params.phys.system,
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
    comp_shape: tuple[int, ...],
    itemsize: int,
) -> None:
    """Stream each local shard to the tar slab-by-slab via kvikIO.

    Each component's chunk lives at ``comp_offsets[comp]`` inside the
    single archive; all writes are at that base plus the in-component
    byte offset, directly from GPU memory (the slab is a contiguous
    view of the shard -- no staging, no copies).  *state* is in the I/O
    layout at the true mode counts (:func:`_to_io_layout`), so this is
    exactly one write per component per device.
    """
    import cupy as cp
    import kvikio

    a_true, kz_true, kx_true = comp_shape
    for shard in state.addressable_shards:
        a_start, na = _a_ranges(
            _shard_device_index(shard), a_true, _n_devices()
        )
        if na == 0:
            continue
        cp_vec = cp.from_dlpack(shard.data)
        _require_dense(cp_vec)
        off = _a_offset(a_start, kz_true, kx_true) * itemsize
        with cp_vec.device, kvikio.CuFile(str(tar_path), "r+") as f:
            for comp in range(_n_components()):
                slab = cp_vec[comp][:na]
                n = f.write(slab, file_offset=comp_offsets[comp] + off)
                # A short write leaves the sparse-reserved zeros of the
                # skeleton in place -- a snapshot that loads fine and is
                # blank in the middle.  The host writer guards this too,
                # and the I/O layout made each slab tens of MiB, so
                # there is something substantial to lose.
                if n != slab.nbytes:
                    raise OSError(
                        f"short GDS write to {tar_path}: {n} of "
                        f"{slab.nbytes} bytes (component {comp}); "
                        "the archive is incomplete."
                    )


def _read_chunks_gds(
    tar_path: Path,
    comp_offsets: dict[int, int],
    comp_shape: tuple[int, ...],
    dtype: np.dtype,
) -> list[Array]:
    r"""Read each device's leading-axis slab via kvikIO into an
    I/O-layout shard (np-agnostic), directly into GPU memory (the slab
    is a contiguous view of the shard -- no staging buffers), one read
    per component.  The caller reshards the assembled array back to
    the solver layout."""
    import cupy as cp
    import kvikio

    itemsize = dtype.itemsize
    a_true, kz_true, kx_true = comp_shape
    local_shape = _io_local_shape(a_true)
    per_device: list[Array] = []
    for device in jax.local_devices():
        a_start, na = _a_ranges(
            _mesh_device_index(device), a_true, _n_devices()
        )
        # Bind cupy to *this* device's hardware ordinal, not to its
        # position in ``local_devices()``.  The byte ranges below come
        # from ``_mesh_device_index(device)``, so allocating on a
        # different GPU would hand each device another one's slab --
        # and ``make_array_from_single_device_arrays`` would accept it
        # (one array per device either way), silently.
        with cp.cuda.Device(_cuda_ordinal(device)):
            vec = cp.zeros(local_shape, dtype=dtype)
            if na > 0:
                off = _a_offset(a_start, kz_true, kx_true) * itemsize
                with kvikio.CuFile(str(tar_path), "r") as f:
                    for comp in range(_n_components()):
                        slab = vec[comp][:na]
                        n = f.read(slab, file_offset=comp_offsets[comp] + off)
                        # Short read = truncated archive; the buffer was
                        # zero-allocated, so without this the tail of
                        # the state silently stays blank (see the host
                        # reader).
                        if n != slab.nbytes:
                            raise OSError(
                                f"short GDS read from {tar_path}: "
                                f"{n} of {slab.nbytes} bytes "
                                f"(component {comp}); the archive "
                                "is truncated."
                            )
            per_device.append(jnp.from_dlpack(vec))
    return per_device


# ── Host I/O ──────────────────────────────────────────────


def _write_serialized(
    write_fn: Callable,
    state: Array,
    tar_path: Path,
    comp_offsets: dict[int, int],
    comp_shape: tuple[int, ...],
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
            write_fn(state, tar_path, comp_offsets, comp_shape, itemsize)
        _barrier(f"snapshot_serial_{r}")


def _write_chunks_host(
    state: Array,
    tar_path: Path,
    comp_offsets: dict[int, int],
    comp_shape: tuple[int, ...],
    itemsize: int,
) -> None:
    """Stream each local shard to the tar slab-by-slab via host I/O.

    *state* is in the I/O layout at the true mode counts
    (:func:`_to_io_layout`), so a device's slab is one write per
    component.  When cupy is available (NVIDIA GPU platforms) each slab
    is staged through ``cupy.asnumpy``; otherwise (CPU runs, non-NVIDIA
    GPUs) the full shard is copied once with ``np.asarray`` and the
    slabs are written directly from it.  Either way the extra host
    memory is one slab, i.e. a whole component
    (``shard / n_components``) rather than a plane -- the trade that
    turns ~1e5 small writes into a handful of big ones.
    """
    a_true, kz_true, kx_true = comp_shape
    try:
        import cupy as cp
    except ImportError:
        cp = None
    for shard in state.addressable_shards:
        a_start, na = _a_ranges(
            _shard_device_index(shard), a_true, _n_devices()
        )
        if na == 0:
            continue
        if cp is not None:
            try:
                vec = cp.from_dlpack(shard.data)
            except Exception:
                cp = None
        if cp is None:
            vec = np.asarray(shard.data)
        else:
            _require_dense(vec)
        off = _a_offset(a_start, kz_true, kx_true) * itemsize
        with open(tar_path, "r+b") as f:
            for comp in range(_n_components()):
                slab = vec[comp][:na]
                f.seek(comp_offsets[comp] + off)
                buf = cp.asnumpy(slab) if cp is not None else slab
                n = f.write(buf)
                if n != buf.nbytes:
                    raise OSError(
                        f"short write to {tar_path}: {n} of "
                        f"{buf.nbytes} bytes (component {comp}); "
                        "the archive is incomplete."
                    )


def _read_chunks_host(
    tar_path: Path,
    comp_offsets: dict[int, int],
    comp_shape: tuple[int, ...],
    dtype: np.dtype,
) -> list[Array]:
    r"""Read each device's leading-axis slab via host I/O into an
    I/O-layout shard (np-agnostic).

    When cupy is available (NVIDIA GPU platforms), the output buffer
    is allocated on GPU and each component's slab is copied onto its
    contiguous view via ``cupy.ndarray.set``.  Otherwise (CPU runs,
    non-NVIDIA GPUs), the output is assembled on the host with
    ``readinto`` (no temporaries) and transferred at the end via
    ``jax.device_put``.  Extra host memory: one slab, i.e. a component
    (``shard / n_components``).  The caller reshards the assembled
    array back to the solver layout.
    """
    itemsize = dtype.itemsize
    a_true, kz_true, kx_true = comp_shape
    local_shape = _io_local_shape(a_true)
    try:
        import cupy as cp
    except ImportError:
        cp = None
    per_device: list[Array] = []
    for device in jax.local_devices():
        a_start, na = _a_ranges(
            _mesh_device_index(device), a_true, _n_devices()
        )
        if cp is not None:
            try:
                # This device's hardware ordinal -- see the GDS reader.
                with cp.cuda.Device(_cuda_ordinal(device)):
                    vec = cp.zeros(local_shape, dtype=dtype)
                    if na > 0:
                        off = _a_offset(a_start, kz_true, kx_true) * itemsize
                        with open(tar_path, "rb") as f:
                            for comp in range(_n_components()):
                                dst = vec[comp][:na]
                                f.seek(comp_offsets[comp] + off)
                                raw = f.read(dst.nbytes)
                                # A short read raises here, in
                                # ``reshape`` -- see the plain branch.
                                dst.set(
                                    np.frombuffer(raw, dtype=dtype).reshape(
                                        dst.shape
                                    )
                                )
                    per_device.append(jnp.from_dlpack(vec))
                continue
            except Exception:
                cp = None
        vec = np.zeros(local_shape, dtype=dtype)
        if na > 0:
            off = _a_offset(a_start, kz_true, kx_true) * itemsize
            with open(tar_path, "rb") as f:
                for comp in range(_n_components()):
                    dst = vec[comp][:na]
                    f.seek(comp_offsets[comp] + off)
                    n = f.readinto(dst)
                    if n != dst.nbytes:
                        # Short read = truncated archive.  Without this
                        # the tail of the state is silently left as the
                        # zeros it was allocated with (the cupy branch
                        # raises on its own, via frombuffer's reshape).
                        raise OSError(
                            f"short read from {tar_path}: {n} of "
                            f"{dst.nbytes} bytes (component "
                            f"{comp}); the archive is truncated."
                        )
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
    one uncompressed tar, in the solver's native spectral layout at
    true (unpadded) mode counts -- no transpose anywhere.  Process 0
    lays out the whole archive first; every device then writes its
    disjoint byte ranges into the reserved chunk regions.  The write
    goes to a ``.partial`` sibling that is renamed into place once
    complete (:data:`_PARTIAL_SUFFIX`), so *path* never names a
    half-written snapshot and an interrupted save leaves the previous
    one intact.

    On a multi-device mesh the state is first resharded onto the I/O
    layout (:func:`_to_io_layout`) so those ranges are contiguous;
    the caller's array is left untouched (no buffer donation -- the
    solver keeps stepping the state it just snapshotted).

    Parameters
    ----------
    state:
        Spectral state, shape ``(n_components, *spec_shape)``, complex
        dtype: the perturbation velocity for the base-flow systems,
        the **total** field for the force-driven dean and the two
        viscoelastic systems (the latter 9 components -- velocity +
        physical conformation components).
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
    # Everything is written to a sibling and renamed at the end, so
    # the final name never names a half-written archive (see
    # :data:`_PARTIAL_SUFFIX`).
    partial = path.with_name(path.name + _PARTIAL_SUFFIX)
    comp_shape = _true_spec_shape()
    dtype_name = _zarr3_dtype_name()
    itemsize = _np_dtype(dtype_name).itemsize
    on_disk = (_n_components(), *comp_shape)

    if sharding.main_device:
        path.parent.mkdir(parents=True, exist_ok=True)
        zarr_bytes = _zarr_json_bytes(on_disk, (1, *comp_shape), dtype_name)
        meta_bytes = _metadata_bytes(t, it, isnap)
        stats_bytes = None if stats is None else _stats_json_bytes(stats)
        _write_tar_skeleton(
            partial, comp_shape, itemsize, meta_bytes, zarr_bytes, stats_bytes
        )
    _barrier("snapshot_create")

    comp_offsets = snapshot_component_offsets(partial)

    use_gds = _gds_available()
    if use_gds:
        sharding.print("Snapshot: using GDS path")
    write_fn = _write_chunks_gds if use_gds else _write_chunks_host

    # One reshard, then every device writes contiguous byte ranges.
    # Collective, so it must happen outside the serial write mode's
    # rank-ordered section -- and before it, since that section only
    # reorders the writes.
    state = _to_io_layout(state)

    if params.outs.snapshot_write_mode == "serial":
        _write_serialized(
            write_fn, state, partial, comp_offsets, comp_shape, itemsize
        )
    else:
        write_fn(state, partial, comp_offsets, comp_shape, itemsize)
    # Every write has landed and every file handle is closed (the
    # engines all use ``with``), so the archive is complete and can
    # take its real name.
    _barrier("snapshot_write")
    if sharding.main_device:
        partial.replace(path)
    # Only after this does the snapshot exist for a reader, on every
    # process -- so a rank that returns early cannot report a
    # checkpoint the run has not actually committed.
    _barrier("snapshot_commit")


def load_snapshot(
    path: str | Path,
) -> tuple[Array, float, int]:
    r"""Load a spectral state from a single-file snapshot.

    Each current device reads a contiguous slab of the file's
    *slowest* axis (the I/O layout, :func:`_io_spec`) and the
    assembled array is then resharded back to the solver layout, so a
    snapshot can be resumed at any ``(np0, np1)`` configuration.  No
    full-array inverse transpose is performed: the reshard is an
    exchange of whole planes, not a reordering of bytes.

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
    # The stored chunk shape (the snapshot's native shape; its ny may
    # differ from the current run's -- kz/kx equality is enforced by
    # ``validate_snapshot_params``).
    comp_shape = tuple(meta["native_shape"][1:])
    dtype = _np_dtype(meta["dtype"])

    # Detect ny mismatch (wall-bounded only).
    snap_native = tuple(meta["native_shape"])
    curr_true = (_n_components(), *_true_spec_shape())
    ny_mismatch = snap_native != curr_true

    if ny_mismatch:
        # Stored params use public names (res.ny is "nr" for the
        # cylindrical/annular flows); look it up via the alias.
        from .flows.registry import stored_value

        snap_ny = stored_value(meta["params"], meta["system"], "res", "ny")
        assembly_shape = (
            _n_components(),
            snap_ny - 1 if _is_periodic() else snap_ny,
            sharding.nz_spec,
            sharding.nx_spec,
        )
    else:
        assembly_shape = (_n_components(), *sharding.spec_shape)

    comp_offsets = snapshot_component_offsets(path)
    if _gds_available():
        sharding.print("Snapshot: using GDS path")
        per_device = _read_chunks_gds(path, comp_offsets, comp_shape, dtype)
    else:
        per_device = _read_chunks_host(path, comp_offsets, comp_shape, dtype)

    # The read lands in the I/O layout: contiguous leading-axis slabs
    # zero-padded to a divisible length, carrying the true mode counts
    # rather than the solver's padded ones.  ``_from_io_layout_core``
    # undoes all three, each where its axis is local.
    a_true = assembly_shape[1]
    io_shape = (
        assembly_shape[0],
        _a_local(a_true, _n_devices()) * _n_devices(),
        _kz_true(),
        _kx_true(),
    )
    state = jax.make_array_from_single_device_arrays(
        io_shape,
        NamedSharding(sharding.mesh, _io_spec()),
        per_device,
    )
    if _n_devices() > 1 or io_shape != assembly_shape:
        state = _from_io_layout_core(state, a_true)
    return state, meta["t"], meta["it"]


def validate_snapshot_params(
    path: str | Path,
) -> None:
    r"""Check that snapshot metadata matches current parameters.

    Raises :class:`SnapshotMismatchError` on critical mismatches
    (resolution, precision, or flow system).  The device count is
    never a reason: every spectral axis is auto-padded to divide the
    mesh (``round_up_padded``), which is what makes resume
    np-agnostic in the first place.  Prints
    warnings for non-critical differences and an info line when
    the device count differs (resume is np-agnostic).  Stored
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
