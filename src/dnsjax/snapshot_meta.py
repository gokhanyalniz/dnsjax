"""Standard-library snapshot metadata helpers (JAX-free).

This module reads the parts of a snapshot (the format:
:mod:`dnsjax.snapshot`) that must be available *before* JAX is
configured -- the embedded parameters and the component byte offsets --
using only the standard library (``tarfile`` / ``json``).  It imports
nothing from the rest of the package, so it is a safe leaf dependency
for both :mod:`dnsjax.parameters` (which reads a resumed snapshot's
parameters before the distributed backend is up) and
:mod:`dnsjax.snapshot` (no import cycle).  It also hosts the
stdlib-only :func:`git_hash` provenance helper, printed at solver
startup and recorded in every snapshot's metadata, and
:func:`write_sidecar_json`, the atomic writer every ``.bin`` stream's
JSON sidecar is created with.

Design notes
------------
**Truncated archives.**  A short archive is where a snapshot goes wrong
in practice -- an interrupted copy, a full disk, a job killed
mid-write -- and it is refused when the archive is opened, before a
byte of state is read: ``tarfile`` walks to the next header by seeking
past each member's data, so any truncation that cuts into a component
fails there.  Measured: only a cut landing exactly at the end of the
last chunk, removing nothing but the end-of-archive marker, still
parses, and that file's data is complete.  Untranslated, the caller
sees ``ReadError: unexpected end of data`` from wherever the member
list happened to be walked, naming neither the file nor the reason, and
on a resume it reads as a dnsjax bug rather than a damaged checkpoint;
hence :func:`_snapshot_tar`.  The per-span short-transfer guards of
:mod:`dnsjax.snapshot` are not the truncation defence: they cover a
short read or write of an intact file, which POSIX permits on a network
filesystem.

**Spawning git without ``vfork``.**  CPython spawns through
``posix_spawn`` only for an executable named with its directory and --
in a build without ``os.POSIX_SPAWN_CLOSEFROM``, uv's interpreters
among them -- with no descriptors to close (git inherits the caller's
inheritable ones for its few milliseconds), hence ``shutil.which`` and
``close_fds=False`` in :func:`git_hash`.  CPython's other path is
``vfork``, whose child runs in the caller's memory until it execs: on a
solver rank that child runs any exec wrapper the launcher interposed,
and such a wrapper can corrupt the parent (observed with Spindle 0.13:
rank 0 died between the ``Distribution initialized`` and ``Code
version`` lines, on ARCHER2 and on one local process).
``posix_spawn`` runs no interposed code in the child and, unlike
``fork``, neither copies a large multithreaded process's page tables
nor runs the fork handlers of the MPI and RPC libraries it has loaded.

**Atomic sidecars.**  A ``.bin`` stream writer (an extension
section's, say) creates its JSON sidecar on the **main** process while
**every** rank tests the same path's existence to choose between
"create" and "validate and append".  A plain ``open(path, "w")`` makes
the path exist before its content does, so a rank whose test lands
inside that window loads zero bytes and dies in ``json.load``
(``JSONDecodeError: Expecting value: line 1 column 1``) -- a race seen
on a fresh output directory under ``mpirun -np 2``, and measured at 138
of 300 attempts with a concurrent reader: a routine multi-process
failure, not a rare one.  Committing by rename
(:func:`write_sidecar_json`) makes the path appear only once complete,
so the loser of the race sees either no file (and, not being the main
process, does nothing) or the whole of it; the rename is atomic on any
POSIX filesystem because the ``.partial`` sibling shares the directory.
A run record a later resume reads back is written the same way, where
a truncated write would be equally fatal.
"""

import contextlib
import functools
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path

#: Tar member holding the JSON metadata.
META_MEMBER = "_dnsjax_meta.json"

#: Optional tar member holding the embedded ``get_stats`` diagnostics.
STATS_MEMBER = "_dnsjax_stats.json"

#: Tar member prefix for the zarr3 component chunks
#: (``state/c/{component}/0/0/0``).
_CHUNK_PREFIX = "state/c/"
_CHUNK_SUFFIX = "/0/0/0"

#: Tar member prefix of the optional solver-carried fields
#: (``carry/c/{slot}/0/0/0``; the :mod:`dnsjax.snapshot` docstring),
#: written by :func:`dnsjax.snapshot.save_snapshot` under
#: ``outs.snapshot_embed_carry`` and read back only by the solver's own
#: resume.  Every other reader walks ``state/c/`` and never sees it.
CARRY_PREFIX = "carry/c/"

#: Tar member prefix of the optional static pressure
#: (``pressure/c/0/0/0/0``): one more zarr3 array, of one component,
#: written beside a reduced-resolution snapshot's ``state`` by
#: :mod:`dnsjax.lowres` and read by
#: :func:`dnsjax.analysis.snapshot_export.read_pressure`.  Like
#: ``carry/`` it is invisible to every reader that walks ``state/c/``.
PRESSURE_PREFIX = "pressure/c/"

#: The ``kind`` of a tar that holds a solver state.  Every snapshot
#: ever written is one and carries no ``kind`` key at all, so its
#: absence means this; the writer records the key only for the other
#: kinds (arrays such as 3-D spectra), which share the container and not
#: its meaning.
STATE_KIND = "state"


class SnapshotArchiveError(ValueError):
    """A snapshot file exists but cannot be read as an archive."""


@contextlib.contextmanager
def _snapshot_tar(path: Path):
    """Open a snapshot archive, naming a damaged one.

    Every read in this module goes through here, so a truncated or
    corrupt file raises :class:`SnapshotArchiveError` naming the file
    and the likely cause, rather than ``tarfile``'s bare ``ReadError``
    (Design notes: "Truncated archives").
    """
    try:
        # ``"r:"``, never ``"r"``: a compressed archive would open
        # transparently, pass every member check, and then be read at
        # the raw byte offsets of the decompressed stream.
        with tarfile.open(path, "r:") as tf:
            yield tf
    except tarfile.ReadError as exc:
        raise SnapshotArchiveError(
            f"{path} is not a readable snapshot archive ({exc}); it is "
            "truncated or corrupt, or compressed, which no reader accepts "
            "-- an interrupted write or copy "
            "leaves exactly this.  A snapshot is written to a "
            "'.partial' file and renamed, so a complete file under "
            "the final name should never be short."
        ) from exc


@functools.cache
def git_hash() -> str:
    """Best-effort git revision of the running dnsjax source tree.

    Returns ``git describe --always --dirty --abbrev=12`` resolved
    from this file's directory: the abbreviated commit hash of the
    checkout the package is imported from, ``-dirty``-suffixed when
    the tree has uncommitted changes (and tag-prefixed if a tag is
    reachable).  Returns ``"unknown"`` when the source tree is not a
    git checkout (e.g. an installed wheel) or git is unavailable.
    Cached, so at most one git process per process.

    git is spawned through ``posix_spawn``, never ``vfork``: a
    launcher's exec wrapper runs in a ``vfork`` child and can corrupt
    the solver rank that spawned it (Design notes: "Spawning git
    without vfork").
    """
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        proc = subprocess.run(
            [
                git,
                "-C",
                str(Path(__file__).resolve().parent),
                "describe",
                "--always",
                "--dirty",
                "--abbrev=12",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            close_fds=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    version = proc.stdout.strip()
    return version if proc.returncode == 0 and version else "unknown"


def write_sidecar_json(path: str | Path, payload: dict) -> None:
    """Write *payload* to *path* atomically (commit by rename).

    The JSON goes to a ``.partial`` sibling that is ``os.replace``-d
    onto *path*, so *path* appears only once complete, as
    :mod:`dnsjax.snapshot` does for the tar itself.  A crash mid-write
    leaves the ``.partial`` behind, which no reader globs for.  Every
    ``.bin`` stream's sidecar and run record are written this way,
    because other ranks test the path while the main process writes it
    (Design notes: "Atomic sidecars").
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def is_snapshot_file(path: str | Path) -> bool:
    """True when *path* is a dnsjax single-file snapshot.

    A dnsjax snapshot is an uncompressed tar containing a
    :data:`META_MEMBER` member; testing for that member, not a suffix,
    tells a real snapshot from any other file the caller was handed.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        with tarfile.open(path, "r:") as tf:
            return META_MEMBER in tf.getnames()
    except tarfile.ReadError:
        return False


#: Oldest readable snapshot ``format_version``.  Format 6 stores the
#: state in the solver's native spectral layout, in physical
#: components for every family, with the embedded ``params`` dump in
#: the flow-relevant public-named surface representation
#: (:mod:`dnsjax.snapshot`).  A pre-6 file differs in at least one of
#: those conventions and would be silently misread, so anything older
#: is rejected -- never translated (no compatibility shim by design).
MIN_FORMAT_VERSION: int = 6


def read_snapshot_meta(path: str | Path) -> dict:
    """Return the parsed ``_dnsjax_meta.json`` member of a snapshot.

    The single version choke point: every consumer (resume, offline
    analysis, scripts) reads metadata through here or through
    :func:`read_snapshot_json`, and a snapshot older than
    :data:`MIN_FORMAT_VERSION` is rejected rather than misread under
    the wrong params convention.
    """
    path = Path(path)
    with _snapshot_tar(path) as tf:
        raw = _meta_bytes(tf, path)
    return _checked_meta(path, raw)


def read_snapshot_json(path: str | Path) -> tuple[dict, bytes, dict | None]:
    """Read both JSON members of a snapshot with one open.

    Returns ``(meta, meta_bytes, stats)``: the parsed metadata, checked
    as :func:`read_snapshot_meta` checks it; the ``_dnsjax_meta.json``
    member's bytes exactly as stored; and the parsed
    ``_dnsjax_stats.json``, or ``None`` when the snapshot carries none
    (:func:`read_snapshot_stats`).  Only the two members' data is read:
    the walk of the member list reads the tar headers alone, seeking
    past every component chunk, so the cost is the same for a state of
    any size.
    """
    path = Path(path)
    with _snapshot_tar(path) as tf:
        raw = _meta_bytes(tf, path)
        stats = None
        if STATS_MEMBER in tf.getnames():
            member = tf.extractfile(STATS_MEMBER)
            if member is not None:
                stats = json.loads(member.read())
    return _checked_meta(path, raw), raw, stats


def _meta_bytes(tf: tarfile.TarFile, path: Path) -> bytes:
    """The stored bytes of the open archive's :data:`META_MEMBER`."""
    if META_MEMBER not in tf.getnames():
        raise ValueError(
            f"{path} is not a dnsjax snapshot: no {META_MEMBER} member."
        )
    return tf.extractfile(META_MEMBER).read()


def _checked_meta(path: Path, raw: bytes) -> dict:
    """Parse metadata bytes, refusing a pre-:data:`MIN_FORMAT_VERSION` one."""
    meta = json.loads(raw)
    version = meta.get("format_version", 0)
    if version < MIN_FORMAT_VERSION:
        raise ValueError(
            f"{path} has snapshot format_version {version}; this code "
            f"reads version {MIN_FORMAT_VERSION}+ only (the stored "
            "component basis, the on-disk array layout, and the "
            "embedded parameter dump changed representation across "
            "versions, and old snapshots are not translated)."
        )
    return meta


def read_snapshot_stats(path: str | Path) -> dict | None:
    """Return the parsed ``_dnsjax_stats.json`` member of a snapshot.

    Returns ``None`` when the snapshot carries no embedded stats (the
    member is optional, written only when ``outs.snapshot_embed_stats``
    is on and stats were supplied to :func:`dnsjax.snapshot.save_snapshot`).
    """
    path = Path(path)
    with _snapshot_tar(path) as tf:
        if STATS_MEMBER not in tf.getnames():
            return None
        member = tf.extractfile(STATS_MEMBER)
        if member is None:
            return None
        return json.loads(member.read())


#: Bytes per element of the dtypes the archive writer emits: complex
#: for a state (:func:`dnsjax.snapshot._zarr3_dtype_name`), real for
#: a real array output.  This module is deliberately numpy-free, so the
#: size is tabulated rather than looked up; an unrecognized name skips
#: the size check below instead of inventing a number.
_ITEMSIZE = {"complex64": 8, "complex128": 16, "float32": 4, "float64": 8}


def _check_chunks_match_meta(
    path: Path, meta_raw: bytes | None, sizes: dict[int, int]
) -> None:
    """The chunks must hold exactly what ``native_shape`` claims.

    The raw offset I/O in :mod:`dnsjax.snapshot` computes every read
    and write position arithmetically from ``native_shape`` and never
    consults the member it lands in, so if the two ever disagree a
    reader walks off the end of one chunk and into the next
    component's bytes -- and returns them as state.  Both come from
    one call in the writer, an invariant a refactor could break with
    nothing downstream able to tell: the wrong bytes are well-formed
    complex numbers.

    Cheap enough to do unconditionally: the sizes are already in the
    tar headers being walked, and the metadata member is ~2 KiB.
    """
    if meta_raw is None:
        return
    meta = json.loads(meta_raw)
    shape = meta.get("native_shape")
    itemsize = _ITEMSIZE.get(meta.get("dtype"))
    if not shape or itemsize is None:
        return
    if len(sizes) != shape[0]:
        raise SnapshotArchiveError(
            f"{path} declares {shape[0]} state components but holds "
            f"{len(sizes)} component chunks."
        )
    expected = itemsize
    for extent in shape[1:]:
        expected *= extent
    wrong = {c: n for c, n in sizes.items() if n != expected}
    if wrong:
        dims = " x ".join(str(d) for d in shape[1:])
        raise SnapshotArchiveError(
            f"{path}: the metadata says each component is {dims} of "
            f"{meta.get('dtype')} ({expected} bytes), but component "
            f"chunk(s) {sorted(wrong)} hold "
            f"{sorted(set(wrong.values()))}.  The archive and the "
            "metadata describing it disagree, and reading it would "
            "run past the end of a chunk into the next component."
        )


def _chunk_members(
    path: Path, prefix: str
) -> tuple[dict[int, int], dict[int, int], bytes | None]:
    """``(offsets, sizes, meta_raw)`` of the ``prefix{i}/0/0/0`` chunks."""
    offsets: dict[int, int] = {}
    sizes: dict[int, int] = {}
    meta_raw: bytes | None = None
    with _snapshot_tar(path) as tf:
        for m in tf.getmembers():
            name = m.name
            if name == META_MEMBER:
                member = tf.extractfile(m)
                meta_raw = None if member is None else member.read()
            elif name.startswith(prefix) and name.endswith(_CHUNK_SUFFIX):
                comp = int(name[len(prefix) :].split("/", 1)[0])
                offsets[comp] = m.offset_data
                sizes[comp] = m.size
    return offsets, sizes, meta_raw


def snapshot_component_offsets(path: str | Path) -> dict[int, int]:
    """Map each state component to its data byte offset in the tar.

    The returned offset is ``tarfile.TarInfo.offset_data`` -- the first
    byte of the component's raw chunk inside the archive -- used as the
    base for the raw offset I/O in :mod:`dnsjax.snapshot`.  The component
    count is the number of chunks: 3 for the velocity-only systems,
    the flow's ``n_components`` for one storing more; the chunks must
    be a contiguous range ``0..N-1``.

    The one place that hands out byte positions to trust, so it checks
    them against the metadata that describes them
    (:func:`_check_chunks_match_meta`) rather than leaving that to each
    caller.
    """
    path = Path(path)
    offsets, sizes, meta_raw = _chunk_members(path, _CHUNK_PREFIX)
    if not offsets or set(offsets) != set(range(len(offsets))):
        raise SnapshotArchiveError(
            f"{path} is missing component chunks (found {sorted(offsets)})."
        )
    _check_chunks_match_meta(path, meta_raw, sizes)
    return offsets


def _member_offsets(
    path: Path, prefix: str, label: str, n_named: Callable[[dict], int]
) -> dict[int, int] | None:
    """Checked byte offsets of an optional member's chunks, or ``None``.

    ``None`` when the archive holds no ``prefix`` chunks.  Otherwise
    they are checked like the state's: a contiguous ``0..N-1`` range,
    ``N`` the count *n_named* reads off the metadata, each exactly one
    component of ``native_shape[1:]``.
    """
    offsets, sizes, meta_raw = _chunk_members(path, prefix)
    if not offsets:
        return None
    meta = json.loads(meta_raw) if meta_raw is not None else {}
    count = n_named(meta)
    if set(offsets) != set(range(count)):
        raise SnapshotArchiveError(
            f"{path} holds {label} chunks {sorted(offsets)} but its "
            f"metadata names {count} {label} field(s)."
        )
    shape = meta.get("native_shape")
    itemsize = _ITEMSIZE.get(meta.get("dtype"))
    if shape and itemsize is not None:
        expected = itemsize
        for extent in shape[1:]:
            expected *= extent
        wrong = {c: n for c, n in sizes.items() if n != expected}
        if wrong:
            raise SnapshotArchiveError(
                f"{path}: {label} chunk(s) {sorted(wrong)} hold "
                f"{sorted(set(wrong.values()))} bytes, not one component "
                f"({expected} bytes)."
            )
    return offsets


def snapshot_carry_offsets(path: str | Path) -> dict[int, int] | None:
    """Byte offsets of the optional ``carry/`` chunks, or ``None``.

    ``None`` when the archive has none (every snapshot but a
    pipe-family one written with ``outs.snapshot_embed_carry``).  The
    chunks are checked like the state's: a contiguous ``0..N-1`` range,
    ``N`` the metadata's ``carried`` count, each exactly one component
    of ``native_shape[1:]``.
    """
    return _member_offsets(
        Path(path),
        CARRY_PREFIX,
        "carried",
        lambda meta: len(meta.get("carried") or []),
    )


def snapshot_pressure_offsets(path: str | Path) -> dict[int, int] | None:
    """Byte offset of the optional ``pressure/`` chunk, or ``None``.

    ``None`` when the archive carries no pressure (every full snapshot;
    a reduced one written with ``lowres.pressure`` off).  Otherwise
    ``{0: offset}``, checked against the metadata's ``pressure`` entry
    and against one component of ``native_shape[1:]``.
    """
    return _member_offsets(
        Path(path),
        PRESSURE_PREFIX,
        "pressure",
        lambda meta: 1 if meta.get("pressure") else 0,
    )


def snapshot_kind(meta: dict) -> str:
    """A tar's ``kind``: :data:`STATE_KIND` unless it records another."""
    return str(meta.get("kind", STATE_KIND))


def checkpoint_refusal(meta: dict, path: str | Path) -> str | None:
    """Why the tar described by *meta* cannot seed a run, or ``None``.

    Two kinds of dnsjax tar are not checkpoints though they share the
    container: an output of another kind (a ``kind`` other than
    :data:`STATE_KIND`), which holds something other than a field, and
    the reduced-resolution snapshots of :mod:`dnsjax.lowres` (a
    ``lowres`` entry), which hold a filtered field without what a
    resume needs.  Shared by the resume path
    (:func:`dnsjax.parameters.read_snapshot_params`,
    :func:`dnsjax.snapshot.validate_snapshot_params`) and every other
    reader that wants a checkpoint.
    """
    kind = snapshot_kind(meta)
    if kind != STATE_KIND:
        return f"{path} is a {kind!r} file, not a state snapshot."
    if meta.get("lowres") is not None:
        return (
            f"{path} is a reduced-resolution output snapshot (lowres), "
            "not a checkpoint: resume from a full state*.tar."
        )
    return None
