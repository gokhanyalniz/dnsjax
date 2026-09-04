"""Standard-library snapshot metadata helpers (JAX-free).

A dnsjax snapshot is a single **uncompressed tar** archive wrapping a
zarr3 store plus a JSON metadata member (see :mod:`dnsjax.snapshot`).
This module reads the parts that must be available *before* JAX is
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
"""

import contextlib
import functools
import json
import os
import subprocess
import tarfile
from pathlib import Path

#: Tar member holding the JSON metadata.
META_MEMBER = "_dnsjax_meta.json"

#: Optional tar member holding the embedded ``get_stats`` diagnostics.
STATS_MEMBER = "_dnsjax_stats.json"

#: Tar member prefix for the zarr3 component chunks
#: (``state/c/{component}/0/0/0``).
_CHUNK_PREFIX = "state/c/"
_CHUNK_SUFFIX = "/0/0/0"


class SnapshotArchiveError(ValueError):
    """A snapshot file exists but cannot be read as an archive."""


@contextlib.contextmanager
def _snapshot_tar(path: Path):
    """Open a snapshot archive, naming a damaged one.

    A short archive is where a snapshot goes wrong in practice -- an
    interrupted copy, a full disk, a job killed mid-write -- and it is
    caught *here* rather than by the readers downstream: ``tarfile``
    walks to the next header by seeking past each member's data, so
    any truncation that cuts into a component is refused before a
    single byte of state is read (measured: only a cut that lands
    exactly at the end of the last chunk, removing nothing but the
    end-of-archive marker, still parses -- and that file's data is
    complete).

    What it says while refusing is the point.  Untranslated, the
    caller sees ``ReadError: unexpected end of data`` raised from
    wherever the member list happened to be walked, naming neither
    the file nor the reason -- and the same exception on a resume
    reads as a dnsjax bug rather than a damaged checkpoint.  So every
    read in this module goes through here.

    (The per-span short-transfer guards in :mod:`dnsjax.snapshot` are
    therefore *not* the truncation defence -- they cover a short read
    or write of an intact file, which POSIX permits on a network
    filesystem.)
    """
    try:
        with tarfile.open(path, "r") as tf:
            yield tf
    except tarfile.ReadError as exc:
        raise SnapshotArchiveError(
            f"{path} is not a readable snapshot archive ({exc}); it is "
            "truncated or corrupt -- an interrupted write or copy "
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
    Cached, so at most one subprocess call per process.
    """
    try:
        proc = subprocess.run(
            [
                "git",
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
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    version = proc.stdout.strip()
    return version if proc.returncode == 0 and version else "unknown"


def write_sidecar_json(path: str | Path, payload: dict) -> None:
    """Write *payload* to *path* atomically (commit by rename).

    Every ``.bin`` stream writer -- :mod:`dnsjax.extensions.probes`,
    :mod:`dnsjax.extensions.forcing`, :mod:`dnsjax.twin._binstream` --
    creates its JSON sidecar on the **main** process while **every**
    rank tests the same path's existence to choose between "create"
    and "validate and append".  A plain ``open(path, "w")`` makes the
    path exist before its content does, so a rank whose test lands
    inside that window loads zero bytes and dies in ``json.load``:

        json.decoder.JSONDecodeError: Expecting value: line 1 column 1

    -- a genuine multi-process race, seen on a fresh output directory
    under ``mpirun -np 2``.  Writing a ``.partial`` sibling and
    ``os.replace``-ing it makes the path appear only once complete,
    so the loser of the race either sees no file (and, not being the
    main process, does nothing) or sees the whole of it.  Same
    directory, so the rename is atomic on any POSIX filesystem; it is
    the commit-by-rename discipline :mod:`dnsjax.snapshot` already
    uses for the tar itself.  A crash mid-write leaves the
    ``.partial`` behind rather than a truncated sidecar, which no
    reader globs for.

    :mod:`dnsjax.twin.driver` writes ``twin.json`` through it too:
    that file is read back by a later resume, where a truncated write
    would be just as fatal.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def is_snapshot_file(path: str | Path) -> bool:
    """True when *path* is a dnsjax single-file snapshot.

    A dnsjax snapshot is an uncompressed tar containing a
    :data:`META_MEMBER` member; testing for that member (rather than a
    suffix convention) is what lets the caller tell a real snapshot
    from any other file it was handed.
    """
    path = Path(path)
    if not path.is_file():
        return False
    if not tarfile.is_tarfile(path):
        return False
    with _snapshot_tar(path) as tf:
        return META_MEMBER in tf.getnames()


#: Oldest readable snapshot ``format_version``.  Format 6 stores the
#: state in the solver's native spectral layout, in physical
#: components for every family (cylindrical/annular `$u_r$`,
#: `$u_\theta$`, the physical conformation tensor), with the embedded
#: ``params`` dump in the flow-relevant public-named surface
#: representation.  A pre-6 file differs in at least one of those
#: conventions and would be silently misread, so anything older is
#: rejected -- never translated (no compatibility shim by design).
MIN_FORMAT_VERSION: int = 6


def read_snapshot_meta(path: str | Path) -> dict:
    """Return the parsed ``_dnsjax_meta.json`` member of a snapshot.

    The single version choke point: every consumer (resume, offline
    analysis, scripts) reads metadata through here, and a snapshot
    older than :data:`MIN_FORMAT_VERSION` is rejected with a clear
    message rather than misread under the wrong params convention.
    """
    path = Path(path)
    with _snapshot_tar(path) as tf:
        member = tf.extractfile(META_MEMBER)
        if member is None:
            raise ValueError(f"{path} has no {META_MEMBER} member.")
        meta = json.loads(member.read())
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


#: Bytes per element of the complex dtypes the snapshot writer emits
#: (:func:`dnsjax.snapshot._zarr3_dtype_name`).  This module is
#: deliberately numpy-free, so the size is tabulated rather than
#: looked up; an unrecognised name skips the size check below instead
#: of inventing a number.
_ITEMSIZE = {"complex64": 8, "complex128": 16}


def _check_chunks_match_meta(
    path: Path, meta_raw: bytes | None, sizes: dict[int, int]
) -> None:
    """The chunks must hold exactly what ``native_shape`` claims.

    The raw offset I/O in :mod:`dnsjax.snapshot` computes every read
    and write position arithmetically from ``native_shape`` and never
    consults the member it lands in, so if the two ever disagree a
    reader walks off the end of one chunk and into the next
    component's bytes -- and returns them as state.  Both come from
    one call in the writer today, which is exactly the sort of
    invariant that holds until someone refactors around it, and
    nothing downstream could tell afterwards: the wrong bytes are
    well-formed complex numbers.

    Cheap enough to do unconditionally -- the sizes are already in the
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


def snapshot_component_offsets(path: str | Path) -> dict[int, int]:
    """Map each state component to its data byte offset in the tar.

    The returned offset is ``tarfile.TarInfo.offset_data`` -- the first
    byte of the component's raw chunk inside the archive -- used as the
    base for the raw offset I/O in :mod:`dnsjax.snapshot`.  The component
    count is the number of chunks: 3 for the velocity-only systems, 9
    for the viscoelastic ones (3 velocity + 6 conformation); the chunks
    must be a contiguous range ``0..N-1``.

    This is the one place that hands out byte positions to trust, so it
    is also where they are checked against the metadata that describes
    them (:func:`_check_chunks_match_meta`) -- rather than leaving each
    caller to remember.
    """
    path = Path(path)
    offsets: dict[int, int] = {}
    sizes: dict[int, int] = {}
    meta_raw: bytes | None = None
    with _snapshot_tar(path) as tf:
        for m in tf.getmembers():
            name = m.name
            if name == META_MEMBER:
                member = tf.extractfile(m)
                meta_raw = None if member is None else member.read()
            elif name.startswith(_CHUNK_PREFIX) and name.endswith(
                _CHUNK_SUFFIX
            ):
                comp = int(name[len(_CHUNK_PREFIX) :].split("/", 1)[0])
                offsets[comp] = m.offset_data
                sizes[comp] = m.size
    if not offsets or set(offsets) != set(range(len(offsets))):
        raise SnapshotArchiveError(
            f"{path} is missing component chunks (found {sorted(offsets)})."
        )
    _check_chunks_match_meta(path, meta_raw, sizes)
    return offsets
