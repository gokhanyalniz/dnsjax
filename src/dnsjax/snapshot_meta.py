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
startup and recorded in every snapshot's metadata.
"""

import functools
import json
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
    except OSError, subprocess.SubprocessError:
        return "unknown"
    version = proc.stdout.strip()
    return version if proc.returncode == 0 and version else "unknown"


def is_snapshot_file(path: str | Path) -> bool:
    """True when *path* is a dnsjax single-file snapshot.

    A dnsjax snapshot is an uncompressed tar containing a
    :data:`META_MEMBER` member; this distinguishes it from a legacy
    ``.npz`` (a zip) or any other file, so the caller can dispatch the
    initial condition without a suffix convention.
    """
    path = Path(path)
    if not path.is_file():
        return False
    if not tarfile.is_tarfile(path):
        return False
    with tarfile.open(path, "r") as tf:
        return META_MEMBER in tf.getnames()


def read_snapshot_meta(path: str | Path) -> dict:
    """Return the parsed ``_dnsjax_meta.json`` member of a snapshot."""
    path = Path(path)
    with tarfile.open(path, "r") as tf:
        member = tf.extractfile(META_MEMBER)
        if member is None:
            raise ValueError(f"{path} has no {META_MEMBER} member.")
        return json.loads(member.read())


def read_snapshot_stats(path: str | Path) -> dict | None:
    """Return the parsed ``_dnsjax_stats.json`` member of a snapshot.

    Returns ``None`` when the snapshot carries no embedded stats (the
    member is optional, written only when ``outs.snapshot_embed_stats``
    is on and stats were supplied to :func:`dnsjax.snapshot.save_snapshot`).
    """
    path = Path(path)
    with tarfile.open(path, "r") as tf:
        if STATS_MEMBER not in tf.getnames():
            return None
        member = tf.extractfile(STATS_MEMBER)
        if member is None:
            return None
        return json.loads(member.read())


def snapshot_component_offsets(path: str | Path) -> dict[int, int]:
    """Map each state component to its data byte offset in the tar.

    The returned offset is ``tarfile.TarInfo.offset_data`` -- the first
    byte of the component's raw chunk inside the archive -- used as the
    base for the raw offset I/O in :mod:`dnsjax.snapshot`.  The component
    count is the number of chunks: 3 for the velocity-only systems, 9
    for the viscoelastic system (3 velocity + 6 conformation); the chunks
    must be a contiguous range ``0..N-1``.
    """
    path = Path(path)
    offsets: dict[int, int] = {}
    with tarfile.open(path, "r") as tf:
        for m in tf.getmembers():
            name = m.name
            if name.startswith(_CHUNK_PREFIX) and name.endswith(_CHUNK_SUFFIX):
                comp = int(name[len(_CHUNK_PREFIX) :].split("/", 1)[0])
                offsets[comp] = m.offset_data
    if not offsets or set(offsets) != set(range(len(offsets))):
        raise ValueError(
            f"{path} is missing component chunks (found {sorted(offsets)})."
        )
    return offsets
