"""Standard-library snapshot metadata helpers (JAX-free).

A dnsjax snapshot is a single **uncompressed tar** archive wrapping a
zarr3 store plus a JSON metadata member (see :mod:`dnsjax.snapshot`).
This module reads the parts that must be available *before* JAX is
configured -- the embedded parameters and the component byte offsets --
using only the standard library (``tarfile`` / ``json``).  It imports
nothing from the rest of the package, so it is a safe leaf dependency
for both :mod:`dnsjax.parameters` (which reads a resumed snapshot's
parameters before the distributed backend is up) and
:mod:`dnsjax.snapshot` (no import cycle).
"""

import json
import tarfile
from pathlib import Path

#: Tar member holding the JSON metadata.
META_MEMBER = "_dnsjax_meta.json"

#: Tar member prefix for the zarr3 component chunks
#: (``state/c/{component}/0/0/0``).
_CHUNK_PREFIX = "state/c/"
_CHUNK_SUFFIX = "/0/0/0"


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


def snapshot_component_offsets(path: str | Path) -> dict[int, int]:
    """Map each velocity component to its data byte offset in the tar.

    The returned offset is ``tarfile.TarInfo.offset_data`` -- the first
    byte of the component's raw chunk inside the archive -- used as the
    base for the raw offset I/O in :mod:`dnsjax.snapshot`.
    """
    path = Path(path)
    offsets: dict[int, int] = {}
    with tarfile.open(path, "r") as tf:
        for m in tf.getmembers():
            name = m.name
            if name.startswith(_CHUNK_PREFIX) and name.endswith(_CHUNK_SUFFIX):
                comp = int(name[len(_CHUNK_PREFIX) :].split("/", 1)[0])
                offsets[comp] = m.offset_data
    if set(offsets) != {0, 1, 2}:
        raise ValueError(
            f"{path} is missing component chunks (found {sorted(offsets)})."
        )
    return offsets
