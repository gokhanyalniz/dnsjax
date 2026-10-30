#!/usr/bin/env python3
r"""Cluster check: does a dnsjax archive hold every byte its writers sent?

:func:`dnsjax.snapshot.write_archive` lays an archive out with every
chunk's data region sparse-reserved
(:func:`dnsjax.snapshot._write_tar_skeleton`), and each device then
writes its own rows of every chunk.  A byte range that no write reached
is still a hole in the file, which the filesystem reports from metadata
alone (``SEEK_HOLE`` / ``SEEK_DATA``): no field data is read, so the
scan costs the same on any file size and suits a login node.  A complete
archive has no hole.

Use it on an archive written while the filesystem was failing, such as
one whose run log shows a retried write (the ``retrying in`` lines of
:func:`dnsjax.snapshot._retry_transient`).  The writer commits an
archive only once every process's ``fsync`` has succeeded, which relies
on the filesystem reporting a dropped write to the process that made
it; this scan checks the file instead of that contract.  A retried
write sends the same bytes again from the same state, so a range that
an earlier attempt landed holds the right data either way.  Each hole
is located by the member, component and leading-axis rows it falls in,
with the mesh devices that write those rows
(:func:`dnsjax.snapshot._a_ranges`), for every archive the writer
produces: snapshots, reduced-resolution snapshots and the twin cubes.
Any other file is scanned too, its holes given as byte ranges.

**Positive control.**  A filesystem without hole reporting answers
through the kernel's generic fallback, which calls the whole file data,
so a clean scan there would mean nothing.  Before scanning, the script
writes a probe file in each directory it scans (one byte after a
:data:`PROBE_HOLE` hole), ``fsync``-s it, checks that the hole is
reported and deletes it; the files of a directory whose probe shows no
hole are left undecided.

**Limits.**  A filesystem tracks holes per block.  A dropped range that
shares a block with bytes another write landed leaves zeros in an
allocated block, which no hole scan can see; any dropped range of two
blocks or more covers a whole block, so the scan reports it.  Scan the
file the run wrote, on its filesystem: a copy made without preserving
holes (``rsync`` without ``--sparse``, for one) stores the zeros as
data.  Scan an archive written without trouble beside it: a
filesystem that stores all-zero blocks as holes (ZFS with compression,
for one) also reports the rows that hold zeros in every state, in both.

Run from the run folder; the output is the report::

    .venv/bin/python lab/scripts/diagnostics/snapshot_holes.py \
        state00000_twin.tar state00000.tar

Exit status: 0 when no file has a hole and every probe showed its hole;
1 when a file has a hole; 2 when no file has a hole but a probe showed
none: the filesystem does not report holes, and the scan cannot decide
(read the file instead).
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dnsjax.snapshot_meta import _ITEMSIZE, META_MEMBER

#: Length of the probe file's hole: a multiple of, and at least, the
#: allocation unit of the filesystems a run writes to (a Lustre stripe,
#: a GPFS block).
PROBE_HOLE = 64 << 20

#: A component chunk member, ``{array}/c/{component}/0/0/0``
#: (:func:`dnsjax.snapshot._write_tar_skeleton`).
_CHUNK = re.compile(r"(?P<array>.+)/c/(?P<comp>\d+)/0/0/0")


@dataclass(frozen=True)
class Chunk:
    """One component chunk of an archive: its byte range and its rows.

    ``rows`` is the extent of the leading axis, the one the writer
    splits over the devices; ``row_bytes`` is the size of one row.
    """

    array: str
    comp: int
    start: int
    size: int
    rows: int
    row_bytes: int


def holes(path: Path) -> list[tuple[int, int]]:
    """``(start, length)`` of every hole in *path*, from metadata alone.

    Walks ``SEEK_HOLE`` / ``SEEK_DATA``; the implicit hole at the end
    of the file is not one.
    """
    found = []
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        pos = 0
        while pos < size:
            start = os.lseek(fd, pos, os.SEEK_HOLE)
            if start >= size:
                break
            try:
                pos = os.lseek(fd, start, os.SEEK_DATA)
            except OSError as error:
                if error.errno != errno.ENXIO:
                    raise
                pos = size  # the hole runs to the end of the file
            found.append((start, pos - start))
    finally:
        os.close(fd)
    return found


def reports_holes(directory: Path) -> bool:
    """Whether the filesystem holding *directory* reports holes.

    Writes a probe file there, one byte after a :data:`PROBE_HOLE`
    hole, and ``fsync``-s it so that the hole is the storage's, not the
    page cache's; the probe is deleted once scanned.  The byte is not
    zero, which a filesystem storing all-zero blocks as holes would
    drop.
    """
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=".snapshot_holes_probe_"
    ) as f:
        f.seek(PROBE_HOLE)
        f.write(b"\x01")
        f.flush()
        os.fsync(f.fileno())
        return bool(holes(Path(f.name)))


def chunks(path: Path) -> tuple[list[Chunk], int, int | None]:
    """A dnsjax archive's chunks, declared chunk count and device count.

    The declared count sums the component counts of the ``zarr.json``
    members, which precede every chunk; ``tarfile`` stops at a zeroed
    header, so a hole reaching one leaves fewer chunks than declared.
    The device count is the writer's mesh size, from the archive's
    metadata.  ``([], 0, None)`` for a file that is not a dnsjax
    archive; reads the tar headers and the small JSON members only.
    """
    arrays, n_devices, members = {}, None, []
    try:
        with tarfile.open(path, "r:") as tf:
            for member in tf:
                members.append(member)
                if member.name == META_MEMBER:
                    meta = json.load(tf.extractfile(member))
                    n_devices = meta.get("n_devices")
                elif member.name.endswith("/zarr.json"):
                    array = member.name.removesuffix("/zarr.json")
                    arrays[array] = json.load(tf.extractfile(member))
    except tarfile.ReadError:
        return [], 0, None
    found = []
    for member in members:
        match = _CHUNK.fullmatch(member.name)
        if match is None or match["array"] not in arrays:
            continue
        zarr = arrays[match["array"]]
        shape = zarr["shape"]
        itemsize = _ITEMSIZE[zarr["data_type"]]
        found.append(
            Chunk(
                array=match["array"],
                comp=int(match["comp"]),
                start=member.offset_data,
                size=member.size,
                rows=shape[1],
                row_bytes=math.prod(shape[2:]) * itemsize,
            )
        )
    declared = sum(zarr["shape"][0] for zarr in arrays.values())
    return found, declared, n_devices


def locate(
    start: int, length: int, found: list[Chunk], n_devices: int | None
) -> list[str]:
    """Where the hole ``[start, start + length)`` falls, one line a part.

    A row ``r`` of a chunk with ``rows`` rows is written by mesh device
    ``r // ceil(rows / n_devices)`` (:func:`dnsjax.snapshot._a_ranges`).
    """
    end, lines, located = start + length, [], 0
    for chunk in found:
        lo = max(start, chunk.start)
        hi = min(end, chunk.start + chunk.size)
        if lo >= hi:
            continue
        located += hi - lo
        r0 = (lo - chunk.start) // chunk.row_bytes
        r1 = (hi - 1 - chunk.start) // chunk.row_bytes
        line = (
            f"{chunk.array} component {chunk.comp}, rows {r0}-{r1} of "
            f"{chunk.rows}"
        )
        if n_devices:
            per_device = -(-chunk.rows // n_devices)
            line += (
                f" (devices {r0 // per_device}-{r1 // per_device} of "
                f"{n_devices})"
            )
        lines.append(f"{line}: {hi - lo} bytes")
    if located < length:
        lines.append(f"outside the component data: {length - located} bytes")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "files", nargs="+", type=Path, help="archives (or any files) to scan"
    )
    args = ap.parse_args()

    probes: dict[Path, bool] = {}
    found_hole = undecided = False
    for path in args.files:
        directory = path.resolve().parent
        if directory not in probes:
            probes[directory] = reports_holes(directory)
            print(
                f"probe in {directory}: "
                + (
                    "hole reported"
                    if probes[directory]
                    else "no hole reported; this filesystem does not "
                    "report holes"
                )
            )
        if not probes[directory]:
            undecided = True
            print(f"{path}: undecided")
            continue
        file_holes = holes(path)
        if not file_holes:
            print(f"{path}: no holes ({path.stat().st_size} bytes)")
            continue
        found_hole = True
        total = sum(length for _, length in file_holes)
        print(f"{path}: {len(file_holes)} hole(s), {total} bytes")
        found, declared, n_devices = chunks(path)
        if not found:
            print(
                "  no component chunk located: not a dnsjax archive, or a "
                "hole reaches its first member headers"
            )
        elif len(found) < declared:
            print(
                f"  {len(found)} of {declared} component chunks located: a "
                "hole reaches a member header, and the bytes after it "
                "count as outside the component data"
            )
        for start, length in file_holes:
            print(f"  [{start}, {start + length}):")
            for line in locate(start, length, found, n_devices):
                print(f"    {line}")
    if found_hole:
        return 1
    return 2 if undecided else 0


if __name__ == "__main__":
    sys.exit(main())
