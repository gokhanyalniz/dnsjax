r"""Rate-capped reads of the lab's binary streams.  JAX-free.

The lab's diagnostic streams (``twin_yspectra.bin``,
``twin_ybudget.bin``, ``twin_spectra.bin`` and their reference streams,
``probes.bin``, ``forcing.bin``) are fixed-size records behind a JSON
sidecar, and a production run's reach several GB.  Their readers read at
full disk speed by default -- a memory map, or ``np.fromfile`` -- and
every one of them can read at a capped rate instead, through this
module:

- :func:`read_records` reads a range of whole records: by
  ``np.fromfile`` without a throttle, by ``pread`` at the throttle's
  rate with one;
- :class:`CappedRecords` stands in for a stream's memory map
  (:func:`map_records`), reading a field of a record only when it is
  indexed;
- :func:`read_column` reads one field of every record (the ``t`` column
  a reader deduplicates on), one ``pread`` per record;
- :func:`paced` charges a file that another library reads whole (a
  cube, a reduced snapshot, a ``.dat`` stream) to the process's cap.

A ``pread`` here switches readahead off on its descriptor
(``POSIX_FADV_RANDOM``) and releases the pages it read from the page
cache (``POSIX_FADV_DONTNEED``), so that the traffic is what was asked
for and nothing accumulates in memory (Design notes, "Why pread, not a
memory map").

The cap
-------
One throttle for the process (Design notes, "One throttle per
process"), off until set: :func:`set_read_cap`, or :func:`capped` for a
block, in MB/s (`$10^6$` bytes per second).  The ``--read-cap`` of
``lab/scripts/twin_spectral_maps.py`` and
``lab/scripts/twin_member_check.py`` sets it for their run.  A reader's
``read_cap`` argument caps that one call: :func:`throttle` gives it a
throttle of its own, chained to the process's, so a call's cap only
ever lowers the rate.  Whether a stream is memory-mapped or read
through :class:`CappedRecords` is decided when it is opened, so set the
cap before opening it.

The cap bounds the **average** rate.  A capped ``pread`` is at most
:data:`_CHUNK_BYTES`, or one record where a record is larger, read at
full speed and then paid for; a file read whole under :func:`paced` is
one burst of its size, paid for before it starts.

Design notes
------------
**Why a rate cap.**  Sustained reads of several GB at full disk speed
have been followed by fatal machine-check exceptions in a WSL2 guest
(WSL 2.6.3, Linux 6.6.87.2): among them a member check reading about
20 GB in under a minute, and a byte comparison of two 4.5 GB snapshots.
Reads of a few hundred MB, and long CPU-bound jobs, were not, and a
memory limit on the process did not prevent it.  Reads by ``pread`` at
25 MB/s, each read's pages released, read 20 GB there without one.  One
more came ten seconds into a read capped at 12.5 MB/s, with the page
cache already holding some 12 GB from another source: the cap lowers
the risk, it does not remove it.

**Why pread, not a memory map.**  A memory map reads whole pages, lets
the kernel read ahead around each fault, and keeps what it read cached:
walking a stream reads it at full speed and leaves it in memory.  An
explicit ``pread`` of the bytes a reader asks for, with readahead off
and the pages released after each read, makes the traffic exactly the
request and lets the throttle see every byte.  Where
``os.posix_fadvise`` is absent (macOS) the reads are still capped, and
their pages stay cached.

**Whole pages.**  The disk transfers, and the page cache holds, whole
pages, so a read is charged for the page span it touches: the 8-byte
``t`` of a record costs a page, and the ``t`` column of a stream of `$N$`
records `$N$` pages.  A read narrower than its span would otherwise be
undercharged by up to two pages.

**One throttle per process.**  A reader of several streams reads them
in turn: an ensemble's members, a stream and its reference.  Throttles
of their own would each carry a second of unused allowance through the
others' turns, so that `$k$` of them could spend `$k$` seconds' worth at
once; one throttle for the process holds the whole of its traffic to
the rate.  A call's own ``read_cap`` throttle charges the process's as
well, which therefore saves up no allowance while it runs, and the
lower of the two rates holds.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import mmap
import os
import time
import weakref
from collections.abc import Iterator
from pathlib import Path

import numpy as np

__all__ = [
    "MAX_READ_CAP",
    "CappedRecords",
    "Throttle",
    "bytes_read",
    "capped",
    "map_records",
    "paced",
    "parse_read_cap",
    "read_column",
    "read_records",
    "set_read_cap",
    "throttle",
]

#: Bytes in the MB of every cap here: the decimal megabyte.
_MB: float = 1e6

#: The largest cap accepted, in MB/s.  No disk reads 100 GB/s, so a
#: larger value is a rate given in bytes per second by mistake.
MAX_READ_CAP: float = 1e5

#: Whole records up to this many bytes per capped read, and at least one
#: record however large: a burst of this size at full speed, then paid
#: for.  One record of a production stream, a few MB, is the size per
#: read the cap has been used with (Design notes, "Why a rate cap").
_CHUNK_BYTES: int = 8 * 2**20

#: The unit the disk transfers and the page cache holds (Design notes,
#: "Whole pages").
_PAGE: int = mmap.PAGESIZE

#: The longest idle stretch a throttle carries forward, in seconds.
_ALLOWANCE: float = 1.0

#: The clock and the sleep the throttles use, module-level so that a
#: test can drive them.
_clock = time.monotonic
_sleep = time.sleep

_FADVISE: bool = hasattr(os, "posix_fadvise")
_PREADV: bool = hasattr(os, "preadv")

#: Bytes charged to a throttle since the process started.
_charged: int = 0


class Throttle:
    """An average read rate with at most one second of unused allowance.

    *rate* is in bytes per second, ``None`` for no limit.  *parent* is
    charged after this throttle, so a chain holds to the lowest of its
    rates.  One thread.
    """

    def __init__(
        self, rate: float | None = None, parent: Throttle | None = None
    ) -> None:
        self.rate = rate
        self.parent = parent
        self.free_at = _clock()

    def wait(self, nbytes: int) -> None:
        """Block until *nbytes* more fit under every rate of the chain."""
        if self.rate is not None:
            now = _clock()
            self.free_at = (
                max(self.free_at, now - _ALLOWANCE) + nbytes / self.rate
            )
            if self.free_at > now:
                _sleep(self.free_at - now)
        if self.parent is not None:
            self.parent.wait(nbytes)


#: The process's throttle: its rate is changed in place, so that a
#: stand-in opened under a cap follows the cap.
_PROCESS = Throttle()


def _rate(mb_per_s: float) -> float:
    """A cap in MB/s as bytes per second, refused outside the range."""
    value = float(mb_per_s)
    if not (math.isfinite(value) and 0.0 < value <= MAX_READ_CAP):
        raise ValueError(
            f"read cap {mb_per_s!r} MB/s: expected a rate in (0, "
            f"{MAX_READ_CAP:g}] MB/s, 1 MB being 10^6 bytes; a larger "
            "value is a rate in bytes per second."
        )
    return value * _MB


def parse_read_cap(text: str) -> float:
    """``--read-cap`` on a command line: MB/s, refused as the cap is.

    An ``argparse`` ``type``, so that a bad value is a usage error.
    """
    try:
        value = float(text)
        _rate(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def set_read_cap(mb_per_s: float | None) -> float | None:
    """Set the process's cap, in MB/s; ``None`` lifts it.

    Returns the previous cap, in MB/s or ``None``, for restoring it.
    """
    previous = None if _PROCESS.rate is None else _PROCESS.rate / _MB
    _PROCESS.rate = None if mb_per_s is None else _rate(mb_per_s)
    return previous


@contextlib.contextmanager
def capped(mb_per_s: float | None) -> Iterator[None]:
    """The process's cap at *mb_per_s* for the block, then as it was.

    ``None`` leaves the cap as it is.
    """
    if mb_per_s is None:
        yield
        return
    previous = set_read_cap(mb_per_s)
    try:
        yield
    finally:
        set_read_cap(previous)


def throttle(read_cap: float | None = None) -> Throttle | None:
    """The throttle a read takes, from a reader's *read_cap* (MB/s).

    A number gives a throttle of its own at that rate, chained to the
    process's (module docstring, "The cap"), with no allowance at its
    start; ``None`` the process's, when a cap is set; otherwise
    ``None``, a read at full speed.
    """
    if read_cap is not None:
        return Throttle(_rate(read_cap), parent=_PROCESS)
    return _PROCESS if _PROCESS.rate is not None else None


def bytes_read() -> int:
    """Bytes charged to a throttle since the process started.

    In whole pages (Design notes, "Whole pages"); a read without a
    throttle is not counted.
    """
    return _charged


def _span(offset: int, nbytes: int) -> tuple[int, int]:
    """The ``(start, length)`` of the pages a read touches."""
    start = offset - offset % _PAGE
    stop = -(-(offset + nbytes) // _PAGE) * _PAGE
    return start, stop - start


def _open(path: Path) -> int:
    """A read-only descriptor with readahead off."""
    fd = os.open(path, os.O_RDONLY)
    if _FADVISE:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    return fd


def _bytes(array: np.ndarray) -> memoryview:
    """A writable byte view of a C-contiguous array."""
    return memoryview(array.reshape(-1).view(np.uint8))


def _pread_into(
    fd: int,
    buffer: memoryview,
    offset: int,
    limit: Throttle | None,
    path: Path,
) -> None:
    """Fill *buffer* from *offset*, charged first and released after."""
    global _charged
    start, length = _span(offset, len(buffer))
    if limit is not None:
        limit.wait(length)
        _charged += length
    done = 0
    while done < len(buffer):
        if _PREADV:
            got = os.preadv(fd, [buffer[done:]], offset + done)
        else:
            data = os.pread(fd, len(buffer) - done, offset + done)
            got = len(data)
            buffer[done : done + got] = data
        if got == 0:
            raise OSError(
                f"{path}: short read at offset {offset + done} "
                f"({done} of {len(buffer)} bytes)"
            )
        done += got
    if _FADVISE:
        os.posix_fadvise(fd, start, length, os.POSIX_FADV_DONTNEED)


def _complete(path: Path, dtype: np.dtype) -> int:
    """The number of complete records the file holds."""
    return path.stat().st_size // dtype.itemsize


def _field(dtype: np.dtype, name: str) -> tuple[np.dtype, int]:
    """A field's ``(dtype, offset)``, refused as numpy refuses it."""
    fields = dtype.fields or {}
    if name not in fields:
        raise ValueError(f"no field of name {name}")
    sub, offset = fields[name][:2]
    return sub, offset


def read_records(
    path: str | Path,
    dtype: np.dtype,
    *,
    start: int = 0,
    stop: int | None = None,
    throttle: Throttle | None = None,
) -> np.ndarray:
    """The complete records ``[start, stop)`` of a fixed-size-record file.

    *stop* defaults to the last complete record, and a partial trailing
    record is never read; an empty range gives an empty array.  Without
    a *throttle* this is ``np.fromfile`` of the range; with one, a
    ``pread`` per chunk of whole records (:data:`_CHUNK_BYTES`) straight
    into the returned array, each charged to the throttle and its pages
    released.  The two return the same bytes.
    """
    path, dtype = Path(path), np.dtype(dtype)
    if start < 0 or (stop is not None and stop < 0):
        raise ValueError(f"record range [{start}, {stop}) is negative")
    n = _complete(path, dtype)
    stop = n if stop is None else min(int(stop), n)
    start = min(int(start), stop)
    count, size = stop - start, dtype.itemsize
    if throttle is None:
        return np.fromfile(path, dtype=dtype, count=count, offset=start * size)
    out = np.empty(count, dtype=dtype)
    if count == 0:
        return out
    raw = out.view(np.uint8)
    per = max(1, _CHUNK_BYTES // size)
    fd = _open(path)
    try:
        for a in range(0, count, per):
            b = min(a + per, count)
            _pread_into(
                fd,
                memoryview(raw[a * size : b * size]),
                (start + a) * size,
                throttle,
                path,
            )
    finally:
        os.close(fd)
    return out


def read_column(
    path: str | Path,
    dtype: np.dtype,
    name: str,
    *,
    throttle: Throttle | None = None,
) -> np.ndarray:
    """One field of every complete record, a ``pread`` per record.

    ``(n_records, *field shape)`` in the stored dtype.  Readahead is off
    and each read's pages are released whether or not a *throttle*
    charges them, so a pass over a stream's ``t`` reads a page per
    record, never a readahead window around each.
    """
    path, dtype = Path(path), np.dtype(dtype)
    sub, offset = _field(dtype, name)
    n = _complete(path, dtype)
    out = np.empty((n, *sub.shape), dtype=sub.base)
    fd = _open(path)
    try:
        for i in range(n):
            _pread_into(
                fd,
                _bytes(out[i : i + 1]),
                i * dtype.itemsize + offset,
                throttle,
                path,
            )
    finally:
        os.close(fd)
    return out


class _Field:
    """One field of a :class:`CappedRecords`, read row by row."""

    def __init__(self, records: CappedRecords, name: str) -> None:
        self._records = records
        self._sub, self._offset = _field(records.dtype, name)

    def __getitem__(self, rows) -> np.ndarray:
        records, sub = self._records, self._sub
        if isinstance(rows, (int, np.integer)):
            out = np.empty((1, *sub.shape), dtype=sub.base)
            records._read(out, records._at(rows) + self._offset)
            return out[0]
        if isinstance(rows, slice):
            rows = np.arange(len(records))[rows]
        rows = np.asarray(rows)
        if rows.ndim == 0:
            return self[int(rows)]
        if not np.issubdtype(rows.dtype, np.integer):
            raise IndexError("arrays used as indices must be of integer type")
        flat = rows.reshape(-1)
        out = np.empty((flat.size, *sub.shape), dtype=sub.base)
        for k, row in enumerate(flat):
            records._read(out[k : k + 1], records._at(row) + self._offset)
        return out.reshape(*rows.shape, *sub.shape)


class CappedRecords:
    r"""A stream's records read on demand: a stand-in for ``np.memmap``.

    Serves the indexing the lab's stream readers use, with numpy's
    results and exceptions:

    - ``records["t"]``, the ``t`` column, read once when first asked
      for (:func:`read_column`) and kept, read-only, and so
      ``records["t"][i]``;
    - ``records[name][rows]``: the field *name* at *rows* -- an int, a
      numpy integer, a negative index, a slice, or an integer array of
      any shape, empty included -- shaped ``(*rows.shape, *field
      shape)`` (the field's own shape for one row), in the stored
      dtype, one ``pread`` of that field per row;
    - ``records[i]``, an int or a numpy integer: the whole record in one
      ``pread``, a read-only ``np.void``, so ``records[i][name]`` and
      ``records[i].__getitem__`` work as on a memory map;
    - ``len(records)``, ``records.shape``, ``records.dtype``.

    An unknown field raises ``ValueError``, a row out of range
    ``IndexError``, as numpy does.  Every read is charged to *throttle*
    (none: no limit) and its pages released; readahead is off on the
    descriptor, which closes with the object.
    """

    def __init__(
        self,
        path: str | Path,
        dtype: np.dtype,
        *,
        throttle: Throttle | None = None,
    ) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.shape = (_complete(self.path, self.dtype),)
        self._throttle = throttle
        self._fd = _open(self.path)
        weakref.finalize(self, os.close, self._fd)
        self._t: np.ndarray | None = None

    def __len__(self) -> int:
        return self.shape[0]

    def _at(self, row) -> int:
        """The byte offset of record *row*, bounds-checked as numpy does."""
        n = self.shape[0]
        i = int(row)
        if not -n <= i < n:
            raise IndexError(
                f"index {i} is out of bounds for axis 0 with size {n}"
            )
        return (i % n) * self.dtype.itemsize

    def _read(self, target: np.ndarray, offset: int) -> None:
        """Fill the C-contiguous *target* from byte *offset*."""
        _pread_into(
            self._fd, _bytes(target), offset, self._throttle, self.path
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "t":
                if self._t is None:
                    self._t = read_column(
                        self.path, self.dtype, "t", throttle=self._throttle
                    )
                    self._t.setflags(write=False)
                return self._t
            return _Field(self, key)
        if isinstance(key, (int, np.integer)):
            out = np.empty(1, dtype=self.dtype)
            self._read(out, self._at(key))
            out.setflags(write=False)
            return out[0]
        raise TypeError(
            f"{type(key).__name__} index: a stream stand-in takes a field "
            "name or a record number"
        )


def map_records(
    path: str | Path, dtype: np.dtype, *, throttle: Throttle | None
) -> np.memmap | CappedRecords:
    """A stream's complete records, memory-mapped or read on demand.

    ``np.memmap`` (read-only) when *throttle* is ``None``, else
    :class:`CappedRecords` charging it; the caller decides, usually
    with :func:`throttle`.  A file without a complete record is refused.
    """
    path, dtype = Path(path), np.dtype(dtype)
    n = _complete(path, dtype)
    if n == 0:
        raise ValueError(f"{path}: no complete records")
    if throttle is None:
        return np.memmap(path, dtype=dtype, mode="r", shape=(n,))
    return CappedRecords(path, dtype, throttle=throttle)


@contextlib.contextmanager
def paced(*paths: str | Path) -> Iterator[None]:
    """Charge files another library reads whole to the process's cap.

    Before the block, the files' sizes in whole pages are charged to
    the process's throttle; after it, their pages are released.  Nothing
    happens without a cap, and a path that is not a file is left to the
    library to refuse.  The library still reads each file at full speed:
    the cap holds on average, not within one file (module docstring,
    "The cap").
    """
    global _charged
    if _PROCESS.rate is None:
        yield
        return
    files = [Path(p) for p in paths if Path(p).is_file()]
    for file in files:
        _, length = _span(0, file.stat().st_size)
        _PROCESS.wait(length)
        _charged += length
    try:
        yield
    finally:
        if _FADVISE:
            for file in files:
                fd = os.open(file, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
