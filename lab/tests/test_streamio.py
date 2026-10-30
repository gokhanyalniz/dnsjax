r"""Rate-capped reads of the lab's streams (``dnsjax_lab.analysis.streamio``).

The lab's binary-stream readers read at full speed by default and at a
capped rate through this module.  On synthetic record files written
with NumPy (no solver), this pins:

1. **The throttle.**  On an injected clock: bytes at a rate take their
   time once the allowance is spent; an idle stretch carries forward
   one second and no more; a chained throttle holds to the lower rate
   and charges its parent, which therefore saves up nothing while it
   runs; a read is charged its page span.
2. **The process's cap.**  Set and lifted in MB/s, refused at zero,
   below, not finite or above
   :data:`~dnsjax_lab.analysis.streamio.MAX_READ_CAP` (a rate in bytes
   per second); ``capped`` restores what was set; ``throttle`` resolves
   a reader's ``read_cap``.
3. **``read_records``** returns the same bytes capped and uncapped for
   every range (whole, partial, empty, past the end), never a partial
   trailing record, with records below and above the chunk size.
4. **``CappedRecords`` is a memory map** on every indexing pattern the
   lab's readers use -- the ``t`` column, a field at an int, a numpy
   integer, a negative index, a slice, an integer array of any shape
   and an empty one, a whole record and its fields -- and raises
   numpy's exceptions where they do.
5. **Readahead off, pages released under a throttle.**
   ``posix_fadvise`` is called with ``POSIX_FADV_RANDOM`` at every
   ``pread`` descriptor's open, and with ``POSIX_FADV_DONTNEED`` on the
   page span of every throttled read and of no other.
6. **``map_records``** maps without a throttle, reads on demand with
   one, and refuses a file without a complete record; **``paced``**
   charges whole files only under a cap.

Usage::

    uv run python lab/tests/test_streamio.py
"""

from __future__ import annotations

import sys

from dnsjax_lab.analysis import streamio as sio

sys.stdout.reconfigure(line_buffering=True)

assert "jax" not in sys.modules, (
    "importing dnsjax_lab.analysis.streamio must not import JAX"
)

import contextlib  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

#: A record that is not a whole number of pages, nor a divisor of one,
#: so that reads straddle page boundaries: 8 + 60 + 8 = 76 bytes.
DTYPE = np.dtype([("t", "<f8"), ("a", "<f4", (3, 5)), ("b", "<i8")])
N = 37
PAGE = sio._PAGE


class _Clock:
    """A clock that a sleep advances, and the sleeps it saw."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.slept += seconds


@contextlib.contextmanager
def _fake_clock():
    clock = _Clock()
    saved = sio._clock, sio._sleep
    sio._clock, sio._sleep = clock, clock.sleep
    try:
        yield clock
    finally:
        sio._clock, sio._sleep = saved


@contextlib.contextmanager
def _stream(n: int = N, trailing: int = 20, dtype: np.dtype = DTYPE):
    """A file of *n* distinct records and a partial trailing one."""
    rec = np.zeros(n, dtype=dtype)
    rng = np.random.default_rng(7)
    for name in dtype.names:
        field = rec[name]
        rec[name] = rng.integers(-1000, 1000, size=field.shape).astype(
            field.dtype
        )
    rec["t"] = 0.5 * np.arange(n)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stream.bin"
        raw = rec.tobytes()
        path.write_bytes(raw + raw[:trailing])
        yield path, rec


def _raises(call, kind: type[BaseException]) -> type[BaseException]:
    try:
        call()
    except kind as exc:
        return type(exc)
    raise AssertionError(f"{call} raised no {kind.__name__}")


def test_throttle() -> None:
    """The rate, the one-second allowance, chains, and page spans."""
    with _fake_clock() as clock:
        limit = sio.Throttle(100.0)  # bytes per second
        limit.wait(100)
        limit.wait(50)
        # No allowance at the start: 150 bytes take 1.5 s.
        assert math.isclose(clock.slept, 1.5), clock.slept
        clock.now += 10.0  # idle: one second carries forward, no more
        before = clock.slept
        limit.wait(100)
        assert clock.slept == before
        limit.wait(100)
        assert math.isclose(clock.slept - before, 1.0), clock.slept

        # A chain holds to its lower rate, whichever end holds it.
        for child_rate, parent_rate in ((100.0, 1000.0), (1000.0, 100.0)):
            start = clock.slept
            parent = sio.Throttle(parent_rate)
            child = sio.Throttle(child_rate, parent=parent)
            for _ in range(10):
                child.wait(100)
            assert math.isclose(clock.slept - start, 10.0, rel_tol=0.11), (
                child_rate,
                clock.slept - start,
            )

        # A chained call charges its parent, which saves up nothing;
        # an unchained one leaves the parent a second's allowance.
        for chained, parent_waits in ((True, True), (False, False)):
            parent = sio.Throttle(100.0)
            child = sio.Throttle(100.0, parent=parent if chained else None)
            for _ in range(10):
                child.wait(100)
            before = clock.slept
            parent.wait(100)
            assert (clock.slept > before) == parent_waits, chained

    assert sio._span(10, 5) == (0, PAGE)
    assert sio._span(PAGE - 2, 4) == (0, 2 * PAGE)
    assert sio._span(PAGE, PAGE) == (PAGE, PAGE)
    with _stream() as (path, rec):
        # The t column charges each record's page span, and only a
        # throttled read is counted.
        before = sio.bytes_read()
        t = sio.read_column(path, DTYPE, "t")
        assert sio.bytes_read() == before
        t = sio.read_column(path, DTYPE, "t", throttle=sio.Throttle())
        assert np.array_equal(t, rec["t"])
        spans = sum(sio._span(i * DTYPE.itemsize, 8)[1] for i in range(N))
        assert sio.bytes_read() - before == spans
    print("throttle: OK")


def test_process_cap() -> None:
    """``set_read_cap`` / ``capped`` / ``throttle`` and their refusals."""
    assert sio.set_read_cap(None) is None
    assert sio.throttle() is None
    assert sio.set_read_cap(25) is None
    assert sio.throttle() is sio._PROCESS
    assert sio._PROCESS.rate == 25e6
    own = sio.throttle(5)
    assert own is not sio._PROCESS
    assert own.rate == 5e6 and own.parent is sio._PROCESS
    with sio.capped(2.5):
        assert sio._PROCESS.rate == 2.5e6
        with sio.capped(None):  # leaves the cap as it is
            assert sio._PROCESS.rate == 2.5e6
    assert sio._PROCESS.rate == 25e6
    assert sio.set_read_cap(None) == 25.0
    with sio.capped(1.0):
        assert sio.throttle() is sio._PROCESS
    assert sio.throttle() is None
    for bad in (0, -1.0, math.nan, math.inf, 25e6):
        try:
            sio.set_read_cap(bad)
        except ValueError as exc:
            assert "bytes per second" in str(exc), exc
        else:
            raise AssertionError(f"read cap {bad} was accepted")
        _raises(lambda bad=bad: sio.throttle(bad), ValueError)
    assert sio.throttle() is None  # nothing set by a refusal
    print("process cap: OK")


def test_read_records() -> None:
    """Capped and uncapped give the same records, for every range."""
    ranges = ((0, None), (3, 17), (5, 5), (30, 100), (40, None), (0, 0))
    with _stream() as (path, rec):
        for chunk in (sio._CHUNK_BYTES, 100, 500, 1):
            saved, sio._CHUNK_BYTES = sio._CHUNK_BYTES, chunk
            try:
                for start, stop in ranges:
                    want = rec[start:stop]
                    fast = sio.read_records(
                        path, DTYPE, start=start, stop=stop
                    )
                    slow = sio.read_records(
                        path,
                        DTYPE,
                        start=start,
                        stop=stop,
                        throttle=sio.Throttle(),
                    )
                    for got in (fast, slow):
                        assert got.dtype == DTYPE
                        assert got.tobytes() == want.tobytes(), (
                            chunk,
                            start,
                            stop,
                        )
            finally:
                sio._CHUNK_BYTES = saved
        _raises(lambda: sio.read_records(path, DTYPE, start=-1), ValueError)
    # A record larger than the chunk is still read whole.
    big = np.dtype([("t", "<f8"), ("u", "<f8", (3, 41, 2))])
    with _stream(n=5, dtype=big) as (path, rec):
        saved, sio._CHUNK_BYTES = sio._CHUNK_BYTES, big.itemsize // 3
        try:
            got = sio.read_records(path, big, throttle=sio.Throttle())
        finally:
            sio._CHUNK_BYTES = saved
        assert got.tobytes() == rec.tobytes()
    print("read_records: OK")


def test_capped_records_is_a_memory_map() -> None:
    """Every indexing pattern the readers use, against ``np.memmap``."""
    with _stream() as (path, rec):
        mm = np.memmap(path, dtype=DTYPE, mode="r", shape=(N,))
        cr = sio.CappedRecords(path, DTYPE, throttle=sio.Throttle())
        assert len(cr) == len(mm) == N
        assert cr.shape == mm.shape and cr.dtype == mm.dtype
        assert cr.dtype.names == mm.dtype.names
        assert np.array_equal(cr["t"], mm["t"])
        assert not cr["t"].flags.writeable
        assert cr["t"][4] == mm["t"][4]
        rows_cases = (
            3,
            np.int64(5),
            -1,
            slice(2, 9, 3),
            slice(None),
            np.array([4, 0, 4]),
            np.array([[1, 2], [3, 4]]),
            np.array([], dtype=np.int64),
            np.int64(0) + np.arange(3),
            np.array(6),
        )
        for name in ("a", "b"):
            for rows in rows_cases:
                want, got = np.asarray(mm[name][rows]), cr[name][rows]
                assert got.shape == want.shape, (name, rows)
                assert got.dtype == want.dtype, (name, rows)
                assert np.array_equal(got, want), (name, rows)
        for i in (7, np.int64(2), -3):
            got, want = cr[i], mm[i]
            assert isinstance(got, np.void) and got.dtype == DTYPE
            for name in DTYPE.names:
                assert np.array_equal(got[name], want[name])
                assert np.array_equal(
                    got.__getitem__(name), want.__getitem__(name)
                )
        # numpy's exceptions, where numpy raises them.
        for index in (
            lambda r: r["nope"],
            lambda r: r["a"][N],
            lambda r: r["a"][-N - 1],
            lambda r: r[N],
            lambda r: r["a"][np.array([0.0, 1.0])],
        ):
            try:
                index(mm)
            except (ValueError, IndexError) as exc:
                kind = type(exc)
            else:
                raise AssertionError("np.memmap raised nothing")
            assert _raises(lambda index=index: index(cr), kind) is kind
    print("CappedRecords is a memory map: OK")


def test_fadvise() -> None:
    """Readahead off at open; a throttled read's page span released."""
    if not sio._FADVISE:
        print("fadvise: skipped (no os.posix_fadvise here)")
        return
    calls: list[tuple[int, int, int, int]] = []
    real = os.posix_fadvise

    def record(fd, offset, length, advice):
        calls.append((fd, offset, length, advice))
        return real(fd, offset, length, advice)

    random_, dontneed = os.POSIX_FADV_RANDOM, os.POSIX_FADV_DONTNEED
    with _stream() as (path, rec):
        os.posix_fadvise = record
        try:
            cr = sio.CappedRecords(path, DTYPE, throttle=sio.Throttle())
            assert [c[1:] for c in calls] == [(0, 0, random_)]
            calls.clear()
            cr["a"][3]
            offset = 3 * DTYPE.itemsize + DTYPE.fields["a"][1]
            span = sio._span(offset, DTYPE.fields["a"][0].itemsize)
            assert [c[1:] for c in calls] == [(*span, dontneed)]
            calls.clear()
            sio.read_records(
                path, DTYPE, start=2, stop=9, throttle=sio.Throttle()
            )
            assert calls[0][1:] == (0, 0, random_)
            assert calls[1:] == [
                (
                    calls[1][0],
                    *sio._span(2 * DTYPE.itemsize, 7 * DTYPE.itemsize),
                    dontneed,
                )
            ]
            # Without a throttle readahead is still off, and nothing
            # is released.
            calls.clear()
            sio.read_column(path, DTYPE, "t")
            assert [c[1:] for c in calls] == [(0, 0, random_)]
            calls.clear()
            sio.read_column(path, DTYPE, "t", throttle=sio.Throttle())
            assert [c[1:] for c in calls[1:]] == [
                (*sio._span(i * DTYPE.itemsize, 8), dontneed) for i in range(N)
            ]
            calls.clear()
            sio.CappedRecords(path, DTYPE)["a"][3]
            assert [c[1:] for c in calls] == [(0, 0, random_)]
        finally:
            os.posix_fadvise = real
    print("fadvise: OK")


def test_map_records_and_paced() -> None:
    """The map/stand-in choice, an empty file, and whole-file charges."""
    with _stream() as (path, rec):
        mapped = sio.map_records(path, DTYPE, throttle=None)
        assert isinstance(mapped, np.memmap)
        stand_in = sio.map_records(path, DTYPE, throttle=sio.Throttle())
        assert isinstance(stand_in, sio.CappedRecords)
        assert np.array_equal(stand_in["b"][:], mapped["b"])
        short = path.with_name("short.bin")
        short.write_bytes(rec.tobytes()[:50])
        _raises(
            lambda: sio.map_records(short, DTYPE, throttle=None), ValueError
        )
        missing = path.with_name("missing.bin")
        before = sio.bytes_read()
        with sio.paced(path, short, missing):
            pass
        assert sio.bytes_read() == before  # no cap: nothing charged
        with sio.capped(sio.MAX_READ_CAP), sio.paced(path, short, missing):
            pass
        want = sum(sio._span(0, p.stat().st_size)[1] for p in (path, short))
        assert sio.bytes_read() - before == want
    print("map_records and paced: OK")


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for case in tests:
        case()
        print(f"  PASS  {case.__name__}")
    assert sio.throttle() is None, "a test left a cap set"
    assert "jax" not in sys.modules, "a test pulled JAX in"
    print(f"\nAll {len(tests)} tests passed.")
