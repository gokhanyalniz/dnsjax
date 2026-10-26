r"""The run status files and the graceful stop request.

A run keeps ``RUNNING`` in its directory while it lives and leaves
``FINISHED``, ``STOPPED`` or ``TERMINATED`` behind; deleting
``RUNNING``, or a ``SIGUSR1`` to any of its processes, asks it to stop
gracefully (:mod:`dnsjax.run_status`).  This script pins that contract
in three tiers (the live-run machinery: ``tests/_run_status.py``):

1. **Units** (no solver launch, no JAX): every transition of
   :class:`~dnsjax.run_status.RunStatus` -- the end file and exit code
   of each way out, the note and snapshot lines, a stale ``RUNNING``
   replaced (dead pid, another host), a live one refused with nothing
   touched, a peer process recording only its own exception, the
   teardown bound armed on exactly the multi-process ``TERMINATED``
   ends, and, in a child process blocked at exit in native code, that
   bound exiting with status 1 and writing nothing -- plus the stop
   poll and :func:`~dnsjax.run_status.loop_outcome`.
2. **One process** (``dnsjax``, no launcher): a natural finish;
   ``rm RUNNING`` and ``SIGUSR1`` mid-run, each ending ``STOPPED`` with
   exit 0 and a final snapshot whose time the last ``stats.dat`` row
   carries; ``SIGTERM`` mid-run and during setup (exit 143), a resume
   at the other precision (one ``dnsjax: error:`` line, exit 1) and a
   corrector that cannot converge (exit 4), each ending
   ``TERMINATED``; a second launch into a live run's directory
   refused, the live run untouched.
3. **Two processes** (``mpirun -np 2``): ``rm RUNNING``; ``SIGUSR1`` to
   rank 1 alone, which only an agreed stop survives (a rank leaving
   the loop alone hangs both); ``SIGUSR1`` to ``mpirun``, which must
   forward it.  Each must end with both ranks shut down.  And an
   ``OSError`` on rank 1 alone mid-write (``tests/_peer_error.py``),
   which must end the launch within the teardown bound rather than
   leave rank 0 waiting in the snapshot barrier.

Run as a script::

    uv run python tests/test_run_status.py              # all three
    uv run python tests/test_run_status.py --unit-only  # tier 1
    uv run python tests/test_run_status.py --no-mpi     # tiers 1 and 2
    uv run python tests/test_run_status.py --mpi-only   # tier 3
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _run_status import (  # noqa: E402
    LONG,
    SOLVER,
    Run,
    end_files,
    fields,
    live_natural,
    live_stop,
    run_cases,
    solver,
)

from dnsjax import run_status as rs  # noqa: E402

# --- Tier 1: units ----------------------------------------------------------


def _expect_exit(d: Path, exc: BaseException) -> None:
    """Enter a status in *d*, raise *exc* inside, check it re-raises."""
    try:
        with rs.RunStatus(directory=d):
            raise exc
    except type(exc):
        return
    raise AssertionError(f"{type(exc).__name__} was swallowed")


def unit_outcomes(d: Path) -> None:
    """Each driver outcome writes its file; RUNNING goes; lines hold."""
    for outcome in (
        rs.Outcome(rs.FINISHED, "stop.max_sim_time reached"),
        rs.Outcome(rs.STOPPED, "stop requested (RUNNING removed)"),
        rs.Outcome(rs.TERMINATED, "corrector failed", rs.EXIT_CORRECTOR),
    ):
        with rs.RunStatus(directory=d) as status:
            running = fields(d / rs.RUNNING)
            assert running["pid"] == str(os.getpid()), running
            assert running["host"] == socket.gethostname(), running
            status.snapshot_written("state00003.tar", 1.5, 150)
            status.outcome = outcome
        assert end_files(d) == [outcome.status], end_files(d)
        got = fields(d / outcome.status)
        assert got["reason"] == outcome.reason, got
        assert got["exit"] == str(outcome.exit_code), got
        assert got["started"] == running["started"], got
        assert got["snapshot"].startswith("state00003.tar (t = 1.5"), got
        assert outcome.status.lower() in got, got


def unit_exceptions(d: Path) -> None:
    """Every exception leaving the run is TERMINATED, and re-raised."""
    cases: list[tuple[BaseException, str | None, str, str]] = [
        (SystemExit(3), "non-finite E' at t = 1", "non-finite E'", "3"),
        (SystemExit(143), None, "signal 15 (SIGTERM)", "143"),
        (SystemExit(1), None, "exit code 1", "1"),
        (
            SystemExit("dnsjax: error: bad\nmore"),
            None,
            "dnsjax: error: bad",
            "1",
        ),
        (RuntimeError("boom\ntrace"), None, "RuntimeError: boom", "1"),
        (KeyboardInterrupt(), None, "signal 2 (SIGINT)", "130"),
    ]
    for exc, note, reason, code in cases:

        def body(exc=exc, note=note):
            with rs.RunStatus(directory=d) as status:
                if note is not None:
                    status.note(note)
                raise exc

        try:
            body()
        except BaseException as caught:
            assert caught is exc, caught
        else:
            raise AssertionError(f"{exc!r} was swallowed")
        assert end_files(d) == [rs.TERMINATED], (exc, end_files(d))
        got = fields(d / rs.TERMINATED)
        assert got["reason"].startswith(reason), (exc, got)
        assert got["exit"] == code, (exc, got)
        assert "process" not in got, got  # one process: no process line
    # SystemExit(0) is a clean end.
    with rs.RunStatus(directory=d):
        pass
    assert end_files(d) == [rs.FINISHED], end_files(d)


def unit_stale(d: Path) -> None:
    """A dead or foreign RUNNING is replaced, old end files cleared."""
    dead = subprocess.Popen(["true"])
    dead.wait()
    for host, pid in (
        (socket.gethostname(), dead.pid),
        ("another-host.invalid", os.getpid() + 0),
    ):
        (d / rs.RUNNING).write_text(
            f"started: then\nhost: {host}\npid: {pid}\n"
        )
        (d / rs.FINISHED).write_text("finished: earlier\n")
        (d / rs.TERMINATED).write_text("terminated: earlier\n")
        with rs.RunStatus(directory=d) as status:
            assert end_files(d) == [rs.RUNNING], end_files(d)
            assert fields(d / rs.RUNNING)["pid"] == str(os.getpid())
            status.outcome = rs.Outcome(rs.FINISHED, "ok")
        assert end_files(d) == [rs.FINISHED], end_files(d)


def unit_live_refusal(d: Path) -> None:
    """A RUNNING naming a live local process refuses, touching nothing."""
    with subprocess.Popen(["sleep", "60"]) as live:
        try:
            boot = rs._boot_id()
            text = (
                f"started: now\nhost: {socket.gethostname()}\n"
                f"pid: {live.pid}\n" + (f"boot: {boot}\n" if boot else "")
            )
            (d / rs.RUNNING).write_text(text)
            (d / rs.FINISHED).write_text("finished: earlier\n")
            try:
                with rs.RunStatus(directory=d):
                    raise AssertionError("entered beside a live run")
            except SystemExit as exc:
                assert "names a live process" in str(exc.code), exc.code
            assert (d / rs.RUNNING).read_text() == text, "RUNNING touched"
            assert end_files(d) == [rs.FINISHED, rs.RUNNING], end_files(d)
            # Another boot of this host: the pid is someone else's.
            if boot:
                (d / rs.RUNNING).write_text(
                    text.replace(f"boot: {boot}", "boot: another-boot")
                )
                with rs.RunStatus(directory=d) as status:
                    status.outcome = rs.Outcome(rs.FINISHED, "ok")
        finally:
            live.kill()
    for name in (rs.RUNNING, *rs.END_FILES):
        (d / name).unlink(missing_ok=True)


def unit_peer(d: Path) -> None:
    """A peer appends its own exception only; process 0 owns RUNNING."""
    agree, bound = rs._agree, rs._bound_teardown
    rs._agree = lambda flag: flag  # two processes, without JAX
    # A real bound would end this script (unit_teardown_bound tests it).
    rs._bound_teardown = lambda: None
    try:
        main = rs.RunStatus(0, 2, directory=d).__enter__()
        peer = rs.RunStatus(1, 2, directory=d).__enter__()
        # A peer's SystemExit (a shared refusal, a signal) records nothing.
        assert peer.__exit__(SystemExit, SystemExit(143), None) is False
        assert end_files(d) == [rs.RUNNING], end_files(d)
        peer = rs.RunStatus(1, 2, directory=d).__enter__()
        err = RuntimeError("RESOURCE_EXHAUSTED on rank 1")
        peer.__exit__(RuntimeError, err, None)
        assert end_files(d) == [rs.RUNNING, rs.TERMINATED], end_files(d)
        main.__exit__(SystemExit, SystemExit(143), None)
        assert end_files(d) == [rs.TERMINATED], end_files(d)
        blocks = (d / rs.TERMINATED).read_text().split("\n\n")
        assert len(blocks) == 2, blocks
        first = fields(d / rs.TERMINATED)
        assert first["reason"].startswith("RuntimeError: RESOURCE"), first
        assert first["process"].startswith("1 of 2 on "), first
        assert "process: 0 of 2 on " in blocks[1], blocks[1]
        assert "signal 15 (SIGTERM)" in blocks[1], blocks[1]
    finally:
        rs._agree, rs._bound_teardown = agree, bound


def unit_teardown_bound(d: Path) -> None:
    """A TERMINATED end bounds the teardown on several processes only."""
    agree, bound = rs._agree, rs._bound_teardown
    armed: list[int] = []
    rs._agree = lambda flag: flag  # two processes, without JAX
    rs._bound_teardown = lambda: armed.append(1)
    try:
        cases = [
            # (process index, count, exception, outcome, armed)
            (0, 1, RuntimeError("one process"), None, False),
            (1, 2, OSError(5, "Input/output error"), None, True),
            (1, 2, SystemExit(143), None, True),
            (0, 2, SystemExit(3), None, True),
            (0, 2, None, rs.Outcome(rs.TERMINATED, "corrector", 4), True),
            (0, 2, None, rs.Outcome(rs.STOPPED, "requested"), False),
            (0, 2, None, None, False),
        ]
        for index, count, exc, outcome, want in cases:
            armed.clear()
            status = rs.RunStatus(index, count, directory=d).__enter__()
            status.outcome = outcome
            status.__exit__(type(exc) if exc else None, exc, None)
            assert bool(armed) == want, (index, count, exc, outcome)
            for name in (rs.RUNNING, *rs.END_FILES):
                (d / name).unlink(missing_ok=True)
    finally:
        rs._agree, rs._bound_teardown = agree, bound


# A child that arms the bound, then blocks at exit in native code
# holding the GIL (``ctypes.PyDLL`` keeps it), as a teardown may.
_BLOCKED_EXIT = """
import atexit, ctypes
from dnsjax import run_status as rs
rs.TEARDOWN_GRACE = 1
rs._bound_teardown()
atexit.register(ctypes.PyDLL(None).sleep, 60)
"""


def unit_teardown_exit(d: Path) -> None:
    """The armed bound exits a blocked teardown, status 1, writing nothing."""
    start = time.monotonic()
    child = subprocess.run(
        [sys.executable, "-c", _BLOCKED_EXIT],
        capture_output=True,
        text=True,
        timeout=60,
    )
    took = time.monotonic() - start
    assert child.returncode == 1, f"exit {child.returncode}"
    assert child.stdout == child.stderr == "", (child.stdout, child.stderr)
    assert took < 30, f"exited after {took:.1f} s"


def unit_poll(d: Path) -> None:
    """The poll: RUNNING removed, SIGUSR1, the clock; a peer never stats."""

    def each(flags):
        return tuple(bool(f) for f in flags)

    rs.install_stop_signal()
    with rs.RunStatus(directory=d) as status:
        assert status.stop_flags(False, each) == (False, None)
        assert status.stop_flags(True, each) == (True, None)
        os.kill(os.getpid(), signal.SIGUSR1)
        try:
            assert status.stop_flags(False, each) == (False, "SIGUSR1")
        finally:
            rs._stop_signalled = False
        (d / rs.RUNNING).unlink()
        assert status.stop_flags(False, each) == (False, "RUNNING removed")
        status.outcome = rs.Outcome(rs.STOPPED, "requested")
    assert end_files(d) == [rs.STOPPED], end_files(d)
    # A peer reports only what the gather brings back.
    peer = rs.RunStatus(1, 2, directory=d)
    seen: list[tuple] = []
    assert peer.stop_flags(False, lambda f: seen.append(f) or each(f)) == (
        False,
        None,
    )
    assert seen == [(False, False, False)], seen
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)


def unit_loop_outcome(d: Path) -> None:
    """Precedence: corrector failure, then a request, then the criteria."""
    lo = rs.loop_outcome
    kw = dict(request=None, out_of_time=False, laminarized=False)
    got = lo(1.0, 10, corrector_error=None, **kw)
    assert got.status == rs.FINISHED and "max_sim_time" in got.reason, got
    got = lo(1.0, 10, corrector_error=None, **{**kw, "out_of_time": True})
    assert "max_wall_time" in got.reason, got
    got = lo(1.0, 10, corrector_error=None, **{**kw, "laminarized": True})
    assert got.reason.startswith("laminarized"), got
    got = lo(
        1.0,
        10,
        corrector_error=None,
        request="SIGUSR1",
        out_of_time=True,
        laminarized=False,
    )
    assert got == rs.Outcome(
        rs.STOPPED, "stop requested (SIGUSR1) at t = 1.000000e+00, it = 10"
    ), got
    got = lo(1.0, 10, corrector_error=0.5, **{**kw, "request": "SIGUSR1"})
    assert (got.status, got.exit_code) == (rs.TERMINATED, 4), got


UNITS: list[Callable[[Path], None]] = [
    unit_outcomes,
    unit_exceptions,
    unit_stale,
    unit_live_refusal,
    unit_peer,
    unit_teardown_bound,
    unit_teardown_exit,
    unit_poll,
    unit_loop_outcome,
]


# --- Tiers 2 and 3: live runs -----------------------------------------------


def live_sigterm(d: Path, ctx: dict) -> None:
    """SIGTERM stays immediate: exit 143, TERMINATED, no RUNNING."""
    run = Run(solver(1, [*SOLVER, *LONG]), d)
    run.wait_for("First iteration over")
    run.proc.send_signal(signal.SIGTERM)
    rc = run.finish()
    assert rc == 128 + signal.SIGTERM, f"exit {rc}"
    assert end_files(d) == [rs.TERMINATED], end_files(d)
    got = fields(d / rs.TERMINATED)
    assert got["reason"] == "signal 15 (SIGTERM)", got
    assert got["exit"] == "143", got


def live_sigterm_setup(d: Path, ctx: dict) -> None:
    """SIGTERM during setup, before any stream exists: exit 143, TERMINATED.

    The solver claims ``RUNNING`` as soon as its JAX runtime is
    configured, minutes before the first step at production size; the
    scheduler's SIGTERM in that window must still end in ``TERMINATED``
    rather than leave ``RUNNING`` behind.  A larger box makes the window
    long enough for the signal to land in it, which the case checks.
    """
    big = list(SOLVER)
    for name, value in (
        ("--res.nx", "48"),
        ("--res.ny", "97"),
        ("--res.nz", "48"),
    ):
        big[big.index(name) + 1] = value
    run = Run(solver(1, [*big, *LONG]), d)
    deadline = time.monotonic() + 600
    while not (d / rs.RUNNING).exists():
        if run.proc.poll() is not None:
            raise AssertionError(f"exit {run.proc.returncode} before RUNNING")
        if time.monotonic() > deadline:
            run.finish(0)
            raise AssertionError("no RUNNING within 600 s")
        time.sleep(0.05)
    seen = time.monotonic()
    run.proc.send_signal(signal.SIGTERM)
    print(f"RUNNING after {seen - (deadline - 600):.2f} s; SIGTERM sent")
    rc = run.finish()
    assert "First iteration over" not in run.stdout, (
        "the signal landed after setup: raise the box"
    )
    assert rc == 128 + signal.SIGTERM, f"exit {rc}"
    assert end_files(d) == [rs.TERMINATED], end_files(d)
    got = fields(d / rs.TERMINATED)
    assert got["reason"] == "signal 15 (SIGTERM)", got


def live_mismatch(d: Path, ctx: dict) -> None:
    """A precision mismatch on resume: one refusal line, exit 1, TERMINATED.

    ``res.double_precision`` is never compared by the parameter layers
    (a resume may run at either precision), so the snapshot's own
    check is the first to refuse; it must do so the way every other
    resume refusal does, not with a traceback.
    """
    args = [
        *SOLVER,
        "--init.snapshot",
        str(ctx["parent"]),
        "--res.double_precision",
        "False",
        "--stop.max_sim_time",
        "0.05",
    ]
    run = Run(solver(1, args), d)
    rc = run.finish()
    assert rc == 1, f"exit {rc}"
    assert "Traceback" not in run.stderr, "a traceback, not a refusal"
    assert "dnsjax: error:" in run.stderr and "precision" in run.stderr, (
        run.stderr[-300:]
    )
    assert end_files(d) == [rs.TERMINATED], end_files(d)
    got = fields(d / rs.TERMINATED)
    assert got["reason"].startswith("dnsjax: error:"), got
    assert got["exit"] == "1", got


def live_corrector(d: Path, ctx: dict) -> None:
    """A corrector that cannot converge: final snapshot, exit 4."""
    args = [*SOLVER, *LONG, "--step.corrector_tolerance", "1e-300"]
    run = Run(solver(1, args), d)
    rc = run.finish()
    assert rc == rs.EXIT_CORRECTOR, f"exit {rc}"
    assert end_files(d) == [rs.TERMINATED], end_files(d)
    got = fields(d / rs.TERMINATED)
    assert got["reason"].startswith("corrector failed to converge"), got
    assert got["exit"] == "4", got
    name = got["snapshot"].split(" ")[0]
    assert name != "state00000.tar" and (d / name).is_file(), got


def live_refusal(d: Path, ctx: dict) -> None:
    """A second launch into a live run's directory is refused."""
    first = Run(solver(1, [*SOLVER, *LONG]), d)
    first.wait_for("First iteration over")
    before = (d / rs.RUNNING).read_text()
    second = Run(solver(1, [*SOLVER, *LONG]), d)
    rc = second.finish()
    try:
        assert rc == 1, f"second launch exited {rc}"
        assert "names a live process" in second.stderr, "no refusal"
        assert (d / rs.RUNNING).read_text() == before, "RUNNING touched"
        assert end_files(d) == [rs.RUNNING], end_files(d)
    finally:
        (d / rs.RUNNING).unlink(missing_ok=True)
        rc = first.finish()
    assert rc == 0, f"the live run exited {rc}"
    assert end_files(d) == [rs.STOPPED], end_files(d)


def live_peer_error(d: Path, ctx: dict) -> None:
    """An I/O error on rank 1 alone ends both ranks, not a hang.

    Rank 1 raises in the initial snapshot's write while rank 0 waits
    in its barrier (``tests/_peer_error.py``): rank 1 records its
    error, then exits within the shortened teardown bound, writing
    nothing, and ``mpirun`` ends rank 0.
    """
    shutil.copy(ctx["pids"], d / "pids.sh")
    helper = Path(__file__).resolve().parent / "_peer_error.py"
    cmd = ["mpirun", "-np", "2", "./pids.sh", sys.executable, str(helper)]
    run = Run([*cmd, "--dist.np1", "2", *SOLVER, *LONG], d)
    rc = run.finish(timeout=120)
    assert rc is not None, "hung after rank 1's error"
    assert rc != 0, f"exit {rc}"
    got = fields(d / rs.TERMINATED)
    assert got["reason"].startswith("OSError: [Errno 5]"), got
    assert got["process"].startswith("1 of 2 on "), got
    assert "Timeout (" not in run.stderr, "the watchdog wrote to stderr"


ONE_PROCESS = [
    ("natural", live_natural),
    ("remove", live_stop(1, "remove")),
    ("usr1", live_stop(1, "launched")),
    ("sigterm", live_sigterm),
    ("sigterm-setup", live_sigterm_setup),
    ("mismatch", live_mismatch),
    ("corrector", live_corrector),
    ("refusal", live_refusal),
]
TWO_PROCESSES = [
    ("mpi-remove", live_stop(2, "remove")),
    ("mpi-usr1-rank1", live_stop(2, "rank1")),
    ("mpi-usr1-mpirun", live_stop(2, "launched")),
    ("mpi-peer-error", live_peer_error),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tier = ap.add_mutually_exclusive_group()
    tier.add_argument("--unit-only", action="store_true")
    tier.add_argument("--no-mpi", action="store_true")
    tier.add_argument("--mpi-only", action="store_true")
    cli = ap.parse_args()

    cases: list[tuple[str, Callable]] = []
    if not cli.mpi_only:
        cases += [(f.__name__, f) for f in UNITS]
    if not cli.unit_only:
        if not cli.mpi_only:
            cases += ONE_PROCESS
        if not cli.no_mpi:
            if shutil.which("mpirun") is None:
                print("mpirun is needed for the two-process tier")
                return 1
            cases += TWO_PROCESSES
    return run_cases(cases)


if __name__ == "__main__":
    sys.exit(main())
