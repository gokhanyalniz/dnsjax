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
   touched, a peer process recording only its own exception -- plus
   the stop poll and :func:`~dnsjax.run_status.loop_outcome`.
2. **One process** (``dnsjax``, no launcher): a natural finish;
   ``rm RUNNING`` and ``SIGUSR1`` mid-run, each ending ``STOPPED`` with
   exit 0 and a final snapshot whose time the last ``stats.dat`` row
   carries; ``SIGTERM`` ending ``TERMINATED`` with exit 143; a corrector
   that cannot converge ending ``TERMINATED`` with exit 4; a second
   launch into a live run's directory refused, the live run untouched.
3. **Two processes** (``mpirun -np 2``): ``rm RUNNING``; ``SIGUSR1`` to
   rank 1 alone, which only an agreed stop survives (a rank leaving
   the loop alone hangs both); ``SIGUSR1`` to ``mpirun``, which must
   forward it.  Each must end with both ranks shut down.

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
    agree = rs._agree
    rs._agree = lambda flag: flag  # two processes, without JAX
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
        rs._agree = agree


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


ONE_PROCESS = [
    ("natural", live_natural),
    ("remove", live_stop(1, "remove")),
    ("usr1", live_stop(1, "launched")),
    ("sigterm", live_sigterm),
    ("corrector", live_corrector),
    ("refusal", live_refusal),
]
TWO_PROCESSES = [
    ("mpi-remove", live_stop(2, "remove")),
    ("mpi-usr1-rank1", live_stop(2, "rank1")),
    ("mpi-usr1-mpirun", live_stop(2, "launched")),
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
