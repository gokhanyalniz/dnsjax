r"""The run status files and the graceful stop request.

A run keeps ``RUNNING`` in its directory while it lives and leaves
``FINISHED``, ``STOPPED`` or ``TERMINATED`` behind; deleting
``RUNNING``, or a ``SIGUSR1`` to any of its processes, asks it to stop
gracefully (:mod:`dnsjax.run_status`).  This script pins that contract
in three tiers:

1. **Units** (no solver launch, no JAX): every transition of
   :class:`~dnsjax.run_status.RunStatus` -- the end file and exit code
   of each way out, the note and snapshot lines, a stale ``RUNNING``
   replaced (dead pid, another host), a live one refused with nothing
   touched, a peer process recording only its own exception -- plus
   the stop poll and :func:`~dnsjax.run_status.loop_outcome`.
2. **One process** (``dnsjax`` and ``dnsjax-twin``, no launcher):
   a natural finish; ``rm RUNNING`` and ``SIGUSR1`` mid-run, each
   ending ``STOPPED`` with exit 0 and a final snapshot (a final pair
   for the twin) whose time the last ``stats.dat`` row carries;
   ``SIGTERM`` ending ``TERMINATED`` with exit 143; a corrector that
   cannot converge ending ``TERMINATED`` with exit 4; a second launch
   into a live run's directory refused, the live run untouched.
3. **Two processes** (``mpirun -np 2``): ``rm RUNNING``; ``SIGUSR1`` to
   rank 1 alone, which only an agreed stop survives (a rank leaving
   the loop alone hangs both); ``SIGUSR1`` to ``mpirun``, which must
   forward it; the twin's ``rm RUNNING``.  Each must end with both
   ranks shut down.

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
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import _pump, report  # noqa: E402

from dnsjax import run_status as rs  # noqa: E402

_BIN = Path(sys.executable).parent

# A box small enough that a step is milliseconds, with no horizon and
# no periodic snapshot: only the stop under test ends the run, and the
# final snapshot is the stop's own.
_SOLVER = [
    "--phys.system",
    "plane-couette",
    "--phys.re",
    "400",
    "--res.nx",
    "16",
    "--res.ny",
    "33",
    "--res.nz",
    "16",
    "--init.random_seed",
    "1",
    "--stop.check_laminarization",
    "False",
    "--outs.it_snapshot",
    "0",
    "--outs.it_stats",
    "1",
]
_LONG = ["--stop.max_sim_time", "1e6"]

# Each rank records its pid (``exec`` keeps it), so one rank alone can
# be signalled.
_PIDS = """#!/bin/sh
echo $$ > "pid.${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"
exec "$@"
"""


# --- Tier 1: units ----------------------------------------------------------


def _fields(path: Path) -> dict[str, str]:
    return rs.read_fields(path)


def _end_files(d: Path) -> list[str]:
    return sorted(n for n in (rs.RUNNING, *rs.END_FILES) if (d / n).exists())


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
            running = _fields(d / rs.RUNNING)
            assert running["pid"] == str(os.getpid()), running
            assert running["host"] == socket.gethostname(), running
            status.snapshot_written("state00003.tar", 1.5, 150)
            status.outcome = outcome
        assert _end_files(d) == [outcome.status], _end_files(d)
        got = _fields(d / outcome.status)
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
        assert _end_files(d) == [rs.TERMINATED], (exc, _end_files(d))
        got = _fields(d / rs.TERMINATED)
        assert got["reason"].startswith(reason), (exc, got)
        assert got["exit"] == code, (exc, got)
        assert "process" not in got, got  # one process: no process line
    # SystemExit(0) is a clean end.
    with rs.RunStatus(directory=d):
        pass
    assert _end_files(d) == [rs.FINISHED], _end_files(d)


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
            assert _end_files(d) == [rs.RUNNING], _end_files(d)
            assert _fields(d / rs.RUNNING)["pid"] == str(os.getpid())
            status.outcome = rs.Outcome(rs.FINISHED, "ok")
        assert _end_files(d) == [rs.FINISHED], _end_files(d)


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
            assert _end_files(d) == [rs.FINISHED, rs.RUNNING], _end_files(d)
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
        assert _end_files(d) == [rs.RUNNING], _end_files(d)
        peer = rs.RunStatus(1, 2, directory=d).__enter__()
        err = RuntimeError("RESOURCE_EXHAUSTED on rank 1")
        peer.__exit__(RuntimeError, err, None)
        assert _end_files(d) == [rs.RUNNING, rs.TERMINATED], _end_files(d)
        main.__exit__(SystemExit, SystemExit(143), None)
        assert _end_files(d) == [rs.TERMINATED], _end_files(d)
        blocks = (d / rs.TERMINATED).read_text().split("\n\n")
        assert len(blocks) == 2, blocks
        first = _fields(d / rs.TERMINATED)
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
    assert _end_files(d) == [rs.STOPPED], _end_files(d)
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


class _Run:
    """A launched run whose output is watched while it runs."""

    def __init__(self, cmd: list[str], cwd: Path) -> None:
        env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
        env.update(
            PYTHONUNBUFFERED="1", NO_COLOR="1", DNSJAX_QUIET_STARTUP="1"
        )
        print("+", " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        self.out: list[str] = []
        self.err: list[str] = []
        self.pumps = [
            threading.Thread(target=_pump, args=(src, sink, acc), daemon=True)
            for src, sink, acc in (
                (self.proc.stdout, sys.stdout, self.out),
                (self.proc.stderr, sys.stderr, self.err),
            )
        ]
        for t in self.pumps:
            t.start()

    @property
    def stdout(self) -> str:
        return "".join(self.out)

    @property
    def stderr(self) -> str:
        return "".join(self.err)

    def wait_for(self, text: str, timeout: float = 600) -> None:
        """Block until *text* is in stdout; raise if the run ends first."""
        deadline = time.monotonic() + timeout
        while text not in self.stdout:
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"exited {self.proc.returncode} before {text!r}"
                )
            if time.monotonic() > deadline:
                self.finish(0)
                raise AssertionError(f"no {text!r} within {timeout} s")
            time.sleep(0.2)

    def finish(self, timeout: float = 300) -> int | None:
        """Wait for the exit (``None``: hung, then SIGTERM, then killed)."""
        try:
            rc: int | None = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # SIGTERM first: mpirun passes it on to its ranks, where a
            # SIGKILL of mpirun alone can orphan them.
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            rc = None
        for t in self.pumps:
            t.join(timeout=5)
        return rc


def _solver(np: int, args: list[str], exe: str = "dnsjax") -> list[str]:
    base = [str(_BIN / exe)]
    if np > 1:
        base = ["mpirun", "-np", str(np), "./pids.sh", *base]
        args = ["--dist.np1", str(np), *args]
    return [*base, *args]


def _check_stopped(
    d: Path, run: _Run, rc: int | None, request: str, np: int, twin: bool
) -> None:
    """Check a graceful stop: exit 0 and ``STOPPED``, a final snapshot
    the last ``stats.dat`` row reaches, every process shut down."""
    if rc is None:
        raise AssertionError("hung after the stop request")
    if rc != 0:
        raise AssertionError(f"exit {rc} after the stop request")
    if _end_files(d) != [rs.STOPPED]:
        raise AssertionError(f"status files {_end_files(d)}")
    got = _fields(d / rs.STOPPED)
    if f"stop requested ({request})" not in got["reason"]:
        raise AssertionError(f"reason {got['reason']!r}")
    snap = got.get("snapshot", "")
    name = snap.split(" ")[0]
    if not name or name == "state00000.tar" or not (d / name).is_file():
        raise AssertionError(f"no final snapshot ({snap!r})")
    if twin and not (d / name.replace(".tar", "_twin.tar")).is_file():
        raise AssertionError(f"no partner of {name}")
    t_snap = float(snap.split("t = ")[1].split(",")[0])
    rows = [
        ln
        for ln in (d / "stats.dat").read_text().splitlines()
        if not ln.startswith("#")
    ]
    t_last = float(rows[-1].split()[0])
    if abs(t_last - t_snap) > 1e-6 * max(1.0, abs(t_snap)):
        raise AssertionError(
            f"stats.dat ends at t = {t_last}, the snapshot is at {t_snap}"
        )
    if f"Stop requested ({request})" not in run.stdout:
        raise AssertionError("no 'Stop requested' line")
    shutdowns = run.stderr.count("Shutdown at")
    if shutdowns != np:
        raise AssertionError(f"{shutdowns} of {np} processes shut down")


def live_natural(d: Path, ctx: dict) -> None:
    """A run reaching its horizon leaves FINISHED and no RUNNING."""
    run = _Run(_solver(1, [*_SOLVER, "--stop.max_sim_time", "0.05"]), d)
    rc = run.finish()
    assert rc == 0, f"exit {rc}"
    assert _end_files(d) == [rs.FINISHED], _end_files(d)
    got = _fields(d / rs.FINISHED)
    assert "stop.max_sim_time reached" in got["reason"], got
    assert got["exit"] == "0", got
    assert got["snapshot"].startswith("state00001.tar"), got
    ctx["parent"] = d / "state00000.tar"


def _live_stop(np: int, how: str, twin: bool = False):
    def case(d: Path, ctx: dict) -> None:
        if twin:
            args = [
                "--init.snapshot",
                str(ctx["parent"]),
                "--twin.e0",
                "1e-6",
                "--twin.seed",
                "3",
                "--stop.check_laminarization",
                "False",
                "--outs.it_snapshot",
                "0",
                "--outs.it_stats",
                "1",
                *_LONG,
            ]
            cmd = _solver(np, args, "dnsjax-twin")
        else:
            cmd = _solver(np, [*_SOLVER, *_LONG])
        if np > 1:
            shutil.copy(ctx["pids"], d / "pids.sh")
        run = _Run(cmd, d)
        run.wait_for("First iteration over")
        if how == "remove":
            (d / rs.RUNNING).unlink()
            request = "RUNNING removed"
        elif how == "rank1":
            pid = int((d / "pid.1").read_text())
            os.kill(pid, signal.SIGUSR1)
            request = "SIGUSR1"
        else:  # the launched process: the solver, or mpirun
            run.proc.send_signal(signal.SIGUSR1)
            request = "SIGUSR1"
        _check_stopped(d, run, run.finish(), request, np, twin)

    return case


def live_sigterm(d: Path, ctx: dict) -> None:
    """SIGTERM stays immediate: exit 143, TERMINATED, no RUNNING."""
    run = _Run(_solver(1, [*_SOLVER, *_LONG]), d)
    run.wait_for("First iteration over")
    run.proc.send_signal(signal.SIGTERM)
    rc = run.finish()
    assert rc == 128 + signal.SIGTERM, f"exit {rc}"
    assert _end_files(d) == [rs.TERMINATED], _end_files(d)
    got = _fields(d / rs.TERMINATED)
    assert got["reason"] == "signal 15 (SIGTERM)", got
    assert got["exit"] == "143", got


def live_corrector(d: Path, ctx: dict) -> None:
    """A corrector that cannot converge: final snapshot, exit 4."""
    args = [*_SOLVER, *_LONG, "--step.corrector_tolerance", "1e-300"]
    run = _Run(_solver(1, args), d)
    rc = run.finish()
    assert rc == rs.EXIT_CORRECTOR, f"exit {rc}"
    assert _end_files(d) == [rs.TERMINATED], _end_files(d)
    got = _fields(d / rs.TERMINATED)
    assert got["reason"].startswith("corrector failed to converge"), got
    assert got["exit"] == "4", got
    name = got["snapshot"].split(" ")[0]
    assert name != "state00000.tar" and (d / name).is_file(), got


def live_refusal(d: Path, ctx: dict) -> None:
    """A second launch into a live run's directory is refused."""
    first = _Run(_solver(1, [*_SOLVER, *_LONG]), d)
    first.wait_for("First iteration over")
    before = (d / rs.RUNNING).read_text()
    second = _Run(_solver(1, [*_SOLVER, *_LONG]), d)
    rc = second.finish()
    try:
        assert rc == 1, f"second launch exited {rc}"
        assert "names a live process" in second.stderr, "no refusal"
        assert (d / rs.RUNNING).read_text() == before, "RUNNING touched"
        assert _end_files(d) == [rs.RUNNING], _end_files(d)
    finally:
        (d / rs.RUNNING).unlink(missing_ok=True)
        rc = first.finish()
    assert rc == 0, f"the live run exited {rc}"
    assert _end_files(d) == [rs.STOPPED], _end_files(d)


ONE_PROCESS = [
    ("natural", live_natural),
    ("remove", _live_stop(1, "remove")),
    ("usr1", _live_stop(1, "launched")),
    ("sigterm", live_sigterm),
    ("corrector", live_corrector),
    ("refusal", live_refusal),
    ("twin-remove", _live_stop(1, "remove", twin=True)),
]
TWO_PROCESSES = [
    ("mpi-remove", _live_stop(2, "remove")),
    ("mpi-usr1-rank1", _live_stop(2, "rank1")),
    ("mpi-usr1-mpirun", _live_stop(2, "launched")),
    ("mpi-twin-remove", _live_stop(2, "remove", twin=True)),
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
            if cli.mpi_only:
                # The twin's parent comes from the natural run.
                cases.append(("natural", live_natural))
            cases += TWO_PROCESSES

    passed, failures = 0, []
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_run_status_"))
    ctx: dict = {"pids": tmp / "pids.sh"}
    ctx["pids"].write_text(_PIDS)
    ctx["pids"].chmod(ctx["pids"].stat().st_mode | stat.S_IXUSR)
    try:
        for name, case in cases:
            d = tmp / name
            d.mkdir()
            try:
                if name.startswith("unit_"):
                    case(d)
                else:
                    case(d, ctx)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL  {name}: {exc}")
                failures.append((name, str(exc)))
            else:
                print(f"  PASS  {name}")
                passed += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return report(passed, failures)


if __name__ == "__main__":
    sys.exit(main())
