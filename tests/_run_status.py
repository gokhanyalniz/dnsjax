r"""Live runs of the run-status tests, core and lab.

The launched-run machinery both suites' ``test_run_status.py`` share
(the contract they pin: :mod:`dnsjax.run_status`): :class:`Run` watches
a launched run's output while it runs, :func:`check_stopped` judges a
graceful stop, :func:`live_natural` is the natural finish whose initial
snapshot a case of another entry point may start from, :func:`live_stop`
builds a stop case, and :func:`run_cases` runs a list of cases and
reports.
"""

from __future__ import annotations

import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from _live import _pump, report

from dnsjax import run_status as rs

BIN = Path(sys.executable).parent

# A box small enough that a step is milliseconds, with no horizon and
# no periodic snapshot: only the stop under test ends the run, and the
# final snapshot is the stop's own.
SOLVER = [
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
LONG = ["--stop.max_sim_time", "1e6"]

# Each rank records its pid (``exec`` keeps it), so one rank alone can
# be signalled.
PIDS = """#!/bin/sh
echo $$ > "pid.${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}"
exec "$@"
"""


def fields(path: Path) -> dict[str, str]:
    return rs.read_fields(path)


def end_files(d: Path) -> list[str]:
    return sorted(n for n in (rs.RUNNING, *rs.END_FILES) if (d / n).exists())


class Run:
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


def solver(np: int, args: list[str], exe: str = "dnsjax") -> list[str]:
    base = [str(BIN / exe)]
    if np > 1:
        base = ["mpirun", "-np", str(np), "./pids.sh", *base]
        args = ["--dist.np1", str(np), *args]
    return [*base, *args]


def check_stopped(
    d: Path,
    run: Run,
    rc: int | None,
    request: str,
    np: int,
    also: Callable[[Path, str], str | None] | None = None,
) -> None:
    """Check a graceful stop: exit 0 and ``STOPPED``, a final snapshot
    the last ``stats.dat`` row reaches, every process shut down.

    *also(d, snapshot_name)* adds an entry point's own check of what
    its stop wrote (a failure string, or ``None``).
    """
    if rc is None:
        raise AssertionError("hung after the stop request")
    if rc != 0:
        raise AssertionError(f"exit {rc} after the stop request")
    if end_files(d) != [rs.STOPPED]:
        raise AssertionError(f"status files {end_files(d)}")
    got = fields(d / rs.STOPPED)
    if f"stop requested ({request})" not in got["reason"]:
        raise AssertionError(f"reason {got['reason']!r}")
    snap = got.get("snapshot", "")
    name = snap.split(" ")[0]
    if not name or name == "state00000.tar" or not (d / name).is_file():
        raise AssertionError(f"no final snapshot ({snap!r})")
    if also is not None and (reason := also(d, name)):
        raise AssertionError(reason)
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
    run = Run(solver(1, [*SOLVER, "--stop.max_sim_time", "0.05"]), d)
    rc = run.finish()
    assert rc == 0, f"exit {rc}"
    assert end_files(d) == [rs.FINISHED], end_files(d)
    got = fields(d / rs.FINISHED)
    assert "stop.max_sim_time reached" in got["reason"], got
    assert got["exit"] == "0", got
    assert got["snapshot"].startswith("state00001.tar"), got
    ctx["parent"] = d / "state00000.tar"


def live_stop(
    np: int,
    how: str,
    launch: Callable[[dict], tuple[str, list[str]]] | None = None,
    also: Callable[[Path, str], str | None] | None = None,
):
    """A case stopping a live run on *np* processes, by *how*.

    *how*: ``"remove"`` (delete ``RUNNING``), ``"rank1"`` (``SIGUSR1``
    to rank 1 alone) or ``"launched"`` (``SIGUSR1`` to the launched
    process).  *launch(ctx)* gives another entry point and its
    arguments (default: the solver, long-running); *also* is passed to
    :func:`check_stopped`.
    """

    def case(d: Path, ctx: dict) -> None:
        exe, args = (
            ("dnsjax", [*SOLVER, *LONG]) if launch is None else launch(ctx)
        )
        cmd = solver(np, args, exe)
        if np > 1:
            shutil.copy(ctx["pids"], d / "pids.sh")
        run = Run(cmd, d)
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
        check_stopped(d, run, run.finish(), request, np, also)

    return case


def run_cases(cases: list[tuple[str, Callable]]) -> int:
    """Run *cases* in fresh directories of one temporary tree; report.

    A case is ``(name, f)``: ``f(d)`` for a name starting ``unit_``,
    else ``f(d, ctx)``, where ``ctx`` carries the rank-pid wrapper
    (``"pids"``) and what an earlier case leaves for a later one (the
    natural run's ``"parent"``).
    """
    passed, failures = 0, []
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_run_status_"))
    ctx: dict = {"pids": tmp / "pids.sh"}
    ctx["pids"].write_text(PIDS)
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
