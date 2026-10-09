r"""Run status files and the graceful stop request.

A solver run (``dnsjax``, ``dnsjax-twin``) keeps one status file in
its run directory (the working directory) while it is live, and leaves
one behind when it ends:

``RUNNING``
    Written by process 0 as soon as the JAX runtime is configured
    (``--help``, ``--sample-toml`` and a parameter error exit earlier
    and write nothing).  **Deleting it asks the run to stop
    gracefully**, as does a ``SIGUSR1`` sent to any of its processes
    (or to ``mpirun``, which forwards it to every rank).
``FINISHED``
    The run ended on one of its own criteria: ``stop.max_sim_time``,
    ``stop.max_wall_time`` or relaminarization.  Exit code 0.
``STOPPED``
    The run stopped on request (``RUNNING`` deleted, or ``SIGUSR1``).
    Exit code 0.
``TERMINATED``
    Anything else the process lived to record.  The exit code says
    which: :data:`EXIT_CORRECTOR` (4) for a corrector that failed to
    converge, :data:`EXIT_NON_FINITE` (3) for the non-finite guard,
    ``128 + n`` for signal ``n`` (SIGTERM, SIGINT), and 1 for a setup
    refusal or an uncaught exception.

A ``RUNNING`` with no live process behind it belongs to a run that was
given no chance to record its end: ``SIGKILL`` (the scheduler's last
resort, the out-of-memory killer), a lost node, or a native abort
inside XLA or MPI.  No Python code runs on any of those.

The stop request
----------------
A request is one more exit criterion of the stepping loop beside the
wall-clock budget, judged with it at the ``outs.it_error_check`` host
sync (:meth:`RunStatus.stop_flags`), so it takes the closing path that
budget already takes: the final stats row and probe sample, the final
snapshot under ``outs.snapshot_save_final`` (default on), the final
reduced snapshot when due, and every stream flushed.  A request made
during setup or compilation is honoured before the first step, with
no step taken.  The decision is collective: process 0 alone looks for
``RUNNING`` and every process reports its own ``SIGUSR1``, and one
``any_process_each`` makes the answer the same on every process, so
all of them leave the loop on the same step.  A process that left
alone would wait forever in the next collective, and so would the
others.

``SIGTERM`` and ``SIGINT`` keep their immediate meaning (each driver's
handler flushes the streams and exits ``128 + n``).  A scheduler sends
``SIGKILL`` soon after ``SIGTERM`` (SLURM's ``KillWait``, often 30
s), which may not leave time for a final snapshot.  For a job time
limit, set ``stop.max_wall_time`` below it, or ask the scheduler for
an early ``SIGUSR1`` (SLURM: ``--signal=USR1@<seconds>``).

The files
---------
Each file is ``key: value`` lines, its own event's timestamp first
(ISO 8601, local time with its offset).  The keys that follow are
``started``, ``reason``, ``exit`` and ``snapshot``.  The last is the
newest snapshot this run wrote, the point a resume continues from.
``RUNNING`` instead carries what identifies the live run: ``host``,
``pid``, ``boot`` (the kernel boot id, where there is one),
``processes`` and ``job`` (``SLURM_JOB_ID``, when set).

Process 0 writes ``RUNNING``, ``FINISHED`` and ``STOPPED``.  It records
every abnormal end in ``TERMINATED``.  Another process appends a block
of its own (with a ``process`` line) only for an uncaught exception.
That is the one end it may live to record while process 0 sits in a
collective until it is killed.  Every write is complete or absent: a
whole file goes through a ``.partial`` rename, and ``TERMINATED``
blocks are appended and ``fsync``-ed.  An end file is written before
``RUNNING`` is removed, so the directory always holds one status file.

A start removes the ``FINISHED``, ``STOPPED`` or ``TERMINATED`` of an
earlier segment.  It refuses, on every process and touching no file,
when ``RUNNING`` names a live process on this machine (same host,
same boot), since two runs would then append to one set of streams.
Any other ``RUNNING`` it reports and replaces: either the run that
wrote it was killed (above), or that run is live on another host,
which nothing here can see.

This module is stdlib-only and imports JAX lazily (for the start-up
agreement alone), so the entry points import it before JAX is
configured.

Design notes
------------
**Why SIGTERM stays immediate.**  A graceful ``SIGTERM`` would finish
the ``it_error_check`` interval and write the final snapshot before
exiting.  Under a scheduler's ``SIGKILL`` deadline that can lose the
race, and a run killed before its closing flush loses every row still
buffered, which the immediate handler writes out first.  ``SIGUSR1``
is the graceful signal, because a scheduler can send it minutes
ahead.

**Why a stale RUNNING is replaced, not refused.**  A chained cluster
job that resumes after a time-limit kill always finds its
predecessor's ``RUNNING`` (``SIGKILL`` leaves it), and almost always
on another node.  Refusing there would break every chain.  The refusal
is kept for the case it can prove: a live process with that pid on
this machine since this boot.

**What the poll costs.**  One process pays a ``stat`` of ``RUNNING``
per ``outs.it_error_check`` steps and no collective.  Several pay one
gather of three ``int32`` there, where only a ``stop.max_wall_time``
run paid one (of one ``int32``) before.  It sits at the host sync the
corrector-error check already makes, so asynchronous dispatch loses
nothing more.  Measured on CPU, two processes on one AMD Ryzen 7 PRO
7840U over the MPI collectives, as the mean of 500 calls after 20
warm-up ones: 0.30 ms for the three-flag gather, against 0.21 ms for
the one-flag gather.  At the default cadence that is 0.03 ms per step:
under 1 % of a step even for a `$16 \times 33 \times 16$` plane
Couette box (a few ms per step there), and a smaller share wherever a
step takes longer.

**Why SIGUSR1 restarts system calls.**  A process-directed signal can
be delivered to any thread that does not block it, an XLA or MPI
thread included.  Python installs its handlers without ``SA_RESTART``,
so a blocking call in such a thread would return ``EINTR``.
``signal.siginterrupt(SIGUSR1, False)`` asks for the restart (the
calls the kernel never restarts still return ``EINTR``, as for any
signal).  The two immediate signals need nothing of the kind: the run
ends either way.
"""

import os
import signal
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RUNNING = "RUNNING"
FINISHED = "FINISHED"
STOPPED = "STOPPED"
TERMINATED = "TERMINATED"
#: The files a run leaves behind; a start removes an earlier segment's.
END_FILES: tuple[str, ...] = (FINISHED, STOPPED, TERMINATED)

#: Exit code of the non-finite guard (``FATAL: non-finite ...``).
EXIT_NON_FINITE: int = 3
#: Exit code of a run whose corrector failed to converge.
EXIT_CORRECTOR: int = 4

# Set by the SIGUSR1 handler, read at each stop poll.  A flag only:
# the handler runs between two bytecodes of the main thread, wherever
# the loop happens to be.
_stop_signalled: bool = False


def _on_stop_signal(signum: int, frame: object) -> None:
    global _stop_signalled
    _stop_signalled = True


def install_stop_signal() -> None:
    """Make ``SIGUSR1`` request a graceful stop (:class:`RunStatus`).

    Call first thing in an entry point, before JAX starts: until a
    handler is installed, the signal's default action kills the
    process.  System calls it interrupts are restarted (Design notes,
    "Why SIGUSR1 restarts system calls").
    """
    signal.signal(signal.SIGUSR1, _on_stop_signal)
    signal.siginterrupt(signal.SIGUSR1, False)


@dataclass(frozen=True)
class Outcome:
    """How a run ended: its status file, the reason, the exit code."""

    status: str
    reason: str
    exit_code: int = 0


def loop_outcome(
    t: float,
    it: int,
    *,
    corrector_error: float | None,
    request: str | None,
    out_of_time: bool,
    laminarized: bool,
) -> Outcome:
    """The :class:`Outcome` of a stepping loop that ended on a criterion.

    *corrector_error* is the largest corrector error when the corrector
    failed to converge, ``None`` when it did (or ran a fixed count);
    *request* is the stop request (:meth:`RunStatus.stop_flags`).  A
    failed corrector wins over a request, and a request over the
    run's own criteria; a loop with none of these flags set reached
    its horizon.
    """
    at = f"at t = {t:.6e}, it = {it}"
    if corrector_error is not None:
        return Outcome(
            TERMINATED,
            "corrector failed to converge in the steps up to "
            f"t = {t:.6e}, it = {it} (max error {corrector_error:.3e})",
            EXIT_CORRECTOR,
        )
    if request is not None:
        return Outcome(STOPPED, f"stop requested ({request}) {at}")
    if laminarized:
        return Outcome(FINISHED, f"laminarized {at}")
    if out_of_time:
        return Outcome(FINISHED, f"stop.max_wall_time reached {at}")
    return Outcome(FINISHED, f"stop.max_sim_time reached {at}")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _boot_id() -> str | None:
    """The kernel's boot id, or ``None`` where there is none (non-Linux)."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def read_fields(path: Path) -> dict[str, str]:
    """The ``key: value`` fields of a status file's first block.

    A ``TERMINATED`` written by several processes holds one block
    each; this reads the first.  Raises ``OSError`` when *path* cannot
    be read.
    """
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if sep:
            fields.setdefault(key.strip(), value.strip())
    return fields


def _format(fields: dict[str, object]) -> str:
    return "".join(
        f"{key}: {value}\n"
        for key, value in fields.items()
        if value is not None
    )


def _write(path: Path, text: str) -> None:
    """Write *path* whole: a ``.partial`` sibling, ``fsync``, rename."""
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)


def _append(path: Path, text: str) -> None:
    """Append one block to *path*, blank-line separated, ``fsync``-ed."""
    with open(path, "a") as f:
        if f.tell() > 0:
            f.write("\n")
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, under another user
    return True


def _live_owner(path: Path) -> str | None:
    """Why *path* belongs to a live run on this machine, or ``None``.

    Live means the ``host`` and ``boot`` it records are this machine's
    and the ``pid`` it records is running.  Where either side has no
    boot id, the host and pid decide alone; a refusal there can be
    lifted by deleting the file.
    """
    try:
        fields = read_fields(path)
        pid = int(fields["pid"])
    except (OSError, KeyError, ValueError):
        return None
    if fields.get("host") != socket.gethostname() or pid == os.getpid():
        return None
    boot, recorded = _boot_id(), fields.get("boot")
    if boot is not None and recorded is not None and boot != recorded:
        return None
    if not _alive(pid):
        return None
    return (
        f"{path} names a live process (pid {pid} on "
        f"{fields['host']}, started {fields.get('started', '?')}): "
        "another run is using this directory.  Wait for it to end "
        "(deleting RUNNING asks it to stop); if that process is not "
        "a dnsjax run, delete RUNNING and launch again."
    )


def _agree(flag: bool) -> bool:
    """Process 0's *flag* on every process (a collective)."""
    import numpy as np
    from jax.experimental.multihost_utils import broadcast_one_to_all

    return bool(broadcast_one_to_all(np.array([int(flag)], np.int32))[0])


def _signal_name(n: int) -> str:
    try:
        return signal.Signals(n).name
    except ValueError:
        return "?"


def _ended_by(exc: BaseException) -> tuple[str, int]:
    """``(reason, exit code)`` for an exception that ends the process."""
    if isinstance(exc, SystemExit):
        code = exc.code
        if isinstance(code, int):
            if 128 < code < 128 + 65:
                n = code - 128
                return f"signal {n} ({_signal_name(n)})", code
            return f"exit code {code}", code
        # SystemExit("message"): Python prints it and exits 1.
        text = str(code).strip().splitlines()
        return (text[0] if text else "exit code 1"), 1
    if isinstance(exc, KeyboardInterrupt):
        # Python ends an uncaught interrupt by re-raising SIGINT.
        return f"signal 2 ({_signal_name(2)})", 128 + 2
    text = str(exc).strip().splitlines()
    name = type(exc).__name__
    return (f"{name}: {text[0]}" if text else name), 1


class RunStatus:
    """The run directory's status file, as a context manager.

    *process_index* and *process_count* are this process's place in
    the run (``jax.process_index()`` / ``jax.process_count()``, so
    construct it after the JAX runtime is configured).  Entering checks
    and replaces an earlier ``RUNNING`` (a collective on a
    multi-process run, so every process enters at the same point) and
    writes the new one; leaving writes the end file from
    :attr:`outcome`, or ``TERMINATED`` when an exception (a
    ``SystemExit`` included) is leaving, which it re-raises.  The
    module docstring is the contract.
    """

    def __init__(
        self,
        process_index: int = 0,
        process_count: int = 1,
        directory: str | os.PathLike = ".",
    ) -> None:
        self.index = process_index
        self.count = process_count
        self.main = process_index == 0
        self.dir = Path(directory).resolve()
        #: Set by the entry point from its driver's return; ``None``
        #: (no driver return) records a plain ``FINISHED``.
        self.outcome: Outcome | None = None
        self._started: str | None = None
        self._note: str | None = None
        self._snapshot: str | None = None
        self._owns_running = False

    @property
    def running(self) -> Path:
        return self.dir / RUNNING

    def __enter__(self) -> "RunStatus":
        refusal = _live_owner(self.running) if self.main else None
        refused = refusal is not None
        if self.count > 1:
            refused = _agree(refused)
        if refused:
            # Raised before anything is written, so no file of the
            # live run is touched (``__exit__`` does not run either).
            raise SystemExit(
                f"dnsjax: error: {refusal}"
                if refusal is not None
                else "dnsjax: error: process 0 found another run live "
                "in this directory"
            )
        if self.main:
            if self.running.exists():
                try:
                    old = self.running.read_text().rstrip()
                except OSError:
                    old = "(unreadable)"
                print(
                    "Replacing the RUNNING of an earlier run, which "
                    "either ended without recording it (SIGKILL, a "
                    "lost node) or is live on another host:\n  "
                    + old.replace("\n", "\n  "),
                    flush=True,
                )
            for name in END_FILES:
                (self.dir / name).unlink(missing_ok=True)
            self._started = _now()
            _write(
                self.running,
                _format(
                    {
                        "started": self._started,
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "boot": _boot_id(),
                        "processes": self.count,
                        "job": os.environ.get("SLURM_JOB_ID"),
                    }
                ),
            )
            self._owns_running = True
        return self

    def stop_flags(
        self,
        late: bool,
        any_process_each: Callable[[Sequence[bool]], tuple[bool, ...]],
    ) -> tuple[bool, str | None]:
        """Poll the stop criteria: ``(out_of_time, request)``.

        *late* is this process's own wall-clock verdict; *request* is
        ``"RUNNING removed"`` (process 0 found it gone), ``"SIGUSR1"``
        (some process received one) or ``None``.  Both are OR-ed over
        the processes by *any_process_each*
        (:meth:`dnsjax.sharding.Sharding.any_process_each`), so every
        process must poll at the same point of the loop.
        """
        removed = self._owns_running and not self.running.exists()
        late, removed, signalled = any_process_each(
            (late, removed, _stop_signalled)
        )
        if removed:
            return late, "RUNNING removed"
        return late, "SIGUSR1" if signalled else None

    def note(self, reason: str) -> None:
        """Record why the run is about to exit (the ``reason`` line)."""
        self._note = reason

    def snapshot_written(self, name: str, t: float, it: int) -> None:
        """Record the newest snapshot (the ``snapshot`` line)."""
        self._snapshot = f"{name} (t = {t:.6e}, it = {it})"

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None or (
            isinstance(exc, SystemExit) and exc.code in (0, None)
        ):
            outcome = self.outcome or Outcome(FINISHED, "completed")
        else:
            reason, code = _ended_by(exc)
            outcome = Outcome(TERMINATED, self._note or reason, code)
        if self.main:
            self._record(outcome)
        elif isinstance(exc, Exception):
            # A peer records only its own exception (the module
            # docstring says why), with the process it happened on.
            self._record(outcome, peer=True)
        return False

    def _record(self, outcome: Outcome, peer: bool = False) -> None:
        fields: dict[str, object] = {
            outcome.status.lower(): _now(),
            "started": self._started,
            "reason": outcome.reason,
            "exit": outcome.exit_code,
        }
        if outcome.status == TERMINATED and self.count > 1:
            fields["process"] = (
                f"{self.index} of {self.count} on {socket.gethostname()}"
            )
        fields["snapshot"] = self._snapshot
        path = self.dir / outcome.status
        if outcome.status == TERMINATED:
            _append(path, _format(fields))
        else:
            _write(path, _format(fields))
        if self._owns_running and not peer:
            self.running.unlink(missing_ok=True)
            self._owns_running = False
