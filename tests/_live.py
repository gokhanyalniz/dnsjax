"""Live-teeing subprocess runner shared by the test scripts.

:func:`run_live` is a drop-in replacement for
``subprocess.run(cmd, capture_output=True, text=True, ...)``: it
streams the child's stdout/stderr line-by-line to this process's
stdout/stderr *while* accumulating both, so callers keep reading
``result.stdout`` / ``result.stderr`` exactly as with
``subprocess.run``, and an agent
tailing the run sees progress immediately (and can abort early).

The child environment gets ``PYTHONUNBUFFERED=1`` (prompt output from
the child and its own children, e.g. ``mpirun`` ranks) and, unless the
caller's environment already sets it, ``DNSJAX_QUIET_STARTUP=1`` (skip
the dnsjax solver's ~60-80 line startup parameter dump, which is pure
repeated noise across the many solver launches of a smoke test; inert
for non-solver children).  Both are added to *env*, or to a copy of
``os.environ`` when *env* is ``None``; a given *env* is otherwise
passed through unchanged, so deliberately stripped environments stay
stripped.

Semantics mirrored from ``subprocess.run``: on *timeout* expiry the
child is killed and :class:`subprocess.TimeoutExpired` is raised with
the output accumulated so far; ``check=True`` raises
:class:`subprocess.CalledProcessError` on a nonzero exit.  Two
deliberate differences: decoding uses ``errors="replace"`` (a stray
invalid byte must not kill the pump), and the relative ordering
between the stdout and stderr streams is not guaranteed (it never was
observable for captured output).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Sequence
from typing import IO, TextIO


def _pump(src: IO[str], sink: TextIO, chunks: list[str]) -> None:
    """Copy *src* to *sink* line-by-line, accumulating in *chunks*."""
    for line in src:
        chunks.append(line)
        sink.write(line)
        sink.flush()
    src.close()


def run_live(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike | None = None,
    check: bool = False,
    echo: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run *cmd* teeing its output live; mirrors ``subprocess.run``.

    *echo* prints a flushed ``+ <cmd>`` line before launching, so a
    tailing agent knows what is running (no caller parses the
    parent's own stdout, only the child pipes returned here).
    """
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    # Quiet the dnsjax solver's verbose startup parameter dump: a smoke
    # test may spawn the solver dozens of times and the dump (~60-80
    # lines) is redundant with the launched command.  ``setdefault`` so
    # ``DNSJAX_QUIET_STARTUP=0`` in the environment can re-enable it when
    # debugging.  Inert for non-solver children (they ignore it).
    child_env.setdefault("DNSJAX_QUIET_STARTUP", "1")
    if echo:
        print("+", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=child_env,
        cwd=cwd,
    )
    out: list[str] = []
    err: list[str] = []
    pumps = [
        threading.Thread(
            target=_pump, args=(proc.stdout, sys.stdout, out), daemon=True
        ),
        threading.Thread(
            target=_pump, args=(proc.stderr, sys.stderr, err), daemon=True
        ),
    ]
    for t in pumps:
        t.start()
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        for t in pumps:
            # Bounded: surviving grandchildren (mpirun ranks) may
            # hold the pipes open past the kill.
            t.join(timeout=5)
        raise subprocess.TimeoutExpired(
            cmd, timeout, output="".join(out), stderr="".join(err)
        ) from None
    except BaseException:
        proc.kill()
        raise
    for t in pumps:
        t.join()
    result = subprocess.CompletedProcess(
        proc.args, returncode, "".join(out), "".join(err)
    )
    if check and returncode != 0:
        raise subprocess.CalledProcessError(
            returncode, proc.args, output=result.stdout, stderr=result.stderr
        )
    return result


#: Clip width for one summary reason (see :func:`report`).
_REASON_CLIP = 200


def report(passed: int, failures: Sequence[tuple[str, str]]) -> int:
    """Print the closing summary; return the process exit code.

    *failures* is one ``(name, reason)`` per failed entry.  The
    reasons are repeated **after** the counts on purpose: these
    scripts tee thousands of lines of child output, so the inline
    ``FAIL`` line scrolls far out of a ``tail`` and leaves
    ``"16 passed, 1 failed."`` as the only visible signal -- which
    says nothing about *which* entry broke or why, and costs a full
    re-run to find out.  A run's outcome must be readable from its
    last few lines alone.

    Reasons are collapsed to their first line and clipped, so a long
    assertion message cannot push the summary out of a short tail
    either; the full text is still inline at the point of failure.
    """
    print(f"\n{passed} passed, {len(failures)} failed.")
    for name, reason in failures:
        head = " ".join(str(reason).split())
        if len(head) > _REASON_CLIP:
            head = head[: _REASON_CLIP - 3] + "..."
        print(f"  FAILED  {name}: {head}")
    return 1 if failures else 0
