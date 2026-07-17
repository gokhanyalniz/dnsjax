"""Live-teeing subprocess runner shared by the test scripts.

:func:`run_live` is a drop-in replacement for
``subprocess.run(cmd, capture_output=True, text=True, ...)``: it
streams the child's stdout/stderr line-by-line to this process's
stdout/stderr *while* accumulating both, so callers keep reading
``result.stdout`` / ``result.stderr`` exactly as before, and an agent
tailing the run sees progress immediately (and can abort early).

The child environment gets ``PYTHONUNBUFFERED=1`` -- added to *env*,
or to a copy of ``os.environ`` when *env* is ``None``; a given *env*
is otherwise passed through unchanged, so deliberately stripped
environments stay stripped -- which makes the child and its own
children (e.g. ``mpirun`` ranks) emit promptly.

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
