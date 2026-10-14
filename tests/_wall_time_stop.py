r"""The skewed two-rank runs of the wall-clock stop tests, core and lab.

Each process times ``stop.max_wall_time`` from its own start, so the
budget runs out at a different moment on each.  Were each process to
decide alone, a budget running out between two processes' readings
would hang the run: one would leave the loop for the closing
collectives, the other enter another step's, and both would wait for
the rest until the job was killed.  The drivers therefore decide it
together (``sharding.Sharding.any_process``, at the
``outs.it_error_check`` cadence).

Here rank 1 starts ``--skew`` seconds after rank 0, which puts that
split on every run rather than on an unlucky one: both drivers must
stop on the budget, inside the stepping loop, and shut down on both
ranks, leaving a ``FINISHED`` that names the budget.  A per-process
stop would hang on each case until ``timeout`` ended it.

:func:`run_skewed` runs one such case and judges it; :func:`parse_cli`
and :func:`delay_script` are the scripts' shared setup.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from _live import run_live

BIN = Path(sys.executable).parent

# Rank 1 sleeps first, so its clock reads ``skew`` seconds behind rank
# 0's for the whole run (the ranks step in lockstep once both are up).
DELAY = """#!/bin/sh
[ "${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}" = 1 ] && sleep "$SKEW"
exec "$@"
"""

# A box small enough that a step is milliseconds, run with no horizon
# and no snapshot past the initial one, so only the budget can stop it.
COMMON = [
    "--dist.np1",
    "2",
    "--stop.max_sim_time",
    "1e6",
    "--stop.check_laminarization",
    "False",
    "--outs.it_snapshot",
    "0",
    "--outs.snapshot_save_final",
    "False",
]
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
]


def run_skewed(exe, args, cwd, delay, cli) -> str | None:
    """One 2-rank run under the skew; ``None`` when it stopped cleanly."""
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["NO_COLOR"] = "1"
    env["SKEW"] = str(cli.skew)
    limit = cli.budget + cli.grace
    # coreutils ``timeout`` ends a hung run with SIGTERM, which mpirun
    # passes on to its ranks; run_live's own (later) kill is SIGKILL,
    # which can orphan them.
    cmd = ["timeout", str(limit), "mpirun", "-np", "2", str(delay)]
    cmd += [str(BIN / exe), *COMMON, *args]
    cmd += ["--stop.max_wall_time", f"PT{cli.budget}S"]
    try:
        res = run_live(cmd, cwd=cwd, env=env, timeout=limit + 60)
    except subprocess.TimeoutExpired:
        return f"{exe} outlived its {limit + 60} s backstop"
    if res.returncode == 124:
        return (
            f"{exe} hung past its {cli.budget} s budget (killed at "
            f"{limit} s): the ranks parted at the wall-clock stop"
        )
    if res.returncode != 0:
        return f"{exe} exited {res.returncode}"
    if "First iteration over" not in res.stdout:
        return (
            f"{exe} spent its {cli.budget} s budget before stepping, so "
            "the stop was not tested inside the loop: raise --budget"
        )
    if "Stopped" not in res.stdout:
        return f"{exe} did not report its stop"
    shutdowns = res.stderr.count("Shutdown at")
    if shutdowns != 2:
        return f"{exe}: {shutdowns} of 2 ranks shut down"
    # The budget is one of the run's own criteria (dnsjax.run_status).
    finished = Path(cwd) / "FINISHED"
    if not finished.is_file() or (Path(cwd) / "RUNNING").exists():
        return f"{exe} left no FINISHED, or left its RUNNING"
    if "stop.max_wall_time reached" not in finished.read_text():
        return f"{exe}: FINISHED does not name the wall-clock budget"
    return None


def parse_cli(description: str) -> argparse.Namespace | None:
    """The scripts' ``--budget`` / ``--skew`` / ``--grace``, or ``None``.

    ``None`` (after saying why) when ``mpirun`` or coreutils
    ``timeout`` is missing.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--budget",
        type=int,
        default=60,
        help="stop.max_wall_time in seconds (default 60)",
    )
    ap.add_argument(
        "--skew",
        type=float,
        default=3.0,
        help="seconds rank 1 starts after rank 0 (default 3)",
    )
    ap.add_argument(
        "--grace",
        type=int,
        default=60,
        help="seconds past the budget before a run counts as hung",
    )
    cli = ap.parse_args()
    if shutil.which("mpirun") is None or shutil.which("timeout") is None:
        print("mpirun and coreutils timeout are needed")
        return None
    return cli


def delay_script(tmp: Path) -> Path:
    """Write the rank-1 delay wrapper into *tmp*; its path."""
    delay = tmp / "delay.sh"
    delay.write_text(DELAY)
    delay.chmod(delay.stat().st_mode | stat.S_IXUSR)
    return delay
