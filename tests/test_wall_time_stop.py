r"""The wall-clock stop, on processes whose clocks disagree.

Each process times ``stop.max_wall_time`` from its own start, so the
budget runs out at a different moment on each.  A stop each process
decided alone hung any run whose budget ran out between two processes'
readings: one left the loop for the closing collectives, the other
entered another step's, and both waited for the rest until the job
was killed.  The drivers therefore decide it together
(``sharding.Sharding.any_process``, at the ``outs.it_error_check``
cadence).

Here rank 1 starts ``--skew`` seconds after rank 0, which puts that
split on every run rather than on an unlucky one: both drivers must
stop on the budget, inside the stepping loop, and shut down on both
ranks.  The per-process stop hangs on each case until ``timeout``
ends it.

1. **Solver** (``mpirun -np 2 dnsjax``): plane Couette from a seeded
   random field; its initial snapshot is the twin's parent.
2. **Twin** (``mpirun -np 2 dnsjax-twin``) from that parent.

Run as a script (``--budget`` must leave the loop a few seconds after
the setup; ``--skew`` must exceed a step)::

    uv run python tests/test_wall_time_stop.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import report, run_live  # noqa: E402

_BIN = Path(sys.executable).parent

# Rank 1 sleeps first, so its clock reads ``skew`` seconds behind rank
# 0's for the whole run (the ranks step in lockstep once both are up).
_DELAY = """#!/bin/sh
[ "${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}" = 1 ] && sleep "$SKEW"
exec "$@"
"""

# A box small enough that a step is milliseconds, run with no horizon
# and no snapshot past the initial one, so only the budget can stop it.
_COMMON = [
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
]


def _run_skewed(exe, args, cwd, delay, cli) -> str | None:
    """One 2-rank run under the skew; ``None`` when it stopped cleanly."""
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["NO_COLOR"] = "1"
    env["SKEW"] = str(cli.skew)
    limit = cli.budget + cli.grace
    # coreutils ``timeout`` ends a hung run with SIGTERM, which mpirun
    # passes on to its ranks; run_live's own (later) kill is SIGKILL,
    # which can orphan them.
    cmd = ["timeout", str(limit), "mpirun", "-np", "2", str(delay)]
    cmd += [str(_BIN / exe), *_COMMON, *args]
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
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_wall_stop_"))
    try:
        delay = tmp / "delay.sh"
        delay.write_text(_DELAY)
        delay.chmod(delay.stat().st_mode | stat.S_IXUSR)
        solver, twin = tmp / "solver", tmp / "twin"
        solver.mkdir()
        twin.mkdir()
        results = [
            ("solver", _run_skewed("dnsjax", _SOLVER, solver, delay, cli))
        ]
        parent = solver / "state00000.tar"
        twin_args = [
            "--init.snapshot",
            str(parent),
            "--twin.e0",
            "1e-6",
            "--twin.seed",
            "3",
        ]
        results.append(
            (
                "twin",
                _run_skewed("dnsjax-twin", twin_args, twin, delay, cli)
                if parent.is_file()
                else "no parent snapshot (the solver case wrote none)",
            )
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, reason in results:
        print(f"  {'PASS' if reason is None else 'FAIL'}  {name}")
    failures = [(n, r) for n, r in results if r is not None]
    return report(len(results) - len(failures), failures)


if __name__ == "__main__":
    sys.exit(main())
