r"""The run status files and the stop request, for ``dnsjax-twin``.

The lab's cases of ``tests/test_run_status.py`` (the live-run
machinery: ``tests/_run_status.py``): ``rm RUNNING`` mid-run ends a
twin run ``STOPPED`` with exit 0 and a final snapshot pair whose time
the last ``stats.dat`` row carries, on one process and on two
(``mpirun -np 2``, where both ranks must shut down).  The twin's parent
is the initial snapshot of the core's natural-finish case, run here
first and judged the same way.

Run as a script::

    uv run python lab/tests/test_run_status.py             # both tiers
    uv run python lab/tests/test_run_status.py --no-mpi    # one process
    uv run python lab/tests/test_run_status.py --mpi-only  # two
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import _repo  # noqa: F401 -- the core suite's helpers
from _run_status import LONG, live_natural, live_stop, run_cases

sys.stdout.reconfigure(line_buffering=True)


def _twin(ctx: dict) -> tuple[str, list[str]]:
    """A long twin run from the natural run's initial snapshot."""
    return "dnsjax-twin", [
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
        *LONG,
    ]


def _partner(d: Path, name: str) -> str | None:
    """A twin stop writes the reference snapshot and its partner."""
    if not (d / name.replace(".tar", "_twin.tar")).is_file():
        return f"no partner of {name}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    tier = ap.add_mutually_exclusive_group()
    tier.add_argument("--no-mpi", action="store_true")
    tier.add_argument("--mpi-only", action="store_true")
    cli = ap.parse_args()

    # The twin's parent comes from the natural run.
    cases: list[tuple[str, Callable]] = [("natural", live_natural)]
    if not cli.mpi_only:
        cases.append(("twin-remove", live_stop(1, "remove", _twin, _partner)))
    if not cli.no_mpi:
        if shutil.which("mpirun") is None:
            print("mpirun is needed for the two-process tier")
            return 1
        cases.append(
            ("mpi-twin-remove", live_stop(2, "remove", _twin, _partner))
        )
    return run_cases(cases)


if __name__ == "__main__":
    sys.exit(main())
