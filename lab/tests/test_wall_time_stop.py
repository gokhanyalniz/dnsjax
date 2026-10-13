r"""The wall-clock stop of ``dnsjax-twin``, on processes whose clocks disagree.

The lab's case of ``tests/test_wall_time_stop.py`` (the mechanism:
``tests/_wall_time_stop.py``): ``mpirun -np 2 dnsjax-twin``, with rank
1 started ``--skew`` seconds after rank 0, must stop on its
``stop.max_wall_time`` budget inside the stepping loop and shut down on
both ranks, leaving a ``FINISHED`` that names the budget.  Its parent
is the initial snapshot of the core's solver case, run here first and
judged the same way.

Run as a script (``--budget`` must leave the loop a few seconds after
the setup; ``--skew`` must exceed a step)::

    uv run python lab/tests/test_wall_time_stop.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import _repo  # noqa: F401 -- the core suite's helpers
from _live import report
from _wall_time_stop import SOLVER, delay_script, parse_cli, run_skewed

sys.stdout.reconfigure(line_buffering=True)


def main() -> int:
    cli = parse_cli(__doc__.splitlines()[0])
    if cli is None:
        return 1
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_wall_stop_"))
    try:
        delay = delay_script(tmp)
        solver, twin = tmp / "solver", tmp / "twin"
        solver.mkdir()
        twin.mkdir()
        results = [
            ("solver", run_skewed("dnsjax", SOLVER, solver, delay, cli))
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
                run_skewed("dnsjax-twin", twin_args, twin, delay, cli)
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
