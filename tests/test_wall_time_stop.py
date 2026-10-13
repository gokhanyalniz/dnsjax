r"""The wall-clock stop, on processes whose clocks disagree.

Rank 1 starts ``--skew`` seconds after rank 0, so each process's
``stop.max_wall_time`` budget runs out at a different moment: the
solver must still stop on the budget, inside the stepping loop, and
shut down on both ranks, leaving a ``FINISHED`` that names the budget
(the mechanism: ``tests/_wall_time_stop.py``).  The case is
``mpirun -np 2 dnsjax`` on plane Couette from a seeded random field.

Run as a script (``--budget`` must leave the loop a few seconds after
the setup; ``--skew`` must exceed a step)::

    uv run python tests/test_wall_time_stop.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

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
        solver = tmp / "solver"
        solver.mkdir()
        results = [
            ("solver", run_skewed("dnsjax", SOLVER, solver, delay, cli))
        ]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, reason in results:
        print(f"  {'PASS' if reason is None else 'FAIL'}  {name}")
    failures = [(n, r) for n, r in results if r is not None]
    return report(len(results) - len(failures), failures)


if __name__ == "__main__":
    sys.exit(main())
