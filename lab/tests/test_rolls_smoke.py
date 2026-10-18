r"""Localized-rolls IC smoke tests for the lab's flows.

The lab's table of ``tests/test_rolls_smoke.py`` (the runner:
``tests/_rolls_smoke.py``; what a run is: the core script's
docstring): the curved pipe's toroidal rolls and Dean's.  The curved
pipe, a total-field flow, must print its laminar flux at ``t = 0``.

Usage::

    uv run python lab/tests/test_rolls_smoke.py
"""

from __future__ import annotations

import _repo  # noqa: F401 -- the core suite's helpers
from _rolls_smoke import main

# Same Reynolds numbers / boxes as the random smoke.
SYSTEMS: list[dict] = [
    {
        # The builder adds the laminar profile, so the t = 0 stats line
        # (before any step) carries its flux; a bare-rolls start would
        # print 0 there and leave the first bulk correction to supply it.
        "name": "curved-pipe",
        "expect_pattern": r"t = 0\.00 .*Ub_s=5\.000e-01",
        "args": [
            "--phys.system",
            "curved-pipe",
            "--phys.re",
            "1800",
            "--geo.curvature",
            "0.3",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "dean",
        "args": [
            "--phys.system",
            "dean",
            "--phys.re",
            "1000",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
        ],
    },
]


if __name__ == "__main__":
    main(SYSTEMS)
