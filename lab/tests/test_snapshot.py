"""Snapshot round-trip tests for the lab's flows.

The lab's cases of ``tests/test_snapshot.py`` (the machinery and the
workers: ``tests/_snapshot.py``), on the flows whose state carries
more than the velocity -- the 9-component viscoelastic states (3
velocity + 6 conformation-tensor):

- save -> load round trips of viscoelastic-dean (single-device,
  ``np 1 -> 2`` and on a ``(2, 2)`` mesh, the only 9-component *save*
  on a multi-device mesh) and viscoelastic-pipe (whose family routes
  through the pipe's regrid and basis paths);
- the Fourier-resolution regrid of a 9-component state;
- an ``nr``-mismatch load assembles the 9-component state at the
  snapshot's radial count (``snapshot._n_components`` from the
  metadata);
- the parity-aware radial regrid of all nine stored components of the
  viscoelastic pipe, each in its own class
  (``FlowSpec.component_parity``): the conformation slots are only
  reachable through a viscoelastic flow in the cylindrical family.

Run as a script::

    uv run python lab/tests/test_snapshot.py
"""

from __future__ import annotations

import sys

import _repo  # noqa: F401 -- the core suite's helpers
from _live import report
from _snapshot import (
    run_case,
    run_load_shape_case,
    run_mode_case,
    run_parity_regrid_case,
)

# (name, system, layout, write_mode, save_np, load_np,
#  save_np0, load_np0)
CASES: list[tuple[str, str, str, str, int, int, int, int]] = [
    # ── viscoelastic (9-component state: 3 velocity + 6 tensor) ──
    (
        "viscoelastic-dean",
        "viscoelastic-dean",
        "walled",
        "concurrent",
        1,
        1,
        1,
        1,
    ),
    (
        "viscoelastic-dean np 1->2",
        "viscoelastic-dean",
        "walled",
        "concurrent",
        1,
        2,
        1,
        1,
    ),
    (
        "viscoelastic-pipe",
        "viscoelastic-pipe",
        "walled",
        "concurrent",
        1,
        1,
        1,
        1,
    ),
    (
        "viscoelastic-pipe np 1->2",
        "viscoelastic-pipe",
        "walled",
        "concurrent",
        1,
        2,
        1,
        1,
    ),
    (
        # The I/O layout carries the component axis whole and the
        # engines loop over it, so a component-count slip would land
        # every component but the first at the wrong file offset.
        "viscoelastic-dean 2D",
        "viscoelastic-dean",
        "walled",
        "concurrent",
        4,
        4,
        2,
        2,
    ),
]

MODE_CASES: list[dict] = [
    {
        "name": "viscoelastic 9-component regrid",
        "system": "viscoelastic-pipe",
        "save_res": (8, 9, 8),
        "load_res": (16, 9, 16),
    },
]


if __name__ == "__main__":
    results: list[tuple[str, str | None]] = [
        (case[0], run_case(*case)) for case in CASES
    ]
    results.extend(
        (case["name"], run_mode_case(**case)) for case in MODE_CASES
    )
    results.append(
        (
            "viscoelastic nr 8->16 load shape",
            run_load_shape_case("viscoelastic-dean"),
        )
    )
    results.append(
        (
            "viscoelastic-pipe nr 8->10 regrid",
            run_parity_regrid_case("viscoelastic-pipe"),
        )
    )
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))
