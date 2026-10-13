r"""Adaptive-dt guards for the lab's flows.

The lab's cases of ``tests/test_adaptive.py`` (the workers and the leaf
tables: ``tests/_adaptive.py``): the two viscoelastic flows, with
conformation diffusion enabled so the ``Hc_op`` rebuild is exercised
(``kappa = 0`` would drop the operator entirely).  Each subclasses its
base geometry's flow and reaches the IMM through it, so its leaf
tables are that geometry's plus the flag-independent ``Hc_op``.

Usage::

    uv run python lab/tests/test_adaptive.py
"""

from __future__ import annotations

import argparse
import sys

import _repo  # noqa: F401 -- the core suite's helpers
from _adaptive import (
    CLOSURE_LEAVES,
    DROPPED_LEAVES,
    LEAVES,
    LX,
    run_worker,
    worker,
)

sys.stdout.reconfigure(line_buffering=True)

#: The sPTT rheology, with nonzero conformation diffusion.
_PHYS = {"el": 20.0, "wi": 20.0, "beta": 0.8, "epsilon": 0.0, "kappa": 1e-3}

#: ``system -> case`` (the keys: :func:`_adaptive.worker`).
CASES: dict[str, dict] = {
    "viscoelastic-pipe": {
        "leaves": LEAVES["pipe"] + ("Hc_op",),
        "closure": CLOSURE_LEAVES["pipe"],
        "dropped": DROPPED_LEAVES["pipe"],
        "stepper": (
            "dnsjax_lab.geometries.wall_bounded.cylindrical_viscoelastic"
            ":build_viscoelastic_stepper"
        ),
        "phys": _PHYS,
        "geo": {"lx": LX},
    },
    "viscoelastic-dean": {
        "leaves": LEAVES["taylor-couette"] + ("Hc_op",),
        "closure": CLOSURE_LEAVES["taylor-couette"],
        "dropped": DROPPED_LEAVES["taylor-couette"],
        "stepper": (
            "dnsjax_lab.geometries.wall_bounded.annular_viscoelastic"
            ":build_viscoelastic_stepper"
        ),
        "phys": _PHYS,
        "geo": {"lx": LX},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", metavar="SYSTEM", help=argparse.SUPPRESS)
    parser.add_argument(
        "--backend", default="pallas", choices=["pallas", "dense"]
    )
    parser.add_argument(
        "--legacy-imm", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker:
        worker(
            args.worker, args.backend, not args.legacy_imm, CASES[args.worker]
        )
        return
    for system in CASES:
        run_worker(__file__, system, "pallas")
    print("\nAll adaptive-dt checks passed.")


if __name__ == "__main__":
    main()
