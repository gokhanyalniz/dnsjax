#!/usr/bin/env python3
r"""Adaptive-dt guards (offline, in-process, no ``mpirun``).

Pins the ``step.adaptive`` machinery end to end:

- **Controller units** (JAX-free, parent process):
  :func:`dnsjax.adaptive.propose_dt` -- growth capped by
  ``dt_max_change`` and ``dt_max``, shrink uncapped but floored by
  ``dt_min`` / ``dt_min_change``, the relative deadband (both
  directions, incl. exactly at the threshold), and the
  zero/non-finite-CFL grow-to-cap branch.
- **Rebuild-vs-fresh leaf parity** (subprocess per system, forced
  single CPU device, x64): build the flow at ``DT0``, step once, then
  ``set_dt(DT1)`` and compare every ``dt``-dependent leaf against a
  freshly constructed flow at ``DT1`` (direct ``params.step.dt``
  assignment before construction -- the ``test_cnab2`` idiom).  The
  jitted rebuild's fusion may differ from the eager setup build, so
  the comparison is a tight ``allclose``, not bitwise.
- **Step parity at the new dt**: one CN/AB2 and one iterative-CN step
  of the module steppers (rebuilt leaves, ``ab2_kappa`` reset) vs the
  same steps of freshly built steppers at ``DT1`` -- catches any
  ``params.step.dt`` still baked into the module trace.
- **No-recompile guard**: with ``jax_log_compiles`` captured, a
  second ``set_dt`` plus plain+measured steps of both schemes must
  produce **zero** JAX log records (no trace, no compile) -- the
  core promise that a dt change swaps pytree leaves only.
- **Live-dt embed**: ``recorded_params_dump`` follows a mutated
  ``params.step.dt`` (the invariant that makes every snapshot embed
  the adapted dt).

The plane-Couette case runs under both solver backends (pallas and
dense).  The workers and the leaf tables: ``tests/_adaptive.py``; the
lab's flows have cases of their own, ``lab/tests/test_adaptive.py``.

Usage::

    uv run python tests/test_adaptive.py                 # everything
    uv run python tests/test_adaptive.py --unit-only     # controller
    uv run python tests/test_adaptive.py --system pipe   # one system
"""

from __future__ import annotations

import argparse
import sys

from _adaptive import (
    CLOSURE_LEAVES,
    DROPPED_LEAVES,
    LEAVES,
    run_unit_checks,
    run_worker,
    worker,
)

sys.stdout.reconfigure(line_buffering=True)

#: ``system -> case`` (the keys: :func:`_adaptive.worker`).
CASES: dict[str, dict] = {
    "plane-couette": {
        "leaves": LEAVES["plane-couette"],
        "closure": CLOSURE_LEAVES["plane-couette"],
        "dropped": DROPPED_LEAVES["plane-couette"],
        "stepper": (
            "dnsjax.geometries.wall_bounded.cartesian:build_cartesian_stepper"
        ),
    },
    "pipe": {
        "leaves": LEAVES["pipe"],
        "closure": CLOSURE_LEAVES["pipe"],
        "dropped": DROPPED_LEAVES["pipe"],
        "stepper": (
            "dnsjax.geometries.wall_bounded.cylindrical"
            ":build_cylindrical_stepper"
        ),
    },
    "taylor-couette": {
        "leaves": LEAVES["taylor-couette"],
        "closure": CLOSURE_LEAVES["taylor-couette"],
        "dropped": DROPPED_LEAVES["taylor-couette"],
        "stepper": (
            "dnsjax.geometries.wall_bounded.annular:build_annular_stepper"
        ),
        "phys": {"re1": 100.0, "re2": 0.0},
        "geo": {"lx": 5.0, "lz": 5.0, "eta": 0.5},
    },
    "kolmogorov": {
        "leaves": LEAVES["kolmogorov"],
        "closure": CLOSURE_LEAVES["kolmogorov"],
        "stepper": (
            "dnsjax.geometries.triply_periodic.triply_periodic"
            ":build_triply_periodic_stepper"
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", metavar="SYSTEM", help=argparse.SUPPRESS)
    parser.add_argument(
        "--backend", default="pallas", choices=["pallas", "dense"]
    )
    parser.add_argument(
        "--system",
        action="append",
        choices=list(CASES),
        help="restrict to one or more systems (repeatable)",
    )
    parser.add_argument(
        "--legacy-imm", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="run only the JAX-free controller units",
    )
    args = parser.parse_args()

    if args.worker:
        worker(
            args.worker, args.backend, not args.legacy_imm, CASES[args.worker]
        )
        return

    run_unit_checks()
    print("  PASS  propose_dt units")
    if args.unit_only:
        print("\nAdaptive-dt unit checks passed (--unit-only).")
        return

    systems = args.system or list(CASES)
    cases = [(s, "pallas") for s in systems]
    if "plane-couette" in systems:
        # The dense backend shares the rebuild contract; one geometry
        # covers its DenseJAXSolver/from_factors path.
        cases.append(("plane-couette", "dense"))
    # The legacy ``res.consistent_imm = False`` path has a different
    # dt-dependent leaf set in every geometry (it carries pressure
    # responses where the default carries wall-normal-velocity ones);
    # one case per implementation.
    cases += [
        (s, "pallas", False)
        for s in ("plane-couette", "taylor-couette", "pipe")
        if s in systems
    ]
    for system, backend, *rest in cases:
        run_worker(__file__, system, backend, rest[0] if rest else True)
    print("\nAll adaptive-dt checks passed.")


if __name__ == "__main__":
    main()
