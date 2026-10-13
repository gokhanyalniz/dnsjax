r"""Shared pieces of the random-IC and localized-rolls smoke tests.

Each smoke script runs every row of its table as an ``mpirun ... -m
dnsjax`` subprocess in a fresh temporary directory (the geometry
modules capture global singletons at import time, and a fresh
directory keeps ``parameters.toml`` out and ``stats.dat`` apart).  A
row is a dict: ``name``, the CLI ``args`` (naming the flow with
``--phys.system``) and the runner's own optional keys.

:func:`check_run` holds the criteria every completed run must meet:

1. ``"Corrector failed to converge"`` absent from stdout;
2. the run reached the end (final ``t`` `$\geq$` ``max_sim_time -
   dt``): it was not cut short by corrector divergence (the main loop
   stops once ``err`` reaches ``corrector_tolerance``);
3. the final corrector error is finite and below
   ``corrector_tolerance`` (catches a late divergence in the last
   ``it_error_check`` steps);
4. every numeric value on the final summary line is finite (NaN/Inf
   print as ``nan``/``inf``).

The exit code is the runners' own check.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from collections.abc import Callable

from _live import report

# Default time-stepping ``corrector_tolerance`` (TimeStepping model
# default; the subprocesses run in temp dirs, so no parameters.toml).
CORRECTOR_TOLERANCE = 1e-5

# A signed float, or ``nan`` / ``inf`` (how non-finite values print).
_NUM = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
# ``\bt`` avoids matching the ``t`` in ``c/it =`` ('i' is a word char, so
# there is no word boundary before that ``t``).
T_PATTERN = re.compile(rf"\bt\s*=\s*({_NUM})")
ERR_PATTERN = re.compile(rf"\berr\s*=\s*({_NUM})")
VALUE_PATTERN = re.compile(rf"=\s*({_NUM})")


def res_flags(row_args: list[str], nx: str, ny: str, nz: str) -> list[str]:
    """A row's resolution flags, under its flow's public names.

    The cylindrical and annular flows take the aliased names (axial
    ``nz``, radial ``nr``, azimuthal ``ntheta``); the internal meaning
    (``nx`` axial, ``ny`` radial, ``nz`` azimuthal) is unchanged.  The
    flow's geometry comes from the registry.
    """
    from dnsjax.flows.registry import spec_for

    system = row_args[row_args.index("--phys.system") + 1]
    if spec_for(system).geometry in ("cylindrical", "annular"):
        return ["--res.nz", nx, "--res.nr", ny, "--res.ntheta", nz]
    return ["--res.nx", nx, "--res.ny", ny, "--res.nz", nz]


def final_summary_line(stdout: str) -> str | None:
    """Return the final ``t = ... err = ...`` summary line, if any.

    The per-step error appears only on the end-of-run summary line, so
    the last line carrying ``err =`` is that summary (the initial
    warm-up ``t = 0.00 ...`` print has no ``err``).
    """
    summary: str | None = None
    for line in stdout.splitlines():
        if ERR_PATTERN.search(line):
            summary = line
    return summary


def check_run(stdout: str, name: str, max_sim_time: float, dt: float) -> str:
    """Validate one completed run; return the summary line for the PASS.

    Raises ``AssertionError`` on any failed criterion.
    """
    if "failed to converge" in stdout:
        raise AssertionError(f"{name}: corrector failed to converge")

    summary = final_summary_line(stdout)
    if summary is None:
        raise AssertionError(
            f"{name}: no end-of-run summary line found\n{stdout[-2000:]}"
        )

    t_match = T_PATTERN.search(summary)
    err_match = ERR_PATTERN.search(summary)
    if t_match is None or err_match is None:
        raise AssertionError(f"{name}: cannot parse summary line: {summary!r}")

    t_final = float(t_match.group(1))
    err = float(err_match.group(1))

    # Reached the end (not cut short by corrector divergence).  The last
    # step lands t on (or just past) max_sim_time; allow one dt of slack.
    if not (t_final >= max_sim_time - dt):
        raise AssertionError(
            f"{name}: run ended early at t={t_final} (< {max_sim_time}); "
            "integration did not complete"
        )

    # Final corrector error finite and converged (catches a divergence
    # in the last it_error_check steps that the loop did not stop on).
    if not (math.isfinite(err) and err < CORRECTOR_TOLERANCE):
        raise AssertionError(
            f"{name}: final corrector error {err:.3e} not in "
            f"[0, {CORRECTOR_TOLERANCE:.0e})"
        )

    # No NaN / Inf anywhere on the summary line (every diagnostic finite).
    values = [float(v) for v in VALUE_PATTERN.findall(summary)]
    if not all(math.isfinite(v) for v in values):
        raise AssertionError(
            f"{name}: non-finite value on summary line: {summary!r}"
        )

    return summary


def run_rows(
    systems: list[dict],
    run_one: Callable[[dict, argparse.Namespace], None],
    args: argparse.Namespace,
) -> int:
    """Run *systems* (``args.systems`` of them, if given); the exit code."""
    if args.systems:
        wanted = set(args.systems)
        unknown = wanted - {s["name"] for s in systems}
        if unknown:
            print(f"Unknown system(s): {sorted(unknown)}")
            return 2
        systems = [s for s in systems if s["name"] in wanted]
    passed = 0
    failures: list[tuple[str, str]] = []
    for system in systems:
        try:
            run_one(system, args)
            passed += 1
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            print(f"  FAIL  {system['name']}: {exc}")
            failures.append((system["name"], str(exc)))
    return report(passed, failures)
