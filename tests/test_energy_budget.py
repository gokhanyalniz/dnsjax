#!/usr/bin/env python3
r"""Energy-budget closure guard: ``dE/dt == I - D`` to truncation order.

The wall-bounded analogue of ``tests/test_monochromatic.py``'s exact
Parseval identities (which do not otherwise exist for the wall-bounded
families).  Each config steps a **resolved** flow a short time via the
``dnsjax`` console script, then checks the discrete total-energy budget

.. math::
    \frac{dE}{dt} = I - D

closes -- i.e. the stepped-state divergence residual and the
``D1``-enstrophy-vs-``D2``-Laplacian summation-by-parts gap are *inert*,
injecting no ``O(1)`` source into the balance -- **on and off** the
radial-operator flags (``res.consistent_imm``, ``res.pipe_axis_fit``).

The residual is finite-difference/truncation level: a central-difference
``dE/dt`` at ``dt`` (``O(dt^2)`` on the smooth part) plus the SBP defect,
both convergent.  The guard is deliberately loose (``< BUDGET_TOL``
relative to the *term* magnitudes ``max(|I|, |D|, |dE/dt|)``, **excluding
the first-step projection transient**) so it catches an ``O(1)`` leak,
not the expected ``O(dt)`` finite-difference size.  Normalising by the
term magnitudes -- not by ``I - D``, which is near-zero for a
laminar-dominated pipe roll -- keeps it well-conditioned in every case.

Background: memory ``reference_imm_divergence_residual`` (the residual is
a convergent truncation error, physically inert for resolved fields) and
the ``Resolution.consistent_imm`` / ``pipe_axis_fit`` docstrings.  The
pipe ``consistent_imm`` entry needs a resolved IC (``localized_rolls``);
a grid-white random draw makes its composed ``D2`` diverge (see the
``pipe_axis_fit`` entry, which stays stable on a random IC).

Run directly (needs ``mpirun``; each config launches ``dnsjax`` once)::

    uv run python tests/test_energy_budget.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

BIN = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "dnsjax")

DT = 0.01
STEPS = 40  # total steps; the budget is read after the startup skip
SKIP = 12  # drop the first-step projection transient before judging
BUDGET_TOL = 0.05  # max|dE/dt-(I-D)| / max(|I|,|D|,|dE/dt|), steady

# Each entry: (label, system, [extra CLI flags]).  A random IC (the
# default start mode) unless the flags select localized_rolls.  Cartesian
# and annular gates are random-IC-stable; the pipe consistent_imm gate is
# not, so it (and its plain / axis-fit siblings, for a like-for-like
# resolved comparison) start from a localized-rolls spot.
_ROLLS = [
    "--init.localized_rolls",
    "True",
    "--init.localized_rolls_amplitude",
    "0.15",
    "--geo.lz",
    "8.0",
]
CONFIGS = [
    ("plane-couette off", "plane-couette", ["--res.consistent_imm", "False"]),
    ("plane-couette on", "plane-couette", ["--res.consistent_imm", "True"]),
    (
        "taylor-couette off",
        "taylor-couette",
        ["--res.consistent_imm", "False"],
    ),
    ("taylor-couette on", "taylor-couette", ["--res.consistent_imm", "True"]),
    ("pipe plain", "pipe", [*_ROLLS, "--res.pipe_axis_fit", "False"]),
    ("pipe axis-fit", "pipe", [*_ROLLS, "--res.pipe_axis_fit", "True"]),
    ("pipe consistent_imm", "pipe", [*_ROLLS, "--res.consistent_imm", "True"]),
]


def _base_flags(system: str) -> list[str]:
    """Resolution / Reynolds per family (moderate, resolved)."""
    if system == "plane-couette":
        return [
            "--phys.re",
            "500",
            "--res.nx",
            "16",
            "--res.nz",
            "16",
            "--res.ny",
            "33",
        ]
    if system == "taylor-couette":
        return [
            "--phys.re1",
            "100",
            "--phys.re2",
            "-100",
            "--geo.eta",
            "0.5",
            "--res.nz",
            "16",
            "--res.nr",
            "48",
            "--res.ntheta",
            "16",
        ]
    # pipe
    return [
        "--phys.re",
        "3000",
        "--res.nz",
        "24",
        "--res.nr",
        "64",
        "--res.ntheta",
        "16",
    ]


def _columns(stats: Path) -> dict[str, np.ndarray]:
    """Parse stats.dat by header name (columns are sorted-key order)."""
    with open(stats) as fh:
        header = fh.readline().split()
    data = np.loadtxt(stats, skiprows=1)
    return {name: data[:, i] for i, name in enumerate(header)}


def _run(label: str, system: str, flags: list[str]) -> str | None:
    """Launch one config; return ``None`` on a closed budget else why."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "mpirun",
            "-np",
            "1",
            BIN,
            "--phys.system",
            system,
            *_base_flags(system),
            *flags,
            "--step.dt",
            str(DT),
            "--stop.max_sim_time",
            str(STEPS * DT),
            "--outs.it_stats",
            "1",
            "--outs.it_corrector",
            "1",
            "--outs.it_error_check",
            "1",
        ]
        env = {**os.environ, "DNSJAX_QUIET_STARTUP": "1"}
        proc = run_live(cmd, cwd=tmp, timeout=400, env=env)
        if proc.returncode != 0:
            return f"{label}: dnsjax exit {proc.returncode}"
        cols = _columns(Path(tmp) / "stats.dat")
    t, E, inp, D = cols["t"], cols["E"], cols["I"], cols["D"]
    if len(t) < SKIP + 5:
        return f"{label}: only {len(t)} steps (need > {SKIP + 5})"
    dEdt = np.gradient(E, t)
    resid = np.abs(dEdt - (inp - D))[SKIP:-1]
    scale = max(np.max(np.abs(inp)), np.max(np.abs(D)), np.max(np.abs(dEdt)))
    rel = float(np.max(resid) / scale)
    ok = rel < BUDGET_TOL
    print(
        f"  {'PASS' if ok else 'FAIL'}: {label:22s} "
        f"steady max|dE/dt-(I-D)|/scale = {rel:.2e} "
        f"(< {BUDGET_TOL})"
    )
    return None if ok else f"{label}: budget residual {rel:.2e}"


def main() -> None:
    print(
        "Energy-budget closure: dE/dt == I - D to truncation order, on "
        "and off the radial-operator flags (offline; mpirun -np 1).",
        flush=True,
    )
    results = [(lbl, _run(lbl, sys_, fl)) for lbl, sys_, fl in CONFIGS]
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))


if __name__ == "__main__":
    main()
