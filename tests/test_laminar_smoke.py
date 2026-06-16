"""Laminar smoke tests for all wall-bounded flows.

Runs each wall-bounded system from its laminar state for a few
time steps at low resolution, verifying that:

- The corrector converges in a single iteration.
- The stepping error is `$O(10^{-18})$` or less.
- The perturbation energy stays at `$O(10^{-32})$` or less.
- The CFL diagnostic (``steps.dat``, written every
  ``it_steps = 1`` steps) matches the laminar base flow: for
  the Cartesian flows `$\\mathrm{CFL}_x
  = \\Delta t \\, \\max|U_x| \\, n_x / l_x$` exactly (the
  wall-normal and spanwise columns are roundoff-sized), for
  the pipe `$0 < \\mathrm{CFL}_z < \\Delta t \\, n_x / l_x$`.

Each system is tested in a separate subprocess because the
geometry modules capture global singletons at import time.
Each subprocess runs in its own temporary directory, so
``stats.dat`` / ``steps.dat`` are per-system (and the repo's
``parameters.toml`` is not loaded; the smoke runs use the
parameter-model defaults plus the CLI arguments).

Usage (single device)::

    uv run python tests/test_laminar_smoke.py

Usage (two devices via MPI)::

    uv run python tests/test_laminar_smoke.py --np 2

With ``--np N`` (N > 1), each test invokes
``mpirun -np N python -m dnsjax --dist.np0 NP0 --dist.np1 NP1 ...``
where ``NP0 * NP1 == N``.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ── configuration ────────────────────────────────────────────────────

SYSTEMS: list[dict] = [
    {
        "name": "plane-couette",
        "args": [
            "--phys.system",
            "plane-couette",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "plane-poiseuille",
        "args": [
            "--phys.system",
            "plane-poiseuille",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "plane-poiseuille-cbv",
        "args": [
            "--phys.system",
            "plane-poiseuille",
            "--phys.driving",
            "constant_bulk_velocity",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "plane-couette-block-spanwise",
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.block_mean_spanwise_velocity",
            "True",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "plane-poiseuille-block-spanwise",
        "args": [
            "--phys.system",
            "plane-poiseuille",
            "--phys.block_mean_spanwise_velocity",
            "True",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "pipe",
        "args": [
            "--phys.system",
            "pipe",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "pipe-cbv",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.driving",
            "constant_bulk_velocity",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "pipe-block-spanwise",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.block_mean_spanwise_velocity",
            "True",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "taylor-couette",
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "100",
            "--phys.re2",
            "0",
            "--geo.eta",
            "0.5",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
    {
        "name": "taylor-couette-block-spanwise",
        "args": [
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "100",
            "--phys.re2",
            "0",
            "--geo.eta",
            "0.5",
            "--phys.block_mean_spanwise_velocity",
            "True",
            "--init.start_from_laminar",
            "True",
            "--stop.max_sim_time",
            "0.04",
            "--outs.it_stats",
            "1",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--res.ny",
            "27",
        ],
    },
]

ERR_PATTERN = re.compile(r"err\s*=\s*([\d.eE+\-]+)")
EP_PATTERN = re.compile(r"E'\s*=\s*([\d.eE+\-]+)")

ERR_THRESHOLD = 1e-14
EP_THRESHOLD = 1e-28

# Values shared by every smoke system (dt and lx are the
# parameter-model defaults; the subprocesses run in temporary
# directories, so no parameters.toml interferes).
SMOKE_DT = 0.01
SMOKE_NX = 4
SMOKE_NZ = 4
SMOKE_LX = 4.0
SMOKE_N_STEPS = 4  # max_sim_time 0.04 / dt 0.01
# Spectral-grid spacing convention: Delta_x = lx / nx.
CFL_X_LAMINAR = SMOKE_DT * SMOKE_NX / SMOKE_LX
# Taylor-Couette laminar azimuthal CFL: dt * max(U_theta/r) * nz/(2 pi)
# (theta period is the literal 2*pi).  For re1=100, re2=0, eta=0.5 the
# maximum of U_theta/r is 1, at the inner wall r1 = 1 where
# U_theta(r1) = 1.
CFL_TH_LAMINAR_TC = SMOKE_DT * 1.0 * SMOKE_NZ / (2 * math.pi)
CFL_REL_TOL = 1e-7
CFL_ZERO_TOL = 1e-12  # roundoff-sized perturbation columns

# ── helpers ──────────────────────────────────────────────────────────


def _build_command(
    system_args: list[str], np_count: int, np0: int = 1
) -> list[str]:
    """Build the subprocess command for a single system."""
    base = [
        "mpirun",
        "-np",
        str(np_count),
        sys.executable,
        "-m",
        "dnsjax",
        "--dist.np0",
        str(np0),
        "--dist.np1",
        str(np_count // np0),
        "--outs.it_steps",
        "1",
    ]
    return base + system_args


def _parse_diagnostics(
    stdout: str,
) -> tuple[float | None, float | None]:
    """Extract the last ``err`` and ``E'`` from stdout."""
    last_err: float | None = None
    last_ep: float | None = None
    for line in stdout.splitlines():
        m_err = ERR_PATTERN.search(line)
        if m_err:
            last_err = float(m_err.group(1))
        m_ep = EP_PATTERN.search(line)
        if m_ep:
            last_ep = float(m_ep.group(1))
    return last_err, last_ep


def _check_steps_file(workdir: Path, name: str) -> None:
    """Validate ``steps.dat`` against laminar CFL expectations."""
    steps_file = workdir / "steps.dat"
    if not steps_file.exists():
        raise AssertionError(f"{name}: steps.dat was not written")

    lines = [ln for ln in steps_file.read_text().splitlines() if ln.strip()]
    header = lines[0].split()

    # Dicts returned through ``jit`` are canonicalised to sorted
    # key order, so compare as sets and index columns by name
    # ("t" is always written first by ``_flush_stats``).
    cylindrical = name.startswith("pipe")
    annular = name.startswith("taylor-couette")
    if cylindrical or annular:
        per_dir = ("CFL_z", "CFL_r", "CFL_th")
    else:
        per_dir = ("CFL_x", "CFL_y", "CFL_z")
    expected_cols = {"t", "CFL", *per_dir}
    if header[0] != "t" or set(header) != expected_cols:
        raise AssertionError(
            f"{name}: steps.dat header {header} != {expected_cols}"
        )
    col = {n: i for i, n in enumerate(header)}

    rows = [[float(v) for v in ln.split()] for ln in lines[1:]]
    if len(rows) != SMOKE_N_STEPS:
        raise AssertionError(
            f"{name}: expected {SMOKE_N_STEPS} steps.dat rows, got {len(rows)}"
        )
    if rows[0][col["t"]] != 0.0:
        raise AssertionError(
            f"{name}: first steps.dat row at t={rows[0][col['t']]}, not 0"
        )

    # Index of the laminar (base-flow) direction within ``per_dir``:
    # the streamwise/axial column (0) for Cartesian and pipe, but the
    # azimuthal CFL_th column (2) for the Taylor-Couette base flow.
    active_i = 2 if annular else 0

    for row in rows:
        if not all(math.isfinite(v) and v >= 0.0 for v in row):
            raise AssertionError(f"{name}: bad steps.dat row {row}")
        vals = [row[col[d]] for d in per_dir]
        total = row[col["CFL"]]
        active = vals[active_i]
        # Perturbation-only columns: roundoff-sized for the laminar
        # state (not exactly zero after the first step).
        roundoff = [v for i, v in enumerate(vals) if i != active_i]
        if any(v > CFL_ZERO_TOL for v in roundoff):
            raise AssertionError(
                f"{name}: nonzero perturbation CFL in row {row}"
            )
        if cylindrical:
            # max U_z = max (1 - r^2) < 1 on the half-CGL grid
            # (r = 0 is not a node), but only barely below it.
            if not 0.9 * CFL_X_LAMINAR < active < CFL_X_LAMINAR:
                raise AssertionError(
                    f"{name}: CFL_z {active} outside "
                    f"(0.9, 1) x {CFL_X_LAMINAR}"
                )
        elif annular:
            # max U_theta/r = 1 at the inner wall r1 = 1.
            if abs(active - CFL_TH_LAMINAR_TC) > (
                CFL_REL_TOL * CFL_TH_LAMINAR_TC
            ):
                raise AssertionError(
                    f"{name}: CFL_th {active} != {CFL_TH_LAMINAR_TC}"
                )
        else:
            # max |U_x| = 1: at the walls (Couette) or at the
            # y = 0 node (Poiseuille; ny = 27 is odd).
            if abs(active - CFL_X_LAMINAR) > CFL_REL_TOL * CFL_X_LAMINAR:
                raise AssertionError(
                    f"{name}: CFL_x {active} != {CFL_X_LAMINAR}"
                )
        if abs(total - active) > CFL_REL_TOL * active:
            raise AssertionError(
                f"{name}: CFL total {total} != active {active}"
            )


# ── test runner ──────────────────────────────────────────────────────


def run_smoke_test(system: dict, np_count: int, np0: int = 1) -> None:
    """Run a single laminar smoke test (in a fresh directory)."""
    name = system["name"]
    cmd = _build_command(system["args"], np_count, np0)

    with tempfile.TemporaryDirectory(prefix=f"smoke_{name}_") as workdir:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workdir,
        )

        if result.returncode != 0:
            print(f"  FAIL  {name}: exit code {result.returncode}")
            print(result.stdout[-2000:] if result.stdout else "(no stdout)")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
            raise AssertionError(
                f"{name} exited with code {result.returncode}"
            )

        last_err, last_ep = _parse_diagnostics(result.stdout)

        if last_err is None:
            raise AssertionError(f"{name}: could not parse 'err' from output")
        if last_ep is None:
            raise AssertionError(f'{name}: could not parse "E\'" from output')

        if last_err > ERR_THRESHOLD:
            raise AssertionError(
                f"{name}: stepping error {last_err:.3e} > {ERR_THRESHOLD:.0e}"
            )
        if last_ep > EP_THRESHOLD:
            raise AssertionError(
                f"{name}: perturbation energy {last_ep:.3e} "
                f"> {EP_THRESHOLD:.0e}"
            )

        _check_steps_file(Path(workdir), name)

    print(f"  PASS  {name}  (err={last_err:.2e}, E'={last_ep:.2e})")


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Laminar smoke tests for wall-bounded flows",
    )
    parser.add_argument(
        "--np",
        type=int,
        default=1,
        help="Number of devices (uses mpirun when > 1)",
    )
    parser.add_argument(
        "--np0",
        type=int,
        default=1,
        help="np0 mesh axis (wall-normal / kz split)",
    )
    args = parser.parse_args()

    passed = 0
    failed = 0
    for system in SYSTEMS:
        try:
            run_smoke_test(system, args.np, args.np0)
            passed += 1
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            print(f"  FAIL  {system['name']}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
