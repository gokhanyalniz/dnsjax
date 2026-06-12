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
SMOKE_LX = 4.0
SMOKE_N_STEPS = 4  # max_sim_time 0.04 / dt 0.01
# Spectral-grid spacing convention: Delta_x = lx / nx.
CFL_X_LAMINAR = SMOKE_DT * SMOKE_NX / SMOKE_LX
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
    if cylindrical:
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

    for row in rows:
        streamwise = row[col[per_dir[0]]]
        second = row[col[per_dir[1]]]
        third = row[col[per_dir[2]]]
        total = row[col["CFL"]]
        if not all(math.isfinite(v) and v >= 0.0 for v in row):
            raise AssertionError(f"{name}: bad steps.dat row {row}")
        # Perturbation-only columns: roundoff-sized for the
        # laminar state (not exactly zero after the first step).
        if second > CFL_ZERO_TOL or third > CFL_ZERO_TOL:
            raise AssertionError(
                f"{name}: nonzero perturbation CFL in row {row}"
            )
        if cylindrical:
            # max U_z = max (1 - r^2) < 1 on the half-CGL grid
            # (r = 0 is not a node), but only barely below it.
            if not 0.9 * CFL_X_LAMINAR < streamwise < CFL_X_LAMINAR:
                raise AssertionError(
                    f"{name}: CFL_z {streamwise} outside "
                    f"(0.9, 1) x {CFL_X_LAMINAR}"
                )
            if abs(total - streamwise) > CFL_REL_TOL * streamwise:
                raise AssertionError(
                    f"{name}: CFL total {total} != CFL_z {streamwise}"
                )
        else:
            # max |U_x| = 1: at the walls (Couette) or at the
            # y = 0 node (Poiseuille; ny = 27 is odd).
            if abs(streamwise - CFL_X_LAMINAR) > CFL_REL_TOL * CFL_X_LAMINAR:
                raise AssertionError(
                    f"{name}: CFL_x {streamwise} != {CFL_X_LAMINAR}"
                )
            if abs(total - CFL_X_LAMINAR) > CFL_REL_TOL * CFL_X_LAMINAR:
                raise AssertionError(
                    f"{name}: CFL total {total} != {CFL_X_LAMINAR}"
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
