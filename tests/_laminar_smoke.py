r"""The runner of the laminar smoke tests, core and lab.

Each script's table (``tests/test_laminar_smoke.py``,
``lab/tests/test_laminar_smoke.py``) holds its rows and its account of
them; :func:`main` runs one.  Each row runs its system from the laminar
state for a few steps at low resolution, in its own subprocess (the
geometry modules capture global singletons at import time) and its own
temporary directory (so ``stats.dat`` / ``steps.dat`` /
``corrector.dat`` are per-row and the repo's ``parameters.toml`` is
not loaded), with ``--stop.check_laminarization False``: the laminar
``E'`` `$\approx 10^{-32}$` would otherwise trip the default
laminarization check and end the run immediately.  Every row then
meets:

- its ``check`` of the run's output (default :func:`check_perturbation`:
  a stepping error of `$O(10^{-18})$` or less and a perturbation energy
  of `$O(10^{-32})$` or less);
- :func:`check_steps_file` (``steps.dat``, written every step): the CFL
  of the laminar base flow in the default moving frame;
- :func:`check_corrector_file` (``corrector.dat``, written every step):
  ``c = 0`` (a single corrector step) and an error within the row's
  ``err_threshold`` (default roundoff) on every step.

Row keys beyond ``name`` and ``args``: ``cfl_active`` (required),
``step_columns``, ``check``, ``err_threshold``, ``console`` (launch the
installed ``dnsjax`` console script under ``mpirun``) and
``expect_stdout``.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from _live import report, run_live

ERR_PATTERN = re.compile(r"err\s*=\s*([\d.eE+\-]+)")
EP_PATTERN = re.compile(r"E'\s*=\s*([\d.eE+\-]+)")
# ``\b`` + uppercase isolates the E / I / D stats keys from E', Ub_*,
# tau_*, and the lowercase 'e' of scientific notation.
E_PATTERN = re.compile(r"\bE\s*=\s*([\d.eE+\-]+)")
I_PATTERN = re.compile(r"\bI\s*=\s*([\d.eE+\-]+)")
D_PATTERN = re.compile(r"\bD\s*=\s*([\d.eE+\-]+)")

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
# Moving-frame laminar CFL in the grid direction.  phys.u_grid defaults
# to the laminar bulk velocity, so pipe / Poiseuille evolve in a moving
# frame and the active CFL measures dt * max|U - U_grid| * n / l:
#   pipe:        max|(1 - r^2) - 1/2| = 1/2 at the r = 1 wall node.
#   Poiseuille:  max|(1 - y^2) - 2/3| = 2/3 at the y = +/-1 walls.
# Couette keeps U_grid = 0 (max|y| = 1), i.e. the unchanged
# CFL_X_LAMINAR.  Both maxima sit exactly on grid nodes, so (unlike the
# laboratory-frame pipe, whose max between nodes was only range-checkable)
# these are tight equalities.
CFL_GRID_LAMINAR_PIPE = 0.5 * CFL_X_LAMINAR
CFL_GRID_LAMINAR_POISEUILLE = (2.0 / 3.0) * CFL_X_LAMINAR
# Taylor-Couette laminar azimuthal CFL: dt * max(U_theta/r) * nz/(2 pi)
# (theta period is the literal 2*pi).  For re1=100, re2=0, eta=0.5 the
# maximum of U_theta/r is 1, at the inner wall r1 = 1 where
# U_theta(r1) = 1.
CFL_TH_LAMINAR_TC = SMOKE_DT * 1.0 * SMOKE_NZ / (2 * math.pi)
CFL_REL_TOL = 1e-7
CFL_ZERO_TOL = 1e-12  # roundoff-sized perturbation columns

# ── helpers ──────────────────────────────────────────────────────────


def console_script() -> Path | None:
    """The installed ``dnsjax`` console script, or ``None``.

    Looked up next to the running interpreter (``.venv/bin/dnsjax``
    after ``uv sync``), so the smoke children exercise the same
    environment either way.
    """
    script = Path(sys.executable).with_name("dnsjax")
    return script if script.exists() else None


def build_command(
    system_args: list[str],
    np_count: int,
    np0: int = 1,
    platform: str = "cpu",
    console: bool = False,
) -> list[str]:
    """Build the subprocess command for a single system.

    *platform* is forwarded as ``--dist.platform`` so the whole suite
    can run on a GPU (``--dist.platform cuda``); with ``--np N`` this is
    ``mpirun -np N`` processes, one device each.  With *console* the
    child is the installed ``dnsjax`` console script instead of
    ``python -m dnsjax`` (both run :func:`dnsjax.__main__.main`).
    """
    entry = (
        [str(console_script())]
        if console
        else [sys.executable, "-m", "dnsjax"]
    )
    base = [
        "mpirun",
        "-np",
        str(np_count),
        *entry,
        "--dist.platform",
        platform,
        "--dist.np0",
        str(np0),
        "--dist.np1",
        str(np_count // np0),
        "--outs.it_steps",
        "1",
        # Corrector diagnostic every step; it_error_check <= it_corrector.
        "--outs.it_corrector",
        "1",
        "--outs.it_error_check",
        "1",
        # Laminarization check off: laminar runs sit at E' ~ 1e-32.
        "--stop.check_laminarization",
        "False",
    ]
    return base + system_args


def parse_diagnostics(
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


def check_steps_file(workdir: Path, system: dict) -> None:
    """Validate ``steps.dat`` against the row's laminar CFL.

    The per-direction columns follow the flow's geometry (from the
    registry): ``CFL_z/r/th`` for the cylindrical and annular flows,
    ``CFL_x/y/z`` otherwise, plus the row's ``step_columns``.  The
    laminar base flow is streamwise in the Cartesian and cylindrical
    geometries and azimuthal in the annular one, so that column is the
    active one and every other is roundoff-sized.  The row's
    ``cfl_active`` is the active column's exact value (the laminar
    maxima sit on grid nodes, so it is a tight equality), or ``None``
    for a flow with no simple analytic laminar CFL, whose active column
    must then be positive and constant across steps (near-steady).
    """
    from dnsjax.flows.registry import spec_for

    name = system["name"]
    args = system["args"]
    geometry = spec_for(args[args.index("--phys.system") + 1]).geometry
    steps_file = workdir / "steps.dat"
    if not steps_file.exists():
        raise AssertionError(f"{name}: steps.dat was not written")

    lines = [ln for ln in steps_file.read_text().splitlines() if ln.strip()]
    header = lines[0].lstrip("#").split()

    # Dicts returned through ``jit`` are canonicalized to sorted
    # key order, so compare as sets and index columns by name
    # ("t" is always written first by ``_flush_stats``).
    if geometry in ("cylindrical", "annular"):
        per_dir = ("CFL_z", "CFL_r", "CFL_th")
    else:
        per_dir = ("CFL_x", "CFL_y", "CFL_z")
    extra = system.get("step_columns", ())
    expected_cols = {"t", "dt", "CFL", *per_dir, *extra}
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

    active_i = 2 if geometry == "annular" else 0
    expected_active = system["cfl_active"]

    active_vals = []
    for row in rows:
        if not all(math.isfinite(v) and v >= 0.0 for v in row):
            raise AssertionError(f"{name}: bad steps.dat row {row}")
        vals = [row[col[d]] for d in per_dir]
        total = row[col["CFL"]]
        # Fixed-dt run: the dt column must hold --step.dt in every
        # row (relative tolerance covers the single-precision store
        # + text round-trip).
        if abs(row[col["dt"]] - SMOKE_DT) > 1e-6 * SMOKE_DT:
            raise AssertionError(
                f"{name}: dt column {row[col['dt']]} != {SMOKE_DT}"
            )
        active = vals[active_i]
        active_vals.append(active)
        # Off-axis columns: roundoff-sized for the laminar state (not
        # exactly zero after the first step).
        roundoff = [v for i, v in enumerate(vals) if i != active_i]
        if any(v > CFL_ZERO_TOL for v in roundoff):
            raise AssertionError(f"{name}: nonzero off-axis CFL in row {row}")
        if expected_active is not None and (
            abs(active - expected_active) > CFL_REL_TOL * expected_active
        ):
            raise AssertionError(
                f"{name}: active CFL {active} != {expected_active}"
            )
        if abs(total - active) > CFL_REL_TOL * active:
            raise AssertionError(
                f"{name}: CFL total {total} != active {active}"
            )

    if expected_active is None:
        # Near-steady: the active CFL positive and constant across steps.
        a_min, a_max = min(active_vals), max(active_vals)
        if a_min <= 0.0:
            raise AssertionError(f"{name}: non-positive active CFL {a_min}")
        if a_max - a_min > CFL_REL_TOL * a_max:
            raise AssertionError(
                f"{name}: active CFL not constant ([{a_min}, {a_max}])"
            )


def check_corrector_file(
    workdir: Path, name: str, err_threshold: float = ERR_THRESHOLD
) -> None:
    """Validate ``corrector.dat`` for a laminar run.

    The laminar state converges in a single corrector step, so for every
    recorded step the iteration count ``c`` (iterations *beyond* the
    first) is 0 and the final corrector error is roundoff-sized.  With
    ``it_corrector = 1`` there is one row per step, the first at
    ``t = 0`` (same cadence/format as ``steps.dat``).  ``err_threshold``
    is relaxed for a flow whose state carries magnitudes that put the
    corrector-error floor at FD truncation rather than roundoff (the
    row's ``err_threshold``).
    """
    corr_file = workdir / "corrector.dat"
    if not corr_file.exists():
        raise AssertionError(f"{name}: corrector.dat was not written")

    lines = [ln for ln in corr_file.read_text().splitlines() if ln.strip()]
    header = lines[0].lstrip("#").split()
    if header[0] != "t" or set(header) != {"t", "c", "error"}:
        raise AssertionError(
            f"{name}: corrector.dat header {header} != [t, c, error]"
        )
    col = {n: i for i, n in enumerate(header)}

    rows = [[float(v) for v in ln.split()] for ln in lines[1:]]
    if len(rows) != SMOKE_N_STEPS:
        raise AssertionError(
            f"{name}: expected {SMOKE_N_STEPS} corrector.dat rows, "
            f"got {len(rows)}"
        )
    if rows[0][col["t"]] != 0.0:
        raise AssertionError(
            f"{name}: first corrector.dat row at t={rows[0][col['t']]}, not 0"
        )
    for row in rows:
        c_val = row[col["c"]]
        err_val = row[col["error"]]
        if c_val != 0.0:
            raise AssertionError(
                f"{name}: corrector count c={c_val} != 0 (row {row})"
            )
        if not (math.isfinite(err_val) and 0.0 <= err_val <= err_threshold):
            raise AssertionError(
                f"{name}: corrector error {err_val} not in "
                f"[0, {err_threshold:.0e}] (row {row})"
            )


def check_perturbation(stdout: str, name: str) -> tuple[float, float]:
    """A base-flow run: roundoff stepping error and perturbation energy.

    The default ``check`` of a row; returns ``(last_err, last_ep)`` for
    the PASS summary.
    """
    last_err, last_ep = parse_diagnostics(stdout)
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
            f"{name}: perturbation energy {last_ep:.3e} > {EP_THRESHOLD:.0e}"
        )
    return last_err, last_ep


# ── console checks ───────────────────────────────────────────────────


def check_console_help() -> None:
    """``dnsjax --help`` must print usage, exit 0, and touch nothing.

    Regression guard for the pre-refactor console script, which had no
    reachable CLI parse and would have silently started a default
    simulation on ``--help``.  Runs without ``mpirun`` (the help path
    exits before any JAX / distributed init).
    """
    script = console_script()
    if script is None:
        print("  SKIP  console-help: console script not installed")
        return
    with tempfile.TemporaryDirectory(prefix="smoke_help_") as workdir:
        result = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=workdir,
            # Deterministic plain help: a FORCE_COLOR in the host env
            # (agent harnesses set one) breaks the substring checks.
            env={**os.environ, "NO_COLOR": "1"},
        )
        if result.returncode != 0:
            raise AssertionError(
                f"console-help: exit code {result.returncode}"
            )
        if "usage: dnsjax" not in result.stdout:
            raise AssertionError("console-help: no usage text in stdout")
        leftovers = sorted(p.name for p in Path(workdir).iterdir())
        if leftovers:
            raise AssertionError(f"console-help: created files {leftovers}")
    print("  PASS  console-help (usage printed, no side effects)")


def check_bare_launch(row_args: list[str]) -> None:
    """A one-process run needs no ``mpirun``, and says so when it does.

    The solver only stands up JAX's distributed runtime when it is one
    of several processes, so a lone run needs no launcher, no
    coordinator and no MPI on the machine at all.  This walks the
    whole path -- console script, no ``mpirun`` anywhere -- and then
    checks the other half of the contract: one process may not hold
    several *CPU* devices (that is oversubscription, not
    parallelism), and asking for it must fail early, naming the launch
    that works, rather than at the device count deep in ``sharding``.
    """
    script = console_script()
    if script is None:
        print("  SKIP  bare-launch: console script not installed")
        return
    env = {**os.environ, "NO_COLOR": "1"}
    args = row_args + ["--stop.check_laminarization", "False"]

    with tempfile.TemporaryDirectory(prefix="smoke_bare_") as workdir:
        result = subprocess.run(
            [str(script), *args],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workdir,
            env=env,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"bare-launch: exit code {result.returncode}\n"
                f"{result.stdout[-2000:]}"
            )
        last_err, last_ep = parse_diagnostics(result.stdout)
        if last_err is None or last_err > ERR_THRESHOLD:
            raise AssertionError(
                f"bare-launch: stepping error {last_err} > {ERR_THRESHOLD:.0e}"
            )
        if last_ep is None or last_ep > EP_THRESHOLD:
            raise AssertionError(
                f"bare-launch: perturbation energy {last_ep} "
                f"> {EP_THRESHOLD:.0e}"
            )

    with tempfile.TemporaryDirectory(prefix="smoke_bare_cpu_") as workdir:
        refused = subprocess.run(
            [str(script), *args, "--dist.np0", "2", "--dist.np1", "2"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=workdir,
            env=env,
        )
        message = refused.stdout + refused.stderr
        if refused.returncode == 0:
            raise AssertionError(
                "bare-launch: 4 CPU devices in one process was allowed"
            )
        if "mpirun -np 4" not in message:
            raise AssertionError(
                f"bare-launch: refusal does not name the working "
                f"launch: {message[-500:]}"
            )
    print(f"  PASS  bare-launch  (err={last_err:.2e}, E'={last_ep:.2e})")


# ── test runner ──────────────────────────────────────────────────────


def run_smoke_test(
    system: dict, np_count: int, np0: int = 1, platform: str = "cpu"
) -> None:
    """Run a single laminar smoke test (in a fresh directory)."""
    name = system["name"]
    console = system.get("console", False)
    if console and console_script() is None:
        print(f"  SKIP  {name}: console script not installed (uv sync)")
        return
    cmd = build_command(system["args"], np_count, np0, platform, console)

    with tempfile.TemporaryDirectory(prefix=f"smoke_{name}_") as workdir:
        result = run_live(cmd, timeout=300, cwd=workdir)

        if result.returncode != 0:
            print(f"  FAIL  {name}: exit code {result.returncode}")
            print(result.stdout[-2000:] if result.stdout else "(no stdout)")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
            raise AssertionError(
                f"{name} exited with code {result.returncode}"
            )

        expect = system.get("expect_stdout")
        if callable(expect):
            expect = expect(np_count // np0)
            if expect is None and "rounded" in result.stdout:
                raise AssertionError(
                    f"{name}: unexpected padded-size rounding note"
                )
        if expect is not None and result.stdout.count(expect) != 1:
            raise AssertionError(
                f"{name}: expected exactly one {expect!r} in stdout "
                f"(got {result.stdout.count(expect)})"
            )

        check = system.get("check", check_perturbation)
        last_err, last_ep = check(result.stdout, name)
        check_steps_file(Path(workdir), system)
        check_corrector_file(
            Path(workdir),
            name,
            err_threshold=system.get("err_threshold", ERR_THRESHOLD),
        )

    print(f"  PASS  {name}  (err={last_err:.2e}, E'={last_ep:.2e})")


# ── main ─────────────────────────────────────────────────────────────


def main(systems: list[dict], *, console_checks: bool) -> None:
    """Run *systems* and exit with the verdict.

    With *console_checks*, first the ``dnsjax --help`` preflight and the
    bare launch (on the first row's arguments).
    """
    sys.stdout.reconfigure(line_buffering=True)
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
    parser.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend forwarded to each `python -m dnsjax` child "
        "(default cpu).  Use cuda to run the suite on GPU(s).",
    )
    args = parser.parse_args()

    print(
        f"Laminar smoke tests on platform '{args.platform}' via "
        f"mpirun -np {args.np} (np0={args.np0}, "
        f"np1={args.np // args.np0}); each child prints its own device "
        "banner.",
        flush=True,
    )

    passed = 0
    failures: list[tuple[str, str]] = []
    checks = []
    if console_checks:
        checks = [
            ("console-help", check_console_help),
            ("bare-launch", lambda: check_bare_launch(systems[0]["args"])),
        ]
    for label, fn in checks:
        try:
            fn()
            passed += 1
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            print(f"  FAIL  {label}: {exc}")
            failures.append((label, str(exc)))
    for system in systems:
        try:
            run_smoke_test(system, args.np, args.np0, args.platform)
            passed += 1
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            print(f"  FAIL  {system['name']}: {exc}")
            failures.append((system["name"], str(exc)))

    sys.exit(report(passed, failures))
