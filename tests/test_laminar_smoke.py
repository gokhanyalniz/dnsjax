"""Laminar smoke tests for all wall-bounded flows.

Runs each wall-bounded system from its laminar state for a few
time steps at low resolution, verifying that:

- The corrector converges in a single iteration.
- The stepping error is `$O(10^{-18})$` or less.
- The perturbation energy stays at `$O(10^{-32})$` or less.
- The CFL diagnostic (``steps.dat``, written every
  ``it_steps = 1`` steps) matches the laminar base flow in the
  (default) moving frame of reference: the active grid-direction
  column is `$\\Delta t \\, \\max|U - U_{grid}| \\, n / l$` and the
  remaining columns are roundoff-sized.  ``phys.u_grid`` defaults to
  the laminar bulk velocity, so the active CFL is `$\\max|U_x| = 1$`
  for plane-Couette (`$U_{grid} = 0$`), `$2/3$` of it for
  plane-Poiseuille (`$U_{grid} = 2/3$`, max at the walls), `$1/2$` of
  it for the pipe (`$U_{grid} = 1/2$`, max at the `$r = 1$` wall), and
  the azimuthal `$\\max|U_\\theta/r| = 1$` for Taylor-Couette
  (`$U_{grid} = 0$`).
- The corrector diagnostic (``corrector.dat``, written every
  ``it_corrector = 1`` steps) records ``c = 0`` (a single corrector
  step) and a roundoff-sized error for every step.

Dean flow is force-driven and integrates the **total** field, so it is
checked differently: started from the *analytical* laminar profile, its
``E'`` (the perturbation kinetic energy of the deviation from that
profile) stays tiny, the corrector still converges (``err``
`$O(10^{-14})$`), the energy balance `$I \\approx D$` holds, and the
total energy is near-steady.  Its azimuthal ``CFL_th`` is the active
column (radial / axial are roundoff).

Each system is tested in a separate subprocess because the
geometry modules capture global singletons at import time.
Each subprocess runs in its own temporary directory, so
``stats.dat`` / ``steps.dat`` / ``corrector.dat`` are per-system
(and the repo's ``parameters.toml`` is not loaded; the smoke runs
use the parameter-model defaults plus the CLI arguments).

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
    {
        "name": "dean",
        "args": [
            "--phys.system",
            "dean",
            "--geo.eta",
            "0.5",
            "--phys.re",
            "100",
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
        "name": "dean-block-spanwise",
        "args": [
            "--phys.system",
            "dean",
            "--geo.eta",
            "0.5",
            "--phys.re",
            "100",
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
        # Viscoelastic (sPTT) Dean flow: total-field 9-component state.
        # At epsilon = kappa = 0 the analytical laminar pair (azimuthal
        # velocity + pointwise sPTT-equilibrium conformation) is the
        # *exact* discrete steady state, so E' (velocity deviation) stays
        # ~1e-18 and the corrector converges in a single step.  A modest
        # wi = el (=> Re = 1) keeps the conformation magnitude O(10) so
        # the corrector error is FD-truncation-, not overflow-, limited.
        # Checked differently from the perturbation flows (own branch):
        # near-steady energy and the polymer energy balance I = D_s - W_p.
        "name": "viscoelastic-dean",
        "args": [
            "--phys.system",
            "viscoelastic-dean",
            "--phys.epsilon",
            "0",
            "--phys.kappa",
            "0",
            "--phys.wi",
            "5",
            "--phys.el",
            "5",
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

# Dean-flow diagnostics (total-field, analytical near-steady laminar
# profile).  E' (deviation kinetic energy) is matched by ``EP_PATTERN``
# above; ``\b`` + uppercase isolates the E / I / D keys from E', Ub_*,
# tau_*, and the lowercase 'e' of scientific notation.
E_PATTERN = re.compile(r"\bE\s*=\s*([\d.eE+\-]+)")
I_PATTERN = re.compile(r"\bI\s*=\s*([\d.eE+\-]+)")
D_PATTERN = re.compile(r"\bD\s*=\s*([\d.eE+\-]+)")

# E' is the deviation energy ||u - U_lam||^2 / 2 from the analytical
# laminar Dean profile.  Observed ~2e-17 at ny=27, fd_order=4 (the old
# norm deviation ~6e-9, squared and halved).  The energy balance
# |I-D|/D (~8e-6) and total-energy drift (~3e-8) have generous margins,
# still catching a wrong forcing sign/magnitude (O(1) departures).
DEAN_EP_THRESHOLD = 5e-13
DEAN_IB_TOL = 1e-3
DEAN_E_DRIFT_TOL = 1e-4

# Viscoelastic-dean diagnostics (total-field, 9-component).  The
# conformation carries magnitude O(10) (TrC ~ 70 at wi = 5), so the
# corrector error and E' floor sit at FD truncation (~1e-10 / ~1e-17),
# well above the perturbation flows' roundoff but far below the
# corrector tolerance.  The polymer energy balance is I = D_s - W_p
# (solvent dissipation minus polymer work; W_p < 0 as polymers dissipate).
DS_PATTERN = re.compile(r"D_s\s*=\s*([\d.eE+\-]+)")
WP_PATTERN = re.compile(r"W_p\s*=\s*([\d.eE+\-]+)")
VE_ERR_THRESHOLD = 1e-7
VE_EP_THRESHOLD = 1e-12
VE_BALANCE_TOL = 5e-3
VE_E_DRIFT_TOL = 1e-4

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
# lab-frame pipe, whose max between nodes was only range-checkable)
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
    # Viscoelastic-dean is a force-driven annular (total-field) flow: it
    # shares Dean's near-steady azimuthal-CFL structure but additionally
    # writes a ``TrC_max`` column (the max conformation trace).
    viscoelastic = name.startswith("viscoelastic")
    dean = name.startswith("dean") or viscoelastic
    if cylindrical or annular or dean:
        per_dir = ("CFL_z", "CFL_r", "CFL_th")
    else:
        per_dir = ("CFL_x", "CFL_y", "CFL_z")
    expected_cols = {"t", "CFL", *per_dir}
    if viscoelastic:
        expected_cols = expected_cols | {"TrC_max"}
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
    # azimuthal CFL_th column (2) for the annular (Taylor-Couette /
    # Dean) base flow.
    active_i = 2 if (annular or dean) else 0

    # Expected active CFL = dt * max|U - U_grid| * n / l for the
    # laminar base flow in the (default) moving frame: pipe at the
    # r = 1 wall node, Poiseuille and Couette at the walls, and
    # Taylor-Couette (U_grid = 0) at the inner wall via
    # max|U_theta/r| = 1 -- all attained exactly at grid nodes, so
    # tight equalities.  Dean has no simple analytic laminar CFL, so
    # its active column is instead checked to be positive and constant
    # across steps (near-steady) below.
    if dean:
        expected_active = None
    elif cylindrical:
        expected_active = CFL_GRID_LAMINAR_PIPE
    elif annular:
        expected_active = CFL_TH_LAMINAR_TC
    elif name.startswith("plane-poiseuille"):
        expected_active = CFL_GRID_LAMINAR_POISEUILLE
    else:  # plane-couette (U_grid = 0)
        expected_active = CFL_X_LAMINAR

    active_vals = []
    for row in rows:
        if not all(math.isfinite(v) and v >= 0.0 for v in row):
            raise AssertionError(f"{name}: bad steps.dat row {row}")
        vals = [row[col[d]] for d in per_dir]
        total = row[col["CFL"]]
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

    if dean:
        # Near-steady: azimuthal CFL positive and constant across steps.
        a_min, a_max = min(active_vals), max(active_vals)
        if a_min <= 0.0:
            raise AssertionError(f"{name}: non-positive azimuthal CFL {a_min}")
        if a_max - a_min > CFL_REL_TOL * a_max:
            raise AssertionError(
                f"{name}: azimuthal CFL not constant ([{a_min}, {a_max}])"
            )


def _check_corrector_file(
    workdir: Path, name: str, err_threshold: float = ERR_THRESHOLD
) -> None:
    """Validate ``corrector.dat`` for a laminar run.

    The laminar state converges in a single corrector step, so for every
    recorded step the iteration count ``c`` (iterations *beyond* the
    first) is 0 and the final corrector error is roundoff-sized.  With
    ``it_corrector = 1`` there is one row per step, the first at
    ``t = 0`` (same cadence/format as ``steps.dat``).  ``err_threshold``
    is relaxed for the viscoelastic total field, whose large
    conformation magnitude puts the corrector-error floor at FD
    truncation rather than roundoff.
    """
    corr_file = workdir / "corrector.dat"
    if not corr_file.exists():
        raise AssertionError(f"{name}: corrector.dat was not written")

    lines = [ln for ln in corr_file.read_text().splitlines() if ln.strip()]
    header = lines[0].split()
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


def _check_dean(stdout: str, name: str) -> tuple[float, float]:
    """Validate a Dean (analytical near-steady laminar) run.

    Dean integrates the total field; the error metric is ``E'``, the
    perturbation kinetic energy of the deviation from the analytical
    laminar profile, alongside corrector convergence (``err``), the
    energy balance ``I ~= D``, and a near-steady total energy ``E``.
    Returns ``(last_err, last_ep)`` for the PASS summary.
    """
    last_err: float | None = None
    ep_vals: list[float] = []
    e_vals: list[float] = []
    i_vals: list[float] = []
    d_vals: list[float] = []
    for line in stdout.splitlines():
        m = ERR_PATTERN.search(line)
        if m:
            last_err = float(m.group(1))
        for pat, acc in (
            (EP_PATTERN, ep_vals),
            (E_PATTERN, e_vals),
            (I_PATTERN, i_vals),
            (D_PATTERN, d_vals),
        ):
            m = pat.search(line)
            if m:
                acc.append(float(m.group(1)))

    if last_err is None or not (ep_vals and e_vals and i_vals and d_vals):
        raise AssertionError(f"{name}: could not parse Dean diagnostics")

    if last_err > ERR_THRESHOLD:
        raise AssertionError(
            f"{name}: stepping error {last_err:.3e} > {ERR_THRESHOLD:.0e}"
        )

    last_ep = ep_vals[-1]
    if last_ep > DEAN_EP_THRESHOLD:
        raise AssertionError(
            f"{name}: deviation energy from laminar {last_ep:.3e} "
            f"> {DEAN_EP_THRESHOLD:.0e}"
        )

    last_i, last_d = i_vals[-1], d_vals[-1]
    if last_i <= 0.0 or last_d <= 0.0:
        raise AssertionError(
            f"{name}: non-positive I={last_i:.3e} or D={last_d:.3e}"
        )
    if abs(last_i - last_d) > DEAN_IB_TOL * last_d:
        raise AssertionError(
            f"{name}: energy balance off: I={last_i:.6e}, D={last_d:.6e}"
        )

    if abs(e_vals[-1] - e_vals[0]) > DEAN_E_DRIFT_TOL * e_vals[0]:
        raise AssertionError(
            f"{name}: energy drift {e_vals[0]:.6e} -> {e_vals[-1]:.6e}"
        )

    return last_err, last_ep


def _check_viscoelastic_dean(stdout: str, name: str) -> tuple[float, float]:
    r"""Validate a viscoelastic-dean laminar run (`$\epsilon=\kappa=0$`).

    The analytical laminar pair is the exact discrete steady state, so
    ``E'`` (velocity deviation energy) is FD-truncation-tiny and the
    corrector converges; additionally the polymer energy balance
    ``I = D_s - W_p`` holds and the total energy is near-steady.  Returns
    ``(last_err, last_ep)`` for the PASS summary.
    """
    last_err: float | None = None
    ep_vals: list[float] = []
    e_vals: list[float] = []
    i_vals: list[float] = []
    ds_vals: list[float] = []
    wp_vals: list[float] = []
    for line in stdout.splitlines():
        m = ERR_PATTERN.search(line)
        if m:
            last_err = float(m.group(1))
        for pat, acc in (
            (EP_PATTERN, ep_vals),
            (E_PATTERN, e_vals),
            (I_PATTERN, i_vals),
            (DS_PATTERN, ds_vals),
            (WP_PATTERN, wp_vals),
        ):
            m = pat.search(line)
            if m:
                acc.append(float(m.group(1)))

    if last_err is None or not (
        ep_vals and e_vals and i_vals and ds_vals and wp_vals
    ):
        raise AssertionError(
            f"{name}: could not parse viscoelastic diagnostics"
        )

    if last_err > VE_ERR_THRESHOLD:
        raise AssertionError(
            f"{name}: stepping error {last_err:.3e} > {VE_ERR_THRESHOLD:.0e}"
        )

    last_ep = ep_vals[-1]
    if last_ep > VE_EP_THRESHOLD:
        raise AssertionError(
            f"{name}: deviation energy from laminar {last_ep:.3e} "
            f"> {VE_EP_THRESHOLD:.0e}"
        )

    # Polymer energy balance I = D_s - W_p at the near-steady state.
    last_i, last_ds, last_wp = i_vals[-1], ds_vals[-1], wp_vals[-1]
    balance = last_ds - last_wp
    if last_i <= 0.0:
        raise AssertionError(f"{name}: non-positive I={last_i:.3e}")
    if abs(last_i - balance) > VE_BALANCE_TOL * abs(last_i):
        raise AssertionError(
            f"{name}: energy balance off: I={last_i:.6e}, "
            f"D_s - W_p={balance:.6e}"
        )

    if abs(e_vals[-1] - e_vals[0]) > VE_E_DRIFT_TOL * e_vals[0]:
        raise AssertionError(
            f"{name}: energy drift {e_vals[0]:.6e} -> {e_vals[-1]:.6e}"
        )

    return last_err, last_ep


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

        if name.startswith("viscoelastic"):
            last_err, last_ep = _check_viscoelastic_dean(result.stdout, name)
            _check_steps_file(Path(workdir), name)
            _check_corrector_file(
                Path(workdir), name, err_threshold=VE_ERR_THRESHOLD
            )
            print(f"  PASS  {name}  (err={last_err:.2e}, E'={last_ep:.2e})")
            return

        if name.startswith("dean"):
            last_err, last_ep = _check_dean(result.stdout, name)
            _check_steps_file(Path(workdir), name)
            _check_corrector_file(Path(workdir), name)
            print(f"  PASS  {name}  (err={last_err:.2e}, E'={last_ep:.2e})")
            return

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
        _check_corrector_file(Path(workdir), name)

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
