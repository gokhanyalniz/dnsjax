#!/usr/bin/env python3
r"""The applied mean-mode driving recorded in ``stats.dat``.

Under ``phys.driving = "constant_bulk_velocity"`` (pipe,
plane-Poiseuille) or ``phys.block_mean_spanwise_velocity`` (the
Cartesian and annular families) the solver holds a mean velocity
component fixed with a rank-1 correction, and the body force that
correction represents -- `$\Pi' = -\partial p'/\partial s$`, the
**applied forcing**, positive when it accelerates the flow -- is
recorded as the last ``stats.dat`` column (``-dPds'`` / ``-dPdn'`` /
``-dPdz'``).  It is a *step* quantity threaded out of the jitted
corrector, because it is the bulk of the **pre**-correction solve and
so is not recoverable from the accepted state, whose bulk is zero by
construction.

Four checks, in increasing strength:

- **laminar**: from `$u' = 0$` the applied forcing is *exactly* zero,
  for every flow that can apply one;
- **constraint**: the quantity being held (perturbation bulk velocity)
  really is zero to machine precision, so the recorded number is the
  price of an enforced constraint rather than of a drifting one;
- **identity** (the decisive plumbing guard): re-running the converged
  corrector iterate with the driving switched *off* reproduces the
  uncorrected solve `$u_\mathrm{unc}$`, and
  `$-\mathrm{bulk}(u_\mathrm{unc})\,H_\mathrm{bulk}^{-1}$` equals the
  recorded value to machine precision.  This pins the reported number
  to the one the corrector applied -- not to a re-derivation of it;
- **wall shear**: on a *non-laminar* run the recorded value agrees with
  the independent wall-shear inference
  `$\Pi' = -\nu(\tau_t - \tau_b)/2$` (pipe: `$-2\nu\tau_z$`), which
  follows from the mean-mode momentum balance once the constraint
  holds the bulk fixed.  The two are **not** identical at finite
  resolution: they differ by the mean-mode bulk of the discrete
  nonlinear term, a wall-normal truncation residual the continuum
  identity assumes away.  Measured for plane-Poiseuille at
  ``nx = nz = 16``, ``dt = 0.005``, smooth random IC:

  ===========  =========  =============
  fixed t=0.1  ny         relative gap
  ===========  =========  =============
               33         6.5
               65         0.14
               129        0.040
  ===========  =========  =============

  ===========  =========  =============
  fixed ny=65  t          relative gap
  ===========  =========  =============
               0.1        0.14
               0.5        0.031
               2.0        0.0035
  ===========  =========  =============

  -- converging on both axes, so the check runs at a resolved ``ny``
  on a developed field and the gap doubles as an under-resolution
  diagnostic.  Do not tighten it without re-measuring the table.

Run directly::

    uv run python tests/test_driving.py
    uv run python tests/test_driving.py --unit-only   # skip the CLI runs
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent
DNSJAX = REPO / ".venv" / "bin" / "dnsjax"

# Flows that can apply a mean-mode driving, with the knob that turns it
# on and the column it then writes.  ``pipe``/``plane-poiseuille`` drive
# the streamwise/axial direction to hold the bulk; the others block the
# undriven one (spanwise for Cartesian, **axial** for the annulus).
LAMINAR_CASES = [
    ("plane-poiseuille", {"driving": "constant_bulk_velocity"}, "-dPds'"),
    ("pipe", {"driving": "constant_bulk_velocity"}, "-dPdz'"),
    ("plane-couette", {"block_mean_spanwise_velocity": True}, "-dPdn'"),
    ("taylor-couette", {"block_mean_spanwise_velocity": True}, "-dPdz'"),
]

# The wall-shear cross-check: resolved ny, developed field (see the
# module docstring's tables for why these numbers and this tolerance).
WS_TOL = 2e-2
WS_CASES = [
    ("plane-poiseuille", "-dPds'"),
    ("pipe", "-dPdz'"),
]


def _cfg(
    system: str, phys: dict, ny: int = 17, step: dict | None = None
) -> None:
    """Configure the singletons for *system* before importing JAX."""
    step = step or {}
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    p: dict = {"system": system, **phys}
    geo: dict = {}
    if system == "taylor-couette":
        p.update(re1=100.0, re2=0.0)
        geo["eta"] = 0.5
    else:
        p["re"] = 100.0
    if system in ("plane-poiseuille", "plane-couette"):
        geo.update(lx=4.0, lz=4.0)
    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": "cpu"},
            phys=p,
            geo=geo,
            res={
                "nx": 8,
                "ny": ny,
                "nz": 8,
                "fd_order": 4,
                "double_precision": True,
            },
            step={"dt": 0.005, **step},
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)


def _flow_module(system: str):
    import importlib

    from dnsjax.flows.registry import spec_for

    return importlib.import_module(spec_for(system).flow_module)


# ── in-process checks (one subprocess per configuration) ─────────────


def _worker_laminar(system: str, knob: str, column: str) -> int:
    phys = (
        {"driving": "constant_bulk_velocity"}
        if knob == "driving"
        else {"block_mean_spanwise_velocity": True}
    )
    _cfg(system, phys)
    import jax.numpy as jnp

    mod = _flow_module(system)
    state = mod.init_state()
    to_solver = getattr(mod, "to_solver_basis", lambda s: s)
    *_, aux = mod.predict_and_fully_correct(jnp.copy(to_solver(state)))

    ok = list(aux) == [column]
    print(
        f"  {'PASS' if ok else 'FAIL'}: {system} column is {column!r} "
        f"(got {list(aux)})"
    )
    val = float(aux[column]) if ok else float("nan")
    exact = ok and val == 0.0
    print(
        f"  {'PASS' if exact else 'FAIL'}: {system} laminar driving is "
        f"exactly zero (got {val!r})"
    )

    # The wall-shear inference must agree there too, and exactly: a
    # zero state has zero wall shear.
    inferred = mod.get_driving(state)
    same = list(inferred) == [column] and float(inferred[column]) == 0.0
    print(
        f"  {'PASS' if same else 'FAIL'}: {system} get_driving matches at "
        f"t0 (got {[(k, float(v)) for k, v in inferred.items()]})"
    )
    return 0 if (ok and exact and same) else 1


def _worker_identity(system: str) -> int:
    r"""The recorded value **is** the applied one, to machine precision.

    Steps a random state with the driving on, then re-runs the *same*
    corrector call with the driving switched off -- a fresh trace, since
    the branch is a trace-time ``params`` read -- to recover the
    uncorrected solve, and checks
    `$-\mathrm{bulk}(u_\mathrm{unc})\,H_\mathrm{bulk}^{-1}$` against it.
    """
    # Drive the corrector to its **fixed point**: the recorded value
    # comes from the iteration that produced ``u1``, so re-deriving it
    # from ``u1`` is one iteration ahead, and the two then agree only to
    # the corrector's remaining error (measured 2e-5 / 6e-4 at the
    # default tolerance).  At the fixed point the identity is exact.
    _cfg(
        system,
        {"driving": "constant_bulk_velocity"},
        ny=27,
        step={
            "corrector_tolerance": 1e-14,
            "max_corrector_iterations": 200,
        },
    )
    import jax.numpy as jnp

    from dnsjax.geometries.wall_bounded._base import extract_mean_mode
    from dnsjax.ic.random_field import generate_random_state
    from dnsjax.parameters import params

    mod = _flow_module(system)
    flow = mod.flow
    if system == "pipe":
        from dnsjax.geometries.wall_bounded import cylindrical as geo

        column = "-dPdz'"
    else:
        from dnsjax.geometries.wall_bounded import cartesian as geo

        column = "-dPds'"
    to_solver = getattr(mod, "to_solver_basis", lambda s: s)

    u0 = to_solver(generate_random_state(0.1, 0.6, 11))
    u1, err, num_c, aux = mod.predict_and_fully_correct(jnp.copy(u0))
    recorded = float(aux[column])

    rhs_n = geo._get_rhs(u0, geo.fourier, flow)
    rhs_c = geo._get_rhs(u1, geo.fourier, flow)
    # Switch the driving off and re-trace: the corrector then returns
    # the *uncorrected* solve the recorded value was formed from.
    params.phys.driving = "constant_pressure_gradient"
    u_unc, _corr, empty = geo._correct(u0, u1, rhs_n, rhs_c, geo.fourier, flow)
    params.phys.driving = "constant_bulk_velocity"

    mean = extract_mean_mode(
        mod.from_solver_basis(u_unc)
        if hasattr(mod, "from_solver_basis")
        else u_unc
    ).real
    if system == "pipe":
        bulk = 2 * float(jnp.dot(flow.y_weights, mean[0]))
    else:
        from dnsjax.parameters import derived_params

        mean_s = (
            mean[0] * derived_params.cos_tilt
            + mean[2] * derived_params.sin_tilt
        )
        bulk = float(jnp.dot(flow.y_weights, mean_s)) / 2
    expected = -bulk * float(flow.H_bulk_inv)

    scale = max(abs(expected), abs(recorded), 1e-300)
    rel = abs(recorded - expected) / scale
    ok = empty == {} and rel < 1e-9 and abs(recorded) > 1e-14
    print(
        f"  {'PASS' if ok else 'FAIL'}: {system} recorded == applied  "
        f"(recorded {recorded:+.8e}, re-derived {expected:+.8e}, "
        f"corrector {int(num_c)} it, err {float(err):.1e}, "
        f"rel {rel:.2e})"
    )

    # And the constraint it buys: the accepted bulk is machine zero.
    mean1 = extract_mean_mode(
        mod.from_solver_basis(u1) if hasattr(mod, "from_solver_basis") else u1
    ).real
    if system == "pipe":
        bulk1 = 2 * float(jnp.dot(flow.y_weights, mean1[0]))
    else:
        bulk1 = float(jnp.dot(flow.y_weights, mean1[0])) / 2
    tight = abs(bulk1) < 1e-16
    print(
        f"  {'PASS' if tight else 'FAIL'}: {system} perturbation bulk is "
        f"machine zero (|Ub'| = {abs(bulk1):.2e})"
    )
    return 0 if (ok and tight) else 1


# ── CLI checks ───────────────────────────────────────────────────────


def _cli(
    tmp: str, system: str, extra: list[str]
) -> subprocess.CompletedProcess:
    cmd = [str(DNSJAX), "--phys.system", system, *extra]
    env = dict(os.environ, NO_COLOR="1")
    return run_live(cmd, timeout=1200, cwd=tmp, env=env)


def _columns(path: Path) -> list[str]:
    return path.read_text().splitlines()[0].lstrip("#").split()


def _check_wall_shear(system: str, column: str) -> str | None:
    """Recorded vs wall-shear inference on a developed, resolved run."""
    with tempfile.TemporaryDirectory() as tmp:
        res = ["--res.nx", "16", "--res.nz", "16", "--res.ny", "65"]
        if system == "pipe":
            res = ["--res.nz", "16", "--res.ntheta", "16", "--res.nr", "65"]
        proc = _cli(
            tmp,
            system,
            [
                "--phys.driving",
                "constant_bulk_velocity",
                *res,
                "--step.dt",
                "0.005",
                "--stop.max_sim_time",
                "2.0",
                "--outs.it_stats",
                "40",
                "--outs.snapshot_save_initial",
                "False",
                "--outs.snapshot_save_final",
                "False",
                "--init.random_amplitude",
                "0.15",
                "--init.random_smoothness",
                "0.9",
                "--init.random_seed",
                "5",
            ],
        )
        if proc.returncode != 0:
            print(proc.stdout[-1500:])
            return f"{system} run exit {proc.returncode}"
        f = Path(tmp) / "stats.dat"
        cols = _columns(f)
        if cols[-1] != column:
            return f"{system}: last column is {cols[-1]!r}, not {column!r}"
        d = np.loadtxt(f)
        i = {n: k for k, n in enumerate(cols)}
        rec = d[-1, i[column]]
        if system == "pipe":
            inf = -2 * d[-1, i["tau'_z"]]
        else:
            inf = -(d[-1, i["tau'_s,t"]] - d[-1, i["tau'_s,b"]]) / 2
        rel = abs(rec - inf) / max(abs(inf), 1e-300)
        print(
            f"  {system}: recorded {rec:+.5e}  wall-shear {inf:+.5e}  "
            f"rel {rel:.2e} (tol {WS_TOL})"
        )
        if not rel < WS_TOL:
            return f"{system}: recorded vs wall-shear rel {rel:.2e}"
    return None


def _check_absent() -> str | None:
    """No knob, no column -- a default run's layout is untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        proc = _cli(
            tmp,
            "plane-poiseuille",
            [
                "--res.nx",
                "8",
                "--res.nz",
                "8",
                "--res.ny",
                "17",
                "--step.dt",
                "0.005",
                "--stop.max_sim_time",
                "0.02",
                "--outs.it_stats",
                "1",
                "--outs.snapshot_save_initial",
                "False",
                "--outs.snapshot_save_final",
                "False",
                "--init.start_from_laminar",
                "True",
            ],
        )
        if proc.returncode != 0:
            return f"CPG run exit {proc.returncode}"
        cols = _columns(Path(tmp) / "stats.dat")
        bad = [c for c in cols if c.startswith("-dP")]
        print(f"  constant_pressure_gradient columns: {cols}")
        if bad:
            return f"CPG run wrote a driving column {bad}"
    return None


def main() -> None:
    if "--worker" in sys.argv:
        kind = sys.argv[sys.argv.index("--worker") + 1]
        if kind == "laminar":
            sys.exit(_worker_laminar(sys.argv[3], sys.argv[4], sys.argv[5]))
        sys.exit(_worker_identity(sys.argv[3]))

    unit_only = "--unit-only" in sys.argv
    results: list[tuple[str, str | None]] = []

    for system, phys, column in LAMINAR_CASES:
        knob = "driving" if "driving" in phys else "block"
        print(f"=== laminar: {system} ({column}) ===")
        proc = run_live(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "laminar",
                system,
                knob,
                column,
            ],
            timeout=600,
        )
        results.append(
            (
                f"laminar {system}",
                None if proc.returncode == 0 else f"exit {proc.returncode}",
            )
        )

    for system in ("plane-poiseuille", "pipe"):
        print(f"=== identity: {system} ===")
        proc = run_live(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "identity",
                system,
            ],
            timeout=600,
        )
        results.append(
            (
                f"identity {system}",
                None if proc.returncode == 0 else f"exit {proc.returncode}",
            )
        )

    if not unit_only:
        print("=== wall-shear cross-check (non-laminar, resolved) ===")
        for system, column in WS_CASES:
            results.append(
                (f"wall shear {system}", _check_wall_shear(system, column))
            )
        print("=== column absent without a driving knob ===")
        results.append(("column absent under CPG", _check_absent()))

    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))


if __name__ == "__main__":
    main()
