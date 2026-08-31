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
injecting no ``O(1)`` source into the balance -- under **both**
formulations of ``res.consistent_imm``: the default reconstruction
scheme and the legacy primitive `$(v, p)$` one.

The residual is finite-difference/truncation level: a central-difference
``dE/dt`` at ``dt`` (``O(dt^2)`` on the smooth part) plus the SBP defect,
both convergent.  The guard is deliberately loose (``< BUDGET_TOL``
relative to the *term* magnitudes ``max(|I|, |D|, |dE/dt|)``, **excluding
the first-step projection transient**) so it catches an ``O(1)`` leak,
not the expected ``O(dt)`` finite-difference size.  Normalising by the
term magnitudes -- not by ``I - D``, which is near-zero for a
laminar-dominated pipe roll -- keeps it well-conditioned in every case.

Background: the residual is a convergent truncation error, physically
inert for resolved fields -- the ``Resolution.consistent_imm``
docstring.  The pipe entries start
from a resolved IC (``localized_rolls``) to keep the two formulations
comparable; neither *needs* it.

Plane-Poiseuille carries the only mean streamwise pressure gradient in
this file, so it is the only flow whose ``I`` has driving-dependent
branches, and the only Cartesian config with a moving frame
(``phys.u_grid = 2/3``).  Both branches are covered, plus spanwise
blocking on both Cartesian flows: ``I`` has **no** spanwise term, so
those entries fail visibly if the blocking force did any work.  Each
entry with an active hold additionally checks that the force is
non-zero and the bulk it holds is zero to rounding -- "no work" is
only informative when the force is real.  Which `$\Pi$` estimate
belongs in ``I``, with the measured table: the
``_check_applied_vs_inferred`` docstring.

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
# default start mode) unless the flags select localized_rolls; the pipe
# entries use a localized-rolls spot so both flag settings are judged
# on the same resolved field.
_ROLLS = [
    "--init.localized_rolls",
    "True",
    "--init.localized_rolls_amplitude",
    "0.15",
    "--geo.lz",
    "8.0",
]
#: ``(label, system, extra CLI flags)``.  Each system runs both
#: formulations: ``default`` is the shipped reconstruction scheme,
#: ``legacy`` the primitive `$(v, p)$` one, both passed explicitly so
#: the pair stays a contrast if the model default ever moves again.
CONFIGS = [
    (
        "plane-couette default",
        "plane-couette",
        ["--res.consistent_imm", "True"],
    ),
    (
        "plane-couette legacy",
        "plane-couette",
        ["--res.consistent_imm", "False"],
    ),
    (
        "taylor-couette default",
        "taylor-couette",
        ["--res.consistent_imm", "True"],
    ),
    (
        "taylor-couette legacy",
        "taylor-couette",
        ["--res.consistent_imm", "False"],
    ),
    ("pipe default", "pipe", [*_ROLLS, "--res.consistent_imm", "True"]),
    ("pipe legacy", "pipe", [*_ROLLS, "--res.consistent_imm", "False"]),
    # Plane-Poiseuille is the only flow here whose mean streamwise
    # pressure gradient does work, so it is the only one whose ``I``
    # has driving-dependent branches (``plane_poiseuille.py``'s
    # ``I_cpg`` / ``I_cbv``) -- and the only Cartesian config with a
    # moving frame (``phys.u_grid = 2/3``).  The plain flow keeps the
    # default/legacy pairing; the three driving variants do not,
    # because ``_apply_bulk_corrections`` is called by *both* IMM
    # paths (``cartesian._imm_iteration_vw`` and
    # ``_cartesian_primitive_imm``), so pairing them re-runs one
    # shared function.
    (
        "plane-poiseuille cpg",
        "plane-poiseuille",
        ["--res.consistent_imm", "True"],
    ),
    (
        "plane-poiseuille cpg legacy",
        "plane-poiseuille",
        ["--res.consistent_imm", "False"],
    ),
    # Constant bulk velocity: the applied gradient is time-varying and
    # its work is ``U_b_lam * (-dPds')`` with ``U_b_lam = 2/3``, the
    # one row of the driving table where neither factor vanishes.
    (
        "plane-poiseuille cbv",
        "plane-poiseuille",
        ["--phys.driving", "constant_bulk_velocity"],
    ),
    # Spanwise blocking applies a real, time-varying uniform force
    # ``-dPdn'`` that ``I`` has **no term for** -- correctly, because
    # the bulk it holds is *zero*, so the force does no work.  If that
    # argument were wrong the budget here would visibly fail to close,
    # which is what makes these two entries a test and not a
    # restatement.
    (
        "plane-poiseuille cbv+span",
        "plane-poiseuille",
        [
            "--phys.driving",
            "constant_bulk_velocity",
            "--phys.block_mean_spanwise_velocity",
            "True",
        ],
    ),
    (
        "plane-couette span",
        "plane-couette",
        ["--phys.block_mean_spanwise_velocity", "True"],
    ),
]


#: Wall-normal resolution of the plane-Poiseuille entries.  Sized with
#: ``scripts/wall_normal_resolution.py match``: at ``fd_order 8`` on the
#: CGL grid ``ny = 49`` resolves what 33 Chebyshev polynomials do to
#: 1 %, and gives ``y+_wall = 0.3`` / ``dy+_centre = 8.8`` at the
#: ``Re_tau ~ 135`` this ``Re`` and box reach.
_PP_NY = "49"


#: Pins the random IC every launch here starts from.  An unset seed is
#: drawn from the OS entropy pool (:mod:`dnsjax.seeding`), which would
#: put ``BUDGET_TOL`` and the applied-vs-inferred gaps on a different
#: trajectory each run; ``1`` is the value they were measured at.
_SEED_FLAGS = ["--init.random_seed", "1"]


def _base_flags(system: str) -> list[str]:
    """Resolution / Reynolds per family (moderate, resolved).

    Carries :data:`_SEED_FLAGS`, so every launch below is on the one
    measured trajectory.
    """
    if system == "plane-poiseuille":
        # The minimal-channel box (lx+ ~ 270, lz+ ~ 108 at Re_tau ~ 135)
        # at the transitional Re_b = 2000.
        return [
            *_SEED_FLAGS,
            "--phys.re",
            "3000",
            "--geo.lx",
            "2.0",
            "--geo.lz",
            "0.8",
            "--res.nx",
            "16",
            "--res.nz",
            "16",
            "--res.ny",
            _PP_NY,
            "--res.fd_order",
            "8",
        ]
    if system == "plane-couette":
        return [
            *_SEED_FLAGS,
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
            *_SEED_FLAGS,
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
        *_SEED_FLAGS,
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
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(stats)
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
    if not ok:
        return f"{label}: budget residual {rel:.2e}"
    return _check_held_bulks(label, flags, cols)


def _check_held_bulks(label: str, flags: list[str], cols: dict) -> str | None:
    r"""A held bulk closes the budget for a *reason*; check both halves.

    A uniform mean-mode force does work (force) `$\times$` (bulk along
    it).  ``I`` has a streamwise term and **no spanwise one at all**, so
    the spanwise entries only test the held-at-zero argument if the
    force they hold with is genuinely non-zero -- otherwise "no work"
    is true for the trivial reason and the entry proves nothing.  The
    same applies to ``constant_bulk_velocity``, where ``I``'s single
    streamwise term is the whole of the driving work.

    Requires, per active knob: the column exists, the applied force is
    non-zero somewhere, and the bulk it holds is zero to rounding.
    """
    held = []
    if "constant_bulk_velocity" in flags:
        held.append(("-dPds'", "Ub'_s"))
    if "--phys.block_mean_spanwise_velocity" in flags:
        held.append(("-dPdn'", "Ub'_n"))
    for key, bulk_key in held:
        if key not in cols:
            return f"{label}: stats.dat has no {key} column"
        force = float(np.max(np.abs(cols[key][1:])))
        bulk = float(np.max(np.abs(cols[bulk_key])))
        if force <= 0.0:
            return (
                f"{label}: the applied {key} is identically zero, so "
                "its doing no work says nothing"
            )
        if bulk > 1e-14:
            return f"{label}: {bulk_key} = {bulk:.2e} is not held to zero"
        print(
            f"        {key}: |force| up to {force:.2e} against "
            f"|{bulk_key}| <= {bulk:.1e}"
        )
    return None


# ── Applied vs inferred pressure-gradient work ───────────────────────

#: Wall-normal ladder for the applied-vs-inferred comparison, and the
#: horizon it runs to.  Both matter: ``test_driving.py``'s table shows
#: the wall-shear inference converging in ``ny`` *and* in ``t``.
_APPLIED_NY = (33, 49, 65, 97)
#: Run long enough for the driving to develop, and judge only the
#: developed window.  This is not fussiness: from a random IC the
#: applied `$-\Pi$` is ~1e-6, so its work is ~0.2 % of ``I`` and a
#: short run would pass however wrong the inference was.  By ``t = 40``
#: it is ~1e-3, i.e. ~60 % of ``I`` -- only there does the comparison
#: test anything.
_APPLIED_T = 60.0
_APPLIED_FROM = 40.0
#: Below this the wall-normal grid is simply too coarse for the
#: budget to close at all (see the table): those rungs are
#: measured and printed, not asserted on.
_APPLIED_RESOLVED = 49


def _pp_input_from_applied(cols: dict, re: float) -> np.ndarray:
    r"""Plane-Poiseuille ``I`` rebuilt from the **applied** driving.

    Under ``constant_bulk_velocity`` the total streamwise gradient is
    the laminar one plus the corrector's own `$\Pi$`, and the held
    bulk is `$U_{b,\mathrm{lam}} = 2/3$`, so
    `$I = -\Pi_\mathrm{tot} U_b
    = I_\mathrm{lam} + U_{b,\mathrm{lam}}\,(-\Pi)$` with
    `$I_\mathrm{lam} = 4/(3Re)$` (``plane_poiseuille.py``).  The
    ``-dPds'`` column *is* `$-\Pi$` (the applied forcing, sign carried
    in the name), so this needs no sign flip.
    """
    return 4.0 / (3.0 * re) + (2.0 / 3.0) * cols["-dPds'"]


def _budget_residual(cols: dict, inp: np.ndarray) -> float:
    """``max|dE/dt - (I - D)| / scale`` for a given ``I`` estimate."""
    t, E, D = cols["t"], cols["E"], cols["D"]
    dEdt = np.gradient(E, t)
    win = (t >= _APPLIED_FROM) & (t < t[-1])
    resid = np.abs(dEdt - (inp - D))[win]
    scale = max(
        np.max(np.abs(inp[win])),
        np.max(np.abs(D[win])),
        np.max(np.abs(dEdt[win])),
    )
    return float(np.max(resid) / scale)


def _check_applied_vs_inferred() -> str | None:
    r"""Which `$\Pi$` estimate belongs in ``I``, and why.

    ``get_stats`` is a function of the *state* alone, so under
    ``constant_bulk_velocity`` it can only infer `$\Pi$` from the wall
    shear (``plane_poiseuille.py``'s ``dpds_pert``).  That drops
    `$\mathrm{bulk}(\bar N_s)$`, the mean-mode bulk of the discrete
    nonlinear term -- continuously zero, a finite wall-normal
    truncation residual discretely.  The corrector's **applied** value
    is a different, exact quantity, recorded in ``stats.dat`` as
    ``-dPds'`` since the driving column shipped, so the total-field
    budget can be closed both ways and the two compared.

    Measured here (plane-Poiseuille MFU, ``Re = 3000``, ``nx = nz =
    16``, judged over the developed window ``t`` in ``[40, 60]``):

    ======  ==================  =================  ==============
    ``ny``  residual, ``I``     residual, applied  `$\Pi$` gap
    ======  ==================  =================  ==============
    33      6.6e-1              6.5e-1             7.7e-1
    49      3.7e-2              1.5e-1             2.3e-1
    65      1.2e-2              2.3e-2             2.0e-2
    97      3.0e-3              5.3e-3             9.3e-3
    ======  ==================  =================  ==============

    Two things to read off, and the second is the non-obvious one.

    The `$\Pi$` gap **converges** (7.7e-1 to 9.3e-3): the wall-shear
    inference and the applied force agree in the limit, as they must.
    At ``ny = 33`` nothing closes at all -- that rung is here to show
    what an under-resolved wall-normal grid looks like, not to pass.

    But the *inferred* `$\Pi$` closes the budget **better** than the
    applied one at every resolved rung.  That is not a statement about
    which estimate is right -- the applied force is the exact input
    rate, full stop -- but about ``I - D`` being an incomplete discrete
    budget.  Writing
    `$I_{ws} = I_{app} - \tfrac23\,\mathrm{bulk}(\bar N_s)$`, the true
    discrete defect of `$dE/dt = I_{app} - D$` is the nonlinear term's
    own energy contribution `$\langle u\cdot N(u)\rangle$` -- zero
    continuously -- whose dominant part is
    `$-U_b\,\mathrm{bulk}(\bar N_s)$` with `$U_b = 2/3$` held.  The two
    truncation errors are *the same quantity*, so they cancel.  (Check
    at ``ny = 49``: the residuals differ by 1.1e-1 of a 1.2e-3 scale,
    i.e. 1.4e-4, against `$\tfrac23\times$` the 2.3e-1 gap on a
    `$\Pi$` of 1e-3, i.e. 1.6e-4.)

    So **do not** "fix" ``get_stats`` to report the applied column:
    that would make ``I`` a better estimate of the physical input rate
    and a worse partner for ``D`` in the discrete budget.  The applied
    column stays the right answer to "what force is being applied",
    which is what ``tests/test_twin_budget.py`` uses it for.

    Asserts the two convergences, that the ``I`` column closes for
    ``ny >= _APPLIED_RESOLVED``, and that the cancellation above still
    holds (the inferred residual stays the smaller).
    """
    re = 3000.0
    rows: list[tuple[int, float, float, float]] = []
    for ny in _APPLIED_NY:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                "mpirun",
                "-np",
                "1",
                BIN,
                "--phys.system",
                "plane-poiseuille",
                *_base_flags("plane-poiseuille"),
                "--res.ny",
                str(ny),
                "--phys.driving",
                "constant_bulk_velocity",
                "--step.dt",
                str(DT),
                "--stop.max_sim_time",
                str(_APPLIED_T),
                "--outs.it_stats",
                "1",
                "--outs.it_error_check",
                "1",
                "--stop.check_laminarization",
                "False",
                "--init.random_amplitude",
                "0.2",
            ]
            env = {**os.environ, "DNSJAX_QUIET_STARTUP": "1"}
            proc = run_live(cmd, cwd=tmp, timeout=900, env=env)
            if proc.returncode != 0:
                return f"applied-vs-inferred ny={ny}: exit {proc.returncode}"
            cols = _columns(Path(tmp) / "stats.dat")
        if "-dPds'" not in cols:
            return f"applied-vs-inferred ny={ny}: stats.dat has no -dPds'"
        applied = _pp_input_from_applied(cols, re)
        rel_inf = _budget_residual(cols, cols["I"])
        rel_app = _budget_residual(cols, applied)
        # Both estimates of Pi' -- the applied column and the inference
        # I backs out of the I column -- relative to the applied one.
        win = cols["t"] >= _APPLIED_FROM
        pi_app = cols["-dPds'"][win]
        pi_inf = ((cols["I"] - 4.0 / (3.0 * re)) / (2.0 / 3.0))[win]
        gap = float(np.max(np.abs(pi_app - pi_inf)) / np.max(np.abs(pi_app)))
        rows.append((ny, rel_inf, rel_app, gap))
        print(
            f"    ny={ny:3d}  residual(I column)={rel_inf:.2e}  "
            f"residual(applied)={rel_app:.2e}  Pi' gap={gap:.2e}",
            flush=True,
        )

    gaps = [r[3] for r in rows]
    if not all(b < a for a, b in zip(gaps, gaps[1:], strict=False)):
        trail = " -> ".join(f"{g:.2e}" for g in gaps)
        return (
            "applied-vs-inferred: the two Pi' definitions did not "
            f"converge to each other in ny ({trail})"
        )
    infs = [r[1] for r in rows]
    if not all(b < a for a, b in zip(infs, infs[1:], strict=False)):
        trail = " -> ".join(f"{v:.2e}" for v in infs)
        return f"applied-vs-inferred: I-column residual not falling ({trail})"
    for ny, rel_inf, _rel_app, _gap in rows:
        if ny >= _APPLIED_RESOLVED and rel_inf >= BUDGET_TOL:
            return (
                f"applied-vs-inferred: the I column fails to close at "
                f"ny={ny} ({rel_inf:.2e})"
            )
    worse = [
        r[0] for r in rows if r[0] >= _APPLIED_RESOLVED and not r[1] < r[2]
    ]
    if worse:
        return (
            "applied-vs-inferred: the applied column closed better at "
            f"ny={worse}; the cancellation described in this function's "
            "docstring has stopped holding, so re-derive it before "
            "trusting either estimate"
        )
    print(
        "  PASS: applied-vs-inferred  (the I column closes and converges; "
        "the two Pi' definitions converge to each other; the wall-shear "
        "estimate stays the better budget partner)"
    )
    return None


def main() -> None:
    print(
        "Energy-budget closure: dE/dt == I - D to truncation order, "
        "under both res.consistent_imm formulations "
        "(offline; mpirun -np 1).",
        flush=True,
    )
    results = [(lbl, _run(lbl, sys_, fl)) for lbl, sys_, fl in CONFIGS]
    results.append(("applied-vs-inferred I", _check_applied_vs_inferred()))
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))


if __name__ == "__main__":
    main()
