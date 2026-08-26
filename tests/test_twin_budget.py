#!/usr/bin/env python3
r"""Twin-run budget closure across the Cartesian flows and driving modes.

``tests/test_twin_driver.py`` guards the *driver* -- streams, restart,
guards -- on one plane-Couette configuration with no driving knob set.
This script guards the *physics* of the difference-field budget
(:mod:`dnsjax.twin.diagnostics`) where that one cannot reach: on
plane-Poiseuille, and with the mean-mode driving constraints active.
Three code paths reach a twin run only through plane-Poiseuille --
the moving frame ``phys.u_grid = 2/3`` (whose cancellation the
"Frame invariance" note argues but never measured), a *curved* base
flow inside ``rU`` (plane-Couette's ``dU/dy`` is constant), and the
``-dPds'_d`` / ``-dPdn'_d`` columns of ``twin.dat``, which are
unreachable without a driving knob.  Each of them would show up as an
unclosed budget, so the closure measurement below is what tests them.

Three questions, all measured rather than argued.

1. **Does the budget close, and converge?**  Per component,
   `$\partial_t E_X = P_X + T_X - \epsilon_X$` up to spatial truncation
   and the `$O(\Delta t^2)$` stepping error
   (:func:`dnsjax.analysis.twin.series.closure_residuals` owns the
   definition).  Each configuration is measured on a wall-normal
   ladder, and the residuals must *shrink* along it: a missing or
   mis-signed term would not.

2. **Is the budget complete under an active driving constraint?**  The
   budget has no forcing column.  It does not need one because the
   omitted work `$\Delta\pi\cdot\mathrm{bulk}(\Delta u)$` vanishes --
   but by a *different* mechanism per setting, and neither mechanism is
   self-evident:

   ===================  ===============  ==================
   setting              `$\Delta\pi$`    bulk `$(\Delta u)$`
   ===================  ===============  ==================
   force free (CPG,     **zero**         non-zero
   plane-Couette,
   spanwise unblocked)
   bulk held (CBV,      non-zero         **zero**
   spanwise blocked)
   ===================  ===============  ==================

   So the test measures **both** factors in every configuration and
   bounds their product against the budget's own terms.  Measuring both
   is the point: asserting only the vanishing one would pass vacuously
   in exactly the case where a real term had gone missing.  A free bulk
   velocity is emphatically not a zero one -- plane-Couette acquires a
   streamwise bulk spontaneously despite having no streamwise pressure
   gradient at all, and so does the spanwise bulk of both flows when the
   block is off.

Measured
--------
Closure residuals, **max over ``twin.seed`` 3/5/7** at each rung
(20 budget samples each; the driving legs are in the next table):

===========  ====  =======  =======  =======  =======
config       ny    dU       du1      du2      T_tot
===========  ====  =======  =======  =======  =======
pp-cpg        49   1.4e-5   1.0e-3   1.5e-2   7.6e-2
              65   1.0e-5   1.1e-3   3.7e-3   1.8e-2
              97   7.6e-6   2.0e-4   1.5e-3   9.3e-3
pp-cbv        49   1.4e-5   3.6e-3   1.1e-2   4.6e-2
              65   1.5e-5   8.9e-4   3.6e-3   3.2e-2
              97   2.2e-5   1.1e-4   1.3e-3   5.0e-3
pp-cbv-span   49   1.7e-5   1.6e-3   3.7e-3   1.4e-2
              65   2.4e-5   6.5e-4   4.1e-3   7.8e-3
              97   2.5e-5   4.6e-5   5.2e-4   1.4e-3
pc            33   5.9e-5   1.8e-3   2.3e-3   2.6e-2
              49   1.3e-4   3.6e-4   1.5e-4   7.4e-4
              65   4.2e-4   5.6e-5   4.6e-5   8.4e-5
pc-span       33   1.6e-5   3.2e-3   2.3e-3   1.2e-2
              49   2.3e-5   5.6e-4   1.1e-4   7.5e-4
              65   1.3e-5   8.5e-5   4.9e-5   9.4e-5
===========  ====  =======  =======  =======  =======

All 15 (config, seed) ladders shrink on ``du1``, ``du2`` and
``T_tot``; ``dU`` does not, and is bounded absolutely instead.  Note
the two flows land in different regimes: plane-Couette closes an order
*tighter* on ``du2``/``T_tot`` and an order *looser* on ``dU``.

The driving legs, at the coarsest rung of each (they are
resolution-independent -- both are exact statements, not truncated
ones):

===========  ==========  ==========  ==========  ==========
config       bulk_s      dpi_s/pi_s  bulk_n      dpi_n/pi_n
===========  ==========  ==========  ==========  ==========
pp-cpg       4.5e-3      -- (no      7.5e-3      -- (no
                         force)                  force)
pp-cbv       8.8e-18     1.0e-2      8.6e-3      -- (no
                                                 force)
pp-cbv-span  1.2e-17     1.2e-2      4.3e-18     4.0e-3
pc           2.4e-3      -- (no      1.7e-3      -- (no
                         force)                  force)
pc-span      2.6e-3      -- (no      1.7e-18     1.8e-1
                         force)
===========  ==========  ==========  ==========  ==========

Every held bulk is 2.7e-18 to 4.7e-17 -- rounding, as it must be for a
rank-1 algebraic projection applied at every corrector iterate.  Every
free bulk is 2.4e-4 to 8.6e-3.  The omitted work
`$\Delta\pi\,\mathrm{bulk}(\Delta u)$` never exceeds **5.1e-16** of the
``dU`` balance it would join.

3. **Does the wall-normal-resolved budget regroup to this one?**
   :func:`_yresolved` reads ``twin_ybudget.bin`` alongside
   ``twin_budget.dat`` and checks that `$\sum_k \int$` of the
   production and viscous densities reproduce ``P_tot`` and
   ``eps_tot``.  Those identities are algebraic -- the same Parseval
   sum regrouped -- so they hold to rounding at every rung, which is
   what makes them a sharp regression test for the marginal
   reduction, the `$\pm k_z$` fold and the density normalisation, on
   real turbulence rather than the synthetic states of
   ``tests/test_twin_unit.py``.  Their two truncation-limited
   companions are printed per rung but **not** asserted; the note on
   ``_CONVERGE`` says why.

Configurations are **minimal flow units started from random
perturbations**: the plane-Poiseuille box is `$2.0 \times 2 \times
0.8$` at `$Re = 3000$` (`$Re_b = 2000$`, `$Re_\tau \approx 135$`, so
`$l_x^+ \approx 270$`, `$l_z^+ \approx 108$`), plane-Couette the
Hamilton-Kim-Waleffe `$1.75\pi \times 2 \times 1.2\pi$` at `$Re = 400$`.
Wall-normal ladders are sized with ``scripts/wall_normal_resolution.py
match`` and sit **above** the literature Chebyshev counts (at
``fd_order = 8`` on the CGL grid, ``ny = 49`` resolves what 33 Chebyshev
polynomials do to 1 %): the closure residual is a discretisation error,
so the resolution that settles it is not the resolution that settles the
physics.

Each rung spins its **own** parent up with the solver rather than
re-gridding one fine parent down.  Both would work, but a truncated fine
state carries more grid-scale content than the coarse scheme would ever
produce, which is the adverse case for the very FD adjointness defect
being measured -- it would manufacture convergence.  A per-rung parent
is a bona fide solution of its own discretisation.  (It is also why
these numbers are not comparable to ``test_twin_driver.py``'s, which
uses ``fd_order = 4`` and an *unstepped* random field as its parent.)

Single-process throughout, so unlike ``test_twin_driver.py`` this script
needs no ``mpirun`` (a lone process needs no launcher).

Usage::

    uv run python tests/test_twin_budget.py
    uv run python tests/test_twin_budget.py --only pp-cbv
    uv run python tests/test_twin_budget.py --quick
    uv run python tests/test_twin_budget.py --only pp-cbv \
        --ladder 49 65 97 --seeds 3 5 7
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

_REPO = Path(__file__).resolve().parent.parent
BIN = str(_REPO / ".venv" / "bin" / "dnsjax")

from dnsjax.analysis.snapshot_export import read_state  # noqa: E402
from dnsjax.analysis.snapshot_ops import integrate  # noqa: E402
from dnsjax.analysis.twin.lengths import partner_of  # noqa: E402
from dnsjax.analysis.twin.series import (  # noqa: E402
    ClosureResiduals,
    closure_residuals,
    read_dat,
    read_twin,
)

assert "jax" not in sys.modules, "the readers here must stay JAX-free"

#: Fixed step for every run (the twin driver requires a fixed ``dt``).
DT = 0.01
#: Perturbation energy of the twin partner.  ~0.5 % of the reference's
#: own `$E'$` at these configurations: well above the cancellation floor
#: of ``E_d(t0) == e0`` and still a perturbation.
E0 = 1e-4
#: Budget cadence, in steps.
IT_BUDGET = 5

_HKW_LX = 1.75 * math.pi
_HKW_LZ = 1.2 * math.pi

#: ``name`` selects with ``--only``.  ``spin`` is the parent spin-up
#: horizon and ``horizon`` the twin's own, both in advective units.
#: ``keys`` are the driving columns the configuration must produce --
#: empty means the run must produce *none*, which is itself asserted.
#: Absolute bounds at each flow's **finest** rung, ~10x above the worst
#: of three ``twin.seed`` values there (the measured table is in this
#: module's docstring).  Ten, not the two or three that would look
#: tighter: a fixed rung's residual varies by 1.2x-31x across seeds, so
#: a 2x margin would be fitted to one draw.  Two sets because the two
#: flows sit in genuinely different places -- plane-Couette closes an
#: order tighter on ``du2``/``T_tot`` and an order *looser* on ``dU``.
_PP_BOUNDS = {"dU": 3e-4, "du1": 2e-3, "du2": 1.5e-2, "T_tot": 1e-1}
_PC_BOUNDS = {"dU": 5e-3, "du1": 1e-3, "du2": 5e-4, "T_tot": 1e-3}

#: ``dU`` is bounded absolutely but **not** asserted to converge: it
#: grows under refinement (plane-Couette: 5.9e-5 -> 1.3e-4 -> 4.2e-4
#: over its ladder), because the mean-mode difference budget it is
#: normalised by shrinks faster than its own absolute residual does.
#: The same exclusion, for the same reason, as in
#: ``tests/test_twin_driver.py``.
#:
#: ``T_bins`` and ``pi_flux`` (:func:`_yresolved`) are **measured and
#: printed per rung but not yet asserted**, here or as absolute
#: bounds.  Their convergence has so far only been established on
#: random solenoidal states -- ``ny`` 15/31/63/127 giving
#: 3.1e-1 -> 8.3e-3 -> 2.0e-3 -> 1.2e-4 and
#: 1.8e-2 -> 6.5e-4 -> 2.6e-5 -> 2.1e-6 -- and this file's discipline
#: is that a bound comes from a measured sweep over *these*
#: configurations and seeds, not from a plausible one elsewhere.  Run
#: ``--measure`` across the ladder and add them here.
_CONVERGE = ("du1", "du2", "T_tot")

CONFIGS: list[dict] = [
    {
        "name": "pp-cpg",
        "bounds": _PP_BOUNDS,
        "converge": _CONVERGE,
        "system": "plane-poiseuille",
        "args": ["--phys.re", "3000", "--geo.lx", "2.0", "--geo.lz", "0.8"],
        "nx": 24,
        "nz": 24,
        "ladder": [49, 65, 97],
        "spin": 50.0,
        "horizon": 1.0,
        "keys": [],
    },
    {
        # The one configuration where the applied force is genuinely
        # different between the two runs *and* time-varying.
        "name": "pp-cbv",
        "bounds": _PP_BOUNDS,
        "converge": _CONVERGE,
        "system": "plane-poiseuille",
        "args": [
            "--phys.re",
            "3000",
            "--geo.lx",
            "2.0",
            "--geo.lz",
            "0.8",
            "--phys.driving",
            "constant_bulk_velocity",
        ],
        "nx": 24,
        "nz": 24,
        "ladder": [49, 65, 97],
        "spin": 50.0,
        "horizon": 1.0,
        "keys": ["-dPds'"],
    },
    {
        "name": "pp-cbv-span",
        "bounds": _PP_BOUNDS,
        "converge": _CONVERGE,
        "system": "plane-poiseuille",
        "args": [
            "--phys.re",
            "3000",
            "--geo.lx",
            "2.0",
            "--geo.lz",
            "0.8",
            "--phys.driving",
            "constant_bulk_velocity",
            "--phys.block_mean_spanwise_velocity",
            "True",
        ],
        "nx": 24,
        "nz": 24,
        "ladder": [49, 65, 97],
        "spin": 50.0,
        "horizon": 1.0,
        "keys": ["-dPds'", "-dPdn'"],
    },
    {
        "name": "pc",
        "bounds": _PC_BOUNDS,
        "converge": _CONVERGE,
        "system": "plane-couette",
        "args": [
            "--phys.re",
            "400",
            "--geo.lx",
            repr(_HKW_LX),
            "--geo.lz",
            repr(_HKW_LZ),
        ],
        "nx": 24,
        "nz": 24,
        "ladder": [33, 49, 65],
        "spin": 50.0,
        "horizon": 1.0,
        "keys": [],
    },
    {
        # Spanwise blocking with no streamwise driving at all: the
        # ``-dPdn'_d`` leg on its own.
        "name": "pc-span",
        "bounds": _PC_BOUNDS,
        "converge": _CONVERGE,
        "system": "plane-couette",
        "args": [
            "--phys.re",
            "400",
            "--geo.lx",
            repr(_HKW_LX),
            "--geo.lz",
            repr(_HKW_LZ),
            "--phys.block_mean_spanwise_velocity",
            "True",
        ],
        "nx": 24,
        "nz": 24,
        "ladder": [33, 49, 65],
        "spin": 50.0,
        "horizon": 1.0,
        "keys": ["-dPdn'"],
    },
]


# ── Runs ─────────────────────────────────────────────────────────────


def _common(cfg: dict, ny: int) -> list[str]:
    return [
        "--phys.system",
        cfg["system"],
        *cfg["args"],
        "--res.nx",
        str(cfg["nx"]),
        "--res.nz",
        str(cfg["nz"]),
        "--res.ny",
        str(ny),
        "--res.fd_order",
        "8",
        "--res.double_precision",
        "True",
        "--step.dt",
        repr(DT),
        "--stop.check_laminarization",
        "False",
    ]


def _spin_up(cfg: dict, ny: int, workdir: Path, spin: float) -> Path:
    """Run the solver from a random IC; return the final snapshot.

    Cached per ``(config, ny)``: ``twin.seed`` selects the *partner*,
    so a seed sweep shares one reference trajectory -- which is also
    the only way the sweep isolates the partner's effect.  The
    reference's own ``init.random_seed`` is pinned for the same reason.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    done = [
        p
        for p in sorted(workdir.glob("state*.tar"))
        if not p.name.endswith("_twin.tar")
    ]
    if done:
        return done[-1]
    cmd = [
        BIN,
        *_common(cfg, ny),
        "--init.random_amplitude",
        "0.2",
        # Pinned: an unset seed is drawn from the OS entropy pool
        # (:mod:`dnsjax.seeding`), so the reference trajectory -- and
        # with it the measured bounds below and the directory cache
        # above -- would differ every run.
        "--init.random_seed",
        "1",
        "--stop.max_sim_time",
        repr(spin),
        "--outs.it_stats",
        "100",
    ]
    env = {**os.environ, "DNSJAX_QUIET_STARTUP": "1"}
    proc = run_live(cmd, cwd=workdir, env=env, timeout=3600)
    assert proc.returncode == 0, (
        f"{cfg['name']} ny={ny}: spin-up exit {proc.returncode}"
    )
    snaps = sorted(workdir.glob("state*.tar"))
    snaps = [p for p in snaps if not p.name.endswith("_twin.tar")]
    assert snaps, f"{cfg['name']} ny={ny}: spin-up wrote no snapshot"
    return snaps[-1]


def _twin(
    cfg: dict,
    ny: int,
    parent: Path,
    workdir: Path,
    seed: int,
    horizon: float,
) -> Path:
    """Run ``dnsjax-twin`` off *parent*; return the member directory."""
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "dnsjax.twin",
        *_common(cfg, ny),
        "--init.snapshot",
        str(parent),
        "--twin.e0",
        repr(E0),
        "--twin.seed",
        str(seed),
        "--twin.bins",
        "True",
        "--twin.it_budget",
        str(IT_BUDGET),
        "--twin.it_ybudget",
        str(IT_BUDGET),
        # A horizon past the parent snapshot, which is what
        # ``stop.max_sim_time`` means (relative to init.t0).
        "--stop.max_sim_time",
        repr(horizon),
        "--outs.it_stats",
        "1",
        "--outs.stats_precision",
        "17",
        "--outs.it_snapshot",
        str(max(1, int(round(horizon / DT)) // 2)),
    ]
    env = {**os.environ, "DNSJAX_QUIET_STARTUP": "1"}
    proc = run_live(cmd, cwd=workdir, env=env, timeout=3600)
    assert proc.returncode == 0, (
        f"{cfg['name']} ny={ny} seed={seed}: twin exit {proc.returncode}"
    )
    return workdir


# ── The omitted driving-work term ────────────────────────────────────


def _tilt(params) -> tuple[float, float]:
    theta = math.radians(float(params.geo.tilt_degree))
    return math.cos(theta), math.sin(theta)


def _difference_bulk(member: Path) -> dict[str, float]:
    r"""`$\mathrm{bulk}(\Delta u)$` in the tilted (s, n) frame.

    Read from the **paired snapshots** the driver writes, because no
    stream reports the partner's bulk velocity: ``stats.dat`` carries
    the reference's alone.  The volume average is the same quantity the
    budget's `$\langle\cdot\rangle$` uses, so the product with the
    recorded force difference is directly comparable to a budget term.

    Returns the largest magnitude over the pairs present, keyed ``s``
    and ``n``.
    """
    pairs = [
        p
        for p in sorted(member.glob("state*.tar"))
        if not p.name.endswith("_twin.tar")
    ]
    assert pairs, f"{member}: the twin run wrote no snapshot pair"
    worst = {"s": 0.0, "n": 0.0}
    for ref in pairs:
        twin = partner_of(ref)
        a, b = read_state(ref), read_state(twin)
        cos_t, sin_t = _tilt(a.params)
        volume = float(a.params.geo.lx) * float(a.params.geo.lz) * 2.0
        bulk = [
            float(integrate(bi - ai, a.params, a.physical_coords)) / volume
            for ai, bi in zip(a.physical, b.physical, strict=True)
        ]
        got = {
            "s": bulk[0] * cos_t + bulk[2] * sin_t,
            "n": -bulk[0] * sin_t + bulk[2] * cos_t,
        }
        for k, v in got.items():
            worst[k] = max(worst[k], abs(v))
    return worst


def _driving_work(cfg: dict, member: Path) -> dict[str, float]:
    r"""Bound the budget's omitted `$\Delta\pi\,\mathrm{bulk}(\Delta u)$`.

    Returns, per constrained direction, the largest force difference
    (``dpi_*``), the largest difference-field bulk (``bulk_*``), their
    product (``work_*``) and that product relative to the smallest
    budget term the same run resolved (``ratio_*``) -- the number that
    decides whether the omission is defensible.
    """
    twin_cols = read_dat(member / "twin.dat")
    stats_cols = read_dat(member / "stats.dat")
    budget = read_dat(member / "twin_budget.dat")
    keys = list(cfg["keys"])

    present = [k for k in ("-dPds'", "-dPdn'") if f"{k}_d" in twin_cols]
    assert present == keys, (
        f"{cfg['name']}: twin.dat carries driving columns {present}, "
        f"expected {keys}"
    )
    for key in keys:
        assert key in stats_cols, f"{cfg['name']}: stats.dat lacks {key}"

    bulk = _difference_bulk(member)
    # The omitted work is a *mean-mode* term -- the force lives on the
    # (0, 0) mode alone -- so the balance it would join is the ``dU``
    # component's, and that is the scale it must be small against.
    # Not "the smallest budget term": ``P_dU(dU,rU)`` is structurally
    # zero, because a divergence-free field with no-slip walls has an
    # identically vanishing mean wall-normal velocity, so
    # `$(\Delta U\cdot\nabla)$` of a mean profile is zero pointwise.
    p_du = sum(budget[n] for n in budget if n.startswith("P_dU("))
    t_du = sum(budget[n] for n in budget if n.startswith("T_dU("))
    scale = float(np.abs(p_du + t_du - budget["eps_dU"]).max())
    assert scale > 0.0, f"{cfg['name']}: the dU budget is identically zero"

    out: dict[str, float] = {"budget_dU": scale}
    for key, axis in (("-dPds'", "s"), ("-dPdn'", "n")):
        # Row 0 is the wall-shear *inference*, not the applied value
        # (no step precedes it); it is a poor estimator at these ny and
        # must not enter the measurement.
        dpi = (
            float(np.abs(twin_cols[f"{key}_d"][1:]).max())
            if f"{key}_d" in twin_cols
            else 0.0
        )
        pi = (
            float(np.abs(stats_cols[key][1:]).max())
            if key in stats_cols
            else 0.0
        )
        out[f"dpi_{axis}"] = dpi
        out[f"pi_{axis}"] = pi
        out[f"bulk_{axis}"] = bulk[axis]
        out[f"work_{axis}"] = dpi * bulk[axis]
        out[f"ratio_{axis}"] = dpi * bulk[axis] / scale
    return out


# ── The measurement ──────────────────────────────────────────────────


def _measure(
    cfg: dict, ny: int, root: Path, seed: int, spin: float, horizon: float
) -> tuple[ClosureResiduals, dict[str, float], dict[str, float]]:
    parent = _spin_up(cfg, ny, root / f"{cfg['name']}_ny{ny}_spin", spin)
    member = _twin(
        cfg,
        ny,
        parent,
        root / f"{cfg['name']}_ny{ny}_s{seed}_twin",
        seed,
        horizon,
    )
    series = read_twin(member)
    return (
        closure_residuals(series),
        _driving_work(cfg, member),
        _yresolved(member),
    )


#: The held bulk is pinned by a rank-1 algebraic projection, so it is
#: zero to rounding, not to a solver tolerance.  Measured 4e-18 to
#: 2e-17 across the configurations here.
BULK_HELD_TOL = 1e-12
#: A *free* bulk must be demonstrably non-zero, or the "the force is
#: what vanishes" leg proves nothing.  Measured O(1e-3 .. 1e-2).
BULK_FREE_MIN = 1e-8
#: Likewise the force difference on a held direction: it has to be a
#: real fraction of the reference's own applied force, or the "the bulk
#: is what vanishes" leg is equally vacuous.  Measured ~1e-2.
DPI_FRACTION_MIN = 1e-4
#: The omitted work relative to the dU balance it would join.
#: Measured 1e-17 to 1e-18.
WORK_RATIO_TOL = 1e-10
#: Interior budget samples a rung must contribute for its residual to
#: be a maximum over something rather than a single draw.
MIN_SAMPLES = 15


# ── The wall-normal-resolved budget on the same ladder ───────────────


def _yresolved(member: Path) -> dict[str, float]:
    r"""Relate ``twin_ybudget.bin`` to ``twin_budget.dat``.

    Four numbers, of two kinds.  ``P_exact`` and ``V_exact`` are
    *algebraic*: `$\sum_k \int$` of the production terms and of the
    viscous term are the same Parseval sum ``twin_budget.dat``
    already forms, only regrouped, so they hold to rounding at every
    rung and on any state.  They are the load-bearing check that the
    marginal reduction, the `$\pm k_z$` fold and the density
    normalisation are all right -- here on real turbulence rather
    than the synthetic states of ``tests/test_twin_unit.py``.

    ``T_bins`` and ``pi_flux`` are *truncation*-limited and converge
    rather than hold.  ``T_bins`` compares the `$k$`-set sums of the
    transfer terms against the per-component transport of the paper's
    (2.14)-(2.16); the two differ by the same-bin triads
    `$\tfrac12\langle(\mathbf{b}\cdot\nabla)|\Delta\mathbf{a}|^2
    \rangle$` that the paper's lists omit because they vanish for
    solenoidal `$\mathbf{b}$` -- continuously.  Discretely they
    vanish only as the integration-by-parts residual that makes
    ``T_tot`` non-zero, so ``T_bins`` and ``T_tot`` are one quantity
    seen twice.  ``pi_flux`` is `$\max_k|\int\Pi\,dy|$`, which the
    continuity identity forces to zero at every mode, normalised by
    the largest `$|\int\mathcal{V}\,dy|$` over the same modes.
    """
    from dnsjax.analysis.twin import integrate_y, read_twin_ybudget

    data = read_twin_ybudget(member)
    budget = read_dat(member / "twin_budget.dat")
    t_b = np.round(budget["t"], 10)
    index = {t: i for i, t in enumerate(t_b)}
    out = {"P_exact": 0.0, "V_exact": 0.0, "T_bins": 0.0, "pi_flux": 0.0}

    def bins(name: str, k: int) -> np.ndarray:
        x0 = integrate_y(data, f"{name}_x0")[k]
        x = integrate_y(data, f"{name}_x")[k]
        return np.array([x0[0], x0[1:].sum(), (x - x0).sum()])

    for k, t in enumerate(np.round(data.t, 10)):
        i = index.get(t)
        if i is None:  # the driver's unconditional final row
            continue
        p_sum = (
            integrate_y(data, "P_U_x")[k].sum()
            + integrate_y(data, "P_r_x")[k].sum()
        )
        v_sum = -integrate_y(data, "V_z")[k].sum()
        p_ref = max(abs(budget["P_tot"][i]), 1e-300)
        e_ref = max(abs(budget["eps_tot"][i]), 1e-300)
        out["P_exact"] = max(
            out["P_exact"], abs(p_sum - budget["P_tot"][i]) / p_ref
        )
        out["V_exact"] = max(
            out["V_exact"], abs(v_sum - budget["eps_tot"][i]) / e_ref
        )
        want = np.array(
            [
                sum(budget[c][i] for c in budget if c.startswith(f"T_{b}("))
                for b in ("dU", "du1", "du2")
            ]
        )
        got = bins("T_ref", k) + bins("T_self", k)
        scale = max(np.abs(want).max(), 1e-300)
        out["T_bins"] = max(out["T_bins"], np.abs(got - want).max() / scale)
        pi = np.abs(integrate_y(data, "Pi_x")[k])
        visc = np.abs(integrate_y(data, "V_x")[k])
        out["pi_flux"] = max(
            out["pi_flux"], pi.max() / max(visc.max(), 1e-300)
        )
    return out


#: The two algebraic identities of :func:`_yresolved` are exact up to
#: float summation order over `$O(10^4)$` modes and `$N_y$` quadrature
#: nodes, so the bound is a rounding allowance, not a physics one.
YEXACT_TOL = 1e-9


def _check_closure(
    cfg: dict, rows: dict[int, dict[str, float]], ladder: list[int]
) -> list[str]:
    r"""Absolute bounds at the finest rung, plus ladder shrinkage.

    *rows* holds, per rung, the **worst over the seeds run**.  That
    matters: a single seed's residual is not reproducible to better
    than an order of magnitude (measured spreads of 1.2x to 31x across
    three ``twin.seed`` values at a fixed rung), so a one-seed ladder
    comparison can flip on noise alone.  Taking the max over seeds is
    what makes the shrinkage assertion a statement about the
    discretisation rather than about the draw -- and it is why the
    absolute bounds below carry ~10x margin over the measured worst
    case rather than the 2-3x that would look tighter.
    """
    bad: list[str] = []
    finest = rows[ladder[-1]]
    for name, bound in cfg["bounds"].items():
        if not finest[name] < bound:
            bad.append(
                f"ny={ladder[-1]} {name}={finest[name]:.2e} >= {bound:.2e}"
            )
    if len(ladder) < 2:
        return bad
    for name in cfg["converge"]:
        seq = [rows[ny][name] for ny in ladder]
        if not seq[-1] < seq[0]:
            trail = " -> ".join(f"{v:.2e}" for v in seq)
            bad.append(f"{name} did not shrink along the ladder ({trail})")
    return bad


def _check_driving(cfg: dict, work: dict[str, float]) -> list[str]:
    """The omitted-term legs, both factors measured (see the header)."""
    bad: list[str] = []
    held = {
        "s": "-dPds'" in cfg["keys"],
        "n": "-dPdn'" in cfg["keys"],
    }
    for axis, is_held in held.items():
        bulk, dpi, pi = (
            work[f"bulk_{axis}"],
            work[f"dpi_{axis}"],
            work[f"pi_{axis}"],
        )
        if is_held:
            if not bulk < BULK_HELD_TOL:
                bad.append(f"bulk_{axis}={bulk:.2e} not held to zero")
            if not (pi > 0 and dpi > DPI_FRACTION_MIN * pi):
                bad.append(
                    f"dpi_{axis}={dpi:.2e} vs pi_{axis}={pi:.2e}: the "
                    "force difference is too small for the zero-work "
                    "result to mean anything"
                )
        else:
            if not bulk > BULK_FREE_MIN:
                bad.append(
                    f"bulk_{axis}={bulk:.2e} is not measurably free; "
                    "the vanishing-force leg proves nothing"
                )
            if dpi != 0.0:
                bad.append(f"dpi_{axis}={dpi:.2e} on an undriven axis")
        if not work[f"ratio_{axis}"] < WORK_RATIO_TOL:
            bad.append(
                f"omitted work along {axis} is {work[f'ratio_{axis}']:.2e} "
                "of the dU balance"
            )
    return bad


def run_config(cfg: dict, args: argparse.Namespace) -> str | None:
    """Measure one configuration across its ladder; return failure."""
    ladder = args.ladder or cfg["ladder"]
    spin = args.spin if args.spin is not None else cfg["spin"]
    horizon = args.horizon if args.horizon is not None else cfg["horizon"]
    seeds = args.seeds or [3]
    print(f"\n[{cfg['name']}] ladder={ladder} seeds={seeds}", flush=True)

    bad: list[str] = []
    rows: dict[int, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(prefix="twin_budget_") as tmp:
        root = Path(tmp)
        for ny in ladder:
            worst: dict[str, float] = {}
            rows[ny] = worst
            for seed in seeds:
                resid, work, yres = _measure(
                    cfg, ny, root, seed, spin, horizon
                )
                for name, value in resid.components.items():
                    worst[name] = max(worst.get(name, 0.0), value)
                comp = resid.components
                print(
                    f"  ny={ny:3d} seed={seed}  "
                    f"dU={comp['dU']:.2e} du1={comp['du1']:.2e} "
                    f"du2={comp['du2']:.2e} T_tot={comp['T_tot']:.2e} "
                    f"({resid.n_samples} samples)",
                    flush=True,
                )
                print(
                    "           driving: "
                    + " ".join(f"{k}={work[k]:.3e}" for k in sorted(work)),
                    flush=True,
                )
                for name in ("T_bins", "pi_flux"):
                    worst[name] = max(worst.get(name, 0.0), yres[name])
                print(
                    "           (y,k): "
                    + " ".join(f"{k}={yres[k]:.2e}" for k in sorted(yres)),
                    flush=True,
                )
                if not _measuring(args):
                    for name in ("P_exact", "V_exact"):
                        if not yres[name] < YEXACT_TOL:
                            bad.append(
                                f"ny={ny} seed={seed}: {name}="
                                f"{yres[name]:.2e} >= {YEXACT_TOL:.0e} -- "
                                "the (y,k) budget no longer regroups to "
                                "the volume-averaged one"
                            )
                    bad += [
                        f"ny={ny} seed={seed}: {why}"
                        for why in _check_driving(cfg, work)
                    ]
                    if resid.n_samples < MIN_SAMPLES:
                        bad.append(
                            f"ny={ny} seed={seed}: only "
                            f"{resid.n_samples} budget samples"
                        )
    if _measuring(args):
        return None
    bad += _check_closure(cfg, rows, ladder)
    return "; ".join(bad) if bad else None


def _measuring(args: argparse.Namespace) -> bool:
    """Report-only mode: the bounds below came from exactly this."""
    return bool(args.quick or args.measure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Twin-run budget closure across flows / driving modes."
    )
    parser.add_argument(
        "--only", default=None, help="Substring filter on the config name"
    )
    parser.add_argument(
        "--ladder",
        type=int,
        nargs="*",
        default=None,
        help="Override the wall-normal ladder",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="twin.seed values (default: one seed)",
    )
    parser.add_argument("--spin", type=float, default=None)
    parser.add_argument("--horizon", type=float, default=None)
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Report the numbers without asserting bounds",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Coarsest rung only, short spin-up (smoke, not a guard)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configs = CONFIGS
    if args.only:
        configs = [c for c in configs if args.only in c["name"]]
        assert configs, f"--only {args.only!r} matches no config"
    if args.quick:
        args.spin = args.spin or 5.0
        args.horizon = args.horizon or 0.3
        configs = [{**c, "ladder": c["ladder"][:1]} for c in configs]

    passed, failures = 0, []
    for cfg in configs:
        why = run_config(cfg, args)
        if why is None:
            passed += 1
        else:
            print(f"  FAIL  {cfg['name']}: {why}")
            failures.append((cfg["name"], why))
    sys.exit(report(passed, failures))


if __name__ == "__main__":
    main()
