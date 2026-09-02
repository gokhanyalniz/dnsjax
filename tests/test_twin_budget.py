#!/usr/bin/env python3
r"""Twin-run budget closure across the Cartesian flows and driving modes.

``tests/test_twin_driver.py`` guards the *driver* -- streams, restart,
guards -- on one plane-Couette configuration with no driving knob set.
This script guards the *physics* of the difference-field budget
(:mod:`dnsjax.twin.diagnostics`) where that one cannot reach: on
plane-Poiseuille, and with the mean-mode driving constraints active.
Three code paths reach a twin run only through these configurations --
the moving frame ``phys.u_grid = 2/3`` (whose cancellation the
"Frame invariance" note argues but never measured), a *curved* base
flow inside ``rU`` (plane-Couette's ``dU/dy`` is constant), and an
active mean-mode driving constraint, which no run without a driving
knob can reach.  Each of them would show up as an unclosed budget, so
the closure measurement below is what tests them.

Two questions, both measured rather than argued.

1. **Does the budget close, and converge?**  Per component,
   `$\partial_t E_X = P_X + T_X - \epsilon_X$` up to spatial truncation
   and the `$O(\Delta t^2)$` stepping error
   (:func:`dnsjax.analysis.twin.series.closure_residuals` owns the
   definition).  Each configuration is measured on a wall-normal
   ladder, and the residuals must *shrink* along it: a missing or
   mis-signed term would not.

Measured
--------
Closure residuals, **max over ``twin.seed`` 3/5/7** at each rung
(20 budget samples each).  The ``pp-cbv`` block was re-measured at
the current partner construction; **the other four blocks predate
it** -- the `$(0,0)$` perturbation became `$\Pi$`-preserving after
they were taken, which changes the partner for a given ``twin.seed``
and so every residual here.  They are kept because the bounds below
were fitted to them and still hold with room; re-measure before
tightening any of them.

===========  ====  =======  =======  =======  =======
config       ny    dU       du1      du2      T_tot
===========  ====  =======  =======  =======  =======
pp-cpg        49   1.4e-5   1.0e-3   1.5e-2   7.6e-2
              65   1.0e-5   1.1e-3   3.7e-3   1.8e-2
              97   7.6e-6   2.0e-4   1.5e-3   9.3e-3
pp-cbv        49   1.4e-5   5.9e-4   4.7e-3   1.7e-1
              65   2.5e-5   2.6e-4   1.3e-2   3.1e-2
              97   3.3e-5   2.4e-4   3.1e-4   6.7e-3
             129   8.7e-6   2.7e-5   9.3e-6   6.0e-4
             161   1.5e-5   3.2e-6   4.5e-5   1.4e-5
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

The `$(y, k)$` numbers of :func:`_yresolved` on the same ``pp-cbv``
ladder, same seeds, in the **default convective form**:

====  ========  ========  ========  ========
ny    P_exact   V_exact   T_bins    pi_flux
====  ========  ========  ========  ========
  49  1.6e-15   5.7e-16   3.3e-1    7.0e-4
  65  8.0e-16   6.1e-16   5.8e-2    7.2e-4
  97  3.2e-15   8.0e-16   2.2e-2    5.3e-5
 129  1.7e-15   9.3e-16   1.1e-3    1.0e-5
 161  1.4e-15   6.0e-16   2.2e-4    4.1e-7
====  ========  ========  ========  ========

``P_exact`` and ``V_exact`` are the algebraic pair and sit at the
float floor; ``T_bins`` and ``pi_flux`` are truncation-limited and
converge, tracking ``T_tot`` rung for rung because all three are the
same discrete integration-by-parts residual seen from different
sides.

Under ``twin.rotational_ybudget`` the set changes (``T_zero`` and
``N_tot`` in place of ``T_bins``; ``P_exact`` measured against
``P_lift``) -- see :func:`_yresolved`.  ``T_zero`` there is exact by
pointwise algebra and so has no `$n_y$` dependence at all, which is
what that form buys.

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
    uv run python tests/test_twin_budget.py --only pp-cbv,pc --quick
    uv run python tests/test_twin_budget.py --only pp-cbv \
        --ladder 49 65 97 --seeds 3 5 7

``--only`` matches config names **exactly** and takes a
comma-separated list, so ``pp-cbv`` does not also select
``pp-cbv-span``.
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
#: The truncation-limited `$(y, k)$` numbers of :func:`_yresolved`
#: (``T_bins`` / ``pi_flux`` in the default convective form,
#: ``N_tot`` / ``pi_flux`` under ``twin.rotational_ybudget``) are
#: **measured and printed for every configuration but asserted only
#: where they have been swept** -- today ``pp-cbv``, which carries
#: them in its own ``bounds`` / ``converge`` entries
#: (:data:`_YRES_BOUNDS`).  This file's discipline is that a bound
#: comes from a measured sweep over *that* configuration and its
#: seeds, not from a plausible one elsewhere: run ``--measure`` on
#: another config and extend it the same way.  The algebraic
#: identities need no sweep and are asserted everywhere under
#: :data:`YEXACT_TOL`.
_CONVERGE = ("du1", "du2", "T_tot")

#: Absolute bounds on the truncation-limited `$(y, k)$` numbers at the
#: **finest** rung of a ladder, from the ``--measure`` sweep in the
#: module docstring.  Carried with the same ~10x margin as the closure
#: bounds, and over the worst of three seeds, for the same reason: a
#: fixed rung's residual is not reproducible to better than an order
#: of magnitude across ``twin.seed``.  These sit at ``pp-cbv``'s
#: default finest rung (``ny = 97``: 2.2e-2 and 5.3e-5), not at the
#: 161 the docstring table reaches.  Convective-form values: the
#: rotational ``pi_flux`` is a couple of decades larger, its ``Wp``
#: being the work of the rougher Bernoulli pressure.
_YRES_BOUNDS: dict[str, float] = {"T_bins": 2e-1, "pi_flux": 5e-4}


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
    },
    {
        # The one configuration where the applied force is genuinely
        # different between the two runs *and* time-varying.
        "name": "pp-cbv",
        # The one configuration swept for the (y, k) numbers as well
        # (the ladder in this module's docstring), so it is the one
        # that asserts them.
        "bounds": {**_PP_BOUNDS, **_YRES_BOUNDS},
        "converge": (*_CONVERGE, *_YRES_BOUNDS),
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
    },
    {
        # Spanwise blocking with no streamwise driving at all: a
        # constrained mean mode in the spanwise direction alone.
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
        # The convective leg reads the (y, k) budget back per k-set
        # bin, which is the k_x = 0 plane's job and is off by default.
        "--twin.x0_planes",
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


# ── The measurement ──────────────────────────────────────────────────


def _measure(
    cfg: dict, ny: int, root: Path, seed: int, spin: float, horizon: float
) -> tuple[ClosureResiduals, dict[str, float]]:
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
    return closure_residuals(series), _yresolved(member)


#: Interior budget samples a rung must contribute for its residual to
#: be a maximum over something rather than a single draw.
MIN_SAMPLES = 15


# ── The wall-normal-resolved budget on the same ladder ───────────────


def _yresolved(member: Path) -> dict[str, float]:
    r"""Relate ``twin_ybudget.bin`` to ``twin_budget.dat``.

    The stream names its own budget form in the sidecar's ``terms``,
    and the identities available differ between the two, so this
    dispatches on that rather than on a flag.

    *Algebraic* -- the same Parseval sum regrouped, so they hold to
    rounding at every rung and on any state.  They are the
    load-bearing check that the marginal reduction, the `$\pm k_z$`
    fold and the density normalisation are all right, here on real
    turbulence rather than the synthetic states of
    ``tests/test_twin_unit.py``:

    - ``V_exact``: `$-\sum_k\int \mathcal{V}$` against ``eps_tot``.
      Form-independent -- ``V`` is the same array either way.
    - ``P_exact``: the production identity.  **Convectively** the
      whole of it, `$\sum_k\int(P_U + P_r)$` against ``P_tot``;
      **rotationally** only the mean-gradient part, `$\sum_k\int
      P_\text{lift}$` against the three ``P_*(*,rU)`` columns, since
      the rotational production is no longer an algebraic identity.
    - ``T_zero`` (rotational only): `$\max_y|\sum_k T(y)|$` over both
      transfer terms, relative to `$\max_y\sum_k|T|$`.  Zero because
      `$\mathbf{a}\cdot(\mathbf{a}\times\mathbf{b}) = 0$` pointwise,
      hence resolution-independent.  It has **no convective
      counterpart**: there `$\sum_k T(y)$` is the turbulent transport
      of difference energy, a real flux that is zero only after
      integrating in `$y$`.

    *Truncation*-limited, converging rather than holding:

    - ``T_bins`` (convective only): the `$k$`-set sums of the transfer
      terms against the per-component transport of the paper's
      (2.14)-(2.16).  The two differ by the same-bin triads
      `$\tfrac12\langle(\mathbf{b}\cdot\nabla)|\Delta\mathbf{a}|^2
      \rangle$` that the paper's lists omit because they vanish for
      solenoidal `$\mathbf{b}$` -- continuously.  Discretely they
      vanish only as the integration-by-parts residual that makes
      ``T_tot`` non-zero, so ``T_bins`` and ``T_tot`` are one quantity
      seen twice.
    - ``N_tot`` (rotational only): `$\sum_k\int(P_U + P_r)$` against
      ``P_tot + T_tot``.  The two nonlinear forms differ by the
      gradient of `$\mathbf{u}^{(1)}\!\cdot\Delta\mathbf{u} +
      |\Delta\mathbf{u}|^2/2$`, whose work is a wall-normal flux --
      zero continuously, and discretely the same residual again.
    - ``pi_flux``: `$\max_k|\int W_p\,\mathrm{d}y|$`, which the
      continuity identity forces to zero at every mode, normalised by
      the largest `$|\int\mathcal{V}\,\mathrm{d}y|$` over the same
      modes.  Form-*dependent* in magnitude though not in kind: the
      rotational ``Wp`` is the work of the Bernoulli pressure, a
      larger and rougher field than the static one, so its product-
      rule residual is correspondingly larger.
    """
    from dnsjax.analysis.twin import integrate_y, read_twin_ybudget

    data = read_twin_ybudget(member)
    rotational = "P_lift" in data.meta["terms"]
    budget = read_dat(member / "twin_budget.dat")
    t_b = np.round(budget["t"], 10)
    index = {t: i for i, t in enumerate(t_b)}
    names = ["P_exact", "V_exact", "pi_flux"]
    names += ["T_zero", "N_tot"] if rotational else ["T_bins"]
    out = dict.fromkeys(names, 0.0)
    rows = [f"P_{b}({b},rU)" for b in ("dU", "du1", "du2")]

    def bins(name: str, k: int) -> np.ndarray:
        x0 = integrate_y(data, f"{name}_x0")[k]
        x = integrate_y(data, f"{name}_x")[k]
        return np.array([x0[0], x0[1:].sum(), (x - x0).sum()])

    def transport_bins(i: int) -> np.ndarray:
        """The paper's per-component transport at ``twin_budget`` row *i*."""
        return np.array(
            [
                sum(budget[c][i] for c in budget if c.startswith(f"T_{b}("))
                for b in ("dU", "du1", "du2")
            ]
        )

    for k, t in enumerate(np.round(data.t, 10)):
        i = index.get(t)
        if i is None:  # the driver's unconditional final row
            continue
        v_sum = -integrate_y(data, "V_z")[k].sum()
        e_ref = max(abs(budget["eps_tot"][i]), 1e-300)
        out["V_exact"] = max(
            out["V_exact"], abs(v_sum - budget["eps_tot"][i]) / e_ref
        )

        p_rot = (
            integrate_y(data, "P_U_x")[k].sum()
            + integrate_y(data, "P_r_x")[k].sum()
        )
        if rotational:
            want_p = sum(budget[r][i] for r in rows)
            out["P_exact"] = max(
                out["P_exact"],
                abs(integrate_y(data, "P_lift_x")[k].sum() - want_p)
                / max(abs(want_p), 1e-300),
            )
            for name in ("T_vort", "T_self"):
                dens = data[f"{name}_x"][k]
                scale = max(np.abs(dens).sum(axis=-1).max(), 1e-300)
                out["T_zero"] = max(
                    out["T_zero"], np.abs(dens.sum(axis=-1)).max() / scale
                )
            n_conv = budget["P_tot"][i] + budget["T_tot"][i]
            out["N_tot"] = max(
                out["N_tot"],
                abs(p_rot - n_conv) / max(abs(budget["P_tot"][i]), 1e-300),
            )
        else:
            out["P_exact"] = max(
                out["P_exact"],
                abs(p_rot - budget["P_tot"][i])
                / max(abs(budget["P_tot"][i]), 1e-300),
            )
            want = transport_bins(i)
            got = bins("T_ref", k) + bins("T_self", k)
            scale = max(np.abs(want).max(), 1e-300)
            out["T_bins"] = max(
                out["T_bins"], np.abs(got - want).max() / scale
            )

        pi = np.abs(integrate_y(data, "Wp_x")[k])
        visc = np.abs(integrate_y(data, "V_x")[k])
        out["pi_flux"] = max(
            out["pi_flux"], pi.max() / max(visc.max(), 1e-300)
        )
    return out


#: The three algebraic identities of :func:`_yresolved` are exact up to
#: float summation order over `$O(10^4)$` modes and `$N_y$` quadrature
#: nodes, so the bound is a rounding allowance, not a physics one.
YEXACT_TOL = 1e-9

#: Names asserted under :data:`YEXACT_TOL`.
_YEXACT = ("P_exact", "V_exact", "T_zero")


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
        # A (y, k) name is absent when the run used the other budget
        # form, which names a different set (:func:`_yresolved`).
        if name not in finest:
            continue
        if not finest[name] < bound:
            bad.append(
                f"ny={ladder[-1]} {name}={finest[name]:.2e} >= {bound:.2e}"
            )
    if len(ladder) < 2:
        return bad
    for name in (n for n in cfg["converge"] if n in rows[ladder[0]]):
        seq = [rows[ny][name] for ny in ladder]
        if not seq[-1] < seq[0]:
            trail = " -> ".join(f"{v:.2e}" for v in seq)
            bad.append(f"{name} did not shrink along the ladder ({trail})")
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
                resid, yres = _measure(cfg, ny, root, seed, spin, horizon)
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
                for name, value in yres.items():
                    worst[name] = max(worst.get(name, 0.0), value)
                print(
                    "           (y,k): "
                    + " ".join(f"{k}={yres[k]:.2e}" for k in sorted(yres)),
                    flush=True,
                )
                if not _measuring(args):
                    for name in (n for n in _YEXACT if n in yres):
                        if not yres[name] < YEXACT_TOL:
                            bad.append(
                                f"ny={ny} seed={seed}: {name}="
                                f"{yres[name]:.2e} >= {YEXACT_TOL:.0e} -- "
                                "the (y,k) budget no longer regroups to "
                                "the volume-averaged one"
                            )
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
        "--only",
        default=None,
        help="Comma-separated config names, matched exactly",
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
        # Exact names, not a substring filter: ``--only pp-cbv`` must
        # not silently drag in ``pp-cbv-span`` and double the ladder.
        want = [n.strip() for n in args.only.split(",") if n.strip()]
        known = [c["name"] for c in CONFIGS]
        unknown = [n for n in want if n not in known]
        assert not unknown, (
            f"--only: unknown config(s) {unknown}; choose from {known}"
        )
        configs = [c for c in configs if c["name"] in want]
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
