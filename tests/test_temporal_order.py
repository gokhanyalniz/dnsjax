#!/usr/bin/env python3
r"""Temporal-order guard for the two time-stepping schemes (offline).

Pins the second-order accuracy of ``step.scheme == "cnab2"`` against
``"iterative-cn"`` with two complementary studies, each stepping
in-process on 1 forced CPU device (no ``mpirun``), one subprocess per
``(system, scheme, dt)`` (the singletons capture ``dt`` at trace time,
so a ``dt`` sweep needs a subprocess per value):

- **Kolmogorov (triply-periodic), absolute order**: the pressure
  projection is algebraically exact there and the triply-periodic
  cnab2 step has *no corrector*, so cnab2 self-convergence against a
  fine-``dt`` cnab2 reference is floor-free and falls at a clean
  slope 2 under ``dt`` halving.  Iterative-CN runs at the largest
  and smallest ``dt`` are compared against the same reference to pin
  that both schemes converge to the *same* limit (same equation),
  which upgrades the self-convergence into an absolute-order
  statement; icn's error there is floor-dominated and first order
  (below), so the check is a ceiling plus ~linear decay, not
  proximity to cnab2's own error.
- **plane-Couette (wall-bounded), scheme-difference order**: both
  schemes share the same IMM projection-splitting error, so the
  *difference* of their final states at matched ``dt`` isolates the
  nonlinear treatment (iterated-CN vs AB2 + implicit coupling) and
  must also fall at slope ~2.  Runs with the default
  ``implicit_mean_coupling`` on, so the mean-flow coupling's CN
  treatment is covered by the order check too.
- **plane-Couette, ``res.consistent_imm`` contrast**: the difference
  proxy above cancels the shared projection error, so it cannot judge
  a change that *removes* that error.  This study instead measures
  each configuration's own self-convergence and asserts the flag
  strictly improves both the error size and its decay rate (measured:
  1.3e-2 at order ~0.5 -> 3.3e-5 at order ~1.1).  The wall-bounded
  absolute order is **not** 2 either way -- the projection splitting
  sets it -- which is exactly why the study above compares schemes
  rather than dts.

Corrector-bearing runs use a tight tolerance (``1e-9``) so
fixed-point error does not pollute the truncation-error measurement,
and every step **asserts** its corrector converged -- an unconverged
corrector silently degrades a step to first order (exactly what an
early version of this test measured on Kolmogorov: clean *order 1*).

**Triply-periodic corrector floor (pre-existing).**  The
triply-periodic iterative-CN corrector does not converge to machine
precision: it stalls on a near-neutral direction (per-iteration rate
~0.9995 while the bulk contracts at ~0.006/iteration) at a
``dt^2``-scaled amplitude -- measured ``~0.02 dt^2`` at this config
(``2.98e-7`` at ``dt = 0.004``), bit-identical on the pre-``optimize``
``read_state`` branch, so it is a property of the algebraic
projection/CN interplay, not a regression.  Accumulated over ``T/dt``
steps (``O(T C dt)``) it dominates icn's global Kolmogorov error:
measured ``4.97e-2`` at ``dt = 0.004`` vs ``1.38e-2`` at
``dt = 0.001`` against the cnab2 reference -- *first-order* decay to
the same limit, ~70x above cnab2's own error at ``dt = 0.004``.
This is what produced the order-1 measurement above, and it is (part
of) the known "Kolmogorov corrector stall" noted in
``tests/test_random_smoke.py``.  The wall-bounded corrector has no
such floor (converges to ``~5e-11`` in 4-5 iterations at these
configs).  Hence: the Kolmogorov reference is corrector-free cnab2,
the per-step assert uses a ``dt^2``-scaled threshold for
Kolmogorov's corrector-bearing steps (the cnab2 self-start and the
icn cross-check), and the icn cross-check asserts same-limit
convergence (absolute ceiling + ~linear decay under a 4x ``dt``
reduction), not proximity to cnab2's error.

cnab2 self-starts exactly like ``__main__``: prime the AB2 history
with a discarded ``step_cnab2(copy(u0), zeros)`` call, take the first
step with iterative-CN, then chain cnab2 (the steppers donate their
arguments, hence the copies).

**Variable-step (vardt) studies.**  Two further studies drive the
adaptive-dt machinery under a *prescribed* dt sequence (no
controller) through the flow module's ``set_dt`` /
``reset_ab2_kappa`` hooks, sweeping the AB2 step ratio
`$\kappa = \Delta t_n / \Delta t_{n-1}$` through 1/2, 1, and 2 with
an on-device operator rebuild at every change.  Both must fall at
slope ~2 under base-dt halving -- a first-order-in-``dt`` kappa bug
shows up directly here.

- ``kolmogorov-vardt``: the dense dyadic pattern ``(d, d/2, d/2)``
  (a dt change at 2 of every 3 steps; each period sums exactly to
  ``2d``), measured as absolute order against the fixed-fine-dt
  reference -- pins the kappa-weighted plain-AB2 forcing and the
  ``ldt_1``/``ildt_2`` rebuild.
- ``plane-couette-vardt``: four equal-duration blocks
  ``(d, d/2, d, d/2)`` -- a **fixed count** of dt changes (3) for
  every base ``d`` -- measured as the cnab2-vs-icn difference order
  at the *matched* sequence; pins ``_cnab2_lbf_core``'s kappa, the
  Hk/IMM rebuild, and iterative-CN under mid-run ``set_dt``.  The
  change count must stay fixed because the wall-bounded IMM
  projection-splitting error -- the schemes' *shared*, dominant
  absolute error (measured ~6e-2 at ``dt = 0.01`` here, at fixed
  and variable dt alike; the periodic projection is algebraically
  exact and has no analogue) -- decorrelates between the schemes by
  `$O(\Delta t^2)$` per dt change: under per-step alternation
  (`$O(1/\Delta t)$` changes) their difference degrades to
  `$O(\Delta t)$` even with exact kappa weights, while a fixed
  change count keeps it `$O(\Delta t^2)$`.  The exact-kappa
  application at the change steps themselves is pinned separately by
  ``tests/test_adaptive.py``'s carry-cancellation identity.

::

    uv run python tests/test_temporal_order.py            # all studies
    uv run python tests/test_temporal_order.py --study kolmogorov
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import run_live

sys.stdout.reconfigure(line_buffering=True)

NX, NY, NZ = 8, 17, 8
NY_PERIODIC = 16
LX, LZ = 5.0, 5.0
SMOOTH, SEED = 0.4, 1

# Fixed horizon; every dt below divides it exactly.
T_END = 0.32
# Kolmogorov (absolute order vs a fine-dt corrector-free cnab2
# reference): dts sit below the known Kolmogorov iterative-CN
# corrector-rate cap (~0.005; see the random-smoke SYSTEMS note) so
# the icn cross-check and the cnab2 self-start step both converge.
AMP_KOLM = 0.02
DTS_KOLM = [0.004, 0.002, 0.001]
DT_REF = 0.0001  # Kolmogorov cnab2 reference, 3200 steps
# plane-Couette (scheme-difference order at matched dt).
AMP_PC = 0.1
DTS_PC = [0.01, 0.005, 0.0025]
# Self-convergence reference for the consistent_imm study (4x the
# finest DTS_PC entry).
DT_SELF_REF = 0.000625

# Corrector setup: converge to TOL, assert every corrector-bearing
# step reached TOL_ASSERT -- except Kolmogorov's, whose corrector
# stalls at a pre-existing dt^2-scaled floor (~0.02 dt^2 here; see
# the module docstring): its threshold is dt^2-scaled with margin.
TOL = 1e-9
TOL_ASSERT = 1e-7
KOLM_FLOOR = 0.05  # * dt^2

# Accepted slope band for order 2 (log2 error ratio per dt halving).
ORDER_LO, ORDER_HI = 1.6, 2.4

STUDIES = [
    "kolmogorov",
    "plane-couette",
    "plane-couette-consistent-imm",
    "kolmogorov-vardt",
    "plane-couette-vardt",
]
WORKER_SYSTEMS = ["kolmogorov", "plane-couette"]

FLOW_MODULES = {
    "plane-couette": "dnsjax.flows.wall_bounded.plane_couette",
    "kolmogorov": "dnsjax.flows.triply_periodic.monochromatic",
}


def _worker(
    system: str,
    scheme: str,
    dt: float,
    out: str,
    vardt: bool,
    consistent_imm: bool = False,
) -> None:
    """Integrate to ``T_END`` with (*system*, *scheme*, *dt*); save the
    final spectral state to *out* (.npy).  With *vardt*, step the
    dyadic ``(dt, dt/2, dt/2)`` pattern via ``set_dt`` (docstring)."""
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

    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": "cpu"},
            phys={"system": system, "re": 100.0},
            geo={"lx": LX, "lz": LZ},
            res={
                "nx": NX,
                "ny": NY_PERIODIC if system == "kolmogorov" else NY,
                "nz": NZ,
                "fd_order": 4,
                "consistent_imm": consistent_imm,
                "double_precision": True,
            },
            step={
                "scheme": scheme,
                "dt": dt,
                "corrector_tolerance": TOL,
                "max_corrector_iterations": 60,
            },
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)

    import importlib

    import jax.numpy as jnp

    fmod = importlib.import_module(FLOW_MODULES[system])
    from dnsjax.random_field import generate_random_state

    amp = AMP_KOLM if system == "kolmogorov" else AMP_PC
    # ICs are physical; the steppers work in the solver basis (the
    # same single crossing ``__main__`` performs).  The order study
    # compares states to each other, so it stays in that one basis.
    to_solver = getattr(fmod, "to_solver_basis", lambda s: s)
    state = to_solver(generate_random_state(amp, SMOOTH, SEED))
    if vardt and system == "kolmogorov":
        # Dense dyadic alternation: a change at 2 of every 3 steps.
        n_periods = round(T_END / (2 * dt))
        assert abs(2 * dt * n_periods - T_END) < 1e-12
        seq = [dt, dt / 2, dt / 2] * n_periods
    elif vardt:
        # Wall-bounded: four equal-duration blocks (d, d/2, d, d/2)
        # -- a fixed count of dt changes (3) for every base d, so the
        # per-change projection-splitting decorrelation stays
        # O(dt^2) total (module docstring).
        n_hi = round(T_END / 4 / dt)
        n_lo = 2 * n_hi
        seq = ([dt] * n_hi + [dt / 2] * n_lo) * 2
        assert abs(sum(seq) - T_END) < 1e-12
    else:
        n_steps = round(T_END / dt)
        assert abs(n_steps * dt - T_END) < 1e-12
        seq = [dt] * n_steps

    # Kolmogorov's corrector-bearing steps stall at the pre-existing
    # dt^2-scaled floor (module docstring); its cnab2 steps have no
    # corrector and report err = 0.
    thresh = (
        max(TOL_ASSERT, KOLM_FLOOR * dt * dt)
        if system == "kolmogorov"
        else TOL_ASSERT
    )

    def _converged(err, i: int) -> None:
        # An unconverged corrector silently degrades the temporal
        # order; fail the worker loudly instead.
        assert float(err) <= thresh, (
            f"{system} {scheme} dt={dt}: corrector not converged at "
            f"step {i} (err {float(err):.3e} > {thresh:.1e})"
        )

    # The dt-change discipline mirrors ``__main__``'s controller:
    # ``set_dt`` before the first step at a new dt (it also sets the
    # AB2 ratio kappa = new/old), ``reset_ab2_kappa`` after exactly
    # one step at the new dt.
    prev_dt = dt
    kappa_pending = False
    if scheme == "cnab2":
        # __main__ bootstrap: discarded priming call seeds the AB2
        # history, the first integration step is iterative-CN.
        _, carry, _, _ = fmod.step_cnab2(
            jnp.copy(state), jnp.zeros_like(state)
        )
        for i, step_dt in enumerate(seq):
            if step_dt != prev_dt:
                fmod.set_dt(step_dt)
                kappa_pending = True
            elif kappa_pending:
                fmod.reset_ab2_kappa()
                kappa_pending = False
            if i == 0:
                state, err, _ = fmod.predict_and_fully_correct(state)
            else:
                state, carry, err, _ = fmod.step_cnab2(state, carry)
            _converged(err, i)
            prev_dt = step_dt
    else:
        for i, step_dt in enumerate(seq):
            if step_dt != prev_dt:
                fmod.set_dt(step_dt)
            state, err, _ = fmod.predict_and_fully_correct(state)
            _converged(err, i)
            prev_dt = step_dt

    np.save(out, np.asarray(state))


def _run(
    system: str,
    scheme: str,
    dt: float,
    out: Path,
    *,
    vardt: bool = False,
    consistent_imm: bool = False,
) -> None:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        system,
        "--scheme",
        scheme,
        "--dt",
        repr(dt),
        "--out",
        str(out),
    ]
    if vardt:
        cmd.append("--vardt")
    if consistent_imm:
        cmd.append("--consistent-imm")
    result = run_live(cmd)
    if result.returncode != 0:
        raise SystemExit(f"worker failed: {system} {scheme} dt={dt}")


def _err(a: Path, b: Path) -> float:
    # Relative L2: the max-abs norm is dominated by the noisy high-k
    # tail of the random IC and hides the asymptotic slope.
    x, y = np.load(a), np.load(b)
    return float(np.linalg.norm(x - y) / np.linalg.norm(y))


def _check_orders(errs: list[float], label: str) -> None:
    orders = [np.log2(e1 / e2) for e1, e2 in zip(errs, errs[1:], strict=False)]
    print(f"{label}: errors {[f'{e:.3e}' for e in errs]}")
    print(f"{label}: orders {[f'{o:.2f}' for o in orders]}")
    assert all(e > 1e-12 for e in errs), (
        f"{label}: error at roundoff, slope meaningless ({errs})"
    )
    assert all(ORDER_LO <= o <= ORDER_HI for o in orders), (
        f"{label}: temporal order not ~2: {orders}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=STUDIES, default=None)
    parser.add_argument("--worker", choices=WORKER_SYSTEMS, default=None)
    parser.add_argument("--scheme", default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--vardt", action="store_true")
    parser.add_argument("--consistent-imm", action="store_true")
    args = parser.parse_args()

    if args.worker:
        _worker(
            args.worker,
            args.scheme,
            args.dt,
            args.out,
            args.vardt,
            args.consistent_imm,
        )
        return

    print(
        "Temporal-order guards: offline, 1 forced CPU device per "
        "(system, scheme, dt) (device-agnostic convergence checks; no "
        "GPU path).",
        flush=True,
    )
    studies = [args.study] if args.study else STUDIES
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)

        def _kolm_ref() -> Path:
            # Corrector-free fixed-fine-dt cnab2 reference, shared by
            # the kolmogorov and kolmogorov-vardt studies (built on
            # first use).
            ref = tdir / "kolm_ref.npy"
            if not ref.exists():
                _run("kolmogorov", "cnab2", DT_REF, ref)
            return ref

        if "kolmogorov" in studies:
            print("=== kolmogorov: cnab2 order (corrector-free ref) ===")
            ref = _kolm_ref()
            errs = []
            for dt in DTS_KOLM:
                out = tdir / f"kolm_cnab2_{dt}.npy"
                _run("kolmogorov", "cnab2", dt, out)
                errs.append(_err(out, ref))
            _check_orders(errs, "kolmogorov cnab2")
            # Same-limit cross-check: iterative-CN must land on the
            # same reference (same equation).  Its error here is the
            # accumulated corrector floor -- first order in dt and
            # ~70x cnab2's own error (module docstring) -- so assert
            # an absolute ceiling plus ~linear decay under a 4x dt
            # reduction (a wrong-equation offset would plateau).
            e_icn = []
            for dt in (DTS_KOLM[0], DTS_KOLM[-1]):
                icn = tdir / f"kolm_icn_{dt}.npy"
                _run("kolmogorov", "iterative-cn", dt, icn)
                e_icn.append(_err(icn, ref))
                print(f"kolmogorov icn@{dt}: {e_icn[-1]:.3e}")
            assert e_icn[0] <= 0.2, (
                f"iterative-cn far from the cnab2 limit: {e_icn[0]:.3e}"
            )
            ratio = e_icn[0] / e_icn[1]
            assert 2.0 <= ratio <= 8.0, (
                f"icn error not ~first-order toward the cnab2 "
                f"limit: {e_icn[0]:.3e} -> {e_icn[1]:.3e} "
                f"(ratio {ratio:.2f})"
            )

        if "plane-couette" in studies:
            print("=== plane-couette: cnab2 vs icn difference order ===")
            errs = []
            for dt in DTS_PC:
                a = tdir / f"pc_icn_{dt}.npy"
                b = tdir / f"pc_cnab2_{dt}.npy"
                _run("plane-couette", "iterative-cn", dt, a)
                _run("plane-couette", "cnab2", dt, b)
                errs.append(_err(b, a))
            _check_orders(errs, "plane-couette cnab2-icn")

        if "plane-couette-consistent-imm" in studies:
            # NOT the scheme-difference proxy of the study above: that
            # proxy works only while the shared IMM projection error
            # dominates *both* schemes and cancels in the difference,
            # and ``res.consistent_imm`` is precisely what removes it.
            # The honest measurement is each configuration's own
            # self-convergence against a fine-dt run of the same
            # configuration.
            print("=== plane-couette: consistent_imm self-convergence ===")
            slopes, first = {}, {}
            for cimm in (False, True):
                tag = "on" if cimm else "off"
                ref = tdir / f"pc_sc_{tag}_ref.npy"
                _run(
                    "plane-couette",
                    "iterative-cn",
                    DT_SELF_REF,
                    ref,
                    consistent_imm=cimm,
                )
                errs = []
                for dt in DTS_PC:
                    out = tdir / f"pc_sc_{tag}_{dt}.npy"
                    _run(
                        "plane-couette",
                        "iterative-cn",
                        dt,
                        out,
                        consistent_imm=cimm,
                    )
                    errs.append(_err(out, ref))
                orders = [
                    np.log2(a / b)
                    for a, b in zip(errs, errs[1:], strict=False)
                ]
                print(
                    f"consistent_imm={tag}: errors "
                    f"{[f'{e:.3e}' for e in errs]}  orders "
                    f"{[f'{o:.2f}' for o in orders]}"
                )
                slopes[tag], first[tag] = min(orders), errs[0]

            # The flag must strictly improve both the size of the
            # error and its decay rate.  Absolute numbers are recorded
            # rather than pinned: the ungated absolute order is ~0.5
            # (the wall-bounded projection-splitting error, ~6e-2 at
            # dt = 0.01), the gated one ~1.1 at ~1.2e-4.  What must
            # never regress is the *contrast*.
            assert first["off"] / first["on"] > 50.0, (
                "consistent_imm did not shrink the absolute temporal "
                f"error: {first['off']:.3e} -> {first['on']:.3e}"
            )
            assert slopes["on"] > slopes["off"] + 0.3, (
                "consistent_imm did not improve the convergence rate: "
                f"{slopes['off']:.2f} -> {slopes['on']:.2f}"
            )
            assert slopes["on"] > 1.0, (
                f"consistent_imm convergence rate below 1: {slopes['on']:.2f}"
            )

        if "kolmogorov-vardt" in studies:
            print("=== kolmogorov: cnab2 vardt order (set_dt seq) ===")
            ref = _kolm_ref()
            errs = []
            for dt in DTS_KOLM:
                out = tdir / f"kolm_cnab2_var_{dt}.npy"
                _run("kolmogorov", "cnab2", dt, out, vardt=True)
                errs.append(_err(out, ref))
            _check_orders(errs, "kolmogorov cnab2-vardt")

        if "plane-couette-vardt" in studies:
            print("=== plane-couette: vardt cnab2 vs icn difference ===")
            errs = []
            for dt in DTS_PC:
                a = tdir / f"pc_icn_var_{dt}.npy"
                b = tdir / f"pc_cnab2_var_{dt}.npy"
                _run("plane-couette", "iterative-cn", dt, a, vardt=True)
                _run("plane-couette", "cnab2", dt, b, vardt=True)
                errs.append(_err(b, a))
            _check_orders(errs, "plane-couette vardt cnab2-icn")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
