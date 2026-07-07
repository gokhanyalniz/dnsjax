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
arguments, hence the copies)::

    uv run python tests/test_temporal_order.py            # both studies
    uv run python tests/test_temporal_order.py --study kolmogorov
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

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

# Corrector setup: converge to TOL, assert every corrector-bearing
# step reached TOL_ASSERT -- except Kolmogorov's, whose corrector
# stalls at a pre-existing dt^2-scaled floor (~0.02 dt^2 here; see
# the module docstring): its threshold is dt^2-scaled with margin.
TOL = 1e-9
TOL_ASSERT = 1e-7
KOLM_FLOOR = 0.05  # * dt^2

# Accepted slope band for order 2 (log2 error ratio per dt halving).
ORDER_LO, ORDER_HI = 1.6, 2.4

STUDIES = ["kolmogorov", "plane-couette"]

FLOW_MODULES = {
    "plane-couette": "dnsjax.flows.wall_bounded.plane_couette",
    "kolmogorov": "dnsjax.flows.triply_periodic.monochromatic",
}


def _worker(system: str, scheme: str, dt: float, out: str) -> None:
    """Integrate to ``T_END`` with (*system*, *scheme*, *dt*); save the
    final spectral state to *out* (.npy)."""
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
    state = generate_random_state(amp, SMOOTH, SEED)
    n_steps = round(T_END / dt)
    assert abs(n_steps * dt - T_END) < 1e-12

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

    if scheme == "cnab2":
        # __main__ bootstrap: discarded priming call seeds the AB2
        # history, the first integration step is iterative-CN.
        _, carry, _, _ = fmod.step_cnab2(
            jnp.copy(state), jnp.zeros_like(state)
        )
        state, err, _ = fmod.predict_and_fully_correct(state)
        _converged(err, 0)
        for i in range(n_steps - 1):
            state, carry, err, _ = fmod.step_cnab2(state, carry)
            _converged(err, i + 1)
    else:
        for i in range(n_steps):
            state, err, _ = fmod.predict_and_fully_correct(state)
            _converged(err, i)

    np.save(out, np.asarray(state))


def _run(system: str, scheme: str, dt: float, out: Path) -> None:
    result = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
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
    parser.add_argument("--worker", choices=STUDIES, default=None)
    parser.add_argument("--scheme", default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.worker:
        _worker(args.worker, args.scheme, args.dt, args.out)
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

        if "kolmogorov" in studies:
            print("=== kolmogorov: cnab2 order (corrector-free ref) ===")
            ref = tdir / "kolm_ref.npy"
            _run("kolmogorov", "cnab2", DT_REF, ref)
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

    print("ALL PASSED")


if __name__ == "__main__":
    main()
