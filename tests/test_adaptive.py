#!/usr/bin/env python3
r"""Adaptive-dt guards (offline, in-process, no ``mpirun``).

Pins the ``step.adaptive`` machinery end to end:

- **Controller units** (JAX-free, parent process):
  :func:`dnsjax.adaptive.propose_dt` -- growth capped by
  ``dt_max_change`` and ``dt_max``, shrink uncapped but floored by
  ``dt_min`` / ``dt_min_change``, the relative deadband (both
  directions, incl. exactly at the threshold), and the
  zero/non-finite-CFL grow-to-cap branch.
- **Rebuild-vs-fresh leaf parity** (subprocess per system, forced
  single CPU device, x64): build the flow at ``DT0``, step once, then
  ``set_dt(DT1)`` and compare every ``dt``-dependent leaf against a
  freshly constructed flow at ``DT1`` (direct ``params.step.dt``
  assignment before construction -- the ``test_cnab2`` idiom).  The
  jitted rebuild's fusion may differ from the eager setup build, so
  the comparison is a tight ``allclose``, not bitwise.
- **Step parity at the new dt**: one CN/AB2 and one iterative-CN step
  of the module steppers (rebuilt leaves, ``ab2_kappa`` reset) vs the
  same steps of freshly built steppers at ``DT1`` -- catches any
  ``params.step.dt`` still baked into the module trace.
- **No-recompile guard**: with ``jax_log_compiles`` captured, a
  second ``set_dt`` plus plain+measured steps of both schemes must
  produce **zero** JAX log records (no trace, no compile) -- the
  core promise that a dt change swaps pytree leaves only.
- **Live-dt embed**: ``recorded_params_dump`` follows a mutated
  ``params.step.dt`` (the invariant that makes every snapshot embed
  the adapted dt).

The plane-Couette case runs under both solver backends (pallas and
dense); ``viscoelastic-dean`` runs with conformation diffusion
enabled so the ``Hc_op`` rebuild is exercised.

Usage::

    uv run python tests/test_adaptive.py                 # everything
    uv run python tests/test_adaptive.py --unit-only     # controller
    uv run python tests/test_adaptive.py --system pipe   # one system
"""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import sys

from _live import run_live

sys.stdout.reconfigure(line_buffering=True)

# Small but nontrivial resolutions (see ``test_cnab2``).
NX, NY, NZ = 8, 17, 8
NY_PERIODIC = 16
LX, LZ = 5.0, 5.0
AMP, SMOOTH, SEED = 0.1, 0.4, 1

# The dt ladder exercised by ``set_dt`` (all well inside stability).
DT0, DT1, DT2 = 0.01, 0.004, 0.007
PARITY_RTOL = 1e-10

SYSTEMS = [
    "plane-couette",
    "pipe",
    "taylor-couette",
    "viscoelastic-dean",
    "kolmogorov",
]

FLOW_MODULES = {
    "plane-couette": "dnsjax.flows.wall_bounded.plane_couette",
    "pipe": "dnsjax.flows.wall_bounded.pipe",
    "taylor-couette": "dnsjax.flows.wall_bounded.taylor_couette",
    "viscoelastic-dean": "dnsjax.flows.wall_bounded.viscoelastic_dean",
    "kolmogorov": "dnsjax.flows.triply_periodic.monochromatic",
}

GEO_MODULES = {
    "plane-couette": "dnsjax.geometries.wall_bounded.cartesian",
    "pipe": "dnsjax.geometries.wall_bounded.cylindrical",
    "taylor-couette": "dnsjax.geometries.wall_bounded.annular",
    "viscoelastic-dean": (
        "dnsjax.geometries.wall_bounded.annular_viscoelastic"
    ),
    "kolmogorov": "dnsjax.geometries.triply_periodic.triply_periodic",
}

STEPPER_BUILDERS = {
    "plane-couette": "build_cartesian_stepper",
    "pipe": "build_cylindrical_stepper",
    "taylor-couette": "build_annular_stepper",
    "viscoelastic-dean": "build_viscoelastic_stepper",
    "kolmogorov": "build_triply_periodic_stepper",
}

# The dt-dependent leaf set per system (the geometry
# ``_build_dt_leaves`` contract; ``ab2_kappa`` is deliberately
# excluded -- it tracks the change history, not dt itself).
LEAVES = {
    "plane-couette": (
        "dt",
        "Hk_op",
        "v1",
        "v2",
        "q1",
        "q2",
        "M_inv",
        "h_bulk_response",
        "H_bulk_inv",
    ),
    "pipe": (
        "dt",
        "Hk_op",
        "v_plus_1",
        "v_minus_1",
        "q_z_1",
        "M_inv",
        "h_bulk_response",
        "H_bulk_inv",
    ),
    "taylor-couette": (
        "dt",
        "Hk_op",
        "v_plus_1",
        "v_minus_1",
        "q_z_1",
        "v_plus_2",
        "v_minus_2",
        "q_z_2",
        "M_inv",
        "h_bulk_response",
        "H_bulk_inv",
    ),
    "kolmogorov": ("dt", "ldt_1", "ildt_2"),
}
LEAVES["viscoelastic-dean"] = LEAVES["taylor-couette"] + ("Hc_op",)

# How ``res.consistent_imm`` changes the dt-dependent leaf set.  A
# missing key here would be silent: ``set_dt`` only assigns what the
# rebuild returns, so a stale column would pair a new-dt ``Hk_op``
# with an old-dt response.
#
# All three geometries switch to a reconstruction scheme, which has no
# pressure: the primitive scheme's pressure-response columns go away
# (``DROPPED_LEAVES``) and are replaced by the wall-normal-velocity
# responses of the two-solve chain.  ``Lk_op`` is deliberately absent
# from both sets -- flag-on it holds the ``dt``-free recovery operator,
# so ``set_dt`` must not rebuild it.
CLOSURE_LEAVES = {
    "plane-couette": ("phi1", "phi2"),
    # One u_r column per wall.
    "taylor-couette": ("ur_1", "ur_2"),
    # The pipe's single wall gives one (a 1x1 influence matrix).
    "pipe": ("ur_1",),
}

# Leaves a ``res.consistent_imm`` build does *not* have.
DROPPED_LEAVES = {
    "plane-couette": ("q1", "q2"),
    "taylor-couette": (
        "v_plus_1",
        "v_minus_1",
        "q_z_1",
        "v_plus_2",
        "v_minus_2",
        "q_z_2",
    ),
    "pipe": ("v_plus_1", "v_minus_1", "q_z_1"),
}


# ── controller units (JAX-free) ──────────────────────────────────


def run_unit_checks() -> None:
    """``propose_dt`` restriction/deadband semantics, pure floats."""
    from dnsjax.adaptive import propose_dt

    kw = dict(
        cfl_target=0.5,
        dt_min=1e-6,
        dt_max=1.0,
        dt_min_change=0.0,
        dt_max_change=1.2,
        dt_threshold=0.05,
    )
    # Growth is capped by the per-evaluation ratio ...
    assert propose_dt(0.1, 0.01, **kw) == 1.2 * 0.01
    # ... and by dt_max (ideal 4.5, ratio cap 1.08, dt_max 1.0).
    assert propose_dt(0.1, 0.9, **kw) == 1.0
    # Zero / non-finite CFL carries no signal: grow to the caps.
    assert propose_dt(0.0, 0.01, **kw) == 1.2 * 0.01
    assert propose_dt(float("nan"), 0.01, **kw) == 1.2 * 0.01
    # Exact ideal when unclamped (a shrink beyond the deadband).
    assert math.isclose(propose_dt(1.0, 0.01, **kw), 0.005, rel_tol=1e-14)
    # Shrink is uncapped by default (two decades in one evaluation)...
    assert math.isclose(propose_dt(50.0, 0.01, **kw), 1e-4, rel_tol=1e-12)
    # ... floored at dt_min ...
    assert propose_dt(1e9, 0.01, **kw) == 1e-6
    # ... and dt_min_change floors the ratio when set.
    kw_ratio = kw | {"dt_min_change": 0.5}
    assert propose_dt(50.0, 0.01, **kw_ratio) == 0.5 * 0.01
    # Relative deadband: sub-threshold proposals keep dt exactly,
    # in both directions and exactly at the threshold boundary.
    assert propose_dt(0.5 / 1.04, 0.01, **kw) == 0.01
    assert propose_dt(0.5 / 0.96, 0.01, **kw) == 0.01
    kw_thr = kw | {"dt_threshold": 0.2}
    assert propose_dt(0.1, 0.01, **kw_thr) == 0.01  # cap 1.2 == thr


# ── worker (subprocess per system, forced 1 CPU device) ──────────


def _configure(
    system: str, backend: str, consistent_imm: bool = False
) -> None:
    """Configure JAX + the parameter singletons (1 CPU device, x64)."""
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

    phys: dict = {"system": system, "re": 100.0}
    geo: dict = {"lx": LX, "lz": LZ}
    if system == "taylor-couette":
        phys.update(re1=100.0, re2=0.0)
        geo["eta"] = 0.5
    elif system == "viscoelastic-dean":
        # Nonzero conformation diffusion so the Hc_op rebuild path is
        # exercised (kappa = 0 would drop the operator entirely).
        phys = {
            "system": system,
            "el": 20.0,
            "wi": 20.0,
            "beta": 0.8,
            "epsilon": 0.0,
            "kappa": 1e-3,
        }
        geo = {"lx": LX}

    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": "cpu"},
            phys=phys,
            geo=geo,
            res={
                "nx": NX,
                "ny": NY_PERIODIC if system == "kolmogorov" else NY,
                "nz": NZ,
                "fd_order": 4,
                "consistent_imm": consistent_imm,
                "double_precision": True,
            },
            step={"scheme": "cnab2", "dt": DT0},
            solver={"backend": backend},
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)


class _LogCapture(logging.Handler):
    """Collects every record routed through the ``jax`` logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _leaf_arrays(val: object) -> dict[str, object]:
    """Comparable array set of a leaf (operators -> their factors)."""
    if hasattr(val, "L"):  # PerModeBandedPallasOperator
        return {".L": val.L, ".U": val.U}
    if hasattr(val, "lu"):  # DenseJAXSolver
        return {".lu": val.lu, ".perm": val.perm}
    return {"": val}


def _worker(system: str, backend: str, consistent_imm: bool = False) -> None:
    _configure(system, backend, consistent_imm)

    import jax
    import jax.numpy as jnp
    import numpy as np

    fmod = importlib.import_module(FLOW_MODULES[system])
    gmod = importlib.import_module(GEO_MODULES[system])

    from dnsjax.parameters import params
    from dnsjax.random_field import generate_random_state

    # ICs are physical; the steppers work in the solver basis (the
    # same single crossing ``__main__`` performs).
    to_solver = getattr(fmod, "to_solver_basis", lambda s: s)
    state0 = to_solver(generate_random_state(AMP, SMOOTH, SEED))

    # Warm every stepper variant at DT0 (donated args -> copies).
    _, carry, _, _ = fmod.step_cnab2(jnp.copy(state0), jnp.zeros_like(state0))
    *_, m0 = fmod.step_cnab2_measured(jnp.copy(state0), jnp.copy(carry))
    fmod.predict_and_fully_correct(jnp.copy(state0))
    *_, m1 = fmod.predict_and_fully_correct_measured(jnp.copy(state0))
    assert float(m0["dt"]) == DT0 and float(m1["dt"]) == DT0

    # First set_dt: compiles the leaf rebuild (once), swaps leaves.
    fmod.set_dt(DT1)
    assert float(fmod.flow.dt) == DT1
    assert math.isclose(float(fmod.flow.ab2_kappa), DT1 / DT0)

    # -- rebuild-vs-fresh leaf parity ------------------------------
    params.step.dt = DT1  # direct assignment before construction
    fresh = type(fmod.flow)()
    dropped = DROPPED_LEAVES.get(system, ()) if consistent_imm else ()
    leaf_names = tuple(n for n in LEAVES[system] if n not in dropped) + (
        CLOSURE_LEAVES[system] if consistent_imm else ()
    )
    for name in leaf_names:
        got_leaf = getattr(fmod.flow, name)
        want_leaf = getattr(fresh, name)
        for suffix, got in _leaf_arrays(got_leaf).items():
            want = _leaf_arrays(want_leaf)[suffix]
            got_np, want_np = np.asarray(got), np.asarray(want)
            if suffix == ".perm":
                assert np.array_equal(got_np, want_np), name + suffix
                continue
            scale = max(1.0, float(np.max(np.abs(want_np))))
            np.testing.assert_allclose(
                got_np,
                want_np,
                rtol=PARITY_RTOL,
                atol=PARITY_RTOL * scale,
                err_msg=f"{system}: leaf {name}{suffix}",
            )
    print(
        f"{system}: {len(leaf_names)} rebuilt leaves match a "
        f"fresh dt={DT1} build"
    )

    # -- step parity at the new dt ---------------------------------
    # Fresh steppers at DT1 bake the (migrated) dt everywhere a
    # stale ``params.step.dt`` read would hide; identical inputs
    # through both must agree.  kappa is reset so both flows step
    # with the uniform-step AB2 weights.
    fmod.reset_ab2_kappa()
    tup = getattr(gmod, STEPPER_BUILDERS[system])(fresh)
    a_cn, a_carry, *_ = fmod.step_cnab2(jnp.copy(state0), jnp.copy(carry))
    b_cn, b_carry, *_ = tup[5](jnp.copy(state0), jnp.copy(carry))
    np.testing.assert_allclose(
        np.asarray(a_cn),
        np.asarray(b_cn),
        rtol=PARITY_RTOL,
        atol=PARITY_RTOL,
        err_msg=f"{system}: cnab2 step parity",
    )
    np.testing.assert_allclose(
        np.asarray(a_carry),
        np.asarray(b_carry),
        rtol=PARITY_RTOL,
        atol=PARITY_RTOL,
        err_msg=f"{system}: cnab2 carry parity",
    )
    a_ic, *_ = fmod.predict_and_fully_correct(jnp.copy(state0))
    b_ic, *_ = tup[3](jnp.copy(state0))
    np.testing.assert_allclose(
        np.asarray(a_ic),
        np.asarray(b_ic),
        rtol=PARITY_RTOL,
        atol=PARITY_RTOL,
        err_msg=f"{system}: iterative-cn step parity",
    )
    print(
        f"{system}: module steppers at set_dt({DT1}) == fresh "
        "steppers (cnab2 + iterative-cn)"
    )

    # -- no-recompile guard ----------------------------------------
    # Everything is warm: a further set_dt + every stepper variant
    # must neither trace nor compile anything (leaf swaps only).
    jax.block_until_ready((a_cn, a_ic))
    cap = _LogCapture()
    jax_logger = logging.getLogger("jax")
    old_level = jax_logger.level
    jax.config.update("jax_log_compiles", True)
    jax_logger.addHandler(cap)
    jax_logger.setLevel(logging.DEBUG)
    try:
        fmod.set_dt(DT2)
        s_cn, c_cn, *_ = fmod.step_cnab2(jnp.copy(state0), jnp.copy(carry))
        *_, m2 = fmod.step_cnab2_measured(jnp.copy(s_cn), jnp.copy(c_cn))
        s_ic, *_ = fmod.predict_and_fully_correct(jnp.copy(state0))
        *_, m3 = fmod.predict_and_fully_correct_measured(jnp.copy(state0))
        jax.block_until_ready((s_ic, m2["dt"], m3["dt"]))
    finally:
        jax_logger.removeHandler(cap)
        jax_logger.setLevel(old_level)
        jax.config.update("jax_log_compiles", False)
    assert not cap.messages, (
        f"{system}: set_dt({DT2}) retraced/recompiled:\n"
        + "\n".join(cap.messages)
    )
    assert float(m2["dt"]) == DT2 and float(m3["dt"]) == DT2
    print(
        f"{system}: set_dt({DT2}) + all stepper variants: 0 traces, 0 compiles"
    )

    # -- kappa carry-cancellation identity -------------------------
    # The variable-step AB2 forcing is
    # F = N^n + (kappa/2) (N^n - carry): with carry == N^n the step
    # must be kappa-independent, and with a different carry it must
    # not be.  Pins that kappa multiplies exactly (N^n - carry) in
    # BOTH cnab2 branches (the wall-bounded _cnab2_lbf_core and the
    # plain triply-periodic forcing).
    _, n_ref, *_ = fmod.step_cnab2(jnp.copy(state0), jnp.zeros_like(state0))
    fmod.reset_ab2_kappa()
    a_id, *_ = fmod.step_cnab2(jnp.copy(state0), jnp.copy(n_ref))
    fmod.flow.ab2_kappa = jnp.asarray(0.7, dtype=fmod.flow.dt.dtype)
    b_id, *_ = fmod.step_cnab2(jnp.copy(state0), jnp.copy(n_ref))
    c_id, *_ = fmod.step_cnab2(jnp.copy(state0), jnp.zeros_like(state0))
    fmod.reset_ab2_kappa()
    np.testing.assert_allclose(
        np.asarray(a_id),
        np.asarray(b_id),
        rtol=1e-9,
        atol=1e-11,
        err_msg=f"{system}: step not kappa-independent at carry==N^n",
    )
    assert not np.allclose(
        np.asarray(b_id), np.asarray(c_id), rtol=1e-9, atol=1e-11
    ), f"{system}: kappa has no effect on the AB2 history term"
    print(f"{system}: kappa multiplies exactly (N^n - carry)")

    # -- live-dt snapshot embed ------------------------------------
    from dnsjax.param_surface import recorded_params_dump

    params.step.dt = 0.123456
    dump = recorded_params_dump(params)
    assert dump["step"]["dt"] == 0.123456, dump["step"]
    print(f"{system}: recorded_params_dump follows the live dt")


# ── runner ───────────────────────────────────────────────────────


def _run_worker(
    system: str, backend: str, consistent_imm: bool = False
) -> None:
    label = f"{system}[{backend}]"
    if consistent_imm:
        label += "[consistent_imm]"
    print(f"=== {label} ===")
    result = run_live(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--worker",
            system,
            "--backend",
            backend,
        ]
        + (["--consistent-imm"] if consistent_imm else [])
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{label}: worker failed (rc={result.returncode})"
        )
    print(f"  PASS  {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", metavar="SYSTEM", help=argparse.SUPPRESS)
    parser.add_argument(
        "--backend", default="pallas", choices=["pallas", "dense"]
    )
    parser.add_argument(
        "--system",
        action="append",
        choices=SYSTEMS,
        help="restrict to one or more systems (repeatable)",
    )
    parser.add_argument(
        "--consistent-imm", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="run only the JAX-free controller units",
    )
    args = parser.parse_args()

    if args.worker:
        _worker(args.worker, args.backend, args.consistent_imm)
        return

    run_unit_checks()
    print("  PASS  propose_dt units")
    if args.unit_only:
        print("\nAdaptive-dt unit checks passed (--unit-only).")
        return

    systems = args.system or SYSTEMS
    cases = [(s, "pallas") for s in systems]
    if "plane-couette" in systems:
        # The dense backend shares the rebuild contract; one geometry
        # covers its DenseJAXSolver/from_factors path.
        cases.append(("plane-couette", "dense"))
    # ``res.consistent_imm`` changes the dt-dependent leaf set in every
    # geometry (the reconstruction scheme has no pressure, so its
    # wall-normal-velocity columns replace the pressure responses);
    # one case per implementation.
    cases += [
        (s, "pallas", True)
        for s in ("plane-couette", "taylor-couette", "pipe")
        if s in systems
    ]
    for system, backend, *rest in cases:
        _run_worker(system, backend, bool(rest and rest[0]))
    print("\nAll adaptive-dt checks passed.")


if __name__ == "__main__":
    main()
