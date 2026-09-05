#!/usr/bin/env python3
r"""Automatic-differentiation guards (offline, in-process workers).

Pins the claim ``docs/differentiability.md`` makes: with an opt-in
fixed corrector count a dnsjax time step admits **reverse-mode**
automatic differentiation, and the gradient is right.

- **Reverse mode works, and matches a finite difference.**  For
  kolmogorov, plane-couette and pipe, under both time-stepping
  schemes, `$\partial E'(u^{n})/\partial u^{0}$` from :func:`jax.grad`
  is compared against a central difference along a random direction.
  This is the whole claim; a gradient that returns an array without
  being the derivative would pass every structural check.
- **The knob is load-bearing.**  The same configuration with the
  dynamic corrector (``step.corrector_iterations = 0``, the default)
  must *fail*, naming ``lax.while_loop``.  Without this row the
  fixed-count rows would pass whether or not the feature did anything.
- **A fixed count is the same integrator when it is long enough.**  A
  step taken with ``corrector_iterations`` well above what the dynamic
  corrector used reproduces the dynamic stepper's state to
  ``corrector_tolerance``.  The corrector count is read when
  ``make_stepper`` runs, so the second stepper is rebuilt in place --
  the ``tests/test_cnab2.py`` idiom for ``split_corrector``.
- **``solver.pallas_kernel`` selects the sweep.**
  :func:`dnsjax.solvers._kernel_path` must honour all three states, and
  the trace-only ``_force_kernel_path`` override must still win.

The banded solve's own adjoint -- the transposed Pallas sweep, the
kernel's ``custom_vjp`` against the portable sweep's autodiff, and the
finite differences on the operator cotangents including the
reciprocated diagonal slot -- is a property of the solver and is pinned
in ``tests/test_banded_solver.py`` instead.

Each row runs in its own subprocess: the parameter singletons and the
jitted steppers capture their configuration at import and trace time::

    uv run python tests/test_autodiff.py
    uv run python tests/test_autodiff.py --only grad-pipe
"""

from __future__ import annotations

import argparse
import os
import sys

from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

NX, NY, NZ = 8, 17, 8
NY_PERIODIC = 16
LX, LZ = 5.0, 5.0
AMP, SMOOTH, SEED = 0.1, 0.4, 1
WALL_SMOOTH, WALL_CONF = 0.4, 0.14
N_STEPS = 2
#: Fixed count for the differentiable rows.  Above what the dynamic
#: corrector uses at this ``dt`` (1--2 iterations for every flow), so
#: these steps are the converged ones.
N_FIXED = 3
#: Relative agreement demanded of gradient vs central difference.  The
#: floor is the difference itself: a central difference at
#: ``eps = 1e-6`` in double precision is good to about 1e-9.
FD_RTOL = 1e-7

FLOW_MODULES = {
    "kolmogorov": "dnsjax.flows.triply_periodic.monochromatic",
    "plane-couette": "dnsjax.flows.wall_bounded.plane_couette",
    "pipe": "dnsjax.flows.wall_bounded.pipe",
}
STEPPER_BUILDERS = {
    "plane-couette": (
        "dnsjax.geometries.wall_bounded.cartesian",
        "build_cartesian_stepper",
    ),
}

#: ``(name, system, scheme)`` of the finite-difference-verified rows.
GRAD_ROWS = (
    ("grad-kolmogorov-icn", "kolmogorov", "iterative-cn"),
    ("grad-kolmogorov-cnab2", "kolmogorov", "cnab2"),
    ("grad-plane-couette-icn", "plane-couette", "iterative-cn"),
    ("grad-plane-couette-cnab2", "plane-couette", "cnab2"),
    ("grad-pipe-icn", "pipe", "iterative-cn"),
)
OTHER_ROWS = (
    "dynamic-corrector-refused",
    "fixed-matches-dynamic",
    "kernel-path-selection",
)


def _configure(
    system: str, scheme: str, n_fixed: int, tol: float | None = None
) -> None:
    """Configure JAX and the parameter singletons (1 forced CPU device).

    Must run before importing ``sharding`` or any geometry module.
    """
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    periodic = system == "kolmogorov"
    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": "cpu"},
            phys={"system": system, "re": 100.0},
            geo={"lx": LX} if system == "pipe" else {"lx": LX, "lz": LZ},
            res={
                "nx": NX,
                "ny": NY_PERIODIC if periodic else NY,
                "nz": NZ,
                "fd_order": 4,
                "double_precision": True,
            },
            step={
                "scheme": scheme,
                "dt": 0.005,
                "corrector_iterations": n_fixed,
                **({} if tol is None else {"corrector_tolerance": tol}),
            },
        )
    )
    padded_res.set_padded_resolution(params)
    validate_parameters()


def _initial_state():
    """A random divergence-free state for the configured flow."""
    from dnsjax.ic.random_field import generate_random_state

    return generate_random_state(AMP, SMOOTH, WALL_SMOOTH, WALL_CONF, SEED)


def _energy_after_steps(mod, scheme: str):
    """``state -> E'`` after ``N_STEPS`` steps, safe to differentiate.

    The steppers donate their inputs, so a caller keeping its own state
    passes a copy -- otherwise the finite-difference evaluations below
    would consume the very array they perturb.
    """
    import jax.numpy as jnp

    def energy(state):
        s = jnp.copy(state)
        if scheme == "cnab2":
            carry = jnp.zeros_like(s)
            _, carry, *_ = mod.step_cnab2(jnp.copy(s), carry)
            for _ in range(N_STEPS):
                s, carry, *_ = mod.step_cnab2(s, carry)
        else:
            for _ in range(N_STEPS):
                s, *_ = mod.predict_and_fully_correct(s)
        return mod.get_perturbation_energy(s)

    return energy


# ── the rows ─────────────────────────────────────────────────────


def _check_grad(system: str, scheme: str) -> None:
    """``jax.grad`` of the stepped energy, against a central difference."""
    import importlib

    _configure(system, scheme, N_FIXED)

    import jax
    import jax.numpy as jnp

    mod = importlib.import_module(FLOW_MODULES[system])
    energy = _energy_after_steps(mod, scheme)
    s0 = _initial_state()

    _, tangent = jax.jvp(energy, (s0,), (jnp.ones_like(s0) * 1e-3,))
    assert jnp.isfinite(tangent), "jax.jvp produced a non-finite tangent"

    g = jax.grad(energy)(s0)
    direction = jax.random.normal(
        jax.random.key(0), s0.shape, dtype=jnp.float64
    ).astype(s0.dtype)
    eps = 1e-6
    fd = float(energy(s0 + eps * direction) - energy(s0 - eps * direction))
    fd /= 2 * eps
    # jax.grad of a real-valued function of a complex input returns the
    # conjugate cotangent, so the directional derivative is
    # Re<conj(g), d>.
    ad = float(jnp.real(jnp.sum(jnp.conj(g) * direction)))
    rel = abs(fd - ad) / max(abs(fd), 1e-300)
    assert abs(fd) > 1e-12, (
        f"{system}/{scheme}: the finite difference is ~0, so the "
        "comparison would pass vacuously; pick another direction"
    )
    assert rel < FD_RTOL, (
        f"{system}/{scheme}: grad {ad:.12e} vs central difference "
        f"{fd:.12e} (rel {rel:.2e})"
    )
    print(
        f"{system}/{scheme}: jvp ok; grad fd={fd:+.9e} ad={ad:+.9e} "
        f"(rel {rel:.1e})"
    )


def _check_dynamic_refused() -> None:
    """The default (dynamic) corrector must *not* differentiate.

    Without this the fixed-count rows would pass whether or not
    ``corrector_iterations`` did anything.
    """
    import importlib

    _configure("plane-couette", "iterative-cn", 0)

    import jax

    mod = importlib.import_module(FLOW_MODULES["plane-couette"])
    energy = _energy_after_steps(mod, "iterative-cn")
    s0 = _initial_state()
    try:
        jax.grad(energy)(s0)
    except Exception as exc:  # noqa: BLE001 - the expected outcome
        assert "while_loop" in str(exc), (
            "the dynamic corrector failed reverse mode for an "
            f"unexpected reason: {type(exc).__name__}: {exc}"
        )
        print(
            "dynamic corrector: jax.grad refused on lax.while_loop, as "
            "it must -- the fixed count is what unlocks it"
        )
        return
    raise AssertionError(
        "jax.grad succeeded with the dynamic corrector: either JAX "
        "gained a while_loop transpose rule (good news, retire the "
        "knob) or this test no longer reaches the corrector"
    )


def _check_fixed_matches_dynamic() -> None:
    """A long enough fixed count is the dynamic corrector's own answer.

    The count is read once when ``make_stepper`` runs, so the second
    stepper is rebuilt after changing it -- the ``test_cnab2.py``
    ``split_corrector`` idiom.

    Run at a **tight** ``corrector_tolerance`` (also read at trace
    time, so no operator rebuild): at the shipped 1e-5 and this ``dt``
    the first correction already converges, and the row would compare a
    fixed count against a corrector that never looped -- true, but
    vacuous.  The precondition is asserted rather than assumed.
    """
    import importlib

    _configure("plane-couette", "iterative-cn", 0, tol=1e-12)

    import jax.numpy as jnp

    from dnsjax.parameters import params

    mod = importlib.import_module(FLOW_MODULES["plane-couette"])
    geo_name, builder_name = STEPPER_BUILDERS["plane-couette"]
    builder = getattr(importlib.import_module(geo_name), builder_name)

    s0 = _initial_state()
    dyn, err_dyn, c_dyn, _ = mod.predict_and_fully_correct(jnp.copy(s0))

    params.step.corrector_iterations = int(c_dyn) + 4
    _, fixed_step, _, _, _, _, _ = builder(mod.flow)
    fix, _err, c_fix, _ = fixed_step(jnp.copy(s0))

    assert int(c_dyn) >= 1, (
        "the dynamic corrector converged without looping, so this row "
        "would not compare a fixed count against an iterated one"
    )
    scale = float(jnp.max(jnp.abs(dyn)))
    diff = float(jnp.max(jnp.abs(fix - dyn))) / scale
    assert int(c_fix) == int(c_dyn) + 3, (
        f"fixed stepper ran {int(c_fix)} extra corrections, expected "
        f"{int(c_dyn) + 3}"
    )
    assert diff < params.step.corrector_tolerance, (
        f"fixed count {int(c_dyn) + 4} vs dynamic corrector: states "
        f"differ by {diff:.3e} (rel), tolerance "
        f"{params.step.corrector_tolerance:.1e}"
    )
    print(
        f"fixed n={int(c_dyn) + 4} reproduces the dynamic corrector "
        f"(c={int(c_dyn)}, err={float(err_dyn):.2e}) to {diff:.2e} rel"
    )


def _check_kernel_path() -> None:
    """``solver.pallas_kernel`` decides the sweep; the test override wins."""
    _configure("plane-couette", "iterative-cn", 0)

    import jax

    import dnsjax.solvers as solvers_mod
    from dnsjax.parameters import params

    assert jax.default_backend() != "gpu", "this row assumes a CPU box"
    try:
        assert not solvers_mod._kernel_path(), "unset should follow the CPU"
        params.solver.pallas_kernel = True
        assert solvers_mod._kernel_path(), "true should force the kernel"
        params.solver.pallas_kernel = False
        assert not solvers_mod._kernel_path(), "false should forbid it"
        solvers_mod._force_kernel_path = True
        assert solvers_mod._kernel_path(), (
            "the trace-only override must still win over the parameter, "
            "or the CUDA-lowering guards cannot reach the kernel"
        )
    finally:
        solvers_mod._force_kernel_path = False
        params.solver.pallas_kernel = None
    print("_kernel_path: unset/true/false and the override all honoured")


WORKERS = {
    "dynamic-corrector-refused": _check_dynamic_refused,
    "fixed-matches-dynamic": _check_fixed_matches_dynamic,
    "kernel-path-selection": _check_kernel_path,
}


def main() -> None:
    names = [name for name, _, _ in GRAD_ROWS] + list(OTHER_ROWS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=names, default=None)
    parser.add_argument(
        "--only", default=None, help="Run only rows whose name contains this."
    )
    args = parser.parse_args()

    if args.worker:
        for name, system, scheme in GRAD_ROWS:
            if name == args.worker:
                _check_grad(system, scheme)
                return
        WORKERS[args.worker]()
        return

    print(
        "Autodiff guards: offline, 1 forced CPU device per row "
        "(the banded solve's own adjoint lives in "
        "tests/test_banded_solver.py).",
        flush=True,
    )
    selected = [n for n in names if not args.only or args.only in n]
    if not selected:
        raise SystemExit(f"--only {args.only!r} matched no row")
    passed, failures = 0, []
    for name in selected:
        print(f"=== {name} ===", flush=True)
        result = run_live([sys.executable, __file__, "--worker", name])
        if result.returncode == 0:
            passed += 1
        else:
            failures.append((name, "worker failed"))
    sys.exit(report(passed, failures))


if __name__ == "__main__":
    main()
