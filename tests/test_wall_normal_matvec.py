r"""Step parity across ``solver.wall_normal_matvec``: GEMM vs stencil.

Every wall-normal FD matrix is a ``YMatrix``
(``geometries/wall_bounded/_base``), applied as a dense GEMM or as a
stencil per ``solver.wall_normal_matvec``.  The unit tests pin the
operator equality (``test_cartesian.py``, ``test_cylindrical.py``,
``test_annular.py``); this script pins the integration: the as-run
predictor-corrector step, compiled once per knob value in one process
(``jax.clear_caches()`` between), from the same random state, must land
on the same state to machine precision with the same corrector count.
A call site still holding a raw matrix, a ghost operand not sliced to
its trimmed corner, or a stencil on the wrong axis would each show as
an O(1) difference.

One worker per case, covering every geometry's derivative call sites:
plane Couette (Cartesian), the pipe, the curved pipe (its continuity
rows rebuild a parity-reduced ``D1`` from ``D1_pos`` and the trimmed
ghost), Taylor-Couette (annular) and viscoelastic Dean (the
component-leading 9-, 6- and 3-field tensor stacks); then the call
sites those five cannot reach: the legacy primitive pass of each
geometry (``res.consistent_imm = False``: the ``_*_primitive_imm``
modules' own ``D1``/``D2``/``A_base`` stacks) and the viscoelastic pipe
(its parity-reduced tensor stacks).  The two paths sum the same
products in a different order, and the solves amplify that by their
conditioning, so the bound is a few orders above round-off and many
below any real defect (measured: ``<= 1.2e-13`` here, ``1.5e-12`` for
the pipe at ``32 x 97 x 64``).

Each case needs its own process: the parameter singletons and the
jitted steppers capture ``params`` at import / trace time.

Run as a script::

    uv run python tests/test_wall_normal_matvec.py
"""

from __future__ import annotations

import argparse
import os
import sys

from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

BOUND = 1e-11

#: Per worker: the system (the case name unless given) and its
#: ``phys`` / ``geo`` / ``init`` / ``res`` overrides.
CASES: dict[str, dict[str, object]] = {
    "plane-couette": {"phys": {"re": 400.0}, "geo": {"lx": 6.0, "lz": 3.0}},
    "pipe": {"phys": {"re": 1800.0}, "geo": {"lx": 5.0}},
    "curved-pipe": {
        "phys": {"re": 1800.0},
        "geo": {"lx": 5.0, "curvature": 0.3},
    },
    "taylor-couette": {
        "phys": {"re1": 100.0, "re2": 0.0},
        "geo": {"eta": 0.5},
    },
    "viscoelastic-dean": {
        "phys": {"wi": 20.0, "el": 20.0},
        "geo": {"lx": 5.0},
        "init": {"random_conformation_amplitude": 10.0},
    },
}

_LEGACY = {"consistent_imm": False}

# The call sites the five above do not reach.
CASES |= {
    "plane-couette-legacy": {
        **CASES["plane-couette"],
        "system": "plane-couette",
        "res": _LEGACY,
    },
    "pipe-legacy": {**CASES["pipe"], "system": "pipe", "res": _LEGACY},
    "taylor-couette-legacy": {
        **CASES["taylor-couette"],
        "system": "taylor-couette",
        "res": _LEGACY,
    },
    "viscoelastic-pipe": {
        "phys": {"wi": 20.0, "el": 0.02},
        "geo": {"lx": 5.0},
        "init": {"random_conformation_amplitude": 10.0},
    },
}


def _worker(name: str) -> None:
    """Compile the as-run step under both knob values and compare."""
    os.environ.setdefault("NPROC", "1")
    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    case = CASES[name]
    system = case.get("system", name)
    update_parameters(
        Parameters(
            phys={"system": system, **case["phys"]},
            geo=case["geo"],
            init=case.get("init", {}),
            res={"nx": 8, "ny": 33, "nz": 16, **case.get("res", {})},
            step={"dt": 0.002},
        )
    )
    padded_res.set_padded_resolution(params)
    validate_parameters()
    configure_jax_platform(platform_from_argv())

    import importlib

    import jax
    import jax.numpy as jnp
    import numpy as np

    from dnsjax.flows.registry import spec_for
    from dnsjax.geometries.wall_bounded._base import YMatrix
    from dnsjax.ic.random_field import generate_random_state

    mod = importlib.import_module(spec_for(system).flow_module)
    state = generate_random_state(
        params.init.random_amplitude,
        params.init.random_smoothness,
        params.init.random_wall_smoothness,
        params.init.random_wall_confinement,
        7,
    )
    state = getattr(mod, "to_solver_basis", lambda s: s)(state)

    flow = mod.flow
    held = [
        getattr(flow, name)
        for name in ("D1", "D2", "A_base", "D1_pos", "A_base_pos")
        if hasattr(flow, name)
    ]
    assert held and all(isinstance(m, YMatrix) for m in held), system

    # The as-run program: the jitted inner step with the singletons as
    # arguments, fished from the bound wrapper's closure.
    fn = mod.predict_and_fully_correct
    cells = dict(
        zip(
            fn.__code__.co_freevars,
            (c.cell_contents for c in fn.__closure__),
            strict=True,
        )
    )
    step = cells["_predict_and_fully_correct_jit"]
    fourier = cells["fourier"]

    out = {}
    for knob in ("dense", "banded"):
        params.solver.wall_normal_matvec = knob
        jax.clear_caches()
        compiled = step.lower(state, fourier, flow).compile()
        new, _, num_c, _ = compiled(jnp.copy(state), fourier, flow)
        out[knob] = (np.asarray(new), int(num_c))
    params.solver.wall_normal_matvec = "auto"

    (s_d, c_d), (s_b, c_b) = out["dense"], out["banded"]
    assert c_d == c_b, f"corrector counts differ: {c_d} vs {c_b}"
    rel = float(np.abs(s_d - s_b).max() / np.abs(s_d).max())
    print(f"RESULT {rel:.6e}", flush=True)


# ── orchestrator ─────────────────────────────────────────────────────


def main(only: list[str] | None) -> None:
    print(
        "GEMM vs stencil step parity (solver.wall_normal_matvec), one "
        f"subprocess per flow, bound {BOUND:.0e} relative (CPU).",
        flush=True,
    )
    passed, failures = 0, []
    for case in only or CASES:
        print(f"\n--- {case} ---", flush=True)
        proc = run_live(
            [sys.executable, os.path.abspath(__file__), "--worker", case],
            timeout=1200,
        )
        lines = [
            ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")
        ]
        if proc.returncode != 0 or not lines:
            err = (proc.stderr or proc.stdout).strip()[-400:]
            print(f"FAIL {case}: {err}", flush=True)
            failures.append((case, err))
            continue
        rel = float(lines[-1].split()[1])
        if rel > BOUND:
            reason = f"states differ by {rel:.2e} > {BOUND:.0e}"
            print(f"FAIL {case}: {reason}", flush=True)
            failures.append((case, reason))
            continue
        print(f"PASS {case}: states agree to {rel:.2e}", flush=True)
        passed += 1
    sys.exit(report(passed, failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", metavar="CASE")
    parser.add_argument(
        "--only", nargs="+", choices=list(CASES), help="a subset of cases"
    )
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        main(args.only)
