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

One worker per flow, covering every geometry's derivative call sites:
plane Couette (Cartesian), the pipe, the curved pipe (its continuity
rows rebuild a parity-reduced ``D1`` from ``D1_pos`` and the trimmed
ghost), Taylor-Couette (annular) and viscoelastic Dean (the
component-leading 9-, 6- and 3-field tensor stacks).  The two paths
sum the same products in a different order, and the solves amplify
that by their conditioning, so the bound is a few orders above
round-off and many below any real defect (measured: ``<= 1e-13``
here, ``1.5e-12`` for the pipe at ``32 x 97 x 64``).

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

#: ``(system, physics, geometry, initiation)`` per worker.
CASES: dict[str, tuple[dict, dict, dict]] = {
    "plane-couette": ({"re": 400.0}, {"lx": 6.0, "lz": 3.0}, {}),
    "pipe": ({"re": 1800.0}, {"lx": 5.0}, {}),
    "curved-pipe": ({"re": 1800.0}, {"lx": 5.0, "curvature": 0.3}, {}),
    "taylor-couette": ({"re1": 100.0, "re2": 0.0}, {"eta": 0.5}, {}),
    "viscoelastic-dean": (
        {"wi": 20.0, "el": 20.0},
        {"lx": 5.0},
        {"random_conformation_amplitude": 10.0},
    ),
}


def _worker(system: str) -> None:
    """Compile the as-run step under both knob values and compare."""
    os.environ.setdefault("NPROC", "1")
    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    phys, geo, init = CASES[system]
    update_parameters(
        Parameters(
            phys={"system": system, **phys},
            geo=geo,
            init=init,
            res={"nx": 8, "ny": 33, "nz": 16},
            step={"dt": 0.002},
        )
    )
    padded_res.set_padded_resolution(params)
    validate_parameters()
    configure_jax_platform("cpu")

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
    for system in only or CASES:
        print(f"\n--- {system} ---", flush=True)
        proc = run_live(
            [sys.executable, os.path.abspath(__file__), "--worker", system],
            timeout=1200,
        )
        lines = [
            ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")
        ]
        if proc.returncode != 0 or not lines:
            err = (proc.stderr or proc.stdout).strip()[-400:]
            print(f"FAIL {system}: {err}", flush=True)
            failures.append((system, err))
            continue
        rel = float(lines[-1].split()[1])
        if rel > BOUND:
            reason = f"states differ by {rel:.2e} > {BOUND:.0e}"
            print(f"FAIL {system}: {reason}", flush=True)
            failures.append((system, reason))
            continue
        print(f"PASS {system}: states agree to {rel:.2e}", flush=True)
        passed += 1
    sys.exit(report(passed, failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", metavar="SYSTEM")
    parser.add_argument(
        "--only", nargs="+", choices=list(CASES), help="a subset of flows"
    )
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
    else:
        main(args.only)
