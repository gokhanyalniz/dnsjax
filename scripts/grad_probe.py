#!/usr/bin/env python3
r"""Probe which dnsjax stepping configurations differentiate.

Answers one question per configuration: does a time step admit forward
mode (:func:`jax.jvp`), and does it admit reverse mode
(:func:`jax.grad`)?  The quantity differentiated is the perturbation
kinetic energy after a few steps, with respect to the initial spectral
state -- the shape every adjoint-based use starts from (optimal
perturbations, sensitivity, control).  Each reverse-mode result is
cross-checked against a central difference along a random direction,
because "``jax.grad`` returned an array" is not the same claim as
"``jax.grad`` returned the derivative".

**Reading the table.**  ``jax.grad`` fails on any configuration whose
corrector iterates to a tolerance: a dynamic trip count
(``lax.while_loop``) has no reverse rule.  ``step.corrector_iterations
= n`` fixes the count, lowers to a scan, and differentiates -- the
opt-in the rest of the table is about.  The exception is
triply-periodic ``cnab2``, whose explicit-AB2 step runs no corrector at
all and so differentiates unconditionally.  See
``docs/differentiability.md``.

**One subprocess per configuration.**  The parameter singletons and the
jitted steppers capture their configuration at import and trace time,
so a sweep cannot run in one process; the script re-executes itself
with ``--child`` per row and parses one ``@@RESULT`` line back, the
same idiom as ``scripts/solver_benchmark.py``.  It imports nothing from
``tests/``.

Usage::

    uv run python scripts/grad_probe.py                  # 12 rows, CPU
    uv run python scripts/grad_probe.py --full           # cross product
    uv run python scripts/grad_probe.py --dist.platform cuda

The default is CPU in double precision.  ``--dist.platform cuda`` is
the run this box cannot do: it exercises the real Triton lowering of
the transposed banded sweep that backs the kernel's ``custom_vjp``
(``solvers._pallas_banded_solve_t``), which interpret mode and a CUDA
lowering check between them do not cover -- the same footing the
forward kernel shipped on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python3"
SELF = Path(__file__).resolve()
RESULT_TAG = "@@RESULT "

#: ``(system, scheme, corrector_iterations, backend, pallas_kernel)``.
#: ``backend``/``pallas_kernel`` are ``None`` where the flow's surface
#: does not carry them (the triply-periodic family has no ``[solver]``
#: fields: its implicit step is diagonal in spectral space).
_Row = tuple[str, str, int, str | None, bool | None]

#: The curated default: every axis covered at least once, 12 rows.
ROWS: tuple[_Row, ...] = (
    ("kolmogorov", "iterative-cn", 0, None, None),
    ("kolmogorov", "iterative-cn", 3, None, None),
    ("kolmogorov", "cnab2", 0, None, None),
    ("kolmogorov", "cnab2", 3, None, None),
    ("plane-couette", "iterative-cn", 0, "pallas", None),
    ("plane-couette", "iterative-cn", 3, "pallas", None),
    ("plane-couette", "cnab2", 0, "pallas", None),
    ("plane-couette", "cnab2", 3, "pallas", None),
    ("plane-couette", "iterative-cn", 3, "dense", None),
    ("plane-couette", "iterative-cn", 3, "pallas", False),
    ("pipe", "iterative-cn", 3, "pallas", None),
    ("pipe", "cnab2", 3, "pallas", None),
)

SYSTEMS = ("kolmogorov", "plane-couette", "pipe")
FLOW_MODULES = {
    "kolmogorov": "dnsjax.flows.triply_periodic.monochromatic",
    "plane-couette": "dnsjax.flows.wall_bounded.plane_couette",
    "pipe": "dnsjax.flows.wall_bounded.pipe",
}
#: Small enough that a row costs seconds, large enough that the
#: nonlinear term, the influence-matrix pass and the banded solve all
#: run on more than a trivial mode set.
NX, NY, NZ = 8, 17, 8
NY_PERIODIC = 16
N_STEPS = 2
SEED = 1


def full_rows(platform: str) -> tuple[_Row, ...]:
    """The cross product behind ``--full``."""
    kernels: tuple[bool | None, ...] = (None, False)
    if platform in ("cuda", "rocm"):
        kernels += (True,)
    rows: list[_Row] = []
    for system in SYSTEMS:
        for scheme in ("iterative-cn", "cnab2"):
            for n in (0, 3):
                if system == "kolmogorov":
                    rows.append((system, scheme, n, None, None))
                    continue
                for backend in ("pallas", "dense"):
                    for kern in kernels if backend == "pallas" else (None,):
                        rows.append((system, scheme, n, backend, kern))
    return tuple(rows)


def label(row: _Row) -> str:
    """One-line human name for a configuration."""
    system, scheme, n, backend, kern = row
    bits = [system, scheme, "dynamic" if n == 0 else f"fixed n={n}"]
    if backend is not None:
        bits.append(backend)
    if kern is not None:
        bits.append(f"pallas_kernel={str(kern).lower()}")
    return " / ".join(bits)


# ── the child: one configuration, measured ───────────────────────


def run_child(a: argparse.Namespace) -> None:
    """Measure one configuration and print its ``@@RESULT`` line."""
    from dnsjax.bootstrap import configure_jax_platform

    configure_jax_platform(a.platform)

    import importlib

    import jax
    import jax.numpy as jnp

    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    periodic = a.system == "kolmogorov"
    solver: dict = {}
    if a.backend:
        solver["backend"] = a.backend
    if a.pallas_kernel != "unset":
        solver["pallas_kernel"] = a.pallas_kernel == "true"
    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": a.platform},
            phys={"system": a.system, "re": 100.0},
            geo={"lx": 5.0} if a.system == "pipe" else {"lx": 5.0, "lz": 5.0},
            res={
                "nx": NX,
                "ny": NY_PERIODIC if periodic else NY,
                "nz": NZ,
                "fd_order": 4,
                "double_precision": True,
            },
            step={
                "scheme": a.scheme,
                "dt": 0.005,
                "corrector_iterations": a.corrector_iterations,
            },
            solver=solver,
        )
    )
    padded_res.set_padded_resolution(params)
    validate_parameters()

    mod = importlib.import_module(FLOW_MODULES[a.system])
    from dnsjax.ic.random_field import generate_random_state

    s0 = generate_random_state(0.1, 0.4, 0.4, 0.14, SEED)

    def energy(state):
        """Perturbation energy after ``N_STEPS`` steps from *state*.

        The steppers donate their inputs, so a caller that keeps its
        own state hands them a copy.
        """
        s = jnp.copy(state)
        if a.scheme == "cnab2":
            carry = jnp.zeros_like(s)
            _, carry, *_ = mod.step_cnab2(jnp.copy(s), carry)
            for _ in range(N_STEPS):
                s, carry, *_ = mod.step_cnab2(s, carry)
        else:
            for _ in range(N_STEPS):
                s, *_ = mod.predict_and_fully_correct(s)
        return mod.get_perturbation_energy(s)

    rec: dict = {"jvp": None, "grad": None, "fd": None, "ad": None}
    try:
        _, tangent = jax.jvp(energy, (s0,), (jnp.ones_like(s0) * 1e-3,))
        rec["jvp"] = "ok"
        rec["jvp_value"] = float(tangent)
    except Exception as exc:  # noqa: BLE001 - reported, not handled
        rec["jvp"] = f"{type(exc).__name__}: {_first_line(exc)}"

    try:
        g = jax.grad(energy)(s0)
        direction = jax.random.normal(
            jax.random.key(0), s0.shape, dtype=jnp.float64
        ).astype(s0.dtype)
        eps = 1e-6
        fd = float(
            energy(s0 + eps * direction) - energy(s0 - eps * direction)
        ) / (2 * eps)
        # jax.grad of a real function of a complex input returns the
        # conjugate cotangent, so the directional derivative is
        # Re<conj(g), d>.
        ad = float(jnp.real(jnp.sum(jnp.conj(g) * direction)))
        rec["grad"] = "ok"
        rec["fd"], rec["ad"] = fd, ad
        rec["rel"] = abs(fd - ad) / max(abs(fd), 1e-300)
    except Exception as exc:  # noqa: BLE001 - reported, not handled
        rec["grad"] = f"{type(exc).__name__}: {_first_line(exc)}"

    print(RESULT_TAG + json.dumps(rec))


def _first_line(exc: BaseException) -> str:
    """The first line of an exception, clipped."""
    text = str(exc).strip().splitlines()
    return (text[0] if text else "")[:200]


def _reason(message: str) -> str:
    """A table-width reason from a failure message.

    Reverse mode's refusal names the primitive it could not transpose,
    which is the whole content of the cell; the rest of JAX's message
    is a paragraph of advice that would wreck the table.
    """
    if "while_loop" in message:
        return "refused: `lax.while_loop`"
    return message[:60]


# ── the parent: spawn, collect, tabulate ─────────────────────────


def spawn(row: _Row, platform: str, timeout: int) -> dict:
    """Run one configuration in its own process; parse its result."""
    system, scheme, n, backend, kern = row
    cmd = [
        str(PY), str(SELF), "--child",
        "--system", system,
        "--scheme", scheme,
        "--corrector-iterations", str(n),
        "--platform", platform,
        "--pallas-kernel",
        "unset" if kern is None else str(kern).lower(),
    ]  # fmt: skip
    if backend:
        cmd += ["--backend", backend]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO
        )
    except subprocess.TimeoutExpired:
        return {"jvp": "timeout", "grad": "timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG) :])
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    return {
        "jvp": "crash",
        "grad": "crash",
        "note": tail[-1][:90] if tail else f"exit {proc.returncode}",
    }


def render(rows: tuple[_Row, ...], results: list[dict]) -> str:
    """The Markdown table."""
    head = "| configuration | `jax.jvp` | `jax.grad` | vs central difference |"
    out = [head, "|---|---|---|---|"]
    for row, rec in zip(rows, results, strict=True):
        jvp = (
            "ok"
            if rec.get("jvp") == "ok"
            else f"**{_reason(str(rec.get('jvp')))}**"
        )
        grad = (
            "ok"
            if rec.get("grad") == "ok"
            else f"**{_reason(str(rec.get('grad')))}**"
        )
        if rec.get("grad") == "ok":
            check = (
                f"fd={rec['fd']:+.6e} ad={rec['ad']:+.6e} "
                f"(rel {rec['rel']:.1e})"
            )
        else:
            check = "—"
        out.append(f"| {label(row)} | {jvp} | {grad} | {check} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--system", choices=SYSTEMS, help=argparse.SUPPRESS)
    ap.add_argument("--scheme", help=argparse.SUPPRESS)
    ap.add_argument(
        "--corrector-iterations", type=int, default=0, help=argparse.SUPPRESS
    )
    ap.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pallas-kernel", default="unset", help=argparse.SUPPRESS)
    ap.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend the children run on (default cpu).",
    )
    ap.add_argument("--platform", dest="platform", help=argparse.SUPPRESS)
    ap.add_argument(
        "--full",
        action="store_true",
        help="Sweep the whole cross product instead of the 12 curated rows.",
    )
    ap.add_argument(
        "--timeout", type=int, default=900, help="Per-child timeout (s)."
    )
    a = ap.parse_args()

    if a.child:
        if a.system is None:
            ap.error("--child requires --system")
        run_child(a)
        return

    rows = full_rows(a.platform) if a.full else ROWS
    print(f"grad_probe: {len(rows)} configurations on {a.platform}\n")
    results = []
    for i, row in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] {label(row)}", flush=True)
        results.append(spawn(row, a.platform, a.timeout))

    print("\n" + render(rows, results) + "\n")
    # A dynamic corrector refusing reverse mode is the documented
    # behaviour, not a probe failure; anything else is.
    unexpected = [
        (label(r), rec)
        for r, rec in zip(rows, results, strict=True)
        if rec.get("grad") != "ok"
        and "while_loop" not in str(rec.get("grad", ""))
    ]
    if unexpected:
        print(
            "Reverse mode failed for a reason other than a dynamic corrector:"
        )
        for name, rec in unexpected:
            print(f"  {name}: {rec.get('grad')}")
    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
