"""Discrete continuity of a stepped state, with and without
``res.consistent_imm``.

The influence-matrix method's continuity argument (Kleiser-Schumann;
Canuto, Hussaini, Quarteroni & Zang 1988, sec. 7.3) is derived for
*continuous* differentiation operators.  Discretely it needs
`$\\nabla\\cdot\\nabla = L_k$` (i.e. `$D_1 D_1 = D_2$`) and
`$[D_1, D_2] = 0$`, and -- separately -- an accounting of the momentum
wall rows the Dirichlet replacement discards.  With none of that,
a stepped state's discrete divergence is **O(1) relative**: not a bug,
but not zero either.  ``res.consistent_imm`` supplies both halves at
once (see the ``Resolution`` docstring).

Each case measures ``max|div| / max|individual term|`` on the true-mode
slice of a state that has been through one full predictor-corrector
step, using the solver's *own* divergence assembly (the one
``dnsjax.analysis.snapshot_ops.divergence`` mirrors).  The measurement
must use a **stepped** state: a divergence-free field cannot
discriminate two divergence operators -- both return zero -- so an
un-stepped check is unfalsifiable.  The real-FFT `$k = 0$` plane is
excluded, since ``random_field`` solves continuity only off it by
design.

What is asserted, and why the bounds differ per geometry:

- **Cartesian** (plane-couette): the closure is *exact*, so the
  stepped-state divergence must collapse to round-off.
- **Annular** (taylor-couette): bounded by the discrete commutator
  `$[D_1, 1/r] \\ne -1/r^2$`, which no choice of a single `$D_1$`
  removes.  A large improvement is required, not exactness.
- **Pipe**: opts in via the `$x = r^2$` parity operators (which make the
  near-axis `$1/r$` commutator exact for one parity) plus a 1-wall
  `$\\hat\\sigma$` closure.  A large improvement is required, not
  exactness: the structural invariant
  `$\\mathrm{diag}(\\Theta) + \\mathrm{diag}(\\Phi) = 2/r^2$` forbids
  both radial parities' commutators vanishing at once, so a stepped
  state keeps the other parity's residual (see the
  ``Resolution.consistent_imm`` docs).  Uses the deterministic
  axis-regular rolls IC (a grid-white random draw would swamp it with
  under-resolved-noise divergence).

Each case needs its own process: the parameter singletons and the
jitted steppers capture ``params`` at import / trace time.

Run as a script::

    uv run python tests/test_imm_continuity.py
"""

from __future__ import annotations

import argparse
import os
import sys

from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

NY = 25
FD_ORDER = 8
# The pipe's rolls IC starts divergence-free; step a short horizon so
# the accumulated projection-splitting divergence is a clear signal.
PIPE_STEPS = 10

# ``--ny`` overrides NY for both the driver and its workers, so the
# resolution dependence can be swept.  ``ny = 25`` is the suite
# default because it is fast, and is the **only** resolution the
# bounds are pinned at; other ``--ny`` values report without
# asserting (``main``).  Measured, seed 7:
#
#     ny    plane-couette off / on     taylor-couette off / on
#     25    4.48e-02 / 4.20e-14        5.94e-02 / 7.14e-06
#     97    1.11e-03 / 1.32e-12        5.43e-04 / 1.48e-08
#
# The ungated residual **falls** with resolution here (~30x over
# 25 -> 97, seed-robust over seeds 7/11/23), and the gated one stays
# far below it -- the closure solve is not conditioning-limited at
# realistic sizes.
#
# These numbers replace a set that was 2-3 orders larger and grew as
# `$N_y^2$` (2.08e-01 -> 9.86e-01 for plane-couette off).  That growth
# was the *initial condition*, not the scheme: ``_column_draw`` was
# grid-white in the wall-normal direction, and the boundary term's
# `$D_1[1,0] \sim N_y^2$` amplified its Nyquist content.  With
# ``random_field._wall_normal_filter`` supplying the missing
# wall-normal factor of the smoothness envelope, that content is gone.
# **The rationale recorded for ``res.consistent_imm`` elsewhere
# (``Resolution.consistent_imm``, ``wall_bounded/CLAUDE.md``, the plan
# file) still quotes the old numbers and needs re-baselining** -- the
# flag still earns its keep, but its "before" was inflated by the IC.

# (label, system, consistent_imm, max relative divergence allowed).
# The gate-off bounds are loose regression pins on today's behaviour
# (measured ~4e-2 / ~6e-2 at NY); the gate-on bounds are the claim.
CASES = [
    ("plane-couette  off", "plane-couette", False, 1e0),
    ("plane-couette  on", "plane-couette", True, 1e-11),
    ("taylor-couette off", "taylor-couette", False, 1e0),
    ("taylor-couette on", "taylor-couette", True, 1e-3),
    # The pipe opts in via the x = r^2 axis operators + 1-wall closure
    # (measured ~2.8e-2 / ~5.6e-5 at NY, ``PIPE_STEPS`` rolls steps;
    # ~500x).  The gate cannot reach the Cartesian machine-zero because
    # the structural invariant forbids both radial parities' 1/r
    # commutators vanishing at once -- see the
    # ``Resolution.consistent_imm`` docs / plan.
    ("pipe           off", "pipe", False, 1e0),
    ("pipe           on", "pipe", True, 1e-3),
]

# Gate-off floors: the flag must *demonstrably* change something, so
# each off case also asserts the residual is at least this large.
# Without it the "on" bounds could pass for the wrong reason.
OFF_FLOOR = 1e-3


# ── worker (runs in its own process) ─────────────────────────────────


def _worker(system: str, consistent_imm: bool, ny: int) -> None:
    import numpy as np

    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        Geometry,
        Initiation,
        Parameters,
        Physics,
        Resolution,
        TimeStepping,
        params,
        update_parameters,
        validate_parameters,
    )

    configure_jax_platform("cpu")

    phys: dict = {"system": system}
    geo: dict = {}
    if system == "taylor-couette":
        phys |= {"re1": 200.0, "re2": -100.0}
        geo |= {"eta": 0.5}
    elif system == "pipe":
        # A higher Re accumulates a clear projection-splitting
        # divergence from the (otherwise well-behaved) axis-regular
        # rolls IC over ``PIPE_STEPS``.
        phys |= {"re": 3000.0}
    else:
        phys |= {"re": 500.0}
    update_parameters(
        Parameters(
            phys=Physics(**phys),
            geo=Geometry(**geo),
            res=Resolution(
                nx=12,
                ny=ny,
                nz=12,
                fd_order=FD_ORDER,
                consistent_imm=consistent_imm,
            ),
            init=(
                # The pipe uses the deterministic axis-regular rolls IC
                # (a grid-white random draw has no continuum limit near
                # the axis, so its divergence is under-resolved noise and
                # the gate shows no clean gain -- see the plan file).
                Initiation(
                    localized_rolls=True, localized_rolls_amplitude=0.15
                )
                if system == "pipe"
                else Initiation(random_amplitude=0.2, random_seed=7)
            ),
            step=TimeStepping(dt=0.01),
        )
    )
    validate_parameters()

    import jax.numpy as jnp

    from dnsjax.flows.registry import spec_for

    mod = __import__(spec_for(system).flow_module, fromlist=["x"])
    flow, fourier = mod.flow, mod.fourier

    # Cartesian has no basis crossing; cylindrical/annular carry the
    # decoupled u_+/u_- solver basis and must be converted first.
    to_solver = getattr(mod, "to_solver_basis", lambda s: s)
    from_solver = getattr(mod, "from_solver_basis", lambda s: s)

    if system == "pipe":
        from dnsjax.localized_rolls import generate_localized_rolls

        state = generate_localized_rolls(0.15, 0.5, 1.0)
        e0 = float(mod.get_perturbation_energy(state))
        state = state * float(np.sqrt(1e-2 / e0))
    else:
        from dnsjax.random_field import generate_random_state

        state = generate_random_state(0.2, 0.4, 7)
    # A random IC produces an O(1) divergence in a single step; the
    # axis-regular rolls IC (pipe) starts divergence-free and its
    # projection-splitting divergence accumulates over a few steps, so
    # the pipe is stepped a short horizon to expose a clear signal.
    n_steps = PIPE_STEPS if system == "pipe" else 1
    stepped = jnp.copy(to_solver(state))
    for _ in range(n_steps):
        stepped, _, _ = mod.predict_and_fully_correct(stepped)
    stepped = from_solver(stepped)

    def divergence(st) -> float:
        """Relative divergence, assembled as ``_imm_iteration`` does."""
        nz, nx = params.res.nz, params.res.nx
        s = np.asarray(st)[:, :, : nz - 1, : nx // 2]
        if system == "plane-couette":
            D1 = np.asarray(flow.D1)
            dy = np.einsum("ij, jzx -> izx", D1, s[1])
            kx = np.asarray(fourier.kx)[..., : nx // 2]
            kz = np.asarray(fourier.kz)[:, : nz - 1]
            terms = [1j * kx * s[0], dy, 1j * kz * s[2]]
        elif system == "pipe":
            # The pipe's divergence is assembled in the solver's own
            # u_+/u_- basis (``_imm_iteration``'s ``div_n``), not the
            # physical triad: near the axis the physical ``u_r/r`` and
            # ``i m u_theta/r`` terms are individually huge but cancel,
            # which would inflate ``scale`` and deflate the ratio.  The
            # parity-reduced radial D1 (parity (-1)^{m+1}) is the x = r^2
            # operator under ``res.consistent_imm``.
            D1p = np.asarray(flow.D1_pos)
            D1g = np.asarray(flow.D1_ghost)
            gg = D1g.shape[0]
            psv = -(np.asarray(fourier.m_is_even) * 2 - 1)[0, : nz - 1]
            inv_r = np.asarray(flow.inv_r)[:, None, None]
            m = np.asarray(fourier.m)[:, : nz - 1]
            kz = np.asarray(fourier.kz)[..., : nx // 2]
            up = s[1] + 1j * s[2]  # u_r + i u_theta
            um = s[1] - 1j * s[2]  # u_r - i u_theta

            def _dy_v(u):
                o = np.einsum("ij, jzx -> izx", D1p, u)
                o[:gg] += psv * np.einsum("ij, jzx -> izx", D1g, u)
                return o

            terms = [
                (_dy_v(up) + (m + 1) * inv_r * up) / 2,
                (_dy_v(um) + (1 - m) * inv_r * um) / 2,
                1j * kz * s[0],
            ]
        else:  # annular (taylor-couette)
            D1 = np.asarray(flow.D1)
            dy = np.einsum("ij, jzx -> izx", D1, s[1])
            inv_r = np.asarray(flow.inv_r)[:, None, None]
            kz = np.asarray(fourier.kz)[..., : nx // 2]
            m = np.asarray(fourier.m)[:, : nz - 1]
            terms = [dy, s[1] * inv_r, 1j * m * s[2] * inv_r, 1j * kz * s[0]]
        # Drop the real-FFT k = 0 plane (see the module docstring).
        d = np.abs(sum(terms)[..., 1:]).max()
        scale = max(np.abs(t[..., 1:]).max() for t in terms)
        return float(d / scale)

    rel_ic = divergence(state)
    rel = divergence(stepped)
    print(f"IC rel div = {rel_ic:.2e}", flush=True)
    print(f"RESULT {rel:.6e}", flush=True)
    assert rel_ic < 1e-12, f"IC is not divergence-free: {rel_ic:.2e}"


def _worker_pipe_accepts() -> None:
    """``res.consistent_imm`` must be settable on the pipe (the opt-in
    landed; the flag is no longer deferred)."""
    from dnsjax.parameters import (
        Parameters,
        Physics,
        Resolution,
        update_parameters,
        validate_parameters,
    )

    update_parameters(
        Parameters(
            phys=Physics(system="pipe", re=3000.0),
            res=Resolution(consistent_imm=True),
        )
    )
    validate_parameters()  # must not raise
    print("RESULT accepted", flush=True)


# ── orchestrator ─────────────────────────────────────────────────────


def _run(label: str, args: list[str]) -> tuple[bool, str, float]:
    proc = run_live(
        [sys.executable, os.path.abspath(__file__), "--worker", *args],
        timeout=1200,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[-400:], 0.0
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    if not line:
        return False, "no RESULT line", 0.0
    value = line[-1].split()[1]
    non_numeric = value in ("rejected", "accepted")
    return True, "", (0.0 if non_numeric else float(value))


def main(ny: int) -> None:
    print(
        f"IMM discrete-continuity tests: ny={ny}, fd_order={FD_ORDER}, "
        "one full step from a random divergence-free IC, one "
        "subprocess per case (CPU).",
        flush=True,
    )
    passed, failures = 0, []
    # The bounds below are pinned at ``NY`` only; ``--ny`` is a
    # *diagnostic* sweep (see the module docstring), reported and not
    # asserted, because both bounds are absolute numbers that move with
    # resolution -- and, since the wall-normal smoothness envelope
    # landed, move a long way.
    asserted = ny == NY
    if not asserted:
        print(
            f"  (ny={ny} != {NY}: values are reported, not asserted)",
            flush=True,
        )

    for label, system, cimm, bound in CASES:
        print(f"\n--- {label} ---", flush=True)
        ok, err, rel = _run(
            label,
            ["--system", system, "--ny", str(ny)]
            + (["--consistent-imm"] if cimm else []),
        )
        if not ok:
            print(f"FAIL {label}: {err}", flush=True)
            failures.append((label, err))
            continue
        if not asserted:
            print(f"REPORT {label}: rel div {rel:.2e}")
            passed += 1
            continue
        if rel > bound:
            reason = f"rel div {rel:.2e} > {bound:.0e}"
        elif not cimm and rel < OFF_FLOOR:
            reason = (
                f"rel div {rel:.2e} < {OFF_FLOOR:.0e} with the flag off "
                "-- the 'on' bound would pass for the wrong reason"
            )
        else:
            print(f"PASS {label}: rel div {rel:.2e} <= {bound:.0e}")
            passed += 1
            continue
        print(f"FAIL {label}: {reason}", flush=True)
        failures.append((label, reason))

    print("\n--- pipe accepts the flag ---", flush=True)
    ok, err, _ = _run("pipe accepts", ["--pipe-accepts"])
    if ok:
        print("PASS pipe accepts res.consistent_imm")
        passed += 1
    else:
        print(f"FAIL pipe accepts: {err}", flush=True)
        failures.append(("pipe accepts", err))

    sys.exit(report(passed, failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--system")
    parser.add_argument("--consistent-imm", action="store_true")
    parser.add_argument("--pipe-accepts", action="store_true")
    parser.add_argument("--ny", type=int, default=NY)
    args = parser.parse_args()
    if args.pipe_accepts:
        _worker_pipe_accepts()
    elif args.worker:
        _worker(args.system, args.consistent_imm, args.ny)
    else:
        main(args.ny)
