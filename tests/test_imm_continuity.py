"""Discrete continuity of a stepped state, with and without
``res.consistent_imm``.

The influence-matrix method's continuity argument (Kleiser-Schumann;
Canuto, Hussaini, Quarteroni & Zang 1988, sec. 7.3) is derived for
*continuous* differentiation operators.  Discretely it needs
`$\\nabla\\cdot\\nabla = L_k$` (i.e. `$D_1 D_1 = D_2$`) and
`$[D_1, D_2] = 0$`, and -- separately -- an accounting of the momentum
wall rows the Dirichlet replacement discards.  With none of that,
a stepped state's discrete divergence is **O(1) relative**: not a bug,
but not zero either.  ``res.consistent_imm`` fixes it in all three
geometries by *reformulating* -- advancing the wall-normal velocity
and vorticity and reconstructing the tangential components, so
continuity is algebra rather than something a solve has to deliver
(see the ``Resolution`` docstring).

Each case measures ``max|div| / max|individual term|`` on the true-mode
slice of a state that has been through one full predictor-corrector
step, using the solver's *own* divergence assembly (the one
``dnsjax.analysis.snapshot_ops.divergence`` mirrors).  The measurement
must use a **stepped** state: a divergence-free field cannot
discriminate two divergence operators -- both return zero -- so an
un-stepped check is unfalsifiable.  The real-FFT `$k = 0$` plane is
excluded, since ``random_field`` solves continuity only off it by
design.

What is asserted: with the flag on, continuity holds *by algebra* at
every row and for any operator, grid or axis fit, so all three
geometries must sit at round-off -- and, unlike an `$h^p$` truncation
residual, **at every `$N_y$`**, which is why the gate-on bounds are
asserted across the whole ``--ny`` sweep.  Before 2026-07-26 the
annulus and the pipe used operator identities instead and were pinned
at `$8\\times10^{-6}$` / `$5.6\\times10^{-5}$`, floors set by a
commutator each could not remove; the reformulation has no such floor.
The plane-couette off case additionally reports the momentum side of
the trade (see ``_worker``): what relocating the residual out of
continuity costs in momentum units.

The pipe uses the deterministic axis-regular rolls IC.  That is no
longer a *requirement* (the composed `$D_2$` that made a grid-white
draw unstable is gone -- ``tests/test_random_smoke.py`` now carries
random-IC pipe entries under the flag); it stays because a grid-white
draw near the axis swamps the off-case measurement with
under-resolved-noise divergence.

A final ``--pipe-accepts`` subprocess asserts the flag is *settable*
on the pipe surface at all (it was a deferred hard error before the
spin-quad opt-in landed).

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
# default because it is fast, and is the only resolution the
# *annular/pipe* bounds are pinned at; other ``--ny`` values report
# those without asserting (``main``).  The Cartesian gate-on bound is
# asserted at every ny -- see the module docstring.  Measured, seed 7:
#
#     ny    plane-couette off / on     taylor-couette off / on
#     25    4.48e-02 / 2.91e-16        6.39e-02 / 5.62e-16
#     97    1.11e-03 / 1.63e-15        5.66e-04 / 1.92e-15
#
# and, on the pipe's rolls IC (PIPE_STEPS steps), 2.84e-02 / 2.07e-15
# at ny = 25 and 3.88e-15 gated at ny = 97.
#
# The ungated residual **falls** with resolution here (~40x over
# 25 -> 97, seed-robust over seeds 7/11/23).  Every gated one stays at
# round-off and follows no `$h^p$` law at all (the mild ny growth is
# the longer `$D_1$` dot product), because the reconstruction makes
# continuity an algebraic identity rather than something a solve has
# to deliver.  The gated annular/pipe numbers replace 8.00e-06 /
# 1.58e-08 and 5.6e-05, the floors of the operator-identity route
# retired on 2026-07-26.
#
# These numbers replace a set that was 2-3 orders larger and grew as
# `$N_y^2$` (2.08e-01 -> 9.86e-01 for plane-couette off).  That growth
# was the *initial condition*, not the scheme: ``_column_draw`` was
# grid-white in the wall-normal direction, and the boundary term's
# `$D_1[1,0] \sim N_y^2$` amplified its Nyquist content.  With
# ``random_field._wall_normal_filter`` supplying the missing
# wall-normal factor of the smoothness envelope, that content is gone.
# ``Resolution.consistent_imm`` records this table (re-baselined
# 2026-07-24).

# (label, system, consistent_imm, max relative divergence allowed).
# The gate-off bounds are loose regression pins on today's behaviour
# (measured ~4e-2 / ~6e-2 at NY); the gate-on bounds are the claim.
# Every gate-on bound is round-off, by algebra rather than by a solve,
# and pinned tight enough that the operator-identity mechanisms these
# replaced (4.2e-14 Cartesian, 8.0e-06 annular, 5.6e-05 pipe) could
# not pass it.
CASES = [
    ("plane-couette  off", "plane-couette", False, 1e0),
    ("plane-couette  on", "plane-couette", True, 1e-13),
    ("taylor-couette off", "taylor-couette", False, 1e0),
    ("taylor-couette on", "taylor-couette", True, 1e-13),
    ("pipe           off", "pipe", False, 1e0),
    ("pipe           on", "pipe", True, 1e-13),
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
            # parity-reduced radial D1 (parity (-1)^{m+1}) is the
            # mirrored fold -- there is one radial construction since
            # the x = r^2 fit was retired.
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

    if system == "plane-couette" and not consistent_imm:
        # The momentum side of the ``consistent_imm`` trade, measured
        # on this direct-fit-operator stepped state.  Both numbers are
        # what the *gated* schemes pay to buy continuity, priced on
        # the same un-gated field:
        #
        # - ``CHI-MOM``: an *upper bound* on the momentum price of the
        #   Cartesian v-omega_y route.  That route relocates the
        #   divergence residual out of continuity and into the
        #   chi-momentum equation (which it never solves), so what
        #   continuity gains, that equation loses:
        #   `$\\max|\\tilde H \\delta| / \\max|\\tilde H u|$` with
        #   `$\\delta = (i k_x, i k_z)\\,Q/k^2$`, i.e. this step's own
        #   divergence residual expressed in momentum units
        #   (4.5e-2 at ny = 25 -> 1.6e-3 at ny = 97).  It is only a
        #   bound because it assumes all of Q lands in the tangential
        #   pair at fixed v; differencing the two schemes' stepped
        #   states directly gives 2.4e-3 / 3.2e-5, ~20x smaller (the
        #   ``Resolution.consistent_imm`` docs).  Either way it is
        #   truncation-level and refines, and -- the point of the
        #   reformulation -- no solve reads it back, so it does not
        #   re-excite (the post-hoc projection that relocated the same
        #   residual *while* the solve still imposed chi-momentum was
        #   violently unstable; the ``_imm_iteration`` docs).
        # - ``COMPOSED-D2``: what the *retired* operator-identity route
        #   paid instead,
        #   `$\\nu\\,\\max|(D_1 D_1 - D_2)\\,u| / \\max|\\tilde H u|$`
        #   -- the extra viscous truncation composed operators inject
        #   into every momentum equation (full CN weight; upper
        #   bound).  ~2 orders below ``CHI-MOM`` here: that is the real
        #   price of the reformulation, and it bought exactness at
        #   every row, on any grid, with a narrower band and one fewer
        #   solve.  Kept as a measurement because it is the only
        #   quantitative comparison left of the two routes.
        #
        # Interior rows and the k = 0 plane excluded, like the
        # divergence measure.
        nz, nx = params.res.nz, params.res.nx
        s = np.asarray(stepped)[:, :, : nz - 1, : nx // 2]
        D1 = np.asarray(flow.D1)
        D2 = np.asarray(flow.D2)
        kx = np.asarray(fourier.kx)[..., : nx // 2]
        kz = np.asarray(fourier.kz)[:, : nz - 1]
        k2 = kx**2 + kz**2
        cw = params.step.implicitness
        nu = 1.0 / params.phys.re
        dt = params.step.dt

        q = 1j * kx * s[0] + np.einsum("ij, jzx -> izx", D1, s[1])
        q = q + 1j * kz * s[2]
        q[0] = 0.0
        q[-1] = 0.0
        q = np.where(k2 > 0, q / np.where(k2 > 0, k2, 1.0), 0.0)

        def h_tilde(f):
            visc = np.einsum("ij, jzx -> izx", D2, f) - k2 * f
            return f / dt - cw * nu * visc

        num = max(
            np.abs(h_tilde(d))[1:-1, :, 1:].max()
            for d in (1j * kx * q, 1j * kz * q)
        )
        den = max(np.abs(h_tilde(s[i]))[1:-1, :, 1:].max() for i in (0, 2))
        r_chi = num / den
        comp = np.einsum("ij, cjzx -> cizx", D1 @ D1 - D2, s)
        r_comp = nu * np.abs(comp)[:, 1:-1, :, 1:].max() / den
        print(f"CHI-MOM relocation = {r_chi:.6e}", flush=True)
        print(f"COMPOSED-D2 momentum price = {r_comp:.6e}", flush=True)

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
    # The gate-*off* bounds are pinned at ``NY`` only; ``--ny`` is a
    # *diagnostic* sweep for them (see the module docstring), reported
    # and not asserted, because they are absolute numbers that move
    # with resolution -- and, since the wall-normal smoothness envelope
    # landed, move a long way.  The gate-on bounds are ny-independent
    # and always asserted.
    asserted = ny == NY
    if not asserted:
        print(
            f"  (ny={ny} != {NY}: the gate-off values are reported,"
            " not asserted; every gate-on bound is ny-independent and"
            " still asserted)",
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
        # Every gate-on claim is an algebraic identity, so its bound
        # holds at any resolution -- asserting it across a ``--ny``
        # sweep is the refinement-flatness guard.
        flat = cimm
        if not asserted and not flat:
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
