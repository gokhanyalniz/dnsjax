r"""Discrete continuity of a stepped state, under both IMM formulations.

The legacy ``res.consistent_imm = False`` path leaves a stepped state a
truncation-level discrete divergence, amplified at the walls
(converging under refinement): not a bug, but not zero either.  The
**default** removes it in all three geometries by *reformulating* --
advancing the wall-normal velocity and vorticity and reconstructing the
tangential components, so continuity is algebra rather than something a
solve has to deliver.  The discrete identities the legacy path misses,
the reformulation and its measurements: ``res.consistent_imm``'s field
description and the :mod:`dnsjax.parameters` Design notes ("Consistent
influence matrix").  Both are exercised here, each passed explicitly.

Each case measures ``max|div| / max|individual term|`` on the true-mode
slice of a state that has been through one full predictor-corrector
step (``PIPE_STEPS`` on the pipe), using the solver's *own* divergence
assembly (the one ``dnsjax.analysis.snapshot_ops.divergence``
mirrors).  The measurement must use a **stepped** state: a
divergence-free field cannot discriminate two divergence operators --
both return zero -- so an un-stepped check is unfalsifiable.  The
measure excludes the real-FFT `$k = 0$` plane; the bounds and floors
below are measured on that slice.

What is asserted: under the default, continuity holds *by algebra* at
every row and for any operator, grid or axis fit, so all three
geometries must sit at round-off -- and, unlike an `$h^p$` truncation
residual, **at every `$N_y$`**, which is why the default bounds are
asserted across the whole ``--ny`` sweep.  Operator identities would
instead leave the annulus and the pipe at `$8\times10^{-6}$` /
`$5.6\times10^{-5}$`, floors set by a commutator each cannot remove;
the reconstruction has no such floor.  Each legacy case must exceed
``OFF_FLOOR`` and stay under a loose regression pin, both asserted
only at the default ``ny = 25`` (at another ``--ny`` its value is
reported: an absolute number that moves a long way with resolution).
The plane-couette legacy case additionally reports the momentum side of
the trade (see ``_worker``): what relocating the residual out of
continuity costs in momentum units.

The pipe uses the deterministic axis-regular rolls IC.  A random IC
steps stably too (``tests/test_random_smoke.py`` carries random-IC
pipe entries), but a grid-white draw has no continuum limit near the
axis: its under-resolved-noise divergence swamps the legacy-case
measurement, and the gate shows no clean gain.

A final ``--pipe-accepts`` subprocess asserts the flag is *settable*
on the pipe surface at all.

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
# default because it is fast, and the only resolution the *legacy*
# bounds are pinned at (module docstring).  Measured, seed 7:
#
#     ny    plane-couette legacy / default  taylor-couette legacy / default
#     25    4.48e-02 / 2.91e-16             6.39e-02 / 5.62e-16
#     97    1.11e-03 / 1.63e-15             5.66e-04 / 1.92e-15
#
# and, on the pipe's rolls IC (PIPE_STEPS steps), 2.84e-02 / 1.08e-15
# at ny = 25 and 1.51e-05 / 8.24e-15 at ny = 97.
#
# The legacy residual **falls** with resolution in every geometry
# (~40x over 25 -> 97 on plane-couette, at each of seeds
# 7/11/23; ~110x on taylor-couette and ~1900x on the pipe).  Every
# default one stays at round-off (the mild ny growth is the longer
# `$D_1$` dot product).
#
# The legacy numbers depend on the IC's wall-normal smoothness:
# ``random_field.wall_normal_filter`` keeps grid-white Nyquist
# content out of the boundary term's `$D_1[1,0] \sim N_y^2$`
# amplification (without it they are 2-3 orders larger and *grow* as
# `$N_y^2$`).  The :mod:`dnsjax.parameters` Design notes ("Consistent
# influence matrix") record this table.

# (label, system, consistent_imm, max relative divergence allowed).
# Both formulations are named explicitly, so the pair stays a contrast
# whatever the model default is.  The legacy bounds are loose
# regression pins on the current behaviour (the table above); the
# default bounds are the claim, round-off and tight enough that an
# operator-identity mechanism could not pass them on the annulus or
# the pipe (floors 8.0e-06 / 5.6e-05; the Cartesian one, 4.2e-14, sits
# below the bound).
CASES = [
    ("plane-couette  legacy", "plane-couette", False, 1e0),
    ("plane-couette  default", "plane-couette", True, 1e-13),
    ("taylor-couette legacy", "taylor-couette", False, 1e0),
    ("taylor-couette default", "taylor-couette", True, 1e-13),
    ("pipe           legacy", "pipe", False, 1e0),
    ("pipe           default", "pipe", True, 1e-13),
]

# Legacy floors: the two formulations must *demonstrably* differ, so
# each legacy case also asserts the residual is at least this large.
# Without it the default bounds could pass for the wrong reason.
OFF_FLOOR = 1e-3


# ── worker (runs in its own process) ─────────────────────────────────


def _worker(system: str, consistent_imm: bool, ny: int) -> None:
    import numpy as np

    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
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
    configure_jax_platform(platform_from_argv())

    import jax.numpy as jnp

    from dnsjax.flows.registry import spec_for

    mod = __import__(spec_for(system).flow_module, fromlist=["x"])
    flow, fourier = mod.flow, mod.fourier

    # Cartesian has no basis crossing; cylindrical/annular carry the
    # decoupled u_+/u_- solver basis and must be converted first.
    to_solver = getattr(mod, "to_solver_basis", lambda s: s)
    from_solver = getattr(mod, "from_solver_basis", lambda s: s)

    if system == "pipe":
        from dnsjax.ic.localized_rolls import generate_localized_rolls

        state = generate_localized_rolls(0.15, 0.5, 1.0)
        e0 = float(mod.get_perturbation_energy(state))
        state = state * float(np.sqrt(1e-2 / e0))
    else:
        from dnsjax.ic.random_field import generate_random_state

        state = generate_random_state(0.2, 0.4, 0.4, 0.14, 7)
    # A random IC gives a clear legacy divergence in one step.
    n_steps = PIPE_STEPS if system == "pipe" else 1
    stepped = jnp.copy(to_solver(state))
    for _ in range(n_steps):
        stepped, *_ = mod.predict_and_fully_correct(stepped)
    stepped = from_solver(stepped)

    def divergence(st) -> float:
        """Relative divergence, assembled as ``_imm_iteration`` does."""
        nz, nx = params.res.nz, params.res.nx
        s = np.asarray(st)[:, :, : nz - 1, : nx // 2]
        if system == "plane-couette":
            D1 = np.asarray(flow.D1.dense)
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
            # mirrored fold, the only radial construction.
            D1p = np.asarray(flow.D1_pos.dense)
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
                o[:gg] += psv * np.einsum(
                    "ij, jzx -> izx", D1g, u[: D1g.shape[1]]
                )
                return o

            terms = [
                (_dy_v(up) + (m + 1) * inv_r * up) / 2,
                (_dy_v(um) + (1 - m) * inv_r * um) / 2,
                1j * kz * s[0],
            ]
        else:  # annular (taylor-couette)
            D1 = np.asarray(flow.D1.dense)
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
        # on this direct-fit-operator stepped state: what the default
        # scheme pays to buy continuity, priced on the same legacy
        # field.
        #
        # - ``CHI-MOM``: an *upper bound* on the momentum price of the
        #   Cartesian v-omega_y route, which relocates the divergence
        #   residual out of continuity and into the chi-momentum
        #   equation (which it never solves):
        #   `$\max|\tilde H \delta| / \max|\tilde H u|$` with
        #   `$\delta = (i k_x, i k_z)\,Q/k^2$`, i.e. this step's own
        #   divergence residual expressed in momentum units
        #   (4.5e-2 at ny = 25 -> 1.6e-3 at ny = 97).  It is only a
        #   bound because it assumes all of Q lands in the tangential
        #   pair at fixed v; differencing the two schemes' stepped
        #   states directly gives 2.4e-3 / 3.2e-5, ~20x smaller.  Why
        #   the residual is harmless (truncation-level, refining, read
        #   back by no solve): the ``parameters`` Design notes
        #   ("Consistent influence matrix", *Price*); a post-hoc
        #   projection relocating the same residual *while* the solve
        #   still imposes chi-momentum is unstable (the
        #   ``_imm_iteration`` docs).
        #
        # Interior rows and the k = 0 plane excluded, like the
        # divergence measure.
        nz, nx = params.res.nz, params.res.nx
        s = np.asarray(stepped)[:, :, : nz - 1, : nx // 2]
        D1 = np.asarray(flow.D1.dense)
        D2 = np.asarray(flow.D2.dense)
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
        print(f"CHI-MOM relocation = {r_chi:.6e}", flush=True)

    print(f"RESULT {rel:.6e}", flush=True)
    assert rel_ic < 1e-12, f"IC is not divergence-free: {rel_ic:.2e}"


def _worker_pipe_accepts() -> None:
    """``res.consistent_imm`` must be settable on the pipe."""
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
    # The legacy bounds hold at ``NY`` only (module docstring).
    asserted = ny == NY
    if not asserted:
        print(
            f"  (ny={ny} != {NY}: the legacy values are reported,"
            " not asserted; every default bound is ny-independent and"
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
        # The default bounds hold at any ny: the refinement-flatness
        # guard.
        flat = cimm
        if not asserted and not flat:
            print(f"REPORT {label}: rel div {rel:.2e}")
            passed += 1
            continue
        if rel > bound:
            reason = f"rel div {rel:.2e} > {bound:.0e}"
        elif not cimm and rel < OFF_FLOOR:
            reason = (
                f"rel div {rel:.2e} < {OFF_FLOOR:.0e} on the legacy "
                "path -- the default bound would pass for the wrong "
                "reason"
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
