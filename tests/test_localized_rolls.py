#!/usr/bin/env python3
r"""Construction self-test for the in-process IC builders.

Builds the deterministic streamwise-localized-rolls IC
(``dnsjax.ic.localized_rolls``) for each wall-bounded flow and checks the
*construction* properties the smoke test (``tests/test_rolls_smoke.py``)
cannot -- without time-stepping -- plus, in the same subprocess (no
extra launch), the divergence guard on the random-field IC
(``dnsjax.ic.random_field``, the default start mode).  Flows in
``RANDOM_ONLY`` run the random half only, and their random state
carries the checks the rolls state carries elsewhere:

- **finiteness** of the built spectral state;
- **exact no-slip** at the wall nodes (the wall ``y`` / ``r`` slice of
  the *velocity* block is identically zero, never transformed): both
  walls for Cartesian / annular, the outer wall ``r = 1`` for the
  cylindrical family (the inner end is the axis); for the total-field
  flows (Dean, viscoelastic) the *total* velocity still vanishes,
  while their conformation block does not -- its wall condition is
  ``grad^2 c = 0``, not a Dirichlet zero;
- **bit-identical determinism** (two builds in one process agree) and
  **device-count independence** (the true modes are identical at
  ``(np0, np1) = (1, 1)``, ``(1, 2)`` and ``(2, 1)``) -- the
  no-replication, per-device build must not depend on the mesh; on
  the random state that independence is to ``RAND_CROSS_TOL`` rather
  than bit-exact, for the reason recorded there;
- an **exactly zero (kx, kz) = (0, 0) mode** on every component, so a
  spot leaves the field's bulk velocity and wall shear untouched (the
  "Mean-free by construction" note in :mod:`dnsjax.ic.localized_rolls`);
  checked at a box whose ``L / wavelength`` is not an integer, the only
  regime in which the roll factors have a DC bin to leak;
- a **loose truncation-level discrete-divergence bound** (the analytic
  profiles are continuously divergence-free; the discrete divergence is
  only FD-truncation-sized and is projected out by the first corrector
  step at run start): Cartesian is ~machine-zero (the FD order
  differentiates the quartic profile exactly), the rational-profile
  annular/pipe are `$O(10^{-5})$`;
- a **peak-velocity / domain-scaling guard** (``_check_peak_scaling``,
  subprocess ``--peak``): built at two resolved box sizes, the sampled
  ``max|u'|`` is within ``PEAK_TOL`` (12%) of ``amplitude`` and
  domain-independent -- guarding against a regression to the old
  single-mode rolls whose amplitude grew in proportion to the box
  length.

Each ``(system, np0, np1)`` configuration runs in its own subprocess with
forced CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count``), mirroring
``tests/test_snapshot.py`` -- so the multi-device cases need no MPI.  Run
directly::

    uv run python tests/test_localized_rolls.py            # all systems
    uv run python tests/test_localized_rolls.py --system pipe   # one cfg

Each subprocess writes its true-mode array to a ``.npy`` so the driver can
compare across device counts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

# Small resolutions chosen so the multi-device cases exercise spectral
# padding: nx // 2 = 5 (odd -> k_x padded for np1 = 2); nz - 1 = 11 (odd
# -> k_z / m padded for np0 = 2).
NX, NY, NZ = 10, 15, 12
LX, LZ = 5.0, 5.0
# ``L / WAVELENGTH = 5/3`` is deliberately **not** an integer: that is
# the only regime in which a roll factor has a nonzero DC bin, so it is
# what makes the mean-mode check below a real guard.  At an integral
# ratio (the shipped defaults, and the old 2.5 here) the leak this
# checks for is machine-zero by coincidence and every build passes.
AMP, WIDTH, WAVELENGTH = 0.1, 1.5, 3.0  # physical width / cross-roll wave

SYSTEMS = [
    "plane-couette",
    "plane-poiseuille",
    "pipe",
    "taylor-couette",
    "dean",
    "viscoelastic-dean",
    "viscoelastic-pipe",
]

# Flows that run the random-field half of the worker only, with the
# cross-device comparison riding the random state instead of the rolls
# one.  Both 9-component flows exercise the builders'
# component-count-agnostic path (the conformation block is carried
# along; the velocity block is what continuity is solved on) -- and
# ``viscoelastic-pipe`` additionally runs the full rolls half, so the
# 9-component rolls path is covered there.
RANDOM_ONLY = ["viscoelastic-dean"]

# Configurations (np0, np1) to build at; (1, 1) is the reference.
CONFIGS = [(1, 1), (1, 2), (2, 1)]

DIV_TOL = 5e-2  # loose truncation-level discrete-divergence bound

# Random-field IC guard: amplitude / smoothness / seed, and the
# *relative* divergence bound over the whole field, k_z = 0 included
# (the builders solve continuity on every mode, so this is machine-zero).
RAND_AMP, RAND_SMOOTH, RAND_SEED = 0.2, 0.4, 7
# The wall-normal pair, at the shipped defaults: the divergence bound
# below is what proves the scale-dependent wall window is structurally
# inert (it reshapes a column, it does not touch the closure).
RAND_WALL_SMOOTH, RAND_WALL_CONF = 0.4, 0.14
RAND_DIV_TOL = 1e-11

# Cross-device-count comparison tolerance for the random state
# (``RANDOM_ONLY`` flows; the rolls state is compared bit-exactly).
# The random builders draw per column in host NumPy -- bit-exact by
# construction -- but their final rescaling divides by a *device-side
# global reduction* (``get_norm2_*`` over the sharded mode axes),
# whose summation order, and so whose last ulp, depends on the mesh.
# That single factor multiplies the whole block: measured <= 2 ulp
# relative, with the laminar mean-mode column -- added after the
# scaling -- bit-identical.  The structural property, exact discrete
# continuity, is bit-exact at every configuration and is checked as
# such above.
RAND_CROSS_TOL = 1e-13

# Peak-velocity / domain-scaling guard.  The spot is peak-normalized so
# max|u'| = amplitude and is domain-independent (the old single-mode
# construction blew up the cross-stream velocity ~ box length).  Built at
# two resolved boxes (the spot fits and is well sampled at both) with the
# spanwise/axial wavelength a fixed physical value, so peak ~ amplitude
# must hold at both and agree across them.
PEAK_SYSTEMS = ["plane-couette", "pipe", "taylor-couette"]
PEAK_AMP, PEAK_WIDTH, PEAK_WAVE = 0.1, 2.0, 4.0
PEAK_BOXES = ((20.0, 48), (40.0, 96))  # (box length, resolution)
PEAK_TOL = 0.12  # |peak/amp - 1| and cross-box agreement bound


# ── parameter / JAX setup (forced CPU devices) ───────────────────


def _configure(system: str, np0: int, np1: int) -> None:
    """Configure JAX and the dnsjax parameter singletons for *np0*x*np1*
    forced CPU devices.  Must run before importing ``sharding`` / the
    geometry modules."""
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={np0 * np1}"
    )
    os.environ["NPROC"] = "1"

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
        phys.update(re1=100.0, re2=-100.0)
        geo["eta"] = 0.5
    elif system == "dean":
        geo["eta"] = 0.5
    elif system in ("viscoelastic-dean", "viscoelastic-pipe"):
        # Re := Wi/El is derived (the annulus additionally takes its
        # geometry from ``geo.delta``); the rheology defaults (spec)
        # are left alone.
        phys.pop("re")

    update_parameters(
        Parameters(
            dist={"np0": np0, "np1": np1, "platform": "cpu"},
            phys=phys,
            geo=geo,
            res={
                "nx": NX,
                "ny": NY,
                "nz": NZ,
                "fd_order": 4,
                "double_precision": True,
            },
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)


# ── per-geometry discrete divergence (host numpy) ────────────────


def _max_divergence(true: np.ndarray, system: str) -> float:
    r"""Max |discrete divergence| of the true-mode state (host numpy).

    The FD matrices are built with the run's **resolved** wall-normal
    grid selection (``geo.grid_type`` etc.), not the builder defaults
    -- pipe resolves to ``half-cgl``, whose `$D_1$` differs from the
    default grid's, and a mismatched reference would silently measure
    the wrong operator.  The whole field is measured, including the
    `$k_z = 0$` plane: the random-field builders now solve continuity
    on every mode (see :mod:`dnsjax.ic.random_field`).
    """
    from dnsjax.flows.registry import cylindrical_systems
    from dnsjax.operators import complex_harmonics, real_harmonics
    from dnsjax.parameters import derived_params, params

    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz
    fd = params.res.fd_order
    grid_args = (
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    kx_real = np.asarray(real_harmonics(nx)) * (2 * np.pi / params.geo.lx)
    if system in ("plane-couette", "plane-poiseuille"):
        from dnsjax.geometries.wall_bounded.cartesian import (
            build_cartesian_grid,
        )

        _, d1, _, _ = build_cartesian_grid(ny, fd, *grid_args)
        d1 = np.asarray(d1)
        kz = np.asarray(complex_harmonics(nz)) * (2 * np.pi / params.geo.lz)
        duy = np.einsum("ij,jzx->izx", d1, true[1])
        div = (
            duy
            + 1j * kz[None, :, None] * true[2]
            + 1j * kx_real[None, None, :] * true[0]
        )
        return float(np.max(np.abs(div)))

    # pipe / annular: native (u_z, u_r, u_theta) over (r, m, k_z,ax).
    # Keyed on the *geometry* list, which spans each geometry's
    # viscoelastic member too (``dnsjax.flows.registry``).
    m = params.geo.m0 * np.asarray(complex_harmonics(nz))
    if system in cylindrical_systems:
        from dnsjax.geometries.wall_bounded.cylindrical import (
            build_cylindrical_grid,
        )

        _, d1_even, d1_odd, _, _, _, inv_r = build_cylindrical_grid(
            ny, fd, *grid_args
        )
        d1_even, d1_odd = np.asarray(d1_even), np.asarray(d1_odd)
        inv_r = np.asarray(inv_r)
    else:  # taylor-couette / dean / viscoelastic-dean
        from dnsjax.geometries.wall_bounded.annular import build_annular_grid

        r1, r2 = derived_params.r_inner, derived_params.r_outer
        _, d1, _, _, inv_r = build_annular_grid(ny, fd, r1, r2, *grid_args)
        d1, inv_r = np.asarray(d1), np.asarray(inv_r)

    div = np.zeros_like(true[0])
    for im, mv in enumerate(m):
        if system in cylindrical_systems:
            d1v = d1_even if (mv + 1) % 2 == 0 else d1_odd
        else:
            d1v = d1
        ur, uth, uz = true[1][:, im, :], true[2][:, im, :], true[0][:, im, :]
        div_perp = (
            d1v @ ur + inv_r[:, None] * ur + 1j * mv * inv_r[:, None] * uth
        )
        div[:, im, :] = div_perp + 1j * kx_real[None, :] * uz
    return float(np.max(np.abs(div)))


# ── worker ───────────────────────────────────────────────────────


def _run_worker(system: str, np0: int, np1: int, out_npy: str) -> int:
    _configure(system, np0, np1)

    from dnsjax.parameters import params

    nx, nz = params.res.nx, params.res.nz
    passed = failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        print(f"  {'PASS' if ok else 'FAIL'}: {name}  {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    # Strip mesh padding -> true modes (padding is at the global axis end).
    def _true(state: np.ndarray) -> np.ndarray:
        return state[:3, :, : nz - 1, : nx // 2]

    true: np.ndarray | None = None
    if system not in RANDOM_ONLY:
        from dnsjax.flows.registry import cylindrical_systems
        from dnsjax.ic.localized_rolls import generate_localized_rolls

        state1 = np.asarray(generate_localized_rolls(AMP, WIDTH, WAVELENGTH))
        state2 = np.asarray(generate_localized_rolls(AMP, WIDTH, WAVELENGTH))
        true = _true(state1)

        check("finite", bool(np.all(np.isfinite(state1))))
        check("determinism (build twice)", np.array_equal(state1, state2))

        # Exact no-slip at the wall nodes (axis 1 = y / r), on the
        # **velocity** block: a viscoelastic total-field state also
        # carries the laminar conformation there, whose wall values are
        # O(1)-to-O(Wi^2) by construction (its BC is grad^2 c = 0, not
        # a Dirichlet zero).  ``[:3]`` is the whole state for every
        # velocity-only flow.
        vel1 = state1[:3]
        if system in cylindrical_systems:
            wall = float(np.max(np.abs(vel1[:, -1])))  # outer wall r = 1
        else:
            wall = max(
                float(np.max(np.abs(vel1[:, 0]))),
                float(np.max(np.abs(vel1[:, -1]))),
            )
        check(
            "exact no-slip at walls", wall < 1e-12, f"max|u|_wall={wall:.2e}"
        )

        div = _max_divergence(true, system)
        check(
            "divergence truncation-level", div < DIV_TOL, f"max|div|={div:.2e}"
        )

        # The spot contributes **nothing** to the (0, 0) mean mode, so
        # it moves neither the bulk velocity nor the wall shear.  Stated
        # against an ``amplitude = 0`` build so it reads the same for
        # the perturbation flows (whose mean column is then identically
        # zero) and the total-field ones (whose mean column carries the
        # analytical laminar profile).  Bit-exact, not a tolerance --
        # the roll factors' DC bins are hard zeros (``_zero_dc``).
        zero_amp = np.asarray(generate_localized_rolls(0.0, WIDTH, WAVELENGTH))
        same_mean = np.array_equal(state1[:, :, 0, 0], zero_amp[:, :, 0, 0])
        moved = float(
            np.max(np.abs(state1[:, :, 0, 0] - zero_amp[:, :, 0, 0]))
        )
        check(
            "spot adds nothing to the (0, 0) mean mode",
            same_mean,
            f"max|delta u(0,0)|={moved:.2e}",
        )

    # ── random-field IC: exact discrete continuity on the *whole*
    # field, k_z = 0 plane included.  Unlike the analytic rolls
    # (truncation-level), the random builders *solve* continuity per mode
    # -- u_z for k_z != 0, u_theta for k_z = 0 (m != 0), u_r = 0 at the
    # mean -- so the residual is machine-zero, a sharp guard on the
    # per-geometry divergence expression each builder inverts.
    from dnsjax.ic.random_field import generate_random_state

    rand = np.asarray(
        generate_random_state(
            RAND_AMP, RAND_SMOOTH, RAND_WALL_SMOOTH, RAND_WALL_CONF, RAND_SEED
        )
    )
    rand_true = _true(rand)
    scale = float(np.max(np.abs(rand_true)))
    rdiv = _max_divergence(rand_true, system)
    check(
        "random IC divergence-free (full field)",
        rdiv < RAND_DIV_TOL * scale,
        f"max|div|={rdiv:.2e} scale={scale:.2e}",
    )

    if true is None:
        # Rolls-less flow: the random state carries the finiteness,
        # determinism and cross-device checks instead -- and it carries
        # them for *all* its components (the conformation block
        # included), not just the velocity block continuity is solved
        # on.
        check("finite", bool(np.all(np.isfinite(rand))))
        rand2 = np.asarray(
            generate_random_state(
                RAND_AMP,
                RAND_SMOOTH,
                RAND_WALL_SMOOTH,
                RAND_WALL_CONF,
                RAND_SEED,
            )
        )
        check("determinism (build twice)", np.array_equal(rand, rand2))
        true = rand[:, :, : nz - 1, : nx // 2]
    np.save(out_npy, true)

    print(f"\n[{system} {np0}x{np1}] {passed} passed, {failed} failed.")
    return 1 if failed else 0


# ── driver ───────────────────────────────────────────────────────


def _run_config(
    system: str, np0: int, np1: int, out_npy: str
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--system",
        system,
        "--np0",
        str(np0),
        "--np1",
        str(np1),
        "--out",
        out_npy,
    ]
    return run_live(cmd, timeout=300)


def _run_system(system: str) -> str | None:
    """Run all configs for one system; ``None`` on a full pass.

    On failure returns the one-line reason for the summary, quoting
    the worker's own first ``FAIL:`` line where it has one -- the
    parent otherwise only knows the exit code.
    """
    print(f"=== {system} ===")
    bad: list[str] = []
    ref: np.ndarray | None = None
    with tempfile.TemporaryDirectory() as tmp:
        for np0, np1 in CONFIGS:
            out = str(Path(tmp) / f"true_{np0}x{np1}.npy")
            proc = _run_config(system, np0, np1, out)
            tag = f"np0={np0} np1={np1}"
            if proc.returncode != 0:
                print(f"  FAIL: {tag} worker exit {proc.returncode}")
                print(proc.stdout[-1500:])
                print(proc.stderr[-1500:])
                inner = next(
                    (
                        ln.strip()
                        for ln in proc.stdout.splitlines()
                        if ln.strip().startswith("FAIL:")
                    ),
                    "",
                )
                bad.append(
                    f"{tag} worker exit {proc.returncode}"
                    + (f" ({inner})" if inner else "")
                )
                continue
            arr = np.load(out)
            if ref is None:
                ref = arr
            else:
                same = arr.shape == ref.shape and (
                    np.allclose(arr, ref, rtol=RAND_CROSS_TOL, atol=0.0)
                    if system in RANDOM_ONLY
                    else np.array_equal(arr, ref)
                )
                print(
                    f"  {'PASS' if same else 'FAIL'}: {tag} matches (1, 1) "
                    "true modes"
                )
                if not same:
                    bad.append(f"{tag} differs from the (1, 1) true modes")
    return "; ".join(bad) if bad else None


def _peak_build(system: str, box: float, n: int) -> int:
    """Subprocess: build rolls at a resolved box; print ``PEAK=max|u'|``."""
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    os.environ["NPROC"] = "1"
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
    geo: dict = {"lx": box, "lz": box}
    if system == "taylor-couette":
        phys.update(re1=100.0, re2=-100.0)
        geo["eta"] = 0.5
    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": "cpu"},
            phys=phys,
            geo=geo,
            res={
                "nx": n,
                "ny": 33,
                "nz": n,
                "fd_order": 4,
                "double_precision": True,
            },
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)

    from dnsjax.ic.localized_rolls import generate_localized_rolls
    from dnsjax.operators import spec_to_phys_2d

    state = generate_localized_rolls(PEAK_AMP, PEAK_WIDTH, PEAK_WAVE)
    pf = np.asarray(spec_to_phys_2d(state))
    # Native components are orthonormal in every geometry:
    # |u'|^2 is the plain component sum.
    mag = np.sqrt(np.sum(pf**2, axis=0))
    print(f"PEAK={float(mag.max())}")
    return 0


def _check_peak_scaling() -> str | None:
    """Peak |u'| = amplitude, domain-independent (subprocess per build).

    ``None`` on a full pass, else the one-line summary reason.
    """
    print("=== peak |u'| = amplitude (domain-independent spot) ===")
    bad: list[str] = []
    for system in PEAK_SYSTEMS:
        peaks = []
        for box, n in PEAK_BOXES:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--peak",
                "--system",
                system,
                "--box",
                str(box),
                "--n",
                str(n),
            ]
            proc = run_live(cmd, timeout=300)
            line = next(
                (
                    ln
                    for ln in proc.stdout.splitlines()
                    if ln.startswith("PEAK=")
                ),
                None,
            )
            if proc.returncode != 0 or line is None:
                print(f"  FAIL: {system} box={box} worker error")
                print(proc.stderr[-800:])
                bad.append(f"{system} box={box} worker error")
                break
            peaks.append(float(line.split("=")[1]))
        else:
            near = all(abs(p / PEAK_AMP - 1) < PEAK_TOL for p in peaks)
            indep = abs(peaks[0] - peaks[1]) / PEAK_AMP < PEAK_TOL
            print(
                f"  {'PASS' if near and indep else 'FAIL'}: {system}  "
                f"peak(L{int(PEAK_BOXES[0][0])})={peaks[0]:.4f} "
                f"peak(L{int(PEAK_BOXES[1][0])})={peaks[1]:.4f} "
                f"(amp={PEAK_AMP})"
            )
            if not (near and indep):
                bad.append(
                    f"{system} peaks {peaks[0]:.4f}/{peaks[1]:.4f} "
                    f"vs amp={PEAK_AMP}"
                )
    return "; ".join(bad) if bad else None


def main() -> None:
    if "--peak" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--peak", action="store_true")
        p.add_argument("--system", required=True)
        p.add_argument("--box", type=float, required=True)
        p.add_argument("--n", type=int, required=True)
        a = p.parse_args()
        sys.exit(_peak_build(a.system, a.box, a.n))

    if "--system" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--system", required=True)
        p.add_argument("--np0", type=int, default=1)
        p.add_argument("--np1", type=int, default=1)
        p.add_argument("--out", required=True)
        a = p.parse_args()
        sys.exit(_run_worker(a.system, a.np0, a.np1, a.out))

    print(
        "Localized-rolls construction self-test: offline, forced CPU "
        "devices (device-count independence is the property under test; "
        "no GPU path).",
        flush=True,
    )
    # Each check returns None when it passes, else its one-line reason;
    # ``report`` repeats the failures after the counts (see _live).
    results: list[tuple[str, str | None]] = [
        (system, _run_system(system)) for system in SYSTEMS
    ]
    results.append(("peak |u'| scaling", _check_peak_scaling()))
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))


if __name__ == "__main__":
    main()
