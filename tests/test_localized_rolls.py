#!/usr/bin/env python3
r"""Construction self-test for ``dnsjax.localized_rolls``.

Builds the deterministic streamwise-localized-rolls IC for each
wall-bounded flow and checks the *construction* properties the smoke test
(``tests/test_rolls_smoke.py``) cannot -- without time-stepping:

- **finiteness** of the built spectral state;
- **exact no-slip** at the wall nodes (the wall ``y`` / ``r`` slice is
  identically zero, never transformed): both walls for Cartesian /
  annular, the outer wall ``r = 1`` for pipe (the inner end is the axis);
  for Dean the *total* field (perturbation + laminar) still vanishes;
- **bit-identical determinism** (two builds in one process agree) and
  **device-count independence** (the true modes are identical at
  ``(np0, np1) = (1, 1)``, ``(1, 2)`` and ``(2, 1)``) -- the
  no-replication, per-device build must not depend on the mesh;
- a **loose truncation-level discrete-divergence bound** (the analytic
  profiles are continuously divergence-free; the discrete divergence is
  only FD-truncation-sized and is projected out by the first corrector
  step at run start).

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

# Small resolutions chosen so the multi-device cases exercise spectral
# padding: nx // 2 = 5 (odd -> k_x padded for np1 = 2); nz - 1 = 11 (odd
# -> k_z / m padded for np0 = 2).
NX, NY, NZ = 10, 15, 12
LX, LZ = 5.0, 5.0
AMP, WIDTH, WAVELENGTH = 0.1, 1.5, 2.5  # physical width / cross-roll wave

SYSTEMS = [
    "plane-couette",
    "plane-poiseuille",
    "pipe",
    "taylor-couette",
    "dean",
]

# Configurations (np0, np1) to build at; (1, 1) is the reference.
CONFIGS = [(1, 1), (1, 2), (2, 1)]

DIV_TOL = 5e-2  # loose truncation-level discrete-divergence bound

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
    r"""Max |discrete divergence| of the true-mode state (host numpy)."""
    from dnsjax.operators import complex_harmonics, real_harmonics
    from dnsjax.parameters import derived_params, params

    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz
    fd = params.res.fd_order
    kx_real = np.asarray(real_harmonics(nx)) * (2 * np.pi / params.geo.lx)
    if system in ("plane-couette", "plane-poiseuille"):
        from dnsjax.geometries.wall_bounded.cartesian import (
            build_cartesian_grid,
        )

        _, d1, _, _ = build_cartesian_grid(ny, fd)
        d1 = np.asarray(d1)
        kz = np.asarray(complex_harmonics(nz)) * (2 * np.pi / params.geo.lz)
        duy = np.einsum("ij,jzx->izx", d1, true[1])
        div = (
            duy
            + 1j * kz[None, :, None] * true[2]
            + 1j * kx_real[None, None, :] * true[0]
        )
        return float(np.max(np.abs(div)))

    # pipe / annular: decoupled (u_z, u_+, u_-) over (r, m, k_z,ax).
    m = np.asarray(complex_harmonics(nz))
    if system == "pipe":
        from dnsjax.geometries.wall_bounded.cylindrical import (
            build_cylindrical_grid,
        )

        _, d1_even, d1_odd, _, _, _, inv_r = build_cylindrical_grid(ny, fd)
        d1_even, d1_odd = np.asarray(d1_even), np.asarray(d1_odd)
        inv_r = np.asarray(inv_r)
    else:  # taylor-couette / dean
        from dnsjax.geometries.wall_bounded.annular import build_annular_grid

        r1, r2 = derived_params.r_inner, derived_params.r_outer
        _, d1, _, _, inv_r = build_annular_grid(ny, fd, r1, r2)
        d1, inv_r = np.asarray(d1), np.asarray(inv_r)

    div = np.zeros_like(true[0])
    for im, mv in enumerate(m):
        if system == "pipe":
            d1pm = d1_even if (mv + 1) % 2 == 0 else d1_odd
        else:
            d1pm = d1
        up, um, uz = true[1][:, im, :], true[2][:, im, :], true[0][:, im, :]
        div_rad = (
            d1pm @ up
            + (mv + 1) * inv_r[:, None] * up
            + d1pm @ um
            + (1 - mv) * inv_r[:, None] * um
        ) / 2.0
        div[:, im, :] = div_rad + 1j * kx_real[None, :] * uz
    return float(np.max(np.abs(div)))


# ── worker ───────────────────────────────────────────────────────


def _run_worker(system: str, np0: int, np1: int, out_npy: str) -> int:
    _configure(system, np0, np1)

    from dnsjax.localized_rolls import generate_localized_rolls
    from dnsjax.parameters import params

    state1 = np.asarray(generate_localized_rolls(AMP, WIDTH, WAVELENGTH))
    state2 = np.asarray(generate_localized_rolls(AMP, WIDTH, WAVELENGTH))

    nx, nz = params.res.nx, params.res.nz
    # Strip mesh padding -> true modes (padding is at the global axis end).
    true = state1[:, :, : nz - 1, : nx // 2]
    np.save(out_npy, true)

    passed = failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        print(f"  {'PASS' if ok else 'FAIL'}: {name}  {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    check("finite", bool(np.all(np.isfinite(state1))))
    check("determinism (build twice)", np.array_equal(state1, state2))

    # Exact no-slip at the wall nodes (axis 1 = y / r).
    if system == "pipe":
        wall = float(np.max(np.abs(state1[:, -1])))  # outer wall r = 1
    else:
        wall = max(
            float(np.max(np.abs(state1[:, 0]))),
            float(np.max(np.abs(state1[:, -1]))),
        )
    check("exact no-slip at walls", wall < 1e-12, f"max|u|_wall={wall:.2e}")

    div = _max_divergence(true, system)
    check("divergence truncation-level", div < DIV_TOL, f"max|div|={div:.2e}")

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
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def _run_system(system: str) -> bool:
    """Run all configs for one system; return True on full pass."""
    print(f"=== {system} ===")
    ok = True
    ref: np.ndarray | None = None
    with tempfile.TemporaryDirectory() as tmp:
        for np0, np1 in CONFIGS:
            out = str(Path(tmp) / f"true_{np0}x{np1}.npy")
            proc = _run_config(system, np0, np1, out)
            tag = f"np0={np0} np1={np1}"
            if proc.returncode != 0:
                ok = False
                print(f"  FAIL: {tag} worker exit {proc.returncode}")
                print(proc.stdout[-1500:])
                print(proc.stderr[-1500:])
                continue
            # Echo the worker's own per-config checks.
            for line in proc.stdout.splitlines():
                if line.lstrip().startswith(("PASS", "FAIL")):
                    print(f"  [{tag}] {line.strip()}")
            arr = np.load(out)
            if ref is None:
                ref = arr
            else:
                same = arr.shape == ref.shape and np.array_equal(arr, ref)
                print(
                    f"  {'PASS' if same else 'FAIL'}: {tag} matches (1, 1) "
                    "true modes"
                )
                ok = ok and same
    return ok


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

    from dnsjax.localized_rolls import generate_localized_rolls
    from dnsjax.operators import spec_to_phys_2d

    state = generate_localized_rolls(PEAK_AMP, PEAK_WIDTH, PEAK_WAVE)
    pf = np.asarray(spec_to_phys_2d(state))
    if system in ("plane-couette", "plane-poiseuille"):
        mag = np.sqrt(np.sum(pf**2, axis=0))
    else:  # (u_z, u_+, u_-): |u'|^2 = |u_z|^2 + (|u_+|^2 + |u_-|^2)/2
        mag = np.sqrt(
            np.abs(pf[0]) ** 2 + (np.abs(pf[1]) ** 2 + np.abs(pf[2]) ** 2) / 2
        )
    print(f"PEAK={float(mag.max())}")
    return 0


def _check_peak_scaling() -> bool:
    """Peak |u'| = amplitude, domain-independent (subprocess per build)."""
    print("=== peak |u'| = amplitude (domain-independent spot) ===")
    ok = True
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
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
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
                ok = False
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
            ok = ok and near and indep
    return ok


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

    failures = 0
    for system in SYSTEMS:
        if not _run_system(system):
            failures += 1
    if not _check_peak_scaling():
        failures += 1
    if failures:
        print(f"\n{failures} check(s) FAILED.")
        sys.exit(1)
    print("\nAll systems passed.")


if __name__ == "__main__":
    main()
