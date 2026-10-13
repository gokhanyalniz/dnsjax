r"""Transient-growth analysis (`dnsjax.analysis.transient_growth`).

Three test layers, run from the parent process:

* **Host unit checks** (JAX-free): the profile reader / regridder /
  mode selector, exercised by importing the module directly (it keeps
  JAX out of module scope, like the rest of ``dnsjax.analysis``).
* **In-process workers** (one forced-CPU subprocess per system; the
  shared pieces: ``tests/_transient_growth.py``): the
  per-flow ``frozen_profile_flow`` hook reproduces the builtin laminar
  coupling, and the linear step is block-diagonal per Fourier mode (the
  property the propagator build relies on).
* **CLI subprocesses** (``python -m dnsjax.analysis.transient_growth``):
  a per-system smoke run, the wall-BC / folder / snapshot-export
  features, and the published values.

Published values (each a single mode; digits are the paper values,
matched here on the solver's FD-in-``y`` discretization to ~2 %):

* plane-Poiseuille ``Re=1000``, ``(alpha,beta)=(0,2.044)``:
  ``G_max ~ 196`` at ``t ~ 76`` (Reddy & Henningson 1993; Butler &
  Farrell 1992).
* plane-Couette ``Re=1000``, ``(alpha,beta)=(0.035,1.60)``:
  ``G_max ~ 1185`` at ``t ~ 117`` (Butler & Farrell 1992).
* pipe ``Re=3000``, ``m=1``, streamwise-independent (``alpha=0``):
  ``G_max = 649`` at ``t = 147`` (Schmid & Henningson 1994, p. 217 --
  their radius / centreline-velocity / ``Re = U_cl a / nu`` scaling is
  exactly dnsjax's, so ``Re=3000`` is directly comparable).
* Taylor-Couette (annular), two checks: (a) the ``Re_i=100, Re_o=0,
  eta=1/2`` case is linearly **unstable** (positive spectral abscissa --
  the centrifugal Taylor-vortex onset); (b) the ``eta=0.881, Re_i=591,
  Re_o=-2588`` counter-rotating case has its global optimum at
  ``n=10, k=1.994`` with ``G_max = 71.58`` (Maretzke, Hof & Avila 2014,
  table 3, cross-validated against Meseguer 2002). dnsjax ``re1``/``re2``
  equal Maretzke ``Re_i``/``Re_o`` (both use the gap width as the length
  scale); ``G_max`` is a dimensionless energy ratio, so the advective-
  vs-viscous time normalization is immaterial to it.

``--slow`` adds the Orszag (1971) plane-Poiseuille eigenvalue
(``Re=1e4, alpha=1``): the extracted generator's leading eigenvalue has
``Re(lambda) ~ +0.00374`` (a spectrum check -- FD-in-``y`` limits the
precision, so it is matched loosely).

The lab's flows have their own cases, ``lab/tests/test_transient_growth.py``.

Run: ``uv run python tests/test_transient_growth.py`` (``--fast`` skips
the published values; ``--slow`` adds Orszag; ``--system`` /
``--worker`` select one flow).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import run_live
from _transient_growth import (
    EXTRA_ARGS,
    MODULE,
    REPO,
    ccf_profile,
    cli_smoke,
    close,
    load,
    run_tg,
    worker,
    write_ccf_laminar,
    write_profile,
)

sys.stdout.reconfigure(line_buffering=True)


def _write_laminar(system: str, path: Path, **kw) -> None:
    """Write the analytic laminar total profile for *system*."""
    if system in ("plane-couette", "plane-poiseuille"):
        y = np.linspace(1.0, -1.0, 401)
        write_profile(path, y, y if system == "plane-couette" else 1.0 - y**2)
    elif system == "pipe":
        r = np.linspace(1.0, 1e-3, 401)
        write_profile(path, r, 1.0 - r**2)
    else:  # taylor-couette (circular-Couette)
        write_ccf_laminar(path, kw["re1"], kw["re2"], kw["eta"])


def _smoke(system: str, phys_args: list[str], **kw) -> dict:
    return {
        "write_laminar": lambda path: _write_laminar(system, path, **kw),
        "phys_args": phys_args,
    }


#: Per system: the in-process worker's parameters, laminar profile and
#: curl bound (polynomial profiles are FD-exact; the circular-Couette
#: ``B0/r`` curl carries FD error), and the CLI smoke's profile and
#: flags.
CASES: dict[str, dict] = {
    "plane-couette": {
        "worker": {
            "phys": {"re": 100.0},
            "geo": {},
            "profile": lambda flow, d: np.asarray(flow.ys),
            "curl_tol": 1e-12,
        },
        "smoke": _smoke("plane-couette", ["--phys.re", "100"]),
    },
    "plane-poiseuille": {
        "worker": {
            "phys": {"re": 100.0},
            "geo": {},
            "profile": lambda flow, d: 1.0 - np.asarray(flow.ys) ** 2,
            "curl_tol": 1e-12,
        },
        "smoke": _smoke("plane-poiseuille", ["--phys.re", "100"]),
    },
    "pipe": {
        "worker": {
            "phys": {"re": 100.0},
            "geo": {},
            "profile": lambda flow, d: 1.0 - np.asarray(flow.rs) ** 2,
            "curl_tol": 1e-12,
        },
        "smoke": _smoke("pipe", ["--phys.re", "100"]),
    },
    "taylor-couette": {
        "worker": {
            "phys": {"re1": 100.0, "re2": 0.0},
            "geo": {"eta": 0.5},
            "profile": ccf_profile,
            "curl_tol": 3e-4,
        },
        "smoke": _smoke(
            "taylor-couette",
            [
                "--phys.re1",
                "50",
                "--phys.re2",
                "0",
                "--geo.eta",
                "0.5",
                "--geo.lz",
                "6.2832",
            ],
            re1=50.0,
            re2=0.0,
            eta=0.5,
        ),
    },
}


# ── host unit checks (JAX-free) ──────────────────────────────────


def _test_host_units() -> None:
    """Reader / regridder / mode-selector, imported directly."""
    from dnsjax.analysis.transient_growth import (
        _read_profile,
        _regrid_profile,
        _select_modes,
    )

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # descending file -> ascending arrays
        y = np.linspace(1.0, -1.0, 51)
        np.savetxt(p / "ok.txt", np.column_stack([y, 1 - y**2]))
        yy, uu = _read_profile(p / "ok.txt")
        assert yy[0] < yy[-1] and abs(yy[0] + 1) < 1e-12, "not ascending"
        assert abs(uu[0]) < 1e-12, "wall value wrong"

        # ascending (wrong direction) rejected
        np.savetxt(p / "asc.txt", np.column_stack([-y, 1 - y**2]))
        _expect_exit(lambda: _read_profile(p / "asc.txt"), "descending")
        # one column rejected
        np.savetxt(p / "one.txt", y)
        _expect_exit(lambda: _read_profile(p / "one.txt"), "two columns")

    # regrid: identity fast path is bit-exact and flagged
    yc = np.linspace(-1.0, 1.0, 33)
    u_id, interp = _regrid_profile(yc, yc**2, yc, 8, 1e-12)
    assert not interp and np.array_equal(u_id, yc**2), "identity path"
    # regrid: smooth interpolation is accurate
    yf = np.linspace(-1.0, 1.0, 257)
    u_rg, interp = _regrid_profile(yf, np.sin(yf), yc, 8, 1e-12)
    assert interp and np.max(np.abs(u_rg - np.sin(yc))) < 1e-8, "interp"
    # regrid: insufficient coverage rejected
    _expect_exit(
        lambda: _regrid_profile(
            np.linspace(-0.5, 0.5, 20), np.zeros(20), yc, 8, 1e-12
        ),
        "cover",
    )

    # mode selection
    i2, i3 = _select_modes("all", 4, 3)
    assert len(i2) == 4 * 3 - 1 and (0, 0) not in set(zip(i2, i3, strict=True))
    i2, i3 = _select_modes("1,0;2,1", 4, 3)
    assert list(zip(i2, i3, strict=True)) == [(1, 0), (2, 1)]
    _expect_exit(lambda: _select_modes("0,0", 4, 3), "mean")
    _expect_exit(lambda: _select_modes("9,9", 4, 3), "range")
    print("  host units: reader / regridder / mode-selector  OK")


def _expect_exit(fn, needle: str) -> None:
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), f"wrong error: {exc}"
        return
    raise SystemExit(f"expected SystemExit containing {needle!r}")


def _test_jax_free() -> None:
    """`import dnsjax.analysis` must not pull in JAX."""
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, dnsjax.analysis; assert 'jax' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if res.returncode != 0:
        raise SystemExit("dnsjax.analysis import pulled in JAX")
    print("  JAX-free import guarantee  OK")


# ── CLI feature checks ───────────────────────────────────────────


def _test_wall_bc() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        y = np.linspace(1.0, -1.0, 101)
        np.savetxt(p / "bad.txt", np.column_stack([y, 1 - y**2 + 0.1]))
        cmd = [
            sys.executable,
            "-m",
            MODULE,
            "--tg.profile",
            str(p / "bad.txt"),
            "--tg.out_dir",
            str(p),
            "--phys.system",
            "plane-poiseuille",
            "--res.ny",
            "20",
            "--res.nx",
            "4",
            "--res.nz",
            "4",
            "--tg.modes",
            "1,0",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=p)
        assert "wall" in res.stdout.lower() and "FAILED" in res.stdout, (
            "wall-BC violation not rejected"
        )
    print("  wall-BC rejection  OK")


def _test_folder() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        src = p / "in"
        src.mkdir()
        _write_laminar("plane-poiseuille", src / "a.txt")
        y = np.linspace(1.0, -1.0, 201)
        np.savetxt(
            src / "b.txt",
            np.column_stack([y, (1 - y**2) * (1 + 0.05 * np.cos(y))]),
        )
        out = p / "out"
        run_tg(
            src,
            out,
            [
                "--phys.system",
                "plane-poiseuille",
                "--phys.re",
                "1000",
                "--res.ny",
                "20",
                "--res.nx",
                "4",
                "--res.nz",
                "4",
                "--geo.lz",
                "3.073985",
                "--tg.modes",
                "1,0",
                "--tg.t_max",
                "80",
                "--tg.nt",
                "12",
            ],
        )
        assert (out / "a_tg.npz").is_file() and (out / "b_tg.npz").is_file()
        assert (out / "a_tg_summary.txt").is_file()
    print("  folder mode (2 profiles -> 2 bundles)  OK")


def _test_export() -> None:
    """Export a mode's optimum; read it back JAX-free and check it."""
    from dnsjax.analysis import integrate, read_state

    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        prof = p / "lam.txt"
        _write_laminar("plane-poiseuille", prof)
        run_tg(
            prof,
            p,
            [
                "--phys.system",
                "plane-poiseuille",
                "--phys.re",
                "1000",
                "--res.ny",
                "28",
                "--res.nx",
                "4",
                "--res.nz",
                "4",
                "--geo.lz",
                "3.073985",
                "--tg.modes",
                "1,0",
                "--tg.t_max",
                "80",
                "--tg.nt",
                "12",
                "--tg.export_snapshot",
                "1,0",
                "--tg.export_amplitude",
                "1e-4",
            ],
        )
        seed = p / "lam_tg_seed_m1_0.tar"
        assert seed.is_file(), "seed snapshot not written"
        sd = read_state(seed, return_physical=True)
        assert sd.params.phys.system == "plane-poiseuille"
        assert np.all(np.isfinite(sd.physical)), "non-finite seed"
        lx, lz = sd.params.geo.lx, sd.params.geo.lz
        e = float(
            integrate(
                np.sum(np.abs(sd.physical) ** 2, axis=0),
                sd.params,
                sd.physical_coords,
            )
        )
        e /= 2.0 * (2.0 * lx * lz)  # mean energy density
        close(e, 1e-4, 0.02, "seed E'")
    print("  snapshot export + JAX-free read-back  OK")


# ── published values ───────────────────────────────────────────────


def _anchor_pp() -> None:
    print("  plane-Poiseuille Re=1000 (0, 2.044):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _write_laminar("plane-poiseuille", p / "l.txt")
        run_tg(
            p / "l.txt",
            p,
            [
                "--phys.system",
                "plane-poiseuille",
                "--phys.re",
                "1000",
                "--res.ny",
                "80",
                "--res.nx",
                "4",
                "--res.nz",
                "4",
                "--res.fd_order",
                "6",
                "--geo.lz",
                "3.073985",
                "--tg.modes",
                "1,0",
                "--tg.t_max",
                "150",
                "--tg.nt",
                "60",
            ],
        )
        z = load(p, "l")
        close(float(z["G_max"][0]), 196.0, 0.02, "G_max")
        close(float(z["t_opt"][0]), 76.0, 0.04, "t_opt")
        assert float(z["spectral_abscissa"][0]) < 0, "PP should be stable"


def _anchor_pc() -> None:
    print("  plane-Couette Re=1000 (0.035, 1.60):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _write_laminar("plane-couette", p / "l.txt")
        run_tg(
            p / "l.txt",
            p,
            [
                "--phys.system",
                "plane-couette",
                "--phys.re",
                "1000",
                "--res.ny",
                "80",
                "--res.nx",
                "4",
                "--res.nz",
                "4",
                "--res.fd_order",
                "6",
                "--geo.lx",
                "179.5199",
                "--geo.lz",
                "3.92699",
                "--tg.modes",
                "1,1",
                "--tg.t_max",
                "230",
                "--tg.nt",
                "70",
            ],
        )
        z = load(p, "l")
        close(float(z["G_max"][0]), 1185.0, 0.02, "G_max")
        close(float(z["t_opt"][0]), 117.0, 0.04, "t_opt")


def _anchor_pipe() -> None:
    print("  pipe Re=3000, m=1 (Schmid & Henningson 1994, p. 217):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _write_laminar("pipe", p / "l.txt")
        run_tg(
            p / "l.txt",
            p,
            [
                "--phys.system",
                "pipe",
                "--phys.re",
                "3000",
                "--res.nr",
                "72",
                "--res.nz",
                "4",
                "--res.ntheta",
                "4",
                "--res.fd_order",
                "6",
                "--geo.lz",
                "12.566",
                "--tg.modes",
                "1,0",
                "--tg.t_max",
                "300",
                "--tg.nt",
                "60",
            ],
        )
        z = load(p, "l")
        close(float(z["G_max"][0]), 649.0, 0.02, "G_max")
        close(float(z["t_opt"][0]), 147.0, 0.04, "t_opt")
        assert float(z["spectral_abscissa"][0]) < 0, "pipe should be stable"


def _anchor_tc() -> None:
    print("  Taylor-Couette (annular):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # (a) Re_i=100, Re_o=0, eta=1/2 is linearly unstable (the
        # centrifugal Taylor-vortex onset): positive spectral abscissa.
        _write_laminar(
            "taylor-couette", p / "u.txt", re1=100.0, re2=0.0, eta=0.5
        )
        run_tg(
            p / "u.txt",
            p,
            [
                "--phys.system",
                "taylor-couette",
                "--phys.re1",
                "100",
                "--phys.re2",
                "0",
                "--geo.eta",
                "0.5",
                "--res.nr",
                "48",
                "--res.nz",
                "6",
                "--res.ntheta",
                "6",
                "--res.fd_order",
                "6",
                "--geo.lz",
                "6.2832",
                "--tg.modes",
                "0,2",
                "--tg.t_max",
                "20",
                "--tg.nt",
                "20",
            ],
        )
        z = load(p, "u")
        assert float(z["spectral_abscissa"][0]) > 0, (
            "Re_i=100 TC must be linearly unstable"
        )
        print(
            f"    unstable mode spectral abscissa "
            f"{float(z['spectral_abscissa'][0]):.4g} > 0  OK"
        )
        # (b) Maretzke et al. 2014, table 3 (row 1; cross-validated
        # against Meseguer 2002): eta=0.881, Re_i=591, Re_o=-2588 -> the
        # global optimum is at n=10, k=1.994 with G_max=71.58.  The
        # axial length lz is set so mode i3=1 gives the axial
        # wavenumber k=1.994; ntheta=22 makes the azimuthal mode m=10
        # available at i2=10.
        _write_laminar(
            "taylor-couette", p / "m.txt", re1=591.0, re2=-2588.0, eta=0.881
        )
        run_tg(
            p / "m.txt",
            p,
            [
                "--phys.system",
                "taylor-couette",
                "--phys.re1",
                "591",
                "--phys.re2",
                "-2588",
                "--geo.eta",
                "0.881",
                "--res.nr",
                "64",
                "--res.nz",
                "4",
                "--res.ntheta",
                "22",
                "--res.fd_order",
                "8",
                "--geo.lz",
                "3.15108",
                "--tg.modes",
                "10,1",
                "--tg.t_max",
                "40",
                "--tg.nt",
                "40",
            ],
        )
        z = load(p, "m")
        assert float(z["spectral_abscissa"][0]) < 0, "Maretzke case stable"
        close(float(z["G_max"][0]), 71.58, 0.01, "Maretzke G_max")


def _test_wedge_equivalence() -> None:
    """The m0 wedge reproduces the full-circle physics for mode m = m0.

    The TG linear step is FFT-free and block-diagonal per Fourier mode,
    so the reduced operator for physical wavenumber m = 4 is identical
    whether it sits at index i2 = 4 on the full circle (m0 = 1) or at
    i2 = 1 on the quarter-annulus wedge (m0 = 4).  G(t) must agree.
    """
    print("  wedge equivalence (m0 = 4 vs full circle, m = 4):")
    # Re-independent structural check; a co-rotating, linearly stable
    # circular-Couette flow (eta = 0.71, Re_o/Re_i = 0.797, with
    # non-trivial transient growth G_max ~ 1.47) keeps the linear step
    # well conditioned at coarse resolution.
    re1, re2, eta = 1000.0, 796.8476357267952, 0.71
    common = [
        "--phys.system",
        "taylor-couette",
        "--phys.re1",
        str(re1),
        "--phys.re2",
        str(re2),
        "--geo.eta",
        str(eta),
        "--res.nr",
        "48",
        "--res.nz",
        "4",
        "--res.fd_order",
        "6",
        "--geo.lz",
        "0.5",
        "--tg.t_max",
        "60",
        "--tg.nt",
        "30",
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        write_ccf_laminar(p / "w.txt", re1, re2, eta)
        run_tg(
            p / "w.txt",
            p,
            [*common, "--res.ntheta", "10", "--tg.modes", "4,0"],
        )
        full = load(p, "w")
        run_tg(
            p / "w.txt",
            p,
            [
                *common,
                "--geo.m0",
                "4",
                "--res.ntheta",
                "4",
                "--tg.modes",
                "1,0",
            ],
        )
        wedge = load(p, "w")
    gf, gw = float(full["G_max"][0]), float(wedge["G_max"][0])
    assert abs(gf - gw) <= 1e-9 * max(1.0, abs(gf)), (
        f"wedge G_max {gw} != full-circle {gf}"
    )
    # Both resolve the same physical azimuthal wavenumber m = 4, from
    # different stored indices (i2 = 4 full circle, i2 = 1 on the wedge).
    assert float(full["mode_wn2"][0]) == 4.0, full["mode_wn2"]
    assert float(wedge["mode_wn2"][0]) == 4.0, wedge["mode_wn2"]
    assert int(full["mode_i2"][0]) == 4 and int(wedge["mode_i2"][0]) == 1
    print(f"    G_max wedge {gw:.6f} == full circle {gf:.6f} (m=4)  OK")


def _anchor_orszag() -> None:
    """Orszag 1971 unstable OS eigenvalue (spectrum check)."""
    print("  Orszag PP Re=1e4 alpha=1 (leading eigenvalue):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _write_laminar("plane-poiseuille", p / "l.txt")
        run_tg(
            p / "l.txt",
            p,
            [
                "--phys.system",
                "plane-poiseuille",
                "--phys.re",
                "10000",
                "--res.ny",
                "160",
                "--res.nx",
                "4",
                "--res.nz",
                "4",
                "--res.fd_order",
                "8",
                "--geo.lz",
                "6.2832",
                "--geo.lx",
                "6.2832",
                "--tg.modes",
                "0,1",
                "--tg.t_max",
                "5",
                "--tg.nt",
                "6",
            ],
        )
        z = load(p, "l")
        lead = float(z["spectral_abscissa"][0])
        # Orszag: c = 0.23752649 + 0.00373967 i, growth = alpha*c_i.
        close(lead, 0.00373967, 0.15, "Re(lambda_max)")
        assert lead > 0, "unstable OS mode not found"


# ── runner ───────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--legacy-imm",
        action="store_true",
        help="run the published values on the legacy "
        "res.consistent_imm=False path",
    )
    ap.add_argument("--worker", choices=list(CASES), default=None)
    ap.add_argument("--system", choices=list(CASES), default=None)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="structural checks only (skip the published values)",
    )
    ap.add_argument(
        "--slow", action="store_true", help="add the Orszag eigenvalue check"
    )
    args = ap.parse_args()
    if args.legacy_imm:
        EXTRA_ARGS.extend(["--res.consistent_imm", "False"])

    if args.worker:
        worker(args.worker, **CASES[args.worker]["worker"])
        return

    systems = [args.system] if args.system else list(CASES)

    print(
        "transient-growth tests (single CPU device; CLI subprocesses "
        "+ per-system in-process workers).",
        flush=True,
    )

    print("[host units]", flush=True)
    _test_jax_free()
    _test_host_units()

    print("[hooks + block-diagonality]", flush=True)
    for system in systems:
        res = run_live(
            [sys.executable, __file__, "--worker", system], cwd=REPO
        )
        if res.returncode != 0:
            raise SystemExit(f"{system}: worker failed")

    print("[CLI features]", flush=True)
    for system in systems:
        cli_smoke(system, **CASES[system]["smoke"])
    if "taylor-couette" in systems:
        _test_wedge_equivalence()
    _test_wall_bc()
    _test_folder()
    _test_export()

    if not args.fast:
        print("[literature anchors]", flush=True)
        anchors = {
            "plane-poiseuille": _anchor_pp,
            "plane-couette": _anchor_pc,
            "pipe": _anchor_pipe,
            "taylor-couette": _anchor_tc,
        }
        for system in systems:
            anchors[system]()
        if args.slow:
            _anchor_orszag()

    print("ALL PASSED")


if __name__ == "__main__":
    main()
