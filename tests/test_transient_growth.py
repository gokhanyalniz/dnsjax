r"""Transient-growth analysis (`dnsjax.analysis.transient_growth`).

Three test layers, run from the parent process:

* **Host unit checks** (JAX-free): the profile reader / regridder /
  mode selector, exercised by importing the module directly (it keeps
  JAX out of module scope, like the rest of ``dnsjax.analysis``).
* **In-process workers** (one forced-CPU subprocess per system): the
  per-flow ``frozen_profile_flow`` hook reproduces the builtin laminar
  coupling, and the linear step is block-diagonal per Fourier mode (the
  property the propagator build relies on).
* **CLI subprocesses** (``python -m dnsjax.analysis.transient_growth``):
  a per-system smoke run, the wall-BC / folder / snapshot-export
  features, and the literature anchors.

Literature anchors (each a single mode; digits are the paper values,
matched here on the solver's FD-in-``y`` discretisation to ~2 %):

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
  vs-viscous time normalisation is immaterial to it.

``--slow`` adds the Orszag (1971) plane-Poiseuille eigenvalue
(``Re=1e4, alpha=1``): the extracted generator's leading eigenvalue has
``Re(lambda) ~ +0.00374`` (a spectrum check -- FD-in-``y`` limits the
precision, so it is matched loosely).

Run: ``uv run python tests/test_transient_growth.py`` (``--fast`` skips
the anchors; ``--slow`` adds Orszag; ``--system`` / ``--worker`` select
one flow).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import run_live

sys.stdout.reconfigure(line_buffering=True)

REPO = Path(__file__).resolve().parent.parent
MODULE = "dnsjax.analysis.transient_growth"
SYSTEMS = [
    "plane-couette",
    "plane-poiseuille",
    "pipe",
    "taylor-couette",
    "quasi-keplerian",
]


# ── laminar profile files (top-wall-first / descending) ──────────


def _ccf(re1: float, re2: float, eta: float) -> tuple[float, float]:
    """Circular-Couette ``A0``, ``B0`` (``U_theta = A0 r + B0/r``)."""
    re_ref = re1 if re1 > 0 else re2
    a0 = (re2 - eta * re1) / ((1 + eta) * re_ref)
    b0 = eta * (re1 - eta * re2) / ((1 + eta) * (1 - eta) ** 2 * re_ref)
    return a0, b0


def _qk_re2(re1: float, eta: float, r_omega: float) -> float:
    """Quasi-Keplerian derived outer Reynolds number Re_o."""
    return re1 * (1 - eta + r_omega) / (eta * r_omega - (1 - eta))


def _write_laminar(system: str, path: Path, **kw) -> None:
    """Write the analytic laminar total profile for *system*."""
    if system in ("plane-couette", "plane-poiseuille"):
        y = np.linspace(1.0, -1.0, 401)
        u = y if system == "plane-couette" else 1.0 - y**2
        np.savetxt(path, np.column_stack([y, u]))
    elif system == "pipe":
        r = np.linspace(1.0, 1e-3, 401)
        np.savetxt(path, np.column_stack([r, 1.0 - r**2]))
    else:  # taylor-couette / quasi-keplerian (circular-Couette)
        eta = kw["eta"]
        re2 = (
            _qk_re2(kw["re1"], eta, kw["r_omega"])
            if system == "quasi-keplerian"
            else kw["re2"]
        )
        a0, b0 = _ccf(kw["re1"], re2, eta)
        r = np.linspace(1.0 / (1 - eta), eta / (1 - eta), 401)  # r2 -> r1
        np.savetxt(path, np.column_stack([r, a0 * r + b0 / r]))


# ── CLI driver ───────────────────────────────────────────────────


#: Appended to every anchor run by ``--consistent-imm``, which selects
#: the reconstruction scheme in *every* wall-bounded geometry: the
#: anchors then check the reformulated propagator against the same
#: published digits -- the strongest available statement that the
#: reformulation did not perturb the linear physics (the eigenvalue
#: content of the operator these anchors measure is exactly what an
#: Orr-Sommerfeld/Squire check would test).  Measured agreement with
#: the ungated propagator: 4-6 significant figures on every anchor.
EXTRA_ARGS: list[str] = []


def _run_tg(profile: Path, out_dir: Path, args: list[str]) -> str:
    """Invoke the CLI; raise on failure.

    Runs in the profile's directory (always a temp dir here), not the
    repo root: the CLI loads a ``./parameters.toml`` when present, and
    the repo's flow-specific one must not leak into these runs.
    """
    cmd = [
        sys.executable,
        "-m",
        MODULE,
        "--tg.profile",
        str(profile),
        "--tg.out_dir",
        str(out_dir),
        *args,
        *EXTRA_ARGS,
    ]
    res = run_live(cmd, cwd=profile.parent)
    if res.returncode != 0 or "FAILED" in res.stdout:
        raise SystemExit(f"tg run failed: {' '.join(args)}")
    return res.stdout


def _load(out_dir: Path, stem: str) -> dict:
    return dict(np.load(out_dir / f"{stem}_tg.npz", allow_pickle=True))


def _close(got: float, want: float, rtol: float, name: str) -> None:
    if abs(got - want) > rtol * abs(want):
        raise SystemExit(
            f"{name}: got {got:.6g}, want {want:.6g} (rtol {rtol})"
        )
    print(f"    {name}: {got:.6g} (want ~{want:.6g})  OK")


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


# ── in-process worker: hooks + block-diagonality ─────────────────


def _worker(system: str) -> None:
    """Per-system: hook == builtin coupling, and block-diagonality."""
    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        Parameters,
        derived_params,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    configure_jax_platform("cpu")
    phys = {"system": system, "re": 100.0}
    geo: dict = {}
    if system == "taylor-couette":
        phys = {"system": system, "re1": 100.0, "re2": 0.0}
        geo = {"eta": 0.5}
    elif system == "quasi-keplerian":
        phys = {"system": system, "re1": 100.0, "r_omega": -1.2}
        geo = {"eta": 0.71}
    update_parameters(
        Parameters(
            phys=phys,
            geo=geo,
            res={
                "nx": 8,
                "ny": 17,
                "nz": 8,
                "fd_order": 4,
                "double_precision": True,
            },
            step={
                "dt": 0.01,
                "implicitness": 1.0,
                "implicit_mean_coupling": False,
                "corrector_tolerance": 1e-12,
                "max_corrector_iterations": 60,
            },
        )
    )
    validate_parameters()
    padded_res.set_padded_resolution(params)

    import importlib

    import jax.numpy as jnp

    from dnsjax.sharding import sharding
    from dnsjax.timestep import make_stepper

    fmod = importlib.import_module(
        f"dnsjax.flows.wall_bounded.{system.replace('-', '_')}"
    )
    geo_name = {
        "plane-couette": "cartesian",
        "plane-poiseuille": "cartesian",
        "pipe": "cylindrical",
        "taylor-couette": "annular",
        "quasi-keplerian": "annular",
    }[system]
    gmod = importlib.import_module(
        f"dnsjax.geometries.wall_bounded.{geo_name}"
    )
    flow = fmod.flow

    # Laminar total profile on the code grid (relevant component).
    if system == "plane-couette":
        prof = np.asarray(flow.ys)
    elif system == "plane-poiseuille":
        prof = 1.0 - np.asarray(flow.ys) ** 2
    elif system == "pipe":
        prof = 1.0 - np.asarray(flow.rs) ** 2
    else:
        a0, b0 = derived_params.ccf_A, derived_params.ccf_B
        rs = np.asarray(flow.rs)
        prof = a0 * rs + b0 / rs

    frozen = fmod.frozen_profile_flow(
        jnp.asarray(prof, dtype=sharding.float_type)
    )
    db = float(np.max(np.abs(np.asarray(frozen.base_flow - flow.base_flow))))
    dc = float(
        np.max(np.abs(np.asarray(frozen.curl_base_flow - flow.curl_base_flow)))
    )
    # Polynomial profiles are FD-exact; the circular-Couette B0/r curl
    # (taylor-couette / quasi-keplerian) carries FD error.
    ctol = 3e-4 if system in ("taylor-couette", "quasi-keplerian") else 1e-12
    assert db < 1e-12, f"{system}: base_flow hook != builtin ({db:.2e})"
    assert dc < ctol, f"{system}: curl hook != builtin ({dc:.2e})"

    # Block-diagonality: a basis vector at all modes vs one mode gives
    # the identical column at that mode.  Everything here stays in the
    # geometry's solver basis (the property is about modes, not
    # components, and both runs use the same stepper); the physical
    # wrapper lives in ``transient_growth._linear_step``.
    raw = make_stepper(
        gmod._l_bf, gmod._predict, gmod._correct, gmod._norm, None, None
    )
    pfc = raw[2]
    ny = params.res.ny
    n2, n3 = sharding.spec_shape[1], sharding.spec_shape[2]
    i2, i3 = 1, 0

    def _state(all_modes: bool):
        st = jnp.zeros(
            (3, ny, n2, n3),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
        if all_modes:
            st = st.at[0, ny // 2].set(1.0).at[:, :, 0, 0].set(0.0)
        else:
            st = st.at[0, ny // 2, i2, i3].set(1.0)
        return st

    col_all = np.asarray(pfc(_state(True), gmod.fourier, flow)[0])[
        :, :, i2, i3
    ]
    col_one = np.asarray(pfc(_state(False), gmod.fourier, flow)[0])[
        :, :, i2, i3
    ]
    bd = float(np.max(np.abs(col_all - col_one)))
    assert bd < 1e-9, f"{system}: not block-diagonal ({bd:.2e})"
    print(
        f"  {system}: hook==builtin (base {db:.1e}, curl {dc:.1e}); "
        f"block-diagonal ({bd:.1e})  OK"
    )


# ── CLI feature checks ───────────────────────────────────────────


def _test_cli_smoke(system: str) -> None:
    """A per-system tiny run: G(0)=1, stable, forced overrides applied."""
    cyl_annular = system in ("pipe", "taylor-couette", "quasi-keplerian")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        prof = p / "lam.txt"
        res_flags = (
            ["--res.nr", "20", "--res.nz", "8", "--res.ntheta", "8"]
            if cyl_annular
            else ["--res.ny", "20", "--res.nx", "8", "--res.nz", "8"]
        )
        extra = [
            "--phys.system",
            system,
            *res_flags,
            "--tg.t_max",
            "40",
            "--tg.nt",
            "12",
            "--tg.modes",
            "1,1",
        ]
        if system == "taylor-couette":
            _write_laminar(system, prof, re1=50.0, re2=0.0, eta=0.5)
            extra += [
                "--phys.re1",
                "50",
                "--phys.re2",
                "0",
                "--geo.eta",
                "0.5",
                "--geo.lz",
                "6.2832",
            ]
        elif system == "quasi-keplerian":
            _write_laminar(system, prof, re1=50.0, r_omega=-1.2, eta=0.71)
            extra += [
                "--phys.re1",
                "50",
                "--phys.r_omega",
                "-1.2",
                "--geo.eta",
                "0.71",
                "--geo.lz",
                "6.2832",
            ]
        else:
            _write_laminar(system, prof)
            extra += ["--phys.re", "100"]
        _run_tg(prof, p, extra)
        z = _load(p, "lam")
        assert abs(z["G"][0, 0] - 1.0) < 1e-9, "G(0) != 1"
        assert np.all(np.isfinite(z["G"])), "non-finite G"
        assert float(z["extraction_residual"][0]) < 1e-9, "eig residual"
        assert float(z["G_max"][0]) >= 1.0 - 1e-9, "G_max < 1"
        pj = json.loads(str(z["params_json"]))
        assert pj["phys"]["u_grid"] == 0.0, "u_grid not zeroed"
        assert pj["step"]["implicitness"] == 1.0, "theta != 1"
    print(f"  {system}: CLI smoke (G(0)=1, u_grid=0, theta=1)  OK")


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
        _run_tg(
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
        _run_tg(
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
        _close(e, 1e-4, 0.02, "seed E'")
    print("  snapshot export + JAX-free read-back  OK")


# ── literature anchors ───────────────────────────────────────────


def _anchor_pp() -> None:
    print("  plane-Poiseuille Re=1000 (0, 2.044):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _write_laminar("plane-poiseuille", p / "l.txt")
        _run_tg(
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
        z = _load(p, "l")
        _close(float(z["G_max"][0]), 196.0, 0.02, "G_max")
        _close(float(z["t_opt"][0]), 76.0, 0.04, "t_opt")
        assert float(z["spectral_abscissa"][0]) < 0, "PP should be stable"


def _anchor_pc() -> None:
    print("  plane-Couette Re=1000 (0.035, 1.60):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _write_laminar("plane-couette", p / "l.txt")
        _run_tg(
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
        z = _load(p, "l")
        _close(float(z["G_max"][0]), 1185.0, 0.02, "G_max")
        _close(float(z["t_opt"][0]), 117.0, 0.04, "t_opt")


def _anchor_pipe() -> None:
    print("  pipe Re=3000, m=1 (Schmid & Henningson 1994, p. 217):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        _write_laminar("pipe", p / "l.txt")
        _run_tg(
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
        z = _load(p, "l")
        _close(float(z["G_max"][0]), 649.0, 0.02, "G_max")
        _close(float(z["t_opt"][0]), 147.0, 0.04, "t_opt")
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
        _run_tg(
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
        z = _load(p, "u")
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
        _run_tg(
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
        z = _load(p, "m")
        assert float(z["spectral_abscissa"][0]) < 0, "Maretzke case stable"
        _close(float(z["G_max"][0]), 71.58, 0.01, "Maretzke G_max")


def _anchor_qk() -> None:
    print("  quasi-Keplerian (annular):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # Axially-periodic quasi-Keplerian optimal transient growth,
        # Shi et al., Phys. Fluids 29, 044107 (2017), Table III case I:
        # eta = 0.71, R_Omega = -1.2, Re_i = 1e4, axial wavenumber
        # k_z = 0, azimuthal m = 4 -> G_opt = 13.04 at t_opt/tau_d = 27.
        # The regime is linearly stable (negative spectral abscissa); the
        # growth is purely non-modal.  Code time == tau_d, so t_opt is in
        # the paper's units.  m = 4 sits at index i2 = 4 for ntheta = 10.
        _write_laminar(
            "quasi-keplerian", p / "k.txt", re1=1.0e4, r_omega=-1.2, eta=0.71
        )
        _run_tg(
            p / "k.txt",
            p,
            [
                "--phys.system",
                "quasi-keplerian",
                "--phys.re1",
                "10000",
                "--phys.r_omega",
                "-1.2",
                "--geo.eta",
                "0.71",
                "--res.nr",
                "128",
                "--res.nz",
                "4",
                "--res.ntheta",
                "10",
                "--res.fd_order",
                "8",
                "--geo.lz",
                "0.5",
                "--tg.modes",
                "4,0",
                "--tg.t_max",
                "60",
                "--tg.nt",
                "60",
            ],
        )
        z = _load(p, "k")
        assert float(z["spectral_abscissa"][0]) < 0, "QK must be stable"
        _close(float(z["G_max"][0]), 13.04, 0.01, "QK G_opt")
        _close(float(z["t_opt"][0]), 27.0, 0.02, "QK t_opt")


def _test_wedge_equivalence() -> None:
    """The m0 wedge reproduces the full-circle physics for mode m = m0.

    The TG linear step is FFT-free and block-diagonal per Fourier mode,
    so the reduced operator for physical wavenumber m = 4 is identical
    whether it sits at index i2 = 4 on the full circle (m0 = 1) or at
    i2 = 1 on the quarter-annulus wedge (m0 = 4).  G(t) must agree.
    """
    print("  wedge equivalence (m0 = 4 vs full circle, m = 4):")
    # Re-independent structural check; a mild Re_i (with non-trivial
    # transient growth G_max ~ 1.47) keeps the linear step well
    # conditioned at coarse resolution.
    common = [
        "--phys.system",
        "quasi-keplerian",
        "--phys.re1",
        "1000",
        "--phys.r_omega",
        "-1.2",
        "--geo.eta",
        "0.71",
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
        _write_laminar(
            "quasi-keplerian", p / "w.txt", re1=1000.0, r_omega=-1.2, eta=0.71
        )
        _run_tg(
            p / "w.txt",
            p,
            [*common, "--res.ntheta", "10", "--tg.modes", "4,0"],
        )
        full = _load(p, "w")
        _run_tg(
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
        wedge = _load(p, "w")
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
        _run_tg(
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
        z = _load(p, "l")
        lead = float(z["spectral_abscissa"][0])
        # Orszag: c = 0.23752649 + 0.00373967 i, growth = alpha*c_i.
        _close(lead, 0.00373967, 0.15, "Re(lambda_max)")
        assert lead > 0, "unstable OS mode not found"


# ── runner ───────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--consistent-imm",
        action="store_true",
        help="run the anchors with res.consistent_imm on",
    )
    ap.add_argument("--worker", choices=SYSTEMS, default=None)
    ap.add_argument("--system", choices=SYSTEMS, default=None)
    ap.add_argument(
        "--fast",
        action="store_true",
        help="structural checks only (skip anchors)",
    )
    ap.add_argument(
        "--slow", action="store_true", help="add the Orszag eigenvalue check"
    )
    args = ap.parse_args()
    if args.consistent_imm:
        EXTRA_ARGS.extend(["--res.consistent_imm", "True"])

    if args.worker:
        _worker(args.worker)
        return

    systems = [args.system] if args.system else SYSTEMS

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
        _test_cli_smoke(system)
    if "quasi-keplerian" in systems:
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
            "quasi-keplerian": _anchor_qk,
        }
        for system in systems:
            anchors[system]()
        if args.slow:
            _anchor_orszag()

    print("ALL PASSED")


if __name__ == "__main__":
    main()
