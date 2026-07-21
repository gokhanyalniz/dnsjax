#!/usr/bin/env python3
r"""Tests for ``scripts/snapshot_import.py`` (native-field converter).

Each flow family is exercised in its own subprocess (the geometry
``fourier`` singleton is built once at import, one system per
process).  Run directly::

    uv run python tests/test_snapshot_import.py            # all families
    uv run python tests/test_snapshot_import.py --system pipe   # one

The converter takes input already in dnsjax's **native** component/axis
order (shape ``(3, ny, nz, nx)``; pipe/TC components
``(u_z, u_r, u_theta)`` over ``(r, theta, z_ax)``, every component a
real field), so the checks pin the *native* contract:

- **single-mode placement**: a pure cosine along one native input axis
  lands in exactly the expected dnsjax spectral slot with the
  ``norm="forward"`` amplitude 1/2 -- pins the native component basis,
  the axis mapping, and the normalisation.
- **no swap / no mixing** (pipe / TC): ``u_z`` -> ``state[0]``,
  ``u_r`` -> ``state[1]``, ``u_theta`` -> ``state[2]`` independently
  (pipe and TC are identical; the converter neither swaps axes nor
  mixes components).
- **mode order**: the converter's truncated `$k_x$` / `$k_z$` / `$m$`
  axes reproduce the geometry ``fourier`` singleton's wavenumbers.
- **spectral-input round-trip**: a numpy ``fft`` of the native field
  (real axis always axis 3; several ``input_norm``) converts to the same
  state as the physical path.
- **loadability**: ``convert_field_to_snapshot`` (the documented
  one-shot configure + pack + write entry point) then ``load_snapshot``
  round-trips the state and records the wall-normal grid.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

# Small, distinct resolutions so axis mix-ups cannot hide.
NX, NY, NZ = 8, 9, 12  # ny is grid points (wall-bounded); periodic uses 10

# system -> (family, full public-named configure kwargs).  The
# cylindrical/annular flows use their public aliases (nz axial <->
# internal nx, nr radial <-> ny, ntheta azimuthal <-> nz, lz axial
# length <-> lx), so the internal sizes stay (NX, NY|10, NZ) for every
# system.
SYSTEMS = {
    "plane-couette": (
        "cartesian",
        {"nx": NX, "ny": NY, "nz": NZ, "lx": 4.0, "lz": 4.0, "re": 200.0},
    ),
    "kolmogorov": (
        "periodic",
        {"nx": NX, "ny": 10, "nz": NZ, "lx": 4.0, "lz": 4.0, "re": 200.0},
    ),
    "pipe": (
        "pipe",
        {"nz": NX, "nr": NY, "ntheta": NZ, "lz": 4.0, "re": 200.0},
    ),
    "taylor-couette": (
        "annular",
        {
            "nz": NX,
            "nr": NY,
            "ntheta": NZ,
            "lz": 4.0,
            "re1": 100.0,
            "re2": 0.0,
            "eta": 0.5,
        },
    ),
}


# ── helpers ──────────────────────────────────────────────────────


def _cos_mode(n: int, q: int) -> np.ndarray:
    """Real cosine of integer mode ``q`` on ``n`` points."""
    return np.cos(2 * np.pi * q * np.arange(n) / n)


def _ch_index(n: int, q: int) -> int:
    """Index of wavenumber ``q`` in ``complex_harmonics(n)``."""
    from dnsjax.operators import complex_harmonics

    return int(np.asarray(complex_harmonics(n)).tolist().index(q))


def _wall_normal_grid(family: str, ny: int) -> np.ndarray | None:
    """An ascending wall-normal/radial grid on the canonical domain."""
    if family == "periodic":
        return None
    if family == "cartesian":
        return -np.cos(np.pi * np.arange(ny) / (ny - 1))  # CGL on [-1, 1]
    if family == "pipe":
        return np.linspace(1.0, ny, ny) / ny  # (0, 1], ends at 1
    return np.linspace(1.0, 2.0, ny)  # annular [r1, r2] for eta = 0.5


def _make_input_spectral(p0, periodic, input_norm):
    """numpy native spectral input: full ``fft`` along every Fourier axis
    (real axis is always axis 3), truncated to ``nx // 2`` on the real
    axis (the rest is implied by Hermitian / joint symmetry)."""
    out = np.fft.fft(p0, axis=3, norm=input_norm)  # real axis (nx)
    out = np.fft.fft(out, axis=2, norm=input_norm)  # complex axis (nz)
    if periodic:
        out = np.fft.fft(out, axis=1, norm=input_norm)  # k_y axis (ny)
    return out[..., : p0.shape[3] // 2]


# ── per-system test body ─────────────────────────────────────────


def _run_one(system: str) -> int:
    family, cfg = SYSTEMS[system]
    ny = cfg["ny"] if "ny" in cfg else cfg["nr"]

    import snapshot_import as si

    # Strictness of the public surface: an internal / irrelevant name
    # must be rejected before any singleton is touched.
    bad = "nx" if "nr" in cfg else "ntheta"
    try:
        si.configure_target(system, **{**cfg, bad: 4})
    except ValueError as exc:
        assert bad in str(exc), exc
    else:
        raise AssertionError(f"{system}: {bad!r} accepted")

    si.configure_target(
        system,
        wall_normal_grid=_wall_normal_grid(family, ny),
        **cfg,
    )

    from jax import numpy as jnp

    from dnsjax.operators import complex_harmonics, real_harmonics

    passed = failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}  {detail}")

    # Native physical layout is (3, ny, nz, nx) for every family.
    sizes = (ny, NZ, NX)

    def _single_mode_field(comp: int, axis: int, q: int) -> np.ndarray:
        p = np.zeros((3, *sizes))
        bcast = [None, None, None]
        bcast[axis - 1] = slice(None)
        p[comp] = _cos_mode(sizes[axis - 1], q)[tuple(bcast)]
        return p

    def _amax(a) -> float:
        return float(jnp.max(jnp.abs(a)))

    # ── single-mode placement (per family) ──────────────────────
    if family in ("cartesian", "periodic"):
        # Native axes (axis1, axis2, axis3) = (y, z, x): streamwise x is
        # the real axis (3), spanwise z the complex axis (2), wall-normal
        # y is untransformed (Cartesian) or the k_y axis (periodic).
        ysl = slice(None) if family == "cartesian" else _ch_index(ny, 0)

        # u_x along streamwise x (real axis 3) at mode qx -> kz=0, kx=qx
        qx = 2
        s = si.to_spectral_state(_single_mode_field(0, 3, qx))
        exp = jnp.zeros_like(s).at[0, ysl, 0, qx].set(0.5)
        e = _amax(s - exp)
        check("u_x x-mode placement", e < 1e-10, f"{e:.1e}")

        # u_x along spanwise z (complex axis 2) at mode qz -> +/-qz in kz
        qz = 3
        s = si.to_spectral_state(_single_mode_field(0, 2, qz))
        exp = (
            jnp.zeros_like(s)
            .at[0, ysl, _ch_index(NZ, qz), 0]
            .set(0.5)
            .at[0, ysl, _ch_index(NZ, -qz), 0]
            .set(0.5)
        )
        e = _amax(s - exp)
        check("u_x z-mode placement", e < 1e-10, f"{e:.1e}")

        if family == "periodic":
            # u_x along shearwise y (k_y axis 1) at mode qy -> +/-qy in ky
            qy = 2
            s = si.to_spectral_state(_single_mode_field(0, 1, qy))
            exp = (
                jnp.zeros_like(s)
                .at[0, _ch_index(ny, qy), 0, 0]
                .set(0.5)
                .at[0, _ch_index(ny, -qy), 0, 0]
                .set(0.5)
            )
            e = _amax(s - exp)
            check("u_x y-mode placement", e < 1e-10, f"{e:.1e}")

    else:  # pipe / annular: native (u_z, u_r, u_theta) over (r, th, z_ax)
        # Identical for pipe and TC: u_z is component 0 along the axial
        # real axis (3); u_r/u_theta are components 1/2 along the
        # azimuthal complex axis (2).  No swap, no component mixing.
        k0 = 2
        s = si.to_spectral_state(_single_mode_field(0, 3, k0))
        exp = jnp.zeros_like(s).at[0, :, 0, k0].set(0.5)
        e = _amax(s - exp)
        check("u_z axial-mode placement", e < 1e-10, f"{e:.1e}")

        m0 = 3
        # u_r (component 1) along azimuthal m -> state[1]; others zero.
        s = si.to_spectral_state(_single_mode_field(1, 2, m0))
        exp_ur = (
            jnp.zeros_like(s[1])
            .at[:, _ch_index(NZ, m0), 0]
            .set(0.5)
            .at[:, _ch_index(NZ, -m0), 0]
            .set(0.5)
        )
        e = _amax(s[1] - exp_ur)
        ok = e < 1e-10 and _amax(s[0]) < 1e-12 and _amax(s[2]) < 1e-12
        check("u_r azimuthal placement (no mixing)", ok, f"{e:.1e}")

        # u_theta (component 2) along azimuthal m -> state[2]; rest zero.
        s = si.to_spectral_state(_single_mode_field(2, 2, m0))
        exp_uth = (
            jnp.zeros_like(s[2])
            .at[:, _ch_index(NZ, m0), 0]
            .set(0.5)
            .at[:, _ch_index(NZ, -m0), 0]
            .set(0.5)
        )
        e = _amax(s[2] - exp_uth)
        ok = e < 1e-10 and _amax(s[0]) < 1e-12 and _amax(s[1]) < 1e-12
        check("u_theta azimuthal placement (no mixing)", ok, f"{e:.1e}")

    # ── mode order vs the fourier singleton ─────────────────────
    fourier = _import_fourier(family)
    rh = np.asarray(real_harmonics(NX))
    ch = np.asarray(complex_harmonics(NZ))
    if family in ("cartesian", "periodic"):
        kx_int = np.rint(np.asarray(fourier.kx).ravel() / (2 * np.pi / 4.0))
        kz_int = np.rint(np.asarray(fourier.kz).ravel() / (2 * np.pi / 4.0))
        ok = np.array_equal(kx_int, rh) and np.array_equal(kz_int, ch)
        check("kx/kz mode order matches fourier", ok)
    else:
        # axial k_z is the real axis (period lx=4); azimuthal m is integer.
        kz_int = np.rint(np.asarray(fourier.kz).ravel() / (2 * np.pi / 4.0))
        m_int = np.asarray(fourier.m).ravel()
        ok = np.array_equal(kz_int, rh) and np.array_equal(m_int, ch)
        check("axial-kz/azimuthal-m order matches fourier", ok)

    # ── spectral-input round-trip ───────────────────────────────
    # Physical input is real in every family (native components).
    rng = np.random.default_rng(0)
    p0 = rng.standard_normal((3, *sizes))
    s_phys = si.to_spectral_state(p0, space="physical")
    periodic = family == "periodic"
    max_err = 0.0
    for input_norm in ("backward", "forward", "ortho"):
        spec = _make_input_spectral(p0, periodic, input_norm)
        s_spec = si.to_spectral_state(
            spec, space="spectral", input_norm=input_norm
        )
        max_err = max(max_err, _amax(s_spec - s_phys))
    check("spectral-input round-trip", max_err < 1e-9, f"err {max_err:.1e}")

    # ── loadability (via the one-shot entry point) ──────────────
    from dnsjax.snapshot import load_snapshot, read_metadata

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "snap.tar")
        # The documented one-call API (configure + pack + write); the
        # target was configured identically above, so the re-configure
        # is idempotent and the packed state must equal s_phys.
        si.convert_field_to_snapshot(
            p0,
            out,
            system=system,
            space="physical",
            t=1.5,
            it=7,
            wall_normal_grid=_wall_normal_grid(family, ny),
            **cfg,
        )
        state2, t2, it2 = load_snapshot(out)
        meta = read_metadata(Path(out))
        e = _amax(state2 - s_phys)
        grid_ok = (
            meta["wall_normal_grid"] is None
            if periodic
            else len(meta["wall_normal_grid"]) == ny
        )
        ok = (
            tuple(state2.shape) == tuple(s_phys.shape)
            and e < 1e-12
            and t2 == 1.5
            and it2 == 7
            and meta["system"] == system
            and grid_ok
        )
        check("one-shot convert + save/load round-trip", ok, f"{e:.1e}")

    print(f"\n[{system}] {passed} passed, {failed} failed.")
    return 1 if failed else 0


def _import_fourier(family: str):
    if family == "cartesian":
        from dnsjax.geometries.wall_bounded.cartesian import fourier
    elif family == "periodic":
        from dnsjax.geometries.triply_periodic.triply_periodic import fourier
    elif family == "pipe":
        from dnsjax.geometries.wall_bounded.cylindrical import fourier
    else:
        from dnsjax.geometries.wall_bounded.annular import fourier
    return fourier


# ── driver ───────────────────────────────────────────────────────


def main() -> None:
    if "--system" in sys.argv:
        system = sys.argv[sys.argv.index("--system") + 1]
        sys.exit(_run_one(system))

    print(
        "snapshot_import native-contract tests: offline, "
        "device-independent (native layout/normalisation checks; no "
        "GPU path).",
        flush=True,
    )
    # ``report`` repeats each failure after the counts (see _live),
    # quoting the worker's own first ``FAIL:`` line -- the parent
    # otherwise only knows the exit code.
    results: list[tuple[str, str | None]] = []
    for system in SYSTEMS:
        print(f"=== {system} ===")
        proc = run_live(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--system",
                system,
            ],
            timeout=600,
        )
        if proc.returncode == 0:
            results.append((system, None))
            continue
        inner = next(
            (
                ln.strip()
                for ln in proc.stdout.splitlines()
                if ln.strip().startswith("FAIL:")
            ),
            "",
        )
        results.append(
            (
                system,
                f"worker exit {proc.returncode}"
                + (f" ({inner})" if inner else ""),
            )
        )

    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))


if __name__ == "__main__":
    main()
