#!/usr/bin/env python3
r"""Tests for ``scripts/snapshot_import.py`` (external-field converter).

Each flow family is exercised in its own subprocess (the geometry
``fourier`` singleton is built once at import, as in
``scripts/random_field.py``).  Run directly::

    uv run python tests/test_snapshot_import.py            # all families
    uv run python tests/test_snapshot_import.py --system pipe   # one

Per system the checks are:

- **single-mode placement**: a pure cosine along one input axis lands in
  exactly the expected dnsjax spectral slot with the ``norm="forward"``
  amplitude 1/2 -- pins the component basis, the axis mapping (including
  the Taylor-Couette axial/azimuthal swap), and the normalisation.
- **mode order**: the converter's truncated `$k_x$` / `$k_z$` / `$m$`
  axes reproduce the geometry ``fourier`` singleton's wavenumbers.
- **`$u_\pm$` mixing** (pipe / TC): `$u_r = 0 \Rightarrow u_+ = -u_-$`.
- **spectral-input round-trip**: a numpy ``rfft``/``fft`` of the field
  (either real axis, several ``input_norm``) converts to the same state.
- **loadability**: ``save`` then ``load_snapshot`` round-trips the state
  and records the wall-normal grid.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

# Small, distinct resolutions so axis swaps cannot hide.
NX, NY, NZ = 8, 9, 12  # ny is grid points (wall-bounded); periodic uses 10

# system -> (family, extra configure kwargs)
SYSTEMS = {
    "plane-couette": ("cartesian", {}),
    "kolmogorov": ("periodic", {}),
    "pipe": ("pipe", {}),
    "taylor-couette": ("annular", {"re1": 100.0, "re2": 0.0, "eta": 0.5}),
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


def _input_sizes(family: str) -> tuple[int, int, int]:
    ny = 10 if family == "periodic" else NY
    if family == "annular":
        return (NZ, ny, NX)  # streamwise=theta(nz), wn=r(ny), spanwise=ax(nx)
    return (NX, ny, NZ)


def _make_input_spectral(p0, periodic, real_axis, input_norm):
    """numpy spectral input (rfft along ``real_axis``, fft elsewhere)."""
    fourier_axes = {1, 2, 3} if periodic else {1, 3}
    real_ax = 1 if real_axis == "streamwise" else 3
    out = np.fft.rfft(p0, axis=real_ax, norm=input_norm)
    for ax in sorted(fourier_axes - {real_ax}):
        out = np.fft.fft(out, axis=ax, norm=input_norm)
    return out


# ── per-system test body ─────────────────────────────────────────


def _run_one(system: str) -> int:
    family, extra = SYSTEMS[system]
    ny = 10 if family == "periodic" else NY

    import snapshot_import as si

    si.configure_target(
        system,
        NX,
        ny,
        NZ,
        lx=4.0,
        lz=4.0,
        wall_normal_grid=_wall_normal_grid(family, ny),
        re=200.0,
        **extra,
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

    sizes = _input_sizes(family)

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
        # The wall-normal axis is untransformed (Cartesian) or the ky
        # axis (periodic): a field constant in y spans all y rows there
        # but lives only at ky=0 here.
        ysl = slice(None) if family == "cartesian" else _ch_index(ny, 0)

        # u_x along streamwise x (real axis) at mode qx -> kz=0, kx=qx, 0.5
        qx = 2
        s = si.to_spectral_state(_single_mode_field(0, 1, qx))
        exp = jnp.zeros_like(s).at[0, ysl, 0, qx].set(0.5)
        e = _amax(s - exp)
        check("u_x x-mode placement", e < 1e-10, f"{e:.1e}")

        # u_x along spanwise z (full axis) at mode qz -> +/-qz in kz
        qz = 3
        s = si.to_spectral_state(_single_mode_field(0, 3, qz))
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
            # u_x along shearwise y (full axis) at mode qy -> +/-qy in ky
            qy = 2
            s = si.to_spectral_state(_single_mode_field(0, 2, qy))
            exp = (
                jnp.zeros_like(s)
                .at[0, _ch_index(ny, qy), 0, 0]
                .set(0.5)
                .at[0, _ch_index(ny, -qy), 0, 0]
                .set(0.5)
            )
            e = _amax(s - exp)
            check("u_x y-mode placement", e < 1e-10, f"{e:.1e}")

    else:  # pipe / annular: (u_z, u_+, u_-) over (r, m, k_z)
        # u_z along the AXIAL direction (dnsjax real axis k_z) -> [0,:,0,k0]
        if family == "pipe":
            uz_comp, ax_axis, th_comp, th_axis = 0, 1, 2, 3
        else:  # annular / TC: input (u_theta, u_r, u_z); axial is axis 3
            uz_comp, ax_axis, th_comp, th_axis = 2, 3, 0, 1
        k0 = 2
        s = si.to_spectral_state(_single_mode_field(uz_comp, ax_axis, k0))
        exp = jnp.zeros_like(s).at[0, :, 0, k0].set(0.5)
        e = _amax(s - exp)
        check("u_z axial-mode placement", e < 1e-10, f"{e:.1e}")

        # u_theta along the AZIMUTHAL direction (m), u_r = 0 -> u_+ = i
        # u_theta, u_- = -i u_theta: state[1] = -state[2], |u_+| = 0.5.
        m0 = 3
        s = si.to_spectral_state(_single_mode_field(th_comp, th_axis, m0))
        exp_up = (
            jnp.zeros_like(s[1])
            .at[:, _ch_index(NZ, m0), 0]
            .set(0.5j)
            .at[:, _ch_index(NZ, -m0), 0]
            .set(0.5j)
        )
        e = _amax(s[1] - exp_up)
        ok = e < 1e-10 and _amax(s[2] + s[1]) < 1e-12 and _amax(s[0]) < 1e-12
        check("u_theta azimuthal-mode + u_pm mixing", ok, f"{e:.1e}")

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
    rng = np.random.default_rng(0)
    p0 = rng.standard_normal((3, *sizes))
    s_phys = si.to_spectral_state(p0, space="physical")
    periodic = family == "periodic"
    max_err = 0.0
    for real_axis in ("streamwise", "spanwise"):
        for input_norm in ("backward", "forward", "ortho"):
            spec = _make_input_spectral(p0, periodic, real_axis, input_norm)
            s_spec = si.to_spectral_state(
                spec,
                space="spectral",
                real_axis=real_axis,
                input_norm=input_norm,
            )
            max_err = max(max_err, _amax(s_spec - s_phys))
    check("spectral-input round-trip", max_err < 1e-9, f"err {max_err:.1e}")

    # ── loadability ─────────────────────────────────────────────
    from dnsjax.snapshot import load_snapshot, read_metadata

    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "snap.tar")
        si.write_snapshot(s_phys, out, t=1.5, it=7)
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
        check("snapshot save/load round-trip", ok, f"{e:.1e}")

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

    failures = 0
    for system in SYSTEMS:
        print(f"=== {system} ===")
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--system",
                system,
            ],
        )
        failures += proc.returncode != 0
    if failures:
        print(f"\n{failures} system(s) FAILED.")
        sys.exit(1)
    print("\nAll systems passed.")


if __name__ == "__main__":
    main()
