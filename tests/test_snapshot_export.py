#!/usr/bin/env python3
r"""Tests for the JAX-free snapshot analysis API (``dnsjax.analysis``).

Validates :func:`dnsjax.analysis.read_state` (spectral<->physical
transforms, the snapshot-native layout / component basis, the
coordinate tuples, component and wall-normal subsetting, and
object-like params/stats access) plus the field operators: the
public ``derivative``/``gradient`` wrappers (axis wiring, ``ik``
scaling, plain ``D1``, the pipe parity requirement), ``curl`` against
the solver's ``_curl_fn`` node-for-node, the **viscoelastic**
conformation recipes (components 3..8) against the solver's
``_spin_to_phys_combos``, and ``integrate`` -- all **without
importing JAX in the test process**, so the import-time JAX-free
guarantee is asserted directly.

Fixtures are generated in JAX subprocesses (forced CPU devices, one per
flow system, no MPI), mirroring ``tests/test_localized_rolls.py``: each
writes ``state.tar`` (a random divergence-free IC) and, for the
wall-bounded families, ``omega.tar`` -- dnsjax's own ``_curl_fn`` of
that state, saved raw in the ``(.,r,θ)`` / ``(x,y,z)`` basis -- so the
analysis ``curl`` can be checked against the solver node-for-node.

Run directly::

    uv run python tests/test_snapshot_export.py            # all systems
    uv run python tests/test_snapshot_export.py --system pipe
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

NX, NY, NZ = 8, 24, 8
LX, LZ = 5.0, 5.0
RE = 100.0

# (system, family); families exercise both basis paths + parity
# (pipe) + the 9-component conformation schema (viscoelastic-dean).
SYSTEMS = [
    ("plane-couette", "cartesian"),
    ("pipe", "cylindrical"),
    ("taylor-couette", "annular"),
    ("viscoelastic-dean", "annular"),
    ("kolmogorov", "triply_periodic"),
]

CURL_TOL = 1e-10  # vs dnsjax _curl_fn (expect machine precision)
DIV_TOL = 1e-8  # divergence of the divergence-free modes (k != 0)
RT_TOL = 1e-10  # transform round-trip (all families)
VOL_TOL = 1e-10  # integrate(ones) vs analytic volume


# ── fixture generation (JAX subprocess; forced CPU) ──────────────


def _generate(system: str, outdir: str) -> None:
    """Write ``state.tar`` (+ ``omega.tar``) for *system*.

    Runs inside the ``--gen`` subprocess; this is the only code path
    that imports JAX.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    phys: dict = {"system": system}
    geo: dict = {"lx": LX, "lz": LZ}
    init: dict = {}
    if system == "taylor-couette":
        phys.update(re1=RE, re2=-RE)  # re derives from re1
        geo["eta"] = 0.5
    elif system == "viscoelastic-dean":
        phys.update(wi=5.0, el=5.0)  # re = wi/el = 1 (derived)
        geo["eta"] = 0.5
        init["random_conformation_amplitude"] = 10.0
    else:
        phys["re"] = RE

    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1, "platform": "cpu"},
            phys=phys,
            geo=geo,
            res={
                "nx": NX,
                "ny": NY,
                "nz": NZ,
                "fd_order": 4,
                "double_precision": True,
            },
            init=init,
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)
    validate_parameters()

    import jax.numpy as jnp

    from dnsjax.random_field import generate_random_state
    from dnsjax.snapshot import save_snapshot

    state = generate_random_state(0.2, 0.4, 1)

    if system in ("plane-couette", "plane-poiseuille"):
        from dnsjax.flows.wall_bounded.plane_couette import flow, get_stats
        from dnsjax.geometries.wall_bounded.cartesian import _curl_fn, fourier

        omega = _curl_fn(state, fourier, flow)
    elif system == "pipe":
        from dnsjax.flows.wall_bounded.pipe import flow, get_stats
        from dnsjax.geometries.wall_bounded.cylindrical import (
            _curl_fn,
            fourier,
        )

        uz, up, um = state[0], state[1], state[2]
        srthz = jnp.array([uz, (up + um) / 2, -1j * (up - um) / 2])
        omega = _curl_fn(srthz, fourier, flow)
    elif system == "taylor-couette":
        from dnsjax.flows.wall_bounded.taylor_couette import flow, get_stats
        from dnsjax.geometries.wall_bounded.annular import _curl_fn, fourier

        uz, up, um = state[0], state[1], state[2]
        srthz = jnp.array([uz, (up + um) / 2, -1j * (up - um) / 2])
        omega = _curl_fn(srthz, fourier, flow)
    elif system == "viscoelastic-dean":
        # 9-component state; the velocity curl path is the annular one
        # (pinned by the taylor-couette case).  The ground truth here
        # is the *conformation* conversion: the solver's spin -> phys
        # combos on the stored chunks, saved in the analysis component
        # order 3..8 (c_zz, c_rz, c_thz, c_rr, c_thth, c_rth).
        from dnsjax.flows.wall_bounded.viscoelastic_dean import get_stats
        from dnsjax.geometries.wall_bounded.annular_viscoelastic import (
            _spin_to_phys_combos,
        )

        c_rr, c_thth, c_rth, c_rz, c_thz, c_zz = _spin_to_phys_combos(
            state[3], state[4], state[5], state[6], state[7], state[8]
        )
        conf_true = jnp.stack([c_zz, c_rz, c_thz, c_rr, c_thth, c_rth])
        # Solver state axes (r, m, k_ax) -> snapshot-native (r, k_ax, m)
        # (the walled on-disk layout read_state returns).
        np.save(
            os.path.join(outdir, "conf_true.npy"),
            np.asarray(conf_true).transpose(0, 1, 3, 2),
        )
        omega = None
    else:  # triply-periodic: curl is pure Fourier, no ground-truth dump
        from dnsjax.flows.triply_periodic.monochromatic import get_stats

        omega = None

    stats = {k: float(v) for k, v in get_stats(state).items()}
    save_snapshot(
        state, 0.0, 0, os.path.join(outdir, "state.tar"), stats=stats, isnap=0
    )
    if omega is not None:
        save_snapshot(omega, 0.0, 0, os.path.join(outdir, "omega.tar"))


def _gen_subprocess(system: str, outdir: str) -> None:
    """Spawn the ``--gen`` generation subprocess (forced 1 CPU device)."""
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    env["NPROC"] = "1"
    subprocess.run(
        [sys.executable, __file__, "--gen", system, outdir],
        check=True,
        env=env,
    )


# ── checks (NumPy only; JAX must NOT be imported here) ────────────


def _relerr(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / (denom if denom else 1.0))


def _check_system(system: str, family: str, outdir: str) -> None:
    from dnsjax.analysis import (
        _core,
        curl,
        derivative,
        divergence,
        gradient,
        integrate,
        read_state,
        to_physical,
        to_spectral,
    )

    d = Path(outdir)
    st = read_state(d / "state.tar", return_spectral=True)

    # physical & spectral present, right arity, real / finite physical
    assert st.physical is not None and len(st.physical) == 3
    assert st.spectral is not None and len(st.spectral) == 3
    for comp in st.physical:
        a = np.asarray(comp)
        assert np.isrealobj(a) and np.all(np.isfinite(a)), system
    # coordinate tuples are axis-ordered and match the data shape
    shape = np.asarray(st.spectral[0]).shape
    assert tuple(len(c) for c in st.spectral_coords) == shape, system
    pshape = np.asarray(st.physical[0]).shape
    assert tuple(len(c) for c in st.physical_coords) == pshape, system

    # params / stats object-like access (viscoelastic re = wi/el = 1,
    # rehydrated by params_namespace from the derived internal field)
    exp_re = 1.0 if system == "viscoelastic-dean" else RE
    assert float(st.params.phys.re) == exp_re, system
    assert st.params.phys.system == system, system
    assert st.stats is not None, system
    _ = st.stats[next(iter(st.stats.keys()))]  # item access works

    # transform round-trip: machine-precision exact for every family.
    # The transform requires u_theta Hermitian on the real (k_z) axis;
    # the cyl/annular IC draws the k_z=0 plane with u_r, u_theta Hermitian
    # (u_+/u_- conjugate partners), so the returned (u_z, u_r, u_theta)
    # basis round-trips like cartesian/periodic.
    back = to_spectral(to_physical(st.spectral, st.params), st.params)
    rt = max(_relerr(back[i], st.spectral[i]) for i in range(3))
    assert rt < RT_TOL, f"{system}: roundtrip {rt:.2e}"

    # curl vs dnsjax _curl_fn (raw chunks, same basis/order)
    omega_path = d / "omega.tar"
    if omega_path.exists():
        my_omega = curl(st.spectral, st.params, st.spectral_coords)
        meta = _core.read_meta(omega_path)
        raw = _core.read_chunks(omega_path, meta, [0, 1, 2])
        cerr = max(_relerr(my_omega[i], raw[i]) for i in range(3))
        assert cerr < CURL_TOL, f"{system}: curl vs dnsjax {cerr:.2e}"

    # viscoelastic conformation recipes vs the solver's spin -> phys
    # conversion (components 3..8; ground truth from _generate)
    conf_path = d / "conf_true.npy"
    if conf_path.exists():
        conf = read_state(
            d / "state.tar",
            return_physical=False,
            return_spectral=True,
            components=tuple(range(3, 9)),
        )
        truth = np.load(conf_path)
        cerr = max(_relerr(conf.spectral[i], truth[i]) for i in range(6))
        assert cerr < 1e-13, f"{system}: conformation recipes {cerr:.2e}"

    # public derivative/gradient wrappers (wiring: axis mapping, the
    # coord pairing, and the pipe parity requirement); u0 is u_x / u_z.
    info = _core.geometry_info(st.params)
    u0 = np.asarray(st.spectral[0])
    par = {"cylindrical_parity": "u_z"} if info.family == "cylindrical" else {}
    grad = gradient(u0, st.params, st.spectral_coords, **par)
    for ax in range(3):
        dax = derivative(
            u0, info.name[ax], st.params, st.spectral_coords, **par
        )
        assert _relerr(grad[ax], dax) == 0.0, f"{system} grad[{ax}]"
        if info.kind[ax] != "grid":
            # Fourier axis: exact ik against the returned wavenumbers.
            k = np.asarray(st.spectral_coords[ax])
            kshape = [1, 1, 1]
            kshape[ax] = len(k)
            ref = 1j * k.reshape(kshape) * u0
            assert _relerr(grad[ax], ref) < 1e-13, f"{system} d ax{ax}"
    if info.walled and info.family != "cylindrical":
        # Plain-D1 grid axis (the pipe's parity path is pinned by the
        # curl-vs-solver check above).
        from dnsjax.fd import build_diff_matrices

        d1, _ = build_diff_matrices(
            np.asarray(st.spectral_coords[0], dtype=float),
            int(st.params.res.fd_order),
        )
        ref = np.einsum("ij,jkl->ikl", d1, u0)
        assert _relerr(grad[0], ref) < 1e-12, f"{system} d1 axis"
    if info.family == "cylindrical":
        try:
            derivative(u0, "r", st.params, st.spectral_coords)
        except ValueError:
            pass
        else:
            raise AssertionError("pipe radial derivative needs parity")

    # divergence of the divergence-free modes (drop the real-axis DC
    # plane: random_field solves continuity per axis-1 mode via 1/k, so
    # only its k=0 plane carries divergence -- left to the corrector).
    div = np.asarray(divergence(st.spectral, st.params, st.spectral_coords))
    om = curl(st.spectral, st.params, st.spectral_coords)
    scale = max(np.linalg.norm(np.asarray(o).ravel()) for o in om)
    div_nz = div[:, 1:, :] if info.walled else div
    rel_div = np.linalg.norm(div_nz.ravel()) / scale
    assert rel_div < DIV_TOL, f"{system}: div {rel_div:.2e}"

    # integrate(ones) == analytic volume (quadrature is exact here)
    ones = np.ones_like(np.asarray(st.physical[0]).real)
    vol = float(integrate(ones, st.params, st.physical_coords))
    exp = 1.0
    for ax in range(3):
        if info.kind[ax] == "grid":
            g = np.asarray(st.physical_coords[ax])
            if info.family == "cartesian":
                exp *= g[-1] - g[0]
            elif info.family == "cylindrical":
                exp *= 0.5 * g[-1] ** 2  # int_0^rmax r dr
            else:
                exp *= 0.5 * (g[-1] ** 2 - g[0] ** 2)
        else:
            exp *= info.length[ax]
    assert abs(vol - exp) / exp < VOL_TOL, f"{system}: vol {vol} != {exp}"

    _check_subsetting(system, family, d, st)
    print(f"  {system}: OK")


def _check_subsetting(system, family, d, full) -> None:
    """Component / wall-normal subsets equal the full read, sliced."""
    from dnsjax.analysis import _core, read_state

    # single-component read returns a 1-tuple equal to the full slice
    # (cyl/annular u_r pulls the stored u_± pair and reconstructs it).
    one = read_state(d / "state.tar", return_spectral=True, components=(1,))
    assert len(one.spectral) == 1 and len(one.physical) == 1, system
    assert _relerr(one.spectral[0], full.spectral[1]) < 1e-14, system
    assert _relerr(one.physical[0], full.physical[1]) < 1e-14, system

    # wall-normal subset: nearest grid points, sliced field + grid.
    info = _core.geometry_info(full.params)
    yax = info.wall_normal_axis
    ygrid = np.asarray(full.physical_coords[yax])
    want = [float(ygrid[1]), float(ygrid[len(ygrid) // 2])]
    sub = read_state(d / "state.tar", wall_normal_points=want)
    idx = _core.nearest_unique_indices(ygrid, want)
    sg = np.asarray(sub.physical_coords[yax])
    assert np.allclose(sg, ygrid[idx]), system
    for c in range(3):
        ref = np.take(np.asarray(full.physical[c]), idx, axis=yax)
        assert _relerr(sub.physical[c], ref) < 1e-12, f"{system} wn sub"


# ── driver ───────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs=2, metavar=("SYSTEM", "OUTDIR"))
    ap.add_argument("--system", default=None)
    args = ap.parse_args()

    if args.gen is not None:
        _generate(args.gen[0], args.gen[1])
        return 0

    print(
        "Snapshot-export API tests: offline, 1 forced CPU device per "
        "system (the analysis API is JAX-free; device-independent, no "
        "GPU path).",
        flush=True,
    )

    # The import-time JAX-free guarantee: importing the API in this
    # process must not pull in JAX.
    import dnsjax.analysis  # noqa: F401

    assert "jax" not in sys.modules, "JAX leaked into dnsjax.analysis!"
    print("JAX-free import: OK")

    systems = SYSTEMS
    if args.system is not None:
        systems = [s for s in SYSTEMS if s[0] == args.system]
        if not systems:
            print(f"unknown system {args.system!r}")
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        for system, family in systems:
            out = os.path.join(tmp, system)
            os.makedirs(out, exist_ok=True)
            _gen_subprocess(system, out)
            _check_system(system, family, out)

    print("\nAll snapshot-export tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
