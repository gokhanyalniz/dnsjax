r"""Shared machinery of the snapshot-export API tests, core and lab.

Each script lists its cases and calls :func:`main`: every case is
generated in a JAX subprocess (:func:`generate`, forced CPU device) and
checked in the parent with NumPy alone (:func:`check_system`), so the
import-time JAX-free guarantee of :mod:`dnsjax.analysis` is asserted
directly.  What the checks pin: the core script's docstring.

A case is a dict: ``system``, optional ``m0`` (the azimuthal wedge),
``phys`` / ``geo`` / ``init`` overrides, ``fixtures`` (see
:func:`generate`), ``exp_re`` (the ``phys.re`` the read-back must give,
default ``RE``) and ``extra_check`` (``(system, state_data,
geometry_info, ones)``, run after the volume check, for checks of the
flow's own).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

NX, NY, NZ = 8, 24, 8
LX, LZ = 5.0, 5.0
RE = 100.0

CURL_TOL = 1e-10  # vs dnsjax _curl_fn (expect machine precision)
DIV_TOL = 1e-8  # divergence of the divergence-free modes (k != 0)
DIV_PARITY_TOL = 1e-12  # vs the solver's own divergence, on DIV_PROBE
#: Per-component scaling that turns the solenoidal IC into a field
#: with O(1) divergence.  Operator *equality* is only observable on a
#: field that is not divergence-free -- on the IC both operators
#: return zero whatever they are, which is why ``DIV_TOL`` above
#: cannot catch a mismatch.
DIV_PROBE = (1.0, 2.0, 3.0)
RT_TOL = 1e-10  # transform round-trip (all families)
VOL_TOL = 1e-10  # integrate(ones) vs analytic volume


# ── fixture generation (JAX subprocess; forced CPU) ──────────────


def solver_divergence(state, flow, fourier):
    r"""dnsjax's own discrete divergence of a **solver-basis** *state*.

    Transcribed from stage 1 of each family's ``_imm_iteration`` (the
    only place the solver forms `$\nabla\cdot\mathbf{u}$`), so the
    check downstream compares the analysis operator against the
    discretization the influence-matrix pass constrains -- including
    the pipe's parity-reduced radial ``D1`` and the decoupled
    `$(u_z, u_\pm)$` assembly, which no physical-component reference
    could pin without restating the analysis formula itself.
    """
    import jax.numpy as jnp

    from dnsjax.flows.registry import cartesian_systems, cylindrical_systems
    from dnsjax.geometries.wall_bounded._base import apply_y_matrix
    from dnsjax.parameters import params

    system = params.phys.system
    if system in cartesian_systems:
        u_n, v_n, w_n = state[0], state[1], state[2]
        dy_v_n = apply_y_matrix(flow.D1, v_n)
        return 1j * fourier.kx * u_n + dy_v_n + 1j * fourier.kz * w_n

    uz_n, up_n, um_n = state[0], state[1], state[2]
    m = fourier.m
    stacked = jnp.stack([up_n, um_n], axis=1)
    if system in cylindrical_systems:
        # (-1)^{m+1} parity of u_pm: D1_pos +/- the folded ghost rows.
        parity_sign_v = -(fourier.m_is_even * 2 - 1)
        g, gc = flow.D1_ghost.shape
        dy = apply_y_matrix(flow.D1_pos, stacked, component_axis=1)
        dy_ghost = apply_y_matrix(
            flow.D1_ghost, stacked[:gc], component_axis=1
        )
        dy_all = dy.at[:g].add(parity_sign_v * dy_ghost)
    else:  # annular: no axis, plain D1
        dy_all = apply_y_matrix(flow.D1, stacked, component_axis=1)
    inv_r = flow.inv_r[:, None, None]
    return (
        (dy_all[:, 0] + (m + 1) * inv_r * up_n) / 2
        + (dy_all[:, 1] + (1 - m) * inv_r * um_n) / 2
        + 1j * fourier.kz * uz_n
    )


def generate(case: dict, system: str, outdir: str, m0: int) -> None:
    """Write *case*'s fixtures (``state.tar`` and the rest) at wedge *m0*.

    Runs inside the ``--gen`` subprocess; the only code path that
    imports JAX.  The case's ``fixtures(state, probe, outdir)`` returns
    ``(get_stats, omega, div_true)``: the flow's stats function, the
    solver's own curl of the state (or ``None``) and its divergence of
    *probe* (or ``None``); it may save ``extra_true.npy``, the stored
    components past the velocity.
    """
    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    phys = {"system": system, **case.get("phys", {"re": RE})}
    geo = {"lx": LX, "lz": LZ, **case.get("geo", {})}
    if m0 != 1:
        # Azimuthal wedge (cylindrical / annular surfaces only); the
        # spec's derive then overwrites geo.lz with 2*pi/m0.
        geo["m0"] = m0

    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1},
            phys=phys,
            geo=geo,
            res={
                "nx": NX,
                "ny": NY,
                "nz": NZ,
                "fd_order": 4,
                "double_precision": True,
            },
            init=case.get("init", {}),
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)
    validate_parameters()
    configure_jax_platform(platform_from_argv())

    import jax.numpy as jnp

    from dnsjax.ic.random_field import generate_random_state
    from dnsjax.snapshot import save_snapshot

    state = generate_random_state(0.2, 0.4, 0.4, 0.14, 1)
    # The divergence ground truth is taken on a deliberately
    # non-solenoidal probe (see DIV_PROBE); the checker rebuilds the
    # same scaling from the stored state.
    probe = state[:3] * jnp.asarray(DIV_PROBE)[:, None, None, None]
    get_stats, omega, div_true = case["fixtures"](state, probe, outdir)

    stats = {k: float(v) for k, v in get_stats(state).items()}
    save_snapshot(
        state, 0.0, 0, os.path.join(outdir, "state.tar"), stats=stats, isnap=0
    )
    if omega is not None:
        save_snapshot(omega, 0.0, 0, os.path.join(outdir, "omega.tar"))
    if div_true is not None:
        np.save(os.path.join(outdir, "div_true.npy"), np.asarray(div_true))


def gen_subprocess(script: str, system: str, outdir: str, m0: int) -> None:
    """Spawn the ``--gen`` generation subprocess (forced 1 CPU device)."""
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    env["NPROC"] = "1"
    subprocess.run(
        [sys.executable, script, "--gen", system, outdir, str(m0)],
        check=True,
        env=env,
    )


# ── checks (NumPy only; JAX must NOT be imported here) ────────────


def relerr(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / (denom if denom else 1.0))


def check_system(system: str, outdir: str, case: dict) -> None:
    """The NumPy-only checks of one case's generated fixture directory."""
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

    # params / stats object-like access (a derived re is rehydrated by
    # params_namespace from the derived internal field)
    assert float(st.params.phys.re) == case.get("exp_re", RE), system
    assert st.params.phys.system == system, system
    assert st.stats is not None, system
    _ = st.stats[next(iter(st.stats.keys()))]  # item access works

    # transform round-trip: machine-precision exact for every family.
    # Every stored component is the transform of a real field (the IC
    # draws the real-FFT-axis-0 plane per-component Hermitian), so the
    # (u_z, u_r, u_theta) basis round-trips like cartesian/periodic.
    back = to_spectral(to_physical(st.spectral, st.params), st.params)
    rt = max(relerr(back[i], st.spectral[i]) for i in range(3))
    assert rt < RT_TOL, f"{system}: roundtrip {rt:.2e}"

    # curl vs dnsjax _curl_fn (raw chunks, same basis/order)
    omega_path = d / "omega.tar"
    if omega_path.exists():
        my_omega = curl(st.spectral, st.params, st.spectral_coords)
        meta = _core.read_meta(omega_path)
        raw = _core.read_chunks(omega_path, meta, [0, 1, 2])
        cerr = max(relerr(my_omega[i], raw[i]) for i in range(3))
        assert cerr < CURL_TOL, f"{system}: curl vs dnsjax {cerr:.2e}"

    # read-back of the components past the velocity vs the stored
    # slice (ground truth from the case's fixtures)
    extra_path = d / "extra_true.npy"
    if extra_path.exists():
        truth = np.load(extra_path)
        extra = read_state(
            d / "state.tar",
            return_physical=False,
            return_spectral=True,
            components=tuple(range(3, 3 + len(truth))),
        )
        cerr = max(
            relerr(extra.spectral[i], truth[i]) for i in range(len(truth))
        )
        assert cerr < 1e-13, f"{system}: extra components {cerr:.2e}"

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
        assert relerr(grad[ax], dax) == 0.0, f"{system} grad[{ax}]"
        if info.kind[ax] != "grid":
            # Fourier axis: exact ik against the returned wavenumbers.
            k = np.asarray(st.spectral_coords[ax])
            kshape = [1, 1, 1]
            kshape[ax] = len(k)
            ref = 1j * k.reshape(kshape) * u0
            assert relerr(grad[ax], ref) < 1e-13, f"{system} d ax{ax}"
    if info.walled and info.family != "cylindrical":
        # Plain-D1 grid axis (the pipe's parity path is pinned by the
        # curl-vs-solver check above).
        from dnsjax.fd import build_diff_matrices

        d1, _ = build_diff_matrices(
            np.asarray(st.spectral_coords[0], dtype=float),
            int(st.params.res.fd_order),
        )
        ref = np.einsum("ij,jkl->ikl", d1, u0)
        assert relerr(grad[0], ref) < 1e-12, f"{system} d1 axis"
    if info.family == "cylindrical":
        try:
            derivative(u0, "r", st.params, st.spectral_coords)
        except ValueError:
            pass
        else:
            raise AssertionError("pipe radial derivative needs parity")

    # divergence of the divergence-free modes (drop the real-axis DC
    # plane: random_field solves continuity per real-axis mode via 1/k,
    # so only its k=0 plane carries divergence -- left to the
    # corrector).  The real-FFT axis is axis 2 in the native layout.
    div = np.asarray(divergence(st.spectral, st.params, st.spectral_coords))
    om = curl(st.spectral, st.params, st.spectral_coords)
    scale = max(np.linalg.norm(np.asarray(o).ravel()) for o in om)
    div_nz = div[:, :, 1:] if info.walled else div
    rel_div = np.linalg.norm(div_nz.ravel()) / scale
    assert rel_div < DIV_TOL, f"{system}: div {rel_div:.2e}"

    # divergence vs the solver's own operator, node-for-node.  The
    # check above cannot see an operator mismatch (both sides return
    # zero on a solenoidal field), so this one runs on DIV_PROBE --
    # the per-component rescaling of the same state, which carries
    # O(1) divergence.  The ground truth came from the solver's
    # decoupled basis, so this also pins the physical <-> u_pm
    # equivalence the stored basis relies on.
    div_path = d / "div_true.npy"
    if div_path.exists():
        probe = [
            np.asarray(c) * s
            for c, s in zip(st.spectral, DIV_PROBE, strict=True)
        ]
        mine = np.asarray(divergence(probe, st.params, st.spectral_coords))
        derr = relerr(mine, np.load(div_path))
        assert derr < DIV_PARITY_TOL, (
            f"{system}: divergence vs dnsjax {derr:.2e}"
        )

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
    if "extra_check" in case:
        case["extra_check"](system, st, info, ones)

    check_subsetting(system, d, st)
    wedge = "" if info.azimuthal_m0 == 1 else f" (m0={info.azimuthal_m0})"
    print(f"  {system}{wedge}: OK")


def check_subsetting(system, d, full) -> None:
    """Component / wall-normal subsets equal the full read, sliced."""
    from dnsjax.analysis import _core, read_state

    # single-component read returns a 1-tuple equal to the full slice
    # (each component reads exactly its own stored chunk).
    one = read_state(d / "state.tar", return_spectral=True, components=(1,))
    assert len(one.spectral) == 1 and len(one.physical) == 1, system
    assert relerr(one.spectral[0], full.spectral[1]) < 1e-14, system
    assert relerr(one.physical[0], full.physical[1]) < 1e-14, system

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
        assert relerr(sub.physical[c], ref) < 1e-12, f"{system} wn sub"


# ── driver ───────────────────────────────────────────────────────


def main(script: str, cases: list[dict]) -> int:
    """Generate and check *cases* (``--gen`` runs one generator)."""
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs=3, metavar=("SYSTEM", "OUTDIR", "M0"))
    ap.add_argument("--system", default=None)
    args = ap.parse_args()

    def find(system: str, m0: int) -> dict:
        return next(
            c for c in cases if c["system"] == system and c.get("m0", 1) == m0
        )

    if args.gen is not None:
        system, outdir, m0 = args.gen[0], args.gen[1], int(args.gen[2])
        generate(find(system, m0), system, outdir, m0)
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

    chosen = cases
    if args.system is not None:
        chosen = [c for c in cases if c["system"] == args.system]
        if not chosen:
            print(f"unknown system {args.system!r}")
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        for case in chosen:
            system, m0 = case["system"], case.get("m0", 1)
            out = os.path.join(tmp, f"{system}-m0{m0}")
            os.makedirs(out, exist_ok=True)
            gen_subprocess(script, system, out, m0)
            check_system(system, out, case)

    # The promise holds past the reads too: a flow's ``analysis_metric``
    # module, imported on its snapshot's first read, is JAX-free.
    assert "jax" not in sys.modules, "JAX leaked into a read (analysis_metric)"
    print("JAX-free reads: OK")

    print("\nAll snapshot-export tests passed.")
    return 0
