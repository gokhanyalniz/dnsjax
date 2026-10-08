r"""Reduced-resolution snapshots (``[lowres]``) and the static pressure.

Each case runs in its own worker process -- the geometry and sharding
singletons are captured at import, and several CPU devices need
``--xla_force_host_platform_device_count`` before JAX starts:

1. **reduce** -- a deterministic state of each geometry family is
   written once through :class:`dnsjax.lowres.LowResWriter`, and what
   :func:`dnsjax.analysis.read_state` / :func:`~dnsjax.analysis.read_pressure`
   read back must equal a host-NumPy reduction of the full-resolution
   field: the Fourier axes cut by harmonic (the wrap order kept), the
   wall-normal axis mapped by :func:`dnsjax.fd.build_interpolation_matrix`
   between :func:`dnsjax.fd.grid_nodes` grids -- per azimuthal parity
   class and per component for the pipe, ``geo.m0 = 2`` so the
   physical `$m$` decides the class -- the velocity's wall rows
   zeroed, and the pressure's `$(0, 0)$` mode re-pinned to exactly
   zero at the upper wall.  Plane Poiseuille also carries its static
   pressure, reduced the same way; the viscoelastic pipe exercises the
   nine-component parity table; Kolmogorov cuts its third Fourier axis
   instead of interpolating.  The metadata must describe the reduced
   field (shape, grid, ``res`` under the public names, the ``lowres``
   entry).  Each case runs on one device and on a ``(2, 2)`` mesh, and
   the parent checks the two files hold the same data to round-off.
2. **pressure** -- plane Poiseuille and plane Couette (moving walls,
   where only the convective source keeps the mean mode's zero Neumann
   row exact): ``_cartesian_pressure.static_pressure`` equals the
   twin's tested difference path with a zero reference
   (``DifferencePressure.solve`` on ``_convective_sources(0, state)``);
   the interior Poisson residual and the influence-matrix wall closure
   `$(D_1\partial_t\hat v)|_w$` vanish to round-off on a solenoidal
   state; and its `$(0, 0)$` profile is `$-\langle v'^2\rangle$`, the
   mean `$y$`-momentum balance, to the wall-normal truncation.
3. **cube** -- a replicated real array written by
   ``write_archive`` as a twin cube reads back exactly through
   :func:`dnsjax.analysis.twin.read_cube` (whole, and as single
   wall-distance slabs); the field readers and the resume path refuse
   it, and the resume path refuses a reduced snapshot too.

Run as a script: ``uv run python tests/test_lowres.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

#: ``(phys, geo, res, lowres)`` per reduce case, internal names.
CASES: dict[str, tuple[dict, dict, dict, dict]] = {
    "plane-poiseuille": (
        {"system": "plane-poiseuille", "re": 400.0},
        {},
        {"nx": 12, "ny": 17, "nz": 12},
        {"nx": 8, "ny": 9, "nz": 8},
    ),
    "pipe": (
        {"system": "pipe", "re": 400.0},
        {"m0": 2},
        {"nx": 8, "ny": 16, "nz": 12},
        {"nx": 4, "ny": 9, "nz": 8},
    ),
    "viscoelastic-pipe": (
        {"system": "viscoelastic-pipe", "wi": 5.0, "el": 5.0},
        {},
        {"nx": 8, "ny": 12, "nz": 8},
        {"nx": 4, "ny": 7, "nz": 4},
    ),
    "taylor-couette": (
        {"system": "taylor-couette", "re1": 100.0, "re2": 0.0},
        {"eta": 0.5},
        {"nx": 8, "ny": 17, "nz": 12},
        {"nx": 8, "ny": 9, "nz": 4},
    ),
    "kolmogorov": (
        {"system": "kolmogorov", "re": 20.0},
        {},
        {"nx": 8, "ny": 12, "nz": 8},
        {"nx": 4, "ny": 6, "nz": 8},
    ),
}

#: The tolerances: a GEMM and a reshard (the state), an FFT-based
#: solve on top (the pressure), relative to the field's largest entry.
RTOL_STATE = 1e-13
RTOL_PRESSURE = 1e-11


def _configure(ndev: int, phys: dict, geo: dict, res: dict, lowres: dict):
    """The parameter singletons, then JAX on *ndev* CPU devices."""
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ndev}"
    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    np0, np1 = (2, 2) if ndev == 4 else (1, 1)
    update_parameters(
        Parameters(
            phys=phys,
            geo=geo,
            res={**res, "fd_order": 4, "double_precision": True},
            dist={"np0": np0, "np1": np1},
            lowres={"it_lowres": 1, **lowres},
        )
    )
    validate_parameters()
    padded_res.set_padded_resolution(params)
    configure_jax_platform(platform_from_argv())
    return params


def _mode_value(np, c, a, qz, qx):
    """A deterministic complex entry keyed on component, row and mode."""
    return (
        (1.0 + 0.1 * c + 0.03 * a)
        * np.exp(1j * (0.3 * c + 0.2 * a + 0.7 * qz + 0.4 * qx))
        / (1.0 + qz * qz + qx * qx)
    )


# ── reduce ───────────────────────────────────────────────────────────


def _host_reduce(np, field, *, scalar: bool):
    """The reference reduction of a true-mode ``(C, A, kz, kx)`` field."""
    from dnsjax.fd import build_interpolation_matrix, grid_nodes
    from dnsjax.flows.registry import periodic_systems
    from dnsjax.harmonics import complex_harmonics, stored_mode_counts
    from dnsjax.parameters import derived_params, params
    from dnsjax.snapshot import (
        PARITY_EVEN_STORED,
        wall_regrid_geometry,
        wall_velocity_rows,
    )

    res, lo = params.res, params.lowres

    def wrap_keep(n_src: int, n_dst: int) -> list[int]:
        pos, neg = stored_mode_counts(n_dst)
        return list(range(pos)) + list(range(n_src - neg, n_src))

    if params.phys.system in periodic_systems:
        field = field[:, wrap_keep(res.ny - 1, lo.ny - 1)]
    else:
        family = wall_regrid_geometry()

        def nodes(n):
            return grid_nodes(
                family,
                n,
                params.geo.grid_type,
                params.geo.grid_stretch,
                derived_params.r_inner,
                derived_params.r_outer,
            )

        T = build_interpolation_matrix(
            nodes(res.ny), nodes(lo.ny), family, res.fd_order
        )
        if isinstance(T, tuple):
            m_phys = complex_harmonics(res.nz) * params.geo.m0
            m_even = (m_phys % 2 == 0)[None, :, None]
            parity = (True,) if scalar else PARITY_EVEN_STORED
            rows = []
            for c in range(field.shape[0]):
                a, b = T if parity[c] else T[::-1]
                rows.append(
                    np.where(
                        m_even,
                        np.einsum("ij,jzx->izx", a, field[c]),
                        np.einsum("ij,jzx->izx", b, field[c]),
                    )
                )
            field = np.stack(rows)
        else:
            field = np.einsum("ij,cjzx->cizx", T, field)
        if not scalar:
            for row in wall_velocity_rows():
                field[:3, row] = 0.0
        else:
            # The pressure's gauge, re-pinned after the truncation: its
            # (0, 0) mode zero at the upper wall.
            top = field[:, -1, 0, 0].copy()
            field[:, :, 0, 0] -= top[:, None]
    field = field[:, :, wrap_keep(res.nz - 1, lo.nz - 1)]
    return field[:, :, :, : lo.nx // 2]


def _reduce_worker(system: str, ndev: int, out: Path) -> None:
    phys, geo, res, lowres = CASES[system]
    params = _configure(ndev, phys, geo, res, lowres)

    import numpy as np

    from dnsjax.analysis import read_pressure, read_state
    from dnsjax.flows.registry import periodic_systems, spec_for
    from dnsjax.harmonics import complex_harmonics, real_harmonics
    from dnsjax.lowres import LowResWriter
    from dnsjax.snapshot import assemble_local_shards, read_metadata

    periodic = system in periodic_systems
    nz_true, nx_true = params.res.nz - 1, params.res.nx // 2
    a_true = params.res.ny - 1 if periodic else params.res.ny
    qz, qx = complex_harmonics(params.res.nz), real_harmonics(params.res.nx)

    def fill(buf, kz0, nkz, kx0, nkx):
        c = np.arange(buf.shape[0])[:, None, None, None]
        a = np.arange(a_true)[None, :, None, None]
        z = qz[kz0 : kz0 + nkz][None, None, :, None]
        x = qx[kx0 : kx0 + nkx][None, None, None, :]
        buf[:, :a_true, :nkz, :nkx] = _mode_value(np, c, a, z, x)

    state = assemble_local_shards(fill)
    writer = LowResWriter()
    path = out / f"{system}_{ndev}.tar"
    writer.write_state(state, 0.5, 42, path)

    full = np.asarray(state)[:, :a_true, :nz_true, :nx_true]
    expected = _host_reduce(np, full, scalar=False)
    got = np.stack(
        read_state(
            path,
            return_physical=False,
            return_spectral=True,
            components=range(full.shape[0]),
        ).spectral
    )
    assert got.shape == expected.shape, (got.shape, expected.shape)
    err = np.abs(got - expected).max() / np.abs(expected).max()
    assert err < RTOL_STATE, f"state: relative error {err:.3e}"

    spec = spec_for(system)
    meta = read_metadata(path)
    lo = params.lowres
    stored = meta["params"]["res"]
    for axis in ("nx", "ny", "nz"):
        assert stored[spec.alias("res", axis)] == getattr(lo, axis), stored
    assert meta["native_shape"] == [full.shape[0], *expected.shape[1:]]
    assert meta["lowres"]["field"] == "state", meta["lowres"]
    assert meta["isnap"] is None and "carried" not in meta
    if not periodic:
        assert len(meta["wall_normal_grid"]) == lo.ny

    if writer.pressure is not None:
        p_full = np.asarray(writer.static_pressure(state))[
            None, :, :nz_true, :nx_true
        ]
        p_expected = _host_reduce(np, p_full, scalar=True)[0]
        p_got = read_pressure(
            path, return_physical=False, return_spectral=True
        ).spectral[0]
        err = np.abs(p_got - p_expected).max() / np.abs(p_expected).max()
        assert err < RTOL_PRESSURE, f"pressure: relative error {err:.3e}"
        assert meta["pressure"]["kind"] == "static", meta["pressure"]
        # The gauge the metadata states, exactly, after the reduction.
        assert p_got[-1, 0, 0] == 0.0, p_got[-1, 0, 0]
    print(f"PASS reduce {system} on {ndev} device(s)")


# ── pressure ─────────────────────────────────────────────────────────


def _pressure_worker(system: str) -> None:
    params = _configure(
        4,
        {"system": system, "re": 400.0},
        {},
        {"nx": 8, "ny": 33, "nz": 8},
        {},
    )

    import jax.numpy as jnp
    import numpy as np

    from dnsjax.geometries.wall_bounded._base import apply_y_matrix
    from dnsjax.geometries.wall_bounded._cartesian_pressure import (
        PoissonPressure,
        convective_nonlinear,
        static_pressure,
    )
    from dnsjax.snapshot import assemble_local_shards
    from dnsjax.twin import diagnostics as td
    from dnsjax.twin.pressure import DifferencePressure

    flow, fourier = td.flow, td.fourier
    y = np.asarray(flow.ys)
    d1 = np.asarray(flow.D1.dense)
    kx1 = 2.0 * np.pi / params.geo.lx

    # A solenoidal state: one streamwise mode (k_x, k_z) = (1, 0) and
    # its mean, built so the discrete divergence is zero --
    # ``u = (i / k_x) D1 v`` with this run's own ``D1``.
    v_hat = 0.05 * (1.0 - y**2) ** 2
    u_hat = (1j / kx1) * (d1 @ v_hat)
    w_hat = 0.02 * (1.0 - y**2) * (1.0 + y)

    def fill(buf, kz0, nkz, kx0, nkx):
        if kz0 == 0 and kx0 <= 1 < kx0 + nkx:
            buf[0, :, 0, 1 - kx0] = u_hat
            buf[1, :, 0, 1 - kx0] = v_hat
            buf[2, :, 0, 1 - kx0] = w_hat * (1.0 + 0.5j)
        if kz0 == 0 and kx0 == 0:
            buf[0, :, 0, 0] = 0.1 * (1.0 - y**2)

    state = assemble_local_shards(fill)
    op = PoissonPressure(flow, fourier)
    p = static_pressure(state, op, fourier, flow)

    # The oracle: the twin's difference path with a zero reference --
    # the same algebra in another arrangement (it transforms the zero
    # reference, batches the gradients together), so round-off through
    # the Poisson solve: 1-2e-12 measured, a 50x margin below.
    dp = DifferencePressure(flow, fourier)
    src = td._convective_sources(jnp.zeros_like(state), state, fourier, flow)
    p_ref = dp.solve(src.delta, src.div_n, src.n_hat[1], flow, fourier)
    scale = float(np.abs(np.asarray(p_ref)).max())
    err = float(np.abs(np.asarray(p - p_ref)).max()) / scale
    assert err < 1e-10, f"oracle: relative error {err:.3e}"

    # Interior Poisson residual and the influence-matrix closure.
    n_hat, div_n = convective_nonlinear(state, fourier, flow)
    lap = apply_y_matrix(flow.D2, p) - fourier.k2 * p
    poisson = np.abs(np.asarray((lap - div_n)[1:-1])).max()
    assert poisson < 1e-10 * max(1.0, float(np.abs(div_n).max())), poisson
    v = state[1]
    dtv = (
        n_hat[1]
        - apply_y_matrix(flow.D1, p)
        + (apply_y_matrix(flow.D2, v) - fourier.k2 * v) / params.phys.re
    )
    d1b = flow.D1_bnd
    closure = np.array(
        jnp.stack(
            [
                jnp.einsum("j,jzx->zx", d1b[0], dtv),
                jnp.einsum("j,jzx->zx", d1b[-1], dtv),
            ]
        )
    )
    closure[:, 0, 0] = 0.0  # vacuous at the mean mode (M = 0)
    assert np.abs(closure).max() < 1e-9, np.abs(closure).max()

    # The mean mode: -<v'^2>, the pair weight 2 at k_x > 0.  The gap
    # is the wall-normal truncation of the discrete balance (1.0e-3 at
    # ny = 33, fd_order = 4, both flows) -- deterministic, not
    # round-off -- so the bound only has to sit above it.
    p00 = np.asarray(p)[:, 0, 0].real
    vv = 2.0 * np.abs(v_hat) ** 2
    err = np.abs(p00 + vv).max() / vv.max()
    assert err < 5e-3, f"mean mode vs -<v'^2>: relative {err:.3e}"
    print(f"PASS pressure {system} (mean-mode truncation {err:.1e})")


# ── cube ─────────────────────────────────────────────────────────────


def _cube_worker(out: Path) -> None:
    params = _configure(
        4,
        {"system": "plane-couette", "re": 400.0},
        {},
        {"nx": 8, "ny": 9, "nz": 8},
        {},
    )

    import jax
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from dnsjax.analysis import read_state
    from dnsjax.analysis.twin import read_cube
    from dnsjax.parameters import read_snapshot_params
    from dnsjax.sharding import sharding
    from dnsjax.snapshot import replicated_to_io_layout, write_archive

    host = np.arange(3 * 5 * 4 * 3, dtype=np.float64).reshape(3, 5, 4, 3)
    cube = jax.device_put(host, NamedSharding(sharding.mesh, P()))
    meta = {
        "format_version": 6,
        "kind": "twin_spectra3d",
        "cube_version": 1,
        "system": params.phys.system,
        "native_shape": [3, 5, 4, 3],
        "dtype": "float64",
        "fields": ["e_u", "e_v", "e_w"],
        "t": 1.5,
        "it": 7,
        "wall_distance": [0.0, 0.1, 0.2, 0.5, 1.0],
        "y": [-1.0, -0.9, -0.8, -0.5, 0.0],
        "iy": [0, 1, 2, 3, 4],
        "kz_harmonics": [0, 1, 2, 3],
        "kx_harmonics": [0, 1, 2],
        "lx": 2.0,
        "lz": 1.0,
    }
    path = out / "cube.tar"
    write_archive(
        [("state", replicated_to_io_layout(cube))],
        (5, 4, 3),
        "float64",
        path,
        meta=lambda: meta,
    )
    whole = read_cube(path)
    assert np.array_equal(np.stack([whole[n] for n in meta["fields"]]), host)
    slab = read_cube(path, y_rows=[1, 3])
    assert np.array_equal(slab["e_v"], host[1, [1, 3]])
    assert np.allclose(slab.wall_distance, [0.1, 0.5])
    for refuse in (read_state, read_snapshot_params):
        try:
            refuse(path)
        except ValueError as exc:
            assert "twin_spectra3d" in str(exc), exc
        else:
            raise AssertionError(f"{refuse.__name__} read a cube")
    print("PASS cube")


# ── parent ───────────────────────────────────────────────────────────


def _run(args: list[str], name: str) -> str | None:
    result = run_live(
        [sys.executable, __file__, "--worker", *args], timeout=900
    )
    if result.returncode != 0 or "PASS" not in result.stdout:
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        return " | ".join(tail) or f"exit {result.returncode}"
    return None


def _same_data(system: str, out: Path) -> str | None:
    """The 1-device and (2, 2) files hold the same data (round-off)."""
    import numpy as np

    from dnsjax.analysis import read_pressure, read_state
    from dnsjax.parameters import read_snapshot_params

    a, b = out / f"{system}_1.tar", out / f"{system}_4.tar"
    for reader in (read_state, read_pressure):
        try:
            sa = reader(a, return_physical=False, return_spectral=True)
        except ValueError:
            continue  # no pressure member for this flow
        sb = reader(b, return_physical=False, return_spectral=True)
        for x, y in zip(sa.spectral, sb.spectral, strict=True):
            err = np.abs(x - y).max() / max(np.abs(x).max(), 1e-300)
            if err > RTOL_PRESSURE:
                return f"{reader.__name__}: meshes differ by {err:.3e}"
    try:
        read_snapshot_params(a)
    except ValueError as exc:
        if "not a checkpoint" not in str(exc):
            return f"unexpected refusal: {exc}"
    else:
        return "a reduced snapshot was accepted as a checkpoint"
    return None


def main() -> int:
    results: list[tuple[str, str | None]] = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        for system in CASES:
            for ndev in (1, 4):
                name = f"reduce {system} ({ndev} device(s))"
                results.append(
                    (
                        name,
                        _run(
                            [
                                "--action",
                                "reduce",
                                "--system",
                                system,
                                "--ndev",
                                str(ndev),
                                "--dir",
                                tmp,
                            ],
                            name,
                        ),
                    )
                )
            results.append(
                (f"reduce {system}: meshes agree", _same_data(system, out))
            )
        for system in ("plane-poiseuille", "plane-couette"):
            name = f"static pressure {system}"
            results.append(
                (
                    name,
                    _run(["--action", "pressure", "--system", system], name),
                )
            )
        results.append(
            (
                "cube container",
                _run(["--action", "cube", "--dir", tmp], "cube"),
            )
        )
    for name, reason in results:
        print(("FAIL " if reason else "PASS ") + name)
    failures = [(n, r) for n, r in results if r is not None]
    return report(len(results) - len(failures), failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--action", choices=["reduce", "pressure", "cube"])
    parser.add_argument("--system")
    parser.add_argument("--ndev", type=int, default=4)
    parser.add_argument("--dir")
    args = parser.parse_args()
    if args.worker:
        if args.action == "reduce":
            _reduce_worker(args.system, args.ndev, Path(args.dir))
        elif args.action == "pressure":
            _pressure_worker(args.system)
        else:
            _cube_worker(Path(args.dir))
        sys.exit(0)
    sys.exit(main())
