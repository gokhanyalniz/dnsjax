r"""Shared workers of the ``[lowres]`` tests, core and lab.

Each case runs in its own worker process -- the geometry and sharding
singletons are captured at import, and several CPU devices need
``--xla_force_host_platform_device_count`` before JAX starts.  A
*reduce* case writes a deterministic state of its flow once through
:class:`dnsjax.lowres.LowResWriter`, and what
:func:`dnsjax.analysis.read_state` / :func:`~dnsjax.analysis.read_pressure`
read back must equal a host-NumPy reduction of the full-resolution
field: the Fourier axes cut by harmonic (the wrap order kept), the
wall-normal axis mapped by :func:`dnsjax.fd.build_interpolation_matrix`
between :func:`dnsjax.fd.grid_nodes` grids -- per azimuthal parity
class and per component for the pipe (the flow's
``FlowSpec.component_parity``) -- the velocity's wall rows zeroed, and
the pressure's `$(0, 0)$` mode re-pinned to exactly zero at the upper
wall.  The metadata must describe the reduced field (shape, grid,
``res`` under the public names, the ``lowres`` entry).  Each case runs
on one device and on a ``(2, 2)`` mesh, and the parent checks the two
files hold the same data to round-off (:func:`reduce_rows`).

:func:`solenoidal_state` is the static-pressure checks' state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from _live import run_live

#: The tolerances: a GEMM and a reshard (the state), an FFT-based
#: solve on top (the pressure), relative to the field's largest entry.
RTOL_STATE = 1e-13
RTOL_PRESSURE = 1e-11


def configure(ndev: int, phys: dict, geo: dict, res: dict, lowres: dict):
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


def mode_value(np, c, a, qz, qx):
    """A deterministic complex entry keyed on component, row and mode."""
    return (
        (1.0 + 0.1 * c + 0.03 * a)
        * np.exp(1j * (0.3 * c + 0.2 * a + 0.7 * qz + 0.4 * qx))
        / (1.0 + qz * qz + qx * qx)
    )


# ── reduce ───────────────────────────────────────────────────────────


def host_reduce(np, field, *, scalar: bool):
    """The reference reduction of a true-mode ``(C, A, kz, kx)`` field."""
    from dnsjax.fd import build_interpolation_matrix, grid_nodes
    from dnsjax.flows.registry import periodic_systems
    from dnsjax.harmonics import complex_harmonics, stored_mode_counts
    from dnsjax.parameters import derived_params, params
    from dnsjax.snapshot import (
        stored_component_parity,
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
            parity = (True,) if scalar else stored_component_parity()
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
            # The pressure's gauge, re-pinned after the truncation.
            top = field[:, -1, 0, 0].copy()
            field[:, :, 0, 0] -= top[:, None]
    field = field[:, :, wrap_keep(res.nz - 1, lo.nz - 1)]
    return field[:, :, :, : lo.nx // 2]


def reduce_worker(case: tuple, system: str, ndev: int, out: Path) -> None:
    """Write *case*'s state reduced, and check the files against NumPy."""
    phys, geo, res, lowres = case
    params = configure(ndev, phys, geo, res, lowres)

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
        buf[:, :a_true, :nkz, :nkx] = mode_value(np, c, a, z, x)

    state = assemble_local_shards(fill)
    writer = LowResWriter()
    path = out / f"{system}_{ndev}.tar"
    writer.write_state(state, 0.5, 42, path)

    full = np.asarray(state)[:, :a_true, :nz_true, :nx_true]
    expected = host_reduce(np, full, scalar=False)
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
        p_expected = host_reduce(np, p_full, scalar=True)[0]
        p_got = read_pressure(
            path, return_physical=False, return_spectral=True
        ).spectral[0]
        err = np.abs(p_got - p_expected).max() / np.abs(p_expected).max()
        assert err < RTOL_PRESSURE, f"pressure: relative error {err:.3e}"
        assert meta["pressure"]["kind"] == "static", meta["pressure"]
        # The gauge the metadata states, exactly, after the reduction.
        assert p_got[-1, 0, 0] == 0.0, p_got[-1, 0, 0]
    print(f"PASS reduce {system} on {ndev} device(s)")


# ── the static pressure's state ──────────────────────────────────────


def solenoidal_state(params, flow):
    r"""A solenoidal Cartesian state and its ``(u, v, w)`` profiles.

    One streamwise mode `$(k_x, k_z) = (1, 0)$` and its mean, built so
    the discrete divergence is zero -- ``u = (i / k_x) D1 v`` with this
    run's own ``D1``.  Returns ``(state, v_hat)``.
    """
    import numpy as np

    from dnsjax.snapshot import assemble_local_shards

    y = np.asarray(flow.ys)
    d1 = np.asarray(flow.D1.dense)
    kx1 = 2.0 * np.pi / params.geo.lx
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

    return assemble_local_shards(fill), v_hat


# ── parent ───────────────────────────────────────────────────────────


def run_worker(script: str, args: list[str]) -> str | None:
    """Run *script*'s worker on *args*; the failure tail, or ``None``."""
    result = run_live([sys.executable, script, "--worker", *args], timeout=900)
    if result.returncode != 0 or "PASS" not in result.stdout:
        tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
        return " | ".join(tail) or f"exit {result.returncode}"
    return None


def same_data(system: str, out: Path) -> str | None:
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


def reduce_rows(
    script: str, cases: dict, tmp: str
) -> list[tuple[str, str | None]]:
    """Every reduce case of *script* on 1 and 4 devices, and their match."""
    out = Path(tmp)
    results: list[tuple[str, str | None]] = []
    for system in cases:
        for ndev in (1, 4):
            name = f"reduce {system} ({ndev} device(s))"
            args = ["--action", "reduce", "--system", system]
            args += ["--ndev", str(ndev), "--dir", tmp]
            results.append((name, run_worker(script, args)))
        results.append(
            (f"reduce {system}: meshes agree", same_data(system, out))
        )
    return results
