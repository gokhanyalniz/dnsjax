r"""Reduced-resolution snapshots (``[lowres]``) and the static pressure.

Each case runs in its own worker process (the shared workers and the
reduce check: ``tests/_lowres.py``):

1. **reduce** -- plane Poiseuille, with its static pressure reduced the
   same way; the pipe at ``geo.m0 = 2``, so the physical `$m$` decides
   the parity class; Taylor-Couette; and Kolmogorov, which cuts its
   third Fourier axis instead of interpolating.  Each reduced file
   equals the host-NumPy reduction of its full field, on one device and
   on a ``(2, 2)`` mesh, and the resume path refuses it.
2. **pressure** -- plane Poiseuille and plane Couette (moving walls,
   where only the convective source keeps the mean mode's zero Neumann
   row exact): the interior Poisson residual and the influence-matrix
   wall closure `$(D_1\partial_t\hat v)|_w$` of
   ``_cartesian_pressure.static_pressure`` vanish to round-off on a
   solenoidal state, and its `$(0, 0)$` profile is
   `$-\langle v'^2\rangle$`, the mean `$y$`-momentum balance, to the
   wall-normal truncation.
3. **kind** -- a replicated real array written by ``write_archive``
   under a ``kind`` of its own is refused by the field readers and by
   the resume path.

Run as a script: ``uv run python tests/test_lowres.py``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from _live import report
from _lowres import (
    configure,
    reduce_rows,
    reduce_worker,
    run_worker,
    solenoidal_state,
)

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


# ── pressure ─────────────────────────────────────────────────────────


def _pressure_worker(system: str) -> None:
    params = configure(
        4,
        {"system": system, "re": 400.0},
        {},
        {"nx": 8, "ny": 33, "nz": 8},
        {},
    )

    import importlib

    import jax.numpy as jnp
    import numpy as np

    from dnsjax.flows.registry import spec_for
    from dnsjax.geometries.wall_bounded._base import apply_y_matrix
    from dnsjax.geometries.wall_bounded._cartesian_pressure import (
        PoissonPressure,
        convective_nonlinear,
        static_pressure,
    )
    from dnsjax.geometries.wall_bounded.cartesian import fourier

    flow = importlib.import_module(spec_for(system).flow_module).flow
    state, v_hat = solenoidal_state(params, flow)
    op = PoissonPressure(flow, fourier)
    p = static_pressure(state, op, fourier, flow)

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


# ── kind ─────────────────────────────────────────────────────────────


def _kind_worker(out: Path) -> None:
    """An archive of another ``kind`` is refused as a state and a parent."""
    params = configure(
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
    from dnsjax.parameters import read_snapshot_params
    from dnsjax.sharding import sharding
    from dnsjax.snapshot import replicated_to_io_layout, write_archive

    host = np.arange(3 * 5 * 4 * 3, dtype=np.float64).reshape(3, 5, 4, 3)
    cube = jax.device_put(host, NamedSharding(sharding.mesh, P()))
    meta = {
        "format_version": 6,
        "kind": "example_cube",
        "system": params.phys.system,
        "native_shape": [3, 5, 4, 3],
        "dtype": "float64",
        "t": 1.5,
        "it": 7,
    }
    path = out / "cube.tar"
    write_archive(
        [("state", replicated_to_io_layout(cube))],
        (5, 4, 3),
        "float64",
        path,
        meta=lambda: meta,
    )
    for refuse in (read_state, read_snapshot_params):
        try:
            refuse(path)
        except ValueError as exc:
            assert "example_cube" in str(exc), exc
        else:
            raise AssertionError(f"{refuse.__name__} read another kind")
    print("PASS kind")


# ── parent ───────────────────────────────────────────────────────────


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        results = reduce_rows(__file__, CASES, tmp)
        for system in ("plane-poiseuille", "plane-couette"):
            results.append(
                (
                    f"static pressure {system}",
                    run_worker(
                        __file__, ["--action", "pressure", "--system", system]
                    ),
                )
            )
        results.append(
            (
                "another kind refused",
                run_worker(__file__, ["--action", "kind", "--dir", tmp]),
            )
        )
    for name, reason in results:
        print(("FAIL " if reason else "PASS ") + name)
    failures = [(n, r) for n, r in results if r is not None]
    return report(len(results) - len(failures), failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--action", choices=["reduce", "pressure", "kind"])
    parser.add_argument("--system")
    parser.add_argument("--ndev", type=int, default=4)
    parser.add_argument("--dir")
    args = parser.parse_args()
    if args.worker:
        if args.action == "reduce":
            reduce_worker(
                CASES[args.system], args.system, args.ndev, Path(args.dir)
            )
        elif args.action == "pressure":
            _pressure_worker(args.system)
        else:
            _kind_worker(Path(args.dir))
        sys.exit(0)
    sys.exit(main())
