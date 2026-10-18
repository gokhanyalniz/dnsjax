r"""Reduced-resolution snapshots (``[lowres]``), for the lab.

The lab's cases of ``tests/test_lowres.py`` (the shared workers:
``tests/_lowres.py``), each in its own worker process:

1. **reduce** -- the viscoelastic pipe, whose nine stored components
   exercise the spec's parity table (``FlowSpec.component_parity``).
2. **pressure reference** -- plane Poiseuille and plane Couette:
   ``_cartesian_pressure.static_pressure`` equals the twin's tested
   difference path with a zero reference (``DifferencePressure.solve``
   on ``_convective_sources(0, state)``).
3. **cube** -- a replicated real array written by ``write_archive`` as
   a twin cube reads back exactly through
   :func:`dnsjax_lab.analysis.twin.read_cube` (whole, and as single
   wall-distance slabs).

Run as a script: ``uv run python lab/tests/test_lowres.py``.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import _repo  # noqa: F401 -- the core suite's helpers
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
    "viscoelastic-pipe": (
        {"system": "viscoelastic-pipe", "wi": 5.0, "el": 5.0},
        {},
        {"nx": 8, "ny": 12, "nz": 8},
        {"nx": 4, "ny": 7, "nz": 4},
    ),
}


# ── pressure reference ───────────────────────────────────────────────


def _reference_worker(system: str) -> None:
    """The static pressure against the twin's difference path."""
    params = configure(
        4,
        {"system": system, "re": 400.0},
        {},
        {"nx": 8, "ny": 33, "nz": 8},
        {},
    )

    import jax.numpy as jnp
    import numpy as np
    from dnsjax_lab.twin import diagnostics as td
    from dnsjax_lab.twin.pressure import DifferencePressure

    from dnsjax.geometries.wall_bounded._cartesian_pressure import (
        PoissonPressure,
        static_pressure,
    )

    flow, fourier = td.flow, td.fourier
    state, _ = solenoidal_state(params, flow)
    p = static_pressure(state, PoissonPressure(flow, fourier), fourier, flow)

    # The same algebra in another arrangement (it transforms the zero
    # reference, batches the gradients together), so round-off through
    # the Poisson solve: 1-2e-12 measured, a 50x margin below.
    dp = DifferencePressure(flow, fourier)
    src = td._convective_sources(jnp.zeros_like(state), state, fourier, flow)
    p_ref = dp.solve(src.delta, src.div_n, src.n_hat[1], flow, fourier)
    scale = float(np.abs(np.asarray(p_ref)).max())
    err = float(np.abs(np.asarray(p - p_ref)).max()) / scale
    assert err < 1e-10, f"reference: relative error {err:.3e}"
    print(f"PASS pressure reference {system}")


# ── cube ─────────────────────────────────────────────────────────────


def _cube_worker(out: Path) -> None:
    """A twin cube reads back exactly, whole and by wall-distance slab."""
    params = configure(
        4,
        {"system": "plane-couette", "re": 400.0},
        {},
        {"nx": 8, "ny": 9, "nz": 8},
        {},
    )

    import jax
    import numpy as np
    from dnsjax_lab.analysis.twin import read_cube
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

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
    print("PASS cube")


# ── parent ───────────────────────────────────────────────────────────


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        results = reduce_rows(__file__, CASES, tmp)
        for system in ("plane-poiseuille", "plane-couette"):
            results.append(
                (
                    f"static pressure reference {system}",
                    run_worker(
                        __file__, ["--action", "reference", "--system", system]
                    ),
                )
            )
        results.append(
            (
                "cube container",
                run_worker(__file__, ["--action", "cube", "--dir", tmp]),
            )
        )
    for name, reason in results:
        print(("FAIL " if reason else "PASS ") + name)
    failures = [(n, r) for n, r in results if r is not None]
    return report(len(results) - len(failures), failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--action", choices=["reduce", "reference", "cube"])
    parser.add_argument("--system")
    parser.add_argument("--ndev", type=int, default=4)
    parser.add_argument("--dir")
    args = parser.parse_args()
    if args.worker:
        if args.action == "reduce":
            reduce_worker(
                CASES[args.system], args.system, args.ndev, Path(args.dir)
            )
        elif args.action == "reference":
            _reference_worker(args.system)
        else:
            _cube_worker(Path(args.dir))
        sys.exit(0)
    sys.exit(main())
