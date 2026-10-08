r"""No host array goes through JAX's cross-process equality check.

``jax.device_put`` of a host (NumPy) value onto a sharding that spans
several processes first checks that every process passed the same
value, by gathering all ``W`` copies onto every process
(``jax.experimental.multihost_utils.assert_equal``): each process
briefly holds about twice ``W`` times the array, so the cost grows
with the job, not with the problem.  On ARCHER2 that check, on the two
dense wall-normal derivative matrices alone, was the per-rank memory
peak of every 4-node layout (``sharding.Sharding.distribute`` has the
numbers).  Host arrays reach the mesh through ``distribute`` instead,
which builds each process's shards from its own copy.

Each case runs the solver, or the twin driver, on two ranks
(``mpirun -np 2``) with a recorder on ``assert_equal``, installed as
the runtime comes up, and then places one host array the old way as a
positive control: a case whose recorder never sees the control fails
(the check moved and this test went blind), as does any other recorded
payload over :data:`_LIMIT` bytes.  The expected small one is
``snapshot._barrier``'s 4-byte name hash
(``multihost_utils.sync_global_devices``), on the snapshot writes.

The cases build every flow family and step it a few times: Cartesian
with ``[probes]``, with ``[force]`` (its per-kick columns), resumed
onto a changed ``ny`` (the regrid matrix), with adaptive ``dt`` and
``[lowres]``, as the parent of a twin member (``dnsjax-twin``, its
streams and reduced differences on), and in single precision;
Kolmogorov from localized rolls; the pipe with ``[force]`` (its mode
columns) and with cnab2; the curved pipe; Taylor-Couette, Dean and the
quasi-Keplerian wedge; and both viscoelastic flows.  In single
precision the check itself failed every multi-process setup: it
compares the gathered copies, canonicalized to float32, with the
float64 host originals, so no float64 array survives it.

Run as a script (``--only`` takes case names; a ``-force``,
``-regrid`` or ``-twin`` case brings the case it reads from along)::

    uv run python tests/test_host_placement.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

#: Largest recorded payload, in bytes, that does not fail a case: one
#: scalar, so that every host *array* placed the old way fails, however
#: small the test grid makes it.
_LIMIT = 8

# Every case: a few steps of a small box; the initial snapshot is kept
# (the regrid and twin cases start from the Cartesian one), no other
# snapshot is written.
_RUN = [
    "--dist.np1",
    "2",
    "--stop.max_sim_time",
    "0.02",
    "--stop.check_laminarization",
    "False",
    "--outs.it_snapshot",
    "0",
    "--outs.snapshot_save_final",
    "False",
]
_FIXED_DT = ["--step.adaptive", "False", "--step.dt", "0.005"]
_SEED = ["--init.random_seed", "1"]
_COMMON = [*_RUN, *_FIXED_DT, *_SEED]
_CART = ["--res.nx", "16", "--res.ny", "17", "--res.nz", "16"]
_CYL = ["--res.nz", "16", "--res.nr", "17", "--res.ntheta", "16"]
_PROBES = ["--probes.modes", "1,1", "--probes.it_probes", "1"]
_FORCE = [
    "--force.modes",
    "1,1",
    "--force.amplitude",
    "1e-4",
    "--force.it_force",
    "2",
    "--force.seed",
    "1",
    *_PROBES,
]
_LOWRES = [
    "--lowres.it_lowres",
    "1",
    "--lowres.nx",
    "8",
    "--lowres.ny",
    "9",
    "--lowres.nz",
    "8",
]
_TWIN = [
    "--twin.e0",
    "1e-6",
    "--twin.seed",
    "3",
    "--twin.it_spectra",
    "1",
    "--twin.it_yspectra",
    "1",
    "--twin.it_ybudget",
    "1",
    "--twin.it_spectra3d",
    "1",
    "--twin.it_budget3d",
    "1",
    "--twin.it_lowres_delta",
    "1",
]
_POISEUILLE = ["--phys.system", "plane-poiseuille", "--phys.re", "2000"]
_PIPE = ["--phys.system", "pipe", "--phys.re", "1800", "--geo.lz", "5"]

#: ``(name, entry point, arguments, source case or None)``.  A
#: ``-force`` case writes its profiles on its source case's wall-normal
#: grid; the ``-regrid`` and ``-twin`` cases start from its initial
#: snapshot.
_CASES: list[tuple[str, str, list[str], str | None]] = [
    ("cartesian", "dnsjax", [*_COMMON, *_POISEUILLE, *_CART, *_PROBES], None),
    (
        "cartesian-force",
        "dnsjax",
        [*_COMMON, *_POISEUILLE, *_CART, *_FORCE],
        "cartesian",
    ),
    (
        "cartesian-regrid",
        "dnsjax",
        [
            *_COMMON,
            *_POISEUILLE,
            "--res.nx",
            "16",
            "--res.ny",
            "19",
            "--res.nz",
            "16",
        ],
        "cartesian",
    ),
    (
        "cartesian-adaptive-lowres",
        "dnsjax",
        [
            *_RUN,
            *_SEED,
            "--step.adaptive",
            "True",
            "--step.dt",
            "0.005",
            "--step.dt_max",
            "0.01",
            *_POISEUILLE,
            *_CART,
            *_LOWRES,
            "--lowres.pressure",
            "True",
        ],
        None,
    ),
    (
        "cartesian-single",
        "dnsjax",
        [
            *_COMMON,
            *_POISEUILLE,
            *_CART,
            "--res.double_precision",
            "False",
        ],
        None,
    ),
    (
        "cartesian-twin",
        "dnsjax-twin",
        [*_RUN, *_FIXED_DT, *_TWIN, *_LOWRES],
        "cartesian",
    ),
    (
        "kolmogorov-rolls",
        "dnsjax",
        [
            *_COMMON,
            "--phys.system",
            "kolmogorov",
            "--phys.re",
            "100",
            "--init.localized_rolls",
            "True",
            "--res.nx",
            "16",
            "--res.ny",
            "16",
            "--res.nz",
            "16",
        ],
        None,
    ),
    ("pipe", "dnsjax", [*_COMMON, *_PIPE, *_CYL], None),
    ("pipe-force", "dnsjax", [*_COMMON, *_PIPE, *_CYL, *_FORCE], "pipe"),
    (
        "pipe-cnab2",
        "dnsjax",
        [*_COMMON, *_PIPE, *_CYL, "--step.scheme", "cnab2"],
        None,
    ),
    (
        "curved-pipe",
        "dnsjax",
        [
            *_COMMON,
            "--phys.system",
            "curved-pipe",
            "--phys.re",
            "1800",
            "--geo.curvature",
            "0.3",
            "--geo.lz",
            "5",
            *_CYL,
        ],
        None,
    ),
    (
        "taylor-couette",
        "dnsjax",
        [
            *_COMMON,
            "--phys.system",
            "taylor-couette",
            "--phys.re1",
            "400",
            "--phys.re2",
            "-400",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
            *_CYL,
        ],
        None,
    ),
    (
        "dean",
        "dnsjax",
        [
            *_COMMON,
            "--phys.system",
            "dean",
            "--phys.re",
            "1000",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
            *_CYL,
        ],
        None,
    ),
    (
        "quasi-keplerian-wedge",
        "dnsjax",
        [
            *_COMMON,
            "--phys.system",
            "quasi-keplerian",
            "--phys.re1",
            "100",
            "--phys.r_omega",
            "-1.2",
            "--geo.eta",
            "0.71",
            "--geo.m0",
            "2",
            *_CYL,
        ],
        None,
    ),
    (
        "viscoelastic-pipe",
        "dnsjax",
        [
            *_COMMON,
            "--phys.system",
            "viscoelastic-pipe",
            "--phys.wi",
            "20",
            "--phys.el",
            "0.02",
            "--init.random_conformation_amplitude",
            "10",
            "--geo.lz",
            "5",
            *_CYL,
        ],
        None,
    ),
    (
        "viscoelastic-dean",
        "dnsjax",
        [
            *_COMMON,
            "--phys.system",
            "viscoelastic-dean",
            "--phys.wi",
            "20",
            "--phys.el",
            "20",
            "--init.random_conformation_amplitude",
            "10",
            "--geo.lz",
            "5",
            *_CYL,
        ],
        None,
    ),
]

#: The module behind each entry point (``pyproject.toml``).
_ENTRIES = {"dnsjax": "dnsjax.__main__", "dnsjax-twin": "dnsjax.twin.driver"}


def _nbytes(leaf: object) -> int:
    """A checked leaf's size, without materializing a device array."""
    dtype = getattr(leaf, "dtype", None)
    if dtype is None:
        leaf = np.asarray(leaf)
        dtype = leaf.dtype
    count = int(np.prod(np.shape(leaf), dtype=np.int64))
    return count * np.dtype(dtype).itemsize


def _child(entry_point: str, argv: list[str]) -> int:
    """One rank: *entry_point* under the recorder, then the control.

    The recorder goes in right after ``configure_jax_runtime``: JAX is
    configured then, and nothing has been placed yet.  Process 0
    prints one ``HOST_PLACEMENT`` line per recorded leaf and the run's
    wall-normal grid (``WALL_GRID``, for a ``-force`` case's profiles).
    """
    import importlib

    entry = importlib.import_module(_ENTRIES[entry_point])

    records: list[dict] = []
    control = [False]
    package = Path(importlib.import_module("dnsjax").__file__).parent
    package = package.resolve()
    configure = entry.configure_jax_runtime

    def configure_and_record(*args: object, **kwargs: object) -> bool:
        main_device = configure(*args, **kwargs)
        import jax
        from jax.experimental import multihost_utils

        check = multihost_utils.assert_equal

        def recorder(in_tree: object, fail_message: str = "") -> None:
            frames = [
                f
                for f in traceback.extract_stack()
                if Path(f.filename).resolve().is_relative_to(package)
            ]
            site = (
                f"{Path(frames[-1].filename).resolve().relative_to(package)}"
                f":{frames[-1].lineno}"
                if frames
                else "outside dnsjax"
            )
            for leaf in jax.tree.leaves(in_tree):
                records.append(
                    {
                        "bytes": _nbytes(leaf),
                        "shape": list(np.shape(leaf)),
                        "dtype": str(getattr(leaf, "dtype", "")),
                        "site": site,
                        "control": control[0],
                    }
                )
            return check(in_tree, fail_message)

        multihost_utils.assert_equal = recorder
        return main_device

    entry.configure_jax_runtime = configure_and_record
    entry.main(argv)

    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from dnsjax.parameters import derived_params
    from dnsjax.sharding import sharding

    control[0] = True
    jax.device_put(np.arange(16.0), NamedSharding(sharding.mesh, P()))
    if jax.process_index() == 0:
        for rec in records:
            print("HOST_PLACEMENT", json.dumps(rec), flush=True)
        grid = derived_params.wall_normal_grid
        print("WALL_GRID", json.dumps(list(grid) if grid else None))
    return 0


def _write_profiles(path: Path, system: str, grid: list[float]) -> None:
    """Two channels of random 3-component profiles for mode ``(1, 1)``,
    on *grid* (``extensions/forcing.py`` refuses any other)."""
    rng = np.random.default_rng(7)
    shape = (2, 3, len(grid))
    profiles = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    np.savez(
        path, system=system, code_grid=np.asarray(grid), profiles_1_1=profiles
    )


def _run(
    entry: str, args: list[str], cwd: Path, timeout: int
) -> tuple[str | None, list[float] | None]:
    """One 2-rank case: ``(failure or None, its wall-normal grid)``."""
    from _live import run_live

    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["NO_COLOR"] = "1"
    # coreutils ``timeout`` ends a hung run with SIGTERM, which mpirun
    # passes on to its ranks; run_live's later SIGKILL can orphan them.
    cmd = ["timeout", str(timeout), "mpirun", "-np", "2", sys.executable]
    cmd += [str(Path(__file__).resolve()), "--child", entry, *args]
    try:
        res = run_live(cmd, cwd=cwd, env=env, timeout=timeout + 60)
    except subprocess.TimeoutExpired:
        return f"outlived its {timeout + 60} s backstop", None
    if res.returncode != 0:
        return f"exited {res.returncode}", None
    records, grid = [], None
    for line in res.stdout.splitlines():
        if line.startswith("HOST_PLACEMENT "):
            records.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("WALL_GRID "):
            grid = json.loads(line.split(" ", 1)[1])
    if not any(r["control"] for r in records):
        return (
            "the recorder never saw the control placement: JAX no longer "
            "checks there, so this test cannot see a regression",
            grid,
        )
    over = [r for r in records if not r["control"] and r["bytes"] > _LIMIT]
    if over:
        sites = "; ".join(
            f"{r['site']} {tuple(r['shape'])} {r['dtype']} {r['bytes']} B"
            for r in over
        )
        return (
            f"{len(over)} placement(s) went through the check: {sites}",
            grid,
        )
    return None, grid


def main() -> int:
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        return _child(sys.argv[i + 1], sys.argv[i + 2 :])
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--only",
        nargs="+",
        choices=[name for name, _, _, _ in _CASES],
        help="run these cases (and the cases they read from)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="seconds per case before it counts as hung (default 600)",
    )
    cli = ap.parse_args()
    if shutil.which("mpirun") is None or shutil.which("timeout") is None:
        print("mpirun and coreutils timeout are needed")
        return 1
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _live import report

    wanted = set(cli.only or [name for name, _, _, _ in _CASES])
    wanted |= {src for name, _, _, src in _CASES if name in wanted and src}
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_host_placement_"))
    results: list[tuple[str, str | None]] = []
    grids: dict[str, list[float] | None] = {}
    try:
        for name, entry, args, source in _CASES:
            if name not in wanted:
                continue
            cwd = tmp / name
            cwd.mkdir()
            args = list(args)
            if name.endswith("-force"):
                if not grids.get(source):
                    results.append((name, f"no grid from case {source}"))
                    continue
                system = args[args.index("--phys.system") + 1]
                _write_profiles(cwd / "profiles.npz", system, grids[source])
                args += ["--force.profiles", str(cwd / "profiles.npz")]
            elif name.endswith(("-regrid", "-twin")):
                parent = tmp / source / "state00000.tar"
                if not parent.is_file():
                    results.append((name, f"no snapshot from case {source}"))
                    continue
                args += ["--init.snapshot", str(parent)]
            reason, grids[name] = _run(entry, args, cwd, cli.timeout)
            results.append((name, reason))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, reason in results:
        print(f"  {'PASS' if reason is None else 'FAIL'}  {name}")
    failures = [(n, r) for n, r in results if r is not None]
    return report(len(results) - len(failures), failures)


if __name__ == "__main__":
    sys.exit(main())
