r"""Shared machinery of the host-placement tests, core and lab.

The contract the scripts pin, and why:

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

Each case runs an entry point (the solver, or a plugin's driver) on two ranks
(``mpirun -np 2``) with a recorder on ``assert_equal``, installed as
the runtime comes up, and then places one host array the old way as a
positive control: a case whose recorder never sees the control fails
(the check moved and this test went blind), as does any other recorded
payload over :data:`LIMIT` bytes.  The expected small one is
``snapshot._barrier``'s 4-byte name hash
(``multihost_utils.sync_global_devices``), on the snapshot writes.

Each script lists its cases and calls :func:`main`.
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
from _live import report, run_live

#: Largest recorded payload, in bytes, that does not fail a case: one
#: scalar, so that every host *array* placed the old way fails, however
#: small the test grid makes it.
LIMIT = 8

# Every case: a few steps of a small box; the initial snapshot is kept
# (a case may start from its source case's), no other snapshot is
# written.
RUN = [
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
FIXED_DT = ["--step.adaptive", "False", "--step.dt", "0.005"]
SEED = ["--init.random_seed", "1"]
COMMON = [*RUN, *FIXED_DT, *SEED]
CART = ["--res.nx", "16", "--res.ny", "17", "--res.nz", "16"]
CYL = ["--res.nz", "16", "--res.nr", "17", "--res.ntheta", "16"]
LOWRES = [
    "--lowres.it_lowres",
    "1",
    "--lowres.nx",
    "8",
    "--lowres.ny",
    "9",
    "--lowres.nz",
    "8",
]
POISEUILLE = ["--phys.system", "plane-poiseuille", "--phys.re", "2000"]
PIPE = ["--phys.system", "pipe", "--phys.re", "1800", "--geo.lz", "5"]


def nbytes(leaf: object) -> int:
    """A checked leaf's size, without materializing a device array."""
    dtype = getattr(leaf, "dtype", None)
    if dtype is None:
        leaf = np.asarray(leaf)
        dtype = leaf.dtype
    count = int(np.prod(np.shape(leaf), dtype=np.int64))
    return count * np.dtype(dtype).itemsize


def child(module: str, argv: list[str]) -> int:
    """One rank: the entry point *module* under the recorder, then the control.

    The recorder goes in right after ``configure_jax_runtime``: JAX is
    configured then, and nothing has been placed yet.  Process 0
    prints one ``HOST_PLACEMENT`` line per recorded leaf and the run's
    wall-normal grid (``WALL_GRID``, for a later case built on it).
    """
    import importlib

    entry = importlib.import_module(module)

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
                        "bytes": nbytes(leaf),
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


def run(
    script: str, entry: str, args: list[str], cwd: Path, timeout: int
) -> tuple[str | None, list[float] | None]:
    """One 2-rank case: ``(failure or None, its wall-normal grid)``."""
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["NO_COLOR"] = "1"
    # coreutils ``timeout`` ends a hung run with SIGTERM, which mpirun
    # passes on to its ranks; run_live's later SIGKILL can orphan them.
    cmd = ["timeout", str(timeout), "mpirun", "-np", "2", sys.executable]
    cmd += [str(Path(script).resolve()), "--child", entry, *args]
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
    over = [r for r in records if not r["control"] and r["bytes"] > LIMIT]
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


def main(
    script: str,
    cases: list[tuple],
    entries: dict[str, str],
    description: str,
) -> int:
    """Run *script*'s *cases*; ``--child`` runs one rank of one case.

    A case is ``(name, entry point, arguments, source case or None,
    prepare or None)``; *entries* maps each entry point to its module.
    ``prepare(cwd, args, source_dir, source_grid)`` returns the
    case's arguments, completed from its source case's directory and
    wall-normal grid, or a failure string.
    """
    sys.stdout.reconfigure(line_buffering=True)
    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        return child(entries[sys.argv[i + 1]], sys.argv[i + 2 :])
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--only",
        nargs="+",
        choices=[case[0] for case in cases],
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

    wanted = set(cli.only or [case[0] for case in cases])
    wanted |= {case[3] for case in cases if case[0] in wanted and case[3]}
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_host_placement_"))
    results: list[tuple[str, str | None]] = []
    grids: dict[str, list[float] | None] = {}
    try:
        for name, entry, args, source, prepare in cases:
            if name not in wanted:
                continue
            cwd = tmp / name
            cwd.mkdir()
            args = list(args)
            if prepare is not None:
                args = prepare(cwd, args, tmp / source, grids.get(source))
                if isinstance(args, str):
                    results.append((name, args))
                    continue
            reason, grids[name] = run(script, entry, args, cwd, cli.timeout)
            results.append((name, reason))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for name, reason in results:
        print(f"  {'PASS' if reason is None else 'FAIL'}  {name}")
    failures = [(n, r) for n, r in results if r is not None]
    return report(len(results) - len(failures), failures)


def from_snapshot(cwd: Path, args: list[str], source: Path, grid) -> list[str]:
    """A ``prepare`` that starts the case from its source's first snapshot."""
    parent = source / "state00000.tar"
    if not parent.is_file():
        return f"no snapshot from case {source.name}"
    return [*args, "--init.snapshot", str(parent)]
