#!/usr/bin/env python3
r"""Layout and scaling sweep for one production run (JAX-free driver).

Before a production campaign on a new machine, the questions are which
process/device layout makes a step cheapest, what it costs in memory,
and how the step scales with the number of nodes.  This driver launches
the ordinary solver (``.venv/bin/dnsjax``) on one fixed problem over a
matrix of layouts, parses each run's closing summary (``s/t``,
``s/rhs``, the peak memory, the start-up phases), and prints one table
(and, with ``--csv``, one machine-readable row per run).  It launches,
it does not configure: the problem -- system, resolution, horizon -- is
whatever ``--solver-args`` (and ``--toml``) say, so the sweep measures
the run you intend to make.

Targets
-------
``--target cpu`` (a many-core node or several, e.g. 2 x 64-core AMD
EPYC 7742 per node): one rank per core, each pinned to one XLA thread
(the solver's own ``NPROC=1`` default).  Two launchers:

``--launcher mpirun`` (the default; one node)
    ``mpirun -np N --map-by core --bind-to core`` (Open MPI;
    ``--mpirun-args`` replaces the binding flags for another MPI).
    Sweeps the rank count (``--ranks``).
``--launcher srun`` (Slurm; run the driver inside the allocation)
    ``srun --nodes --ntasks --ntasks-per-node --cpus-per-task
    --hint=nomultithread --distribution --kill-on-bad-exit=1``.
    Sweeps the node count (``--nodes``), the ranks per node
    (``--tasks-per-node``), the cores per rank (``--cpus-per-task``,
    default the node's cores divided evenly, which is how an
    underpopulated node keeps its ranks spread over every memory
    channel) and the task placement (``--distribution``).  The run
    directories must be on a filesystem every node sees
    (``--workdir``): a rank on another node cannot enter the driver
    node's ``/tmp``.

At each rank count every ``(np0, np1)`` factorisation runs, or only
those whose ``np1`` is listed in ``--np1``.  Export
``MPITRAMPOLINE_LIB`` first (``docs/cpu-collectives.md``): without it
the collectives run over ``gloo``, which is not what a production run
should measure.

``--target gpu`` (e.g. 4 x NVIDIA H200 in one node)
    one process spanning every visible GPU (the launch the
    ``Distribution`` docstring recommends for a single node); sweeps the
    mesh (every factorisation of ``--gpus``), optionally the Pallas
    tile (``--tiles``) and the precision (``--precisions``).

Arms and repeats
----------------
``--variant NAME=ARGS`` adds solver flags as a named arm (e.g.
``chunks2="--solver.rhs_transform_chunks 2"``); ``--exe NAME=PATH``
names a ``dnsjax`` executable per arm, so two checkouts of the code
can be compared in one sweep.  ``--repeats R`` runs the whole row list
R times, reversing the order on every second pass, so drift over the
allocation does not favour whichever arm runs first.

What is reported
----------------
- ``s/t`` (wall seconds per unit of simulated time) and ``s/rhs``,
  both from the solver's own clock, which starts after the first,
  compiling, step;
- the efficiency against the fastest row at the smallest node count
  (``srun``, node-based: ``N0 T(N0) / (N T(N))``) or the smallest rank
  count (``mpirun``, rank-based), and with ``srun`` the cost in node
  hours per unit of simulated time (``CU/t``);
- the peak memory: the CPU ``Peak host memory`` line (per rank, and
  the fullest node with shared pages counted once) or the GPU
  ``Peak device memory`` line;
- the start-up phases from the solver's own timestamps: process start
  to distributed runtime (imports, rank discovery), to the first step
  (initial condition, operators), and the first step itself (its
  compile);
- each row's padding: ``--dry-run`` prints, for every layout, what
  the start-up diagnostics would report, without launching anything.

Rows that fail (a mesh the mode counts cannot split, an out-of-memory
kill, a timeout) are reported with their status and the tail of their
output, and the sweep continues.  Choose a horizon of a few tens of
steps at least, e.g. with ``--stop.max_wall_time`` so a row on more
nodes takes more steps in the same time.

Usage (from the repository root)::

    # what would run, with each layout's padding
    .venv/bin/python scripts/node_benchmark.py --target cpu --dry-run \
        --solver-args "--phys.system pipe --phys.re 2300 --geo.lz 20 \
            --res.nz 256 --res.nr 48 --res.ntheta 96 \
            --stop.max_sim_time 0.5 --outs.snapshot_save_initial False \
            --outs.snapshot_save_final False"
    # one node under Open MPI
    .venv/bin/python scripts/node_benchmark.py --target cpu \
        --ranks 16 32 64 128 --solver-args "..."
    # inside a Slurm allocation: 1-4 nodes, two grids per count
    .venv/bin/python scripts/node_benchmark.py --target cpu \
        --launcher srun --nodes 1 2 4 --tasks-per-node 128 \
        --np1 128 64 --repeats 2 --workdir "$PWD/bench" \
        --csv bench.csv --toml parameters.toml --solver-args "..."
    # the GPU node
    .venv/bin/python scripts/node_benchmark.py --target gpu --gpus 4 \
        --tiles 2,32 1,32 2,64 --precisions double single \
        --solver-args "..."
    # harness self-check on any machine (2 CPU ranks, tiny problem)
    .venv/bin/python scripts/node_benchmark.py --cpu-smoke
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solver_benchmark import (  # noqa: E402
    CIT_PATTERN,
    SPRHS_PATTERN,
    SPT_PATTERN,
    _echo_tail,
    _to_text,
)

REPO = Path(__file__).resolve().parent.parent
DNSJAX = REPO / ".venv" / "bin" / "dnsjax"

#: The solver's closing peak-memory lines.
PEAK_PATTERN = re.compile(r"Peak device memory: ([\d.]+) GiB")
HOST_PATTERN = re.compile(
    r"Peak host memory: ([\d.]+) GiB per rank .*?, ([\d.]+) GiB per node"
)
#: Start-up timestamps (``Alive at`` is per rank, on stderr).
_STAMP = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)"
ALIVE_PATTERN = re.compile(rf"Alive at {_STAMP}")
INIT_PATTERN = re.compile(rf"Distribution initialized at {_STAMP}")
START_PATTERN = re.compile(rf"Started timestepping at {_STAMP}")
FIRST_PATTERN = re.compile(rf"First iteration over at {_STAMP}")
COLLECTIVES_PATTERN = re.compile(r"CPU cross-process collectives: (\S+)")
OOM_PATTERN = re.compile(r"oom[-_ ]kill|oom killed|out of memory", re.I)

#: Open MPI's rank-per-core binding.
OMPI_BINDING = ["--map-by", "core", "--bind-to", "core"]
#: The ``srun`` flags every row carries.
SRUN_FIXED = ["--hint=nomultithread", "--kill-on-bad-exit=1"]

#: The ``--cpu-smoke`` problem: small enough for a laptop.
SMOKE_ARGS = (
    "--phys.system plane-couette --phys.re 330 --geo.lx 5 --geo.lz 5 "
    "--res.nx 16 --res.ny 17 --res.nz 16 --stop.max_sim_time 0.1 "
    "--outs.snapshot_save_initial False --outs.snapshot_save_final False"
)


@dataclass
class Row:
    """One launch of the solver and, after it ran, what it reported."""

    label: str
    cmd: list[str]
    nodes: int = 1
    ranks: int = 1
    tpn: int | None = None
    cpt: int | None = None
    dist: str = ""
    np0: int = 1
    np1: int = 1
    variant: str = ""
    exe: str = ""
    repeat: int = 0
    result: dict = field(default_factory=dict)


def _factorisations(n: int) -> list[tuple[int, int]]:
    """Every ``(np0, np1)`` with ``np0 * np1 == n``."""
    return [(a, n // a) for a in range(1, n + 1) if n % a == 0]


def _named(items: list[str], default: tuple[str, str]) -> list[tuple]:
    """``NAME=VALUE`` strings as pairs; *default* when none are given."""
    if not items:
        return [default]
    out = []
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=VALUE, got {item!r}")
        out.append((name, value))
    return out


def _layouts(n: int, np1_filter: list[str]) -> list[tuple[int, int]]:
    """The factorisations of *n* the ``--np1`` filter keeps."""
    pairs = _factorisations(n)
    if np1_filter and np1_filter != ["all"]:
        keep = {int(v) for v in np1_filter}
        pairs = [(a, b) for a, b in pairs if b in keep]
    return pairs


def _cpu_rows(a: argparse.Namespace) -> list[Row]:
    """The CPU sweep, ``mpirun`` or ``srun``, before repeats."""
    solver = shlex.split(a.solver_args)
    variants = _named(a.variant, ("base", ""))
    exes = _named(a.exe, ("dnsjax", str(DNSJAX)))
    rows = []
    if a.launcher == "mpirun":
        binding = shlex.split(a.mpirun_args) if a.mpirun_args else OMPI_BINDING
        for n in a.ranks:
            for (np0, np1), (vn, va), (en, ep) in itertools.product(
                _layouts(n, a.np1), variants, exes
            ):
                cmd = [
                    "mpirun",
                    *(["--oversubscribe"] if a.oversubscribe else []),
                    "-np",
                    str(n),
                    *binding,
                    ep,
                    *_dist_args(np0, np1),
                    *solver,
                    *shlex.split(va),
                ]
                label = f"{n:4d} ranks  ({np0:3d}, {np1:3d})"
                rows.append(
                    Row(
                        _arm_label(label, vn, en, variants, exes),
                        cmd,
                        ranks=n,
                        np0=np0,
                        np1=np1,
                        variant=vn,
                        exe=en,
                    )
                )
        return rows
    extra = shlex.split(a.srun_args) if a.srun_args else []
    for nodes, tpn, dist in itertools.product(
        a.nodes, a.tasks_per_node, a.distribution
    ):
        cpts = a.cpus_per_task or [max(a.cores_per_node // tpn, 1)]
        for cpt in cpts:
            n = nodes * tpn
            for (np0, np1), (vn, va), (en, ep) in itertools.product(
                _layouts(n, a.np1), variants, exes
            ):
                cmd = [
                    "srun",
                    f"--nodes={nodes}",
                    f"--ntasks={n}",
                    f"--ntasks-per-node={tpn}",
                    f"--cpus-per-task={cpt}",
                    f"--distribution={dist}",
                    *SRUN_FIXED,
                    *extra,
                    ep,
                    *_dist_args(np0, np1),
                    *solver,
                    *shlex.split(va),
                ]
                label = (
                    f"{nodes:3d}n {tpn:3d}t c{cpt} {_short(dist)} "
                    f"({np0:3d},{np1:3d})"
                )
                rows.append(
                    Row(
                        _arm_label(label, vn, en, variants, exes),
                        cmd,
                        nodes=nodes,
                        ranks=n,
                        tpn=tpn,
                        cpt=cpt,
                        dist=dist,
                        np0=np0,
                        np1=np1,
                        variant=vn,
                        exe=en,
                    )
                )
    return rows


def _dist_args(np0: int, np1: int) -> list[str]:
    return [
        "--dist.platform",
        "cpu",
        "--dist.np0",
        str(np0),
        "--dist.np1",
        str(np1),
    ]


def _short(dist: str) -> str:
    """``block:cyclic`` -> ``bc``, for the row label."""
    return "".join(part[:1] for part in dist.split(":"))


def _arm_label(label: str, vn: str, en: str, variants, exes) -> str:
    if len(variants) > 1:
        label += f" {vn}"
    if len(exes) > 1:
        label += f" [{en}]"
    return label


def _gpu_rows(a: argparse.Namespace) -> list[Row]:
    """The GPU sweep."""
    solver = shlex.split(a.solver_args)
    tiles = [tuple(int(x) for x in t.split(",")) for t in a.tiles] or [None]
    rows = []
    for (np0, np1), tile, prec in itertools.product(
        _factorisations(a.gpus), tiles, a.precisions
    ):
        extra = []
        if tile is not None:
            extra += [
                "--solver.pallas_block_m0",
                str(tile[0]),
                "--solver.pallas_block_m1",
                str(tile[1]),
            ]
        extra += ["--res.double_precision", str(prec == "double")]
        cmd = [
            str(DNSJAX),
            "--dist.platform",
            "cuda",
            "--dist.np0",
            str(np0),
            "--dist.np1",
            str(np1),
            *extra,
            *solver,
        ]
        tag = f" tile {tile[0]}x{tile[1]}" if tile else ""
        rows.append(
            Row(
                f"({np0}, {np1}){tag} {prec}",
                cmd,
                ranks=1,
                np0=np0,
                np1=np1,
            )
        )
    return rows


def _padding(a: argparse.Namespace, rows: list[Row]) -> dict:
    """What each layout pads, from the resolved parameters (JAX-free).

    Mirrors :mod:`dnsjax.sharding`: the wall-normal axis and the stored
    ``kz`` modes padded to multiples of ``np0``, the stored ``kx`` modes
    to multiples of ``np1``, the oversampled ``z`` rounded to an
    FFT-friendly multiple of ``np1``.  Returns ``{(np0, np1): text}``.
    """
    from dnsjax.bootstrap import resolve_parameters
    from dnsjax.flows.registry import periodic_systems
    from dnsjax.parameters import PaddedResolution, params, round_up_padded

    resolve_parameters(
        shlex.split(a.solver_args),
        toml_path=Path(a.toml) if a.toml else False,
    )
    p = params.model_copy(deep=True)
    nx, ny, nz = p.res.nx, p.res.ny, p.res.nz
    periodic = p.phys.system in periodic_systems

    def sizes(np0: int, np1: int) -> tuple[int, int, int, int]:
        q = p.model_copy(deep=True)
        q.dist.np0, q.dist.np1 = np0, np1
        pr = PaddedResolution()
        pr.set_padded_resolution(q)
        y = pr.ny_padded if periodic else round_up_padded(ny, np0)
        return (
            y,
            pr.nz_padded,
            round_up_padded(nz - 1, np0),
            (round_up_padded(nx // 2, np1)),
        )

    y1, z1, kz1, kx1 = sizes(1, 1)
    out = {}
    for row in rows:
        key = (row.np0, row.np1)
        if key in out:
            continue
        y, z, kz, kx = sizes(*key)
        out[key] = (
            f"y {y1}->{y}, z_pad {z1}->{z}, kz {kz1}->{kz}, "
            f"kx {kx1}->{kx}: physical x{y * z / (y1 * z1):.3f}, "
            f"spectral x{kz * kx / (kz1 * kx1):.3f}"
        )
    return out


def _stamp(pattern: re.Pattern, text: str, first: bool = True):
    hits = pattern.findall(text)
    if not hits:
        return None
    stamps = sorted(datetime.fromisoformat(h) for h in hits)
    return stamps[0] if first else stamps[-1]


def _parse(out: str, err: str, status: int) -> dict:
    """The run's closing summary and start-up phases."""
    rec: dict = {}
    for key, pat in (
        ("s_per_t", SPT_PATTERN),
        ("s_per_rhs", SPRHS_PATTERN),
        ("c_per_it", CIT_PATTERN),
        ("peak_device_gib", PEAK_PATTERN),
    ):
        hits = pat.findall(out)
        if hits:
            rec[key] = float(hits[-1])
    host = HOST_PATTERN.findall(out)
    if host:
        rec["host_rank_gib"] = float(host[-1][0])
        rec["host_node_gib"] = float(host[-1][1])
    coll = COLLECTIVES_PATTERN.findall(out)
    if coll:
        rec["collectives"] = coll[-1]
    alive = _stamp(ALIVE_PATTERN, err + out)
    init = _stamp(INIT_PATTERN, out)
    start = _stamp(START_PATTERN, out)
    first = _stamp(FIRST_PATTERN, out)
    if alive and init:
        rec["t_init_s"] = (init - alive).total_seconds()
    if init and start:
        rec["t_setup_s"] = (start - init).total_seconds()
    if start and first:
        rec["t_first_step_s"] = (first - start).total_seconds()
    if status == -1:
        rec["status"] = "timeout"
    elif status != 0:
        oom = OOM_PATTERN.search(err + out) is not None
        rec["status"] = "OOM" if oom else f"exit {status}"
    elif "s_per_t" not in rec:
        rec["status"] = "no summary"
    else:
        rec["status"] = "ok"
    return rec


def _run(row: Row, a: argparse.Namespace, index: int) -> dict:
    """Run one row in its own directory; return its parsed record."""
    if a.workdir:
        wd = Path(a.workdir) / f"row{index:03d}"
        wd.mkdir(parents=True, exist_ok=True)
        ctx = nullcontext(str(wd))
    else:
        ctx = tempfile.TemporaryDirectory(prefix="dnsjax_node_")
    with ctx as wd:
        if a.toml:
            shutil.copy(a.toml, Path(wd) / "parameters.toml")
        t0 = time.time()
        try:
            proc = subprocess.run(
                row.cmd,
                cwd=wd,
                capture_output=True,
                text=True,
                timeout=a.timeout,
            )
            out, err, status = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            out, err, status = _to_text(exc.stdout), _to_text(exc.stderr), -1
        t1 = time.time()
        if a.workdir:
            (Path(wd) / "out.log").write_text(out)
            (Path(wd) / "err.log").write_text(err)
    rec = _parse(out, err, status)
    rec.update(start_unix=round(t0, 1), end_unix=round(t1, 1), rowdir=wd)
    if rec["status"] != "ok":
        _echo_tail(" ".join(row.cmd[-8:]), out, err)
    return rec


def _nodelist() -> list[str]:
    """The allocation's hosts in Slurm order (empty outside Slurm)."""
    nodes = os.environ.get("SLURM_JOB_NODELIST")
    if not nodes or shutil.which("scontrol") is None:
        return []
    proc = subprocess.run(
        ["scontrol", "show", "hostnames", nodes],
        capture_output=True,
        text=True,
    )
    return proc.stdout.split() if proc.returncode == 0 else []


def _summarise(rows: list[Row], a: argparse.Namespace) -> None:
    """Print the table; efficiencies against the best smallest row."""
    srun = a.target == "cpu" and a.launcher == "srun"
    size = (lambda r: r.nodes) if srun else (lambda r: r.ranks)
    ok = [r for r in rows if r.result.get("status") == "ok"]
    ref = None
    if ok:
        n_ref = min(size(r) for r in ok)
        ref = n_ref * min(r.result["s_per_t"] for r in ok if size(r) == n_ref)
    head = f"\n{'layout':<44} {'s/t':>10} {'eff':>5}"
    if srun:
        head += f" {'CU/t':>8}"
    head += f" {'mem GiB':>13} {'start s':>8}  status"
    print(head)
    for r in rows:
        res = r.result
        spt = res.get("s_per_t")
        eff = f"{ref / (size(r) * spt):5.2f}" if spt and ref else "    -"
        line = f"{r.label:<44} "
        line += f"{spt:10.3e}" if spt else f"{'-':>10}"
        line += f" {eff}"
        if srun:
            line += f" {r.nodes * spt / 3600:8.2e}" if spt else f" {'-':>8}"
        if "host_rank_gib" in res:
            mem = f"{res['host_rank_gib']:.2f}/{res['host_node_gib']:.1f}"
        elif "peak_device_gib" in res:
            mem = f"{res['peak_device_gib']:.2f}"
        else:
            mem = "-"
        start = sum(
            res.get(k, 0.0)
            for k in ("t_init_s", "t_setup_s", "t_first_step_s")
        )
        line += f" {mem:>13} {start:8.1f}  {res.get('status', '-')}"
        print(line)
    if srun:
        print(
            "\neff: N0 T(N0) / (N T(N)) against the fastest row at the "
            "smallest node count;\nCU/t: node hours per unit of simulated "
            "time; mem: peak GiB per rank / per node;\nstart: process "
            "start to the end of the first (compiling) step."
        )


def _write_csv(rows: list[Row], path: str, hosts: list[str]) -> None:
    keys = [
        "label",
        "nodes",
        "ranks",
        "tasks_per_node",
        "cpus_per_task",
        "distribution",
        "np0",
        "np1",
        "variant",
        "exe",
        "repeat",
        "status",
        "s_per_t",
        "s_per_rhs",
        "c_per_it",
        "host_rank_gib",
        "host_node_gib",
        "peak_device_gib",
        "t_init_s",
        "t_setup_s",
        "t_first_step_s",
        "collectives",
        "padding",
        "start_unix",
        "end_unix",
        "rowdir",
        "nodelist",
        "command",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "label": r.label.strip(),
                    "nodes": r.nodes,
                    "ranks": r.ranks,
                    "tasks_per_node": r.tpn,
                    "cpus_per_task": r.cpt,
                    "distribution": r.dist,
                    "np0": r.np0,
                    "np1": r.np1,
                    "variant": r.variant,
                    "exe": r.exe,
                    "repeat": r.repeat,
                    "nodelist": ",".join(hosts[: r.nodes]) if hosts else "",
                    "command": shlex.join(r.cmd),
                    **{k: r.result.get(k, "") for k in keys if k in r.result},
                }
            )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--target", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--solver-args", default="", help="the problem")
    ap.add_argument(
        "--toml", default=None, help="copied into each run as parameters.toml"
    )
    ap.add_argument("--launcher", choices=("mpirun", "srun"), default="mpirun")
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--mpirun-args", default=None)
    ap.add_argument("--oversubscribe", action="store_true")
    ap.add_argument("--nodes", type=int, nargs="+", default=[1])
    ap.add_argument("--tasks-per-node", type=int, nargs="+", default=[128])
    ap.add_argument("--cores-per-node", type=int, default=128)
    ap.add_argument("--cpus-per-task", type=int, nargs="+", default=None)
    ap.add_argument("--distribution", nargs="+", default=["block:block"])
    ap.add_argument("--srun-args", default=None, help="extra srun flags")
    ap.add_argument(
        "--np1", nargs="+", default=["all"], help="np1 values to keep"
    )
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--exe", action="append", default=[])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workdir", default=None, help="keep run dirs here")
    ap.add_argument("--csv", default=None, help="write one row per run")
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument(
        "--precisions",
        nargs="+",
        choices=("double", "single"),
        default=["double"],
    )
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-smoke", action="store_true")
    a = ap.parse_args()

    if a.cpu_smoke:
        a.target, a.launcher, a.ranks, a.oversubscribe = (
            "cpu",
            "mpirun",
            [1, 2],
            True,
        )
        a.solver_args = a.solver_args or SMOKE_ARGS
        a.timeout = min(a.timeout, 600.0)
    if not a.solver_args and not a.toml:
        ap.error("--solver-args or --toml is required (the problem)")
    launcher = a.launcher if a.target == "cpu" else None
    if launcher and not (a.dry_run or shutil.which(launcher)):
        ap.error(f"{launcher} not on PATH")
    if a.target == "cpu" and not os.environ.get("MPITRAMPOLINE_LIB"):
        print(
            "note: MPITRAMPOLINE_LIB is unset, so multi-rank runs use "
            "gloo collectives (docs/cpu-collectives.md)."
        )

    base = _cpu_rows(a) if a.target == "cpu" else _gpu_rows(a)
    pads = _padding(a, base) if a.target == "cpu" else {}
    if a.dry_run:
        for row in base:
            print(f"{row.label}: {shlex.join(row.cmd)}")
            if (row.np0, row.np1) in pads:
                print(f"    padding: {pads[(row.np0, row.np1)]}")
        return 0

    rows = []
    for rep in range(a.repeats):
        order = base if rep % 2 == 0 else base[::-1]
        for row in order:
            rows.append(Row(**{**row.__dict__, "repeat": rep, "result": {}}))
    hosts = _nodelist()
    for i, row in enumerate(rows):
        print(f"== [{i + 1}/{len(rows)}] {row.label}", flush=True)
        row.result = _run(row, a, i)
        if (row.np0, row.np1) in pads:
            row.result["padding"] = pads[(row.np0, row.np1)]
        res = row.result
        print(
            f"   {res['status']}"
            + (f", {res['s_per_t']:.3e} s/t" if "s_per_t" in res else "")
            + (
                f", {res['host_node_gib']:.0f} GiB per node"
                if "host_node_gib" in res
                else ""
            ),
            flush=True,
        )
    _summarise(rows, a)
    if a.csv:
        _write_csv(rows, a.csv, hosts)
        print(f"\nwrote {a.csv}")
    return 0 if all(r.result.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
