#!/usr/bin/env python3
r"""Layout sweep for one production run on one node (JAX-free driver).

Before a production campaign on a new node, the questions are which
process/device layout makes a step cheapest and what it costs in
memory.  This driver launches the ordinary solver
(``.venv/bin/dnsjax``) on one fixed problem over a matrix of layouts,
parses each run's closing summary (``s/t``, and the peak device memory
where the backend reports it), and prints one table.  It launches, it
does not configure: the problem -- system, resolution, horizon -- is
whatever ``--solver-args`` says, so the sweep measures the run you
intend to make.

Targets
-------
``--target cpu`` (a many-core node, e.g. 2 x 64-core AMD EPYC 7742):
    one MPI rank per core, each pinned to one XLA thread (the solver's
    own ``NPROC=1`` default), launched as ``mpirun -np N --map-by core
    --bind-to core`` (Open MPI; ``--mpirun-args`` replaces the binding
    flags for another MPI).  Sweeps the rank count (``--ranks``) and,
    at each, every ``(np0, np1)`` factorisation.  Reports ``s/t`` and
    the strong-scaling efficiency against the fastest layout of the
    smallest rank count.  Export ``MPITRAMPOLINE_LIB`` first
    (``docs/cpu-collectives.md``): without it the collectives run over
    ``gloo``, which is not what a production run should measure.
``--target gpu`` (e.g. 4 x NVIDIA H200 in one node):
    one process spanning every visible GPU (the launch the
    ``Distribution`` docstring recommends for a single node); sweeps the
    mesh (every factorisation of ``--gpus``), optionally the Pallas
    tile (``--tiles``) and the precision (``--precisions``).  Reports
    ``s/t`` and the peak device memory.

Rows that fail (a mesh the mode counts cannot split, an out-of-memory
tile) are reported with their exit status and the tail of their
output, and the sweep continues.  Each row runs in its own scratch
directory (the solver writes its ``.dat`` files there).  Choose a
horizon of a few tens of steps at least: the summary's clock starts
after the first, compiling, step.

Usage (from the repository root)::

    # what would run
    .venv/bin/python scripts/node_benchmark.py --target cpu --dry-run \
        --solver-args "--phys.system pipe --phys.re 2300 --geo.lz 20 \
            --res.nz 256 --res.nr 48 --res.ntheta 96 \
            --stop.max_sim_time 0.5 --outs.snapshot_save_initial False \
            --outs.snapshot_save_final False"
    # the CPU node
    .venv/bin/python scripts/node_benchmark.py --target cpu \
        --ranks 16 32 64 128 --solver-args "..."
    # the GPU node
    .venv/bin/python scripts/node_benchmark.py --target gpu --gpus 4 \
        --tiles 2,32 1,32 2,64 --precisions double single \
        --solver-args "..."
    # harness self-check on any machine (2 CPU ranks, tiny problem)
    .venv/bin/python scripts/node_benchmark.py --cpu-smoke
"""

from __future__ import annotations

import argparse
import itertools
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solver_benchmark import SPT_PATTERN, _echo_tail, _to_text  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DNSJAX = REPO / ".venv" / "bin" / "dnsjax"

#: The solver's closing peak-memory line (GPU backends only).
PEAK_PATTERN = re.compile(r"Peak device memory: ([\d.]+) GiB")

#: Open MPI's rank-per-core binding.
OMPI_BINDING = ["--map-by", "core", "--bind-to", "core"]

#: The ``--cpu-smoke`` problem: small enough for a laptop.
SMOKE_ARGS = (
    "--phys.system plane-couette --phys.re 330 --geo.lx 5 --geo.lz 5 "
    "--res.nx 16 --res.ny 17 --res.nz 16 --stop.max_sim_time 0.1 "
    "--outs.snapshot_save_initial False --outs.snapshot_save_final False"
)


def _factorisations(n: int) -> list[tuple[int, int]]:
    """Every ``(np0, np1)`` with ``np0 * np1 == n``."""
    return [(a, n // a) for a in range(1, n + 1) if n % a == 0]


def _run(
    cmd: list[str], timeout: float
) -> tuple[float | None, float | None, str]:
    """Run one layout; return ``(s_per_t, peak_gib, status)``."""
    with tempfile.TemporaryDirectory(prefix="dnsjax_node_") as wd:
        try:
            proc = subprocess.run(
                cmd, cwd=wd, capture_output=True, text=True, timeout=timeout
            )
            out, err, status = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            out, err, status = _to_text(exc.stdout), _to_text(exc.stderr), -1
    spt = SPT_PATTERN.findall(out)
    peak = PEAK_PATTERN.findall(out)
    if status != 0 or not spt:
        _echo_tail(" ".join(cmd[-8:]), out, err)
        label = "timeout" if status == -1 else f"exit {status}"
        return None, None, label if status else "no summary"
    return float(spt[-1]), (float(peak[-1]) if peak else None), "ok"


def _cpu_rows(a: argparse.Namespace) -> list[tuple[str, int, list[str]]]:
    """``(label, n_ranks, command)`` for the CPU sweep."""
    binding = shlex.split(a.mpirun_args) if a.mpirun_args else OMPI_BINDING
    solver = shlex.split(a.solver_args)
    rows = []
    for n in a.ranks:
        for np0, np1 in _factorisations(n):
            cmd = [
                "mpirun",
                *(["--oversubscribe"] if a.oversubscribe else []),
                "-np",
                str(n),
                *binding,
                str(DNSJAX),
                "--dist.platform",
                "cpu",
                "--dist.np0",
                str(np0),
                "--dist.np1",
                str(np1),
                *solver,
            ]
            rows.append((f"{n:4d} ranks  ({np0:3d}, {np1:3d})", n, cmd))
    return rows


def _gpu_rows(a: argparse.Namespace) -> list[tuple[str, int, list[str]]]:
    """``(label, n_devices, command)`` for the GPU sweep."""
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
        rows.append((f"({np0}, {np1}){tag} {prec}", a.gpus, cmd))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--target", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--solver-args", default="", help="the problem")
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--mpirun-args", default=None)
    ap.add_argument("--oversubscribe", action="store_true")
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
        a.target, a.ranks, a.oversubscribe = "cpu", [1, 2], True
        a.solver_args = a.solver_args or SMOKE_ARGS
        a.timeout = min(a.timeout, 600.0)
    if not a.solver_args:
        ap.error("--solver-args is required (the problem to measure)")
    if a.target == "cpu" and not (a.dry_run or shutil.which("mpirun")):
        ap.error("mpirun not on PATH")
    if a.target == "cpu" and not os.environ.get("MPITRAMPOLINE_LIB"):
        print(
            "note: MPITRAMPOLINE_LIB is unset, so multi-rank runs use "
            "gloo collectives (docs/cpu-collectives.md)."
        )

    rows = _cpu_rows(a) if a.target == "cpu" else _gpu_rows(a)
    if a.dry_run:
        for label, _n, cmd in rows:
            print(f"{label}: {shlex.join(cmd)}")
        return 0

    results = []
    for label, n, cmd in rows:
        print(f"== {label}", flush=True)
        spt, peak, status = _run(cmd, a.timeout)
        results.append((label, n, spt, peak, status))

    ok = [(n, spt) for _l, n, spt, _p, s in results if s == "ok"]
    ref = None
    if ok:
        n_ref = min(n for n, _ in ok)
        ref = n_ref * min(spt for n, spt in ok if n == n_ref)
    print(f"\n{'layout':<34} {'s/t':>11} {'eff':>6} {'peak GiB':>9}  status")
    for label, n, spt, peak, status in results:
        eff = f"{ref / (n * spt):6.2f}" if spt and ref else "     -"
        spt_s = f"{spt:11.4e}" if spt else f"{'-':>11}"
        peak_s = f"{peak:9.2f}" if peak is not None else f"{'-':>9}"
        print(f"{label:<34} {spt_s} {eff} {peak_s}  {status}")
    return 0 if all(r[4] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
