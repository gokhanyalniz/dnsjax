#!/usr/bin/env python3
r"""Per-node memory sampler for multi-process runs (JAX-free).

An out-of-memory kill is decided on a node's total, at one moment, and
it takes the run's closing diagnostics with it.  This sampler runs
beside the solver, one instance per node, and writes what the node
held every ``--interval`` seconds to ``memwatch_<host>.csv``, flushed
per sample, so the record survives the kill it explains:

- ``MemTotal`` and ``MemAvailable`` (``/proc/meminfo``): the node as
  the kernel sees it, page cache excluded;
- the job's cgroup usage and its peak, where the kernel exposes them
  (cgroup v1 or v2; the level above the job's step directories when
  ``SLURM_JOB_ID`` is set, else the sampler's own cgroup) -- the figure
  an out-of-memory kill is decided on;
- the solver processes on the node -- ``dnsjax`` / ``dnsjax-twin``
  console scripts and ``python -m dnsjax`` -- with the sum and maximum
  of their resident set, and the sum of their proportional set
  (``Pss``, shared pages split between their users, read every
  ``--pss-every`` samples since it costs more).

``summary`` reads the files back: per node, the peak and when it
happened, relative to the idle baseline before the solver started; with
``--rows`` (the CSV ``node_benchmark.py --csv`` writes, which records
each run's start and end time) also the peak within each run, so an
out-of-memory failure is placed against its layout and, through the
solver's own start-up timestamps, its phase.

Usage, inside a Slurm job (``--overlap`` lets the sampler share the
allocation with the runs it watches)::

    srun --overlap --nodes="$SLURM_NNODES" --ntasks-per-node=1 \
        .venv/bin/python scripts/memory_watch.py sample --out mem &
    ...                        # the runs (srun, node_benchmark.py, ...)
    touch mem/stop; wait       # every sampler exits at its next sample
    .venv/bin/python scripts/memory_watch.py summary mem --rows bench.csv

Locally (any Linux machine): start ``sample`` in the background, run
the solver, then ``touch`` the stop file.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import socket
import sys
import time
from pathlib import Path

GIB = 2**30
KIB = 1024

#: Columns of a ``memwatch_<host>.csv``.
COLUMNS = (
    "unix",
    "mem_total",
    "mem_available",
    "cgroup_current",
    "cgroup_peak",
    "n_proc",
    "rss_sum",
    "rss_max",
    "pss_sum",
)

#: Executables whose processes count as the solver's.
SOLVERS = ("dnsjax", "dnsjax-twin")


def _meminfo() -> tuple[int, int]:
    """``(MemTotal, MemAvailable)`` in bytes."""
    vals = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                vals[key] = int(rest.split()[0]) * KIB
    return vals.get("MemTotal", 0), vals.get("MemAvailable", 0)


def _cgroup_files() -> tuple[Path | None, Path | None]:
    """The (current, peak) usage files of the job's memory cgroup.

    The job level is the path up to its ``job_<SLURM_JOB_ID>``
    component (Slurm's cgroup plugins name it so in both hierarchies);
    outside Slurm, the sampler's own cgroup.  ``None`` where a file does
    not exist (``memory.peak`` needs Linux 5.19 under cgroup v2).
    """
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None, None
    job = os.environ.get("SLURM_JOB_ID")

    def job_level(path: str) -> str:
        if job:
            parts = path.split("/")
            for i, part in enumerate(parts):
                if part == f"job_{job}":
                    return "/".join(parts[: i + 1])
        return path

    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # cgroup v2
            base = Path("/sys/fs/cgroup") / job_level(path).lstrip("/")
            cur, peak = base / "memory.current", base / "memory.peak"
        elif "memory" in controllers.split(","):  # cgroup v1
            base = Path("/sys/fs/cgroup/memory") / job_level(path).lstrip("/")
            cur = base / "memory.usage_in_bytes"
            peak = base / "memory.max_usage_in_bytes"
        else:
            continue
        return (
            cur if cur.is_file() else None,
            peak if peak.is_file() else None,
        )
    return None, None


def _read_int(path: Path | None) -> int | str:
    if path is None:
        return ""
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return ""


def _is_solver(argv: list[str]) -> bool:
    """A ``dnsjax`` console script or ``python -m dnsjax[.twin]``."""
    names = [os.path.basename(a) for a in argv[:3]]
    if any(n in SOLVERS for n in names):
        return True
    return len(argv) > 2 and argv[1] == "-m" and argv[2].startswith("dnsjax")


def _solver_procs(pss: bool) -> tuple[int, int, int, int | str]:
    """``(count, rss_sum, rss_max, pss_sum)`` of this user's solvers."""
    me, uid = os.getpid(), os.getuid()
    n = rss_sum = rss_max = pss_sum = 0
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            raw = Path(entry.path, "cmdline").read_bytes()
            if not _is_solver(raw.decode(errors="replace").split("\0")):
                continue
            rss = 0
            for line in Path(entry.path, "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * KIB
                    break
            if pss:
                rollup = Path(entry.path, "smaps_rollup").read_text()
                for line in rollup.splitlines():
                    if line.startswith("Pss:"):
                        pss_sum += int(line.split()[1]) * KIB
                        break
        except (OSError, ValueError):
            continue  # the process ended between the listing and the read
        n += 1
        rss_sum += rss
        rss_max = max(rss_max, rss)
    return n, rss_sum, rss_max, (pss_sum if pss else "")


def sample(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stop_file = Path(args.stop_file) if args.stop_file else out / "stop"
    path = out / f"memwatch_{socket.gethostname()}.csv"
    cur_file, peak_file = _cgroup_files()
    stopping = []

    def _stop(signum, frame):
        stopping.append(signum)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    t_end = time.time() + args.duration if args.duration else None
    k = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        while not stopping and not stop_file.exists():
            if t_end is not None and time.time() > t_end:
                break
            total, avail = _meminfo()
            n, rss_sum, rss_max, pss_sum = _solver_procs(
                k % args.pss_every == 0
            )
            w.writerow(
                (
                    f"{time.time():.1f}",
                    total,
                    avail,
                    _read_int(cur_file),
                    _read_int(peak_file),
                    n,
                    rss_sum,
                    rss_max,
                    pss_sum,
                )
            )
            fh.flush()
            k += 1
            time.sleep(args.interval)
    return 0


def _load(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            rows.append(
                {k: (float(v) if v != "" else None) for k, v in rec.items()}
            )
    return rows


def _gib(x: float | None) -> str:
    return f"{x / GIB:7.2f}" if x is not None else f"{'-':>7}"


def summary(args: argparse.Namespace) -> int:
    files = sorted(Path(args.dir).glob("memwatch_*.csv"))
    if not files:
        print(f"no memwatch_*.csv in {args.dir}")
        return 1
    nodes = {}
    for f in files:
        data = _load(f)
        if data:
            nodes[f.stem.removeprefix("memwatch_")] = data
    print(
        "Per node (GiB): used = MemTotal - MemAvailable; base = the "
        "least used\nwhile no solver ran; cgroup = the job's usage "
        "peak; pss/rss = summed\nover the solver processes; rss1 = the "
        "largest single process.\n"
    )
    print(
        f"{'node':<16} {'total':>7} {'base':>7} {'used':>7} {'cgroup':>7} "
        f"{'pss':>7} {'rss':>7} {'rss1':>7} {'procs':>5}  peak at"
    )
    for host, data in nodes.items():
        idle = [
            r["mem_total"] - r["mem_available"]
            for r in data
            if not r["n_proc"]
        ]
        used = [r["mem_total"] - r["mem_available"] for r in data]
        i_peak = max(range(len(data)), key=lambda i: used[i])
        cg = [
            v
            for r in data
            for v in (r["cgroup_peak"], r["cgroup_current"])
            if v is not None
        ]
        pss = [r["pss_sum"] for r in data if r["pss_sum"] is not None]
        when = time.strftime("%H:%M:%S", time.localtime(data[i_peak]["unix"]))
        print(
            f"{host:<16} {_gib(data[0]['mem_total'])} "
            f"{_gib(min(idle) if idle else None)} {_gib(used[i_peak])} "
            f"{_gib(max(cg) if cg else None)} "
            f"{_gib(max(pss) if pss else None)} "
            f"{_gib(max(r['rss_sum'] for r in data))} "
            f"{_gib(max(r['rss_max'] for r in data))} "
            f"{int(max(r['n_proc'] for r in data)):5d}  {when}"
        )
    if not args.rows:
        return 0
    print(
        "\nPer run (the largest node; GiB above that node's idle base, "
        "and the largest\nsingle process):\n"
    )
    print(f"{'run':<52} {'status':>10} {'node':>7} {'rss1':>7}")
    with open(args.rows, newline="") as fh:
        for run in csv.DictReader(fh):
            t0, t1 = float(run["start_unix"]), float(run["end_unix"])
            best, rss1 = None, None
            for data in nodes.values():
                idle = [
                    r["mem_total"] - r["mem_available"]
                    for r in data
                    if not r["n_proc"]
                ]
                base = min(idle) if idle else 0.0
                window = [r for r in data if t0 <= r["unix"] <= t1 + 2]
                if not window:
                    continue
                used = max(r["mem_total"] - r["mem_available"] for r in window)
                if best is None or used - base > best:
                    best = used - base
                top = max(r["rss_max"] for r in window)
                rss1 = top if rss1 is None else max(rss1, top)
            print(
                f"{run['label'][:52]:<52} {run.get('status', ''):>10} "
                f"{_gib(best)} {_gib(rss1)}"
            )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample", help="sample this node until stopped")
    sp.add_argument("--out", required=True, help="directory for the CSV")
    sp.add_argument("--interval", type=float, default=1.0)
    sp.add_argument(
        "--pss-every",
        type=int,
        default=5,
        help="read Pss every N samples (it costs more than RSS)",
    )
    sp.add_argument("--stop-file", default=None, help="default: <out>/stop")
    sp.add_argument(
        "--duration", type=float, default=None, help="stop after N s"
    )
    sm = sub.add_parser("summary", help="summarise a sample directory")
    sm.add_argument("dir")
    sm.add_argument(
        "--rows", default=None, help="node_benchmark.py --csv output"
    )
    args = ap.parse_args()
    return sample(args) if args.cmd == "sample" else summary(args)


if __name__ == "__main__":
    sys.exit(main())
