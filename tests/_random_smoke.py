r"""The runner of the random-IC smoke tests, core and lab.

Each script's table (``tests/test_random_smoke.py``,
``lab/tests/test_random_smoke.py``) holds its rows and its account of
them; :func:`main` runs one.  Per row, beyond :mod:`_smoke`'s criteria:
the exit code is 0, and on CPU the closing ``Peak host memory`` and
``Resident host memory`` lines are there and count every rank (their
all-gather is a collective each rank must reach).

Row keys this runner reads, beyond ``name`` and ``args``: ``force_np``
/ ``force_np0`` / ``res`` / ``oversubscribe`` / ``max_sim_time`` /
``it_stats`` / ``check_laminarization`` (overriding the suite
defaults), ``force_dt`` and ``slack_dt`` (the step and the end-of-run
``t`` slack), ``omit_random_flag`` (the default start mode),
``expect_pattern``, ``expect_nonfinite_abort``, ``check_error_max`` and
``lowres``.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

from _live import run_live
from _smoke import (
    ERR_PATTERN,
    check_run,
    res_flags,
    run_rows,
)


def _build_command(
    system: dict, args: argparse.Namespace, dt: float
) -> list[str]:
    """Build the ``mpirun ... -m dnsjax`` command for one system.

    Per-system ``force_np`` / ``force_np0`` / ``res`` /
    ``oversubscribe`` / ``max_sim_time`` / ``it_stats`` /
    ``check_laminarization`` override the suite defaults (used by the
    multi-device padded entries and the nan-guard entry); ``lowres``
    adds the ``[lowres]`` flags (:func:`_check_lowres`).
    """
    np_count = system.get("force_np", args.np)
    np0 = system.get("force_np0", args.np0)
    res = system.get("res", {})
    nx = str(res.get("nx", args.res))
    ny = str(res.get("ny", args.res))
    nz = str(res.get("nz", args.res))
    flags = res_flags(system["args"], nx, ny, nz)
    max_sim_time = system.get("max_sim_time", args.max_sim_time)
    it_stats = system.get("it_stats", args.it_stats)

    base = ["mpirun"]
    if system.get("oversubscribe"):
        base.append("--oversubscribe")
    base += [
        "-np",
        str(np_count),
        sys.executable,
        "-m",
        "dnsjax",
        "--dist.platform",
        args.platform,
        "--dist.np0",
        str(np0),
        "--dist.np1",
        str(np_count // np0),
    ]
    # In-process random divergence-free IC (no snapshot file).  The
    # default-IC entry omits this flag to verify that random is the
    # default start mode when nothing else is selected.
    if not system.get("omit_random_flag"):
        base += ["--init.random_field", "True"]
    base += [
        "--init.random_amplitude",
        str(args.amplitude),
        "--init.random_smoothness",
        str(args.smoothness),
        "--init.random_seed",
        str(args.seed),
        *flags,
        "--step.dt",
        str(dt),
        "--stop.max_sim_time",
        str(max_sim_time),
        "--outs.it_stats",
        str(it_stats),
        # Laminarization check off unless the entry keeps it on: a
        # decaying transient must not cut the run short before
        # max_sim_time.  The multi-process entries keep it on (their
        # E' stays decades above the threshold), because its E' read
        # is a jitted program only a real multi-process run can fail
        # to trace.
        "--stop.check_laminarization",
        str(system.get("check_laminarization", False)),
    ]
    lowres = system.get("lowres")
    if lowres:
        base += ["--lowres.it_lowres", str(lowres["it_lowres"])]
        for axis, value in lowres["res"].items():
            base += [f"--lowres.{axis}", str(value)]
        if lowres["pressure"]:
            base += ["--lowres.pressure", "True"]
    return base + system["args"]


def _check_lowres(
    workdir: Path, system: dict, max_sim_time: float, dt: float
) -> str:
    """Read a run's ``[lowres]`` files back (the entry's ``lowres``).

    One file per aligned step, the initial condition included; each
    with the reduced resolution recorded under the flow's public names,
    the reduced stored shape, finite and non-zero, and a ``pressure/``
    member exactly when the entry expects one.  Written by a
    multi-process run, so a global array baked into one of the
    writer's jitted programs fails here.
    """
    import numpy as np

    from dnsjax.analysis import read_pressure, read_state
    from dnsjax.snapshot_meta import read_snapshot_meta

    name, spec = system["name"], system["lowres"]
    files = sorted((workdir / "lowres").glob("lowres_*.tar"))
    its = [int(f.stem.rsplit("_", 1)[1]) for f in files]
    expected = list(range(0, round(max_sim_time / dt) + 1, spec["it_lowres"]))
    if its != expected:
        raise AssertionError(
            f"{name}: lowres files at steps {its}, expected {expected}"
        )
    for f in files:
        meta = read_snapshot_meta(f)
        stored = {k: meta["params"]["res"][k] for k in spec["res"]}
        if stored != spec["res"]:
            raise AssertionError(f"{name}: {f.name} records res {stored}")
        fields = np.stack(
            read_state(f, return_physical=False, return_spectral=True).spectral
        )
        if fields.shape != spec["shape"]:
            raise AssertionError(
                f"{name}: {f.name} holds {fields.shape}, "
                f"expected {spec['shape']}"
            )
        if not np.isfinite(fields).all() or not np.abs(fields).max() > 0:
            raise AssertionError(f"{name}: {f.name} is non-finite or zero")
        if bool(meta.get("pressure")) != spec["pressure"]:
            raise AssertionError(
                f"{name}: {f.name} pressure member "
                f"{bool(meta.get('pressure'))}, expected {spec['pressure']}"
            )
        if spec["pressure"]:
            p = read_pressure(
                f, return_physical=False, return_spectral=True
            ).spectral[0]
            if p.shape != spec["shape"][1:] or not np.isfinite(p).all():
                raise AssertionError(
                    f"{name}: {f.name} pressure {p.shape} or non-finite"
                )
    return f"{len(files)} lowres files"


def _check_error_max(corrector_dat: Path, summary: str, name: str) -> None:
    """The closing ``err =`` covers every step since the last check.

    Compared at the summary's printed precision: the first
    ``corrector.dat`` row (step 1's error) rounded the same way must not
    exceed it.
    """
    rows = [
        ln.split()
        for ln in corrector_dat.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not rows:
        raise AssertionError(f"{name}: corrector.dat has no rows")
    first = float(f"{float(rows[0][2]):.3e}")
    err_match = ERR_PATTERN.search(summary)
    if err_match is None:
        raise AssertionError(f"{name}: no err on the summary line")
    closing = float(err_match.group(1))
    if closing < first:
        raise AssertionError(
            f"{name}: closing err {closing:.3e} below step 1's "
            f"{first:.3e} -- the check dropped an earlier step's error"
        )


# ── test runner ──────────────────────────────────────────────────────


def run_smoke_test(system: dict, args: argparse.Namespace) -> None:
    """Run a single random-IC smoke test (in a fresh directory).

    Further per-entry keys: ``expect_pattern`` (regex that must match
    the run's stdout, e.g. the adaptive-dt "[adaptive]" change line)
    and ``slack_dt`` (the end-of-run ``t`` slack for runs whose step
    grows -- the adaptive entries pass their ``dt_max``).
    """
    name = system["name"]
    # ``force_dt`` overrides the global --dt outright (the adaptive
    # entries start from their own dt; the nan-guard entry needs a dt
    # *larger* than any sensible default).
    dt = system.get("force_dt", args.dt)
    cmd = _build_command(system, args, dt)

    with tempfile.TemporaryDirectory(prefix=f"rand_{name}_") as workdir:
        result = run_live(cmd, timeout=args.timeout, cwd=workdir)

        if system.get("expect_nonfinite_abort"):
            # Inverted criteria: the run must have aborted itself via
            # the non-finite guard -- exit code 3 (mpirun -np 1
            # propagates the child's code) and the FATAL diagnostic.
            fatal_lines = [
                line
                for line in result.stdout.splitlines()
                if "FATAL: non-finite" in line
            ]
            if result.returncode != 3 or not fatal_lines:
                print(f"  FAIL  {name}: exit code {result.returncode}")
                print(
                    result.stdout[-2000:] if result.stdout else "(no stdout)"
                )
                print(
                    result.stderr[-2000:] if result.stderr else "(no stderr)"
                )
                raise AssertionError(
                    f"{name}: expected the non-finite guard abort (exit 3 "
                    f"+ 'FATAL: non-finite ...'), got exit "
                    f"{result.returncode} with {len(fatal_lines)} FATAL "
                    "line(s)"
                )
            print(f"  PASS  {name}  ({fatal_lines[0].strip()})")
            return

        if result.returncode != 0:
            print(f"  FAIL  {name}: exit code {result.returncode}")
            print(result.stdout[-2000:] if result.stdout else "(no stdout)")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
            raise AssertionError(
                f"{name} exited with code {result.returncode}"
            )

        # Default-IC entry: confirm the run actually took the random
        # branch (no snapshot, no explicit mode => random default).
        if system.get("omit_random_flag") and (
            "Started from an in-process random IC" not in result.stdout
        ):
            raise AssertionError(
                f"{name}: default start mode did not select the random IC "
                "('Started from an in-process random IC' missing from stdout)"
            )

        summary = check_run(
            result.stdout,
            name,
            system.get("max_sim_time", args.max_sim_time),
            system.get("slack_dt", dt),
        )

        if args.platform == "cpu":
            n_ranks = system.get("force_np", args.np)
            for line in (
                "Peak host memory",
                "Resident host memory at the end",
            ):
                mem = re.search(
                    rf"{line}: [\d.]+ GiB per rank \(max of (\d+)[,)]",
                    result.stdout,
                )
                if mem is None or int(mem.group(1)) != n_ranks:
                    raise AssertionError(
                        f"{name}: no '{line}' line over {n_ranks} "
                        "rank(s) in stdout"
                    )

        pattern = system.get("expect_pattern")
        if pattern and re.search(pattern, result.stdout) is None:
            raise AssertionError(
                f"{name}: expected pattern {pattern!r} not found in stdout"
            )
        if system.get("check_error_max"):
            _check_error_max(Path(workdir) / "corrector.dat", summary, name)
        if system.get("lowres"):
            read_back = _check_lowres(
                Path(workdir),
                system,
                system.get("max_sim_time", args.max_sim_time),
                dt,
            )
            summary = f"{summary.strip()}; {read_back}"

    print(f"  PASS  {name}  ({summary.strip()})")


# ── main ─────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-IC smoke tests",
    )
    parser.add_argument(
        "--np", type=int, default=1, help="Number of devices (mpirun -np)"
    )
    parser.add_argument(
        "--np0",
        type=int,
        default=1,
        help="np0 mesh axis (wall-normal / kz split)",
    )
    parser.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend forwarded to each `python -m dnsjax` child "
        "(default cpu).  Use cuda to run the suite on GPU(s).",
    )
    parser.add_argument(
        "--res", type=int, default=32, help="Cubic resolution nx=ny=nz"
    )
    parser.add_argument("--max-sim-time", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--amplitude", type=float, default=0.1)
    # The model default, so the smoke test drives the initial
    # condition users actually get.  The random IC's *wall-normal*
    # shaping (``init.random_wall_smoothness`` /
    # ``random_wall_confinement``) is not a flag here: it is
    # wall-bounded-only, so a shared flag would have to be withheld
    # from the kolmogorov row, and taking the model default covers
    # every row at once.
    parser.add_argument("--smoothness", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--it-stats", type=int, default=10)
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Subset of system names to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-system subprocess timeout in seconds",
    )
    return parser.parse_args()


def main(systems: list[dict]) -> None:
    """Run the smoke test over *systems* and exit with its verdict."""
    sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args()
    print(
        f"Random-IC smoke tests on platform '{args.platform}' via "
        f"mpirun (default -np {args.np}, np0={args.np0}; some entries "
        "force their own device count).  Each child prints its own "
        "device banner.",
        flush=True,
    )
    sys.exit(run_rows(systems, run_smoke_test, args))
