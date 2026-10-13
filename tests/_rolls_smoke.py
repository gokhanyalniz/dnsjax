r"""The runner of the localized-rolls smoke tests, core and lab.

Each script's table (``tests/test_rolls_smoke.py``,
``lab/tests/test_rolls_smoke.py``) holds its rows and its account of
them; :func:`main` runs one, each row from the deterministic
localized-rolls IC (the in-process ``init.localized_rolls`` start mode,
no snapshot file), judged by the exit code and :mod:`_smoke`'s
criteria.  Row keys beyond ``name`` and ``args``: ``force_np`` /
``force_np0`` / ``res`` / ``oversubscribe`` (overriding the suite
defaults) and ``expect_pattern``.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile

from _live import run_live
from _smoke import check_run, res_flags, run_rows


def _build_command(
    system: dict, args: argparse.Namespace, dt: float
) -> list[str]:
    """Build the ``mpirun ... -m dnsjax`` command for one system.

    Per-system ``force_np`` / ``force_np0`` / ``res`` / ``oversubscribe``
    override the suite defaults (used by the multi-device padded entry).
    """
    np_count = system.get("force_np", args.np)
    np0 = system.get("force_np0", args.np0)
    res = system.get("res", {})
    nx = str(res.get("nx", args.res))
    ny = str(res.get("ny", args.res))
    nz = str(res.get("nz", args.res))
    flags = res_flags(system["args"], nx, ny, nz)

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
        # In-process deterministic localized-rolls IC (no snapshot file).
        "--init.localized_rolls",
        "True",
        "--init.localized_rolls_amplitude",
        str(args.amplitude),
        "--init.localized_rolls_width",
        str(args.width),
        "--init.localized_rolls_wavelength",
        str(args.wavelength),
        *flags,
        "--step.dt",
        str(dt),
        "--stop.max_sim_time",
        str(args.max_sim_time),
        "--outs.it_stats",
        str(args.it_stats),
        # Laminarization check off: a decaying transient must not cut
        # the run short before max_sim_time.
        "--stop.check_laminarization",
        "False",
    ]
    return base + system["args"]


# ── test runner ──────────────────────────────────────────────────────


def run_smoke_test(system: dict, args: argparse.Namespace) -> None:
    """Run a single localized-rolls smoke test (in a fresh directory).

    An entry's optional ``expect_pattern`` is a regex the run's stdout
    must match (e.g. a total-field flow's ``t = 0`` flux).
    """
    name = system["name"]
    dt = args.dt
    cmd = _build_command(system, args, dt)

    with tempfile.TemporaryDirectory(prefix=f"rolls_{name}_") as workdir:
        result = run_live(cmd, timeout=args.timeout, cwd=workdir)

        if result.returncode != 0:
            print(f"  FAIL  {name}: exit code {result.returncode}")
            print(result.stdout[-2000:] if result.stdout else "(no stdout)")
            print(result.stderr[-2000:] if result.stderr else "(no stderr)")
            raise AssertionError(
                f"{name} exited with code {result.returncode}"
            )

        summary = check_run(result.stdout, name, args.max_sim_time, dt)
        pattern = system.get("expect_pattern")
        if pattern is not None and re.search(pattern, result.stdout) is None:
            raise AssertionError(f"{name}: no match for {pattern!r}")

    print(f"  PASS  {name}  ({summary.strip()})")


# ── main ─────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Localized-rolls IC smoke tests (wall-bounded flows)",
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
    # Shorter than test_random_smoke's t = 1.0 on purpose: the unique
    # content here is the rolls IC *through* the solver (construction
    # + the first stretch of nonlinear evolution); the long-horizon
    # integration of the same flows is the random smoke's job.
    parser.add_argument("--max-sim-time", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--width", type=float, default=1.5)
    parser.add_argument("--wavelength", type=float, default=4.0)
    parser.add_argument("--it_stats", type=int, default=10)
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
        f"Localized-rolls smoke tests on platform '{args.platform}' via "
        f"mpirun (default -np {args.np}, np0={args.np0}; the mpi-pad "
        "entry forces its own device count).  Each child prints its own "
        "device banner.",
        flush=True,
    )
    sys.exit(run_rows(systems, run_smoke_test, args))
