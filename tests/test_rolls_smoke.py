"""Localized-rolls IC smoke tests: nonlinear integration, wall-bounded.

Starts each wall-bounded rolls-builder variant (plane-couette for the
Cartesian builder, pipe, taylor-couette, dean) from the deterministic
streamwise-localized-rolls IC (the in-process ``init.localized_rolls``
start mode, no snapshot file) at a transitional Reynolds number on a
small domain, and integrates a short time (``t = 0.25``; the
long-horizon integration of the same flows is
``test_random_smoke.py``'s job at ``t = 1``) -- verifying the run
completes with no error, NaN, or blow-up.  The rolls drive the full
**nonlinear** path (the laminar smoke test cannot, since ``u' = 0``
there), like ``tests/test_random_smoke.py`` but with a deterministic
IC.

Reuses ``test_random_smoke``'s ``_check_run`` (the five success criteria:
exit 0, reached the end, no ``"Corrector failed"``, finite + converged
final error, all summary values finite).  The triply-periodic family is
excluded -- the rolls are a wall-normal cross-plane structure, defined for
wall-bounded systems only.

Includes a **forced multi-device, padding-inducing** case (``mpirun
--oversubscribe -np 2`` with ``np1 = 2`` and ``nx`` chosen so
``nx // 2`` is not divisible by ``np1``) that exercises the rolls'
sharded per-device spectral build with spectral `$k_x$` padding -- the
regression guard for the no-replication construction.

Usage (single device)::

    uv run python tests/test_rolls_smoke.py

Faster subset (one system, coarse / short)::

    uv run python tests/test_rolls_smoke.py \
        --systems plane-couette --res 16 --max-sim-time 0.2

The whole standard suite at two devices via MPI::

    uv run python tests/test_rolls_smoke.py --np 2
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile

from _live import report, run_live
from test_random_smoke import _check_run

sys.stdout.reconfigure(line_buffering=True)

# ── configuration ────────────────────────────────────────────────────

# Same Reynolds numbers / boxes as test_random_smoke (minus kolmogorov;
# the rolls are wall-bounded only).  Each standard entry runs at the
# suite's ``--np`` (default 1); the trailing ``*-mpi-pad`` entry always
# runs multi-device with a padding-inducing resolution.
SYSTEMS: list[dict] = [
    {
        "name": "plane-couette",
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
    {
        # plane-poiseuille needs no entry: the Cartesian rolls builder
        # is shared with plane-couette (above), rolls construction is
        # unit-tested in test_localized_rolls.py, and per-flow
        # nonlinear integration lives in test_random_smoke.py.
        "name": "pipe",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--geo.lz",
            "5",
        ],
    },
    {
        # ``res.consistent_imm`` on the pipe, from a *resolved* rolls
        # IC.  This entry predates the 2026-07-26 reformulation, when
        # the flag's composed D2 = D1*D1 was not grid-scale-dissipative
        # and a grid-white draw was genuinely unstable, so a rolls IC
        # was the only option.  It is now redundant with the random-IC
        # entries in ``test_random_smoke.py`` as a *stability* gate and
        # stays as the resolved-IC coverage of the same path.
        "name": "pipe-consistent-imm",
        "args": [
            "--phys.system",
            "pipe",
            "--phys.re",
            "1800",
            "--res.consistent_imm",
            "True",
            "--geo.lz",
            "5",
        ],
    },
    {
        "name": "taylor-couette",
        "args": [
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
        ],
    },
    {
        "name": "dean",
        "args": [
            "--phys.system",
            "dean",
            "--phys.re",
            "1000",
            "--geo.eta",
            "0.5",
            "--geo.lz",
            "5",
        ],
    },
    {
        # Regression guard for the sharded (no-replication) build with
        # spectral padding: 2 devices on the kx axis, nx // 2 = 17 not
        # divisible by np1 = 2 (padded to 18).  Always multi-device.
        "name": "plane-couette-mpi-pad",
        "force_np": 2,
        "force_np0": 1,
        "oversubscribe": True,
        "res": {"nx": 34, "ny": 32, "nz": 32},
        "args": [
            "--phys.system",
            "plane-couette",
            "--phys.re",
            "330",
            "--geo.lx",
            "5",
            "--geo.lz",
            "5",
        ],
    },
]

CORRECTOR_TOLERANCE = 1e-5

# ── helpers ──────────────────────────────────────────────────────────


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
    # Cylindrical/annular flows take the aliased public resolution
    # names (axial nz, radial nr, azimuthal ntheta); the internal
    # meaning (nx = axial, ny = radial, nz = azimuthal) is unchanged.
    cyl_annular = any(
        s in system["args"]
        for s in (
            "pipe",
            "taylor-couette",
            "quasi-keplerian",
            "dean",
            "viscoelastic-dean",
        )
    )
    res_flags = (
        ["--res.nz", nx, "--res.nr", ny, "--res.ntheta", nz]
        if cyl_annular
        else ["--res.nx", nx, "--res.ny", ny, "--res.nz", nz]
    )

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
        *res_flags,
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
    """Run a single localized-rolls smoke test (in a fresh directory)."""
    name = system["name"]
    dt = min(args.dt, system.get("max_dt", math.inf))
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

        summary = _check_run(result.stdout, name, args.max_sim_time, dt)

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


if __name__ == "__main__":
    args = _parse_args()

    systems = SYSTEMS
    if args.systems:
        wanted = set(args.systems)
        systems = [s for s in SYSTEMS if s["name"] in wanted]
        unknown = wanted - {s["name"] for s in SYSTEMS}
        if unknown:
            print(f"Unknown system(s): {sorted(unknown)}")
            sys.exit(2)

    print(
        f"Localized-rolls smoke tests on platform '{args.platform}' via "
        f"mpirun (default -np {args.np}, np0={args.np0}; the mpi-pad "
        "entry forces its own device count).  Each child prints its own "
        "device banner.",
        flush=True,
    )

    passed = 0
    failures: list[tuple[str, str]] = []
    for system in systems:
        try:
            run_smoke_test(system, args)
            passed += 1
        except (AssertionError, subprocess.TimeoutExpired) as exc:
            print(f"  FAIL  {system['name']}: {exc}")
            failures.append((system["name"], str(exc)))

    sys.exit(report(passed, failures))
