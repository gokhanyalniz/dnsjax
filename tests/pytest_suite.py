"""Pytest bridge over the core suite's standalone test scripts.

The machinery and the markers: ``tests/_bridge.py``.  The lab's
scripts have their own bridge, ``lab/tests/pytest_suite.py``.

The quick ``mpirun`` rows that ``-m "not slow"`` adds: seeding,
device_grid, host_placement, mpi_communicators, wall_time_stop and
run_status.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR))

from _bridge import MPI, NO_MPI_ONLY, SLOW, Row, script_test  # noqa: E402

# One row per invocation.  Scripts appearing twice: ``test_resume``
# (offline ``--unit-only`` subset + full mpirun run),
# ``test_device_grid``/``test_seeding`` (the no-mpirun unit fallback +
# the full run), ``test_driving`` (the always-run unit half + the slow
# full run: its full row launches no mpirun, so it is marked slow only
# and its unit row carries no NO_MPI_ONLY, or the fast tiers would never
# run it), ``test_transient_growth`` (offline ``--fast`` structure
# checks + the slow full run with the published values),
# ``test_run_status`` (the ``--no-mpi`` tiers + the ``--mpi-only`` one,
# disjoint), and ``test_laminar_smoke`` (single-device + the ``--np 2``
# mesh row whose reason is spelled out at that entry).
_SCRIPTS: list[Row] = [
    ("test_adaptive.py", (), (), 1800),
    ("test_annular.py", (), (), 1800),
    # Eight in-process (forced CPU device) subprocesses, each a
    # few differentiated steps; no mpirun.
    ("test_autodiff.py", (), (), 1800),
    ("test_banded_solver.py", (), (), 1800),
    ("test_banded_solver_sharded.py", (), (), 1800),
    ("test_bootstrap.py", (), (), 1800),
    ("test_cartesian.py", (), (), 1800),
    ("test_cnab2.py", (), (), 1800),
    ("test_cylindrical.py", (), (), 1800),
    # Offline (no mpirun), but it runs five full solver steps in
    # subprocesses, so it is a slow row.
    ("test_imm_continuity.py", (), SLOW, 3600),
    ("test_integration.py", (), (), 1800),
    # Parses the core's sources; no JAX, a few seconds.
    ("test_layering.py", (), (), 1800),
    # Nine children importing the core under fake entry points; no
    # JAX, a few seconds.
    ("test_plugins.py", (), (), 1800),
    ("test_localized_rolls.py", (), (), 1800),
    # In-process (forced CPU device) workers, one per case; no mpirun.
    ("test_lowres.py", (), (), 1800),
    ("test_mean_mask.py", (), (), 1800),
    # Four in-process (forced CPU device) subprocesses, no mpirun.
    ("test_mean_mode.py", (), (), 1800),
    ("test_monochromatic.py", (), (), 1800),
    ("test_padding.py", (), (), 1800),
    ("test_param_surface.py", (), (), 1800),
    ("test_device_grid.py", ("--unit-only",), NO_MPI_ONLY, 1800),
    ("test_driving.py", ("--unit-only",), (), 1800),
    ("test_seeding.py", ("--unit-only",), NO_MPI_ONLY, 1800),
    ("test_resume.py", ("--unit-only",), (), 1800),
    # ~2 min: the units plus six one-process solver launches (no
    # launcher); the mpirun tier is its own row below.
    ("test_run_status.py", ("--no-mpi",), (), 1800),
    # ~25 s: four one-process solver launches of four steps each.
    ("test_run_hooks.py", (), (), 1800),
    ("test_snapshot.py", (), (), 1800),
    ("test_snapshot_export.py", (), (), 1800),
    ("test_snapshot_import.py", (), (), 1800),
    ("test_snapshot_perturb.py", (), (), 1800),
    ("test_transient_growth.py", ("--fast",), (), 1800),
    ("test_wedge.py", (), (), 1800),
    # Six in-process (forced CPU device) workers, one per case, each
    # compiling one step twice; no mpirun.
    ("test_wall_normal_matvec.py", (), (), 1800),
    ("test_temporal_order.py", (), SLOW, 3600),
    ("test_energy_budget.py", (), MPI + SLOW, 3600),
    # Three (2, 2) launches of a few steps, two of them over a
    # simulated second host; about a minute.
    ("test_device_grid.py", (), MPI, 1800),
    ("test_driving.py", (), SLOW, 2400),
    # ~2.5 min: nine 2-rank launches of a few steps each (every core
    # flow family, [lowres], adaptive dt, cnab2, single precision, a
    # wedge, a regrid resume) -- not a SLOW row.
    ("test_host_placement.py", (), MPI, 1800),
    ("test_laminar_smoke.py", (), MPI + SLOW, 3600),
    # Under 10 s: two 4-rank launches, the second over a simulated
    # second host, no stepping -- not a SLOW row.
    ("test_mpi_communicators.py", (), MPI, 1800),
    # Multi-device, and specifically on the np1 (k_x / axial) axis:
    # every entry runs at the runner default --np 1 otherwise, which
    # is how a total np1 > 1 failure of res.consistent_imm reached
    # production in the cylindrical and annular families -- the
    # geometries build wavenumber arrays that are unsharded on that
    # axis, so a spec mismatch there is invisible single-device.
    ("test_laminar_smoke.py", ("--np", "2"), MPI + SLOW, 3600),
    ("test_random_smoke.py", (), MPI + SLOW, 3600),
    # MPI for the cross-process seed-agreement row alone; the solver
    # launches are all a handful of steps each.
    ("test_seeding.py", (), MPI, 1800),
    ("test_resume.py", (), MPI + SLOW, 3600),
    ("test_rolls_smoke.py", (), MPI + SLOW, 3600),
    ("test_transient_growth.py", (), SLOW, 3600),
    # ~1.5 min: one solver run to its 60 s wall-clock budget on two
    # ranks -- not a SLOW row.
    ("test_wall_time_stop.py", (), MPI, 1800),
    # ~1.5 min: three short two-rank runs, each stopped on request.
    ("test_run_status.py", ("--mpi-only",), MPI, 1800),
]

test_script = script_test(_TESTS_DIR, _SCRIPTS)
