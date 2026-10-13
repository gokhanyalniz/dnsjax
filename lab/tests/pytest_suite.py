"""Pytest bridge over the lab suite's standalone test scripts.

The machinery and the markers: ``tests/_bridge.py``, shared with the
core suite.

The quick ``mpirun`` rows that ``-m "not slow"`` adds: probes, forcing,
twin_postprocess, wall_time_stop, run_status and host_placement.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR))

import _repo  # noqa: E402, F401 -- the core suite's helpers
from _bridge import MPI, NO_MPI_ONLY, SLOW, Row, script_test  # noqa: E402

# One row per invocation.  Scripts appearing twice:
# ``test_forcing``/``test_probes``/``test_twin_postprocess`` (the
# no-mpirun unit fallback + the full run), ``test_run_status`` (the
# ``--no-mpi`` tier + the ``--mpi-only`` one, disjoint) and
# ``test_laminar_smoke`` (single-device + the ``--np 2`` mesh row) and
# ``test_transient_growth`` (offline ``--fast`` structure checks + the
# slow run with the published value).
_SCRIPTS: list[Row] = [
    # Seven checks, one subprocess each (the geometry captures kappa at
    # import); ~3 min, no mpirun.
    ("test_adaptive.py", (), (), 1800),
    ("test_cnab2.py", (), (), 1800),
    ("test_curved_pipe.py", (), (), 1800),
    ("test_cylindrical.py", (), (), 1800),
    ("test_forcing.py", ("--unit-only",), NO_MPI_ONLY, 1800),
    # In-process (forced CPU device) workers, one per case; no mpirun.
    ("test_localized_rolls.py", (), (), 1800),
    ("test_lowres.py", (), (), 1800),
    ("test_param_surface.py", (), (), 1800),
    ("test_probes.py", ("--unit-only",), NO_MPI_ONLY, 1800),
    ("test_quasi_keplerian.py", (), (), 1800),
    # ~1 min: the natural run (the twin's parent) and one one-process
    # twin launch; the mpirun tier is its own row below.
    ("test_resume.py", (), (), 1800),
    ("test_run_status.py", ("--no-mpi",), (), 1800),
    # The offline gating check plus three single-process launches (a
    # parent run, a fresh twin start and its paired resume); no mpirun.
    ("test_seeding.py", (), (), 1800),
    ("response/test_ensemble.py", (), (), 1800),
    ("response/test_lim.py", (), (), 1800),
    ("response/test_operator_tools.py", (), (), 1800),
    ("response/test_probes_reader.py", (), (), 1800),
    ("response/test_ssi.py", (), (), 1800),
    ("test_snapshot.py", (), (), 1800),
    ("test_snapshot_export.py", (), (), 1800),
    ("test_snapshot_perturb.py", (), (), 1800),
    ("test_transient_growth.py", ("--fast",), (), 1800),
    ("test_twin_analysis.py", (), (), 1800),
    ("test_twin_postprocess.py", ("--unit-only",), NO_MPI_ONLY, 1800),
    # Offline and matplotlib-only; skips itself with a message where
    # the ``plots`` dependency group is not installed.
    ("test_twin_spectral_maps.py", (), (), 1800),
    ("test_twin_unit.py", (), (), 1800),
    ("test_viscoelastic.py", (), (), 1800),
    ("test_viscoelastic_pipe.py", (), (), 1800),
    # Three in-process (forced CPU device) workers, one per case, each
    # compiling one step twice; no mpirun.
    ("test_wall_normal_matvec.py", (), (), 1800),
    ("test_forcing.py", (), MPI, 1800),
    ("test_probes.py", (), MPI, 1800),
    # ~3 min: ten 2-rank launches of a few steps each ([probes],
    # [force], a twin member, the lab flows) -- not a SLOW row.
    ("test_host_placement.py", (), MPI, 1800),
    ("test_laminar_smoke.py", (), MPI + SLOW, 3600),
    # Multi-device on the np1 (k_x / axial) axis, as the core's row of
    # the same name (tests/pytest_suite.py says why).
    ("test_laminar_smoke.py", ("--np", "2"), MPI + SLOW, 3600),
    ("test_random_smoke.py", (), MPI + SLOW, 3600),
    ("test_rolls_smoke.py", (), MPI + SLOW, 3600),
    ("test_transient_growth.py", (), SLOW, 3600),
    # No MPI: single-process throughout (a lone process needs no
    # launcher).  ~20 min at the default single seed -- the
    # per-rung 50-advective-unit parent spin-ups dominate.
    ("test_twin_budget.py", (), SLOW, 3600),
    ("test_twin_driver.py", (), MPI + SLOW, 3600),
    # ~2.5 min: three short twin members plus the reconstructions of
    # them, the last on a (2, 1) mesh -- not a SLOW row.
    ("test_twin_postprocess.py", (), MPI, 1800),
    # ~2.5 min: one solver and one twin run, each to its 60 s wall-clock
    # budget on two ranks -- not a SLOW row.
    ("test_wall_time_stop.py", (), MPI, 1800),
    # ~1 min: the natural run, then one two-rank twin run stopped on
    # request.
    ("test_run_status.py", ("--mpi-only",), MPI, 1800),
]

test_script = script_test(_TESTS_DIR, _SCRIPTS)
