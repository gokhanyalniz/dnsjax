"""Pytest bridge over the lab suite's standalone test scripts.

The machinery and the markers: ``tests/_bridge.py``, shared with the
core suite.

The quick ``mpirun`` rows that ``-m "not slow"`` adds: probes, forcing
and twin_postprocess.
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
# no-mpirun unit fallback + the full run).
_SCRIPTS: list[Row] = [
    # Seven checks, one subprocess each (the geometry captures kappa at
    # import); ~3 min, no mpirun.
    ("test_curved_pipe.py", (), (), 1800),
    ("test_forcing.py", ("--unit-only",), NO_MPI_ONLY, 1800),
    ("test_probes.py", ("--unit-only",), NO_MPI_ONLY, 1800),
    ("response/test_ensemble.py", (), (), 1800),
    ("response/test_lim.py", (), (), 1800),
    ("response/test_operator_tools.py", (), (), 1800),
    ("response/test_probes_reader.py", (), (), 1800),
    ("response/test_ssi.py", (), (), 1800),
    ("test_twin_analysis.py", (), (), 1800),
    ("test_twin_postprocess.py", ("--unit-only",), NO_MPI_ONLY, 1800),
    # Offline and matplotlib-only; skips itself with a message where
    # the ``plots`` dependency group is not installed.
    ("test_twin_spectral_maps.py", (), (), 1800),
    ("test_twin_unit.py", (), (), 1800),
    ("test_viscoelastic.py", (), (), 1800),
    ("test_viscoelastic_pipe.py", (), (), 1800),
    ("test_forcing.py", (), MPI, 1800),
    ("test_probes.py", (), MPI, 1800),
    # No MPI: single-process throughout (a lone process needs no
    # launcher).  ~20 min at the default single seed -- the
    # per-rung 50-advective-unit parent spin-ups dominate.
    ("test_twin_budget.py", (), SLOW, 3600),
    ("test_twin_driver.py", (), MPI + SLOW, 3600),
    # ~2.5 min: three short twin members plus the reconstructions of
    # them, the last on a (2, 1) mesh -- not a SLOW row.
    ("test_twin_postprocess.py", (), MPI, 1800),
]

test_script = script_test(_TESTS_DIR, _SCRIPTS)
