"""JAX-free offline analysis of twin-run (``dnsjax-twin``) outputs.

Layout (one module per concern; none is imported by the top-level
``dnsjax.analysis`` namespace, mirroring ``analysis/response``):

- :mod:`.series` -- readers for the ``.dat`` scalar streams
  (``twin.dat`` / ``twin_budget.dat`` / ``stats.dat``) and the
  ``twin.json`` member record; per-component budget sums and the
  budget-closure residuals.
- :mod:`.ensemble` -- member-tree aggregation of the twin streams on
  aligned relative time, and the growth-rate fits (`$\\lambda$` from
  the exponential phase, the algebraic-phase linear rate).
- :mod:`.spectra` -- reader for the ``twin_spectra.bin`` stream
  and the decorrelation ratio.
- :mod:`.lengths` -- integral length scales of the difference field
  from a paired snapshot.

Everything here is importable without JAX (the
``tests/test_twin_analysis.py`` guarantee).
"""

from .ensemble import (
    aggregate_members,
    fit_exponential_rate,
    fit_linear_rate,
)
from .lengths import (
    integral_lengths,
    integral_lengths_from_modes,
    partner_of,
)
from .series import (
    ClosureResiduals,
    TwinSeries,
    budget_sums,
    closure_residuals,
    read_dat,
    read_twin,
)
from .spectra import (
    TwinSpectraData,
    decorrelation_ratio,
    read_twin_spectra,
)

__all__ = [
    "ClosureResiduals",
    "TwinSeries",
    "TwinSpectraData",
    "aggregate_members",
    "budget_sums",
    "closure_residuals",
    "decorrelation_ratio",
    "fit_exponential_rate",
    "fit_linear_rate",
    "integral_lengths",
    "integral_lengths_from_modes",
    "partner_of",
    "read_dat",
    "read_twin",
    "read_twin_spectra",
]
