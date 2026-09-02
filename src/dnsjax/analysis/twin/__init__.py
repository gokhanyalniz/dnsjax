"""JAX-free offline analysis of twin-run (``dnsjax-twin``) outputs.

Layout (one module per concern; none is imported by the top-level
``dnsjax.analysis`` namespace, mirroring ``analysis/response``):

- :mod:`.series` -- readers for the ``.dat`` scalar streams
  (``twin.dat`` / ``twin_budget.dat``) and the ``twin.json`` member
  record, plus the column-generic :func:`~.series.read_dat` that loads
  any of them and the per-state ``stats.dat`` / ``stats_twin.dat``
  pair as well; per-component budget sums, the
  budget-closure residuals, and :func:`~.series.uniform_grid`, which
  selects a stream's own cadence grid out of the off-grid rows a
  resume and the final row add.
- :mod:`.ensemble` -- member-tree aggregation of the twin streams on
  aligned relative time, and the growth-rate fits (`$\\lambda$` from
  the exponential phase, the algebraic-phase linear rate).
- :mod:`.spectra` -- reader for the ``twin_spectra.bin`` stream
  and the decorrelation ratio.
- :mod:`.yspectra` -- readers for the wall-normal-resolved
  ``twin_yspectra.bin`` / ``twin_ybudget.bin`` streams, the
  sidecar-driven record layout they share with the memory-mapped
  reader in ``scripts/twin_spectral_maps.py``, the quadrature
  contraction, the three-bin energies recovered from them, and the
  total-in-`$(y, k)$` fluctuation energy (the total with the
  `$(0, 0)$` mode removed) that a difference spectrum is normalised
  by.
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
    uniform_grid,
)
from .spectra import (
    TwinSpectraData,
    decorrelation_ratio,
    read_twin_spectra,
)
from .yspectra import (
    LEGACY_SUFFIXES,
    YResolvedData,
    bin_energies,
    fluctuation_energy,
    fluctuation_profile,
    integrate_y,
    mean_free_spectrum,
    mean_mode_name,
    mean_mode_profile,
    read_twin_ybudget,
    read_twin_yspectra,
    record_dtype,
    stored_fields,
    stored_suffixes,
)

__all__ = [
    "LEGACY_SUFFIXES",
    "ClosureResiduals",
    "TwinSeries",
    "TwinSpectraData",
    "YResolvedData",
    "aggregate_members",
    "bin_energies",
    "budget_sums",
    "closure_residuals",
    "decorrelation_ratio",
    "fit_exponential_rate",
    "fit_linear_rate",
    "fluctuation_energy",
    "fluctuation_profile",
    "integral_lengths",
    "integral_lengths_from_modes",
    "integrate_y",
    "mean_free_spectrum",
    "mean_mode_name",
    "mean_mode_profile",
    "partner_of",
    "read_dat",
    "read_twin",
    "read_twin_spectra",
    "read_twin_ybudget",
    "read_twin_yspectra",
    "record_dtype",
    "stored_fields",
    "stored_suffixes",
    "uniform_grid",
]
