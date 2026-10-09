r"""JAX-free offline analysis of twin-run (``dnsjax-twin``) outputs.

Layout (one module per concern; none is imported by the top-level
``dnsjax.analysis`` namespace, mirroring ``analysis/response``):

- :mod:`.series` -- readers for the ``.dat`` scalar streams
  (``twin.dat`` / ``twin_budget.dat``) and the ``twin.json`` member
  record; the column-generic :func:`~.series.read_dat`, which also
  loads the per-state ``stats.dat`` / ``stats_twin.dat`` pair;
  per-component budget sums and the budget-closure residuals;
  :func:`~.series.relative_time`, which reads a member's sample times
  as whole steps since its perturbation; and
  :func:`~.series.uniform_grid`, which keeps a stream's own cadence
  grid and drops the off-grid rows a resume and the final row add.
- :mod:`.ensemble` -- member-tree aggregation of the twin streams on
  aligned relative time, and the growth-rate fits (`$\lambda$` from
  the exponential phase, the algebraic-phase linear rate).
- :mod:`.spectra` -- readers for the ``twin_spectra.bin`` stream
  and its reference counterpart (either layout), and the
  decorrelation ratio.
- :mod:`.yspectra` -- readers for the wall-normal-resolved
  ``twin_yspectra.bin`` / ``twin_ybudget.bin`` streams and the
  sidecar-driven record layout they share with the memory-mapped
  reader in ``scripts/twin_spectral_maps.py``; the quadrature
  contraction; the three-bin energies recovered from them; the
  fluctuation energy in `$(y, k)$` (the total with the `$(0, 0)$` mode
  removed) that normalizes a difference spectrum; the shape overlap of
  two such spectra, which ``scripts/random_ic_calibrate.py`` scores an
  initial condition with; and the budget regrouped into the terms of
  the difference-energy balance (:func:`~.yspectra.balance_term`).
- :mod:`.cubes` -- readers for the 3-D ``(y, k_z, k_x)`` energy and
  budget cubes (``twin_spectra3d/``, ``twin_spectra3d_ref/``,
  ``twin_budget3d/``), one dnsjax tar per sample.
- :mod:`.lengths` -- integral length scales of the difference field
  from a paired snapshot.
- :mod:`.moments` -- the log-coordinate moments of a `$(y, k)$`
  density (centroid, spreads, tilt) from raw sums that are linear in
  the field, and the split of their rates over the terms of a budget.
- :mod:`.growth` -- growth-law diagnostics of a decorrelating pair:
  the logarithmic rate, the `$\gamma$`-`$R$` diagram's slope, the
  local algebraic exponent, the bound-free variable `$-\ln(1 - R)$`
  and the reference laws they are read against.

Everything here is importable without JAX (the
``tests/test_twin_analysis.py`` guarantee).
"""

from .cubes import (
    Cube,
    CubeSeries,
    cube_files,
    read_cube,
    read_cubes,
    read_twin_budget3d,
    read_twin_spectra3d,
    read_twin_spectra3d_ref,
)
from .ensemble import (
    aggregate_members,
    fit_exponential_rate,
    fit_linear_rate,
)
from .growth import (
    algebraic_exponent,
    bound_free,
    decorrelation_rate,
    log_rate,
    log_slope,
    logistic_rate,
    longest_window,
)
from .lengths import (
    integral_lengths,
    integral_lengths_from_modes,
    partner_of,
)
from .moments import (
    LogMoments,
    MomentRates,
    log_moment_sums,
    log_moments,
    moment_rates,
)
from .series import (
    ClosureResiduals,
    TwinSeries,
    budget_sums,
    closure_residuals,
    read_dat,
    read_twin,
    relative_time,
    uniform_grid,
)
from .spectra import (
    TwinSpectraData,
    TwinSpectraRefData,
    decorrelation_ratio,
    read_twin_spectra,
    read_twin_spectra_ref,
)
from .yspectra import (
    BALANCE_PARTS,
    BALANCE_SOURCES,
    BALANCE_TERMS,
    LEGACY_SUFFIXES,
    YResolvedData,
    balance_term,
    balance_terms,
    bin_energies,
    fluctuation_energy,
    fluctuation_profile,
    integrate_y,
    mean_free_spectrum,
    mean_mode_name,
    mean_mode_profile,
    read_twin_ybudget,
    read_twin_yspectra,
    read_twin_yspectra_ref,
    record_dtype,
    shape_alignment,
    stored_fields,
    stored_suffixes,
)

__all__ = [
    "BALANCE_PARTS",
    "BALANCE_SOURCES",
    "BALANCE_TERMS",
    "ClosureResiduals",
    "Cube",
    "CubeSeries",
    "LEGACY_SUFFIXES",
    "LogMoments",
    "MomentRates",
    "TwinSeries",
    "TwinSpectraData",
    "TwinSpectraRefData",
    "YResolvedData",
    "aggregate_members",
    "algebraic_exponent",
    "balance_term",
    "balance_terms",
    "bin_energies",
    "bound_free",
    "budget_sums",
    "closure_residuals",
    "cube_files",
    "decorrelation_rate",
    "decorrelation_ratio",
    "fit_exponential_rate",
    "fit_linear_rate",
    "fluctuation_energy",
    "fluctuation_profile",
    "integral_lengths",
    "integral_lengths_from_modes",
    "integrate_y",
    "log_moment_sums",
    "log_moments",
    "log_rate",
    "log_slope",
    "logistic_rate",
    "longest_window",
    "mean_free_spectrum",
    "mean_mode_name",
    "mean_mode_profile",
    "moment_rates",
    "partner_of",
    "read_cube",
    "read_cubes",
    "read_dat",
    "read_twin",
    "read_twin_budget3d",
    "read_twin_spectra",
    "read_twin_spectra3d",
    "read_twin_spectra3d_ref",
    "read_twin_spectra_ref",
    "read_twin_ybudget",
    "read_twin_yspectra",
    "read_twin_yspectra_ref",
    "record_dtype",
    "relative_time",
    "shape_alignment",
    "stored_fields",
    "stored_suffixes",
    "uniform_grid",
]
