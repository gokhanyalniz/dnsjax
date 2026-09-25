"""Initial-condition generators (the ``init`` start modes).

Submodules (import them explicitly; this ``__init__`` stays empty so
each keeps its own import-order guarantee -- all are importable before
JAX is configured, see their module docstrings):

- :mod:`dnsjax.ic.random_field` -- random divergence-free IC
  generators (``init.random_field``, the default start mode).
- :mod:`dnsjax.ic.localized_rolls` -- deterministic localized-spot
  ("turbulent spot") IC generators (``init.localized_rolls``).
- :mod:`dnsjax.ic.mean_mode` -- the conservation laws a perturbation
  of the ``(k_x, k_z) = (0, 0)`` mode must respect (Cartesian flows),
  shared by the random IC, the twin partner and
  ``scripts/snapshot_perturb.py``.
"""
