"""Initial-condition generators (the ``init`` start modes).

Submodules (import them explicitly; this ``__init__`` stays empty so
each generator keeps its own import-order guarantee -- both are
importable before JAX is configured, see their module docstrings):

- :mod:`dnsjax.ic.random_field` -- random divergence-free IC
  generators (``init.random_field``, the default start mode).
- :mod:`dnsjax.ic.localized_rolls` -- deterministic localized-spot
  ("turbulent spot") IC generators (``init.localized_rolls``).
"""
