"""Twin-run (``dnsjax-twin``) driver package.

Submodules (import them explicitly; this ``__init__`` stays empty so
importing it never touches JAX or the singletons -- ``diagnostics``
imports the Cartesian geometry at module scope, and ``driver``
registers the ``[twin]`` extension at import):

- :mod:`dnsjax.twin.driver` -- the ``dnsjax-twin`` entry point:
  lockstep twin stepping, the ``[twin]`` extension section, paired
  snapshots/resume, the ``twin.dat``/``twin_budget.dat`` streams.
- :mod:`dnsjax.twin.diagnostics` -- difference-field diagnostics
  (component masks, energies, the 27-term budget, ``(k_z, k_x)``
  spectra).
- :mod:`dnsjax.twin.spectra` -- the ``twin_spectra.bin`` stream
  writer (JAX-free reader: :mod:`dnsjax.analysis.twin.spectra`).

``python -m dnsjax.twin`` runs the ``__main__`` shim, which hands
off to :func:`dnsjax.twin.driver.main`.
"""
