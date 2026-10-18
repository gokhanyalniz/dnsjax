"""Twin-run (``dnsjax-twin``) driver package.

Import the submodules explicitly: this ``__init__`` stays empty so
importing it never touches JAX or the singletons (``diagnostics``
imports the Cartesian geometry at module scope, and ``driver``
registers the ``[twin]`` extension at import).

- :mod:`dnsjax_lab.twin.driver`, the ``dnsjax-twin`` entry point:
  lockstep twin stepping, the ``[twin]`` extension section, paired
  snapshots/resume, the ``twin.dat``/``twin_budget.dat`` streams;
  ``python -m dnsjax_lab.twin`` runs the ``__main__`` shim, which
  hands off to :func:`dnsjax_lab.twin.driver.main`.
- What the streams carry: :mod:`dnsjax_lab.twin.diagnostics`
  (component masks, energies, the 27-term budget, ``(k_z, k_x)``
  spectra, and the wall-normal-resolved spectra and spectral budget
  that supersede the three-bin split) and :mod:`dnsjax_lab.twin.pressure`
  (the difference-field pressure, on the IMM's own wall closure; the
  one budget term that a `$y$`-resolved balance cannot omit).
- The writers: :mod:`dnsjax_lab.twin.spectra` (``twin_spectra*.bin``)
  and :mod:`dnsjax_lab.twin.yspectra` (``twin_yspectra*.bin``,
  ``twin_ybudget.bin``), on the buffered-binary state machine of
  :mod:`dnsjax_lab.twin._binstream`, and :mod:`dnsjax_lab.twin.cubes`
  (the 3-D cube directories); each names its JAX-free reader in
  :mod:`dnsjax_lab.analysis.twin`.
"""
