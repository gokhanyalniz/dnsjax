"""Command-line utilities that read the solver's files.

Each module is a console script that meets the solver through the files
a run writes and imports only the standard library and the JAX-free
leaves, so it starts in well under a second wherever ``dnsjax`` is
installed:

- :mod:`dnsjax.tools.snapshot_info`: ``dnsjax-info``, the metadata and
  embedded stats of snapshots.
"""
