---
paths:
  - "src/dnsjax/snapshot*.py"
  - "src/dnsjax/{__main__,param_surface}.py"
  - "src/dnsjax/flows/registry.py"
  - "src/dnsjax/analysis/**/*.py"
  - "src/dnsjax/twin/driver.py"
  - "scripts/{snapshot_perturb,snapshot_figure,ensemble_setup,twin_postprocess}.py"
  - "tests/test_{snapshot,snapshot_export,snapshot_import,snapshot_perturb,resume}.py"
---

# Snapshots and resume

- A snapshot is one uncompressed tar wrapping a zarr3 store at
  `format_version` 6; readers reject older ones (no translation, by
  design). Layout, the raw offset I/O and the `.partial`
  commit-by-rename: the `snapshot.py` and `snapshot_meta.py` module
  docstrings. On disk is the solver-native layout: never transpose it.
- Stored components are the physical ones for every family (the
  cylindrical/annular solver basis is converted at the write/read
  boundary). The stored state is the perturbation `u'` for base-flow
  systems (laminar = a zero array) and the total field for
  `flows.registry.total_field_systems`.
- The pipe family's solver state carries two slots it does not store:
  `from_solver_basis` drops them, `to_solver_basis` re-derives them,
  and only the optional `carry/` member (`outs.snapshot_embed_carry`)
  keeps them, restored on an unchanged trajectory
  (`snapshot.load_snapshot_carry`).
- The embedded params dump is the public-named surface
  (`param_surface.recorded_params_dump`); read it back with
  `flows.registry.internalize_stored` / `stored_value`, which raise on
  a core-section key this version does not define.
- A resume is np-agnostic, needs matching precision, and regrids every
  changed axis at load, by index (so an `lx`/`lz` change rescales what
  the surviving modes mean): wall-normal by interpolation
  (`__main__._interpolate_if_needed`), Fourier axes by padding or
  truncation (`snapshot._from_io_layout_core`). `t`/`it`/`isnap`
  continue only while `parameters.trajectory_defining_changes` is
  empty: a `phys`, `geo`, `res` or `[force]` change starts a new
  trajectory unless `init.force_resume`.
- An interrupted save leaves `*.tar.partial`, which `*.tar` globs skip
  (`scripts/ensemble_setup.py` relies on it).
