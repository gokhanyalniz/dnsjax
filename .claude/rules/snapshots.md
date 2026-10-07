---
paths:
  - "src/dnsjax/snapshot*.py"
  - "src/dnsjax/lowres.py"
  - "src/dnsjax/twin/cubes.py"
  - "src/dnsjax/{__main__,param_surface}.py"
  - "src/dnsjax/flows/registry.py"
  - "src/dnsjax/analysis/**/*.py"
  - "src/dnsjax/twin/driver.py"
  - "scripts/{snapshot_perturb,snapshot_figure,ensemble_setup,twin_postprocess}.py"
  - "tests/test_{snapshot,snapshot_export,snapshot_import,snapshot_perturb,resume}.py"
---

# Snapshots and resume

- A snapshot is one uncompressed tar wrapping a zarr3 store; readers
  reject a `format_version` below `snapshot_meta.MIN_FORMAT_VERSION`
  (no translation, by design). Layout, the raw offset I/O and the
  `.partial` commit-by-rename: the `snapshot.py` and `snapshot_meta.py`
  module docstrings. On disk is the solver-native layout: never
  transpose it.
- Stored components are the physical ones for every family (the
  cylindrical/annular solver basis is converted at the write/read
  boundary); a base-flow system's laminar state is a zero array.
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
  changed axis at load (`__main__._interpolate_if_needed`,
  `snapshot._from_io_layout_core`). `t`/`it`/`isnap` continue only
  while `parameters.trajectory_defining_changes` is empty, unless
  `init.force_resume`.
- An interrupted save leaves `*.tar.partial`, which `*.tar` globs skip
  (`scripts/ensemble_setup.py` relies on it).
- Every dnsjax tar goes through `snapshot.write_archive`: states, the
  `[lowres]` files and the twin cubes. A new array output does too,
  y-major, with a metadata `kind` if it is not a state (absent means a
  state). Optional members beside `state/`: `carry/` and `pressure/`.
- A non-state `kind` or a `lowres` entry is not a checkpoint:
  `snapshot_meta.checkpoint_refusal` refuses it on every resume path
  and the parent harvest, and the field readers refuse a non-state
  `kind`.
