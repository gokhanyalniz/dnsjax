# dnsjax-twin

`driver.py` is the console script (`[twin]`, paired snapshots and
resume, stream wiring; `__main__.py` is its `python -m` shim);
`diagnostics.py`, `pressure.py`, `_binstream.py`, `spectra.py`,
`yspectra.py` and `cubes.py` compute and write the streams; the
readers are `dnsjax.analysis.twin`. Formats and derivations: those module
docstrings. The human overview: `README.md` here.

- Every twin cadence counts from the member's own perturbation step
  (`twin.json`'s `parent_it`), not from the absolute `it`, so members
  meet on `t - parent_t` counted in steps.
- Both states write the per-state solver streams (`stats_twin.dat`,
  `steps_twin.dat`, `corrector_twin.dat`; byte-identical to the
  reference's at `twin.e0 = 0`), each with its own state's driving
  columns; `twin.dat` has none, and `[probes]` is reference-only.
- A resume never re-perturbs, and every `_TWIN_MATCH_KEYS` entry must
  match the recorded `twin.json`, a back-filled legacy value included
  (the `_TWIN_LEGACY_DEFAULTS` comment).
- `analysis.twin.stored_fields` / `record_dtype` own the `(y, k)`
  stream layouts, for the eager reader and for
  `scripts/twin_spectral_maps.py`'s memory map alike.
- The reference spectra are streams of their own (`*_ref`), but a
  member recorded before the split resumes writing them inside the
  difference streams (`includes_ref` in their sidecars; the driver's
  "Diagnostic streams"). Every reader and the spectral-maps script
  accept both layouts, mixed ensembles included; keep it that way.
- Per-sample outputs (the 3-D cubes, `lowres/`, `lowres_delta/`) are
  dnsjax tars through `snapshot.write_archive`, one file per step;
  `_STREAM_FILES` names their directories so a fresh start refuses
  stale ones.
- Change a `[twin]` default only after reading its field description
  and the Design notes in `driver.py`: the defaults decide what a run
  costs and what its streams can answer.
- Offline tools: `scripts/twin_postprocess.py`,
  `scripts/twin_spectral_maps.py` and `scripts/ensemble_setup.py
  build-twin` (`scripts/CLAUDE.md`).
