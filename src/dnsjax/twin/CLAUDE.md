# dnsjax-twin

`driver.py` is the console script (`[twin]`, paired snapshots and
resume, stream wiring; `__main__.py` is its `python -m` shim);
`diagnostics.py`, `pressure.py`, `_binstream.py`, `spectra.py` and
`yspectra.py` compute and write the streams; the readers are
`dnsjax.analysis.twin`. Formats and derivations: those module
docstrings. The human overview: `README.md` here.

- Every twin cadence counts from the member's own perturbation step
  (`twin.json`'s `parent_it`), not from the absolute `it`, so members
  meet on `t - parent_t` counted in steps. Members recorded before
  that rule carry displaced grids: `analysis.twin.aggregate_members`
  and `scripts/twin_spectral_maps.py` refuse them unless given
  `align_atol` (`--align-atol`).
- Both states write the per-state solver streams (`stats_twin.dat`,
  `steps_twin.dat`, `corrector_twin.dat`; byte-identical to the
  reference's at `twin.e0 = 0`); `[probes]` is reference-only.
- A resume never re-perturbs, and every `_TWIN_MATCH_KEYS` entry must
  match the recorded `twin.json`, a back-filled legacy value included
  (the `_TWIN_LEGACY_DEFAULTS` comment).
- `analysis.twin.stored_fields` / `record_dtype` own the three
  `(y, k)` stream layouts, for the eager reader and for
  `scripts/twin_spectral_maps.py`'s memory map alike.
- The `[twin]` defaults decide what a run costs and what its streams
  can answer; each field's description in `driver.py` says why.
- Offline tools: `scripts/twin_postprocess.py`,
  `scripts/twin_spectral_maps.py` and `scripts/ensemble_setup.py
  build-twin` (`scripts/CLAUDE.md`).
