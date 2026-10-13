# dnsjax.analysis

The external-facing snapshot API: `snapshot_export.read_state` and
`snapshot_ops` over the `_core.py` engine.

- `import dnsjax.analysis` must never import JAX, even transitively:
  nothing on that path (`__init__.py`, what it imports, the JAX-free
  leaves, every flow spec) may import it. Why it can hold: the
  `_core.py` docstring; asserted by `tests/test_snapshot_export.py`.
- Outside that guarantee, and never imported from a JAX-free module:
  `transient_growth.py` (the JAX-based TG CLI, every JAX import
  deferred) and `snapshot_import.py` (the inbound direction; needs the
  solver runtime, imports JAX in-function). The lab's analysis
  (`dnsjax_lab.analysis`: `response/` may use JAX and SciPy, lazily;
  `twin/` is JAX-free) is never imported from here.
- Return data in the stored layout: a chunk reshaped to
  `meta["native_shape"][1:]` is the solver's own layout, and coordinate
  tuples are ordered to match.
- `divergence` and `curl` reproduce the solver's discrete operators
  node for node (pinned by `test_snapshot_export.py`): re-run it after
  changing any primitive. The pipe's parity follows the physical
  `m = m0·h` (`GeometryInfo.azimuthal_m0` -> `_core.radial_derivative`),
  exactly as `cylindrical.Fourier.m_is_even` does. A flow with a
  metric takes the solver's metric-weighted forms through
  `GeometryInfo.metric` (the lab's toroidal pipe:
  `dnsjax_lab.analysis.toroidal.ToroidalMetric`, which agrees to
  roundoff on the default `phys.curvature_padding` grid): change it
  with `dnsjax_lab.geometries.wall_bounded.cylindrical_curved`.
- The `*_SYSTEMS` sets mirror the registry, whose geometry and
  total-field axes overlap (`flows/registry.py`).
- The lab's `response/`: ensemble members aggregate on relative time
  while `dnsjax_lab.extensions.probes` gates `probes.bin` on the
  absolute `it % it_probes`, so harvest parents at multiples of the
  probe cadence or `response.ensemble.aggregate_tree` refuses the set.
  The pipeline and the route trade-offs:
  `dnsjax_lab/analysis/response/__init__.py`.
- `transient_growth.single_mode_state` / `mode_state_energy` are
  shared with `scripts/snapshot_perturb.py`, and `--tg.save_operator`
  exports the per-mode generators the lab's `response/` consumes:
  change either end with the other.
