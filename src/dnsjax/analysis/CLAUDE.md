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
  solver runtime, imports JAX in-function).
- `_core._axial_metric` imports a flow's `analysis_metric` factory when
  its snapshot is first read, and the registry loads every installed
  plugin's spec module: both must stay JAX-free, which
  `tests/test_snapshot_export.py` asserts before and after its reads.
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
- `transient_growth.single_mode_state` / `mode_state_energy` are
  shared with `scripts/snapshot_perturb.py`, and `--tg.save_operator`
  exports the per-mode generators the lab's `response/` consumes:
  change either end with the other.
