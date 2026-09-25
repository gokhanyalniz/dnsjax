# dnsjax.analysis

The external-facing snapshot API: `snapshot_export.read_state` and
`snapshot_ops` over the `_core.py` engine.

- `import dnsjax.analysis` must never import JAX, even transitively:
  nothing on that path (`__init__.py`, what it imports, the JAX-free
  leaves, every flow spec) may import it. Why it can hold: the
  `_core.py` docstring; asserted by `tests/test_snapshot_export.py`.
- Outside that guarantee, and never imported from a JAX-free module:
  `transient_growth.py` (the JAX-based TG CLI, every JAX import
  deferred), `snapshot_import.py` (the inbound direction; needs the
  solver runtime, imports JAX in-function) and `response/` (may use JAX
  and SciPy, lazily). `twin/` is JAX-free but not imported by
  `__init__.py`.
- Return data in the stored layout: a chunk reshaped to
  `meta["native_shape"][1:]` is the solver's own layout, and coordinate
  tuples are ordered to match. Never transpose.
- `divergence` and `curl` reproduce the solver's discrete operators
  node for node (pinned by `test_snapshot_export.py`): re-run it after
  changing any primitive. The pipe's parity follows the physical
  `m = m0·h` (`GeometryInfo.azimuthal_m0` -> `_core.radial_derivative`),
  exactly as `cylindrical.Fourier.m_is_even` does.
- The `*_SYSTEMS` sets mirror the registry. The geometry, rheology and
  total-field axes overlap, so an ordered branch mixing them tests
  rheology first (`flows/registry.py`).
- `response/`: ensemble members aggregate on relative time while
  `__main__` gates `probes.bin` on the absolute `it % it_probes`, so
  harvest parents at multiples of the probe cadence or
  `response.ensemble.aggregate_tree` refuses the set. The pipeline and
  the route trade-offs: `response/__init__.py`.
- `transient_growth.py`: the scope is `WALL_BOUNDED_TG_SYSTEMS`
  (derived from `FlowSpec.total_field`); `--tg.save_operator` exports
  the per-mode generators `response/` consumes; `single_mode_state` /
  `mode_state_energy` are shared with `scripts/snapshot_perturb.py`.
