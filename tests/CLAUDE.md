# Tests

Every test is a standalone script (`uv run python tests/<name>.py`);
`pytest_suite.py` is the bridge, one `_SCRIPTS` row per invocation
(add one for a new script). Markers and tiers: `README.md` here. What a
script pins and its flags: its module docstring.

## Writing a test

- Configure the singletons once, at module top, before importing
  `sharding` or a geometry module (`test_cartesian.py` assigns
  `params.*` directly; the others call `update_parameters()`). A test
  that re-calls `update_parameters()` mutates the shared
  `params`/`derived_params` and must restore the module configuration
  before returning (only `test_annular.py` does).
- `taylor-couette` needs `phys.re1`, `phys.re2` and `geo.eta` set
  before the singletons are built (`update_parameters()` raises
  otherwise); the unit tests use `100` / `0` / `0.5`.
  `quasi-keplerian` takes `re1` / `r_omega` / `eta` instead (tests:
  `-1.2` and `0.71` for the last two).
- A test that builds its own FD reference grid passes the resolved
  selection, `build_*_grid(ny, params.res.fd_order,
  params.geo.wall_grid, params.geo.grid_type, params.geo.grid_stretch)`,
  never the builder defaults: they are not the resolved grid (the pipe
  resolves to `half-cgl`).
- Base flows are covered by the laminar smoke, not by per-field unit
  tests.
- The laminar smoke has `u' = 0`, so it never exercises the nonlinear
  term (a broken advection can still report `err = 0`);
  `test_random_smoke.py` does.
- A slip that only multi-process runs expose (a global array baked
  into a jit) needs a real `mpirun` row; forced CPU devices in one
  process do not catch it.
- A script whose children stream output uses `_live.run_live` and
  `_live.report`. Shared response fixtures: `response/_common.py`.

## Which script a change reaches

Banded solve and Pallas kernel:
- `test_banded_solver.py` (parity, whole tiles, cuda lowering, the
  adjoint), `test_banded_solver_sharded.py` (a `(2, 2)` mesh).

Geometry operators and grids:
- `test_cartesian.py`, `test_cylindrical.py`, `test_annular.py`
  (operators, band-vs-dense parity); `test_curved_pipe.py` (`--only`);
  `test_viscoelastic.py` (annular sPTT), `test_viscoelastic_pipe.py`.
- `test_integration.py` (quadrature, interpolation),
  `test_mean_mask.py` (one-hot under padding), `test_padding.py`
  (padded sizes, FFT exactness, `chunked_transform`).

Stepping:
- `test_laminar_smoke.py` (every wall-bounded flow but the curved pipe;
  `--np`, `--np0`), `test_random_smoke.py` (the seven stepping
  machineries and variants; `--np`).
- `test_cnab2.py`, `test_temporal_order.py`, `test_adaptive.py`,
  `test_imm_continuity.py` (`--ny`), `test_energy_budget.py`,
  `test_autodiff.py` (`--only`), `test_monochromatic.py` (Kolmogorov).

Parameters, bootstrap, seeds:
- `test_param_surface.py` (registry, surfaces, the total-field axis),
  `test_bootstrap.py`, `test_seeding.py` (`--unit-only`).

Initial conditions:
- `test_localized_rolls.py`, `test_rolls_smoke.py` (6 roll builders),
  `test_mean_mode.py` (`--unit-only`), `test_snapshot_perturb.py`.

Snapshots, resume, analysis:
- `test_snapshot.py`, `test_resume.py` (`--unit-only`),
  `test_snapshot_import.py`, `test_snapshot_export.py` (re-run after
  changing a primitive), `test_transient_growth.py` (`--fast`),
  `test_quasi_keplerian.py`.

Streams: `test_probes.py`, `test_forcing.py`, `test_driving.py` (each
`--unit-only`).

Twin: `test_twin_unit.py`, `test_twin_driver.py` (`--only`, `--seed`,
`--mean-free`), `test_twin_budget.py` (`--only`, `--ladder`,
`--seeds`, `--measure`, `--quick`), `test_twin_analysis.py`,
`test_twin_postprocess.py` (`--unit-only`), `test_twin_spectral_maps.py`
(skips without the `plots` group).

Response: `response/test_probes_reader.py`,
`response/test_operator_tools.py`, `response/test_ensemble.py`,
`response/test_lim.py`, `response/test_ssi.py`.
