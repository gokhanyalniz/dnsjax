# Tests

Every test is a standalone script (`uv run python tests/<name>.py`);
`pytest_suite.py` is the bridge, one `_SCRIPTS` row per invocation
(add one for a new script). Markers and tiers: `README.md` here. What a
script pins and its flags: its module docstring.

## Writing a test

- Configure the singletons once, at module top, before importing
  `sharding` or a geometry module. A test that re-calls
  `update_parameters()` mutates the shared `params`/`derived_params`
  and must restore the module configuration before returning (as
  `test_annular.py` does).
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
- Normalise a roundoff guard by its input, never by an output the
  operation may legitimately shrink, and set its margin to a decade
  after sweeping seeds and sizes: summation order alone moves such a
  quantity by about 1.5x.
- A guard for a silent failure must fail on the pre-fix code with a
  real input. A stub keeps the real input's awkward property (a missing
  attribute, a PAX tar header), or the test passes for the same reason
  the bug shipped.
- A script whose children stream output uses `_live.run_live` and
  `_live.report`. Shared response fixtures: `response/_common.py`.
- A launch across two hosts fakes the second one with
  `_fake_host.py` (gloo or the MPI collectives alike).

## Which script a change reaches

Banded solve and Pallas kernel:
- `test_banded_solver.py` (parity, whole tiles, cuda lowering, the
  adjoint), `test_banded_solver_sharded.py` (a `(2, 2)` mesh, also for
  the wall-normal stencil).

Geometry operators and grids:
- `test_cartesian.py`, `test_cylindrical.py`, `test_annular.py`
  (operators, band-vs-dense parity); `test_curved_pipe.py` (`--only`);
  `test_viscoelastic.py` (annular sPTT), `test_viscoelastic_pipe.py`.
- `test_integration.py` (quadrature, interpolation),
  `test_mean_mask.py` (one-hot under padding), `test_padding.py`
  (padded sizes, FFT exactness, `chunked_transform`).

Stepping:
- `test_laminar_smoke.py` (every wall-bounded flow but the curved
  pipe), `test_random_smoke.py` (every stepping machinery and its
  variants).
- `test_cnab2.py`, `test_temporal_order.py`, `test_adaptive.py`,
  `test_imm_continuity.py`, `test_energy_budget.py`,
  `test_autodiff.py` (`--only`), `test_monochromatic.py` (Kolmogorov).
- `test_wall_normal_matvec.py` (`--only`): one step under the GEMM and
  the stencil (`solver.wall_normal_matvec`) must agree, per geometry
  and legacy pass.
- `test_wall_time_stop.py` (`mpirun`): both drivers stop on
  `stop.max_wall_time` together when the ranks' clocks disagree.

Parameters, bootstrap, mesh, seeds:
- `test_param_surface.py` (registry, surfaces, the total-field axis),
  `test_bootstrap.py`, `test_seeding.py` (`--unit-only`).
- `test_device_grid.py` (`--unit-only`): the mesh across nodes, two
  hosts faked on one machine under `mpirun`.
- `test_host_placement.py` (`mpirun`, `--only`): no host array reaches
  the mesh through JAX's cross-process gather
  (`sharding.Sharding.distribute`).
- `test_mpi_communicators.py` (`mpirun`): the mesh's MPI communicators
  are open before a collective can run off the MPI thread
  (`sharding._warm_communicators`).

Initial conditions:
- `test_localized_rolls.py`, `test_rolls_smoke.py` (the roll builders),
  `test_mean_mode.py` (`--unit-only`), `test_snapshot_perturb.py`.

Snapshots, resume, analysis:
- `test_snapshot.py`, `test_resume.py` (`--unit-only`),
  `test_snapshot_import.py`, `test_snapshot_export.py` (re-run after
  changing a primitive), `test_transient_growth.py` (`--fast`),
  `test_quasi_keplerian.py`.
- `test_lowres.py` (`[lowres]` files, the static pressure, the cube
  container); multi-process `[lowres]` writes: the `*-mpi-pad` rows of
  `test_random_smoke.py`.

Streams: `test_probes.py`, `test_forcing.py`, `test_driving.py` (each
`--unit-only`).

Twin: `test_twin_unit.py`, `test_twin_driver.py` (`--only`),
`test_twin_budget.py` (`--only`, `--quick`), `test_twin_analysis.py`,
`test_twin_postprocess.py` (`--unit-only`), `test_twin_spectral_maps.py`
(skips without the `plots` group).

Response: `response/test_probes_reader.py`,
`response/test_operator_tools.py`, `response/test_ensemble.py`,
`response/test_lim.py`, `response/test_ssi.py`.
