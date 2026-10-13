# Tests

Every test is a standalone script (`uv run python tests/<name>.py`,
the lab's `lab/tests/<name>.py`); each suite's `pytest_suite.py` is its
bridge, one `_SCRIPTS` row per invocation (add one for a new script).
Markers and tiers: `README.md` here. What a script pins and its flags:
its module docstring. "(both suites)" below marks a script with a lab
half of the same name in `lab/tests/`.

## Writing a test

- Configure the singletons once per process, before importing
  `sharding` or a geometry module (a test that needs several
  configurations runs one subprocess per configuration). A test that
  re-calls `update_parameters()` mutates the shared
  `params`/`derived_params` and must restore the module configuration
  before returning (as `test_annular.py` does).
- Offline multi-device tests force CPU devices with
  `XLA_FLAGS=--xla_force_host_platform_device_count=N` and set
  `params.dist.np0`/`np1` before importing `sharding`. Never combine
  that with `mpirun`.
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
  `_live.report`. Shared response fixtures:
  `lab/tests/response/_common.py`.
- A launch across two hosts fakes the second one with
  `_fake_host.py` (gloo or the MPI collectives alike).
- A script covering core and lab flows is split in two of one name:
  the shared runner in `tests/_<name>.py`, free of side effects at
  import; the core rows in `tests/test_<name>.py`; the lab rows in
  `lab/tests/test_<name>.py`, whose first import is
  `import _repo  # noqa: F401`. A shared runner never branches on a
  system name: it takes registry lists, spec fields or per-row hooks.

## Which script a change reaches

Core and lab: `test_layering.py`: the core never names the lab
(imports, system strings, lab fields and sections); `test_plugins.py`:
what a plugin's entry points add (faked), and each loader's refusals.

Banded solve and Pallas kernel:
- `test_banded_solver.py` (parity, whole tiles, cuda lowering, the
  adjoint), `test_banded_solver_sharded.py` (a `(2, 2)` mesh, also for
  the wall-normal stencil).

Geometry operators and grids:
- `test_cartesian.py`, `test_cylindrical.py` (both suites),
  `test_annular.py` (operators, band-vs-dense parity); the lab's
  `test_curved_pipe.py` (`--only`), `test_viscoelastic.py` (annular
  sPTT), `test_viscoelastic_pipe.py`.
- `test_integration.py` (quadrature, interpolation),
  `test_mean_mask.py` (one-hot under padding), `test_padding.py`
  (padded sizes, FFT exactness, `chunked_transform`).

Stepping:
- `test_laminar_smoke.py` (both suites: every wall-bounded flow but
  the curved pipe), `test_random_smoke.py` (both suites: every stepping
  machinery and its variants).
- `test_cnab2.py` and `test_adaptive.py` (both suites),
  `test_temporal_order.py`, `test_imm_continuity.py`,
  `test_energy_budget.py`, `test_autodiff.py` (`--only`),
  `test_monochromatic.py` (Kolmogorov).
- `test_wall_normal_matvec.py` (both suites, `--only`): one step under
  the GEMM and the stencil (`solver.wall_normal_matvec`) must agree,
  per geometry and legacy pass.
- `test_wall_time_stop.py` (both suites, `mpirun`): both drivers stop
  on `stop.max_wall_time` together when the ranks' clocks disagree.
- `test_run_status.py` (both suites; `--unit-only`, `--no-mpi`,
  `--mpi-only`): the status files and the stop request, in both
  drivers.

Parameters, bootstrap, mesh, seeds:
- `test_param_surface.py` (both suites: registry, surfaces and their
  `--help`, the total-field axis), `test_bootstrap.py`,
  `test_seeding.py` (both suites, `--unit-only`).
- `test_device_grid.py` (`--unit-only`): the mesh across nodes, two
  hosts faked on one machine under `mpirun`.
- `test_host_placement.py` (both suites, `mpirun`, `--only`): no host
  array reaches the mesh through JAX's cross-process gather
  (`sharding.Sharding.distribute`).
- `test_mpi_communicators.py` (`mpirun`): the mesh's MPI communicators
  are open before a collective can run off the MPI thread
  (`sharding._warm_communicators`).

Initial conditions:
- `test_localized_rolls.py`, `test_rolls_smoke.py` (the roll builders;
  both suites), `test_mean_mode.py` (`--unit-only`),
  `test_snapshot_perturb.py` (both suites).

Snapshots, resume, analysis:
- `test_snapshot.py`, `test_resume.py` (`--unit-only`),
  `test_snapshot_export.py` (re-run after changing a primitive),
  `test_transient_growth.py` (`--fast`), all both suites;
  `test_snapshot_import.py`, `test_wedge.py` (`geo.m0`; the lab's
  `test_quasi_keplerian.py` holds its flow's cases).
- `test_lowres.py` (both suites: `[lowres]` files, the static
  pressure, the cube container); multi-process `[lowres]` writes: the
  `*-mpi-pad` rows of `test_random_smoke.py`.

Streams: `test_driving.py`; `test_run_hooks.py` (an extension
section's `RunHooks`: call order, inert default, modify after every
save, abort on a report); the lab's `test_probes.py`,
`test_forcing.py` (each `--unit-only`).

Twin (all in `lab/tests/`): `test_twin_unit.py`, `test_twin_driver.py`
(`--only`), `test_twin_budget.py` (`--only`, `--quick`),
`test_twin_analysis.py`, `test_twin_postprocess.py` (`--unit-only`),
`test_twin_spectral_maps.py` (skips without the `plots` group).

Response (all in `lab/tests/response/`): `test_probes_reader.py`,
`test_operator_tools.py`, `test_ensemble.py`, `test_lim.py`,
`test_ssi.py`.
