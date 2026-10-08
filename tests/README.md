# Tests

53 standalone scripts. Each one is a program with a `__main__` runner,
runnable on its own:

```bash
uv run python tests/test_cartesian.py
uv run python tests/test_transient_growth.py --system pipe
uv run python tests/test_laminar_smoke.py --np 2
```

The whole suite runs on **CPU**. Nothing here needs a GPU: the
Pallas/Triton kernel is validated in interpret mode and separately
*lowered* for CUDA inside an abstract GPU mesh, so a compilation
regression is caught on a machine with no GPU in it.

## Why scripts and not plain pytest cases

`dnsjax` captures its configuration in module-level singletons — the
resolved parameters, the device mesh, each geometry's wavenumber arrays
— at **import** time, and JAX's platform and precision must be fixed
before any of that happens. A test therefore owns its process:

- Importing a test module *is* configuring it. If pytest collected these
  files, the top level would execute with the singletons unset.
- A sweep over the resolution or any parameter but `dt` needs one
  process per value, because the value is baked into the jitted steppers
  when they are traced. `dt` alone is rebuilt in place (the flow
  module's `set_dt`), which the temporal-order study relies on.
- Several scripts launch real `mpirun` multi-device runs, or spawn
  subprocesses with forced CPU device counts, to exercise sharded paths
  that are invisible on one device.

So the scripts are the source of truth, and each of them decides its own
platform, precision and device layout before importing anything from
`dnsjax.sharding` onwards.

## The pytest bridge

`pytest_suite.py` is the only file pytest collects. It shells each
script out as a subprocess, asserts a zero exit code, and surfaces the
output tail on failure — the scripts stay authoritative, and `pytest`
becomes a way to run them all with markers and a summary.

```bash
uv run pytest -m "not slow and not mpi"   # offline: no mpirun launch
uv run pytest -m "not slow"               # + the quick mpirun rows
uv run pytest                             # everything available
uv run pytest -k padding                  # one script
```

Output streams live as each script runs (`_live.py` tees it, and
`addopts = -s` is set in `pyproject.toml`), so a long run shows progress
and can be aborted early.

Two markers:

- **`mpi`** — the script launches solver runs through `mpirun` (even at
  `-np 1`). Skipped automatically when `mpirun` is not on `PATH`; where
  a script has a `--unit-only` half, that half runs instead.
- **`slow`** — full solver integration runs, `dt` sweeps, or the
  published transient-growth values. Minutes each.

Adding a test script means adding one row to `_SCRIPTS` in
`pytest_suite.py`.

## Running them well

Run **one** heavy suite at a time. Each invocation is already serial
internally and deliberately leaves JAX's CPU thread pool unpinned, so
concurrent invocations oversubscribe the machine and have produced
spurious failures. Read a verdict from a captured log rather than from a
`tail` of a run you have not seen in full, and treat a failure that does
not reproduce on a clean serial rerun as contention.

## What each script covers

Every script's module docstring states what it pins and how to run its
variants; the claims the scripts back are mapped in
[`../docs/validation.md`](../docs/validation.md). By area:

| Area | Scripts (`test_` prefix dropped) |
|---|---|
| Banded solve and Pallas kernel | `banded_solver`, `banded_solver_sharded` |
| Geometry operators and grids | `cartesian`, `cylindrical`, `annular`, `curved_pipe`, `viscoelastic`, `viscoelastic_pipe`, `integration`, `mean_mask`, `padding` |
| Stepping | `laminar_smoke`, `random_smoke`, `cnab2`, `temporal_order`, `adaptive`, `imm_continuity`, `energy_budget`, `autodiff`, `monochromatic`, `wall_normal_matvec`, `wall_time_stop` |
| Parameters, bootstrap, device mesh, seeds | `param_surface`, `bootstrap`, `seeding`, `device_grid`, `host_placement`, `mpi_communicators` |
| Initial conditions | `localized_rolls`, `rolls_smoke`, `mean_mode`, `snapshot_perturb` |
| Snapshots, resume, analysis | `snapshot`, `resume`, `snapshot_import`, `snapshot_export`, `transient_growth`, `quasi_keplerian`, `lowres` |
| Diagnostic streams | `probes`, `forcing`, `driving` |
| Twin runs | `twin_unit`, `twin_driver`, `twin_budget`, `twin_analysis`, `twin_postprocess`, `twin_spectral_maps` |
| Response analysis (`response/`) | `probes_reader`, `operator_tools`, `ensemble`, `lim`, `ssi` |
