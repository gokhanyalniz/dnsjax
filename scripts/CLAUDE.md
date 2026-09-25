# Offline tools

Each script's module docstring is its manual. Only
`twin_spectral_maps.py` and `snapshot_figure.py` need matplotlib: run
them as `uv run --group plots python scripts/<name>.py` (`uv sync`
alone does not install the `plots` group). The seeds in the benchmarks
and diagnostics are fixed on purpose.

- `snapshot_perturb.py`: inject a scaled single-mode perturbation into
  a snapshot (`[perturb]`).
- `ensemble_setup.py`: JAX-free `harvest` / `build` / `build-twin` of
  ensemble member trees from a snapshot archive.
- `twin_postprocess.py`: rebuild a twin member's streams from its
  snapshot pairs (`[recon]`).
- `twin_spectral_maps.py`: `(λ, y)` and `(y, t)` maps of the twin
  `(y, k)` streams over an ensemble.
- `random_ic_calibrate.py`: score and sweep the random IC's `(y, k)`
  shape against a recorded ensemble (`[calib]`).
- `snapshot_figure.py`: velocity-plane figures and animations (the
  `docs/figures/` sources); `_figure_common.py` holds its JAX-free
  per-flow helpers.
- `wall_normal_resolution.py`: size `res.ny` / `fd_order` /
  `geo.grid_type` against a Chebyshev order (Cartesian).
- `pallas_tiling_diagnostic.py`: GPU harness for the Triton
  partial-tile miscompile.
- `pallas_solve_profile.py`: GPU profile of the banded solve and its
  share of a step (`--solve-only`, `--steps-only`, `--cpu-smoke`).
- `solver_benchmark.py`: pallas-vs-dense validation and benchmark
  (`--cpu-bench`, `--cpu-smoke`).
- `node_benchmark.py`: JAX-free layout sweep of one problem on one
  node (`--dry-run`, `--cpu-smoke`).
- `grad_probe.py`: forward/reverse differentiability matrix with a
  finite-difference check (`--full`, `--dist.platform cuda`).
- `gds_probe.py`: cluster diagnostic for the snapshot GDS path
  (`--env-only`, `--end-to-end`, `--end-to-end-only`, `--cpu-smoke`).
