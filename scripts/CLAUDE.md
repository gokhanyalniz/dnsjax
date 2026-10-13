# Offline tools

Each script's module docstring is its manual. Only
`snapshot_figure.py` and the lab's `twin_spectral_maps.py` and
`performance/scaling_figure.py` need matplotlib: run them as
`uv run --group plots python <path>` (`uv sync` alone does not install
the `plots` group). The seeds in the benchmarks and diagnostics are
fixed on purpose.

- `snapshot_perturb.py`: inject a scaled single-mode perturbation into
  a snapshot (`[perturb]`).
- `snapshot_figure.py`: velocity-plane figures and animations (the
  `docs/figures/` sources); `_figure_common.py` holds its JAX-free
  per-flow helpers.
- `wall_normal_resolution.py`: size `res.ny` / `fd_order` /
  `geo.grid_type` against a Chebyshev order (Cartesian).

The lab's scripts, in `lab/scripts/`:
- `ensemble_setup.py`: JAX-free `harvest` / `build` / `build-twin` of
  ensemble member trees from a snapshot archive.
- `twin_postprocess.py`: rebuild a twin member's streams from its
  snapshot pairs (`[recon]`).
- `twin_spectral_maps.py`: `(λ, y)` maps, `(y, t)` / `(λ, t)`
  histories, moment budgets, decorrelation fronts and growth-law
  figures of the twin `(y, k)` streams over an ensemble.
- `random_ic_calibrate.py`: score and sweep the random IC's `(y, k)`
  shape against a recorded ensemble (`[calib]`).
- `performance/solver_benchmark.py`: pallas-vs-dense validation and
  benchmark, every flow (`--cpu-smoke`).
- `performance/node_benchmark.py`: JAX-free layout and strong-scaling
  sweep of one problem, `mpirun` or `srun` over nodes, each row's
  `stats.dat` checked against a reference (`--dry-run`, `--cpu-smoke`);
  imports its sibling `solver_benchmark.py`.
- `performance/pallas_solve_profile.py`: GPU profile of the banded solve
  and its share of a step (`--cpu-smoke`).
- `performance/memory_budget.py`: per-rank memory of the as-run step on
  any layout, offline (forced CPU devices).
- `performance/memory_watch.py`: JAX-free per-node memory sampler.
- `performance/scaling_figure.py`: the strong-scaling figure (light and
  dark SVG) and tables of `docs/scaling.md`, from
  `docs/figures/*-scaling.csv`.
- `diagnostics/grad_probe.py`: forward/reverse differentiability matrix
  with a finite-difference check.
- `diagnostics/pallas_tiling_diagnostic.py`: GPU harness for the Triton
  partial-tile miscompile.
- `diagnostics/gds_probe.py`: cluster diagnostic for the snapshot GDS
  path (`--cpu-smoke`).
