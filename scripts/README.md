# Offline tools

Standalone programs that meet the solver through its snapshots and
streams, or launch it as a subprocess; none of them changes how a
simulation runs. Each script's module docstring is its manual. The
figure scripts need the `plots` dependency group:

```bash
uv run --group plots python scripts/snapshot_figure.py ...
```

Several take `--cpu-smoke`, a tiny CPU run that exercises the script
without its target hardware, or `--dry-run`, which prints what would
run; their docstrings say which.

## Figures

| Script | What it does |
|---|---|
| `snapshot_figure.py` | velocity-plane figures and animations from a snapshot, JAX-free (the figures in the README and the docs); `_figure_common.py` holds its per-flow helpers |
| `scaling_figure.py` | the strong-scaling figure and tables of [`docs/scaling.md`](../docs/scaling.md), from the committed CSV of measured runs |
| `twin_spectral_maps.py` | premultiplied $(\lambda, y)$ maps, histories, moment budgets, decorrelation fronts and growth laws of the [twin](../src/dnsjax/twin/README.md) $(y, k)$ streams over an ensemble |

## Initial conditions and ensembles

| Script | What it does |
|---|---|
| `snapshot_perturb.py` | injects a scaled single-mode perturbation into a snapshot (`[perturb]`) |
| `ensemble_setup.py` | harvests statistically independent parents from a run and builds member trees for [response](../src/dnsjax/analysis/response/README.md) ensembles (`build`) or twin ensembles (`build-twin`), JAX-free |
| `random_ic_calibrate.py` | scores and sweeps the random initial condition's $(y, k)$ shape against a recorded twin ensemble (`[calib]`) |
| `twin_postprocess.py` | rebuilds a twin member's difference-field streams from its snapshot pairs (`[recon]`) |

## Resolution, performance and memory

| Script | What it does |
|---|---|
| `wall_normal_resolution.py` | how many wall-normal modes a finite-difference grid resolves: sizes `res.ny`, `fd_order` and `geo.grid_type` against a Chebyshev order (Cartesian) |
| `node_benchmark.py` | a layout and strong-scaling sweep of one production problem under `mpirun` or `srun`, each row's `stats.dat` checked against a reference; JAX-free driver |
| `memory_budget.py` | per-rank memory of the step as run, on any layout, from XLA's buffer assignment, offline |
| `memory_watch.py` | per-node memory sampler for multi-process runs, with a summary of each node's peak and typical use, JAX-free |
| `solver_benchmark.py` | the Pallas banded backend against the dense reference solver: validation and timing |
| `pallas_solve_profile.py` | GPU profile of the banded solve and its share of a step |

## Diagnostics

| Script | What it does |
|---|---|
| `grad_probe.py` | which stepping configurations differentiate in forward and reverse mode, each gradient checked against a finite difference ([`docs/differentiability.md`](../docs/differentiability.md)) |
| `pallas_tiling_diagnostic.py` | GPU harness that bisects the partial-tile Triton miscompile the kernel's whole-tile padding avoids |
| `gds_probe.py` | cluster diagnostic for the snapshot write path through GPUDirect Storage |
