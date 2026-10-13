# dnsjax-lab

The second package in this repository: features developed alongside
the [`dnsjax`](../README.md) solver and kept out of its releases. It is
never published — no PyPI upload, no Zenodo archive (its `pyproject.toml`
carries the `Private :: Do Not Upload` classifier, which PyPI refuses)
— and lives in this repository only.

The lab plugs into the solver through the same entry points any
package would use ([`docs/extending.md`](../docs/extending.md)): its
flows through `dnsjax.flows`, their parameter fields through
`dnsjax.parameters`, and its run-loop sections through
`dnsjax.extensions`. The solver never imports the lab, and
`tests/test_layering.py` fails on any solver file that names it.

## Install and run

`uv sync` at the repository root installs the lab with the solver, in
one environment (`uv sync --no-group lab` leaves it out). Its flows then
appear in `dnsjax --help` beside the solver's own, and run the same way:

```bash
mkdir -p /tmp/run && cd /tmp/run
/path/to/dnsjax/.venv/bin/dnsjax --phys.system dean --phys.re 1000 \
  --geo.eta 0.5 --stop.max_sim_time 10
```

## Flows

| Flow | Geometry | Laminar base / driving | Defining controls |
|---|---|---|---|
| **Curved pipe** | cylindrical (toroidal) | uniform body force on a pipe bent onto a circle, total field; no closed-form laminar state | `re`, `curvature`, `lz` |
| **Viscoelastic pipe** | cylindrical (sPTT) | pressure-driven, 9-component total field | `el`, `wi`, `beta`, `epsilon`, `kappa`, `lz` |
| **Quasi-Keplerian** | annular | $U_\theta = A_0 r + B_0/r$, Rayleigh-stable co-rotation | `re1`, `r_omega`, `eta`, `lz` |
| **Dean** | annular | azimuthal body force (a pressure-driven *curved* channel), total field | `re`, `eta`, `lz` |
| **Viscoelastic Dean** | annular (sPTT) | azimuthal body force (a pressure-driven *curved* channel), 9-component total field | `el`, `wi`, `beta`, `epsilon`, `kappa`, `delta`, `lz` |

Their equations and conventions:
[`docs/numerics.md`](docs/numerics.md). What they are checked against:
[`docs/validation.md`](docs/validation.md).

## Runs and analyses

| | |
|---|---|
| **Twin-run perturbation growth** | `dnsjax-twin` steps a reference snapshot and a perturbed copy in lockstep, streaming difference-field energy, $y$-resolved spectra with the matching budget, $(k_z, k_x)$ spectra, and subsampled $(y, k_z, k_x)$ cubes of both — [`twin/`](src/dnsjax_lab/twin/README.md) |
| **Runtime streams** | `[probes]` records the wall-normal profiles of chosen spectral modes, `[force]` drives them with white-in-time kicks — [`extensions/`](src/dnsjax_lab/extensions/README.md) |
| **Response and identification** | from a turbulent run to a data-driven linear operator, three interchangeable ways — [`analysis/response/`](src/dnsjax_lab/analysis/response/README.md) |

## Scripts

Analysis and ensembles, in [`scripts/`](scripts/):

| Script | What it does |
|---|---|
| `twin_spectral_maps.py` | premultiplied $(\lambda, y)$ maps, histories, moment budgets, decorrelation fronts and growth laws of the twin $(y, k)$ streams over an ensemble (needs the `plots` group) |
| `ensemble_setup.py` | harvests statistically independent parents from a run and builds member trees for response ensembles (`build`) or twin ensembles (`build-twin`), JAX-free |
| `random_ic_calibrate.py` | scores and sweeps the random initial condition's $(y, k)$ shape against a recorded twin ensemble (`[calib]`) |
| `twin_postprocess.py` | rebuilds a twin member's difference-field streams from its snapshot pairs (`[recon]`) |

Performance, in [`scripts/performance/`](scripts/performance/), kept
apart from the analysis scripts; none of them changes how a simulation
runs:

| Script | What it does |
|---|---|
| `solver_benchmark.py` | the Pallas banded backend against the dense reference solver, every flow: validation and timing |
| `node_benchmark.py` | a layout and strong-scaling sweep of one production problem under `mpirun` or `srun`, each row's `stats.dat` checked against a reference; JAX-free driver |
| `pallas_solve_profile.py` | GPU profile of the banded solve and its share of a step |
| `memory_budget.py` | per-rank memory of the step as run, on any layout, from XLA's buffer assignment, offline |
| `memory_watch.py` | per-node memory sampler for multi-process runs, with a summary of each node's peak and typical use, JAX-free |
| `scaling_figure.py` | the strong-scaling figure and tables of [`docs/scaling.md`](../docs/scaling.md), from the committed CSV of measured runs (needs the `plots` group) |

Functionality checks, in [`scripts/diagnostics/`](scripts/diagnostics/):

| Script | What it does |
|---|---|
| `grad_probe.py` | which stepping configurations differentiate in forward and reverse mode, each gradient checked against a finite difference ([`docs/differentiability.md`](../docs/differentiability.md)) |
| `pallas_tiling_diagnostic.py` | GPU harness that bisects the partial-tile Triton miscompile the kernel's whole-tile padding avoids |
| `gds_probe.py` | cluster diagnostic for the snapshot write path through GPUDirect Storage |

## Tests

36 standalone scripts in [`tests/`](tests/), laid out like the
solver's ([`tests/README.md`](../tests/README.md)): one `uv run pytest`
at the repository root runs both suites. A script covering both the
solver's flows and the lab's is split in two of the same name, one in
each suite, sharing a runner in the solver's `tests/`.

## Limitations

- **The viscoelastic laminar states are exact only at $\epsilon = 0$.**
  At the default $\epsilon = 10^{-3}$ the sPTT `start_from_laminar`
  profile does not balance momentum exactly, so those two flows carry a
  small residual and their laminarization check cannot fire. Fixing it
  needs a shooting solve; the size of the discrepancy is recorded in the
  flow modules.
- **`dnsjax-twin` is Cartesian wall-bounded, fixed step.**

## References

- N. Phan-Thien and R. I. Tanner, *A new constitutive equation derived from
  network theory*, J. Non-Newtonian Fluid Mech. **2**(4), 353–365 (1977). The
  (simplified) Phan-Thien–Tanner viscoelastic model.

The papers each analysis or flow reproduces are cited at its
implementing code and in the subpackage READMEs.
