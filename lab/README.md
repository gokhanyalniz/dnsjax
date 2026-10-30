# dnsjax-lab

The second package in this repository: flows and features in
development alongside the [`dnsjax`](../README.md) solver.

The lab plugs into the solver through the same entry points any
package would use ([`docs/extending.md`](../docs/extending.md)): its
flows through `dnsjax.flows`, their parameter fields through
`dnsjax.parameters`, and its `[probes]` and `[force]` sections through
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

## Scripts

Performance, in [`scripts/performance/`](scripts/performance/); none
of them changes how a simulation runs:

| Script | What it does |
|---|---|
| `solver_benchmark.py` | the Pallas banded backend against the dense reference solver on the flows its `SIZES` and `EXTRA_ENTRIES` tables list: validation and timing |
| `node_benchmark.py` | a layout and strong-scaling sweep of one production problem — CPU ranks under `mpirun` or `srun`, or one process over a node's GPUs — each row's `stats.dat`, when recorded, checked against a reference; JAX-free driver |
| `pallas_solve_profile.py` | GPU profile of the banded solve and its share of a step |
| `memory_budget.py` | per-rank memory of the step as run, on any layout, from XLA's buffer assignment, offline |
| `memory_watch.py` | per-node memory sampler for multi-process runs, with a summary of each node's peak and typical use, JAX-free |
| `step_sharing_census.py` | which step-invariant wall-normal GEMMs one iterative-CN step shares between its predictor and correction passes, from the compiled program |
| `scaling_figure.py` | the strong-scaling figure and tables of [`docs/scaling.md`](../docs/scaling.md), from the committed CSV of measured runs (needs the `plots` group) |

Functionality checks, in [`scripts/diagnostics/`](scripts/diagnostics/):

| Script | What it does |
|---|---|
| `grad_probe.py` | which stepping configurations differentiate in forward and reverse mode, each gradient checked against a finite difference ([`docs/differentiability.md`](../docs/differentiability.md)) |
| `pallas_tiling_diagnostic.py` | GPU harness that bisects the observed miscompile of the Pallas banded kernel's masked partial-tile path, which the kernel's whole-tile padding avoids, and checks that the padded solve passes |
| `gds_probe.py` | cluster diagnostic for the snapshot's GPUDirect Storage path: whether it engages, and what limits its transfer rate |
| `snapshot_holes.py` | whether every byte of an archive written while the filesystem was failing reached it: the holes left in the archive's sparse layout, each placed by member, component, rows and writing devices, from metadata alone, with a sparse probe file as the control |

## Tests

36 standalone scripts in [`tests/`](tests/), laid out like the
solver's, one `uv run pytest` at the repository root running both
suites. The conventions, and the split of a script covering both the
solver's flows and the lab's: [`tests/README.md`](../tests/README.md).

## Limitations

- **The viscoelastic laminar states are exact only at
  $\epsilon = \kappa = 0$.** At the defaults the sPTT
  `start_from_laminar` pair is not a steady state (its velocity neglects
  the shear-thinning $\epsilon$ brings, its conformation the stress
  diffusion $\kappa$ brings), so `E'` keeps a floor and the
  laminarization check cannot fire. Fixing it needs a nonlinear solve for
  the velocity profile; the defect, not small at the defaults, is
  measured in the flow modules' Design notes.

## References

- N. Phan-Thien and R. I. Tanner, *A new constitutive equation derived from
  network theory*, J. Non-Newtonian Fluid Mech. **2**(4), 353–365 (1977). The
  (simplified) Phan-Thien–Tanner viscoelastic model.

The papers the lab's code reproduces are cited at the implementing
code.
