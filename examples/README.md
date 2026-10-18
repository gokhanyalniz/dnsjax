# Examples

Four worked configurations across three of the solver's four
geometries. Each is a single annotated `parameters.toml` plus a short
README, sized to reach turbulence on **one laptop core in minutes**, and
each starts from a deterministic localized-roll perturbation — so two
runs of the same file take the same trajectory, with no seed to record.
(The annular geometry, that of Taylor–Couette flow, has no example
here: in the linearly stable regimes its transition threshold is
amplitude- and geometry-specific, with no standard minimal unit to
copy.)

| example | geometry | what it shows |
|---|---|---|
| [`kolmogorov/`](kolmogorov/) | triply-periodic | the simplest full nonlinear path in the repository: no walls, so no influence matrix and no banded solve between the initial condition and the answer |
| [`plane-couette-re500/`](plane-couette-re500/) | Cartesian | the minimal flow unit — the smallest box that holds turbulence between two shearing walls, and a turbulent state with a finite lifetime |
| [`plane-poiseuille-re5000/`](plane-poiseuille-re5000/) | Cartesian | a small channel box: subcritical transition below the linear-stability limit, under constant-bulk driving |
| [`pipe-re2500/`](pipe-re2500/) | cylindrical | a short periodic pipe at a transitional Reynolds number, where turbulence in a cell this short is transient |

A run reads `parameters.toml` from its working directory and writes its
output files there, so give each run a scratch directory of its own:

```bash
mkdir -p /tmp/run && cd /tmp/run
cp /path/to/dnsjax/examples/kolmogorov/parameters.toml .
/path/to/dnsjax/.venv/bin/dnsjax
```

Any value in the file can be overridden on the command line
(`--phys.re 200`), and `dnsjax --help <system>` lists everything a given
flow accepts.

**On a GPU**, sync the environment with JAX's CUDA build
(`uv sync --extra cuda13` in the clone; the other backends:
[On a GPU](../README.md#on-a-gpu)) and add one flag:

```bash
/path/to/dnsjax/.venv/bin/dnsjax --dist.platform cuda
```

**Reading the result.** Each example writes `stats.dat`, one row per
diagnostic interval (its format and columns:
[`docs/running.md`](../docs/running.md#the-diagnostic-streams)). For the
three wall-bounded examples the column that says whether the flow is
turbulent is the **dissipation `D`**, measured against its laminar value —
not the perturbation energy `E'` on its own, which also carries the mean
profile's deviation from laminar and can move for reasons that have
nothing to do with fluctuations: it can fall smoothly while the flow is
still fully turbulent. Kolmogorov flow is the exception, where `E'` is
the reliable indicator and `D` runs the other way; its README explains
why. Each README says what to expect.

**Two of them hold the flow rate.** The pipe and plane-Poiseuille
examples set `phys.driving` to `constant_bulk_velocity`: the flow rate
is held fixed and the pressure gradient is whatever it takes to maintain
it. That is the right way round for an example: the Reynolds number
keeps meaning what it says once the flow is turbulent, and the applied
forcing, the last column of `stats.dat` (positive when accelerating),
rises above its laminar value exactly when the flow does — a second,
independent reading of the same event as `D`. Under the default
`constant_pressure_gradient` the roles swap: the forcing is fixed and
the flow rate drops instead.

**They may stop early.** In wall-bounded boxes this small, turbulence
can be transient: it has a finite, stochastic lifetime and eventually
decays back to the laminar state, sometimes before the horizon.
`stop.check_laminarization` is on by default, so a run whose
perturbation energy collapses ends cleanly and says so instead of
integrating a laminar flow to the horizon. The check reads `E'`, which
falls below its threshold later than `D` returns to laminar, since the
mean profile relaxes on the viscous timescale: a run can reach its
horizon with the flow already laminar but `E'` still decaying.

**How big they are.** Each runs in double precision (the default):

| example | resolution | horizon |
|---|---|---|
| `kolmogorov` | $32^3$ | 200 |
| `plane-couette-re500` | $16 \times 33 \times 16$ ($n_x, n_y, n_z$) | 200 |
| `plane-poiseuille-re5000` | $16 \times 49 \times 16$ ($n_x, n_y, n_z$) | 250 |
| `pipe-re2500` | $24 \times 32 \times 32$ ($n_z, n_r, n_\theta$) | 200 |

They run on a GPU unchanged, but they are not GPU-sized problems: cases
this small do not come close to filling one, so there is no reason to
expect a GPU to help here and some reason to expect it to hurt. Sizing a
run for a GPU node is in
[`docs/scaling.md`](../docs/scaling.md#target-nodes), and how a
production-sized run scales, from 1 to 64 CPU nodes, in
[`docs/scaling.md`](../docs/scaling.md#strong-scaling-on-archer2).

The production-scale run these are the small counterpart to — a
100-diameter pipe over 500 advective time units — is walked through flag
by flag in [`docs/running.md`](../docs/running.md).

## Notebook

[The notebook](notebooks/read_snapshot_numpy_only.ipynb) reads a
snapshot with NumPy and nothing else. It is not a fifth flow
example: it runs the solver as a *subprocess* and post-processes what
that wrote, so the notebook kernel never imports JAX or the solver at
all — which is the claim it exists to check, and its last cell does.

## On a SLURM cluster

[`slurm/cpu.slurm`](slurm/cpu.slurm) is a job script for a production
run on CPU nodes, self-contained and commented throughout. Copy it
into the run folder (the one holding `parameters.toml`), fill in the
fields marked `<...>`, and submit it there with `sbatch cpu.slurm`.
Its sections say what changes for each run (the nodes, tasks, memory
and time, the device grid `(np0, np1)` and the solver's wall-clock
budget), what is set once per machine (the dnsjax checkout, its
Python, the MPI wrapper of [CPU collectives](../docs/cpu-collectives.md),
Spindle), and which defaults it applies. Every simulation parameter
stays in `parameters.toml`. By default the job also samples each node's
memory into `memwatch/<job id>/` and prints the peak and typical use per
node at the end, with a per-node sampler the full repository provides
([the lab](../README.md#the-lab)); a checkout without it skips the
record. It was tested on ARCHER2, an HPE Cray EX system.
