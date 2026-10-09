# Running a simulation

A worked example, the four start modes, the seed contract, and what a
run writes while it goes. Start at the [README](../README.md) for the
solver itself, at [`configuration.md`](configuration.md) for how the
parameter layers combine, and at [`snapshots.md`](snapshots.md) for what
comes out at the end.

## Launching

A run that fits in one process is launched directly, with no MPI involved;
only a multi-process run goes through `mpirun -np N`, invoking the
environment's `dnsjax` console script directly — `uv run` does not compose
with `mpirun`, and `python -m dnsjax` is the equivalent module form. Output
files (`stats.dat`, snapshots, …) are written to the current directory, so
launch from a scratch directory:

```bash
.venv/bin/dnsjax \
  --phys.system pipe \
  --phys.re 2300 \
  --geo.lz 200 \
  --geo.grid_type half-cgl \
  --res.nz 512 --res.nr 48 --res.ntheta 96 --res.fd_order 8 \
  --step.scheme iterative-cn --step.dt 0.01 \
  --init.localized_rolls True \
  --init.localized_rolls_amplitude 0.2 --init.localized_rolls_width 2.0 \
  --stop.max_sim_time 500 \
  --outs.it_stats 100 --outs.it_snapshot 5000 \
  --dist.platform cpu
```

Every flow exposes only the parameters that apply to it, under the names
natural to its geometry — `dnsjax --help` lists the global parameters and
the implemented flows, `dnsjax --help pipe` the pipe's own surface, and
`dnsjax --sample-toml pipe` prints an annotated configuration template. A
parameter that does not belong to the selected flow is an error (on the
command line and in `parameters.toml` alike), not a silently ignored knob.

## Reading the example

The example above runs a **100-diameter pipe at Re = 2300**, started from
a compact localized-roll perturbation, on a single CPU device. Every
problem-defining parameter — the physics, the geometry, the resolution,
and the time integrator — is written out explicitly, so switching to
another flow is a matter of editing values rather than learning the
defaults.

It is a production-scale run: seven million real degrees of freedom
carried over 500 advective time units. For something that finishes while
you watch, [`examples/`](../examples/) ships four laptop-sized
configurations that reach turbulence in minutes on one core.

Reading the flags:

- `--phys.system pipe --phys.re 2300` — the flow and its Reynolds number.
- `--geo.lz 200` — the axial length is 100 pipe diameters ($D = 2$). The
  azimuthal extent is not settable: it is the full circle, or the
  $2\pi/m_0$ wedge when the `--geo.m0` symmetry restriction is used.
- `--geo.grid_type half-cgl` — the radial grid; `half-cgl` is the default
  for a pipe under `iterative-cn`, while `cnab2` uses `rigged-cgl` instead
  (both are halves of a Chebyshev grid that avoid the axis — see
  [Grids](numerics.md#grids)).
- `--res.nz 512 --res.nr 48 --res.ntheta 96` — axial, radial, and azimuthal
  resolution, with eighth-order (the default) finite differences in the
  radial direction.
- `--step.scheme iterative-cn --step.dt 0.01` — the default
  predictor–corrector integrator at a wall-bounded-safe step.
- `--init.localized_rolls …` — a compact, deterministic finite-amplitude
  perturbation (peak amplitude 0.2) that seeds transition.
- `--stop.max_sim_time 500` — integrate 500 advective units past the
  initial condition, so here $t = 500$ (transition develops over
  $O(100)$ units; the run also stops early if the flow relaminarizes).
- `--dist.platform cpu` — a single CPU device.

This configuration fits comfortably in laptop memory — the
[Memory footprint](scaling.md#memory-footprint) section shows how to
estimate any configuration. **Switching flows** is a one-line change:
`--phys.system taylor-couette --phys.re1 … --phys.re2 … --geo.eta …`, or
`--phys.system kolmogorov --geo.lx … --geo.lz …`, and so on per the
[flow table](../README.md#flows-and-geometries).

## The TOML form

The same run can be expressed as a `parameters.toml` in the working
directory:

```toml
[phys]
system = "pipe"
re = 2300            # bulk/diameter Reynolds number (= centreline/radius)

[geo]
lz = 200.0           # axial length = 100 pipe diameters
# grid_type defaults to "half-cgl" for pipe + iterative-cn (auto-resolved)

[res]
nz = 512             # axial Fourier modes
nr = 48              # radial finite-difference points
ntheta = 96          # azimuthal Fourier modes
fd_order = 8

[init]
localized_rolls = true
localized_rolls_amplitude = 0.2   # peak |u'| of the perturbation
localized_rolls_width = 2.0       # axial localization half-width

[step]
dt = 0.01
scheme = "iterative-cn"

[outs]
it_stats = 100
it_snapshot = 5000

[stop]
max_sim_time = 500.0
# check_laminarization = true (default) stops the run if the flow relaminarizes
```

## Start modes

`--init.localized_rolls` is one of **four start modes**, resolved in a
fixed precedence: a supplied `init.snapshot` wins over everything, then
`start_from_laminar` (the analytical base state), then
`localized_rolls`, then `random_field` — which is the **default**, so a
run with no snapshot and no explicit mode starts from a random
divergence-free field. The random builder takes
`--init.random_amplitude` / `_smoothness` / `_seed`, the wall-normal
pair `_wall_smoothness` / `_wall_confinement`, plus
`_conformation_amplitude` where it applies; the roll builder adds
`--init.localized_rolls_wavelength` to the amplitude and width shown
above. On the two plane channels the random field can also perturb the
$(k_x, k_z) = (0, 0)$ mean profile (`--init.random_mean_flow`, off by
default), conditioned on that mode's conservation laws — an unchanged
mean pressure gradient, which under no-slip is compatibility at both
walls, and an unchanged bulk velocity in each direction whose mean the
driving holds — so the perturbation reaches the mean flow without
contradicting what the run is holding fixed.
Every other flow declares the field and refuses it,
rather than appearing to offer something it does not implement. A path
given to `--init.snapshot` that is not a dnsjax snapshot **aborts**
rather than falling through to the random default, so a typo cannot
quietly start a different calculation.

## Seeds

Leave `--init.random_seed` unset and the run **draws one from the
system entropy pool**, prints it with its source, and records it in the
snapshot — so a batch of runs launched the same way explores different
realizations, and any one of them replays exactly by passing its
printed seed back. The same holds for `--twin.seed` and `--force.seed`.
A run that draws nothing (laminar, rolls, or a resume) never asks for
entropy; one that would draw and cannot reach a source stops rather
than falling back to a fixed value.

## Moving frame

One default worth knowing: the pipes integrate in a frame translating at
their laminar bulk velocity $1/2$, and plane-Poiseuille in one at $2/3$,
and their snapshots are stored in that frame; pass `--phys.u_grid 0` for
the lab frame (see
[Temporal discretization](numerics.md#temporal-discretization)).

## What a run writes

The code's git revision, the final working
parameters, and the physical-space resolution are printed at startup; the
first step takes noticeably longer than the rest (JIT compilation); and a
timing summary is printed at the end. Statistics stream to `stats.dat`
(with `steps.dat` and `corrector.dat` for the CFL and corrector
diagnostics), and snapshots appear as `state00000.tar` (the initial
condition), `state00001.tar`, and so on. Runs end gracefully — at
`max_sim_time`, at an ISO 8601 `stop.max_wall_time` budget, on
relaminarization, or on request (below), each writing a final snapshot
first; on SIGTERM/SIGINT they flush the diagnostic buffers and exit at
once — so interrupted runs stay consistent with their outputs; a NaN or
inf in any diagnostic instead aborts the run at once with a line naming
the quantity, rather than spending the budget on a broken state.

### Stopping a run

While a run lives, its directory holds a file named `RUNNING` (its
start time, host, process id and, under SLURM, job id). To stop the
run gracefully, delete it:

```bash
rm RUNNING
```

Within `outs.it_error_check` steps (10 by default) every process
leaves the stepping loop together. The run writes its final snapshot
and flushes every stream, exactly as at the end of a `max_sim_time`
horizon. `kill -USR1` sent to any of its processes, or to `mpirun`
(which forwards it to every rank), does the same. A SLURM job
launched with `srun` can have the scheduler send that signal to every
task ahead of its time limit: `#SBATCH --signal=USR1@300` sends it five
minutes before. Setting `stop.max_wall_time` below the limit does the
same without a signal. The example job script
[`examples/slurm/cpu.slurm`](../examples/slurm/cpu.slurm) sets both,
with the signal as the backstop.

When the run ends, `RUNNING` is replaced by one file saying how:

| File | The run | Exit code |
|---|---|---|
| `FINISHED` | reached `max_sim_time` or `max_wall_time`, or relaminarized | 0 |
| `STOPPED` | stopped on request | 0 |
| `TERMINATED` | failed: the corrector did not converge (4); a NaN or inf (3); SIGTERM or SIGINT (128 + the signal number); anything else (1) | as listed |

Each file gives the start and end times, the reason and the newest
snapshot the run wrote, which a resume continues from. A `RUNNING`
left with no live process belongs to a run that was killed outright
(SIGKILL, out of memory, a lost node). The next launch in that
directory reports it and replaces it. A launch refuses to start next
to a `RUNNING` whose process is still alive on the same machine. The
details are in the
[`run_status`](../src/dnsjax/run_status.py) module docstring.

### The diagnostic streams

Each `.dat` stream opens with a `#`-commented header row naming its
columns (`t` first) — so `np.loadtxt` reads one directly — and is
appended to across resumes. `stats.dat` carries the
flow's physical diagnostics: the perturbation and total kinetic
energies `E'` and `E`, and the energy input rate `I` against the
dissipation `D`, which satisfy $dE/dt = I - D$ to truncation order —
a closure the test suite pins. The wall-bounded flows add per-wall
shear stresses and bulk velocities under the names natural to the
geometry (`tau'_s,b`/`tau'_s,t` and `Ub'_s`/`Ub'_n` in the channels,
`tau'_z`/`tau'_th` in the pipe, inner/outer pairs in the annulus),
primed on the flows that evolve a perturbation and unprimed on the
four that integrate the total field. The viscoelastic flows report
the solvent dissipation `D_s` in place of `D` and add the polymer
work `W_p`, the elastic energy `E_p`, and the mean conformation
trace `TrC`. A run holding a bulk velocity or a mean
spanwise velocity fixed appends one further column per constrained
direction (`-dPds'` / `-dPdn'` / `-dPdz'`): the mean-mode **forcing**
the corrector applied over that step, positive when accelerating. Two
optional binary streams — a spectral-mode probe stream and a
stochastic-forcing log — are available through the `[probes]` and
`[force]` sections; see
[`src/dnsjax/extensions`](../src/dnsjax/extensions/README.md).

### Reduced-resolution snapshots

The `[lowres]` section writes the state at a coarser resolution on a
cadence of its own, for when full snapshots at that cadence would not
fit on disk:

```bash
.venv/bin/dnsjax \
  --phys.system plane-poiseuille --phys.re 4200 \
  --res.nx 128 --res.ny 97 --res.nz 128 \
  --lowres.it_lowres 100 --lowres.nx 64 --lowres.ny 49 --lowres.nz 64
```

Every `lowres.it_lowres` steps (counted like `outs.it_snapshot`) the
run writes `lowres/lowres_<it>.tar`, the step count zero-padded to ten
digits. The targets take the flow's own `res` names (`lowres.nz`,
`lowres.nr` and `lowres.ntheta` on the pipes and the annulus). Each
defaults to the run's own count, and a set one may not exceed it. Like
`[outs]`, the section is inherited on resume; `--lowres.it_lowres 0`
switches it off ([`configuration.md`](configuration.md)).

Each field is computed at the run's resolution and only then reduced.
The Fourier axes drop their highest modes. The wall-normal axis is
interpolated onto the same grid type at the reduced count, exactly as a
resume onto a new grid does it: spectrally between Chebyshev grids. A
custom `geo.wall_grid` has no grid at another count, so it keeps its
own.

On plane Couette and plane Poiseuille, `lowres.pressure` (on by
default there) adds the static pressure perturbation $p'$, computed
from the full-resolution field. It is the pressure consistent with the
discrete dynamics, in the gauge where its mean profile is zero at the
upper wall, re-pinned there after the reduction. The mean pressure
gradient is the driving, which `stats.dat` records.

The files are ordinary snapshots for reading —
`dnsjax.analysis.read_state` opens one, and `read_pressure` its
pressure — but they are not checkpoints: they carry nothing a solver
needs beyond the fields, and a resume refuses them
([`snapshots.md`](snapshots.md)).
