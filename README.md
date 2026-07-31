# dnsjax

**A GPU-accelerated pseudo-spectral solver for direct numerical simulation
of the 3D incompressible Navier–Stokes equations, written in
[JAX](https://github.com/jax-ml/jax).**

![Python](https://img.shields.io/badge/python-%E2%89%A53.14-blue)
![JAX](https://img.shields.io/badge/backend-JAX-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

`dnsjax` integrates the incompressible Navier–Stokes equations by a
**pseudo-spectral** treatment of the periodic directions (Fourier) combined
with **banded finite differences** in up to one wall-bounded direction, where
the **influence-matrix method** enforces incompressibility together with the
wall boundary conditions. Because it is written in JAX, the same source runs
on **CPUs, GPUs, and TPUs**, on a single device or sharded across many, and in
single- or double-precision. Time advancement defaults to a second-order,
semi-implicit predictor–corrector scheme (an iterative Crank–Nicolson); a
Crank–Nicolson / Adams–Bashforth alternative trades the corrector's stability
margin for a single FFT evaluation per step.

## Highlights

- **Eight flow systems across four geometries** — pipe, Taylor–Couette,
  quasi-Keplerian, Dean, viscoelastic Dean, plane-Poiseuille,
  plane-Couette, and Kolmogorov flow.
- **Runs anywhere JAX runs** — CPU, GPU, or TPU, on one device or many, from
  the same code path.
- **Second-order semi-implicit time stepping** — an iterative Crank–Nicolson
  predictor–corrector by default, stable well past the advective CFL limit;
  an optional Crank–Nicolson / Adams–Bashforth scheme (`cnab2`) costs a
  single FFT evaluation per step in exchange for an advective-CFL step
  restriction. Both are built from one stepper factory.
- **A custom banded-LU GPU kernel** — a Pallas/Triton per-mode wall-normal
  solver with $O(N_y p)$ storage, validated against a dense reference
  solver.
- **Non-modal stability built in** — 3D linear optimal energy growth $G(t)$
  around an arbitrary wall-normal profile, reusing the solver's own linear
  step.
- **A coupled viscoelastic model** — a simplified Phan-Thien–Tanner (sPTT)
  conformation-tensor flow riding the same component-agnostic machinery.
- **Portable data** — snapshots are plain tar + zarr3, written in parallel
  directly from device memory, readable with standard tools and a
  dependency-light NumPy reader; resume is device-count-agnostic.
- **Extensively tested** — 34 standalone test scripts (also runnable
  through a pytest bridge) pin the numerics, the machinery, and the
  multi-device behavior, and the optimal-growth module reproduces
  published values — see
  [Testing and validation](#testing-and-validation).

## Flows and geometries

| Flow | Geometry | Laminar base / driving | Defining controls |
|---|---|---|---|
| **Pipe** | cylindrical | $U_z = 1 - r^2$, pressure-driven | `re`, axial length `lz` |
| **Taylor–Couette** | annular | $U_\theta = A_0 r + B_0/r$, wall rotation | `re1`, `re2`, `eta` |
| **Quasi-Keplerian** | annular | $U_\theta = A_0 r + B_0/r$, Rayleigh-stable co-rotation | `re1`, `r_omega`, `eta` |
| **Dean** | annular | azimuthal body force, total field | `re`, `eta` |
| **Viscoelastic Dean** | annular (sPTT) | azimuthal body force, 9-component total field | `el`, `wi`, `beta`, `epsilon`, `kappa`, `delta` |
| **Plane-Poiseuille** | cartesian | $U = 1 - y^2$, pressure-driven | `re`, `lx`, `lz`, `tilt_degree` |
| **Plane-Couette** | cartesian | $U = y$, wall-driven | `re`, `lx`, `lz`, `tilt_degree` |
| **Kolmogorov** | triply-periodic | $U = \sin(2\pi y / L_y)$, sine body force | `re`, `lx`, `lz` |

A few conventions worth knowing:

- **Reynolds number.** `re` sets the viscosity $\nu = 1/Re$. For the pipe it
  is simultaneously the centerline–radius and the bulk-velocity–diameter
  Reynolds number (the factors of two cancel in the chosen normalization).
- **Driving.** The pressure-driven flows (pipe, plane-Poiseuille) accept
  `phys.driving = "constant_bulk_velocity"` to hold the bulk velocity
  fixed instead of the mean pressure gradient, and every wall-bounded
  flow but the pipe can pin the mean velocity of its undriven
  homogeneous direction to zero (`phys.block_mean_spanwise_velocity`) —
  the spanwise mean in the channels, the axial mean in the annulus.
- **Taylor–Couette rotation.** `re1` and `re2` are the inner and outer
  cylinder Reynolds numbers on a unit gap, with `re1 >= 0` and `re2` free to
  be negative. The sign pattern selects the configuration: inner-driven
  (`re1 > 0, re2 = 0`), outer-driven (`re1 = 0, re2 > 0`), co-rotating (same
  signs), or counter-rotating (`re2 < 0`); `eta = r_1/r_2` is the radius
  ratio. The quasi-Keplerian flow is the same annulus parameterized by
  `re1`, the rotation number `r_omega` on the quasi-Keplerian half-line
  $R_\Omega < -1$, and `eta`, with the outer Reynolds number `re2`
  derived from them.
- **Viscoelastic controls.** `el` is the elasticity number and sets
  $Re = Wi/El$; `wi` is the Weissenberg number; `beta` the solvent-to-total
  viscosity ratio; `epsilon` the sPTT extensibility; `kappa` an artificial
  stress diffusivity; `delta` the inner radius (the gap is fixed at 2).
- **Viscoelastic memory.** With 9 state components, the viscoelastic
  right-hand side inverse-transforms a 36-field batch every step, and that
  batch dominates the step's peak memory. `solver.rhs_transform_chunks`
  (item 4 in [Additional features](#additional-features)) splits it to
  cut the batch's transform-stage peak roughly $k$-fold at identical
  results; it defaults to `1` (the fused batch) because chunking costs
  throughput — more FFT dispatches and reshard rounds — that is only
  worth paying when a run is memory-bound.
- **Grid axes.** Each flow's parameters use the names natural to its
  geometry: the cylindrical and annular flows expose `lz` (axial
  length), `nz` (axial modes), `nr` (radial points), and `ntheta`
  (azimuthal modes); their azimuthal extent is not a free length — it
  is the full circle, or the $2\pi/m_0$ wedge under `--geo.m0`. The
  wall-normal extent is fixed by the geometry (the channel spans
  $[-1, 1]$, the pipe radius is 1, the annulus $[r_1, r_2]$, and the
  periodic box uses $L_y = 4$).

## Installation

```bash
git clone https://github.com/gokhanyalniz/dnsjax.git
cd dnsjax
uv sync
```

The only prerequisite is [`uv`](https://docs.astral.sh/uv/): `uv sync`
provisions the pinned Python (3.14) by itself and installs the dependencies.
An MPI runtime (`mpirun`) is used to *launch* simulation runs — even
single-process ones — but is not needed for the installation or by the
post-processing API. The default install pulls a CPU build of JAX. To run on
**CUDA GPUs**, replace `jax` with the CUDA-13 build:

```bash
uv add "jax[cuda13]"    # rewrites the jax requirement, re-locks, and re-syncs
```

(equivalently, change the `jax>=…` line in `pyproject.toml` to
`jax[cuda13]>=…` and run `uv sync`). The CUDA wheels are Linux x86-64 only.

## Running a simulation

The example below runs a **100-diameter pipe at Re = 2300**, started from a
compact localized-roll perturbation, on a single CPU device. Every
problem-defining parameter — the physics, the geometry, the resolution, and
the time integrator — is written out explicitly, so switching to another flow
is a matter of editing values rather than learning the defaults.

A `dnsjax` run is always launched through `mpirun` (even for one process),
invoking the environment's `dnsjax` console script directly — `uv run` does
not compose with `mpirun`, and `python -m dnsjax` is the equivalent module
form. Output files (`stats.dat`, snapshots, …) are written to the current
directory, so launch from a scratch directory:

```bash
mpirun -np 1 .venv/bin/dnsjax \
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

Reading the flags:

- `--phys.system pipe --phys.re 2300` — the flow and its Reynolds number.
- `--geo.lz 200` — the axial length is 100 pipe diameters ($D = 2$). The
  azimuthal extent is not settable: it is the full circle, or the
  $2\pi/m_0$ wedge when the `--geo.m0` symmetry restriction is used.
- `--geo.grid_type half-cgl` — the radial grid; `half-cgl` is the default
  for a pipe under `iterative-cn`, while `cnab2` uses `rigged-cgl` instead
  (both are halves of a Chebyshev grid that avoid the axis — see
  [Grids](#grids)).
- `--res.nz 512 --res.nr 48 --res.ntheta 96` — axial, radial, and azimuthal
  resolution, with eighth-order (the default) finite differences in the
  radial direction.
- `--step.scheme iterative-cn --step.dt 0.01` — the default
  predictor–corrector integrator at a wall-bounded-safe step.
- `--init.localized_rolls …` — a compact, deterministic finite-amplitude
  perturbation (peak amplitude 0.2) that seeds transition.
- `--stop.max_sim_time 500` — stop at $t = 500$ advective units (transition
  develops over $O(100)$ units; the run also stops early if the flow
  relaminarizes).
- `--dist.platform cpu` — a single CPU device.

One default worth knowing: the pipe integrates in a frame translating at the
laminar bulk velocity $1/2$, and its snapshots are stored in that frame;
pass `--phys.u_grid 0` for the lab frame (see
[Temporal discretization](#temporal-discretization)).

This configuration fits comfortably in laptop memory — the
[Memory footprint](#memory-footprint) section shows how to estimate any
configuration. **Switching flows** is a one-line change:
`--phys.system taylor-couette --phys.re1 … --phys.re2 … --geo.eta …`, or
`--phys.system kolmogorov --geo.lx … --geo.lz …`, and so on per the table
above.

The same run can be expressed as a `parameters.toml` in the working
directory (shipped as the repository default):

```toml
[phys]
system = "pipe"
re = 2300            # bulk/diameter Reynolds number (= centerline/radius; D = 2)

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

What to expect while it runs: the code's git revision, the final working
parameters, and the physical-space resolution are printed at startup, the
first step takes noticeably longer than the rest (JIT compilation), and a
timing summary is printed at the end. Statistics stream to `stats.dat`
(with `steps.dat` and `corrector.dat` for the CFL and corrector
diagnostics), and snapshots appear as `state00000.tar` (the initial
condition), `state00001.tar`, and so on. Runs end gracefully — at
`max_sim_time`, at an ISO 8601 `stop.max_wall_time` budget (writing a
final snapshot first), on relaminarization, or on SIGTERM/SIGINT (flushing
the diagnostic buffers) — so interrupted runs stay consistent with their
outputs; a NaN or inf in any diagnostic instead aborts the run at once
with a line naming the quantity, rather than spending the budget on a
broken state.

## Parameter layering

Configuration is applied in layers, lowest priority first:

**Per-flow defaults → parameters embedded in a resumed snapshot →
`parameters.toml` → command-line flags.**

Only explicitly set fields override a lower layer, and validation runs once
after the final layer. Every layer is parsed against the **selected flow's
parameter surface**: only that flow's parameters exist (an irrelevant key is
a hard error naming the flow), fields go by their geometry-natural public
names (a pipe has `--geo.lz`/`--res.nz`/`--res.nr`/`--res.ntheta` where a
plane channel has `--geo.lx`/`--res.nx`/`--res.ny`/`--res.nz`), and per-flow
defaults (the pipe's moving frame `u_grid = 0.5`, its scheme-dependent
`grid_type`, the viscoelastic rheology values) are materialized before
printing or recording. The parameters that must be known before JAX
initializes — `dist.np0`, `dist.np1`, `dist.platform`, and
`res.double_precision` — are never inherited from a snapshot, nor are the
resume-decision fields `init.snapshot` and `init.force_resume` (recorded
for lineage only), and the entire `solver` section is execution-only.

`uv run dnsjax --help` shows the global parameters and the flow list,
`--help <system>` one flow's full surface with per-field descriptions, and
`--sample-toml <system>` an annotated `parameters.toml` template with every
default commented out (all exit at the parser — no `mpirun` needed). The
authoritative field-by-field documentation lives in
[`src/dnsjax/parameters.py`](src/dnsjax/parameters.py) and the per-flow
specs under `src/dnsjax/flows/*/specs/`.

## Memory footprint

Every contribution below scales linearly with the point count
$n_x n_y n_z$ — nothing grows faster under the default backends — and the
total divides by $n_{p0} \cdot n_{p1}$ across devices. At the default double
precision one real number is 8 bytes, so a *field* of $n_x n_y n_z$ reals
occupies $n_x n_y n_z / 2^{27}$ GiB; that is the unit used below. Single
precision (`res.double_precision = false`) halves everything and roughly
doubles the throughput of the bandwidth-bound FFT stages on GPUs
(considerably more on consumer GPUs, which throttle double-precision
arithmetic), at reduced accuracy. Assuming the default 3/2 dealiasing and
the default backends:

- **Spectral state** — exactly $n_c$ fields, with $n_c = 3$ velocity
  components (9 for viscoelastic Dean): one component is
  $(n_x/2) \cdot n_y \cdot (n_z - 1)$ complex numbers ($n_y - 1$ in place
  of $n_y$ for the periodic box), i.e.
  $\approx n_x n_y n_z$ reals. The time stepper holds about three further
  state-sized arrays within a step, and `cnab2` carries one across steps
  (for the wall-bounded systems its allocated peak still matches the
  default scheme's, whose corrector branch XLA keeps reserved); Dean and
  viscoelastic Dean keep one extra state-sized laminar reference.
- **Nonlinear term, every step** — the rotational form inverse-transforms a
  6-field batch (velocity + vorticity) to the oversampled grid, multiplies
  pointwise, and forward-transforms the 3 product fields. Counting the held
  fields, the products, and the one to two batch-sized intermediates inside
  the transforms, the working set is $W \approx 15\text{–}21$ oversampled
  fields; each oversampled field is $(3/2)^2 = 2.25$ fields for
  wall-bounded systems (the wall-normal direction is never oversampled) and
  $(3/2)^3 = 3.375$ fields for triply-periodic ones. How much of this
  coexists is decided by XLA's buffer reuse, so treat the upper end as the
  sizing estimate. The viscoelastic right-hand side instead transforms a
  36-field batch with 9 outputs, and `solver.rhs_transform_chunks = k` —
  the knob applies to every flow's batch, but bites here — cuts its
  transform-stage share $k$-fold at identical results.
- **Wall-normal operators** — the Pallas backend stores no-pivot banded LU
  factors: $(2p + 1) \cdot n_y$ reals per matrix per Fourier mode, with the
  half-bandwidth $p$ equal to `fd_order`, over the $(n_z - 1)(n_x/2)$ mode
  plane — that is $m (2p + 1)/2$ fields for $m$ banded matrices, the one
  term that grows with `fd_order`. Here $m = 2$ for
  plane-Couette/Poiseuille, $4$ for pipe, Taylor–Couette,
  quasi-Keplerian, and Dean, and $10$ for viscoelastic Dean, plus
  $v = 3\text{–}6$ field-sized boundary-response vectors ($v/2$
  fields). Switching to
  `solver.backend = "dense"` replaces $(2p + 1)$ by $n_y$ per matrix — the
  one super-linear option, and the reason Pallas is the wall-bounded
  default. Triply-periodic systems store no matrices at all (their implicit
  solve is diagonal in spectral space), only four real coefficient arrays
  — wavenumber and inverse-Laplacian factors, $\approx 2$ fields.

Summing these, the leading-order total per device is

```math
\text{wall-bounded:} \qquad
  \Bigl[\, 4 n_c + \tfrac{9}{4} W +
    \tfrac{1}{2} \bigl( m (2p + 1) + v \bigr) \Bigr]
  \, \frac{n_x n_y n_z}{2^{27} \, n_{p0} n_{p1}} \ \text{GiB},
```

```math
\text{triply-periodic:} \qquad
  \Bigl[\, 4 n_c + \tfrac{27}{8} W + 2 \Bigr]
  \, \frac{n_x n_y n_z}{2^{27} \, n_{p0} n_{p1}} \ \text{GiB},
```

with $W \approx 15\text{–}21$ as above (for viscoelastic Dean,
$W \approx 45 + 72/k$ with $k$ = `rhs_transform_chunks`) and
$(n_c, m, v) = (3, 2, 4)$ for the plane flows, $(3, 4, 3)$ for the pipe,
$(3, 4, 6)$ for Taylor–Couette, quasi-Keplerian, and Dean, and
$(9, 10, 6)$ for viscoelastic Dean. The sum is an upper estimate —
XLA's buffer reuse typically realizes less — and halves at single
precision. Off the stepping path, a snapshot write reshards the state
onto an I/O layout before moving each device's bytes directly to disk
(staging through host memory only when GPUDirect Storage is
unavailable) — a transient second state-sized copy on multi-device
runs, nothing extra on a single device — and the on-device diagnostic
buffers are resolution-independent.

## Array layout by geometry

The solver keeps one internal axis order for every flow — physical
`[axis0, axis1, axis2]` and spectral `[axis0, axis1, axis2]` — and the
physical meaning of each axis is set by the geometry (a row per
geometry, not per flow). The leading axis is device-local; the two
sharded axes are split by `np0` and `np1` (elaborated under
[Parallelization](#parallelization)). Role abbreviations: **sw**
streamwise, **wn** wall-normal, **sh** shearwise, **sp** spanwise.

| Geometry | Velocity components `(0, 1, 2)` | Physical `[0, 1, 2]` | Spectral `[0, 1, 2]` | `np0` splits | `np1` splits |
|---|---|---|---|---|---|
| Triply-periodic (Kolmogorov) | $(u_x, u_y, u_z)$ = (sw, sh, sp) | $[y, z, x]$ | $[k_y, k_z, k_x]$ | $y$ / $k_z$ | $z$ / $k_x$ |
| Cartesian (plane-Poiseuille/Couette) | $(u_x, u_y, u_z)$ = (sw, wn, sp) | $[y, z, x]$ | $[y, k_z, k_x]$ | $y$ / $k_z$ | $z$ / $k_x$ |
| Cylindrical (pipe) | $(u_z, u_r, u_\theta)$ = (sw, wn, sp) | $[r, \theta, z]$ | $[r, k_\theta, k_z]$ | $r$ / $k_\theta$ | $\theta$ / $k_z$ |
| Annular (Taylor–Couette, quasi-Keplerian, Dean, viscoelastic Dean) | $(u_z, u_r, u_\theta)$ = (**sp**, wn, **sw**) | $[r, \theta, z]$ | $[r, k_\theta, k_z]$ | $r$ / $k_\theta$ | $\theta$ / $k_z$ |

Each `np0` / `np1` cell reads *physical axis* / *spectral axis*.
Velocity components are stored in `(streamwise, wall-normal, spanwise)`
order for every geometry **except the annulus**, which reuses the
pipe's axial-first $(u_z, u_r, u_\theta)$ order so the solver's shared,
right-handed curl / cross / finite-difference operators apply
unchanged. Because the annular main flow is azimuthal, its streamwise
velocity is component 2 ($u_\theta$) and its spanwise velocity is
component 0 ($u_z$) — the sole departure from the component-order
convention.

## Parallelization

The device grid is $(n_{p0}, n_{p1})$, and the two axes distribute the data
differently:

- **`np0`** splits the wall-normal axis ($y$ / $r$) in physical space and the
  spanwise / azimuthal wavenumber axis ($k_z$ / $m$) in spectral space. The
  split is padding-free when `np0` divides both the wall-normal point
  count (`ny`, or `nr`) and the stored mode count ($n_z - 1$, or
  $n_\theta - 1$); otherwise the layer zero-pads to the next multiple
  and strips the padding around the reshard (the stored mode count is
  odd, so a one-mode pad is the norm — and harmless).
- **`np1`** splits the spanwise / azimuthal axis ($z$ / $\theta$) in
  physical space and the streamwise / axial wavenumber axis ($k_x$) in
  spectral space. The spectral side is auto-padded the same way
  (padding-free when `np1` divides the streamwise / axial mode count,
  $n_x/2$ or $n_z/2$); on the physical side the oversampled size
  ($3/2 \times$ the base resolution of that axis at the default
  oversampling) is rounded up to the next FFT-friendly multiple of
  `np1` when needed (see
  [Spatial discretization](#spatial-discretization)), which amounts to a
  sliver of extra oversampling.
- Independently of the device grid, the **Pallas banded solver** tiles each
  device's $(k_z, k_x)$ mode plane in blocks of
  (`solver.pallas_block_m0`, `solver.pallas_block_m1`) $= (2, 32)$ and pads
  up to whole tiles. The padded modes cost memory and solve work in
  proportion to the round-up, so per-device mode counts
  $(n_z - 1)/n_{p0}$ and $(n_x/2)/n_{p1}$ near multiples of the block sizes
  are optimal; both knobs are adjustable when the mode plane is small.

No divisibility choice is rejected, and none of the padding — for the
device grid or for FFT-friendly sizes — is silent: every adjustment is
reported by a one-line startup diagnostic, so its (usually marginal) cost
stays visible.

Crucially, **every device holds the full wall-normal extent in spectral
space**, so the per-mode banded solves need no communication. The forward and
inverse FFTs move data between layouts with two reshards implemented as a
`shard_map` with explicit `reshard` calls; with `np0 = 1` the decomposition
collapses to the one-dimensional $k_x$ / $z$ split. `jax.device_count()`
must equal $n_{p0} \cdot n_{p1}$.

The pipe example above on a $2 \times 2$ device grid (`nr = 48` and
$n_z/2 = 256$ split evenly; `ntheta = 96` gives 144 padded azimuthal
points, divisible by 2; the only round-up is the harmless one-mode pad
of the 95 stored azimuthal modes):

```bash
# CPU: one device per process
mpirun -np 4 .venv/bin/dnsjax \
  --dist.np0 2 --dist.np1 2 --dist.platform cpu \
  --phys.system pipe --phys.re 2300 --geo.lz 200 \
  --res.nz 512 --res.nr 48 --res.ntheta 96 \
  --init.localized_rolls True --stop.max_sim_time 500
```

```bash
# GPU: a single process addressing all four GPUs on the node
mpirun -np 1 .venv/bin/dnsjax \
  --dist.np0 2 --dist.np1 2 --dist.platform cuda \
  --phys.system pipe --phys.re 2300 --geo.lz 200 \
  --res.nz 512 --res.nr 48 --res.ntheta 96 \
  --init.localized_rolls True --stop.max_sim_time 500
```

Because `np0 * np1` counts *devices* rather than processes, a single-node
multi-GPU run is most reliably launched as one process that addresses every
visible GPU; multi-node runs use one process per node spanning that node's
GPUs. The `Distribution` docstring in `parameters.py` covers the SLURM
launch details.

## Snapshots and external data access

A snapshot is a **single uncompressed tar archive** (format version 6)
wrapping a **zarr3** store, a JSON metadata member (parameters, grid,
lineage, and the writing code's git revision), and one contiguous chunk
per state component (three velocity components, or nine for the
viscoelastic flow). Each chunk is stored **in the solver's native
spectral layout** at true (unpadded) mode counts — saving, loading, and
reading never transpose — and in **physical components** for every
geometry: the cylindrical and annular families convert from the
solver's decoupled $u_\pm$/spin working basis at the write/read
boundary. The embedded parameters are the flow-relevant,
resolved values under their public names — the same representation the
startup printout and `--sample-toml` use; snapshots written before
format version 6 embed a different layout, basis, or representation and
are rejected rather than translated. A write first reshards the state,
inside `jit`, onto the file's own layout — one contiguous span per
device — and each device then writes its disjoint byte ranges into the
one file in parallel: directly between GPU memory and disk when
GPUDirect Storage is available, through the host otherwise, with a
concurrent mode for POSIX/parallel filesystems and a rank-ordered
serial mode for filesystems where concurrent writes are unsafe. The
bytes land in `<name>.tar.partial` and are renamed into place only once
complete, so a killed job leaves the previous snapshot intact and never
a truncated archive that could pass for a valid one; on read, the chunk
layout is checked against the metadata, and a damaged archive raises an
error naming the file and the cause.
The stored field is the spectral **perturbation** $\mathbf{u}'$ for the
base-flow systems (the laminar state is a zero array) and the **total**
field for Dean and viscoelastic Dean. The archive is readable with ordinary
tools — `tar xf` yields a valid zarr3 store, and in the worst case each
chunk is raw little-endian complex data for `numpy.fromfile`. Resume is
agnostic to the device count (precision must match — a mismatch
rejects), and re-grids a changed wall-normal grid on load — spectrally
when both grids are CGL-family, by a local order-`fd_order` stencil for
tanh or custom grids.

For post-processing, `dnsjax.analysis.snapshot_export.read_state` reads a
snapshot into NumPy arrays **without importing JAX or the solver runtime**,
pulling only the requested data off disk:

```python
from dnsjax.analysis.snapshot_export import read_state

data = read_state("state00001.tar")   # NumPy only — no JAX, no solver
u_z, u_r, u_theta = data.physical     # pipe: real fields, native (r, θ, z)
r, theta, z = data.physical_coords    # matching coordinate arrays
re = data.params.phys.re              # embedded parameters

# Cartesian systems return (u_x, u_y, u_z) in the native (y, z, x) layout:
u_x, u_y, u_z = read_state("state00002.tar").physical

# Select components, read just two wall-normal slabs off disk, and also
# return the spectral coefficients:
data = read_state(
    "state00001.tar",
    components=(0,),
    wall_normal_points=(0.2, 0.8),
    return_spectral=True,
)
```

The companion `dnsjax.analysis.snapshot_ops` module provides `divergence`,
`curl`, `gradient`, and `integrate` that reproduce the solver's *discrete*
operators node-for-node, and `scripts/snapshot_import.py` covers the
reverse direction: packing a velocity field produced elsewhere (by
another simulator, say) into a valid snapshot — velocity flows only,
the nine-component viscoelastic state being readable but not importable.

The importer is a library (not a CLI) and **assumes the field is already
in dnsjax's native layout**: components leading, axes $(y, z, x)$ for the
Cartesian and triply-periodic systems and $(r, \theta, z)$ for the
cylindrical and annular flows (pipe, Taylor-Couette, quasi-Keplerian,
Dean) — whose components are $(u_z, u_r, u_\theta)$ — so any axis
permutation and component reordering from the source code's conventions
is the caller's first step.
Two conventions to keep in mind. The resolutions are the solver's
nominal (physical) mode counts *without* the 3/2 dealiasing expansion —
never include dealiasing zero-padding in the field or the resolution
parameters. And every wall-bounded flow needs its wall-normal/radial
grid points, **ascending** in dnsjax's convention: bottom wall $-1$ to
top wall $+1$ (Cartesian), near-axis to the outer wall on $(0, 1]$
(pipe), inner to outer radius (Taylor-Couette); the triply-periodic
systems take no grid. Parameters go by the flow's public names, exactly
as on the CLI:

```python
import sys

import numpy as np

sys.path.insert(0, "scripts")   # snapshot_import is a library, not a CLI
from snapshot_import import convert_field_to_snapshot

# Plane-Couette: perturbation u' with components (u_x, u_y, u_z) over
# native axes (y, z, x) — shape (3, ny, nz, nx) — already in dnsjax's
# layout, sampled on the ascending wall-normal grid ys of length ny.
u = np.load("external_field.npy")           # (3, 65, 128, 128)
ys = -np.cos(np.linspace(0.0, np.pi, 65))   # CGL: -1 (bottom) → +1 (top)
convert_field_to_snapshot(
    u, "ic_plane_couette.tar",
    system="plane-couette", nx=128, ny=65, nz=128,
    lx=4.0, lz=4.0, wall_normal_grid=ys, re=400.0,
    space="physical",
)

# Pipe: (u_z, u_r, u_θ) over (r, θ, z), shape (3, nr, ntheta, nz); lz
# is the axial period (the sole free length — the azimuthal extent is
# the wedge 2π/m0), and rs ascends over the radii on (0, 1].
convert_field_to_snapshot(
    u_pipe, "ic_pipe.tar",
    system="pipe", nz=96, nr=49, ntheta=128,
    lz=6.0, m0=1, wall_normal_grid=rs, re=3000.0,
    space="physical",
)

# Taylor-Couette: same layout and resolution names as the pipe, driven
# by re1/re2/eta; rs_tc ascends from r_in = η/(1−η) to r_out = 1/(1−η).
convert_field_to_snapshot(
    u_tc, "ic_taylor_couette.tar",
    system="taylor-couette", nz=64, nr=49, ntheta=128,
    lz=4.0, wall_normal_grid=rs_tc,
    re1=400.0, re2=-200.0, eta=0.875,
    space="physical",
)
```

`space="spectral"` accepts already-transformed input in the same axis
order, with one restriction on where the half spectrum may sit: only the
**last** axis (the streamwise `nx` / axial `nz` slot) is the real-FFT
axis, holding the `nx//2` non-negative modes (Nyquist optional, dropped);
the other Fourier axes must carry full two-sided spectra, and
`input_norm` names the source's FFT normalization. A source that
`rfft`-ed a different axis must be permuted so its half axis lands last.
The result is an ordinary snapshot: start a run from it with
`--init.snapshot ic_plane_couette.tar` (a wall-normal grid differing from
the run's is re-gridded at load).

## Governing equations and numerics

### Equations

The solver advances the non-dimensional incompressible Navier–Stokes
equations,

```math
\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u}
  = -\nabla p + \frac{1}{Re}\nabla^2 \mathbf{u},
\qquad \nabla \cdot \mathbf{u} = 0 .
```

Most systems evolve the **perturbation** $\mathbf{u}'$ about an analytical
laminar profile $\mathbf{U}$, using the **rotational form** of the nonlinear
term,

```math
\mathrm{NL} = \mathbf{u}' \times \boldsymbol{\omega}' +
  \mathbf{u}' \times (\nabla \times \mathbf{U}) +
  \mathbf{U} \times \boldsymbol{\omega}',
\qquad \boldsymbol{\omega}' = \nabla \times \mathbf{u}',
```

in which the base-flow self-interaction is a pure gradient absorbed by the
pressure. The force-driven Dean and viscoelastic-Dean systems instead
integrate the **total** field with a mean-mode azimuthal body force.

The viscoelastic flow couples a symmetric **conformation tensor** $\mathbf{c}$
through a simplified Phan-Thien–Tanner constitutive law,

```math
\partial_t \mathbf{c} + (\mathbf{u}\cdot\nabla)\mathbf{c} -
  (\nabla\mathbf{u})^\top \mathbf{c} - \mathbf{c}\,(\nabla\mathbf{u})
  = \kappa \nabla^2 \mathbf{c} -
    \frac{1 - 3\epsilon + \epsilon\,\mathrm{tr}\,\mathbf{c}}{Wi}
    (\mathbf{c} - \mathbf{I}),
```

with the polymer stress feeding momentum as
$\tfrac{1-\beta}{Re\,Wi}\nabla\cdot\mathbf{c}$ and solvent viscosity
$\nu = \beta/Re$. This grows the state to nine components on the same solver
machinery.

### Spatial discretization

Periodic directions are treated **pseudo-spectrally** with Fourier
transforms; the single wall-bounded direction uses **banded finite
differences** with Fornberg weights of order `fd_order` (half-bandwidth $p$
equal to `fd_order`). The quadratic nonlinearity is dealiased with the
**3/2 rule** — physical fields are evaluated on a
$\tfrac{3}{2}$-oversampled grid and the product is truncated back — and the
Nyquist mode is dropped on every stored spectral axis (FFTs use
`norm="forward"`). The dealiasing pad carries no parity constraint — the
omitted Nyquist mode re-enters as a zero in its exact wrap-order slot for
even and odd pads alike — so any spanwise / azimuthal `nz` is accepted
(likewise `ny` for the triply-periodic box). The oversampled sizes are
then rounded up, with a startup note, to **7-smooth** lengths (no prime
factor beyond 7, so every transform takes the fast FFT radix kernels
whatever the base resolution) that also divide evenly across the device
grid; the streamwise real-FFT axis, never sharded, gets the smoothness
rounding only. The extra slots carry only zero modes, so the rounding is
physically neutral.

### Temporal discretization

Two second-order, semi-implicit schemes share the same predictor and
influence-matrix pressure solve:

- **`iterative-cn`** (default) — a semi-implicit Euler predictor followed
  by an iterative Crank–Nicolson corrector that makes the nonlinear term
  implicit through its fixed point. This costs $2 + n_c \approx 3$ FFT
  evaluations per step ($n_c$ the corrector's iteration count) and is
  stable well past the advective CFL limit.
- **`cnab2`** — Crank–Nicolson viscous term with an explicit second-order
  Adams–Bashforth nonlinear term, costing a **single** FFT evaluation per
  step (roughly a threefold saving on CFL-limited turbulent runs), at the
  price of an advective-CFL step restriction. The wall-bounded systems
  keep the wall-stiff coupling implicit through an FFT-free corrector; a
  step whose corrector fails to contract falls back to one full
  `iterative-cn` step.

The `implicitness` knob $c$ is the Crank–Nicolson weight ($c = 0.5$ is the
trapezoidal rule), with `corrector_tolerance` and `max_corrector_iterations`
governing the fixed point. An **opt-in split corrector**
(`split_corrector`, off by default) iterates the wall-stiff linear coupling
FFT-free between full right-hand-side refreshes; it only helps when the step
is pushed near the corrector iteration cap and is otherwise slower, hence the
default. A related `implicit_mean_coupling` (on by default) folds the
instantaneous mean-flow coupling into the implicit term.

Both schemes can run at a fixed `step.dt` or under **adaptive CFL time
stepping** (`step.adaptive`): the main loop re-reads the measured total
CFL every `cfl_cadence` steps and rescales the step toward the
`cfl_target` setpoint, bounded by `dt_min`/`dt_max`, per-change growth
and shrink limiters, and a relative deadband that suppresses churn from
CFL noise. An accepted change rebuilds the $\Delta t$-dependent
implicit operators on the device — a few implicit solves, no
recompilation — and under `cnab2` the following Adams–Bashforth step
is ratio-weighted (variable-step AB2), so both schemes remain
second-order accurate.
Snapshots embed the live `dt`, so a resume continues at the adapted
step.

A moving frame of reference (`u_grid`) translates the domain along the
streamwise / axial direction and is integrated implicitly by both schemes —
convenient for following traveling structures. By default the frame moves at
the laminar bulk velocity: $1/2$ for the pipe, $2/3$ for plane-Poiseuille,
and zero for the others (Dean's driving is azimuthal, so its axial bulk
vanishes). The pipe and plane-Poiseuille flows therefore integrate, and
store snapshots, in the moving frame unless `u_grid` is set to `0` (see also
item 9 in [Additional features](#additional-features)).

### Grids

The default wall-normal grid is a **Chebyshev–Gauss–Lobatto (CGL)** grid for
the cartesian and annular geometries. The pipe's radial grid is half of a
CGL grid, in one of two variants: the **rigged-CGL** grid is the positive
half of a CGL grid with an *odd* number of points — whose middle node falls
exactly on the axis and is dropped — placing its innermost node about one
grid spacing from the axis, while the **half-CGL** grid is the positive half
of a CGL grid with an *even* number of points (no node ever sits on the
axis), placing it about half as far. `cnab2` defaults to the rigged grid:
the admissible step of its explicit azimuthal advection grows with the
innermost node's radius. `iterative-cn` integrates the near-axis
coupling implicitly and defaults to the finer-resolving half-CGL grid.
Optional tanh stretching (`grid_type = "tanh"`, or `"half-tanh"` for
the pipe's one-sided variant, with the `grid_stretch` factor) and fully
**custom grids** (via a `geo.wall_grid` file) are supported; quadrature is
spectral Clenshaw–Curtis on CGL grids (a weighted parity variant on the
pipe's half grids) and an order-`fd_order` composite rule otherwise.

### The influence-matrix method

Enforcing incompressibility together with the wall boundary conditions is the
central difficulty of a wall-bounded spectral discretization: the wall-normal
momentum equation supplies only the *interior* pressure Poisson problem,
while the correct wall pressure boundary condition is fixed *indirectly* by
requiring $\nabla \cdot \mathbf{u} = 0$ at the walls. The **Kleiser–Schumann
influence-matrix method** resolves this by precomputing a small set of
homogeneous responses once, so that each time step recovers the boundary
condition with a tiny per-mode solve — $1 \times 1$ for the pipe's single
wall, $2 \times 2$ for the two-walled cartesian and annular geometries —
after which the velocity is corrected by linearity. The precomputation
happens once, and the per-step boundary work stays a handful of small
per-mode operations, which is what keeps the wall-bounded solve inexpensive
at scale.

The influence-matrix solve enforces the wall conditions exactly, but the
interior divergence of a stepped state retains a truncation-level
residual. An opt-in reformulation (`res.consistent_imm`) eliminates it:
advance the wall-normal velocity and vorticity, reconstruct the
tangential components, and no discrete pressure appears — the stepped
state's divergence sits at round-off at any resolution, on the same
banded operators, with fewer solves and less operator storage. The trade
is a truncation-level tangential-momentum residual that nothing feeds
back; the discrete-divergence and energy-budget tests pin both
formulations.

## Testing and validation

The test suite is 34 standalone scripts under `tests/`, run directly
(`uv run python tests/test_cartesian.py`) or through the optional pytest
bridge — `uv run pytest` shells each script out as a subprocess, with
`mpi`/`slow` markers and the scripts staying the source of truth — and
several of them launch real `mpirun` multi-device runs. Among the
guarantees they pin:

- **Solvers and operators** — the Pallas banded kernel against a dense
  reference solver, per-geometry operators and matvecs against NumPy
  constructions, and CUDA-lowering guards that catch Triton compilation
  regressions on CPU-only machines.
- **The physics** — laminar states step at machine precision, random
  initial conditions integrate through the full nonlinear path for every
  distinct stepping machinery (six of the eight flows; the other two
  bind machinery those six cover), localized spots integrate for every
  wall-bounded spot builder, and second-order temporal convergence is
  pinned — absolute on the periodic box, scheme-against-scheme for the
  wall-bounded systems, whose absolute order the influence-matrix
  splitting sets.
- **The machinery** — snapshot round-trips readable by standard tools,
  device-count-agnostic resume with lineage checks, and the JAX-free
  import guarantee of the analysis API.
- **Physical validation** — the transient-growth module reproduces published
  optimal-growth values for all five of its flows to about 2% or better.

`scripts/` adds benchmark and diagnostic tools: `solver_benchmark.py`
(Pallas-vs-dense validation and benchmark, including multi-GPU),
`pallas_solve_profile.py` (where the banded solve's time goes),
`pallas_tiling_diagnostic.py` (a GPU miscompile-isolation harness),
`pivot_stability_survey.py` (the evidence behind the no-pivot LU
stability tolerances), `gds_probe.py` (whether the GPUDirect Storage
snapshot path is engaged, and what starves it), and
`corrector_invariance_probe.py` (where the corrector's loop-invariant
work goes and whether hoisting it would pay).

## References

The numerics follow these references:

- A. P. Willis, *The Openpipeflow Navier–Stokes solver*, SoftwareX **6**,
  124–127 (2017). Predictor–corrector time stepping and the decoupled
  $u_\pm$ pipe/annular formulation.
- L. Kleiser and U. Schumann, *Treatment of incompressibility and boundary
  conditions in 3-D numerical spectral simulations of plane channel flows*,
  in *Proc. 3rd GAMM Conference on Numerical Methods in Fluid Mechanics*,
  165–173, Vieweg (1980). The influence-matrix method.
- B. Fornberg, *Calculation of weights in finite difference formulas*, SIAM
  Review **40**(3), 685–691 (1998). Finite-difference weights on non-uniform
  grids.
- L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM (2000). Clenshaw–Curtis
  quadrature and spectral differentiation.
- J. A. C. Weideman and S. C. Reddy, *A MATLAB differentiation matrix suite*,
  ACM Trans. Math. Softw. **26**(4), 465–519 (2000). Differentiation and
  interpolation matrices.
- N. Phan-Thien and R. I. Tanner, *A new constitutive equation derived from
  network theory*, J. Non-Newtonian Fluid Mech. **2**(4), 353–365 (1977). The
  (simplified) Phan-Thien–Tanner viscoelastic model.

## License and citation

Released under the [MIT License](LICENSE), © 2025 Gökhan Yalnız. If `dnsjax`
supports your work, a citation of this repository
(<https://github.com/gokhanyalniz/dnsjax>) is appreciated.

## Additional features

A closer look at what is in the box, beyond the core solver:

1. **Non-modal optimal-growth analysis.** `dnsjax.analysis.transient_growth`
   computes 3D linear optimal energy growth $G(t)$ around an arbitrary
   wall-normal total profile for the pipe, Taylor–Couette, quasi-Keplerian,
   plane-Poiseuille, and plane-Couette flows, reusing the solver's own
   linear step for each Fourier mode. It runs on a single device
   (GPU-capable) and reproduces published optimal-growth values for all
   five flows to about 2% or better.

   ```bash
   python -m dnsjax.analysis.transient_growth --help
   ```

2. **A custom banded-LU GPU kernel with a dense reference solver.** The
   per-mode wall-normal solves run through a custom Pallas/Triton banded-LU
   sweep that stores $O(N_y p)$ factors instead of the dense $O(N_y^2)$, and
   a dense reference solver validates it numerically. Kernels are checked
   both in Pallas interpret mode and by lowering to CUDA on CPU-only
   machines.

3. **Multi-device sharding on CPU/GPU/TPU.** A two-axis $(n_{p0}, n_{p1})$
   device mesh with an in-FFT reshard pipeline distributes the work while
   keeping the wall-normal solves communication-free — see
   [Parallelization](#parallelization).

4. **A memory–throughput dial for the nonlinear term.**
   `solver.rhs_transform_chunks = k` splits the batched inverse transform of
   the pseudo-spectral right-hand side into $k$ balanced groups, cutting its
   transform-stage working set roughly $k$-fold at identical results — see
   [Memory footprint](#memory-footprint). The default `1` keeps the single
   fused batch, which is throughput-optimal (one FFT dispatch and one
   reshard round per pipeline stage); raise it to fit a memory-bound run,
   most effectively for viscoelastic Dean, whose 36-field batch dominates
   the step's peak.

5. **Standard-tools-readable snapshots and a JAX-free reader.** The tar +
   zarr3 format — written in parallel, directly from GPU memory when
   GPUDirect Storage is available — and the NumPy-only `read_state` cleanly
   separate the runtime from the analysis API — see
   [Snapshots and external data access](#snapshots-and-external-data-access).

6. **Robust resume.** Snapshots resume across any device count
   (precision must match), re-grid a changed wall-normal grid on load,
   and track lineage — including the recording code's git revision,
   echoed at startup when resuming — distinguishing a genuine
   continuation from a new trajectory when the physics or geometry
   changes.

7. **Laminarization auto-stop.** A run terminates automatically once the
   perturbation energy drops below `stop.laminarization_threshold`, so
   relaminarization events are captured without babysitting — natural for
   lifetime and edge-of-chaos studies.

8. **Initial-condition generators.** Divergence-free random fields (the
   default start mode) and deterministic, compactly localized "turbulent
   spots" are both built in, sharded and reproducible independent of the
   device count.

9. **Moving frame of reference.** The `u_grid` parameter integrates the flow
   in a frame translating along the streamwise / axial direction, implicitly
   in both time schemes — convenient for following traveling structures. It
   defaults to the laminar bulk velocity ($1/2$ pipe, $2/3$
   plane-Poiseuille, zero otherwise); set it to `0` for the lab frame.

10. **Buffered, crash-consistent diagnostics with a non-finite guard.**
    Statistics, CFL, and corrector diagnostics stream to `stats.dat`,
    `steps.dat`, and `corrector.dat`, buffered on-device and flushed
    before snapshots and on termination so they stay consistent with the
    saved state. Every flushed value is checked: a NaN or inf aborts the
    run with one line naming the quantity (exit code 3), keeping the
    offending rows on disk for post-mortem and writing no snapshot of the
    broken state.

11. **Wall-time-aware graceful shutdown.** `stop.max_wall_time` takes an
    ISO 8601 duration and ends the run cleanly — final statistics, a final
    snapshot, flushed diagnostics — before a queue kills it, and
    SIGTERM/SIGINT are caught and flush the diagnostic buffers.

12. **External-data import.** `scripts/snapshot_import.py` is a small
    library that packs a velocity field produced elsewhere into a valid
    snapshot, so external data enters the solver and the analysis API as a
    first-class state.

13. **Adaptive CFL time stepping.** `step.adaptive` re-selects the time
    step at runtime from the measured CFL (setpoint `cfl_target`,
    bounds `dt_min`/`dt_max`, per-change limiters and a deadband),
    rebuilding the $\Delta t$-dependent implicit operators on the device
    with no recompilation and, under `cnab2`, ratio-weighting the next
    Adams–Bashforth step — see
    [Temporal discretization](#temporal-discretization).

14. **Machine-precision discrete incompressibility.** The opt-in
    `res.consistent_imm` advances the wall-normal velocity and vorticity
    and reconstructs the tangential components, eliminating the discrete
    pressure — the stepped state's divergence drops from truncation
    level to round-off at any resolution, on the same banded operators
    with fewer solves and less operator storage, and the energy budget
    closes tighter. Available for every wall-bounded flow; changing it
    on resume starts a new trajectory — see
    [The influence-matrix method](#the-influence-matrix-method).

## Use of AI

The first version of this solver — the triply-periodic geometry, the
predictor–corrector stepper, and the Kolmogorov flow — was designed and
written entirely by hand. The extension to the wall-bounded geometries grew
out of that core design with extensive use of LLM coding assistants.
Throughout, the design, the numerical formulation, and the validation
strategy are the author's, and every assisted change was planned, reviewed,
and iterated on by the author, with correctness checked against the test
suite.
