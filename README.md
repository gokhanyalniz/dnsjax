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

- **Seven flow systems across four geometries** — pipe, Taylor–Couette, Dean,
  viscoelastic Dean, plane-Poiseuille, plane-Couette, and Kolmogorov flow.
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
  dependency-light NumPy reader; resume is device- and precision-agnostic.
- **Extensively tested** — 19 standalone test scripts pin the numerics, the
  machinery, and the multi-device behavior, and the optimal-growth module
  reproduces published values — see
  [Testing and validation](#testing-and-validation).

## Flows and geometries

| Flow | Geometry | Laminar base / driving | Defining controls |
|---|---|---|---|
| **Pipe** | cylindrical | $U_z = 1 - r^2$, pressure-driven | `re`, axial length `lx` |
| **Taylor–Couette** | annular | $U_\theta = A_0 r + B_0/r$, wall rotation | `re1`, `re2`, `eta` |
| **Dean** | annular | azimuthal body force, total field | `re`, `eta` |
| **Viscoelastic Dean** | annular (sPTT) | azimuthal body force, 9-component total field | `el`, `wi`, `beta`, `epsilon`, `kappa`, `delta` |
| **Plane-Poiseuille** | cartesian | $U = 1 - y^2$, pressure-driven | `re`, `lx`, `lz`, `tilt_degree` |
| **Plane-Couette** | cartesian | $U = y$, wall-driven | `re`, `lx`, `lz`, `tilt_degree` |
| **Kolmogorov** | triply-periodic | $U = \sin(2\pi y / L_y)$, sine body force | `re`, `lx`, `lz` |

A few conventions worth knowing:

- **Reynolds number.** `re` sets the viscosity $\nu = 1/Re$. For the pipe it
  is simultaneously the centerline–radius and the bulk-velocity–diameter
  Reynolds number (the factors of two cancel in the chosen normalization).
- **Taylor–Couette rotation.** `re1` and `re2` are the inner and outer
  cylinder Reynolds numbers on a unit gap, with `re1 >= 0` and `re2` free to
  be negative. The sign pattern selects the configuration: inner-driven
  (`re1 > 0, re2 = 0`), outer-driven (`re1 = 0, re2 > 0`), co-rotating (same
  signs), or counter-rotating (`re2 < 0`); `eta = r_1/r_2` is the radius
  ratio.
- **Viscoelastic controls.** `el` is the elasticity number and sets
  $Re = Wi/El$; `wi` is the Weissenberg number; `beta` the solvent-to-total
  viscosity ratio; `epsilon` the sPTT extensibility; `kappa` an artificial
  stress diffusivity; `delta` the inner radius (the gap is fixed at 2).
- **Grid axes.** For the cylindrical and annular geometries the roles are
  swapped relative to the cartesian intuition: `nx` resolves the axial
  (streamwise) direction, `nz` the azimuthal direction, and `ny` the radial
  direction. The wall-normal extent $L_y$ is fixed by the geometry (the
  channel spans $[-1, 1]$, the pipe radius is 1, the annulus $[r_1, r_2]$,
  and the periodic box uses $L_y = 4$).

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
  --geo.lx 200 \
  --geo.grid_type half-cgl \
  --res.nx 512 --res.ny 48 --res.nz 96 --res.fd_order 4 \
  --step.scheme iterative-cn --step.dt 0.01 \
  --init.localized_rolls True \
  --init.localized_rolls_amplitude 0.2 --init.localized_rolls_width 2.0 \
  --stop.max_sim_time 500 \
  --outs.it_stats 100 --outs.it_snapshot 5000 \
  --dist.platform cpu
```

Reading the flags:

- `--phys.system pipe --phys.re 2300` — the flow and its Reynolds number.
- `--geo.lx 200` — the axial length is 100 pipe diameters ($D = 2$). The
  azimuthal length is fixed at $2\pi$, so `--geo.lz` is never set for a pipe.
- `--geo.grid_type half-cgl` — the radial grid; `half-cgl` is the default
  for a pipe under `iterative-cn`, while `cnab2` uses the rigged-CGL grid
  instead (both are halves of a Chebyshev grid that avoid the axis — see
  [Grids](#grids)).
- `--res.nx 512 --res.ny 48 --res.nz 96` — axial, radial, and azimuthal
  resolution, with fourth-order finite differences in the radial direction.
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
lx = 200.0           # axial length = 100 pipe diameters; lz is fixed at 2*pi
# grid_type defaults to "half-cgl" for pipe + iterative-cn (auto-resolved)

[res]
nx = 512             # axial Fourier modes
ny = 48              # radial finite-difference points
nz = 96              # azimuthal Fourier modes
fd_order = 4

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

What to expect while it runs: the final working parameters and the
physical-space resolution are printed at startup, the first step takes
noticeably longer than the rest (JIT compilation), and a timing summary is
printed at the end. Statistics stream to `stats.dat` (with `steps.dat` and
`corrector.dat` for the CFL and corrector diagnostics), and snapshots appear
as `state00000.tar` (the initial condition), `state00001.tar`, and so on.
Runs end gracefully — at `max_sim_time`, at an ISO 8601
`stop.max_wall_time` budget (writing a final snapshot first), on
relaminarization, or on SIGTERM/SIGINT (flushing the diagnostic buffers) —
so interrupted runs stay consistent with their outputs.

## Parameter layering

Configuration is applied in layers, lowest priority first:

**Pydantic defaults → parameters embedded in a resumed snapshot →
`parameters.toml` → command-line flags.**

Only explicitly set fields override a lower layer, and validation runs once
after the final layer. The parameters that must be known before JAX
initializes — `dist.np0`, `dist.np1`, `dist.platform`, and
`res.double_precision` — are never inherited from a snapshot, and the entire
`solver` section is execution-only. Run `uv run dnsjax --help` for the full
command-line interface (it exits at the parser, so no `mpirun` is needed);
the authoritative field-by-field documentation lives in
[`src/dnsjax/parameters.py`](src/dnsjax/parameters.py).

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
  $(n_x/2) \cdot n_y \cdot (n_z - 1)$ complex numbers, i.e.
  $\approx n_x n_y n_z$ reals. The time stepper holds about three further
  state-sized arrays within a step, and `cnab2` carries one across steps
  (its allocated peak still matches the default scheme's, whose corrector
  branch XLA keeps reserved); Dean and viscoelastic Dean keep one extra
  state-sized laminar reference.
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
  36-field batch with 9 outputs, and `solver.rhs_transform_chunks = k` cuts
  its transform-stage share $k$-fold at identical results.
- **Wall-normal operators** — the Pallas backend stores no-pivot banded LU
  factors: $(2p + 1) \cdot n_y$ reals per matrix per Fourier mode, with the
  half-bandwidth $p$ equal to `fd_order`, over the $(n_z - 1)(n_x/2)$ mode
  plane — that is $m (2p + 1)/2$ fields for $m$ banded matrices, the one
  term that grows with `fd_order`. Here $m = 2$ for
  plane-Couette/Poiseuille, $4$ for pipe, Taylor–Couette, and Dean, and
  $10$ for viscoelastic Dean, plus $v = 3\text{–}6$ field-sized
  boundary-response vectors ($v/2$ fields). Switching to
  `solver.backend = "dense"` replaces $(2p + 1)$ by $n_y$ per matrix — the
  one super-linear option, and the reason Pallas is the wall-bounded
  default. Triply-periodic systems store no matrices at all (their implicit
  solve is diagonal in spectral space), only $\approx 4$ fields of
  wavenumber and inverse-Laplacian coefficients.

Summing these, the leading-order total per device is

```math
\text{wall-bounded:} \qquad
  \Bigl[\, 4 n_c + \tfrac{9}{4} W +
    \tfrac{1}{2} \bigl( m (2p + 1) + v \bigr) \Bigr]
  \, \frac{n_x n_y n_z}{2^{27} \, n_{p0} n_{p1}} \ \text{GiB},
```

```math
\text{triply-periodic:} \qquad
  \Bigl[\, 4 n_c + \tfrac{27}{8} W + 4 \Bigr]
  \, \frac{n_x n_y n_z}{2^{27} \, n_{p0} n_{p1}} \ \text{GiB},
```

with $W \approx 15\text{–}21$ as above (for viscoelastic Dean,
$W \approx 45 + 72/k$ with $k$ = `rhs_transform_chunks`) and
$(n_c, m, v) = (3, 2, 4)$ for the plane flows, $(3, 4, 3)$ for the pipe,
$(3, 4, 6)$ for Taylor–Couette and Dean, and $(9, 10, 6)$ for viscoelastic
Dean. The sum is an upper estimate — XLA's buffer reuse typically realizes
less — and halves at single precision. Off the stepping path, snapshot
writes move each device's bytes directly to disk (staging through host
memory only when GPUDirect Storage is unavailable) and the on-device
diagnostic buffers are resolution-independent, so the optional I/O adds no
resolution-scaled device memory.

## Parallelization

The device grid is $(n_{p0}, n_{p1})$, and the two axes distribute the data
differently:

- **`np0`** splits the wall-normal axis ($y$ / $r$) in physical space and the
  spanwise / azimuthal wavenumber axis ($k_z$ / $m$) in spectral space. The
  split is padding-free when `np0` divides both `ny` and the stored mode
  count $n_z - 1$; otherwise the layer zero-pads to the next multiple and
  strips the padding around the reshard ($n_z - 1$ is odd, so a one-mode
  pad is the norm — and harmless).
- **`np1`** splits the spanwise axis ($z$) in physical space and the
  streamwise wavenumber axis ($k_x$) in spectral space. The spectral side
  is auto-padded the same way (padding-free when `np1` divides $n_x/2$);
  on the physical side the oversampled size
  `nz_padded = oversampling_factor * nz / 2` is rounded up to a multiple of
  `np1` when needed, which amounts to a sliver of extra oversampling.
- Independently of the device grid, the **Pallas banded solver** tiles each
  device's $(k_z, k_x)$ mode plane in blocks of
  (`solver.pallas_block_m0`, `solver.pallas_block_m1`) $= (2, 32)$ and pads
  up to whole tiles. The padded modes cost memory and solve work in
  proportion to the round-up, so per-device mode counts
  $(n_z - 1)/n_{p0}$ and $(n_x/2)/n_{p1}$ near multiples of the block sizes
  are optimal; both knobs are adjustable when the mode plane is small.

No divisibility choice is rejected, and none of the padding is silent:
every adjustment is reported by a one-line startup diagnostic, so its
(usually marginal) cost stays visible.

Crucially, **every device holds the full wall-normal extent in spectral
space**, so the per-mode banded solves need no communication. The forward and
inverse FFTs move data between layouts with two reshards implemented as a
`shard_map` with explicit `reshard` calls; with `np0 = 1` the decomposition
collapses to the one-dimensional $k_x$ / $z$ split. `jax.device_count()`
must equal $n_{p0} \cdot n_{p1}$.

The pipe example above on a $2 \times 2$ device grid (`ny = 48` and
$n_x/2 = 256$ split evenly; `nz = 96` gives `nz_padded = 144`, divisible
by 2 — an entirely padding-free choice):

```bash
# CPU: one device per process
mpirun -np 4 .venv/bin/dnsjax \
  --dist.np0 2 --dist.np1 2 --dist.platform cpu \
  --phys.system pipe --phys.re 2300 --geo.lx 200 \
  --res.nx 512 --res.ny 48 --res.nz 96 \
  --init.localized_rolls True --stop.max_sim_time 500
```

```bash
# GPU: a single process addressing all four GPUs on the node
mpirun -np 1 .venv/bin/dnsjax \
  --dist.np0 2 --dist.np1 2 --dist.platform cuda \
  --phys.system pipe --phys.re 2300 --geo.lx 200 \
  --res.nx 512 --res.ny 48 --res.nz 96 \
  --init.localized_rolls True --stop.max_sim_time 500
```

Because `np0 * np1` counts *devices* rather than processes, a single-node
multi-GPU run is most reliably launched as one process that addresses every
visible GPU; multi-node runs use one process per node spanning that node's
GPUs. The `Distribution` docstring in `parameters.py` covers the SLURM
launch details.

## Snapshots and external data access

A snapshot is a **single uncompressed tar archive** (format version 3)
wrapping a **zarr3** store, a JSON metadata member (parameters, grid, and
lineage), and one contiguous chunk per state component (three velocity
components, or nine for the viscoelastic flow). Each device writes its
disjoint byte ranges into the one file in parallel — directly between GPU
memory and disk when GPUDirect Storage is available, through the host
otherwise — with a concurrent mode for POSIX/parallel filesystems and a
rank-ordered serial mode for filesystems where concurrent writes are
unsafe.
The stored field is the spectral **perturbation** $\mathbf{u}'$ for the
base-flow systems (the laminar state is a zero array) and the **total**
field for Dean and viscoelastic Dean. The archive is readable with ordinary
tools — `tar xf` yields a valid zarr3 store, and in the worst case each
chunk is raw little-endian complex data for `numpy.fromfile`. Resume is
agnostic to the device count and precision, and re-grids a changed
wall-normal grid on load.

For post-processing, `dnsjax.analysis.snapshot_export.read_state` reads a
snapshot into NumPy arrays **without importing JAX or the solver runtime**,
pulling only the requested data off disk:

```python
from dnsjax.analysis.snapshot_export import read_state

data = read_state("state00001.tar")   # NumPy only — no JAX, no solver
u_z, u_r, u_theta = data.physical     # pipe: real fields, native (r, z, θ)
r, z, theta = data.physical_coords    # matching coordinate arrays
re = data.params.phys.re              # embedded parameters

# Cartesian systems return (u_x, u_y, u_z) in the native (y, x, z) layout:
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
operators node-for-node, and `scripts/snapshot_import.py` covers the reverse
direction: packing a velocity field produced elsewhere into a valid
snapshot.

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
even and odd pads alike — so any spanwise / azimuthal `nz` runs as-is on a
single device (likewise `ny` for the triply-periodic box); the oversampled
sizes are rounded up (with a startup note) only when a device-grid axis
must divide them evenly. The streamwise real-FFT axis is never rounded.

### Temporal discretization

Two second-order, semi-implicit schemes share the same predictor and
influence-matrix pressure solve:

- **`iterative-cn`** (default) — an explicit Euler predictor followed by an
  iterative Crank–Nicolson corrector that makes the nonlinear term implicit
  through its fixed point. This costs $2 + c \approx 3$ FFT evaluations per
  step and is stable well past the advective CFL limit.
- **`cnab2`** — Crank–Nicolson viscous term with an explicit second-order
  Adams–Bashforth nonlinear term, costing a **single** FFT evaluation per
  step (roughly a threefold saving on CFL-limited turbulent runs), at the
  price of an advective-CFL step restriction.

The `implicitness` knob $c$ is the Crank–Nicolson weight ($c = 0.5$ is the
trapezoidal rule), with `corrector_tolerance` and `max_corrector_iterations`
governing the fixed point. An **opt-in split corrector**
(`split_corrector`, off by default) iterates the wall-stiff linear coupling
FFT-free between full right-hand-side refreshes; it only helps when the step
is pushed near the corrector iteration cap and is otherwise slower, hence the
default. A related `implicit_mean_coupling` (on by default) folds the
instantaneous mean-flow coupling into the implicit term.

A moving frame of reference (`u_grid`) translates the domain along the
streamwise / axial direction and is integrated implicitly by both schemes —
convenient for following traveling structures. By default the frame moves at
the laminar bulk velocity: $1/2$ for the pipe, $2/3$ for plane-Poiseuille,
and zero for the others (Dean's driving is azimuthal, so its axial bulk
vanishes). The pipe and plane-Poiseuille flows therefore integrate, and
store snapshots, in the moving frame unless `u_grid` is set to `0` (see also
item 8 in [Additional features](#additional-features)).

### Grids

The default wall-normal grid is a **Chebyshev–Gauss–Lobatto (CGL)** grid for
the cartesian and annular geometries. The pipe's radial grid is half of a
CGL grid, in one of two variants: the **rigged-CGL** grid is the positive
half of a CGL grid with an *odd* number of points — whose middle node falls
exactly on the axis and is dropped — placing its innermost node about one
grid spacing from the axis, while the **half-CGL** grid is the positive half
of a CGL grid with an *even* number of points (no node ever sits on the
axis), placing it about half as far. `cnab2` defaults to the rigged grid:
its explicit azimuthal advection near the axis limits the time step in
proportion to that innermost radius. `iterative-cn` integrates the
near-axis coupling implicitly and defaults to the finer-resolving half-CGL
grid. Optional `tanh` stretching (`grid_stretch`) and fully **custom grids**
(via a `geo.wall_grid` file) are supported; quadrature is spectral
Clenshaw–Curtis on CGL grids and an order-`fd_order` composite rule
otherwise.

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

## Testing and validation

The test suite is 19 standalone scripts under `tests/`, run directly
(`uv run python tests/test_cartesian.py` — no test framework), several of
which launch real `mpirun` multi-device runs. Among the guarantees they pin:

- **Solvers and operators** — the Pallas banded kernel against a dense
  reference solver, per-geometry operators and matvecs against NumPy
  constructions, and CUDA-lowering guards that catch Triton compilation
  regressions on CPU-only machines.
- **The physics** — laminar states step at machine precision, random and
  localized initial conditions integrate through the full nonlinear path for
  all seven flows, and both time steppers converge at second order.
- **The machinery** — snapshot round-trips readable by standard tools,
  device-count- and precision-agnostic resume with lineage checks, and the
  JAX-free import guarantee of the analysis API.
- **Physical validation** — the transient-growth module reproduces published
  optimal-growth values for all four of its flows to about 2%.

`scripts/` adds benchmark and diagnostic tools: `solver_benchmark.py`
(Pallas-vs-dense validation and benchmark, including multi-GPU),
`pallas_solve_profile.py` (where the banded solve's time goes),
`pallas_tiling_diagnostic.py` (a GPU miscompile-isolation harness), and
`pivot_stability_survey.py` (the evidence behind the no-pivot LU stability
tolerances).

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
   wall-normal total profile for the pipe, Taylor–Couette, plane-Poiseuille,
   and plane-Couette flows, reusing the solver's own linear step for each
   Fourier mode. It runs on a single device (GPU-capable) and reproduces
   published optimal-growth values for all four flows to about 2%.

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

4. **Standard-tools-readable snapshots and a JAX-free reader.** The tar +
   zarr3 format — written in parallel, directly from GPU memory when
   GPUDirect Storage is available — and the NumPy-only `read_state` cleanly
   separate the runtime from the analysis API — see
   [Snapshots and external data access](#snapshots-and-external-data-access).

5. **Robust resume.** Snapshots resume across any device count and precision,
   re-grid a changed wall-normal grid on load, and track lineage,
   distinguishing a genuine continuation from a new trajectory when the
   physics or geometry changes.

6. **Laminarization auto-stop.** A run terminates automatically once the
   perturbation energy drops below a threshold, so relaminarization events
   are captured without babysitting — natural for lifetime and
   edge-of-chaos studies.

7. **Initial-condition generators.** Divergence-free random fields (the
   default start mode) and deterministic, compactly localized "turbulent
   spots" are both built in, sharded and reproducible independent of the
   device count.

8. **Moving frame of reference.** The `u_grid` parameter integrates the flow
   in a frame translating along the streamwise / axial direction, implicitly
   in both time schemes — convenient for following traveling structures. It
   defaults to the laminar bulk velocity ($1/2$ pipe, $2/3$
   plane-Poiseuille, zero otherwise); set it to `0` for the lab frame.

9. **Buffered, crash-consistent diagnostics.** Statistics, CFL, and corrector
   diagnostics stream to `stats.dat`, `steps.dat`, and `corrector.dat`,
   buffered on-device and flushed around snapshots and on termination so they
   stay consistent with the saved state.

10. **Wall-time-aware graceful shutdown.** `stop.max_wall_time` takes an
    ISO 8601 duration and ends the run cleanly — final statistics, a final
    snapshot, flushed diagnostics — before a queue kills it, and
    SIGTERM/SIGINT are caught and flush the diagnostic buffers.

11. **External-data import.** `scripts/snapshot_import.py` is a small
    library that packs a velocity field produced elsewhere into a valid
    snapshot, so external data enters the solver and the analysis API as a
    first-class state.

## Use of AI

The first version of this solver — the triply-periodic geometry, the
predictor–corrector stepper, and the Kolmogorov flow — was designed and
written entirely by hand. The extension to the wall-bounded geometries grew
out of that core design with extensive use of LLM coding assistants.
Throughout, the design, the numerical formulation, and the validation
strategy are the author's, and every assisted change was planned, reviewed,
and iterated on by the author, with correctness checked against the test
suite.
