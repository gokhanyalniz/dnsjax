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
single- or double-precision. Time advancement uses one of two second-order,
semi-implicit schemes.

## Highlights

- **Seven flow systems across four geometries** — pipe, Taylor–Couette, Dean,
  viscoelastic Dean, plane-Poiseuille, plane-Couette, and Kolmogorov flow.
- **Runs anywhere JAX runs** — CPU, GPU, or TPU, on one device or many, from
  the same code path.
- **Two second-order integrators** — an iterative Crank–Nicolson
  predictor–corrector and a Crank–Nicolson / Adams–Bashforth scheme that costs
  a single FFT evaluation per step — built from one stepper factory.
- **A custom banded-LU GPU kernel** — a Pallas/Triton per-mode wall-normal
  solver with $O(N_y p)$ storage, backed by a dense reference solver for
  validation.
- **Non-modal stability built in** — 3D linear optimal energy growth $G(t)$
  around an arbitrary wall-normal profile, reusing the solver's own linear
  step.
- **A coupled viscoelastic model** — a simplified Phan-Thien–Tanner (sPTT)
  conformation-tensor flow riding the same component-agnostic machinery.
- **Portable data** — snapshots are plain tar + zarr3, readable with standard
  tools and a dependency-light NumPy reader; resume is device- and
  precision-agnostic.

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
  Reynolds number (the factors of two cancel in the chosen normalization), so
  `re = 2300` is exactly the classical transitional value.
- **Taylor–Couette rotation.** `re1` and `re2` are the inner and outer
  cylinder Reynolds numbers on a unit gap, with `re1 >= 0` and `re2` free to be
  negative. The sign pattern selects the configuration: inner-driven
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
  channel spans $[-1, 1]$, the pipe radius is 1, the annulus $[r_1, r_2]$, and
  the periodic box uses $L_y = 4$).

## Installation

```bash
git clone https://github.com/gokhanyalniz/dnsjax.git
cd dnsjax
uv sync
```

Requires Python ≥ 3.14 and [`uv`](https://docs.astral.sh/uv/); an MPI runtime
is needed for multi-device runs. The default install pulls a CPU build of
JAX. To run on **CUDA GPUs**, replace `jax` with the CUDA-13 build:

```bash
uv add "jax[cuda13]"    # rewrites the jax requirement, re-locks, and re-syncs
```

(equivalently, change the `jax>=…` line in `pyproject.toml` to
`jax[cuda13]>=…` and run `uv sync`). The CUDA wheels are Linux x86-64 only.

## Running a simulation

The flagship example below runs a **100-diameter pipe at Re = 2300**, started
from a localized-roll perturbation (a "puff"), on a single CPU device. Every
problem-defining parameter — the physics, the geometry, the resolution, and
the time integrator — is written out explicitly, so switching to another flow
is a matter of editing values rather than learning the defaults.

A `python -m dnsjax` run is always launched through `mpirun` (even for one
process), invoking the environment's Python directly. Output files
(`stats.dat`, snapshots, …) are written to the current directory, so launch
from a scratch directory:

```bash
mpirun -np 1 .venv/bin/python -m dnsjax \
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
- `--geo.grid_type half-cgl` — the radial grid; `half-cgl` is the default for a
  pipe under `iterative-cn` (it is invalid under `cnab2`, which uses a
  rigged Chebyshev–Gauss–Lobatto grid).
- `--res.nx 512 --res.ny 48 --res.nz 96` — axial, radial, and azimuthal
  resolution, with fourth-order finite differences in the radial direction.
- `--step.scheme iterative-cn --step.dt 0.01` — the default predictor–corrector
  integrator at a wall-bounded-safe step.
- `--init.localized_rolls …` — a compact, deterministic finite-amplitude spot
  (amplitude 0.2) that seeds transition.
- `--stop.max_sim_time 500` — stop at $t = 500$ advective units (transition
  develops over $O(100)$ units; the run also stops early if the puff
  relaminarizes).
- `--dist.platform cpu` — a single CPU device.

This configuration needs about **0.3 GiB resident and 0.6 GiB peak** memory —
comfortable on a laptop — though the physics is compute-bound over hundreds of
advective units, so a GPU (see [Parallelization](#parallelization)) is the
practical choice for production. **Switching flows** is a one-line change:
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
localized_rolls_amplitude = 0.2   # peak |u'| of the puff
localized_rolls_width = 2.0       # axial localization half-width

[step]
dt = 0.01
scheme = "iterative-cn"

[outs]
it_stats = 100
it_snapshot = 5000

[stop]
max_sim_time = 500.0
# check_laminarization = true (default) stops the run if the puff relaminarizes
```

## Parameter layering

Configuration is applied in layers, lowest priority first:

**Pydantic defaults → parameters embedded in a resumed snapshot →
`parameters.toml` → command-line flags.**

Only explicitly set fields override a lower layer, and validation runs once
after the final layer. The parameters that must be known before JAX
initializes — `dist.np0`, `dist.np1`, `dist.platform`, and
`res.double_precision` — are never inherited from a snapshot, and the entire
`solver` section is execution-only. Run `uv run dnsjax --help` for the full
command-line interface; the authoritative field-by-field documentation lives
in [`src/dnsjax/parameters.py`](src/dnsjax/parameters.py).

## Memory footprint

Memory is dominated by two multipliers: **oversampling** — the nonlinear term
is evaluated on a $\tfrac32$-padded grid, so the physical working set is
$2.25\times$ the spectral mode count for wall-bounded flows and $3.375\times$
for triply-periodic flows — and **precision** ($2\times$ for the default
double precision; halve every figure below for single). Resident memory is the
spectral state plus the banded wall-normal operators; the transient peak adds
the oversampled physical fields of one nonlinear-term evaluation. All totals
scale as $n^3$; per device they scale as the global figure divided by
$n_{p0} \cdot n_{p1}$.

The table gives **runtime / peak** GiB, at double precision, on a single
device, for a cubic resolution $n^3$. Wall-bounded rows use the production
**Pallas** banded backend; the triply-periodic row uses its algebraic
(diagonal) implicit solve, its only backend.

| Flow (geometry) | 64³ | 128³ | 192³ | 256³ |
|---|---|---|---|---|
| Pipe (cylindrical) | 0.03 / 0.06 | 0.25 / 0.46 | 0.84 / 1.6 | 2.0 / 3.7 |
| Taylor–Couette, Dean (annular) | 0.05 / 0.08 | 0.39 / 0.60 | 1.3 / 2.0 | 3.1 / 4.8 |
| Plane-Poiseuille, Plane-Couette (cartesian) | 0.03 / 0.06 | 0.25 / 0.46 | 0.84 / 1.6 | 2.0 / 3.7 |
| Viscoelastic Dean (annular, 9-component) | 0.12 / 0.28 | 0.97 / 2.2 | 3.3 / 7.5 | 7.8 / 18 |
| Kolmogorov (triply-periodic) | 0.01 / 0.05 | 0.08 / 0.39 | 0.26 / 1.3 | 0.63 / 3.2 |

Notes:

- The **peak** column is the leading transient term (the oversampled physical
  batch); a measured high-water mark is somewhat higher owing to FFT stage
  buffers and per-corrector temporaries.
- The **viscoelastic** peak is set by a fused ~36-field right-hand-side
  transform; `solver.rhs_transform_chunks = k` reduces that transient roughly
  $k$-fold at identical results.
- Real runs are anisotropic; the flagship pipe above ($512\times48\times96$)
  needs ≈ 0.3 / 0.6 GiB.

## Parallelization

The device grid is $(n_{p0}, n_{p1})$, and the two axes distribute the data
differently:

- **`np0`** splits the wall-normal axis ($y$ / $r$) in physical space and the
  spanwise / azimuthal wavenumber axis ($k_z$ / $m$) in spectral space.
- **`np1`** splits the spanwise axis ($z$) in physical space and the
  streamwise wavenumber axis ($k_x$) in spectral space.

Crucially, **every device holds the full wall-normal extent in spectral
space**, so the per-mode banded solves need no communication. The forward and
inverse FFTs move data between layouts with two reshards implemented as a
`shard_map` with explicit `reshard` calls; with `np0 = 1` the decomposition
collapses to the one-dimensional $k_x$ / $z$ split.

The layer **auto-pads** when a mode count is not divisible by the device grid:
the spectral $k_z$ axis is zero-padded to the next multiple of `np0` (and the
physical $y$ axis with it, stripped after the reshard). The remaining
**restrictions** are that `nz_padded = oversampling_factor * nz // 2` must be
divisible by `np1`, that `jax.device_count()` must equal $n_{p0} \cdot n_{p1}$,
and that the complex spanwise axis requires an even oversampling difference (a
very small `nz` such as 6 is rejected; use 8 or more). A power-of-two `ny`
avoids the wall-normal padding entirely.

The flagship pipe run on a $2 \times 2$ device grid (here `nz = 96` gives
`nz_padded = 144`, divisible by 2, and `ny = 48` splits evenly):

```bash
# CPU: one device per process
mpirun -np 4 .venv/bin/python -m dnsjax \
  --dist.np0 2 --dist.np1 2 --dist.platform cpu \
  --phys.system pipe --phys.re 2300 --geo.lx 200 \
  --res.nx 512 --res.ny 48 --res.nz 96 \
  --init.localized_rolls True --stop.max_sim_time 500
```

```bash
# GPU: a single process addressing all four GPUs on the node
mpirun -np 1 .venv/bin/python -m dnsjax \
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
components, or nine for the viscoelastic flow). The stored field is the
spectral **perturbation** $\mathbf{u}'$ for the base-flow systems (the laminar
state is a zero array) and the **total** field for Dean and viscoelastic Dean.
The archive is readable with ordinary tools — `tar xf` yields a valid zarr3
store, and in the worst case each chunk is raw little-endian complex data for
`numpy.fromfile`. Resume is agnostic to the device count and precision, and
re-grids a changed wall-normal grid on load.

For post-processing, `dnsjax.analysis.snapshot_export.read_state` reads a
snapshot into NumPy arrays **without importing JAX or the solver runtime**,
pulling only the requested components (and wall-normal slabs) off disk:

```python
from dnsjax.analysis.snapshot_export import read_state

data = read_state("state00001.tar")   # NumPy only — no JAX, no solver runtime
u_z, u_r, u_theta = data.physical      # real fields, native (r, z, θ) layout
r, z, theta = data.physical_coords     # matching coordinate arrays
re = data.params.phys.re               # embedded parameters
```

The companion `dnsjax.analysis.snapshot_ops` module provides `divergence`,
`curl`, `gradient`, and `integrate` that reproduce the solver's *discrete*
operators node-for-node.

## Governing equations and numerics

### Equations

The solver advances the non-dimensional incompressible Navier–Stokes
equations,

$$
\partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla)\mathbf{u}
  = -\nabla p + \frac{1}{Re}\nabla^2 \mathbf{u},
\qquad \nabla \cdot \mathbf{u} = 0 .
$$

Most systems evolve the **perturbation** $\mathbf{u}'$ about an analytical
laminar profile $\mathbf{U}$, using the **rotational form** of the nonlinear
term,

$$
\mathrm{NL} = \mathbf{u}' \times \boldsymbol{\omega}'
  + \mathbf{u}' \times (\nabla \times \mathbf{U})
  + \mathbf{U} \times \boldsymbol{\omega}',
\qquad \boldsymbol{\omega}' = \nabla \times \mathbf{u}',
$$

in which the base-flow self-interaction is a pure gradient absorbed by the
pressure. The force-driven Dean and viscoelastic-Dean systems instead
integrate the **total** field with a mean-mode azimuthal body force.

The viscoelastic flow couples a symmetric **conformation tensor** $\mathbf{c}$
through a simplified Phan-Thien–Tanner constitutive law,

$$
\partial_t \mathbf{c} + (\mathbf{u}\cdot\nabla)\mathbf{c}
  - (\nabla\mathbf{u})^\top \mathbf{c} - \mathbf{c}\,(\nabla\mathbf{u})
  = \kappa \nabla^2 \mathbf{c}
  - \frac{1 - 3\epsilon + \epsilon\,\mathrm{tr}\,\mathbf{c}}{Wi}
    (\mathbf{c} - \mathbf{I}),
$$

with the polymer stress feeding momentum as
$\tfrac{1-\beta}{Re\,Wi}\nabla\cdot\mathbf{c}$ and solvent viscosity
$\nu = \beta/Re$. This grows the state to nine components on the same solver
machinery.

### Spatial discretization

Periodic directions are treated **pseudo-spectrally** with Fourier
transforms; the single wall-bounded direction uses **banded finite
differences** with Fornberg weights of order `fd_order` (half-bandwidth
$p = $ `fd_order`). The quadratic nonlinearity is dealiased with the **3/2
rule** — physical fields are evaluated on a $\tfrac32$-oversampled grid and the
product is truncated back — and the Nyquist mode is dropped on every stored
spectral axis (FFTs use `norm="forward"`).

### Temporal discretization

Two second-order, semi-implicit schemes share the same predictor and
influence-matrix pressure solve:

- **`iterative-cn`** (default) — an explicit Euler predictor followed by an
  iterative Crank–Nicolson corrector that makes the nonlinear term implicit
  through its fixed point. This costs $2 + c \approx 3$ FFT evaluations per
  step and is stable well past the advective CFL limit.
- **`cnab2`** — Crank–Nicolson viscous term with an explicit second-order
  Adams–Bashforth nonlinear term, costing a **single** FFT evaluation per step
  (roughly a threefold saving on CFL-limited turbulent runs), at the price of
  an advective-CFL step restriction.

The `implicitness` knob $c$ is the Crank–Nicolson weight ($c = 0.5$ is the
trapezoidal rule), with `corrector_tolerance` and `max_corrector_iterations`
governing the fixed point. An **opt-in split corrector**
(`split_corrector`, off by default) iterates the wall-stiff linear coupling
FFT-free between full right-hand-side refreshes; it only helps when the step is
pushed near the corrector iteration cap and is otherwise slower, hence the
default. A related `implicit_mean_coupling` (on by default) folds the
instantaneous mean-flow coupling into the implicit term. A moving frame of
reference (`u_grid`) translates the domain along the homogeneous direction and
is integrated implicitly by both schemes — convenient for capturing traveling
or rotating states.

### Grids

The default wall-normal grid is a **Chebyshev–Gauss–Lobatto (CGL)** grid for
the cartesian and annular geometries. The cylindrical geometry uses a
**half-CGL** grid under `iterative-cn` (and a rigged-CGL grid under `cnab2`),
chosen so that the innermost node sits far enough from the axis to relax the
near-axis azimuthal-advection time-step limit. Optional `tanh` stretching
(`grid_stretch`) and fully **custom grids** (via a `geo.wall_grid` file) are
supported; quadrature is spectral Clenshaw–Curtis on CGL grids and an
order-`fd_order` composite rule otherwise.

### The influence-matrix method

Enforcing incompressibility together with the wall boundary conditions is the
central difficulty of a wall-bounded spectral discretization: the wall-normal
momentum equation supplies only the *interior* pressure Poisson problem, while
the correct wall pressure boundary condition is fixed *indirectly* by
requiring $\nabla \cdot \mathbf{u} = 0$ at the walls. The **Kleiser–Schumann
influence-matrix method** resolves this by precomputing a small set of
homogeneous responses once, so that each time step recovers the boundary
condition with a tiny per-mode solve — $1 \times 1$ for the pipe's single wall,
$2 \times 2$ for the two-walled cartesian and annular geometries — after which
the velocity is corrected by linearity. Algebraically this is a
**Schur-complement (capacitance-matrix) reduction** whose boundary correction
is a low-rank (Woodbury / Sherman–Morrison) update, which is what keeps the
wall-bounded solve inexpensive at scale.

## References

The numerics follow several standard references (bibliographic details are
worth confirming against the originals):

- A. P. Willis, *The Openpipeflow Navier–Stokes solver*, SoftwareX **6**,
  124–127 (2017). Predictor–corrector time stepping and the decoupled
  $u_\pm$ pipe/annular formulation.
- L. Kleiser and U. Schumann, *Treatment of incompressibility and boundary
  conditions in three-dimensional numerical spectral simulations of plane
  channel flows* (1980). The influence-matrix method.
- C. Canuto, M. Y. Hussaini, A. Quarteroni, and T. A. Zang, *Spectral
  Methods*, Springer (2006–2007). Spectral methods and the influence-matrix
  treatment.
- B. Fornberg, *Calculation of weights in finite difference formulas*, SIAM
  Review **40**(3), 685–691 (1998). Finite-difference weights on non-uniform
  grids.
- L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM (2000). Clenshaw–Curtis
  quadrature and spectral differentiation.
- J. A. C. Weideman and S. C. Reddy, *A MATLAB differentiation matrix suite*,
  ACM TOMS **26**(4), 465–519 (2000). Differentiation and interpolation
  matrices.
- P. J. Schmid and D. S. Henningson, *Stability and Transition in Shear
  Flows*, Springer (2001). Non-modal stability and transient growth.
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
   Fourier mode. It runs on a single device (GPU-capable) and has been
   validated against literature anchors for all four flows.

   ```bash
   python -m dnsjax.analysis.transient_growth --help
   ```

2. **A custom banded-LU GPU kernel with a dense oracle.** The per-mode
   wall-normal solves run through a hand-written Pallas/Triton banded-LU
   sweep that stores $O(N_y p)$ factors instead of the dense $O(N_y^2)$, and a
   dense reference solver validates it numerically. Kernels are checked both
   in Pallas interpret mode and by lowering to CUDA on CPU-only machines.

3. **Multi-device sharding on CPU/GPU/TPU.** A two-axis $(n_{p0}, n_{p1})$
   device mesh with an in-FFT reshard pipeline distributes the work while
   keeping the wall-normal solves communication-free — see
   [Parallelization](#parallelization).

4. **Standard-tools-readable snapshots and a JAX-free reader.** The tar +
   zarr3 format and the NumPy-only `read_state` cleanly separate the runtime
   from the analysis API — see
   [Snapshots and external data access](#snapshots-and-external-data-access).

5. **Robust resume.** Snapshots resume across any device count and precision,
   re-grid a changed wall-normal grid on load, and track lineage, distinguishing
   a genuine continuation from a new trajectory when the physics or geometry
   changes.

6. **Laminarization auto-stop.** A run terminates automatically once the
   perturbation energy drops below a threshold, so relaminarization events are
   captured without babysitting — natural for lifetime and edge-of-chaos
   studies.

7. **Initial-condition generators.** Divergence-free random fields (the
   default start mode) and deterministic, compactly localized "turbulent
   spots" are both built in, sharded and reproducible independent of the
   device count.

8. **Moving frame of reference.** The `u_grid` parameter integrates the flow
   in a frame translating along the homogeneous direction, integrated
   implicitly by both time schemes — convenient for traveling and rotating
   states.

9. **Buffered, crash-consistent diagnostics.** Statistics, CFL, and corrector
   diagnostics stream to `stats.dat`, `steps.dat`, and `corrector.dat`,
   buffered on-device and flushed around snapshots and on termination so they
   stay consistent with the saved state.
