# dnsjax

**Work in progress!**

A GPU-accelerated & parallelized 3D pseudo-spectral solver for direct numerical simulations of Navier–Stokes equations in [JAX](https://github.com/jax-ml/jax).

Through JAX, it can run on (multiple) CPUs, GPUs, and TPUs.

## Flows

Currently implemented are:
- **Kolmogorov flow** — a *fully-periodic* geometry with monochromatic sine forcing. **[Periodic orbits of Kolmogorov flow](https://doi.org/10.1103/PhysRevLett.126.244502) integrate with dnsjax!**
- **Waleffe flow** — a *fully-periodic* geometry with monochromatic cosine forcing (the Ry symmetry constraint is not yet enforced).
- **Decaying-box turbulence** — a *fully-periodic* geometry with no forcing.
- **Plane-Couette flow** — a *wall-bounded* channel geometry driven by wall motion, with laminar base flow $U(y) = y$ on $y \in [-1, 1]$.

Intent is to eventually add other wall-bounded flows, such as plane-Poiseuille flow, pipe flow, and Taylor-Couette flow.

## Usage

Parameters are layered: Pydantic defaults → optional `parameters.toml` in the working directory → command-line arguments (see `uv run dnsjax --help`). Keys are grouped by section — `[phys]`, `[geo]`, `[res]`, `[step]`, `[stop]`, `[dist]`, `[solver]`, `[debug]`, `[outs]`, `[init]` — mirroring the Pydantic models in `src/dnsjax/parameters.py`.

## Implementation details

### Formulation

The solver evolves the **perturbation** $\mathbf{u}'$ around an analytical laminar base flow $\mathbf{U}(y)$ (e.g. $\sin(2\pi y / L_y)$ for Kolmogorov, $y$ for plane-Couette). The nonlinear term uses the **rotational form**,

$$\mathrm{NL} = \mathbf{u}' \times \boldsymbol{\omega}' + \mathbf{u}' \times \nabla \times \mathbf{U} + \mathbf{U} \times \boldsymbol{\omega}' + \mathbf{U} \times \nabla \times \mathbf{U},$$

obtained by expanding $(\mathbf{u}' + \mathbf{U}) \times \nabla \times (\mathbf{u}' + \mathbf{U})$; the base-flow curl and self-interaction are precomputed once per run. Pressure is treated implicitly to enforce $\nabla \cdot \mathbf{u}' = 0$.

### Pseudo-spectral discretisation

All Fourier directions use the **3/2 rule** for dealiasing: physical fields live on a $\tfrac{3}{2}$-oversampled grid, the nonlinear product is evaluated there, and the result is truncated back. FFTs use `norm="forward"` (divide by $N$ on the forward transform, no factor on the inverse) and omit the Nyquist mode on every stored spectral axis.

Spectral array layouts:
- **Triply-periodic flows** — `(ny-1, nz-1, nx//2)` in $[k_y, k_z, k_x]$ order; 3D real FFT.
- **Wall-bounded flows** — `(nz-1, nx//2, ny)` in $[k_z, k_x, y]$ order (y stays in grid-point space); 2D real FFT.

### Time stepping

Integration follows the second-order predictor-corrector scheme of [openpipeflow](https://openpipeflow.org) (Willis 2017): an explicit Euler predictor followed by iterative Crank–Nicolson correctors until the correction norm drops below `params.step.corrector_tolerance`. The `params.step.implicitness` knob $c$ slides between explicit ($c = 0$), standard Crank–Nicolson ($c = 0.5$), and fully implicit ($c = 1$). `timestep.make_stepper()` turns four flow-specific callables (RHS, predictor, corrector, norm) into JIT-compiled `predict_and_correct` and `iterate_correction` functions with buffer donation. After each step a divergence correction zeroes $\nabla \cdot \mathbf{u}'$ and the mean mode.

### Triply-periodic flows

The Helmholtz operator $\tfrac{1}{\Delta t} - c\,\tfrac{\nabla^2}{\mathrm{Re}}$ is diagonal in Fourier space, so each implicit solve reduces to a pointwise multiply by precomputed coefficients `ldt_1`, `ildt_2`. The mean mode $(k_y, k_z, k_x) = \mathbf{0}$ is zeroed as a passive constant shift.

### Wall-bounded flows

For plane-Couette, wall-normal derivatives use finite differences on a Chebyshev–Gauss–Lobatto grid; `params.res.fd_order` sets the stencil order $p$. `fd.fornberg_weights` computes weights on the non-uniform grid and `fd.build_diff_matrices` assembles the banded $D_1$ and $D_2$ matrices (boundary rows use one-sided stencils).

The pressure Poisson equation with preliminary Neumann boundary conditions is solved via the **influence-matrix method** (IMM): two homogeneous solutions `p1`, `p2` and an influence matrix `M_inv` are precomputed once so that, during time stepping, the correct pressure boundary condition — determined by the normal derivative of the wall-normal velocity at the walls — can be recovered with a small $2 \times 2$ solve, after which pressure and then velocity are updated with the correct pressure gradient.

#### Memory footprint of the IMM solvers

Plane-Couette uses per-mode LU factorizations of the finite-difference operators $L_k$ and $H_k$, one pair per Fourier mode $(k_z, k_x)$. Two storage backends are available through `params.solver.backend`:

- **`banded`** (default): LU factors are kept in LAPACK's banded `ab` packed format, shape $(N_{k_z}, N_{k_x}, 3p+1, N_y)$ with $p$ = `params.res.fd_order`. The $L_k$ and $H_k^-$ actions are rebuilt on the fly from the shared $(N_y, N_y)$ second-derivative matrix $D_2$, so the per-mode matrices themselves are not stored. Total footprint scales as $N_{k_z} \cdot N_{k_x} \cdot (3p+1) \cdot N_y$ — linear in `nx`, linear in `nz`, and **linear in `ny`**.
- **`dense`**: five $(N_{k_z}, N_{k_x}, N_y, N_y)$ arrays per run ($L_k$, $H_k$, $H_k^-$, and the two LU caches). Total footprint scales as $N_{k_z} \cdot N_{k_x} \cdot N_y^2$ — linear in `nx`, linear in `nz`, **quadratic in `ny`**. Kept as a reference / rollback path.

Both backends use $N_{k_z} = $ `nz - 1` and $N_{k_x} = $ `nx // 2`. The factor $\sim N_y / (3p+1)$ saving along `ny` is what lets the banded backend scale to the resolutions needed for wall-bounded turbulence. Concretely, at `nx = nz = 64`, `fd_order = 4`, double precision:

| `ny` | Dense   | Banded | Saving |
|------|---------|--------|--------|
| 64   | 672 MB  | 54 MB  | 92%    |
| 128  | 2.69 GB | 108 MB | 96%    |
| 256  | 10.7 GB | 217 MB | 98%    |
| 512  | 42.9 GB | 434 MB | 99%    |

### Multi-device parallelism and precision

Physical arrays are sharded on the $z$ axis, spectral arrays on the $k_x$ axis; the reshard lives inside `fft.py` as a `shard_map` with an explicit `reshard` between the two layouts. Device setup is driven by `params.dist` (CPU / CUDA / ROCm / TPU, number of processes). `params.res.double_precision` toggles `jax_enable_x64` before any JAX array is created, so float32 and float64 runs share the same code path.

## To-do

### Features
- [ ] Initialize from random states
- [ ] Save to disk stats with a cache
- [ ] Save to disk states
- [ ] Include the [periodic orbits for Kolmogorov flow](https://github.com/gokhanyalniz/dnsbox/wiki/Periodic-orbits)
- [ ] Enforce the Ry symmetry for Waleffe flow
- [x] Banded LU with split-complex solves for wall-bounded flows
- [x] Finite-differences + the influence-matrix method for wall-bounded flows
- [x] Solve for perturbations around the laminar state
- [x] Save diagnostics (parameters, sharding, benchmarks, runtime) to a log file
- [x] Read parameters from a TOML configuration file
- [x] Command-line interface

#### Future
- [ ] Newton solver (with auto-differentiation)
- [ ] Flow symmetries
