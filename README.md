# dnsjax

**Work in progress!**

A GPU-accelerated & parallelized 3D pseudo-spectral solver for direct numerical simulations of Navier–Stokes equations in [JAX](https://github.com/jax-ml/jax).

Through JAX, it can run on (multiple) CPUs, GPUs, and TPUs.

## Flows

Currently implemented are:
- **Kolmogorov flow** — a *fully-periodic* geometry with monochromatic sine forcing. **[Periodic orbits of Kolmogorov flow](https://doi.org/10.1103/PhysRevLett.126.244502) integrate with dnsjax!**
- **Waleffe flow** — a *fully-periodic* geometry with monochromatic cosine forcing (the Ry symmetry constraint is not yet enforced).
- **Decaying-box turbulence** — a *fully-periodic* geometry with no forcing.

Intent is to eventually add other wall-bounded flows, such as plane-Couette flow, plane-Poiseuille flow, pipe flow, and Taylor-Couette flow.

## Usage

Parameters are layered: Pydantic defaults → optional `parameters.toml` in the working directory → command-line arguments (see `uv run dnsjax --help`). Keys are grouped by section — `[phys]`, `[geo]`, `[res]`, `[step]`, `[stop]`, `[dist]`, `[solver]`, `[debug]`, `[outs]`, `[init]` — mirroring the Pydantic models in `src/dnsjax/parameters.py`.

## Implementation details

### Formulation

The solver evolves the **perturbation** $\mathbf{u}'$ around an analytical laminar base flow $\mathbf{U}(y)$ (e.g. $\sin(2\pi y / L_y)$ for Kolmogorov). The nonlinear term uses the **rotational form**,

$$\mathrm{NL} = \mathbf{u}' \times \boldsymbol{\omega}' + \mathbf{u}' \times \nabla \times \mathbf{U} + \mathbf{U} \times \boldsymbol{\omega}' + \mathbf{U} \times \nabla \times \mathbf{U},$$

obtained by expanding $(\mathbf{u}' + \mathbf{U}) \times \nabla \times (\mathbf{u}' + \mathbf{U})$; the base-flow curl and self-interaction are precomputed once per run. Pressure is treated implicitly to enforce $\nabla \cdot \mathbf{u}' = 0$.

### Pseudo-spectral discretisation

All Fourier directions use the **3/2 rule** for dealiasing: physical fields live on a $\tfrac{3}{2}$-oversampled grid, the nonlinear product is evaluated there, and the result is truncated back. FFTs use `norm="forward"` (divide by $N$ on the forward transform, no factor on the inverse) and omit the Nyquist mode on every stored spectral axis.

Spectral array layout: `(ny-1, nz-1, nx//2)` in $[k_y, k_z, k_x]$ order.

### Time stepping

Integration follows the second-order predictor-corrector scheme of [openpipeflow](https://openpipeflow.org) (Willis 2017): an explicit Euler predictor followed by iterative Crank–Nicolson correctors until the correction norm drops below `params.step.corrector_tolerance`. The `params.step.implicitness` knob $c$ slides between explicit ($c = 0$), standard Crank–Nicolson ($c = 0.5$), and fully implicit ($c = 1$). `timestep.make_stepper()` turns four flow-specific callables (RHS, predictor, corrector, norm) into JIT-compiled `predict_and_correct` and `iterate_correction` functions with buffer donation. After each step a divergence correction zeroes $\nabla \cdot \mathbf{u}'$ and the mean mode.

### Triply-periodic flows

The Helmholtz operator $\tfrac{1}{\Delta t} - c\,\tfrac{\nabla^2}{\mathrm{Re}}$ is diagonal in Fourier space, so each implicit solve reduces to a pointwise multiply by precomputed coefficients `ldt_1`, `ildt_2`. The mean mode $(k_y, k_z, k_x) = \mathbf{0}$ is zeroed as a passive constant shift.

### Multi-device parallelism and precision

Physical arrays are sharded on the $z$ axis, spectral arrays on the $k_x$ axis; the reshard lives inside `fft.py` as a `shard_map` with an explicit `reshard` between the two layouts. Device setup is driven by `params.dist` (CPU / CUDA / ROCm / TPU, number of processes). `params.res.double_precision` toggles `jax_enable_x64` before any JAX array is created, so float32 and float64 runs share the same code path.

## To-do

### Features
- [ ] Initialize from random states
- [ ] Save to disk stats with a cache
- [ ] Save to disk states
- [ ] Include the [periodic orbits for Kolmogorov flow](https://github.com/gokhanyalniz/dnsbox/wiki/Periodic-orbits)
- [ ] Enforce the Ry symmetry for Waleffe flow
- [x] Solve for perturbations around the laminar state
- [x] Save diagnostics (parameters, sharding, benchmarks, runtime) to a log file
- [x] Read parameters from a TOML configuration file
- [x] Command-line interface

#### Future
- [ ] Newton solver (with auto-differentiation)
- [ ] Flow symmetries
