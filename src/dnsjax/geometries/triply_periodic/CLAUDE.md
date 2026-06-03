## Triply-periodic geometry

### Module layout

- `triply_periodic.py`: Fourier class, spectral differential operators (derivative, curl, div, grad, laplacian, inverse_laplacian), norms, TriplyPeriodicFlow base dataclass, init_state, algebraic Helmholtz predict/correct, correct_divergence, build_triply_periodic_stepper factory

### TriplyPeriodicFlow base dataclass

Fields precomputed by the constructor:
- `ldt_1`: explicit time-stepping coefficient `$1/\Delta t + (1-c)\,\nabla^2/\mathrm{Re}$`
- `ildt_2`: inverse implicit coefficient `$(1/\Delta t - c\,\nabla^2/\mathrm{Re})^{-1}$`
- Both are zeroed at the mean mode `$(k_x, k_y, k_z) = 0$`

Subclasses must set `base_flow` and `curl_base_flow` after calling `super().__post_init__()`.

### Stepper factory (triply-periodic layer)

`build_triply_periodic_stepper(flow)` wraps `timestep.make_stepper()` directly and additionally returns `correct_velocity` (where the divergence-free constraint is enforced algebraically rather than by the IMM). Flow modules call the builder at module level to expose the public interface consumed by `__main__`.

### Key differences from wall-bounded

- Helmholtz inversion is algebraic (pointwise multiply by `$ildt\_2 = (1/\Delta t - c\,\nabla^2/\mathrm{Re})^{-1}$` where `$c$` is the implicitness parameter) -- no matrix solves or LU factorisations
- Divergence correction is a separate post-step projection (not built into an IMM)
- Spectral layout: `(ny-1, nz_spec, nx_spec)` with `[ky, kz, kx]` -- ky fully local, kz sharded by np0, kx sharded by np1. With `np0=1`, collapses to the original 1D scheme (only kx distributed).

### Parallelization (double decomposition)

The 3D FFT uses the same two-reshard pipeline as the wall-bounded 2D FFT, with an additional y-FFT step:

- Forward: x-FFT -> [reshard #1: z<->kx] -> z-FFT -> [reshard #2: y<->kz] -> y-FFT
- Physical `(y_np0, z_np1, x)` -> spectral `(ky, kz_np0, kx_np1)`

After reshard #2, the full y-extent becomes local on each device, enabling the y-FFT on the full extent. The output ky axis is fully local; kz is sharded by np0; kx is sharded by np1. Wall-bounded keeps y in grid-point space throughout (no y-FFT); triply-periodic adds the y-FFT after reshard #2 brings y local. The two reshard operations handle the same z<->kx and y<->kz exchanges regardless of geometry.

`np0` requires `ny_padded` (= `oversampling_factor * ny // 2`) to be divisible; if not, `ny_padded` is automatically bumped to the next multiple (marginally more oversampling, physically neutral).

### Pressure projection

Incompressibility is enforced in two places:
- Inside `_get_rhs`: algebraic pressure Poisson solve (`$\nabla^2 p = \nabla \cdot \mathbf{NL}$`, inverted via `inverse_laplacian`) projects the nonlinear term to be divergence-free before the Helmholtz step
- After time stepping: `correct_divergence` (returned as `correct_velocity` by the builder) removes any residual divergence accumulated during the corrector iterations

### Flows

- `flows/triply_periodic/monochromatic.py`: MonochromaticFlow(TriplyPeriodicFlow) -- base flow and forcing for Kolmogorov / Waleffe / decaying-box; diagnostics (E, I, D, E')
