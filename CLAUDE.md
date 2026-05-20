## Project Overview

`dnsjax` is a GPU-accelerated pseudo-spectral + finite-differences DNS solver for the 3D incompressible Navier-Stokes equations, written in JAX. It targets triply-periodic flows (Kolmogorov, Waleffe, decaying-box) and wall-bounded flows (plane-Couette, plane-Poiseuille, pipe). The solver uses a predictor-corrector time integration scheme (Euler + iterative Crank-Nicolson, following Willis 2017 / openpipeflow).

## Commands

### Prerequisites

Python >=3.14, `uv`, MPI (for multi-device runs).

### Setup

`uv sync`

### Lint

`uv run ruff check --fix`

### Run tests

Single file: `uv run python tests/test_cartesian.py`
Laminar smoke (multi-device): `uv run python tests/test_laminar_smoke.py --np 2`

### Smoke test (laminar time stepping)

`mpirun -np 2 python -m dnsjax --dist.np 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 27`

The laminar state should time step with a single corrector step, with stepping error of O(-18) or less, and perturbation energy of O(-32) or less. Run existing tests listed in section Tests below if you touch the modules they test.

## Documentation instructions

Keep docstrings, comments (in LaTeX for math for both) and typing up-to-date. In the future MkDocs will be used with MathJax, escape LaTeX commands appropriately. Keep documentation lines in code to 79 characters wide. Keep CLAUDE.md files up-to-date (root and subdirectory files).

## Architecture

### Package layout (`src/dnsjax/`)

```
__main__.py          # Entry point: parse params, init JAX, run time-stepping loop
parameters.py        # Pydantic parameter models; global singletons params, derived_params, padded_res
sharding.py          # JAX multi-device mesh; global singleton sharding (types, partition specs, shapes)
operators.py         # Wavenumber helpers (real_harmonics, complex_harmonics); vector cross product; batched 2D / vmapped 3D phys<->spec FFT wrappers
fft.py               # 3D/2D real FFT with 3/2-rule dealiasing; shard_map for multi-device
rhs.py               # Rotational-form nonlinear term (shared across flow types)
timestep.py          # make_stepper() factory: produces JIT-compiled predict_and_correct, iterate_correction, predict_and_fully_correct (fused corrector loop via lax.while_loop)
fd.py                # Finite-difference utilities (Fornberg weights, D1/D2 matrices, composite quadrature weights for arbitrary non-uniform grids)
solvers.py           # Geometry-independent linear solvers: DenseJAXSolver (batched dense LU), PerModeBandedOperator (SPIKE block-partitioned banded solver), SPIKE factorisation, block-partitioning, and partition validation utilities
geometries/
  wall_bounded/      # Wall-bounded geometry family (see wall_bounded/CLAUDE.md)
    _base.py         # Shared wall-bounded infrastructure: integrate_scalar, get_inprod/get_norm2/get_norm, init_state, build_wall_bounded_stepper factory, phys_to_spec/spec_to_phys aliases
    cartesian.py     # Fourier class, Clenshaw-Curtis weights, CartesianFlow base dataclass (with tilt and constant-bulk-velocity support), on-device IMM operator assembly (Lk/Hk builders for dense and banded backends), Kleiser-Schumann IMM iteration, build_cartesian_stepper factory
    cylindrical.py   # Fourier class, get_norm2_cyl, CylindricalFlow base dataclass, half-CGL radial grid, parity-reduced FD matrices, decoupled u+/u-/uz operators (Lk, Hk_plus, Hk_minus, Hk_z), 1x1 IMM, build_cylindrical_stepper factory
  triply_periodic/   # Triply-periodic geometry family (see triply_periodic/CLAUDE.md)
    triply_periodic.py # Fourier class, spectral diff ops (curl, div, grad, laplacian), norms, TriplyPeriodicFlow base dataclass, algebraic Helmholtz predict/correct, divergence correction, build_triply_periodic_stepper factory
flows/
  wall_bounded/
    plane_couette.py   # PlaneCouetteFlow(CartesianFlow): plane-Couette base flow U(y) = y with tilt; diagnostics
    plane_poiseuille.py # PlanePoiseuilleFlow(CartesianFlow): plane-Poiseuille base flow Us = 1-y^2 with tilt; diagnostics (E', dPds')
    pipe.py            # PipeFlow(CylindricalFlow): pipe base flow Uz = 1 - r^2; diagnostics
  triply_periodic/
    monochromatic.py   # MonochromaticFlow(TriplyPeriodicFlow): base flow and forcing for Kolmogorov / Waleffe / decaying-box; diagnostics (E, I, D, E')
```

### Code-exploration constraints

The two geometry families are **completely independent**. The directory structure enforces this:

- `geometries/wall_bounded/` and `flows/wall_bounded/` are unrelated to `geometries/triply_periodic/` and `flows/triply_periodic/`. Do not explore across families unless explicitly prompted.
- Wall-bounded family documentation: `src/dnsjax/geometries/wall_bounded/CLAUDE.md`
- Triply-periodic family documentation: `src/dnsjax/geometries/triply_periodic/CLAUDE.md`

### Key design patterns

**Global singletons at module level**: `params`, `derived_params`, `padded_res` (from `parameters.py`), `sharding` (from `sharding.py`), and a geometry-specific `fourier` (from the respective geometry module) are all instantiated at import time and mutated by `update_parameters()`. Every module imports and uses these directly. This means import order matters: JAX must be configured (`jax_enable_x64`, platform, distributed) *before* importing any module that uses `sharding` or a geometry module. The `__main__.py` enforces this by deferring `import jax` and flow-module imports until after configuration.

**Stepper factory pattern (two layers)**: `timestep.make_stepper()` is the shared core -- it takes four geometry-general callables (`get_rhs_fn`, `predict_fn`, `correct_fn`, `norm_fn`) and returns JIT-compiled `predict_and_correct` / `iterate_correction` / `predict_and_fully_correct`, threading extra `*args` (typically `fourier` and a `flow` dataclass instance) through to the callables. `predict_and_fully_correct` fuses the predictor and all corrector iterations into a single JIT scope via `lax.while_loop`, eliminating per-iteration GPU-to-CPU synchronisation; it is the primary path used by `__main__`. Each geometry family wraps `make_stepper` in its own builder (see family-specific CLAUDE.md).

**Spectral array layout**: Spectral fields have shape `(ny-1, nz-1, nx//2)` for periodic flows, `(nz-1, nx//2, ny)` for wall-bounded flows (where y stays in grid-point space). Physical fields, after 3/2-rule oversampling, have shape `(ny_padded, nz_padded, nx_padded)` for periodic flows, `(ny, nz_padded, nx_padded)` for wall-bounded flows. Nyquist modes are omitted on all stored spectral axes. For multi-device: spectral arrays are sharded on the last (triply-periodic) or second-to-last (wall-bounded) axis (kx for both), physical arrays on the second-to-last (z). The reshard between layouts happens inside `fft.py`.

**Perturbation formulation**: The solver evolves the perturbation `u'` around the laminar base flow `U(y)`. The nonlinear term in `rhs.py` uses the rotational form of the perturbation equation: `NL = u' x omega' + u' x curl(U) + U x omega'`. The base-flow self-interaction `U x curl(U) = grad(|U|^2/2)` is omitted because it is a pure gradient absorbed by the pressure. All three cross-product contributions are computed per output component in a single fused `jnp.array` expression (`_fused_nonlinear`), eliminating intermediate concatenation and scatter kernels. Base flow fields (`base_flow`, `curl_base_flow`) are precomputed once in the flow dataclass constructor.

**JAX pytree registration**: Geometry base dataclasses (`TriplyPeriodicFlow`, `CartesianFlow`, `CylindricalFlow`) and their flow subclasses (`MonochromaticFlow(TriplyPeriodicFlow)`, `PlaneCouetteFlow(CartesianFlow)`, `PlanePoiseuilleFlow(CartesianFlow)`, `PipeFlow(CylindricalFlow)`), along with the geometry-specific `Fourier` classes and the solver dataclasses (`DenseJAXSolver`, `PerModeBandedOperator`), are registered as JAX pytrees via `register_dataclass_pytree()` in `sharding.py`, allowing them to be passed through `@jit` boundaries as static-like arguments.

### Parameter layering

Defaults (Pydantic models) -> `parameters.toml` -> CLI args. `update_parameters()` only applies explicitly-set fields, leaving unset fields at their current values.

### Configuration (`parameters.toml`)

Key sections: `[phys]` (re, system, oversampling_factor, oversample_y, driving: `"constant_pressure_gradient"` (default) or `"constant_bulk_velocity"` for pipe and plane-Poiseuille flows, block_mean_spanwise_velocity: `false` (default) -- zeroes mean-mode spanwise bulk velocity for Cartesian flows), `[geo]` (lx, lz, tilt_degree), `[res]` (nx, ny, nz, fd_order, double_precision), `[init]` (start_from_laminar, snapshot, t0, it0), `[outs]` (it_stats), `[step]` (dt, implicitness, corrector_tolerance, max_corrector_iterations), `[stop]` (max_sim_time, max_wall_time as ISO 8601), `[dist]` (np, platform), `[solver]` (backend: `"banded"` or `"dense"`, spike_block_size: optional target SPIKE block size `m`).

### JAX-specific notes

- Explicit mode sharding is used globally rather than Auto mode, which propagates shardings on arrays for most operations. Do not use `jax.lax.with_sharding_constraint`.
- Avoid allocating a global array first and then distributing it with `jax.device_put` to devices after when such an array can be directly allocated on individual devices via the `out_sharding` argument for array-allocating calls like `jnp.zeros`, `ndarray.at.get(...)` and `ndarray.at.set(...)` etc. When this is not possible, do not use `jnp.asarray` just to avoid a `jax.device_put`.
- `jax_enable_x64` is set from `params.res.double_precision` before JAX initializes arrays.
- Buffer donation (`donate_argnums`) is used on main time-stepping functions to reuse memory.
- The first time step is excluded from benchmark statistics because it includes JIT compilation overhead.
- FFT normalization uses `norm="forward"` (divides by N on forward, no factor on inverse).

## Tests
All to be kept up-to-date as the respective modules change:
- `tests/test_banded_solver.py` contains geometry-independent SPIKE solver tests.
- `tests/test_cartesian.py` contains Cartesian operator and matvec tests.
- `tests/test_cylindrical.py` contains cylindrical operator and matvec tests.
- `tests/test_integration.py` contains quadrature weight tests.
- `tests/test_laminar_smoke.py` runs all wall-bounded flows from laminar state (via subprocess/mpirun) checking stepping error and perturbation energy.
