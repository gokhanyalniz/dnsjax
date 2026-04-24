## Project Overview

`dnsjax` is a GPU-accelerated pseudo-spectral + finite-differences DNS solver for the 3D incompressible Navier-Stokes equations, written in JAX. It targets triply-periodic flows (Kolmogorov, Waleffe, decaying-box) and wall-bounded flows (plane-Couette). The solver uses a predictor-corrector time integration scheme (Euler + iterative Crank-Nicolson, following Willis 2017 / openpipeflow).

## Debugging instructions

You can run `uv run ruff check --fix` for linting and static checks. For runtime tests, you may run a given flow from its laminar state a few time steps with low resolution on two devices, like: `mpirun -np 2 python -m dnsjax --dist.np 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 9`. The laminar state should time step with a single corrector step, with stepping error of O(-18) or less, and perturbation energy of O(-32) or less.

## Documentation instructions

Add and update docstrings and comments (in LaTeX for math) for any new code and any changed code. In the future MkDocs will be used with MathJax, escape LaTeX commands apppropriately. Keep documentation lines in code to 79 characters wide. Keep CLAUDE.md up-to-date.

## Architecture

### Package layout (`src/dnsjax/`)

```
__main__.py          # Entry point: parse params, init JAX, run time-stepping loop
parameters.py        # Pydantic parameter models; global singletons params, derived_params, padded_res
sharding.py          # JAX multi-device mesh; global singleton sharding (types, partition specs, shapes)
operators.py         # Wavenumber helpers (real_harmonics, complex_harmonics); vector cross product; vmapped 3D/2D phys<->spec FFT wrappers
fft.py               # 3D/2D real FFT with 3/2-rule dealiasing; shard_map for multi-device
rhs.py               # Rotational-form nonlinear term (shared across flow types)
timestep.py          # make_stepper() factory: produces JIT-compiled predict_and_correct, iterate_correction
bench.py             # @timer decorator; timers dict for function timing
fd.py                # Finite-difference utilities (Fornberg weights, D1/D2 matrices)
geometries/
  triply_periodic.py # Fourier class, spectral diff ops (curl, div, grad, laplacian), norms, TriplyPeriodicFlow base dataclass, algebraic Helmholtz predict/correct, divergence correction, build_triply_periodic_stepper factory
  cartesian.py       # Fourier class, norms with Simpson y-integration, CartesianFlow base dataclass, IMM precomputation, LU solvers (dense/banded), Kleiser-Schumann IMM iteration, build_cartesian_stepper factory
flows/
  monochromatic.py   # MonochromaticFlow(TriplyPeriodicFlow): base flow and forcing for Kolmogorov / Waleffe / decaying-box; diagnostics (E, I, D, E')
  plane_couette.py   # PlaneCouetteFlow(CartesianFlow): plane-Couette base flow U(y) = y; diagnostics
```

### Key design patterns

**Global singletons at module level**: `params`, `derived_params`, `padded_res` (from `parameters.py`), `sharding` (from `sharding.py`), and a geometry-specific `fourier` (from `geometries/triply_periodic.py` or `geometries/cartesian.py`) are all instantiated at import time and mutated by `update_parameters()`. Every module imports and uses these directly. This means import order matters: JAX must be configured (`jax_enable_x64`, platform, distributed) *before* importing any module that uses `sharding` or a geometry module. The `__main__.py` enforces this by deferring `import jax` and flow-module imports until after configuration.

**Stepper factory pattern (two layers)**: `timestep.make_stepper()` is the shared core — it takes four geometry-general callables (`get_rhs_fn`, `predict_fn`, `correct_fn`, `norm_fn`) and returns JIT-compiled `predict_and_correct` / `iterate_correction`, threading extra `*args` (typically `fourier` and a `flow` dataclass instance) through to the callables. Each geometry module wraps this in a higher-level builder — `build_triply_periodic_stepper(flow)` in `geometries/triply_periodic.py` and `build_cartesian_stepper(flow)` in `geometries/cartesian.py` — that captures the geometry's `_get_rhs` / `_predict` / `_correct` / `_norm`, calls `make_stepper`, and binds the `fourier` and `flow` singletons into closures. The builder returns `(predict_and_correct, iterate_correction, init_state_bound)` for wall-bounded geometries, plus `correct_velocity` for triply-periodic (where the divergence-free constraint is enforced algebraically rather than by the IMM). Flow modules call the builder at module level to expose the public interface consumed by `__main__`.

**Spectral array layout**: Spectral fields have shape `(ny-1, nz-1, nx//2)` for periodic flows or `(nz-1, nx//2, ny)` for wall-bounded flows (where y stays in grid-point space). Physical fields have shape `(ny_padded, nz_padded, nx_padded)` after 3/2-rule oversampling. Nyquist modes are omitted on all stored spectral axes. For multi-device: spectral arrays are sharded on the last (triply-periodic) or second-to-last (wall-bounded) axis (kx), physical arrays on the second-to-last (z). The reshard between layouts happens inside `fft.py`.

**Perturbation formulation**: The solver evolves the perturbation `u'` around the laminar base flow `U(y)`. The nonlinear term in `rhs.py` uses the rotational form: `NL = u' x omega' + u' x curl(U) + U x omega' + U x curl(U)`. Base flow terms are precomputed once in the flow dataclass constructor.

**JAX pytree registration**: Geometry base dataclasses (`TriplyPeriodicFlow`, `CartesianFlow`) and their flow subclasses (`MonochromaticFlow(TriplyPeriodicFlow)`, `PlaneCouetteFlow(CartesianFlow)`), along with the geometry-specific `Fourier` classes and the solver dataclasses (`DenseJAXSolver`, `PerModeBandedOperator`), are registered as JAX pytrees via `register_dataclass_pytree()` in `sharding.py`, allowing them to be passed through `@jit` boundaries as static-like arguments.

**Wall-bounded flows use the influence-matrix method (IMM)**: For plane-Couette, the pressure Poisson equation with preliminary Neumann BCs is solved via LU-factored matrices (`Lk`, `Hk`). Only the per-mode *operator factors* are built on the CPU; all IMM homogeneous data (`p1, p2, v1, v2, q1, q2, M_inv`) is derived on-device by `CartesianFlow._derive_imm_homogeneous_data` from the already-factored GPU operator, avoiding a second dense LU per mode. `params.solver.backend` selects the operator-factor storage format: `"banded"` (default) runs LAPACK `dgbtrf` per mode on the CPU and ships LU factors in LAPACK's packed `ab` layout of shape `(Nkz, Nkx, 3p+1, Ny)` with `p = params.res.fd_order`; the triangular solves then run on device via two `lax.scan`s (see `_banded_solve_device` and `PerModeBandedOperator` in `geometries/cartesian.py`), and the `Lk` and `Hk_minus` matvecs are reconstructed on the fly from the shared `D2` / `D1` matrices by `_lk_matvec` / `_hk_minus_matvec`. `"dense"` ships the raw `(Nkz, Nkx, Ny, Ny)` matrices (plus cached `Lk`, `Hk_minus`) to the GPU and factors them on-device via `DenseJAXSolver` — a reference path kept for parity with the banded backend. The IMM uses the homogeneous solutions (`p1`, `p2`) and influence matrix `M_inv` during timestepping to find the correct pressure boundary condition from the normal derivative of the wall-normal velocity at the wall; pressure is then solved with that BC, and velocity is updated with the corresponding pressure gradient. Operator factors are distributed across devices via `jax.make_array_from_callback`; homogeneous data inherits the kx-sharded layout from the GPU operator.

### Parameter layering

Defaults (Pydantic models) -> `parameters.toml` -> CLI args. `update_parameters()` only applies explicitly-set fields, leaving unset fields at their current values.

### Configuration (`parameters.toml`)

Key sections: `[phys]` (re, forcing, oversampling_factor), `[geo]` (lx, lz, tilt_degree), `[res]` (nx, ny, nz, fd_order, double_precision), `[step]` (dt, implicitness, corrector_tolerance), `[stop]` (max_sim_time, max_wall_time as ISO 8601), `[debug]` (time_functions), `[dist]` (np, platform), `[solver]` (backend: `"banded"` or `"dense"`).

### JAX-specific notes

- `jax_enable_x64` is set from `params.res.double_precision` before JAX initializes arrays.
- Buffer donation (`donate_argnums`) is used on main time-stepping functions to reuse memory.
- `@jit` and `@timer` decorators are stacked with `@timer` outermost so timing wraps the JIT-compiled function.
- The first time step is excluded from benchmark statistics because it includes JIT compilation overhead.
- FFT normalization uses `norm="forward"` (divides by N on forward, no factor on inverse).
