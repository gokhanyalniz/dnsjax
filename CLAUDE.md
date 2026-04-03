## Project Overview

`dnsjax` is a GPU-accelerated pseudo-spectral DNS solver for the 3D incompressible Navier-Stokes equations, written in JAX. It targets triply-periodic flows (Kolmogorov, Waleffe, decaying-box). The solver uses a predictor-corrector time integration scheme (Euler + iterative Crank-Nicolson, following Willis 2017 / openpipeflow).

## Debugging instructions

Do not run the code for runtime testing. You can run `uv run ruff check --fix src/` for linting and to check code syntax.

## Documentation instructions

Add and update docstrings and comments (in LaTeX for math) for any new code and any changed code. In the future MkDocs will be used with MathJax, escape LaTeX commands apppropriately. Keep documentation lines in code to 79 characters wide.

## Architecture

### Package layout (`src/dnsjax/`)

```
__main__.py          # Entry point: parse params, init JAX, run time-stepping loop
parameters.py        # Pydantic parameter models; global singletons params, derived_params, padded_res
sharding.py          # JAX multi-device mesh; global singleton sharding (types, partition specs, shapes)
operators.py         # Spectral operators (cross product); Fourier wavenumber dataclass; phys<->spec FFT wrappers
fft.py               # 3D real FFT with 3/2-rule dealiasing; shard_map for multi-device
rhs.py               # Rotational-form nonlinear term (shared across flow types)
timestep.py          # make_stepper() factory: produces JIT-compiled predict_and_correct, iterate_correction
bench.py             # @timer decorator; timers dict for function timing
geometries/
  triply_periodic.py # Spectral differential operators (curl, div, grad, laplacian) for periodic domains
flows/
  monochromatic.py   # Full flow interface for periodic systems (Kolmogorov, Waleffe, decaying-box)
```

### Key design patterns

**Global singletons at module level**: `params`, `derived_params`, `padded_res` (from `parameters.py`), `sharding` (from `sharding.py`), and `fourier` (from `operators.py`) are all instantiated at import time and mutated by `update_parameters()`. Every module imports and uses these directly. This means import order matters: JAX must be configured (`jax_enable_x64`, platform, distributed) *before* importing any module that uses `sharding` or `operators`. The `__main__.py` enforces this by deferring `import jax` and flow-module imports until after configuration.

**Stepper factory pattern**: `timestep.make_stepper()` takes four flow-specific callables (`get_rhs_fn`, `predict_fn`, `correct_fn`, `norm_fn`) and returns two JIT-compiled functions. Each flow module wires up flow-specific helpers and calls `make_stepper()` at module level, producing `predict_and_correct` and `iterate_correction`. The factory passes extra `*args` (typically `fourier` and `flow` dataclass instances) through to the callables.

**Spectral array layout**: Spectral fields have shape `(ny-1, nz-1, nx//2)`. Physical fields have shape `(ny_padded, nz_padded, nx_padded)` after 3/2-rule oversampling. Nyquist modes are omitted on all stored spectral axes. For multi-device: spectral arrays are sharded on the last axis (kx), physical arrays on the second-to-last (z). The reshard between layouts happens inside `fft.py`.

**Perturbation formulation**: The solver evolves the perturbation `u'` around the laminar base flow `U(y)`. The nonlinear term in `rhs.py` uses the rotational form: `NL = u' x omega' + u' x curl(U) + U x omega' + U x curl(U)`. Base flow terms are precomputed once in the flow dataclass constructor.

**JAX pytree registration**: Flow dataclasses (e.g. `TriplyPeriodicFlow`) are registered as JAX pytrees via `register_dataclass_pytree()` in `sharding.py`, allowing them to be passed through `@jit` boundaries as static-like arguments.

### Parameter layering

Defaults (Pydantic models) -> `parameters.toml` -> CLI args. `update_parameters()` only applies explicitly-set fields, leaving unset fields at their current values.

### Configuration (`parameters.toml`)

Key sections: `[phys]` (re, forcing, oversampling_factor), `[geo]` (lx, lz, tilt_degree), `[res]` (nx, ny, nz, fd_order, double_precision), `[step]` (dt, implicitness, corrector_tolerance), `[stop]` (max_sim_time, max_wall_time as ISO 8601), `[debug]` (correct_divergence, time_functions, measure_corrections), `[dist]` (np, platform).

### JAX-specific notes

- `jax_enable_x64` is set from `params.res.double_precision` before JAX initializes arrays.
- Buffer donation (`donate_argnums`) is used on main time-stepping functions to reuse memory.
- `@jit` and `@timer` decorators are stacked with `@timer` outermost so timing wraps the JIT-compiled function.
- The first time step is excluded from benchmark statistics because it includes JIT compilation overhead.
- FFT normalization uses `norm="forward"` (divides by N on forward, no factor on inverse).
