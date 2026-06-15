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
Laminar smoke (1D multi-device): `uv run python tests/test_laminar_smoke.py --np 2`
Laminar smoke (2D multi-device): `uv run python tests/test_laminar_smoke.py --np 4 --np0 2`

### Smoke test (laminar time stepping)

Any `python -m dnsjax` run must be launched via `mpirun` (even single-process: `mpirun -np 1 ...`); `__main__` unconditionally initializes the JAX distributed backend. Under `mpirun`, invoke the interpreter as `.venv/bin/python` directly (`uv run` does not compose with `mpirun`). A run writes `stats.dat`/`steps.dat` (and any snapshots) to the cwd, so launch manual smoke/debug runs from a scratch dir, using the absolute path to the repo's `.venv/bin/python`.

`mpirun -np 2 python -m dnsjax --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 27`

For double parallelisation (tanh grid recommended for clean ny divisibility):
`mpirun -np 4 python -m dnsjax --dist.np0 2 --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 28 --geo.grid_type tanh`

The laminar state should time step with a single corrector step, with stepping error of O(-18) or less, and perturbation energy of O(-32) or less. Run existing tests listed in section Tests below if you touch the modules they test.

### Generate random initial condition

`uv run python scripts/random_field.py --system plane-couette --nx 128 --ny 65 --nz 128 --amplitude 0.1 --smoothness 0.4 --seed 1 --output random_ic`

Generates a divergence-free random perturbation (obeying BCs) and saves it as a zarr3 snapshot. Load with `--init.snapshot random_ic --init.start_from_laminar False`. Supports all flow systems. Run `--test` for self-verification: it checks the configured system's generator and exits with a pass/fail status, writing no snapshot (so `--output` is not needed in this mode). See `scripts/random_field.py` docstring for the full algorithm and CLI options.

## Documentation instructions

Keep docstrings, comments (in LaTeX for math for both) and typing up-to-date. In the future MkDocs will be used with MathJax, escape LaTeX commands appropriately. Keep documentation lines in code to 79 characters wide. Keep CLAUDE.md files up-to-date (root and subdirectory files).

**Documentation layering**: detailed descriptions of algorithms, array shapes, mathematical formulations, and per-function behaviour belong in code docstrings and comments. CLAUDE.md files serve as a concise index for AI agents: structural overview, cross-cutting constraints, copy-paste commands, and pointers to the relevant code (e.g. "see `module.py` module docstring" or "see `function` docstring in `file.py`"). When adding new functionality, put the detail in the code and add a brief entry or pointer in the appropriate CLAUDE.md only if it introduces a new module, a cross-cutting pattern, or a non-obvious constraint that isn't discoverable from a single file's docstrings.

## Architecture

### Package layout (`src/dnsjax/`)

```
__main__.py           Entry point; import-order enforcement, stats buffering, snapshot resume
parameters.py         Pydantic parameter models; global singletons params, derived_params, padded_res
sharding.py           JAX multi-device mesh; global singleton sharding; pytree registration
operators.py          Wavenumber helpers; vmapped 3D/2D FFT wrappers; vector cross product
fft.py                3D/2D real FFT with 3/2-rule dealiasing; shard_map; double-parallelisation reshards
rhs.py                Rotational-form perturbation nonlinear term (shared across flow types); measure_fn hook for physical-space measurements
measurements.py       Physical-space measurements consumed via the rhs.py hook (currently the CFL diagnostic, get_cfl)
timestep.py           make_stepper() factory; JIT-compiled predict_and_correct / predict_and_fully_correct (+ _measured variant)
fd.py                 FD utilities (Fornberg weights, D1/D2, quadrature weights, interpolation matrices); NumPy-only, importable standalone without JAX/params setup
solvers.py            Geometry-independent linear solvers: DenseJAXSolver, PerModeBandedOperator (SPIKE)
snapshot.py           Snapshot save/load: zarr3, np-agnostic resume, raw offset I/O (GDS or host)
geometries/
  wall_bounded/       Wall-bounded geometry family (see wall_bounded/CLAUDE.md)
    _base.py          Shared wall-bounded infrastructure (norms, init_state, stepper builder)
    cartesian.py      Cartesian: Fourier, CGL grid, CartesianFlow, IMM, Lk/Hk operators
    cylindrical.py    Cylindrical: Fourier, half-CGL grid, CylindricalFlow, decoupled u+/u-/uz, 1x1 IMM
    annular.py        Annular (concentric cylinders): Fourier, CGL grid on [r1,r2], AnnularFlow, decoupled u+/u-/uz (no parity/ghost, no r=0), 2x2 IMM
  triply_periodic/    Triply-periodic geometry family (see triply_periodic/CLAUDE.md)
    triply_periodic.py  Fourier, spectral diff ops, TriplyPeriodicFlow, algebraic Helmholtz, divergence correction
flows/
  wall_bounded/
    plane_couette.py    PlaneCouetteFlow(CartesianFlow): U(y) = y with tilt
    plane_poiseuille.py PlanePoiseuilleFlow(CartesianFlow): Us = 1-y^2 with tilt
    pipe.py             PipeFlow(CylindricalFlow): Uz = 1 - r^2
    taylor_couette.py   TaylorCouetteFlow(AnnularFlow): circular-Couette Uθ = A0 r + B0/r (shear-driven, no pressure gradient)
  triply_periodic/
    monochromatic.py    MonochromaticFlow(TriplyPeriodicFlow): Kolmogorov / Waleffe / decaying-box
```

### Code-exploration constraints

The two geometry families are **completely independent**. The directory structure enforces this:

- `geometries/wall_bounded/` and `flows/wall_bounded/` are unrelated to `geometries/triply_periodic/` and `flows/triply_periodic/`. Do not explore across families unless explicitly prompted.
- Wall-bounded family documentation: `src/dnsjax/geometries/wall_bounded/CLAUDE.md`
- Triply-periodic family documentation: `src/dnsjax/geometries/triply_periodic/CLAUDE.md`

### Key design patterns

**Global singletons and import order**: `params`, `derived_params`, `padded_res` (from `parameters.py`), `sharding` (from `sharding.py`), and a geometry-specific `fourier` are instantiated at import time and mutated by `update_parameters()`. JAX must be configured (`jax_enable_x64`, platform, distributed) *before* importing any module that uses `sharding` or a geometry module. See `__main__.py` module docstring.

**Stepper factory (two layers)**: `timestep.make_stepper()` takes four geometry-general callables and returns JIT-compiled stepping functions, including `predict_and_fully_correct` (fused corrector loop via `lax.while_loop`, the primary path). Each geometry family wraps it in its own builder that binds the `fourier` and `flow` singletons. See `timestep.py`, `_base.py`, and `triply_periodic.py` docstrings.

**Spectral array layout and sharding**: see `sharding.py` module docstring for shapes, partition specs, and the `(np0, np1)` device mesh. See `fft.py` module docstring for the reshard pipeline and spectral padding.

**Perturbation formulation**: the solver evolves `u'` around laminar `U(y)`. The rotational-form nonlinear term and base-flow gradient elimination are documented in the `rhs.py` module docstring.

**JAX pytree registration**: `register_dataclass_pytree()` in `sharding.py` registers geometry dataclasses, flow subclasses, solver classes, and Fourier classes as JAX pytrees. See its docstring for details.

### Parameter layering

Defaults (Pydantic models) -> `parameters.toml` -> CLI args. `update_parameters()` only applies explicitly-set fields, leaving unset fields at their current values.

### Configuration (`parameters.toml`)

See `parameters.py` classes for full documentation. Key sections:

| Section    | Key fields                                                                                             |
|------------|--------------------------------------------------------------------------------------------------------|
| `[phys]`   | `re`, `re1`/`re2` (Taylor-Couette inner/outer cylinder Reynolds numbers; derive `re := Re_ref`), `system`, `oversampling_factor`, `oversample_y`, `driving` (`"constant_pressure_gradient"` / `"constant_bulk_velocity"`), `block_mean_spanwise_velocity` (Taylor-Couette: blocks the mean axial velocity) |
| `[geo]`    | `lx`, `lz`, `tilt_degree`, `eta` (Taylor-Couette radius ratio r1/r2), `wall_grid` (custom grid file), `grid_type` (`"tanh"` / `"cgl"`), `grid_stretch` |
| `[res]`    | `nx`, `ny`, `nz`, `fd_order`, `double_precision`                                                      |
| `[init]`   | `start_from_laminar`, `snapshot`, `t0`, `it0`                                                          |
| `[outs]`   | `it_stats`, `it_steps` (CFL diagnostic cadence -> `steps.dat`), `it_snapshot`, `it_error_check` (host-sync cadence for corrector convergence), `nbuffer`, `stats_precision`, `snapshot_write_mode` (`"concurrent"` / `"serial"`) |
| `[step]`   | `dt`, `implicitness`, `corrector_tolerance`, `max_corrector_iterations`                                |
| `[stop]`   | `max_sim_time`, `max_wall_time` (ISO 8601)                                                            |
| `[dist]`   | `np0` (wall-normal / kz axis), `np1` (spanwise / kx axis), `platform`                                 |
| `[solver]` | `backend` (`"banded"` / `"dense"`), `spike_block_size`, `block_thomas`                                 |

The default `parameters.toml` contains only `[phys] [geo] [res] [init] [outs] [step] [stop]`; `[dist]` and `[solver]` rely on model defaults -- set them via CLI (e.g. `--dist.np1 2`, `--solver.backend dense`) or by adding the section.

### Diagnostics (`stats.dat`, `steps.dat`)

On-device buffered stats, flushed periodically to `stats.dat`. The CFL diagnostic (every `outs.it_steps` steps, measured from physical-space velocity inside the nonlinear-term evaluation -- see `measurements.py` and the `rhs.py` `measure_fn` hook) is buffered and flushed to `steps.dat` the same way. See `__main__.py` module docstring for the buffering mechanism and file format.

### Snapshots

Zarr3 format with 3 combined per-component files (np-agnostic resume at any `(np0, np1)` configuration). `_dnsjax_meta.json` stores simulation time, iteration, layout, grid, and full params. When the wall-normal grid differs from the snapshot's, the state is interpolated at load time (`_interpolate_if_needed` in `__main__.py`; interpolation methods in `fd.py`). See `snapshot.py` module docstring for on-disk layouts, I/O engines, memory, and write modes.

### JAX-specific notes

- Explicit mode sharding is used globally rather than Auto mode, which propagates shardings on arrays for most operations. Do not use `jax.lax.with_sharding_constraint`.
- Avoid allocating a global array first and then distributing it with `jax.device_put` to devices after when such an array can be directly allocated on individual devices via the `out_sharding` argument for array-allocating calls like `jnp.zeros`, `ndarray.at.get(...)` and `ndarray.at.set(...)` etc. When this is not possible, do not use `jnp.asarray` just to avoid a `jax.device_put`.
- `jax_enable_x64` is set from `params.res.double_precision` before JAX initializes arrays.
- JAX has no zero-copy complex<->real bitcast (`lax.bitcast_convert_type` rejects complex; `.view()` lowers to scatter). Real-operator x complex-field GEMMs/solves use an explicit trailing re/im split at half the promoted-complex FLOPs — reuse `apply_y_matrix` (`geometries/wall_bounded/_base.py`) or the `solvers.py` pattern.
- Buffer donation (`donate_argnums`) is used on main time-stepping functions to reuse memory.
- The first time step is excluded from benchmark statistics because it includes JIT compilation overhead.
- FFT normalization uses `norm="forward"` (divides by N on forward, no factor on inverse).
- Dicts returned from jitted functions (`get_stats`, measurement dicts) are canonicalized to **sorted key order** by pytree flattening — this sets the column order of `stats.dat` / `steps.dat`; never assume insertion order.

## Scripts
- `scripts/spike_partition_info.py`: display SPIKE block-partition trade-offs for a given resolution.
- `scripts/random_field.py`: generate a random divergence-free perturbation and save as a zarr3 snapshot. Supports all flow systems (Cartesian wall-bounded, cylindrical, annular, triply-periodic). Uses `build_cartesian_grid` / `build_cylindrical_grid` / `build_annular_grid` from the geometry modules for grid/FD setup without constructing the full flow dataclass. Taylor-Couette needs `--re1 --re2 --eta`. Per-mode divergence-free enforcement uses NumPy loops (not JAX) to avoid tracing overhead; all other array work uses JAX. Run with `--test` for self-verification (divergence-free, wall BCs, norm, Hermitian symmetry, seed determinism); the self-test runs the block for the configured `--system` (cartesian or annular).

## Tests
All to be kept up-to-date as the respective modules change:
- `tests/test_banded_solver.py` contains geometry-independent SPIKE solver tests.
- `tests/test_cartesian.py` contains Cartesian operator and matvec tests.
- `tests/test_cylindrical.py` contains cylindrical operator and matvec tests.
- `tests/test_annular.py` contains annular (Taylor-Couette) operator/matvec tests, the 2x2 SPIKE-vs-dense parity, and circular-Couette coefficient (A0/B0) checks.
- `tests/test_integration.py` contains quadrature weight tests.
- `tests/test_mean_mask.py` checks that padding slots carry nonzero placeholder wavenumbers and `Fourier.mean_mask` is the unique k^2 = 0 (mean) mode under forced spectral padding (subprocess, forced CPU devices).
- `tests/test_laminar_smoke.py` runs all wall-bounded flows from laminar state (via subprocess/mpirun) checking stepping error, perturbation energy, and the `steps.dat` CFL columns against analytic laminar values. Each subprocess runs in a temp dir: `parameters.toml` is not loaded (model defaults + CLI args only).
- `tests/test_snapshot.py` round-trips snapshots (save/load equality, np-agnostic resume, `load_y_slice`) for all on-disk layouts via the host I/O path (subprocess per system/device-count, multi-device via forced CPU devices).
