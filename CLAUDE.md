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

`mpirun -np 2 python -m dnsjax --dist.np 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 27`

For double parallelisation (ny must be divisible by np0; tanh grid recommended):
`mpirun -np 4 python -m dnsjax --dist.np 4 --dist.np0 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 28 --geo.grid_type tanh`

The laminar state should time step with a single corrector step, with stepping error of O(-18) or less, and perturbation energy of O(-32) or less. Run existing tests listed in section Tests below if you touch the modules they test.

### Generate random initial condition

`uv run python scripts/random_field.py --system plane-couette --nx 128 --ny 65 --nz 128 --amplitude 0.1 --smoothness 0.4 --seed 1 --output random_ic`

Generates a divergence-free random perturbation (obeying BCs) and saves it as a zarr3 snapshot. Load with `--init.snapshot random_ic --init.start_from_laminar False`. Supports all flow systems. Run `--test` for self-verification. See `scripts/random_field.py` docstring for the full algorithm and CLI options.

## Documentation instructions

Keep docstrings, comments (in LaTeX for math for both) and typing up-to-date. In the future MkDocs will be used with MathJax, escape LaTeX commands appropriately. Keep documentation lines in code to 79 characters wide. Keep CLAUDE.md files up-to-date (root and subdirectory files).

## Architecture

### Package layout (`src/dnsjax/`)

```
__main__.py          # Entry point: parse params, init JAX, run time-stepping loop; buffers diagnostics on-device, flushes to stats.dat at intervals
parameters.py        # Pydantic parameter models; global singletons params, derived_params, padded_res
sharding.py          # JAX multi-device mesh; global singleton sharding (types, partition specs, shapes)
operators.py         # Wavenumber helpers (real_harmonics, complex_harmonics); vector cross product; batched 2D / vmapped 3D phys<->spec FFT wrappers
fft.py               # 3D/2D real FFT with 3/2-rule dealiasing; shard_map for multi-device
rhs.py               # Rotational-form nonlinear term (shared across flow types)
timestep.py          # make_stepper() factory: produces JIT-compiled predict_and_correct, iterate_correction, predict_and_fully_correct (fused corrector loop via lax.while_loop)
fd.py                # Finite-difference utilities (Fornberg weights, D1/D2 matrices, composite quadrature weights for arbitrary non-uniform grids)
solvers.py           # Geometry-independent linear solvers: DenseJAXSolver (batched dense LU), PerModeBandedOperator (SPIKE block-partitioned banded solver), SPIKE factorisation, block-partitioning, and partition validation utilities
snapshot.py          # Snapshot save/load: zarr3 format, 3 combined per-component clean-global files (np-agnostic resume), selectable y_major/native layout, raw offset I/O (GDS/kvikIO or host), streamed slab-by-slab (no full transpose); save_snapshot, load_snapshot, load_y_slice, validate_snapshot_params
geometries/
  wall_bounded/      # Wall-bounded geometry family (see wall_bounded/CLAUDE.md)
    _base.py         # Shared wall-bounded infrastructure: integrate_scalar, get_inprod/get_norm2/get_norm, get_pert_enstrophy, init_state, build_wall_bounded_stepper factory, phys_to_spec/spec_to_phys aliases, extract_mean_mode
    cartesian.py     # Fourier class, Clenshaw-Curtis weights, build_cartesian_grid (shared grid/FD/weights factory), CartesianFlow base dataclass (with tilt and constant-bulk-velocity support), on-device IMM operator assembly (Lk/Hk builders for dense and banded backends), Kleiser-Schumann IMM iteration, build_cartesian_stepper factory
    cylindrical.py   # Fourier class, get_norm2_cyl, build_half_cgl_grid, build_parity_reduced_matrices, build_cylindrical_grid (shared grid/FD/weights factory), CylindricalFlow base dataclass, half-CGL radial grid, parity-reduced FD matrices, decoupled u+/u-/uz operators (Lk, Hk_plus, Hk_minus, Hk_z), 1x1 IMM, build_cylindrical_stepper factory
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

**Spectral array layout**: Spectral fields have shape `(ny-1, nz_spec, nx_spec)` for periodic flows, `(nz_spec, nx_spec, ny)` for wall-bounded flows (where y stays in grid-point space). `nz_spec` and `nx_spec` may exceed the true mode counts (`nz-1` and `nx//2`) by up to `np0-1` or `np1-1` zero-padded dummy modes for 2D mesh divisibility. Physical fields, after 3/2-rule oversampling, have shape `(ny_padded, nz_padded, nx_padded)` for periodic flows, `(ny, nz_padded, nx_padded)` for wall-bounded flows. Nyquist modes are omitted on all stored spectral axes. For multi-device: the device mesh has shape `(np0, np1)` with `np0 * np1 = np`. In spectral space, `kz` is sharded by `np0` and `kx` by `np1`; in physical space, `y` is sharded by `np0` and `z` by `np1`. When `np0 == 1` (default), this collapses to the original 1D scheme on `kx` / `z`. Two reshards happen inside `fft.py`: one Ns-way (`z ↔ kx`) and one Nr-way (`y ↔ kz`); when either mesh axis is 1, its reshard is skipped.

**Perturbation formulation**: The solver evolves the perturbation `u'` around the laminar base flow `U(y)`. The nonlinear term in `rhs.py` uses the rotational form of the perturbation equation: `NL = u' x omega' + u' x curl(U) + U x omega'`. The base-flow self-interaction `U x curl(U) = grad(|U|^2/2)` is omitted because it is a pure gradient absorbed by the pressure. All three cross-product contributions are computed per output component in a single fused `jnp.array` expression (`_fused_nonlinear`), eliminating intermediate concatenation and scatter kernels. Base flow fields (`base_flow`, `curl_base_flow`) are precomputed once in the flow dataclass constructor.

**JAX pytree registration**: Geometry base dataclasses (`TriplyPeriodicFlow`, `CartesianFlow`, `CylindricalFlow`) and their flow subclasses (`MonochromaticFlow(TriplyPeriodicFlow)`, `PlaneCouetteFlow(CartesianFlow)`, `PlanePoiseuilleFlow(CartesianFlow)`, `PipeFlow(CylindricalFlow)`), along with the geometry-specific `Fourier` classes and the solver dataclasses (`DenseJAXSolver`, `PerModeBandedOperator`), are registered as JAX pytrees via `register_dataclass_pytree()` in `sharding.py`, allowing them to be passed through `@jit` boundaries as static-like arguments.

### Parameter layering

Defaults (Pydantic models) -> `parameters.toml` -> CLI args. `update_parameters()` only applies explicitly-set fields, leaving unset fields at their current values.

### Configuration (`parameters.toml`)

Key sections: `[phys]` (re, system, oversampling_factor, oversample_y, driving: `"constant_pressure_gradient"` (default) or `"constant_bulk_velocity"` for pipe and plane-Poiseuille flows, block_mean_spanwise_velocity: `false` (default) -- zeroes mean-mode spanwise bulk velocity for Cartesian flows), `[geo]` (lx, lz, tilt_degree, wall_grid: optional path to a custom wall-normal grid file -- see wall_bounded/CLAUDE.md for file format and grid selection, grid_type: optional `"tanh"` or `"cgl"` for built-in grids, grid_stretch: stretching parameter for tanh grids, default 1.5), `[res]` (nx, ny, nz, fd_order, double_precision), `[init]` (start_from_laminar, snapshot, t0, it0), `[outs]` (it_stats, it_snapshot, nstats: stats-buffer flush size, default 100; stats_precision: significant digits in stats.dat, default 9; snapshot_layout: `"y_major"` (default) or `"native"` wall-bounded on-disk layout, periodic always native; snapshot_write_mode: `"concurrent"` (default) or `"serial"` rank-ordered multi-process writes for NFS-like filesystems), `[step]` (dt, implicitness, corrector_tolerance, max_corrector_iterations), `[stop]` (max_sim_time, max_wall_time as ISO 8601), `[dist]` (np, np0: wall-normal / kz mesh axis, default 1; np1: spanwise / kx mesh axis, default np // np0; platform), `[solver]` (backend: `"banded"` or `"dense"`, spike_block_size: optional target SPIKE block size `m`, block_thomas: `true` (default) -- use block-Thomas `lax.scan` solves for the SPIKE reduced system, `false` for the original batched cuSOLVER `lu_solve`). The default `parameters.toml` contains only `[phys] [geo] [res] [init] [outs] [step] [stop]`; `[dist]` and `[solver]` are valid sections that rely on model defaults -- set them via CLI (e.g. `--dist.np 2`, `--solver.backend dense`) or by adding the section.

### Diagnostics (`stats.dat`)

`get_stats` output is buffered on-device in a fixed `(nstats, n_cols)` array (one row every `it_stats` steps) and flushed to the `stats.dat` text file once `nstats` rows accumulate, with a final flush at shutdown (`_flush_stats` in `__main__.py`). Buffering avoids a host-device sync every `it_stats` steps -- only the periodic flush transfers to the host. `stats.dat` (written by the main device, appended) has a header row of column names (`t` plus the `get_stats` keys) and one whitespace-aligned row per sample at `stats_precision` significant digits.

### Snapshots

`snapshot.py` saves and loads spectral perturbation velocity in zarr3 format as **three combined per-component files** (one zarr3 chunk per velocity component), each a clean global array with kx de-interleaved across devices. Because every file holds the full kx range, a snapshot can be **resumed at any device count** (np-agnostic): on load, each current device reads only its own kx sub-range. `_dnsjax_meta.json` alongside the store embeds `t`, `it`, the on-disk `layout`, the global shapes, `wall_normal_grid` (the wall-normal grid points as a float array for wall-bounded flows), and the full `params.model_dump()` for resume validation. When the current wall-normal grid (default or custom) differs from the snapshot's grid (different `ny` or different point locations), the state is automatically interpolated in the wall-normal direction at load time (`_interpolate_if_needed` in `__main__.py`). CGL-to-CGL uses spectrally optimal Chebyshev coefficient truncation/extension; half-CGL-to-half-CGL uses parity-aware Chebyshev interpolation; general cases use barycentric Lagrange interpolation. Wall BCs are enforced after interpolation; residual divergence is projected out by the first corrector step.

On-disk layout per component is `D = (A, kx_global, B)`. Wall-bounded is selectable via `params.outs.snapshot_layout`: `"y_major"` (default) stores `(y, kx, kz)` so y is slowest, making a y-slice a contiguous read (`load_y_slice`); `"native"` stores `(kz, kx, y)` (zero-copy slab writes). Periodic flows ignore the option and always use `periodic_native` `(kz, kx, ky)`. **No full-array transpose is ever materialised** on save or load: the field is streamed one `(local_kx, len(B))` slab at a time, so peak extra memory is at most one slab (zero for `native`), never a second copy of the field. This was the motivation for the layout — the previous transpose-based scheme transiently doubled GPU memory.

I/O engine: data is written/read with **raw offset I/O** on both backends (TensorStore writes at chunk granularity, so per-device sub-range writes to a shared chunk would race / read-modify-write); TensorStore is used only to create the zarr3 metadata. kvikIO/GDS gives GPU-direct slab I/O when available (`kvikio`, `cupy`, optional, detected at runtime); the host fallback does device-to-host copy then `seek`/`write`. The 3 chunk files are pre-sized (`ftruncate`) by the main process before a barrier so concurrent disjoint-range writes (incl. multi-host) are safe.

Multi-process write ordering is set by `params.outs.snapshot_write_mode`: `"concurrent"` (default) lets every process write its disjoint byte ranges at once (POSIX/parallel filesystems); `"serial"` does rank-ordered (token-passing) writes via per-rank barriers (`_write_serialized`) so only one process holds a chunk file open at a time -- for NFS-like filesystems where concurrent writes can corrupt data, relying on close-to-open consistency. It is a no-op for single-process runs.

Loading from a zarr3 snapshot directory (detected by `Path.is_dir()`) overrides `params.init.t0` and `params.init.it0` from the snapshot metadata. Legacy `.npz` files (detected as plain files) still go through the geometry-specific `init_state` functions. The on-disk format is `format_version: 2`; version-1 snapshots are not readable.

### JAX-specific notes

- Explicit mode sharding is used globally rather than Auto mode, which propagates shardings on arrays for most operations. Do not use `jax.lax.with_sharding_constraint`.
- Avoid allocating a global array first and then distributing it with `jax.device_put` to devices after when such an array can be directly allocated on individual devices via the `out_sharding` argument for array-allocating calls like `jnp.zeros`, `ndarray.at.get(...)` and `ndarray.at.set(...)` etc. When this is not possible, do not use `jnp.asarray` just to avoid a `jax.device_put`.
- `jax_enable_x64` is set from `params.res.double_precision` before JAX initializes arrays.
- Buffer donation (`donate_argnums`) is used on main time-stepping functions to reuse memory.
- The first time step is excluded from benchmark statistics because it includes JIT compilation overhead.
- FFT normalization uses `norm="forward"` (divides by N on forward, no factor on inverse).

## Scripts
- `scripts/spike_partition_info.py`: display SPIKE block-partition trade-offs for a given resolution.
- `scripts/random_field.py`: generate a random divergence-free perturbation and save as a zarr3 snapshot. Supports all flow systems (Cartesian wall-bounded, cylindrical, triply-periodic). Uses `build_cartesian_grid` / `build_cylindrical_grid` from the geometry modules for grid/FD setup without constructing the full flow dataclass. Per-mode divergence-free enforcement uses NumPy loops (not JAX) to avoid tracing overhead; all other array work uses JAX. Run with `--test` for self-verification (divergence-free, wall BCs, norm, Hermitian symmetry, seed determinism).

## Tests
All to be kept up-to-date as the respective modules change:
- `tests/test_banded_solver.py` contains geometry-independent SPIKE solver tests.
- `tests/test_cartesian.py` contains Cartesian operator and matvec tests.
- `tests/test_cylindrical.py` contains cylindrical operator and matvec tests.
- `tests/test_integration.py` contains quadrature weight tests.
- `tests/test_laminar_smoke.py` runs all wall-bounded flows from laminar state (via subprocess/mpirun) checking stepping error and perturbation energy.
- `tests/test_snapshot.py` round-trips snapshots (save/load equality, np-agnostic resume, `load_y_slice`) for all on-disk layouts via the host I/O path (subprocess per system/device-count, multi-device via forced CPU devices).
