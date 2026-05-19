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

Keep docstrings, comments (in LaTeX for math for both) and typing up-to-date. In the future MkDocs will be used with MathJax, escape LaTeX commands appropriately. Keep documentation lines in code to 79 characters wide. Keep CLAUDE.md up-to-date.

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
  wall_bounded.py    # Shared wall-bounded infrastructure: integrate_scalar, get_inprod/get_norm2/get_norm, init_state, build_wall_bounded_stepper factory, phys_to_spec/spec_to_phys aliases
  triply_periodic.py # Fourier class, spectral diff ops (curl, div, grad, laplacian), norms, TriplyPeriodicFlow base dataclass, algebraic Helmholtz predict/correct, divergence correction, build_triply_periodic_stepper factory
  cartesian.py       # Fourier class, Clenshaw-Curtis weights, CartesianFlow base dataclass (with tilt and constant-bulk-velocity support), on-device IMM operator assembly (Lk/Hk builders for dense and banded backends), Kleiser-Schumann IMM iteration, build_cartesian_stepper factory
  cylindrical.py     # Fourier class, get_norm2_cyl, CylindricalFlow base dataclass, half-CGL radial grid, parity-reduced FD matrices, decoupled u+/u-/uz operators (Lk, Hk_plus, Hk_minus, Hk_z), 1x1 IMM, build_cylindrical_stepper factory
flows/
  monochromatic.py   # MonochromaticFlow(TriplyPeriodicFlow): base flow and forcing for Kolmogorov / Waleffe / decaying-box; diagnostics (E, I, D, E')
  plane_couette.py   # PlaneCouetteFlow(CartesianFlow): plane-Couette base flow U(y) = y with tilt; diagnostics
  plane_poiseuille.py # PlanePoiseuilleFlow(CartesianFlow): plane-Poiseuille base flow Us = 1-y^2 with tilt; diagnostics (E', dPds')
  pipe.py            # PipeFlow(CylindricalFlow): pipe base flow Uz = 1 - r^2; diagnostics
```

### Code-exploration constraints

Wall-bounded geometries (`cartesian.py`, `cylindrical.py`) and flows (`plane_couette.py`, `plane_poiseuille.py`, `pipe.py`) are completely independent from and design-wise mostly unrelated to the triply-periodic geometry (`triply_periodic.py`) and the respective monochromatic flows (`monochromatic.py`). In the development of future wall-bounded geometries and flows, do not explore the triply-periodic geometry and monochromatic flows, unless explicitly prompted to do so.

### Key design patterns

**Global singletons at module level**: `params`, `derived_params`, `padded_res` (from `parameters.py`), `sharding` (from `sharding.py`), and a geometry-specific `fourier` (from `geometries/triply_periodic.py`, `geometries/cartesian.py`, or `geometries/cylindrical.py`) are all instantiated at import time and mutated by `update_parameters()`. Every module imports and uses these directly. This means import order matters: JAX must be configured (`jax_enable_x64`, platform, distributed) *before* importing any module that uses `sharding` or a geometry module. The `__main__.py` enforces this by deferring `import jax` and flow-module imports until after configuration.

**Stepper factory pattern (two layers)**: `timestep.make_stepper()` is the shared core — it takes four geometry-general callables (`get_rhs_fn`, `predict_fn`, `correct_fn`, `norm_fn`) and returns JIT-compiled `predict_and_correct` / `iterate_correction` / `predict_and_fully_correct`, threading extra `*args` (typically `fourier` and a `flow` dataclass instance) through to the callables. `predict_and_fully_correct` fuses the predictor and all corrector iterations into a single JIT scope via `lax.while_loop`, eliminating per-iteration GPU-to-CPU synchronisation; it is the primary path used by `__main__`. For wall-bounded geometries, `build_wall_bounded_stepper()` in `geometries/wall_bounded.py` wraps `make_stepper` and binds the `fourier` and `flow` singletons into closures, returning `(predict_and_correct, iterate_correction, init_state_bound, predict_and_fully_correct)`. Each wall-bounded geometry module provides a thin `build_*_stepper(flow)` that passes its geometry-specific `_get_rhs` / `_predict` / `_correct` / `_norm` to `build_wall_bounded_stepper`. The triply-periodic geometry wraps `make_stepper` directly in `build_triply_periodic_stepper(flow)` and additionally returns `correct_velocity` (where the divergence-free constraint is enforced algebraically rather than by the IMM). Flow modules call the builder at module level to expose the public interface consumed by `__main__`.

**Spectral array layout**: Spectral fields have shape `(ny-1, nz-1, nx//2)` for periodic flows, `(nz-1, nx//2, ny)` for wall-bounded flows (where y stays in grid-point space). Physical fields, after 3/2-rule oversampling, have shape `(ny_padded, nz_padded, nx_padded)` for periodic flows, `(ny, nz_padded, nx_padded)` for wall-bounded flows. Nyquist modes are omitted on all stored spectral axes. For multi-device: spectral arrays are sharded on the last (triply-periodic) or second-to-last (wall-bounded) axis (kx for both), physical arrays on the second-to-last (z). The reshard between layouts happens inside `fft.py`.

**Perturbation formulation**: The solver evolves the perturbation `u'` around the laminar base flow `U(y)`. The nonlinear term in `rhs.py` uses the rotational form of the perturbation equation: `NL = u' x omega' + u' x curl(U) + U x omega'`. The base-flow self-interaction `U x curl(U) = grad(|U|^2/2)` is omitted because it is a pure gradient absorbed by the pressure. All three cross-product contributions are computed per output component in a single fused `jnp.array` expression (`_fused_nonlinear`), eliminating intermediate concatenation and scatter kernels. Base flow fields (`base_flow`, `curl_base_flow`) are precomputed once in the flow dataclass constructor.

**JAX pytree registration**: Geometry base dataclasses (`TriplyPeriodicFlow`, `CartesianFlow`, `CylindricalFlow`) and their flow subclasses (`MonochromaticFlow(TriplyPeriodicFlow)`, `PlaneCouetteFlow(CartesianFlow)`, `PlanePoiseuilleFlow(CartesianFlow)`, `PipeFlow(CylindricalFlow)`), along with the geometry-specific `Fourier` classes and the solver dataclasses (`DenseJAXSolver`, `PerModeBandedOperator`), are registered as JAX pytrees via `register_dataclass_pytree()` in `sharding.py`, allowing them to be passed through `@jit` boundaries as static-like arguments.

**Wall-bounded flows use the influence-matrix method (IMM)**:
- The pressure Poisson equation with preliminary Neumann BCs is solved via LU-factored matrices (`Lk`, `Hk`). The entire per-mode setup runs on the device: FD matrices `D1`/`D2` are built using JAX arrays with Python control flow (outside `@jit`) and distributed to devices once, after which `Lk` and `Hk` are assembled and factorised with no further host↔device traffic.
- All IMM homogeneous data (`p1, p2, v1, v2, q1, q2, M_inv`) is derived by `CartesianFlow._derive_imm_homogeneous_data` from the already-factored GPU operator.
- `params.solver.backend` selects the operator-factor storage format:
  - `"banded"` (default): SPIKE algorithm (Polizzi & Sameh 2006) partitions each banded `(Ny, Ny)` operator into `P` contiguous blocks of size `m = Ny/P` (with `m >= 2p`, `p = params.res.fd_order`) and factors each block as a dense `(m, m)` LU via cuSOLVER's batched LU (`jax.scipy.linalg.lu_factor`). Spike matrices `V_i = A_i^{-1} B_i`, `W_i = A_i^{-1} C_i` capture off-block coupling, and a small dense reduced system of size `2Pp` is also LU-factored once. At solve time (inside the JIT'd IMM iteration), per-block LU solves, a tiny reduced solve, and a spike reconstruction replace the old sequential `lax.scan` — all cuSOLVER-batched. Storage is `O(Ny·m)` per mode; no `(Nkz, Nkx, Ny, Ny)` array is ever materialised.
  - `"dense"`: builds the full `(Nkz, Nkx, Ny, Ny)` matrices on the GPU via `_build_Lk_dense_gpu`/`_build_Hk_dense_gpu`, LU-factors them on-device via `DenseJAXSolver`, then discards the originals — a reference path kept for parity with the banded backend.
- Solver infrastructure: geometry-independent code (`DenseJAXSolver`, `PerModeBandedOperator`, `_spike_factor`, `_choose_block_partition`, `_extract_banded_corners`) lives in `solvers.py`; `_build_Lk_blocks_gpu`/`_build_Hk_blocks_gpu` in `geometries/cartesian.py` assemble the per-block dense operators and coupling corners using those helpers.
- Both backends apply `Lk` and `Hk_minus` matvecs matrix-free via `_lk_matvec` / `_hk_minus_matvec`, reconstructing the operator action on the fly from the shared `D2` / `D1` FD matrices (no per-mode operator matrices are stored).
- IMM iteration: homogeneous solutions (`p1`, `p2`) and influence matrix `M_inv` find the correct pressure BC from the normal derivative of wall-normal velocity at the wall; pressure is then solved with that BC, and velocity is updated with the corresponding pressure gradient. Operator factors and homogeneous data inherit the kx-sharded layout from the broadcast against `fourier.k2`.
- Constant bulk velocity: with `params.phys.driving == "constant_bulk_velocity"`, `CartesianFlow._precompute_bulk_response` solves `Hk h = 1` (zero Dirichlet wall BCs) at the mean mode; after each IMM iteration, the mean-mode streamwise velocity is corrected by `G * h` where `G = -Ub_pert / H_bulk` and `H_bulk = dot(y_weights, h) / 2`.
- Block mean spanwise velocity: with `params.phys.block_mean_spanwise_velocity == True`, each IMM iteration additionally zeroes the perturbation bulk velocity in the spanwise direction `(-sin θ, 0, cos θ)`. Uses the same `h_bulk_response` / `H_bulk_inv` as the streamwise constant-bulk-velocity enforcement (the Helmholtz operator at the mean mode is identical for all horizontal velocity components). The two corrections are orthogonal and independent.
- Tilt: both Cartesian flows support tilted domains via `cos_tilt`/`sin_tilt` (from `derived_params.tilt_rad`), which rotate the streamwise direction in the (x, z) plane.

**Cylindrical geometry and decoupled velocity formulation**:
- The cylindrical Navier-Stokes vector Laplacian couples `u_r` and `u_theta` through `1/r^2` terms. Following Openpipeflow (Willis 2017), `geometries/cylindrical.py` decouples them via `u+ = u_r + i u_theta`, `u- = u_r - i u_theta`, reducing the vector problem to three scalar Helmholtz equations.
- Each component has an **effective azimuthal mode** `m_eff` that governs its scalar Laplacian structure (`D2 + (1/r)D1 - m_eff^2/r^2`): `m_eff = m+1` for `u+`, `m_eff = m-1` for `u-`, `m_eff = m` for `u_z`. Despite different `m_eff`, `u+` and `u-` share the **same parity** `(-1)^{m+1}` — parity is kinematic (how a field transforms under `r -> -r` on the auxiliary grid), while `m_eff` determines the operator spectrum.
- Radial grid: half-CGL on `(0, 1]` with `Nr = ny` points, formed by taking the positive half of a `2Nr`-point CGL grid on `[-1, 1]`. No grid point falls at `r = 0`; regularity is enforced by parity-reduced FD matrices built by mirroring the grid and folding ghost unknowns: `D_reduced = D_pos ± D_ghost_flipped`, where the sign depends on parity.
- Two base operators `A_base_even` and `A_base_odd` (`D2 + diag(1/r)*D1` with even/odd parity) differ only in the first ~p rows (near the centre).
- Three Helmholtz operators `Hk_plus`, `Hk_minus`, `Hk_z` are built per velocity component (with the appropriate `m_eff^2/r^2` diagonal shift), factored separately, then stacked into a single combined `Hk_op` with a leading component axis (order: plus, minus, z). Both `DenseJAXSolver` and `PerModeBandedOperator` support batched-operator dispatch: when the factor arrays have one extra leading dimension, `solve()` vmaps over both operator and RHS, issuing one batched kernel launch instead of three.
- The pipe has only one physical wall at `r = 1`, giving a `1x1` influence matrix (scalar `alpha` per mode) instead of the Cartesian `2x2`. Homogeneous data consists of 4 arrays (`p1`, `v_plus_1`, `v_minus_1`, `q_z_1`) plus scalar `M_inv`.
- SPIKE block construction reuses `solvers.py` with a parity-dependent first block: pre-built for both parities, selected per mode via `jnp.where`.
- Matrix-free matvecs decompose into a common part (`D_pos`) plus a parity-dependent ghost correction for the first ~p entries.
- Velocity ordering: state array stores `(u_z, u+, u-)`, matching the Cartesian convention of (streamwise, wall-normal, spanwise); the physical representation follows the same convention as `(u_z, u_r, u_theta)`. `_get_rhs` converts between the two for the nonlinear term, and `_curl_fn` implements the cylindrical curl in spectral space.
- Constant bulk velocity: with `params.phys.driving == "constant_bulk_velocity"`, each IMM iteration adds a uniform mean pressure gradient `G` to the mean-mode `u_z` Helmholtz RHS (via a Helmholtz-consistent post-solve correction `uz += G * h`, where `h = Hk_z^{-1} [1,...,1,0]` is precomputed) to enforce zero perturbation bulk velocity; `G = -Ub_pert / H_bulk` where `H_bulk = 2 int_0^1 h r dr`.

### Parameter layering

Defaults (Pydantic models) -> `parameters.toml` -> CLI args. `update_parameters()` only applies explicitly-set fields, leaving unset fields at their current values.

### Configuration (`parameters.toml`)

Key sections: `[phys]` (re, system, oversampling_factor, oversample_y, driving: `"constant_pressure_gradient"` (default) or `"constant_bulk_velocity"` for pipe and plane-Poiseuille flows, block_mean_spanwise_velocity: `false` (default) — zeroes mean-mode spanwise bulk velocity for Cartesian flows), `[geo]` (lx, lz, tilt_degree), `[res]` (nx, ny, nz, fd_order, double_precision), `[step]` (dt, implicitness, corrector_tolerance), `[stop]` (max_sim_time, max_wall_time as ISO 8601), `[dist]` (np, platform), `[solver]` (backend: `"banded"` or `"dense"`, spike_block_size: optional target SPIKE block size `m`).

### JAX-specific notes

- Explicit mode sharding is used globally rather than Auto mode, which propagates shardings on arrays for most operations. Do not use `jax.lax.with_sharding_constraint`.
- Avoid allocating a global array first and then distributing it with `jax.device_put` to devices after when such an array can be directly allocated on individual devices via the `out_sharding` argument for array-allocating calls like `jnp.zeros`, `ndarray.at.get(...)` and `ndarray.at.set(...)` etc. When this is not possible, do not use `jnp.asarray` just to avoid a `jax.device_put`.
- `jax_enable_x64` is set from `params.res.double_precision` before JAX initializes arrays.
- Buffer donation (`donate_argnums`) is used on main time-stepping functions to reuse memory.
- The first time step is excluded from benchmark statistics because it includes JIT compilation overhead.
- FFT normalization uses `norm="forward"` (divides by N on forward, no factor on inverse).

### Common optimization patterns
- When the aim is to operate on a quantity derivable from the mean mode (streamwise *and* spanwise wavenumber equal to zero), first index to the mean mode, and then operate, when this indexing and the desired operations commute. You can use the function `extract_mean_mode` for this purpose in wall-bounded geometries.

## Tests
All to be kept up-to-date as the respective modules change:
- `tests/test_banded_solver.py` contains geometry-independent SPIKE solver tests.
- `tests/test_cartesian.py` contains Cartesian operator and matvec tests.
- `tests/test_cylindrical.py` contains cylindrical operator and matvec tests.
- `tests/test_integration.py` contains quadrature weight tests.
- `tests/test_laminar_smoke.py` runs all wall-bounded flows from laminar state (via subprocess/mpirun) checking stepping error and perturbation energy.
