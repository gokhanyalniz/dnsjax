## Project Overview

`dnsjax` is a GPU-accelerated pseudo-spectral + finite-differences DNS solver for the 3D incompressible Navier-Stokes equations, written in JAX. It targets triply-periodic flows (Kolmogorov, Waleffe, decaying-box) and wall-bounded flows (plane-Couette, plane-Poiseuille, pipe). The solver uses a predictor-corrector time integration scheme (Euler + iterative Crank-Nicolson, following Willis 2017 / openpipeflow).

## Commands

### Prerequisites

Python >=3.14, `uv`, MPI (for multi-device runs).

### Setup

`uv sync`

### Lint

`uv run ruff check --fix`

Line length is 79 for **all** lines (ruff `line-length = 79`, E501), not only docstrings/comments.

### Run tests

All tests are standalone scripts run directly (`uv run python tests/test_*.py`) -- there is no `pytest`/CI runner, and they rely on `__main__` setup plus shared module-level singletons (so `pytest` collection misbehaves).

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

Generates a divergence-free random perturbation (obeying BCs) and saves it as a zarr3 snapshot. Load with `--init.snapshot random_ic --init.start_from_laminar False`. Supports all flow systems. For Dean (`--system dean`, total-field) the analytical laminar profile is added to the perturbation. Run `--test` for self-verification: it checks the configured system's generator and exits with a pass/fail status, writing no snapshot (so `--output` is not needed in this mode). See `scripts/random_field.py` docstring for the full algorithm and CLI options.

The generators live in `dnsjax.random_field` (`generate_random_state`). To start a run from a random IC **without** a snapshot file (no disk round-trip), use the in-process start mode `--init.random_field True` (with `--init.random_amplitude` / `--init.random_smoothness` / `--init.random_seed` / `--init.random_mean_flow`) instead of this script. This is what `tests/test_random_smoke.py` drives.

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
random_field.py       Random divergence-free IC generators per geometry family + generate_random_state dispatch; shared by scripts/random_field.py and the in-process init.random_field start mode
geometries/
  wall_bounded/       Wall-bounded geometry family (see wall_bounded/CLAUDE.md)
    _base.py          Shared wall-bounded infrastructure (norms, init_state, stepper builder)
    cartesian.py      Cartesian: Fourier, CGL grid, CartesianFlow, IMM, Lk/Hk operators
    cylindrical.py    Cylindrical: Fourier, half-CGL grid, CylindricalFlow, decoupled u+/u-/uz, 1x1 IMM
    annular.py        Annular (concentric cylinders): Fourier, CGL grid on [r1,r2], AnnularFlow, decoupled u+/u-/uz (no parity/ghost, no r=0), 2x2 IMM, optional mean-mode azimuthal body force (pi_theta); dean_laminar_u_theta
  triply_periodic/    Triply-periodic geometry family (see triply_periodic/CLAUDE.md)
    triply_periodic.py  Fourier, spectral diff ops, TriplyPeriodicFlow, algebraic Helmholtz, divergence correction
flows/
  wall_bounded/
    plane_couette.py    PlaneCouetteFlow(CartesianFlow): U(y) = y with tilt
    plane_poiseuille.py PlanePoiseuilleFlow(CartesianFlow): Us = 1-y^2 with tilt
    pipe.py             PipeFlow(CylindricalFlow): Uz = 1 - r^2
    taylor_couette.py   TaylorCouetteFlow(AnnularFlow): circular-Couette Uθ = A0 r + B0/r (shear-driven, no pressure gradient)
    dean.py             DeanFlow(AnnularFlow): force-driven Dean flow between stationary cylinders (azimuthal body force; integrates the TOTAL field, no base flow / no E')
  triply_periodic/
    monochromatic.py    MonochromaticFlow(TriplyPeriodicFlow): Kolmogorov / Waleffe / decaying-box
```

### Code-exploration constraints

The two geometry families are **completely independent**. The directory structure enforces this:

- `geometries/wall_bounded/` and `flows/wall_bounded/` are unrelated to `geometries/triply_periodic/` and `flows/triply_periodic/`. Do not explore across families unless explicitly prompted.
- Wall-bounded family documentation: `src/dnsjax/geometries/wall_bounded/CLAUDE.md`
- Triply-periodic family documentation: `src/dnsjax/geometries/triply_periodic/CLAUDE.md`

### Adding a flow system

To add a flow `X`: (1) add `"X"` to the relevant `*_systems` list in `parameters.py` (this auto-extends the `phys.system` Literal); (2) add/extend the geometry branch in `update_parameters()` if it needs derived params; (3) create `flows/<family>/X.py` exporting `predict_and_fully_correct`, `predict_and_fully_correct_measured`, `init_state`, `get_stats` (periodic flows also export `correct_velocity`); (4) add an `elif` to the flow dispatch in `__main__.py`; (5) add a `tests/test_laminar_smoke.py` SYSTEMS entry (the smoke test parses `err=`/`E'=` from stdout — a flow without a perturbation energy `E'` (e.g. total-field) needs its own check branch) and a `tests/test_random_smoke.py` SYSTEMS entry (random-IC integration: pick a Reynolds number above transition onset and a small domain).

### Key design patterns

**Global singletons and import order**: `params`, `derived_params`, `padded_res` (from `parameters.py`), `sharding` (from `sharding.py`), and a geometry-specific `fourier` are instantiated at import time and mutated by `update_parameters()`. JAX must be configured (`jax_enable_x64`, platform, distributed) *before* importing any module that uses `sharding` or a geometry module. See `__main__.py` module docstring. A module that must stay importable *before* JAX is configured but still uses JAX internally (e.g. `random_field.py`) keeps `import jax` out of module scope: lazy imports inside functions, and `from jax import Array` under `if TYPE_CHECKING:` (annotations are stringised by `from __future__ import annotations`).

**Stepper factory (two layers)**: `timestep.make_stepper()` takes four geometry-general callables and returns JIT-compiled stepping functions, including `predict_and_fully_correct` (fused corrector loop via `lax.while_loop`, the primary path). Each geometry family wraps it in its own builder that binds the `fourier` and `flow` singletons. See `timestep.py`, `_base.py`, and `triply_periodic.py` docstrings.

**Corrector convergence is `dt`-limited, not CFL-limited**: the iterative Crank-Nicolson corrector's contraction rate scales with `step.dt`. A `corrector failed to converge` at *low* CFL with the final error only marginally above `step.corrector_tolerance` means the step is too large to contract within `max_corrector_iterations` — reduce `dt` (or raise the iteration cap); it is not an advective-CFL / blow-up. (Random-IC Kolmogorov stalls at ~1.4e-5 at `dt=0.01` but converges in one corrector step at `dt=0.005`; the wall-bounded flows are fine at 0.01.)

**Spectral array layout and sharding**: see `sharding.py` module docstring for shapes, partition specs, and the `(np0, np1)` device mesh. See `fft.py` module docstring for the reshard pipeline and spectral padding.

**Perturbation formulation**: the solver evolves `u'` around laminar `U(y)`. The rotational-form nonlinear term and base-flow gradient elimination are documented in the `rhs.py` module docstring.

**Moving frame of reference**: `phys.u_grid` translates the wall-bounded frame along the grid direction (`∂_t → ∂_t − U_grid ∂_grid`). It is treated *explicitly* by advecting with `U − U_grid ê₀` (physical component 0) in both the rotational nonlinear term and the CFL diagnostic — the precomputed `base_flow_adv_padded` (see `geometries/wall_bounded/_base.py` `pad_base_flow`, which documents the rotational-form identity). Lowers the advective CFL and de-advects snapshots; the stored `u'` and all stats are frame-invariant.

**JAX pytree registration**: `register_dataclass_pytree()` in `sharding.py` registers geometry dataclasses, flow subclasses, solver classes, and Fourier classes as JAX pytrees. See its docstring for details.

### Parameter layering

Lowest priority first: defaults (Pydantic models) -> parameters embedded in a resumed snapshot -> `parameters.toml` -> CLI args. `update_parameters()` only applies explicitly-set, non-`None` fields, leaving unset fields at their current values; `validate_parameters()` is called once after the final layer for cross-field checks. Snapshot params are read by `read_snapshot_params()` from the snapshot's `_dnsjax_meta.json` -- but the JAX-setup fields `dist.np0`/`np1`/`platform` and `res.double_precision` are *not* inherited (resume is device-/precision-agnostic). `__main__.py` resolves the resume snapshot (explicit CLI `init.snapshot` over TOML) and applies the layers in order before JAX/singleton setup.

### Configuration (`parameters.toml`)

See `parameters.py` classes for full documentation. Key sections:

| Section    | Key fields                                                                                             |
|------------|--------------------------------------------------------------------------------------------------------|
| `[phys]`   | `re`, `re1`/`re2` (Taylor-Couette inner/outer cylinder Reynolds numbers; derive `re := Re_ref`), `system`, `oversampling_factor`, `oversample_y`, `driving` (`"constant_pressure_gradient"` / `"constant_bulk_velocity"`), `block_mean_spanwise_velocity` (Taylor-Couette: blocks the mean axial velocity), `u_grid` (moving-frame speed along the grid direction -- streamwise `x` for Cartesian, axial `z` for cyl/annular; `None` -> laminar bulk: 1/2 pipe, 2/3 plane-Poiseuille, 0 otherwise; resolved onto `derived_params.u_grid`) |
| `[geo]`    | `lx`, `lz`, `tilt_degree`, `eta` (Taylor-Couette radius ratio r1/r2), `wall_grid` (custom grid file), `grid_type` (`"tanh"` / `"cgl"`), `grid_stretch` |
| `[res]`    | `nx`, `ny`, `nz`, `fd_order`, `double_precision`                                                      |
| `[init]`   | `start_from_laminar`, `snapshot`, `t0`, `it0`, `random_field` (in-process random IC; `random_amplitude`/`random_smoothness`/`random_seed`/`random_mean_flow`) |
| `[outs]`   | `it_stats`, `it_steps` (CFL diagnostic cadence -> `steps.dat`), `it_snapshot`, `it_corrector` (corrector diagnostic cadence -> `corrector.dat`; requires `it_error_check <= it_corrector`), `it_error_check` (host-sync cadence for corrector convergence), `nbuffer`, `stats_precision`, `snapshot_write_mode` (`"concurrent"` / `"serial"`) |
| `[step]`   | `dt`, `implicitness`, `corrector_tolerance`, `max_corrector_iterations`                                |
| `[stop]`   | `max_sim_time`, `max_wall_time` (ISO 8601)                                                            |
| `[dist]`   | `np0` (wall-normal / kz axis), `np1` (spanwise / kx axis), `platform`                                 |
| `[solver]` | `backend` (`"banded"` / `"dense"`), `spike_block_size`, `block_thomas`                                 |

The default `parameters.toml` contains only `[phys] [geo] [res] [init] [outs] [step] [stop]`; `[dist]` and `[solver]` rely on model defaults -- set them via CLI (e.g. `--dist.np1 2`, `--solver.backend dense`) or by adding the section.

### Diagnostics (`stats.dat`, `steps.dat`, `corrector.dat`)

On-device buffered stats, flushed periodically to `stats.dat`. The CFL diagnostic (every `outs.it_steps` steps, measured from physical-space velocity inside the nonlinear-term evaluation -- see `measurements.py` and the `rhs.py` `measure_fn` hook) is buffered and flushed to `steps.dat` the same way. The corrector diagnostic (every `outs.it_corrector` steps: iteration count `c` and final error, both already returned by every step) is buffered and flushed to `corrector.dat` the same way. Every flush is `fsync`-ed, so rows reach disk immediately once the on-device buffer flushes (`_flush_stats`, shared by all three streams). See `__main__.py` module docstring for the buffering mechanism and file format.

### Snapshots

Zarr3 format with 3 combined per-component files (np-agnostic resume at any `(np0, np1)` configuration). The stored state is the spectral **perturbation** `u'` only — the base flow lives in `flow.base_flow`, not the state, so a laminar snapshot is a zero array. `_dnsjax_meta.json` stores simulation time, iteration, layout, grid, and full params. When the wall-normal grid differs from the snapshot's, the state is interpolated at load time (`_interpolate_if_needed` in `__main__.py`; interpolation methods in `fd.py`). See `snapshot.py` module docstring for on-disk layouts, I/O engines, memory, and write modes.

### JAX-specific notes

- Explicit mode sharding is used globally rather than Auto mode, which propagates shardings on arrays for most operations. Do not use `jax.lax.with_sharding_constraint`.
- Avoid allocating a global array first and then distributing it with `jax.device_put` to devices after when such an array can be directly allocated on individual devices via the `out_sharding` argument for array-allocating calls like `jnp.zeros`, `ndarray.at.get(...)` and `ndarray.at.set(...)` etc. When this is not possible, do not use `jnp.asarray` just to avoid a `jax.device_put`.
- `jax_enable_x64` is set from `params.res.double_precision` before JAX initializes arrays.
- JAX has no zero-copy complex<->real bitcast (`lax.bitcast_convert_type` rejects complex; `.view()` lowers to scatter). Real-operator x complex-field GEMMs/solves use an explicit trailing re/im split at half the promoted-complex FLOPs — reuse `apply_y_matrix` (`geometries/wall_bounded/_base.py`) or the `solvers.py` pattern.
- Buffer donation (`donate_argnums`) is used on main time-stepping functions to reuse memory.
- The first time step is excluded from benchmark statistics because it includes JIT compilation overhead.
- FFT normalization uses `norm="forward"` (divides by N on forward, no factor on inverse).
- Dicts returned from jitted functions (`get_stats`, measurement dicts) are canonicalized to **sorted key order** by pytree flattening — this sets the column order of `stats.dat` / `steps.dat`; never assume insertion order.
- A flow dataclass is a registered pytree, so every array field is traced into the jitted steppers; keep data needed only *outside* jit (e.g. a precomputed initial/laminar state) at module level, not as a flow field.

## Scripts
- `scripts/spike_partition_info.py`: display SPIKE block-partition trade-offs for a given resolution.
- `scripts/random_field.py`: thin CLI wrapper over `dnsjax.random_field` (which holds the generators, shared with the in-process `init.random_field` start mode); generates a random divergence-free perturbation and saves it as a zarr3 snapshot. Supports all flow systems (Cartesian wall-bounded, cylindrical, annular, triply-periodic). Uses `build_cartesian_grid` / `build_cylindrical_grid` / `build_annular_grid` from the geometry modules for grid/FD setup without constructing the full flow dataclass. Taylor-Couette needs `--re1 --re2 --eta`; Dean needs `--eta` (and `--re`) and adds the analytical laminar profile (`dean_laminar_u_theta`) to the perturbation to form the total-field IC. Per-mode divergence-free enforcement uses NumPy loops (not JAX) to avoid tracing overhead; all other array work uses JAX. Run with `--test` for self-verification (divergence-free, wall BCs, norm, Hermitian symmetry, seed determinism, and for Dean the total-field wall BCs); the self-test runs the block for the configured `--system` (cartesian, cylindrical, or annular).
- `scripts/snapshot_import.py`: **library** (not a CLI) for converting an external simulator's velocity field (physical- or spectral-space, no dealiasing padding, layout `[component, streamwise, wall-normal, spanwise]`) into a dnsjax zarr3 snapshot, for import into future per-simulator CLIs. Public API: `configure_target` (JAX/params singleton setup, one system per process like `random_field.py`), `to_spectral_state`, `write_snapshot`, `convert_field_to_snapshot`, `validate_state`. Stores the field on the supplied wall-normal grid (recorded in metadata, interpolated at load). **Perturbation only** — no base-flow subtraction; input must already be a perturbation. For pipe/TC it forms `u_± = u_r ± i u_θ` and follows dnsjax's axial→`nx`/azimuthal→`nz` mapping (so for Taylor-Couette streamwise resolution is `nz`, spanwise `nx`). See the module docstring for the per-system layout tables, the FFT/normalization algorithm, and the `real_axis`/`input_norm` options for spectral input.

## Tests
All to be kept up-to-date as the respective modules change:
- `tests/test_banded_solver.py` contains geometry-independent SPIKE solver tests.
- `tests/test_cartesian.py` contains Cartesian operator and matvec tests.
- `tests/test_cylindrical.py` contains cylindrical operator and matvec tests.
- `tests/test_annular.py` contains annular (Taylor-Couette) operator/matvec tests, the 2x2 SPIKE-vs-dense parity, and circular-Couette coefficient (A0/B0) checks.
- `tests/test_integration.py` contains quadrature weight tests.
- `tests/test_mean_mask.py` checks that padding slots carry nonzero placeholder wavenumbers and `Fourier.mean_mask` is the unique k^2 = 0 (mean) mode under forced spectral padding (subprocess, forced CPU devices).
- `tests/test_laminar_smoke.py` runs all wall-bounded flows from laminar state (via subprocess/mpirun) checking stepping error, perturbation energy, the `steps.dat` CFL columns against analytic laminar values, and the `corrector.dat` columns (`c = 0`, roundoff-sized error). Each subprocess runs in a temp dir: `parameters.toml` is not loaded (model defaults + CLI args only). Caveat: the laminar state has `u'=0`, so all `ω'`/`u'`-proportional terms vanish — this checks the base-flow fixed point, time-stepping, and CFL diagnostic but **not** the rotational nonlinear term (a wrong `rhs.py`/advection change can still report `err=0`). Validate such changes with a non-laminar run (`tests/test_random_smoke.py`, or a manual `scripts/random_field.py` IC), comparing a transform-invariant diagnostic (e.g. `E'`) across configs and confirming convergence as `dt → 0`. **Dean** is checked differently (total-field, no `E'`): started from the *analytical* laminar profile (only a near-fixed-point on the FD grid), it verifies the deviation `dU` from that profile stays tiny, the corrector converges, the energy balance `I ≈ D` holds, and the energy is near-steady; its azimuthal `CFL_th` is the active column. (Note: the FD enstrophy diagnostic `D` underestimates the true dissipation for under-resolved rough fields, so the `dE/dt = I − D` budget is exact only for resolved/smooth fields — confirmed by convergence as the field smooths.)
- `tests/test_random_smoke.py` exercises time integration for all 6 flows from a random divergence-free IC (the in-process `--init.random_field` start mode, no snapshot), at a Reynolds number above transition onset on a small domain (default 32³, 5×5×5, to `t = 1`). Checks each run exits 0, reaches the end (no early corrector divergence), and ends finite and converged (`err < corrector_tolerance`). Complements the laminar smoke test by driving the **nonlinear** path; transition is not expected to develop by `t = 1` (success = clean integration, not turbulence). Subprocess per system in a temp dir; CLI knobs `--np`/`--res`/`--dt`/`--max-sim-time`/`--systems`. Kolmogorov is dt-capped at 0.005 (corrector-rate limit, not advective CFL; see the file's `SYSTEMS`).
- `tests/test_snapshot.py` round-trips snapshots (save/load equality, np-agnostic resume, `load_y_slice`) for all on-disk layouts via the host I/O path (subprocess per system/device-count, multi-device via forced CPU devices).
- `tests/test_snapshot_import.py` validates `scripts/snapshot_import.py` (subprocess per geometry family): single-mode placement (axis mapping, component basis incl. the TC axial/azimuthal swap, normalization), mode order vs the `fourier` singleton, `u_±` mixing, spectral-input round-trips (both `real_axis`, several `input_norm`), and snapshot save/load. Offline (no `mpirun`).
- In-process geometry tests (`test_{cartesian,cylindrical,annular}.py`) build their `flow`/`fourier` singletons **once** from a module-top `update_parameters()`; a test that re-calls `update_parameters()` mutates the shared `params`/`derived_params` (the singletons keep the import-time config, so read a re-derived value off `derived_params`, not `flow.*`) and must restore the module config before returning (tests run in definition order, so an unrestored mutation leaks into later tests).
- Any test exercising `taylor-couette` must set `params.phys.re1`/`re2` and `params.geo.eta` (suite-standard `100`/`0`/`0.5`) before resolution derivation / singleton construction — all three default to `None` and the TC branch of `update_parameters()` raises otherwise.
