## Project Overview

`dnsjax` is a GPU-accelerated pseudo-spectral + finite-differences DNS
solver for the 3D incompressible Navier-Stokes equations, written in
JAX. Flow systems: triply-periodic (Kolmogorov, Waleffe, decaying-box)
and wall-bounded (plane-Couette, plane-Poiseuille, pipe,
Taylor-Couette, force-driven Dean, and viscoelastic (sPTT) Dean with a
coupled conformation tensor). Two selectable second-order
time-integration schemes (`step.scheme`): the default
predictor-corrector `"iterative-cn"` (Euler + iterative Crank-Nicolson,
following Willis 2017 / openpipeflow) and `"cnab2"` (Crank-Nicolson
viscous + explicit Adams-Bashforth nonlinear, one FFT evaluation per
step); see the "Time-stepping scheme" note below.

## Commands

### Prerequisites

Python >=3.14, `uv`, MPI (for multi-device runs).

### Setup

`uv sync`

### Lint

`uv run ruff check --fix`

Line length is 79 for **all** lines (ruff `line-length = 79`, E501),
not only docstrings/comments.

### Run tests

All tests are standalone scripts run directly
(`uv run python tests/test_*.py`) -- there is no `pytest`/CI runner,
and they rely on `__main__` setup plus shared module-level singletons
(so `pytest` collection misbehaves).

Two ways to get multiple devices, and they do **not** mix:
offline/in-process tests (import modules directly) force CPU devices
via `XLA_FLAGS=--xla_force_host_platform_device_count=N` + set
`params.dist.np0`/`np1` before importing `sharding` (the
`test_snapshot.py` / `test_localized_rolls.py` pattern, no MPI); a
`python -m dnsjax` run needs real `mpirun -np N` (its distributed init
gives 1 device/process, so `mpirun -np 1` + forced device count fails
with "# of devices visible (1) != np0*np1"). Default-suite
multi-device subprocess entries use `mpirun --oversubscribe -np 2` for
portability on few cores.

Single file: `uv run python tests/test_cartesian.py`
Laminar smoke (1D multi-device):
`uv run python tests/test_laminar_smoke.py --np 2`
Laminar smoke (2D multi-device):
`uv run python tests/test_laminar_smoke.py --np 4 --np0 2`

### Smoke test (laminar time stepping)

Any `python -m dnsjax` run must be launched via `mpirun` (even
single-process: `mpirun -np 1 ...`); `__main__` unconditionally
initializes the JAX distributed backend. Under `mpirun`, invoke the
interpreter as `.venv/bin/python` directly (`uv run` does not compose
with `mpirun`). A run writes `stats.dat`/`steps.dat` (and any
snapshots) to the cwd, so launch manual smoke/debug runs from a
scratch dir, using the absolute path to the repo's `.venv/bin/python`.

`mpirun -np 2 python -m dnsjax --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 27`

For double parallelisation (tanh grid recommended for clean ny
divisibility):

`mpirun -np 4 python -m dnsjax --dist.np0 2 --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 28 --geo.grid_type tanh`

The laminar state should time step with a single corrector step, with
stepping error of O(-18) or less, and perturbation energy of O(-32) or
less. Run existing tests listed in section Tests below if you touch
the modules they test.

### Generate random initial condition

In-process only (no offline script): `--init.random_field True` is the
**default** start mode when no snapshot is given (with
`--init.random_amplitude` / `--init.random_smoothness` /
`--init.random_seed` / `--init.random_mean_flow`; viscoelastic-dean
adds `--init.random_conformation_amplitude`). All flow systems; the
total-field dean/viscoelastic-dean add the analytical laminar profile
(+ the sPTT-equilibrium conformation). Generators live in
`dnsjax.random_field` (`generate_random_state`): per-device build (no
replication), device-count-independent, padded-mesh-safe -- see the
module docstring. Two validation caveats (detail in the
`random_field.py` docstrings): the real-FFT-axis DC plane is left
non-divergence-free for the first corrector step (exclude it when
validating a divergence operator; Cartesian is fully div-free), and
the cyl/annular `k_z=0` reality constraint is on `u_r`/`u_θ` -- `u_±`
are conjugate *partners*, not individually Hermitian
(`random_field._hermitian_column`, `pm_pair`).

### Generate localized-rolls ("turbulent spot") initial condition

In-process only (no offline script): `--init.localized_rolls True`
(with `--init.localized_rolls_amplitude` / `_width` / `_wavelength`);
wall-bounded systems only, higher precedence than the default random
field. A compact fixed-physical structure, localized in every
homogeneous direction and peak-normalized so `max|u'| = amplitude` at
any box size; the total-field dean/viscoelastic-dean add the
analytical laminar profile (+ conformation). Generators live in
`dnsjax.localized_rolls` (`generate_localized_rolls`); construction,
per-geometry pairing, and the sharded separable build are in its
module docstring.

### Triggering transition to turbulence

The solver reproduces the linear physics **quantitatively**
(Orr-Sommerfeld growth rates, lift-up streak growth), so a flow that
decays is a *regime/time* matter, **not** a solver bug. Transition
develops over O(100) advective time units (the smoke tests only reach
`t=1`); it needs `Re` above the sustainment threshold (plane-Couette
≳ 350-500), a domain ≳ the minimal flow unit, and a finite-amplitude
perturbation (random `amplitude ≈ 0.1-0.2`, or a localized spot strong
enough to break down). Near-minimal boxes give transient (decaying)
turbulence; robust sustainment needs a larger box / higher `Re`.

## Documentation instructions

Keep docstrings, comments (in LaTeX for math for both) and typing
up-to-date. In the future MkDocs will be used with MathJax, escape
LaTeX commands appropriately (prefer raw docstrings: `\t`/`\f` in
non-raw strings silently become control characters). Keep
documentation lines in code to 79 characters wide. Keep CLAUDE.md
files up-to-date (root and subdirectory files). `README.md` is
human-facing and may lag the code, so treat the CLAUDE.md files and
code docstrings as authoritative and do not sync code to the README.

**Documentation layering**: detailed descriptions of algorithms, array
shapes, mathematical formulations, and per-function behaviour belong
in code docstrings and comments. CLAUDE.md files serve as a concise
index for AI agents: structural overview, cross-cutting constraints,
copy-paste commands, and pointers to the relevant code. When adding
functionality, put the detail in the code and add a brief entry or
pointer in the appropriate CLAUDE.md only if it introduces a new
module, a cross-cutting pattern, or a non-obvious constraint that
isn't discoverable from a single file's docstrings. Tests follow the
same rule: one line per test file in the Tests section below; the
description of what a test covers lives in that test's module
docstring.

## Architecture

### Package layout (`src/dnsjax/`)

Per-module detail lives in each module's docstring; one line each here.

```
__main__.py           Entry point: import-order enforcement, stats
                      buffering, snapshot resume/lineage
parameters.py         Pydantic parameter models; singletons params,
                      derived_params, padded_res;
                      trajectory_defining_changes
sharding.py           Multi-device (np0, np1) mesh; singleton sharding;
                      register_dataclass_pytree; layouts + specs
operators.py          Wavenumber helpers (re-exports harmonics.py in
                      jnp.asarray; pad_harmonics), FFT wrappers,
                      cross product
harmonics.py          Stdlib/NumPy-only (JAX-free) wavenumber
                      sequences; leaf shared with dnsjax.analysis
fft.py                3D/2D real FFT, 3/2-rule dealiasing, shard_map
                      reshard pipeline, spectral padding
rhs.py                Rotational-form perturbation nonlinear term;
                      measure_fn hook
measurements.py       Physical-space measurements (get_cfl)
timestep.py           make_stepper() factory:
                      predict_and_fully_correct (+_measured),
                      step_cnab2 (+_measured), _cnab2_lbf_core
fd.py                 NumPy-only FD utilities (JAX-free): Fornberg
                      D1/D2, quadrature rules, interpolation matrices,
                      tanh grids
solvers.py            Geometry-independent solvers: DenseJAXSolver,
                      PerModeBandedOperator (SPIKE),
                      PerModeBandedPallasOperator (mode-tiled Triton
                      banded sweep on GPU, pure-JAX sweep on CPU;
                      no-pivot LU + pivoted-SPIKE fallback; mode-inner
                      .solve contract, a shard_map-local region with
                      per-shard tile-padded factors; shared
                      banded-assembly helpers used by every geometry)
                      -- see _pallas_banded_solve / .solve docstrings
snapshot.py           Single-file (tar/zarr3) snapshot save/load, raw
                      offset I/O (GDS or host); assemble_local_shards
snapshot_meta.py      Stdlib-only (JAX-free) snapshot tar metadata
                      helpers
random_field.py       Random divergence-free IC generators
                      (init.random_field, the default start mode)
localized_rolls.py    Deterministic localized-spot IC generators
                      (init.localized_rolls)
geometries/
  wall_bounded/       _base.py, cartesian.py, cylindrical.py,
                      annular.py, annular_viscoelastic.py -- see
                      wall_bounded/CLAUDE.md
  triply_periodic/    triply_periodic.py -- see its CLAUDE.md
flows/
  wall_bounded/       plane_couette, plane_poiseuille, pipe,
                      taylor_couette, dean, viscoelastic_dean --
                      base flows/driving in wall_bounded/CLAUDE.md
  triply_periodic/    monochromatic.py: Kolmogorov/Waleffe/decaying-box
analysis/             External-facing JAX-free snapshot post-processing
                      API -- see analysis/CLAUDE.md
```

### Code-exploration constraints

The two geometry families are **completely independent**. The
directory structure enforces this:

- `geometries/wall_bounded/` and `flows/wall_bounded/` are unrelated to
  `geometries/triply_periodic/` and `flows/triply_periodic/`. Do not
  explore across families unless explicitly prompted.
- Wall-bounded family documentation:
  `src/dnsjax/geometries/wall_bounded/CLAUDE.md`
- Triply-periodic family documentation:
  `src/dnsjax/geometries/triply_periodic/CLAUDE.md`

### Adding a flow system

To add a flow `X`: (1) add `"X"` to the relevant `*_systems` list in
`parameters.py` (this auto-extends the `phys.system` Literal); (2)
add/extend the geometry branch in `update_parameters()` if it needs
derived params; (3) create `flows/<family>/X.py` exporting
`predict_and_fully_correct`, `predict_and_fully_correct_measured`,
`step_cnab2`, `step_cnab2_measured`, `init_state`, `get_stats`,
`get_perturbation_energy` (the cheap `E'` read for the laminarization
check); (4) add an
`elif` to the flow dispatch in `__main__.py`; (5) add SYSTEMS entries
to `tests/test_laminar_smoke.py` (a flow without a perturbation `E'`
needs its own check branch) and `tests/test_random_smoke.py` (pick a
Reynolds number above transition onset, small domain); (6) add `"X"`
to the matching `*_SYSTEMS` frozenset in `analysis/_core.py` (unknown
systems raise there). A flow whose state is not the 3 velocity
components (e.g. the 9-component viscoelastic state) also drives the
component count in `snapshot.py`/`snapshot_meta.py` (`_n_components`,
metadata-driven) and needs an `analysis/_core.py` component schema
(`geometry_info` / `_component_recipes`); the IC builders and the
FFT/sharding/stepper machinery are component-count-agnostic (leading
state axis replicated).

### Key design patterns

**Global singletons and import order**: `params`, `derived_params`,
`padded_res` (from `parameters.py`), `sharding` (from `sharding.py`),
and a geometry-specific `fourier` are module-level singletons.
`update_parameters()` mutates `params`/`derived_params` (with
`padded_res.set_padded_resolution(params)` applied alongside in
`__main__.py`); `sharding` and the geometry modules capture the
configuration at import time -- so JAX must be configured
(`jax_enable_x64`, platform, distributed) and parameters final
*before* importing any module that uses `sharding` or a geometry
module (see the `__main__.py` module docstring). A module that must
stay importable earlier (e.g. `random_field.py`) keeps `import jax`
out of module scope (lazy in-function imports; `from jax import
Array` under `TYPE_CHECKING`). The `fourier` singleton's wavenumber
arrays are global multi-device arrays: per-process host code cannot
`np.asarray` them -- recompute host-side from
`harmonics.real_harmonics`/`complex_harmonics` × `2π/L`.

**Stepper factory (two layers)**: `timestep.make_stepper()` takes four
required geometry-general callables plus two optional ones
(`get_rhs_measured_fn`, `l_bf_fn`) and returns JIT-compiled stepping
functions, including `predict_and_fully_correct` (fused corrector
loop, the primary path) and `step_cnab2`. Each geometry family wraps
it in its own builder that binds the `fourier` and `flow` singletons.
See the `make_stepper` docstring, `_base.py`, and
`triply_periodic.py`.

**Time-stepping scheme (`step.scheme`)**: both schemes are 2nd-order
and share the predictor/IMM-pressure solve. `"iterative-cn"`
(default): nonlinear term implicit via the corrector fixed-point
iteration; `2+c ≈ 3` FFT evals/step; stable well past the advective
CFL. `"cnab2"`: explicit AB2 nonlinear, **one** FFT eval/step (~3×
fewer FFTs on CFL-limited runs). Wall-bounded cnab2 advances only the
self-advection `u'×ω'` explicitly; the wall-stiff linear base-flow
coupling `L_bf` and (default-on `step.implicit_mean_coupling`) the
instantaneous mean-flow coupling `L_mf` stay implicit via an
**FFT-free** corrector (geometry `_l_bf`; `corrector_tolerance` /
`max_corrector_iterations` apply). The first step runs `iterative-cn`
while a discarded priming call seeds the AB2 history (`rhs_prev`,
carried by `__main__.py`, not persisted in snapshots); a
non-converging coupling corrector auto-falls back to a full
`iterative-cn` step (`lax.cond`). The residual `dt` bound is the
explicit self-advection CFL (pipe: near-axis azimuthal `CFL_th` --
the reason rigged-CGL is the default radial grid; Cartesian: near-wall
`Δy ~ 1/N²`); strongly non-normal regimes (counter-rotating
Taylor-Couette) want `iterative-cn` or a smaller `dt`; triply-periodic
cnab2 is the plain no-corrector AB2 step. `implicitness` sets the CN
weight in both schemes. Full detail incl. measured `dt` limits: the
`TimeStepping` docstring in `parameters.py`; implementation:
`step_cnab2`/`_cnab2_lbf_core` (`timestep.py`), `base_flow_coupling`/
`_l_bf` (`geometries/wall_bounded/`); guards: `tests/test_cnab2.py`,
`tests/test_temporal_order.py`.

**Corrector convergence is `dt`-limited, not CFL-limited**: the
iterative Crank-Nicolson corrector's contraction rate scales with
`step.dt`. A `corrector failed to converge` at *low* CFL with the
final error only marginally above `step.corrector_tolerance` means the
step is too large to contract within `max_corrector_iterations` --
reduce `dt` (or raise the iteration cap); it is not an advective-CFL /
blow-up. (Random-IC Kolmogorov stalls at ~1.4e-5 at `dt=0.01` but
converges in one corrector step at `dt=0.005`; the wall-bounded flows
are fine at 0.01.)

**Spectral array layout and sharding**: see the `sharding.py` module
docstring for shapes, partition specs, and the `(np0, np1)` device
mesh. See the `fft.py` module docstring for the reshard pipeline and
spectral padding.

**Perturbation formulation**: the solver evolves `u'` around laminar
`U(y)`; the rotational-form nonlinear term and base-flow gradient
elimination are documented in the `rhs.py` module docstring. The
force-driven dean/viscoelastic-dean systems instead integrate the
**total** field (`base_flow = 0`, mean-mode body force).

**Moving frame of reference (`phys.u_grid`)**: translates the
wall-bounded frame along the grid direction (`None` → laminar bulk:
1/2 pipe, 2/3 plane-Poiseuille, 0 otherwise; periodic systems reject
it). Implemented in convective form, added spectrally in each
geometry's `_get_rhs_core` **and** `_l_bf`, so both schemes integrate
it implicitly; only the CFL diagnostic advects with the frame-relative
velocity. It does **not** relax cnab2's explicit self-advection CFL
(`u'×ω'` is frame-invariant). Fields drift between frames, so a
changed `u_grid` on resume is trajectory-defining. Detail: the
`u_grid` field docs in `parameters.py` and `pad_base_flow` in
`_base.py`.

**JAX pytree registration**: `register_dataclass_pytree()` in
`sharding.py` registers geometry dataclasses, flow subclasses, solver
classes, and Fourier classes as JAX pytrees. See its docstring.

**Performance/memory trade-offs** (detail lives in the owning
docstrings/comments): operator storage & GPU speed order pallas <
banded < dense (`solver.backend` comments in `parameters.py`); SPIKE
reduced-system memory vs latency (`solver.block_thomas` /
`spike_block_size` comments, `scripts/spike_partition_info.py`);
Pallas whole-tile mode-plane padding — factors pre-padded once at
construction, per device shard inside the shard_map-local solve,
overhead matters only for small planes
(`from_banded_factors` in `solvers.py`, `pallas_block_m0` comments);
the viscoelastic 36-field RHS transform batch vs peak memory
(`solver.rhs_transform_chunks`; `_get_rhs_core` in
`annular_viscoelastic.py`); cnab2 is a throughput win, not a
peak-memory one (`step_cnab2` docstring in `timestep.py`); the
dominant global memory multipliers are `phys.oversampling_factor`
(dealiased grid, ~2.25x physical points at the default 3) and
`res.double_precision` (2x).

### Parameter layering

Lowest priority first: defaults (Pydantic models) -> parameters
embedded in a resumed snapshot -> `parameters.toml` -> CLI args.
`update_parameters()` only applies explicitly-set, non-`None` fields;
`validate_parameters()` runs once after the final layer. Snapshot
params are read by `read_snapshot_params()` from the snapshot's
`_dnsjax_meta.json` -- but the JAX-setup fields `dist.np0`/`np1`/
`platform` and `res.double_precision` are *not* inherited (resume is
device-/precision-agnostic). `__main__.py` resolves the resume
snapshot (explicit CLI `init.snapshot` over TOML) and applies the
layers in order before JAX/singleton setup.

### Configuration (`parameters.toml`)

See the `parameters.py` model docstrings for full documentation (the
`Initiation` docstring for start-mode precedence, `TimeStepping` for
the schemes, `Solver` for the Pallas knobs). Key fields:

| Section    | Key fields                                          |
|------------|-----------------------------------------------------|
| `[phys]`   | `re`, `re1`/`re2` (Taylor-Couette inner/outer; derive `re := Re_ref`), `system`, `oversampling_factor`, `oversample_y`, `driving`, `block_mean_spanwise_velocity` (mean spanwise velocity: axial for TC, z for Cartesian), `u_grid`; viscoelastic-dean only: `el`, `wi`, `beta`, `epsilon`, `kappa` (`re := wi/el` derived) |
| `[geo]`    | `lx`, `lz`, `tilt_degree`, `eta` (TC radius ratio), `delta` (viscoelastic-dean inner radius; radii `(δ, δ+2)`), `wall_grid` (custom grid file; always overrides generation), `grid_type` (`"cgl"` / `"half-cgl"` / `"tanh"`; cylindrical default = rigged-CGL, `"half-cgl"` cylindrical + `iterative-cn` only), `grid_stretch` |
| `[res]`    | `nx`, `ny`, `nz`, `fd_order`, `double_precision`    |
| `[init]`   | Start-mode precedence: `snapshot` > `start_from_laminar` > `localized_rolls` > `random_field` (default **on**). `snapshot`, `t0`, `it0`, `isnap0`, `force_resume`, `random_amplitude`/`_smoothness`/`_seed`/`_mean_flow`/`_conformation_amplitude`, `localized_rolls_amplitude`/`_width`/`_wavelength` |
| `[outs]`   | `it_stats`, `it_steps` (CFL cadence -> `steps.dat`), `it_snapshot`, `it_corrector` (-> `corrector.dat`; requires `it_error_check <= it_corrector`), `it_error_check` (host-sync cadence), `nbuffer`, `stats_precision`, `snapshot_write_mode`, `snapshot_pad_width`, `snapshot_embed_stats`, `snapshot_save_initial`, `snapshot_save_final` (last three default on, independent of `it_snapshot`) |
| `[step]`   | `dt`, `scheme` (`"iterative-cn"` / `"cnab2"`, both supported for every flow), `implicitness`, `corrector_tolerance`, `max_corrector_iterations`, `implicit_mean_coupling` |
| `[stop]`   | `max_sim_time`, `max_wall_time` (ISO 8601), `check_laminarization` (default on; terminate when `E'` < `laminarization_threshold`, default `1e-9`) |
| `[dist]`   | `np0` (wall-normal / kz axis), `np1` (spanwise / kx axis), `platform` |
| `[solver]` | `backend` (`"pallas"` default / `"banded"` / `"dense"`; `banded` recommended for CPU-heavy or multi-GPU work until multi-GPU Pallas is validated), `pallas_force_pivoting`, `pallas_block_m0`/`m1` (mode tile, default 2/32), `pallas_stability_tol`, `pallas_num_warps`/`pallas_num_stages`, `spike_block_size`, `block_thomas`, `rhs_transform_chunks` (viscoelastic RHS memory knob) |

The default `parameters.toml` contains only
`[phys] [geo] [res] [init] [outs] [step] [stop]`; `[dist]` and
`[solver]` rely on model defaults -- set them via CLI (e.g.
`--dist.np1 2`, `--solver.backend dense`) or by adding the section.

### Diagnostics (`stats.dat`, `steps.dat`, `corrector.dat`)

On-device buffered stats, flushed periodically (fsync-ed) to
`stats.dat`. The CFL diagnostic (every `outs.it_steps` steps, measured
inside the nonlinear evaluation via the `rhs.py` `measure_fn` hook --
see `measurements.py`) goes to `steps.dat`, and the corrector
diagnostic (every `outs.it_corrector` steps) to `corrector.dat`, the
same way. All three streams are also flushed at shutdown, after the
first (JIT-heavy) step, around snapshot writes, and on SIGTERM/SIGINT,
so the `.dat` files stay consistent with the snapshots and survive an
interruption. Buffering mechanism and file format: the `__main__.py`
module docstring.

### Snapshots

A snapshot is a single uncompressed tar (`format_version: 3`) wrapping
a zarr3 store: `_dnsjax_meta.json` (time, iteration, lineage index
`isnap`, layout, grid, full params; read JAX-free via
`snapshot_meta.py`), `state/zarr.json`, and one contiguous chunk per
state component (3, or 9 for viscoelastic; metadata-driven, validated
on resume). Readable with standard tools and no dnsjax (`tar xf`
yields a zarr store + plain-JSON metadata); each device writes its
disjoint byte ranges directly into the one file (raw offset I/O / GDS
preserved; never compressed). Resume is np- and precision-agnostic;
a different wall-normal grid is interpolated at load
(`_interpolate_if_needed` in `__main__.py`). The stored state is the
spectral perturbation `u'` for the base-flow systems (a laminar
snapshot is a zero array) and the **total** field for
dean/viscoelastic-dean.

Snapshots are named `state{isnap:0Nd}.tar`
(`N = outs.snapshot_pad_width`); `isnap` starts at `init.isnap0` and
bumps on every write. By default the IC of any non-continuation start
is saved as `state00000.tar`, the final state is saved on termination
(deduped), and every snapshot embeds its `get_stats` dict as
`_dnsjax_stats.json`. On resume, `t`/`it`/`isnap` continue only when
`parameters.trajectory_defining_changes(meta["params"])` is empty (no
`phys`/`geo`/`res` override); any such change starts a **new
trajectory** at `t=it=isnap=0` unless `init.force_resume` is set --
orthogonal to the hard `nx`/`nz`/`system`/`precision` rejects of
`validate_snapshot_params`. Full detail: `snapshot.py` and
`__main__.py` module docstrings.

### JAX-specific notes

- Explicit mode sharding is used globally rather than Auto mode. Do
  not use `jax.lax.with_sharding_constraint`.
- Allocate sharded arrays directly on devices (`out_sharding` argument
  of `jnp.zeros`, `.at[...].get/set`, etc.) instead of allocating
  globally and redistributing with `jax.device_put`; when direct
  allocation is not possible, do not substitute `jnp.asarray` for
  `jax.device_put`.
- `jax_enable_x64` is set from `params.res.double_precision` before
  JAX initializes arrays.
- JAX has no zero-copy complex<->real bitcast. Real-operator ×
  complex-field GEMMs/solves use an explicit trailing re/im split --
  reuse `apply_y_matrix` (`geometries/wall_bounded/_base.py`) or the
  `solvers.py` pattern.
- Buffer donation: `predict_and_fully_correct(_measured)` donate
  `state`; `step_cnab2(_measured)` donate `state` **and** `carry`. Any
  caller that reuses an input afterwards must pass `jnp.copy` -- the
  `__main__` warm-up/priming calls do, and Dean's `init_state` copies
  the module-level `_laminar_state` for the same reason.
- The first time step is excluded from benchmark statistics (JIT
  compilation overhead).
- FFT normalization uses `norm="forward"`.
- Dicts returned from jitted functions (`get_stats`, measurements) are
  canonicalized to **sorted key order** by pytree flattening -- this
  sets the column order of `stats.dat`/`steps.dat`; never assume
  insertion order.
- A flow dataclass is a registered pytree, so every array field is
  traced into the jitted steppers; keep data needed only *outside* jit
  (e.g. a precomputed laminar state) at module level, not as a flow
  field.
- A `dt` / resolution / `params` sweep needs a **subprocess per
  value**: they are captured into the singletons and jitted steppers
  at import/trace time (the `test_*` subprocess-per-config idiom).
- Pallas/Triton GPU kernels: interpret mode (CPU) validates numerics
  but **not** Triton's lowering restrictions; compile-check on the
  GPU-less dev box with
  `jax.jit(f).trace(*a).lower(lowering_platforms=("cuda",))`. The
  restrictions, layout rules, and the partial-tile miscompile (pad
  tiled arrays to whole tiles) are documented in the
  `_pallas_banded_solve` docstring; `test_pallas_cuda_lowering` is the
  regression guard.

## Scripts

- `scripts/spike_partition_info.py`: display SPIKE block-partition
  trade-offs for a given resolution.
- `scripts/snapshot_import.py`: **library** (not a CLI) converting a
  velocity field already in dnsjax's native component/axis structure
  into a single-file snapshot. Public API: `configure_target`,
  `to_spectral_state`, `write_snapshot`, `convert_field_to_snapshot`,
  `validate_state`. Perturbation-only, velocity-only (viscoelastic
  rejected); any external layout permutation and `u_±` mixing are the
  caller's responsibility. See its module docstring for the layout
  table and normalization options.
- `scripts/pallas_tiling_diagnostic.py`: GPU construct-bisection
  harness that localised the Triton partial-tile miscompile and
  confirms the pad-to-whole-tiles fix (run on real GPU).
- `scripts/pallas_solve_profile.py`: GPU diagnostic for where the
  Pallas banded solve's time goes (profiled the matvec transpose
  sources).

## Tests

All to be kept up-to-date as the respective modules change. Detail
lives in each test file's module docstring; entries here are
one-liners. Cross-cutting notes:

- The laminar smoke test has `u' = 0`, so it does **not** exercise the
  rotational nonlinear term (a wrong advection change can still report
  `err=0`); `test_random_smoke.py` drives that path.
- In-process geometry tests configure the singletons **once** at
  module top (`test_cylindrical.py`/`test_annular.py` via
  `update_parameters()`, `test_cartesian.py` via direct `params.*`
  assignment); a test that re-calls `update_parameters()` (only
  `test_annular.py` does) mutates the shared `params`/`derived_params`
  and must restore the module config before returning.
- Any test exercising `taylor-couette` must set `params.phys.re1`/
  `re2` and `params.geo.eta` before singleton construction (all three
  default to `None` and `update_parameters()` raises otherwise); the
  unit tests use `100`/`0`/`0.5`, the smoke/integration tests
  counter-rotating values.
- A tiny complex-FFT-axis nz trips "Difference (n - N) = 3 cannot be
  odd" in the 3/2-rule dealiasing (nz=6 fails, nz=8/32 work; the real
  axis nx=6 is fine -- different rule).

- `tests/test_banded_solver.py`: geometry-independent SPIKE + Pallas
  banded backend (interpret parity incl. pad-to-whole-tiles,
  compile-only cuda-lowering guard, `_decide_pallas_or_spike`).
- `tests/test_banded_solver_sharded.py`: shard_map-local Pallas solve
  on a forced (2, 2) mesh (per-shard factor tile padding, sharded
  `.solve` oracle parity, component-axis layouts).
- `tests/test_cartesian.py`: Cartesian operator/matvec tests + Pallas
  band-vs-dense parity.
- `tests/test_cylindrical.py`: cylindrical operator/matvec tests +
  Pallas band-vs-dense parity (guards the shared-assembly refactor).
- `tests/test_annular.py`: annular operator/matvec tests, SPIKE- and
  Pallas-vs-dense parity, circular-Couette A0/B0 checks.
- `tests/test_viscoelastic.py`: sPTT conformation-tensor machinery
  (spin conversions, tensor Laplacian vs reference, laminar fixed
  point, `Hc` parity, Frobenius norm, fused-RHS FFT count).
- `tests/test_integration.py`: quadrature weights and interpolation
  matrices.
- `tests/test_cnab2.py`: CN/AB2 structural guards -- split exactness,
  `L_mf` oracle, carry-seed independence, jaxpr FFT-count guards,
  viscoelastic split.
- `tests/test_temporal_order.py`: second-order temporal accuracy
  (Kolmogorov cnab2 self-convergence + icn cross-check, plane-Couette
  scheme-difference slope).
- `tests/test_mean_mask.py`: padding slots carry placeholder
  wavenumbers; `mean_mask` is the unique k^2 = 0 mode under forced
  spectral padding.
- `tests/test_laminar_smoke.py`: laminar fixed-point smoke for all
  wall-bounded flows (subprocess/mpirun; total-field Dean and
  viscoelastic-dean energy-balance branches).
- `tests/test_random_smoke.py`: random-IC nonlinear integration for
  all 7 flows, plus cnab2 entries, the default-IC entry, and the
  multi-device-padding / Pallas-backend regression entries.
- `tests/test_snapshot.py`: snapshot round-trips, np-agnostic resume,
  standard-tools readability, 9-component viscoelastic case, isnap /
  stats members.
- `tests/test_resume.py`: snapshot lineage and resume policy (offline
  `trajectory_defining_changes` + grid-validation units; mpirun
  integration; `--unit-only` to skip the latter).
- `tests/test_snapshot_import.py`: `scripts/snapshot_import.py`
  native-contract validation (offline).
- `tests/test_snapshot_export.py`: `dnsjax.analysis` API vs solver
  ground truth (JAX-free import guarantee, curl parity at machine
  precision).
- `tests/test_rolls_smoke.py`: localized-rolls IC integration for the
  5 wall-bounded flows (incl. a multi-device padding case).
- `tests/test_localized_rolls.py`: rolls construction self-test
  (no-slip, determinism, device-count independence, divergence bound,
  peak-scaling guard).
