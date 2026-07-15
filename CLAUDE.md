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

`uv sync` (also installs the `dnsjax` console script into `.venv/bin`)

### Lint

`uv run ruff check --fix`

Line length is 79 for **all** lines (ruff `line-length = 79`, E501),
not only docstrings/comments.

### Run tests

All tests are standalone scripts (`uv run python tests/test_*.py`,
the source of truth; they rely on `__main__` setup + shared
module-level singletons, so `pytest` must never import them). An
optional pytest bridge (`tests/pytest_suite.py`, the **only** file
pytest collects via `[tool.pytest.ini_options] python_files`) runs
each script as a subprocess: `uv run pytest -m "not slow"` for the
offline loop, plain `uv run pytest` for everything (`mpi`-marked
scripts auto-skip without `mpirun`).

Two ways to get multiple devices, and they do **not** mix:
offline/in-process tests force CPU devices via
`XLA_FLAGS=--xla_force_host_platform_device_count=N` + set
`params.dist.np0`/`np1` before importing `sharding` (no MPI); a
`dnsjax` run needs real `mpirun -np N` (1 device/process).

Single file: `uv run python tests/test_cartesian.py`
Laminar smoke (1D multi-device):
`uv run python tests/test_laminar_smoke.py --np 2`
Laminar smoke (2D multi-device):
`uv run python tests/test_laminar_smoke.py --np 4 --np0 2`

### Smoke test (laminar time stepping)

Any solver run must be launched via `mpirun` (even `mpirun -np 1 ...`);
under `mpirun` invoke `.venv/bin/dnsjax` directly (`uv run` does not
compose with `mpirun`; `python -m dnsjax` is the equivalent module
form, same `__main__.main()`). `uv run dnsjax --help` needs no mpirun
(exits at the parser, no side effects). A run writes its `.dat`
files/snapshots to the cwd, so launch from a scratch dir. `np0 * np1`
counts devices, not processes (a single process can address all GPUs on
a node); launch recipe, SLURM discipline, and the per-task-visibility
trap: the `Distribution` docstring in `parameters.py`.

`mpirun -np 2 .venv/bin/dnsjax --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 27`

For double parallelisation on the default CGL grid (pick `ny` divisible
by `np0` for an even split; the sharding layer auto-pads otherwise, and
every auto-pad -- spectral, physical, padded-size rounding, Pallas
tiles -- prints a startup diagnostic):

`mpirun -np 4 .venv/bin/dnsjax --dist.np0 2 --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 28`

The laminar state should step with a single corrector, stepping error
O(-18) or less and perturbation energy O(-32) or less. Run the tests in
the Tests section below if you touch the modules they test.

### Generate random initial condition

In-process only (no offline script): `--init.random_field True` is the
**default** start mode when no snapshot is given (with
`--init.random_amplitude` / `_smoothness` / `_seed` / `_mean_flow`;
viscoelastic-dean adds `--init.random_conformation_amplitude`). All
flow systems; total-field dean/viscoelastic-dean add the analytical
laminar profile (+ sPTT-equilibrium conformation). Generators
(`generate_random_state`), the per-device/padded-mesh-safe build, and
the divergence/Hermitian validation caveats: the `random_field.py`
module docstring.

### Generate localized-rolls ("turbulent spot") initial condition

In-process only (no offline script): `--init.localized_rolls True`
(with `--init.localized_rolls_amplitude` / `_width` / `_wavelength`);
wall-bounded systems only, higher precedence than the default random
field. A compact peak-normalized spot localized in every homogeneous
direction (total-field dean/viscoelastic-dean add the laminar
profile). Construction, per-geometry pairing, and the sharded
separable build: the `localized_rolls.py` module docstring.

### Triggering transition to turbulence

The solver reproduces the linear physics **quantitatively**, so a flow
that decays is a *regime/time* matter, **not** a solver bug: transition
develops over O(100) advective units (smoke tests only reach `t=1`) and
needs `Re` above sustainment (plane-Couette ≳ 350-500), a domain ≳ the
minimal flow unit, and a finite-amplitude perturbation (random
`amplitude ≈ 0.1-0.2`, or a strong localized spot). Near-minimal boxes
give transient (decaying) turbulence.

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
__main__.py           Entry point main() (console script `dnsjax` +
                      python -m), stats buffering, snapshot
                      resume/lineage
bootstrap.py          Shared entry-point setup, used by all entry
                      points (solver, scripts, tests):
                      resolve_parameters (CLI/toml/snapshot layering),
                      configure_jax_runtime (distributed init),
                      configure_jax_platform/platform_from_argv
                      (single-process --dist.platform)
parameters.py         Pydantic parameter models (JAX-free); singletons
                      params, derived_params, padded_res;
                      trajectory_defining_changes; round_up_padded
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
probes.py             Spectral-mode probe stream (outs.probe_modes):
                      sharded-gather mode extractor + buffered binary
                      writer (probes.bin + probes.json sidecar)
forcing.py            White-in-time stochastic mode kicks ([force]
                      section): sharded scatter-add mode injector
                      (the extractor's dual) + buffered forcing.bin
                      coefficient log; reader/identification in
                      analysis/response/ssi.py
timestep.py           make_stepper() factory:
                      predict_and_fully_correct (+_measured),
                      step_cnab2 (+_measured), _cnab2_lbf_core
fd.py                 NumPy-only FD utilities (JAX-free): Fornberg
                      D1/D2, quadrature rules, interpolation matrices,
                      tanh grids
solvers.py            Geometry-independent linear solvers:
                      DenseJAXSolver (reference/oracle) and
                      PerModeBandedPallasOperator (production banded
                      sweep) -- see _pallas_banded_solve / .solve
                      docstrings
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
                      taylor_couette, quasi_keplerian, dean,
                      viscoelastic_dean -- base flows/driving in
                      wall_bounded/CLAUDE.md
  triply_periodic/    monochromatic.py: Kolmogorov/Waleffe/decaying-box
analysis/             External-facing JAX-free snapshot post-processing
                      API (+ the JAX-based transient_growth CLI and the
                      response/ subpackage: probe reader, operator
                      tools, ensemble/LIM/SSI operator identification)
                      -- see analysis/CLAUDE.md
```

### Transient-growth analysis

`python -m dnsjax.analysis.transient_growth` computes 3D linear optimal
energy growth `G(t)` around an arbitrary wall-normal **total** profile
`U(y)` for the five base-flow wall-bounded flows (plane-couette/
poiseuille, pipe, taylor-couette, quasi-keplerian; Dean out of scope),
reusing the solver's own linear step per Fourier mode. Single-device, GPU-runnable
(`--dist.platform cuda`). Full math, the `frozen_profile_flow` hook,
and the CLI/output spec: the module docstring and `analysis/CLAUDE.md`.
`--save-operator` additionally exports each mode's reduced generator
(`<stem>_tg_op.npz`) for the `analysis/response/` post-processing
(controllability modes, growth curves, ensemble/LIM/SSI operator
identification).

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
`padded_res` (`parameters.py`), `sharding` (`sharding.py`), and a
geometry `fourier` are module-level singletons captured at import time
-- so JAX must be configured and parameters final (`update_parameters()`)
*before* importing `sharding` or any geometry module (the setup
contract: the `bootstrap.py` module docstring). Earlier-importable modules (e.g. `random_field.py`) keep
`import jax` out of module scope. `fourier`'s wavenumber arrays are
global multi-device arrays -- host code recomputes them from
`harmonics.real_harmonics`/`complex_harmonics` × `2π/L`, never
`np.asarray`.

**Stepper factory (two layers)**: `timestep.make_stepper()` builds the
JIT-compiled stepping functions (the primary `predict_and_fully_correct`
and `step_cnab2`) from geometry-general callables; each geometry family
wraps it in a builder that binds the `fourier`/`flow` singletons. See
the `make_stepper` docstring, `_base.py`, and `triply_periodic.py`.

**Time-stepping scheme (`step.scheme`)**: both 2nd-order, sharing the
predictor/IMM-pressure solve. `"iterative-cn"` (default) makes the
nonlinear term implicit via the corrector fixed point (`2+c ≈ 3` FFT
evals/step, stable past the advective CFL); `"cnab2"` advances it
explicitly (AB2), **one** FFT eval/step (~3× fewer on CFL-limited runs)
— wall-bounded cnab2 keeps the wall-stiff coupling `_l_bf` implicit via
an FFT-free corrector, staying advective-CFL-bound. `implicitness` sets
the CN weight; `implicit_mean_coupling` (default on) and
`split_corrector` (opt-in, default off) tune the coupling. Full detail
(measured `dt` limits, per-geometry CFL, the split/mean-coupling
rationale): the `TimeStepping` docstring (`parameters.py`);
implementation `timestep.py`; guards `tests/test_cnab2.py`,
`tests/test_temporal_order.py`.

**Corrector convergence is `dt`-limited, not CFL-limited**: the
iterative-CN corrector's contraction rate scales with `step.dt`, so a
`corrector failed to converge` at *low* CFL (final error only just above
`corrector_tolerance`) means the step is too large to contract within
`max_corrector_iterations` — reduce `dt` (or raise the cap), not a
blow-up. (Random-IC Kolmogorov needs `dt=0.005`, capped in
`test_random_smoke.py`; the wall-bounded flows are fine at 0.01.)

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
wall-bounded frame along the grid direction (`None` → laminar bulk;
periodic systems reject it), integrated implicitly by both schemes. It
does **not** relax cnab2's explicit self-advection CFL (`u'×ω'` is
frame-invariant), and a changed `u_grid` on resume is
trajectory-defining (fields drift between frames). Detail: the `u_grid`
field docs (`parameters.py`) and `pad_base_flow` (`_base.py`).

**JAX pytree registration**: `register_dataclass_pytree()` in
`sharding.py` registers geometry dataclasses, flow subclasses, solver
classes, and Fourier classes as JAX pytrees. See its docstring.

**Performance/memory trade-offs** (detail in the owning
docstrings/comments): pallas beats dense in storage and speed
(`solver.backend`); whole-tile mode-plane padding (`from_banded_factors`
/ `pallas_block_m0` in `solvers.py`); the RHS transform batch vs peak
memory, all flows (`solver.rhs_transform_chunks`, applied by
`fft.chunked_transform`; the ~36-field viscoelastic batch is where it
bites — the `fft.py` memory note also records why fusing the dealias
zero-pad into the FFT is a dead end); cnab2 is a throughput win, not a
peak-memory one (`step_cnab2` in `timestep.py`).
The dominant global memory multipliers are `phys.oversampling_factor`
(~2.25× physical points at the default 3) and `res.double_precision`
(2×).

### Parameter layering

Lowest priority first: defaults (Pydantic models) → resumed-snapshot
params → `parameters.toml` → CLI args. `update_parameters()` applies
only explicitly-set, non-`None` fields; `validate_parameters()` runs
once after the final layer. The JAX-setup fields `dist.np0`/`np1`/
`platform` and `res.double_precision` are *not* inherited from a
snapshot (resume is device-/precision-agnostic). Detail:
`bootstrap.py` (`resolve_parameters`).

Two fields carry per-family defaults **re-resolved on every
`update_parameters()` call** unless explicitly set through a layer:
`solver.backend` (periodic → `"dense"`, wall-bounded → `"pallas"`) and
`geo.grid_type` (wall-bounded → `"cgl"`, except cylindrical +
`iterative-cn` → `"half-cgl"`; periodic / `wall_grid` → `None`). Scripts
and tests must set these via `update_parameters(Parameters(solver=...,
geo=...))` — a direct `params.solver.backend = ...` assignment is
silently overwritten by the re-resolution (never enters
`_user_set_fields`).

### Configuration (`parameters.toml`)

See the `parameters.py` model docstrings for full documentation (the
`Initiation` docstring for start-mode precedence, `TimeStepping` for
the schemes, `Solver` for the Pallas knobs). Key fields:

| Section    | Key fields                                          |
|------------|-----------------------------------------------------|
| `[phys]`   | `re`, `re1`/`re2` (TC), `re1`/`r_omega` (quasi-keplerian; `re2` derived), `system`, `oversampling_factor`, `oversample_y`, `driving`, `block_mean_spanwise_velocity`, `u_grid`; viscoelastic-dean: `el`, `wi`, `beta`, `epsilon`, `kappa` |
| `[geo]`    | `lx`, `lz`, `tilt_degree`, `eta` (TC), `m0` (annular/cylindrical azimuthal wedge, `lz = 2π/m0`), `delta` (viscoelastic-dean), `wall_grid`, `grid_type` (`"cgl"`/`"half-cgl"`/`"tanh"`), `grid_stretch` |
| `[res]`    | `nx`, `ny`, `nz`, `fd_order`, `double_precision`    |
| `[init]`   | Start-mode precedence: `snapshot` > `start_from_laminar` > `localized_rolls` > `random_field` (default **on**). `t0`, `it0`, `isnap0`, `force_resume`, `random_*`, `localized_rolls_*` |
| `[outs]`   | `it_stats`, `it_steps`, `it_snapshot`, `it_corrector`, `it_error_check`, `probe_modes`, `it_probes`, `nbuffer`, `stats_precision`, `snapshot_write_mode`, `snapshot_pad_width`, `snapshot_embed_stats`, `snapshot_save_initial`, `snapshot_save_final` |
| `[step]`   | `dt`, `scheme` (`"iterative-cn"`/`"cnab2"`), `implicitness`, `corrector_tolerance`, `max_corrector_iterations`, `implicit_mean_coupling`, `split_corrector` |
| `[force]`  | White-in-time stochastic mode kicks (all-or-none; trajectory-defining): `modes`, `profiles` (npz of channel profiles), `amplitude`, `it_force` (multiple of `it_probes`), `n_channels`, `seed` — the `StochasticForcing` docstring + `forcing.py` |
| `[stop]`   | `max_sim_time`, `max_wall_time` (ISO 8601), `check_laminarization`, `laminarization_threshold` |
| `[dist]`   | `np0` (wall-normal / kz axis), `np1` (spanwise / kx axis), `platform` |
| `[solver]` | `backend` (`"pallas"`/`"dense"`), `pallas_block_m0`/`m1`, `pallas_stability_tol`, `rhs_transform_chunks` |

The default `parameters.toml` contains only
`[phys] [geo] [res] [init] [outs] [step] [stop]`; `[force]`, `[dist]`
and `[solver]` rely on model defaults -- set them via CLI (e.g.
`--dist.np1 2`, `--force.modes "3,0"`) or by adding the section.

### Diagnostics (`stats.dat`, `steps.dat`, `corrector.dat`, `probes.bin`, `forcing.bin`)

Three on-device buffered scalar streams, flushed periodically
(fsync-ed): `get_stats` → `stats.dat`, the CFL diagnostic
(`outs.it_steps`, via the `rhs.py` `measure_fn` hook) → `steps.dat`,
the corrector diagnostic (`outs.it_corrector`) → `corrector.dat`; plus
the binary spectral-mode probe stream (`outs.probe_modes`/`it_probes`,
wall-bounded only) → `probes.bin` + `probes.json` sidecar (format and
sharded gather: the `probes.py` module docstring; JAX-free reader:
`dnsjax.analysis.response.probes`), and the stochastic-kick
coefficient log (`[force]`) → `forcing.bin` + `forcing.json` (format,
kick timing vs probes/snapshots, resume PRNG continuation: the
`forcing.py` module docstring; reader:
`dnsjax.analysis.response.ssi`). All are also flushed at
shutdown, after the first step, before snapshot writes, and on
SIGTERM/SIGINT, so they stay consistent with snapshots. Every flushed
row and host-synced scalar is guarded against NaN/inf: a hit prints
one `FATAL: non-finite ...` line naming the quantity, skips the final
snapshot, and exits with code 3 (detail: the `__main__.py` module
docstring). Buffering mechanism and file format: the `__main__.py`
module docstring.

### Snapshots

A snapshot is a single uncompressed tar (`format_version: 3`) wrapping a
zarr3 store — `_dnsjax_meta.json` (params/grid/lineage + the writing
code's git hash, printed at startup too; JAX-free via
`snapshot_meta.py`), `state/zarr.json`, and one contiguous chunk per
state component (3, or 9 for viscoelastic; metadata-driven). Readable
with standard tools and no dnsjax; each device writes its disjoint byte
ranges directly (raw offset I/O / GDS, never compressed). The stored
state is the spectral perturbation `u'` for base-flow systems (laminar =
zero array), the **total** field for dean/viscoelastic-dean. Resume is
np-/precision-agnostic and re-grids a changed wall-normal grid at load;
`t`/`it`/`isnap` continue only when
`trajectory_defining_changes(meta["params"])` is empty — a `phys`/`geo`/
`res` override starts a **new trajectory** unless `init.force_resume`
(distinct from the hard `nx`/`nz`/`system`/`precision` rejects). Full
detail: `snapshot.py` and `__main__.py` module docstrings.

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
- Diagnostic scripts and offline / in-process tests select the JAX
  backend from `--dist.platform` (default cpu) via
  `configure_jax_platform` / `platform_from_argv` (`bootstrap.py`),
  before importing `sharding` or any geometry module -- so
  `--dist.platform cuda` runs the real Pallas kernels on GPU. In-process
  multi-device tests force CPU (`--xla_force_host_platform_device_count`,
  CPU-only); real multi-GPU uses `mpirun` (`test_random_smoke.py --np`).
- Pallas/Triton GPU kernels: interpret mode (CPU) validates numerics
  but **not** Triton's lowering; compile-check on the GPU-less dev box
  with `jax.jit(f).trace(*a).lower(lowering_platforms=("cuda",))`. The
  lowering/layout rules, the partial-tile miscompile (pad tiled arrays
  to whole tiles), and the `check_vma=False` a `pallas_call` inside a
  `shard_map` needs are in the `_pallas_banded_solve` docstring; guards
  `test_pallas_cuda_lowering`, `test_pallas_cuda_lowering_sharded_solve`.

## Scripts

One line each; full rationale/usage in each script's module docstring.

- `scripts/snapshot_import.py`: **library** (not a CLI) packing a
  native-layout velocity field into a snapshot (perturbation/velocity
  only).
- `scripts/snapshot_perturb.py`: CLI + library injecting a scaled
  single-mode perturbation (transient-growth optimal, controllability
  mode, or raw profile) into an existing snapshot; solver-exact
  `--amplitude-energy`, `--negate` for antithetic pairs.
- `scripts/ensemble_setup.py`: JAX-free `harvest`/`build` CLI turning
  a snapshot archive into ensemble member run trees (perturbed seeds,
  per-member `parameters.toml` + probe stream, `run_commands.txt`,
  antithetic/baseline pairing); aggregation lives in
  `dnsjax.analysis.response.ensemble`.
- `scripts/pallas_tiling_diagnostic.py`: GPU harness that localised the
  Triton partial-tile miscompile and confirms the pad-to-whole-tiles fix.
- `scripts/pallas_solve_profile.py`: GPU diagnostic for where the Pallas
  banded solve's time goes (+ `_imm_iteration` stage breakdown, cnab2
  composition, mode-tile knob sweep; `--cpu-smoke`).
- `scripts/solver_benchmark.py`: pallas-vs-dense validation & benchmark
  incl. multi-GPU correctness with a SLURM preflight ladder
  (`--cpu-bench`, `--cpu-smoke`).
- `scripts/pivot_stability_survey.py`: CPU survey of the no-pivot
  banded-LU stability checks across the config space (finding: no real
  config trips them).

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
  counter-rotating values. `quasi-keplerian` is the same annular flow
  parameterized by `re1`/`r_omega`/`eta` (with `re2` derived on the
  quasi-Keplerian half-line `R_Ω < -1`); tests use `re1`/`-1.2`/`0.71`.
- The annular / cylindrical azimuthal wedge (`geo.m0 > 1`) reduces the
  domain to `θ ∈ [0, 2π/m0)` and resolves only `m = m0·j`; it genuinely
  cuts azimuthal cost/memory by `m0` at fixed `nz` (all array/FFT sizes
  are `nz`-driven; `m0` only scales wavenumber values). Cylindrical
  parity (`m_is_even`) tracks the physical `m0·j`.
- The 3/2-rule pad on a complex FFT axis (z, periodic y) has no parity
  constraint (`zeropad_fft`/`truncate_fft` place modes exactly for odd
  pads and odd sizes); `set_padded_resolution` auto-rounds every
  padded FFT size for np1/np0 divisibility **and** to FFT-friendly
  7-smooth lengths with a startup note (e.g. nz=6, np1=2: nz_padded
  9->10; nz=94: 141->144; `round_up_padded_smooth` in
  `parameters.py`), so every (n, np) combination runs. The real axis
  nx gets only the smoothness rounding (never sharded, so no
  divisibility part).

- `tests/pytest_suite.py`: the pytest bridge (subprocess per script,
  `mpi`/`slow` markers; the only pytest-collected file — see Run
  tests).
- `tests/test_banded_solver.py`: geometry-independent Pallas banded
  backend (interpret parity, cuda-lowering guard, check contract).
- `tests/test_banded_solver_sharded.py`: shard_map-local Pallas solve on
  a forced (2, 2) mesh (per-shard tile padding, `.solve` oracle parity).
- `tests/test_cartesian.py`: Cartesian operator/matvec + Pallas
  band-vs-dense parity.
- `tests/test_cylindrical.py`: cylindrical operator/matvec + Pallas
  band-vs-dense parity.
- `tests/test_annular.py`: annular operator/matvec, Pallas-vs-dense
  parity, circular-Couette A0/B0 checks.
- `tests/test_viscoelastic.py`: sPTT conformation-tensor machinery
  (spins, tensor Laplacian, laminar fixed point, `Hc`, fused-RHS FFTs).
- `tests/test_integration.py`: quadrature weights and interpolation
  matrices.
- `tests/test_cnab2.py`: CN/AB2 + split-corrector structural guards
  (split exactness, `L_mf` oracle, FFT-count jaxprs, gate-off).
- `tests/test_temporal_order.py`: second-order temporal accuracy
  (Kolmogorov cnab2 self-convergence, plane-Couette scheme slope).
- `tests/test_mean_mask.py`: `mean_mask` is the unique k²=0 mode under
  forced spectral padding.
- `tests/test_padding.py`: padded-size rounding (`round_up_padded`/
  `round_up_padded_smooth` units; primary/fallback/smooth rounding
  subprocess cases with the diagnostic asserted) + FFT exactness cases
  (odd-pad, odd-nz, fused spec-pad on a forced (2, 2) mesh) +
  `chunked_transform` bit-parity.
- `tests/test_laminar_smoke.py`: laminar fixed-point smoke for all
  wall-bounded flows (subprocess/mpirun; Dean/viscoelastic branches;
  console-script `--help`/run entries, an nz-padding entry, and
  annular/cylindrical azimuthal-wedge entries `quasi-keplerian-wedge`
  / `pipe-wedge`).
- `tests/test_random_smoke.py`: random-IC nonlinear integration for all
  8 flows (+ cnab2, default-IC, gate-off, multi-device-padding,
  chunked-RHS entries, and a nan-guard entry asserting the forced
  blow-up aborts with exit 3).
- `tests/test_quasi_keplerian.py`: quasi-keplerian control-parameter
  derivation (Re_o/Re_s/μ/q from re1/r_omega/eta, pinned to the
  literature line η=0.71, R_Ω=-1.2), regime/validation errors, and the
  annular + cylindrical azimuthal-wedge Fourier units (m0-scaled `m`,
  lz-based CFL, cylindrical physical-parity `m_is_even`); offline,
  subprocess-per-case.
- `tests/test_probes.py`: runtime spectral-mode probe stream (sharded
  extractor exactness on a forced (2, 2) mesh, writer semantics,
  parameter validation; mpirun laminar/random solver runs behind
  `--unit-only`).
- `tests/test_forcing.py`: runtime stochastic kicks (sharded injector
  + kick placement bit-exactness on a forced (2, 2) mesh, coefficient
  stream + PRNG resume-skip, profile/parameter validation; mpirun
  forced-laminar runs behind `--unit-only`: exported-propagator
  trajectory prediction to ~1e-4 and split-resume stream equality).
- `tests/test_snapshot_perturb.py`: `scripts/snapshot_perturb.py`
  injection (bit-exactness + conjugate partner, energy convention,
  antithesis, real TG/controllability npz sources, error paths).
- `tests/response/test_probes_reader.py`: JAX-free probe reader
  (synthetic streams; mean profile, Re_tau, profile-file round trip).
- `tests/response/test_operator_tools.py`: Gramian/controllability/
  growth-curve units + `--save-operator` export faithfulness
  (eig(A), growth_curve vs the stored G) + the controllability CLI.
- `tests/response/test_ensemble.py`: harvest/build orchestration
  (real seeds, dry-run), exact antithetic aggregation on synthetic
  member streams, and direct operator identification recovery
  against a known restricted operator.
- `tests/response/test_lim.py`: LIM identification (lag-consistent
  estimator exactness, conditioning rejection, Ornstein-Uhlenbeck
  statistical recovery + Gramian covariance identity, file/CLI
  pipeline on a real operator bundle).
- `tests/response/test_ssi.py`: SSI identification (cross-covariance
  estimator exactness + causality, discrete-Lyapunov units and
  forced-variance prediction, statistical file/CLI pipeline on a
  real operator bundle, forcing-reader error paths).
- `tests/test_snapshot.py`: snapshot round-trips, np-agnostic resume,
  standard-tools readability, 9-component viscoelastic case.
- `tests/test_resume.py`: snapshot lineage and resume policy (offline
  units + mpirun integration; `--unit-only`).
- `tests/test_snapshot_import.py`: `scripts/snapshot_import.py`
  native-contract validation (offline).
- `tests/test_snapshot_export.py`: `dnsjax.analysis` API vs solver
  ground truth (JAX-free import guarantee, curl parity).
- `tests/test_rolls_smoke.py`: localized-rolls IC integration for the 5
  wall-bounded flows (incl. a multi-device padding case).
- `tests/test_localized_rolls.py`: rolls construction self-test (no-slip,
  determinism, device-count independence, divergence bound).
- `tests/test_transient_growth.py`: transient-growth analysis (JAX-free
  host units, per-flow hooks, CLI features, PP/PC/pipe/TC/quasi-keplerian
  anchors -- the QK anchor pins the axially-periodic quasi-Keplerian
  optimal growth G_opt=13.04 -- plus an m0-wedge-vs-full-circle
  equivalence check).
