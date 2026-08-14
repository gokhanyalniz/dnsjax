## Project Overview

`dnsjax` is a GPU-accelerated pseudo-spectral + finite-differences DNS
solver for the 3D incompressible Navier-Stokes equations, written in
JAX. Flow systems: triply-periodic (Kolmogorov)
and wall-bounded (plane-Couette, plane-Poiseuille, pipe, Taylor-Couette,
quasi-Keplerian, force-driven Dean, and two viscoelastic (sPTT) flows
with a coupled conformation tensor: axially driven pipe and
azimuthally driven Dean). Two selectable second-order time
integrators (`step.scheme`): the predictor-corrector `"iterative-cn"`
(default) and `"cnab2"`; see the "Time-stepping scheme" note below.

## Commands

### Prerequisites

Python >=3.14, `uv`, MPI (for multi-device runs).

### Setup

`uv sync` (also installs the `dnsjax` console script into `.venv/bin`)

### Lint

`uv run ruff check --fix`
`uv run ruff format src tests scripts`

The commit hook (`prek.toml`) runs both. Do not bare `uv run ruff
format`: it also reformats `README.md`'s code blocks, which
deliberately lag. Line length is 79 for **all** lines (ruff
`line-length = 79`, E501), not only docstrings/comments.

### Run tests

Tests are standalone scripts (`uv run python tests/test_*.py`);
`tests/pytest_suite.py` is the subprocess bridge over them. Markers,
why pytest must never *import* a script, and the live-output
plumbing: its module docstring.

`uv run pytest -m "not slow and not mpi"`  the offline loop
`uv run pytest -m "not slow"`              + the two quick mpirun rows
`uv run pytest`                            everything

Background a long run and tail it for progress -- and kill it early
when something is clearly wrong instead of waiting out the timeout.

Let a backgrounded run's own completion signal reach you;
**never poll for it** with `until ! pgrep -f "tests/test_x"`. The
polling shell's own command line contains the pattern, so `pgrep -f`
matches itself, the condition never goes false, and the loop
outlives the run it was watching. Queue several suites by chaining
them with `&&` inside one backgrounded command, not by waiting in
between. Where a process check is genuinely unavoidable, bracket the
pattern so it cannot match its own text (`tes[t]_x`) and cap the
iteration count.

Run **one** heavy suite at a time. Each invocation is already serial
internally (no xdist; the smoke suites loop one `mpirun` at a time),
and the in-process ones deliberately leave JAX's CPU thread pool
unpinned (the reason, and why not to "fix" it:
`configure_jax_platform` in `bootstrap.py`; the `mpirun` children go
through `configure_jax_runtime` instead, which pins one thread per
rank). Concurrent invocations oversubscribe instead, and have produced
spurious aborts (a signal-6 in an `mpirun` child; a smoke entry that
failed once and passed identically on rerun). Corollaries: never
read a *verdict*
from a `tail` of a run you have not seen in full -- capture it
(`> log 2>&1`) and grep the file; never change **behaviour** while a
suite is running -- each script is launched as a subprocess, so later
ones would run different semantics than earlier ones and the verdict
covers no single tree (docstrings and comments are fine: they change
nothing the run measures); and a failure that does not
reproduce on a clean serial rerun was contention -- say so rather
than silently re-running.

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
counts devices, not processes; launch recipe, SLURM discipline, and
the per-task-visibility trap: the `Distribution` docstring in
`parameters.py`.

`mpirun -np 2 .venv/bin/dnsjax --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 27`

For double parallelisation on the default CGL grid (pick `ny`
divisible by `np0`; every auto-pad -- spectral, physical, padded-size
rounding, Pallas tiles -- prints a startup diagnostic):

`mpirun -np 4 .venv/bin/dnsjax --dist.np0 2 --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 28`

The laminar state should step with a single corrector, stepping error
O(-18) or less and perturbation energy O(-32) or less. Run the tests in
the Tests section below if you touch the modules they test.

### Initial conditions

In-process only (no offline script). The four modes and their
precedence: the `Initiation` docstring (`parameters.py`). Knobs:
`--init.snapshot`, `--init.start_from_laminar`,
`--init.localized_rolls` + `_amplitude`/`_width`/`_wavelength`
(wall-bounded only), `--init.random_field` (the **default**) +
`--init.random_amplitude` / `_smoothness` / `_seed` / `_mean_flow` /
`_conformation_amplitude`. Construction, per-geometry pairing, the
sharded/padded-mesh-safe builds, and the divergence/Hermitian
caveats: the `ic/random_field.py` and `ic/localized_rolls.py` module
docstrings.

### Triggering transition to turbulence

A flow that decays is a *regime/time* matter, **not** a solver bug (the
linear physics is reproduced quantitatively): transition develops over
O(100) advective units (smoke tests only reach `t=1`) and needs `Re`
above sustainment (plane-Couette ≳ 350-500), a domain ≳ the minimal flow
unit, and a finite-amplitude perturbation (random `amplitude ≈ 0.1-0.2`,
or a strong localized spot). Near-minimal boxes give transient
(decaying) turbulence.

## Documentation instructions

Keep docstrings, comments (in LaTeX for math for both) and typing
up-to-date. In the future MkDocs will be used with MathJax, escape
LaTeX commands appropriately (prefer raw docstrings: `\t`/`\f` in
non-raw strings silently become control characters). Keep
documentation lines in code to 79 characters wide. Keep CLAUDE.md
files up-to-date (root and subdirectory files). The **four**
`README.md` files -- root plus `extensions/`, `twin/` and
`analysis/response/` -- are human-facing and may lag the code, so treat
the CLAUDE.md files and code docstrings as authoritative and do not
sync code to a README.

**Documentation layering.** CLAUDE.md files are an index for AI agents,
not a manual. A line earns its place only if it is (a) a command to
copy-paste, (b) a pointer/index entry, or (c) a constraint spanning ≥2
modules. Anything explaining *how* or *why* a single module works goes
in that module's docstring, and a CLAUDE.md line never restates a
docstring it points at — point at it instead. Algorithms, array shapes,
mathematical formulations, and per-function behaviour are always code
docs. Tests follow the same rule: one line per test file in the Tests
section below; what a test covers lives in its module docstring.

## Architecture

### Package layout (`src/dnsjax/`)

Per-module detail lives in each module's docstring; one line each here.

```
__main__.py           Entry point main() (console script `dnsjax` +
                      python -m), stats buffering, snapshot
                      resume/lineage
bootstrap.py          Shared entry-point setup for solver, scripts and
                      tests: resolve_parameters, configure_jax_runtime,
                      configure_jax_platform/platform_from_argv
parameters.py         Pydantic parameter models (JAX-free); singletons
                      params, derived_params, padded_res;
                      trajectory_defining_changes; round_up_padded
flow_spec.py          JAX-free FieldSpec/DeferredSpec/FlowSpec
                      dataclasses (per-flow surface declarations)
param_surface.py      Per-flow surface models (CLI/TOML), aliases,
                      internalize/externalize, recorded_params_dump,
                      sample-TOML + startup-printout renderers
extensions/
  __init__.py         ParamExtension registry + built-in [probes] and
                      [force] section models; singletons
                      probes_params/force_params
  probes.py           Spectral-mode probe stream ([probes] extension):
                      sharded-gather extractor + probes.bin writer
  forcing.py          White-in-time stochastic mode kicks ([force]
                      extension): sharded scatter-add injector (the
                      extractor's dual) + forcing.bin coefficient log
sharding.py           Multi-device (np0, np1) mesh; singleton sharding;
                      register_dataclass_pytree; layouts + specs
operators.py          Wavenumber helpers (re-exports harmonics.py in
                      jnp.asarray; pad_harmonics), FFT wrappers
harmonics.py          Stdlib/NumPy-only (JAX-free) wavenumber
                      sequences + parse_mode_pairs; leaf shared with
                      dnsjax.analysis
fft.py                3D/2D real FFT, 3/2-rule dealiasing, shard_map
                      reshard pipeline, spectral padding
rhs.py                Rotational-form perturbation nonlinear term;
                      measure_fn hook
measurements.py       Physical-space measurements (get_cfl)
timestep.py           make_stepper() factory:
                      predict_and_fully_correct (+_measured),
                      step_cnab2 (+_measured), _cnab2_lbf_core
adaptive.py           JAX-free CFL time-step controller
                      (propose_dt) behind step.adaptive
fd.py                 NumPy-only FD utilities (JAX-free): Fornberg
                      D1/D2, quadrature rules, interpolation matrices,
                      tanh grids
solvers.py            Geometry-independent linear solvers:
                      DenseJAXSolver (reference/oracle) and
                      PerModeBandedPallasOperator (production banded
                      sweep)
snapshot.py           Single-file (tar/zarr3) snapshot save/load, raw
                      offset I/O (GDS or host); assemble_local_shards
snapshot_meta.py      Stdlib-only (JAX-free) snapshot tar metadata
                      helpers
ic/
  random_field.py     Random divergence-free IC generators
                      (init.random_field, the default start mode)
  localized_rolls.py  Deterministic localized-spot IC generators
                      (init.localized_rolls)
twin/
  driver.py           dnsjax-twin console script (also python -m
                      dnsjax.twin): lockstep twin-run (predictability)
                      driver, [twin] extension, paired
                      snapshots/resume, twin.dat streams
  diagnostics.py      Difference-field diagnostics: component masks,
                      energies, the 27-term budget, (kz,kx) spectra
  spectra.py          TwinSpectraStream: twin_spectra.bin writer
                      (reader dnsjax.analysis.twin.spectra)
geometries/
  wall_bounded/       _base.py, cartesian.py, cylindrical.py,
                      annular.py, the three
                      _*_primitive_imm.py legacy-IMM siblings,
                      _viscoelastic_common.py,
                      _viscoelastic_stepping.py,
                      cylindrical_viscoelastic.py,
                      annular_viscoelastic.py -- see
                      wall_bounded/CLAUDE.md
  triply_periodic/    triply_periodic.py -- see its CLAUDE.md
flows/
  registry.py         JAX-free flow-spec registry: SPECS, spec_for,
                      all_systems, *_systems lists, GLOBAL_FIELDS,
                      internalize_stored/stored_value
  wall_bounded/       plane_couette, plane_poiseuille, pipe,
                      viscoelastic_pipe, taylor_couette,
                      quasi_keplerian (both bind the shared
                      _circular_couette.py machinery), dean,
                      viscoelastic_dean -- base flows/driving in
                      wall_bounded/CLAUDE.md; specs/ holds their
                      JAX-free parameter FlowSpecs
  triply_periodic/    monochromatic.py: Kolmogorov;
                      specs/ holds their JAX-free parameter FlowSpecs
analysis/             External-facing JAX-free snapshot post-processing
                      API (+ the JAX-based transient_growth CLI and the
                      response/ and twin/ subpackages) -- see
                      analysis/CLAUDE.md
```

### Twin-run perturbation growth (`dnsjax-twin`)

`dnsjax-twin` steps a reference snapshot and a perturbed copy
(random divergence-free field of exact energy `twin.e0`) in lockstep
and streams difference-field diagnostics: component energies
(`twin.dat`), the production/transport/dissipation budget
(`twin_budget.dat`, `twin.it_budget`), and (kz,kx) energy spectra
(`twin_spectra.bin`, `twin.it_spectra`). Cartesian wall-bounded
flows, fixed dt, launched like the solver (mpirun, scratch dir):

`mpirun -np 1 .venv/bin/dnsjax-twin --init.snapshot parent.tar
--twin.e0 1e-6 --twin.seed 3 --stop.max_sim_time <t_parent + 10>`

Start/resume rules (partner snapshot + `twin.json` decide; a resume
never re-perturbs), stream formats, and the frame-invariance /
dissipation-form notes: the `twin/driver.py` and
`twin/diagnostics.py` module docstrings. Ensembles:
`ensemble_setup.py build-twin` +
`dnsjax.analysis.twin`.

### Transient-growth analysis

`python -m dnsjax.analysis.transient_growth` computes 3D linear optimal
energy growth `G(t)` around an arbitrary wall-normal **total** profile
`U(y)` for the five base-flow wall-bounded flows (plane-couette/
poiseuille, pipe, taylor-couette, quasi-keplerian; Dean out of scope),
reusing the solver's own linear step per Fourier mode. Single-device,
GPU-runnable (`--dist.platform cuda`). It parses the shared per-flow
surface (`bootstrap.resolve_parameters`; public names, strict) plus its
own `[tg]`/`--tg.*` extension section (`TGParams`). **`G_max` needs a
converged wall-normal resolution**: at an unconverged `ny`/`nr` the
reported optimum is an artefact (why, and the recipe: the module
docstring's "Converging N_y" section). `--tg.save_operator`
additionally exports each mode's reduced generator
(`<stem>_tg_op.npz`) for the
`analysis/response/` post-processing. Math, the `frozen_profile_flow`
hook, the CLI/output spec, and the `--tg.dt` trade-off: the module
docstring and `analysis/CLAUDE.md`.

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

To add a flow `X`: (1) create `flows/<family>/specs/X.py` with a
JAX-free `FlowSpec` (surface fields, defaults, hooks, `flow_module`,
`n_components` — the `flow_spec.py` docstrings; the existing specs and
`_family.py` fragments are the template) and add it to the `SPECS`
tuple in `flows/<family>/specs/__init__.py` (`flows/registry.py`'s
`SPECS` is derived from those, not edited) — this auto-extends the
`phys.system` Literal, the `--help`/TOML surface, `--sample-toml`,
snapshot metadata, the flow dispatch, and the `analysis/_core.py`
frozensets; (2) create the
`flow_module` (`flows/<family>/X.py`) exporting the stepping surface
(the function list: the `FlowSpec.flow_module` docs; its
`get_perturbation_energy` is the cheap `E'` laminarization read);
(3) add SYSTEMS entries to `tests/test_laminar_smoke.py` (a flow
without a perturbation `E'` needs its own check branch) and
`tests/test_random_smoke.py` (pick a Reynolds number above transition
onset, small domain).

A flow whose state is not the 3 velocity components (e.g. the
9-component viscoelastic state) declares it via `FlowSpec.n_components`
(the snapshot writer/loader reads it) and needs an `analysis/_core.py`
component schema (`geometry_info`); the IC builders and the
FFT/sharding/stepper machinery are component-count-agnostic (leading
state axis replicated).

### Key design patterns

**Global singletons and import order**: `params`, `derived_params`,
`padded_res` (`parameters.py`), `sharding` (`sharding.py`), and a
geometry `fourier` are module-level singletons captured at import time
-- so JAX must be configured and parameters final (`update_parameters()`)
*before* importing `sharding` or any geometry module (the setup
contract: the `bootstrap.py` module docstring). Earlier-importable
modules (e.g. `ic/random_field.py`) keep `import jax` out of module
scope.
`fourier`'s wavenumber arrays are global multi-device arrays -- host
code recomputes them from `harmonics.real_harmonics`/`complex_harmonics`
× `2π/L`, never `np.asarray`.

**Stepper factory (two layers)**: `timestep.make_stepper()` builds the
JIT-compiled stepping functions from geometry-general callables; each
geometry family wraps it in a builder that binds the `fourier`/`flow`
singletons. See the `make_stepper` docstring, `_base.py`, and
`triply_periodic.py`.

**Time-stepping scheme (`step.scheme`)**: both 2nd-order, sharing the
predictor and the geometry's IMM implicit solve. `"iterative-cn"`
(default) makes the nonlinear term implicit via the corrector fixed
point (stable past the advective CFL); `"cnab2"` advances it
explicitly (AB2) at **one** FFT eval/step, and wall-bounded keeps the
wall-stiff coupling `_l_bf` implicit via an FFT-free corrector.
Full detail (measured `dt` limits,
per-geometry CFL, `implicitness`, `implicit_mean_coupling`,
`split_corrector`, and the corrector-contraction `dt` limit — a
`corrector failed to converge` at *low* CFL means reduce `dt`, not a
blow-up): the `TimeStepping` docstring (`parameters.py`); implementation
`timestep.py`; guards `tests/test_cnab2.py`,
`tests/test_temporal_order.py`. Adaptive CFL `dt` (`step.adaptive`,
knobs + controller law in the `TimeStepping` docstring and
`adaptive.py`; on-device operator rebuild via the builders'
`set_dt`, no recompile): guards `tests/test_adaptive.py` and the
vardt studies in `tests/test_temporal_order.py`.

**Spectral array layout and sharding**: see the `sharding.py` module
docstring for shapes, partition specs, and the `(np0, np1)` device
mesh. See the `fft.py` module docstring for the reshard pipeline and
spectral padding.

**Perturbation formulation**: the solver evolves `u'` around laminar
`U(y)` (`rhs.py` module docstring). The force-driven
dean/viscoelastic-dean/viscoelastic-pipe systems instead integrate the
**total** field (`base_flow = 0`, mean-mode body force).

**Component basis (cylindrical/annular only)**: the state is carried
in the decoupled `u_±`/spin solver basis and *observed* in physical
components — so **anything handing a freshly built (physical) state to
a stepper must convert first**, and a state crosses at most once,
never back. Cartesian and triply-periodic carry physical components
always. Rules and rationale: `geometries/wall_bounded/CLAUDE.md`.

**Moving frame of reference (`phys.u_grid`)**: translates the
wall-bounded frame along the grid direction (`None` → laminar bulk;
periodic systems reject it), integrated implicitly by both schemes. It
does **not** relax cnab2's explicit self-advection CFL (`u'×ω'` is
frame-invariant), and a changed `u_grid` on resume is
trajectory-defining. Detail: the `u_grid` field docs (`parameters.py`)
and `pad_base_flow` (`_base.py`).

**JAX pytree registration**: `register_dataclass_pytree()`
(`sharding.py`) registers geometry, flow, solver, and Fourier classes as
JAX pytrees. See its docstring.

**Performance/memory trade-offs** (detail in the owning
docstrings/comments): `solver.backend` (pallas beats dense in storage
and speed); whole-tile mode-plane padding (`solvers.py`); the RHS
transform batch vs peak memory (`solver.rhs_transform_chunks`, applied
by `fft.chunked_transform`; the ~36-field viscoelastic batch is where it
bites); cnab2 is a throughput win, not a peak-memory one (`timestep.py`).
The dominant global memory multipliers are `phys.oversampling_factor`
(at the default 3: ~2.25× physical points wall-bounded, ~3.375×
periodic -- only periodic flows oversample `y`) and
`res.double_precision` (2×).

### Parameter layering

The layer order, the per-flow surface (`param_surface.py`) every
user-facing layer parses against, and `--help <system>` /
`--sample-toml <system>`: `bootstrap.resolve_parameters` (other entry
points pass `toml_path`/`extensions`/`prog` — the TG CLI is the
template). Aliased fields go by public names (cylindrical/annular:
`lz`/`nz`/`nr`/`ntheta` for internal
`geo.lx`/`res.nx`/`res.ny`/`res.nz`). Never inherited from a
snapshot: the JAX-setup fields `dist.np0`/`np1`/`platform` and
`res.double_precision` (chosen per run; a precision mismatch with the
snapshot rejects), the whole `[solver]` section (execution-only,
`read_snapshot_params` strips it), and the resume-decision fields
`init.snapshot`/`init.force_resume` (recorded for lineage only).
A stored **core-section** parameter this version does not define is a
hard error in `internalize_stored` — `[solver]` alone is exempt
(note-and-drop), because it is stripped only *after* internalizing.

Per-flow `FieldSpec` defaults (`phys.u_grid`, `geo.grid_type`, the
viscoelastic rheology values, ...) are **re-materialized on every
`update_parameters()` call** unless explicitly set through a layer
(`_materialized_defaults` restore-then-materialize). Scripts and
tests must set these via `update_parameters(Parameters(...))` — a
direct `params.geo.grid_type = ...` assignment is silently
overwritten on the next pass (never enters `_user_set_fields`).

### Configuration (`parameters.toml`)

Full field documentation: the `parameters.py` model docstrings (the
`Initiation` docstring for start-mode precedence, `TimeStepping` for the
schemes, `Solver` for the Pallas knobs, `Distribution` for the launch
contract), surfaced per flow by `--help <system>` /
`--sample-toml <system>` and validated per flow as in "Parameter
layering" above.

| Section    | Purpose                                             |
|------------|-----------------------------------------------------|
| `[phys]`   | Reynolds numbers, `system`, oversampling, driving, `u_grid`; viscoelastic rheology (`el`/`wi`/`beta`/`epsilon`/`kappa`) |
| `[geo]`    | Domain lengths/tilt, `eta`, `m0` (azimuthal wedge), `delta`, wall-normal grid selection |
| `[res]`    | Resolution (`nx`/`ny`/`nz`, or `nz`/`nr`/`ntheta`), `fd_order`, `consistent_imm`, `double_precision` |
| `[init]`   | Start mode (see "Initial conditions" above) + `t0`/`it0`/`isnap0`/`force_resume` |
| `[outs]`   | Diagnostic cadences, buffering, snapshot write policy |
| `[step]`   | `dt` + scheme knobs + adaptive-CFL knobs (`TimeStepping`) |
| `[stop]`   | Sim-/wall-time limits, laminarization check          |
| `[dist]`   | `np0` (wall-normal / kz axis), `np1` (spanwise / kx axis), `platform` |
| `[solver]` | Backend selection + Pallas tiling / RHS chunking (wall-bounded; `rhs_transform_chunks` is global) |
| `[probes]` | Extension (`extensions/`): spectral-mode probe stream (wall-bounded) |
| `[force]`  | Extension: white-in-time stochastic mode kicks; all-or-none and trajectory-defining (wall-bounded, non-viscoelastic) |
| `[twin]`   | Extension (registered by `dnsjax-twin` only): twin-run seed/energy/cadences (Cartesian wall-bounded, fixed dt) |

The default `parameters.toml` contains only
`[phys] [geo] [res] [init] [outs] [step] [stop]`; the rest rely on
model defaults -- set them via CLI (e.g. `--dist.np1 2`,
`--force.modes "3,0"`, `--probes.modes "0,0;3,0"`) or by adding the
section. Analysis CLIs and scripts register further extension sections
on their own surfaces (`[tg]` for the transient-growth driver,
`[perturb]` for `scripts/snapshot_perturb.py`).

### Diagnostics (`stats.dat`, `steps.dat`, `corrector.dat`, `probes.bin`, `forcing.bin`)

Three on-device buffered scalar streams, flushed periodically
(fsync-ed): `get_stats` → `stats.dat`, the CFL diagnostic
(`outs.it_steps`, via the `rhs.py` `measure_fn` hook) → `steps.dat`, the
corrector diagnostic (`outs.it_corrector`) → `corrector.dat`; plus the
binary spectral-mode probe stream (`probes.modes`/`probes.it_probes`,
wall-bounded only) → `probes.bin` + `probes.json`
(`extensions/probes.py`; JAX-free reader
`dnsjax.analysis.response.probes`), and the stochastic-kick
coefficient log (`[force]`) → `forcing.bin` + `forcing.json`
(`extensions/forcing.py`; reader `dnsjax.analysis.response.ssi`).

Every `.dat` header row is `#`-commented (`_write_dat_header` in
`__main__.py`, shared with `twin/driver.py`; the `#` eats one space of
the first column's padding), so a bare `np.loadtxt` reads any stream —
and every reader must `lstrip("#")` the header line before splitting
it.

Every stream with a sidecar carries a `format_version` enforced
against the reader's `MIN_FORMAT_VERSION`, like the snapshot one —
four writer/reader pairs: `extensions/probes.py` and `forcing.py` →
`analysis/response/probes.py` and `ssi.py`; `twin/spectra.py` and
`twin/driver.py` (`twin.json`) → `analysis/twin/spectra.py` and
`series.py`. Bump writer and reader together when the stored *meaning*
changes (rationale: the writers' docstrings).

Every flushed row and host-synced scalar is guarded against NaN/inf: a
hit prints one `FATAL: non-finite ...` line naming the quantity, skips
the final snapshot, and exits with code **3**. Buffering mechanism,
flush points, file format, and the guard: the `__main__.py` module
docstring.

### Snapshots

A single uncompressed tar (`format_version: 6`) wrapping a zarr3
store; `snapshot_meta.read_snapshot_meta` rejects `< 6` (no
translation of old snapshots, by design). Archive layout, the
standard-tools contract, the native no-transpose byte layout, the raw
offset I/O / GDS write path, the `.partial` commit-by-rename (which
also makes `*.tar` globs skip an interrupted save), and the metadata
(incl. the writing code's git hash): the `snapshot.py` and
`__main__.py` module docstrings.

Cross-cutting: the stored components are the **physical** ones for
every family — the cyl/annular `u_±`/spin working basis is converted
at the write/read boundary (`wall_bounded/CLAUDE.md`) — and the stored
state is the spectral perturbation `u'` for base-flow systems (laminar
= zero array), the **total** field for dean/viscoelastic-dean/
viscoelastic-pipe. The embedded `params` dump is the flow-relevant,
resolved, **public-named** surface representation plus the relevant
extension sections (`param_surface.recorded_params_dump`); readers map
it back via `flows.registry.internalize_stored` / `stored_value`.

Resume is np-agnostic (precision must match — a mismatch rejects) and
re-grids a changed wall-normal grid at load; `t`/`it`/`isnap` continue
only when
`trajectory_defining_changes(meta["params"])` is empty — a `phys`/`geo`/
`res` override or a `[force]` change starts a **new trajectory** unless
`init.force_resume` (distinct from the hard resolution/system/precision
rejects).

### JAX-specific notes

- Explicit mode sharding is used globally rather than Auto mode. Do
  not use `jax.lax.with_sharding_constraint`.
- Allocate sharded arrays directly on devices (`out_sharding` argument
  of `jnp.zeros`, `.at[...].get/set`, etc.) instead of allocating
  globally and redistributing with `jax.device_put`; when direct
  allocation is not possible, do not substitute `jnp.asarray` for
  `jax.device_put`.
- **Resharding an existing multi-device array** (vs a host→device
  ingest, the `device_put` above): do it **inside `jax.jit`** with
  `jax.sharding.reshard`, moving **one mesh axis per step** -- eager
  `device_put` redistributes piecewise (338× slower than jitted),
  and moving both axes at once replicates the whole array on every
  device (`Involuntary full rematerialization`; shows only at
  `np0 > 1, np1 > 1`, so audit on a `(2, 2)` mesh). Jitting costs a
  compile: for reshards that run repeatedly, not once-per-run ones.
  Pattern + numbers: `snapshot.py`'s `_via_mid` /
  `_to_io_layout_core`; codebase-wide audit recipe:
  `~/.claude/plans/reshard-audit-jitted-collectives.md`.
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
- The wall-bounded per-mode solve dispatches on the **live backend**:
  `solvers._kernel_path()` picks both the solve body and the factor
  *storage*, so they cannot disagree. A CPU run never reaches
  `pallas_call`; it sweeps in pure JAX and stores the plain `U`
  diagonal at the true mode plane, where the kernel path stores the
  reciprocal and pads to whole tiles. The mode-inner **layout** is
  shared and is measured-optimal on both -- three CPU-native layouts
  are faster in isolation and slower end to end, monotonically. A test
  flipping `_force_kernel_path` must do so *before* building the
  operator. Numbers and the trap: the `solvers.py` docstrings -- do not
  re-derive from an isolated solve timing.
- Pallas/Triton GPU kernels: interpret mode (CPU) validates numerics
  but **not** Triton's lowering; compile-check on the GPU-less dev box
  by lowering for cuda **inside an abstract GPU mesh** (since JAX 0.11
  a bare `lower(lowering_platforms=("cuda",))` raises `No supported GPU
  devices found` -- Triton reads the compute capability off the mesh
  context's abstract device). Recipe, and the extra mesh swap a
  `shard_map`-wrapped kernel needs: `_abstract_gpu_mesh` in
  `tests/test_banded_solver.py`. The
  lowering/layout rules and the partial-tile miscompile (pad tiled
  arrays to whole tiles): the `_pallas_banded_solve` docstring; the
  `check_vma=False` a `pallas_call` inside a `shard_map` needs:
  `PerModeBandedPallasOperator.solve`. Guards
  `test_pallas_cuda_lowering` and
  `test_pallas_cuda_lowering_sharded_solve` (both in
  `tests/test_banded_solver.py`).

## Scripts

All under `scripts/`; full rationale/usage in each module docstring.

- `snapshot_import.py`: **library** (not a CLI) packing a
  native-layout velocity field into a snapshot.
- `snapshot_perturb.py`: CLI + library injecting a scaled single-mode
  perturbation into an existing snapshot.
- `ensemble_setup.py`: JAX-free `harvest`/`build`/`build-twin` CLI
  building ensemble member run trees from a snapshot archive.
- `wall_normal_resolution.py`: JAX-free `resolve`/`match`/`box` CLI
  sizing `res.ny`/`fd_order`/`geo.grid_type` against a Chebyshev
  expansion of a given order (Cartesian family only).
- `pallas_tiling_diagnostic.py`: GPU harness for the Triton
  partial-tile miscompile (localised it; confirms the fix).
- `pallas_solve_profile.py`: GPU diagnostic for the Pallas banded
  solve's time breakdown (`--cpu-smoke`).
- `solver_benchmark.py`: pallas-vs-dense validation & benchmark incl.
  multi-GPU correctness (`--cpu-bench`, `--cpu-smoke`).
- `gds_probe.py`: cluster diagnostic for the snapshot GDS path
  (`--env-only`, `--end-to-end`, `--end-to-end-only`, `--cpu-smoke`).

## Tests

All under `tests/`, to be kept up-to-date as the respective modules
change. What each covers lives in its module docstring; entries here
are one-liners. Cross-cutting notes:

- The laminar smoke test has `u' = 0`, so it does **not** exercise the
  rotational nonlinear term (a wrong advection change can still report
  `err=0`); `test_random_smoke.py` drives that path.
- In-process geometry tests configure the singletons **once** at
  module top (`test_cylindrical.py`/`test_annular.py`/
  `test_viscoelastic.py`/`test_viscoelastic_pipe.py` via
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

- `pytest_suite.py`: the pytest bridge — see Run tests.
- `_live.py`: `run_live` (tee-ing subprocess runner) + `report`
  (summary that re-prints each failure after the counts). Use both in
  any script whose children stream output.
- `response/_common.py`: shared fixtures of the response
  identification tests.
- `test_banded_solver.py`: geometry-independent Pallas banded backend.
- `test_banded_solver_sharded.py`: shard_map-local Pallas solve on a
  forced (2, 2) mesh.
- `test_cartesian.py`: Cartesian operators + band-vs-dense parity.
- `test_cylindrical.py`: cylindrical operators + band-vs-dense parity.
- `test_annular.py`: annular operators + band-vs-dense parity.
- `test_viscoelastic.py`: the annular sPTT geometry.
- `test_viscoelastic_pipe.py`: the cylindrical sPTT geometry (both
  also carry the no-cross-geometry-import guard).
- `test_integration.py`: quadrature weights and interpolation matrices.
- `test_cnab2.py`: CN/AB2 + split-corrector structural guards.
- `test_imm_continuity.py`: stepped-state discrete divergence, default
  vs legacy `res.consistent_imm` (`--ny`).
- `test_energy_budget.py`: stepped total-energy budget closure.
- `test_adaptive.py`: the adaptive-dt machinery.
- `test_temporal_order.py`: second-order temporal accuracy, fixed and
  variable step.
- `test_mean_mask.py`: `mean_mask` is the unique k²=0 mode under
  forced spectral padding.
- `test_monochromatic.py`: Kolmogorov `get_stats` identities.
- `test_padding.py`: padded-size rounding + FFT exactness +
  `chunked_transform` bit-parity.
- `test_laminar_smoke.py`: laminar fixed-point smoke, all wall-bounded
  flows (subprocess/mpirun; `--np`/`--np0`).
- `test_random_smoke.py`: random-IC nonlinear integration for the 7
  distinct stepping machineries + the scheme/flag variants (`--np`).
- `test_quasi_keplerian.py`: quasi-keplerian control parameters and
  the azimuthal wedge.
- `test_param_surface.py`: flow-spec registry + per-flow surface
  machinery.
- `test_probes.py`: runtime spectral-mode probe stream
  (`--unit-only` skips the mpirun runs).
- `test_forcing.py`: runtime stochastic kicks (`--unit-only`).
- `test_snapshot_perturb.py`: `scripts/snapshot_perturb.py` injection.
- `response/test_probes_reader.py`: JAX-free probe reader.
- `response/test_operator_tools.py`: Gramian/controllability/
  growth-curve units + the `--tg.save_operator` export.
- `response/test_ensemble.py`: harvest/build orchestration, antithetic
  aggregation, direct operator identification.
- `response/test_lim.py`: LIM identification.
- `response/test_ssi.py`: SSI identification.
- `test_snapshot.py`: snapshot round-trips, np-agnostic resume, the
  multi-device I/O layout, and the integrity guards.
- `test_resume.py`: snapshot lineage and resume policy (`--unit-only`).
- `test_snapshot_import.py`: `scripts/snapshot_import.py`
  native-contract validation (offline).
- `test_snapshot_export.py`: `dnsjax.analysis` API vs solver ground
  truth (**re-run when changing a primitive**).
- `test_rolls_smoke.py`: localized-rolls IC integration, 4 variants.
- `test_localized_rolls.py`: IC construction self-test (rolls + the
  random-field divergence guard).
- `test_transient_growth.py`: transient-growth analysis, incl. the
  per-flow literature anchors (`--fast` skips them).
- `test_twin_unit.py`: twin diagnostics on a (2,2) mesh.
- `test_twin_driver.py`: `dnsjax-twin` integration via mpirun
  (`--only <frag>` runs a subset).
- `test_twin_analysis.py`: JAX-free `analysis.twin` readers/
  aggregation/fits/lengths + `build-twin` end to end.
