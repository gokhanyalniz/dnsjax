# dnsjax: notes for coding agents

`dnsjax`: GPU-accelerated pseudo-spectral + finite-difference DNS of
the 3D incompressible Navier-Stokes equations, in JAX. Two independent
geometry families: triply-periodic (Kolmogorov) and wall-bounded
(plane Couette and Poiseuille, pipe, curved pipe, Taylor-Couette,
quasi-Keplerian, Dean, and the sPTT viscoelastic pipe and Dean). Human
docs: `README.md`, `docs/`, `CONTRIBUTING.md`.

## Commands

Prerequisites: Python >=3.12 (`.python-version` pins 3.14), `uv`, and
MPI only for multi-process runs (a lone process needs none, even across
GPUs). Moving the floor: `CONTRIBUTING.md` "Python versions".

- `uv sync`: the environment, plus the `dnsjax` and `dnsjax-twin`
  console scripts in `.venv/bin`.
- `uv run ruff check --fix` and `uv run ruff format src tests scripts`
  (never a bare `ruff format`: it also rewrites the Python blocks in the
  Markdown docs and the example notebook, laid out for reading). Lines
  are at most 79 columns. `prek.toml` (at commit) and
  `.claude/settings.json` (on each edited `.py`) run both.
- `uv run pytest -m "not slow and not mpi"` (the offline loop, ~22
  min), `-m "not slow"` (adds the quick `mpirun` rows), `uv run pytest`
  (everything); one script: `uv run python tests/<name>.py`. Which
  script a change needs: `tests/CLAUDE.md`.
- Figures: `uv run --group plots python scripts/<name>.py`
  (`scripts/CLAUDE.md`).
- A flow's parameter surface: `uv run dnsjax --help <system>` and
  `uv run dnsjax --sample-toml <system>`.

## Running the solver

- Launch from a scratch directory: a run reads `./parameters.toml` and
  writes its `.dat` streams and snapshots to the cwd. Templates:
  `examples/*/parameters.toml`.
- Several processes: `mpirun -np N .venv/bin/dnsjax ...` (`uv run` does
  not compose with `mpirun`; `python -m dnsjax` is the same entry
  point). One process: `uv run dnsjax ...`; it may span several GPUs
  (`--dist.np0 4`) but not several CPU devices. `np0 * np1` counts
  devices. The launch contract, SLURM and the per-task-visibility trap:
  the `parameters.Distribution` docstring.
- Laminar smoke on a 1D and a 2D mesh (pick `ny` divisible by `np0`).
  Expect one corrector per step, a stepping error of O(1e-18) or less
  and `E'` of O(1e-32) or less:

  `mpirun -np 2 .venv/bin/dnsjax --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 27`

  `mpirun -np 4 .venv/bin/dnsjax --dist.np0 2 --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 28`

- Twin runs (Cartesian wall-bounded flows, fixed `dt`, launched like
  the solver; `src/dnsjax/twin/CLAUDE.md`):

  `.venv/bin/dnsjax-twin --init.snapshot parent.tar --twin.e0 1e-6 --twin.seed 3 --stop.max_sim_time 10`

- Transient growth of the base-flow wall-bounded flows:
  `uv run python -m dnsjax.analysis.transient_growth` (one device;
  `--dist.platform cuda` on a GPU). Converge `ny`/`nr` first: at an
  unconverged resolution the reported `G_max` is an artefact (the
  module docstring, "Converging N_y").
- Offline multi-device tests force CPU devices with
  `XLA_FLAGS=--xla_force_host_platform_device_count=N` and set
  `params.dist.np0`/`np1` before importing `sharding`. Never combine
  that with `mpirun`.
- A flow that decays is a matter of regime or time, not a solver bug.
  Sustained turbulence needs O(100) advective time units (smoke tests
  reach `t = 1`), `Re` above sustainment (plane Couette: ~350-500) and a
  finite-amplitude perturbation in a domain at least the minimal flow
  unit, which is itself only transiently turbulent.

## Test-run discipline

- Before running anything, ask what the change can actually reach and
  run the cheapest check that covers exactly that, often a one-line
  `uv run python -c "import ..."`. A docstring, comment or type-hint
  change needs no run.
- Run one heavy suite at a time: the in-process tests leave JAX's CPU
  thread pool unpinned on purpose (`bootstrap.configure_jax_platform`;
  do not "fix" it), so concurrent suites oversubscribe and abort
  spuriously. A failure that does not reproduce on a clean serial rerun
  was contention: say so rather than re-running silently.
- Background a long run with its output captured (`> log 2>&1`), queue
  suites with `&&` inside one background command, and read the verdict
  by grepping the log, never from a `tail` of a run you have not seen
  in full. Let the run's own completion signal reach you. **Never poll
  with `until ! pgrep -f "tests/test_x"`**: the pattern matches the
  polling shell itself. If a process check is unavoidable, bracket the
  pattern (`tes[t]_x`) and cap the iterations.
- While a suite runs, change only docstrings and comments: each script
  is a subprocess, so a behaviour change leaves the verdict covering no
  single tree.

## Documentation rules

- The agent notes (every `CLAUDE.md` and `.claude/rules/*.md`) are an
  index. A line stays only if an agent needs it before, or instead of,
  reading the code that holds the answer: a command, a rule for the
  agent's own conduct, a convention for code not yet written, an
  invariant whose two ends live in different modules, or a pointer to a
  `module.symbol` docstring. How and why one module works, measured
  numbers, history and per-function behaviour belong in its docstring;
  a note never restates one, and code never points into a note.
- Put a note in the narrowest file whose load scope covers every file
  where its mistake could be made: this root (always loaded), a
  directory's `CLAUDE.md` (loaded when a file there is read), or a
  `.claude/rules/` file whose `paths:` globs name the files it governs
  (each glob must match a tracked file). The indexes (the package map
  here, `tests/`, `scripts/`) are one line per file.
- Sizes are guides, counted at the 79-column wrap. The root aims at
  about 200 lines; every line costs every session, so grow it only with
  the utmost care, after asking whether a directory note or a rule file
  covers the files where the mistake would be made. Those load only
  with their files and have more room (about 120 lines is comfortable,
  not a limit), but the line test still decides what goes in.
- Keep docstrings, comments, type hints and these notes current, at 79
  columns. Math is LaTeX: inline `` `$...$` ``, display `.. math::`. A
  docstring containing a backslash is raw (`r"""`): in a plain one `\t`
  becomes a TAB and a trailing `\` eats its newline. In a `.md` file an
  inline `$...$` never spans a line break (GitHub renders it raw).
- The human-facing docs change in the same pass as what they describe:
  `README.md`, `CONTRIBUTING.md`, `tests/README.md`, `docs/*.md`,
  `examples/**/README.md`, and the READMEs of `extensions/`, `twin/`
  and `analysis/response/`. They are pointer-first; where one
  disagrees with a docstring, fix the docstring first.
- Nothing committed carries a placeholder, an invisible marker or a
  claim whose backing artefact has not landed: no `TODO(author)`, no
  `DRAFT:`, no empty section, no badge for a DOI that does not exist.
  Omit what cannot be finished truthfully.

## Package map (`src/dnsjax/`)

JAX-free leaves, importable before JAX is configured: `parameters.py`,
`flow_spec.py`, `harmonics.py`, `fd.py`, `adaptive.py`,
`snapshot_meta.py`, `seeding.py`, `flows/registry.py`, the flow
`specs/`, and `analysis/` (bar the members its notes name). Reuse them
(extract a helper into one rather than copying it), and put a module
used only outside the solver in a subpackage, not at this level.

```
__main__.py       the `dnsjax` entry point: run loop, streams, resume
bootstrap.py      setup for every entry point: parameters, JAX, ranks, seeds
parameters.py     pydantic models; params / derived_params / padded_res
flow_spec.py      FieldSpec / DeferredSpec / FlowSpec
param_surface.py  per-flow CLI/TOML surfaces, public names, params dump
extensions/       [probes] and [force]: sections and their streams
sharding.py       the (np0, np1) mesh singleton, layouts, pytree registration
fft.py            real FFTs, 3/2 dealiasing, shard_map reshards, padding
operators.py      wavenumbers (from harmonics.py), FFT wrappers
rhs.py            rotational nonlinear term; measure_fn and metric hooks
measurements.py   CFL
timestep.py       make_stepper: the iterative-cn and cnab2 steppers
adaptive.py       CFL time-step controller
fd.py             FD weights, quadrature, interpolation, grids
solvers.py        dense (reference) and banded Pallas per-mode solves
snapshot.py       snapshot save/load (snapshot_meta.py: tar metadata)
seeding.py        the seed contract
ic/               random_field (the default IC), localized_rolls, mean_mode
twin/             dnsjax-twin
geometries/       wall_bounded/, triply_periodic/
flows/            registry.py; per-family flow modules and specs/
analysis/         snapshot API, transient_growth, snapshot_import,
                  response/, twin/
```

## Invariants everywhere

- The two geometry families are independent: `geometries/` and
  `flows/` each split into `wall_bounded/` and `triply_periodic/`, and
  nothing crosses. Do not read across families unless asked.
- `params`, `derived_params`, `padded_res`, `sharding` and each
  geometry's `fourier` are module singletons built at import.
  Configure JAX and finalize the parameters (`update_parameters()`)
  before importing `sharding` or any geometry module. Every entry point
  (solver, scripts, tests) does that through `dnsjax.bootstrap`, whose
  module docstring is the contract; never inline `jax.config.update` /
  `jax.distributed.initialize`. Modules importable earlier keep
  `import jax` out of module scope.
- Base-flow systems evolve and store the perturbation `u'` about the
  laminar `U(y)`; the force-driven systems (`FlowSpec.total_field`: the
  curved pipe, Dean, and the viscoelastic pipe and Dean) evolve and
  store the total field.
- Cylindrical and annular states are carried in a solver basis and
  observed in physical components: convert a freshly built physical
  state before stepping it (`geometries/wall_bounded/CLAUDE.md`).
- A `dt`, resolution or parameter sweep needs one process per value:
  the value is captured in the singletons and the jitted steppers.
- Pre- and post-processing belong in a `scripts/` tool that meets the
  solver through snapshots, not in a new solver parameter or runtime
  path.

## Other notes

- `.claude/rules/`, each loaded when you read a file it governs (read
  one directly before working in its area if you have not opened such
  a file yet): `jax.md` (any Python), `stepping.md`, `parameters.md`,
  `initial-conditions.md`, `snapshots.md`, `diagnostics.md`,
  `pallas.md`.
- Directory notes (`CLAUDE.md`): `tests/`, `scripts/`, and in
  `src/dnsjax/`: `geometries/wall_bounded/`, `geometries/triply_periodic/`,
  `flows/`, `twin/`, `analysis/`.
