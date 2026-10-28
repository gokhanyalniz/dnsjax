# dnsjax: notes for coding agents

`dnsjax`: GPU-accelerated pseudo-spectral + finite-difference DNS of
the 3D incompressible Navier-Stokes equations, in JAX. Two independent
geometry families: triply-periodic (Kolmogorov) and wall-bounded
(plane Couette and Poiseuille, pipe, Taylor-Couette; in development
in the lab, `lab/`: curved pipe, quasi-Keplerian, Dean, and the sPTT
viscoelastic pipe and Dean).
Human docs: `README.md`, `docs/`, `CONTRIBUTING.md`.

## Commands

Prerequisites: Python at `pyproject.toml`'s `requires-python` or newer
(`.python-version` pins the newest release; both, and moving them:
`CONTRIBUTING.md` "Python versions"), `uv`, and MPI only for
multi-process runs (a lone process needs none, even across GPUs).

- `uv sync`: the environment, plus the `dnsjax`, `dnsjax-info` and
  `dnsjax-twin` console scripts in `.venv/bin`.
- `uv run ruff check --fix` and `uv run ruff format src tests scripts
  lab/src lab/tests lab/scripts` (never a bare `ruff format`: it also
  rewrites the Python blocks in the Markdown docs and the example
  notebook, laid out for reading). Lines are at most 79 columns.
  `prek.toml` runs both at commit.
- `uv run pytest -m "not slow and not mpi"` (the offline loop, a heavy
  suite), `-m "not slow"` (adds the quick `mpirun` rows), `uv run pytest`
  (everything), each over both suites; one script: `uv run python
  tests/<name>.py` (`lab/tests/` for the lab's). Which script a change
  needs: `tests/CLAUDE.md`.
- Figures: `uv run --group plots python scripts/<name>.py`
  (`scripts/CLAUDE.md`).
- A flow's parameter surface: `uv run dnsjax --help <system>` and
  `uv run dnsjax --sample-toml <system>`.
- A snapshot's clock, parameters and stats, its state unread: `uv run
  dnsjax-info [--time] [--provenance] [--stats] [--raw] state*.tar`
  (`--raw` for JSON; the `tools.snapshot_info` docstring).

## Running the solver

- Launch from a scratch directory: a run reads `./parameters.toml` and
  writes its `.dat` streams and snapshots to the cwd. Templates:
  `examples/*/parameters.toml`.
- Stop a live run with `rm RUNNING` (or SIGUSR1): it ends on a final
  snapshot. Status files, exit codes: the `run_status` docstring.
- Several processes: `mpirun -np N .venv/bin/dnsjax ...` (`uv run` does
  not compose with `mpirun`; `python -m dnsjax` is the same entry
  point). One process: `uv run dnsjax ...`; it may span several GPUs
  (`--dist.np0 4`) but not several CPU devices. `np0 * np1` counts
  devices. The launch contract, SLURM and the per-task-visibility trap:
  the `parameters.Distribution` docstring.
- Laminar smoke on a 1D and a 2D mesh (any `ny` runs; one that `np0`
  divides skips the y auto-pad).
  Expect `c/it = 0.00` (`c` counts corrections past the first), a
  stepping error of O(1e-18) or less and `E'` of O(1e-32) or less:

  `mpirun -np 2 .venv/bin/dnsjax --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 27`

  `mpirun -np 4 .venv/bin/dnsjax --dist.np0 2 --dist.np1 2 --phys.system plane-couette --init.start_from_laminar True --stop.max_sim_time 0.04 --outs.it_stats 1 --res.nx 4 --res.nz 4 --res.ny 28`

- Twin runs (the lab's; Cartesian wall-bounded flows, fixed `dt`,
  launched like the solver; `lab/src/dnsjax_lab/twin/CLAUDE.md`):

  `.venv/bin/dnsjax-twin --init.snapshot parent.tar --twin.e0 1e-6 --twin.seed 3 --stop.max_sim_time 10`

- Transient growth (base-flow wall-bounded flows, one device):
  `uv run python -m dnsjax.analysis.transient_growth`. Converge `ny` /
  `nr` first: an unconverged `G_max` is an artefact (the module
  docstring, "Converging N_y").
- A flow that decays is a matter of regime or time, not a solver bug.
  Sustained turbulence needs O(100) advective time units (smoke tests
  reach at most `t = 1`), `Re` above sustainment (plane Couette:
  ~350-500) and a finite-amplitude perturbation in a domain at least the
  minimal flow unit, which is itself only transiently turbulent.

## Test-run discipline

- Before running anything, ask what the change can reach and
  run the cheapest check that covers exactly that, often a one-line
  `uv run python -c "import ..."`. A docstring, comment or type-hint
  change needs no run.
- Run one heavy suite at a time: the in-process tests leave JAX's CPU
  thread pool unpinned on purpose (`bootstrap.configure_jax_platform`;
  do not "fix" it), so concurrent suites oversubscribe and abort
  spuriously. A failure that does not reproduce on a clean serial rerun
  was contention: say so rather than re-running silently.
- A heavy suite can take the machine down. First record in persistent
  memory what resuming the work needs: goal, changes, what is
  verified, commands and log paths (logs outside the scratchpad), what
  is left. Keep it current; delete it when done.
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

## Conduct

- A difference at machine epsilon is a footnote, not a defect. Only an
  exact-equality guard cares (`twin.e0 = 0`, the laminar smoke's
  `err = 0`), and then the finding is the guard.
- Before replacing a specialised numerical routine (quadrature,
  interpolation, a solve) with a general one, compare their accuracy on
  the special case, and keep the specialised one where it wins.
- Fix a problem at its source where possible; where not, make the
  exception explicit where it lives, with the reason.
- Commit a reusable, valuable tool a session writes (a check, a
  harness, a converter) rather than leave it in the scratchpad or
  `~/.cache`: in `lab/scripts/<topic>/` (`scripts/` if a solver user
  needs it), its docstring the manual, with a line in the script index
  and the README that list its directory, so a later session reruns
  it, not rebuilds it.

## Documentation rules

- The agent notes (every `CLAUDE.md` and `.claude/rules/*.md`) are an
  index, not documentation: how and why a module works, measured
  numbers and history belong in its docstring, and code never points
  into a note. Writing a note: `.claude/rules/agent-notes.md`, which
  loads with any note you open.
- Keep docstrings, comments, type hints and these notes current, at 79
  columns.
- The human-facing docs change in the same pass as what they describe:
  the READMEs, `CONTRIBUTING.md`, `docs/` and `lab/docs/` (the set, and
  their conventions: the `paths:` of `.claude/rules/human-docs.md`).
- Nothing committed carries a placeholder, an invisible marker or a
  claim whose backing artefact has not landed: no `TODO(author)`, no
  `DRAFT:`, no empty section, no badge for a DOI that does not exist.
  Omit what cannot be finished truthfully.

## Package map (`src/dnsjax/`)

JAX-free leaves, importable before JAX is configured: `parameters.py`,
`param_surface.py`, `flow_spec.py`, `harmonics.py`, `fd.py`,
`adaptive.py`, `snapshot_meta.py`, `seeding.py`, `run_status.py`,
`extensions/`, `flows/registry.py`,
the flow `specs/`, `tools/`, and `analysis/` (bar the members its
notes name).
Reuse them (extract a helper into one rather than copying it), and put
a module used only outside the solver in a subpackage, not at this
level.

```
__main__.py       the `dnsjax` entry point: run loop, streams, resume
bootstrap.py      setup for every entry point: parameters, JAX, ranks, seeds
run_status.py     RUNNING / FINISHED / STOPPED / TERMINATED, the stop request
parameters.py     pydantic models; params / derived_params / padded_res
flow_spec.py      FieldSpec / DeferredSpec / FlowSpec
param_surface.py  per-flow CLI/TOML surfaces, public names, params dump
extensions/       extension sections: registry, run-loop hooks
sharding.py       the (np0, np1) mesh singleton, layouts, pytree registration
fft.py            real FFTs, 3/2 dealiasing, shard_map reshards, padding
harmonics.py      integer wavenumber sequences of the spectral axes
operators.py      wavenumbers (from harmonics.py), FFT wrappers
rhs.py            rotational nonlinear term; measure_fn and metric hooks
measurements.py   CFL
timestep.py       make_stepper: the iterative-cn and cnab2 steppers
adaptive.py       CFL time-step controller
fd.py             FD weights, quadrature, interpolation, grids
solvers.py        dense (reference) and banded Pallas per-mode solves
snapshot.py       snapshot save/load
snapshot_meta.py  snapshot tar metadata, stdlib only
lowres.py         [lowres]: reduced-resolution snapshots (+ pressure)
seeding.py        the seed contract
ic/               random_field (the default IC), localized_rolls, mean_mode
geometries/       wall_bounded/, triply_periodic/
flows/            registry.py; per-family flow modules and specs/
analysis/         snapshot API, transient_growth, snapshot_import
tools/            console utilities: dnsjax-info (snapshot metadata)
```

## Invariants everywhere

- The two geometry families are independent: `geometries/` and
  `flows/` each split into `wall_bounded/` and `triply_periodic/`, and
  nothing crosses. Do not read across families unless asked.
- `params`, `derived_params`, `padded_res`, `sharding` and each
  geometry's `fourier` are module singletons built at import.
  Configure JAX and finalize the parameters (`update_parameters()`,
  then `validate_parameters()` and `padded_res.set_padded_resolution`)
  before importing `sharding` or any geometry module. Every entry point
  (solver, scripts, tests) does that through `dnsjax.bootstrap`, whose
  module docstring is the contract; never inline `jax.config.update` /
  `jax.distributed.initialize`. Modules importable earlier keep
  `import jax` out of module scope.
- Base-flow systems evolve and store the perturbation `u'` about the
  laminar `U(y)`; the force-driven systems (`FlowSpec.total_field`,
  collected in `flows.registry.total_field_systems`) evolve and store
  the total field.
- Cylindrical and annular states are carried in a solver basis and
  observed in physical components: convert a freshly built physical
  state before stepping it (`geometries/wall_bounded/CLAUDE.md`).
- A resolution or parameter sweep needs one process per value: the
  value is captured in the singletons and the jitted steppers. `dt`
  alone changes in place, through the flow module's `set_dt` with
  `params.step.dt` kept in step, as `__main__`'s adaptive loop does.
- Pre- and post-processing belong in an offline tool, placed as Conduct
  says, that meets the solver through snapshots, not in a new solver
  parameter or runtime path.

## Core and lab

- One uv workspace, two packages: `dnsjax` (the root, released) and
  `dnsjax-lab` (`lab/`, in development; its index: `lab/CLAUDE.md`).
  `uv sync` installs both.
- One way only: the core never imports or names the lab
  (`tests/test_layering.py`). The lab plugs in through the
  `dnsjax.flows` / `dnsjax.parameters` / `dnsjax.extensions` entry
  points and may import core internals, so a core change runs both
  suites.
- A lab feature that needs the core adds or widens a seam the core's
  own flows use too, in the same commit.
- A release (PyPI's sdist and wheel; the tag's GitHub archive, which
  a Zenodo deposit takes) leaves out what `.gitattributes` lists, the
  lab and every agent file (`CLAUDE.md`, `.claude/`) among them: a new
  kind of agent file gets a pattern there and in `pyproject.toml`'s
  `[tool.uv.build-backend]`. `lab/scripts/release/prepare_release.py`
  cuts it; only the author pushes the tag, publishes or creates the
  GitHub release.

## Other notes

- `.claude/rules/`, each loaded when you read, edit or write a file it
  governs (read one directly before working in its area if you have not
  opened such a file yet): `jax.md` and `docstrings.md` (any Python),
  `stepping.md`, `parameters.md`, `initial-conditions.md`,
  `snapshots.md`, `diagnostics.md`, `pallas.md`, `human-docs.md` (the
  human-facing docs), `agent-notes.md` (these notes).
- Directory notes (`CLAUDE.md`): `tests/`, `scripts/`, in
  `src/dnsjax/`: `geometries/wall_bounded/`, `geometries/triply_periodic/`,
  `flows/`, `analysis/`; the lab's `lab/` (its index, scripts included)
  and `lab/src/dnsjax_lab/twin/`.

## Commit messages

Audience, in priority order: (1) an LLM reading the history to
understand this codebase; (2) a technical reader assessing the work.

- Subject: imperative mood, at most 72 characters and ideally 50, no
  trailing full stop.
- Body: wrapped at 72 characters, blank line after the subject, present
  for any non-trivial change. State why the change was made and what
  changed at the level of modules, equations, numerical schemes or data
  structures. Do not restate the diff.
- Name the concrete thing: the solver, boundary condition, geometry,
  spectral transform, time-stepper, JAX transformation, sharding
  strategy. "Update solver" is a failure.
- Never use a vague verb as the whole subject: update, improve, fix
  stuff, cleanup, various changes, misc.
- No self-deprecation, no profanity, no informal aside, no emoji, and
  no trailer block (`Co-authored-by:` included): AI assistance is
  disclosed once, in the README's "Use of AI" section
  (`.claude/settings.json`'s `attribution` turns Claude Code's off).
- Plain words: the banned words of `lab/scripts/docs/banned_words.py`
  (`TONE`, `WORDS`) apply to commit messages as to docstrings. A banned
  word is permitted only where it is the correct and necessary
  technical term (the cadence `anchor` argument of `lowres.lowres_due`
  and the twin driver).
- `lab/scripts/docs/check_commit_message.py` rejects what of this and
  the next section a script can judge: prek's commit-msg hook runs it
  on every commit (`uv run prek install` installs both hooks).
  `--commits origin/main..HEAD` checks what a push would publish,
  including commits the hook never saw (`--no-verify`,
  `git commit-tree`).

## Commit hashes

- Code, configuration, tests and documentation never depend on or
  record a commit hash: a hash does not survive a history rewrite.
- A commit message never cites a commit hash; it cites the commit's
  subject line, in quotes.
- Snapshot and run metadata may record a commit hash, and should record
  the commit subject line alongside it so the provenance survives a
  rewrite.

## History

- Keep `main` linear: never create a merge commit (GitHub refuses a
  push to `main` that carries one). Land a branch by rebasing it onto
  `main` and fast-forwarding `main` to it.
- Rebase an unpushed branch onto `main`, and fast-forward `main` to it,
  without asking: it rewrites only commits no remote holds. After `git
  fetch` (and `git merge --ff-only origin/main` on a `main` that is
  behind), a branch is unpushed when `git rev-list main..<branch> --not
  --remotes` lists every commit `git rev-list main..<branch>` does.
  Then `git rebase --committer-date-is-author-date main <branch>`, `git
  switch main` and `git merge --ff-only <branch>`; rebase `main`'s own
  unpushed commits onto `origin/main` the same way. On a conflict, `git
  rebase --abort` and ask: a resolution is a change the user has not
  seen.
- Never run any other history-rewriting git command without explicit
  per-instance approval: a rebase of a pushed commit, filter-branch,
  filter-repo, reset --hard, a ref update onto an existing branch or
  tag (`branch -f`, `update-ref`, `tag -f`) and reflog expiry or
  pruning.
- `git push --force` and `--force-with-lease` are the user's action,
  never the agent's.
- Every commit carries one author and committer identity, that of
  HEAD's author (`git log -1 --format='%an <%ae>'`), with the committer
  date equal to the author date. An amend resets the committer date;
  amend with `GIT_COMMITTER_DATE="$(git log -1 --format=%aD)" git commit
  --amend`.
