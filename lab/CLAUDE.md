# dnsjax-lab

The repository's second package, never released (the root note's "Core
and lab"): flows, `dnsjax-twin`, the `[probes]` / `[force]` sections
and analyses built on the core. The human overview: `README.md` here;
the lab flows' maths and checks: `docs/`.

## Layout (`src/dnsjax_lab/`)

```
flows/            SPECS; wall_bounded/: the flow modules and specs/
geometries/       wall_bounded/: the curved, viscoelastic, forced-annulus
                  operators and stepping
parameters.py     FIELDS: the lab flows' fields in the core sections
extensions/       [probes] and [force]: sections and their streams
twin/             dnsjax-twin (its own CLAUDE.md)
analysis/         twin/ (the twin streams' readers, JAX-free),
                  response/, toroidal.py
harmonics.py      the curved pipe's metric harmonics
```

## Meeting the core

- The entry points' objects (`lab/pyproject.toml`):
  `dnsjax_lab.flows.SPECS` (listed after the core's flows in each
  geometry), `dnsjax_lab.parameters.FIELDS` and the
  `dnsjax_lab.extensions` module, which registers its sections when
  imported. All three stay JAX-free, like the core's specs.
- Adding a lab flow: its `FlowSpec` in `flows/wall_bounded/specs/`,
  appended to `SPECS`; a new field in `FIELDS` and a `FieldSpec` in the
  spec; the flow module the spec names; rows in the lab halves of
  `tests/test_random_smoke.py` (and `test_laminar_smoke.py` for a
  closed-form laminar state). What a spec may import: the root's
  `docs/extending.md`.

## Analysis

- `analysis/response/` may use JAX and SciPy, lazily; `analysis/twin/`
  is JAX-free. Ensemble members aggregate on relative time while
  `extensions/probes.py` gates `probes.bin` on the absolute
  `it % it_probes`, so harvest parents at multiples of the probe
  cadence or `response.ensemble.aggregate_tree` refuses the set. The
  pipeline and the route trade-offs: `analysis/response/__init__.py`.
  Its per-mode generators come from the core's `--tg.save_operator`:
  change either end with the other.
- `analysis/toroidal.ToroidalMetric` is the curved pipe's
  `GeometryInfo.metric`: change it with
  `geometries/wall_bounded/cylindrical_curved.py`.

## Tests

`tests/` mirrors the core's suite and shares its runners. The
conventions, the core/lab split of a script and the which-script index
of both suites: the core's `tests/CLAUDE.md` (read it before writing a
lab test).

## Scripts (`scripts/`)

Each script's module docstring is its manual. `twin_spectral_maps.py`
and `performance/scaling_figure.py` need matplotlib: `uv run --group
plots python <path>`. The seeds in the benchmarks and diagnostics are
fixed on purpose.

- `ensemble_setup.py`: JAX-free `harvest` / `build` / `build-twin` of
  ensemble member trees from a snapshot archive.
- `twin_postprocess.py`: rebuild a twin member's streams from its
  snapshot pairs (`[recon]`).
- `twin_spectral_maps.py`: `(λ, y)` maps, `(y, t)` / `(λ, t)`
  histories, moment budgets, decorrelation fronts and growth-law
  figures of the twin `(y, k)` streams over an ensemble.
- `random_ic_calibrate.py`: score and sweep the random IC's `(y, k)`
  shape against a recorded ensemble (`[calib]`).
- `performance/solver_benchmark.py`: pallas-vs-dense validation and
  benchmark on the flows its `SIZES` / `EXTRA_ENTRIES` tables list
  (`--cpu-smoke`).
- `performance/node_benchmark.py`: JAX-free layout and strong-scaling
  sweep of one problem, `mpirun` or `srun` over nodes, each row's
  `stats.dat` checked against a reference (`--dry-run`, `--cpu-smoke`);
  imports its sibling `solver_benchmark.py`.
- `performance/pallas_solve_profile.py`: GPU profile of the banded solve
  and its share of a step (`--cpu-smoke`).
- `performance/memory_budget.py`: per-rank memory of the as-run step on
  any layout, offline (forced CPU devices).
- `performance/memory_watch.py`: JAX-free per-node memory sampler.
- `performance/scaling_figure.py`: the strong-scaling figure (light and
  dark SVG) and tables of `docs/scaling.md`, from
  `docs/figures/*-scaling.csv`.
- `diagnostics/grad_probe.py`: forward/reverse differentiability matrix
  with a finite-difference check.
- `diagnostics/pallas_tiling_diagnostic.py`: GPU harness for the Triton
  partial-tile miscompile.
- `diagnostics/gds_probe.py`: cluster diagnostic for the snapshot GDS
  path (`--cpu-smoke`).
- `refactor/verify.py`: the gate for a refactor that must change
  nothing: capture a commit in a throwaway worktree and compare it with
  a reference capture (`--only` narrows the cases).
- `refactor/capture.py`: a tree's behavioural fingerprint through its
  public command lines; `refactor/compare.py` diffs two, with reviewed
  differences allowed by pattern.
- `refactor/refingerprint.py`: recompute a capture's HLO fingerprints.
- `refactor/analysis_fp.py`: fingerprint the JAX-free analysis API on a
  capture's snapshots, and compare two.
- `refactor/check_imports.py`: every `dnsjax` / `dnsjax_lab` import in
  the repository resolves, checked without importing.
- `refactor/prose_scan.py`: the lab's vocabulary in the core's prose,
  for review.
- `release/prepare_release.py`: `cut VERSION`, a release's own commit
  and tag (the lab stripped, relocked, the PyPI README pinned, the
  archive's lab links absolute); `strip`, the same `pyproject.toml`
  edit for CI; `files`, an sdist against an archive. Never push its
  tag.
- `release/pypi_readme.py`: write `docs/pypi/README.md` from the root
  `README.md` (`--check`, in CI's lint job).
- `release/ci_job.py`: run one CI workflow job's steps locally in a
  throwaway clone (`core-only`: a release's checks; `--with pyyaml`).
