# dnsjax-lab

The repository's second package (the root note's "Core and lab"):
flows, `dnsjax-twin`, the `[probes]` / `[force]` sections and analyses
in development, built on the core. The human overview: `README.md` here;
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

Each script's module docstring is its manual; the human index is
`README.md`'s "Scripts" for `performance/` and `diagnostics/`, and the
`README.md` of each of `refactor/`, `release/` and `docs/`.
`twin_spectral_maps.py` and `performance/scaling_figure.py` need
matplotlib: `uv run --group plots python <path>`. The seeds in the
benchmarks and diagnostics are fixed on purpose.

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
  sweep of one problem, `mpirun` or `srun` over nodes (`--target gpu`:
  one process spanning every visible GPU), each row's `stats.dat`
  checked against a reference when `--outs.it_stats` writes one
  (`--dry-run`, `--cpu-smoke`); imports its sibling
  `solver_benchmark.py`.
- `performance/pallas_solve_profile.py`: GPU profile of the banded solve
  and its share of a step (`--cpu-smoke`).
- `performance/memory_budget.py`: per-rank memory of the as-run step on
  any layout, offline (forced CPU devices).
- `performance/memory_watch.py`: JAX-free per-node memory sampler.
- `performance/step_sharing_census.py`: which `t^n` GEMMs one
  iterative-CN step shares between its predictor and correction
  passes (an HLO dot census, CPU).
- `performance/scaling_figure.py`: the strong-scaling figure (light and
  dark SVG) and tables of `docs/scaling.md`, from
  `docs/figures/*-scaling.csv`.
- `diagnostics/grad_probe.py`: forward/reverse differentiability matrix
  with a finite-difference check.
- `diagnostics/pallas_tiling_diagnostic.py`: GPU harness for, and
  regression check of, the observed miscompile of the masked
  partial-tile path.
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
- `release/prepare_release.py`: `cut VERSION [--doi DOI]`, a release's
  own commit and tag; `strip`, the same `pyproject.toml` edit for CI;
  `files`, an sdist against an archive; `asset REF OUT`, the with-lab
  archive the release workflow attaches. Never push its tag.
- `release/pypi_readme.py`: write `docs/pypi/README.md` from the root
  `README.md` (`--check`, in CI's lint job).
- `release/ci_job.py`: run one CI workflow job's steps locally in a
  throwaway clone (`core-only`: a release's checks; `--with pyyaml`).
- `release/check_extras.py`: each extra of `dnsjax` is the `jax` extra
  of the same name and the installed JAX provides it (CI's lint job).
- `docs/check_links.py`: relative links and in-page `#` targets of the given
  Markdown files (GitHub slugs).
- `docs/check_refs.py`: every backticked module, symbol, member,
  parameter field, flag and path in the tree's docstrings, comments and
  docs resolves; its index serves `docs/check_snake.py`, the backticked
  snake_case tokens that name nothing.
- `docs/check_docstrings.py`: raw-string, math-form, doubled-word and
  tone-word checks over the tracked Python tree; `docs/lint_docs.py`:
  banned words (`WORDS`), multi-line summaries and split roles in the
  given files.
- `docs/banned_words.py`: `TONE` and `WORDS`, the banned-word patterns
  the prose checks share.
- `docs/check_commit_message.py`: a commit message against the commit
  rules (prek's commit-msg hook); `--commits <revs>`, recorded ones.
- `docs/check_returns.py`: return arity and documented parameters
  against each signature (and a public `Parameters` section omitting
  one).
- `docs/docs_only.py`: proves a diff against a base ref changes
  docstrings, comments and `description=` strings only;
  `docs/prose_only.py`, the same per file and stricter (tokens,
  directives, runtime-read docstrings, TOML/YAML/shell/notebook).
- `docs/prose_atoms.py`: the numbers, code spans, math and qualifiers a
  docs change drops, each traced to where it now lives (`--tree`).
- `docs/doc_pointers.py`: removed headings, titles, banners or
  numbering with their remaining references; `--inbound <files>`.
- `docs/dup_prose.py`: repeated prose passages across the tree.
- `docs/md_lint.py`: added Markdown lines against the prose rules.
- `docs/parse_cmds.py`: parse-validates one documented command line
  (solver, twin or a TOML), JAX-free.
- `docs/check_render.py`: renders Markdown through GitHub's API (a POST
  per file) and compares its math with the source.
- `docs/oxford.py` (`--dry`; its ranges: `docs/doc_ranges.py`),
  `docs/reflow.py`, `docs/underline.py`: mechanical rewrites of
  documentation: Oxford spelling, one reflowed paragraph, section
  underlines.
- `docs/doc_inventory.py`: per-file docstring and comment dumps (under
  `OUT_DIR`, default `./dump`) and the missing docstrings (printed).
- `docs/crossref.py`: bibliographic metadata of a DOI or a query from
  Crossref.
