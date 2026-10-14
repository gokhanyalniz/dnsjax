# Contributing

## Setup

```bash
uv sync
```

That provisions the pinned Python, installs the dependencies and the
`dnsjax` console script, and puts `pytest`, `ruff` and `prek` in the dev
environment. Figure scripts additionally need the `plots` group:
`uv run --group plots python scripts/snapshot_figure.py`.

The repository is a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
of two packages sharing one lockfile and one `.venv`: `dnsjax`, the
released solver at the root, and `dnsjax-lab` in `lab/`, which is
developed alongside it but never released
([`lab/README.md`](lab/README.md)). `uv sync` installs both, the lab in
editable mode with its own console script, `dnsjax-twin`. The solver
finds the lab only through its entry points and never imports it:
`tests/test_layering.py` fails on any core file that names it. A change
to the solver that a lab feature needs adds or widens a seam the
solver's own flows use too, and is tested by both suites.

## Python versions

Three separate things, and `uv lock --upgrade` changes none of them — it
only moves package versions within the constraints they set:

| | what it is | changed by |
|---|---|---|
| `.python-version` | the interpreter `uv sync` builds `.venv` with | `uv python pin <ver>`; keep it on the newest release |
| `requires-python` in `pyproject.toml` | the **oldest** version supported | hand-edited, only for the reasons below |
| `requires-python` in `uv.lock` | a mirror of the above | rewritten by `uv lock` |

Develop on the newest Python. The floor is a separate decision, and it
moves only when one of these holds:

1. **A dependency forces it.** uv says so outright — set the floor below
   what the dependencies allow and `uv lock` refuses, naming the version
   to use: *"The `requires-python` value (>=3.11) includes Python
   versions that are not supported by your dependencies (e.g.
   numpy>=2.5.1 only supports >=3.12)."* That is the whole check. The
   current `>=3.12` is exactly where JAX and NumPy put it.
2. **The floor has started costing something.** Because uv resolves
   universally, a package that drops the floor version is not an error:
   uv either holds it back, or **forks** the lock — a newer version for
   newer interpreters, an older one for the floor — at which point the
   two CI matrix jobs are no longer testing the same dependencies. A
   fork shows up as a duplicated package:

   ```bash
   grep -E '^name = ' uv.lock | sort | uniq -d   # empty means no fork
   ```

   Raising the floor is then a judgement call, not a requirement.
3. **The code wants a feature only a newer Python has**, deliberately,
   with the reason recorded.

Never raise it merely because the development machine has moved on.

Moving the floor touches several places, none of which derive from
each other: `requires-python` and the classifiers in `pyproject.toml`,
`requires-python` in `lab/pyproject.toml`, the `smoke` matrix and the
`core-only` job's floor venv in `.github/workflows/ci.yml`, the floor
venv in `.github/workflows/release.yml`, the Python badge in
`README.md`, and the Prerequisites line in the repository's root
`CLAUDE.md`. **Lowering** it additionally requires a real run under the
new floor, in this order: `py_compile` the tree, import it, then run
the offline suite. Source-level compatibility is necessary but not
sufficient, and only a real run finds things like reliance on deferred
annotation evaluation.

## Lint and format

```bash
uv run ruff check --fix
uv run ruff format src tests scripts lab/src lab/tests lab/scripts
```

Line length is **79** for code and prose alike (the Markdown docs'
tables and HTML excepted). Name the directories when formatting — a bare
`uv run ruff format` also reformats the Python blocks in the Markdown
docs and the example notebook, which are laid out to be read rather than
to satisfy the formatter.

`prek.toml` configures the commit hook, which runs both:

```bash
uv run prek install
```

No tracked file carries an email address, the package metadata
included: CI's lint job fails on one.

## Tests

Every test is a standalone script; `pytest` is a bridge over them,
collecting the solver's suite (`tests/`) and the lab's (`lab/tests/`)
together. Why, how to run one, and what the `mpi` / `slow` markers
mean: [`tests/README.md`](tests/README.md).

```bash
uv run pytest -m "not slow and not mpi"   # the offline loop
uv run pytest -m "not slow"               # + the quick mpirun rows
uv run pytest                             # everything available
```

Run **one** heavy suite at a time: each invocation is already serial
internally and leaves JAX's CPU thread pool unpinned, so concurrent runs
oversubscribe the machine and produce spurious failures.

Pick the check that fits what a change can actually reach. A docs or
type-hint pass cannot change behaviour, and a test run that cannot tell
"the change is fine" from "the change was never involved" buys nothing.

## Adding a flow system

Two files: a `FlowSpec` under `src/dnsjax/flows/<family>/specs/`, added
to that package's `SPECS` tuple, and the flow module it names; a flow
in another package registers its spec through an entry point instead.
The surfaces, the dispatch and the analysis geometry sets all derive
from the registry. The full recipe, including what a spec may and may
not import: [`docs/extending.md`](docs/extending.md).

New flows also need a row in `tests/test_random_smoke.py`, and one in
`tests/test_laminar_smoke.py` when the flow has a closed-form laminar
state.

## Documentation

Docstrings, comments and type hints stay current with the code, at 79
columns; math is LaTeX. A docstring states its contract first and its
reasons in a sentence or two; measurements, rejected alternatives and
history go in a closing "Design notes" section of the module docstring.
The human-facing documents — `README.md`, this file, the `docs/` pages,
`tests/README.md`, `scripts/README.md`, the `examples/` READMEs and the
lab's READMEs and `lab/docs/` pages — are updated in the same change
that makes them wrong. They are written pointer-first: they link the
module docstring that owns a detail rather than restating it, which is
what keeps that affordable. `docs/pypi/README.md`, the README as PyPI
shows it, is regenerated rather than edited (Releases, below).

The mechanical checks of the prose, from links and anchors and
backticked names against the tree to docstring style, Oxford spelling,
parse-validated command lines and a proof that a change is docs-only,
live in `lab/scripts/docs/`; each script's docstring is its manual.

Nothing committed carries a placeholder: no `TODO(author)`, no draft
block, no section waiting on a number. Content that cannot be finished
truthfully is left out until it can be.

## Commits

Imperative, sentence case, no prefix, and specific about what changed —
`Refuse a twin stream recorded under other parameters` rather
than `fix streams`. Wrap the body at 72 columns and use it for the
reasoning: what was wrong, and why this is the fix.

## Releases

A release holds the solver alone, in three artefacts: the sdist and
the wheel on PyPI, and the GitHub release of the tag, whose source
archive is what Zenodo keeps. The sdist and the archive hold the same
files (the package, its tests, docs, examples and scripts, the README
and the citation); neither holds the lab, this guide, the files
written for coding agents, the development configuration or `uv.lock`.
The exclusions are the `export-ignore` lines of `.gitattributes` for
the archive and `source-include` under `[tool.uv.build-backend]` in
`pyproject.toml` for the sdist; the wheel is the package alone. The
sdist and the archive both install with `uv sync`, resolving afresh;
the lock stays in the repository. The GitHub release alone also
carries the repository with the lab, as an asset the workflow attaches
(step 4 below).

`main` is not released as it stands: its `pyproject.toml` installs the
lab with the solver (the `lab` dependency group, a workspace source),
which a tree without `lab/` cannot resolve, and the published README
is trimmed and its links pinned. So each release is a commit of its
own, cut by `lab/scripts/release/prepare_release.py` and reachable
only from its tag; `main` is untouched.

### Once

- GitHub: under `Settings → Environments`, `pypi` and `testpypi`, each
  with a required reviewer. The workflow deploys through them and
  waits for the approval.
- PyPI and TestPyPI: an account with two-factor authentication and a
  pending trusted publisher for the project `dnsjax`: owner
  `gokhanyalniz`, repository `dnsjax`, workflow `release.yml`,
  environment `pypi` on PyPI and `testpypi` on TestPyPI. No token is
  stored anywhere; a pending publisher reserves nothing, the first
  upload does.
- Zenodo: log in with GitHub and switch `gokhanyalniz/dnsjax` on
  before the first release. From then on Zenodo archives every release
  published on GitHub, from its source archive alone: a release's
  assets never reach it.

### Rehearsal

Under Actions, run the Release workflow by hand on `main`. It strips
the lab from `main`'s `pyproject.toml` as a release commit does, runs
every check of step 3, waits for the `testpypi` reviewer and uploads
to TestPyPI (a repeated version is skipped, not an error); no tag and
no GitHub release. Then, in a fresh environment:

```bash
uv venv /tmp/rehearsal
uv pip install --python /tmp/rehearsal/bin/python \
  --index-strategy unsafe-best-match \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ dnsjax
/tmp/rehearsal/bin/dnsjax --help pipe
```

### Each release

1. On `main`, set the version in `pyproject.toml` and `CITATION.cff`,
   commit and push, and let CI pass: its `core-only` job builds the
   release artefacts from that commit and checks them. The same
   checks locally, from a commit of your own:

   ```bash
   uv run --no-project --with pyyaml python lab/scripts/release/ci_job.py core-only
   ```

2. Cut the release commit and the tag, first as a dry run, which
   prints the diff and leaves nothing behind:

   ```bash
   uv run python lab/scripts/release/prepare_release.py cut 0.2.0 --dry-run
   uv run python lab/scripts/release/prepare_release.py cut 0.2.0
   git show --stat v0.2.0
   ```

   `cut` refuses a dirty tree and an existing tag, then, in a
   throwaway worktree of `HEAD`, strips the lab from `pyproject.toml`,
   relocks `uv.lock`, sets `date-released` in `CITATION.cff` (today,
   or `--date`), trims the README to the sections a release carries,
   regenerates the PyPI README with its links pinned to the tag,
   points every link the archive's Markdown makes to a file the
   archive leaves out at the tag on GitHub, commits and tags
   (annotated, `v0.2.0`).

3. Push the tag and nothing else:

   ```bash
   git push origin v0.2.0
   ```

   This runs `.github/workflows/release.yml`, which checks that the
   tag points at a release commit and that the tag, `pyproject.toml`
   and `CITATION.cff` name one version; builds the sdist and the
   wheel; checks both for lab and agent files, and the sdist against
   the tag archive file for file; fetches the tag's zipball from
   GitHub, checks it the same way and installs it with `uv sync`;
   installs the wheel alone on the floor Python; then waits for the
   `pypi` reviewer.

4. Under Actions, open the run and approve the `pypi` deployment. The
   workflow publishes to PyPI, builds the with-lab asset from the
   tag's parent commit (the repository the release was cut from, with
   the lab, this guide and uv's files, without the agent files and
   the CI configuration) and creates a *draft* GitHub release of the
   tag with generated notes, the sdist, the wheel and that asset
   attached.

5. On GitHub, review the draft: the notes, the three attached files,
   and the source archives (no `lab/`, no agent files, no `uv.lock`,
   no `CONTRIBUTING.md`). Publish it. Zenodo archives the source
   archive and mints a version DOI, and on the first release the
   concept DOI.

6. Afterwards, in one commit on `main`: the concept DOI as `doi:` in
   `CITATION.cff` and a DOI badge in the README. Nothing in the
   repository promises a DOI before then.

A release can be deleted on GitHub, assets included; the tag, the PyPI
files and Zenodo's copy stay.

PyPI shows `docs/pypi/README.md`, the README with absolute links and
plain-text math. Regenerate it after every change to `README.md`:

```bash
uv run python lab/scripts/release/pypi_readme.py
```

CI's lint job fails while it is stale.
