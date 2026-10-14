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
`Refuse every stale twin stream, and price two costs honestly` rather
than `fix streams`. Wrap the body at 72 columns and use it for the
reasoning: what was wrong, and why this is the fix.

## Releases

A release holds the solver alone. The PyPI sdist and the GitHub
archive of the release tag that Zenodo keeps hold the same files (the
package, its tests, docs, examples and scripts, and the top-level
documents), and the archive adds `uv.lock`; neither holds the lab, the
files written for coding agents or the development configuration (the
exclusions are `.gitattributes`'s for the archive and
`[tool.uv.build-backend]`'s in `pyproject.toml` for the sdist and the
wheel).
The wheel is the package alone. The sdist and the archive install with
uv: the archive with its `uv.lock` (`uv sync --locked`), the sdist
without one. `main` cannot be released as it stands, since its
`pyproject.toml` installs the lab with the solver, so each release is
cut as a commit of its own, reachable only from its tag, by
`lab/scripts/release/prepare_release.py`. With the version bumped in
both `pyproject.toml` and `CITATION.cff` on `main`:

```bash
uv run python lab/scripts/release/prepare_release.py cut 0.2.0 --dry-run
uv run python lab/scripts/release/prepare_release.py cut 0.2.0
git push origin v0.2.0
```

`cut` strips the lab from `pyproject.toml`, relocks, dates the
citation, trims the README to the sections a release carries, pins the
PyPI README's links to the tag, points the archive's links into the lab
at the tag on GitHub, commits and tags.
Pushing the tag runs `.github/workflows/release.yml`, which checks the
artefacts and the archive, then publishes to PyPI once the `pypi`
environment's reviewer approves and drafts the GitHub release.
Publishing that draft is what has Zenodo archive it. Running the
workflow by hand rehearses the build on TestPyPI. CI's `core-only` job
checks the release build on every push of every branch and on every
pull request, from that push's own commit; to run the same checks
before pushing, from a commit of your own:

```bash
uv run --no-project --with pyyaml python lab/scripts/release/ci_job.py core-only
```

PyPI shows `docs/pypi/README.md`, the README with absolute links and
plain-text math. Regenerate it after every change to `README.md`:

```bash
uv run python lab/scripts/release/pypi_readme.py
```

CI's lint job fails while it is stale.
