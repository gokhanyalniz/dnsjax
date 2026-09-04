# Contributing

## Setup

```bash
uv sync
```

That provisions the pinned Python, installs the dependencies and the
`dnsjax` / `dnsjax-twin` console scripts, and puts `pytest`, `ruff` and
`prek` in the dev environment. Figure scripts additionally need the
`plots` group: `uv run --group plots python scripts/snapshot_figure.py`.

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

Moving the floor touches five places, none of which derive from each
other: `requires-python` and the classifiers in `pyproject.toml`, the
`smoke` matrix in `.github/workflows/ci.yml`, the Python badge in
`README.md`, and the Prerequisites line in `CLAUDE.md`. **Lowering** it
additionally requires running the offline suite under the new floor —
source-level compatibility is necessary but not sufficient, and only a
real run finds things like reliance on deferred annotation evaluation.

## Lint and format

```bash
uv run ruff check --fix
uv run ruff format src tests scripts
```

Line length is **79 for every line**, code and prose alike. Name the
directories when formatting — a bare `uv run ruff format` also reformats
the code blocks in `README.md`, which are written to be read rather than
to satisfy the formatter.

`prek.toml` configures the commit hook, which runs both:

```bash
uv run prek install
```

## Tests

Every test is a standalone script; `pytest` is a bridge over them. Why,
how to run one, and what the `mpi` / `slow` markers mean:
[`tests/README.md`](tests/README.md).

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
to that package's `SPECS` tuple, and the flow module it names. The
surfaces, the dispatch and the analysis geometry sets all derive from
the registry. The full recipe, including what a spec may and may not
import: [`docs/extending.md`](docs/extending.md).

New flows also need a row in `tests/test_laminar_smoke.py` and one in
`tests/test_random_smoke.py`.

## Documentation

Docstrings, comments and type hints stay current with the code, at 79
columns; math is LaTeX. The human-facing documents — `README.md`, this
file, the `docs/` pages, `tests/README.md`, and the three subpackage
READMEs — are updated in the same change that makes them wrong. They
are written pointer-first: they link the module docstring that owns a
detail rather than restating it, which is what keeps that affordable.

Nothing committed carries a placeholder: no `TODO(author)`, no draft
block, no section waiting on a number. Content that cannot be finished
truthfully is left out until it can be.

## Commits

Imperative, sentence case, no prefix, and specific about what changed —
`Refuse every stale twin stream, and price two costs honestly` rather
than `fix streams`. Wrap the body at 72 columns and use it for the
reasoning: what was wrong, and why this is the fix.
