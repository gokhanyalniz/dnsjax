# Contributing

## Setup

```bash
uv sync
```

That provisions the pinned Python, installs the dependencies and the
`dnsjax` / `dnsjax-twin` console scripts, and puts `pytest`, `ruff` and
`prek` in the dev environment. Figure scripts additionally need the
`plots` group: `uv run --group plots python scripts/snapshot_figure.py`.

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
columns; math is LaTeX. The human-facing documents — `README.md`,
`NUMERICS.md`, `SCALING.md`, the `docs/` pages, `tests/README.md`, and
the three subpackage READMEs — are updated in the same change that makes
them wrong. They are written pointer-first: they link the module
docstring that owns a detail rather than restating it, which is what
keeps that affordable.

Nothing committed carries a placeholder: no `TODO(author)`, no draft
block, no section waiting on a number. Content that cannot be finished
truthfully is left out until it can be.

## Commits

Imperative, sentence case, no prefix, and specific about what changed —
`Refuse every stale twin stream, and price two costs honestly` rather
than `fix streams`. Wrap the body at 72 columns and use it for the
reasoning: what was wrong, and why this is the fix.
