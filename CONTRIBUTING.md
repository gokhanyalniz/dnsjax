# Contributing

Work on a branch of a clone, and send the branch or open a pull request
against `main`. What follows is what a change needs to pass.

## Setup

```bash
uv sync
uv run prek install
```

`uv sync` provisions the pinned Python, installs the dependencies and
the `dnsjax` console script, and puts `pytest`, `ruff` and `prek` in the
environment; `prek install` installs the commit hooks ([Lint and
format](#lint-and-format), [Commits](#commits)). Figure scripts also
need the `plots` group: `uv run --group plots python
scripts/snapshot_figure.py`. An accelerator is an extra of JAX's name,
`uv sync --extra cuda13` (the README's Quick start lists them).

The repository is a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
of two packages in one lockfile and one `.venv`: `dnsjax`, the solver
at the root, and `dnsjax-lab` in `lab/`, flows and features in
development alongside it ([`lab/README.md`](lab/README.md)). `uv sync`
installs both, the lab with its own console script, `dnsjax-twin`. The
solver finds the lab only through its entry points and never imports
it: `tests/test_layering.py` fails on a core file that names it. A
solver change a lab feature needs adds or widens a seam the solver's
own flows use too, and is tested by both suites.

## Python versions

`.python-version` pins the interpreter `uv sync` builds `.venv` with,
the newest release; `requires-python` in `pyproject.toml` is the oldest
version supported, where JAX and NumPy put theirs (`uv lock` refuses a
lower one, naming the version). A change moves neither unless it wants
a feature only a newer Python has, with the reason recorded. The floor
is named in `pyproject.toml` (`requires-python`, the classifiers),
`lab/pyproject.toml`, the `smoke` matrix and the `core-only` job of
`.github/workflows/ci.yml`, `.github/workflows/release.yml` and the
README's Python badge and Quick start, which names the pinned version
too. Lowering it needs a real run under the new floor, `py_compile` of
the tree, then an import, then the offline suite: only a run finds a
reliance on a newer interpreter, such as deferred annotation
evaluation.

## Lint and format

```bash
uv run ruff check --fix
uv run ruff format src tests scripts lab/src lab/tests lab/scripts
```

Line length is **79** for code and prose alike (the Markdown docs'
tables and HTML excepted). Name the directories when formatting: a bare
`uv run ruff format` also reformats the Python blocks in the Markdown
docs and the example notebook, which are laid out to be read.

`prek.toml` configures the two commit hooks: ruff check and format at
commit, and the rules of [Commits](#commits) on the message, through
[`check_commit_message.py`](lab/scripts/docs/check_commit_message.py).

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

Pick the check that fits what a change can reach. A docs or type-hint
pass cannot change behaviour, and a test run that cannot tell "the
change is fine" from "the change was never involved" buys nothing.

## Adding a flow system

Two files: a `FlowSpec` under `src/dnsjax/flows/<family>/specs/`, added
to that package's `SPECS` tuple, and the flow module it names; a flow in
another package registers its spec through an entry point instead. The
surfaces, the dispatch and the analysis geometry sets all derive from
the registry. The full recipe, including what a spec may and may not
import: [`docs/extending.md`](docs/extending.md).

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
module docstring that owns a detail rather than restating it.

`docs/pypi/README.md`, the README as PyPI shows it, is generated, never
edited; CI's lint job fails while it is stale. After any change to
`README.md`:

```bash
uv run python lab/scripts/release/pypi_readme.py
```

The mechanical checks of the prose, from links, their in-page targets
and backticked names against the tree to docstring style, Oxford
spelling and parse-validated command lines, live in `lab/scripts/docs/`;
each script's docstring is its manual.

Nothing committed carries a placeholder: no `TODO(author)`, no draft
block, no section waiting on a number. Content that cannot be finished
truthfully is left out until it can be.

## Commits

Imperative, sentence case, no prefix, at most 72 characters (ideally
50) with no trailing full stop, and specific about what changed —
`Refuse a twin stream recorded under other parameters` rather than `fix
streams`. Wrap the body at 72 columns and use it for the reasoning: why
the change was made, and what changed at the level of modules,
equations, numerical schemes or data structures. No trailer block, and
no commit hash: cite another commit by its subject, in quotes, since a
hash does not survive a history rewrite. The banned words of the
docstring checks ([`banned_words.py`](lab/scripts/docs/banned_words.py))
apply to messages too. The commit-message hook checks these rules as
far as a script can; `uv run python
lab/scripts/docs/check_commit_message.py --commits origin/main..HEAD`
checks what a push would publish.

`main` is linear: a branch is brought up to date by rebasing it onto
`main`, never by merging `main` into it, and lands by fast-forward.
