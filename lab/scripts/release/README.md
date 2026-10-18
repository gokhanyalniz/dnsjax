# Release tooling

How a release is cut: the docstring of `prepare_release.py`. Each
script's module docstring is its manual.

| Script | What it does |
|---|---|
| `prepare_release.py` | cuts a release as a commit and tag of its own, recording the DOI reserved on Zenodo (`--doi`), and pushes nothing; the edits it makes, from `pyproject.toml` and `uv.lock` to the README and the citation, are listed in its docstring. `files` checks that an sdist and an archive hold the same files; `asset` builds the archive of the repository a GitHub release attaches |
| `pypi_readme.py` | writes `docs/pypi/README.md`, the README as PyPI shows it, with absolute links and images and plain-text math; `--check` fails on a stale copy |
| `ci_job.py` | runs one GitHub workflow job's steps locally, in a throwaway clone of a commit — `core-only` checks a release's artefacts before anything is pushed |
| `check_extras.py` | checks that each extra of `dnsjax` (`cuda13`, `tpu`, ...) is the `jax` extra of the same name and that the installed JAX provides it; CI's lint job runs it |
