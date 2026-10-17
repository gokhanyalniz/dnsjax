#!/usr/bin/env python3
r"""Check that every extra of ``dnsjax`` is an extra the installed JAX has.

Usage::

    uv run python lab/scripts/release/check_extras.py

``pyproject.toml``'s ``[project.optional-dependencies]`` forwards each
accelerator extra (``cuda13``, ``tpu``, ...) to the ``jax`` extra of
the same name, which is how the README presents them.  JAX renames
these as its backends move (``cuda`` became ``cuda12`` and ``cuda13``;
``rocm`` became ``rocm7-local``), and a resolver meeting an extra a
package does not provide warns and installs the package without it
(observed with uv 0.12.7): a ``uv lock --upgrade`` onto such a JAX
would silently turn ``dnsjax[rocm7-local]`` into the CPU build.  This
reads the extras from ``pyproject.toml`` and the ``Provides-Extra``
entries of the ``jax`` the environment holds (the locked one after
``uv sync``), and exits 1 naming each extra of ``dnsjax`` whose
requirement is not ``jax[<the same name>]`` or whose name JAX does not
provide.  CI's lint job runs it.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The repository holding this script (``lab/scripts/release/``).
REPO = HERE.parents[2]

#: A requirement forwarding to extras of ``jax``: the extras it names.
_JAX_EXTRAS = re.compile(r"jax\[([^\]]+)\]\s*(?:[<>=!~].*)?")


def main() -> int:
    with open(REPO / "pyproject.toml", "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    jax = importlib.metadata.metadata("jax")
    provided = set(jax.get_all("Provides-Extra") or [])
    version = jax["Version"]
    problems = []
    for name, requirements in extras.items():
        for requirement in requirements:
            m = _JAX_EXTRAS.fullmatch(requirement)
            names = [e.strip() for e in m.group(1).split(",")] if m else []
            if names != [name]:
                problems.append(
                    f"{name}: {requirement!r} is not jax[{name}], the "
                    "extra of the same name"
                )
            for extra in names:
                if extra not in provided:
                    problems.append(f"{name}: jax {version} has no {extra!r}")
    if problems:
        print("\n".join(problems))
        return 1
    print(f"{len(extras)} extras, each one jax {version} provides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
