#!/usr/bin/env python3
r"""Capture a commit and compare it with a reference capture.

Usage::

    verify.py COMMIT NAME [--ref NAME] [--work DIR] [--jobs N]
              [--only REGEX] [-- COMPARE_OPTION ...]

Checks *COMMIT* out into a throwaway worktree (``WORK/wt-NAME``),
installs it there (``uv sync``, so the worktree has the ``.venv`` the
capture drives), captures it into ``WORK/cap-NAME`` (``capture.py``),
compares that with the reference capture ``WORK/cap-REF``
(``compare.py``, which receives every option after ``--``) and removes
the worktree.  Everything is logged to ``WORK/logs/verify-NAME.log``.
``--only`` passes a run/text filter to the capture and the comparison,
for a quick look at a subset.  Without a reference capture it only
captures, which is how a refactor's baseline is made::

    verify.py main baseline                  # the reference, once
    verify.py HEAD step1                     # each step after it
    verify.py HEAD step2 --ref step1 -- --allow 'text help\.'

*WORK* defaults to ``~/.cache/dnsjax-refactor``.  A capture takes about
12 minutes at the default ``--jobs 4`` and about 0.9 GB on disk, most of
it the HLO dumps, and is kept for later comparisons; run one at a time.
Exit status: the comparison's (1 on an unallowed difference), 2 when
the checkout, the install or the capture fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The repository holding this script (``lab/scripts/refactor/``).
REPO = HERE.parents[2]


def _run(cmd: list, log, cwd: Path | None = None) -> int:
    """Run *cmd*, appending its output to *log* and echoing it."""
    print("+", " ".join(str(c) for c in cmd), file=log, flush=True)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        log.write(line)
        log.flush()
        sys.stdout.write(line)
    return proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("commit")
    ap.add_argument("name")
    ap.add_argument("--ref", default="baseline")
    ap.add_argument("--work", type=Path, default="~/.cache/dnsjax-refactor")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default=None)
    # Split at the first ``--`` by hand: an ``argparse.REMAINDER``
    # positional would also swallow the options given after NAME.
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a, options = ap.parse_args(argv[:cut]), argv[cut + 1 :]
    work = a.work.expanduser().resolve()
    tree, capture = work / f"wt-{a.name}", work / f"cap-{a.name}"
    ref = work / f"cap-{a.ref}"
    (work / "logs").mkdir(parents=True, exist_ok=True)
    git = ["git", "-C", REPO]
    with open(work / "logs" / f"verify-{a.name}.log", "w") as log:
        subprocess.run(
            [*git, "worktree", "remove", "--force", tree], capture_output=True
        )
        subprocess.run(["rm", "-rf", capture], check=True)
        try:
            if _run(
                [*git, "worktree", "add", "--detach", tree, a.commit], log
            ):
                return 2
            if _run(["uv", "sync", "-q"], log, cwd=tree):
                return 2
            if _run(
                [
                    sys.executable,
                    HERE / "capture.py",
                    tree,
                    capture,
                    "--jobs",
                    a.jobs,
                    *(["--only", a.only] if a.only else []),
                ],
                log,
            ):
                return 2
        finally:
            subprocess.run(
                [*git, "worktree", "remove", "--force", tree],
                capture_output=True,
            )
        if not ref.is_dir():
            print(f"no reference {ref}: captured only", file=log)
            print(f"no reference {ref}: captured only")
            return 0
        # A subset capture compares as that subset, or every reference
        # entry it left out would count as missing.
        if a.only and "--only" not in options:
            options = [*options, "--only", a.only]
        return _run(
            [sys.executable, HERE / "compare.py", ref, capture, *options], log
        )


if __name__ == "__main__":
    sys.exit(main())
