#!/usr/bin/env python3
r"""Run one job of a GitHub workflow locally, in a throwaway clone.

Usage::

    uv run --no-project --with pyyaml python lab/scripts/release/ci_job.py \
        JOB [--workflow ci.yml] [--commit REF] [--work DIR] [--keep]

Clones the repository at *REF* (default ``HEAD``; commit what the run
should see) into ``WORK/clone`` (*WORK* defaults to a new temporary
directory), then runs *JOB*'s ``run:`` steps from
``.github/workflows/WORKFLOW`` in order, each with ``bash -e`` from the
clone's root -- GitHub's default shell -- under the workflow's, the
job's and the step's ``env:``, with ``GITHUB_WORKSPACE`` set to the
clone and ``RUNNER_TEMP`` to ``WORK/runner-temp``.  A ``uses:`` step is
skipped (the clone stands in for ``actions/checkout``, the local ``uv``
for ``setup-uv``), and so is a step whose ``if:`` names an event
(``github.event_name``), reported as skipped.  Any other GitHub
expression (``${{ ... }}``) is refused: the jobs this runs are the ones
a release depends on, ``lint`` and ``core-only``, which need none.
Each step's output goes to the terminal; the run stops at the first
failing step, and the exit status is that step's.  The clone and the
runner directory are removed afterwards (with *WORK*, if this run made
it) unless ``--keep``.

It checks a release before anything is pushed: the ``core-only`` job
proves that the stripped package builds, holds no lab or agent file,
installs alone and runs the solver.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
#: The repository holding this script (``lab/scripts/release/``).
REPO = HERE.parents[2]


def _env(*layers: dict | None) -> dict[str, str]:
    """The merged ``env:`` layers, refusing a GitHub expression."""
    out: dict[str, str] = {}
    for layer in layers:
        for key, value in (layer or {}).items():
            if "${{" in str(value):
                raise SystemExit(f"env {key}: an expression ({value})")
            out[key] = str(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("job")
    ap.add_argument("--workflow", default="ci.yml")
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    path = REPO / ".github" / "workflows" / a.workflow
    workflow = yaml.safe_load(path.read_text())
    jobs = workflow.get("jobs", {})
    if a.job not in jobs:
        raise SystemExit(f"no job {a.job!r} in {a.workflow}: {list(jobs)}")
    job = jobs[a.job]
    if "if" in job or "strategy" in job:
        raise SystemExit(f"job {a.job!r} needs a GitHub context to run")
    base = _env(workflow.get("env"), job.get("env"))

    made = a.work is None
    work = a.work or Path(tempfile.mkdtemp(prefix=f"ci-{a.job}-"))
    work = work.expanduser().resolve()
    clone, runner_temp = work / "clone", work / "runner-temp"
    shutil.rmtree(clone, ignore_errors=True)
    shutil.rmtree(runner_temp, ignore_errors=True)
    runner_temp.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", "--no-checkout", str(REPO), str(clone)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "checkout", "-q", "--detach", a.commit],
        check=True,
    )
    print(f"{a.workflow} {a.job} at {a.commit} in {clone}", flush=True)

    rc = 0
    try:
        for k, step in enumerate(job.get("steps", []), 1):
            name = step.get("name") or step.get("uses") or step.get("run")
            name = str(name).strip().splitlines()[0]
            if "uses" in step:
                print(f"--- [{k}] skipped (uses): {name}", flush=True)
                continue
            cond = str(step.get("if", ""))
            if "github.event_name" in cond:
                print(f"--- [{k}] skipped (if: {cond}): {name}", flush=True)
                continue
            if "${{" in step["run"] or cond:
                raise SystemExit(f"step {k} ({name}): an expression")
            env = {
                **os.environ,
                **base,
                **_env(step.get("env")),
                "GITHUB_WORKSPACE": str(clone),
                "RUNNER_TEMP": str(runner_temp),
            }
            print(f"--- [{k}] {name}", flush=True)
            rc = subprocess.run(
                ["bash", "-e", "-c", step["run"]], cwd=clone, env=env
            ).returncode
            if rc != 0:
                print(f"--- [{k}] FAILED (exit {rc}): {name}", flush=True)
                break
        else:
            print(f"--- {a.job}: every run step passed", flush=True)
    finally:
        if not a.keep:
            shutil.rmtree(clone, ignore_errors=True)
            shutil.rmtree(runner_temp, ignore_errors=True)
            if made:
                shutil.rmtree(work, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
