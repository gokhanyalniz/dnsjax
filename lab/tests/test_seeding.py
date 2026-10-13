r"""Seed-resolution tests for the lab's sections and ``dnsjax-twin``.

The lab's part of ``tests/test_seeding.py`` (the shared fixtures:
``tests/_seeding.py``):

1. **Gating** (offline): ``resolve_run_seeds`` resolves ``force.seed``
   only once the enabling ``[force]`` quartet is configured, checked
   with the entropy source disabled, so a spurious draw *fails*.
2. **Twin adopt-or-draw** (subprocess): a fresh ``dnsjax-twin`` start
   with no ``--twin.seed`` draws and records it, and the paired resume
   then adopts it from ``twin.json`` instead of drawing -- without
   which every un-pinned twin run would fail its first resume on the
   ``_TWIN_MATCH_KEYS`` seed check.  The mismatch guard itself is
   still exercised by ``lab/tests/test_twin_driver.py``.

Run as a script::

    uv run python lab/tests/test_seeding.py             # both
    uv run python lab/tests/test_seeding.py --unit-only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import _repo  # noqa: F401 -- the core suite's helpers
from _live import report
from _seeding import BIN, RUN_ARGS, no_entropy, run

from dnsjax.seeding import SOURCE_CLI, SOURCE_DRAWN

_TWIN_SEED_RE = re.compile(r"^twin\.seed = (\d+) \((.+)\)$", re.M)


def run_force_gating() -> str | None:
    """An unconfigured ``[force]`` resolves no seed, entropy disabled."""
    from dnsjax_lab.extensions import force_params

    from dnsjax.bootstrap import ResolvedSetup, resolve_run_seeds
    from dnsjax.flows.registry import spec_for
    from dnsjax.parameters import (
        Parameters,
        update_parameters,
        validate_parameters,
    )

    try:
        setup = ResolvedSetup(
            system="plane-couette",
            spec=spec_for("plane-couette"),
            params_from_disk=False,
            snapshot_path=None,
            snapshot_params_used=False,
            seed_layers={"init.random_seed": SOURCE_CLI},
        )
        update_parameters(
            Parameters(
                phys={"system": "plane-couette"},
                res={"nx": 8, "ny": 27, "nz": 8},
                init={"random_seed": 4242},
            )
        )
        validate_parameters()
        if force_params.modes is not None:  # pragma: no cover
            return "the [force] section leaked in from another test"
        with no_entropy():
            if len(resolve_run_seeds(setup)) != 1:
                return "an unconfigured [force] resolved a seed"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def run_twin_adoption(timeout: float) -> str | None:
    """A fresh twin draws; its paired resume adopts, and so proceeds."""
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_seed_twin_"))
    try:
        parent, member = tmp / "parent", tmp / "m0"
        parent.mkdir()
        member.mkdir()
        res = run(
            [
                str(BIN / "dnsjax"),
                *RUN_ARGS,
                "--init.random_seed",
                "1",
                "--outs.it_snapshot",
                "2",
            ],
            parent,
            timeout,
        )
        if res.returncode != 0:
            return f"the parent run exited {res.returncode}"
        snaps = sorted(parent.glob("state*.tar"))
        if not snaps:
            return "the parent run wrote no snapshot"
        shutil.copy(snaps[-1], member / snaps[-1].name)

        def twin_args(snapshot: str, horizon: str) -> list[str]:
            # ``max_sim_time`` is the horizon past the snapshot the
            # launch starts from, not an absolute stop time.
            return [
                str(BIN / "dnsjax-twin"),
                *RUN_ARGS,
                "--init.snapshot",
                snapshot,
                "--stop.max_sim_time",
                horizon,
                "--twin.e0",
                "1e-6",
                "--twin.it_energy",
                "1",
            ]

        # Fresh start: draws, prints, records.
        res = run(twin_args(snaps[-1].name, "0.005"), member, timeout)
        if res.returncode != 0:
            return f"the fresh twin start exited {res.returncode}"
        match = _TWIN_SEED_RE.search(res.stdout)
        if match is None or match.group(2) != SOURCE_DRAWN:
            got = None if match is None else match.group(2)
            return f"an unset twin.seed was reported as {got!r}"
        seed = int(match.group(1))
        recorded = json.loads((member / "twin.json").read_text())["seed"]
        if recorded != seed:
            return f"twin.json recorded {recorded}, the run drew {seed}"

        # Paired resume, still unset: adopts rather than re-drawing --
        # a fresh draw would fail the _TWIN_MATCH_KEYS seed check.  A
        # resume points at the newest pair the member wrote, as the
        # driver's own "no partner snapshot" diagnostic instructs.
        pairs = sorted(
            q
            for q in member.glob("state*.tar")
            if not q.name.endswith("_twin.tar")
            and q.with_name(f"{q.stem}_twin.tar").exists()
        )
        if not pairs:
            return "the fresh twin start wrote no reference/partner pair"
        res = run(twin_args(pairs[-1].name, "0.01"), member, timeout)
        if res.returncode != 0:
            return (
                f"the paired resume exited {res.returncode}; an unset "
                "twin.seed did not adopt the recorded one"
            )
        match = _TWIN_SEED_RE.search(res.stdout)
        if match is None or int(match.group(1)) != seed:
            got = None if match is None else match.group(1)
            return f"the resume reported seed {got}, expected {seed}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return None


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed-resolution tests")
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only the offline gating check (skip the twin runs)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-run subprocess timeout in seconds",
    )
    cli = parser.parse_args()
    results: list[tuple[str, str | None]] = [
        ("[force] seed gating", run_force_gating()),
    ]
    if not cli.unit_only:
        results.append(("twin adopt-or-draw", run_twin_adoption(cli.timeout)))
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))
