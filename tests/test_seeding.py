r"""Seed-resolution tests: the ``unset means draw one`` contract.

Covers :mod:`dnsjax.seeding` and the two resolution points that use it
(``bootstrap.resolve_seed`` / ``resolve_run_seeds``, and the twin
driver's late adopt-or-draw):

1. **Units** (offline, no JAX): :func:`draw_seed` stays inside the
   62-bit range and does not repeat; :func:`split_seed` /
   :func:`join_seed` round-trip exactly at the range's ends -- the
   property the cross-process ``int32`` payload rests on, since
   ``jax_enable_x64`` follows ``res.double_precision`` and an
   ``int64`` payload would be truncated in single precision.
2. **Refusal**: with ``os.urandom`` raising, an unset seed that the run
   would draw with exits naming the flag that fixes it, while a
   supplied seed still resolves -- i.e. the refusal is scoped to the
   draw, not to the run.
3. **Gating**: ``resolve_run_seeds`` resolves ``init.random_seed`` only
   when the start mode is the random IC (a laminar / rolls /
   snapshot-resume run needs no entropy at all) and ``force.seed``
   only when the enabling ``[force]`` quartet is configured.  Both are
   checked with the entropy source disabled, so a spurious draw
   *fails* rather than passing silently.
4. **Provenance**: the layer -> label mapping, and that a seed a layer
   supplied is used unchanged and reported with that layer's label,
   never re-drawn.  Writing a drawn seed back is also checked to touch
   nothing else: it goes through ``update_parameters``, which re-runs
   the flow's derive hook and the restore-then-materialize pass
   *after* ``validate_parameters`` already ran.
5. **Reproducibility** (subprocess): a solver run with no
   ``--init.random_seed`` prints the drawn seed, and re-running at that
   printed value reproduces ``stats.dat`` byte for byte.  This is the
   guard that the seed printed is the seed used.
6. **Multi-process agreement** (``mpirun -np 2``): an unset seed is
   drawn once and broadcast, so the two ranks build one random field.
   Checked against a single-process run at the printed seed -- the
   failure this catches (per-rank draws) is silent otherwise: the
   field stays divergence-free and correctly normalised, and the
   snapshot records rank 0's seed for a trajectory no seed reproduces.
7. **Twin adopt-or-draw** (subprocess): a fresh ``dnsjax-twin`` start
   with no ``--twin.seed`` draws and records it, and the paired resume
   then adopts it from ``twin.json`` instead of drawing -- without
   which every un-pinned twin run would fail its first resume on the
   ``_TWIN_MATCH_KEYS`` seed check.  The mismatch guard itself is
   still exercised by ``tests/test_twin_driver.py``.

Run as a script::

    uv run python tests/test_seeding.py             # units + CLI runs
    uv run python tests/test_seeding.py --unit-only
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import report, run_live  # noqa: E402

from dnsjax.seeding import (  # noqa: E402
    SEED_BITS,
    SOURCE_CLI,
    SOURCE_DRAWN,
    SOURCE_SNAPSHOT,
    SOURCE_TOML,
    NoEntropySource,
    draw_seed,
    join_seed,
    missing_entropy_message,
    seed_note,
    split_seed,
)

_MAX_SEED = (1 << SEED_BITS) - 1
_BIN = Path(sys.executable).parent
_SEED_RE = re.compile(r"^init\.random_seed = (\d+) \((.+)\)$", re.M)
_TWIN_SEED_RE = re.compile(r"^twin\.seed = (\d+) \((.+)\)$", re.M)

# Smallest plane-Couette box that still exercises the random IC on both
# mesh axes; two steps is enough for stats.dat to differ between seeds.
_RUN_ARGS = [
    "--phys.system",
    "plane-couette",
    "--res.nx",
    "8",
    "--res.nz",
    "8",
    "--res.ny",
    "27",
    "--step.dt",
    "0.005",
    "--stop.max_sim_time",
    "0.015",
    "--outs.it_stats",
    "1",
]


@contextlib.contextmanager
def _no_entropy():
    """``os.urandom`` unavailable, the way a stripped platform reports it.

    ``draw_seed`` catches ``OSError`` and ``NotImplementedError``; this
    raises the former, which is what a sandbox denying ``/dev/urandom``
    produces.
    """
    original = os.urandom

    def fail(_n):
        raise OSError("Function not implemented")

    os.urandom = fail
    try:
        yield
    finally:
        os.urandom = original


def _run(cmd, cwd, timeout):
    """A solver/twin child with the startup dump *kept*.

    ``run_live`` sets ``DNSJAX_QUIET_STARTUP=1`` by default, which
    suppresses the resolved-parameter dump -- but not the seed line,
    which is deliberately outside that gate.  Pinning it to ``0`` here
    would hide a regression that moved the line inside it, so leave the
    default and parse what a quiet run prints.
    """
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    env["NO_COLOR"] = "1"
    return run_live(cmd, cwd=cwd, env=env, timeout=timeout)


# ── 1. Units ─────────────────────────────────────────────────────────


def run_unit_checks() -> str | None:
    """Draw range, non-repetition, and exact transport round-trips."""
    try:
        seeds = {draw_seed() for _ in range(64)}
        for s in seeds:
            if not 0 <= s <= _MAX_SEED:
                return f"draw_seed() returned {s}, outside 0..{_MAX_SEED}"
        if len(seeds) < 64:
            return f"draw_seed() repeated: {len(seeds)} distinct of 64"

        for s in (0, 1, _MAX_SEED, *sorted(seeds)[:4]):
            high, low = split_seed(s)
            if not (0 <= high < (1 << 31) and 0 <= low < (1 << 31)):
                return f"split_seed({s}) -> ({high}, {low}) is not int32-safe"
            if join_seed(high, low) != s:
                return f"join_seed(split_seed({s})) != {s}"

        # Out of range must be rejected, not silently truncated: a
        # truncated seed would name a different trajectory.
        try:
            split_seed(_MAX_SEED + 1)
        except ValueError:
            pass
        else:
            return "split_seed accepted an out-of-range seed"

        note = seed_note("init.random_seed", 7, SOURCE_DRAWN)
        if note != f"init.random_seed = 7 ({SOURCE_DRAWN})":
            return f"seed_note format changed: {note!r}"
        msg = missing_entropy_message(
            "init.random_seed", "--init.random_seed", "no source"
        )
        if "--init.random_seed" not in msg or "[init]" not in msg:
            return f"refusal message names neither flag nor section: {msg!r}"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


# ── 2-4. Resolution, gating and provenance ───────────────────────────


def run_resolution_checks() -> str | None:
    """``resolve_seed`` / ``resolve_run_seeds`` with entropy disabled.

    Disabling the source is what turns "did not draw" from an
    unobservable claim into a failing one: any check below that
    silently drew would exit instead of returning.
    """
    from dnsjax.bootstrap import (
        ResolvedSetup,
        _seed_layers,
        resolve_run_seeds,
        resolve_seed,
    )
    from dnsjax.flows.registry import spec_for
    from dnsjax.parameters import (
        Parameters,
        derived_params,
        params,
        update_parameters,
        validate_parameters,
    )

    try:
        if draw_seed() is None:  # pragma: no cover - shape guard
            return "draw_seed() returned None"

        # Layer -> label, highest priority winning.  The core seed
        # rides the core sections and the two extension seeds their
        # own, so both halves of each layer are exercised.
        found = _seed_layers(
            (SOURCE_CLI, {"init": {"random_seed": 5}}, {}),
            (
                SOURCE_TOML,
                {"init": {"random_seed": 6}},
                {"twin": {"seed": 7}},
            ),
            (
                SOURCE_SNAPSHOT,
                {"init": {"random_seed": 8}},
                {"force": {"seed": 9}, "twin": {"seed": 10}},
            ),
        )
        want = {
            "init.random_seed": SOURCE_CLI,
            "twin.seed": SOURCE_TOML,
            "force.seed": SOURCE_SNAPSHOT,
        }
        if found != want:
            return f"seed provenance is {found!r}, expected {want!r}"
        # A layer that set no seed contributes none.
        if _seed_layers((SOURCE_CLI, {"init": {"random_amplitude": 1.0}}, {})):
            return "a layer with no seed reported one"

        with _no_entropy():
            # A supplied seed never reaches the entropy pool.
            got = resolve_seed("init.random_seed", "--init.random_seed", 42)
            if got != 42:
                return f"a supplied seed was not used unchanged: {got}"
            try:
                draw_seed()
            except NoEntropySource:
                pass
            else:
                return "the _no_entropy() fixture did not disable the source"
            # An unset seed refuses, naming the flag.
            try:
                resolve_seed("init.random_seed", "--init.random_seed", None)
            except SystemExit as exc:
                if "--init.random_seed" not in str(exc):
                    return f"refusal does not name the flag: {exc}"
            else:
                return "an unset seed resolved with no entropy source"

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
            )
        )
        validate_parameters()

        with _no_entropy():
            # (a) The random IC is the start mode but the seed is set:
            #     used unchanged, reported with the layer's label.
            update_parameters(Parameters(init={"random_seed": 4242}))
            notes = resolve_run_seeds(setup)
            want = seed_note("init.random_seed", 4242, SOURCE_CLI)
            if notes != [want]:
                return f"provenance note is {notes!r}, expected [{want!r}]"

            # (b) A start mode that draws nothing needs no entropy --
            #     and must not report a seed it never resolved.
            update_parameters(Parameters(init={"start_from_laminar": True}))
            notes = resolve_run_seeds(setup)
            if notes:
                return f"a laminar start resolved seeds: {notes!r}"

            # (c) An unconfigured [force] likewise.
            from dnsjax.extensions import force_params

            if force_params.modes is not None:  # pragma: no cover
                return "the [force] section leaked in from another test"
            update_parameters(Parameters(init={"start_from_laminar": False}))
            update_parameters(Parameters(init={"random_seed": 4242}))
            if len(resolve_run_seeds(setup)) != 1:
                return "an unconfigured [force] resolved a seed"

        # (d) With entropy available, an unset seed is drawn, lands in
        #     ``params`` (so the snapshot records what ran) and is
        #     reported as drawn.
        params.init.random_seed = None
        notes = resolve_run_seeds(setup)
        if params.init.random_seed is None:
            return "a drawn seed was not written back into params"
        if notes != [
            seed_note(
                "init.random_seed", params.init.random_seed, SOURCE_DRAWN
            )
        ]:
            return f"drawn seed reported as {notes!r}"

        # (e) Writing the seed back goes through ``update_parameters``,
        #     which re-runs the flow's derive hook and the
        #     restore-then-materialize pass -- *after*
        #     ``validate_parameters`` has already run.  That extra pass
        #     must touch nothing but the seed, or the resolution point
        #     would silently re-resolve part of the configuration.
        before = params.model_dump(mode="json")
        d_before = {
            f.name: repr(getattr(derived_params, f.name))
            for f in dataclasses.fields(derived_params)
        }
        update_parameters(Parameters(init={"random_seed": 12345}))
        after = params.model_dump(mode="json")
        changed = {
            f"{sec}.{key}"
            for sec in before
            for key in before[sec]
            if before[sec][key] != after[sec][key]
        }
        if changed != {"init.random_seed"}:
            return (
                "writing the seed back also changed "
                f"{sorted(changed - {'init.random_seed'})}"
            )
        d_after = {
            f.name: repr(getattr(derived_params, f.name))
            for f in dataclasses.fields(derived_params)
        }
        if d_before != d_after:
            return "writing the seed back re-derived the derived params"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


# ── 5. End-to-end reproducibility ────────────────────────────────────


def run_reproducibility(timeout: float) -> str | None:
    """The printed seed re-runs to the same trajectory."""
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_seed_"))
    try:
        drawn, rerun = tmp / "drawn", tmp / "rerun"
        drawn.mkdir()
        rerun.mkdir()
        res = _run([str(_BIN / "dnsjax"), *_RUN_ARGS], drawn, timeout)
        if res.returncode != 0:
            return f"the unseeded run exited {res.returncode}"
        match = _SEED_RE.search(res.stdout)
        if match is None:
            return "no 'init.random_seed = N (source)' line in the banner"
        seed, source = match.group(1), match.group(2)
        if source != SOURCE_DRAWN:
            return f"an unset seed was reported as {source!r}"

        res = _run(
            [
                str(_BIN / "dnsjax"),
                *_RUN_ARGS,
                "--init.random_seed",
                seed,
            ],
            rerun,
            timeout,
        )
        if res.returncode != 0:
            return f"the re-seeded run exited {res.returncode}"
        again = _SEED_RE.search(res.stdout)
        if again is None or again.group(2) != SOURCE_CLI:
            got = None if again is None else again.group(2)
            return f"an explicit seed was reported as {got!r}"

        a = (drawn / "stats.dat").read_bytes()
        b = (rerun / "stats.dat").read_bytes()
        if a != b:
            return "re-running at the printed seed changed stats.dat"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return None


# ── 6. Cross-process agreement ───────────────────────────────────────


def run_multiprocess_agreement(timeout: float) -> str | None:
    """Two ranks draw one seed between them, not one each."""
    import numpy as np

    if shutil.which("mpirun") is None:
        return "mpirun not found"
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_seed_mpi_"))
    try:
        multi, single = tmp / "np2", tmp / "np1"
        multi.mkdir()
        single.mkdir()
        res = _run(
            [
                "mpirun",
                "-np",
                "2",
                str(_BIN / "dnsjax"),
                "--dist.np1",
                "2",
                *_RUN_ARGS,
            ],
            multi,
            timeout,
        )
        if res.returncode != 0:
            return f"the 2-process run exited {res.returncode}"
        seeds = {m.group(1) for m in _SEED_RE.finditer(res.stdout)}
        if len(seeds) != 1:
            return f"the ranks reported {len(seeds)} distinct seeds: {seeds}"
        seed = seeds.pop()

        res = _run(
            [str(_BIN / "dnsjax"), *_RUN_ARGS, "--init.random_seed", seed],
            single,
            timeout,
        )
        if res.returncode != 0:
            return f"the 1-process run exited {res.returncode}"

        a = np.loadtxt(multi / "stats.dat")
        b = np.loadtxt(single / "stats.dat")
        if a.shape != b.shape:
            return f"stats.dat shapes differ: {a.shape} vs {b.shape}"
        # Not bit-identity: a different device count reassociates the
        # global reductions.  A per-rank draw would be O(1) off, not
        # O(eps), so the tolerance is not load-bearing.
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        if not rel < 1e-10:
            return (
                f"the 2-process trajectory differs from the 1-process "
                f"one at the same seed (max rel {rel:.3e}); the ranks "
                "did not agree on a seed"
            )
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return None


# ── 7. Twin adopt-or-draw ────────────────────────────────────────────


def run_twin_adoption(timeout: float) -> str | None:
    """A fresh twin draws; its paired resume adopts, and so proceeds."""
    tmp = Path(tempfile.mkdtemp(prefix="dnsjax_seed_twin_"))
    try:
        parent, member = tmp / "parent", tmp / "m0"
        parent.mkdir()
        member.mkdir()
        res = _run(
            [
                str(_BIN / "dnsjax"),
                *_RUN_ARGS,
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

        def twin_args(snapshot: str, t_end: str) -> list[str]:
            return [
                str(_BIN / "dnsjax-twin"),
                *_RUN_ARGS,
                "--init.snapshot",
                snapshot,
                "--stop.max_sim_time",
                t_end,
                "--twin.e0",
                "1e-6",
                "--twin.it_energy",
                "1",
            ]

        # Fresh start: draws, prints, records.
        res = _run(twin_args(snaps[-1].name, "0.02"), member, timeout)
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
        res = _run(twin_args(pairs[-1].name, "0.03"), member, timeout)
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
        help="Run only the offline units (skip the solver/twin runs)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-run subprocess timeout in seconds",
    )
    cli = parser.parse_args()

    # Each check returns None when it passes, else its one-line reason;
    # ``report`` repeats the failures after the counts (see _live).
    results: list[tuple[str, str | None]] = [
        ("seeding units", run_unit_checks()),
        ("resolution / gating / provenance", run_resolution_checks()),
    ]
    if not cli.unit_only:
        results += [
            ("drawn-seed reproducibility", run_reproducibility(cli.timeout)),
            (
                "cross-process agreement",
                run_multiprocess_agreement(cli.timeout),
            ),
            ("twin adopt-or-draw", run_twin_adoption(cli.timeout)),
        ]

    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))
