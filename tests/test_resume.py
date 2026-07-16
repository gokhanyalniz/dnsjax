"""Resume-lineage tests: snapshot counter, naming, and trajectory policy.

Two layers:

1. **Offline units** (pure Pydantic, no JAX / no ``mpirun``):
   :func:`dnsjax.parameters.trajectory_defining_changes` against
   stored dumps in the public-named (v4) representation
   (``recorded_params_dump``) -- identical params yield no change; a
   ``phys``/``geo``/``res`` override is reported; the JAX-setup skip
   field ``res.double_precision`` and non-trajectory sections
   (``step``) are ignored; a stored key the current surface no longer
   defines (e.g. the retired ``geo.axis_gap``) is dropped, while
   switching ``geo.grid_type`` (rigged <-> half-CGL) *is* a trajectory
   change; the trajectory-defining ``force`` extension section is
   compared alongside; a pre-v4 ``format_version`` is rejected.  Plus
   ``run_grid_validation_checks`` (the half-CGL rules: rejected for
   non-cylindrical systems and with ``cnab2``, accepted on the pipe
   with ``iterative-cn``) and ``run_grid_default_resolution_checks``
   (``update_parameters`` resolves an unset ``geo.grid_type`` from
   the flow spec: cylindrical ``"half-cgl"`` under ``iterative-cn``
   / ``"rigged-cgl"`` under cnab2, ``"cgl"`` for Cartesian/annular,
   ``None`` for periodic and ``wall_grid`` runs; a user-set value
   survives layer flips).

2. **Subprocess integration** driving ``python -m dnsjax`` (via
   ``mpirun``, one device) end to end in temporary directories,
   asserting the new snapshot behaviour:

   - a non-snapshot start saves its IC as ``state00000.tar`` and a final
     ``stateNNNNN.tar`` on termination, each carrying an ``isnap``
     metadata field and an embedded ``_dnsjax_stats.json`` member;
   - a resume with unchanged Physics/Geometry/Resolution **continues**
     the lineage (no IC re-save; numbering picks up at the resumed
     index + 1) -- even with competing ``--init.random_field`` /
     ``--init.start_from_laminar`` flags passed, confirming a provided
     snapshot takes precedence over every in-process init mode;
   - a resume with a changed ``phys.re`` starts a **new trajectory**
     (``t=it=isnap=0``, a fresh ``state00000.tar``, diagnostic on
     stdout) unless ``init.force_resume`` is set, which forces a
     continuation (no fresh ``state00000.tar``).

Run as a script::

    uv run python tests/test_resume.py            # unit + integration
    uv run python tests/test_resume.py --unit-only
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile

# ── configuration ────────────────────────────────────────────────────

# A short, low-resolution plane-Couette run from a random IC.  Three
# steps at dt = 0.01 (it_snapshot = 1) produce state00000 (IC),
# state00001/2 (periodic) and a final snapshot.
RUN1_ARGS: list[str] = [
    "--phys.system",
    "plane-couette",
    "--phys.re",
    "330",
    "--geo.lx",
    "5",
    "--geo.lz",
    "5",
    "--res.nx",
    "8",
    "--res.ny",
    "16",
    "--res.nz",
    "8",
    "--init.random_field",
    "True",
    "--init.random_amplitude",
    "0.1",
    "--init.random_seed",
    "1",
    "--step.dt",
    "0.01",
    "--outs.it_snapshot",
    "1",
    "--outs.it_stats",
    "1",
    "--stop.max_sim_time",
    "0.03",
    "--stop.check_laminarization",
    "False",
]

_SNAP_RE = re.compile(r"^state(\d+)\.tar$")


# ── unit test ────────────────────────────────────────────────────────


def run_unit_checks() -> bool:
    """Offline units of ``trajectory_defining_changes`` and the
    ``read_snapshot_params`` solver-section skip (no JAX/mpirun).

    Stored dumps are built with ``recorded_params_dump`` -- the
    public-named (v4) representation snapshots actually embed -- so
    these units exercise the internalize-on-compare path.
    """
    from dnsjax.param_surface import recorded_params_dump
    from dnsjax.parameters import params, trajectory_defining_changes

    name = "trajectory_defining_changes"
    snap = recorded_params_dump(params)  # baseline == current params

    try:
        # Identical params -> a continuation (no changes).
        assert trajectory_defining_changes(snap) == [], "identical"

        # A phys / geo / res override is detected and reported.
        for section, key in (("phys", "re"), ("geo", "lx"), ("res", "ny")):
            model = getattr(params, section)
            old = getattr(model, key)
            setattr(model, key, old + 1)
            changes = trajectory_defining_changes(snap)
            setattr(model, key, old)
            assert any(c.startswith(f"{section}.{key}:") for c in changes), (
                section,
                key,
                changes,
            )

        # res.double_precision is a JAX-setup skip field -> not a change.
        old_dp = params.res.double_precision
        params.res.double_precision = not old_dp
        changes = trajectory_defining_changes(snap)
        params.res.double_precision = old_dp
        assert changes == [], ("double_precision", changes)

        # A non-trajectory section (step) is ignored entirely.
        old_dt = params.step.dt
        params.step.dt = old_dt + 1
        changes = trajectory_defining_changes(snap)
        params.step.dt = old_dt
        assert changes == [], ("step.dt", changes)

        # A stored key the current surface no longer defines (e.g. the
        # retired geo.axis_gap) is dropped by the internalization (with
        # a note), so such a snapshot resumes as a clean continuation.
        snap_legacy = recorded_params_dump(params)
        snap_legacy["geo"]["axis_gap"] = 0
        changes = trajectory_defining_changes(snap_legacy)
        assert changes == [], ("legacy axis_gap ignored", changes)

        # Switching geo.grid_type (rigged <-> half-CGL) *is* a
        # trajectory change (the radial grid differs).
        old_gt = params.geo.grid_type
        params.geo.grid_type = "half-cgl"
        changes = trajectory_defining_changes(snap)
        params.geo.grid_type = old_gt
        assert any(c.startswith("geo.grid_type:") for c in changes), (
            "grid_type change",
            changes,
        )

        # The trajectory-defining ``force`` extension: turning kicks
        # on over a kick-free snapshot is a trajectory change; a
        # stored section matching the live singleton is not.
        from dnsjax.extensions import force_params

        force_params.amplitude = 0.5
        changes = trajectory_defining_changes(snap)
        assert any(c.startswith("force.amplitude:") for c in changes), (
            "force extension change",
            changes,
        )
        snap_forced = recorded_params_dump(params)  # embeds [force]
        changes = trajectory_defining_changes(snap_forced)
        force_params.amplitude = None
        assert changes == [], ("force extension match", changes)

        # A snapshot embedding solver params the current model no
        # longer defines (a retired backend name and its knobs) must
        # still resume: ``read_snapshot_params`` drops the
        # execution-only [solver] section before validation.  The
        # stored [probes]/[force] sections come back as extension
        # overlays (pure defaults here: nothing was configured).
        import io
        import json
        import tarfile
        import tempfile
        from pathlib import Path

        from dnsjax.parameters import read_snapshot_params
        from dnsjax.snapshot_meta import META_MEMBER

        legacy = recorded_params_dump(params)
        legacy["solver"] = {
            "backend": "some-retired-backend",
            "retired_knob": 8,
        }
        payload = json.dumps(
            {"format_version": 4, "system": "plane-couette", "params": legacy}
        ).encode()
        with tempfile.NamedTemporaryFile(suffix=".tar") as fh:
            with tarfile.open(fh.name, "w") as tf:
                info = tarfile.TarInfo(META_MEMBER)
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            snap = read_snapshot_params(Path(fh.name))
        assert snap is not None, "legacy snapshot params unreadable"
        loaded, ext_overlays = snap
        assert loaded.solver.backend == "pallas", (
            "retired solver params inherited",
            loaded.solver,
        )
        assert set(ext_overlays) == {"probes", "force"}, ext_overlays
        from dnsjax.extensions import EXTENSIONS

        for sec_name, sec in ext_overlays.items():
            defaults = EXTENSIONS[sec_name].model().model_dump(mode="json")
            assert sec == defaults, (
                "unconfigured overlay must equal defaults",
                sec_name,
                sec,
            )

        # A pre-v4 snapshot (old params representation) is rejected
        # outright at the metadata read.
        payload3 = json.dumps({"format_version": 3, "params": legacy}).encode()
        with tempfile.NamedTemporaryFile(suffix=".tar") as fh:
            with tarfile.open(fh.name, "w") as tf:
                info = tarfile.TarInfo(META_MEMBER)
                info.size = len(payload3)
                tf.addfile(info, io.BytesIO(payload3))
            try:
                read_snapshot_params(Path(fh.name))
            except ValueError as exc:
                assert "format_version" in str(exc), exc
            else:
                raise AssertionError("format_version 3 was accepted")
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return False

    print(f"  PASS  {name}")
    return True


def run_grid_validation_checks() -> bool:
    """Offline unit: ``geo.grid_type='half-cgl'`` validation rules."""
    from dnsjax.parameters import params, validate_parameters

    name = "grid_type half-cgl validation"
    save = (
        params.phys.system,
        params.geo.grid_type,
        params.step.scheme,
    )

    def raises() -> bool:
        try:
            validate_parameters()
        except ValueError:
            return True
        return False

    ok = True
    try:
        # half-CGL on a non-cylindrical system -> rejected.
        params.phys.system = "plane-couette"
        params.geo.grid_type = "half-cgl"
        params.step.scheme = "iterative-cn"
        ok = ok and raises()

        # half-CGL on the pipe with the explicit cnab2 scheme ->
        # rejected (destabilises the explicit scheme near the axis).
        params.phys.system = "pipe"
        params.step.scheme = "cnab2"
        ok = ok and raises()

        # half-CGL on the pipe with iterative-cn -> accepted.
        params.step.scheme = "iterative-cn"
        ok = ok and not raises()
    finally:
        (params.phys.system, params.geo.grid_type, params.step.scheme) = save

    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def _restore_params(saved_dump: dict, saved_user_set: set) -> None:
    """Restore the sections the grid units mutate (plus the
    ``_user_set_fields`` layer bookkeeping)."""
    import dnsjax.parameters as P

    for section in ("phys", "geo", "res", "step"):
        model = getattr(P.params, section)
        for key, value in saved_dump[section].items():
            setattr(model, key, value)
    P._user_set_fields.clear()
    P._user_set_fields.update(saved_user_set)


def run_grid_default_resolution_checks() -> bool:
    """Offline unit: scheme-dependent ``geo.grid_type`` resolution.

    ``update_parameters`` resolves an unset ``grid_type`` to a
    concrete per-family/per-scheme value (re-resolved on every layer,
    like ``solver.backend``): cylindrical half-CGL under
    ``iterative-cn``, rigged (``"cgl"``) under cnab2, ``"cgl"`` for
    Cartesian/annular, ``None`` for periodic systems and custom
    ``wall_grid`` runs; an explicitly-set value is never overridden.
    """
    import dnsjax.parameters as P
    from dnsjax.parameters import Parameters, params, update_parameters

    name = "grid_type default resolution"
    saved_dump = params.model_dump(mode="json")
    saved_user_set = set(P._user_set_fields)

    def resolve(**layers) -> str | None:
        update_parameters(Parameters(**layers))
        return params.geo.grid_type

    try:
        # Cylindrical + iterative-cn (the defaults) -> half-CGL.
        P._user_set_fields.discard(("geo", "grid_type"))
        got = resolve(phys={"system": "pipe"}, step={"scheme": "iterative-cn"})
        assert got == "half-cgl", ("pipe + iterative-cn", got)

        # A later cnab2 layer re-resolves to rigged, and back.
        got = resolve(step={"scheme": "cnab2"})
        assert got == "rigged-cgl", ("pipe + cnab2", got)
        got = resolve(step={"scheme": "iterative-cn"})
        assert got == "half-cgl", ("pipe + iterative-cn again", got)

        # Cartesian / annular -> full CGL under either scheme.
        got = resolve(phys={"system": "plane-couette"})
        assert got == "cgl", ("cartesian", got)
        got = resolve(phys={"system": "dean"}, geo={"eta": 0.5})
        assert got == "cgl", ("annular", got)

        # A user-set value survives system / scheme flips.
        got = resolve(phys={"system": "pipe"}, geo={"grid_type": "half-tanh"})
        assert got == "half-tanh", ("user-set", got)
        got = resolve(step={"scheme": "cnab2"})
        assert got == "half-tanh", ("user-set survives", got)

        # Periodic systems stay None.
        P._user_set_fields.discard(("geo", "grid_type"))
        params.geo.grid_type = None
        got = resolve(phys={"system": "kolmogorov"})
        assert got is None, ("periodic", got)

        # A custom wall_grid keeps grid_type unset (in particular, no
        # spurious "cannot set both" from an earlier resolution).
        got = resolve(
            phys={"system": "plane-couette"}, step={"scheme": "cnab2"}
        )
        assert got == "cgl", ("pre-wall_grid", got)
        with tempfile.NamedTemporaryFile(suffix=".txt") as fh:
            got = resolve(geo={"wall_grid": fh.name})
        assert got is None, ("wall_grid", got)
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return False
    finally:
        _restore_params(saved_dump, saved_user_set)

    print(f"  PASS  {name}")
    return True


def _snap_indices(workdir: str) -> list[int]:
    """Sorted ``isnap`` indices of the ``stateNNNNN.tar`` files in dir."""
    out = []
    for fname in os.listdir(workdir):
        m = _SNAP_RE.match(fname)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


# JAX backend for the mpirun integration children; set from
# --dist.platform in __main__ (cuda runs the resume integration on a GPU).
_PLATFORM = "cpu"


def _run_dnsjax(
    workdir: str, extra_args: list[str], timeout: float
) -> subprocess.CompletedProcess:
    """Run ``mpirun -np 1 python -m dnsjax <extra>`` with cwd *workdir*."""
    cmd = [
        "mpirun",
        "-np",
        "1",
        sys.executable,
        "-m",
        "dnsjax",
        "--dist.platform",
        _PLATFORM,
        "--dist.np0",
        "1",
        "--dist.np1",
        "1",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=workdir,
    )


def _fail(stage: str, res: subprocess.CompletedProcess) -> None:
    print(f"  FAIL  integration: {stage} exit {res.returncode}")
    print(res.stdout[-2000:] if res.stdout else "(no stdout)")
    print(res.stderr[-2000:] if res.stderr else "(no stderr)")


def run_integration(timeout: float) -> bool:
    """End-to-end snapshot-lineage and resume-policy checks."""
    from dnsjax.snapshot_meta import read_snapshot_meta, read_snapshot_stats

    # Reuse the random-smoke "clean integration" check for run 1.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_random_smoke import _check_run

    name = "resume lineage"
    base = tempfile.mkdtemp(prefix="resume_")
    work1 = os.path.join(base, "run1")
    work_new = os.path.join(base, "new_traj")
    work_force = os.path.join(base, "force")
    for d in (work1, work_new, work_force):
        os.makedirs(d)

    try:
        # --- Run 1: fresh start from a random IC ---------------------
        r1 = _run_dnsjax(work1, RUN1_ARGS, timeout)
        if r1.returncode != 0:
            _fail("run1", r1)
            return False
        _check_run(r1.stdout, "run1", 0.03, 0.01)  # raises on a bad run

        idx1 = _snap_indices(work1)
        assert 0 in idx1, f"IC state00000.tar missing: {idx1}"
        final1 = max(idx1)
        assert final1 >= 1, f"no final snapshot beyond the IC: {idx1}"
        final_path = os.path.abspath(
            os.path.join(work1, f"state{final1:05d}.tar")
        )

        ic_meta = read_snapshot_meta(os.path.join(work1, "state00000.tar"))
        assert ic_meta["isnap"] == 0, ic_meta["isnap"]
        assert ic_meta["t"] == 0.0, ic_meta["t"]
        ic_stats = read_snapshot_stats(os.path.join(work1, "state00000.tar"))
        assert ic_stats, f"IC snapshot has no embedded stats: {ic_stats}"

        final_meta = read_snapshot_meta(final_path)
        assert final_meta["isnap"] == final1, final_meta["isnap"]
        assert final_meta["t"] > 0.0, final_meta["t"]
        assert read_snapshot_stats(final_path), "final snapshot lacks stats"

        # --- Run 2: continuation resume (same params, same dir) ------
        # Also pass competing in-process init flags (random_field +
        # start_from_laminar): a provided snapshot must take precedence
        # over every other init mode, so this still resumes (and, with
        # unchanged params, continues the trajectory).
        r2 = _run_dnsjax(
            work1,
            [
                "--init.snapshot",
                final_path,
                "--init.random_field",
                "True",
                "--init.start_from_laminar",
                "True",
                "--outs.it_snapshot",
                "1",
                "--stop.max_sim_time",
                "0.05",
            ],
            timeout,
        )
        if r2.returncode != 0:
            _fail("run2 (continuation)", r2)
            return False
        assert "NEW trajectory" not in r2.stdout, "unexpected new trajectory"
        assert "Resumed from snapshot" in r2.stdout, r2.stdout[-1500:]
        # Snapshot precedence: neither in-process IC was used despite the
        # competing flags above.
        assert "in-process random IC" not in r2.stdout, (
            "snapshot must take precedence over --init.random_field"
        )
        idx2 = _snap_indices(work1)
        assert (final1 + 1) in idx2, (final1, idx2)
        assert max(idx2) > final1, (final1, idx2)

        # --- Run 3: changed phys.re -> new trajectory (fresh dir) ----
        r3 = _run_dnsjax(
            work_new,
            [
                "--init.snapshot",
                final_path,
                "--phys.re",
                "660",
                "--outs.it_snapshot",
                "1",
                "--stop.max_sim_time",
                "0.03",
            ],
            timeout,
        )
        if r3.returncode != 0:
            _fail("run3 (new trajectory)", r3)
            return False
        assert "NEW trajectory" in r3.stdout, r3.stdout[-1500:]
        idx3 = _snap_indices(work_new)
        assert 0 in idx3, f"new trajectory must save state00000: {idx3}"
        new_ic = read_snapshot_meta(os.path.join(work_new, "state00000.tar"))
        assert new_ic["t"] == 0.0, new_ic["t"]
        assert new_ic["params"]["phys"]["re"] == 660, new_ic["params"]["phys"][
            "re"
        ]

        # --- Run 4: changed phys.re + force_resume -> continuation ---
        r4 = _run_dnsjax(
            work_force,
            [
                "--init.snapshot",
                final_path,
                "--phys.re",
                "660",
                "--init.force_resume",
                "True",
                "--outs.it_snapshot",
                "1",
                "--stop.max_sim_time",
                "0.05",
            ],
            timeout,
        )
        if r4.returncode != 0:
            _fail("run4 (force_resume)", r4)
            return False
        assert "NEW trajectory" not in r4.stdout, "force_resume reset anyway"
        assert 0 not in _snap_indices(work_force), (
            "force_resume continuation must not save state00000"
        )
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return False
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)

    print(f"  PASS  {name}")
    return True


# ── main ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume-lineage tests")
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only the offline unit (skip the mpirun integration)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-run subprocess timeout in seconds",
    )
    parser.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=["cpu", "cuda", "rocm", "tpu"],
        help="JAX backend for the mpirun integration children "
        "(default cpu; cuda runs the resume integration on a GPU).",
    )
    cli = parser.parse_args()
    _PLATFORM = cli.platform

    if not cli.unit_only:
        print(
            f"Resume integration on platform '{_PLATFORM}' via "
            "mpirun -np 1; each child prints its own device banner. "
            "(Offline units are device-independent.)",
            flush=True,
        )

    passed = failed = 0
    offline = [
        run_unit_checks(),
        run_grid_validation_checks(),
        run_grid_default_resolution_checks(),
    ]
    for ok in (
        offline if cli.unit_only else [*offline, run_integration(cli.timeout)]
    ):
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
