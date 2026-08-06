"""Resume-lineage tests: snapshot counter, naming, and trajectory policy.

Two layers:

1. **Offline units** (pure Pydantic, no JAX / no ``mpirun``):
   :func:`dnsjax.parameters.trajectory_defining_changes` against
   stored dumps in the public-named representation
   (``recorded_params_dump``) -- identical params yield no change; a
   ``phys``/``geo``/``res`` override is reported; the JAX-setup skip
   field ``res.double_precision`` and non-trajectory sections
   (``step``) are ignored; a stored core-section key this version does
   not define is a hard ``ValueError``, while the execution-only
   ``[solver]`` section is exempt (and dropped wholesale before
   validation, so a snapshot embedding retired solver knobs still
   resumes); switching ``geo.grid_type`` (rigged <-> half-CGL) *is* a
   trajectory change; the trajectory-defining ``force`` extension
   section is compared alongside; a pre-v6 ``format_version`` is
   rejected.  Plus
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
     continuation (no fresh ``state00000.tar``);
   - an adaptive-dt run (``step.adaptive``) embeds its live (grown)
     ``dt`` in every snapshot; a resume with no explicit ``--step.dt``
     continues at the adapted value (and keeps adapting), while an
     explicit ``--step.dt`` / ``--step.adaptive False`` override beats
     the snapshot layer -- in both cases the lineage continues
     (``step.*`` is not trajectory-defining);
   - an ``--init.snapshot`` that is not a dnsjax snapshot (a typo'd or
     unrelated path) exits nonzero with a naming diagnostic instead of
     falling through to an in-process init mode.

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

from _live import report, run_live

sys.stdout.reconfigure(line_buffering=True)

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


def run_unit_checks() -> str | None:
    """Offline units of ``trajectory_defining_changes`` and the
    ``read_snapshot_params`` solver-section skip (no JAX/mpirun).

    Stored dumps are built with ``recorded_params_dump`` -- the
    public-named representation snapshots actually embed -- so
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

        # A non-trajectory section (step) is ignored entirely -- incl.
        # the adaptive-dt knobs, so a resumed run may adapt onward.
        old_dt = params.step.dt
        params.step.dt = old_dt + 1
        changes = trajectory_defining_changes(snap)
        params.step.dt = old_dt
        assert changes == [], ("step.dt", changes)
        old_ad = params.step.adaptive
        params.step.adaptive = not old_ad
        changes = trajectory_defining_changes(snap)
        params.step.adaptive = old_ad
        assert changes == [], ("step.adaptive", changes)

        # A stored core-section key this version does not define is a
        # hard error: the snapshot means something by it, and resuming
        # against a setup that differs from the stored one with nothing
        # reporting it is the failure mode being refused here.
        snap_unknown = recorded_params_dump(params)
        snap_unknown["geo"]["no_such_field"] = 0
        try:
            trajectory_defining_changes(snap_unknown)
        except ValueError as exc:
            assert "geo.no_such_field" in str(exc), exc
        else:
            raise AssertionError("unknown stored geo key did not raise")

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

        stored = recorded_params_dump(params)
        stored["solver"] = {
            "backend": "some-retired-backend",
            "retired_knob": 8,
        }
        payload = json.dumps(
            {"format_version": 6, "system": "plane-couette", "params": stored}
        ).encode()
        with tempfile.NamedTemporaryFile(suffix=".tar") as fh:
            with tarfile.open(fh.name, "w") as tf:
                info = tarfile.TarInfo(META_MEMBER)
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))
            snap = read_snapshot_params(Path(fh.name))
        assert snap is not None, "stored snapshot params unreadable"
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

        # A pre-v6 snapshot (v5: the decoupled u_pm / spin component
        # basis; v4: the old on-disk array layout) is rejected outright
        # at the metadata read.
        for old in (5, 4):
            payload_old = json.dumps(
                {"format_version": old, "params": stored}
            ).encode()
            with tempfile.NamedTemporaryFile(suffix=".tar") as fh:
                with tarfile.open(fh.name, "w") as tf:
                    info = tarfile.TarInfo(META_MEMBER)
                    info.size = len(payload_old)
                    tf.addfile(info, io.BytesIO(payload_old))
                try:
                    read_snapshot_params(Path(fh.name))
                except ValueError as exc:
                    assert "format_version" in str(exc), exc
                else:
                    raise AssertionError(f"format_version {old} was accepted")
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return str(exc)

    print(f"  PASS  {name}")
    return None


def run_grid_validation_checks() -> str | None:
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
    return None if ok else "a half-cgl accept/reject rule did not hold"


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


def run_grid_default_resolution_checks() -> str | None:
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
        return str(exc)
    finally:
        _restore_params(saved_dump, saved_user_set)

    print(f"  PASS  {name}")
    return None


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
    return run_live(cmd, timeout=timeout, cwd=workdir)


def _fail(stage: str, res: subprocess.CompletedProcess) -> str:
    """Print the failure detail; return the one-line summary reason."""
    reason = f"{stage} exit {res.returncode}"
    print(f"  FAIL  integration: {reason}")
    print(res.stdout[-2000:] if res.stdout else "(no stdout)")
    print(res.stderr[-2000:] if res.stderr else "(no stderr)")
    return reason


def run_integration(timeout: float) -> str | None:
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
            return _fail("run1", r1)
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
            return _fail("run2 (continuation)", r2)
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
            return _fail("run3 (new trajectory)", r3)
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
            return _fail("run4 (force_resume)", r4)
        assert "NEW trajectory" not in r4.stdout, "force_resume reset anyway"
        assert 0 not in _snap_indices(work_force), (
            "force_resume continuation must not save state00000"
        )

        # --- Run 5: adaptive dt -> snapshots embed the live dt -------
        # cfl_cadence 1 + dt_threshold 0 with a tiny CFL grows dt by
        # 1.2x after every step, so the final snapshot must embed an
        # adapted dt > the initial 0.01.
        work_ad = os.path.join(base, "adaptive")
        work_override = os.path.join(base, "override")
        os.makedirs(work_ad)
        os.makedirs(work_override)
        r5 = _run_dnsjax(
            work_ad,
            RUN1_ARGS
            + [
                "--step.adaptive",
                "True",
                "--step.dt_max",
                "0.02",
                "--step.cfl_cadence",
                "1",
                "--step.dt_threshold",
                "0",
            ],
            timeout,
        )
        if r5.returncode != 0:
            return _fail("run5 (adaptive)", r5)
        assert "[adaptive]" in r5.stdout, "no adaptive dt change logged"
        idx5 = _snap_indices(work_ad)
        final5 = max(idx5)
        final5_path = os.path.abspath(
            os.path.join(work_ad, f"state{final5:05d}.tar")
        )
        meta5 = read_snapshot_meta(final5_path)
        ad_dt = meta5["params"]["step"]["dt"]
        assert ad_dt > 0.01, f"live dt not embedded: {ad_dt}"
        assert meta5["params"]["step"]["adaptive"] is True, meta5["params"]

        # --- Run 6: resume with no --step.dt -> continues at the
        # adapted dt (snapshot layer) and keeps adapting -------------
        r6 = _run_dnsjax(
            work_ad,
            [
                "--init.snapshot",
                final5_path,
                "--outs.it_snapshot",
                "1",
                "--stop.max_sim_time",
                "0.1",
            ],
            timeout,
        )
        if r6.returncode != 0:
            return _fail("run6 (adaptive resume)", r6)
        assert "NEW trajectory" not in r6.stdout, "adaptive dt reset lineage"
        assert "Resumed from snapshot" in r6.stdout, r6.stdout[-1500:]
        idx6 = _snap_indices(work_ad)
        assert max(idx6) > final5, (final5, idx6)
        meta6 = read_snapshot_meta(
            os.path.join(work_ad, f"state{max(idx6):05d}.tar")
        )
        assert meta6["params"]["step"]["dt"] >= ad_dt, (
            "resume did not continue at the adapted dt",
            ad_dt,
            meta6["params"]["step"]["dt"],
        )

        # --- Run 7: an explicit --step.dt / --step.adaptive override
        # beats the snapshot layer (still a continuation: step.* is
        # not trajectory-defining) -----------------------------------
        r7 = _run_dnsjax(
            work_override,
            [
                "--init.snapshot",
                final5_path,
                "--step.dt",
                "0.005",
                "--step.adaptive",
                "False",
                "--outs.it_snapshot",
                "1",
                "--stop.max_sim_time",
                "0.1",
            ],
            timeout,
        )
        if r7.returncode != 0:
            return _fail("run7 (dt override)", r7)
        assert "NEW trajectory" not in r7.stdout, "dt override reset lineage"
        assert "[adaptive]" not in r7.stdout, "adaptive off yet controller ran"
        last7 = max(_snap_indices(work_override))
        meta7 = read_snapshot_meta(
            os.path.join(work_override, f"state{last7:05d}.tar")
        )
        assert meta7["params"]["step"]["dt"] == 0.005, meta7["params"]["step"]
        assert meta7["params"]["step"]["adaptive"] is False, meta7["params"][
            "step"
        ]

        # --- Run 8: --init.snapshot that is not a dnsjax snapshot ----
        # Must refuse loudly.  Falling through to an in-process mode
        # would start a run that silently computes something else --
        # a typo'd path is the common case -- so the competing
        # random_field flag below must NOT rescue it.
        work_bad = os.path.join(base, "bad_snapshot")
        os.makedirs(work_bad)
        not_a_snapshot = os.path.join(base, "not_a_snapshot.tar")
        with open(not_a_snapshot, "wb") as fh:
            fh.write(b"this is not a tar archive\n")
        r8 = _run_dnsjax(
            work_bad,
            RUN1_ARGS + ["--init.snapshot", not_a_snapshot],
            timeout,
        )
        assert r8.returncode != 0, (
            "a non-snapshot --init.snapshot must not start a run"
        )
        assert "is not a dnsjax snapshot file" in r8.stdout, r8.stdout[-1500:]
        assert not _snap_indices(work_bad), (
            f"refused run wrote snapshots: {_snap_indices(work_bad)}"
        )
    except AssertionError as exc:
        print(f"  FAIL  {name}: {exc}")
        return str(exc)
    finally:
        import shutil

        shutil.rmtree(base, ignore_errors=True)

    print(f"  PASS  {name}")
    return None


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

    # Each check returns None when it passes, else its one-line reason;
    # ``report`` repeats the failures after the counts (see _live).
    results: list[tuple[str, str | None]] = [
        ("snapshot-meta units", run_unit_checks()),
        ("grid_type half-cgl validation", run_grid_validation_checks()),
        ("grid_type default resolution", run_grid_default_resolution_checks()),
    ]
    if not cli.unit_only:
        results.append(("resume lineage", run_integration(cli.timeout)))

    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))
