"""Resume checks of the lab's ``[probes]`` and ``[force]`` sections.

The lab's part of ``tests/test_resume.py`` (the shared fixtures:
``tests/_resume.py``), offline (pure Pydantic, no JAX, no ``mpirun``):

1. The trajectory-defining ``force`` section is compared alongside the
   core sections: turning kicks on over a kick-free snapshot is a
   trajectory change, a stored section matching the live one is not.
2. An inherited ``[probes]`` / ``[force]`` configuration switched off
   by its cadence's ``0`` returns to its defaults instead of failing
   its "set together" check on what it inherited, and the recorded
   dump then leaves both sections out (a section is recorded only
   while it is configured).

Run as a script::

    uv run python lab/tests/test_resume.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import _repo  # noqa: F401 -- the core suite's helpers
from _live import report
from _resume import write_meta_tar

sys.stdout.reconfigure(line_buffering=True)


def run_force_trajectory() -> str | None:
    """``[force]`` is a trajectory-defining section."""
    from dnsjax_lab.extensions import force_params

    from dnsjax.param_surface import recorded_params_dump
    from dnsjax.parameters import params, trajectory_defining_changes

    snap = recorded_params_dump(params)
    try:
        assert "force" not in snap, ("unconfigured [force] recorded", snap)
        force_params.modes = "1,1"
        force_params.amplitude = 0.5
        changes = trajectory_defining_changes(snap)
        assert any(c.startswith("force.amplitude:") for c in changes), (
            "force extension change",
            changes,
        )
        snap_forced = recorded_params_dump(params)
        assert snap_forced.get("force", {}).get("amplitude") == 0.5, (
            "configured [force] not recorded",
            snap_forced.get("force"),
        )
        changes = trajectory_defining_changes(snap_forced)
        assert changes == [], ("force extension match", changes)
    except AssertionError as exc:
        return str(exc)
    finally:
        force_params.modes = None
        force_params.amplitude = None
    return None


def run_off_cadences() -> str | None:
    """An inherited ``[probes]`` / ``[force]`` switches off at ``0``."""
    from dnsjax_lab.extensions import force_params, probes_params

    import dnsjax.parameters as P
    from dnsjax.bootstrap import resolve_parameters
    from dnsjax.extensions import reset_extensions
    from dnsjax.param_surface import recorded_params_dump
    from dnsjax.parameters import params

    saved = params.model_dump()
    saved_user_set = set(P._user_set_fields)
    configured = dict(
        recorded_params_dump(params),
        probes={"modes": "1,1", "it_probes": 5},
        force={
            "modes": "1,1",
            "profiles": "profiles.npz",
            "amplitude": 0.1,
            "it_force": 5,
            "seed": 7,
        },
    )
    meta = {"format_version": 6, "system": "plane-couette"}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            forced = Path(tmp) / "forced.tar"
            write_meta_tar(forced, meta | {"params": configured})
            resolve_parameters(
                [
                    "--init.snapshot",
                    str(forced),
                    "--probes.it_probes",
                    "0",
                    "--force.it_force",
                    "0",
                ],
                toml_path=False,
            )
            for values in (probes_params, force_params):
                fresh = type(values)()
                assert values == fresh, ("section not reset", values)
            dump = recorded_params_dump(params)
            assert "probes" not in dump and "force" not in dump, (
                "switched-off sections recorded",
                {k: dump.get(k) for k in ("probes", "force")},
            )
    except AssertionError as exc:
        return str(exc)
    finally:
        for section, values in saved.items():
            model = getattr(params, section)
            for key, value in values.items():
                setattr(model, key, value)
        P._user_set_fields.clear()
        P._user_set_fields.update(saved_user_set)
        reset_extensions()
    return None


if __name__ == "__main__":
    results = [
        ("[force] trajectory-defining", run_force_trajectory()),
        ("[probes]/[force] off cadences", run_off_cadences()),
    ]
    for name, reason in results:
        print(f"  {'PASS' if reason is None else 'FAIL'}  {name}")
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))
