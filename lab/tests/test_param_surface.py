r"""Parameter-surface tests of the lab's flows and sections.

The lab's part of ``tests/test_param_surface.py`` (the shared pieces:
``tests/_param_surface.py``), offline and JAX-free:

- the total-field flows are the lab's four force-driven ones, and the
  transient-growth scope takes quasi-Keplerian after the core flows;
- the sPTT surfaces: viscoelastic Dean takes ``geo.m0`` and the
  spanwise blocking, accepts ``[probes]`` and refuses ``[force]``; the
  pipe refuses an sPTT field;
- quasi-Keplerian's derived ``re2`` (a direct assignment is
  overwritten) and its rehydrate; viscoelastic Dean's defaults, wedge
  and derived ``re``;
- the ``[probes]`` / ``[force]`` sections end to end: surface split,
  layering, validation dispatch, and the metadata dump;
- ``dnsjax-twin --help`` lists every ``[twin]`` field, and the
  transient-growth help carries ``[probes]`` exactly where the flow
  accepts it.

Run as a script::

    uv run python lab/tests/test_param_surface.py
"""

from __future__ import annotations

import os
import subprocess
import sys

import _repo  # noqa: F401 -- the core suite's helpers
from _param_surface import check, finish, reset

sys.stdout.reconfigure(line_buffering=True)


def case_total_field() -> None:
    from dnsjax.flows import registry as R

    total = {"curved-pipe", "dean", "viscoelastic-dean", "viscoelastic-pipe"}
    check(
        set(R.total_field_systems) == total,
        "total-field systems are the four force-driven flows",
        R.total_field_systems,
    )
    # In a subprocess: importing the driver registers its section.
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dnsjax.analysis.transient_growth import "
            "WALL_BOUNDED_TG_SYSTEMS as s; print(','.join(s))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    check(
        r.returncode == 0
        and r.stdout.strip()
        == "plane-couette,plane-poiseuille,pipe,taylor-couette,"
        "quasi-keplerian",
        "transient-growth scope adds quasi-keplerian",
        r.stdout.strip() or r.stderr[-300:],
    )


def case_surfaces() -> None:
    from pydantic import ValidationError

    import dnsjax.param_surface as PS
    from dnsjax.flows.registry import spec_for

    pipe_model = PS.build_surface_model(spec_for("pipe"), settings=False)
    try:
        pipe_model.model_validate({"phys": {"el": 80.0}})
        check(False, "pipe rejects an sPTT field")
    except ValidationError:
        check(True, "pipe rejects an sPTT field")

    # viscoelastic-dean: m0 and block_mean are on the surface now.
    from dnsjax.extensions import relevant_extensions

    ve = spec_for("viscoelastic-dean")
    ve_exts = tuple(relevant_extensions("viscoelastic-dean").values())
    ve_model = PS.build_surface_model(ve, settings=False, extensions=ve_exts)
    v = ve_model.model_validate(
        {
            "geo": {"m0": 2},
            "phys": {"block_mean_spanwise_velocity": True},
        }
    )
    core, _ = PS.split_extensions(v.model_dump(exclude_unset=True), ve_exts)
    core = PS.internalize(core, ve)
    check(
        core["geo"] == {"m0": 2}
        and core["phys"] == {"block_mean_spanwise_velocity": True},
        "viscoelastic m0/block_mean accepted",
        core,
    )
    # The [probes] extension is on the viscoelastic surface; [force]
    # is not (its relevance excludes viscoelastic systems) -- even
    # with the flow-relevant extensions attached.
    check(
        [e.name for e in ve_exts] == ["probes"],
        "viscoelastic relevant extensions",
        [e.name for e in ve_exts],
    )
    ve_model.model_validate({"probes": {"modes": "3,0", "it_probes": 5}})
    check(True, "viscoelastic accepts [probes]")
    try:
        ve_model.model_validate({"force": {"modes": "3,0"}})
        check(False, "viscoelastic rejects [force]")
    except ValidationError:
        check(True, "viscoelastic rejects [force]")


def case_derive() -> None:
    import math

    import dnsjax.parameters as P

    # quasi-keplerian: re2 derived; a direct assignment is overwritten.
    reset()
    P.params.phys.re2 = 123.0  # direct assignment, silently overwritten
    P.update_parameters(
        P.Parameters(
            phys={"system": "quasi-keplerian", "re1": 500.0, "r_omega": -1.2},
            geo={"eta": 0.71},
        )
    )
    re2 = P.params.phys.re2
    check(
        re2 is not None and abs(re2 - 398.4238178633975) < 1e-9,
        "quasi-keplerian re2 derived",
        re2,
    )
    check(P.params.phys.re == 500.0, "quasi-keplerian re = re1")
    P.validate_parameters()

    # viscoelastic-dean: reference defaults materialized, wedge lz,
    # re := wi/el.
    reset()
    P.update_parameters(
        P.Parameters(phys={"system": "viscoelastic-dean"}, geo={"m0": 2})
    )
    check(P.params.phys.el == 80.0, "viscoelastic el default")
    check(P.params.phys.wi == 105.0, "viscoelastic wi default")
    check(P.params.geo.delta == 11.0, "viscoelastic delta default")
    check(
        math.isclose(P.params.geo.lx, 2 * math.pi),
        "viscoelastic axial default 2*pi",
    )
    check(
        math.isclose(P.params.geo.lz, math.pi),
        "viscoelastic wedge lz = 2*pi/m0",
        P.params.geo.lz,
    )
    check(
        math.isclose(P.params.phys.re, 105.0 / 80.0),
        "viscoelastic re = wi/el",
    )
    check(
        P.derived_params.r_inner == 11.0 and P.derived_params.r_outer == 13.0,
        "viscoelastic radii",
    )
    P.validate_parameters()  # m0 = 2 is allowed for viscoelastic now


def case_externalize() -> None:
    import math

    import dnsjax.param_surface as PS
    import dnsjax.parameters as P
    from dnsjax.flows.registry import internalize_stored, spec_for

    # Rehydrate matches derive for the hidden-derived QK fields.
    reset()
    P.update_parameters(
        P.Parameters(
            phys={"system": "quasi-keplerian", "re1": 500.0, "r_omega": -1.2},
            geo={"eta": 0.71},
        )
    )
    P.validate_parameters()
    qk = spec_for("quasi-keplerian")
    stored = PS.externalize(P.params, qk)
    check(
        "re2" not in stored["phys"] and "re" not in stored["phys"],
        "QK hides derived re/re2",
        sorted(stored["phys"]),
    )
    back = internalize_stored(stored, "quasi-keplerian", rehydrate=True)
    check(
        math.isclose(back["phys"]["re2"], P.params.phys.re2),
        "QK rehydrate matches derive (re2)",
        (back["phys"].get("re2"), P.params.phys.re2),
    )
    check(
        math.isclose(back["phys"]["re"], P.params.phys.re),
        "QK rehydrate matches derive (re)",
    )


def case_extensions() -> None:
    from dnsjax_lab.extensions import force_params, probes_params
    from pydantic_settings import CliApp

    import dnsjax.param_surface as PS
    import dnsjax.parameters as P
    from dnsjax.extensions import (
        apply_extension_layer,
        relevant_extensions,
        reset_extensions,
    )
    from dnsjax.flows.registry import spec_for
    from dnsjax.param_surface import recorded_params_dump

    check("jax" not in sys.modules, "extensions stay JAX-free")

    # CLI parse: extension flags land in the overlays, not the core.
    pp = spec_for("plane-poiseuille")
    exts = tuple(relevant_extensions("plane-poiseuille").values())
    check(
        [e.name for e in exts] == ["probes", "force"],
        "plane-poiseuille relevant extensions",
        [e.name for e in exts],
    )
    cli_model = PS.build_surface_model(pp, settings=True, extensions=exts)
    args = [
        "--phys.re",
        "500",
        "--probes.modes",
        "3,0",
        "--probes.it_probes",
        "10",
    ]
    src = PS.make_cli_source(
        cli_model, system="plane-poiseuille", cli_args=args
    )
    parsed = CliApp.run(cli_model, cli_args=args, cli_settings_source=src)
    core, overlays = PS.split_extensions(
        parsed.model_dump(exclude_unset=True), exts
    )
    check(
        core.get("phys") == {"re": 500.0} and "probes" not in core,
        "extension flags split from core",
        core,
    )
    check(
        overlays == {"probes": {"modes": "3,0", "it_probes": 10}},
        "extension overlay contents",
        overlays,
    )

    # apply_extension_layer mutates the live singleton in place.
    try:
        apply_extension_layer(overlays)
        check(
            probes_params.modes == "3,0" and probes_params.it_probes == 10,
            "apply_extension_layer mutates singleton",
        )

        # The recorded params dump carries the extension sections and
        # the flow-relevant public-named core surface.
        reset()  # also resets the extension singletons
        apply_extension_layer(overlays)
        P.update_parameters(P.Parameters(phys={"system": "plane-poiseuille"}))
        dump = recorded_params_dump(P.params)
        check(
            dump.get("probes") == {"modes": "3,0", "it_probes": 10}
            and "force" in dump,
            "recorded_params_dump embeds extensions",
            {k: dump.get(k) for k in ("probes", "force")},
        )
        check(
            "el" not in dump["phys"]
            and dump["phys"]["system"] == "plane-poiseuille",
            "recorded_params_dump is the filtered public surface",
            sorted(dump["phys"]),
        )

        # Irrelevant system: the sections leave the metadata dump...
        reset()
        P.update_parameters(P.Parameters(phys={"system": "kolmogorov"}))
        dump = recorded_params_dump(P.params)
        check(
            "probes" not in dump and "force" not in dump,
            "recorded_params_dump filters irrelevant extensions",
        )
        # ... but a *configured* section on an unsupported system is
        # still rejected (direct assignment bypasses the strict
        # surface; the validate hooks dispatch unconditionally).
        force_params.modes = "3,0"
        force_params.profiles = "prof.npz"
        force_params.amplitude = 1e-3
        force_params.it_force = 2
        try:
            P.validate_parameters()
            check(False, "force on periodic rejected")
        except ValueError as ex:
            check(
                "wall-bounded" in str(ex),
                "force on periodic rejected",
                ex,
            )
    finally:
        reset_extensions()
    check(
        probes_params.modes is None and force_params.modes is None,
        "reset_extensions restores defaults",
    )


def case_entry_smoke() -> None:
    import tempfile

    tmp = tempfile.mkdtemp(prefix="param_surface_lab_")

    def run(*args: str, module: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp,
            env={**os.environ, "NO_COLOR": "1"},
        )

    # An entry point's own section is on its bare --help, and its flow
    # help carries only the built-in sections that flow accepts.  In
    # subprocesses: importing either driver registers its section.
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dnsjax_lab.twin.driver import TwinParams as m; "
            "print(' '.join(m.model_fields))",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    twin_flags = {f"--twin.{name}" for name in r.stdout.split()}
    r = run("--help", module="dnsjax_lab.twin")
    missing = sorted(f for f in twin_flags if f not in r.stdout)
    check(
        r.returncode == 0 and twin_flags and not missing,
        "dnsjax-twin --help lists every [twin] field",
        missing or r.stderr[-300:],
    )
    tg = "dnsjax.analysis.transient_growth"
    for args, shown, hidden in (
        ((), ("--tg.",), ("--probes.", "--force.")),
        (("kolmogorov",), ("--tg.",), ("--probes.", "--force.")),
        (("viscoelastic-pipe",), ("--tg.", "--probes."), ("--force.",)),
    ):
        r = run("--help", *args, module=tg)
        check(
            r.returncode == 0
            and all(s in r.stdout for s in shown)
            and not any(h in r.stdout for h in hidden),
            f"transient-growth --help {' '.join(args)}: "
            f"shows {shown}, hides {hidden}",
            r.stderr[-300:],
        )


def main() -> int:
    case_total_field()
    case_surfaces()
    case_derive()
    case_externalize()
    case_extensions()
    case_entry_smoke()
    return finish("lab parameter-surface")


if __name__ == "__main__":
    sys.exit(main())
