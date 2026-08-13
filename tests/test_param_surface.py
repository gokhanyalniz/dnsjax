r"""Flow-spec registry and per-flow parameter-surface tests.

Offline and JAX-free end to end: the registry (``dnsjax.flows.
registry``), the spec hooks wired into ``update_parameters`` /
``validate_parameters`` (per-flow defaults materialization, derive,
grid defaults, strict relevance for direct assignment), and the
surface machinery (``dnsjax.param_surface``): dynamic CLI/TOML models,
alias round-trips, strict rejection of irrelevant parameters,
deferred-feature messages, ``externalize``/``internalize_stored``
mapping, and the annotated sample-TOML rendering.  The entry-point
smoke cases additionally shell out ``python -m dnsjax --help`` /
``--help <system>`` / ``--sample-toml`` (no ``mpirun``: help exits at
the parser).

Cases mutate the ``params`` singleton and restore it via ``_reset``
(fresh section models + cleared explicit-set tracking + reset
extension singletons), so ordering is free.  The extension cases
cover the ``probes``/``force`` sections end to end: surface split,
layering, validation dispatch, and the metadata dump.

Run as a script::

    uv run python tests/test_param_surface.py
    uv run python tests/test_param_surface.py --skip-entry-smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib

sys.stdout.reconfigure(line_buffering=True)

FAILURES: list[str] = []


def check(cond: bool, label: str, detail: object = "") -> None:
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if not cond else ""))
    if not cond:
        FAILURES.append(label)


def _reset() -> None:
    import dnsjax.parameters as P
    from dnsjax.extensions import reset_extensions

    fresh = P.Parameters()
    for section in P.Parameters.model_fields:
        setattr(P.params, section, getattr(fresh, section))
    P._user_set_fields.clear()
    P._materialized_defaults.clear()
    reset_extensions()


# ── Case A: registry / internal-model coherence (JAX-free) ───────────


def case_coherence() -> None:
    from dnsjax.flows import registry as R
    from dnsjax.parameters import Parameters

    check("jax" not in sys.modules, "registry+surface stay JAX-free")

    models = {
        section: field.annotation
        for section, field in Parameters.model_fields.items()
    }
    for section, name in R.GLOBAL_FIELDS:
        check(
            name in models[section].model_fields,
            f"global field exists: {section}.{name}",
        )
    for system, spec in R.SPECS.items():
        for fs in spec.fields:
            check(
                fs.name in models[fs.section].model_fields,
                f"{system}: field exists: {fs.section}.{fs.name}",
            )
        for d in spec.deferred:
            check(
                d.name in models[d.section].model_fields,
                f"{system}: deferred field exists: {d.section}.{d.name}",
            )
        # No public-name collisions within a section (aliases must not
        # shadow another surface field).
        publics: set[tuple[str, str]] = set()
        for fs in spec.fields:
            key = (fs.section, fs.public_name)
            check(
                key not in publics,
                f"{system}: unique public name {key}",
            )
            publics.add(key)
        # Global fields are never re-declared by a spec.
        overlap = set(spec.field_map) & set(R.GLOBAL_FIELDS)
        check(not overlap, f"{system}: no global overlap", overlap)


# ── Case B: surface strictness + alias round-trip ────────────────────


def case_surface_strictness() -> None:
    from pydantic import ValidationError

    import dnsjax.param_surface as PS
    from dnsjax.flows.registry import spec_for

    pipe = spec_for("pipe")
    toml_model = PS.build_surface_model(pipe, settings=False)

    # Aliased names parse; internal names for aliased fields do not.
    v = toml_model.model_validate(
        {"res": {"nz": 512, "nr": 48, "ntheta": 96}, "geo": {"lz": 200.0}}
    )
    core = PS.internalize(v.model_dump(exclude_unset=True), pipe)
    check(
        core == {"geo": {"lx": 200.0}, "res": {"nx": 512, "ny": 48, "nz": 96}},
        "pipe alias internalize",
        core,
    )
    for bad in (
        {"phys": {"el": 80.0}},  # viscoelastic-only field
        {"geo": {"lx": 4.0}},  # internal name of an aliased field
        {"res": {"ny": 48}},  # internal name of an aliased field
        {"phys": {"block_mean_spanwise_velocity": True}},  # not pipe
        {"nonsense": {"x": 1}},  # unknown section
    ):
        try:
            toml_model.model_validate(bad)
            check(False, f"pipe rejects {bad}")
        except ValidationError:
            check(True, f"pipe rejects {bad}")

    # plane-couette: no driving on the surface (amendment: cbv banned).
    couette = spec_for("plane-couette")
    couette_model = PS.build_surface_model(couette, settings=False)
    try:
        couette_model.model_validate({"phys": {"driving": "x"}})
        check(False, "couette rejects driving")
    except ValidationError:
        check(True, "couette rejects driving")

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

    # Periodic: wall-bounded fields are rejected outright; the
    # random-IC mean-mode knob stays (the periodic generator honours
    # it).
    kol = spec_for("kolmogorov")
    kol_model = PS.build_surface_model(kol, settings=False)
    for bad in (
        {"geo": {"grid_type": "cgl"}},
        {"solver": {"backend": "dense"}},
    ):
        try:
            kol_model.model_validate(bad)
            check(False, f"kolmogorov rejects {bad}")
        except ValidationError:
            check(True, f"kolmogorov rejects {bad}")
    kol_model.model_validate({"init": {"random_mean_flow": True}})
    check(True, "kolmogorov accepts init.random_mean_flow")


def case_cli_parse() -> None:
    from pydantic_settings import CliApp

    import dnsjax.param_surface as PS
    from dnsjax.flows.registry import spec_for

    pipe = spec_for("pipe")
    cli_model = PS.build_surface_model(pipe, settings=True)

    args = [
        "--phys.re",
        "2300",
        "--res.nz",
        "512",
        "--res.nr",
        "48",
        "--res.ntheta",
        "96",
        "--geo.lz",
        "200.0",
    ]
    src = PS.make_cli_source(cli_model, system="pipe", cli_args=args)
    parsed = CliApp.run(cli_model, cli_args=args, cli_settings_source=src)
    dump = parsed.model_dump(exclude_unset=True)
    core = PS.internalize(dump, pipe)
    check(
        core.get("phys") == {"re": 2300.0}
        and core.get("res") == {"nx": 512, "ny": 48, "nz": 96}
        and core.get("geo") == {"lx": 200.0},
        "pipe CLI parse + internalize",
        core,
    )

    # An irrelevant flag exits with argparse code 2 and the flow-aware
    # message (the source parses eagerly at construction).
    bad = ["--phys.el", "80"]
    try:
        src = PS.make_cli_source(cli_model, system="pipe", cli_args=bad)
        CliApp.run(cli_model, cli_args=bad, cli_settings_source=src)
        check(False, "pipe CLI rejects --phys.el")
    except SystemExit as ex:
        check(ex.code == 2, "pipe CLI rejects --phys.el", ex.code)


# ── Case C: deferred-feature messages ────────────────────────────────


def case_deferred() -> None:
    import dnsjax.param_surface as PS
    from dnsjax.flows.registry import spec_for

    pipe = spec_for("pipe")
    model = PS.build_surface_model(pipe, settings=False)
    v = model.model_validate({"geo": {"tilt_degree": 10.0}})
    try:
        PS.internalize(v.model_dump(exclude_unset=True), pipe)
        check(False, "pipe tilt deferred")
    except ValueError as ex:
        check("not implemented yet" in str(ex), "pipe tilt deferred", ex)

    kol = spec_for("kolmogorov")
    model = PS.build_surface_model(kol, settings=False)
    for field, value in (
        ("u_grid", 0.5),
        ("u_grid", 0.0),  # even zero: the field is deferred wholesale
    ):
        v = model.model_validate({"phys": {field: value}})
        try:
            PS.internalize(v.model_dump(exclude_unset=True), kol)
            check(False, f"kolmogorov {field}={value} deferred")
        except ValueError as ex:
            check(
                "not implemented yet" in str(ex),
                f"kolmogorov {field}={value} deferred",
                ex,
            )


# ── Case D: update_parameters spec dispatch ──────────────────────────


def case_derive_and_defaults() -> None:
    import dnsjax.parameters as P

    # pipe: u_grid materialized, scheme-dependent grid default.
    _reset()
    P.update_parameters(P.Parameters(phys={"system": "pipe"}))
    check(P.params.phys.u_grid == 0.5, "pipe u_grid materialized")
    check(P.derived_params.u_grid == 0.5, "pipe derived u_grid")
    check(
        P.params.geo.grid_type == "half-cgl",
        "pipe iterative-cn grid default",
        P.params.geo.grid_type,
    )
    P.update_parameters(P.Parameters(step={"scheme": "cnab2"}))
    check(
        P.params.geo.grid_type == "rigged-cgl",
        "pipe cnab2 grid default",
        P.params.geo.grid_type,
    )
    import math

    check(
        math.isclose(P.params.geo.lz, 2 * math.pi),
        "pipe lz = 2*pi (m0=1)",
    )
    P.validate_parameters()

    # A later system layer re-materializes stale per-flow defaults.
    _reset()
    P.update_parameters(P.Parameters(phys={"system": "pipe"}))
    P.update_parameters(P.Parameters(phys={"system": "plane-couette"}))
    check(
        P.params.phys.u_grid == 0.0,
        "couette u_grid re-materialized",
        P.params.phys.u_grid,
    )
    check(
        P.params.geo.grid_type == "cgl",
        "couette grid default",
        P.params.geo.grid_type,
    )
    P.validate_parameters()

    # quasi-keplerian: re2 derived; a direct assignment is overwritten.
    _reset()
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
    _reset()
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

    # taylor-couette still validates its control parameters.
    _reset()
    try:
        P.update_parameters(P.Parameters(phys={"system": "taylor-couette"}))
        check(False, "taylor-couette requires eta")
    except ValueError as ex:
        check("geo.eta" in str(ex), "taylor-couette requires eta", ex)


def case_validate_guards() -> None:
    import dnsjax.parameters as P

    # m0 on a Cartesian flow: rejected (direct assignment path).
    _reset()
    P.update_parameters(P.Parameters(phys={"system": "plane-couette"}))
    P.params.geo.m0 = 3
    try:
        P.validate_parameters()
        check(False, "m0 rejected for cartesian")
    except ValueError as ex:
        check("wedge" in str(ex), "m0 rejected for cartesian", ex)

    # constant_bulk_velocity is banned for plane-couette.
    _reset()
    P.update_parameters(P.Parameters(phys={"system": "plane-couette"}))
    P.params.phys.driving = "constant_bulk_velocity"
    try:
        P.validate_parameters()
        check(False, "couette cbv rejected")
    except ValueError as ex:
        check("plane-couette" in str(ex), "couette cbv rejected", ex)

    # Cross-family grid names are rejected at validation.
    _reset()
    P.update_parameters(
        P.Parameters(phys={"system": "pipe"}, geo={"grid_type": "half-cgl"})
    )
    P.params.geo.grid_type = "cgl"  # direct assignment of a wrong name
    try:
        P.validate_parameters()
        check(False, "pipe rejects plain cgl")
    except ValueError as ex:
        check("grid_type" in str(ex), "pipe rejects plain cgl", ex)

    # half-cgl requires iterative-cn.
    _reset()
    try:
        P.update_parameters(
            P.Parameters(
                phys={"system": "pipe"},
                geo={"grid_type": "half-cgl"},
                step={"scheme": "cnab2"},
            )
        )
        P.validate_parameters()
        check(False, "half-cgl needs iterative-cn")
    except ValueError as ex:
        check("half-cgl" in str(ex), "half-cgl needs iterative-cn", ex)

    # u_grid on a periodic system: deferred error (direct assignment).
    _reset()
    P.update_parameters(P.Parameters(phys={"system": "kolmogorov"}))
    P.params.phys.u_grid = 0.5
    try:
        P.validate_parameters()
        check(False, "periodic u_grid deferred")
    except ValueError as ex:
        check(
            "not implemented yet" in str(ex),
            "periodic u_grid deferred",
            ex,
        )


# ── Case E: externalize / internalize_stored ─────────────────────────


def case_externalize() -> None:
    import math

    import dnsjax.param_surface as PS
    import dnsjax.parameters as P
    from dnsjax.flows.registry import internalize_stored, spec_for

    _reset()
    P.update_parameters(
        P.Parameters(
            phys={"system": "pipe", "re": 2300.0},
            res={"nx": 64, "ny": 24, "nz": 32},
        )
    )
    P.validate_parameters()
    pipe = spec_for("pipe")
    out = PS.externalize(P.params, pipe)
    check(
        set(out["res"])
        == {
            "nz",
            "nr",
            "ntheta",
            "fd_order",
            "consistent_imm",
            "double_precision",
        },
        "externalize res keys public",
        sorted(out["res"]),
    )
    check(
        out["res"]["nz"] == 64
        and out["res"]["nr"] == 24
        and out["res"]["ntheta"] == 32,
        "externalize alias values",
        out["res"],
    )
    check("el" not in out["phys"], "externalize filters el")
    check(
        "lz" in out["geo"] and "lx" not in out["geo"], "externalize geo public"
    )
    check(out["phys"]["u_grid"] == 0.5, "externalize materialized u_grid")
    check("tilt_degree" not in out["geo"], "externalize skips deferred")

    back = internalize_stored(out, "pipe", rehydrate=True)
    check(
        back["res"]
        == {
            "nx": 64,
            "ny": 24,
            "nz": 32,
            "fd_order": P.params.res.fd_order,
            "consistent_imm": P.params.res.consistent_imm,
            "double_precision": True,
        },
        "internalize_stored res",
        back["res"],
    )
    check(
        math.isclose(back["geo"]["lz"], 2 * math.pi),
        "rehydrate fills internal lz",
        back["geo"].get("lz"),
    )

    # Rehydrate matches derive for the hidden-derived QK fields.
    _reset()
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


# ── Case E2: extension sections (probes / force) ─────────────────────


def case_extensions() -> None:
    from pydantic_settings import CliApp

    import dnsjax.param_surface as PS
    import dnsjax.parameters as P
    from dnsjax.extensions import (
        apply_extension_layer,
        force_params,
        probes_params,
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
        _reset()  # also resets the extension singletons
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
        _reset()
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


# ── Case F: sample-TOML round trip ───────────────────────────────────


def case_sample_toml() -> None:
    import dnsjax.param_surface as PS
    from dnsjax.extensions import relevant_extensions
    from dnsjax.flows.registry import SPECS

    for system, spec in SPECS.items():
        exts = tuple(relevant_extensions(system).values())
        text = PS.render_sample_toml(spec, exts)
        data = tomllib.loads(text)
        # Section headers parse as empty tables; the only active value
        # must be phys.system (every default is commented out --
        # extension sections included).
        active = {s: v for s, v in data.items() if v}
        check(
            active == {"phys": {"system": system}},
            f"sample toml {system}: only system active",
            active,
        )
        check(
            all(f"[{e.name}]" in text for e in exts),
            f"sample toml {system}: extension sections rendered",
        )
        model = PS.build_surface_model(spec, settings=False, extensions=exts)
        model.model_validate(data)
        check(True, f"sample toml {system} validates")


# ── Case G: entry-point smoke (help / sample-toml / strict errors) ───


def case_entry_smoke() -> None:
    import tempfile

    # Hermetic cwd: the entry point reads ./parameters.toml, so a repo
    # root config must not shape these checks.
    tmp = tempfile.mkdtemp(prefix="param_surface_smoke_")

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "dnsjax", *args],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp,
            # Deterministic plain help: a FORCE_COLOR in the host env
            # (agent harnesses set one) breaks the substring checks.
            env={**os.environ, "NO_COLOR": "1"},
        )

    r = run("--help")
    check(r.returncode == 0, "--help exits 0", r.stderr[-300:])
    check(
        "flows (geometry: systems):" in r.stdout
        and "plane-couette" in r.stdout,
        "--help lists flows",
    )
    check(
        "--phys.el" not in r.stdout and "--geo.eta" not in r.stdout,
        "--help hides flow-specific fields",
    )
    check(
        "Flow system to integrate" in r.stdout,
        "--help shows descriptions",
    )

    r = run("--help", "pipe")
    check(r.returncode == 0, "--help pipe exits 0", r.stderr[-300:])
    check(
        "--res.nr" in r.stdout and "--res.ntheta" in r.stdout,
        "--help pipe shows aliases",
    )
    check("--phys.el" not in r.stdout, "--help pipe hides el")
    check("rigged-cgl" in r.stdout, "--help pipe shows grid choices")

    r = run("--phys.system", "taylor-couette", "--help")
    check(
        r.returncode == 0 and "--phys.re1" in r.stdout,
        "--phys.system X --help works",
    )

    r = run("--sample-toml", "pipe")
    check(r.returncode == 0, "--sample-toml exits 0", r.stderr[-300:])
    try:
        data = tomllib.loads(r.stdout)
        check(
            data.get("phys") == {"system": "pipe"},
            "--sample-toml output parses clean",
            data,
        )
    except tomllib.TOMLDecodeError as ex:
        check(False, "--sample-toml output parses clean", ex)

    r = run("--phys.system", "pipe", "--phys.el", "80")
    check(r.returncode == 2, "irrelevant CLI flag exits 2", r.returncode)
    check(
        "pipe" in r.stderr and "--phys.el" in r.stderr,
        "irrelevant CLI flag names flow",
        r.stderr[-300:],
    )


# ── runner ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-entry-smoke",
        action="store_true",
        help="skip the python -m dnsjax subprocess cases",
    )
    args = parser.parse_args()

    case_coherence()
    case_surface_strictness()
    case_cli_parse()
    case_deferred()
    case_derive_and_defaults()
    case_validate_guards()
    case_externalize()
    case_extensions()
    case_sample_toml()
    if not args.skip_entry_smoke:
        case_entry_smoke()

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll parameter-surface tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
