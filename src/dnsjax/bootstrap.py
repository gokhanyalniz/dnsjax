r"""Shared entry-point setup: parameter layering and JAX runtime init.

Every dnsjax entry point -- the ``dnsjax`` / ``dnsjax-twin`` console
scripts (``python -m dnsjax`` / ``-m dnsjax.twin``), the analysis
CLIs, the diagnostic scripts, and the offline tests -- must finalize
parameters and
configure JAX in the same order **before** importing
:mod:`dnsjax.sharding` or any geometry module, because those modules
capture ``params`` / ``padded_res`` / the devices in module-level
singletons at import time:

1. Finalize the global ``params``: :func:`resolve_parameters` (the
   production CLI / TOML / snapshot layering over the per-flow
   parameter surface), or a script's own ``update_parameters`` calls
   followed by ``validate_parameters()`` and
   ``padded_res.set_padded_resolution(params)``.
2. Configure the JAX runtime: :func:`configure_jax_runtime` (the
   production multi-process path), or the single-process
   :func:`configure_jax_platform` (typically fed by
   :func:`platform_from_argv`).
3. Only then import :mod:`dnsjax.sharding` and the geometry / flow
   modules.

Production parsing is *surface-based* and two-pass:
:func:`peek_run_context` scans the raw argv / ``parameters.toml`` (no
pydantic) for the flow system (CLI ``--phys.system`` > TOML > resumed
snapshot > model default), ``--help`` (with an optional flow
positional: ``dnsjax --help pipe``), and ``--sample-toml``; then the
per-flow surface model (:mod:`dnsjax.param_surface`) parses the CLI
and validates the TOML strictly -- a parameter that does not apply to
the selected flow is an error, aliased fields go by their public
names, and per-flow defaults are materialized.  The flow-relevant
extension sections (:mod:`dnsjax.extensions`; e.g. ``[probes]``,
``[force]``) ride the same surface: each layer's extension overlay is
applied to the extension singletons right after its core
``update_parameters`` call, snapshot layer included.

This module is JAX-free at import time (JAX is imported inside the
configuration functions), so it is always safe to import first.
"""

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError
from pydantic_settings import CliApp

from .extensions import apply_extension_layer, relevant_extensions
from .flow_spec import FlowSpec
from .flows.registry import all_systems, spec_for
from .param_surface import (
    build_surface_model,
    flow_list_epilog,
    internalize,
    make_cli_source,
    render_sample_toml,
    split_extensions,
)
from .parameters import (
    Parameters,
    Physics,
    padded_res,
    params,
    read_snapshot_params,
    update_parameters,
    validate_parameters,
)

_VALID_PLATFORMS: tuple[str, ...] = ("cpu", "cuda", "rocm", "tpu")


def platform_from_argv(
    argv: list[str] | None = None, default: str = "cpu"
) -> str:
    """Extract ``--dist.platform`` from *argv* (default ``sys.argv``).

    A single-process script or offline test must know its JAX backend
    *before* importing :mod:`dnsjax.sharding` or any geometry module
    (which capture the platform at import), i.e. before its own
    ``argparse`` runs.  This does a minimal early parse of the same
    ``--dist.platform`` flag the production CLI accepts -- both
    ``--dist.platform cuda`` and ``--dist.platform=cuda`` -- and returns
    *default* when it is absent (unknown flags are ignored).  Pair it
    with :func:`configure_jax_platform`.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dist.platform", dest="platform", default=default)
    known, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    return known.platform


def configure_jax_platform(
    platform: str, *, double_precision: bool = True
) -> None:
    """Select the JAX backend for a single-process script or offline test.

    Records *platform* on :data:`params` (``params.dist.platform``) and
    configures JAX for the platform/precision -- the single-process half
    of :func:`configure_jax_runtime`, without the multi-process
    ``jax.distributed.initialize`` and the CPU-threading ``XLA_FLAGS``
    that only the production entry point needs.  The thread pool stays
    unpinned on purpose -- a single-process test or script *should*
    use the whole box; do not add the production pinning here.  Must
    be called *before* importing :mod:`dnsjax.sharding` or any
    geometry module.

    After this, :data:`sharding` reports the active device unambiguously
    (its banner reads the live device, so a stale ``params.dist.platform``
    can no longer contradict it), and ``--dist.platform cuda`` (via
    :func:`platform_from_argv`) runs the real Pallas / Triton kernels on a
    GPU from any script or test, not just the production entry point.

    Also line-buffers ``sys.stdout``, so piped output (agent-tailed
    runs, SLURM logs) arrives per line instead of in 8 KiB blocks.
    """
    # Guarded: wrapped/captured stdout objects may lack
    # ``reconfigure``; the switch flushes anything already buffered.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if platform not in _VALID_PLATFORMS:
        raise ValueError(
            f"unknown platform {platform!r}; expected one of "
            f"{_VALID_PLATFORMS}"
        )
    params.dist.platform = platform

    import jax

    jax.config.update("jax_enable_x64", double_precision)
    jax.config.update("jax_platforms", platform)


# ── Production surface parsing ───────────────────────────────────


@dataclass(frozen=True)
class RunContext:
    """Pre-parse peek at an invocation (raw argv / TOML scan).

    ``system`` is the flow every surface is built for, resolved with
    priority CLI ``--phys.system`` > ``parameters.toml`` > the resumed
    snapshot's stored system > the model default.  ``help_system`` is
    the flow named for ``--help`` -- a bare positional
    (``dnsjax --help pipe``) or an explicit CLI ``--phys.system``;
    ``None`` means global help (bare ``dnsjax --help`` shows the
    global parameters and the flow list even when a ``parameters.toml``
    selects a system).  ``sample_toml`` is the flow named by
    ``--sample-toml`` (``None`` = not requested).
    """

    system: str
    help_requested: bool
    help_system: str | None
    sample_toml: str | None
    snapshot_path: Path | None
    raw_toml: dict | None


def _scan_flag(argv: list[str], name: str) -> str | None:
    """Last value of ``--name v`` / ``--name=v`` in *argv* (raw scan)."""
    value = None
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            value = argv[i + 1]
        elif tok.startswith(name + "="):
            value = tok.split("=", 1)[1]
    return value


def _check_system(name: str | None, origin: str) -> None:
    if name is not None and name not in all_systems():
        raise SystemExit(
            f"dnsjax: error: unknown flow system {name!r} ({origin}); "
            f"available: {', '.join(all_systems())}"
        )


def peek_run_context(
    argv: list[str], toml_path: Path | Literal[False] | None = None
) -> RunContext:
    """Scan *argv* and the parameters TOML for the run context.

    A raw string/`tomllib` scan (no pydantic): the flow system that
    selects the parameter surface, the ``--help`` /``--sample-toml``
    requests and their optional flow positional, and the snapshot to
    resume (needed both for the system peek and the layering).
    *toml_path* overrides the default ``./parameters.toml`` (an entry
    point with its own TOML-path flag passes it); an explicit path
    that does not exist is an error, whereas the default is optional.
    ``False`` skips the TOML layer entirely -- for entry points whose
    configuration is snapshot + CLI only (e.g. the offline
    ``snapshot_perturb`` injector), which must not pick up an
    unrelated ``parameters.toml`` from the working directory.
    """
    raw_toml = None
    if toml_path is False:
        params_file = None
    else:
        if toml_path is not None and not toml_path.is_file():
            raise SystemExit(f"parameters file not found: {toml_path}")
        params_file = (
            toml_path if toml_path is not None else Path("parameters.toml")
        )
    if params_file is not None and params_file.is_file():
        with open(params_file, "rb") as fh:
            raw_toml = tomllib.load(fh)

    systems = set(all_systems())
    help_requested = any(tok in ("-h", "--help") for tok in argv)

    cli_system = _scan_flag(argv, "--phys.system")
    _check_system(cli_system, "--phys.system")
    toml_system = ((raw_toml or {}).get("phys") or {}).get("system")
    _check_system(toml_system, "parameters.toml [phys] system")

    # A bare token naming a system is the --help / --sample-toml flow
    # selector (``dnsjax --help pipe``).  A system name in a flag-value
    # position (the previous token is a ``--flag`` that takes a value)
    # is not a positional.
    positional = None
    for i, tok in enumerate(argv):
        if tok not in systems:
            continue
        prev = argv[i - 1] if i > 0 else ""
        if prev.startswith("--") and prev not in (
            "--help",
            "--sample-toml",
        ):
            continue
        positional = tok
    _check_system(positional, "positional")

    sample_toml = None
    if "--sample-toml" in argv:
        i = argv.index("--sample-toml")
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        sample_toml = nxt if nxt in systems else (positional or "")

    snap = _scan_flag(argv, "--init.snapshot")
    if snap is None and raw_toml is not None:
        snap = (raw_toml.get("init") or {}).get("snapshot")
    snapshot_path = Path(snap) if snap else None

    snap_system = None
    if (
        cli_system is None
        and toml_system is None
        and (snapshot_path is not None)
    ):
        snap_system = _snapshot_system(snapshot_path)

    default_system = Physics.model_fields["system"].default
    system = cli_system or toml_system or snap_system or default_system
    if sample_toml == "":
        sample_toml = system

    return RunContext(
        system=system,
        help_requested=help_requested,
        help_system=positional or cli_system if help_requested else None,
        sample_toml=sample_toml,
        snapshot_path=snapshot_path,
        raw_toml=raw_toml,
    )


def _snapshot_system(path: Path) -> str | None:
    """The stored system of a dnsjax snapshot (``None`` otherwise)."""
    from .snapshot_meta import is_snapshot_file, read_snapshot_meta

    if not is_snapshot_file(path):
        return None
    return read_snapshot_meta(path).get("system")


def _format_toml_errors(exc: ValidationError, system: str) -> str:
    """Human-readable strict-TOML failure (flow-aware for extras)."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        if err["type"] == "extra_forbidden":
            lines.append(
                f"parameters.toml: '{loc}' is not a parameter of flow "
                f"{system!r}; see `dnsjax --help {system}`"
            )
        else:
            lines.append(f"parameters.toml: {loc}: {err['msg']}")
    return "dnsjax: error:\n  " + "\n  ".join(lines)


def _flow_epilog(spec: FlowSpec, prog: str = "dnsjax") -> str:
    return (
        f"{spec.system}: {spec.summary}.\n"
        f"All flows also accept the global parameters: `{prog} --help`."
    )


@dataclass(frozen=True)
class ResolvedSetup:
    """What the production startup banner needs from the layering.

    ``system``/``spec`` identify the resolved flow (and its parameter
    surface, e.g. for the startup printout); ``params_from_disk``
    whether a ``parameters.toml`` was loaded; ``snapshot_path`` /
    ``snapshot_params_used`` which snapshot (if any) contributed its
    embedded parameters as the lowest layer.
    """

    system: str
    spec: FlowSpec
    params_from_disk: bool
    snapshot_path: Path | None
    snapshot_params_used: bool


def resolve_parameters(
    cli_args: list[str] | None = None,
    *,
    toml_path: Path | Literal[False] | None = None,
    extensions: tuple | None = None,
    prog: str = "dnsjax",
) -> ResolvedSetup:
    r"""Resolve the production parameter layers into the global ``params``.

    Layering, lowest priority first: code defaults (incl. the flow's
    materialized default overrides) -> the embedded parameters of the
    snapshot being resumed (whichever ``init.snapshot`` was set
    explicitly, CLI over TOML) -> a ``parameters.toml`` in the current
    directory -> the command line (*cli_args*; ``None`` parses
    ``sys.argv``).

    All user-facing input goes through the *per-flow surface*
    (:mod:`dnsjax.param_surface`): only the selected flow's parameters
    parse, under their public names (``--help`` / ``--help <system>``
    documents them; ``--sample-toml <system>`` prints an annotated
    template), and an irrelevant or deferred parameter is a hard error
    on the CLI and in the TOML alike.  A resumed snapshot storing a
    different system than the run selects is rejected outright.

    The keyword parameters let other entry points reuse the exact
    production flow: *toml_path* overrides the default
    ``./parameters.toml`` lookup (``False`` skips the TOML layer
    entirely -- see :func:`peek_run_context`); *extensions* pins the
    extension
    sections riding the surface (``None`` -- the production default --
    selects every registered extension relevant to the resolved flow,
    :func:`dnsjax.extensions.relevant_extensions`; an entry point with
    its own section, e.g. the transient-growth ``[tg]``, passes an
    explicit tuple); *prog* names the program in help/usage/errors.

    Ends with ``validate_parameters()`` and
    ``padded_res.set_padded_resolution(params)``, so ``params`` and
    ``padded_res`` are final on return.  Must run *before*
    :func:`configure_jax_runtime` (which reads the resolved platform /
    precision) and before importing :mod:`dnsjax.sharding`.
    """
    argv = list(sys.argv[1:] if cli_args is None else cli_args)
    ctx = peek_run_context(argv, toml_path=toml_path)

    def _exts(system: str) -> tuple:
        if extensions is not None:
            return extensions
        return tuple(relevant_extensions(system).values())

    if ctx.sample_toml is not None:
        print(
            render_sample_toml(
                spec_for(ctx.sample_toml), _exts(ctx.sample_toml)
            ),
            end="",
        )
        raise SystemExit(0)

    if ctx.help_requested:
        spec = spec_for(ctx.help_system) if ctx.help_system else None
        # Global help (no flow named) shows no *flow-relevant* built-in
        # sections, but an entry point's own explicit sections (e.g.
        # the transient-growth [tg]) are system-independent and stay.
        help_exts = (
            _exts(spec.system)
            if spec is not None
            else (extensions if extensions is not None else ())
        )
        model = build_surface_model(spec, settings=True, extensions=help_exts)
        epilog = (
            flow_list_epilog(prog)
            if spec is None
            else _flow_epilog(spec, prog)
        )
        # The source parses eagerly: building it with ["--help"]
        # prints the help text and exits 0.
        make_cli_source(
            model,
            system=ctx.help_system,
            epilog=epilog,
            cli_args=["--help"],
            prog=prog,
        )
        raise SystemExit(0)  # not reached; the parser exits

    spec = spec_for(ctx.system)
    exts = _exts(ctx.system)
    surface_cls = build_surface_model(spec, settings=True, extensions=exts)
    toml_cls = build_surface_model(spec, settings=False, extensions=exts)

    # CLI layer -- parsed first (fail fast, exits on irrelevant
    # flags), applied last.
    src = make_cli_source(
        surface_cls,
        system=ctx.system,
        epilog=_flow_epilog(spec, prog),
        cli_args=argv,
        prog=prog,
    )
    parsed_cli = CliApp.run(
        surface_cls, cli_args=argv, cli_settings_source=src
    )
    cli_core, cli_ext = split_extensions(
        parsed_cli.model_dump(exclude_unset=True), exts
    )
    try:
        cli_layer = internalize(cli_core, spec)
    except ValueError as exc:  # deferred feature
        raise SystemExit(f"dnsjax: error: {exc}") from None

    # TOML layer -- validated strictly against the flow's surface.
    toml_layer: dict = {}
    toml_ext: dict[str, dict] = {}
    if ctx.raw_toml is not None:
        try:
            toml_obj = toml_cls.model_validate(ctx.raw_toml)
            toml_core, toml_ext = split_extensions(
                toml_obj.model_dump(exclude_unset=True), exts
            )
            toml_layer = internalize(toml_core, spec)
        except ValidationError as exc:
            raise SystemExit(_format_toml_errors(exc, ctx.system)) from None
        except ValueError as exc:  # deferred feature
            raise SystemExit(
                f"dnsjax: error: parameters.toml: {exc}"
            ) from None

    # Snapshot layer (lowest above defaults): whichever
    # ``init.snapshot`` was set explicitly, CLI over TOML.
    snap_value = (cli_layer.get("init") or {}).get("snapshot") or (
        toml_layer.get("init") or {}
    ).get("snapshot")
    snapshot_path = Path(snap_value) if snap_value is not None else None
    snapshot_params_used = False
    if snapshot_path is not None:
        # An unreadable snapshot (below ``MIN_FORMAT_VERSION``, or
        # malformed metadata) raises ``ValueError`` from deep inside
        # ``snapshot_meta``.  Present it the way every other
        # resume-time failure here is presented -- one ``dnsjax:
        # error:`` line -- rather than as a raw traceback; the message
        # already names the version and the reason.
        try:
            stored_system = _snapshot_system(snapshot_path)
        except ValueError as exc:
            raise SystemExit(f"dnsjax: error: {exc}") from None
        if stored_system is not None and stored_system != ctx.system:
            raise SystemExit(
                f"dnsjax: error: snapshot '{snapshot_path}' stores "
                f"system {stored_system!r} but the run selects "
                f"{ctx.system!r} (snapshots do not convert across "
                "flows)."
            )
        snap = read_snapshot_params(snapshot_path)
        if snap is not None:
            snap_params, snap_ext = snap
            update_parameters(snap_params)
            apply_extension_layer(snap_ext)
            snapshot_params_used = True

    # Higher-priority layers: parameters.toml, then CLI arguments
    # (core sections and extension sections alike).  Empty core layers
    # still run (the spec derive / default materialization must fire
    # at least once).
    update_parameters(Parameters.model_validate(toml_layer))
    apply_extension_layer(toml_ext)
    update_parameters(Parameters.model_validate(cli_layer))
    apply_extension_layer(cli_ext)

    validate_parameters()
    padded_res.set_padded_resolution(params)
    return ResolvedSetup(
        system=ctx.system,
        spec=spec,
        params_from_disk=ctx.raw_toml is not None,
        snapshot_path=snapshot_path,
        snapshot_params_used=snapshot_params_used,
    )


def configure_jax_runtime(distributed: bool = True) -> bool:
    r"""Configure the JAX runtime from the resolved global ``params``.

    The production path (``distributed=True``, used by the ``dnsjax``
    console script / ``python -m dnsjax``): pin the CPU threading via
    ``NPROC`` / ``XLA_FLAGS`` before JAX is imported (CPU platform
    only; one thread per rank, see the ``Distribution`` docstring),
    select platform and precision from ``params.dist.platform`` /
    ``params.res.double_precision`` (:func:`configure_jax_platform`),
    and initialize the multi-process runtime
    (``jax.distributed.initialize()``, which auto-detects the
    launcher).  ``distributed=False`` runs only the platform/precision
    half -- for single-process drivers whose parameters were resolved
    through :func:`resolve_parameters`; scripts with their own CLI
    usually call :func:`configure_jax_platform` directly instead.

    Returns whether this is the main process
    (``jax.process_index() == 0``), the gate for rank-0 printing.
    Must be called *before* importing :mod:`dnsjax.sharding` or any
    geometry module.
    """
    if distributed and params.dist.platform == "cpu":
        # ``NPROC`` is what actually sizes the CPU thread pool (see the
        # ``Distribution`` docstring); ``setdefault`` so a wide node can
        # override it from the environment.  The Eigen flag only makes
        # sense alongside the 1-thread pin, and it is *prepended* --
        # XLA reads a leading token without ``--`` as a flagfile name
        # and dies, so the composed value must start with a flag, and
        # a user's own ``XLA_FLAGS`` must survive.
        threads = os.environ.setdefault("NPROC", "1")
        if threads == "1":
            existing = os.environ.get("XLA_FLAGS", "")
            os.environ["XLA_FLAGS"] = (
                f"--xla_cpu_multi_thread_eigen=false {existing}".rstrip()
            )

    configure_jax_platform(
        params.dist.platform,
        double_precision=params.res.double_precision,
    )

    import jax

    if distributed:
        jax.distributed.initialize()
    return bool(jax.process_index() == 0)
