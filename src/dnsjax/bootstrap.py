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
   production multi-process path -- launcher bootstrap and, on CPU,
   the cross-process collectives backend), or the single-process
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
import re
import sys
import tomllib
import zlib
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


# ── Multi-process runtime ────────────────────────────────────────


# (rank, size, local rank, local size) as published by the MPI
# launcher, in priority order.  Open MPI is what dnsjax is launched
# under in practice and the only family verified here; the other two
# are the published contracts of the MPICH-derived stacks (Intel MPI
# and Cray MPICH set the ``PMI_*`` pair, MVAPICH2 its own) and are
# untested -- they cost four lines and are the difference between
# "falls back to JAX's detection, which knows no MPI but Open MPI 4"
# and "runs".
_RANK_VARS: tuple[tuple[str, str, str, str], ...] = (
    (
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
    ),
    (
        "MV2_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_SIZE",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_SIZE",
    ),
    ("PMI_RANK", "PMI_SIZE", "MPI_LOCALRANKID", "MPI_LOCALNRANKS"),
)

# The Open MPI daemon URI: ORTE's name (4.x, the one JAX reads) and
# the PRRTE spelling it would have taken in 5.x.  The mixed case is
# verbatim -- an MCA parameter is exported as
# ``<PROJECT>_MCA_<param>`` -- so the capitalized spellings ruff
# suggests (SIM112) are not the variables.  In practice this is an
# Open MPI *4* source only: a 5.0.10 PBS site publishes no variable
# whose name contains ``hnp_uri`` at all (measured), which is why the
# chain below cannot lean on it.
_HNP_URI_VARS: tuple[str, ...] = (
    "OMPI_MCA_orte_hnp_uri",  # noqa: SIM112
    "PRTE_MCA_prte_hnp_uri",  # noqa: SIM112
)

# Seeds for the coordinator port, first one set wins.  Ordered
# launch-scoped before job-scoped: see :func:`_coordinator_port`.
_PORT_VARS: tuple[str, ...] = (
    "PMIX_NAMESPACE",  # one PMIx namespace per launch, any PMIx stack
    "PBS_JOBID",  # PBS Pro / OpenPBS / Torque
    "SLURM_JOB_ID",
    "LSB_JOBID",  # LSF
    "JOB_ID",  # SGE / UGE / Altair Grid Engine
)


@dataclass(frozen=True)
class _Ranks:
    """This process's place in the job, off the launcher environment.

    ``local_size`` is the number of ranks of this job on this node
    (``None`` when the launcher does not publish it); ``family`` names
    the variable that supplied the layout, for diagnostics.
    """

    rank: int
    size: int
    local_rank: int
    local_size: int | None
    family: str


def _launcher_ranks() -> _Ranks | None:
    """The launcher's rank layout for this process, else ``None``.

    The first family of :data:`_RANK_VARS` that publishes rank, size
    and local rank wins; a family whose values do not parse is skipped
    rather than trusted, so a stale or malformed variable falls
    through to JAX's own detection instead of misplacing the process.
    """
    for rank_v, size_v, local_v, local_size_v in _RANK_VARS:
        rank = os.environ.get(rank_v)
        size = os.environ.get(size_v)
        local = os.environ.get(local_v)
        if not rank or not size or not local:
            continue
        local_size = os.environ.get(local_size_v)
        try:
            return _Ranks(
                rank=int(rank),
                size=int(size),
                local_rank=int(local),
                local_size=int(local_size) if local_size else None,
                family=rank_v,
            )
        except ValueError:
            continue
    return None


def _rank_marker() -> str | None:
    r"""A variable saying this process is one rank of several, if any.

    The negative of this is what licenses skipping the distributed
    runtime on a launcher-free machine, so it is a *deliberately* wide
    net -- every stack's "which rank am I" variable, whether or not
    :func:`_launcher_ranks` can build a layout out of its family.

    Getting it wrong in the permissive direction is the worst failure
    in the bootstrap: a multi-rank launch nobody here recognises would
    have every rank believe it is alone, run the whole problem, and
    overwrite the others' ``.dat`` files in the shared directory,
    silently.  A stray marker in a login shell, the other direction,
    costs one error message that names the variable.
    """
    markers = (
        *(family[0] for family in _RANK_VARS),
        "PMIX_RANK",  # any PMIx launcher
        "PMI_ID",  # older MPICH
        "MPI_LOCALRANKID",  # Intel MPI / MPICH, without PMI_RANK
        "SLURM_PROCID",
        "ALPS_APP_PE",  # Cray ALPS
        "FLUX_TASK_RANK",
        "JSM_NAMESPACE_RANK",  # LSF jsrun
    )
    return next((name for name in markers if os.environ.get(name)), None)


def _solo_launch() -> bool:
    """Is this process the whole job?

    The launcher's own count when it published one, else the absence
    of any marker (:func:`_rank_marker`).  A lone process has nothing
    to coordinate, so this is what decides whether the distributed
    runtime is started at all -- see :func:`_bootstrap_distributed`.
    """
    ranks = _launcher_ranks()
    if ranks is not None:
        return ranks.size == 1
    return _rank_marker() is None


def _apply_local_device_ids() -> None:
    r"""Narrow this process's devices the way ``initialize`` would.

    ``JAX_LOCAL_DEVICE_IDS`` first, else the launcher's local rank --
    the same two sources, in the same order, that
    ``jax.distributed.initialize`` reads, and the same thing it does
    with them: set ``jax_cuda_visible_devices`` /
    ``jax_rocm_visible_devices`` (``jax/_src/distributed.py``; the
    flags are consumed when the GPU client is built).  Doing it here
    is what lets a lone process skip that call without changing which
    devices it ends up with.

    Nothing is applied when neither source says anything, which is the
    bare single-process launch: it takes every visible device, so one
    process spans a whole multi-GPU node with no MPI in sight.  On CPU
    the flags are inert, so this needs no platform test.
    """
    import jax

    visible = os.environ.get("JAX_LOCAL_DEVICE_IDS")
    if not visible:
        ranks = _launcher_ranks()
        if ranks is None:
            return
        visible = str(ranks.local_rank)
    ids = ",".join(str(int(i)) for i in visible.split(","))
    jax.config.update("jax_cuda_visible_devices", ids)
    jax.config.update("jax_rocm_visible_devices", ids)


def _coordinator_port() -> str:
    r"""A port every rank of *this launch* derives identically.

    ``JAX_COORDINATOR_PORT`` (JAX's own override, which the explicit
    bootstrap bypasses along with the rest of its detection) wins;
    otherwise the first seed of :data:`_PORT_VARS` that is set is
    mapped into the same ephemeral range JAX's own plugins pick from
    -- by checksum, for the reason at the end of this docstring.

    The order is not cosmetic.  A *scheduler job id* is shared by every
    ``mpirun`` in the job, so seeding on it hands two concurrent
    launches -- the ensemble members of one allocation
    (``scripts/ensemble_setup.py``) -- the same port, whereupon the
    second run's rank 0 connects to the first run's coordination
    service and is killed by it (``INTERNAL: wrong service
    incarnation``, then ``signal 6``; measured, not inferred).
    ``PMIX_NAMESPACE`` identifies the *launch* and is published
    identically to every rank by every PMIx-based launcher (Open MPI 4
    and 5, PRRTE, Flux, Slurm's PMIx plugin), so it comes first and the
    job ids are the fallback for launchers that publish none.  That
    every rank agrees on it is the PMIx contract, and is measured:
    one namespace shared by four ranks on four *different* nodes of a
    scattered Open MPI 5.0.10 PBS job, plus every single-node job
    sampled.  The string
    (``prterun-<launch host>-<pid>@<n>``) names the launch and its
    mother superior rather than the local daemon, which is why it is
    node-invariant -- had it not been, ranks would have derived
    different ports and failed to connect, a startup timeout rather
    than a wrong answer.

    The seed is *checksummed* rather than reduced modulo the range
    directly, because these identifiers are not uniformly distributed
    and one of them is pathological: an Open MPI PMIx namespace is the
    ORTE job id plus one, and job ids are multiples of ``2^12``, so
    ``namespace % 2**12`` is **1** for every launch on the machine
    (measured across four).  JAX's own Open MPI plugin escapes that by
    dividing by ``2^12`` first, which only works because it knows its
    seed is a job id; a checksum works for any format a launcher
    invents, and ``zlib.crc32`` is stable across processes where
    ``hash`` (PYTHONHASHSEED) is not -- ranks must agree.
    """
    override = os.environ.get("JAX_COORDINATOR_PORT")
    if override:
        return override
    seed = ""
    for name in _PORT_VARS:
        # "1240793089", "prterun-node01-3271@1", "8817.pbs01": taken
        # whole, whatever the launcher's format.
        seed = os.environ.get(name, "")
        if seed:
            break
    # Ephemeral range [65535 - 2^12 + 1, 65535], as JAX picks one.
    return str(zlib.crc32(seed.encode()) % 2**12 + (65535 - 2**12 + 1))


def _first_field(path: str | None) -> str | None:
    """First whitespace-separated field of *path*'s first line."""
    if not path:
        return None
    try:
        with open(path) as fh:
            fields = fh.readline().split()
    except OSError:
        return None
    return fields[0] if fields else None


def _hnp_host() -> str | None:
    """The launch node's address out of the Open MPI daemon URI.

    ``OMPI_MCA_orte_hnp_uri`` is what JAX's own Open MPI plugin keys
    on, and Open MPI 5 does not replace it: PRRTE exports no
    ``hnp_uri`` under any name (measured on 5.0.10), which is why 5.x
    falls through that plugin and why the sources around this one --
    not this one -- are what carry it.  The URI reads
    ``<jobid>.<vpid>;tcp://<ip>[,<ip>...]:<port>`` (or ``tcp6://`` with
    the address bracketed), and the host is the node ``mpirun`` itself
    runs on -- rank 0's node under every mapping that does not reorder
    the hosts explicitly.  The parse is JAX's.  The PRRTE spelling is
    kept as a free catch in case some build does export it.
    """
    uri = ""
    for name in _HNP_URI_VARS:
        uri = os.environ.get(name, "")
        if uri:
            break
    if not uri:
        return None
    match = re.search(r"tcp://(.+?)[,:]|tcp6://\[(.+?)[,\]]", uri)
    if match is None:
        return None
    return next((g for g in match.groups() if g is not None), None)


def _slurm_host() -> str | None:
    """First host of the SLURM node list, ``None`` outside SLURM.

    ``SLURM_STEP_NODELIST`` exists only inside a step (``srun``, where
    JAX's own SLURM plugin already fires); an ``sbatch``/``salloc``
    script running ``mpirun`` has ``SLURM_JOB_NODELIST`` only, which is
    the case this covers.  The bracket forms (``node001``,
    ``node001,host2``, ``node[001-0015],host2``,
    ``node[001,007-015],host2``) are parsed as JAX parses them.
    """
    node_list = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get(
        "SLURM_JOB_NODELIST"
    )
    if not node_list:
        return None
    ind = next(
        (i for i, ch in enumerate(node_list) if ch in {",", "["}),
        len(node_list),
    )
    if ind == len(node_list) or node_list[ind] == ",":
        return node_list[:ind]
    suffix = node_list[ind + 1 :]
    ind2 = next((i for i, ch in enumerate(suffix) if ch in {",", "-"}), None)
    return f"{node_list[:ind]}{suffix[:ind2]}"


def _coordinator_host(ranks: _Ranks) -> str | None:
    r"""The node rank 0 lands on, from the launch environment.

    JAX's coordination *service* is started by rank 0 and binds
    ``[::]:port``, so the address every rank connects to has to resolve
    to rank 0's node.  Sources, in order:

    1. **All ranks on this node** (``local size == size``) -- loopback,
       which needs no name resolution, no reachable interface and no
       site variable at all.  This is what makes a laptop, a Mac
       (Homebrew ships Open MPI 5) and a single-node batch job work
       with nothing exported.
    2. The launcher's own daemon URI (:func:`_hnp_host`) -- generic
       across schedulers, and what JAX does for Open MPI 4.
    3. The scheduler's node list: ``PBS_NODEFILE``, the SLURM node list
       (:func:`_slurm_host`), LSF's ``LSB_DJOB_HOSTFILE`` / ``LSB_HOSTS``,
       Grid Engine's ``PE_HOSTFILE``.  Each names the job's first node,
       which is where ``mpirun`` runs and where rank 0 lands under the
       default by-slot mapping, and every rank reads the same value.
       Confirmed on a scattered 4-node PBS job, which is where it
       could have failed: the first entry is rank 0's node even under
       ``--map-by node``, spelled as the FQDN where ``hostname``
       gives the short name -- the better thing to connect to anyway.

    Every source is pinned offline (``tests/test_bootstrap.py``), but
    only the first two have been exercised by a real launcher here;
    the scheduler entries are the published contracts of those
    queueing systems.  A site matching none is one
    ``JAX_COORDINATOR_ADDRESS`` export away, and
    :func:`_bootstrap_distributed` says so by name.
    """
    if ranks.local_size is not None and ranks.local_size == ranks.size:
        return "127.0.0.1"
    hnp = _hnp_host()
    if hnp:
        return hnp
    for name in ("PBS_NODEFILE", "LSB_DJOB_HOSTFILE", "PE_HOSTFILE"):
        host = _first_field(os.environ.get(name))
        if host:
            return host
    slurm = _slurm_host()
    if slurm:
        return slurm
    return (os.environ.get("LSB_HOSTS", "").split() or [None])[0]


def _launcher_params() -> dict[str, object] | None:
    r"""The four distributed parameters off the launcher, else ``None``.

    Supplying all four makes ``jax.distributed.initialize`` skip
    cluster detection altogether; anything less complete returns
    ``None`` and leaves that detection to run unchanged.

    The rank layout comes from the launcher (:func:`_launcher_ranks`)
    and the coordinator from ``JAX_COORDINATOR_ADDRESS``, else from
    :func:`_coordinator_host` and :func:`_coordinator_port`.  That
    combination is what covers Open MPI 5, whose PRRTE launcher
    dropped the ``OMPI_MCA_orte_hnp_uri`` JAX's own plugin keys on
    while still publishing ``OMPI_COMM_WORLD_*``: the rank layout is
    right there and only the address is genuinely missing.

    ``JAX_LOCAL_DEVICE_IDS`` is honoured rather than overwritten -- JAX
    reads it only along the detection path this bypasses, and the
    single-process multi-device launch of the ``Distribution``
    docstring is built on it.
    """
    ranks = _launcher_ranks()
    if ranks is None:
        return None
    address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if address is None:
        host = _coordinator_host(ranks)
        if host is None:
            return None
        address = f"{host}:{_coordinator_port()}"
    visible = os.environ.get("JAX_LOCAL_DEVICE_IDS")
    return {
        "coordinator_address": address,
        "num_processes": ranks.size,
        "process_id": ranks.rank,
        "local_device_ids": (
            [int(i) for i in visible.split(",")]
            if visible
            else [ranks.local_rank]
        ),
    }


def _undetectable_launcher(exc: ValueError) -> str:
    r"""The failure message for a launcher nothing could identify.

    Which half is missing decides the advice, and getting it wrong
    costs a debugging session: a process with no layout to complete
    cannot be helped by exporting ``JAX_COORDINATOR_ADDRESS``, which
    only moves the failure on to ``Number of processes must be
    defined`` (measured).  Reaching this at all means a marker said
    "one rank of several" -- a lone process never gets here, it skips
    the runtime entirely -- so the marker is the thing to name.
    """
    ranks = _launcher_ranks()
    if ranks is None:
        return (
            f"dnsjax: error: {exc}  {_rank_marker()} is set, so this "
            "looks like one rank of a multi-process launch, but "
            "neither dnsjax nor JAX could work out the rank layout. "
            "Launch under mpirun, or -- if this really is a single "
            f"process -- unset {_rank_marker()}."
        )
    return (
        f"dnsjax: error: {exc}  {ranks.family} says this is rank "
        f"{ranks.rank} of {ranks.size}, but no coordinator address "
        "could be derived (tried JAX_COORDINATOR_ADDRESS, the Open "
        "MPI daemon URI, PBS_NODEFILE, the SLURM node list, "
        "LSB_DJOB_HOSTFILE / LSB_HOSTS and PE_HOSTFILE); export "
        "JAX_COORDINATOR_ADDRESS=<rank-0 host>:<port>."
    )


def _bootstrap_distributed() -> None:
    r"""Start JAX's multi-process runtime.

    The launcher environment first, when it describes itself fully
    (:func:`_launcher_params`), else JAX's own cluster detection --
    Slurm, the cloud environments, Open MPI 4.  Rank discovery only;
    it never selects a collective backend.

    **One process means no distributed runtime, ever**
    (:func:`_solo_launch`) -- whether the launcher said so or nothing
    in the environment claims otherwise.  There is nothing to
    coordinate, and ``process_index`` / ``process_count`` are 0 / 1
    either way.  What that buys is not tidiness: a lone run then needs
    no coordinator address at all, so it runs on a laptop, a Mac or a
    login node with no MPI installed and no launcher to interrogate;
    it binds no port, so concurrent single-rank ensemble members
    cannot collide on one; and it cannot hang out JAX's 300 s
    initialization timeout on an address that was never reachable.
    Verified by construction and by measurement: the only
    coordinator-dependent call in the tree is ``snapshot._barrier``,
    itself gated on ``process_count() > 1``, and a full one-process run
    with ``jax.distributed.initialize`` made fatal-if-called
    reproduces the normal run's ``stats.dat`` and snapshot byte for
    byte.

    Skipping it costs nothing in device selection, because
    :func:`_apply_local_device_ids` does by hand the one thing
    ``initialize`` would have done with ``JAX_LOCAL_DEVICE_IDS`` and
    the local rank.  A launch under ``mpirun -np 1`` therefore keeps
    exactly the devices it has today, while a bare one -- which no
    source narrows -- takes every visible device, which is what lets a
    single process drive a whole multi-GPU node without MPI.
    """
    import jax

    if _solo_launch():
        _apply_local_device_ids()
        return
    explicit = _launcher_params()
    if explicit is not None:
        jax.distributed.initialize(**explicit)
        return
    try:
        jax.distributed.initialize()
    except ValueError as exc:  # nothing identified the launcher
        raise SystemExit(_undetectable_launcher(exc)) from None


def _mpiwrapper_lib() -> str | None:
    r"""Find the MPIwrapper library for MPItrampoline, and export it.

    ``jaxlib`` links MPItrampoline statically but ships no MPI: its MPI
    collectives dlopen the library named by ``MPITRAMPOLINE_LIB`` --
    the thin wrapper MPIwrapper builds around the site's own MPI (the
    ``README.md`` "Installation" section) -- and abort without it.
    Honour that variable when it points at a real file, otherwise look
    for ``libmpiwrapper.so`` on ``LD_LIBRARY_PATH`` and write the hit
    back into the environment.  The scan is a plain path test on
    purpose: ``ctypes.util.find_library`` would shell out to
    ``ldconfig`` / ``gcc`` on every rank.

    **Call this before ``import jax``.**  MPItrampoline reads the
    variable while jaxlib is being loaded, so exporting a discovered
    path afterwards is silently too late -- the run reaches its first
    collective and dies with ``MPITRAMPOLINE_LIB is not set`` (measured
    here, not inferred).
    """
    lib = os.environ.get("MPITRAMPOLINE_LIB")
    if lib:
        return lib if Path(lib).is_file() else None
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if entry and (cand := Path(entry) / "libmpiwrapper.so").is_file():
            os.environ["MPITRAMPOLINE_LIB"] = str(cand)
            return str(cand)
    return None


def _select_cpu_collectives(lib: str | None) -> str:
    r"""Pick the CPU cross-process collectives backend; report it.

    JAX defaults to **gloo** (TCP).  Routing the collectives through
    MPI instead is the faster choice on a cluster and costs nothing to
    arrange -- a multi-device CPU run is one process per device, so it
    is under ``mpirun`` by definition -- so take it whenever the
    MPItrampoline wrapper library *lib* was
    found (:func:`_mpiwrapper_lib`, which had to run earlier).  JAX's
    own ``JAX_CPU_COLLECTIVES_IMPLEMENTATION`` wins over that -- JAX
    applies it itself (the flag is an enum *state*, which reads the
    environment), so what is left here is the dispatch pin below,
    which MPI needs however it was selected.

    Two conditions come with MPI, neither of them a choice.  Nothing
    may touch ``MPI_Init`` before XLA: XLA's is unguarded and ignores
    its own return value, so a process that had already initialized
    MPI aborts (Open MPI 4.1.6: ``mpi_init: invoked multiple times``).
    And CPU async dispatch has to go, which is why this also sets
    ``jax_cpu_enable_async_dispatch``: XLA's MPI backend requires every
    communicator request to come from the thread that called
    ``MPI_Init``, while PjRt's CPU client hands an executable to its
    own thread pool as soon as the inputs are not already available.
    That only starts happening once a run has real work in flight, so
    the failure looks like a mesh or machine quirk rather than a race
    -- ``Communicator requested from a thread that is not the one MPI
    was initialized from`` -- and it is fatal.  Reproduced at
    ``np0=4 x np1=4`` on 16 ranks and fixed by the inline dispatch,
    which does not cost MPI its advantage: 0.80 s/t against gloo's
    1.14 (4 ranks, plane-Couette 32^3, interleaved).

    The choice is made per rank against that rank's own filesystem, and
    reported by rank 0 only, so the wrapper library has to be visible
    identically everywhere: a node that cannot see it picks gloo while
    its peers pick MPI, and the run hangs with nothing said.  Exporting
    ``MPITRAMPOLINE_LIB`` in the job script is what guarantees that.

    Returns the one-line startup diagnostic: which backend a run got is
    a performance-relevant decision made by the launch environment
    rather than by a parameter, so it is never silent.
    """
    import jax

    chosen = os.environ.get("JAX_CPU_COLLECTIVES_IMPLEMENTATION")
    if chosen:
        if chosen == "mpi":
            jax.config.update("jax_cpu_enable_async_dispatch", False)
        return (
            f"CPU cross-process collectives: {chosen} "
            "(JAX_CPU_COLLECTIVES_IMPLEMENTATION)."
        )
    if lib is None:
        # Set-but-missing is the interesting half: the advice below is
        # exactly what such a user already did, so name the path
        # instead (a typo, or a $HOME the compute nodes do not mount).
        stale = os.environ.get("MPITRAMPOLINE_LIB")
        if stale:
            return (
                "CPU cross-process collectives: gloo (TCP).  "
                f"MPITRAMPOLINE_LIB names '{stale}', which is not a "
                "file on this rank, so MPI was not taken."
            )
        return (
            "CPU cross-process collectives: gloo (TCP).  MPI is "
            "usually faster: build MPIwrapper and point "
            "MPITRAMPOLINE_LIB at its libmpiwrapper.so (README.md, "
            "'Installation')."
        )
    jax.config.update("jax_cpu_collectives_implementation", "mpi")
    jax.config.update("jax_cpu_enable_async_dispatch", False)
    return f"CPU cross-process collectives: MPI via MPItrampoline ({lib})."


def configure_jax_runtime(distributed: bool = True) -> bool:
    r"""Configure the JAX runtime from the resolved global ``params``.

    The production path (``distributed=True``, used by the ``dnsjax``
    console script / ``python -m dnsjax``): pin the CPU threading via
    ``NPROC`` / ``XLA_FLAGS`` before JAX is imported (CPU platform
    only; one thread per rank, see the ``Distribution`` docstring),
    select platform and precision from ``params.dist.platform`` /
    ``params.res.double_precision`` (:func:`configure_jax_platform`),
    bootstrap the multi-process runtime
    (:func:`_bootstrap_distributed`), and choose the CPU collective
    backend (:func:`_select_cpu_collectives`, multi-device CPU runs
    only -- a single rank has no cross-process collectives to route).
    ``distributed=False`` runs only the platform/precision half -- for
    single-process drivers whose parameters were resolved through
    :func:`resolve_parameters`; scripts with their own CLI usually
    call :func:`configure_jax_platform` directly instead.

    Four environment variables steer the multi-process setup, none
    of them a dnsjax parameter (they describe the machine, not the
    flow): ``JAX_COORDINATOR_ADDRESS`` (and ``JAX_COORDINATOR_PORT``)
    complete the rank bootstrap, ``MPITRAMPOLINE_LIB`` names the
    MPIwrapper library, and ``JAX_CPU_COLLECTIVES_IMPLEMENTATION``
    forces JAX's collective backend.  See the helpers above and the
    ``Distribution`` docstring.

    A **one-process** launch needs none of the multi-process half and
    does not get it (:func:`_bootstrap_distributed`), so a run that
    fits in one process needs no MPI on the machine at all -- not even
    ``mpirun -np 1``.  On CPU that means exactly one device: several
    CPU devices in one process is oversubscription rather than
    parallelism, so asking for it is refused here, with the launch
    that does work.  (The offline tests do force host devices, through
    :func:`configure_jax_platform`; this is the production path.)

    Returns whether this is the main process
    (``jax.process_index() == 0``), the gate for rank-0 printing.
    Must be called *before* importing :mod:`dnsjax.sharding` or any
    geometry module.
    """
    devices = params.dist.np0 * params.dist.np1
    if (
        distributed
        and params.dist.platform == "cpu"
        and devices > 1
        and _solo_launch()
    ):
        raise SystemExit(
            f"dnsjax: error: this is one process, but np0 * np1 = "
            f"{devices} asks for {devices} CPU devices.  A CPU run "
            "takes one device per process: launch it as "
            f"`mpirun -np {devices} ...`."
        )
    mpiwrapper = None
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
        # Before JAX is imported: MPItrampoline reads the variable as
        # jaxlib loads (see ``_mpiwrapper_lib``).  Exporting it even
        # for a run that ends up on gloo costs nothing -- nothing
        # dlopens the wrapper unless the collectives do.
        mpiwrapper = _mpiwrapper_lib()

    configure_jax_platform(
        params.dist.platform,
        double_precision=params.res.double_precision,
    )

    import jax

    # The CPU client reads the collectives implementation when the
    # backend is built -- which the ``jax.process_index()`` below is
    # the first call to trigger -- so the choice has to be made before
    # that, while the rank that reports it is only known after.
    # The device count, not ``jax.process_count()``, is what gates the
    # choice: that call goes through ``jax.devices()`` and would build
    # the very backend the choice has to precede.  Every multi-process
    # CPU launch is one device per rank, so the two agree there.
    note = None
    if distributed:
        _bootstrap_distributed()
        if (
            params.dist.platform == "cpu"
            and params.dist.np0 * params.dist.np1 > 1
        ):
            note = _select_cpu_collectives(mpiwrapper)
    main = bool(jax.process_index() == 0)
    if note is not None and main:
        print(note, flush=True)
    return main
