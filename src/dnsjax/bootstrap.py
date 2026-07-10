r"""Shared entry-point setup: parameter layering and JAX runtime init.

Every dnsjax entry point -- the ``dnsjax`` console script /
``python -m dnsjax``, the diagnostic scripts, the offline tests, and
future drivers (e.g. Newton solvers) -- must finalize parameters and
configure JAX in the same order **before** importing
:mod:`dnsjax.sharding` or any geometry module, because those modules
capture ``params`` / ``padded_res`` / the devices in module-level
singletons at import time:

1. Finalize the global ``params``: :func:`resolve_parameters` (the
   production CLI / TOML / snapshot layering), or a script's own
   ``update_parameters`` calls followed by ``validate_parameters()``
   and ``padded_res.set_padded_resolution(params)``.
2. Configure the JAX runtime: :func:`configure_jax_runtime` (the
   production multi-process path), or the single-process
   :func:`configure_jax_platform` (typically fed by
   :func:`platform_from_argv`).
3. Only then import :mod:`dnsjax.sharding` and the geometry / flow
   modules.

This module is JAX-free at import time (JAX is imported inside the
configuration functions), so it is always safe to import first.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import CliApp

from .parameters import (
    CLIParameters,
    Parameters,
    padded_res,
    params,
    read_parameters,
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
    import sys

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
    that only the production entry point needs.  Must be called *before*
    importing :mod:`dnsjax.sharding` or any geometry module.

    After this, :data:`sharding` reports the active device unambiguously
    (its banner reads the live device, so a stale ``params.dist.platform``
    can no longer contradict it), and ``--dist.platform cuda`` (via
    :func:`platform_from_argv`) runs the real Pallas / Triton kernels on a
    GPU from any script or test, not just the production entry point.
    """
    if platform not in _VALID_PLATFORMS:
        raise ValueError(
            f"unknown platform {platform!r}; expected one of "
            f"{_VALID_PLATFORMS}"
        )
    params.dist.platform = platform

    import jax

    jax.config.update("jax_enable_x64", double_precision)
    jax.config.update("jax_platforms", platform)


def _resume_snapshot_path(
    params_cli: Parameters, params_in: Parameters | None
) -> Path | None:
    """Return the snapshot path to resume from (CLI over TOML).

    Inspects only the *explicitly set* ``init.snapshot`` of each layer
    (via ``exclude_unset``), so an unset field never shadows a lower
    layer.  Returns ``None`` when neither layer sets it (a laminar start,
    or the path lives only in the code defaults -- which is ``None``).
    """
    for layer in (params_cli, params_in):
        if layer is None:
            continue
        init = layer.model_dump(exclude_unset=True).get("init") or {}
        snap = init.get("snapshot")
        if snap is not None:
            return Path(snap)
    return None


@dataclass(frozen=True)
class ResolvedSetup:
    """What the production startup banner needs from the layering.

    ``params_cli`` is the parsed CLI layer (kept for callers that need
    to inspect explicitly-set fields), ``params_from_disk`` whether a
    ``parameters.toml`` was loaded, ``snapshot_path`` /
    ``snapshot_params_used`` which snapshot (if any) contributed its
    embedded parameters as the lowest layer.
    """

    params_cli: Parameters
    params_from_disk: bool
    snapshot_path: Path | None
    snapshot_params_used: bool


def resolve_parameters(cli_args: list[str] | None = None) -> ResolvedSetup:
    r"""Resolve the production parameter layers into the global ``params``.

    Layering, lowest priority first: code defaults -> the embedded
    parameters of the snapshot being resumed (whichever
    ``init.snapshot`` was set explicitly, CLI over TOML) -> a
    ``parameters.toml`` in the current directory -> the command line
    (*cli_args*; ``None`` parses ``sys.argv``).  ``--help`` is handled
    by the CLI parser (prints usage and raises ``SystemExit``).  Ends
    with ``validate_parameters()`` and
    ``padded_res.set_padded_resolution(params)``, so ``params`` and
    ``padded_res`` are final on return.  Must run *before*
    :func:`configure_jax_runtime` (which reads the resolved platform /
    precision) and before importing :mod:`dnsjax.sharding`.
    """
    params_cli = CliApp.run(CLIParameters, cli_args=cli_args)

    params_file = Path("parameters.toml")
    params_in = (
        read_parameters(params_file) if Path.is_file(params_file) else None
    )

    # Configuration layers, lowest priority first.  The snapshot to
    # resume from is whichever ``init.snapshot`` the user set explicitly
    # (CLI over TOML); its embedded parameters form the lowest layer
    # above the code defaults, so a resume inherits the snapshot's
    # configuration unless TOML or the CLI override it.
    snapshot_path = _resume_snapshot_path(params_cli, params_in)
    snapshot_params_used = False
    if snapshot_path is not None:
        snap_params = read_snapshot_params(snapshot_path)
        if snap_params is not None:
            update_parameters(snap_params)
            snapshot_params_used = True

    # Higher-priority layers: parameters.toml, then CLI arguments.
    if params_in is not None:
        update_parameters(params_in)
    update_parameters(params_cli)

    validate_parameters()
    padded_res.set_padded_resolution(params)
    return ResolvedSetup(
        params_cli=params_cli,
        params_from_disk=params_in is not None,
        snapshot_path=snapshot_path,
        snapshot_params_used=snapshot_params_used,
    )


def configure_jax_runtime(distributed: bool = True) -> bool:
    r"""Configure the JAX runtime from the resolved global ``params``.

    The production path (``distributed=True``, used by the ``dnsjax``
    console script / ``python -m dnsjax``): pin the CPU threading via
    ``XLA_FLAGS`` before JAX is imported (CPU platform only), select
    platform and precision from ``params.dist.platform`` /
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
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
        )
        os.environ["NPROC"] = "1"

    configure_jax_platform(
        params.dist.platform,
        double_precision=params.res.double_precision,
    )

    import jax

    if distributed:
        jax.distributed.initialize()
    return bool(jax.process_index() == 0)
