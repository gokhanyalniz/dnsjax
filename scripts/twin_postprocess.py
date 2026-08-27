#!/usr/bin/env python3
r"""Rebuild the twin difference-field streams from snapshot pairs.

``dnsjax-twin`` writes its diagnostics *online*, so a member directory
recorded before a stream existed carries no trace of it -- and the
trajectory cannot be re-run, since it **is** the data.  What such a
directory does still hold is everything the diagnostics are computed
from: the lockstep snapshot pairs ``state{isnap}.tar`` (reference) +
``state{isnap}_twin.tar`` (partner), written at identical
`$(t, \mathrm{it})$`.

This script walks those pairs and feeds the driver's own writers:

- ``twin.dat`` -- :func:`dnsjax.twin.diagnostics.twin_energies`
- ``twin_yspectra.bin`` (+ sidecar) --
  :func:`~dnsjax.twin.diagnostics.twin_yspectra`
- ``twin_ybudget.bin`` (+ sidecar) --
  :func:`~dnsjax.twin.diagnostics.twin_ybudget`

Nothing is reimplemented: the same jitted diagnostics, the same
:class:`~dnsjax.twin.pressure.DifferencePressure`, the same
:class:`~dnsjax.twin._binstream.BinStream` writers and the same
``.dat`` machinery the live driver uses, so a rebuilt record is the
number the run would have written, bit for bit.

Why this needs JAX (and cannot live in :mod:`dnsjax.analysis`):
:func:`~dnsjax.twin.diagnostics._marginals_replicated` is a
``shard_map`` with a ``psum``, and
:func:`~dnsjax.twin.diagnostics._difference_sources` needs padded
dealiased transforms, the geometry's ``D1``/``D2``/``D1_bnd``/base
profile, and a factored Neumann Poisson operator.

Two things a reconstruction cannot give back
============================================
**The sampling grid is the snapshot grid.**  A live stream samples
every ``twin.it_yspectra`` steps; this one samples wherever
``outs.it_snapshot`` left a pair.  That is why the rebuilt streams go
to a *sub-directory* (``--recon.out``, default ``recon/``) and never
overwrite the run's own ``twin.dat``: the two are different time
series, and the original carries the fine per-step sampling the
growth-rate fits want.  The sub-directory is itself a valid member
directory -- ``twin.json`` is copied into it -- so
:func:`dnsjax.analysis.twin.series.read_twin`,
:func:`~dnsjax.analysis.twin.yspectra.read_twin_yspectra` and
:func:`~dnsjax.analysis.twin.yspectra.read_twin_ybudget` read it as
they would a live one.

**The driving columns are inferred, not applied.**  Under a driving
constraint (``phys.driving = "constant_bulk_velocity"`` or
``phys.block_mean_spanwise_velocity``) ``twin.dat`` carries a
``-dPds'_d`` / ``-dPdn'_d`` column.  In a live run that is the
*applied* mean-mode force threaded out of the corrector -- a **step**
quantity, which no snapshot records.  Here the column is
``get_driving(state2) - get_driving(state1)``, the wall-shear
inference, under the same name.  That is the right substitute rather
than a compromise:

- it is exactly what the driver itself writes for its own ``t = t0``
  row, for exactly this reason (:mod:`dnsjax.__main__`, the comment at
  the read site);
- it is the same inference the ``Pi`` term of ``twin_ybudget`` already
  uses, through
  :func:`~dnsjax.geometries.wall_bounded.cartesian.mean_driving_from_profile`,
  so the rebuilt ``twin.dat`` and ``twin_ybudget.bin`` agree with each
  other;
- the two differ by a wall-normal truncation residual, which shrinks
  with ``res.ny``.  Measured on plane-Poiseuille at ``re = 400``,
  ``nx = nz = 16``, ``constant_bulk_velocity`` with the spanwise mean
  blocked -- the worst relative gap over the stepped rows of a 5-sample
  member (each rung is its own trajectory, so read the trend, not the
  levels):

  =========  ==============  ==============
  ``res.ny``   ``-dPds'_d``    ``-dPdn'_d``
  =========  ==============  ==============
  17         3.7 %           19 %
  33         3.4 %           4.4 %
  65         1.0 %           0.93 %
  =========  ==============  ==============

  At a resolution coarse enough the gap can exceed the term itself
  (``tests/test_driving.py`` measures the same residual against the
  solver's own applied value, where it runs to 60x the viscous term at
  ``ny = 27``), so read a rebuilt driving column as a converged
  quantity only where the run's own ``res.ny`` is converged.

The startup banner says so whenever the column set is non-empty, and
``twin_postprocess.json`` records ``driving: "inferred"``.  Under the
``constant_pressure_gradient`` default there is no such column and the
reconstruction is exact.

Parameters, and why the grid is pinned
======================================
Parameters come from a snapshot in the folder (the first complete
pair, or ``--init.snapshot`` to choose another) through the ordinary
layering, with the ``parameters.toml`` layer switched off -- a member's
own TOML carries ``[twin]`` keys, which do not ride this script's
surface.  ``res.double_precision`` is taken from the snapshot, the only
value :func:`~dnsjax.snapshot.validate_snapshot_params` accepts.

Every selected snapshot must then share that one's ``native_shape``
and wall-normal grid, or the script refuses.  Each stream sidecar
carries a *single* ``ny`` / ``y`` / ``y_weights`` / ``n_kz`` / ``n_kx``
for the whole file, so a folder spanning a mid-run regrid cannot be
described by one stream: interpolating onto the parameter snapshot's
grid would silently relabel the records that were computed on the
other one.  Refusing names both grids and leaves the split to the
caller.

Cost
====
Per pair: two snapshot loads, one ``twin_ybudget`` sample (~33 field
transforms, ~0.9 of a twin step) and two cheap reductions.  Peak memory
matches a live run with ``twin.it_ybudget`` set -- that program is the
high-water mark.  The jitted programs compile once and are reused
across pairs.

Launch it like the solver: a lone process needs no launcher (and on GPU
can span every visible device with ``--dist.np0``); several processes
take ``mpirun``.

Usage::

    uv run python scripts/twin_postprocess.py --recon.dir member/
    uv run python scripts/twin_postprocess.py --recon.dir member/ \
        --recon.out /scratch/recon --recon.stride 4 --dist.platform cuda
    mpirun -np 4 .venv/bin/python scripts/twin_postprocess.py \
        --recon.dir member/ --dist.np0 4
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from dnsjax.analysis.twin.lengths import partner_of
from dnsjax.extensions import ParamExtension, register_extension
from dnsjax.flows.registry import cartesian_systems
from dnsjax.snapshot_meta import git_hash, read_snapshot_meta

_PROG = "python scripts/twin_postprocess.py"

#: ``twin_postprocess.json`` schema version.  A provenance record on
#: the ``ensemble_setup.py`` ``members.json`` model -- nothing reads it
#: back, so it carries no reader floor (unlike the stream sidecars).
PROVENANCE_VERSION: int = 1

#: The output files this script owns; an existing one is refused rather
#: than appended to (both writer families append silently on a matching
#: sidecar, and ``_ScalarStream`` re-writes no header).
_OUTPUTS: tuple[str, ...] = (
    "twin.dat",
    "twin_yspectra.bin",
    "twin_yspectra.json",
    "twin_ybudget.bin",
    "twin_ybudget.json",
)

#: Relative tolerance of the built-in energy identity (``--recon.check``).
_CHECK_TOL: float = 1e-10


class ReconParams(BaseModel):
    r"""Twin stream reconstruction section (this script's knobs).

    ``dir`` is a ``dnsjax-twin`` member directory; the rebuilt streams
    land in ``dir / out`` (or ``out`` when absolute).  Everything else
    -- the flow, the grid, ``step.dt``, the wall-normal quadrature --
    is inherited from the snapshots themselves; see the module
    docstring.
    """

    model_config = ConfigDict(extra="forbid")

    dir: str = Field(
        default=".",
        description=(
            "The dnsjax-twin run directory holding the "
            "state*.tar / state*_twin.tar pairs."
        ),
    )
    out: str = Field(
        default="recon",
        description=(
            "Output directory for the rebuilt streams; relative to "
            "recon.dir unless absolute.  Never the run directory "
            "itself: its twin.dat is a different time series."
        ),
    )
    stride: int = Field(
        default=1,
        ge=1,
        description="Process every n-th snapshot pair.",
    )
    first: int | None = Field(
        default=None,
        ge=0,
        description="Lowest isnap to process (inclusive); unset = all.",
    )
    last: int | None = Field(
        default=None,
        ge=0,
        description="Highest isnap to process (inclusive); unset = all.",
    )
    spectra_ref: bool = Field(
        default=True,
        description=(
            "Also record the reference state's spectra alongside the "
            "difference field's (the twin.spectra_ref convention); "
            "turning it off also skips computing them."
        ),
    )
    bins: bool | None = Field(
        default=None,
        description=(
            "Write the three-bin columns in twin.dat (twin.bins); "
            "unset adopts the source twin.json's value, else off "
            "(analysis.twin.bin_energies recovers them from "
            "twin_yspectra.bin anyway)."
        ),
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "Delete existing streams in the output directory instead "
            "of refusing to touch them."
        ),
    )
    check: bool = Field(
        default=True,
        description=(
            "Verify per record that both spectral marginals integrate "
            "back to twin.dat's E_d, and report the worst deviation."
        ),
    )


def _validate_recon(values: ReconParams, params) -> None:
    """Structural checks the pydantic field constraints cannot express."""
    if values.stride < 1:
        raise ValueError("recon.stride must be >= 1.")
    if (
        values.first is not None
        and values.last is not None
        and values.last < values.first
    ):
        raise ValueError(
            f"recon.last ({values.last}) is below recon.first "
            f"({values.first}); the range selects nothing."
        )
    if params.phys.system not in cartesian_systems:
        raise ValueError(
            "the twin streams exist for the Cartesian wall-bounded "
            f"flows only (system {params.phys.system!r})."
        )
    if params.res.nz % 2:
        # dnsjax.twin.diagnostics._fold_kz's requirement, copied here
        # because the [twin] validate hook does not run under this
        # entry point.
        raise ValueError(
            f"the wall-normal-resolved streams need an even res.nz "
            f"(got {params.res.nz}): the k_z axis is stored folded "
            "onto |k_z|, and at odd nz the highest negative mode has "
            "no positive partner to fold onto "
            "(dnsjax.twin.diagnostics._fold_kz)."
        )
    if params.step.adaptive:
        raise ValueError(
            "the twin streams assume a fixed time step "
            "(step.adaptive = False); the sidecars record one dt."
        )


RECON_EXTENSION = register_extension(
    ParamExtension(
        name="recon",
        model=ReconParams,
        relevant=lambda system: system in cartesian_systems,
        summary="Offline twin-stream reconstruction (this script).",
        validate=_validate_recon,
        # Post-processing configuration, not trajectory state.
        record_in_metadata=False,
    )
)

#: Live ``[recon]`` values (resolved by ``resolve_parameters`` in main).
recon_params: ReconParams = RECON_EXTENSION.values


class Pair(NamedTuple):
    """One lockstep snapshot pair and the metadata that describes it."""

    isnap: int
    t: float
    it: int
    reference: Path
    partner: Path
    shape: tuple[int, ...]
    grid: tuple[float, ...] | None


def discover_pairs(directory: Path) -> tuple[list[Pair], list[Path]]:
    """The complete snapshot pairs in *directory*, ordered in time.

    Returns ``(pairs, orphans)``: the references whose partner is
    missing are reported rather than guessed at.  ``*_twin.tar`` files
    are never treated as references (the ``ensemble_setup.py harvest``
    rule -- a partner is a perturbed copy at the same ``t``, not a
    sample of its own), and an interrupted save is skipped for free by
    the ``*.tar`` glob (``snapshot.py`` commits by renaming a
    ``.partial`` sibling).  A pair whose two halves disagree on
    ``(t, it)`` raises: it is a crash between the driver's two writes,
    and differencing across times would return plausible noise.
    """
    pairs: list[Pair] = []
    orphans: list[Path] = []
    for reference in sorted(directory.glob("state*.tar")):
        if reference.name.endswith("_twin.tar"):
            continue
        partner = partner_of(reference)
        if not partner.is_file():
            orphans.append(reference)
            continue
        meta = read_snapshot_meta(reference)
        pmeta = read_snapshot_meta(partner)
        if (meta["t"], meta["it"]) != (pmeta["t"], pmeta["it"]):
            raise SystemExit(
                f"{_PROG}: error: {reference.name} is at "
                f"(t={meta['t']}, it={meta['it']}) but its partner "
                f"{partner.name} is at (t={pmeta['t']}, "
                f"it={pmeta['it']}); the pair is inconsistent (a crash "
                "between the two writes?)."
            )
        if meta["native_shape"] != pmeta["native_shape"]:
            raise SystemExit(
                f"{_PROG}: error: {reference.name} and "
                f"{partner.name} store different shapes "
                f"({meta['native_shape']} vs {pmeta['native_shape']})."
            )
        grid = meta.get("wall_normal_grid")
        pairs.append(
            Pair(
                isnap=int(meta["isnap"]),
                t=float(meta["t"]),
                it=int(meta["it"]),
                reference=reference,
                partner=partner,
                shape=tuple(meta["native_shape"]),
                grid=None if grid is None else tuple(grid),
            )
        )
    pairs.sort(key=lambda p: (p.it, p.t))
    # A resume seam writes the parent segment's final pair and the
    # child's initial pair at the same clock under two isnaps; keep the
    # first, as the stream readers do (``series._drop_seam_duplicates``).
    seen: set[tuple[float, int]] = set()
    unique: list[Pair] = []
    for pair in pairs:
        if (pair.t, pair.it) in seen:
            continue
        seen.add((pair.t, pair.it))
        unique.append(pair)
    return unique, orphans


def select_pairs(pairs: list[Pair], values: ReconParams) -> list[Pair]:
    """Apply the ``first`` / ``last`` / ``stride`` selection."""
    kept = [
        p
        for p in pairs
        if (values.first is None or p.isnap >= values.first)
        and (values.last is None or p.isnap <= values.last)
    ]
    return kept[:: values.stride]


def check_uniform_grid(pairs: list[Pair]) -> int | None:
    """The constant step spacing of *pairs*, or ``None`` if uneven.

    Recorded as the sidecars' ``it_yspectra`` / ``it_ybudget``, which
    are stream match keys -- so a second run at a different stride into
    the same output directory is refused instead of interleaved.
    """
    if len(pairs) < 2:
        return None
    gaps = {b.it - a.it for a, b in zip(pairs, pairs[1:], strict=False)}
    if len(gaps) != 1:
        return None
    gap = gaps.pop()
    return gap if gap >= 1 else None


def check_one_grid(pairs: list[Pair], reference: Pair) -> None:
    """Refuse a selection spanning more than one discrete grid."""
    for p in pairs:
        if p.shape != reference.shape:
            raise SystemExit(
                f"{_PROG}: error: {p.reference.name} stores shape "
                f"{list(p.shape)} but the parameters come from "
                f"{reference.reference.name}, which stores "
                f"{list(reference.shape)}.  One stream describes one "
                "grid; select a single-resolution range with "
                "--recon.first / --recon.last, or point "
                "--init.snapshot at the other side."
            )
        if p.grid != reference.grid:
            raise SystemExit(
                f"{_PROG}: error: {p.reference.name} was written on a "
                "different wall-normal grid than "
                f"{reference.reference.name}; one stream sidecar "
                "carries one y / y_weights, so the two cannot share a "
                "file.  Select one side with --recon.first / "
                "--recon.last."
            )


def _prepare_output(out: Path, values: ReconParams, main: bool) -> None:
    """Create *out* and clear or refuse pre-existing streams.

    Runs on **every** rank: the ``mkdir`` and the ``unlink`` are both
    idempotent, so no barrier is needed before the writers stat the
    same paths, and the refusal must be unanimous.
    """
    out.mkdir(parents=True, exist_ok=True)
    existing = [name for name in _OUTPUTS if (out / name).is_file()]
    if not existing:
        return
    if not values.overwrite:
        raise SystemExit(
            f"{_PROG}: error: {out} already holds "
            f"{', '.join(existing)}.  The writers would append to "
            "them, mixing two sample grids in one file; pass "
            "--recon.overwrite to replace them, or choose another "
            "--recon.out."
        )
    if main:
        sizes = ", ".join(
            f"{name} ({(out / name).stat().st_size} B)" for name in existing
        )
        print(f"[recon] replacing existing output: {sizes}", flush=True)
    for name in existing:
        (out / name).unlink(missing_ok=True)


def _source_record(directory: Path, main: bool) -> dict:
    """The member's ``twin.json``, or ``{}`` with a warning."""
    path = directory / "twin.json"
    if not path.is_file():
        if main:
            print(
                f"[recon] warning: no twin.json in {directory}; the "
                "stream sidecars will record a null seed / e0 and the "
                "readers will fall back to the first sample for "
                "t_rel.",
                flush=True,
            )
        return {}
    with open(path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # ``_scan_flag`` is bootstrap's own raw pre-parse (the one
    # ``peek_run_context`` uses to find the flow before any model
    # exists); the run directory has to be known one step earlier
    # still, since it supplies the snapshot the parameters come from.
    from dnsjax.bootstrap import _scan_flag, resolve_parameters
    from dnsjax.parameters import params

    if any(a in ("-h", "--help", "--sample-toml") for a in argv):
        # ``resolve_parameters`` renders the surface and exits; the
        # run-directory scan below would refuse first otherwise.
        resolve_parameters(
            argv,
            toml_path=False,
            extensions=(RECON_EXTENSION,),
            prog=_PROG,
        )
        return 0

    directory = Path(_scan_flag(argv, "--recon.dir") or ".")
    if not directory.is_dir():
        raise SystemExit(
            f"{_PROG}: error: --recon.dir {directory} is not a directory"
        )
    pairs, orphans = discover_pairs(directory)
    if not pairs:
        raise SystemExit(
            f"{_PROG}: error: no complete state*.tar / state*_twin.tar "
            f"pair in {directory}"
        )
    if _scan_flag(argv, "--init.snapshot") is None:
        argv = [*argv, "--init.snapshot", str(pairs[0].reference)]

    # Snapshot + CLI layers only: an ensemble member's own
    # parameters.toml carries [twin] keys, which are not on this
    # surface and would be a hard error.
    resolve_parameters(
        argv,
        toml_path=False,
        extensions=(RECON_EXTENSION,),
        prog=_PROG,
    )
    values = recon_params

    param_snapshot = Path(params.init.snapshot).resolve()
    param_pair = next(
        (p for p in pairs if p.reference.resolve() == param_snapshot),
        pairs[0],
    )
    kept = select_pairs(pairs, values)
    if not kept:
        raise SystemExit(
            f"{_PROG}: error: the isnap selection (first="
            f"{values.first}, last={values.last}, stride="
            f"{values.stride}) keeps none of the {len(pairs)} pairs in "
            f"{directory}"
        )
    check_one_grid(kept, param_pair)
    # A resolution override would leave the loaded state's axes at the
    # stored sizes while every derived object follows params.
    expect = (params.res.ny, params.res.nz - 1, params.res.nx // 2)
    if param_pair.shape[1:] != expect:
        raise SystemExit(
            f"{_PROG}: error: the resolved resolution "
            f"(ny={params.res.ny}, nz={params.res.nz}, "
            f"nx={params.res.nx}) does not match what "
            f"{param_pair.reference.name} stores "
            f"({list(param_pair.shape[1:])} = (ny, nz-1, nx//2)); this "
            "script rebuilds a stream on the snapshots' own grid and "
            "does not regrid."
        )

    # The snapshot's precision, before JAX initializes any array:
    # validate_snapshot_params rejects any other choice anyway.
    stored_dp = bool(
        read_snapshot_meta(param_pair.reference)["params"]["res"][
            "double_precision"
        ]
    )
    params.res.double_precision = stored_dp

    out = Path(values.out)
    if not out.is_absolute():
        out = directory / out
    if out.resolve() == directory.resolve():
        raise SystemExit(
            f"{_PROG}: error: --recon.out resolves to the run "
            "directory itself.  Its twin.dat is the run's own, on a "
            "finer sample grid; the rebuilt streams go beside it, not "
            "over it."
        )

    # Every refusal above is free; from here on the run costs a JAX
    # init and a geometry build.
    from dnsjax.bootstrap import configure_jax_runtime

    main_device = configure_jax_runtime()

    it_stride = check_uniform_grid(kept)
    source = _source_record(directory, main_device)
    bins = bool(
        source.get("bins", False) if values.bins is None else values.bins
    )

    if main_device:
        print(f"[recon] source: {directory.resolve()}", flush=True)
        print(f"[recon] output: {out.resolve()}", flush=True)
        print(
            f"[recon] parameters from {param_pair.reference.name} "
            f"(double_precision={stored_dp})",
            flush=True,
        )
        for path in orphans:
            print(
                f"[recon] warning: skipping {path.name}: no partner "
                f"{partner_of(path).name}",
                flush=True,
            )
        print(
            f"[recon] {len(kept)} of {len(pairs)} pairs: isnap "
            f"{kept[0].isnap}..{kept[-1].isnap}, t {kept[0].t:g}.."
            f"{kept[-1].t:g}, it {kept[0].it}..{kept[-1].it}, cadence "
            + (f"{it_stride} steps" if it_stride else "non-uniform"),
            flush=True,
        )
        if it_stride is None and len(kept) > 1:
            print(
                "[recon] warning: the pairs are not evenly spaced in "
                "it; the sidecars record a null cadence.",
                flush=True,
            )

    _prepare_output(out, values, main_device)

    # Everything below captures the singletons at import: params are
    # final and the JAX runtime is configured (the bootstrap contract).
    # ``dnsjax.twin.driver`` registers the [twin] section on import,
    # which is inert here: the explicit ``extensions=`` tuple above
    # keeps it off this surface, and the stream writers take their
    # values as an argument rather than importing the singleton
    # (:mod:`dnsjax.twin.spectra` says why).
    import importlib

    import numpy as np
    from jax import numpy as jnp

    from dnsjax.__main__ import _stats_row as _row
    from dnsjax.flows.registry import spec_for
    from dnsjax.sharding import sharding
    from dnsjax.snapshot import load_snapshot, validate_snapshot_params
    from dnsjax.twin import diagnostics
    from dnsjax.twin.driver import TwinParams, _ScalarStream
    from dnsjax.twin.pressure import DifferencePressure
    from dnsjax.twin.yspectra import TwinYBudgetStream, TwinYSpectraStream

    get_driving = getattr(
        importlib.import_module(spec_for(params.phys.system).flow_module),
        "get_driving",
        None,
    )

    stream_values = TwinParams(
        e0=source.get("e0"),
        seed=source.get("seed"),
        smoothness=source.get("smoothness", TwinParams().smoothness),
        bins=bins,
        spectra_ref=values.spectra_ref,
        it_yspectra=it_stride,
        it_ybudget=it_stride,
    )
    y_weights = diagnostics.flow.y_weights
    pressure = DifferencePressure(diagnostics.flow, diagnostics.fourier)

    def drive_difference(state1, state2) -> dict:
        """The wall-shear inference of `$\\Delta\\Pi$` (module docstring)."""
        if get_driving is None:
            return {}
        d1, d2 = get_driving(state1), get_driving(state2)
        return {f"{k}_d": d2[k] - d1[k] for k in d1}

    # One pair is loaded up front so the twin.dat column set is known
    # before the stream opens (the driver's warm-up, for the same
    # reason); the streams then open together, so a refusal from one
    # cannot leave the others half-written.
    state1, _, _ = load_snapshot(kept[0].reference)
    state2, _, _ = load_snapshot(kept[0].partner)
    tvals = diagnostics.twin_energies(state1, state2, bins=bins)
    drive_d = drive_difference(state1, state2)
    del state1, state2

    if main_device and drive_d:
        print(
            "[recon] note: the "
            + " / ".join(drive_d)
            + " column(s) carry the wall-shear *inference* of the "
            "driving difference, not the corrector's applied value "
            "(a step quantity no snapshot records); see the module "
            "docstring.",
            flush=True,
        )

    twin_stream = _ScalarStream(
        out / "twin.dat",
        list(tvals.keys()) + list(drive_d.keys()),
        jnp=jnp,
        sharding=sharding,
    )
    yspectra_stream = TwinYSpectraStream(
        stream_values, np.asarray(y_weights).tolist(), directory=out
    )
    ybudget_stream = TwinYBudgetStream(
        stream_values, np.asarray(y_weights).tolist(), directory=out
    )
    streams = (twin_stream, yspectra_stream, ybudget_stream)

    def abort(reason: str) -> None:
        """The ``dnsjax.__main__`` non-finite contract: FATAL, exit 3."""
        sharding.print(f"FATAL: {reason}; aborting.")
        for stream in streams:
            stream.flush(check=False)
        sys.exit(3)

    worst = 0.0
    for n, pair in enumerate(kept, start=1):
        validate_snapshot_params(pair.reference)
        validate_snapshot_params(pair.partner)
        state1, t, _ = load_snapshot(pair.reference)
        state2, _, _ = load_snapshot(pair.partner)

        tvals = diagnostics.twin_energies(state1, state2, bins=bins)
        drive_d = drive_difference(state1, state2)
        yvals = diagnostics.twin_yspectra(
            state1, state2, ref=values.spectra_ref
        )
        ybvals = diagnostics.twin_ybudget(state1, state2, pressure)

        for bad in (
            twin_stream.push(_row(tvals, drive_d), t),
            yspectra_stream.record(yvals, t),
            ybudget_stream.record(ybvals, t),
        ):
            if bad is not None:
                abort(bad)

        if values.check:
            e_d = float(tvals["E_d"])
            for axis in ("e_x", "e_z"):
                got = float(jnp.einsum("j,cjk->", y_weights, yvals[axis]))
                dev = abs(got - e_d) / e_d if e_d > 0 else abs(got)
                worst = max(worst, dev)
                if dev > _CHECK_TOL:
                    sharding.print(
                        f"[recon] CHECK FAILED at t = {t:.6e}: "
                        f"sum_k int {axis} = {got:.12e} but E_d = "
                        f"{e_d:.12e} (relative {dev:.3e})"
                    )
        if main_device and (n % 10 == 0 or n == len(kept)):
            print(
                f"[recon] [{n}/{len(kept)}] isnap {pair.isnap}, t = {t:g}",
                flush=True,
            )
        del state1, state2

    for stream in streams:
        bad = stream.flush()
        if bad is not None:
            abort(bad)

    if main_device:
        # Copied verbatim so the rebuilt directory is a member
        # directory: ``series.read_twin`` reads ``parent_t`` from it.
        if (directory / "twin.json").is_file():
            shutil.copy2(directory / "twin.json", out / "twin.json")
        with open(out / "twin_postprocess.json", "w") as f:
            json.dump(
                {
                    "format_version": PROVENANCE_VERSION,
                    "source_dir": str(directory.resolve()),
                    "param_snapshot": str(param_pair.reference.resolve()),
                    "streams": [
                        name for name in _OUTPUTS if not name.endswith(".json")
                    ],
                    "n_pairs": len(kept),
                    "n_pairs_available": len(pairs),
                    "stride": values.stride,
                    "isnap_first": kept[0].isnap,
                    "isnap_last": kept[-1].isnap,
                    "it_first": kept[0].it,
                    "it_last": kept[-1].it,
                    "it_stride": it_stride,
                    "t_first": kept[0].t,
                    "t_last": kept[-1].t,
                    "bins": bins,
                    "spectra_ref": values.spectra_ref,
                    "driving": "inferred" if drive_d else "none",
                    "created": datetime.now(UTC).isoformat(),
                    "git_hash": git_hash(),
                },
                f,
                indent=2,
            )
        print(
            f"[recon] wrote {len(kept)} records to {out}/"
            + (
                f"; worst energy-identity deviation {worst:.2e}"
                if values.check
                else ""
            ),
            flush=True,
        )
    if values.check and worst > _CHECK_TOL:
        sharding.print(
            f"[recon] FAILED: the spectral marginals miss twin.dat's "
            f"E_d by {worst:.3e} (tolerance {_CHECK_TOL:.0e}); the "
            "streams are written but should not be trusted."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
