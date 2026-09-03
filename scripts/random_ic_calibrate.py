r"""Calibrate the random initial condition against a twin ensemble.

The random perturbation of :mod:`dnsjax.ic.random_field` has a shape
in `$(y, k)$` -- set by ``init.random_smoothness``,
``random_wall_smoothness`` and ``random_wall_confinement`` -- and a
turbulent difference field has one of its own, which it settles onto
within a few advective time units whatever it started from.  The gap
between the two is a transient that every twin run pays before its
growth rate means anything.  This script measures the gap, so a
default can be *calibrated* rather than assumed.

**What it computes.**  The initial condition is built by the shipped
generator (:func:`dnsjax.ic.random_field.generate_random_state`) and
marginalised by the shipped stream function
(:func:`dnsjax.twin.diagnostics.twin_yspectra`, evaluated against a
zero reference so the difference field *is* the perturbation) -- so
the map compared here and the `$t = 0$` record of a real
``twin_yspectra.bin`` are the same object, by construction rather
than by agreement.  The target is the ensemble- and time-averaged
shape of one or more recorded members over a window of their
exponential-growth phase.  The two are compared by
:func:`dnsjax.analysis.twin.yspectra.shape_alignment`, the
`$y$`-weighted Bhattacharyya overlap `$A \in [0, 1]$`, with the
`$(0, 0)$` mode taken off both: it is the wall-parallel mean, not
part of the shape.

**What `$A$` buys, in time.**  Measured against the same target,
`$1 - A(t)$` of a recorded member decays exponentially at a rate
`$\mu$` that is a property of the flow, not of the initial condition.
So no calibration removes the alignment transient; it only starts it
further along, by

.. math::
    \Delta t = \frac{1}{\mu}\,
        \ln\frac{1 - A_{\text{old}}}{1 - A_{\text{new}}} .

The script fits `$\mu$` from the ensemble it was given and reports
`$\Delta t$` against the recorded members' own `$A(0)$`, so the
answer is in advective time units rather than in an abstract overlap.
It is printed as ``shape lead`` and not ``head start`` on purpose: it
is how much sooner this field would *look* settled, which is not the
same as how much sooner it grows -- see the next paragraph.

**Reading the output.**  A bare run reports one configuration --
whatever the surface resolved, which with no overrides is the shipped
default.  ``--calib.sweep_smoothness`` and its two siblings take
comma-separated lists and report their outer product, one line each,
rebuilding the field per entry (a per-mode host loop: seconds at a
production resolution, so a dozen entries is a coffee, not a job).
The seed is `` init.random_seed`` as usual, and `$A$` is essentially
seed-independent -- it averages over every resolved mode, and moves
by under 0.005 across seeds at a production box -- so a sweep needs
one seed, not an ensemble.

**A(0) is a shape score, not a growth predictor.**  This is the
caveat that decides how the output should be used, and it is measured
rather than hedged.  At an HKW minimal plane-Couette box
(`$Re = 400$`, `$Re_\tau \approx 34$`) the score peaks at
``random_smoothness`` `$\approx 0.15$` and calls it worth thirteen
time units -- while the difference energy actually measured over 60
advective units goes 2.07 decades at `$s = 0.4$`, 1.15 at `$0.15$` and
0.09 at `$0.04$`, decaying outright for the first five units at the
last (two seeds, both).  The reason is physical: an attractor's
small-scale content is sustained by transfer from larger scales, so
seeding it directly feeds `$k^2/Re$` rather than the instability.
Read `$A(0)$` as "how much of this field is in the wrong part of
`$(y, k)$`", and confirm any change of ``random_smoothness`` with a
growth run before adopting it.  ``random_wall_confinement``, whose
optimum this tool also reports, was adopted on the strength of being
*neutral* in that same growth measurement.

**Scope.**  Cartesian wall-bounded flows, like ``dnsjax-twin`` itself:
the target has to come from a recorded twin stream, and those exist
only there.  The measure itself
(:func:`~dnsjax.analysis.twin.yspectra.shape_alignment`) is
geometry-general and lives with the readers.

Usage::

    python scripts/random_ic_calibrate.py \
        --phys.system plane-poiseuille --phys.re 4200 \
        --geo.lx 12.566370614 --geo.lz 6.283185307 \
        --res.nx 192 --res.ny 215 --res.nz 160 \
        --init.random_seed 1 --init.random_mean_flow True \
        --calib.target /data/twin/KMM4200/twins-redo \
        --calib.window "15,30" \
        --calib.sweep_smoothness "0.4,0.06,0.04,0.03" \
        --calib.sweep_wall_confinement "0,0.11,0.14,0.20"

``--calib.target`` is either one member directory or a tree of them
(``--calib.members`` is the glob, default ``twin*``).  Flow, box,
resolution and grid must match the recorded stream's -- a mismatch is
refused rather than interpolated, because the two maps would not be
on the same axes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dnsjax.extensions import ParamExtension, register_extension
from dnsjax.flows.registry import cartesian_systems

_PROG = "python scripts/random_ic_calibrate.py"

#: Records whose alignment feeds the ``mu`` fit, on the **rise** only:
#: below the first the shape is still in its initial burst, and above
#: the second it has arrived.  The prefix matters -- ``A(t)`` climbs to
#: ~1 across the target window and then drifts back down as the field
#: saturates, so a band alone would fit both slopes at once and return
#: about zero.
_FIT_RANGE = (0.5, 0.98)


class CalibParams(BaseModel):
    r"""Calibration-run knobs (``[calib]``), this script only.

    Everything else -- the flow, the box, the resolution, the grid and
    the three ``init.random_*`` shape knobs -- comes from the ordinary
    per-flow surface (:func:`dnsjax.bootstrap.resolve_parameters`), so
    a calibration is configured exactly like the run it is calibrating.
    """

    model_config = ConfigDict(extra="forbid")

    target: Path | None = Field(
        default=None,
        description=(
            "Recorded twin member directory, or a tree of them, whose "
            "twin_yspectra stream provides the target shape."
        ),
    )
    members: str = Field(
        default="twin*",
        description=("Glob selecting member directories under a target tree."),
    )
    window: str = Field(
        default="15,30",
        description=(
            "Time window of the target, as 'lo,hi' in advective units "
            "since each member's perturbation."
        ),
    )
    marginal: Literal["x", "z"] = Field(
        default="x",
        description=(
            "Which stored marginal to compare: 'x' is resolved in k_z "
            "(summed over k_x), 'z' resolved in k_x."
        ),
    )
    sweep_smoothness: str | None = Field(
        default=None,
        description=(
            "Comma-separated init.random_smoothness values to report "
            "instead of the resolved one."
        ),
    )
    sweep_wall_smoothness: str | None = Field(
        default=None,
        description=("Comma-separated init.random_wall_smoothness values."),
    )
    sweep_wall_confinement: str | None = Field(
        default=None,
        description=(
            "Comma-separated init.random_wall_confinement values; "
            "zero is the unnarrowed wall window."
        ),
    )


def _validate_calib(values: CalibParams, params) -> None:
    if values.target is None:
        return
    if params.phys.system not in cartesian_systems:
        raise ValueError(
            "calib.target: the calibration compares against a "
            "dnsjax-twin stream, so it supports the Cartesian "
            f"wall-bounded flows only (system {params.phys.system!r})."
        )
    _parse_window(values.window)
    for name in (
        "sweep_smoothness",
        "sweep_wall_smoothness",
        "sweep_wall_confinement",
    ):
        raw = getattr(values, name)
        if raw is not None:
            _parse_floats(raw, f"calib.{name}")


CALIB_EXTENSION = register_extension(
    ParamExtension(
        name="calib",
        model=CalibParams,
        relevant=lambda system: system in cartesian_systems,
        summary="Random-IC calibration target (this script's knobs).",
        validate=_validate_calib,
        # A measurement run, not trajectory state.
        record_in_metadata=False,
    )
)

#: Live ``[calib]`` values (resolved by ``resolve_parameters``).
calib_params: CalibParams = CALIB_EXTENSION.values


# ── Argument parsing ─────────────────────────────────────────────


def _parse_window(raw: str) -> tuple[float, float]:
    """``"lo,hi"`` as a pair, refusing anything else."""
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"calib.window {raw!r}: expected 'lo,hi' in advective units"
        )
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"calib.window {raw!r}: {exc}") from exc
    if not hi > lo:
        raise ValueError(f"calib.window {raw!r}: needs hi > lo")
    return lo, hi


def _parse_floats(raw: str, label: str) -> list[float]:
    """A comma-separated sweep list."""
    out: list[float] = []
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError as exc:
            raise ValueError(f"{label}: {tok!r} is not a number") from exc
    if not out:
        raise ValueError(f"{label}: empty list")
    return out


# ── The target ───────────────────────────────────────────────────


def member_dirs(target: Path, glob: str) -> list[Path]:
    """The member directories *target* names: itself if it holds a
    stream, else its *glob* children that do."""
    if (target / "twin_yspectra.json").is_file():
        return [target]
    found = sorted(
        d for d in target.glob(glob) if (d / "twin_yspectra.json").is_file()
    )
    if not found:
        raise SystemExit(
            f"{_PROG}: error: no twin_yspectra stream under {target} "
            f"(glob {glob!r})"
        )
    return found


def _member_shapes(
    directory: Path, marginal: str
) -> tuple[dict, np.ndarray, np.ndarray]:
    r"""``(meta, t_rel, shapes)`` for one member.

    *shapes* is ``(n_t, n_y, n_k)``, component-summed, the `$(0, 0)$`
    mode removed, each record normalised to unit total -- the shape
    alone.  Read through a memory map: the eager reader would pull a
    gigabyte per member for the handful of records a window needs
    (:func:`dnsjax.analysis.twin.yspectra.record_dtype` exists for
    exactly this).
    """
    from dnsjax.analysis.twin.yspectra import (
        mean_free_spectrum,
        mean_mode_name,
        mean_mode_profile,
        record_dtype,
    )

    meta = json.loads((directory / "twin_yspectra.json").read_text())
    dtype = record_dtype(meta, "twin_yspectra")
    records = np.memmap(directory / "twin_yspectra.bin", dtype=dtype, mode="r")
    weights = np.asarray(meta["y_weights"], dtype=np.float64)
    name00 = mean_mode_name(meta, "e")

    parent_t = None
    sidecar = directory / "twin.json"
    if sidecar.is_file():
        parent_t = json.loads(sidecar.read_text()).get("parent_t")
    times = np.asarray(records["t"], dtype=np.float64)
    if parent_t is None:
        parent_t = float(times[0])
    t_rel = times - float(parent_t)

    out = np.empty((len(records), int(meta["ny"]), _n_k(meta, marginal)))
    for i in range(len(records)):
        field = records[f"e_{marginal}"][i].astype(np.float64).sum(0)
        mean = mean_mode_profile(
            records[name00][i].astype(np.float64), name00
        ).sum(0)
        field = mean_free_spectrum(field, mean)
        out[i] = field / float(np.einsum("j,jk->", weights, field))
    del records
    return meta, t_rel, out


def _n_k(meta: dict, marginal: str) -> int:
    return int(meta["n_kz"] if marginal == "x" else meta["n_kx"])


def build_target(
    members: list[Path], marginal: str, window: tuple[float, float]
) -> tuple[dict, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    r"""The averaged target shape, plus each member's own series.

    Returns ``(meta, target, series)``: *target* is the mean of every
    record whose time since the perturbation lies in *window*, over
    every member, renormalised; *series* pairs each member's `$t$`
    with its per-record shapes, for the `$\mu$` fit.
    """
    from dnsjax.analysis.twin.yspectra import shape_alignment  # noqa: F401

    meta0 = None
    total = None
    count = 0
    series: list[tuple[np.ndarray, np.ndarray]] = []
    for directory in members:
        meta, t_rel, shapes = _member_shapes(directory, marginal)
        if meta0 is None:
            meta0, total = meta, np.zeros_like(shapes[0])
        elif (meta["ny"], _n_k(meta, marginal)) != (
            meta0["ny"],
            _n_k(meta0, marginal),
        ):
            raise SystemExit(
                f"{_PROG}: error: {directory} stores "
                f"({meta['ny']}, {_n_k(meta, marginal)}) where the "
                "first member stores "
                f"({meta0['ny']}, {_n_k(meta0, marginal)}); an "
                "ensemble must share its axes."
            )
        picked = (t_rel >= window[0]) & (t_rel <= window[1])
        total += shapes[picked].sum(axis=0)
        count += int(picked.sum())
        series.append((t_rel, shapes))
    if count == 0:
        raise SystemExit(
            f"{_PROG}: error: no records in the window {window}; the "
            "members cover "
            f"[{series[0][0][0]:.3g}, {series[0][0][-1]:.3g}]"
        )
    weights = np.asarray(meta0["y_weights"], dtype=np.float64)
    target = total / float(np.einsum("j,jk->", weights, total))
    return meta0, target, series


def fit_mu(
    series: list[tuple[np.ndarray, np.ndarray]],
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    r"""``(mu, A0, t, A)`` of the recorded members.

    `$A(t)$` is the ensemble mean overlap with *target* and `$\mu$`
    the decay rate of `$1 - A$` over :data:`_FIT_RANGE`, least squares
    in `$\ln(1 - A)$`.  `$A_0$` is the members' own starting overlap,
    i.e. what the initial condition they were run with was worth.
    """
    from dnsjax.analysis.twin.yspectra import shape_alignment

    grid = series[0][0]
    stack = np.array(
        [
            [shape_alignment(s, target, weights) for s in shapes]
            for _, shapes in series
        ]
    )
    align = stack.mean(axis=0)
    arrived = np.flatnonzero(align > _FIT_RANGE[1])
    end = int(arrived[0]) if arrived.size else align.size
    band = np.zeros(align.shape, dtype=bool)
    band[:end] = align[:end] >= _FIT_RANGE[0]
    if band.sum() < 3:
        return float("nan"), float(align[0]), grid, align
    slope = np.polyfit(grid[band], np.log1p(-align[band]), 1)[0]
    return -float(slope), float(align[0]), grid, align


# ── The candidate initial condition ──────────────────────────────


def ic_shape(marginal: str) -> tuple[np.ndarray, np.ndarray]:
    r"""``(y_weights, shape)`` of the perturbation the resolved
    ``init.random_*`` knobs build.

    Marginalised by :func:`dnsjax.twin.diagnostics.twin_yspectra`
    against a zero reference, so the map is the stream's own `$t = 0$`
    record and not a re-derivation of it.
    """
    import jax.numpy as jnp

    from dnsjax.ic.random_field import generate_random_state
    from dnsjax.parameters import params
    from dnsjax.twin.diagnostics import flow, twin_yspectra

    delta = generate_random_state(
        params.init.random_amplitude,
        params.init.random_smoothness,
        params.init.random_wall_smoothness,
        params.init.random_wall_confinement,
        params.init.random_seed,
        params.init.random_mean_flow,
    )
    spectra = twin_yspectra(jnp.zeros_like(delta), delta, ref=False, x0=False)
    field = np.asarray(spectra[f"e_{marginal}"], dtype=np.float64).sum(0)
    mean = np.asarray(spectra["e_xz00"], dtype=np.float64).sum(0)
    field = field.copy()
    field[:, 0] -= mean
    weights = np.asarray(flow.y_weights, dtype=np.float64)
    return weights, field / float(np.einsum("j,jk->", weights, field))


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from dnsjax.analysis.twin.yspectra import shape_alignment
    from dnsjax.bootstrap import configure_jax_platform, resolve_parameters
    from dnsjax.parameters import params, update_parameters

    resolve_parameters(argv, extensions=(CALIB_EXTENSION,), prog=_PROG)
    if calib_params.target is None:
        raise SystemExit(f"{_PROG}: error: --calib.target is required")
    if params.dist.np0 * params.dist.np1 != 1:
        raise SystemExit(f"{_PROG}: error: single-device (np0*np1 = 1)")

    window = _parse_window(calib_params.window)
    marginal = calib_params.marginal
    members = member_dirs(calib_params.target, calib_params.members)
    print(f"Target: {len(members)} member(s), t in {window}", flush=True)
    meta, target, series = build_target(members, marginal, window)
    weights = np.asarray(meta["y_weights"], dtype=np.float64)
    mu, a_recorded, _, align = fit_mu(series, target, weights)
    print(
        f"Recorded members: A(0) = {a_recorded:.4f}, "
        f"mu = {mu:.4f} (1/mu = {1.0 / mu:.2f} time units)",
        flush=True,
    )

    configure_jax_platform(params.dist.platform, double_precision=True)
    from dnsjax.twin.diagnostics import flow  # noqa: F401  (builds it)

    if (int(meta["ny"]), _n_k(meta, marginal)) != (
        params.res.ny,
        params.res.nz // 2 if marginal == "x" else params.res.nx // 2,
    ):
        raise SystemExit(
            f"{_PROG}: error: this run's grid does not match the "
            f"recorded stream's (ny {params.res.ny} vs {meta['ny']}); "
            "the two maps would not share axes."
        )

    sweeps = {
        "random_smoothness": _parse_floats(
            calib_params.sweep_smoothness, "calib.sweep_smoothness"
        )
        if calib_params.sweep_smoothness
        else [params.init.random_smoothness],
        "random_wall_smoothness": _parse_floats(
            calib_params.sweep_wall_smoothness,
            "calib.sweep_wall_smoothness",
        )
        if calib_params.sweep_wall_smoothness
        else [params.init.random_wall_smoothness],
        "random_wall_confinement": _parse_floats(
            calib_params.sweep_wall_confinement,
            "calib.sweep_wall_confinement",
        )
        if calib_params.sweep_wall_confinement
        else [params.init.random_wall_confinement],
    }
    print(
        f"\n{'smoothness':>12} {'wall_smooth':>12} {'wall_conf':>13}"
        f" {'A(0)':>8} {'shape lead':>11}",
        flush=True,
    )
    for s_h in sweeps["random_smoothness"]:
        for s_w in sweeps["random_wall_smoothness"]:
            for conf in sweeps["random_wall_confinement"]:
                # Through ``update_parameters``: a direct assignment to
                # a materialized field is overwritten on the next pass
                # (root CLAUDE.md, "Parameter layering").
                params.init.random_smoothness = s_h
                params.init.random_wall_smoothness = s_w
                params.init.random_wall_confinement = conf
                update_parameters(params)
                _, shape = ic_shape(marginal)
                a_new = shape_alignment(shape, target, weights)
                gain = (
                    np.log((1.0 - a_recorded) / (1.0 - a_new)) / mu
                    if np.isfinite(mu) and a_new < 1.0
                    else float("nan")
                )
                print(
                    f"{s_h:12.4g} {s_w:12.4g} {conf:13.4g}"
                    f" {a_new:8.4f} {gain:+10.2f} t",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
