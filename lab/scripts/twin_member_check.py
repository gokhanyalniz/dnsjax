#!/usr/bin/env python3
r"""Check one finished twin member: complete, consistent, long enough.

Reads a ``dnsjax-twin`` member directory, without JAX, and reports in
five sections:

- **Streams.**  Every ``.dat`` stream, ``(y, k)`` stream, cube and
  reduced-snapshot directory holds a sample at every multiple of its
  cadence before the last step, counted in steps from the member's
  parent (``twin.json``'s ``parent_it``, as the driver counts them);
  no ``.bin`` stream ends in a partial record; every value is finite.
  Repeated and off-grid rows (a resume seam, a final row) are
  reported, not refused.
- **Consistency.**  The ``(y, k)`` energy spectra summed over `$y$`
  and `$k$` against ``twin.dat``'s ``E_d`` and ``E_ref``, to that
  file's printed precision (``outs.stats_precision`` significant
  digits), and the two marginals against each other; the
  ``twin_ybudget`` balance summed over `$y$` and `$k$` against
  `$\mathrm{d}E_\Delta/\mathrm{d}t$`, a five-point derivative of
  ``twin.dat``'s ``E_d``, to :data:`CLOSURE_TOL` of the summed term
  magnitudes (Design notes, "Closure tolerance").
- **Solver.**  The corrector's passes and largest error, checked
  against ``step.corrector_tolerance`` when the corrector converges
  rather than runs a fixed count, and each state's largest CFL number.
- **Snapshots.**  The ``state*.tar`` pairs, and a final pair at the
  last step: the resume point and the next parents.  The status file
  and how much of the ``stop.max_sim_time`` horizon the run covered
  are reported.
- **Run length.**  The first time the difference reaches
  `$R = E_\Delta / E_\mathrm{sat} =` :data:`LEVEL` in each band
  (:data:`N_SINGLE` single harmonics per axis, then octaves) and each
  of :data:`WALL_RANGES`, the whole channel first.  `$E_\mathrm{sat}$`
  is twice the reference's band energy without the `$(0, 0)$` mode,
  averaged over the distinct reference instants of this member and of
  every ``--pool`` member
  (:func:`dnsjax_lab.analysis.twin.ensemble.distinct_instants`); the
  `$(0, 0)$` mode is taken off the difference too.  A cell that has
  not reached :data:`LEVEL` by the last sample calls for a longer
  run: resume the member from its final pair with
  ``--stop.max_sim_time`` set to the extension, in the code's time
  unit and counted from the resumed clock (the twin ``README.md``,
  "Trajectory bookkeeping").

Usage::

    uv run python lab/scripts/twin_member_check.py MEMBER \
        --re-tau 510 [--pool OTHER_MEMBER ...]

Pool the ensemble's other members: one member's reference covers only
a few correlation times of the box-scale bands, which sets how well
their saturation level is known (Design notes, "The level").  A pool
member must share the member's grid and box, and holds reference
spectra of its own (``twin_yspectra_ref.bin``, or inside
``twin_yspectra.bin`` for a member recorded before that split).

Times are reported in outer units, `$t\,u_\tau/h$`, with `$t^+$` and
the code's `$h/U_\mathrm{cl}$` beside the run-length verdict, from the
reference flow's time-averaged `$Re_\tau$` (``--re-tau``): a required
input, never measured here (the twin ``README.md``, "Offline
analysis").  Wall distances are in the half-height `$h$`, the unit of
the stored ``y`` in `$[-1, 1]$`.

Exit status: 0 when every check passes and every cell reached
:data:`LEVEL`; 1 when a check fails; 2 when the checks pass but a cell
has not reached :data:`LEVEL` (extend the member).

Design notes
------------
**The level.**  In five plane-Poiseuille members at
`$Re_\tau \approx 510$` in an `$8\pi h \times 2h \times \pi h$` box, run
250 to 500 `$h/U_\mathrm{cl}$`, every band with `$k_x \ge 4$` or
`$k_z \ge 3$` (harmonics) reached `$R = 0.9$` by about 90
`$h/U_\mathrm{cl}$` (3.3 `$h/u_\tau$`), and the box-scale bands
(`$k_x \le 2$`, `$k_z \le 2$`) as late as about 240 (8.7
`$h/u_\tau$`), in one member of the five.  Once saturated, those bands'
`$R$` averaged over 250 `$h/U_\mathrm{cl}$` ranged from 0.8 to 1.3
between members: their reference band-energy autocorrelation falls
below 0.1 at lags of 25 to 65 `$h/U_\mathrm{cl}$`, so a run of that
length holds only a few independent samples of them.  A level near 1,
such as 0.95, is then crossed or missed by those fluctuations, not by
the decorrelation; 0.9 is the level the run length is planned on.  The
check is a flag for a longer run, not a measurement: the crossing
times themselves belong to an ensemble analysis.

**The wall-distance ranges.**  Outer ranges, because the cells that
cross last are the outer, box-scale ones; inner wall distances cross
earlier at every band, and outer ranges need no `$Re_\tau$` to place.

**Closure tolerance.**  On the member the level was set from, the
five-point derivative of the 9-digit ``E_d`` matched the stored
balance to a median of `$1.5 \times 10^{-5}$` of the summed term
magnitudes, and to `$10^{-3}$` at worst, in the first few
`$h/U_\mathrm{cl}$`, where the seed adjusts and `$E_\Delta$` curves
fastest.  A wrong or missing term reads at order one, so
:data:`CLOSURE_TOL` sits a decade above the worst agreement and a
hundredfold below a defect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from dnsjax_lab.analysis.twin.cubes import read_cube
from dnsjax_lab.analysis.twin.ensemble import distinct_instants
from dnsjax_lab.analysis.twin.series import read_dat, relative_time
from dnsjax_lab.analysis.twin.yspectra import (
    BALANCE_PARTS,
    balance_term,
    balance_terms,
    mean_mode_name,
    mean_mode_profile,
    record_dtype,
)

from dnsjax.analysis import read_meta, read_state
from dnsjax.run_status import FINISHED, RUNNING, STOPPED, TERMINATED
from dnsjax.run_status import read_fields as read_status

#: The level of `$R$` the run length is judged at (Design notes, "The
#: level").
LEVEL: float = 0.9

#: Wall-distance ranges, in `$h$`, beside the whole channel (Design
#: notes, "The wall-distance ranges").  The last one is closed.
WALL_RANGES: tuple[tuple[float, float], ...] = (
    (0.0, 0.1),
    (0.1, 0.3),
    (0.3, 0.6),
    (0.6, 1.0),
)

#: Harmonics ``0 .. N_SINGLE - 1`` of each axis are bands of their own,
#: the box-scale ones the run length turns on; octaves follow.
N_SINGLE: int = 4

#: Largest closure residual of the ``twin_ybudget`` balance against
#: `$\mathrm{d}E_\Delta/\mathrm{d}t$`, relative to the summed term
#: magnitudes (Design notes, "Closure tolerance").
CLOSURE_TOL: float = 1e-2

#: Agreement of the two complete marginals' totals, which differ by
#: summation order alone.
MARGINAL_TOL: float = 1e-10

#: The cube and reduced-snapshot directories, if present.
TAR_DIRS: tuple[str, ...] = (
    "twin_spectra3d",
    "twin_spectra3d_ref",
    "twin_budget3d",
    "lowres",
    "lowres_delta",
)


class Report:
    """Prints check and note lines and counts the failed checks."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def section(self, title: str) -> None:
        print(f"\n== {title}")

    def check(self, ok: bool, text: str) -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {text}")
        if not ok:
            self.failures.append(text)

    def note(self, text: str) -> None:
        print(f"       {text}")


class Member:
    """One member directory: its record, clock and ``(y, k)`` streams."""

    def __init__(self, path: Path) -> None:
        self.path = path
        record = path / "twin.json"
        if not record.is_file():
            raise SystemExit(f"error: no twin.json in {path}")
        self.meta = json.loads(record.read_text())
        self.params = self.meta["params"]
        self.dt = float(self.meta["dt"])
        self.parent_t = float(self.meta["parent_t"])
        self.parent_it = int(self.meta["parent_it"])

    def steps(self, t: np.ndarray, where: str) -> np.ndarray:
        """Absolute times *t* as whole steps since the parent."""
        rel = relative_time(
            np.asarray(t, dtype=np.float64), self.meta, 0.0, where
        )
        return np.rint(rel / self.dt).astype(np.int64)

    def stream(self, stem: str):
        """``(sidecar, records, trailing bytes)`` of a ``.bin`` stream."""
        sidecar = json.loads((self.path / f"{stem}.json").read_text())
        dtype = record_dtype(sidecar, stem)
        size = (self.path / f"{stem}.bin").stat().st_size
        n, rest = divmod(size, dtype.itemsize)
        records = np.memmap(
            self.path / f"{stem}.bin", dtype=dtype, mode="r", shape=(n,)
        )
        return sidecar, records, rest

    def reference(self):
        """``(sidecar, records)`` of the reference spectra, either layout."""
        if (self.path / "twin_yspectra_ref.json").is_file():
            sidecar, records, _ = self.stream("twin_yspectra_ref")
            return sidecar, records
        sidecar, records, _ = self.stream("twin_yspectra")
        if not sidecar.get("includes_ref"):
            raise SystemExit(f"error: {self.path} has no reference spectra")
        return sidecar, records


def _grid(
    report: Report, name: str, steps: np.ndarray, cadence: int, last: int
) -> None:
    """Every multiple of *cadence* below *last* present (stream check)."""
    on = steps[steps % cadence == 0]
    missing = np.setdiff1d(np.arange(0, last, cadence), on)
    report.check(
        missing.size == 0,
        f"{name}: {steps.size} samples, every {cadence} steps from "
        f"{steps.min()} to {steps.max()}"
        + (
            f"; {missing.size} missing, the first at step {missing[0]}"
            if missing.size
            else ""
        ),
    )
    repeated = on.size - np.unique(on).size
    off = int((steps % cadence != 0).sum())
    if repeated or off:
        report.note(f"{repeated} repeated and {off} off-grid sample(s)")


def _cadence(recorded, steps: np.ndarray) -> int:
    """The recorded cadence, or the smallest gap when none is recorded."""
    if recorded:
        return int(recorded)
    gaps = np.diff(np.unique(steps))
    return int(gaps.min()) if gaps.size else 1


def check_dat(report: Report, member: Member, last: int) -> dict:
    """The ``.dat`` streams on their cadences, finite; returns them."""
    outs = member.params["outs"]
    plan = {
        "twin.dat": member.meta.get("it_energy"),
        "stats.dat": outs.get("it_stats"),
        "stats_twin.dat": outs.get("it_stats"),
        "steps.dat": outs.get("it_steps"),
        "steps_twin.dat": outs.get("it_steps"),
        "corrector.dat": outs.get("it_corrector"),
        "corrector_twin.dat": outs.get("it_corrector"),
    }
    streams = {}
    for name, cadence in plan.items():
        path = member.path / name
        if not path.is_file():
            if cadence:
                report.check(False, f"{name}: absent")
            continue
        columns = read_dat(path)
        steps = member.steps(columns["t"], str(path))
        _grid(report, name, steps, _cadence(cadence, steps), last)
        bad = [k for k, v in columns.items() if not np.isfinite(v).all()]
        if bad:
            report.check(False, f"{name}: non-finite in {bad}")
        streams[name] = (steps, columns)
    return streams


def check_bin(report: Report, member: Member, last: int) -> None:
    """The ``(y, k)`` streams: whole records on their cadences, finite."""
    for stem, key in (
        ("twin_yspectra", "it_yspectra"),
        ("twin_yspectra_ref", "it_yspectra_ref"),
        ("twin_ybudget", "it_ybudget"),
    ):
        if not (member.path / f"{stem}.bin").is_file():
            report.note(f"{stem}.bin: not recorded")
            continue
        sidecar, records, rest = member.stream(stem)
        report.check(rest == 0, f"{stem}.bin: {rest} trailing byte(s)")
        steps = member.steps(records["t"], stem)
        cadence = _cadence(sidecar.get(key), steps)
        _grid(report, f"{stem}.bin", steps, cadence, last)
        bad = 0
        for i in range(records.shape[0]):
            for name in records.dtype.names:
                bad += int(not np.isfinite(records[name][i]).all())
        report.check(bad == 0, f"{stem}.bin: {bad} non-finite field(s)")


def check_tars(report: Report, member: Member, last: int) -> None:
    """The cube and reduced-snapshot directories, file by file."""
    for kind in TAR_DIRS:
        folder = member.path / kind
        files = sorted(folder.glob(f"{kind}_*.tar")) if folder.is_dir() else []
        if not files:
            report.note(f"{kind}/: not recorded")
            continue
        steps = (
            np.array([int(f.stem.rsplit("_", 1)[1]) for f in files])
            - member.parent_it
        )
        recorded = {
            "lowres": member.params["lowres"].get("it_lowres"),
            "lowres_delta": member.meta.get("it_lowres_delta"),
        }.get(kind, member.meta.get(f"it_{kind.removeprefix('twin_')}"))
        _grid(report, f"{kind}/", steps, _cadence(recorded, steps), last)
        bad = []
        for f in files:
            try:
                if kind.startswith("lowres"):
                    fields = read_state(
                        f, return_physical=False, return_spectral=True
                    ).spectral
                else:
                    fields = tuple(read_cube(f).fields.values())
            except ValueError as error:  # a damaged archive
                bad.append(f"{f.name} ({error})")
                continue
            if not all(np.isfinite(v).all() for v in fields):
                bad.append(f.name)
        report.check(
            not bad,
            f"{kind}/: every file readable and finite"
            + (f"; not: {', '.join(bad[:3])}" if bad else ""),
        )


def _wall_masks(y: np.ndarray) -> list[np.ndarray]:
    """The whole channel, then each of :data:`WALL_RANGES`."""
    if not np.isclose(np.abs(y).max(), 1.0):
        raise SystemExit("error: the stored y does not span [-1, 1]")
    wall = 1.0 - np.abs(y)
    masks = [np.ones(y.size, dtype=bool)]
    for i, (a, b) in enumerate(WALL_RANGES):
        top = wall <= b if i == len(WALL_RANGES) - 1 else wall < b
        masks.append((wall >= a) & top)
    return masks


def _bands(n: int) -> list[np.ndarray]:
    """Single harmonics below :data:`N_SINGLE`, then octaves to *n*."""
    bands = [np.array([m]) for m in range(min(N_SINGLE, n))]
    low = N_SINGLE
    while low < n:
        bands.append(np.arange(low, min(2 * low, n)))
        low *= 2
    return bands


def _band_energies(record, prefix, sidecar, masks, bands) -> np.ndarray:
    """One record's energy per wall range and band, `$(0, 0)$` off.

    Returns ``(2, n_ranges, n_bands)``: the `$k_x$` bands (from the
    `$k_z$`-summed ``*_z`` marginal) then the `$k_z$` bands (from
    ``*_x``), each padded to the longer band list with ``nan``.
    """
    w = np.asarray(sidecar["y_weights"], dtype=np.float64)
    name = mean_mode_name(sidecar, prefix)
    mean = mean_mode_profile(
        np.asarray(record[name], dtype=np.float64), name
    ).sum(axis=0)
    out = np.full((2, len(masks), max(len(b) for b in bands)), np.nan)
    for a, suffix in enumerate(("z", "x")):
        plane = np.asarray(record[f"{prefix}_{suffix}"], dtype=np.float64)
        plane = plane.sum(axis=0)
        plane[:, 0] -= mean
        for j, mask in enumerate(masks):
            per_k = w[mask] @ plane[mask]
            for b, band in enumerate(bands[a]):
                out[a, j, b] = per_k[band].sum()
    return out


def _first_passage(t: np.ndarray, r: np.ndarray, level: float) -> float:
    """The first upward crossing of *level*, linear in time; ``nan``."""
    above = np.flatnonzero(r >= level)
    if above.size == 0:
        return float("nan")
    i = int(above[0])
    if i == 0:
        return float(t[0])
    return float(
        t[i - 1] + (level - r[i - 1]) * (t[i] - t[i - 1]) / (r[i] - r[i - 1])
    )


def run_length(member: Member, pool: list[Member], re_tau: float) -> bool:
    """The run-length table and verdict; ``True`` when every cell crossed."""
    sidecar, records, _ = member.stream("twin_yspectra")
    y = np.asarray(sidecar["y"], dtype=np.float64)
    masks = _wall_masks(y)
    bands = (_bands(int(sidecar["n_kx"])), _bands(int(sidecar["n_kz"])))
    grid_keys = ("y", "y_weights", "kz_harmonics", "kx_harmonics", "lx", "lz")

    # The saturation level over the members' distinct reference instants.
    members = [member, *pool]
    references = [m.reference() for m in members]
    for m, (ref_sidecar, _) in zip(members, references, strict=True):
        if any(ref_sidecar[k] != sidecar[k] for k in grid_keys):
            raise SystemExit(f"error: {m.path} is not on the member's grid")
    picks, n_instants, _ = distinct_instants(
        [np.asarray(r["t"], dtype=np.float64) for _, r in references],
        [m.parent_t for m in members],
        [str(m.path) for m in members],
    )
    total = 0.0
    span = [np.inf, -np.inf]
    for (ref_sidecar, ref), rows in zip(references, picks, strict=True):
        for i in rows:
            total = total + _band_energies(
                ref[i], "r", ref_sidecar, masks, bands
            )
            span = [min(span[0], ref["t"][i]), max(span[1], ref["t"][i])]
    e_sat = 2.0 * total / n_instants

    steps = member.steps(records["t"], "twin_yspectra")
    t = steps * member.dt
    energy = np.stack(
        [
            _band_energies(records[i], "e", sidecar, masks, bands)
            for i in range(records.shape[0])
        ]
    )
    with np.errstate(invalid="ignore"):
        crossing = np.apply_along_axis(
            lambda r: _first_passage(t, r, LEVEL), 0, energy / e_sat
        )

    outer = re_tau / float(member.params["phys"]["re"])
    print(
        f"  E_sat: {n_instants} distinct reference instants of "
        f"{len(members)} member(s), t = {span[0]:.2f} .. {span[1]:.2f} "
        f"({(span[1] - span[0]) * outer:.2f} h/u_tau)"
    )
    print(
        f"  First time R = E_d / E_sat reaches {LEVEL:g}, in t u_tau/h "
        f"('-': not by t u_tau/h = {t[-1] * outer:.2f}):"
    )
    heads = ["all y"] + [f"{a:g}-{b:g}h" for a, b in WALL_RANGES]
    print("  " + f"{'band':36s}" + "".join(f"{h:>9s}" for h in heads))
    lengths = (float(sidecar["lx"]), float(sidecar["lz"]))
    unreached: list[str] = []
    latest = (-np.inf, "")
    for a, axis in enumerate(("k_x", "k_z")):
        for b, band in enumerate(bands[a]):
            lo, hi = int(band[0]), int(band[-1])
            name = f"{axis} {lo}" if lo == hi else f"{axis} {lo}-{hi}"
            if lo == hi:
                lam = "inf" if lo == 0 else f"{lengths[a] / lo:.3g}"
            else:
                lam = f"{lengths[a] / hi:.2g}-{lengths[a] / lo:.2g}"
            cells = []
            for j, head in enumerate(heads):
                tc = crossing[a, j, b]
                if np.isnan(tc):
                    cells.append(f"{'-':>9s}")
                    unreached.append(f"{name}, {head}")
                else:
                    cells.append(f"{tc * outer:9.2f}")
                    if tc > latest[0]:
                        latest = (tc, f"{name}, {head}")
            label = f"{name} (lambda/h {lam})"
            print("  " + f"{label:36s}" + "".join(cells))

    plus = re_tau * outer
    if not unreached:
        tc, where = latest
        print(
            f"  Every cell reached R = {LEVEL:g}; the last at t u_tau/h = "
            f"{tc * outer:.2f} (t+ = {tc * plus:.0f}, {tc:.1f} h/U_cl): "
            f"{where}."
        )
        return True
    print(
        f"  {len(unreached)} cell(s) below R = {LEVEL:g} at the last "
        f"sample: {'; '.join(unreached)}."
    )
    print(
        "  Extend the member: resume it from its final pair with "
        "--stop.max_sim_time set to the extension, in h/U_cl "
        f"(1 h/U_cl = {outer:.4f} h/u_tau)."
    )
    return False


def check_consistency(report: Report, member: Member, dat: dict) -> None:
    """The spectra against ``twin.dat``, and the budget's closure."""
    steps_e, twin = dat["twin.dat"]
    digits = int(member.params["outs"]["stats_precision"])
    tol = 10.0 ** (1 - digits)
    index = {int(s): i for i, s in enumerate(steps_e)}
    for stem, prefix, column in (
        ("twin_yspectra", "e", "E_d"),
        ("twin_yspectra_ref", "r", "E_ref"),
    ):
        if not (member.path / f"{stem}.bin").is_file():
            continue
        sidecar, records, _ = member.stream(stem)
        w = np.asarray(sidecar["y_weights"], dtype=np.float64)
        steps = member.steps(records["t"], stem)
        worst, worst_marginals, n = 0.0, 0.0, 0
        for i, s in enumerate(steps):
            if int(s) not in index:
                continue
            by_x = np.einsum("cjk,j->", records[f"{prefix}_x"][i], w)
            by_z = np.einsum("cjk,j->", records[f"{prefix}_z"][i], w)
            target = twin[column][index[int(s)]]
            worst = max(worst, abs(by_x / target - 1.0))
            worst_marginals = max(worst_marginals, abs(by_z / by_x - 1.0))
            n += 1
        report.check(
            n > 0 and worst <= tol,
            f"{stem}.bin: summed spectra = twin.dat {column} to "
            f"{worst:.1e} over {n} samples (tolerance {tol:.0e})",
        )
        report.check(
            worst_marginals <= MARGINAL_TOL,
            f"{stem}.bin: the two marginals' totals agree to "
            f"{worst_marginals:.1e}",
        )

    if not (member.path / "twin_ybudget.bin").is_file():
        return
    sidecar, records, _ = member.stream("twin_ybudget")
    try:
        terms = [t for t in balance_terms(sidecar) if t not in BALANCE_PARTS]
    except ValueError:
        report.note("twin_ybudget.bin: rotational form, closure not checked")
        return
    w = np.asarray(sidecar["y_weights"], dtype=np.float64)
    h = int(member.meta["it_energy"])
    energy = twin["E_d"]
    residuals, where = [], []
    for i, s in enumerate(member.steps(records["t"], "twin_ybudget")):
        near = [int(s) + d * h for d in (-2, -1, 1, 2)]
        if not all(k in index for k in near):
            continue
        e = [energy[index[k]] for k in near]
        record = records[i]
        rate = (e[0] - 8.0 * e[1] + 8.0 * e[2] - e[3]) / (12.0 * h * member.dt)
        parts = [
            float(
                np.einsum(
                    "jk,j->",
                    balance_term(record.__getitem__, sidecar, term, "x"),
                    w,
                )
            )
            for term in terms
        ]
        residuals.append(abs(sum(parts) - rate) / sum(abs(p) for p in parts))
        where.append(int(s))
    if not residuals:
        report.note("twin_ybudget.bin: no record with twin.dat neighbours")
        return
    res = np.asarray(residuals)
    report.check(
        res.max() <= CLOSURE_TOL,
        f"twin_ybudget.bin: balance = dE_d/dt to median {np.median(res):.1e}"
        f", worst {res.max():.1e} at step {where[int(res.argmax())]}, over "
        f"{res.size} samples (tolerance {CLOSURE_TOL:.0e})",
    )


def check_solver(report: Report, member: Member, dat: dict) -> None:
    """Corrector passes and error, and each state's largest CFL."""
    step = member.params["step"]
    converging = not step.get("corrector_iterations")
    for name in ("corrector.dat", "corrector_twin.dat"):
        if name in dat:
            columns = dat[name][1]
            text = (
                f"{name}: at most {columns['c'].max():.0f} correction(s) "
                f"past the first pass, error at most "
                f"{columns['error'].max():.2e}"
            )
            if converging:
                tol = float(step["corrector_tolerance"])
                report.check(
                    columns["error"].max() < tol,
                    text + f" (tolerance {tol:g})",
                )
            else:
                report.note(text + " (a fixed count of passes)")
    for name in ("steps.dat", "steps_twin.dat"):
        if name in dat:
            report.note(f"{name}: largest CFL {dat[name][1]['CFL'].max():.3f}")


def check_snapshots(report: Report, member: Member, last: int) -> None:
    """The status file, the horizon covered, and the snapshot pairs."""
    for status in (FINISHED, STOPPED, TERMINATED, RUNNING):
        path = member.path / status
        if path.is_file():
            fields = read_status(path)
            report.note(
                f"status {status}: {fields.get('reason', '')}"
                + (f" (exit {fields['exit']})" if "exit" in fields else "")
            )
            break
    else:
        report.note("no status file")
    horizon = member.params["stop"].get("max_sim_time")
    if horizon:
        report.note(
            f"covered {last * member.dt:g} of the {float(horizon):g} "
            "stop.max_sim_time horizon (h/U_cl, from the run's own start)"
        )
    pairs = []
    for ref in sorted(member.path.glob("state*.tar")):
        if ref.stem.endswith("_twin"):
            continue
        partner = ref.with_name(f"{ref.stem}_twin.tar")
        if not partner.is_file():
            report.note(f"{ref.name}: no partner, not a pair")
            continue
        its = {int(read_meta(f)["it"]) for f in (ref, partner)}
        report.check(len(its) == 1, f"{ref.name} pair: one step, {its}")
        pairs.append((its.pop() - member.parent_it, ref.name))
    if pairs:
        report.note(
            "pairs at steps " + ", ".join(f"{s} ({n})" for s, n in pairs)
        )
    report.check(
        bool(pairs) and max(pairs)[0] == last,
        f"final pair at the last step, {last}"
        + ("" if pairs else ": no state*.tar pair in the directory"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="twin_member_check.py",
        description=__doc__.split("\n\n")[0],
        allow_abbrev=False,
    )
    parser.add_argument("member", type=Path, help="the member directory")
    parser.add_argument(
        "--re-tau",
        type=float,
        required=True,
        help="the reference flow's time-averaged Re_tau (sets u_tau)",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        nargs="+",
        default=[],
        help="other members whose reference spectra join the saturation level",
    )
    args = parser.parse_args(argv)
    member = Member(args.member)
    pool = [Member(p) for p in args.pool]
    report = Report()

    report.section(f"Streams of {member.path}")
    twin = read_dat(member.path / "twin.dat")
    last = int(member.steps(twin["t"], "twin.dat").max())
    dat = check_dat(report, member, last)
    check_bin(report, member, last)
    check_tars(report, member, last)

    report.section("Consistency")
    check_consistency(report, member, dat)

    report.section("Solver")
    check_solver(report, member, dat)

    report.section("Snapshots")
    check_snapshots(report, member, last)

    report.section("Run length")
    crossed = run_length(member, pool, args.re_tau)

    print()
    if report.failures:
        print(f"{len(report.failures)} check(s) failed.")
        return 1
    print("Every check passed" + ("." if crossed else "; extend the member."))
    return 0 if crossed else 2


if __name__ == "__main__":
    raise SystemExit(main())
