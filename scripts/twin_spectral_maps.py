r"""Premultiplied `$(\lambda, y)$` maps of the twin `$(y, k)$` streams.

Renders one figure per recorded sample of ``twin_yspectra.bin`` and
``twin_ybudget.bin`` (:mod:`dnsjax.twin.yspectra`), ensemble-averaged
over a set of ``dnsjax-twin`` member directories, in the style of the
premultiplied spectral maps of Cho, Hwang & Choi, *J. Fluid Mech.*
**854**, 474-504 (2018) -- their figures 3 and 11: wavelength on a
logarithmic abscissa, wall-normal position on a logarithmic ordinate,
filled contours plus contour lines, inner units on the primary axes
and outer units on the secondary ones.

Both wavenumber marginals are drawn (``_z`` gives `$\lambda_x$`, the
paper shows only `$\lambda_z$`), plus the stored `$k_x = 0$` plane.

What "premultiplied" means here
===============================
A stored entry is the energy (or rate) held by one **discrete** mode
band, not a spectral density: summing the entries over the stored
one-sided axis and contracting with ``y_weights`` returns the volume
average.  The density is therefore ``entry / dk`` with
`$\Delta k = 2\pi/L$`, and since `$k_m = m\,\Delta k$` for the integer
harmonic `$m$` the premultiplied spectrum is

.. math::  k\,\Phi(y, k) = m \times \text{entry}(y, m) ,

independent of the box length.  The `$m = 0$` column carries no
premultiplied content and is dropped, which is also what makes the
maps read as fluctuation spectra: the wall-parallel mean of a state
lives at `$(0, 0)$` alone, so at every plotted `$m \ge 1$` the stored
perturbation-about-laminar spectrum ``r_*`` *is* the spectrum of the
fluctuation about the `$x$`-`$z$` mean.

Stored entries are additionally divided by ``volume_fac`` (the
channel's wall-normal extent) so that the plain `$y$`-average of a
profile is the volume average.  Multiplying it back gives the local
density, which is the quantity the paper plots; that is
:attr:`MapOptions.volume_fac` and it is on by default.  What fixes
that factor empirically is the ``r_*`` half of the spectra stream: at
plane-Poiseuille `$Re = 4200$`, `$Re_\tau = 178.6$`, it puts
`$\max k_z E^{x+}_{uu} \approx 3.8$` at `$y^+ \approx 14$`,
`$\lambda_z^+ \approx 130$` -- the textbook near-wall peak.  Without
it every map is a factor of two low.

Inner units
===========
With `$h = U_\mathrm{cl} = 1$` in the code's non-dimensionalisation,
`$\nu = 1/Re$` and `$u_\tau = Re_\tau/Re$`, so

.. math::
    \lambda^+ = \lambda\,Re_\tau, \quad y^+ = (1 - |y|)\,Re_\tau,
    \quad t^+ = t\,Re_\tau^2/Re, \quad
    E^+ = E\,Re^2/Re_\tau^2, \quad
    \mathcal{B}^+ = \mathcal{B}\,Re^3/Re_\tau^4 ,

the last two for an energy and for a budget rate (`$u_\tau^2$` and
`$u_\tau^4/\nu$`).  ``Re_tau`` is a **measured** input, never derived
here.

Usage
=====
matplotlib is not a solver dependency; it lives in the ``plots``
dependency group::

    uv run --group plots python scripts/twin_spectral_maps.py \
        --members RUN1 RUN2 --out FIGDIR \
        --re 4200 --re-tau 178.62135279727977 --stride 10

Every number above is a knob; ``--help`` lists the rest (contour
levels, colour maps, figure width, which channel half, outer
units, filename padding, frame and series selection).

As a library (a notebook on the cluster, one stream at a time)::

    from twin_spectral_maps import (
        MapOptions, Units, draw_map, make_map, open_series)

    s = open_series(["twin1", "twin2"], "twin_ybudget", stride=10)
    opts = MapOptions(Units(re=4200.0, re_tau=178.62135279727977))
    m = make_map(s, "P_U_x", frame=24, options=opts)
    draw_map(ax, m)

:func:`open_series` memory-maps each member and reads only the
selected records, so a single stream costs megabytes rather than the
gigabyte the eager reader
:func:`dnsjax.analysis.twin.yspectra.read_twin_yspectra` would pull
in; the record layout, the format-version floors and the
duplicate-``t`` policy are that reader's, mirrored here.  Members are
aligned on **relative** time `$t - t_0$`, as
:func:`dnsjax.analysis.twin.ensemble.aggregate_members` does, because
members start from different parent snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dnsjax.analysis.twin.yspectra import (
    MIN_YBUDGET_VERSION,
    MIN_YSPECTRA_VERSION,
)

#: Decimals used to key relative times when intersecting members.
#: The cadence is ``it_* * dt`` (order 1 here) and the stored times
#: carry accumulated round-off of order `$10^{-9}$`; the repo reader's
#: alignment tolerance is the same order.
_T_DECIMALS: int = 6

#: The two streams this script understands, with their reader floors.
STEMS: dict[str, int] = {
    "twin_yspectra": MIN_YSPECTRA_VERSION,
    "twin_ybudget": MIN_YBUDGET_VERSION,
}

#: Velocity components of the ``twin_yspectra`` leading axis.
COMPONENTS: tuple[str, ...] = ("u", "v", "w")

#: Budget terms excluded from the ``sum`` virtual field: ``eps`` is
#: the pseudo-dissipation companion of ``V`` (not a separate sink) and
#: ``P_lift`` sits outside the sum by construction.  What is left adds
#: up to `$\partial_t \hat e(y, k)$` -- see "Two budget forms" in
#: :mod:`dnsjax.twin.diagnostics`.
NON_ADDITIVE_TERMS: frozenset[str] = frozenset({"eps", "P_lift"})

#: Panel labels for the budget terms, matching the appendix notation.
TERM_LABELS: dict[str, str] = {
    "P_U": r"\mathcal{P}^{U}",
    "P_r": r"\mathcal{P}^{r}",
    "T_ref": r"\mathcal{T}^{\mathrm{ref}}",
    "T_self": r"\mathcal{T}^{\mathrm{self}}",
    "T_vort": r"\mathcal{T}^{\mathrm{vort}}",
    "V": r"\mathcal{V}",
    "eps": r"\hat{\varepsilon}",
    "Wp": r"\mathcal{W}",
    "P_lift": r"\mathcal{P}^{\mathrm{lift}}",
    "sum": r"\partial_t\hat{e}",
}

#: ``(wavenumber symbol, energy superscript)`` per stored suffix.
MARGINALS: dict[str, tuple[str, str]] = {
    "x": ("k_z", "x"),
    "z": ("k_x", "z"),
    "x0": ("k_z", "x0"),
}

#: LaTeX preamble matching the ``perturbation_dynamics`` write-up.
LATEX_PREAMBLE: str = r"""
\usepackage[p]{stickstootext}
\usepackage[scaled=1.05,stix2,vvarbb]{newtxmath}
\usepackage[defaultsans,proportional,scale=0.955]{lato}
"""

#: Text width of that document, in inches (its ``\linewidth``).
PAGE_LINEWIDTH: float = 6.61546

#: Title offset (points) that clears a secondary abscissa's own label,
#: and colour-bar offset (inches) that clears a secondary ordinate's.
_TITLE_PAD: float = 12.0
_CBAR_PAD: float = 0.72

#: Upper bound on the number of labelled colour-bar ticks.
_BAR_TICKS: int = 6

#: Inches a panel's decorations take beside and above/below its axes
#: box (ordinate label + colour bar + secondary ordinate; title +
#: secondary abscissa + abscissa label), and the figure's suptitle.
_COL_OVERHEAD: float = 2.00
_ROW_OVERHEAD: float = 1.35
_SUP_OVERHEAD: float = 0.45


# ── Units ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Units:
    r"""Outer `$\to$` inner unit conversions for one measured flow.

    *re* is the code's ``phys.re`` and *re_tau* the **measured**
    friction Reynolds number; nothing here re-measures it.  With
    *wall* false every conversion is the identity and the labels
    revert to outer units (`$h$`, `$U_\mathrm{cl}$`).
    """

    re: float
    re_tau: float
    wall: bool = True

    @property
    def u_tau(self) -> float:
        r"""`$u_\tau/U_\mathrm{cl} = Re_\tau/Re$`."""
        return self.re_tau / self.re

    def length(self, values):
        r"""`$\lambda \to \lambda^+ = \lambda\,Re_\tau$`."""
        return values * self.re_tau if self.wall else values

    def time(self, t: float) -> float:
        r"""`$t \to t^+ = t\,Re_\tau^2/Re$`."""
        return t * self.re_tau**2 / self.re

    def energy(self, values):
        r"""`$E \to E/u_\tau^2$`."""
        return values / self.u_tau**2 if self.wall else values

    def rate(self, values):
        r"""`$\mathcal{B} \to \mathcal{B}\,\nu/u_\tau^4$`."""
        if not self.wall:
            return values
        return values / (self.u_tau**4 * self.re)

    def convert(self, values, kind: str):
        """Dispatch :meth:`energy` / :meth:`rate` on *kind*."""
        return self.energy(values) if kind == "energy" else self.rate(values)

    @property
    def lambda_label(self) -> str:
        """Abscissa label (primary axis)."""
        return r"$\lambda^+$" if self.wall else r"$\lambda/h$"

    @property
    def y_label(self) -> str:
        """Ordinate label (primary axis)."""
        return r"$y^+$" if self.wall else r"$y/h$"

    def norm_suffix(self, kind: str) -> str:
        """Normalisation appended to a panel title."""
        if not self.wall:
            return ""
        if kind == "energy":
            return r"/u_\tau^2"
        return r"\,\nu/u_\tau^4"


# ── Stream reading ───────────────────────────────────────────────────


def _record_dtype(meta: dict, stem: str) -> np.dtype:
    """The stream's fixed-size record layout, from its sidecar.

    Mirrors the field table of
    :func:`dnsjax.analysis.twin.yspectra.read_twin_yspectra` /
    :func:`~dnsjax.analysis.twin.yspectra.read_twin_ybudget`; the
    dtype is what lets a record be read without the eager reader's
    whole-file pass.
    """
    ny = int(meta["ny"])
    n_kz, n_kx = int(meta["n_kz"]), int(meta["n_kx"])
    per_suffix = (("x", n_kz), ("z", n_kx), ("x0", n_kz))
    value_dtype = meta["value_dtype"]
    if stem == "twin_yspectra":
        prefixes = ("e", "r") if bool(meta["includes_ref"]) else ("e",)
        names = [
            (f"{p}_{suf}", (len(COMPONENTS), ny, n))
            for p in prefixes
            for suf, n in per_suffix
        ]
    else:
        names = [
            (f"{term}_{suf}", (ny, n))
            for term in meta["terms"]
            for suf, n in per_suffix
        ]
    return np.dtype([("t", "<f8")] + [(n, value_dtype, sh) for n, sh in names])


@dataclass(frozen=True)
class _Member:
    """One member's memory-mapped stream, deduplicated in time."""

    path: Path
    meta: dict
    records: np.memmap
    rows: np.ndarray  # record indices, ascending, unique in t
    t_rel: np.ndarray  # their times relative to the member's first


def _open_member(path: Path, stem: str) -> _Member:
    """Memory-map one member and resolve its usable records."""
    bin_path, json_path = path / f"{stem}.bin", path / f"{stem}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"no sidecar {json_path}")
    with open(json_path) as fh:
        meta = json.load(fh)
    version = int(meta.get("format_version", 0))
    if version < STEMS[stem]:
        raise ValueError(
            f"{json_path}: format_version {version} predates the "
            f"reader floor {STEMS[stem]}."
        )
    dtype = _record_dtype(meta, stem)
    # A kill mid-write leaves a partial trailing record; the complete
    # prefix is intact (append-only, fsync per flush).
    n_records = bin_path.stat().st_size // dtype.itemsize
    if n_records == 0:
        raise ValueError(f"{bin_path}: no complete records")
    records = np.memmap(bin_path, dtype=dtype, mode="r", shape=(n_records,))
    t = np.asarray(records["t"], dtype=np.float64)
    # Resume-by-append can repeat a seam row: keep its first copy,
    # the eager reader's policy.
    rows = np.sort(np.unique(t, return_index=True)[1])
    return _Member(path, meta, records, rows, t[rows] - t[rows[0]])


@dataclass
class YSeries:
    r"""An ensemble-averaged, subsampled `$(y, k)$` stream.

    Fields are read and averaged on demand (and cached), so opening a
    series is cheap however long the run.  ``index`` carries each
    frame's record number on the members' common relative-time grid --
    the number the output filenames use.
    """

    stem: str
    members: tuple[_Member, ...]
    rows: np.ndarray  # (n_members, n_frames) record index per member
    index: np.ndarray  # (n_frames,) position on the common grid
    t_rel: np.ndarray  # (n_frames,) relative time
    meta: dict  # the first member's sidecar
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def y(self) -> np.ndarray:
        """Wall-normal grid, `$y \\in [-1, 1]$`."""
        return np.asarray(self.meta["y"], dtype=np.float64)

    @property
    def y_weights(self) -> np.ndarray:
        """Its quadrature weights (they sum to ``volume_fac``)."""
        return np.asarray(self.meta["y_weights"], dtype=np.float64)

    @property
    def volume_fac(self) -> float:
        """The wall-normal extent every stored entry is divided by."""
        return float(self.meta["volume_fac"])

    def harmonics(self, marginal: str) -> np.ndarray:
        """Integer harmonics of a marginal's wavenumber axis."""
        key = "kx_harmonics" if marginal == "z" else "kz_harmonics"
        return np.asarray(self.meta[key], dtype=np.float64)

    def wavelengths(self, marginal: str) -> np.ndarray:
        r"""`$\lambda = L/m$` for `$m \ge 1$`, in outer units.

        Ascending, i.e. reversed against the harmonic order, which is
        the order :func:`make_map` puts the wavenumber axis in.
        """
        length = float(self.meta["lx" if marginal == "z" else "lz"])
        return (length / self.harmonics(marginal)[1:])[::-1]

    @property
    def terms(self) -> tuple[str, ...]:
        """Budget term names (``twin_ybudget`` only)."""
        return tuple(self.meta.get("terms", ()))

    @property
    def prefixes(self) -> tuple[str, ...]:
        """Spectra prefixes present (``twin_yspectra`` only)."""
        if self.stem != "twin_yspectra":
            return ()
        return ("e", "r") if bool(self.meta["includes_ref"]) else ("e",)

    def field(self, name: str) -> np.ndarray:
        """Ensemble mean of one field over the selected frames.

        Shape ``(n_frames, 3, n_y, n_k)`` for the spectra and
        ``(n_frames, n_y, n_k)`` for the budget.  The virtual name
        ``sum_<suffix>`` adds the budget terms that make up
        `$\\partial_t \\hat e$` (:data:`NON_ADDITIVE_TERMS`).
        """
        if name in self._cache:
            return self._cache[name]
        base, _, suffix = name.rpartition("_")
        if base == "sum":
            parts = [
                self.field(f"{term}_{suffix}")
                for term in self.terms
                if term not in NON_ADDITIVE_TERMS
            ]
            if not parts:
                raise ValueError(f"{self.stem}: no additive budget terms")
            value = np.sum(parts, axis=0)
        else:
            total = None
            for member, rows in zip(self.members, self.rows, strict=True):
                block = np.asarray(
                    member.records[name][rows], dtype=np.float64
                )
                total = block if total is None else total + block
            value = total / len(self.members)
        self._cache[name] = value
        return value


def _locate(key: np.ndarray, wanted: np.ndarray, path: Path) -> np.ndarray:
    """Positions of *wanted* in the ascending *key*, verified.

    ``searchsorted`` is silent on a key that is not sorted, and an
    append-only stream that was somehow written out of order would
    then pair a frame with the wrong record; the equality check is
    what turns that into an error.
    """
    where = np.searchsorted(key, wanted)
    if where.size and (
        where.max() >= key.size or not np.array_equal(key[where], wanted)
    ):
        raise ValueError(f"{path}: sample times are not sorted ascending")
    return where


def open_series(
    members: list[str | Path],
    stem: str,
    *,
    stride: int = 1,
    first: int = 0,
    last: int | None = None,
) -> YSeries:
    """Open one stream across *members* on their common time grid.

    *stride* keeps every *stride*-th record (the "subsample by 10" of
    a long run), *first* / *last* clip the common grid before that.
    Members are aligned on relative time `$t - t_0$`: they start from
    different parent snapshots, so absolute times do not match.
    """
    if stem not in STEMS:
        raise ValueError(f"unknown stream {stem!r}; expected {set(STEMS)}")
    if not members:
        raise ValueError("need at least one member directory")
    opened = [_open_member(Path(m), stem) for m in members]

    keys = [np.round(m.t_rel, _T_DECIMALS) for m in opened]
    common = keys[0]
    for other in keys[1:]:
        common = np.intersect1d(common, other, assume_unique=False)
    if common.size == 0:
        raise ValueError("members share no relative sample time")
    common = common[first : (None if last is None else last + 1)][::stride]
    rows = np.stack(
        [
            member.rows[_locate(key, common, member.path)]
            for member, key in zip(opened, keys, strict=True)
        ]
    )
    index = _locate(keys[0], common, opened[0].path)
    return YSeries(
        stem=stem,
        members=tuple(opened),
        rows=rows,
        index=index,
        t_rel=common,
        meta=opened[0].meta,
    )


# ── Map construction ─────────────────────────────────────────────────


@dataclass(frozen=True)
class MapOptions:
    r"""Everything that turns a stored field into a plotted map.

    *half* picks how the two channel halves collapse onto the single
    wall distance a `$y^+$` ordinate needs -- ``"mean"`` averages them
    (the grid is symmetric, so this is an arithmetic mean at matching
    `$y^+$`), ``"lower"`` / ``"upper"`` keep one wall, which is how a
    run's own asymmetry is inspected.  *volume_fac* multiplies the
    stored `$y$`-mean density back to the local density the paper
    plots; *premultiply* applies the `$m \times$` factor of the module
    docstring.

    *smooth* is a centred running mean over that many adjacent
    wavenumbers, applied last.  It is off (``1``) by default and is
    presentation only: a two-member ensemble mean of an *instantaneous*
    field is genuinely rough mode to mode -- unlike the paper's
    long-time averages of a stationary flow -- and the transfer terms
    show it.  Widening this bins the map; it does not denoise it.
    """

    units: Units
    half: str = "mean"
    volume_fac: bool = True
    premultiply: bool = True
    smooth: int = 1


@dataclass(frozen=True)
class Map:
    r"""One panel's data: ``values`` on the `$(\lambda, y)$` grid."""

    lam: np.ndarray  # (n_lam,) wavelength, plotted units
    y: np.ndarray  # (n_y,) wall distance, plotted units
    values: np.ndarray  # (n_y, n_lam)
    title: str  # LaTeX panel title, with normalisation
    name: str  # the stored field it came from


def _select_half(
    values: np.ndarray, y: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the channel onto one wall distance (:class:`MapOptions`).

    The wall-normal grid is symmetric about the centreline, so index
    ``j`` pairs with ``n_y - 1 - j`` and, for odd ``n_y``, the
    mid-plane with itself -- which is why ``"mean"`` needs no special
    case there.  Returns the collapsed values and the wall distance
    `$1 - |y|$` of the retained rows, ascending from the wall.
    """
    n_half = (y.size + 1) // 2
    lower = values[..., :n_half, :]
    upper = values[..., ::-1, :][..., :n_half, :]
    if mode == "mean":
        kept = 0.5 * (lower + upper)
    elif mode == "lower":
        kept = lower
    elif mode == "upper":
        kept = upper
    else:
        raise ValueError(f"half must be mean/lower/upper, not {mode!r}")
    return kept, 1.0 + y[:n_half]


def _running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred running mean over the trailing (wavenumber) axis.

    The window is truncated at both ends rather than padded, so no
    value is invented at the first and last wavenumbers, and an even
    *window* is widened by one: a mean over an even number of
    neighbours has no centre, and an off-centre one would shift every
    map half a wavenumber.
    """
    if window <= 1:
        return values
    window += 1 - window % 2
    kernel = np.ones(window) / window
    ones = np.ones(values.shape[-1])
    norm = np.convolve(ones, kernel, mode="same")
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), -1, values
    )
    return smoothed / norm


def field_title(
    series: YSeries,
    name: str,
    component: int | None,
    options: MapOptions,
) -> str:
    """The LaTeX panel title for one stored (or virtual) field."""
    base, _, suffix = name.rpartition("_")
    wavenumber, superscript = MARGINALS[suffix]
    kind = "energy" if series.stem == "twin_yspectra" else "rate"
    factor = f"{wavenumber}\\," if options.premultiply else ""
    if kind == "rate":
        body = TERM_LABELS.get(base, base.replace("_", r"\_"))
    else:
        delta = r"\Delta " if base == "e" else ""
        if component is None:
            body = (
                rf"\sum_\alpha E^{{{superscript}}}_{{{delta}\alpha}}"
                if base == "e"
                else rf"\sum_\alpha E^{{{superscript}}}_{{\alpha}}"
            )
        else:
            sub = f"{delta}{COMPONENTS[component]}"
            body = rf"E^{{{superscript}}}_{{{sub}}}"
    return f"${factor}{body}{options.units.norm_suffix(kind)}$"


def make_map(
    series: YSeries,
    name: str,
    frame: int,
    *,
    options: MapOptions,
    component: int | None = None,
) -> Map:
    """Build one premultiplied map from a stored (or virtual) field.

    *name* is a stored field such as ``e_x`` / ``P_U_z``, or the
    virtual ``sum_x``; *component* selects a velocity component of a
    ``twin_yspectra`` field (``None`` sums the three).  *frame* indexes
    the series' subsampled records.
    """
    suffix = name.rpartition("_")[2]
    values = series.field(name)[frame]
    if series.stem == "twin_yspectra":
        values = values.sum(axis=0) if component is None else values[component]
    elif component is not None:
        raise ValueError(f"{series.stem}: {name} has no component axis")

    harmonics = series.harmonics(suffix)
    values = values[:, 1:]  # m = 0 carries no premultiplied content
    if options.premultiply:
        values = values * harmonics[None, 1:]
    if options.volume_fac:
        values = values * series.volume_fac
    kind = "energy" if series.stem == "twin_yspectra" else "rate"
    values = options.units.convert(values, kind)
    values = values[:, ::-1]  # ascending in wavelength, as the axis is

    values, wall_distance = _select_half(values, series.y, options.half)
    values = _running_mean(values, options.smooth)
    return Map(
        lam=options.units.length(series.wavelengths(suffix)),
        y=options.units.length(wall_distance),
        values=values,
        title=field_title(series, name, component, options),
        name=name,
    )


# ── Drawing ──────────────────────────────────────────────────────────


def _bar_ticks(levels: np.ndarray) -> np.ndarray:
    """At most :data:`_BAR_TICKS` of the contour levels, zero kept."""
    every = max(1, math.ceil(levels.size / _BAR_TICKS))
    if every == 1:
        return levels
    zero = int(np.argmin(np.abs(levels)))
    return levels[zero % every :: every]


def nice_step(raw: float) -> float:
    """Round a level spacing up to the next 1 / 2 / 2.5 / 5 decade step.

    Levels on round numbers are what makes a colour bar readable, and
    rounding *up* keeps the level count at or below the request.
    """
    exponent = math.floor(math.log10(raw))
    mantissa = raw / 10.0**exponent
    for candidate in (1.0, 2.0, 2.5, 5.0):
        if mantissa <= candidate:
            return candidate * 10.0**exponent
    return 10.0 ** (exponent + 1)


def contour_levels(
    values: np.ndarray,
    n_levels: int,
    quantile: float | None = None,
    nice: bool = True,
) -> tuple[np.ndarray, bool, float]:
    """Contour levels for one map, and whether it is non-negative.

    A non-negative field gets bands from one step up to the peak,
    leaving the lowest band unfilled so the empty corners of the map
    stay white (the paper's convention).  A signed field gets the same
    step reflected about zero and trimmed to the data range, with a
    colour scale symmetric about zero so that zero is the neutral
    colour whatever the trim.

    The zero level itself is **dropped**, which is the signed
    counterpart of that unfilled lowest band: an instantaneous
    difference-field budget is tiny and sign-alternating wherever it
    is not active -- near the wall, and at the smallest scales -- so a
    zero contour there tracks round-off and draws a picket fence
    across regions three orders below the first real level.  Without
    it the near-zero band is one neutral-coloured band, as it should
    be, and no line is drawn through it.

    With *nice* the step is rounded up to a round number
    (:func:`nice_step`), which is what keeps the colour-bar labels
    short across panels whose magnitudes differ by decades.

    *quantile* (0-1) clips the peak to a quantile of ``|values|``
    instead of its maximum -- a guard against one near-wall cell
    setting the scale.  Returns ``(levels, non_negative, vmax)``, the
    last being the end of the colour scale, not of the data.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.asarray([]), True, 0.0
    magnitude = np.abs(finite)
    peak = (
        float(magnitude.max())
        if quantile is None
        else float(np.quantile(magnitude, quantile))
    )
    if peak <= 0.0:
        return np.asarray([]), bool(finite.min() >= 0.0), 0.0
    step = peak / max(n_levels, 2)  # a filled band needs two levels
    if nice:
        step = nice_step(step)
    if finite.min() >= 0.0:
        top = math.ceil(peak / step)
        levels = np.arange(1, top + 1) * step
        return levels, True, float(levels[-1])
    lo = min(math.floor(float(finite.min()) / step), -1)
    hi = max(math.ceil(float(finite.max()) / step), 1)
    levels = np.concatenate([np.arange(lo, 0), np.arange(1, hi + 1)]) * step
    return levels, False, float(np.abs(levels).max())


def draw_map(
    ax,
    map_: Map,
    *,
    units: Units,
    n_levels: int = 10,
    cmap_positive: str = "Greys",
    cmap_signed: str = "RdBu_r",
    quantile: float | None = None,
    nice: bool = True,
    lines: bool = True,
    colorbar: bool = True,
    secondary: bool = True,
    title: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Draw one premultiplied map on *ax*; returns the contour set.

    Filled contours plus (redundant, deliberately) contour lines at
    the same levels; grey-scale for a non-negative field and a
    blue-white-red scale otherwise (:func:`contour_levels`).  With
    *secondary* the outer-unit axes are added opposite the inner-unit
    ones, as in the paper -- and the title is then lifted clear of the
    top one's own label.
    """
    ax.set_xscale("log")
    ax.set_yscale("log")
    above_wall = map_.y > 0.0  # the wall itself has no log position
    y, values = map_.y[above_wall], map_.values[above_wall]
    levels, non_negative, vmax = contour_levels(
        values, n_levels, quantile, nice
    )

    filled = None
    if levels.size:
        cmap = cmap_positive if non_negative else cmap_signed
        span = (0.0, vmax) if non_negative else (-vmax, vmax)
        filled = ax.contourf(
            map_.lam,
            y,
            values,
            levels=levels,
            cmap=cmap,
            norm=Normalize(*span),
        )
        if lines:
            ax.contour(
                map_.lam,
                y,
                values,
                levels=levels,
                colors="k",
                linewidths=0.3,
                alpha=0.7,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "identically zero",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    ax.set_xlim(*(xlim or (map_.lam.min(), map_.lam.max())))
    ax.set_ylim(*(ylim or (y.min(), y.max())))
    ax.set_xlabel(units.lambda_label)
    ax.set_ylabel(units.y_label)

    divider = make_axes_locatable(ax)
    twinned = secondary and units.wall
    if twinned:
        # The outer-unit twins of the two inner-unit axes; both are a
        # plain division by Re_tau, as in the paper's frames.
        outer = (lambda v: v / units.re_tau, lambda v: v * units.re_tau)
        ax.secondary_xaxis("top", functions=outer).set_xlabel(r"$\lambda/h$")
        ax.secondary_yaxis("right", functions=outer).set_ylabel(r"$y/h$")
    if title:
        # tight_layout sizes the cell from the tight bbox (child axes
        # included), but set_title still anchors to the axes box, so
        # the pad has to clear the secondary abscissa by hand.
        ax.set_title(map_.title, pad=_TITLE_PAD if twinned else None)
    if colorbar and filled is not None:
        cax = divider.append_axes(
            "right", size="3.5%", pad=_CBAR_PAD if twinned else 0.1
        )
        bar = ax.figure.colorbar(filled, cax=cax, ticks=_bar_ticks(levels))
        # Three significant digits inline, rather than a shared
        # exponent: the offset box sits over the secondary ordinate.
        bar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{v:.3g}")
        )
        bar.ax.tick_params(labelsize="small")
    return filled


# ── Figures ──────────────────────────────────────────────────────────


def _suptitle(series: YSeries, frame: int, units: Units) -> str:
    r"""``t`` in simulation units and in inner units."""
    t = float(series.t_rel[frame])
    # One math group with `\;` spacers: LaTeX collapses literal spaces
    # between two groups, mathtext knows no `\quad`.
    return (
        rf"$t = {t:.6g}\,h/U_\mathrm{{cl}},"
        rf"\;\;\; t^+ = {units.time(t):.6g}$"
    )


@dataclass(frozen=True)
class PlotStyle:
    """Figure-level knobs shared by the two builders."""

    width: float = PAGE_LINEWIDTH
    aspect: float = 1.3  # of the axes box, not of the grid cell
    ncols: int = 2
    n_levels: int = 10
    cmap_positive: str = "Greys"
    cmap_signed: str = "RdBu_r"
    quantile: float | None = None
    nice: bool = True
    lines: bool = True
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    dpi: int = 200


def _grid(n_panels: int, style: PlotStyle):
    """A figure holding *n_panels* panels, sized from the axes box.

    A panel's decorations -- the ordinate label, the colour bar and
    the two secondary axes -- cost a fixed amount of the cell whatever
    the figure width, so the free parameter is the **axes box**, not
    the cell: :data:`_COL_OVERHEAD` and :data:`_ROW_OVERHEAD` are what
    the decorations take, and :attr:`PlotStyle.aspect` is the box's
    own height/width.
    """
    nrows = math.ceil(n_panels / style.ncols)
    box_w = max(style.width / style.ncols - _COL_OVERHEAD, 0.5)
    height = nrows * (box_w * style.aspect + _ROW_OVERHEAD) + _SUP_OVERHEAD
    fig, axes = plt.subplots(
        nrows, style.ncols, figsize=(style.width, height), squeeze=False
    )
    return fig, axes.ravel()


def _panel_figure(
    series: YSeries,
    frame: int,
    panels: list[tuple[str, int | None]],
    options: MapOptions,
    style: PlotStyle,
):
    """One figure, one ``(name, component)`` map per panel.

    Shared body of :func:`figure_spectra` and :func:`figure_budget`:
    lay the grid out, draw each map, blank the unused cells and title
    the whole with the sample time in both unit systems.
    """
    fig, axes = _grid(len(panels), style)
    for ax, (name, component) in zip(axes, panels, strict=False):
        map_ = make_map(
            series, name, frame, options=options, component=component
        )
        draw_map(
            ax,
            map_,
            units=options.units,
            n_levels=style.n_levels,
            cmap_positive=style.cmap_positive,
            cmap_signed=style.cmap_signed,
            quantile=style.quantile,
            nice=style.nice,
            lines=style.lines,
            xlim=style.xlim,
            ylim=style.ylim,
        )
    for ax in axes[len(panels) :]:
        ax.set_axis_off()
    fig.suptitle(_suptitle(series, frame, options.units))
    fig.tight_layout()
    return fig


def figure_spectra(
    series: YSeries,
    frame: int,
    prefix: str,
    marginal: str,
    *,
    options: MapOptions,
    style: PlotStyle,
):
    """The componentwise spectra of one prefix and marginal.

    Panels: the three velocity components and their sum -- the layout
    of the paper's figure 11.
    """
    name = f"{prefix}_{marginal}"
    panels = [(name, c) for c in range(len(COMPONENTS))] + [(name, None)]
    return _panel_figure(series, frame, panels, options, style)


def figure_budget(
    series: YSeries,
    frame: int,
    marginal: str,
    *,
    options: MapOptions,
    style: PlotStyle,
):
    """Every stored budget term of one marginal, plus their sum."""
    names = [*series.terms, "sum"]
    panels = [(f"{term}_{marginal}", None) for term in names]
    return _panel_figure(series, frame, panels, options, style)


def apply_rcparams(usetex: bool, font_size: float = 11.0) -> None:
    """The write-up's matplotlib style (fonts, preamble, tight bbox)."""
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "text.latex.preamble": LATEX_PREAMBLE if usetex else "",
            "font.size": font_size,
            "axes.titlesize": font_size * 0.9,
            "savefig.bbox": "tight",
            "lines.linewidth": 1,
        }
    )


def resolve_usetex(choice: str) -> bool:
    """``on`` / ``off`` / ``auto`` (LaTeX + dvipng on the PATH)."""
    if choice == "on":
        return True
    if choice == "off":
        return False
    return bool(shutil.which("latex") and shutil.which("dvipng"))


# ── Series registry and driver ───────────────────────────────────────


def available_series(
    spectra: YSeries | None, budget: YSeries | None
) -> dict[str, tuple[str, str, str]]:
    """``tag -> (stream, prefix-or-empty, marginal)`` for what is on disk."""
    out: dict[str, tuple[str, str, str]] = {}
    if spectra is not None:
        for prefix in spectra.prefixes:
            for marginal in MARGINALS:
                out[f"spectra_{prefix}_{marginal}"] = (
                    "twin_yspectra",
                    prefix,
                    marginal,
                )
    if budget is not None:
        for marginal in MARGINALS:
            out[f"budget_{marginal}"] = ("twin_ybudget", "", marginal)
    return out


def render_series(
    series: YSeries,
    tag: str,
    prefix: str,
    marginal: str,
    out_dir: Path,
    *,
    options: MapOptions,
    style: PlotStyle,
    fmt: str = "png",
    pad: int | None = None,
    quiet: bool = False,
) -> list[Path]:
    """Render every frame of one series into ``out_dir/tag``.

    Filenames are ``<tag>_<index>.<fmt>`` with *index* the record's
    position on the members' common time grid, zero-padded so a
    lexical sort is the time order.
    """
    target = out_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    width = pad or len(str(int(series.index.max())))
    written: list[Path] = []
    for frame in range(series.t_rel.size):
        if prefix:
            fig = figure_spectra(
                series, frame, prefix, marginal, options=options, style=style
            )
        else:
            fig = figure_budget(
                series, frame, marginal, options=options, style=style
            )
        path = target / f"{tag}_{int(series.index[frame]):0{width}d}.{fmt}"
        fig.savefig(path, dpi=style.dpi)
        plt.close(fig)
        written.append(path)
        if not quiet:
            print(f"  {path.name}  t = {series.t_rel[frame]:g}", flush=True)
    return written


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface; every documented number is a knob."""
    p = argparse.ArgumentParser(
        prog="twin_spectral_maps.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--members",
        nargs="+",
        required=True,
        metavar="DIR",
        help="dnsjax-twin run directories to ensemble-average",
    )
    p.add_argument(
        "--out", required=True, type=Path, help="output directory root"
    )
    p.add_argument("--re", type=float, required=True, help="phys.re")
    p.add_argument(
        "--re-tau",
        type=float,
        required=True,
        help="measured friction Reynolds number (never re-measured)",
    )
    p.add_argument(
        "--stride", type=int, default=10, help="keep every Nth record"
    )
    p.add_argument("--first", type=int, default=0, help="first record kept")
    p.add_argument(
        "--last", type=int, default=None, help="last record kept (inclusive)"
    )
    p.add_argument(
        "--series",
        nargs="+",
        default=None,
        metavar="TAG",
        help="subset of series tags to render (default: all present)",
    )
    p.add_argument("--levels", type=int, default=10, help="contour levels")
    p.add_argument("--cmap-positive", default="Greys")
    p.add_argument("--cmap-signed", default="RdBu_r")
    p.add_argument(
        "--exact-levels",
        action="store_true",
        help="space levels by vmax/levels instead of a round step",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="clip the colour scale to this quantile of |values|",
    )
    p.add_argument(
        "--no-lines", action="store_true", help="drop contour lines"
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=1,
        help="running mean over this many adjacent wavenumbers",
    )
    p.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="wavelength axis limits, in the plotted units",
    )
    p.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="wall-normal axis limits, in the plotted units",
    )
    p.add_argument("--width", type=float, default=PAGE_LINEWIDTH)
    p.add_argument(
        "--aspect",
        type=float,
        default=1.3,
        help="axes-box height/width of one panel",
    )
    p.add_argument("--ncols", type=int, default=2)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--format", default="png", help="savefig extension")
    p.add_argument(
        "--pad", type=int, default=None, help="filename zero-padding width"
    )
    p.add_argument(
        "--outer-units",
        action="store_true",
        help="plot in h / U_cl instead of wall units",
    )
    p.add_argument(
        "--half",
        choices=("mean", "lower", "upper"),
        default="mean",
        help="how the two channel halves collapse onto one y+",
    )
    p.add_argument(
        "--no-volume-fac",
        action="store_true",
        help="plot the stored y-mean density, not the local one",
    )
    p.add_argument("--usetex", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    """Render every series of both streams for one member set."""
    args = build_parser().parse_args(argv)
    apply_rcparams(resolve_usetex(args.usetex))
    options = MapOptions(
        units=Units(args.re, args.re_tau, wall=not args.outer_units),
        half=args.half,
        volume_fac=not args.no_volume_fac,
        smooth=args.smooth,
    )
    style = PlotStyle(
        width=args.width,
        aspect=args.aspect,
        ncols=args.ncols,
        n_levels=args.levels,
        cmap_positive=args.cmap_positive,
        cmap_signed=args.cmap_signed,
        quantile=args.quantile,
        nice=not args.exact_levels,
        lines=not args.no_lines,
        xlim=None if args.xlim is None else tuple(args.xlim),
        ylim=None if args.ylim is None else tuple(args.ylim),
        dpi=args.dpi,
    )

    opened: dict[str, YSeries | None] = {}
    for stem in STEMS:
        try:
            opened[stem] = open_series(
                args.members,
                stem,
                stride=args.stride,
                first=args.first,
                last=args.last,
            )
        except FileNotFoundError as exc:
            print(f"skipping {stem}: {exc}", file=sys.stderr)
            opened[stem] = None

    registry = available_series(
        opened["twin_yspectra"], opened["twin_ybudget"]
    )
    tags = args.series or list(registry)
    unknown = [t for t in tags if t not in registry]
    if unknown:
        raise SystemExit(
            f"unknown series {unknown}; available: {list(registry)}"
        )
    if not tags:
        raise SystemExit("no stream found under the given members")

    if not args.quiet:
        print(
            f"{len(args.members)} member(s), Re = {args.re:g}, "
            f"Re_tau = {args.re_tau:g}, stride {args.stride}, "
            f"usetex {plt.rcParams['text.usetex']}",
            flush=True,
        )
    for tag in tags:
        stem, prefix, marginal = registry[tag]
        series = opened[stem]
        assert series is not None
        if not args.quiet:
            print(
                f"{tag}: {series.t_rel.size} frames, "
                f"t = {series.t_rel[0]:g}..{series.t_rel[-1]:g}",
                flush=True,
            )
        render_series(
            series,
            tag,
            prefix,
            marginal,
            args.out,
            options=options,
            style=style,
            fmt=args.format,
            pad=args.pad,
            quiet=args.quiet,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
