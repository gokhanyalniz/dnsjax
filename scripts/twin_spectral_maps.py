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

Premultiplication
=================
A stored entry is the energy (or rate) held by one **discrete** mode
band, not a spectral density: summing the entries over the stored
one-sided axis and contracting with ``y_weights`` returns the volume
average.  The density is therefore ``entry / dk`` with
`$\Delta k = 2\pi/L$`, and since `$k_m = m\,\Delta k$` for the integer
harmonic `$m$`, `$k\,\Phi = m \times \text{entry}$`, independent of the
box length.

Both axes are logarithmic, so **both** get a premultiplier -- the
paper's (2.8), `$\int\!\!\int \Phi\,\mathrm{d}k\,\mathrm{d}y =
\int\!\!\int k\,y\,\Phi\,\mathrm{d}\log k\,\mathrm{d}\log y$`.  The
plotted quantity is therefore

.. math::  y\,k\,\Phi(y, k) = y \times m \times \text{entry}(y, m) ,

with `$y$` the wall distance **in the plotted units** (`$y^+$` by
default), which is what makes equal areas on the map equal energy in
those same units.  ``--premultiply k`` drops the `$y$` factor and
``none`` drops both.

The `$m = 0$` column carries no premultiplied content and is dropped,
which is also what makes the maps read as fluctuation spectra: the
wall-parallel mean of a state lives at `$(0, 0)$` alone, so at every
plotted `$m \ge 1$` the stored perturbation-about-laminar spectrum
``r_*`` *is* the spectrum of the fluctuation about the `$x$`-`$z$`
mean.

Stored entries are additionally divided by ``volume_fac`` (the
channel's wall-normal extent) so that the plain `$y$`-average of a
profile is the volume average.  Multiplying it back gives the local
density, which is the quantity the paper plots; that is
:attr:`MapOptions.volume_fac` and it is on by default.  What fixes
that factor empirically is the ``r_*`` half of the spectra stream: at
plane-Poiseuille `$Re = 4200$`, `$Re_\tau = 178.6$`, its `$k$`-only
premultiplied map puts `$\max k_z E^{x+}_{uu} \approx 3.8$` at
`$y^+ \approx 14$`, `$\lambda_z^+ \approx 130$` -- the textbook
near-wall peak.  Without it every map is a factor of two low.

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

Ensemble averaging
==================
Members are averaged sample by sample on **relative** time
`$t - t_\mathrm{parent}$`, the clock since the perturbation, as
:func:`dnsjax.analysis.twin.ensemble.aggregate_members` does: members
start from different parent snapshots, so their absolute times do not
match.  `$t_\mathrm{parent}$` comes from each member's ``twin.json``
when the record is there and from the stream's first sample
otherwise, and the frames rendered are the **intersection** of the
members' relative grids, so a short or resumed member restricts the
set rather than corrupting it.  With one shared ``dt`` and cadence
this selects the same samples as matching by index, but keyed on the
clock, which is what survives a member whose stream starts elsewhere.
Any number of members works; ``--tree`` takes an
``ensemble_setup.py build-twin`` tree and uses every member its
``members.json`` lists.

Folding the channel
===================
``--half mean`` (the default) averages the two channel halves at
matching wall distance, which is legitimate and free statistics
because the reflection `$R_y:\,(u,v,w)(x,y,z) \mapsto
(u,-v,w)(x,-y,z)$` is a symmetry of the flow.  Every stored quantity
is `$R_y$`-**even**, so the fold is a plain arithmetic mean with no
sign flips: the spectra are moduli; `$\mathcal{P}^U$` flips both
`$\Delta\hat v$` and `$\partial_y U$`; `$\mathcal{V}$`,
`$\hat\varepsilon$` and `$\mathcal{W}$` pair each odd factor with a
`$\partial_y$` or with the `$v$` slot; and each transfer term carries
an even number of odd factors for the same reason.  The mid-plane
pairs with itself and is **not** double counted, and the grid is
checked for the symmetry the fold assumes rather than trusted.
``--half lower`` / ``upper`` keep one wall instead, which is how a
run's own asymmetry is inspected.

Colour scales
=============
Non-negativity is **declared**, not inferred: the energies and the
pseudo-dissipation are sums of squares (:data:`NON_NEGATIVE`) and get
the grey scale, everything else the diverging one.  The declaration is
asserted against the data once per series and a negative excursion is
reported with its size relative to the peak -- at round-off it is
truncation and the map is drawn regardless; anything larger is worth
looking at, and the map is still drawn.  ``--signs-from-data`` infers
the sign instead, for a stream this list does not cover.

The `$y$` grid is the solver's own (CGL by default), and nothing here
assumes it is uniform: ``contourf`` / ``contour`` are handed the
coordinate arrays, so a contour lands at the wall distance it belongs
to.  Clustering at the wall makes that grid *coarse in* `$\log y$`
there -- its first plotted cell spans 0.6 of a decade -- which is
where ``--fill pcolormesh`` earns its place: the same bands, one flat
cell per sample on geometric-mean edges (:func:`log_edges`), with no
interpolation between them.  The contour lines are drawn on top
either way.

Every colour scale, per frame or frozen, is read off the **plotted**
quantity -- premultiplied, folded, in inner units, over exactly the
rows the logarithmic ordinate shows (:meth:`Map.drawn`).  The colour
bar therefore labels the same numbers the contours do.

Each panel's scale is frozen (``--clim series``, the default) on the
ensemble-global extremes of that quantity over the rendered frames, so
one panel means the same thing in every figure of a sequence and the
colour bar can be read once.  The price is the growth phase: a
difference field that saturates four decades above its initial energy
leaves the first frames below the first contour level, and they come
out blank rather than rescaled.  ``--clim frame`` rescales every
figure to its own peak instead, which is what shows the *shape* while
the amplitude is still climbing -- at the cost of a colour bar that
moves under you.  The sign family is decided once for the whole series
either way, so a panel never changes colour map mid-run.

Figure geometry
===============
**One decade is the same length on both axes** -- the constraint that
makes a `$\lambda \propto y$` band read at 45 degrees -- so the axes
box is `$(\text{decade} \times D_\lambda,\ \text{decade} \times D_y)$`
for the decade counts `$D$` of the plotted limits, and
``ax.set_aspect(1)`` holds it there.  That fixes the box's shape from
the limits and leaves only its scale free, which ``--width`` sets
(default 6.61546 in, the write-up's ``\linewidth``); ``--decade``
sets it directly instead.  A **square** box therefore needs equal
decade counts, which is a choice of ``--ylim`` rather than of sizing:
on the full stored range `$y^+ \in [0.02, 179]$` against
`$\lambda_z^+ \in [14, 1122]$` the box is 2.1 times taller than wide,
and an eight-panel budget figure is correspondingly tall.

Usage
=====
matplotlib is not a solver dependency; it lives in the ``plots``
dependency group::

    uv run --group plots python scripts/twin_spectral_maps.py \
        --members RUN1 RUN2 --out FIGDIR \
        --re 4200 --re-tau 178.62135279727977 --stride 10

Every number above is a knob; ``--help`` lists the rest.

As a library (a notebook on the cluster, one stream at a time)::

    from twin_spectral_maps import (
        MapOptions, Units, draw_map, make_map, open_series)

    s = open_series(["twin1", "twin2"], "twin_ybudget", stride=10)
    opts = MapOptions(Units(re=4200.0, re_tau=178.62135279727977))
    m = make_map(s, "P_U_x", frame=24, options=opts)
    draw_map(ax, m, units=opts.units)

:func:`open_series` memory-maps each member and reads only the
selected records, so a single stream costs megabytes rather than the
gigabyte the eager reader
:func:`dnsjax.analysis.twin.yspectra.read_twin_yspectra` would pull
in; the record layout, the format-version floors and the
duplicate-``t`` policy are that reader's, mirrored here.
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
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import FuncFormatter

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

#: Fields that are non-negative **by construction**, keyed by the base
#: name: the two spectra prefixes are `$\tfrac12|\hat u|^2$` sums and
#: ``eps`` is `$\nu(|\partial_y\hat u|^2 + k^2|\hat u|^2)$`.  ``V`` is
#: deliberately absent -- the operator (discrete-Laplacian) viscous
#: form is *not* sign-definite, as :mod:`dnsjax.twin.diagnostics`
#: ("Dissipation form") sets out, however negative it happens to come
#: out in any given run.
NON_NEGATIVE: frozenset[str] = frozenset({"e", "r", "eps"})

#: Below this fraction of the peak, a negative excursion in a declared
#: non-negative field is reported as truncation rather than a defect.
#: These are sums of squares, so round-off is the only mechanism and
#: it lands many orders below this.
SIGN_TOLERANCE: float = 1e-9

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

#: ``(wavelength axis, energy superscript)`` per stored suffix.  A
#: suffix names the axis that was **summed over**, so ``_x`` is the
#: `$k_z$` marginal and its abscissa is `$\lambda_z$`.  The axis letter
#: is what both the panel title and the two abscissa labels subscript,
#: which is why it is stored rather than a ready-made ``k_z``.
MARGINALS: dict[str, tuple[str, str]] = {
    "x": ("z", "x"),
    "z": ("x", "z"),
    "x0": ("z", "x0"),
}

#: LaTeX preamble matching the ``perturbation_dynamics`` write-up.
LATEX_PREAMBLE: str = r"""
\usepackage[p]{stickstootext}
\usepackage[scaled=1.05,stix2,vvarbb]{newtxmath}
\usepackage[defaultsans,proportional,scale=0.955]{lato}
"""

#: Text width of that document, in inches (its ``\linewidth``).
PAGE_LINEWIDTH: float = 6.61546

#: Panel margins, in inches, at the default font size.  The layout is
#: placed explicitly (:func:`panel_geometry`) rather than by
#: ``tight_layout``, because the equal-decade rule fixes the axes box
#: and everything else has to be budgeted around it.
_M_LEFT: float = 0.62  # ordinate label + its tick labels
_M_RIGHT: float = 0.05  # trailing strip
_M_TOP: float = 0.78  # top tick labels + secondary label + title
_M_BOTTOM: float = 0.55  # bottom tick labels + abscissa label
_RIGHT_AXIS: float = 0.62  # secondary ordinate, right of the box
_CBAR_PAD: float = 0.08
_CBAR_WIDTH: float = 0.10
_CBAR_LABELS: float = 0.50
_COL_GAP: float = 0.12
_ROW_GAP: float = 0.10
_SUP_HEIGHT: float = 0.36

#: Title offset in points, inside the top margin budgeted above.
_TITLE_PAD: float = 12.0

#: Upper bound on the number of labelled colour-bar ticks.
_BAR_TICKS: int = 6


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
    def suffix(self) -> str:
        """``+`` on a symbol that is plotted in inner units."""
        return "^+" if self.wall else ""

    def lambda_label(self, axis: str = "", *, outer: bool = False) -> str:
        r"""Abscissa label for the wavelength of one marginal.

        *axis* is that wavelength's own subscript -- ``z`` on the
        `$k_z$` marginal, ``x`` on the `$k_x$` one (:data:`MARGINALS`)
        -- and is empty only for a map that is not a single marginal.
        *outer* forces outer units, which is what the secondary
        abscissa wants while the primary one is in wall units.
        """
        lam = rf"\lambda_{{{axis}}}" if axis else r"\lambda"
        if outer or not self.wall:
            return rf"${lam}/h$"
        return rf"${lam}^+$"

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
    t_rel: np.ndarray  # their times since the perturbation


def _perturbation_time(path: Path, first_sample: float) -> float:
    """The member's `$t_\\mathrm{parent}$`, the clock the ensemble uses.

    ``twin.json``'s ``parent_t`` when the member record is there --
    what :mod:`dnsjax.analysis.twin.series` uses, and the only one
    that is right for a member whose stream begins at a resume rather
    than at the perturbation.  Otherwise the stream's first sample.
    """
    record = path / "twin.json"
    if record.is_file():
        with open(record) as fh:
            meta = json.load(fh)
        if meta.get("parent_t") is not None:
            return float(meta["parent_t"])
    return first_sample


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
    t0 = _perturbation_time(path, float(t[rows[0]]))
    return _Member(path, meta, records, rows, t[rows] - t0)


@dataclass
class YSeries:
    r"""An ensemble-averaged, subsampled `$(y, k)$` stream.

    Fields are read and averaged on demand (and cached), so opening a
    series is cheap however long the run and however many members it
    has.  ``index`` carries each frame's position on the members'
    common relative-time grid -- the number the output filenames use.
    """

    stem: str
    members: tuple[_Member, ...]
    rows: np.ndarray  # (n_members, n_frames) record index per member
    index: np.ndarray  # (n_frames,) position on the common grid
    t_rel: np.ndarray  # (n_frames,) time since the perturbation
    meta: dict  # the first member's sidecar
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def n_members(self) -> int:
        """How many member streams are being averaged."""
        return len(self.members)

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
            value = total / self.n_members
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


def tree_members(tree: str | Path) -> list[Path]:
    """Member directories of an ``ensemble_setup.py build-twin`` tree.

    Reads the tree's ``members.json`` index, the same one
    :func:`dnsjax.analysis.twin.ensemble.aggregate_members` walks.
    """
    tree = Path(tree)
    with open(tree / "members.json") as fh:
        spec = json.load(fh)
    if spec.get("kind") != "twin":
        raise ValueError(
            f"{tree}/members.json is not a twin tree "
            f"(kind = {spec.get('kind')!r})."
        )
    members = spec.get("members") or []
    if not members:
        raise ValueError(f"{tree}/members.json lists no members")
    return [tree / record["dir"] for record in members]


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
    Members are aligned on time since the perturbation (module
    docstring, "Ensemble averaging"); any number of them works.
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
    return YSeries(
        stem=stem,
        members=tuple(opened),
        rows=rows,
        index=_locate(keys[0], common, opened[0].path),
        t_rel=common,
        meta=opened[0].meta,
    )


# ── Map construction ─────────────────────────────────────────────────


@dataclass(frozen=True)
class MapOptions:
    r"""Everything that turns a stored field into a plotted map.

    *premultiply* is ``"ky"`` (the paper's (2.8), both axes
    logarithmic), ``"k"`` or ``"none"``; *half* is how the two channel
    halves collapse onto the one wall distance a `$y^+$` ordinate
    needs (module docstring, "Folding the channel"); *volume_fac*
    multiplies the stored `$y$`-mean density back to the local density
    the paper plots.

    *smooth* is a centred running mean over that many adjacent
    wavenumbers, applied last.  It is off (``1``) by default and is
    presentation only: a few-member ensemble mean of an
    *instantaneous* field is genuinely rough mode to mode -- unlike
    the paper's long-time averages of a stationary flow -- and the
    transfer terms show it.  Widening this bins the map; it does not
    denoise it.
    """

    units: Units
    premultiply: str = "ky"
    half: str = "mean"
    volume_fac: bool = True
    smooth: int = 1


@dataclass(frozen=True)
class Map:
    r"""One panel's data: ``values`` on the `$(\lambda, y)$` grid."""

    lam: np.ndarray  # (n_lam,) wavelength, plotted units
    y: np.ndarray  # (n_y,) wall distance, plotted units
    values: np.ndarray  # (n_y, n_lam)
    title: str  # LaTeX panel title, with normalisation
    name: str  # the stored field it came from
    non_negative: bool  # declared or inferred; sets the colour family

    @property
    def lam_axis(self) -> str:
        r"""Which wavelength the abscissa is: ``x`` or ``z``.

        Read off the stored suffix of :attr:`name`, the same way
        :func:`field_title` reads the panel title's wavenumber, so a
        panel and its axes cannot label different marginals.
        """
        return MARGINALS[self.name.rpartition("_")[2]][0]

    def drawn(self) -> tuple[np.ndarray, np.ndarray]:
        r"""The rows a logarithmic ordinate can show: ``y > 0``.

        The wall row has no position on a log axis, so it is dropped
        from the plot -- and therefore from the colour scale too,
        which is why :func:`scan_panels` and :func:`draw_map` both go
        through here rather than reading ``values`` directly.  It
        matters under ``--premultiply k`` / ``none``, where the wall
        value of a budget term is not zero (`$\hat\varepsilon$` is
        largest there); under the default ``ky`` the `$y$` factor
        zeroes that row anyway.
        """
        above_wall = self.y > 0.0
        return self.y[above_wall], self.values[above_wall]


def _select_half(
    values: np.ndarray, y: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the channel onto one wall distance.

    Index ``j`` pairs with ``n_y - 1 - j`` -- checked against the grid
    rather than assumed -- and for odd ``n_y`` the mid-plane pairs with
    itself, so ``"mean"`` neither needs a special case there nor
    double counts it.  Every stored quantity is even under the
    reflection, so the mean carries no sign flips; the argument is in
    the module docstring, "Folding the channel".  Returns the
    collapsed values and the wall distance `$1 - |y|$` of the retained
    rows, ascending from the wall.
    """
    if not np.allclose(y, -y[::-1], rtol=0.0, atol=1e-12):
        raise ValueError(
            "the wall-normal grid is not symmetric about the centreline, "
            "so the R_y fold does not apply; use --half lower / upper"
        )
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
    norm = np.convolve(np.ones(values.shape[-1]), kernel, mode="same")
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), -1, values
    )
    return smoothed / norm


def field_kind(series: YSeries) -> str:
    """``"energy"`` or ``"rate"``, by which stream *series* is."""
    return "energy" if series.stem == "twin_yspectra" else "rate"


def declared_non_negative(name: str) -> bool:
    """Whether :data:`NON_NEGATIVE` covers a stored (or virtual) name."""
    return name.rpartition("_")[0] in NON_NEGATIVE


def field_title(
    series: YSeries,
    name: str,
    component: int | None,
    options: MapOptions,
) -> str:
    """The LaTeX panel title for one stored (or virtual) field."""
    base, _, suffix = name.rpartition("_")
    axis, superscript = MARGINALS[suffix]
    wavenumber = rf"k_{{{axis}}}"
    kind = field_kind(series)
    plus = options.units.suffix
    factor = {
        "ky": rf"{wavenumber}{plus} y{plus}\,",
        "k": rf"{wavenumber}{plus}\,",
        "none": "",
    }[options.premultiply]
    if kind == "rate":
        body = TERM_LABELS.get(base, base.replace("_", r"\_"))
    else:
        delta = r"\Delta " if base == "e" else ""
        if component is None:
            inner = rf"{delta}\alpha" if base == "e" else r"\alpha"
            body = rf"\sum_\alpha E^{{{superscript}}}_{{{inner}}}"
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
    non_negative: bool | None = None,
) -> Map:
    """Build one premultiplied map from a stored (or virtual) field.

    *name* is a stored field such as ``e_x`` / ``P_U_z``, or the
    virtual ``sum_x``; *component* selects a velocity component of a
    ``twin_yspectra`` field (``None`` sums the three).  *frame* indexes
    the series' subsampled records.  *non_negative* overrides the
    declaration of :data:`NON_NEGATIVE` -- what ``--signs-from-data``
    and :func:`scan_panels` feed back in.
    """
    suffix = name.rpartition("_")[2]
    values = series.field(name)[frame]
    if series.stem == "twin_yspectra":
        values = values.sum(axis=0) if component is None else values[component]
    elif component is not None:
        raise ValueError(f"{series.stem}: {name} has no component axis")

    values = values[:, 1:]  # m = 0 carries no premultiplied content
    if options.premultiply != "none":
        values = values * series.harmonics(suffix)[None, 1:]
    if options.volume_fac:
        values = values * series.volume_fac
    values = options.units.convert(values, field_kind(series))
    values = values[:, ::-1]  # ascending in wavelength, as the axis is

    values, wall_distance = _select_half(values, series.y, options.half)
    y = options.units.length(wall_distance)
    if options.premultiply == "ky":
        # The second logarithmic axis needs its own premultiplier, in
        # the units the ordinate is drawn in (module docstring).
        values = values * y[:, None]
    values = _running_mean(values, options.smooth)
    return Map(
        lam=options.units.length(series.wavelengths(suffix)),
        y=y,
        values=values,
        title=field_title(series, name, component, options),
        name=name,
        non_negative=(
            declared_non_negative(name)
            if non_negative is None
            else non_negative
        ),
    )


# ── Colour scales ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PanelScale:
    """One panel's series-global range and its sign family."""

    lo: float
    hi: float
    non_negative: bool


def scan_panels(
    series: YSeries,
    panels: list[tuple[str, int | None]],
    options: MapOptions,
    *,
    declared: bool = True,
) -> tuple[dict[tuple[str, int | None], PanelScale], list[str]]:
    """Series-global extremes per panel, and the sign-check report.

    Walks every frame of every panel once -- the member means are
    cached, so this is a set of NumPy reductions over arrays already
    in memory -- and returns ``{(name, component): PanelScale}`` plus
    the lines to print.  With *declared* the sign family comes from
    :data:`NON_NEGATIVE` and the data only **checks** it; without, it
    is inferred from the series-global minimum, which is what a stream
    the declaration does not cover needs.

    Either way the family is decided once for the whole series, so a
    panel cannot change colour map from frame to frame.
    """
    scales: dict[tuple[str, int | None], PanelScale] = {}
    notes: list[str] = []
    for name, component in panels:
        lo, hi = math.inf, -math.inf
        for frame in range(series.t_rel.size):
            _, values = make_map(
                series,
                name,
                frame,
                options=options,
                component=component,
                non_negative=True,  # irrelevant here; decided below
            ).drawn()
            finite = values[np.isfinite(values)]
            if finite.size:
                lo = min(lo, float(finite.min()))
                hi = max(hi, float(finite.max()))
        if not math.isfinite(lo):
            lo = hi = 0.0
        label = (
            name if component is None else f"{name}[{COMPONENTS[component]}]"
        )
        if declared and declared_non_negative(name):
            non_negative = True
            peak = max(abs(lo), abs(hi))
            if lo < 0.0 and peak > 0.0:
                ratio = -lo / peak
                verdict = (
                    "round-off, i.e. truncation"
                    if ratio < SIGN_TOLERANCE
                    else "ABOVE round-off -- worth a look"
                )
                notes.append(
                    f"  {label}: declared non-negative, min/peak = "
                    f"-{ratio:.3e} ({verdict}); drawn as non-negative"
                )
        else:
            non_negative = lo >= 0.0
        scales[(name, component)] = PanelScale(lo, hi, non_negative)
    return scales, notes


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
    *,
    non_negative: bool,
    data_range: tuple[float, float] | None = None,
    quantile: float | None = None,
    nice: bool = True,
) -> tuple[np.ndarray, float]:
    """Contour levels for one map, and the end of its colour scale.

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

    *data_range* freezes the scale on a range computed elsewhere
    (``--clim series``); without it the map's own extremes are used.
    *quantile* (0-1) clips the peak to a quantile of ``|values|``
    instead of its maximum -- a guard against one near-wall cell
    setting the scale.  With *nice* the step is rounded up to a round
    number (:func:`nice_step`), which is what keeps the colour-bar
    labels short across panels whose magnitudes differ by decades.
    """
    finite = values[np.isfinite(values)]
    if data_range is not None:
        lo, hi = data_range
    elif finite.size:
        lo, hi = float(finite.min()), float(finite.max())
    else:
        return np.asarray([], dtype=float), 0.0
    peak = max(abs(lo), abs(hi))
    if quantile is not None and finite.size:
        peak = float(np.quantile(np.abs(finite), quantile))
    if peak <= 0.0:
        return np.asarray([], dtype=float), 0.0

    step = peak / max(n_levels, 2)  # a filled band needs two levels
    if nice:
        step = nice_step(step)
    if non_negative:
        levels = np.arange(1, math.ceil(peak / step) + 1) * step
        return levels, float(levels[-1])
    below = min(math.floor(lo / step), -1)
    above = max(math.ceil(hi / step), 1)
    levels = (
        np.concatenate([np.arange(below, 0), np.arange(1, above + 1)]) * step
    )
    return levels, float(np.abs(levels).max())


def log_edges(centres: np.ndarray) -> np.ndarray:
    r"""Cell edges around log-spaced *centres*: the geometric means.

    ``pcolormesh`` wants edges, and on a logarithmic axis the edge
    between two samples is their geometric mean, not their arithmetic
    one -- the difference is visible in the near-wall cells, where the
    CGL grid is coarsest **in `$\log y$`** (its first plotted cell
    spans 0.6 of a decade).  The two outermost edges are extrapolated
    by the same half-step.
    """
    mid = np.sqrt(centres[:-1] * centres[1:])
    return np.concatenate(
        [[centres[0] ** 2 / mid[0]], mid, [centres[-1] ** 2 / mid[-1]]]
    )


def _bar_ticks(levels: np.ndarray) -> np.ndarray:
    """At most :data:`_BAR_TICKS` of the contour levels, zero kept."""
    every = max(1, math.ceil(levels.size / _BAR_TICKS))
    if every == 1:
        return levels
    zero = int(np.argmin(np.abs(levels)))
    return levels[zero % every :: every]


# ── Drawing ──────────────────────────────────────────────────────────


def draw_map(
    ax,
    map_: Map,
    *,
    units: Units,
    n_levels: int = 10,
    cmap_positive: str = "Greys",
    cmap_signed: str = "RdBu_r",
    data_range: tuple[float, float] | None = None,
    quantile: float | None = None,
    nice: bool = True,
    fill: str = "contour",
    lines: bool = True,
    cax=None,
    secondary: bool = True,
    title: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Draw one premultiplied map on *ax*; returns the contour set.

    Filled contours plus (redundant, deliberately) contour lines at
    the same levels; grey-scale when ``map_.non_negative`` and a
    blue-white-red scale otherwise.  Both axes are logarithmic and
    ``set_aspect(1)`` holds one decade to the same length on each --
    the layout of :func:`panel_geometry` sizes the box so that costs
    nothing.  *cax* is where the colour bar goes (``None``: no bar).
    """
    ax.set_xscale("log")
    ax.set_yscale("log")
    y, values = map_.drawn()
    levels, vmax = contour_levels(
        values,
        n_levels,
        non_negative=map_.non_negative,
        data_range=data_range,
        quantile=quantile,
        nice=nice,
    )

    filled = None
    if levels.size:
        cmap = cmap_positive if map_.non_negative else cmap_signed
        span = (0.0, vmax) if map_.non_negative else (-vmax, vmax)
        if fill == "pcolormesh":
            # The same bands, drawn per cell instead of interpolated
            # between samples -- the honest rendering of a grid that
            # is coarse in log y near the wall (:func:`log_edges`).
            shaded = plt.get_cmap(cmap).copy()
            shaded.set_under((1.0, 1.0, 1.0, 0.0))  # as contourf leaves it
            filled = ax.pcolormesh(
                log_edges(map_.lam),
                log_edges(y),
                values,
                cmap=shaded,
                norm=BoundaryNorm(levels, shaded.N, extend="max"),
            )
        else:
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
    # One decade, one length, on both axes: the constraint the whole
    # layout is built around.  A no-op once panel_geometry has sized
    # the box, and the guarantee if anything else moves the limits.
    ax.set_aspect(1.0, adjustable="box", anchor="C")
    ax.set_xlabel(units.lambda_label(map_.lam_axis))
    ax.set_ylabel(units.y_label)

    if secondary and units.wall:
        # The outer-unit twins of the two inner-unit axes; both are a
        # plain division by Re_tau, as in the paper's frames.
        outer = (lambda v: v / units.re_tau, lambda v: v * units.re_tau)
        ax.secondary_xaxis("top", functions=outer).set_xlabel(
            units.lambda_label(map_.lam_axis, outer=True)
        )
        ax.secondary_yaxis("right", functions=outer).set_ylabel(r"$y/h$")
    if title:
        ax.set_title(map_.title, pad=_TITLE_PAD)
    if cax is not None and filled is not None:
        bar = ax.figure.colorbar(filled, cax=cax, ticks=_bar_ticks(levels))
        # Three significant digits inline, rather than a shared
        # exponent: the offset box sits over the secondary ordinate.
        bar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{v:.3g}")
        )
        bar.ax.tick_params(labelsize="small")
    elif cax is not None:
        cax.set_axis_off()
    return filled


# ── Figures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlotStyle:
    """Figure-level knobs shared by the two builders.

    *decade* is inches per decade, the one free parameter the
    equal-decade rule leaves; ``None`` derives it from *width* so the
    figure comes out exactly that wide (module docstring, "Figure
    geometry").
    """

    width: float = PAGE_LINEWIDTH
    decade: float | None = None
    ncols: int = 2
    n_levels: int = 10
    cmap_positive: str = "Greys"
    cmap_signed: str = "RdBu_r"
    quantile: float | None = None
    nice: bool = True
    fill: str = "contour"
    lines: bool = True
    freeze_clim: bool = True
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    dpi: int = 200


#: Inches a panel needs to the right of its axes box, and the pitch
#: from one column's box to the next -- which has to carry the *next*
#: column's ordinate labels (``_M_LEFT``) as well, not only this
#: column's colour bar.
_COL_AFTER: float = _RIGHT_AXIS + _CBAR_PAD + _CBAR_WIDTH + _CBAR_LABELS
_COL_PITCH_EXTRA: float = _COL_AFTER + _COL_GAP + _M_LEFT


@dataclass(frozen=True)
class Geometry:
    """A figure's placement, in inches and figure fractions."""

    fig_w: float
    fig_h: float
    box_w: float
    box_h: float
    nrows: int
    ncols: int

    def axes_rect(self, panel: int) -> tuple[float, float, float, float]:
        """``[left, bottom, width, height]`` of one panel's axes box."""
        row, col = divmod(panel, self.ncols)
        left = _M_LEFT + col * (self.box_w + _COL_PITCH_EXTRA)
        top = (
            _SUP_HEIGHT
            + row * (_M_TOP + self.box_h + _M_BOTTOM + _ROW_GAP)
            + _M_TOP
        )
        return (
            left / self.fig_w,
            1.0 - (top + self.box_h) / self.fig_h,
            self.box_w / self.fig_w,
            self.box_h / self.fig_h,
        )

    def cbar_rect(self, panel: int) -> tuple[float, float, float, float]:
        """``[left, bottom, width, height]`` of its colour bar."""
        left, bottom, _, height = self.axes_rect(panel)
        return (
            left + (self.box_w + _RIGHT_AXIS + _CBAR_PAD) / self.fig_w,
            bottom,
            _CBAR_WIDTH / self.fig_w,
            height,
        )


def panel_geometry(
    n_panels: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    style: PlotStyle,
) -> Geometry:
    """Size a figure so one decade is one length on both axes.

    The axes box is ``decade`` inches per decade in **both**
    directions, so its shape follows from the limits alone; the
    remaining freedom is the scale, taken from ``style.decade`` when
    given and otherwise from ``style.width``, which then comes out
    exact.  Everything around the box is a fixed inch budget (the
    ``_M_*`` module constants), so the figure height is whatever the
    panels need.
    """
    nrows = math.ceil(n_panels / style.ncols)
    decades_x = math.log10(xlim[1] / xlim[0])
    decades_y = math.log10(ylim[1] / ylim[0])
    if style.decade is not None:
        decade = style.decade
    else:
        usable = (
            style.width
            - _M_LEFT
            - _M_RIGHT
            - (style.ncols - 1) * (_COL_GAP + _M_LEFT)
        )
        box_w = usable / style.ncols - _COL_AFTER
        if box_w <= 0.1:
            raise ValueError(
                f"--width {style.width:g} in leaves no room for "
                f"{style.ncols} columns: labels and colour bar take about "
                f"{_COL_AFTER + _M_LEFT:.2f} in of each.  Widen the "
                "figure, drop a column, or set --decade."
            )
        decade = box_w / decades_x
    box_w, box_h = decade * decades_x, decade * decades_y
    fig_w = (
        _M_LEFT
        + style.ncols * (box_w + _COL_AFTER)
        + (style.ncols - 1) * (_COL_GAP + _M_LEFT)
        + _M_RIGHT
    )
    fig_h = (
        _SUP_HEIGHT
        + nrows * (_M_TOP + box_h + _M_BOTTOM)
        + (nrows - 1) * _ROW_GAP
    )
    return Geometry(fig_w, fig_h, box_w, box_h, nrows, style.ncols)


def _suptitle(series: YSeries, frame: int, units: Units) -> str:
    r"""``t`` in simulation units and in inner units."""
    t = float(series.t_rel[frame])
    # One math group with `\;` spacers: LaTeX collapses literal spaces
    # between two groups, mathtext knows no `\quad`.
    return (
        rf"$t = {t:.6g}\,h/U_\mathrm{{cl}},"
        rf"\;\;\; t^+ = {units.time(t):.6g}$"
    )


def panel_figure(
    series: YSeries,
    frame: int,
    panels: list[tuple[str, int | None]],
    options: MapOptions,
    style: PlotStyle,
    scales: dict[tuple[str, int | None], PanelScale],
):
    """One figure, one ``(name, component)`` map per panel.

    Shared body of :func:`spectra_panels` and :func:`budget_panels`
    figures.  Every panel of a figure shares one wavelength axis and
    one wall-normal grid, so a single :func:`panel_geometry` sizes
    them all.
    """
    maps = [
        make_map(
            series,
            name,
            frame,
            options=options,
            component=component,
            non_negative=scales[(name, component)].non_negative,
        )
        for name, component in panels
    ]
    drawn_y = maps[0].drawn()[0]
    xlim = style.xlim or (maps[0].lam.min(), maps[0].lam.max())
    ylim = style.ylim or (drawn_y.min(), drawn_y.max())
    geometry = panel_geometry(len(panels), xlim, ylim, style)

    fig = plt.figure(figsize=(geometry.fig_w, geometry.fig_h))
    for panel, (map_, key) in enumerate(zip(maps, panels, strict=True)):
        scale = scales[key]
        draw_map(
            fig.add_axes(geometry.axes_rect(panel)),
            map_,
            units=options.units,
            n_levels=style.n_levels,
            cmap_positive=style.cmap_positive,
            cmap_signed=style.cmap_signed,
            data_range=(scale.lo, scale.hi) if style.freeze_clim else None,
            quantile=style.quantile,
            nice=style.nice,
            fill=style.fill,
            lines=style.lines,
            cax=fig.add_axes(geometry.cbar_rect(panel)),
            xlim=xlim,
            ylim=ylim,
        )
    fig.suptitle(
        _suptitle(series, frame, options.units),
        y=1.0 - 0.3 * _SUP_HEIGHT / geometry.fig_h,
        va="top",
    )
    return fig


def spectra_panels(prefix: str, marginal: str) -> list[tuple[str, int | None]]:
    """The four panels of a spectra figure (the paper's figure 11)."""
    name = f"{prefix}_{marginal}"
    return [(name, c) for c in range(len(COMPONENTS))] + [(name, None)]


def budget_panels(
    series: YSeries, marginal: str
) -> list[tuple[str, int | None]]:
    """Every stored budget term of one marginal, plus their sum."""
    return [(f"{t}_{marginal}", None) for t in (*series.terms, "sum")]


def apply_rcparams(usetex: bool, font_size: float = 11.0) -> None:
    """The write-up's matplotlib style (fonts, preamble, exact size).

    ``savefig.bbox`` is deliberately **not** ``"tight"``: the layout
    budgets its own margins in inches (:func:`panel_geometry`), and
    cropping them away would make the saved figure narrower than
    ``--width``.
    """
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "text.latex.preamble": LATEX_PREAMBLE if usetex else "",
            "font.size": font_size,
            "axes.titlesize": font_size * 0.9,
            "savefig.bbox": None,
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
    """``tag -> (stream, prefix-or-empty, marginal)`` for what is here."""
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
    declared_signs: bool = True,
    fmt: str = "png",
    pad: int | None = None,
    quiet: bool = False,
) -> list[Path]:
    """Render every frame of one series into ``out_dir/tag``.

    Filenames are ``<tag>_<index>.<fmt>`` with *index* the record's
    position on the members' common time grid, zero-padded so a
    lexical sort is the time order.  The series is scanned once first
    (:func:`scan_panels`) for the sign check and the frozen scale.
    """
    panels = (
        spectra_panels(prefix, marginal)
        if prefix
        else budget_panels(series, marginal)
    )
    scales, notes = scan_panels(
        series, panels, options, declared=declared_signs
    )
    if notes and not quiet:
        print("\n".join(notes), flush=True)

    target = out_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    width = pad or len(str(int(series.index.max())))
    written: list[Path] = []
    for frame in range(series.t_rel.size):
        fig = panel_figure(series, frame, panels, options, style, scales)
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
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--members",
        nargs="+",
        metavar="DIR",
        help="dnsjax-twin run directories to ensemble-average",
    )
    source.add_argument(
        "--tree",
        metavar="ROOT",
        help="ensemble_setup.py build-twin tree; every member is used",
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
    p.add_argument(
        "--premultiply",
        choices=("ky", "k", "none"),
        default="ky",
        help="premultiplier: both log axes, wavenumber only, or neither",
    )
    p.add_argument(
        "--clim",
        choices=("series", "frame"),
        default="series",
        help="colour scale frozen on the whole series, or per figure",
    )
    p.add_argument(
        "--signs-from-data",
        action="store_true",
        help="infer non-negativity instead of declaring it",
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
        "--fill",
        choices=("contour", "pcolormesh"),
        default="contour",
        help="filled contours, or one flat cell per sample",
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
    p.add_argument(
        "--width",
        type=float,
        default=PAGE_LINEWIDTH,
        help="figure width in inches (sets the decade length)",
    )
    p.add_argument(
        "--decade",
        type=float,
        default=None,
        help="inches per decade; overrides --width as the scale",
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
    members = (
        tree_members(args.tree)
        if args.tree
        else [Path(m) for m in args.members]
    )
    options = MapOptions(
        units=Units(args.re, args.re_tau, wall=not args.outer_units),
        premultiply=args.premultiply,
        half=args.half,
        volume_fac=not args.no_volume_fac,
        smooth=args.smooth,
    )
    style = PlotStyle(
        width=args.width,
        decade=args.decade,
        ncols=args.ncols,
        n_levels=args.levels,
        cmap_positive=args.cmap_positive,
        cmap_signed=args.cmap_signed,
        quantile=args.quantile,
        nice=not args.exact_levels,
        fill=args.fill,
        lines=not args.no_lines,
        freeze_clim=args.clim == "series",
        xlim=None if args.xlim is None else tuple(args.xlim),
        ylim=None if args.ylim is None else tuple(args.ylim),
        dpi=args.dpi,
    )

    opened: dict[str, YSeries | None] = {}
    for stem in STEMS:
        try:
            opened[stem] = open_series(
                members,
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
            f"{len(members)} member(s), Re = {args.re:g}, "
            f"Re_tau = {args.re_tau:g}, stride {args.stride}, "
            f"premultiply {args.premultiply}, half {args.half}, "
            f"clim {args.clim}, usetex {plt.rcParams['text.usetex']}",
            flush=True,
        )
    for tag in tags:
        stem, prefix, marginal = registry[tag]
        series = opened[stem]
        if series is None:  # pragma: no cover - the registry guards this
            continue
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
            declared_signs=not args.signs_from_data,
            fmt=args.format,
            pad=args.pad,
            quiet=args.quiet,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
