r"""Premultiplied `$(\lambda, y)$` maps of the twin `$(y, k)$` streams.

Renders one figure per recorded sample of ``twin_yspectra.bin`` and
``twin_ybudget.bin`` (:mod:`dnsjax.twin.yspectra`), ensemble-averaged
over a set of ``dnsjax-twin`` member directories, following the
premultiplied spectral maps of Cho, Hwang & Choi, *J. Fluid Mech.*
**854**, 474-504 (2018) -- their figures 3 and 11: wavelength on a
logarithmic abscissa, filled contours plus contour lines, inner units
on the primary axes and outer units on the secondary ones.  Where
these depart from the paper is the premultiplier: `$k$` alone, not
its `$k\,y$`, which is the commoner convention for a spectrum on a
logarithmic ordinate (next section).  ``--yscale linear`` swaps the
ordinate instead.

What is drawn
=============
Both wavenumber marginals of the **spectra** stream (``_z`` gives
`$\lambda_x$`; the paper shows only `$\lambda_z$`), and nothing else
unless asked.  Two sets are held back behind their own flag rather
than behind a tag name the caller has to know:

- ``--budget`` adds the ``twin_ybudget`` series, one figure per
  stored term;
- ``--x0`` adds the `$k_x = 0$` plane, where the stream has one --
  ``twin.x0_planes``, or any member recorded before that plane became
  opt-in.  It is a slice of the mode plane rather than a marginal of
  it, which is also why it is left absolute (below).

``--series`` overrides both: it names exact tags, and
:func:`available_series` lists what a given member set offers.  The
`$(0, 0)$` mode ``xz00`` is never a tag -- it is one mode, with no
abscissa to put it on -- and reaches the figures only as
`$E^{\mathrm{ref}}$`.

Premultiplication
=================
A stored entry is the energy (or rate) held by one **discrete** mode
band, not a spectral density: summing the entries over the stored
one-sided axis and contracting with ``y_weights`` returns the volume
average.  The density is therefore ``entry / dk`` with
`$\Delta k = 2\pi/L$`, and since `$k_m = m\,\Delta k$` for the integer
harmonic `$m$`, `$k\,\Phi = m \times \text{entry}$`, independent of the
box length.

A logarithmic axis needs a premultiplier if equal areas along it are
to be equal energy.  The abscissa always asks for one,

.. math::  k\,\Phi(y, k) = m \times \text{entry}(y, m) ,

which is ``--premultiply k``, the default and the near-universal
convention for these maps: a vertical cut then reads as the spectrum
*at* that wall distance, whose area over `$\log\lambda$` is the local
variance, and the ordinate is logarithmic to put the near-wall region
and the log layer in one frame rather than to be integrated over.

``ky`` adds the second factor -- the paper's (2.8),
`$\int\!\!\int \Phi\,\mathrm{d}k\,\mathrm{d}y = \int\!\!\int
k\,y\,\Phi\,\mathrm{d}\log k\,\mathrm{d}\log y$` -- which makes the
area of the *whole map* the energy rather than one wall distance's
variance.  That is the reading the paper's budget argument needs and
it is the minority one; it is a knob, not the default.  ``none``
drops both factors.

**Its `$y$` is the wall distance in the plotted units, so (2.8) is
exact only under ``--outer-units``.**  A premultiplier cancels the
units of what it multiplies -- `$k^+\Phi^+ = k\Phi$`, which is why
the `$m \times \text{entry}$` above serves a `$\lambda^+$` abscissa
and a `$\lambda/h$` one alike -- so the area-true pair takes `$k$`
and `$y$` in the **same** units:
`$k^+y^+\Phi^+ = k\,y\,\Phi/u_\tau^2$`, outer `$k$` and outer `$y$`.
What is plotted is that with `$y^+$` in place of `$y$`, so a
wall-unit ``ky`` map has area `$Re_\tau E^+$`, not `$E^+$`.  Its
shape, its axes and its contour spacing are the paper's; read an
*area* off ``--outer-units``, where the two factors agree.

The scale and the premultiplier are independent: pairing a linear
ordinate with ``ky``, or a logarithmic one with ``none``, is legal
and simply not area-true in `$y$`.

The `$m = 0$` column is dropped whatever the premultiplier, because
`$\lambda = L/m$` has no position on a wavelength axis.  That is also
what makes the maps read as fluctuation spectra: the wall-parallel
mean of a state lives at `$(0, 0)$` alone, so at every plotted
`$m \ge 1$` the stored perturbation-about-laminar spectrum ``r_*``
*is* the spectrum of the fluctuation about the `$x$`-`$z$` mean.

Stored entries are additionally divided by ``volume_fac`` (the
channel's wall-normal extent) so that contracting a profile with
``y_weights`` gives the volume average.  Multiplying it back gives the
**local** density at that `$y$`, which is what the literature plots;
that is :attr:`MapOptions.volume_fac` and it is on by default.  The
paper settles both halves of that convention: its (2.5) defines
`$\hat e = (|\hat u|^2 + |\hat v|^2 + |\hat w|^2)/2$`, so the
`$\tfrac12$` the writer carries is the standard one too, and its
(2.7)-(2.8) integrate `$\int_0^h \ldots \mathrm{d}y$` over the
physical wall distance, never over a channel-averaged one.  The
numbers agree: at plane-Poiseuille `$Re = 4200$`,
`$Re_\tau = 178.6$`, the `$k$`-premultiplied reference spectrum --
read before the next section's normalisation divides it through --
peaks at `$k_z E^{x+}_{uu} \approx 3.8$`, at `$y^+ \approx 14$`,
`$\lambda_z^+ \approx 130$`, the textbook near-wall peak.  Without
the factor every map is a factor of two low.

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
this selects the same samples as matching by index, but read off the
clock, which is what survives a member whose stream starts
elsewhere.  What the clock cannot tell is whether two members are
the same *flow*: their sidecars are therefore compared first, and a
set that disagrees on the grid, the mode axes or the stored meaning
is refused (:data:`_SHARED_KEYS`).

Two members meet on that clock to within a **tolerance**
(:data:`_T_ATOL`), never on a rounded key.  They reach one sample
time by different arithmetic -- accumulating from different parents
-- so their stored times differ in the last bits, and a pair
straddling a bin's edge rounds apart however fine the bin.  On this
grid that silently *drops* a frame, which is indistinguishable from
a member simply being short.  Any number of members works; ``--tree``
takes an
``ensemble_setup.py build-twin`` tree and uses every member its
``members.json`` lists.

Reference normalisation
=======================
The two true marginals of the spectra stream -- the difference
spectra ``e_x`` / ``e_z`` and the reference spectra ``r_x`` / ``r_z``
-- are drawn in units of the reference field's own fluctuation
energy,

.. math::
    E^{\mathrm{ref}}_\alpha = \bigl\langle R_\alpha
        - R^{00}_\alpha \bigr\rangle_t ,
    \quad R_\alpha = \sum_m \sum_j w_j\, r^{x}_\alpha(y_j, m) ,
    \quad R^{00}_\alpha = \sum_j w_j\, r^{xz00}_\alpha(y_j) ,

the total reference energy over `$y$` **and** `$k$` less its
`$(0, 0)$` mode, one number per component (and their sum for the
summed panel) for the whole series.  Both sums are wall-normal
**averages** and not integrals: `$\sum_j w_j$` is ``volume_fac``, by
which the stored entries are already divided.

The mean mode leaves the total because it is the wall-parallel mean,
common to both states of a pair and never decorrelating; it dominates
the total otherwise, a plane-Poiseuille `$R_u$` being some 89 % its
own `$(0, 0)$` mode, and every panel would be read against a number
that is mostly mean flow.  It comes off the stored `$(0, 0)$` mode
itself -- or, on a member recorded before that field existed, off
index 0 of the `$k_x = 0$` plane, which is the same number
(:func:`~dnsjax.analysis.twin.yspectra.mean_mode_name`).  What it is
*not* is ``r_x[..., 0]``, the whole `$k_z = 0$` column.

Dividing by a constant is a rescaling and nothing else: the shape of
a map is untouched and only the rounded level step
(:func:`nice_step`) so much as notices.  What it buys is a **common
denominator** -- the difference map and the reference map of one
component are then on one scale, and so are two runs at different
amplitudes -- and the reading that goes with it: two statistically
identical fields that have decorrelated completely differ by twice
the energy of either, mode by mode, so a saturated pair's ``e_*``
panel is twice its ``r_*`` panel.  Each panel's title carries its own
`$E^{\mathrm{ref}}$`, so the absolute values are a multiplication
away.  (The mode-by-mode sibling of this, each mode over its *own*
reference energy, is
:func:`dnsjax.analysis.twin.decorrelation_ratio`.)

The premultiplier and ``volume_fac`` reach the numerator alone,
exactly as they reach an absolute panel.  The energy unit conversion
would cancel between the two halves, so neither half takes it, and
``--outer-units`` therefore reaches these panels only through the
`$y$` of ``--premultiply ky`` -- the `$Re_\tau$` of
"Premultiplication" above, and here the whole difference between the
two unit systems -- and through the `$E^{\mathrm{ref}}$` the title
reports, which is in the units the rest of the figure is drawn in.

The `$k_x = 0$` panels, when ``--x0`` asks for them, are left
absolute.  ``e_x`` and ``e_z`` each sum the other wavenumber away, so
each is a complete sum over the mode plane and a fraction of
`$E^{\mathrm{ref}}$` is a fraction of a whole; the `$k_x = 0$` plane
is a slice of that plane instead, and normalising it by its own total
would cost exactly what the shared denominator buys.

`$E^{\mathrm{ref}}$` is averaged over **distinct reference
instants**: the members of an ensemble are subsampled from one long
turbulent run, so their reference halves are not independent --
members built from one parent snapshot carry bit-identical ``r_*``,
and members from different parents repeat each other wherever their
windows overlap.  Samples are therefore grouped on **absolute** time
(unlike the ensemble alignment above, which is relative) and each
instant counted once.  Grouped to the same tolerance as the frame
grid above and for the same reason, the cost of a rounded key here
being the mirror of the cost there: a straddling pair counted twice,
and the reference state it names double-weighted in the average.
Every record of every stream feeds that average, independently of
``--stride`` / ``--first`` / ``--last``; ``--ref-stride`` subsamples
it for a cheaper pass.

That key is exactly right for **one** reference trajectory and wrong
without it -- members subsampled from two *different* turbulent runs
would be merged wherever their absolute times met.  Nothing recorded
today separates them: ``twin.json`` carries ``parent`` and
``parent_t`` (the member's own start) and the harvest manifest the
source ``run_dir``, but neither is a signature of the trajectory, and
neither travels into the stream.  Until one does, the report prints
how many distinct parent snapshots are in play beside the instant
count, so a member set that is not one trajectory is at least visible.

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
checked for the symmetry the fold assumes rather than trusted --
which binds ``upper`` as well, since it labels its rows with the
*opposite* half's wall distances, but not ``lower``, where `$1 + y$`
is the wall distance whatever the far half does.
``--half lower`` / ``upper`` keep one wall instead, which is how a
run's own asymmetry is inspected.

Colour scales
=============
Non-negativity is **declared**, not inferred: the energies and the
pseudo-dissipation are sums of squares, which the division by a
positive `$E^{\mathrm{ref}}$` leaves them (:data:`NON_NEGATIVE`), so
those get the grey scale and everything else the diverging one.  The
declaration is asserted against the data once per series and a
negative excursion is reported with its size relative to the peak --
at round-off it is truncation and the map is drawn regardless;
anything larger is worth looking at, and the map is still drawn.
``--signs-from-data`` infers the sign instead, for a stream this list
does not cover.

Both fills are handed the *same* band colours, so ``--fill contour``
and ``--fill pcolormesh`` differ in geometry and in nothing else, and
a signed scale is **two-slope**: zero sits on the colour map's neutral
centre exactly and each side is stretched on its own, so the most
negative band is the darkest blue and the most positive the darkest
red however lopsided the range (:func:`band_colors`).

``--levels`` sets the level **step**, ``peak / levels``, and not the
level count, which is bounded by it rather than equal to it: at or
below it on a non-negative field, because the step is rounded up to a
round number, and up to twice it on a signed one, which spends that
step on both sides of zero (:func:`contour_levels`).

The `$y$` grid is the solver's own (CGL by default), and nothing here
assumes it is uniform: ``contourf`` / ``contour`` are handed the
coordinate arrays, so a contour lands at the wall distance it belongs
to.  Where the samples are sparse, ``--fill pcolormesh`` earns its
place: the same bands, one flat cell per sample on midpoint edges
(:func:`cell_edges` -- geometric on a logarithmic axis, arithmetic on
a linear one), with no interpolation between them.  Which end is
sparse follows the ordinate's scale: wall clustering makes the grid
coarse in `$\log y$` at the wall (its first plotted cell spans 0.6 of
a decade) and coarse in plain `$y$` at the centreline.  The contour
lines are drawn on top either way.

Every colour scale, per frame or frozen, is read off the **plotted**
quantity -- premultiplied, folded, in inner units, over exactly the
rows the axes box shows: the ordinate's floor and limits included,
not merely the wall row a logarithmic axis cannot place
(:meth:`Map.drawn`, :func:`y_limits`).  The colour bar therefore
labels the same numbers the contours do, which matters most for
`$\hat\varepsilon$`: its peak is at the wall, below the default
floor, and would otherwise set a scale no visible contour reaches.

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
The abscissa is sized by its decade count: the axes box is
`$\text{decade} \times D_\lambda$` wide for the `$D_\lambda$` decades
of the plotted limits, leaving only the scale free.  ``--width`` sets
it (default 6.61546 in, the write-up's ``\linewidth``) and
``--decade`` sets the decade length directly instead.

The height follows the **ordinate's scale**.  Logarithmic (the
default): the same decade length applies to it as well -- **one
decade, one length, on both axes**, the constraint that makes a
`$\lambda \propto y$` band read at 45 degrees -- and
``ax.set_aspect(1)`` holds it there, so the box's shape follows from
the limits alone.  The default floor at `$y^+ = 1$`
(:data:`Y_FLOOR_PLUS`) is most of what sets it: on
`$y^+ \in [1, 179]$` against `$\lambda_z^+ \in [14, 1122]$` the box
is 1.2 times taller than wide, where the grid's full
`$y^+ \in [0.02, 179]$` would make it 2.1 and an eight-panel budget
figure correspondingly tall.  ``--ylim`` trims it further.

Linear: the height is ``--box-aspect`` times the width, 1 (square) by
default, a linear axis having no decades to match.  ``set_aspect`` is
deliberately *not* applied there, where it would be matching decades
against data units.

Everything around the box is a fixed inch budget (the ``_M_*``
constants), with one thing free: the top margin grows by a line
(:data:`_TITLE_LINE`) for the two-line title a normalised panel
carries, so a figure never has to choose between the reported
`$E^{\mathrm{ref}}$` and its neighbour's tick labels.

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
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.ticker import FuncFormatter

from dnsjax.analysis.twin.yspectra import (
    MIN_YBUDGET_VERSION,
    MIN_YSPECTRA_VERSION,
    fluctuation_energy,
    mean_mode_name,
    mean_mode_profile,
    record_dtype,
    stored_suffixes,
)

#: Tolerance within which two sample times are held to be the same
#: instant -- of the members' shared frame grid (:func:`open_series`,
#: on relative time) and of the reference average
#: (:meth:`YSeries._distinct_instants`, on absolute time).
#:
#: Two members reach one instant by different arithmetic -- one
#: accumulates to it from its own parent, the other starts there --
#: so their stored times differ in the last bits.  A *rounded key*
#: will not do however fine: two values a picosecond apart still land
#: in different bins when they straddle one bin's edge, and the pair
#: is then two instants rather than one, which silently drops a frame
#: from the shared grid and double-weights a reference state.
#:
#: This sits three decades above the round-off it has to absorb (the
#: stored times accumulate order `$10^{-9}$`) and five below the
#: sampling cadence, ``it_* * dt``, of order 1 here.  That second side
#: matters as much: matching is transitive, so samples closer together
#: than this would chain into one instant.  :func:`_open_member`
#: refuses such a stream rather than let it.
_T_ATOL: float = 1e-6

#: The two streams this script understands, with their reader floors.
STEMS: dict[str, int] = {
    "twin_yspectra": MIN_YSPECTRA_VERSION,
    "twin_ybudget": MIN_YBUDGET_VERSION,
}

#: Sidecar keys every member of a set must agree on before their
#: streams may be averaged: the grid, the mode axes and the stored
#: meaning -- everything a figure reads off the **first** member alone
#: (:attr:`YSeries.meta`) and then applies to the ensemble mean.  This
#: is deliberately *not* the writers' ``_MATCH_KEYS``
#: (:mod:`dnsjax.twin.yspectra`), which answer a different question --
#: may this run append to that file -- and so pin ``dt``,
#: ``it_yspectra`` / ``it_ybudget`` and ``double_precision`` as well.
#: Here those three may differ: the members meet on a *clock*
#: (:func:`_match`), so a member on another cadence contributes the
#: samples it shares and no more, and every read is cast to float64
#: whatever ``value_dtype`` says.  ``twin`` (seed / ``e0`` /
#: smoothness) is what an ensemble varies and is never compared.
_SHARED_KEYS: tuple[str, ...] = (
    "format_version",
    "system",
    "ny",
    "n_kz",
    "n_kx",
    "kz_harmonics",
    "kx_harmonics",
    "lx",
    "lz",
    "y",
    "y_weights",
    "volume_fac",
)

#: Added to :data:`_SHARED_KEYS` per stream: what sets the field table,
#: and so what a stored name means, rather than the grid it lives on.
#: ``suffixes`` is normalised onto every member's sidecar by
#: :func:`_open_member`, so a set of pre-``xz00`` members compares on
#: the legacy triple rather than on a key none of them has -- and a
#: set that mixes layouts is refused by name.
_STREAM_KEYS: dict[str, tuple[str, ...]] = {
    "twin_yspectra": ("includes_ref", "suffixes"),
    "twin_ybudget": ("terms", "suffixes"),
}

#: Velocity components of the ``twin_yspectra`` leading axis.
COMPONENTS: tuple[str, ...] = ("u", "v", "w")

#: Stored suffixes whose panels are drawn relative to
#: `$E^{\mathrm{ref}}$` (module docstring, "Reference
#: normalisation").  ``x0`` is deliberately absent: it is a slice of
#: the mode plane, not a complete sum over it.
NORMALISED_MARGINALS: frozenset[str] = frozenset({"x", "z"})

#: Default bottom of a **logarithmic** ordinate, in wall units.  The
#: grid reaches far below it (`$y^+ \approx 0.02$` at the resolutions
#: these runs use), and nothing but `$\hat\varepsilon$` reaches its
#: first contour level down there, so the decade below `$y^+ = 1$`
#: buys a taller box and no information.  A linear ordinate keeps the
#: wall itself, which is a position it can show.
Y_FLOOR_PLUS: float = 1.0

#: Records read from a memory-mapped stream at a time while the
#: reference normalisation is accumulated.  Each is reduced to three
#: numbers immediately, so this bounds that pass's memory.
_REF_CHUNK: int = 64

#: Fields that are non-negative **by construction**, keyed by the base
#: name: the two spectra prefixes are `$\tfrac12|\hat u|^2$` sums,
#: which a division by a positive `$E^{\mathrm{ref}}$` leaves them,
#: and ``eps`` is `$\nu(|\partial_y\hat u|^2 + k^2|\hat u|^2)$`.
#: ``V`` is deliberately absent -- the operator (discrete-Laplacian)
#: viscous form is *not* sign-definite, as
#: :mod:`dnsjax.twin.diagnostics` ("Dissipation form") sets out,
#: however negative it happens to come out in any given run.
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

#: ``(wavelength axis, energy superscript)`` per **drawable** stored
#: suffix.  A suffix names the axis that was **summed over**, so
#: ``_x`` is the `$k_z$` marginal and its abscissa is
#: `$\lambda_z$`.  The axis letter is what both the panel title and
#: the two abscissa labels subscript, which is why it is stored rather
#: than a ready-made ``k_z``.
#:
#: ``xz00`` is deliberately absent: it is one mode, with no wavenumber
#: axis to put on an abscissa.  It reaches the figures only through
#: `$E^{\mathrm{ref}}$`.  ``x0`` is here but is **not** drawn by
#: default (:data:`DEFAULT_MARGINALS`); only a pre-``xz00`` stream or
#: a run under ``twin.x0_planes`` carries it at all.
MARGINALS: dict[str, tuple[str, str]] = {
    "x": ("z", "x"),
    "z": ("x", "z"),
    "x0": ("z", "x0"),
}

#: The marginals rendered unless ``--x0`` asks for the rest, and the
#: streams rendered unless ``--budget`` does.  Both defaults are about
#: what is worth looking at rather than what is on disk: the two true
#: marginals of the difference and reference spectra.
DEFAULT_MARGINALS: frozenset[str] = frozenset({"x", "z"})
DEFAULT_STEMS: frozenset[str] = frozenset({"twin_yspectra"})

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
_M_TOP: float = 0.78  # top tick labels + secondary label + one title line
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

#: What each title line beyond the first adds to ``_M_TOP``, in
#: inches at the default font size: a panel normalised by
#: `$E^{\mathrm{ref}}$` reports it on a second line, and ``_M_TOP``
#: has no slack to absorb one.
_TITLE_LINE: float = 0.17

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

    The eager reader's own
    :func:`dnsjax.analysis.twin.yspectra.record_dtype` -- shared
    rather than mirrored, so the two cannot disagree about which
    layout a sidecar describes.  What this script needs it for is the
    memory map: the dtype is what lets a record be read without the
    whole-file pass :func:`~dnsjax.analysis.twin.yspectra.read_twin_yspectra`
    makes.
    """
    return record_dtype(meta, stem)


@dataclass(frozen=True)
class _Member:
    """One member's memory-mapped stream, deduplicated in time."""

    path: Path
    meta: dict
    records: np.memmap
    rows: np.ndarray  # record indices, ascending, unique in t
    t_abs: np.ndarray  # their absolute simulation times
    t_rel: np.ndarray  # the same, since the perturbation
    parent: str  # its parent snapshot, "" when unrecorded


def _check_cadence(path: Path, times: np.ndarray) -> None:
    """A stream's own samples must be ascending and well separated.

    Both consumers of these times -- the shared frame grid and the
    reference average -- match them to :data:`_T_ATOL`, and matching
    is transitive, so two samples closer than that would be one
    instant.  Ascending order is the other assumption: the streams are
    append-only, and both consumers reach a member's records through
    ``searchsorted``, which is silent on a key that is not sorted and
    would pair a frame with the wrong record.  Neither is trusted.
    """
    gap = float(np.min(np.diff(times))) if times.size > 1 else np.inf
    if gap <= 0.0:
        raise ValueError(f"{path}: sample times are not sorted ascending")
    if gap <= _T_ATOL:
        raise ValueError(
            f"{path}: samples are {gap:g} apart in time, at or below "
            f"the {_T_ATOL:g} tolerance that decides whether two of "
            "them are the same instant; distinct instants would be "
            "chained into one."
        )


def _twin_record(path: Path) -> dict:
    """A member's parsed ``twin.json``, or ``{}`` when it has none.

    Two fields are read off it.  ``parent_t`` is the member's
    `$t_\\mathrm{parent}$`, the clock the ensemble aligns on -- what
    :mod:`dnsjax.analysis.twin.series` uses, and the only one that is
    right for a member whose stream begins at a resume rather than at
    the perturbation; the stream's first sample stands in when the
    record is absent.  ``parent`` names the snapshot the reference
    trajectory was picked up from, which the reference normalisation
    reports (module docstring, "Reference normalisation").
    """
    record = path / "twin.json"
    if not record.is_file():
        return {}
    with open(record) as fh:
        return json.load(fh)


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
    # Written in so the per-member sidecar comparison has a key to
    # compare (_STREAM_KEYS); a pre-``xz00`` sidecar has none.
    meta["suffixes"] = list(stored_suffixes(meta))
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
    t_abs = t[rows]
    _check_cadence(path, t_abs)
    record = _twin_record(path)
    parent_t = record.get("parent_t")
    t0 = float(t_abs[0]) if parent_t is None else float(parent_t)
    return _Member(
        path,
        meta,
        records,
        rows,
        t_abs,
        t_abs - t0,
        str(record.get("parent", "")),
    )


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
    ref_stride: int = 1  # subsampling of the reference normalisation
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _reference: tuple[np.ndarray, list[str]] | None = field(
        default=None, repr=False
    )

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
    def suffixes(self) -> tuple[str, ...]:
        """Stored marginals, normalised onto the sidecar on open."""
        return tuple(self.meta["suffixes"])

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

    def reference_scale(self) -> np.ndarray:
        r"""`$E^{\mathrm{ref}}_\alpha$` per component, cached.

        The reference field's total-in-`$(y, k)$` energy without its
        `$(0, 0)$` mode, averaged over the **distinct** reference
        instants of the member set: the one number every normalised
        panel of a component is divided by (module docstring,
        "Reference normalisation").
        """
        return self._resolved_reference()[0]

    def reference_report(self) -> list[str]:
        """The lines describing that normalisation, for printing."""
        return self._resolved_reference()[1]

    def _resolved_reference(self) -> tuple[np.ndarray, list[str]]:
        """The reference normalisation and its report, built once."""
        if self._reference is None:
            self._reference = self._build_reference()
        return self._reference

    def _build_reference(self) -> tuple[np.ndarray, list[str]]:
        """Accumulate the reference normalisation over the members."""
        if "r" not in self.prefixes:
            raise ValueError(
                f"{self.stem}: the stream carries no reference spectra "
                "(twin.spectra_ref was off), so there is no E_ref to "
                "normalise the difference spectra by."
            )
        picks, n_instants, n_samples = self._distinct_instants()
        mean_name = mean_mode_name(self.meta, "r")
        total = np.zeros(len(COMPONENTS))
        for member, rows in zip(self.members, picks, strict=True):
            if rows.size == 0:  # every instant is another member's
                continue
            self._check_marginals(member, int(rows[0]))
            for start in range(0, rows.size, _REF_CHUNK):
                take = rows[start : start + _REF_CHUNK]
                total += fluctuation_energy(
                    np.asarray(member.records["r_x"][take], dtype=np.float64),
                    mean_mode_profile(
                        np.asarray(
                            member.records[mean_name][take], dtype=np.float64
                        ),
                        mean_name,
                    ),
                    self.y_weights,
                ).sum(axis=0)
        scale = total / n_instants
        if not np.all(scale > 0.0):
            raise ValueError(
                "the reference fluctuation energy is not positive in "
                f"every component ({scale.tolist()}); a spectrum "
                "cannot be normalised by it."
            )
        parents = {m.parent for m in self.members if m.parent}
        report = [
            f"reference normalisation, {self.n_members} member(s): "
            f"{n_instants} distinct instants of {n_samples} samples"
            + (f", stride {self.ref_stride}" if self.ref_stride > 1 else "")
            + f"; {len(parents) or 'unrecorded'} parent snapshot(s)",
            "  E_ref = "
            + "  ".join(
                f"{c} {v:.6g}" for c, v in zip(COMPONENTS, scale, strict=True)
            )
            + f"  sum {scale.sum():.6g}",
        ]
        return scale, report

    def _distinct_instants(self) -> tuple[list[np.ndarray], int, int]:
        r"""Record indices covering each reference instant once.

        Returns one ascending index array per member -- together they
        name every distinct instant exactly once -- with the instant
        and sample counts.  Samples are grouped by **proximity in
        absolute time** rather than by a rounded key: the two are the
        same until a pair straddles a bin edge, where a key splits
        them and a tolerance does not (:data:`_T_ATOL`).  Whichever
        member sorts first owns a shared instant; they hold the same
        reference state, so it does not matter which.

        The grouping is transitive, so the tolerance has to stay well
        below the sampling cadence; :func:`_check_cadence` has already
        refused a stream where it does not.
        """
        times = [m.t_abs[:: self.ref_stride] for m in self.members]
        rows = [m.rows[:: self.ref_stride] for m in self.members]
        flat = np.concatenate(times)
        owner = np.concatenate(
            [np.full(t.size, i, dtype=int) for i, t in enumerate(times)]
        )
        index = np.concatenate([np.arange(t.size) for t in times])
        order = np.argsort(flat, kind="stable")
        opening = np.ones(order.size, dtype=bool)
        opening[1:] = np.diff(flat[order]) > _T_ATOL
        keep, owned = index[order[opening]], owner[order[opening]]
        picks = [
            np.sort(member_rows[keep[owned == i]])
            for i, member_rows in enumerate(rows)
        ]
        return picks, int(opening.sum()), int(flat.size)

    def _check_marginals(self, member: _Member, row: int) -> None:
        """Both marginals must report the same reference total.

        ``r_x`` sums over `$k_x$` and ``r_z`` over `$k_z$`, so each is
        already a complete one-sided sum over the mode plane and the
        two are an independent reading of the same number -- a
        mismatch is a convention slip, not noise.  Checked once per
        member, on the first record its instants contribute.
        """
        block = member.records[row]
        mean_name = mean_mode_name(member.meta, "r")
        mean = mean_mode_profile(
            np.asarray(block[mean_name], dtype=np.float64), mean_name
        )
        by_x, by_z = (
            fluctuation_energy(
                np.asarray(block[name], dtype=np.float64),
                mean,
                self.y_weights,
            )
            for name in ("r_x", "r_z")
        )
        tol = 1e-9 if member.meta["value_dtype"] == "<f8" else 1e-4
        if not np.allclose(
            by_x, by_z, rtol=tol, atol=tol * float(np.max(np.abs(by_x)))
        ):
            raise ValueError(
                f"{member.path}: the k_z and k_x marginals disagree on "
                f"the reference energy at t = {member.records['t'][row]:g} "
                f"({by_x.tolist()} vs {by_z.tolist()}); one of them is "
                "not a complete sum over the mode plane."
            )


def _sidecar_mismatch(
    first: dict, other: dict, keys: tuple[str, ...]
) -> list[str]:
    """Which of *keys* two members' sidecars disagree on.

    The wall-normal grid and its weights are compared to the same
    ``1e-12`` the driver uses when it re-derives a grid it already
    holds (``twin/driver.py``), so a member written by another build
    is not refused over a last-bit difference; everything else,
    including the integer harmonic lists, is compared exactly.
    """
    bad: list[str] = []
    for key in keys:
        a, b = first.get(key), other.get(key)
        if key in ("y", "y_weights"):
            same = np.shape(a) == np.shape(b) and np.allclose(
                np.asarray(a, dtype=np.float64),
                np.asarray(b, dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            )
        else:
            same = a == b
        if not same:
            bad.append(key)
    return bad


def _match(times: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Index in ascending *times* of each *wanted* value, or ``-1``.

    Nearest neighbour within :data:`_T_ATOL`.  A tolerance rather than
    an equality on rounded keys: two members reach one sample time by
    different arithmetic, so a pair that straddles a bin's edge would
    round apart and drop the frame from the shared grid -- silently,
    since a shorter grid is exactly what a short member produces.
    Both arrays are ascending and separated by more than the
    tolerance (:func:`_check_cadence`), so the match is unambiguous.
    """
    if times.size == 0:
        return np.full(wanted.shape, -1, dtype=int)
    after = np.searchsorted(times, wanted)
    before = np.clip(after - 1, 0, times.size - 1)
    after = np.clip(after, 0, times.size - 1)
    nearer = np.abs(times[before] - wanted) <= np.abs(times[after] - wanted)
    nearest = np.where(nearer, before, after)
    return np.where(np.abs(times[nearest] - wanted) <= _T_ATOL, nearest, -1)


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
    ref_stride: int = 1,
) -> YSeries:
    """Open one stream across *members* on their common time grid.

    *stride* keeps every *stride*-th record (the "subsample by 10" of
    a long run), *first* / *last* clip the common grid before that.
    Members are aligned on time since the perturbation (module
    docstring, "Ensemble averaging"); any number of them works.

    *ref_stride* subsamples the reference normalisation instead, and
    is the only one of the four that does not select what is drawn:
    that average runs over every record either way, on absolute time
    (module docstring, "Reference normalisation").

    Members that do not agree on the grid, the mode axes or the
    stored meaning (:data:`_SHARED_KEYS`) are **refused**: a figure
    reads all three off the first member and would otherwise label an
    average of incommensurate streams with one member's axes.
    """
    if stem not in STEMS:
        raise ValueError(f"unknown stream {stem!r}; expected {set(STEMS)}")
    if not members:
        raise ValueError("need at least one member directory")
    opened = [_open_member(Path(m), stem) for m in members]
    keys = _SHARED_KEYS + _STREAM_KEYS[stem]
    for member in opened[1:]:
        differs = _sidecar_mismatch(opened[0].meta, member.meta, keys)
        if differs:
            raise ValueError(
                f"{member.path}: its {stem}.json disagrees with "
                f"{opened[0].path}'s on {', '.join(differs)}, so these "
                "members are not one ensemble and their streams cannot "
                "be averaged -- the grid, the mode axes and the panel "
                "labels are all read off the first member alone."
            )

    # The shared grid is the first member's own times, thinned to
    # those every other member also has (:func:`_match`); its own
    # record positions are then the frame index the filenames carry.
    grids = [member.t_rel for member in opened]
    keep = np.arange(grids[0].size)
    for other in grids[1:]:
        keep = keep[_match(other, grids[0][keep]) >= 0]
    if keep.size == 0:
        raise ValueError("members share no relative sample time")
    selected = keep[first : (None if last is None else last + 1)][::stride]
    if selected.size == 0:
        raise ValueError(
            f"first={first} / last={last} / stride={stride} select none "
            f"of the {keep.size} sample time(s) the members share."
        )
    keep = selected
    common = grids[0][keep]
    rows = np.stack(
        [
            member.rows[_match(grid, common)]
            for member, grid in zip(opened, grids, strict=True)
        ]
    )
    return YSeries(
        stem=stem,
        members=tuple(opened),
        rows=rows,
        index=keep,
        t_rel=common,
        meta=opened[0].meta,
        ref_stride=ref_stride,
    )


# ── Map construction ─────────────────────────────────────────────────


@dataclass(frozen=True)
class MapOptions:
    r"""Everything that turns a stored field into a plotted map.

    *premultiply* is ``"k"`` (the default), ``"ky"`` (the paper's
    (2.8)) or ``"none"``, and *y_log* draws the ordinate logarithmic
    (the default); the two are independent, and the default pair is
    the usual convention for a spectrum rather than the paper's own
    (module docstring, "Premultiplication").  *half* is how the two
    channel halves collapse onto the one wall distance a `$y^+$` axis
    needs (module docstring, "Folding the channel"); *volume_fac*
    multiplies the stored `$y$`-mean density back to the local density
    the literature plots.  All three reach a normalised spectra panel
    exactly as they reach an absolute one: the normalisation divides
    the numerator by a constant and stops there (module docstring,
    "Reference normalisation").

    *smooth* is a centred running mean over that many adjacent
    wavenumbers, applied last.  It is off (``1``) by default and is
    presentation only: a few-member ensemble mean of an
    *instantaneous* field is genuinely rough mode to mode -- unlike
    the paper's long-time averages of a stationary flow -- and the
    transfer terms show it.  Widening this bins the map; it does not
    denoise it.
    """

    units: Units
    premultiply: str = "k"
    half: str = "mean"
    volume_fac: bool = True
    smooth: int = 1
    y_log: bool = True


@dataclass(frozen=True)
class Map:
    r"""One panel's data: ``values`` on the `$(\lambda, y)$` grid."""

    lam: np.ndarray  # (n_lam,) wavelength, plotted units
    y: np.ndarray  # (n_y,) wall distance, plotted units
    values: np.ndarray  # (n_y, n_lam)
    title: str  # LaTeX panel title, with normalisation
    name: str  # the stored field it came from
    non_negative: bool  # declared or inferred; sets the colour family
    y_log: bool = False  # whether the ordinate is drawn logarithmic

    @property
    def lam_axis(self) -> str:
        r"""Which wavelength the abscissa is: ``x`` or ``z``.

        Read off the stored suffix of :attr:`name`, the same way
        :func:`field_title` reads the panel title's wavenumber, so a
        panel and its axes cannot label different marginals.
        """
        return MARGINALS[self.name.rpartition("_")[2]][0]

    def drawn(
        self, ylim: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""The rows the ordinate shows, optionally within *ylim*.

        The wall row has no position on a **logarithmic** axis, so it
        is dropped there -- from the plot and therefore from the
        colour scale too, which is why :func:`scan_panels` and
        :func:`draw_map` both go through here rather than reading
        ``values`` directly.  It matters under ``--premultiply k`` /
        ``none``, where the wall value of a budget term is not zero
        (`$\hat\varepsilon$` is largest there); under ``ky`` the
        `$y$` factor zeroes that row anyway.  On a linear ordinate the
        row is an ordinary sample and is kept.

        *ylim* additionally drops what falls outside the axis box,
        which is how the colour scale stays a scale of what is
        *visible* once the ordinate has a floor (:data:`Y_FLOOR_PLUS`)
        rather than of every stored row.  One row is kept beyond each
        end: the fill interpolates between samples, so the pair
        straddling a limit still colours the strip inside it.
        :func:`draw_map` calls both forms: the restricted rows set its
        levels, and the unrestricted ones are what it draws, so a
        contour still reaches the edge of the box.
        """
        if self.y_log:
            shown = self.y > 0.0
        else:
            shown = np.ones(self.y.size, dtype=bool)
        keep = shown.copy()
        if ylim is not None:
            keep &= (self.y >= ylim[0]) & (self.y <= ylim[1])
            inside = np.flatnonzero(keep)
            if inside.size:  # the interpolation neighbours, if any
                keep[max(int(inside[0]) - 1, 0)] = True
                keep[min(int(inside[-1]) + 1, keep.size - 1)] = True
            keep &= shown
        return self.y[keep], self.values[keep]


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
    wall_distance = _half_grid(y, mode)
    n_half = wall_distance.size
    lower = values[..., :n_half, :]
    upper = values[..., ::-1, :][..., :n_half, :]
    if mode == "mean":
        kept = 0.5 * (lower + upper)
    elif mode == "lower":
        kept = lower
    else:
        kept = upper
    return kept, wall_distance


def _half_grid(y: np.ndarray, mode: str) -> np.ndarray:
    r"""The wall distances a fold retains, ascending from the wall.

    Split out of :func:`_select_half` because the ordinate's limits
    are a property of the grid alone (:func:`y_limits`), settled once
    for a series rather than per panel and per frame.
    """
    if mode not in ("mean", "lower", "upper"):
        raise ValueError(f"half must be mean/lower/upper, not {mode!r}")
    if y.size > 1 and y[0] >= y[-1]:
        raise ValueError(
            "the wall-normal grid is not ascending, so 1 + y is not the "
            "wall distance; the streams carry the solver's own grid, "
            "which runs from the lower wall up."
        )
    if mode != "lower" and not np.allclose(y, -y[::-1], rtol=0.0, atol=1e-12):
        raise ValueError(
            "the wall-normal grid is not symmetric about the centreline, "
            "so the R_y fold has no partner row to average against, and "
            "--half upper no wall distance to label its rows with; use "
            "--half lower, which needs neither."
        )
    return 1.0 + y[: (y.size + 1) // 2]


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


def y_limits(
    series: YSeries,
    options: MapOptions,
    ylim: tuple[float, float] | None = None,
) -> tuple[float, float]:
    r"""The ordinate limits every panel of a series shares.

    ``--ylim`` when it is given, and otherwise the folded grid's own
    range -- floored at :data:`Y_FLOOR_PLUS` on a logarithmic
    ordinate, converted into whatever units the ordinate is drawn in
    so that the floor is the same wall distance either way.  The floor
    never *extends* the axis past the data: a grid whose first point
    is already above it keeps that point.

    Settled from the grid alone, so :func:`scan_panels` can read a
    colour scale over exactly the rows the box will show and
    :func:`panel_figure` can size the box from the same numbers.
    """
    if ylim is not None:
        return ylim
    y = options.units.length(_half_grid(series.y, options.half))
    top = float(y.max())
    if not options.y_log:
        return float(y.min()), top
    above_wall = float(y[y > 0.0].min())
    floor = options.units.length(Y_FLOOR_PLUS / options.units.re_tau)
    return max(above_wall, floor), top


def field_kind(series: YSeries) -> str:
    """``"energy"`` or ``"rate"``, by which stream *series* is."""
    return "energy" if series.stem == "twin_yspectra" else "rate"


def declared_non_negative(name: str) -> bool:
    """Whether :data:`NON_NEGATIVE` covers a stored (or virtual) name."""
    return name.rpartition("_")[0] in NON_NEGATIVE


def normalises(series: YSeries, name: str) -> bool:
    r"""Whether a field is drawn relative to `$E^{\mathrm{ref}}$`.

    The two complete marginals of a spectra stream that carries its
    reference half, and nothing else: a budget term is not an energy
    (and names no prefix, so a budget stream is excluded by the same
    test), the `$k_x = 0$` plane is not a complete sum over the mode
    plane, and a stream written without ``twin.spectra_ref`` has no
    `$E^{\mathrm{ref}}$` to offer (module docstring, "Reference
    normalisation").  Cheap, so a caller can ask before paying for
    :meth:`YSeries.reference_scale`.
    """
    base, _, suffix = name.rpartition("_")
    return (
        base in series.prefixes
        and suffix in NORMALISED_MARGINALS
        and "r" in series.prefixes
    )


def reference_norm(
    series: YSeries, name: str, component: int | None
) -> float | None:
    r"""The `$E^{\mathrm{ref}}$` one panel divides by, or ``None``.

    The summed panel takes the summed reference, so it is one ratio
    of sums rather than a sum of three ratios.
    """
    if not normalises(series, name):
        return None
    scale = series.reference_scale()
    return float(scale.sum() if component is None else scale[component])


def reference_symbol(component: int | None) -> str:
    r"""LaTeX for the `$E^{\mathrm{ref}}$` one panel divides by."""
    ref = r"E^{\mathrm{ref}}"
    if component is None:
        return rf"\sum_\alpha {ref}_{{\alpha}}"
    return rf"{ref}_{{{COMPONENTS[component]}}}"


def latex_float(value: float, digits: int = 4) -> str:
    """*value* to *digits* significant figures, as LaTeX math.

    ``%g``'s exponent is spelled out, so a title reads
    `$1.234 \\times 10^{-3}$` rather than ``1.234e-03``.
    """
    text = f"{value:.{digits}g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def field_title(
    series: YSeries,
    name: str,
    component: int | None,
    options: MapOptions,
) -> str:
    r"""The LaTeX panel title for one stored (or virtual) field.

    A normalised panel (:func:`normalises`) gets a second line
    carrying its `$E^{\mathrm{ref}}$` in the units the figure is
    drawn in, which is what makes its colour bar recoverable as an
    absolute one; :func:`panel_geometry` budgets the extra line.
    """
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
    scale = reference_norm(series, name, component)
    if scale is None:
        return f"${factor}{body}{options.units.norm_suffix(kind)}$"
    ref = reference_symbol(component)
    return (
        f"${factor}{body}/{ref}$\n"
        f"${ref}{options.units.norm_suffix(kind)} = "
        f"{latex_float(options.units.energy(scale))}$"
    )


def make_map(
    series: YSeries,
    name: str,
    frame: int,
    *,
    options: MapOptions,
    component: int | None = None,
    non_negative: bool | None = None,
) -> Map:
    r"""Build one premultiplied map from a stored (or virtual) field.

    *name* is a stored field such as ``e_x`` / ``P_U_z``, or the
    virtual ``sum_x``; *component* selects a velocity component of a
    ``twin_yspectra`` field (``None`` sums the three).  *frame*
    indexes the series' subsampled records.  *non_negative* overrides
    the declaration of :data:`NON_NEGATIVE` -- what
    ``--signs-from-data`` and :func:`scan_panels` feed back in.

    A spectra panel of a complete marginal is divided through by the
    series' `$E^{\mathrm{ref}}$` (:func:`reference_norm`), **after**
    the component reduction so that the summed panel is one ratio of
    sums rather than a sum of three ratios.
    """
    suffix = name.rpartition("_")[2]
    values = series.field(name)[frame]
    if series.stem == "twin_yspectra":
        values = values.sum(axis=0) if component is None else values[component]
    elif component is not None:
        raise ValueError(f"{series.stem}: {name} has no component axis")

    values = values[:, 1:]  # lambda = L/m has no place at m = 0
    if options.premultiply != "none":
        values = values * series.harmonics(suffix)[None, 1:]
    if options.volume_fac:
        values = values * series.volume_fac
    scale = reference_norm(series, name, component)
    if scale is None:
        values = options.units.convert(values, field_kind(series))
    else:
        # An energy over an energy: the unit conversion cancels
        # between the two halves, so neither half takes it and the
        # title reports E_ref in the plotted units instead (module
        # docstring, "Reference normalisation").
        values = values / scale
    values = values[:, ::-1]  # ascending in wavelength, as the axis is

    values, wall_distance = _select_half(values, series.y, options.half)
    y = options.units.length(wall_distance)
    if options.premultiply == "ky":
        # A second logarithmic axis needs its own premultiplier, in
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
        y_log=options.y_log,
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
    ylim: tuple[float, float] | None = None,
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

    *ylim* restricts the scan to the rows the axes box will show
    (:meth:`Map.drawn`), which is what keeps the colour bar a legend
    for the visible map rather than for a near-wall peak the ordinate
    floors away.
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
            ).drawn(ylim)
            finite = values[np.isfinite(values)]
            if finite.size:
                lo = min(lo, float(finite.min()))
                hi = max(hi, float(finite.max()))
        if not math.isfinite(lo):
            lo = hi = 0.0
        label = (
            name if component is None else f"{name}[{COMPONENTS[component]}]"
        )
        non_negative = declared_non_negative(name) if declared else lo >= 0.0
        if declared and non_negative:
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
) -> np.ndarray:
    """Contour levels for one map.

    A non-negative field gets bands from one step up to the peak,
    leaving the lowest band unfilled so the empty corners of the map
    stay white (the paper's convention).  A signed field gets the same
    step reflected about zero and trimmed to the data range; which
    colour each of those bands is then given, and why zero lands on
    the neutral one however lopsided the trim, is
    :func:`band_colors`.

    *n_levels* sets the **step**, ``peak / n_levels``, not the level
    count: it is how many bands reach from zero to the larger of the
    two extremes.  The count that comes out is therefore only bounded
    by it -- at or below on a non-negative field (``nice`` rounds the
    step up, never down, by a factor under two), and up to twice it on
    a signed one, which spends that step on both sides of zero.

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
        return np.asarray([], dtype=float)
    peak = max(abs(lo), abs(hi))
    if quantile is not None and finite.size:
        peak = float(np.quantile(np.abs(finite), quantile))
    if peak <= 0.0:
        return np.asarray([], dtype=float)

    step = peak / max(n_levels, 2)  # a filled band needs two levels
    if nice:
        step = nice_step(step)
    if non_negative:
        return np.arange(1, math.ceil(peak / step) + 1) * step
    below = min(math.floor(lo / step), -1)
    above = max(math.ceil(hi / step), 1)
    return (
        np.concatenate([np.arange(below, 0), np.arange(1, above + 1)]) * step
    )


def band_colors(
    levels: np.ndarray, cmap: str, *, non_negative: bool
) -> tuple[ListedColormap, BoundaryNorm]:
    r"""One colour per filled band, and the norm that selects it.

    ``contourf`` colours a band by its **midpoint** and ``pcolormesh``
    by the interval a value falls in, so the two agree only if they
    are handed the same table: a :class:`~matplotlib.colors.
    ListedColormap` holding one colour per band, indexed by a
    :class:`~matplotlib.colors.BoundaryNorm` on *levels*.  Then
    ``--fill contour`` and ``--fill pcolormesh`` differ in geometry
    and in nothing else.

    The colours themselves are read off the band midpoints through a
    *value*-linear norm, so intensity still tracks magnitude within a
    side:

    - non-negative: white at zero to the darkest colour at the top
      band, and everything below the first level is left transparent
      (the deliberately unfilled lowest band);
    - signed: **two-slope** -- zero sits exactly on the colour map's
      neutral centre, and each side is scaled on its own so that the
      most negative band is the darkest blue and the most positive the
      darkest red.  That is deliberately *not* symmetric in intensity:
      a one-sided term such as `$\mathcal{P}^{U}$`, whose negative
      excursion is a percent of its positive one, would otherwise
      spend the entire blue half of the colour map on a single band
      and read as unsigned.

    A signed field that never changes sign is the same statement with
    one side empty, and gets the whole ramp of the side it does use.
    Note that "neutral" is the colour map's own centre, which for the
    default ``RdBu_r`` is ColorBrewer's near-white ``#f7f6f6`` rather
    than the page; ``--cmap-signed bwr`` centres on pure white.
    """
    base = plt.get_cmap(cmap)
    layers = 0.5 * (levels[:-1] + levels[1:])  # what contourf colours
    if non_negative:
        scale = Normalize(0.0, float(layers[-1]))
    else:
        lo, hi = min(float(layers[0]), 0.0), max(float(layers[-1]), 0.0)
        if lo == 0.0 and hi == 0.0:  # the zero-straddling band alone
            scale = Normalize(-1.0, 1.0)
        else:
            span = max(-lo, hi)
            scale = TwoSlopeNorm(
                vcenter=0.0,
                vmin=lo if lo < 0.0 else -span,
                vmax=hi if hi > 0.0 else span,
            )
    shaded = ListedColormap(base(scale(layers)))
    shaded.set_under((1.0, 1.0, 1.0, 0.0))  # below the first level
    shaded.set_over(shaded(shaded.N - 1))  # only --quantile reaches it
    return shaded, BoundaryNorm(levels, shaded.N)


def cell_edges(centres: np.ndarray, *, log: bool) -> np.ndarray:
    r"""Cell edges around *centres*: the midpoints between them.

    ``pcolormesh`` wants edges, and the edge between two samples is
    their arithmetic mean on a linear axis and their **geometric**
    mean on a logarithmic one -- the difference is visible in the
    near-wall cells of a `$\log y$` ordinate, where the CGL grid is
    coarsest in that measure (its first plotted cell spans 0.6 of a
    decade).  The two outermost edges are extrapolated by the same
    half-step, so a linear ordinate's first edge falls *inside* the
    wall; the axis limits clip it back.
    """
    if log:
        mid = np.sqrt(centres[:-1] * centres[1:])
        return np.concatenate(
            [[centres[0] ** 2 / mid[0]], mid, [centres[-1] ** 2 / mid[-1]]]
        )
    mid = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate(
        [[2.0 * centres[0] - mid[0]], mid, [2.0 * centres[-1] - mid[-1]]]
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
    blue-white-red scale otherwise.  The abscissa is logarithmic and
    the ordinate follows ``map_.y_log``; when it is logarithmic too,
    ``set_aspect(1)`` holds one decade to the same length on each --
    the layout of :func:`panel_geometry` sizes the box so that costs
    nothing.  *cax* is where the colour bar goes (``None``: no bar).

    *ylim* both limits the axis and restricts the rows the **levels**
    are read from (:meth:`Map.drawn`), so the colour bar is a legend
    for the visible map whether the scale is frozen (*data_range*,
    already restricted by :func:`scan_panels`) or taken from this
    frame.  Everything inside the box is still drawn from the
    unrestricted array.
    """
    ax.set_xscale("log")
    ax.set_yscale("log" if map_.y_log else "linear")
    y, values = map_.drawn()
    # The scale is a legend for what the box shows, so the levels are
    # read off the rows inside *ylim* -- while the fill still gets the
    # unrestricted array, so a contour reaches the edge of the box
    # (:meth:`Map.drawn`).  Under ``--clim series`` *data_range* has
    # already been restricted the same way (:func:`scan_panels`); this
    # is what makes ``--clim frame`` and ``--quantile`` agree with it.
    scaled = values if ylim is None else map_.drawn(ylim)[1]
    levels = contour_levels(
        scaled,
        n_levels,
        non_negative=map_.non_negative,
        data_range=data_range,
        quantile=quantile,
        nice=nice,
    )

    filled = None
    if levels.size > 1:  # a single level bounds no band
        shaded, norm = band_colors(
            levels,
            cmap_positive if map_.non_negative else cmap_signed,
            non_negative=map_.non_negative,
        )
        if fill == "pcolormesh":
            # The same bands, drawn per cell instead of interpolated
            # between samples -- the honest rendering of a grid that
            # is coarse wherever it is (:func:`cell_edges`).
            filled = ax.pcolormesh(
                cell_edges(map_.lam, log=True),
                cell_edges(y, log=map_.y_log),
                values,
                cmap=shaded,
                norm=norm,
            )
        else:
            # ``extend`` only where something can land above the
            # top level: without ``--quantile`` the levels cover the
            # data, and an arrowed colour bar would be a lie.
            filled = ax.contourf(
                map_.lam,
                y,
                values,
                levels=levels,
                cmap=shaded,
                norm=norm,
                extend="neither" if quantile is None else "max",
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
    if map_.y_log:
        # One decade, one length, on both axes: the constraint a
        # log-log layout is built around.  A no-op once
        # panel_geometry has sized the box, and the guarantee if
        # anything else moves the limits.  Meaningless across a
        # log/linear pair, where it would match decades to data units.
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

    *decade* is inches per decade of the abscissa, the one free
    parameter the decade rule leaves; ``None`` derives it from *width*
    so the figure comes out exactly that wide.  *box_aspect* is the
    axes box's height over its width, and applies only to a **linear**
    ordinate -- a logarithmic one takes the same decade length as the
    abscissa instead (module docstring, "Figure geometry").
    """

    width: float = PAGE_LINEWIDTH
    decade: float | None = None
    box_aspect: float = 1.0
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
    """A figure's placement, in inches and figure fractions.

    *m_top* is the one margin that is not a module constant: it
    carries however many title lines the panels have
    (:func:`panel_geometry`).
    """

    fig_w: float
    fig_h: float
    box_w: float
    box_h: float
    nrows: int
    ncols: int
    m_top: float = _M_TOP

    def axes_rect(self, panel: int) -> tuple[float, float, float, float]:
        """``[left, bottom, width, height]`` of one panel's axes box."""
        row, col = divmod(panel, self.ncols)
        left = _M_LEFT + col * (self.box_w + _COL_PITCH_EXTRA)
        top = (
            _SUP_HEIGHT
            + row * (self.m_top + self.box_h + _M_BOTTOM + _ROW_GAP)
            + self.m_top
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
    *,
    y_log: bool,
    title_lines: int = 1,
) -> Geometry:
    r"""Size a figure from the abscissa's decade count.

    The axes box is ``decade`` inches per decade of the wavelength
    axis; the scale comes from ``style.decade`` when given and
    otherwise from ``style.width``, which then comes out exact.  Its
    height is ``style.box_aspect`` times that width on a linear
    ordinate and the same decade length again on a logarithmic one,
    where the shape therefore follows from the limits alone.
    Everything around the box is a fixed inch budget (the ``_M_*``
    module constants), so the figure height is whatever the panels
    need.

    *title_lines* is the tallest panel title of the figure, in lines:
    the top margin grows by :data:`_TITLE_LINE` for each one past the
    first, which is what makes room for the `$E^{\mathrm{ref}}$` a
    normalised panel reports (:func:`field_title`).
    """
    nrows = math.ceil(n_panels / style.ncols)
    m_top = _M_TOP + (title_lines - 1) * _TITLE_LINE
    decades_x = math.log10(xlim[1] / xlim[0])
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
    box_w = decade * decades_x
    box_h = (
        decade * math.log10(ylim[1] / ylim[0])
        if y_log
        else box_w * style.box_aspect
    )
    fig_w = (
        _M_LEFT
        + style.ncols * (box_w + _COL_AFTER)
        + (style.ncols - 1) * (_COL_GAP + _M_LEFT)
        + _M_RIGHT
    )
    fig_h = (
        _SUP_HEIGHT
        + nrows * (m_top + box_h + _M_BOTTOM)
        + (nrows - 1) * _ROW_GAP
    )
    return Geometry(fig_w, fig_h, box_w, box_h, nrows, style.ncols, m_top)


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
    xlim = style.xlim or (maps[0].lam.min(), maps[0].lam.max())
    ylim = y_limits(series, options, style.ylim)
    geometry = panel_geometry(
        len(panels),
        xlim,
        ylim,
        style,
        y_log=options.y_log,
        title_lines=1 + max(m.title.count("\n") for m in maps),
    )

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
    r"""``tag -> (stream, prefix-or-empty, marginal)`` for what is here.

    One tag per stored prefix and **drawable** stored marginal --
    :data:`MARGINALS` intersected with what the sidecar says the
    stream carries, so a default run offers no ``_x0`` tag because it
    has no such field, and ``xz00`` never becomes a tag at all.  Which
    tags come out normalised by `$E^{\mathrm{ref}}$` is
    :func:`normalises`, not a tag of its own, and which of them are
    rendered *by default* is :func:`default_series`.
    """
    out: dict[str, tuple[str, str, str]] = {}
    if spectra is not None:
        for prefix in spectra.prefixes:
            for marginal in spectra.suffixes:
                if marginal in MARGINALS:
                    out[f"spectra_{prefix}_{marginal}"] = (
                        "twin_yspectra",
                        prefix,
                        marginal,
                    )
    if budget is not None:
        for marginal in budget.suffixes:
            if marginal in MARGINALS:
                out[f"budget_{marginal}"] = ("twin_ybudget", "", marginal)
    return out


def default_series(
    registry: dict[str, tuple[str, str, str]],
    *,
    x0: bool = False,
    budget: bool = False,
) -> list[str]:
    """The tags rendered when ``--series`` names none.

    Two things are held back, each behind its own flag rather than a
    tag the caller has to know the name of.  The `$k_x = 0$` plane is
    a slice of the mode plane, not a marginal of it, and only a legacy
    or ``twin.x0_planes`` stream has one; the whole ``twin_ybudget``
    set is a figure per term, which is worth asking for rather than
    getting.  ``--series`` overrides both: naming a tag renders it.
    """
    return [
        tag
        for tag, (stem, _, marginal) in registry.items()
        if (budget or stem in DEFAULT_STEMS)
        and (x0 or marginal in DEFAULT_MARGINALS)
    ]


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

    The reference normalisation a spectra series may carry is a
    property of the member set rather than of a tag, so it is
    :meth:`YSeries.reference_report` and its caller's to print, once.
    """
    panels = (
        spectra_panels(prefix, marginal)
        if prefix
        else budget_panels(series, marginal)
    )
    scales, notes = scan_panels(
        series,
        panels,
        options,
        declared=declared_signs,
        ylim=y_limits(series, options, style.ylim),
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
        help="exact series tags to render, overriding --budget / --x0 "
        "(default: the spectra marginals present)",
    )
    p.add_argument(
        "--budget",
        action="store_true",
        help="also render the twin_ybudget series (one figure per "
        "term); off by default",
    )
    p.add_argument(
        "--x0",
        action="store_true",
        help="also render the k_x = 0 plane series, where the stream "
        "carries one (twin.x0_planes, or a pre-xz00 member)",
    )
    p.add_argument(
        "--premultiply",
        choices=("ky", "k", "none"),
        default="k",
        help="premultiplier: both axes (log ordinate), wavenumber "
        "only, or neither",
    )
    p.add_argument(
        "--yscale",
        choices=("log", "linear"),
        default="log",
        help="wall-normal axis scale; linear keeps the wall row and "
        "takes --box-aspect for the panel shape",
    )
    p.add_argument(
        "--ref-stride",
        type=int,
        default=1,
        help="keep every Nth record of the E_ref average (default: "
        "every one, whatever --stride is)",
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
    p.add_argument(
        "--levels",
        type=int,
        default=10,
        help="bands from zero to the peak; sets the step, not the count",
    )
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
        help="wall-normal axis limits, in the plotted units "
        "(default: the grid, floored at y+ = 1 when logarithmic)",
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
    p.add_argument(
        "--box-aspect",
        type=float,
        default=1.0,
        help="axes box height over width, linear ordinate only",
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
        y_log=args.yscale == "log",
    )
    style = PlotStyle(
        width=args.width,
        decade=args.decade,
        box_aspect=args.box_aspect,
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
                ref_stride=args.ref_stride,
            )
        except FileNotFoundError as exc:
            print(f"skipping {stem}: {exc}", file=sys.stderr)
            opened[stem] = None

    registry = available_series(
        opened["twin_yspectra"], opened["twin_ybudget"]
    )
    tags = args.series or default_series(
        registry, x0=args.x0, budget=args.budget
    )
    unknown = [t for t in tags if t not in registry]
    if unknown:
        raise SystemExit(
            f"unknown series {unknown}; available: {list(registry)}"
        )
    if not tags:
        raise SystemExit(
            "no stream found under the given members"
            if not registry
            else "every series present is held back by default; add "
            f"--budget / --x0, or name one of: {list(registry)}"
        )
    held = [t for t in registry if t not in tags]
    if held and not args.quiet:
        print(f"not rendering {' '.join(held)}", flush=True)

    if not args.quiet:
        print(
            f"{len(members)} member(s), Re = {args.re:g}, "
            f"Re_tau = {args.re_tau:g}, stride {args.stride}, "
            f"premultiply {args.premultiply}, yscale {args.yscale}, "
            f"half {args.half}, clim {args.clim}, "
            f"usetex {plt.rcParams['text.usetex']}",
            flush=True,
        )
        # E_ref is one number per component for the whole member set,
        # so its report belongs here, not once per tag that uses it.
        spectra = opened["twin_yspectra"]
        chosen = [
            f"{prefix}_{marginal}"
            for stem, prefix, marginal in (registry[tag] for tag in tags)
            if stem == "twin_yspectra"
        ]
        if spectra is not None and any(
            normalises(spectra, name) for name in chosen
        ):
            print("\n".join(spectra.reference_report()), flush=True)
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
