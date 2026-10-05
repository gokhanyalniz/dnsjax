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
A tag for each series (:class:`SeriesSpec`), in one of several figure
families.  A `$(\lambda, y)$` **map** is one figure per recorded
sample; every other family is one figure (or a few) for the whole run:
a **spacetime** map, a **history**, a moment budget, a decorrelation
front, a growth-law figure.  What a bare invocation draws is three sets
of maps, each on both wavenumber marginals (``_z`` gives `$\lambda_x$`;
the paper shows only `$\lambda_z$`), and the histories of the tracked
quantities:

- the **difference** spectra, a panel per velocity component and one
  for their sum;
- their **shape** maps, the same panels under the `$k\,y^+$`
  premultiplier and each frame over its own peak ("Shape maps");
- the ``twin_ybudget`` series: the terms of the difference-energy
  balance, regrouped from the stored densities
  (:func:`~dnsjax.analysis.twin.yspectra.balance_term`) and each drawn
  as its contribution to `$\partial_t E_\Delta$`, on a 3 x 3 grid
  (:data:`MAP_PANELS`).  The driving input, one mode, is drawn only by
  a spacetime map, with the pressure transport
  (:data:`SPACETIME_PANELS`).  ``--no-budget`` drops the set;
- the **histories** of the difference spectra, of their shape maps and
  of the production: the `$k$`-sum against wall distance and time, and
  the wall-normal average against wavelength and time, each
  premultiplied after its sum ("History maps").  ``--no-history``
  drops them.

The panels whose spectra keep one continuous bulk carry the track of
their peak and the ellipse of their size and tilt ("Peak tracking",
"Size and tilt"); ``--no-frames`` keeps those and the whole-run figures
and skips the frames.  Everything else is held back behind its own
flag rather than behind a tag name the caller has to know, and the
switches have the same shape:

- ``--reference`` adds the reference field's spectra, as maps -- and,
  under ``--spacetime``, their `$k$`-summed map too.  They are the
  turbulent flow's own, statistically the same in every frame, which
  is why they are not drawn unasked.  They come from
  ``twin_yspectra_ref.bin``, or from ``twin_yspectra.bin`` itself for a
  member recorded before the reference had a stream of its own; one
  set may mix the two layouts (:class:`_Reference`).  On another
  cadence than the difference stream's (``twin.it_yspectra_ref``) the
  reference *maps* need frames the two share, while the normalisation
  below averages every reference sample either way;
- ``--decorr`` adds `$\mathcal{R}$`, the decorrelation over a
  `$k$`-resolved reference, as maps ("Decorrelation");
- ``--decorr-k`` adds `$\mathcal{R}^k$`, the decorrelation over a
  `$k$`-summed one, as maps -- and, under ``--spacetime``, its
  `$k$`-summed map too, that being the only spacetime series a
  decorrelation has;
- ``--spacetime`` adds the `$k$`-summed `$(y, t)$` maps of whatever
  else is selected: the difference spectra always, the reference
  spectra under ``--reference``, `$\mathcal{R}^k$` under
  ``--decorr-k``, the budget unless ``--no-budget`` ("Spacetime
  maps");
- ``--x0`` adds the `$k_x = 0$` plane, where the stream has one --
  ``twin.x0_planes``, or any member recorded before that plane became
  opt-in.  It is a slice of the mode plane rather than a marginal of
  it, which is also why it is left absolute (below) and why neither
  decorrelation is offered for it;
- ``--moment-budget`` adds the budget of the difference energy's
  log-coordinate moments, in wall distance and on each marginal
  ("Moment budget"); it reads both streams;
- ``--front`` adds the decorrelation front, the time each
  `$(\lambda, y)$` cell rises through half its decorrelated level
  ("Decorrelation front");
- ``--growth`` adds the growth-law figures: the summary of the total
  difference energy, the bands of each marginal and of the wall
  distance, and the streaks, rolls and waves ("Growth laws").

``--series`` overrides all of them: it names exact tags, and
:func:`available_series` lists what a given member set offers.  The
`$(0, 0)$` mode ``xz00`` is never a tag -- it is one mode, with no
abscissa to put it on -- and reaches the figures only through the
reference quantities it is subtracted from.

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
it is the minority one; it is a knob, not the default, except on the
shape maps, which take it whatever the knob says ("Shape maps").
``none`` drops both factors.

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

Members out of phase
====================
Members meet on the relative clock because ``dnsjax-twin`` counts
every cadence from the member's own perturbation step, so its samples
sit at `$t_\mathrm{parent} + n\,c\,\Delta t$` whatever iteration
number its parent snapshot carried.  A member recorded before that
was true (the driver gated on the absolute step counter instead)
carries a grid displaced by `$(-\mathrm{it}_\mathrm{parent}
\bmod c)\,\Delta t$`, a phase set by the bookkeeping index of the
snapshot it was harvested from.  Members whose parents differed in
that residue then have no relative sample time in common but
`$t = 0$`, which the driver records unconditionally -- a single
frame per stream, and nothing to say why.

That set is **refused** (:func:`_check_phase`), with the measured
offset and the ``--align-atol`` that accepts it.  Widening the
tolerance pairs each frame with every member's *nearest* sample
rather than its own, which is exact for nothing and useful for a
recorded ensemble that cannot be run again: the frames come back, and
each averages fields recorded up to that far apart on their own
clocks.  Half the cadence is the cap -- a nearest neighbour is never
further than that from a uniform grid -- and the run report prints
the widest spread any frame actually pairs across
(:meth:`YSeries.alignment_spread`), so the size of the approximation
is on the page rather than in the flag.

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
away.  What a *resolved* denominator buys instead -- a ratio rather
than a rescaling -- is the next section.

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

Two members meet on that key when their parents are separated by a
whole number of sample cadences, which is what a harvest spacing
that is a multiple of ``it_* * dt`` gives (``dnsjax-twin`` counts
each member's cadence from its own perturbation step, so it is the
parents' separation that decides this, not their ``it`` residues).
Parents spaced otherwise put the members on interleaved absolute
grids: there is then nothing to deduplicate, the report's instant
count comes out at the sample count, and the average becomes
coverage-weighted -- the middle of the covered window, where every
member's stream overlaps, carries more weight than its ends -- rather
than uniform over the union.  It is a normalisation constant either
way (both estimate the same time average, and a constant moves no
contour), and the two counts side by side say which one was taken.

That key is exactly right for **one** reference trajectory and wrong
without it -- members subsampled from two *different* turbulent runs
would be merged wherever their absolute times met.  Nothing recorded
today separates them: ``twin.json`` carries ``parent`` and
``parent_t`` (the member's own start) and the harvest manifest the
source ``run_dir``, but neither is a signature of the trajectory, and
neither travels into the stream.  Until one does, the report prints
how many distinct parent snapshots are in play beside the instant
count, so a member set that is not one trajectory is at least visible.

What that pass actually accumulates is the **mean reference record**
-- both complete ``r_*`` marginals, with the `$(0, 0)$` mode
subtracted once, on arrival (:meth:`YSeries.reference_spectrum`).
Never a `$k_x = 0$` plane the stream may also carry: nothing divides
by one, and reading it would be a third more of the pass.
`$E^{\mathrm{ref}}$` is then
one of its three reductions, and the next section's two divisors are
the other two, so "a reference divisor never carries the `$(0, 0)$`
mode" is a statement about one array rather than three that could
drift.  It is exact: every step from a stored entry to
`$E^{\mathrm{ref}}$` is linear, so the mean of the reductions is the
reduction of the mean.

Decorrelation
=============
Dividing by a `$y$`- and `$k$`-independent scalar is a rescaling: it
moves no contour.  Dividing by a **resolved** reference is a
decorrelation, and there are two of them, differing only in what the
divisor does with `$k$`:

.. math::
    D_\alpha(y) = \bigl\langle \textstyle\sum_m r_\alpha(y, m)
        - r^{00}_\alpha(y) \bigr\rangle_t , \\
    \mathcal{R}^k_\alpha(y, m) = \frac{e_\alpha(y, m)}
        {2\,D_\alpha(y)} , \qquad
    \mathcal{R}_\alpha(y, m) = \frac{e_\alpha(y, m)}
        {2\,\langle r_\alpha(y, m)\rangle_t} .

Both saturate at 1, by the reading of "Reference normalisation"
applied where it is resolved: two statistically identical fields that
have decorrelated completely differ by twice the energy of either.
`$\mathcal{R}$` is the sharper of the two, each mode against its own
reference energy; `$\mathcal{R}^k$` weighs every mode against the same
number and so still says which modes carry the difference.  The
summed panel of either is one ratio of sums -- `$\sum_\alpha$` on each
half separately -- as the `$E^{\mathrm{ref}}$` panels are.

**Only the reference loses its `$(0, 0)$` mode.**  That mode is the
wall-parallel mean, common to both states of a pair and never
decorrelating, and it dominates the reference total -- a
plane-Poiseuille `$R_u$` is some 89 % its own.  The perturbation's is
its own business and stays, which matters nowhere on these maps (the
`$m = 0$` column is dropped from every one of them) and matters very
much to their `$k$`-sums, next section.  The subtraction still reaches
the `$m = 0$` **column** of `$\mathcal{R}$`'s divisor, which is where
that mode is stored: a column that is not drawn is not a column that
may be wrong.

**`$\mathcal{R}^k$` is premultiplied and `$\mathcal{R}$` is not.**  A
`$k$`-independent divisor leaves the map additive in `$k$`, so the
premultiplier still buys equal-areas-equal-energy on a logarithmic
abscissa and `$\sum_m \mathcal{R}^k$` is a quantity in its own right
(the spacetime map).  A `$k$`-resolved one destroys that additivity,
and its `$k$` would cancel against the numerator's in any case.
Neither takes ``volume_fac`` or the unit conversion, which cancel
between a ratio's two halves, so ``--no-volume-fac`` and
``--outer-units`` do not move either.

Where the reference vanishes -- the wall row, exactly, where every
velocity component does -- the ratio is ``nan`` rather than an
infinity (:func:`_ratio`): ``nan`` is dropped from every colour scale
here and left unpainted by both fills, and the default `$y^+ = 1$`
floor puts that row outside the box regardless.

One subtlety the fold introduces: the mean of two ratios is not the
ratio of two means once the divisor depends on `$y$`, so a
decorrelation would otherwise depend on whether ``--half mean`` ran
before or after the division.  The divisor is therefore symmetrised
about the centreline first (:func:`_symmetrise_y`), which makes the
two orders identical and the folded map the ratio of the folded
halves.

(The instantaneous sibling of these, each mode over the reference
energy *of that frame* rather than of the run, is
:func:`dnsjax.analysis.twin.decorrelation_ratio`.)

Shape maps
==========
The difference spectra once more, as ``spectra_s_x`` /
``spectra_s_z``, drawn for *where* their energy sits rather than for
how much of it there is.  Each panel of each frame is the absolute
quantity of the ``e_*`` panel under the `$k\,y$` premultiplier,
whatever ``--premultiply`` says, divided by its own peak:

.. math::
    S_\alpha(y, \lambda) = \frac{k\,y\,\Phi_\alpha(y, k)}
        {\max_{y,\,\lambda} k\,y\,\Phi_\alpha(y, k)} .

The premultiplier is the area-true pair of "Premultiplication": equal
areas on the two logarithmic axes are equal energy, so the map shows
how the energy is distributed over wall distance as well as over
wavelength, where the `$k$`-premultiplied map shows its local density
at each wall distance.  The peak is the frame's own, read over the
rows the box shows -- the ordinate's floor and ``--ylim`` included, as
for every colour scale ("Colour scales") -- and taken last, after the
fold and ``--smooth``, so the map reaches 1 there.  The summed panel
is over the peak of the summed spectrum: one ratio, not a sum of
three.

The colour scale is therefore `$[0, 1]$` by construction, the same
bands in every frame whatever ``--clim`` says, and ``--quantile`` does
not reach it.  A growth phase decades below saturation draws as
legibly as saturation does, which the frozen ``e_*`` scale cannot give
and ``--clim frame`` gives only with a colour bar that moves.  The
band below the first level stays unfilled, as on every non-negative
map here.

Whatever is constant over a frame cancels from the map: the unit
conversion, ``volume_fac`` and `$E^{\mathrm{ref}}$`, which a shape map
is therefore not divided by -- its peak is its scale, and a time
average adds nothing to that, so a shape-only run never reads the
reference.  So do the units of `$y$`: a shape map is the same in wall
and outer units, the `$Re_\tau$` of "Premultiplication" included.  All
of them survive in the peak, which each panel's title reports on its
second line, in the plotted units, as an `$E^{\mathrm{ref}}$` is
reported: the absolute ``ky`` map is the shape map times it.  The
panels carry the track of their peak, as the ``e_*`` panels do ("Peak
tracking").

Spacetime maps
==============
Sum a `$(y, k)$` stream over `$k$` and what is left is a `$(y, t)$`
field, which is one figure for a whole run rather than one per frame:
wall distance across, on the same scale and floor as the maps'
ordinate, and time up.  ``--spacetime`` asks for them, and every
selected `$k$`-summable series then has one -- the difference spectra,
the reference spectra under ``--reference``, `$\mathcal{R}^k$` under
``--decorr-k``, the `$k_x = 0$` slice under ``--x0``, the budget
terms unless ``--no-budget`` -- and each
is **marginal-free**, `$\sum_m e_x = \sum_m e_z$` being two readings
of one complete sum over the mode plane.  So there is one figure per
quantity, its panels the three components and their sum (the budget's,
its terms and theirs), not one figure per marginal
(:func:`check_k_sum` asserts that rather than assuming it).

Which modes the sum covers is the whole convention, and it is the
previous section's: a **reference** spectrum loses its `$(0, 0)$`
mode, everything else keeps every mode it has.  So the difference
map is the total difference energy at that wall distance,
`$\mathcal{R}^k$`'s `$k$`-sum is that over `$2D_\alpha(y)$`, and a
reference map is `$D_\alpha(y)$` without its time average -- whose
wall-normal average is `$E^{\mathrm{ref}}$` exactly.

**Nothing here is premultiplied**, whatever ``--premultiply`` says:
`$\sum_m m\,\text{entry}$` is not a sum of energies, and a spacetime
map has no logarithmic wavelength axis for a premultiplier to serve.
``volume_fac``, the unit conversion and `$E^{\mathrm{ref}}$` reach an
absolute panel exactly as they reach a `$(\lambda, y)$` one.

Each series is drawn **twice**, once with the banded linear colour
scale the rest of the module uses and once logarithmically, on one
range: the two are two readings of the same numbers.  An energy is
bounded below by zero, which a logarithmic scale cannot reach, so its
floor is declared -- ``--log-decades`` below the peak, or the smallest
positive value drawn where the data spans fewer (:func:`log_floor`) --
and the colour bar extends downward to say so.  A series that changes
sign (a budget term does) gets no logarithmic figure at all, with a
line saying why.

Beside each pair goes a ``.npz`` (:func:`write_spacetime_npz`): the
drawn arrays, their axes in both unit systems, the divisor or
`$E^{\mathrm{ref}}$` each panel took, every factor that was and was
not applied, and the stream metadata the figures were labelled from.
Enough to redraw a panel, or to undo its normalisation, without them.

History maps
============
The tracked quantities against time, each reduced over one coordinate
and premultiplied by the other only **after** the reduction: the
`$k$`-sum against wall distance, times `$y$` in the plotted units
(``<tag>_y``), and the wall-normal average against the wavelength of
each marginal, times `$k$` (``<tag>_x`` for `$\lambda_z$`, ``<tag>_z``
for `$\lambda_x$`).  Each is then area-true on its logarithmic axis:
`$y \int \Phi\,\mathrm{d}k = \int k\,y\,\Phi\,\mathrm{d}\ln k$` and
`$k \int \Phi\,\mathrm{d}y = \int k\,y\,\Phi\,\mathrm{d}\ln y$`, so the
pair are the two marginals of the shape map's density, and their
moments the shape map's centroid and spreads, one coordinate at a
time.

Three series: ``history_e``, the difference spectra over their
`$E^{\mathrm{ref}}$` (u, v, w, sum); ``history_s``, the same with each
time row over its own peak, on `$[0, 1]$` -- the shape-map rule, row by
row, so every unit factor and the premultiplier's `$Re_\tau$` cancel;
and ``history_budget``, the production row `$\mathcal{P}_\Delta$`,
`$\mathcal{P}_\Delta^{\mathbf{U}}$`, `$\mathcal{P}_\Delta^{\tilde
{\mathbf{u}}}$` (:data:`HISTORY_BUDGET_PANELS`).  The wall distance's
`$y$` is the plotted one, so a wall-unit ``_y`` map carries the
`$Re_\tau$` of "Premultiplication"; the wavelength average is the
volume average per mode (:func:`half_weights`: the half-channel
quadrature, doubled, which a symmetric field contracts to the whole
channel), and ``volume_fac`` does not reach it.  The `$m = 0$` column
is dropped from the wavelength maps, as from every map.

The non-negative panels carry the 10, 50 and 90 % quantiles of each
time row's as-drawn density (:func:`quantile_curves`) -- where the
bulk sits and how wide it is along that coordinate.  An absolute
history is drawn under both colour scales, as a spacetime map is; a
shape or budget history under the linear one.  ``--first`` / ``--last``
choose the time window, which matters: a run saturated for most of its
length draws its migration in the bottom fifth of the box.

Moment budget
=============
How each term of the balance moves the moments of the difference
energy (:func:`moment_budget`).  The density is the energy itself --
each folded off-wall cell weighted by :func:`half_weights`, so a cell's
weight times its entry is its exact share of the volume average -- in
the component sum, the only one the component-summed budget can
explain.  If `$\partial_t e = \sum_B B$` then for any function
`$\varphi$` of the cell

.. math::
    \frac{\mathrm{d}\langle\varphi\rangle}{\mathrm{d}t}
      = E^{-1} \sum w\,(\varphi - \langle\varphi\rangle)\,
        \partial_t e ,

so every moment's rate is a sum over the terms
(:func:`~dnsjax.analysis.twin.moments.moment_rates`): ``moments_y`` in
physical space -- the `$k$`-sum over every mode, the history
``_y`` map's distribution, with `$\mathrm{d}\ln E/\mathrm{d}t$`,
`$\mathrm{d}\langle\ln y\rangle/\mathrm{d}t$` and the rate of
`$\sigma^2_{\ln y}$` -- and ``moments_x`` / ``moments_z`` over the
joint `$(\ln\lambda, \ln y)$` distribution of `$m \ge 1$`, whose
wavelength moments are the history ``_x`` / ``_z`` maps', adding the
wavelength centroid and variance and the covariance.  The groups are
:data:`MOMENT_TERMS`, the production in its two parts; the physical
space budget's pressure group carries the driving input too.

Each figure draws every group, their sum, and the moments' own rate by
second-order differences.  The identity is exact, so where those two
part, it is the stream's closure or the sampling cadence: on a run's
first few samples the moments change faster than a difference at the
stream's cadence can follow.  A net drift is a small residual of
large terms of opposite sign, so read the closure against the largest
term, not against the drift.  The two streams must share their frames.

Decorrelation front
===================
The time each `$(\lambda, y)$` cell decorrelates (``front_x``,
``front_z``): the last upward crossing of ``--front-level`` (one half
by default) by the mode-by-mode ratio `$\mathcal{R} =
e/(2\langle r\rangle_t)$`, interpolated between frames
(:func:`front_times`).  The last crossing rather than the first, so a
seed that starts above the level and dips below it is not timed at
the start; a cell that never stays above it by the end of the record
is left white.  Four panels per marginal, the components and their
sum, on one colour range of every time the boxes show, the iso-time
lines drawn: the front's position at those times.

Growth laws
===========
Which law the difference energy grows by, and when
(:mod:`dnsjax.analysis.twin.growth`).  With `$R = E/E_\mathrm{sat}$`,
`$E_\mathrm{sat}$` twice the reference's fluctuation energy, each law
is a straight line on one set of axes: an exponential in `$\ln E$`
against `$t$` and flat in the log-log `$\gamma$`-`$R$` diagram
(`$\gamma = \mathrm{d}\ln E/\mathrm{d}t$`); an algebraic law
`$E \propto (t - t_0)^\alpha$` a line of slope `$-1/\alpha$` in that
diagram; constant-rate decorrelation, the twin correlation `$C = 1 - R$`
decaying as `$e^{-\nu t}$`, a line of slope `$\nu$` in `$-\ln(1 - R)$`
against `$t$`.  Saturation alone -- the logistic at the exponential's
rate -- bends the `$\gamma$`-`$R$` diagram only as `$R \to 1$`, by
`$-R/(1 - R)$`, and gives `$-\ln(1 - R)$` a late slope equal to the
early rate; a slowing at `$R \ll 1$`, or a late rate well below the
early one, is therefore no inflection.

``growth_global`` is the summary: the total difference energy on each
member's ``twin.dat`` cadence (aligned on whole steps since the
perturbation; the spectra stream's totals where any member lacks one,
the figure saying which), the member geometric mean, on the four sets
of axes (:func:`growth_summary_figure`), with an exponential phase and
a constant-rate decorrelation phase marked by a stated criterion
(:func:`growth_phases`: the longest window within
:data:`GROWTH_TOLERANCE` of its own mean).  ``growth_x`` /
``growth_z`` follow one band per octave of `$m$` and the `$m = 0$`
column (the `$(0, 0)$` mode off both halves), ``growth_y`` a row per
octave of wall distance, each against twice its own reference share,
with -- where the budget stream shares the frames -- the band budget
per unit energy: the same-`$k$` production `$\mathcal{P}^{\mathbf{U}}
_\Delta$`, which grows a band in proportion to itself, against the
cross-scale terms, which feed it from other bands whatever its own
energy (the production and the transport, for a row).
``growth_ssp`` splits the `$k_x$` marginal into the self-sustaining
process's parts -- the streaks `$\Delta u$` and rolls `$\Delta v$`,
`$\Delta w$` at `$k_x = 0$`, the waves at `$k_x \neq 0$` (the
three-bin split of :func:`~dnsjax.analysis.twin.yspectra.bin_energies`,
its streak bin by component) -- with the `$k_x = 0$` plane's budget per
unit energy: the lift-up that builds it and the transfer out of it.

Folding the channel
===================
``--half mean`` (the default) averages the two channel halves at
matching wall distance, which is legitimate and free statistics
because the flow is statistically symmetric about its mid-plane:
under the reflection `$R_y:\,(u,v,w)(x,y,z) \mapsto
(u,-v,w)(x,-y,z)$` for plane Poiseuille, and under the rotation
`$(u,v,w)(x,y,z) \mapsto (-u,-v,w)(-x,-y,z)$` for plane Couette, whose
`$x \to -x$` both marginals are blind to (a stored entry pairs `$\pm k$`
already).  Every stored quantity is **even** under the flow's own
symmetry, so the fold is a plain arithmetic mean with no sign flips.
For the reflection: the spectra are moduli; every balance term is a
sum of stored densities that are each even --
`$\mathcal{P}_\Delta^{\mathbf{U}}$` flips both `$\Delta\hat v$` and
`$\partial_y U$`, the viscous and pressure densities pair each odd
factor with a `$\partial_y$` or with the `$v$` slot, and each
advective one carries an even number of odd factors for the same
reason.  The mid-plane pairs with itself and is **not** double
counted, and the grid is checked for the symmetry the fold assumes
rather than trusted -- which binds ``upper`` as well, since it labels
its rows with the *opposite* half's wall distances, but not
``lower``, where `$1 + y$` is the wall distance whatever the far half
does.
``--half lower`` / ``upper`` keep one wall instead, which is how a
run's own asymmetry is inspected.

Colour scales
=============
Non-negativity is **declared**, not inferred: the energies are sums
of squares, which the division by a positive `$E^{\mathrm{ref}}$` --
and either decorrelation's by a positive reference -- leaves them
(:data:`NON_NEGATIVE`), so those get the grey scale and everything
else the diverging one.  The budget's `$-\mathcal{D}_\Delta$` is minus
a sum of squares, declared non-positive (:data:`NON_POSITIVE`) and
drawn on the diverging scale, whose negative side it fills.  Either
declaration is asserted against the data once per series and an
excursion across zero is reported with its size relative to the peak --
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

Every colour scale, per frame, ramped or frozen, is read off the
**plotted** quantity -- premultiplied, folded, in inner units, over
exactly the rows the axes box shows: the ordinate's floor and limits
included, not merely the wall row a logarithmic axis cannot place
(:meth:`Map.drawn`, :func:`y_limits`).  The colour bar therefore
labels the same numbers the contours do, which matters most for
`$-\mathcal{D}_\Delta$`: its peak is at the wall, below the default
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
moves under you.

``--clim ramped`` sits between the two: each figure is frozen on the
extremes of every frame up to and including its own
(:meth:`PanelScale.data_range`), each side of a signed panel on its
own.  That scale never shrinks and always contains the frame's own
extremes, so nothing is clipped; while the field grows it is the
frame's own scale, and from the frame that holds the series extreme
onward it is the frozen one exactly.  In between -- through
saturation -- it still rises a little whenever a frame sets a new
extreme, which is what freezing on the frames seen so far costs
against freezing on all of them.  The sign family is decided once for
the whole series in every mode, so a panel never changes colour map
mid-run.

A shape map takes none of the three: each of its frames is over its
own peak, so its scale is `$[0, 1]$` in every frame by construction
("Shape maps").

Peak tracking
=============
The panels whose spectra keep one continuous bulk -- every panel of
the difference spectra and of their shape maps, and
`$\mathcal{P}_\Delta$` and `$\mathcal{P}_\Delta^{\mathbf{U}}$` among
the budget terms (:data:`TRACKED`) -- carry the track of their peak:
a point where it is in that frame, and a thin line through where it
has been since the first (:func:`draw_track`).  Red on a grey map and
black on a signed one, each haloed in white so it stays legible on the
darkest band.

The peak is not the largest value, which moves a whole cell at a time
and jumps between near-equal maxima, but the **centroid of the top
band** (:func:`peak_centroid`): the plotted quantity `$f$` treated as
a density over the plotted plane, restricted to the cells at or above
`$(1 - 1/n)$` of the frame's own peak for ``--levels`` `$n$` -- the
top band the map would get on its own exact scale -- and averaged
there,

.. math::
    \ln\lambda_c = \frac{\sum_{ij} w_{ij} \ln\lambda_j}
        {\sum_{ij} w_{ij}} , \quad
    \ln y_c = \frac{\sum_{ij} w_{ij} \ln y_i}{\sum_{ij} w_{ij}} ,
    \qquad w_{ij} = f_{ij}\,\Delta\ln\lambda_j\,\Delta\ln y_i ,

the `$\Delta$` being the grid's trapezoidal widths: both axes are
non-uniform, and an unweighted sum would drift toward wherever the
grid is fine.  A linear ordinate puts `$y$` in place of `$\ln y$`
throughout.  The band is the frame's own rather than the drawn colour
scale's, so the track is the same under every ``--clim``: it exists in
a growth-phase frame that a frozen scale leaves blank, and it never
collapses onto the one cell around the maximum, which the drawn top
band does whenever the peak has just crossed a level.  The rows are
the ones the box shows, as for the colour scale (:meth:`Map.drawn`).

Beside the frames of each tracked series goes a directory
``<tag>_track`` holding one figure of `$y_c$` and `$\lambda_c$`
against time, a line per tracked panel, and its ``.npz``
(:func:`render_tracks`) -- a directory of its own, so a glob over the
series' frames still matches frames alone.  The size and tilt of the
same panels go beside it ("Size and tilt").

Size and tilt
=============
Where the track says where a panel's peak is, its **moments** say how
large and how tilted the whole field is: the plotted quantity read as
a density on the plotted plane -- value times the trapezoidal widths
in `$\ln\lambda$` and `$\ln y$`, over the rows the box shows, the
track's own measure applied to every cell rather than to the top band
(:func:`map_moment_sums`) -- has a centroid, spreads
`$\sigma_{\ln\lambda}$` and `$\sigma_{\ln y}$`, an area measure
`$\sqrt{\det C}$` (the one-sigma ellipse's area over `$\pi$`), a
correlation `$\rho$` and a ridge slope `$b = C_{\lambda y}/C_{yy}$`,
the slope of the energy-weighted line through the row-by-row mean of
`$\ln\lambda$`, so that `$b = 1$` is `$\lambda \propto y$`
(:mod:`dnsjax.analysis.twin.moments`).

**Only constants cancel from them** -- a shape map's per-frame peak,
`$E^{\mathrm{ref}}$`, ``volume_fac``, the unit factors: every moment
is a ratio over the mass.  The premultiplier does not cancel; it is
what decides the weighting.  On a shape map the `$k\,y$`
premultiplier is the Jacobian of `$(k, y) \to (\ln k, \ln y)$`,
`$\Phi\,\mathrm{d}k\,\mathrm{d}y = k\,y\,\Phi\,\mathrm{d}\ln
k\,\mathrm{d}\ln y$`, so each cell's weight is its energy -- to the
quadrature, and to the half cells the trapezoid gives the first and
last wavelength and the rows at the box's edges; there the moments are
those of the energy distribution.  On a `$k$`-premultiplied map (a
local spectrum per unit `$y$`) each row is weighted by `$1/y$` against
its energy instead, so its centroid sits nearer the wall: the moments
of what that map shows, not of where the energy is.  A budget panel's
are its positive part's, the negative share beside them.

Every tracked frame draws its one-sigma ellipse, dashed, in the
track's colour (:func:`draw_ellipse`, built in the log coordinates so
it is an ellipse on the logarithmic axes), and the track directory
gets ``<tag>_moments`` -- the five against time, a line per tracked
panel -- and its ``.npz``.  The principal-axis angle is used for the
drawing only: it swings freely when the ellipse is nearly round, where
`$\rho$` and `$b$` do not.

Figure geometry
===============
The abscissa is sized by its decade count: the axes box is
`$\text{decade} \times D_\lambda$` wide for the `$D_\lambda$` decades
of the plotted limits, leaving only the scale free.  ``--width`` sets
it (default 6.61546 in, the write-up's ``\linewidth``) and
``--decade`` sets the decade length directly instead.

``--width`` fits that length to a figure of ``--ncols`` columns, the
spectra's two, and every figure of the run takes it.  A budget figure
is three columns whatever ``--ncols`` says (:data:`BUDGET_NCOLS`), its
nine panels in the rows :data:`MAP_PANELS` sets, so its panels are the
spectra's size and the figure is wider than ``--width`` -- half again
as wide, at the defaults.  The page's width cannot hold three
columns: each carries its ordinate labels, its secondary axis and its
colour bar, two inches apiece before any box.

The height follows the **ordinate's scale**.  Logarithmic (the
default): the same decade length applies to it as well -- **one
decade, one length, on both axes**, the constraint that makes a
`$\lambda \propto y$` band read at 45 degrees -- and
``ax.set_aspect(1)`` holds it there, so the box's shape follows from
the limits alone.  The default floor at `$y^+ = 1$`
(:data:`Y_FLOOR_PLUS`) is most of what sets it: on
`$y^+ \in [1, 179]$` against `$\lambda_z^+ \in [14, 1122]$` the box
is 1.2 times taller than wide, where the grid's full
`$y^+ \in [0.02, 179]$` would make it 2.1 and every figure
correspondingly tall.  ``--ylim`` trims it further.

Linear: the height is ``--box-aspect`` times the width, 1 (square) by
default, a linear axis having no decades to match.  ``set_aspect`` is
deliberately *not* applied there, where it would be matching decades
against data units.

A spacetime panel is the linear case with the axes swapped in
character: its abscissa is the wall distance, logarithmic by default
and so still sized by the decade rule, and its ordinate is time, which
takes ``--box-aspect``.  Under ``--yscale linear`` there are no
decades on either axis, and the box takes the fitted width instead
(``x_log`` in :func:`panel_geometry`).

Everything around the box is a fixed inch budget (the ``_M_*``
constants), with one thing free: the top margin grows by a line
(:data:`_TITLE_LINE`) for the two-line title a normalised panel
carries, so a figure never has to choose between the reported
`$E^{\mathrm{ref}}$` (or a shape map's peak) and its neighbour's tick
labels.

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
        MapOptions, Units, draw_map, draw_spacetime, make_map,
        make_spacetime, open_series)

    s = open_series(["twin1", "twin2"], "twin_ybudget", stride=10)
    opts = MapOptions(Units(re=4200.0, re_tau=178.62135279727977))
    m = make_map(s, "prod_x", frame=24, options=opts)
    draw_map(ax, m, units=opts.units)

    e = open_series(["twin1", "twin2"], "twin_yspectra", stride=10)
    r = make_spacetime(e, "decorr_k", options=opts, component=0)
    draw_spacetime(ax, r, units=opts.units, scale="log")

The whole-run families on a long run, without redrawing its frames::

    uv run --group plots python scripts/twin_spectral_maps.py \
        --members RUN1 RUN2 --out FIGDIR --re 4200 \
        --re-tau 178.62135279727977 --stride 1 --last 150 \
        --no-frames --growth --front --moment-budget

:func:`open_series` memory-maps each member and reads only the
records a figure draws -- the selected frames of the fields a map
shows, and for a spacetime map not even those whole: each record is
summed over `$k$` as it is read -- so a stream costs megabytes rather
than the gigabytes the eager reader
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
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
    LogNorm,
    Normalize,
    TwoSlopeNorm,
)
from matplotlib.ticker import FuncFormatter

from dnsjax.analysis.twin.growth import (
    bound_free,
    decorrelation_rate,
    log_rate,
    log_slope,
    logistic_rate,
    longest_window,
)
from dnsjax.analysis.twin.moments import (
    LogMoments,
    log_moment_sums,
    log_moments,
    moment_rates,
)
from dnsjax.analysis.twin.series import read_twin, uniform_grid
from dnsjax.analysis.twin.yspectra import (
    BALANCE_PARTS,
    BALANCE_TERMS,
    MIN_YBUDGET_VERSION,
    MIN_YSPECTRA_REF_VERSION,
    MIN_YSPECTRA_VERSION,
    balance_term,
    balance_terms,
    fluctuation_energy,
    fluctuation_profile,
    mean_free_spectrum,
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
#: set that mixes layouts is refused by name.  ``has_ref`` is written
#: the same way: whether a member has reference spectra at all,
#: whichever of the two layouts holds them (:class:`_Reference`), so a
#: set mixing them is one ensemble.
_STREAM_KEYS: dict[str, tuple[str, ...]] = {
    "twin_yspectra": ("has_ref", "suffixes"),
    "twin_ybudget": ("terms", "suffixes"),
}

#: Velocity components of the ``twin_yspectra`` leading axis.
COMPONENTS: tuple[str, ...] = ("u", "v", "w")

#: The two virtual spectra bases, built from the stored ``e_*`` and
#: ``r_*`` rather than read, so neither names a stored field.  They
#: differ only in what their divisor does with `$k$`, and everything
#: else follows from that (module docstring, "Decorrelation"):
#: :data:`DECORR_K` divides by a `$k$`-summed reference and stays
#: additive in `$k$`, so it keeps the premultiplier and has a
#: `$k$`-summed sibling among the spacetime maps; :data:`DECORR`
#: divides mode by mode and has neither.
DECORR_K: str = "decorr_k"
DECORR: str = "decorr"

#: The virtual base of the shape maps: the stored ``e_*`` again,
#: premultiplied by `$k\,y$` whatever ``--premultiply`` says and each
#: frame divided by its own peak, so that it spans `$[0, 1]$` in every
#: frame (module docstring, "Shape maps").
SHAPE: str = "s"

#: The two figure families a series tag can belong to
#: (:class:`SeriesSpec`): a `$(\lambda, y)$` map per recorded sample,
#: or one `$(y, t)$` map for the whole run.
MAP: str = "map"
SPACETIME: str = "spacetime"

#: The figure families beyond those two, each one figure (or a few)
#: for the whole run: the premultiplied `$(y, t)$` and `$(\lambda, t)$`
#: histories of the tracked quantities ("History maps"), the budget of
#: the spectra's log-coordinate moments ("Moment budget"), the
#: decorrelation front ("Decorrelation front") and the growth-law
#: diagnostics ("Growth laws").
HISTORY: str = "history"
MOMENT_BUDGET: str = "moments"
FRONT: str = "front"
GROWTH: str = "growth"

#: Stored suffixes whose panels are drawn relative to
#: `$E^{\mathrm{ref}}$` (module docstring, "Reference
#: normalisation").  ``x0`` is deliberately absent: it is a slice of
#: the mode plane, not a complete sum over it.
NORMALISED_MARGINALS: frozenset[str] = frozenset({"x", "z"})

#: Default bottom of a **logarithmic** ordinate, in wall units.  The
#: grid reaches far below it (`$y^+ \approx 0.02$` at the resolutions
#: these runs use), and nothing but `$\mathcal{D}_\Delta$` reaches its
#: first contour level down there, so the decade below `$y^+ = 1$`
#: buys a taller box and no information.  A linear ordinate keeps the
#: wall itself, which is a position it can show.
Y_FLOOR_PLUS: float = 1.0

#: Records read from a memory-mapped stream at a time while the
#: reference normalisation is accumulated.  Each is reduced onto the
#: running mean immediately, so this bounds that pass's memory.
_REF_CHUNK: int = 64

#: Decades below a spacetime map's peak that its **logarithmic**
#: colour scale reaches, unless the data itself spans fewer
#: (:func:`log_floor`).  A difference field climbing from ``twin.e0``
#: to saturation covers several, which is the reading that scale is
#: for; six of them keeps the growth phase legible without spending
#: the colour map on round-off.
LOG_DECADES: float = 6.0

#: Bands per decade of that scale.  Fine enough to read as a
#: continuous ramp, since only the decade boundaries are labelled and
#: drawn as contour lines -- a line per band would be mush.
_LOG_BANDS_PER_DECADE: int = 8

#: Fields that are non-negative **by construction**, keyed by the base
#: name: the two spectra prefixes are `$\tfrac12|\hat u|^2$` sums,
#: which a division by a positive `$E^{\mathrm{ref}}$` leaves them.
#: Both decorrelations are one of those sums over twice another, and a
#: shape map one of them over its own peak, so they inherit it.
NON_NEGATIVE: frozenset[str] = frozenset({"e", "r", DECORR, DECORR_K, SHAPE})

#: Budget terms that are non-positive **by construction**: ``diss`` is
#: `$-\mathcal{D}_\Delta$`, minus a sum of squares.  Drawn on the
#: signed scale, whose negative side it fills, and checked against the
#: data as :data:`NON_NEGATIVE` is.  ``tr_visc`` is deliberately
#: absent: it carries ``V``'s operator (discrete-Laplacian) form,
#: which is not sign-definite (:mod:`dnsjax.twin.diagnostics`,
#: "Dissipation form").
NON_POSITIVE: frozenset[str] = frozenset({"diss"})

#: Below this fraction of the peak, an excursion across zero in a
#: declared one-signed field is reported as truncation rather than a
#: defect.  These are sums of squares, so round-off is the only
#: mechanism and it lands many orders below this.
SIGN_TOLERANCE: float = 1e-9

#: Budget terms excluded from the ``sum`` virtual field: the two parts
#: of ``prod``, which would count production twice.  What is left adds
#: up to `$\partial_t E_\Delta$` at every `$(y, k)$`
#: (:func:`~dnsjax.analysis.twin.yspectra.balance_term`).
NON_ADDITIVE_TERMS: frozenset[str] = BALANCE_PARTS

#: Budget panels that draw a sum of balance terms rather than one
#: (:func:`balance_field`): ``press_input`` is `$\mathcal{I}_\Delta -
#: \mathcal{T}^{\Delta\mathbf{u}}_{\Delta p}$`, the stored ``Wp``.
COMBINED_PANELS: dict[str, tuple[str, ...]] = {
    "press_input": ("tr_press", "input"),
}

#: The panels of a `$(\lambda, y)$` budget map, in the order drawn:
#: with the ``sum`` :func:`budget_panels` appends, the three rows of
#: the budget's grid (:data:`BUDGET_NCOLS`) -- the production and its
#: two parts; the dissipation and the viscous and pressure transports;
#: the two advective transports and the sum.  ``input`` lives at
#: `$(0, 0)$` alone, in the `$m = 0$` column no map draws, so the
#: pressure panel is the pressure transport alone.
MAP_PANELS: tuple[str, ...] = (
    "prod",
    "prod_mean",
    "prod_fluct",
    "diss",
    "tr_visc",
    "tr_press",
    "tr_ref",
    "tr_self",
)

#: The panels of a spacetime budget map: those of a map, each in its
#: place, the pressure panel carrying the driving input as well, which
#: a `$k$`-sum does show.  Every panel but the two parts of ``prod``
#: then adds up to the ``sum``, `$\partial_t E_\Delta$`.
SPACETIME_PANELS: tuple[str, ...] = tuple(
    "press_input" if term == "tr_press" else term for term in MAP_PANELS
)

#: The panels of the budget's history maps: the production and its two
#: parts, the first row of the budget's grid.  The two tracked terms
#: (:data:`TRACKED`) and the fluctuation part that completes them --
#: the term that carries the growth phase, which a history of the
#: other two alone would leave unexplained.
HISTORY_BUDGET_PANELS: tuple[str, ...] = ("prod", "prod_mean", "prod_fluct")

#: The quantiles of each time row's as-drawn density that a history map
#: draws as lines, in the order they are drawn: the median and the two
#: that bound the central 80 % (module docstring, "History maps").
HISTORY_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)

#: The moment budget's term groups, in the order drawn: the balance
#: terms of :data:`~dnsjax.analysis.twin.yspectra.BALANCE_TERMS` with
#: the production split into its two parts and the whole left out (it
#: would count twice), so the groups add up to the rate.  The physical
#: space budget swaps ``tr_press`` for ``press_input``, its pressure
#: transport with the driving input it carries at `$(0, 0)$`.
MOMENT_TERMS: tuple[str, ...] = (
    "prod_mean",
    "prod_fluct",
    "diss",
    "tr_visc",
    "tr_press",
    "tr_ref",
    "tr_self",
)

#: The decorrelation level a front map times the crossing of (module
#: docstring, "Decorrelation front"): half the decorrelated value, the
#: level at which a cell holds half the energy it will hold.
FRONT_LEVEL: float = 0.5

#: The relative band a growth phase must stay within to be marked as
#: one (:func:`~dnsjax.analysis.twin.growth.longest_window`): a rate
#: within 10 % of its own mean over the window.
GROWTH_TOLERANCE: float = 0.1

#: Above this saturation fraction `$-\ln(1 - R)$` is noise: sampling
#: fluctuations take `$R$` to 1 and beyond at saturation.
GROWTH_R_MAX: float = 0.95

#: Columns of every budget figure, map or spacetime alike, whatever
#: ``--ncols`` sets for the spectra: its nine panels are the three rows
#: of :data:`MAP_PANELS` (module docstring, "Figure geometry").
BUDGET_NCOLS: int = 3

#: Panel labels for the budget terms in the write-up's notation, as
#: ``(sign, symbol)``: each panel is the term's contribution to
#: `$\partial_t E_\Delta$`, and a contribution's minus sign leads its
#: title, ahead of any premultiplier (:func:`field_title`).
TERM_LABELS: dict[str, tuple[str, str]] = {
    "prod": ("", r"\mathcal{P}_\Delta"),
    "prod_mean": ("", r"\mathcal{P}_\Delta^{\mathbf{U}}"),
    "prod_fluct": ("", r"\mathcal{P}_\Delta^{\tilde{\mathbf{u}}}"),
    "input": ("", r"\mathcal{I}_\Delta"),
    "diss": ("-", r"\mathcal{D}_\Delta"),
    "tr_self": ("-", r"\mathcal{T}_{E_\Delta}^{\Delta\mathbf{u}}"),
    "tr_ref": ("-", r"\mathcal{T}_{E_\Delta}^{\mathbf{u}}"),
    "tr_visc": ("-", r"\mathcal{T}_{E_\Delta}^{\nu}"),
    "tr_press": ("-", r"\mathcal{T}_{\Delta p}^{\Delta\mathbf{u}}"),
    "press_input": (
        "",
        r"(\mathcal{I}_\Delta - \mathcal{T}_{\Delta p}^{\Delta\mathbf{u}})",
    ),
    "sum": ("", r"\partial_t E_\Delta"),
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

#: The marginals rendered unless ``--x0`` asks for the rest: the two
#: true ones, a default about what is worth looking at rather than
#: what is on disk.
DEFAULT_MARGINALS: frozenset[str] = frozenset({"x", "z"})

#: The map panels that carry the track of their peak (module
#: docstring, "Peak tracking"): every panel of the two
#: difference-spectra marginals and of their shape maps, and the
#: production and its mean-shear part on the budget's -- the
#: quantities whose spectra keep one continuous bulk, which is what
#: makes one centroid a location.  Never the reference spectra, a
#: decorrelation or the `$k_x = 0$` slice.
TRACKED: frozenset[str] = frozenset(
    {
        "e_x",
        "e_z",
        f"{SHAPE}_x",
        f"{SHAPE}_z",
        "prod_x",
        "prod_z",
        "prod_mean_x",
        "prod_mean_z",
    }
)

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
#: inches at the default font size: a normalised panel reports what it
#: was divided by on a second line -- `$E^{\mathrm{ref}}$`, or a shape
#: map's peak -- and ``_M_TOP`` has no slack to absorb one.
_TITLE_LINE: float = 0.17

#: Upper bound on the number of labelled colour-bar ticks.
_BAR_TICKS: int = 6

#: A track's colour on a map, by the map's colour family (keyed on
#: :attr:`Map.non_negative`): red reads on every grey band, black on
#: every red and blue one (:func:`draw_track`).
_TRACK_COLOURS: dict[bool, str] = {True: "#e41a1c", False: "black"}

#: A track's history line and current point, in points.  Both are
#: haloed in white, ``_TRACK_HALO`` wider than the line and
#: ``_TRACK_EDGE`` round the point, which is what keeps them legible
#: on the darkest band, where the colour alone is not.
_TRACK_LINE: float = 0.9
_TRACK_MARKER: float = 4.5
_TRACK_EDGE: float = 0.8
_TRACK_HALO: float = 1.2

#: The track figure's lines, one per tracked panel in panel order
#: (:func:`track_figure`): the write-up's categorical order, checked
#: for colour-vision separation on white (worst pair `$\Delta E$` 11.0
#: deutan, 8.6 tritan, 15.6 normal), each paired with a dash pattern so
#: identity never rests on hue alone -- the orange sits at 2.3:1
#: against the page.  Four is as many as a tracked figure has.
_TRACK_SERIES: tuple[str, ...] = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
_TRACK_DASHES: tuple = (
    "-",
    (0, (5, 2)),
    (0, (1.5, 1.5)),
    (0, (7, 2, 1.5, 2)),
)

#: The track figure's height, in inches, at ``--width``: two rows,
#: their twin axes and a legend underneath.
_TRACK_HEIGHT: float = 4.6

#: Height of each row of a line-figure grid (moments, moment budget,
#: growth laws), in inches, legend and labels included.
_ROW_HEIGHT: float = 1.55

#: The moment budget's seven term groups (:data:`MOMENT_TERMS`), in
#: order: the data-viz reference palette's seven categorical slots,
#: validated for colour-vision separation on white (worst adjacent pair
#: `$\Delta E$` 9.1 protan, normal-vision floor 19.6).  Three sit below
#: 3:1 contrast against the page, so each also carries a dash pattern
#: and the legend names every line; their sum is black and the
#: finite-difference check a grey dashed line.
_TERM_SERIES: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
)
_TERM_DASHES: tuple = (
    "-",
    (0, (5, 2)),
    (0, (1.5, 1.5)),
    (0, (7, 2, 1.5, 2)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (8, 3)),
    (0, (2, 2)),
)

#: Colour map of a family of lines ordered by a scale (a wavelength or
#: a wall distance): one hue, light to dark with the scale, so order
#: reads as lightness and no line is lost on the page (the ramp's
#: palest quarter is never used).
_BAND_CMAP: str = "Blues"
_BAND_RANGE: tuple[float, float] = (0.3, 1.0)


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

    def time(self, t: float | np.ndarray) -> float | np.ndarray:
        r"""`$t \to t^+ = t\,Re_\tau^2/Re$`.

        Unconditional, unlike :meth:`length` and :meth:`energy`: the
        figure titles report `$t^+$` beside the outer `$t$` whatever
        units the axes are in.  An axis wants :meth:`plotted_time`.
        """
        return t * self.re_tau**2 / self.re

    def plotted_time(self, t: float | np.ndarray) -> float | np.ndarray:
        r"""`$t$` in whichever units the figure is drawn in."""
        return self.time(t) if self.wall else t

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
        """Wall-distance axis label (primary axis)."""
        return r"$y^+$" if self.wall else r"$y/h$"

    @property
    def t_label(self) -> str:
        """Time axis label (primary axis) of a spacetime map."""
        return r"$t^+$" if self.wall else r"$t\,U_\mathrm{cl}/h$"

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
class _Reference:
    """A member's reference spectra, memory-mapped and deduplicated.

    Their own stream, ``twin_yspectra_ref.bin``, on its own cadence
    (``twin.it_yspectra_ref``, by default the difference stream's) --
    or, for a member recorded before the reference had a stream of its
    own, the ``r_*`` fields of ``twin_yspectra.bin`` itself
    (``includes_ref``), which is then this same memory map.  Every
    ``r_*`` read goes through here, so the two layouts read alike.
    """

    meta: dict
    records: np.memmap
    rows: np.ndarray
    t_abs: np.ndarray
    t_rel: np.ndarray


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
    ref: _Reference | None = None  # its reference spectra, if any


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
    r"""A member's parsed ``twin.json``, or ``{}`` when it has none.

    Two fields are read off it.  ``parent_t`` is the member's
    `$t_\mathrm{parent}$`, the clock the ensemble aligns on -- what
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


def _map_stream(path: Path, stem: str, floor: int):
    """``(meta, records, rows, t_abs)`` of one stream, deduplicated."""
    bin_path, json_path = path / f"{stem}.bin", path / f"{stem}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"no sidecar {json_path}")
    with open(json_path) as fh:
        meta = json.load(fh)
    version = int(meta.get("format_version", 0))
    if version < floor:
        raise ValueError(
            f"{json_path}: format_version {version} predates the "
            f"reader floor {floor}."
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
    return meta, records, rows, t_abs


def _open_member(path: Path, stem: str) -> _Member:
    """Memory-map one member and resolve its usable records."""
    meta, records, rows, t_abs = _map_stream(path, stem, STEMS[stem])
    record = _twin_record(path)
    parent_t = record.get("parent_t")
    t0 = float(t_abs[0]) if parent_t is None else float(parent_t)
    ref = None
    if stem == "twin_yspectra":
        if bool(meta.get("includes_ref")):
            ref = _Reference(meta, records, rows, t_abs, t_abs - t0)
        elif (path / "twin_yspectra_ref.json").is_file():
            r_meta, r_records, r_rows, r_t = _map_stream(
                path, "twin_yspectra_ref", MIN_YSPECTRA_REF_VERSION
            )
            ref = _Reference(r_meta, r_records, r_rows, r_t, r_t - t0)
        meta["has_ref"] = ref is not None
    return _Member(
        path,
        meta,
        records,
        rows,
        t_abs,
        t_abs - t0,
        str(record.get("parent", "")),
        ref,
    )


def balance_field(read, meta: dict, name: str) -> np.ndarray:
    """A field by name, the balance terms regrouped on the way.

    ``<term>_<suffix>`` with *term* one of
    :data:`~dnsjax.analysis.twin.yspectra.BALANCE_TERMS` is built from
    the stored densities
    (:func:`~dnsjax.analysis.twin.yspectra.balance_term`), and with
    *term* one of :data:`COMBINED_PANELS` it is the sum of the balance
    terms that panel names; any other name is a stored field and comes
    back from *read* unchanged.  *read* returns a stored field by name,
    which is what lets the three readers here -- the cached ensemble
    mean (:meth:`YSeries.field`), the chunked `$k$`-sum
    (:meth:`YSeries.reduced`) and one frame of one member
    (:func:`_frame_mean`) -- share the one definition.
    """
    base, _, suffix = name.rpartition("_")
    if base in COMBINED_PANELS:
        terms = COMBINED_PANELS[base]
    elif base in BALANCE_TERMS:
        terms = (base,)
    else:
        return read(name)
    return sum(balance_term(read, meta, term, suffix) for term in terms)


@dataclass
class YSeries:
    r"""An ensemble-averaged, subsampled `$(y, k)$` stream.

    Fields are read and averaged on demand (and cached), so opening a
    series is cheap however long the run and however many members it
    has.  ``index`` carries each frame's position in the **first**
    member's own deduplicated record sequence -- the number the output
    filenames use.  That is the frame label rather than a position on
    the shared grid, and deliberately: it is strictly increasing in
    time (so a lexical sort of the filenames is the time order) and it
    survives ``--stride`` / ``--first`` / ``--last`` naming the same
    frame the same way, which a position within the *selection* would
    not.  It skips a number wherever the first member has a sample the
    others lack.
    """

    stem: str
    members: tuple[_Member, ...]
    rows: np.ndarray  # (n_members, n_frames) record index per member
    index: np.ndarray  # (n_frames,) row in members[0], the frame label
    t_rel: np.ndarray  # (n_frames,) time since the perturbation
    t_members: np.ndarray  # (n_members, n_frames) each member's own
    matched: np.ndarray  # (n_members,) hits on members[0]'s full grid
    meta: dict  # the first member's sidecar
    ref_stride: int = 1  # subsampling of the reference normalisation
    # (n_members, n_frames) reference record per frame, -1 where a
    # member's reference has no sample there; None without references.
    ref_rows: np.ndarray | None = None
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _reference: tuple[dict[str, np.ndarray], list[str]] | None = field(
        default=None, repr=False
    )

    @property
    def n_members(self) -> int:
        """How many member streams are being averaged."""
        return len(self.members)

    def grid_report(self) -> str | None:
        """Which members cost the shared grid samples, if any did.

        The frame grid is the **intersection** of the members'
        relative grids, so one member that samples elsewhere silently
        shortens it -- to a single frame in the limit, which is what a
        member set recorded off the shared clock collapses to and what
        says nothing on its own.  ``None`` when every member carries
        every one of the first's sample times; otherwise the line
        naming the shortfall, member by member, whatever caused it.
        The phase case is diagnosed before this (:func:`_check_phase`)
        and named as such; this is the catch-all behind it.
        """
        full = self.members[0].t_rel.size
        if int(np.min(self.matched)) == full:
            return None
        return (
            f"{self.stem}: {int(np.min(self.matched))} of "
            f"{full} sample times of {self.members[0].path} are in "
            "every member; per member "
            + ", ".join(
                f"{member.path.name} {int(hits)}"
                for member, hits in zip(
                    self.members, self.matched, strict=True
                )
            )
        )

    def alignment_spread(self) -> float:
        """The widest relative-time spread inside one frame.

        Zero for a set aligned exactly (the default): every member
        contributes the same relative instant.  Under
        ``--align-atol`` it is how far apart on their own clocks the
        two extreme members of the worst frame were recorded --
        what the widened tolerance actually bought and cost.
        """
        return float(
            np.max(self.t_members.max(axis=0) - self.t_members.min(axis=0))
        )

    @property
    def y(self) -> np.ndarray:
        r"""Wall-normal grid, `$y \in [-1, 1]$`."""
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
        """The balance terms of a ``twin_ybudget`` stream.

        :data:`~dnsjax.analysis.twin.yspectra.BALANCE_TERMS`, regrouped
        from the stored densities (:func:`balance_field`); a stream the
        balance cannot be built from is refused here, before any figure
        is.  Empty for the spectra stream.
        """
        if self.stem != "twin_ybudget":
            return ()
        return balance_terms(self.meta)

    @property
    def suffixes(self) -> tuple[str, ...]:
        """Stored marginals, normalised onto the sidecar on open."""
        return tuple(self.meta["suffixes"])

    @property
    def prefixes(self) -> tuple[str, ...]:
        """Spectra prefixes present (``twin_yspectra`` only)."""
        if self.stem != "twin_yspectra":
            return ()
        return ("e", "r") if bool(self.meta.get("has_ref")) else ("e",)

    def source(self, index: int, name: str) -> tuple[np.memmap, np.ndarray]:
        """``(records, rows)`` holding *name* for member *index*.

        A difference or budget field reads the member's own stream on
        the frames; an ``r_*`` field its reference (:class:`_Reference`),
        whose rows on the frames were matched by relative time when
        the series opened.  A frame its reference has no sample at --
        only a ``twin.it_yspectra_ref`` off the difference cadence
        does that -- refuses the read rather than borrowing a
        neighbour.
        """
        member = self.members[index]
        if not name.startswith("r_"):
            return member.records, self.rows[index]
        rows = self.ref_rows[index]
        missing = int((rows < 0).sum())
        if missing:
            raise ValueError(
                f"{member.path}: its reference spectra have no sample at "
                f"{missing} of the {rows.size} selected frames (recorded "
                "on another twin.it_yspectra_ref than twin.it_yspectra); "
                "a reference map needs one at every frame -- select the "
                "frames the two cadences share with --stride, or leave "
                "the reference maps out."
            )
        return member.ref.records, rows

    def field(self, name: str) -> np.ndarray:
        r"""Ensemble mean of one field over the selected frames.

        Shape ``(n_frames, 3, n_y, n_k)`` for the spectra and
        ``(n_frames, n_y, n_k)`` for the budget.  A budget name is a
        balance term (``prod_x``, ...) or a stored density
        (``P_U_x``, ...), and the virtual ``sum_<suffix>`` adds the
        balance terms that make up `$\partial_t E_\Delta$`
        (:data:`NON_ADDITIVE_TERMS`).  A balance term is a linear
        combination of stored densities, so it is taken of their
        ensemble means.
        """
        if name in self._cache:
            return self._cache[name]
        base, _, suffix = name.rpartition("_")
        if base == "sum":
            value = np.sum(
                [self.field(n) for n in self.additive(suffix)], axis=0
            )
        elif self.stem == "twin_ybudget" and (
            base in COMBINED_PANELS or base in BALANCE_TERMS
        ):
            value = balance_field(self.field, self.meta, name)
        else:
            total = None
            for index in range(self.n_members):
                records, rows = self.source(index, name)
                block = np.asarray(records[name][rows], dtype=np.float64)
                total = block if total is None else total + block
            value = total / self.n_members
        self._cache[name] = value
        return value

    def additive(self, suffix: str) -> list[str]:
        r"""The balance terms that add up to `$\partial_t E_\Delta$`.

        Every term of one marginal but :data:`NON_ADDITIVE_TERMS` --
        what the virtual ``sum_<suffix>`` adds, wherever it is read.
        The driving input is among them whether or not a panel draws
        it, so the sum is the whole rate.
        """
        names = [
            f"{term}_{suffix}"
            for term in self.terms
            if term not in NON_ADDITIVE_TERMS
        ]
        if not names:
            raise ValueError(f"{self.stem}: no additive budget terms")
        return names

    def reduced(self, reduce) -> np.ndarray:
        r"""Ensemble mean of a per-record reduction, read in chunks.

        *reduce* is handed ``read(name)``, the float64 block of one
        stored field over up to :data:`_REF_CHUNK` of one member's
        selected records, and returns its reduction with the records
        still on the leading axis.  Nothing is cached and no field is
        ever held whole: where :meth:`field` keeps ``n_frames``
        records of `$(3, n_y, n_k)$`, a `$k$`-sum wants
        `$(3, n_y)$` of each, and at every sample of a long production
        member the first is gigabytes (module docstring, "Spacetime
        maps").
        """
        total = None
        n_frames = self.rows.shape[1]
        for index in range(self.n_members):
            parts = []
            for start in range(0, n_frames, _REF_CHUNK):
                frames = slice(start, start + _REF_CHUNK)

                def read(name: str, index=index, frames=frames) -> np.ndarray:
                    records, rows = self.source(index, name)
                    return np.asarray(
                        records[name][rows[frames]], dtype=np.float64
                    )

                parts.append(reduce(read))
            block = np.concatenate(parts, axis=0)
            total = block if total is None else total + block
        return total / self.n_members

    def reference_spectrum(self, marginal: str) -> np.ndarray:
        r"""`$\langle r_\alpha(y, m)\rangle_t$`, `$(0, 0)$` mode off.

        ``(3, n_y, n_k)``: the member set's mean reference spectrum
        over the **distinct** reference instants, with the mean mode
        taken off its `$m = 0$` column
        (:func:`~dnsjax.analysis.twin.yspectra.mean_free_spectrum`).
        The divisor of `$\mathcal{R}$`, and the array the other two
        readings below reduce -- so "a reference divisor never carries
        the `$(0, 0)$` mode" is one statement about one array rather
        than three that could drift (module docstring,
        "Decorrelation").

        Only the two **complete** marginals are held
        (:data:`NORMALISED_MARGINALS`), which is every one a divisor
        is ever built from: neither decorrelation is offered for the
        `$k_x = 0$` plane, and normalising that plane by a reference
        of the whole mode plane is what "Reference normalisation"
        declines to do.  Asking for another marginal is a caller bug
        rather than a missing feature, so it says so.
        """
        spectra = self._resolved_reference()[0]
        if marginal not in spectra:
            raise ValueError(
                f"no reference spectrum is held for the {marginal!r} "
                f"marginal (only {sorted(spectra)}): a divisor is built "
                "from a complete sum over the mode plane, which the "
                "k_x = 0 plane is not."
            )
        return spectra[marginal]

    def reference_profile(self) -> np.ndarray:
        r"""`$D_\alpha(y)$`, ``(3, n_y)``: that spectrum's `$k$`-sum.

        The reference field's fluctuation energy at each wall
        distance, and the divisor of `$\mathcal{R}^k$`.  Read off the
        `$k_z$` marginal; the `$k_x$` one gives the same profile
        (:func:`~dnsjax.analysis.twin.yspectra.fluctuation_profile`),
        which :meth:`_check_marginals` has already asserted.
        """
        return self.reference_spectrum("x").sum(axis=-1)

    def reference_scale(self) -> np.ndarray:
        r"""`$E^{\mathrm{ref}}_\alpha$` per component, ``(3,)``.

        That profile's wall-normal average: the reference field's
        total-in-`$(y, k)$` energy without its `$(0, 0)$` mode, the
        one number every normalised panel of a component is divided
        by (module docstring, "Reference normalisation").
        """
        return np.einsum("j,cj->c", self.y_weights, self.reference_profile())

    def reference_report(self) -> list[str]:
        """The lines describing that normalisation, for printing."""
        return self._resolved_reference()[1]

    def _resolved_reference(self) -> tuple[dict[str, np.ndarray], list[str]]:
        """The mean reference spectra and their report, built once."""
        if self._reference is None:
            self._reference = self._build_reference()
        return self._reference

    def _build_reference(self) -> tuple[dict[str, np.ndarray], list[str]]:
        r"""Accumulate the mean reference record over the members.

        One pass over the distinct instants, accumulating the two
        complete reference marginals and the `$(0, 0)$` mode;
        everything a normalised or decorrelation panel divides by is a
        reduction of what comes out (:meth:`reference_spectrum`).
        Accumulating the marginals rather than the scalar is what
        makes the `$y$`- and `$k$`-resolved divisors available at all,
        and it is exact: every step from the stored entry to
        `$E^{\mathrm{ref}}$` is linear, so the mean of the reductions
        is the reduction of the mean.  It reads three fields per
        record where the scalar needed one; ``--ref-stride`` is the
        lever if that pass is the expensive one.
        """
        if "r" not in self.prefixes:
            raise ValueError(
                f"{self.stem}: the stream carries no reference spectra "
                "(twin.spectra_ref was off), so there is no E_ref to "
                "normalise the difference spectra by."
            )
        picks, n_instants, n_samples = self._distinct_instants()
        mean_name = mean_mode_name(self.meta, "r")
        # The two complete marginals only, never the ``x0`` plane a
        # stream may also carry: nothing divides by it
        # (:meth:`reference_spectrum`), and reading it here would be a
        # third more of this pass -- which is the whole stream, every
        # member, and the most expensive thing the script does.
        wanted = [suf for suf in self.suffixes if suf in NORMALISED_MARGINALS]
        totals = {suf: np.zeros(()) for suf in wanted}
        mean_total = np.zeros(())
        for member, rows in zip(self.members, picks, strict=True):
            if rows.size == 0:  # every instant is another member's
                continue
            self._check_marginals(member, int(rows[0]))
            records = member.ref.records
            for start in range(0, rows.size, _REF_CHUNK):
                take = rows[start : start + _REF_CHUNK]
                mean_total = mean_total + mean_mode_profile(
                    np.asarray(records[mean_name][take], dtype=np.float64),
                    mean_name,
                ).sum(axis=0)
                for suf in wanted:
                    totals[suf] = totals[suf] + np.asarray(
                        records[f"r_{suf}"][take], dtype=np.float64
                    ).sum(axis=0)
        mean_mode = mean_total / n_instants
        spectra = {
            suf: mean_free_spectrum(total / n_instants, mean_mode)
            for suf, total in totals.items()
        }
        scale = np.einsum("j,cjk->c", self.y_weights, spectra["x"])
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
        return spectra, report

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
        times = [m.ref.t_abs[:: self.ref_stride] for m in self.members]
        rows = [m.ref.rows[:: self.ref_stride] for m in self.members]
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
        block = member.ref.records[row]
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
                f"the reference energy at t = "
                f"{member.ref.records['t'][row]:g} "
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


def _nearest(times: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """Index in ascending *times* of the nearest value to each *wanted*.

    Ties (a *wanted* exactly between two samples) go to the earlier
    one; :func:`_match` applies the tolerance.
    """
    after = np.searchsorted(times, wanted)
    before = np.clip(after - 1, 0, times.size - 1)
    after = np.clip(after, 0, times.size - 1)
    nearer = np.abs(times[before] - wanted) <= np.abs(times[after] - wanted)
    return np.where(nearer, before, after)


def _match(
    times: np.ndarray, wanted: np.ndarray, atol: float = _T_ATOL
) -> np.ndarray:
    """Index in ascending *times* of each *wanted* value, or ``-1``.

    Nearest neighbour within *atol*.  A tolerance rather than an
    equality on rounded keys: two members reach one sample time by
    different arithmetic, so a pair that straddles a bin's edge would
    round apart and drop the frame from the shared grid -- silently,
    since a shorter grid is exactly what a short member produces.

    At the default :data:`_T_ATOL` both arrays are ascending and
    separated by far more than the tolerance
    (:func:`_check_cadence`), so at most one sample can match.  Under
    ``--align-atol`` (:func:`open_series`) the tolerance approaches
    half a cadence and several may fall inside it; the nearest is
    still one sample, and a tolerance above half a cadence -- where
    two frames could claim it -- is refused there.
    """
    if times.size == 0:
        return np.full(wanted.shape, -1, dtype=int)
    nearest = _nearest(times, wanted)
    return np.where(np.abs(times[nearest] - wanted) <= atol, nearest, -1)


def _cadence(grids: list[np.ndarray]) -> float:
    """The coarsest member's sampling interval, or ``inf``.

    The median gap: a stream carries a few rows off its own cadence
    grid (a resume seam, the driver's unconditional final row) and the
    median is the cadence anyway.  ``inf`` when no member has two
    samples, which leaves both the phase test and the tolerance below
    inert -- there is no grid to have a phase on.
    """
    gaps = [float(np.median(np.diff(g))) for g in grids if g.size > 1]
    return min(gaps) if gaps else float("inf")


def _phase_offsets(grids: list[np.ndarray], cadence: float) -> np.ndarray:
    r"""Each later member's grid displacement against the first's.

    A member samples at `$p + n\,c$`; this returns `$p_i - p_0$` for
    every member after the first, folded onto
    `$[-c/2, c/2)$`.  Zero to the last bits for members recorded on
    one clock, and a nonzero value is a **phase** offset -- the whole
    grid displaced, so the members have no relative sample time in
    common beyond whatever coincides by accident.

    Modulo the cadence rather than a distance, so that members
    covering *different stretches* of the relative clock -- a short
    member, one that starts later -- still read as in phase; their
    grids do not overlap and that is the intersection's business, not
    this one's.  A median rather than a mean, so the handful of
    off-grid rows a stream carries cannot move it, and the
    **difference** is folded as well as each phase, so two in-phase
    members landing on opposite sides of the fold read as the same
    phase rather than as a whole cadence apart.

    ``dnsjax-twin`` anchors every cadence on the member's own
    perturbation step, so a set recorded by it is in phase whatever
    iteration numbers its parent snapshots carried
    (``dnsjax.twin.driver``, "Sample-cadence anchor").  Members
    written before that are not, whenever their parents' ``it``
    differed modulo the cadence.
    """
    if len(grids) < 2 or not np.isfinite(cadence):
        return np.zeros(0)

    def fold(x):
        return (x + 0.5 * cadence) % cadence - 0.5 * cadence

    phases = [float(np.median(fold(g))) for g in grids]
    return fold(np.array(phases[1:]) - phases[0])


def _check_phase(
    opened: list[_Member],
    grids: list[np.ndarray],
    atol: float,
    cadence: float,
) -> None:
    r"""Refuse a member set whose relative grids are out of phase.

    The members' sample times must name the same relative instants to
    within *atol*, or the shared frame grid below is whatever few
    times coincide by accident -- in the worst case `$t = 0$` alone,
    which the driver records unconditionally.  That produced a single
    frame per stream and said nothing, so it is an error here instead
    (module docstring, "Members out of phase").

    The message quotes the offset (:func:`_phase_offsets`) and the
    ``--align-atol`` that would accept it, which is never more than
    half a cadence: a folded offset cannot be, and neither can the
    distance to the nearest sample of a uniform grid.
    """
    offsets = _phase_offsets(grids, cadence)
    if offsets.size == 0 or float(np.max(np.abs(offsets))) <= atol:
        return
    worst = float(np.max(np.abs(offsets)))
    named = ", ".join(
        f"{member.path}: {offset:+.6g}"
        for member, offset in zip(opened[1:], offsets, strict=True)
        if abs(offset) > atol
    )
    raise ValueError(
        f"{opened[0].path}: the members' sample grids are out of "
        f"phase on the relative clock t - t_parent, by up to "
        f"{worst:.6g} against this one ({named}), which is more than "
        f"the {atol:g} that decides whether two of them are the same "
        "instant -- so they share no frame beyond t = 0.  A member "
        "whose parent snapshot sat at an iteration number that is "
        "not a multiple of the sample cadence was recorded on a "
        "displaced grid; dnsjax-twin anchors the cadence on the "
        "member's own perturbation step and no longer does that.  To "
        "use the members as recorded, pass --align-atol "
        f"{min(1.05 * worst, 0.5 * cadence):.6g} (up to half the "
        f"{cadence:.6g} cadence), which pairs each frame with every "
        "member's nearest sample instead of its own."
    )


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
    align_atol: float = _T_ATOL,
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

    *align_atol* is how far apart two members' samples may be and
    still be one frame (module docstring, "Members out of phase").
    The default :data:`_T_ATOL` demands the same relative instant and
    a set that cannot meet it is **refused**, naming the offset it
    would take; raising it accepts members whose grids are displaced
    in phase, at the cost of averaging fields recorded up to that far
    apart on their own clocks.
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
    # Half a cadence is as wide as the tolerance may go.  It buys
    # nothing beyond that -- a nearest neighbour inside a uniform grid
    # is never further away -- and it costs: past the end of a short
    # member's grid the nearest sample is its last one, at any
    # distance, so a wider tolerance would quietly extend that member
    # over every later frame instead of ending its contribution.
    cadence = _cadence(grids)
    if align_atol > 0.5 * cadence:
        raise ValueError(
            f"align_atol = {align_atol:g} is more than half the "
            f"{cadence:g} sample cadence.  Nothing above that pairs "
            "a frame with a nearer sample; it only lets a member "
            "whose stream ends early keep contributing its last "
            "record to every frame after it."
        )
    _check_phase(opened, grids, align_atol, cadence)
    matched = np.array(
        [int((_match(g, grids[0], align_atol) >= 0).sum()) for g in grids]
    )
    keep = np.arange(grids[0].size)
    for other in grids[1:]:
        keep = keep[_match(other, grids[0][keep], align_atol) >= 0]
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
    picks = [_match(grid, common, align_atol) for grid in grids]
    return YSeries(
        stem=stem,
        members=tuple(opened),
        rows=np.stack(
            [
                member.rows[pick]
                for member, pick in zip(opened, picks, strict=True)
            ]
        ),
        index=keep,
        t_rel=common,
        t_members=np.stack(
            [grid[pick] for grid, pick in zip(grids, picks, strict=True)]
        ),
        matched=matched,
        meta=opened[0].meta,
        ref_stride=ref_stride,
        ref_rows=reference_rows(opened, common, align_atol),
    )


def reference_rows(
    members, frames: np.ndarray, atol: float = _T_ATOL
) -> np.ndarray | None:
    """Each member's reference record at each frame, or ``None``.

    ``(n_members, n_frames)``: the reference record matched to each
    relative time in *frames* the way the frames themselves are
    (:func:`_match`), ``-1`` where a member's reference has no sample
    -- which only a ``twin.it_yspectra_ref`` off the difference
    cadence leaves (:meth:`YSeries.source`).  ``None`` unless every
    member has reference spectra.
    """
    if not all(member.ref is not None for member in members):
        return None
    rows = []
    for member in members:
        pick = _match(member.ref.t_rel, frames, atol)
        rows.append(
            np.where(pick >= 0, member.ref.rows[np.maximum(pick, 0)], -1)
        )
    return np.stack(rows)


# ── Map construction ─────────────────────────────────────────────────


@dataclass(frozen=True)
class MapOptions:
    r"""Everything that turns a stored field into a plotted map.

    *premultiply* is ``"k"`` (the default), ``"ky"`` (the paper's
    (2.8)) or ``"none"``, and *y_log* draws the ordinate logarithmic
    (the default); the two are independent, and the default pair is
    the usual convention for a spectrum rather than the paper's own
    (module docstring, "Premultiplication").  `$\mathcal{R}$` and the
    shape maps fix their own premultiplier instead
    (:func:`map_premultiplier`).  *half* is how the two
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


def _drawn_rows(
    y: np.ndarray, y_log: bool, ylim: tuple[float, float] | None
) -> np.ndarray:
    """Which wall distances an axis shows, within *ylim*, as a mask.

    The rows :meth:`Map.drawn` keeps and the columns
    :meth:`SpacetimeMap.drawn` keeps, by the rules the first of them
    documents, and the rows a shape map's peak is read over
    (:func:`make_map`), before there is a :class:`Map` to ask.
    """
    shown = y > 0.0 if y_log else np.ones(y.size, dtype=bool)
    keep = shown.copy()
    if ylim is not None:
        keep &= (y >= ylim[0]) & (y <= ylim[1])
        inside = np.flatnonzero(keep)
        if inside.size:  # the interpolation neighbours, if any
            keep[max(int(inside[0]) - 1, 0)] = True
            keep[min(int(inside[-1]) + 1, keep.size - 1)] = True
        keep &= shown
    return keep


@dataclass(frozen=True)
class Map:
    r"""One panel's data: ``values`` on the `$(\lambda, y)$` grid.

    ``peak`` is set on a shape map alone: the frame's own peak it was
    divided by, which is what :func:`draw_map` reads to put it on the
    `$[0, 1]$` scale it has by construction.
    """

    lam: np.ndarray  # (n_lam,) wavelength, plotted units
    y: np.ndarray  # (n_y,) wall distance, plotted units
    values: np.ndarray  # (n_y, n_lam)
    title: str  # LaTeX panel title, with normalisation
    name: str  # the stored field it came from
    non_negative: bool  # declared or inferred; sets the colour family
    y_log: bool = False  # whether the ordinate is drawn logarithmic
    peak: float | None = None  # a shape map's divisor, plotted units

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
        keep = _drawn_rows(self.y, self.y_log, ylim)
        return self.y[keep], self.values[keep]


def _select_half(
    values: np.ndarray, y: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the channel onto one wall distance.

    Index ``j`` pairs with ``n_y - 1 - j`` -- checked against the grid
    rather than assumed -- and for odd ``n_y`` the mid-plane pairs with
    itself, so ``"mean"`` neither needs a special case there nor
    double counts it.  Every stored quantity is even under the flow's
    own mid-plane symmetry (plane Poiseuille's reflection, plane
    Couette's rotation), so the mean carries no sign flips; the
    argument is in the module docstring, "Folding the channel".  Returns the
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
            "so the mid-plane fold has no partner row to average against, "
            "and --half upper no wall distance to label its rows with; "
            "use --half lower, which needs neither."
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


def _ratio(numerator: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """*numerator* over *divisor*, ``nan`` where that is undefined.

    A reference energy vanishes at the wall, where every velocity
    component does, so the wall row of a decorrelation is `$0/0$` --
    and on a small enough member set a high wavenumber can be empty
    too.  Those become ``nan`` rather than an error, a warning or an
    infinity: ``nan`` is what every colour scale here already drops
    (:func:`scan_panels` filters on ``isfinite``) and what
    ``contourf`` / ``pcolormesh`` already leave unpainted, whereas an
    infinity would set a scale no contour could reach.
    """
    out = np.full(np.broadcast_shapes(numerator.shape, divisor.shape), np.nan)
    np.divide(numerator, divisor, out=out, where=divisor > 0.0)
    return out


def _symmetrise_y(values: np.ndarray, half: str, axis: int = -2) -> np.ndarray:
    r"""A divisor made symmetric about the centreline, under a fold.

    ``--half mean`` averages `$y$` with `$-y$`, and the mean of two
    ratios is not the ratio of two means once the divisor depends on
    `$y$` -- so a decorrelation would come out depending on whether
    the fold ran before or after the division.  Symmetrising the
    divisor first removes the choice: division by a symmetric profile
    commutes with the fold exactly, and the folded map is then the
    ratio of the folded halves, which is what a channel-averaged
    figure claims to show.  ``lower`` / ``upper`` fold nothing and
    keep each row's own divisor.

    The reference is mid-plane symmetric in the mean anyway, so this
    moves the numbers by the run's own asymmetry and no more; what it
    buys is that the answer no longer depends on the order.
    """
    if half != "mean":
        return values
    return 0.5 * (values + np.flip(values, axis=axis))


def map_divisor(
    series: YSeries, base: str, marginal: str, half: str
) -> np.ndarray | None:
    r"""The `$(3, n_y, n_k)$` divisor of a decorrelation, or ``None``.

    `$\mathcal{R}$` divides mode by mode and `$\mathcal{R}^k$` by the
    same reference summed over `$k$`, broadcast back over it; both
    come off :meth:`YSeries.reference_spectrum`, which has the
    `$(0, 0)$` mode off already, and both are symmetrised for the fold
    (:func:`_symmetrise_y`).  The caller applies the factor of two.
    """
    if base == DECORR:
        divisor = series.reference_spectrum(marginal)
    elif base == DECORR_K:
        divisor = series.reference_profile()[..., None]
    else:
        return None
    return _symmetrise_y(divisor, half)


def map_premultiplier(base: str, options: MapOptions) -> str:
    r"""Which premultiplier a map of *base* takes: ``k``, ``ky`` or ``none``.

    ``--premultiply``'s, for `$\mathcal{R}^k$` and every absolute map,
    and fixed for the other two.  `$\mathcal{R}$` takes none: it
    divides mode by mode, so its `$k$` would cancel against its
    divisor's, and the map is no longer additive in `$k$` for a `$y$`
    factor to act on either (module docstring, "Decorrelation").  A
    shape map takes both, whatever ``--premultiply`` says (module
    docstring, "Shape maps").
    """
    if base == DECORR:
        return "none"
    if base == SHAPE:
        return "ky"
    return options.premultiply


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
    """Whether :data:`NON_NEGATIVE` covers a name.

    Stored (``e_x``), virtual (``decorr_k_x``, ``diss_z``) or bare
    (``e``, ``sum``): a trailing marginal is stripped and anything
    else is looked up whole, so the `$k$`-summed spacetime bases
    resolve to the same declaration their marginals do.
    """
    base, _, suffix = name.rpartition("_")
    return (base if suffix in MARGINALS else name) in NON_NEGATIVE


def declared_non_positive(name: str) -> bool:
    """Whether :data:`NON_POSITIVE` covers a name, looked up as
    :func:`declared_non_negative` looks one up."""
    base, _, suffix = name.rpartition("_")
    return (base if suffix in MARGINALS else name) in NON_POSITIVE


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
    r"""*value* to *digits* significant figures, as LaTeX math.

    ``%g``'s exponent is spelled out, so a title reads
    `$1.234 \times 10^{-3}$` rather than ``1.234e-03``.
    """
    text = f"{value:.{digits}g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def panel_symbol(
    series: YSeries, name: str, component: int | None
) -> tuple[str, str]:
    r"""``(sign, symbol)`` of the quantity one map panel draws.

    Before any premultiplier or normalisation: a balance term's
    write-up symbol and the sign of its contribution
    (:data:`TERM_LABELS`), or a spectrum's `$E$` with its marginal and
    component.  What :func:`field_title` builds a title on and what a
    track's legend names its panel by (:func:`track_figure`).  Not for
    a decorrelation, whose symbol carries no marginal.
    """
    base, _, suffix = name.rpartition("_")
    if field_kind(series) == "rate":
        return TERM_LABELS.get(base, ("", base.replace("_", r"\_")))
    superscript = MARGINALS[suffix][1]
    # A shape map draws the difference spectrum too.
    delta = r"\Delta " if base in ("e", SHAPE) else ""
    if component is None:
        return "", rf"\sum_\alpha E^{{{superscript}}}_{{{delta}\alpha}}"
    return "", rf"E^{{{superscript}}}_{{{delta}{COMPONENTS[component]}}}"


def panel_label(series: YSeries, base: str, component: int | None) -> str:
    """Which panel of its figure a field is, in plain text.

    Its term on a budget figure, its component (or ``sum``) on a
    spectra one: what a spacetime ``.npz`` and a track ``.npz`` name
    their rows by.
    """
    if series.stem == "twin_ybudget":
        return base
    return "sum" if component is None else COMPONENTS[component]


def field_title(
    series: YSeries,
    name: str,
    component: int | None,
    options: MapOptions,
    peak: float | None = None,
) -> str:
    r"""The LaTeX panel title for one stored (or virtual) field.

    A normalised panel (:func:`normalises`) gets a second line
    carrying its `$E^{\mathrm{ref}}$` in the units the figure is
    drawn in, which is what makes its colour bar recoverable as an
    absolute one; :func:`panel_geometry` budgets the extra line.  A
    shape map's second line carries its *peak* the same way, for the
    same reason.
    """
    base, _, suffix = name.rpartition("_")
    axis = MARGINALS[suffix][0]
    wavenumber = rf"k_{{{axis}}}"
    kind = field_kind(series)
    plus = options.units.suffix
    factor = {
        "ky": rf"{wavenumber}{plus} y{plus}\,",
        "k": rf"{wavenumber}{plus}\,",
        "none": "",
    }[map_premultiplier(base, options)]
    if base in (DECORR, DECORR_K):
        # Which marginal it is comes off the abscissa (and off the
        # premultiplier where there is one), as it does for a
        # spacetime panel: a decorrelation carries no marginal
        # superscript, that slot being spent on the k of R^k.
        sup = "^{k}" if base == DECORR_K else ""
        sub = "" if component is None else f"_{{{COMPONENTS[component]}}}"
        return rf"${factor}\mathcal{{R}}{sup}{sub}$"
    sign, body = panel_symbol(series, name, component)
    if base == SHAPE:
        # Over the frame's own peak, which keeps the units the map has
        # shed and is reported in the plotted ones, as an E_ref is
        # (module docstring, "Shape maps").
        top = r"\max"
        return (
            f"${factor}{body}/{top}$\n"
            f"${top}{options.units.norm_suffix(kind)} = "
            f"{latex_float(peak)}$"
        )
    scale = reference_norm(series, name, component)
    if scale is None:
        return f"${sign}{factor}{body}{options.units.norm_suffix(kind)}$"
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
    ylim: tuple[float, float] | None = None,
) -> Map:
    r"""Build one premultiplied map from a stored (or virtual) field.

    *name* is a stored field such as ``e_x``, a balance term such as
    ``prod_z`` (:func:`balance_field`), or one of the virtual
    ``sum_x`` / ``decorr_x`` / ``decorr_k_x`` / ``s_x``;
    *component* selects a velocity component of a ``twin_yspectra``
    field (``None`` sums the three).  *frame* indexes the series'
    subsampled records.  *non_negative* overrides the declaration of
    :data:`NON_NEGATIVE` -- what ``--signs-from-data`` and
    :func:`scan_panels` feed back in.

    A spectra panel of a complete marginal is divided through by the
    series' `$E^{\mathrm{ref}}$` (:func:`reference_norm`), and a
    ``decorr*`` panel by its own resolved divisor
    (:func:`map_divisor`) -- both **after** the component reduction,
    so that the summed panel is one ratio of sums rather than a sum of
    three ratios.  The divisor is applied before the `$m = 0$` column
    is dropped: that column is not drawn, but it is the one whose
    divisor a `$(0, 0)$`-carrying reference would get wrong, and a
    ratio that is only right where it happens to be plotted is not
    worth having.

    A shape panel (``s_*``) is the absolute ``e_*`` panel under the
    `$k\,y$` premultiplier, divided **last** by its own peak over the
    rows the box shows -- *ylim*, :func:`y_limits`' default when
    ``None`` -- so that it spans `$[0, 1]$` there; the peak is
    :attr:`Map.peak` and its title's second line (module docstring,
    "Shape maps").  Every other map ignores *ylim*.
    """
    base, _, suffix = name.rpartition("_")
    decorr = base in (DECORR, DECORR_K)
    stored = f"e_{suffix}" if decorr or base == SHAPE else name
    values = series.field(stored)[frame]
    if series.stem == "twin_yspectra":
        divisor = map_divisor(series, base, suffix, options.half)
        if component is None:
            values = values.sum(axis=0)
            divisor = None if divisor is None else divisor.sum(axis=0)
        else:
            values = values[component]
            divisor = None if divisor is None else divisor[component]
        if divisor is not None:
            values = _ratio(values, 2.0 * divisor)
    elif component is not None:
        raise ValueError(f"{series.stem}: {name} has no component axis")

    values = values[:, 1:]  # lambda = L/m has no place at m = 0
    premultiply = map_premultiplier(base, options)
    if premultiply != "none":
        values = values * series.harmonics(suffix)[None, 1:]
    if not decorr:
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
    if premultiply == "ky":
        # A second logarithmic axis needs its own premultiplier, in
        # the units the ordinate is drawn in (module docstring).
        values = values * y[:, None]
    values = _running_mean(values, options.smooth)
    peak = None
    if base == SHAPE:
        # Over the frame's own peak, read off the rows the box shows,
        # as every colour scale is; an identically zero field has none
        # and stays zero rather than going nan.
        rows = _drawn_rows(y, options.y_log, y_limits(series, options, ylim))
        shown = values[rows]
        finite = shown[np.isfinite(shown)]
        peak = float(finite.max()) if finite.size else 0.0
        if peak > 0.0:
            values = values / peak
    return Map(
        lam=options.units.length(series.wavelengths(suffix)),
        y=y,
        values=values,
        title=field_title(series, name, component, options, peak),
        name=name,
        non_negative=(
            declared_non_negative(name)
            if non_negative is None
            else non_negative
        ),
        y_log=options.y_log,
        peak=peak,
    )


# ── Colour scales ────────────────────────────────────────────────────


@dataclass(frozen=True)
class PanelScale:
    """One panel's series-global range and its sign family.

    *frame_lo* / *frame_hi* are each frame's own extremes over the same
    rows, ``nan`` where a frame has nothing finite: what ``--clim
    ramped`` accumulates (:meth:`data_range`).  A spacetime panel, one
    figure for the whole run, has neither.
    """

    lo: float
    hi: float
    non_negative: bool
    frame_lo: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    frame_hi: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )

    def data_range(self, clim: str, frame: int) -> tuple[float, float] | None:
        """The range one frame's levels are read from, under *clim*.

        ``series`` is the frozen range, ``frame`` none (the figure
        reads its own), and ``ramped`` the extremes of every frame up
        to and including *frame*, each side separately (module
        docstring, "Colour scales").  That range always contains the
        frame's own, never shrinks, and is the frozen range itself from
        the frame holding the series extreme onward.  Before any frame
        has anything finite it is ``(0, 0)``, which draws as
        identically zero, as an empty series' frozen range does.
        """
        if clim == "series":
            return self.lo, self.hi
        if clim == "frame":
            return None
        if clim != "ramped":
            raise ValueError(f"clim must be series/frame/ramped, not {clim!r}")
        seen_lo = self.frame_lo[: frame + 1]
        seen_hi = self.frame_hi[: frame + 1]
        if np.all(np.isnan(seen_lo)):
            return 0.0, 0.0
        return float(np.nanmin(seen_lo)), float(np.nanmax(seen_hi))


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
    floors away.  Each frame's own extremes over those rows are kept
    as well, for ``--clim ramped`` (:meth:`PanelScale.data_range`).
    """
    scales: dict[tuple[str, int | None], PanelScale] = {}
    notes: list[str] = []
    for name, component in panels:
        frame_lo = np.full(series.t_rel.size, np.nan)
        frame_hi = np.full(series.t_rel.size, np.nan)
        for frame in range(series.t_rel.size):
            _, values = make_map(
                series,
                name,
                frame,
                options=options,
                component=component,
                non_negative=True,  # irrelevant here; decided below
                ylim=ylim,
            ).drawn(ylim)
            finite = values[np.isfinite(values)]
            if finite.size:
                frame_lo[frame] = finite.min()
                frame_hi[frame] = finite.max()
        seen = np.isfinite(frame_lo)
        lo = float(frame_lo[seen].min()) if seen.any() else 0.0
        hi = float(frame_hi[seen].max()) if seen.any() else 0.0
        label = (
            name if component is None else f"{name}[{COMPONENTS[component]}]"
        )
        non_negative = declared_non_negative(name) if declared else lo >= 0.0
        note = None
        if declared and non_negative:
            note = _sign_note(label, lo, hi)
        elif declared and declared_non_positive(name):
            note = _sign_note(label, lo, hi, positive=False)
        if note is not None:
            notes.append(note)
        scales[(name, component)] = PanelScale(
            lo, hi, non_negative, frame_lo, frame_hi
        )
    return scales, notes


def _sign_note(
    label: str, lo: float, hi: float, *, positive: bool = True
) -> str | None:
    """What a declared one-signed field's excursion across zero earns.

    *positive* is the :data:`NON_NEGATIVE` declaration, whose negative
    minimum is judged; otherwise the :data:`NON_POSITIVE` one, whose
    positive maximum is judged the same way.  ``None`` where there is
    nothing to say.  Shared by the two scans (:func:`scan_panels`,
    :func:`spacetime_scales`), which differ in what they sweep and not
    in how they judge a sign.
    """
    peak = max(abs(lo), abs(hi))
    stray = -lo if positive else hi
    if stray <= 0.0 or peak <= 0.0:
        return None
    ratio = stray / peak
    verdict = (
        "round-off, i.e. truncation"
        if ratio < SIGN_TOLERANCE
        else "ABOVE round-off -- worth a look"
    )
    if not positive:
        return (
            f"  {label}: declared non-positive, max/peak = "
            f"{ratio:.3e} ({verdict}); drawn signed"
        )
    return (
        f"  {label}: declared non-negative, min/peak = "
        f"-{ratio:.3e} ({verdict}); drawn as non-negative"
    )


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
    (``--clim series`` / ``ramped``); without it the map's own extremes
    are used.
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
    levels: np.ndarray, cmap: str, *, non_negative: bool, log: bool = False
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
      a one-sided term such as `$\mathcal{P}_\Delta^{\mathbf{U}}$`,
      whose negative excursion is a percent of its positive one, would
      otherwise spend the entire blue half of the colour map on a
      single band and read as unsigned.

    A signed field that never changes sign is the same statement with
    one side empty, and gets the whole ramp of the side it does use.
    Note that "neutral" is the colour map's own centre, which for the
    default ``RdBu_r`` is ColorBrewer's near-white ``#f7f6f6`` rather
    than the page; ``--cmap-signed bwr`` centres on pure white.

    With *log* the bands are read logarithmically instead -- their
    **geometric** midpoints through a
    :class:`~matplotlib.colors.LogNorm` spanning the level range --
    which is what makes a decade of a spacetime map's log scale
    (:func:`log_levels`) one fixed step of colour.  A value-linear
    norm over log-spaced bands would spend the whole ramp on the top
    decade.  It implies non-negative levels and takes the same
    unfilled floor: what falls below the first is left transparent,
    and the colour bar's ``extend`` is what says so.
    """
    base = plt.get_cmap(cmap)
    layers = 0.5 * (levels[:-1] + levels[1:])  # what contourf colours
    if log:
        layers = np.sqrt(levels[:-1] * levels[1:])
        scale = LogNorm(float(levels[0]), float(levels[-1]))
    elif non_negative:
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


def log_floor(values: np.ndarray, decades: float) -> float:
    r"""Where a **logarithmic** colour scale should stop, from below.

    An energy is bounded below by zero, which a logarithmic scale
    cannot reach, so one has to be told where to stop.  Two answers
    are informative and this takes whichever is the higher:

    - *decades* below the peak, which is what a growth phase needs --
      a difference field climbs several decades from ``twin.e0`` to
      saturation, and a scale that reached the smallest number in the
      array would spend most of itself on round-off;
    - the smallest positive value drawn, where the data spans fewer
      decades than that, so no empty range is invented below it.

    Read off exactly the values a caller passes -- the drawn window,
    not the stored array (:meth:`SpacetimeMap.drawn`).  Returns
    ``0.0`` when nothing positive is drawn, which is the caller's
    signal that there is no logarithmic map to make.
    """
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return 0.0
    return max(float(positive.max()) / 10.0**decades, float(positive.min()))


def log_levels(
    lo: float, hi: float, per_decade: int = _LOG_BANDS_PER_DECADE
) -> np.ndarray:
    """Log-spaced band edges covering ``[lo, hi]``.

    The logarithmic counterpart of :func:`contour_levels`, and
    deliberately not a *round*-stepped one: the labels a reader wants
    off this scale are the decades, which :func:`_decade_ticks` picks
    whatever the band count, so the bands themselves are free to be
    fine enough to read as a ramp.
    """
    if not 0.0 < lo < hi:
        return np.asarray([], dtype=float)
    bands = max(int(math.ceil(math.log10(hi / lo) * per_decade)), 1)
    return np.logspace(math.log10(lo), math.log10(hi), bands + 1)


def _decade_ticks(levels: np.ndarray) -> np.ndarray:
    """The whole decades inside a logarithmic scale's range.

    Its colour-bar ticks and its contour lines both: a line per band
    would be mush, and a decade is the one spacing a log scale can be
    read against.  Falls back to the range's ends where it spans less
    than a decade.
    """
    lo, hi = float(levels[0]), float(levels[-1])
    ticks = 10.0 ** np.arange(
        math.ceil(math.log10(lo)), math.floor(math.log10(hi)) + 1
    )
    return ticks if ticks.size else np.asarray([lo, hi])


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
    for the visible map whether the scale is frozen or ramped
    (*data_range*, already restricted by :func:`scan_panels`) or taken
    from this frame.  Everything inside the box is still drawn from the
    unrestricted array.

    A shape map (:attr:`Map.peak`) is drawn on `$[0, 1]$` whatever
    *data_range* and *quantile* say: it is over its own peak, so that
    is its scale in every frame by construction (module docstring,
    "Shape maps").
    """
    if map_.peak is not None:
        data_range, quantile = (0.0, 1.0), None
    ax.set_xscale("log")
    ax.set_yscale("log" if map_.y_log else "linear")
    y, values = map_.drawn()
    # The scale is a legend for what the box shows, so the levels are
    # read off the rows inside *ylim* -- while the fill still gets the
    # unrestricted array, so a contour reaches the edge of the box
    # (:meth:`Map.drawn`).  Under ``--clim series`` / ``ramped``
    # *data_range* has already been restricted the same way
    # (:func:`scan_panels`); this is what makes ``--clim frame`` and
    # ``--quantile`` agree with it.
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
    so a figure of *ncols* columns comes out exactly that wide.
    *ncols* is a spectra figure's column count; a budget figure is
    :data:`BUDGET_NCOLS` wide whatever it says, at the same panel size.
    *box_aspect* is the axes box's height over its width, and applies
    only to a **linear** ordinate -- a logarithmic one takes the same
    decade length as the abscissa instead (module docstring, "Figure
    geometry").  *clim* is ``--clim``: ``series``, ``frame`` or
    ``ramped`` (:meth:`PanelScale.data_range`).
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
    clim: str = "series"
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


def _fitted_box_width(style: PlotStyle) -> float:
    """The axes-box width that makes a figure exactly ``--width``."""
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
    return box_w


def panel_geometry(
    n_panels: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    style: PlotStyle,
    *,
    y_log: bool,
    x_log: bool = True,
    title_lines: int = 1,
    ncols: int | None = None,
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

    *x_log* is what a **spacetime** map turns off: its abscissa is the
    wall distance, which ``--yscale linear`` draws linearly and whose
    limits then start at the wall, where a decade count is
    `$\log 0$`.  The box takes the fitted width in that case, and the
    decade rule applies to neither axis.

    *title_lines* is the tallest panel title of the figure, in lines:
    the top margin grows by :data:`_TITLE_LINE` for each one past the
    first, which is what makes room for the `$E^{\mathrm{ref}}$` a
    normalised panel reports (:func:`field_title`).

    *ncols* is the figure's own column count where it is not
    ``style.ncols``: a budget figure's (:func:`figure_columns`).  The
    decade length is fitted on ``style.ncols`` either way, so such a
    figure keeps the spectra's panel size and is wider or narrower than
    ``style.width`` instead.
    """
    columns = style.ncols if ncols is None else ncols
    nrows = math.ceil(n_panels / columns)
    m_top = _M_TOP + (title_lines - 1) * _TITLE_LINE
    if x_log:
        decades_x = math.log10(xlim[1] / xlim[0])
        decade = (
            style.decade
            if style.decade is not None
            else _fitted_box_width(style) / decades_x
        )
        box_w = decade * decades_x
    else:
        box_w = _fitted_box_width(style)
        decade = style.decade if style.decade is not None else box_w
    box_h = (
        decade * math.log10(ylim[1] / ylim[0])
        if y_log
        else box_w * style.box_aspect
    )
    fig_w = (
        _M_LEFT
        + columns * (box_w + _COL_AFTER)
        + (columns - 1) * (_COL_GAP + _M_LEFT)
        + _M_RIGHT
    )
    fig_h = (
        _SUP_HEIGHT
        + nrows * (m_top + box_h + _M_BOTTOM)
        + (nrows - 1) * _ROW_GAP
    )
    return Geometry(fig_w, fig_h, box_w, box_h, nrows, columns, m_top)


def figure_columns(series: YSeries) -> int | None:
    """A figure's own column count: :data:`BUDGET_NCOLS` for a budget.

    ``None`` -- ``--ncols`` -- for anything else.  Shared by the map and
    the spacetime figures, so the two budget figures cannot be laid out
    differently (module docstring, "Figure geometry").
    """
    return BUDGET_NCOLS if series.stem == "twin_ybudget" else None


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
    tracks: dict[tuple[str, int | None], PeakTrack] | None = None,
):
    """One figure, one ``(name, component)`` map per panel.

    Shared body of :func:`spectra_panels` and :func:`budget_panels`
    figures.  Every panel of a figure shares one wavelength axis and
    one wall-normal grid, so a single :func:`panel_geometry` sizes
    them all.  A panel with an entry in *tracks* carries its peak's
    track up to this frame (:func:`draw_track`).
    """
    ylim = y_limits(series, options, style.ylim)
    maps = [
        make_map(
            series,
            name,
            frame,
            options=options,
            component=component,
            non_negative=scales[(name, component)].non_negative,
            ylim=ylim,
        )
        for name, component in panels
    ]
    xlim = style.xlim or (maps[0].lam.min(), maps[0].lam.max())
    geometry = panel_geometry(
        len(panels),
        xlim,
        ylim,
        style,
        y_log=options.y_log,
        title_lines=1 + max(m.title.count("\n") for m in maps),
        ncols=figure_columns(series),
    )

    fig = plt.figure(figsize=(geometry.fig_w, geometry.fig_h))
    for panel, (map_, key) in enumerate(zip(maps, panels, strict=True)):
        ax = fig.add_axes(geometry.axes_rect(panel))
        draw_map(
            ax,
            map_,
            units=options.units,
            n_levels=style.n_levels,
            cmap_positive=style.cmap_positive,
            cmap_signed=style.cmap_signed,
            data_range=scales[key].data_range(style.clim, frame),
            quantile=style.quantile,
            nice=style.nice,
            fill=style.fill,
            lines=style.lines,
            cax=fig.add_axes(geometry.cbar_rect(panel)),
            xlim=xlim,
            ylim=ylim,
        )
        if tracks and key in tracks:
            draw_track(
                ax,
                tracks[key],
                frame,
                colour=_TRACK_COLOURS[map_.non_negative],
                y_log=map_.y_log,
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
    """Every map panel of one marginal (:data:`MAP_PANELS`), plus their
    sum, once the stream is known to support them."""
    balance_terms(series.meta)
    return [(f"{t}_{marginal}", None) for t in (*MAP_PANELS, "sum")]


# ── Peak tracking ────────────────────────────────────────────────────


def _trapezoid_widths(x: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights on the ascending grid *x*.

    Half the gap to each neighbour, and half the one gap at either
    end, so they sum to the grid's span: the rule for a field known
    between the first sample and the last and no further, which is
    where a filled contour paints it.  A lone sample is the whole grid.
    """
    if x.size < 2:
        return np.ones(x.size)
    gaps = 0.5 * np.diff(x)
    widths = np.zeros(x.size)
    widths[:-1] += gaps
    widths[1:] += gaps
    return widths


def peak_centroid(
    map_: Map, n_levels: int, ylim: tuple[float, float] | None = None
) -> tuple[float, float]:
    r"""``(lambda_c, y_c)``: the centroid of one map's top band.

    The cells at or above `$(1 - 1/n)$` of the frame's own peak, for
    *n_levels* `$n$`, weighted by the plotted value times their
    trapezoidal widths in `$\ln\lambda$` and `$\ln y$` -- `$y$` on a
    linear ordinate -- and averaged in those same coordinates (module
    docstring, "Peak tracking").  Over the rows the box shows, *ylim*
    read as :meth:`Map.drawn` reads it for the colour scale.  Both in
    the map's plotted units; ``nan`` where nothing is positive.
    """
    y, values = map_.drawn(ylim)
    finite = np.isfinite(values)
    peak = float(values[finite].max()) if finite.any() else 0.0
    if not peak > 0.0:
        return math.nan, math.nan
    top = finite & (values >= (1.0 - 1.0 / max(n_levels, 2)) * peak)
    ln_lam = np.log(map_.lam)
    along_y = np.log(y) if map_.y_log else y
    weight = (
        np.where(top, values, 0.0)
        * _trapezoid_widths(along_y)[:, None]
        * _trapezoid_widths(ln_lam)[None, :]
    )
    mass = float(weight.sum())
    if not mass > 0.0:
        return math.nan, math.nan
    lam_c = math.exp(float(weight.sum(axis=0) @ ln_lam) / mass)
    y_c = float(weight.sum(axis=1) @ along_y) / mass
    return lam_c, math.exp(y_c) if map_.y_log else y_c


def map_moment_sums(
    map_: Map, ylim: tuple[float, float] | None = None
) -> tuple[np.ndarray, float]:
    r"""The raw moment sums of one map as drawn, and its negative share.

    The plotted field read as a density on the plotted plane -- value
    times the trapezoidal widths in `$\ln\lambda$` and `$\ln y$`
    (`$y$` on a linear ordinate), over the rows the box shows: the
    measure of :func:`peak_centroid`, applied to the whole map rather
    than to its top band (module docstring, "Size and tilt").  Returns
    :func:`~dnsjax.analysis.twin.moments.log_moment_sums` of the
    positive part, and the mass of the negative part over that of the
    positive one -- zero for a spectrum, the measure of what a signed
    budget panel's moments leave out.
    """
    y, values = map_.drawn(ylim)
    ln_lam = np.log(map_.lam)
    along_y = np.log(y) if map_.y_log else y
    widths = (
        _trapezoid_widths(along_y)[:, None]
        * _trapezoid_widths(ln_lam)[None, :]
    )
    finite = np.where(np.isfinite(values), values, 0.0)
    sums = log_moment_sums(np.maximum(finite, 0.0), widths, ln_lam, along_y)
    negative = float((np.maximum(-finite, 0.0) * widths).sum())
    share = negative / float(sums[0]) if sums[0] > 0.0 else math.nan
    return sums, share


@dataclass(frozen=True)
class PeakTrack:
    """One panel's top-band centroid, frame by frame (:func:`track_peak`).

    ``lam`` and ``y`` are ``(n_frames,)``, in the map's plotted units,
    and ``nan`` at a frame with nothing positive.  ``moments`` are the
    whole map's, frame by frame (:func:`map_moment_sums`) -- natural
    logs of the plotted units, `$y$` itself on a linear ordinate --
    and ``negative`` each frame's negative share.
    """

    lam: np.ndarray
    y: np.ndarray
    moments: LogMoments | None = None
    negative: np.ndarray | None = None


def track_peak(
    series: YSeries,
    name: str,
    component: int | None,
    *,
    options: MapOptions,
    n_levels: int,
    ylim: tuple[float, float] | None = None,
) -> PeakTrack:
    """The centroid of one panel's top band, over every frame.

    One :func:`make_map` per frame, all read before any figure is drawn
    so that each frame can carry the history up to itself.  It sees
    neither the colour scale nor ``--clim``: the band is the frame's
    own (:func:`peak_centroid`).  The same maps give the whole map's
    moments (:func:`map_moment_sums`), so the size and tilt cost no
    second pass.
    """
    points, sums, negative = [], [], []
    for frame in range(series.t_rel.size):
        map_ = make_map(
            series,
            name,
            frame,
            options=options,
            component=component,
            ylim=ylim,
        )
        points.append(peak_centroid(map_, n_levels, ylim))
        frame_sums, share = map_moment_sums(map_, ylim)
        sums.append(frame_sums)
        negative.append(share)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return PeakTrack(
        lam=points[:, 0],
        y=points[:, 1],
        moments=log_moments(np.asarray(sums)),
        negative=np.asarray(negative),
    )


def draw_track(
    ax, track: PeakTrack, frame: int, *, colour: str, y_log: bool = True
) -> None:
    """A peak's history up to *frame* as a thin line, *frame* as a point.

    Both above the map and haloed in white (:data:`_TRACK_HALO`,
    :data:`_TRACK_EDGE`), which is what keeps a red track legible on a
    grey map's black top band and a black one on the darkest red.  A
    frame with nothing positive has no point and breaks the line.  The
    axes limits are the map's, already fixed, so neither moves them.

    Where the track carries the map's moments, the frame's one-sigma
    ellipse is drawn as well (:func:`draw_ellipse`), dashed, in the same
    colour: where the whole field sits, how large and how tilted,
    beside where its peak is.
    """
    halo = [
        patheffects.withStroke(
            linewidth=_TRACK_LINE + _TRACK_HALO, foreground="white"
        )
    ]
    ax.plot(
        track.lam[: frame + 1],
        track.y[: frame + 1],
        color=colour,
        linewidth=_TRACK_LINE,
        path_effects=halo,
        zorder=3,
    )
    if np.isfinite(track.lam[frame]) and np.isfinite(track.y[frame]):
        ax.plot(
            track.lam[frame],
            track.y[frame],
            linestyle="none",
            marker="o",
            markersize=_TRACK_MARKER,
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=_TRACK_EDGE,
            zorder=4,
        )
    if track.moments is not None:
        draw_ellipse(ax, track.moments, frame, colour=colour, y_log=y_log)


def ellipse_points(
    moments: LogMoments, frame: int, n_points: int = 121
) -> tuple[np.ndarray, np.ndarray] | None:
    r"""The one-sigma ellipse of one frame, in the moments' coordinates.

    ``(x, y)``: `$\mu + L(\cos\theta, \sin\theta)$` for `$L$` the
    Cholesky factor of the covariance, so the curve is the ellipse
    whatever its tilt -- built in the log coordinates and only then
    mapped back, so that it is an ellipse on the logarithmic axes.
    ``None`` where the frame has no positive-definite covariance.
    """
    mean = np.array([moments.mean_lam[frame], moments.mean_y[frame]])
    cov = np.array(
        [
            [moments.var_lam[frame], moments.cov[frame]],
            [moments.cov[frame], moments.var_y[frame]],
        ]
    )
    if not (np.isfinite(mean).all() and np.isfinite(cov).all()):
        return None
    try:
        factor = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return None
    theta = np.linspace(0.0, 2.0 * np.pi, n_points)
    pts = mean[:, None] + factor @ np.vstack([np.cos(theta), np.sin(theta)])
    return pts[0], pts[1]


def draw_ellipse(
    ax, moments: LogMoments, frame: int, *, colour: str, y_log: bool = True
) -> None:
    """One frame's one-sigma ellipse on a map, dashed and haloed."""
    pts = ellipse_points(moments, frame)
    if pts is None:
        return
    halo = [
        patheffects.withStroke(
            linewidth=_TRACK_LINE + _TRACK_HALO, foreground="white"
        )
    ]
    ax.plot(
        np.exp(pts[0]),
        np.exp(pts[1]) if y_log else pts[1],
        color=colour,
        linewidth=_TRACK_LINE,
        linestyle=(0, (3, 1.5)),
        path_effects=halo,
        zorder=3,
    )


def track_figure(
    series: YSeries,
    keys: list[tuple[str, int | None]],
    tracks: dict[tuple[str, int | None], PeakTrack],
    options: MapOptions,
    style: PlotStyle,
):
    r"""The tracks of one series against time: `$y_c$` above `$\lambda_c$`.

    A line per tracked panel, in panel order, coloured and dashed by
    :data:`_TRACK_SERIES` / :data:`_TRACK_DASHES` and named by the
    panel's symbol (:func:`panel_symbol`).  Both ordinates are on the
    maps' scales and in their units, with the outer-unit twins the maps
    carry, and time is the spacetime maps' axis.
    """
    units = options.units
    t = units.plotted_time(series.t_rel)
    axis = MARGINALS[keys[0][0].rpartition("_")[2]][0]
    fig, (ax_y, ax_lam) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(style.width, _TRACK_HEIGHT),
        layout="constrained",
    )
    for index, key in enumerate(keys):
        sign, symbol = panel_symbol(series, *key)
        line = {
            "color": _TRACK_SERIES[index],
            "linestyle": _TRACK_DASHES[index],
            "label": f"${sign}{symbol}$",
        }
        ax_y.plot(t, tracks[key].y, **line)
        ax_lam.plot(t, tracks[key].lam, **line)
    ax_y.set_yscale("log" if options.y_log else "linear")
    ax_lam.set_yscale("log")
    ax_y.set_ylabel(units.y_label)
    ax_lam.set_ylabel(units.lambda_label(axis))
    ax_lam.set_xlabel(units.t_label)
    ax_lam.set_xlim(float(t[0]), float(t[-1]))
    for ax in (ax_y, ax_lam):
        ax.grid(True, color="0.88", linewidth=0.5)
    if units.wall:
        # The maps' outer-unit twins, and the spacetime maps' for time.
        lengths = (lambda v: v / units.re_tau, lambda v: v * units.re_tau)
        factor = units.re_tau**2 / units.re
        times = (lambda v: v / factor, lambda v: v * factor)
        ax_y.secondary_yaxis("right", functions=lengths).set_ylabel(r"$y/h$")
        ax_lam.secondary_yaxis("right", functions=lengths).set_ylabel(
            units.lambda_label(axis, outer=True)
        )
        ax_y.secondary_xaxis("top", functions=times).set_xlabel(
            r"$t\,U_\mathrm{cl}/h$"
        )
    handles, labels = ax_y.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(keys),
        frameon=False,
    )
    fig.suptitle(_spacetime_suptitle(series, units))
    return fig


def write_track_npz(
    path: Path,
    series: YSeries,
    keys: list[tuple[str, int | None]],
    tracks: dict[tuple[str, int | None], PeakTrack],
    options: MapOptions,
    n_levels: int,
) -> Path:
    """Dump the tracks of one series, and what they were taken from.

    A row per tracked panel, in the plotted units the maps use (the
    factors that undo them beside), with the band's threshold and the
    measure: enough to redraw the figure, or to set a track against
    another estimate of the same peak, without it.
    """
    units = options.units
    payload = {
        "lam": np.stack([tracks[key].lam for key in keys]),
        "y": np.stack([tracks[key].y for key in keys]),
        "fields": np.asarray([name for name, _ in keys]),
        "panels": np.asarray(
            [
                panel_label(series, name.rpartition("_")[0], component)
                for name, component in keys
            ]
        ),
        "t": series.t_rel,
        "t_plotted": units.plotted_time(series.t_rel),
        "index": series.index,
        "threshold": 1.0 - 1.0 / max(n_levels, 2),
        "measure": (
            "value x trapezoidal widths in ln(lambda) and "
            + ("ln(y)" if options.y_log else "y")
            + "; centroid of ln(lambda) and "
            + ("ln(y)" if options.y_log else "y")
        ),
        "length_factor": units.length(1.0),
        "time_factor": units.plotted_time(1.0),
        "wall_units": units.wall,
        "re": units.re,
        "re_tau": units.re_tau,
        "half": options.half,
        # What the tracked maps took, which a shape map fixes itself.
        "premultiply": map_premultiplier(
            keys[0][0].rpartition("_")[0], options
        ),
        "volume_fac_applied": options.volume_fac,
        "smooth": options.smooth,
        "stem": series.stem,
        "n_members": series.n_members,
        "members": np.asarray([str(m.path) for m in series.members]),
    }
    np.savez_compressed(path, **payload)
    return path


def _decades(values: np.ndarray, log: bool) -> np.ndarray:
    """A natural-log spread in decades, or as it is on a linear axis."""
    return values / math.log(10.0) if log else values


def moments_figure(
    series: YSeries,
    keys: list[tuple[str, int | None]],
    tracks: dict[tuple[str, int | None], PeakTrack],
    options: MapOptions,
    style: PlotStyle,
):
    r"""How large and how tilted each tracked panel is, against time.

    Five rows sharing the time axis, a line per tracked panel in the
    track figure's order, colours and dashes (:func:`track_figure`):
    the spreads `$\sigma_{\ln\lambda}$` and `$\sigma_{\ln y}$` in
    decades, the area measure `$\sqrt{\det C}$` in decades squared,
    the correlation `$\rho$` and the ridge slope `$b$`, whose
    `$b = 1$` (`$\lambda \propto y$`) is drawn for reference (module
    docstring, "Size and tilt").  On a linear ordinate the `$y$`
    entries are in the plotted units instead of decades.
    """
    units = options.units
    t = units.plotted_time(series.t_rel)
    log_y = options.y_log
    axis = MARGINALS[keys[0][0].rpartition("_")[2]][0]
    rows = (
        ("sd_lam", rf"$\sigma_{{\ln\lambda_{axis}}}$ (dec)"),
        (
            "sd_y",
            r"$\sigma_{\ln y}$ (dec)" if log_y else r"$\sigma_y$",
        ),
        (
            "sqrt_det",
            r"$\sqrt{\det C}$ (dec$^2$)" if log_y else r"$\sqrt{\det C}$",
        ),
        ("rho", r"$\rho$"),
        ("slope", r"$b = C_{\lambda y}/C_{yy}$"),
    )
    fig, axes = plt.subplots(
        len(rows),
        1,
        sharex=True,
        figsize=(style.width, _ROW_HEIGHT * len(rows) + 0.6),
        layout="constrained",
    )
    for index, key in enumerate(keys):
        m = tracks[key].moments
        sign, symbol = panel_symbol(series, *key)
        line = {
            "color": _TRACK_SERIES[index],
            "linestyle": _TRACK_DASHES[index],
            "label": f"${sign}{symbol}$",
        }
        values = {
            "sd_lam": _decades(m.sd_lam, True),
            "sd_y": _decades(m.sd_y, log_y),
            "sqrt_det": m.sqrt_det / (math.log(10.0) ** (2 if log_y else 1)),
            "rho": m.rho,
            "slope": m.slope,
        }
        for ax, (name, _) in zip(axes, rows, strict=True):
            ax.plot(t, values[name], **line)
    for ax, (name, label) in zip(axes, rows, strict=True):
        ax.set_ylabel(label)
        ax.grid(True, color="0.88", linewidth=0.5)
        if name == "rho":
            ax.axhline(0.0, color="0.5", linewidth=0.6)
        if name == "slope" and log_y:
            ax.axhline(1.0, color="0.5", linewidth=0.6, linestyle=":")
    axes[-1].set_xlabel(units.t_label)
    axes[-1].set_xlim(float(t[0]), float(t[-1]))
    if units.wall:
        factor = units.re_tau**2 / units.re
        axes[0].secondary_xaxis(
            "top", functions=(lambda v: v / factor, lambda v: v * factor)
        ).set_xlabel(r"$t\,U_\mathrm{cl}/h$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(keys),
        frameon=False,
    )
    fig.suptitle(_spacetime_suptitle(series, units))
    return fig


def write_moments_npz(
    path: Path,
    series: YSeries,
    keys: list[tuple[str, int | None]],
    tracks: dict[tuple[str, int | None], PeakTrack],
    options: MapOptions,
) -> Path:
    """Dump the moments of each tracked panel, and how they were taken.

    A row per tracked panel: the centroid in the plotted units, the
    spreads and `$\\sqrt{\\det C}$` in natural-log units (decades are a
    division by `$\\ln 10$` away), the correlation, the ridge slope,
    the principal-axis angle, the mass and the negative share.
    """
    units = options.units
    moments = [tracks[key].moments for key in keys]

    def stack(name: str) -> np.ndarray:
        return np.stack([np.asarray(getattr(m, name)) for m in moments])

    centre_y = stack("mean_y")
    payload = {
        "lam_c": np.exp(stack("mean_lam")),
        "y_c": np.exp(centre_y) if options.y_log else centre_y,
        "sd_lam": stack("sd_lam"),
        "sd_y": stack("sd_y"),
        "sqrt_det": stack("sqrt_det"),
        "rho": stack("rho"),
        "slope": stack("slope"),
        "angle": stack("angle"),
        "var_lam": stack("var_lam"),
        "var_y": stack("var_y"),
        "cov": stack("cov"),
        "mass": stack("mass"),
        "negative_share": np.stack([tracks[key].negative for key in keys]),
        "fields": np.asarray([name for name, _ in keys]),
        "panels": np.asarray(
            [
                panel_label(series, name.rpartition("_")[0], component)
                for name, component in keys
            ]
        ),
        "t": series.t_rel,
        "t_plotted": units.plotted_time(series.t_rel),
        "index": series.index,
        "measure": (
            "plotted value x trapezoidal widths in ln(lambda) and "
            + ("ln(y)" if options.y_log else "y")
            + " over the rows the box shows; positive part"
        ),
        "y_log": options.y_log,
        "premultiply": map_premultiplier(
            keys[0][0].rpartition("_")[0], options
        ),
        "smooth": options.smooth,
        "stem": series.stem,
        "n_members": series.n_members,
        "members": np.asarray([str(m.path) for m in series.members]),
        **_units_payload(options),
    }
    np.savez_compressed(path, **payload)
    return path


def render_tracks(
    series: YSeries,
    tag: str,
    keys: list[tuple[str, int | None]],
    tracks: dict[tuple[str, int | None], PeakTrack],
    out_dir: Path,
    *,
    options: MapOptions,
    style: PlotStyle,
    fmt: str = "png",
    quiet: bool = False,
) -> list[Path]:
    """The track figure of one series and its ``.npz``, in ``<tag>_track``.

    A directory of their own beside the series' frames rather than
    among them, so that a glob over the frames matches frames alone.  A
    selection of fewer than two frames has no time axis to draw and
    gets the ``.npz`` alone, as a spacetime map does.
    """
    target = out_dir / f"{tag}_track"
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if series.t_rel.size >= 2:
        fig = track_figure(series, keys, tracks, options, style)
        path = target / f"{tag}_track.{fmt}"
        fig.savefig(path, dpi=style.dpi)
        plt.close(fig)
        written.append(path)
    elif not quiet:
        print(
            f"  {tag}_track: no figure, a track needs two sample times "
            f"and this selection has {series.t_rel.size}",
            flush=True,
        )
    written.append(
        write_track_npz(
            target / f"{tag}_track.npz",
            series,
            keys,
            tracks,
            options,
            style.n_levels,
        )
    )
    if all(tracks[key].moments is not None for key in keys):
        if series.t_rel.size >= 2:
            fig = moments_figure(series, keys, tracks, options, style)
            path = target / f"{tag}_moments.{fmt}"
            fig.savefig(path, dpi=style.dpi)
            plt.close(fig)
            written.append(path)
        written.append(
            write_moments_npz(
                target / f"{tag}_moments.npz", series, keys, tracks, options
            )
        )
    if not quiet:
        for path in written:
            print(f"  {path.name}", flush=True)
    return written


# ── Spacetime maps ───────────────────────────────────────────────────


@dataclass(frozen=True)
class SpacetimeMap:
    r"""One panel's `$(y, t)$` field, and what produced it.

    ``values`` is ``(n_t, n_y)`` -- time down the first axis, as
    ``contourf`` wants it against ``t`` on the ordinate -- in the
    plotted units, folded, and **never** premultiplied.  Its
    ``provenance`` is what the ``.npz`` beside the figures carries
    (:func:`write_spacetime_npz`).
    """

    y: np.ndarray  # (n_y,) wall distance, plotted units
    t: np.ndarray  # (n_t,) time since the perturbation, plotted units
    values: np.ndarray  # (n_t, n_y)
    title: str  # LaTeX panel title
    name: str  # the base field it came from
    label: str  # which panel of the figure it is
    non_negative: bool
    y_log: bool = False
    provenance: dict = field(default_factory=dict, repr=False)

    def drawn(
        self, ylim: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""The columns the wall-distance axis shows, within *ylim*.

        :meth:`Map.drawn` transposed: here `$y$` is the **abscissa**,
        so it is columns rather than rows that a logarithmic axis
        cannot place at the wall and that the axis box's limits cut
        off.  Everything downstream of the colour scale goes through
        this, which is what keeps a wall peak the ordinate floors away
        from setting a scale no visible contour reaches.
        """
        keep = _drawn_rows(self.y, self.y_log, ylim)
        return self.y[keep], self.values[:, keep]


def _frame_mean(series: YSeries, name: str, frame: int) -> np.ndarray:
    """The ensemble mean of one stored field at one frame.

    :meth:`YSeries.field` reads and caches every selected frame, which
    is what the maps want and what a one-off cross-check does not.
    """
    total = None
    for index in range(series.n_members):

        def read(stored: str, index=index) -> np.ndarray:
            records, rows = series.source(index, stored)
            return np.asarray(records[stored][rows[frame]], dtype=np.float64)

        block = balance_field(read, series.meta, name)
        total = block if total is None else total + block
    return total / series.n_members


def k_summed(series: YSeries, base: str, marginal: str = "") -> np.ndarray:
    r"""A stored field summed over `$k$`, ``(n_t, [3,] n_y)``.

    Read record by record and summed on arrival
    (:meth:`YSeries.reduced`), so the whole field is never held.

    Which modes that sum covers is the whole of the convention behind
    every spacetime map (module docstring, "Spacetime maps"): a
    **reference** spectrum loses its `$(0, 0)$` mode, because that is
    the wall-parallel mean, is common to both states of a twin pair
    and never decorrelates; everything else -- the difference spectra
    and the budget terms alike -- keeps every mode it has, so the sum
    is the whole of that quantity at that wall distance.

    *marginal* empty means the complete sum, which is the same number
    off either marginal and is read off `$k_z$`'s; the two are
    compared once per series (:func:`check_k_sum`).  ``"x0"`` asks for
    the `$k_x = 0$` slice instead, which is a different quantity and
    keeps its name.
    """
    suffix = marginal or "x"
    if base == "r":
        name, mean_name = f"r_{suffix}", mean_mode_name(series.meta, "r")
        return series.reduced(
            lambda read: fluctuation_profile(
                read(name), mean_mode_profile(read(mean_name), mean_name)
            )
        )
    names = series.additive(suffix) if base == "sum" else [f"{base}_{suffix}"]
    return series.reduced(
        lambda read: sum(
            balance_field(read, series.meta, n).sum(axis=-1) for n in names
        )
    )


def check_k_sum(series: YSeries, base: str) -> None:
    r"""Both marginals must give the same complete `$k$`-sum.

    ``<base>_x`` sums over `$k_x$` and ``<base>_z`` over `$k_z$`, so
    each is already a complete one-sided sum over the mode plane and
    summing either over its own axis is the same total at every `$y$`
    -- which is exactly why a `$k$`-summed map is marginal-free and is
    drawn once rather than twice.  The claim is asserted rather than
    argued, on the first frame, off single records.

    The reference half has this checked per member and per instant
    already (:meth:`YSeries._check_marginals`); this is its
    difference-field twin, and the one that covers a budget term.
    """
    by_x, by_z = (
        _frame_mean(series, f"{base}_{suffix}", 0).sum(axis=-1)
        for suffix in ("x", "z")
    )
    tol = 1e-9 if series.meta["value_dtype"] == "<f8" else 1e-4
    if not np.allclose(
        by_x, by_z, rtol=tol, atol=tol * float(np.max(np.abs(by_x)))
    ):
        raise ValueError(
            f"{series.members[0].path}: the k_z and k_x marginals of "
            f"{base} disagree on the k-summed profile at "
            f"t = {series.t_rel[0]:g}; one of them is not a complete "
            "sum over the mode plane."
        )


def spacetime_normalises(series: YSeries, base: str, marginal: str) -> bool:
    r"""Whether a `$k$`-summed panel is drawn over `$E^{\mathrm{ref}}$`.

    :func:`normalises` one axis down: the complete sums of a spectra
    stream that carries its reference half, and nothing else.  False
    for the `$k_x = 0$` slice (not a sum over the whole mode plane),
    for a budget term (not an energy) and for `$\mathcal{R}^k$`
    (which has its own, resolved, divisor).  Cheap, so a caller can
    ask before paying for :meth:`YSeries.reference_scale`.
    """
    return (
        series.stem == "twin_yspectra"
        and not marginal
        and base in ("e", "r")
        and "r" in series.prefixes
    )


def spacetime_norm(
    series: YSeries, base: str, marginal: str, component: int | None
) -> float | None:
    r"""The `$E^{\mathrm{ref}}$` one `$k$`-summed panel divides by."""
    if not spacetime_normalises(series, base, marginal):
        return None
    scale = series.reference_scale()
    return float(scale.sum() if component is None else scale[component])


def spacetime_title(
    series: YSeries,
    base: str,
    marginal: str,
    component: int | None,
    options: MapOptions,
    *,
    premultiply: bool = False,
) -> str:
    r"""The LaTeX panel title for one `$k$`-summed field.

    No premultiplier appears unless *premultiply* (a history map's,
    module docstring "History maps"), and then it is the wall
    distance's, in the plotted units.  A prime marks the one quantity
    here whose `$(0, 0)$` mode has been removed, the reference energy;
    the difference energy carries all of its modes and no prime
    (:func:`k_summed`).  A shape history is over each row's own peak,
    which varies along time, so its title reports none.
    """
    kind = field_kind(series)
    factor = rf"y{options.units.suffix}\," if premultiply else ""
    if base == DECORR_K:
        sub = "" if component is None else f"_{{{COMPONENTS[component]}}}"
        return rf"$\mathcal{{R}}^{{k}}{sub}$"
    if kind == "rate":
        sign, body = TERM_LABELS.get(base, ("", base.replace("_", r"\_")))
        return f"${sign}{factor}{body}{options.units.norm_suffix(kind)}$"
    superscript = f"^{{{MARGINALS[marginal][1]}}}" if marginal else ""
    symbol = "{E'}" if base == "r" else "{E}"
    delta = r"\Delta " if base in ("e", SHAPE) else ""
    if component is None:
        body = rf"\sum_\alpha {symbol}{superscript}_{{{delta}\alpha}}"
    else:
        sub = f"{delta}{COMPONENTS[component]}"
        body = rf"{symbol}{superscript}_{{{sub}}}"
    if base == SHAPE:
        return rf"${factor}{body}/\max_y$"
    scale = spacetime_norm(series, base, marginal, component)
    if scale is None:
        return f"${factor}{body}{options.units.norm_suffix(kind)}$"
    ref = reference_symbol(component)
    return (
        f"${factor}{body}/{ref}$\n"
        f"${ref}{options.units.norm_suffix(kind)} = "
        f"{latex_float(options.units.energy(scale))}$"
    )


def _row_peaks(
    values: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each row of *values* over its own peak among the columns *keep*.

    Returns the scaled rows and the peaks.  A row with nothing positive
    keeps its zeros rather than going ``nan``, as a shape map's frame
    does.
    """
    shown = np.where(keep[None, :] & np.isfinite(values), values, -np.inf)
    peaks = shown.max(axis=1, initial=-np.inf)
    peaks = np.where(peaks > 0.0, peaks, 0.0)
    scaled = np.divide(
        values,
        peaks[:, None],
        out=np.array(values, dtype=np.float64),
        where=peaks[:, None] > 0.0,
    )
    return scaled, peaks


def make_spacetime(
    series: YSeries,
    base: str,
    marginal: str = "",
    *,
    options: MapOptions,
    component: int | None = None,
    non_negative: bool | None = None,
    premultiply: bool = False,
    ylim: tuple[float, float] | None = None,
) -> SpacetimeMap:
    r"""Build one `$(y, t)$` map from a `$k$`-summed field.

    *base* is a stored prefix (``e`` / ``r``), a budget term (or the
    virtual ``sum``), :data:`DECORR_K`, which is ``e`` over twice
    the reference profile, or :data:`SHAPE`, which is ``e`` again;
    *marginal* is empty for the complete sum and ``"x0"`` for the
    `$k_x = 0$` slice.

    No `$k$` premultiplier reaches these whatever ``--premultiply``
    says: `$\sum_m m\,\text{entry}$` is not a sum of energies, and
    there is no logarithmic wavelength axis left for one to serve.
    ``volume_fac`` and the unit conversion reach an absolute panel
    exactly as they reach a `$(\lambda, y)$` one, and cancel out of a
    ratio.  The divisor is symmetrised before the fold
    (:func:`_symmetrise_y`).

    *premultiply* multiplies by the wall distance in the plotted units
    **after** the sum and the fold -- a history map (module docstring,
    "History maps") -- so that equal areas over `$\ln y$` are equal
    energy.  :data:`SHAPE` implies it, and then divides each time row
    by its own peak over the columns the box shows (*ylim*,
    :func:`y_limits`' default when ``None``), which the provenance
    records as ``row_peaks``.
    """
    ratio = base == DECORR_K
    shape = base == SHAPE
    premultiply = premultiply or shape
    values = k_summed(series, "e" if ratio or shape else base, marginal)
    divisor = None
    if series.stem == "twin_yspectra":
        if ratio:
            divisor = _symmetrise_y(
                series.reference_profile(), options.half, axis=-1
            )
        if component is None:
            values = values.sum(axis=1)
            divisor = None if divisor is None else divisor.sum(axis=0)
        else:
            values = values[:, component]
            divisor = None if divisor is None else divisor[component]
    elif component is not None:
        raise ValueError(f"{series.stem}: {base} has no component axis")

    scale = None
    if divisor is not None:
        values = _ratio(values, 2.0 * divisor)
    else:
        if options.volume_fac:
            values = values * series.volume_fac
        scale = spacetime_norm(series, base, marginal, component)
        if scale is None:
            values = options.units.convert(values, field_kind(series))
        else:
            values = values / scale
    # _select_half folds the second-to-last axis, the wavenumber one
    # on a map; a k-summed profile borrows a length-1 axis for it.
    folded, wall_distance = _select_half(
        values[..., None], series.y, options.half
    )
    if divisor is not None:
        # Recorded on the *plotted* grid, so that multiplying it back
        # through recovers the numerator.  Symmetric already under
        # ``mean``, so this fold only selects the rows.
        divisor = _select_half(divisor[..., None], series.y, options.half)[0]
        divisor = 2.0 * divisor[..., 0]
    y = options.units.length(wall_distance)
    folded = folded[..., 0]
    if premultiply:
        folded = folded * y[None, :]
    row_peaks = None
    if shape:
        keep = _drawn_rows(y, options.y_log, y_limits(series, options, ylim))
        folded, row_peaks = _row_peaks(folded, keep)
    name = f"{base}_{marginal}" if marginal else base
    return SpacetimeMap(
        y=y,
        t=options.units.plotted_time(series.t_rel),
        values=folded,
        title=spacetime_title(
            series,
            base,
            marginal,
            component,
            options,
            premultiply=premultiply,
        ),
        name=name,
        label=panel_label(series, base, component),
        non_negative=(
            declared_non_negative(name)
            if non_negative is None
            else non_negative
        ),
        y_log=options.y_log,
        provenance={
            "divisor": divisor,
            "e_ref": scale,
            "kind": "ratio" if ratio else field_kind(series),
            "premultiplier": "y" if premultiply else "none",
            "row_peaks": row_peaks,
        },
    )


def half_weights(series: YSeries) -> np.ndarray:
    r"""Quadrature weights of the rows a fold keeps, doubled.

    The weights of :func:`_half_grid`'s rows, each doubled to stand for
    its mirror row as well -- except the mid-plane of an odd grid,
    which is its own mirror -- so that contracting a folded profile
    with them returns the full-channel contraction of a symmetric one.
    The stored entries are already divided by ``volume_fac``, so that
    contraction is a wall-normal **average** (module docstring,
    "Premultiplication").
    """
    w = series.y_weights
    n = (w.size + 1) // 2
    out = 2.0 * w[:n]
    if w.size % 2:
        out[-1] = w[n - 1]
    return out


def y_averaged(
    series: YSeries, base: str, marginal: str, half: str
) -> np.ndarray:
    r"""A stored field averaged over the wall distance, ``(n_t, [3,] n_k)``.

    Read record by record and contracted on arrival
    (:meth:`YSeries.reduced`), so the whole field is never held.  Each
    record is folded first (:func:`_select_half`, so ``--half`` reaches
    it as it reaches a map) and then contracted with
    :func:`half_weights`: under ``mean`` that is the full-channel
    average, which is the per-mode contribution to the volume-averaged
    total.  *base* is ``e`` or a budget term (balance terms regrouped
    on the way, :func:`balance_field`); every mode is kept, `$m = 0$`
    included, so the caller decides what an axis can show.
    """
    weights = half_weights(series)
    name = f"{base}_{marginal}"

    def reduce(read) -> np.ndarray:
        folded, _ = _select_half(
            balance_field(read, series.meta, name), series.y, half
        )
        return np.einsum("j,...jk->...k", weights, folded)

    return series.reduced(reduce)


@dataclass(frozen=True)
class ScaleMap:
    r"""One panel's `$(\lambda, t)$` field: a wall-normal average.

    ``values`` is ``(n_t, n_lam)``, time down the first axis and the
    wavelength ascending, in the plotted units and premultiplied by
    `$k$` (module docstring, "History maps").  ``name`` carries the
    marginal suffix, which is what :attr:`lam_axis` reads.
    """

    lam: np.ndarray  # (n_lam,) wavelength, ascending, plotted units
    t: np.ndarray  # (n_t,) time since the perturbation, plotted units
    values: np.ndarray  # (n_t, n_lam)
    title: str
    name: str
    label: str
    non_negative: bool
    provenance: dict = field(default_factory=dict, repr=False)

    @property
    def lam_axis(self) -> str:
        """Which wavelength the abscissa is: ``x`` or ``z``."""
        return MARGINALS[self.name.rpartition("_")[2]][0]

    def drawn(
        self, xlim: tuple[float, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The columns inside *xlim*, with one neighbour either side.

        :meth:`SpacetimeMap.drawn` for a wavelength abscissa: there is
        no wall column to drop, only the box's limits, and the
        neighbours are kept because the fill interpolates.
        """
        keep = np.ones(self.lam.size, dtype=bool)
        if xlim is not None:
            keep = (self.lam >= xlim[0]) & (self.lam <= xlim[1])
            inside = np.flatnonzero(keep)
            if inside.size:
                keep[max(int(inside[0]) - 1, 0)] = True
                keep[min(int(inside[-1]) + 1, keep.size - 1)] = True
        return self.lam[keep], self.values[:, keep]


def scaletime_title(
    series: YSeries,
    base: str,
    marginal: str,
    component: int | None,
    options: MapOptions,
) -> str:
    r"""The LaTeX panel title for one wall-normal-averaged field.

    The `$k$` premultiplier of the marginal's own wavenumber, in the
    plotted units, ahead of a `$\langle\cdot\rangle_y$` average; a
    spectrum is over `$E^{\mathrm{ref}}$` as on a map (second line),
    a shape history over each row's own peak.
    """
    kind = field_kind(series)
    plus = options.units.suffix
    factor = rf"k_{{{MARGINALS[marginal][0]}}}{plus}\,"
    if kind == "rate":
        sign, body = TERM_LABELS.get(base, ("", base.replace("_", r"\_")))
        return (
            rf"${sign}{factor}\langle {body}\rangle_y"
            rf"{options.units.norm_suffix(kind)}$"
        )
    _, body = panel_symbol(series, f"e_{marginal}", component)
    if base == SHAPE:
        return rf"${factor}\langle {body}\rangle_y/\max_\lambda$"
    scale = reference_norm(series, f"e_{marginal}", component)
    if scale is None:
        return (
            rf"${factor}\langle {body}\rangle_y"
            rf"{options.units.norm_suffix(kind)}$"
        )
    ref = reference_symbol(component)
    return (
        rf"${factor}\langle {body}\rangle_y/{ref}$" + "\n"
        f"${ref}{options.units.norm_suffix(kind)} = "
        f"{latex_float(options.units.energy(scale))}$"
    )


def make_scaletime(
    series: YSeries,
    base: str,
    marginal: str,
    *,
    options: MapOptions,
    component: int | None = None,
    non_negative: bool | None = None,
) -> ScaleMap:
    r"""Build one `$(\lambda, t)$` map: a wall-normal average, `$\times k$`.

    The history counterpart of :func:`make_spacetime` with the roles of
    the two coordinates swapped (module docstring, "History maps"):
    :func:`y_averaged` first, then the component reduction, the
    `$m = 0$` column dropped (no position on a wavelength axis), and
    only **then** the premultiplier `$m$`, so that equal areas over
    `$\ln\lambda$` are equal energy.  A difference spectrum is divided
    by its `$E^{\mathrm{ref}}$` (:func:`reference_norm`, the same
    number as the maps'), a budget term takes the unit conversion, and
    :data:`SHAPE` divides each time row by its own peak instead, which
    the provenance records.  ``volume_fac`` does not apply: an average
    over `$y$` has no local density to restore.
    """
    shape = base == SHAPE
    stored = "e" if shape else base
    values = y_averaged(series, stored, marginal, options.half)
    if series.stem == "twin_yspectra":
        values = (
            values.sum(axis=1) if component is None else (values[:, component])
        )
    elif component is not None:
        raise ValueError(f"{series.stem}: {base} has no component axis")
    values = values[:, 1:] * series.harmonics(marginal)[None, 1:]
    scale = None
    if series.stem == "twin_yspectra" and not shape:
        scale = reference_norm(series, f"e_{marginal}", component)
    if scale is None:
        values = options.units.convert(values, field_kind(series))
    else:
        values = values / scale
    values = values[:, ::-1]  # ascending in wavelength, as the axis is
    row_peaks = None
    if shape:
        values, row_peaks = _row_peaks(
            values, np.ones(values.shape[1], dtype=bool)
        )
    name = f"{base}_{marginal}"
    return ScaleMap(
        lam=options.units.length(series.wavelengths(marginal)),
        t=options.units.plotted_time(series.t_rel),
        values=values,
        title=scaletime_title(series, base, marginal, component, options),
        name=name,
        label=panel_label(series, base, component),
        non_negative=(
            declared_non_negative(name)
            if non_negative is None
            else non_negative
        ),
        provenance={
            "e_ref": scale,
            "kind": field_kind(series),
            "premultiplier": "k",
            "row_peaks": row_peaks,
        },
    )


def spacetime_panels(
    series: YSeries, spec: SeriesSpec
) -> list[tuple[str, int | None]]:
    """Which ``(base, component)`` panels a spacetime figure carries.

    A spectra series shows its three components and their sum; a
    budget series :data:`SPACETIME_PANELS` and their sum, whose
    pressure panel carries the driving input that a map's leaves out
    (:data:`MAP_PANELS`).  The base is what :func:`make_spacetime`
    takes.
    """
    if spec.stem == "twin_ybudget":
        balance_terms(series.meta)
        return [(term, None) for term in (*SPACETIME_PANELS, "sum")]
    return [(spec.base, c) for c in (*range(len(COMPONENTS)), None)]


def spacetime_maps(
    series: YSeries, spec: SeriesSpec, options: MapOptions
) -> list[SpacetimeMap]:
    """Every panel of one spacetime series, built once.

    Both figures of the pair and the ``.npz`` are made from these, so
    the three cannot disagree about what was summed.
    """
    panels = spacetime_panels(series, spec)
    if not spec.marginal:  # a complete sum claims to be marginal-free
        check_k_sum(series, "e" if spec.base == DECORR_K else panels[0][0])
    return [
        make_spacetime(
            series, base, spec.marginal, options=options, component=c
        )
        for base, c in panels
    ]


def spacetime_scales(
    maps: list[SpacetimeMap],
    ylim: tuple[float, float] | None,
    *,
    declared: bool = True,
    decades: float = LOG_DECADES,
) -> tuple[list[PanelScale], list[float], list[str]]:
    """Per-panel range, logarithmic floor, and the sign-check report.

    :func:`scan_panels` for a series that is one figure: there are no
    frames to sweep, so each panel's range is read straight off the
    columns its axes box will show (:meth:`SpacetimeMap.drawn`) and
    both figures of the pair are drawn against it -- which is what
    makes the linear and logarithmic versions two readings of one
    scale rather than two scales.
    """
    scales: list[PanelScale] = []
    floors: list[float] = []
    notes: list[str] = []
    for map_ in maps:
        values = map_.drawn(ylim)[1]
        finite = values[np.isfinite(values)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 0.0
        non_negative = (
            declared_non_negative(map_.name) if declared else lo >= 0.0
        )
        label = f"{map_.name}[{map_.label}]"
        note = None
        if declared and non_negative:
            note = _sign_note(label, lo, hi)
        elif declared and declared_non_positive(map_.name):
            note = _sign_note(label, lo, hi, positive=False)
        if note is not None:
            notes.append(note)
        scales.append(PanelScale(lo, hi, non_negative))
        floors.append(log_floor(values, decades))
    return scales, floors, notes


def draw_spacetime(
    ax,
    map_: SpacetimeMap,
    *,
    units: Units,
    scale: str = "linear",
    n_levels: int = 10,
    cmap_positive: str = "Greys",
    cmap_signed: str = "RdBu_r",
    data_range: tuple[float, float] | None = None,
    floor: float | None = None,
    decades: float = LOG_DECADES,
    nice: bool = True,
    fill: str = "contour",
    lines: bool = True,
    cax=None,
    secondary: bool = True,
    title: bool = True,
    ylim: tuple[float, float] | None = None,
):
    r"""Draw one `$(y, t)$` map on *ax*; returns the fill artist.

    Wall distance on the abscissa, wall on the left and on whichever
    scale the `$(\lambda, y)$` maps use their ordinate; time up the
    ordinate, always linear.  No ``set_aspect``: a decade of `$y$`
    against a time interval is not a ratio that means anything.

    *scale* picks the colour scale.  ``"linear"`` is the banded one
    the rest of the module uses, unchanged; ``"log"`` spends
    :func:`log_levels` on the decades from *floor* up, colours them
    logarithmically (:func:`band_colors`), draws its contour lines and
    labels its bar on the decades alone, and extends the bar downward
    to say that something falls below the floor.  *floor* defaults to
    :func:`log_floor` of the columns inside *ylim* at *decades*, which
    is what :func:`render_spacetime` hands it anyway.

    The body is :func:`draw_time_map`'s, which a wavelength abscissa
    (:class:`ScaleMap`) shares.
    """
    return draw_time_map(
        ax,
        map_,
        units=units,
        scale=scale,
        n_levels=n_levels,
        cmap_positive=cmap_positive,
        cmap_signed=cmap_signed,
        data_range=data_range,
        floor=floor,
        decades=decades,
        nice=nice,
        fill=fill,
        lines=lines,
        cax=cax,
        secondary=secondary,
        title=title,
        lim=ylim,
    )


def _time_map_abscissa(
    map_, units: Units
) -> tuple[np.ndarray, bool, str, str]:
    """``(coordinate, log, label, outer label)`` of a map against time."""
    if isinstance(map_, ScaleMap):
        return (
            map_.lam,
            True,
            units.lambda_label(map_.lam_axis),
            units.lambda_label(map_.lam_axis, outer=True),
        )
    return map_.y, map_.y_log, units.y_label, r"$y/h$"


def draw_time_map(
    ax,
    map_: SpacetimeMap | ScaleMap,
    *,
    units: Units,
    scale: str = "linear",
    n_levels: int = 10,
    cmap_positive: str = "Greys",
    cmap_signed: str = "RdBu_r",
    data_range: tuple[float, float] | None = None,
    floor: float | None = None,
    decades: float = LOG_DECADES,
    nice: bool = True,
    fill: str = "contour",
    lines: bool = True,
    cax=None,
    secondary: bool = True,
    title: bool = True,
    lim: tuple[float, float] | None = None,
    quantiles: np.ndarray | None = None,
):
    r"""Draw a map against time -- `$(y, t)$` or `$(\lambda, t)$`.

    :func:`draw_spacetime`'s rules for either abscissa: the wall
    distance of a :class:`SpacetimeMap` on the maps' ordinate scale, or
    the logarithmic wavelength of a :class:`ScaleMap`; time up the
    ordinate, linear.  *lim* is the abscissa's limits, and the levels
    are read off the columns inside it while the fill gets them all,
    so a contour reaches the edge of the box.

    *quantiles* is ``(n_t, n_q)`` positions on the abscissa
    (:func:`quantile_curves`), drawn as lines over the fill: the median
    solid, the others dashed, haloed in white like a track
    (:func:`draw_track`).
    """
    x, x_log, x_label, x_outer = _time_map_abscissa(map_, units)
    ax.set_xscale("log" if x_log else "linear")
    _, values = map_.drawn()
    scaled = values if lim is None else map_.drawn(lim)[1]
    log = scale == "log"
    if log:
        peak = (data_range or (0.0, float("-inf")))[1]
        if data_range is None:
            finite = scaled[np.isfinite(scaled)]
            peak = float(finite.max()) if finite.size else 0.0
        if floor is None:
            floor = log_floor(scaled, decades)
        levels = log_levels(floor, peak)
    else:
        levels = contour_levels(
            scaled,
            n_levels,
            non_negative=map_.non_negative,
            data_range=data_range,
            nice=nice,
        )
    x = map_.drawn()[0]

    filled = None
    if levels.size > 1:  # a single level bounds no band
        shaded, norm = band_colors(
            levels,
            cmap_positive if (log or map_.non_negative) else cmap_signed,
            non_negative=map_.non_negative,
            log=log,
        )
        if fill == "pcolormesh":
            filled = ax.pcolormesh(
                cell_edges(x, log=x_log),
                cell_edges(map_.t, log=False),
                values,
                cmap=shaded,
                norm=norm,
            )
        else:
            filled = ax.contourf(
                x,
                map_.t,
                values,
                levels=levels,
                cmap=shaded,
                norm=norm,
                extend="min" if log else "neither",
            )
        if lines:
            ax.contour(
                x,
                map_.t,
                values,
                levels=_decade_ticks(levels) if log else levels,
                colors="k",
                linewidths=0.3,
                alpha=0.7,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "identically zero" if not log else "nothing above the floor",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    if quantiles is not None:
        draw_quantiles(
            ax, quantiles, map_.t, colour=_TRACK_COLOURS[map_.non_negative]
        )

    ax.set_xlim(*(lim or (x.min(), x.max())))
    ax.set_ylim(float(map_.t.min()), float(map_.t.max()))
    ax.set_xlabel(x_label)
    ax.set_ylabel(units.t_label)
    if secondary and units.wall:
        # The outer-unit twins of both axes, as draw_map's are: a
        # division by Re_tau for the length, by Re_tau^2/Re for time.
        lengths = (lambda v: v / units.re_tau, lambda v: v * units.re_tau)
        factor = units.re_tau**2 / units.re
        times = (lambda v: v / factor, lambda v: v * factor)
        ax.secondary_xaxis("top", functions=lengths).set_xlabel(x_outer)
        ax.secondary_yaxis("right", functions=times).set_ylabel(
            r"$t\,U_\mathrm{cl}/h$"
        )
    if title:
        ax.set_title(map_.title, pad=_TITLE_PAD)
    if cax is not None and filled is not None:
        bar = ax.figure.colorbar(
            filled,
            cax=cax,
            ticks=_decade_ticks(levels) if log else _bar_ticks(levels),
        )
        bar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{v:.3g}")
        )
        bar.ax.tick_params(labelsize="small")
    elif cax is not None:
        cax.set_axis_off()
    return filled


def quantile_curves(
    x: np.ndarray,
    values: np.ndarray,
    quantiles: tuple[float, ...] = HISTORY_QUANTILES,
    *,
    x_log: bool = True,
) -> np.ndarray:
    r"""Quantiles of each row's as-drawn density, ``(n_t, n_q)``.

    Each row of *values* -- a history map's premultiplied field on the
    columns *x* the box shows -- is a density along its axis; times the
    trapezoidal widths in `$\ln x$` (`$x$` on a linear axis) it is a
    mass per column, the same measure the peak tracks and the size and
    tilt descriptors use (module docstring, "Size and tilt").  The
    quantiles interpolate the cumulative mass, each column's mass
    centred on its own sample, in `$\ln x$`.  Negative values carry no
    mass, and a row with none is ``nan`` throughout.
    """
    along = np.log(x) if x_log else np.asarray(x, dtype=np.float64)
    mass = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    mass = mass * _trapezoid_widths(along)[None, :]
    total = mass.sum(axis=1)
    cumulative = np.cumsum(mass, axis=1) - 0.5 * mass
    out = np.full((values.shape[0], len(quantiles)), np.nan)
    for row in np.flatnonzero(total > 0.0):
        cdf = cumulative[row] / total[row]
        out[row] = np.interp(quantiles, cdf, along)
    return np.exp(out) if x_log else out


def draw_quantiles(
    ax, quantiles: np.ndarray, t: np.ndarray, *, colour: str
) -> None:
    """The quantile lines of a map against time, haloed in white.

    The middle column of *quantiles* (the median of
    :data:`HISTORY_QUANTILES`) solid, the rest dashed; the axes limits
    are the map's, already fixed or about to be, so neither moves them.
    """
    halo = [
        patheffects.withStroke(
            linewidth=_TRACK_LINE + _TRACK_HALO, foreground="white"
        )
    ]
    middle = quantiles.shape[1] // 2
    for column in range(quantiles.shape[1]):
        ax.plot(
            quantiles[:, column],
            t,
            color=colour,
            linewidth=_TRACK_LINE,
            linestyle="-" if column == middle else (0, (4, 2)),
            path_effects=halo,
            zorder=3,
        )


def _spacetime_suptitle(series: YSeries, units: Units) -> str:
    r"""The window the map covers, in both time units."""
    first, last = float(series.t_rel[0]), float(series.t_rel[-1])
    return (
        rf"${series.n_members}$ member(s), "
        rf"$t = {first:.6g}\ldots{last:.6g}\,h/U_\mathrm{{cl}},"
        rf"\;\;\; t^+ = {units.time(first):.6g}"
        rf"\ldots{units.time(last):.6g}$"
    )


def spacetime_figure(
    series: YSeries,
    maps: list[SpacetimeMap],
    scales: list[PanelScale],
    floors: list[float],
    options: MapOptions,
    style: PlotStyle,
    *,
    scale: str,
    ylim: tuple[float, float],
):
    """One figure of the pair, one `$k$`-summed panel per component."""
    return time_map_figure(
        series,
        maps,
        scales,
        floors,
        options,
        style,
        scale=scale,
        lim=ylim,
    )


def time_map_figure(
    series: YSeries,
    maps: list[SpacetimeMap] | list[ScaleMap],
    scales: list[PanelScale],
    floors: list[float],
    options: MapOptions,
    style: PlotStyle,
    *,
    scale: str,
    lim: tuple[float, float],
    quantiles: list[np.ndarray | None] | None = None,
):
    """One figure of maps against time, one panel per map.

    The panels share an abscissa -- the wall distance, or one
    marginal's wavelength -- whose limits are *lim*; the geometry is
    :func:`panel_geometry`'s for a spacetime map, the decade rule
    applying to a logarithmic abscissa and ``--box-aspect`` to time.
    *quantiles* holds each panel's :func:`quantile_curves`, or
    ``None`` where a panel draws none.
    """
    tlim = (float(maps[0].t.min()), float(maps[0].t.max()))
    _, x_log, _, _ = _time_map_abscissa(maps[0], options.units)
    geometry = panel_geometry(
        len(maps),
        lim,
        tlim,
        style,
        y_log=False,
        x_log=x_log,
        title_lines=1 + max(m.title.count("\n") for m in maps),
        ncols=figure_columns(series),
    )
    fig = plt.figure(figsize=(geometry.fig_w, geometry.fig_h))
    for panel, map_ in enumerate(maps):
        draw_time_map(
            fig.add_axes(geometry.axes_rect(panel)),
            map_,
            units=options.units,
            scale=scale,
            n_levels=style.n_levels,
            cmap_positive=style.cmap_positive,
            cmap_signed=style.cmap_signed,
            data_range=(scales[panel].lo, scales[panel].hi),
            floor=floors[panel],
            nice=style.nice,
            fill=style.fill,
            lines=style.lines,
            cax=fig.add_axes(geometry.cbar_rect(panel)),
            lim=lim,
            quantiles=None if quantiles is None else quantiles[panel],
        )
    fig.suptitle(
        _spacetime_suptitle(series, options.units),
        y=1.0 - 0.3 * _SUP_HEIGHT / geometry.fig_h,
        va="top",
    )
    return fig


def write_spacetime_npz(
    path: Path,
    series: YSeries,
    spec: SeriesSpec,
    maps: list[SpacetimeMap],
    scales: list[PanelScale],
    floors: list[float],
    options: MapOptions,
    ylim: tuple[float, float],
) -> Path:
    r"""Dump everything behind one pair of spacetime figures.

    The plotted arrays as they are drawn -- unrestricted by *ylim*,
    which is stored beside them -- their axes in **both** unit
    systems, the divisor or `$E^{\mathrm{ref}}$` each panel was
    divided by, every factor that was or was not applied, and the
    stream metadata the figures were labelled from.  Enough to redraw
    a panel, to undo the normalisation, or to check a number against
    the stream, without the figures.
    """
    units = options.units
    meta = {
        key: series.meta.get(key)
        for key in (
            *_SHARED_KEYS,
            *_STREAM_KEYS[series.stem],
            "dt",
            "value_dtype",
            "git_hash",
            "twin",
            "it_yspectra",
            "it_ybudget",
        )
        if key in series.meta
    }
    divisors = [m.provenance["divisor"] for m in maps]
    e_refs = [m.provenance["e_ref"] for m in maps]
    payload = {
        "values": np.stack([m.values for m in maps]),
        "panels": np.asarray([m.label for m in maps]),
        "fields": np.asarray([m.name for m in maps]),
        "kinds": np.asarray([m.provenance["kind"] for m in maps]),
        "t": series.t_rel,
        "t_plotted": maps[0].t,
        "y": _half_grid(series.y, options.half),
        "y_plotted": maps[0].y,
        "y_full": series.y,
        "y_weights": series.y_weights,
        # Which records went in: the frame labels the figures are
        # named by (:attr:`YSeries.index`), after --stride / --first /
        # --last.
        "index": series.index,
        "clim": np.asarray([(s.lo, s.hi) for s in scales]),
        "log_floor": np.asarray(floors),
        "ylim": np.asarray(ylim),
        # The divisor of a decorrelation is y-resolved and already
        # carries its factor of two; E_ref is the scalar an absolute
        # spectra panel was divided by.  nan where a panel took
        # neither, so both stay one array per figure.
        "divisor": np.asarray(
            [
                np.full(maps[0].y.size, np.nan) if d is None else d
                for d in divisors
            ]
        ),
        "e_ref": np.asarray(
            [np.nan if s is None else s for s in e_refs], dtype=np.float64
        ),
        "half": options.half,
        "premultiplied": False,
        "volume_fac_applied": options.volume_fac,
        "volume_fac": series.volume_fac,
        "wall_units": units.wall,
        "re": units.re,
        "re_tau": units.re_tau,
        "u_tau": units.u_tau,
        "energy_factor": units.energy(1.0),
        "rate_factor": units.rate(1.0),
        "length_factor": units.length(1.0),
        "time_factor": units.plotted_time(1.0),
        "mode_sum": (
            "every mode; a reference (r_*) loses its (0, 0) mode"
            if not spec.marginal
            else "every mode of the k_x = 0 plane; a reference (r_*) "
            "loses its (0, 0) mode"
        ),
        "stem": series.stem,
        "marginal": spec.marginal,
        "n_members": series.n_members,
        "members": np.asarray([str(m.path) for m in series.members]),
        "parents": np.asarray([m.parent for m in series.members]),
        "meta_json": json.dumps(meta),
    }
    np.savez_compressed(path, **payload)
    return path


def render_spacetime(
    series: YSeries,
    spec: SeriesSpec,
    tag: str,
    out_dir: Path,
    *,
    options: MapOptions,
    style: PlotStyle,
    decades: float = LOG_DECADES,
    declared_signs: bool = True,
    fmt: str = "png",
    quiet: bool = False,
) -> list[Path]:
    """Render one spacetime series: two figures and their ``.npz``.

    ``<tag>_lin`` and ``<tag>_log`` are the same panels, the same
    ranges and the same fold under the two colour scales; the
    logarithmic one is skipped, with a line saying so, for a series
    any of whose panels is signed -- a budget term's `$k$`-sum changes
    sign, and a logarithmic scale of it would be a lie.

    A selection of fewer than two sample times has no time axis to
    draw and is skipped whole, with its own line: that is a
    ``--stride`` / ``--first`` / ``--last`` away on a short run, and
    it must not take the per-sample figures of the same run down with
    it.  The ``.npz`` is written either way -- one row is still the
    data.
    """
    maps = spacetime_maps(series, spec, options)
    ylim = y_limits(series, options, style.ylim)
    scales, floors, notes = spacetime_scales(
        maps, ylim, declared=declared_signs, decades=decades
    )
    if notes and not quiet:
        print("\n".join(notes), flush=True)

    target = out_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    drawable = series.t_rel.size >= 2
    if not drawable and not quiet:
        print(
            f"  {tag}: no figures, a spacetime map needs two sample "
            f"times and this selection has {series.t_rel.size}",
            flush=True,
        )
    for scale in ("lin", "log") if drawable else ():
        if scale == "log" and not all(s.non_negative for s in scales):
            if not quiet:
                print(
                    f"  {tag}: no logarithmic figure, the series changes sign",
                    flush=True,
                )
            continue
        fig = spacetime_figure(
            series,
            maps,
            scales,
            floors,
            options,
            style,
            scale="log" if scale == "log" else "linear",
            ylim=ylim,
        )
        path = target / f"{tag}_{scale}.{fmt}"
        fig.savefig(path, dpi=style.dpi)
        plt.close(fig)
        written.append(path)
        if not quiet:
            print(f"  {path.name}", flush=True)
    written.append(
        write_spacetime_npz(
            target / f"{tag}.npz",
            series,
            spec,
            maps,
            scales,
            floors,
            options,
            ylim,
        )
    )
    if not quiet:
        print(f"  {written[-1].name}", flush=True)
    return written


# ── History maps ─────────────────────────────────────────────────────


def history_panels(
    series: YSeries, spec: SeriesSpec
) -> list[tuple[str, int | None]]:
    """Which ``(base, component)`` panels a history figure carries.

    A spectra series (difference or shape) shows its three components
    and their sum; a budget series the production row
    (:data:`HISTORY_BUDGET_PANELS`).
    """
    if spec.stem == "twin_ybudget":
        balance_terms(series.meta)
        return [(term, None) for term in HISTORY_BUDGET_PANELS]
    return [(spec.base, c) for c in (*range(len(COMPONENTS)), None)]


def history_maps(
    series: YSeries,
    spec: SeriesSpec,
    options: MapOptions,
    ylim: tuple[float, float] | None = None,
) -> dict[str, list]:
    r"""Every panel of one history series, ``{"y": [...], "x": [...], ...}``.

    The `$(y, t)$` maps under ``"y"`` -- marginal-free, so one set,
    which :func:`check_k_sum` asserts rather than assumes -- and one
    `$(\lambda, t)$` set per stored true marginal under its suffix.
    *ylim* is what a shape history reads its row peaks over.
    """
    panels = history_panels(series, spec)
    stored = "e" if spec.stem == "twin_yspectra" else panels[0][0]
    check_k_sum(series, stored)
    out: dict[str, list] = {
        "y": [
            make_spacetime(
                series,
                base,
                options=options,
                component=c,
                premultiply=True,
                ylim=ylim,
            )
            for base, c in panels
        ]
    }
    for marginal in ("x", "z"):
        if marginal in series.suffixes:
            out[marginal] = [
                make_scaletime(
                    series, base, marginal, options=options, component=c
                )
                for base, c in panels
            ]
    return out


def _stream_meta(series: YSeries) -> dict:
    """The sidecar keys a written ``.npz`` records, as a dict."""
    return {
        key: series.meta.get(key)
        for key in (
            *_SHARED_KEYS,
            *_STREAM_KEYS[series.stem],
            "dt",
            "value_dtype",
            "git_hash",
            "twin",
            "it_yspectra",
            "it_ybudget",
        )
        if key in series.meta
    }


def _units_payload(options: MapOptions) -> dict:
    """The unit factors every written ``.npz`` carries."""
    units = options.units
    return {
        "half": options.half,
        "wall_units": units.wall,
        "re": units.re,
        "re_tau": units.re_tau,
        "u_tau": units.u_tau,
        "energy_factor": units.energy(1.0),
        "rate_factor": units.rate(1.0),
        "length_factor": units.length(1.0),
        "time_factor": units.plotted_time(1.0),
    }


def write_history_npz(
    path: Path,
    series: YSeries,
    key: str,
    maps: list,
    scales: list[PanelScale],
    floors: list[float],
    quantiles: list[np.ndarray | None],
    options: MapOptions,
    lim: tuple[float, float],
) -> Path:
    r"""Dump one set of history maps and everything behind them.

    The drawn arrays (unrestricted by *lim*, which is stored beside
    them), the abscissa in both unit systems, the premultiplier, each
    panel's `$E^{\mathrm{ref}}$` or row peaks, the quantile lines and
    the factors that were and were not applied -- enough to redraw a
    panel or undo its scaling without the figure.
    """
    first = maps[0]
    if key == "y":
        abscissa, x, x_plotted = (
            "y",
            _half_grid(series.y, options.half),
            first.y,
        )
    else:
        axis = MARGINALS[key][0]
        abscissa = f"lambda_{axis}"
        x = series.wavelengths(key)
        x_plotted = first.lam
    n_t = first.t.size
    peaks = [m.provenance.get("row_peaks") for m in maps]
    e_refs = [m.provenance.get("e_ref") for m in maps]
    n_q = len(HISTORY_QUANTILES)
    payload = {
        "values": np.stack([m.values for m in maps]),
        "panels": np.asarray([m.label for m in maps]),
        "fields": np.asarray([m.name for m in maps]),
        "t": series.t_rel,
        "t_plotted": first.t,
        "index": series.index,
        "abscissa": abscissa,
        "x": x,
        "x_plotted": x_plotted,
        "premultiplier": first.provenance["premultiplier"],
        "row_peaks": np.asarray(
            [np.full(n_t, np.nan) if p is None else p for p in peaks]
        ),
        "e_ref": np.asarray(
            [np.nan if v is None else v for v in e_refs], dtype=np.float64
        ),
        "quantile_levels": np.asarray(HISTORY_QUANTILES),
        "quantiles": np.asarray(
            [
                np.full((n_t, n_q), np.nan) if q is None else q
                for q in quantiles
            ]
        ),
        "clim": np.asarray([(sc.lo, sc.hi) for sc in scales]),
        "log_floor": np.asarray(floors),
        "lim": np.asarray(lim),
        "volume_fac_applied": bool(options.volume_fac and key == "y"),
        "volume_fac": series.volume_fac,
        "stem": series.stem,
        "n_members": series.n_members,
        "members": np.asarray([str(m.path) for m in series.members]),
        "parents": np.asarray([m.parent for m in series.members]),
        "meta_json": json.dumps(_stream_meta(series)),
        **_units_payload(options),
    }
    np.savez_compressed(path, **payload)
    return path


def render_history(
    series: YSeries,
    spec: SeriesSpec,
    tag: str,
    out_dir: Path,
    *,
    options: MapOptions,
    style: PlotStyle,
    decades: float = LOG_DECADES,
    declared_signs: bool = True,
    fmt: str = "png",
    quiet: bool = False,
) -> list[Path]:
    r"""Render one history series: its `$(y, t)$` and `$(\lambda, t)$` maps.

    ``<tag>_y_*`` is the wall-distance history and ``<tag>_x_*`` /
    ``<tag>_z_*`` the `$\lambda_z$` / `$\lambda_x$` ones, each under
    the linear colour scale and, where every panel is non-negative and
    not a shape history, the logarithmic one as well, which reads the
    growth phase (module docstring, "History maps").  A shape history
    is drawn on `$[0, 1]$`, its rows each over their own peak.  The
    non-negative panels carry their quantile lines
    (:func:`quantile_curves`); each set's ``.npz`` sits beside its
    figures.  A selection of fewer than two sample times draws no
    figure and still writes the data, as a spacetime map does.
    """
    ylim = y_limits(series, options, style.ylim)
    sets = history_maps(series, spec, options, ylim)
    shape = spec.base == SHAPE
    target = out_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    drawable = series.t_rel.size >= 2
    if not drawable and not quiet:
        print(
            f"  {tag}: no figures, a history needs two sample times and "
            f"this selection has {series.t_rel.size}",
            flush=True,
        )
    for key, maps in sets.items():
        if key == "y":
            lim = ylim
            x_log = options.y_log
        else:
            lim = style.xlim or (
                float(maps[0].lam.min()),
                float(maps[0].lam.max()),
            )
            x_log = True
        scales, floors, notes = spacetime_scales(
            maps, lim, declared=declared_signs, decades=decades
        )
        if notes and not quiet:
            print("\n".join(notes), flush=True)
        if shape:
            scales = [PanelScale(0.0, 1.0, True) for _ in maps]
        quantiles = []
        for map_, sc in zip(maps, scales, strict=True):
            if not sc.non_negative:
                quantiles.append(None)
                continue
            x, values = map_.drawn(lim)
            quantiles.append(quantile_curves(x, values, x_log=x_log))
        for scale in ("lin", "log") if drawable else ():
            if scale == "log" and (
                shape or not all(sc.non_negative for sc in scales)
            ):
                continue
            fig = time_map_figure(
                series,
                maps,
                scales,
                floors,
                options,
                style,
                scale="log" if scale == "log" else "linear",
                lim=lim,
                quantiles=quantiles,
            )
            path = target / f"{tag}_{key}_{scale}.{fmt}"
            fig.savefig(path, dpi=style.dpi)
            plt.close(fig)
            written.append(path)
            if not quiet:
                print(f"  {path.name}", flush=True)
        written.append(
            write_history_npz(
                target / f"{tag}_{key}.npz",
                series,
                key,
                maps,
                scales,
                floors,
                quantiles,
                options,
                lim,
            )
        )
        if not quiet:
            print(f"  {written[-1].name}", flush=True)
    return written


# ── Moment budget ────────────────────────────────────────────────────

#: The rows of a moment budget figure: ``(MomentRates field, label)``
#: in the order drawn, for the wall-distance budget (physical space)
#: and for a marginal's joint one (Fourier space).
_MOMENT_ROWS_Y: tuple[tuple[str, str], ...] = (
    ("mass", r"$\mathrm{d}\ln E/\mathrm{d}t$"),
    ("mean_y", r"$\mathrm{d}\langle\ln y\rangle/\mathrm{d}t$"),
    ("var_y", r"$\mathrm{d}\sigma^2_{\ln y}/\mathrm{d}t$"),
)
_MOMENT_ROWS_JOINT: tuple[tuple[str, str], ...] = (
    ("mass", r"$\mathrm{d}\ln E'/\mathrm{d}t$"),
    ("mean_lam", r"$\mathrm{d}\langle\ln\lambda\rangle/\mathrm{d}t$"),
    ("mean_y", r"$\mathrm{d}\langle\ln y\rangle/\mathrm{d}t$"),
    ("var_lam", r"$\mathrm{d}\sigma^2_{\ln\lambda}/\mathrm{d}t$"),
    ("var_y", r"$\mathrm{d}\sigma^2_{\ln y}/\mathrm{d}t$"),
    ("cov", r"$\mathrm{d}C_{\lambda y}/\mathrm{d}t$"),
)


def moment_terms(marginal: str) -> tuple[str, ...]:
    """The term groups of one moment budget (:data:`MOMENT_TERMS`).

    The wall-distance budget covers every mode, `$(0, 0)$` included,
    so its pressure group carries the driving input as well
    (``press_input``); a marginal's covers `$m \\ge 1$`, where the
    input has no entry and the pressure transport is all there is.
    """
    if marginal:
        return MOMENT_TERMS
    return tuple(
        "press_input" if term == "tr_press" else term for term in MOMENT_TERMS
    )


def _moment_cells(
    series: YSeries, marginal: str, options: MapOptions
) -> tuple[np.ndarray, np.ndarray, np.ndarray, slice]:
    r"""``(weights, ln_lam, ln_y, columns)`` of a moment budget's cells.

    The folded rows off the wall (the wall row has no `$\ln y$`, and
    no energy either), weighted by :func:`half_weights`, so a cell's
    weight times its entry is its exact share of the volume-averaged
    energy: the distribution of energy itself, not of a map.  A
    marginal's columns are `$m \ge 1$`, ascending in `$m$` as stored,
    with `$\ln\lambda$` in the plotted units; the wall-distance budget
    sums every column into one, at a dummy `$\ln\lambda = 0$`.
    Both coordinates are in the plotted units, which shifts a mean and
    moves no rate.
    """
    y = options.units.length(_half_grid(series.y, options.half))
    rows = y > 0.0
    weights = half_weights(series)[rows][:, None]
    ln_y = np.log(y[rows])
    if not marginal:
        return weights, np.zeros(1), ln_y, slice(None)
    lam = options.units.length(
        float(series.meta["lx" if marginal == "z" else "lz"])
        / series.harmonics(marginal)[1:]
    )
    return weights, np.log(lam), ln_y, slice(1, None)


def moment_budget_sums(
    spectra: YSeries,
    budget: YSeries,
    marginal: str,
    options: MapOptions,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    r"""Raw moment sums of the energy and of each term group, per frame.

    ``(energy (n_t, 6), terms (n_t, n_groups, 6), groups)``, ensemble
    means read in chunks (:meth:`YSeries.reduced`) -- exact, the sums
    being linear (:mod:`dnsjax.analysis.twin.moments`).  The energy is
    the component-summed difference spectrum, the only one the
    component-summed budget can explain.  *marginal* empty is the
    wall-distance budget over every mode (the history `$(y, t)$`
    map's distribution); ``x`` / ``z`` the joint `$(\ln\lambda,
    \ln y)$` one over `$m \ge 1$`.
    """
    weights, ln_lam, ln_y, columns = _moment_cells(spectra, marginal, options)
    suffix = marginal or "x"
    off_wall = options.units.length(_half_grid(spectra.y, options.half)) > 0.0

    def cells(values: np.ndarray) -> np.ndarray:
        folded, _ = _select_half(values, spectra.y, options.half)
        folded = folded[..., off_wall, :]
        if marginal:
            return folded[..., columns]
        return folded.sum(axis=-1, keepdims=True)

    def energy(read) -> np.ndarray:
        e = np.asarray(read(f"e_{suffix}"), dtype=np.float64).sum(axis=1)
        return log_moment_sums(cells(e), weights, ln_lam, ln_y)

    groups = moment_terms(marginal)

    def terms(read) -> np.ndarray:
        return np.stack(
            [
                log_moment_sums(
                    cells(balance_field(read, budget.meta, f"{g}_{suffix}")),
                    weights,
                    ln_lam,
                    ln_y,
                )
                for g in groups
            ],
            axis=1,
        )

    return spectra.reduced(energy), budget.reduced(terms), groups


def _check_shared_frames(spectra: YSeries, budget: YSeries) -> None:
    """A moment budget needs both streams on one frame grid."""
    if spectra.t_rel.shape != budget.t_rel.shape or not np.allclose(
        spectra.t_rel, budget.t_rel, rtol=0.0, atol=_T_ATOL
    ):
        raise ValueError(
            "the moment budget sets the rates of the spectra stream's "
            "moments against the budget stream's terms frame by frame, "
            "so both need the same frames; these were recorded on "
            f"different grids ({spectra.t_rel.size} vs "
            f"{budget.t_rel.size} frames).  Select frames both streams "
            "carry with --stride / --first / --last."
        )


def moment_budget(
    spectra: YSeries,
    budget: YSeries,
    marginal: str,
    options: MapOptions,
) -> dict:
    r"""Every term's share of every moment rate, and the check on them.

    A dict of ``(n_groups, n_t)`` contributions per
    :class:`~dnsjax.analysis.twin.moments.MomentRates` field, their
    sum over the groups, and the same rate by second-order finite
    differences of the moments themselves -- the closure check: the
    identity is exact, so whatever separates the two is the budget
    stream's own closure and the sampling cadence.  Rates are per unit
    of plotted time.
    """
    _check_shared_frames(spectra, budget)
    e_sums, t_sums, groups = moment_budget_sums(
        spectra, budget, marginal, options
    )
    per_time = 1.0 / options.units.plotted_time(1.0)
    rates = [moment_rates(e_sums, t_sums[:, g]) for g in range(len(groups))]
    m = log_moments(e_sums)
    t = spectra.t_rel
    fields = ("mass", "mean_lam", "mean_y", "var_lam", "var_y", "cov")
    with np.errstate(divide="ignore", invalid="ignore"):
        own = {
            "mass": np.gradient(np.log(m.mass), t),
            "mean_lam": np.gradient(m.mean_lam, t),
            "mean_y": np.gradient(m.mean_y, t),
            "var_lam": np.gradient(m.var_lam, t),
            "var_y": np.gradient(m.var_y, t),
            "cov": np.gradient(m.cov, t),
        }
    contributions = {
        name: per_time * np.stack([getattr(r, name) for r in rates])
        for name in fields
    }
    return {
        "groups": groups,
        "contributions": contributions,
        "total": {k: v.sum(axis=0) for k, v in contributions.items()},
        "finite_difference": {k: per_time * v for k, v in own.items()},
        "moments": m,
    }


def moment_budget_figure(
    series: YSeries,
    result: dict,
    marginal: str,
    options: MapOptions,
    style: PlotStyle,
):
    r"""The moment budget against time, a row per moment rate.

    Each term group a line (:data:`_TERM_SERIES`, dashed for a second
    cue), their sum black and the finite-difference rate of the
    moments grey and dashed: where the two agree, the terms account
    for all of the motion (module docstring, "Moment budget").
    """
    units = options.units
    t = units.plotted_time(series.t_rel)
    rows = _MOMENT_ROWS_JOINT if marginal else _MOMENT_ROWS_Y
    fig, axes = plt.subplots(
        len(rows),
        1,
        sharex=True,
        figsize=(style.width, _ROW_HEIGHT * len(rows) + 0.9),
        layout="constrained",
    )
    for ax, (name, label) in zip(axes, rows, strict=True):
        for index, group in enumerate(result["groups"]):
            sign, symbol = TERM_LABELS[group]
            ax.plot(
                t,
                result["contributions"][name][index],
                color=_TERM_SERIES[index],
                linestyle=_TERM_DASHES[index],
                linewidth=1.0,
                label=f"${sign}{symbol}$",
            )
        ax.plot(
            t, result["total"][name], color="black", linewidth=1.3, label="sum"
        )
        ax.plot(
            t,
            result["finite_difference"][name],
            color="0.55",
            linewidth=1.3,
            linestyle=(0, (2, 1.5)),
            label="moments, differenced",
        )
        ax.axhline(0.0, color="0.75", linewidth=0.5)
        # Per unit of the plotted time: t+ in wall units.
        ax.set_ylabel(
            label.replace(r"\mathrm{d}t$", r"\mathrm{d}t^+$")
            if units.wall
            else label
        )
        ax.grid(True, color="0.92", linewidth=0.5)
    axes[-1].set_xlabel(units.t_label)
    axes[-1].set_xlim(float(t[0]), float(t[-1]))
    if units.wall:
        factor = units.re_tau**2 / units.re
        axes[0].secondary_xaxis(
            "top", functions=(lambda v: v / factor, lambda v: v * factor)
        ).set_xlabel(r"$t\,U_\mathrm{cl}/h$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=5,
        frameon=False,
    )
    where = (
        "wall distance, every mode"
        if not marginal
        else rf"$(\lambda_{{{MARGINALS[marginal][0]}}}, y)$, $m \ge 1$"
    )
    fig.suptitle(
        "Moment budget of the difference energy: "
        + where
        + "\n"
        + _spacetime_suptitle(series, units)
    )
    return fig


def write_moment_budget_npz(
    path: Path,
    series: YSeries,
    result: dict,
    marginal: str,
    options: MapOptions,
) -> Path:
    """Dump a moment budget: contributions, sum, check and moments."""
    m = result["moments"]
    fields = ("mass", "mean_lam", "mean_y", "var_lam", "var_y", "cov")
    payload = {
        "groups": np.asarray(result["groups"]),
        "rates": np.asarray(fields),
        "contributions": np.stack(
            [result["contributions"][f] for f in fields], axis=1
        ),
        "total": np.stack([result["total"][f] for f in fields]),
        "finite_difference": np.stack(
            [result["finite_difference"][f] for f in fields]
        ),
        "mass": m.mass,
        "mean_lam": m.mean_lam,
        "mean_y": m.mean_y,
        "var_lam": m.var_lam,
        "var_y": m.var_y,
        "cov": m.cov,
        "t": series.t_rel,
        "t_plotted": options.units.plotted_time(series.t_rel),
        "index": series.index,
        "marginal": marginal,
        "cells": (
            "folded rows off the wall, every mode summed"
            if not marginal
            else "folded rows off the wall, m >= 1"
        ),
        "weights": "half_weights (exact energies)",
        "coordinates": "ln of the plotted wavelength and wall distance",
        "rate_units": "per unit of plotted time",
        "n_members": series.n_members,
        "members": np.asarray([str(mm.path) for mm in series.members]),
        "meta_json": json.dumps(_stream_meta(series)),
        **_units_payload(options),
    }
    np.savez_compressed(path, **payload)
    return path


def render_moment_budget(
    spectra: YSeries,
    budget: YSeries,
    spec: SeriesSpec,
    tag: str,
    out_dir: Path,
    *,
    options: MapOptions,
    style: PlotStyle,
    fmt: str = "png",
    quiet: bool = False,
) -> list[Path]:
    """Render one moment budget: its figure and its ``.npz``."""
    result = moment_budget(spectra, budget, spec.marginal, options)
    target = out_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if spectra.t_rel.size >= 3:
        fig = moment_budget_figure(
            spectra, result, spec.marginal, options, style
        )
        path = target / f"{tag}.{fmt}"
        fig.savefig(path, dpi=style.dpi)
        plt.close(fig)
        written.append(path)
    elif not quiet:
        print(
            f"  {tag}: no figure, a rate needs three sample times and "
            f"this selection has {spectra.t_rel.size}",
            flush=True,
        )
    written.append(
        write_moment_budget_npz(
            target / f"{tag}.npz", spectra, result, spec.marginal, options
        )
    )
    if not quiet:
        for path in written:
            print(f"  {path.name}", flush=True)
    return written


# ── Decorrelation front ──────────────────────────────────────────────


def front_times(
    t: np.ndarray, ratio: np.ndarray, level: float = FRONT_LEVEL
) -> np.ndarray:
    r"""When each cell's ratio rises through *level* for good.

    *ratio* is ``(n_t, ...)``; the result has its trailing shape.  The
    time is the **last** upward crossing of *level*, interpolated
    linearly between the two frames that bracket it: the earliest time
    after which the ratio stays at or above *level*.  A cell that is
    never below it has ``t[0]`` (a seed already above the level), one
    that ends below it ``nan`` (it has not decorrelated by the end of
    the record).  Taking the last crossing rather than the first is
    what keeps a seed that starts above the level and dips below it
    from being timed at ``t[0]``.
    """
    r = np.asarray(ratio, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    above = np.where(np.isfinite(r), r >= level, False)
    below = ~above
    n = r.shape[0]
    any_below = below.any(axis=0)
    last_below = n - 1 - np.argmax(below[::-1], axis=0)
    out = np.where(any_below, np.nan, t[0])
    crossed = any_below & above[-1]
    i0 = np.where(crossed, last_below, 0)
    i1 = np.minimum(i0 + 1, n - 1)
    r0 = np.take_along_axis(r, i0[None], axis=0)[0]
    r1 = np.take_along_axis(r, i1[None], axis=0)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip((level - r0) / (r1 - r0), 0.0, 1.0)
    return np.where(crossed, t[i0] + frac * (t[i1] - t[i0]), out)


def front_maps(
    series: YSeries,
    marginal: str,
    options: MapOptions,
    level: float = FRONT_LEVEL,
) -> list[Map]:
    r"""The four `$t_{1/2}$` panels of one marginal: u, v, w and the sum.

    Each from the mode-by-mode decorrelation
    `$\mathcal{R} = e/(2\langle r\rangle_t)$` (the ``decorr`` maps'
    ratio, :func:`map_divisor`, symmetrised before the fold), the summed
    panel one ratio of sums; then folded, `$m = 0$` dropped, ascending
    in wavelength, and timed (:func:`front_times`) in the plotted
    units.  A :class:`Map` per panel, so the maps' geometry serves
    them.
    """
    e = series.field(f"e_{marginal}")
    divisor = map_divisor(series, DECORR, marginal, options.half)
    t = options.units.plotted_time(series.t_rel)
    maps = []
    for component in (*range(len(COMPONENTS)), None):
        if component is None:
            ratio = _ratio(e.sum(axis=1), 2.0 * divisor.sum(axis=0))
        else:
            ratio = _ratio(e[:, component], 2.0 * divisor[component])
        ratio = ratio[..., 1:][..., ::-1]
        folded, wall_distance = _select_half(ratio, series.y, options.half)
        _, symbol = panel_symbol(series, f"e_{marginal}", component)
        maps.append(
            Map(
                lam=options.units.length(series.wavelengths(marginal)),
                y=options.units.length(wall_distance),
                values=front_times(t, folded, level),
                title=rf"$t{options.units.suffix}_{{1/2}}$ of ${symbol}$",
                name=f"e_{marginal}",
                non_negative=True,
                y_log=options.y_log,
            )
        )
    return maps


def draw_front(
    ax,
    map_: Map,
    *,
    units: Units,
    data_range: tuple[float, float],
    n_levels: int = 10,
    cmap: str = "Blues",
    cax=None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    r"""One `$t_{1/2}$` panel: every band filled, the iso-time lines drawn.

    Unlike a spectrum's map no band is left unfilled -- the earliest
    times are as much data as the latest -- and a cell that never stays
    above the level (``nan``) is left white, which the colour bar's
    range never reaches.  The contour lines are the front's positions
    at the labelled times.
    """
    ax.set_xscale("log")
    ax.set_yscale("log" if map_.y_log else "linear")
    y, values = map_.drawn()
    lo, hi = data_range
    step = nice_step(max(hi - lo, 1e-12) / max(n_levels, 2))
    levels = np.arange(
        math.floor(lo / step) * step, math.ceil(hi / step) * step + step, step
    )
    filled = None
    if levels.size > 1 and np.isfinite(values).any():
        filled = ax.contourf(
            map_.lam, y, values, levels=levels, cmap=cmap, extend="neither"
        )
        ax.contour(
            map_.lam,
            y,
            values,
            levels=levels,
            colors="k",
            linewidths=0.3,
            alpha=0.7,
        )
    ax.set_xlim(*(xlim or (map_.lam.min(), map_.lam.max())))
    ax.set_ylim(*(ylim or (y.min(), y.max())))
    if map_.y_log:
        ax.set_aspect(1.0, adjustable="box", anchor="C")
    ax.set_xlabel(units.lambda_label(map_.lam_axis))
    ax.set_ylabel(units.y_label)
    if units.wall:
        outer = (lambda v: v / units.re_tau, lambda v: v * units.re_tau)
        ax.secondary_xaxis("top", functions=outer).set_xlabel(
            units.lambda_label(map_.lam_axis, outer=True)
        )
        ax.secondary_yaxis("right", functions=outer).set_ylabel(r"$y/h$")
    ax.set_title(map_.title, pad=_TITLE_PAD)
    if cax is not None and filled is not None:
        bar = ax.figure.colorbar(filled, cax=cax, ticks=_bar_ticks(levels))
        bar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, _pos: f"{v:.3g}")
        )
        bar.ax.tick_params(labelsize="small")
        bar.set_label(units.t_label, fontsize="small")
    elif cax is not None:
        cax.set_axis_off()
    return filled


def render_front(
    series: YSeries,
    spec: SeriesSpec,
    tag: str,
    out_dir: Path,
    *,
    options: MapOptions,
    style: PlotStyle,
    level: float = FRONT_LEVEL,
    cmap: str = "Blues",
    fmt: str = "png",
    quiet: bool = False,
) -> list[Path]:
    r"""Render one marginal's decorrelation front: a figure and an ``.npz``.

    The four panels share one colour range, the extremes of every
    finite `$t_{1/2}$` the box shows, so a colour is one time across
    the components (module docstring, "Decorrelation front").
    """
    maps = front_maps(series, spec.marginal, options, level)
    ylim = y_limits(series, options, style.ylim)
    shown = np.concatenate(
        [m.drawn(ylim)[1][np.isfinite(m.drawn(ylim)[1])] for m in maps]
    )
    data_range = (
        (float(shown.min()), float(shown.max())) if shown.size else (0.0, 1.0)
    )
    target = out_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    xlim = style.xlim or (float(maps[0].lam.min()), float(maps[0].lam.max()))
    geometry = panel_geometry(
        len(maps), xlim, ylim, style, y_log=options.y_log
    )
    fig = plt.figure(figsize=(geometry.fig_w, geometry.fig_h))
    for panel, map_ in enumerate(maps):
        draw_front(
            fig.add_axes(geometry.axes_rect(panel)),
            map_,
            units=options.units,
            data_range=data_range,
            n_levels=style.n_levels,
            cmap=cmap,
            cax=fig.add_axes(geometry.cbar_rect(panel)),
            xlim=xlim,
            ylim=ylim,
        )
    fig.suptitle(
        rf"Time $\mathcal{{R}} = e/2\langle r\rangle$ rises through "
        rf"${level:g}$ for good (white: not by the end)"
        + "\n"
        + _spacetime_suptitle(series, options.units),
        y=1.0 - 0.15 * _SUP_HEIGHT / geometry.fig_h,
        va="top",
        fontsize="medium",
    )
    path = target / f"{tag}.{fmt}"
    fig.savefig(path, dpi=style.dpi)
    plt.close(fig)
    units = options.units
    npz = target / f"{tag}.npz"
    np.savez_compressed(
        npz,
        t_half_plotted=np.stack([m.values for m in maps]),
        t_half=np.stack([m.values for m in maps]) / units.plotted_time(1.0),
        lam=series.wavelengths(spec.marginal),
        lam_plotted=maps[0].lam,
        y=_half_grid(series.y, options.half),
        y_plotted=maps[0].y,
        panels=np.asarray(["u", "v", "w", "sum"]),
        level=level,
        ratio="e / (2 <r>_t), mode by mode, (0,0) mode off the reference",
        n_members=series.n_members,
        members=np.asarray([str(m.path) for m in series.members]),
        meta_json=json.dumps(_stream_meta(series)),
        **_units_payload(options),
    )
    if not quiet:
        print(f"  {path.name}\n  {npz.name}", flush=True)
    return [path, npz]


# ── Growth laws ──────────────────────────────────────────────────────

#: The budget groups a growth figure reads per unit band energy: the
#: same-`$k$` mean-shear production, and the three cross-scale terms
#: (module docstring, "Growth laws") -- the fluctuation production and
#: the two advective transfers, which only move energy between modes
#: and wall distances -- with the dissipation that drains each band.
GROWTH_TERMS: tuple[str, ...] = (
    "prod_mean",
    "prod_fluct",
    "tr_ref",
    "tr_self",
    "diss",
)

#: The wall distances, in wall units, whose rows a ``growth_y`` figure
#: follows: one per octave from the viscous sublayer to the centreline,
#: each the nearest row of the grid.
GROWTH_Y_PLUS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)


@dataclass(frozen=True)
class GrowthCurves:
    r"""A family of growth curves on one clock (module docstring,
    "Growth laws").

    ``energy`` is ``(n_curves, n_t)`` and ``saturation`` ``(n_curves,)``
    -- twice the reference's, the energy two independent fields
    differ by -- so `$R$` = ``energy / saturation[:, None]``.
    ``coords`` places each curve on a scale (a wavelength or a wall
    distance, plotted units) where one exists, ``members`` holds each
    member's own energy for a single curve, and ``rates`` the budget
    groups per unit energy, ``{group: (n_curves, n_t)}``, where the
    budget stream was read.
    """

    t: np.ndarray
    energy: np.ndarray
    saturation: np.ndarray
    labels: list[str]
    coords: np.ndarray | None = None
    members: np.ndarray | None = None
    rates: dict[str, np.ndarray] | None = None
    source: str = "twin_yspectra"

    @property
    def r(self) -> np.ndarray:
        """`$R = E/E_\\mathrm{sat}$`, per curve."""
        return self.energy / self.saturation[:, None]

    @property
    def gamma(self) -> np.ndarray:
        r"""`$\gamma = \mathrm{d}\ln E/\mathrm{d}t$`, per curve."""
        return log_rate(self.t, self.energy)

    @property
    def f(self) -> np.ndarray:
        r"""`$-\ln(1 - R)$`, ``nan`` above :data:`GROWTH_R_MAX`."""
        r = self.r
        return np.where(r <= GROWTH_R_MAX, bound_free(r), np.nan)


def _y_averaged_many(
    series: YSeries, names: list[str], half: str
) -> list[np.ndarray]:
    """Several stored or balance fields averaged over `$y$`, in one pass.

    One array per name: ``(n_t, [3,] n_k)``, or ``(n_t, [3])`` for a
    `$(0, 0)$`-mode name -- suffix ``xz00``, or ``x0:00`` for a legacy
    ``x0`` plane's first column -- whose profile has no wavenumber
    axis.  The records are read once whatever the number of names
    (:meth:`YSeries.reduced`), each field flattened for the pass and
    reshaped after it.
    """
    weights = half_weights(series)
    shapes: list[tuple[int, ...]] = []

    def one(read, name: str) -> np.ndarray:
        if name.endswith(("_xz00", "_x0:00")):
            if name.endswith("_x0:00"):
                values = balance_field(read, series.meta, name[:-3])[..., 0]
            else:
                values = balance_field(read, series.meta, name)
            folded, _ = _select_half(values[..., None], series.y, half)
            return np.einsum("j,...j->...", weights, folded[..., 0])
        folded, _ = _select_half(
            balance_field(read, series.meta, name), series.y, half
        )
        return np.einsum("j,...jk->...k", weights, folded)

    def reduce(read) -> np.ndarray:
        parts = [one(read, n) for n in names]
        if not shapes:
            shapes.extend(p.shape[1:] for p in parts)
        return np.concatenate(
            [p.reshape(p.shape[0], -1) for p in parts], axis=1
        )

    flat = series.reduced(reduce)
    out, start = [], 0
    for shape in shapes:
        size = int(np.prod(shape)) if shape else 1
        out.append(
            flat[:, start : start + size].reshape(flat.shape[0], *shape)
        )
        start += size
    return out


def _mean_mode_suffix(series: YSeries) -> str:
    """How a stream stores its `$(0, 0)$` mode (:func:`_y_averaged_many`)."""
    return "xz00" if "xz00" in series.suffixes else "x0:00"


def _reference_bands(series: YSeries, marginal: str, half: str) -> np.ndarray:
    r"""`$\langle r\rangle_t$` averaged over `$y$`, ``(3, n_k)``, mean-free."""
    folded, _ = _select_half(
        series.reference_spectrum(marginal), series.y, half
    )
    return np.einsum("j,...jk->...k", half_weights(series), folded)


def _band_rates(
    budget: YSeries | None,
    marginal: str,
    columns: np.ndarray,
    energy: np.ndarray,
    half: str,
) -> dict[str, np.ndarray] | None:
    r"""The budget groups per unit band energy, ``{group: (n_bands, n_t)}``.

    Read off the budget stream's *marginal* at the band *columns*, the
    `$(0, 0)$` mode taken off column 0 as it is off the energy, and
    divided by the band's *energy* ``(n_bands, n_t)``.  ``None``
    without a budget stream, or one on other frames.
    """
    if budget is None:
        return None
    try:
        balance_terms(budget.meta)
    except ValueError:
        return None
    mm = _mean_mode_suffix(budget)
    names = [f"{g}_{marginal}" for g in GROWTH_TERMS]
    mean_names = [f"{g}_{mm}" for g in GROWTH_TERMS]
    values = _y_averaged_many(budget, names + mean_names, half)
    out = {}
    for index, group in enumerate(GROWTH_TERMS):
        bands = values[index][:, columns].T.copy()
        mean = values[len(GROWTH_TERMS) + index]
        bands[columns == 0] -= mean
        with np.errstate(divide="ignore", invalid="ignore"):
            out[group] = bands / energy
    return out


def _band_columns(n_k: int) -> np.ndarray:
    """The bands a growth figure follows: `$m = 0$` and every octave."""
    octaves = [1]
    while octaves[-1] * 2 < n_k:
        octaves.append(octaves[-1] * 2)
    return np.asarray([0, *octaves])


def growth_bands(
    series: YSeries,
    marginal: str,
    options: MapOptions,
    budget: YSeries | None = None,
) -> GrowthCurves:
    r"""The difference energy of one marginal's bands, on the frames.

    Component-summed and averaged over `$y$` (:func:`y_averaged`): the
    `$m = 0$` column -- with the `$(0, 0)$` mode taken off it, the
    mean-flow difference being no fluctuation -- and one band per
    octave of `$m$`, each against twice its own reference energy.
    The budget groups per unit band energy come along where *budget*
    is given on the same frames.
    """
    mm = _mean_mode_suffix(series)
    band, mean = _y_averaged_many(
        series, [f"e_{marginal}", f"e_{mm}"], options.half
    )
    k = band.sum(axis=1)  # (n_t, n_k)
    k[:, 0] -= mean.sum(axis=1)
    columns = _band_columns(k.shape[1])
    ref = _reference_bands(series, marginal, options.half).sum(axis=0)
    energy = k[:, columns].T
    lam = options.units.length(
        float(series.meta["lx" if marginal == "z" else "lz"])
        / np.maximum(series.harmonics(marginal)[columns], 1)
    )
    axis = MARGINALS[marginal][0]
    plus = options.units.suffix
    labels = [
        rf"$k_{axis} = 0$"
        if c == 0
        else rf"$\lambda_{axis}{plus} = {lam[i]:.0f}$"
        for i, c in enumerate(columns)
    ]
    coords = np.where(columns == 0, np.inf, lam)
    rates = None
    if budget is not None and np.allclose(
        budget.t_rel, series.t_rel, rtol=0.0, atol=_T_ATOL
    ):
        rates = _band_rates(budget, marginal, columns, energy, options.half)
    return GrowthCurves(
        t=series.t_rel,
        energy=energy,
        saturation=2.0 * ref[columns],
        labels=labels,
        coords=coords,
        rates=rates,
    )


def growth_rows(
    series: YSeries,
    options: MapOptions,
    budget: YSeries | None = None,
) -> GrowthCurves:
    r"""The difference energy at a few wall distances, every mode summed.

    The `$k$`-summed profile (:func:`k_summed`) less its `$(0, 0)$`
    mode, at the rows nearest :data:`GROWTH_Y_PLUS`, against twice the
    reference's fluctuation profile there.  Where *budget* is given,
    the production and the total transport per unit energy, each row
    being fed by its own production and by what transport brings it.
    """
    weightless = k_summed(series, "e").sum(axis=1)  # (n_t, n_y)
    mm_name = mean_mode_name(series.meta, "e")
    mean = series.reduced(
        lambda read: mean_mode_profile(read(mm_name), mm_name).sum(axis=1)
    )
    profile, wall = _select_half(
        (weightless - mean)[..., None], series.y, options.half
    )
    profile = profile[..., 0]
    ref, _ = _select_half(
        series.reference_profile().sum(axis=0)[:, None], series.y, options.half
    )
    ref = ref[:, 0]
    y_plus = wall * options.units.re_tau
    rows = sorted(
        {int(np.argmin(np.abs(y_plus - target))) for target in GROWTH_Y_PLUS}
    )
    rows = [r for r in rows if y_plus[r] > 0.0]
    energy = profile[:, rows].T
    coords = options.units.length(wall[rows])
    labels = [rf"$y{options.units.suffix} = {c:.3g}$" for c in coords]
    rates = None
    if budget is not None and np.allclose(
        budget.t_rel, series.t_rel, rtol=0.0, atol=_T_ATOL
    ):
        try:
            balance_terms(budget.meta)
            groups = {
                "prod": ["prod"],
                "transport": ["tr_visc", "tr_press", "tr_ref", "tr_self"],
            }
            rates = {}
            mm = _mean_mode_suffix(budget)
            for key, terms in groups.items():
                total = sum(k_summed(budget, term) for term in terms)
                mean_mode = sum(
                    budget.reduced(
                        lambda read, term=term: (
                            balance_field(read, budget.meta, f"{term}_xz00")
                            if mm == "xz00"
                            else balance_field(
                                read, budget.meta, f"{term}_x0"
                            )[..., 0]
                        )
                    )
                    for term in terms
                )
                folded, _ = _select_half(
                    (total - mean_mode)[..., None], series.y, options.half
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    rates[key] = folded[..., 0][:, rows].T / energy
        except ValueError:
            rates = None
    return GrowthCurves(
        t=series.t_rel,
        energy=energy,
        saturation=2.0 * ref[rows],
        labels=labels,
        coords=coords,
        rates=rates,
    )


def growth_ssp(
    series: YSeries,
    options: MapOptions,
    budget: YSeries | None = None,
) -> GrowthCurves:
    r"""The difference energy of the self-sustaining process's three parts.

    Off the `$k_x$` marginal, per component: the **streaks**, `$\Delta
    u$` at `$k_x = 0$`; the **rolls**, `$\Delta v$` and `$\Delta w$`
    there; the **waves**, every component at `$k_x \neq 0$` -- the
    `$(0, 0)$` mode taken off the first two, each against twice its
    reference share (module docstring, "Growth laws").  These are the
    three-bin quantities of :func:`~dnsjax.analysis.twin.yspectra.
    bin_energies` with the streak bin split by component.  The budget
    stream is component-summed, so where *budget* is given the rates
    are those of the whole `$k_x = 0$` plane (streaks and rolls
    together) and of the waves, both per unit of their own energy:
    the lift-up `$\mathcal{P}^{\mathbf{U}}_\Delta$` and the transfer
    out of the plane are the SSP's growth and breakdown legs.
    """
    mm = _mean_mode_suffix(series)
    ez, mean = _y_averaged_many(series, ["e_z", f"e_{mm}"], options.half)
    ref = _reference_bands(series, "z", options.half)  # (3, n_kx)
    streak = ez[:, 0, 0] - mean[:, 0]
    roll = ez[:, 1, 0] + ez[:, 2, 0] - mean[:, 1] - mean[:, 2]
    wave = ez[:, :, 1:].sum(axis=(1, 2))
    energy = np.stack([streak, roll, wave])
    saturation = 2.0 * np.array(
        [ref[0, 0], ref[1, 0] + ref[2, 0], ref[:, 1:].sum()]
    )
    rates = None
    if budget is not None and np.allclose(
        budget.t_rel, series.t_rel, rtol=0.0, atol=_T_ATOL
    ):
        try:
            balance_terms(budget.meta)
            bmm = _mean_mode_suffix(budget)
            names = [f"{g}_z" for g in GROWTH_TERMS]
            means = [f"{g}_{bmm}" for g in GROWTH_TERMS]
            b = _y_averaged_many(budget, names + means, options.half)
            plane = streak + roll
            rates = {}
            for index, group in enumerate(GROWTH_TERMS):
                kx0 = b[index][:, 0] - b[len(GROWTH_TERMS) + index]
                waves = b[index][:, 1:].sum(axis=-1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rates[group] = np.stack([kx0 / plane, waves / wave])
        except ValueError:
            rates = None
    return GrowthCurves(
        t=series.t_rel,
        energy=energy,
        saturation=saturation,
        labels=[
            r"streaks $\Delta u,\ k_x = 0$",
            r"rolls $\Delta v, \Delta w,\ k_x = 0$",
            r"waves, $k_x \neq 0$",
        ],
        rates=rates,
    )


def growth_global(series: YSeries, options: MapOptions) -> GrowthCurves:
    r"""The total difference energy, every member, on its finest clock.

    Each member's ``twin.dat`` where every member has one -- the
    ``twin.it_energy`` cadence, finer than the spectra's -- aligned on
    whole steps since the perturbation (:func:`~dnsjax.analysis.twin.
    series.relative_time`) and restricted to the frames' window; the
    spectra stream's own per-member totals otherwise.  The curve is the
    **geometric** member mean, `$\exp\langle\ln E\rangle$`: a rate is a
    logarithmic derivative, and the arithmetic mean would let the
    fastest member set it.  `$E_\mathrm{sat}$` is twice the reference's
    fluctuation energy (:meth:`YSeries.reference_scale`).
    """
    lo, hi = float(series.t_rel[0]), float(series.t_rel[-1])
    curves = []
    source = "twin.dat"
    try:
        for member in series.members:
            twin = read_twin(member.path)
            dt, on_grid = uniform_grid(twin.t)
            t_rel = twin.t_rel[on_grid]
            step = float(twin.meta["dt"]) if twin.meta else dt
            keep = (t_rel >= lo - 0.5 * step) & (t_rel <= hi + 0.5 * step)
            keys = np.rint(t_rel[keep] / step).astype(np.int64)
            curves.append((keys, twin.energies["E_d"][on_grid][keep], step))
        common = curves[0][0]
        for keys, _, _ in curves[1:]:
            common = np.intersect1d(common, keys)
        energies = np.stack(
            [e[np.searchsorted(keys, common)] for keys, e, _ in curves]
        )
        t = common * curves[0][2]
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        # Every member's twin.dat or none: a mixed set would put the
        # members on two clocks.  Which source was used is recorded.
        energies, t = member_energies(series), series.t_rel
        source = f"twin_yspectra ({type(exc).__name__}: {exc})"
    with np.errstate(divide="ignore"):
        geometric = np.exp(np.log(energies).mean(axis=0))
    saturation = 2.0 * float(series.reference_scale().sum())
    return GrowthCurves(
        t=t,
        energy=geometric[None, :],
        saturation=np.array([saturation]),
        labels=[r"$E_\Delta$"],
        members=energies,
        source=source,
    )


def member_energies(series: YSeries) -> np.ndarray:
    """Each member's total difference energy on the frames, ``(n, n_t)``."""
    weights = series.y_weights
    out = []
    for index in range(series.n_members):
        records, rows = series.source(index, "e_x")
        e = np.asarray(records["e_x"][rows], dtype=np.float64)
        out.append(np.einsum("j,tcjk->t", weights, e))
    return np.stack(out)


@dataclass(frozen=True)
class GrowthPhases:
    r"""The two phases a growth curve is marked by, by stated criteria.

    ``exponential`` and ``decorrelation`` are ``(t_start, t_stop,
    rate)`` in outer time, or ``None``: the longest window over which
    `$\gamma$` -- respectively `$\mathrm{d}f/\mathrm{d}t$`, below
    :data:`GROWTH_R_MAX` and after the exponential window -- stays
    within :data:`GROWTH_TOLERANCE` of its own mean
    (:func:`~dnsjax.analysis.twin.growth.longest_window`).  The rate of
    the first is the exponential's `$\gamma_0$`, of the second the
    decorrelation rate `$\nu$`.
    """

    exponential: tuple[float, float, float] | None
    decorrelation: tuple[float, float, float] | None


def growth_phases(
    t: np.ndarray, energy: np.ndarray, saturation: float
) -> GrowthPhases:
    """One curve's exponential and decorrelation phases.

    The windows :class:`GrowthPhases` describes, by its criteria.
    """
    r = energy / saturation
    gamma = log_rate(t, energy)
    start, stop, gamma0 = longest_window(
        gamma, GROWTH_TOLERANCE, valid=(gamma > 0.0) & (r < 0.5)
    )
    exponential = None
    if stop > start:
        exponential = (float(t[start]), float(t[stop - 1]), gamma0)
    f = np.where(r <= GROWTH_R_MAX, bound_free(r), np.nan)
    df = np.gradient(f, t)
    after = t > (exponential[1] if exponential else -np.inf)
    a, b, nu = longest_window(
        df, GROWTH_TOLERANCE, valid=after & (r <= GROWTH_R_MAX) & (df > 0.0)
    )
    decorrelation = None
    if b > a:
        decorrelation = (float(t[a]), float(t[b - 1]), nu)
    return GrowthPhases(exponential, decorrelation)


#: Pale tints that mark the two phases on a growth figure's time axes,
#: behind the data: the exponential phase and the decorrelation phase.
_PHASE_TINTS: tuple[str, str] = ("#dbe8f8", "#fbe0d2")


def _shade_phases(ax, phases: GrowthPhases, units: Units) -> None:
    """Tint the two phases' windows on a time axis, behind everything."""
    for window, tint in zip(
        (phases.exponential, phases.decorrelation), _PHASE_TINTS, strict=True
    ):
        if window is not None:
            ax.axvspan(
                units.plotted_time(window[0]),
                units.plotted_time(window[1]),
                color=tint,
                zorder=0,
                linewidth=0,
            )


def _time_axis(ax, t_plotted: np.ndarray, units: Units, *, top: bool) -> None:
    """Grid, limits and (on the top row) the outer-time twin of a time axis."""
    ax.set_xlim(float(t_plotted[0]), float(t_plotted[-1]))
    ax.grid(True, color="0.92", linewidth=0.5)
    if top and units.wall:
        factor = units.re_tau**2 / units.re
        ax.secondary_xaxis(
            "top", functions=(lambda v: v / factor, lambda v: v * factor)
        ).set_xlabel(r"$t\,U_\mathrm{cl}/h$")


def _slope_key(ax, r_end: float, g_end: float) -> None:
    r"""A key of algebraic slopes on a log-log `$\gamma$`-`$R$` diagram.

    Grey segments ending at one point and rising to the left, of slope
    `$-1/\alpha$` for `$\alpha$` = 1, 2 and 10, each labelled at its far
    end: an algebraic phase `$E \propto (t - t_0)^\alpha$` runs parallel
    to its own (:mod:`dnsjax.analysis.twin.growth`), an exponential to
    none of them.  Placed by the caller where the data are not.
    """
    span = 1.0  # decades of R
    for alpha in (1.0, 2.0, 10.0):
        rr = np.array([r_end / 10.0**span, r_end])
        gg = g_end * (rr / r_end) ** (-1.0 / alpha)
        ax.plot(rr, gg, color="0.5", linewidth=0.9, zorder=1)
        ax.annotate(
            rf"$\alpha = {alpha:g}$",
            (rr[0], gg[0]),
            textcoords="offset points",
            xytext=(-2, 0),
            fontsize="xx-small",
            color="0.3",
            ha="right",
            va="center",
        )
    # One legend entry names the family, so no note crowds the key.
    ax.plot(
        [],
        [],
        color="0.5",
        linewidth=0.9,
        label=r"algebraic, $(t - t_0)^\alpha$: slope $-1/\alpha$",
    )


def _saturation_time(
    t: np.ndarray, r: np.ndarray, level: float = 0.98
) -> float:
    """When *r* first reaches *level*, or the end of the record."""
    hit = np.flatnonzero(r >= level)
    return float(t[hit[0]]) if hit.size else float(t[-1])


def growth_summary_figure(
    series: YSeries,
    curves: GrowthCurves,
    phases: GrowthPhases,
    options: MapOptions,
    style: PlotStyle,
):
    r"""Which growth law holds when, read off four sets of axes.

    Each law is a straight line on one of them (module docstring,
    "Growth laws"), and the two marked phases are tinted alike on every
    time axis -- the exponential blue, constant-rate decorrelation
    orange:

    (a) `$\ln R$` against `$t$`: an exponential is a line, fitted on
        its window and continued both ways, so the departure from it
        shows;
    (b) `$\ln\gamma$` against `$\ln R$`: an exponential is flat,
        saturation alone bends the curve only near `$R = 1$` (the
        logistic at the exponential's rate), an algebraic law runs
        parallel to one of the keyed slopes `$-1/\alpha$`, and
        constant-rate decorrelation follows `$\nu(1 - R)/R$`;
    (c) `$-\ln(1 - R)$` against `$t$`: constant-rate decorrelation is a
        line of slope `$\nu$`, where the logistic at the exponential's
        rate, through the same half-saturation time, would end at slope
        `$\gamma_0$`;
    (d) `$R$` against `$t$`, with the tangent at the inflection and the
        logistic of the same peak slope: the stretch that looks linear
        is that inflection, as narrow as the logistic's, at an
        `$O(1)$` fraction of saturation.

    The time axes stop a fifth past the time `$R$` reaches 0.98.
    """
    units = options.units
    t = curves.t
    tp = units.plotted_time(t)
    energy = curves.energy[0]
    sat = float(curves.saturation[0])
    r = energy / sat
    gamma = log_rate(t, energy)
    f = np.where(r <= GROWTH_R_MAX, bound_free(r), np.nan)
    t_end = min(float(t[-1]), 1.15 * _saturation_time(t, r, GROWTH_R_MAX))
    tp_end = units.plotted_time(t_end)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(style.width * 1.15, style.width * 0.95),
        layout="constrained",
    )
    (ax_e, ax_g), (ax_f, ax_r) = axes
    exp_w, dec_w = phases.exponential, phases.decorrelation
    blue, vermillion = _TRACK_SERIES[0], _TRACK_SERIES[3]

    def time_axes(ax, top: bool) -> None:
        _shade_phases(ax, phases, units)
        ax.set_xlim(0.0 if tp[0] <= 0 else float(tp[0]), tp_end)
        ax.grid(True, color="0.92", linewidth=0.5)
        ax.set_xlabel(units.t_label)
        if top and units.wall:
            factor = units.re_tau**2 / units.re
            ax.secondary_xaxis(
                "top", functions=(lambda v: v / factor, lambda v: v * factor)
            ).set_xlabel(r"$t\,U_\mathrm{cl}/h$")

    # (a) ln R against t.
    time_axes(ax_e, top=True)
    if curves.members is not None:
        for member in curves.members:
            ax_e.plot(tp, member / sat, color="0.82", linewidth=0.4)
    ax_e.plot(
        tp,
        r,
        color="black",
        linewidth=1.4,
        label=r"$R$, member geometric mean",
    )
    if exp_w is not None:
        sel = (t >= exp_w[0]) & (t <= exp_w[1])
        slope, intercept = np.polyfit(t[sel], np.log(r[sel]), 1)
        span = exp_w[1] - exp_w[0]
        tt = np.linspace(max(t[0], exp_w[0] - span), exp_w[1] + 2.5 * span, 50)
        ax_e.plot(
            units.plotted_time(tt),
            np.exp(intercept + slope * tt),
            color=blue,
            linestyle=(0, (5, 2)),
            linewidth=1.3,
            label=(
                rf"exponential, $\gamma_0 = {exp_w[2]:.3g}\,"
                rf"U_\mathrm{{cl}}/h$"
            ),
        )
    ax_e.set_yscale("log")
    ax_e.set_ylim(r[r > 0].min() * 0.5, 2.0)
    ax_e.set_ylabel(r"$R = E_\Delta/E_\mathrm{sat}$")
    ax_e.legend(fontsize="x-small", loc="lower right", frameon=False)
    for window, colour, name in (
        (exp_w, blue, "exponential"),
        (dec_w, vermillion, "decorrelation"),
    ):
        if window is not None:
            ax_e.annotate(
                name,
                (units.plotted_time(0.5 * (window[0] + window[1])), 1.0),
                xycoords=("data", "axes fraction"),
                textcoords="offset points",
                xytext=(0, -3),
                ha="center",
                va="top",
                fontsize="xx-small",
                color=colour,
            )

    # (b) ln gamma against ln R.
    ok = (gamma > 0.0) & (r > 0.0) & np.isfinite(gamma)
    peak = int(np.nanargmax(np.where(ok, gamma, -np.inf)))
    path = ok & (np.arange(t.size) >= peak)
    r_lo = float(r[path].min())
    g_hi = float(gamma[path].max())
    g_lo = max(float(gamma[path & (r < 0.99)].min()), 3e-3)
    ax_g.set_xscale("log")
    ax_g.set_yscale("log")
    ax_g.set_xlim(r_lo * 0.5, 1.5)
    ax_g.set_ylim(g_lo * 0.6, g_hi * 1.8)
    ax_g.plot(
        r[path],
        gamma[path],
        color="black",
        linewidth=1.4,
        zorder=3,
        label=r"$\gamma(R)$",
    )
    for window, colour in ((exp_w, blue), (dec_w, vermillion)):
        if window is not None:
            sel = path & (t >= window[0]) & (t <= window[1])
            ax_g.plot(
                r[sel],
                gamma[sel],
                color=colour,
                linewidth=4.0,
                alpha=0.45,
                zorder=2,
                solid_capstyle="round",
            )
    rr = np.logspace(math.log10(r_lo * 0.5), math.log10(1.5), 300)
    if exp_w is not None:
        ax_g.plot(
            rr,
            logistic_rate(rr, exp_w[2]),
            color="black",
            linewidth=0.9,
            linestyle=(0, (5, 2)),
            label=r"saturation alone, $\gamma_0(1 - R)$",
        )
    if dec_w is not None:
        big = rr >= 0.03
        ax_g.plot(
            rr[big],
            decorrelation_rate(rr[big], dec_w[2]),
            color=vermillion,
            linewidth=1.0,
            linestyle=(0, (1.5, 1.2)),
            label=r"constant-rate decorrelation, $\nu(1 - R)/R$",
        )
    y_lo, y_hi = math.log10(g_lo * 0.6), math.log10(g_hi * 1.8)
    _slope_key(
        ax_g,
        r_end=10.0 ** (math.log10(r_lo) + 1.8),
        g_end=10.0 ** (y_lo + 0.1 * (y_hi - y_lo)),
    )
    for stamp in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0):
        i = int(np.argmin(np.abs(t - stamp)))
        if path[i] and abs(t[i] - stamp) < 1.0 and r[i] < 1.0:
            ax_g.plot(
                r[i], gamma[i], "o", color="black", markersize=2.5, zorder=4
            )
            ax_g.annotate(
                rf"$t = {stamp:g}$",
                (r[i], gamma[i]),
                textcoords="offset points",
                xytext=(-4, -2),
                ha="right",
                va="top",
                fontsize="xx-small",
            )
    ax_g.set_xlabel(r"$R$")
    ax_g.set_ylabel(
        r"$\gamma = \mathrm{d}\ln E_\Delta/\mathrm{d}t$"
        r" ($U_\mathrm{cl}/h$)"
    )
    ax_g.grid(True, color="0.92", linewidth=0.5, which="both")
    ax_g.legend(
        fontsize="x-small",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncols=2,
        frameon=False,
    )

    # (c) -ln(1 - R) against t.
    time_axes(ax_f, top=False)
    ax_f.plot(tp, f, color="black", linewidth=1.4, label=r"$-\ln(1 - R)$")
    if dec_w is not None:
        sel = (t >= dec_w[0]) & (t <= dec_w[1]) & np.isfinite(f)
        slope, intercept = np.polyfit(t[sel], f[sel], 1)
        span = dec_w[1] - dec_w[0]
        tt = np.linspace(dec_w[0] - 0.7 * span, dec_w[1] + 0.5 * span, 50)
        ax_f.plot(
            units.plotted_time(tt),
            intercept + slope * tt,
            color=vermillion,
            linestyle=(0, (5, 2)),
            linewidth=1.3,
            label=rf"constant rate, $\nu = {dec_w[2]:.3g}\,U_\mathrm{{cl}}/h$",
        )
    if exp_w is not None:
        half = np.flatnonzero(r >= 0.5)
        if half.size:
            t_half = float(t[half[0]])
            r_log = 1.0 / (1.0 + np.exp(-exp_w[2] * (t - t_half)))
            f_log = np.where(r_log <= GROWTH_R_MAX, bound_free(r_log), np.nan)
            ax_f.plot(
                tp,
                f_log,
                color="black",
                linewidth=0.9,
                linestyle=(0, (1.5, 1.5)),
                label=r"logistic at $\gamma_0$, same $t$ at $R = 1/2$",
            )
    ax_f.set_ylim(0.0, float(bound_free(np.array(GROWTH_R_MAX))) * 1.05)
    ax_f.set_ylabel(r"$f = -\ln(1 - R) = -\ln C$")
    ax_f.legend(fontsize="x-small", loc="upper left", frameon=False)

    # (d) R against t: the inflection, and the logistic of its slope.
    time_axes(ax_r, top=False)
    ax_r.plot(tp, r, color="black", linewidth=1.4, label=r"$R$")
    dr = np.gradient(r, t)
    k = int(np.nanargmax(np.where(r < 0.9, dr, -np.inf)))
    peak_slope = float(dr[k])
    if peak_slope > 0.0:
        rate = 4.0 * peak_slope
        shift = math.log(1.0 / min(max(r[k], 1e-9), 1.0 - 1e-9) - 1.0)
        ax_r.plot(
            tp,
            1.0 / (1.0 + np.exp(-rate * (t - t[k]) + shift)),
            color="black",
            linewidth=0.9,
            linestyle=(0, (1.5, 1.5)),
            label="logistic with that peak slope",
        )
        width = 1.0 / rate
        tt = np.linspace(t[k] - 2.0 * width, t[k] + 2.0 * width, 20)
        ax_r.plot(
            units.plotted_time(tt),
            r[k] + peak_slope * (tt - t[k]),
            color=blue,
            linestyle=(0, (5, 2)),
            linewidth=1.2,
            label="tangent at the inflection",
        )
        ax_r.plot(
            units.plotted_time(t[k]),
            r[k],
            "o",
            color=blue,
            markersize=3.5,
            zorder=4,
        )
    ax_r.set_ylim(0.0, 1.08)
    ax_r.set_ylabel(r"$R$")
    ax_r.legend(fontsize="x-small", loc="lower right", frameon=False)
    for ax, letter in zip(axes.flat, "abcd", strict=True):
        ax.set_title(f"({letter})", loc="left", fontsize="small")
    origin = (
        "twin.dat" if curves.source == "twin.dat" else "the spectra stream"
    )
    fig.suptitle(
        "Growth of the difference energy: exponential (blue), "
        "constant-rate decorrelation (orange)\n"
        + _spacetime_suptitle(series, units)
        + f", from {origin}",
        fontsize="small",
    )
    return fig


def _band_colours(curves: GrowthCurves) -> list:
    """One colour per curve: lightness by its scale, black for `$k = 0$`."""
    cmap = plt.get_cmap(_BAND_CMAP)
    coords = curves.coords
    finite = coords[np.isfinite(coords)]
    lo, hi = np.log(finite.min()), np.log(finite.max())
    out = []
    for c in coords:
        if not np.isfinite(c):
            out.append("black")
            continue
        x = 0.5 if hi == lo else (np.log(c) - lo) / (hi - lo)
        out.append(
            cmap(_BAND_RANGE[0] + x * (_BAND_RANGE[1] - _BAND_RANGE[0]))
        )
    return out


def growth_bands_figure(
    series: YSeries,
    curves: GrowthCurves,
    title: str,
    options: MapOptions,
    style: PlotStyle,
    *,
    rate_panels: tuple[tuple[str, tuple[str, ...]], ...],
):
    r"""The growth laws of a family of bands, and what feeds each.

    The `$\gamma$`-`$R$` diagram and `$-\ln(1 - R)$` per band, as on
    the summary figure, and two of the band budget's groups per unit
    band energy (*rate_panels*: a title and the groups it adds) --
    the same-`$k$` and the cross-scale terms for a wavelength band, the
    production and the transport for a wall distance.  Lightness is
    the band's scale; the `$k = 0$` band is black.
    """
    units = options.units
    t = curves.t
    tp = units.plotted_time(t)
    colours = _band_colours(curves)
    n_rows = 2 if curves.rates else 1
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(style.width * 1.15, style.width * 0.48 * n_rows),
        layout="constrained",
        squeeze=False,
    )
    ax_g, ax_f = axes[0]
    r_all, gamma_all = curves.r, curves.gamma
    for index, label in enumerate(curves.labels):
        r, gamma = r_all[index], gamma_all[index]
        ok = (gamma > 0.0) & (r > 0.0)
        peak = int(np.nanargmax(np.where(ok, gamma, -np.inf)))
        path = ok & (np.arange(t.size) >= peak)
        dash = (0, (4, 2)) if not np.isfinite(curves.coords[index]) else "-"
        ax_g.plot(
            r[path],
            gamma[path],
            color=colours[index],
            linewidth=1.1,
            linestyle=dash,
            label=label,
        )
        ax_f.plot(
            tp,
            curves.f[index],
            color=colours[index],
            linewidth=1.1,
            linestyle=dash,
        )
    g0 = float(
        np.nanmedian(
            np.nanmax(np.where(gamma_all > 0, gamma_all, np.nan), axis=1)
        )
    )
    rr = np.logspace(-6, 0, 200)
    ax_g.plot(
        rr,
        logistic_rate(rr, g0),
        color="black",
        linewidth=0.8,
        linestyle=(0, (5, 2)),
        label=r"$\gamma_0(1 - R)$",
    )
    ax_g.set_xscale("log")
    ax_g.set_yscale("log")
    r_lo = float(np.nanmin(np.where(r_all > 0, r_all, np.nan)))
    ax_g.set_xlim(max(r_lo, 1e-8) * 0.5, 1.5)
    ax_g.set_ylim(3e-3, 1.0)
    _slope_key(
        ax_g, r_end=10.0 ** (math.log10(max(r_lo, 1e-8)) + 3.0), g_end=5e-3
    )
    ax_g.set_xlabel(r"$R$")
    ax_g.set_ylabel(r"$\gamma$ ($U_\mathrm{cl}/h$)")
    ax_g.grid(True, color="0.92", linewidth=0.5, which="both")
    ax_f.set_ylim(0.0, float(bound_free(np.array(GROWTH_R_MAX))) * 1.05)
    ax_f.set_ylabel(r"$-\ln(1 - R)$")
    ax_f.set_xlabel(units.t_label)
    _time_axis(ax_f, tp, units, top=True)
    if curves.rates:
        for ax, (heading, groups) in zip(axes[1], rate_panels, strict=True):
            for index in range(len(curves.labels)):
                total = sum(curves.rates[g][index] for g in groups)
                dash = (
                    (0, (4, 2))
                    if not np.isfinite(curves.coords[index])
                    else "-"
                )
                ax.plot(
                    tp,
                    total,
                    color=colours[index],
                    linewidth=1.0,
                    linestyle=dash,
                )
            ax.axhline(0.0, color="0.6", linewidth=0.6)
            ax.set_ylabel(heading + r" ($U_\mathrm{cl}/h$)")
            ax.set_xlabel(units.t_label)
            _time_axis(ax, tp, units, top=False)
    fig.legend(
        *ax_g.get_legend_handles_labels(),
        loc="outside lower center",
        ncols=5,
        fontsize="x-small",
        frameon=False,
    )
    fig.suptitle(
        title + "\n" + _spacetime_suptitle(series, units), fontsize="small"
    )
    return fig


def growth_ssp_figure(
    series: YSeries,
    curves: GrowthCurves,
    options: MapOptions,
    style: PlotStyle,
):
    r"""The self-sustaining process's three parts, as growth curves.

    Their shares of the difference energy against the reference's
    (dashed), their `$\gamma$`-`$R$` diagrams, their `$-\ln(1 - R)$`
    with each one's constant-rate window fitted, and -- where the
    budget was read -- the `$k_x = 0$` plane's budget per unit energy:
    the lift-up that builds it, the transfer out of it (breakdown) and
    the rest (module docstring, "Growth laws").
    """
    units = options.units
    t = curves.t
    tp = units.plotted_time(t)
    colours = _TRACK_SERIES[:3]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(style.width * 1.15, style.width * 0.95),
        layout="constrained",
    )
    (ax_s, ax_g), (ax_f, ax_b) = axes
    total = curves.energy.sum(axis=0)
    shares = curves.saturation / curves.saturation.sum()
    for index, label in enumerate(curves.labels):
        ax_s.plot(
            tp,
            curves.energy[index] / total,
            color=colours[index],
            linestyle=_TRACK_DASHES[index],
            linewidth=1.2,
            label=label,
        )
        ax_s.axhline(
            shares[index],
            color=colours[index],
            linewidth=0.8,
            linestyle=(0, (1, 2)),
        )
    ax_s.set_yscale("log")
    ax_s.set_ylabel(r"share of $E_\Delta$ (dotted: reference)")
    ax_s.set_xlabel(units.t_label)
    _time_axis(ax_s, tp, units, top=True)
    ax_s.legend(
        fontsize="x-small",
        loc="center right",
        bbox_to_anchor=(1.0, 0.42),
        frameon=False,
    )
    r_all, gamma_all, f_all = curves.r, curves.gamma, curves.f
    for index in range(len(curves.labels)):
        r, gamma = r_all[index], gamma_all[index]
        ok = (gamma > 0.0) & (r > 0.0)
        peak = int(np.nanargmax(np.where(ok, gamma, -np.inf)))
        path = ok & (np.arange(t.size) >= peak)
        ax_g.plot(
            r[path],
            gamma[path],
            color=colours[index],
            linestyle=_TRACK_DASHES[index],
            linewidth=1.2,
        )
        ax_f.plot(
            tp,
            f_all[index],
            color=colours[index],
            linestyle=_TRACK_DASHES[index],
            linewidth=1.2,
        )
        phases = growth_phases(
            t, curves.energy[index], float(curves.saturation[index])
        )
        if phases.decorrelation is not None:
            a, b, nu = phases.decorrelation
            sel = (t >= a) & (t <= b) & np.isfinite(f_all[index])
            if sel.sum() >= 2:
                slope, intercept = np.polyfit(t[sel], f_all[index][sel], 1)
                ax_f.plot(
                    units.plotted_time(t[sel]),
                    intercept + slope * t[sel],
                    color=colours[index],
                    linewidth=3.0,
                    alpha=0.35,
                    label=rf"$\nu = {nu:.3g}$",
                )
    g0 = float(
        np.nanmedian(
            np.nanmax(np.where(gamma_all > 0, gamma_all, np.nan), axis=1)
        )
    )
    rr = np.logspace(-6, 0, 200)
    ax_g.plot(
        rr,
        logistic_rate(rr, g0),
        color="black",
        linewidth=0.8,
        linestyle=(0, (5, 2)),
        label=r"$\gamma_0(1 - R)$",
    )
    ax_g.set_xscale("log")
    ax_g.set_yscale("log")
    r_lo = float(np.nanmin(np.where(r_all > 0, r_all, np.nan)))
    ax_g.set_xlim(max(r_lo, 1e-8) * 0.5, 1.5)
    ax_g.set_ylim(3e-3, 1.0)
    _slope_key(
        ax_g, r_end=10.0 ** (math.log10(max(r_lo, 1e-8)) + 3.0), g_end=5e-3
    )
    ax_g.set_xlabel(r"$R$")
    ax_g.set_ylabel(r"$\gamma$ ($U_\mathrm{cl}/h$)")
    ax_g.grid(True, color="0.92", linewidth=0.5, which="both")
    ax_g.legend(fontsize="x-small", loc="upper right", frameon=False)
    ax_f.set_ylim(0.0, float(bound_free(np.array(GROWTH_R_MAX))) * 1.05)
    ax_f.set_ylabel(r"$-\ln(1 - R)$")
    ax_f.set_xlabel(units.t_label)
    ax_f.legend(
        fontsize="x-small",
        loc="upper left",
        frameon=False,
        title=r"decorrelation rate ($U_\mathrm{cl}/h$)",
        title_fontsize="x-small",
    )
    _time_axis(ax_f, tp, units, top=False)
    if curves.rates:
        terms = (
            ("prod_mean", r"lift-up $\mathcal{P}^{\mathbf{U}}_\Delta$"),
            ("prod_fluct", r"$\mathcal{P}^{\tilde{\mathbf{u}}}_\Delta$"),
            (
                "transfer",
                r"transfer, $-\mathcal{T}^{\mathbf{u}}_{E_\Delta}"
                r" - \mathcal{T}^{\Delta\mathbf{u}}_{E_\Delta}$",
            ),
            ("diss", r"$-\mathcal{D}_\Delta$"),
        )
        slots = {"prod_mean": 0, "prod_fluct": 1, "transfer": 5, "diss": 2}
        for key, label in terms:
            if key == "transfer":
                values = curves.rates["tr_ref"][0] + curves.rates["tr_self"][0]
            else:
                values = curves.rates[key][0]
            ax_b.plot(
                tp,
                values,
                color=_TERM_SERIES[slots[key]],
                linestyle=_TERM_DASHES[slots[key]],
                linewidth=1.2,
                label=label,
            )
        ax_b.axhline(0.0, color="0.6", linewidth=0.6)
        ax_b.set_ylabel(
            r"$k_x = 0$ plane, per unit energy ($U_\mathrm{cl}/h$)"
        )
        ax_b.set_xlabel(units.t_label)
        ax_b.legend(fontsize="x-small", loc="upper right", frameon=False)
        _time_axis(ax_b, tp, units, top=False)
    else:
        ax_b.set_axis_off()
    for ax, letter in zip(axes.flat, "abcd", strict=True):
        ax.set_title(f"({letter})", loc="left", fontsize="small")
    fig.suptitle(
        "Streaks, rolls and waves of the difference field\n"
        + _spacetime_suptitle(series, units),
        fontsize="small",
    )
    return fig


def write_growth_npz(
    path: Path,
    series: YSeries,
    curves: GrowthCurves,
    options: MapOptions,
    phases: list[GrowthPhases],
) -> Path:
    """Dump a growth figure's curves, their derived laws and the phases."""

    def window(w):
        return np.full(3, np.nan) if w is None else np.asarray(w, dtype=float)

    payload = {
        "t": curves.t,
        "t_plotted": options.units.plotted_time(curves.t),
        "energy": curves.energy,
        "saturation": curves.saturation,
        "r": curves.r,
        "gamma": curves.gamma,
        "f": curves.f,
        "log_slope": log_slope(curves.r, curves.gamma),
        "labels": np.asarray(curves.labels),
        "coords": (
            np.full(len(curves.labels), np.nan)
            if curves.coords is None
            else curves.coords
        ),
        "exponential": np.stack([window(p.exponential) for p in phases]),
        "decorrelation": np.stack([window(p.decorrelation) for p in phases]),
        "windows": "(t_start, t_stop, rate) in outer time; rate in U_cl/h",
        "tolerance": GROWTH_TOLERANCE,
        "r_max": GROWTH_R_MAX,
        "rate_units": "U_cl/h (outer time)",
        "source": curves.source,
        "n_members": series.n_members,
        "members": np.asarray([str(m.path) for m in series.members]),
        "meta_json": json.dumps(_stream_meta(series)),
        **_units_payload(options),
    }
    if curves.members is not None:
        payload["member_energy"] = curves.members
    if curves.rates:
        for group, values in curves.rates.items():
            payload[f"rate_{group}"] = values
    np.savez_compressed(path, **payload)
    return path


def render_growth(
    spectra: YSeries,
    budget: YSeries | None,
    spec: SeriesSpec,
    tag: str,
    out_dir: Path,
    *,
    options: MapOptions,
    style: PlotStyle,
    fmt: str = "png",
    quiet: bool = False,
) -> list[Path]:
    """Render one growth-law figure and its ``.npz`` (module docstring)."""
    kind = spec.marginal
    if kind == "global":
        curves = growth_global(spectra, options)
    elif kind == "ssp":
        curves = growth_ssp(spectra, options, budget)
    elif kind == "y":
        curves = growth_rows(spectra, options, budget)
    else:
        curves = growth_bands(spectra, kind, options, budget)
    phases = [
        growth_phases(curves.t, curves.energy[i], float(curves.saturation[i]))
        for i in range(curves.energy.shape[0])
    ]
    target = out_dir / tag
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if curves.t.size >= 5:
        if kind == "global":
            fig = growth_summary_figure(
                spectra, curves, phases[0], options, style
            )
        elif kind == "ssp":
            fig = growth_ssp_figure(spectra, curves, options, style)
        elif kind == "y":
            fig = growth_bands_figure(
                spectra,
                curves,
                "Growth at fixed wall distances",
                options,
                style,
                rate_panels=(
                    (r"production$/E$", ("prod",)),
                    (r"transport$/E$", ("transport",)),
                ),
            )
        else:
            axis = MARGINALS[kind][0]
            fig = growth_bands_figure(
                spectra,
                curves,
                rf"Growth of the $\lambda_{axis}$ bands",
                options,
                style,
                rate_panels=(
                    (
                        r"same-$k$: $\mathcal{P}^{\mathbf{U}}_\Delta/E$",
                        ("prod_mean",),
                    ),
                    (
                        r"cross-scale: $(\mathcal{P}^{\tilde{\mathbf{u}}}"
                        r"_\Delta - \mathcal{T}^{\mathbf{u}}"
                        r" - \mathcal{T}^{\Delta\mathbf{u}})/E$",
                        ("prod_fluct", "tr_ref", "tr_self"),
                    ),
                ),
            )
        path = target / f"{tag}.{fmt}"
        fig.savefig(path, dpi=style.dpi)
        plt.close(fig)
        written.append(path)
    written.append(
        write_growth_npz(
            target / f"{tag}.npz", spectra, curves, options, phases
        )
    )
    if not quiet:
        for path in written:
            print(f"  {path.name}", flush=True)
    return written


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


@dataclass(frozen=True)
class SeriesSpec:
    """What one series tag names.

    *base* is a stored spectra prefix (``e`` / ``r``), one of the two
    virtual decorrelations, the virtual :data:`SHAPE`, or empty for a
    budget series, whose panels are its terms.  *marginal* is empty
    exactly when `$k$` has been summed away, which is what makes such
    a series marginal-free.
    """

    stem: str  # which stream it reads
    base: str  # its field, or "" for the budget
    marginal: str  # "x" / "z" / "x0", or "" when k is summed
    family: str  # :data:`MAP` or :data:`SPACETIME`


def available_series(
    spectra: YSeries | None, budget: YSeries | None
) -> dict[str, SeriesSpec]:
    r"""``tag -> SeriesSpec`` for what a member set offers.

    One `$(\lambda, y)$` tag per stored prefix and **drawable** stored
    marginal -- :data:`MARGINALS` intersected with what the sidecar
    says the stream carries, so a default run offers no ``_x0`` tag
    because it has no such field, and ``xz00`` never becomes a tag at
    all -- plus a shape tag and a decorrelation tag per true marginal,
    and a spacetime tag per `$k$`-summable quantity.

    Neither decorrelation is offered for the `$k_x = 0$` plane, for
    the reason its panels are already left absolute: it is a slice of
    the mode plane and its divisor would be built from the whole.
    Both need the stream's reference half; a shape map needs only the
    difference spectra it redraws.  Which tags come out
    normalised by `$E^{\mathrm{ref}}$` is :func:`normalises` /
    :func:`spacetime_normalises`, not a tag of its own, and which are
    rendered *by default* is :func:`default_series`.
    """
    out: dict[str, SeriesSpec] = {}
    if spectra is not None:
        for prefix in spectra.prefixes:
            for marginal in spectra.suffixes:
                if marginal in MARGINALS:
                    out[f"spectra_{prefix}_{marginal}"] = SeriesSpec(
                        "twin_yspectra", prefix, marginal, MAP
                    )
            out[f"spacetime_{prefix}"] = SeriesSpec(
                "twin_yspectra", prefix, "", SPACETIME
            )
            if "x0" in spectra.suffixes:
                out[f"spacetime_{prefix}_x0"] = SeriesSpec(
                    "twin_yspectra", prefix, "x0", SPACETIME
                )
        for marginal in ("x", "z"):
            out[f"spectra_{SHAPE}_{marginal}"] = SeriesSpec(
                "twin_yspectra", SHAPE, marginal, MAP
            )
        if "r" in spectra.prefixes:
            for base in (DECORR_K, DECORR):
                for marginal in ("x", "z"):
                    out[f"spectra_{base}_{marginal}"] = SeriesSpec(
                        "twin_yspectra", base, marginal, MAP
                    )
            out[f"spacetime_{DECORR_K}"] = SeriesSpec(
                "twin_yspectra", DECORR_K, "", SPACETIME
            )
        for base in ("e", SHAPE):
            out[f"history_{base}"] = SeriesSpec(
                "twin_yspectra", base, "", HISTORY
            )
    if budget is not None:
        for marginal in budget.suffixes:
            if marginal in MARGINALS:
                out[f"budget_{marginal}"] = SeriesSpec(
                    "twin_ybudget", "", marginal, MAP
                )
        out["spacetime_budget"] = SeriesSpec("twin_ybudget", "", "", SPACETIME)
        out["history_budget"] = SeriesSpec("twin_ybudget", "", "", HISTORY)
    if spectra is not None and "r" in spectra.prefixes:
        for marginal in ("x", "z"):
            if marginal in spectra.suffixes:
                out[f"front_{marginal}"] = SeriesSpec(
                    "twin_yspectra", "e", marginal, FRONT
                )
    if spectra is not None and "r" in spectra.prefixes:
        for kind in ("global", "x", "z", "y", "ssp"):
            if kind in ("x", "z") and kind not in spectra.suffixes:
                continue
            out[f"growth_{kind}"] = SeriesSpec(
                "twin_yspectra", "e", kind, GROWTH
            )
    if spectra is not None and budget is not None:
        out["moments_y"] = SeriesSpec("twin_ybudget", "", "", MOMENT_BUDGET)
        for marginal in ("x", "z"):
            if marginal in spectra.suffixes and marginal in budget.suffixes:
                out[f"moments_{marginal}"] = SeriesSpec(
                    "twin_ybudget", "", marginal, MOMENT_BUDGET
                )
    return out


def default_series(
    registry: dict[str, SeriesSpec],
    *,
    x0: bool = False,
    budget: bool = True,
    reference: bool = False,
    decorr: bool = False,
    decorr_k: bool = False,
    spacetime: bool = False,
    history: bool = True,
    moment_budget: bool = False,
    front: bool = False,
    growth: bool = False,
) -> list[str]:
    r"""The tags rendered when ``--series`` names none.

    Three families are drawn unasked, all as maps on the two
    wavenumber marginals: the difference spectra, their shape maps,
    and the ``twin_ybudget`` set, which ``--no-budget`` (*budget*
    false) drops.  Five are held back, each behind its own flag rather
    than a tag the caller has to know the name of, and each of the
    five is a ``False`` here: the reference spectra are the turbulent
    flow's own, statistically the same in every frame; the
    `$k_x = 0$` plane is a slice of the mode plane rather than a
    marginal of it (and only a legacy or ``twin.x0_planes`` stream has
    one); the two decorrelations and the `$k$`-summed `$(y, t)$` maps
    are second readings of the same records, and a rendering run pays
    for each in full.  ``--series`` overrides every one of them:
    naming a tag renders it.

    The composite cases fall out of the predicate rather than being
    special-cased.  ``spacetime_decorr_k`` needs *both* ``decorr_k``
    (it is a :data:`DECORR_K` base) and ``spacetime`` (it is a
    :data:`SPACETIME` family), which is what makes `$\mathcal{R}^k$`'s
    spacetime map follow its maps; ``spacetime_r`` needs *reference*
    and ``spacetime`` the same way; and a ``spacetime_*_x0`` needs
    ``x0`` as well, its marginal being one.
    """
    held = {"r": reference, DECORR: decorr, DECORR_K: decorr_k}
    return [
        tag
        for tag, spec in registry.items()
        if (budget or spec.stem != "twin_ybudget")
        and (x0 or spec.marginal not in set(MARGINALS) - DEFAULT_MARGINALS)
        and held.get(spec.base, True)
        and (spacetime or spec.family != SPACETIME)
        and (history or spec.family != HISTORY)
        and (moment_budget or spec.family != MOMENT_BUDGET)
        and (front or spec.family != FRONT)
        and (growth or spec.family != GROWTH)
    ]


def needs_reference(series: YSeries, spec: SeriesSpec) -> bool:
    """Whether rendering *spec* touches the reference average.

    What decides whether ``main`` prints that average's report, and
    so whether a run pays for the pass at all: a budget-only,
    absolute-only or shape-only selection never builds it.  *series*
    is the spectra series, whatever stream *spec* names -- the
    reference average is one and lives there.
    """
    if spec.stem != "twin_yspectra":
        return False
    if spec.base in (DECORR, DECORR_K):
        return True
    if spec.family == HISTORY:
        return spec.base == "e" and "r" in series.prefixes
    if spec.family in (FRONT, GROWTH):
        return True
    if spec.family == SPACETIME:
        return spacetime_normalises(series, spec.base, spec.marginal)
    return normalises(series, f"{spec.base}_{spec.marginal}")


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
    frames: bool = True,
) -> list[Path]:
    """Render every frame of one series into ``out_dir/tag``.

    Filenames are ``<tag>_<index>.<fmt>`` with *index* the frame
    label of :attr:`YSeries.index`, zero-padded so a lexical sort is
    the time order.  The series is scanned once first
    (:func:`scan_panels`) for the sign check and the frozen scale, and
    its :data:`TRACKED` panels once more for their tracks and moments
    (:func:`track_peak`), which every frame draws and
    :func:`render_tracks` writes out beside the frames.

    Without *frames* (``--no-frames``) neither the scan nor the frames
    are made, only the tracks and moments and their figures: what a
    re-render of those needs, without redrawing every sample.

    The reference normalisation a spectra series may carry is a
    property of the member set rather than of a tag, so it is
    :meth:`YSeries.reference_report` and its caller's to print, once.
    """
    panels = (
        spectra_panels(prefix, marginal)
        if prefix
        else budget_panels(series, marginal)
    )
    ylim = y_limits(series, options, style.ylim)
    scales: dict = {}
    if frames:
        scales, notes = scan_panels(
            series,
            panels,
            options,
            declared=declared_signs,
            ylim=ylim,
        )
        if notes and not quiet:
            print("\n".join(notes), flush=True)
    tracked = [key for key in panels if key[0] in TRACKED]
    tracks = {
        key: track_peak(
            series,
            *key,
            options=options,
            n_levels=style.n_levels,
            ylim=ylim,
        )
        for key in tracked
    }

    target = out_dir / tag
    width = pad or len(str(int(series.index.max())))
    written: list[Path] = []
    if frames:
        target.mkdir(parents=True, exist_ok=True)
    for frame in range(series.t_rel.size if frames else 0):
        fig = panel_figure(
            series, frame, panels, options, style, scales, tracks
        )
        path = target / f"{tag}_{int(series.index[frame]):0{width}d}.{fmt}"
        fig.savefig(path, dpi=style.dpi)
        plt.close(fig)
        written.append(path)
        if not quiet:
            print(f"  {path.name}  t = {series.t_rel[frame]:g}", flush=True)
    if tracked:
        written += render_tracks(
            series,
            tag,
            tracked,
            tracks,
            out_dir,
            options=options,
            style=style,
            fmt=fmt,
            quiet=quiet,
        )
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
        "--align-atol",
        type=float,
        default=_T_ATOL,
        metavar="T",
        help="how far apart (in t) two members' samples may be and "
        "still be one frame; raise it to use members recorded on "
        "phase-displaced grids (default: exact alignment)",
    )
    p.add_argument(
        "--series",
        nargs="+",
        default=None,
        metavar="TAG",
        help="exact series tags to render, overriding every selection "
        "switch (default: the difference-spectra, shape and budget "
        "marginals present)",
    )
    p.add_argument(
        "--budget",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render the twin_ybudget series, the terms of the "
        "difference-energy balance on a 3 x 3 grid; on unless "
        "--no-budget",
    )
    p.add_argument(
        "--reference",
        action="store_true",
        help="also render the reference field's spectra (their k-summed "
        "map too, under --spacetime); off by default",
    )
    p.add_argument(
        "--x0",
        action="store_true",
        help="also render the k_x = 0 plane series, where the stream "
        "carries one (twin.x0_planes, or a pre-xz00 member)",
    )
    p.add_argument(
        "--decorr-k",
        action="store_true",
        help="also render R^k, the decorrelation over a k-summed "
        "reference (its spacetime map too, under --spacetime); off by "
        "default",
    )
    p.add_argument(
        "--decorr",
        action="store_true",
        help="also render R, the decorrelation over a k-resolved "
        "reference; off by default",
    )
    p.add_argument(
        "--spacetime",
        action="store_true",
        help="also render the k-summed (y, t) maps and their .npz, "
        "for whatever else is selected; off by default",
    )
    p.add_argument(
        "--frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw one figure per sample of the map series; --no-frames "
        "keeps only their tracks, moments and the whole-run figures",
    )
    p.add_argument(
        "--history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render the premultiplied (y, t) and (lambda, t) histories "
        "of the tracked quantities (history_e, history_s, "
        "history_budget); on unless --no-history",
    )
    p.add_argument(
        "--moment-budget",
        action="store_true",
        help="also render the budget of the difference spectrum's "
        "log-coordinate moments (moments_y, moments_x, moments_z); "
        "needs both streams; off by default",
    )
    p.add_argument(
        "--front",
        action="store_true",
        help="also render the decorrelation front, the time each "
        "(lambda, y) cell's R rises through --front-level for good "
        "(front_x, front_z); needs the reference; off by default",
    )
    p.add_argument(
        "--front-level",
        type=float,
        default=FRONT_LEVEL,
        help="the decorrelation level the front map times",
    )
    p.add_argument(
        "--cmap-front",
        default="Blues",
        help="sequential colour map of the front's times",
    )
    p.add_argument(
        "--growth",
        action="store_true",
        help="also render the growth-law figures (growth_global, the "
        "summary; growth_x / growth_z / growth_y per band; growth_ssp, "
        "streaks, rolls and waves); needs the reference; off by default",
    )
    p.add_argument(
        "--log-decades",
        type=float,
        default=LOG_DECADES,
        help="decades below the peak a spacetime map's logarithmic "
        "colour scale reaches, where the data spans that many",
    )
    p.add_argument(
        "--premultiply",
        choices=("ky", "k", "none"),
        default="k",
        help="premultiplier: both axes (log ordinate), wavenumber "
        "only, or neither; the shape maps always take both",
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
        choices=("series", "frame", "ramped"),
        default="series",
        help="colour scale frozen on the whole series, per figure, or "
        "ramped: frozen on the frames so far, so it grows with the "
        "field and ends on the series scale",
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
        help="width of a spectra figure in inches, which sets the decade "
        "length every figure shares",
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
    p.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="columns of a spectra figure, which --width fits the panel "
        "size of every figure to; a budget figure is always 3 columns "
        "of those panels, and so wider",
    )
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
        clim=args.clim,
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
                align_atol=args.align_atol,
            )
        except FileNotFoundError as exc:
            print(f"skipping {stem}: {exc}", file=sys.stderr)
            opened[stem] = None

    registry = available_series(
        opened["twin_yspectra"], opened["twin_ybudget"]
    )
    tags = args.series or default_series(
        registry,
        x0=args.x0,
        budget=args.budget,
        reference=args.reference,
        decorr=args.decorr,
        decorr_k=args.decorr_k,
        spacetime=args.spacetime,
        history=args.history,
        moment_budget=args.moment_budget,
        front=args.front,
        growth=args.growth,
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
            else "every series present is held back; add --reference / "
            "--decorr / --decorr-k / --spacetime / --x0, drop "
            f"--no-budget, or name one of: {list(registry)}"
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
        # Never silent: a shared grid shorter than the first member's
        # own says which member shortened it
        # (:meth:`YSeries.grid_report`), and a widened tolerance says
        # how far apart the frames it recovered actually pair.
        for stem, series in opened.items():
            if series is None:
                continue
            shortfall = series.grid_report()
            if shortfall is not None:
                print(shortfall, flush=True)
            if args.align_atol > _T_ATOL:
                print(
                    f"{stem}: members aligned to {args.align_atol:g} "
                    f"in t; frames pair samples up to "
                    f"{series.alignment_spread():.6g} apart on the "
                    "relative clock",
                    flush=True,
                )
        # The reference average is one pass for the whole member set,
        # so its report belongs here, not once per tag that uses it.
        spectra = opened["twin_yspectra"]
        if spectra is not None and any(
            needs_reference(spectra, registry[tag]) for tag in tags
        ):
            print("\n".join(spectra.reference_report()), flush=True)
    for tag in tags:
        spec = registry[tag]
        series = opened[spec.stem]
        if series is None:  # pragma: no cover - the registry guards this
            continue
        if not args.quiet:
            print(
                f"{tag}: {series.t_rel.size} frames, "
                f"t = {series.t_rel[0]:g}..{series.t_rel[-1]:g}",
                flush=True,
            )
        if spec.family == GROWTH:
            render_growth(
                series,
                opened["twin_ybudget"],
                spec,
                tag,
                args.out,
                options=options,
                style=style,
                fmt=args.format,
                quiet=args.quiet,
            )
            continue
        if spec.family == FRONT:
            render_front(
                series,
                spec,
                tag,
                args.out,
                options=options,
                style=style,
                level=args.front_level,
                cmap=args.cmap_front,
                fmt=args.format,
                quiet=args.quiet,
            )
            continue
        if spec.family == MOMENT_BUDGET:
            render_moment_budget(
                opened["twin_yspectra"],
                opened["twin_ybudget"],
                spec,
                tag,
                args.out,
                options=options,
                style=style,
                fmt=args.format,
                quiet=args.quiet,
            )
            continue
        if spec.family == HISTORY:
            render_history(
                series,
                spec,
                tag,
                args.out,
                options=options,
                style=style,
                decades=args.log_decades,
                declared_signs=not args.signs_from_data,
                fmt=args.format,
                quiet=args.quiet,
            )
            continue
        if spec.family == SPACETIME:
            render_spacetime(
                series,
                spec,
                tag,
                args.out,
                options=options,
                style=style,
                decades=args.log_decades,
                declared_signs=not args.signs_from_data,
                fmt=args.format,
                quiet=args.quiet,
            )
            continue
        render_series(
            series,
            tag,
            spec.base,
            spec.marginal,
            args.out,
            options=options,
            style=style,
            declared_signs=not args.signs_from_data,
            fmt=args.format,
            pad=args.pad,
            quiet=args.quiet,
            frames=args.frames,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
