r"""Twin-run difference-field diagnostics (Cartesian wall-bounded).

Online diagnostics of the difference field
`$\Delta\mathbf{u} = \mathbf{u}^{(2)} - \mathbf{u}^{(1)}$` between the
two DNS states the ``dnsjax-twin`` driver (:mod:`dnsjax.twin.driver`)
steps
in lockstep, following the methodology of Egerique-de-la-Concha &
Hwang, *J. Fluid Mech.* **1036**, A52 (2026).  Both states store the
spectral perturbation about the *same* laminar base flow, so
``state2 - state1`` is exactly the spectral `$\Delta\mathbf{u}$` --
the base flow cancels identically and every solver-measure norm
helper (:func:`~dnsjax.geometries.wall_bounded._base.get_norm2`)
applies to it unchanged.

State preconditions
-------------------
Everything here takes the two states to be what the solver produces:
divergence-free, with no-slip at both walls.  One consequence is used
throughout and is worth stating once, because it is what several terms
below are entitled to omit rather than evaluate --
`$\Delta\hat v_{00} \equiv 0$`, the mean mode's wall-normal
component.  Continuity at `$k^2 = 0$` reads
`$\partial_y \hat v_{00} = 0$` and no-slip pins
`$\hat v_{00}(\pm 1) = 0$`, so the profile is identically zero; and
it holds **exactly**, not to truncation, at every point a state can
enter: both influence-matrix paths zero that plane outright
(``cartesian._imm_iteration_vw`` stage 6 and
``_cartesian_primitive_imm._imm_iteration_vp``), every initial
condition honours it (:mod:`dnsjax.ic.random_field` sets it, the
localized rolls are mean-free, ``scripts/snapshot_perturb.py``
refuses a mean-mode profile that violates it), a wall-normal regrid is
a linear map on a profile that is zero for all `$y$`, and a Fourier
regrid preserves mode zero.

So a mean profile `$\mathbf{P} = (P_x, 0, P_z)$` advects with
`$\mathrm{i}(k_xP_x + k_zP_z)$` alone -- mode-diagonal, hence
FFT-free, and with no `$P_y\,\partial_y$` half to form.  The sites
that rely on it: ``term_b_mean`` in :func:`_twin_budget_jit`,
``mean_advect`` in :func:`_convective_sources`, the two-component
form of :func:`_driving_density`, and the mean-mode gauge argument
of :mod:`dnsjax.twin.pressure` (where
`$\Delta\hat v_{00} \equiv 0$` is what makes the fluctuating
pressure do no work at `$(0,0)$`).  A state that violated it would not
be an admissible incompressible no-slip field in the first place, and
the whole `$(y, k)$` half of this module would be wrong for it, not
just those terms.

Component decomposition
-----------------------
Fields are split by their wall-parallel Fourier support into the
three components of the reference (mean / streak / streamwise-varying
triad):

- mean `$\Delta U$`: the `$(k_z, k_x) = (0, 0)$` mode
  (``fourier.mean_mask``, one-hot by the padding-slot invariant --
  see :func:`dnsjax.operators.pad_harmonics`);
- streaks `$\Delta u_1$`: `$k_x = 0$`, `$k_z \ne 0$` (the
  streamwise-averaged fluctuation);
- streamwise-varying `$\Delta u_2$`: `$k_x \ne 0$`.

The three masks partition the whole mode grid: spectral padding
slots carry nonzero placeholder wavenumbers
(:func:`dnsjax.operators.pad_harmonics`), so `$k_x$` padding lands in
the `$\Delta u_2$` mask and `$k_z$` padding (at `$k_x = 0$`) in the
`$\Delta u_1$` mask -- both weight identically-zero state entries and
are inert.  Consequently

.. math::
    E_{\Delta U} + E_{\Delta u_1} + E_{\Delta u_2} = E_\Delta

holds to rounding (a guard in ``tests/test_twin_unit.py``), where
each energy is the volume-averaged
`$E_X = \|X\|^2 / 2$` in the solver measure (Parseval over the
wall-parallel modes with the real-FFT weight ``k_metric``, quadrature
``y_weights`` wall-normally, divided by ``derived_params.volume_fac``
-- identical to every flow's ``get_perturbation_energy``).
`$E_{\Delta u_1}$` is additionally split per velocity component
(``E_du1_x`` / ``E_du1_y`` / ``E_du1_z``): the streamwise dominance
of the streak difference field is the lift-up signature (fig. 11 of
the paper).

Budget terms
------------
:func:`twin_budget` evaluates the volume-averaged energy budget of
the three components (eqs. 2.7-2.17 of the paper): 12 production and
12 transport terms of the form

.. math::
    -\langle \mathbf{a} \cdot (\mathbf{b} \cdot \nabla)
    \mathbf{c} \rangle,

with `$(\mathbf{a}, \mathbf{b}, \mathbf{c})$` triples over the six
decomposed fields -- `$\Delta U, \Delta u_1, \Delta u_2$` (``dU`` /
``du1`` / ``du2``) and the reference's `$U^{(1)}, u_1^{(1)},
u_2^{(1)}$` (``rU`` / ``ru1`` / ``ru2``; `$U^{(1)}$` *includes the
laminar base profile*, so the terms are those of the total field) --
plus the three dissipations
`$\epsilon_{\Delta X} = -\langle \Delta X \cdot \nabla^2 \Delta X
\rangle / Re$` (the paper's eq. 2.17; see "Dissipation form" below)
and the consistency sums ``P_tot`` / ``T_tot`` / ``eps_tot``.  Column
names encode the triple, e.g. ``P_du1(du1,rU)`` is
`$-\langle \Delta u_1 \cdot (\Delta u_1 \cdot \nabla) U^{(1)}
\rangle$`.  The transport terms cancel pairwise by parts (each
advector appears symmetrically), so ``T_tot`` vanishes up to spatial
truncation; per component,
`$\partial_t E_X = P_X + T_X - \epsilon_X$` closes up to the
(pressure-projection + quadrature/FD-adjointness) truncation error
and the `$O(\Delta t^2)$` stepping error -- the guard in
``tests/test_twin_driver.py``.

Four evaluation classes, the first three FFT-free:

- **c mean** (`$\mathbf{c} \in \{U^{(1)}, \Delta U\}$`, 7 terms):
  `$(\mathbf{b}\cdot\nabla)\mathbf{c} = b_y\, \partial_y c_i(y)$`,
  so the term is a per-`$y$` Parseval cross-mean of `$(a_i, b_y)$`
  against `$\partial_y c_i$` -- no transform.
- **b mean** (`$\mathbf{b} = \Delta U$`, 2 terms): advection by a
  `$y$`-profile is diagonal in `$(k_z, k_x)$`:
  `$(i k_x b_x + i k_z b_z)\hat{c} + b_y \partial_y \hat{c}$`.
- **a mean** (`$\mathbf{a} = \Delta U$`, 6 terms): the `$(0,0)$`
  projection of the quadratic `$(\mathbf{b}\cdot\nabla)\mathbf{c}$`
  is a Parseval cross of `$\nabla\mathbf{c}$` with `$\mathbf{b}$`.
- **triple-fluctuating** (9 terms):
  `$\mathbf{q} = (\mathbf{b}\cdot\nabla)\mathbf{c}$` is formed on
  the padded physical grid (alias-free for a quadratic product),
  transformed back, and paired spectrally with
  `$\hat{\mathbf{a}}$` -- fully alias-controlled, never a third
  physical field.  Pairs are grouped by `$\mathbf{c}$` (one gradient
  set each) with the three advector fields' physical forms cached:
  69 single-field transforms per sample.  **A sample costs
  `$\sim\!0.9$` of a twin step** (both states), measured on CPU at
  plane-Couette `$48^3$` and `$64^3$` -- the same ratio at both, and
  the whole of it transform time.  So ``it_budget`` reads directly as
  a throughput tax: `$1$` nearly doubles the run, `$10$` costs
  `$\sim\!9\,\%$`.

  **69 is the floor for this pairing**, and the grouping is the
  right way round.  The nine rows use 3 distinct `$\mathbf{b}$`,
  4 distinct `$\mathbf{c}$` and **8** distinct
  `$(\mathbf{b},\mathbf{c})$` pairs -- 8, not 9, because
  ``(du2, ru2)`` serves two production rows that differ only in
  `$\mathbf{a}$`, and one `$\mathbf{q}$` covers both.  So
  `$3\times3 = 9$` advector, `$4\times9 = 36$` gradient and
  `$8\times3 = 24$` back-transforms, each of which some row needs.
  Caching the *advectors* and re-deriving the gradients is what
  makes it minimal: there are fewer distinct `$\mathbf{b}$` than
  `$\mathbf{c}$`, so the other grouping would hold 36 gradient
  fields to save nothing.  Writing `$(\mathbf{b}\cdot\nabla)
  \mathbf{c} = \nabla\cdot(\mathbf{b}\mathbf{c})$` (legal, both are
  solenoidal) trades the 36 gradient transforms for `$8\times9$`
  tensor back-transforms: 93, worse.

  The pass holds ~21 padded physical fields at once -- the 9 cached
  advectors, the 9 gradients of the current `$\mathbf{c}$` and
  `$\mathbf{q}$` -- one gradient set at a time, which an optimization
  barrier per `$\mathbf{c}$` enforces (without it XLA formed the four
  sets together).  That is a count of live fields; the program's
  measured peak is about twice it ("Memory" below).  Two knobs trade
  against footprint if it ever binds, and they act on **different**
  transients -- neither is applied by default (both cost throughput,
  and the budget is a cadenced diagnostic):

  - ``solver.rhs_transform_chunks`` bounds the *transform-stage*
    transient inside one :func:`dnsjax.fft.chunked_transform` call
    (the padded intermediates of a 9-field batch), for the same
    69 single-field transforms in more, smaller dispatches.  It
    leaves the ~21 live fields alone; measured, it takes the peak
    from 43 to 37.
  - Moving the advector transform *inside* the pair loop is what
    cuts those: the 9 cached `$\mathbf{b}$` fields become 3 live
    ones, 21 `$\to$` 15 live fields, for 84 transforms instead of
    69 (+22 %) since each of the 8 pairs then re-transforms its own
    advector.  A count off the shapes; not measured.

`$U^{(1)}$` and `$\Delta U$` are needed only as `$(3, N_y)$`
profiles in the a/b/c mean slots (no advecting-`$U^{(1)}$` term
exists: self-advection contributes no energy and is excluded from
the paper's lists), except `$\Delta U$`'s full masked field, which
`$\epsilon_{\Delta U}$` needs anyway.

Dissipation form
----------------
`$\epsilon_{\Delta X}$` is evaluated in the discrete-Laplacian
(operator) form `$-\langle \Delta X \cdot (\nabla_h^2 + D_2) \Delta
X\rangle / Re$` -- the operator the solver's implicit viscous update
actually applies -- rather than the positive-definite quadratic form
`$\langle |\nabla \Delta X|^2 \rangle / Re$` of
:func:`~dnsjax.geometries.wall_bounded._base.get_pert_enstrophy`
(which ``get_stats`` keeps).  Continuously the two coincide; the
discrete pair is not summation-by-parts in the quadrature inner
product, so they differ by
`$\Delta X^{T}(D_1^{T} W D_1 + W D_2)\,\Delta X$`.  That defect is a
truncation error *of the resolved part only*: at ``fd_order = 8`` it
is `$<10^{-4}$` for a decaying wall-normal spectrum but `$\sim 40\,\%$`
for content at half the grid scale -- and a difference field is the
adverse case, since it re-populates the grid scale as the grid
refines (measured flat at `$\sim 3\,\%$` from `$N_y = 17$` to
`$257$`, against `$6\times10^{-3} \to 5\times10^{-10}$` for a fixed
smooth field).  Only the operator form -- the one the implicit
viscous update actually applies -- therefore closes the discrete
budget `$\partial_t E_X = P_X + T_X - \epsilon_X$` against the
stepped states.  The price is positivity: unlike the quadratic form
this one is not positive-definite (the symmetric part of `$-W D_2$`
has genuinely negative eigenvalues), which is why ``get_stats``
keeps the other.

Wall-normal-resolved spectra
----------------------------
:func:`twin_yspectra` and :func:`twin_ybudget` replace the bin index
with the wavenumber itself and stop integrating over `$y$`.  The
three-bin split above *is* a three-bin partition of the
`$(k_x, k_z)$` plane, and the paper restricts it to minimal flow
units (its caveat after eq. 2.5); above that it stops resolving
anything.

The stored objects are the two marginals of the per-mode density,
plus its `$(0, 0)$` mode:

.. math::
    E_\Delta^x[\alpha](y, k_z) = \sum_{k_x} \hat{e}_\alpha , \qquad
    E_\Delta^z[\alpha](y, k_x) = \sum_{k_z} \hat{e}_\alpha , \qquad
    E_\Delta^{xz00}[\alpha](y) = \hat{e}_\alpha(y, 0, 0) ,

with `$\hat{e}_\alpha = \tfrac12 |\Delta\hat u_\alpha|^2$` per
velocity component.  **Energy first, then the sum over the other
wavenumber** -- not the energy of the averaged velocity.  Under the
``norm="forward"`` convention that sum *is* the streamwise average of
the energy, so it is the standard one-dimensional spectrum; it
closes (summing either marginal over its axis and integrating in
`$y$` returns `$E_\Delta$` exactly, where averaging first returns
only the `$k_x = 0$` content); and it *contains* the other reading,
the `$k_x = 0$` plane

.. math::
    E_\Delta^{x0}[\alpha](y, k_z) = \hat{e}_\alpha(y, 0, k_z) ,

of which `$E^{xz00}$` is the `$k_z = 0$` column.  With the full plane
the three-bin split comes back resolved in `$k_z$`:

.. math::
    E_{\Delta U} = \textstyle\int \sum_\alpha E^{x0}_\alpha(y, 0)
        = \int \sum_\alpha E^{xz00}_\alpha ,
    \quad
    E_{\Delta u_1} = \int \sum_\alpha \sum_{k_z>0} E^{x0}_\alpha ,
    \quad
    E_{\Delta u_2} = \int \sum_\alpha \sum_{k_z}
        (E^{x}_\alpha - E^{x0}_\alpha)

-- the three numbers of the old binning, per `$k_z$`.  The
`$\pm k_z$` fold this requires is not cosmetic: :func:`_fold_kz`.

Only `$E^{xz00}$` is stored unconditionally.  The plane it comes from
is **opt-in** (``twin.x0_planes``, default off): it is a third of
every record and a third of the sample's collective, and the two
marginals are what a `$(y, k)$` reading actually wants.  With it off
the three numbers still come back, because the first column of the
`$k_x$` marginal is the whole plane summed over `$k_z$`:

.. math::
    E_{\Delta u_1} = \textstyle\int \sum_\alpha
        \bigl(E^{z}_\alpha(y, 0) - E^{xz00}_\alpha\bigr) ,
    \qquad
    E_{\Delta u_2} = \int \sum_\alpha \sum_{k_x > 0} E^{z}_\alpha ,

which is what :func:`~dnsjax.analysis.twin.bin_energies` evaluates.
So with or without the plane the stored marginals are a strict
refinement of the three-bin split rather than a replacement, which is
why ``twin.bins`` can stay off.  What goes with the plane is only the
`$k_z$` resolution of the two fluctuating bins: `$E_{\Delta u_2}$`
stays resolved in `$k_x$`, `$E_{\Delta u_1}$` in neither.  The mode is
also what :func:`~dnsjax.analysis.twin.yspectra.fluctuation_energy`
subtracts.

Spectral budget
---------------
Contracting the difference momentum equation with
`$\sigma_{k_x}\Delta\hat{\mathbf{u}}^*$` at each mode gives one
production, one transfer, one viscous and one pressure term per
`$(y, k)$` -- and the paper's (2.7)-(2.9) are `$k$`-set sums of them,
so the 12 + 12 expansions of (2.11)-(2.16) have nothing left to say
and are not reproduced.  :func:`ybudget_terms`, default
(**convective**) form:

- ``P_U``: production against the reference mean profile,
  `$-\sigma_k \mathrm{Re}\{\Delta\hat u_i^* \Delta\hat v\}\,
  \partial_y U^{(1)}_i$` -- diagonal in `$k$`, no transform, and the
  paper's dominant long-time (lift-up) term now resolved in
  `$(y, k)$`;
- ``P_r``: production against the reference *fluctuation* gradients;
- ``T_ref`` / ``T_self``: transfer by the reference fluctuation and
  by the difference field's own advection;
- ``V`` / ``eps``: the viscous term in the operator form (the one
  that closes -- "Dissipation form" above) and the positive-definite
  pseudo-dissipation `$\nu|\nabla\Delta\mathbf{u}|^2$`.  Their
  difference is the wall-normal diffusion flux;
- ``Wp``: the work done by the pressure *gradient*, from
  :mod:`dnsjax.twin.pressure`.  Not the mean-mode driving --
  although at `$(0,0)$`, where the fluctuating pressure does no work,
  it is exactly that (:func:`_driving_density`; "Mean-mode driving"
  below fixes the sign convention).

`$\sum_k \int$` of ``P_U + P_r`` and of ``-V`` reproduce
``twin_budget.dat``'s ``P_tot`` and ``eps_tot`` to rounding --
algebraic identities, the same Parseval sum regrouped.  The transfer
terms match their per-bin counterparts only up to the discrete
integration-by-parts residual that makes ``T_tot`` nonzero in the
first place (measured on the ladder in ``tests/test_twin_budget.py``).

**What `$\sum_k T(y)$` is, and is not.**  Summed over `$k$` at fixed
`$y$` the two transfer terms give

.. math::
    -\partial_y\langle v^{(1)}|\Delta\mathbf{u}|^2/2\rangle_{xz}
    -\partial_y\langle \Delta v|\Delta\mathbf{u}|^2/2\rangle_{xz}
    = -\partial_y\langle v^{(2)}|\Delta\mathbf{u}|^2/2\rangle_{xz},

the turbulent transport of difference energy -- carried by the
**perturbed** member, because `$\Delta\mathbf{u}$` is advected by
`$\mathbf{u}^{(1)} + \Delta\mathbf{u}$` and the two halves above are
exactly ``T_ref`` and ``T_self``.  Dropping the second would be
dropping one of the two terms.  A genuine wall-normal flux, zero only
after integrating in `$y$` (`$v^{(2)}$` vanishes at both walls too).
It is *not* a quantity that should vanish pointwise, and a build in
which it did would have lost that transport.  What does vanish, up to
truncation, is the `$y$`-integral.

The `$k$`-resolved budget is **cheaper** than the three-bin one: 33
field transforms against 69 (:func:`_convective_sources`), because
binning no longer forces a separate physical product per bin pair.
What it adds instead is the pressure -- one factored Poisson operator
held for the run, plus its two homogeneous columns.  Pressure is the
one term the volume-averaged budget omits for free and a localised
one cannot; :mod:`dnsjax.twin.pressure` has the whole argument, and
its "Cost" section the footprint.

Its *transient*, unlike its transform count, is not automatically the
smaller of the two.  Held together the three padded physical sets
`$\mathbf{b}$`, `$\nabla\mathbf{c}^{(1)}$` and
`$\nabla\Delta\mathbf{c}$`, with the product being formed, are
6 + 9 + 9 + 3 = 27 live fields, *more* than the three-bin pass's ~21
(counted the same way) despite half the transforms.  So
:func:`_convective_sources` forms the two gradient sets one at a time,
the reference's consumer `$q_p$` between them: 6 + 9 + 3.  Statement
order does not get that on its own -- XLA formed both sets together
-- so an optimization barrier holds it there.  What the program then
peaks at is measured rather than counted in "Memory" below: about
twice the live-field count, the rest being transform-pipeline
transient and spectral operands.

Two budget forms
----------------
``twin.rotational_ybudget`` swaps the nonlinear term for the
**rotational** one the solver actually integrates
(:mod:`dnsjax.rhs`).  Off by default.  The two forms differ by
`$\nabla\phi$` with `$\phi = \mathbf{u}^{(1)}\!\cdot\Delta\mathbf{u}
+ |\Delta\mathbf{u}|^2/2$`; contracted with
`$\Delta\hat{\mathbf{u}}^*$` and reduced by continuity that is
`$\partial_y\mathrm{Re}\{\hat\phi\,\Delta\hat v^*\}$`, a wall-normal
flux -- so the **volume totals agree** (continuously; discretely up
to the same integration-by-parts residual, measured as ``N_tot``),
while the *densities* move between production, transfer and ``Wp``.

Under it, :func:`ybudget_terms` becomes ``P_U, P_r, T_vort, T_self,
V, eps, Wp, P_lift``:

- ``T_vort`` / ``T_self`` are `$\Delta\hat{\mathbf{u}}^*\cdot(\Delta
  \mathbf{u}\times\mathbf{b})$`, and `$\mathbf{a}\cdot(\mathbf{a}
  \times\mathbf{b}) = 0$` pointwise, so `$\sum_k T(y) = 0$` at every
  `$y$` **exactly** -- at any resolution, on any grid, needing
  neither continuity nor no-slip.  That is a different statement from
  the convective one above, not a repaired version of it: the
  turbulent transport has moved, and not as a block -- its
  `$\Delta v$` half into ``Wp``, which is now the work of the
  Bernoulli pressure, and its `$\mathbf{u}^{(1)}$` half into the
  production.
- ``P_U`` carries the lift-up production, but its density is
  `$U^{(1)}_i\partial_y\langle\Delta u_i\Delta v\rangle$` rather than
  the classical `$-\langle\Delta u_i\Delta v\rangle\partial_yU^{(1)}
  _i$`; the two share a `$y$`-integral and differ by the flux
  `$\partial_y(U^{(1)}_iR_i)$`, `$R_i$` the `$u$`-`$v$` co-spectrum.
  ``P_U`` alone is also not Galilean-invariant -- a shift of
  `$\mathbf{U}^{(1)}$` moves density between it and ``Wp``, cancelling
  mode by mode.
- ``P_lift`` is therefore stored as well: it **is** the convective
  ``P_U``, unchanged, sitting outside the sum the other seven make.
  It is the frame-invariant production density, it is unrecoverable
  from anything else stored (`$R_i$` is a cross-spectrum and only the
  diagonal energies are written), and its `$k$`-sum still reproduces
  the three ``P_*(*,rU)`` columns exactly.

What the form buys: those two exact identities, and
`$\hat{\mathcal{N}}$` becoming the solver's own RHS difference -- so
the recovered pressure is the Bernoulli pressure the influence matrix
actually closes on, and the whole term is *checkable* against the
solver instead of argued (``tests/test_twin_unit.py``).  It is also
cheaper: 21 field transforms against 33, and a measured peak of 33
padded components against 37 ("Memory" below).  What it costs:
``P_r``, ``T_vort`` and ``T_self`` no longer map onto the paper's
terms at all, so `$\sum_k\int(P_U + P_r)$` is
``P_tot + T_tot`` up to truncation rather than ``P_tot`` exactly, and
the fluctuation half of the production has no convective counterpart
in the stream.  That is why the convective form is the default.

Reference halves and 3-D cubes
------------------------------
Each energy spectrum has a difference half (``e``) and, under
``twin.spectra_ref``, a reference half (``r``), written to separate
streams on separate cadences: ``twin.it_spectra_ref``,
``it_yspectra_ref`` and ``it_spectra3d_ref``, each defaulting to its
difference cadence.  Both halves are static flags of one program, so a
half that is not due is never traced, and at the default cadences one
program computes both, as before the split.  :mod:`dnsjax.twin.driver`
says how a member recorded before the split goes on.

The cubes keep the third axis that the `$(y, k)$` streams sum over.
They hold the same per-mode densities, folded and subsampled
(:func:`cube_selection`, :func:`_cube_replicated`):

- `$y$` is folded about the centreline (the mean of the two halves);
- `$k_z$` is folded onto `$|k_z|$`, summing each pair as
  :func:`_fold_kz` does;
- each axis is then thinned to points spaced uniformly in the
  logarithm of the wall distance or of the wavenumber.

Every density is computed at full resolution before any of this, so
the kept entries are exact.  With no thinning, summing a cube over
`$k_x$` (or `$|k_z|$`) gives the `$y$`-folded ``*_x`` (or ``*_z``)
marginal of the matching stream (``tests/test_twin_unit.py``).  The
budget cube holds the convective terms only, and shares the 2-D
budget's density pass when both are due.  The file layout is in
:mod:`dnsjax.twin.cubes`.

Frame invariance
----------------
A moving frame (``phys.u_grid``, e.g. the plane-Poiseuille default
`$2/3$`) is a change of *coordinate*, not of velocity: the solver adds
the convective term `$+\,\mathrm{i}k_x U_{grid}\mathbf{u}'$` to its
RHS (``cartesian._get_rhs_core`` / ``_l_bf``) and leaves the stored
field alone.  So the **energy** terms are frame-invariant for two
independent reasons -- the mean of both states shifts by the same
constant, which cancels in `$\Delta\mathbf{u}$`, and the frame term
is `$\mathrm{i}a\,\Delta\hat{\mathbf{u}}$` with real `$a$`, whose
contraction against `$\Delta\hat{\mathbf{u}}^*$` is purely
imaginary and so contributes exactly zero *per mode* -- the same
mechanism that removes :func:`_convective_sources`' two
``mean_advect`` terms from the densities, and
`$\Delta\mathbf{u}\times\boldsymbol{\Omega}^{(1)}$` from
:func:`_rotational_sources`'.

A moving frame is **not** the Galilean redefinition of "Two budget
forms" above: ``phys.u_grid`` leaves the stored field and
`$\mathbf{U}^{(1)}$` untouched, so it moves nothing between ``P_U``
and ``Wp``.

The **pressure** is the exception, and it is why both source builders
add the frame term to `$\hat{\mathcal N}$` explicitly rather than
relying on the argument above.  The `$U^{(1)}$` profile does *not*
enter every term through `$\partial_y$`: ``mean_advect`` reads the
profile itself.  (Rotationally the requirement is sharper still --
`$\hat{\mathcal N}$` has to equal the solver's own RHS difference
term for term, which the frame term is part of, and the profiles read
are the lab-frame ones ``get_nonlin`` uses rather than the shifted
``base_flow_adv_padded``; see ``_base.pad_base_flow``.)  The error
from omitting it in `$\widehat{\nabla\cdot\mathcal N}$` would be
`$\mathrm{i}k_x U_{grid}\,\nabla\!\cdot\!\Delta\mathbf{u}$` and
in the wall closure `$\mathrm{i}k_x U_{grid}(D_1\Delta\hat
v)|_w$`.  The second is machine-zero under either scheme -- each
influence matrix imposes `$(D_1 v)|_w = 0$` exactly, the legacy one in
the wall-divergence stages of ``_imm_iteration_vp`` -- and the first
only under the default ``res.consistent_imm``, whose discrete
continuity holds everywhere: the legacy flag's states carry an
`$O(1)$` relative interior divergence.  Carrying the term costs one
mode-diagonal multiply and makes :mod:`dnsjax.twin.pressure`'s "right
under either ``res.consistent_imm``" claim unconditional.

Mean-mode driving
-----------------
**Sign convention** (:mod:`dnsjax.ic.mean_mode` derives it, and it
holds codebase-wide): `$\Pi_s$` is the `$(0,0)$` mode of
`$\partial p/\partial s$` -- the mean pressure gradient -- so the
applied *force* is `$-\Pi_s$`, positive when it accelerates the flow,
which is what ``stats.dat``'s ``-dPds'`` column carries.

There is deliberately **no forcing column** in the budget, for either
driving knob.  ``_apply_bulk_corrections``
(:mod:`~dnsjax.geometries.wall_bounded.cartesian`) applies a *scalar*
body force on the `$(0,0)$` mode alone -- `$-\Pi_s$` under
``phys.driving = "constant_bulk_velocity"``, `$-\Pi_n$` under
``phys.block_mean_spanwise_velocity`` -- so its work on a field is
exactly (force) `$\times$` (that field's bulk velocity along the
forced direction).  On the *difference* field that is
`$-\Delta\Pi \cdot \mathrm{bulk}(\Delta u)$`, and every supported
setting annihilates one of the two factors, by a different mechanism:

- **force free** (``constant_pressure_gradient``, and plane-Couette,
  which carries no ``driving`` field at all): the applied force is the
  same constant in both runs, so `$\Delta\Pi = 0$`.  Note this says
  nothing about `$\mathrm{bulk}(\Delta u)$`, which is genuinely
  non-zero here -- an undriven direction acquires a bulk velocity
  spontaneously, plane-Couette's streamwise one included.
- **bulk held**: both runs hold the *same* bulk value, so
  `$\mathrm{bulk}(\Delta u) = 0$` -- exactly, because the correction is
  a rank-1 algebraic projection satisfied at every corrector iterate,
  not a converged feedback loop.  Here `$\Delta\Pi$` is the non-zero
  factor: the two runs apply genuinely different, time-varying forces.

The cancellation is *exact* rather than approximate only because the
same quadrature ``flow.y_weights`` defines the bulk in all three
places that matter: the corrector's constraint, the held-mean
constraint row of the partner's `$(0,0)$` initial perturbation
(:mod:`dnsjax.ic.mean_mode`), and :func:`get_inprod` here.  Guard --
measuring both factors, so neither leg can pass vacuously:
``tests/test_twin_budget.py``.

Contrast the *total* field, whose budget does carry the term
(``I`` in each flow's ``get_stats``): there the held streamwise bulk
of plane-Poiseuille is `$U_b = 2/3 \ne 0$`, so the mean pressure
gradient does real, time-varying work.  The spanwise block is the
degenerate case in which the held value is zero, so it does none.

Sharding
--------
The masks derive from ``fourier.kx`` (spec ``P(None, None, a1)``) and
``fourier.mean_mask`` (``P(None, a0, a1)``) through *binary* ops
only, which infer the combined partition spec -- the
``jnp.broadcast_to``-keeps-the-source-spec trap (see the precedent
note in ``cylindrical.py``, ``_imm_iteration_vw``) cannot arise
because no mask is ever materialised standalone at full shape.
Reductions are plain ``get_norm2`` sums over the sharded axes;
outputs are replicated scalars.

Memory
------
Measured, not counted: each program's peak transient from XLA's own
buffer assignment (``temp_size_in_bytes`` of
``jax.jit(f).lower(...).compile().memory_analysis()``), in padded
physical components -- one real field on the 3/2-rule grid,
`$18\,n_x n_y n_z$` bytes at double precision.  CPU, plane-Poiseuille
at `$64\times65\times64$` and `$96\times97\times96$` (the two agree
to 1 %), at ``solver.rhs_transform_chunks`` 1 and 3:

- :func:`twin_budget` (three-bin): 43 / 37;
- :func:`twin_ybudget`: 37 / 31 convective, 33 / 27 rotational, and
  the same with the 3-D cube, alone or beside the marginals;
- :func:`difference_pressure` (the ``lowres_delta/`` pressure): 34 /
  29, the budget's sources and solve without its densities;
- for scale, the time step itself: 30 / 28 iterative-CN, 31 / 29
  CN/AB2;
- :func:`twin_energies` 2.6 (7.9 under ``twin.bins``), the two
  spectra samples 1.5 and 1.3, the 3-D energy cubes 1.3 -- none of
  them a peak.

Every program is lowered as the run calls it, the singletons passed as
arguments.  Baked in as constants instead (a ``jit`` of the bound
step) the step measures 20 / 18 and 28 / 26: that program holds no
copies of the banded factors inside its corrector loop, which the
program a run executes does (visible in its compiled HLO).

So either budget, when enabled, is the run's high-water mark, since
the device allocator's pool grows to the maximum over every program;
without one, the difference pressure sits level with the step or up to
a sixth above it.
A count of live fields misses about half: every batched transform
carries some two padded fields of pipeline transient per field in
flight on top of its output (a 3-field :func:`spec_to_phys` alone
peaks at 6, a 9-field one at 18), plus the spectral operands around
it.  The two optimization barriers (:func:`_convective_sources`,
:func:`_twin_budget_jit`) are worth 47 `$\to$` 37 and
61 `$\to$` 43 of these, for the same numbers to rounding and no
measurable CPU time.  A GPU schedule is its own: size a job against
the driver's closing ``Peak device memory`` line, not this list.
"""

import importlib
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import numpy as np
from jax import Array, jit, lax, shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ..fft import chunked_transform
from ..flows.registry import cartesian_systems, spec_for
from ..geometries.wall_bounded._base import (
    apply_y_matrix,
    extract_mean_modes,
    get_inprod,
    get_norm2,
    integrate_scalar,
    phys_to_spec,
    spec_to_phys,
)
from ..geometries.wall_bounded._cartesian_pressure import mean_advect
from ..geometries.wall_bounded.cartesian import (
    DRIVING_KEY_N,
    DRIVING_KEY_S,
    Fourier,
    _curl_fn,
    fourier,
    mean_driving_from_profile,
)
from ..parameters import derived_params, params
from ..sharding import sharding
from .pressure import DifferencePressure

if params.phys.system not in cartesian_systems:  # pragma: no cover
    raise RuntimeError(
        "dnsjax.twin.diagnostics supports the Cartesian wall-bounded "
        f"flows only (system {params.phys.system!r}); the [twin] "
        "surface should have rejected this configuration."
    )

#: The selected flow's module (shared singletons with the driver via
#: the import cache); its ``flow`` instance carries the grid
#: quadrature ``y_weights`` (and, for the budget terms, ``D1`` and
#: the laminar base profile).
_flow_mod = importlib.import_module(spec_for(params.phys.system).flow_module)
flow = _flow_mod.flow


def component_masks(fourier_: Fourier) -> tuple[Array, Array, Array]:
    r"""The mean / streak / streamwise-varying mode masks.

    Returns ``(m_mean, m_u1, m_u2)`` boolean masks broadcastable
    against the spectral state's trailing ``(N_y, N_{k_z}, N_{k_x})``
    axes (shapes ``(1, N_{k_z}, N_{k_x})``, ditto, and
    ``(1, 1, N_{k_x})``).  Built from ``fourier_`` fields through
    binary ops only (see the module docstring's sharding note); cheap
    enough to rebuild inside every jitted diagnostic, keeping the
    jaxprs free of captured device-array constants.
    """
    m_mean = fourier_.mean_mask
    m_u1 = (fourier_.kx == 0) & ~m_mean
    m_u2 = fourier_.kx != 0
    return m_mean, m_u1, m_u2


@partial(jit, static_argnames=("bins",))
def _twin_energies_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    *,
    bins: bool,
) -> dict[str, Array]:
    r"""Difference-field energies (see the module docstring).

    Always:

    - ``E_d``: total `$E_\Delta = \|\Delta\mathbf{u}\|^2/2$`.
    - ``E_ref``: the reference state's own `$E'$` (context for
      saturation levels and the laminarization read).

    Under ``bins`` (``twin.bins``), additionally the three-bin
    decomposition of the reference paper:

    - ``E_dU`` / ``E_du1`` / ``E_du2``: the mean / streak /
      streamwise-varying components (computed independently; their
      sum equals ``E_d`` to rounding -- a deliberate consistency
      guard, not a redundancy to trade away);
    - ``E_du1_x`` / ``E_du1_y`` / ``E_du1_z``: `$E_{\Delta u_1}$`
      per velocity component.

    ``bins`` is **static**: the two column sets are separate compiled
    programs, and the flag is pinned in ``twin.json`` so a resume
    cannot append one to the other.  With it off, the two masked
    copies (``delta * m_*``) -- the pair of full-state temporaries
    that make this call a few percent of a twin step at
    ``it_energy = 1`` -- are never built.  The scale-resolved
    successor is :func:`twin_yspectra`, from which all three bin
    energies are recoverable (module docstring, "Wall-normal-resolved
    spectra").

    Keys are chosen so their *sorted* order (the ``twin.dat`` column
    order -- dicts returned through ``jit`` are canonicalised, see
    :mod:`dnsjax.measurements`) groups the components readably.
    """
    k_metric = fourier_.k_metric
    w = flow_.y_weights
    delta = state2 - state1
    out = {
        "E_d": get_norm2(delta, k_metric, w) / 2,
        "E_ref": get_norm2(state1, k_metric, w) / 2,
    }
    if not bins:
        return out
    m_mean, m_u1, m_u2 = component_masks(fourier_)
    du1 = delta * m_u1
    return out | {
        "E_dU": get_norm2(delta * m_mean, k_metric, w) / 2,
        "E_du1": get_norm2(du1, k_metric, w) / 2,
        "E_du1_x": get_norm2(du1[0:1], k_metric, w) / 2,
        "E_du1_y": get_norm2(du1[1:2], k_metric, w) / 2,
        "E_du1_z": get_norm2(du1[2:3], k_metric, w) / 2,
        "E_du2": get_norm2(delta * m_u2, k_metric, w) / 2,
    }


def twin_energies(
    state1: Array, state2: Array, *, bins: bool
) -> dict[str, Array]:
    """Wrapper around ``_twin_energies_jit`` binding the singletons."""
    return _twin_energies_jit(state1, state2, fourier, flow, bins=bins)


# ── Budget terms (see the module docstring's "Budget terms") ─────────

#: The paper's production triples ``(a, b, c)`` of
#: `$-\langle a \cdot (b \cdot \nabla) c \rangle$` (eqs. 2.11-2.13),
#: with ``d*`` the difference components and ``r*`` the reference's.
_PRODUCTION: tuple[tuple[str, str, str], ...] = (
    ("dU", "dU", "rU"),
    ("dU", "du1", "ru1"),
    ("dU", "du2", "ru2"),
    ("du1", "du1", "rU"),
    ("du1", "dU", "ru1"),
    ("du1", "du1", "ru1"),
    ("du1", "du2", "ru2"),
    ("du2", "dU", "ru2"),
    ("du2", "du1", "ru2"),
    ("du2", "du2", "rU"),
    ("du2", "du2", "ru1"),
    ("du2", "du2", "ru2"),
)

#: The transport triples (eqs. 2.14-2.16).  Each advector ``b`` acts
#: symmetrically on an ``(a, c)`` pair across two rows, so the twelve
#: terms cancel pairwise by parts (``T_tot`` ~ 0).
_TRANSPORT: tuple[tuple[str, str, str], ...] = (
    ("dU", "ru1", "du1"),
    ("dU", "du1", "du1"),
    ("dU", "ru2", "du2"),
    ("dU", "du2", "du2"),
    ("du1", "ru1", "dU"),
    ("du1", "du1", "dU"),
    ("du1", "ru2", "du2"),
    ("du1", "du2", "du2"),
    ("du2", "ru2", "dU"),
    ("du2", "du2", "dU"),
    ("du2", "ru2", "du1"),
    ("du2", "du2", "du1"),
)

#: Fields that are pure mean profiles (see the module docstring).
_MEANS: frozenset[str] = frozenset({"dU", "rU"})


def budget_names() -> list[str]:
    """The ``twin_budget`` keys (unsorted; JIT sorts the columns)."""
    names = [
        f"{kind}_{a}({b},{c})"
        for kind, table in (("P", _PRODUCTION), ("T", _TRANSPORT))
        for a, b, c in table
    ]
    names += ["eps_dU", "eps_du1", "eps_du2", "P_tot", "T_tot", "eps_tot"]
    return names


@jit
def _twin_budget_jit(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> dict[str, Array]:
    r"""The 24 advective terms, 3 dissipations, and consistency sums.

    Term math, evaluation classes, and naming: the module docstring.
    All Python control flow below is trace-time (static tables).
    """
    k_metric = fourier_.k_metric
    kx = fourier_.kx
    kz = fourier_.kz
    D1 = flow_.D1
    w = flow_.y_weights
    vf = derived_params.volume_fac
    re = params.phys.re

    delta = state2 - state1
    m_mean, m_u1, m_u2 = component_masks(fourier_)
    full = {
        "dU": delta * m_mean,
        "du1": delta * m_u1,
        "du2": delta * m_u2,
        "ru1": state1 * m_u1,
        "ru2": state1 * m_u2,
    }
    # One collective for the pair (:func:`._base.extract_mean_modes`);
    # cadenced, so this is tidiness rather than a measured win.
    mean_delta, mean_ref = extract_mean_modes(delta, state1)
    prof = {
        "dU": mean_delta.real,
        "rU": mean_ref.real + flow_.base_flow[:, :, 0, 0],
    }

    def d_dy_prof(p: Array) -> Array:
        """FD wall-normal derivative of a ``(3, Ny)`` profile."""
        return jnp.einsum("ij,cj->ci", D1, p)

    def xz_mean_cross(f: Array, g: Array) -> Array:
        r"""``(C, Ny)`` profile of the `$xz$`-mean of the product of
        the real fields with spectral coefficients *f* (``(C,...)``)
        and *g* (``(1,...)``, broadcast) -- Parseval with the
        real-FFT weight."""
        return jnp.sum(k_metric * (f * jnp.conj(g)).real, axis=(2, 3))

    def grad_spec(c: Array) -> tuple[Array, Array, Array]:
        return 1j * kx * c, apply_y_matrix(D1, c), 1j * kz * c

    def term_c_mean(a: str, b: str, c: str) -> Array:
        dyc = d_dy_prof(prof[c])
        cross = xz_mean_cross(full[a], full[b][1:2])
        return -integrate_scalar(jnp.sum(dyc * cross, axis=0), w) / vf

    def term_b_mean(a: str, b: str, c: str) -> Array:
        # `$b$` is only ever `$\Delta U$`, whose wall-normal row is an
        # identical zero (module docstring, "State preconditions"), so
        # the `$b_y\,\partial_y$` half of the advection is a
        # full-field GEMM against exact zero and is not formed.
        bx = prof[b][0][:, None, None]
        bz = prof[b][2][:, None, None]
        adv = 1j * (bx * kx + bz * kz) * full[c]
        return -get_inprod(full[a], adv, k_metric, w)

    def term_a_mean(a: str, b: str, c: str) -> Array:
        dxc, dyc, dzc = grad_spec(full[c])
        mean_prof = (
            xz_mean_cross(dxc, full[b][0:1])
            + xz_mean_cross(dyc, full[b][1:2])
            + xz_mean_cross(dzc, full[b][2:3])
        )
        return -integrate_scalar(jnp.sum(prof[a] * mean_prof, axis=0), w) / vf

    out: dict[str, Array] = {}

    # Triple-fluctuating terms: grouped by c (one gradient set each),
    # the advector physical forms cached across the pass.
    fluct = [
        (kind, a, b, c)
        for kind, table in (("P", _PRODUCTION), ("T", _TRANSPORT))
        for a, b, c in table
        if not ({a, b, c} & _MEANS)
    ]
    b_names = tuple(dict.fromkeys(t[2] for t in fluct))
    b_stack = jnp.concatenate([full[n] for n in b_names], axis=0)
    b_phys_all = chunked_transform(spec_to_phys, b_stack)
    b_phys = {
        n: b_phys_all[3 * i : 3 * (i + 1)] for i, n in enumerate(b_names)
    }
    for c in dict.fromkeys(t[3] for t in fluct):
        grad_stack = jnp.concatenate(grad_spec(full[c]), axis=0)
        # Rows [j * 3 + i] = the j-derivative of component i.
        grad_phys = chunked_transform(spec_to_phys, grad_stack)
        for b in dict.fromkeys(t[2] for t in fluct if t[3] == c):
            q_phys = jnp.stack(
                [
                    sum(b_phys[b][j] * grad_phys[3 * j + i] for j in range(3))
                    for i in range(3)
                ]
            )
            q_spec = chunked_transform(phys_to_spec, q_phys)
            for kind, a, b_t, c_t in fluct:
                if (b_t, c_t) == (b, c):
                    out[f"{kind}_{a}({b},{c})"] = -get_inprod(
                        full[a], q_spec, k_metric, w
                    )
        # One gradient set live at a time.  The four are independent
        # of one another, so without this XLA forms them together
        # (module docstring, "Budget terms").
        out, b_phys, full = lax.optimization_barrier((out, b_phys, full))

    # Mean-slot terms (FFT-free classes; priority c > b > a keeps the
    # dispatch unambiguous for multi-mean triples).
    for kind, table in (("P", _PRODUCTION), ("T", _TRANSPORT)):
        for a, b, c in table:
            name = f"{kind}_{a}({b},{c})"
            if name in out:
                continue
            if c in _MEANS:
                out[name] = term_c_mean(a, b, c)
            elif b in _MEANS:
                out[name] = term_b_mean(a, b, c)
            else:
                out[name] = term_a_mean(a, b, c)

    # Dissipations (the discrete-Laplacian form -- the module
    # docstring's "Dissipation form") and the consistency sums.
    for x in ("dU", "du1", "du2"):
        lap = -fourier_.k2 * full[x] + apply_y_matrix(flow_.D2, full[x])
        out[f"eps_{x}"] = -get_inprod(full[x], lap, k_metric, w) / re
    out["P_tot"] = sum(out[f"P_{a}({b},{c})"] for a, b, c in _PRODUCTION)
    out["T_tot"] = sum(out[f"T_{a}({b},{c})"] for a, b, c in _TRANSPORT)
    out["eps_tot"] = out["eps_dU"] + out["eps_du1"] + out["eps_du2"]
    return out


def twin_budget(state1: Array, state2: Array) -> dict[str, Array]:
    """Wrapper around ``_twin_budget_jit`` binding the singletons."""
    return _twin_budget_jit(state1, state2, fourier, flow)


# ── (kz, kx) energy spectra ──────────────────────────────────────────


def _mode_energy_replicated(field: Array, w: Array, k_metric: Array) -> Array:
    r"""Per-mode energy of a spectral field, replicated across devices.

    `$E(k_z, k_x) = \tfrac{1}{2}\,\mathrm{metric}\,
    \int |\hat{u}|^2\, w\, \mathrm{d}y \,/\, V$` summed over the
    velocity components -- so summing the returned array over the
    true modes reproduces the total energy exactly (the twin.dat
    convention).  Each device reduces its own ``(k_z, k_x)`` tile and
    scatters it into a zero global-shape array at its mesh position;
    a ``psum`` over both mesh axes assembles the **replicated**
    global spectrum (the disjoint-tile analogue of
    ``extract_mean_mode``) -- required because the writer's rank-0
    host transfer needs a fully-addressable array under multi-process
    launches.  Shape ``(N_{k_z}, N_{k_x})`` *padded* sizes; the
    padding rows/columns weight zero data and are stripped by the
    caller.

    Called twice per sample (difference and reference), i.e. two
    ``psum``\ s where one stacked call would do.  Deliberately left
    alone: the collective carries one mode plane
    (`$\sim$`1 MB at a `$1024\times257\times256$` target) against the
    `$\sim$`3 GB *spectral field* each call streams through the
    einsum, so merging them would save `$\sim$`0.03 % of a diagnostic
    that already runs on a cadence.  The cost here is the field pass,
    and there is only one of those per state either way.
    """
    nz_spec, nx_spec = sharding.spec_shape[1], sharding.spec_shape[2]
    vf = derived_params.volume_fac

    def _local(shard: Array, w_loc: Array, k_metric_loc: Array) -> Array:
        # ``(z conj(z)).real`` rather than ``abs(z) ** 2``: the latter
        # takes a square root per element only to square it back (the
        # ``get_inprod`` / ``xz_mean_cross`` idiom, and one fewer
        # rounding on the sum-equals-``E_d`` identity).
        e_loc = (
            jnp.einsum("j,cjkl->kl", w_loc, (shard * jnp.conj(shard)).real)
            * k_metric_loc[0]
        )
        row0 = lax.axis_index("np0") * e_loc.shape[0]
        col0 = lax.axis_index("np1") * e_loc.shape[1]
        full = jnp.zeros((nz_spec, nx_spec), dtype=e_loc.dtype)
        full = lax.dynamic_update_slice(full, e_loc, (row0, col0))
        return lax.psum(full, ("np0", "np1"))

    gathered = shard_map(
        _local,
        mesh=sharding.mesh,
        in_specs=(
            sharding.spec_vector_shard,
            P(None),
            P(None, None, sharding.a1),
        ),
        out_specs=P(None, None),
    )(field, w, k_metric)
    return gathered / (2.0 * vf)


@partial(jit, static_argnames=("ref", "delta"))
def _twin_spectra_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    *,
    ref: bool,
    delta: bool = True,
) -> dict[str, Array]:
    r"""``(k_z, k_x)`` energy spectra of the difference and reference.

    ``e_delta`` is the per-mode `$E_\Delta(k_z, k_x)$` and, under
    *ref* (``twin.spectra_ref``), ``e_ref`` the reference state's own
    spectrum (their ratio `$E_\Delta / 2 E^{(1)}$` is the offline
    decorrelation measure: fully decorrelated independent fields give
    1).  True modes only (padding stripped); summing ``e_delta``
    reproduces ``twin.dat``'s ``E_d`` to rounding (a
    ``tests/test_twin_unit.py`` guard).

    *ref* and *delta* are **static**, like ``bins`` on
    :func:`_twin_energies_jit`: a half that is off is never traced,
    field pass and ``psum`` alike.  The two halves go to two streams,
    each on its own cadence (``twin.it_spectra`` /
    ``twin.it_spectra_ref``, the second defaulting to the first), so a
    sample asks for whichever are due; at the default cadences that is
    both, in one program.
    """
    n2 = params.res.nz - 1
    n3 = params.res.nx // 2
    w = flow_.y_weights
    k_metric = fourier_.k_metric
    out: dict[str, Array] = {}
    if delta:
        out["e_delta"] = _mode_energy_replicated(state2 - state1, w, k_metric)[
            :n2, :n3
        ]
    if ref:
        out["e_ref"] = _mode_energy_replicated(state1, w, k_metric)[:n2, :n3]
    return out


def twin_spectra_2d(
    state1: Array, state2: Array, *, ref: bool = True, delta: bool = True
) -> dict[str, Array]:
    """Wrapper around ``_twin_spectra_jit`` binding the singletons."""
    return _twin_spectra_jit(
        state1, state2, fourier, flow, ref=ref, delta=delta
    )


# ── Wall-normal-resolved marginal spectra ────────────────────────────


def marginal_bin_counts() -> tuple[int, int]:
    r"""``(n_{k_z}, n_{k_x})`` of the folded marginal axes.

    Both are one-sided: `$n_z/2$` and `$n_x/2$` bins, carrying
    integer wavenumbers `$0, 1, \dots$` (``harmonics.real_harmonics``
    of each axis' full count).  See :func:`_fold_kz` for why the
    `$k_z$` axis is folded rather than stored two-sided.

    Refuses an odd ``res.nz``, where the stored `$k_z$` band would be
    asymmetric and :func:`_fold_kz` would have no slot for the
    outermost negative mode.  ``parameters.validate_parameters``
    refuses an odd Fourier count outright, for the same reason and
    everywhere, so this is unreachable through any entry point; it
    keeps the invariant with the function whose contract states it
    rather than only with the parameter layer.
    """
    if params.res.nz % 2:
        raise ValueError(
            "marginal_bin_counts needs an even res.nz (got "
            f"{params.res.nz}): the wall-normal-resolved streams fold "
            "the k_z axis onto |k_z|, and at odd nz the outermost "
            "negative mode has no positive partner (_fold_kz)."
        )
    return params.res.nz // 2, params.res.nx // 2


def _fold_kz(a: Array) -> Array:
    r"""Fold a stored `$k_z$` axis onto `$|k_z|$`.

    *a* is a replicated array whose **last** axis is the stored
    full-complex `$k_z$` axis, padding already stripped
    (`$n_z - 1$` entries in FFT wrap order,
    :func:`dnsjax.harmonics.complex_harmonics`).  Returns `$n_z/2$`
    entries, entry `$j$` being the sum of the `$\pm j$` pair.

    **Why the fold is mandatory, not a convenience.**  The stored
    half-plane carries `$k_x \ge 0$` with the conjugate-pair weight
    ``k_metric``, so a stored entry is the energy of the *pair*
    `$\{(k_x, k_z), (-k_x, -k_z)\}$`.  Marginalising that over
    `$k_x$` therefore does **not** give the two-sided spectrum at
    `$k_z$` -- the partner of `$(k_x > 0, k_z)$` sits at `$-k_z$`.
    Only after summing the `$\pm k_z$` pair do the two agree:

    .. math::
        \sum_{k_x \ge 0} \sigma_{k_x} |\hat{u}(k_x, k_z)|^2
        + (k_z \to -k_z)
        = \sum_{k_x} |\hat{u}(k_x, k_z)|^2 + (k_z \to -k_z) .

    The `$k_x$` marginal needs no such fold: summing over the whole
    stored `$k_z$` axis already covers both partners.
    """
    npos = params.res.nz // 2
    return a[..., :npos].at[..., 1:].add(a[..., npos:][..., ::-1])


def marginal_suffixes(x0: bool) -> tuple[str, ...]:
    r"""The stored marginal names, in stored order.

    ``x`` / ``z`` are the two marginals and ``xz00`` the `$(0, 0)$`
    mode; ``x0``, the full `$k_x = 0$` plane, joins them under
    ``twin.x0_planes``.  This is the order
    :func:`_marginals_replicated` returns and the order both writers
    (:mod:`dnsjax.twin.yspectra`) lay a record out in, so the two
    cannot drift apart.
    """
    return ("x", "z", "x0", "xz00") if x0 else ("x", "z", "xz00")


def _marginals_replicated(density: Array, *, x0: bool) -> dict[str, Array]:
    r"""The marginals of a per-mode density, replicated.

    *density* is a **real** `$(C, N_y, N_{k_z}, N_{k_x})$` array in
    the spectral layout, already carrying its ``k_metric`` weight and
    any prefactor -- so that summing it over `$(k_z, k_x)$` and
    integrating over `$y$` with ``y_weights`` reproduces the scalar
    the same quantity gives in ``twin.dat`` / ``twin_budget.dat``.
    The leading axis is free: three velocity components for the
    energies, one per term for the budget.

    Returns one entry per :func:`marginal_suffixes` name, each
    replicated with the spectral padding stripped:

    - ``x`` `$(C, N_y, n_z/2)$`: summed over `$k_x$` and folded onto
      `$|k_z|$` (:func:`_fold_kz`) -- the `$x$`-averaged spectrum;
    - ``z`` `$(C, N_y, n_x/2)$`: summed over `$k_z$` -- the
      `$z$`-averaged spectrum;
    - ``xz00`` `$(C, N_y)$`: the `$(k_x, k_z) = (0, 0)$` mode alone.
      It takes no fold -- `$k_z = 0$` is its own `$\pm$` partner, so
      :func:`_fold_kz` would leave it exactly as it is -- and it is
      *not* ``x[..., 0]`` (the whole `$k_z = 0$` column, summed over
      `$k_x$`) nor ``z[..., 0]`` (the whole `$k_x = 0$` plane, summed
      over `$k_z$`);
    - ``x0`` `$(C, N_y, n_z/2)$`, under *x0* only: the `$k_x = 0$`
      plane, folded like ``x`` -- the spectrum *of the
      streamwise-averaged field*, which splits each `$k_z$` of ``x``
      into its `$k_x = 0$` part (`$\Delta u_1$`, or `$\Delta U$` at
      `$k_z = 0$`) and its `$\Delta u_2$` part (module docstring).

    Each device reduces its own `$(k_z, k_x)$` tile, scatters its
    blocks into zero global-shape arrays at its mesh position, and one
    ``psum`` over both mesh axes assembles the replicated result --
    the pattern of :func:`_mode_energy_replicated`, and required for
    the same reason (the writer's rank-0 host transfer needs a
    fully-addressable array).  The blocks share one collective:
    unlike the two `$(k_z, k_x)$` planes there, they are reductions of
    the *same* field pass, so there is nothing to gain by splitting
    them.  The fold runs **after** the ``psum``: the `$\pm k_z$`
    partners live on different ``np0`` devices.

    What *x0* costs is therefore not a field pass but a third of the
    payload -- one more `$(C, N_y, N_{k_z}^{\mathrm{spec}})$` block in
    a collective that otherwise carries
    `$N_{k_z}^{\mathrm{spec}} + N_{k_x}^{\mathrm{spec}} + 1$` columns
    -- and a third of every stored record.  ``xz00`` is the single
    trailing column, which is why it is unconditional.
    """
    nz_spec, nx_spec = sharding.spec_shape[1], sharding.spec_shape[2]

    def _local(d: Array) -> Array:
        nkz_loc, nkx_loc = d.shape[2], d.shape[3]
        row0 = lax.axis_index("np0") * nkz_loc
        col0 = lax.axis_index("np1") * nkx_loc
        c, ny = d.shape[0], d.shape[1]
        zeros_z = jnp.zeros((c, ny, nz_spec), dtype=d.dtype)
        zeros_x = jnp.zeros((c, ny, nx_spec), dtype=d.dtype)
        blocks = [
            lax.dynamic_update_slice_in_dim(
                zeros_z, jnp.sum(d, axis=3), row0, 2
            ),
            lax.dynamic_update_slice_in_dim(
                zeros_x, jnp.sum(d, axis=2), col0, 2
            ),
        ]
        if x0:
            # ``k_x = 0`` is local column 0 of the first device column
            # only; every other device contributes exact zeros.
            x0_loc = jnp.where(col0 == 0, d[:, :, :, 0], 0.0)
            blocks.append(
                lax.dynamic_update_slice_in_dim(zeros_z, x0_loc, row0, 2)
            )
        # ``(0, 0)`` lives on one device and is already global-shaped
        # at length 1, so it is appended rather than scattered.
        blocks.append(
            jnp.where((row0 == 0) & (col0 == 0), d[:, :, 0, 0], 0.0)[..., None]
        )
        return lax.psum(jnp.concatenate(blocks, axis=2), ("np0", "np1"))

    gathered = shard_map(
        _local,
        mesh=sharding.mesh,
        in_specs=(sharding.spec_vector_shard,),
        out_specs=P(None, None, None),
    )(density)

    n2 = params.res.nz - 1
    n3 = params.res.nx // 2
    out = {
        "x": _fold_kz(gathered[..., :n2]),
        "z": gathered[..., nz_spec : nz_spec + n3],
    }
    offset = nz_spec + nx_spec
    if x0:
        out["x0"] = _fold_kz(gathered[..., offset : offset + n2])
        offset += nz_spec
    out["xz00"] = gathered[..., offset]
    return {suf: out[suf] for suf in marginal_suffixes(x0)}


def _energy_density(state: Array, fourier_: Fourier) -> Array:
    r"""Per-component, per-mode energy density in `$y$`.

    `$\tfrac{1}{2}\sigma_{k_x}|\hat{u}_c|^2 / V$`, shaped
    `$(3, N_y, N_{k_z}, N_{k_x})$` -- so
    `$\sum_j w_j \sum_{c,k}$` of it is the solver-measure energy.
    ``(z conj(z)).real`` rather than ``abs(z) ** 2``: see
    :func:`_mode_energy_replicated`.
    """
    return (state * jnp.conj(state)).real * (
        fourier_.k_metric / (2.0 * derived_params.volume_fac)
    )


@partial(jit, static_argnames=("ref", "x0", "delta"))
def _twin_yspectra_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    *,
    ref: bool,
    x0: bool,
    delta: bool = True,
) -> dict[str, Array]:
    r"""Wall-normal-resolved componentwise spectra (module docstring).

    ``e_<suffix>`` are the difference field's marginals of
    :func:`_marginals_replicated`, per velocity component, one per
    :func:`marginal_suffixes` name; under *ref* (``twin.spectra_ref``)
    the ``r_<suffix>`` set is the reference state's.  Every array is a
    `$y$`-**density**: integrate with ``flow.y_weights`` (shipped in
    the stream's sidecar) to get the per-`$k$` energy, and sum over
    `$k$` for ``twin.dat``'s ``E_d``.

    *ref*, *delta* and *x0* are all **static**, like ``bins`` on
    :func:`_twin_energies_jit`.  The two halves -- each a full real
    `$(3, N_y, N_{k_z}, N_{k_x})$` density **and** a ``psum`` -- go to
    two streams on their own cadences (``twin.it_yspectra`` /
    ``twin.it_yspectra_ref``, the second defaulting to the first), and
    a half that is not due is never traced.  *x0*
    (``twin.x0_planes``) is a third of each collective rather than a
    field pass, and with it off nothing about the `$k_x = 0$` plane is
    traced at all.  The writers pin *x0* in each stream's sidecar
    (``suffixes``), so a resume cannot flip it mid-stream.
    """
    out: dict[str, Array] = {}
    halves = []
    if delta:
        halves.append(("e", state2 - state1))
    if ref:
        halves.append(("r", state1))
    for prefix, field_ in halves:
        out |= {
            f"{prefix}_{suf}": v
            for suf, v in _marginals_replicated(
                _energy_density(field_, fourier_), x0=x0
            ).items()
        }
    return out


def twin_yspectra(
    state1: Array,
    state2: Array,
    *,
    ref: bool = True,
    x0: bool = False,
    delta: bool = True,
) -> dict[str, Array]:
    """Wrapper around ``_twin_yspectra_jit`` binding the singletons."""
    return _twin_yspectra_jit(
        state1, state2, fourier, flow, ref=ref, x0=x0, delta=delta
    )


# ── Three-dimensional (y, k_z, k_x) cubes ────────────────────────────


def log_spaced_indices(coord: np.ndarray, n: int | None) -> np.ndarray:
    r"""Indices of *coord* spaced uniformly in its logarithm.

    *coord* is ascending with ``coord[0]`` the zero of the axis (the
    wall, or `$k = 0$`) and every other entry positive.  *n* targets
    are laid out by ``geomspace`` from ``coord[1]`` to ``coord[-1]``,
    each is snapped to the entry nearest it in log distance, the
    duplicates go, and index 0 is always kept -- so the result is
    ascending, holds both ends, and has at most ``n + 1`` entries;
    fewer wherever the axis is coarser than the targets (the integer
    harmonics at low `$k$`, a CGL grid near the wall).  *n* ``None``,
    or at least the number of positive entries, keeps every index.
    """
    coord = np.asarray(coord, dtype=np.float64)
    if n is None or n >= coord.size - 1:
        return np.arange(coord.size)
    positive = np.log(coord[1:])
    targets = np.log(np.geomspace(coord[1], coord[-1], n))
    nearest = np.abs(positive[None, :] - targets[:, None]).argmin(axis=1)
    return np.concatenate([[0], np.unique(nearest) + 1])


@dataclass(frozen=True)
class CubeSelection:
    r"""The static index sets of the 3-D twin cubes (hashable).

    ``iy`` are the kept rows of the lower half, ascending from the wall,
    and ``iy_mirror`` their partners ``ny - 1 - iy`` on the upper half;
    ``kz_pos`` the stored index of `$+m$` for each kept `$|k_z| = m$`
    (`$m = 0$` first, always) and ``kz_neg`` that of `$-m$` for each
    kept `$m \ge 1$`, in the same order; ``kx`` the kept stored (equal:
    harmonic) `$k_x$` indices.  A ``jit`` static argument, so one per
    run compiles once.
    """

    iy: tuple[int, ...]
    iy_mirror: tuple[int, ...]
    kz_pos: tuple[int, ...]
    kz_neg: tuple[int, ...]
    kx: tuple[int, ...]


def cube_selection(
    y,
    n_y: int | None,
    n_kz: int | None,
    n_kx: int | None,
) -> CubeSelection:
    r"""The cubes' rows and modes for this run (module docstring).

    *y* is the run's ascending wall-normal grid.  The `$y$` fold pairs
    row `$j$` with `$n_y - 1 - j$`, so the grid must be symmetric about
    the centreline -- checked, to the run's precision, rather than
    assumed (a custom ``geo.wall_grid`` need not be) -- and the log
    spacing runs in the wall distance `$1 + y_j$` of the lower half.
    `$k_z$` folds onto `$|k_z|$`, `$m = 0 \ldots n_z/2 - 1$`, and `$k_x$`
    keeps its `$m = 0 \ldots n_x/2 - 1$`; each axis is subsampled by
    :func:`log_spaced_indices` with its own count.
    """
    y = np.asarray(y, dtype=np.float64)
    tol = 1e-12 if params.res.double_precision else 1e-6
    if not np.allclose(y, -y[::-1], rtol=0.0, atol=tol):
        raise ValueError(
            "the 3-D twin cubes fold y about the centreline, and this "
            "wall-normal grid is not symmetric about it (a custom "
            "geo.wall_grid?); leave twin.it_spectra3d / it_budget3d "
            "unset for it."
        )
    ny = y.size
    iy = log_spaced_indices(1.0 + y[: (ny + 1) // 2], n_y)
    nz, nx = params.res.nz, params.res.nx
    mz = log_spaced_indices(np.arange(nz // 2), n_kz)
    return CubeSelection(
        iy=tuple(int(j) for j in iy),
        iy_mirror=tuple(int(ny - 1 - j) for j in iy),
        kz_pos=tuple(int(m) for m in mz),
        kz_neg=tuple(int(nz - 1 - m) for m in mz if m >= 1),
        kx=tuple(int(m) for m in log_spaced_indices(np.arange(nx // 2), n_kx)),
    )


def _cube_replicated(density: Array, sel: CubeSelection) -> Array:
    r"""A per-mode density folded, subsampled and replicated.

    *density* is a real `$(C, N_y, N_{k_z}, N_{k_x})$` array in the
    spectral layout, as :func:`_marginals_replicated` takes it.
    Returns `$(C, n_y, n_{k_z}, n_{k_x})$` on *sel*, replicated -- the
    writer's host transfer needs a fully addressable array.

    One ``shard_map``: each device folds `$y$` locally (the axis is not
    sharded; the mean of a row and its mirror, the mid-plane pairing
    with itself), takes the kept `$k_z$` indices -- both members of
    each `$\pm m$` pair -- and `$k_x$` indices out of its own tile
    (a clipped local index and a range mask, the global offsets being
    its mesh position), and one ``psum`` over both mesh axes assembles
    them.  The `$k_z$` fold, a sum, runs **after** the ``psum``: the
    partners live on different ``np0`` devices
    (:func:`_marginals_replicated`).  Folding commutes with taking
    entries, so the result is the folded full-resolution cube on the
    kept points exactly.
    """
    iy = np.asarray(sel.iy)
    iy_mirror = np.asarray(sel.iy_mirror)
    kz_idx = np.asarray(sel.kz_pos + sel.kz_neg)
    kx_idx = np.asarray(sel.kx)

    def _local(d: Array) -> Array:
        nkz_loc, nkx_loc = d.shape[2], d.shape[3]
        row0 = lax.axis_index("np0") * nkz_loc
        col0 = lax.axis_index("np1") * nkx_loc
        d = 0.5 * (jnp.take(d, iy, axis=1) + jnp.take(d, iy_mirror, axis=1))
        local = kz_idx - row0
        mine = (local >= 0) & (local < nkz_loc)
        d = jnp.take(d, jnp.clip(local, 0, nkz_loc - 1), axis=2)
        d = jnp.where(mine[None, None, :, None], d, 0.0)
        local = kx_idx - col0
        mine = (local >= 0) & (local < nkx_loc)
        d = jnp.take(d, jnp.clip(local, 0, nkx_loc - 1), axis=3)
        d = jnp.where(mine[None, None, None, :], d, 0.0)
        return lax.psum(d, ("np0", "np1"))

    gathered = shard_map(
        _local,
        mesh=sharding.mesh,
        in_specs=(sharding.spec_vector_shard,),
        out_specs=P(None, None, None, None),
    )(density)
    n_pos = len(sel.kz_pos)
    # ``kz_neg`` pairs with ``kz_pos[1:]`` (every kept m >= 1).
    return gathered[:, :, :n_pos].at[:, :, 1:].add(gathered[:, :, n_pos:])


@partial(jit, static_argnames=("ref", "delta", "sel"))
def _twin_spectra3d_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    *,
    ref: bool,
    delta: bool,
    sel: CubeSelection,
) -> dict[str, Array]:
    r"""The 3-D componentwise energy cubes (module docstring).

    ``e`` is the difference field's `$(3, n_y, n_{k_z}, n_{k_x})$`
    cube and ``r`` the reference state's, each traced only when asked
    for: they go to two directories on two cadences
    (``twin.it_spectra3d`` / ``twin.it_spectra3d_ref``, the second
    defaulting to the first).  Same density as :func:`twin_yspectra`,
    so the full-sampling cube summed over either wavenumber is that
    stream's marginal, folded in `$y$`.
    """
    out: dict[str, Array] = {}
    if delta:
        out["e"] = _cube_replicated(
            _energy_density(state2 - state1, fourier_), sel
        )
    if ref:
        out["r"] = _cube_replicated(_energy_density(state1, fourier_), sel)
    return out


def twin_spectra3d(
    state1: Array,
    state2: Array,
    sel: CubeSelection,
    *,
    ref: bool = True,
    delta: bool = True,
) -> dict[str, Array]:
    """Wrapper around ``_twin_spectra3d_jit`` binding the singletons."""
    return _twin_spectra3d_jit(
        state1, state2, fourier, flow, ref=ref, delta=delta, sel=sel
    )


# ── Wall-normal-resolved spectral budget ─────────────────────────────

#: ``twin_ybudget`` term names, in stored order, one set per budget
#: form (module docstring, "Two budget forms").  The **convective**
#: set is the default, and its sums over the three bins' `$k$`-sets
#: return the paper's (2.7)-(2.9) (the 12 + 12 terms of (2.11)-(2.16)
#: dissolve into it); the **rotational** set is
#: ``twin.rotational_ybudget``.  In both, ``V`` is the viscous term in
#: the operator (discrete-Laplacian) form -- the one that makes the
#: budget close, matching ``twin_budget``'s ``eps_*`` -- and ``eps``
#: its positive-definite pseudo-dissipation companion; the two differ
#: by the wall-normal diffusion flux.  ``Wp`` is the work done by the
#: pressure *gradient*, not the mean-mode driving -- though at
#: `$(0,0)$` it is exactly that (:func:`_driving_density`).
CONVECTIVE_TERMS: tuple[str, ...] = (
    "P_U",
    "P_r",
    "T_ref",
    "T_self",
    "V",
    "eps",
    "Wp",
)

#: The rotational set.  ``P_lift`` is the convective ``P_U`` carried
#: unchanged, and sits **outside** the sum the other seven make.
ROTATIONAL_TERMS: tuple[str, ...] = (
    "P_U",
    "P_r",
    "T_vort",
    "T_self",
    "V",
    "eps",
    "Wp",
    "P_lift",
)


def ybudget_terms(rotational: bool) -> tuple[str, ...]:
    """The stored term names for the selected budget form."""
    return ROTATIONAL_TERMS if rotational else CONVECTIVE_TERMS


class _Sources(NamedTuple):
    r"""The products the budget and the pressure share.

    Built once per sample by :func:`_difference_sources`, whose field
    transforms are the budget's whole cost, so the pressure rides on
    them rather than repeating them.

    *advective* and *trailing* are the form-specific operands, each
    already signed so that its density is ``work(b)`` -- the leading
    terms of :func:`ybudget_terms` and (rotational only) the ones
    stored after ``Wp``.  Everything else is form-independent.
    """

    delta: Array
    advective: tuple[Array, ...]
    trailing: tuple[Array, ...]
    n_hat: Array  # the full nonlinear term, as it enters d_t(Du)
    div_n: Array  # its discrete divergence
    prof_dU: Array  # (3, Ny) mean-mode difference profile


def _cross(a: Array, b: Array) -> Array:
    r"""`$\mathbf{a} \times \mathbf{b}$`, fused per component.

    One ``jnp.array`` expression per output component (the
    :func:`dnsjax.rhs._fused_nonlinear` shape), so no intermediate
    concatenation or scatter kernel is emitted.  Either operand may be
    a `$(3, N_y, 1, 1)$` profile broadcast over the Fourier axes; such
    a product is mode-diagonal and therefore transform-free.

    Deliberately not
    :func:`~dnsjax.geometries.wall_bounded._base.base_flow_coupling`:
    that helper returns the *sum* `$\mathbf{u}\times\nabla\times
    \mathbf{U} + \mathbf{U}\times\boldsymbol{\omega}$` of two cross
    products, and the budget needs them as separate terms.  Expressing
    it through this one would split its single fused stack in two on
    the solver's corrector path.
    """
    return jnp.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def _difference_sources(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    rotational: bool,
) -> _Sources:
    """Evaluate the difference field's nonlinear term, in either form.

    *rotational* is resolved at trace time (it reaches here as a
    ``static_argnames`` entry of :func:`_twin_ybudget_jit`), so only
    the selected branch is ever traced.
    """
    if rotational:
        return _rotational_sources(state1, state2, fourier_, flow_)
    return _convective_sources(state1, state2, fourier_, flow_)


def _convective_sources(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> _Sources:
    r"""The convective nonlinear term (the default form).

    Twenty-four ``spec_to_phys`` (the two advectors and the two
    gradient sets) and nine back -- fewer than the 69 the three-bin
    :func:`twin_budget` needs, because binning no longer forces a
    separate physical product per bin pair.  The two gradient sets are
    formed **one at a time** -- the reference's only consumer is
    `$q_p$`, and an optimization barrier keeps XLA to that order -- so
    the live padded physical set is 6 + 9 + 3 with the product being
    formed, not 6 + 9 + 9 + 3; the module docstring's "Memory" section
    has what the program measures.

    Three exact simplifications, all of them of the same mean-mode
    kind (`$U^{(1)}_y = \Delta U_y = 0$` by continuity plus no-slip,
    so a mean profile advects with `$\mathrm{i}(k_xP_x + k_zP_z)$`
    alone -- mode-diagonal, and therefore free of any transform):

    - **The advectors are mean-free.**  The mean part of either
      advector contributes `$\mathrm{i}(k_xU_x + k_zU_z)|\Delta\hat
      u|^2$` to the energy, purely imaginary and so exactly zero per
      mode, which keeps a large, exactly-cancelling term out of the
      transforms.  The *pressure* needs the full term, so ``n_hat``
      adds the mean-mode pieces back spectrally, at no transform cost.
    - **The production advector splits.**  `$(\Delta\mathbf{u}\cdot
      \nabla)\mathbf{u}'^{(1)} = (\Delta\mathbf{u}'\cdot\nabla)
      \mathbf{u}'^{(1)} + (\Delta\mathbf{U}\cdot\nabla)
      \mathbf{u}'^{(1)}$`, and the second half is mode-diagonal.
      That is what makes the advector set **two** fields rather than
      three: `$\Delta\mathbf{u}$` never needs a physical form of its
      own.  The split is exact -- a `$k = 0$` times fluctuation
      product cannot alias -- and it drops three forward transforms
      *and* three live padded fields.
    - **The moving frame is mode-diagonal too.**  In a moving frame
      (``phys.u_grid``) the solver's RHS carries
      `$+\,\mathrm{i}k_xU_{grid}\mathbf{u}'$`, so the difference
      field's `$\hat{\mathcal N}$` carries `$+\,\mathrm{i}k_x
      U_{grid}\Delta\hat{\mathbf{u}}$`.  It contributes nothing to the
      energy densities (imaginary again) but it *is* part of the
      pressure's source, and is added here so ``div_n`` and
      ``n_hat[1]`` match the solver term for term under either
      ``res.consistent_imm`` (module docstring, "Frame invariance").
    """
    kx, kz = fourier_.kx, fourier_.kz
    d1 = flow_.D1
    m_mean, _, _ = component_masks(fourier_)

    delta = state2 - state1
    # `$\mathbf{u}'^{(1)} = \mathbf{u}^{(1)} - \mathbf{U}^{(1)}$`:
    # the laminar base flow lives entirely at `$(0,0)$`, so the
    # reference fluctuation is ``state1`` with its mean mode removed
    # -- no base-flow arithmetic enters.
    ref_f = state1 * ~m_mean
    delta_f = delta * ~m_mean

    mean_delta, mean_ref = extract_mean_modes(delta, state1)
    prof_dU = mean_delta.real
    prof_rU = mean_ref.real + flow_.base_flow[:, :, 0, 0]
    dy_rU = jnp.einsum("ij,cj->ci", d1, prof_rU)

    def grad_spec(c: Array) -> Array:
        r"""Nine rows; row ``3 * d + i`` is `$\partial_d c_i$`."""
        return jnp.concatenate(
            [1j * kx * c, apply_y_matrix(d1, c), 1j * kz * c], axis=0
        )

    def advect(b_phys: Array, grad_phys: Array) -> Array:
        r"""`$(\mathbf{b}\cdot\nabla)\mathbf{c}$`, back to spectral."""
        return chunked_transform(
            phys_to_spec,
            jnp.stack(
                [
                    sum(b_phys[j] * grad_phys[3 * j + i] for j in range(3))
                    for i in range(3)
                ]
            ),
        )

    adv = chunked_transform(
        spec_to_phys, jnp.concatenate([ref_f, delta_f], axis=0)
    )
    # One gradient set at a time: the reference's only consumer is
    # ``q_p``, so finishing it first retires those nine padded fields
    # before the difference's nine exist.
    grad_ref = chunked_transform(spec_to_phys, grad_spec(ref_f))
    # `$(\Delta\mathbf{u}\cdot\nabla)\mathbf{u}'^{(1)}$`, split so the
    # advector is the mean-free half (docstring).
    q_p = advect(adv[3:6], grad_ref) + mean_advect(prof_dU, ref_f, kx, kz)
    # Statement order alone does not get that order: ``grad_del``
    # depends on nothing ``q_p`` does, and XLA schedules the two
    # nine-field transforms together.  The barrier makes the second
    # wait for ``q_p`` -- a fifth off this program's peak, and the
    # same numbers (docstring).
    q_p, adv, delta = lax.optimization_barrier((q_p, adv, delta))

    grad_del = chunked_transform(spec_to_phys, grad_spec(delta))
    q_tr = advect(adv[0:3], grad_del)
    q_ts = advect(adv[3:6], grad_del)
    q_pu = delta[1] * dy_rU[:, :, None, None]

    n_hat = -(
        q_p
        + q_tr
        + q_ts
        + q_pu
        + mean_advect(prof_rU, delta, kx, kz)
        + mean_advect(prof_dU, delta, kx, kz)
    )
    u_grid = derived_params.u_grid
    if u_grid:
        # The moving frame's convective term (docstring): energy-inert,
        # part of the pressure's source.
        n_hat = n_hat + (1j * u_grid) * kx * delta
    # The solver's own discrete divergence
    # (``cartesian._imm_iteration_vp`` stage 1).
    div_n = (
        1j * kx * n_hat[0] + apply_y_matrix(d1, n_hat[1]) + 1j * kz * n_hat[2]
    )
    # Advection enters `$\partial_t\Delta\mathbf{u}$` with a minus, so
    # the operands are handed over negated and ``work`` stays uniform
    # across the two forms.  Sign flips are exact.
    return _Sources(
        delta,
        (-q_pu, -q_p, -q_tr, -q_ts),
        (),
        n_hat,
        div_n,
        prof_dU,
    )


def _rotational_sources(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> _Sources:
    r"""The rotational nonlinear term (``twin.rotational_ybudget``).

    Both states carry the perturbation about the *same* base flow
    `$\mathbf{U}_b$`, so differencing :mod:`dnsjax.rhs`'s rotational
    form collapses to three terms in the *total* reference fields,

    .. math::
        \Delta\mathcal{N} = \mathbf{u}^{(1)}\times\Delta
        \boldsymbol{\omega} + \Delta\mathbf{u}\times
        \boldsymbol{\omega}^{(1)} + \Delta\mathbf{u}\times\Delta
        \boldsymbol{\omega} ,

    and splitting each reference field into its mean profile and its
    fluctuation gives the five products below.  This is the term the
    solver itself integrates, so ``n_hat`` equals
    ``cartesian._get_rhs(state2) - _get_rhs(state1)`` to rounding --
    a property the convective form cannot have, and the guard
    ``tests/test_twin_unit.py::test_nonlinear_matches_solver``.

    **Two are mode-diagonal, hence transform-free.**  A `$k = 0$`
    profile times a fluctuation cannot alias, so evaluating
    `$\mathbf{U}^{(1)}\times\Delta\boldsymbol{\omega}$` and
    `$\Delta\mathbf{u}\times\boldsymbol{\Omega}^{(1)}$` spectrally is
    *equal*, not merely equivalent, to the padded physical-space form.
    `$\overline{\boldsymbol{\omega}'^{(1)}} = \nabla\times\overline{
    \mathbf{u}'^{(1)}}$` because the curl is mode-diagonal, so one
    :func:`~dnsjax.geometries.wall_bounded._base.extract_mean_modes`
    collective yields all three mean profiles -- the ``cartesian._l_bf``
    precedent.

    **`$\Delta\mathbf{u}\times\boldsymbol{\Omega}^{(1)}$` is kept in
    `$\hat{\mathcal N}$` and dropped from the densities.**  Its
    contraction is `$\mathrm{Re}\{\hat{\mathbf{a}}^*\cdot(\hat{
    \mathbf{a}}\times\boldsymbol{\Omega})\} = \mathrm{Re}\{
    \boldsymbol{\Omega}\cdot(\hat{\mathbf{a}}^*\times\hat{\mathbf{a}}
    )\}$`, and `$\hat{\mathbf{a}}^*\times\hat{\mathbf{a}}$` is purely
    imaginary, so it is exactly zero **per mode** for a real profile
    -- the same imaginary-contraction omission the convective form
    makes for its mean advectors.

    **Cost: 21 field transforms against the convective 33** -- twelve
    ``spec_to_phys`` (`$\Delta\mathbf{u}$`, `$\Delta\boldsymbol{
    \omega}$`, `$\boldsymbol{\omega}_f^{(1)}$`, `$\mathbf{u}_f^{(1)}$`)
    and nine back, one per product.  The statement order below keeps
    at most **12** padded fields live, the product being formed
    included (the convective order's is 18 by that count): the
    two `$\Delta\mathbf{u}$` products are formed and retired first,
    which lets `$\boldsymbol{\omega}_f^{(1)}$` and
    `$\Delta\mathbf{u}$` go before `$\mathbf{u}_f^{(1)}$` arrives.
    That is a count of live fields, not the program's peak, which its
    one nine-field transform sets: 33 padded components measured,
    against the convective 37 (module docstring, "Memory").
    """
    kx, kz = fourier_.kx, fourier_.kz
    d1 = flow_.D1
    m_mean, _, _ = component_masks(fourier_)

    delta = state2 - state1
    omega_d = _curl_fn(delta, fourier_, flow_)
    omega_r = _curl_fn(state1, fourier_, flow_)

    # One collective for the three mean modes (docstring).
    mean_delta, mean_ref, mean_om = extract_mean_modes(delta, state1, omega_r)
    prof_dU = mean_delta.real
    prof_rU = mean_ref.real + flow_.base_flow[:, :, 0, 0]
    prof_rom = mean_om.real + flow_.curl_base_flow[:, :, 0, 0]
    ref_f = state1 * ~m_mean
    omg_f = omega_r * ~m_mean

    # Mode-diagonal, transform-free (docstring).
    q_pu = _cross(prof_rU[:, :, None, None], omega_d)
    q_om = _cross(delta, prof_rom[:, :, None, None])
    dy_rU = jnp.einsum("ij,cj->ci", d1, prof_rU)
    q_lift = delta[1] * dy_rU[:, :, None, None]

    def back(product: Array) -> Array:
        r"""One padded physical product back to spectral."""
        return chunked_transform(phys_to_spec, product)

    phys = chunked_transform(
        spec_to_phys, jnp.concatenate([delta, omega_d, omg_f], axis=0)
    )
    q_tv = back(_cross(phys[0:3], phys[6:9]))
    q_ts = back(_cross(phys[0:3], phys[3:6]))
    # `$\mathbf{u}_f^{(1)}$` last: by here `$\boldsymbol{\omega}_f^{(1)}$`
    # and `$\Delta\mathbf{u}$` are dead, so the peak stays at 12.
    q_pr = back(_cross(chunked_transform(spec_to_phys, ref_f), phys[3:6]))

    n_hat = q_pu + q_pr + q_om + q_tv + q_ts
    u_grid = derived_params.u_grid
    if u_grid:
        n_hat = n_hat + (1j * u_grid) * kx * delta
    div_n = (
        1j * kx * n_hat[0] + apply_y_matrix(d1, n_hat[1]) + 1j * kz * n_hat[2]
    )
    # The rotational term enters `$\partial_t\Delta\mathbf{u}$` with a
    # plus; ``P_lift`` is the classical `$-\langle\ldots\rangle$` and
    # is negated here for the same uniformity.
    return _Sources(
        delta, (q_pu, q_pr, q_tv, q_ts), (-q_lift,), n_hat, div_n, prof_dU
    )


def _ybudget_densities(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    pressure: DifferencePressure,
    rotational: bool,
) -> Array:
    r"""The per-mode budget densities, stacked.

    Returns a real ``(n_terms, N_y, N_{k_z}, N_{k_x})`` array in
    :func:`ybudget_terms` order, component-summed, each already
    carrying ``k_metric`` and divided by ``volume_fac`` -- so summing
    over `$(k_z, k_x)$` and integrating with ``y_weights`` gives the
    rate the corresponding scalar diagnostic reports.
    """
    k2, k_metric = fourier_.k2, fourier_.k_metric
    vf = derived_params.volume_fac
    nu = 1.0 / params.phys.re
    d1, d2 = flow_.D1, flow_.D2
    src = _difference_sources(state1, state2, fourier_, flow_, rotational)
    delta = src.delta

    def work(b: Array) -> Array:
        r"""`$+\sigma_{k_x}\sum_i\mathrm{Re}\{\Delta\hat u_i^* b_i\}/V$`.

        The rate at which a term *b* of `$\partial_t\Delta\hat{
        \mathbf{u}}$` feeds the modal energy.  Both source builders
        hand over operands already signed for it, so this is the one
        contraction the whole budget uses.
        """
        return jnp.sum((jnp.conj(delta) * b).real, axis=0) * (k_metric / vf)

    # Viscous: the operator form (closure-consistent, matching
    # ``twin_budget``'s ``eps_*``) and the positive-definite
    # pseudo-dissipation `$\nu|\nabla\Delta u|^2$`; their difference
    # is the wall-normal diffusion flux.
    visc = work(apply_y_matrix(d2, delta) - k2 * delta) * nu
    dy_delta = apply_y_matrix(d1, delta)
    eps = (
        jnp.sum(
            (dy_delta * jnp.conj(dy_delta)).real
            + k2 * (delta * jnp.conj(delta)).real,
            axis=0,
        )
        * nu
        * (k_metric / vf)
    )

    p_hat = pressure.solve(delta, src.div_n, src.n_hat[1], flow_, fourier_)
    wp = pressure.work_density(delta, p_hat, flow_, fourier_)
    wp = (
        wp
        + _driving_density(src.prof_dU, flow_) * component_masks(fourier_)[0]
    )

    return jnp.stack(
        [
            *(work(b) for b in src.advective),
            visc,
            eps,
            wp,
            *(work(b) for b in src.trailing),
        ]
    )


def _driving_density(prof_dU: Array, flow_: object) -> Array:
    r"""Mean-mode driving work density, shape ``(N_y, 1, 1)``.

    The `$(0,0)$` mode's pressure term is not the fluctuating pressure
    (which does no work there: `$\Delta\hat v_{00}\equiv 0$` and the
    horizontal gradients vanish) but the applied driving.  With
    `$\Pi$` the mean pressure gradient (the module docstring's
    "Mean-mode driving" fixes the sign) the force is `$-\Pi$`, so the
    density is
    `$-\Delta\Pi_s \Delta U_s(y) - \Delta\Pi_n \Delta U_n(y)$` -- and
    what ``mean_driving_from_profile`` returns is `$-\Delta\Pi$`
    already, keyed ``-dPds'`` / ``-dPdn'``, so the code below adds it
    with a plus.  Its wall-normal average (the stored density's
    `$y$`-integral, ``volume_fac`` being divided out) is
    `$-\Delta\Pi \cdot U_\text{bulk}(\Delta u) = 0$`
    exactly -- at constant flow rate both members hold the same bulk,
    at fixed pressure gradient `$\Delta\Pi = 0$`.  Under a held bulk
    its *density* is not zero, so a `$y$`-resolved budget needs it;
    at a fixed gradient the density vanishes identically too, and
    this returns exact zeros.

    `$-\Delta\Pi$` is the **wall-shear inference** of the two members'
    driving, differenced; that is deliberately the better budget
    partner than the corrector's applied value (the `$t = t_0$`
    reasoning in :mod:`dnsjax.__main__`).  Returns exact zeros when no
    driving constraint is active.

    Taken from the **difference mean profile alone**, not from the two
    states: the inference is linear, so the difference of the two
    members' driving is exactly the driving evaluated on
    `$\Delta\mathbf{U}$` -- hence
    :func:`~dnsjax.geometries.wall_bounded.cartesian.mean_driving_from_profile`
    on ``prof_dU``, which :func:`_difference_sources` has already
    extracted.  Calling ``mean_driving`` on each state instead costs two
    further ``extract_mean_mode`` collectives per sample, on top of the
    one already spent on the same two mean modes (separate
    ``shard_map`` regions, so XLA cannot merge them) -- a third of this
    sample's collectives for nothing, and the ``psum`` is
    latency-bound:
    :func:`~dnsjax.geometries.wall_bounded._base.extract_mean_modes`.
    """
    drive = mean_driving_from_profile(prof_dU, flow_)
    cos_t, sin_t = derived_params.cos_tilt, derived_params.sin_tilt
    dens = jnp.zeros_like(prof_dU[0])
    if DRIVING_KEY_S in drive:
        dens = dens + drive[DRIVING_KEY_S] * (
            prof_dU[0] * cos_t + prof_dU[2] * sin_t
        )
    if DRIVING_KEY_N in drive:
        dens = dens + drive[DRIVING_KEY_N] * (
            -prof_dU[0] * sin_t + prof_dU[2] * cos_t
        )
    return (dens / derived_params.volume_fac)[:, None, None]


@partial(jit, static_argnames=("rotational", "x0", "marginals", "cube"))
def _twin_ybudget_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    pressure: DifferencePressure,
    *,
    rotational: bool,
    x0: bool,
    marginals: bool = True,
    cube: CubeSelection | None = None,
) -> dict[str, Array]:
    r"""Wall-normal-resolved spectral budget (module docstring).

    Returns ``<term>_<suffix>`` for each name in
    :func:`ybudget_terms` and each in :func:`marginal_suffixes`, the
    marginals of :func:`_marginals_replicated`.  Every array is a
    `$y$`-density: integrate with ``flow.y_weights`` for the
    per-`$k$` rate, and sum over `$k$` for the corresponding
    volume-averaged rate.

    *rotational* and *x0* are both **static**, like ``bins`` on
    :func:`_twin_energies_jit` and ``ref`` on
    :func:`_twin_yspectra_jit`.  *rotational* selects the budget form,
    so only one of the two source builders is ever traced, and the
    term list it names is what the stream's records are shaped by;
    *x0* (``twin.x0_planes``) selects the suffix list, and with it off
    the `$k_x = 0$` plane is never formed.

    *marginals* and *cube* select the outputs, both static: the 2-D
    marginals above, and under *cube* the 3-D budget cube ``cube``,
    `$(n_{terms}, n_y, n_{k_z}, n_{k_x})$` on that selection
    (:func:`_cube_replicated`).  A sample on which both streams are due
    builds the densities -- the whole cost -- once.  The cube is
    convective-only (the rotational form is being retired, and the
    ``[twin]`` validation refuses the pair).
    """
    stacked = _ybudget_densities(
        state1, state2, fourier_, flow_, pressure, rotational
    )
    out: dict[str, Array] = {}
    if marginals:
        reduced = _marginals_replicated(stacked, x0=x0)
        for i, name in enumerate(ybudget_terms(rotational)):
            for suf, value in reduced.items():
                out[f"{name}_{suf}"] = value[i]
    if cube is not None:
        if rotational:
            raise ValueError("the 3-D budget cube is convective-only")
        out["cube"] = _cube_replicated(stacked, cube)
    return out


def twin_ybudget(
    state1: Array,
    state2: Array,
    pressure: DifferencePressure,
    *,
    rotational: bool = False,
    x0: bool = False,
    marginals: bool = True,
    cube: CubeSelection | None = None,
) -> dict[str, Array]:
    """Wrapper around ``_twin_ybudget_jit`` binding the singletons."""
    return _twin_ybudget_jit(
        state1,
        state2,
        fourier,
        flow,
        pressure,
        rotational=rotational,
        x0=x0,
        marginals=marginals,
        cube=cube,
    )


@jit
def _difference_pressure_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    pressure: DifferencePressure,
) -> tuple[Array, Array]:
    r"""`$(\Delta\hat{\mathbf{u}}, \Delta\hat{p})$`, the static pair.

    The convective sources and the solve of the budget's ``Wp``, so the
    reduced difference snapshots (``twin.it_lowres_delta``) store
    exactly the pressure the budget's work is computed from, formed
    from `$\Delta\mathbf{u}$` directly rather than as a difference of
    two `$O(1)$` pressures.
    """
    src = _convective_sources(state1, state2, fourier_, flow_)
    return src.delta, pressure.solve(
        src.delta, src.div_n, src.n_hat[1], flow_, fourier_
    )


def difference_pressure(
    state1: Array, state2: Array, pressure: DifferencePressure
) -> tuple[Array, Array]:
    """Wrapper around ``_difference_pressure_jit`` binding the singletons."""
    return _difference_pressure_jit(state1, state2, fourier, flow, pressure)


@partial(jit, static_argnames=("rotational",))
def _twin_pressure_check_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    pressure: DifferencePressure,
    *,
    rotational: bool,
) -> dict[str, Array]:
    r"""Residuals of the difference-pressure solve.

    Built from the *same* :func:`_difference_sources` the budget uses,
    so nothing here can drift from what ``Pi`` was computed with.
    Returns:

    - ``poisson``: the interior Poisson residual
      `$(D_2 - k^2)\Delta\hat p - \widehat{\nabla\cdot\mathcal N}$`,
      machine zero;
    - ``closure``: `$(D_1 \partial_t \Delta\hat v)|_w$`, the wall
      condition that *was* imposed -- machine zero at every mode but
      `$(0,0)$`, where the influence matrix is structurally singular
      and `$\Delta\hat v \equiv 0$` makes it vacuous;
    - ``neumann``: `$(D_1\Delta\hat p - \hat{\mathcal N}_y
      - Re^{-1}D_2\Delta\hat v)|_w$`, the analytic condition the IMM
      closure declines to impose -- a wall-normal truncation
      diagnostic that must shrink with ``res.ny``, not an error;
    - ``n_hat``, ``div_n``, ``dy_dtv``: the nonlinear term, the
      Poisson source and `$\partial_y\,\partial_t\Delta\hat v$` --
      what the residuals above are measured *against*, returned so a
      caller can normalise by them, and so ``n_hat`` can be pinned
      against the solver's own RHS.  All **full-size**
      `$(3, N_y, N_{k_z}, N_{k_x})$` / `$(N_y, N_{k_z}, N_{k_x})$`
      complex arrays -- this entry point is for tests, not for a
      cadenced stream.
    """
    src = _difference_sources(state1, state2, fourier_, flow_, rotational)
    delta = src.delta
    p_hat = pressure.solve(delta, src.div_n, src.n_hat[1], flow_, fourier_)
    lap = apply_y_matrix(flow_.D2, p_hat) - fourier_.k2 * p_hat
    dtv = (
        src.n_hat[1]
        - apply_y_matrix(flow_.D1, p_hat)
        + (apply_y_matrix(flow_.D2, delta[1]) - fourier_.k2 * delta[1])
        / params.phys.re
    )
    d1b = flow_.D1_bnd
    return {
        "poisson": (lap - src.div_n)[1:-1],
        "closure": jnp.stack(
            [
                jnp.einsum("j,jzx->zx", d1b[0], dtv),
                jnp.einsum("j,jzx->zx", d1b[-1], dtv),
            ]
        ),
        "neumann": pressure.neumann_residual(
            delta, p_hat, src.n_hat[1], flow_
        ),
        "n_hat": src.n_hat,
        "div_n": src.div_n,
        "dy_dtv": apply_y_matrix(flow_.D1, dtv),
    }


def twin_pressure_check(
    state1: Array,
    state2: Array,
    pressure: DifferencePressure,
    *,
    rotational: bool = False,
) -> dict[str, Array]:
    """Wrapper around ``_twin_pressure_check_jit`` binding the singletons."""
    return _twin_pressure_check_jit(
        state1, state2, fourier, flow, pressure, rotational=rotational
    )
