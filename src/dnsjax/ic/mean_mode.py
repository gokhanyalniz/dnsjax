r"""Mean-mode `$(k_x, k_z) = (0, 0)$` perturbation constraints.

A perturbation of the mean mode is not free: the `$x$`-`$z$` averaged
momentum balance ties the mean profile to the boundary conditions and
to the driving.  This module turns those relations into the
**discrete, homogeneous** conditions a mean-mode perturbation must
satisfy, and projects a candidate profile onto the subspace they
define.  Cartesian only (plane-Couette / plane-Poiseuille); every
other flow defers ``init.random_mean_flow`` until its own laws are
worked out.

What the conditions deliver: a state and its perturbed copy share
every quantity the run **controls**, so their difference is an
initial-condition perturbation and nothing else --

- the mean pressure gradient `$\Pi$`, in **both** tangential
  directions and under **every** driving.  It is imposed and shared
  where the driving fixes it, and pinned to the state where the
  driving holds a bulk instead, which is the extra row of case B
  below;
- the bulk velocity, in each direction whose bulk the driving
  **holds**.  Where the bulk is free it is a response, not a control,
  and the pair's bulks differ exactly as any two states' would --
  deliberately unconstrained.

That is what a twin run (:mod:`dnsjax.twin.driver`) needs of
`$\delta$`, and it is what a solver run needs of a random IC built
around the laminar profile.

Derivation
----------
Write `$\langle\cdot\rangle$` for the `$x$`-`$z$` average, `$\nu =
1/Re$`, and let `$\Pi$` be the mean pressure gradient along a
homogeneous direction (`$\Pi = 0$` for plane-Couette).  Continuity plus
no-slip give `$\langle v\rangle = 0$`, so the mean-mode momentum
balance of a tangential component `$u$` is

.. math::
    \partial_t \langle u\rangle = \partial_y \tau - \Pi, \qquad
    \tau = -\langle v u\rangle + \nu\,\partial_y\langle u\rangle ,

with `$\tau$` the **total** shear of the **total** field.  Two
relations follow:

1. **Regardless of the driving.**  No-slip pins `$\langle u\rangle$`
   at each wall, so `$\partial_t\langle u\rangle|_{\pm 1} = 0$` and
   hence `$\partial_y\tau(\pm 1) = \Pi$` at **each wall
   independently**.  At a wall `$v = \partial_y v = 0$` (no-slip plus
   continuity), so `$\langle v u\rangle$` and
   `$\partial_y\langle v u\rangle$` both vanish there and the relation
   reduces to

   .. math:: \nu\,\partial_y^2\langle u\rangle(\pm 1) = \Pi .

   It holds at all times, for any driving, and it is the
   **first-order compatibility condition** between the initial data
   and the no-slip boundary condition: data violating it is
   inconsistent with `$\partial_t\langle u\rangle|_{\text{wall}} = 0$`
   and launches a singular near-wall adjustment layer.

2. **When the direction's bulk is held fixed**, integrating the
   balance across the channel (the `$\partial_y\tau$` term telescopes,
   the Reynolds stress vanishes at both walls) gives

   .. math::
       \Pi = \frac{\tau(+1) - \tau(-1)}{2}
           = \frac{\nu}{2}\Bigl[
             \partial_y\langle u\rangle(+1)
             - \partial_y\langle u\rangle(-1)\Bigr] ,

   which is the wall-shear inference
   :func:`dnsjax.geometries.wall_bounded.cartesian.mean_driving`
   already reports (up to its sign convention: that column is the
   applied forcing `$-\Pi$`).  `$\Pi$` is then a **response**: it is
   whatever the state's own wall shears demand, rather than a number
   the run imposes.

Measured, on a controlled pair -- plane-Poiseuille at ``re = 400``,
``ny = 65``, ``fd_order = 6``, ``constant_bulk_velocity``,
``dt = 2e-4``, an otherwise laminar state carrying one mean-mode
profile of peak ``0.05``.  Both profiles are **odd** in `$y$`, so both
have exactly zero bulk and (their `$\partial_y\delta$` being even)
equal wall shears -- both held-bulk rows are inert for both, and they
differ only in `$\partial_y^2\delta(\pm 1)$`, which is `$0$` for
`$\sin(\pi y)$` and `$\mp 6$` (scaled) for `$y(1-y^2)$`:

    t            0.0002   0.001    0.005    0.02
    compatible   1.9e-9   9.7e-9   4.8e-8   1.9e-7    (linear in t)
    violating    1.1e-6   3.5e-6   7.8e-6   1.5e-5    (~ sqrt(t))

as `$\tau'_{s,b}(t) - \tau'_{s,b}(0)$`.  The violating trace divided by
`$\sqrt{t}$` is flat to 1.5 % across two decades (1.10, 1.11, 1.10,
1.09 `$\times 10^{-4}$`) -- the `$\sqrt{\nu t}$` adjustment layer,
exactly -- while the compatible one is linear (its first step takes
1.00 % of the 100-step change, the uniform-rate value; the violating
one takes 6.9 %).  The excursion is 80-550x larger, worst at the
earliest times, and it contaminates a physical diagnostic: the applied
driving ``-dPds'`` peaks at ``3.7e-7`` for the violating start against
``2.0e-12`` -- machine zero -- for the compatible one, though both
hold the bulk at ``9e-18``.

Reduction to homogeneous conditions
-----------------------------------
The solver stores the perturbation `$u'$` about the laminar base flow
`$U$`, which satisfies both relations exactly -- and *discretely* so,
since `$D_2\,(1-y^2) = -2$` and `$D_2\,y = 0$` are exact for
``fd_order >= 2``.  Both relations are linear, so with `$\delta(y)$`
the perturbation of one tangential mean profile, and
`$\Delta\Pi$` the difference in mean pressure gradient it induces,
relation 1 becomes

.. math:: \nu\,\partial_y^2\delta(\pm 1) = \Delta\Pi .

`$\Delta\Pi = 0$` is imposed: a perturbation may not change the
gradient that drives the flow (see the header).  Relation 1 then
reads `$\partial_y^2\delta(\pm 1) = 0$` **whatever the driving** --
the factor `$\nu$` drops out with it, so the constraints are
**Reynolds-number free** and purely geometric/discrete.  What the
driving decides is only how `$\Delta\Pi = 0$` is *spelt*, since it
fixes whether `$\Pi$` is imposed or a response.

With the wall-normal grid ascending (index ``0`` is `$y = -1$`, index
``-1`` is `$y = +1$`; see ``build_cartesian_grid``):

**Case A** -- that direction carries no held mean, so `$\Pi$` is
imposed (zero for plane-Couette, the laminar value under
``constant_pressure_gradient``) and `$\Delta\Pi = 0$` needs no row of
its own:

.. math:: \delta(\pm 1) = 0, \qquad (D_2\delta)_0 = (D_2\delta)_{-1} = 0 .

**Case B** -- that direction's mean is held
(``phys.driving = "constant_bulk_velocity"`` streamwise,
``phys.block_mean_spanwise_velocity`` spanwise), so relation 2
evaluates `$\Pi$` on the state itself.  `$\Delta\Pi = 0$` becomes the
equal-wall-shear row, and the held bulk adds one more:

.. math::
    \delta(\pm 1) = 0, \qquad (D_2\delta)_0 = (D_2\delta)_{-1} = 0,
    \qquad (D_1\delta)_{-1} = (D_1\delta)_0, \qquad
    \mathbf{w}\cdot\delta = 0 ,

with `$\mathbf{w}$` the grid's own quadrature weights (the same ones
``_apply_bulk_corrections`` integrates with).  The two curvature rows
are **the same rows in both cases**; case B adds the two that keep
them true under a responding `$\Pi$`.

Anchors, measured at ``cgl``/``ny = 65``/``fd_order = 6``:
`$\sin(k\pi(y+1)/2)$` satisfies case A for every `$k$`;
`$\sin(m\pi y)$` satisfies case B (odd, so both extra rows are
inert).  The even quartic `$(1-y^2)(5-y^2)$` separates them --
machine-exact on the two curvature rows and `$O(1)$` on each of case
B's extras (``1.000`` and ``0.640``, its wall shears being `$\mp 8$`
and its bulk ``6.4``) -- which is what makes the four rows
independent statements.  Laminar plane-Poiseuille `$1-y^2$` violates
**every** row of both cases (``1, 1`` and ``1, 1, 1, 0.667``): legal
as a base flow, never as a perturbation.

The two directions are the **tilted** ones,
`$\delta_s = \delta_x\cos\theta + \delta_z\sin\theta$` and
`$\delta_n = -\delta_x\sin\theta + \delta_z\cos\theta$`, and they can
be in different cases (plane-Poiseuille driven at constant bulk with a
free spanwise mean).  The wall-normal component's mean mode is
identically zero by continuity and is never touched here.

Enforcement: condition, do not project blindly
----------------------------------------------
The constrained subspace has codimension 2 (A) or 4 (B) inside the
wall-vanishing profiles.  Projecting a candidate along a *fixed*
complement is a poor generator: the natural complement
`$\mathrm{span}\{(1-y^2),\, y(1-y^2)\}$` is exactly where a windowed
random draw puts its near-wall curvature, so the "correction" comes out
an order of magnitude larger than the input and the profile degenerates
into a fixed polynomial shape.

:func:`project_profile` instead conditions the generator's **own
ensemble**.  The random Cartesian draw is `$\delta = M z$` with
`$M = \mathrm{diag}(\text{window})\,F$` (`$F$` the wall-normal
smoothness filter) and `$z \sim N(0, I)$`, so the distribution of
`$\delta$` given `$C\delta = 0$` is centred on

.. math::
    \delta' = \delta - K C^{\mathsf T}\,(C K C^{\mathsf T})^{-1}
              C \delta , \qquad K = M M^{\mathsf T} .

That is the orthogonal projection in the ensemble's own metric, and it
buys four things at once:

- it cannot amplify -- `$E\|\delta'\|^2 = \mathrm{tr}(KP) \le
  \mathrm{tr}(K) = E\|\delta\|^2$`;
- `$K$` low-passes the (spiky, one-sided) `$D_2$` wall stencils, so
  the correction is a smooth profile rather than a near-wall spike;
- `$K$`'s window factor makes the correction vanish at the walls
  exactly, so **no-slip survives for free** -- which is also why the
  two no-slip rows are deliberately *not* part of `$C$`
  (`$K e_0 = 0$` would make `$C K C^{\mathsf T}$` singular);
- it is idempotent, and for a profile that is already compatible in the
  continuum it moves only by the discrete truncation residual.

Functions
---------
constraint_rows:
    The `$(m, N_y)$` homogeneous constraint rows of one direction.
smoothing_kernel:
    The ensemble covariance `$K$` the projection is taken in.
project_profile:
    Condition one profile on ``constraint_rows``.
constraint_residuals:
    Per-relation *relative* residuals, for reporting and for guards.
build_cartesian_projector:
    The `$(\delta_x, \delta_z) \to (\delta_x, \delta_z)$` map for the
    live flow: tilt rotation plus each direction's own case.
check_cartesian_mean_profile:
    Human-readable violations of a candidate `$(0,0)$` column
    (``scripts/snapshot_perturb.py``'s guard); empty when compatible.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy import ndarray

from ..parameters import derived_params, params

# Relative floor added to the ensemble covariance's spectrum
# (:func:`smoothing_kernel`).  At an extreme ``random_smoothness`` the
# filter retains only four or five effective wall-normal modes (index
# 6 at ``s = 0.95``, index 4 at ``0.99``, ``ny = 65``), and the case-B
# constraint rows -- which need four -- go linearly dependent after
# smoothing, making ``C K C^T`` singular.  The floor is a white
# component *inside* the wall-vanishing space (it keeps the window
# factor), so it restores full rank without breaking no-slip.  At the
# default smoothness it sits 29 filter indices down and is invisible:
# switching it from ``1e-6`` moved the projected draw's retained
# energy by 0.1 % there, while cutting ``cond(C K C^T)`` by a decade
# at ``s = 0.95``-``0.99`` (1.3e12 -> 1.3e11) -- which is what the
# fourth row needed.
_KERNEL_FLOOR = 1e-5

# Residual below which a relation counts as satisfied.  A profile that
# is compatible in the *continuum* still has a truncation-level, not a
# machine-level, discrete residual, so this is measured rather than set
# to an epsilon.  Worst relative residual over the analytically
# compatible families -- ``sin(k pi (y+1)/2)`` for case A and
# ``sin(m pi y)`` for case B -- restricted to modes the grid resolves
# (>= 8 points per wavelength), maximised over ``fd_order`` in
# ``{4, 6, 8}``:
#
#     ny        17       25       33       65      129
#     cgl    2.9e-4   4.3e-5   1.1e-5   3.6e-7   1.1e-8
#     tanh   5.6e-4   8.2e-5   1.4e-4   4.9e-6   1.7e-6
#
# (the floor at large ``ny`` is roundoff, not truncation: the CGL
# near-wall ``D_2`` entries grow like ``ny^4``).  A profile that
# genuinely violates a relation scores `$O(1)$`: ``1 - y^2`` scores
# ``1.0`` on every row of both cases except case B's bulk row, at
# ``0.67``.  So this sits ~1 decade above the worst compatible case
# and ~2 below the violating one.  An *unresolved* profile (~2 points
# per wavelength) scores ``1.0`` and is rejected, correctly: the grid
# cannot carry it compatibly.  Do not tighten without re-measuring
# the table.
COMPAT_TOL = 5e-3

# Relation labels, in the row order of :func:`constraint_rows`.
_LABELS_A = ("d(tau)/dy(y=-1) = Pi", "d(tau)/dy(y=+1) = Pi")
_LABELS_B = (
    *_LABELS_A,
    "mean pressure gradient unchanged",
    "bulk velocity held",
)


def constraint_rows(
    D1: ndarray,
    D2: ndarray,
    y_weights: ndarray,
    *,
    fixed_bulk: bool,
) -> ndarray:
    r"""Homogeneous constraint rows for one tangential direction.

    Returns the `$(m, N_y)$` matrix `$C$` whose rows are the relations
    derived in the module docstring, in the order of :data:`_LABELS_A`
    / :data:`_LABELS_B`: `$m = 2$` in case A (*fixed_bulk* false) and
    `$m = 4$` in case B.  The first two rows -- relation 1, the
    vanishing wall curvature -- are the **same in both cases**; case B
    adds the two rows that make them so under a responding `$\Pi$`.
    The no-slip rows are **not** included (see
    the module docstring).  Rows are unscaled -- the projection is
    invariant under row scaling, and :func:`constraint_residuals`
    applies each relation's own physical scale.

    Parameters
    ----------
    D1, D2:
        Wall-normal first/second-derivative matrices, ``(Ny, Ny)``.
    y_weights:
        Wall-normal quadrature weights, ``(Ny,)`` -- the same ones the
        solver's bulk-velocity constraint integrates with.
    fixed_bulk:
        Whether this direction's mean is held (case B).
    """
    D1 = np.asarray(D1, dtype=np.float64)
    D2 = np.asarray(D2, dtype=np.float64)
    rows = [D2[0].copy(), D2[-1].copy()]
    if fixed_bulk:
        # Relation 2 evaluates the responding Pi on delta itself, so
        # "the perturbation leaves Pi alone" is the wall-shear
        # difference; relation 3 holds the bulk the driving holds.
        rows.append((D1[-1] - D1[0]) / 2.0)
        rows.append(np.asarray(y_weights, dtype=np.float64).copy())
    return np.stack(rows)


def constraint_residuals(
    delta: ndarray,
    D1: ndarray,
    D2: ndarray,
    y_weights: ndarray,
    *,
    fixed_bulk: bool,
) -> ndarray:
    r"""Per-relation **relative** residual of a candidate profile.

    Each relation is scaled by its own magnitude -- the curvature
    relations by `$\max|D_2\delta|$`, the pressure-gradient relation
    by `$\max|D_1\delta|$`, the bulk relation by
    `$\bigl(\sum_j |w_j|\bigr)\max|\delta|$`.  The result is
    dimensionless and `$O(1)$` for a grossly incompatible profile,
    truncation-level for one that is compatible in the continuum, so a
    single tolerance (:data:`COMPAT_TOL`) covers every resolution.

    An identically zero profile has zero residual by convention.
    """
    delta = np.asarray(delta, dtype=np.float64)
    C = constraint_rows(D1, D2, y_weights, fixed_bulk=fixed_bulk)
    raw = C @ delta

    d2 = np.asarray(D2, dtype=np.float64) @ delta
    scale = np.max(np.abs(d2))
    scales = [scale, scale]
    if fixed_bulk:
        d1 = np.asarray(D1, dtype=np.float64) @ delta
        scales.append(np.max(np.abs(d1)))
        w = np.asarray(y_weights, dtype=np.float64)
        scales.append(float(np.sum(np.abs(w))) * np.max(np.abs(delta)))
    out = np.array(
        [
            0.0 if s <= 0.0 else abs(r) / s
            for r, s in zip(raw, scales, strict=True)
        ]
    )
    return out


def smoothing_kernel(window: ndarray, wn_filter: ndarray) -> ndarray:
    r"""Ensemble covariance `$K = M M^{\mathsf T}$` of the random draw.

    *window* is the wall window applied to a tangential component
    (`$1 - y^2$`) and *wn_filter* the wall-normal smoothness filter
    `$F$` (``random_field._wall_normal_filter``), so that the generated
    profile is `$\delta = \mathrm{diag}(\text{window})\,F z$` with
    `$z \sim N(0, I)$`.  A relative floor :data:`_KERNEL_FLOOR` is
    added to the filter's spectrum before the window is applied.
    """
    w = np.asarray(window, dtype=np.float64)
    F = np.asarray(wn_filter, dtype=np.float64)
    core = F @ F.T
    n = core.shape[0]
    core = core + (_KERNEL_FLOOR * np.trace(core) / n) * np.eye(n)
    return (w[:, None] * core) * w[None, :]


def project_profile(delta: ndarray, C: ndarray, K: ndarray) -> ndarray:
    r"""Condition *delta* on `$C\delta = 0$` in the `$K$` metric.

    `$\delta' = \delta - K C^{\mathsf T} (C K C^{\mathsf T})^{-1}
    C\delta$` (module docstring).  Rows of `$C$` are normalised to unit
    length first -- the `$D_2$` wall rows carry `$1/h^2 \sim 10^5$`
    entries on a wall-clustered grid -- which leaves the projection
    itself unchanged.
    """
    delta = np.asarray(delta, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    norms = np.linalg.norm(C, axis=1)
    Cn = C / np.where(norms > 0.0, norms, 1.0)[:, None]
    KCt = np.asarray(K, dtype=np.float64) @ Cn.T  # (Ny, m)
    lam = np.linalg.solve(Cn @ KCt, Cn @ delta)
    return delta - KCt @ lam


def _tilt_cases() -> tuple[float, float, bool, bool]:
    """``(cos, sin, fixed_bulk_s, fixed_bulk_n)`` for the live flow."""
    return (
        derived_params.cos_tilt,
        derived_params.sin_tilt,
        params.phys.driving == "constant_bulk_velocity",
        params.phys.block_mean_spanwise_velocity,
    )


def build_cartesian_projector(
    D1: ndarray,
    D2: ndarray,
    y_weights: ndarray,
    window: ndarray,
    wn_filter: ndarray,
) -> Callable[[ndarray, ndarray], tuple[ndarray, ndarray]]:
    r"""Build the `$(0,0)$` projector for the live Cartesian flow.

    Returns a map `$(\delta_x, \delta_z) \to (\delta_x, \delta_z)$` on
    real ``(Ny,)`` profiles: rotate into the tilted `$(s, n)$` pair,
    condition each on its own case (streamwise held by
    ``phys.driving``, spanwise by ``phys.block_mean_spanwise_velocity``
    -- read from the live singletons), rotate back.  The kernel and
    both constraint sets are built once here, outside any per-mode
    loop.
    """
    cos_t, sin_t, bulk_s, bulk_n = _tilt_cases()
    K = smoothing_kernel(window, wn_filter)
    C_s = constraint_rows(D1, D2, y_weights, fixed_bulk=bulk_s)
    C_n = constraint_rows(D1, D2, y_weights, fixed_bulk=bulk_n)

    def project(dx: ndarray, dz: ndarray) -> tuple[ndarray, ndarray]:
        d_s = dx * cos_t + dz * sin_t
        d_n = -dx * sin_t + dz * cos_t
        d_s = project_profile(d_s, C_s, K)
        d_n = project_profile(d_n, C_n, K)
        return (d_s * cos_t - d_n * sin_t, d_s * sin_t + d_n * cos_t)

    return project


def check_cartesian_mean_profile(
    vec: ndarray,
    D1: ndarray,
    D2: ndarray,
    y_weights: ndarray,
) -> list[str]:
    r"""Violations of a candidate `$(0,0)$` column, one line each.

    *vec* is the ``(3, Ny)`` profile a caller means to add at the mean
    mode, in physical components `$(u_x, u_y, u_z)$`.  Checks, for the
    live flow: reality, a zero wall-normal component (continuity forces
    `$\langle v\rangle \equiv 0$`), no-slip, and each tilted
    direction's relations against :data:`COMPAT_TOL`.  Returns an empty
    list when the profile is compatible; used by
    ``scripts/snapshot_perturb.py``, which rejects rather than
    silently reshaping a profile its caller owns.
    """
    cos_t, sin_t, bulk_s, bulk_n = _tilt_cases()
    vec = np.asarray(vec)
    bad: list[str] = []

    scale = max(float(np.max(np.abs(vec))), 1.0)
    if np.iscomplexobj(vec):
        im = float(np.max(np.abs(vec.imag)))
        if im > 1e-13 * scale:
            bad.append(
                f"profile must be real (max|Im| = {im:.3e}); the mean "
                "mode is its own conjugate partner"
            )
        vec = vec.real
    if float(np.max(np.abs(vec[1]))) > 1e-13 * scale:
        bad.append(
            "wall-normal component must vanish at the mean mode "
            "(continuity with no-slip forces <v> = 0)"
        )
    wall = max(abs(float(vec[0, 0])), abs(float(vec[0, -1])))
    wall = max(wall, abs(float(vec[2, 0])), abs(float(vec[2, -1])))
    if wall > 1e-13 * scale:
        bad.append(f"no-slip violated at a wall (max|u'| = {wall:.3e})")

    d_s = vec[0] * cos_t + vec[2] * sin_t
    d_n = -vec[0] * sin_t + vec[2] * cos_t
    for name, d, fixed in (
        ("streamwise", d_s, bulk_s),
        ("spanwise", d_n, bulk_n),
    ):
        res = constraint_residuals(d, D1, D2, y_weights, fixed_bulk=fixed)
        labels = _LABELS_B if fixed else _LABELS_A
        for label, r in zip(labels, res, strict=True):
            if r > COMPAT_TOL:
                bad.append(
                    f"{name} mean profile violates '{label}': relative "
                    f"residual {r:.3e} > {COMPAT_TOL:.0e}"
                )
    return bad
