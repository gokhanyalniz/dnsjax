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

Component decomposition
-----------------------
Fields are split by their wall-parallel Fourier support into the
three components of the reference (mean / streak / streamwise-varying
triad):

- mean `$\Delta U$`: the `$(k_z, k_x) = (0, 0)$` mode
  (``fourier.mean_mask``, one-hot by the padding-slot invariant --
  see "Mean mode and padding modes" in
  ``geometries/wall_bounded/CLAUDE.md``);
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

  Two knobs trade against footprint if this program's peak ever
  binds, and they act on **different** transients -- neither is
  applied by default (both cost throughput, and the budget is a
  cadenced diagnostic):

  - ``solver.rhs_transform_chunks`` bounds the *transform-stage*
    transient inside one :func:`dnsjax.fft.chunked_transform` call
    (the padded intermediates of a 9-field batch), for the same
    69 single-field transforms in more, smaller dispatches.  It
    leaves the ~21 live fields alone.
  - Moving the advector transform *inside* the pair loop is what
    cuts those: the 9 cached `$\mathbf{b}$` fields become 3 live
    ones, peak 21 `$\to$` 15 (-29 %), for 84 transforms instead of
    69 (+22 %) since each of the 8 pairs then re-transforms its own
    advector.

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
plus the `$k_x = 0$` plane:

.. math::
    E_\Delta^x[\alpha](y, k_z) = \sum_{k_x} \hat{e}_\alpha , \qquad
    E_\Delta^z[\alpha](y, k_x) = \sum_{k_z} \hat{e}_\alpha , \qquad
    E_\Delta^{x0}[\alpha](y, k_z) = \hat{e}_\alpha(y, 0, k_z) ,

with `$\hat{e}_\alpha = \tfrac12 |\Delta\hat u_\alpha|^2$` per
velocity component.  **Energy first, then the sum over the other
wavenumber** -- not the energy of the averaged velocity.  Under the
``norm="forward"`` convention that sum *is* the streamwise average of
the energy, so it is the standard one-dimensional spectrum; it
closes (summing either marginal over its axis and integrating in
`$y$` returns `$E_\Delta$` exactly, where averaging first returns
only the `$k_x = 0$` content); and it *contains* the other reading,
which is exactly the stored `$k_x = 0$` plane.  That plane is also
what makes these a strict refinement rather than a replacement:

.. math::
    E_{\Delta U} = \textstyle\int \sum_\alpha E^{x0}_\alpha(y, 0),
    \quad
    E_{\Delta u_1} = \int \sum_\alpha \sum_{k_z>0} E^{x0}_\alpha ,
    \quad
    E_{\Delta u_2} = \int \sum_\alpha \sum_{k_z}
        (E^{x}_\alpha - E^{x0}_\alpha)

-- the three numbers of the old binning, now `$k_z$`-resolved, which
is why ``twin.bins`` can stay off.  The `$\pm k_z$` fold this
requires is not cosmetic: :func:`_fold_kz`.

Spectral budget
---------------
Contracting the difference momentum equation with
`$\sigma_{k_x}\Delta\hat{\mathbf{u}}^*$` at each mode gives one
production, one transfer, one viscous and one pressure term per
`$(y, k)$` -- and the paper's (2.7)-(2.9) are `$k$`-set sums of them,
so the 12 + 12 expansions of (2.11)-(2.16) have nothing left to say
and are not reproduced.  :data:`YBUDGET_TERMS`:

- ``P_U``: production against the reference mean profile,
  `$-\sigma_k \mathrm{Re}\{\Delta\hat u_i^* \Delta\hat v\}\,
  \partial_y U^{(1)}_i$` -- diagonal in `$k$`, no transform, and the
  paper's dominant long-time (lift-up) term now resolved in
  `$(y, k)$`;
- ``P_r``: production against the reference *fluctuation* gradients;
- ``T_ref`` / ``T_self``: transfer by the reference fluctuation and
  by the difference field's own advection.  Each sums to zero over
  all `$(y, k)$`, so they are pure redistribution -- interscale as
  well as wall-normal;
- ``V`` / ``eps``: the viscous term in the operator form (the one
  that closes -- "Dissipation form" above) and the positive-definite
  pseudo-dissipation `$\nu|\nabla\Delta\mathbf{u}|^2$`.  Their
  difference is the wall-normal diffusion flux;
- ``Pi``: the pressure work, from :mod:`dnsjax.twin.pressure`.

`$\sum_k \int$` of ``P_U + P_r`` and of ``-V`` reproduce
``twin_budget.dat``'s ``P_tot`` and ``eps_tot`` to rounding --
algebraic identities, the same Parseval sum regrouped.  The transfer
terms match their per-bin counterparts only up to the discrete
integration-by-parts residual that makes ``T_tot`` nonzero in the
first place (measured on the ladder in ``tests/test_twin_budget.py``).

The `$k$`-resolved budget is **cheaper** than the three-bin one: 33
field transforms against 69 (:func:`_difference_sources`), because
binning no longer forces a separate physical product per bin pair.
What it adds instead is the pressure -- one factored Poisson
operator held for the run.  Pressure is the one term the
volume-averaged budget omits for free and a localised one cannot;
:mod:`dnsjax.twin.pressure` has the whole argument.

Frame invariance
----------------
A moving frame (``phys.u_grid``, e.g. the plane-Poiseuille default
`$2/3$`) shifts the streamwise mean of *both* states by the same
constant, which cancels in `$\Delta\mathbf{u}$`; the only remaining
carrier is the `$U^{(1)}$` profile, which enters every budget term
through `$\partial_y$` alone.  All quantities here are therefore
frame-invariant and need no ``u_grid`` handling.

Mean-mode driving
-----------------
There is deliberately **no forcing column** in the budget, for either
driving knob.  ``_apply_bulk_corrections``
(:mod:`~dnsjax.geometries.wall_bounded.cartesian`) applies a *scalar*
body force on the `$(0,0)$` mode alone -- `$\pi_s$` under
``phys.driving = "constant_bulk_velocity"``, `$\pi_n$` under
``phys.block_mean_spanwise_velocity`` -- so its work on a field is
exactly (force) `$\times$` (that field's bulk velocity along the
forced direction).  On the *difference* field that is
`$\Delta\pi \cdot \mathrm{bulk}(\Delta u)$`, and every supported
setting annihilates one of the two factors, by a different mechanism:

- **force free** (``constant_pressure_gradient``, and plane-Couette,
  which carries no ``driving`` field at all): the applied force is the
  same constant in both runs, so `$\Delta\pi = 0$`.  Note this says
  nothing about `$\mathrm{bulk}(\Delta u)$`, which is genuinely
  non-zero here -- an undriven direction acquires a bulk velocity
  spontaneously, plane-Couette's streamwise one included.
- **bulk held**: both runs hold the *same* bulk value, so
  `$\mathrm{bulk}(\Delta u) = 0$` -- exactly, because the correction is
  a rank-1 algebraic projection satisfied at every corrector iterate,
  not a converged feedback loop.  Here `$\Delta\pi$` is the non-zero
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
"""

import importlib
from functools import partial
from typing import NamedTuple

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
from ..geometries.wall_bounded.cartesian import (
    DRIVING_KEY_N,
    DRIVING_KEY_S,
    Fourier,
    fourier,
    mean_driving,
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
        bx = prof[b][0][:, None, None]
        by = prof[b][1][:, None, None]
        bz = prof[b][2][:, None, None]
        adv = 1j * (bx * kx + bz * kz) * full[c] + by * apply_y_matrix(
            D1, full[c]
        )
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


@jit
def _twin_spectra_jit(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> dict[str, Array]:
    r"""``(k_z, k_x)`` energy spectra of the difference and reference.

    ``e_delta`` is the per-mode `$E_\Delta(k_z, k_x)$` and ``e_ref``
    the reference state's own spectrum (their ratio
    `$E_\Delta / 2 E^{(1)}$` is the offline decorrelation measure:
    fully decorrelated independent fields give 1).  True modes only
    (padding stripped); summing ``e_delta`` reproduces ``twin.dat``'s
    ``E_d`` to rounding (a ``tests/test_twin_unit.py`` guard).
    """
    n2 = params.res.nz - 1
    n3 = params.res.nx // 2
    w = flow_.y_weights
    k_metric = fourier_.k_metric
    delta = state2 - state1
    return {
        "e_delta": _mode_energy_replicated(delta, w, k_metric)[:n2, :n3],
        "e_ref": _mode_energy_replicated(state1, w, k_metric)[:n2, :n3],
    }


def twin_spectra_2d(state1: Array, state2: Array) -> dict[str, Array]:
    """Wrapper around ``_twin_spectra_jit`` binding the singletons."""
    return _twin_spectra_jit(state1, state2, fourier, flow)


# ── Wall-normal-resolved marginal spectra ────────────────────────────


def marginal_bin_counts() -> tuple[int, int]:
    r"""``(n_{k_z}, n_{k_x})`` of the folded marginal axes.

    Both are one-sided: `$n_z/2$` and `$n_x/2$` bins, carrying
    integer wavenumbers `$0, 1, \dots$` (``harmonics.real_harmonics``
    of each axis' full count).  See :func:`_fold_kz` for why the
    `$k_z$` axis is folded rather than stored two-sided.
    """
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


def _marginals_replicated(density: Array) -> tuple[Array, Array, Array]:
    r"""The three marginals of a per-mode density, replicated.

    *density* is a **real** `$(C, N_y, N_{k_z}, N_{k_x})$` array in
    the spectral layout, already carrying its ``k_metric`` weight and
    any prefactor -- so that summing it over `$(k_z, k_x)$` and
    integrating over `$y$` with ``y_weights`` reproduces the scalar
    the same quantity gives in ``twin.dat`` / ``twin_budget.dat``.
    The leading axis is free: three velocity components for the
    energies, one per term for the budget.

    Returns ``(m_x, m_z, m_x0)``, each replicated with the spectral
    padding stripped:

    - ``m_x`` `$(C, N_y, n_z/2)$`: summed over `$k_x$` and folded onto
      `$|k_z|$` (:func:`_fold_kz`) -- the `$x$`-averaged spectrum;
    - ``m_z`` `$(C, N_y, n_x/2)$`: summed over `$k_z$` -- the
      `$z$`-averaged spectrum;
    - ``m_x0`` `$(C, N_y, n_z/2)$`: the `$k_x = 0$` plane alone,
      folded the same way -- the spectrum *of the streamwise-averaged
      field*, which is what recovers the `$\Delta U$` / `$\Delta u_1$`
      / `$\Delta u_2$` binning from ``m_x`` (module docstring).

    Each device reduces its own `$(k_z, k_x)$` tile, scatters the
    three blocks into zero global-shape arrays at its mesh position,
    and one ``psum`` over both mesh axes assembles the replicated
    result -- the pattern of :func:`_mode_energy_replicated`, and
    required for the same reason (the writer's rank-0 host transfer
    needs a fully-addressable array).  The three blocks share one
    collective: unlike the two `$(k_z, k_x)$` planes there, they are
    reductions of the *same* field pass, so there is nothing to gain
    by splitting them.  The fold runs **after** the ``psum``: the
    `$\pm k_z$` partners live on different ``np0`` devices.
    """
    nz_spec, nx_spec = sharding.spec_shape[1], sharding.spec_shape[2]

    def _local(d: Array) -> Array:
        nkz_loc, nkx_loc = d.shape[2], d.shape[3]
        row0 = lax.axis_index("np0") * nkz_loc
        col0 = lax.axis_index("np1") * nkx_loc
        c, ny = d.shape[0], d.shape[1]
        zeros_z = jnp.zeros((c, ny, nz_spec), dtype=d.dtype)
        zeros_x = jnp.zeros((c, ny, nx_spec), dtype=d.dtype)
        # ``k_x = 0`` is local column 0 of the first device column
        # only; every other device contributes exact zeros.
        x0_loc = jnp.where(col0 == 0, d[:, :, :, 0], 0.0)
        blocks = (
            lax.dynamic_update_slice_in_dim(
                zeros_z, jnp.sum(d, axis=3), row0, 2
            ),
            lax.dynamic_update_slice_in_dim(
                zeros_x, jnp.sum(d, axis=2), col0, 2
            ),
            lax.dynamic_update_slice_in_dim(zeros_z, x0_loc, row0, 2),
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
    m_x = gathered[..., :n2]
    m_z = gathered[..., nz_spec : nz_spec + n3]
    m_x0 = gathered[..., nz_spec + nx_spec :][..., :n2]
    return _fold_kz(m_x), m_z, _fold_kz(m_x0)


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


@jit
def _twin_yspectra_jit(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> dict[str, Array]:
    r"""Wall-normal-resolved componentwise spectra (module docstring).

    ``e_x`` / ``e_z`` / ``e_x0`` are the difference field's three
    marginals of :func:`_marginals_replicated`, per velocity
    component; ``r_x`` / ``r_z`` / ``r_x0`` the reference state's.
    Every array is a `$y$`-**density**: integrate with
    ``flow.y_weights`` (shipped in the stream's sidecar) to get the
    per-`$k$` energy, and sum over `$k$` for ``twin.dat``'s ``E_d``.
    """
    delta = state2 - state1
    e_x, e_z, e_x0 = _marginals_replicated(_energy_density(delta, fourier_))
    r_x, r_z, r_x0 = _marginals_replicated(_energy_density(state1, fourier_))
    return {
        "e_x": e_x,
        "e_z": e_z,
        "e_x0": e_x0,
        "r_x": r_x,
        "r_z": r_z,
        "r_x0": r_x0,
    }


def twin_yspectra(state1: Array, state2: Array) -> dict[str, Array]:
    """Wrapper around ``_twin_yspectra_jit`` binding the singletons."""
    return _twin_yspectra_jit(state1, state2, fourier, flow)


# ── Wall-normal-resolved spectral budget ─────────────────────────────

#: ``twin_ybudget`` term names, in stored order.  ``V`` is the viscous
#: term in the operator (discrete-Laplacian) form -- the one that makes
#: the budget close, matching ``twin_budget``'s ``eps_*`` -- and ``eps``
#: its positive-definite pseudo-dissipation companion; the two differ
#: by the wall-normal diffusion flux (module docstring).
YBUDGET_TERMS: tuple[str, ...] = (
    "P_U",
    "P_r",
    "T_ref",
    "T_self",
    "V",
    "eps",
    "Pi",
)


class _Sources(NamedTuple):
    r"""The advective products the budget and the pressure share.

    Built once per sample by :func:`_difference_sources` -- the 33
    field transforms are the budget's whole cost, so the pressure
    rides on them rather than repeating them.
    """

    delta: Array
    q_p: Array  # (Du . grad) u'^(1)
    q_tr: Array  # (u'^(1) . grad) Du
    q_ts: Array  # (Du' . grad) Du
    q_pu: Array  # Dv d_y U^(1), the lift-up term
    n_hat: Array  # the full nonlinear term
    div_n: Array  # its discrete divergence
    prof_dU: Array  # (3, Ny) mean-mode difference profile


def _difference_sources(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> _Sources:
    r"""Evaluate the difference field's advective terms.

    Twenty-four forward transforms (the three advectors and the two
    gradient sets) and nine back -- fewer than the 69 the three-bin
    :func:`twin_budget` needs, because binning no longer forces a
    separate physical product per bin pair.

    Two exact simplifications: the mean part of either *advector*
    contributes identically zero to the energy (`$U_y^{(1)} = \Delta
    U_y = 0$` by continuity plus no-slip, and what remains is
    `$i(k_xU_x + k_zU_z)|\Delta\hat u|^2$`, purely imaginary), so
    the transport terms take the mean-free advectors -- which also
    keeps a large, exactly-cancelling term out of the transforms.
    The *pressure* needs the full term, so ``n_hat`` adds the three
    mean-mode pieces back spectrally, at no transform cost.
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

    def mean_advect(prof: Array) -> Array:
        r"""`$(\mathbf{P}\cdot\nabla)\Delta\mathbf{u}$`, diagonal in
        `$k$`; the wall-normal row of either mean profile vanishes."""
        return (
            1j
            * (kx * prof[0][:, None, None] + kz * prof[2][:, None, None])
            * delta
        )

    adv = chunked_transform(
        spec_to_phys, jnp.concatenate([delta, ref_f, delta_f], axis=0)
    )
    grad_ref = chunked_transform(spec_to_phys, grad_spec(ref_f))
    grad_del = chunked_transform(spec_to_phys, grad_spec(delta))

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

    q_p = advect(adv[0:3], grad_ref)
    q_tr = advect(adv[3:6], grad_del)
    q_ts = advect(adv[6:9], grad_del)
    q_pu = delta[1] * dy_rU[:, :, None, None]

    n_hat = -(
        q_p + q_tr + q_ts + q_pu + mean_advect(prof_rU) + mean_advect(prof_dU)
    )
    # The solver's own discrete divergence
    # (``cartesian._imm_iteration_vp`` stage 1).
    div_n = (
        1j * kx * n_hat[0] + apply_y_matrix(d1, n_hat[1]) + 1j * kz * n_hat[2]
    )
    return _Sources(delta, q_p, q_tr, q_ts, q_pu, n_hat, div_n, prof_dU)


def _ybudget_densities(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    pressure: DifferencePressure,
) -> Array:
    r"""The seven per-mode budget densities, stacked.

    Returns a real ``(7, N_y, N_{k_z}, N_{k_x})`` array in
    :data:`YBUDGET_TERMS` order, component-summed, each already
    carrying ``k_metric`` and divided by ``volume_fac`` -- so summing
    over `$(k_z, k_x)$` and integrating with ``y_weights`` reproduces
    the corresponding scalar of ``twin_budget.dat``.
    """
    k2, k_metric = fourier_.k2, fourier_.k_metric
    vf = derived_params.volume_fac
    nu = 1.0 / params.phys.re
    d1, d2 = flow_.D1, flow_.D2
    src = _difference_sources(state1, state2, fourier_, flow_)
    delta = src.delta

    def pair(b: Array) -> Array:
        r"""`$-\sigma_{k_x}\sum_i\mathrm{Re}\{\Delta\hat u_i^* b_i\}/V$`."""
        return -jnp.sum((jnp.conj(delta) * b).real, axis=0) * (k_metric / vf)

    # Viscous: the operator form (closure-consistent, matching
    # ``twin_budget``'s ``eps_*``) and the positive-definite
    # pseudo-dissipation `$\nu|\nabla\Delta u|^2$`; their difference
    # is the wall-normal diffusion flux.
    visc = pair(-(apply_y_matrix(d2, delta) - k2 * delta)) * nu
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
    pi = pressure.work_density(delta, p_hat, flow_, fourier_)
    pi = (
        pi
        + _driving_density(state1, state2, src.prof_dU, flow_)
        * component_masks(fourier_)[0]
    )

    return jnp.stack(
        [
            pair(src.q_pu),
            pair(src.q_p),
            pair(src.q_tr),
            pair(src.q_ts),
            visc,
            eps,
            pi,
        ]
    )


def _driving_density(
    state1: Array, state2: Array, prof_dU: Array, flow_: object
) -> Array:
    r"""Mean-mode driving work density, shape ``(N_y, 1, 1)``.

    The `$(0,0)$` mode's pressure term is not the fluctuating pressure
    (which does no work there: `$\Delta\hat v_{00}\equiv 0$` and the
    horizontal gradients vanish) but the applied driving,
    `$\Delta\Pi_s \Delta U_s(y) + \Delta\Pi_n \Delta U_n(y)$`.  Its
    `$y$`-integral is `$\Delta\Pi \cdot U_\text{bulk}(\Delta u) = 0$`
    exactly -- at constant flow rate both members hold the same bulk,
    at fixed pressure gradient `$\Delta\Pi = 0$` -- but its *density*
    is not, so a `$y$`-resolved budget needs it.

    `$\Delta\Pi$` is the **wall-shear inference**
    (:func:`~dnsjax.geometries.wall_bounded.cartesian.mean_driving`)
    of each member's driving, differenced; that is deliberately the
    better budget partner than the corrector's applied value (the
    `$t = t_0$` reasoning in :mod:`dnsjax.__main__`).  Returns exact
    zeros when no driving constraint is active.
    """
    drive1 = mean_driving(state1, flow_)
    drive2 = mean_driving(state2, flow_)
    cos_t, sin_t = derived_params.cos_tilt, derived_params.sin_tilt
    dens = jnp.zeros_like(prof_dU[0])
    if DRIVING_KEY_S in drive1:
        dens = dens + (drive2[DRIVING_KEY_S] - drive1[DRIVING_KEY_S]) * (
            prof_dU[0] * cos_t + prof_dU[2] * sin_t
        )
    if DRIVING_KEY_N in drive1:
        dens = dens + (drive2[DRIVING_KEY_N] - drive1[DRIVING_KEY_N]) * (
            -prof_dU[0] * sin_t + prof_dU[2] * cos_t
        )
    return (dens / derived_params.volume_fac)[:, None, None]


@jit
def _twin_ybudget_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    pressure: DifferencePressure,
) -> dict[str, Array]:
    r"""Wall-normal-resolved spectral budget (module docstring).

    Returns ``<term>_x`` / ``<term>_z`` / ``<term>_x0`` for each of
    :data:`YBUDGET_TERMS`, the three marginals of
    :func:`_marginals_replicated`.  Every array is a `$y$`-density:
    integrate with ``flow.y_weights`` for the per-`$k$` rate, and sum
    over `$k$` for the corresponding ``twin_budget.dat`` column.
    """
    stacked = _ybudget_densities(state1, state2, fourier_, flow_, pressure)
    m_x, m_z, m_x0 = _marginals_replicated(stacked)
    out: dict[str, Array] = {}
    for i, name in enumerate(YBUDGET_TERMS):
        out[f"{name}_x"] = m_x[i]
        out[f"{name}_z"] = m_z[i]
        out[f"{name}_x0"] = m_x0[i]
    return out


def twin_ybudget(
    state1: Array, state2: Array, pressure: DifferencePressure
) -> dict[str, Array]:
    """Wrapper around ``_twin_ybudget_jit`` binding the singletons."""
    return _twin_ybudget_jit(state1, state2, fourier, flow, pressure)


@jit
def _twin_pressure_check_jit(
    state1: Array,
    state2: Array,
    fourier_: Fourier,
    flow_: object,
    pressure: DifferencePressure,
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
    - ``neumann``: `$(D_1\Delta\hat p - Re^{-1}D_2\Delta\hat v)|_w$`,
      the analytic condition the IMM closure declines to impose -- a
      wall-normal truncation diagnostic that must shrink with
      ``res.ny``, not an error.
    """
    src = _difference_sources(state1, state2, fourier_, flow_)
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
        "neumann": pressure.neumann_residual(delta, p_hat, flow_),
        "div_n": src.div_n,
        "dy_dtv": apply_y_matrix(flow_.D1, dtv),
    }


def twin_pressure_check(
    state1: Array, state2: Array, pressure: DifferencePressure
) -> dict[str, Array]:
    """Wrapper around ``_twin_pressure_check_jit`` binding the singletons."""
    return _twin_pressure_check_jit(state1, state2, fourier, flow, pressure)
