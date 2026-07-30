r"""Linear transient (optimal energy) growth around arbitrary profiles.

Compute the three-dimensional linear transient-growth spectrum of a
wall-bounded flow **linearised about an arbitrary wall-normal total
profile** `$\mathbf{U}(y)$` -- not necessarily a laminar / stationary
solution.  Supported systems: ``plane-couette`` / ``plane-poiseuille``
(Cartesian), ``pipe`` (cylindrical), ``taylor-couette`` and
``quasi-keplerian`` (annular).  The force-driven Dean / viscoelastic-Dean
flows are out of scope.

Run as a CLI (single process, single device; GPU with
``--dist.platform cuda`` and ``CUDA_VISIBLE_DEVICES=0``)::

    python -m dnsjax.analysis.transient_growth --tg.profile U.txt \
        --phys.system plane-poiseuille --phys.re 1000 \
        --res.ny 96 --res.nx 4 --res.nz 4 --geo.lz 3.074 \
        --tg.modes "1,0"

The command line and ``parameters.toml`` go through the shared
per-flow surface (:func:`dnsjax.bootstrap.resolve_parameters`): flow
parameters under their public names (``--help <system>`` documents
them -- e.g. the pipe resolution is ``--res.nr``/``--res.ntheta``),
strict relevance, plus this driver's own knobs as the ``[tg]``
extension section (``--tg.<field>``; :class:`TGParams`).

The profile file has two whitespace-separated columns: the wall-normal
grid points **top wall first (descending)** -- the same convention as a
``geo.wall_grid`` file -- and the *total* profile value there (the
streamwise `$u_x$` for the Cartesian flows, axial `$u_z$` for pipe,
azimuthal `$u_\theta$` for Taylor-Couette).  A directory may be given
instead of a file, in which case every file in it is processed.

Why this reuses the solver exactly
==================================
The base profile enters the solver only through the pair of flow arrays
``base_flow`` / ``curl_base_flow`` that the FFT-free linear coupling
`$L_{bf} = \mathbf{u}'\times\nabla\times\mathbf{U} +
\mathbf{U}\times\boldsymbol{\omega}'$` reads (each geometry's ``_l_bf``;
:func:`dnsjax.geometries.wall_bounded._base.base_flow_coupling`).  This
coupling is **exact for any wall-parallel `$y$`-only profile**: because
`$\mathbf{U}\cdot\nabla\mathbf{U}\equiv 0$`, the base self-interaction is
a pure gradient absorbed by the pressure (see the :mod:`dnsjax.rhs`
module docstring), so linearising about a *non-solution* profile
introduces **no extra terms** in the Jacobian -- the base residual is a
constant forcing that does not enter the linear operator.  Every other
operator (the viscous Helmholtz `$H_k$`, the pressure Poisson `$L_k$`,
and the influence-matrix data enforcing `$\nabla\cdot\mathbf{u}'=0$` and
the wall BCs) is profile-independent, so a per-profile flow is a shallow
copy with the two arrays swapped
(:func:`dnsjax.geometries.wall_bounded._base.frozen_profile_flow`; each
flow module's ``frozen_profile_flow(profile)`` builder).

Mathematical formulation
========================
Per non-mean Fourier mode `$(k_2, k_3)$` the linearised dynamics are
`$d\mathbf{q}/dt = \mathcal{A}\,\mathbf{q}$` on the complex state
`$\mathbf{q}\in\mathbb{C}^{n}$`, `$n = 3 N_y$` (three velocity
components on the wall-normal grid), restricted to the discretely
divergence-free, no-slip subspace `$S$`.  Transient growth measures how
much the perturbation *energy* can transiently amplify even when every
eigenvalue of `$\mathcal{A}$` decays (a non-normal-operator effect):

.. math::
    G(t) = \max_{\mathbf{q}(0)\neq 0}
    \frac{\lVert \mathbf{q}(t)\rVert_E^2}{\lVert \mathbf{q}(0)\rVert_E^2},
    \qquad
    G_{\max} = \max_{t\ge 0} G(t)\ \text{at}\ t = t_{\mathrm{opt}}.

The energy norm `$\lVert\mathbf{q}\rVert_E^2 = \mathbf{q}^{H} W
\mathbf{q}$` is the solver's own kinetic energy (``get_norm2*``): `$W$`
is diagonal, `$W = \mathrm{diag}_c(m_c\,w_y)$` with the wall-normal
quadrature weights `$w_y$` (``flow.y_weights``; the radial Jacobian
`$r$` is baked in for pipe / Taylor-Couette) and the component metric
`$m_c = 1$` in every family: this driver works in the **physical**
component basis throughout (``_linear_step`` converts the
cylindrical / annular solver basis at the stepper boundary), and a
physical triad is pointwise orthonormal, so ``get_norm2*`` is the
plain component sum.  The Hermitian ``k_metric`` factor and ``volume_fac`` are
constants for a fixed mode and cancel in the ratio `$G$`, so they are
omitted (all reported quantities -- `$G$`, `$G_{\max}$`, the abscissae
-- are ratios or rates invariant under an overall scaling of `$W$`).

The generator `$\mathcal{A}$` is extracted from **one** solver step.
With CN weight `$\theta = 1$` (backward Euler,
``step.implicit_mean_coupling = False``) the converged implicit step is

.. math::
    \Phi = (I - \Delta t\,\mathcal{A})^{-1}\quad\text{on } S,

realised exactly by the influence-matrix pressure solve (see the
``_imm_iteration`` docstring in
:mod:`dnsjax.geometries.wall_bounded.cartesian` and
:func:`dnsjax.timestep.make_stepper`).  Backward Euler is an *exact
rational function* of `$\mathcal{A}$`, so inverting the relation
recovers `$\mathcal{A}$` to rounding -- `$\Delta t$` is a probe, **not**
an accuracy knob, and there is no time-discretisation error.

The propagator is the solver's, so it inherits ``res.consistent_imm``.
In every wall-bounded geometry that flag selects the reconstruction
scheme, whose solenoidal subspace `$S$` is *exactly* the discrete one:
a non-solenoidal basis vector's tangential part maps to zero in a
single step rather than decaying over several, so `$\Phi$` is singular
on the complement by construction.  That is the intended behaviour on
`$S$` and does not affect `$G(t)$`, which is computed there -- but it
does mean the raw propagator is rank-deficient off it.

The pipeline per mode is:

1. **Propagator.**  `$\Phi$` (an `$n\times n$` matrix) is built column
   by column: the `$j$`-th unit vector `$\mathbf{e}_{(c,j)}$` placed at
   *every* selected mode at once, stepped once, gives column `$(c,j)$`
   of *every* mode's `$\Phi$` simultaneously (the linear step is
   block-diagonal in `$(k_2,k_3)$` because the only `$k=0$` content is
   the fixed base flow).  So the whole propagator set costs just `$3
   N_y$` cheap FFT-free linear steps, independent of the mode count.

2. **Subspace reduction.**  `$\Phi$` maps `$\mathbb{C}^n$` into `$S =
   \mathrm{range}(\Phi)$`.  A singular-value decomposition exposes a
   clean numerical rank `$r = \dim S$` (`$\approx 2 N_y$`; the null
   space is the constraint-violating directions the influence matrix
   projects out).  The leading `$r$` **left** singular vectors `$V$`
   are an orthonormal basis of `$S$`, and `$\Phi_S = V^{H}\Phi V$` is
   the exact `$r\times r$` restriction (`$\Phi V \subseteq \mathrm{span}
   V$`).

3. **Generator.**  `$\Phi_S = (I - \Delta t\,\mathcal{A}_S)^{-1}$`
   shares eigenvectors with `$\mathcal{A}_S$`, so its host
   eigendecomposition `$\Phi_S Y = Y\,\mathrm{diag}(\mu)$` yields the
   generator spectrum directly, `$\lambda = (1 - 1/\mu)/\Delta t$`;
   `$\mathcal{A}$` is never formed or inverted (that would amplify
   the stiff-mode round-off).  An unresolved stiff mode
   (`$|\mu| \le \tfrac12$`) whose round-off phase flips
   `$\mathrm{Re}\,\lambda$` above the resolved spectral abscissa is
   clamped to instant decay.  The reported ``extraction_residual`` is
   the relative eigendecomposition residual
   `$\lVert \Phi_S Y - Y\,\mathrm{diag}(\mu)\rVert_F /
   \lVert\Phi_S\rVert_F$`.

4. **Energy metric.**  `$M = V^{H} W V$` (`$r\times r$`, Hermitian
   positive-definite), `$F = \mathrm{chol}(M)$` upper (so `$M =
   F^{H}F$` and `$\lVert\mathbf{a}\rVert_E = \lVert F\mathbf{a}
   \rVert_2$` in the `$V$` coordinates).  In energy-orthonormal
   coordinates the propagator is `$B(t) = F\,e^{t\mathcal{A}}\,F^{-1}$`.

5. **Restriction to the resolved eigenspace.**  Growth is measured on
   the probe-*resolved* eigenspace only (`$|\mu| > \tfrac12$`, i.e.
   `$|\lambda| \lesssim 2/\Delta t$`).  Its energy-coordinate
   eigenvectors are factorised `$E_{\mathrm{res}} = QR$` (`$Q$`
   orthonormal, so the 2-norm in `$Q$` coordinates *is* the energy
   norm), giving the reduced generator `$\mathcal{A}_{\mathrm{res}} =
   Q^{H}\mathcal{A}Q = R\,\mathrm{diag}(\lambda_{\mathrm{res}})
   \,R^{-1}$` -- already in eigenform, so no ``expm`` is needed.
   This restriction is **not** an optimisation: carrying the
   unresolved modes would turn the propagator into a *non-orthogonal
   spectral projector* the instant their `$e^{t\lambda}$` dies, so
   the computed `$G$` would jump from `$1$` to `$\lVert$`projector
   `$\rVert^2 \ggg 1$` at `$t = 0^{+}$` -- a mesh-dependent
   discontinuity with no physical meaning that can dominate
   `$G_{\max}$` outright.  On the restriction `$G(0) = 1$` holds
   *continuously*, and `$G$` is a rigorous **lower bound** on the
   full-space growth, converging from below as the resolved window
   widens.

6. **Growth & optima.**  `$G(t) = \sigma_{\max}(B(t))^2$` with
   `$B(t) = R\,\mathrm{diag}(e^{t\lambda_{\mathrm{res}}})\,R^{-1}$`;
   the stiff factors underflow to zero for `$t > 0$` rather than
   overflowing.  This needs a well-conditioned eigenbasis, so a
   (near-)defective `$\Phi_S$` is rejected at
   `$\mathrm{cond}(R) > 10^{12}$`.  `$t_{\mathrm{opt}}$` refines the
   grid maximum by golden section.  The leading singular triplet of
   `$B(t_{\mathrm{opt}})$` gives the optimal input (right singular
   vector `$\mathbf{v}_1$`) and its response (left singular vector
   `$\mathbf{u}_1$`): in the full state, `$\mathbf{q}(0) = V
   F^{-1}Q\mathbf{v}_1$` and `$\mathbf{q}(t_{\mathrm{opt}}) =
   \sigma_1\, V F^{-1}Q\mathbf{u}_1$`.

7. **Abscissae & spectrum.**  Spectral abscissa `$\max_i\mathrm{Re}\,
   \lambda_i(\mathcal{A})$` (asymptotic growth rate) over the
   post-clamp -- i.e. *probe-resolved* -- spectrum; the leading
   eigenvalues are also stored, accurate only inside the resolved
   window `$|\lambda| \lesssim 1/\Delta t$` (beyond it the probe
   compresses `$\mu \to 0$`).  Numerical abscissa
   `$\lambda_{\max}\big((\tilde{\mathcal{A}} +
   \tilde{\mathcal{A}}^{H})/2\big)$` with `$\tilde{\mathcal{A}} =
   F\mathcal{A}F^{-1}$` (the maximum instantaneous energy growth rate,
   `$= \tfrac{1}{2}\,dG/dt|_{0}$`), compressed onto the resolved
   eigenspace `$|\mu| > \tfrac12$`: the unresolved near-wall FD modes
   carry spurious, mesh-dependent instantaneous growth that would
   otherwise swamp the diagnostic.  The nonsymmetric eigensolve runs
   on the host (:func:`numpy.linalg.eig`; JAX's ``eig`` is CPU-only).

Conventions and choices
=======================
- **Mean mode** `$(0,0)$` is excluded: the influence-matrix mean branch
  (bulk-velocity / spanwise-blocking projections, mean-mode driving)
  makes it affine rather than a clean linear block, and it is not part
  of a `$(k_2,k_3)\neq 0$` optimal-growth analysis.
- **Dealiasing** needs no action: the linear step is entirely FFT-free
  (spectral coupling + banded solves), so the 3/2 padding pipeline is
  never entered -- there is no aliasing to remove.
- **Moving frame:** ``phys.u_grid`` is forced to `$0$` (lab frame)
  unless the user sets it.  `$G$` is frame-invariant (a per-mode phase);
  a moving frame only shifts each eigenvalue by `$-i k_0 U_{grid}$`.
- **Backward Euler probe** (`$\theta = 1$`): exact generator recovery;
  the extraction is best conditioned near `$\Delta t \approx 10^{-2}$`
  (large `$\Delta t$` ill-conditions `$\Phi_S$`; tiny `$\Delta t$`
  cancels in `$(1 - 1/\mu)/\Delta t$`).  A trapezoidal `$\theta=\tfrac12$`
  Cayley variant `$\mathcal{A} = \tfrac{2}{\Delta t}(\Phi_S -
  I)(\Phi_S + I)^{-1}$` is an alternative, not implemented here.

Choosing the knobs
==================
The defaults suit the five systems at moderate `$Re$`; every failure
mode below is guarded per mode with an explicit error.

- ``--tg.dt`` (0.01): the probe step sets *conditioning* and the
  *resolved spectral window*, not accuracy.  Only eigenvalues with
  `$|\lambda| \lesssim 2/\Delta t$` are resolved (beyond, the probe
  compresses `$\mu \to 0$`), and those are exactly the modes `$G$` is
  measured on (step 5).  It therefore cuts **both** ways and the
  default is a balance, not a floor:

  * *Reduce* it to resolve more of the spectrum (a slow mode you need
    must satisfy `$|\lambda| \lesssim 2/\Delta t$`) or to fix a
    diverging corrector (contraction `$\propto \Delta t$`).
  * *Raise* it to tighten the window.  A wider window admits the
    fast, wall-clustered FD eigenmodes, which are **discretisation
    artefacts** unless `$N_y$` resolves them: they are strongly
    non-normal, and their short-time growth is real for the *discrete*
    operator while having no continuum counterpart.  It shows up as an
    early, mesh-dependent `$G$` spike that decays long before the
    physical optimum -- and, being a genuine property of `$\mathcal{A}
    _{\mathrm{res}}$`, no post-processing can subtract it.  Measured
    (quasi-Keplerian `$Re_i = 10^4$`, `$m = 4$`, `$N_y = 128$`), the
    numerical abscissa falls `$+162 \to +82 \to +57$` for `$\Delta t =
    0.003 \to 0.01 \to 0.03$`, and the rank gap *widens* over the same
    range (`$1.3\times10^{3} \to 1.4\times10^{3} \to 3.7\times10^{3}$`).

  The fix for a contaminated `$G_{\max}$` is therefore **`$N_y$`, not
  `$\Delta t$`** (see the convergence recipe below).
- ``--tg.corrector_tolerance`` (1e-11) / ``--tg.max_corrector_iterations``
  (200): the tolerance bounds each propagator column's error and so
  floors every reported quantity (the achieved
  ``corrector_error_max`` is in the outputs).  On the "failed to
  converge" error reduce ``--tg.dt`` first (faster contraction),
  raise the iteration cap second.
- ``--tg.rank_tol`` (1e-11) / ``--tg.rank_gap_min`` (1e3): the relative
  cutoff must land inside the singular-value cliff between the
  physical spectrum (whose floor is `$\sigma/\sigma_0 \sim
  1/(\Delta t\,|\lambda_{\mathrm{stiff}}|)$`) and the
  constraint-violating null space (whose floor is set by
  ``--tg.corrector_tolerance``, *not* by machine epsilon -- the columns
  are only converged that far); the gap check verifies that it did.
  On a "no clean rank gap" failure inspect the printed tail, then
  move ``--tg.rank_tol`` into the observed gap; lower ``--tg.rank_gap_min``
  only if the true cliff is genuinely that shallow.  ``--tg.dt`` also
  moves the gap, but *raising* it is what widens the cliff in
  practice (measured under ``--tg.dt`` above) -- both floors respond,
  so the naive "reduce it to lift the physical floor" is unreliable;
  read the printed tail rather than assuming a direction.
- ``--tg.t_max`` (default `$0.25\,Re$`): covers the classic optima,
  `$t_{\mathrm{opt}} \approx 0.05$`--`$0.15\,Re$`, of the five flows
  (Taylor-Couette / quasi-Keplerian use the derived reference ``re``:
  ``re1``, or ``re2`` when outer-driven).  Raise it when a mode prints the
  "G still rising at t_max" warning.
- ``--tg.nt`` (65): `$G(t)$`-grid density.  The golden-section
  refinement only polishes the bracket around the *grid* argmax, so
  raise it when `$G(t)$` may be multimodal or narrow-peaked; it also
  sets the ``--tg.save_all_times`` storage.
- ``--tg.t_chunk`` (16): batch size of the device SVDs over the time
  grid (`$\sim$` ``t_chunk`` `$\times\,r^2$` complex temporaries).
  Raise on GPU for throughput, lower to bound device memory at
  large ``ny``.
- ``--tg.interp_order`` (8): Fornberg stencil width ``order + 1`` for
  profile regridding.  High order suits smooth profiles; drop to 2-4
  for noisy (experimental / binned DNS-mean) data, where a wide
  stencil amplifies the noise.
- ``--tg.wall_bc_tol`` (1e-6): the perturbation BCs are homogeneous, so
  the *total* profile must carry the exact laminar wall values -- a
  real mismatch is a different BC problem, not a tolerance matter.
  Loosen only for wall values off by numerical artefacts
  (interpolation, output truncation).
- ``--tg.grid_match_tol`` (1e-12): identical-grid fast-path detector;
  raise it (e.g. to 1e-8) when the profile file stores the code grid
  with fewer printed digits, to skip a pointless interpolation.
- ``--tg.n_eig`` (20): how many leading eigenvalues are *stored*; no
  accuracy effect.  Trustworthy only inside the resolved window
  `$|\lambda| \lesssim 1/\Delta t$` (see ``--tg.dt``).
- ``--tg.save_all_times``: stores the optimal pair at every grid time,
  `$2 K n_t (3 N_y)$` complex values over `$K$` modes -- the
  dominant output for mode sweeps.
- ``--tg.save_operator``: also writes ``<stem>_tg_op.npz`` with each
  mode's reduced generator `$\mathcal{A}$` (restricted to the
  probe-resolved eigenspace, in an orthonormal energy-coordinate
  basis), the bases `$V, F, Q$`, and the coordinate contract -- the
  input for controllability / growth-curve / identification
  post-processing
  (:mod:`dnsjax.analysis.response.operator_tools`; the storage
  layout: the ``_write_operator_npz`` docstring).
- ``--tg.export_amplitude`` (1e-4): volume-averaged energy `$E'$` of
  the exported seed (the solver's own measure).  The default keeps
  the seeded DNS initially linear; raise it for finite-amplitude
  (nonlinear, transition-triggering) seeding.

Converging `$N_y$` (do this before trusting `$G_{\max}$`)
=========================================================
The physical `$G(t)$` converges *fast* in `$N_y$`; the artefact of
step 5 converges *slowly*, so **`$N_y$` is set by the artefact, not by
the physics**.  Measured for quasi-Keplerian `$Re_i = 10^4$`,
`$m = 4$`, `$\Delta t = 10^{-2}$` (`$G_{\max} = 13.04$` at
`$t_{\mathrm{opt}} = 27$`):

===========  =====================  ===================  ==============
`$N_y$`      `$G$` at `$t \ge 4$`   early spike          `$\omega_
                                                         {\mathrm{num}}$`
===========  =====================  ===================  ==============
64           5 digits correct       `$G \approx 10^{2}$`  `$+516$`
128          5 digits correct       `$G \approx 8$`       `$+82$`
===========  =====================  ===================  ==============

So `$N_y = 64$` already nails the physical curve, yet reports
`$G_{\max} \sim 10^{2}$` at `$t \approx 2$`: the optimum is the
artefact.  `$N_y = 128$` pushes the spike under the physical peak and
the reported optimum becomes the physical one.

Recipe: raise `$N_y$` until (a) `$G_{\max}$` and `$t_{\mathrm{opt}}$`
stop moving, and (b) `$t_{\mathrm{opt}}$` is **not** in the early
transient.  The sharpest single indicator is the **numerical
abscissa**: it is `$O(1)$` for the continuum (a Reynolds-Orr
production bound) but mesh-dependent while the near-wall modes are
unresolved, so a value orders above the physical shear rate means the
early `$G$` is artefact-dominated -- regardless of how clean the
`$G_{\max}$` looks.  Sampling `$G(t)$` densely at small `$t$` (small
``--tg.t_max``, large ``--tg.nt``) exposes the spike directly; a coarse
`$t$` grid can *step over* it and report a plausible-looking optimum.

Cost & memory: a propagator build is `$3 N_y$` FFT-free linear steps
(independent of the mode count) plus a host SVD + eigensolve per
mode; the propagator set is held on the host at `$K (3 N_y)^2$`
complex128 (the estimate is printed at startup).  For large mode
sweeps split ``--tg.modes`` across separate processes (one device
each) -- the pipeline is embarrassingly parallel over modes.

Outputs
=======
Per profile ``<stem>``: a human-readable ``<stem>_tg_summary.txt`` (one
row per mode) and a self-describing ``<stem>_tg.npz`` (grids, the
interpolated profile, the resolved parameters, and per-mode `$G(t)$`,
`$G_{\max}$`, `$t_{\mathrm{opt}}$`, abscissae, leading eigenvalues,
singular values, and the optimal input / response at
`$t_{\mathrm{opt}}$` -- optionally at every requested time with
``--tg.save_all_times``).  For a linearly *unstable* profile (positive
spectral abscissa) `$G(t)$` grows without bound, so `$G_{\max}$` /
`$t_{\mathrm{opt}}$` merely reflect the end of the time grid (the
"still rising" warning fires) and the abscissae are the meaningful
outputs.  ``--tg.export_snapshot "i2,i3"`` writes the chosen
mode's optimal perturbation as a standard dnsjax snapshot (with the
`$k=0$`-plane conjugate partner filled in for a real physical field) to
seed a DNS run.

Future work: eigenvector output
================================
Only the eigen*values* are stored.  The eigen*vectors* of
`$\mathcal{A}$` (the linear-stability modes) are already in hand:
the columns `$\mathbf{y}_i^{(r)}$` of `$Y$` from the `$\Phi_S$`
eigendecomposition in ``_analyze_mode`` (same eigenvectors as
`$\mathcal{A}_S$`).  To expose them: energy-normalise so
`$\lVert F\mathbf{y}_i^{(r)}\rVert_2 = 1$`, lift to the full state
`$\mathbf{y}_i = V\mathbf{y}_i^{(r)}$`, sort by
`$\mathrm{Re}\,\lambda_i$`, and store alongside ``eigvals`` (adding a
`$k=0$`-plane conjugate partner for a snapshot export exactly as the
optimal-perturbation export does).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .. import harmonics
from ..bootstrap import (
    _scan_flag,
    configure_jax_platform,
    peek_run_context,
    resolve_parameters,
)
from ..extensions import ParamExtension, register_extension
from ..fd import local_interpolation_matrix
from ..parameters import (
    Parameters,
    _user_set_fields,
    derived_params,
    padded_res,
    params,
    update_parameters,
    validate_parameters,
)
from ..snapshot_meta import git_hash

if TYPE_CHECKING:
    from jax import Array

# The TG scope: the wall-bounded *base-flow* (perturbation-form)
# systems, whose flow modules export ``frozen_profile_flow``.  The
# force-driven Dean / viscoelastic-Dean flows integrate the total
# field around ``base_flow = 0`` and are out of scope, so this cannot
# be derived from the spec families; the modules and geometry come
# from the registry (``FlowSpec.flow_module`` / ``family``) in
# :func:`_dispatch`.
WALL_BOUNDED_TG_SYSTEMS = (
    "plane-couette",
    "plane-poiseuille",
    "pipe",
    "taylor-couette",
    "quasi-keplerian",
)

# Component labels in the stored state basis, per family.
_COMPONENT_LABELS = {
    "cartesian": ("u_x", "u_y", "u_z"),
    "cylindrical": ("u_z", "u_r", "u_theta"),
    "annular": ("u_z", "u_r", "u_theta"),
}
# Energy component metric m_c in the stored basis, per family (every
# native basis is a pointwise orthonormal physical triad).
_COMPONENT_METRIC = {
    "cartesian": (1.0, 1.0, 1.0),
    "cylindrical": (1.0, 1.0, 1.0),
    "annular": (1.0, 1.0, 1.0),
}
# TG-owned step fields; warn if the user tries to set them.
_TG_OWNED_STEP = (
    "dt",
    "implicitness",
    "scheme",
    "implicit_mean_coupling",
    "corrector_tolerance",
    "max_corrector_iterations",
    "split_corrector",
)

#: Program name shown in this CLI's usage / help / error hints.
_PROG = "python -m dnsjax.analysis.transient_growth"


# ── Configuration models ─────────────────────────────────────────


class TGParams(BaseModel):
    r"""Transient-growth driver knobs: the ``[tg]`` extension section.

    Parsed as ``--tg.<field>`` CLI flags / a ``[tg]`` TOML section on
    the shared per-flow parameter surface
    (:func:`dnsjax.bootstrap.resolve_parameters`), alongside the
    flow's own parameters under their public names.  Knob guidance
    (especially the ``dt`` conditioning trade-off and the rank/
    resolvedness cuts): the module docstring.
    """

    model_config = ConfigDict(extra="forbid")

    profile: str | None = Field(
        default=None,
        description=(
            "Two-column profile file (wall-normal grid descending "
            "from the top wall; total profile value), or a folder of "
            "such files.  Required."
        ),
    )
    out_dir: str = Field(
        default=".",
        description="Output directory for the *_tg.npz results.",
    )
    parameters: str | None = Field(
        default=None,
        description=(
            "Path of the parameters TOML to load (CLI-only; "
            "default: ./parameters.toml if present)."
        ),
    )
    modes: str = Field(
        default="all",
        description=(
            "'all' (every non-mean mode) or 'i2,i3;i2,i3;...' "
            "(i2 = kz/m axis, i3 = kx/axial axis)."
        ),
    )
    t_max: float | None = Field(
        default=None,
        gt=0,
        description="End of the G(t) grid (default 0.25*Re).",
    )
    nt: int = Field(
        default=65,
        ge=2,
        description="Number of G(t) grid points including t=0.",
    )
    dt: float = Field(
        default=0.01,
        gt=0,
        description=(
            "Backward-Euler probe step; sets conditioning and the "
            "resolved-mode window |lambda| <~ 1/dt."
        ),
    )
    corrector_tolerance: float = Field(
        default=1e-11,
        gt=0,
        description="Corrector tolerance of the linear probe step.",
    )
    max_corrector_iterations: int = Field(
        default=200,
        ge=1,
        description="Corrector iteration cap of the linear probe step.",
    )
    rank_tol: float = Field(
        default=1e-11,
        gt=0,
        description="Relative singular-value cutoff for the rank.",
    )
    rank_gap_min: float = Field(
        default=1e3,
        gt=0,
        description="Required s[r-1]/s[r] gap at the rank cut.",
    )
    interp_order: int = Field(
        default=8,
        ge=1,
        description=(
            "Accuracy order of the profile regridding (local "
            "Fornberg stencils of interp_order+1 points; same "
            "convention as res.fd_order)."
        ),
    )
    wall_bc_tol: float = Field(
        default=1e-6,
        gt=0,
        description="Relative wall-value tolerance of the profile.",
    )
    grid_match_tol: float = Field(
        default=1e-12,
        gt=0,
        description="Same-grid fast-path tolerance.",
    )
    n_eig: int = Field(
        default=20,
        ge=1,
        description="Leading eigenvalues stored per mode.",
    )
    t_chunk: int = Field(
        default=16,
        ge=1,
        description="SVD batch size over the time grid.",
    )
    save_all_times: bool = Field(
        default=False,
        description="Store the optimal pair at every grid time.",
    )
    save_operator: bool = Field(
        default=False,
        description=(
            "Also write <stem>_tg_op.npz: per-mode reduced generator "
            "A, resolved basis Q, energy factor F, subspace basis V."
        ),
    )
    export_snapshot: str | None = Field(
        default=None,
        description=(
            "'i2,i3': export that mode's optimal as a dnsjax snapshot."
        ),
    )
    export_amplitude: float = Field(
        default=1e-4,
        gt=0,
        description="Perturbation energy E' of the exported seed.",
    )
    export_which: Literal["input", "response"] = Field(
        default="input",
        description="Export the optimal input or its t_opt response.",
    )


TG_EXTENSION = register_extension(
    ParamExtension(
        name="tg",
        model=TGParams,
        relevant=lambda system: system in WALL_BOUNDED_TG_SYSTEMS,
        summary="Transient-growth driver (this CLI's own knobs).",
        # Analysis-run config, not trajectory state: never embedded in
        # snapshot metadata (an exported seed must not carry it).
        record_in_metadata=False,
    )
)

#: Live ``[tg]`` values (resolved by ``resolve_parameters`` in
#: :func:`main`; registration is import-time, so any entry point that
#: imports this module ahead of parsing gets the section).
tg_params: TGParams = TG_EXTENSION.values


def _config_json(cfg: TGParams) -> str:
    """The resolved ``[tg]`` knobs as a provenance JSON blob."""
    return json.dumps(cfg.model_dump(), sort_keys=True)


@dataclasses.dataclass
class ModeResult:
    """Per-mode transient-growth results."""

    i2: int
    i3: int
    wn2: float
    wn3: float
    rank: int
    sv_gap: float
    extraction_residual: float
    G: np.ndarray  # (nt,)
    G_max: float
    t_opt: float
    sigma_opt: float
    spectral_abscissa: float
    numerical_abscissa: float
    eigvals: np.ndarray  # (n_eig,) complex
    singular_values: np.ndarray  # (n,) real
    opt_input: np.ndarray  # (3, Ny) complex, stored basis
    opt_response: np.ndarray  # (3, Ny) complex, stored basis
    opt_input_t: np.ndarray | None  # (nt, 3, Ny) or None
    opt_response_t: np.ndarray | None
    sigma_t: np.ndarray | None  # (nt,) or None
    # ``--tg.save_operator`` payload (None otherwise); the coordinate
    # contract is documented in ``_write_operator_npz``.
    op_V: np.ndarray | None = None  # (n, r) subspace basis
    op_F: np.ndarray | None = None  # (r, r) upper Cholesky, M = F^H F
    op_Q: np.ndarray | None = None  # (r, r_res) resolved basis
    op_A: np.ndarray | None = None  # (r_res, r_res) reduced generator
    op_lam: np.ndarray | None = None  # (r_res,) resolved eigenvalues


# ── Host-side helpers (JAX-free) ─────────────────────────────────


def _gather_profiles(path_str: str) -> list[Path]:
    """A single file -> ``[file]``; a directory -> its sorted files."""
    path = Path(path_str)
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.is_file())
        if not files:
            raise SystemExit(f"no files in profile directory {path}")
        return files
    if path.is_file():
        return [path]
    raise SystemExit(f"profile path not found: {path}")


def _read_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column profile file into ascending ``(y, u)``.

    Column 0 is the wall-normal grid **descending** from the top wall
    (the ``geo.wall_grid`` convention); both columns are reversed to the
    code's ascending order.
    """
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise SystemExit(
            f"{path}: expected two columns (grid, profile), got "
            f"shape {data.shape}"
        )
    y_raw = data[:, 0]
    if not np.all(np.diff(y_raw) < 0):
        raise SystemExit(
            f"{path}: first column must be strictly descending "
            "(top wall first)"
        )
    return y_raw[::-1].copy(), data[::-1, 1].copy()


def _regrid_profile(
    y_user: np.ndarray,
    u_user: np.ndarray,
    y_code: np.ndarray,
    order: int,
    match_tol: float,
) -> tuple[np.ndarray, bool]:
    """Interpolate the profile onto the code grid if needed.

    Returns ``(u_on_code_grid, interpolated)``.  Identical grids take a
    bit-exact fast path.  Otherwise a local Fornberg interpolation
    (:func:`dnsjax.fd.local_interpolation_matrix`) maps the arbitrary
    monotone user grid onto the code grid.
    """
    if (
        len(y_user) == len(y_code)
        and np.max(np.abs(y_user - y_code)) < match_tol
    ):
        return u_user.copy(), False
    span_lo, span_hi = y_user[0], y_user[-1]
    # 1e-9 absolute slack: endpoint round-off in written profile
    # files (a printed grid may miss the wall by its format), not
    # a knob -- interpolation never extrapolates beyond it.
    if y_code[0] < span_lo - 1e-9 or y_code[-1] > span_hi + 1e-9:
        raise SystemExit(
            f"profile grid [{span_lo:.4f}, {span_hi:.4f}] does not "
            f"cover the code grid [{y_code[0]:.4f}, {y_code[-1]:.4f}]"
        )
    mat = local_interpolation_matrix(y_user, y_code, order)
    return mat @ u_user, True


def _select_modes(
    spec: str, n2: int, n3: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the ``--tg.modes`` spec to index arrays (mean excluded)."""
    if spec.strip() == "all":
        pairs = [
            (i2, i3)
            for i2 in range(n2)
            for i3 in range(n3)
            if not (i2 == 0 and i3 == 0)
        ]
    else:
        try:
            parsed = harmonics.parse_mode_pairs(spec)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        pairs = []
        for i2, i3 in parsed:
            if i2 == 0 and i3 == 0:
                raise SystemExit("the mean mode (0,0) is excluded")
            if not (0 <= i2 < n2 and 0 <= i3 < n3):
                raise SystemExit(
                    f"mode ({i2},{i3}) out of range (0..{n2 - 1}, 0..{n3 - 1})"
                )
            pairs.append((i2, i3))
    if not pairs:
        raise SystemExit("no modes selected")
    arr = np.asarray(pairs, dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def _wavenumber_arrays(
    family: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    """Physical wavenumbers for the two spectral axes (host-recompute).

    Axis 2 (from ``nz``) is spanwise `$k_z$` / azimuthal `$m$`; axis 3
    (from ``nx``, the real-FFT axis) is streamwise `$k_x$` / axial.
    Recomputed from :mod:`dnsjax.harmonics` scaled by `$2\\pi/L$`, per
    the global-array caveat in the ``fourier`` singleton docstrings.
    """
    nx, nz = params.res.nx, params.res.nz
    ch = np.asarray(harmonics.complex_harmonics(nz), dtype=np.float64)
    rh = np.asarray(harmonics.real_harmonics(nx), dtype=np.float64)
    kax = rh * (2.0 * np.pi / params.geo.lx)
    if family == "cartesian":
        return ch * (2.0 * np.pi / params.geo.lz), kax, ("beta", "alpha")
    # Annular / cylindrical: physical azimuthal wavenumber m = m0 * j
    # over the wedge (m0 = 1 full circle; see geo.m0).
    return ch * params.geo.m0, kax, ("m", "k_ax")


# ── Parameter layering ───────────────────────────────────────────


def _configure_parameters(argv: list[str]) -> None:
    """Apply defaults < parameters.toml < CLI < forced TG overrides.

    The layering runs through the shared production surface
    (:func:`dnsjax.bootstrap.resolve_parameters`): flow parameters
    under their public names, strict relevance, ``--help`` /
    ``--help <system>`` / ``--sample-toml``, plus the ``[tg]``
    extension section carrying this driver's own knobs
    (``--tg.<field>``).  The flow-relevant *solver-run* extension
    sections (``[probes]``, ``[force]``) are accepted too, so a
    ``parameters.toml`` shared with a probed production run parses --
    the driver ignores them (a note is printed when one is
    configured); bare ``--help`` shows only ``[tg]``.  It then forces
    the TG-required settings (single device, backward-Euler linear
    probe, no mean coupling, tight corrector, lab frame, double
    precision) so the geometry / flow singletons -- imported *after*
    the platform is configured -- bake them into the operators.
    """
    from ..extensions import relevant_extensions

    toml_flag = _scan_flag(argv, "--tg.parameters")
    toml_path = Path(toml_flag) if toml_flag is not None else None
    ctx = peek_run_context(argv, toml_path=toml_path)
    if ctx.help_requested and ctx.help_system is None:
        ext_map = {"tg": TG_EXTENSION}
    else:
        # ``tg`` is registered, so it is already relevant for every
        # TG-supported system; setdefault keeps the section (and its
        # unknown-flag errors) for unsupported systems too, which the
        # explicit system check below rejects with a clearer message.
        ext_map = dict(relevant_extensions(ctx.system))
        ext_map.setdefault("tg", TG_EXTENSION)
    resolve_parameters(
        argv,
        toml_path=toml_path,
        extensions=tuple(ext_map.values()),
        prog=_PROG,
    )

    from ..extensions import force_params, probes_params

    ignored = [
        name
        for name, values in (
            ("probes", probes_params),
            ("force", force_params),
        )
        if values.modes is not None
    ]
    if ignored:
        print(
            f"[tg] note: the {', '.join(ignored)} section(s) configure "
            "solver runs; the transient-growth driver ignores them."
        )

    # ``tg.parameters`` names the TOML to load, so it can only come
    # from the command line -- a value set inside a TOML would name a
    # (different) file that was never read.
    if toml_flag is None and tg_params.parameters is not None:
        raise SystemExit(
            "dnsjax: error: tg.parameters is CLI-only "
            "(--tg.parameters PATH selects the TOML to load; setting "
            "it inside a [tg] section has no effect)."
        )

    # Snapshot user intent *before* forcing.
    u_grid_user_set = ("phys", "u_grid") in _user_set_fields
    for f in _TG_OWNED_STEP:
        if ("step", f) in _user_set_fields:
            print(
                f"[tg] note: step.{f} is set by the TG driver "
                "(tg.dt / tg.corrector_tolerance / "
                "tg.max_corrector_iterations); the provided value "
                "is ignored."
            )
    if ("res", "double_precision") in _user_set_fields and not (
        params.res.double_precision
    ):
        print("[tg] note: transient growth always runs double precision.")

    if params.dist.np0 * params.dist.np1 != 1:
        raise SystemExit(
            "transient growth is single-device (np0*np1 must be 1); "
            "parallelise across profiles with separate processes."
        )
    if params.phys.system not in WALL_BOUNDED_TG_SYSTEMS:
        raise SystemExit(
            f"system {params.phys.system!r} is not supported "
            f"(choose one of {', '.join(WALL_BOUNDED_TG_SYSTEMS)})"
        )

    phys_force: dict[str, Any] = {}
    if not u_grid_user_set:
        phys_force["u_grid"] = 0.0
    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1},
            res={"double_precision": True},
            step={
                "scheme": "iterative-cn",
                "dt": tg_params.dt,
                "implicitness": 1.0,
                "implicit_mean_coupling": False,
                "corrector_tolerance": tg_params.corrector_tolerance,
                "max_corrector_iterations": (
                    tg_params.max_corrector_iterations
                ),
            },
            phys=phys_force,
        )
    )
    validate_parameters()
    padded_res.set_padded_resolution(params)

    print(
        "[tg] forced overrides: scheme=iterative-cn theta=1.0 "
        f"dt={tg_params.dt} implicit_mean_coupling=False "
        f"u_grid={derived_params.u_grid} np0=np1=1 "
        f"platform={params.dist.platform}"
    )


# ── Device-side pipeline ─────────────────────────────────────────


def _dispatch(system: str) -> tuple[Any, Any, str]:
    """Import the flow / geometry modules for *system*.

    Both come from the flow spec: the flow module is
    ``FlowSpec.flow_module``, the geometry module is the family
    (``cartesian``/``cylindrical``/``annular`` name their
    ``geometries.wall_bounded`` module directly).
    """
    import importlib

    from ..flows.registry import spec_for

    spec = spec_for(system)
    fmod = importlib.import_module(spec.flow_module)
    gmod = importlib.import_module(
        f"dnsjax.geometries.wall_bounded.{spec.family.replace('-', '_')}"
    )
    return fmod, gmod, spec.family


def _linear_step(gmod: Any, fmod: Any = None):
    """Return the pure-linear implicit step ``(state, fourier, flow)``.

    Feeds the geometry's FFT-free linear coupling ``_l_bf`` as the RHS
    and no ``l_bf_fn`` (unsplit corrector), so the converged
    predict-and-fully-correct is the exact `$\\theta$`-implicit linear
    step of viscous + coupling + influence-matrix pressure, with the
    nonlinear self-advection never formed.

    On the cylindrical/annular flows every array crossing the raw
    stepper is in the decoupled `$u_\\pm$` solver basis, so the
    returned step wraps it and the propagator -- and everything this
    driver exports -- stays in **physical** components.  Cartesian
    carries physical components already, in both ``res.consistent_imm``
    states, so it is returned unwrapped.
    """
    from ..timestep import make_stepper

    raw = make_stepper(
        gmod._l_bf,
        gmod._predict,
        gmod._correct,
        gmod._norm,
        None,
        None,
    )
    step = raw[2]  # predict_and_fully_correct(state, *args)
    # The pair lives on the *flow* module (the cylindrical/annular
    # ones re-export the pure ``_base`` algebra), never the geometry
    # module -- so look there, and take the absence as "no basis".
    basis_mod = fmod if fmod is not None else gmod
    to_solver = getattr(basis_mod, "to_solver_basis", None)
    if to_solver is None:
        return step
    from_solver = basis_mod.from_solver_basis

    def step_physical(state, *args):
        out, err, num_c = step(to_solver(state), *args)
        return from_solver(out), err, num_c

    return step_physical


def _build_propagators(
    step: Any,
    fourier: Any,
    frozen_flow: Any,
    i2_arr: np.ndarray,
    i3_arr: np.ndarray,
    ny: int,
    tol: float,
) -> tuple[np.ndarray, float, int]:
    """Assemble the per-mode propagators over the selected modes.

    Steps the `$3 N_y$` wall-normal unit vectors placed at the selected
    (non-mean) modes only -- block-diagonality keeps every other mode
    zero, so the shared corrector norm reflects just these modes (a
    near-mean mode in a very long box would otherwise stall the whole
    step) -- reading column `$(c,j)$` off each.  Returns ``(phi,
    err_max, nc_max)`` with ``phi`` of shape ``(K, n, n)`` complex, `$n
    = 3 N_y$`, component-major index `$c N_y + j$`.
    """
    import jax.numpy as jnp

    from ..sharding import sharding

    n2, n3 = sharding.spec_shape[1], sharding.spec_shape[2]
    n = 3 * ny
    k = len(i2_arr)
    phi = np.empty((k, n, n), dtype=np.complex128)
    i2_dev = jnp.asarray(i2_arr)
    i3_dev = jnp.asarray(i3_arr)
    err_max = 0.0
    nc_max = 0
    for c in range(3):
        for j in range(ny):
            st = (
                jnp.zeros(
                    (3, ny, n2, n3),
                    dtype=sharding.complex_type,
                    out_sharding=sharding.spec_vector_shard,
                )
                .at[c, j, i2_dev, i3_dev]
                .set(1.0)  # only the selected (non-mean) modes
            )
            out, err, nc = step(st, fourier, frozen_flow)
            err_f = float(err)
            if not np.isfinite(err_f) or err_f > tol:
                raise SystemExit(
                    f"linear step failed to converge (component {c}, "
                    f"row {j}: err={err_f:.2e} > {tol:.1e}); reduce "
                    "--tg.dt or raise --tg.max_corrector_iterations."
                )
            err_max = max(err_max, err_f)
            nc_max = max(nc_max, int(nc))
            cols = np.asarray(out[:, :, i2_dev, i3_dev])  # (3, ny, K)
            phi[:, :, c * ny + j] = cols.reshape(n, k).T
    return phi, err_max, nc_max


def _energy_weight_diag(family: str, flow: Any) -> np.ndarray:
    """Diagonal energy weights `$W$` (component-major, length `$3N_y$`).

    Matches the solver's ``get_norm2*`` kinetic energy up to the
    per-mode ``k_metric`` / ``volume_fac`` constants (which cancel in
    `$G$`): `$m_c\\,w_y$` with `$m_c = 1$` in every native basis (kept
    as an explicit per-component metric for the exported
    ``energy_weights``' self-description).
    """
    w_y = np.asarray(flow.y_weights, dtype=np.float64)
    metric = _COMPONENT_METRIC[family]
    return np.concatenate([m * w_y for m in metric])


def _analyze_mode(
    phi_k: np.ndarray,
    w_diag: np.ndarray,
    dt: float,
    t_grid: np.ndarray,
    i2: int,
    i3: int,
    wn2: float,
    wn3: float,
    cfg: TGParams,
    jgrowth: Any,
    jfull: Any,
) -> ModeResult:
    """Full transient-growth analysis of one mode's propagator."""
    import jax.numpy as jnp

    ny = phi_k.shape[0] // 3
    n = phi_k.shape[0]
    # Subspace reduction.  The physical subspace S = range(Phi) is
    # separated from the constraint-violating null space by a wide
    # singular-value gap: the physical spectrum spans many orders (the
    # stiff wall-normal viscous modes reach sigma ~ 1/(dt*|lambda|) ~
    # 1e-7..1e-9), then a sharp cliff drops to the ~machine-epsilon null
    # floor (sigma/s0 ~ 1e-14).  ``rank_tol`` (relative, default 1e-11)
    # sits in that gap; the gap and reduced-conditioning checks catch a
    # threshold that lands mid-spectrum.
    u, s, _ = np.linalg.svd(phi_k)
    rank = int(np.sum(s > cfg.rank_tol * s[0]))
    rank = min(max(rank, 1), n)
    gap = (
        float(s[rank - 1] / s[rank])
        if rank < n and s[rank] > 0
        else float("inf")
    )
    if gap < cfg.rank_gap_min:
        raise SystemExit(
            f"mode ({i2},{i3}): no clean rank gap at rank {rank} "
            f"(s[{rank - 1}]/s[{rank}] = {gap:.2e} < "
            f"{cfg.rank_gap_min:.1e}); tail "
            f"{s[max(rank - 3, 0) : rank + 3]}; move --tg.rank_tol into "
            "the observed gap, or try raising --tg.dt."
        )
    v = u[:, :rank]  # (n, r) range basis
    phi_s = v.conj().T @ phi_k @ v  # (r, r)

    # Generator via the eigendecomposition of Phi_S (not by inverting
    # I - dt*A, which would amplify the stiff-mode errors).  Phi_S =
    # (I - dt A_S)^{-1} shares eigenvectors with A_S, so
    # eig(Phi_S) = (mu, Y) gives the generator eigenvalues lambda =
    # (1 - 1/mu)/dt directly.  The stiff FD wall-clustering modes get
    # huge negative lambda; exp(t*lambda) then *underflows* to 0 for
    # t > 0 (they decay instantly) rather than overflowing a dense
    # matrix exponential.
    mu, y_vec = np.linalg.eig(phi_s)
    lam = (1.0 - 1.0 / mu) / dt
    eig_res = float(
        np.linalg.norm(phi_s @ y_vec - y_vec * mu) / np.linalg.norm(phi_s)
    )
    # For the ultra-stiff modes mu ~ 0, and lambda = (1 - 1/mu)/dt
    # amplifies the eig round-off: a tiny spurious phase in mu can flip
    # Re(lambda) to a huge *positive* value (unphysical growth that
    # overflows exp).  A mode the probe cannot resolve (|mu| <= 1/2)
    # cannot physically grow faster than the resolved spectral abscissa;
    # clamp any such mode to instant decay so it cannot pollute the
    # reported spectrum.
    resolved = np.abs(mu) > 0.5
    if not resolved.any():
        raise SystemExit(
            f"mode ({i2},{i3}): no probe-resolved eigenvalues (every "
            f"|mu| <= 1/2 at --tg.dt {dt:g}); raise --tg.dt to widen "
            "the resolved window."
        )
    omega_res = float(np.max(lam.real[resolved]))
    spurious = (~resolved) & (lam.real > omega_res)
    lam = np.where(spurious, -1e30, lam)

    # Energy metric and the eigenvectors in energy coordinates.
    m_mat = v.conj().T @ (w_diag[:, None] * v)  # V^H W V
    f_mat = np.linalg.cholesky(m_mat).conj().T  # upper, M = F^H F
    f_inv = np.linalg.inv(f_mat)
    e_mat = f_mat @ y_vec  # eigenvectors in energy coords

    # Restrict to the probe-resolved eigenspace *before* measuring
    # growth.  Q is orthonormal in the energy coordinates -- so the
    # 2-norm there is the energy norm -- and the reduced generator
    # follows from invariance (E_res = Q R, so A_res = Q^H A Q =
    # R diag(lam_res) R^{-1}, already in eigenform: no expm needed).
    # Carrying the *unresolved* modes into G instead would make the
    # propagator a non-orthogonal spectral projector the instant their
    # exp(t*lambda) dies: G would jump from 1 to the projector norm at
    # t = 0+ (a discontinuity of many orders, mesh-dependent and
    # physically meaningless) and mask the true optimum.  On the
    # restriction G(0) = 1 holds continuously, and G is a rigorous
    # *lower bound* on the full-space growth that converges from below
    # as the resolved window widens (larger wall-normal resolution
    # res.ny/nr at fixed --tg.dt).
    q_mat, r_mat = np.linalg.qr(e_mat[:, resolved])
    lam_res = lam[resolved]
    cond_e = np.linalg.cond(r_mat)
    if not np.isfinite(cond_e) or cond_e > 1e12:
        raise SystemExit(
            f"mode ({i2},{i3}): eigenvector matrix is ill-conditioned "
            f"(cond = {cond_e:.2e}); Phi_S is (near) defective -- "
            "reduce the wall-normal resolution (res.ny/nr) or "
            "change --tg.dt."
        )
    r_inv = np.linalg.inv(r_mat)
    a_res = (r_mat * lam_res[None, :]) @ r_inv
    e_dev = jnp.asarray(r_mat)
    lam_dev = jnp.asarray(lam_res)
    w_dev = jnp.asarray(r_inv)

    def geval(t: float) -> float:
        return float(
            np.asarray(jgrowth(e_dev, lam_dev, w_dev, jnp.asarray([t])))[0]
        )

    # Growth curve G(t) = sigma_max(E diag(exp(t*lambda)) W)^2.
    g_vals = []
    for lo in range(0, len(t_grid), cfg.t_chunk):
        ts = jnp.asarray(t_grid[lo : lo + cfg.t_chunk])
        g_vals.append(np.asarray(jgrowth(e_dev, lam_dev, w_dev, ts)))
    g_curve = np.concatenate(g_vals) ** 2
    g_curve[0] = 1.0  # exact at t=0

    # Optimal time by golden-section refinement of the grid maximum.
    imax = int(np.argmax(g_curve))
    if 0 < imax < len(t_grid) - 1:
        t_opt = _golden_max(geval, t_grid[imax - 1], t_grid[imax + 1])
    else:
        t_opt = float(t_grid[imax])
        if imax == len(t_grid) - 1:
            print(
                f"[tg] mode ({i2},{i3}): G still rising at t_max; "
                "increase --tg.t_max."
            )
    s_opt, u1, v1 = jfull(e_dev, lam_dev, w_dev, float(t_opt))
    sigma_opt = float(np.asarray(s_opt)[0])
    g_max = sigma_opt**2

    # Optimal input / response lifted to the full state (3, Ny), back
    # out through the resolved eigenspace (Q), the energy factor (F)
    # and the range basis (V).
    def _lift(vec_r: np.ndarray, scale: float) -> np.ndarray:
        full = v @ (f_inv @ (q_mat @ vec_r))
        return scale * full.reshape(3, ny)

    opt_input = _lift(np.asarray(v1), 1.0)
    opt_response = _lift(np.asarray(u1), sigma_opt)

    # Optimal pair at every requested time (optional).
    opt_input_t = opt_response_t = sigma_t = None
    if cfg.save_all_times:
        in_t = np.empty((len(t_grid), 3, ny), dtype=np.complex128)
        re_t = np.empty_like(in_t)
        sig_t = np.empty(len(t_grid), dtype=np.float64)
        for it, t in enumerate(t_grid):
            st, u1t, v1t = jfull(e_dev, lam_dev, w_dev, float(t))
            st0 = float(np.asarray(st)[0])
            sig_t[it] = st0
            in_t[it] = _lift(np.asarray(v1t), 1.0)
            re_t[it] = _lift(np.asarray(u1t), st0)
        opt_input_t, opt_response_t, sigma_t = in_t, re_t, sig_t

    # Spectrum and abscissae.  The spectral abscissa uses all modes; the
    # numerical abscissa (max instantaneous energy growth) is taken on
    # the *probe-resolved* subspace |dt*lambda| < 1 (|mu| > 1/2): faster
    # modes are heavily compressed by the backward-Euler probe (mu -> 0),
    # so their extracted lambda -- and the near-wall FD modes' spurious
    # non-normal instantaneous growth -- are unresolved and would swamp
    # the diagnostic with a mesh-dependent value.
    order = np.argsort(-lam.real)
    lam_sorted = lam[order]
    spectral_abscissa = float(lam_sorted[0].real)
    sym = 0.5 * (a_res + a_res.conj().T)
    numerical_abscissa = float(np.max(np.linalg.eigvalsh(sym)))
    op_v = op_f = op_q = op_a = op_lam = None
    if cfg.save_operator:
        # ``A_res`` is exactly the generator whose growth is reported,
        # so ``operator_tools.growth_curve`` on the exported bundle
        # reproduces the stored ``G`` (not merely approximates it).
        op_v, op_f, op_q, op_a = v, f_mat, q_mat, a_res
        op_lam = lam_res
    extraction_residual = eig_res
    n_eig = min(cfg.n_eig, rank)

    return ModeResult(
        i2=i2,
        i3=i3,
        wn2=wn2,
        wn3=wn3,
        rank=rank,
        sv_gap=gap,
        extraction_residual=extraction_residual,
        G=g_curve,
        G_max=g_max,
        t_opt=float(t_opt),
        sigma_opt=sigma_opt,
        spectral_abscissa=spectral_abscissa,
        numerical_abscissa=numerical_abscissa,
        eigvals=lam_sorted[:n_eig],
        singular_values=s,
        opt_input=opt_input,
        opt_response=opt_response,
        opt_input_t=opt_input_t,
        opt_response_t=opt_response_t,
        sigma_t=sigma_t,
        op_V=op_v,
        op_F=op_f,
        op_Q=op_q,
        op_A=op_a,
        op_lam=op_lam,
    )


def _golden_max(g: Any, lo: float, hi: float, iters: int = 40) -> float:
    """Golden-section search for the interior maximum of ``g(t)``."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0

    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    gc, gd = g(c), g(d)
    for _ in range(iters):
        if gc < gd:
            lo, c, gc = c, d, gd
            d = lo + inv_phi * (hi - lo)
            gd = g(d)
        else:
            hi, d, gd = d, c, gc
            c = hi - inv_phi * (hi - lo)
            gc = g(c)
        if hi - lo < 1e-6 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def _wall_bc_check(
    frozen: Any, builtin: Any, family: str, tol: float
) -> float:
    """Verify the profile matches the flow's wall values.

    Compares the frozen base flow to the builtin (laminar) one at the
    wall rows -- ``[-1]`` for the cylindrical family (the sole wall; no
    `$r=0$` wall), ``[0, -1]`` otherwise.  Returns the relative wall
    residual; raises if it exceeds *tol*.
    """
    fb = np.asarray(frozen.base_flow[:, :, 0, 0])
    bb = np.asarray(builtin.base_flow[:, :, 0, 0])
    rows = [-1] if family == "cylindrical" else [0, -1]
    scale = float(np.max(np.abs(bb))) + 1e-30
    resid = float(np.max(np.abs(fb[:, rows] - bb[:, rows]))) / scale
    if resid > tol:
        raise SystemExit(
            f"profile wall values differ from the flow BCs by "
            f"{resid:.2e} (> {tol:.1e}); the total profile must satisfy "
            "the same wall boundary conditions as the laminar flow."
        )
    return resid


# ── Output ───────────────────────────────────────────────────────


def _write_summary(
    path: Path,
    results: list[ModeResult],
    labels: tuple[str, str],
    profile_file: Path,
    interpolated: bool,
    wall_resid: float,
    err_max: float,
    nc_max: int,
) -> None:
    """Write (and print) the human-readable summary table."""
    lines: list[str] = []
    lines.append(f"# transient growth: {params.phys.system}")
    re_str = f"re={params.phys.re}"
    if params.phys.system == "taylor-couette":
        re_str = (
            f"re1={params.phys.re1} re2={params.phys.re2} eta={params.geo.eta}"
        )
    elif params.phys.system == "quasi-keplerian":
        re_str = (
            f"re1={params.phys.re1} r_omega={params.phys.r_omega} "
            f"eta={params.geo.eta} (re2={params.phys.re2})"
        )
    lines.append(
        f"# {re_str} tilt={params.geo.tilt_degree} "
        f"lx={params.geo.lx} lz={params.geo.lz}"
    )
    lines.append(
        f"# res: nx={params.res.nx} ny={params.res.ny} "
        f"nz={params.res.nz} fd_order={params.res.fd_order} "
        f"grid={params.geo.grid_type}"
    )
    lines.append(
        f"# overrides: theta=1 dt={params.step.dt} "
        f"u_grid={derived_params.u_grid} "
        f"corr_tol={params.step.corrector_tolerance}"
    )
    lines.append(
        f"# profile: {profile_file.name} interpolated={interpolated} "
        f"wall_residual={wall_resid:.2e}"
    )
    lines.append(
        f"# propagator: corrector_error_max={err_max:.2e} "
        f"corrector_iterations_max={nc_max}"
    )
    lines.append(
        f"# columns: i2 i3 {labels[0]} {labels[1]} rank G_max t_opt "
        "spec_absc num_absc extract_res sv_gap"
    )
    for r in sorted(results, key=lambda m: -m.G_max):
        lines.append(
            f"{r.i2:4d} {r.i3:4d} {r.wn2:12.6g} {r.wn3:12.6g} "
            f"{r.rank:5d} {r.G_max:14.6e} {r.t_opt:12.5g} "
            f"{r.spectral_abscissa:13.5e} {r.numerical_abscissa:13.5e} "
            f"{r.extraction_residual:11.2e} {r.sv_gap:9.2e}"
        )
    best = max(results, key=lambda m: m.G_max)
    lines.append(
        f"# max G_max = {best.G_max:.6e} at mode "
        f"({best.i2},{best.i3}), t_opt = {best.t_opt:.5g}"
    )
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text, end="")


def _write_npz(
    path: Path,
    results: list[ModeResult],
    family: str,
    labels: tuple[str, str],
    cfg: TGParams,
    profile_file: Path,
    y_code: np.ndarray,
    y_user: np.ndarray,
    u_user: np.ndarray,
    u_code: np.ndarray,
    interpolated: bool,
    t_grid: np.ndarray,
    w_diag: np.ndarray,
    err_max: float,
    nc_max: int,
) -> None:
    """Write the self-describing per-profile ``.npz`` bundle."""
    results = sorted(results, key=lambda m: (m.i2, m.i3))

    def stack(attr: str) -> np.ndarray:
        return np.stack([getattr(r, attr) for r in results])

    out: dict[str, Any] = {
        "readme": (
            "dnsjax transient growth. Modes indexed (i2, i3): i2 = "
            "kz/m axis, i3 = kx/axial axis. Vectors are (3, Ny) in the "
            "stored physical basis "
            + str(_COMPONENT_LABELS[family])
            + ". G is the energy growth on t_grid."
        ),
        "system": params.phys.system,
        "family": family,
        "params_json": params.model_dump_json(),
        "tg_config_json": _config_json(cfg),
        "profile_file": str(profile_file),
        "component_labels": np.asarray(_COMPONENT_LABELS[family]),
        "wavenumber_labels": np.asarray(labels),
        "code_grid": y_code,
        "user_grid": y_user,
        "user_values": u_user,
        "profile_on_grid": u_code,
        "interpolated": bool(interpolated),
        "energy_weights": w_diag,
        "t_grid": t_grid,
        "corrector_error_max": err_max,
        "corrector_iterations_max": nc_max,
        "mode_i2": np.asarray([r.i2 for r in results]),
        "mode_i3": np.asarray([r.i3 for r in results]),
        "mode_wn2": np.asarray([r.wn2 for r in results]),
        "mode_wn3": np.asarray([r.wn3 for r in results]),
        "rank": np.asarray([r.rank for r in results]),
        "sv_gap": np.asarray([r.sv_gap for r in results]),
        "extraction_residual": np.asarray(
            [r.extraction_residual for r in results]
        ),
        "G": stack("G"),
        "G_max": np.asarray([r.G_max for r in results]),
        "t_opt": np.asarray([r.t_opt for r in results]),
        "sigma_opt": np.asarray([r.sigma_opt for r in results]),
        "spectral_abscissa": np.asarray(
            [r.spectral_abscissa for r in results]
        ),
        "numerical_abscissa": np.asarray(
            [r.numerical_abscissa for r in results]
        ),
        "eigvals": stack("eigvals"),
        "singular_values": stack("singular_values"),
        "opt_input": stack("opt_input"),
        "opt_response": stack("opt_response"),
    }
    if cfg.save_all_times:
        out["opt_input_t"] = stack("opt_input_t")
        out["opt_response_t"] = stack("opt_response_t")
        out["sigma_t"] = stack("sigma_t")
    np.savez(path, **out)
    print(f"[tg] wrote {path}")


def _write_operator_npz(
    path: Path,
    results: list[ModeResult],
    family: str,
    cfg: TGParams,
    profile_file: Path,
    y_code: np.ndarray,
    u_code: np.ndarray,
    w_diag: np.ndarray,
    t_grid: np.ndarray,
) -> None:
    r"""Write the ``--tg.save_operator`` bundle (``<stem>_tg_op.npz``).

    Per mode (keys suffixed ``_{i2}_{i3}``; ranks are ragged across
    modes, so no stacking):

    - ``A``: the reduced generator (``r_res x r_res``) in an
      orthonormal basis of the **energy coordinates**, restricted to
      the probe-resolved eigenspace (`$|\mu| > 1/2$`).  The clamped
      stiff eigenvalues (`$-10^{30}$`) would poison any explicitly
      formed operator (a matrix exponential or Lyapunov solve), and
      truncating modes with `$|\lambda| \gtrsim 1/\Delta t$` changes
      downstream results by the same `$O(\Delta t)$` class the
      numerical-abscissa compression already accepts.
    - ``Q``: that basis (``r x r_res``), ``F``: the upper Cholesky of
      the subspace energy metric (``M = F^H F``, ``r x r``), ``V``:
      the divergence-free subspace basis (``n x r``, ``n = C Ny``
      component-major, index ``c*Ny + j``), ``lam``: ``eig(A)``.

    Coordinate contract (also in the ``readme`` key): full state
    `$q = V F^{-1} Q\,a$`; projection `$a = Q^H F V^H q$`; the plain
    2-norm of `$a$` is the energy seminorm `$q^H\,\mathrm{diag}(
    \mathtt{energy\_weights})\,q$`.  The solver's volume-averaged
    `$E'$` of a physically real single-``(i2,i3)``-mode state is
    `$\lVert a\rVert_2^2 / \mathtt{volume\_fac}$` (the real-FFT
    ``mode_k_metric`` and the ``i3 = 0`` conjugate partner supply the
    same factor 2).

    Consumers: :mod:`dnsjax.analysis.response.operator_tools`
    (controllability modes, growth curves, subspace restriction).
    """
    results = sorted(results, key=lambda m: (m.i2, m.i3))
    out: dict[str, Any] = {
        "readme": (
            "dnsjax transient-growth operator export. Per mode "
            "(suffix _{i2}_{i3}): A = reduced generator (r_res x "
            "r_res) in an orthonormal basis Q of the energy "
            "coordinates, restricted to the probe-resolved "
            "eigenspace; F = chol (upper) of the subspace energy "
            "metric; V = subspace basis (n x r, n = C*Ny, "
            "component-major c*Ny+j); lam = eig(A). Contract: "
            "q = V F^-1 Q a; a = Q^H F V^H q; ||a||_2^2 = "
            "q^H diag(energy_weights) q; solver E' of a real "
            "single-mode state = ||a||_2^2 / volume_fac."
        ),
        "system": params.phys.system,
        "family": family,
        "params_json": params.model_dump_json(),
        "tg_config_json": _config_json(cfg),
        "profile_file": str(profile_file),
        "component_labels": np.asarray(_COMPONENT_LABELS[family]),
        "code_grid": y_code,
        "profile_on_grid": u_code,
        "energy_weights": w_diag,
        "tg_dt": params.step.dt,
        "volume_fac": float(derived_params.volume_fac),
        "t_grid": t_grid,
        "mode_i2": np.asarray([r.i2 for r in results]),
        "mode_i3": np.asarray([r.i3 for r in results]),
        "mode_wn2": np.asarray([r.wn2 for r in results]),
        "mode_wn3": np.asarray([r.wn3 for r in results]),
        "mode_k_metric": np.asarray(
            [1.0 if r.i3 == 0 else 2.0 for r in results]
        ),
        "rank": np.asarray([r.rank for r in results]),
        "rank_resolved": np.asarray([r.op_A.shape[0] for r in results]),
    }
    for r in results:
        sfx = f"_{r.i2}_{r.i3}"
        out["A" + sfx] = r.op_A
        out["Q" + sfx] = r.op_Q
        out["F" + sfx] = r.op_F
        out["V" + sfx] = r.op_V
        out["lam" + sfx] = r.op_lam
    np.savez(path, **out)
    print(f"[tg] wrote {path}")


# ── Single-mode state helpers (shared with scripts) ─────────────

#: family -> the geometry module's kinetic-energy norm function.
_NORM2_BY_FAMILY = {
    "cartesian": "get_norm2",
    "cylindrical": "get_norm2_cyl",
    "annular": "get_norm2_annular",
}


def single_mode_state(vec: np.ndarray, i2: int, i3: int) -> Array:
    r"""Zero spectral state carrying *vec* at global mode ``(i2, i3)``.

    *vec* is ``(C, Ny)`` complex in the stored component basis.  On
    the real-FFT plane (``i3 == 0``, ``i2 > 0``) the conjugate partner
    at ``((nz-1) - i2, 0)`` is filled so the physical field is real --
    the plain conjugate, every native component being the transform of
    a real field.  The mean mode ``(0, 0)`` is a valid target (no
    partner; the caller owns its reality).

    Requires the unpadded single-device spectral layout (storage
    index == true mode index), which the transient-growth driver and
    the offline scripts guarantee (``np0 = np1 = 1``).

    Shared by the ``--tg.export_snapshot`` seed writer and
    ``scripts/snapshot_perturb.py``.
    """
    import jax.numpy as jnp

    from ..sharding import sharding

    vec = np.asarray(vec)
    n2, n3 = sharding.spec_shape[1], sharding.spec_shape[2]
    n2_true = params.res.nz - 1
    if n2 != n2_true:
        raise SystemExit(
            "single_mode_state requires the unpadded single-device "
            f"spectral layout (axis 2 holds {n2} slots for {n2_true} "
            "true modes; run with np0 = np1 = 1)"
        )
    state = jnp.zeros(
        (vec.shape[0], vec.shape[1], n2, n3),
        dtype=sharding.complex_type,
        out_sharding=sharding.spec_vector_shard,
    )
    state = state.at[:, :, i2, i3].set(jnp.asarray(vec))
    # Conjugate partner on the real-FFT (i3=0) plane for a real field.
    if i3 == 0 and i2 > 0:
        partner = np.conj(vec)
        state = state.at[:, :, n2_true - i2, 0].set(jnp.asarray(partner))
    return state


def mode_state_energy(
    state: Array, family: str, gmod: Any, flow: Any
) -> float:
    r"""The solver's volume-averaged perturbation energy `$E'$`.

    ``get_norm2*``-based, so amplitude conventions built on it (the
    ``--tg.export_amplitude`` seed, ``snapshot_perturb``'s
    ``--amplitude-energy``) agree exactly with the solver's own
    ``E'`` diagnostic.
    """
    norm2 = getattr(gmod, _NORM2_BY_FAMILY[family])
    return float(norm2(state, gmod.fourier.k_metric, flow.y_weights)) / 2.0


@contextmanager
def _seed_metadata_params():
    """Production-default metadata for an exported seed snapshot.

    The driver forces analysis stepping (backward-Euler ``theta = 1``,
    ``tg.dt``, a tight corrector) and the lab frame onto the live
    ``params``; embedding those in the exported seed would make a
    production resume inherit them silently through the snapshot
    parameter layer.  A seed is pure state plus its
    trajectory-defining ``phys``/``geo``/``res`` identity, so for the
    metadata dump the ``step`` section, the moving-frame speed, and
    the solver-run extension sections (``probes``/``force``, possibly
    configured by a shared production TOML this driver ignores) are
    swapped to their production defaults, then restored.
    """
    from ..extensions import EXTENSIONS
    from ..flow_spec import UNSET
    from ..flows.registry import spec_for
    from ..parameters import TimeStepping

    recorded = {
        name: ext for name, ext in EXTENSIONS.items() if ext.record_in_metadata
    }
    saved_step = params.step
    saved_u_grid = params.phys.u_grid
    saved_ext = {
        name: ext.values.model_dump() for name, ext in recorded.items()
    }
    default_u = spec_for(params.phys.system).default_for("phys", "u_grid")
    params.step = TimeStepping()
    params.phys.u_grid = None if default_u is UNSET else default_u
    for ext in recorded.values():
        fresh = ext.model()
        for field_name in ext.model.model_fields:
            setattr(ext.values, field_name, getattr(fresh, field_name))
    try:
        yield
    finally:
        params.step = saved_step
        params.phys.u_grid = saved_u_grid
        for name, values in saved_ext.items():
            for field_name, value in values.items():
                setattr(recorded[name].values, field_name, value)


def _export_snapshot(
    out_dir: Path,
    stem: str,
    results: list[ModeResult],
    fmod: Any,
    gmod: Any,
    family: str,
    cfg: TGParams,
) -> None:
    """Export a chosen mode's optimal perturbation as a snapshot.

    The embedded metadata carries production-default ``step``/frame/
    extension sections (:func:`_seed_metadata_params`), not this
    driver's forced analysis configuration.
    """
    import jax

    from ..snapshot import save_snapshot

    try:
        (pair,) = harmonics.parse_mode_pairs(cfg.export_snapshot)
    except ValueError as exc:
        raise SystemExit(
            f"--tg.export_snapshot takes one 'i2,i3' pair: {exc}"
        ) from None
    i2, i3 = pair
    match = [r for r in results if r.i2 == i2 and r.i3 == i3]
    if not match:
        raise SystemExit(
            f"--tg.export_snapshot mode ({i2},{i3}) was not computed"
        )
    r = match[0]
    vec = r.opt_input if cfg.export_which == "input" else r.opt_response
    state = single_mode_state(vec, i2, i3)
    energy = mode_state_energy(state, family, gmod, fmod.flow)
    scale = float(np.sqrt(cfg.export_amplitude / energy))
    state = state * scale

    path = out_dir / f"{stem}_tg_seed_m{i2}_{i3}.tar"
    with _seed_metadata_params():
        save_snapshot(jax.block_until_ready(state), 0.0, 0, path, isnap=0)
    print(
        f"[tg] wrote {path} (E'={cfg.export_amplitude:g}); resume with\n"
        f"     mpirun -np 1 python -m dnsjax "
        f"--phys.system {params.phys.system} --init.snapshot {path}"
    )


# ── Orchestration ────────────────────────────────────────────────


def _run(cfg: TGParams) -> int:
    """Process every profile file (parameters already resolved)."""
    import jax
    import jax.numpy as jnp

    from ..sharding import sharding

    fmod, gmod, family = _dispatch(params.phys.system)
    fourier = gmod.fourier
    step = _linear_step(gmod, fmod)
    ny = params.res.ny
    y_code = np.asarray(derived_params.wall_normal_grid, dtype=np.float64)
    w_diag = _energy_weight_diag(family, fmod.flow)
    n2, n3 = sharding.spec_shape[1], sharding.spec_shape[2]
    i2_arr, i3_arr = _select_modes(cfg.modes, n2, n3)
    wn2_all, wn3_all, labels = _wavenumber_arrays(family)
    # Forced np0 = np1 = 1 means sharding never pads the spectral
    # axes, so mode indices map 1:1 onto the harmonic arrays.
    assert len(wn2_all) == n2 and len(wn3_all) == n3
    t_max = cfg.t_max if cfg.t_max is not None else 0.25 * params.phys.re
    t_grid = np.linspace(0.0, t_max, cfg.nt)

    def _propagator(e_mat: Array, lam: Array, w_mat: Array, t: Array):
        # B(t) = E diag(exp(t*lambda)) W in energy coordinates; stiff
        # modes (huge negative Re lambda) underflow to zero for t > 0.
        return e_mat @ (jnp.exp(t * lam)[:, None] * w_mat)

    @jax.jit
    def jgrowth(e_mat: Array, lam: Array, w_mat: Array, ts: Array) -> Array:
        return jax.vmap(
            lambda t: jnp.linalg.svd(
                _propagator(e_mat, lam, w_mat, t), compute_uv=False
            )[0]
        )(ts)

    @jax.jit
    def jfull(
        e_mat: Array, lam: Array, w_mat: Array, t: Array
    ) -> tuple[Array, Array, Array]:
        u_m, s_m, vh_m = jnp.linalg.svd(_propagator(e_mat, lam, w_mat, t))
        return s_m, u_m[:, 0], vh_m[0, :].conj()

    n_bytes = len(i2_arr) * (3 * ny) ** 2 * 16
    print(
        f"[tg] {len(i2_arr)} mode(s), n=3*Ny={3 * ny}; propagator "
        f"host memory ~{n_bytes / 1e6:.0f} MB; {3 * ny} linear steps "
        "per profile."
    )

    out_dir = Path(tg_params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for profile_file in _gather_profiles(tg_params.profile):
        try:
            _process_profile(
                profile_file,
                cfg,
                fmod,
                gmod,
                family,
                step,
                fourier,
                ny,
                y_code,
                w_diag,
                i2_arr,
                i3_arr,
                wn2_all,
                wn3_all,
                labels,
                t_grid,
                jgrowth,
                jfull,
                out_dir,
            )
        except SystemExit as exc:
            failures += 1
            print(f"[tg] {profile_file.name}: FAILED -- {exc}")
    if failures:
        print(f"[tg] {failures} profile(s) failed.")
        return 1
    return 0


def _process_profile(
    profile_file: Path,
    cfg: TGParams,
    fmod: Any,
    gmod: Any,
    family: str,
    step: Any,
    fourier: Any,
    ny: int,
    y_code: np.ndarray,
    w_diag: np.ndarray,
    i2_arr: np.ndarray,
    i3_arr: np.ndarray,
    wn2_all: np.ndarray,
    wn3_all: np.ndarray,
    labels: tuple[str, str],
    t_grid: np.ndarray,
    jgrowth: Any,
    jfull: Any,
    out_dir: Path,
) -> None:
    """Run the full pipeline for one profile file."""
    import jax.numpy as jnp

    from ..sharding import sharding

    y_user, u_user = _read_profile(profile_file)
    u_code, interpolated = _regrid_profile(
        y_user, u_user, y_code, cfg.interp_order, cfg.grid_match_tol
    )
    frozen = fmod.frozen_profile_flow(
        jnp.asarray(u_code, dtype=sharding.float_type)
    )
    wall_resid = _wall_bc_check(frozen, fmod.flow, family, cfg.wall_bc_tol)

    phi, err_max, nc_max = _build_propagators(
        step,
        fourier,
        frozen,
        i2_arr,
        i3_arr,
        ny,
        params.step.corrector_tolerance,
    )
    results = [
        _analyze_mode(
            phi[k],
            w_diag,
            params.step.dt,
            t_grid,
            int(i2_arr[k]),
            int(i3_arr[k]),
            float(wn2_all[i2_arr[k]]),
            float(wn3_all[i3_arr[k]]),
            cfg,
            jgrowth,
            jfull,
        )
        for k in range(len(i2_arr))
    ]

    stem = profile_file.stem
    _write_summary(
        out_dir / f"{stem}_tg_summary.txt",
        results,
        labels,
        profile_file,
        interpolated,
        wall_resid,
        err_max,
        nc_max,
    )
    _write_npz(
        out_dir / f"{stem}_tg.npz",
        results,
        family,
        labels,
        cfg,
        profile_file,
        y_code,
        y_user,
        u_user,
        u_code,
        interpolated,
        t_grid,
        w_diag,
        err_max,
        nc_max,
    )
    if cfg.save_operator:
        _write_operator_npz(
            out_dir / f"{stem}_tg_op.npz",
            results,
            family,
            cfg,
            profile_file,
            y_code,
            u_code,
            w_diag,
            t_grid,
        )
    if cfg.export_snapshot is not None:
        _export_snapshot(out_dir, stem, results, fmod, gmod, family, cfg)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    _configure_parameters(args)  # exits on --help / --sample-toml
    if tg_params.profile is None:
        raise SystemExit(
            f"{_PROG}: error: --tg.profile is required (a profile "
            "file or a folder of profile files)"
        )
    configure_jax_platform(params.dist.platform, double_precision=True)
    print("Code version:", git_hash(), flush=True)
    # A frozen copy: later mutation of the singleton (another
    # entry point, tests) cannot drift a run in progress.
    return _run(tg_params.model_copy(deep=True))


if __name__ == "__main__":
    sys.exit(main())
