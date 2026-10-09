r"""Difference-field pressure for the wall-normal-resolved budget.

The volume-averaged budget of Egerique-de-la-Concha & Hwang (*J.
Fluid Mech.* **1036**, A52, 2026) carries no pressure term, and is
right not to: `$\Delta\mathbf{u}\cdot\nabla\Delta p =
\nabla\cdot(\Delta p\,\Delta\mathbf{u})$` is a divergence, so it
integrates away over the domain.  That cancellation belongs to the
*integral*, not the integrand.  Resolve the budget in `$y$` (as
:func:`dnsjax.twin.diagnostics.twin_ybudget` does) and the term
reappears as a wall-normal flux -- zero net, but comparable to
production near the wall, and the mechanism by which the wall blocks
and redistributes a scale's energy.  Resolve it per velocity
component and it becomes the pressure--strain redistribution: the
only way the mean shear's energy reaches `$\Delta v$`, whose
production against the mean profile vanishes identically (its
production against the reference *fluctuations* does not).

This module recovers `$\Delta\hat{p}$` so that term can be measured
rather than left as a hole.  The solve itself -- the Poisson problem
on the interior rows and the influence-matrix wall closure
`$(D_1\,\partial_t\Delta\hat v)|_w = 0$` -- is the geometry's
:class:`~dnsjax.geometries.wall_bounded._cartesian_pressure.PoissonPressure`,
whose module docstring derives it; nothing in it reads the field's
nonlinear term except as a source.  The same object serves the
reduced-resolution snapshots' static pressure of one state
(:mod:`dnsjax.lowres`), so a twin run holds one operator for both.

The source
----------
`$\mathcal{N}$` is the difference field's nonlinear term,

.. math::
    \mathcal{N} = -(\mathbf{u}^{(1)}\cdot\nabla)\Delta\mathbf{u}
    - (\Delta\mathbf{u}\cdot\nabla)\mathbf{u}^{(1)}
    - (\Delta\mathbf{u}\cdot\nabla)\Delta\mathbf{u} ,

in the **convective** form by default
(:func:`dnsjax.twin.diagnostics._convective_sources`), which makes
`$\Delta\hat p$` the difference of the two members' static pressures.
Under ``twin.rotational_ybudget`` it is instead the rotational term the
solver itself integrates (:mod:`dnsjax.rhs`),
`$\mathbf{u}^{(1)}\times\Delta\boldsymbol{\omega} + \Delta\mathbf{u}
\times\boldsymbol{\omega}^{(1)} + \Delta\mathbf{u}\times\Delta
\boldsymbol{\omega}$`, and then `$\Delta\hat p$` comes back as the
difference of the two members' **Bernoulli** pressures,
`$\Delta p + \mathbf{u}^{(1)}\!\cdot\Delta\mathbf{u} +
|\Delta\mathbf{u}|^2/2$`.  The two differ by a gradient, so the total
work is unchanged; the `$y$`-density is not
(:mod:`dnsjax.twin.diagnostics`, "Two budget forms").

The mean mode
-------------
`$\Delta\hat{v} \equiv 0$` at `$(k_z, k_x) = (0,0)$` and the horizontal
gradients vanish there, so the *fluctuating* pressure does no work at
that mode whatever its gauge.  What *does* act on the mean mode is the
applied driving `$-\Delta\Pi$` (`$\Pi$` the mean pressure gradient --
the sign convention is fixed in :mod:`dnsjax.twin.diagnostics`,
"Mean-mode driving"); that density is added by
:func:`dnsjax.twin.diagnostics.twin_ybudget`, which has the mean
profile to hand.

Cost
----
Resident, held for the run -- so it is built only when a consumer is
enabled (``twin.it_ybudget``, ``twin.it_budget3d``, or ``[lowres]``
pressure under either reduced-snapshot cadence): one extra factored
operator the size of ``flow.Lk_op`` plus the two homogeneous columns
(~12 % on top at ``fd_order = 8``) and ``M_inv``.  Per sample: one
banded solve and a handful of `$D_1$` matvecs, against the 33 field
transforms the budget itself costs (21 in the rotational form).
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array
from jax import numpy as jnp

from ..geometries.wall_bounded._base import apply_y_matrix
from ..geometries.wall_bounded._cartesian_pressure import PoissonPressure
from ..geometries.wall_bounded.cartesian import CartesianFlow, Fourier
from ..parameters import derived_params, params
from ..sharding import register_dataclass_pytree


@register_dataclass_pytree
@dataclass(init=False)
class DifferencePressure(PoissonPressure):
    r"""The difference field's pressure, and the work it does.

    :class:`~dnsjax.geometries.wall_bounded._cartesian_pressure.PoissonPressure`
    -- the factored operator, its homogeneous columns and the wall
    closure's influence matrix, built once -- plus the two readings
    the twin budget makes of the pressure it solves for.  Registered
    as a pytree in its own right (registration is per class); its
    fields are the base class's.
    """

    def work_density(
        self,
        delta: Array,
        p_hat: Array,
        flow_: CartesianFlow,
        fourier_: Fourier,
    ) -> Array:
        r"""`$\sum_\alpha W_\alpha(y, k)$`, the pressure work density.

        .. math::
            W_\alpha = -\sigma_{k_x}\,
            \mathrm{Re}\{\Delta\hat{u}_\alpha^*\,
            (\partial_\alpha \Delta p)^{\widehat{\ }}\} ,

        stored as :func:`~dnsjax.twin.diagnostics.ybudget_terms`'
        ``Wp`` -- **not** the mean-mode driving, although at
        `$(0,0)$` the two coincide (``diagnostics._driving_density``).

        Evaluated **componentwise and summed**, not through the
        equivalent flux form `$-\sigma\,\partial_y
        \mathrm{Re}\{\Delta\hat{p}\Delta\hat{v}^*\}$`: the two agree
        only up to the discrete product-rule error, and this is the
        one that appears in `$\partial_t e$`.  (The flux form is the
        *interpretation* -- it is what makes `$\int W\,dy = 0$` at
        every `$k$`, and comparing them is a check, not a shortcut.
        It follows from continuity alone, so it holds for the
        Bernoulli `$\Delta\hat p$` exactly as it did for the static
        one -- of a larger field, hence a larger absolute residual:
        the ``pi_flux`` column of ``tests/test_twin_budget.py``.)

        Returned as a `$y$`-density divided by ``volume_fac``, like
        every other :func:`dnsjax.twin.diagnostics.twin_ybudget` term.
        """
        grad = (
            1j * fourier_.kx * p_hat,
            apply_y_matrix(flow_.D1, p_hat),
            1j * fourier_.kz * p_hat,
        )
        work = sum((jnp.conj(delta[i]) * grad[i]).real for i in range(3))
        return -work * (fourier_.k_metric / derived_params.volume_fac)

    def neumann_residual(
        self,
        delta: Array,
        p_hat: Array,
        n_y: Array,
        flow_: CartesianFlow,
    ) -> Array:
        r"""The wall residual of the analytic Neumann condition.

        `$(D_1\Delta\hat p - \hat{\mathcal N}_y - Re^{-1}D_2\Delta\hat
        v)|_w$`, the condition the IMM closure does *not* impose, as a
        wall-normal truncation diagnostic: it must shrink
        with ``res.ny``.  Shape ``(2, Nkz, Nkx)`` complex, walls
        ``[bottom, top]``.

        `$\hat{\mathcal N}_y$` is carried unconditionally because
        it is form-dependent.  Evaluating the `$y$`-momentum equation at
        a wall where `$\partial_t\Delta\hat v = \Delta\hat v = 0$`
        gives `$(\partial_y\Delta\hat p)|_w = \hat{\mathcal N}_y|_w +
        Re^{-1}(D_2\Delta\hat v)|_w$`.  Convectively every term of
        `$\mathcal{N}$` carries a velocity factor that no-slip kills,
        so `$\hat{\mathcal N}_y|_w$` is machine-zero and subtracting
        it is a no-op.  Rotationally it is `$(\mathbf{U}_w\cdot
        \partial_y\Delta\mathbf{u}_\parallel)|_w$` -- non-zero
        wherever the wall moves (plane-Couette) -- and is exactly
        `$\partial_y(\mathbf{u}^{(1)}\!\cdot\Delta\mathbf{u})|_w$`,
        the Bernoulli part of `$\Delta\hat p$`, so subtracting it
        leaves the same quantity in both forms.
        """
        d1 = flow_.D1_bnd
        d2v = apply_y_matrix(flow_.D2, delta[1]) / params.phys.re
        return jnp.stack(
            [
                jnp.einsum("j,jzx->zx", d1[0], p_hat) - n_y[0] - d2v[0],
                jnp.einsum("j,jzx->zx", d1[-1], p_hat) - n_y[-1] - d2v[-1],
            ]
        )
