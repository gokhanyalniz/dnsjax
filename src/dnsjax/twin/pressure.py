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
component and it becomes the pressure--strain redistribution, the
*only* source `$\Delta v$` has.

This module recovers `$\Delta\hat{p}$` so that term can be measured
rather than left as a hole.

Interior equation
-----------------
Take the divergence of the difference momentum equation; both members
are solenoidal, so the pressure obeys, per wall-parallel mode,

.. math::
    (D_2 - k^2)\,\Delta\hat{p} = \widehat{\nabla\cdot\mathcal{N}},
    \qquad
    \mathcal{N} = -(\mathbf{u}^{(1)}\cdot\nabla)\Delta\mathbf{u}
    - (\Delta\mathbf{u}\cdot\nabla)\mathbf{u}^{(1)}
    - (\Delta\mathbf{u}\cdot\nabla)\Delta\mathbf{u} ,

on the **interior** rows.  That operator is
:func:`~dnsjax.geometries.wall_bounded.cartesian.build_poisson_operator`.

`$\mathcal{N}$` is written above in the **convective** form, the
default (:func:`dnsjax.twin.diagnostics._convective_sources`).  Under
``twin.rotational_ybudget`` it is instead the rotational term the
solver itself integrates (:mod:`dnsjax.rhs`),
`$\mathbf{u}^{(1)}\times\Delta\boldsymbol{\omega} + \Delta\mathbf{u}
\times\boldsymbol{\omega}^{(1)} + \Delta\mathbf{u}\times\Delta
\boldsymbol{\omega}$`, and then
`$\Delta\hat p$` comes back as the difference of the two members'
**Bernoulli** pressures, `$\Delta p + \mathbf{u}^{(1)}\!\cdot
\Delta\mathbf{u} + |\Delta\mathbf{u}|^2/2$` -- the pressure the
influence matrix actually closes on, rather than one reconstructed
from an operator the solver never applies.  The two differ by a
gradient, so the total work is unchanged; the `$y$`-density is not
(:mod:`dnsjax.twin.diagnostics`, "Two budget forms").  Everything
below is form-independent: nothing in the solve reads
`$\mathcal{N}$` except as a source.

Wall closure: the IMM one, not the textbook one
-----------------------------------------------
Two rows are free.  The obvious choice is the analytic Neumann
condition `$(D_1\Delta\hat{p})|_w = Re^{-1}(D_2\Delta\hat{v})|_w$`
-- the `$y$`-momentum equation evaluated at the wall.  **The
influence-matrix method deliberately declines it**:
:func:`~dnsjax.geometries.wall_bounded._cartesian_primitive_imm._imm_iteration_vp`
states that "the wall BC is determined indirectly by enforcing
continuity `$\nabla\cdot u = 0$` at the walls", because with a
discrete operator the analytic condition and discrete continuity are
not the same constraint.  This module follows the scheme, not the
textbook.

The reconstructed time derivative of the difference field is

.. math::
    \partial_t \Delta\hat{\mathbf{u}}
    = \hat{\mathcal{N}} - \nabla\Delta\hat{p}
      + Re^{-1}(D_2 - k^2)\Delta\hat{\mathbf{u}} ,

divergence-free on the interior rows by the equation above, so the
two free rows are fixed by asking the same of it at the walls,
`$(D_1\,\partial_t\Delta\hat{v})|_{w} = 0$` -- the wall-parallel
components of `$\partial_t\Delta\mathbf{u}$` vanish there because
no-slip holds for all `$t$`, so this is exactly
``_imm_iteration_vp``'s stage-4 residual.  Nothing in that argument
reads `$\mathcal{N}$`, so the closure is the same under either
nonlinear form.  Writing
`$\Delta\hat{p} = p_P + \alpha_1 p_1 + \alpha_2 p_2$` with
`$L_k p_P = \hat f$` (wall rows zeroed) and `$L_k p_i = e_i$` (unit
wall data) makes that residual affine in `$\alpha$`, so

.. math::
    M_{ji} = \bigl(D_1 D_1 p_i\bigr)\big|_{w_j}, \qquad
    M\alpha = \bigl(D_1 (r - p_P)\bigr)\big|_{w},
    \qquad r = \hat{\mathcal{N}}_y
      + Re^{-1}(D_2 - k^2)\Delta\hat{v} .

Same Schur-complement structure as
:func:`~dnsjax.geometries.wall_bounded._cartesian_primitive_imm.derive_homogeneous_data`,
but **cheaper**: there is no velocity being stepped, so no Helmholtz
solves enter -- `$p_1$`, `$p_2$` and `$M^{-1}$` are built once at
construction and only `$p_P$` is solved per sample.

The closure is right under either ``res.consistent_imm``: both
schemes deliver discrete continuity at the walls (the default
delivers it everywhere, at machine epsilon), so requiring the same of
`$\partial_t\Delta\mathbf{u}$` is consistent with the dynamics that
produced the state.  What the analytic Neumann condition then does is
supply an *independent* check:
:meth:`DifferencePressure.neumann_residual` measures how far
`$(D_1\Delta\hat{p} - \hat{\mathcal{N}}_y
- Re^{-1}D_2\Delta\hat{v})|_w$` is from zero, a wall-normal
truncation diagnostic of the same kind as the applied-vs-inferred
driving gap (``tests/test_driving.py``), which must shrink with
``res.ny``.  The `$\hat{\mathcal{N}}_y|_w$` term is machine-zero in
the convective form and genuinely non-zero in the rotational one, so
it is carried unconditionally: that method's docstring has the
derivation.

The mean mode
-------------
`$\Delta\hat{v} \equiv 0$` at `$(k_z, k_x) = (0,0)$` (continuity plus
no-slip), and `$k_x = k_z = 0$` kills the horizontal gradients, so
the *fluctuating* pressure does no work there.  `$k^2 = 0$` is the
one singular system, so ``build_poisson_operator`` swaps that mode's
upper Neumann row for a Dirichlet pin; both homogeneous columns are
then harmonic (`$p_1$` affine in `$y$`, `$p_2$` constant), so
`$D_1D_1p_i = 0$`, `$M$` is identically zero, and ``M_inv`` handles
that by zeroing -- `$\alpha = 0$`.  What comes back is therefore the
mean pressure the interior equation determines, free only in the
constant the pin fixes; not a pure gauge, but doing no work either
way.  What *does* act on the mean mode is the applied driving
`$-\Delta\Pi$` (`$\Pi$` the mean pressure gradient -- the sign
convention is fixed in :mod:`dnsjax.twin.diagnostics`, "Mean-mode
driving"); that density is added by
:func:`dnsjax.twin.diagnostics.twin_ybudget`, which has the mean
profile to hand.

Cost
----
Resident, held for the run -- so it is built only when
``twin.it_ybudget`` is set:

- one extra factored operator, the size of ``flow.Lk_op``
  (`$(N_{k_z}, N_{k_x}, N_y, 2p+1)$` banded factors);
- the two homogeneous columns `$p_1$`, `$p_2$`: real
  `$(N_y, N_{k_z}, N_{k_x})$` fields, so together `$2/(2p+1)$` of the
  factors -- ~12 % on top at ``fd_order = 8``.  They are what makes
  the runtime superposition solve-free, and they are not compressible:
  the columns depend on the mode through `$k^2$`, which every mode has
  its own value of.
- ``M_inv``, `$(N_{k_z}, N_{k_x}, 2, 2)$` -- negligible beside those.

Per sample: one banded solve and a handful of `$D_1$` matvecs, against
the ~21 field transforms the budget itself costs.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array
from jax import numpy as jnp

from ..geometries.wall_bounded._base import apply_y_matrix
from ..geometries.wall_bounded.cartesian import (
    CartesianFlow,
    Fourier,
    build_poisson_operator,
)
from ..parameters import derived_params, params
from ..sharding import register_dataclass_pytree, sharding
from ..solvers import DenseJAXSolver, PerModeBandedPallasOperator


@register_dataclass_pytree
@dataclass(init=False)
class DifferencePressure:
    r"""The difference field's pressure, and the work it does.

    Construction factors the Neumann Poisson operator and derives the
    two homogeneous columns and the `$2\times2$` influence matrix --
    all state-independent, all done once.  Hold one instance for the
    run; see the module docstring for its memory cost.

    A **pytree**, like the flow and solver dataclasses it is built
    from, and for the same reason: every field is a global
    multi-device array, so the jitted diagnostics must take it as an
    *argument*.  A ``static_argnames`` entry would embed the factors
    in the trace as constants -- which works on one process and
    raises ``Closing over jax.Array that spans non-addressable (non
    process local) devices`` on the first multi-process run.
    """

    op: DenseJAXSolver | PerModeBandedPallasOperator
    p1: Array
    p2: Array
    m_inv: Array

    def __init__(self, flow_: CartesianFlow, fourier_: Fourier) -> None:
        self.op = build_poisson_operator(flow_, fourier_)
        zeros = jnp.zeros(
            sharding.spec_shape,
            dtype=sharding.float_type,
            out_sharding=sharding.spec_scalar_shard,
        )
        # `$L_k p_i = e_i$`: unit Neumann data at wall `$i$`, no
        # interior source.  Real operator, real data, real columns.
        self.p1 = self.op.solve(zeros.at[0].set(1.0))
        self.p2 = self.op.solve(zeros.at[-1].set(1.0))

        d1 = flow_.D1_bnd
        dd1 = apply_y_matrix(flow_.D1, self.p1)
        dd2 = apply_y_matrix(flow_.D1, self.p2)
        m00 = jnp.einsum("j,jzx->zx", d1[0], dd1)
        m01 = jnp.einsum("j,jzx->zx", d1[0], dd2)
        m10 = jnp.einsum("j,jzx->zx", d1[-1], dd1)
        m11 = jnp.einsum("j,jzx->zx", d1[-1], dd2)
        # At `$k^2 = 0$` both columns are harmonic, so `$M \equiv 0$`
        # and every `$\alpha$` is admissible (the residual it would
        # correct is identically zero there: `$\Delta\hat{v} = 0$`).
        # Zero ``M_inv`` to pick `$\alpha = 0$`, keeping the regular
        # branch NaN-free before the selection -- the
        # ``derive_homogeneous_data`` idiom.  Padding modes carry
        # nonzero placeholder `$k^2$` and take the regular branch;
        # their values are inert.
        is_mean = fourier_.mean_mask[0]
        det = m00 * m11 - m01 * m10
        safe = jnp.where(is_mean, 1.0, det)
        self.m_inv = jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.where(is_mean, 0.0, m11 / safe),
                        jnp.where(is_mean, 0.0, -m01 / safe),
                    ],
                    axis=-1,
                ),
                jnp.stack(
                    [
                        jnp.where(is_mean, 0.0, -m10 / safe),
                        jnp.where(is_mean, 0.0, m00 / safe),
                    ],
                    axis=-1,
                ),
            ],
            axis=-2,
        )

    def solve(
        self,
        delta: Array,
        div_n: Array,
        n_y: Array,
        flow_: CartesianFlow,
        fourier_: Fourier,
    ) -> Array:
        r"""`$\Delta\hat{p}$` from the difference field's own sources.

        Parameters
        ----------
        delta:
            The spectral difference field, ``(3, Ny, Nkz, Nkx)``.
        div_n:
            `$\widehat{\nabla\cdot\mathcal{N}}$`, ``(Ny, Nkz, Nkx)``.
        n_y:
            `$\hat{\mathcal{N}}_y$`, the wall-normal component of the
            nonlinear term, same shape.  Needed only for the wall
            closure's residual.
        flow\_, fourier\_:
            The geometry singletons.

        Returns
        -------
        :
            `$\Delta\hat{p}$`, ``(Ny, Nkz, Nkx)`` complex, gauge-pinned
            at the mean mode.
        """
        p_part = self.op.solve(div_n.at[0].set(0.0).at[-1].set(0.0))
        v = delta[1]
        # `$r = \hat{\mathcal{N}}_y + Re^{-1}(D_2 - k^2)\Delta\hat v$`
        r = (
            n_y
            + (apply_y_matrix(flow_.D2, v) - fourier_.k2 * v) / params.phys.re
        )
        # `$b_j = (D_1 r)|_{w_j} - (D_1 D_1 p_P)|_{w_j}$`: one `$D_1$`
        # on `$r$` (read at the wall row), two on the pressure, since
        # the pressure enters `$\partial_t\Delta\hat v$` already
        # differentiated.
        g = apply_y_matrix(flow_.D1, p_part)
        d1 = flow_.D1_bnd
        b = jnp.stack(
            [
                jnp.einsum("j,jzx->zx", d1[0], r - g),
                jnp.einsum("j,jzx->zx", d1[-1], r - g),
            ],
            axis=-1,
        )
        alpha = jnp.einsum("zxab,zxb->zxa", self.m_inv, b)
        return p_part + alpha[..., 0] * self.p1 + alpha[..., 1] * self.p2

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
        `$(0,0)$` the two coincide (:func:`dnsjax.twin.diagnostics.
        _driving_density`).

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
        r"""`$(D_1\Delta\hat p - \hat{\mathcal N}_y
        - Re^{-1}D_2\Delta\hat v)|_w$`.

        The analytic Neumann condition the IMM closure does *not*
        impose, as a wall-normal truncation diagnostic: it must shrink
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
