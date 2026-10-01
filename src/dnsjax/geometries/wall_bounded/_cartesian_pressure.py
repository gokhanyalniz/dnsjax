r"""Static pressure of a Cartesian wall-bounded field, on the IMM closure.

The default `$v$`-`$\omega_y$` step (``res.consistent_imm``) never forms
a pressure, and the legacy primitive step forms the
Crank-Nicolson-weighted Bernoulli one.  This module recovers the
instantaneous **static** pressure of a stored field from the field
alone, consistently with the discrete dynamics that produced it.  Two
consumers: the reduced-resolution snapshots of :mod:`dnsjax.lowres`
(the perturbation pressure `$p'$` of one state) and the twin budget's
difference pressure `$\Delta p$` (:mod:`dnsjax.twin.pressure`).  Both
solve the same problem with a different source; nothing below reads
the source except as a source.

Interior equation
-----------------
Take the divergence of the momentum equation; the field is solenoidal,
so per wall-parallel mode

.. math::
    (D_2 - k^2)\,\hat{p} = \widehat{\nabla\cdot\mathcal{N}}

on the **interior** rows, `$\mathcal{N}$` the nonlinear term as it
enters `$\partial_t\hat{\mathbf{u}}$` (the moving-frame term included).
The operator is
:func:`~dnsjax.geometries.wall_bounded.cartesian.build_poisson_operator`.
For a perturbation `$\mathbf{u}'$` about the laminar
`$\mathbf{U}_b(y)$`, :func:`convective_nonlinear` builds the
**convective** form

.. math::
    \mathcal{N} = -(\mathbf{u}\cdot\nabla)\mathbf{u}
    + (\mathbf{U}_b\cdot\nabla)\mathbf{U}_b ,

whose pressure is the static one directly.  The rotational term the
solver integrates (:mod:`dnsjax.rhs`) would instead return the
Bernoulli head `$p' + \mathbf{U}_b\!\cdot\mathbf{u}' + |\mathbf{u}'|^2/2$`;
the two differ by a gradient continuously and by the FD product-rule
error discretely.  The convective form is the one used because its
wall value is exact: every term of it carries a velocity factor that
no-slip kills, so `$\hat{\mathcal{N}}_y|_w = 0$` -- which is what the
mean mode's zero Neumann row below assumes.  The rotational
`$\hat{\mathcal{N}}_y|_w$` is `$(\mathbf{U}_w\cdot\partial_y
\mathbf{u}_\parallel)|_w$`, non-zero at plane Couette's moving walls.

Wall closure: the IMM one, not the textbook one
-----------------------------------------------
Two rows are free.  The obvious choice is the analytic Neumann
condition `$(D_1\hat{p})|_w = \hat{\mathcal{N}}_y|_w +
Re^{-1}(D_2\hat{v})|_w$` -- the `$y$`-momentum equation at the wall.
**The influence-matrix method deliberately declines it**:
:func:`~dnsjax.geometries.wall_bounded._cartesian_primitive_imm._imm_iteration_vp`
"enforces continuity at the walls" instead, because with a discrete
operator the analytic condition and discrete continuity are not the
same constraint.  The default step imposes the discrete condition too
-- `$(D_1\hat v)|_w = 0$`, exactly, at every step -- and the closure
here is that condition's time derivative.  The reconstructed

.. math::
    \partial_t \hat{\mathbf{u}}
    = \hat{\mathcal{N}} - \nabla\hat{p}
      + Re^{-1}(D_2 - k^2)\hat{\mathbf{u}}

is divergence-free on the interior rows by the equation above, so the
two free rows ask the same of it at the walls,
`$(D_1\,\partial_t\hat{v})|_{w} = 0$` (no-slip holds for all `$t$`, so
the wall-parallel components of `$\partial_t\mathbf{u}$` vanish
there).  Writing `$\hat{p} = p_P + \alpha_1 p_1 + \alpha_2 p_2$` with
`$L_k p_P = \hat f$` (wall rows zeroed) and `$L_k p_i = e_i$` (unit
wall data) makes that residual affine in `$\alpha$`:

.. math::
    M_{ji} = \bigl(D_1 D_1 p_i\bigr)\big|_{w_j}, \qquad
    M\alpha = \bigl(D_1 r - D_1 D_1 p_P\bigr)\big|_{w},
    \qquad r = \hat{\mathcal{N}}_y
      + Re^{-1}(D_2 - k^2)\hat{v} .

The Schur-complement structure of
:func:`~dnsjax.geometries.wall_bounded._cartesian_primitive_imm.derive_homogeneous_data`,
but cheaper -- no velocity is stepped, so no Helmholtz solve enters:
`$p_1$`, `$p_2$` and `$M^{-1}$` are built once, and a sample costs
one banded solve.  Right under either ``res.consistent_imm``: both
schemes deliver discrete continuity at the walls (the default delivers
it everywhere).  The analytic condition then supplies an independent
truncation diagnostic (``DifferencePressure.neumann_residual``,
:mod:`dnsjax.twin.pressure`).

The mean mode
-------------
`$\hat v \equiv 0$` at `$(k_z, k_x) = (0,0)$` (continuity plus no-slip)
and `$k^2 = 0$` is the one singular system, so ``build_poisson_operator``
swaps that mode's upper Neumann row for a Dirichlet pin: both
homogeneous columns are harmonic, `$M \equiv 0$`, ``m_inv`` zeroes to
pick `$\alpha = 0$`, and the mean pressure is what the interior
equation and the zero Neumann row at the lower wall determine, in the
gauge `$\hat p_{00} = 0$` at the upper wall.  Continuously that is
`$\bar p'(y) = -\langle v'^2\rangle_{xz}(y)$` for a perturbation (the
mean `$y$`-momentum balance, and `$\langle v'^2\rangle = 0$` at both
walls).  The mean pressure *gradient* is not here: it is the driving,
a uniform body force, which ``stats.dat`` records.

Cost
----
Resident, held for the run: one extra factored operator the size of
``flow.Lk_op`` and the two real `$(N_y, N_{k_z}, N_{k_x})$` columns --
some 12 % on top at ``fd_order = 8``.  Build it only when a consumer
is enabled.  Per sample, :func:`static_pressure` takes 15 single-field
transforms (:func:`convective_nonlinear`) and one banded solve.  Its
peak transient, measured as :mod:`dnsjax.twin.diagnostics` ("Memory")
measures every program: 17 / 15 padded physical components at
``solver.rhs_transform_chunks`` 1 / 3, about half the time step's own
(30 / 28 iterative-CN), so a ``[lowres]`` pressure sample does not
set a run's peak on CPU (a GPU schedules its own).
"""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array, jit, lax
from jax import numpy as jnp

from ...fft import chunked_transform
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding
from ...solvers import DenseJAXSolver, PerModeBandedPallasOperator
from ._base import (
    apply_y_matrix,
    extract_mean_mode,
    phys_to_spec,
    spec_to_phys,
)
from .cartesian import CartesianFlow, Fourier, build_poisson_operator


@register_dataclass_pytree
@dataclass(init=False)
class PoissonPressure:
    r"""The factored Poisson problem and its two homogeneous columns.

    Construction factors the Neumann Poisson operator and derives the
    two homogeneous columns and the `$2\times2$` influence matrix --
    all field-independent, all done once.  Hold one instance for the
    run (its cost: the module docstring).

    A **pytree**, like the flow and solver dataclasses it is built
    from, and for the same reason: every field is a global
    multi-device array, so the jitted callers must take it as an
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
        # correct is identically zero there: `$\hat{v} = 0$`).  Zero
        # ``M_inv`` to pick `$\alpha = 0$`, keeping the regular branch
        # NaN-free before the selection -- the
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
        field: Array,
        div_n: Array,
        n_y: Array,
        flow_: CartesianFlow,
        fourier_: Fourier,
    ) -> Array:
        r"""`$\hat{p}$` from a field's own sources.

        Parameters
        ----------
        field:
            The spectral velocity field, ``(3, Ny, Nkz, Nkx)``: a
            perturbation, or the twin's difference field.  Only its
            wall-normal component is read.
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
            `$\hat{p}$`, ``(Ny, Nkz, Nkx)`` complex, gauge-pinned at
            the mean mode (module docstring).
        """
        p_part = self.op.solve(div_n.at[0].set(0.0).at[-1].set(0.0))
        v = field[1]
        # `$r = \hat{\mathcal{N}}_y + Re^{-1}(D_2 - k^2)\hat v$`
        r = (
            n_y
            + (apply_y_matrix(flow_.D2, v) - fourier_.k2 * v) / params.phys.re
        )
        # `$b_j = (D_1 r)|_{w_j} - (D_1 D_1 p_P)|_{w_j}$`: one `$D_1$`
        # on `$r$` (read at the wall row), two on the pressure, since
        # the pressure enters `$\partial_t\hat v$` already
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


def mean_advect(prof: Array, field: Array, kx: Array, kz: Array) -> Array:
    r"""`$(\mathbf{P}\cdot\nabla)\mathbf{f}$`, `$\mathbf{P}$` a mean profile.

    Diagonal in `$k$`, so FFT-free.  The wall-normal row of any mean
    profile vanishes (continuity plus no-slip at `$k = 0$`; the twin
    module's "State preconditions"), so only the two wall-parallel
    rows of *prof* (``(3, Ny)``) enter.
    """
    return (
        1j
        * (kx * prof[0][:, None, None] + kz * prof[2][:, None, None])
        * field
    )


def convective_nonlinear(
    state: Array, fourier_: Fourier, flow_: CartesianFlow
) -> tuple[Array, Array]:
    r"""A perturbation's convective nonlinear term and its divergence.

    With `$\mathbf{u}' = \bar{\mathbf{u}}'(y) + \mathbf{u}'_f$` split
    into its `$(0, 0)$` mode and fluctuation, and
    `$\mathbf{P} = \mathbf{U}_b + \bar{\mathbf{u}}'$` the whole mean
    profile,

    .. math::
        -\mathcal{N} = (\mathbf{u}'_f\cdot\nabla)\mathbf{u}'
        + \mathrm{i}(k_xP_x + k_zP_z)\,\hat{\mathbf{u}}'
        + \hat v'\,\partial_y\mathbf{U}_b ,

    with `$(\mathbf{P}\cdot\nabla)\mathbf{P} = 0$` for a parallel
    profile.  Only the first term is a product of fluctuations: its
    advector is transformed once (3 transforms), and the gradient of
    each component in turn (3 each) meets it on the padded grid and
    goes back (1 each) -- 15 transforms, with one component's
    gradient live at a time (an optimization barrier per component;
    statement order alone does not hold XLA to it).  The other two
    terms are mode-diagonal.  In a moving frame (``phys.u_grid``) the
    solver's `$+\mathrm{i}k_xU_{grid}\mathbf{u}'$` is added, so
    ``div_n`` and ``n_hat[1]`` match the solver term for term (the
    twin module's "Frame invariance").  This is
    :func:`dnsjax.twin.diagnostics._convective_sources` with a zero
    reference, without the transforms of that zero.

    Returns ``(n_hat, div_n)``: `$\hat{\mathcal{N}}$`
    ``(3, Ny, Nkz, Nkx)`` and the solver's own discrete divergence of
    it (``cartesian._imm_iteration_vp`` stage 1).
    """
    kx, kz = fourier_.kx, fourier_.kz
    d1 = flow_.D1
    base = flow_.base_flow[:, :, 0, 0]
    prof = extract_mean_mode(state).real + base
    dy_base = jnp.einsum("ij,cj->ci", d1, base)

    adv = chunked_transform(spec_to_phys, state * ~fourier_.mean_mask)
    rows: list[Array] = []
    for i in range(3):
        c = state[i]
        grad = chunked_transform(
            spec_to_phys,
            jnp.stack([1j * kx * c, apply_y_matrix(d1, c), 1j * kz * c]),
        )
        product = adv[0] * grad[0] + adv[1] * grad[1] + adv[2] * grad[2]
        rows.append(chunked_transform(phys_to_spec, product[None])[0])
        # One component's gradient live at a time: the three are
        # independent, so without this XLA forms them together.
        rows, adv, state = lax.optimization_barrier((rows, adv, state))

    n_hat = -(
        jnp.stack(rows)
        + mean_advect(prof, state, kx, kz)
        + state[1] * dy_base[:, :, None, None]
    )
    u_grid = derived_params.u_grid
    if u_grid:
        n_hat = n_hat + (1j * u_grid) * kx * state
    div_n = (
        1j * kx * n_hat[0] + apply_y_matrix(d1, n_hat[1]) + 1j * kz * n_hat[2]
    )
    return n_hat, div_n


@jit
def static_pressure(
    state: Array,
    pressure: PoissonPressure,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> Array:
    r"""The static pressure perturbation `$\hat{p}'$` of a state.

    *state* is the physical spectral perturbation the solver stores
    (``(3, Ny, Nkz, Nkx)``); returns ``(Ny, Nkz, Nkx)`` in the same
    layout.  Every array argument is a global array and reaches the
    program as an argument (``.claude/rules/jax.md``).
    """
    n_hat, div_n = convective_nonlinear(state, fourier_, flow_)
    return pressure.solve(state, div_n, n_hat[1], flow_, fourier_)
