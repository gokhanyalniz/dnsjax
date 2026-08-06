r"""Viscoelastic (sPTT) extension of the cylindrical (pipe) geometry.

Adds a symmetric conformation tensor `$\mathbf{c}$` to the cylindrical
geometry, coupling it to the velocity via the polymer-stress
divergence.  The time-integrated state grows from the 3 velocity
components to **9**: 3 velocity + 6 independent symmetric-tensor
components, carried as a single stacked spectral array
``(9, Nr, Nm, Nkz)`` in the solver's spin basis

.. math::
    [\,u_z,\; u_+,\; u_-,\;
      c_{zz},\; c_{z+},\; c_{z-},\; c_{+-},\; c_{++},\; c_{--}\,],

with `$u_\pm = u_r \pm i u_\theta$` the decoupled velocity of
:mod:`~dnsjax.geometries.wall_bounded.cylindrical` and the analogous
tensor spin projections.  Everything outside the time stepper --
snapshots, diagnostics, probes, initial conditions, the analysis
package -- uses the physical layout
`$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
c_{\theta\theta}, c_{r\theta})$`, and a given state crosses at most
once.  Both layouts, the conversion pair, the spin weights and the
pointwise physical-space arithmetic live in
:mod:`._viscoelastic_common`, shared with the annular sPTT geometry.

What lives where
----------------
The time stepper itself -- the fused pseudo-spectral RHS, the FFT-free
CN/AB2 coupling, the conformation Crank-Nicolson update, the
predictor / corrector / norm, the `$H_c$` builders and the stepper
factory -- is shared with the annular sPTT geometry and lives in
:mod:`._viscoelastic_stepping`.  **This** module owns the pipe half:
the flow dataclass, the analytical laminar profiles, the per-slot axis
parity, and the adapter methods the shared stepper dispatches on
(parity-reduced radial derivatives, a single wall row, the axial body
force -- see :class:`ViscoelasticCylindricalFlow`).

Governing equations (sPTT)
--------------------------
.. math::
    \partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u}
      &= -\nabla p + \tfrac{\beta}{\mathrm{Re}}\nabla^2\mathbf{u}
      + \tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}\nabla\cdot\mathbf{c}
      + \boldsymbol{\Pi}, \\
    \partial_t \mathbf{c} + \mathbf{u}\cdot\nabla\mathbf{c}
      - (\nabla\mathbf{u})^{\!\top}\!\cdot\mathbf{c}
      - \mathbf{c}\cdot\nabla\mathbf{u}
      &= \kappa\nabla^2\mathbf{c}
      - \tfrac{\mathbf{c}-\mathbb{I}}{\mathrm{Wi}}
        (1 - 3\epsilon + \epsilon\,\mathrm{tr}\,\mathbf{c}),

with no-slip `$\mathbf{u}=0$` and `$\nabla^2\mathbf{c}=0$` at the
single wall `$r = 1$`, regularity at the axis carried by parity (see
below), and the **axial** body force `$\Pi_z = 4/\mathrm{Re}$` (a
uniform mean pressure gradient; see
:mod:`~dnsjax.flows.wall_bounded.viscoelastic_pipe`).  All products are
at most quadratic (`$\mathrm{tr}(\mathbf{c})\,\mathbf{c}$`), so the
existing 3/2-rule dealiasing is exact.

Axis parity of the conformation tensor
--------------------------------------
The pipe has no `$r = 0$` grid point and no axis boundary row:
regularity is imposed by the parity-reduced FD matrices, whose stencils
span across the axis on the mirrored auxiliary grid (the
``cylindrical.py`` module docstring).  Each field needs its parity
class there.

A spin-`$s$` quantity picks up `$e^{im\pi} = (-1)^m$` from the Fourier
mode and `$(-1)^s$` from the reversal of `$\hat e_r, \hat e_\theta$`
across the origin, so

.. math::
    \text{parity} = (-1)^{m+s}, \qquad
    m_{\mathrm{eff}} = m + s,

one rule for the whole state.  Equivalently, in the physical basis a
component's class is set by how many of its indices are in
`$\{r, \theta\}$` -- each such index flips sign:

=========================================  ==========  ================
component                                  spin `$s$`  parity
=========================================  ==========  ================
`$u_z$`                                    0           `$(-1)^m$`
`$u_\pm$` (`$u_r, u_\theta$`)              `$\pm1$`    `$(-1)^{m+1}$`
`$c_{zz}$`, `$c_{+-}$`                     0           `$(-1)^m$`
`$c_{z\pm}$` (`$c_{rz}, c_{\theta z}$`)    `$\pm1$`    `$(-1)^{m+1}$`
`$c_{\pm\pm}$` (`$c_{rr}, c_{\theta\theta},
c_{r\theta}$`)                             `$\pm2$`    `$(-1)^m$`
=========================================  ==========  ================

So the six tensor components fall into the **two existing** parity
classes -- no new operator family, only a per-slot selection.  Parity
depends on `$s \bmod 2$` while `$m_{\mathrm{eff}}$` depends on `$s$`
itself, exactly as for `$u_\pm$` (same parity, different operator).
The class is set by the **physical** `$m = m_0 j$`, so the azimuthal
wedge folds in unchanged (``Fourier.m_is_even``).

Like the velocity, the conformation is held to the parity condition
only -- smoothness across the axis -- and not to the stronger
spin-weighted `$r^{|m+s|}$` vanishing rate, which the parity-reduced
discretisation does not represent for either field.

Every radial derivative therefore carries a per-slot sign, built by
:func:`_parity_signs` from the spin weights (the flow's
``rhs_radial_derivatives`` / ``div_c_radial_derivatives`` /
``tensor_abase_matvec`` adapters below).  Those GEMMs stack their
inputs **y-leading** (``component_axis=1``), which is both the
transpose-free layout and the one whose ghost scatter-add lands on the
radial axis (:func:`~.cylindrical._parity_y_matvec`).

Time integration
----------------
Both ``iterative-cn`` (default) and ``cnab2`` schemes are supported.
``get_rhs`` returns the full 9-component nonlinear RHS -- velocity
(`$\mathbf{u}\times\boldsymbol\omega$` + `$\boldsymbol\Pi$` + FFT-free
polymer divergence
`$\tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}\nabla\cdot\mathbf{c}$`) and
conformation (`$-\mathbf{u}\cdot\nabla\mathbf{c} + (\nabla u)^{\!\top}c
+ c\nabla u - \tfrac{f(\mathrm{tr}\,c)}{\mathrm{Wi}}(c-\mathbb{I})$`) --
built from a single fused pseudo-spectral evaluation (one batched
inverse transform of ~36 fields, one batched forward transform of the
9 outputs; the vorticity is free from the velocity-gradient tensor).
The predictor/corrector then solves the velocity via the cylindrical
`$1\times1$` IMM
(:func:`~dnsjax.geometries.wall_bounded.cylindrical._imm_iteration`,
solvent viscosity `$\nu = \beta/\mathrm{Re}$`) and the conformation via
a Crank-Nicolson Helmholtz solve per spin component
(`$H_c = \tfrac1{\Delta t}I - \theta\kappa\nabla^2$`, one Laplacian BC
wall row).  The polymer divergence is added to the velocity **sources**,
so both IMM schemes project it with no change of their own.  With
`$\kappa = 0$` the transport is purely hyperbolic (no wall BC) and the
conformation update degenerates to the explicit CN combination.

``res.consistent_imm`` is supported and needs nothing from this module:
the reconstruction scheme consumes the polymer divergence as one more
source.

The ``cnab2`` scheme (one FFT/step) makes the FFT-free linear/mean
coupling implicit via ``_viscoelastic_stepping._l_bf`` -- velocity
mean-flow coupling + polymer-stress divergence, conformation mean
advection / mean-shear stretching + linear relaxation (all gated /
structured so the explicit AB2 remainder is the pure
fluctuation-fluctuation nonlinearity plus the nonlinear relaxation) --
and advances that remainder explicitly.  It reproduces ``iterative-cn``
to O(`$\Delta t^2$`) at ~1 FFT/step versus ~4 (the coupled tensor system
inherits the wall-bounded velocity's reduced projection-splitting
order, shared by both schemes).
"""

from dataclasses import dataclass, field
from typing import ClassVar

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from ...parameters import params
from ...sharding import register_dataclass_pytree, sharding
from ...solvers import (
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _banded_from_dense,
)
from ._base import (
    integrate_scalar,  # noqa: F401 -- for the flow module
)
from ._viscoelastic_common import (
    C_FROB_SQRT_SPIN,  # noqa: F401 -- re-exported
    N_VE_COMPONENTS,  # noqa: F401 -- re-exported
    PHYS_COMBO_SPIN,
    TENSOR_SPIN,
    from_spin_basis,
    get_norm2_conformation,  # noqa: F401 -- re-exported for the flow
    narrow_abase_wall_row,
    solve_ptt_f,
    spin_to_phys_combos,  # noqa: F401 -- re-exported (test_cnab2)
    to_spin_basis,
)

# The shared sPTT stepping surface lives in
# ``._viscoelastic_stepping``; the names re-exported here are the
# ones consumed *through this module* -- by the flow module
# (``_div_c``) and by the tests that reach it by string
# (``test_cnab2.py`` and the per-geometry viscoelastic tests).
from ._viscoelastic_stepping import (
    _build_Hc_band_gpu,  # noqa: F401
    _build_Hc_dense_gpu,  # noqa: F401
    _build_hc_operator,
    _div_c,  # noqa: F401
    _get_rhs,  # noqa: F401
    _l_bf,  # noqa: F401
    _tensor_laplacian_spin,  # noqa: F401
)
from ._viscoelastic_stepping import (
    build_viscoelastic_stepper as _build_stepper,
)
from .cylindrical import (
    CFL_NAMES,
    CylindricalFlow,
    Fourier,
    _imm_iteration,
    _parity_y_matvec,
    build_parity_reduced_matrices,
    fourier,
)
from .cylindrical import (
    _build_dt_leaves as _cyl_dt_leaves,
)
from .cylindrical import (
    _l_bf as _cyl_l_bf,
)

#: Role aliases for the basis boundary (see ``cylindrical.py``).
to_solver_basis = to_spin_basis
from_solver_basis = from_spin_basis

# Spin weights of the fused radial-derivative batch of the shared
# ``_get_rhs_core`` -- the velocity triad (u_r, u_theta, u_z) followed
# by the physical tensor combos -- and of the three columns ``_div_c``
# differentiates (c_rr, c_r_theta, c_rz).  Only ``s % 2`` matters (the
# parity class); see the module docstring.
_DR_BATCH_SPIN = np.concatenate([[1, 1, 0], PHYS_COMBO_SPIN])
_DIV_C_SPIN = np.array([2, 2, 1])


# ── Per-slot axis parity ────────────────────────────────────────────


def _parity_signs(spins: np.ndarray, fourier_: Fourier) -> Array:
    r"""Per-slot ghost sign `$(-1)^{m+s}$`, shaped ``(1, C, Nm, 1)``.

    Broadcasts against a y-leading ``(g, C, Nm, Nkz)`` ghost product, so
    one :func:`~.cylindrical._parity_y_matvec` call can carry a whole
    stack of differently-signed components.  Only the parity of *spins*
    is read: `$s$` even selects the `$(-1)^m$` class, `$s$` odd the
    `$(-1)^{m+1}$` one.
    """
    psp = fourier_.m_is_even * 2 - 1  # (-1)^m
    psv = -psp  # (-1)^{m+1}
    return jnp.stack([psp if s % 2 == 0 else psv for s in spins], axis=1)


def _mean_parity_signs(spins: np.ndarray) -> np.ndarray:
    r"""The same signs at the mean mode `$m = 0$`, as plain scalars.

    A mean profile carries `$m = 0$`, so `$(-1)^{m+s} = (-1)^s$` is a
    constant and the ghost correction needs no mode-dependent mask.
    """
    return np.where(spins % 2 == 0, 1.0, -1.0)


# ── Analytical laminar profiles (JAX-free, build-time) ──────────────


def viscoelastic_laminar_profiles(
    rs: np.ndarray, D1_even: np.ndarray, wi: float, eps: float
) -> np.ndarray:
    r"""9-component laminar `$r$`-profiles for the axially driven sPTT
    pipe (complex ``(9, Nr)``), in the **physical** state layout
    `$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` -- these feed initial conditions
    and the flow's laminar reference, both of which live outside the
    solver (:func:`to_spin_basis` converts when one enters it).

    Velocity: the Hagen-Poiseuille profile `$W(r) = 1 - r^2$`, which is
    the exact balance of `$\Pi_z = 4/\mathrm{Re}$` against the *total*
    (solvent + polymer) stress at `$\epsilon = 0$`; the shear-thinning
    correction at `$\epsilon > 0$` is neglected, as in the annular twin.

    Conformation: the pointwise sPTT equilibrium on the **discrete**
    local shear `$S = D_1 W$` (no curvature term -- the flow is
    unidirectional and axial, so `$\bar L_{rz}$` is the only nonzero
    velocity gradient), which makes the conformation slice of the RHS
    vanish identically at every `$\epsilon$`:

    .. math::
        c_{rr} = c_{\theta\theta} = 1, \quad
        c_{rz} = \frac{\mathrm{Wi}\,S}{f}, \quad
        c_{zz} = 1 + \frac{2(\mathrm{Wi}\,S)^2}{f^2}, \quad
        f^3 - f^2 = 2\epsilon(\mathrm{Wi}\,S)^2,

    with `$c_{r\theta} = c_{\theta z} = 0$`.  The sheared pair is
    `$(c_{rz}, c_{zz})$` here, against the annulus's
    `$(c_{r\theta}, c_{\theta\theta})$`.

    *D1_even* is the even-parity radial FD matrix: `$W$` is even about
    the axis, so that is the operator the solver itself would use, and
    taking the shear discretely is what makes the equilibrium exact on
    the grid rather than only in the continuum.  Pure (NumPy,
    build-time); shared by the flow's laminar state and the
    viscoelastic random / rolls ICs.
    """
    rs_np = np.asarray(rs)
    u_z = 1.0 - rs_np**2
    shear = np.asarray(D1_even) @ u_z
    wis = wi * shear
    f = solve_ptt_f(2.0 * eps * wis**2)
    x = wis / f  # c_rz
    c_zz = 1.0 + 2.0 * x**2
    zeros = np.zeros_like(rs_np, dtype=np.complex128)
    ones = np.ones_like(rs_np, dtype=np.complex128)
    return np.stack(
        [
            u_z,  # u_z
            zeros,  # u_r
            zeros,  # u_theta
            c_zz,  # c_zz
            x,  # c_rz
            zeros,  # c_theta_z
            ones,  # c_rr
            ones,  # c_theta_theta
            zeros,  # c_r_theta
        ]
    ).astype(np.complex128)


def parity_d1_even(rs: np.ndarray, fd_order: int) -> np.ndarray:
    r"""The assembled even-parity radial `$D_1$` on *rs* (host NumPy).

    The flow stores `$D_1$` split into its parity-independent part and
    the near-axis ghost correction -- the right form for a matvec, but
    not for the two build-time consumers that need the matrix itself:
    the laminar profiles' discrete shear, and the narrow Laplacian BC
    wall row (whose one-sided stencil never reaches the axis, so the
    even matrix serves both parity bands there).  A small,
    once-per-run NumPy build.
    """
    return build_parity_reduced_matrices(np.asarray(rs), fd_order)[0]


# ── Viscoelastic cylindrical flow dataclass ─────────────────────────

_WallBoundedOp = DenseJAXSolver | PerModeBandedPallasOperator


@register_dataclass_pytree
@dataclass
class ViscoelasticCylindricalFlow(CylindricalFlow):
    r"""Precomputed data for viscoelastic (sPTT) pipe flow.

    Extends
    :class:`~dnsjax.geometries.wall_bounded.cylindrical.CylindricalFlow`
    (radial grid, parity-reduced FD matrices, `$1\times1$` IMM operators
    -- built with solvent viscosity `$\nu = \beta/\mathrm{Re}$` via
    ``derived_params.nu``) with the conformation-tensor machinery: the
    stacked Crank-Nicolson Helmholtz operator ``Hc_op`` (6 spin
    components, `$m_{\mathrm{eff}} = m + s$` on the `$(-1)^{m+s}$`
    parity band; two share `$m_{\mathrm{eff}} = m$`) and the physical
    `$1/r$` profile on the padded grid.  When `$\kappa = 0$` no
    Helmholtz operator is built (``Hc_op = None``): the conformation
    transport is hyperbolic and the update is the explicit CN
    combination.

    The methods below are the **pipe half of the adapter surface** the
    shared stepper (:mod:`._viscoelastic_stepping`) dispatches on,
    resolved once at trace time; being methods rather than fields they
    add no pytree leaf.  In short: parity-reduced radial derivatives
    carrying the per-slot `$(-1)^{m+s}$` sign, a single
    `$\nabla^2 c = 0$` wall row (the axis is closed by the parity
    reduction), and an axial mean-mode body force.

    ``pi_z`` is that body force, zero here and set by the flow subclass
    (:class:`~dnsjax.flows.wall_bounded.viscoelastic_pipe`), which also
    zeros the base flow (total-field integration).
    """

    #: CFL column labels (a ``ClassVar``: as an annotated field this
    #: tuple would become two static pytree entries per flatten).
    cfl_names: ClassVar[tuple[str, str, str]] = CFL_NAMES

    tensor_spin: Array = field(init=False)
    inv_r_padded: Array = field(init=False)
    pi_z: Array = field(init=False)
    Hc_op: _WallBoundedOp | None = field(init=False, default=None)
    # Narrow Laplacian BC wall row of Hc, stored as a leaf so the
    # jitted adaptive-dt rebuild (``_build_dt_leaves``) can reuse it
    # (its NumPy build cannot run on tracers).  ``None`` (aux) while
    # kappa == 0, where no Hc exists.
    hc_narrowN: Array | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Build velocity grid / FD matrices / 1x1 IMM (nu = beta/re).
        super().__post_init__()

        self.tensor_spin = jax.device_put(
            jnp.asarray(TENSOR_SPIN, dtype=sharding.float_type),
            sharding.no_shard,
        )

        # Mean-mode axial body force; the flow subclass overwrites it.
        self.pi_z = jnp.zeros(
            params.res.ny,
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )

        Nr = params.res.ny
        ny_phys = Nr + sharding.ny_y_pad
        inv_r_pad = np.zeros(ny_phys, dtype=sharding.float_type)
        inv_r_pad[:Nr] = np.asarray(self.inv_r)
        self.inv_r_padded = jax.device_put(
            inv_r_pad.reshape(ny_phys, 1, 1), sharding.no_shard
        )

        if params.phys.kappa == 0:
            # Hyperbolic conformation transport: no diffusion, no wall BC.
            self.Hc_op = None
            return

        self._build_conformation_operator()

    def _build_conformation_operator(self) -> None:
        r"""Build the stacked 6-component `$H_c$` Crank-Nicolson operator.

        Stores the narrow Laplacian BC wall row as the ``hc_narrowN``
        leaf (its JAX-free NumPy build cannot run on tracers, so the
        jitted adaptive-``dt`` rebuild reuses it), optionally
        pre-checks the no-pivot LU at ``dt_max`` (``step.adaptive``;
        the velocity `$H_k$` analogue), and delegates the
        assembly/factorization to
        :func:`._viscoelastic_stepping._build_hc_operator`.

        The wall row is one-sided at `$r = 1$`, so its `$D_2$` stencil
        never reaches the axis and the row is parity-independent -- it
        is built from the even-parity `$D_1$` and used for both bands.
        """
        rs_np = np.asarray(self.rs)
        D1_even = parity_d1_even(rs_np, params.res.fd_order)
        rowN_np = narrow_abase_wall_row(
            rs_np, D1_even, params.res.fd_order, inner=False
        )
        self.hc_narrowN = jax.device_put(rowN_np, sharding.no_shard)

        if params.step.adaptive and params.solver.backend == "pallas":
            # Verify the no-pivot LU where the Helmholtz diagonal is
            # least dominant; adaptive rebuilds at dt <= dt_max then
            # skip the check (solvers._factor_pallas_operator).
            _build_hc_operator(
                params.step.dt_max, fourier, self, label="Hc(dt_max)"
            )
        self.Hc_op = _build_hc_operator(self.dt, fourier, self, label="Hc")

    # ── Adapter surface (see ``_viscoelastic_stepping``) ────────────

    def rhs_radial_derivatives(
        self,
        fields: tuple[Array, ...],
        combos: Array,
        fourier_: Fourier,
    ) -> Array:
        r"""The fused 9-field radial-derivative batch of the RHS.

        One parity-reduced `$D_1$` GEMM **pair** over the 3 velocity
        components and the 6 conformation combos -- one pair instead of
        two -- stacked y-leading ``(Nr, 9, Nm, Nkz)``, which is both
        transpose-free and the layout whose ghost scatter-add lands on
        the radial axis.  *fields* is the flat 9-tuple the stack wants;
        *combos* (the same six tensor entries already materialised as
        one array) is the annulus's preferred form and unused here.
        Returns ``(9, Nr, Nm, Nkz)``.
        """
        dr_y = _parity_y_matvec(
            self.D1_pos,
            self.D1_ghost,
            jnp.stack(fields, axis=1),
            _parity_signs(_DR_BATCH_SPIN, fourier_),
            component_axis=1,
        )
        return jnp.swapaxes(dr_y, 0, 1)

    def div_c_radial_derivatives(
        self, c_rr: Array, c_rth: Array, c_rz: Array, fourier_: Fourier
    ) -> Array:
        r"""`$(\partial_r c_{rr}, \partial_r c_{r\theta},
        \partial_r c_{rz})$`, one batched parity-reduced `$D_1$` GEMM.

        `$c_{rr}$` and `$c_{r\theta}$` are in the `$(-1)^m$` class,
        `$c_{rz}$` in the `$(-1)^{m+1}$` one.  Returns
        ``(3, Nr, Nm, Nkz)``.
        """
        dr_y = _parity_y_matvec(
            self.D1_pos,
            self.D1_ghost,
            jnp.stack([c_rr, c_rth, c_rz], axis=1),
            _parity_signs(_DIV_C_SPIN, fourier_),
            component_axis=1,
        )
        return jnp.swapaxes(dr_y, 0, 1)

    def tensor_abase_matvec(self, c_spin: Array, fourier_: Fourier) -> Array:
        r"""`$A_{\mathrm{base}}^{(\sigma)} c
        = (\partial_r^2 + \tfrac1r\partial_r)c$` on the 6 spin slots,
        each on its own `$(-1)^{m+s}$` parity band.  ``(6, Nr, Nm,
        Nkz)``."""
        inv_r_y = self.inv_r[:, None, None, None]  # against (Nr,6,Nm,Nkz)
        par = _parity_signs(TENSOR_SPIN, fourier_)
        # y-leading (Nr, 6, Nm, Nkz): transpose-free GEMM, and the ghost
        # scatter lands on the radial axis.
        c_y = jnp.swapaxes(c_spin, 0, 1)
        D2_c = _parity_y_matvec(
            self.D2_pos, self.D2_ghost, c_y, par, component_axis=1
        )
        D1_c = _parity_y_matvec(
            self.D1_pos, self.D1_ghost, c_y, par, component_axis=1
        )
        return jnp.swapaxes(D2_c + inv_r_y * D1_c, 0, 1)

    def mean_profile_dr(self, prof: Array, spin: int) -> Array:
        r"""`$\partial_r$` of one `$m = 0$` profile, ``(Nr,)``.

        The parity-reduced `$D_1$` on a bare `$(N_r,)$` mean profile is
        a direct matmul plus the near-axis ghost rows, at the `$m = 0$`
        constant sign `$(-1)^s$` of the profile's *spin* (`$+1$` for
        the even `$\bar u_z$`, `$-1$` for the odd `$\bar u_\theta$`) --
        no mode-dependent mask needed (the same shortcut
        ``pipe.frozen_profile_flow`` takes).
        """
        g = self.D1_ghost.shape[0]
        sign = _mean_parity_signs(np.array([spin]))[0]
        return (self.D1_pos @ prof).at[:g].add(sign * (self.D1_ghost @ prof))

    def add_mean_body_force(
        self, nl_z: Array, nl_r: Array, nl_th: Array, fourier_: Fourier
    ) -> tuple[Array, Array, Array]:
        """Add the axial body force ``pi_z`` at the mean mode."""
        return (
            nl_z
            + jnp.where(fourier_.mean_mask, self.pi_z[:, None, None], 0.0),
            nl_r,
            nl_th,
        )

    def zero_hc_wall_rows(self, R: Array) -> Array:
        r"""Zero the `$H_c$` RHS at the single `$\nabla^2 c = 0$` wall
        row (`$r = 1$`); the axis carries no row."""
        return R.at[:, -1].set(0.0)

    def hc_wall_rows(self) -> tuple[tuple[int, Array], ...]:
        r"""``((row index, narrow BC row),)`` -- the outer wall only.

        The index is a non-negative host int: ``_banded_wall_row``'s
        static column arithmetic needs that form.
        """
        return ((self.hc_narrowN.shape[0] - 1, self.hc_narrowN),)

    def hc_spin_bases(
        self,
        fourier_: Fourier,
        spins: tuple[int, ...],
        *,
        banded: bool,
        p: int,
    ) -> list[Array]:
        r"""The per-spin `$H_c$` base operator, aligned with *spins*.

        The parity band follows `$s \bmod 2$` (module docstring): the
        odd-`$s$` slots ride the `$(-1)^{m+1}$` band, the rest the
        `$(-1)^m$` one, selected per mode by ``jnp.where`` on
        ``m_is_even``.  Broadcast to the operator's mode layout, as
        :func:`~dnsjax.solvers._assemble_banded_operator` requires of
        its caller.
        """
        even_c = fourier_.m_is_even[0, ..., None]  # (Nm, 1, 1)
        odd_c = 1.0 - even_c
        if banded:
            band_even = _banded_from_dense(self.A_base_even, p)
            band_odd = _banded_from_dense(self.A_base_odd, p)
            return [
                jnp.where(
                    odd_c if s % 2 else even_c, band_even[None], band_odd[None]
                )[:, None]
                for s in spins
            ]
        return [
            jnp.where(
                (odd_c if s % 2 else even_c)[..., None],
                self.A_base_even,
                self.A_base_odd,
            )
            for s in spins
        ]

    def imm_iteration(
        self,
        u_prev: Array,
        u_pred: Array,
        rhs_prev: Array,
        rhs_next: Array,
        fourier_: Fourier,
    ) -> tuple[Array, Array]:
        r"""The cylindrical `$1\times1$` influence-matrix velocity pass."""
        return _imm_iteration(
            u_prev, u_pred, rhs_prev, rhs_next, fourier_, self
        )

    def velocity_l_bf(self, vel: Array, fourier_: Fourier) -> Array:
        """The cylindrical FFT-free base/mean-flow velocity coupling."""
        return _cyl_l_bf(vel, fourier_, self)

    def base_dt_leaves(
        self, dt: Array, fourier_: Fourier
    ) -> dict[str, object]:
        """The cylindrical velocity ``dt``-dependent leaves."""
        return _cyl_dt_leaves(dt, fourier_, self)


# ── Stepper factory ─────────────────────────────────────────────────


def build_viscoelastic_stepper(flow: ViscoelasticCylindricalFlow):
    """Build time-stepping functions for a viscoelastic pipe flow.

    Binds this geometry's ``fourier`` singleton to the shared
    :func:`._viscoelastic_stepping.build_viscoelastic_stepper`, which
    documents the returned 7-tuple.
    """
    return _build_stepper(flow, fourier)
