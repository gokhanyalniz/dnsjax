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

Every radial derivative below therefore carries a per-slot sign, built
by :func:`_parity_signs` from the spin weights.  Those GEMMs stack
their inputs **y-leading** (``component_axis=1``), which is both the
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
source.  A flag-on blow-up found while porting this flow was
misattributed to that polymer source; it was a defect in the
cylindrical pass's lagged wall data, reproducing at `$\beta = 1$` and
in the Newtonian pipe, and is fixed there
(``cylindrical._imm_iteration_vw`` carries the measurements; the
flow-level summary is in
:mod:`~dnsjax.flows.wall_bounded.viscoelastic_pipe`).

The ``cnab2`` scheme (one FFT/step) makes the FFT-free linear/mean
coupling implicit via :func:`_l_bf` -- velocity mean-flow coupling +
polymer-stress divergence, conformation mean advection / mean-shear
stretching + linear relaxation (all gated / structured so the explicit
AB2 remainder is the pure fluctuation-fluctuation nonlinearity plus the
nonlinear relaxation) -- and advances that remainder explicitly.  It
reproduces ``iterative-cn`` to O(`$\Delta t^2$`) at ~1 FFT/step versus
~4 (the coupled tensor system inherits the wall-bounded velocity's
reduced projection-splitting order, shared by both schemes).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from ...fft import chunked_transform
from ...measurements import get_cfl
from ...operators import phys_to_spec_2d, spec_to_phys_2d
from ...parameters import derived_params, params
from ...sharding import register_dataclass_pytree, sharding
from ...solvers import (
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _assemble_banded_operator,
    _banded_diag_column,
    _banded_from_dense,
    _banded_wall_row,
    _build_pallas_operator,
    _factor_pallas_operator,
)
from ._base import (
    extract_mean_mode,
    integrate_scalar,  # noqa: F401 -- for the flow module
)
from ._viscoelastic_common import (
    C_FROB_SQRT_SPIN,  # noqa: F401 -- re-exported
    N_VE_COMPONENTS,  # noqa: F401 -- re-exported
    PHYS_COMBO_SPIN,
    TENSOR_SPIN,
    combined_norm,
    conformation_coupling_core,
    div_c_assemble,
    from_spin_basis,
    get_norm2_conformation,  # noqa: F401 -- re-exported for the flow
    narrow_abase_wall_row,
    phys_combos_to_spin,
    pointwise_rhs,
    solve_ptt_f,
    spin_to_phys_combos,
    to_spin_basis,
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

# Spin weights of the fused radial-derivative batch of ``_get_rhs_core``
# -- the velocity triad (u_r, u_theta, u_z) followed by the physical
# tensor combos -- and of the three columns ``_div_c`` differentiates
# (c_rr, c_r_theta, c_rz).  Only ``s % 2`` matters (the parity class);
# see the module docstring.
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


# ── H_c Helmholtz operator builders (per spin component) ────────────


def _build_Hc_dense_gpu(
    A_base_even: Array,
    A_base_odd: Array,
    narrowN: Array,
    m_is_even_c: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    kappa: float,
) -> Array:
    r"""Dense `$H_c = \tfrac1{\Delta t} I - c\kappa\nabla^2$` for one spin
    component (dense backend).  Interior rows carry the diagonal
    Helmholtz shift on the parity-selected base operator; the single
    wall row (`$r = 1$`) is the narrow Laplacian BC row
    `$A_{\mathrm{base}} - (m_{\mathrm{eff}}^2/r^2 + k_z^2) I$`.  The
    axis needs no row -- the parity reduction closes it."""
    Nr = A_base_even.shape[0]
    dtype = A_base_even.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)
    diag_coeff = 1.0 / dt + c * kappa * (meff2 * inv_r2 + kz2)  # (Nm,Nkz,Nr)
    A_base = jnp.where(m_is_even_c[..., None], A_base_even, A_base_odd)
    Hc = diag_coeff[..., None] * eye_Nr - c * kappa * A_base
    # Wall row: narrow Laplacian BC (mode-dependent diagonal shift).
    shiftN = meff2 * inv_r2[-1] + kz2  # (Nm, Nkz, 1)
    Hc = Hc.at[..., -1, :].set(narrowN[None, None] - shiftN * eye_Nr[-1])
    return Hc


def _build_Hc_band_gpu(
    band_even: Array,
    band_odd: Array,
    narrowN: Array,
    m_is_even_c: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    kappa: float,
    p: int,
) -> Array:
    r"""Banded `$H_c$` for one spin component (Pallas backend), layout
    ``(Nm, Nkz, Nr, 2p+1)``; one narrow Laplacian BC wall row."""
    Nr = band_even.shape[0]
    band_base = jnp.where(m_is_even_c, band_even[None], band_odd[None])
    diag = 1.0 / dt + c * kappa * (meff2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    e = _banded_diag_column(p, band_base.dtype)
    # Narrow BC band (mode-constant) minus the mode-dependent shift.
    bandN = _banded_wall_row(narrowN, Nr - 1, p)  # (2p+1,)
    shiftN = meff2 * inv_r2[-1] + kz2  # (Nm, Nkz, 1)
    return _assemble_banded_operator(
        band_base[:, None], -c * kappa, diag, [(Nr - 1, bandN - shiftN * e)]
    )


# ── Spectral tensor operators (FFT-free) ────────────────────────────


def _tensor_laplacian_spin(
    c_spin: Array, fourier_: Fourier, flow_: ViscoelasticCylindricalFlow
) -> Array:
    r"""Spin-diagonal tensor Laplacian, `$(6, N_r, N_m, N_{kz})$`.

    `$(\nabla^2 c)_{\text{spin }s} = A_{\mathrm{base}}^{(\sigma)} c
    - (m_{\mathrm{eff}}^2/r^2 + k_z^2) c$` with
    `$m_{\mathrm{eff}} = m + s$` and the parity-reduced
    `$A_{\mathrm{base}}$` selected per spin slot (module docstring).
    """
    inv_r_y = flow_.inv_r[:, None, None, None]  # against (Nr, 6, Nm, Nkz)
    par = _parity_signs(TENSOR_SPIN, fourier_)
    # y-leading (Nr, 6, Nm, Nkz): transpose-free GEMM, and the ghost
    # scatter lands on the radial axis.
    c_y = jnp.swapaxes(c_spin, 0, 1)
    D2_c = _parity_y_matvec(
        flow_.D2_pos, flow_.D2_ghost, c_y, par, component_axis=1
    )
    D1_c = _parity_y_matvec(
        flow_.D1_pos, flow_.D1_ghost, c_y, par, component_axis=1
    )
    Abase_c = jnp.swapaxes(D2_c + inv_r_y * D1_c, 0, 1)

    m = fourier_.m  # (1, Nm, 1)
    meff = m + flow_.tensor_spin[:, None, None, None]  # (6, 1, Nm, 1)
    meff2_over_r2 = (meff**2) * flow_.inv_r2[None, :, None, None]
    return Abase_c - (meff2_over_r2 + fourier_.kz2) * c_spin


def _div_c(
    c_rr: Array,
    c_thth: Array,
    c_rth: Array,
    c_rz: Array,
    c_thz: Array,
    c_zz: Array,
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
) -> tuple[Array, Array, Array]:
    r"""Spectral divergence of the symmetric tensor (FFT-free).

    One batched parity-reduced `$D_1$` GEMM for the radial derivatives
    -- `$c_{rr}$` and `$c_{r\theta}$` are in the `$(-1)^m$` class,
    `$c_{rz}$` in the `$(-1)^{m+1}$` one -- then the shared curvature
    assembly (:func:`._viscoelastic_common.div_c_assemble`, which
    carries the component formulas).  The result lands in the classes
    the velocity sources need: `$(\nabla\cdot c)_z$` with `$N_z$`,
    `$(\nabla\cdot c)_{r,\theta}$` with `$N_{r,\theta}$`.
    """
    dr_y = _parity_y_matvec(
        flow_.D1_pos,
        flow_.D1_ghost,
        jnp.stack([c_rr, c_rth, c_rz], axis=1),
        _parity_signs(_DIV_C_SPIN, fourier_),
        component_axis=1,
    )
    return div_c_assemble(
        jnp.swapaxes(dr_y, 0, 1),
        c_rr,
        c_thth,
        c_rth,
        c_rz,
        c_thz,
        c_zz,
        1j * fourier_.m,
        1j * fourier_.kz,
        flow_.inv_r[:, None, None],
    )


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

    ``pi_z`` is the mean-mode axial body force, zero here and set by
    the flow subclass
    (:class:`~dnsjax.flows.wall_bounded.viscoelastic_pipe`), which also
    zeros the base flow (total-field integration).
    """

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
        assembly/factorization to :func:`_build_hc_operator`.

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


def _build_hc_operator(
    dt: float | Array,
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
    *,
    label: str | None,
) -> _WallBoundedOp:
    r"""Factored 6-component `$H_c$` at *dt*.

    Five distinct `$(m_{\mathrm{eff}}^2, \text{parity})$` operators
    (`$s = 0, \pm1, \pm2$`) are built and stacked into the 6-component
    order `$(c_{zz}, c_{z+}, c_{z-}, c_{+-}, c_{++}, c_{--})$`, so the
    `$s = 0$` operator serves both `$c_{zz}$` and `$c_{+-}$`.  The
    parity band follows `$s \bmod 2$` (module docstring): the
    `$s = \pm1$` slots ride the `$(-1)^{m+1}$` band, the rest the
    `$(-1)^m$` one.

    The stacked storage duplicates that shared operator's factors, and
    why that is left alone is the annular twin's
    (``annular_viscoelastic._build_hc_operator``).

    *label* selects the pallas factorization path: a string runs the
    setup-checked :func:`solvers._build_pallas_operator` under that
    diagnostic label; ``None`` runs the unchecked, jittable
    :func:`solvers._factor_pallas_operator` (the ``set_dt``
    rebuild).  The dense backend is pivoted and ignores *label*.
    The wall row comes from the ``hc_narrowN`` leaf.
    """
    kappa = params.phys.kappa
    c_impl = params.step.implicitness
    m_s = fourier_.m[0, ..., None]  # (Nm, 1, 1)
    kz2_s = fourier_.kz2[0, ..., None]  # (1, Nkz, 1)
    m_is_even_s = fourier_.m_is_even[0, ..., None]  # (Nm, 1, 1)
    m_is_even_v = 1.0 - m_is_even_s
    # (m_eff^2, parity mask) for spin s = 0, +1, -1, +2, -2.
    per_spin = {
        s: ((m_s + s) ** 2, m_is_even_v if s % 2 else m_is_even_s)
        for s in (0, 1, -1, 2, -2)
    }
    order = [0, 1, -1, 0, 2, -2]

    if params.solver.backend == "pallas":
        p_band = flow_.Lk_op.L.shape[1]
        band_even = _banded_from_dense(flow_.A_base_even, p_band)
        band_odd = _banded_from_dense(flow_.A_base_odd, p_band)
        bands = [
            _build_Hc_band_gpu(
                band_even,
                band_odd,
                flow_.hc_narrowN,
                per_spin[s][1],
                per_spin[s][0],
                flow_.inv_r2,
                kz2_s,
                dt,
                c_impl,
                kappa,
                p_band,
            )
            for s in order
        ]
        if label is not None:
            return _build_pallas_operator(bands, label)
        return _factor_pallas_operator(bands)

    def _dense(s: int) -> DenseJAXSolver:
        return DenseJAXSolver(
            _build_Hc_dense_gpu(
                flow_.A_base_even,
                flow_.A_base_odd,
                flow_.hc_narrowN,
                per_spin[s][1],
                per_spin[s][0],
                flow_.inv_r2,
                kz2_s,
                dt,
                c_impl,
                kappa,
            )
        )

    solvers_by_spin = {s: _dense(s) for s in (0, 1, -1, 2, -2)}
    return DenseJAXSolver.from_factors(
        lu=jnp.stack([solvers_by_spin[s].lu for s in order]),
        perm=jnp.stack([solvers_by_spin[s].perm for s in order]),
    )


def _build_dt_leaves(
    dt: Array,
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
) -> dict[str, object]:
    r"""Rebuild every ``dt``-dependent flow leaf at the traced *dt*.

    The cylindrical velocity set (`$H_k$` group + IMM leaves;
    ``cylindrical._build_dt_leaves``, with the solvent
    `$\nu = \beta/\mathrm{Re}$` via ``derived_params.nu``) plus the
    conformation `$H_c$` (unchecked factorization,
    :func:`_build_hc_operator`) when diffusion is active.  At
    `$\kappa = 0$` ``Hc_op`` is ``None`` (static aux) and stays out
    of the rebuild -- the trace-time branch matches construction.
    """
    leaves = _cyl_dt_leaves(dt, fourier_, flow_)
    if flow_.Hc_op is not None:
        leaves["Hc_op"] = _build_hc_operator(dt, fourier_, flow_, label=None)
    return leaves


# ── Fused pseudo-spectral RHS ───────────────────────────────────────


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
    measure_fn: Callable[[Array, Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    r"""Evaluate the full 9-component nonlinear RHS ``rhs_no_lapl``.

    One batched inverse transform of ~36 fields (velocity, velocity
    gradient `$L_{ij}$`, physical tensor, and its 18 advection
    derivatives), the shared pointwise physical-space stage
    (:func:`._viscoelastic_common.pointwise_rhs`), one batched forward
    transform of the 9 outputs.  The viscous / diffusive Laplacians are
    added implicitly by the predictor/corrector, so they are absent
    here.  See the module docstring.

    The nine radial derivatives (3 velocity + 6 conformation combos)
    are one parity-reduced GEMM pair, stacked y-leading with the
    per-slot signs of :func:`_parity_signs`; at the default
    ``solver.rhs_transform_chunks = 1`` the inverse/forward transforms
    are each a **single batched** FFT over all fields.  The
    memory-vs-throughput trade of that knob, and the deferred
    interleaved-transform refinement, are the same as in the annular
    twin (``annular_viscoelastic._get_rhs_core``).
    """
    im = 1j * fourier_.m
    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]

    # ── Spectral prep ──
    u_z, u_plus, u_minus = state[0], state[1], state[2]
    u_r = (u_plus + u_minus) / 2
    u_th = -0.5j * (u_plus - u_minus)

    # Conformation physical combos (still spectral here; cs_* denotes
    # the spectral tensor combos, distinct from the physical crr.. that
    # the pointwise stage sees).
    cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz = spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    combos = jnp.array([cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz])

    # Single batched parity-reduced D1 GEMM over the 3 radial velocity
    # derivatives (velocity gradient L_ij = d_i u_j) and the 6 radial
    # conformation advection derivatives -- one GEMM pair instead of
    # two.  y-leading (Nr, 9, Nm, Nkz): transpose-free, and the ghost
    # scatter-add lands on the radial axis.
    dr_y = _parity_y_matvec(
        flow_.D1_pos,
        flow_.D1_ghost,
        jnp.stack(
            [u_r, u_th, u_z, cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz],
            axis=1,
        ),
        _parity_signs(_DR_BATCH_SPIN, fourier_),
        component_axis=1,
    )
    dr_all = jnp.swapaxes(dr_y, 0, 1)  # (9, Nr, Nm, Nkz)
    Lrr, Lrth, Lrz = dr_all[0], dr_all[1], dr_all[2]
    dr_c = dr_all[3:9]  # (6, Nr, Nm, Nkz)
    Lthr = im * inv_r * u_r - inv_r * u_th
    Lthth = im * inv_r * u_th + inv_r * u_r
    Lthz = im * inv_r * u_z
    Lzr = ikz * u_r
    Lzth = ikz * u_th
    Lzz = ikz * u_z

    # Spectral advection derivatives of the conformation combos.
    dth_c = im * combos
    dz_c = ikz * combos

    # ── Batched inverse transform (36 fields) ──
    L_spec = jnp.array([Lrr, Lrth, Lrz, Lthr, Lthth, Lthz, Lzr, Lzth, Lzz])
    u_spec = jnp.array([u_z, u_r, u_th])
    stack = jnp.concatenate([u_spec, L_spec, combos, dr_c, dth_c, dz_c])
    phys = chunked_transform(spec_to_phys_2d, stack)

    # ── Pointwise physical-space stage (shared, coordinate-level) ──
    wi = params.phys.wi
    out_phys, om_phys, trc = pointwise_rhs(
        phys, flow_.inv_r_padded, wi, params.phys.epsilon
    )

    # ── Single batched forward transform (9 outputs) ──
    out_spec = phys_to_spec_2d(out_phys)
    NL_z, NL_r, NL_th = out_spec[0], out_spec[1], out_spec[2]

    # Axial body force at the mean mode.
    NL_z = NL_z + jnp.where(fourier_.mean_mask, flow_.pi_z[:, None, None], 0.0)
    # FFT-free polymer-stress divergence coef * div(c).
    coef = (1.0 - params.phys.beta) / (params.phys.re * wi)
    div_r, div_th, div_z = _div_c(
        cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz, fourier_, flow_
    )
    NL_z = NL_z + coef * div_z
    NL_r = NL_r + coef * div_r
    NL_th = NL_th + coef * div_th

    rhs_uz = NL_z
    rhs_up = NL_r + 1j * NL_th
    rhs_um = NL_r - 1j * NL_th

    # Conformation outputs -> spin components.
    Nc_spin = phys_combos_to_spin(
        out_spec[3],
        out_spec[4],
        out_spec[5],
        out_spec[6],
        out_spec[7],
        out_spec[8],
    )

    rhs = jnp.concatenate([jnp.array([rhs_uz, rhs_up, rhs_um]), Nc_spin])

    # Moving-frame convective term (mode-diagonal on every component).
    u_grid = derived_params.u_grid
    if u_grid != 0:
        rhs = rhs + (1j * u_grid) * fourier_.kz * state

    if measure_fn is None:
        return rhs
    measurements = measure_fn(phys[:3], om_phys, trc)
    return rhs, measurements


def _get_rhs(
    state: Array, fourier_: Fourier, flow_: ViscoelasticCylindricalFlow
) -> Array:
    """Evaluate the 9-component nonlinear RHS."""
    return _get_rhs_core(state, fourier_, flow_, None)


def _get_rhs_measured(
    state: Array, fourier_: Fourier, flow_: ViscoelasticCylindricalFlow
) -> tuple[Array, dict[str, Array]]:
    """Evaluate the RHS + CFL / max-tr(c) measurements."""

    def _measure(
        u_phys: Array, om_phys: Array, trc: Array
    ) -> dict[str, Array]:
        meas = get_cfl(
            u_phys,
            flow_.base_flow_adv_padded,
            flow_.cfl_inv_spacing,
            CFL_NAMES,
            flow_.dt,
        )
        meas["TrC_max"] = jnp.max(trc)
        return meas

    return _get_rhs_core(state, fourier_, flow_, _measure)


# ── FFT-free linear / mean coupling (CN/AB2 scheme) ─────────────────


def _conformation_coupling(
    state: Array,
    combos: tuple[Array, Array, Array, Array, Array, Array],
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
) -> Array:
    r"""FFT-free linear/mean conformation coupling, 6 spin components.

    The cylindrical binding of
    :func:`._viscoelastic_common.conformation_coupling_core` (which
    carries the term-by-term account of what the CN/AB2 scheme makes
    implicit here): this supplies the instantaneous mean velocity
    profile and its **parity-reduced** radial gradients, plus the
    always-implicit moving-frame convective term.

    The mean profiles sit at `$m = 0$`, so their parity signs are the
    constants `$(-1)^s$` -- `$+1$` for the even `$\bar u_z$`, `$-1$`
    for the odd `$\bar u_\theta$` -- and no mode-dependent mask is
    needed (the same shortcut ``pipe.frozen_profile_flow`` takes).
    """
    mean = None
    if params.step.implicit_mean_coupling:
        # Instantaneous mean velocity profile (u_z, u_r, u_theta); the
        # mean u_r is structurally 0, so its d_r term vanishes.
        u_z, u_plus, u_minus = state[0], state[1], state[2]
        u_r = (u_plus + u_minus) / 2
        u_th = -0.5j * (u_plus - u_minus)
        mean_vel = extract_mean_mode(jnp.array([u_z, u_r, u_th]))  # (3, Nr)
        # Mean velocity gradients: the parity-reduced D1 on the bare
        # (N_r,) mean profiles is a direct matmul plus the near-axis
        # ghost rows, at the m = 0 constant signs.
        g = flow_.D1_ghost.shape[0]
        s_uz, s_uth = _mean_parity_signs(np.array([0, 1]))
        d_uz = (
            (flow_.D1_pos @ mean_vel[0])
            .at[:g]
            .add(s_uz * (flow_.D1_ghost @ mean_vel[0]))
        )
        d_uth = (
            (flow_.D1_pos @ mean_vel[2])
            .at[:g]
            .add(s_uth * (flow_.D1_ghost @ mean_vel[2]))
        )
        mean = (
            mean_vel[0][:, None, None],
            mean_vel[2][:, None, None],
            d_uz[:, None, None],
            d_uth[:, None, None],
            1j * fourier_.m,
            1j * fourier_.kz,
            flow_.inv_r[:, None, None],
        )

    conf = conformation_coupling_core(
        combos,
        jnp.where(fourier_.mean_mask, 1.0, 0.0),
        params.phys.epsilon,
        params.phys.wi,
        mean,
    )

    u_grid = derived_params.u_grid
    if u_grid != 0:
        conf = conf + (1j * u_grid) * fourier_.kz * state[3:]
    return conf


def _l_bf(
    state: Array, fourier_: Fourier, flow_: ViscoelasticCylindricalFlow
) -> Array:
    r"""FFT-free linear coupling for the CN/AB2 scheme, all 9 components.

    Velocity slice: the cylindrical base/mean-flow coupling
    (:func:`~dnsjax.geometries.wall_bounded.cylindrical._l_bf`,
    including the moving-frame term) plus the **polymer-stress
    divergence**
    `$\tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}\nabla\cdot\mathbf{c}$`
    (the elastic velocity`$\leftrightarrow$`conformation coupling,
    linear in `$c$` and FFT-free).  Conformation slice:
    :func:`_conformation_coupling`.

    ``step_cnab2`` advances the explicit remainder
    `$\text{get\_rhs} - \text{\_l\_bf}$` (pure fluctuation-fluctuation
    advection / stretching + nonlinear relaxation + the constant body
    force) with AB2 and makes this coupling implicit through the
    FFT-free corrector.  For the total-field viscoelastic pipe the mean
    coupling (velocity *and* the large mean conformation profile) is
    the dominant stiffness.
    """
    vel_lbf = _cyl_l_bf(state[:3], fourier_, flow_)

    cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz = spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    coef = (1.0 - params.phys.beta) / (params.phys.re * params.phys.wi)
    div_r, div_th, div_z = _div_c(
        cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz, fourier_, flow_
    )
    vel_lbf = vel_lbf + coef * jnp.array(
        [div_z, div_r + 1j * div_th, div_r - 1j * div_th]
    )

    conf_lbf = _conformation_coupling(
        state,
        (cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz),
        fourier_,
        flow_,
    )
    return jnp.concatenate([vel_lbf, conf_lbf])


# ── Conformation Crank-Nicolson update ──────────────────────────────


def _c_cn_update(
    c_n: Array,
    Nc_n: Array,
    Nc_j: Array,
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
) -> Array:
    r"""Crank-Nicolson conformation update (6 spin components).

    Solves `$H_c c^{new} = \tfrac1{\Delta t} c^n + (1-\theta)\kappa
    \nabla^2 c^n + \theta N_c^j + (1-\theta) N_c^n$` with the **single**
    wall-row RHS zeroed (the `$\nabla^2 c = 0$` BC at `$r = 1$`; the
    axis carries no row, its regularity being built into the
    parity-reduced band).  With `$\kappa = 0$` there is no diffusion /
    wall BC and the update degenerates to
    `$c^{new} = c^n + \Delta t(\theta N_c^j + (1-\theta) N_c^n)$`.
    """
    dt = flow_.dt
    c_impl = params.step.implicitness
    nl = c_impl * Nc_j + (1.0 - c_impl) * Nc_n
    if flow_.Hc_op is None:  # kappa == 0 (trace-time branch)
        return c_n + dt * nl
    kappa = params.phys.kappa
    lap_cn = _tensor_laplacian_spin(c_n, fourier_, flow_)
    R = (1.0 / dt) * c_n + (1.0 - c_impl) * kappa * lap_cn + nl
    R = R.at[:, -1].set(0.0)  # zero the wall-row RHS
    return flow_.Hc_op.solve(R)


# ── Predictor / corrector / norm ────────────────────────────────────


def _correct(
    state_prev: Array,
    prediction: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
) -> tuple[Array, Array]:
    """Coupled velocity-IMM + conformation-CN corrector.

    Velocity: the cylindrical `$1\\times1$` influence-matrix iteration
    (which sees the polymer divergence only through the sources, so it
    needs no viscoelastic knowledge).  Conformation: the Crank-Nicolson
    Helmholtz update.  The returned correction stacks both so the single
    convergence norm covers `$u$` and `$c$`.
    """
    vel_new, vel_corr = _imm_iteration(
        state_prev[:3],
        prediction[:3],
        rhs_prev[:3],
        rhs_next[:3],
        fourier_,
        flow_,
    )
    c_new = _c_cn_update(
        state_prev[3:], rhs_prev[3:], rhs_next[3:], fourier_, flow_
    )
    c_corr = c_new - prediction[3:]
    state_new = jnp.concatenate([vel_new, c_new])
    correction = jnp.concatenate([vel_corr, c_corr])
    return state_new, correction


def _predict(
    state_n: Array,
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
) -> Array:
    """Euler predictor (nonlinear at `$u^n$`, viscous/diffusive CN)."""
    prediction, _ = _correct(
        state_n, state_n, rhs_no_lapl, rhs_no_lapl, fourier_, flow_
    )
    return prediction


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: ViscoelasticCylindricalFlow,
) -> Array:
    r"""Combined L2 convergence norm, `$\sqrt{\|u\|^2 + \|c\|_F^2}$`
    (:func:`._viscoelastic_common.combined_norm`)."""
    return combined_norm(correction, fourier_.k_metric, flow_.y_weights)


# ── Stepper factory ─────────────────────────────────────────────────


def build_viscoelastic_stepper(flow: ViscoelasticCylindricalFlow):
    """Build time-stepping functions for a viscoelastic pipe flow.

    Returns the same 9-tuple as
    :func:`~dnsjax.geometries.wall_bounded._base.build_wall_bounded_stepper`
    (incl. the adaptive-dt ``set_dt`` / ``reset_ab2_kappa``, backed
    by this module's ``_build_dt_leaves``).
    ``_l_bf`` (the FFT-free linear/mean coupling: velocity mean-flow
    coupling + polymer-stress divergence, conformation mean advection /
    stretching / linear relaxation) is passed so the CN/AB2 scheme
    treats it implicitly and the explicit AB2 remainder stays pure
    fluctuation-fluctuation nonlinearity.
    """
    from ._base import build_wall_bounded_stepper

    return build_wall_bounded_stepper(
        _get_rhs,
        _predict,
        _correct,
        _norm,
        fourier,
        flow,
        _get_rhs_measured,
        _l_bf,
        dt_leaves_fn=_build_dt_leaves,
    )
