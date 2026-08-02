r"""Viscoelastic (sPTT) extension of the annular geometry.

Adds a symmetric conformation tensor `$\mathbf{c}$` to the annular
(concentric-cylinder) geometry, coupling it to the velocity via the
polymer-stress divergence.  The time-integrated state grows from the 3
velocity components to **9**: 3 velocity + 6 independent symmetric-tensor
components, carried as a single stacked spectral array
``(9, Nr, Nm, Nkz)`` in the layout

.. math::
    [\,u_z,\; u_+,\; u_-,\;
      c_{zz},\; c_{z+},\; c_{z-},\; c_{+-},\; c_{++},\; c_{--}\,]

where the velocity uses the decoupled `$u_\pm = u_r \pm i u_\theta$`
formulation of :mod:`~dnsjax.geometries.wall_bounded.annular`, and the
tensor uses the analogous **spin** projections

.. math::
    c_{z\pm} = c_{rz} \pm i c_{\theta z}, \qquad
    c_{+-}   = c_{rr} + c_{\theta\theta}, \qquad
    c_{\pm\pm} = (c_{rr} - c_{\theta\theta}) \pm 2 i c_{r\theta}.

As in the annular geometry this is the solver's **working** basis.
Outside the time stepper -- snapshots, diagnostics, probes, initial
conditions, the analysis package -- the state is the physical
9-component layout

.. math::
    [\,u_z,\; u_r,\; u_\theta,\;
      c_{zz},\; c_{rz},\; c_{\theta z},\;
      c_{rr},\; c_{\theta\theta},\; c_{r\theta}\,],

and a given state crosses between the two at most once
(:func:`to_spin_basis` / :func:`from_spin_basis`, driven by
:mod:`dnsjax.__main__`).

Spin diagonalisation of the tensor Laplacian
--------------------------------------------
The cylindrical Laplacian couples the physical tensor components through
`$1/r^2$` terms, exactly as the vector Laplacian couples `$u_r, u_\theta$`.
Writing `$\partial_\theta \to im$` and collecting the basis-rotation
generator `$\mathcal R$` (the `$\partial_\theta$` action on the tensor
basis), the angular part of the Laplacian is
`$\tfrac{1}{r^2}(\mathcal R + im)^2$`.  Each spin projection is an
eigenvector of `$\mathcal R$` with eigenvalue `$is$` (spin weight
`$s$`), so `$(\mathcal R + im)^2 \to -(m + s)^2$` and the tensor
Laplacian **diagonalises**:

.. math::
    (\nabla^2 \mathbf{c})_{\text{spin }s} =
    \Bigl[\partial_r^2 + \tfrac{1}{r}\partial_r
    - \tfrac{(m+s)^2}{r^2} - k_z^2\Bigr]\,\mathbf{c}_{\text{spin }s},

with spin weights `$s = 0$` for `$c_{zz}, c_{+-}$`, `$s = \pm1$` for
`$c_{z\pm}$`, and `$s = \pm2$` for `$c_{\pm\pm}$` -- the same mechanism as
`$u_\pm$` (`$m_{\mathrm{eff}} = m \pm 1$`).  Each spin component therefore
diffuses through a scalar Helmholtz solve with its own
`$m_{\mathrm{eff}}$`, reusing the annular dense/Pallas machinery.

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

with no-slip `$\mathbf{u}=0$` and `$\nabla^2\mathbf{c}=0$` at both walls,
and the azimuthal body force `$\Pi_\theta = (r_1+r_2)/(\mathrm{Re}\,r)$`
(see :mod:`~dnsjax.flows.wall_bounded.viscoelastic_dean`).  All products
are at most quadratic (`$\mathrm{tr}(\mathbf{c})\,\mathbf{c}$`), so the
existing 3/2-rule dealiasing is exact.

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
inverse transform of ~36 fields, one batched forward transform of the 9
outputs; the vorticity is free from the velocity-gradient tensor).  The
predictor/corrector then solves the velocity via the annular 2x2 IMM
(:func:`~dnsjax.geometries.wall_bounded.annular._imm_iteration`,
solvent viscosity `$\nu = \beta/\mathrm{Re}$`) and the conformation via a
Crank-Nicolson Helmholtz solve per spin component (`$H_c = \tfrac1{\Delta
t}I - \theta\kappa\nabla^2$`, Laplacian BC wall rows).  With
`$\kappa = 0$` the transport is purely hyperbolic (no wall BC) and the
conformation update degenerates to the explicit CN combination.

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
    apply_y_matrix,
    extract_mean_mode,
    integrate_scalar,  # noqa: F401 -- re-exported for the flow module
)
from ._viscoelastic_common import (
    C_FROB_SQRT_SPIN,  # noqa: F401 -- re-exported
    N_VE_COMPONENTS,  # noqa: F401 -- re-exported
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
from .annular import (
    CFL_NAMES,
    AnnularFlow,
    Fourier,
    _imm_iteration,
    fourier,
)
from .annular import (
    _build_dt_leaves as _annular_dt_leaves,
)
from .annular import (
    _l_bf as _annular_l_bf,
)

# ── State layout ────────────────────────────────────────────────────
#
# Solver basis: state[0:3] = velocity (u_z, u_+, u_-); state[3:9] =
# conformation spin components (c_zz, c_z+, c_z-, c_+-, c_++, c_--).
# Physical (everything outside the stepper): (u_z, u_r, u_theta) +
# (c_zz, c_rz, c_theta_z, c_rr, c_theta_theta, c_r_theta).  The layout,
# the spin weights, the basis pair and the pointwise physical-space
# arithmetic are geometry-free and shared with the pipe's viscoelastic
# geometry: :mod:`._viscoelastic_common`.

#: Role aliases for the basis boundary (see ``cylindrical.py``).
to_solver_basis = to_spin_basis
from_solver_basis = from_spin_basis


# ── Analytical laminar profiles (JAX-free, build-time) ──────────────


def viscoelastic_laminar_profiles(
    rs: np.ndarray, D1: np.ndarray, r1: float, r2: float, wi: float, eps: float
) -> np.ndarray:
    r"""9-component laminar `$r$`-profiles for a force-driven annular
    sPTT flow (complex ``(9, Nr)``), in the **physical** state layout
    `$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` -- these feed initial conditions
    and the flow's laminar reference, both of which live outside the
    solver (:func:`to_spin_basis` converts when one enters it).

    Velocity: the azimuthal profile `$U_\theta(r)$` (body-force
    coefficient `$C = r_1 + r_2$`).  Conformation: the pointwise sPTT
    equilibrium on the **discrete** local shear `$S = D_1 U_\theta -
    U_\theta/r$` (see the
    :mod:`~dnsjax.flows.wall_bounded.viscoelastic_dean` module
    docstring): `$c_{rr} = c_{zz} = 1$`, `$c_{r\theta} =
    \mathrm{Wi}\,S/f$`, `$c_{\theta\theta} = 1 + 2 c_{r\theta}^2$`.
    Pure (NumPy, build-time); shared by the flow's laminar state and
    the viscoelastic random / rolls ICs.
    """
    from .annular import annular_forced_laminar_u_theta

    rs_np = np.asarray(rs)
    u_theta = np.asarray(
        annular_forced_laminar_u_theta(jnp.asarray(rs_np), r1, r2, r1 + r2)
    )
    shear = np.asarray(D1) @ u_theta - u_theta / rs_np
    wis = wi * shear
    f = solve_ptt_f(2.0 * eps * wis**2)
    x = wis / f  # c_r_theta
    c_thth = 1.0 + 2.0 * x**2  # c_theta_theta
    zeros = np.zeros_like(rs_np, dtype=np.complex128)
    ones = np.ones_like(rs_np, dtype=np.complex128)
    return np.stack(
        [
            zeros,  # u_z
            zeros,  # u_r
            u_theta,  # u_theta
            ones,  # c_zz
            zeros,  # c_rz
            zeros,  # c_theta_z
            ones,  # c_rr
            c_thth,  # c_theta_theta
            x,  # c_r_theta
        ]
    ).astype(np.complex128)


# ── Narrow (banded-storage-fitting) Laplacian BC wall rows ──────────


def _narrow_abase_wall_rows(
    rs: np.ndarray, D1: np.ndarray, fd_order: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""Both narrow-`$D_2$` `$A_{\mathrm{base}}$` wall rows.

    The annulus carries a `$\nabla^2 c = 0$` BC row at each of its two
    walls; :func:`._viscoelastic_common.narrow_abase_wall_row` builds
    one (and documents why the regular `$D_2$` boundary row cannot be
    used).  Returns the two full-length `$(N_r,)$` rows.
    """
    return (
        narrow_abase_wall_row(rs, D1, fd_order, inner=True),
        narrow_abase_wall_row(rs, D1, fd_order, inner=False),
    )


# ── H_c Helmholtz operator builders (per spin component) ────────────


def _build_Hc_dense_gpu(
    A_base: Array,
    narrow0: Array,
    narrowN: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    kappa: float,
) -> Array:
    r"""Dense `$H_c = \tfrac1{\Delta t} I - c\kappa\nabla^2$` for one spin
    component (dense backend).  Interior rows carry the diagonal
    Helmholtz shift; both wall rows are the narrow Laplacian BC row
    `$A_{\mathrm{base}} - (m_{\mathrm{eff}}^2/r^2 + k_z^2) I$`."""
    Nr = A_base.shape[0]
    dtype = A_base.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)
    diag_coeff = 1.0 / dt + c * kappa * (meff2 * inv_r2 + kz2)  # (Nm,Nkz,Nr)
    Hc = diag_coeff[..., None] * eye_Nr - c * kappa * A_base
    # Wall rows: narrow Laplacian BC (mode-dependent diagonal shift).
    shift0 = meff2 * inv_r2[0] + kz2  # (Nm, Nkz, 1)
    shiftN = meff2 * inv_r2[-1] + kz2
    row0 = narrow0[None, None] - shift0 * eye_Nr[0]  # (Nm, Nkz, Nr)
    rowN = narrowN[None, None] - shiftN * eye_Nr[-1]
    Hc = Hc.at[..., 0, :].set(row0)
    Hc = Hc.at[..., -1, :].set(rowN)
    return Hc


def _build_Hc_band_gpu(
    A_base: Array,
    narrow0: Array,
    narrowN: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    kappa: float,
    p: int,
) -> Array:
    r"""Banded `$H_c$` for one spin component (Pallas backend), layout
    ``(Nm, Nkz, Nr, 2p+1)``; narrow Laplacian BC wall rows."""
    Nr = A_base.shape[0]
    band_base = _banded_from_dense(A_base, p)
    diag = 1.0 / dt + c * kappa * (meff2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    e = _banded_diag_column(p, band_base.dtype)
    # Narrow BC bands (mode-constant) minus the mode-dependent shift.
    band0 = _banded_wall_row(narrow0, 0, p)  # (2p+1,)
    bandN = _banded_wall_row(narrowN, Nr - 1, p)
    shift0 = meff2 * inv_r2[0] + kz2  # (Nm, Nkz, 1)
    shiftN = meff2 * inv_r2[-1] + kz2
    row0 = band0 - shift0 * e  # (Nm, Nkz, 2p+1)
    rowN = bandN - shiftN * e
    return _assemble_banded_operator(
        band_base, -c * kappa, diag, [(0, row0), (Nr - 1, rowN)]
    )


# ── Spectral tensor operators (FFT-free) ────────────────────────────


def _tensor_laplacian_spin(
    c_spin: Array, fourier_: Fourier, flow_: ViscoelasticAnnularFlow
) -> Array:
    r"""Spin-diagonal tensor Laplacian, `$(6, N_r, N_m, N_{kz})$`.

    `$(\nabla^2 c)_{\text{spin }s} = A_{\mathrm{base}} c
    - (m_{\mathrm{eff}}^2/r^2 + k_z^2) c$` with
    `$m_{\mathrm{eff}} = m + s$` per spin component.
    """
    inv_r = flow_.inv_r[:, None, None]
    Abase_c = apply_y_matrix(flow_.D2, c_spin) + inv_r * apply_y_matrix(
        flow_.D1, c_spin
    )
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
    flow_: ViscoelasticAnnularFlow,
) -> tuple[Array, Array, Array]:
    r"""Spectral divergence of the symmetric tensor (FFT-free).

    One batched `$D_1$` GEMM for the radial derivatives, then the
    shared curvature assembly
    (:func:`._viscoelastic_common.div_c_assemble`, which carries the
    component formulas).
    """
    dr = apply_y_matrix(
        flow_.D1, jnp.array([c_rr, c_rth, c_rz])
    )  # (3, Nr, Nm, Nkz)
    return div_c_assemble(
        dr,
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


# ── Viscoelastic annular flow dataclass ─────────────────────────────

_WallBoundedOp = DenseJAXSolver | PerModeBandedPallasOperator


@register_dataclass_pytree
@dataclass
class ViscoelasticAnnularFlow(AnnularFlow):
    r"""Precomputed data for viscoelastic (sPTT) annular flow.

    Extends :class:`~dnsjax.geometries.wall_bounded.annular.AnnularFlow`
    (velocity radial grid, FD matrices, 2x2 IMM operators -- built with
    solvent viscosity `$\nu = \beta/\mathrm{Re}$` via
    ``derived_params.nu``) with the conformation-tensor machinery: the
    stacked Crank-Nicolson Helmholtz operator ``Hc_op`` (6 spin
    components, `$m_{\mathrm{eff}} = m + s$`; two share
    `$m_{\mathrm{eff}} = m$`) and the physical `$1/r$` profile on the
    padded grid.  When `$\kappa = 0$` no Helmholtz operator is built
    (``Hc_op = None``): the conformation transport is hyperbolic and the
    update is the explicit CN combination.

    Subclasses (:class:`~dnsjax.flows.wall_bounded.viscoelastic_dean`)
    set ``pi_theta`` and zero the base flow (total-field integration).
    """

    tensor_spin: Array = field(init=False)
    inv_r_padded: Array = field(init=False)
    Hc_op: _WallBoundedOp | None = field(init=False, default=None)
    # Narrow Laplacian BC wall rows of Hc, stored as leaves so the
    # jitted adaptive-dt rebuild (``_build_dt_leaves``) can reuse
    # them (their NumPy build cannot run on tracers).  ``None`` (aux)
    # while kappa == 0, where no Hc exists.
    hc_narrow0: Array | None = field(init=False, default=None)
    hc_narrowN: Array | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Build velocity grid / FD matrices / 2x2 IMM (nu = beta/re).
        super().__post_init__()

        self.tensor_spin = jax.device_put(
            jnp.asarray(TENSOR_SPIN, dtype=sharding.float_type),
            sharding.no_shard,
        )

        Nr = params.res.ny
        ny_phys = Nr + sharding.ny_y_pad
        inv_r_pad = np.zeros(ny_phys, dtype=sharding.float_type)
        inv_r_pad[:Nr] = np.asarray(self.inv_r)
        self.inv_r_padded = jax.device_put(
            inv_r_pad.reshape(ny_phys, 1, 1), sharding.no_shard
        )

        kappa = params.phys.kappa
        if kappa == 0:
            # Hyperbolic conformation transport: no diffusion, no wall BC.
            self.Hc_op = None
            return

        self._build_conformation_operator()

    def _build_conformation_operator(self) -> None:
        r"""Build the stacked 6-component `$H_c$` Crank-Nicolson operator.

        Stores the narrow Laplacian BC wall rows as the
        ``hc_narrow0`` / ``hc_narrowN`` leaves (their JAX-free NumPy
        build cannot run on tracers, so the jitted adaptive-``dt``
        rebuild reuses them), optionally pre-checks the no-pivot LU
        at ``dt_max`` (``step.adaptive``; the velocity `$H_k$`
        analogue), and delegates the assembly/factorization to
        :func:`_build_hc_operator`.
        """
        # Full narrow Laplacian BC wall rows (JAX-free build).
        row0_np, rowN_np = _narrow_abase_wall_rows(
            np.asarray(self.rs), np.asarray(self.D1), params.res.fd_order
        )
        self.hc_narrow0 = jax.device_put(row0_np, sharding.no_shard)
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
    flow_: ViscoelasticAnnularFlow,
    *,
    label: str | None,
) -> _WallBoundedOp:
    r"""Factored 6-component `$H_c$` at *dt*.

    Five distinct `$m_{\mathrm{eff}}^2$` operators (`$m, m\pm1,
    m\pm2$`) are built and stacked into the 6-component order
    `$(c_{zz}, c_{z+}, c_{z-}, c_{+-}, c_{++}, c_{--})$`, so the
    `$m_{\mathrm{eff}} = m$` operator serves both `$c_{zz}$` and
    `$c_{+-}$`.

    The stacked storage **duplicates** that shared operator's
    factors (slot 0 and slot 3 hold the same data -- ~1/6 of the
    ``Hc_op`` memory), because the uniform stacked ``.solve``
    contract pairs component ``i`` of the RHS with operator ``i``.
    Deduplicating would need a nonuniform component-to-operator
    solve mapping (5 operators against 6 RHS components) in every
    backend -- deferred as not worth the contract complexity for a
    small, setup-persistent array (the velocity ``Hk_op`` stack and
    the per-step transform transients are far larger).

    *label* selects the pallas factorization path: a string runs the
    setup-checked :func:`solvers._build_pallas_operator` under that
    diagnostic label; ``None`` runs the unchecked, jittable
    :func:`solvers._factor_pallas_operator` (the ``set_dt``
    rebuild).  The dense backend is pivoted and ignores *label*.
    Wall rows come from the ``hc_narrow0``/``hc_narrowN`` leaves.
    """
    kappa = params.phys.kappa
    c_impl = params.step.implicitness
    m_s = fourier_.m[0, ..., None]  # (Nm, 1, 1)
    kz2_s = fourier_.kz2[0, ..., None]  # (1, Nkz, 1)
    # m_eff^2 for spin s = 0, +1, -1, +2, -2 (the 5 distinct values).
    meff2 = {s: (m_s + s) ** 2 for s in (0, 1, -1, 2, -2)}

    if params.solver.backend == "pallas":
        # Half-width read back from the already-factored, dt-independent
        # Lk, exactly as ``annular._hk_bands`` does: a static shape, so
        # it works inside the jitted ``set_dt`` rebuild, and it is
        # *measured* rather than assumed to be ``fd_order`` -- an
        # under-sized band truncates entries silently
        # (``fd.matrix_half_bandwidth``).  Read inside this branch:
        # the dense backend's ``Lk_op`` is a ``DenseJAXSolver``, which
        # carries no band.
        fd_p = flow_.Lk_op.L.shape[1]
        # Six per-spin banded operators (slot 3 repeats s = 0),
        # stacked into one homogeneous operator.
        bands = [
            _build_Hc_band_gpu(
                flow_.A_base,
                flow_.hc_narrow0,
                flow_.hc_narrowN,
                meff2[s],
                flow_.inv_r2,
                kz2_s,
                dt,
                c_impl,
                kappa,
                fd_p,
            )
            for s in (0, 1, -1, 0, 2, -2)
        ]
        if label is not None:
            return _build_pallas_operator(bands, label)
        return _factor_pallas_operator(bands)

    def _dense(s: int) -> DenseJAXSolver:
        H = _build_Hc_dense_gpu(
            flow_.A_base,
            flow_.hc_narrow0,
            flow_.hc_narrowN,
            meff2[s],
            flow_.inv_r2,
            kz2_s,
            dt,
            c_impl,
            kappa,
        )
        return DenseJAXSolver(H)

    solvers_by_spin = {s: _dense(s) for s in (0, 1, -1, 2, -2)}
    order = [0, 1, -1, 0, 2, -2]
    return DenseJAXSolver.from_factors(
        lu=jnp.stack([solvers_by_spin[s].lu for s in order]),
        perm=jnp.stack([solvers_by_spin[s].perm for s in order]),
    )


def _build_dt_leaves(
    dt: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> dict[str, object]:
    r"""Rebuild every ``dt``-dependent flow leaf at the traced *dt*.

    The annular velocity set (`$H_k$` group + IMM leaves;
    ``annular._build_dt_leaves``, with the solvent
    `$\nu = \beta/\mathrm{Re}$` via ``derived_params.nu``) plus the
    conformation `$H_c$` (unchecked factorization,
    :func:`_build_hc_operator`) when diffusion is active.  At
    `$\kappa = 0$` ``Hc_op`` is ``None`` (static aux) and stays out
    of the rebuild -- the trace-time branch matches construction.
    """
    leaves = _annular_dt_leaves(dt, fourier_, flow_)
    if flow_.Hc_op is not None:
        leaves["Hc_op"] = _build_hc_operator(dt, fourier_, flow_, label=None)
    return leaves


# ── Fused pseudo-spectral RHS ───────────────────────────────────────


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
    measure_fn: Callable[[Array, Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    r"""Evaluate the full 9-component nonlinear RHS ``rhs_no_lapl``.

    One batched inverse transform of ~36 fields (velocity, velocity
    gradient `$L_{ij}$`, physical tensor, and its 18 advection
    derivatives), pointwise physical-space arithmetic, one batched
    forward transform of the 9 outputs.  The viscous / diffusive
    Laplacians are added implicitly by the predictor/corrector, so they
    are absent here.  See the module docstring.

    The two radial-derivative GEMMs (3 velocity + 6 conformation combos)
    are fused into one ``apply_y_matrix`` call, and at the default
    ``solver.rhs_transform_chunks = 1`` the inverse/forward transforms
    are each a **single batched** FFT over all fields (pinned by
    ``test_fused_rhs_transform_count``).

    **Memory vs throughput** (``solver.rhs_transform_chunks``): this
    36-field inverse transform dominates a viscoelastic step's peak
    memory -- not the held physical outputs themselves, but the
    transform's padded intermediate stage buffers (~2 complex copies
    of the whole batch at the dealiased size; see :mod:`dnsjax.fft`).
    The shared :func:`dnsjax.fft.chunked_transform` applies the knob:
    ``k`` balanced groups cut that transient by ~``k`` at the cost of
    ``k``x the FFT dispatches (and ``k`` smaller reshard rounds per
    stage on multi-device runs); the results are identical.

    **Deferred optimisation (interleaved transform/accumulate)**:
    chunking caps only the transform transient -- all 36 physical
    fields must still coexist as inputs of the single pointwise
    stage, so they plus the 9 outputs (~45 oversampled fields) are
    the floor the knob cannot cut.  That floor is decomposable
    because the pointwise stage has sparse field incidence: the 18
    advection derivatives are strictly per-component (only
    `$\mathrm{adv}(c_i)$` reads the
    `$(\partial_r, \partial_\theta, \partial_z) c_i$` triple),
    while the velocities, `$L_{ij}$`, and tensor combos are shared.
    Interleaving would hold the shared fields, then per component
    transform its derivative triple, multiply-accumulate its
    advection contribution into the output, and let the triple die
    before the next component's transform -- cutting the held floor
    to ~30 fields (further if the `$L_{ij}$` contributions are
    accumulated and freed first).  Deferred because it hard-codes
    chunking's throughput cost even when memory is not tight: the
    fused one-pass pointwise stage shatters into per-group kernels,
    the outputs are re-read/re-written once per group, the transform
    batches become permanently small, the schedule is specific to
    this RHS's term structure (unlike the flow-agnostic
    ``chunked_transform``), and the freeing relies on XLA liveness
    rather than construction.  The 9-output forward transform stays
    fused for the related reason that all outputs exist before it
    starts, so chunking it could shave only its own minor transient.
    """
    im = 1j * fourier_.m
    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]

    # ── Spectral prep ──
    u_z, u_plus, u_minus = state[0], state[1], state[2]
    u_r = (u_plus + u_minus) / 2
    u_th = -0.5j * (u_plus - u_minus)

    # Conformation physical combos (still spectral here; cs_* denotes
    # the spectral tensor combos, distinct from the physical crr.. below).
    cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz = spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    combos = jnp.array([cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz])

    # Single batched D1 GEMM over the 3 radial velocity derivatives
    # (velocity gradient L_ij = d_i u_j) and the 6 radial conformation
    # advection derivatives -- one GEMM instead of two (bit-identical;
    # the per-field matmul is batch-independent).
    dr_all = apply_y_matrix(
        flow_.D1,
        jnp.concatenate([jnp.array([u_r, u_th, u_z]), combos]),
    )
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
    # One fused batch by default; ``solver.rhs_transform_chunks = k``
    # (trace-time, static) splits it into k balanced groups to cap the
    # transform-stage transient -- see the docstring.
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

    # Azimuthal body force at the mean mode.
    NL_th = NL_th + jnp.where(
        fourier_.mean_mask, flow_.pi_theta[:, None, None], 0.0
    )
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
    state: Array, fourier_: Fourier, flow_: ViscoelasticAnnularFlow
) -> Array:
    """Evaluate the 9-component nonlinear RHS."""
    return _get_rhs_core(state, fourier_, flow_, None)


def _get_rhs_measured(
    state: Array, fourier_: Fourier, flow_: ViscoelasticAnnularFlow
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
    flow_: ViscoelasticAnnularFlow,
) -> Array:
    r"""FFT-free linear/mean conformation coupling, 6 spin components.

    The annular binding of
    :func:`._viscoelastic_common.conformation_coupling_core` (which
    carries the term-by-term account of what the CN/AB2 scheme makes
    implicit here): this supplies the instantaneous mean velocity
    profile and its **plain-`$D_1$`** radial gradients -- the annulus
    has no axis, so no parity enters -- plus the always-implicit
    moving-frame convective term.
    """
    mean = None
    if params.step.implicit_mean_coupling:
        # Instantaneous mean velocity profile (u_z, u_r, u_theta); the
        # mean u_r is structurally 0, so its d_r term vanishes.
        u_z, u_plus, u_minus = state[0], state[1], state[2]
        u_r = (u_plus + u_minus) / 2
        u_th = -0.5j * (u_plus - u_minus)
        mean_vel = extract_mean_mode(jnp.array([u_z, u_r, u_th]))  # (3, Nr)
        # Mean velocity gradient profiles: D1 on the bare (N_r,) mean
        # profiles is a direct matmul (no Fourier axes here).
        mean = (
            mean_vel[0][:, None, None],
            mean_vel[2][:, None, None],
            (flow_.D1 @ mean_vel[0])[:, None, None],  # d_r u_z
            (flow_.D1 @ mean_vel[2])[:, None, None],  # d_r u_theta
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
    state: Array, fourier_: Fourier, flow_: ViscoelasticAnnularFlow
) -> Array:
    r"""FFT-free linear coupling for the CN/AB2 scheme, all 9 components.

    Velocity slice: the annular base/mean-flow coupling (:func:`~dnsjax.
    geometries.wall_bounded.annular._l_bf`, including the moving-frame
    term) plus the **polymer-stress divergence**
    `$\tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}\nabla\cdot\mathbf{c}$`
    (the elastic velocity`$\leftrightarrow$`conformation coupling, linear
    in `$c$` and FFT-free).  Conformation slice:
    :func:`_conformation_coupling`.

    ``step_cnab2`` advances the explicit remainder
    `$\text{get\_rhs} - \text{\_l\_bf}$` (pure fluctuation-fluctuation
    advection / stretching + nonlinear relaxation + the constant body
    force) with AB2 and makes this coupling implicit through the
    FFT-free corrector.  For total-field viscoelastic Dean the mean
    coupling (velocity *and* the large mean conformation profile) is the
    dominant stiffness, exactly as the mean-flow coupling is for
    Newtonian Dean.
    """
    vel_lbf = _annular_l_bf(state[:3], fourier_, flow_)

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
    flow_: ViscoelasticAnnularFlow,
) -> Array:
    r"""Crank-Nicolson conformation update (6 spin components).

    Solves `$H_c c^{new} = \tfrac1{\Delta t} c^n + (1-\theta)\kappa
    \nabla^2 c^n + \theta N_c^j + (1-\theta) N_c^n$` with the wall-row
    RHS zeroed (the `$\nabla^2 c = 0$` BC).  With `$\kappa = 0$` there is
    no diffusion / wall BC and the update degenerates to
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
    R = R.at[:, 0].set(0.0).at[:, -1].set(0.0)  # zero wall-row RHS
    return flow_.Hc_op.solve(R)


# ── Predictor / corrector / norm ────────────────────────────────────


def _correct(
    state_prev: Array,
    prediction: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> tuple[Array, Array]:
    """Coupled velocity-IMM + conformation-CN corrector.

    Velocity: the annular 2x2 influence-matrix iteration.  Conformation:
    the Crank-Nicolson Helmholtz update.  The returned correction stacks
    both so the single convergence norm covers `$u$` and `$c$`.
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
    flow_: ViscoelasticAnnularFlow,
) -> Array:
    """Euler predictor (nonlinear at `$u^n$`, viscous/diffusive CN)."""
    prediction, _ = _correct(
        state_n, state_n, rhs_no_lapl, rhs_no_lapl, fourier_, flow_
    )
    return prediction


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> Array:
    r"""Combined L2 convergence norm, `$\sqrt{\|u\|^2 + \|c\|_F^2}$`
    (:func:`._viscoelastic_common.combined_norm`)."""
    return combined_norm(correction, fourier_.k_metric, flow_.y_weights)


# ── Stepper factory ─────────────────────────────────────────────────


def build_viscoelastic_stepper(flow: ViscoelasticAnnularFlow):
    """Build time-stepping functions for a viscoelastic annular flow.

    Returns the same 9-tuple as
    :func:`~dnsjax.geometries.wall_bounded._base.build_wall_bounded_stepper`
    (incl. the adaptive-dt ``set_dt`` / ``reset_ab2_kappa``, backed
    by this module's ``_build_dt_leaves``).
    ``_l_bf`` (the FFT-free linear/mean coupling: velocity mean-flow
    coupling + polymer-stress divergence, conformation mean advection /
    stretching / linear relaxation) is passed so the CN/AB2 scheme treats
    it implicitly and the explicit AB2 remainder stays pure
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
