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

What lives where
----------------
The time stepper itself -- the fused pseudo-spectral RHS, the FFT-free
CN/AB2 coupling, the conformation Crank-Nicolson update, the
predictor / corrector / norm, the `$H_c$` builders and the stepper
factory -- is shared with the pipe's sPTT geometry and lives in
:mod:`._viscoelastic_stepping`; the coordinate-level algebra under it
(spin `$\leftrightarrow$` physical maps, the pointwise physical-space
kernel, the `$\nabla\cdot c$` curvature assembly) in
:mod:`._viscoelastic_common`.  **This** module owns the annular half:
the flow dataclass, the analytical laminar profiles, the narrow
Laplacian BC wall rows, and the adapter methods the shared stepper
dispatches on (plain `$D_1$`/`$D_2$` radial derivatives, two wall
rows, the azimuthal body force -- see
:class:`ViscoelasticAnnularFlow`).

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
      - \boldsymbol{\Pi}, \\
    \partial_t \mathbf{c} + \mathbf{u}\cdot\nabla\mathbf{c}
      - (\nabla\mathbf{u})^{\!\top}\!\cdot\mathbf{c}
      - \mathbf{c}\cdot\nabla\mathbf{u}
      &= \kappa\nabla^2\mathbf{c}
      - \tfrac{\mathbf{c}-\mathbb{I}}{\mathrm{Wi}}
        (1 - 3\epsilon + \epsilon\,\mathrm{tr}\,\mathbf{c}),

with no-slip `$\mathbf{u}=0$` and `$\nabla^2\mathbf{c}=0$` at both walls,
and the azimuthal body force `$-\Pi_\theta = (r_1+r_2)/(\mathrm{Re}\,r)$`
(see :mod:`~dnsjax.flows.wall_bounded.viscoelastic_dean`).  All products
are at most quadratic (`$\mathrm{tr}(\mathbf{c})\,\mathbf{c}$`), so the
existing 3/2-rule dealiasing is exact.

Time integration
----------------
Both ``iterative-cn`` (default) and ``cnab2`` schemes are supported.
``get_rhs`` returns the full 9-component nonlinear RHS -- velocity
(`$\mathbf{u}\times\boldsymbol\omega$` + `$-\boldsymbol\Pi$` + FFT-free
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
    apply_y_matrix,
    integrate_scalar,  # noqa: F401 -- re-exported for the flow module
)
from ._viscoelastic_common import (
    C_FROB_SQRT_SPIN,  # noqa: F401 -- re-exported
    N_VE_COMPONENTS,  # noqa: F401 -- re-exported
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

    The methods below are the **annular half of the adapter surface**
    the shared stepper (:mod:`._viscoelastic_stepping`) dispatches on,
    resolved once at trace time; being methods rather than fields they
    add no pytree leaf.  In short: plain `$D_1$`/`$D_2$` radial
    derivatives (no axis, so no parity anywhere), two `$\nabla^2 c = 0$`
    wall rows, and an azimuthal mean-mode body force.

    Subclasses (:class:`~dnsjax.flows.wall_bounded.viscoelastic_dean`)
    set ``force_theta`` and zero the base flow (total-field integration).
    """

    #: CFL column labels (a ``ClassVar``: as an annotated field this
    #: tuple would become two static pytree entries per flatten).
    cfl_names: ClassVar[tuple[str, str, str]] = CFL_NAMES

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
        :func:`._viscoelastic_stepping._build_hc_operator`.
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

    # ── Adapter surface (see ``_viscoelastic_stepping``) ────────────

    def rhs_radial_derivatives(
        self,
        fields: tuple[Array, ...],
        combos: Array,
        fourier_: Fourier,
    ) -> Array:
        r"""The fused 9-field radial-derivative batch of the RHS.

        One `$D_1$` GEMM over the 3 velocity components and the 6
        conformation combos -- one GEMM instead of two (bit-identical;
        the per-field matmul is batch-independent).  *fields* is the
        flat 9-tuple `$(u_r, u_\theta, u_z, c_{rr}, \ldots)$` and
        *combos* the already-materialised `$(6, \ldots)$` tensor batch:
        the annulus concatenates behind the velocity triple (reusing
        that buffer), the pipe stacks the flat tuple y-leading, so the
        shared caller hands over both forms.  Returns
        ``(9, Nr, Nm, Nkz)``.
        """
        return apply_y_matrix(
            self.D1,
            jnp.concatenate(
                [jnp.array([fields[0], fields[1], fields[2]]), combos]
            ),
        )

    def div_c_radial_derivatives(
        self, c_rr: Array, c_rth: Array, c_rz: Array, fourier_: Fourier
    ) -> Array:
        r"""`$(\partial_r c_{rr}, \partial_r c_{r\theta},
        \partial_r c_{rz})$`, one batched `$D_1$` GEMM,
        ``(3, Nr, Nm, Nkz)``."""
        return apply_y_matrix(self.D1, jnp.array([c_rr, c_rth, c_rz]))

    def tensor_abase_matvec(self, c_spin: Array, fourier_: Fourier) -> Array:
        r"""`$A_{\mathrm{base}} c = (\partial_r^2 + \tfrac1r\partial_r)c$`
        on the 6 spin slots, ``(6, Nr, Nm, Nkz)``.

        One GEMM against the **precomputed** ``AnnularFlow.A_base``
        (which the implicit bands are already built from), not a
        `$D_2$` matvec, a `$D_1$` matvec and a field-sized `$1/r$`
        multiply-add between them: `$D_1 c$` has no other consumer here
        (``div_c_radial_derivatives`` is its own, narrower stack),
        which is the premise the curvilinear fusion needs.  6
        field-GEMMs instead of 12, one fewer
        ``(6, N_r, N_m, N_{k_z})`` transient, and -- since this stack
        is component-leading -- two of the four field-sized transposes
        ``apply_y_matrix`` emits at ``component_axis = 0``.

        **It buys no measurable wall time on CPU**: an interleaved A/B
        at `$64^3$` gives -0.7 % at ``num_c = 0`` and -0.7 % again at
        ``num_c = 3-4`` (chained, so the field develops), both well
        inside an 11-25 % within-arm spread.  Two operating points
        agreeing is what makes this a wash rather than an unresolved
        measurement.  Kept on the FLOPs and the dropped transient, not
        on a timing.  The full record, the pipe twin, and why the GPU
        case does not follow:
        ``cylindrical_viscoelastic.tensor_abase_matvec``.
        """
        return apply_y_matrix(self.A_base, c_spin)

    def mean_profile_dr(self, prof: Array, spin: int) -> Array:
        r"""`$\partial_r$` of one `$m = 0$` profile, ``(Nr,)``.

        A direct matmul: no Fourier axes here, and no axis, so *spin*
        (which the pipe reads as its parity) is unused.
        """
        return self.D1 @ prof

    def add_mean_body_force(
        self, nl_z: Array, nl_r: Array, nl_th: Array, fourier_: Fourier
    ) -> tuple[Array, Array, Array]:
        """Add the azimuthal body force ``force_theta`` at the mean mode."""
        return (
            nl_z,
            nl_r,
            nl_th
            + jnp.where(
                fourier_.mean_mask, self.force_theta[:, None, None], 0.0
            ),
        )

    def zero_hc_wall_rows(self, R: Array) -> Array:
        r"""Zero the `$H_c$` RHS at both `$\nabla^2 c = 0$` wall rows."""
        return R.at[:, 0].set(0.0).at[:, -1].set(0.0)

    def hc_wall_rows(self) -> tuple[tuple[int, Array], ...]:
        r"""``((row index, narrow BC row), ...)`` for both walls.

        Indices are non-negative host ints: ``_banded_wall_row``'s
        static column arithmetic needs that form.
        """
        return (
            (0, self.hc_narrow0),
            (self.hc_narrow0.shape[0] - 1, self.hc_narrowN),
        )

    def hc_spin_bases(
        self,
        fourier_: Fourier,
        spins: tuple[int, ...],
        *,
        banded: bool,
        p: int,
    ) -> list[Array]:
        r"""The per-spin `$H_c$` base operator, aligned with *spins*.

        The annulus has no axis, so one `$A_{\mathrm{base}}$` serves
        every spin slot (only `$m_{\mathrm{eff}}^2$` differs, and the
        shared builder applies that): the band extraction runs once
        rather than per slot.  The pipe instead selects a parity band
        per slot.
        """
        base = _banded_from_dense(self.A_base, p) if banded else self.A_base
        return [base] * len(spins)

    def imm_iteration(
        self,
        u_prev: Array,
        u_pred: Array,
        rhs_prev: Array,
        rhs_next: Array,
        fourier_: Fourier,
    ) -> tuple[Array, Array, dict[str, Array]]:
        """The annular 2x2 influence-matrix velocity pass.

        Third return: the velocity pass' corrector-side *aux*
        diagnostics, passed straight through (the geometry owns
        what goes in it).
        """
        return _imm_iteration(
            u_prev, u_pred, rhs_prev, rhs_next, fourier_, self
        )

    def velocity_l_bf(self, vel: Array, fourier_: Fourier) -> Array:
        """The annular FFT-free base/mean-flow velocity coupling."""
        return _annular_l_bf(vel, fourier_, self)

    def base_dt_leaves(
        self, dt: Array, fourier_: Fourier
    ) -> dict[str, object]:
        """The annular velocity ``dt``-dependent leaves."""
        return _annular_dt_leaves(dt, fourier_, self)


# ── Stepper factory ─────────────────────────────────────────────────


def build_viscoelastic_stepper(flow: ViscoelasticAnnularFlow):
    """Build time-stepping functions for a viscoelastic annular flow.

    Binds this geometry's ``fourier`` singleton to the shared
    :func:`._viscoelastic_stepping.build_viscoelastic_stepper`, which
    documents the returned 7-tuple.
    """
    return _build_stepper(flow, fourier)
