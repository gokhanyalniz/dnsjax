r"""Viscoelastic (sPTT) extension of the annular geometry.

Adds a symmetric conformation tensor `$\mathbf{c}$` to the annular
(concentric-cylinder) geometry, coupling it to the velocity via the
polymer-stress divergence.  The time-integrated state grows from the 3
velocity components to **9** (3 velocity + 6 independent
symmetric-tensor components), one stacked spectral array
``(9, Nr, Nm, Nkz)``: in the solver, the decoupled `$u_\pm$` velocity
of :mod:`~dnsjax.geometries.wall_bounded.annular` and the analogous
tensor spin projections; everywhere else, the physical components.
Both layouts, the crossing between them and the spin weights:
:mod:`._viscoelastic_common` ("State layout", "Spin weights").

What lives where
----------------
The time stepper itself is shared with the pipe's sPTT geometry and
lives in :mod:`._viscoelastic_stepping`, the coordinate-level algebra
under it in :mod:`._viscoelastic_common`.  **This** module owns the
annular half: the flow dataclass, the analytical laminar profiles, the
narrow Laplacian BC wall rows, and the adapter methods the shared
stepper dispatches on (plain `$D_1$`/`$D_2$` radial derivatives, two
wall rows, the azimuthal body force -- see
:class:`ViscoelasticAnnularFlow`).

Spin diagonalization of the tensor Laplacian
--------------------------------------------
Each spin component diffuses through a scalar Helmholtz solve with its
own `$m_{\mathrm{eff}} = m + s$` (:mod:`._viscoelastic_common`,
"Spin weights"), reusing the annular dense/Pallas machinery; the
annulus has no axis, so no parity enters.

Governing equations (sPTT)
--------------------------
The system of :mod:`._viscoelastic_common` ("The model") with
`$\mathbf{f} = -\boldsymbol{\Pi}$`, no-slip `$\mathbf{u}=0$` and
`$\nabla^2\mathbf{c}=0$` at both walls, and the azimuthal body force
`$-\Pi_\theta = (r_1+r_2)/(\mathrm{Re}\,r)$`
(:mod:`~dnsjax_lab.flows.wall_bounded.viscoelastic_dean`).

Time integration
----------------
The shared stepper (:mod:`._viscoelastic_stepping`, both schemes) with
the velocity on the annular `$2\times2$` IMM
(:func:`~dnsjax.geometries.wall_bounded.annular._imm_iteration`,
solvent viscosity `$\nu = \beta/\mathrm{Re}$`) and the conformation's
Crank-Nicolson Helmholtz solve carrying a Laplacian BC row at each
wall.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from dnsjax.geometries.wall_bounded._base import (
    apply_y_matrix,
)
from dnsjax.geometries.wall_bounded.annular import (
    CFL_NAMES,
    AnnularFlow,
    Fourier,
    _imm_iteration,
    fourier,
)
from dnsjax.geometries.wall_bounded.annular import (
    _build_dt_leaves as _annular_dt_leaves,
)
from dnsjax.geometries.wall_bounded.annular import (
    _l_bf as _annular_l_bf,
)
from dnsjax.parameters import params
from dnsjax.sharding import register_dataclass_pytree, sharding
from dnsjax.solvers import (
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _banded_from_dense,
)
from dnsjax_lab.geometries.wall_bounded._viscoelastic_common import (
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
from dnsjax_lab.geometries.wall_bounded._viscoelastic_stepping import (
    _build_Hc_band_gpu,  # noqa: F401
    _build_Hc_dense_gpu,  # noqa: F401
    _build_hc_operator,
    _div_c,  # noqa: F401
    _get_rhs,  # noqa: F401
    _l_bf,  # noqa: F401
    _tensor_laplacian_spin,  # noqa: F401
)
from dnsjax_lab.geometries.wall_bounded._viscoelastic_stepping import (
    build_viscoelastic_stepper as _build_stepper,
)

# ── State layout ────────────────────────────────────────────────────
#
# The layout, the spin weights, the basis pair and the pointwise
# physical-space arithmetic are geometry-free and shared with the
# pipe's viscoelastic geometry: :mod:`._viscoelastic_common`.

#: Role aliases for the basis boundary (see ``cylindrical.py``).
to_solver_basis = to_spin_basis
from_solver_basis = jax.jit(from_spin_basis)


# ── Analytical laminar profiles (build-time) ──────────────────────


def viscoelastic_laminar_profiles(
    rs: np.ndarray, D1: np.ndarray, r1: float, r2: float, wi: float, eps: float
) -> np.ndarray:
    r"""9-component laminar `$r$`-profiles of a force-driven annular sPTT flow.

    Complex ``(9, Nr)``, in the **physical** state layout
    `$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` -- these feed initial conditions
    and the flow's laminar reference, both of which live outside the
    solver (:func:`to_spin_basis` converts when one enters it).

    Velocity: the azimuthal profile `$U_\theta(r)$` (body-force
    coefficient `$C = r_1 + r_2$`).  Conformation: the pointwise sPTT
    equilibrium on the **discrete** local shear `$S = D_1 U_\theta -
    U_\theta/r$` (see the
    :mod:`~dnsjax_lab.flows.wall_bounded.viscoelastic_dean` module
    docstring): `$c_{rr} = c_{zz} = 1$`, `$c_{r\theta} =
    \mathrm{Wi}\,S/f$`, `$c_{\theta\theta} = 1 + 2 c_{r\theta}^2$`.
    A build-time host computation returning NumPy (the shared
    `$U_\theta$` formula runs in JAX); shared by the flow's laminar state
    and the viscoelastic random / rolls ICs.
    """
    from dnsjax_lab.geometries.wall_bounded._annular_forced import (
        annular_forced_laminar_u_theta,
    )

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
    (built with the solvent viscosity `$\nu = \beta/\mathrm{Re}$` via
    ``derived_params.nu``) with the conformation-tensor machinery: the
    stacked Crank-Nicolson Helmholtz operator ``Hc_op`` (6 spin
    components, `$m_{\mathrm{eff}} = m + s$`; two share
    `$m_{\mathrm{eff}} = m$`) and the physical `$1/r$` profile on the
    padded grid.  When `$\kappa = 0$` no Helmholtz operator is built
    (``Hc_op = None``): the conformation transport is hyperbolic and the
    update is the explicit CN combination.

    The methods below are the **annular half of the adapter surface** of
    :mod:`._viscoelastic_stepping`: plain `$D_1$`/`$D_2$` radial
    derivatives (no axis, so no parity anywhere), two
    `$\nabla^2 c = 0$` wall rows, and an azimuthal mean-mode body force.

    Subclasses (:mod:`~dnsjax_lab.flows.wall_bounded.viscoelastic_dean`)
    set ``force_theta`` and zero the base flow (total-field integration).
    """

    #: CFL column labels (a ``ClassVar``; why: the adapter table of
    #: :mod:`._viscoelastic_stepping`).
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
        self.inv_r_padded = sharding.distribute(
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

        Stores the narrow Laplacian BC wall rows as the ``hc_narrow0`` /
        ``hc_narrowN`` leaves (their field comment says why), optionally
        pre-checks the no-pivot LU at ``dt_max`` (``step.adaptive``; the
        velocity `$H_k$` analogue), and delegates the
        assembly/factorization to
        :func:`._viscoelastic_stepping._build_hc_operator`.
        """
        # Full narrow Laplacian BC wall rows (JAX-free build).
        row0_np, rowN_np = _narrow_abase_wall_rows(
            np.asarray(self.rs), np.asarray(self.D1.dense), params.res.fd_order
        )
        self.hc_narrow0 = sharding.distribute(row0_np, sharding.no_shard)
        self.hc_narrowN = sharding.distribute(rowN_np, sharding.no_shard)

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
        *combos* the already-materialized `$(6, \ldots)$` tensor batch:
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
        r"""Radial derivatives of `$c_{rr}$`, `$c_{r\theta}$`, `$c_{rz}$`.

        One batched `$D_1$` GEMM, ``(3, Nr, Nm, Nkz)``.
        """
        return apply_y_matrix(self.D1, jnp.array([c_rr, c_rth, c_rz]))

    def tensor_abase_matvec(self, c_spin: Array, fourier_: Fourier) -> Array:
        r"""`$A_{\mathrm{base}} c$` on the 6 spin slots, ``(6, Nr, Nm, Nkz)``.

        `$A_{\mathrm{base}} c = (\partial_r^2 + \tfrac1r\partial_r)c$`,
        one GEMM against the precomputed ``AnnularFlow.A_base`` (which
        the implicit bands are already built from): `$D_1 c$` has no
        other consumer here (``div_c_radial_derivatives`` is its own,
        narrower stack), the premise of
        :func:`~dnsjax.geometries.wall_bounded.cylindrical._build_A_base`.
        6 field-GEMMs instead of 12, one fewer
        ``(6, N_r, N_m, N_{k_z})`` transient, and -- this stack being
        component-leading -- two of the four field-sized transposes
        ``apply_y_matrix`` emits at ``component_axis = 0``.  Kept on
        those grounds, not on a timing: it buys no measurable wall time
        on CPU (the A/B, which covers this flow too: the
        :mod:`~dnsjax_lab.geometries.wall_bounded.cylindrical_viscoelastic`
        Design notes, "Fusing `$A_{\mathrm{base}}$` on the tensor slots").
        """
        return apply_y_matrix(self.A_base, c_spin)

    def mean_profile_dr(self, prof: Array, spin: int) -> Array:
        r"""`$\partial_r$` of one `$m = 0$` profile, ``(Nr,)``.

        A direct matmul: no Fourier axes here, and no axis, so *spin*
        (which the pipe reads as its parity) is unused.
        """
        return self.D1.dense @ prof

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
        a_base = self.A_base.dense
        base = _banded_from_dense(a_base, p) if banded else a_base
        return [base] * len(spins)

    def imm_iteration(
        self,
        u_prev: Array,
        u_pred: Array,
        rhs_prev: Array,
        rhs_next: Array,
        fourier_: Fourier,
        carried: Array | None = None,
    ) -> tuple[Array, Array, dict[str, Array], Array | None]:
        """The annular 2x2 influence-matrix velocity pass.

        Third return: the velocity pass' corrector-side *aux*
        diagnostics, passed straight through (the geometry owns
        what goes in it); fourth, ``None`` -- the annular pass carries
        no slots (``AnnularFlow.n_carried``), and *carried* is always
        ``None`` here.
        """
        return (
            *_imm_iteration(
                u_prev, u_pred, rhs_prev, rhs_next, fourier_, self
            ),
            None,
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
