r"""Geometry-parametrised time stepping shared by the sPTT flows.

The sPTT extension exists in two wall-bounded geometries -- the annulus
(:mod:`~dnsjax.geometries.wall_bounded.annular_viscoelastic`) and the
pipe (:mod:`~dnsjax.geometries.wall_bounded.cylindrical_viscoelastic`).
:mod:`._viscoelastic_common` holds what is *coordinate-level* identical
between them (the spin `$\leftrightarrow$` physical algebra, the
pointwise physical-space kernel, the `$\nabla\cdot c$` curvature
assembly).  This module holds the layer above it: the **stepping
functions themselves** -- the fused pseudo-spectral RHS, the FFT-free
CN/AB2 coupling, the conformation Crank-Nicolson update, the
predictor / corrector / norm, the `$H_c$` builders and the stepper
factory -- written once against a small per-geometry adapter surface.

Everything that genuinely differs between the two geometries is a
**method on the flow dataclass**, resolved once at trace time by
attribute lookup (a flow is a jit *argument*, rebuilt by
:func:`~dnsjax.sharding.register_dataclass_pytree`'s unflatten with the
concrete class), so the dispatch costs nothing at run time.  Methods
are not ``dataclasses.fields``, so they add no pytree leaf and no
memory either.

Adapter surface
---------------
Each viscoelastic flow class provides, beyond the data fields the
functions below read directly (``inv_r``, ``inv_r2``, ``inv_r_padded``,
``tensor_spin``, ``Hc_op``, ``Lk_op``, ``dt``, ``y_weights``,
``base_flow_adv_padded``, ``cfl_inv_spacing``):

===============================  ====================================
member                           what it owns
===============================  ====================================
``cfl_names``                    the CFL column labels (a
                                 ``ClassVar`` -- as an annotated
                                 *field* it would become two pytree
                                 leaves)
``rhs_radial_derivatives``       the fused 9-field `$\partial_r$`
                                 batch of :func:`_get_rhs_core`
``div_c_radial_derivatives``     the 3-field `$\partial_r$` batch of
                                 :func:`_div_c`
``tensor_abase_matvec``          `$A_{\mathrm{base}} c
                                 = (\partial_r^2 + \tfrac1r
                                 \partial_r)c$` on the 6 spin slots
``mean_profile_dr``              `$\partial_r$` of one `$m = 0$`
                                 profile
``add_mean_body_force``          the mean-mode driving `$\Pi$`
``zero_hc_wall_rows``            the `$\nabla^2 c = 0$` RHS rows
``hc_wall_rows``                 ``((row index, narrow BC row), ...)``
``hc_spin_bases``                the per-spin `$A_{\mathrm{base}}$`
                                 (parity-selected, banded or dense)
``imm_iteration``                the velocity influence-matrix pass
``velocity_l_bf``                the velocity base/mean-flow coupling
``base_dt_leaves``               the velocity ``dt``-dependent leaves
===============================  ====================================

The annulus takes plain `$D_1$`/`$D_2$` matvecs, has two walls and an
azimuthal body force; the pipe takes parity-reduced ones on the
`$(-1)^{m+s}$` bands, has one wall (the axis is closed by parity) and
an axial body force.  Each module's docstring carries its own
derivation.
"""

# Deferred annotations: the two type aliases below exist only under
# ``TYPE_CHECKING`` (importing either geometry at runtime would build
# both families' grids), and ``jax.jit`` inspects these signatures.
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from jax import Array
from jax import numpy as jnp

from ...fft import chunked_transform
from ...measurements import get_cfl
from ...operators import phys_to_spec_2d, spec_to_phys_2d
from ...parameters import derived_params, params
from ...solvers import (
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _assemble_banded_operator,
    _banded_diag_column,
    _banded_wall_row,
    _build_pallas_operator,
    _factor_pallas_operator,
)
from ._base import extract_mean_mode
from ._viscoelastic_common import (
    combined_norm,
    conformation_coupling_core,
    div_c_assemble,
    phys_combos_to_spin,
    pointwise_rhs,
    spin_to_phys_combos,
)

if TYPE_CHECKING:
    from .annular import Fourier as _AnnularFourier
    from .annular_viscoelastic import ViscoelasticAnnularFlow
    from .cylindrical import Fourier as _CylindricalFourier
    from .cylindrical_viscoelastic import ViscoelasticCylindricalFlow

    #: Either viscoelastic geometry's ``Fourier`` singleton.
    Fourier = _AnnularFourier | _CylindricalFourier
    #: Either viscoelastic flow (the adapter surface above).
    ViscoelasticFlow = ViscoelasticAnnularFlow | ViscoelasticCylindricalFlow

_WallBoundedOp = DenseJAXSolver | PerModeBandedPallasOperator

#: The five distinct `$m_{\mathrm{eff}}^2 = (m + s)^2$` operators.
_SPINS = (0, 1, -1, 2, -2)
#: The six stacked `$H_c$` slots `$(c_{zz}, c_{z+}, c_{z-}, c_{+-},
#: c_{++}, c_{--})$`; slot 3 repeats the `$s = 0$` operator.
_SPIN_ORDER = (0, 1, -1, 0, 2, -2)


# ── H_c Helmholtz operator builders (per spin component) ────────────


def _build_Hc_dense_gpu(
    A_base: Array,
    wall_rows: tuple[tuple[int, Array], ...],
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    kappa: float,
) -> Array:
    r"""Dense `$H_c = \tfrac1{\Delta t} I - c\kappa\nabla^2$` for one spin
    component (dense backend).

    Interior rows carry the diagonal Helmholtz shift on *A_base* (the
    geometry's already parity-selected base operator,
    ``flow.hc_spin_bases``); each entry of *wall_rows* overwrites its
    row with the narrow Laplacian BC row `$A_{\mathrm{base}} -
    (m_{\mathrm{eff}}^2/r^2 + k_z^2) I$`.  The annulus passes two
    (`$r_1$`, `$r_2$`), the pipe one (`$r = 1$`; its axis needs no row,
    the parity reduction closes it).
    """
    Nr = A_base.shape[-1]
    eye_Nr = jnp.eye(Nr, dtype=A_base.dtype)
    diag_coeff = 1.0 / dt + c * kappa * (meff2 * inv_r2 + kz2)  # (Nm,Nkz,Nr)
    Hc = diag_coeff[..., None] * eye_Nr - c * kappa * A_base
    for idx, narrow in wall_rows:
        # Narrow Laplacian BC (mode-dependent diagonal shift).
        shift = meff2 * inv_r2[idx] + kz2  # (Nm, Nkz, 1)
        Hc = Hc.at[..., idx, :].set(narrow[None, None] - shift * eye_Nr[idx])
    return Hc


def _build_Hc_band_gpu(
    band_base: Array,
    wall_rows: tuple[tuple[int, Array], ...],
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    kappa: float,
    p: int,
) -> Array:
    r"""Banded `$H_c$` for one spin component (Pallas backend), layout
    ``(Nm, Nkz, Nr, 2p+1)``; narrow Laplacian BC wall rows.

    *band_base* is the geometry's parity-selected base band, already
    broadcast to the operator's mode layout (``flow.hc_spin_bases``);
    *wall_rows* as in :func:`_build_Hc_dense_gpu`.
    """
    diag = 1.0 / dt + c * kappa * (meff2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    e = _banded_diag_column(p, band_base.dtype)
    walls = []
    for idx, narrow in wall_rows:
        # Narrow BC band (mode-constant) minus the mode-dependent shift.
        band_w = _banded_wall_row(narrow, idx, p)  # (2p+1,)
        shift = meff2 * inv_r2[idx] + kz2  # (Nm, Nkz, 1)
        walls.append((idx, band_w - shift * e))
    return _assemble_banded_operator(band_base, -c * kappa, diag, walls)


def _build_hc_operator(
    dt: float | Array,
    fourier_: Fourier,
    flow_: ViscoelasticFlow,
    *,
    label: str | None,
) -> _WallBoundedOp:
    r"""Factored 6-component `$H_c$` at *dt*.

    Five distinct operators (`$m_{\mathrm{eff}}^2 = (m+s)^2$` for
    `$s = 0, \pm1, \pm2$`, on the pipe additionally on the
    `$s \bmod 2$` parity band) are built and stacked into the
    6-component order `$(c_{zz}, c_{z+}, c_{z-}, c_{+-}, c_{++},
    c_{--})$`, so the `$s = 0$` operator serves both `$c_{zz}$` and
    `$c_{+-}$`.

    The stacked storage **duplicates** that shared operator's factors
    (slot 0 and slot 3 hold the same data -- ~1/6 of the ``Hc_op``
    memory), because the uniform stacked ``.solve`` contract pairs
    component ``i`` of the RHS with operator ``i``.  Deduplicating
    would need a nonuniform component-to-operator solve mapping (5
    operators against 6 RHS components) in every backend -- deferred
    as not worth the contract complexity for a small,
    setup-persistent array (the velocity ``Hk_op`` stack and the
    per-step transform transients are far larger).

    *label* selects the pallas factorization path: a string runs the
    setup-checked :func:`solvers._build_pallas_operator` under that
    diagnostic label; ``None`` runs the unchecked, jittable
    :func:`solvers._factor_pallas_operator` (the ``set_dt`` rebuild).
    The dense backend is pivoted and ignores *label*.  Wall rows come
    from the geometry's stored narrow-BC leaves
    (``flow.hc_wall_rows``).
    """
    kappa = params.phys.kappa
    c_impl = params.step.implicitness
    m_s = fourier_.m[0, ..., None]  # (Nm, 1, 1)
    kz2_s = fourier_.kz2[0, ..., None]  # (1, Nkz, 1)
    meff2 = {s: (m_s + s) ** 2 for s in _SPINS}
    walls = flow_.hc_wall_rows()

    if params.solver.backend == "pallas":
        # Half-width read back from the already-factored, dt-independent
        # Lk, exactly as each geometry's ``_hk_bands`` does: a static
        # shape, so it works inside the jitted ``set_dt`` rebuild, and
        # it is *measured* rather than assumed to be ``fd_order`` -- an
        # under-sized band truncates entries silently
        # (``fd.matrix_half_bandwidth``).  The dense backend's
        # ``Lk_op`` carries no band, hence the read sits here.
        p = flow_.Lk_op.L.shape[1]
        bands = [
            _build_Hc_band_gpu(
                base,
                walls,
                meff2[s],
                flow_.inv_r2,
                kz2_s,
                dt,
                c_impl,
                kappa,
                p,
            )
            for s, base in zip(
                _SPIN_ORDER,
                flow_.hc_spin_bases(fourier_, _SPIN_ORDER, banded=True, p=p),
                strict=True,
            )
        ]
        if label is not None:
            return _build_pallas_operator(bands, label)
        return _factor_pallas_operator(bands)

    solvers_by_spin = {
        s: DenseJAXSolver(
            _build_Hc_dense_gpu(
                base, walls, meff2[s], flow_.inv_r2, kz2_s, dt, c_impl, kappa
            )
        )
        for s, base in zip(
            _SPINS,
            flow_.hc_spin_bases(fourier_, _SPINS, banded=False, p=0),
            strict=True,
        )
    }
    return DenseJAXSolver.from_factors(
        lu=jnp.stack([solvers_by_spin[s].lu for s in _SPIN_ORDER]),
        perm=jnp.stack([solvers_by_spin[s].perm for s in _SPIN_ORDER]),
    )


def _build_dt_leaves(
    dt: Array,
    fourier_: Fourier,
    flow_: ViscoelasticFlow,
) -> dict[str, object]:
    r"""Rebuild every ``dt``-dependent flow leaf at the traced *dt*.

    The geometry's velocity set (`$H_k$` group + IMM leaves, from its
    ``_build_dt_leaves`` via ``flow.base_dt_leaves``, with the solvent
    `$\nu = \beta/\mathrm{Re}$` through ``derived_params.nu``) plus the
    conformation `$H_c$` (unchecked factorization,
    :func:`_build_hc_operator`) when diffusion is active.  At
    `$\kappa = 0$` ``Hc_op`` is ``None`` (static aux) and stays out of
    the rebuild -- the trace-time branch matches construction.
    """
    leaves = flow_.base_dt_leaves(dt, fourier_)
    if flow_.Hc_op is not None:
        leaves["Hc_op"] = _build_hc_operator(dt, fourier_, flow_, label=None)
    return leaves


# ── Spectral tensor operators (FFT-free) ────────────────────────────


def _tensor_laplacian_spin(
    c_spin: Array, fourier_: Fourier, flow_: ViscoelasticFlow
) -> Array:
    r"""Spin-diagonal tensor Laplacian, `$(6, N_r, N_m, N_{kz})$`.

    `$(\nabla^2 c)_{\text{spin }s} = A_{\mathrm{base}} c
    - (m_{\mathrm{eff}}^2/r^2 + k_z^2) c$` with
    `$m_{\mathrm{eff}} = m + s$` per spin component; the radial part
    `$A_{\mathrm{base}} = \partial_r^2 + \tfrac1r\partial_r$` is the
    geometry's (``flow.tensor_abase_matvec`` -- plain FD on the
    annulus, parity-reduced per spin slot on the pipe).
    """
    Abase_c = flow_.tensor_abase_matvec(c_spin, fourier_)
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
    flow_: ViscoelasticFlow,
) -> tuple[Array, Array, Array]:
    r"""Spectral divergence of the symmetric tensor (FFT-free).

    One batched geometry `$D_1$` GEMM for the radial derivatives of
    `$(c_{rr}, c_{r\theta}, c_{rz})$` (``flow.div_c_radial_derivatives``
    -- on the pipe a parity-reduced pair, `$c_{rr}$` and
    `$c_{r\theta}$` in the `$(-1)^m$` class, `$c_{rz}$` in the
    `$(-1)^{m+1}$` one), then the shared curvature assembly
    (:func:`._viscoelastic_common.div_c_assemble`, which carries the
    component formulas).  The result lands in the classes the velocity
    sources need.
    """
    dr = flow_.div_c_radial_derivatives(c_rr, c_rth, c_rz, fourier_)
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


# ── Fused pseudo-spectral RHS ───────────────────────────────────────


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: ViscoelasticFlow,
    measure_fn: Callable[[Array, Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    r"""Evaluate the full 9-component nonlinear RHS ``rhs_no_lapl``.

    One batched inverse transform of ~36 fields (velocity, velocity
    gradient `$L_{ij}$`, physical tensor, and its 18 advection
    derivatives), the shared pointwise physical-space stage
    (:func:`._viscoelastic_common.pointwise_rhs`), one batched forward
    transform of the 9 outputs.  The viscous / diffusive Laplacians are
    added implicitly by the predictor/corrector, so they are absent
    here.  See the two geometry module docstrings.

    The nine radial derivatives (3 velocity + 6 conformation combos)
    are one batched GEMM (``flow.rhs_radial_derivatives``; a
    parity-reduced GEMM *pair* on the pipe), and at the default
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
    # the spectral tensor combos, distinct from the physical crr.. that
    # the pointwise stage sees).
    cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz = spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    combos = jnp.array([cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz])

    # Single batched D1 GEMM over the 3 radial velocity derivatives
    # (velocity gradient L_ij = d_i u_j) and the 6 radial conformation
    # advection derivatives -- one GEMM (pair) instead of two.
    dr_all = flow_.rhs_radial_derivatives(
        (u_r, u_th, u_z, cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz),
        combos,
        fourier_,
    )  # (9, Nr, Nm, Nkz)
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
    # Mean-mode body force (azimuthal on the annulus, axial in the pipe).
    NL_z, NL_r, NL_th = flow_.add_mean_body_force(
        out_spec[0], out_spec[1], out_spec[2], fourier_
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
    state: Array, fourier_: Fourier, flow_: ViscoelasticFlow
) -> Array:
    """Evaluate the 9-component nonlinear RHS."""
    return _get_rhs_core(state, fourier_, flow_, None)


def _get_rhs_measured(
    state: Array, fourier_: Fourier, flow_: ViscoelasticFlow
) -> tuple[Array, dict[str, Array]]:
    """Evaluate the RHS + CFL / max-tr(c) measurements."""

    def _measure(
        u_phys: Array, om_phys: Array, trc: Array
    ) -> dict[str, Array]:
        meas = get_cfl(
            u_phys,
            flow_.base_flow_adv_padded,
            flow_.cfl_inv_spacing,
            flow_.cfl_names,
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
    flow_: ViscoelasticFlow,
) -> Array:
    r"""FFT-free linear/mean conformation coupling, 6 spin components.

    Binds :func:`._viscoelastic_common.conformation_coupling_core`
    (which carries the term-by-term account of what the CN/AB2 scheme
    makes implicit here): this supplies the instantaneous mean velocity
    profile and its radial gradients -- the one geometry-dependent step
    (``flow.mean_profile_dr``: a plain `$D_1$` matmul on the annulus,
    the parity-reduced one at the `$m = 0$` constant signs `$(-1)^s$`
    in the pipe) -- plus the always-implicit moving-frame convective
    term.
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
        # profiles, at the spin weight of each (0 for u_z, 1 for
        # u_theta -- only the pipe reads it, as its axis parity).
        d_uz = flow_.mean_profile_dr(mean_vel[0], 0)
        d_uth = flow_.mean_profile_dr(mean_vel[2], 1)
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


def _l_bf(state: Array, fourier_: Fourier, flow_: ViscoelasticFlow) -> Array:
    r"""FFT-free linear coupling for the CN/AB2 scheme, all 9 components.

    Velocity slice: the geometry's base/mean-flow coupling
    (``flow.velocity_l_bf``, including the moving-frame term) plus the
    **polymer-stress divergence**
    `$\tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}\nabla\cdot\mathbf{c}$`
    (the elastic velocity`$\leftrightarrow$`conformation coupling,
    linear in `$c$` and FFT-free).  Conformation slice:
    :func:`_conformation_coupling`.

    ``step_cnab2`` advances the explicit remainder
    `$\text{get\_rhs} - \text{\_l\_bf}$` (pure fluctuation-fluctuation
    advection / stretching + nonlinear relaxation + the constant body
    force) with AB2 and makes this coupling implicit through the
    FFT-free corrector.  For these total-field flows the mean coupling
    (velocity *and* the large mean conformation profile) is the
    dominant stiffness, exactly as the mean-flow coupling is for
    Newtonian Dean.
    """
    vel_lbf = flow_.velocity_l_bf(state[:3], fourier_)

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
    flow_: ViscoelasticFlow,
) -> Array:
    r"""Crank-Nicolson conformation update (6 spin components).

    Solves `$H_c c^{new} = \tfrac1{\Delta t} c^n + (1-\theta)\kappa
    \nabla^2 c^n + \theta N_c^j + (1-\theta) N_c^n$` with the wall-row
    RHS zeroed (the `$\nabla^2 c = 0$` BC; ``flow.zero_hc_wall_rows``
    -- two rows on the annulus, one in the pipe, whose axis carries no
    row).  With `$\kappa = 0$` there is no diffusion / wall BC and the
    update degenerates to `$c^{new} = c^n + \Delta t(\theta N_c^j +
    (1-\theta) N_c^n)$`.
    """
    dt = flow_.dt
    c_impl = params.step.implicitness
    nl = c_impl * Nc_j + (1.0 - c_impl) * Nc_n
    if flow_.Hc_op is None:  # kappa == 0 (trace-time branch)
        return c_n + dt * nl
    kappa = params.phys.kappa
    lap_cn = _tensor_laplacian_spin(c_n, fourier_, flow_)
    R = (1.0 / dt) * c_n + (1.0 - c_impl) * kappa * lap_cn + nl
    R = flow_.zero_hc_wall_rows(R)
    return flow_.Hc_op.solve(R)


# ── Predictor / corrector / norm ────────────────────────────────────


def _correct(
    state_prev: Array,
    prediction: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: ViscoelasticFlow,
) -> tuple[Array, Array]:
    """Coupled velocity-IMM + conformation-CN corrector.

    Velocity: the geometry's influence-matrix iteration
    (``flow.imm_iteration``, which sees the polymer divergence only
    through the sources, so it needs no viscoelastic knowledge).
    Conformation: the Crank-Nicolson Helmholtz update.  The returned
    correction stacks both so the single convergence norm covers `$u$`
    and `$c$`.
    """
    vel_new, vel_corr = flow_.imm_iteration(
        state_prev[:3],
        prediction[:3],
        rhs_prev[:3],
        rhs_next[:3],
        fourier_,
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
    flow_: ViscoelasticFlow,
) -> Array:
    """Euler predictor (nonlinear at `$u^n$`, viscous/diffusive CN)."""
    prediction, _ = _correct(
        state_n, state_n, rhs_no_lapl, rhs_no_lapl, fourier_, flow_
    )
    return prediction


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: ViscoelasticFlow,
) -> Array:
    r"""Combined L2 convergence norm, `$\sqrt{\|u\|^2 + \|c\|_F^2}$`
    (:func:`._viscoelastic_common.combined_norm`)."""
    return combined_norm(correction, fourier_.k_metric, flow_.y_weights)


# ── Stepper factory ─────────────────────────────────────────────────


def build_viscoelastic_stepper(flow: ViscoelasticFlow, fourier: Fourier):
    """Build time-stepping functions for a viscoelastic flow.

    Returns the same 9-tuple as
    :func:`~dnsjax.geometries.wall_bounded._base.build_wall_bounded_stepper`
    (incl. the adaptive-dt ``set_dt`` / ``reset_ab2_kappa``, backed by
    this module's ``_build_dt_leaves``).  :func:`_l_bf` (the FFT-free
    linear/mean coupling: velocity mean-flow coupling + polymer-stress
    divergence, conformation mean advection / stretching / linear
    relaxation) is passed so the CN/AB2 scheme treats it implicitly and
    the explicit AB2 remainder stays pure fluctuation-fluctuation
    nonlinearity.

    *fourier* is the geometry's own singleton: this module never
    imports either geometry (that would build both families' grids on
    any viscoelastic import), so each geometry module passes its own
    through its one-line ``build_viscoelastic_stepper``.
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
