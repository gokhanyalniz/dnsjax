r"""Legacy primitive `$(u_\pm, p)$` influence-matrix path, annular.

``res.consistent_imm`` is **on by default**; the annular implicit step
is then the `$u_r$`-`$\omega_r$` reformulation in :mod:`.annular`
(:func:`~dnsjax.geometries.wall_bounded.annular._imm_iteration_vw`),
which never forms a pressure.  Setting the flag to ``False`` selects the
**legacy** scheme kept here: the primitive Kleiser-Schumann
influence-matrix method, which solves `$(u_z, u_+, u_-)$` against a
pressure Poisson solve and enforces continuity only at the two walls.
It is retained for reference and for reproducing older trajectories; a
state it steps carries an `$O(1)$` *relative* discrete divergence.  The
full comparison and the measured ledger: the
``Resolution.consistent_imm`` docs (``parameters.py``) and the shared
scheme record on
:func:`~dnsjax.geometries.wall_bounded.cartesian._imm_iteration`.

Everything here is reachable **only** when the flag is off, so
``annular.py`` imports this module lazily inside its flag-off branches
and the default path never imports it at all.  The dependency runs the
other way at module scope -- this module imports the shared `$H_k$`
builders and types from ``annular.py`` -- which is why the import must
be deferred there rather than declared at the top of that file.

Contents: the Neumann-BC pressure Poisson operator in both storage
backends (:func:`_build_Lk_band_gpu`, :func:`_build_Lk_dense_gpu`), the
three-family `$H_k$` group `$(+, -, z)$` this scheme needs
(:func:`_hk_bands`, :func:`_hk_dense_op` -- the default path builds the
two-family spin pair instead), the matrix-free applies
(:func:`_abase_matvec`, :func:`_lk_matvec`), the homogeneous-column /
influence-matrix derivation (:func:`derive_homogeneous_data`), and the
step itself (:func:`_imm_iteration_vp`).
"""

from jax import Array
from jax import numpy as jnp

from ...parameters import derived_params, params
from ...sharding import sharding
from ...solvers import (
    DenseJAXSolver,
    _assemble_banded_operator,
    _banded_diag_column,
    _banded_from_dense,
    _banded_wall_row,
)
from ._base import apply_y_matrix, extract_mean_mode, integrate_scalar
from .annular import (
    AnnularFlow,
    Fourier,
    _build_Hk_band_gpu,
    _build_Hk_dense_gpu,
)

# ── Pressure Poisson operator (Neumann BCs) ─────────────────


def _build_Lk_band_gpu(
    D1: Array,
    A_base: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
    p: int,
) -> Array:
    r"""Build `$L_k$` in banded storage for the Pallas backend.

    Same operator as :func:`_build_Lk_dense_gpu`
    (`$L_k = A_{\mathrm{base}} - (m^2/r^2 + k_z^2) I$`), assembled
    directly in banded layout ``(Nm, Nkz, Nr, 2p+1)`` from the base
    band ``_banded_from_dense(A_base, p)``, with no ``(Nr, Nr)`` per
    mode.  The two-wall row-setting mirrors the Cartesian builder:
    Neumann `$D_1$` rows at the inner (row 0) and outer (row Nr-1)
    walls, with a mean-mode identity pin at the outer wall.  No parity
    selection (single `$A_{\mathrm{base}}$`).
    """
    Nr = A_base.shape[0]
    band_base = _banded_from_dense(A_base, p)  # (Nr, 2p+1)
    diag = -(m2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    inner = _banded_wall_row(D1[0], 0, p)  # Neumann, inner wall
    neumann_outer = _banded_wall_row(D1[-1], Nr - 1, p)  # Neumann, outer
    outer = jnp.where(
        mean_mask, _banded_diag_column(p, band_base.dtype), neumann_outer
    )  # (Nm, Nkz, 2p+1)
    return _assemble_banded_operator(
        band_base, 1.0, diag, [(0, inner), (Nr - 1, outer)]
    )


def _build_Lk_dense_gpu(
    D1: Array,
    A_base: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
) -> Array:
    r"""Build dense `$L_k$` on GPU (dense backend only)."""
    Nr = A_base.shape[0]
    dtype = A_base.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)

    diag_shift = -(m2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    Lk = A_base[None, None] + diag_shift[..., None] * eye_Nr

    Lk = Lk.at[..., 0, :].set(D1[0, :])  # Neumann inner
    pin = eye_Nr[-1, :]
    rowN = jnp.where(mean_mask, pin, D1[-1, :])  # Neumann outer / pin mean
    Lk = Lk.at[..., -1, :].set(rowN)
    return Lk


# ── Three-family Helmholtz group (+, -, z) ───────────────────


def _hk_bands(
    dt: float | Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> list[Array]:
    r"""Assemble the banded `$H_k$` group (+, -, z) at *dt*.

    Single-sources the band assembly for the setup-checked build, the
    adaptive ``dt_max`` stability pre-check, and the jitted ``set_dt``
    rebuild (:func:`_build_dt_leaves`).  Pallas backend only.

    The half-width is read back from the already-factored (and
    ``dt``-independent) `$L_k$`, whose ``L`` factor is
    ``(Nr, p, Nm, Nkz)`` -- a static shape, so this works inside
    ``jit`` where a host-side measurement on the traced ``A_base``
    could not.
    """
    m_s = fourier_.m[0, ..., None]
    kz2_s = fourier_.kz2[0, ..., None]
    # Solvent viscosity ``derived_params.nu``: 1/re for Newtonian
    # Taylor-Couette / Dean, beta/re for the viscoelastic subclass.
    return [
        _build_Hk_band_gpu(
            flow_.A_base,
            meff2,
            flow_.inv_r2,
            kz2_s,
            dt,
            params.step.implicitness,
            derived_params.nu,
            flow_.Lk_op.L.shape[1],
        )
        for meff2 in ((m_s + 1) ** 2, (m_s - 1) ** 2, m_s**2)
    ]


def _hk_dense_op(
    dt: float | Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> DenseJAXSolver:
    r"""Factored dense stacked `$H_k$` (+, -, z) at *dt* (dense
    backend)."""
    m_s = fourier_.m[0, ..., None]
    kz2_s = fourier_.kz2[0, ..., None]
    ops = [
        DenseJAXSolver(
            _build_Hk_dense_gpu(
                flow_.A_base,
                meff2,
                flow_.inv_r2,
                kz2_s,
                dt,
                params.step.implicitness,
                derived_params.nu,
            )
        )
        for meff2 in ((m_s + 1) ** 2, (m_s - 1) ** 2, m_s**2)
    ]
    return DenseJAXSolver.from_factors(
        lu=jnp.stack([o.lu for o in ops]),
        perm=jnp.stack([o.perm for o in ops]),
    )


# ── Matrix-free matvecs ────────────────────────────────


def _abase_matvec(u: Array, flow_: AnnularFlow) -> Array:
    r"""Apply `$A_{\mathrm{base}} u = (D_2 + (1/r) D_1)\,u$`.

    One GEMM against the precomputed operator (``AnnularFlow.A_base``,
    which the implicit bands are already built from), not a `$D_2$` and
    a `$D_1$` with a field-sized `$1/r$` multiply-add between them.
    """
    return apply_y_matrix(flow_.A_base, u)


def _lk_matvec(
    u: Array,
    flow_: AnnularFlow,
    fourier_: Fourier,
) -> Array:
    r"""Apply `$L_k u = A_{\mathrm{base}} u - (m^2/r^2 + k_z^2) u$`.

    Neumann wall rows at both walls; the outer-wall row pins
    `$p_{N_r-1}$` at the mean mode (the only `$k^2 = 0$` system).
    """
    Abase_u = _abase_matvec(u, flow_)
    inv_r2 = flow_.inv_r2[:, None, None]
    out = Abase_u - (fourier_.m2 * inv_r2 + fourier_.kz2) * u

    inner = jnp.einsum("j, jmz -> mz", flow_.D1_bnd[0], u)
    outer_neumann = jnp.einsum("j, jmz -> mz", flow_.D1_bnd[1], u)
    outer = jnp.where(fourier_.mean_mask[0], u[-1], outer_neumann)
    return out.at[0].set(inner).at[-1].set(outer)


# ── Homogeneous columns and influence matrix ────────────────


def derive_homogeneous_data(
    flow_: AnnularFlow, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
) -> None:
    r"""Fill the homogeneous responses and the `$2 \times 2$` ``M_inv``.

    The legacy half of
    :meth:`~dnsjax.geometries.wall_bounded.annular.AnnularFlow._derive_imm_homogeneous_data`
    (the dispatcher, which calls this when ``res.consistent_imm`` is
    off).

    Two unit-wall pressures (`$L_k p_i = e_i$`, `$e_1$` at the inner
    wall, `$e_2$` at the outer) give, via the pressure gradient and
    the Helmholtz solves, the `$u_\pm$` responses ``v_plus_i``,
    ``v_minus_i`` and the axial potentials ``q_z_i``.  The `$u_r$`
    part is zeroed at the mean mode (continuity forces `$u_r \equiv
    0$` there).  The influence matrix
    `$M_{ji} = D_{1,\mathrm{wall}_j} \cdot (v_{+,i} + v_{-,i})/2$`
    is `$2 \times 2$`; ``M_inv`` is its inverse, set to zero at the
    mean mode (where `$d_{\mathrm{wall}} = 0$`, so the correction
    vanishes regardless).
    """
    # This run-once setup stays in the mode-outer (Nm, Nkz, Nr)
    # layout: the influence-matrix einsums below operate on it and
    # the results are transposed to field layout (Nr, Nm, Nkz) at
    # the end.  ``.solve`` now takes a mode-inner field, so each
    # setup solve is wrapped (transpose in, transpose out) to keep
    # this layout.  FUTURE: rebuild this setup natively mode-inner to
    # drop the wrappers -- the hot path already is; here it only
    # relocates a one-time transpose, so it is deferred.
    e_inner = (
        jnp.zeros(
            (Nm, Nkz, Nr),
            dtype=sharding.float_type,
            out_sharding=sharding.spec_imm_corr_shard,
        )
        .at[..., 0]
        .set(1.0)
    )
    e_outer = (
        jnp.zeros(
            (Nm, Nkz, Nr),
            dtype=sharding.float_type,
            out_sharding=sharding.spec_imm_corr_shard,
        )
        .at[..., -1]
        .set(1.0)
    )
    p1_s = flow_.Lk_op.solve(e_inner.transpose(2, 0, 1)).transpose(1, 2, 0)
    p2_s = flow_.Lk_op.solve(e_outer.transpose(2, 0, 1)).transpose(1, 2, 0)

    m_s = fourier_.m[0, ..., None]  # (Nm, 1, 1)
    m_over_r_s = m_s * flow_.inv_r  # (Nm, 1, Nr)
    mean_s = fourier_.mean_mask[0, ..., None]  # (Nm, Nkz, 1)

    def _helm_responses(p_s: Array) -> tuple[Array, Array, Array]:
        D1_p = jnp.einsum("ij, mzj -> mzi", flow_.D1, p_s)
        rhs_v_plus = -(D1_p - m_over_r_s * p_s)
        rhs_v_minus = -(D1_p + m_over_r_s * p_s)
        rhs_v_plus = rhs_v_plus.at[..., 0].set(0.0).at[..., -1].set(0.0)
        rhs_v_minus = rhs_v_minus.at[..., 0].set(0.0).at[..., -1].set(0.0)
        q_rhs = p_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
        stacked = jnp.stack([rhs_v_plus, rhs_v_minus, q_rhs])
        res = flow_.Hk_op.solve(stacked.transpose(0, 3, 1, 2)).transpose(
            0, 2, 3, 1
        )
        vp, vm = res[0], res[1]
        # Zero the u_r part at the mean mode, preserving u_theta.
        vr = jnp.where(mean_s, (vp + vm) / 2, 0.0)
        return vp - vr, vm - vr, res[2]

    vp1, vm1, qz1 = _helm_responses(p1_s)
    vp2, vm2, qz2 = _helm_responses(p2_s)

    # 2x2 influence matrix M[j, i] = D1_bnd[j] . u_r^(i).
    ur1 = (vp1 + vm1) / 2
    ur2 = (vp2 + vm2) / 2
    M00 = jnp.einsum("j, mzj -> mz", flow_.D1_bnd[0], ur1)
    M01 = jnp.einsum("j, mzj -> mz", flow_.D1_bnd[0], ur2)
    M10 = jnp.einsum("j, mzj -> mz", flow_.D1_bnd[1], ur1)
    M11 = jnp.einsum("j, mzj -> mz", flow_.D1_bnd[1], ur2)

    is_mean = fourier_.mean_mask[0]  # (Nm, Nkz)
    det = M00 * M11 - M01 * M10
    safe_det = jnp.where(is_mean, 1.0, det)
    # Mean mode: u_r is zeroed and d_wall = 0 there, so the
    # correction vanishes; M_inv = 0 keeps it NaN-free.
    inv_00 = jnp.where(is_mean, 0.0, M11 / safe_det)
    inv_01 = jnp.where(is_mean, 0.0, -M01 / safe_det)
    inv_10 = jnp.where(is_mean, 0.0, -M10 / safe_det)
    inv_11 = jnp.where(is_mean, 0.0, M00 / safe_det)
    flow_.M_inv = jnp.stack(
        [
            jnp.stack([inv_00, inv_01], axis=-1),
            jnp.stack([inv_10, inv_11], axis=-1),
        ],
        axis=-2,
    )

    # Transpose to field layout (Nr, Nm, Nkz).
    flow_.v_plus_1 = vp1.transpose(2, 0, 1)
    flow_.v_minus_1 = vm1.transpose(2, 0, 1)
    flow_.q_z_1 = qz1.transpose(2, 0, 1)
    flow_.v_plus_2 = vp2.transpose(2, 0, 1)
    flow_.v_minus_2 = vm2.transpose(2, 0, 1)
    flow_.q_z_2 = qz2.transpose(2, 0, 1)

    # Static aux-data (not traced leaves) here: the default
    # scheme's columns.
    flow_.ur_1 = flow_.ur_2 = None


# ── The step ─────────────────────────────────────────


def _imm_iteration_vp(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> tuple[Array, Array]:
    r"""Primitive `$(u_\pm, p)$` influence-matrix pass (legacy).

    Combines the cylindrical `$u_\pm$` divergence / pressure-gradient
    structure with the Cartesian two-wall `$2 \times 2$` influence
    matrix.  Stages (plus mean-mode projections):

    1. **Poisson RHS** from the cylindrical divergence
       `$(D_1 u_+ + (m{+}1)/r\,u_+)/2 + (D_1 u_- + (1{-}m)/r\,u_-)/2
       + ik_z u_z$`.
    2. **Particular pressure** `$L_k p_P = \hat f_P$` (both Neumann wall
       rows zeroed).
    3. **Helmholtz solves** for `$u_{+,-,z}$` against
       `$(\nabla p)_\pm = D_1 p \mp (m/r) p$`, `$(\nabla p)_z = ik_z p$`
       (both Dirichlet wall rows zeroed; mean-mode `$u_r$` removed).
    4. **Wall divergence residual** (2-vector) `$d_{\mathrm{wall}} =
       D_{1,\mathrm{bnd}} \cdot (u_{+,arb} + u_{-,arb})/2$`.
    5. **Influence matrix** `$\boldsymbol\alpha = -M^{-1}
       d_{\mathrm{wall}}$`.
    6. **Correction** `$u_\pm = u_{\pm,arb} + \alpha_1 v_{\pm,1}
       + \alpha_2 v_{\pm,2}$`, `$u_z = u_{z,arb} - ik_z(\alpha_1 q_{z,1}
       + \alpha_2 q_{z,2})$`.
    7. **Zero mean-mode** `$u_r$` (preserve `$u_\theta$`).
    8. *(optional)* If ``block_mean_spanwise_velocity``, zero the
       mean-mode perturbation bulk axial velocity `$u_z$`.
    """
    c = params.step.implicitness
    dt = flow_.dt
    nu = derived_params.nu  # solvent viscosity (see AnnularFlow.__post_init__)

    uz_n, up_n, um_n = velocity_n[0], velocity_n[1], velocity_n[2]
    NLz_n, NLp_n, NLm_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    NLz_j, NLp_j, NLm_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]
    m = fourier_.m
    mean_mask = fourier_.mean_mask

    m_plus_1_sq = (m + 1) ** 2
    m_minus_1_sq = (m - 1) ** 2
    m_sq = fourier_.m2

    # Batch the D1 derivatives for the divergence and the explicit
    # Hk^- matvec (u_z included) into a single GEMM; only the
    # just-solved pP needs a second D1 after the Poisson solve below.
    # Stack y-leading (N_r, 7, ...) so the batched D1 GEMM (shared by
    # the divergence and the Hk^- matvec below) contracts the leading
    # wall-normal axis transpose-free; the component axis is 1.
    all_v = jnp.stack([up_n, um_n, uz_n, NLp_j, NLp_n, NLm_j, NLm_n], axis=1)
    dy_all = apply_y_matrix(flow_.D1, all_v, component_axis=1)

    # ``dnsjax.analysis`` mirrors this operator in physical
    # components; changing it here means changing
    # ``snapshot_ops.divergence`` and the transcription in
    # ``tests/test_snapshot_export.py`` (``_solver_divergence``),
    # which pins the two together.
    div_n = (
        (dy_all[:, 0] + (m + 1) * inv_r * up_n) / 2
        + (dy_all[:, 1] + (1 - m) * inv_r * um_n) / 2
        + ikz * uz_n
    )
    div_NLj = (
        (dy_all[:, 3] + (m + 1) * inv_r * NLp_j) / 2
        + (dy_all[:, 5] + (1 - m) * inv_r * NLm_j) / 2
        + ikz * NLz_j
    )
    div_NLn = (
        (dy_all[:, 4] + (m + 1) * inv_r * NLp_n) / 2
        + (dy_all[:, 6] + (1 - m) * inv_r * NLm_n) / 2
        + ikz * NLz_n
    )

    Lk_d = _lk_matvec(div_n, flow_, fourier_)
    f_hat = div_n / dt + c * div_NLj + (1 - c) * div_NLn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure (both Neumann wall rows zeroed).
    f_hat_P = f_hat.at[0].set(0.0).at[-1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: pressure gradient and explicit Hk^- matvec.  D1 of the
    # velocity (u_+, u_-, u_z) was already formed above as dy_all[:3];
    # only the just-solved pP needs a fresh D1.
    # y-leading (N_r, 3, ...) Hk construction: the D2 GEMM and the
    # reused D1_vel stay transpose-free (component axis 1); the solve
    # takes this layout (component_axis=1) and we unstack.  inv_r/inv_r2
    # get a trailing axis to broadcast over C; kz2/mean_mask are
    # trailing-mode broadcasts (layout-invariant).
    vel_n_stack = jnp.stack([up_n, um_n, uz_n], axis=1)  # (N_r, 3, ...)
    D1_pP = apply_y_matrix(flow_.D1, pP)
    D1_vel = dy_all[:, :3]
    m_over_r = m * inv_r

    grad_pP_plus = D1_pP - m_over_r * pP
    grad_pP_minus = D1_pP + m_over_r * pP
    grad_pP_z = ikz * pP

    inv_r_y = inv_r[..., None]  # (N_r, 1, 1, 1) over the C axis
    # Kept as `$D_2 + (1/r) D_1$` rather than the fused
    # ``flow_.A_base`` the other three sites use: here `$D_1$` of
    # `$u_\pm$` is **shared** with the divergence above, so fusing
    # would add a 3-wide `$A_{\mathrm{base}}$` GEMM to save a 3-wide
    # `$D_2$` one and shrink the shared batch by one -- 10 field-GEMMs
    # to 9, against halving it where `$D_1$` has no other consumer.
    D2_all = apply_y_matrix(flow_.D2, vel_n_stack, component_axis=1)
    Abase_stack = D2_all + inv_r_y * D1_vel
    meff2_stack = jnp.stack([m_plus_1_sq, m_minus_1_sq, m_sq], axis=1)
    inv_r2 = flow_.inv_r2[:, None, None, None]  # (N_r, 1, 1, 1)
    lapl_stack = (
        Abase_stack - (meff2_stack * inv_r2 + fourier_.kz2) * vel_n_stack
    )
    Hk_minus_stack = (1.0 / dt) * vel_n_stack + (1.0 - c) * nu * lapl_stack
    # Identity wall rows at both walls.
    Hk_minus_stack = Hk_minus_stack.at[0].set(vel_n_stack[0])
    Hk_minus_stack = Hk_minus_stack.at[-1].set(vel_n_stack[-1])

    R_stack = (
        Hk_minus_stack
        - jnp.stack([grad_pP_plus, grad_pP_minus, grad_pP_z], axis=1)
        + c * jnp.stack([NLp_j, NLm_j, NLz_j], axis=1)
        + (1 - c) * jnp.stack([NLp_n, NLm_n, NLz_n], axis=1)
    )
    # Zero Dirichlet wall rows (both walls).
    R_stack = R_stack.at[0].set(0.0).at[-1].set(0.0)

    # Mean mode: zero the u_r part of the +/- RHS so u_r = 0 there.
    Rr_corr = jnp.where(mean_mask, (R_stack[:, 0] + R_stack[:, 1]) / 2, 0.0)
    R_stack = R_stack.at[:, 0].add(-Rr_corr)
    R_stack = R_stack.at[:, 1].add(-Rr_corr)

    arb_stack = flow_.Hk_op.solve(R_stack, component_axis=1)
    up_arb, um_arb, uz_arb = (
        arb_stack[:, 0],
        arb_stack[:, 1],
        arb_stack[:, 2],
    )

    # Stage 4: wall divergence residual (inner, outer).
    ur_arb = (up_arb + um_arb) / 2
    d_wall = jnp.einsum("bj, jmz -> mzb", flow_.D1_bnd, ur_arb)  # (Nm, Nkz, 2)
    d_wall = jnp.where(mean_mask[0][..., None], 0.0, d_wall)

    # Stage 5: influence-matrix algebra (2x2).
    alpha = -jnp.einsum("mzab, mzb -> mza", flow_.M_inv, d_wall)
    alpha1 = alpha[..., 0][None]  # (1, Nm, Nkz)
    alpha2 = alpha[..., 1][None]

    # Stage 6: corrected velocity.
    up_new = up_arb + alpha1 * flow_.v_plus_1 + alpha2 * flow_.v_plus_2
    um_new = um_arb + alpha1 * flow_.v_minus_1 + alpha2 * flow_.v_minus_2
    q_corr = alpha1 * flow_.q_z_1 + alpha2 * flow_.q_z_2

    # Stage 7: zero mean-mode u_r, preserving u_theta.
    ur_corr = jnp.where(mean_mask, (up_new + um_new) / 2, 0.0)
    up_new = up_new - ur_corr
    um_new = um_new - ur_corr

    if params.phys.block_mean_spanwise_velocity:
        # Zero the mean-mode perturbation bulk axial velocity.  At the
        # mean mode alpha = 0 and ikz = 0, so uz_arb already equals the
        # uncorrected uz there; reading the bulk from uz_arb fuses the
        # IMM and bulk corrections.
        mean_uz = extract_mean_mode(uz_arb[None])[0].real
        bulk_uz = (
            integrate_scalar(mean_uz, flow_.y_weights)
            / derived_params.volume_fac
        )
        uz_new = (
            uz_arb
            - ikz * q_corr
            + jnp.where(
                mean_mask,
                -bulk_uz
                * flow_.H_bulk_inv
                * flow_.h_bulk_response[:, None, None],
                0.0,
            )
        )
    else:
        uz_new = uz_arb - ikz * q_corr

    velocity_new = jnp.array([uz_new, up_new, um_new])
    correction = velocity_new - velocity_j

    return velocity_new, correction
