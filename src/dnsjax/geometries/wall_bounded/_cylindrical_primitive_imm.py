r"""Legacy primitive `$(u_\pm, p)$` influence-matrix path, cylindrical.

``res.consistent_imm`` is **on by default**; the pipe's implicit step is
then the spin-quad reformulation in :mod:`.cylindrical`
(:func:`~dnsjax.geometries.wall_bounded.cylindrical._imm_iteration_vw`),
which never forms a pressure.  Setting the flag to ``False`` selects the
**legacy** scheme kept here: the primitive Kleiser-Schumann
influence-matrix method, which solves `$(u_z, u_+, u_-)$` against a
pressure Poisson solve and enforces continuity only at the wall.  It is
retained for reference and for reproducing older trajectories; a state
it steps carries an `$O(1)$` *relative* discrete divergence.  The full
comparison and the measured ledger: the ``Resolution.consistent_imm``
docs (``parameters.py``) and the shared scheme record on
:func:`~dnsjax.geometries.wall_bounded.cartesian._imm_iteration`.

Everything here is reachable **only** when the flag is off, so
``cylindrical.py`` imports this module lazily inside its flag-off
branches and the default path never imports it at all.  The dependency
runs the other way at module scope -- this module imports the shared
`$H_k$` builders and types from ``cylindrical.py`` -- which is why the
import must be deferred there rather than declared at the top of that
file.

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
from ._base import apply_y_matrix, extract_mean_mode
from .cylindrical import (
    CylindricalFlow,
    Fourier,
    _build_Hk_band_gpu,
    _build_Hk_dense_gpu,
)

# ── Pressure Poisson operator (Neumann BC) ──────────────────


def _build_Lk_band_gpu(
    D1_wall: Array,
    band_even: Array,
    band_odd: Array,
    m_is_even: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
    p: int,
) -> Array:
    r"""Build `$L_k$` in banded storage for the Pallas backend.

    Same operator as :func:`_build_Lk_dense_gpu`,
    but assembled directly in banded layout
    ``(Nm, Nkz, Nr, 2p+1)`` (``band[..., i, d] = L_k[..., i, i-p+d]``)
    from the base-operator bands, with no ``(Nr, Nr)`` per mode.

    Parameters
    ----------
    D1_wall:
        Last row of `$D_1$` (parity-independent), shape ``(Nr,)``.
    band_even, band_odd:
        Banded `$A_{\mathrm{base}}$` for even/odd parity,
        shape ``(Nr, 2p+1)``.
    m_is_even, m2:
        Pressure parity selector and `$m^2$`, shape ``(Nm, 1, 1)``.
    inv_r2:
        `$1/r_j^2$`, shape ``(Nr,)``.
    kz2:
        `$k_z^2$`, shape ``(1, Nkz, 1)``.
    mean_mask:
        Mean-mode boolean mask, shape ``(Nm, Nkz, 1)``.
    p:
        FD order (half-bandwidth).
    """
    Nr = band_even.shape[0]
    band_base = jnp.where(m_is_even, band_even[None], band_odd[None])
    diag = -(m2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    # Single wall (r = 1): Neumann D1[-1, :] in band form, identity
    # (pin) at the mean mode; r = 0 regularity is built into the
    # parity-reduced base band, so no inner-wall row.
    neumann = _banded_wall_row(D1_wall, Nr - 1, p)
    wall = jnp.where(
        mean_mask, _banded_diag_column(p, band_base.dtype), neumann
    )  # (Nm, Nkz, 2p+1)
    return _assemble_banded_operator(
        band_base[:, None], 1.0, diag, [(Nr - 1, wall)]
    )


def _build_Lk_dense_gpu(
    D1_wall: Array,
    A_base_even: Array,
    A_base_odd: Array,
    m_is_even: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
) -> Array:
    r"""Build dense `$L_k$` on GPU (dense backend only).

    Returns the full ``(Nm, Nkz, Nr, Nr)`` pressure Poisson
    operator.  The parity-dependent row selection is handled
    by ``jnp.where`` on the ``m_is_even`` mask.
    """
    Nr = A_base_even.shape[0]
    dtype = A_base_even.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)

    m2_over_r2 = m2 * inv_r2  # (Nm, 1, Nr)
    diag_shift = -(m2_over_r2 + kz2)  # (Nm, Nkz, Nr)

    Lk_even = A_base_even[None, None] + diag_shift[..., None] * eye_Nr
    Lk_odd = A_base_odd[None, None] + diag_shift[..., None] * eye_Nr
    Lk = jnp.where(m_is_even[..., None], Lk_even, Lk_odd)

    # Wall BC: Neumann D1[-1,:] for all modes, pin at the mean.
    D1_wall_1d = D1_wall.ravel()
    pin = eye_Nr[-1, :]
    wall_row = jnp.where(mean_mask, pin, D1_wall_1d)
    Lk = Lk.at[..., -1, :].set(wall_row)

    return Lk


# ── Three-family Helmholtz group (+, -, z) ───────────────────


def _hk_bands(
    dt: float | Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> list[Array]:
    r"""Assemble the banded `$H_k$` group (+, -, z) at *dt*.

    Single-sources the band assembly for the setup-checked build, the
    adaptive ``dt_max`` stability pre-check, and the jitted ``set_dt``
    rebuild (:func:`_build_dt_leaves`).  Pallas backend only.

    The half-width is read back from the already-factored (and
    ``dt``-independent) `$L_k$`, whose ``L`` factor is
    ``(Nr, p, Nm, Nkz)`` -- a static shape, so this works inside
    ``jit`` (``set_dt``) where a host-side ``matrix_half_bandwidth`` on
    the traced ``A_base`` could not.  It is ``fd_order`` in **both**
    flag states: ``res.consistent_imm`` swaps in a band-preserving
    Dirichlet recovery operator, so no shipped configuration widens
    the band.
    """
    p_band = flow_.Lk_op.L.shape[1]
    m_s = fourier_.m[0, ..., None]
    kz2_s = fourier_.kz2[0, ..., None]
    m_is_even_s = fourier_.m_is_even[0, ..., None]
    # u_+/u_- carry parity (-1)^{m+1}; u_z carries (-1)^m.
    m_is_even_v = 1.0 - m_is_even_s
    band_even = _banded_from_dense(flow_.A_base_even, p_band)
    band_odd = _banded_from_dense(flow_.A_base_odd, p_band)
    groups = (
        (m_is_even_v, (m_s + 1) ** 2),
        (m_is_even_v, (m_s - 1) ** 2),
        (m_is_even_s, m_s**2),
    )
    return [
        _build_Hk_band_gpu(
            band_even,
            band_odd,
            parity,
            meff2,
            flow_.inv_r2,
            kz2_s,
            dt,
            params.step.implicitness,
            derived_params.nu,
            p_band,
        )
        for parity, meff2 in groups
    ]


def _hk_dense_op(
    dt: float | Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> DenseJAXSolver:
    r"""Factored dense stacked `$H_k$` (+, -, z) at *dt* (dense
    backend)."""
    m_s = fourier_.m[0, ..., None]
    kz2_s = fourier_.kz2[0, ..., None]
    m_is_even_s = fourier_.m_is_even[0, ..., None]
    m_is_even_v = 1.0 - m_is_even_s
    groups = (
        (m_is_even_v, (m_s + 1) ** 2),
        (m_is_even_v, (m_s - 1) ** 2),
        (m_is_even_s, m_s**2),
    )
    ops = [
        DenseJAXSolver(
            _build_Hk_dense_gpu(
                flow_.A_base_even,
                flow_.A_base_odd,
                parity,
                meff2,
                flow_.inv_r2,
                kz2_s,
                dt,
                params.step.implicitness,
                derived_params.nu,
            )
        )
        for parity, meff2 in groups
    ]
    return DenseJAXSolver.from_factors(
        lu=jnp.stack([o.lu for o in ops]),
        perm=jnp.stack([o.perm for o in ops]),
    )


# ── Matrix-free matvecs ────────────────────────────────


def _abase_matvec(
    u: Array,
    flow_: CylindricalFlow,
    parity_sign: Array,
) -> Array:
    r"""Apply `$A_{\mathrm{base}}^{(\sigma)} u$` matrix-free.

    .. math::
        A_{\mathrm{base}}^{(\sigma)} u
        = \underbrace{(D_{2,\mathrm{pos}} + (1/r)\,
          D_{1,\mathrm{pos}})\,u}_{\text{common part}}
        + (-1)^{m_{\mathrm{eff}}}
          \underbrace{(\widetilde{D}_{2,\mathrm{ghost}}
          + (1/r)\,\widetilde{D}_{1,\mathrm{ghost}})
          \,u}_{\text{ghost correction}}

    The ghost correction matrices are stored row-sliced to
    their `$g \sim p/2$` nonzero rows (near the pipe centre,
    where stencils cross `$r = 0$`), so the ghost GEMMs and
    the scatter-add touch only the first `$g$` radial points.

    Parameters
    ----------
    u:
        Field, shape ``(Nr, Nm, Nkz)``.
    flow\_:
        Cylindrical flow data (uses ``D1_pos``,
        ``D2_pos``, ``D1_ghost``, ``D2_ghost``,
        ``inv_r``).
    parity_sign:
        `$(-1)^{m_{\mathrm{eff}}}$`, shape
        ``(1, Nm, 1)``.
    """
    inv_r = flow_.inv_r[:, None, None]
    D2_u = apply_y_matrix(flow_.D2_pos, u)
    D1_u = apply_y_matrix(flow_.D1_pos, u)
    common = D2_u + inv_r * D1_u

    g = flow_.D1_ghost.shape[0]
    D2g_u = apply_y_matrix(flow_.D2_ghost, u)
    D1g_u = apply_y_matrix(flow_.D1_ghost, u)
    ghost = D2g_u + inv_r[:g] * D1g_u

    return common.at[:g].add(parity_sign * ghost)


def _lk_matvec(
    u: Array,
    flow_: CylindricalFlow,
    fourier_: Fourier,
) -> Array:
    r"""Apply `$L_k u$` for the pressure Poisson operator.

    Matrix-free evaluation:
    `$L_k u = A_{\mathrm{base}}^{(\sigma_p)} u
    - (m^2/r^2 + k_z^2) u$`, with Neumann wall row and
    mean-mode pin.

    Parity for pressure: `$(-1)^m$`, so parity_sign =
    ``m_is_even * 2 - 1`` (``+1`` for even, ``-1`` for odd).
    """
    parity_sign = fourier_.m_is_even * 2 - 1

    Abase_u = _abase_matvec(u, flow_, parity_sign)
    inv_r2 = flow_.inv_r2[:, None, None]
    out = Abase_u - (fourier_.m2 * inv_r2 + fourier_.kz2) * u

    # Wall row: Neumann D1[-1,:] for all modes, pin at the mean.
    D1_wall_row = flow_.D1_wall.ravel()
    wall_val = jnp.einsum("j, jmz -> mz", D1_wall_row, u)
    bot = jnp.where(fourier_.mean_mask[0], u[-1], wall_val)
    return out.at[-1].set(bot)


# ── Homogeneous columns and influence matrix ────────────────


def derive_homogeneous_data(
    flow_: CylindricalFlow, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
) -> None:
    r"""Fill the homogeneous responses and the `$1 \times 1$` ``M_inv``.

    The legacy half of
    :meth:`~dnsjax.geometries.wall_bounded.cylindrical.CylindricalFlow._derive_imm_homogeneous_data`
    (the dispatcher, which calls this when ``res.consistent_imm`` is
    off).  A single unit-wall pressure `$L_k p_1 = e_{\mathrm{wall}}$`
    gives, through the pressure gradient and the Helmholtz solves, the
    `$u_\pm$` responses ``v_plus_1``/``v_minus_1`` and the axial
    potential ``q_z_1``; the influence "matrix" is the scalar
    `$M = D_{1,\mathrm{wall}} \cdot (v_+ + v_-)/2$` and ``M_inv`` its
    reciprocal (the pipe has one wall).
    """
    # This run-once setup stays in the mode-outer (Nm, Nkz, Nr)
    # layout: the influence-matrix einsums below operate on it and
    # the results are transposed to field layout (Nr, Nm, Nkz) at
    # the end.  ``.solve`` now takes a mode-inner field, so each
    # setup solve is wrapped (transpose in, transpose out) to keep
    # this layout.  FUTURE: rebuild this setup natively mode-inner to
    # drop the wrappers -- the hot path already is; here it only
    # relocates a one-time transpose, so it is deferred.
    e_wall = (
        jnp.zeros(
            (Nm, Nkz, Nr),
            dtype=sharding.float_type,
            out_sharding=sharding.spec_imm_corr_shard,
        )
        .at[..., -1]
        .set(1.0)
    )
    p1_s = flow_.Lk_op.solve(e_wall.transpose(2, 0, 1)).transpose(1, 2, 0)

    # Pressure gradient components for the +/- equations.
    # The ghost matrix holds only its g nonzero rows; its
    # contribution lands in the first g radial entries.
    parity_sign_p_s = fourier_.m_is_even[0, ..., None] * 2 - 1
    g = flow_.D1_ghost.shape[0]
    ghost_p1 = jnp.einsum("ij, mzj -> mzi", flow_.D1_ghost, p1_s)
    D1_p1 = jnp.einsum("ij, mzj -> mzi", flow_.D1_pos, p1_s)
    D1_p1 = D1_p1.at[..., :g].add(parity_sign_p_s * ghost_p1)
    m_s = fourier_.m[0, ..., None]  # (Nm, 1, 1)
    m_over_r_s = m_s * flow_.inv_r  # (Nm, 1, Nr)

    rhs_v_plus = -(D1_p1 - m_over_r_s * p1_s)
    rhs_v_minus = -(D1_p1 + m_over_r_s * p1_s)
    rhs_v_plus = rhs_v_plus.at[..., -1].set(0.0)
    rhs_v_minus = rhs_v_minus.at[..., -1].set(0.0)
    q_rhs = p1_s.at[..., -1].set(0.0)

    # Batched solve: component order (plus, minus, z).
    rhs_stack = jnp.stack([rhs_v_plus, rhs_v_minus, q_rhs])
    result_stack = flow_.Hk_op.solve(
        rhs_stack.transpose(0, 3, 1, 2)
    ).transpose(0, 2, 3, 1)
    vp1_s = result_stack[0]
    vm1_s = result_stack[1]
    qz1_s = result_stack[2]

    # Zero the u_r part at the mean mode, preserving u_theta.
    mean_s = fourier_.mean_mask[0, ..., None]  # (Nm, Nkz, 1)
    vr_corr = jnp.where(mean_s, (vp1_s + vm1_s) / 2, 0.0)
    vp1_s = vp1_s - vr_corr
    vm1_s = vm1_s - vr_corr

    # 1x1 influence matrix.
    D1_wall_row = flow_.D1_wall.ravel()  # (Nr,)
    ur_1 = (vp1_s + vm1_s) / 2
    M = jnp.einsum("j, mzj -> mz", D1_wall_row, ur_1)

    is_mean = fourier_.mean_mask[0]  # (Nm, Nkz)
    safe_M = jnp.where(is_mean, 1.0, M)
    flow_.M_inv = jnp.where(is_mean, 0.0, 1.0 / safe_M)

    # Transpose to field layout (Nr, Nm, Nkz).
    flow_.v_plus_1 = vp1_s.transpose(2, 0, 1)
    flow_.v_minus_1 = vm1_s.transpose(2, 0, 1)
    flow_.q_z_1 = qz1_s.transpose(2, 0, 1)

    # Static aux-data (not traced leaves) here: the default
    # scheme's column.
    flow_.ur_1 = None


# ── The step ─────────────────────────────────────────


def _imm_iteration_vp(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> tuple[Array, Array]:
    r"""Primitive `$(u_\pm, p)$` influence-matrix pass (legacy).

    The pipe's single wall at `$r = 1$` gives a `$1 \times 1$`
    influence matrix (scalar `$\alpha$` per mode).

    Six stages (plus mean-mode projections):

    1. **Poisson RHS**: cylindrical divergence of momentum in
       `$(u_z, u_+, u_-)$` components:

       .. math::
           \nabla\!\cdot\!\mathbf{u}
           = \frac{D_1 u_+ + (m+1)/r\;u_+}{2}
           + \frac{D_1 u_- + (1-m)/r\;u_-}{2}
           + ik_z\,u_z

    2. **Particular pressure**: `$L_k p_P = \hat{f}_P$` with
       zero Neumann wall row.
    3. **Helmholtz solves**: three separate solves with
       `$H_{k,+}$`, `$H_{k,-}$`, `$H_{k,z}$`.  Pressure
       gradient in `$(+, -, z)$`:

       .. math::
           (\nabla p)_+ = D_1 p - (m/r)\,p, \quad
           (\nabla p)_- = D_1 p + (m/r)\,p, \quad
           (\nabla p)_z = ik_z\,p

    4. **Wall divergence residual**:
       `$d_{\mathrm{wall}} = D_{1,\mathrm{wall}}
       \cdot (u_{+,arb} + u_{-,arb})/2$`
    5. **Influence matrix**: `$\alpha = -M^{-1} d_{\mathrm{wall}}$`
    6. **Correction**:
       `$u_+ = u_{+,arb} + \alpha\,v_{+,1}$`,
       `$u_- = u_{-,arb} + \alpha\,v_{-,1}$`,
       `$u_z = u_{z,arb} - ik_z\,\alpha\,q_{z,1}$`.
    7. **Zero mean-mode** `$u_r$`: continuity
       `$(1/r)\,\partial(r u_r)/\partial r = 0$` plus
       no-slip at `$r = 1$` forces `$u_r \equiv 0$` at the
       mean mode.  The `$u_\theta$` part of `$u_\pm$` is
       preserved.
    8. *(optional)* If ``constant_bulk_velocity``, zero the
       mean-mode perturbation bulk `$u_z$`.
    """
    c = params.step.implicitness
    dt = flow_.dt
    nu = derived_params.nu

    uz_n, up_n, um_n = velocity_n[0], velocity_n[1], velocity_n[2]
    NLz_n, NLp_n, NLm_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    NLz_j, NLp_j, NLm_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]
    m = fourier_.m

    # Parity signs for each component type.
    parity_sign_p = fourier_.m_is_even * 2 - 1  # (-1)^m
    parity_sign_v = -parity_sign_p  # (-1)^{m+1}

    m_plus_1_sq = (m + 1) ** 2
    m_minus_1_sq = (m - 1) ** 2
    m_sq = fourier_.m2

    # Batch all D1 y-derivatives with (-1)^{m+1} parity into
    # one GEMM each for D1_pos and D1_ghost (2 instead of 4);
    # the ghost GEMM covers only its g nonzero rows.
    g = flow_.D1_ghost.shape[0]
    # Stack y-leading (N_r, 6, ...) so the batched D1 GEMM contracts the
    # leading wall-normal axis transpose-free; the component axis is 1.
    all_vparity = jnp.stack([up_n, um_n, NLp_j, NLp_n, NLm_j, NLm_n], axis=1)
    dy_common = apply_y_matrix(flow_.D1_pos, all_vparity, component_axis=1)
    dy_ghost = apply_y_matrix(flow_.D1_ghost, all_vparity, component_axis=1)
    dy_all = dy_common.at[:g].add(parity_sign_v * dy_ghost)

    # Cylindrical divergence at time n.  ``dnsjax.analysis`` mirrors
    # this operator in physical components; changing it here means
    # changing ``snapshot_ops.divergence`` and the transcription in
    # ``tests/test_snapshot_export.py`` (``_solver_divergence``),
    # which pins the two together.
    div_n = (
        (dy_all[:, 0] + (m + 1) * inv_r * up_n) / 2
        + (dy_all[:, 1] + (1 - m) * inv_r * um_n) / 2
        + ikz * uz_n
    )

    # Divergence of nonlinear terms at times n and j.
    div_NLj = (
        (dy_all[:, 2] + (m + 1) * inv_r * NLp_j) / 2
        + (dy_all[:, 4] + (1 - m) * inv_r * NLm_j) / 2
        + ikz * NLz_j
    )
    div_NLn = (
        (dy_all[:, 3] + (m + 1) * inv_r * NLp_n) / 2
        + (dy_all[:, 5] + (1 - m) * inv_r * NLm_n) / 2
        + ikz * NLz_n
    )

    Lk_d = _lk_matvec(div_n, flow_, fourier_)

    f_hat = div_n / dt + c * div_NLj + (1 - c) * div_NLn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure.
    f_hat_P = f_hat.at[-1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: Helmholtz solves for each component.  The Hk construction
    # is built **y-leading** ``(N_r, C, ...)`` so the batched D1/D2 GEMMs
    # contract the leading wall-normal axis transpose-free (component axis
    # 1); the solve takes that layout directly (``component_axis=1``) and
    # we unstack.  ``inv_r``/``inv_r2`` get a trailing axis to broadcast
    # over the C axis; ``kz2``/``mean_mask`` are trailing-mode broadcasts
    # (layout-invariant).
    inv_r_y = inv_r[..., None]  # (N_r, 1, 1, 1) over the C axis
    vel_n_stack = jnp.stack([up_n, um_n, uz_n], axis=1)  # (N_r, 3, ...)
    pP_and_vel = jnp.concatenate([pP[:, None], vel_n_stack], axis=1)
    D1_batch = apply_y_matrix(flow_.D1_pos, pP_and_vel, component_axis=1)
    D1g_batch = apply_y_matrix(flow_.D1_ghost, pP_and_vel, component_axis=1)

    # pP pressure gradient (parity (-1)^m -> parity_sign_p).
    D1_pP = D1_batch[:, 0].at[:g].add(parity_sign_p * D1g_batch[:, 0])
    m_over_r = m * inv_r  # (1, Nm, 1) * (Nr, 1, 1) → (Nr, Nm, 1)

    grad_pP_plus = D1_pP - m_over_r * pP
    grad_pP_minus = D1_pP + m_over_r * pP
    grad_pP_z = ikz * pP

    # Batched `$H_k^-$` matvec for all three components (y-leading).
    D1_vel = D1_batch[:, 1:]
    D1g_vel = D1g_batch[:, 1:]
    D2_all = apply_y_matrix(flow_.D2_pos, vel_n_stack, component_axis=1)
    D2g_all = apply_y_matrix(flow_.D2_ghost, vel_n_stack, component_axis=1)
    common_hk = D2_all + inv_r_y * D1_vel
    ghost_hk = D2g_all + inv_r_y[:g] * D1g_vel
    parity_hk = jnp.stack(
        [parity_sign_v, parity_sign_v, parity_sign_p], axis=1
    )
    Abase_stack = common_hk.at[:g].add(parity_hk * ghost_hk)
    meff2_stack = jnp.stack([m_plus_1_sq, m_minus_1_sq, m_sq], axis=1)
    inv_r2 = flow_.inv_r2[:, None, None, None]  # (N_r, 1, 1, 1)
    lapl_stack = (
        Abase_stack - (meff2_stack * inv_r2 + fourier_.kz2) * vel_n_stack
    )
    Hk_minus_stack = (1.0 / dt) * vel_n_stack + (1.0 - c) * nu * lapl_stack
    Hk_minus_stack = Hk_minus_stack.at[-1].set(vel_n_stack[-1])

    R_stack = (
        Hk_minus_stack
        - jnp.stack([grad_pP_plus, grad_pP_minus, grad_pP_z], axis=1)
        + c * jnp.stack([NLp_j, NLm_j, NLz_j], axis=1)
        + (1 - c) * jnp.stack([NLp_n, NLm_n, NLz_n], axis=1)
    )

    # Zero wall BC (Dirichlet no-slip).
    R_stack = R_stack.at[-1].set(0.0)

    # Zero the u_r part of the +/- RHS at the mean mode so
    # the Helmholtz solves produce u_r = 0 there.  At m=0,
    # Hk_plus and Hk_minus are identical (m_eff^2 = 1, same
    # parity), so the antisymmetric RHS gives up = -um.
    Rr_corr = jnp.where(
        fourier_.mean_mask, (R_stack[:, 0] + R_stack[:, 1]) / 2, 0.0
    )
    R_stack = R_stack.at[:, 0].add(-Rr_corr)
    R_stack = R_stack.at[:, 1].add(-Rr_corr)

    # Batched Helmholtz solve (y-leading, component axis 1).
    arb_stack = flow_.Hk_op.solve(R_stack, component_axis=1)
    up_arb, um_arb, uz_arb = (
        arb_stack[:, 0],
        arb_stack[:, 1],
        arb_stack[:, 2],
    )

    # Stage 4: wall divergence residual.
    D1_wall_row = flow_.D1_wall.ravel()
    ur_arb = (up_arb + um_arb) / 2
    d_wall = jnp.einsum("j, jmz -> mz", D1_wall_row, ur_arb)

    # Mean mode: pressure is a gauge; zero the residual.
    d_wall = jnp.where(fourier_.mean_mask[0], 0.0, d_wall)

    # Stage 5: influence matrix correction (scalar per mode).
    alpha = (-flow_.M_inv * d_wall)[None]  # (1, Nm, Nkz)
    # Stage 6: corrected velocity.
    up_new = up_arb + alpha * flow_.v_plus_1
    um_new = um_arb + alpha * flow_.v_minus_1
    qz_corr = alpha * flow_.q_z_1

    # Stage 7: zero mean-mode u_r, preserving u_theta.
    ur_corr = jnp.where(fourier_.mean_mask, (up_new + um_new) / 2, 0.0)
    up_new = up_new - ur_corr
    um_new = um_new - ur_corr

    # Constant-bulk-velocity enforcement: add a uniform mean
    # pressure gradient G to the mean-mode u_z Helmholtz RHS
    # so that the perturbation bulk velocity is zero.
    # Equivalent post-solve form: uz += G * h, where
    # h = Hk_z^{-1} [1,...,1,0] and G = -Ub_pert / H_bulk.
    # At the mean mode alpha = 0 and ikz = 0, so uz_arb
    # already equals the uncorrected uz_new there; reading
    # the bulk from uz_arb lets the IMM correction and the
    # bulk correction fuse into a single expression.  The
    # write mask is ``mean_mask``: no other mode (padding
    # included) receives the correction.
    if params.phys.driving == "constant_bulk_velocity":
        mean_uz = extract_mean_mode(uz_arb[None])[0].real
        bulk_uz = 2 * jnp.dot(flow_.y_weights, mean_uz)
        uz_new = (
            uz_arb
            - ikz * qz_corr
            + jnp.where(
                fourier_.mean_mask,
                -bulk_uz
                * flow_.H_bulk_inv
                * flow_.h_bulk_response[:, None, None],
                0.0,
            )
        )
    else:
        uz_new = uz_arb - ikz * qz_corr

    velocity_new = jnp.array([uz_new, up_new, um_new])
    correction = velocity_new - velocity_j

    return velocity_new, correction
