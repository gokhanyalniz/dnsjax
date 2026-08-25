r"""Legacy primitive `$(v, p)$` influence-matrix path, Cartesian.

``res.consistent_imm`` is **on by default**; the Cartesian implicit step
is then the `$v$`-`$\omega_y$` reformulation in :mod:`.cartesian`
(:func:`~dnsjax.geometries.wall_bounded.cartesian._imm_iteration_vw`),
which never forms a pressure.  Setting the flag to ``False`` selects the
**legacy** scheme kept here: the primitive Kleiser-Schumann
influence-matrix method, which solves the three velocity components
against a pressure Poisson solve and enforces continuity only at the two
walls.  It is retained for reference and for reproducing older
trajectories; a state it steps carries an `$O(1)$` *relative* discrete
divergence.  The full comparison, the measured ledger and the four other
repairs that were tried and retired: the ``Resolution.consistent_imm``
docs (``parameters.py``) and
:func:`~dnsjax.geometries.wall_bounded.cartesian._imm_iteration`.

Everything here is reachable **only** when the flag is off, so
``cartesian.py`` imports this module lazily inside its flag-off branches
and the default path never imports it at all.  The dependency runs the
other way at module scope -- this module imports the shared operators and
types from ``cartesian.py`` -- which is why the import must be deferred
there rather than declared at the top of that file.

Contents: the Neumann-BC pressure Poisson operator in both storage
backends (:func:`_build_Lk_band_gpu`, :func:`_build_Lk_dense_gpu`), its
matrix-free apply (:func:`_lk_matvec`), the homogeneous-column /
influence-matrix derivation (:func:`derive_homogeneous_data`), and the
step itself (:func:`_imm_iteration_vp`).
"""

import jax
from jax import Array
from jax import numpy as jnp

from ...parameters import params
from ...solvers import (
    _assemble_banded_operator,
    _banded_diag_column,
    _banded_from_dense,
    _banded_wall_row,
)
from ._base import apply_y_matrix
from .cartesian import (
    CartesianFlow,
    Fourier,
    _apply_bulk_corrections,
    _hk_minus_matvec,
)

# ── Pressure Poisson operator (Neumann BCs) ───────────────────────


def _build_Lk_band_gpu(
    D1: Array,
    D2: Array,
    k2: Array,
    mean_mask: Array,
    p: int,
) -> Array:
    r"""Build `$L_k$` in banded storage for the Pallas backend.

    Same Neumann-BC pressure Poisson operator
    `$L_k = D_2 - k^2 I$` as :func:`_build_Lk_dense_gpu`,
    but assembled directly in banded
    layout ``(Nkz, Nkx, Ny, 2p+1)``
    (``band[..., i, d] = L_k[..., i, i-p+d]``) from the base band
    ``_banded_from_dense(D2, p)``, with no ``(Ny, Ny)`` per mode.  The
    `$-k^2$` shift is constant across rows; Neumann `$D_1$` rows sit at
    **both** walls (rows 0 and ``Ny-1``), with a mean-mode identity pin
    at the outer wall (the only `$k^2 = 0$` system).

    Parameters
    ----------
    D1, D2:
        First/second-derivative matrices, ``(Ny, Ny)``.
    k2:
        `$k_x^2 + k_z^2$`, ``(Nkz, Nkx, 1)``.
    mean_mask:
        Mean-mode boolean mask, same shape as *k2*.
    p:
        FD order (half-bandwidth).
    """
    Ny = D2.shape[-1]
    band_D2 = _banded_from_dense(D2, p)  # (Ny, 2p+1)
    diag = -k2  # (Nkz, Nkx, 1), constant across rows
    inner = _banded_wall_row(D1[0], 0, p)  # Neumann, inner wall
    neumann_outer = _banded_wall_row(D1[-1], Ny - 1, p)  # Neumann, outer
    outer = jnp.where(
        mean_mask, _banded_diag_column(p, band_D2.dtype), neumann_outer
    )  # (Nkz, Nkx, 2p+1)
    return _assemble_banded_operator(
        band_D2, 1.0, diag, [(0, inner), (Ny - 1, outer)]
    )


def _build_Lk_dense_gpu(
    D1: Array, D2: Array, k2: Array, mean_mask: Array
) -> Array:
    """Build the Neumann-BC Laplacian `$L_k$` in dense form on GPU.

    Used only by the ``"dense"`` solver backend; allocates
    `$(N_{kz}, N_{kx}, N_y, N_y)$`.  No CPU path.

    Parameters follow :func:`_build_Lk_band_gpu` (sans ``p``);
    the output is the full dense operator.
    """
    Ny = D2.shape[-1]
    eye = jnp.eye(Ny, dtype=D2.dtype)
    # Lk_interior[..., i, j] = D2[i, j] - k2 * delta_{i, j}
    Lk = D2[None, None, :, :] - k2[..., None] * eye

    # Row 0: D1[0, :] for all modes (Neumann).
    Lk = Lk.at[..., 0, :].set(D1[0, :])

    # Row -1: D1[-1, :] for all modes; pin row [0, ..., 0, 1]
    # at the mean mode.  mean_mask is (Nkz, Nkx, 1); `jnp.where`
    # broadcasts the (Ny,) branches to (Nkz, Nkx, Ny).
    pin = eye[-1, :]  # (Ny,)
    row_N = jnp.where(mean_mask, pin, D1[-1, :])
    Lk = Lk.at[..., -1, :].set(row_N)

    return Lk


def _lk_matvec(
    u: Array,
    flow_: CartesianFlow,
    fourier_: Fourier,
) -> Array:
    r"""Apply `$L_k u$` for the Neumann-BC pressure Poisson operator.

    Matrix-free evaluation that avoids storing the per-mode
    ``(Nkz, Nkx, Ny, Ny)`` operator.  The interior of the
    output is `$D_2 u - k^2 u$`; the wall rows use `$D_1$`
    to encode Neumann BCs, except for the mean mode (the
    only `$k^2 = 0$` mode) where the top-wall row pins
    `$p_{N_y-1} = 0$` (matching
    :func:`_build_Lk_dense_gpu`).

    Parameters
    ----------
    u:
        Field, shape ``(Ny, Nkz, Nkx)``.
    flow\_:
        Cartesian flow data (uses ``D2``, ``D1_bnd``).
    fourier\_:
        Wavenumber grids (uses ``k2``, ``mean_mask``).
    """
    D2u = apply_y_matrix(flow_.D2, u)
    out = D2u - fourier_.k2 * u
    bot = jnp.einsum("j, jzx -> zx", flow_.D1_bnd[0], u)
    top_neumann = jnp.einsum("j, jzx -> zx", flow_.D1_bnd[-1], u)
    top = jnp.where(fourier_.mean_mask[0], u[-1], top_neumann)
    return out.at[0].set(bot).at[-1].set(top)


# ── Homogeneous columns and influence matrix ──────────────────────


def derive_homogeneous_data(
    flow_: CartesianFlow,
    fourier_: Fourier,
    e_cols: list[Array],
) -> None:
    r"""Fill ``v1``, ``v2``, ``q1``, ``q2`` and ``M_inv`` on *flow_*.

    The legacy half of
    :meth:`~dnsjax.geometries.wall_bounded.cartesian.CartesianFlow._derive_imm_homogeneous_data`
    (the dispatcher, which builds *e_cols* and calls this when
    ``res.consistent_imm`` is off).  Both backends converge here once
    ``Lk_op`` and ``Hk_op`` are in place; nothing else on the CPU needs
    another LU solve -- everything below runs against the
    already-factored device operator.

    In Schur-complement notation, the arrays ``p1, p2, v1, v2, q1, q2``
    are the columns of `$A_{II}^{-1}\,A_{IB}$` (the interior-to-boundary
    coupling through the factored interior operator), and ``M_inv`` is
    `$S^{-1}$` where `$S$` is the `$2 \times 2$` Schur complement
    (influence / capacitance matrix).  See :func:`_imm_iteration_vp` for
    the full context.  The homogeneous pressures ``p1``, ``p2`` are
    needed only within this derivation (the IMM never assembles the
    pressure), so they are not stored on the dataclass.

    The mean mode (the only `$k^2 = 0$` system) is handled analytically:
    ``M`` has a zero second column there (`$p_2 \equiv 1$` is a pressure
    gauge), so the `$2 \times 2$` inverse is replaced by
    `$[[1/M_{00}, 0], [0, 0]]$`.  The ``jnp.where`` around ``safe_det``
    keeps the regular branch NaN-free before the selection happens.
    Padding modes take the regular branch (their placeholder
    `$k^2 \ne 0$` systems are as well-posed as physical ones); the
    values are inert, multiplied only by the exactly-zero wall residuals
    of zero fields.

    After ``M_inv`` is built, ``v1`` and ``v2`` are zeroed at the mean
    mode so the IMM velocity correction produces zero there (continuity
    forces `$v \equiv 0$` at `$k^2 = 0$`).  The zeroing must follow the
    ``M_inv`` computation, which uses the original ``v1`` to evaluate
    `$1/M_{00}$`.

    Parameters
    ----------
    flow\_:
        The flow being built; every field below is written on it.
    fourier\_:
        Wavenumber grids (uses ``mean_mask``).
    e_cols:
        The two unit wall vectors, mode-outer ``(Nkz, Nkx, Ny)``.
    """
    # This run-once setup stays in the mode-outer (Nkz, Nkx, Ny)
    # layout: the influence-matrix einsums below operate on it and
    # the results are transposed to field layout (Ny, Nkz, Nkx) at
    # the end.  ``.solve`` now takes a mode-inner field, so each
    # setup solve is wrapped (transpose in, transpose out) to keep
    # this layout.  FUTURE: rebuild this setup natively mode-inner to
    # drop the wrappers -- the hot path already is; here it only
    # relocates a one-time transpose, so it is deferred.
    e1_b, e2_b = e_cols

    p1_s = flow_.Lk_op.solve(e1_b.transpose(2, 0, 1)).transpose(1, 2, 0)
    p2_s = flow_.Lk_op.solve(e2_b.transpose(2, 0, 1)).transpose(1, 2, 0)

    rhs_v1 = -jnp.einsum("ij, zxj -> zxi", flow_.D1, p1_s)
    rhs_v2 = -jnp.einsum("ij, zxj -> zxi", flow_.D1, p2_s)
    rhs_v1 = rhs_v1.at[..., 0].set(0.0).at[..., -1].set(0.0)
    rhs_v2 = rhs_v2.at[..., 0].set(0.0).at[..., -1].set(0.0)
    v1_s = flow_.Hk_op.solve(rhs_v1.transpose(2, 0, 1)).transpose(1, 2, 0)
    v2_s = flow_.Hk_op.solve(rhs_v2.transpose(2, 0, 1)).transpose(1, 2, 0)

    q_rhs1 = p1_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
    q_rhs2 = p2_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
    q1_s = flow_.Hk_op.solve(q_rhs1.transpose(2, 0, 1)).transpose(1, 2, 0)
    q2_s = flow_.Hk_op.solve(q_rhs2.transpose(2, 0, 1)).transpose(1, 2, 0)

    # Influence matrix `$M_{ji} = (D_1 v_i)|_{\\text{wall}_j}$`.
    M00 = jnp.einsum("j, zxj -> zx", flow_.D1_bnd[0], v1_s)
    M01 = jnp.einsum("j, zxj -> zx", flow_.D1_bnd[0], v2_s)
    M10 = jnp.einsum("j, zxj -> zx", flow_.D1_bnd[-1], v1_s)
    M11 = jnp.einsum("j, zxj -> zx", flow_.D1_bnd[-1], v2_s)

    is_mean = fourier_.mean_mask[0]
    det = M00 * M11 - M01 * M10
    safe_det = jnp.where(is_mean, 1.0, det)
    inv_00 = jnp.where(is_mean, 1.0 / M00, M11 / safe_det)
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

    # Transpose to field layout (Ny, Nkz, Nkx).
    flow_.v1 = v1_s.transpose(2, 0, 1)
    flow_.v2 = v2_s.transpose(2, 0, 1)
    flow_.q1 = q1_s.transpose(2, 0, 1)
    flow_.q2 = q2_s.transpose(2, 0, 1)

    # Zero homogeneous wall-normal velocity at the mean mode.
    flow_.v1 = jnp.where(fourier_.mean_mask, 0.0, flow_.v1)
    flow_.v2 = jnp.where(fourier_.mean_mask, 0.0, flow_.v2)


# ── The step ──────────────────────────────────────────────────────


def _imm_iteration_vp(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: CartesianFlow,
) -> tuple[Array, Array, dict[str, Array]]:
    r"""Kleiser-Schumann influence-matrix method.

    The y-momentum equation supplies only the *interior* Poisson
    equation for pressure; the wall BC is determined indirectly by
    enforcing continuity `$\nabla \cdot u = 0$` at the walls.

    Nine stages (six core IMM stages, then three mean-mode projections):

    1. Build the interior Poisson RHS from divergence of momentum.
    2. Solve Poisson for the particular pressure `$p_P$` with
       arbitrary (zero) Neumann BCs.
    3. Solve Helmholtz for all three particular velocity components
       `$u_{arb}, v_{arb}, w_{arb}$` against `$p_P$` (zero
       Dirichlet BCs).
    4. Compute wall divergence residual
       `$d_{\mathrm{wall}} = (D_1 v_{arb})|_{\mathrm{wall}}$`
       (since `$u = w = 0$` at walls).
    5. Apply the influence matrix
       `$\alpha = -M^{-1} d_{\mathrm{wall}}$`.
    6. Assemble the corrected pressure and all three corrected
       velocity components via Helmholtz linearity, with no
       further Helmholtz solves:

       - `$p = p_P + \alpha_1 p_1 + \alpha_2 p_2$`
       - `$v = v_{arb} + \alpha_1 v_1 + \alpha_2 v_2$`
       - `$u = u_{arb} - i k_x \Delta q$`
       - `$w = w_{arb} - i k_z \Delta q$`

       where `$\Delta q = \alpha_1 q_1 + \alpha_2 q_2$` and
       `$q_i = H_k^{-1} p_i$` (precomputed), using the
       factorisation `$u^{(i)} = -i k_x q_i$`,
       `$w^{(i)} = -i k_z q_i$` (the scalar `$-i k_x$`,
       `$-i k_z$` commute with `$H_k^{-1}$` per mode).
    7. Zero the mean-mode wall-normal velocity `$v$`.
       Continuity `$\partial v / \partial y = 0$` plus
       no-slip at both walls forces `$v \equiv 0$` there;
       the projection prevents accumulation of numerical
       noise from the Helmholtz RHS.
    8. *(optional)* If ``constant_bulk_velocity``, zero the
       mean-mode perturbation bulk velocity in the streamwise
       direction `$(\cos\theta, 0, \sin\theta)$`.
    9. *(optional)* If ``block_mean_spanwise_velocity``, zero
       the mean-mode perturbation bulk velocity in the
       spanwise direction `$(-\sin\theta, 0, \cos\theta)$`.

    Steps 7--9 are orthogonal projections and do not
    interfere; all mean-mode projections and writes go
    through ``mean_mask``.  Padding modes need no writes:
    their fields are identically zero (the forward FFT
    re-zeroes the padding slots on every evaluation), their
    placeholder-wavenumber operators are regular, and the
    IMM corrections vanish there.

    Mathematical equivalence
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The IMM is a **Schur-complement (capacitance-matrix)
    reduction**.  The coupled pressure--velocity system has a
    `$2 \times 2$` block structure with interior unknowns
    (`$I$`) and boundary unknowns (`$B$`).  The influence
    matrix `$M$` is the Schur complement
    `$S = A_{BB} - A_{BI}\,A_{II}^{-1}\,A_{IB}$`; the
    homogeneous data (``p1, p2, v1, v2, q1, q2``) are the
    columns of `$A_{II}^{-1}\,A_{IB}$`.  The correction
    (stage 6) is a **rank-2 low-rank update** to the particular
    solution -- the same algebraic structure as the **Woodbury
    matrix identity** applied to boundary conditions.  The
    bulk-velocity correction (step 8) is a **rank-1
    Sherman--Morrison update**.  Cylindrical: same structure
    with a `$1 \times 1$` Schur complement (one wall at
    `$r = 1$`).

    The discrete continuity this scheme does *not* deliver -- the
    reason it is the legacy path rather than the default -- is
    documented on
    :func:`~dnsjax.geometries.wall_bounded.cartesian._imm_iteration`.
    """
    c = params.step.implicitness
    dt = flow_.dt
    nu = 1.0 / params.phys.re

    u_n, v_n, w_n = velocity_n[0], velocity_n[1], velocity_n[2]
    Nu_n, Nv_n, Nw_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    Nu_j, Nv_j, Nw_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    mean_mask = fourier_.mean_mask

    # Horizontal spectral-derivative factors, reused across every stage.
    ikx = 1j * fourier_.kx
    ikz = 1j * fourier_.kz

    # Batch the three D1 y-derivatives into one GEMM, stacked y-leading
    # (N_y, 3, ...) so the contraction is transpose-free; unstack to 3-d.
    dy_stack = apply_y_matrix(
        flow_.D1, jnp.stack([v_n, Nv_j, Nv_n], axis=1), component_axis=1
    )
    dy_v_n, dy_Nv_j, dy_Nv_n = dy_stack[:, 0], dy_stack[:, 1], dy_stack[:, 2]

    # d_hat^n (discrete divergence at time n; ~0 after first step).
    # ``dnsjax.analysis`` mirrors this operator; changing it here
    # means changing ``snapshot_ops.divergence`` and the
    # transcription in ``tests/test_snapshot_export.py``
    # (``_solver_divergence``), which pins the two together.
    d_hat_n = ikx * u_n + dy_v_n + ikz * w_n

    # Stage 1: interior pressure Poisson RHS.
    div_Nj = ikx * Nu_j + dy_Nv_j + ikz * Nw_j
    div_Nn = ikx * Nu_n + dy_Nv_n + ikz * Nw_n

    Lk_d = _lk_matvec(d_hat_n, flow_, fourier_)

    f_hat = d_hat_n / dt + c * div_Nj + (1 - c) * div_Nn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure with ZERO Neumann BCs.
    f_hat_P = f_hat.at[0].set(0.0).at[-1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: Helmholtz solves for all three velocity components
    # against the particular pressure p_P (zero Dirichlet BCs).  The
    # three components share the same :math:`H_k` operator per mode,
    # so the explicit matvec, the wall-row zeroing, and the final
    # solve are all batched over the component axis — one kernel
    # launch each instead of three sequential ones.
    #
    # This Hk path stays **component-leading** (unlike the y-leading
    # curl/divergence matvecs above): it has a single D2 GEMM (the
    # vmapped _hk_minus_matvec), and velocity_n / nonlin_j / nonlin_n
    # all arrive component-leading, so a y-leading conversion would add
    # three transposes to remove the one matvec's two -- a net loss.
    # (Cylindrical/annular convert theirs -- several batched matvecs to
    # amortise; see those modules.)
    dx_pP = ikx * pP
    dy_pP = apply_y_matrix(flow_.D1, pP)
    dz_pP = ikz * pP
    grad_pP = jnp.stack([dx_pP, dy_pP, dz_pP])  # (3, Ny, Nkz, Nkx)

    Hk_minus_stack = jax.vmap(
        _hk_minus_matvec,
        in_axes=(0, None, None),
    )(velocity_n, flow_, fourier_)

    R_stack = Hk_minus_stack - grad_pP + c * nonlin_j + (1 - c) * nonlin_n
    R_stack = R_stack.at[:, 0].set(0.0).at[:, -1].set(0.0)

    # Zero v-component RHS at the mean mode so the Helmholtz
    # solve itself returns v = 0 there.
    R_stack = R_stack.at[1].set(jnp.where(mean_mask, 0.0, R_stack[1]))

    arb_stack = flow_.Hk_op.solve(R_stack)
    u_arb, v_arb, w_arb = arb_stack[0], arb_stack[1], arb_stack[2]

    # Stage 4: wall divergence residual. At walls u=w=0 (no-slip),
    # so div u|_wall = D1 v|_wall.
    d_wall = jnp.einsum("bj, jzx -> zxb", flow_.D1_bnd, v_arb)

    # Mean-mode top-wall residual is a pressure gauge; zero it.
    d_wall = d_wall.at[..., 1].set(
        jnp.where(mean_mask[0], 0.0, d_wall[..., 1])
    )

    # Stage 5: influence matrix algebra alpha = -M_inv @ d_wall.
    alpha = -jnp.einsum("zxab, zxb -> zxa", flow_.M_inv, d_wall)
    alpha1 = alpha[..., 0][None]
    alpha2 = alpha[..., 1][None]

    # Stage 6: corrected velocity components via Helmholtz
    # linearity — no additional Helmholtz solves.  The corrected
    # pressure (pP + alpha1 p1 + alpha2 p2) is never assembled:
    # only velocity is stepped.
    v_new = v_arb + alpha1 * flow_.v1 + alpha2 * flow_.v2

    # Horizontal corrections factor through the scalar potential Δq,
    # since u^(i) = -ikx q_i and w^(i) = -ikz q_i (the -ikx, -ikz
    # scalar factors commute with Hk linearity per mode).
    q_new = alpha1 * flow_.q1 + alpha2 * flow_.q2

    # Stage 7: zero mean-mode wall-normal velocity.
    v_new = jnp.where(mean_mask, 0.0, v_new)
    u_new = u_arb - ikx * q_new
    w_new = w_arb - ikz * q_new

    u_new, w_new, aux = _apply_bulk_corrections(u_new, w_new, mean_mask, flow_)

    velocity_new = jnp.array([u_new, v_new, w_new])

    correction = velocity_new - velocity_j

    return velocity_new, correction, aux
