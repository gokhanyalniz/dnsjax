"""Plane Couette flow: wall-bounded shear between two moving plates.

The base flow is `$U(y) = y$` where `$y \\in [-1, 1]$`,
with walls moving at `$\\pm 1$`.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from jax import Array, jit
from jax import numpy as jnp

from ..bench import timer
from ..fd import build_diff_matrices
from ..geometries.cartesian import (
    DenseJAXSolver,
    Fourier,
    IMMChunker,
    PerModeBandedOperator,
    fourier,
    get_norm2,
)
from ..operators import (
    phys_to_spec_2d,
    spec_to_phys_2d,
)
from ..parameters import params
from ..rhs import get_nonlin
from ..sharding import register_dataclass_pytree, sharding
from ..timestep import make_stepper


@register_dataclass_pytree
@dataclass
class PlaneCouetteFlow:
    """Precomputed data for plane Couette flow."""

    ys: Array = field(init=False)
    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    nonlin_base_flow: Array = field(init=False)
    D1: Array = field(init=False)
    D2: Array = field(init=False)
    D1_bnd: Array = field(init=False)
    D2_bnd: Array = field(init=False)
    Lk_op: DenseJAXSolver | PerModeBandedOperator = field(init=False)
    Hk_op: DenseJAXSolver | PerModeBandedOperator = field(init=False)
    # ``Lk`` and ``Hk_minus`` are only populated under
    # ``params.solver.backend == "dense"`` for the matrix-free matvecs
    # at :func:`_imm_iteration`.  Under ``"banded"`` they stay ``None``
    # and the matvecs are computed on the fly from ``D2``.
    Lk: Array | None = field(init=False, default=None)
    Hk_minus: Array | None = field(init=False, default=None)
    p1: Array = field(init=False)
    p2: Array = field(init=False)
    v1: Array = field(init=False)
    v2: Array = field(init=False)
    q1: Array = field(init=False)
    q2: Array = field(init=False)
    M_inv: Array = field(init=False)

    def __post_init__(self) -> None:
        """Build CGL grid, base flow, and IMM operators.

        Constructs the Chebyshev-Gauss-Lobatto grid for
        the wall-normal coordinate ``y`` in ``[-1, 1]``,
        the laminar base flow ``U(y) = y`` and its derived
        quantities, FD matrices D1 and D2, and all per-mode
        IMM operators via ``IMMChunker``.
        """
        self.ys = -jnp.cos(
            jnp.arange(params.res.ny, dtype=sharding.float_type)
            * jnp.pi
            / (params.res.ny - 1)
        )

        base_flow_np = self.ys.copy()
        dy_base_flow_np = jnp.ones(params.res.ny, dtype=sharding.float_type)

        self.base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[0]
            .set(base_flow_np)[:, :, None, None]
        )
        self.curl_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[2]
            .set(-dy_base_flow_np)[:, :, None, None]
        )
        self.nonlin_base_flow = (
            jnp.zeros(
                (3, params.res.ny),
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            .at[1]
            .set(base_flow_np * dy_base_flow_np)[:, :, None, None]
        )

        D1, D2 = build_diff_matrices(np.array(self.ys), params.res.fd_order)
        self.D1 = jax.device_put(jnp.array(D1), sharding.no_shard)
        self.D2 = jax.device_put(jnp.array(D2), sharding.no_shard)
        self.D1_bnd = jax.device_put(
            jnp.array(D1[[0, -1], :]), sharding.no_shard
        )
        self.D2_bnd = jax.device_put(
            jnp.array(D2[[0, -1], :]), sharding.no_shard
        )

        kx_global = fourier.kx_global
        kz_global = fourier.kz_global
        Nkz = len(kz_global)
        Nkx = len(kx_global)
        Ny = params.res.ny

        chunker = IMMChunker(
            ys_arr=np.array(self.ys),
            kx_global=kx_global,
            kz_global=kz_global,
            p=params.res.fd_order,
            dt=params.step.dt,
            c=params.step.implicitness,
            nu=1.0 / params.phys.re,
            D1_arr=D1,
            D2_arr=D2,
            backend=params.solver.backend,
        )

        # Homogeneous IMM data — always needed.
        self.p1 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "p1"),
        )
        self.p2 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "p2"),
        )
        self.v1 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "v1"),
        )
        self.v2 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "v2"),
        )
        self.q1 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "q1"),
        )
        self.q2 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny),
            sharding.spec_imm_corr_shard,
            lambda idx: chunker.get_chunk(idx, "q2"),
        )
        self.M_inv = jax.make_array_from_callback(
            (Nkz, Nkx, 2, 2),
            sharding.spec_dy_op_shard,
            lambda idx: chunker.get_chunk(idx, "M_inv"),
        )

        # Operator caches: LAPACK-packed banded LU or legacy dense.
        if params.solver.backend == "banded":
            p = params.res.fd_order
            ab_rows = 3 * p + 1  # = 2*kl + ku + 1 with kl = ku = p
            Lk_ab = jax.make_array_from_callback(
                (Nkz, Nkx, ab_rows, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Lk_ab"),
            )
            Lk_piv = jax.make_array_from_callback(
                (Nkz, Nkx, Ny),
                sharding.spec_imm_corr_shard,
                lambda idx: chunker.get_chunk(idx, "Lk_piv"),
            )
            Hk_ab = jax.make_array_from_callback(
                (Nkz, Nkx, ab_rows, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Hk_ab"),
            )
            Hk_piv = jax.make_array_from_callback(
                (Nkz, Nkx, Ny),
                sharding.spec_imm_corr_shard,
                lambda idx: chunker.get_chunk(idx, "Hk_piv"),
            )
            self.Lk_op = PerModeBandedOperator(ab=Lk_ab, ipiv=Lk_piv)
            self.Hk_op = PerModeBandedOperator(ab=Hk_ab, ipiv=Hk_piv)
        else:
            self.Lk = jax.make_array_from_callback(
                (Nkz, Nkx, Ny, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Lk"),
            )
            Hk = jax.make_array_from_callback(
                (Nkz, Nkx, Ny, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Hk"),
            )
            self.Hk_minus = jax.make_array_from_callback(
                (Nkz, Nkx, Ny, Ny),
                sharding.spec_dy_op_shard,
                lambda idx: chunker.get_chunk(idx, "Hk_minus"),
            )
            self.Lk_op = DenseJAXSolver(self.Lk)
            self.Hk_op = DenseJAXSolver(Hk)


flow: PlaneCouetteFlow = PlaneCouetteFlow()


def init_state(snapshot: str | None) -> Array:
    """Initialize the flow state (velocity_spec)."""
    if params.init.start_from_laminar:
        velocity_spec = jnp.zeros(
            shape=(3, *sharding.spec_shape),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
    elif snapshot is not None:
        snapshot_arr = jnp.load(snapshot)["velocity_phys_nonexpanded"].astype(
            sharding.float_type
        )
        velocity_phys = jax.device_put(
            snapshot_arr, sharding.phys_vector_shard
        )
        # velocity_phys = velocity_phys.at[...].subtract(flow.base_flow)
        velocity_spec = phys_to_spec_2d(velocity_phys)
        velocity_phys = spec_to_phys_2d(velocity_spec)

    else:
        sharding.print("Provide an initial condition.")
        sharding.exit(code=1)

    return velocity_spec


def _curl_fn(
    state: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> Array:
    """Spectral curl with 1D FD in y and spectral derivatives in x and z."""
    u, v, w = state[0], state[1], state[2]

    dy_u = jnp.einsum("ij, zxj -> zxi", flow_.D1, u)
    dy_w = jnp.einsum("ij, zxj -> zxi", flow_.D1, w)

    dx_v = 1j * fourier_.kx * v
    dz_v = 1j * fourier_.kz * v
    dx_w = 1j * fourier_.kx * w
    dz_u = 1j * fourier_.kz * u

    omega_x = dy_w - dz_v
    omega_y = dz_u - dx_w
    omega_z = dx_v - dy_u

    return jnp.array([omega_x, omega_y, omega_z])


def _get_rhs(
    state: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> Array:
    """Evaluate non-linear RHS terms."""
    nonlin = get_nonlin(
        state,
        flow_.base_flow,
        flow_.curl_base_flow,
        flow_.nonlin_base_flow,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier_, flow_),
    )
    return nonlin


def _matvec_4d(op: Array, x: Array) -> Array:
    """Apply operator of shape (Nkz, Nkx, Ny, Ny)
    to field of shape (Nkz, Nkx, Ny).
    """
    return jnp.einsum("zxij, zxj -> zxi", op, x)


def _lk_matvec(
    u: Array,
    D2: Array,
    D1_bnd: Array,
    k2: Array,
    k2_is_zero: Array,
) -> Array:
    """Apply `$L_k u$` for the Neumann-BC pressure Poisson operator.

    Matrix-free evaluation that avoids storing the per-mode
    ``(Nkz, Nkx, Ny, Ny)`` operator.  The interior of the output is
    `$D_2 u - k^2 u$`; the wall rows use `$D_1$` to encode Neumann
    BCs, except for the `$k^2 = 0$` mean mode where row 0 pins
    `$p_0 = 0$` (matching :func:`build_Lk_neumann`).

    Parameters
    ----------
    u:
        Field, shape ``(Nkz, Nkx, Ny)``.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    D1_bnd:
        Boundary rows `$D_1[0,:]$`, `$D_1[-1,:]$`, shape ``(2, Ny)``.
    k2:
        Squared horizontal wavenumber, broadcasting as
        ``(Nkz, Nkx, 1)``.
    k2_is_zero:
        Boolean mask ``k2 == 0``, same shape as *k2*.

    Returns
    -------
    :
        ``Lk @ u`` with the same shape and dtype as *u*.
    """
    D2u = jnp.einsum("ij, zxj -> zxi", D2, u)
    out = D2u - k2 * u
    bot_neumann = jnp.einsum("j, zxj -> zx", D1_bnd[0], u)
    bot = jnp.where(k2_is_zero[..., 0], u[..., 0], bot_neumann)
    top = jnp.einsum("j, zxj -> zx", D1_bnd[-1], u)
    return out.at[..., 0].set(bot).at[..., -1].set(top)


def _hk_minus_matvec(
    u: Array, D2: Array, k2: Array, dt: float, c: float, nu: float
) -> Array:
    """Apply `$H_k^- u$` for the explicit-side Helmholtz operator.

    Matrix-free evaluation of ``flow_.Hk_minus @ u``:
    `$\\tfrac{1}{\\Delta t} u + (1 - c) \\nu (D_2 u - k^2 u)$` in the
    interior, with identity wall rows (`$u|_\\text{wall}$` unchanged).

    Parameters
    ----------
    u:
        Field, shape ``(Nkz, Nkx, Ny)``.
    D2:
        Second-derivative matrix, shape ``(Ny, Ny)``.
    k2:
        Squared horizontal wavenumber, broadcasting as
        ``(Nkz, Nkx, 1)``.
    dt:
        Time step.
    c:
        Implicitness parameter.
    nu:
        Kinematic viscosity `$1/\\mathrm{Re}$`.

    Returns
    -------
    :
        ``Hk_minus @ u`` with the same shape and dtype as *u*.
    """
    D2u = jnp.einsum("ij, zxj -> zxi", D2, u)
    out = (1.0 / dt) * u + (1.0 - c) * nu * (D2u - k2 * u)
    return out.at[..., 0].set(u[..., 0]).at[..., -1].set(u[..., -1])


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> tuple[Array, Array]:
    """Kleiser-Schumann influence-matrix method.

    The y-momentum equation supplies only the *interior* Poisson
    equation for pressure; the wall BC is determined indirectly by
    enforcing continuity `$\\nabla \\cdot u = 0$` at the walls.

    Six stages:
    1. Build the interior Poisson RHS from divergence of momentum.
    2. Solve Poisson for the particular pressure `$p_P$` with
       arbitrary (zero) Neumann BCs.
    3. Solve Helmholtz for all three particular velocity components
       `$u_{arb}, v_{arb}, w_{arb}$` against `$p_P$` (zero Dirichlet
       BCs).
    4. Compute wall divergence residual `$d_{\\mathrm{wall}} = (D_1
       v_{arb})|_{\\mathrm{wall}}$` (since `$u = w = 0$` at walls).
    5. Apply the influence matrix `$\\alpha = -M^{-1} d_{\\mathrm{wall}}$`.
    6. Assemble the corrected pressure and all three corrected
       velocity components via Helmholtz linearity, with no further
       Helmholtz solves:

       - `$p = p_P + \\alpha_1 p_1 + \\alpha_2 p_2$`
       - `$v = v_{arb} + \\alpha_1 v_1 + \\alpha_2 v_2$`
       - `$u = u_{arb} - i k_x \\Delta q$`
       - `$w = w_{arb} - i k_z \\Delta q$`

       where `$\\Delta q = \\alpha_1 q_1 + \\alpha_2 q_2$` and
       `$q_i = H_k^{-1} p_i$` (precomputed), using the factorisation
       `$u^{(i)} = -i k_x q_i$`, `$w^{(i)} = -i k_z q_i$` (the scalar
       `$-i k_x$`, `$-i k_z$` commute with `$H_k^{-1}$` per mode).
    """
    c = params.step.implicitness
    dt = params.step.dt
    nu = 1.0 / params.phys.re

    u_n, v_n, w_n = velocity_n[0], velocity_n[1], velocity_n[2]
    Nu_n, Nv_n, Nw_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    Nu_j, Nv_j, Nw_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    # Squared horizontal wavenumber, broadcastable to (Nkz, Nkx, Ny).
    k2 = fourier_.k2
    k2_is_zero = fourier_.k2_is_zero

    # d_hat^n (discrete divergence at time n; ~0 after first step).
    dy_v_n = jnp.einsum("ij, zxj -> zxi", flow_.D1, v_n)
    d_hat_n = 1j * fourier_.kx * u_n + dy_v_n + 1j * fourier_.kz * w_n

    # Stage 1: interior pressure Poisson RHS.
    dy_Nv_j = jnp.einsum("ij, zxj -> zxi", flow_.D1, Nv_j)
    dy_Nv_n = jnp.einsum("ij, zxj -> zxi", flow_.D1, Nv_n)
    div_Nj = 1j * fourier_.kx * Nu_j + dy_Nv_j + 1j * fourier_.kz * Nw_j
    div_Nn = 1j * fourier_.kx * Nu_n + dy_Nv_n + 1j * fourier_.kz * Nw_n

    if params.solver.backend == "banded":
        Lk_d = _lk_matvec(d_hat_n, flow_.D2, flow_.D1_bnd, k2, k2_is_zero)
    else:
        Lk_d = _matvec_4d(flow_.Lk, d_hat_n)

    f_hat = d_hat_n / dt + c * div_Nj + (1 - c) * div_Nn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure with ZERO Neumann BCs.
    f_hat_P = f_hat.at[..., 0].set(0.0).at[..., -1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: Helmholtz solves for all three velocity components
    # against the particular pressure p_P (zero Dirichlet BCs).
    dx_pP = 1j * fourier_.kx * pP
    dy_pP = jnp.einsum("ij, zxj -> zxi", flow_.D1, pP)
    dz_pP = 1j * fourier_.kz * pP

    if params.solver.backend == "banded":
        Hku_n = _hk_minus_matvec(u_n, flow_.D2, k2, dt, c, nu)
        Hkv_n = _hk_minus_matvec(v_n, flow_.D2, k2, dt, c, nu)
        Hkw_n = _hk_minus_matvec(w_n, flow_.D2, k2, dt, c, nu)
    else:
        Hku_n = _matvec_4d(flow_.Hk_minus, u_n)
        Hkv_n = _matvec_4d(flow_.Hk_minus, v_n)
        Hkw_n = _matvec_4d(flow_.Hk_minus, w_n)

    Ru = Hku_n - dx_pP + c * Nu_j + (1 - c) * Nu_n
    Rv = Hkv_n - dy_pP + c * Nv_j + (1 - c) * Nv_n
    Rw = Hkw_n - dz_pP + c * Nw_j + (1 - c) * Nw_n

    Ru = Ru.at[..., 0].set(0.0).at[..., -1].set(0.0)
    Rv = Rv.at[..., 0].set(0.0).at[..., -1].set(0.0)
    Rw = Rw.at[..., 0].set(0.0).at[..., -1].set(0.0)

    u_arb = flow_.Hk_op.solve(Ru)
    v_arb = flow_.Hk_op.solve(Rv)
    w_arb = flow_.Hk_op.solve(Rw)

    # Stage 4: wall divergence residual. At walls u=w=0 (no-slip),
    # so div u|_wall = D1 v|_wall.
    d_wall = jnp.einsum("bj, zxj -> zxb", flow_.D1_bnd, v_arb)

    # Mean mode (k²=0) bottom-wall residual is a pressure gauge; zero it.
    d_wall = d_wall.at[..., 0].set(
        jnp.where(k2_is_zero[..., 0], 0.0, d_wall[..., 0])
    )

    # Stage 5: influence matrix algebra alpha = -M_inv @ d_wall.
    alpha = -jnp.einsum("zxab, zxb -> zxa", flow_.M_inv, d_wall)
    alpha1 = alpha[..., 0][..., None]
    alpha2 = alpha[..., 1][..., None]

    # Stage 6: corrected pressure and all three velocity components
    # via Helmholtz linearity — no additional Helmholtz solves.
    # p_new = pP + alpha1 * flow_.p1 + alpha2 * flow_.p2
    v_new = v_arb + alpha1 * flow_.v1 + alpha2 * flow_.v2

    # Horizontal corrections factor through the scalar potential Δq,
    # since u^(i) = -ikx q_i and w^(i) = -ikz q_i (the -ikx, -ikz
    # scalar factors commute with Hk linearity per mode).
    q_new = alpha1 * flow_.q1 + alpha2 * flow_.q2
    u_new = u_arb - 1j * fourier_.kx * q_new
    w_new = w_arb - 1j * fourier_.kz * q_new

    velocity_new = jnp.array([u_new, v_new, w_new])

    correction = velocity_new - velocity_j

    return velocity_new, correction


def _predict(
    velocity_n: Array,
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> Array:
    """Euler predictor (Willis 2017 j=0) via Kleiser-Schumann IMM."""
    nonlin_n = rhs_no_lapl

    prediction_state, _ = _imm_iteration(
        velocity_n, velocity_n, nonlin_n, nonlin_n, fourier_, flow_
    )
    return prediction_state


def _correct(
    state_prev: Array,
    prediction_state: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector (Willis 2017 j>0) via Kleiser-Schumann IMM."""
    velocity_n = state_prev
    velocity_j = prediction_state

    nonlin_n = rhs_prev
    nonlin_j = rhs_next

    prediction_state_new, correction = _imm_iteration(
        velocity_n, velocity_j, nonlin_n, nonlin_j, fourier_, flow_
    )
    return prediction_state_new, correction


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> Array:
    """L2 convergence norm."""
    return jnp.sqrt(get_norm2(correction, fourier_.k_metric, flow_.ys))


_predict_and_correct_jit, _iterate_correction_jit = make_stepper(
    _get_rhs, _predict, _correct, _norm
)


def predict_and_correct(
    state: Array,
) -> tuple[Array, Array, Array]:
    """Predictor-corrector step with bound singletons."""
    return _predict_and_correct_jit(state, fourier, flow)


def iterate_correction(
    state_prev: Array,
    prediction: Array,
    rhs_prev: Array,
) -> tuple[Array, Array, Array]:
    """One corrector iteration with bound singletons."""
    return _iterate_correction_jit(
        state_prev, prediction, rhs_prev, fourier, flow
    )


@timer("velocity/correct_velocity")
@jit(donate_argnums=0)
def correct_velocity(
    state: Array,
) -> tuple[Array, dict[str, Array | None] | None]:
    """Pass-through, IMM satisfies strict numerical continuity inherently."""
    norm_corrections = None if not params.debug.measure_corrections else {}
    return state, norm_corrections


phys_to_spec = phys_to_spec_2d
spec_to_phys = spec_to_phys_2d


# ── Diagnostic statistics ────────────────────────────────────────────────
#
def get_perturbation_energy(state: Array, fourier_: Any, flow_: Any) -> Array:
    """Perturbation kinetic energy `$E' = \\|\\mathbf{u}'\\|^2 / 2$`."""
    return get_norm2(state, fourier_.k_metric, flow_.ys) / 2


@jit
def _get_stats_jit(
    state: Array, fourier_: Any, flow_: Any
) -> dict[str, Array]:
    """Compute diagnostic statistics: E, I, D, E'."""
    perturbation_energy = get_perturbation_energy(state, fourier_, flow_)
    # input = get_input(state, fourier_, flow_)
    # dissipation = get_dissipation(state, input, fourier_, flow_)
    # energy = get_energy(perturbation_energy, input, fourier_, flow_)

    stats = {
        # "E": energy,
        # "I": input,
        # "D": dissipation,
        "E'": perturbation_energy,
    }

    return stats


@timer("stats")
def get_stats(state: Array) -> dict[str, Array]:
    return _get_stats_jit(state, fourier, flow)
