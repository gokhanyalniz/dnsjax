"""Plane Couette flow: wall-bounded shear between two moving plates.

The base flow is `$U(y) = y$` on the Chebyshev-Gauss-Lobatto grid
`$y \\in [-1, 1]$`, with walls moving at `$\\pm 1$`.
"""

from dataclasses import dataclass, field

import jax
import numpy as np
from jax import Array, jit
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ..bench import timer
from ..fd import build_diff_matrices
from ..geometries.cartesian import (
    DenseJAXSolver,
    Fourier,
    IMMChunker,
    LineaxBandedSolver,
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
    D2_bnd: Array = field(init=False)
    Lk: Array = field(init=False)
    Lk_solver: DenseJAXSolver | LineaxBandedSolver = field(init=False)
    Hk: Array = field(init=False)
    Hk_solver: DenseJAXSolver | LineaxBandedSolver = field(init=False)
    Hk_minus: Array = field(init=False)
    p1: Array = field(init=False)
    p2: Array = field(init=False)
    M_inv: Array = field(init=False)
    k2: Array = field(init=False)

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
            jnp.zeros((3, params.res.ny), dtype=sharding.float_type)
            .at[0]
            .set(base_flow_np)[:, :, None, None]
        )
        self.curl_base_flow = (
            jnp.zeros((3, params.res.ny), dtype=sharding.float_type)
            .at[2]
            .set(-dy_base_flow_np)[:, :, None, None]
        )
        self.nonlin_base_flow = (
            jnp.zeros((3, params.res.ny), dtype=sharding.float_type)
            .at[1]
            .set(base_flow_np * dy_base_flow_np)[:, :, None, None]
        )

        D1, D2 = build_diff_matrices(np.array(self.ys), params.res.fd_order)
        self.D1 = jax.device_put(jnp.array(D1), sharding.no_shard)
        self.D2 = jax.device_put(jnp.array(D2), sharding.no_shard)
        self.D2_bnd = jax.device_put(
            jnp.array(D2[[0, -1], :]), sharding.no_shard
        )

        kx_global = np.array(fourier.kx.reshape(-1))
        kz_global = np.array(fourier.kz.reshape(-1))
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
        )

        self.Lk = jax.make_array_from_callback(
            (Nkz, Nkx, Ny, Ny),
            sharding.spec_dy_op_shard,
            lambda idx: chunker.get_chunk(idx, "Lk"),
        )
        self.Hk = jax.make_array_from_callback(
            (Nkz, Nkx, Ny, Ny),
            sharding.spec_dy_op_shard,
            lambda idx: chunker.get_chunk(idx, "Hk"),
        )
        self.Hk_minus = jax.make_array_from_callback(
            (Nkz, Nkx, Ny, Ny),
            sharding.spec_dy_op_shard,
            lambda idx: chunker.get_chunk(idx, "Hk_minus"),
        )
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
        self.M_inv = jax.make_array_from_callback(
            (Nkz, Nkx, 2, 2),
            sharding.spec_dy_op_shard,
            lambda idx: chunker.get_chunk(idx, "M_inv"),
        )
        self.k2 = jax.make_array_from_callback(
            (Nkz, Nkx),
            sharding.spec_k2_op_shard,
            lambda idx: chunker.get_chunk(idx, "k2"),
        )

        if params.solver.use_lineax:
            try:
                import lineax as lx

                SolverClass = lambda m: LineaxBandedSolver(
                    m, params.res.fd_order + 1, params.res.fd_order + 1
                )
            except ImportError:
                raise ImportError(
                    "Lineax is not installed! Use 'pip install lineax' or set use_lineax=False."
                )
        else:
            SolverClass = DenseJAXSolver

        self.Lk_solver = SolverClass(self.Lk)
        self.Hk_solver = SolverClass(self.Hk)


flow: PlaneCouetteFlow = PlaneCouetteFlow()


def _compute_static_pressure(velocity_spec: Array) -> Array:
    """Solve the continuous pressure Poisson equation
    on the un-advanced snapshot.

    1. `$\\nabla^2 p = \\nabla \\cdot \\mathbf{N}$`
    2. `$\\partial p/\\partial y = N_v + \\nu \\frac{\\partial^2 v}{\\partial y^2}$`
       at boundaries
    """
    nonlin = get_nonlin(
        velocity_spec,
        flow.base_flow,
        flow.curl_base_flow,
        flow.nonlin_base_flow,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier, flow),
    )

    u, v, w = velocity_spec
    Nu, Nv, Nw = nonlin

    dy_Nv = jnp.einsum("ij, zxj -> zxi", flow.D1, Nv)
    div_N = 1j * fourier.kx * Nu + dy_Nv + 1j * fourier.kz * Nw

    # Pressure Neumann BC: only wall values are needed
    D2_v_bnd = jnp.einsum("bj, zxj -> zxb", flow.D2_bnd, v)
    g_0 = Nv[..., 0] + D2_v_bnd[..., 0] / params.phys.re
    g_1 = Nv[..., -1] + D2_v_bnd[..., 1] / params.phys.re

    # Solve particular pressure
    f_P = div_N.at[..., 0].set(0.0).at[..., -1].set(0.0)
    pP = flow.Lk_solver.solve(f_P)

    # Calculate continuous boundary mismatch
    r_bot = -flow.k2 * pP[..., 0] + g_0
    r_top = -flow.k2 * pP[..., -1] + g_1
    r = jnp.stack([r_bot, r_top], axis=-1)

    # Apply influence matrix algebra mapping constraints
    alpha = jnp.einsum("zxab, zxb -> zxa", flow.M_inv, r)
    alpha1 = alpha[..., 0][..., None]
    alpha2 = alpha[..., 1][..., None]

    p_new = pP + alpha1 * flow.p1 + alpha2 * flow.p2

    return p_new


def init_state(snapshot: str | None) -> tuple[Array, Array]:
    """Initialize the flow state (velocity_spec, pressure_spec)."""
    if params.init.start_from_laminar:
        velocity_spec = jnp.zeros(
            shape=(3, *sharding.spec_shape),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
    elif snapshot is not None:
        snapshot_arr = jnp.load(snapshot)["velocity_phys"].astype(
            sharding.float_type
        )
        velocity_phys = jax.device_put(
            snapshot_arr, sharding.phys_vector_shard
        )
        velocity_phys = velocity_phys.at[...].subtract(flow.base_flow)
        velocity_spec = phys_to_spec_2d(velocity_phys)
    else:
        sharding.print("Provide an initial condition.")
        sharding.exit(code=1)

    pressure_spec = jax.device_put(
        _compute_static_pressure(velocity_spec),
        sharding.spec_scalar_shard,
    )

    return velocity_spec, pressure_spec


def _curl_fn(
    state: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> Array:
    """Spectral curl with 1D FD in y and spectral derivatives in x and z."""
    u, v, w = state

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
    state: tuple[Array, Array],
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> Array:
    """Evaluate non-linear RHS terms."""
    velocity_spec, _ = state
    nonlin = get_nonlin(
        velocity_spec,
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


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    pressure_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> tuple[tuple[Array, Array], Array]:
    """Openpipeflow fractional-step algorithm (Section 5.2).

    Four stages:
    1. Solve Helmholtz for v (Dirichlet BCs).
    2. Build pressure Poisson RHS with Neumann BCs
       from the y-momentum equation at the walls.
    3. Solve pressure via the influence-matrix method
       (particular solution + homogeneous correction).
    4. Solve Helmholtz for u and w.
    """
    c = params.step.implicitness

    u_n, v_n, w_n = velocity_n
    Nu_n, Nv_n, Nw_n = nonlin_n
    Nu_j, Nv_j, Nw_j = nonlin_j

    # Pre-calculate d_hat^n
    dy_v_n = jnp.einsum("ij, zxj -> zxi", flow_.D1, v_n)
    d_hat_n = 1j * fourier_.kx * u_n + dy_v_n + 1j * fourier_.kz * w_n

    # Stage 1: Solve for v^{(j+1)}
    dy_p_j = jnp.einsum("ij, zxj -> zxi", flow_.D1, pressure_j)
    Rv = _matvec_4d(flow_.Hk_minus, v_n) - dy_p_j + c * Nv_j + (1 - c) * Nv_n
    v_new = flow_.Hk_solver.solve(Rv)

    # Stage 2: Residuals and RHS for pressure
    dy_Nv_j = jnp.einsum("ij, zxj -> zxi", flow_.D1, Nv_j)
    dy_Nv_n = jnp.einsum("ij, zxj -> zxi", flow_.D1, Nv_n)
    div_Nj = 1j * fourier_.kx * Nu_j + dy_Nv_j + 1j * fourier_.kz * Nw_j
    div_Nn = 1j * fourier_.kx * Nu_n + dy_Nv_n + 1j * fourier_.kz * Nw_n

    Lk_d = _matvec_4d(flow_.Lk, d_hat_n)

    f_hat = (
        d_hat_n / params.step.dt
        + c * div_Nj
        + (1 - c) * div_Nn
        + (1 - c) * (1.0 / params.phys.re) * Lk_d
    )

    # Pressure Neumann BC: only wall values are needed,
    # so use boundary rows of D2 instead of full operators.
    D2_v_new_bnd = jnp.einsum("bj, zxj -> zxb", flow_.D2_bnd, v_new)
    D2_v_n_bnd = jnp.einsum("bj, zxj -> zxb", flow_.D2_bnd, v_n)
    nu = 1.0 / params.phys.re
    g_0 = (
        v_n[..., 0] / params.step.dt
        + c * Nv_j[..., 0]
        + (1 - c) * Nv_n[..., 0]
        + c * nu * D2_v_new_bnd[..., 0]
        + (1 - c) * nu * D2_v_n_bnd[..., 0]
    )
    g_1 = (
        v_n[..., -1] / params.step.dt
        + c * Nv_j[..., -1]
        + (1 - c) * Nv_n[..., -1]
        + c * nu * D2_v_new_bnd[..., 1]
        + (1 - c) * nu * D2_v_n_bnd[..., 1]
    )

    # Stage 3: Solve pressure via influence matrix
    f_hat_P = f_hat.at[..., 0].set(0.0).at[..., -1].set(0.0)
    pP = flow_.Lk_solver.solve(f_hat_P)  # particular solution

    r_bot = -flow_.k2 * pP[..., 0] + g_0
    r_top = -flow_.k2 * pP[..., -1] + g_1
    r = jnp.stack([r_bot, r_top], axis=-1)

    # IMM homogeneous solution coefficients
    alpha = jnp.einsum("zxab, zxb -> zxa", flow_.M_inv, r)
    alpha1 = alpha[..., 0][..., None]
    alpha2 = alpha[..., 1][..., None]

    p_new = pP + alpha1 * flow_.p1 + alpha2 * flow_.p2

    # Stage 4: Solve for u and w
    dx_p_new = 1j * fourier_.kx * p_new
    dz_p_new = 1j * fourier_.kz * p_new

    Ru = _matvec_4d(flow_.Hk_minus, u_n) - dx_p_new + c * Nu_j + (1 - c) * Nu_n
    Rw = _matvec_4d(flow_.Hk_minus, w_n) - dz_p_new + c * Nw_j + (1 - c) * Nw_n

    u_new = flow_.Hk_solver.solve(Ru)
    w_new = flow_.Hk_solver.solve(Rw)

    velocity_new = jnp.array([u_new, v_new, w_new])

    correction = velocity_new - velocity_j

    state_new = (velocity_new, p_new)
    return state_new, correction


def _predict(
    state: tuple[Array, Array],
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> tuple[Array, Array]:
    """Euler predictor step mapping j=0 over Openpipeflow IMM."""
    velocity_n, pressure_n = state
    nonlin_n = rhs_no_lapl

    prediction_state, _ = _imm_iteration(
        velocity_n, velocity_n, pressure_n, nonlin_n, nonlin_n, fourier_, flow_
    )
    return prediction_state


def _correct(
    state_prev: tuple[Array, Array],
    prediction_state: tuple[Array, Array],
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: PlaneCouetteFlow,
) -> tuple[tuple[Array, Array], Array]:
    """Crank-Nicolson corrector mapping j>0 over Openpipeflow IMM."""
    velocity_n, _ = state_prev
    velocity_j, pressure_j = prediction_state

    nonlin_n = rhs_prev
    nonlin_j = rhs_next

    prediction_state_new, correction = _imm_iteration(
        velocity_n, velocity_j, pressure_j, nonlin_n, nonlin_j, fourier_, flow_
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
    state: tuple[Array, Array],
) -> tuple[tuple[Array, Array], Array, Array]:
    """Predictor-corrector step with bound singletons."""
    return _predict_and_correct_jit(state, fourier, flow)


def iterate_correction(
    state_prev: tuple[Array, Array],
    prediction: tuple[Array, Array],
    rhs_prev: Array,
):
    """One corrector iteration with bound singletons."""
    return _iterate_correction_jit(
        state_prev, prediction, rhs_prev, fourier, flow
    )


@timer("velocity/correct_velocity")
@jit(donate_argnums=0)
def correct_velocity(
    state: tuple[Array, Array],
) -> tuple[tuple[Array, Array], dict[str, Array | None] | None]:
    """Pass-through, IMM satisfies strict numerical continuity inherently."""
    norm_corrections = None if not params.debug.measure_corrections else {}
    return state, norm_corrections


@jit
def get_stats(state: tuple[Array, Array]) -> dict[str, Array]:
    """Diagnostic statistics placeholder."""
    return {}


phys_to_spec = phys_to_spec_2d
spec_to_phys = spec_to_phys_2d
