"""Plane Couette flow: wall-bounded shear between two moving plates.

The base flow is ``U(y) = y`` on the Chebyshev-Gauss-Lobatto grid
``y in [-1, 1]``, with walls moving at ``+/-1``.
"""

from dataclasses import dataclass, field
from functools import partial

import numpy as np
import jax
from jax import Array, jit, vmap
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ..bench import timer
from ..fd import precompute_imm
from ..operators import (
    fourier,
    phys_to_spec_2d,
    spec_to_phys_2d,
)
from ..parameters import padded_res, params
from ..rhs import get_nonlin
from ..sharding import sharding
from ..timestep import make_stepper
from ..velocity import get_norm2


@dataclass
class PlaneCouetteFlow:
    """Precomputed data for plane Couette flow."""

    ys: Array = field(init=False)
    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    nonlin_base_flow: Array = field(init=False)
    D1: Array = field(init=False)
    D2: Array = field(init=False)
    Lk: Array = field(init=False)
    Hk: Array = field(init=False)
    Hk_minus: Array = field(init=False)
    p1: Array = field(init=False)
    p2: Array = field(init=False)
    M_inv: Array = field(init=False)
    k2: Array = field(init=False)

    def __post_init__(self):
        self.ys = -jnp.cos(
            jnp.arange(params.res.ny, dtype=sharding.float_type)
            * jnp.pi
            / (params.res.ny - 1)
        )

        base_flow_np = self.ys.copy()
        dy_base_flow_np = jnp.ones(params.res.ny, dtype=sharding.float_type)
        
        self.base_flow = jnp.zeros((3, params.res.ny), dtype=sharding.float_type).at[0].set(base_flow_np)[:, :, None, None]
        self.curl_base_flow = jnp.zeros((3, params.res.ny), dtype=sharding.float_type).at[2].set(-dy_base_flow_np)[:, :, None, None]
        self.nonlin_base_flow = jnp.zeros((3, params.res.ny), dtype=sharding.float_type).at[1].set(base_flow_np * dy_base_flow_np)[:, :, None, None]

        __imm_numpy = precompute_imm(
            y=np.array(self.ys),
            kx_vals=np.array(fourier.kx.reshape(-1)),
            kz_vals=np.array(fourier.kz.reshape(-1)),
            p=params.res.fd_order,
            dt=params.step.dt,
            c=params.step.implicitness,
            nu=1.0 / params.phys.re,
        )

        shard_4d = P(None, *sharding.axis_names, None, None)
        shard_3d = P(None, *sharding.axis_names, None)
        shard_2d = P(None, *sharding.axis_names)
        
        self.D1 = jax.device_put(jnp.array(__imm_numpy["D1"]), sharding.no_shard)
        self.D2 = jax.device_put(jnp.array(__imm_numpy["D2"]), sharding.no_shard)
        
        self.Lk = jax.device_put(jnp.array(__imm_numpy["Lk"]), shard_4d)
        self.Hk = jax.device_put(jnp.array(__imm_numpy["Hk"]), shard_4d)
        self.Hk_minus = jax.device_put(jnp.array(__imm_numpy["Hk_minus"]), shard_4d)
        
        self.p1 = jax.device_put(jnp.array(__imm_numpy["p1"]), shard_3d)
        self.p2 = jax.device_put(jnp.array(__imm_numpy["p2"]), shard_3d)
        self.M_inv = jax.device_put(jnp.array(__imm_numpy["M_inv"]), shard_4d)
        
        self.k2 = jax.device_put(jnp.array(__imm_numpy["k2"]), shard_2d)


flow: PlaneCouetteFlow = PlaneCouetteFlow()


def _compute_static_pressure(velocity_spec: Array) -> Array:
    """Solve the continuous pressure Poisson equation on the un-advanced snapshot.
    
    1. grad^2 p = div N
    2. dp/dy = N_v + nu * d2v/dy2 at boundaries
    """
    nonlin = get_nonlin(
        velocity_spec,
        flow.base_flow,
        flow.curl_base_flow,
        flow.nonlin_base_flow,
        spec_to_phys_2d,
        phys_to_spec_2d,
        _curl_fn,
    )
    
    u, v, w = velocity_spec
    Nu, Nv, Nw = nonlin
    
    dy_Nv = jnp.tensordot(flow.D1, Nv, axes=(1, 0))
    div_N = 1j * fourier.kx * Nu + dy_Nv + 1j * fourier.kz * Nw
    
    D2_v = jnp.tensordot(flow.D2, v, axes=(1, 0))
    
    g_full = Nv + D2_v / params.phys.re
    g_0 = g_full[0]
    g_1 = g_full[-1]
    
    # Solve particular pressure
    f_P = div_N.at[0].set(0.0).at[-1].set(0.0)
    f_P_b = jnp.transpose(f_P, (1, 2, 0))
    pP_b = jnp.linalg.solve(flow.Lk, f_P_b)
    pP = jnp.transpose(pP_b, (2, 0, 1))
    
    # Calculate continuous boundary mismatch
    r_bot = -flow.k2 * pP[0] + g_0
    r_top = -flow.k2 * pP[-1] + g_1
    r = jnp.stack([r_bot, r_top], axis=-1)
    
    # Apply influence matrix algebra mapping constraints
    alpha = jnp.einsum('zxab, zxb -> zxa', flow.M_inv, r)
    alpha1 = alpha[..., 0][..., None]
    alpha2 = alpha[..., 1][..., None]
    
    p_new_b = jnp.transpose(pP, (1, 2, 0)) + alpha1 * flow.p1 + alpha2 * flow.p2
    p_new = jnp.transpose(p_new_b, (2, 0, 1))
    
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
        snapshot_arr = jnp.load(snapshot)["velocity_phys"].astype(sharding.float_type)
        velocity_phys = jax.device_put(snapshot_arr, sharding.phys_vector_shard)
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


def _curl_fn(state: Array) -> Array:
    """Spectral curl with 1D FD in y and spectral derivatives in x and z."""
    u, v, w = state
    
    dy_u = jnp.tensordot(flow.D1, u, axes=(1, 0))
    dy_w = jnp.tensordot(flow.D1, w, axes=(1, 0))
    
    dx_v = 1j * fourier.kx * v
    dz_v = 1j * fourier.kz * v
    dx_w = 1j * fourier.kx * w
    dz_u = 1j * fourier.kz * u
    
    omega_x = dy_w - dz_v
    omega_y = dz_u - dx_w
    omega_z = dx_v - dy_u
    
    return jnp.array([omega_x, omega_y, omega_z])


def _get_rhs(state: tuple[Array, Array]) -> Array:
    """Evaluate non-linear RHS terms."""
    velocity_spec, _ = state
    nonlin = get_nonlin(
        velocity_spec,
        flow.base_flow,
        flow.curl_base_flow,
        flow.nonlin_base_flow,
        spec_to_phys_2d,
        phys_to_spec_2d,
        _curl_fn,
    )
    return nonlin


def _einsum_4d(op: Array, x: Array) -> Array:
    """Apply operator of shape (Nkz, Nkx, Ny, Ny) to (Ny, Nkz, Nkx)."""
    return jnp.einsum('zxij, jzx -> izx', op, x)


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    pressure_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
) -> tuple[tuple[Array, Array], Array]:
    """Openpipeflow fractional-step algorithm (imm.tex, Section 5.2)."""
    c = params.step.implicitness
    
    u_n, v_n, w_n = velocity_n
    Nu_n, Nv_n, Nw_n = nonlin_n
    Nu_j, Nv_j, Nw_j = nonlin_j
    
    # Pre-calculate d_hat^n
    dy_v_n = jnp.tensordot(flow.D1, v_n, axes=(1, 0))
    d_hat_n = 1j * fourier.kx * u_n + dy_v_n + 1j * fourier.kz * w_n
    
    # Stage 1: Solve for v^{(j+1)}
    dy_p_j = jnp.tensordot(flow.D1, pressure_j, axes=(1, 0))
    Rv = _einsum_4d(flow.Hk_minus, v_n) - dy_p_j + c * Nv_j + (1 - c) * Nv_n
    Rv_b = jnp.transpose(Rv, (1, 2, 0))
    v_new_b = jnp.linalg.solve(flow.Hk, Rv_b)
    v_new = jnp.transpose(v_new_b, (2, 0, 1))
    
    # Stage 2: Residuals and RHS for pressure
    dy_Nv_j = jnp.tensordot(flow.D1, Nv_j, axes=(1, 0))
    dy_Nv_n = jnp.tensordot(flow.D1, Nv_n, axes=(1, 0))
    div_Nj = 1j * fourier.kx * Nu_j + dy_Nv_j + 1j * fourier.kz * Nw_j
    div_Nn = 1j * fourier.kx * Nu_n + dy_Nv_n + 1j * fourier.kz * Nw_n
    
    Lk_d = _einsum_4d(flow.Lk, d_hat_n)
    
    f_hat = (
        d_hat_n / params.step.dt
        + c * div_Nj
        + (1 - c) * div_Nn
        + (1 - c) * (1.0 / params.phys.re) * Lk_d
    )
    
    D2_v_new = jnp.tensordot(flow.D2, v_new, axes=(1, 0))
    Lk_v_n = _einsum_4d(flow.Lk, v_n)
    
    g_full = (
        v_n / params.step.dt 
        + c * Nv_j
        + (1 - c) * Nv_n
        + c * (1.0 / params.phys.re) * D2_v_new
        + (1 - c) * (1.0 / params.phys.re) * Lk_v_n 
    )
    g_0 = g_full[0]
    g_1 = g_full[-1]
    
    # Stage 3: Solve pressure via influence matrix
    f_hat_P = f_hat.at[0].set(0.0).at[-1].set(0.0)
    
    f_hat_P_b = jnp.transpose(f_hat_P, (1, 2, 0))
    pP_b = jnp.linalg.solve(flow.Lk, f_hat_P_b)
    pP = jnp.transpose(pP_b, (2, 0, 1))
    
    r_bot = -flow.k2 * pP[0] + g_0
    r_top = -flow.k2 * pP[-1] + g_1
    r = jnp.stack([r_bot, r_top], axis=-1)
    
    alpha = jnp.einsum('zxab, zxb -> zxa', flow.M_inv, r)
    alpha1 = alpha[..., 0][..., None]
    alpha2 = alpha[..., 1][..., None]
    
    p_new_b = jnp.transpose(pP, (1, 2, 0)) + alpha1 * flow.p1 + alpha2 * flow.p2
    p_new = jnp.transpose(p_new_b, (2, 0, 1))
    
    # Stage 4: Solve for u and w
    dx_p_new = 1j * fourier.kx * p_new
    dz_p_new = 1j * fourier.kz * p_new
    
    Ru = _einsum_4d(flow.Hk_minus, u_n) - dx_p_new + c * Nu_j + (1 - c) * Nu_n
    Rw = _einsum_4d(flow.Hk_minus, w_n) - dz_p_new + c * Nw_j + (1 - c) * Nw_n
    
    Ru_b = jnp.transpose(Ru, (1, 2, 0))
    Rw_b = jnp.transpose(Rw, (1, 2, 0))
    
    u_new_b = jnp.linalg.solve(flow.Hk, Ru_b)
    w_new_b = jnp.linalg.solve(flow.Hk, Rw_b)
    
    u_new = jnp.transpose(u_new_b, (2, 0, 1))
    w_new = jnp.transpose(w_new_b, (2, 0, 1))
    
    velocity_new = jnp.array([u_new, v_new, w_new])
    
    correction = velocity_new - velocity_j
    
    state_new = (velocity_new, p_new)
    return state_new, correction


def _predict(state: tuple[Array, Array], rhs_no_lapl: Array) -> tuple[Array, Array]:
    """Euler predictor step mapping j=0 over Openpipeflow IMM."""
    velocity_n, pressure_n = state
    nonlin_n = rhs_no_lapl
    
    prediction_state, _ = _imm_iteration(
        velocity_n, velocity_n, pressure_n, nonlin_n, nonlin_n
    )
    return prediction_state


def _correct(
    state_prev: tuple[Array, Array], 
    prediction_state: tuple[Array, Array], 
    rhs_prev: Array, 
    rhs_next: Array
) -> tuple[tuple[Array, Array], Array]:
    """Crank-Nicolson corrector mapping j>0 over Openpipeflow IMM."""
    velocity_n, _ = state_prev
    velocity_j, pressure_j = prediction_state
    
    nonlin_n = rhs_prev
    nonlin_j = rhs_next
    
    prediction_state_new, correction = _imm_iteration(
        velocity_n, velocity_j, pressure_j, nonlin_n, nonlin_j
    )
    return prediction_state_new, correction


def _norm(correction: Array) -> Array:
    """L2 convergence norm."""
    return jnp.sqrt(get_norm2(correction, fourier.k_metric, flow.ys))


predict_and_correct, iterate_correction = make_stepper(
    _get_rhs, _predict, _correct, _norm
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
