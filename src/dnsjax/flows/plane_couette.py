"""Plane Couette flow: wall-bounded shear between two moving plates.

The base flow is `$U(y) = y$` on the Chebyshev-Gauss-Lobatto grid
`$y \in [-1, 1]$`, with walls moving at `$\pm 1$`.
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from jax import Array, jit
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ..bench import timer
from ..fd import build_diff_matrices, precompute_imm
from ..operators import (
    fourier,
    phys_to_spec_2d,
    spec_to_phys_2d,
)
from ..parameters import params
from ..rhs import get_nonlin
from ..sharding import register_dataclass_pytree, sharding
from ..timestep import make_stepper
from ..velocity import get_norm2


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
    Lk: Array = field(init=False)
    Lk_solver: Any = field(init=False)
    Hk: Array = field(init=False)
    Hk_solver: Any = field(init=False)
    Hk_minus: Array = field(init=False)
    p1: Array = field(init=False)
    p2: Array = field(init=False)
    M_inv: Array = field(init=False)
    k2: Array = field(init=False)

    def __post_init__(self) -> None:
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

        kx_global = np.array(fourier.kx.reshape(-1))
        kz_global = np.array(fourier.kz.reshape(-1))
        Nkz = len(kz_global)
        Nkx = len(kx_global)
        Ny = params.res.ny

        class IMMChunker:
            def __init__(self, ys_arr, p, dt, c, nu, D1_arr, D2_arr):
                self.ys_arr = ys_arr
                self.p = p
                self.dt = dt
                self.c = c
                self.nu = nu
                self.D1_arr = D1_arr
                self.D2_arr = D2_arr
                self.cache = {}

            def get_chunk(
                self, indices: tuple[slice, ...], key: str
            ) -> np.ndarray:
                slice_kz, slice_kx = indices[0], indices[1]
                cache_key = (
                    slice_kz.start,
                    slice_kz.stop,
                    slice_kx.start,
                    slice_kx.stop,
                )
                if cache_key not in self.cache:
                    self.cache[cache_key] = precompute_imm(
                        y=self.ys_arr,
                        kx_vals=kx_global[slice_kz],
                        kz_vals=kz_global[slice_kx],
                        p=self.p,
                        dt=self.dt,
                        c=self.c,
                        nu=self.nu,
                        D1=self.D1_arr,
                        D2=self.D2_arr,
                    )
                return self.cache[cache_key][key]

        chunker = IMMChunker(
            ys_arr=np.array(self.ys),
            p=params.res.fd_order,
            dt=params.step.dt,
            c=params.step.implicitness,
            nu=1.0 / params.phys.re,
            D1_arr=D1,
            D2_arr=D2,
        )

        shard_4d = P(None, *sharding.axis_names, None, None)
        shard_3d = P(None, *sharding.axis_names, None)
        shard_2d = P(None, *sharding.axis_names)

        self.Lk = jax.make_array_from_callback(
            (Nkz, Nkx, Ny, Ny),
            shard_4d,
            lambda idx: chunker.get_chunk(idx, "Lk"),
        )
        self.Hk = jax.make_array_from_callback(
            (Nkz, Nkx, Ny, Ny),
            shard_4d,
            lambda idx: chunker.get_chunk(idx, "Hk"),
        )
        self.Hk_minus = jax.make_array_from_callback(
            (Nkz, Nkx, Ny, Ny),
            shard_4d,
            lambda idx: chunker.get_chunk(idx, "Hk_minus"),
        )
        self.p1 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny), shard_3d, lambda idx: chunker.get_chunk(idx, "p1")
        )
        self.p2 = jax.make_array_from_callback(
            (Nkz, Nkx, Ny), shard_3d, lambda idx: chunker.get_chunk(idx, "p2")
        )
        self.M_inv = jax.make_array_from_callback(
            (Nkz, Nkx, 2, 2),
            shard_4d,
            lambda idx: chunker.get_chunk(idx, "M_inv"),
        )
        self.k2 = jax.make_array_from_callback(
            (Nkz, Nkx), shard_2d, lambda idx: chunker.get_chunk(idx, "k2")
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


import typing

import jax.scipy.linalg as sla


@jax.jit
def _lu_solve(lu_pivots: tuple[Array, Array], b: Array) -> Array:
    """Batched LU solve across 2D (k_z, k_x) Fourier modes."""

    def solve_single(lu_piv, vec):
        return sla.lu_solve(lu_piv, vec)

    return jax.vmap(jax.vmap(solve_single))(lu_pivots, b)


@register_dataclass_pytree
@dataclass
class DenseJAXSolver:
    """The current mathematically optimal dense LU cache."""

    matrix: Array
    lu: Array = field(init=False)
    piv: Array = field(init=False)

    def __post_init__(self):
        @jax.jit
        def batched_lu_factor(A: Array) -> tuple[Array, Array]:
            return jax.vmap(jax.vmap(sla.lu_factor))(A)

        self.lu, self.piv = batched_lu_factor(self.matrix)

    def solve(self, rhs: Array) -> Array:
        return _lu_solve((self.lu, self.piv), rhs)


@register_dataclass_pytree
@dataclass
class LineaxBandedSolver:
    """The Lineax sparse operator path."""

    matrix: Array
    lower_band: int
    upper_band: int
    operator: Any = field(init=False)

    def __post_init__(self):
        raise NotImplementedError(
            "Lineax banded packing is pending extraction implementation!"
        )

    def solve(self, rhs: Array) -> Array:
        raise NotImplementedError


flow: PlaneCouetteFlow = PlaneCouetteFlow()


def _compute_static_pressure(velocity_spec: Array) -> Array:
    """Solve the continuous pressure Poisson equation
    on the un-advanced snapshot.

    1. `$\nabla^2 p = \nabla \cdot \mathbf{N}$`
    2. `$\partial p/\partial y = N_v + \nu \frac{\partial^2 v}{\partial y^2}$`
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

    dy_Nv = jnp.tensordot(flow.D1, Nv, axes=(1, 0))
    div_N = 1j * fourier.kx * Nu + dy_Nv + 1j * fourier.kz * Nw

    D2_v = jnp.tensordot(flow.D2, v, axes=(1, 0))

    g_full = Nv + D2_v / params.phys.re
    g_0 = g_full[0]
    g_1 = g_full[-1]

    # Solve particular pressure
    f_P = div_N.at[0].set(0.0).at[-1].set(0.0)
    f_P_b = jnp.transpose(f_P, (1, 2, 0))
    pP_b = flow.Lk_solver.solve(f_P_b)
    pP = jnp.transpose(pP_b, (2, 0, 1))

    # Calculate continuous boundary mismatch
    r_bot = -flow.k2 * pP[0] + g_0
    r_top = -flow.k2 * pP[-1] + g_1
    r = jnp.stack([r_bot, r_top], axis=-1)

    # Apply influence matrix algebra mapping constraints
    alpha = jnp.einsum("zxab, zxb -> zxa", flow.M_inv, r)
    alpha1 = alpha[..., 0][..., None]
    alpha2 = alpha[..., 1][..., None]

    p_new_b = (
        jnp.transpose(pP, (1, 2, 0)) + alpha1 * flow.p1 + alpha2 * flow.p2
    )
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


def _curl_fn(state: Array, fourier_: Any, flow_: Any) -> Array:
    """Spectral curl with 1D FD in y and spectral derivatives in x and z."""
    u, v, w = state

    dy_u = jnp.tensordot(flow_.D1, u, axes=(1, 0))
    dy_w = jnp.tensordot(flow_.D1, w, axes=(1, 0))

    dx_v = 1j * fourier_.kx * v
    dz_v = 1j * fourier_.kz * v
    dx_w = 1j * fourier_.kx * w
    dz_u = 1j * fourier_.kz * u

    omega_x = dy_w - dz_v
    omega_y = dz_u - dx_w
    omega_z = dx_v - dy_u

    return jnp.array([omega_x, omega_y, omega_z])


def _get_rhs(state: tuple[Array, Array], fourier_: Any, flow_: Any) -> Array:
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


def _einsum_4d(op: Array, x: Array) -> Array:
    """Apply operator of shape (Nkz, Nkx, Ny, Ny) to (Ny, Nkz, Nkx)."""
    return jnp.einsum("zxij, jzx -> izx", op, x)


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    pressure_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Any,
    flow_: Any,
) -> tuple[tuple[Array, Array], Array]:
    """Openpipeflow fractional-step algorithm (imm.tex, Section 5.2)."""
    c = params.step.implicitness

    u_n, v_n, w_n = velocity_n
    Nu_n, Nv_n, Nw_n = nonlin_n
    Nu_j, Nv_j, Nw_j = nonlin_j

    # Pre-calculate d_hat^n
    dy_v_n = jnp.tensordot(flow_.D1, v_n, axes=(1, 0))
    d_hat_n = 1j * fourier_.kx * u_n + dy_v_n + 1j * fourier_.kz * w_n

    # Stage 1: Solve for v^{(j+1)}
    dy_p_j = jnp.tensordot(flow_.D1, pressure_j, axes=(1, 0))
    Rv = _einsum_4d(flow_.Hk_minus, v_n) - dy_p_j + c * Nv_j + (1 - c) * Nv_n
    Rv_b = jnp.transpose(Rv, (1, 2, 0))
    v_new_b = flow_.Hk_solver.solve(Rv_b)
    v_new = jnp.transpose(v_new_b, (2, 0, 1))

    # Stage 2: Residuals and RHS for pressure
    dy_Nv_j = jnp.tensordot(flow_.D1, Nv_j, axes=(1, 0))
    dy_Nv_n = jnp.tensordot(flow_.D1, Nv_n, axes=(1, 0))
    div_Nj = 1j * fourier_.kx * Nu_j + dy_Nv_j + 1j * fourier_.kz * Nw_j
    div_Nn = 1j * fourier_.kx * Nu_n + dy_Nv_n + 1j * fourier_.kz * Nw_n

    Lk_d = _einsum_4d(flow_.Lk, d_hat_n)

    f_hat = (
        d_hat_n / params.step.dt
        + c * div_Nj
        + (1 - c) * div_Nn
        + (1 - c) * (1.0 / params.phys.re) * Lk_d
    )

    D2_v_new = jnp.tensordot(flow_.D2, v_new, axes=(1, 0))
    Lk_v_n = _einsum_4d(flow_.Lk, v_n)

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
    pP_b = flow_.Lk_solver.solve(f_hat_P_b)
    pP = jnp.transpose(pP_b, (2, 0, 1))

    r_bot = -flow_.k2 * pP[0] + g_0
    r_top = -flow_.k2 * pP[-1] + g_1
    r = jnp.stack([r_bot, r_top], axis=-1)

    alpha = jnp.einsum("zxab, zxb -> zxa", flow_.M_inv, r)
    alpha1 = alpha[..., 0][..., None]
    alpha2 = alpha[..., 1][..., None]

    p_new_b = (
        jnp.transpose(pP, (1, 2, 0)) + alpha1 * flow_.p1 + alpha2 * flow_.p2
    )
    p_new = jnp.transpose(p_new_b, (2, 0, 1))

    # Stage 4: Solve for u and w
    dx_p_new = 1j * fourier_.kx * p_new
    dz_p_new = 1j * fourier_.kz * p_new

    Ru = _einsum_4d(flow_.Hk_minus, u_n) - dx_p_new + c * Nu_j + (1 - c) * Nu_n
    Rw = _einsum_4d(flow_.Hk_minus, w_n) - dz_p_new + c * Nw_j + (1 - c) * Nw_n

    Ru_b = jnp.transpose(Ru, (1, 2, 0))
    Rw_b = jnp.transpose(Rw, (1, 2, 0))

    u_new_b = flow_.Hk_solver.solve(Ru_b)
    w_new_b = flow_.Hk_solver.solve(Rw_b)

    u_new = jnp.transpose(u_new_b, (2, 0, 1))
    w_new = jnp.transpose(w_new_b, (2, 0, 1))

    velocity_new = jnp.array([u_new, v_new, w_new])

    correction = velocity_new - velocity_j

    state_new = (velocity_new, p_new)
    return state_new, correction


def _predict(
    state: tuple[Array, Array],
    rhs_no_lapl: Array,
    fourier_: Any,
    flow_: Any,
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
    fourier_: Any,
    flow_: Any,
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


def _norm(correction: Array, fourier_: Any, flow_: Any) -> Array:
    """L2 convergence norm."""
    return jnp.sqrt(get_norm2(correction, fourier_.k_metric, flow_.ys))


_predict_and_correct_jit, _iterate_correction_jit = make_stepper(
    _get_rhs, _predict, _correct, _norm
)


def predict_and_correct(
    state: tuple[Array, Array],
) -> tuple[tuple[Array, Array], Array, Array]:
    return _predict_and_correct_jit(state, fourier, flow)


def iterate_correction(
    state_prev: tuple[Array, Array],
    prediction: tuple[Array, Array],
    rhs_prev: Array,
):
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
