"""Unit tests for the cylindrical geometry operators.

Tests cover:

1. Half-CGL grid properties (positive, monotone, endpoint).
2. Parity-reduced FD matrices vs full auxiliary grid reference.
3. `$A_{\\mathrm{base}}$` dense operator vs NumPy reference.
4. ``_abase_matvec`` matrix-free vs dense reference.
5. ``_lk_matvec`` vs per-mode NumPy reference.
6. SPIKE vs dense parity for `$L_k$`, `$H_{k,+}$`,
   `$H_{k,-}$`, `$H_{k,z}$`.
7. ``get_norm2_cyl`` correctness.
8. Composite integration weights on the half-CGL grid.

Run as a script via ``uv run python tests/test_cylindrical.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

from dnsjax.parameters import params  # noqa: E402

params.phys.system = "pipe"
params.res.nx = 4
params.res.ny = 16
params.res.nz = 4
params.res.fd_order = 4
params.res.double_precision = True

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.fd import (  # noqa: E402
    build_diff_matrices,
    build_integration_weights,
)
from dnsjax.flows.pipe import flow as pipe_flow  # noqa: E402
from dnsjax.geometries.cylindrical import (  # noqa: E402
    _abase_matvec,
    _build_A_base,
    _build_half_cgl_grid,
    _build_Hk_blocks_gpu,
    _build_Hk_dense_gpu,
    _build_Lk_blocks_gpu,
    _build_Lk_dense_gpu,
    _build_parity_reduced_matrices,
    _lk_matvec,
    fourier,
    get_norm2_cyl,
)
from dnsjax.geometries.wall_bounded import get_norm2  # noqa: E402
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedOperator,
    _spike_factor,
    validate_spike_partition,
)

# ── helpers ──────────────────────────────────────────────────────────


def _build_Lk_reference_cyl(
    m_val: float,
    kz_val: float,
    A_base: np.ndarray,
    D1_wall_row: np.ndarray,
    inv_r2: np.ndarray,
) -> np.ndarray:
    r"""Build single-mode cylindrical `$L_k$` in NumPy.

    `$L_k = A_{\mathrm{base}} - (m^2/r^2 + k_z^2)\,I$` with
    Neumann wall row (last row), or pin for mean mode.
    """
    diag_shift = m_val**2 * inv_r2 + kz_val**2
    Lk = A_base.copy() - np.diag(diag_shift)
    is_mean = m_val == 0.0 and kz_val == 0.0
    if is_mean:
        Lk[-1, :] = 0.0
        Lk[-1, -1] = 1.0
    else:
        Lk[-1, :] = D1_wall_row
    return Lk


# ── tests ────────────────────────────────────────────────────────────

# Group A: Grid and FD matrices


def test_half_cgl_grid_properties() -> None:
    """Half-CGL grid: strictly positive, monotone, endpoint = 1."""
    for Nr in [8, 16, 32]:
        rs = np.asarray(_build_half_cgl_grid(Nr))
        assert rs.shape == (Nr,), f"Nr={Nr}: wrong shape {rs.shape}"
        assert np.all(rs > 0), f"Nr={Nr}: non-positive point"
        assert_allclose(
            rs[-1],
            1.0,
            atol=1e-14,
            err_msg=f"Nr={Nr}: last point != 1",
        )
        diffs = np.diff(rs)
        assert np.all(diffs > 0), f"Nr={Nr}: not monotonically increasing"


def test_parity_reduced_matrices_vs_full_grid() -> None:
    """Parity-reduced matrices match the full auxiliary grid."""
    Nr = params.res.ny
    p = params.res.fd_order
    rs = _build_half_cgl_grid(Nr)

    D1_even, D2_even, D1_odd, D2_odd, D1_pos, D2_pos = (
        _build_parity_reduced_matrices(rs, p)
    )

    # Reference: build full auxiliary grid explicitly.
    aux_grid = jnp.concatenate([-rs[::-1], rs])
    D1_full, D2_full = build_diff_matrices(aux_grid, p)

    D1_full_np = np.asarray(D1_full)
    D2_full_np = np.asarray(D2_full)

    D1_pos_ref = D1_full_np[Nr:, Nr:]
    D1_ghost_flipped = D1_full_np[Nr:, :Nr][:, ::-1]
    D2_pos_ref = D2_full_np[Nr:, Nr:]
    D2_ghost_flipped = D2_full_np[Nr:, :Nr][:, ::-1]

    assert_allclose(
        np.asarray(D1_pos),
        D1_pos_ref,
        atol=1e-14,
        err_msg="D1_pos mismatch",
    )
    assert_allclose(
        np.asarray(D2_pos),
        D2_pos_ref,
        atol=1e-14,
        err_msg="D2_pos mismatch",
    )
    assert_allclose(
        np.asarray(D1_even),
        D1_pos_ref + D1_ghost_flipped,
        atol=1e-14,
        err_msg="D1_even mismatch",
    )
    assert_allclose(
        np.asarray(D1_odd),
        D1_pos_ref - D1_ghost_flipped,
        atol=1e-14,
        err_msg="D1_odd mismatch",
    )
    assert_allclose(
        np.asarray(D2_even),
        D2_pos_ref + D2_ghost_flipped,
        atol=1e-14,
        err_msg="D2_even mismatch",
    )
    assert_allclose(
        np.asarray(D2_odd),
        D2_pos_ref - D2_ghost_flipped,
        atol=1e-14,
        err_msg="D2_odd mismatch",
    )


# Group B: Operators


def test_A_base_matches_reference() -> None:
    r"""``_build_A_base`` matches `$D_2 + \mathrm{diag}(1/r) D_1$`."""
    Nr = params.res.ny
    p = params.res.fd_order
    rs = _build_half_cgl_grid(Nr)
    inv_r = 1.0 / rs

    D1_even, D2_even, D1_odd, D2_odd, _, _ = _build_parity_reduced_matrices(
        rs, p
    )

    for label, D1, D2 in [
        ("even", D1_even, D2_even),
        ("odd", D1_odd, D2_odd),
    ]:
        A_base = np.asarray(_build_A_base(D1, D2, inv_r))
        ref = np.asarray(D2) + np.diag(np.asarray(inv_r)) @ np.asarray(D1)
        assert_allclose(
            A_base,
            ref,
            atol=1e-12,
            err_msg=f"A_base ({label} parity)",
        )


def test_abase_matvec_matches_dense() -> None:
    """``_abase_matvec`` matches dense ``A_base @ u``."""
    Nr = params.res.ny
    p = params.res.fd_order
    rs = _build_half_cgl_grid(Nr)
    inv_r = 1.0 / rs

    D1_even, D2_even, D1_odd, D2_odd, D1_pos, D2_pos = (
        _build_parity_reduced_matrices(rs, p)
    )
    D1_ghost = D1_even - D1_pos
    D2_ghost = D2_even - D2_pos
    A_even = np.asarray(_build_A_base(D1_even, D2_even, inv_r))
    A_odd = np.asarray(_build_A_base(D1_odd, D2_odd, inv_r))

    flow_ = SimpleNamespace(
        D1_pos=D1_pos,
        D2_pos=D2_pos,
        D1_ghost=D1_ghost,
        D2_ghost=D2_ghost,
        inv_r=inv_r,
    )

    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(10)
    u_np = rng.standard_normal((Nm, Nkz, Nr)) + 1j * rng.standard_normal(
        (Nm, Nkz, Nr)
    )
    u = jnp.asarray(u_np)

    for label, parity_val, A_ref in [
        ("even", 1.0, A_even),
        ("odd", -1.0, A_odd),
    ]:
        parity_sign = jnp.full((Nm, 1, 1), parity_val)
        got = np.asarray(_abase_matvec(u, flow_, parity_sign))
        ref = np.einsum("ij, mzj -> mzi", A_ref, u_np)
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"parity={label}",
        )


def test_lk_matvec_matches_reference() -> None:
    r"""``_lk_matvec`` matches per-mode NumPy reference."""
    Nr = params.res.ny
    inv_r2 = np.asarray(pipe_flow.inv_r2)
    D1_wall_row = np.asarray(pipe_flow.D1_wall).ravel()
    A_even = np.asarray(pipe_flow.A_base_even)
    A_odd = np.asarray(pipe_flow.A_base_odd)

    m_vals = np.asarray(fourier.m).ravel()
    kz_vals = np.asarray(fourier.kz).ravel()
    m_is_even_np = np.asarray(fourier.m_is_even).ravel()

    Nm = len(m_vals)
    Nkz = len(kz_vals)
    rng = np.random.default_rng(60)
    u_np = rng.standard_normal((Nm, Nkz, Nr)) + 1j * rng.standard_normal(
        (Nm, Nkz, Nr)
    )

    # Reference: build Lk per mode and apply.
    ref = np.zeros_like(u_np)
    for mi in range(Nm):
        A_base = A_even if m_is_even_np[mi] else A_odd
        for ki in range(Nkz):
            Lk = _build_Lk_reference_cyl(
                m_vals[mi],
                kz_vals[ki],
                A_base,
                D1_wall_row,
                inv_r2,
            )
            ref[mi, ki] = Lk @ u_np[mi, ki]

    u = jax.device_put(
        jnp.asarray(u_np),
        sharding.spec_imm_corr_shard,
    )
    got = np.asarray(_lk_matvec(u, pipe_flow, fourier))
    assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_spike_vs_dense_on_cylindrical_operators() -> None:
    """SPIKE matches dense for cylindrical Lk/Hk_plus/Hk_minus/Hk_z."""
    Nr = params.res.ny
    p = params.res.fd_order
    P_blk, m_blk = validate_spike_partition(Nr, p, "Nr")

    m_is_even_p = fourier.m_is_even
    m_is_even_v = 1.0 - fourier.m_is_even

    m_sq = fourier.m2
    m_plus_1_sq = (fourier.m + 1) ** 2
    m_minus_1_sq = (fourier.m - 1) ** 2

    dt = params.step.dt
    c = params.step.implicitness
    nu = 1.0 / params.phys.re

    A_even = pipe_flow.A_base_even
    A_odd = pipe_flow.A_base_odd
    inv_r2 = pipe_flow.inv_r2
    kz2 = fourier.kz2
    k2_is_zero = fourier.k2_is_zero
    D1_wall = pipe_flow.D1_wall.ravel()

    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(70)
    b = rng.standard_normal((Nm, Nkz, Nr)) + 1j * rng.standard_normal(
        (Nm, Nkz, Nr)
    )
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_imm_corr_shard)

    # --- Lk ---
    Lk_A, Lk_B, Lk_C = _build_Lk_blocks_gpu(
        D1_wall,
        A_even,
        A_odd,
        m_is_even_p,
        m_sq,
        inv_r2,
        kz2,
        k2_is_zero,
        p,
        P_blk,
        m_blk,
    )
    Lk_banded = PerModeBandedOperator(*_spike_factor(Lk_A, Lk_B, Lk_C))
    Lk_dense = DenseJAXSolver(
        _build_Lk_dense_gpu(
            D1_wall,
            A_even,
            A_odd,
            m_is_even_p,
            m_sq,
            inv_r2,
            kz2,
            k2_is_zero,
        )
    )
    x_b = np.array(Lk_banded.solve(rhs))
    x_d = np.array(Lk_dense.solve(rhs))
    assert_allclose(x_b, x_d, atol=1e-9, rtol=1e-9, err_msg="Lk")

    # --- Hk_plus (meff = m+1, vel parity) ---
    Hp_A, Hp_B, Hp_C = _build_Hk_blocks_gpu(
        A_even,
        A_odd,
        m_is_even_v,
        m_plus_1_sq,
        inv_r2,
        kz2,
        dt,
        c,
        nu,
        p,
        P_blk,
        m_blk,
    )
    Hp_banded = PerModeBandedOperator(*_spike_factor(Hp_A, Hp_B, Hp_C))
    Hp_dense = DenseJAXSolver(
        _build_Hk_dense_gpu(
            A_even,
            A_odd,
            m_is_even_v,
            m_plus_1_sq,
            inv_r2,
            kz2,
            dt,
            c,
            nu,
        )
    )
    assert_allclose(
        np.asarray(Hp_banded.solve(rhs)),
        np.asarray(Hp_dense.solve(rhs)),
        atol=1e-9,
        rtol=1e-9,
        err_msg="Hk_plus",
    )

    # --- Hk_minus (meff = m-1, vel parity) ---
    Hm_A, Hm_B, Hm_C = _build_Hk_blocks_gpu(
        A_even,
        A_odd,
        m_is_even_v,
        m_minus_1_sq,
        inv_r2,
        kz2,
        dt,
        c,
        nu,
        p,
        P_blk,
        m_blk,
    )
    Hm_banded = PerModeBandedOperator(*_spike_factor(Hm_A, Hm_B, Hm_C))
    Hm_dense = DenseJAXSolver(
        _build_Hk_dense_gpu(
            A_even,
            A_odd,
            m_is_even_v,
            m_minus_1_sq,
            inv_r2,
            kz2,
            dt,
            c,
            nu,
        )
    )
    assert_allclose(
        np.asarray(Hm_banded.solve(rhs)),
        np.asarray(Hm_dense.solve(rhs)),
        atol=1e-9,
        rtol=1e-9,
        err_msg="Hk_minus",
    )

    # --- Hk_z (meff = m, pressure parity) ---
    Hz_A, Hz_B, Hz_C = _build_Hk_blocks_gpu(
        A_even,
        A_odd,
        m_is_even_p,
        m_sq,
        inv_r2,
        kz2,
        dt,
        c,
        nu,
        p,
        P_blk,
        m_blk,
    )
    Hz_banded = PerModeBandedOperator(*_spike_factor(Hz_A, Hz_B, Hz_C))
    Hz_dense = DenseJAXSolver(
        _build_Hk_dense_gpu(
            A_even,
            A_odd,
            m_is_even_p,
            m_sq,
            inv_r2,
            kz2,
            dt,
            c,
            nu,
        )
    )
    assert_allclose(
        np.asarray(Hz_banded.solve(rhs)),
        np.asarray(Hz_dense.solve(rhs)),
        atol=1e-9,
        rtol=1e-9,
        err_msg="Hk_z",
    )


# Group C: Norms


def test_get_norm2_cyl() -> None:
    r"""``get_norm2_cyl`` matches manual `$(|u_+|^2+|u_-|^2)/2+|u_z|^2$`."""
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    Nr = params.res.ny

    rng = np.random.default_rng(80)
    s_shape = (3, Nm, Nkz, Nr)
    state_np = rng.standard_normal(s_shape) + 1j * rng.standard_normal(s_shape)
    state = jnp.asarray(state_np)

    k_m = fourier.k_metric
    y_w = pipe_flow.y_weights
    got = float(get_norm2_cyl(state, k_m, y_w))

    # Reference: separate norm2 calls.
    pm = jnp.stack([state[1], state[2]])
    pm_norm2 = float(get_norm2(pm, k_m, y_w))
    uz_norm2 = float(get_norm2(state[0:1], k_m, y_w))
    ref = pm_norm2 / 2 + uz_norm2

    assert_allclose(got, ref, atol=1e-12, err_msg="get_norm2_cyl")


# Group D: Integration


def test_cylindrical_integration_weights() -> None:
    """Composite weights on half-CGL: sum and polynomial exactness."""
    p = params.res.fd_order
    for Nr in [8, 16, 32]:
        rs = _build_half_cgl_grid(Nr)
        rs_np = np.asarray(rs)
        w = build_integration_weights(rs, p=p)
        w_np = np.asarray(w)

        r_min = rs_np[0]
        interval_length = 1.0 - r_min
        assert_allclose(
            np.sum(w_np),
            interval_length,
            atol=1e-12,
            err_msg=f"Nr={Nr}: weight sum",
        )

        for d in range(p + 1):
            computed = float(np.dot(w_np, rs_np**d))
            exact = (1.0 - r_min ** (d + 1)) / (d + 1)
            assert_allclose(
                computed,
                exact,
                atol=1e-10,
                err_msg=f"Nr={Nr}, degree={d}",
            )


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
