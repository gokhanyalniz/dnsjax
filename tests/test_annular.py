"""Unit tests for the annular (Taylor-Couette) geometry operators.

Tests cover:

1. Annular grid properties (spans ``[r1, r2]``, monotone, endpoints).
2. `$A_{\\mathrm{base}}$` dense operator vs NumPy reference.
3. ``_abase_matvec`` matrix-free vs dense reference.
4. ``_lk_matvec`` vs per-mode NumPy reference (Neumann at both walls,
   pin at the mean mode).
5. SPIKE vs dense parity for `$L_k$`, `$H_{k,+}$`, `$H_{k,-}$`,
   `$H_{k,z}$`.
6. ``get_norm2_annular`` correctness.
7. Circular-Couette coefficients `$A_0$`, `$B_0$` vs the per-case
   reference forms and wall values.
8. Composite integration weights with radial Jacobian on ``[r1, r2]``.

Run as a script via ``uv run python tests/test_annular.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

from dnsjax.parameters import (  # noqa: E402
    Parameters,
    derived_params,
    params,
    update_parameters,
)

# Case 1 (inner-driven): re1 = 100, re2 = 0, eta = 0.5 -> r1 = 1, r2 = 2.
update_parameters(
    Parameters(
        phys={"system": "taylor-couette", "re1": 100.0, "re2": 0.0},
        geo={"eta": 0.5},
        res={
            "nx": 4,
            "ny": 16,
            "nz": 4,
            "fd_order": 4,
            "double_precision": True,
        },
    )
)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.fd import build_integration_weights  # noqa: E402
from dnsjax.flows.wall_bounded.taylor_couette import (  # noqa: E402
    flow as tc_flow,
)
from dnsjax.geometries.wall_bounded import get_norm2  # noqa: E402
from dnsjax.geometries.wall_bounded.annular import (  # noqa: E402
    _abase_matvec,
    _build_Hk_blocks_gpu,
    _build_Hk_dense_gpu,
    _build_Lk_blocks_gpu,
    _build_Lk_dense_gpu,
    _lk_matvec,
    build_annular_grid,
    fourier,
    get_norm2_annular,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    _spike_factor,
    validate_spike_partition,
)

R1 = derived_params.r_inner
R2 = derived_params.r_outer


# ── helpers ──────────────────────────────────────────────────────────


def _build_Lk_reference_annular(
    m_val: float,
    kz_val: float,
    A_base: np.ndarray,
    D1_bnd: np.ndarray,
    inv_r2: np.ndarray,
) -> np.ndarray:
    r"""Build single-mode annular `$L_k$` in NumPy.

    `$L_k = A_{\mathrm{base}} - (m^2/r^2 + k_z^2)\,I$` with Neumann
    rows at both walls (inner row 0, outer row -1), and an outer-wall
    pin for the mean mode.
    """
    diag_shift = m_val**2 * inv_r2 + kz_val**2
    Lk = A_base.copy() - np.diag(diag_shift)
    Lk[0, :] = D1_bnd[0]  # Neumann inner
    is_mean = m_val == 0.0 and kz_val == 0.0
    if is_mean:
        Lk[-1, :] = 0.0
        Lk[-1, -1] = 1.0  # pin outer wall
    else:
        Lk[-1, :] = D1_bnd[1]  # Neumann outer
    return Lk


# ── tests ────────────────────────────────────────────────────────────

# Group A: Grid


def test_annular_grid_properties() -> None:
    """Annular grid: spans [r1, r2], strictly increasing, endpoints."""
    for ny in [8, 16, 33]:
        rs, _, _, _, _ = build_annular_grid(ny, 4, R1, R2)
        rs_np = np.asarray(rs)
        assert rs_np.shape == (ny,), f"ny={ny}: wrong shape"
        assert_allclose(rs_np[0], R1, atol=1e-14, err_msg=f"ny={ny}: r[0]")
        assert_allclose(rs_np[-1], R2, atol=1e-14, err_msg=f"ny={ny}: r[-1]")
        assert np.all(np.diff(rs_np) > 0), f"ny={ny}: not increasing"


# Group B: Operators


def test_A_base_matches_reference() -> None:
    r"""``_build_A_base`` matches `$D_2 + \mathrm{diag}(1/r) D_1$`."""
    A_base = np.asarray(tc_flow.A_base)
    D1 = np.asarray(tc_flow.D1)
    D2 = np.asarray(tc_flow.D2)
    inv_r = np.asarray(tc_flow.inv_r)
    ref = D2 + np.diag(inv_r) @ D1
    assert_allclose(A_base, ref, atol=1e-12, err_msg="A_base")


def test_abase_matvec_matches_dense() -> None:
    """``_abase_matvec`` matches dense ``A_base @ u``."""
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    A_base = np.asarray(tc_flow.A_base)

    flow_ = SimpleNamespace(D1=tc_flow.D1, D2=tc_flow.D2, inv_r=tc_flow.inv_r)

    rng = np.random.default_rng(10)
    u_np = rng.standard_normal((Nr, Nm, Nkz)) + 1j * rng.standard_normal(
        (Nr, Nm, Nkz)
    )
    u = jnp.asarray(u_np)

    got = np.asarray(_abase_matvec(u, flow_))
    ref = np.einsum("ij, jmz -> imz", A_base, u_np)
    assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_lk_matvec_matches_reference() -> None:
    r"""``_lk_matvec`` matches per-mode NumPy reference."""
    Nr = params.res.ny
    inv_r2 = np.asarray(tc_flow.inv_r2)
    D1_bnd = np.asarray(tc_flow.D1_bnd)
    A_base = np.asarray(tc_flow.A_base)

    m_vals = np.asarray(fourier.m).ravel()
    kz_vals = np.asarray(fourier.kz).ravel()
    Nm = len(m_vals)
    Nkz = len(kz_vals)

    rng = np.random.default_rng(60)
    u_np = rng.standard_normal((Nr, Nm, Nkz)) + 1j * rng.standard_normal(
        (Nr, Nm, Nkz)
    )

    ref = np.zeros_like(u_np)
    for mi in range(Nm):
        for ki in range(Nkz):
            Lk = _build_Lk_reference_annular(
                m_vals[mi], kz_vals[ki], A_base, D1_bnd, inv_r2
            )
            ref[:, mi, ki] = Lk @ u_np[:, mi, ki]

    u = jax.device_put(jnp.asarray(u_np), sharding.spec_scalar_shard)
    got = np.asarray(_lk_matvec(u, tc_flow, fourier))
    assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_spike_vs_dense_on_annular_operators() -> None:
    """SPIKE matches dense for annular Lk/Hk_plus/Hk_minus/Hk_z."""
    Nr = params.res.ny
    p = params.res.fd_order
    P_blk, m_blk = validate_spike_partition(Nr, p, "Nr")

    m_s = fourier.m[0, ..., None]
    kz2_s = fourier.kz2[0, ..., None]
    mean_s = fourier.mean_mask[0, ..., None]
    m_sq = m_s**2
    m_plus_1_sq = (m_s + 1) ** 2
    m_minus_1_sq = (m_s - 1) ** 2

    dt = params.step.dt
    c = params.step.implicitness
    nu = 1.0 / params.phys.re

    D1 = tc_flow.D1
    A_base = tc_flow.A_base
    inv_r2 = tc_flow.inv_r2

    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(70)
    b = rng.standard_normal((Nm, Nkz, Nr)) + 1j * rng.standard_normal(
        (Nm, Nkz, Nr)
    )
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_imm_corr_shard)

    # --- Lk ---
    Lk_banded = _spike_factor(
        *_build_Lk_blocks_gpu(
            D1, A_base, m_sq, inv_r2, kz2_s, mean_s, p, P_blk, m_blk
        )
    )
    Lk_dense = DenseJAXSolver(
        _build_Lk_dense_gpu(D1, A_base, m_sq, inv_r2, kz2_s, mean_s)
    )
    assert_allclose(
        np.asarray(Lk_banded.solve(rhs)),
        np.asarray(Lk_dense.solve(rhs)),
        atol=1e-9,
        rtol=1e-9,
        err_msg="Lk",
    )

    # --- Hk_plus / Hk_minus / Hk_z ---
    for label, meff2 in [
        ("Hk_plus", m_plus_1_sq),
        ("Hk_minus", m_minus_1_sq),
        ("Hk_z", m_sq),
    ]:
        Hk_banded = _spike_factor(
            *_build_Hk_blocks_gpu(
                A_base, meff2, inv_r2, kz2_s, dt, c, nu, p, P_blk, m_blk
            )
        )
        Hk_dense = DenseJAXSolver(
            _build_Hk_dense_gpu(A_base, meff2, inv_r2, kz2_s, dt, c, nu)
        )
        assert_allclose(
            np.asarray(Hk_banded.solve(rhs)),
            np.asarray(Hk_dense.solve(rhs)),
            atol=1e-9,
            rtol=1e-9,
            err_msg=label,
        )


# Group C: Norms


def test_get_norm2_annular() -> None:
    r"""``get_norm2_annular`` matches `$(|u_+|^2+|u_-|^2)/2+|u_z|^2$`."""
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    Nr = params.res.ny

    rng = np.random.default_rng(80)
    s_shape = (3, Nr, Nm, Nkz)
    state_np = rng.standard_normal(s_shape) + 1j * rng.standard_normal(s_shape)
    state = jnp.asarray(state_np)

    k_m = fourier.k_metric
    y_w = tc_flow.y_weights
    got = float(get_norm2_annular(state, k_m, y_w))

    pm = jnp.stack([state[1], state[2]])
    ref = float(get_norm2(pm, k_m, y_w)) / 2 + float(
        get_norm2(state[0:1], k_m, y_w)
    )
    assert_allclose(got, ref, atol=1e-12, err_msg="get_norm2_annular")


# Group D: Control parameters


def _check_ccf(re1: float, re2: float, eta: float) -> None:
    """Set the (re1, re2, eta) triple and verify A0, B0, wall values."""
    update_parameters(
        Parameters(
            phys={"system": "taylor-couette", "re1": re1, "re2": re2},
            geo={"eta": eta},
        )
    )
    A0 = derived_params.ccf_A
    B0 = derived_params.ccf_B
    re_ref = re1 if re1 > 0 else re2
    A0_ref = (re2 - eta * re1) / ((1 + eta) * re_ref)
    B0_ref = eta * (re1 - eta * re2) / ((1 + eta) * (1 - eta) ** 2 * re_ref)
    assert_allclose(A0, A0_ref, atol=1e-13, err_msg="A0")
    assert_allclose(B0, B0_ref, atol=1e-13, err_msg="B0")

    r1 = eta / (1 - eta)
    r2 = 1 / (1 - eta)
    # Wall values: U_theta(r1) = (re1>0 ? 1 : 0), U_theta(r2) =
    # (re1>0 ? re2/re1 : 1).
    u1 = A0 * r1 + B0 / r1
    u2 = A0 * r2 + B0 / r2
    if re1 > 0:
        assert_allclose(u1, 1.0, atol=1e-13, err_msg="U_theta(r1)")
        assert_allclose(u2, re2 / re1, atol=1e-13, err_msg="U_theta(r2)")
    else:
        assert_allclose(u1, 0.0, atol=1e-13, err_msg="U_theta(r1)")
        assert_allclose(u2, 1.0, atol=1e-13, err_msg="U_theta(r2)")


def test_ccf_coefficients() -> None:
    """Circular-Couette coefficients match the per-case forms."""
    _check_ccf(100.0, 0.0, 0.5)  # case 1, outer fixed
    _check_ccf(100.0, 50.0, 0.4)  # case 1, co-rotating
    _check_ccf(100.0, -30.0, 0.6)  # case 1, counter-rotating
    _check_ccf(0.0, 100.0, 0.5)  # case 2, outer-driven
    # Restore the module-level case-1 configuration.
    update_parameters(
        Parameters(
            phys={"system": "taylor-couette", "re1": 100.0, "re2": 0.0},
            geo={"eta": 0.5},
        )
    )


# Group E: Integration


def test_annular_integration_weights() -> None:
    """Composite weights and radial Jacobian integrate r-moments."""
    p = params.res.fd_order
    for ny in [8, 16, 33]:
        rs, _, _, y_weights, _ = build_annular_grid(ny, p, R1, R2)
        rs_np = np.asarray(rs)
        yw = np.asarray(y_weights)
        w = np.asarray(build_integration_weights(rs_np, p=p))

        # Raw weights: exact for polynomials up to degree p.
        for d in range(p + 1):
            computed = float(np.dot(w, rs_np**d))
            exact = (R2 ** (d + 1) - R1 ** (d + 1)) / (d + 1)
            assert_allclose(
                computed,
                exact,
                atol=1e-10,
                rtol=1e-10,
                err_msg=f"ny={ny}, raw degree={d}",
            )
        # Jacobian weights integrate the radial measure:
        # sum(w_j r_j) = int_{r1}^{r2} r dr = (r2^2 - r1^2)/2.
        assert_allclose(
            np.sum(yw),
            (R2**2 - R1**2) / 2,
            atol=1e-10,
            err_msg=f"ny={ny}: sum(y_weights)",
        )


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
