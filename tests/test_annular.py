"""Unit tests for the annular (Taylor-Couette) geometry operators.

Tests cover:

1. Annular grid properties (spans ``[r1, r2]``, monotone, endpoints).
2. `$A_{\\mathrm{base}}$` dense operator vs NumPy reference.
3. ``_abase_matvec`` matrix-free vs dense reference.
4. ``_lk_matvec`` vs per-mode NumPy reference (Neumann at both walls,
   pin at the mean mode).
5. Pallas band-vs-dense parity for `$L_k$` and the three `$H_k$`
   operators (banded storage == ``banded(dense)``, no-pivot banded
   solve == dense solve).
6. ``get_norm2_annular`` correctness.
7. Circular-Couette coefficients `$A_0$`, `$B_0$` vs the per-case
   reference forms and wall values.
8. Affine-mapped Clenshaw-Curtis integration weights (spectral) with
   the radial Jacobian on ``[r1, r2]``.

Run as a script via ``uv run python tests/test_annular.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

# Select the JAX backend from --dist.platform (default cpu) before the
# geometry import below builds sharding.  --dist.platform cuda runs the
# Pallas band-vs-dense parity on a GPU.
from dnsjax.parameters import (  # noqa: E402
    Parameters,
    configure_jax_platform,
    derived_params,
    params,
    platform_from_argv,
    update_parameters,
)

configure_jax_platform(platform_from_argv())

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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.flows.wall_bounded.taylor_couette import (  # noqa: E402
    flow as tc_flow,
)
from dnsjax.geometries.wall_bounded import get_norm2  # noqa: E402
from dnsjax.geometries.wall_bounded.annular import (  # noqa: E402
    _abase_matvec,
    _build_Hk_band_gpu,
    _build_Hk_dense_gpu,
    _build_Lk_band_gpu,
    _build_Lk_dense_gpu,
    _lk_matvec,
    build_annular_grid,
    fourier,
    get_norm2_annular,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _banded_factor,
    _banded_from_dense,
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


def test_pallas_vs_dense_on_annular_operators() -> None:
    r"""``PerModeBandedPallasOperator`` matches ``DenseJAXSolver`` on
    annular Lk/Hk_plus/Hk_minus/Hk_z.

    Validates the Pallas band assembly (``_build_{Lk,Hk}_band_gpu``):
    the banded operator equals ``banded(dense)`` exactly, and the
    no-pivot banded sweep (CPU pure-JAX path) reproduces the dense
    solve on a complex RHS.
    """
    Nr = params.res.ny
    p = params.res.fd_order

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
    rng = np.random.default_rng(71)
    b = rng.standard_normal((Nr, Nm, Nkz)) + 1j * rng.standard_normal(
        (Nr, Nm, Nkz)
    )
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_scalar_shard)

    to_band = jax.vmap(jax.vmap(lambda A: _banded_from_dense(A, p)))

    def _check(label: str, band: jax.Array, full: jax.Array) -> None:
        assert_allclose(
            np.asarray(band),
            np.asarray(to_band(full)),
            atol=1e-12,
            err_msg=f"{label} assembly",
        )
        pallas = PerModeBandedPallasOperator.from_banded_factors(
            *_banded_factor(band)
        )
        dense = DenseJAXSolver(full)
        assert_allclose(
            np.asarray(pallas.solve(rhs)),
            np.asarray(dense.solve(rhs)),
            atol=1e-9,
            rtol=1e-9,
            err_msg=label,
        )

    _check(
        "Lk",
        _build_Lk_band_gpu(D1, A_base, m_sq, inv_r2, kz2_s, mean_s, p),
        _build_Lk_dense_gpu(D1, A_base, m_sq, inv_r2, kz2_s, mean_s),
    )
    for label, meff2 in [
        ("Hk_plus", m_plus_1_sq),
        ("Hk_minus", m_minus_1_sq),
        ("Hk_z", m_sq),
    ]:
        _check(
            label,
            _build_Hk_band_gpu(A_base, meff2, inv_r2, kz2_s, dt, c, nu, p),
            _build_Hk_dense_gpu(A_base, meff2, inv_r2, kz2_s, dt, c, nu),
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


def test_narrow_gap_limit() -> None:
    r"""``I_lam`` reduces to the plane-Couette value as `$\eta \to 1$`.

    For pure inner rotation (``re2 = 0``) the laminar constant
    `$I_{\mathrm{lam}} = 4 B_0^2 / (\mathrm{Re}\,r_1^2 r_2^2)$` satisfies
    the closed form `$I_{\mathrm{lam}}\,\mathrm{Re} = 4/(1+\eta)^2$`
    (``ccf_B`` already carries the `$1/\mathrm{Re}$` factor, so this is
    Reynolds-independent), which tends to the plane-Couette value 1 as
    the gap narrows.
    """
    re = 100.0
    prev_err = None
    for eta in (0.9, 0.99, 0.999):
        update_parameters(
            Parameters(
                phys={"system": "taylor-couette", "re1": re, "re2": 0.0},
                geo={"eta": eta},
            )
        )
        B0 = derived_params.ccf_B
        r1 = derived_params.r_inner
        r2 = derived_params.r_outer
        I_lam_Re = 4.0 * B0**2 / (r1**2 * r2**2)  # = I_lam * Re
        assert_allclose(
            I_lam_Re,
            4.0 / (1 + eta) ** 2,
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"eta={eta}: I_lam*Re closed form",
        )
        err = abs(I_lam_Re - 1.0)
        if prev_err is not None:
            assert err < prev_err, f"eta={eta}: I_lam*Re not converging to 1"
        prev_err = err
    assert prev_err < 2e-3, "eta=0.999: I_lam*Re not within 2e-3 of 1"

    # Restore the module-level case-1 configuration.
    update_parameters(
        Parameters(
            phys={"system": "taylor-couette", "re1": 100.0, "re2": 0.0},
            geo={"eta": 0.5},
        )
    )


# Group E: Integration


def test_annular_integration_weights() -> None:
    """Default annular grid uses affine-mapped Clenshaw-Curtis:
    ``y_weights`` integrate ``f * r`` spectrally -- exact for the
    radial measure and polynomial moments, and near machine precision
    for a smooth non-polynomial (an ``fd_order`` composite rule would
    be only ``~1e-7`` at this resolution)."""
    p = params.res.fd_order
    for ny in [8, 16, 33]:
        rs, _, _, y_weights, _ = build_annular_grid(ny, p, R1, R2)
        rs_np = np.asarray(rs)
        yw = np.asarray(y_weights)
        # sum(w_j r_j) = int_{r1}^{r2} r dr = (r2^2 - r1^2)/2 (exact).
        assert_allclose(
            np.sum(yw),
            (R2**2 - R1**2) / 2,
            atol=1e-12,
            err_msg=f"ny={ny}: radial measure",
        )
        # int r^2 * r dr = (r2^4 - r1^4)/4 (CC exact for this poly).
        assert_allclose(
            float(yw @ rs_np**2),
            (R2**4 - R1**4) / 4,
            atol=1e-12,
            err_msg=f"ny={ny}: r^3 moment",
        )
        # Smooth non-polynomial: affine CC is spectral, so near
        # machine precision once resolved.
        grid = np.linspace(R1, R2, 2_000_001)
        ref = np.trapezoid(np.cos(2.0 * grid) * grid, grid)
        err = abs(float(yw @ np.cos(2.0 * rs_np)) - ref)
        if ny == 33:
            assert err < 1e-9, f"ny={ny}: CC not spectral, err={err:.2e}"


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
