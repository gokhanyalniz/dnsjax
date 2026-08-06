r"""Unit tests for the Cartesian geometry operators.

Tests cover:

1. ``_lk_matvec`` matches a NumPy reference on CGL and custom grids.
2. ``_hk_minus_matvec`` matches a NumPy reference.
3. ``get_norm2`` matches a manual Parseval/quadrature sum.
4. Pallas band-vs-dense parity: the ``_build_{Lk,Hk,Lk_dir}_band_gpu``
   banded storage equals ``banded(dense)``, and the no-pivot banded
   solve equals the dense solve.
5. The two algebraic identities the ``res.consistent_imm``
   `$v$`-`$\omega_y$` scheme rests on: the reconstruction of
   `$(u, w)$` from `$(D_1 v, \omega_y)$` is exactly solenoidal, and
   the source projections annihilate a discrete gradient exactly.

Run as a script via ``uv run python tests/test_cartesian.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

# Select the JAX backend from --dist.platform (default cpu) and enable
# float64 before importing any dnsjax module that captures the platform
# (sharding does so at import).  ``--dist.platform cuda`` runs the Pallas
# band-vs-dense parity on a GPU.
from dnsjax.bootstrap import (
    configure_jax_platform,
    platform_from_argv,
)
from dnsjax.parameters import (
    derived_params,
    params,
)

sys.stdout.reconfigure(line_buffering=True)

configure_jax_platform(platform_from_argv())

params.phys.system = "plane-couette"
params.res.nx = 4
params.res.ny = 16
params.res.nz = 4
params.res.fd_order = 4
params.res.double_precision = True

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import Array  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.fd import (  # noqa: E402
    build_diff_matrices,
    matrix_half_bandwidth,
)
from dnsjax.geometries.wall_bounded import get_norm2  # noqa: E402
from dnsjax.geometries.wall_bounded.cartesian import (  # noqa: E402
    _build_Hk_band_gpu,
    _build_Hk_dense_gpu,
    _build_Lk_band_gpu,
    _build_Lk_dense_gpu,
    _build_Lk_dir_band_gpu,
    _build_Lk_dir_dense_gpu,
    _hk_minus_matvec,
    _lk_matvec,
    build_cartesian_grid,
    fourier,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _banded_factor,
    _banded_from_dense,
)

# ── helpers ──────────────────────────────────────────────────────────


def _build_Lk_reference(
    k2_val: float, D1: np.ndarray, D2: np.ndarray
) -> np.ndarray:
    r"""Build single-mode `$L_k$` (Neumann-BC Laplacian) in NumPy."""
    Ny = D2.shape[0]
    Lk = D2.copy() - k2_val * np.eye(Ny)
    Lk[0, :] = D1[0, :]
    if k2_val == 0.0:
        Lk[-1, :] = 0.0
        Lk[-1, -1] = 1.0
    else:
        Lk[-1, :] = D1[-1, :]
    return Lk


def _build_Hk_minus_reference(
    k2_val: float,
    D2: np.ndarray,
    dt: float,
    c: float,
    nu: float,
) -> np.ndarray:
    r"""Build single-mode `$H_k^-$` (explicit Helmholtz) in NumPy."""
    Ny = D2.shape[0]
    eye = np.eye(Ny)
    Hk_minus = (1.0 / dt) * eye + (1.0 - c) * nu * (D2 - k2_val * eye)
    Hk_minus[0, :] = eye[0, :]
    Hk_minus[-1, :] = eye[-1, :]
    return Hk_minus


def _perturbed_cgl_grid(Ny: int, seed: int = 42) -> np.ndarray:
    """Monotonically increasing non-uniform grid on [-1, 1]."""
    rng = np.random.default_rng(seed)
    pts = np.sort(rng.uniform(-1, 1, Ny))
    pts[0] = -1.0
    pts[-1] = 1.0
    return pts


# ── tests ────────────────────────────────────────────────────────────


def test_lk_matvec_matches_reference() -> None:
    r"""``_lk_matvec`` matches reference `$L_k u$` from D1/D2."""
    Ny, p = 17, 4
    y = -jnp.cos(jnp.arange(Ny) * jnp.pi / (Ny - 1))
    D1, D2 = build_diff_matrices(y, p)

    flow_ = SimpleNamespace(D2=D2, D1_bnd=D1[[0, -1], :])

    rng = np.random.default_rng(30)
    for kz, kx in [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]:
        k2_val = kx**2 + kz**2
        Lk = _build_Lk_reference(k2_val, np.asarray(D1), np.asarray(D2))
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Lk @ u

        u_j = jnp.asarray(u)[:, None, None]
        fourier_ = SimpleNamespace(
            k2=jnp.asarray([[[k2_val]]]),
            mean_mask=jnp.asarray([[[k2_val]]]) == 0.0,
        )
        got = np.asarray(_lk_matvec(u_j, flow_, fourier_))[:, 0, 0]
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"kz={kz}, kx={kx}",
        )


def test_hk_minus_matvec_matches_reference() -> None:
    r"""``_hk_minus_matvec`` matches reference `$H_k^- u$` from D2."""
    Ny, p = 17, 4
    dt = params.step.dt
    c = params.step.implicitness
    nu = 1.0 / params.phys.re
    y = -jnp.cos(jnp.arange(Ny) * jnp.pi / (Ny - 1))
    _, D2 = build_diff_matrices(y, p)

    flow_ = SimpleNamespace(D2=D2, dt=jnp.asarray(dt))

    rng = np.random.default_rng(40)
    for kz, kx in [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]:
        k2_val = kx**2 + kz**2
        Hk_minus = _build_Hk_minus_reference(k2_val, np.asarray(D2), dt, c, nu)
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Hk_minus @ u

        u_j = jnp.asarray(u)[:, None, None]
        fourier_ = SimpleNamespace(k2=jnp.asarray([[[k2_val]]]))
        got = np.asarray(_hk_minus_matvec(u_j, flow_, fourier_))[:, 0, 0]
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"kz={kz}, kx={kx}",
        )


def test_lk_matvec_on_custom_grid() -> None:
    r"""``_lk_matvec`` matches reference on a non-CGL grid."""
    Ny, p = 17, 4
    y_np = _perturbed_cgl_grid(Ny)
    y = jnp.asarray(y_np, dtype=jnp.float64)
    D1, D2 = build_diff_matrices(y, p)

    flow_ = SimpleNamespace(D2=D2, D1_bnd=D1[[0, -1], :])

    rng = np.random.default_rng(50)
    for kz, kx in [(0.0, 0.0), (0.0, 1.7), (2.0, 3.0)]:
        k2_val = kx**2 + kz**2
        Lk = _build_Lk_reference(k2_val, np.asarray(D1), np.asarray(D2))
        u = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        ref = Lk @ u

        u_j = jnp.asarray(u)[:, None, None]
        fourier_ = SimpleNamespace(
            k2=jnp.asarray([[[k2_val]]]),
            mean_mask=jnp.asarray([[[k2_val]]]) == 0.0,
        )
        got = np.asarray(_lk_matvec(u_j, flow_, fourier_))[:, 0, 0]
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"kz={kz}, kx={kx}",
        )


def test_get_norm2_cartesian() -> None:
    r"""``get_norm2`` matches a manual Parseval/quadrature sum.

    Pins the spectral contraction: sum `$|u|^2$` over components and the
    two periodic wavenumber axes with the real-FFT multiplicity
    ``k_metric``, integrate over `$y$` with the CGL quadrature weights,
    and normalise by ``derived_params.volume_fac``.
    """
    Ny = params.res.ny
    Nkz = params.res.nz - 1
    Nkx = params.res.nx // 2

    _, _, _, y_weights = build_cartesian_grid(Ny, params.res.fd_order)

    rng = np.random.default_rng(80)
    s_shape = (3, Ny, Nkz, Nkx)
    state_np = rng.standard_normal(s_shape) + 1j * rng.standard_normal(s_shape)
    state = jnp.asarray(state_np)

    k_metric = fourier.k_metric
    got = float(get_norm2(state, k_metric, y_weights))

    # Manual reference mirroring ``get_inprod`` in _base.py.
    k_metric_np = np.asarray(k_metric)
    y_w_np = np.asarray(y_weights)
    integrand = (np.abs(state_np) ** 2 * k_metric_np).sum(axis=(0, 2, 3))
    ref = float(y_w_np @ integrand) / derived_params.volume_fac

    assert_allclose(got, ref, atol=1e-12, err_msg="get_norm2 (cartesian)")


def test_measured_band_half_width() -> None:
    r"""``matrix_half_bandwidth`` sizes the band correctly.

    The banded builders take the half-width from the *assembled*
    operator rather than assuming ``fd_order`` (``CartesianFlow.
    __post_init__``).  Two properties matter: with
    the direct-fit `$D_2$` every Cartesian operator now uses must
    reproduce ``fd_order`` exactly -- that is what pins the band -- and
    a composed `$D_1 D_1$` (the annular/pipe ``consistent_imm``
    operators) must measure wider, its boundary-adjacent rows reaching
    further than the direct fit.
    Rows 0 and Ny-1 are excluded because every operator overwrites
    them with BC rows.
    """
    for ny in (25, 49):
        for p in (4, 6, 8):
            y = -np.cos(np.arange(ny) * np.pi / (ny - 1))
            D1, D2 = build_diff_matrices(y, p)
            assert matrix_half_bandwidth(D2, (0, -1)) == p, (ny, p)
            wide = matrix_half_bandwidth(D1 @ D1, (0, -1))
            assert wide > p, (ny, p, wide)
            # Nothing outside the measured band survives truncation.
            band = np.asarray(_banded_from_dense(jnp.asarray(D1 @ D1), wide))
            rebuilt = np.zeros_like(D1)
            for d in range(2 * wide + 1):
                for i in range(ny):
                    j = i - wide + d
                    if 0 <= j < ny:
                        rebuilt[i, j] = band[i, d]
            assert_allclose(rebuilt[1:-1], (D1 @ D1)[1:-1], atol=0.0)


def test_pallas_vs_dense_on_cartesian_operators() -> None:
    r"""``PerModeBandedPallasOperator`` matches ``DenseJAXSolver``.

    Validates the Pallas band assembly of all three Cartesian
    operators -- the Neumann `$L_k$`, the Dirichlet `$H_k$`, and the
    Dirichlet `$L_k$` the `$v$`-`$\omega_y$` scheme solves `$v$` with:
    each banded operator equals ``banded(dense)`` exactly, and the
    no-pivot banded sweep (CPU pure-JAX path) reproduces the dense
    solve on a complex RHS.

    Run twice: with the direct-fit `$D_2$` at the ``fd_order`` band,
    and with `$D_2 = D_1 D_1$` at its wider measured band (the
    annular/pipe ``consistent_imm`` operators) -- the assembly, the
    no-pivot factorisation and the Pallas sweep all have to hold at
    both widths.
    """
    Ny = params.res.ny
    y = -jnp.cos(jnp.arange(Ny) * jnp.pi / (Ny - 1))
    D1, D2_fit = build_diff_matrices(y, params.res.fd_order)
    for D2 in (D2_fit, D1 @ D1):
        _check_band_vs_dense(D1, D2, matrix_half_bandwidth(D2, (0, -1)))


def _check_band_vs_dense(D1: Array, D2: Array, p: int) -> None:
    """One (operator pair, band width) case of the parity check."""
    Ny = params.res.ny
    dt, c, nu = 0.01, 0.5, 1.0 / 1000.0
    k2_s = fourier.k2[0, ..., None]
    mean_s = fourier.mean_mask[0, ..., None]

    Lk_band = _build_Lk_band_gpu(D1, D2, k2_s, mean_s, p)
    Hk_band = _build_Hk_band_gpu(D2, k2_s, dt, c, nu, p)
    Ld_band = _build_Lk_dir_band_gpu(D2, k2_s, p)
    Lk_full = _build_Lk_dense_gpu(D1, D2, k2_s, mean_s)
    Hk_full = _build_Hk_dense_gpu(D2, k2_s, dt, c, nu)
    Ld_full = _build_Lk_dir_dense_gpu(D2, k2_s)

    # Band assembly equals banded(dense).
    to_band = jax.vmap(jax.vmap(lambda A: _banded_from_dense(A, p)))
    for band, full in (
        (Lk_band, Lk_full),
        (Hk_band, Hk_full),
        (Ld_band, Ld_full),
    ):
        assert_allclose(
            np.asarray(band), np.asarray(to_band(full)), atol=1e-12
        )

    # No-pivot banded solve reproduces the dense solve.
    pallas_ops = [
        PerModeBandedPallasOperator.from_banded_factors(*_banded_factor(b))
        for b in (Lk_band, Hk_band, Ld_band)
    ]
    dense_ops = [DenseJAXSolver(f) for f in (Lk_full, Hk_full, Ld_full)]

    Nkz, Nkx = int(fourier.k2.shape[1]), int(fourier.k2.shape[2])
    rng = np.random.default_rng(21)
    b = rng.standard_normal((Ny, Nkz, Nkx)) + 1j * rng.standard_normal(
        (Ny, Nkz, Nkx)
    )
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_scalar_shard)

    for pallas_op, dense_op in zip(pallas_ops, dense_ops, strict=True):
        assert_allclose(
            np.asarray(pallas_op.solve(rhs)),
            np.asarray(dense_op.solve(rhs)),
            atol=1e-9,
            rtol=1e-9,
        )


def test_vw_reconstruction_is_exactly_solenoidal() -> None:
    r"""The `$v$`-`$\omega_y$` reconstruction zeroes the divergence.

    The identity ``res.consistent_imm`` rests on
    (:func:`~dnsjax.geometries.wall_bounded.cartesian._imm_iteration_vw`):
    for any `$v$` and `$\omega_y$`,

    .. math::
        u = \frac{i}{k^2}(k_x D_1 v - k_z \omega_y), \qquad
        w = \frac{i}{k^2}(k_z D_1 v + k_x \omega_y)

    satisfies `$i k_x u + D_1 v + i k_z w = 0$` at **every** row --
    walls included -- as algebra, so in floating point the residual is
    a few ulps and does not care about the grid, the operator or the
    wavenumber.  Checked with the production `$D_1$` (the same object
    the divergence diagnostic uses) on random data, including the
    `$k_x = 0$` and `$k_z = 0$` lines, where a route that recovered
    one component from continuity alone would have been singular.
    """
    Ny = params.res.ny
    y = -np.cos(np.arange(Ny) * np.pi / (Ny - 1))
    D1, _ = build_diff_matrices(y, params.res.fd_order)
    rng = np.random.default_rng(3)

    for kx, kz in ((1.0, 0.0), (0.0, 2.5), (3.0, -4.0), (0.05, 0.05)):
        k2 = kx * kx + kz * kz
        v = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        om = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        dy_v = D1 @ v
        u = 1j * (kx * dy_v - kz * om) / k2
        w = 1j * (kz * dy_v + kx * om) / k2

        div = 1j * kx * u + dy_v + 1j * kz * w
        scale = max(np.abs(t).max() for t in (1j * kx * u, dy_v, 1j * kz * w))
        assert np.abs(div).max() < 100 * np.finfo(float).eps * scale, (
            kx,
            kz,
            np.abs(div).max() / scale,
        )

        # The round trip is exact too: the reconstruction reproduces
        # the vorticity it was built from.
        assert_allclose(1j * kz * u - 1j * kx * w, om, rtol=1e-13)

        # Condition number exactly 1 (M^H M = k^2 I), so the only
        # amplification is the uniform 1/sqrt(k2).
        M = np.array([[1j * kx, 1j * kz], [1j * kz, -1j * kx]])
        assert_allclose(M.conj().T @ M, k2 * np.eye(2), atol=1e-13)


def test_vw_source_projections_kill_gradients() -> None:
    r"""`$S_\varphi$` and `$S_\omega$` annihilate a discrete gradient.

    This is the discrete pressure elimination of the
    `$v$`-`$\omega_y$` scheme, and it is *exact* -- not exact-up-to-
    truncation -- because the per-mode scalar `$k^2$` commutes with
    `$D_1$`.  Applying the two projections to the gradient of an
    arbitrary field `$q$`, `$(i k_x q,\; D_1 q,\; i k_z q)$`, must
    give zero to round-off with the direct-fit `$D_2$` in play (the
    primitive scheme's elimination, by contrast, leaves the
    `$(D_2 - D_1^2)$` and `$[D_1, D_2]$` obstruction).
    """
    Ny = params.res.ny
    y = -np.cos(np.arange(Ny) * np.pi / (Ny - 1))
    D1, _ = build_diff_matrices(y, params.res.fd_order)
    rng = np.random.default_rng(5)

    for kx, kz in ((1.0, 0.0), (0.0, 2.5), (3.0, -4.0)):
        k2 = kx * kx + kz * kz
        q = rng.standard_normal(Ny) + 1j * rng.standard_normal(Ny)
        n_u, n_v, n_w = 1j * kx * q, D1 @ q, 1j * kz * q

        s_phi = -k2 * n_v - D1 @ (1j * kx * n_u + 1j * kz * n_w)
        s_om = 1j * kz * n_u - 1j * kx * n_w

        # Both residuals are normalised by the size of the terms that
        # cancel; a component-wise scale would vanish on the k_x = 0
        # and k_z = 0 lines, which are exactly the ones worth testing.
        assert np.abs(s_phi).max() < 1e-12 * k2 * np.abs(n_v).max(), (kx, kz)
        assert np.abs(s_om).max() < 1e-12 * k2 * np.abs(q).max(), (kx, kz)


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
