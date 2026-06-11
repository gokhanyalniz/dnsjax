"""Unit tests for quadrature weights.

Tests cover:

1. Clenshaw-Curtis weights on CGL grids (``clenshaw_curtis_weights``):
   weight sum, polynomial exactness, even/odd ny, spectral convergence.
2. Composite polynomial weights on arbitrary non-uniform grids
   (``build_integration_weights``): weight sum, polynomial exactness,
   convergence rate.

Run as a script via ``uv run python tests/test_integration.py``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from dnsjax.parameters import params  # noqa: E402

params.phys.system = "plane-couette"
params.res.nx = 4
params.res.ny = 16
params.res.nz = 4
params.res.double_precision = True

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.fd import (  # noqa: E402
    barycentric_interpolation_matrix,
    build_integration_weights,
    chebyshev_interpolation_matrix,
    half_cgl_interpolation_matrices,
    tanh_one_sided_grid,
)
from dnsjax.geometries.wall_bounded import integrate_scalar  # noqa: E402
from dnsjax.geometries.wall_bounded.cartesian import (  # noqa: E402
    clenshaw_curtis_weights,
)


def _cgl_grid(ny: int) -> jnp.ndarray:
    """CGL grid on [-1, 1]."""
    return -jnp.cos(jnp.arange(ny, dtype=jnp.float64) * jnp.pi / (ny - 1))


def _exact_monomial_integral(d: int) -> float:
    r"""Exact :math:`\int_{-1}^{1} y^d\,dy`."""
    if d % 2 == 1:
        return 0.0
    return 2.0 / (d + 1)


# ── Clenshaw-Curtis tests ──────────────────────────────────────────


def test_cc_weight_sum():
    """CC weights must sum to 2 for any ny >= 2."""
    for ny in [3, 4, 5, 8, 17, 27, 28, 32, 33, 64]:
        w = clenshaw_curtis_weights(ny)
        assert_allclose(
            float(jnp.sum(w)),
            2.0,
            atol=1e-14,
            err_msg=f"ny={ny}",
        )


def test_cc_polynomial_exactness():
    """CC with ny points must be exact for degree <= ny - 1."""
    for ny in [3, 4, 5, 8, 17, 28, 33]:
        ys = _cgl_grid(ny)
        w = clenshaw_curtis_weights(ny)
        for d in range(ny):
            computed = float(jnp.dot(w, ys**d))
            exact = _exact_monomial_integral(d)
            assert_allclose(
                computed,
                exact,
                atol=1e-12,
                err_msg=f"ny={ny}, degree={d}",
            )


def test_cc_even_ny():
    """Even ny values must work (previously unsupported)."""
    for ny in [4, 8, 28, 32]:
        w = clenshaw_curtis_weights(ny)
        ys = _cgl_grid(ny)
        assert_allclose(
            float(jnp.sum(w)),
            2.0,
            atol=1e-14,
            err_msg=f"weight sum, ny={ny}",
        )
        for d in range(ny):
            computed = float(jnp.dot(w, ys**d))
            exact = _exact_monomial_integral(d)
            assert_allclose(
                computed,
                exact,
                atol=1e-12,
                err_msg=f"ny={ny}, degree={d}",
            )


def test_cc_spectral_convergence():
    """Error on exp(y) must decrease exponentially."""
    exact = np.exp(1.0) - np.exp(-1.0)
    prev_err = None
    for ny in [5, 9, 17, 33]:
        w = clenshaw_curtis_weights(ny)
        ys = _cgl_grid(ny)
        err = abs(float(jnp.dot(w, jnp.exp(ys))) - exact)
        if prev_err is not None and prev_err > 1e-15:
            assert err < prev_err * 0.01, (
                f"ny={ny}: error {err:.2e} not << {prev_err:.2e}"
            )
        prev_err = err
    assert err < 1e-14, f"final error {err:.2e} not < 1e-14"


def test_cc_integrate_scalar():
    """integrate_scalar must match direct dot product."""
    ny = 27
    w = clenshaw_curtis_weights(ny)
    ys = _cgl_grid(ny)
    f = jnp.sin(jnp.pi * ys)
    assert_allclose(
        float(integrate_scalar(f, w)),
        float(jnp.dot(w, f)),
        atol=1e-15,
    )


# ── Composite polynomial quadrature tests ──────────────────────────


def _perturbed_grid(ny: int, seed: int = 42) -> jnp.ndarray:
    """Monotonically increasing non-uniform grid on [-1, 1]."""
    rng = np.random.default_rng(seed)
    pts = np.sort(rng.uniform(-1, 1, ny))
    pts[0] = -1.0
    pts[-1] = 1.0
    return jnp.asarray(pts, dtype=jnp.float64)


def test_composite_weight_sum():
    """Composite weights must sum to interval length."""
    for ny in [5, 8, 16, 33]:
        for grid_fn in [_cgl_grid, _perturbed_grid]:
            ys = grid_fn(ny)
            w = build_integration_weights(ys, p=4)
            interval_length = float(ys[-1] - ys[0])
            assert_allclose(
                float(jnp.sum(w)),
                interval_length,
                atol=1e-12,
                err_msg=f"ny={ny}, grid={grid_fn.__name__}",
            )


def test_composite_polynomial_exactness():
    """Composite with stencil order p must be exact for
    degree <= p on non-uniform grids."""
    for p in [2, 4, 6]:
        ny = 20
        ys = _perturbed_grid(ny)
        w = build_integration_weights(ys, p=p)
        for d in range(p + 1):
            computed = float(jnp.dot(w, ys**d))
            exact = _exact_monomial_integral(d)
            assert_allclose(
                computed,
                exact,
                atol=1e-10,
                err_msg=f"p={p}, degree={d}",
            )


def test_composite_convergence_rate():
    """Error must decrease as O(h^{p+1})."""
    p = 4
    exact = np.exp(1.0) - np.exp(-1.0)
    errors = []
    nys = [11, 21, 41, 81]
    for ny in nys:
        ys = _cgl_grid(ny)
        w = build_integration_weights(ys, p=p)
        err = abs(float(jnp.dot(w, jnp.exp(ys))) - exact)
        errors.append(err)
    for i in range(1, len(errors)):
        ratio = errors[i - 1] / max(errors[i], 1e-16)
        assert ratio > 4, f"ny={nys[i]}: ratio {ratio:.1f} (expected >> 4)"


def _radial_weights(rs: np.ndarray, p: int) -> np.ndarray:
    """Radial weights as built by ``build_cylindrical_grid``:
    composite rule over the full disc (``left_edge=0.0``)
    times the radial Jacobian."""
    return build_integration_weights(rs, p, left_edge=0.0) * rs


def test_radial_weights_full_disc():
    r"""Radial weights must cover [0, 1] including the
    `$[0, r_0]$` segment below the first grid point.

    The integrand `$g = r f$` is interpolated per interval,
    so the rule is exact whenever `$g$` is a polynomial of
    degree `$\le p$`: checks `$\int_0^1 r\,dr = 1/2$`,
    `$\int_0^1 r^3\,dr = 1/4$`, and the laminar pipe bulk
    `$2\int_0^1 (1-r^2)\,r\,dr = 1/2$`.
    """
    cases = [
        ("half-cgl", _half_cgl_grid),
        ("tanh", lambda nr: tanh_one_sided_grid(nr, 1.5)),
    ]
    for name, grid_fn in cases:
        for nr in [8, 16, 32, 64]:
            rs = np.asarray(grid_fn(nr), dtype=np.float64)
            for p in [4, 8]:
                if nr < 2 * p:
                    continue
                yw = _radial_weights(rs, p)
                assert_allclose(
                    yw.sum(),
                    0.5,
                    atol=1e-12,
                    err_msg=f"int r dr, {name}, nr={nr}, p={p}",
                )
                assert_allclose(
                    (yw * rs**2).sum(),
                    0.25,
                    atol=1e-12,
                    err_msg=f"int r^3 dr, {name}, nr={nr}, p={p}",
                )
                assert_allclose(
                    2 * (yw * (1 - rs**2)).sum(),
                    0.5,
                    atol=1e-12,
                    err_msg=f"laminar bulk, {name}, nr={nr}, p={p}",
                )


def test_radial_weights_convergence_rate():
    r"""Full-disc radial error must decrease as
    `$O(h^{p+1})$` for a non-polynomial smooth-even
    integrand (no `$O(N_r^{-2})$` floor from a missing
    `$[0, r_0]$` segment).

    Uses `$\int_0^1 e^{r^2} r\,dr = (e-1)/2$`.
    """
    exact = (np.exp(1.0) - 1.0) / 2.0
    for p in [2, 4]:
        errors = []
        nrs = [8, 16, 32, 64]
        for nr in nrs:
            rs = np.asarray(_half_cgl_grid(nr), dtype=np.float64)
            yw = _radial_weights(rs, p)
            errors.append(abs((yw * np.exp(rs**2)).sum() - exact))
        for i in range(1, len(errors)):
            ratio = errors[i - 1] / max(errors[i], 1e-16)
            assert ratio > 4, (
                f"p={p}, nr={nrs[i]}: ratio {ratio:.1f} (expected >> 4)"
            )


def test_radial_weights_left_edge_validation():
    """left_edge above the first grid point must raise."""
    rs = np.asarray(_half_cgl_grid(8), dtype=np.float64)
    raised = False
    try:
        build_integration_weights(rs, 4, left_edge=2 * rs[0])
    except ValueError:
        raised = True
    assert raised, "left_edge > y[0] should raise ValueError"


def test_composite_vs_cc_on_cgl():
    """On a CGL grid, both methods must agree for low-degree
    polynomials."""
    ny = 17
    p = 4
    ys = _cgl_grid(ny)
    w_cc = clenshaw_curtis_weights(ny)
    w_comp = build_integration_weights(ys, p=p)
    for d in range(p + 1):
        f = ys**d
        assert_allclose(
            float(jnp.dot(w_cc, f)),
            float(jnp.dot(w_comp, f)),
            atol=1e-12,
            err_msg=f"degree={d}",
        )


# ── Interpolation tests ──────────────────────────────────────────


def test_chebyshev_interp_identity():
    """CGL N -> N must give the identity matrix."""
    for ny in [5, 17, 33]:
        T = chebyshev_interpolation_matrix(ny, ny)
        assert_allclose(T, np.eye(ny), atol=1e-12, err_msg=f"ny={ny}")


def test_chebyshev_interp_polynomial():
    """Low-degree polynomials must survive CGL interpolation."""
    ny_old, ny_new = 17, 33
    T = chebyshev_interpolation_matrix(ny_old, ny_new)
    y_old = np.asarray(_cgl_grid(ny_old))
    y_new = np.asarray(_cgl_grid(ny_new))
    for d in range(ny_old):
        f_old = y_old**d
        f_new = T @ f_old
        assert_allclose(
            f_new,
            y_new**d,
            atol=1e-10,
            err_msg=f"degree={d}",
        )


def test_chebyshev_interp_truncation():
    """Downsampling must preserve low-degree content."""
    ny_old, ny_new = 33, 17
    T = chebyshev_interpolation_matrix(ny_old, ny_new)
    y_old = np.asarray(_cgl_grid(ny_old))
    y_new = np.asarray(_cgl_grid(ny_new))
    for d in range(ny_new):
        f_old = y_old**d
        f_new = T @ f_old
        assert_allclose(
            f_new,
            y_new**d,
            atol=1e-10,
            err_msg=f"degree={d}",
        )


def _half_cgl_grid(nr):
    N_full = 2 * nr
    s = np.cos(np.arange(N_full) * np.pi / (N_full - 1))
    return -s[nr:]


def test_half_cgl_interp_even_parity():
    """Even-parity fields must survive half-CGL interpolation."""
    nr_old, nr_new = 16, 24
    T_even, _ = half_cgl_interpolation_matrices(nr_old, nr_new)
    r_old = _half_cgl_grid(nr_old)
    r_new = _half_cgl_grid(nr_new)
    # Even-parity test: f(r) = r^2 (even function of r)
    for d in [0, 2, 4]:
        f_old = r_old**d
        f_new = T_even @ f_old
        assert_allclose(
            f_new,
            r_new**d,
            atol=1e-10,
            err_msg=f"even degree={d}",
        )


def test_half_cgl_interp_odd_parity():
    """Odd-parity fields must survive half-CGL interpolation."""
    nr_old, nr_new = 16, 24
    _, T_odd = half_cgl_interpolation_matrices(nr_old, nr_new)
    r_old = _half_cgl_grid(nr_old)
    r_new = _half_cgl_grid(nr_new)
    for d in [1, 3, 5]:
        f_old = r_old**d
        f_new = T_odd @ f_old
        assert_allclose(
            f_new,
            r_new**d,
            atol=1e-10,
            err_msg=f"odd degree={d}",
        )


def test_barycentric_polynomial():
    """Barycentric must exactly interpolate polynomials."""
    ny_old = 17
    ny_new = 25
    y_old = np.asarray(_perturbed_grid(ny_old))
    y_new = np.asarray(_perturbed_grid(ny_new, seed=99))
    T = barycentric_interpolation_matrix(y_old, y_new)
    for d in range(ny_old):
        f_old = y_old**d
        f_new = T @ f_old
        expected = y_new**d
        assert_allclose(
            f_new,
            expected,
            atol=1e-7,
            err_msg=f"degree={d}",
        )


def test_barycentric_vs_chebyshev():
    """On CGL grids, barycentric and Chebyshev must agree."""
    ny_old, ny_new = 17, 25
    T_cheb = chebyshev_interpolation_matrix(ny_old, ny_new)
    y_old = np.asarray(_cgl_grid(ny_old))
    y_new = np.asarray(_cgl_grid(ny_new))
    T_bary = barycentric_interpolation_matrix(y_old, y_new)
    f_old = np.sin(np.pi * y_old)
    assert_allclose(
        T_cheb @ f_old,
        T_bary @ f_old,
        atol=1e-10,
    )


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
