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

from dnsjax.fd import build_integration_weights  # noqa: E402
from dnsjax.geometries.cartesian import (  # noqa: E402
    clenshaw_curtis_weights,
    integrate_scalar_in_y,
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


def test_cc_integrate_scalar_in_y():
    """integrate_scalar_in_y must match direct dot product."""
    ny = 27
    w = clenshaw_curtis_weights(ny)
    ys = _cgl_grid(ny)
    f = jnp.sin(jnp.pi * ys)
    assert_allclose(
        float(integrate_scalar_in_y(f, w)),
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


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
