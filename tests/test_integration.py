"""Unit tests for quadrature weights and interpolation matrices.

Tests cover:

1. Clenshaw-Curtis weights on CGL grids (``clenshaw_curtis_weights``):
   weight sum, polynomial exactness, even/odd ny, spectral convergence.
2. Composite polynomial weights on arbitrary non-uniform grids
   (``build_integration_weights``): weight sum, polynomial exactness,
   convergence rate.
3. The parity-free axis-augmented radial quadrature (the cylindrical
   solver rule): positivity and even+odd exactness.
4. Interpolation matrices: Chebyshev, the ``cgl_axis_gap`` detector,
   the spectral ``cgl_parity_interpolation_matrices`` (half / rigged /
   mixed, both parities), and the local Fornberg fallback (bounded
   Lebesgue vs a global fit's blow-up).

Run as a script via ``uv run python tests/test_integration.py``.
"""

from __future__ import annotations

# Select the JAX backend from --dist.platform (default cpu) before
# importing any dnsjax module that captures the platform (the geometry
# import below builds sharding).  This suite is quadrature / interpolation
# math -- device-agnostic -- but honours --dist.platform for consistency.
from dnsjax.bootstrap import (  # noqa: E402
    configure_jax_platform,
    platform_from_argv,
)
from dnsjax.parameters import (  # noqa: E402
    params,
)

configure_jax_platform(platform_from_argv())

params.phys.system = "plane-couette"
params.res.nx = 4
params.res.ny = 16
params.res.nz = 4
params.res.double_precision = True

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.fd import (  # noqa: E402
    build_integration_weights,
    cgl_axis_gap,
    cgl_parity_interpolation_matrices,
    cgl_radial_quadrature_weights,
    chebyshev_interpolation_matrix,
    clenshaw_curtis_weights,
    local_grid_spacing,
    local_interpolation_matrix,
    tanh_one_sided_grid,
)
from dnsjax.geometries.wall_bounded import integrate_scalar  # noqa: E402


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
    """Exercise ``build_integration_weights``'s ``left_edge`` option:
    the composite rule over the full disc (``left_edge=0.0``) times
    the radial Jacobian.  (The cylindrical solver no longer uses this
    -- it uses the spectral parity CC / axis-augmented rule -- but
    ``left_edge`` remains a supported feature.)"""
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


def _radial_cgl_grid(nr, gap):
    """Cylindrical radial grid: outer ``nr`` positive points of a
    ``2*nr + gap``-point CGL grid (gap 0 = half-CGL, 1 = rigged-CGL)."""
    n_full = 2 * nr + gap
    s = -np.cos(np.arange(n_full) * np.pi / (n_full - 1))
    return s[nr + gap :]


def _augmented_radial_weights(rs, p):
    """The solver's parity-free full-disc rule: integrate g = f*r on
    the axis-augmented grid [0, *rs], then drop the axis node."""
    r_aug = np.concatenate([[0.0], rs])
    return build_integration_weights(r_aug, p)[1:] * rs


def test_augmented_axis_quadrature():
    """Parity-free axis-augmented radial quadrature (the solver rule):
    strictly positive AND exact for even *and odd* monomials up to the
    interior order.  The retired even-parity gap rule erred O(r0^3) on
    odd integrands (e.g. the mean u_theta); the augmented rule uses the
    axis node g(0)=0, which holds for any bounded f -- no parity."""
    p = 4
    for gap in (0, 1):
        for nr in (8, 16, 32):
            rs = _radial_cgl_grid(nr, gap)
            yw = _augmented_radial_weights(rs, p)
            assert np.all(yw > 0), f"gap={gap} nr={nr}: negative weight"
            # int_0^1 r^d * r dr = 1/(d+2), exact for d+1 <= p, both
            # parities (odd d included -- the regression guard).
            for d in (0, 1, 2, 3):
                assert_allclose(
                    float(yw @ rs**d),
                    1.0 / (d + 2),
                    atol=1e-12,
                    err_msg=f"gap={gap} nr={nr} d={d}",
                )


def test_cgl_radial_quadrature_weights():
    """Parity-specific spectral radial quadrature (the pipe's CC):
    (w_even, w_odd) are both strictly positive (definite energy norm)
    and each is spectral for its parity -- exact for the polynomial
    moments int r^d * r dr = 1/(d+2) (even d via w_even, odd d via
    w_odd) and machine precision on a smooth integrand.  None for a
    non-CGL grid (caller falls back to the composite rule)."""
    fine = np.linspace(0.0, 1.0, 2_000_001)
    ref_even = float(np.trapezoid(np.cos(2.0 * fine) * fine, fine))
    ref_odd = float(np.trapezoid(fine * np.cos(2.0 * fine) * fine, fine))
    for gap in (0, 1):
        for nr in (8, 16, 48):
            rs = _radial_cgl_grid(nr, gap)
            w_even, w_odd = cgl_radial_quadrature_weights(rs, 4)
            assert np.all(w_even > 0), f"gap={gap} nr={nr}: w_even<0"
            assert np.all(w_odd > 0), f"gap={gap} nr={nr}: w_odd<0"
            for d in (0, 2):  # even moments via w_even
                assert_allclose(
                    float(w_even @ rs**d), 1.0 / (d + 2), atol=1e-12
                )
            for d in (1, 3):  # odd moments via w_odd
                assert_allclose(
                    float(w_odd @ rs**d), 1.0 / (d + 2), atol=1e-12
                )
            # Spectral on smooth integrands (once resolved: nr=8 is too
            # coarse for cos(2r), but the polynomial moments are exact
            # at every nr).
            if nr >= 16:
                assert abs(w_even @ np.cos(2.0 * rs) - ref_even) < 1e-10
                assert abs(w_odd @ (rs * np.cos(2.0 * rs)) - ref_odd) < 1e-10
    # Non-CGL grid -> None (caller uses the composite fallback).
    assert cgl_radial_quadrature_weights(np.linspace(0.1, 1.0, 20), 4) is None


def test_cgl_axis_gap_detector():
    """cgl_axis_gap: half-CGL -> 0, rigged-CGL -> 1, non-CGL -> None."""
    for nr in (12, 24, 48):
        assert cgl_axis_gap(_radial_cgl_grid(nr, 0)) == 0
        assert cgl_axis_gap(_radial_cgl_grid(nr, 1)) == 1
    assert cgl_axis_gap(np.linspace(0.1, 1.0, 24)) is None
    assert cgl_axis_gap(np.asarray(tanh_one_sided_grid(24, 1.5))) is None


def test_cgl_parity_interpolation_spectral():
    """Spectral parity interpolation between radial CGL grids (half,
    rigged, and mixed): near machine precision for both parities and
    orders of magnitude better than the local fd_order fallback."""

    def even(r):  # sigma = +1
        return np.cos(0.6 * np.pi * r) + 0.3 * r**2

    def odd(r):  # sigma = -1
        return r * np.cos(2.0 * r)

    nr_old, nr_new = 20, 28
    for go, gn in [(0, 0), (1, 1), (1, 0), (0, 1)]:
        ro = _radial_cgl_grid(nr_old, go)
        rn = _radial_cgl_grid(nr_new, gn)
        t_even, t_odd = cgl_parity_interpolation_matrices(
            nr_old, nr_new, go, gn, 4
        )
        e_even = np.max(np.abs(t_even @ even(ro) - even(rn)))
        e_odd = np.max(np.abs(t_odd @ odd(ro) - odd(rn)))
        assert e_even < 1e-8, f"gap {go}->{gn} even: {e_even:.2e}"
        assert e_odd < 1e-10, f"gap {go}->{gn} odd: {e_odd:.2e}"
        e_local = np.max(
            np.abs(local_interpolation_matrix(ro, rn, 4) @ even(ro) - even(rn))
        )
        assert e_local > 100 * e_even, (
            f"gap {go}->{gn}: spectral ({e_even:.2e}) not beating "
            f"local ({e_local:.2e})"
        )


def test_local_interp_polynomial_exactness():
    """Local ``order``-stencil interpolation is exact up to degree
    ``order`` (each stencil is an ``order``-degree Lagrange fit)."""
    order = 4
    y_old = np.asarray(_perturbed_grid(19))
    y_new = np.asarray(_perturbed_grid(27, seed=7))
    T = local_interpolation_matrix(y_old, y_new, order)
    for d in range(order + 1):
        assert_allclose(
            T @ y_old**d, y_new**d, atol=1e-10, err_msg=f"degree={d}"
        )


def test_local_interp_bounded_on_radial_grids():
    """The local Fornberg fallback stays well-conditioned (Lebesgue
    ``O(10)``) on the lopsided radial CGL grids, in contrast to a
    *global* Lagrange fit whose Lebesgue constant blows up ``>=1e6``
    there (the reason the fallback is local, not global; detected CGL
    grids take the spectral parity path instead)."""

    def global_lagrange_lebesgue(old, new):
        # Barycentric (global degree-N) interpolation Lebesgue const.
        w = np.array(
            [
                1.0 / np.prod(old[j] - np.delete(old, j))
                for j in range(len(old))
            ]
        )
        rows = []
        for x in new:
            d = x - old
            if np.any(np.abs(d) < 1e-14):  # x coincides with a node
                rows.append(1.0)
                continue
            t = w / d
            rows.append(np.abs(t / t.sum()).sum())
        return max(rows)

    for nr in (24, 32, 48):
        old = _radial_cgl_grid(nr, 0)  # half-CGL
        new = _radial_cgl_grid(nr, 1)  # rigged-CGL
        leb_loc = (
            np.abs(local_interpolation_matrix(old, new, 4)).sum(axis=1).max()
        )
        leb_global = global_lagrange_lebesgue(old, new)
        assert leb_loc < 20.0, f"nr={nr}: local Lebesgue {leb_loc:.2e}"
        assert leb_global >= 1e6, (
            f"nr={nr}: global Lebesgue {leb_global:.2e} "
            "(guard the motivating blow-up)"
        )


def test_local_interp_radial_convergence():
    """Local radial interpolation of a smooth field converges under
    grid refinement (both interpolation and axis-ward extrapolation
    directions stay bounded and small)."""

    def smooth(r):  # even near r = 0, u_z-like
        return np.cos(0.5 * np.pi * r) + 0.3 * r**2

    prev = {"up": np.inf, "down": np.inf}
    for nr in (16, 32, 64):
        old0, old1 = _radial_cgl_grid(nr, 0), _radial_cgl_grid(nr, 1)
        # half -> rigged (pure interpolation; new r_0 above old r_0)
        e_up = np.max(
            np.abs(
                local_interpolation_matrix(old0, old1, 4) @ smooth(old0)
                - smooth(old1)
            )
        )
        # rigged -> half (extrapolation toward the axis)
        e_down = np.max(
            np.abs(
                local_interpolation_matrix(old1, old0, 4) @ smooth(old1)
                - smooth(old0)
            )
        )
        assert e_up < prev["up"], f"nr={nr}: up err not decreasing"
        assert e_down < prev["down"], f"nr={nr}: down err not decreasing"
        prev = {"up": e_up, "down": e_down}
    assert prev["up"] < 1e-6 and prev["down"] < 1e-4


# ── Local grid spacing (CFL diagnostic) ────────────────────────────


def test_local_grid_spacing():
    """Min-neighbour spacing, one-sided at the ends."""
    nodes = np.array([0.0, 1.0, 3.0, 4.0, 8.0])
    expected = np.array([1.0, 1.0, 1.0, 1.0, 4.0])
    assert_allclose(local_grid_spacing(nodes), expected)

    # Uniform grid: constant spacing everywhere.
    uniform = np.linspace(-1.0, 1.0, 9)
    assert_allclose(local_grid_spacing(uniform), 0.25)

    # CGL grid: finest near the walls, coarsest in the centre;
    # one-sided ends equal the first/last gap.
    cgl = np.asarray(_cgl_grid(11))
    sp = local_grid_spacing(cgl)
    gaps = np.diff(cgl)
    assert_allclose(sp[0], gaps[0])
    assert_allclose(sp[-1], gaps[-1])
    assert_allclose(sp[1:-1], np.minimum(gaps[:-1], gaps[1:]))
    assert np.argmax(sp) == 5  # centre of the grid


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
