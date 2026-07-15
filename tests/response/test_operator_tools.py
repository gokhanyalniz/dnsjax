r"""Operator-tools tests (``dnsjax.analysis.response.operator_tools``).

Two layers:

1. **Analytic units** (NumPy/SciPy, no solver): Lyapunov/Gramian
   closed forms (`$A = -aI \Rightarrow X = I/2a$`, on both the SciPy
   and the eigendecomposition-fallback paths), controllability-mode
   ordering on a normal operator, stability rejection, growth curves
   against `$e^{2\lambda_{\max} t}$` (normal) and a dense
   ``scipy.linalg.expm`` reference (non-normal), the
   input-response curve of an eigenvector, full-rank ``restrict``
   identity, and the non-orthonormal-restriction rejection.  The
   module import itself must not pull JAX (asserted first); the
   growth-curve calls then enable float64 JAX explicitly.

2. **Export faithfulness** (subprocess): a laminar plane-Poiseuille
   transient-growth run with ``--save-operator`` (the real writer);
   the loaded bundle must satisfy the coordinate contract
   (``T_proj @ T_lift = I``), reproduce the stored resolved
   eigenvalues (assignment-free nearest matching), and
   ``growth_curve(A, t_grid)`` must match the stored ``G`` curve
   *exactly* (rtol 1e-9) -- the CLI reports growth on the same
   resolved-eigenspace restriction it exports, so this doubles as the
   guard on that restriction.  The controllability CLI then exports
   lifted modes whose energy Gram matrix is the identity.

Usage::

    uv run python tests/response/test_operator_tools.py
"""

from __future__ import annotations

import sys

from dnsjax.analysis.response import operator_tools as ot

assert "jax" not in sys.modules, "importing operator_tools must not import JAX"

import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

NX, NY, NZ = 8, 33, 8


# ── Analytic units ───────────────────────────────────────────────────


def test_gramian_closed_forms() -> None:
    r"""`$A = -aI \Rightarrow X = I/(2a)$` on both solver paths; an
    unstable `$A$` is rejected."""
    r, a = 5, 0.7
    a_mat = -a * np.eye(r, dtype=complex)
    for x in (
        ot.controllability_gramian(a_mat),  # scipy path (installed)
        ot._gramian_eig_closed_form(a_mat),  # fallback path
    ):
        assert_allclose(x, np.eye(r) / (2 * a), atol=1e-12)

    # A random stable non-normal case: both paths agree and satisfy
    # the Lyapunov equation.
    rng = np.random.default_rng(0)
    m = rng.standard_normal((r, r)) + 1j * rng.standard_normal((r, r))
    a_mat = m - (np.max(np.linalg.eigvals(m).real) + 1.0) * np.eye(r)
    x1 = ot.controllability_gramian(a_mat)
    x2 = ot._gramian_eig_closed_form(a_mat)
    assert_allclose(x1, x2, atol=1e-10)
    resid = a_mat @ x1 + x1 @ a_mat.conj().T + np.eye(r)
    assert np.max(np.abs(resid)) < 1e-12

    try:
        ot.controllability_gramian(np.eye(2))
    except ValueError as e:
        assert "not stable" in str(e)
    else:
        raise AssertionError("unstable A was accepted")


def test_controllability_modes_ordering() -> None:
    r"""Normal `$A = \mathrm{diag}(-1, -4)$`: `$X = \mathrm{diag}(1/2,
    1/8)$`, so the leading mode is `$e_1$`."""
    vals, p = ot.controllability_modes(np.diag([-1.0, -4.0]), 2)
    assert_allclose(vals, [0.5, 0.125], atol=1e-14)
    assert_allclose(np.abs(p), np.eye(2), atol=1e-12)
    assert np.max(np.abs(p.conj().T @ p - np.eye(2))) < 1e-13


def test_growth_curves() -> None:
    """Normal + non-normal growth curves and the input response."""
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    ts = np.linspace(0.0, 2.0, 9)
    # Normal: G(t) = exp(2 lambda_max t).
    g = ot.growth_curve(np.diag([-1.0, -2.0]), ts, t_chunk=4)
    assert_allclose(g, np.exp(-2.0 * ts), rtol=1e-12)

    # Non-normal: dense scipy reference.
    from scipy.linalg import expm as scipy_expm

    a_mat = np.array([[-1.0, 5.0], [0.0, -2.0]], dtype=complex)
    g = ot.growth_curve(a_mat, ts)
    ref = np.array(
        [
            np.linalg.svd(scipy_expm(t * a_mat), compute_uv=False)[0] ** 2
            for t in ts
        ]
    )
    assert_allclose(g, ref, rtol=1e-11)
    assert g.max() > 1.5  # genuinely transient growth

    # Input response of an eigenvector: exp(2 Re(lambda) t); the
    # envelope bounds it.
    r = ot.input_response_curve(
        np.diag([-1.0, -2.0]), np.array([0.0, 1.0]), ts
    )
    assert_allclose(r, np.exp(-4.0 * ts), rtol=1e-12)
    assert np.all(r <= g + 1e-12)

    try:
        ot.input_response_curve(a_mat, np.zeros(2), ts)
    except ValueError:
        pass
    else:
        raise AssertionError("zero input was accepted")


def test_restrict() -> None:
    """Full-rank restriction is exact; non-orthonormal columns are
    rejected."""
    rng = np.random.default_rng(1)
    a_mat = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    p_full, _ = np.linalg.qr(
        rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    )
    assert_allclose(
        ot.restrict(a_mat, p_full),
        p_full.conj().T @ a_mat @ p_full,
        atol=1e-14,
    )
    try:
        ot.restrict(a_mat, 2.0 * p_full[:, :2])
    except ValueError as e:
        assert "orthonormal" in str(e)
    else:
        raise AssertionError("non-orthonormal restriction was accepted")


# ── Export faithfulness (subprocess against the real writer) ─────────


def _run_tg_with_operator(tmp: Path) -> tuple[Path, Path]:
    """Laminar plane-Poiseuille TG run with ``--save-operator``."""
    y = np.cos(np.pi * np.arange(NY) / (NY - 1))
    with open(tmp / "lam.txt", "w") as f:
        for yi, ui in zip(y, 1.0 - y**2, strict=True):
            f.write(f"{yi:+.17e} {ui:+.17e}\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnsjax.analysis.transient_growth",
            "--profile",
            str(tmp / "lam.txt"),
            "--out-dir",
            str(tmp),
            "--modes",
            "1,0;3,0",
            "--nt",
            "17",
            "--save-operator",
            "--phys.system",
            "plane-poiseuille",
            "--res.nx",
            str(NX),
            "--res.ny",
            str(NY),
            "--res.nz",
            str(NZ),
            "--res.fd_order",
            "4",
        ],
        capture_output=True,
        text=True,
        cwd=tmp,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp / "lam_tg.npz", tmp / "lam_tg_op.npz"


def test_export_faithfulness_and_cli() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        tg_npz, op_npz = _run_tg_with_operator(tmp)
        assert ot.available_modes(op_npz) == [(1, 0), (3, 0)]
        op = ot.load_operator(op_npz, 1, 0)
        assert op.system == "plane-poiseuille"
        assert op.family == "cartesian"
        assert op.k_metric == 1.0 and op.volume_fac == 2.0
        r_res = op.A.shape[0]
        assert op.Q.shape == (op.F.shape[0], r_res)
        assert op.V.shape == (3 * NY, op.F.shape[0])

        # Coordinate contract: projection is a left inverse of the
        # lift on the resolved subspace.
        assert np.max(np.abs(op.T_proj @ op.T_lift - np.eye(r_res))) < 1e-12

        # eig(A) reproduces the stored resolved eigenvalues
        # (assignment-free nearest matching; complex sorting cannot
        # pair reliably).
        eig_a = np.linalg.eigvals(op.A)
        for lam in op.lam:
            d = np.min(np.abs(eig_a - lam)) / (1.0 + np.abs(lam))
            assert d < 1e-8, (lam, d)

        # growth_curve(A) reproduces the stored G(t).  The agreement is
        # exact (not approximate): the CLI measures growth on the same
        # resolved-eigenspace restriction it exports as ``A``, via its
        # eigenform rather than ``expm``, so this pins both the export
        # faithfulness and that restriction.  Admitting the unresolved
        # modes into the reported G -- which turns the propagator into
        # a spectral projector at t = 0+ (the ``_analyze_mode`` step-5
        # note) -- breaks this at small t by orders of magnitude.
        with np.load(tg_npz) as z:
            g_stored = z["G"][0]
            t_grid = z["t_grid"]
        g = ot.growth_curve(op.A, t_grid)
        assert g[0] == 1.0
        assert_allclose(g, g_stored, rtol=1e-9)

        # The Gramian of the exported operator solves its Lyapunov
        # equation.
        x = ot.controllability_gramian(op.A)
        resid = op.A @ x + x @ op.A.conj().T + np.eye(r_res)
        assert np.max(np.abs(resid)) < 1e-9

        # Controllability CLI: lifted modes are W-orthonormal.
        cont_npz = tmp / "cont.npz"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnsjax.analysis.response.operator_tools",
                "--operator",
                str(op_npz),
                "--modes",
                "1,0",
                "--n-modes",
                "5",
                "--out",
                str(cont_npz),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with np.load(cont_npz) as z:
            lifted = z["profiles_1_0"]
            gram_eigvals = z["gram_eigvals_1_0"]
            assert str(np.asarray(z["system"])) == "plane-poiseuille"
            assert_allclose(z["code_grid"], op.y)
        assert lifted.shape == (5, 3, NY)
        assert np.all(np.diff(gram_eigvals) <= 0) and gram_eigvals[0] > 0
        flat = lifted.reshape(5, -1)
        gram = flat.conj() @ (op.w_diag[:, None] * flat.T)
        assert_allclose(gram, np.eye(5), atol=1e-10)

        # Restricting A to the leading controllability modes gives a
        # growth curve bounded by the full one.
        _, p = ot.controllability_modes(op.A, 5)
        g_r = ot.growth_curve(ot.restrict(op.A, p), t_grid)
        assert np.all(g_r <= g * (1.0 + 1e-9))


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
