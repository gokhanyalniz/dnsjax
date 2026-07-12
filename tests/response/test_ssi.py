r"""SSI cross-covariance identification tests (``response.ssi``).

All offline (no solver runs; the solver-side kick machinery is
``tests/test_forcing.py``):

1. **Estimator exactness**: kick/response windows constructed with
   *only* intra-window responses (no cross-kick contamination) --
   the empirical-covariance regression recovers ``M(l)`` to
   roundoff, ``identify_generator`` recovers ``L``, and the
   causality entry reports exactly the planted lag-0 level.
2. **Discrete Lyapunov**: the SciPy-free eigendecomposition fallback
   matches ``solve_discrete_lyapunov``, and
   ``predicted_forced_variance`` matches a long kicked-linear-system
   simulation (pre-kick sampling convention).
3. **File pipeline (statistical)**: a kicked linear system with an
   independent stochastic background, simulated on a real
   transient-growth operator bundle and written as
   ``probes.bin``/``forcing.bin`` streams -- ``identify_ssi``
   recovers the restricted reference generator to ~10 %, with a
   small causality level, and pools two runs.
4. **read_forcing error paths**: missing sidecar, truncated record.

Usage::

    uv run python tests/response/test_ssi.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from dnsjax.parameters import (  # noqa: E402
    Parameters,
    padded_res,
    params,
    update_parameters,
    validate_parameters,
)

NX, NY, NZ = 8, 25, 8
update_parameters(
    Parameters(
        phys={"system": "plane-poiseuille"},
        res={
            "nx": NX,
            "ny": NY,
            "nz": NZ,
            "fd_order": 4,
            "double_precision": True,
        },
    )
)
validate_parameters()
padded_res.set_padded_resolution(params)

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.analysis.response import operator_tools as ot  # noqa: E402
from dnsjax.analysis.response.ensemble import (  # noqa: E402
    identify_generator,
)
from dnsjax.analysis.response.ssi import (  # noqa: E402
    _discrete_lyapunov_eig,
    cross_propagators,
    identify_ssi,
    predicted_forced_variance,
    read_forcing,
)

IT_PROBES = 10
DT = float(params.step.dt)
DELTA = IT_PROBES * DT


def _stable_generator(m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    return a - (np.max(np.linalg.eigvals(a).real) + 1.0) * np.eye(m)


def _cn_noise(rng, *shape) -> np.ndarray:
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)


# ── Estimator exactness ──────────────────────────────────────────────


def test_estimator_exactness() -> None:
    """Uncontaminated windows: M(l) and L exact; causality = planted."""
    from scipy.linalg import expm

    m, n_kicks, max_lag, eps = 3, 40, 5, 0.01
    l_true = _stable_generator(m, seed=1)
    rng = np.random.default_rng(2)
    w = _cn_noise(rng, n_kicks, m)
    b_mats = [expm(lag * DELTA * l_true) for lag in range(max_lag + 1)]
    resp = np.zeros((n_kicks, max_lag + 1, m), dtype=complex)
    for lag in range(1, max_lag + 1):
        resp[:, lag] = eps * (w @ b_mats[lag].T)
    zero_level = 1e-6
    resp[:, 0] = zero_level * _cn_noise(rng, n_kicks, m)

    pairs, diag = cross_propagators(
        [(w, resp)], [1, 3], DELTA, eps, demean=False
    )
    for (tau, m_mat), lag in zip(pairs, [1, 3], strict=True):
        assert np.isclose(tau, lag * DELTA)
        assert_allclose(m_mat, b_mats[lag], atol=1e-12)
    # The lag-0 regression sees only the planted noise (scaled by
    # 1/eps like every M).
    assert diag["causality"] < 10 * zero_level / eps
    assert diag["n_kicks"] == n_kicks

    l_hat, fit = identify_generator(pairs)
    assert_allclose(l_hat, l_true, atol=1e-10)
    assert max(fit["residuals"]) < 1e-10

    # Pooling two window sets stays exact.
    pairs2, _ = cross_propagators(
        [(w, resp), (w[: n_kicks // 2], resp[: n_kicks // 2])],
        [2],
        DELTA,
        eps,
        demean=False,
    )
    assert_allclose(pairs2[0][1], b_mats[2], atol=1e-12)


# ── Lyapunov helpers ─────────────────────────────────────────────────


def test_discrete_lyapunov() -> None:
    """Eig fallback == Bartels-Stewart; prediction == simulation."""
    from scipy.linalg import expm, solve_discrete_lyapunov

    m = 4
    a = _stable_generator(m, seed=3)
    e_mat = expm(0.3 * a)
    rng = np.random.default_rng(4)
    c = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    q = c @ c.conj().T
    assert_allclose(
        _discrete_lyapunov_eig(e_mat, q),
        solve_discrete_lyapunov(e_mat, q),
        atol=1e-10,
    )

    # predicted_forced_variance vs a long kicked simulation with
    # pre-kick sampling (no background).
    eps, dt_force, n = 0.05, 0.3, 200_000
    p, _ = np.linalg.qr(rng.standard_normal((m, 2)))
    e_f = expm(dt_force * a)
    x = np.zeros(m, dtype=complex)
    acc, rng2 = 0.0, np.random.default_rng(5)
    for _ in range(n):
        acc += float(np.sum(np.abs(p.conj().T @ x) ** 2))  # pre-kick
        x = e_f @ (x + eps * (p @ _cn_noise(rng2, 2)))
    var_pred = predicted_forced_variance(a, p, eps, dt_force)
    assert np.isclose(acc / n, var_pred, rtol=0.05), (acc / n, var_pred)

    # Unstable A is rejected.
    try:
        predicted_forced_variance(np.eye(2) * 0.1, np.eye(2), 1.0, 1.0)
    except ValueError as e:
        assert "stable" in str(e)
    else:
        raise AssertionError("unstable A was accepted")


# ── File pipeline on a real operator bundle ──────────────────────────


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    assert result.returncode == 0, (
        " ".join(str(c) for c in cmd)
        + "\n"
        + result.stdout[-3000:]
        + result.stderr[-3000:]
    )
    return result


def _operator_artifacts(tmp: Path) -> tuple[Path, Path]:
    """Real TG --save-operator + controllability bundles (mode (3,0))."""
    y = np.cos(np.pi * np.arange(NY) / (NY - 1))
    with open(tmp / "lam.txt", "w") as f:
        for yi, ui in zip(y, 1.0 - y**2, strict=True):
            f.write(f"{yi:+.17e} {ui:+.17e}\n")
    _run(
        [
            sys.executable,
            "-m",
            "dnsjax.analysis.transient_growth",
            "--profile",
            str(tmp / "lam.txt"),
            "--out-dir",
            str(tmp),
            "--modes",
            "3,0",
            "--nt",
            "9",
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
        cwd=tmp,
    )
    op_npz = tmp / "lam_tg_op.npz"
    cont_npz = tmp / "cont.npz"
    _run(
        [
            sys.executable,
            "-m",
            "dnsjax.analysis.response.operator_tools",
            "--operator",
            str(op_npz),
            "--n-modes",
            "4",
            "--out",
            str(cont_npz),
        ]
    )
    return op_npz, cont_npz


def _write_probe_stream(directory: Path, u: np.ndarray) -> None:
    """probes.bin/json for one synthetic run probing mode (3, 0)."""
    nt = u.shape[0]
    sidecar = {
        "format_version": 1,
        "modes": [[3, 0]],
        "wavenumbers": [[3, 0]],
        "n_components": 3,
        "component_labels": ["u_x", "u_y", "u_z"],
        "ny": NY,
        "wall_normal_grid": [
            float(v) for v in np.cos(np.pi * np.arange(NY) / (NY - 1))
        ],
        "value_dtype": "<f8",
        "it_probes": IT_PROBES,
        "dt": DT,
        "system": "plane-poiseuille",
        "double_precision": True,
        "git_hash": "synthetic",
        "params": {},
    }
    rec_dtype = np.dtype([("t", "<f8"), ("u", "<f8", (1, 3, NY, 2))])
    rec = np.zeros(nt, dtype=rec_dtype)
    rec["t"] = np.arange(nt) * DELTA
    rec["u"][..., 0] = u.real
    rec["u"][..., 1] = u.imag
    with open(directory / "probes.json", "w") as f:
        json.dump(sidecar, f)
    (directory / "probes.bin").write_bytes(rec.tobytes())


def _write_forcing_stream(
    directory: Path,
    t: np.ndarray,
    w: np.ndarray,
    eps: float,
    it_force: int,
    profiles: Path,
) -> None:
    """forcing.bin/json for one synthetic run forcing mode (3, 0)."""
    m = w.shape[1]
    sidecar = {
        "format_version": 1,
        "modes": [[3, 0]],
        "wavenumbers": [[3, 0]],
        "n_channels": m,
        "amplitude": eps,
        "it_force": it_force,
        "seed": 0,
        "dt": DT,
        "system": "plane-poiseuille",
        "profiles": str(profiles),
        "profiles_sha256": "synthetic",
        "git_hash": "synthetic",
        "params": {},
    }
    rec_dtype = np.dtype([("t", "<f8"), ("w", "<f8", (1, m, 2))])
    rec = np.zeros(len(t), dtype=rec_dtype)
    rec["t"] = t
    rec["w"][:, 0, :, 0] = w.real
    rec["w"][:, 0, :, 1] = w.imag
    with open(directory / "forcing.json", "w") as f:
        json.dump(sidecar, f)
    (directory / "forcing.bin").write_bytes(rec.tobytes())


def _simulate_forced_run(
    directory: Path,
    op: ot.OperatorData,
    p: np.ndarray,
    l_sim: np.ndarray,
    cont_npz: Path,
    n_kicks: int,
    eps: float,
    bg: float,
    seed: int,
) -> None:
    """Kicked linear dynamics + independent background, streamed out.

    Simulates dynamics **closed on the basis** (generator ``l_sim``;
    the identification's own model class -- the Galerkin gap of a
    real subspace is physics, measured by the residuals, not part of
    this plumbing test) on the probe grid with the runtime's pre-kick
    sampling convention (kick every ``c = 5`` samples), an
    independent white background in the basis coordinates, and writes
    the matching ``probes.*``/``forcing.*`` pair.
    """
    from scipy.linalg import expm

    c = 10
    nt = n_kicks * c + 1
    m = p.shape[1]
    e_mat = expm(DELTA * l_sim)
    rng = np.random.default_rng(seed)
    w = _cn_noise(rng, n_kicks, m)

    x = np.zeros(m, dtype=complex)
    b = np.empty((nt, m), dtype=complex)
    for n in range(nt):
        b[n] = x  # pre-kick sample
        if n % c == 0 and n // c < n_kicks:
            x = x + eps * w[n // c]
        x = e_mat @ x + bg * _cn_noise(rng, m)

    coords = b @ p.T  # a = P b: lift the basis coords
    u = (coords @ op.T_lift.T).reshape(nt, 1, 3, NY)
    directory.mkdir(exist_ok=True)
    _write_probe_stream(directory, u)
    t_kicks = np.arange(n_kicks) * c * DELTA
    _write_forcing_stream(directory, t_kicks, w, eps, c * IT_PROBES, cont_npz)


def test_identify_ssi_files() -> None:
    """Statistical recovery through the full file pipeline (+pooling,
    causality, sidecar-default basis, CLI).

    The simulated law is the restricted reference operator with extra
    damping: the laminar test operator is nearly neutral (spectral
    abscissa ~ -1e-3), so its kicked response is a barely-decaying
    random walk that no reasonable kick count can average down --
    real turbulent-mean operators are damped, and the estimator's own
    exactness is pinned above.  The identified L is compared against
    the simulated law; the reference growth curve of the undamped
    operator is checked structurally.
    """
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        op_npz, cont_npz = _operator_artifacts(tmp)
        op = ot.load_operator(op_npz, 3, 0)
        p = ot.recover_basis(op, ot.load_modes_npz(cont_npz, 3, 0))
        a_r = ot.restrict(op.A, p)
        l_sim = a_r - 0.5 * np.eye(p.shape[1])  # decay time ~ 2
        eps = 1.0

        for k, run in enumerate(("run_a", "run_b")):
            _simulate_forced_run(
                tmp / run,
                op,
                p,
                l_sim,
                cont_npz,
                1500,
                eps,
                bg=0.02,
                seed=10 + k,
            )

        # Lags comparable to the decay time: at short lags M is
        # near-identity and logm/tau amplifies the estimator noise
        # (the module docstring's short-lag warning, seen live).
        result = identify_ssi(
            [tmp / "run_a", tmp / "run_b"],
            3,
            0,
            op_npz,
            lags=[5 * DELTA, 10 * DELTA],
        )
        err = np.linalg.norm(result["L"] - l_sim) / np.linalg.norm(l_sim)
        assert err < 0.10, err
        assert result["causality"] < 0.1, result["causality"]
        # nt = 10 * n_kicks + 1, so even the last kick's 10-lag
        # window fits: nothing is clipped.
        assert result["n_kicks"] == 2 * 1500
        # Positive, finite variance report (the prediction uses the
        # full exported A; its exactness is unit-tested above).
        assert result["var_measured"] > 0
        assert result["var_forced_predicted"] > 0

        out = tmp / "ssi.npz"
        _run(
            [
                sys.executable,
                "-m",
                "dnsjax.analysis.response.ssi",
                "--runs",
                str(tmp / "run_a"),
                str(tmp / "run_b"),
                "--mode",
                "3,0",
                "--operator",
                str(op_npz),
                "--lags",
                f"{5 * DELTA},{10 * DELTA}",
                "--out",
                str(out),
            ]
        )
        with np.load(out) as z:
            # G_id must track the growth curve of the simulated law;
            # G_ref (the undamped reference restriction) is checked
            # structurally.
            g_sim = ot.growth_curve(l_sim, z["t_grid"])
            rel = np.max(np.abs(z["G_id"] - g_sim) / g_sim)
            assert rel < 0.3, rel
            assert z["G_ref"].shape == z["G_id"].shape
            assert np.isfinite(z["G_ref"]).all()
            assert bool(z["stable"])


def test_read_forcing_errors() -> None:
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        try:
            read_forcing(tmp)
        except FileNotFoundError as e:
            assert "sidecar" in str(e)
        else:
            raise AssertionError("missing sidecar was accepted")

        rng = np.random.default_rng(6)
        _write_forcing_stream(
            tmp,
            np.arange(3) * DELTA,
            _cn_noise(rng, 3, 2),
            0.1,
            IT_PROBES,
            tmp / "none.npz",
        )
        # Truncate the last record: it is dropped with a note.
        raw = (tmp / "forcing.bin").read_bytes()
        (tmp / "forcing.bin").write_bytes(raw[:-5])
        data = read_forcing(tmp)
        assert data.w.shape == (2, 1, 2)


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
