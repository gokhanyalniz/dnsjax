r"""LIM identification tests (``dnsjax.analysis.response.lim``).

All offline (no solver runs; the probe streams are synthesised, the
operator bundle comes from a real transient-growth ``--save-operator``
subprocess as in ``test_ensemble.py``):

1. **Estimator exactness**: for noiseless linear data
   ``b_k = e^{k dt L} b_0`` the lag-consistent overlap estimator
   returns ``M(tau) = e^{tau L}`` to roundoff (single segment and
   pooled segments), and ``identify_generator`` recovers ``L``.
   Ill-excited coordinates raise the conditioning error.
2. **Statistical recovery**: a discrete Ornstein-Uhlenbeck process
   with the exact per-step noise covariance ``Q = X - E X E^H``
   (``X`` the continuous Lyapunov solution, so the process's
   stationary covariance is ``X`` and ``C(l) = E^l X`` exactly in
   expectation) -- LIM recovers ``L`` to a few %, and the sample
   ``C(0)`` matches the controllability Gramian (the WP3 Lyapunov
   identity).
3. **File pipeline + CLI**: synthetic ``probes.bin`` streams built by
   lifting basis coordinates through a real operator bundle;
   ``identify_lim`` recovers the restricted reference generator
   exactly (noiseless, ``demean=False``), and the CLI writes
   ``G_id ~= G_ref``.
4. **projected_fluctuations units**: ``t_min`` cut, demeaning, and
   the non-uniform-sample rejection.

Usage::

    uv run python tests/response/test_lim.py
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(line_buffering=True)

import jax  # noqa: E402

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
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from _live import run_live  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.analysis.response import operator_tools as ot  # noqa: E402
from dnsjax.analysis.response.ensemble import (  # noqa: E402
    identify_generator,
)
from dnsjax.analysis.response.lim import (  # noqa: E402
    identify_lim,
    lagged_propagators,
    projected_fluctuations,
)
from dnsjax.analysis.response.probes import ProbeData  # noqa: E402

IT_PROBES = 10
DT = float(params.step.dt)
DELTA = IT_PROBES * DT


def _stable_generator(m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    return a - (np.max(np.linalg.eigvals(a).real) + 1.0) * np.eye(m)


def _trajectory(l_mat: np.ndarray, b0: np.ndarray, nt: int) -> np.ndarray:
    from scipy.linalg import expm

    e_mat = expm(DELTA * l_mat)
    out = np.empty((nt, len(b0)), dtype=complex)
    out[0] = b0
    for k in range(1, nt):
        out[k] = e_mat @ out[k - 1]
    return out


def _ou_series(l_mat: np.ndarray, nt: int, seed: int) -> np.ndarray:
    """Exactly-stationary discrete OU samples of ``db = L b + xi``.

    Per-step noise covariance ``Q = X - E X E^H`` (``X`` the
    continuous unit-forcing Lyapunov solution), so the stationary
    covariance is ``X`` and ``C(l) = E^l X`` in expectation."""
    from scipy.linalg import expm

    m = l_mat.shape[0]
    e_mat = expm(DELTA * l_mat)
    x_cov = ot.controllability_gramian(l_mat)
    q = x_cov - e_mat @ x_cov @ e_mat.conj().T
    vals, vecs = np.linalg.eigh(0.5 * (q + q.conj().T))
    q_sqrt = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))

    rng = np.random.default_rng(seed)
    noise = (
        rng.standard_normal((nt, m)) + 1j * rng.standard_normal((nt, m))
    ) / np.sqrt(2.0)
    b = np.empty((nt, m), dtype=complex)
    b[0] = q_sqrt @ noise[0]
    for k in range(1, nt):
        b[k] = e_mat @ b[k - 1] + q_sqrt @ noise[k]
    return b


# ── Estimator exactness ──────────────────────────────────────────────


def test_noiseless_exactness() -> None:
    """Noiseless linear data: M(tau) exact, L recovered, pooled too."""
    from scipy.linalg import expm

    m = 4
    l_true = _stable_generator(m, seed=1)
    rng = np.random.default_rng(2)
    seg1 = _trajectory(
        l_true, rng.standard_normal(m) + 1j * rng.standard_normal(m), 40
    )
    pairs, diag = lagged_propagators([seg1], [1, 3], DELTA)
    for (tau, m_mat), lag in zip(pairs, [1, 3], strict=True):
        assert np.isclose(tau, lag * DELTA)
        assert_allclose(m_mat, expm(tau * l_true), atol=1e-9)
    assert diag["n_samples"] == [39, 37]

    l_hat, _ = identify_generator(pairs)
    assert_allclose(l_hat, l_true, atol=1e-8)

    # Pooling two segments stays exact.
    seg2 = _trajectory(
        l_true, rng.standard_normal(m) + 1j * rng.standard_normal(m), 25
    )
    pairs2, _ = lagged_propagators([seg1, seg2], [2], DELTA)
    assert_allclose(pairs2[0][1], expm(2 * DELTA * l_true), atol=1e-9)


def test_conditioning_rejection() -> None:
    """A coordinate the data never excites raises the C(0) error."""
    m = 3
    l_true = _stable_generator(m, seed=3)
    seg = _trajectory(l_true, np.array([1.0, 0.5, 0.2]) + 0j, 30)
    seg[:, 2] = 0.0  # kill one coordinate
    try:
        lagged_propagators([seg], [1], DELTA)
    except ValueError as e:
        assert "condition" in str(e)
    else:
        raise AssertionError("rank-deficient C(0) was accepted")


# ── Statistical recovery (Ornstein-Uhlenbeck) ────────────────────────


def test_ou_recovery() -> None:
    """LIM on an exactly-stationary OU process: L to a few %, and the
    sample covariance matches the controllability Gramian."""
    m = 4
    l_true = _stable_generator(m, seed=4)
    b = _ou_series(l_true, nt=100_000, seed=5)

    pairs, _ = lagged_propagators([b], [1, 2, 4], DELTA)
    l_hat, _ = identify_generator(pairs)
    err = np.linalg.norm(l_hat - l_true) / np.linalg.norm(l_true)
    assert err < 0.05, err

    # Stationary sample covariance vs the Lyapunov solution.
    x_cov = ot.controllability_gramian(l_true)
    c0 = (b.T @ np.conj(b)) / b.shape[0]
    cov_err = np.linalg.norm(c0 - x_cov) / np.linalg.norm(x_cov)
    assert cov_err < 0.05, cov_err


# ── projected_fluctuations units ─────────────────────────────────────


def _probe_data(t: np.ndarray, u: np.ndarray) -> ProbeData:
    """Minimal in-memory ProbeData for one probed mode (3, 0)."""
    return ProbeData(
        t=t,
        u=u,
        modes=np.asarray([[3, 0]]),
        wavenumbers=np.asarray([[3, 0]]),
        y=np.linspace(1.0, -1.0, NY),
        component_labels=["u_x", "u_y", "u_z"],
        meta={"it_probes": IT_PROBES, "dt": DT},
    )


def test_projected_fluctuations_units() -> None:
    n = 3 * NY
    t_proj = np.eye(4, n, dtype=complex)  # first 4 flat entries
    rng = np.random.default_rng(6)
    u = rng.standard_normal((10, 1, 3, NY)) + 1j * rng.standard_normal(
        (10, 1, 3, NY)
    )
    t = np.arange(10) * DELTA

    t_out, b = projected_fluctuations(_probe_data(t, u), t_proj, 3, 0)
    assert b.shape == (10, 4)
    assert_allclose(b.mean(axis=0), 0.0, atol=1e-14)  # demeaned
    flat = u[:, 0].reshape(10, -1)
    assert_allclose(b, flat[:, :4] - flat[:, :4].mean(axis=0), atol=1e-14)

    # t_min cut.
    t_out, b = projected_fluctuations(
        _probe_data(t, u), t_proj, 3, 0, t_min=4.5 * DELTA
    )
    assert len(t_out) == 5 and t_out[0] >= 4.5 * DELTA

    # Non-uniform samples (an overlapping resume) are rejected.
    t_bad = t.copy()
    t_bad[7:] += DELTA  # a gap
    try:
        projected_fluctuations(_probe_data(t_bad, u), t_proj, 3, 0)
    except ValueError as e:
        assert "uniform" in str(e)
    else:
        raise AssertionError("non-uniform samples were accepted")


# ── File pipeline + CLI on a real operator bundle ────────────────────


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    result = run_live(cmd, **kw)
    assert result.returncode == 0, (
        " ".join(str(c) for c in cmd)
        + "\n"
        + result.stdout[-3000:]
        + result.stderr[-3000:]
    )
    return result


def _operator_artifacts(tmp: Path) -> tuple[Path, Path]:
    """Real TG operator export + controllability bundle (mode (3,0))."""
    y = np.cos(np.pi * np.arange(NY) / (NY - 1))
    with open(tmp / "lam.txt", "w") as f:
        for yi, ui in zip(y, 1.0 - y**2, strict=True):
            f.write(f"{yi:+.17e} {ui:+.17e}\n")
    _run(
        [
            sys.executable,
            "-m",
            "dnsjax.analysis.transient_growth",
            "--tg.profile",
            str(tmp / "lam.txt"),
            "--tg.out_dir",
            str(tmp),
            "--tg.modes",
            "3,0",
            "--tg.nt",
            "9",
            "--tg.save_operator",
            "True",
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


def _lifted_stream(
    directory: Path, op: ot.OperatorData, p: np.ndarray, b: np.ndarray
) -> None:
    """Write a probes.bin whose projection reproduces ``b`` exactly:
    ``u = T_lift (P b)`` and ``P^H T_proj u = b``."""
    coords = b @ p.T  # (nt, r_res): a_k = P b_k
    u = (coords @ op.T_lift.T).reshape(b.shape[0], 1, 3, NY)
    directory.mkdir(exist_ok=True)
    _write_probe_stream(directory, u)


def test_identify_lim_files_and_cli() -> None:
    """identify_lim on lifted synthetic streams recovers the restricted
    reference generator (exactly for noiseless data; statistically
    through the demeaning CLI on OU data), with G_id tracking G_ref."""
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        op_npz, cont_npz = _operator_artifacts(tmp)
        op = ot.load_operator(op_npz, 3, 0)
        p = ot.recover_basis(op, ot.load_modes_npz(cont_npz, 3, 0))
        a_r = ot.restrict(op.A, p)

        # Noiseless data through the full file pipeline: exact.  The
        # synthesised law is a mild synthetic generator, not the
        # physical a_r: a single decaying trajectory of the real
        # operator loses its fast controllability directions within
        # one probe interval and trips the conditioning guard -- LIM
        # genuinely needs persistent excitation (the OU stream below
        # provides it for the physical dynamics).
        rng = np.random.default_rng(7)
        l_syn = _stable_generator(4, seed=17)
        run_det = tmp / "run_det"
        _lifted_stream(
            run_det,
            op,
            p,
            _trajectory(
                l_syn,
                rng.standard_normal(4) + 1j * rng.standard_normal(4),
                48,
            ),
        )
        result = identify_lim(
            [run_det],
            3,
            0,
            op_npz,
            lags=[2 * DELTA, 5 * DELTA],
            modes_npz=cont_npz,
            demean=False,
        )
        err = np.linalg.norm(result["L"] - l_syn) / np.linalg.norm(l_syn)
        assert err < 1e-6, err
        assert max(result["residuals"]) < 1e-8

        # Stationary OU stream through the CLI (default demeaning).
        run_ou = tmp / "run_ou"
        _lifted_stream(run_ou, op, p, _ou_series(a_r, nt=20_000, seed=8))
        out = tmp / "lim.npz"
        _run(
            [
                sys.executable,
                "-m",
                "dnsjax.analysis.response.lim",
                "--probes",
                str(run_ou),
                "--mode",
                "3,0",
                "--operator",
                str(op_npz),
                "--modes-npz",
                str(cont_npz),
                "--lags",
                f"{2 * DELTA},{5 * DELTA}",
                "--out",
                str(out),
            ]
        )
        with np.load(out) as z:
            assert z["G_id"].shape == z["G_ref"].shape == z["t_grid"].shape
            assert bool(z["stable"])
            rel = np.max(np.abs(z["G_id"] - z["G_ref"]) / z["G_ref"])
            assert rel < 0.25, rel


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
