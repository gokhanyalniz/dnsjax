r"""Ensemble machinery tests (``ensemble_setup.py`` + ``response.ensemble``).

Covers the orchestration + aggregation + identification chain without
solver runs (the probe streams of the "members" are synthesised; the
solver-side probe integration is ``tests/test_probes.py``, and the
full chain is a documented manual rehearsal):

1. **harvest + build (real artifacts)**: two real mini snapshots ->
   ``harvest`` manifest (spacing/t-min honoured) -> ``build``
   ``--dry-run`` (prints the plan, writes nothing) -> real ``build``
   with an ``--npy`` source and antithetic pairing: member dirs with
   ``seed.tar`` (via the real ``snapshot_perturb`` subprocesses, the
   antithetic pair seeds exactly mirrored about the parent),
   generated ``parameters.toml``, ``run_commands.txt``, and
   ``members.json``.
2. **Antithetic aggregation is exact**: synthetic member probe
   streams ``u_pm = base_k(t) +- eps resp(t)`` with a large random
   per-pair background -> ``aggregate_tree`` recovers ``eps resp``
   to roundoff (the background and even orders cancel identically).
3. **identify_generator units**: exact multi-horizon recovery of a
   known generator; branch-cut / singular / shape rejections.
4. **Direct identification end-to-end vs a known operator**: a real
   transient-growth ``--save-operator`` bundle + controllability
   basis (real writers, subprocess); synthetic basis responses
   generated from the *restriction* of the exported operator, plus
   noise -> ``identify_from_responses`` recovers it (noise-free to
   1e-8, noisy to 5%), and the ``identify`` CLI writes matching
   ``G_id``/``G_ref`` curves.

Usage::

    uv run python tests/response/test_ensemble.py
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
    aggregate_tree,
    identify_from_responses,
    identify_generator,
)
from dnsjax.snapshot import (  # noqa: E402
    assemble_local_shards,
    load_snapshot,
    save_snapshot,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_SETUP = _REPO / "scripts" / "ensemble_setup.py"
IT_PROBES = 10
DT = float(params.step.dt)
NT = 6  # probe samples per member (t_rel = 0 .. 5*IT_PROBES*DT)


def _make_parent(path: Path, t: float, it: int, seed: int) -> None:
    rng = np.random.default_rng(seed)

    def fill_local(buf, kz_start, nkz, kx_start, nkx):
        shape = (3, NY, nkz, nkx)
        buf[:, :, :nkz, :nkx] = rng.standard_normal(
            shape
        ) + 1j * rng.standard_normal(shape)

    save_snapshot(assemble_local_shards(fill_local), t, it, path, isnap=0)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    assert result.returncode == 0, (
        " ".join(str(c) for c in cmd)
        + "\n"
        + result.stdout[-3000:]
        + result.stderr[-3000:]
    )
    return result


def _build_tree(tmp: Path) -> Path:
    """Real harvest + build (antithetic, --npy source) -> tree path."""
    parents = tmp / "parents"
    parents.mkdir()
    _make_parent(parents / "state00007.tar", t=10.0, it=1000, seed=1)
    _make_parent(parents / "state00008.tar", t=12.0, it=1200, seed=2)
    _make_parent(parents / "state00009.tar", t=15.0, it=1500, seed=3)

    manifest = tmp / "manifest.json"
    _run(
        [
            sys.executable,
            str(_SETUP),
            "harvest",
            "--run-dir",
            str(parents),
            "--t-min",
            "9.0",
            "--spacing",
            "4.0",
            "--n",
            "2",
            "--out",
            str(manifest),
        ]
    )
    with open(manifest) as f:
        picked = json.load(f)["snapshots"]
    # Spacing 4 from t=10 skips t=12 and takes t=15.
    assert [s["t"] for s in picked] == [10.0, 15.0]

    rng = np.random.default_rng(4)
    vec = rng.standard_normal((3, NY)) + 1j * rng.standard_normal((3, NY))
    np.save(tmp / "vec.npy", vec)

    tree = tmp / "members"
    build_args = [
        sys.executable,
        str(_SETUP),
        "build",
        "--manifest",
        str(manifest),
        "--tree",
        str(tree),
        "--mode",
        "3,0",
        "--npy",
        str(tmp / "vec.npy"),
        "--amplitude-energy",
        "1e-6",
        "--pairing",
        "antithetic",
        "--horizon",
        str(NT * IT_PROBES * DT),  # note: NT intervals
        "--probe-modes",
        "3,0",
        "--it-probes",
        str(IT_PROBES),
    ]
    dry = _run([*build_args, "--dry-run"])
    assert not tree.exists()  # dry run writes nothing
    assert "m0000_p" in dry.stdout and "m0001_m" in dry.stdout
    assert "run-cmd" in dry.stdout and "seed-cmd" in dry.stdout

    _run(build_args)
    return tree


def test_harvest_build_and_aggregate() -> None:
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        tree = _build_tree(tmp)

        with open(tree / "members.json") as f:
            spec = json.load(f)
        assert spec["mode"] == [3, 0]
        assert [m["dir"] for m in spec["members"]] == [
            "m0000_p",
            "m0000_m",
            "m0001_p",
            "m0001_m",
        ]
        assert (tree / "run_commands.txt").read_text().count("mpirun") == 4

        # The antithetic seeds are exact mirrors about the parent.
        for k, parent_t in ((0, 10.0), (1, 15.0)):
            parent = np.asarray(
                load_snapshot(spec["members"][2 * k]["parent"])[0]
            )
            plus, t_p, _ = load_snapshot(tree / f"m{k:04d}_p" / "seed.tar")
            minus, t_m, _ = load_snapshot(tree / f"m{k:04d}_m" / "seed.tar")
            assert t_p == t_m == parent_t
            assert_allclose(
                np.asarray(plus) - parent,
                -(np.asarray(minus) - parent),
                atol=0,
            )
            toml = (tree / f"m{k:04d}_p" / "parameters.toml").read_text()
            assert 'probe_modes = "3,0"' in toml
            assert f"max_sim_time = {parent_t + NT * IT_PROBES * DT!r}" in toml

        # Synthetic member probe streams: huge per-pair background,
        # shared response -> antithetic aggregation is exact.
        rng = np.random.default_rng(5)
        resp = rng.standard_normal((NT, 1, 3, NY)) + 1j * rng.standard_normal(
            (NT, 1, 3, NY)
        )
        eps = 1e-3
        for k, parent_t in ((0, 10.0), (1, 15.0)):
            base = 10.0 * (
                rng.standard_normal((NT, 1, 3, NY))
                + 1j * rng.standard_normal((NT, 1, 3, NY))
            )
            for tag, sign in (("p", +1), ("m", -1)):
                _write_probe_stream(
                    tree / f"m{k:04d}_{tag}",
                    parent_t,
                    base + sign * eps * resp,
                )

        out = tmp / "response.npz"
        result = aggregate_tree(tree, out)
        assert_allclose(result["mean_u"], eps * resp, atol=1e-14)
        with np.load(out) as z:
            assert int(z["injected_index"]) == 0
            assert int(z["basis_index"]) == -1  # npy source, no basis
            assert_allclose(
                z["t_rel"], np.arange(NT) * IT_PROBES * DT, atol=1e-12
            )


def _write_probe_stream(directory: Path, t0: float, u: np.ndarray) -> None:
    """Hand-written probes.bin/json for one member (mode (3,0))."""
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
    rec["t"] = t0 + np.arange(nt) * IT_PROBES * DT
    rec["u"][..., 0] = u.real
    rec["u"][..., 1] = u.imag
    with open(directory / "probes.json", "w") as f:
        json.dump(sidecar, f)
    (directory / "probes.bin").write_bytes(rec.tobytes())


def test_identify_generator_units() -> None:
    rng = np.random.default_rng(6)
    m = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    l_true = m - (np.max(np.linalg.eigvals(m).real) + 0.5) * np.eye(4)
    from scipy.linalg import expm

    pairs = [(tau, expm(tau * l_true)) for tau in (0.3, 0.7, 1.1)]
    l_hat, diag = identify_generator(pairs)
    assert_allclose(l_hat, l_true, atol=1e-10)
    assert max(diag["residuals"]) < 1e-10

    # Branch-cut rejection: a negative real eigenvalue.
    try:
        identify_generator([(1.0, np.diag([-1.0, 0.5]))])
    except ValueError as e:
        assert "negative real axis" in str(e)
    else:
        raise AssertionError("branch-cut M was accepted")

    # Singular rejection.
    try:
        identify_generator([(1.0, np.diag([0.0, 0.5]))])
    except ValueError as e:
        assert "singular" in str(e)
    else:
        raise AssertionError("singular M was accepted")

    # Shape mismatch.
    try:
        identify_generator([(1.0, np.eye(2)), (2.0, np.eye(3))])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched shapes were accepted")


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


def _synthetic_responses(
    tmp: Path,
    op: ot.OperatorData,
    p: np.ndarray,
    l_true: np.ndarray,
    t_rel: np.ndarray,
    noise: float,
    seed: int,
) -> list[Path]:
    """Response npzs mimicking ``aggregate_tree`` outputs: the basis
    responses of the dynamics ``l_true`` on the basis ``p``."""
    from scipy.linalg import expm

    rng = np.random.default_rng(seed)
    m = p.shape[1]
    files = []
    for j in range(m):
        scale = 0.01 * (1.0 + j)
        coords = np.stack(
            [p @ (expm(t * l_true) @ (scale * np.eye(m)[:, j])) for t in t_rel]
        )  # (nt, r_res)
        u = (coords @ op.T_lift.T).reshape(len(t_rel), 1, 3, NY)
        noise_arr = (
            noise
            * scale
            * (
                rng.standard_normal(u.shape)
                + 1j * rng.standard_normal(u.shape)
            )
        )
        # The t = 0 sample carries no ensemble noise in reality: the
        # pair-combined response at t = 0 is the injected profile
        # itself (members share the parent state exactly there).
        noise_arr[0] = 0.0
        u = u + noise_arr
        path = tmp / f"resp{j}.npz"
        np.savez(
            path,
            t_rel=t_rel,
            mean_u=u,
            modes=np.asarray([[3, 0]]),
            injected_i2=3,
            injected_i3=0,
            injected_index=0,
            basis_index=j,
        )
        files.append(path)
    return files


def test_direct_identification() -> None:
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        op_npz, cont_npz = _operator_artifacts(tmp)
        op = ot.load_operator(op_npz, 3, 0)
        with np.load(cont_npz) as z:
            lifted = np.asarray(z["cont_modes_3_0"])
        p = (lifted.reshape(4, -1) @ op.T_proj.T).T  # (r_res, 4)
        l_true = ot.restrict(op.A, p)
        t_rel = np.linspace(0.0, 2.0, 9)

        # Noise-free: exact recovery.
        files = _synthetic_responses(
            tmp, op, p, l_true, t_rel, noise=0.0, seed=7
        )
        result = identify_from_responses(
            files, op_npz, cont_npz, horizons=[0.5, 1.0, 2.0]
        )
        err = np.linalg.norm(result["L"] - l_true) / np.linalg.norm(l_true)
        assert err < 1e-8, err
        assert max(result["residuals"]) < 1e-8

        # Noisy: 5% recovery.
        files = _synthetic_responses(
            tmp, op, p, l_true, t_rel, noise=1e-4, seed=8
        )
        result = identify_from_responses(
            files, op_npz, cont_npz, horizons=[0.5, 1.0, 2.0]
        )
        err = np.linalg.norm(result["L"] - l_true) / np.linalg.norm(l_true)
        assert err < 0.05, err

        # The identify CLI: G_id tracks G_ref for the noise-free set.
        files = _synthetic_responses(
            tmp, op, p, l_true, t_rel, noise=0.0, seed=9
        )
        out = tmp / "identified.npz"
        _run(
            [
                sys.executable,
                "-m",
                "dnsjax.analysis.response.ensemble",
                "identify",
                "--responses",
                *[str(f) for f in files],
                "--operator",
                str(op_npz),
                "--modes-npz",
                str(cont_npz),
                "--horizons",
                "0.5,1.0,2.0",
                "--out",
                str(out),
            ]
        )
        with np.load(out) as z:
            assert_allclose(z["G_id"], z["G_ref"], rtol=1e-6)
            assert bool(z["stable"])


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
