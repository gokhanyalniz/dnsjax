r"""``scripts/snapshot_perturb.py`` injection tests (offline, np = 1).

Builds small plane-Poiseuille snapshots in-process (synthetic spectral
states via ``snapshot.assemble_local_shards`` + ``save_snapshot``),
runs the injection CLI as a subprocess in a temporary directory, and
verifies against the reloaded arrays:

1. **Exactness**: a ``--perturb.npy`` injection at ``(i2, i3) =
   (3, 0)`` adds exactly ``scale * vec`` at the target column and
   ``scale * conj(vec)`` at the real-FFT conjugate partner
   ``((nz-1)-3, 0)``, leaves every other entry ``==``-identical, and
   preserves the parent's ``(t, it)``.
2. **Energy convention**: ``--perturb.amplitude_energy E0`` onto a
   zero snapshot yields a state whose solver-measure ``E'``
   (``get_perturbation_energy``) is ``E0`` to roundoff.
3. **Antithesis**: ``--perturb.negate`` gives the exact mirror
   perturbation (the ``+``/``-`` deltas cancel identically).
4. **Transient-growth source**: a real (subprocess) TG run on the
   laminar profile produces ``<stem>_tg.npz``; ``--perturb.tg_npz``
   injects a field collinear with its ``opt_input`` row at the
   requested energy -- pinning the npz key contract against the
   actual writer.  Its ``--tg.save_operator`` bundle keeps the
   coordinate contract it documents, and full-state profiles built
   from it inject through ``--perturb.modes_npz``.
5. **Error paths**: out-of-range mode, complex ``(0,0)`` profile,
   mode missing from the TG bundle, wrong-system bundle.
6. **Mean mode** ``(0,0)``: a profile that satisfies the conservation
   laws of :mod:`dnsjax.ic.mean_mode` injects (even under
   ``constant_bulk_velocity``) and writes that one self-conjugate
   column; each law it can break refuses, naming the law, judged on
   the injected profile rather than the resulting state.

Usage::

    uv run python tests/test_snapshot_perturb.py
"""

from __future__ import annotations

from _snapshot_perturb import (
    N2,
    NY,
    configure,
    make_snapshot,
    run_perturb,
    tg_bundle,
)

configure()

import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

import dnsjax.flows.wall_bounded.plane_poiseuille as fmod  # noqa: E402
from dnsjax.snapshot import load_snapshot  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def test_npy_injection_exact() -> None:
    rng = np.random.default_rng(7)
    vec = rng.standard_normal((3, NY)) + 1j * rng.standard_normal((3, NY))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = make_snapshot(tmp / "base.tar")
        np.save(tmp / "vec.npy", vec)
        run_perturb(
            [
                "--init.snapshot",
                str(tmp / "base.tar"),
                "--perturb.out",
                str(tmp / "out.tar"),
                "--perturb.mode",
                "3,0",
                "--perturb.npy",
                str(tmp / "vec.npy"),
                "--perturb.amplitude_scale",
                "0.5",
            ]
        )
        state, t, it = load_snapshot(tmp / "out.tar")
        assert (t, it) == (1.25, 50)
        new = np.asarray(state)
        expected = base.copy()
        expected[:, :, 3, 0] += 0.5 * vec
        expected[:, :, N2 - 3, 0] += 0.5 * np.conj(vec)
        assert_array_equal(new, expected)


def test_amplitude_energy_convention() -> None:
    rng = np.random.default_rng(8)
    vec = rng.standard_normal((3, NY)) + 1j * rng.standard_normal((3, NY))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_snapshot(tmp / "zero.tar", zero=True)
        np.save(tmp / "vec.npy", vec)
        run_perturb(
            [
                "--init.snapshot",
                str(tmp / "zero.tar"),
                "--perturb.out",
                str(tmp / "out.tar"),
                "--perturb.mode",
                "3,0",
                "--perturb.npy",
                str(tmp / "vec.npy"),
                "--perturb.amplitude_energy",
                "1e-4",
            ]
        )
        state, _, _ = load_snapshot(tmp / "out.tar")
        e_prime = float(fmod.get_perturbation_energy(state))
        assert_allclose(e_prime, 1e-4, rtol=1e-12)


def test_negate_antithetic() -> None:
    rng = np.random.default_rng(9)
    vec = rng.standard_normal((3, NY)) + 1j * rng.standard_normal((3, NY))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = make_snapshot(tmp / "base.tar")
        np.save(tmp / "vec.npy", vec)
        common = [
            "--init.snapshot",
            str(tmp / "base.tar"),
            "--perturb.mode",
            "3,0",
            "--perturb.npy",
            str(tmp / "vec.npy"),
            "--perturb.amplitude_energy",
            "1e-6",
        ]
        run_perturb([*common, "--perturb.out", str(tmp / "plus.tar")])
        run_perturb(
            [
                *common,
                "--perturb.out",
                str(tmp / "minus.tar"),
                "--perturb.negate",
                "True",
            ]
        )
        plus = np.asarray(load_snapshot(tmp / "plus.tar")[0])
        minus = np.asarray(load_snapshot(tmp / "minus.tar")[0])
        assert_array_equal(plus - base, -(minus - base))
        assert np.abs(plus - base).max() > 0


def _check_operator_bundle(path: Path) -> np.ndarray:
    r"""The export's coordinate contract; its basis as full states.

    ``q = V F^{-1} Q a`` and ``a = Q^H F V^H q`` invert each other,
    ``||a||^2 = q^H diag(energy_weights) q``, and ``lam = eig(A)``
    (the bundle's ``readme``).  Returns ``V F^{-1} Q`` column by column
    as ``(r_res, 3, NY)`` profiles.
    """
    with np.load(path) as z:
        A, Q, F, V = (np.asarray(z[f"{k}_3_0"]) for k in "AQFV")
        lam = np.asarray(z["lam_3_0"])
        w = np.asarray(z["energy_weights"])
    r_res = A.shape[0]
    assert Q.shape[1] == r_res and V.shape == (3 * NY, F.shape[0])
    assert_allclose(Q.conj().T @ Q, np.eye(r_res), atol=1e-12)
    assert_allclose(V.conj().T @ V, np.eye(V.shape[1]), atol=1e-12)
    assert_allclose(np.triu(F), F, atol=0.0)
    metric = V.conj().T @ (w[:, None] * V)
    assert_allclose(F.conj().T @ F, metric, atol=1e-12 * np.abs(metric).max())
    basis = V @ np.linalg.solve(F, Q)  # columns: q of each unit a
    rng = np.random.default_rng(3)
    a = rng.standard_normal(r_res) + 1j * rng.standard_normal(r_res)
    q = basis @ a
    assert_allclose(Q.conj().T @ (F @ (V.conj().T @ q)), a, atol=1e-10)
    assert_allclose(np.vdot(q, w * q).real, np.vdot(a, a).real, rtol=1e-10)
    got = np.sort_complex(np.linalg.eigvals(A))
    assert_allclose(got, np.sort_complex(lam), atol=1e-9 * np.abs(lam).max())
    return basis.T.reshape(r_res, 3, NY)


def test_tg_npz_source() -> None:
    """End-to-end against the real transient-growth npz writers."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tg_bundle(tmp)
        npz_path = tmp / "lam_tg.npz"
        with np.load(npz_path) as npz:
            opt_input = np.asarray(npz["opt_input"][0])

        make_snapshot(tmp / "zero.tar", zero=True)
        run_perturb(
            [
                "--init.snapshot",
                str(tmp / "zero.tar"),
                "--perturb.out",
                str(tmp / "out.tar"),
                "--perturb.mode",
                "3,0",
                "--perturb.tg_npz",
                str(npz_path),
                "--perturb.which",
                "input",
                "--perturb.amplitude_energy",
                "1e-4",
            ]
        )
        state, _, _ = load_snapshot(tmp / "out.tar")
        new = np.asarray(state)
        col = new[:, :, 3, 0]
        # Collinear with the optimal input (one global positive scale).
        scale = np.abs(col).max() / np.abs(opt_input).max()
        assert scale > 0
        assert_allclose(col, scale * opt_input, atol=1e-12 * scale)
        assert_allclose(new[:, :, N2 - 3, 0], np.conj(col), atol=1e-15 * scale)
        e_prime = float(fmod.get_perturbation_energy(state))
        assert_allclose(e_prime, 1e-4, rtol=1e-12)

        # Error: a mode the bundle does not contain.
        run_perturb(
            [
                "--init.snapshot",
                str(tmp / "zero.tar"),
                "--perturb.out",
                str(tmp / "x.tar"),
                "--perturb.mode",
                "2,0",
                "--perturb.tg_npz",
                str(npz_path),
                "--perturb.amplitude_energy",
                "1e-4",
            ],
            expect_fail="is not in",
        )

        # A modes bundle (--perturb.modes_npz) built from the operator
        # export's own coordinate directions, q = V F^-1 Q e_j: unit
        # energy, solenoidal full-state profiles.
        profiles = _check_operator_bundle(tmp / "lam_tg_op.npz")
        np.savez(
            tmp / "cont.npz",
            system="plane-poiseuille",
            code_grid=np.load(npz_path)["code_grid"],
            profiles_3_0=profiles[:3],
        )
        run_perturb(
            [
                "--init.snapshot",
                str(tmp / "zero.tar"),
                "--perturb.out",
                str(tmp / "cm.tar"),
                "--perturb.mode",
                "3,0",
                "--perturb.modes_npz",
                str(tmp / "cont.npz"),
                "--perturb.index",
                "1",
                "--perturb.amplitude_energy",
                "1e-4",
            ]
        )
        with np.load(tmp / "cont.npz") as z:
            cont_vec = np.asarray(z["profiles_3_0"][1])
        state, _, _ = load_snapshot(tmp / "cm.tar")
        new = np.asarray(state)
        col = new[:, :, 3, 0]
        scale = np.abs(col).max() / np.abs(cont_vec).max()
        assert_allclose(col, scale * cont_vec, atol=1e-12 * scale)
        e_prime = float(fmod.get_perturbation_energy(state))
        assert_allclose(e_prime, 1e-4, rtol=1e-12)

        # --index out of range against the real bundle.
        run_perturb(
            [
                "--init.snapshot",
                str(tmp / "zero.tar"),
                "--perturb.out",
                str(tmp / "x.tar"),
                "--perturb.mode",
                "3,0",
                "--perturb.modes_npz",
                str(tmp / "cont.npz"),
                "--perturb.index",
                "3",
                "--perturb.amplitude_energy",
                "1e-4",
            ],
            expect_fail="out of range",
        )


def test_error_paths() -> None:
    rng = np.random.default_rng(10)
    vec = rng.standard_normal((3, NY)) + 1j * rng.standard_normal((3, NY))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_snapshot(tmp / "base.tar")
        np.save(tmp / "vec.npy", vec)
        common = [
            "--init.snapshot",
            str(tmp / "base.tar"),
            "--perturb.out",
            str(tmp / "x.tar"),
            "--perturb.npy",
            str(tmp / "vec.npy"),
            "--perturb.amplitude_scale",
            "1.0",
        ]
        run_perturb(
            [*common, "--perturb.mode", f"{N2},0"], expect_fail="out of range"
        )
        run_perturb(
            [*common, "--perturb.mode", "0,0"], expect_fail="must be real"
        )

        # Wrong-system bundle (only the "system" key is read before
        # the rejection fires).
        np.savez(tmp / "wrong.npz", system="plane-couette")
        run_perturb(
            [
                "--init.snapshot",
                str(tmp / "base.tar"),
                "--perturb.out",
                str(tmp / "x.tar"),
                "--perturb.mode",
                "3,0",
                "--perturb.tg_npz",
                str(tmp / "wrong.npz"),
                "--perturb.amplitude_scale",
                "1.0",
            ],
            expect_fail="was computed for system",
        )


# ── Mean mode ────────────────────────────────────────────────────────


def _mean_profiles(compatible: bool) -> np.ndarray:
    """A ``(3, NY)`` real mean-mode profile, legal or not.

    Legal: ``u_x = sin(pi y)`` satisfies the case-B relations (held
    bulk) and ``u_z = sin(pi (y+1)/2)`` the case-A ones; both vanish at
    the walls, and ``u_y`` is zero by continuity.  Illegal: ``1 - y^2``
    has curvature ``-2`` at both walls and moves the bulk -- it is the
    laminar shape, legal as a *base flow* and not as a perturbation.
    """
    y = np.asarray(fmod.flow.ys)
    vec = np.zeros((3, NY), dtype=complex)
    if compatible:
        vec[0] = np.sin(np.pi * y)
        vec[2] = np.sin(np.pi * (y + 1) / 2)
    else:
        vec[0] = 1.0 - y**2
    return vec


def test_mean_mode_injection_under_constant_bulk() -> None:
    """A legal (0,0) profile injects under ``constant_bulk_velocity``.

    Exactly the case the old code refused outright, and the mode is
    self-conjugate, so it writes one column and nothing else.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = make_snapshot(tmp / "base.tar")
        np.save(tmp / "vec.npy", _mean_profiles(True))
        run_perturb(
            [
                "--init.snapshot",
                str(tmp / "base.tar"),
                "--perturb.out",
                str(tmp / "out.tar"),
                "--perturb.npy",
                str(tmp / "vec.npy"),
                "--perturb.mode",
                "0,0",
                "--perturb.amplitude_scale",
                "0.25",
                "--phys.driving",
                "constant_bulk_velocity",
            ]
        )
        state, t, it = load_snapshot(tmp / "out.tar")
        assert (t, it) == (1.25, 50)
        expected = base.copy()
        expected[:, :, 0, 0] += 0.25 * _mean_profiles(True)
        assert_array_equal(np.asarray(state), expected)


def test_mean_mode_error_paths() -> None:
    """Every mean-mode rule refuses, naming what it refused.

    The check is on the *injected profile*, not on the resulting
    state: the relations are homogeneous in the perturbation, so a
    parent's own residual is inherited unchanged (the synthetic parent
    here has a complex, wholly incompatible (0,0) column).
    """
    y = np.asarray(fmod.flow.ys)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_snapshot(tmp / "base.tar")

        def _try(vec, expect, extra=()):
            np.save(tmp / "v.npy", vec)
            return run_perturb(
                [
                    "--init.snapshot",
                    str(tmp / "base.tar"),
                    "--perturb.out",
                    str(tmp / "x.tar"),
                    "--perturb.npy",
                    str(tmp / "v.npy"),
                    "--perturb.mode",
                    "0,0",
                    "--perturb.amplitude_scale",
                    "1.0",
                    *extra,
                ],
                expect_fail=expect,
            )

        _try(_mean_profiles(False), "d(tau)/dy")

        wn = _mean_profiles(True)
        wn[1] = 1.0 - y**2
        _try(wn, "wall-normal component must vanish")

        slip = _mean_profiles(True)
        slip[0] = slip[0] + 1.0
        _try(slip, "no-slip violated")

        # The case-A/case-B split, end to end on one profile:
        # ``(1-y^2)^3`` has zero value, slope and curvature at both
        # walls, so it satisfies the compatibility relations either
        # way -- but it carries bulk ``32/35``, which only a held mean
        # forbids.  Accepted with a free bulk, refused with a held one.
        bulk = np.zeros((3, NY), dtype=complex)
        bulk[0] = (1.0 - y**2) ** 3
        np.save(tmp / "b.npy", bulk)
        common = [
            "--init.snapshot",
            str(tmp / "base.tar"),
            "--perturb.out",
            str(tmp / "b.tar"),
            "--perturb.npy",
            str(tmp / "b.npy"),
            "--perturb.mode",
            "0,0",
            "--perturb.amplitude_scale",
            "1.0",
        ]
        run_perturb(common)
        run_perturb(
            [*common, "--phys.driving", "constant_bulk_velocity"],
            expect_fail="bulk velocity held",
        )


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
