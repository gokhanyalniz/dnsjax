r"""``scripts/snapshot_perturb.py`` with the lab's controllability modes.

The lab's part of ``tests/test_snapshot_perturb.py`` (the shared
fixtures: ``tests/_snapshot_perturb.py``): the
:mod:`dnsjax_lab.analysis.response.operator_tools` CLI turns a real
transient-growth ``--tg.save_operator`` bundle into controllability
modes, and ``--perturb.modes_npz`` injects the requested one collinear
with its profile at the requested energy; an index past the bundle's
modes is refused.

Usage::

    uv run python lab/tests/test_snapshot_perturb.py
"""

from __future__ import annotations

import _repo  # noqa: F401 -- the core suite's helpers
from _snapshot_perturb import configure, make_snapshot, run_perturb, tg_bundle

configure()

import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from _live import run_live  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

import dnsjax.flows.wall_bounded.plane_poiseuille as fmod  # noqa: E402
from dnsjax.snapshot import load_snapshot  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)


def test_controllability_modes_source() -> None:
    """operator_tools' modes bundle, injected by --perturb.modes_npz."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tg_bundle(tmp)
        make_snapshot(tmp / "zero.tar", zero=True)
        result = run_live(
            [
                sys.executable,
                "-m",
                "dnsjax_lab.analysis.response.operator_tools",
                "--operator",
                str(tmp / "lam_tg_op.npz"),
                "--n-modes",
                "3",
                "--out",
                str(tmp / "cont.npz"),
            ]
        )
        assert result.returncode == 0, result.stdout + result.stderr
        common = [
            "--init.snapshot",
            str(tmp / "zero.tar"),
            "--perturb.mode",
            "3,0",
            "--perturb.modes_npz",
            str(tmp / "cont.npz"),
            "--perturb.amplitude_energy",
            "1e-4",
        ]
        run_perturb(
            [*common, "--perturb.out", str(tmp / "cm.tar")]
            + ["--perturb.index", "1"]
        )
        with np.load(tmp / "cont.npz") as z:
            cont_vec = np.asarray(z["profiles_3_0"][1])
        state, _, _ = load_snapshot(tmp / "cm.tar")
        col = np.asarray(state)[:, :, 3, 0]
        scale = np.abs(col).max() / np.abs(cont_vec).max()
        assert_allclose(col, scale * cont_vec, atol=1e-12 * scale)
        e_prime = float(fmod.get_perturbation_energy(state))
        assert_allclose(e_prime, 1e-4, rtol=1e-12)

        # --index out of range against the real bundle.
        run_perturb(
            [*common, "--perturb.out", str(tmp / "x.tar")]
            + ["--perturb.index", "3"],
            expect_fail="out of range",
        )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
