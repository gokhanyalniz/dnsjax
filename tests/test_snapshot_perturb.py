r"""``scripts/snapshot_perturb.py`` injection tests (offline, np = 1).

Builds small plane-Poiseuille snapshots in-process (synthetic spectral
states via ``snapshot.assemble_local_shards`` + ``save_snapshot``),
runs the injection CLI as a subprocess in a temporary directory, and
verifies against the reloaded arrays:

1. **Exactness**: a ``--npy`` injection at ``(i2, i3) = (3, 0)`` adds
   exactly ``scale * vec`` at the target column and ``scale *
   conj(vec)`` at the real-FFT conjugate partner ``((nz-1)-3, 0)``,
   leaves every other entry ``==``-identical, and preserves the
   parent's ``(t, it)``.
2. **Energy convention**: ``--amplitude-energy E0`` onto a zero
   snapshot yields a state whose solver-measure ``E'``
   (``get_perturbation_energy``) is ``E0`` to roundoff.
3. **Antithesis**: ``--negate`` gives the exact mirror perturbation
   (the ``+``/``-`` deltas cancel identically).
4. **Transient-growth source**: a real (subprocess) TG run on the
   laminar profile produces ``<stem>_tg.npz``; ``--tg-npz`` injects a
   field collinear with its ``opt_input`` row at the requested
   energy -- pinning the npz key contract against the actual writer.
5. **Error paths**: out-of-range mode, complex ``(0,0)`` profile,
   mode missing from the TG bundle, wrong-system bundle.

Usage::

    uv run python tests/test_snapshot_perturb.py
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
# ``update_parameters`` (not direct ``params.*`` assignment): the
# energy checks below need the derived parameters -- ``volume_fac``
# enters ``get_norm2`` -- which only ``update_parameters`` computes.
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

import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

import dnsjax.flows.wall_bounded.plane_poiseuille as fmod  # noqa: E402
from dnsjax.snapshot import (  # noqa: E402
    assemble_local_shards,
    load_snapshot,
    save_snapshot,
)

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "snapshot_perturb.py"
N2 = NZ - 1  # true complex-axis mode count


def _make_snapshot(path: Path, zero: bool = False, seed: int = 0):
    """Write a synthetic snapshot; return its spectral array."""
    rng = np.random.default_rng(seed)

    def fill_local(buf, kz_start, nkz, kx_start, nkx):
        if zero:
            return
        shape = (3, NY, nkz, nkx)
        buf[:, :, :nkz, :nkx] = rng.standard_normal(
            shape
        ) + 1j * rng.standard_normal(shape)

    state = assemble_local_shards(fill_local)
    save_snapshot(state, 1.25, 50, path, isnap=0)
    return np.asarray(state)


def _run_perturb(args: list[str], expect_fail: str | None = None) -> str:
    """Run the injection CLI; assert success (or the failure text)."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    if expect_fail is None:
        assert result.returncode == 0, output[-2000:]
    else:
        assert result.returncode != 0, output[-2000:]
        assert expect_fail in output, output[-2000:]
    return output


def test_npy_injection_exact() -> None:
    rng = np.random.default_rng(7)
    vec = rng.standard_normal((3, NY)) + 1j * rng.standard_normal((3, NY))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = _make_snapshot(tmp / "base.tar")
        np.save(tmp / "vec.npy", vec)
        _run_perturb(
            [
                "--snapshot",
                str(tmp / "base.tar"),
                "--out",
                str(tmp / "out.tar"),
                "--mode",
                "3,0",
                "--npy",
                str(tmp / "vec.npy"),
                "--amplitude-scale",
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
        _make_snapshot(tmp / "zero.tar", zero=True)
        np.save(tmp / "vec.npy", vec)
        _run_perturb(
            [
                "--snapshot",
                str(tmp / "zero.tar"),
                "--out",
                str(tmp / "out.tar"),
                "--mode",
                "3,0",
                "--npy",
                str(tmp / "vec.npy"),
                "--amplitude-energy",
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
        base = _make_snapshot(tmp / "base.tar")
        np.save(tmp / "vec.npy", vec)
        common = [
            "--snapshot",
            str(tmp / "base.tar"),
            "--mode",
            "3,0",
            "--npy",
            str(tmp / "vec.npy"),
            "--amplitude-energy",
            "1e-6",
        ]
        _run_perturb([*common, "--out", str(tmp / "plus.tar")])
        _run_perturb([*common, "--out", str(tmp / "minus.tar"), "--negate"])
        plus = np.asarray(load_snapshot(tmp / "plus.tar")[0])
        minus = np.asarray(load_snapshot(tmp / "minus.tar")[0])
        assert_array_equal(plus - base, -(minus - base))
        assert np.abs(plus - base).max() > 0


def test_tg_npz_source() -> None:
    """End-to-end against the real transient-growth npz writer."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Laminar total profile on the snapshot's own CGL grid.
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
            capture_output=True,
            text=True,
            cwd=tmp,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        npz_path = tmp / "lam_tg.npz"
        with np.load(npz_path) as npz:
            opt_input = np.asarray(npz["opt_input"][0])

        _make_snapshot(tmp / "zero.tar", zero=True)
        _run_perturb(
            [
                "--snapshot",
                str(tmp / "zero.tar"),
                "--out",
                str(tmp / "out.tar"),
                "--mode",
                "3,0",
                "--tg-npz",
                str(npz_path),
                "--which",
                "input",
                "--amplitude-energy",
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
        _run_perturb(
            [
                "--snapshot",
                str(tmp / "zero.tar"),
                "--out",
                str(tmp / "x.tar"),
                "--mode",
                "2,0",
                "--tg-npz",
                str(npz_path),
                "--amplitude-energy",
                "1e-4",
            ],
            expect_fail="is not in",
        )

        # Controllability-mode source: operator_tools CLI on the
        # --save-operator bundle, then --modes-npz injection.
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnsjax.analysis.response.operator_tools",
                "--operator",
                str(tmp / "lam_tg_op.npz"),
                "--n-modes",
                "3",
                "--out",
                str(tmp / "cont.npz"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        _run_perturb(
            [
                "--snapshot",
                str(tmp / "zero.tar"),
                "--out",
                str(tmp / "cm.tar"),
                "--mode",
                "3,0",
                "--modes-npz",
                str(tmp / "cont.npz"),
                "--index",
                "1",
                "--amplitude-energy",
                "1e-4",
            ]
        )
        with np.load(tmp / "cont.npz") as z:
            cont_vec = np.asarray(z["cont_modes_3_0"][1])
        state, _, _ = load_snapshot(tmp / "cm.tar")
        new = np.asarray(state)
        col = new[:, :, 3, 0]
        scale = np.abs(col).max() / np.abs(cont_vec).max()
        assert_allclose(col, scale * cont_vec, atol=1e-12 * scale)
        e_prime = float(fmod.get_perturbation_energy(state))
        assert_allclose(e_prime, 1e-4, rtol=1e-12)

        # --index out of range against the real bundle.
        _run_perturb(
            [
                "--snapshot",
                str(tmp / "zero.tar"),
                "--out",
                str(tmp / "x.tar"),
                "--mode",
                "3,0",
                "--modes-npz",
                str(tmp / "cont.npz"),
                "--index",
                "3",
                "--amplitude-energy",
                "1e-4",
            ],
            expect_fail="out of range",
        )


def test_error_paths() -> None:
    rng = np.random.default_rng(10)
    vec = rng.standard_normal((3, NY)) + 1j * rng.standard_normal((3, NY))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _make_snapshot(tmp / "base.tar")
        np.save(tmp / "vec.npy", vec)
        common = [
            "--snapshot",
            str(tmp / "base.tar"),
            "--out",
            str(tmp / "x.tar"),
            "--npy",
            str(tmp / "vec.npy"),
            "--amplitude-scale",
            "1.0",
        ]
        _run_perturb(
            [*common, "--mode", f"{N2},0"], expect_fail="out of range"
        )
        _run_perturb([*common, "--mode", "0,0"], expect_fail="must be real")

        # Wrong-system bundle (only the "system" key is read before
        # the rejection fires).
        np.savez(tmp / "wrong.npz", system="plane-couette")
        _run_perturb(
            [
                "--snapshot",
                str(tmp / "base.tar"),
                "--out",
                str(tmp / "x.tar"),
                "--mode",
                "3,0",
                "--tg-npz",
                str(tmp / "wrong.npz"),
                "--amplitude-scale",
                "1.0",
            ],
            expect_fail="was computed for system",
        )


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for tfun in tests:
        tfun()
        print(f"  PASS  {tfun.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
