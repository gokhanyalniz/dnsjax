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
   actual writer.
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
from _live import run_live  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

import dnsjax.flows.wall_bounded.plane_poiseuille as fmod  # noqa: E402
from dnsjax.snapshot import (  # noqa: E402
    assemble_local_shards,
    load_snapshot,
    save_snapshot,
)

sys.stdout.reconfigure(line_buffering=True)

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
        _make_snapshot(tmp / "zero.tar", zero=True)
        np.save(tmp / "vec.npy", vec)
        _run_perturb(
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
        base = _make_snapshot(tmp / "base.tar")
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
        _run_perturb([*common, "--perturb.out", str(tmp / "plus.tar")])
        _run_perturb(
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


def test_tg_npz_source() -> None:
    """End-to-end against the real transient-growth npz writer."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Laminar total profile on the snapshot's own CGL grid.
        y = np.cos(np.pi * np.arange(NY) / (NY - 1))
        with open(tmp / "lam.txt", "w") as f:
            for yi, ui in zip(y, 1.0 - y**2, strict=True):
                f.write(f"{yi:+.17e} {ui:+.17e}\n")
        result = run_live(
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
        assert result.returncode == 0, result.stdout + result.stderr
        npz_path = tmp / "lam_tg.npz"
        with np.load(npz_path) as npz:
            opt_input = np.asarray(npz["opt_input"][0])

        _make_snapshot(tmp / "zero.tar", zero=True)
        _run_perturb(
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
        _run_perturb(
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

        # Controllability-mode source: operator_tools CLI on the
        # --tg.save_operator bundle, then --perturb.modes_npz
        # injection.
        result = run_live(
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
            ]
        )
        assert result.returncode == 0, result.stdout + result.stderr
        _run_perturb(
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
        _run_perturb(
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
        _make_snapshot(tmp / "base.tar")
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
        _run_perturb(
            [*common, "--perturb.mode", f"{N2},0"], expect_fail="out of range"
        )
        _run_perturb(
            [*common, "--perturb.mode", "0,0"], expect_fail="must be real"
        )

        # Wrong-system bundle (only the "system" key is read before
        # the rejection fires).
        np.savez(tmp / "wrong.npz", system="plane-couette")
        _run_perturb(
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
        base = _make_snapshot(tmp / "base.tar")
        np.save(tmp / "vec.npy", _mean_profiles(True))
        _run_perturb(
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
        _make_snapshot(tmp / "base.tar")

        def _try(vec, expect, extra=()):
            np.save(tmp / "v.npy", vec)
            return _run_perturb(
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
        _run_perturb(common)
        _run_perturb(
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
