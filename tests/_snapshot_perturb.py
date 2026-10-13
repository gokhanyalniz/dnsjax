r"""Shared fixtures of the ``scripts/snapshot_perturb.py`` tests.

The core suite's ``test_snapshot_perturb.py`` and the lab's script of
the same name build
the same small plane-Poiseuille configuration (:func:`configure`, once
per process before anything imports ``sharding``), write synthetic
snapshots (:func:`make_snapshot`), run the injection CLI
(:func:`run_perturb`) and a transient-growth export to inject from
(:func:`tg_bundle`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from _live import run_live

NX, NY, NZ = 8, 25, 8
N2 = NZ - 1  # true complex-axis mode count

#: The injection CLI under test.
SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "snapshot_perturb.py"
)


def configure() -> None:
    """The plane-Poiseuille singletons, then JAX."""
    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    # ``update_parameters`` (not direct ``params.*`` assignment): the
    # energy checks need the derived parameters -- ``volume_fac``
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
    configure_jax_platform(platform_from_argv())


def make_snapshot(path: Path, zero: bool = False, seed: int = 0):
    """Write a synthetic snapshot; return its spectral array."""
    from dnsjax.snapshot import assemble_local_shards, save_snapshot

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


def run_perturb(args: list[str], expect_fail: str | None = None) -> str:
    """Run the injection CLI; assert success (or the failure text)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
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


def tg_bundle(tmp: Path) -> None:
    """A real transient-growth run of mode ``(3, 0)``, operator exported.

    On the laminar total profile, on the snapshot's own CGL grid; writes
    ``lam_tg.npz`` and ``lam_tg_op.npz`` into *tmp*.
    """
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
