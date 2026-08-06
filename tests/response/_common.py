"""Shared fixture helpers for the ``tests/response`` scripts.

The three identification test scripts (``test_ensemble.py``,
``test_lim.py``, ``test_ssi.py``) build the same fixtures: a checked
subprocess wrapper (:func:`run`), a real transient-growth operator
export + controllability bundle for plane-Poiseuille mode ``(3, 0)``
at the shared tiny resolution (:func:`operator_artifacts`), and
hand-written synthetic ``probes.bin``/``probes.json`` streams
(:func:`write_probe_stream`).  Each script stays standalone (its own
directory is ``sys.path[0]``, so ``import _common`` works run
directly or through the pytest bridge) and configures the dnsjax
singletons itself; this module imports neither JAX nor dnsjax.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _live import run_live

#: Shared fixture configuration: a real but tiny plane-Poiseuille
#: resolution, and the probe cadence of the synthetic streams (the
#: scripts run at the parameter-model default dt and pass it in).
NX, NY, NZ = 8, 25, 8
IT_PROBES = 10


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """``run_live`` + a zero-exit assert carrying the output tail."""
    result = run_live(cmd, **kw)
    assert result.returncode == 0, (
        " ".join(str(c) for c in cmd)
        + "\n"
        + result.stdout[-3000:]
        + result.stderr[-3000:]
    )
    return result


def operator_artifacts(tmp: Path) -> tuple[Path, Path]:
    """Real TG operator export + controllability bundle (mode (3,0))."""
    y = np.cos(np.pi * np.arange(NY) / (NY - 1))
    with open(tmp / "lam.txt", "w") as f:
        for yi, ui in zip(y, 1.0 - y**2, strict=True):
            f.write(f"{yi:+.17e} {ui:+.17e}\n")
    run(
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
    run(
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


def write_probe_stream(
    directory: Path, u: np.ndarray, dt: float, t0: float = 0.0
) -> None:
    """Hand-written ``probes.bin``/``probes.json`` probing mode (3,0).

    *u* is the ``(nt, 3, NY)`` complex profile series; samples sit at
    ``t0 + k * IT_PROBES * dt``.
    """
    from dnsjax.analysis.response.probes import MIN_FORMAT_VERSION

    nt = u.shape[0]
    sidecar = {
        "format_version": MIN_FORMAT_VERSION,
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
        "dt": dt,
        "system": "plane-poiseuille",
        "double_precision": True,
        "git_hash": "synthetic",
        "params": {},
    }
    rec_dtype = np.dtype([("t", "<f8"), ("u", "<f8", (1, 3, NY, 2))])
    rec = np.zeros(nt, dtype=rec_dtype)
    rec["t"] = t0 + np.arange(nt) * IT_PROBES * dt
    rec["u"][..., 0] = u.real
    rec["u"][..., 1] = u.imag
    with open(directory / "probes.json", "w") as f:
        json.dump(sidecar, f)
    (directory / "probes.bin").write_bytes(rec.tobytes())
