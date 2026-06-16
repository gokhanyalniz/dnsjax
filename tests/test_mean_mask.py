"""Spectral mask and padding-wavenumber tests under forced padding.

Padding slots (appended for 2D mesh divisibility) carry nonzero
beyond-resolution placeholder wavenumbers (``pad_harmonics`` in
``dnsjax.operators``), so the mean mode (global index ``(0, 0)``)
is the only mode with ``k^2 = 0``.  ``Fourier.mean_mask`` is the
one-hot mask of that mode; all mean-mode handling (operator pin
row, influence-matrix mean branch, projections, bulk-velocity
writes) keys on it.  Padding-mode fields stay identically zero
because the forward FFT re-zeroes the padding slots and their
regular operators map zero RHS to zero solutions.

Each case needs its own process because the geometry/sharding
singletons are captured at import time; multiple CPU devices are
obtained via ``--xla_force_host_platform_device_count``.

Run as a script::

    uv run python tests/test_mean_mask.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

NY = 9

# (label, system, nx, nz, np0, np1).  nz = 4 stores nz - 1 = 3
# kz/m modes, padded to 4 under np0 = 2; nx = 6 stores nx // 2 = 3
# kx/kz modes, padded to 4 under np1 = 2 (both-pad cases).
CASES = [
    ("cartesian kz-pad", "plane-couette", 4, 4, 2, 1),
    ("cylindrical kz-pad", "pipe", 4, 4, 2, 1),
    ("annular kz-pad", "taylor-couette", 4, 4, 2, 1),
    ("cartesian both-pad", "plane-couette", 6, 4, 2, 2),
    ("cylindrical both-pad", "pipe", 6, 4, 2, 2),
    ("annular both-pad", "taylor-couette", 6, 4, 2, 2),
]


# ── worker (runs in its own process) ─────────────────────────────────


def _worker(system: str, nx: int, nz: int, np0: int, np1: int) -> None:
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={np0 * np1}"
    )

    import numpy as np

    from dnsjax.parameters import padded_res, params

    params.phys.system = system
    if system == "taylor-couette":
        # Taylor-Couette needs inner/outer Reynolds numbers and a
        # radius ratio; their values do not affect the mask (which
        # depends only on resolution/wavenumbers), so fix them here.
        params.phys.re1 = 100.0
        params.phys.re2 = 0.0
        params.geo.eta = 0.5
    params.res.nx = nx
    params.res.ny = NY
    params.res.nz = nz
    params.res.double_precision = True
    params.dist.np0 = np0
    params.dist.np1 = np1
    params.dist.platform = "cpu"
    padded_res.set_padded_resolution(params)

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.sharding import sharding

    n0_true = nz - 1
    n1_true = nx // 2
    pad0 = (-n0_true) % np0
    pad1 = (-n1_true) % np1
    assert sharding.nz_spec_pad == pad0, sharding.nz_spec_pad
    assert sharding.nx_spec_pad == pad1, sharding.nx_spec_pad
    assert pad0 > 0  # every case forces at least kz/m padding

    if system in ("pipe", "taylor-couette"):
        # Cylindrical and annular share the decoupled azimuthal/axial
        # Fourier layout (m on the kz axis, kz on the kx axis).
        if system == "pipe":
            from dnsjax.geometries.wall_bounded.cylindrical import fourier
        else:
            from dnsjax.geometries.wall_bounded.annular import fourier

        w0 = np.asarray(fourier.m).ravel()  # (nz_spec,)
        w1 = np.asarray(fourier.kz).ravel()  # (nx_spec,)
        k2 = np.asarray(fourier.m2 + fourier.kz2)
    else:
        from dnsjax.geometries.wall_bounded.cartesian import fourier

        w0 = np.asarray(fourier.kz).ravel()  # (nz_spec,)
        w1 = np.asarray(fourier.kx).ravel()  # (nx_spec,)
        k2 = np.asarray(fourier.k2)

    mean_mask = np.asarray(fourier.mean_mask)

    # Padding slots carry nonzero placeholder wavenumbers.
    assert w0.shape == (n0_true + pad0,), w0.shape
    assert w1.shape == (n1_true + pad1,), w1.shape
    assert (w0[n0_true:] != 0).all(), w0
    assert (w1[n1_true:] != 0).all(), w1

    # mean_mask: one-hot at the global (0, 0) mode.
    assert mean_mask.shape == k2.shape, (mean_mask.shape, k2.shape)
    assert mean_mask.sum() == 1, mean_mask.sum()
    assert mean_mask[0, 0, 0]

    # The mean mode is the only k^2 = 0 mode.
    assert ((k2 == 0.0) == mean_mask).all()

    print("worker-ok", flush=True)


# ── orchestrator ─────────────────────────────────────────────────────


def main() -> None:
    for label, system, nx, nz, np0, np1 in CASES:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--system",
                system,
                "--nx",
                str(nx),
                "--nz",
                str(nz),
                "--np0",
                str(np0),
                "--np1",
                str(np1),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or "worker-ok" not in proc.stdout:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"FAIL  mean_mask {label}")
        print(f"  PASS  mean_mask {label}")
    print(f"\nAll {len(CASES)} mean-mask cases passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--system", default="plane-couette")
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--np0", type=int, default=2)
    parser.add_argument("--np1", type=int, default=1)
    args = parser.parse_args()
    if args.worker:
        _worker(args.system, args.nx, args.nz, args.np0, args.np1)
    else:
        main()
