"""Mean-mode mask tests under forced spectral padding.

``Fourier.mean_mask`` must be ``True`` only at the true mean mode
(global index ``(0, 0)``), while ``k2_is_zero`` is also ``True``
at the zero-padded dummy modes appended for 2D mesh divisibility
(their stored wavenumbers are zero).  The IMM bulk-velocity
corrections write through ``mean_mask`` so the dummy modes stay
exactly zero; ``k2_is_zero`` remains in use for operator gauge
fixing.

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

# nz = 4 stores nz - 1 = 3 kz/m modes; with np0 = 2 this is padded
# to 4 (one dummy mode), exercising the mask distinction.
NX, NY, NZ = 4, 9, 4

CASES = [
    ("cartesian", "plane-couette"),
    ("cylindrical", "pipe"),
]


# ── worker (runs in its own process) ─────────────────────────────────


def _worker(system: str) -> None:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

    import numpy as np

    from dnsjax.parameters import padded_res, params

    params.phys.system = system
    params.res.nx = NX
    params.res.ny = NY
    params.res.nz = NZ
    params.res.double_precision = True
    params.dist.np0 = 2
    params.dist.np1 = 1
    params.dist.platform = "cpu"
    padded_res.set_padded_resolution(params)

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.sharding import sharding

    assert sharding.nz_spec_pad == 1, sharding.nz_spec_pad

    if system == "pipe":
        from dnsjax.geometries.wall_bounded.cylindrical import fourier
    else:
        from dnsjax.geometries.wall_bounded.cartesian import fourier

    mean_mask = np.asarray(fourier.mean_mask)
    k2_is_zero = np.asarray(fourier.k2_is_zero)

    assert mean_mask.shape == k2_is_zero.shape, (
        mean_mask.shape,
        k2_is_zero.shape,
    )
    # Exactly one True entry, at the global (0, 0) mode.
    assert mean_mask.sum() == 1, mean_mask.sum()
    assert mean_mask[0, 0, 0]
    # mean_mask is a strict subset of k2_is_zero: the padded dummy
    # mode (stored wavenumbers zero) is in k2_is_zero only.
    assert (mean_mask <= k2_is_zero).all()
    assert k2_is_zero.sum() == 2, k2_is_zero.sum()

    print("worker-ok", flush=True)


# ── orchestrator ─────────────────────────────────────────────────────


def main() -> None:
    for name, system in CASES:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--worker",
                "--system",
                system,
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or "worker-ok" not in proc.stdout:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(f"FAIL  mean_mask {name}")
        print(f"  PASS  mean_mask {name}")
    print(f"\nAll {len(CASES)} mean-mask cases passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--system", default="plane-couette")
    args = parser.parse_args()
    if args.worker:
        _worker(args.system)
    else:
        main()
