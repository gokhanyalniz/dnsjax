#!/usr/bin/env python3
r"""Construction self-test for the in-process IC builders.

Builds the deterministic streamwise-localized-rolls IC
(``dnsjax.ic.localized_rolls``) for each core flow and checks the
*construction* properties the smoke test (``tests/test_rolls_smoke.py``)
cannot -- without time-stepping -- plus, in the same subprocess (no
extra launch), the divergence guard on the random-field IC
(``dnsjax.ic.random_field``, the default start mode).  The workers:
``tests/_localized_rolls.py``, which the lab's flows share
(``lab/tests/test_localized_rolls.py``).  The checks:

- **finiteness** of the built spectral state;
- **exact no-slip** at the wall nodes (the wall ``y`` / ``r`` slice of
  the *velocity* block is identically zero, never transformed): both
  walls for Cartesian / annular, the outer wall ``r = 1`` for the
  cylindrical family (the inner end is the axis); for a total-field
  flow the *total* velocity still vanishes, and a state carrying more
  than the velocity is checked on its velocity block;
- **bit-identical determinism** (two builds in one process agree) and
  **device-count independence** (the true modes are identical at
  ``(np0, np1) = (1, 1)``, ``(1, 2)`` and ``(2, 1)``) -- the
  no-replication, per-device build must not depend on the mesh; on
  the random state that independence is to ``RAND_CROSS_TOL`` rather
  than bit-exact, for the reason recorded there;
- an **exactly zero (kx, kz) = (0, 0) mode** on every component, so a
  spot leaves the field's bulk velocity and wall shear untouched (the
  "Mean-free by construction" note in :mod:`dnsjax.ic.localized_rolls`);
  checked at a box whose ``L / wavelength`` is not an integer, the only
  regime in which the roll factors have a DC bin to leak;
- a **loose truncation-level discrete-divergence bound** (the analytic
  profiles are continuously divergence-free; the discrete divergence is
  only FD-truncation-sized and is projected out by the first corrector
  step at run start): Cartesian and the ``m0 = 1`` pipe are
  ~machine-zero (the FD order differentiates the quartic profile
  exactly), the rational-profile annular and the `$r^{m_0 - 1}$`-raised
  pipe wedge `$O(10^{-5})$`.  **Kolmogorov is held to a
  machine-precision bound instead** (``DIV_TOL_SPECTRAL``, relative):
  with no wall-normal direction there is no FD operator to truncate
  against, `$G'$` is the exact `$\mathrm{i}k_y\hat G$`, and the two
  divergence terms are the same product of the same factors with
  opposite sign -- so a truncation-level bound there would pass on a
  construction that had stopped cancelling;
- **reality**, Kolmogorov only: the periodic family's joint
  `$f(k_y,k_z,0) = \overline{f(-k_y,-k_z,0)}$` condition on the
  `$k_x = 0$` plane, which the rolls get free from separability while
  the random-field builder needs a dedicated helper for it;
- a **peak-velocity / domain-scaling guard** (``_check_peak_scaling``,
  subprocess ``--peak``): built at two resolved box sizes, the sampled
  ``max|u'|`` is within ``PEAK_TOL`` (12%) of ``amplitude`` and
  domain-independent -- which a single-mode construction, whose
  amplitude grows with box length, cannot pass.

Each ``(system, np0, np1)`` configuration runs in its own subprocess with
forced CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count``), mirroring
``tests/test_snapshot.py`` -- so the multi-device cases need no MPI.  Run
directly::

    uv run python tests/test_localized_rolls.py            # all systems
    uv run python tests/test_localized_rolls.py --system pipe --out p.npy

The second form is one worker, one configuration (``--np0`` / ``--np1``,
default 1).  Each worker writes its true-mode array to its ``--out``
``.npy`` so the driver can compare across device counts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from _live import report, run_live
from _localized_rolls import run_case, run_worker

sys.stdout.reconfigure(line_buffering=True)

#: ``name -> case`` (the keys: ``tests/_localized_rolls.py``).  A wedge
#: case is a flow at a non-default azimuthal wedge (``geo.m0``): the
#: pipe rolls' azimuthal factor is one period over the wedge, i.e. the
#: physical ``m = m0``, and their radial profiles must follow it for the
#: field to stay solenoidal and axis-regular
#: (``generate_cylindrical_rolls``) -- which the divergence check
#: measures with the physical ``m``.
CASES: dict[str, dict] = {
    "kolmogorov": {},
    "plane-couette": {},
    "plane-poiseuille": {},
    "pipe": {},
    "pipe-wedge": {"system": "pipe", "geo": {"m0": 2}},
    "taylor-couette": {
        "phys": {"re1": 100.0, "re2": -100.0},
        "geo": {"eta": 0.5},
    },
}

# Peak-velocity / domain-scaling guard.  The spot is peak-normalized so
# max|u'| = amplitude and is domain-independent (a single-mode
# construction, whose amplitude grows with box length, would not be).
# Built at two resolved boxes (the spot fits and is well sampled at
# both) with the spanwise/axial wavelength a fixed physical value, so
# peak ~ amplitude must hold at both and agree across them.
PEAK_SYSTEMS = ["kolmogorov", "plane-couette", "pipe", "taylor-couette"]
PEAK_AMP, PEAK_WIDTH, PEAK_WAVE = 0.1, 2.0, 4.0
PEAK_BOXES = ((20.0, 48), (40.0, 96))  # (box length, resolution)
PEAK_TOL = 0.12  # |peak/amp - 1| and cross-box agreement bound


# ── peak scaling ─────────────────────────────────────────────────


def _peak_build(system: str, box: float, n: int) -> int:
    """Subprocess: build rolls at a resolved box; print ``PEAK=max|u'|``."""
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
    os.environ["NPROC"] = "1"
    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    phys: dict = {"system": system, "re": 100.0}
    geo: dict = {"lx": box, "lz": box}
    if system == "taylor-couette":
        phys.update(re1=100.0, re2=-100.0)
        geo["eta"] = 0.5
    # Kolmogorov's shear direction is Fourier too, and its period is
    # fixed by the geometry -- so ``box`` scales two of three
    # directions there, and ``ny`` is a mode count like the others.
    ny = n if system == "kolmogorov" else 33
    update_parameters(
        Parameters(
            dist={"np0": 1, "np1": 1},
            phys=phys,
            geo=geo,
            res={
                "nx": n,
                "ny": ny,
                "nz": n,
                "fd_order": 4,
                "double_precision": True,
            },
            outs={},
        )
    )
    padded_res.set_padded_resolution(params)
    configure_jax_platform(platform_from_argv())

    from dnsjax.ic.localized_rolls import generate_localized_rolls
    from dnsjax.operators import spec_to_phys, spec_to_phys_2d

    state = generate_localized_rolls(PEAK_AMP, PEAK_WIDTH, PEAK_WAVE)
    # ``spec_to_phys_2d`` leaves the wall-normal axis alone; the
    # triply-periodic state has no such axis and takes the 3D one.
    to_phys = spec_to_phys if system == "kolmogorov" else spec_to_phys_2d
    pf = np.asarray(to_phys(state))
    # Native components are orthonormal in every geometry:
    # |u'|^2 is the plain component sum.
    mag = np.sqrt(np.sum(pf**2, axis=0))
    print(f"PEAK={float(mag.max())}")
    return 0


def _check_peak_scaling() -> str | None:
    """Peak |u'| = amplitude, domain-independent (subprocess per build).

    ``None`` on a full pass, else the one-line summary reason.
    """
    print("=== peak |u'| = amplitude (domain-independent spot) ===")
    bad: list[str] = []
    for system in PEAK_SYSTEMS:
        peaks = []
        for box, n in PEAK_BOXES:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--peak",
                "--system",
                system,
                "--box",
                str(box),
                "--n",
                str(n),
            ]
            proc = run_live(cmd, timeout=300)
            line = next(
                (
                    ln
                    for ln in proc.stdout.splitlines()
                    if ln.startswith("PEAK=")
                ),
                None,
            )
            if proc.returncode != 0 or line is None:
                print(f"  FAIL: {system} box={box} worker error")
                print(proc.stderr[-800:])
                bad.append(f"{system} box={box} worker error")
                break
            peaks.append(float(line.split("=")[1]))
        else:
            near = all(abs(p / PEAK_AMP - 1) < PEAK_TOL for p in peaks)
            indep = abs(peaks[0] - peaks[1]) / PEAK_AMP < PEAK_TOL
            print(
                f"  {'PASS' if near and indep else 'FAIL'}: {system}  "
                f"peak(L{int(PEAK_BOXES[0][0])})={peaks[0]:.4f} "
                f"peak(L{int(PEAK_BOXES[1][0])})={peaks[1]:.4f} "
                f"(amp={PEAK_AMP})"
            )
            if not (near and indep):
                bad.append(
                    f"{system} peaks {peaks[0]:.4f}/{peaks[1]:.4f} "
                    f"vs amp={PEAK_AMP}"
                )
    return "; ".join(bad) if bad else None


def main() -> None:
    if "--peak" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--peak", action="store_true")
        p.add_argument("--system", required=True)
        p.add_argument("--box", type=float, required=True)
        p.add_argument("--n", type=int, required=True)
        a = p.parse_args()
        sys.exit(_peak_build(a.system, a.box, a.n))

    if "--system" in sys.argv:
        p = argparse.ArgumentParser()
        p.add_argument("--system", required=True)
        p.add_argument("--np0", type=int, default=1)
        p.add_argument("--np1", type=int, default=1)
        p.add_argument("--out", required=True)
        a = p.parse_args()
        sys.exit(run_worker(CASES[a.system], a.system, a.np0, a.np1, a.out))

    print(
        "Localized-rolls construction self-test: offline, forced CPU "
        "devices (device-count independence is the property under test; "
        "no GPU path).",
        flush=True,
    )
    # Each check returns None when it passes, else its one-line reason;
    # ``report`` repeats the failures after the counts (see _live).
    results: list[tuple[str, str | None]] = [
        (name, run_case(__file__, case, name)) for name, case in CASES.items()
    ]
    results.append(("peak |u'| scaling", _check_peak_scaling()))
    failures = [(n, r) for n, r in results if r is not None]
    sys.exit(report(len(results) - len(failures), failures))


if __name__ == "__main__":
    main()
