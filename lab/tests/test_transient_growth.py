r"""Transient-growth analysis of the lab's base-flow flows.

The lab's cases of ``tests/test_transient_growth.py`` (the shared
pieces: ``tests/_transient_growth.py``), for quasi-Keplerian:

* the in-process worker (one forced-CPU subprocess): the flow's
  ``frozen_profile_flow`` hook reproduces the builtin laminar coupling,
  and the linear step is block-diagonal per Fourier mode;
* the CLI smoke run;
* the published value: axially-periodic quasi-Keplerian optimal
  transient growth, Shi et al., Phys. Fluids 29, 044107 (2017),
  Table III case I, ``G_opt = 13.04`` (1 % relative) at
  ``t_opt/tau_d = 27`` (2 %), in a linearly stable regime (negative
  spectral abscissa).

Run: ``uv run python lab/tests/test_transient_growth.py`` (``--fast``
skips the published value; ``--worker`` runs the worker).
"""

import argparse
import sys
import tempfile
from pathlib import Path

import _repo  # noqa: F401 -- the core suite's helpers
from _live import run_live
from _transient_growth import (
    REPO,
    ccf_profile,
    cli_smoke,
    close,
    load,
    run_tg,
    worker,
    write_ccf_laminar,
)

sys.stdout.reconfigure(line_buffering=True)

SYSTEM = "quasi-keplerian"


def _qk_re2(re1: float, eta: float, r_omega: float) -> float:
    """Quasi-Keplerian derived outer Reynolds number Re_o."""
    return re1 * (1 - eta + r_omega) / (eta * r_omega - (1 - eta))


def _write_laminar(path: Path, re1: float, r_omega: float, eta: float):
    """The laminar profile, at the derived Re_o."""
    write_ccf_laminar(path, re1, _qk_re2(re1, eta, r_omega), eta)


WORKER = {
    "phys": {"re1": 100.0, "r_omega": -1.2},
    "geo": {"eta": 0.71},
    "profile": ccf_profile,
    # The circular-Couette B0/r curl carries FD error.
    "curl_tol": 3e-4,
}
SMOKE = {
    "write_laminar": lambda path: _write_laminar(path, 50.0, -1.2, 0.71),
    "phys_args": [
        "--phys.re1",
        "50",
        "--phys.r_omega",
        "-1.2",
        "--geo.eta",
        "0.71",
        "--geo.lz",
        "6.2832",
    ],
}


def _anchor_qk() -> None:
    print("  quasi-Keplerian (annular):")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # Table III case I (the module docstring): eta = 0.71, R_Omega =
        # -1.2, Re_i = 1e4, axial wavenumber k_z = 0, azimuthal m = 4;
        # the growth is purely non-modal.  Code time == tau_d, so t_opt
        # is in the paper's units.  m = 4 sits at index i2 = 4 for
        # ntheta = 10.
        _write_laminar(p / "k.txt", 1.0e4, -1.2, 0.71)
        run_tg(
            p / "k.txt",
            p,
            [
                "--phys.system",
                SYSTEM,
                "--phys.re1",
                "10000",
                "--phys.r_omega",
                "-1.2",
                "--geo.eta",
                "0.71",
                "--res.nr",
                "128",
                "--res.nz",
                "4",
                "--res.ntheta",
                "10",
                "--res.fd_order",
                "8",
                "--geo.lz",
                "0.5",
                "--tg.modes",
                "4,0",
                "--tg.t_max",
                "60",
                "--tg.nt",
                "60",
            ],
        )
        z = load(p, "k")
        assert float(z["spectral_abscissa"][0]) < 0, "QK must be stable"
        close(float(z["G_max"][0]), 13.04, 0.01, "QK G_opt")
        close(float(z["t_opt"][0]), 27.0, 0.02, "QK t_opt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument(
        "--fast",
        action="store_true",
        help="structural checks only (skip the published value)",
    )
    args = ap.parse_args()
    if args.worker:
        worker(SYSTEM, **WORKER)
        return
    print("[hooks + block-diagonality]", flush=True)
    res = run_live([sys.executable, __file__, "--worker"], cwd=REPO)
    if res.returncode != 0:
        raise SystemExit(f"{SYSTEM}: worker failed")
    print("[CLI features]", flush=True)
    cli_smoke(SYSTEM, **SMOKE)
    if not args.fast:
        print("[published values]", flush=True)
        _anchor_qk()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
