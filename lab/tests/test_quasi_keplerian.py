r"""Quasi-Keplerian control-parameter tests.

Offline (no solver run) checks for the ``quasi-keplerian`` flow
system.  Each case runs in its own subprocess (the ``test_*``
subprocess-per-config idiom: ``dnsjax.parameters`` captures the global
``params`` once).  The azimuthal wedge (``geo.m0``) it shares with the
core annular flows is checked by the core suite's ``test_wedge.py``.

Cases:

1. ``derive``: the (Re_i, R_Omega, eta) -> Re_o inversion at the
   literature line eta = 0.71, R_Omega = -1.2 -- Re_o/Re_i, the round
   trip R_Omega(Re_i, Re_o), Re_s, mu, the circular-Couette wall values
   U_theta(r1) = 1 / U_theta(r2) = Re_o/Re_i, and the local exponent
   q(r1), q(r2) in (0, 2) (pinned against Table II of the reproduced
   axially-periodic quasi-Keplerian study, Shi et al., Phys. Fluids
   2017: Re_s = 5078.8 at Re_i = 1e4).
2. ``resume``: a consistent externally-supplied ``phys.re2`` (the
   snapshot-resume replay path) is accepted and left unchanged.
3. ``err_romega``: a missing ``r_omega`` and the Rayleigh-line /
   co-rotating rejections (R_Omega >= -1).
4. ``err_re1``: the ``re1`` sign-convention rejections (None, 0, < 0).
5. ``override_re2``: a conflicting directly-assigned or layered
   ``phys.re2`` is silently overwritten by the (re1, r_omega, eta)
   derivation (``re2`` is not a quasi-Keplerian parameter; the CLI/TOML
   surfaces reject it outright).

Usage::

    uv run python lab/tests/test_quasi_keplerian.py
"""

from __future__ import annotations

import argparse
import math
import sys

import _repo  # noqa: F401 -- the core suite's helpers
from _live import run_live

sys.stdout.reconfigure(line_buffering=True)

ETA = 0.71
R_OMEGA = -1.2
# Reference values on the eta = 0.71, R_Omega = -1.2 half-line.
RE_O_RATIO = 0.7968476357267952  # Re_o / Re_i
MU = 0.5657618213660247  # eta * Re_o / Re_i
RE_S_RATIO = 0.5078809106830124  # Re_s / Re_i (-> 5078.8 at Re_i = 1e4)
Q_INNER = 1.7513  # q(r1)
Q_OUTER = 1.5604  # q(r2)


def _re2(re_i: float, eta: float, r_om: float) -> float:
    """Derived outer Reynolds number Re_o (the branch's inversion)."""
    return re_i * (1 - eta + r_om) / (eta * r_om - (1 - eta))


# ── individual cases (each in its own subprocess) ──────────────────


def case_derive() -> None:
    """Pin the (Re_i, R_Omega, eta) -> circular-Couette derivation."""
    from dnsjax.parameters import (
        Parameters,
        derived_params,
        params,
        update_parameters,
        validate_parameters,
    )

    re_i = 1.0e4
    qk = {"system": "quasi-keplerian", "re1": re_i, "r_omega": R_OMEGA}
    update_parameters(
        Parameters(
            phys=qk,
            geo={"eta": ETA},
        )
    )
    validate_parameters()

    re_o = params.phys.re2
    assert params.phys.re == re_i, "re must equal the inner Re_i"
    # Round-trip R_Omega(Re_i, Re_o).
    r_om_back = (1 - ETA) * (re_i + re_o) / (ETA * re_o - re_i)
    assert math.isclose(r_om_back, R_OMEGA, rel_tol=1e-12), r_om_back
    assert math.isclose(re_o / re_i, RE_O_RATIO, rel_tol=1e-9), re_o / re_i
    mu = ETA * re_o / re_i
    assert math.isclose(mu, MU, rel_tol=1e-9), mu
    re_s = 2 * abs(ETA * re_o - re_i) / (1 + ETA)
    assert math.isclose(re_s, RE_S_RATIO * re_i, rel_tol=1e-9), re_s

    # Circular-Couette wall values: U_theta(r1) = 1, U_theta(r2) = ratio.
    a0, b0 = derived_params.ccf_A, derived_params.ccf_B
    r1, r2 = derived_params.r_inner, derived_params.r_outer
    assert math.isclose(a0 * r1 + b0 / r1, 1.0, rel_tol=1e-12)
    assert math.isclose(a0 * r2 + b0 / r2, RE_O_RATIO, rel_tol=1e-9)

    # Local exponent q(r) = 2 B0 / (A0 r^2 + B0), monotone in (0, 2).
    q1 = 2 * b0 / (a0 * r1**2 + b0)
    q2 = 2 * b0 / (a0 * r2**2 + b0)
    assert math.isclose(q1, Q_INNER, rel_tol=1e-3), q1
    assert math.isclose(q2, Q_OUTER, rel_tol=1e-3), q2
    assert 0 < q2 < q1 < 2, (q2, q1)
    print("case-ok")


def case_resume() -> None:
    """A consistent externally-supplied re2 is accepted, unchanged."""
    from dnsjax.parameters import (
        Parameters,
        params,
        update_parameters,
        validate_parameters,
    )

    re_i = 1.0e4
    re_o = _re2(re_i, ETA, R_OMEGA)
    update_parameters(
        Parameters(
            phys={
                "system": "quasi-keplerian",
                "re1": re_i,
                "r_omega": R_OMEGA,
                "re2": re_o,  # the snapshot-resume replay path
            },
            geo={"eta": ETA},
        )
    )
    validate_parameters()
    assert params.phys.re2 == re_o, "consistent re2 must be preserved"
    print("case-ok")


def _expect_valueerror(phys: dict, geo: dict, needle: str) -> None:
    """update_parameters(...) + validate must raise, message ~ needle."""
    from dnsjax.parameters import (
        Parameters,
        update_parameters,
        validate_parameters,
    )

    try:
        update_parameters(Parameters(phys=phys, geo=geo))
        validate_parameters()
    except ValueError as exc:
        assert needle in str(exc), f"{needle!r} not in: {exc}"
        return
    raise AssertionError(f"expected ValueError mentioning {needle!r}")


def case_err_romega() -> None:
    """Missing r_omega and the R_Omega >= -1 (non-quasi-Keplerian)."""
    # Missing r_omega (re1 set, so the r_omega check is the one to fire).
    _expect_valueerror(
        {"system": "quasi-keplerian", "re1": 1.0e4}, {"eta": ETA}, "r_omega"
    )
    # Rayleigh line and co-rotating side both rejected.
    for r_om in (-1.0, -0.5, 0.5):
        _expect_valueerror(
            {"system": "quasi-keplerian", "re1": 1.0e4, "r_omega": r_om},
            {"eta": ETA},
            "Rayleigh",
        )
    print("case-ok")


def case_err_re1() -> None:
    """The re1 sign-convention rejections (None, 0, negative)."""
    # re1 = None (omitted): fresh params, so genuinely unset.
    _expect_valueerror(
        {"system": "quasi-keplerian", "r_omega": R_OMEGA}, {"eta": ETA}, "re1"
    )
    for re_i in (0.0, -5.0):
        _expect_valueerror(
            {"system": "quasi-keplerian", "re1": re_i, "r_omega": R_OMEGA},
            {"eta": ETA},
            "re1",
        )
    print("case-ok")


def case_override_re2() -> None:
    """A conflicting externally-supplied re2 is silently overwritten."""
    from dnsjax.parameters import (
        Parameters,
        params,
        update_parameters,
        validate_parameters,
    )

    re_i = 1.0e4
    update_parameters(
        Parameters(
            phys={
                "system": "quasi-keplerian",
                "re1": re_i,
                "r_omega": R_OMEGA,
                "re2": 123.0,  # conflicting; the derivation wins
            },
            geo={"eta": ETA},
        )
    )
    validate_parameters()
    expected = _re2(re_i, ETA, R_OMEGA)
    assert params.phys.re2 == expected, (
        "conflicting re2 must be overwritten by the derivation",
        params.phys.re2,
        expected,
    )
    print("case-ok")


CASES = {
    "derive": case_derive,
    "resume": case_resume,
    "err_romega": case_err_romega,
    "err_re1": case_err_re1,
    "override_re2": case_override_re2,
}


def _run_case(name: str) -> None:
    """Run one subprocess case and check its stdout for the marker."""
    result = run_live(
        [sys.executable, __file__, "--case", name],
        timeout=900,
    )
    if result.returncode != 0 or "case-ok" not in result.stdout:
        print(result.stdout[-2000:] if result.stdout else "(no stdout)")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        raise AssertionError(f"case {name}: exit {result.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=sorted(CASES), default=None)
    args = parser.parse_args()
    if args.case:
        CASES[args.case]()
        sys.exit(0)

    failed = 0
    for name in sorted(CASES):
        try:
            _run_case(name)
            print(f"  PASS  {name}")
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
    if failed:
        print(f"\n{failed} case(s) FAILED")
        sys.exit(1)
    print("\nAll quasi-keplerian cases passed.")
