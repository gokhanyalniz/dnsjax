r"""Azimuthal-wedge tests (``geo.m0``).

Offline checks of the annular / cylindrical azimuthal wedge, the
``geo.m0`` support in the annular / cylindrical ``Fourier`` classes.
Each case runs in its own subprocess (the ``test_*``
subprocess-per-config idiom: ``dnsjax.parameters`` -- and, for the
wedge-Fourier cases, the sharding / geometry singletons -- capture the
global ``params`` once).

Cases:

1. ``err_m0``: the wedge rejected on a non-annular / non-cylindrical
   system.
2. ``wedge_annular``: m0 = 2 annular Fourier (Taylor-Couette) -- lz =
   pi, the m0-scaled integer azimuthal wavenumbers, the lz-based CFL
   azimuthal spacing, and the unique mean mode.
3. ``wedge_cylindrical``: m0 = 3 pipe Fourier -- the m0-scaled
   wavenumbers and the parity mask ``m_is_even`` tracking the *physical*
   wavenumber parity (the r = 0 axis-regularity condition).
4. ``wedge_nonlinear``: the wedge's **physical-space / nonlinear**
   path, which no other wedge check reaches (the laminar smoke test has
   u' = 0; the transient-growth wedge equivalence is FFT-free).  Steps
   one physical Taylor-Couette field two ways -- as an m0 = 3 wedge at
   nz = 8 and as the full circle at nz = 24 with the coefficients
   re-indexed onto the m0-multiples -- through a full nonlinear step,
   and requires (a) the m0-multiples agree and (b) nothing leaks to the
   other m.  This pins that the inverse FFT of the m = m0*j coefficients
   yields ``nz`` points spanning the *wedge* [0, 2*pi/m0) rather than
   the whole azimuth: were it the latter, the pseudo-spectral product
   would be evaluated on an m0-times too coarse theta grid and (a)
   would fail outright.  Runs two subprocess arms sharing a state file.

Usage::

    uv run python tests/test_wedge.py
"""

from __future__ import annotations

import argparse
import math
import sys

from _live import run_live

sys.stdout.reconfigure(line_buffering=True)

ETA = 0.71


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


def case_err_m0() -> None:
    """The azimuthal wedge is rejected on a non-annular geometry."""
    _expect_valueerror({"system": "plane-couette"}, {"m0": 2}, "wedge")
    print("case-ok")


def case_wedge_annular() -> None:
    """m0 = 2 annular Fourier: lz, m scaling, CFL, mean mode."""
    import numpy as np

    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.harmonics import complex_harmonics
    from dnsjax.parameters import (
        Parameters,
        params,
        update_parameters,
        validate_parameters,
    )

    tc = {"system": "taylor-couette", "re1": 100.0, "re2": 80.0}
    update_parameters(
        Parameters(
            phys=tc,
            geo={"eta": ETA, "m0": 2},
            res={"nx": 4, "nz": 8, "ny": 11},
        )
    )
    validate_parameters()
    configure_jax_platform(platform_from_argv())
    from dnsjax.geometries.wall_bounded.annular import AnnularFlow, fourier

    assert math.isclose(params.geo.lz, math.pi), params.geo.lz
    m_true = np.asarray(fourier.m).ravel()[:7]
    expected = 2 * np.asarray(complex_harmonics(8))
    assert np.array_equal(m_true, expected), (m_true, expected)

    flow = AnnularFlow()
    cfl_th = np.asarray(flow.cfl_inv_spacing)[2, :11, 0, 0]
    expect_cfl = np.asarray(flow.inv_r) * params.res.nz / params.geo.lz
    assert np.allclose(cfl_th, expect_cfl), "CFL_th must use lz = 2*pi/m0"
    assert int(np.asarray(fourier.mean_mask).sum()) == 1, "unique mean mode"
    print("case-ok")


def case_wedge_cylindrical() -> None:
    """m0 = 3 pipe Fourier: m scaling + physical-parity mask."""
    import numpy as np

    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.harmonics import complex_harmonics
    from dnsjax.parameters import (
        Parameters,
        params,
        update_parameters,
        validate_parameters,
    )

    update_parameters(
        Parameters(
            phys={"system": "pipe", "re": 100.0},
            geo={"m0": 3},
            res={"nx": 4, "nz": 8, "ny": 11},
        )
    )
    validate_parameters()
    configure_jax_platform(platform_from_argv())
    from dnsjax.geometries.wall_bounded.cylindrical import fourier

    assert math.isclose(params.geo.lz, 2 * math.pi / 3), params.geo.lz
    m_true = np.asarray(fourier.m).ravel()[:7]
    expected = 3 * np.asarray(complex_harmonics(8))
    assert np.array_equal(m_true, expected), (m_true, expected)
    # m_is_even must track the *physical* wavenumber parity (r=0
    # regularity), not the raw harmonic index.
    par = np.asarray(fourier.m_is_even).ravel()[:7].astype(int)
    assert np.array_equal(par, (np.abs(expected) % 2 == 0).astype(int)), par
    print("case-ok")


# ── nonlinear wedge-vs-full-circle equivalence ──────────────────
#
# The wedge's *physical-space* path.  The FFT is purely index-based
# (it never sees theta), so the inverse transform of the m = m0*j
# coefficients yields ``nz_padded`` points spanning one period of the
# m0-periodic field -- i.e. the wedge [0, 2*pi/m0), at m0-times finer
# spacing than the full circle at the same nz -- and *not* nz_padded
# points spread over the whole azimuth.  Everything below rides on
# that: if it were false, the pseudo-spectral product would be
# evaluated on the wrong grid and the wedge would silently disagree
# with the full circle.
#
# Neither existing wedge check reaches this path: the laminar smoke
# test has u' = 0 (no nonlinear term) and the transient-growth wedge
# equivalence is FFT-free (per-mode linear propagator).  So step the
# *same physical field* two ways -- as an m0 wedge at nz = NZ_WEDGE,
# and as the full circle at nz = m0*NZ_WEDGE with the coefficients
# re-indexed onto the m0-multiples -- through a full nonlinear step
# and require they agree.  Agreement is to round-off, not bit-exact:
# the two runs use different FFT lengths.

WEDGE_M0 = 3
NZ_WEDGE = 8
WEDGE_NX = 6
WEDGE_NY = 15
WEDGE_TOL = 1e-11


def _wedge_common(m0: int, nz: int) -> None:
    """Configure the shared Taylor-Couette wedge/full-circle run."""
    from dnsjax.parameters import Parameters, update_parameters

    update_parameters(
        Parameters(
            phys={"system": "taylor-couette", "re1": 200.0, "re2": 160.0},
            geo={"eta": ETA, "m0": m0, "lx": 2.0},
            res={"nx": WEDGE_NX, "nz": nz, "ny": WEDGE_NY},
            # A tight corrector: the fixed point is only pinned to
            # ``corrector_tolerance``, so it -- not the discretization
            # -- would otherwise floor the two arms' agreement.
            step={
                "dt": 0.005,
                "corrector_tolerance": 1e-14,
                "max_corrector_iterations": 100,
            },
        )
    )


def _azimuthal_index_map(m0: int, nz_wedge: int, nz_full: int):
    """Map a wedge azimuthal index to the full-circle one at the same ``m``.

    Both axes store ``complex_harmonics`` order (Nyquist omitted), so
    the map is looked up by wavenumber value rather than assumed.
    """
    import numpy as np

    from dnsjax.harmonics import complex_harmonics

    h_wedge = complex_harmonics(nz_wedge)
    h_full = complex_harmonics(nz_full)
    idx = []
    for h in h_wedge:
        hit = np.flatnonzero(h_full == m0 * h)
        assert hit.size == 1, f"m = {m0 * h} not representable in full circle"
        idx.append(int(hit[0]))
    return np.asarray(idx)


def _wedge_step(state):
    """One full nonlinear step of the configured Taylor-Couette flow.

    *state* is physical, as the IC builders produce it; the crossing
    into and out of the solver basis mirrors ``__main__``'s, so both
    arms of the comparison stay in physical components.
    """
    from dnsjax.flows.wall_bounded.taylor_couette import (
        from_solver_basis,
        predict_and_fully_correct,
        to_solver_basis,
    )

    out, err, *_ = predict_and_fully_correct(to_solver_basis(state))
    assert float(err) < 1e-13, f"corrector did not converge: err={float(err)}"
    return from_solver_basis(out)


def case_wedge_nonlinear_wedge() -> None:
    """Wedge arm: build a random state, step it, save both."""
    import numpy as np

    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv

    _wedge_common(WEDGE_M0, NZ_WEDGE)
    from dnsjax.parameters import validate_parameters

    validate_parameters()
    configure_jax_platform(platform_from_argv(), double_precision=True)
    from dnsjax.ic.random_field import generate_random_state

    state = generate_random_state(0.2, 0.6, 0.4, 0.14, seed=17)
    state_in = np.asarray(state)  # host copy: the step donates ``state``
    out = _wedge_step(state)
    np.savez(
        sys.argv[sys.argv.index("--state-file") + 1],
        state_in=state_in,
        state_out=np.asarray(out),
    )
    print("case-ok")


def case_wedge_nonlinear_full() -> None:
    """Full-circle arm: embed the wedge state, step, compare."""
    import numpy as np

    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv

    nz_full = WEDGE_M0 * NZ_WEDGE
    _wedge_common(1, nz_full)
    from dnsjax.parameters import validate_parameters

    validate_parameters()
    configure_jax_platform(platform_from_argv(), double_precision=True)
    import jax
    from jax.sharding import NamedSharding

    from dnsjax.sharding import sharding

    ref = np.load(sys.argv[sys.argv.index("--state-file") + 1])
    wedge_in, wedge_out = ref["state_in"], ref["state_out"]
    idx = _azimuthal_index_map(WEDGE_M0, NZ_WEDGE, nz_full)

    # Embed: the same physical field, re-indexed onto m = m0*j.
    # ``spec_shape`` is the per-component (ny, m, k_ax) shape.
    full_in = np.zeros(
        (wedge_in.shape[0], *sharding.spec_shape), dtype=wedge_in.dtype
    )
    full_in[:, :, idx, :] = wedge_in[:, :, : len(idx), :]
    out = np.asarray(
        _wedge_step(
            jax.device_put(
                full_in,
                NamedSharding(sharding.mesh, sharding.spec_vector_shard),
            )
        )
    )

    scale = np.max(np.abs(wedge_out))
    assert scale > 1e-6, f"degenerate reference state (scale {scale:.2e})"

    # 1. The m0-multiples reproduce the wedge step exactly.
    err = np.max(np.abs(out[:, :, idx, :] - wedge_out[:, :, : len(idx), :]))
    assert err / scale < WEDGE_TOL, (
        f"wedge and full circle disagree on the m0-multiples: "
        f"rel err {err / scale:.3e} (a wedge resolved on the *whole* "
        f"azimuth instead of [0, 2*pi/m0) would fail here)"
    )

    # 2. The wedge subspace is closed: nothing leaks to other m.
    mask = np.ones(out.shape[2], dtype=bool)
    mask[idx] = False
    leak = np.max(np.abs(out[:, :, mask, :])) if mask.any() else 0.0
    assert leak / scale < WEDGE_TOL, (
        f"nonlinear term leaked out of the m = 0 mod {WEDGE_M0} "
        f"subspace: rel {leak / scale:.3e}"
    )
    print(f"  wedge==full rel err {err / scale:.2e}, leak {leak / scale:.2e}")
    print("case-ok")


def case_wedge_nonlinear() -> None:
    """Orchestrate the two arms through a shared state file."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        f = str(Path(d) / "wedge.npz")
        for arm in ("wedge_nonlinear_wedge", "wedge_nonlinear_full"):
            r = run_live(
                [sys.executable, __file__, "--case", arm, "--state-file", f],
                timeout=600,
            )
            if r.returncode != 0 or "case-ok" not in r.stdout:
                print(r.stdout[-2000:], r.stderr[-2000:])
                raise AssertionError(f"arm {arm}: exit {r.returncode}")
    print("case-ok")


CASES = {
    "err_m0": case_err_m0,
    "wedge_annular": case_wedge_annular,
    "wedge_cylindrical": case_wedge_cylindrical,
    "wedge_nonlinear": case_wedge_nonlinear,
}
# Arms of ``wedge_nonlinear``; not run standalone (they need
# --state-file and must run in the documented order).
_ARMS = {
    "wedge_nonlinear_wedge": case_wedge_nonlinear_wedge,
    "wedge_nonlinear_full": case_wedge_nonlinear_full,
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
    parser.add_argument(
        "--case", choices=sorted(CASES | _ARMS.keys()), default=None
    )
    parser.add_argument(
        "--state-file", default=None, help="wedge_nonlinear arms only"
    )
    args = parser.parse_args()
    if args.case:
        (CASES | _ARMS)[args.case]()
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
    print("\nAll wedge cases passed.")
