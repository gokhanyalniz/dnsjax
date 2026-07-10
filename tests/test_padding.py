r"""Padded-size rounding tests (mesh divisibility + even-pad parity).

Covers :func:`dnsjax.parameters.round_up_padded` and its two
application sites: ``PaddedResolution.set_padded_resolution`` (the
primary site -- ``params.dist`` is final at every production call) and
the idempotent ``padded_res.apply_rounding`` fallback at
:mod:`dnsjax.sharding` import (entry points that set ``params.dist``
after -- or without -- ``set_padded_resolution``).  Singleton-dependent
cases run in subprocesses (the ``test_*`` subprocess-per-config idiom:
sharding/geometry singletons capture ``params`` at import time), and
each asserts its rounding diagnostic is printed exactly once.

1. Unit: ``round_up_padded`` -- divisibility, parity (including the
   ``divisor = 1`` parity-rescue), the odd-divisor double step, no-op,
   and the impossible corner (even divisor, odd source) raising.
2. ``primary``: nz = 6 with np1 = 2 on 2 forced host CPU devices --
   ``set_padded_resolution`` rounds ``nz_padded`` 9 -> 10 *before* the
   sharding import; ``sharding.phys_shape`` picks it up and a
   spectral <-> physical FFT round-trip runs on the padded grid.
3. ``fallback``: ``params.dist.np1`` assigned *after*
   ``set_padded_resolution`` (the direct-assignment idiom of
   ``test_banded_solver_sharded``) -- the sharding-import fallback
   rounds 10 -> 12 for np1 = 4 and records the note.
4. ``rescue``: single device, nz = 6 -- previously the trace-time
   ``"Difference (n - N) = 3 cannot be odd"`` failure; now rounds
   9 -> 10 and the FFT round-trip runs.

Usage::

    uv run python tests/test_padding.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def test_round_up_padded() -> None:
    """Unit-check the rounding helper (JAX-free)."""
    from dnsjax.parameters import round_up_padded

    cases = [
        ((9, 6, 2), 10),  # divisibility + parity (np1 = 2)
        ((9, 6, 1), 10),  # parity-rescue on a single device
        ((12, 8, 2), 12),  # already valid: no-op
        ((24, 16, 5), 30),  # odd divisor: parity forces a second step
        ((192, 128, 1), 192),  # the defaults are untouched
        ((16, 16, 4), 16),  # zero pad is even
    ]
    for (n_padded, n_source, divisor), expected in cases:
        got = round_up_padded(n_padded, n_source, divisor)
        if got != expected:
            raise AssertionError(
                f"round_up_padded{(n_padded, n_source, divisor)} = "
                f"{got}, expected {expected}"
            )
    try:
        round_up_padded(26, 17, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("even divisor + odd source must raise")


# ── subprocess cases (singletons captured at import time) ───────────


def _fft_round_trip(sharding) -> None:
    """Zero-field spec -> phys -> spec round-trip on the padded grid.

    Shape-driven: exercises ``zeropad_fft`` / ``truncate_fft`` with the
    rounded ``nz_padded`` -- the exact path that raised
    ``"Difference (n - N) ... cannot be odd"`` before the rounding.
    """
    from jax import numpy as jnp

    from dnsjax.operators import phys_to_spec_2d, spec_to_phys_2d

    spec = jnp.zeros(
        (3, *sharding.spec_shape),
        dtype=sharding.complex_type,
        out_sharding=sharding.spec_vector_shard,
    )
    phys = spec_to_phys_2d(spec)
    if phys.shape[2] != sharding.phys_shape[1]:
        raise AssertionError(f"physical z size {phys.shape}")
    back = phys_to_spec_2d(phys)
    if back.shape != spec.shape:
        raise AssertionError(f"round-trip shape {back.shape}")
    if float(jnp.abs(back).max()) != 0.0:
        raise AssertionError("round-trip of zeros not zero")


def case_primary() -> None:
    """np1 = 2, nz = 6: primary rounding before the sharding import."""
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    update_parameters(
        Parameters(
            phys={"system": "plane-couette"},
            res={"nx": 4, "ny": 9, "nz": 6},
            dist={"np0": 1, "np1": 2},
        )
    )
    validate_parameters()
    padded_res.set_padded_resolution(params)
    assert padded_res.nz_padded == 10, padded_res.nz_padded
    assert len(padded_res.notes) == 1, padded_res.notes

    from dnsjax.sharding import sharding

    assert sharding.phys_shape == (9, 10, 6), sharding.phys_shape
    _fft_round_trip(sharding)
    print("case-ok")


def case_fallback() -> None:
    """np1 set after ``set_padded_resolution``: the sharding fallback."""
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    update_parameters(
        Parameters(
            phys={"system": "plane-couette"},
            res={"nx": 4, "ny": 9, "nz": 6},
        )
    )
    padded_res.set_padded_resolution(params)
    assert padded_res.nz_padded == 10  # parity-rescue at np = 1

    # The direct-assignment idiom: the mesh axes change *after*
    # ``padded_res`` was computed; the sharding import must re-round.
    params.dist.np1 = 4
    from dnsjax.sharding import sharding

    assert padded_res.nz_padded == 12, padded_res.nz_padded
    assert sharding.phys_shape == (9, 12, 6), sharding.phys_shape
    print("case-ok")


def case_rescue() -> None:
    """Single device, nz = 6: the old odd-pad failure now runs."""
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    update_parameters(
        Parameters(
            phys={"system": "plane-couette"},
            res={"nx": 4, "ny": 9, "nz": 6},
        )
    )
    padded_res.set_padded_resolution(params)
    assert padded_res.nz_padded == 10, padded_res.nz_padded

    from dnsjax.sharding import sharding

    _fft_round_trip(sharding)
    print("case-ok")


CASES = {
    "primary": case_primary,
    "fallback": case_fallback,
    "rescue": case_rescue,
}
# The rounding diagnostic each case must print exactly once (the
# fallback's stdout also carries the primary "9 to 10" line -- both
# notes are reported at the sharding import).
EXPECT = {
    "primary": "nz_padded rounded from 9 to 10",
    "fallback": "nz_padded rounded from 10 to 12",
    "rescue": "nz_padded rounded from 9 to 10",
}


def _run_case(name: str) -> None:
    """Run one subprocess case and check its stdout."""
    result = subprocess.run(
        [sys.executable, __file__, "--case", name],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:] if result.stdout else "(no stdout)")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        raise AssertionError(f"case {name}: exit {result.returncode}")
    if "case-ok" not in result.stdout:
        raise AssertionError(f"case {name}: missing case-ok marker")
    n_notes = result.stdout.count(EXPECT[name])
    if n_notes != 1:
        raise AssertionError(
            f"case {name}: {EXPECT[name]!r} printed {n_notes} times "
            f"(expected once):\n{result.stdout}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=sorted(CASES), default=None)
    args = parser.parse_args()
    if args.case:
        CASES[args.case]()
        sys.exit(0)

    tests: list[tuple[str, object]] = [
        ("round_up_padded units", test_round_up_padded)
    ]
    tests += [
        (f"subprocess case {n}", lambda n=n: _run_case(n))
        for n in sorted(CASES)
    ]

    failed = 0
    for label, fn in tests:
        try:
            fn()
            print(f"  PASS  {label}")
        except AssertionError as exc:
            print(f"  FAIL  {label}: {exc}")
            failed += 1

    print(f"\n{len(tests) - failed} passed, {failed} failed.")
    sys.exit(1 if failed else 0)
