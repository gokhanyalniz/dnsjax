r"""Padded-size rounding tests (mesh divisibility, FFT-friendly
7-smooth lengths) and odd-pad FFTs.

Covers :func:`dnsjax.parameters.round_up_padded` /
:func:`round_up_padded_smooth` and their two application sites:
``PaddedResolution.set_padded_resolution`` (the primary site --
``params.dist`` is final at every production call) and the idempotent
``padded_res.apply_rounding`` fallback at :mod:`dnsjax.sharding`
import (entry points that set ``params.dist`` after -- or without --
``set_padded_resolution``), plus the parity-free ``zeropad_fft`` /
``truncate_fft`` mode placement those sizes feed.
Singleton-dependent cases run in subprocesses (the
``test_*`` subprocess-per-config idiom: sharding/geometry singletons
capture ``params`` at import time); each asserts its rounding
diagnostic is printed exactly once, or that none appears where the
natural padded size must pass through unrounded.  The two exactness
cases (4 and 6) also pin :func:`dnsjax.fft.chunked_transform`
bit-exactly against the fused batch (``k = 2`` and ``k = 7`` over a
6-field stack), single-device and on the (2, 2) mesh (per-chunk
reshards + the component-axis concatenate of sharded outputs).

1. Unit: ``round_up_padded`` -- divisibility, no-op, and the
   ``divisor <= 1`` passthrough.
2. ``primary``: nz = 6 with np1 = 2 on 2 forced host CPU devices --
   ``set_padded_resolution`` rounds ``nz_padded`` 9 -> 10 *before* the
   sharding import; ``sharding.phys_shape`` picks it up and a
   spectral <-> physical FFT round-trip runs on the padded grid.
3. ``fallback``: ``params.dist.np1`` assigned *after*
   ``set_padded_resolution`` (the direct-assignment idiom of
   ``test_banded_solver_sharded``) -- the sharding-import fallback
   rounds 9 -> 12 for np1 = 4 and records the note.
4. ``odd_pad``: single device, nz = 6 -- ``nz_padded`` stays at the
   natural 9 (an odd dealiasing pad, no rounding note); a random
   Hermitian field round-trips exactly and a single ``k_z = 1`` mode
   lands on the analytic physical profile (mode placement is
   parity-free).
5. ``odd_nz``: single device, nz = 5 -- ``truncate_fft`` keeps
   ``(n - 1) - n // 2`` negative modes, matching the ``n - 1`` stored
   modes of ``harmonics.complex_harmonics`` for odd *n* (formerly a
   shape mismatch); exactness checks as in ``odd_pad``.
6. ``spec_pad``: forced (2, 2) host-CPU mesh with nx = 6 / nz = 4 --
   both spectral divisibility pads engaged, driving the
   ``pad``/``strip`` arguments fused into ``truncate_*`` /
   ``zeropad_*`` (no rounding note; the spec-pad diagnostics are
   expected), with the exactness checks sharded via ``device_put``.
7. ``smooth``: single device, nz = 94 -- the natural pad
   ``141 = 3 * 47`` (a slow generic-radix FFT length) is bumped to
   the 7-smooth 144 with the FFT-friendly note, and the exactness
   checks confirm the extra slots stay out of the physical field.

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
        ((9, 2), 10),  # divisibility (np1 = 2)
        ((9, 1), 9),  # single device: no rounding
        ((12, 2), 12),  # already valid: no-op
        ((24, 5), 25),  # odd divisor
        ((192, 1), 192),  # the defaults are untouched
        ((16, 4), 16),  # already a multiple
    ]
    for (n_padded, divisor), expected in cases:
        got = round_up_padded(n_padded, divisor)
        if got != expected:
            raise AssertionError(
                f"round_up_padded{(n_padded, divisor)} = "
                f"{got}, expected {expected}"
            )


def test_round_up_padded_smooth() -> None:
    """Unit-check the 7-smooth rounding helper (JAX-free)."""
    from dnsjax.parameters import is_fft_smooth, round_up_padded_smooth

    smooth = [1, 2, 6, 7, 9, 10, 144, 147, 150, 2 * 3 * 5 * 7]
    rough = [-3, 0, 11, 13, 141, 142, 143, 146]  # 141=3*47, 142=2*71
    for n in smooth:
        if not is_fft_smooth(n):
            raise AssertionError(f"is_fft_smooth({n}) should be True")
    for n in rough:
        if is_fft_smooth(n):
            raise AssertionError(f"is_fft_smooth({n}) should be False")

    cases = [
        ((141, 1), 144),  # 3*47 -> 2^4 * 3^2
        ((141, 2), 144),  # divisibility (142 = 2*71) + smoothness
        ((150, 1), 150),  # already smooth: no-op
        ((9, 2), 10),  # divisibility alone lands smooth (primary case)
        ((9, 4), 12),  # the fallback case's rounding is unchanged
        ((143, 11), 143),  # non-smooth divisor: plain rounding only
        ((192, 1), 192),  # the defaults are untouched
    ]
    for (n_padded, divisor), expected in cases:
        got = round_up_padded_smooth(n_padded, divisor)
        if got != expected:
            raise AssertionError(
                f"round_up_padded_smooth{(n_padded, divisor)} = "
                f"{got}, expected {expected}"
            )


# ── subprocess cases (singletons captured at import time) ───────────


def _fft_round_trip(sharding) -> None:
    """Zero-field spec -> phys -> spec round-trip on the padded grid.

    Shape-driven: exercises ``zeropad_fft`` / ``truncate_fft`` with
    the padded ``nz_padded`` under the mesh the case forces.
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


def _hermitian_random_spec(sharding):
    """Random spectral field that a real physical field can carry.

    Free on ``kx > 0`` columns (the real FFT implies the conjugate
    half); Hermitian-paired in kz on the ``kx = 0`` plane
    (``c(-kz) = conj(c(kz))``, real mean), with unpaired kz modes
    (the odd-``nz`` band edge) and divisibility-padding slots zeroed.
    Any such field is exactly representable on the padded grid, so
    spec -> phys -> spec must be the identity.
    """
    import numpy as np

    from dnsjax.harmonics import complex_harmonics
    from dnsjax.parameters import params

    rng = np.random.default_rng(7)
    shape = (3, *sharding.spec_shape)
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    kz = list(complex_harmonics(params.res.nz))
    a[:, :, len(kz) :, :] = 0.0  # kz divisibility-padding slots
    a[:, :, :, params.res.nx // 2 :] = 0.0  # kx padding slots
    for i, q in enumerate(kz):
        if q == 0:
            a[:, :, i, 0] = a[:, :, i, 0].real
        elif -q not in kz:
            a[:, :, i, 0] = 0.0  # unpaired band edge (odd nz)
        elif q > 0:
            a[:, :, kz.index(-q), 0] = np.conj(a[:, :, i, 0])
    return a


def _fft_exactness(sharding) -> None:
    r"""Identity + analytic mode-placement checks.

    A random Hermitian field must survive spec -> phys -> spec
    unchanged, and a single `$k_z = 1$` mode must land on
    `$2 \cos(2 \pi j / n_{z,\mathrm{padded}})$` on the padded
    physical grid -- an absolute-placement check, so a
    self-consistent slot swap in the pad/truncate pair cannot pass.
    """
    import jax
    import numpy as np
    from jax import numpy as jnp
    from jax.sharding import NamedSharding

    from dnsjax.harmonics import complex_harmonics
    from dnsjax.operators import phys_to_spec_2d, spec_to_phys_2d
    from dnsjax.parameters import padded_res, params

    shard = NamedSharding(sharding.mesh, sharding.spec_vector_shard)
    spec = jax.device_put(
        _hermitian_random_spec(sharding).astype(sharding.complex_type),
        shard,
    )
    back = phys_to_spec_2d(spec_to_phys_2d(spec))
    err = float(jnp.abs(back - spec).max())
    if err > 1e-12:
        raise AssertionError(f"Hermitian round-trip error {err:.2e}")

    kz = list(complex_harmonics(params.res.nz))
    one = np.zeros((3, *sharding.spec_shape), dtype=sharding.complex_type)
    one[0, :, kz.index(1), 0] = 1.0
    one[0, :, kz.index(-1), 0] = 1.0
    phys = spec_to_phys_2d(jax.device_put(one, shard))
    grid = np.arange(padded_res.nz_padded) / padded_res.nz_padded
    expected = 2.0 * np.cos(2.0 * np.pi * grid)
    err = float(np.abs(np.asarray(phys[0]) - expected[None, :, None]).max())
    if err > 1e-12:
        raise AssertionError(f"mode-placement error {err:.2e}")


def _chunked_parity(sharding) -> None:
    """``chunked_transform`` reproduces the fused batch bit-exactly.

    A 6-field random spectral stack (the ``get_nonlin`` batch size)
    through the batched inverse transform, at ``k = 2`` (balanced
    velocity/vorticity split) and ``k = 7`` (> the field count:
    empty groups skipped, per-field degradation).  Per-field
    transforms are independent, so any difference is a chunking bug.
    """
    import jax
    import numpy as np
    from jax.sharding import NamedSharding

    from dnsjax.fft import chunked_transform
    from dnsjax.operators import spec_to_phys_2d

    rng = np.random.default_rng(11)
    shape = (6, *sharding.spec_shape)
    stack = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    stack = jax.device_put(
        stack.astype(sharding.complex_type),
        NamedSharding(sharding.mesh, sharding.spec_vector_shard),
    )
    fused = np.asarray(spec_to_phys_2d(stack))
    for k in (2, 7):
        chunked = np.asarray(chunked_transform(spec_to_phys_2d, stack, k))
        if not np.array_equal(chunked, fused):
            err = float(np.abs(chunked - fused).max())
            raise AssertionError(
                f"chunked (k={k}) != fused batch (max err {err:.2e})"
            )


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
    assert padded_res.nz_padded == 9  # no rounding at np = 1

    # The direct-assignment idiom: the mesh axes change *after*
    # ``padded_res`` was computed; the sharding import must re-round.
    params.dist.np1 = 4
    from dnsjax.sharding import sharding

    assert padded_res.nz_padded == 12, padded_res.nz_padded
    assert sharding.phys_shape == (9, 12, 6), sharding.phys_shape
    print("case-ok")


def case_odd_pad() -> None:
    """Single device, nz = 6: the natural odd pad (9) runs unrounded."""
    from dnsjax.bootstrap import configure_jax_platform
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
    assert padded_res.nz_padded == 9, padded_res.nz_padded
    assert padded_res.notes == [], padded_res.notes

    configure_jax_platform("cpu")  # x64 for the exactness thresholds
    from dnsjax.sharding import sharding

    assert sharding.phys_shape == (9, 9, 6), sharding.phys_shape
    _fft_round_trip(sharding)
    _fft_exactness(sharding)
    _chunked_parity(sharding)
    print("case-ok")


def case_odd_nz() -> None:
    """Single device, nz = 5: the odd-``nz`` band round-trips exactly."""
    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    update_parameters(
        Parameters(
            phys={"system": "plane-couette"},
            res={"nx": 4, "ny": 9, "nz": 5},
        )
    )
    padded_res.set_padded_resolution(params)
    assert padded_res.nz_padded == 7, padded_res.nz_padded
    assert padded_res.notes == [], padded_res.notes

    configure_jax_platform("cpu")  # x64 for the exactness thresholds
    from dnsjax.sharding import sharding

    assert sharding.spec_shape == (9, 4, 2), sharding.spec_shape
    _fft_round_trip(sharding)
    _fft_exactness(sharding)
    print("case-ok")


def case_spec_pad() -> None:
    """(2, 2) mesh, nx = 6 / nz = 4: both fused spec pads engaged.

    ``nz - 1 = 3`` and ``nx // 2 = 3`` are odd, so both spectral axes
    carry one divisibility-padding mode -- the ``pad``/``strip``
    arguments fused into ``truncate_*`` / ``zeropad_*`` are exercised
    end to end, with the exactness checks confirming the padded slots
    stay out of the physical field.
    """
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    update_parameters(
        Parameters(
            phys={"system": "plane-couette"},
            res={"nx": 6, "ny": 8, "nz": 4},
            dist={"np0": 2, "np1": 2},
        )
    )
    padded_res.set_padded_resolution(params)
    assert padded_res.nz_padded == 6, padded_res.nz_padded
    assert padded_res.notes == [], padded_res.notes

    configure_jax_platform("cpu")  # x64 for the exactness thresholds
    from dnsjax.sharding import sharding

    assert sharding.nz_spec_pad == 1, sharding.nz_spec_pad
    assert sharding.nx_spec_pad == 1, sharding.nx_spec_pad
    _fft_round_trip(sharding)
    _fft_exactness(sharding)
    _chunked_parity(sharding)
    print("case-ok")


def case_smooth() -> None:
    """Single device, nz = 94: the non-smooth natural pad is bumped.

    ``3 * 94 // 2 = 141 = 3 * 47`` would hit the slow generic-radix
    FFT path; ``apply_rounding`` bumps it to the next 7-smooth length
    (144, no divisibility involved on a single device), records the
    FFT-friendly note, and the exactness checks confirm the extra
    slots stay out of the physical field.
    """
    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        update_parameters,
    )

    update_parameters(
        Parameters(
            phys={"system": "plane-couette"},
            res={"nx": 4, "ny": 9, "nz": 94},
        )
    )
    padded_res.set_padded_resolution(params)
    assert padded_res.nz_padded == 144, padded_res.nz_padded

    configure_jax_platform("cpu")  # x64 for the exactness thresholds
    from dnsjax.sharding import sharding

    assert sharding.phys_shape == (9, 144, 6), sharding.phys_shape
    _fft_round_trip(sharding)
    _fft_exactness(sharding)
    print("case-ok")


CASES = {
    "primary": case_primary,
    "fallback": case_fallback,
    "odd_pad": case_odd_pad,
    "odd_nz": case_odd_nz,
    "spec_pad": case_spec_pad,
    "smooth": case_smooth,
}
# The rounding diagnostic each case must print exactly once; ``None``
# marks cases whose natural padded size must pass through unrounded
# (no rounding note at all).
EXPECT: dict[str, str | None] = {
    "primary": "nz_padded rounded from 9 to 10 (np1 divisibility).",
    "fallback": "nz_padded rounded from 9 to 12 (np1 divisibility).",
    "odd_pad": None,
    "odd_nz": None,
    "spec_pad": None,
    "smooth": "nz_padded rounded from 141 to 144 (FFT-friendly size).",
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
    expect = EXPECT[name]
    if expect is None:
        if "rounded" in result.stdout:
            raise AssertionError(
                f"case {name}: unexpected rounding note:\n{result.stdout}"
            )
        return
    n_notes = result.stdout.count(expect)
    if n_notes != 1:
        raise AssertionError(
            f"case {name}: {expect!r} printed {n_notes} times "
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
        ("round_up_padded units", test_round_up_padded),
        ("round_up_padded_smooth units", test_round_up_padded_smooth),
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
