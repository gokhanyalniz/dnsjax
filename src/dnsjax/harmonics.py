r"""Integer wavenumber sequences for the spectral axes (JAX-free).

These NumPy generators are the single source of truth for the Fourier
mode numbering used throughout the solver and by the JAX-free analysis
tooling (:mod:`dnsjax.analysis`).  :mod:`dnsjax.operators` re-exports
them wrapped in ``jnp.asarray`` so the runtime keeps device arrays,
while host-side / external code (which must run without JAX) imports
the NumPy versions directly.

The conventions match the storage layout: the Nyquist mode is always
omitted, so a real-FFT axis carries `$n / 2$` modes and a full-complex
axis carries `$n - 1$` modes (see :mod:`dnsjax.fft`).
"""

import numpy as np
from numpy import ndarray


def real_harmonics(n: int) -> ndarray:
    r"""Non-negative integer wavenumbers for a real-FFT axis.

    The Nyquist mode is omitted, leaving `$n / 2$` modes.

    Parameters
    ----------
    n:
        Full mode count along the axis.

    Returns
    -------
    :
        Wavenumber array `$[0, 1, \dots, n/2 - 1]$`, shape
        ``(n // 2,)``.
    """
    # Omits the Nyquist mode
    return np.arange(0, n // 2, dtype=int)


def parse_mode_pairs(spec: str) -> list[tuple[int, int]]:
    r"""Parse an ``"i2,i3;i2,i3;..."`` spectral-mode list.

    Each pair is a global spectral index: ``i2`` on the complex
    (axis-2) slot and ``i3`` on the real-FFT (axis-3) slot of the
    stored spectral layout -- the same convention as the
    transient-growth CLI ``--modes`` argument.  Whitespace around
    numbers and separators is ignored.  Purely syntactic (this module
    is a JAX-free leaf): no range check against a resolution --
    callers validate bounds themselves.

    Raises ``ValueError`` on malformed pairs, negative indices, or
    duplicates.
    """
    pairs: list[tuple[int, int]] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            raise ValueError(
                f"empty mode entry in {spec!r} (expected 'i2,i3;i2,i3')"
            )
        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"malformed mode {item!r} in {spec!r} (expected 'i2,i3')"
            )
        try:
            i2, i3 = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"non-integer mode {item!r} in {spec!r}"
            ) from None
        if i2 < 0 or i3 < 0:
            raise ValueError(f"negative mode index in {item!r}")
        if (i2, i3) in pairs:
            raise ValueError(f"duplicate mode ({i2},{i3}) in {spec!r}")
        pairs.append((i2, i3))
    return pairs


def complex_harmonics(n: int) -> ndarray:
    r"""Full-complex integer wavenumbers with the Nyquist mode omitted.

    Parameters
    ----------
    n:
        Full mode count along the axis.

    Returns
    -------
    :
        `$n - 1$` wavenumbers in FFT order:
        `$[0, 1, \dots, n/2-1, -n/2+1, \dots, -1]$`.
    """
    qs = (np.arange(n, dtype=int) + n // 2) % n - n // 2
    # Omits the Nyquist mode
    qs_out = np.zeros(n - 1, dtype=int)
    qs_out[: n // 2] = qs[: n // 2]
    qs_out[n // 2 :] = qs[n // 2 + 1 :]
    return qs_out
