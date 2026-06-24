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
