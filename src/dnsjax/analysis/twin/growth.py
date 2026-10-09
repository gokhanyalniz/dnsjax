r"""Growth laws of a decorrelating twin pair.  JAX-free.

A twin difference energy `$E(t)$` grows from its seed and saturates at
`$E_\mathrm{sat}$`, twice the reference field's energy once the two
members are independent.  Which law it follows in between is read
off three quantities, none of which needs a fit window:

- the logarithmic rate `$\gamma = \mathrm{d}\ln E/\mathrm{d}t$`
  (:func:`log_rate`);
- the saturation fraction `$R = E/E_\mathrm{sat}$`, which is
  `$1 - C$` for `$C$` the correlation between the two members;
- the bound-free variable `$f = -\ln(1 - R) = -\ln C$`
  (:func:`bound_free`), which maps `$R \in [0, 1)$` onto
  `$[0, \infty)$`.

What each law looks like, on the axes that make it a straight line:

- **Exponential**, `$E \propto e^{\gamma_0 t}$`: `$\gamma$` is
  constant, so the log-log `$\gamma$`-`$R$` diagram is flat --
  :func:`log_slope` `$s = \mathrm{d}\ln\gamma/\mathrm{d}\ln R = 0$`
  -- and `$\ln E$` is a line in `$t$`.
- **Algebraic**, `$E \propto (t - t_0)^\alpha$`: `$\gamma =
  \alpha/(t - t_0)$` while `$R \propto (t - t_0)^\alpha$`, so
  `$s = -1/\alpha$` is constant and `$1/\gamma$` is a line in `$t$` of
  slope `$1/\alpha$` (:func:`algebraic_exponent` returns that
  `$\alpha$` locally).  What makes such a phase a law rather than a
  passing slope is that it holds where the bound is irrelevant,
  `$R \ll 1$`.
- **Saturation alone**, the logistic `$\dot R = \gamma_0 R (1 - R)$`
  (:func:`logistic_rate`): `$\gamma = \gamma_0 (1 - R)$`, so
  `$s = -R/(1 - R)$` -- flat for `$R \ll 1$`, bending only as
  `$R \to 1$`.  Its `$f$` grows like `$e^{\gamma_0 t}$` early and like
  `$\gamma_0 t$` late: **the same rate at both ends**.
- **Exponential decorrelation**, `$C \propto e^{-\nu t}$`
  (:func:`decorrelation_rate`): `$f$` is a line of slope `$\nu$`, and
  `$\gamma = \nu (1 - R)/R$`.  A late `$\nu$` well below the early
  `$\gamma_0$` is therefore a phase of its own, which the logistic
  cannot produce.

An inflection of `$E$` -- the time where `$\ddot E = 0$` and `$\dot E$`
is stationary -- is none of these.  Every saturating curve has one,
whatever law it grew by: `$E$` is linear in `$t$` there to first order
only, over a width set by the local rate, and at an `$O(1)$` fraction
of saturation by construction, so the slowing it shows is the bound
taking the headroom away.  The `$\gamma$`-`$R$` diagram separates the
two: the bound bends the curve by `$-R/(1-R)$`, a law bends it by a
constant `$-1/\alpha$` already at `$R \ll 1$`.

:func:`longest_window` finds the longest run over which a quantity
stays within a relative band of its own mean -- how a figure marks an
exponential phase (`$\gamma$`) or an exponential-decorrelation phase
(`$\mathrm{d}f/\mathrm{d}t$`) by a criterion it states, rather than by
a window chosen by eye.
"""

from __future__ import annotations

import numpy as np


def log_rate(t: np.ndarray, energy: np.ndarray) -> np.ndarray:
    r"""`$\gamma = \mathrm{d}\ln E/\mathrm{d}t$` along the last axis.

    Second-order differences on the (possibly non-uniform) *t*;
    ``nan`` where *energy* is not positive, whose logarithm does not
    exist.
    """
    energy = np.asarray(energy, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_e = np.where(energy > 0.0, np.log(energy), np.nan)
    return np.gradient(log_e, np.asarray(t, dtype=np.float64), axis=-1)


def log_slope(r: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    r"""`$s = \mathrm{d}\ln\gamma/\mathrm{d}\ln R$` along a trajectory.

    Both differentiated against the sample index, so *r* need not be
    monotonic; ``nan`` where either is not positive or `$R$` stalls.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(gamma > 0.0, np.log(gamma), np.nan)
        lr = np.where(r > 0.0, np.log(r), np.nan)
        dlr = np.gradient(lr, axis=-1)
        return np.where(dlr != 0.0, np.gradient(lg, axis=-1) / dlr, np.nan)


def algebraic_exponent(t: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    r"""The local `$\alpha = (\mathrm{d}\gamma^{-1}/\mathrm{d}t)^{-1}$`.

    Constant over a phase `$E \propto (t - t_0)^\alpha$`, whatever
    `$t_0$`; it diverges for an exponential, whose `$1/\gamma$` is
    flat.  A second derivative of the data, so noisy: read it over a
    window, never at a point.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(gamma != 0.0, 1.0 / gamma, np.nan)
        slope = np.gradient(inv, np.asarray(t, dtype=np.float64), axis=-1)
        return np.where(slope != 0.0, 1.0 / slope, np.nan)


def bound_free(r: np.ndarray) -> np.ndarray:
    r"""`$f = -\ln(1 - R)$`, ``nan`` where `$R \ge 1$`.

    `$R$` reaches and crosses 1 at saturation only through sampling
    noise, and there `$f$` has no value to give; a caller cuts it off
    some way below (``scripts/twin_spectral_maps.py`` at 0.95).
    """
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r < 1.0, -np.log1p(-r), np.nan)


def logistic_rate(r: np.ndarray, gamma0: float) -> np.ndarray:
    r"""`$\gamma = \gamma_0 (1 - R)$`: saturation alone, rate `$\gamma_0$`."""
    return gamma0 * (1.0 - np.asarray(r, dtype=np.float64))


def decorrelation_rate(r: np.ndarray, nu: float) -> np.ndarray:
    r"""`$\gamma = \nu (1 - R)/R$`: exponential decorrelation, rate `$\nu$`."""
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return nu * (1.0 - r) / r


def longest_window(
    values: np.ndarray, tolerance: float, valid: np.ndarray | None = None
) -> tuple[int, int, float]:
    r"""The longest run within *tolerance* (relative) of its own mean.

    Returns ``(start, stop, mean)`` with *stop* exclusive -- the run
    over which every sample satisfies `$|v - \bar v| \le \epsilon
    |\bar v|$` for the run's own mean `$\bar v$` -- or ``(0, 0, nan)``
    when no run of two samples qualifies.  Each run grows from its
    start until the first sample that would break the band, so the
    criterion is greedy and stated rather than tuned.  *valid* masks
    out samples that may not belong to any run (a non-finite rate, or
    `$R$` too close to 1 for `$f$` to mean anything).
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v) if valid is None else np.isfinite(v) & valid
    best = (0, 0, float("nan"))
    n = v.size
    for start in range(n):
        if not ok[start]:
            continue
        total, lo, hi = v[start], v[start], v[start]
        stop = start + 1
        while stop < n and ok[stop]:
            nt = total + v[stop]
            mean = nt / (stop - start + 1)
            nlo, nhi = min(lo, v[stop]), max(hi, v[stop])
            if max(nhi - mean, mean - nlo) > tolerance * abs(mean):
                break
            total, lo, hi = nt, nlo, nhi
            stop += 1
        if stop - start >= 2 and stop - start > best[1] - best[0]:
            best = (start, stop, float(total / (stop - start)))
    return best
