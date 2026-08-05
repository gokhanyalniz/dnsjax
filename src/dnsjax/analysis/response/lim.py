r"""Linear inverse modeling (LIM) from unforced probe streams.

Identifies the linear operator `$L$` governing a mode's fluctuation
statistics from a plain (unforced, unperturbed) turbulent run: under
the LIM hypothesis the fluctuations obey `$d\mathbf{b}/dt =
L\mathbf{b} + \boldsymbol{\xi}$` with `$\boldsymbol{\xi}$` white in
time, so the lagged covariances satisfy `$C(\tau) = e^{\tau L}
C(0)$` and

.. math::
    M(\tau) = C(\tau)\,C(0)^{-1} \approx e^{\tau L},

fed to the same multi-horizon ``logm`` fit as the ensemble
identification (:func:`~dnsjax.analysis.response.ensemble.
identify_generator`).  Unlike the injected-basis routes (ensemble
impulse responses, SSI forcing), LIM needs **no extra runs**: the
input is the probe stream of the production run itself
(``probes.modes``; :mod:`dnsjax.extensions.probes`).  Its price is the
whiteness hypothesis -- when the turbulent forcing of the mode is
correlated in time, `$M(\tau)$` is no longer a semigroup and the
identified `$L$` drifts with the lag; the per-lag reconstruction
residuals expose exactly that drift.

Estimator
=========
Per lag `$\ell$` (in probe samples, `$\tau = \ell\,\Delta$` with
`$\Delta = \mathtt{it\_probes}\cdot dt$`), both covariances are
accumulated over the *same* overlap window and pooled over all
segments (independent runs / stream files):

.. math::
    C(\tau) = \sum_k \mathbf{b}_{k+\ell}\mathbf{b}_k^H, \qquad
    C_0(\tau) = \sum_k \mathbf{b}_k\mathbf{b}_k^H,

so `$M(\tau) = C\,C_0^{-1}$` is exact (to roundoff) for noiseless
linear data -- the anchor the unit tests pin -- and unbiased under
stationarity.  ``b`` are the probe profiles projected onto the
exported operator's energy coordinates
(:func:`~dnsjax.analysis.response.ensemble.project_series`), each
segment's own sample mean subtracted (``demean``).

Knobs (when to tweak)
=====================
- ``--modes-npz`` (+ ``--n-modes``): restrict to the leading
  controllability modes before estimating.  Recommended: the full
  resolved subspace contains weakly excited directions that make
  `$C_0$` ill-conditioned and the fit noise-dominated; the
  restriction is also what makes the result comparable with the
  other identification routes on the same basis.  Without it the
  fit runs on all ``r_res`` coordinates (fine for small operators
  or very long streams; a conditioning error tells you when not).
- ``--lags``: several, spread over the correlation time of the mode:
  short lags weight sampling noise (the ``logm`` divides by
  `$\tau$`), long lags weight the whiteness error and eventually hit
  the decay / branch-cut rejections; disagreement across lags
  (residuals, or refitting per lag) measures the LIM hypothesis
  itself.
- ``--t-min``: cut the initial transient -- the estimator assumes
  stationarity.
- Stream length: the covariance noise decays as
  `$1/\sqrt{n_\mathrm{samples}}$`; pass several ``--probes`` runs to
  pool independent segments.

CLI
===
::

    python -m dnsjax.analysis.response.lim \
        --probes run1/ run2/ --mode 3,0 --operator U_mean_tg_op.npz \
        --modes-npz U_mean_cont.npz --n-modes 10 \
        --lags "0.5,1,2" --t-min 200 --out lim.npz

writes the identified ``L``, its spectrum / stability, per-lag
residuals, and growth curves ``G_id`` (of ``L``) vs ``G_ref`` (of
the reference operator restricted to the same basis) -- the same
output convention as ``ensemble identify``, so the routes are
directly comparable.  SciPy is imported lazily (``logm``); the
growth curves use the JAX-based :mod:`.operator_tools` sweeps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .ensemble import (
    identify_generator,
    project_series,
    snap_sample_lags,
    stability_report,
)
from .probes import ProbeData, _time_mask, read_probes

__all__ = [
    "projected_fluctuations",
    "lagged_propagators",
    "identify_lim",
]


def projected_fluctuations(
    data: ProbeData,
    t_proj: np.ndarray,
    i2: int,
    i3: int,
    p: np.ndarray | None = None,
    t_min: float = 0.0,
    demean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One probe stream's mode series in operator (or basis) coordinates.

    Selects mode ``(i2, i3)``, keeps ``t >= t_min``, projects
    (:func:`~dnsjax.analysis.response.ensemble.project_series`), and
    subtracts the segment's own sample mean (*demean*; keep it on for
    turbulence -- off only for deterministic linear data, where the
    sample mean is dynamics, not statistics).  Returns ``(t, b)``
    with ``b`` of shape ``(nt, m)``; requires the kept samples to be
    uniformly spaced and strictly increasing (a resumed stream that
    re-ran a segment must be cut with *t_min* first).
    """
    k = data.mode_index(i2, i3)
    mask = _time_mask(data, t_min)
    t = data.t[mask]
    if len(t) > 1:
        steps = np.diff(t)
        if not np.allclose(steps, steps[0], rtol=0, atol=1e-9):
            raise ValueError(
                "probe samples are not uniformly spaced after the "
                "t_min cut (an overlapping resume?); raise t_min past "
                "the overlap"
            )
    b = project_series(data.u[mask, k], t_proj, p)
    if demean:
        b = b - b.mean(axis=0)
    return t, b


def lagged_propagators(
    segments: list[np.ndarray], lags: list[int], delta: float
) -> tuple[list[tuple[float, np.ndarray]], dict]:
    r"""Propagator samples `$M(\ell\Delta) = C(\tau) C_0(\tau)^{-1}$`.

    *segments* are ``(nt, m)`` coordinate series (independent runs;
    pooled), *lags* the integer sample lags, *delta* the sample
    interval.  Both covariances use the lag's shared overlap window
    (module docstring), so the returned pairs feed
    :func:`~dnsjax.analysis.response.ensemble.identify_generator`
    directly.  The diagnostics carry per-lag sample counts and the
    condition number of `$C_0$` -- a large one (> 1e8 raises) means
    the coordinates contain directions the data barely excites:
    restrict to fewer basis modes.
    """
    if not segments:
        raise ValueError("no segments given")
    m = segments[0].shape[1]
    pairs: list[tuple[float, np.ndarray]] = []
    counts: list[int] = []
    conds: list[float] = []
    for lag in lags:
        if lag < 1:
            raise ValueError(f"lag {lag} must be >= 1 sample")
        num = np.zeros((m, m), dtype=complex)
        den = np.zeros((m, m), dtype=complex)
        n_used = 0
        for b in segments:
            if b.shape[1] != m:
                raise ValueError(
                    f"inconsistent coordinate counts: {b.shape[1]} vs {m}"
                )
            if b.shape[0] <= lag:
                continue
            x0, x1 = b[:-lag], b[lag:]
            num += x1.T @ np.conj(x0)
            den += x0.T @ np.conj(x0)
            n_used += x0.shape[0]
        if n_used < m:
            raise ValueError(
                f"lag {lag}: only {n_used} sample pairs for {m} "
                "coordinates; the streams are too short for this lag"
            )
        cond = float(np.linalg.cond(den))
        if cond > 1e8:
            raise ValueError(
                f"lag {lag}: C(0) condition number {cond:.2e}; the "
                "data barely excites some coordinate directions -- "
                "restrict to fewer basis modes (--modes-npz/--n-modes) "
                "or provide longer streams"
            )
        # M C0 = C  =>  C0^T M^T = C^T.
        m_mat = np.linalg.solve(den.T, num.T).T
        pairs.append((lag * delta, m_mat))
        counts.append(n_used)
        conds.append(cond)
    return pairs, {"n_samples": counts, "c0_cond": conds}


def identify_lim(
    probes: list[str | Path],
    i2: int,
    i3: int,
    operator: str | Path,
    lags: list[float],
    modes_npz: str | Path | None = None,
    n_modes: int | None = None,
    t_min: float = 0.0,
    demean: bool = True,
) -> dict:
    """Full LIM pipeline: probe streams -> identified generator.

    *probes* are run directories / ``probes.bin`` paths (pooled as
    independent segments; their cadence must agree), *lags* fit times
    (each snapped to the nearest probe interval, >= 1).  With
    *modes_npz* the estimation runs on the recovered controllability
    basis (recommended; ``--n-modes`` truncates it), else on the full
    resolved coordinates.  Returns the ``ensemble identify``-style
    dict (``L``, spectrum, residuals, ``basis``/``op``/``delta`` for
    the CLI's growth curves) plus the estimator diagnostics.
    """
    from .operator_tools import load_modes_npz, load_operator, recover_basis

    op = load_operator(operator, i2, i3)
    p = None
    if modes_npz is not None:
        p = recover_basis(op, load_modes_npz(modes_npz, i2, i3, n_modes))
    elif n_modes is not None:
        raise ValueError("n_modes needs modes_npz (the basis to truncate)")

    segments: list[np.ndarray] = []
    delta: float | None = None
    for path in probes:
        data = read_probes(path)
        _, b = projected_fluctuations(
            data, op.T_proj, i2, i3, p, t_min, demean
        )
        d = float(data.meta["it_probes"]) * float(data.meta["dt"])
        if delta is None:
            delta = d
        elif not np.isclose(d, delta, rtol=0, atol=1e-12):
            raise ValueError(
                f"{path}: probe interval {d:g} differs from the first "
                f"stream's {delta:g}; streams must share the cadence"
            )
        segments.append(b)

    sample_lags = snap_sample_lags(lags, delta, "lim")
    pairs, est_diag = lagged_propagators(segments, sample_lags, delta)
    l_mat, fit_diag = identify_generator(pairs)
    report = stability_report(l_mat)
    return {
        "L": l_mat,
        "eigvals": report["eigvals"],
        "spectral_abscissa": report["spectral_abscissa"],
        "stable": report["stable"],
        "lags": np.asarray([tau for tau, _ in pairs]),
        "residuals": np.asarray(fit_diag["residuals"]),
        "n_samples": np.asarray(est_diag["n_samples"]),
        "c0_cond": np.asarray(est_diag["c0_cond"]),
        "basis": p if p is not None else np.eye(op.A.shape[0], dtype=complex),
        "op": op,
        "delta": delta,
    }


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from ...bootstrap import configure_jax_platform
    from ...harmonics import parse_mode_pairs

    ap = argparse.ArgumentParser(
        prog="python -m dnsjax.analysis.response.lim",
        description="Linear inverse modeling from unforced probe "
        "streams (see the module docstring).",
        allow_abbrev=False,
    )
    ap.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=("cpu", "cuda", "rocm", "tpu"),
        help="JAX backend for the growth-curve sweeps",
    )
    ap.add_argument(
        "--probes",
        nargs="+",
        required=True,
        help="probe streams (run dirs or probes.bin paths), pooled",
    )
    ap.add_argument("--mode", required=True, help='"i2,i3" probed mode')
    ap.add_argument("--operator", required=True, help="<stem>_tg_op.npz")
    ap.add_argument(
        "--modes-npz",
        default=None,
        help="controllability bundle: estimate on its basis "
        "(recommended; see the module docstring)",
    )
    ap.add_argument(
        "--n-modes",
        type=int,
        default=None,
        help="leading basis columns kept (--modes-npz only)",
    )
    ap.add_argument(
        "--lags",
        required=True,
        help='comma list of fit lags in time units, e.g. "0.5,1,2" '
        "(several, spread over the mode's correlation time)",
    )
    ap.add_argument(
        "--t-min",
        type=float,
        default=0.0,
        help="discard the initial transient before this time",
    )
    ap.add_argument(
        "--growth-tmax",
        type=float,
        default=None,
        help="growth-curve extent (default: twice the longest lag)",
    )
    ap.add_argument("--out", required=True, help="output npz path")
    args = ap.parse_args(argv)
    configure_jax_platform(args.platform, double_precision=True)

    pairs = parse_mode_pairs(args.mode)
    if len(pairs) != 1:
        raise SystemExit("--mode takes exactly one 'i2,i3' pair")
    i2, i3 = pairs[0]
    lags = [float(tok) for tok in args.lags.split(",")]

    result = identify_lim(
        args.probes,
        i2,
        i3,
        args.operator,
        lags,
        modes_npz=args.modes_npz,
        n_modes=args.n_modes,
        t_min=args.t_min,
    )

    from .operator_tools import growth_curve, restrict

    op = result.pop("op")
    delta = result.pop("delta")
    basis = result["basis"]
    t_max = args.growth_tmax
    if t_max is None:
        t_max = 2.0 * float(np.max(result["lags"]))
    t_grid = delta * np.arange(int(round(t_max / delta)) + 1)
    g_id = growth_curve(result["L"], t_grid)
    a_ref = restrict(op.A, basis) if args.modes_npz is not None else op.A
    g_ref = growth_curve(a_ref, t_grid)
    np.savez(
        args.out,
        readme=(
            "dnsjax LIM identification. L: generator identified from "
            "lagged covariances of the unforced probe stream, on the "
            "injected/restricted basis (energy-orthonormal "
            "coordinates); G_id/G_ref: growth curves of L and of the "
            "reference operator on the same basis, on t_grid; "
            "residuals: per-lag ||e^{tau L} - M(tau)||_F/||M||_F "
            "(their growth with tau measures the whiteness "
            "hypothesis)."
        ),
        t_grid=t_grid,
        G_id=g_id,
        G_ref=g_ref,
        **result,
    )
    print(
        f"[lim] wrote {args.out}: spectral abscissa "
        f"{result['spectral_abscissa']:+.4e} "
        f"({'stable' if result['stable'] else 'UNSTABLE'}), "
        f"max |G_id - G_ref|/G_ref = "
        f"{float(np.max(np.abs(g_id - g_ref) / g_ref)):.3g}, "
        f"lag residuals {np.round(result['residuals'], 4)}."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
