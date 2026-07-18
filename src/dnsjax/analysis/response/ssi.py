r"""Stochastic-forcing cross-covariance identification (SSI).

Identifies a mode's linear operator from a run driven by the
white-in-time stochastic kicks of the ``force`` section
(:mod:`dnsjax.forcing`): cross-correlating the probe stream with the
*recorded* kick coefficients isolates the coherent forced response
-- the natural turbulence is statistically independent of the
injected noise and drops out of the cross-covariance -- so, unlike
LIM (:mod:`.lim`), **no whiteness hypothesis on the turbulent
background** is needed; the injected forcing is white by
construction.  The price is running the forced experiment (one
production run with ``[force]`` + probes) instead of reusing an
unforced stream.

Estimator
=========
Kicks add `$\varepsilon\,P\,\mathbf{w}_k$` at times `$t_k$`
(`$\mathbf{w}_k \sim \mathcal{CN}(0, I_m)$` recorded in
``forcing.bin``; `$P$` the channel basis, the leading
controllability modes the run was configured with).  Probe samples
at kick times are pre-kick (the :mod:`dnsjax.forcing` timing
convention), so with `$\mathbf{b}(t)$` the probe profiles projected
onto the basis coordinates, the lagged cross-covariance regression

.. math::
    M(\ell\Delta) = \frac{1}{\varepsilon}\,
        \Big[\textstyle\sum_k \mathbf{b}(t_k + \ell\Delta)\,
        \mathbf{w}_k^H\Big]
        \Big[\textstyle\sum_k \mathbf{w}_k^{\vphantom{H}}
        \mathbf{w}_k^H\Big]^{-1}
        \;\longrightarrow\; P^H e^{\ell\Delta\,A} P

estimates the projected propagator at every probe lag
(`$\Delta = \mathtt{it\_probes}\cdot dt$`; responses to *other*
kicks and the turbulent background average out).  The lag-0 sample
correlates only with **earlier** kicks, so `$\lVert M(0)\rVert$` is
a built-in causality check: it measures the pure noise floor of the
estimator and should be small against `$\lVert M(\ell)\rVert
\approx 1$`.  The pairs then feed the shared multi-horizon fit
:func:`~dnsjax.analysis.response.ensemble.identify_generator`,
exactly like the ensemble and LIM routes -- the three
identifications share the fit, the coordinates, and the output
convention, so their operators are directly comparable.

Knobs (when to tweak)
=====================
- ``--lags``: as for the other routes -- several, inside the window
  where the kick response is still detectable; short lags weight the
  noise floor (see the causality number), long ones the response
  decay.  The per-lag residuals arbitrate.
- ``--t-min``: drop kicks before the forced run is statistically
  settled (the stationary forced level builds up over the mode's
  decay time from the forcing onset).
- Noise floor `$\propto 1/\sqrt{n_\mathrm{kicks}}$`: extend the run,
  or pool several independently seeded forced runs (``--runs`` takes
  many).
- The forcing amplitude / channel count are **run-time** knobs
  (``force.amplitude`` / ``force.n_channels``,
  :class:`dnsjax.extensions.ForceParams`); this module reads
  them from the sidecar.  :func:`predicted_forced_variance` gives
  the stationary forced level for planning the amplitude.

CLI
===
::

    python -m dnsjax.analysis.response.ssi \
        --runs run1/ run2/ --mode 3,0 --operator U_mean_tg_op.npz \
        --lags "0.5,1,2" --t-min 200 --out ssi.npz

The channel basis defaults to the profile bundle recorded in each
run's ``forcing.json`` (``--modes-npz`` overrides, e.g. after moving
files).  Outputs mirror ``ensemble identify`` / ``lim``: the
identified ``L``, spectrum/stability, per-lag residuals, causality
level, measured vs predicted variance, and ``G_id``/``G_ref`` growth
curves.  SciPy is imported lazily; growth curves use the JAX-based
:mod:`.operator_tools` sweeps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ensemble import (
    identify_generator,
    project_series,
    snap_sample_lags,
    stability_report,
)
from .probes import ProbeData, _resolve_pair, read_probes

__all__ = [
    "ForcingData",
    "read_forcing",
    "kick_response_windows",
    "cross_propagators",
    "identify_ssi",
    "predicted_forced_variance",
]


# ── Forcing-stream reader ────────────────────────────────────────


@dataclass(frozen=True)
class ForcingData:
    r"""One kick-coefficient stream, fully loaded.

    ``t`` are the kick times, ``w`` the `$\mathcal{CN}(0,1)$`
    coefficients (``(n_kicks, K, m)`` complex128; ``K`` forced modes
    in sidecar order, ``m`` channels); the physical kick was
    ``meta["amplitude"] * sum_j w_j profile_j``.  ``meta`` is the
    full sidecar dict (``meta["profiles"]`` names the channel
    bundle).
    """

    t: np.ndarray
    w: np.ndarray
    modes: np.ndarray
    meta: dict

    def mode_index(self, i2: int, i3: int) -> int:
        """Index of forced mode ``(i2, i3)`` along ``w``'s axis 1."""
        hits = np.nonzero((self.modes[:, 0] == i2) & (self.modes[:, 1] == i3))[
            0
        ]
        if hits.size == 0:
            raise KeyError(
                f"mode ({i2},{i3}) was not forced "
                f"(forced: {self.modes.tolist()})"
            )
        return int(hits[0])


def read_forcing(path: str | Path = ".") -> ForcingData:
    """Load a kick stream (a run directory or the ``forcing.bin``)."""
    bin_path, json_path = _resolve_pair(path, "forcing")
    if not json_path.exists():
        raise FileNotFoundError(f"forcing sidecar {json_path} not found")
    with open(json_path) as f:
        meta = json.load(f)

    modes = np.asarray(meta["modes"], dtype=int)
    m = int(meta["n_channels"])
    record_dtype = np.dtype([("t", "<f8"), ("w", "<f8", (len(modes), m, 2))])
    # Sized read straight into the record dtype (no byte-array
    # copies; the ``read_probes`` idiom).
    n_rec, rem = divmod(bin_path.stat().st_size, record_dtype.itemsize)
    if rem:
        print(
            f"[ssi] {bin_path}: dropping a truncated trailing record "
            f"({rem} of {record_dtype.itemsize} bytes)."
        )
    rec = np.fromfile(bin_path, dtype=record_dtype, count=n_rec)
    t = rec["t"].astype(np.float64)
    if n_rec > 1 and not (np.diff(t) > 0).all():
        print(
            f"[ssi] {bin_path}: non-monotonic kick times (a resume "
            "re-ran a trajectory segment?); filter by t_min."
        )
    w = rec["w"][..., 0] + 1j * rec["w"][..., 1]
    return ForcingData(t=t, w=w.astype(np.complex128), modes=modes, meta=meta)


# ── Estimator ────────────────────────────────────────────────────


def kick_response_windows(
    probe: ProbeData,
    forcing: ForcingData,
    t_proj: np.ndarray,
    i2: int,
    i3: int,
    max_lag: int,
    p: np.ndarray | None = None,
    t_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-kick coefficient / response-window arrays for one run.

    Aligns each kick to its (pre-kick) probe sample -- exact by the
    ``it_force % it_probes == 0`` validation -- and keeps kicks with
    ``t >= t_min`` whose full lag window ``0..max_lag`` fits in the
    probe stream.  Returns ``(w, resp)``: ``(n_kicks, m)``
    coefficients and ``(n_kicks, max_lag + 1, m)`` projected response
    coordinates.
    """
    kf = forcing.mode_index(i2, i3)
    k_probe = probe.mode_index(i2, i3)
    b = project_series(probe.u[:, k_probe], t_proj, p)  # (nt, m)

    if len(probe.t) < 2:
        raise ValueError("the probe stream has fewer than 2 samples")
    delta = float(probe.t[1] - probe.t[0])
    w_rows, resp_rows = [], []
    for k, t_k in enumerate(forcing.t):
        if t_k < t_min:
            continue
        idx = int(round((t_k - probe.t[0]) / delta))
        if idx < 0 or idx + max_lag >= len(probe.t):
            continue
        if abs(probe.t[idx] - t_k) > 1e-9:
            raise ValueError(
                f"kick at t = {t_k:g} does not coincide with a probe "
                "sample (mismatched streams?)"
            )
        w_rows.append(forcing.w[k, kf])
        resp_rows.append(b[idx : idx + max_lag + 1])
    if not w_rows:
        raise ValueError(
            f"no kicks with t >= {t_min} whose {max_lag}-lag window "
            "fits in the probe stream"
        )
    return np.asarray(w_rows), np.asarray(resp_rows)


def cross_propagators(
    windows: list[tuple[np.ndarray, np.ndarray]],
    lags: list[int],
    delta: float,
    eps: float,
    demean: bool = True,
) -> tuple[list[tuple[float, np.ndarray]], dict]:
    r"""Propagator samples from kick/response windows (module formula).

    *windows* are per-run ``(w, resp)`` pairs
    (:func:`kick_response_windows`; pooled with per-run sample means
    subtracted when *demean* -- keep it on for turbulence, off only
    for deterministic synthetic data).  Returns the ``(tau, M)``
    pairs for :func:`~dnsjax.analysis.response.ensemble.
    identify_generator` plus diagnostics: total kick count, the
    empirical channel-covariance condition number, and the
    ``causality`` level `$\lVert M(0)\rVert_2$` (the estimator's
    noise floor; should be small against 1).
    """
    if not windows:
        raise ValueError("no kick/response windows given")
    m = windows[0][0].shape[1]
    max_lag = windows[0][1].shape[1] - 1
    if any(lag < 0 or lag > max_lag for lag in lags):
        raise ValueError(f"lags {lags} outside the 0..{max_lag} window")
    num = np.zeros((max_lag + 1, m, m), dtype=complex)
    den = np.zeros((m, m), dtype=complex)
    n_kicks = 0
    for w, resp in windows:
        if w.shape[1] != m or resp.shape[2] != m:
            raise ValueError(
                "channel/coordinate count mismatch across runs "
                f"({w.shape}, {resp.shape} vs m = {m})"
            )
        if demean:
            w = w - w.mean(axis=0)
            resp = resp - resp.mean(axis=0)
        # num[l] = sum_k resp[k, l] w_k^H ; den = sum_k w_k w_k^H.
        num += np.einsum("klm,kn->lmn", resp, np.conj(w))
        den += w.T @ np.conj(w)
        n_kicks += w.shape[0]
    if n_kicks < m:
        raise ValueError(
            f"only {n_kicks} usable kicks for {m} channels; the "
            "streams are too short"
        )
    cond = float(np.linalg.cond(den))
    # M(l) den = num[l] / eps  =>  den^T M^T = (num/eps)^T.
    m_all = np.stack(
        [
            np.linalg.solve(den.T, (num[lag] / eps).T).T
            for lag in range(max_lag + 1)
        ]
    )
    pairs = [(lag * delta, m_all[lag]) for lag in lags if lag >= 1]
    diag = {
        "n_kicks": n_kicks,
        "w_cond": cond,
        "causality": float(np.linalg.norm(m_all[0], 2)),
    }
    return pairs, diag


def predicted_forced_variance(
    a: np.ndarray, p: np.ndarray, eps: float, dt_force: float
) -> float:
    r"""Stationary forced variance `$\mathrm{tr}(P^H X P)$` on the basis.

    Samples are **pre-kick** (the :mod:`dnsjax.forcing` convention),
    so the sampled state obeys `$x_{n+1} = E\,(x_n + \varepsilon P
    \mathbf{w}_n)$` with `$E = e^{\Delta_f A}$`, and `$X$` solves the
    discrete Lyapunov equation `$X = E\,(X + Q)\,E^H$` with the
    per-kick injection `$Q = \varepsilon^2 P P^H$` -- the kick-forced
    analogue of the continuous controllability Gramian (as
    `$\Delta_f \to 0$` at fixed `$\varepsilon^2/\Delta_f$` the two
    coincide).  Use it to plan ``force.amplitude``: the returned
    level is the expected stationary `$\sum\lvert b_j\rvert^2$` of
    the forced part, to be kept inside the linear window and above
    the natural background you want to beat.  Requires a stable
    `$A$`.
    """
    from scipy.linalg import expm

    a = np.asarray(a)
    p = np.asarray(p)
    e_mat = expm(dt_force * a)
    if np.max(np.abs(np.linalg.eigvals(e_mat))) >= 1.0:
        raise ValueError(
            "A is not stable; the stationary forced variance does not exist"
        )
    q = eps**2 * (p @ p.conj().T)
    q_eff = e_mat @ q @ e_mat.conj().T  # kick propagated to the sample
    try:
        from scipy.linalg import solve_discrete_lyapunov

        x = solve_discrete_lyapunov(e_mat, q_eff)
    except ImportError:  # pragma: no cover - scipy present in dev
        x = _discrete_lyapunov_eig(e_mat, q_eff)
    return float(np.real(np.trace(p.conj().T @ x @ p)))


def _discrete_lyapunov_eig(e_mat: np.ndarray, q: np.ndarray) -> np.ndarray:
    r"""Eigendecomposition closed form of `$X = E X E^H + Q$`.

    With `$E = S\Lambda S^{-1}$`: `$\tilde{X}_{ij} = \tilde{Q}_{ij}
    / (1 - \lambda_i\bar{\lambda}_j)$` in the eigenbasis.  SciPy-free
    fallback, mirroring the continuous-case fallback in
    :mod:`.operator_tools`.
    """
    lam, s = np.linalg.eig(e_mat)
    s_inv = np.linalg.inv(s)
    q_t = s_inv @ q @ s_inv.conj().T
    x_t = q_t / (1.0 - lam[:, None] * np.conj(lam)[None, :])
    return s @ x_t @ s.conj().T


# ── Pipeline ─────────────────────────────────────────────────────


def identify_ssi(
    runs: list[str | Path],
    i2: int,
    i3: int,
    operator: str | Path,
    lags: list[float],
    modes_npz: str | Path | None = None,
    t_min: float = 0.0,
    demean: bool = True,
) -> dict:
    """Full SSI pipeline: forced run(s) -> identified generator.

    *runs* are forced run directories (each holding the
    ``probes.*``/``forcing.*`` pairs; pooled -- their cadence,
    amplitude, and channel count must agree), *lags* fit times
    snapped to the probe grid.  The channel basis is recovered from
    *modes_npz* (default: the profile bundle recorded in the first
    run's sidecar), truncated to the ``n_channels`` the run forced.
    Returns the ``identify``-convention dict plus the SSI
    diagnostics (``causality``, ``n_kicks``, measured vs predicted
    stationary variance).
    """
    from .operator_tools import load_modes_npz, load_operator, recover_basis

    if not runs:
        raise ValueError("no runs given")
    op = load_operator(operator, i2, i3)

    # Kick logs are small; read them all up front for the config
    # checks and the sidecar-recorded channel bundle.
    forcings = [read_forcing(r) for r in runs]
    f0 = forcings[0].meta
    eps = float(f0["amplitude"])
    n_ch = int(f0["n_channels"])
    for r, f in zip(runs, forcings, strict=True):
        if float(f.meta["amplitude"]) != eps or (
            int(f.meta["n_channels"]) != n_ch
        ):
            raise ValueError(
                f"{r}: forcing amplitude/channel count differ from the "
                "first run's; pool only identically configured runs"
            )
    if modes_npz is None:
        modes_npz = f0["profiles"]
        if not Path(modes_npz).exists():
            raise FileNotFoundError(
                f"the sidecar's profile bundle {modes_npz} no longer "
                "exists; pass --modes-npz explicitly"
            )
    p = recover_basis(op, load_modes_npz(modes_npz, i2, i3, n_ch))

    # Probe streams can be multi-GB: read one run at a time and
    # reduce it to its (small) kick/response windows immediately, so
    # peak memory is one stream, not the whole pool.
    delta: float | None = None
    sample_lags: list[int] = []
    max_lag = 0
    windows = []
    for r, fd in zip(runs, forcings, strict=True):
        pd = read_probes(r)
        d = float(pd.meta["it_probes"]) * float(pd.meta["dt"])
        if delta is None:
            delta = d
            sample_lags = snap_sample_lags(lags, delta, "ssi")
            max_lag = max(sample_lags)
        elif not np.isclose(d, delta, rtol=0, atol=1e-12):
            raise ValueError(
                f"{r}: probe interval {d:g} differs from the first "
                f"run's {delta:g}"
            )
        windows.append(
            kick_response_windows(pd, fd, op.T_proj, i2, i3, max_lag, p, t_min)
        )
        del pd  # drop the full stream before reading the next run
    pairs, est_diag = cross_propagators(
        windows, sample_lags, delta, eps, demean
    )
    l_mat, fit_diag = identify_generator(pairs)
    report = stability_report(l_mat)

    # Measured stationary variance of the basis coordinates (forced +
    # natural), against the predicted forced part.
    var_meas = float(
        np.mean(
            [
                np.mean(np.sum(np.abs(resp[:, 0]) ** 2, axis=1))
                for _, resp in windows
            ]
        )
    )
    dt_force = float(f0["it_force"]) * float(f0["dt"])
    from .operator_tools import restrict

    a_ref = restrict(op.A, p)
    try:
        var_pred = predicted_forced_variance(op.A, p, eps, dt_force)
    except ValueError:
        var_pred = float("nan")  # unstable reference operator

    return {
        "L": l_mat,
        "eigvals": report["eigvals"],
        "spectral_abscissa": report["spectral_abscissa"],
        "stable": report["stable"],
        "lags": np.asarray([tau for tau, _ in pairs]),
        "residuals": np.asarray(fit_diag["residuals"]),
        "causality": est_diag["causality"],
        "n_kicks": est_diag["n_kicks"],
        "w_cond": est_diag["w_cond"],
        "var_measured": var_meas,
        "var_forced_predicted": var_pred,
        "basis": p,
        "op": op,
        "a_ref": a_ref,
        "delta": delta,
    }


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from ...bootstrap import configure_jax_platform
    from ...harmonics import parse_mode_pairs

    ap = argparse.ArgumentParser(
        prog="python -m dnsjax.analysis.response.ssi",
        description="Cross-covariance identification from stochastically "
        "forced runs (see the module docstring).",
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
        "--runs",
        nargs="+",
        required=True,
        help="forced run directories (probes.* + forcing.*), pooled",
    )
    ap.add_argument("--mode", required=True, help='"i2,i3" forced mode')
    ap.add_argument("--operator", required=True, help="<stem>_tg_op.npz")
    ap.add_argument(
        "--modes-npz",
        default=None,
        help="channel-profile bundle (default: the path recorded in "
        "the run's forcing.json)",
    )
    ap.add_argument(
        "--lags",
        required=True,
        help='comma list of fit times, e.g. "0.5,1,2" (inside the '
        "kick-response window; see the module docstring)",
    )
    ap.add_argument(
        "--t-min",
        type=float,
        default=0.0,
        help="drop kicks before this time (forced-level spin-up)",
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

    result = identify_ssi(
        args.runs,
        i2,
        i3,
        args.operator,
        lags,
        modes_npz=args.modes_npz,
        t_min=args.t_min,
    )

    from .operator_tools import growth_curve

    result.pop("op")
    a_ref = result.pop("a_ref")
    delta = result.pop("delta")
    t_max = args.growth_tmax
    if t_max is None:
        t_max = 2.0 * float(np.max(result["lags"]))
    t_grid = delta * np.arange(int(round(t_max / delta)) + 1)
    g_id = growth_curve(result["L"], t_grid)
    g_ref = growth_curve(a_ref, t_grid)
    np.savez(
        args.out,
        readme=(
            "dnsjax SSI identification. L: generator identified from "
            "kick/response cross-covariances, on the forced channel "
            "basis (energy-orthonormal coordinates); G_id/G_ref: "
            "growth curves of L and of the reference operator "
            "restricted to the same basis, on t_grid; causality: "
            "||M(0)||_2, the estimator noise floor."
        ),
        t_grid=t_grid,
        G_id=g_id,
        G_ref=g_ref,
        **result,
    )
    print(
        f"[ssi] wrote {args.out}: {result['n_kicks']} kicks, "
        f"causality {result['causality']:.3g}, spectral abscissa "
        f"{result['spectral_abscissa']:+.4e} "
        f"({'stable' if result['stable'] else 'UNSTABLE'}), "
        f"max |G_id - G_ref|/G_ref = "
        f"{float(np.max(np.abs(g_id - g_ref) / g_ref)):.3g}, "
        f"variance measured {result['var_measured']:.3e} vs forced "
        f"prediction {result['var_forced_predicted']:.3e}."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
