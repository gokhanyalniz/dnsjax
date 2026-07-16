r"""Ensemble-response aggregation and direct operator identification.

Post-processing for member trees built by
``scripts/ensemble_setup.py``: combine the members' probe streams
into the ensemble-averaged mode response `$\langle\hat{u}\rangle(t)$`
(``aggregate``), and identify the linear operator governing the
ensemble-averaged perturbation dynamics directly from the responses
to an injected basis (``identify``).

Aggregation
===========
Member probe streams are aligned on **relative** time (each member
continues its parent snapshot's clock; the grids must agree, which
the shared ``it_probes * dt`` cadence guarantees) and pair-combined
per the tree's pairing: antithetic `$(\hat{u}_+ - \hat{u}_-)/2$`
(cancels the common turbulent evolution and all even-order nonlinear
contributions), baseline `$\hat{u}_p - \hat{u}_b$`, or the plain
mean.  The ensemble mean over parents then gives
`$\langle\hat{u}\rangle(t)$` for every probed mode; at `$t = 0$` the
injected mode's entry **is** the injected profile, which is why no
separate input specification is needed downstream.

With a transient-growth operator bundle (``--operator``, the
``--tg.save_operator`` output) the aggregate also reports the measured
energy amplification `$E(t)/E(0)$` of the response against the
linear prediction for *this* input
(:func:`~dnsjax.analysis.response.operator_tools.
input_response_curve` seeded with the measured `$t=0$` projection)
and the optimal-growth envelope `$G(t)$`.

Direct identification
=====================
Inject each of the `$m$` leading controllability modes (one member
tree per basis index ``j``, built with ``--modes-npz ... --index j``),
aggregate each -- the flow's ensemble-averaged impulse responses to
the basis -- and feed the ``m`` response bundles to ``identify``:
the responses are projected onto the basis coordinates
(`$b_j(t) = P^H T_\mathrm{proj}\,\langle\hat{u}\rangle_j(t)$`, with
`$P$` recovered from the controllability bundle via `$P =
T_\mathrm{proj}\,\mathrm{lifted}$`), normalised by their own
`$t = 0$` coefficients, and assembled into
`$M(t_k) \approx e^{t_k L}$`; :func:`identify_generator` then fits

.. math::
    L = \frac{1}{N_\tau}\sum_i \frac{1}{\tau_i}
        \operatorname{logm} M(\tau_i),

with per-horizon branch-cut validity checks and reconstruction
residuals `$\lVert e^{\tau_i L} - M(\tau_i)\rVert_F$`.  The same
:func:`identify_generator` is the shared core for covariance-based
identification variants built on the probe stream.  Outputs include
the identified spectrum (:func:`stability_report`) and the growth
curve of `$L$` in the same energy convention as the exported
operators, comparable against
:func:`~dnsjax.analysis.response.operator_tools.restrict` of the
reference operator on the same basis.

SciPy is imported lazily (``logm``); the growth-curve outputs use the
JAX-based :mod:`.operator_tools` sweeps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .probes import read_probes

__all__ = [
    "project_series",
    "identify_generator",
    "stability_report",
    "aggregate_tree",
    "identify_from_responses",
]


# ── Identification core (shared) ─────────────────────────────────


def project_series(
    u: np.ndarray, t_proj: np.ndarray, p: np.ndarray | None = None
) -> np.ndarray:
    """Project full-state profiles onto operator (or basis) coordinates.

    *u* is ``(nt, C, Ny)`` (one probed mode's time series); the
    component-major flatten matches the operator export's index
    convention (``c*Ny + j``).  Returns ``(nt, r_res)`` energy
    coordinates, or ``(nt, m)`` basis coordinates when *p*
    (``r_res x m``, orthonormal columns) is given.
    """
    u = np.asarray(u)
    a = u.reshape(u.shape[0], -1) @ t_proj.T
    return a if p is None else a @ np.conj(p)


def identify_generator(
    pairs: list[tuple[float, np.ndarray]],
) -> tuple[np.ndarray, dict]:
    r"""Fit the generator from propagator samples
    `$M(\tau_i) \approx e^{\tau_i L}$`.

    ``L = mean_i logm(M_i)/tau_i`` (principal matrix logarithm).  A
    horizon whose `$M_i$` has an eigenvalue on (or hugging) the
    negative real axis is ambiguous under the principal branch and is
    rejected with a ``ValueError`` naming it -- shorten the horizons
    or average more members.  Returns ``(L, diagnostics)`` with
    per-horizon reconstruction residuals
    `$\lVert e^{\tau_i L} - M_i\rVert_F / \lVert M_i\rVert_F$` (large
    residuals at long horizons flag nonlinearity / noise, not a
    failure of the fit itself).
    """
    from scipy.linalg import expm, logm

    if not pairs:
        raise ValueError("no (tau, M) pairs given")
    n = pairs[0][1].shape[0]
    logs = []
    for tau, m_mat in pairs:
        m_mat = np.asarray(m_mat)
        if tau <= 0.0:
            raise ValueError(f"horizon tau = {tau:g} must be positive")
        if m_mat.shape != (n, n):
            raise ValueError(
                f"inconsistent M shapes: {m_mat.shape} vs ({n}, {n})"
            )
        eig = np.linalg.eigvals(m_mat)
        if np.min(np.abs(eig)) < 1e-14:
            raise ValueError(
                f"M(tau={tau:g}) is singular (an eigenvalue ~ 0); the "
                "response has fully decayed -- shorten the horizons"
            )
        # Principal-branch validity: arg(mu) must stay clear of +-pi.
        margin = np.pi - np.max(np.abs(np.angle(eig)))
        if margin < 0.05:
            raise ValueError(
                f"M(tau={tau:g}) has an eigenvalue within {margin:.3f} "
                "rad of the negative real axis; the principal logm "
                "branch is ambiguous -- shorten the horizons or "
                "average more members"
            )
        logs.append(logm(m_mat) / tau)
    l_mat = np.mean(logs, axis=0)

    residuals = []
    for tau, m_mat in pairs:
        recon = expm(tau * l_mat)
        residuals.append(
            float(
                np.linalg.norm(recon - m_mat)
                / max(np.linalg.norm(m_mat), 1e-300)
            )
        )
    return l_mat, {
        "horizons": [tau for tau, _ in pairs],
        "residuals": residuals,
    }


def stability_report(l_mat: np.ndarray) -> dict:
    """Spectrum summary of an identified generator."""
    eig = np.linalg.eigvals(np.asarray(l_mat))
    order = np.argsort(-eig.real)
    eig = eig[order]
    return {
        "eigvals": eig,
        "spectral_abscissa": float(eig[0].real),
        "stable": bool(eig[0].real < 0.0),
    }


# ── Aggregation ──────────────────────────────────────────────────


def _member_response(
    tree: Path, members: list[dict], pairing: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read + align + pair-combine the member probe streams.

    Returns ``(t_rel, mean_u, modes, sidecar_meta)`` with ``mean_u``
    of shape ``(nt, K, C, Ny)``.
    """
    by_dir = {m["dir"]: m for m in members}
    data = {d: read_probes(tree / d) for d in by_dir}

    ref_dir = next(iter(data))
    ref = data[ref_dir]
    t_rel = ref.t - ref.t[0]
    for d, pd in data.items():
        if pd.u.shape != ref.u.shape:
            raise SystemExit(
                f"{d}: probe shape {pd.u.shape} differs from "
                f"{ref_dir}'s {ref.u.shape}"
            )
        if not np.allclose(pd.t - pd.t[0], t_rel, rtol=0, atol=1e-10):
            raise SystemExit(
                f"{d}: relative sample times differ from {ref_dir}'s "
                "(inconsistent it_probes * dt across members?)"
            )
        if pd.modes.tolist() != ref.modes.tolist():
            raise SystemExit(f"{d}: probed modes differ from {ref_dir}'s")

    combos = []
    n_pairs = max(int(m["dir"][1:5]) for m in members) + 1
    for k in range(n_pairs):
        if pairing == "antithetic":
            combos.append(
                0.5 * (data[f"m{k:04d}_p"].u - data[f"m{k:04d}_m"].u)
            )
        elif pairing == "baseline":
            combos.append(data[f"m{k:04d}_p"].u - data[f"m{k:04d}_b"].u)
        else:
            combos.append(data[f"m{k:04d}_p"].u)
    mean_u = np.mean(combos, axis=0)
    return t_rel, mean_u, ref.modes, ref.meta


def aggregate_tree(
    tree: str | Path,
    out: str | Path,
    operator: str | Path | None = None,
) -> dict:
    """Aggregate one member tree into a response bundle (npz).

    Stores the pair-combined ensemble mean of **every** probed mode,
    the injected mode's index, and -- when *operator* (the matching
    ``<stem>_tg_op.npz``) is given -- the measured energy
    amplification alongside the linear input-response prediction and
    the optimal-growth envelope on the same time grid.
    """
    tree = Path(tree)
    with open(tree / "members.json") as f:
        spec = json.load(f)
    t_rel, mean_u, modes, meta = _member_response(
        tree, spec["members"], spec["pairing"]
    )
    i2, i3 = spec["mode"]
    inj = [tuple(m) for m in modes.tolist()].index((i2, i3))

    out_dict: dict[str, Any] = {
        "readme": (
            "dnsjax ensemble response. mean_u: (nt, K, C, Ny) "
            "pair-combined ensemble mean of the probed modes; "
            "injected_index selects the injected mode's row; "
            "mean_u[0, injected_index] is the injected profile. "
            "energy/prediction/envelope (when present): measured "
            "E(t)/E(0) vs the linear input response and the optimal "
            "envelope from the reference operator."
        ),
        "t_rel": t_rel,
        "mean_u": mean_u,
        "modes": modes,
        "injected_i2": i2,
        "injected_i3": i3,
        "injected_index": inj,
        "n_members": len(spec["members"]),
        "pairing": spec["pairing"],
        "amplitude_energy": spec["amplitude_energy"],
        "source_json": json.dumps(spec["source"], sort_keys=True),
        "basis_index": spec["source"].get("index", -1),
        "system": meta["system"],
        "members_json": json.dumps(spec, sort_keys=True, default=str),
    }

    if operator is not None:
        from .operator_tools import (
            growth_curve,
            input_response_curve,
            load_operator,
        )

        op = load_operator(operator, i2, i3)
        a_series = project_series(mean_u[:, inj], op.T_proj)
        e_meas = np.sum(np.abs(a_series) ** 2, axis=1)
        if e_meas[0] <= 0.0:
            raise SystemExit(
                "the t = 0 response projects to zero energy; wrong "
                "operator bundle or an unprobed injection?"
            )
        out_dict["energy"] = e_meas / e_meas[0]
        out_dict["prediction"] = input_response_curve(op.A, a_series[0], t_rel)
        out_dict["envelope"] = growth_curve(op.A, t_rel)
        out_dict["operator_file"] = str(operator)

    np.savez(out, **out_dict)
    print(
        f"[ensemble] wrote {out} ({out_dict['n_members']} members, "
        f"pairing {spec['pairing']}, injected mode ({i2},{i3}))."
    )
    if "energy" in out_dict:
        k = int(np.argmax(out_dict["energy"]))
        print(
            f"[ensemble] measured peak E/E0 = "
            f"{out_dict['energy'][k]:.4g} at t = {t_rel[k]:g}; "
            f"predicted {out_dict['prediction'][k]:.4g} there "
            f"(envelope {out_dict['envelope'][k]:.4g})."
        )
    return out_dict


# ── Direct identification ────────────────────────────────────────


def identify_from_responses(
    response_files: list[str | Path],
    operator: str | Path,
    modes_npz: str | Path,
    horizons: list[float],
) -> dict:
    r"""Identify `$L$` from the responses to an injected basis.

    *response_files*: one ``aggregate_tree`` bundle per injected
    controllability-mode index ``j = 0..m-1`` (matched on their
    recorded ``basis_index``; all for the same ``(i2, i3)``).
    *modes_npz*: the controllability bundle they were injected from.
    *horizons*: fit times (each snapped to the nearest probe sample).
    Use several, spread over the window where the response is still
    linear but not yet decayed: short horizons amplify ensemble noise
    (the ``logm`` fit divides by `$\tau$`), long ones pick up
    nonlinearity and eventually the decay / branch-cut rejections of
    :func:`identify_generator`; the per-horizon residuals show which
    end is failing, so widen or trim the list accordingly.
    """
    op = None
    responses: dict[int, dict] = {}
    i2 = i3 = None
    for path in response_files:
        with np.load(path, allow_pickle=False) as z:
            j = int(z["basis_index"])
            if j < 0:
                raise SystemExit(
                    f"{path} was not built from a --modes-npz basis "
                    "injection (basis_index missing)"
                )
            if j in responses:
                raise SystemExit(f"duplicate basis index {j} ({path})")
            responses[j] = {
                "t": np.asarray(z["t_rel"]),
                "u": np.asarray(z["mean_u"])[:, int(z["injected_index"])],
            }
            if i2 is None:
                i2, i3 = int(z["injected_i2"]), int(z["injected_i3"])
            elif (i2, i3) != (int(z["injected_i2"]), int(z["injected_i3"])):
                raise SystemExit(
                    "response bundles mix injected modes "
                    f"({(i2, i3)} vs {path})"
                )
    m = len(responses)
    if sorted(responses) != list(range(m)):
        raise SystemExit(
            f"basis indices {sorted(responses)} are not 0..{m - 1}"
        )

    from .operator_tools import load_modes_npz, load_operator, recover_basis

    op = load_operator(operator, i2, i3)
    try:
        p = recover_basis(op, load_modes_npz(modes_npz, i2, i3, m))
    except (KeyError, ValueError) as e:
        raise SystemExit(str(e)) from e

    t_ref = responses[0]["t"]
    b_mat = np.stack(
        [project_series(responses[j]["u"], op.T_proj, p) for j in range(m)],
        axis=2,
    )  # (nt, m coords, m inputs)

    # Normalise each column by its own t=0 coefficient; the t=0
    # cross-coefficients must vanish (orthonormal injections).
    scales = np.array([b_mat[0, j, j] for j in range(m)])
    if np.min(np.abs(scales)) == 0.0:
        raise SystemExit("a response has zero t=0 coefficient")
    off = b_mat[0] / np.abs(scales)[None, :] - np.diag(scales / np.abs(scales))
    if np.max(np.abs(off)) > 1e-6:
        raise SystemExit(
            "t=0 responses are not the injected orthonormal basis "
            f"(max off-diagonal {np.max(np.abs(off)):.2e})"
        )
    b_mat = b_mat / scales[None, None, :]

    pairs = []
    used = []
    for tau in horizons:
        k = int(np.argmin(np.abs(t_ref - tau)))
        if k == 0:
            raise SystemExit(f"horizon {tau:g} snaps to t = 0")
        pairs.append((float(t_ref[k]), b_mat[k]))
        used.append(float(t_ref[k]))
    l_mat, diag = identify_generator(pairs)
    report = stability_report(l_mat)
    return {
        "L": l_mat,
        "eigvals": report["eigvals"],
        "spectral_abscissa": report["spectral_abscissa"],
        "stable": report["stable"],
        "horizons": np.asarray(used),
        "residuals": np.asarray(diag["residuals"]),
        "basis": p,
        "op": op,
        "t_ref": t_ref,
    }


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from ...bootstrap import configure_jax_platform

    p = argparse.ArgumentParser(
        prog="python -m dnsjax.analysis.response.ensemble",
        description="Aggregate ensemble member trees / identify the "
        "response operator (see the module docstring).",
        allow_abbrev=False,
    )
    p.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=("cpu", "cuda", "rocm", "tpu"),
        help="JAX backend for the growth-curve sweeps",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("aggregate", help="member tree -> response npz")
    pa.add_argument("--tree", required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument(
        "--operator",
        default=None,
        help="matching _tg_op.npz: adds energy/prediction/envelope",
    )

    pi = sub.add_parser("identify", help="basis-response npzs -> identified L")
    pi.add_argument(
        "--responses",
        nargs="+",
        required=True,
        help="aggregate outputs, one per injected basis index",
    )
    pi.add_argument("--operator", required=True)
    pi.add_argument("--modes-npz", required=True)
    pi.add_argument(
        "--horizons",
        required=True,
        help='comma list of fit times, e.g. "1,2,4" (several, inside '
        "the linear window; see identify_from_responses)",
    )
    pi.add_argument("--out", required=True)

    args = p.parse_args(argv)
    configure_jax_platform(args.platform, double_precision=True)

    if args.command == "aggregate":
        aggregate_tree(args.tree, args.out, args.operator)
        return 0

    horizons = [float(tok) for tok in args.horizons.split(",")]
    result = identify_from_responses(
        args.responses, args.operator, args.modes_npz, horizons
    )
    from .operator_tools import growth_curve, restrict

    op = result.pop("op")
    t_ref = result.pop("t_ref")
    basis = result["basis"]
    g_id = growth_curve(result["L"], t_ref)
    g_ref = growth_curve(restrict(op.A, basis), t_ref)
    np.savez(
        args.out,
        readme=(
            "dnsjax direct operator identification. L: identified "
            "generator on the injected controllability basis "
            "(energy-orthonormal coordinates); G_id/G_ref: growth "
            "curves of L and of the reference operator restricted "
            "to the same basis, on t_grid."
        ),
        t_grid=t_ref,
        G_id=g_id,
        G_ref=g_ref,
        **result,
    )
    print(
        f"[ensemble] wrote {args.out}: spectral abscissa "
        f"{result['spectral_abscissa']:+.4e} "
        f"({'stable' if result['stable'] else 'UNSTABLE'}), "
        f"max |G_id - G_ref|/G_ref = "
        f"{float(np.max(np.abs(g_id - g_ref) / g_ref)):.3g}, "
        f"horizon residuals {np.round(result['residuals'], 4)}."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
