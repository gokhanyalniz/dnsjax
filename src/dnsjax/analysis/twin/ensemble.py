r"""Twin-run ensemble aggregation and growth-rate fits (JAX-free).

A twin ensemble is a member tree built by
``scripts/ensemble_setup.py build-twin``: member directories under a
root, indexed by ``members.json`` (``kind: "twin"``), each holding
the ``twin.dat`` (+ optional ``twin_budget.dat``) streams of one
``dnsjax-twin`` run.  :func:`aggregate_members` stacks every column
on the shared relative-time grid `$t - t_\mathrm{parent}$` (members
start from different parent snapshots, so absolute times differ) and
returns per-column stacks with ensemble mean and standard deviation
(``ddof = 0``, the member spread -- see :func:`aggregate_members` for
the standard-error conversion) -- the inputs of the paper's figures
(Egerique-de-la-Concha & Hwang, *J. Fluid Mech.* **1036**, A52, 2026).
Each member is restricted to its own cadence grid first, so a resumed
member stacks against a fresh one (:func:`_grid_mask`).

Growth-rate fits (least squares over a caller-chosen window):

- :func:`fit_exponential_rate` -- the leading Lyapunov exponent from
  the short-term phase via
  `$\lambda = \tfrac{1}{2}\,\mathrm{d}\log E_\Delta/\mathrm{d}t$`
  (the paper's eq. 3.1: `$E_\Delta \sim e^{2\lambda t}$` because the
  energy is quadratic in the perturbation).
- :func:`fit_linear_rate` -- the algebraic-phase slope
  `$\mathrm{d}E/\mathrm{d}t$`.

CLI: ``python -m dnsjax.analysis.twin.ensemble --tree T --out A.npz``
writes the aggregate bundle (per-column ``stack_*`` / ``mean_*`` /
``std_*`` arrays plus ``t_rel``, ``columns``, and the members.json
provenance).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .series import read_twin, uniform_grid

#: Relative-time alignment tolerance across members (seconds of
#: simulation time; the grids come from one shared dt and cadence).
_T_ATOL = 1e-9


def _grid_mask(t: np.ndarray) -> np.ndarray:
    """One member's on-cadence rows (everything, when too short).

    A stream carries a few rows off its own cadence grid -- the
    driver's unconditional final row always, a resume seam's ``t0``
    row when the snapshot cadence is not a multiple of the sample one
    (:mod:`dnsjax.analysis.twin.series`).  Which of them a member has
    depends on whether and where it was resumed, so members that are
    otherwise identical would not stack; selecting each member's grid
    first is what makes them comparable again.
    """
    try:
        return uniform_grid(t)[1]
    except ValueError:
        return np.ones(t.shape, dtype=bool)


def _stack_group(
    label: str,
    per_member: list[tuple[str, np.ndarray, dict[str, np.ndarray]]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Stack one stream group over members, guarding mismatches.

    *per_member* holds ``(member dir, relative times, columns)``
    triples (``t`` excluded from the columns).  Each member is first
    restricted to its own cadence grid (:func:`_grid_mask`); all must
    then share the column set and the relative-time grid.
    """
    _, t_rel, first_cols = per_member[0]
    keep0 = _grid_mask(t_rel)
    t_rel = t_rel[keep0]
    names = set(first_cols)
    stacks: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for member, rel, columns in per_member:
        if set(columns) != names:
            raise ValueError(
                f"{member}: {label} column set differs from the first member's"
            )
        keep = _grid_mask(rel)
        rel = rel[keep]
        if rel.shape != t_rel.shape or not np.allclose(
            rel, t_rel, rtol=0, atol=_T_ATOL
        ):
            raise ValueError(
                f"{member}: {label} relative time grid differs from "
                "the first member's (different horizon, cadence, or "
                "an incomplete run?)"
            )
        for name in names:
            stacks[name].append(columns[name][keep])
    return t_rel, {name: np.stack(vals) for name, vals in stacks.items()}


def aggregate_members(tree: str | Path, out: str | Path | None = None) -> dict:
    r"""Aggregate a twin member tree; optionally write an ``.npz``.

    Returns ``t_rel`` plus ``stack_<c>`` / ``mean_<c>`` / ``std_<c>``
    for every ``twin.dat`` column ``<c>``; when every member carries
    a budget stream, also ``t_rel_budget`` and the same triple per
    budget column under ``budget_<c>``.  ``columns`` lists the
    aggregated names and ``members_json`` carries the tree's
    provenance verbatim.

    ``std_*`` is NumPy's default **population** standard deviation
    (``ddof = 0``) -- the spread of the members themselves, which is
    the quantity a member-scatter band plots.  It is *not* an
    uncertainty on ``mean_*``: for that, take the standard error
    `$\sigma_{\bar{x}} = \mathrm{std}/\sqrt{N-1}$` (the
    `$\sqrt{N/(N-1)}$` bias correction and the `$1/\sqrt{N}$`
    cancel to this), with ``N = n_members``.  The distinction is 5 %
    at ten members, so state which one a figure shows.
    """
    tree = Path(tree)
    with open(tree / "members.json") as fh:
        spec = json.load(fh)
    if spec.get("kind") != "twin":
        raise ValueError(
            f"{tree}/members.json is not a twin tree "
            f"(kind = {spec.get('kind')!r}); response trees aggregate "
            "with dnsjax.analysis.response.ensemble."
        )
    members = spec["members"]
    if not members:
        raise ValueError("members.json lists no members")

    energy_rows = []
    budget_rows = []
    for record in members:
        series = read_twin(tree / record["dir"])
        t0 = series.t[0] if series.meta is None else series.meta["parent_t"]
        cols = {n: v for n, v in series.energies.items() if n != "t"}
        energy_rows.append((record["dir"], series.t - t0, cols))
        if series.budget is not None:
            bcols = {n: v for n, v in series.budget.items() if n != "t"}
            budget_rows.append((record["dir"], series.budget["t"] - t0, bcols))
    if budget_rows and len(budget_rows) != len(members):
        raise ValueError(
            "some members carry twin_budget.dat and some do not; "
            "the tree is inconsistent."
        )

    t_rel, stacks = _stack_group("energy", energy_rows)
    bundle: dict = {
        "t_rel": t_rel,
        "n_members": len(members),
        "members_json": json.dumps(spec),
    }
    names = sorted(stacks)
    for name, arr in stacks.items():
        bundle[f"stack_{name}"] = arr
        bundle[f"mean_{name}"] = arr.mean(axis=0)
        bundle[f"std_{name}"] = arr.std(axis=0)
    if budget_rows:
        t_rel_b, bstacks = _stack_group("budget", budget_rows)
        bundle["t_rel_budget"] = t_rel_b
        for name, arr in bstacks.items():
            bundle[f"stack_budget_{name}"] = arr
            bundle[f"mean_budget_{name}"] = arr.mean(axis=0)
            bundle[f"std_budget_{name}"] = arr.std(axis=0)
        names += [f"budget_{n}" for n in sorted(bstacks)]
    # Unicode, never ``dtype=object``: an object array would make the
    # whole bundle unreadable under ``np.load(allow_pickle=False)``,
    # the setting every reader in this package opens with.
    bundle["columns"] = np.array(names)
    if out is not None:
        np.savez_compressed(out, **bundle)
    return bundle


def _window(
    t: np.ndarray, e: np.ndarray, t_min: float, t_max: float
) -> tuple[np.ndarray, np.ndarray]:
    sel = (t >= t_min) & (t <= t_max)
    if sel.sum() < 3:
        raise ValueError(
            f"fewer than 3 samples in the fit window [{t_min}, {t_max}]"
        )
    return t[sel], e[sel]


def fit_exponential_rate(
    t: np.ndarray,
    e: np.ndarray,
    t_min: float,
    t_max: float,
) -> tuple[float, float, float]:
    r"""Fit `$E(t) = E_0 e^{2\lambda t}$` over ``[t_min, t_max]``.

    Least squares on `$\log E$`; returns ``(lam, e0_fit, rms)`` with
    ``lam`` the *Lyapunov* rate (half the log-energy slope, the
    paper's eq. 3.1), ``e0_fit`` the fitted `$E$` at ``t = 0`` of the
    given axis, and ``rms`` the log-residual RMS (fit-quality gauge:
    it grows when the window leaks into the transient or algebraic
    phases).
    """
    tw, ew = _window(t, e, t_min, t_max)
    if (ew <= 0).any():
        raise ValueError("non-positive energies in the fit window")
    slope, intercept = np.polyfit(tw, np.log(ew), 1)
    resid = np.log(ew) - (slope * tw + intercept)
    return (
        float(slope / 2.0),
        float(np.exp(intercept)),
        float(np.sqrt(np.mean(resid**2))),
    )


def fit_linear_rate(
    t: np.ndarray,
    e: np.ndarray,
    t_min: float,
    t_max: float,
) -> tuple[float, float, float]:
    r"""Fit `$E(t) = a + r\,t$` over ``[t_min, t_max]``.

    Returns ``(r, a, rms)`` -- the algebraic-phase growth rate, the
    intercept, and the residual RMS (in energy units).
    """
    tw, ew = _window(t, e, t_min, t_max)
    rate, intercept = np.polyfit(tw, ew, 1)
    resid = ew - (rate * tw + intercept)
    return (
        float(rate),
        float(intercept),
        float(np.sqrt(np.mean(resid**2))),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dnsjax.analysis.twin.ensemble",
        description="Aggregate a twin member tree (see the module docstring).",
        allow_abbrev=False,
    )
    parser.add_argument("--tree", required=True, help="member tree root")
    parser.add_argument("--out", required=True, help="output .npz path")
    args = parser.parse_args(argv)
    bundle = aggregate_members(args.tree, args.out)
    print(
        f"[twin-ensemble] {bundle['n_members']} members, "
        f"{bundle['t_rel'].shape[0]} samples -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
