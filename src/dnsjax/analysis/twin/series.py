r"""Readers for the twin driver's scalar streams (JAX-free).

The ``.dat`` streams are the whitespace-aligned text format of
``dnsjax.__main__`` (``#``-commented header row of column names, one
row per sample, column order = the writer dict's *sorted* keys):
parse by **name**, never by position.  Because the header is a
comment, the rows load under a default-flag :func:`numpy.loadtxt`.
A resumed member's stream duplicates one sample
per resume seam (the parent segment's final row and the child's
``t0`` row hold the same state at the same ``t``); :func:`read_twin`
drops the duplicates, keeping the first occurrence -- the probe
reader's convention.

``twin.json`` is the member record the driver writes at the fresh
start (seed, ``e0``, parent snapshot and clock, cadences, git hash,
resolved parameter dump); its ``format_version`` floor here is
:data:`MIN_FORMAT_VERSION`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Oldest ``twin.json`` schema this reader understands
#: (``dnsjax.twin.driver.TWIN_FORMAT_VERSION`` is the writer's).
MIN_FORMAT_VERSION: int = 1


def read_dat(path: str | Path) -> dict[str, np.ndarray]:
    """Read one ``.dat`` stream into ``{column name: values}``.

    The first column is always ``t``.  Rows are returned as written
    (including any resume-seam duplicates); shape ``(n_rows,)`` per
    column.  The names come off the header line with its ``#``
    stripped; the rows need no ``skiprows`` because
    :func:`numpy.loadtxt` drops that line as a comment.
    """
    path = Path(path)
    with open(path) as fh:
        header = fh.readline().lstrip("#").split()
    data = np.loadtxt(path, ndmin=2)
    if data.size == 0:
        return {name: np.empty(0) for name in header}
    if data.shape[1] != len(header):
        raise ValueError(
            f"{path}: {data.shape[1]} columns but {len(header)} header "
            "names; the file is truncated or not a dnsjax .dat stream."
        )
    return {name: data[:, i] for i, name in enumerate(header)}


def _drop_seam_duplicates(
    columns: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Drop later rows whose ``t`` repeats an earlier one exactly."""
    t = columns["t"]
    _, keep = np.unique(t, return_index=True)
    keep.sort()
    if len(keep) == len(t):
        return columns
    return {name: vals[keep] for name, vals in columns.items()}


@dataclass(frozen=True)
class TwinSeries:
    """One member directory's twin streams.

    ``energies`` are the ``twin.dat`` columns and ``budget`` the
    ``twin_budget.dat`` ones (``None`` when the stream was disabled),
    both seam-deduplicated, with ``t`` inside each dict.  ``meta`` is
    the parsed ``twin.json`` (``None`` when absent -- e.g. a stream
    pair copied without its member record).  ``t_rel`` is the time
    since the perturbation, ``t - meta["parent_t"]`` (falling back to
    the first sample when ``meta`` is missing).
    """

    path: Path
    energies: dict[str, np.ndarray]
    budget: dict[str, np.ndarray] | None
    meta: dict | None

    @property
    def t(self) -> np.ndarray:
        return self.energies["t"]

    @property
    def t_rel(self) -> np.ndarray:
        t0 = (
            float(self.meta["parent_t"])
            if self.meta is not None
            else float(self.t[0])
        )
        return self.t - t0


def read_twin(directory: str | Path = ".") -> TwinSeries:
    """Read a member directory's ``twin.dat`` (+ budget + record)."""
    directory = Path(directory)
    dat = directory / "twin.dat"
    if not dat.is_file():
        raise FileNotFoundError(f"no twin.dat in {directory}")
    energies = _drop_seam_duplicates(read_dat(dat))

    budget = None
    budget_path = directory / "twin_budget.dat"
    if budget_path.is_file():
        budget = _drop_seam_duplicates(read_dat(budget_path))

    meta = None
    meta_path = directory / "twin.json"
    if meta_path.is_file():
        with open(meta_path) as fh:
            meta = json.load(fh)
        version = int(meta.get("format_version", 0))
        if version < MIN_FORMAT_VERSION:
            raise ValueError(
                f"{meta_path}: format_version {version} predates the "
                f"reader floor {MIN_FORMAT_VERSION}; re-run the member "
                "with the current driver."
            )
    return TwinSeries(
        path=directory, energies=energies, budget=budget, meta=meta
    )


def budget_sums(budget: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r"""Per-component sums of the budget columns.

    Returns ``P_<x>`` / ``T_<x>`` for ``x`` in ``dU`` / ``du1`` /
    ``du2`` (the individual `$-\langle a\cdot(b\cdot\nabla)c\rangle$`
    columns grouped by their ``a`` slot), alongside the stream's own
    ``eps_*`` and ``*_tot`` columns, so the per-component balance
    `$\partial_t E_X = P_X + T_X - \epsilon_X$` is directly
    plottable.
    """
    out: dict[str, np.ndarray] = {"t": budget["t"]}
    for x in ("dU", "du1", "du2"):
        for kind in ("P", "T"):
            cols = [n for n in budget if n.startswith(f"{kind}_{x}(")]
            if not cols:
                raise ValueError(
                    f"no {kind}_{x}(...) columns in the budget stream"
                )
            out[f"{kind}_{x}"] = sum(budget[n] for n in cols)
        out[f"eps_{x}"] = budget[f"eps_{x}"]
    for name in ("P_tot", "T_tot", "eps_tot"):
        out[name] = budget[name]
    return out


#: Per-component ``(production, transport)`` term counts in the budget
#: stream, keyed by the ``a`` slot of the
#: `$-\langle a\cdot(b\cdot\nabla)c\rangle$` triples (``_PRODUCTION``
#: / ``_TRANSPORT`` in :mod:`dnsjax.twin.diagnostics`).
_TERM_COUNTS: dict[str, tuple[int, int]] = {
    "dU": (3, 4),
    "du1": (4, 4),
    "du2": (5, 4),
}


@dataclass(frozen=True)
class ClosureResiduals:
    r"""Relative budget-closure residuals of one member.

    ``components`` holds the three per-component residuals of
    `$\partial_t E_X = P_X + T_X - \epsilon_X$` (``dU`` / ``du1`` /
    ``du2``) plus ``T_tot``.  Each is a maximum over the sample
    times, normalised to read as a fraction: ``0`` is exact closure,
    ``1`` a residual as large as the budget itself.  ``n_samples``
    is how many budget rows contributed, ``dt`` the ``twin.dat``
    sample spacing the derivative used.  Indexing delegates to
    ``components``, so ``res["du1"]`` works.
    """

    components: dict[str, float]
    n_samples: int
    dt: float

    def __getitem__(self, key: str) -> float:
        return self.components[key]


def _uniform_spacing(t: np.ndarray) -> float:
    """The uniform sample spacing of *t*, or raise."""
    if t.size < 3:
        raise ValueError(
            f"need at least 3 twin.dat samples to difference, got {t.size}"
        )
    steps = np.diff(t)
    dt = float(np.mean(steps))
    if dt <= 0 or np.max(np.abs(steps - dt)) > 1e-6 * dt:
        raise ValueError(
            "twin.dat sample times are not uniformly spaced "
            f"(spread {float(np.max(np.abs(steps - dt))):.3e} about "
            f"dt = {dt:.6g}); the centred difference below assumes a "
            "single fixed cadence"
        )
    return dt


def closure_residuals(series: TwinSeries) -> ClosureResiduals:
    r"""Budget-closure residuals of a member's twin streams.

    Per component `$X$`, the centred-difference `$\partial_t E_X$`
    from ``twin.dat`` is compared against `$P_X + T_X - \epsilon_X$`
    from ``twin_budget.dat`` at every budget sample time that has an
    energy sample on both sides, and the largest absolute mismatch is
    normalised by the largest magnitude either side reaches.
    ``T_tot`` -- which cancels pairwise by parts, so vanishes
    continuously rather than balancing anything -- is normalised by
    the largest individual transport term instead.

    What remains is discrete truncation error (pressure work against
    the interior divergence residual, the FD integration-by-parts
    defect of the wall-normal transport) plus the `$O(\Delta t^2)$`
    stepping error; the dissipation is evaluated in the operator form
    the implicit viscous update actually applies, so the viscous part
    closes exactly (:mod:`dnsjax.twin.diagnostics`, "Dissipation
    form").  The residuals therefore **converge under refinement**,
    which is the property worth asserting of them -- a missing or
    mis-signed term would not.

    The derivative uses the ``twin.dat`` spacing rather than the run's
    ``step.dt``, so it is correct at any ``twin.it_energy``.  Raises
    :class:`ValueError` when the budget stream is absent, is
    structurally not a twin budget, has a non-uniform energy cadence,
    or leaves no interior sample.
    """
    if series.budget is None:
        raise ValueError(
            f"{series.path}: no twin_budget.dat; re-run the member with "
            "twin.it_budget set"
        )
    energies, budget = series.energies, series.budget

    p_all = [n for n in budget if n.startswith("P_") and n != "P_tot"]
    t_all = [n for n in budget if n.startswith("T_") and n != "T_tot"]
    if len(p_all) != 12 or len(t_all) != 12:
        raise ValueError(
            f"{series.path}: expected 12 production and 12 transport "
            f"columns, found {len(p_all)} and {len(t_all)}"
        )

    t_e = energies["t"]
    dt = _uniform_spacing(t_e)
    t_b = budget["t"]
    # Budget rows land on the energy grid by construction (both are
    # sampled before the same step), except the driver's unconditional
    # final row, which can fall off a cadence.  Map by grid index and
    # verify, so an off-grid row is skipped rather than mis-paired.
    idx = np.rint((t_b - t_e[0]) / dt).astype(int)
    on_grid = (
        (idx > 0)
        & (idx + 1 < t_e.size)
        & (np.abs(t_e[np.clip(idx, 0, t_e.size - 1)] - t_b) < 1e-6 * dt)
    )
    if not on_grid.any():
        raise ValueError(
            f"{series.path}: no budget sample has an energy sample on "
            "both sides; the two cadences do not overlap"
        )
    k = np.flatnonzero(on_grid)
    i = idx[k]

    out: dict[str, float] = {}
    for x, (n_p, n_t) in _TERM_COUNTS.items():
        p_cols = [n for n in budget if n.startswith(f"P_{x}(")]
        t_cols = [n for n in budget if n.startswith(f"T_{x}(")]
        if len(p_cols) != n_p or len(t_cols) != n_t:
            raise ValueError(
                f"{series.path}: component {x} has {len(p_cols)}/"
                f"{len(t_cols)} production/transport columns, "
                f"expected {n_p}/{n_t}"
            )
        dedt = (energies[f"E_{x}"][i + 1] - energies[f"E_{x}"][i - 1]) / (
            2 * dt
        )
        rhs = (
            sum(budget[n][k] for n in p_cols)
            + sum(budget[n][k] for n in t_cols)
            - budget[f"eps_{x}"][k]
        )
        scale = float(max(np.abs(dedt).max(), np.abs(rhs).max()))
        out[x] = (
            0.0 if scale == 0.0 else float(np.abs(dedt - rhs).max() / scale)
        )

    t_scale = float(np.abs(np.stack([budget[n] for n in t_all])).max())
    out["T_tot"] = (
        0.0
        if t_scale == 0.0
        else float(np.abs(budget["T_tot"]).max() / t_scale)
    )
    return ClosureResiduals(components=out, n_samples=int(k.size), dt=dt)
