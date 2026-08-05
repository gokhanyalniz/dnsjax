r"""Readers for the twin driver's scalar streams (JAX-free).

The ``.dat`` streams are the whitespace-aligned text format of
``dnsjax.__main__`` (header row of column names, one row per sample,
column order = the writer dict's *sorted* keys): parse by **name**,
never by position.  A resumed member's stream duplicates one sample
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
#: (``dnsjax.twin.TWIN_FORMAT_VERSION`` is the writer's).
MIN_FORMAT_VERSION: int = 1


def read_dat(path: str | Path) -> dict[str, np.ndarray]:
    """Read one ``.dat`` stream into ``{column name: values}``.

    The first column is always ``t``.  Rows are returned as written
    (including any resume-seam duplicates); shape ``(n_rows,)`` per
    column.
    """
    path = Path(path)
    with open(path) as fh:
        header = fh.readline().split()
    data = np.loadtxt(path, skiprows=1, ndmin=2)
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
