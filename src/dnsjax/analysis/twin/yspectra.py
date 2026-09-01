r"""JAX-free readers for the wall-normal-resolved twin streams.

``twin_yspectra.bin`` and ``twin_ybudget.bin``, written by
:mod:`dnsjax.twin.yspectra`.  Both come back as a
:class:`YResolvedData`: a dict of ``(n_t, ..., n_y, n_k)`` float64
arrays, the two wavenumber axes in physical units, the wall-normal
grid and its quadrature weights, and the sidecar.

Everything stored is a `$y$`-**density** already divided by
``volume_fac``.  :func:`integrate_y` contracts one with the weights
to give the per-`$k$` quantity; summing *that* over `$k$` gives the
matching volume-averaged rate.  Both wavenumber axes are one-sided
(`$|k_z|$` folded), so those sums run over the stored axis with no
further weighting.

For the energies that rate is ``twin.dat``'s ``E_d`` exactly.  For
the budget it is a ``twin_budget.dat`` column only where the two
regroup the same Parseval sum, and which columns those are depends on
the form the stream was written in -- the sidecar's ``terms`` names it
(:mod:`dnsjax.twin.diagnostics`, "Two budget forms").  In the
**default convective** form ``-V`` gives ``eps_tot`` and
``P_U + P_r`` gives ``P_tot``, both exactly, while the transfer terms
match their per-bin counterparts only up to the discrete
integration-by-parts residual that makes ``T_tot`` nonzero.  Under
``twin.rotational_ybudget`` the production identity moves to
``P_lift`` (the three mean-gradient production columns) and the rest
of the split differs from ``twin_budget.dat`` by the work of a
gradient -- zero in total, not per `$y$`.

:func:`bin_energies` is the bridge back to the three-bin diagnostics
of Egerique-de-la-Concha & Hwang (*J. Fluid Mech.* **1036**, A52,
2026): the `$k_x = 0$` plane the stream carries alongside the two
marginals is exactly the spectrum of the streamwise-averaged
difference field, so

.. math::
    E_{\Delta U} = \textstyle\int \sum_\alpha e^{x0}_\alpha(y, 0),
    \quad
    E_{\Delta u_1} = \int \sum_\alpha \sum_{k_z > 0} e^{x0}_\alpha ,
    \quad
    E_{\Delta u_2} = \int \sum_\alpha \sum_{k_z}
        \bigl(e^{x}_\alpha - e^{x0}_\alpha\bigr) ,

now resolved in `$k_z$` rather than collapsed to three numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Reader floors (raised with the writers' ``*_FORMAT_VERSION``).
MIN_YSPECTRA_VERSION: int = 1
MIN_YBUDGET_VERSION: int = 2


@dataclass(frozen=True)
class YResolvedData:
    r"""One wall-normal-resolved stream.

    ``fields`` maps a stored name (``e_x``, ``P_U_z``, ...) to its
    ``(n_t, ...)`` array; ``y`` / ``y_weights`` are the wall-normal
    grid and its quadrature rule; ``kz`` / ``kx`` the one-sided
    physical wavenumbers of the ``*_x`` / ``*_x0`` and ``*_z``
    arrays.
    """

    t: np.ndarray
    fields: dict[str, np.ndarray]
    y: np.ndarray
    y_weights: np.ndarray
    kz: np.ndarray
    kx: np.ndarray
    meta: dict

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[name]


def _resolve_pair(path: str | Path, stem: str) -> tuple[Path, Path]:
    path = Path(path)
    if path.is_dir():
        return path / f"{stem}.bin", path / f"{stem}.json"
    if path.suffix == ".json":
        return path.with_suffix(".bin"), path
    return path, path.with_suffix(".json")


def _read(
    path: str | Path,
    stem: str,
    floor: int,
    field_shapes,
) -> YResolvedData:
    """Shared body: sidecar, record dtype, truncation, seam drop."""
    bin_path, json_path = _resolve_pair(path, stem)
    if not json_path.is_file():
        raise FileNotFoundError(f"no sidecar {json_path}")
    with open(json_path) as fh:
        meta = json.load(fh)
    version = int(meta.get("format_version", 0))
    if version < floor:
        raise ValueError(
            f"{json_path}: format_version {version} predates the "
            f"reader floor {floor}; re-run with the current writer."
        )

    ny = int(meta["ny"])
    n_kz, n_kx = int(meta["n_kz"]), int(meta["n_kx"])
    value_dtype = meta["value_dtype"]
    names = field_shapes(meta, ny, n_kz, n_kx)
    record_dtype = np.dtype(
        [("t", "<f8")] + [(n, value_dtype, sh) for n, sh in names]
    )

    raw = np.fromfile(bin_path, dtype=np.uint8)
    n_records = raw.size // record_dtype.itemsize
    if n_records == 0:
        raise ValueError(f"{bin_path}: no complete records")
    if raw.size % record_dtype.itemsize:
        # A kill mid-write leaves a partial trailing record; the
        # complete prefix is intact (append-only + fsync per flush).
        raw = raw[: n_records * record_dtype.itemsize]
    records = raw.view(record_dtype)

    t = records["t"].astype(np.float64)
    keep = np.sort(np.unique(t, return_index=True)[1])
    records, t = records[keep], t[keep]

    kz = (2.0 * np.pi / float(meta["lz"])) * np.asarray(
        meta["kz_harmonics"], dtype=np.float64
    )
    kx = (2.0 * np.pi / float(meta["lx"])) * np.asarray(
        meta["kx_harmonics"], dtype=np.float64
    )
    if kz.shape[0] != n_kz or kx.shape[0] != n_kx:
        raise ValueError(
            f"{json_path}: harmonic lists ({kz.shape[0]}, "
            f"{kx.shape[0]}) do not match the bin counts "
            f"({n_kz}, {n_kx})."
        )
    y = np.asarray(meta["y"], dtype=np.float64)
    w = np.asarray(meta["y_weights"], dtype=np.float64)
    if y.shape[0] != ny or w.shape[0] != ny:
        raise ValueError(
            f"{json_path}: y / y_weights lengths ({y.shape[0]}, "
            f"{w.shape[0]}) do not match ny = {ny}."
        )
    return YResolvedData(
        t=t,
        fields={n: records[n].astype(np.float64) for n, _ in names},
        y=y,
        y_weights=w,
        kz=kz,
        kx=kx,
        meta=meta,
    )


def read_twin_yspectra(path: str | Path = ".") -> YResolvedData:
    """Read ``twin_yspectra`` (a run directory, the ``.bin``, or the
    ``.json``).  Fields ``e_x`` / ``e_z`` / ``e_x0``, each
    ``(n_t, 3, n_y, n_k)``, plus the ``r_*`` triplet when the run set
    ``twin.spectra_ref``."""

    def shapes(meta, ny, n_kz, n_kx):
        prefixes = ("e", "r") if bool(meta["includes_ref"]) else ("e",)
        return [
            (f"{p}_{suf}", (3, ny, n))
            for p in prefixes
            for suf, n in (("x", n_kz), ("z", n_kx), ("x0", n_kz))
        ]

    return _read(path, "twin_yspectra", MIN_YSPECTRA_VERSION, shapes)


def read_twin_ybudget(path: str | Path = ".") -> YResolvedData:
    """Read ``twin_ybudget``.  Fields ``<term>_x`` / ``_z`` / ``_x0``
    for each name in the sidecar's ``terms``, each
    ``(n_t, n_y, n_k)``."""

    def shapes(meta, ny, n_kz, n_kx):
        return [
            (f"{term}_{suf}", (ny, n))
            for term in meta["terms"]
            for suf, n in (("x", n_kz), ("z", n_kx), ("x0", n_kz))
        ]

    return _read(path, "twin_ybudget", MIN_YBUDGET_VERSION, shapes)


def integrate_y(data: YResolvedData, name: str) -> np.ndarray:
    r"""Contract a stored density with the wall-normal quadrature.

    Returns ``(n_t, ..., n_k)``: the per-`$k$` energy or budget rate.
    Summing the result over its last axis gives the corresponding
    ``twin.dat`` / ``twin_budget.dat`` scalar (for the energies, also
    sum over the component axis).
    """
    return np.einsum("j,...jk->...k", data.y_weights, data[name])


def fluctuation_energy(
    marginal: np.ndarray,
    mean_plane: np.ndarray,
    y_weights: np.ndarray,
) -> np.ndarray:
    r"""Total-in-`$(y, k)$` energy without the `$(0, 0)$` mode.

    Array-level, so a memory-mapped reader can share the definition
    with :class:`YResolvedData` (``scripts/twin_spectral_maps.py``
    does).  *marginal* is a **complete** marginal -- ``r_x`` / ``e_x``
    (summed over `$k_x$`) or ``r_z`` / ``e_z`` (summed over `$k_z$`)
    -- and *mean_plane* the `$k_x = 0$` plane ``*_x0`` of the same
    field, both ``(..., n_y, n_k)`` with the leading axes free; the
    result is ``(...)``.

    Either marginal gives the same total, since each already sums the
    other axis, so passing ``r_z`` and ``r_x0`` together is a genuine
    cross-check rather than a second reading of the same numbers.
    Index 0 of a ``*_x0`` axis is `$k_z = 0$` at `$k_x = 0$`, i.e.
    the mean mode alone -- which is why the subtraction needs that
    plane and not ``*_x[..., 0]``, the whole `$k_z = 0$` column.

    For a reference field ``r_*`` the result is the fluctuation
    energy about the wall-parallel mean: the stored spectra are of
    the perturbation about the **laminar** profile, whose `$(0, 0)$`
    mode carries the mean-flow deviation and dominates the streamwise
    total.
    """
    total = np.einsum("j,...jk->...", y_weights, marginal)
    mean_mode = np.einsum("j,...j->...", y_weights, mean_plane[..., 0])
    return total - mean_mode


def bin_energies(data: YResolvedData) -> dict[str, np.ndarray]:
    r"""`$E_{\Delta U}$`, `$E_{\Delta u_1}$`, `$E_{\Delta u_2}$` per record.

    The three-bin decomposition recovered from the stored marginals
    (module docstring), component-summed, as ``E_dU`` / ``E_du1`` /
    ``E_du2`` -- the same numbers ``twin.dat`` carries under
    ``twin.bins``, and the reason that flag can stay off.  Pass a
    ``twin_yspectra`` stream.
    """
    if "e_x0" not in data.fields:
        raise ValueError("bin_energies needs a twin_yspectra stream")
    x0 = integrate_y(data, "e_x0").sum(axis=1)  # (n_t, n_kz)
    x = integrate_y(data, "e_x").sum(axis=1)
    return {
        "E_dU": x0[:, 0],
        "E_du1": x0[:, 1:].sum(axis=1),
        "E_du2": (x - x0).sum(axis=1),
    }
