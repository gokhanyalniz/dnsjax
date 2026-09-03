r"""JAX-free readers for the wall-normal-resolved twin streams.

``twin_yspectra.bin`` and ``twin_ybudget.bin``, written by
:mod:`dnsjax.twin.yspectra`.  Both come back as a
:class:`YResolvedData`: a dict of ``(n_t, ..., n_y, n_k)`` float64
arrays, the two wavenumber axes in physical units, the wall-normal
grid and its quadrature weights, and the sidecar.

Which fields a stream carries is the sidecar's ``suffixes``, and
:func:`stored_fields` is the one place that turns it into a record
layout -- shared with the memory-mapped reader in
``scripts/twin_spectral_maps.py`` rather than mirrored there.  Three
suffix tuples exist, from two eras, and all three are read:

- ``("x", "z", "xz00")``, the default, and ``("x", "z", "x0",
  "xz00")`` under ``twin.x0_planes``;
- ``("x", "z", "x0")``, everything written before ``xz00`` existed.
  Those sidecars have no ``suffixes`` key and a ``format_version``
  below the current writer's; :data:`MIN_YSPECTRA_VERSION` /
  :data:`MIN_YBUDGET_VERSION` are therefore held at the old values
  rather than raised with the writer, which is the exception to the
  usual lockstep rule (:mod:`dnsjax.twin.yspectra` says why).

Everything stored is a `$y$`-**density** already divided by
``volume_fac``.  :func:`integrate_y` contracts one with the weights
to give the per-`$k$` quantity; summing *that* over `$k$` gives the
matching volume-averaged rate.  Both wavenumber axes are one-sided
(`$|k_z|$` folded), so those sums run over the stored axis with no
further weighting.  A ``*_xz00`` field has no wavenumber axis --
it is the `$(0, 0)$` mode alone -- so :func:`integrate_y` returns
its `$y$`-average directly.

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
2026): the `$k_x = 0$` plane is exactly the spectrum of the
streamwise-averaged difference field, so

.. math::
    E_{\Delta U} = \textstyle\int \sum_\alpha e^{x0}_\alpha(y, 0)
        = \int \sum_\alpha e^{xz00}_\alpha ,
    \quad
    E_{\Delta u_1} = \int \sum_\alpha \sum_{k_z > 0} e^{x0}_\alpha ,
    \quad
    E_{\Delta u_2} = \int \sum_\alpha \sum_{k_z}
        \bigl(e^{x}_\alpha - e^{x0}_\alpha\bigr) ,

now resolved in `$k_z$` rather than collapsed to three numbers.  It
therefore needs a stream written under ``twin.x0_planes``; the first
of the three survives on the always-stored ``e_xz00`` alone, and so
does the `$(0, 0)$` mode's other use: taking it back off a spectrum,
in the three reductions :func:`mean_free_spectrum` /
:func:`fluctuation_profile` / :func:`fluctuation_energy`, which are
one definition read at three resolutions -- `$(y, k)$`, `$(y)$` and a
scalar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Reader floors: the oldest layout this module can still *name*, not
#: the current writer version (module docstring).
MIN_YSPECTRA_VERSION: int = 1
MIN_YBUDGET_VERSION: int = 2

#: What a sidecar without a ``suffixes`` key stored, which is every
#: stream written before ``xz00`` existed.
LEGACY_SUFFIXES: tuple[str, ...] = ("x", "z", "x0")

#: Number of velocity components on a ``twin_yspectra`` leading axis.
_N_COMPONENTS: int = 3


@dataclass(frozen=True)
class YResolvedData:
    r"""One wall-normal-resolved stream.

    ``fields`` maps a stored name (``e_x``, ``P_U_z``, ...) to its
    ``(n_t, ...)`` array; ``y`` / ``y_weights`` are the wall-normal
    grid and its quadrature rule; ``kz`` / ``kx`` the one-sided
    physical wavenumbers of the ``*_x`` / ``*_x0`` and ``*_z``
    arrays.  A ``*_xz00`` field has no wavenumber axis.  Which names
    are present is :func:`stored_suffixes` of :attr:`meta`.
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


def stored_suffixes(meta: dict) -> tuple[str, ...]:
    """The marginal suffixes a stream stores, from its sidecar.

    :data:`LEGACY_SUFFIXES` when the key is absent, which is exactly
    the pre-``xz00`` layout (module docstring).
    """
    return tuple(meta.get("suffixes", LEGACY_SUFFIXES))


def stored_fields(meta: dict, stem: str) -> list[tuple[str, tuple[int, ...]]]:
    """``(name, shape)`` per stored field, in stored order.

    The counterpart of the writers' ``_suffix_shapes``
    (:mod:`dnsjax.twin.yspectra`), and the only place either reader
    turns a sidecar into a record layout.  *stem* is
    ``"twin_yspectra"`` or ``"twin_ybudget"``: the first is prefixed
    by ``e`` / ``r`` and carries a component axis, the second by the
    sidecar's ``terms`` and does not.
    """
    ny = int(meta["ny"])
    widths: dict[str, tuple[int, ...]] = {
        "x": (int(meta["n_kz"]),),
        "z": (int(meta["n_kx"]),),
        "x0": (int(meta["n_kz"]),),
        "xz00": (),
    }
    suffixes = stored_suffixes(meta)
    unknown = [suf for suf in suffixes if suf not in widths]
    if unknown:
        raise ValueError(
            f"unknown stored suffix(es) {unknown}; this reader knows "
            f"{sorted(widths)}."
        )
    if stem == "twin_yspectra":
        prefixes = ("e", "r") if bool(meta["includes_ref"]) else ("e",)
        return [
            (f"{p}_{suf}", (_N_COMPONENTS, ny, *widths[suf]))
            for p in prefixes
            for suf in suffixes
        ]
    return [
        (f"{term}_{suf}", (ny, *widths[suf]))
        for term in meta["terms"]
        for suf in suffixes
    ]


def record_dtype(meta: dict, stem: str) -> np.dtype:
    """The stream's fixed-size record layout, from its sidecar alone.

    ``("t", "<f8")`` then :func:`stored_fields`.  Shared with the
    memory-mapped reader in ``scripts/twin_spectral_maps.py``, which
    needs the dtype without the whole-file pass :func:`_read` makes.
    """
    return np.dtype(
        [("t", "<f8")]
        + [
            (name, meta["value_dtype"], shape)
            for name, shape in stored_fields(meta, stem)
        ]
    )


def mean_mode_name(meta: dict, prefix: str) -> str:
    r"""The stored field carrying *prefix*'s `$(0, 0)$` mode.

    ``<prefix>_xz00`` where the stream has it, else the legacy
    ``<prefix>_x0`` whose index 0 is that mode.  Pair it with
    :func:`mean_mode_profile`, which drops the difference.
    """
    suffixes = stored_suffixes(meta)
    for suffix in ("xz00", "x0"):
        if suffix in suffixes:
            return f"{prefix}_{suffix}"
    raise ValueError(
        f"the stream stores {list(suffixes)}, neither the (0, 0) mode "
        "(xz00) nor the k_x = 0 plane (x0) it can be sliced from; "
        "re-run with twin.x0_planes, or with a writer new enough to "
        "store xz00."
    )


def mean_mode_profile(values: np.ndarray, name: str) -> np.ndarray:
    r"""`$(..., n_y)$` from whichever field :func:`mean_mode_name`
    chose: a ``*_xz00`` array is already the profile, a ``*_x0``
    plane needs its `$k_z = 0$` column."""
    return values if name.endswith("_xz00") else values[..., 0]


def _resolve_pair(path: str | Path, stem: str) -> tuple[Path, Path]:
    path = Path(path)
    if path.is_dir():
        return path / f"{stem}.bin", path / f"{stem}.json"
    if path.suffix == ".json":
        return path.with_suffix(".bin"), path
    return path, path.with_suffix(".json")


def _read(path: str | Path, stem: str, floor: int) -> YResolvedData:
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
    names = stored_fields(meta, stem)
    dtype = record_dtype(meta, stem)

    raw = np.fromfile(bin_path, dtype=np.uint8)
    n_records = raw.size // dtype.itemsize
    if n_records == 0:
        raise ValueError(f"{bin_path}: no complete records")
    if raw.size % dtype.itemsize:
        # A kill mid-write leaves a partial trailing record; the
        # complete prefix is intact (append-only + fsync per flush).
        raw = raw[: n_records * dtype.itemsize]
    records = raw.view(dtype)

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
    ``.json``).  Fields ``e_<suffix>`` for each of
    :func:`stored_suffixes`, ``(n_t, 3, n_y, n_k)`` or
    ``(n_t, 3, n_y)``, plus the matching ``r_*`` set when the run set
    ``twin.spectra_ref``."""
    return _read(path, "twin_yspectra", MIN_YSPECTRA_VERSION)


def read_twin_ybudget(path: str | Path = ".") -> YResolvedData:
    """Read ``twin_ybudget``.  Fields ``<term>_<suffix>`` for each
    name in the sidecar's ``terms`` and each of
    :func:`stored_suffixes`, ``(n_t, n_y, n_k)`` or ``(n_t, n_y)``."""
    return _read(path, "twin_ybudget", MIN_YBUDGET_VERSION)


def integrate_y(data: YResolvedData, name: str) -> np.ndarray:
    r"""Contract a stored density with the wall-normal quadrature.

    Returns ``(n_t, ..., n_k)``: the per-`$k$` energy or budget rate.
    Summing the result over its last axis gives the corresponding
    ``twin.dat`` / ``twin_budget.dat`` scalar (for the energies, also
    sum over the component axis).

    A ``*_xz00`` field has no wavenumber axis, so it comes back
    ``(n_t, ...)`` -- already the scalar for that one mode.  The
    branch is on the **name**, which is what fixes a field's layout
    (:func:`stored_fields`), not on the array's rank.
    """
    subscripts = "j,...j->..." if name.endswith("_xz00") else "j,...jk->...k"
    return np.einsum(subscripts, data.y_weights, data[name])


def mean_free_spectrum(
    marginal: np.ndarray,
    mean_mode: np.ndarray,
) -> np.ndarray:
    r"""A stored marginal with the `$(0, 0)$` mode taken off it.

    The first of three reductions of one idea, each the next one's
    input: the spectrum without the mean mode, its `$k$`-sum
    (:func:`fluctuation_profile`), and that profile's wall-normal
    average (:func:`fluctuation_energy`).  All three are array-level,
    so a memory-mapped reader shares them with
    :class:`YResolvedData` (``scripts/twin_spectral_maps.py`` does).

    *marginal* is a **complete** marginal -- ``r_x`` / ``e_x`` (summed
    over `$k_x$`) or ``r_z`` / ``e_z`` (summed over `$k_z$`) -- shaped
    ``(..., n_y, n_k)``, and *mean_mode* the same field's `$(0, 0)$`
    profile, ``(..., n_y)``, with the leading axes free;
    :func:`mean_mode_name` / :func:`mean_mode_profile` produce that
    second argument from whichever of ``*_xz00`` / ``*_x0`` the stream
    carries.

    The mean mode lives in the `$m = 0$` column and **only** there, so
    that is the one column this touches.  What it is *not* is the
    whole of that column: ``*_x[..., 0]`` is every `$k_z = 0$` mode
    summed over `$k_x$` and ``*_z[..., 0]`` every `$k_x = 0$` mode
    summed over `$k_z$`, both of which carry fluctuating modes that
    have every right to be there.

    Use it wherever a **reference** spectrum is a denominator: the
    `$(0, 0)$` mode is the wall-parallel mean, common to both states
    of a twin pair and never decorrelating, and the stored spectra
    are of the perturbation about the **laminar** profile, so it also
    carries the mean-flow deviation and dominates the streamwise
    total.  A returned copy, always float64; the input is untouched.
    """
    out = np.array(marginal, dtype=np.float64)
    out[..., 0] -= mean_mode
    return out


def fluctuation_profile(
    marginal: np.ndarray,
    mean_mode: np.ndarray,
) -> np.ndarray:
    r"""`$(..., n_y)$`: :func:`mean_free_spectrum` summed over `$k$`.

    The energy at each wall distance held by everything but the
    `$(0, 0)$` mode.  Written as the sum less the mode rather than as
    a sum of the copy -- the same number, one array cheaper.

    Either marginal gives the same profile, since each already sums
    the other axis, so passing ``r_z`` against the same mean mode as
    ``r_x`` is a genuine cross-check rather than a second reading of
    the same numbers.
    """
    return marginal.sum(axis=-1) - mean_mode


def fluctuation_energy(
    marginal: np.ndarray,
    mean_mode: np.ndarray,
    y_weights: np.ndarray,
) -> np.ndarray:
    r"""Total-in-`$(y, k)$` energy without the `$(0, 0)$` mode.

    :func:`fluctuation_profile` contracted with the wall-normal
    quadrature; the arguments are that function's, and the result has
    its leading axes.

    Because the `$y$`-weights sum to ``volume_fac`` and the stored
    entries are already divided by it, the contraction is a
    wall-normal **average**: the result is a mean energy density, not
    an integral.
    """
    return np.einsum(
        "j,...j->...", y_weights, fluctuation_profile(marginal, mean_mode)
    )


def shape_alignment(
    p: np.ndarray,
    q: np.ndarray,
    y_weights: np.ndarray,
) -> float:
    r"""How much two `$(y, k)$` distributions overlap, in `$[0, 1]$`.

    The Bhattacharyya coefficient
    `$A = \int \sqrt{p\,q}\,\mathrm{d}y\,\mathrm{d}k$` of two
    ``(n_y, n_k)`` densities, each first normalised to unit total
    against *y_weights*, so it reads **shape only** -- an amplitude
    is divided out, and `$A = 1$` exactly when the two shapes agree.
    Negative entries (a budget term, say) are not distributions and
    are rejected.

    What it is for: the difference field of a twin run settles onto a
    `$(y, k)$` distribution of its own within a few time units,
    whatever it started from, and `$1 - A(t)$` against that
    distribution decays exponentially.  An initial condition is
    therefore worth exactly the head start its own `$A(0)$` buys, and
    this is the number to calibrate one against -- see
    ``scripts/random_ic_calibrate.py``, which is its caller, and
    :mod:`dnsjax.ic.random_field` for what the shipped generator is
    calibrated to.

    Pair it with :func:`mean_free_spectrum`: the `$(0, 0)$` mode is
    the wall-parallel mean and is not part of the shape being
    compared.
    """
    pf = np.asarray(p, dtype=np.float64)
    qf = np.asarray(q, dtype=np.float64)
    if pf.shape != qf.shape:
        raise ValueError(
            f"shape mismatch: {pf.shape} against {qf.shape}; both must "
            "be (n_y, n_k) on the same grid and wavenumber axis."
        )
    if pf.min() < 0.0 or qf.min() < 0.0:
        raise ValueError(
            "shape_alignment compares densities; a negative entry "
            "means the input is not one."
        )
    norms = [float(np.einsum("j,jk->", y_weights, a)) for a in (pf, qf)]
    if min(norms) <= 0.0:
        raise ValueError("shape_alignment needs two non-zero densities.")
    return float(
        np.einsum(
            "j,jk->", y_weights, np.sqrt((pf / norms[0]) * (qf / norms[1]))
        )
    )


def bin_energies(data: YResolvedData) -> dict[str, np.ndarray]:
    r"""`$E_{\Delta U}$`, `$E_{\Delta u_1}$`, `$E_{\Delta u_2}$` per record.

    The three-bin decomposition recovered from the stored marginals
    (module docstring), component-summed, as ``E_dU`` / ``E_du1`` /
    ``E_du2`` -- the same numbers ``twin.dat`` carries under
    ``twin.bins``, and the reason that flag can stay off.  Pass a
    ``twin_yspectra`` stream written under ``twin.x0_planes``: all
    three need the `$k_x = 0$` plane, and only ``E_dU`` survives
    without it, as ``integrate_y(data, "e_xz00").sum(axis=1)``.
    """
    if "e_x0" not in data.fields:
        raise ValueError(
            "bin_energies needs the k_x = 0 plane, which this stream "
            f"does not carry (it stores {list(stored_suffixes(data.meta))}"
            "); re-run or rebuild it with twin.x0_planes."
        )
    x0 = integrate_y(data, "e_x0").sum(axis=1)  # (n_t, n_kz)
    x = integrate_y(data, "e_x").sum(axis=1)
    return {
        "E_dU": x0[:, 0],
        "E_du1": x0[:, 1:].sum(axis=1),
        "E_du2": (x - x0).sum(axis=1),
    }
