r"""Readers for the twin ``(k_z, k_x)`` spectra streams (JAX-free).

Format and conventions: the :mod:`dnsjax.twin.spectra` writer
docstring.  ``twin_spectra.bin`` holds the difference field's per-mode
energy `$E_\Delta(k_z, k_x)$` on the true (unpadded) mode grid --
summing over modes reproduces ``twin.dat``'s ``E_d`` -- and, under
``twin.spectra_ref`` (the default), the reference state's own spectrum
`$E^{(1)}(k_z, k_x)$` is recorded too: in ``twin_spectra_ref.bin``, on
its own cadence (by default the same steps), or -- for a member
recorded before that split -- inside ``twin_spectra.bin`` itself
(``includes_ref``).

:func:`read_twin_spectra` returns either layout the same way: its
``e_ref`` is taken from whichever stream holds the reference, placed
on the difference stream's sample times (both are written from one
``t``, so the match is exact) with ``nan`` rows where the reference
has no sample at that time.  :func:`read_twin_spectra_ref` returns the
reference on its own times.

The readers tolerate a truncated trailing record (a kill mid-write)
and drop exact-duplicate timestamps (resume seams), like the probe
reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Oldest ``twin_spectra.json`` schema this reader understands
#: (``dnsjax.twin.spectra.FORMAT_VERSION`` is the writer's).
MIN_FORMAT_VERSION: int = 1

#: Oldest ``twin_spectra_ref.json`` schema (the writer's is
#: ``dnsjax.twin.spectra.REF_FORMAT_VERSION``).
MIN_REF_FORMAT_VERSION: int = 1


@dataclass(frozen=True)
class TwinSpectraData:
    r"""One twin spectra stream, parsed.

    ``e_delta`` (and ``e_ref``, ``None`` when not recorded) have
    shape ``(n_t, n_kz, n_kx)`` on the true mode grid; ``kz`` /
    ``kx`` are the *physical* wavenumbers (harmonics
    `$\times\, 2\pi/L$`, in the stored axis order); ``meta`` is
    the sidecar dict.
    """

    t: np.ndarray
    e_delta: np.ndarray
    e_ref: np.ndarray | None
    kz: np.ndarray
    kx: np.ndarray
    meta: dict


@dataclass(frozen=True)
class TwinSpectraRefData:
    r"""The reference spectrum on its own sample times.

    ``e_ref`` is ``(n_t, n_kz, n_kx)``; the rest as in
    :class:`TwinSpectraData`, ``meta`` the sidecar it came from.
    """

    t: np.ndarray
    e_ref: np.ndarray
    kz: np.ndarray
    kx: np.ndarray
    meta: dict


def _resolve_pair(path: str | Path, stem: str = "twin_spectra"):
    path = Path(path)
    if path.is_dir():
        return path / f"{stem}.bin", path / f"{stem}.json"
    if path.suffix == ".json":
        return path.with_suffix(".bin"), path
    return path, path.with_suffix(".json")


def _read(bin_path: Path, json_path: Path, floor: int):
    """``(records, t, meta, kz, kx)`` of one stream, deduplicated."""
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

    n2, n3 = int(meta["n2"]), int(meta["n3"])
    value_dtype = meta["value_dtype"]
    if json_path.name == "twin_spectra_ref.json":
        names = ["e_ref"]
    else:
        names = ["e_delta"] + (["e_ref"] if meta["includes_ref"] else [])
    record_dtype = np.dtype(
        [("t", "<f8")] + [(name, value_dtype, (n2, n3)) for name in names]
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
    records = records[keep]
    t = t[keep]

    kz = (2.0 * np.pi / float(meta["lz"])) * np.asarray(
        meta["kz_harmonics"], dtype=np.float64
    )
    kx = (2.0 * np.pi / float(meta["lx"])) * np.asarray(
        meta["kx_harmonics"], dtype=np.float64
    )
    # The stored spectrum drops the padding slots; the harmonic lists
    # are the full true-mode sequences already (n2 / n3 entries).
    if kz.shape[0] != n2 or kx.shape[0] != n3:
        raise ValueError(
            f"{json_path}: harmonic lists ({kz.shape[0]}, "
            f"{kx.shape[0]}) do not match the mode counts "
            f"({n2}, {n3})."
        )
    return records, t, meta, kz, kx


def on_times(
    t: np.ndarray, t_src: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """*values* (sampled at *t_src*) placed on the times *t*.

    Exact matching -- the twin writers stamp both streams of a pair
    with the same ``t`` -- and a ``nan`` row wherever *t_src* has no
    sample.  Float64.
    """
    out = np.full((t.size, *values.shape[1:]), np.nan)
    index = {float(v): i for i, v in enumerate(t_src)}
    for i, v in enumerate(t):
        j = index.get(float(v))
        if j is not None:
            out[i] = values[j]
    return out


def read_twin_spectra_ref(path: str | Path = ".") -> TwinSpectraRefData:
    """The reference spectrum of a run directory, in either layout.

    From ``twin_spectra_ref.bin`` when the run has it, else from the
    ``e_ref`` of a pre-split ``twin_spectra.bin``; refused when neither
    holds one (``twin.spectra_ref`` was off).  *path* is the run
    directory or either stream's ``.bin`` / ``.json``.
    """
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    ref_bin, ref_json = _resolve_pair(directory, "twin_spectra_ref")
    if ref_json.is_file():
        records, t, meta, kz, kx = _read(
            ref_bin, ref_json, MIN_REF_FORMAT_VERSION
        )
        return TwinSpectraRefData(
            t, records["e_ref"].astype(np.float64), kz, kx, meta
        )
    records, t, meta, kz, kx = _read(
        *_resolve_pair(directory), MIN_FORMAT_VERSION
    )
    if not meta["includes_ref"]:
        raise ValueError(
            f"{directory} carries no reference spectra "
            "(twin.spectra_ref was off)."
        )
    return TwinSpectraRefData(
        t, records["e_ref"].astype(np.float64), kz, kx, meta
    )


def read_twin_spectra(path: str | Path = ".") -> TwinSpectraData:
    """Read a stream (a run directory, the ``.bin``, or the ``.json``).

    ``e_ref`` comes from whichever layout holds the reference, on the
    difference stream's times (module docstring); ``None`` when the
    run recorded none.
    """
    bin_path, json_path = _resolve_pair(path)
    records, t, meta, kz, kx = _read(bin_path, json_path, MIN_FORMAT_VERSION)
    if meta["includes_ref"]:
        e_ref = records["e_ref"].astype(np.float64)
    elif json_path.with_name("twin_spectra_ref.json").is_file():
        ref = read_twin_spectra_ref(json_path.parent)
        e_ref = on_times(t, ref.t, ref.e_ref)
    else:
        e_ref = None
    return TwinSpectraData(
        t=t,
        e_delta=records["e_delta"].astype(np.float64),
        e_ref=e_ref,
        kz=kz,
        kx=kx,
        meta=meta,
    )


def decorrelation_ratio(
    data: TwinSpectraData, floor: float = 0.0
) -> np.ndarray:
    r"""`$E_\Delta(k) / (2 E^{(1)}(k))$` per record and mode.

    Two fully decorrelated, statistically identical fields give 1
    (the difference of independent fields carries twice the energy
    of each).  Modes whose reference energy is at or below *floor*
    return ``nan`` (empty reference scales carry no decorrelation
    information).  Requires a stream written with
    ``twin.spectra_ref``.
    """
    if data.e_ref is None:
        raise ValueError(
            "the stream carries no reference spectra "
            "(twin.spectra_ref was off)."
        )
    denom = 2.0 * data.e_ref
    out = np.full_like(data.e_delta, np.nan)
    ok = denom > (2.0 * floor)
    out[ok] = data.e_delta[ok] / denom[ok]
    return out
