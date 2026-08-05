r"""Reader for the twin spectra stream ``twin_spectra.bin`` (JAX-free).

Format and conventions: the :mod:`dnsjax.twin_spectra` writer
docstring.  Each record holds the difference field's per-mode energy
`$E_\Delta(k_z, k_x)$` on the true (unpadded) mode grid -- summing
over modes reproduces ``twin.dat``'s ``E_d`` -- and, when the stream
was written with ``twin.spectra_ref`` (the default), the reference
state's own spectrum `$E^{(1)}(k_z, k_x)$`.

The reader tolerates a truncated trailing record (a kill mid-write)
and drops exact-duplicate timestamps (resume seams), like the probe
reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Oldest ``twin_spectra.json`` schema this reader understands
#: (``dnsjax.twin_spectra.FORMAT_VERSION`` is the writer's).
MIN_FORMAT_VERSION: int = 1


@dataclass(frozen=True)
class TwinSpectraData:
    """One twin spectra stream, parsed.

    ``e_delta`` (and ``e_ref``, ``None`` when not recorded) have
    shape ``(n_t, n_kz, n_kx)`` on the true mode grid; ``kz`` /
    ``kx`` are the *physical* wavenumbers (harmonics
    `$\\times\\, 2\\pi/L$`, in the stored axis order); ``meta`` is
    the sidecar dict.
    """

    t: np.ndarray
    e_delta: np.ndarray
    e_ref: np.ndarray | None
    kz: np.ndarray
    kx: np.ndarray
    meta: dict


def _resolve_pair(path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    if path.is_dir():
        return path / "twin_spectra.bin", path / "twin_spectra.json"
    if path.suffix == ".json":
        return path.with_suffix(".bin"), path
    return path, path.with_suffix(".json")


def read_twin_spectra(path: str | Path = ".") -> TwinSpectraData:
    """Read a stream (a run directory, the ``.bin``, or the ``.json``)."""
    bin_path, json_path = _resolve_pair(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"no sidecar {json_path}")
    with open(json_path) as fh:
        meta = json.load(fh)
    version = int(meta.get("format_version", 0))
    if version < MIN_FORMAT_VERSION:
        raise ValueError(
            f"{json_path}: format_version {version} predates the "
            f"reader floor {MIN_FORMAT_VERSION}; re-run with the "
            "current writer."
        )

    n2, n3 = int(meta["n2"]), int(meta["n3"])
    value_dtype = meta["value_dtype"]
    includes_ref = bool(meta["includes_ref"])
    fields = [("t", "<f8"), ("e_delta", value_dtype, (n2, n3))]
    if includes_ref:
        fields.append(("e_ref", value_dtype, (n2, n3)))
    record_dtype = np.dtype(fields)

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
    return TwinSpectraData(
        t=t,
        e_delta=records["e_delta"].astype(np.float64),
        e_ref=(records["e_ref"].astype(np.float64) if includes_ref else None),
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
