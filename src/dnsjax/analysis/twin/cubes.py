r"""JAX-free readers for the twin's 3-D ``(y, k_z, k_x)`` cubes.

The ``twin_spectra3d/``, ``twin_spectra3d_ref/`` and ``twin_budget3d/``
directories of a ``dnsjax-twin`` run, one tar per sample, written by
:mod:`dnsjax.twin.cubes` (whose docstring has the folding, the
subsampling and what every entry means).  Each file is a dnsjax tar --
the snapshot container, with a ``kind`` of its own -- so it is read
through the snapshot leaves (:mod:`dnsjax.snapshot_meta`,
:func:`dnsjax.analysis._core.read_chunks`), and the y-major layout lets
:func:`read_cube` pull single wall distances off disk (*y_rows*).

:func:`read_cube` reads one file; :func:`read_cubes` and its three
named forms stack a directory's files in step order, optionally a
subset of them (*its*), after checking they share one set of points.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...snapshot_meta import read_snapshot_meta, snapshot_kind
from .._core import read_chunks

#: Oldest ``cube_version`` this reader understands (the writer's is
#: ``dnsjax.twin.cubes.CUBE_VERSION``).
MIN_CUBE_VERSION: int = 1

#: The three cube kinds, each its own directory and file stem.
KINDS: tuple[str, ...] = (
    "twin_spectra3d",
    "twin_spectra3d_ref",
    "twin_budget3d",
)

#: Metadata every file of one series must agree on.
_POINTS: tuple[str, ...] = (
    "fields",
    "iy",
    "kz_harmonics",
    "kx_harmonics",
    "lx",
    "lz",
)


@dataclass(frozen=True)
class Cube:
    r"""One sample.

    ``fields`` maps a field name (``e_u``, ``P_U``, ...) to its
    ``(n_y, n_kz, n_kx)`` array, float64; ``wall_distance`` / ``y`` are
    the kept rows (lower half, ascending from the wall), ``kz`` /
    ``kx`` the kept physical wavenumbers (`$|k_z|$` folded), ``meta``
    the file's metadata.
    """

    t: float
    it: int
    fields: dict[str, np.ndarray]
    wall_distance: np.ndarray
    y: np.ndarray
    kz: np.ndarray
    kx: np.ndarray
    meta: dict

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[name]


@dataclass(frozen=True)
class CubeSeries:
    r"""A directory's samples, stacked: each field ``(n_t, n_y, n_kz,
    n_kx)``, ``t`` / ``it`` ``(n_t,)``; the points and ``meta`` are the
    first file's (every file is checked to share them)."""

    t: np.ndarray
    it: np.ndarray
    fields: dict[str, np.ndarray]
    wall_distance: np.ndarray
    y: np.ndarray
    kz: np.ndarray
    kx: np.ndarray
    meta: dict

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[name]


def read_cube(
    path: str | Path, *, y_rows: Iterable[int] | None = None
) -> Cube:
    """Read one cube file; *y_rows* (indices into its kept rows) reads
    only those wall distances off disk."""
    path = Path(path)
    meta = read_snapshot_meta(path)
    kind = snapshot_kind(meta)
    if kind not in KINDS:
        raise ValueError(
            f"{path} is a {kind!r} file, not a twin cube ({KINDS})."
        )
    version = int(meta.get("cube_version", 0))
    if version < MIN_CUBE_VERSION:
        raise ValueError(
            f"{path}: cube_version {version} predates the reader floor "
            f"{MIN_CUBE_VERSION}."
        )
    rows = None if y_rows is None else np.asarray(list(y_rows), dtype=int)
    names = list(meta["fields"])
    raw = read_chunks(path, meta, range(len(names)), slab_indices=rows)
    pick = slice(None) if rows is None else rows
    return Cube(
        t=float(meta["t"]),
        it=int(meta["it"]),
        fields={
            name: raw[i].astype(np.float64) for i, name in enumerate(names)
        },
        wall_distance=np.asarray(meta["wall_distance"], dtype=np.float64)[
            pick
        ],
        y=np.asarray(meta["y"], dtype=np.float64)[pick],
        kz=(2.0 * np.pi / float(meta["lz"]))
        * np.asarray(meta["kz_harmonics"], dtype=np.float64),
        kx=(2.0 * np.pi / float(meta["lx"]))
        * np.asarray(meta["kx_harmonics"], dtype=np.float64),
        meta=meta,
    )


def cube_files(path: str | Path, kind: str) -> list[Path]:
    """A run directory's (or a kind directory's) files, in step order.

    The step count is zero-padded in every name, so the lexical order
    is the step order.
    """
    path = Path(path)
    directory = path if path.name == kind else path / kind
    return sorted(directory.glob(f"{kind}_*.tar"))


def read_cubes(
    path: str | Path, kind: str, *, its: Iterable[int] | None = None
) -> CubeSeries:
    """Stack a run's *kind* cubes (all of them, or the steps *its*)."""
    if kind not in KINDS:
        raise ValueError(f"unknown cube kind {kind!r}; expected {KINDS}")
    files = cube_files(path, kind)
    if its is not None:
        wanted = {int(i) for i in its}
        files = [f for f in files if int(f.stem.rsplit("_", 1)[1]) in wanted]
    if not files:
        raise FileNotFoundError(f"no {kind} files under {path}")
    cubes = [read_cube(f) for f in files]
    first = cubes[0]
    for cube, f in zip(cubes[1:], files[1:], strict=True):
        differs = [k for k in _POINTS if cube.meta.get(k) != first.meta.get(k)]
        if differs:
            raise ValueError(
                f"{f} disagrees with {files[0]} on {', '.join(differs)}: "
                "the files of one directory hold one set of points."
            )
    return CubeSeries(
        t=np.asarray([c.t for c in cubes]),
        it=np.asarray([c.it for c in cubes]),
        fields={
            name: np.stack([c.fields[name] for c in cubes])
            for name in first.fields
        },
        wall_distance=first.wall_distance,
        y=first.y,
        kz=first.kz,
        kx=first.kx,
        meta=first.meta,
    )


def read_twin_spectra3d(
    path: str | Path = ".", *, its: Iterable[int] | None = None
) -> CubeSeries:
    """The difference field's energy cubes (``e_u, e_v, e_w``)."""
    return read_cubes(path, "twin_spectra3d", its=its)


def read_twin_spectra3d_ref(
    path: str | Path = ".", *, its: Iterable[int] | None = None
) -> CubeSeries:
    """The reference state's energy cubes (``r_u, r_v, r_w``)."""
    return read_cubes(path, "twin_spectra3d_ref", its=its)


def read_twin_budget3d(
    path: str | Path = ".", *, its: Iterable[int] | None = None
) -> CubeSeries:
    """The budget cubes, one field per convective term."""
    return read_cubes(path, "twin_budget3d", its=its)
