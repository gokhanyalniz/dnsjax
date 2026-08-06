r"""Read a dnsjax snapshot into NumPy arrays (no JAX, no zarr3).

:func:`read_state` opens a single-file snapshot and returns the
velocity field in physical and/or spectral space (in the native axis
layout -- the on-disk bytes are the solver's spectral layout, and
nothing is ever transposed), the matching coordinate arrays, and the
embedded parameters and stats.  It reads the **least data possible**:
only the requested velocity components and, for wall-bounded
snapshots, only the requested wall-normal slabs are pulled off disk.

See :mod:`dnsjax.analysis._core` for the native layout table, the
component bases, and the transform conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import _core


class StateData(NamedTuple):
    r"""Result of :func:`read_state` (unpacks in the documented order).

    Attributes
    ----------
    physical:
        Tuple of physical-space velocity components (one real array
        each, native axis layout), or ``None`` when
        ``return_physical`` is off.
    physical_coords:
        Tuple of physical coordinate arrays, one per axis in the data's
        axis order, or ``None``.
    spectral:
        Tuple of spectral velocity components (complex, as stored, in
        the returned basis), or ``None`` when ``return_spectral`` is off.
    spectral_coords:
        Tuple of spectral coordinates (wavenumbers; the wall-normal grid
        sits at its axis for wall-bounded geometries), or ``None``.
    params:
        :class:`~dnsjax.analysis._core.Namespace` over the embedded
        parameters (``params.phys.re`` ...).
    stats:
        :class:`~dnsjax.analysis._core.Namespace` over the embedded
        stats (item access, e.g. ``stats["E'"]``), or ``None`` if the
        snapshot carries none.
    """

    physical: tuple | None
    physical_coords: tuple | None
    spectral: tuple | None
    spectral_coords: tuple | None
    params: _core.Namespace
    stats: _core.Namespace | None


def _validate_components(components, ncomp: int) -> tuple[int, ...]:
    out: list[int] = []
    for c in components:
        c = int(c)
        if not 0 <= c < ncomp:
            raise ValueError(
                f"component {c} out of range; must be in [0, {ncomp - 1}]."
            )
        if c not in out:
            out.append(c)
    if not out:
        raise ValueError("components must request at least one component.")
    return tuple(out)


def read_state(
    path: str | Path,
    *,
    return_physical: bool = True,
    return_spectral: bool = False,
    components=(0, 1, 2),
    wall_normal_points=None,
) -> StateData:
    r"""Read a snapshot's velocity field, grids, parameters and stats.

    Parameters
    ----------
    path:
        Path to a dnsjax single-file (tar) snapshot.
    return_physical:
        Return the physical-space velocity and its grids (default
        ``True``).
    return_spectral:
        Return the stored spectral velocity and its wavenumbers (default
        ``False``).
    components:
        Which components to read from disk (default the 3 velocity
        components).  Order is preserved and duplicates removed; each
        component reads exactly its own stored chunk (the stored
        components are the physical components).  The **viscoelastic**
        system exposes 9 components (velocity ``0..2`` plus the
        physical conformation tensor ``c_zz, c_rz, c_θz, c_rr, c_θθ,
        c_rθ`` = ``3..8``).
    wall_normal_points:
        Optional list/array of wall-normal coordinate *values*.  The
        nearest unique grid points are selected (de-duplicated, sorted)
        and only that slice of the field and the corresponding
        wall-normal grid are returned.  For wall-bounded snapshots only
        those slabs are read from disk (minimal I/O) and the subset
        applies to both physical and spectral output.  For
        triply-periodic snapshots the full ``k_y`` axis is read and
        transformed first, then the physical ``y`` axis is sliced
        (spectral output keeps the full ``k_y`` -- its wall-normal axis
        is a Fourier axis, not a grid).

    Returns
    -------
    :
        A :class:`StateData` named tuple
        ``(physical, physical_coords, spectral, spectral_coords,
        params, stats)``.

    Notes
    -----
    Physical-space components are real; cylindrical/annular return
    ``(u_z, u_r, u_θ)``, all other families ``(u_x, u_y, u_z)``.  The
    stored field is the **perturbation** ``u'`` (the total field for
    Dean); the analytical base flow is not added.
    """
    path = Path(path)
    meta = _core.read_meta(path)
    params = _core.params_namespace(meta)
    stats_raw = _core.read_stats(path)
    stats = _core.Namespace(stats_raw) if stats_raw is not None else None
    info = _core.geometry_info(params)

    out_components = _validate_components(components, len(info.components))
    native_needed = sorted(set(out_components))
    wall_normal_grid = meta.get("wall_normal_grid")

    # Wall-bounded wall-normal subset → read only those outer slabs.
    wn_indices = None
    if wall_normal_points is not None and info.walled:
        wn_indices = _core.nearest_unique_indices(
            wall_normal_grid, wall_normal_points
        )

    raw = _core.read_chunks(path, meta, native_needed, slab_indices=wn_indices)
    native = {c: raw[c] for c in out_components}

    wn_grid = wall_normal_grid
    if wn_indices is not None:
        wn_grid = np.asarray(wall_normal_grid, dtype=float)[wn_indices]

    spectral = spectral_coords = None
    if return_spectral:
        spectral = tuple(native[c] for c in out_components)
        spectral_coords = _core.spectral_coords(info, wn_grid)

    physical = physical_coords = None
    if return_physical:
        phys = {c: _core.inverse_transform(native[c], info) for c in native}
        grids = _core.physical_grids(info, wn_grid)
        # Triply-periodic wall-normal subset is applied after transform.
        if wall_normal_points is not None and not info.walled:
            yax = info.wall_normal_axis
            sel = _core.nearest_unique_indices(grids[yax], wall_normal_points)
            phys = {c: np.take(v, sel, axis=yax) for c, v in phys.items()}
            grids = tuple(
                g[sel] if a == yax else g for a, g in enumerate(grids)
            )
        physical = tuple(phys[c] for c in out_components)
        physical_coords = grids

    return StateData(
        physical, physical_coords, spectral, spectral_coords, params, stats
    )
