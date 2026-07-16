r"""Shared engine for the JAX-free snapshot analysis API.

This module backs :mod:`dnsjax.analysis.snapshot_export` and
:mod:`dnsjax.analysis.snapshot_ops`.  It depends only on NumPy, the
standard library, and dnsjax's JAX-free leaf modules
(:mod:`dnsjax.snapshot_meta`, :mod:`dnsjax.harmonics`,
:mod:`dnsjax.fd`) -- importing it never pulls in JAX.

Snapshot-native (on-disk) layout
--------------------------------
A component chunk read straight off disk and reshaped to
``(a_size, kx_global, b_size)`` *is* the snapshot-native layout; we
never transpose it.  Axis 1 is always the real-FFT axis
(``kx_global = nx // 2``).  Per family:

==================  =======================  =================
family              spectral axes (as read)  physical axes
==================  =======================  =================
cartesian           (y, kx, kz)              (y, x, z)
cylindrical/annular (r, k_axial, m)          (r, z, θ)
viscoelastic (dean) (r, k_axial, m)          (r, z, θ)
triply-periodic     (kz, kx, ky)             (z, x, y)
==================  =======================  =================

Components are ``(u_x, u_y, u_z)`` for cartesian / triply-periodic and
``(u_z, u_r, u_θ)`` for cylindrical / annular (the stored ``u_±`` basis
is converted to ``u_r, u_θ`` at read time; see
:func:`to_returned_basis`).  The viscoelastic-dean system shares the
cylindrical/annular axes with **9 components**: the 3 velocity
components plus the physical conformation tensor
``(c_zz, c_rz, c_θz, c_rr, c_θθ, c_rθ)`` as components ``3..8``, each
combined at read time from its stored spin components (recipes in
``_component_recipes`` / :func:`geometry_info`).

Transforms reinstate the omitted Nyquist mode as zero and use NumPy
``ifft`` / ``irfft`` with ``norm="forward"`` (the inverse of the
solver's forward-normalised transform), differentiating in place along
the stored axes -- no transpose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray

from ..fd import build_diff_matrices
from ..flows import registry as _registry
from ..harmonics import complex_harmonics, real_harmonics
from ..snapshot_meta import (
    is_snapshot_file,
    read_snapshot_meta,
    read_snapshot_stats,
    snapshot_component_offsets,
)

# Flow systems per geometry family, from the JAX-free flow-spec
# registry (the single source of truth: a new flow spec extends these
# automatically).
CARTESIAN_SYSTEMS = frozenset(_registry.cartesian_systems)
CYLINDRICAL_SYSTEMS = frozenset(_registry.cylindrical_systems)
ANNULAR_SYSTEMS = frozenset(_registry.annular_systems)
#: Viscoelastic annular systems (9-component state: 3 velocity + 6
#: symmetric conformation-tensor components).  Annular *geometry*, but a
#: distinct component schema, so kept separate from ``ANNULAR_SYSTEMS``.
VISCOELASTIC_SYSTEMS = frozenset(_registry.viscoelastic_systems)
PERIODIC_SYSTEMS = frozenset(_registry.periodic_systems)

#: Triply-periodic shear-direction box length (fixed length reference;
#: see :mod:`dnsjax.geometries.triply_periodic`).
LY_PERIODIC = 4.0

_NP_DTYPES = {
    "complex128": np.dtype("<c16"),
    "complex64": np.dtype("<c8"),
}


# ── Object-like access to embedded params / stats ────────────


class Namespace:
    r"""Recursive read-only view over a nested dict.

    Gives both attribute access (``params.phys.re``) and item access
    (``stats["E'"]`` -- stats keys such as ``E'`` or ``tau'_s,b`` are
    not valid identifiers).  No pydantic dependency.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", dict(data))

    def _wrap(self, value):
        return Namespace(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            return self._wrap(data[name])
        raise AttributeError(name)

    def __getitem__(self, key: str):
        return self._wrap(object.__getattribute__(self, "_data")[key])

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_data")

    def get(self, key: str, default=None):
        data = object.__getattribute__(self, "_data")
        return self._wrap(data[key]) if key in data else default

    def keys(self):
        return object.__getattribute__(self, "_data").keys()

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def to_dict(self) -> dict:
        """Return the underlying plain dict (a shallow copy)."""
        return dict(object.__getattribute__(self, "_data"))

    def __iter__(self):
        return iter(self.keys())

    def __repr__(self) -> str:
        keys = ", ".join(map(str, self.keys()))
        return f"Namespace({keys})"


def params_namespace(meta: dict) -> Namespace:
    """Internal-named parameter view over snapshot metadata.

    Stored (v4) metadata records the flow-relevant **public** names;
    this maps them back to internal names and rehydrates the
    hidden-derived internal fields (the annular azimuthal ``geo.lz``,
    the derived ``phys.re``/``re2``) via
    :func:`dnsjax.flows.registry.internalize_stored` -- so downstream
    code reads ``params.res.nx`` / ``params.geo.lz`` etc. exactly as
    the solver's live singleton would hold them.
    """
    stored = meta["params"]
    system = meta.get("system") or stored.get("phys", {}).get("system")
    return Namespace(
        _registry.internalize_stored(stored, system, rehydrate=True)
    )


# ── Geometry descriptor ──────────────────────────────────────


@dataclass(frozen=True)
class GeometryInfo:
    r"""Axis semantics for a snapshot, in snapshot-native axis order.

    All tuples are indexed by the on-disk axis (0, 1, 2).
    """

    family: str  # "cartesian" | "cylindrical" | "annular" | "triply_periodic"
    walled: bool
    kind: tuple[str, str, str]  # per axis: "grid" | "real" | "complex"
    name: tuple[str, str, str]  # direction name per axis
    n: tuple[int, int, int]  # full physical size per axis
    length: tuple[float | None, ...]  # physical length (None for grid axis)
    # Returned component labels (axis-0 order).  Length 3 for the
    # velocity-only systems; 9 for the viscoelastic system (velocity +
    # physical conformation-tensor components).
    components: tuple[str, ...]
    wall_normal_axis: int  # physical axis carrying the wall-normal direction
    grid_axis: int | None  # spectral axis stored as a grid (None if periodic)

    def axis_of(self, direction: str) -> int:
        """On-disk axis index of a named direction."""
        try:
            return self.name.index(direction)
        except ValueError:
            raise ValueError(
                f"{self.family} has no direction {direction!r}; "
                f"valid directions are {self.name}."
            ) from None


def geometry_info(params: Namespace) -> GeometryInfo:
    """Build the :class:`GeometryInfo` for a snapshot's parameters."""
    system = str(params.phys.system)
    nx, ny, nz = int(params.res.nx), int(params.res.ny), int(params.res.nz)
    lx, lz = float(params.geo.lx), float(params.geo.lz)

    if system in CARTESIAN_SYSTEMS:
        return GeometryInfo(
            family="cartesian",
            walled=True,
            kind=("grid", "real", "complex"),
            name=("y", "x", "z"),
            n=(ny, nx, nz),
            length=(None, lx, lz),
            components=("u_x", "u_y", "u_z"),
            wall_normal_axis=0,
            grid_axis=0,
        )
    if (
        system in CYLINDRICAL_SYSTEMS
        or system in ANNULAR_SYSTEMS
        or system in VISCOELASTIC_SYSTEMS
    ):
        family = "cylindrical" if system in CYLINDRICAL_SYSTEMS else "annular"
        if system in VISCOELASTIC_SYSTEMS:
            # Velocity + physical conformation-tensor components (the
            # stored spin combos are converted in ``to_returned_basis``).
            components = (
                "u_z",
                "u_r",
                "u_theta",
                "c_zz",
                "c_rz",
                "c_theta_z",
                "c_rr",
                "c_theta_theta",
                "c_r_theta",
            )
        else:
            components = ("u_z", "u_r", "u_theta")
        # Azimuthal length is the wedge extent lz = 2*pi/m0 (m0 = 1 full
        # circle; stored in geo.lz); the real axis is axial (lx).
        return GeometryInfo(
            family=family,
            walled=True,
            kind=("grid", "real", "complex"),
            name=("r", "z", "theta"),
            n=(ny, nx, nz),
            length=(None, lx, lz),
            components=components,
            wall_normal_axis=0,
            grid_axis=0,
        )
    if system in PERIODIC_SYSTEMS:
        return GeometryInfo(
            family="triply_periodic",
            walled=False,
            kind=("complex", "real", "complex"),
            name=("z", "x", "y"),
            n=(nz, nx, ny),
            length=(lz, lx, LY_PERIODIC),
            components=("u_x", "u_y", "u_z"),
            wall_normal_axis=2,
            grid_axis=None,
        )
    raise ValueError(
        f"Unknown system {system!r}: not in any family of the "
        "dnsjax.flows.registry specs this reader has a component "
        "schema for (a new family needs a geometry_info branch)."
    )


# ── Metadata / stats access ──────────────────────────────────


def read_meta(path: str | Path) -> dict:
    """Parsed ``_dnsjax_meta.json`` of a dnsjax snapshot.

    Raises a clear error for a non-snapshot file (e.g. a legacy
    ``.npz``), so callers get a useful message rather than a tar parse
    error.
    """
    path = Path(path)
    if not is_snapshot_file(path):
        raise ValueError(
            f"{path} is not a dnsjax snapshot (an uncompressed tar with a "
            "_dnsjax_meta.json member); legacy .npz files are unsupported."
        )
    return read_snapshot_meta(path)


def read_stats(path: str | Path) -> dict | None:
    """Parsed ``_dnsjax_stats.json`` of a snapshot, or ``None``."""
    return read_snapshot_stats(path)


# ── Raw chunk reading (no transpose, minimal I/O) ────────────


def _np_dtype(name: str) -> np.dtype:
    try:
        return _NP_DTYPES[name]
    except KeyError:
        raise ValueError(f"Unsupported snapshot dtype {name!r}.") from None


def _component_recipes(
    info: GeometryInfo,
) -> list[tuple[tuple[int, ...], Callable[[dict[int, ndarray]], ndarray]]]:
    r"""Per-returned-component ``(native chunks, combine)`` recipes.

    Cartesian / periodic: identity (component ``i`` is native chunk
    ``i``).  Cylindrical / annular: the velocity triad converts the
    stored ``u_±`` pair (`$u_r = (u_+ + u_-)/2$`,
    `$u_\theta = (u_+ - u_-)/2i$`); the **viscoelastic** system adds the
    6 physical conformation components from the stored spin combos
    (`$c_{rz} = (c_{z+} + c_{z-})/2$`, `$c_{rr} = c_{+-}/2 +
    (c_{++} + c_{--})/4$`, etc.).
    """
    if info.family in ("cylindrical", "annular"):
        recipes: list = [
            ((0,), lambda r: r[0]),  # u_z
            ((1, 2), lambda r: (r[1] + r[2]) / 2.0),  # u_r
            ((1, 2), lambda r: (r[1] - r[2]) / 2.0j),  # u_theta
        ]
        if len(info.components) > 3:  # viscoelastic (9 components)
            recipes += [
                ((3,), lambda r: r[3]),  # c_zz
                ((4, 5), lambda r: (r[4] + r[5]) / 2.0),  # c_rz
                ((4, 5), lambda r: (r[4] - r[5]) / 2.0j),  # c_theta_z
                # c_rr, c_theta_theta from c_+-, c_++, c_--.
                ((6, 7, 8), lambda r: r[6] / 2.0 + (r[7] + r[8]) / 4.0),
                ((6, 7, 8), lambda r: r[6] / 2.0 - (r[7] + r[8]) / 4.0),
                ((7, 8), lambda r: (r[7] - r[8]) / 4.0j),  # c_r_theta
            ]
        return recipes
    return [((i,), _pick(i)) for i in range(len(info.components))]


def _pick(i: int) -> Callable[[dict[int, ndarray]], ndarray]:
    """Identity combine for native chunk ``i`` (Cartesian / periodic)."""
    return lambda r: r[i]


def native_components_needed(
    info: GeometryInfo, out_components: tuple[int, ...]
) -> list[int]:
    """Native chunk indices required to build *out_components*.

    For cylindrical/annular, ``u_r`` and ``u_θ`` are each formed from
    the ``u_±`` pair, so requesting either pulls native chunks 1 and 2;
    the viscoelastic conformation components similarly pull their spin
    chunks (see :func:`_component_recipes`).
    """
    recipes = _component_recipes(info)
    need: set[int] = set()
    for c in out_components:
        need.update(recipes[c][0])
    return sorted(need)


def read_chunks(
    path: str | Path,
    meta: dict,
    components,
    slab_indices: ndarray | None = None,
) -> dict[int, ndarray]:
    r"""Read native spectral chunks straight off disk (no transpose).

    Each returned component has the snapshot-native shape
    ``(a_size, kx_global, b_size)`` (or ``(len(slab_indices),
    kx_global, b_size)`` when *slab_indices* selects outer-axis slabs).

    *slab_indices* is only valid for the ``walled`` layout, whose outer
    axis is the wall-normal direction (a slab is contiguous on disk, so
    each is a single ``seek`` + ``read``).
    """
    offsets = snapshot_component_offsets(path)
    _, a_size, kx_global, b_size = meta["on_disk_shape"]
    dtype = _np_dtype(meta["dtype"])
    itemsize = dtype.itemsize
    plane = kx_global * b_size

    out: dict[int, ndarray] = {}
    with open(path, "rb") as f:
        for c in components:
            base = offsets[c]
            if slab_indices is None:
                f.seek(base)
                raw = f.read(a_size * plane * itemsize)
                out[c] = (
                    np.frombuffer(raw, dtype=dtype)
                    .reshape(a_size, kx_global, b_size)
                    .copy()
                )
            else:
                slabs = []
                for i in slab_indices:
                    f.seek(base + int(i) * plane * itemsize)
                    raw = f.read(plane * itemsize)
                    slabs.append(
                        np.frombuffer(raw, dtype=dtype).reshape(
                            kx_global, b_size
                        )
                    )
                out[c] = np.stack(slabs, axis=0)
    return out


def to_returned_basis(
    raw: dict[int, ndarray],
    info: GeometryInfo,
    out_components: tuple[int, ...],
) -> dict[int, ndarray]:
    r"""Map native chunks to the returned component basis.

    Cylindrical/annular convert the stored ``u_±`` (and, for the
    viscoelastic system, the conformation spin combos) to the physical
    components; all other families are identity.  See
    :func:`_component_recipes`.
    """
    recipes = _component_recipes(info)
    return {c: recipes[c][1](raw) for c in out_components}


# ── Nearest wall-normal grid points ──────────────────────────


def nearest_unique_indices(grid, points) -> ndarray:
    """Sorted, de-duplicated nearest-grid-point indices for *points*."""
    grid = np.asarray(grid, dtype=float)
    pts = np.atleast_1d(np.asarray(points, dtype=float))
    idx = sorted({int(np.argmin(np.abs(grid - p))) for p in pts})
    return np.array(idx, dtype=int)


# ── Spectral <-> physical transforms (no transpose) ──────────


def _insert_nyquist(a: ndarray, axis: int, n: int) -> ndarray:
    """Reinstate the omitted Nyquist mode (zero) on a full-FFT axis."""
    return np.insert(a, n // 2, 0, axis=axis)


def _append_real_nyquist(a: ndarray, axis: int) -> ndarray:
    """Append the omitted Nyquist mode (zero) on the real-FFT axis."""
    shape = list(a.shape)
    shape[axis] = 1
    return np.concatenate([a, np.zeros(shape, dtype=a.dtype)], axis=axis)


def inverse_transform(field: ndarray, info: GeometryInfo) -> ndarray:
    """Spectral -> physical for one component, in place (no transpose).

    Returns a real array of the physical-space shape.
    """
    out = np.asarray(field)
    real_axis = -1
    for ax in range(3):
        if info.kind[ax] == "complex":
            out = _insert_nyquist(out, ax, info.n[ax])
            out = np.fft.ifft(out, axis=ax, norm="forward")
        elif info.kind[ax] == "real":
            real_axis = ax
    out = _append_real_nyquist(out, real_axis)
    return np.fft.irfft(
        out, n=info.n[real_axis], axis=real_axis, norm="forward"
    )


def forward_transform(field: ndarray, info: GeometryInfo) -> ndarray:
    """Physical -> spectral for one component (inverse of the above)."""
    arr = np.asarray(field)
    real_axis = next(ax for ax in range(3) if info.kind[ax] == "real")
    out = np.fft.rfft(arr.real, axis=real_axis, norm="forward")
    out = np.take(out, np.arange(info.n[real_axis] // 2), axis=real_axis)
    for ax in range(3):
        if info.kind[ax] == "complex":
            out = np.fft.fft(out, axis=ax, norm="forward")
            out = np.delete(out, info.n[ax] // 2, axis=ax)
    return out


def to_physical(field, params: Namespace):
    """Inverse transform a component (or tuple) given a params view."""
    info = geometry_info(params)
    if isinstance(field, (tuple, list)):
        return tuple(inverse_transform(f, info) for f in field)
    return inverse_transform(field, info)


def to_spectral(field, params: Namespace):
    """Forward transform a component (or tuple) given a params view."""
    info = geometry_info(params)
    if isinstance(field, (tuple, list)):
        return tuple(forward_transform(f, info) for f in field)
    return forward_transform(field, info)


# ── Coordinate builders (axis order matches the data) ────────


def physical_grids(info: GeometryInfo, wall_normal_grid) -> tuple:
    """Physical-space coordinates, one array per axis (axis order)."""
    grids = []
    for ax in range(3):
        if info.kind[ax] == "grid":
            grids.append(np.asarray(wall_normal_grid, dtype=float))
        else:
            n, length = info.n[ax], info.length[ax]
            grids.append(np.arange(n) * (length / n))
    return tuple(grids)


def spectral_coords(info: GeometryInfo, wall_normal_grid) -> tuple:
    """Spectral coordinates: wavenumbers, with the wall-normal grid at
    its (grid) axis for wall-bounded geometries."""
    coords = []
    for ax in range(3):
        kind = info.kind[ax]
        if kind == "grid":
            coords.append(np.asarray(wall_normal_grid, dtype=float))
        elif kind == "real":
            coords.append(
                real_harmonics(info.n[ax]) * (2.0 * np.pi / info.length[ax])
            )
        else:
            coords.append(
                complex_harmonics(info.n[ax]) * (2.0 * np.pi / info.length[ax])
            )
    return tuple(coords)


# ── Differentiation / integration primitives (used by ops) ───


def _broadcast_along(vec: ndarray, axis: int, ndim: int = 3) -> ndarray:
    shape = [1] * ndim
    shape[axis] = np.asarray(vec).shape[0]
    return np.asarray(vec).reshape(shape)


def fourier_derivative(field: ndarray, axis: int, wavenumber) -> ndarray:
    r"""Exact spectral derivative ``× i k`` along a Fourier *axis*."""
    kvec = 1j * _broadcast_along(np.asarray(wavenumber), axis)
    return np.asarray(field) * kvec


def _matmul_axis(mat: ndarray, field: ndarray, axis: int) -> ndarray:
    """Contract an FD matrix *mat* against *field* along *axis*."""
    moved = np.moveaxis(field, axis, 0)
    m = mat.astype(field.dtype) if np.iscomplexobj(field) else mat
    out = np.tensordot(m, moved, axes=([1], [0]))
    return np.moveaxis(out, 0, axis)


def parity_radial_d1(grid, fd_order: int) -> tuple[ndarray, ndarray]:
    r"""Parity-reduced radial ``D1`` pair ``(D1_even, D1_odd)``.

    Mirrors ``build_parity_reduced_matrices`` in
    :mod:`dnsjax.geometries.wall_bounded.cylindrical`: ``D1`` is built
    on the auxiliary mirrored grid ``{-r[::-1], r}`` and reduced with
    the axis parity relation ``u(-r) = (-1)^{m_eff} u(r)``::

        D1_even = D1_pos + D1_ghost_flipped
        D1_odd  = D1_pos - D1_ghost_flipped

    Used for the pipe (cylindrical), whose radial grid reaches the axis
    ``r -> 0``; the annulus has no axis and uses a plain ``D1``.
    """
    rs = np.asarray(grid, dtype=float)
    nr = len(rs)
    aux = np.concatenate([-rs[::-1], rs])
    d1_full, _ = build_diff_matrices(aux, int(fd_order))
    pos = d1_full[nr:, nr:]
    ghost_flipped = d1_full[nr:, :nr][:, ::-1]
    return pos + ghost_flipped, pos - ghost_flipped


def radial_derivative(
    field: ndarray,
    grid,
    fd_order: int,
    info: GeometryInfo,
    parity: str | None = None,
) -> ndarray:
    r"""First derivative along the wall-normal/radial grid axis (axis 0).

    *parity* is ``None`` for the cartesian wall-normal axis and the
    annular radius (a plain ``D1`` matmul).  For the pipe it selects the
    parity-reduced operator per azimuthal mode ``m`` (axis 2): ``"uz"``
    for the ``(-1)^m`` parity of ``u_z``; ``"utheta"`` for the
    ``(-1)^{m+1}`` parity of ``u_r`` / ``u_θ``.
    """
    field = np.asarray(field)
    if parity is None:
        gr = np.asarray(grid, dtype=float)
        d1, _ = build_diff_matrices(gr, int(fd_order))
        return _matmul_axis(d1, field, 0)
    d1_even, d1_odd = parity_radial_d1(grid, fd_order)
    even = _matmul_axis(d1_even, field, 0)
    odd = _matmul_axis(d1_odd, field, 0)
    m_even = complex_harmonics(info.n[2]) % 2 == 0
    use_even = m_even if parity == "uz" else ~m_even
    return np.where(use_even.reshape(1, 1, -1), even, odd)


def derivative_axis(
    field: ndarray,
    axis: int,
    info: GeometryInfo,
    coord: ndarray,
    fd_order: int,
    parity: str | None = None,
) -> ndarray:
    r"""First derivative of a spectral field along one on-disk *axis*.

    Fourier axis: ``× i k`` (exact, per stored mode).  Grid axis
    (always axis 0 for wall-bounded families): the finite-difference
    ``D1`` on *coord*, parity-reduced for the pipe when *parity* is set.
    """
    if info.kind[axis] in ("real", "complex"):
        return fourier_derivative(field, axis, coord)
    return radial_derivative(field, coord, fd_order, info, parity=parity)
