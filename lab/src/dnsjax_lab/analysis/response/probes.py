r"""JAX-free reader for the runtime spectral-mode probe stream.

Reads the ``probes.bin``/``probes.json`` pair written by
:mod:`dnsjax_lab.extensions.probes` (see that docstring for the record layout
and the writer's append-on-resume rules) and provides the small host-side
post-processing steps that need no device: time-averaged mean profiles, the
friction Reynolds number, and profile files consumable by the transient-growth
CLI (``--tg.profile``).

Depends only on NumPy, the standard library, and the JAX-free
:mod:`dnsjax.fd` leaf, so importing it pulls in neither JAX nor SciPy.

Conventions
-----------
- ``u`` is the stored **perturbation** state's mode profiles (the
  total field for the force-driven systems,
  :data:`~dnsjax.analysis._core.TOTAL_FIELD_SYSTEMS`); the mean mode
  ``(0,0)`` therefore records the perturbation's instantaneous mean
  profile, and :func:`mean_profile` adds the closed-form laminar
  profile back to return the **total** streamwise mean.
- Sample times are uniform (``t0 + k * it_probes * dt``) unless a
  resume broke the grid: one from a snapshot off the probe cadence
  records its initial state off the grid, and one that re-ran a
  trajectory segment shows up as non-monotonic ``t`` and is flagged
  with a warning (filter with ``t`` yourself in either case).
- A stream reaches several GB.  :func:`read_probes` holds all of it;
  :func:`iter_probes` reads it in blocks of records, in time order,
  for a single pass that holds one block at a time.  Both take
  ``read_cap``, in MB/s, to read by ``pread`` at most that fast
  (:mod:`dnsjax_lab.analysis.streamio`); unset, the process's cap
  applies, full speed unless one is set.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dnsjax.fd import build_diff_matrices
from dnsjax_lab.analysis import streamio

#: Oldest ``probes.json`` schema this reader accepts (the writer's
#: current version is ``dnsjax_lab.extensions.probes.FORMAT_VERSION``).
#: Version-1 streams carry the cylindrical/annular columns in the solver's
#: decoupled `$(u_z, u_+, u_-)$` basis at the same layout, so every
#: positional consumer downstream would misread them silently.
#: Version-2 streams additionally record the axis-2 ``wavenumbers``
#: of an azimuthal wedge as the harmonic index rather than the
#: physical `$m = m_0 j$`, so anything deriving a wavelength or a
#: spectrum from them is wrong by `$m_0$` (correct only at
#: ``geo.m0 = 1``).
MIN_FORMAT_VERSION: int = 3

#: Bytes of records an :func:`iter_probes` block holds at most (at
#: least one record): with the complex copy of their profiles, a block
#: takes about twice this in memory.
_BLOCK_BYTES: int = 64 * 2**20

#: Laminar streamwise total profiles `$U_s(y)$` of the Cartesian
#: base-flow systems (`$y \in [-1, 1]$`; the tilted frame's
#: streamwise ``s`` projection of the base flow is `$U_s$` itself).
_CARTESIAN_LAMINAR = {
    "plane-couette": lambda y: y,
    "plane-poiseuille": lambda y: 1.0 - y**2,
}


@dataclass(frozen=True)
class ProbeData:
    r"""One probe stream, fully loaded.

    ``u`` has shape ``(n_t, K, C, N_y)`` (complex128): ``K`` probed
    modes in sidecar order, ``C`` state components
    (``component_labels``).  ``modes`` are the ``(i2, i3)`` global
    spectral indices, ``wavenumbers`` the matching harmonics (multiply
    the Cartesian and axial ones by `$2\pi/L$` for physical
    wavenumbers; the azimuthal one is already `$m = m_0 j$`).  ``meta`` is
    the full sidecar dict (``meta["params"]`` is the writing run's
    resolved parameter dump).
    """

    t: np.ndarray
    u: np.ndarray
    modes: np.ndarray
    wavenumbers: np.ndarray
    y: np.ndarray
    component_labels: list[str]
    meta: dict

    def mode_index(self, i2: int, i3: int) -> int:
        """Index of mode ``(i2, i3)`` along ``u``'s axis 1."""
        hits = np.nonzero((self.modes[:, 0] == i2) & (self.modes[:, 1] == i3))[
            0
        ]
        if hits.size == 0:
            raise KeyError(
                f"mode ({i2},{i3}) is not in this stream "
                f"(probed: {self.modes.tolist()})"
            )
        return int(hits[0])


def _resolve_pair(path: str | Path, stem: str = "probes") -> tuple[Path, Path]:
    """Map a run directory / ``<stem>.bin`` path to the file pair.

    Shared with the forcing-stream reader (``stem="forcing"``,
    :mod:`dnsjax_lab.analysis.response.ssi`).
    """
    path = Path(path)
    if path.is_dir():
        return path / f"{stem}.bin", path / f"{stem}.json"
    if path.suffix == ".json":
        return path.with_suffix(".bin"), path
    return path, path.with_suffix(".json")


def _check_format_version(json_path: Path, meta: dict, minimum: int) -> None:
    """Reject a sidecar older than *minimum*.

    The stream analogue of
    :func:`dnsjax.snapshot_meta.read_snapshot_meta`'s version choke
    point (an old stream reads without error: the
    :mod:`dnsjax_lab.extensions.probes` docstring).  Shared with the
    forcing-stream reader (:mod:`dnsjax_lab.analysis.response.ssi`).
    """
    version = meta.get("format_version", 0)
    if version < minimum:
        raise ValueError(
            f"{json_path} has format_version {version}; this reader "
            f"takes version {minimum}+ only.  The record layout is "
            "unchanged, but the component basis, the azimuthal "
            "wavenumbers of a wedge (and, for forcing, the "
            "conjugate-partner rule) changed representation, so an "
            "older stream would be misread silently rather than fail "
            "-- see the writer's FORMAT_VERSION comment."
        )


def _layout(path: str | Path) -> tuple[Path, dict, np.dtype, int]:
    """``(bin_path, meta, record dtype, complete records)`` of a stream.

    Reads the sidecar, refuses one older than :data:`MIN_FORMAT_VERSION`
    and reports a truncated trailing record (a killed writer), which
    neither reader reads.
    """
    bin_path, json_path = _resolve_pair(path)
    if not json_path.exists():
        raise FileNotFoundError(f"probe sidecar {json_path} not found")
    with open(json_path) as f:
        meta = json.load(f)
    _check_format_version(json_path, meta, MIN_FORMAT_VERSION)
    shape = (
        len(meta["modes"]),
        int(meta["n_components"]),
        int(meta["ny"]),
        2,
    )
    record_dtype = np.dtype([("t", "<f8"), ("u", meta["value_dtype"], shape)])
    n_rec, rem = divmod(bin_path.stat().st_size, record_dtype.itemsize)
    if rem:
        print(
            f"[probes] {bin_path}: dropping a truncated trailing "
            f"record ({rem} of {record_dtype.itemsize} bytes)."
        )
    return bin_path, meta, record_dtype, n_rec


def _seam_mask(t: np.ndarray, bin_path: Path) -> np.ndarray:
    """Which records to keep: the last of each run of equal times.

    Exact-duplicate consecutive timestamps are a clean continuation's
    seam (the :mod:`dnsjax_lab.extensions.probes` docstring); each
    drop is noted, and non-monotonic times among the kept records (a
    resume that re-ran a trajectory segment) are warned of and kept.
    """
    keep = np.ones(t.size, dtype=bool)
    if t.size > 1:
        dup = np.diff(t) == 0.0
        if dup.any():
            keep[:-1][dup] = False  # keep the last of each seam
            print(
                f"[probes] {bin_path}: dropped {int(dup.sum())} "
                "duplicate-timestamp record(s) (continuation-resume "
                "seams)."
            )
    kept = t[keep]
    if kept.size > 1 and not (np.diff(kept) > 0).all():
        print(
            f"[probes] {bin_path}: non-monotonic timestamps (a resume "
            "re-ran a trajectory segment?); filter by t before "
            "averaging."
        )
    return keep


def _probe_data(meta: dict, t: np.ndarray, raw: np.ndarray) -> ProbeData:
    """The records' times *t* and ``u`` fields *raw* as a :class:`ProbeData`.

    Real and imaginary parts are written in place: no complex
    temporary beside ``u`` (the streams reach several GB).
    """
    u = np.empty(raw.shape[:-1], dtype=np.complex128)
    u.real = raw[..., 0]
    u.imag = raw[..., 1]
    return ProbeData(
        t=t,
        u=u,
        modes=np.asarray(meta["modes"], dtype=int),
        wavenumbers=np.asarray(meta["wavenumbers"], dtype=int),
        y=np.asarray(meta["wall_normal_grid"], dtype=float),
        component_labels=list(meta["component_labels"]),
        meta=meta,
    )


def read_probes(
    path: str | Path = ".", *, read_cap: float | None = None
) -> ProbeData:
    """Load a probe stream (a run directory or the ``probes.bin``).

    Reconstructs the record dtype from the sidecar, reads every whole
    record (a truncated trailing record -- e.g. from a killed writer
    -- is dropped with a warning), and returns complex128 profiles.
    Exact-duplicate consecutive timestamps (a clean continuation's
    seam: the :mod:`dnsjax_lab.extensions.probes` docstring) are
    dropped, keeping the last, with a note, so downstream consumers
    see a uniform grid across such resumes; genuinely decreasing
    timestamps (a re-run trajectory segment) still only warn.

    *read_cap* (MB/s) reads the stream by ``pread`` at most that fast
    (module docstring).  A stream too large to hold:
    :func:`iter_probes`.
    """
    bin_path, meta, record_dtype, n_rec = _layout(path)
    # Sized read straight into the record dtype: no intermediate
    # byte-array copies (the streams reach multi-GB).
    rec = streamio.read_records(
        bin_path,
        record_dtype,
        stop=n_rec,
        throttle=streamio.throttle(read_cap),
    )
    t = rec["t"].astype(np.float64)
    keep = _seam_mask(t, bin_path)
    if not keep.all():
        rec, t = rec[keep], t[keep]
    return _probe_data(meta, t, rec["u"])


def iter_probes(
    path: str | Path = ".",
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    block_bytes: int = _BLOCK_BYTES,
    read_cap: float | None = None,
) -> Iterator[ProbeData]:
    """A probe stream in blocks of records, in time order, read once.

    The records :func:`read_probes` returns -- the same seams dropped,
    the same notes printed -- with ``t_min <= t <= t_max`` (either
    bound optional), as :class:`ProbeData` blocks of consecutive
    records, at most *block_bytes* of them and at least one record
    each: concatenated, they are :func:`read_probes`' ``t`` and ``u``
    inside that window.  For a single pass over a stream too large to
    hold, such as a covariance accumulation.

    The times are read first, one ``pread`` per record
    (:func:`dnsjax_lab.analysis.streamio.read_column`), then each
    block in one read; nothing before the window's first record or
    after its last is read.  *read_cap* as for :func:`read_probes`,
    one throttle for the whole pass.

    Refused: a window holding no record (as :func:`mean_profile`
    refuses one), and kept times inside it that do not rise strictly
    -- a resume that re-ran a trajectory segment, whose overlap the
    pass would count twice; raise *t_min* past it.
    """
    bin_path, meta, record_dtype, n_rec = _layout(path)
    limit = streamio.throttle(read_cap)
    t = streamio.read_column(
        bin_path, record_dtype, "t", throttle=limit
    ).astype(np.float64)
    keep = _seam_mask(t, bin_path)
    if t_min is not None:
        keep &= t >= t_min
    if t_max is not None:
        keep &= t <= t_max
    rows = np.flatnonzero(keep)
    if rows.size == 0:
        covers = (
            f"the stream covers t = {t.min():g} .. {t.max():g}"
            if t.size
            else "the stream holds no complete record"
        )
        raise ValueError(
            f"{bin_path}: no probe samples with {t_min} <= t <= {t_max} "
            f"({covers})"
        )
    if rows.size > 1 and not (np.diff(t[rows]) > 0).all():
        raise ValueError(
            f"{bin_path}: the kept times between t = {t[rows[0]]:g} and "
            f"{t[rows[-1]]:g} do not rise strictly (a resume re-ran a "
            "trajectory segment); raise t_min past the overlap"
        )
    per = max(1, block_bytes // record_dtype.itemsize)
    first, last = int(rows[0]), int(rows[-1]) + 1
    for start in range(first, last, per):
        stop = min(start + per, last)
        mask = keep[start:stop]
        if not mask.any():
            continue
        rec = streamio.read_records(
            bin_path, record_dtype, start=start, stop=stop, throttle=limit
        )
        yield _probe_data(meta, t[start:stop][mask], rec["u"][mask])


def _time_mask(data: ProbeData, t_min: float) -> np.ndarray:
    mask = data.t >= t_min
    if not mask.any():
        raise ValueError(
            f"no probe samples at t >= {t_min} "
            f"(stream covers t = {data.t[0]:g} .. {data.t[-1]:g})"
        )
    return mask


def mean_profile(
    data: ProbeData, t_min: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""Time-averaged **total** streamwise mean profile `$U_s(y)$`.

    Averages the ``(0,0)`` mode over ``t >= t_min`` (discard the
    transient!), projects onto the tilted streamwise direction
    (`$u_s = u_x\cos\vartheta + u_z\sin\vartheta$`), takes the real
    part (the mean mode of a real field), and adds the closed-form
    laminar profile.  Cartesian base-flow systems only.  Returns
    ``(y, U_s)`` with ``y`` exactly as the stream stores it, i.e. the
    solver's **ascending** grid (bottom wall `$y = -1$` first);
    :func:`write_profile_file` flips it to the top-wall-first order
    the transient-growth ``--tg.profile`` reader expects.
    """
    system = data.meta["system"]
    if system not in _CARTESIAN_LAMINAR:
        raise ValueError(
            f"mean_profile supports {sorted(_CARTESIAN_LAMINAR)}; "
            f"got {system!r}"
        )
    k = data.mode_index(0, 0)
    mask = _time_mask(data, t_min)
    mean_hat = data.u[mask, k].mean(axis=0)  # (C, Ny)

    tilt = math.radians(data.meta["params"]["geo"].get("tilt_degree") or 0.0)
    u_s = (mean_hat[0] * math.cos(tilt) + mean_hat[2] * math.sin(tilt)).real
    return data.y, _CARTESIAN_LAMINAR[system](data.y) + u_s


def re_tau(data: ProbeData, t_min: float = 0.0) -> float:
    r"""Friction Reynolds number of the time-averaged total profile.

    `$Re_\tau = u_\tau h/\nu = \sqrt{Re\,\lvert dU_s/dy\rvert_w}$` in
    code units (`$h = 1$`, `$\nu = 1/Re$`), averaged over both walls;
    the wall derivative uses the run's own ``fd_order`` one-sided
    stencils (:func:`dnsjax.fd.build_diff_matrices` boundary rows).
    """
    y, u_s = mean_profile(data, t_min)
    # Raw public-named sidecar access: ``fd_order``/``re`` are on the
    # Cartesian surfaces (the only ones ``mean_profile`` admits).  A
    # flow whose ``re`` is derived (Taylor-Couette / quasi-Keplerian /
    # viscoelastic-dean) does not record it -- go through
    # ``flows.registry.internalize_stored(..., rehydrate=True)`` if
    # this is ever un-gated.
    p = int(data.meta["params"]["res"]["fd_order"])
    d1, _ = build_diff_matrices(y, p)
    re = float(data.meta["params"]["phys"]["re"])
    # ``y`` is ascending, so row 0 is the *bottom* wall and row -1 the
    # top one; the magnitudes are averaged, so only the naming cares.
    du_lo = abs(float(d1[0] @ u_s))
    du_hi = abs(float(d1[-1] @ u_s))
    return math.sqrt(re * 0.5 * (du_lo + du_hi))


def write_profile_file(
    path: str | Path, y: np.ndarray, profile: np.ndarray
) -> None:
    """Write a two-column total-profile file (top wall first).

    The format the transient-growth CLI ``--tg.profile`` consumes:
    grid points descending from the top wall, then the total profile
    value.  An ascending input grid is flipped.
    """
    y = np.asarray(y, dtype=float)
    profile = np.asarray(profile, dtype=float)
    if y.shape != profile.shape or y.ndim != 1:
        raise ValueError(
            f"y {y.shape} and profile {profile.shape} must be equal-"
            "length 1-D arrays"
        )
    if y[0] < y[-1]:
        y, profile = y[::-1], profile[::-1]
    with open(path, "w") as f:
        for yi, ui in zip(y, profile, strict=True):
            f.write(f"{yi:+.17e} {ui:+.17e}\n")
