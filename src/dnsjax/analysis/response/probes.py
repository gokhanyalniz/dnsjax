r"""JAX-free reader for the runtime spectral-mode probe stream.

Reads the ``probes.bin``/``probes.json`` pair written by
:mod:`dnsjax.probes` (see that docstring for the record layout and the
writer's append-on-resume rules) and provides the small host-side
post-processing steps that need no device: time-averaged mean
profiles, the friction Reynolds number, and profile files consumable
by the transient-growth CLI (``--profile``).

Depends only on NumPy, the standard library, and the JAX-free
:mod:`dnsjax.fd` leaf, so it is safe anywhere the
``import dnsjax.analysis`` guarantee applies.

Conventions
===========
- ``u`` is the stored **perturbation** state's mode profiles (the
  total field for the force-driven Dean systems); the mean mode
  ``(0,0)`` therefore records the perturbation's instantaneous mean
  profile, and :func:`mean_profile` adds the closed-form laminar
  profile back to return the **total** streamwise mean.
- Sample times are uniform by construction (``t0 + k
  \cdot it_probes \cdot dt``); a resumed stream that re-ran a
  trajectory segment shows up as non-monotonic ``t`` and is flagged
  with a warning (filter with ``t`` yourself in that case).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...fd import build_diff_matrices

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
    spectral indices, ``wavenumbers`` the matching integer harmonics
    (multiply by `$2\pi/L$` for physical wavenumbers).  ``meta`` is
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


def _resolve_pair(path: str | Path) -> tuple[Path, Path]:
    """Map a run directory / ``probes.bin`` path to the file pair."""
    path = Path(path)
    if path.is_dir():
        return path / "probes.bin", path / "probes.json"
    if path.suffix == ".json":
        return path.with_suffix(".bin"), path
    return path, path.with_suffix(".json")


def read_probes(path: str | Path = ".") -> ProbeData:
    """Load a probe stream (a run directory or the ``probes.bin``).

    Reconstructs the record dtype from the sidecar, reads every whole
    record (a truncated trailing record -- e.g. from a killed writer
    -- is dropped with a warning), and returns complex128 profiles.
    Exact-duplicate consecutive timestamps -- the benign seam of a
    clean continuation resume, where the parent's final sample and
    the child's t0 sample record the same state -- are dropped
    (keeping the last) with a note, so downstream consumers see a
    uniform grid across resumes; genuinely decreasing timestamps (a
    re-run trajectory segment) still only warn.
    """
    bin_path, json_path = _resolve_pair(path)
    if not json_path.exists():
        raise FileNotFoundError(f"probe sidecar {json_path} not found")
    with open(json_path) as f:
        meta = json.load(f)

    modes = np.asarray(meta["modes"], dtype=int)
    n_components = int(meta["n_components"])
    ny = int(meta["ny"])
    record_dtype = np.dtype(
        [
            ("t", "<f8"),
            (
                "u",
                meta["value_dtype"],
                (len(modes), n_components, ny, 2),
            ),
        ]
    )

    raw = np.fromfile(bin_path, dtype=np.uint8)
    n_rec, rem = divmod(raw.size, record_dtype.itemsize)
    if rem:
        print(
            f"[probes] {bin_path}: dropping a truncated trailing "
            f"record ({rem} of {record_dtype.itemsize} bytes)."
        )
    rec = np.frombuffer(raw.tobytes(), dtype=record_dtype, count=n_rec)

    t = rec["t"].astype(np.float64)
    if n_rec > 1:
        dup = np.diff(t) == 0.0
        if dup.any():
            keep = np.ones(n_rec, dtype=bool)
            keep[:-1][dup] = False  # keep the last of each seam
            print(
                f"[probes] {bin_path}: dropped {int(dup.sum())} "
                "duplicate-timestamp record(s) (continuation-resume "
                "seams)."
            )
            rec, t = rec[keep], t[keep]
    if len(t) > 1 and not (np.diff(t) > 0).all():
        print(
            f"[probes] {bin_path}: non-monotonic timestamps (a resume "
            "re-ran a trajectory segment?); filter by t before "
            "averaging."
        )
    u = rec["u"][..., 0].astype(np.complex128)
    u += 1j * rec["u"][..., 1]

    return ProbeData(
        t=t,
        u=u,
        modes=modes,
        wavenumbers=np.asarray(meta["wavenumbers"], dtype=int),
        y=np.asarray(meta["wall_normal_grid"], dtype=float),
        component_labels=list(meta["component_labels"]),
        meta=meta,
    )


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
    ``(y, U_s)`` with ``y`` in solver order (descending from the top
    wall) -- directly consumable by :func:`write_profile_file`.
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
    p = int(data.meta["params"]["res"]["fd_order"])
    d1, _ = build_diff_matrices(y, p)
    re = float(data.meta["params"]["phys"]["re"])
    du_top = abs(float(d1[0] @ u_s))
    du_bot = abs(float(d1[-1] @ u_s))
    return math.sqrt(re * 0.5 * (du_top + du_bot))


def write_profile_file(
    path: str | Path, y: np.ndarray, profile: np.ndarray
) -> None:
    """Write a two-column total-profile file (top wall first).

    The format the transient-growth CLI ``--profile`` consumes: grid
    points descending from the top wall, then the total profile value.
    An ascending input grid is flipped.
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
