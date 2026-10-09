r"""The twin's 3-D ``(y, k_z, k_x)`` cubes: one dnsjax tar per sample.

Three directories of a ``dnsjax-twin`` run directory, each holding one
file per sample, ``<kind>/<kind>_{it:010d}.tar``:

- ``twin_spectra3d`` (``twin.it_spectra3d``): ``e_u, e_v, e_w``;
- ``twin_spectra3d_ref`` (``twin.it_spectra3d_ref``, by default the
  same steps; under ``twin.spectra_ref``): ``r_u, r_v, r_w``;
- ``twin_budget3d`` (``twin.it_budget3d``): ``P_U, P_r, T_ref,
  T_self, V, eps, Wp``.

``e`` / ``r`` are the difference field's and the reference state's
componentwise energy densities, the budget terms the convective
``twin_ybudget`` terms (:mod:`dnsjax.twin.diagnostics`, "Spectral
budget") -- every term the budget figures are drawn from, and not
their sum.  Each is computed at the run's full resolution, then folded
and subsampled (:func:`dnsjax.twin.diagnostics.cube_selection`):

- `$y$` folded about the centreline, the **mean** of the two halves at
  each wall distance (``twin_spectral_maps.py --half mean``), stored
  on the lower half's rows ascending from the wall;
- `$k_z$` folded onto `$|k_z|$`, the **sum** of each `$\pm k_z$` pair
  (the `$(y, k)$` streams' fold), `$k_x \ge 0$` as stored;
- each axis subsampled uniformly in the logarithm of its coordinate --
  the wall distance, `$|k_z|$`, `$k_x$` -- by its own count
  (``twin.n_y3d`` / ``n_kz3d`` / ``n_kx3d``), the wall row and
  `$k = 0$` always kept.

So entry `$(0, 0)$` of the two wavenumber axes is the mean mode alone
(the ``*_xz00`` of the 2-D streams), and in ``Wp`` it is the driving's
input.  Values have the 2-D streams' meaning: `$y$`-densities divided
by ``volume_fac``, carrying the conjugate-pair weight ``k_metric``.
With no subsampling, summing a cube over `$k_x$` (`$|k_z|$`) gives the
`$y$`-folded ``*_x`` (``*_z``) marginal of the matching 2-D stream.

The container
-------------
The snapshot format (:mod:`dnsjax.snapshot`), written by its own
:func:`~dnsjax.snapshot.write_archive`: an uncompressed tar of
``_dnsjax_meta.json`` and one zarr3 ``state/`` array of real
(``float64`` / ``float32``) chunks, one per field, **y-major**
``(field, y, k_z, k_x)`` like every snapshot -- so a reader pulls single
wall distances as contiguous slabs.  The metadata ``kind`` keeps these
apart from states: the field readers and a resume refuse them
(:func:`dnsjax.snapshot_meta.checkpoint_refusal`), and
:func:`dnsjax.analysis.twin.read_cube` reads them.  It also carries
what a file needs to be read alone: the kept rows (``iy``, ``y``,
``wall_distance``), harmonics (``kz_harmonics`` = `$|m|$`,
``kx_harmonics``), the domain, the counts, the cadence and the twin
provenance.

``cube_version`` is the schema's version, checked against the reader's
floor; bump both when a stored meaning changes.

A file is written atomically and replaces one of the same name, so a
resume rewrites its first sample.  A resumed member's files must agree
with the configured run on the keys in :data:`_MATCH_KEYS` -- checked
against the newest file already in the directory -- or the resume is
refused rather than mixing samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jax import Array

from ..param_surface import recorded_params_dump
from ..parameters import derived_params, params
from ..sharding import sharding
from ..snapshot import replicated_to_io_layout, write_archive
from ..snapshot_meta import git_hash, read_snapshot_meta
from .diagnostics import CubeSelection

#: The cube schema version (reader floor:
#: ``analysis.twin.cubes.MIN_CUBE_VERSION``).
CUBE_VERSION: int = 1

#: The three kinds, each its own directory and file stem.
SPECTRA3D: str = "twin_spectra3d"
SPECTRA3D_REF: str = "twin_spectra3d_ref"
BUDGET3D: str = "twin_budget3d"

#: Metadata that must agree between a resumed member's existing files
#: and the configured run (the cadence key is added per kind).
_MATCH_KEYS: tuple[str, ...] = (
    "kind",
    "cube_version",
    "system",
    "native_shape",
    "dtype",
    "fields",
    "iy",
    "kz_harmonics",
    "kx_harmonics",
    "ny",
    "nz",
    "nx",
    "lx",
    "lz",
    "parent_it",
    "dt",
    "double_precision",
)


class CubeStore:
    """One kind's directory of cube tars (module docstring).

    Construct once per run, then :meth:`write` each sample; it returns
    the non-finite diagnostic the driver aborts on (``None`` when
    clean).  Collective: every process constructs it and calls
    :meth:`write` at the same steps.
    """

    def __init__(
        self,
        kind: str,
        fields: tuple[str, ...],
        sel: CubeSelection,
        cadence_key: str,
        cadence: int,
        twin_values,
        parent_it: int,
        directory: str | Path = ".",
    ) -> None:
        self.kind = kind
        self.fields = tuple(fields)
        self.directory = Path(directory) / kind
        y = np.asarray(derived_params.wall_normal_grid, dtype=np.float64)
        self._cadence_key = cadence_key
        self._static = {
            "format_version": 6,
            "kind": kind,
            "cube_version": CUBE_VERSION,
            "system": params.phys.system,
            "native_shape": [
                len(self.fields),
                len(sel.iy),
                len(sel.kz_pos),
                len(sel.kx),
            ],
            "dtype": "float64" if params.res.double_precision else "float32",
            "fields": list(self.fields),
            "axes": ["field", "wall_distance", "kz", "kx"],
            "ny": params.res.ny,
            "nz": params.res.nz,
            "nx": params.res.nx,
            "iy": list(sel.iy),
            "y": [float(y[j]) for j in sel.iy],
            "wall_distance": [float(1.0 + y[j]) for j in sel.iy],
            "kz_harmonics": list(sel.kz_pos),
            "kx_harmonics": list(sel.kx),
            "lx": params.geo.lx,
            "lz": params.geo.lz,
            "volume_fac": derived_params.volume_fac,
            "selection": {
                "n_y3d": twin_values.n_y3d,
                "n_kz3d": twin_values.n_kz3d,
                "n_kx3d": twin_values.n_kx3d,
            },
            cadence_key: cadence,
            "parent_it": parent_it,
            "dt": params.step.dt,
            "double_precision": params.res.double_precision,
            "twin": {
                "seed": twin_values.seed,
                "e0": twin_values.e0,
                "smoothness": twin_values.smoothness,
                "wall_smoothness": twin_values.wall_smoothness,
                "wall_confinement": twin_values.wall_confinement,
            },
            "note": (
                "y-densities / volume_fac, k_metric weighted; y folded "
                "(mean of the halves, lower-half rows from the wall), "
                "k_z folded onto |k_z| (sum of the +- pair); each axis "
                "subsampled log-uniformly after the full-resolution "
                "computation; entry (k_z, k_x) = (0, 0) is the mean mode"
            ),
        }
        self._check_existing()

    def _check_existing(self) -> None:
        """Refuse a directory whose newest file disagrees with the run.

        Every process reads (a shared filesystem); none writes.
        """
        existing = sorted(self.directory.glob(f"{self.kind}_*.tar"))
        if not existing:
            return
        meta = read_snapshot_meta(existing[-1])
        keys = (*_MATCH_KEYS, self._cadence_key)
        mismatch = [k for k in keys if meta.get(k) != self._static[k]]
        if mismatch:
            raise SystemExit(
                f"[twin] {existing[-1]} does not match this run "
                f"(differs in: {', '.join(mismatch)}); move the "
                f"{self.directory} directory away to start afresh."
            )

    def write(self, cube: Array, t: float, it: int) -> str | None:
        """Write one sample as a dnsjax tar.

        *cube* is ``(n_fields, n_y, n_kz, n_kx)``, replicated
        (``diagnostics._cube_replicated``).
        """
        path = self.directory / f"{self.kind}_{it:010d}.tar"
        static = self._static

        def meta() -> dict:
            return static | {
                "git_hash": git_hash(),
                "t": t,
                "it": it,
                "n_devices": sharding.n_devices,
                "params": recorded_params_dump(params),
            }

        write_archive(
            [("state", replicated_to_io_layout(cube))],
            tuple(static["native_shape"][1:]),
            static["dtype"],
            path,
            meta=meta,
        )
        # Post-write, like every stream: the offending file stays on
        # disk for the post-mortem.
        if not sharding.main_device:
            return None
        finite = np.isfinite(np.asarray(cube))
        if finite.all():
            return None
        field = self.fields[int(np.argwhere(~finite)[0][0])]
        return f"non-finite {self.kind} value in {field} at t = {t:.6e}"
