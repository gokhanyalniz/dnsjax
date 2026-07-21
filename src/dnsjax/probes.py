r"""Spectral-mode probe stream: per-step `$\hat{u}(y)$` time series.

Records the complex wall-normal profiles of a small set of global
spectral modes -- ``probes.modes`` (the ``probes`` extension section;
:mod:`dnsjax.extensions`), an ``"i2,i3;..."`` list in the
stored-layout index convention (axis 2 = complex slot, axis 3 =
real-FFT slot; :func:`dnsjax.harmonics.parse_mode_pairs`) -- every
``probes.it_probes`` steps into a binary ``probes.bin`` next to the
``.dat`` diagnostic streams.  Wall-bounded systems only (the state
layout is ``(C, N_y, N_{k_2}, N_{k_3})``).  Unlike the transient-growth
CLI, the mean mode ``(0,0)`` **is** allowed: its record is the
instantaneous mean profile of the perturbation (add the closed-form
laminar profile for the total; see the reader).

Why a fourth stream: a mode time series at `$O(10^5$--$10^6)$` samples
is far beyond what per-snapshot output can carry (a snapshot per sample
is ~three orders of magnitude more bytes), while the scalar ``.dat``
text streams cannot hold complex per-`$y$` profiles.  The probe stream
is the input for mode statistics (covariances, spectra) and for
ensemble-averaged response curves.

File format
===========
``probes.bin`` is a flat sequence of fixed-size records,

.. code-block:: python

    numpy.dtype([("t", "<f8"), ("u", VAL, (K, C, N_y, 2))])

with ``K`` probed modes, ``C`` state components, the trailing axis
``(re, im)``, and ``VAL = "<f8"``/``"<f4"`` following
``res.double_precision`` (the values carry state precision; ``t`` is
always float64).  A ``probes.json`` sidecar (written once) carries the
schema: the mode list, integer wavenumbers, component labels, the
wall-normal grid, cadence, and the full resolved parameter dump.  The
JAX-free reader is :mod:`dnsjax.analysis.response.probes`.

The record layout is fixed across schema versions, so a stale stream
reads *cleanly* and only its values mean something else -- which makes
:data:`FORMAT_VERSION` the only thing between an old file and a silent
misread.  Bump it whenever the stored meaning changes (not just the
layout), and raise the reader's ``MIN_FORMAT_VERSION`` with it.

Resume semantics: an existing ``probes.bin``/``probes.json`` pair is
appended to iff the sidecar matches the current run (same modes, grid,
components, precision, system, and cadence) -- anything else is a hard
error asking the user to move the old pair away.  A clean continuation
duplicates one sample per seam (the parent's cadence-aligned final
record and the child's t0 record hold the same state at the same
``t``); the reader drops these and flags genuinely non-monotonic
timestamps (a re-run trajectory segment).

Sharded gather
==============
The state's mode axes are sharded (`$k_2$` by ``np0``, `$k_3$` by
``np1``), and slicing a sharded axis of the global array is not
supported under explicit sharding; :func:`build_mode_extractor`
instead gathers inside a ``shard_map`` -- the owning device is
computed from the *local* shard shape and every device contributes
either the column or zeros to a ``psum`` (the
``extract_mean_mode`` pattern of
:mod:`dnsjax.geometries.wall_bounded._base`, generalised to arbitrary
static mode indices).  Probed indices always address *true* modes
(``validate_parameters`` bounds them by the unpadded mode counts), so
spectral padding slots are never read.

Buffering matches the ``.dat`` streams: an on-device
``(nbuffer, K, C, N_y)`` complex buffer plus a host timestamp list,
flushed (appended + ``fsync``-ed) by the main process when full, at
shutdown, before every snapshot, and on a termination signal
(``flush_all_buffers`` in :mod:`dnsjax.__main__`); flushed records are
scanned for non-finite values and a hit aborts the run through the
same FATAL / exit-3 path as the other diagnostics.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
from jax import Array, lax, shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from .extensions import probes_params
from .harmonics import complex_harmonics, parse_mode_pairs, real_harmonics
from .param_surface import recorded_params_dump
from .parameters import (
    cartesian_systems,
    derived_params,
    params,
    viscoelastic_systems,
)
from .sharding import sharding
from .snapshot_meta import git_hash

#: Sidecar schema version.  Version 2 records the switch of the
#: cylindrical/annular columns from the solver's decoupled
#: `$(u_z, u_+, u_-)$` basis (and the conformation spin components)
#: to the physical `$(u_z, u_r, u_\theta)$` one: the *values* changed
#: meaning at fixed layout, so a version-1 stream fed to a
#: positional consumer (``response.lim`` / ``response.ssi``) would be
#: silently misread.  The reader's floor is
#: ``analysis.response.probes.MIN_FORMAT_VERSION``.
FORMAT_VERSION: int = 2

#: Sidecar keys that must match for an append (resume) to proceed.
_MATCH_KEYS: tuple[str, ...] = (
    "format_version",
    "modes",
    "n_components",
    "ny",
    "value_dtype",
    "system",
    "it_probes",
    "dt",
)


def _component_labels(n_components: int) -> list[str]:
    """Component labels of the stored state for the current system."""
    if params.phys.system in cartesian_systems:
        labels = ["u_x", "u_y", "u_z"]
    elif params.phys.system in viscoelastic_systems:
        # Velocity + the 6 stored physical conformation-tensor
        # components (the stored/physical layout).
        labels = [
            "u_z",
            "u_r",
            "u_theta",
            "c_zz",
            "c_rz",
            "c_theta_z",
            "c_rr",
            "c_theta_theta",
            "c_r_theta",
        ]
    else:  # cylindrical / annular velocity basis
        labels = ["u_z", "u_r", "u_theta"]
    if len(labels) != n_components:  # pragma: no cover - defensive
        labels = [f"c{i}" for i in range(n_components)]
    return labels


def build_mode_extractor(
    mode_pairs: list[tuple[int, int]],
) -> Callable[[Array], Array]:
    r"""Build a jitted gather of the probed modes' columns.

    Returns a function ``state -> (K, C, N_y)`` complex, replicated
    across devices.  ``state`` has the wall-bounded spectral layout
    ``(C, N_y, N_{k_2}, N_{k_3})`` with axes 2/3 sharded by
    ``np0``/``np1``; per probed mode the owning device is computed
    from the local shard shape at trace time (the indices are static)
    and contributes the column to a ``psum`` over both mesh axes.

    *state* arrives in the **solver** basis (the stream samples the
    live field), and each column is converted to physical components
    -- the basis of ``_component_labels`` and of the written stream --
    *after* the gather, on a ``(C, N_y)`` slice rather than the whole
    spectral field, so the probe stream stays affordable even at
    ``it_probes = 1``.  (The map is linear and maps zero to zero, so
    it commutes with the owner mask and the ``psum``.)
    """
    pairs = tuple((int(i2), int(i3)) for i2, i3 in mode_pairs)
    if params.phys.system in cartesian_systems:
        to_physical = None
    elif params.phys.system in viscoelastic_systems:
        from .geometries.wall_bounded.annular_viscoelastic import (
            from_solver_basis as _from_solver,
        )

        to_physical = _from_solver
    else:  # cylindrical / annular velocity basis
        from .geometries.wall_bounded._base import from_pm_basis

        to_physical = from_pm_basis

    def _local(shard: Array) -> Array:
        n2_loc, n3_loc = shard.shape[2], shard.shape[3]
        cols = []
        for i2, i3 in pairs:
            owner0, l2 = divmod(i2, n2_loc)
            owner1, l3 = divmod(i3, n3_loc)
            col = shard[:, :, l2, l3]
            is_owner = (lax.axis_index("np0") == owner0) & (
                lax.axis_index("np1") == owner1
            )
            col = lax.psum(
                jnp.where(is_owner, col, jnp.zeros_like(col)),
                ("np0", "np1"),
            )
            cols.append(col if to_physical is None else to_physical(col))
        return jnp.stack(cols)

    return jax.jit(
        shard_map(
            _local,
            mesh=sharding.mesh,
            in_specs=sharding.spec_vector_shard,
            out_specs=P(None, None, None),
        )
    )


class ProbeStream:
    r"""Buffered binary writer for the spectral-mode probe stream.

    Mirrors the ``.dat`` streams' state machine (on-device buffer +
    host timestamps + index, all ranks reset in lockstep, disk I/O on
    the main process only).  Construct once with the initial state
    (shape source), then :meth:`record` each sample and let
    :meth:`flush` run at the ``flush_all_buffers`` sites.  Both return
    a non-finite diagnostic message (main process only) exactly like
    ``_flush_stats``; the caller aborts on it.
    """

    def __init__(self, state: Array, directory: str | Path = ".") -> None:
        self.modes = parse_mode_pairs(probes_params.modes)
        self.nbuffer: int = params.outs.nbuffer
        n_components, ny = int(state.shape[0]), int(state.shape[1])
        self.component_labels = _component_labels(n_components)
        self._extract = build_mode_extractor(self.modes)
        self._buffer = jnp.zeros(
            (self.nbuffer, len(self.modes), n_components, ny),
            dtype=sharding.complex_type,
        )
        self._ts: list[float] = []
        self._idx: int = 0

        value_dtype = "<f8" if params.res.double_precision else "<f4"
        self.record_dtype = np.dtype(
            [
                ("t", "<f8"),
                ("u", value_dtype, (len(self.modes), n_components, ny, 2)),
            ]
        )
        q2 = complex_harmonics(params.res.nz)
        q3 = real_harmonics(params.res.nx)
        self._sidecar = {
            "format_version": FORMAT_VERSION,
            "modes": [[i2, i3] for i2, i3 in self.modes],
            "wavenumbers": [
                [int(q2[i2]), int(q3[i3])] for i2, i3 in self.modes
            ],
            "n_components": n_components,
            "component_labels": self.component_labels,
            "ny": ny,
            "wall_normal_grid": derived_params.wall_normal_grid,
            "value_dtype": value_dtype,
            "it_probes": probes_params.it_probes,
            "dt": params.step.dt,
            "system": params.phys.system,
            "double_precision": params.res.double_precision,
            "git_hash": git_hash(),
            "params": recorded_params_dump(params),
        }
        self.bin_path = Path(directory) / "probes.bin"
        self.json_path = Path(directory) / "probes.json"
        self._open_files()

    def _open_files(self) -> None:
        """Validate/append or create the ``probes.bin``/``.json`` pair.

        Validation runs identically on every rank (shared filesystem,
        as for snapshots) so a mismatch exits all ranks; only the main
        process writes the sidecar.
        """
        if self.bin_path.exists() and not self.json_path.exists():
            raise SystemExit(
                f"[probes] {self.bin_path} exists without its "
                f"{self.json_path.name} sidecar; move it away."
            )
        if self.json_path.exists():
            with open(self.json_path) as f:
                old = json.load(f)
            mismatch = [
                k for k in _MATCH_KEYS if old.get(k) != self._sidecar[k]
            ]
            if mismatch:
                raise SystemExit(
                    "[probes] existing probes.json does not match this "
                    f"run (differs in: {', '.join(mismatch)}); move the "
                    "old probes.bin/probes.json pair away to start a "
                    "fresh stream."
                )
            n_bytes = (
                self.bin_path.stat().st_size if self.bin_path.exists() else 0
            )
            if n_bytes % self.record_dtype.itemsize != 0:
                raise SystemExit(
                    f"[probes] {self.bin_path} size ({n_bytes} B) is not "
                    f"a whole number of {self.record_dtype.itemsize}-B "
                    "records; the file is corrupt or from another "
                    "configuration."
                )
            sharding.print(
                f"[probes] appending to {self.bin_path} "
                f"({n_bytes // self.record_dtype.itemsize} records)."
            )
        elif sharding.main_device:
            with open(self.json_path, "w") as f:
                json.dump(self._sidecar, f, indent=2, default=str)

    def record(self, state: Array, t: float) -> str | None:
        """Buffer one sample; flush (checked) when the buffer fills."""
        self._buffer = self._buffer.at[self._idx].set(self._extract(state))
        self._ts.append(t)
        self._idx += 1
        if self._idx == self.nbuffer:
            return self.flush()
        return None

    def flush(self, check: bool = True) -> str | None:
        """Append the buffered records to ``probes.bin``, durably.

        Disk I/O (and the non-finite scan, after the write so the
        offending records are on disk for post-mortem) runs on the
        main process; the buffer index / timestamp reset runs on all
        ranks so they stay in lockstep.  Returns the non-finite
        diagnostic message, or ``None``.
        """
        if self._idx == 0:
            return None
        bad = None
        if sharding.main_device:
            data = np.asarray(self._buffer[: self._idx])
            rec = np.zeros(self._idx, dtype=self.record_dtype)
            rec["t"] = np.asarray(self._ts)
            rec["u"][..., 0] = data.real
            rec["u"][..., 1] = data.imag
            with open(self.bin_path, "ab") as f:
                f.write(rec.tobytes())
                f.flush()
                os.fsync(f.fileno())
            if check:
                finite = np.isfinite(data)
                if not finite.all():
                    i, k, c, _ = (int(v) for v in np.argwhere(~finite)[0])
                    i2, i3 = self.modes[k]
                    bad = (
                        f"non-finite probe value: mode ({i2},{i3}) "
                        f"component {self.component_labels[c]} at "
                        f"t = {self._ts[i]:.6e}"
                    )
        self._ts.clear()
        self._idx = 0
        return bad
