r"""Spectral-mode probe stream: per-step `$\hat{u}(y)$` time series.

Records the complex wall-normal profiles of a small set of global
spectral modes -- ``probes.modes`` (the ``probes`` extension section;
:mod:`dnsjax.extensions`), an ``"i2,i3;..."`` list of stored-layout
indices (axis 2 = complex slot, axis 3 = real-FFT slot: the
transient-growth CLI's ``--tg.modes`` convention;
:func:`dnsjax.harmonics.parse_mode_pairs`) -- every
``probes.it_probes`` steps into a binary ``probes.bin`` next to the
``.dat`` diagnostic streams.  Wall-bounded systems only (the state
layout is ``(C, N_y, N_{k_2}, N_{k_3})``); on ``curved-pipe`` the
component-0 column is the carried `$w_s = h\,u_s$`, not `$u_s$`
(:class:`dnsjax_lab.extensions.ProbesParams`).  Unlike the
transient-growth CLI, the mean mode ``(0,0)`` **is** allowed: its
record is the instantaneous mean profile of the stored field -- the
perturbation on a base-flow system, whose closed-form laminar profile
gives the total (the reader's ``mean_profile`` adds it for plane
Couette and plane Poiseuille), and the total itself on a force-driven
one.

Why a fourth stream: a mode time series at `$O(10^5)$` to `$O(10^6)$` samples
is far beyond what per-snapshot output can carry (a snapshot per sample
is ~three orders of magnitude more bytes), while the scalar ``.dat``
text streams cannot hold complex per-`$y$` profiles.  The probe stream
is the input for mode statistics (covariances, spectra) and for
ensemble-averaged response curves.

File format
-----------
``probes.bin`` is a flat sequence of fixed-size records,

.. code-block:: python

    numpy.dtype([("t", "<f8"), ("u", VAL, (K, C, N_y, 2))])

with ``K`` probed modes, ``C`` state components, the trailing axis
``(re, im)``, and ``VAL = "<f8"``/``"<f4"`` following
``res.double_precision`` (the values carry state precision; ``t`` is
always float64).  A ``probes.json`` sidecar (written once) carries the
schema: the mode list, integer wavenumbers, component labels, the
wall-normal grid, cadence, and the full resolved parameter dump.  The
JAX-free reader is :mod:`dnsjax_lab.analysis.response.probes`.

The record layout is fixed across schema versions, so a stale stream
reads *cleanly* and only its values mean something else -- which makes
:data:`FORMAT_VERSION` the only thing between an old file and a silent
misread.  Bump it whenever the stored meaning changes (not only the
layout), and raise the reader's ``MIN_FORMAT_VERSION`` with it.

Resume semantics: an existing ``probes.bin``/``probes.json`` pair is
appended to iff the sidecar matches the current run (same modes **and
the wavenumbers they denote**, grid, components, precision, system,
cadence, and ``dt``: :data:`_MATCH_KEYS`) -- anything else is a hard
error asking the user to move the old pair away.  The wavenumbers
matter because a resume may change ``res.nx`` / ``res.nz`` (the Fourier
axes are regridded at load, ``snapshot._from_io_layout_core``), and on
the wrap-ordered `$k_z$` axis a stored *index* in the negative block
then means a different mode; matching the wavenumbers refuses exactly
that, and still appends across a resolution change that leaves every
probed mode where it was.  A clean continuation duplicates one sample
per seam (the parent's cadence-aligned final record and the child's t0
record hold the same state at the same ``t``); the reader drops these
and flags genuinely non-monotonic timestamps (a re-run trajectory
segment).

Sharded gather
--------------
The state's mode axes are sharded (`$k_2$` by ``np0``, `$k_3$` by
``np1``), and slicing a sharded axis of the global array is not
supported under explicit sharding; :func:`build_mode_extractor`
instead gathers inside a ``shard_map`` -- the owning device is
computed at trace time from the *local* shard shape (the indices are
static), and every device contributes either the column or zeros to a
``psum`` over both mesh axes (the ``extract_mean_mode`` pattern of
:mod:`dnsjax.geometries.wall_bounded._base`, generalized to arbitrary
static mode indices).  Probed indices always address *true* modes
(``validate_parameters`` bounds them by the unpadded mode counts), so
spectral padding slots are never read.

Buffering matches the ``.dat`` streams: an on-device
``(nbuffer, K, C, N_y)`` complex buffer plus a host timestamp list and
index, reset on all ranks in lockstep, and flushed (appended +
``fsync``-ed) by the main process when full and at every
``flush_all_buffers`` site of :mod:`dnsjax.__main__` -- after the first
step, before every snapshot, at shutdown and on a termination signal;
flushed records are scanned for non-finite values and a hit aborts the
run through the same FATAL / exit-3 path as the other diagnostics.
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

from dnsjax.extensions import RunHooks
from dnsjax.flows.registry import cartesian_systems, spec_for
from dnsjax.harmonics import (
    complex_harmonics,
    parse_mode_pairs,
    real_harmonics,
)
from dnsjax.param_surface import recorded_params_dump
from dnsjax.parameters import derived_params, params
from dnsjax.sharding import sharding
from dnsjax.snapshot_meta import git_hash, write_sidecar_json
from dnsjax_lab.extensions import probes_params
from dnsjax_lab.flows import VISCOELASTIC_SYSTEMS

#: Sidecar schema version.  Bumped whenever the stored *meaning*
#: changes at fixed layout -- the physical component basis of the
#: columns, the physical (`$m = m_0 j$`) azimuthal wavenumber labels
#: -- so a stale stream cannot be silently misread by a positional
#: consumer (``response.lim`` / ``response.ssi``).  The reader's
#: floor is ``analysis.response.probes.MIN_FORMAT_VERSION``.
FORMAT_VERSION: int = 3

#: Sidecar keys that must match for an append (resume) to proceed.
_MATCH_KEYS: tuple[str, ...] = (
    "format_version",
    "modes",
    # Not the indices alone: a resume may regrid the Fourier axes (the
    # module docstring's resume semantics).
    "wavenumbers",
    "n_components",
    # The *counts* above are not enough: a resume at unchanged ``ny``
    # onto a different ``geo.grid_type`` samples the same number of
    # rows at different `$y$`, and the sidecar (written once, never
    # rewritten) would keep advertising the original grid, so any
    # profile or wall-derived quantity would integrate on the wrong
    # one.  Same for the component *labels* versus their count.
    "component_labels",
    "wall_normal_grid",
    "ny",
    "value_dtype",
    "system",
    "it_probes",
    "dt",
)


def _component_labels(n_components: int) -> list[str]:
    r"""Component labels of the stored state for the current system.

    ``curved-pipe`` takes the cylindrical schema, although its slot 0
    holds the carried `$w_s = h\,u_s$` (``ProbesParams``).
    """
    if params.phys.system in cartesian_systems:
        labels = ["u_x", "u_y", "u_z"]
    elif params.phys.system in VISCOELASTIC_SYSTEMS:
        # Velocity + the 6 stored physical conformation-tensor
        # components (the stored/physical layout).
        labels = list(spec_for(params.phys.system).components)
    else:  # cylindrical / annular velocity basis
        labels = ["u_z", "u_r", "u_theta"]
    if len(labels) != n_components:
        raise ValueError(
            f"no component-label schema for {n_components} components "
            f"({params.phys.system!r})"
        )
    return labels


def build_mode_extractor(
    mode_pairs: list[tuple[int, int]],
    n_components: int | None = None,
) -> Callable[[Array], Array]:
    r"""Build a jitted gather of the probed modes' columns.

    Returns a function ``state -> (K, C, N_y)`` complex, replicated
    across devices, for ``state`` in the wall-bounded spectral layout
    ``(C, N_y, N_{k_2}, N_{k_3})`` with axes 2/3 sharded by
    ``np0``/``np1`` (the gather: the module docstring's "Sharded
    gather").

    Only the first *n_components* slots are gathered (every slot when
    ``None``) -- the flow's physical ones; a pipe state's trailing
    carried slots are solver internals (``_cylindrical_stepping``).
    Where the state is carried in a **solver** basis (cylindrical /
    annular `$u_\pm$`, the viscoelastic spin tensor) it is converted to
    physical components -- the basis of ``_component_labels`` and of
    the written stream -- *after* the gather, on a ``(C, N_y)`` slice
    rather than the whole spectral field, so a sample costs little and
    ``it_probes = 1`` is affordable (the map is linear and maps zero to
    zero, so it commutes with the owner mask and the ``psum``).
    Cartesian carries physical components under both
    ``res.consistent_imm`` formulations and needs no conversion.
    """
    pairs = tuple((int(i2), int(i3)) for i2, i3 in mode_pairs)
    if params.phys.system in cartesian_systems:
        to_physical = None
    elif params.phys.system in VISCOELASTIC_SYSTEMS:
        # The 9-component spin map, from the geometry-free module both
        # viscoelastic geometries share -- importing either geometry
        # here would build the wrong family's singletons.
        from dnsjax_lab.geometries.wall_bounded._viscoelastic_common import (
            from_spin_basis,
        )

        to_physical = from_spin_basis
    else:  # cylindrical / annular velocity basis
        from dnsjax.geometries.wall_bounded._base import from_pm_basis

        to_physical = from_pm_basis

    def _local(shard: Array) -> Array:
        n2_loc, n3_loc = shard.shape[2], shard.shape[3]
        cols = []
        for i2, i3 in pairs:
            owner0, l2 = divmod(i2, n2_loc)
            owner1, l3 = divmod(i3, n3_loc)
            col = shard[:n_components, :, l2, l3]
            is_owner = (lax.axis_index("np0") == owner0) & (
                lax.axis_index("np1") == owner1
            )
            col = lax.psum(
                jnp.where(is_owner, col, jnp.zeros_like(col)),
                ("np0", "np1"),
            )
            cols.append(col if to_physical is None else to_physical(col))
        return jnp.stack(cols)

    extractor = shard_map(
        _local,
        mesh=sharding.mesh,
        in_specs=sharding.spec_vector_shard,
        out_specs=P(None, None, None),
    )
    return jax.jit(extractor)


class ProbeStream:
    r"""Buffered binary writer for the spectral-mode probe stream.

    Mirrors the ``.dat`` streams' state machine (the module docstring's
    buffering paragraph).  Construct once with the initial state (shape
    source), then :meth:`record` each sample and let :meth:`flush` run
    at the ``flush_all_buffers`` sites.  Both return a non-finite
    diagnostic message (main process only) exactly like
    ``_flush_stats``; the caller aborts on it.
    """

    def __init__(self, state: Array, directory: str | Path = ".") -> None:
        self.modes = parse_mode_pairs(probes_params.modes)
        self.nbuffer: int = params.outs.nbuffer
        # The flow's physical component count, not the state's leading
        # size: a pipe state also carries solver slots.
        n_components = spec_for(params.phys.system).n_components
        ny = int(state.shape[1])
        self.component_labels = _component_labels(n_components)
        self._extract = build_mode_extractor(self.modes, n_components)
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
        # Axis-2 harmonics are *physical* wavenumbers: on the
        # cylindrical/annular wedge the azimuthal mode of harmonic
        # `$j$` is `$m = m_0 j$` (``geo.m0 = 1`` is the full circle),
        # exactly as the geometry's ``Fourier.m`` builds it.  Cartesian
        # axis 2 is `$k_z$`, which has no wedge factor.
        m0 = 1 if params.phys.system in cartesian_systems else params.geo.m0
        q2 = complex_harmonics(params.res.nz) * m0
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

        Appends only against a sidecar matching on :data:`_MATCH_KEYS`.
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
            # Atomic, for the same reason as every other sidecar
            # (:func:`~dnsjax.snapshot_meta.write_sidecar_json`).
            write_sidecar_json(self.json_path, self._sidecar)

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


class ProbeHooks(RunHooks):
    r"""The ``[probes]`` section's part in the solver's run loop.

    The section's ``runtime`` factory
    (:class:`~dnsjax.extensions.RunHooks`, which times the hooks).
    Records the initial state (``start``), every ``it_probes`` steps
    the pre-step state (``sample``, which skips ``it == it0``, the start
    record's), and the post-step final state when the run ends on the
    cadence (``finish``): only a cadence-aligned final record (a
    ``stop.max_sim_time`` horizon of a whole number of probe intervals
    ends exactly on the grid) keeps the sample times uniform for the
    readers.  A non-finite record aborts the run, through the hook that
    records it when it fills the buffer (``outs.nbuffer``), else at the
    next checked :meth:`flush`.
    """

    def __init__(self, state: Array, t: float) -> None:
        self.stream = ProbeStream(state)

    def start(self, state: Array, t: float) -> str | None:
        return self.stream.record(state, t)

    def sample(self, it: int, state: Array, t: float) -> str | None:
        if it % probes_params.it_probes == 0 and it > params.init.it0:
            return self.stream.record(state, t)
        return None

    def finish(self, it: int, state: Array, t: float) -> str | None:
        if it > params.init.it0 and it % probes_params.it_probes == 0:
            return self.stream.record(state, t)
        return None

    def flush(self, check: bool) -> str | None:
        return self.stream.flush(check=check)
