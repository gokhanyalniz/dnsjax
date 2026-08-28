r"""Shared buffered-binary-stream machinery for the twin writers.

The three ``dnsjax-twin`` binary streams -- ``twin_spectra.bin``
(:mod:`dnsjax.twin.spectra`), ``twin_yspectra.bin`` and
``twin_ybudget.bin`` (:mod:`dnsjax.twin.yspectra`) -- differ only in
*what* they record.  Everything around that is one state machine,
inherited from ``extensions.probes.ProbeStream``: an on-device buffer
of a few records, host-side timestamps, disk I/O and the sidecar on
the main process, index resets on **all** ranks, an ``fsync``-ed
append, and a post-write non-finite scan whose message the driver
aborts on.  It lives here rather than being written three times.

What a subclass supplies
========================
Its own :data:`FORMAT_VERSION`, ``_MATCH_KEYS`` and sidecar dict --
these are per-stream on purpose, since a stream's stored *meaning*
changes independently of its siblings' -- plus the field table
``((name, shape), ...)`` and the two paths.  Nothing else.

Why the buffer is flat
======================
Fields of one stream need not share a shape: ``twin_yspectra`` mixes
`$(3, N_y, n_{k_z})$` and `$(3, N_y, n_{k_x})$` blocks.  The device
buffer is therefore ``(nbuffer, total_flat)`` with per-field offsets,
and the structured record dtype restores the real shapes at write
time -- so the **on-disk layout is unchanged** by this indirection
(a stream whose fields do share a shape writes exactly the bytes it
wrote before).
"""

import json
import os
from pathlib import Path

import numpy as np
from jax import Array
from jax import numpy as jnp

from ..sharding import sharding
from ..snapshot_meta import write_sidecar_json


class BinStream:
    """Buffered binary record stream with a JSON sidecar.

    Construct once, :meth:`record` each sample, and let :meth:`flush`
    run at the driver's ``flush_all_buffers`` sites; both return the
    non-finite diagnostic message the caller aborts on (``None`` when
    clean).
    """

    def __init__(
        self,
        *,
        fields: tuple[tuple[str, tuple[int, ...]], ...],
        sidecar: dict,
        match_keys: tuple[str, ...],
        bin_path: Path,
        json_path: Path,
        value_dtype: str,
        nbuffer: int,
        float_type=None,
    ) -> None:
        self._fields = fields
        self._match_keys = match_keys
        self._sidecar = sidecar
        self.bin_path = bin_path
        self.json_path = json_path
        self.record_dtype = np.dtype(
            [("t", "<f8")]
            + [(name, value_dtype, shape) for name, shape in fields]
        )
        sizes = [int(np.prod(shape)) for _, shape in fields]
        self._offsets = np.cumsum([0] + sizes)
        self._buffer = jnp.zeros(
            (nbuffer, int(self._offsets[-1])),
            dtype=float_type or sharding.float_type,
        )
        self._nbuffer = nbuffer
        self._ts: list[float] = []
        self._idx: int = 0
        self._open_files()

    def _open_files(self) -> None:
        """Validate/append or create the ``.bin``/``.json`` pair.

        Validation runs identically on every rank (shared
        filesystem); only the main process writes the sidecar.
        """
        if self.bin_path.exists() and not self.json_path.exists():
            raise SystemExit(
                f"[twin] {self.bin_path} exists without its "
                f"{self.json_path.name} sidecar; move it away."
            )
        if self.json_path.exists():
            with open(self.json_path) as f:
                old = json.load(f)
            mismatch = [
                k for k in self._match_keys if old.get(k) != self._sidecar[k]
            ]
            if mismatch:
                raise SystemExit(
                    f"[twin] existing {self.json_path.name} does not "
                    f"match this run (differs in: {', '.join(mismatch)}); "
                    f"move the old {self.bin_path.name}/"
                    f"{self.json_path.name} pair away to start a fresh "
                    "stream."
                )
            n_bytes = (
                self.bin_path.stat().st_size if self.bin_path.exists() else 0
            )
            if n_bytes % self.record_dtype.itemsize != 0:
                raise SystemExit(
                    f"[twin] {self.bin_path} size ({n_bytes} B) is not "
                    f"a whole number of {self.record_dtype.itemsize}-B "
                    "records; the file is corrupt or from another "
                    "configuration."
                )
            sharding.print(
                f"[twin] appending to {self.bin_path} "
                f"({n_bytes // self.record_dtype.itemsize} records)."
            )
        elif sharding.main_device:
            # Atomic: every rank tested ``json_path.exists()`` above,
            # so a path that exists must already be complete
            # (:func:`~dnsjax.snapshot_meta.write_sidecar_json`).
            write_sidecar_json(self.json_path, self._sidecar)

    def record(self, values: dict[str, Array], t: float) -> str | None:
        """Buffer one sample; flush when the buffer fills."""
        flat = jnp.concatenate(
            [values[name].reshape(-1) for name, _ in self._fields]
        )
        self._buffer = self._buffer.at[self._idx].set(flat)
        self._ts.append(t)
        self._idx += 1
        if self._idx == self._nbuffer:
            return self.flush()
        return None

    def flush(self, check: bool = True) -> str | None:
        """Append the buffered records durably; reset on all ranks."""
        if self._idx == 0:
            return None
        bad = None
        if sharding.main_device:
            data = np.asarray(self._buffer[: self._idx])
            rec = np.zeros(self._idx, dtype=self.record_dtype)
            rec["t"] = np.asarray(self._ts)
            for i, (name, shape) in enumerate(self._fields):
                block = data[:, self._offsets[i] : self._offsets[i + 1]]
                rec[name] = block.reshape((self._idx, *shape))
            with open(self.bin_path, "ab") as f:
                f.write(rec.tobytes())
                f.flush()
                os.fsync(f.fileno())
            if check:
                bad = self._non_finite(data)
        self._ts.clear()
        self._idx = 0
        return bad

    def _non_finite(self, data: np.ndarray) -> str | None:
        """Name the first non-finite value, mapping the flat offset back."""
        finite = np.isfinite(data)
        if finite.all():
            return None
        row, col = (int(v) for v in np.argwhere(~finite)[0])
        which = int(np.searchsorted(self._offsets, col, side="right") - 1)
        return (
            f"non-finite {self.bin_path.stem} value in "
            f"{self._fields[which][0]} at t = {self._ts[row]:.6e}"
        )
