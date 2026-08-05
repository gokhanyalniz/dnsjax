r"""Twin-run ``(k_z, k_x)`` energy-spectra stream: ``twin_spectra.bin``.

Records the difference-field per-mode energy spectrum
`$E_\Delta(k_z, k_x)$` (and, by default, the reference state's own
spectrum -- the pair whose ratio `$E_\Delta / 2E^{(1)}$` is the
scale-by-scale decorrelation measure) every ``twin.it_spectra`` steps
of a ``dnsjax-twin`` run (:mod:`dnsjax.twin`), computed by
:func:`dnsjax.twin_diagnostics.twin_spectra_2d`.  The stream is the
high-Reynolds-number replacement for the scalar `$u_1/u_2$` split:
it resolves *which scales* have decorrelated at each time.

Why a stream: the spectrum is `$O(N_{k_z} N_{k_x})$` values per
sample -- far beyond the scalar ``.dat`` streams, three orders of
magnitude below a snapshot.  FFT-free (a masked reduction of data
already in spectral space), so any cadence is cheap.

File format
===========
``twin_spectra.bin`` is a flat sequence of fixed-size records,

.. code-block:: python

    numpy.dtype(
        [("t", "<f8"), ("e_delta", VAL, (N2, N3))]
        + ([("e_ref", VAL, (N2, N3))] if includes_ref else [])
    )

with ``N2 = nz - 1`` true complex modes, ``N3 = nx // 2`` true
real-FFT modes (spectral padding never stored), and
``VAL = "<f8"``/``"<f4"`` per ``res.double_precision``.  The
``twin_spectra.json`` sidecar carries the schema: mode counts, the
integer harmonic lists of both axes (physical wavenumbers =
`$\times\, 2\pi/L$`; :mod:`dnsjax.harmonics`), the domain lengths,
cadence, and the resolved parameter dump.  The JAX-free reader is
:mod:`dnsjax.analysis.twin.spectra`.

:data:`FORMAT_VERSION` follows the probes discipline: the record
layout reads cleanly across schema changes, so bump the version
whenever the stored *meaning* changes and raise the reader's floor
with it.

Resume semantics mirror ``probes.bin``: an existing pair is appended
to iff the sidecar matches (:data:`_MATCH_KEYS`), anything else is a
hard error; a clean continuation duplicates one sample per seam
(dropped by the reader).  Buffering mirrors the ``.dat`` streams
(main-process ``fsync``-ed writes, all-rank index resets, non-finite
scan after the write) at a fixed small depth
(:data:`_NBUFFER`; a record is `$\sim$`MB at production sizes, so
the ``outs.nbuffer`` default would hold hundreds of MB on device).
"""

import json
import os
from pathlib import Path

import numpy as np
from jax import Array
from jax import numpy as jnp

from .harmonics import complex_harmonics, real_harmonics
from .param_surface import recorded_params_dump
from .parameters import params
from .sharding import sharding
from .snapshot_meta import git_hash

#: Sidecar schema version (bump when the stored meaning changes; the
#: reader's floor is ``analysis.twin.spectra.MIN_FORMAT_VERSION``).
FORMAT_VERSION: int = 1

#: Records buffered on device between flushes (deliberately small and
#: fixed: a production-size record is ~1-2 MB, so ``outs.nbuffer``
#: would pin an outsized replicated buffer).
_NBUFFER: int = 8

#: Sidecar keys that must match for an append (resume) to proceed.
_MATCH_KEYS: tuple[str, ...] = (
    "format_version",
    "system",
    "n2",
    "n3",
    "value_dtype",
    "includes_ref",
    "it_spectra",
    "dt",
    "double_precision",
    "lx",
    "lz",
)


class TwinSpectraStream:
    """Buffered binary writer for the twin spectra stream.

    The ``probes.ProbeStream`` state machine: an on-device
    ``(_NBUFFER, n_fields, N2, N3)`` buffer plus host timestamps,
    disk I/O on the main process, index resets on all ranks.
    Construct once, :meth:`record` each ``twin_spectra_2d`` sample,
    and let :meth:`flush` run at the driver's ``flush_all_buffers``
    sites; both return the non-finite diagnostic message the caller
    aborts on.
    """

    def __init__(self, twin_values, directory: str | Path = ".") -> None:
        """*twin_values* is the resolved ``[twin]`` section (the
        driver's ``twin_params`` singleton), passed in rather than
        imported: under ``python -m dnsjax.twin`` the driver module
        is ``__main__``, and a ``from .twin import ...`` here would
        re-execute it as a second module instance whose extension
        re-registration fails."""
        self.includes_ref = bool(twin_values.spectra_ref)
        self.n2 = params.res.nz - 1
        self.n3 = params.res.nx // 2
        self._fields = (
            ("e_delta", "e_ref") if self.includes_ref else ("e_delta",)
        )
        self._buffer = jnp.zeros(
            (_NBUFFER, len(self._fields), self.n2, self.n3),
            dtype=sharding.float_type,
        )
        self._ts: list[float] = []
        self._idx: int = 0

        value_dtype = "<f8" if params.res.double_precision else "<f4"
        shape = (self.n2, self.n3)
        self.record_dtype = np.dtype(
            [("t", "<f8")]
            + [(name, value_dtype, shape) for name in self._fields]
        )
        self._sidecar = {
            "format_version": FORMAT_VERSION,
            "system": params.phys.system,
            "n2": self.n2,
            "n3": self.n3,
            "kz_harmonics": [int(m) for m in complex_harmonics(params.res.nz)],
            "kx_harmonics": [int(m) for m in real_harmonics(params.res.nx)],
            "lx": params.geo.lx,
            "lz": params.geo.lz,
            "value_dtype": value_dtype,
            "includes_ref": self.includes_ref,
            "it_spectra": twin_values.it_spectra,
            "dt": params.step.dt,
            "double_precision": params.res.double_precision,
            "note": (
                "per-mode energy: k_metric/2 * int |u|^2 w dy / V, "
                "component-summed; true modes only; sum == E_d"
            ),
            "twin": {
                "seed": twin_values.seed,
                "e0": twin_values.e0,
                "smoothness": twin_values.smoothness,
            },
            "git_hash": git_hash(),
            "params": recorded_params_dump(params),
        }
        self.bin_path = Path(directory) / "twin_spectra.bin"
        self.json_path = Path(directory) / "twin_spectra.json"
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
                k for k in _MATCH_KEYS if old.get(k) != self._sidecar[k]
            ]
            if mismatch:
                raise SystemExit(
                    "[twin] existing twin_spectra.json does not match "
                    f"this run (differs in: {', '.join(mismatch)}); "
                    "move the old twin_spectra.bin/.json pair away to "
                    "start a fresh stream."
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
            with open(self.json_path, "w") as f:
                json.dump(self._sidecar, f, indent=2, default=str)

    def record(self, spectra: dict[str, Array], t: float) -> str | None:
        """Buffer one ``twin_spectra_2d`` sample; flush when full."""
        sample = jnp.stack([spectra[name] for name in self._fields])
        self._buffer = self._buffer.at[self._idx].set(sample)
        self._ts.append(t)
        self._idx += 1
        if self._idx == _NBUFFER:
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
            for i, name in enumerate(self._fields):
                rec[name] = data[:, i]
            with open(self.bin_path, "ab") as f:
                f.write(rec.tobytes())
                f.flush()
                os.fsync(f.fileno())
            if check:
                finite = np.isfinite(data)
                if not finite.all():
                    i, j, *_ = (int(v) for v in np.argwhere(~finite)[0])
                    bad = (
                        f"non-finite spectra value in "
                        f"{self._fields[j]} at t = {self._ts[i]:.6e}"
                    )
        self._ts.clear()
        self._idx = 0
        return bad
