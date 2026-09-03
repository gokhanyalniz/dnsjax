r"""Twin-run ``(k_z, k_x)`` energy-spectra stream: ``twin_spectra.bin``.

Records the difference-field per-mode energy spectrum
`$E_\Delta(k_z, k_x)$` (and, by default, the reference state's own
spectrum -- the pair whose ratio `$E_\Delta / 2E^{(1)}$` is the
scale-by-scale decorrelation measure) every ``twin.it_spectra`` steps
of a ``dnsjax-twin`` run (:mod:`dnsjax.twin.driver`), computed by
:func:`dnsjax.twin.diagnostics.twin_spectra_2d`.  The stream is the
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

Buffering, resume-by-append, sidecar matching (:data:`_MATCH_KEYS`)
and the post-write non-finite scan are
:class:`dnsjax.twin._binstream.BinStream`'s, shared with the
wall-normal-resolved streams; only :data:`FORMAT_VERSION`, the field
table and the sidecar are this stream's own.  The buffer depth
(:data:`_NBUFFER`) is fixed and small: a record is `$\sim$`MB at
production sizes, so the ``outs.nbuffer`` default would hold
hundreds of MB on device.
"""

from pathlib import Path

from ..harmonics import complex_harmonics, real_harmonics
from ..param_surface import recorded_params_dump
from ..parameters import params
from ..snapshot_meta import git_hash
from ._binstream import BinStream

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


class TwinSpectraStream(BinStream):
    """Buffered binary writer for the twin spectra stream.

    A :class:`~dnsjax.twin._binstream.BinStream` carrying two
    equal-shaped `$(N_2, N_3)$` fields; everything about buffering,
    the sidecar match, the ``fsync``-ed append and the non-finite
    scan lives in the base class.  Construct once, :meth:`record`
    each ``twin_spectra_2d`` sample.
    """

    def __init__(self, twin_values, directory: str | Path = ".") -> None:
        """*twin_values* is the resolved ``[twin]`` section (the
        driver's ``twin_params`` singleton), passed in rather than
        imported: the ``[twin]`` extension is registered by
        :mod:`dnsjax.twin.driver` alone, and taking the values as an
        argument keeps this writer importable and testable without
        pulling in the driver and its import-time registration."""
        self.includes_ref = bool(twin_values.spectra_ref)
        self.n2 = params.res.nz - 1
        self.n3 = params.res.nx // 2
        names = ("e_delta", "e_ref") if self.includes_ref else ("e_delta",)
        value_dtype = "<f8" if params.res.double_precision else "<f4"
        directory = Path(directory)
        super().__init__(
            fields=tuple((n, (self.n2, self.n3)) for n in names),
            sidecar={
                "format_version": FORMAT_VERSION,
                "system": params.phys.system,
                "n2": self.n2,
                "n3": self.n3,
                "kz_harmonics": [
                    int(m) for m in complex_harmonics(params.res.nz)
                ],
                "kx_harmonics": [
                    int(m) for m in real_harmonics(params.res.nx)
                ],
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
                    "wall_smoothness": twin_values.wall_smoothness,
                    "wall_confinement": twin_values.wall_confinement,
                },
                "git_hash": git_hash(),
                "params": recorded_params_dump(params),
            },
            match_keys=_MATCH_KEYS,
            bin_path=directory / "twin_spectra.bin",
            json_path=directory / "twin_spectra.json",
            value_dtype=value_dtype,
            nbuffer=_NBUFFER,
        )
