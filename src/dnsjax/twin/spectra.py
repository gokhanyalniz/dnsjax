r"""Twin-run ``(k_z, k_x)`` energy-spectra streams: ``twin_spectra*.bin``.

``twin_spectra.bin`` records the difference-field per-mode energy
spectrum `$E_\Delta(k_z, k_x)$` every ``twin.it_spectra`` steps of a
``dnsjax-twin`` run (:mod:`dnsjax.twin.driver`), and, under
``twin.spectra_ref`` (the default), ``twin_spectra_ref.bin`` the
reference state's own spectrum every ``twin.it_spectra_ref`` steps --
by default the same steps.  The pair's ratio `$E_\Delta / 2E^{(1)}$`
is the scale-by-scale decorrelation measure.  Both come from
:func:`dnsjax.twin.diagnostics.twin_spectra_2d`.  The stream is the
high-Reynolds-number replacement for the scalar `$u_1/u_2$` split:
it resolves *which scales* have decorrelated at each time.

The reference spectrum is a stream of its own so that it can run on
its own cadence.  A member recorded before the split stored it inside
``twin_spectra.bin`` (``includes_ref: true``, the ``e_ref`` field);
such a member is resumed in that combined layout
(:class:`TwinSpectraStream`'s *includes_ref*), and the readers take
the reference from whichever layout a run holds
(:func:`dnsjax.analysis.twin.spectra.read_twin_spectra_ref`).

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

(``includes_ref`` only in the pre-split layout), and
``twin_spectra_ref.bin`` of ``[("t", "<f8"), ("e_ref", VAL, (N2, N3))]``,
with ``N2 = nz - 1`` true complex modes, ``N3 = nx // 2`` true
real-FFT modes (spectral padding never stored), and
``VAL = "<f8"``/``"<f4"`` per ``res.double_precision``.  Each
``.json`` sidecar carries its stream's schema: mode counts, the
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

#: ``twin_spectra_ref.json`` schema version (reader floor:
#: ``analysis.twin.spectra.MIN_REF_FORMAT_VERSION``).
REF_FORMAT_VERSION: int = 1

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

#: The reference stream's: the same, its own cadence in place of the
#: difference stream's and no ``includes_ref``.
_REF_MATCH_KEYS: tuple[str, ...] = (
    "format_version",
    "system",
    "n2",
    "n3",
    "value_dtype",
    "it_spectra_ref",
    "dt",
    "double_precision",
    "lx",
    "lz",
)


def _common_sidecar(twin_values) -> dict:
    """The sidecar keys both ``(k_z, k_x)`` streams share."""
    return {
        "system": params.phys.system,
        "n2": params.res.nz - 1,
        "n3": params.res.nx // 2,
        "kz_harmonics": [int(m) for m in complex_harmonics(params.res.nz)],
        "kx_harmonics": [int(m) for m in real_harmonics(params.res.nx)],
        "lx": params.geo.lx,
        "lz": params.geo.lz,
        "value_dtype": "<f8" if params.res.double_precision else "<f4",
        "dt": params.step.dt,
        "double_precision": params.res.double_precision,
        "twin": {
            "seed": twin_values.seed,
            "e0": twin_values.e0,
            "smoothness": twin_values.smoothness,
            "wall_smoothness": twin_values.wall_smoothness,
            "wall_confinement": twin_values.wall_confinement,
        },
        "git_hash": git_hash(),
        "params": recorded_params_dump(params),
    }


class TwinSpectraStream(BinStream):
    """Buffered binary writer for the difference spectrum stream.

    A :class:`~dnsjax.twin._binstream.BinStream` carrying `$(N_2, N_3)$`
    fields; everything about buffering, the sidecar match, the
    ``fsync``-ed append and the non-finite scan lives in the base
    class.  Construct once, :meth:`record` each ``twin_spectra_2d``
    sample.

    *includes_ref* is the pre-split layout, ``e_ref`` in the same
    record: only a paired resume of a member recorded in it asks for
    it, to keep appending to its file (module docstring).
    """

    def __init__(
        self,
        twin_values,
        directory: str | Path = ".",
        *,
        includes_ref: bool = False,
    ) -> None:
        """*twin_values* is the resolved ``[twin]`` section (the
        driver's ``twin_params`` singleton), passed in rather than
        imported: the ``[twin]`` extension is registered by
        :mod:`dnsjax.twin.driver` alone, and taking the values as an
        argument keeps this writer importable and testable without
        pulling in the driver and its import-time registration."""
        self.includes_ref = includes_ref
        sidecar = _common_sidecar(twin_values) | {
            "format_version": FORMAT_VERSION,
            "includes_ref": includes_ref,
            "it_spectra": twin_values.it_spectra,
            "note": (
                "per-mode energy: k_metric/2 * int |u|^2 w dy / V, "
                "component-summed; true modes only; sum == E_d"
            ),
        }
        self.n2, self.n3 = sidecar["n2"], sidecar["n3"]
        names = ("e_delta", "e_ref") if includes_ref else ("e_delta",)
        directory = Path(directory)
        super().__init__(
            fields=tuple((n, (self.n2, self.n3)) for n in names),
            sidecar=sidecar,
            match_keys=_MATCH_KEYS,
            bin_path=directory / "twin_spectra.bin",
            json_path=directory / "twin_spectra.json",
            value_dtype=sidecar["value_dtype"],
            nbuffer=_NBUFFER,
        )


class TwinSpectraRefStream(BinStream):
    """Buffered binary writer for the reference spectrum stream.

    ``twin_spectra_ref.bin``: the reference state's ``e_ref`` alone,
    every *cadence* steps (the resolved ``twin.it_spectra_ref``).
    """

    def __init__(
        self, twin_values, cadence: int, directory: str | Path = "."
    ) -> None:
        sidecar = _common_sidecar(twin_values) | {
            "format_version": REF_FORMAT_VERSION,
            "it_spectra_ref": cadence,
            "note": (
                "the reference state's per-mode energy: k_metric/2 * "
                "int |u|^2 w dy / V, component-summed; true modes only"
            ),
        }
        n2, n3 = sidecar["n2"], sidecar["n3"]
        directory = Path(directory)
        super().__init__(
            fields=(("e_ref", (n2, n3)),),
            sidecar=sidecar,
            match_keys=_REF_MATCH_KEYS,
            bin_path=directory / "twin_spectra_ref.bin",
            json_path=directory / "twin_spectra_ref.json",
            value_dtype=sidecar["value_dtype"],
            nbuffer=_NBUFFER,
        )
