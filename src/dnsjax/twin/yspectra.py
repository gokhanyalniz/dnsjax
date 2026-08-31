r"""Wall-normal-resolved twin spectra and budget streams.

Two binary streams written by ``dnsjax-twin``
(:mod:`dnsjax.twin.driver`), both on the wall-normal-resolved,
one-sided marginal bins of
:func:`dnsjax.twin.diagnostics.marginal_bin_counts`:

- ``twin_yspectra.bin`` (``twin.it_yspectra``), from
  :func:`~dnsjax.twin.diagnostics.twin_yspectra`: the componentwise
  difference energy `$E_\Delta^x[u,v,w](y, k_z)$` and
  `$E_\Delta^z[u,v,w](y, k_x)$`, plus the `$k_x = 0$` plane that
  recovers the `$\Delta U$` / `$\Delta u_1$` / `$\Delta u_2$`
  binning;
- ``twin_ybudget.bin`` (``twin.it_ybudget``), from
  :func:`~dnsjax.twin.diagnostics.twin_ybudget`: the component-summed
  budget densities of
  :func:`~dnsjax.twin.diagnostics.ybudget_terms` on the same bins --
  seven of them in the default convective form, eight under
  ``twin.rotational_ybudget``.  The sidecar's ``terms`` names them
  and is a match key, so a resume cannot change form mid-stream.

Why these replace the three-bin diagnostics: the `$\Delta U$` /
`$\Delta u_1$` / `$\Delta u_2$` split is a three-bin partition of the
`$(k_x, k_z)$` plane, and its own authors restrict it to minimal flow
units (Egerique-de-la-Concha & Hwang, *J. Fluid Mech.* **1036**, A52,
2026, after their eq. 2.5).  These streams refine the bin index to
the wavenumber itself and drop the `$y$`-integration, and the old
three numbers remain exactly recoverable from them.

File format
===========
Both are flat sequences of fixed-size records, ``("t", "<f8")``
followed by one entry per field, ``VAL = "<f8"``/``"<f4"`` per
``res.double_precision``:

.. code-block:: text

    twin_yspectra:  e_x,  e_z,  e_x0   (3, ny, n_kz) / (3, ny, n_kx)
                    r_x,  r_z,  r_x0   when twin.spectra_ref
    twin_ybudget:   <term>_x, <term>_z, <term>_x0  for each of
                    ybudget_terms(),   (ny, n_kz) / (ny, n_kx)

Every array is a `$y$`-**density** already divided by
``volume_fac``: contract with the sidecar's ``y_weights`` for the
per-`$k$` quantity, and sum that over `$k$` for the corresponding
``twin.dat`` / ``twin_budget.dat`` scalar.  The sidecars therefore
carry ``y`` and ``y_weights`` outright, so a reader integrates
without rebuilding the grid.

Both wavenumber axes are **one-sided**: ``kz_harmonics`` and
``kx_harmonics`` are :func:`dnsjax.harmonics.real_harmonics` of
``nz`` / ``nx``.  The `$k_z$` axis is folded onto `$|k_z|$` for a
reason that is not cosmetic --
:func:`dnsjax.twin.diagnostics._fold_kz` has it.

Buffering, sidecar matching, resume-by-append and the non-finite scan
are :class:`dnsjax.twin._binstream.BinStream`'s; the JAX-free reader
is :mod:`dnsjax.analysis.twin.yspectra`.  Each stream keeps its own
:data:`FORMAT_VERSION` / ``_MATCH_KEYS``, bumped when *its* stored
meaning changes.
"""

from pathlib import Path

from ..harmonics import real_harmonics
from ..param_surface import recorded_params_dump
from ..parameters import derived_params, params
from ..snapshot_meta import git_hash
from ._binstream import BinStream
from .diagnostics import marginal_bin_counts, ybudget_terms

#: Sidecar schema versions (the reader's floors are
#: ``analysis.twin.yspectra.MIN_*_VERSION``).
YSPECTRA_FORMAT_VERSION: int = 1
YBUDGET_FORMAT_VERSION: int = 2

#: Records buffered on device between flushes.  Smaller than the
#: ``(k_z, k_x)`` stream's: these records carry a wall-normal axis
#: (~0.4 MB for the energies, ~1.0 MB for the budget at
#: `$n_y = 129$`, `$n_x = n_z = 192$`).
_NBUFFER: int = 4

_YSPECTRA_MATCH_KEYS: tuple[str, ...] = (
    "format_version",
    "system",
    "ny",
    "n_kz",
    "n_kx",
    "value_dtype",
    "includes_ref",
    "it_yspectra",
    "dt",
    "double_precision",
    "lx",
    "lz",
)

_YBUDGET_MATCH_KEYS: tuple[str, ...] = (
    "format_version",
    "system",
    "ny",
    "n_kz",
    "n_kx",
    "terms",
    "value_dtype",
    "it_ybudget",
    "dt",
    "double_precision",
    "lx",
    "lz",
)


def _common_sidecar(twin_values, y_weights: list[float]) -> dict:
    """The keys both sidecars share (grid, axes, provenance)."""
    n_kz, n_kx = marginal_bin_counts()
    return {
        "system": params.phys.system,
        "ny": params.res.ny,
        "n_kz": n_kz,
        "n_kx": n_kx,
        "kz_harmonics": [int(m) for m in real_harmonics(params.res.nz)],
        "kx_harmonics": [int(m) for m in real_harmonics(params.res.nx)],
        "lx": params.geo.lx,
        "lz": params.geo.lz,
        "y": list(derived_params.wall_normal_grid or []),
        "y_weights": list(y_weights),
        "volume_fac": derived_params.volume_fac,
        "value_dtype": "<f8" if params.res.double_precision else "<f4",
        "dt": params.step.dt,
        "double_precision": params.res.double_precision,
        "twin": {
            "seed": twin_values.seed,
            "e0": twin_values.e0,
            "smoothness": twin_values.smoothness,
        },
        "git_hash": git_hash(),
        "params": recorded_params_dump(params),
    }


class TwinYSpectraStream(BinStream):
    """``twin_yspectra.bin`` writer (module docstring)."""

    def __init__(
        self,
        twin_values,
        y_weights: list[float],
        directory: str | Path = ".",
    ) -> None:
        self.includes_ref = bool(twin_values.spectra_ref)
        ny = params.res.ny
        n_kz, n_kx = marginal_bin_counts()
        prefixes = ("e", "r") if self.includes_ref else ("e",)
        fields = tuple(
            (f"{p}_{suf}", (3, ny, n))
            for p in prefixes
            for suf, n in (("x", n_kz), ("z", n_kx), ("x0", n_kz))
        )
        sidecar = _common_sidecar(twin_values, y_weights) | {
            "format_version": YSPECTRA_FORMAT_VERSION,
            "includes_ref": self.includes_ref,
            "it_yspectra": twin_values.it_yspectra,
            "note": (
                "componentwise energy density in y: "
                "sigma_kx |u_c|^2 / (2 V); one-sided folded k axes; "
                "y_weights . e_x summed over k == E_d"
            ),
        }
        directory = Path(directory)
        super().__init__(
            fields=fields,
            sidecar=sidecar,
            match_keys=_YSPECTRA_MATCH_KEYS,
            bin_path=directory / "twin_yspectra.bin",
            json_path=directory / "twin_yspectra.json",
            value_dtype=sidecar["value_dtype"],
            nbuffer=_NBUFFER,
        )


class TwinYBudgetStream(BinStream):
    """``twin_ybudget.bin`` writer (module docstring)."""

    def __init__(
        self,
        twin_values,
        y_weights: list[float],
        directory: str | Path = ".",
    ) -> None:
        ny = params.res.ny
        n_kz, n_kx = marginal_bin_counts()
        terms = ybudget_terms(bool(twin_values.rotational_ybudget))
        fields = tuple(
            (f"{term}_{suf}", (ny, n))
            for term in terms
            for suf, n in (("x", n_kz), ("z", n_kx), ("x0", n_kz))
        )
        sidecar = _common_sidecar(twin_values, y_weights) | {
            "format_version": YBUDGET_FORMAT_VERSION,
            "terms": list(terms),
            "it_ybudget": twin_values.it_ybudget,
            "note": (
                "component-summed budget densities in y; the terms "
                "sum to d_t e(y, k), except that under the rotational "
                "form the trailing P_lift (the classical "
                "-Re{Du_i* Dv} d_y U_i) sits outside that sum.  V is "
                "the operator (closure-consistent) viscous form and "
                "eps the pseudo-dissipation; Wp is the work of the "
                "pressure gradient (the applied driving at k = 0)"
            ),
        }
        directory = Path(directory)
        super().__init__(
            fields=fields,
            sidecar=sidecar,
            match_keys=_YBUDGET_MATCH_KEYS,
            bin_path=directory / "twin_ybudget.bin",
            json_path=directory / "twin_ybudget.json",
            value_dtype=sidecar["value_dtype"],
            nbuffer=_NBUFFER,
        )
