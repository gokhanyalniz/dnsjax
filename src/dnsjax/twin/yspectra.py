r"""Wall-normal-resolved twin spectra and budget streams.

Two binary streams written by ``dnsjax-twin``
(:mod:`dnsjax.twin.driver`), both on the wall-normal-resolved,
one-sided marginal bins of
:func:`dnsjax.twin.diagnostics.marginal_bin_counts`:

- ``twin_yspectra.bin`` (``twin.it_yspectra``), from
  :func:`~dnsjax.twin.diagnostics.twin_yspectra`: the componentwise
  difference energy `$E_\Delta^x[u,v,w](y, k_z)$` and
  `$E_\Delta^z[u,v,w](y, k_x)$`, plus their `$(0, 0)$` mode
  `$E_\Delta^{xz00}[u,v,w](y)$`;
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
the wavenumber itself and drop the `$y$`-integration.  Recovering the
old three numbers takes the full `$k_x = 0$` plane, which is
``twin.x0_planes`` and is **off by default**; the `$(0, 0)$` mode --
`$E_{\Delta U}$` on its own, and the mean the reference fluctuation
energy is measured against -- is always there.

File format
===========
Both are flat sequences of fixed-size records, ``("t", "<f8")``
followed by one entry per field, ``VAL = "<f8"``/``"<f4"`` per
``res.double_precision``.  The field table is the outer product of
the prefixes (or the budget terms) with
:func:`~dnsjax.twin.diagnostics.marginal_suffixes`, **term-major**:

.. code-block:: text

    suffix    x           z           x0          xz00
    shape     (.., n_kz)  (.., n_kx)  (.., n_kz)  (..,)
    stored    always      always      x0_planes   always

    twin_yspectra:  e_<suffix>    leading axis (3,) + (ny, ...)
                    r_<suffix>    when twin.spectra_ref
    twin_ybudget:   <term>_<suffix>  for each of ybudget_terms(),
                    leading axis (ny, ...)

The sidecar's ``suffixes`` names that middle row outright, so a
reader never infers the layout: :mod:`dnsjax.analysis.twin.yspectra`
reads it, and falls back to the pre-``xz00`` triple
``("x", "z", "x0")`` below this module's format versions.

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

The ``xz00`` column has no wavenumber axis at all, being one mode.

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
from .diagnostics import (
    marginal_bin_counts,
    marginal_suffixes,
    ybudget_terms,
)

#: Sidecar schema versions.  The reader's floors
#: (``analysis.twin.yspectra.MIN_*_VERSION``) are deliberately **not**
#: raised in step with these: version 2 / 3 added ``xz00`` and made
#: ``x0`` opt-in, which is a change of *layout*, not of what a stored
#: array means, and the sidecar's ``suffixes`` names the layout
#: outright.  Holding the floors is what keeps a member recorded
#: before this readable -- the one thing this pair of streams is
#: expected to do that the snapshot format is not.
YSPECTRA_FORMAT_VERSION: int = 2
YBUDGET_FORMAT_VERSION: int = 3


def _suffix_shapes(x0: bool) -> tuple[tuple[str, tuple[int, ...]], ...]:
    r"""``(suffix, trailing shape)`` in stored order.

    The trailing shape is what a field carries *after* its leading
    axis (the three velocity components, or the wall-normal grid for
    the budget): a wavenumber axis for the two marginals and the
    `$k_x = 0$` plane, nothing at all for the single `$(0, 0)$` mode.
    The reader's counterpart is
    :func:`dnsjax.analysis.twin.yspectra.stored_fields`.
    """
    n_kz, n_kx = marginal_bin_counts()
    widths: dict[str, tuple[int, ...]] = {
        "x": (n_kz,),
        "z": (n_kx,),
        "x0": (n_kz,),
        "xz00": (),
    }
    return tuple((suf, widths[suf]) for suf in marginal_suffixes(x0))


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
    "suffixes",
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
    "suffixes",
    "terms",
    "value_dtype",
    "it_ybudget",
    "dt",
    "double_precision",
    "lx",
    "lz",
)


def _common_sidecar(twin_values, y_weights: list[float]) -> dict:
    """The keys both sidecars share (grid, axes, layout, provenance).

    ``suffixes`` is the layout key: it names the field table both
    streams are laid out by, so a reader never has to infer one, and
    it is a match key in both -- a resume that flipped
    ``twin.x0_planes`` would otherwise append records of a different
    size onto an existing ``.bin``.
    """
    n_kz, n_kx = marginal_bin_counts()
    return {
        "system": params.phys.system,
        "ny": params.res.ny,
        "n_kz": n_kz,
        "n_kx": n_kx,
        "suffixes": list(marginal_suffixes(bool(twin_values.x0_planes))),
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
            "wall_smoothness": twin_values.wall_smoothness,
            "wall_confinement": twin_values.wall_confinement,
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
        prefixes = ("e", "r") if self.includes_ref else ("e",)
        fields = tuple(
            (f"{p}_{suf}", (3, ny, *shape))
            for p in prefixes
            for suf, shape in _suffix_shapes(bool(twin_values.x0_planes))
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
        terms = ybudget_terms(bool(twin_values.rotational_ybudget))
        fields = tuple(
            (f"{term}_{suf}", (ny, *shape))
            for term in terms
            for suf, shape in _suffix_shapes(bool(twin_values.x0_planes))
        )
        sidecar = _common_sidecar(twin_values, y_weights) | {
            "format_version": YBUDGET_FORMAT_VERSION,
            "terms": list(terms),
            "it_ybudget": twin_values.it_ybudget,
            "note": (
                "component-summed budget densities in y.  Summing the "
                "terms gives d_t e(y, k), with two exclusions.  eps is "
                "the positive-definite pseudo-dissipation companion to "
                "the operator (closure-consistent) viscous form V -- "
                "only V is in the balance, and their difference is the "
                "wall-normal diffusion flux.  And where the rotational "
                "form writes it, the trailing P_lift (the classical "
                "-Re{Du_i* Dv} d_y U_i, carried unchanged from the "
                "convective form) also sits outside the sum.  Wp is "
                "the work of the pressure gradient (the applied "
                "driving at k = 0)"
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
