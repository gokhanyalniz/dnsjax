r"""Reduced-resolution snapshots (``[lowres]``), written during a run.

Every ``lowres.it_lowres`` steps the solver writes
``lowres/lowres_{it:010d}.tar``: the stored state -- the velocity, plus
the conformation tensor of the viscoelastic flows -- at the reduced
resolution ``[lowres]`` names, and, under ``lowres.pressure`` (plane
Couette and plane Poiseuille), its static pressure as one more member.
``dnsjax-twin`` writes the same files for its reference state and, on a
cadence of its own, for the difference field (:mod:`dnsjax.twin.driver`).
The point is a cadence the full snapshots cannot afford on disk, which
is also why this runs inside the solver rather than in a ``scripts/``
tool: offline it would need full snapshots at that cadence, and the
pressure has to come from the full-resolution field.

What a file holds
=================
A format-6 snapshot (:mod:`dnsjax.snapshot`), read by
:func:`dnsjax.analysis.read_state` like any other, with

- ``state/``: the physical components, as a normal snapshot stores them;
- ``pressure/`` (optional): one component, the static pressure
  perturbation `$p'$` of a state or the difference `$\Delta p$`
  (:func:`dnsjax.analysis.read_pressure`; the metadata ``pressure``
  entry says which, and the gauge);
- metadata describing the **reduced** field: ``native_shape``, the
  wall-normal grid it lives on, and a parameter dump whose ``res`` is
  the reduced resolution under the flow's public names -- what every
  reader takes the shapes from -- plus a ``lowres`` entry with the
  run's own resolution and the field kind.

No ``carry/`` member and no stats: a reduced snapshot is output, not a
checkpoint, and a resume refuses one
(:func:`dnsjax.snapshot_meta.checkpoint_refusal`).

How a field is reduced
======================
Every field is computed at the run's resolution first -- the pressure
above all, whose source is quadratic in the velocity -- and only then
reduced, axis by axis, where each axis is local:

- **wall-normal**, in the solver layout: the regrid a resume applies
  (:func:`dnsjax.snapshot.apply_wall_normal_regrid`) onto the same grid
  type at the reduced count -- Chebyshev coefficient truncation between
  CGL grids, the parity maps between the pipe's radial CGL grids, the
  local stencil otherwise -- with the velocity's wall rows reset to
  zero and the pressure's `$(0, 0)$` mode re-pinned to zero at the
  upper wall: a truncation moves end values, and both are stated
  exactly (the no-slip condition, the pressure's gauge).  Both grids
  are built in float64 by
  :func:`dnsjax.fd.grid_nodes`, so the CGL path holds in a
  single-precision run too.  A custom ``geo.wall_grid`` keeps its own
  count (``validate_parameters``);
- **periodic** `$k_y$`, in the solver layout: the highest modes
  dropped (:func:`dnsjax.snapshot._resize_wrapped_axis`);
- `$k_z$` and `$k_x$`, inside the snapshot reshard: the highest modes
  dropped at no extra collective (``snapshot._to_io_layout_core``).

Cost
====
Per file: the reshard and write of a reduced field, a wall-normal GEMM,
and with the pressure one static-pressure sample (15 field transforms
and one banded solve,
:func:`~dnsjax.geometries.wall_bounded._cartesian_pressure.static_pressure`),
whose transient is some three fifths of the time step's.  The pressure
operator is resident for the run once enabled: a second banded factor
set the size of the Poisson factors (its module's "Cost").
"""

from __future__ import annotations

import copy
import importlib
from functools import partial
from pathlib import Path

from jax import Array, jit
from jax import numpy as jnp

from .flows.registry import cartesian_systems, periodic_systems, spec_for
from .parameters import derived_params, params
from .snapshot import (
    PARITY_EVEN_STORED,
    StoredLayout,
    _resize_wrapped_axis,
    apply_wall_normal_regrid,
    save_snapshot,
    wall_regrid_geometry,
    wall_regrid_matrix,
    wall_velocity_rows,
)

#: Run-directory subdirectory of the state files.
LOWRES_DIR: str = "lowres"

#: Zero-padded width of the step count in every file name here.
IT_WIDTH: int = 10


def lowres_path(directory: str | Path, stem: str, it: int) -> Path:
    """``<directory>/<stem>_<it>.tar``, the step count zero-padded."""
    return Path(directory) / f"{stem}_{it:0{IT_WIDTH}d}.tar"


def lowres_due(it: int, anchor: int = 0) -> bool:
    """Whether ``[lowres]`` writes the state the run holds at *it*.

    *anchor* is the step the cadence counts from: ``0`` for the solver
    (the absolute count, like ``outs.it_snapshot``), the member's
    ``parent_it`` for ``dnsjax-twin``.
    """
    cadence = params.lowres.it_lowres
    return cadence is not None and (it - anchor) % cadence == 0


@partial(jit, static_argnames=("parity", "zero_rows", "ky"))
def _reduce(
    field: Array,
    T,
    m_even,
    *,
    parity: tuple[bool, ...],
    zero_rows: tuple[int, ...],
    ky: tuple[int, int] | None,
) -> Array:
    """The leading-axis half of a reduction (module docstring).

    *T* ``None`` skips the wall-normal regrid; *ky* ``(src, dst)`` cuts
    the periodic `$k_y$`.  Every array is an argument
    (``.claude/rules/jax.md``).
    """
    if T is not None:
        field = apply_wall_normal_regrid(
            field,
            T,
            m_even=m_even,
            parity_even=parity,
            zero_velocity_walls=zero_rows,
        )
    if ky is not None:
        field = _resize_wrapped_axis(field, 1, *ky)
    return field


@jit
def _pin_mean_top(pressure: Array, mean_mask: Array) -> Array:
    """Re-gauge a reduced ``(1, Ny, Nkz, Nkx)`` pressure.

    Its `$(0, 0)$` mode, zero at the upper wall at full resolution,
    moves there by the wall-normal reduction's truncation error;
    subtracting that end value shifts the mean profile by a constant
    and restores the gauge the metadata states.  *mean_mask* is the
    geometry's one-hot ``fourier.mean_mask``, an argument like every
    array here (``.claude/rules/jax.md``).
    """
    return pressure - jnp.where(mean_mask, pressure[:, -1:], 0.0)


class LowResWriter:
    r"""The reduction of this run's fields to ``[lowres]``.

    Built once, when a consumer is enabled: the targets, the grids, the
    regrid matrix and -- under ``lowres.pressure`` -- the pressure
    operator are field-independent.  *pressure_op* lets a caller that
    already holds one share it (``dnsjax-twin``'s
    :class:`~dnsjax.twin.pressure.DifferencePressure` is one); left
    ``None`` it is built here when the pressure is on.
    """

    def __init__(self, pressure_op=None) -> None:
        self.system = params.phys.system
        self.spec = spec_for(self.system)
        self.periodic = self.system in periodic_systems
        lo, res = params.lowres, params.res
        self.target = {
            axis: getattr(lo, axis) or getattr(res, axis)
            for axis in ("nx", "ny", "nz")
        }
        a = self.target["ny"] - 1 if self.periodic else self.target["ny"]
        self.kz = self.target["nz"] - 1
        self.kx = self.target["nx"] // 2

        self.T = None
        self.m_even = None
        self.ky = None
        grid = derived_params.wall_normal_grid
        if self.periodic:
            if self.target["ny"] != res.ny:
                self.ky = (res.ny - 1, a)
        elif self.target["ny"] != res.ny:
            from .fd import grid_nodes

            family = wall_regrid_geometry()
            nodes = partial(
                grid_nodes,
                family,
                grid_type=params.geo.grid_type,
                grid_stretch=params.geo.grid_stretch,
                r_inner=derived_params.r_inner,
                r_outer=derived_params.r_outer,
            )
            y_new = nodes(self.target["ny"])
            self.T = wall_regrid_matrix(nodes(res.ny), y_new)
            grid = [float(v) for v in y_new]
            if isinstance(self.T, tuple):
                from .geometries.wall_bounded.cylindrical import fourier

                self.m_even = fourier.m_is_even.astype(bool)
        self.wall_rows = () if self.periodic else wall_velocity_rows()

        self.pressure = None
        if lo.pressure and self.system in cartesian_systems:
            from .geometries.wall_bounded._cartesian_pressure import (
                PoissonPressure,
            )
            from .geometries.wall_bounded.cartesian import fourier

            self._flow = importlib.import_module(self.spec.flow_module).flow
            self._fourier = fourier
            self.pressure = (
                pressure_op
                if pressure_op is not None
                else PoissonPressure(self._flow, fourier)
            )

        dump = copy.deepcopy(_recorded_dump())
        for axis in ("nx", "ny", "nz"):
            dump["res"][self.spec.alias("res", axis)] = self.target[axis]
        self._layout = dict(
            a=a,
            kz=self.kz,
            kx=self.kx,
            wall_normal_grid=grid,
            params=dump,
        )
        self.source = {
            self.spec.alias("res", axis): getattr(res, axis)
            for axis in ("nx", "ny", "nz")
        }
        #: The targets under the flow's public names, for messages.
        self.target_public = {
            self.spec.alias("res", axis): self.target[axis]
            for axis in ("nx", "ny", "nz")
        }

    def due(self, it: int, anchor: int = 0) -> bool:
        """:func:`lowres_due` (a method, for callers holding a writer)."""
        return lowres_due(it, anchor)

    def static_pressure(self, state_phys: Array) -> Array:
        """The full-resolution static pressure of a physical state."""
        from .geometries.wall_bounded._cartesian_pressure import (
            static_pressure,
        )

        return static_pressure(
            state_phys, self.pressure, self._fourier, self._flow
        )

    def reduce(self, field: Array, *, scalar: bool = False) -> Array:
        """Regrid the leading axis of a full-resolution *field*.

        The `$k_z$` / `$k_x$` cut happens in the write
        (:meth:`write`); a *scalar* field takes the even parity class
        and keeps its wall rows.
        """
        if self.T is None and self.ky is None:
            return field
        n = field.shape[0]
        return _reduce(
            field,
            self.T,
            self.m_even,
            parity=(True,) * n if scalar else PARITY_EVEN_STORED[:n],
            zero_rows=() if scalar else self.wall_rows,
            ky=self.ky,
        )

    def write(
        self,
        fields: Array,
        pressure: Array | None,
        t: float,
        it: int,
        path: str | Path,
        *,
        field: str = "state",
        pressure_note: str = "",
        extra_meta: dict | None = None,
    ) -> None:
        r"""Reduce and write one file (collective; every process calls it).

        *fields* is the full-resolution physical field to store
        (``(C, Ny, Nkz, Nkx)``) and *pressure* its full-resolution
        pressure (``(Ny, Nkz, Nkx)``, ``None`` for none); *field*
        names what they are (``"state"``, ``"difference"``) and
        *pressure_note* what the pressure is, for the metadata.
        """
        extra = {
            "lowres": {
                "field": field,
                "source_res": self.source,
            }
        }
        if pressure is not None:
            extra["pressure"] = {
                "kind": "static",
                "of": pressure_note or field,
                "gauge": "the (0, 0) mode is zero at the upper wall",
            }
        extra |= extra_meta or {}
        reduced_p = None
        if pressure is not None:
            reduced_p = self.reduce(pressure[None], scalar=True)
            if self.T is not None:
                reduced_p = _pin_mean_top(reduced_p, self._fourier.mean_mask)
        save_snapshot(
            self.reduce(fields),
            t,
            it,
            path,
            isnap=None,
            layout=StoredLayout(**self._layout, extra=extra),
            pressure=reduced_p,
        )

    def write_state(
        self,
        state_phys: Array,
        t: float,
        it: int,
        path: str | Path,
        *,
        extra_meta: dict | None = None,
    ) -> None:
        """:meth:`write` for a state, its static pressure when enabled."""
        pressure = (
            None if self.pressure is None else self.static_pressure(state_phys)
        )
        self.write(
            state_phys,
            pressure,
            t,
            it,
            path,
            pressure_note="perturbation",
            extra_meta=extra_meta,
        )


def _recorded_dump() -> dict:
    """The run's public-named parameter dump (deferred import)."""
    from .param_surface import recorded_params_dump

    return recorded_params_dump(params)
