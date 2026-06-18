#!/usr/bin/env python3
r"""Convert external-simulator velocity fields into dnsjax snapshots.

This is a **library** (not a CLI): it exposes functions that future,
per-simulator CLIs import to pack a velocity field produced by another
DNS code (Fourier x finite-difference x Fourier for wall-bounded flows,
or Fourier x Fourier x Fourier for triply-periodic flows) into dnsjax's
native single-file (tar-wrapped zarr3) snapshot format.  It mirrors how
``scripts/random_field.py`` configures the global parameter singletons
and calls ``snapshot.save_snapshot``, but instead of *generating* a
field it *packs a supplied one*.

Input field layout
------------------
The input velocity field is array-like with layout

    ``[component, streamwise, wall-normal, spanwise]``

(for triply-periodic flows the wall-normal direction is the "shearwise"
direction).  Components are ordered streamwise / wall-normal / spanwise.
Concretely, per system:

=================  =====================  =============================
system family      input components       input axes (1, 2, 3)
=================  =====================  =============================
Cartesian          `$(u_x, u_y, u_z)$`    `$(x, y, z)$`
triply-periodic    `$(u_x, u_y, u_z)$`    `$(x, y, z)$`
pipe               `$(u_z, u_r, u_\theta)$`   `$(z_{ax}, r, \theta)$`
Taylor-Couette     `$(u_\theta, u_r, u_z)$`   `$(\theta, r, z_{ax})$`
=================  =====================  =============================

The field may be supplied in **physical** space or in **spectral**
(Fourier-transformed) space.  In both cases it must be at the native
resolution -- **no 3/2 dealiasing expansion / padding**.  For spectral
input, either the streamwise *or* the spanwise axis may be the
real-to-complex (rfft) axis holding only non-negative wavenumbers; pick
it with ``real_axis`` (``"streamwise"`` <-> `$k_x$`, ``"spanwise"`` <->
`$k_z$`).

dnsjax native layout (what is stored)
-------------------------------------
The snapshot stores the **complex spectral perturbation velocity** at
true (unpadded) resolution; with ``np=1`` here there is no device
padding, so the in-memory state is exactly what is written.  Components
and axes per family:

=================  ==================  ====================
system family      state components    state axes
=================  ==================  ====================
Cartesian          `$(u_x,u_y,u_z)$`   `$(y, k_z, k_x)$`
triply-periodic    `$(u_x,u_y,u_z)$`   `$(k_y, k_z, k_x)$`
pipe / TC          `$(u_z,u_+,u_-)$`   `$(r, m, k_z)$`
=================  ==================  ====================

True (unpadded) shapes are ``(ny, nz-1, nx//2)`` (Cartesian, pipe,
TC) and ``(ny-1, nz-1, nx//2)`` (triply-periodic).

with `$u_\pm = u_r \pm i\,u_\theta$`.  dnsjax always maps the ``nx`` /
real-FFT slot to the streamwise direction for Cartesian/periodic/pipe,
but for **Taylor-Couette the axial direction is the real-FFT (``nx``)
axis and the azimuthal direction the complex (``nz``) axis** -- so the
streamwise (azimuthal) resolution is ``nz`` and the spanwise (axial)
resolution is ``nx`` (confirmed from the ``annular.py`` ``Fourier``
coordinate-mapping table).  This module encapsulates that mapping so
callers always pass the natural ``[streamwise, wall-normal, spanwise]``
field.

Algorithm (single forward path; spectral input adds an inverse pre-step)
-----------------------------------------------------------------------
1. If ``space == "spectral"``: inverse-transform the input to physical
   space -- ``ifft`` along the full-complex Fourier axes, ``irfft``
   along ``real_axis`` (last, to yield a real field) -- using the
   source's normalisation (``input_norm``, numpy naming).  The
   wall-normal / radial axis is never transformed for wall-bounded
   flows.  A missing Nyquist mode is tolerated (zero-filled).
2. Permute the input axes to the dnsjax physical layout
   ``[c, (y|r), (z|theta), (x|z_ax)]`` (axis sizes ``(ny, nz, nx)``).
3. Form the dnsjax component basis: identity for Cartesian/periodic;
   `$(u_z, u_+, u_-)$` for pipe/TC.
4. Forward-transform with ``norm="forward"`` (dnsjax convention): a full
   ``fft`` along every Fourier axis, then drop the Nyquist mode -- keep
   the non-negative half `$[0, n/2)$` on the real axis
   (``operators.real_harmonics``) and ``operators.complex_harmonics``
   order on the full axes.  Using a *full* ``fft`` then truncating the
   real axis (rather than ``rfft``) is required because `$u_\pm$` are
   complex, and is identity-equivalent for real components.  The
   wall-normal / radial axis is left as grid samples for wall-bounded
   flows.

Perturbation only
-----------------
dnsjax snapshots store the perturbation `$\mathbf{u}'$` around the
laminar base flow `$\mathbf{U}$`, which lives in the ``flow`` dataclass
and is **not** part of the state.  This module performs **no** base-flow
subtraction: the input field must already be a perturbation around
dnsjax's base flow for the chosen system.

Wall-normal grid
----------------
For wall-bounded flows the field is stored on the *supplied*
``wall_normal_grid`` (recorded in the snapshot metadata via
``derived_params.wall_normal_grid``); dnsjax interpolates it to the run
grid at load time (``__main__._interpolate_if_needed``).  The grid must
lie on the canonical domain (`$[-1, 1]$` Cartesian, `$(0, 1]$` pipe,
`$[r_1, r_2]$` Taylor-Couette); nondimensionalisation is the caller's
responsibility.

Usage
-----
One conversion per process (the geometry ``fourier`` singleton is built
at import, exactly as in ``random_field.py``)::

    from snapshot_import import convert_field_to_snapshot

    convert_field_to_snapshot(
        field, "ic_snapshot",
        system="plane-couette", nx=128, ny=65, nz=128,
        lx=4.0, lz=4.0, wall_normal_grid=ys, re=400.0,
        space="physical",
    )

or, for finer control::

    import snapshot_import as si
    si.configure_target("kolmogorov", 64, 64, 64, lx=4.0, lz=4.0)
    state = si.to_spectral_state(field, space="spectral",
                                 real_axis="streamwise",
                                 input_norm="backward")
    si.write_snapshot(state, "ic_snapshot", t=0.0, it=0)
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np

__all__ = [
    "configure_target",
    "to_spectral_state",
    "write_snapshot",
    "convert_field_to_snapshot",
    "validate_state",
]

Space = Literal["physical", "spectral"]
RealAxis = Literal["streamwise", "spanwise"]
InputNorm = Literal["backward", "forward", "ortho"]


# ── Geometry descriptors ─────────────────────────────────────────


def _geo_family(system: str) -> str:
    """Map a flow ``system`` to a geometry family string.

    Returns one of ``"cartesian"``, ``"periodic"``, ``"pipe"``,
    ``"annular"``.  Imports the system lists lazily (JAX/params must be
    configured first).
    """
    from dnsjax.parameters import (
        annular_systems,
        cartesian_systems,
        cylindrical_systems,
        periodic_systems,
    )

    if system in cartesian_systems:
        return "cartesian"
    if system in periodic_systems:
        return "periodic"
    if system in cylindrical_systems:
        return "pipe"
    if system in annular_systems:
        return "annular"
    raise ValueError(f"unknown system: {system!r}")


def _input_axis_sizes(
    family: str, nx: int, ny: int, nz: int
) -> tuple[int, int, int]:
    """Physical sizes of the input axes ``(streamwise, wn, spanwise)``.

    dnsjax maps the ``nx`` (real-FFT) slot to the streamwise direction
    for every family *except* Taylor-Couette, whose streamwise
    (azimuthal) direction is the ``nz`` slot and spanwise (axial) the
    ``nx`` slot.
    """
    if family == "annular":
        return (nz, ny, nx)
    return (nx, ny, nz)


def _permute(family: str) -> tuple[int, int, int, int]:
    """Axis permutation: input ``[c, stream, wn, span]`` ->
    dnsjax physical ``[c, (y|r), (z|theta), (x|z_ax)]``."""
    if family == "annular":
        # input [c, theta, r, z_ax] -> [c, r, theta, z_ax]
        return (0, 2, 1, 3)
    # input [c, x|z_ax, y|r, z|theta] -> [c, y|r, z|theta, x|z_ax]
    return (0, 2, 3, 1)


# ── Wavenumber gather (numpy FFT order -> dnsjax order) ──────────


def _full_axis_gather(n: int) -> np.ndarray:
    r"""Indices selecting ``complex_harmonics(n)`` order from a length
    ``n`` FFT output (numpy order).

    Equivalent to deleting index ``n // 2`` (the Nyquist slot), but
    built by matching integer wavenumbers against
    ``operators.complex_harmonics`` so it cannot drift from the solver.
    """
    from dnsjax.operators import complex_harmonics

    fft_freqs = (np.arange(n) + n // 2) % n - n // 2  # numpy FFT order
    target = np.asarray(complex_harmonics(n))
    index_of = {int(q): i for i, q in enumerate(fft_freqs)}
    return np.array([index_of[int(q)] for q in target], dtype=int)


def _real_axis_gather(n: int) -> np.ndarray:
    """Indices selecting ``real_harmonics(n)`` from a length ``n`` FFT
    output: the non-negative half ``[0, n//2)`` (FFT index == wavenumber
    for non-negative modes)."""
    from dnsjax.operators import real_harmonics

    return np.asarray(real_harmonics(n)).astype(int)


# ── Component basis ──────────────────────────────────────────────


def _make_components(family: str, arr: Any) -> Any:
    r"""Map input components (axis 0) to the dnsjax basis.

    Cartesian / periodic: identity ``(u_x, u_y, u_z)``.  Pipe / TC:
    ``(u_z, u_+, u_-)`` with `$u_\pm = u_r \pm i\,u_\theta$`.  ``arr`` is
    the post-permute dnsjax-physical array (axis 0 still in input order).
    """
    from jax import numpy as jnp

    if family in ("cartesian", "periodic"):
        return arr
    if family == "pipe":  # input (u_z, u_r, u_theta)
        u_z, u_r, u_th = arr[0], arr[1], arr[2]
    else:  # annular / TC, input (u_theta, u_r, u_z)
        u_th, u_r, u_z = arr[0], arr[1], arr[2]
    return jnp.stack([u_z, u_r + 1j * u_th, u_r - 1j * u_th], axis=0)


# ── Transforms ───────────────────────────────────────────────────


def _reinsert_full_nyquist(field: Any, axis: int, n: int) -> Any:
    """Restore numpy FFT order (length ``n``) on a full-complex axis.

    A standard numpy-order axis (length ``n``) is returned unchanged; a
    dnsjax-style axis with the Nyquist dropped (length ``n - 1``,
    ``complex_harmonics`` order) gets a zero re-inserted at index
    ``n // 2``.
    """
    from jax import numpy as jnp

    m = field.shape[axis]
    if m == n:
        return field
    if m == n - 1:
        return jnp.insert(field, n // 2, 0, axis=axis)
    raise ValueError(
        f"spectral axis {axis} has length {m}; expected n={n} or n-1"
    )


def _inverse_to_physical(
    field: Any,
    real_axis: RealAxis,
    input_norm: InputNorm,
    periodic: bool,
    sizes: tuple[int, int, int],
) -> Any:
    """Inverse-transform a spectral input to a real physical field.

    Operates in input-axis order ``[c, stream(1), wn(2), span(3)]``.
    The full-complex Fourier axes are inverted with ``ifft``; the
    ``real_axis`` is inverted last with ``irfft`` to produce a real
    field.  ``input_norm`` is the source's forward-transform
    normalisation (numpy naming).
    """
    from jax import numpy as jnp

    phys_size = {1: sizes[0], 2: sizes[1], 3: sizes[2]}
    fourier_axes = {1, 2, 3} if periodic else {1, 3}
    real_ax = 1 if real_axis == "streamwise" else 3
    if real_ax not in fourier_axes:
        raise ValueError(f"real_axis {real_axis!r} is not a Fourier axis")

    out = field
    for ax in sorted(fourier_axes - {real_ax}):
        out = _reinsert_full_nyquist(out, ax, phys_size[ax])
        out = jnp.fft.ifft(out, axis=ax, norm=input_norm)
    out = jnp.fft.irfft(
        out, n=phys_size[real_ax], axis=real_ax, norm=input_norm
    )
    return out


def _forward_to_spectral(
    arr: Any, periodic: bool, nx: int, ny: int, nz: int
) -> Any:
    """Forward-transform a dnsjax-physical array to the spectral state.

    ``arr`` has layout ``[c, (y|r), (z|theta), (x|z_ax)]`` with axis
    sizes ``(ny, nz, nx)``.  Returns the spectral state
    ``[c, (y|k_y), k_z, k_x]`` at true (unpadded) shape.
    """
    from jax import numpy as jnp

    out = jnp.fft.fft(arr, axis=3, norm="forward")  # real axis (nx)
    out = jnp.fft.fft(out, axis=2, norm="forward")  # full axis (nz)
    if periodic:
        out = jnp.fft.fft(out, axis=1, norm="forward")  # full axis (ny)

    out = jnp.take(out, _real_axis_gather(nx), axis=3)
    out = jnp.take(out, _full_axis_gather(nz), axis=2)
    if periodic:
        out = jnp.take(out, _full_axis_gather(ny), axis=1)
    return out


# ── Input validation ─────────────────────────────────────────────


def _validate_input_shape(
    field: Any,
    space: Space,
    real_axis: RealAxis,
    periodic: bool,
    sizes: tuple[int, int, int],
) -> None:
    """Check the input field shape against the configured resolution."""
    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError(
            f"field must have shape (3, stream, wn, span); got {field.shape}"
        )
    n1, n2, n3 = sizes
    if space == "physical":
        if tuple(field.shape[1:]) != (n1, n2, n3):
            raise ValueError(
                f"physical field shape {tuple(field.shape[1:])} != "
                f"expected ({n1}, {n2}, {n3})"
            )
        return
    if space != "spectral":
        raise ValueError(
            f"space must be 'physical' or 'spectral'; got {space!r}"
        )

    real_ax = 1 if real_axis == "streamwise" else 3
    fourier_axes = {1, 2, 3} if periodic else {1, 3}
    phys = {1: n1, 2: n2, 3: n3}
    for ax in (1, 2, 3):
        m = field.shape[ax]
        n = phys[ax]
        if ax == real_ax:
            if m not in (n // 2, n // 2 + 1):
                raise ValueError(
                    f"spectral real axis {ax} length {m}; expected "
                    f"{n // 2} or {n // 2 + 1}"
                )
        elif ax in fourier_axes:
            if m not in (n, n - 1):
                raise ValueError(
                    f"spectral full axis {ax} length {m}; expected {n} or "
                    f"{n - 1}"
                )
        else:  # untransformed wall-normal axis
            if m != n:
                raise ValueError(f"wall-normal axis {ax} length {m} != ny {n}")


def _validate_grid_domain(system: str, family: str, grid: list[float]) -> None:
    """Validate a wall-normal/radial grid lies on the canonical domain.

    Raises on out-of-domain points or a non-ascending grid; warns if the
    walls are not resolved at the domain endpoints.
    """
    from dnsjax.parameters import derived_params

    g = np.asarray(grid)
    if np.any(np.diff(g) <= 0):
        raise ValueError(
            f"{system}: wall_normal_grid must be strictly ascending"
        )

    tol = 1e-9
    if family == "cartesian":
        lo, hi = -1.0, 1.0
    elif family == "pipe":
        lo, hi = 0.0, 1.0  # (0, 1]; r = 0 (the axis) excluded
    else:  # annular / TC
        lo, hi = derived_params.r_inner, derived_params.r_outer

    if g[0] < lo - tol or g[-1] > hi + tol:
        raise ValueError(
            f"{system}: wall_normal_grid spans [{g[0]:.6g}, {g[-1]:.6g}], "
            f"outside the canonical domain [{lo:.6g}, {hi:.6g}]"
        )
    if family == "pipe" and g[0] <= 0.0:
        raise ValueError("pipe: radial grid must exclude r = 0 (use (0, 1])")
    wall_lo = lo if family != "pipe" else hi  # pipe: only the outer wall
    if family == "pipe":
        if abs(g[-1] - hi) > 1e-6:
            print(
                f"  warning: outer wall r=1 not resolved (r[-1]={g[-1]:.6g})"
            )
    else:
        if abs(g[0] - lo) > 1e-6 or abs(g[-1] - hi) > 1e-6:
            print(
                f"  warning: walls not resolved at the domain endpoints "
                f"[{lo:.6g}, {hi:.6g}] (grid ends [{g[0]:.6g}, {g[-1]:.6g}])"
            )
    _ = wall_lo  # documented intent; only used for the message above


# ── Public API ───────────────────────────────────────────────────


def configure_target(
    system: str,
    nx: int,
    ny: int,
    nz: int,
    *,
    lx: float = 4.0,
    lz: float = 4.0,
    wall_normal_grid: Any | None = None,
    double_precision: bool = True,
    fd_order: int = 4,
    tilt_degree: float = 0.0,
    re: float = 1000.0,
    re1: float | None = None,
    re2: float | None = None,
    eta: float | None = None,
    driving: str = "constant_pressure_gradient",
    block_mean_spanwise_velocity: bool = False,
    extra_params: dict[str, Any] | None = None,
) -> None:
    """Configure JAX and the global dnsjax parameter singletons.

    Must be called **once per process, before** ``to_spectral_state`` /
    ``write_snapshot`` (the geometry ``fourier`` singleton is built at
    import).  Mirrors ``random_field._setup_jax_and_params`` (single
    device, CPU).

    Parameters
    ----------
    system, nx, ny, nz:
        Flow system and dnsjax resolution.  For Taylor-Couette ``nx`` is
        the axial (spanwise) and ``nz`` the azimuthal (streamwise)
        resolution; for all other systems ``nx`` is streamwise and ``nz``
        spanwise.  See the module docstring.
    lx, lz:
        Streamwise / spanwise domain lengths (recorded in metadata;
        forced to `$2\\pi$` internally for pipe/TC).
    wall_normal_grid:
        Wall-normal / radial grid (length ``ny``) for wall-bounded flows;
        must be ``None`` for triply-periodic.  Stored in the snapshot and
        interpolated to the run grid at load.
    re, re1, re2, eta:
        Reynolds number(s).  Taylor-Couette requires ``re1``, ``re2``,
        ``eta``.
    extra_params:
        Optional nested dict (e.g. ``{"step": {"dt": 0.01}}``) of any
        further parameters to record in the snapshot metadata.
    """
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    os.environ["NPROC"] = "1"

    import jax

    jax.config.update("jax_enable_x64", double_precision)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.parameters import (
        Parameters,
        derived_params,
        padded_res,
        params,
        periodic_systems,
        update_parameters,
    )

    cli = Parameters(
        dist={"np": 1, "platform": "cpu"},
        phys={
            "system": system,
            "re": re,
            "re1": re1,
            "re2": re2,
            "driving": driving,
            "block_mean_spanwise_velocity": block_mean_spanwise_velocity,
        },
        geo={"lx": lx, "lz": lz, "tilt_degree": tilt_degree, "eta": eta},
        res={
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "fd_order": fd_order,
            "double_precision": double_precision,
        },
        outs={},
    )
    update_parameters(cli)
    if extra_params is not None:
        update_parameters(Parameters(**extra_params))
    padded_res.set_padded_resolution(params)

    family = _geo_family(system)
    if system in periodic_systems:
        if wall_normal_grid is not None:
            raise ValueError(
                "wall_normal_grid is not used for triply-periodic systems"
            )
        derived_params.wall_normal_grid = None
    else:
        if wall_normal_grid is None:
            raise ValueError(
                f"{system} requires wall_normal_grid (length {ny})"
            )
        grid = [float(v) for v in np.asarray(wall_normal_grid).ravel()]
        if len(grid) != ny:
            raise ValueError(f"wall_normal_grid length {len(grid)} != ny {ny}")
        _validate_grid_domain(system, family, grid)
        derived_params.wall_normal_grid = grid


def to_spectral_state(
    field: Any,
    *,
    space: Space = "physical",
    real_axis: RealAxis = "streamwise",
    input_norm: InputNorm = "backward",
) -> Any:
    r"""Pack an input velocity field into dnsjax's spectral state.

    Requires ``configure_target`` to have been called.  ``field`` has
    layout ``[component, streamwise, wall-normal, spanwise]`` at the
    native (unpadded) resolution.

    Parameters
    ----------
    space:
        ``"physical"`` or ``"spectral"`` (Fourier-transformed in the
        periodic directions, no 3/2 padding).
    real_axis:
        For ``space="spectral"``: which axis holds only non-negative
        wavenumbers -- ``"streamwise"`` (`$k_x$`) or ``"spanwise"``
        (`$k_z$`).  Ignored for physical input.
    input_norm:
        For ``space="spectral"``: the source's forward-FFT normalisation
        (numpy naming; default ``"backward"``).  Ignored for physical
        input.

    Returns
    -------
    :
        Complex spectral state of shape ``(3, *spec_shape)``, typed and
        sharded for the current (single-device) configuration.
    """
    import jax
    from jax import numpy as jnp

    from dnsjax.parameters import params
    from dnsjax.sharding import sharding

    system = params.phys.system
    family = _geo_family(system)
    periodic = family == "periodic"
    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz
    sizes = _input_axis_sizes(family, nx, ny, nz)

    field = jnp.asarray(field)
    _validate_input_shape(field, space, real_axis, periodic, sizes)

    if space == "spectral":
        field = _inverse_to_physical(
            field, real_axis, input_norm, periodic, sizes
        )

    arr = jnp.transpose(field, _permute(family))
    arr = _make_components(family, arr)
    state = _forward_to_spectral(arr, periodic, nx, ny, nz)
    state = state.astype(sharding.complex_type)
    return jax.device_put(state, sharding.spec_vector_shard)


def write_snapshot(
    state: Any, output_path: str, *, t: float = 0.0, it: int = 0
) -> None:
    """Write a spectral state to a single-file (tar) snapshot.

    Thin wrapper over ``snapshot.save_snapshot``; records the configured
    parameters, the wall-normal grid, ``t`` and ``it`` in the snapshot's
    ``_dnsjax_meta.json`` member.  *output_path* is the tar file path.
    """
    from dnsjax.snapshot import save_snapshot

    save_snapshot(state, t=t, it=it, path=output_path)


def convert_field_to_snapshot(
    field: Any,
    output_path: str,
    *,
    system: str,
    nx: int,
    ny: int,
    nz: int,
    space: Space = "physical",
    real_axis: RealAxis = "streamwise",
    input_norm: InputNorm = "backward",
    t: float = 0.0,
    it: int = 0,
    **config: Any,
) -> None:
    """One-shot conversion: configure, pack, and write a snapshot.

    ``**config`` is forwarded to ``configure_target`` (``lx``, ``lz``,
    ``wall_normal_grid``, ``re``, ``re1``, ``re2``, ``eta``,
    ``tilt_degree``, ``driving``, ``double_precision``, ``fd_order``,
    ``extra_params``, ...).
    """
    configure_target(system, nx, ny, nz, **config)
    state = to_spectral_state(
        field, space=space, real_axis=real_axis, input_norm=input_norm
    )
    write_snapshot(state, output_path, t=t, it=it)


def validate_state(state: Any, *, atol: float = 1e-8) -> dict[str, float]:
    """Light sanity checks on a converted spectral state (warn-only).

    Checks finiteness and, for wall-bounded flows, the no-slip wall
    boundary condition magnitude (the perturbation should vanish at the
    walls).  Does **not** modify the state and does **not** check
    divergence -- the solver projects residual divergence on the first
    corrector step at load.  Returns the measured metrics.
    """
    import numpy as _np

    s = _np.asarray(state)
    metrics: dict[str, float] = {}
    finite = bool(_np.all(_np.isfinite(s)))
    metrics["all_finite"] = float(finite)
    if not finite:
        print("  warning: state contains non-finite values")

    from dnsjax.parameters import params

    family = _geo_family(params.phys.system)
    if family in ("cartesian", "annular"):
        bc = max(
            float(_np.max(_np.abs(s[:, 0]))),
            float(_np.max(_np.abs(s[:, -1]))),
        )
        metrics["wall_bc"] = bc
        if bc > atol:
            print(
                f"  warning: nonzero perturbation at walls (max |u'|={bc:.2e})"
            )
    elif family == "pipe":
        bc = float(_np.max(_np.abs(s[:, -1])))  # outer wall r = 1
        metrics["wall_bc"] = bc
        if bc > atol:
            print(
                f"  warning: nonzero perturbation at r=1 (max |u'|={bc:.2e})"
            )
    return metrics
