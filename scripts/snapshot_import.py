#!/usr/bin/env python3
r"""Convert a native-layout velocity field into a dnsjax snapshot.

This is a **library** (not a CLI): it exposes functions that future,
per-simulator CLIs import to pack a velocity field -- already arranged in
dnsjax's **native** component/axis structure -- into dnsjax's single-file
(tar-wrapped zarr3) snapshot format.  Like :mod:`dnsjax.random_field`
it configures the global parameter singletons (one system per process)
and calls ``snapshot.save_snapshot``, but instead of *generating* a field
it *packs a supplied one*.  Conversion runs single-device (``np = 1``);
any external ``[streamwise, wall-normal, spanwise]`` -> native
permutation and component mixing is the **caller's** responsibility, not
this module's.

Native input layout (physical and spectral)
-------------------------------------------
The input is array-like with shape ``(3, axis1, axis2, axis3)`` in
dnsjax's native ordering for every geometry:

=================  =====================  =========================
system family      components (axis 0)    axes (1, 2, 3)
=================  =====================  =========================
Cartesian          `$(u_x, u_y, u_z)$`    `$(y, z, x)$`
triply-periodic    `$(u_x, u_y, u_z)$`    `$(y, z, x)$`
pipe / TC          `$(u_z, u_+, u_-)$`    `$(r, \theta, z_{ax})$`
=================  =====================  =========================

with `$u_\pm = u_r \pm i\,u_\theta$` (formed by the caller).  Axis 3 is
the real-FFT (``nx``) slot, axis 2 the complex-FFT (``nz``) slot, axis 1
the wall-normal (``ny``; untransformed for wall-bounded) or `$k_y$`
(periodic) slot.  **Pipe and Taylor-Couette share the same native layout**
``(r, θ, z_ax)``: dnsjax maps the axial direction to the real-FFT (``nx``)
slot and the azimuthal to the complex (``nz``) slot, so for
Taylor-Couette the spanwise (axial) resolution is ``nx`` and the
streamwise (azimuthal) ``nz``.

Physical input has shape ``(3, ny, nz, nx)`` -- real for
Cartesian/periodic, complex for pipe/TC (``u_pm`` are complex).  Spectral
input has the same native layout with the Fourier axes transformed (no
3/2 dealiasing padding): the real axis (3) holds ``nx//2`` non-negative
modes (or ``nx//2 + 1`` with Nyquist), the complex axis (2) ``nz - 1``
modes in ``complex_harmonics`` order (or ``nz`` in numpy-FFT order with
Nyquist), and for periodic the `$k_y$` axis (1) likewise; ``input_norm``
(numpy naming) is the source's forward-FFT normalisation.

dnsjax native state (what is stored)
------------------------------------
The snapshot stores the **complex spectral perturbation velocity** at
true (unpadded) resolution (``np = 1`` -> no device padding):

=================  ==================  ====================
system family      state components    state axes
=================  ==================  ====================
Cartesian          `$(u_x,u_y,u_z)$`   `$(y, k_z, k_x)$`
triply-periodic    `$(u_x,u_y,u_z)$`   `$(k_y, k_z, k_x)$`
pipe / TC          `$(u_z,u_+,u_-)$`   `$(r, m, k_z)$`
=================  ==================  ====================

True shapes are ``(ny, nz-1, nx//2)`` (Cartesian, pipe, TC) and
``(ny-1, nz-1, nx//2)`` (triply-periodic).

Algorithm
---------
- **physical**: forward-transform with ``norm="forward"`` (dnsjax
  convention) -- a full ``fft`` along every Fourier axis, then keep
  ``operators.real_harmonics`` on the real axis and
  ``operators.complex_harmonics`` on the full axes (dropping Nyquist).  A
  *full* ``fft`` then truncation (rather than ``rfft``) is required
  because `$u_\pm$` are complex; it is identity-equivalent for real
  components.  The wall-normal / radial axis is left as grid samples.
- **spectral**: the input is already native, so no transform and **no
  inverse-to-physical** is performed (an ``irfft`` would be invalid --
  the `$(u_+, u_-)$` pair is not individually Hermitian on the real axis;
  only the joint `$\hat u_+(-k)=\overline{\hat u_-(k)}$` holds).  Each
  Fourier axis is reordered to native order (dropping any Nyquist mode)
  and the field is rescaled from ``input_norm`` to dnsjax's ``"forward"``
  convention.

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
at import, the :mod:`dnsjax.random_field` idiom)::

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
        viscoelastic_systems,
    )

    if system in cartesian_systems:
        return "cartesian"
    if system in periodic_systems:
        return "periodic"
    if system in cylindrical_systems:
        return "pipe"
    if system in annular_systems:
        return "annular"
    if system in viscoelastic_systems:
        raise ValueError(
            f"system {system!r} has a 9-component tensor state; "
            "snapshot_import only supports the 3-component velocity "
            "systems (tensor import is not yet implemented)."
        )
    raise ValueError(f"unknown system: {system!r}")


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


# ── Transforms ───────────────────────────────────────────────────


def _to_native_full_axis(field: Any, axis: int, n: int) -> Any:
    """Reorder a full-complex spectral axis to native
    (``complex_harmonics``) order, dropping the Nyquist mode.

    Length ``n - 1`` is already native (pass through); length ``n`` is
    numpy-FFT order (gather, which also drops the Nyquist mode).
    """
    from jax import numpy as jnp

    m = field.shape[axis]
    if m == n - 1:
        return field
    if m == n:
        return jnp.take(field, _full_axis_gather(n), axis=axis)
    raise ValueError(
        f"full spectral axis {axis} length {m}; expected {n} or {n - 1}"
    )


def _norm_factor(input_norm: InputNorm, axis_sizes: list[int]) -> float:
    """Scalar mapping a forward transform in ``input_norm`` to dnsjax's
    ``norm="forward"`` convention, over the transformed ``axis_sizes``."""
    if input_norm == "forward":
        return 1.0
    if input_norm == "backward":
        return float(np.prod([1.0 / n for n in axis_sizes]))
    if input_norm == "ortho":
        return float(np.prod([1.0 / np.sqrt(n) for n in axis_sizes]))
    raise ValueError(
        f"input_norm must be 'backward'/'forward'/'ortho'; got {input_norm!r}"
    )


def _spectral_to_native(
    field: Any,
    input_norm: InputNorm,
    periodic: bool,
    nx: int,
    ny: int,
    nz: int,
) -> Any:
    r"""Convert a native-layout spectral input to the dnsjax state.

    The input is already in native component/axis order; only the Fourier
    mode order / Nyquist presence and the normalisation differ.  Reorders
    each Fourier axis to native order (real axis -> ``[0, nx//2)``; full
    axes -> ``complex_harmonics``, Nyquist dropped) and rescales
    ``input_norm`` -> dnsjax's ``"forward"``.  No inverse-to-physical
    transform is used: `$(u_+, u_-)$` are not individually Hermitian on
    the real axis (see the module docstring).
    """
    field = field[..., : nx // 2]  # real axis (3): keep [0, nx//2)
    field = _to_native_full_axis(field, 2, nz)  # k_z / m
    axis_sizes = [nx, nz]
    if periodic:
        field = _to_native_full_axis(field, 1, ny)  # k_y
        axis_sizes.append(ny)
    return field * _norm_factor(input_norm, axis_sizes)


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
    periodic: bool,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Check the native input field shape against the resolution."""
    if field.ndim != 4 or field.shape[0] != 3:
        raise ValueError(
            f"field must have shape (3, axis1, axis2, axis3); got "
            f"{field.shape}"
        )
    if space == "physical":
        if tuple(field.shape[1:]) != (ny, nz, nx):
            raise ValueError(
                f"physical field shape {tuple(field.shape[1:])} != native "
                f"(ny, nz, nx) = ({ny}, {nz}, {nx})"
            )
        return
    if space != "spectral":
        raise ValueError(
            f"space must be 'physical' or 'spectral'; got {space!r}"
        )

    a1, a2, a3 = field.shape[1], field.shape[2], field.shape[3]
    if periodic:
        if a1 not in (ny - 1, ny):
            raise ValueError(
                f"spectral k_y axis length {a1}; expected {ny - 1} or {ny}"
            )
    elif a1 != ny:
        raise ValueError(f"wall-normal axis length {a1} != ny {ny}")
    if a2 not in (nz - 1, nz):
        raise ValueError(
            f"spectral k_z axis length {a2}; expected {nz - 1} or {nz}"
        )
    if a3 not in (nx // 2, nx // 2 + 1):
        raise ValueError(
            f"spectral real axis length {a3}; expected {nx // 2} or "
            f"{nx // 2 + 1}"
        )


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
    input_norm: InputNorm = "backward",
) -> Any:
    r"""Pack a native-layout velocity field into dnsjax's spectral state.

    Requires ``configure_target`` to have been called.  ``field`` is in
    dnsjax's native component/axis order at the native (unpadded)
    resolution -- physical ``(3, ny, nz, nx)`` or spectral with the
    Fourier axes transformed (see the module docstring).

    Parameters
    ----------
    space:
        ``"physical"`` or ``"spectral"`` (native layout, no 3/2 padding).
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

    periodic = _geo_family(params.phys.system) == "periodic"
    nx, ny, nz = params.res.nx, params.res.ny, params.res.nz

    field = jnp.asarray(field)
    _validate_input_shape(field, space, periodic, nx, ny, nz)

    if space == "physical":
        state = _forward_to_spectral(field, periodic, nx, ny, nz)
    else:
        state = _spectral_to_native(field, input_norm, periodic, nx, ny, nz)
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
    input_norm: InputNorm = "backward",
    t: float = 0.0,
    it: int = 0,
    **config: Any,
) -> None:
    """One-shot conversion: configure, pack, and write a snapshot.

    ``field`` is in dnsjax native layout (see the module docstring).
    ``**config`` is forwarded to ``configure_target`` (``lx``, ``lz``,
    ``wall_normal_grid``, ``re``, ``re1``, ``re2``, ``eta``,
    ``tilt_degree``, ``driving``, ``double_precision``, ``fd_order``,
    ``extra_params``, ...).
    """
    configure_target(system, nx, ny, nz, **config)
    state = to_spectral_state(field, space=space, input_norm=input_norm)
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
