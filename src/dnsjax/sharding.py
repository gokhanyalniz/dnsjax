r"""JAX multi-device mesh setup, precision types, and partition specs.

Initialised at import time from the global ``params``.  The singleton
``sharding`` exposes the device mesh, data-type choices, partition specs
for spectral/physical arrays, and convenience helpers (``print``, ``exit``).

Double parallelisation
----------------------
The device mesh has shape ``(np0, np1)`` with axes ``"np0"`` and
``"np1"``.

- ``np0`` distributes the wall-normal axis (`$y$` / `$r$`) in
  physical space and the spanwise-wavenumber axis (`$k_z$` / `$m$`)
  in spectral space.
- ``np1`` distributes the spanwise axis (`$z$`) in physical space
  and the streamwise-wavenumber axis (`$k_x$`) in spectral space.
- Each device holds the full wall-normal extent in spectral space,
  so FD / SPIKE solves are unchanged.

When ``np0 == 1`` the ``"np0"`` axis is trivially size-1 and all
partition specs collapse to the original 1D decomposition on
`$k_x$` / `$z$`.

Array layout convention
-----------------------
Physical arrays have shape

- ``(ny_padded, nz_padded, nx_padded)`` for triply-periodic,
- ``(ny, nz_padded, nx_padded)`` for wall-bounded,

with axes ``[y, z, x]``.  ``y`` is sharded by ``np0``, ``z`` by
``np1``, ``x`` is local.

Spectral arrays have shape

- ``(ny-1, nz_spec, nx_spec)`` ``[ky, kz, kx]`` for triply-periodic,
- ``(nz_spec, nx_spec, ny)`` ``[kz, kx, y]`` for wall-bounded.

``kz`` is sharded by ``np0``, ``kx`` by ``np1``, and ``y`` / ``ky``
are local.  ``nz_spec`` and ``nx_spec`` may exceed the true mode
counts (``nz - 1`` and ``nx // 2``) by up to ``np0 - 1`` or
``np1 - 1`` zero-padded dummy modes; see :mod:`dnsjax.fft`.
"""

import dataclasses
import sys
from dataclasses import dataclass

import jax
from jax import numpy as jnp
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from .parameters import padded_res, params, periodic_systems


def _pad_to_multiple(n: int, divisor: int) -> int:
    """Round *n* up to the next multiple of *divisor*."""
    if divisor <= 1:
        return n
    return ((n + divisor - 1) // divisor) * divisor


def register_dataclass_pytree[T](cls: type[T]) -> type[T]:
    def _tree_flatten(obj: T) -> tuple[tuple[object, ...], dict[str, object]]:
        children: list[object] = []
        aux_data: dict[str, object] = {}
        for f in dataclasses.fields(cls):
            val = getattr(obj, f.name)
            if (
                isinstance(val, (str, type(None)))
                or callable(val)
                and not isinstance(val, (jax.Array, jnp.ndarray))
            ):
                aux_data[f.name] = val
            else:
                children.append(val)
                aux_data[f.name] = True
        return (tuple(children), aux_data)

    def _tree_unflatten(
        aux_data: dict[str, object], children: tuple[object, ...]
    ) -> T:
        obj = object.__new__(cls)
        child_idx = 0
        for f in dataclasses.fields(cls):
            val_or_flag = aux_data.get(f.name)
            if val_or_flag is True:
                setattr(obj, f.name, children[child_idx])
                child_idx += 1
            else:
                setattr(obj, f.name, val_or_flag)
        return obj

    jax.tree_util.register_pytree_node(cls, _tree_flatten, _tree_unflatten)
    return cls


@dataclass
class Sharding:
    r"""Device mesh, precision, partition specs, and array shapes.

    All class-level attributes are computed eagerly at dataclass
    definition time, so this acts as a module-level singleton
    once ``sharding = Sharding()`` is executed.

    Spectral padding
    ~~~~~~~~~~~~~~~~
    When the spectral mode count (``nz - 1`` for `$k_z$` or
    ``nx // 2`` for `$k_x$`) is not evenly divisible by the
    corresponding mesh axis (``np0`` or ``np1``), the dimension
    is padded to the next multiple with zero (physics-neutral)
    dummy modes.  The padding amount is stored in ``nz_spec_pad``
    and ``nx_spec_pad``; the total stored mode count in
    ``nz_spec`` and ``nx_spec``.  Padding and stripping are
    handled inside :mod:`dnsjax.fft`.

    Physical dimensions (``ny``, ``ny_padded``, ``nz_padded``)
    cannot be padded (it would change the FFT size and
    normalisation), so divisibility by the mesh axis is a hard
    requirement validated at startup.
    """

    np0: int = params.dist.np0
    np1: int = params.dist.np1
    n_devices: int = params.dist.np
    main_device: bool = bool(jax.process_index() == 0)

    devices = jax.devices()
    n_devices_reported: int = len(devices)
    if n_devices_reported != n_devices:
        if main_device:
            print(
                f"# of devices visible ({n_devices_reported}) "
                f"is not equal to np = {n_devices}.",
                flush=True,
            )
        sys.exit(1)

    print(
        f"Working with {n_devices} {params.dist.platform} devices"
        f" (np0={np0}, np1={np1}):",
        *devices,
        flush=True,
    )

    # ── 2D device mesh ────────────────────────────────────────
    mesh = jax.make_mesh(
        (np0, np1),
        axis_names=("np0", "np1"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    jax.set_mesh(mesh)

    # Axis-name helpers: None when the axis is trivially size-1,
    # so that P(a0, ...) becomes P(None, ...) = replicated,
    # avoiding any size-1 collective overhead.
    a0: str | None = "np0" if np0 > 1 else None
    a1: str | None = "np1" if np1 > 1 else None

    # ── Hard divisibility constraints on physical dims ─────────
    _is_periodic: bool = params.phys.system in periodic_systems

    if not _is_periodic and np0 > 1 and params.res.ny % np0 != 0:
        print(
            f"Wall-bounded ny={params.res.ny} is not divisible by "
            f"np0={np0}. Use a tanh grid with power-of-2 ny, or "
            f"choose np0 to divide ny.",
            flush=True,
        )
        sys.exit(1)

    if _is_periodic and np0 > 1:
        _ny_padded_check: int | None = padded_res.ny_padded
        if _ny_padded_check is not None and _ny_padded_check % np0 != 0:
            print(
                f"ny_padded={_ny_padded_check} is not divisible by "
                f"np0={np0}. Choose ny and oversampling_factor so "
                f"that ny_padded = oversampling_factor * ny // 2 is "
                f"a multiple of np0.",
                flush=True,
            )
            sys.exit(1)

    if np1 > 1 and padded_res.nz_padded % np1 != 0:
        print(
            f"nz_padded={padded_res.nz_padded} is not divisible by "
            f"np1={np1}. Choose nz and oversampling_factor so that "
            f"nz_padded = oversampling_factor * nz // 2 is a "
            f"multiple of np1.",
            flush=True,
        )
        sys.exit(1)

    # ── Spectral mode counts (auto-padded for divisibility) ───
    nz_spec: int = _pad_to_multiple(params.res.nz - 1, np0)
    nx_spec: int = _pad_to_multiple(params.res.nx // 2, np1)
    nz_spec_pad: int = nz_spec - (params.res.nz - 1)
    nx_spec_pad: int = nx_spec - (params.res.nx // 2)

    if nz_spec_pad and main_device:
        print(
            f"Spectral kz padded from {params.res.nz - 1} to "
            f"{nz_spec} modes (+{nz_spec_pad} zeros) for np0 "
            f"divisibility.",
            flush=True,
        )
    if nx_spec_pad and main_device:
        print(
            f"Spectral kx padded from {params.res.nx // 2} to "
            f"{nx_spec} modes (+{nx_spec_pad} zeros) for np1 "
            f"divisibility.",
            flush=True,
        )

    # ── FFT-internal partition specs ──────────────────────────
    # Three stages of the FFT pipeline in [y, z/kz, x/kx] order:
    #   phys:  [y_np0, z_np1, x]      P(a0, a1, None)
    #   mid:   [y_np0, z,     kx_np1] P(a0, None, a1)
    #   spec:  [y,     kz_np0, kx_np1] P(None, a0, a1)
    _fft_phys_scalar_shard = P(a0, a1, None)
    _fft_mid_scalar_shard = P(a0, None, a1)
    _fft_spec_scalar_shard = P(None, a0, a1)

    # ── Spectral partition specs ──────────────────────────────
    if _is_periodic:
        # Spectral layout [ky, kz, kx]:
        # ky fully local, kz by np0, kx by np1.
        spec_vector_shard = P(None, None, a0, a1)
        spec_scalar_shard = P(None, a0, a1)
    else:
        # Spectral layout [kz, kx, y]:
        # kz by np0, kx by np1, y fully local.
        spec_vector_shard = P(None, a0, a1, None)
        spec_scalar_shard = P(a0, a1, None)

    # ── Physical partition specs ──────────────────────────────
    # [y, z, x] or [C, y, z, x]:
    # y by np0, z by np1, x fully local.
    phys_vector_shard = P(None, a0, a1, None)
    phys_scalar_shard = P(a0, a1, None)

    no_shard = P(None)

    # ── IMM partition specs (wall-bounded) ────────────────────
    # Leading spectral axes [kz, kx, ...]:
    spec_imm_corr_shard = NamedSharding(mesh, P(a0, a1, None))
    spec_dy_op_shard = NamedSharding(mesh, P(a0, a1, None, None))
    spec_dy_blocks_shard = NamedSharding(mesh, P(a0, a1, None, None, None))
    spec_k2_op_shard = NamedSharding(mesh, P(a0, a1))

    # ── Precision ─────────────────────────────────────────────
    if params.res.double_precision:
        float_type = jnp.float64
        complex_type = jnp.complex128
    else:
        float_type = jnp.float32
        complex_type = jnp.complex64

    # ── Array shapes ──────────────────────────────────────────
    if _is_periodic:
        # Spectral [ky, kz, kx] — ky is unpadded (not sharded).
        spec_shape: tuple[int, ...] = (
            params.res.ny - 1,
            nz_spec,
            nx_spec,
        )
        # Physical [y, z, x]
        phys_shape: tuple[int, ...] = (
            padded_res.ny_padded,
            padded_res.nz_padded,
            padded_res.nx_padded,
        )
        vector_mean_mode: tuple[slice, ...] = tuple(
            [slice(None)] + [slice(0, 1)] * 3
        )
        scalar_mean_mode: tuple[slice, ...] = tuple([slice(0, 1)] * 3)
    else:
        # Spectral [kz, kx, y] — y is unpadded (not sharded).
        spec_shape = (
            nz_spec,
            nx_spec,
            params.res.ny,
        )
        # Physical [y, z, x]
        phys_shape = (
            params.res.ny,
            padded_res.nz_padded,
            padded_res.nx_padded,
        )
        vector_mean_mode: tuple[slice, ...] = tuple(
            [slice(None)] + [slice(0, 1)] * 2 + [slice(None)]
        )
        scalar_mean_mode: tuple[slice, ...] = tuple(
            [slice(0, 1)] * 2 + [slice(None)]
        )

    def exit(self, code: int = 1) -> None:
        """Terminate all processes."""
        sys.exit(code)

    def print(self, *args: object, **kwargs: object) -> None:
        """Print only on the main device (process index 0)."""
        if self.main_device:
            print(*args, **kwargs, flush=True)


sharding: Sharding = Sharding()
