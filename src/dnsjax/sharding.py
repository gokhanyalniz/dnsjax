"""JAX multi-device mesh setup, precision types, and partition specs.

Initialised at import time from the global ``params``.  The singleton
``sharding`` exposes the device mesh, data-type choices, partition specs
for spectral/physical arrays, and convenience helpers (``print``, ``exit``).

Array layout convention
-----------------------
Spectral arrays have shape ``(ny[-1], nz-1, nx//2)`` -- the last axis
(kx, real-FFT, keeping non-negative modes only) is sharded across devices.
Physical arrays have shape ``(ny_padded, nz_padded, nx_padded)`` for
the triply-periodic geometry and ``(ny, nz_padded, nx_padded)`` otherwise.
The second-to-last axis (z) is sharded.  The reshard between these layouts
is handled in :mod:`dnsjax.fft`.
"""

import dataclasses
import sys
from dataclasses import dataclass
from typing import Any

import jax
from jax import numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

from .parameters import padded_res, params, periodic_systems


def register_dataclass_pytree(cls):
    def _tree_flatten(obj):
        children, aux_data = [], {}
        for f in dataclasses.fields(cls):
            val = getattr(obj, f.name)
            if (
                isinstance(val, (str, type(None)))
                or getattr(val, "__call__", None) is not None
                and not isinstance(val, (jax.Array, jnp.ndarray))
            ):
                aux_data[f.name] = val
            else:
                children.append(val)
                aux_data[f.name] = True
        return (tuple(children), aux_data)

    def _tree_unflatten(aux_data, children):
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
    """Device mesh, precision, partition specs, and array shapes.

    All class-level attributes are computed eagerly at dataclass
    definition time (not in ``__init__``), so this acts as a
    module-level singleton once ``sharding = Sharding()`` is executed.
    """

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
        f"Working with {n_devices} {params.dist.platform} devices:",
        *devices,
        flush=True,
    )

    axis_names: tuple[str, ...] = ("gpus",)
    mesh = jax.make_mesh(
        (params.dist.np,),
        axis_names=axis_names,
        axis_types=(AxisType.Explicit,),
    )

    jax.set_mesh(mesh)

    # Partition specs -- last axis of spectral arrays and second-to-last
    # axis of physical arrays are distributed across devices.
    spec_vector_shard = P(None, None, None, *axis_names)
    spec_scalar_shard = P(None, None, *axis_names)

    phys_vector_shard = P(None, None, *axis_names, None)
    phys_scalar_shard = P(None, *axis_names, None)

    no_shard = P(None)

    if params.res.double_precision:
        float_type = jnp.float64
        complex_type = jnp.complex128
    else:
        float_type = jnp.float32
        complex_type = jnp.complex64

    if params.phys.system in periodic_systems:
        # All three directions are Fourier-expanded; the Nyquist mode is
        # omitted in y and z, giving ny-1 and nz-1 stored modes.
        # The (ky, kz, kx) = (0, 0, 0) Fourier mode is the mean mode.
        vector_mean_mode: tuple[slice, ...] = tuple(
            [slice(None)] + [slice(0, 1)] * 3
        )
        scalar_mean_mode: tuple[slice, ...] = tuple([slice(0, 1)] * 3)

        spec_shape: tuple[int, ...] = (
            params.res.ny - 1,
            params.res.nz - 1,
            params.res.nx // 2,
        )

        phys_shape: tuple[int, ...] = (
            padded_res.ny_padded,
            padded_res.nz_padded,
            padded_res.nx_padded,
        )
    else:
        # Wall-bounded: y is in physical (grid-point) space, only x and z
        # are Fourier-expanded.
        # The (kz, kx) = (0, 0) Fourier mode is the mean mode.
        vector_mean_mode: tuple[slice, ...] = tuple(
            [slice(None)] * 2 + [slice(0, 1)] * 2
        )
        scalar_mean_mode: tuple[slice, ...] = tuple(
            [slice(None)] + [slice(0, 1)] * 2
        )

        spec_shape = (
            params.res.ny,
            params.res.nz - 1,
            params.res.nx // 2,
        )

        phys_shape = (
            params.res.ny,
            padded_res.nz_padded,
            padded_res.nx_padded,
        )

    def exit(self, code: int = 1) -> None:
        """Terminate all processes."""
        sys.exit(code)

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only on the main device (process index 0)."""
        if self.main_device:
            print(*args, **kwargs, flush=True)


sharding: Sharding = Sharding()
