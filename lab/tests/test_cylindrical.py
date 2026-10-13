r"""The forcing kick's carried slots, on the pipe geometry.

The lab's part of ``tests/test_cylindrical.py`` (whose case 8b checks
``column_differences`` on columns it builds itself): a ``[force]`` kick
adds velocity columns to a pipe state, so it must add their share of
the two carried spin-quad differences as well, which the forcer builds
column by column (``StochasticForcer._carried_kick``:
``column_differences`` over ``ModeColumns`` holding each column's `$m$`
and `$k_z$`).  The map is linear, so per column it must equal
``kinematic_differences`` of the whole field at that mode -- a check on
the forcer's own wavenumber arithmetic.

Run as a script via ``uv run python lab/tests/test_cylindrical.py``.
"""

from __future__ import annotations

import sys

from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
from dnsjax.parameters import (
    Parameters,
    padded_res,
    params,
    update_parameters,
)

sys.stdout.reconfigure(line_buffering=True)

configure_jax_platform(platform_from_argv())

update_parameters(
    Parameters(
        phys={"system": "pipe"},
        res={
            "nx": 4,
            "ny": 16,
            "nz": 4,
            "fd_order": 4,
            "double_precision": True,
        },
    )
)
padded_res.set_padded_resolution(params)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.flows.wall_bounded.pipe import flow as pipe_flow  # noqa: E402
from dnsjax.geometries.wall_bounded.cylindrical import fourier  # noqa: E402


def test_kick_carried_contribution_matches_the_field_map() -> None:
    r"""A kick moves the carried differences as the field map does.

    A ``[force]`` kick adds velocity columns to a pipe state, so it must
    add their share of the two carried spin-quad differences as well,
    which the forcer builds column by column
    (``StochasticForcer._carried_kick``: ``column_differences`` over
    ``ModeColumns`` holding each column's `$m$` and `$k_z$`).  The map
    is linear, so per column it must equal
    :func:`kinematic_differences` of the whole field at that mode --
    a check on the forcer's own wavenumber arithmetic too.
    """
    from dnsjax_lab.extensions.forcing import StochasticForcer

    from dnsjax.geometries.wall_bounded._cylindrical_stepping import (
        kinematic_differences,
    )

    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    Nr = params.res.ny
    rng = np.random.default_rng(13)
    field = jnp.asarray(
        rng.standard_normal((3, Nr, Nm, Nkz))
        + 1j * rng.standard_normal((3, Nr, Nm, Nkz))
    )
    whole = np.asarray(kinematic_differences(field, fourier, pipe_flow))
    pairs = [(1, 0), (2, 1), (0, 1), (Nm - 1, 1)]
    carried_fn, carry_args = StochasticForcer._carried_kick(None, pairs, True)
    cols = jnp.stack([field[:, :, i2, i3] for i2, i3 in pairs])
    got = np.asarray(carried_fn(cols, *carry_args))
    for k, (i2, i3) in enumerate(pairs):
        assert_allclose(
            got[k],
            whole[:, :, i2, i3],
            rtol=1e-13,
            atol=1e-13 * float(np.abs(whole).max()),
            err_msg=f"column ({i2}, {i3})",
        )


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
