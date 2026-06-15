#!/usr/bin/env python3
r"""Generate a random divergence-free perturbation and save as a
zarr3 snapshot.

The output snapshot is immediately usable as initial condition::

    uv run python scripts/random_field.py \
        --system plane-couette \
        --nx 128 --ny 65 --nz 128 \
        --amplitude 0.1 --smoothness 0.4 --seed 1 \
        --output random_ic

    uv run python -m dnsjax \
        --init.snapshot random_ic \
        --init.start_from_laminar False ...

The wavenumber-dependent amplitude of each Fourier mode
`$(k_x, k_z)$` decays as

.. math::
    A(k_x, k_z) = (1 - s)^{|k_x| + |k_z|}

where `$s$` is the ``--smoothness`` parameter and `$k_x, k_z$`
are the physical wavenumbers.  The field is then normalised
so that the volume-averaged L2 norm equals ``--amplitude``.

**Non-JAX operations**: The per-mode divergence-free
enforcement (step 4) loops over Fourier modes and uses NumPy
for the `$D_1 \mathbf{v}$` matvecs, because Python-level
looping in JAX would incur tracing overhead.  All other
array work uses JAX.
"""

from __future__ import annotations

import argparse
import os
import sys
from math import pi
from pathlib import Path

import numpy as np

# ── CLI ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a random divergence-free perturbation "
            "and save as a zarr3 snapshot."
        ),
    )
    # Simulation parameters
    ap.add_argument(
        "--system",
        required=True,
        choices=[
            "plane-couette",
            "plane-poiseuille",
            "pipe",
            "taylor-couette",
            "kolmogorov",
            "waleffe",
            "decaying-box",
        ],
    )
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--lx", type=float, default=4.0)
    ap.add_argument("--lz", type=float, default=4.0)
    ap.add_argument("--re", type=float, default=1000.0)
    # Taylor-Couette control parameters.
    ap.add_argument("--re1", type=float, default=None)
    ap.add_argument("--re2", type=float, default=None)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--fd-order", type=int, default=4)
    ap.add_argument("--tilt-degree", type=float, default=0.0)
    ap.add_argument("--wall-grid", type=str, default=None)
    ap.add_argument("--single-precision", action="store_true")
    ap.add_argument(
        "--driving",
        choices=["constant_pressure_gradient", "constant_bulk_velocity"],
        default="constant_pressure_gradient",
    )
    ap.add_argument(
        "--block-mean-spanwise-velocity",
        action="store_true",
    )
    ap.add_argument(
        "--parameters-toml",
        type=str,
        default=None,
        help="Load simulation parameters from a TOML file.",
    )

    # Random-field parameters
    ap.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="Target L2 norm of the perturbation.",
    )
    ap.add_argument(
        "--smoothness",
        type=float,
        default=0.4,
        help="Spectral decay rate (0 < s < 1). Higher = smoother.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (NumPy PRNG, device-count independent).",
    )
    ap.add_argument(
        "--mean-flow",
        action="store_true",
        help="Also perturb the mean mode (kx=kz=0).",
    )

    # Output
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the zarr3 snapshot.",
    )

    # Self-test
    ap.add_argument(
        "--test",
        action="store_true",
        help="Run self-contained verification tests and exit.",
    )
    return ap.parse_args()


# ── JAX + singleton setup ────────────────────────────────────────


def _setup_jax_and_params(args: argparse.Namespace) -> None:
    """Configure JAX for CPU and set the global parameter
    singletons exactly as ``__main__.py`` does."""
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    os.environ["NPROC"] = "1"

    import jax

    double = not args.single_precision
    jax.config.update("jax_enable_x64", double)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        read_parameters,
        update_parameters,
    )

    if args.parameters_toml is not None:
        params_in = read_parameters(Path(args.parameters_toml))
        update_parameters(params_in)

    cli_params = Parameters(
        dist={"np": 1, "platform": "cpu"},
        phys={
            "system": args.system,
            "re": args.re,
            "re1": args.re1,
            "re2": args.re2,
            "driving": args.driving,
            "block_mean_spanwise_velocity": (
                args.block_mean_spanwise_velocity
            ),
        },
        geo={
            "lx": args.lx,
            "lz": args.lz,
            "tilt_degree": args.tilt_degree,
            "eta": args.eta,
            "wall_grid": args.wall_grid,
        },
        res={
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "fd_order": args.fd_order,
            "double_precision": double,
        },
        outs={},
    )
    update_parameters(cli_params)
    padded_res.set_padded_resolution(params)


# ── Hermitian-symmetry enforcement ───────────────────────────────

# The real-FFT axis (kx for Cartesian/periodic, kz for cylindrical)
# stores only non-negative wavenumbers.  On the complex-FFT axis
# at kx=0 (or kz=0 for cylindrical), the stored modes must satisfy
# conjugate symmetry for the physical field to be real.  The helper
# below is pure NumPy (no JAX) since it works on the host array.


def _enforce_hermitian_slice(
    slc: np.ndarray,
    n_physical: int,
) -> None:
    """Enforce conjugate symmetry in-place on a 1-D slice.

    ``slc`` has leading shape ``(Nc, ...)`` where
    ``Nc = n_physical - 1`` (Nyquist omitted), indexed by
    ``complex_harmonics(n_physical)``:
    ``[0, 1, ..., n//2-1, -n//2+1, ..., -1]``.

    Parameters
    ----------
    slc:
        The complex-FFT axis slice to fix, with the
        complex-FFT axis as axis 0.
    n_physical:
        Physical-space size of this direction.
    """
    n_pos = n_physical // 2
    Nc = n_physical - 1

    # Index 0 (wavenumber 0) must be real.
    slc[0] = slc[0].real

    # Pair positive kz at index i with negative kz at Nc-i.
    for i in range(1, n_pos):
        slc[Nc - i] = np.conj(slc[i])

    # Odd n: unpaired negative mode (Nyquist partner omitted).
    if n_physical % 2 == 1:
        slc[n_pos] = 0.0


# ── Cartesian wall-bounded generation ────────────────────────────


def _generate_cartesian(args: argparse.Namespace):
    """Generate a random divergence-free Cartesian perturbation.

    Returns a JAX array of shape ``(3, Ny, Nkz, Nkx)``.
    """
    import jax
    from jax import numpy as jnp

    from dnsjax.geometries.wall_bounded._base import get_norm
    from dnsjax.geometries.wall_bounded.cartesian import (
        build_cartesian_grid,
        fourier,
    )
    from dnsjax.parameters import derived_params, params
    from dnsjax.sharding import sharding

    ny = params.res.ny
    Nkz = params.res.nz - 1
    Nkx = params.res.nx // 2

    ys, D1, _, y_weights = build_cartesian_grid(
        ny, params.res.fd_order, params.geo.wall_grid
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(ys)]

    # ── NumPy: per-mode construction (non-JAX) ──────────────
    ys_np = np.asarray(ys)
    D1_np = np.asarray(D1)
    kx_np = np.asarray(fourier.kx).ravel()  # (Nkx,)
    kz_np = np.asarray(fourier.kz).ravel()  # (Nkz,)

    decay = 1.0 - args.smoothness
    window_noslip = 1.0 - ys_np**2

    # Wavenumber decay factors: shape (Nkz, Nkx).
    mode_decay = decay ** (np.abs(kz_np[:, None]) + np.abs(kx_np[None, :]))

    rng = np.random.default_rng(args.seed)
    shape = (3, ny, Nkz, Nkx)
    raw = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    # Apply no-slip window to all components: f(+/-1) = 0.
    raw *= window_noslip[np.newaxis, :, np.newaxis, np.newaxis]

    # Apply wavenumber decay.
    raw *= mode_decay[np.newaxis, np.newaxis, :, :]

    # Fix v so D1@v is exactly zero at both walls.  This is
    # necessary because the derived component (w or u) gets
    # its wall value from D1@v via the continuity equation.
    # Adjust the two near-wall interior points v[1] and
    # v[N-2] to cancel the D1@v residual at the boundaries.
    A_fix = np.array(
        [
            [D1_np[0, 1], D1_np[0, -2]],
            [D1_np[-1, 1], D1_np[-1, -2]],
        ]
    )
    A_fix_inv = np.linalg.inv(A_fix)
    for iz in range(Nkz):
        for ix in range(Nkx):
            v_mode = raw[1, :, iz, ix]
            dv_wall = D1_np[[0, -1], :] @ v_mode  # (2,)
            delta = -A_fix_inv @ dv_wall
            v_mode[1] += delta[0]
            v_mode[-2] += delta[1]

    # Enforce divergence-free per mode (NumPy loop).
    # D1@v is now exactly zero at the walls, so the derived
    # component inherits exact zero wall BCs from u(+/-1)=0.
    for iz in range(Nkz):
        kz_val = kz_np[iz]
        for ix in range(Nkx):
            kx_val = kx_np[ix]
            if kx_val == 0 and kz_val == 0:
                raw[1, :, iz, ix] = 0.0
            elif kz_val != 0:
                dv_dy = D1_np @ raw[1, :, iz, ix]
                raw[2, :, iz, ix] = -(
                    1j * kx_val * raw[0, :, iz, ix] + dv_dy
                ) / (1j * kz_val)
            else:
                dv_dy = D1_np @ raw[1, :, iz, ix]
                raw[0, :, iz, ix] = -dv_dy / (1j * kx_val)

    # Hermitian symmetry at kx=0: fix kz axis for each component.
    # raw shape: (3, Ny, Nkz, Nkx); kx=0 is index 3.
    for c in range(3):
        _enforce_hermitian_slice(raw[c, :, :, 0].T, params.res.nz)

    # Zero mean mode unless --mean-flow.
    if not args.mean_flow:
        raw[:, :, 0, 0] = 0.0

    # ── JAX: normalise and return ────────────────────────────
    state = jax.device_put(
        jnp.asarray(raw, dtype=sharding.complex_type),
        sharding.spec_vector_shard,
    )
    norm = get_norm(state, fourier.k_metric, y_weights)
    state = state * (args.amplitude / norm)
    return state, ys, D1_np, y_weights, fourier


# ── Cylindrical generation ───────────────────────────────────────


def _generate_cylindrical(args: argparse.Namespace):
    r"""Generate a random perturbation for pipe flow.

    Returns a JAX array of shape ``(3, Nr, Nm, Nkz)`` in
    `$(u_z, u_+, u_-)$` form.

    For `$k_z \neq 0$` modes the field is divergence-free by
    construction.  For `$k_z = 0$` modes a small residual
    divergence may remain; the first corrector step of the
    solver projects it out via the IMM.
    """
    import jax
    from jax import numpy as jnp

    from dnsjax.geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
        fourier,
        get_norm2_cyl,
    )
    from dnsjax.parameters import derived_params, params
    from dnsjax.sharding import sharding

    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2

    rs, D1_even, D1_odd, D1_pos, y_weights, inv_r = build_cylindrical_grid(
        Nr, params.res.fd_order, params.geo.wall_grid
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    # ── NumPy: per-mode construction (non-JAX) ──────────────
    rs_np = np.asarray(rs)
    inv_r_np = np.asarray(inv_r)
    D1_even_np = np.asarray(D1_even)
    D1_odd_np = np.asarray(D1_odd)
    kz_np = np.asarray(fourier.kz).ravel()  # (Nkz,)
    m_np = np.asarray(fourier.m).ravel()  # (Nm,)

    decay = 1.0 - args.smoothness
    window_wall = 1.0 - rs_np  # f(1) = 0

    # Wavenumber decay: |kz_phys| + |m|*2*pi/lz.
    kz_abs = np.abs(kz_np)
    m_abs = np.abs(m_np) * 2 * pi / params.geo.lz
    mode_decay = decay ** (kz_abs[None, :] + m_abs[:, None])

    rng = np.random.default_rng(args.seed)
    shape = (3, Nr, Nm, Nkz)
    raw = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    # Apply wall window (r=1).
    raw *= window_wall[np.newaxis, :, np.newaxis, np.newaxis]

    # Apply parity windows near r=0 and wavenumber decay.
    for im in range(Nm):
        m_val = int(m_np[im])
        # u_z parity: (-1)^m -> odd when m odd
        if m_val % 2 != 0:
            raw[0, :, im, :] *= rs_np[:, np.newaxis]
        # u+, u- parity: (-1)^{m+1} -> odd when m even
        if m_val % 2 == 0:
            raw[1, :, im, :] *= rs_np[:, np.newaxis]
            raw[2, :, im, :] *= rs_np[:, np.newaxis]
        raw[:, :, im, :] *= mode_decay[im, :][np.newaxis, np.newaxis, :]

    # Enforce divergence-free for kz != 0 (NumPy loop).
    # Continuity: i*kz*uz + [D1(u+) + (m+1)*u+/r
    #                       + D1(u-) + (1-m)*u-/r] / 2 = 0
    for im in range(Nm):
        m_val = int(m_np[im])
        # u+, u-: parity (-1)^{m+1}
        D1_pm = D1_even_np if (m_val + 1) % 2 == 0 else D1_odd_np

        for ik in range(Nkz):
            kz_val = kz_np[ik]
            if kz_val == 0:
                continue
            u_plus = raw[1, :, im, ik]
            u_minus = raw[2, :, im, ik]
            div_radial = (
                D1_pm @ u_plus
                + (m_val + 1) * inv_r_np * u_plus
                + D1_pm @ u_minus
                + (1 - m_val) * inv_r_np * u_minus
            ) / 2.0
            raw[0, :, im, ik] = -div_radial / (1j * kz_val)

    # Zero mean mode unless --mean-flow.
    if not args.mean_flow:
        raw[:, :, 0, 0] = 0.0

    # Hermitian symmetry at kz=0 (real-FFT axis):
    # fix m axis for each component.
    # raw shape: (3, Nr, Nm, Nkz); kz=0 is index 3.
    for c in range(3):
        _enforce_hermitian_slice(raw[c, :, :, 0].T, params.res.nz)

    # ── JAX: normalise and return ────────────────────────────
    state = jax.device_put(
        jnp.asarray(raw, dtype=sharding.complex_type),
        sharding.spec_vector_shard,
    )

    norm2 = get_norm2_cyl(state, fourier.k_metric, y_weights)
    norm = jnp.sqrt(norm2)
    state = state * (args.amplitude / norm)
    return state


# ── Annular generation ───────────────────────────────────────────


def _generate_annular(args: argparse.Namespace):
    r"""Generate a random perturbation for Taylor-Couette flow.

    Returns a JAX array of shape ``(3, Nr, Nm, Nkz)`` in
    `$(u_z, u_+, u_-)$` form, with no-slip at both walls.

    For `$k_z \neq 0$` modes the field is divergence-free by
    construction (`$u_z$` derived from continuity, with the near-wall
    `$u_\pm$` points adjusted so `$D_1 u_r = 0$` at both walls, so the
    derived `$u_z$` inherits exact zero wall values).  For `$k_z = 0$`
    modes a small residual divergence may remain; the first corrector
    step of the solver projects it out via the IMM.
    """
    import jax
    from jax import numpy as jnp

    from dnsjax.geometries.wall_bounded.annular import (
        build_annular_grid,
        fourier,
        get_norm2_annular,
    )
    from dnsjax.parameters import derived_params, params
    from dnsjax.sharding import sharding

    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2

    r1 = derived_params.r_inner
    r2 = derived_params.r_outer
    rs, D1, _, y_weights, inv_r = build_annular_grid(
        Nr, params.res.fd_order, r1, r2, params.geo.wall_grid
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    # ── NumPy: per-mode construction (non-JAX) ──────────────
    rs_np = np.asarray(rs)
    inv_r_np = np.asarray(inv_r)
    D1_np = np.asarray(D1)
    kz_np = np.asarray(fourier.kz).ravel()  # (Nkz,)
    m_np = np.asarray(fourier.m).ravel()  # (Nm,)

    decay = 1.0 - args.smoothness
    # No-slip window: zero at both walls, peak in the interior.
    window = (rs_np - r1) * (r2 - rs_np)
    window = window / np.max(window)

    kz_abs = np.abs(kz_np)
    m_abs = np.abs(m_np) * 2 * pi / params.geo.lz
    mode_decay = decay ** (kz_abs[None, :] + m_abs[:, None])

    rng = np.random.default_rng(args.seed)
    shape = (3, Nr, Nm, Nkz)
    raw = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    raw *= window[np.newaxis, :, np.newaxis, np.newaxis]
    for im in range(Nm):
        raw[:, :, im, :] *= mode_decay[im, :][np.newaxis, np.newaxis, :]

    # Adjust the two near-wall interior points of u_+ and u_- so that
    # D1 @ u_pm = 0 at both walls; then D1 @ u_r = 0 there, and the
    # u_z derived from continuity inherits exact zero wall values.
    A_fix = np.array(
        [
            [D1_np[0, 1], D1_np[0, -2]],
            [D1_np[-1, 1], D1_np[-1, -2]],
        ]
    )
    A_fix_inv = np.linalg.inv(A_fix)

    # Enforce divergence-free for kz != 0 (NumPy loop).
    # Continuity: i*kz*uz + [D1(u+) + (m+1)*u+/r
    #                       + D1(u-) + (1-m)*u-/r] / 2 = 0
    for im in range(Nm):
        m_val = int(m_np[im])
        for ik in range(Nkz):
            for comp in (1, 2):  # u_+, u_-
                f = raw[comp, :, im, ik]
                d_wall = D1_np[[0, -1], :] @ f  # (2,)
                delta = -A_fix_inv @ d_wall
                f[1] += delta[0]
                f[-2] += delta[1]

            kz_val = kz_np[ik]
            if kz_val == 0:
                continue
            u_plus = raw[1, :, im, ik]
            u_minus = raw[2, :, im, ik]
            div_radial = (
                D1_np @ u_plus
                + (m_val + 1) * inv_r_np * u_plus
                + D1_np @ u_minus
                + (1 - m_val) * inv_r_np * u_minus
            ) / 2.0
            raw[0, :, im, ik] = -div_radial / (1j * kz_val)

    # Zero mean mode unless --mean-flow.
    if not args.mean_flow:
        raw[:, :, 0, 0] = 0.0

    # Hermitian symmetry at kz=0 (real-FFT axis): fix m axis.
    for c in range(3):
        _enforce_hermitian_slice(raw[c, :, :, 0].T, params.res.nz)

    # ── JAX: normalise and return ────────────────────────────
    state = jax.device_put(
        jnp.asarray(raw, dtype=sharding.complex_type),
        sharding.spec_vector_shard,
    )

    norm2 = get_norm2_annular(state, fourier.k_metric, y_weights)
    norm = jnp.sqrt(norm2)
    state = state * (args.amplitude / norm)
    return state


# ── Triply-periodic generation ───────────────────────────────────


def _generate_triply_periodic(args: argparse.Namespace):
    """Generate a random divergence-free periodic perturbation.

    Returns a JAX array of shape ``(3, Nky, Nkz, Nkx)``.
    Uses the Leray projection to enforce incompressibility.
    """
    import jax
    from jax import numpy as jnp

    from dnsjax.geometries.triply_periodic.triply_periodic import (
        fourier,
        get_norm,
    )
    from dnsjax.parameters import params
    from dnsjax.sharding import sharding

    Nky = params.res.ny - 1
    Nkz = params.res.nz - 1
    Nkx = params.res.nx // 2

    # ── NumPy: generate and project (non-JAX) ───────────────
    kx_np = np.asarray(fourier.kx).ravel()  # (Nkx,)
    kz_np = np.asarray(fourier.kz).ravel()  # (Nkz,)
    ky_np = np.asarray(fourier.ky).ravel()  # (Nky,)

    decay = 1.0 - args.smoothness

    # 3D L1-norm decay.
    mode_decay = decay ** (
        np.abs(ky_np[:, None, None])
        + np.abs(kz_np[None, :, None])
        + np.abs(kx_np[None, None, :])
    )

    rng = np.random.default_rng(args.seed)
    shape = (3, Nky, Nkz, Nkx)
    raw = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    raw *= mode_decay[np.newaxis]

    # Leray projection: u_proj = u - k (k . u) / |k|^2.
    kx_3d = kx_np[np.newaxis, np.newaxis, :]
    ky_3d = ky_np[:, np.newaxis, np.newaxis]
    kz_3d = kz_np[np.newaxis, :, np.newaxis]
    k2 = kx_3d**2 + ky_3d**2 + kz_3d**2
    k2_safe = np.where(k2 > 0, k2, 1.0)
    k_dot_u = kx_3d * raw[0] + ky_3d * raw[1] + kz_3d * raw[2]
    proj = k_dot_u / k2_safe
    raw[0] -= kx_3d * proj
    raw[1] -= ky_3d * proj
    raw[2] -= kz_3d * proj

    # Zero mean mode.
    if not args.mean_flow:
        raw[:, 0, 0, 0] = 0.0

    # Hermitian symmetry at kx=0: joint 2D constraint
    # f_hat(ky, kz, 0) = conj(f_hat(-ky, -kz, 0)).
    from dnsjax.operators import complex_harmonics as _ch

    ky_idx = np.asarray(_ch(params.res.ny))
    kz_idx = np.asarray(_ch(params.res.nz))
    ky_map = {int(k): i for i, k in enumerate(ky_idx)}
    kz_map = {int(k): i for i, k in enumerate(kz_idx)}
    visited = set()
    for iy in range(Nky):
        for iz in range(Nkz):
            if (iy, iz) in visited:
                continue
            ky_v, kz_v = int(ky_idx[iy]), int(kz_idx[iz])
            jy = ky_map.get(-ky_v)
            jz = kz_map.get(-kz_v)
            if jy is None or jz is None:
                raw[:, iy, iz, 0] = 0.0
                visited.add((iy, iz))
                continue
            if (iy, iz) == (jy, jz):
                raw[:, iy, iz, 0] = raw[:, iy, iz, 0].real
            else:
                raw[:, jy, jz, 0] = np.conj(raw[:, iy, iz, 0])
            visited.add((iy, iz))
            visited.add((jy, jz))

    # ── JAX: normalise and return ────────────────────────────
    state = jax.device_put(
        jnp.asarray(raw, dtype=sharding.complex_type),
        sharding.spec_vector_shard,
    )
    norm = get_norm(state, fourier.k_metric)
    state = state * (args.amplitude / norm)
    return state


# ── Self-test ────────────────────────────────────────────────────


def _run_tests() -> None:
    """Run self-contained verification for all geometries."""
    from jax import numpy as jnp

    from dnsjax.parameters import (
        annular_systems,
        cartesian_systems,
        params,
    )

    passed = 0
    failed = 0

    def _check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}  {detail}")

    # The generation singletons (Fourier / sharding) are built from
    # ``params`` at import time, so the self-test runs the block for the
    # configured ``--system`` rather than switching systems mid-process.
    system = params.phys.system

    if system in cartesian_systems:
        print(f"Cartesian ({system}):")
        state, ys, D1_np, y_weights, fourier = _generate_cartesian(args)

        Nkz = params.res.nz - 1
        Nkx = params.res.nx // 2
        state_np = np.asarray(state)
        kx_np = np.asarray(fourier.kx).ravel()
        kz_np = np.asarray(fourier.kz).ravel()

        # Divergence-free.
        max_div = 0.0
        for iz in range(Nkz):
            for ix in range(Nkx):
                kx_v = kx_np[ix]
                kz_v = kz_np[iz]
                dv_dy = D1_np @ state_np[1, :, iz, ix]
                div = (
                    1j * kx_v * state_np[0, :, iz, ix]
                    + dv_dy
                    + 1j * kz_v * state_np[2, :, iz, ix]
                )
                max_div = max(max_div, float(np.max(np.abs(div))))
        _check(
            "divergence-free", max_div < 1e-10, f"max |div| = {max_div:.2e}"
        )

        # Wall BCs.
        bc_err = max(
            float(np.max(np.abs(state_np[:, 0]))),
            float(np.max(np.abs(state_np[:, -1]))),
        )
        _check("wall BCs", bc_err < 1e-14, f"max |BC| = {bc_err:.2e}")

        # Norm.
        from dnsjax.geometries.wall_bounded._base import get_norm

        norm = float(get_norm(state, fourier.k_metric, y_weights))
        norm_err = abs(norm - args.amplitude) / args.amplitude
        _check(
            "norm matches target",
            norm_err < 1e-12,
            f"|norm - target| / target = {norm_err:.2e}",
        )

        # Mean mode zero.
        mean_err = float(np.max(np.abs(state_np[:, :, 0, 0])))
        _check(
            "mean mode zero", mean_err < 1e-30, f"max |mean| = {mean_err:.2e}"
        )

        # Hermitian symmetry at kx=0.
        n_pos = params.res.nz // 2
        sym_err = 0.0
        for i in range(1, n_pos):
            j = Nkz - i
            diff = state_np[:, :, j, 0] - np.conj(state_np[:, :, i, 0])
            sym_err = max(sym_err, float(np.max(np.abs(diff))))
        kz0_imag = float(np.max(np.abs(state_np[:, :, 0, 0].imag)))
        sym_err = max(sym_err, kz0_imag)
        _check(
            "Hermitian symmetry at kx=0",
            sym_err < 1e-14,
            f"max sym error = {sym_err:.2e}",
        )

        # Seed determinism.
        state2, *_ = _generate_cartesian(args)
        det_err = float(jnp.max(jnp.abs(state - state2)))
        _check("seed determinism", det_err == 0.0, f"max diff = {det_err:.2e}")

    elif system in annular_systems:
        print(f"Annular ({system}):")
        from dnsjax.geometries.wall_bounded.annular import (
            build_annular_grid,
            fourier,
            get_norm2_annular,
        )
        from dnsjax.parameters import derived_params

        state = _generate_annular(args)
        state_np = np.asarray(state)

        Nm = params.res.nz - 1
        Nkz = params.res.nx // 2
        rs, D1, _, y_weights, inv_r = build_annular_grid(
            params.res.ny,
            params.res.fd_order,
            derived_params.r_inner,
            derived_params.r_outer,
            params.geo.wall_grid,
        )
        D1_np = np.asarray(D1)
        inv_r_np = np.asarray(inv_r)
        kz_np = np.asarray(fourier.kz).ravel()
        m_np = np.asarray(fourier.m).ravel()

        # Divergence-free for kz != 0 (kz = 0 modes carry a residual
        # divergence projected out by the first corrector step).
        max_div = 0.0
        for im in range(Nm):
            m_val = int(m_np[im])
            for ik in range(Nkz):
                kz_v = kz_np[ik]
                if kz_v == 0:
                    continue
                up = state_np[1, :, im, ik]
                um = state_np[2, :, im, ik]
                div_r = (
                    D1_np @ up
                    + (m_val + 1) * inv_r_np * up
                    + D1_np @ um
                    + (1 - m_val) * inv_r_np * um
                ) / 2.0
                div = 1j * kz_v * state_np[0, :, im, ik] + div_r
                max_div = max(max_div, float(np.max(np.abs(div))))
        _check(
            "divergence-free (kz!=0)",
            max_div < 1e-10,
            f"max |div| = {max_div:.2e}",
        )

        # Wall BCs (both walls).
        bc_err = max(
            float(np.max(np.abs(state_np[:, 0]))),
            float(np.max(np.abs(state_np[:, -1]))),
        )
        _check("wall BCs", bc_err < 1e-13, f"max |BC| = {bc_err:.2e}")

        # Norm.
        norm = float(
            jnp.sqrt(get_norm2_annular(state, fourier.k_metric, y_weights))
        )
        norm_err = abs(norm - args.amplitude) / args.amplitude
        _check(
            "norm matches target",
            norm_err < 1e-12,
            f"|norm - target| / target = {norm_err:.2e}",
        )

        # Mean mode zero.
        mean_err = float(np.max(np.abs(state_np[:, :, 0, 0])))
        _check(
            "mean mode zero", mean_err < 1e-30, f"max |mean| = {mean_err:.2e}"
        )

        # Hermitian symmetry at kz=0 (over the azimuthal m axis).
        n_pos = params.res.nz // 2
        sym_err = 0.0
        for i in range(1, n_pos):
            j = Nm - i
            diff = state_np[:, :, j, 0] - np.conj(state_np[:, :, i, 0])
            sym_err = max(sym_err, float(np.max(np.abs(diff))))
        kz0_imag = float(np.max(np.abs(state_np[:, :, 0, 0].imag)))
        sym_err = max(sym_err, kz0_imag)
        _check(
            "Hermitian symmetry at kz=0",
            sym_err < 1e-14,
            f"max sym error = {sym_err:.2e}",
        )

        # Seed determinism.
        state2 = _generate_annular(args)
        det_err = float(jnp.max(jnp.abs(state - state2)))
        _check("seed determinism", det_err == 0.0, f"max diff = {det_err:.2e}")

    else:
        print(
            f"  (self-test implemented for cartesian and annular systems; "
            f"'{system}' skipped)"
        )

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed > 0 else 0)


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    global args
    args = _parse_args()
    _setup_jax_and_params(args)

    from dnsjax.parameters import (
        annular_systems,
        cartesian_systems,
        cylindrical_systems,
        params,
        periodic_systems,
    )

    if args.test:
        _run_tests()
        return

    from dnsjax.sharding import sharding
    from dnsjax.snapshot import save_snapshot

    system = params.phys.system
    if system in cartesian_systems:
        state, ys, _, _, _ = _generate_cartesian(args)
    elif system in cylindrical_systems:
        state = _generate_cylindrical(args)
    elif system in annular_systems:
        state = _generate_annular(args)
    elif system in periodic_systems:
        state = _generate_triply_periodic(args)
    else:
        print(f"Unknown system: {system}")
        sys.exit(1)

    save_snapshot(state, t=0.0, it=0, path=args.output)
    sharding.print(
        f"Saved random perturbation to {args.output}/\n"
        f"  system={system}, "
        f"resolution=({args.nx}, {args.ny}, {args.nz}), "
        f"amplitude={args.amplitude}, "
        f"smoothness={args.smoothness}, "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()
