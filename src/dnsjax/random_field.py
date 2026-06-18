r"""Random divergence-free initial-condition generators.

Builds a random divergence-free perturbation of the base flow for any
implemented flow system, returned as a sharded spectral state ready to
time step.  This is the shared implementation behind both:

- ``scripts/random_field.py`` -- the CLI that saves the state as a
  zarr3 snapshot (and runs the per-geometry self-tests), and
- ``dnsjax.__main__`` -- the in-process random initial-condition start
  mode (``init.random_field``), which avoids a snapshot disk round-trip.

The wavenumber-dependent amplitude of each Fourier mode decays as

.. math::
    A(k) = (1 - s)^{|k_x| + |k_z| (+ |k_y|)}

where `$s$` is the ``smoothness`` argument; the field is then normalised
so the volume-averaged L2 norm equals ``amplitude``.

**Dean flow** (``system == "dean"``) integrates the *total* field, so the
generated divergence-free perturbation is added to the analytical laminar
Dean profile (``add_dean_laminar``) to form the total-field IC.

**Non-JAX operations**: the per-mode divergence-free enforcement loops
over Fourier modes and uses NumPy for the `$D_1 \mathbf{v}$` matvecs,
because Python-level looping in JAX would incur tracing overhead.  All
other array work uses JAX.

**Import-order discipline**: only NumPy and the (JAX-free)
``parameters`` singletons are imported at module top.  ``jax``,
``sharding``, and the geometry modules (which build the ``fourier``
singleton at import) are imported lazily inside each generator, so
importing this module is safe before JAX is configured and before the
flow system is selected.
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np

from .parameters import (
    annular_systems,
    cartesian_systems,
    cylindrical_systems,
    derived_params,
    params,
    periodic_systems,
)

if TYPE_CHECKING:
    # ``Array`` is used only in (stringised) annotations, so it never
    # needs importing at runtime -- keeping this module importable
    # before JAX is configured (see the module docstring).
    from jax import Array

# ── Hermitian-symmetry enforcement ───────────────────────────────

# The real-FFT axis (kx for Cartesian/periodic, kz for cylindrical)
# stores only non-negative wavenumbers.  On the complex-FFT axis
# at kx=0 (or kz=0 for cylindrical), the stored modes must satisfy
# conjugate symmetry for the physical field to be real.  The helper
# below is pure NumPy (no JAX) since it works on the host array.


def enforce_hermitian_slice(
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


def generate_cartesian(
    amplitude: float,
    smoothness: float,
    seed: int,
    mean_flow: bool,
) -> Array:
    """Generate a random divergence-free Cartesian perturbation.

    Returns a JAX array of shape ``(3, Ny, Nkz, Nkx)``.
    """
    import jax
    from jax import numpy as jnp

    from .geometries.wall_bounded._base import get_norm
    from .geometries.wall_bounded.cartesian import (
        build_cartesian_grid,
        fourier,
    )
    from .sharding import sharding

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

    decay = 1.0 - smoothness
    window_noslip = 1.0 - ys_np**2

    # Wavenumber decay factors: shape (Nkz, Nkx).
    mode_decay = decay ** (np.abs(kz_np[:, None]) + np.abs(kx_np[None, :]))

    rng = np.random.default_rng(seed)
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
        enforce_hermitian_slice(raw[c, :, :, 0].T, params.res.nz)

    # Zero mean mode unless mean_flow.
    if not mean_flow:
        raw[:, :, 0, 0] = 0.0

    # ── JAX: normalise and return ────────────────────────────
    state = jax.device_put(
        jnp.asarray(raw, dtype=sharding.complex_type),
        sharding.spec_vector_shard,
    )
    norm = get_norm(state, fourier.k_metric, y_weights)
    state = state * (amplitude / norm)
    return state


# ── Cylindrical generation ───────────────────────────────────────


def generate_cylindrical(
    amplitude: float,
    smoothness: float,
    seed: int,
    mean_flow: bool,
) -> Array:
    r"""Generate a random perturbation for pipe flow.

    Returns a JAX array of shape ``(3, Nr, Nm, Nkz)`` in
    `$(u_z, u_+, u_-)$` form, with no-slip at the wall `$r = 1$`.

    For `$k_z \neq 0$` modes the field is divergence-free by
    construction (`$u_z$` derived from continuity, with the near-wall
    `$u_\pm$` point adjusted so `$D_1 u_r = 0$` at the wall `$r = 1$`,
    so the derived `$u_z$` inherits an exact zero wall value).  The
    inner end `$r = 0$` is the axis (regularity via parity), not a
    wall.  For `$k_z = 0$` modes a small residual divergence may
    remain; the first corrector step of the solver projects it out
    via the IMM.
    """
    import jax
    from jax import numpy as jnp

    from .geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
        fourier,
        get_norm2_cyl,
    )
    from .sharding import sharding

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

    decay = 1.0 - smoothness
    window_wall = 1.0 - rs_np  # f(1) = 0

    # Wavenumber decay: |kz_phys| + |m|*2*pi/lz.
    kz_abs = np.abs(kz_np)
    m_abs = np.abs(m_np) * 2 * pi / params.geo.lz
    mode_decay = decay ** (kz_abs[None, :] + m_abs[:, None])

    rng = np.random.default_rng(seed)
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
    # The near-(outer-)wall interior point of u_+ and u_- is first
    # adjusted so D1_pm @ u_pm = 0 at the wall r=1; since u_pm already
    # vanish at the wall (wall window), D1_pm @ u_r = 0 there too, so
    # the u_z derived from continuity inherits an exact zero wall
    # value.  The inner end r=0 is the axis (regularity via parity),
    # not a wall, so it needs no adjustment.
    for im in range(Nm):
        m_val = int(m_np[im])
        # u+, u-: parity (-1)^{m+1}
        D1_pm = D1_even_np if (m_val + 1) % 2 == 0 else D1_odd_np

        for ik in range(Nkz):
            for comp in (1, 2):  # u_+, u_-
                f = raw[comp, :, im, ik]
                f[-2] += -(D1_pm[-1, :] @ f) / D1_pm[-1, -2]

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

    # Zero mean mode unless mean_flow.
    if not mean_flow:
        raw[:, :, 0, 0] = 0.0

    # Hermitian symmetry at kz=0 (real-FFT axis):
    # fix m axis for each component.
    # raw shape: (3, Nr, Nm, Nkz); kz=0 is index 3.
    for c in range(3):
        enforce_hermitian_slice(raw[c, :, :, 0].T, params.res.nz)

    # ── JAX: normalise and return ────────────────────────────
    state = jax.device_put(
        jnp.asarray(raw, dtype=sharding.complex_type),
        sharding.spec_vector_shard,
    )

    norm2 = get_norm2_cyl(state, fourier.k_metric, y_weights)
    norm = jnp.sqrt(norm2)
    state = state * (amplitude / norm)
    return state


# ── Annular generation ───────────────────────────────────────────


def generate_annular(
    amplitude: float,
    smoothness: float,
    seed: int,
    mean_flow: bool,
) -> Array:
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

    from .geometries.wall_bounded.annular import (
        build_annular_grid,
        fourier,
        get_norm2_annular,
    )
    from .sharding import sharding

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

    decay = 1.0 - smoothness
    # No-slip window: zero at both walls, peak in the interior.
    window = (rs_np - r1) * (r2 - rs_np)
    window = window / np.max(window)

    kz_abs = np.abs(kz_np)
    m_abs = np.abs(m_np) * 2 * pi / params.geo.lz
    mode_decay = decay ** (kz_abs[None, :] + m_abs[:, None])

    rng = np.random.default_rng(seed)
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

    # Zero mean mode unless mean_flow.
    if not mean_flow:
        raw[:, :, 0, 0] = 0.0

    # Hermitian symmetry at kz=0 (real-FFT axis): fix m axis.
    for c in range(3):
        enforce_hermitian_slice(raw[c, :, :, 0].T, params.res.nz)

    # ── JAX: normalise and return ────────────────────────────
    state = jax.device_put(
        jnp.asarray(raw, dtype=sharding.complex_type),
        sharding.spec_vector_shard,
    )

    norm2 = get_norm2_annular(state, fourier.k_metric, y_weights)
    norm = jnp.sqrt(norm2)
    state = state * (amplitude / norm)
    return state


def add_dean_laminar(state: Array) -> Array:
    r"""Add the analytical laminar Dean profile to a perturbation.

    Dean flow integrates the **total** velocity, so a usable initial
    condition is the closed-form laminar azimuthal profile (placed at
    the mean mode) plus the divergence-free random perturbation from
    :func:`generate_annular`.  The laminar profile is axisymmetric and
    zero at both walls, so it preserves the perturbation's
    divergence-free and no-slip properties.  Returns the total spectral
    state in `$(u_z, u_+, u_-)$` form.
    """
    from jax import numpy as jnp

    from .geometries.wall_bounded.annular import (
        build_annular_grid,
        dean_laminar_u_theta,
        fourier,
    )
    from .sharding import sharding

    rs, *_ = build_annular_grid(
        params.res.ny,
        params.res.fd_order,
        derived_params.r_inner,
        derived_params.r_outer,
        params.geo.wall_grid,
    )
    u_theta = dean_laminar_u_theta(rs, params.geo.eta)  # (Nr,) real
    # Place U_theta at the mean mode: u_+ = i U_theta, u_- = -i U_theta.
    u_spec = jnp.where(fourier.mean_mask, u_theta[:, None, None], 0.0)
    laminar = jnp.stack(
        [
            jnp.zeros_like(u_spec, dtype=sharding.complex_type),
            (1j * u_spec).astype(sharding.complex_type),
            (-1j * u_spec).astype(sharding.complex_type),
        ]
    )
    return state + laminar


# ── Triply-periodic generation ───────────────────────────────────


def generate_triply_periodic(
    amplitude: float,
    smoothness: float,
    seed: int,
    mean_flow: bool,
) -> Array:
    """Generate a random divergence-free periodic perturbation.

    Returns a JAX array of shape ``(3, Nky, Nkz, Nkx)``.
    Uses the Leray projection to enforce incompressibility.
    """
    import jax
    from jax import numpy as jnp

    from .geometries.triply_periodic.triply_periodic import (
        fourier,
        get_norm,
    )
    from .sharding import sharding

    Nky = params.res.ny - 1
    Nkz = params.res.nz - 1
    Nkx = params.res.nx // 2

    # ── NumPy: generate and project (non-JAX) ───────────────
    kx_np = np.asarray(fourier.kx).ravel()  # (Nkx,)
    kz_np = np.asarray(fourier.kz).ravel()  # (Nkz,)
    ky_np = np.asarray(fourier.ky).ravel()  # (Nky,)

    decay = 1.0 - smoothness

    # 3D L1-norm decay.
    mode_decay = decay ** (
        np.abs(ky_np[:, None, None])
        + np.abs(kz_np[None, :, None])
        + np.abs(kx_np[None, None, :])
    )

    rng = np.random.default_rng(seed)
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
    if not mean_flow:
        raw[:, 0, 0, 0] = 0.0

    # Hermitian symmetry at kx=0: joint 2D constraint
    # f_hat(ky, kz, 0) = conj(f_hat(-ky, -kz, 0)).
    from .operators import complex_harmonics as _ch

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
    state = state * (amplitude / norm)
    return state


# ── Dispatch ─────────────────────────────────────────────────────


def generate_random_state(
    amplitude: float,
    smoothness: float,
    seed: int,
    mean_flow: bool = False,
) -> Array:
    """Generate a random initial state for the configured flow system.

    Dispatches to the geometry-specific generator for
    ``params.phys.system`` and returns the sharded spectral state (on
    ``sharding.spec_vector_shard``), ready to time step -- the same
    object type that ``init_state`` / ``load_snapshot`` return.  For the
    total-field Dean flow the analytical laminar profile is added to the
    perturbation; every other system returns the perturbation directly.

    Requires JAX to be configured and the parameter singletons set (the
    geometry ``fourier`` singleton is built lazily by the dispatched
    generator's import).
    """
    system = params.phys.system
    if system in cartesian_systems:
        return generate_cartesian(amplitude, smoothness, seed, mean_flow)
    if system in cylindrical_systems:
        return generate_cylindrical(amplitude, smoothness, seed, mean_flow)
    if system in annular_systems:
        state = generate_annular(amplitude, smoothness, seed, mean_flow)
        if system == "dean":
            state = add_dean_laminar(state)
        return state
    if system in periodic_systems:
        return generate_triply_periodic(amplitude, smoothness, seed, mean_flow)
    raise ValueError(f"Unknown system: {system}")
