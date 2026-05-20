r"""Cylindrical geometry: Fourier class, norms, IMM, and solvers.

Provides all geometry-general infrastructure for wall-bounded
cylindrical flows: the ``Fourier`` wavenumber class, the
``CylindricalFlow`` base dataclass (half-diameter radial grid,
parity-reduced FD matrices, IMM operators), spectral solvers
(influence-matrix method, predictor-corrector time stepping), and
diagnostic helpers (norms, perturbation energy).

Decoupled velocity formulation
------------------------------
The cylindrical Navier-Stokes vector Laplacian couples
`$u_r$` and `$u_\theta$` through `$1/r^2$` terms.
Following Openpipeflow (Willis 2017), we decouple them via

.. math::
    u_+ = u_r + i\,u_\theta, \qquad
    u_- = u_r - i\,u_\theta,

reducing the vector problem to three scalar Helmholtz
equations with **effective azimuthal modes**:

.. math::
    m_{\mathrm{eff}} = m + 1 \;\text{for } u_+, \qquad
    m_{\mathrm{eff}} = m - 1 \;\text{for } u_-, \qquad
    m_{\mathrm{eff}} = m     \;\text{for } u_z.

The effective azimuthal mode `$m_{\mathrm{eff}}$` governs the
scalar Laplacian structure: after decoupling, each component
satisfies a Helmholtz equation whose radial operator is
`$\partial_r^2 + (1/r)\partial_r - m_{\mathrm{eff}}^2/r^2$`.

Despite having different `$m_{\mathrm{eff}}$` values, `$u_+$`
and `$u_-$` share the **same parity** `$(-1)^{m+1}$`.  Parity
is a kinematic property (how a field transforms under
`$r \to -r$` on the auxiliary grid), while `$m_{\mathrm{eff}}$`
determines the operator spectrum.  The coincidence
`$(-1)^{m+1} = (-1)^{m-1}$` makes the parity identical.

Parity-reduced FD matrices
--------------------------
A field with azimuthal Fourier mode `$m$` must be
single-valued when analytically continued across the pipe
centre.  The point at radius `$r$` and angle
`$\theta + \pi$` is the same physical point as `$(-r,
\theta)$` on the auxiliary grid.  The factor
`$e^{im\pi} = (-1)^m$` from the Fourier mode, combined
with the reversal of `$\hat{e}_r$` and `$\hat{e}_\theta$`
when crossing the origin, determines the parity of each
field component:

- Pressure `$p$` and axial velocity `$u_z$`:
  parity `$(-1)^m$`, so `$m_{\mathrm{eff}} = m$`.
- `$u_+ = u_r + i\,u_\theta$`: parity `$(-1)^{m+1}$`,
  `$m_{\mathrm{eff}} = m + 1$` in the Helmholtz operator.
- `$u_- = u_r - i\,u_\theta$`: parity `$(-1)^{m+1}$`,
  `$m_{\mathrm{eff}} = m - 1$` in the Helmholtz operator.

Even parity (`$m_{\mathrm{eff}}$` even) means `$g$` is
symmetric about `$r = 0$`: `$g'(0) = 0$` (Neumann-like).
Odd parity (`$m_{\mathrm{eff}}$` odd) means `$g$` is
antisymmetric: `$g(0) = 0$` (Dirichlet-like).

The parity-reduced FD matrices encode these constraints
because the underlying stencils span across `$s = 0$` on
the auxiliary grid.  No explicit regularity BCs or
l'Hopital treatment at `$r = 0$` are needed.

Influence-matrix method (`$1 \times 1$`)
----------------------------------------
The pipe has a single physical wall at `$r = 1$`.
Regularity at `$r = 0$` is handled by the parity-reduced
FD matrices, not by a boundary condition.  This gives a
`$1 \times 1$` influence matrix -- simpler than the
Cartesian `$2 \times 2$` case.

Flow-specific modules (e.g. ``flows.wall_bounded.pipe``) subclass
``CylindricalFlow`` to define the base flow, then call
``build_cylindrical_stepper`` to obtain ready-to-use
time-stepping functions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
from jax import Array
from jax import numpy as jnp

from ...fd import build_diff_matrices, build_integration_weights
from ...operators import (
    complex_harmonics,
    phys_to_spec_2d,
    real_harmonics,
    spec_to_phys_2d,
)
from ...parameters import params
from ...rhs import get_nonlin
from ...sharding import register_dataclass_pytree, sharding
from ...solvers import (
    DenseJAXSolver,
    PerModeBandedOperator,
    _extract_banded_corners,
    _spike_factor,
    validate_spike_partition,
)
from ._base import (
    build_wall_bounded_stepper,
    extract_mean_mode,
    get_inprod,  # noqa: F401 — re-exported
    get_norm,  # noqa: F401 — re-exported
    get_norm2,
    init_state,  # noqa: F401 — re-exported
    integrate_scalar,
    phys_to_spec,  # noqa: F401 — re-exported
    spec_to_phys,  # noqa: F401 — re-exported
)


@register_dataclass_pytree
@dataclass
class Fourier:
    r"""Wavenumber grids for the cylindrical geometry.

    Broadcasting shapes match the spectral layout
    ``(Nm, Nkz, Nr)`` = ``(nz-1, nx//2, ny)``:

    - ``kz``: shape ``(1, nx//2, 1)`` -- axial wavenumber
      (real FFT on the streamwise ``x`` parameter direction).
    - ``m``: shape ``(nz-1, 1, 1)`` -- azimuthal mode number
      (complex FFT on the ``z`` parameter direction with
      `$l_z = 2\pi$`, so integer-valued).

    The coordinate mapping is:

    =============  ===========  ============  =============
    Physical       Parameter    Transform     Wavenumber
    =============  ===========  ============  =============
    `$z_{axial}$`  ``x`` (rfft) real FFT      `$k_z$`
    `$\theta$`     ``z`` (cfft) complex FFT   `$m$` (int)
    `$r$`          ``y`` (FD)   none          grid points
    =============  ===========  ============  =============

    ``k_metric`` equals 2 for `$k_z > 0$` and 1 for
    `$k_z = 0$`, accounting for the Hermitian symmetry of
    the real FFT.

    ``m_is_even`` is a boolean mask ``(nz-1, 1, 1)``
    selecting the azimuthal modes where `$m$` is even, used
    to choose the correct parity-reduced FD matrices.
    """

    kz: Array = field(init=False)
    m: Array = field(init=False)
    k_metric: Array = field(init=False)
    kz2: Array = field(init=False)
    m2: Array = field(init=False)
    k2_is_zero: Array = field(init=False)
    m_is_even: Array = field(init=False)

    def __post_init__(self) -> None:
        kz_vals = real_harmonics(params.res.nx) * 2 * jnp.pi / params.geo.lx
        self.kz = jax.device_put(
            kz_vals.reshape([1, -1, 1]).astype(sharding.float_type),
            sharding.spec_scalar_shard,
        )
        m_vals = complex_harmonics(params.res.nz)
        self.m = jax.device_put(
            m_vals.reshape([-1, 1, 1]).astype(sharding.float_type),
            sharding.no_shard,
        )

        self.k_metric = jnp.where(self.kz == 0, 1, 2).astype(
            sharding.float_type
        )

        self.kz2 = self.kz**2
        self.m2 = self.m**2
        self.k2_is_zero = (self.kz2 + self.m2) == 0.0
        self.m_is_even = (self.m % 2 == 0).astype(sharding.float_type)


fourier: Fourier = Fourier()


# Backward-compatible alias.
integrate_scalar_in_r = integrate_scalar


# ── Cylindrical-specific norms ──────────────────────────────────


def get_pert_enstrophy_cyl(
    state: Array,
    D1_pos: Array,
    D1_ghost: Array,
    m_is_even: Array,
    inv_r: Array,
    m: Array,
    kz2: Array,
    k_metric: Array,
    y_weights: Array,
) -> Array:
    r"""Perturbation enstrophy for the cylindrical geometry.

    Uses the identity
    `$\Omega' = \langle |\nabla \mathbf{u}'|^2 \rangle$`,
    split into radial derivative, azimuthal, and axial terms:

    .. math::
        \Omega' = \langle |D_1 \mathbf{u}'|^2 \rangle
        + \langle |m_{\mathrm{eff}}/r\;\mathbf{u}'|^2 \rangle
        + \langle k_z^2\,|\mathbf{u}'|^2 \rangle

    where `$m_{\mathrm{eff}} = m$` for `$u_z$`,
    `$m + 1$` for `$u_+$`, `$m - 1$` for `$u_-$`.
    The radial derivative uses parity-dependent FD matrices:
    `$D_1 = D_{1,\mathrm{pos}} + (-1)^{m_{\mathrm{eff}}}
    D_{1,\mathrm{ghost}}$`.

    Parameters
    ----------
    state:
        Spectral velocity in `$(u_z, u_+, u_-)$` form,
        shape ``(3, Nm, Nkz, Nr)``.
    D1_pos:
        Common part of first-derivative FD matrix.
    D1_ghost:
        Ghost correction for `$D_1$`.
    m_is_even:
        Boolean mask for even `$m$`, shape ``(Nm, 1, 1)``.
    inv_r:
        `$1/r$` on the radial grid.
    m:
        Azimuthal mode number, shape ``(Nm, 1, 1)``.
    kz2:
        `$k_z^2$`, shape ``(1, Nkz, 1)``.
    k_metric:
        Hermitian-symmetry weight for the real FFT axis.
    y_weights:
        Radial integration weights `$w_j r_j$`.
    """
    # Parity signs: u_z has parity (-1)^m, u_± has (-1)^{m+1}.
    p_sign_z = m_is_even * 2 - 1
    p_sign_pm = -p_sign_z

    # Batched D1 matvecs (2 GEMMs for all 3 components).
    dy_pos = jnp.einsum("ij, cmzj -> cmzi", D1_pos, state)
    dy_ghost = jnp.einsum("ij, cmzj -> cmzi", D1_ghost, state)
    p_signs = jnp.stack([p_sign_z, p_sign_pm, p_sign_pm])
    dy_state = dy_pos + p_signs * dy_ghost

    enstrophy_D1 = get_norm2_cyl(dy_state, k_metric, y_weights)

    # Azimuthal term: m_eff/r * u for each component.
    state_m = jnp.stack(
        [
            m * inv_r * state[0],
            (m + 1) * inv_r * state[1],
            (m - 1) * inv_r * state[2],
        ]
    )
    enstrophy_m = get_norm2_cyl(state_m, k_metric, y_weights)

    # Axial term: kz^2 |u|^2.
    enstrophy_kz = get_norm2_cyl(state, kz2 * k_metric, y_weights)

    return enstrophy_D1 + enstrophy_m + enstrophy_kz


def get_norm2_cyl(state: Array, k_metric: Array, y_weights: Array) -> Array:
    r"""Cylindrical squared L2 norm for `$(u_z, u_+, u_-)$`.

    The physical velocity magnitude satisfies
    `$|u_r|^2 + |u_\theta|^2 + |u_z|^2
    = (|u_+|^2 + |u_-|^2)/2 + |u_z|^2$`,
    so the `$u_\pm$` components carry a factor of 1/2
    relative to `$u_z$`.

    Parameters
    ----------
    state:
        Spectral velocity in `$(u_z, u_+, u_-)$` form,
        shape ``(3, Nm, Nkz, Nr)``.
    k_metric:
        Hermitian-symmetry weight for the real FFT axis.
    y_weights:
        Radial integration weights `$w_j r_j$`.
    """
    u_z, u_plus, u_minus = state[0], state[1], state[2]
    pm_norm2 = get_norm2(jnp.stack([u_plus, u_minus]), k_metric, y_weights)
    uz_norm2 = get_norm2(u_z[None], k_metric, y_weights)
    return pm_norm2 / 2 + uz_norm2


# ── Half-diameter grid and parity-reduced FD matrices ──────────────


def _build_half_cgl_grid(Nr: int) -> Array:
    r"""Build the half-CGL radial grid on `$(0, 1]$`.

    Takes the positive half of a `$2 N_r$`-point CGL grid on
    `$[-1, 1]$`:

    .. math::
        s_j = -\cos\!\bigl(j\pi/(2N_r - 1)\bigr),
        \quad j = 0, \ldots, 2N_r - 1.

    Since `$2 N_r$` is always even, no `$s_j$` falls on
    `$s = 0$`.  The `$N_r$` points with `$s_j > 0$`
    (indices `$j = N_r, \ldots, 2N_r - 1$`) form the radial
    grid `$r_0 < r_1 < \cdots < r_{N_r - 1} = 1$`, with CGL
    clustering near the pipe wall and widest spacing near
    the centre.

    Parameters
    ----------
    Nr:
        Number of radial grid points.

    Returns
    -------
    :
        Radial grid array, shape ``(Nr,)``.
    """
    N_full = 2 * Nr
    s = -jnp.cos(
        jnp.arange(N_full, dtype=sharding.float_type) * jnp.pi / (N_full - 1)
    )
    return s[Nr:]


def _build_parity_reduced_matrices(
    rs: Array, p: int
) -> tuple[Array, Array, Array, Array, Array, Array]:
    r"""Build parity-reduced FD matrices from the auxiliary grid.

    An auxiliary `$2 N_r$`-point grid on `$[-1, 1]$` is formed
    by mirroring: `$\{-r_{N_r-1}, \ldots, -r_0, r_0, \ldots,
    r_{N_r-1}\}$`.  Full-grid FD matrices are built on the
    auxiliary grid, then reduced by substituting the parity
    relation `$u(-r_j) = (-1)^{m_{\mathrm{eff}}} u(r_j)$`:

    .. math::
        D_{\mathrm{reduced}} = D_{\mathrm{pos}}
        + (-1)^{m_{\mathrm{eff}}} \widetilde{D}_{\mathrm{ghost}}

    where `$D_{\mathrm{pos}}$` is the positive-row,
    positive-column block and `$\widetilde{D}_{\mathrm{ghost}}$`
    is the positive-row, ghost-column block with columns
    flipped.

    Returns
    -------
    D1_even, D2_even:
        Parity-reduced matrices for even `$m_{\mathrm{eff}}$`.
    D1_odd, D2_odd:
        Parity-reduced matrices for odd `$m_{\mathrm{eff}}$`.
    D1_pos, D2_pos:
        Common (parity-independent) part: positive-row,
        positive-column block of the full-grid matrices.
    """
    Nr = len(rs)
    aux_grid = jnp.concatenate([-rs[::-1], rs])
    D1_full, D2_full = build_diff_matrices(aux_grid, p)

    D1_pos = D1_full[Nr:, Nr:]
    D1_ghost_flipped = D1_full[Nr:, :Nr][:, ::-1]
    D1_even = D1_pos + D1_ghost_flipped
    D1_odd = D1_pos - D1_ghost_flipped

    D2_pos = D2_full[Nr:, Nr:]
    D2_ghost_flipped = D2_full[Nr:, :Nr][:, ::-1]
    D2_even = D2_pos + D2_ghost_flipped
    D2_odd = D2_pos - D2_ghost_flipped

    return D1_even, D2_even, D1_odd, D2_odd, D1_pos, D2_pos


# ── SPIKE block-partitioned operator builders ─────────────────────


def _build_A_base(D1: Array, D2: Array, inv_r: Array) -> Array:
    r"""Build the radial base operator `$A_{\mathrm{base}}$`.

    .. math::
        A_{\mathrm{base}} = D_2 + \mathrm{diag}(1/r)\,D_1

    Parameters
    ----------
    D1:
        First-derivative matrix, shape ``(Nr, Nr)``.
    D2:
        Second-derivative matrix, shape ``(Nr, Nr)``.
    inv_r:
        `$1/r_j$`, shape ``(Nr,)``.
    """
    return D2 + jnp.diag(inv_r) @ D1


def _build_Lk_blocks_gpu(
    D1_wall: Array,
    A_base_even: Array,
    A_base_odd: Array,
    m_is_even: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    k2_is_zero: Array,
    p: int,
    P: int,
    m_blk: int,
) -> tuple[Array, Array, Array]:
    r"""Build SPIKE block-partitioned `$L_k$` on GPU.

    The pressure Poisson operator uses `$m_{\mathrm{eff}} = m$`
    (same parity as pressure / `$u_z$`):

    .. math::
        L_k = A_{\mathrm{base}}^{(\sigma_p)}
        - (m^2/r^2 + k_z^2)\,I

    where `$\sigma_p$` is even when `$m$` is even, odd when
    `$m$` is odd.  The `$m^2/r^2$` diagonal shift is
    **per-point** (varies with `$j$`), unlike the uniform
    `$k^2$` in the Cartesian case.

    The first block (`$i = 0$`) depends on parity (its first
    `$\sim p$` rows differ between even/odd FD matrices); all
    other blocks are parity-independent.  Per-mode selection
    uses ``jnp.where`` on the parity mask.

    Parameters
    ----------
    D1_wall:
        Last row of `$D_1$` (parity-independent), shape
        ``(Nr,)`` or ``(1, Nr)``.
    A_base_even:
        Base operator with even-parity FD matrices,
        shape ``(Nr, Nr)``.
    A_base_odd:
        Base operator with odd-parity FD matrices,
        shape ``(Nr, Nr)``.
    m_is_even:
        Boolean mask for even `$m$`, shape ``(Nm, 1, 1)``.
    m2:
        `$m^2$`, shape ``(Nm, 1, 1)``.
    inv_r2:
        `$1/r_j^2$`, shape ``(Nr,)``.
    kz2:
        `$k_z^2$`, shape ``(1, Nkz, 1)``.
    k2_is_zero:
        Mean-mode boolean mask.
    p:
        FD order (half-bandwidth).
    P:
        Number of SPIKE blocks.
    m_blk:
        Block size (``Nr // P``).

    Returns
    -------
    A_blocks:
        Diagonal blocks, ``(Nm, Nkz, P, m_blk, m_blk)``.
    B_corner:
        Right-coupling corners, ``(Nm, Nkz, P, p, p)``.
    C_corner:
        Left-coupling corners, ``(Nm, Nkz, P, p, p)``.
    """
    dtype = A_base_even.dtype

    eye_m = jnp.eye(m_blk, dtype=dtype)

    # Per-point diagonal shift: -(m^2/r_j^2 + kz^2) for each
    # radial point in each block.
    # m2_over_r2 has shape (Nm, 1, Nr) after broadcast.
    m2_over_r2 = m2 * inv_r2  # (Nm, 1, Nr)

    # Build even/odd diagonal blocks from A_base.
    def extract_blocks(A_base):
        return jnp.stack(
            [
                A_base[
                    i * m_blk : (i + 1) * m_blk, i * m_blk : (i + 1) * m_blk
                ]
                for i in range(P)
            ]
        )  # (P, m_blk, m_blk)

    A_blks_even = extract_blocks(A_base_even)  # (P, m_blk, m_blk)
    A_blks_odd = extract_blocks(A_base_odd)

    # Block 0 differs by parity; blocks 1..P-1 are identical.
    # Select block 0 per-mode via m_is_even.
    blk0_even = A_blks_even[0]  # (m_blk, m_blk)
    blk0_odd = A_blks_odd[0]
    # Squeeze m_is_even from (Nm, 1, 1) to (Nm, 1, 1)
    # for correct broadcast to (Nm, m_blk, m_blk).
    blk0 = jnp.where(
        m_is_even.ravel()[:, None, None], blk0_even, blk0_odd
    )  # (Nm, m_blk, m_blk)

    # Build each block with its diagonal shift incorporated.
    # Arithmetic with kz2 drives the kx-sharding.
    blocks = []
    for i in range(P):
        r_slice = slice(i * m_blk, (i + 1) * m_blk)
        shift = -(m2_over_r2[..., r_slice] + kz2)
        shift_diag = shift[..., None] * eye_m
        if i == 0:
            block = blk0[:, None, :, :] + shift_diag
        else:
            block = A_blks_even[i][None, None] + shift_diag
        blocks.append(block)
    A_blocks = jnp.stack(blocks, axis=2)

    # BC: wall row (last row of last block) -> Neumann D1[-1,:]
    # for non-mean modes, pin [...,0,1] for (m,kz) = (0,0).
    D1_wall_row = D1_wall[-m_blk:]  # last m_blk entries
    pin_row = jnp.zeros(m_blk, dtype=dtype).at[-1].set(1.0)
    wall_row = jnp.where(k2_is_zero, pin_row, D1_wall_row)
    A_blocks = A_blocks.at[:, :, -1, -1, :].set(wall_row)

    # Coupling corners (parity-independent: from A_base_even
    # which equals A_base_odd for blocks > 0).
    B_raw, C_raw = _extract_banded_corners(A_base_even, P, m_blk, p)
    # Apply the per-point diagonal shift to corners: the shift is
    # diagonal so it doesn't affect off-diagonal coupling corners.
    batch = m2.shape[:1] + kz2.shape[1:2] + (P, p, p)
    B_corner = jnp.broadcast_to(B_raw[None, None], batch)
    C_corner = jnp.broadcast_to(C_raw[None, None], batch)

    return A_blocks, B_corner, C_corner


def _build_Hk_blocks_gpu(
    A_base_even: Array,
    A_base_odd: Array,
    m_is_even_vel: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    nu: float,
    p: int,
    P: int,
    m_blk: int,
) -> tuple[Array, Array, Array]:
    r"""Build SPIKE block-partitioned `$H_k$` on GPU.

    Builds one of the three Helmholtz operators (`$H_{k,+}$`,
    `$H_{k,-}$`, or `$H_{k,z}$`).  The caller supplies the
    appropriate `$m_{\mathrm{eff}}^2$` and parity mask.

    .. math::
        H_k = \frac{1}{\Delta t}\,I
        + c\nu\bigl(m_{\mathrm{eff}}^2/r^2 + k_z^2\bigr)\,I
        - c\nu\,A_{\mathrm{base}}^{(\sigma)}

    The effective azimuthal mode `$m_{\mathrm{eff}}$`
    determines which `$1/r^2$` coefficient appears in the
    diagonal and which parity-reduced FD matrices are used
    for the first block.

    Parameters
    ----------
    A_base_even, A_base_odd:
        Base operators with even/odd parity FD matrices.
    m_is_even_vel:
        Parity mask for this velocity component.
    meff2:
        `$m_{\mathrm{eff}}^2$`, shape ``(Nm, 1, 1)``.
    inv_r2:
        `$1/r_j^2$`, shape ``(Nr,)``.
    kz2:
        `$k_z^2$`, shape ``(1, Nkz, 1)``.
    dt:
        Time step.
    c:
        Implicitness parameter.
    nu:
        Kinematic viscosity `$1/\mathrm{Re}$`.
    p, P, m_blk:
        FD order, block count, block size.

    Returns
    -------
    A_blocks, B_corner, C_corner:
        SPIKE block data with the same layout as
        :func:`_build_Lk_blocks_gpu`.
    """
    dtype = A_base_even.dtype
    eye_m = jnp.eye(m_blk, dtype=dtype)

    meff2_over_r2 = meff2 * inv_r2  # (Nm, 1, Nr)

    def extract_blocks(A_base):
        return jnp.stack(
            [
                A_base[
                    i * m_blk : (i + 1) * m_blk, i * m_blk : (i + 1) * m_blk
                ]
                for i in range(P)
            ]
        )

    A_blks_even = extract_blocks(A_base_even)
    A_blks_odd = extract_blocks(A_base_odd)

    blk0_even = A_blks_even[0]
    blk0_odd = A_blks_odd[0]
    blk0 = jnp.where(
        m_is_even_vel.ravel()[:, None, None], blk0_even, blk0_odd
    )  # (Nm, m_blk, m_blk)

    # Build each block with diagonal shift incorporated.
    # Arithmetic with kz2 drives the kx-sharding.
    blocks = []
    for i in range(P):
        r_slice = slice(i * m_blk, (i + 1) * m_blk)
        diag_val = 1.0 / dt + c * nu * (meff2_over_r2[..., r_slice] + kz2)
        diag_mat = diag_val[..., None] * eye_m
        if i == 0:
            block = -c * nu * blk0[:, None, :, :] + diag_mat
        else:
            block = -c * nu * A_blks_even[i][None, None] + diag_mat
        blocks.append(block)
    A_blocks = jnp.stack(blocks, axis=2)

    # Dirichlet no-slip wall BC: identity row at r = 1
    # (last row of last block).
    eN = jnp.zeros(m_blk, dtype=dtype).at[-1].set(1.0)
    A_blocks = A_blocks.at[:, :, -1, -1, :].set(eN[None, None])

    # Coupling corners: -c*nu * A_base sub-blocks.
    B_raw, C_raw = _extract_banded_corners(
        A_base_even, P, m_blk, p, scale=-c * nu
    )
    batch = meff2.shape[:1] + kz2.shape[1:2] + (P, p, p)
    B_corner = jnp.broadcast_to(B_raw[None, None], batch)
    C_corner = jnp.broadcast_to(C_raw[None, None], batch)

    return A_blocks, B_corner, C_corner


# ── Dense-backend operator builders ───────────────────────────────


def _build_Lk_dense_gpu(
    D1_wall: Array,
    A_base_even: Array,
    A_base_odd: Array,
    m_is_even: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    k2_is_zero: Array,
) -> Array:
    r"""Build dense `$L_k$` on GPU (dense backend only).

    Returns the full ``(Nm, Nkz, Nr, Nr)`` pressure Poisson
    operator.  The parity-dependent row selection is handled
    by ``jnp.where`` on the ``m_is_even`` mask.
    """
    Nr = A_base_even.shape[0]
    dtype = A_base_even.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)

    m2_over_r2 = m2 * inv_r2  # (Nm, 1, Nr)
    diag_shift = -(m2_over_r2 + kz2)  # (Nm, Nkz, Nr)

    Lk_even = A_base_even[None, None] + diag_shift[..., None] * eye_Nr
    Lk_odd = A_base_odd[None, None] + diag_shift[..., None] * eye_Nr
    Lk = jnp.where(m_is_even[..., None], Lk_even, Lk_odd)

    # Wall BC: Neumann D1[-1,:] for non-mean, pin for mean.
    D1_wall_1d = D1_wall.ravel()
    pin = eye_Nr[-1, :]
    wall_row = jnp.where(k2_is_zero, pin, D1_wall_1d)
    Lk = Lk.at[..., -1, :].set(wall_row)

    return Lk


def _build_Hk_dense_gpu(
    A_base_even: Array,
    A_base_odd: Array,
    m_is_even_vel: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    nu: float,
) -> Array:
    r"""Build dense `$H_k$` on GPU (dense backend only).

    Returns the full ``(Nm, Nkz, Nr, Nr)`` Helmholtz operator
    for one velocity component.
    """
    Nr = A_base_even.shape[0]
    dtype = A_base_even.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)

    meff2_over_r2 = meff2 * inv_r2
    diag_coeff = 1.0 / dt + c * nu * (meff2_over_r2 + kz2)

    Hk_even = diag_coeff[..., None] * eye_Nr - c * nu * A_base_even
    Hk_odd = diag_coeff[..., None] * eye_Nr - c * nu * A_base_odd
    Hk = jnp.where(m_is_even_vel[..., None], Hk_even, Hk_odd)

    # Dirichlet no-slip: identity wall row.
    eN = jnp.zeros(Nr, dtype=dtype).at[-1].set(1.0)
    Hk = Hk.at[..., -1, :].set(eN)

    return Hk


# ── CylindricalFlow base dataclass ─────────────────────────────────


@register_dataclass_pytree
@dataclass
class CylindricalFlow:
    r"""Precomputed data for wall-bounded cylindrical flows.

    Subclasses must set ``base_flow`` and ``curl_base_flow``
    *after* calling
    ``super().__post_init__()``, which builds the half-CGL
    radial grid, parity-reduced FD matrices, and all per-mode
    IMM operators.

    The velocity state is stored in decoupled form
    `$(u_z, u_+, u_-)$` where

    .. math::
        u_+ = u_r + i\,u_\theta, \qquad
        u_- = u_r - i\,u_\theta.

    Three separate Helmholtz operators are built:

    - `$H_{k,+}$` with `$m_{\mathrm{eff}} = m + 1$`
    - `$H_{k,-}$` with `$m_{\mathrm{eff}} = m - 1$`
    - `$H_{k,z}$` with `$m_{\mathrm{eff}} = m$`

    The pressure Poisson operator `$L_k$` uses
    `$m_{\mathrm{eff}} = m$`.  Parity selection:

    - `$L_k$` and `$H_{k,z}$` use parity `$(-1)^m$`
      (``m_is_even`` from ``fourier``).
    - `$H_{k,+}$` and `$H_{k,-}$` use parity `$(-1)^{m+1}$`
      (the opposite: ``~m_is_even``).

    Attributes
    ----------
    D1_pos:
        Common (parity-independent) part of the
        first-derivative FD matrix, shape ``(Nr, Nr)``.
    D2_pos:
        Common part of the second-derivative FD matrix,
        shape ``(Nr, Nr)``.
    D1_ghost:
        Ghost correction for `$D_1$`
        (`$D_{1,\mathrm{even}} - D_{1,\mathrm{pos}}$`);
        nonzero only in the first `$\sim p$` rows near
        `$r = 0$`, shape ``(Nr, Nr)``.
    D2_ghost:
        Ghost correction for `$D_2$`
        (`$D_{2,\mathrm{even}} - D_{2,\mathrm{pos}}$`),
        shape ``(Nr, Nr)``.
    D1_wall:
        Last row of `$D_1$` (parity-independent),
        shape ``(1, Nr)``.
    inv_r:
        `$1/r$` on the radial grid.
    inv_r2:
        `$1/r^2$` on the radial grid.
    """

    rs: Array = field(init=False)
    inv_r: Array = field(init=False)
    inv_r2: Array = field(init=False)
    y_weights: Array = field(init=False)
    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    D1_pos: Array = field(init=False)
    D2_pos: Array = field(init=False)
    D1_ghost: Array = field(init=False)
    D2_ghost: Array = field(init=False)
    D1_wall: Array = field(init=False)
    A_base_even: Array = field(init=False)
    A_base_odd: Array = field(init=False)
    Lk_op: DenseJAXSolver | PerModeBandedOperator = field(init=False)
    Hk_op: DenseJAXSolver | PerModeBandedOperator = field(init=False)
    p1: Array = field(init=False)
    v_plus_1: Array = field(init=False)
    v_minus_1: Array = field(init=False)
    q_z_1: Array = field(init=False)
    M_inv: Array = field(init=False)
    h_bulk_response: Array = field(init=False)
    H_bulk_inv: Array = field(init=False)

    def __post_init__(self) -> None:
        r"""Build radial grid, FD matrices, and IMM operators.

        Constructs the half-CGL grid on `$(0, 1]$`, builds
        parity-reduced FD matrices, assembles and factorises
        `$L_k$`, `$H_{k,+}$`, `$H_{k,-}$`, `$H_{k,z}$`
        directly on the device, then derives all homogeneous
        IMM data.
        """
        Nr = params.res.ny
        self.rs = _build_half_cgl_grid(Nr)
        self.inv_r = 1.0 / self.rs
        self.inv_r2 = self.inv_r**2

        # Integration weights with radial Jacobian folded in.
        w = build_integration_weights(self.rs, params.res.fd_order)
        self.y_weights = w * self.rs

        # Parity-reduced FD matrices.
        (
            D1_even,
            D2_even,
            D1_odd,
            D2_odd,
            D1_pos,
            D2_pos,
        ) = _build_parity_reduced_matrices(self.rs, params.res.fd_order)

        self.D1_pos = jax.device_put(D1_pos, sharding.no_shard)
        self.D2_pos = jax.device_put(D2_pos, sharding.no_shard)

        # Ghost correction matrices: the difference between the
        # parity-reduced and the common (pos) part.  Only the first
        # ~p rows are nonzero; we store the full (Nr, Nr) shape for
        # simplicity (matvec cost is dominated by D1_pos/D2_pos).
        self.D1_ghost = jax.device_put(D1_even - D1_pos, sharding.no_shard)
        self.D2_ghost = jax.device_put(D2_even - D2_pos, sharding.no_shard)

        # Wall row of D1 (parity-independent, last row).
        self.D1_wall = jax.device_put(D1_pos[-1:, :], sharding.no_shard)

        # Base operators.
        self.A_base_even = _build_A_base(D1_even, D2_even, self.inv_r)
        self.A_base_odd = _build_A_base(D1_odd, D2_odd, self.inv_r)

        # Distribute grid arrays.
        self.rs = jax.device_put(self.rs, sharding.no_shard)
        self.inv_r = jax.device_put(self.inv_r, sharding.no_shard)
        self.inv_r2 = jax.device_put(self.inv_r2, sharding.no_shard)
        self.y_weights = jax.device_put(self.y_weights, sharding.no_shard)

        Nm = params.res.nz - 1
        Nkz = params.res.nx // 2

        fd_p = params.res.fd_order
        dt = params.step.dt
        c_impl = params.step.implicitness
        nu = 1.0 / params.phys.re

        # Effective azimuthal mode squared for each component.
        m_plus_1_sq = (fourier.m + 1) ** 2  # (Nm, 1, 1)
        m_minus_1_sq = (fourier.m - 1) ** 2
        m_sq = fourier.m2

        # Parity masks:
        # pressure / u_z use (-1)^m  -> m_is_even
        # u_+, u_- use (-1)^{m+1}   -> ~m_is_even (opposite)
        m_is_even_p = fourier.m_is_even  # (Nm, 1, 1)
        m_is_even_v = 1.0 - fourier.m_is_even  # opposite

        if params.solver.backend == "banded":
            P_blk, m_blk = validate_spike_partition(Nr, fd_p, "Nr")

            # Lk
            Lk_A, Lk_B, Lk_C = _build_Lk_blocks_gpu(
                self.D1_wall.ravel(),
                self.A_base_even,
                self.A_base_odd,
                m_is_even_p,
                m_sq,
                self.inv_r2,
                fourier.kz2,
                fourier.k2_is_zero,
                fd_p,
                P_blk,
                m_blk,
            )
            self.Lk_op = PerModeBandedOperator(
                *_spike_factor(Lk_A, Lk_B, Lk_C)
            )

            # Hk_plus (meff = m+1, parity = (-1)^{m+1})
            Hp_A, Hp_B, Hp_C = _build_Hk_blocks_gpu(
                self.A_base_even,
                self.A_base_odd,
                m_is_even_v,
                m_plus_1_sq,
                self.inv_r2,
                fourier.kz2,
                dt,
                c_impl,
                nu,
                fd_p,
                P_blk,
                m_blk,
            )
            lu_p, piv_p, V_p, W_p, rl_p, rp_p = _spike_factor(Hp_A, Hp_B, Hp_C)

            # Hk_minus (meff = m-1, parity = (-1)^{m+1})
            Hm_A, Hm_B, Hm_C = _build_Hk_blocks_gpu(
                self.A_base_even,
                self.A_base_odd,
                m_is_even_v,
                m_minus_1_sq,
                self.inv_r2,
                fourier.kz2,
                dt,
                c_impl,
                nu,
                fd_p,
                P_blk,
                m_blk,
            )
            lu_m, piv_m, V_m, W_m, rl_m, rp_m = _spike_factor(Hm_A, Hm_B, Hm_C)

            # Hk_z (meff = m, parity = (-1)^m)
            Hz_A, Hz_B, Hz_C = _build_Hk_blocks_gpu(
                self.A_base_even,
                self.A_base_odd,
                m_is_even_p,
                m_sq,
                self.inv_r2,
                fourier.kz2,
                dt,
                c_impl,
                nu,
                fd_p,
                P_blk,
                m_blk,
            )
            lu_z, piv_z, V_z, W_z, rl_z, rp_z = _spike_factor(Hz_A, Hz_B, Hz_C)

            # Combined Hk: component order (plus, minus, z).
            self.Hk_op = PerModeBandedOperator(
                lu=jnp.stack([lu_p, lu_m, lu_z]),
                piv=jnp.stack([piv_p, piv_m, piv_z]),
                V=jnp.stack([V_p, V_m, V_z]),
                W=jnp.stack([W_p, W_m, W_z]),
                red_lu=jnp.stack([rl_p, rl_m, rl_z]),
                red_piv=jnp.stack([rp_p, rp_m, rp_z]),
            )

        else:
            # Dense backend
            Lk_dense = _build_Lk_dense_gpu(
                self.D1_wall,
                self.A_base_even,
                self.A_base_odd,
                m_is_even_p,
                m_sq,
                self.inv_r2,
                fourier.kz2,
                fourier.k2_is_zero,
            )
            self.Lk_op = DenseJAXSolver(Lk_dense)

            Hp_dense = _build_Hk_dense_gpu(
                self.A_base_even,
                self.A_base_odd,
                m_is_even_v,
                m_plus_1_sq,
                self.inv_r2,
                fourier.kz2,
                dt,
                c_impl,
                nu,
            )
            Hk_plus_solver = DenseJAXSolver(Hp_dense)

            Hm_dense = _build_Hk_dense_gpu(
                self.A_base_even,
                self.A_base_odd,
                m_is_even_v,
                m_minus_1_sq,
                self.inv_r2,
                fourier.kz2,
                dt,
                c_impl,
                nu,
            )
            Hk_minus_solver = DenseJAXSolver(Hm_dense)

            Hz_dense = _build_Hk_dense_gpu(
                self.A_base_even,
                self.A_base_odd,
                m_is_even_p,
                m_sq,
                self.inv_r2,
                fourier.kz2,
                dt,
                c_impl,
                nu,
            )
            Hk_z_solver = DenseJAXSolver(Hz_dense)

            # Combined Hk: component order (plus, minus, z).
            self.Hk_op = DenseJAXSolver.from_factors(
                lu=jnp.stack(
                    [
                        Hk_plus_solver.lu,
                        Hk_minus_solver.lu,
                        Hk_z_solver.lu,
                    ]
                ),
                piv=jnp.stack(
                    [
                        Hk_plus_solver.piv,
                        Hk_minus_solver.piv,
                        Hk_z_solver.piv,
                    ]
                ),
            )

        self._derive_imm_homogeneous_data(Nm, Nkz, Nr)
        self._precompute_bulk_response(Nm, Nkz, Nr)

    def _derive_imm_homogeneous_data(self, Nm: int, Nkz: int, Nr: int) -> None:
        r"""Fill ``p1``, ``v_plus_1``, ``v_minus_1``,
        ``q_z_1``, and ``M_inv`` on-device.

        The pipe has a single wall at `$r = 1$` (last grid
        point), giving a `$1 \times 1$` influence matrix.

        Homogeneous data (4 solves):

        - `$L_k p_1 = e_1$` (unit RHS at wall)
        - `$H_{k,+} v_{+,1} = -(D_1 - m/r) p_1$`
        - `$H_{k,-} v_{-,1} = -(D_1 + m/r) p_1$`
        - `$H_{k,z} q_{z,1} = p_1$` (scalar potential for
          `$u_z$`: `$u_z^{(1)} = -i k_z q_{z,1}$`)

        The influence matrix (scalar per mode):

        .. math::
            M = D_{1,\mathrm{wall}} \cdot
            \frac{v_{+,1} + v_{-,1}}{2}

        measures `$\partial u_r / \partial r|_{\mathrm{wall}}$`.
        `$M^{-1} = 1/M$` for non-mean modes; `$M^{-1} = 0$`
        for `$(m, k_z) = (0, 0)$`.

        After the solves, the `$u_r$` part of ``v_plus_1``
        and ``v_minus_1`` is zeroed at the mean mode
        (continuity forces `$u_r \\equiv 0$` there), while
        preserving the `$u_\\theta$` part.
        """
        # Unit RHS at wall (last grid point).
        e_wall = (
            jnp.zeros(
                (Nm, Nkz, Nr),
                dtype=sharding.float_type,
                out_sharding=sharding.spec_imm_corr_shard,
            )
            .at[..., -1]
            .set(1.0)
        )
        self.p1 = self.Lk_op.solve(e_wall)

        # Pressure gradient components for the +/- equations.
        # (nabla p)_+ = D1 p - (m/r) p
        # (nabla p)_- = D1 p + (m/r) p
        # p1 has parity (-1)^m -> use parity-corrected D1.
        parity_sign_p = fourier.m_is_even * 2 - 1
        D1_p1 = jnp.einsum(
            "ij, mzj -> mzi", self.D1_pos, self.p1
        ) + parity_sign_p * jnp.einsum(
            "ij, mzj -> mzi", self.D1_ghost, self.p1
        )
        m_over_r = fourier.m * self.inv_r  # (Nm, 1, Nr)

        rhs_v_plus = -(D1_p1 - m_over_r * self.p1)
        rhs_v_minus = -(D1_p1 + m_over_r * self.p1)
        rhs_v_plus = rhs_v_plus.at[..., -1].set(0.0)
        rhs_v_minus = rhs_v_minus.at[..., -1].set(0.0)
        q_rhs = self.p1.at[..., -1].set(0.0)

        # Batched solve: component order (plus, minus, z).
        rhs_stack = jnp.stack([rhs_v_plus, rhs_v_minus, q_rhs])
        result_stack = self.Hk_op.solve(rhs_stack)
        self.v_plus_1 = result_stack[0]
        self.v_minus_1 = result_stack[1]
        self.q_z_1 = result_stack[2]

        # Zero the u_r part at the mean mode, preserving u_theta.
        vr_corr = jnp.where(
            fourier.k2_is_zero,
            (self.v_plus_1 + self.v_minus_1) / 2,
            0.0,
        )
        self.v_plus_1 = self.v_plus_1 - vr_corr
        self.v_minus_1 = self.v_minus_1 - vr_corr

        # 1x1 influence matrix:
        # M = D1_wall . (v_plus_1 + v_minus_1) / 2
        D1_wall_row = self.D1_wall.ravel()  # (Nr,)
        ur_1 = (self.v_plus_1 + self.v_minus_1) / 2
        M = jnp.einsum("j, mzj -> mz", D1_wall_row, ur_1)

        is_mean = fourier.k2_is_zero[..., 0]  # (Nm, Nkz)
        safe_M = jnp.where(is_mean, 1.0, M)
        self.M_inv = jnp.where(is_mean, 0.0, 1.0 / safe_M)

    def _precompute_bulk_response(self, Nm: int, Nkz: int, Nr: int) -> None:
        r"""Precompute the Helmholtz response for constant-bulk-
        velocity enforcement.

        Solves `$H_{k,z}\,h = \mathbf{1}$` (unit uniform RHS,
        zero wall BC) at the mean mode `$(m, k_z) = (0, 0)$`.
        The response `$h(r)$` is the velocity profile produced
        by a unit mean pressure gradient over one implicit time
        step.  Its bulk `$H = 2 \int_0^1 h\,r\,dr$` gives the
        scaling needed to zero the perturbation bulk velocity:

        .. math::
            G = -\frac{U_{b,\mathrm{pert}}}{H}, \qquad
            \bar{u}'_z \;\leftarrow\; \bar{u}'_z + G\,h

        which is equivalent to adding a uniform forcing `$G$`
        to the mean-mode `$u_z$` Helmholtz RHS before solving.
        """
        if params.phys.driving != "constant_bulk_velocity":
            self.h_bulk_response = jnp.zeros(
                Nr,
                dtype=sharding.float_type,
                out_sharding=sharding.no_shard,
            )
            self.H_bulk_inv = jnp.zeros((), dtype=sharding.float_type)
            return

        # Unit uniform RHS at the mean mode only, zero wall BC.
        ones_vec = jnp.ones(Nr, dtype=sharding.float_type).at[-1].set(0.0)
        rhs = jnp.where(fourier.k2_is_zero, ones_vec, 0.0)

        # Solve using the z-component (index 2) of the combined
        # Hk operator via a padded batch (one-time init cost).
        zeros = jnp.zeros_like(rhs)
        h_full = self.Hk_op.solve(jnp.stack([zeros, zeros, rhs]))[2]

        self.h_bulk_response = jax.device_put(
            extract_mean_mode(h_full[None])[0],
            sharding.no_shard,
        )
        H_bulk = 2 * jnp.dot(self.y_weights, self.h_bulk_response)
        self.H_bulk_inv = 1.0 / H_bulk


# ── Solver functions ─────────────────────────────────────────────


def _curl_fn(
    state: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> Array:
    r"""Spectral curl in cylindrical coordinates.

    Input/output in `$(u_z, u_r, u_\theta)$` representation.
    All operations are spectral multiplications (`$im$`,
    `$ik_z$`), diagonal scalings (`$1/r$`), and FD
    matrix-vector products (`$D_1$`).  Uses `$D_{1,pos}$`
    (parity-independent, since the curl is applied to fields
    that already have correct parity).

    .. math::
        \omega_r = \frac{im}{r}\,u_z - ik_z\,u_\theta

    .. math::
        \omega_\theta = ik_z\,u_r - D_1\,u_z

    .. math::
        \omega_z = D_1\,u_\theta + \frac{1}{r}\,u_\theta
                 - \frac{im}{r}\,u_r
    """
    uz, ur, utheta = state[0], state[1], state[2]

    im = 1j * fourier_.m
    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r  # (Nr,)

    # Parity signs: u_theta has parity (-1)^{m+1},
    # u_z has parity (-1)^m.
    parity_sign_p = fourier_.m_is_even * 2 - 1
    parity_sign_v = -parity_sign_p

    # Batch D1_pos and D1_ghost into two GEMMs.
    fields = jnp.stack([utheta, uz])
    dy_common = jnp.einsum("ij, cmzj -> cmzi", flow_.D1_pos, fields)
    dy_ghost = jnp.einsum("ij, cmzj -> cmzi", flow_.D1_ghost, fields)
    dy_utheta = dy_common[0] + parity_sign_v * dy_ghost[0]
    dy_uz = dy_common[1] + parity_sign_p * dy_ghost[1]

    omega_r = im * inv_r * uz - ikz * utheta
    omega_theta = ikz * ur - dy_uz
    omega_z = dy_utheta + inv_r * utheta - im * inv_r * ur

    return jnp.array([omega_z, omega_r, omega_theta])


def _get_rhs(
    state: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> Array:
    r"""Evaluate the nonlinear RHS in `$(u_z, u_+, u_-)$` form.

    1. Convert `$(u_z, u_+, u_-) \to (u_z, u_r, u_\theta)$`.
    2. Compute the rotational-form nonlinear term via
       :func:`~dnsjax.rhs.get_nonlin` with the cylindrical
       curl.
    3. Convert `$(NL_z, NL_r, NL_\theta)
       \to (NL_z, NL_+, NL_-)$`.
    """
    u_z, u_plus, u_minus = state[0], state[1], state[2]
    ur = (u_plus + u_minus) / 2
    utheta = -1j * (u_plus - u_minus) / 2

    state_rthz = jnp.array([u_z, ur, utheta])

    nonlin_rthz = get_nonlin(
        state_rthz,
        flow_.base_flow,
        flow_.curl_base_flow,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier_, flow_),
    )

    NL_z, NL_r, NL_theta = (
        nonlin_rthz[0],
        nonlin_rthz[1],
        nonlin_rthz[2],
    )
    NL_plus = NL_r + 1j * NL_theta
    NL_minus = NL_r - 1j * NL_theta

    return jnp.array([NL_z, NL_plus, NL_minus])


# ── Matrix-free matvecs ──────────────────────────────────────────


def _abase_matvec(
    u: Array,
    flow_: CylindricalFlow,
    parity_sign: Array,
) -> Array:
    r"""Apply `$A_{\mathrm{base}}^{(\sigma)} u$` matrix-free.

    .. math::
        A_{\mathrm{base}}^{(\sigma)} u
        = \underbrace{(D_{2,\mathrm{pos}} + (1/r)\,
          D_{1,\mathrm{pos}})\,u}_{\text{common part}}
        + (-1)^{m_{\mathrm{eff}}}
          \underbrace{(\widetilde{D}_{2,\mathrm{ghost}}
          + (1/r)\,\widetilde{D}_{1,\mathrm{ghost}})
          \,u}_{\text{ghost correction}}

    The ghost correction matrices have nonzero entries only
    in the first `$\sim p$` rows (near the pipe centre, where
    stencils cross `$r = 0$`).

    Parameters
    ----------
    u:
        Field, shape ``(Nm, Nkz, Nr)``.
    flow\_:
        Cylindrical flow data (uses ``D1_pos``,
        ``D2_pos``, ``D1_ghost``, ``D2_ghost``,
        ``inv_r``).
    parity_sign:
        `$(-1)^{m_{\mathrm{eff}}}$`, shape
        ``(Nm, 1, 1)``.
    """
    D2_u = jnp.einsum("ij, mzj -> mzi", flow_.D2_pos, u)
    D1_u = jnp.einsum("ij, mzj -> mzi", flow_.D1_pos, u)
    common = D2_u + flow_.inv_r * D1_u

    D2g_u = jnp.einsum("ij, mzj -> mzi", flow_.D2_ghost, u)
    D1g_u = jnp.einsum("ij, mzj -> mzi", flow_.D1_ghost, u)
    ghost = D2g_u + flow_.inv_r * D1g_u

    return common + parity_sign * ghost


def _lk_matvec(
    u: Array,
    flow_: CylindricalFlow,
    fourier_: Fourier,
) -> Array:
    r"""Apply `$L_k u$` for the pressure Poisson operator.

    Matrix-free evaluation:
    `$L_k u = A_{\mathrm{base}}^{(\sigma_p)} u
    - (m^2/r^2 + k_z^2) u$`, with Neumann wall row and
    mean-mode pin.

    Parity for pressure: `$(-1)^m$`, so parity_sign =
    ``m_is_even * 2 - 1`` (``+1`` for even, ``-1`` for odd).
    """
    parity_sign = fourier_.m_is_even * 2 - 1

    Abase_u = _abase_matvec(u, flow_, parity_sign)
    out = Abase_u - (fourier_.m2 * flow_.inv_r2 + fourier_.kz2) * u

    # Wall row: Neumann D1[-1,:] for non-mean, pin for mean.
    D1_wall_row = flow_.D1_wall.ravel()
    wall_val = jnp.einsum("j, mzj -> mz", D1_wall_row, u)
    bot = jnp.where(fourier_.k2_is_zero[..., 0], u[..., -1], wall_val)
    return out.at[..., -1].set(bot)


# ── IMM iteration (1x1) ─────────────────────────────────────────


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> tuple[Array, Array]:
    r"""Influence-matrix method for cylindrical geometry.

    The pipe's single wall at `$r = 1$` gives a `$1 \times 1$`
    influence matrix (scalar `$\alpha$` per mode).

    Six stages (plus mean-mode projections):

    1. **Poisson RHS**: cylindrical divergence of momentum in
       `$(u_z, u_+, u_-)$` components:

       .. math::
           \nabla\!\cdot\!\mathbf{u}
           = \frac{D_1 u_+ + (m+1)/r\;u_+}{2}
           + \frac{D_1 u_- + (1-m)/r\;u_-}{2}
           + ik_z\,u_z

    2. **Particular pressure**: `$L_k p_P = \hat{f}_P$` with
       zero Neumann wall row.
    3. **Helmholtz solves**: three separate solves with
       `$H_{k,+}$`, `$H_{k,-}$`, `$H_{k,z}$`.  Pressure
       gradient in `$(+, -, z)$`:

       .. math::
           (\nabla p)_+ = D_1 p - (m/r)\,p, \quad
           (\nabla p)_- = D_1 p + (m/r)\,p, \quad
           (\nabla p)_z = ik_z\,p

    4. **Wall divergence residual**:
       `$d_{\mathrm{wall}} = D_{1,\mathrm{wall}}
       \cdot (u_{+,arb} + u_{-,arb})/2$`
    5. **Influence matrix**: `$\alpha = -M^{-1} d_{\mathrm{wall}}$`
    6. **Correction**:
       `$u_+ = u_{+,arb} + \alpha\,v_{+,1}$`,
       `$u_- = u_{-,arb} + \alpha\,v_{-,1}$`,
       `$u_z = u_{z,arb} - ik_z\,\alpha\,q_{z,1}$`.
    7. **Zero mean-mode** `$u_r$`: continuity
       `$(1/r)\,\partial(r u_r)/\partial r = 0$` plus
       no-slip at `$r = 1$` forces `$u_r \equiv 0$` at the
       mean mode.  The `$u_\theta$` part of `$u_\pm$` is
       preserved.
    8. *(optional)* If ``constant_bulk_velocity``, zero the
       mean-mode perturbation bulk `$u_z$`.
    """
    c = params.step.implicitness
    dt = params.step.dt
    nu = 1.0 / params.phys.re

    uz_n, up_n, um_n = velocity_n[0], velocity_n[1], velocity_n[2]
    NLz_n, NLp_n, NLm_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    NLz_j, NLp_j, NLm_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r
    m = fourier_.m

    # Parity signs for each component type.
    parity_sign_p = fourier_.m_is_even * 2 - 1  # (-1)^m
    parity_sign_v = -parity_sign_p  # (-1)^{m+1}

    m_plus_1_sq = (m + 1) ** 2
    m_minus_1_sq = (m - 1) ** 2
    m_sq = fourier_.m2

    # Batch all D1 y-derivatives with (-1)^{m+1} parity into
    # one GEMM each for D1_pos and D1_ghost (2 instead of 4).
    all_vparity = jnp.stack([up_n, um_n, NLp_j, NLp_n, NLm_j, NLm_n])
    dy_common = jnp.einsum("ij, cmzj -> cmzi", flow_.D1_pos, all_vparity)
    dy_ghost = jnp.einsum("ij, cmzj -> cmzi", flow_.D1_ghost, all_vparity)
    dy_all = dy_common + parity_sign_v * dy_ghost

    # Cylindrical divergence at time n.
    div_n = (
        (dy_all[0] + (m + 1) * inv_r * up_n) / 2
        + (dy_all[1] + (1 - m) * inv_r * um_n) / 2
        + ikz * uz_n
    )

    # Divergence of nonlinear terms at times n and j.
    div_NLj = (
        (dy_all[2] + (m + 1) * inv_r * NLp_j) / 2
        + (dy_all[4] + (1 - m) * inv_r * NLm_j) / 2
        + ikz * NLz_j
    )
    div_NLn = (
        (dy_all[3] + (m + 1) * inv_r * NLp_n) / 2
        + (dy_all[5] + (1 - m) * inv_r * NLm_n) / 2
        + ikz * NLz_n
    )

    Lk_d = _lk_matvec(div_n, flow_, fourier_)

    f_hat = div_n / dt + c * div_NLj + (1 - c) * div_NLn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure.
    f_hat_P = f_hat.at[..., -1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: Helmholtz solves for each component.
    # Batch the D1 derivatives of pP and vel_n into shared
    # GEMMs (2 D1 GEMMs instead of 4 separate ones).
    vel_n_stack = jnp.stack([up_n, um_n, uz_n])
    pP_and_vel = jnp.concatenate([pP[None], vel_n_stack])
    D1_batch = jnp.einsum("ij, cmzj -> cmzi", flow_.D1_pos, pP_and_vel)
    D1g_batch = jnp.einsum("ij, cmzj -> cmzi", flow_.D1_ghost, pP_and_vel)

    # pP pressure gradient (parity (-1)^m -> parity_sign_p).
    D1_pP = D1_batch[0] + parity_sign_p * D1g_batch[0]
    m_over_r = m * inv_r

    grad_pP_plus = D1_pP - m_over_r * pP
    grad_pP_minus = D1_pP + m_over_r * pP
    grad_pP_z = ikz * pP

    # Batched `$H_k^-$` matvec for all three components.
    D1_vel = D1_batch[1:]
    D1g_vel = D1g_batch[1:]
    D2_all = jnp.einsum("ij, cmzj -> cmzi", flow_.D2_pos, vel_n_stack)
    D2g_all = jnp.einsum("ij, cmzj -> cmzi", flow_.D2_ghost, vel_n_stack)
    common_hk = D2_all + flow_.inv_r * D1_vel
    ghost_hk = D2g_all + flow_.inv_r * D1g_vel
    parity_hk = jnp.stack([parity_sign_v, parity_sign_v, parity_sign_p])
    Abase_stack = common_hk + parity_hk * ghost_hk
    meff2_stack = jnp.stack([m_plus_1_sq, m_minus_1_sq, m_sq])
    lapl_stack = (
        Abase_stack - (meff2_stack * flow_.inv_r2 + fourier_.kz2) * vel_n_stack
    )
    Hk_minus_stack = (1.0 / dt) * vel_n_stack + (1.0 - c) * nu * lapl_stack
    Hk_minus_stack = Hk_minus_stack.at[..., -1].set(vel_n_stack[..., -1])

    R_stack = (
        Hk_minus_stack
        - jnp.stack([grad_pP_plus, grad_pP_minus, grad_pP_z])
        + c * jnp.stack([NLp_j, NLm_j, NLz_j])
        + (1 - c) * jnp.stack([NLp_n, NLm_n, NLz_n])
    )

    # Zero wall BC (Dirichlet no-slip).
    R_stack = R_stack.at[..., -1].set(0.0)

    # Zero the u_r part of the +/- RHS at the mean mode so
    # the Helmholtz solves produce u_r = 0 there.  At m=0,
    # Hk_plus and Hk_minus are identical (m_eff^2 = 1, same
    # parity), so the antisymmetric RHS gives up = -um.
    Rr_corr = jnp.where(
        fourier_.k2_is_zero, (R_stack[0] + R_stack[1]) / 2, 0.0
    )
    R_stack = R_stack.at[0].add(-Rr_corr)
    R_stack = R_stack.at[1].add(-Rr_corr)

    # Batched Helmholtz solve: component order (plus, minus, z).
    arb_stack = flow_.Hk_op.solve(R_stack)
    up_arb, um_arb, uz_arb = arb_stack[0], arb_stack[1], arb_stack[2]

    # Stage 4: wall divergence residual.
    D1_wall_row = flow_.D1_wall.ravel()
    ur_arb = (up_arb + um_arb) / 2
    d_wall = jnp.einsum("j, mzj -> mz", D1_wall_row, ur_arb)

    # Mean mode: pressure is a gauge; zero the residual.
    d_wall = jnp.where(fourier_.k2_is_zero[..., 0], 0.0, d_wall)

    # Stage 5: influence matrix correction (scalar per mode).
    alpha = -flow_.M_inv * d_wall  # (Nm, Nkz)
    alpha = alpha[..., None]  # (Nm, Nkz, 1)

    # Stage 6: corrected velocity.
    up_new = up_arb + alpha * flow_.v_plus_1
    um_new = um_arb + alpha * flow_.v_minus_1

    # Stage 7: zero mean-mode u_r, preserving u_theta.
    ur_corr = jnp.where(fourier_.k2_is_zero, (up_new + um_new) / 2, 0.0)
    up_new = up_new - ur_corr
    um_new = um_new - ur_corr

    # Constant-bulk-velocity enforcement: add a uniform mean
    # pressure gradient G to the mean-mode u_z Helmholtz RHS
    # so that the perturbation bulk velocity is zero.
    # Equivalent post-solve form: uz += G * h, where
    # h = Hk_z^{-1} [1,...,1,0] and G = -Ub_pert / H_bulk.
    # At the mean mode alpha = 0 and ikz = 0, so uz_arb
    # already equals the uncorrected uz_new there; reading
    # the bulk from uz_arb lets the IMM correction and the
    # bulk correction fuse into a single expression.
    if params.phys.driving == "constant_bulk_velocity":
        mean_uz = extract_mean_mode(uz_arb[None])[0].real
        bulk_uz = 2 * jnp.dot(flow_.y_weights, mean_uz)
        uz_new = (
            uz_arb
            - ikz * alpha * flow_.q_z_1
            + jnp.where(
                fourier_.k2_is_zero,
                -bulk_uz * flow_.H_bulk_inv * flow_.h_bulk_response,
                0.0,
            )
        )
    else:
        uz_new = uz_arb - ikz * alpha * flow_.q_z_1

    velocity_new = jnp.array([uz_new, up_new, um_new])
    correction = velocity_new - velocity_j

    return velocity_new, correction


def _predict(
    velocity_n: Array,
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> Array:
    """Euler predictor via the cylindrical IMM."""
    nonlin_n = rhs_no_lapl
    prediction_state, _ = _imm_iteration(
        velocity_n, velocity_n, nonlin_n, nonlin_n, fourier_, flow_
    )
    return prediction_state


def _correct(
    state_prev: Array,
    prediction_state: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector via the cylindrical IMM."""
    prediction_state_new, correction = _imm_iteration(
        state_prev,
        prediction_state,
        rhs_prev,
        rhs_next,
        fourier_,
        flow_,
    )
    return prediction_state_new, correction


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> Array:
    """L2 convergence norm with cylindrical weighting."""
    return jnp.sqrt(
        get_norm2_cyl(correction, fourier_.k_metric, flow_.y_weights)
    )


# ── Stepper factory ─────────────────────────────────────────────


def build_cylindrical_stepper(
    flow: CylindricalFlow,
) -> tuple[
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array, Array, Array], tuple[Array, Array, Array]],
    Callable[[str | None], Array],
    Callable[[Array], tuple[Array, Array, Array]],
]:
    """Build time-stepping functions for a cylindrical flow.

    Returns ``(predict_and_correct, iterate_correction,
    init_state_bound, predict_and_fully_correct)`` with the
    ``fourier`` and *flow* singletons already bound.
    """
    return build_wall_bounded_stepper(
        _get_rhs, _predict, _correct, _norm, fourier, flow
    )
