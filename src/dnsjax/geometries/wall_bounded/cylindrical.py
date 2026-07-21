r"""Cylindrical geometry: Fourier class, norms, IMM, and solvers.

Provides all geometry-general infrastructure for wall-bounded
cylindrical flows: the ``Fourier`` wavenumber class, the
``CylindricalFlow`` base dataclass (radial CGL grid -- half-CGL
under the default ``iterative-cn`` scheme, rigged-CGL under
``cnab2``, selected by ``geo.grid_type``, parity-reduced FD
matrices, IMM operators),
spectral solvers (influence-matrix method, predictor-corrector
time stepping), and diagnostic helpers (norms, perturbation
energy, centreline interpolation).

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

This is the solver's **working** basis: the state carried between
steps, the RHS, the corrector iterates and every operator below
live in `$(u_z, u_+, u_-)$`.  It is not what anything outside the
time stepper sees -- snapshots, diagnostics, probes, initial
conditions and the analysis package all work in the physical
triad `$(u_z, u_r, u_\theta)$`, and a given state crosses between
the two at most once, at that boundary (``to_pm_basis`` /
``from_pm_basis`` in ``_base.py``, driven by
:mod:`dnsjax.__main__`; the wall-bounded ``CLAUDE.md``).

:func:`_get_rhs_core` and :func:`_l_bf` convert internally because
the real FFT demands it: every physical component is the
transform of a real field and is individually Hermitian-symmetric,
whereas `$u_\pm$` are not
(`$\overline{\hat u_+(k)} = \hat u_-(-k)$`).  The physical-space
fields -- and hence the CFL measurement -- are therefore always
`$(u_z, u_r, u_\theta)$`.

Despite having different `$m_{\mathrm{eff}}$` values, `$u_+$`
and `$u_-$` share the **same parity** `$(-1)^{m+1}$` -- that of
`$u_r$` and `$u_\theta$`, preserved by the pointwise mixing.
Parity is a kinematic property (how a field transforms under
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
- `$u_r$` and `$u_\theta$` share that same `$(-1)^{m+1}$`
  class, so the physical-basis diagnostics and the resume
  regrid use the parity machinery unchanged.

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

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ...fd import (
    axis_extrapolation_weights,
    build_diff_matrices,
    build_integration_weights,
    cgl_radial_quadrature_weights,
    local_grid_spacing,
    matrix_half_bandwidth,
    tanh_one_sided_grid,
)
from ...measurements import get_cfl
from ...operators import (
    complex_harmonics,
    pad_harmonics,
    phys_to_spec_2d,
    real_harmonics,
    spec_to_phys_2d,
)
from ...parameters import derived_params, params
from ...rhs import get_nonlin
from ...sharding import register_dataclass_pytree, sharding
from ...solvers import (
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _assemble_banded_operator,
    _banded_diag_column,
    _banded_from_dense,
    _banded_wall_row,
    _build_pallas_operator,
    _factor_pallas_operator,
)
from ._base import (
    apply_y_matrix,
    base_flow_coupling,
    build_wall_bounded_stepper,
    extract_mean_mode,
    from_pm_basis,
    frozen_profile_flow,  # noqa: F401 — re-exported
    get_inprod,  # noqa: F401 — re-exported
    get_norm,  # noqa: F401 — re-exported
    get_norm2,
    init_state,  # noqa: F401 — re-exported
    integrate_scalar,
    pad_base_flow,  # noqa: F401 — re-exported
    phys_to_spec,  # noqa: F401 — re-exported
    spec_to_phys,  # noqa: F401 — re-exported
    to_pm_basis,
)

#: Role aliases for the basis boundary.  The flow modules re-export
#: these so :mod:`dnsjax.__main__` can move a state between the
#: physical representation every consumer sees and the solver basis,
#: without knowing which geometry it is driving (Cartesian and
#: triply-periodic simply have none).
to_solver_basis = to_pm_basis
from_solver_basis = from_pm_basis


def _ghost_row_count(D1_ghost: np.ndarray, D2_ghost: np.ndarray) -> int:
    r"""Number of leading nonzero rows of the ghost matrices.

    Stencils cross `$r = 0$` only for the first
    `$\sim (p+2)//2$` radial points, so all later rows of the
    ghost corrections vanish and need not be stored or applied.
    """
    nz = np.nonzero(
        np.any(D1_ghost != 0.0, axis=1) | np.any(D2_ghost != 0.0, axis=1)
    )[0]
    return int(nz[-1]) + 1 if nz.size else 1


@register_dataclass_pytree
@dataclass
class Fourier:
    r"""Wavenumber grids for the cylindrical geometry.

    Broadcasting shapes match the spectral layout
    ``(Nr, Nm, Nkz)`` = ``(ny, nz-1, nx//2)``:

    - ``kz``: shape ``(1, 1, nx//2)`` -- axial wavenumber
      (real FFT on the streamwise ``x`` parameter direction).
    - ``m``: shape ``(1, nz-1, 1)`` -- azimuthal mode number
      (complex FFT on the ``z`` parameter direction with
      `$l_z = 2\pi/m_0$`); the resolved modes are the integer
      multiples `$m = m_0 j$` of the wedge fundamental `$m_0$`
      (``geo.m0``; `$m_0 = 1$` is the full circle).

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
    the real FFT (padding columns get 2 — inert, they only
    ever weight zero fields).

    ``m_is_even`` is a boolean mask ``(1, nz_spec, 1)``
    selecting the azimuthal modes where `$m$` is even, used
    to choose the correct parity-reduced FD matrices.  At
    padding slots it follows the parity of the placeholder
    `$m$` values; the selected operators are regular either
    way.

    Padding slots (``nz_spec > nz - 1`` or
    ``nx_spec > nx // 2``, spectral padding for 2D
    divisibility) carry nonzero beyond-resolution
    placeholder wavenumbers (see ``pad_harmonics`` in
    :mod:`dnsjax.operators`): every per-mode operator
    assembled at a padding slot is regular, and the fields
    there are identically zero (the forward FFT re-zeroes
    the padding slots on every evaluation), so the padding
    modes need no special-casing.

    ``mean_mask`` is a boolean mask that is ``True`` only at
    the mean mode `$(m, k_z) = (0, 0)$` (global index
    ``(0, 0)``; padding modes are appended at the end).  The
    mean mode is the only `$m^2 + k_z^2 = 0$` mode, so this
    single mask serves the operator pin row, the
    influence-matrix mean branch, and all mean-mode physics
    (projections and the constant-bulk-velocity write).
    """

    kz: Array = field(init=False)
    m: Array = field(init=False)
    k_metric: Array = field(init=False)
    kz2: Array = field(init=False)
    m2: Array = field(init=False)
    m_is_even: Array = field(init=False)
    mean_mask: Array = field(init=False)

    def __post_init__(self) -> None:
        kz_vals = (
            pad_harmonics(
                real_harmonics(params.res.nx),
                params.res.nx,
                sharding.nx_spec_pad,
            )
            * 2
            * jnp.pi
            / params.geo.lx
        )
        self.kz = jax.device_put(
            kz_vals.reshape([1, 1, -1]).astype(sharding.float_type),
            P(None, None, sharding.a1),
        )

        # Azimuthal wavenumbers m = m0 * harmonic over the wedge
        # l_z = 2*pi/m0 (m0 = 1 is the full circle).  The integer
        # multiply is exact and keeps the padding placeholders nonzero;
        # ``m_is_even`` below then tracks the parity of the *physical* m,
        # i.e. the correct r = 0 axis-regularity condition per mode.
        m_vals = (
            pad_harmonics(
                complex_harmonics(params.res.nz),
                params.res.nz,
                sharding.nz_spec_pad,
            )
            * params.geo.m0
        )
        self.m = jax.device_put(
            m_vals.reshape([1, -1, 1]).astype(sharding.float_type),
            P(None, sharding.a0, None),
        )

        self.k_metric = jnp.where(self.kz == 0, 1, 2).astype(
            sharding.float_type
        )

        self.kz2 = self.kz**2
        self.m2 = self.m**2
        self.m_is_even = (self.m % 2 == 0).astype(sharding.float_type)

        # One-hot at the mean mode (m, kz) = (0, 0): the true
        # modes precede the padding, so it is global index (0, 0).
        # The mean mode is the only m^2 + kz^2 = 0 mode (padding
        # slots carry nonzero placeholder wavenumbers).
        e_m = (
            jnp.zeros(m_vals.shape[0], dtype=sharding.float_type)
            .at[0]
            .set(1.0)
        )
        e_kz = (
            jnp.zeros(kz_vals.shape[0], dtype=sharding.float_type)
            .at[0]
            .set(1.0)
        )
        self.mean_mask = (
            jax.device_put(e_m.reshape([1, -1, 1]), P(None, sharding.a0, None))
            * jax.device_put(
                e_kz.reshape([1, 1, -1]), P(None, None, sharding.a1)
            )
        ) == 1.0


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
    split into radial-derivative, azimuthal, and axial terms.
    The azimuthal term is the covariant azimuthal gradient in
    `$(u_z, u_r, u_\theta)$` components,

    .. math::
        \frac{|im\,u_z|^2 + |im\,u_r - u_\theta|^2
        + |im\,u_\theta + u_r|^2}{r^2},

    pointwise equal to the `$m_{\mathrm{eff}}$`-diagonal form of the
    solver-interior decoupled basis
    (`$|m u_z|^2 + \tfrac{1}{2}|(m{+}1)u_+|^2 +
    \tfrac{1}{2}|(m{-}1)u_-|^2$`).
    The radial derivative uses parity-dependent FD matrices:
    `$D_1 = D_{1,\mathrm{pos}} + (-1)^{m_{\mathrm{eff}}}
    D_{1,\mathrm{ghost}}$` (with `$u_r$`, `$u_\theta$` sharing the
    `$(-1)^{m+1}$` parity class of `$u_\pm$`).

    Parameters
    ----------
    state:
        Spectral velocity in `$(u_z, u_r, u_\theta)$` form,
        shape ``(3, Nr, Nm, Nkz)``.
    D1_pos:
        Common part of first-derivative FD matrix.
    D1_ghost:
        Ghost correction for `$D_1$`, row-sliced to its
        `$g$` nonzero rows: shape ``(g, Nr)``.
    m_is_even:
        Boolean mask for even `$m$`, shape ``(1, Nm, 1)``.
    inv_r:
        `$1/r$` on the radial grid.
    m:
        Azimuthal mode number, shape ``(1, Nm, 1)``.
    kz2:
        `$k_z^2$`, shape ``(1, 1, Nkz)``.
    k_metric:
        Hermitian-symmetry weight for the real FFT axis.
    y_weights:
        Radial integration weights `$w_j r_j$`.
    """
    # Parity signs: u_z has parity (-1)^m, u_r/u_theta (-1)^{m+1}.
    p_sign_z = m_is_even * 2 - 1
    p_sign_v = -p_sign_z

    # Batched D1 matvecs (2 GEMMs for all 3 components; the
    # ghost GEMM covers only its g nonzero rows).
    g = D1_ghost.shape[0]
    dy_pos = apply_y_matrix(D1_pos, state)
    dy_ghost = apply_y_matrix(D1_ghost, state)
    p_signs = jnp.stack([p_sign_z, p_sign_v, p_sign_v])
    dy_state = dy_pos.at[:, :g].add(p_signs * dy_ghost)

    enstrophy_D1 = get_norm2_cyl(dy_state, k_metric, y_weights)

    # Azimuthal term: covariant azimuthal gradient over r.
    inv_r_3d = inv_r[:, None, None]
    im = 1j * m
    state_m = jnp.stack(
        [
            im * inv_r_3d * state[0],
            inv_r_3d * (im * state[1] - state[2]),
            inv_r_3d * (im * state[2] + state[1]),
        ]
    )
    enstrophy_m = get_norm2_cyl(state_m, k_metric, y_weights)

    # Axial term: kz^2 |u|^2.
    enstrophy_kz = get_norm2_cyl(state, kz2 * k_metric, y_weights)

    return enstrophy_D1 + enstrophy_m + enstrophy_kz


def get_norm2_cyl(state: Array, k_metric: Array, y_weights: Array) -> Array:
    r"""Cylindrical squared L2 norm for `$(u_z, u_r, u_\theta)$`.

    The component axis is a pointwise orthonormal physical triad, so
    this is the plain component sum of the shared :func:`get_norm2`;
    kept as a named wrapper for signature symmetry with the other
    geometry norms (the radial Jacobian `$r$` lives in *y_weights*).

    Parameters
    ----------
    state:
        Spectral velocity in `$(u_z, u_r, u_\theta)$` form,
        shape ``(3, Nr, Nm, Nkz)`` (any component count works).
    k_metric:
        Hermitian-symmetry weight for the real FFT axis.
    y_weights:
        Radial integration weights `$w_j r_j$`.
    """
    return get_norm2(state, k_metric, y_weights)


# ── Half-diameter grid and parity-reduced FD matrices ──────────────


def build_radial_cgl_grid(Nr: int, axis_gap: int = 1) -> Array:
    r"""Build the radial CGL grid on `$(0, 1]$` (rigged or half).

    Takes the `$N_r$` outermost positive points of a
    `$(2 N_r + g)$`-point CGL grid on `$[-1, 1]$`
    (`$g$` = *axis_gap* `$\in \{0, 1\}$`):

    .. math::
        s_j = -\cos\!\bigl(j\pi/(2N_r + g - 1)\bigr),
        \quad j = N_r + g, \ldots, 2N_r + g - 1,

    giving `$r_0 < r_1 < \cdots < r_{N_r-1} = 1$` with CGL
    clustering near the pipe wall, near-uniform spacing
    `$\Delta r \approx \pi/(2 N_r)$` near the centre, and
    innermost point

    .. math::
        r_0 = \sin\!\Bigl(\frac{(g+1)\,\pi}{2\,(2N_r+g-1)}\Bigr)
        \approx (g+1)\,\frac{\Delta r}{2}.

    - `$g = 1$` -- the **rigged-CGL** grid (the ``cnab2``
      default).  The odd auxiliary total has a centre point
      exactly on `$r = 0$` (a coordinate singularity, not a
      boundary) which is dropped, landing
      `$r_0 \approx \Delta r$`.
    - `$g = 0$` -- the **half-CGL** grid (the ``iterative-cn``
      default; even auxiliary total, no point on the axis,
      staggered `$r_0 \approx \Delta r/2$`).

    No degree of freedom lives in `$[0, r_0)$` (the parity
    ghosts close the FD stencils across the axis and the
    quadrature covers the segment via the parity-specific
    spectral rule in :func:`build_cylindrical_grid` /
    :func:`~dnsjax.fd.cgl_radial_quadrature_weights`), so
    `$r_0$` is a free discretisation choice.  It bounds the near-axis azimuthal
    advection CFL `$\propto 1/r_0$` -- the pipe's explicit
    (cnab2) timestep limit -- so the rigged grid's
    `$2\times$`-larger `$r_0$` doubles the admissible cnab2
    ``dt`` (measured), which is why it is the ``cnab2``
    default; the tighter half-CGL axis destabilises cnab2 (a
    near-axis explicit instability) and is restricted to
    ``iterative-cn`` (``geo.grid_type = "half-cgl"``), which
    integrates it cleanly, gains its finer near-axis
    resolution, and defaults to it.

    Parameters
    ----------
    Nr:
        Number of radial grid points kept.
    axis_gap:
        `$0$` = half-CGL, `$1$` = rigged-CGL.  Selected from
        ``geo.grid_type`` by :func:`build_cylindrical_grid`
        (not a user-facing config field).

    Returns
    -------
    :
        Radial grid array, shape ``(Nr,)``, ascending, all
        `$r > 0$`, last point `$r = 1$`.
    """
    N_full = 2 * Nr + axis_gap
    s = -jnp.cos(
        jnp.arange(N_full, dtype=sharding.float_type) * jnp.pi / (N_full - 1)
    )
    return s[Nr + axis_gap :]


def build_parity_reduced_matrices(
    rs: Array, p: int, consistent_imm: bool = False
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

    With *consistent_imm* the matrices are instead built from a single
    Fornberg fit on `$x = r^2$` (an axis-regular field is analytic in
    `$x$`):

    .. math::
        D_{1,\mathrm{even}} = 2\,\mathrm{diag}(r)\,D_x, \qquad
        D_{1,\mathrm{odd}} = S
          + \mathrm{diag}(r)\,D_{1,\mathrm{even}}\,S, \quad
        S = \mathrm{diag}(1/r),

    whose discrete `$1/r$` commutator
    `$D_{1,\mathrm{even}}S - S D_{1,\mathrm{odd}} + S^2$` vanishes to
    round-off -- the axis identity the influence-matrix continuity
    argument needs (the mirrored fold above leaves it
    `$O(10^2\text{--}10^3)$`).  `$D_2$` is then the **composed**
    `$D_1 D_1$`, parity-flipping (even data
    `$\to D_{1,\mathrm{even}} \to$` odd `$\to D_{1,\mathrm{odd}} \to$`
    even, and vice versa), so `$\nabla\!\cdot\!\nabla = L_k$` holds
    discretely; the common part is the parity average
    `$D_{\mathrm{pos}} = (D_{\mathrm{even}} + D_{\mathrm{odd}})/2$`, so
    the same ghost machinery applies -- but the ghost
    `$(D_{\mathrm{even}} - D_{\mathrm{odd}})/2$` is now full, not
    near-axis-sparse, and the band widens (`$A_{\mathrm{base}}$` 12 vs 8
    at ``fd_order = 8``; the assembler measures it).  Off by default and
    only ever reached through ``res.consistent_imm`` (the
    ``Resolution.consistent_imm`` docs carry the trade); see the pipe
    branch of :func:`_imm_iteration`.

    Returns
    -------
    D1_even, D2_even:
        Parity-reduced matrices for even `$m_{\mathrm{eff}}$`.
    D1_odd, D2_odd:
        Parity-reduced matrices for odd `$m_{\mathrm{eff}}$`.
    D1_pos, D2_pos:
        Common (parity-independent) part: positive-row,
        positive-column block of the full-grid matrices (the parity
        average under *consistent_imm*).
    """
    Nr = len(rs)
    if consistent_imm:
        # Single Fornberg fit on x = r^2 (see the docstring): the
        # resulting even/odd D1 pair makes the near-axis 1/r commutator
        # exact, and D2 is the composed D1.D1 for discrete continuity.
        rs_np = np.asarray(rs)
        DX, _ = build_diff_matrices(rs_np**2, p)
        R = np.diag(rs_np)
        S = np.diag(1.0 / rs_np)
        D1_even = 2.0 * (R @ DX)
        D1_odd = S + R @ D1_even @ S
        D2_even = D1_odd @ D1_even  # even -> odd -> even
        D2_odd = D1_even @ D1_odd  # odd -> even -> odd
        D1_pos = (D1_even + D1_odd) / 2
        D2_pos = (D2_even + D2_odd) / 2
        return D1_even, D2_even, D1_odd, D2_odd, D1_pos, D2_pos

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


def build_cylindrical_grid(
    ny: int,
    fd_order: int,
    wall_grid: str | None = None,
    grid_type: str | None = None,
    grid_stretch: float = 1.5,
    consistent_imm: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    r"""Build radial grid, parity-reduced D1 matrices, weights,
    and `$1/r$` for the cylindrical geometry.

    Grid selection (precedence):

    1. *wall_grid*: load from file (a custom grid always
       overrides dnsjax's grid generation).
    2. *grid_type*: ``"half-tanh"`` for one-sided tanh stretching
       (outer wall only -- there is no inner wall); ``"half-cgl"``
       for the half-CGL radial grid (``axis_gap = 0``);
       ``"rigged-cgl"`` / ``None`` for the rigged-CGL radial grid
       (``axis_gap = 1``).  The Cartesian/annular names
       (``"cgl"``/``"tanh"``) are rejected.
    3. ``update_parameters`` resolves an unset ``geo.grid_type``
       from the pipe spec: ``"half-cgl"`` under ``iterative-cn``
       and ``"rigged-cgl"`` under ``cnab2``, so params-driven
       callers pass a concrete value; a raw ``None`` here falls
       back to rigged-CGL.

    See :func:`build_radial_cgl_grid` for the rigged vs half-CGL
    construction and the near-axis-CFL rationale.

    Parameters
    ----------
    ny:
        Number of radial grid points (`$N_r$`).
    fd_order:
        Finite-difference stencil half-bandwidth.
    wall_grid:
        Optional path to a custom radial grid file.
        File format: one coordinate per line in
        wall-to-interior order (first line = pipe wall
        `$r = 1$`, last line = closest to centre).
        All `$r > 0$`; `$r = 0$` is excluded.  The code
        reverses to ascending order internally.
    grid_type:
        Named grid type (``"rigged-cgl"`` / ``None`` = rigged-CGL,
        ``"half-cgl"``, or ``"half-tanh"``).
    grid_stretch:
        Stretching parameter for ``grid_type="half-tanh"``.
    consistent_imm:
        ``params.res.consistent_imm``: return the `$x = r^2$`
        parity-reduced ``D1`` pair (see
        :func:`build_parity_reduced_matrices`) instead of the mirrored
        fold, so a consumer that reconstructs the divergence operator
        (the random-IC generator, the analysis package) matches the
        gated solver.  Off by default.

    Returns
    -------
    rs:
        Radial grid on `$(0, 1]$`, shape ``(ny,)``.
    D1_even:
        Even-parity first-derivative matrix, ``(ny, ny)``.
    D1_odd:
        Odd-parity first-derivative matrix, ``(ny, ny)``.
    D1_pos:
        Common (parity-independent) part, ``(ny, ny)``.
    y_weights:
        **Even-parity** radial quadrature weights, shape ``(ny,)``,
        `$\sum_j W_j f_j \approx \int_0^1 f\,r\,dr$` over the full
        disc for an *even* integrand `$f$` -- the energy norm
        (`$|u|^2$`), mean `$u_z$`, dissipation.  On a detected radial
        CGL grid these are the spectral Clenshaw-Curtis-with-weight
        `$r$` weights that bake in the `$r = 0$` reconstruction
        (:func:`~dnsjax.fd.cgl_radial_quadrature_weights`); on a
        custom / tanh grid the parity-agnostic axis-augmented
        composite rule (`$g = f r$` on `$[0, r_0, \ldots]$`, the axis
        a free node since `$g(0) = 0$`).  Strictly positive (a
        definite energy norm), verified at build.
    y_weights_odd:
        **Odd-parity** radial quadrature weights, shape ``(ny,)``,
        for an *odd* integrand (the mean `$u_\theta$`); equal to
        ``y_weights`` on custom / tanh grids (the composite rule is
        parity-agnostic).  A single vector cannot be spectral for
        both parities, so each diagnostic uses the vector matching
        its known parity.
    inv_r:
        `$1/r$` on the grid, shape ``(ny,)``.
    """
    if wall_grid is not None:
        grid_raw = np.loadtxt(wall_grid, dtype=np.float64)
        if len(grid_raw) != ny:
            raise ValueError(
                f"Wall grid file has {len(grid_raw)} points, expected ny={ny}"
            )
        grid = grid_raw[::-1].copy()
        if not np.isclose(grid[-1], 1.0):
            raise ValueError(
                f"Cylindrical wall grid must end at r=1 (got r[-1]={grid[-1]})"
            )
        if grid[0] <= 0.0:
            raise ValueError(
                "Cylindrical wall grid must have all"
                f" r > 0 (got r[0]={grid[0]})"
            )
        rs = jnp.asarray(grid, dtype=sharding.float_type)
    elif grid_type == "half-tanh":
        grid = tanh_one_sided_grid(ny, grid_stretch)
        rs = jnp.asarray(grid, dtype=sharding.float_type)
    elif grid_type in ("half-cgl", "rigged-cgl", None):
        # "rigged-cgl" / None -> rigged (axis_gap = 1); "half-cgl" ->
        # the staggered half grid (axis_gap = 0).  The resolved
        # default is always concrete (pipe spec: half-cgl under
        # iterative-cn, rigged-cgl under cnab2).
        axis_gap = 0 if grid_type == "half-cgl" else 1
        rs = build_radial_cgl_grid(ny, axis_gap)
    else:
        # The Cartesian/annular names ("cgl"/"tanh") do not select a
        # cylindrical radial grid; validate_parameters rejects them
        # upstream -- this guards direct callers.
        raise ValueError(
            f"grid_type {grid_type!r} is not a cylindrical radial "
            "grid; choose 'half-cgl', 'rigged-cgl', or 'half-tanh'."
        )
    inv_r = 1.0 / rs
    rs_np = np.asarray(rs)
    # Full-disc quadrature int_0^1 f r dr with no axis grid point.
    qc = cgl_radial_quadrature_weights(rs_np, fd_order)
    if qc is not None:
        # Detected radial CGL grid (rigged / half): spectral
        # parity-specific weights, baking in the r=0 reconstruction
        # (positive).  A single vector cannot be spectral for both
        # parities, so w_even serves the energy norm and even
        # integrands (mean u_z, dissipation), w_odd the odd mean
        # u_theta -- the caller picks by each diagnostic's known
        # parity.  See fd.cgl_radial_quadrature_weights.
        w_even_np, w_odd_np = qc
    else:
        # Custom / tanh grid: the parity-agnostic axis-augmented
        # composite rule (integrate g = f*r on [0, *rs] with the axis
        # r=0 as a free node, g(0)=0 for any bounded f; fd_order,
        # positive, correct for either parity).
        r_aug = np.concatenate([[0.0], rs_np])
        w_aug = build_integration_weights(r_aug, fd_order)[1:] * rs_np
        w_even_np = w_odd_np = w_aug
    if not (np.all(w_even_np > 0) and np.all(w_odd_np > 0)):
        raise ValueError(
            "Radial quadrature weights are not strictly positive "
            "(the discrete energy norm would be indefinite): the "
            "fd_order is too high for this ny, or the custom wall "
            "grid is pathological near the axis."
        )
    y_weights = jnp.asarray(w_even_np, dtype=sharding.float_type)
    y_weights_odd = jnp.asarray(w_odd_np, dtype=sharding.float_type)

    D1_even, _, D1_odd, _, D1_pos, _ = build_parity_reduced_matrices(
        rs, fd_order, consistent_imm
    )
    return rs, D1_even, D1_odd, D1_pos, y_weights, y_weights_odd, inv_r


def interpolate_to_axis(
    arr: Array,
    rs: Array,
    axis: int = 0,
    order: int | None = None,
    parity: str | None = None,
) -> Array:
    r"""Interpolate an r-dependent array to the centreline `$r = 0$`.

    The radial grid excludes `$r = 0$` by construction (see
    :func:`build_radial_cgl_grid`); this evaluates radial data at
    the axis (spectrally for even-parity data on the CGL grids,
    by local Fornberg extrapolation otherwise; see *parity*), for
    any array carrying an r-varying axis (spectral or physical,
    real or complex, any number of other axes).  Runs host-side
    (weights are NumPy); pass addressable (single-device or fully
    replicated) arrays.

    Parameters
    ----------
    arr:
        Input array with ``arr.shape[axis] == len(rs)``.
    rs:
        Ascending radial grid on `$(0, 1]$` (host-readable, e.g.
        ``np.asarray(derived_params.wall_normal_grid)``).
    axis:
        The radial axis of *arr*.
    order:
        Stencil width minus one; defaults to
        ``params.res.fd_order``.  Ignored on the spectral
        even-parity CGL path.
    parity:
        ``None`` (default): one-sided ``order + 1``-point
        extrapolation -- the only safe general choice for
        *physical-space* arrays, whose `$r \to -r$` continuation
        pairs with `$\theta \to \theta + \pi$` and is therefore
        not a per-column symmetry.  ``"even"``: the data is
        smooth and even in `$r$` (an `$m_{\mathrm{eff}}$`-even
        spectral component, e.g. the mean mode of `$u_z$`); an
        even analytic function is a function of `$x = r^2$` --
        on a detected radial CGL grid the exact spectral
        parity-constrained fit in `$x$`
        (``fd._spectral_even_axis_weights``, exact for even
        polynomials of degree `$\le 2(N_r - 1)$`), on a
        custom/tanh grid the ``order + 1``-point stencil in `$x$`
        (exact to degree `$\le 2\,\mathrm{order}$`).
        ``"odd"``: the data vanishes on the axis identically
        (`$m_{\mathrm{eff}}$`-odd components); returns zeros.

    Returns
    -------
    :
        *arr* with the radial axis removed, evaluated at
        `$r = 0$`.
    """
    if order is None:
        order = params.res.fd_order
    moved = jnp.moveaxis(arr, axis, 0)
    # Shared JAX-free leaf (also behind the rigged-CGL interpolation
    # matrix): spectral even weights span the whole grid on CGL,
    # local rules are zero-padded outside their stencil; either way
    # the full-axis contraction drops the radial axis (odd parity ->
    # zeros).
    w = axis_extrapolation_weights(np.asarray(rs), order, parity)
    w_jax = jnp.asarray(w, dtype=sharding.float_type)
    return jnp.tensordot(w_jax, moved, axes=(0, 0))


# ── Shared radial base operator ───────────────────────────────────


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
    return D2 + inv_r[:, None] * D1


# ── Pallas-backend banded operator builders ───────────────────────


def _build_Lk_band_gpu(
    D1_wall: Array,
    band_even: Array,
    band_odd: Array,
    m_is_even: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
    p: int,
) -> Array:
    r"""Build `$L_k$` in banded storage for the Pallas backend.

    Same operator as :func:`_build_Lk_dense_gpu`,
    but assembled directly in banded layout
    ``(Nm, Nkz, Nr, 2p+1)`` (``band[..., i, d] = L_k[..., i, i-p+d]``)
    from the base-operator bands, with no ``(Nr, Nr)`` per mode.

    Parameters
    ----------
    D1_wall:
        Last row of `$D_1$` (parity-independent), shape ``(Nr,)``.
    band_even, band_odd:
        Banded `$A_{\mathrm{base}}$` for even/odd parity,
        shape ``(Nr, 2p+1)``.
    m_is_even, m2:
        Pressure parity selector and `$m^2$`, shape ``(Nm, 1, 1)``.
    inv_r2:
        `$1/r_j^2$`, shape ``(Nr,)``.
    kz2:
        `$k_z^2$`, shape ``(1, Nkz, 1)``.
    mean_mask:
        Mean-mode boolean mask, shape ``(Nm, Nkz, 1)``.
    p:
        FD order (half-bandwidth).
    """
    Nr = band_even.shape[0]
    band_base = jnp.where(m_is_even, band_even[None], band_odd[None])
    diag = -(m2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    # Single wall (r = 1): Neumann D1[-1, :] in band form, identity
    # (pin) at the mean mode; r = 0 regularity is built into the
    # parity-reduced base band, so no inner-wall row.
    neumann = _banded_wall_row(D1_wall, Nr - 1, p)
    wall = jnp.where(
        mean_mask, _banded_diag_column(p, band_base.dtype), neumann
    )  # (Nm, Nkz, 2p+1)
    return _assemble_banded_operator(
        band_base[:, None], 1.0, diag, [(Nr - 1, wall)]
    )


def _build_Hk_band_gpu(
    band_even: Array,
    band_odd: Array,
    m_is_even_vel: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    nu: float,
    p: int,
) -> Array:
    r"""Build one `$H_k$` Helmholtz operator in banded storage.

    Banded analogue of :func:`_build_Hk_dense_gpu`, laid out as
    ``(Nm, Nkz, Nr, 2p+1)``.
    """
    Nr = band_even.shape[0]
    band_base = jnp.where(m_is_even_vel, band_even[None], band_odd[None])
    diag = 1.0 / dt + c * nu * (meff2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    # Dirichlet no-slip wall: identity row at r = 1.
    eN = _banded_diag_column(p, band_base.dtype)
    return _assemble_banded_operator(
        band_base[:, None], -c * nu, diag, [(Nr - 1, eN)]
    )


# ── Dense-backend operator builders ───────────────────────────────


def _build_Lk_dense_gpu(
    D1_wall: Array,
    A_base_even: Array,
    A_base_odd: Array,
    m_is_even: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
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

    # Wall BC: Neumann D1[-1,:] for all modes, pin at the mean.
    D1_wall_1d = D1_wall.ravel()
    pin = eye_Nr[-1, :]
    wall_row = jnp.where(mean_mask, pin, D1_wall_1d)
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


# Operator backends sharing the ``.solve()`` contract.
_WallBoundedOp = DenseJAXSolver | PerModeBandedPallasOperator


# ── CylindricalFlow base dataclass ─────────────────────────────────


@register_dataclass_pytree
@dataclass
class CylindricalFlow:
    r"""Precomputed data for wall-bounded cylindrical flows.

    Subclasses must set ``base_flow`` and ``curl_base_flow``
    *after* calling
    ``super().__post_init__()``, which builds the radial CGL
    grid (half-CGL or rigged-CGL, per the resolved
    ``geo.grid_type``), parity-reduced FD matrices, and all
    per-mode IMM operators.

    The velocity state is carried through the solver in decoupled
    form `$(u_z, u_+, u_-)$` where

    .. math::
        u_+ = u_r + i\,u_\theta, \qquad
        u_- = u_r - i\,u_\theta,

    and in the physical triad everywhere outside it (the module
    docstring; ``to_pm_basis``/``from_pm_basis``).

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
        (`$D_{1,\mathrm{even}} - D_{1,\mathrm{pos}}$`).
        Nonzero only in the first
        `$g \sim (p+2)//2$` rows near `$r = 0$`, so only
        those rows are stored: shape ``(g, Nr)``.  Applied
        via ``out.at[:g].add(...)`` so the ghost GEMM cost
        is `$g/N_r$` of the pos part instead of doubling it.
    D2_ghost:
        Ghost correction for `$D_2$`
        (`$D_{2,\mathrm{even}} - D_{2,\mathrm{pos}}$`),
        shape ``(g, Nr)`` (same row count).
    D1_wall:
        Last row of `$D_1$` (parity-independent),
        shape ``(1, Nr)``.
    inv_r:
        `$1/r$` on the radial grid.
    inv_r2:
        `$1/r^2$` on the radial grid.
    dt, ab2_kappa:
        Live time step and AB2 step ratio, 0-d array leaves (see
        ``CartesianFlow`` and the builder ``set_dt``).
    """

    dt: Array = field(init=False)
    ab2_kappa: Array = field(init=False)
    rs: Array = field(init=False)
    inv_r: Array = field(init=False)
    inv_r2: Array = field(init=False)
    y_weights: Array = field(init=False)  # even-parity (energy norm)
    y_weights_odd: Array = field(init=False)  # odd-parity (mean u_theta)
    cfl_inv_spacing: Array = field(init=False)
    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    base_flow_padded: Array = field(init=False)
    curl_base_flow_padded: Array = field(init=False)
    base_flow_adv_padded: Array = field(init=False)
    D1_pos: Array = field(init=False)
    D2_pos: Array = field(init=False)
    D1_ghost: Array = field(init=False)
    D2_ghost: Array = field(init=False)
    D1_wall: Array = field(init=False)
    A_base_even: Array = field(init=False)
    A_base_odd: Array = field(init=False)
    Lk_op: _WallBoundedOp = field(init=False)
    Hk_op: _WallBoundedOp = field(init=False)
    v_plus_1: Array = field(init=False)
    v_minus_1: Array = field(init=False)
    q_z_1: Array = field(init=False)
    # Second IMM column of the ``res.consistent_imm`` boundary closure
    # (the 1-wall analogue of the Cartesian ``v3``/``v4``); ``None`` --
    # and therefore static pytree aux-data, not traced leaves -- when
    # the flag is off.  ``M_inv`` is then the ``2 x 2`` influence
    # matrix instead of the ``1 x 1`` scalar.
    v_plus_2: Array | None = field(init=False)
    v_minus_2: Array | None = field(init=False)
    q_z_2: Array | None = field(init=False)
    M_inv: Array = field(init=False)
    h_bulk_response: Array = field(init=False)
    H_bulk_inv: Array = field(init=False)

    def __post_init__(self) -> None:
        r"""Build radial grid, FD matrices, and IMM operators.

        Constructs the radial CGL grid on `$(0, 1]$` (half-CGL or
        rigged-CGL, per the resolved ``geo.grid_type``), builds
        parity-reduced FD matrices,
        assembles and factorises `$L_k$`, `$H_{k,+}$`,
        `$H_{k,-}$`, `$H_{k,z}$` directly on the device, then
        derives all homogeneous IMM data.
        """
        Nr = params.res.ny
        (
            self.rs,
            D1_even,
            D1_odd,
            D1_pos,
            self.y_weights,
            self.y_weights_odd,
            self.inv_r,
        ) = build_cylindrical_grid(
            Nr,
            params.res.fd_order,
            params.geo.wall_grid,
            params.geo.grid_type,
            params.geo.grid_stretch,
            params.res.consistent_imm,
        )
        self.inv_r2 = self.inv_r**2

        derived_params.wall_normal_grid = [
            float(v) for v in np.asarray(self.rs)
        ]

        # Inverse local advection length scales for the CFL
        # diagnostic (:func:`dnsjax.measurements.get_cfl`),
        # per component (u_z, u_r, u_theta), zero in the
        # ny_y_pad rows.  The azimuthal scale is the arc length
        # `$r \Delta\theta$` with `$\Delta\theta = l_z/n_z$`
        # (theta period `$l_z = 2\pi/m_0$` over the wedge;
        # ``geo.lz`` carries this).  Uniform directions use
        # the spectral-resolution spacing `$\Delta = L/n$`;
        # switch to ``padded_res.nx_padded`` / ``nz_padded``
        # for the dealiased-grid convention.
        inv_sp = np.zeros(
            (3, Nr + sharding.ny_y_pad), dtype=sharding.float_type
        )
        inv_sp[0, :Nr] = params.res.nx / params.geo.lx
        inv_sp[1, :Nr] = 1.0 / local_grid_spacing(np.asarray(self.rs))
        inv_sp[2, :Nr] = np.asarray(self.inv_r) * params.res.nz / params.geo.lz
        self.cfl_inv_spacing = jax.device_put(
            inv_sp[:, :, None, None], sharding.no_shard
        )

        # Full parity-reduced matrices (D2 needed for operators).
        (
            D1_even,
            D2_even,
            D1_odd,
            D2_odd,
            D1_pos,
            D2_pos,
        ) = build_parity_reduced_matrices(
            self.rs, params.res.fd_order, params.res.consistent_imm
        )

        self.D1_pos = jax.device_put(D1_pos, sharding.no_shard)
        self.D2_pos = jax.device_put(D2_pos, sharding.no_shard)

        # Ghost correction matrices: the difference between the
        # parity-reduced and the common (pos) part.  Stencils cross
        # r = 0 only near the axis, so just the first g rows are
        # nonzero; only those rows are stored and applied (a full
        # (Nr, Nr) ghost GEMM would cost as much as its pos
        # counterpart, doubling every FD matvec).
        D1_ghost_np = np.asarray(D1_even - D1_pos)
        D2_ghost_np = np.asarray(D2_even - D2_pos)
        g_rows = _ghost_row_count(D1_ghost_np, D2_ghost_np)
        self.D1_ghost = jax.device_put(D1_ghost_np[:g_rows], sharding.no_shard)
        self.D2_ghost = jax.device_put(D2_ghost_np[:g_rows], sharding.no_shard)

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
        self.y_weights_odd = jax.device_put(
            self.y_weights_odd, sharding.no_shard
        )

        Nm = sharding.nz_spec
        Nkz = sharding.nx_spec

        # Banded half-width: measured from the assembled base operator,
        # not assumed.  The wall row (r = 1) is replaced by a BC row in
        # every operator, so its own stencil need not fit.  Equals
        # ``fd_order`` for the mirrored fold's direct-fit D2 and is
        # wider for the ``consistent_imm`` composed D1.D1 (12 vs 8 at
        # ``fd_order = 8``).  Mirrors the Cartesian build; ``_hk_bands``
        # reads it back from the factored ``Lk``.
        p_band = max(
            matrix_half_bandwidth(np.asarray(self.A_base_even), (-1,)),
            matrix_half_bandwidth(np.asarray(self.A_base_odd), (-1,)),
        )
        dt = params.step.dt

        # Live-dt pytree leaves (class docstring; rebuilt by the
        # builder's ``set_dt`` with identical dtype/shape).
        self.dt = jnp.asarray(dt, dtype=sharding.float_type)
        self.ab2_kappa = jnp.ones((), dtype=sharding.float_type)

        # Solver-internal wavenumber arrays: squeeze y dim
        # from field layout (1, Nm, ...) to (Nm, ..., 1).
        m_s = fourier.m[0, ..., None]  # (Nm, 1, 1)
        kz2_s = fourier.kz2[0, ..., None]  # (1, Nkz, 1)
        mean_s = fourier.mean_mask[0, ..., None]  # (Nm, Nkz, 1)
        m_is_even_s = fourier.m_is_even[0, ..., None]  # (Nm, 1, 1)

        m_sq = m_s**2

        # Parity mask: pressure / u_z use (-1)^m -> m_is_even (the
        # u_+/u_- masks live in ``_hk_bands`` / ``_hk_dense_op``).
        m_is_even_p = m_is_even_s

        if params.solver.backend == "pallas":
            # Pallas backend: one-program-per-mode banded sweep.
            # Operators are assembled directly in banded storage (no
            # (Nr, Nr) per mode) and factored by the setup-checked
            # no-pivot banded LU (_build_pallas_operator).
            band_even = _banded_from_dense(self.A_base_even, p_band)
            band_odd = _banded_from_dense(self.A_base_odd, p_band)
            D1_wall_1d = self.D1_wall.ravel()

            # Lk (meff = m, pressure parity).
            Lk_band = _build_Lk_band_gpu(
                D1_wall_1d,
                band_even,
                band_odd,
                m_is_even_p,
                m_sq,
                self.inv_r2,
                kz2_s,
                mean_s,
                p_band,
            )
            self.Lk_op = _build_pallas_operator([Lk_band], "Lk")
            del Lk_band

            # Hk group (plus, minus, z): stacked into one homogeneous
            # operator and stability-checked as a single group.
            if params.step.adaptive:
                # Verify the no-pivot LU where the Helmholtz
                # diagonal is least dominant; adaptive rebuilds at
                # dt <= dt_max then skip the check
                # (solvers._factor_pallas_operator).
                _build_pallas_operator(
                    _hk_bands(params.step.dt_max, fourier, self),
                    "Hk(dt_max)",
                )
            self.Hk_op = _build_pallas_operator(
                _hk_bands(dt, fourier, self), "Hk"
            )

        else:
            # Dense backend: full matrices are built, LU-factored
            # (donated, so the factors reuse their buffers), then
            # dropped — only the factors are kept.
            Lk_dense = _build_Lk_dense_gpu(
                self.D1_wall,
                self.A_base_even,
                self.A_base_odd,
                m_is_even_p,
                m_sq,
                self.inv_r2,
                kz2_s,
                mean_s,
            )
            self.Lk_op = DenseJAXSolver(Lk_dense)
            del Lk_dense

            # Combined Hk: component order (plus, minus, z).
            self.Hk_op = _hk_dense_op(dt, fourier, self)

        self._derive_imm_homogeneous_data(fourier, Nm, Nkz, Nr)
        self._precompute_bulk_response(fourier, Nm, Nkz, Nr)

    def _derive_imm_homogeneous_data(
        self, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
    ) -> None:
        r"""Fill ``v_plus_1``, ``v_minus_1``, ``q_z_1``, and
        ``M_inv`` on-device.

        The homogeneous pressure `$p_1$` stays local: only the
        velocity responses derived from it are needed at runtime
        (the corrected pressure is never assembled -- see the
        Cartesian ``_imm_iteration`` stage-6 note).

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
        `$M^{-1} = 1/M$` for all modes except the mean mode
        `$(m, k_z) = (0, 0)$`, where `$M^{-1} = 0$` (the
        `$u_r$` zeroing below makes `$M = 0$` there).
        Padding modes take the regular `$1/M$` branch (their
        placeholder-wavenumber systems are as well-posed as
        physical ones); the values are inert, multiplied
        only by the exactly-zero wall residuals of zero
        fields.

        After the solves, the `$u_r$` part of ``v_plus_1``
        and ``v_minus_1`` is zeroed at the mean mode
        (continuity forces `$u_r \\equiv 0$` there), while
        preserving the `$u_\\theta$` part.  The zeroing runs
        before ``M`` is assembled.
        """
        # This run-once setup stays in the mode-outer (Nm, Nkz, Nr)
        # layout: the influence-matrix einsums below operate on it and
        # the results are transposed to field layout (Nr, Nm, Nkz) at
        # the end.  ``.solve`` now takes a mode-inner field, so each
        # setup solve is wrapped (transpose in, transpose out) to keep
        # this layout.  FUTURE: rebuild this setup natively mode-inner to
        # drop the wrappers -- the hot path already is; here it only
        # relocates a one-time transpose, so it is deferred.
        e_wall = (
            jnp.zeros(
                (Nm, Nkz, Nr),
                dtype=sharding.float_type,
                out_sharding=sharding.spec_imm_corr_shard,
            )
            .at[..., -1]
            .set(1.0)
        )
        p1_s = self.Lk_op.solve(e_wall.transpose(2, 0, 1)).transpose(1, 2, 0)

        # Pressure gradient components for the +/- equations.
        # The ghost matrix holds only its g nonzero rows; its
        # contribution lands in the first g radial entries.
        parity_sign_p_s = fourier_.m_is_even[0, ..., None] * 2 - 1
        g = self.D1_ghost.shape[0]
        ghost_p1 = jnp.einsum("ij, mzj -> mzi", self.D1_ghost, p1_s)
        D1_p1 = jnp.einsum("ij, mzj -> mzi", self.D1_pos, p1_s)
        D1_p1 = D1_p1.at[..., :g].add(parity_sign_p_s * ghost_p1)
        m_s = fourier_.m[0, ..., None]  # (Nm, 1, 1)
        m_over_r_s = m_s * self.inv_r  # (Nm, 1, Nr)

        rhs_v_plus = -(D1_p1 - m_over_r_s * p1_s)
        rhs_v_minus = -(D1_p1 + m_over_r_s * p1_s)
        rhs_v_plus = rhs_v_plus.at[..., -1].set(0.0)
        rhs_v_minus = rhs_v_minus.at[..., -1].set(0.0)
        q_rhs = p1_s.at[..., -1].set(0.0)

        # Batched solve: component order (plus, minus, z).
        rhs_stack = jnp.stack([rhs_v_plus, rhs_v_minus, q_rhs])
        result_stack = self.Hk_op.solve(
            rhs_stack.transpose(0, 3, 1, 2)
        ).transpose(0, 2, 3, 1)
        vp1_s = result_stack[0]
        vm1_s = result_stack[1]
        qz1_s = result_stack[2]

        # ``res.consistent_imm``: a second homogeneous column (seeded by
        # the interior source, the image of a unit wall correction
        # sigma_hat) and a 2x2 influence matrix -- see
        # :meth:`_derive_imm_closure` and the pipe branch of
        # :func:`_imm_iteration`.
        if params.res.consistent_imm:
            self._derive_imm_closure(
                fourier_,
                p1_s,
                (vp1_s, vm1_s, qz1_s),
                m_over_r_s,
                parity_sign_p_s,
                g,
            )
            return

        # Zero the u_r part at the mean mode, preserving u_theta.
        mean_s = fourier_.mean_mask[0, ..., None]  # (Nm, Nkz, 1)
        vr_corr = jnp.where(mean_s, (vp1_s + vm1_s) / 2, 0.0)
        vp1_s = vp1_s - vr_corr
        vm1_s = vm1_s - vr_corr

        # 1x1 influence matrix.
        D1_wall_row = self.D1_wall.ravel()  # (Nr,)
        ur_1 = (vp1_s + vm1_s) / 2
        M = jnp.einsum("j, mzj -> mz", D1_wall_row, ur_1)

        is_mean = fourier_.mean_mask[0]  # (Nm, Nkz)
        safe_M = jnp.where(is_mean, 1.0, M)
        self.M_inv = jnp.where(is_mean, 0.0, 1.0 / safe_M)

        # Transpose to field layout (Nr, Nm, Nkz).
        self.v_plus_1 = vp1_s.transpose(2, 0, 1)
        self.v_minus_1 = vm1_s.transpose(2, 0, 1)
        self.q_z_1 = qz1_s.transpose(2, 0, 1)

        # Static aux-data (not traced leaves) with the closure off.
        self.v_plus_2 = self.v_minus_2 = self.q_z_2 = None

    def _derive_imm_closure(
        self,
        fourier_: Fourier,
        p1_s: Array,
        neumann_cols: tuple[Array, Array, Array],
        m_over_r_s: Array,
        parity_sign_p_s: Array,
        g: int,
    ) -> None:
        r"""Second homogeneous column and the `$2 \times 2$` influence
        matrix (``res.consistent_imm``; the pipe's 1-wall analogue of
        :meth:`CartesianFlow._derive_imm_closure`).

        Extends the single Neumann-data column already derived by
        :meth:`_derive_imm_homogeneous_data` with one column seeded by
        the interior source -- the image, in the pressure Poisson RHS,
        of a unit boundary correction `$\hat\sigma$` at the wall row of
        the `$u_r$` momentum.  A unit `$u_r = \hat\sigma$` at `$r = 1$`
        enters the (parity-`$(-1)^{m+1}$`) divergence as the wall column
        of `$D_{1,v}$` (the `$u_r/r$` term lives only on the wall row,
        which the interior source zeroes), so the source is
        `$(D_{1,v}[:, -1])_P$`.  Everything downstream is the same solve
        chain as the Neumann column (`$L_k$` for the pressure, `$H_k$`
        for the velocity response and the axial potential).

        Rows of the `$2 \times 2$` influence system, in the
        ``alpha = -M^{-1} d`` convention of :func:`_imm_iteration`:

        - row ``0``: wall divergence `$M_{0,j} = D_{1,\mathrm{wall}}
          \cdot u_r^{(j)}$` -- the pre-existing condition.
        - row ``1``: the closure `$\hat\sigma_1 = $` correction, i.e.
          `$M_{1,j} = c\nu (A_{\mathrm{base},v} u_r^{(j)})|_{\mathrm{wall}}
          + \delta_{j,1} - \delta_{j,0}$`.  `$A_{\mathrm{base},v}$` is the
          `$u_r$` viscous operator (parity `$(-1)^{m+1}$`); at the wall
          `$u_\pm = 0$`, so the Helmholtz diagonal drops and only
          `$A_{\mathrm{base}}$` survives.  The `$-\delta_{j,0}$` is the
          Neumann pressure's own `$(D_1 p)|_{\mathrm{wall}} = \alpha_0$`.

        At the mean mode ``M_inv`` is zeroed outright (`$u_r \equiv 0$`
        and `$i k_z = 0$` there, so every column is inert).
        """
        c = params.step.implicitness
        nu = 1.0 / params.phys.re
        vp1_s, vm1_s, qz1_s = neumann_cols

        # --- Closure column, seeded by the interior source ---
        parity_sign_v_s = -parity_sign_p_s  # (-1)^{m+1}, (Nm, 1, 1)
        src = jnp.zeros_like(p1_s) + self.D1_pos[:, -1]  # (Nm, Nkz, Nr)
        src = src.at[..., :g].add(parity_sign_v_s * self.D1_ghost[:, -1])
        src = src.at[..., -1].set(0.0)  # interior source (wall row zeroed)
        pb_s = self.Lk_op.solve(src.transpose(2, 0, 1)).transpose(1, 2, 0)

        # Pressure gradient of p_beta in +/- (pressure parity (-1)^m),
        # exactly as the Neumann column.
        ghost_pb = jnp.einsum("ij, mzj -> mzi", self.D1_ghost, pb_s)
        D1_pb = jnp.einsum("ij, mzj -> mzi", self.D1_pos, pb_s)
        D1_pb = D1_pb.at[..., :g].add(parity_sign_p_s * ghost_pb)
        rhs_vp2 = (-(D1_pb - m_over_r_s * pb_s)).at[..., -1].set(0.0)
        rhs_vm2 = (-(D1_pb + m_over_r_s * pb_s)).at[..., -1].set(0.0)
        q_rhs2 = pb_s.at[..., -1].set(0.0)
        res2 = self.Hk_op.solve(
            jnp.stack([rhs_vp2, rhs_vm2, q_rhs2]).transpose(0, 3, 1, 2)
        ).transpose(0, 2, 3, 1)
        vp2_s, vm2_s, qz2_s = res2[0], res2[1], res2[2]

        # --- 2x2 influence matrix ---
        D1_wall_row = self.D1_wall.ravel()  # (Nr,)
        # A_base_v wall row per mode: u_+/u_- parity (-1)^{m+1}, so even
        # m -> odd parity -> A_base_odd, odd m -> A_base_even.
        miv = 1.0 - fourier_.m_is_even[0, :, 0]  # (Nm,), 1 if m odd
        abv_wall = jnp.where(
            miv[:, None],
            self.A_base_even[-1][None, :],
            self.A_base_odd[-1][None, :],
        )  # (Nm, Nr)
        ur_1 = (vp1_s + vm1_s) / 2
        ur_2 = (vp2_s + vm2_s) / 2
        m00 = jnp.einsum("j, mzj -> mz", D1_wall_row, ur_1)
        m01 = jnp.einsum("j, mzj -> mz", D1_wall_row, ur_2)
        m10 = c * nu * jnp.einsum("mj, mzj -> mz", abv_wall, ur_1) - 1.0
        m11 = c * nu * jnp.einsum("mj, mzj -> mz", abv_wall, ur_2) + 1.0
        M = jnp.stack(
            [jnp.stack([m00, m01], -1), jnp.stack([m10, m11], -1)], -2
        )  # (Nm, Nkz, 2, 2)

        is_mean = fourier_.mean_mask[0][..., None, None]
        eye2 = jnp.eye(2, dtype=sharding.float_type)
        self.M_inv = jnp.where(
            is_mean, 0.0, jnp.linalg.inv(jnp.where(is_mean, eye2, M))
        )

        # Zero u_r at the mean mode for both columns (continuity forces
        # u_r == 0 there), preserving u_theta; then to field layout.
        mean_s = fourier_.mean_mask[0, ..., None]  # (Nm, Nkz, 1)
        c1 = jnp.where(mean_s, ur_1, 0.0)
        c2 = jnp.where(mean_s, ur_2, 0.0)
        self.v_plus_1 = (vp1_s - c1).transpose(2, 0, 1)
        self.v_minus_1 = (vm1_s - c1).transpose(2, 0, 1)
        self.v_plus_2 = (vp2_s - c2).transpose(2, 0, 1)
        self.v_minus_2 = (vm2_s - c2).transpose(2, 0, 1)
        self.q_z_1 = qz1_s.transpose(2, 0, 1)
        self.q_z_2 = qz2_s.transpose(2, 0, 1)

    def _precompute_bulk_response(
        self, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
    ) -> None:
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

        # Unit uniform RHS at the mean mode only (``mean_mask``;
        # all other modes, padding included, get zero RHS), zero
        # wall BC.  Solver-internal layout (Nm, Nkz, Nr).
        ones_vec = jnp.ones(Nr, dtype=sharding.float_type).at[-1].set(0.0)
        rhs = jnp.where(fourier_.mean_mask[0, ..., None], ones_vec, 0.0)

        # Solve using the z-component (index 2) of the combined
        # Hk operator via a padded batch (one-time init cost).
        zeros = jnp.zeros_like(rhs)
        h_full = self.Hk_op.solve(
            jnp.stack([zeros, zeros, rhs]).transpose(0, 3, 1, 2)
        ).transpose(0, 2, 3, 1)[2]

        # ``reshard`` (not ``device_put``): this method also runs
        # inside the jitted ``set_dt`` rebuild, where placing a
        # traced value is expressed as a resharding.
        self.h_bulk_response = jax.sharding.reshard(
            extract_mean_mode(h_full.transpose(2, 0, 1)[None])[0],
            sharding.no_shard,
        )
        H_bulk = 2 * jnp.dot(self.y_weights, self.h_bulk_response)
        self.H_bulk_inv = 1.0 / H_bulk


def _hk_bands(
    dt: float | Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> list[Array]:
    r"""Assemble the banded `$H_k$` group (+, -, z) at *dt*.

    Single-sources the band assembly for the setup-checked build, the
    adaptive ``dt_max`` stability pre-check, and the jitted ``set_dt``
    rebuild (:func:`_build_dt_leaves`).  Pallas backend only.

    The half-width is read back from the already-factored (and
    ``dt``-independent) `$L_k$`, whose ``L`` factor is
    ``(Nr, p, Nm, Nkz)`` -- a static shape, so this works inside
    ``jit`` (``set_dt``) where a host-side ``matrix_half_bandwidth`` on
    the traced ``A_base`` could not.  Equals ``fd_order`` off the
    ``consistent_imm`` gate and the wider composed band on it.
    """
    p_band = flow_.Lk_op.L.shape[1]
    m_s = fourier_.m[0, ..., None]
    kz2_s = fourier_.kz2[0, ..., None]
    m_is_even_s = fourier_.m_is_even[0, ..., None]
    # u_+/u_- carry parity (-1)^{m+1}; u_z carries (-1)^m.
    m_is_even_v = 1.0 - m_is_even_s
    band_even = _banded_from_dense(flow_.A_base_even, p_band)
    band_odd = _banded_from_dense(flow_.A_base_odd, p_band)
    groups = (
        (m_is_even_v, (m_s + 1) ** 2),
        (m_is_even_v, (m_s - 1) ** 2),
        (m_is_even_s, m_s**2),
    )
    return [
        _build_Hk_band_gpu(
            band_even,
            band_odd,
            parity,
            meff2,
            flow_.inv_r2,
            kz2_s,
            dt,
            params.step.implicitness,
            1.0 / params.phys.re,
            p_band,
        )
        for parity, meff2 in groups
    ]


def _hk_dense_op(
    dt: float | Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> DenseJAXSolver:
    r"""Factored dense stacked `$H_k$` (+, -, z) at *dt* (dense
    backend)."""
    m_s = fourier_.m[0, ..., None]
    kz2_s = fourier_.kz2[0, ..., None]
    m_is_even_s = fourier_.m_is_even[0, ..., None]
    m_is_even_v = 1.0 - m_is_even_s
    groups = (
        (m_is_even_v, (m_s + 1) ** 2),
        (m_is_even_v, (m_s - 1) ** 2),
        (m_is_even_s, m_s**2),
    )
    ops = [
        DenseJAXSolver(
            _build_Hk_dense_gpu(
                flow_.A_base_even,
                flow_.A_base_odd,
                parity,
                meff2,
                flow_.inv_r2,
                kz2_s,
                dt,
                params.step.implicitness,
                1.0 / params.phys.re,
            )
        )
        for parity, meff2 in groups
    ]
    return DenseJAXSolver.from_factors(
        lu=jnp.stack([o.lu for o in ops]),
        perm=jnp.stack([o.perm for o in ops]),
    )


def _build_dt_leaves(
    dt: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> dict[str, object]:
    r"""Rebuild every ``dt``-dependent flow leaf at the traced *dt*.

    The pure counterpart of the ``__post_init__`` operator/IMM setup,
    jitted by the builder's ``set_dt``: assemble the `$H_k$` group at
    *dt*, factor it **unchecked**
    (:func:`solvers._factor_pallas_operator` -- the checked build ran
    at setup, and under ``step.adaptive`` additionally at ``dt_max``,
    the dominance-weakest point), then re-run the unmodified IMM
    derivation on a trace-local shallow copy of *flow_* and collect
    the refreshed leaves.  `$L_k$` is ``dt``-independent and shared.
    The returned leaves match the stored ones in
    shape/dtype/sharding, so swapping them onto the flow singleton
    retraces nothing.
    """
    new = copy.copy(flow_)
    new.dt = dt
    if params.solver.backend == "pallas":
        new.Hk_op = _factor_pallas_operator(_hk_bands(dt, fourier_, new))
    else:
        new.Hk_op = _hk_dense_op(dt, fourier_, new)
    new._derive_imm_homogeneous_data(
        fourier_, sharding.nz_spec, sharding.nx_spec, params.res.ny
    )
    new._precompute_bulk_response(
        fourier_, sharding.nz_spec, sharding.nx_spec, params.res.ny
    )
    leaves = {
        "dt": new.dt,
        "Hk_op": new.Hk_op,
        "v_plus_1": new.v_plus_1,
        "v_minus_1": new.v_minus_1,
        "q_z_1": new.q_z_1,
        "M_inv": new.M_inv,
        "h_bulk_response": new.h_bulk_response,
        "H_bulk_inv": new.H_bulk_inv,
    }
    if params.res.consistent_imm:
        leaves |= {
            "v_plus_2": new.v_plus_2,
            "v_minus_2": new.v_minus_2,
            "q_z_2": new.q_z_2,
        }
    return leaves


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
    matrix-vector products (`$D_1$`).  Radial derivatives use
    the parity-reduced `$D_1$`: the common
    `$D_{1,\mathrm{pos}}$` part plus the ghost correction
    signed by each field's parity (`$(-1)^m$` for `$u_z$`,
    `$(-1)^{m+1}$` for `$u_\theta$`).

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
    inv_r = flow_.inv_r[:, None, None]

    # Parity signs: u_theta has parity (-1)^{m+1},
    # u_z has parity (-1)^m.
    parity_sign_p = fourier_.m_is_even * 2 - 1
    parity_sign_v = -parity_sign_p

    # Batch D1_pos and D1_ghost into two GEMMs; the ghost GEMM
    # covers only its g nonzero rows near the axis.
    g = flow_.D1_ghost.shape[0]
    # Stack y-leading (N_r, 2, ...) so the batched D1 GEMM contracts the
    # leading wall-normal axis transpose-free, then unstack to 3-d.
    fields = jnp.stack([utheta, uz], axis=1)
    dy_common = apply_y_matrix(flow_.D1_pos, fields, component_axis=1)
    dy_ghost = apply_y_matrix(flow_.D1_ghost, fields, component_axis=1)
    dy_utheta = dy_common[:, 0].at[:g].add(parity_sign_v * dy_ghost[:, 0])
    dy_uz = dy_common[:, 1].at[:g].add(parity_sign_p * dy_ghost[:, 1])

    omega_r = im * inv_r * uz - ikz * utheta
    omega_theta = ikz * ur - dy_uz
    omega_z = dy_utheta + inv_r * utheta - im * inv_r * ur

    return jnp.array([omega_z, omega_r, omega_theta])


def _l_bf(
    state: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> Array:
    r"""Linear base-flow coupling (FFT-free), in `$(u_z, u_+, u_-)$`.

    Mirrors :func:`_get_rhs_core`'s conversion to the `$(u_z, u_r,
    u_\theta)$` triad (the cross products are defined on the
    physical triad), but evaluates only the two *linear* base-flow
    cross-product terms (:func:`base_flow_coupling`) -- no Fourier
    transform (the base flow is a radial profile, and
    `$\boldsymbol{\omega}'$` is the spectral :func:`_curl_fn`; that
    curl is the same subexpression ``get_rhs`` builds, so evaluating
    both at one state costs one curl -- XLA CSE, see
    ``_cnab2_lbf_core``).  The pure self-advection
    `$\mathbf{u}' \times \boldsymbol{\omega}' =
    \text{get\_rhs} - \text{\_l\_bf}$` stays explicit; this term (with
    its stiff radial derivative on the wall-clustered grid) is made
    implicit by the CN/AB2 scheme -- see ``step_cnab2`` in
    :mod:`dnsjax.timestep`.

    With ``params.step.implicit_mean_coupling`` (default on) the
    *instantaneous mean-flow* coupling is folded in by adding the
    `$m = k_z = 0$` mean profiles of the `$(u_z, u_r, u_\theta)$`
    state and of `$\boldsymbol{\omega}'$` (the curl being linear and
    mode-diagonal, the mean of the curl *is* the curl of the mean)
    onto the base-flow profiles -- FFT-free
    (``extract_mean_mode`` is a ``psum``); see the Cartesian
    ``_l_bf`` and the ``TimeStepping`` docstring in
    :mod:`dnsjax.parameters`.
    """
    state_rthz = from_pm_basis(state)

    omega = _curl_fn(state_rthz, fourier_, flow_)
    base = flow_.base_flow
    curl_base = flow_.curl_base_flow
    if params.step.implicit_mean_coupling:
        base = base + extract_mean_mode(state_rthz)[:, :, None, None]
        curl_base = curl_base + extract_mean_mode(omega)[:, :, None, None]
    l_bf = to_pm_basis(base_flow_coupling(state_rthz, omega, base, curl_base))
    # Moving frame: the convective frame term (the same expression
    # ``_get_rhs_core`` adds, diagonal in the solver basis) belongs
    # to the linear coupling, so CN/AB2 integrates it implicitly.
    u_grid = derived_params.u_grid
    if u_grid == 0:
        return l_bf
    return l_bf + (1j * u_grid) * fourier_.kz * state


# Per-direction CFL column names, matching the physical-space
# component order (u_z, u_r, u_theta).
CFL_NAMES: tuple[str, str, str] = ("CFL_z", "CFL_r", "CFL_th")


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
    measure_fn: Callable[[Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    r"""Evaluate the nonlinear RHS in `$(u_z, u_+, u_-)$` form.

    1. Convert `$(u_z, u_+, u_-) \to (u_z, u_r, u_\theta)$` -- the
       real FFTs need components that are individually
       Hermitian-symmetric, which `$u_\pm$` are not, so the
       physical-space fields (and the *measure_fn* CFL) are always
       the physical triad.
    2. Compute the rotational-form nonlinear term via
       :func:`~dnsjax.rhs.get_nonlin` with the cylindrical
       curl (and the optional physical-space *measure_fn*).
    3. Convert `$(NL_z, NL_r, NL_\theta)
       \to (NL_z, NL_+, NL_-)$`.
    """
    nonlin_rthz = get_nonlin(
        from_pm_basis(state),
        flow_.base_flow_padded,
        flow_.curl_base_flow_padded,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier_, flow_),
        measure_fn,
    )
    if measure_fn is not None:
        nonlin_rthz, measurements = nonlin_rthz

    rhs = to_pm_basis(nonlin_rthz)
    # Moving frame: convective-form frame term
    # `$+ i k_z U_{grid} \mathbf{u}'$` -- the axial derivative is
    # component-diagonal in the `$(u_z, u_+, u_-)$` basis, so it is
    # added on the solver-basis state (mode-diagonal,
    # divergence-free; see ``pad_base_flow``).
    u_grid = derived_params.u_grid
    if u_grid != 0:
        rhs = rhs + (1j * u_grid) * fourier_.kz * state
    if measure_fn is None:
        return rhs
    return rhs, measurements


def _get_rhs(
    state: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> Array:
    r"""Evaluate the nonlinear RHS in `$(u_z, u_+, u_-)$` form."""
    return _get_rhs_core(state, fourier_, flow_, None)


def _get_rhs_measured(
    state: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate the nonlinear RHS + CFL measurements."""

    def _measure(u_phys: Array, omega_phys: Array) -> dict[str, Array]:
        return get_cfl(
            u_phys,
            flow_.base_flow_adv_padded,
            flow_.cfl_inv_spacing,
            CFL_NAMES,
            flow_.dt,
        )

    return _get_rhs_core(state, fourier_, flow_, _measure)


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

    The ghost correction matrices are stored row-sliced to
    their `$g \sim p/2$` nonzero rows (near the pipe centre,
    where stencils cross `$r = 0$`), so the ghost GEMMs and
    the scatter-add touch only the first `$g$` radial points.

    Parameters
    ----------
    u:
        Field, shape ``(Nr, Nm, Nkz)``.
    flow\_:
        Cylindrical flow data (uses ``D1_pos``,
        ``D2_pos``, ``D1_ghost``, ``D2_ghost``,
        ``inv_r``).
    parity_sign:
        `$(-1)^{m_{\mathrm{eff}}}$`, shape
        ``(1, Nm, 1)``.
    """
    inv_r = flow_.inv_r[:, None, None]
    D2_u = apply_y_matrix(flow_.D2_pos, u)
    D1_u = apply_y_matrix(flow_.D1_pos, u)
    common = D2_u + inv_r * D1_u

    g = flow_.D1_ghost.shape[0]
    D2g_u = apply_y_matrix(flow_.D2_ghost, u)
    D1g_u = apply_y_matrix(flow_.D1_ghost, u)
    ghost = D2g_u + inv_r[:g] * D1g_u

    return common.at[:g].add(parity_sign * ghost)


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
    inv_r2 = flow_.inv_r2[:, None, None]
    out = Abase_u - (fourier_.m2 * inv_r2 + fourier_.kz2) * u

    # Wall row: Neumann D1[-1,:] for all modes, pin at the mean.
    D1_wall_row = flow_.D1_wall.ravel()
    wall_val = jnp.einsum("j, jmz -> mz", D1_wall_row, u)
    bot = jnp.where(fourier_.mean_mask[0], u[-1], wall_val)
    return out.at[-1].set(bot)


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
    dt = flow_.dt
    nu = 1.0 / params.phys.re

    uz_n, up_n, um_n = velocity_n[0], velocity_n[1], velocity_n[2]
    NLz_n, NLp_n, NLm_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    NLz_j, NLp_j, NLm_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]
    m = fourier_.m

    # Parity signs for each component type.
    parity_sign_p = fourier_.m_is_even * 2 - 1  # (-1)^m
    parity_sign_v = -parity_sign_p  # (-1)^{m+1}

    m_plus_1_sq = (m + 1) ** 2
    m_minus_1_sq = (m - 1) ** 2
    m_sq = fourier_.m2

    # Batch all D1 y-derivatives with (-1)^{m+1} parity into
    # one GEMM each for D1_pos and D1_ghost (2 instead of 4);
    # the ghost GEMM covers only its g nonzero rows.
    g = flow_.D1_ghost.shape[0]
    # Stack y-leading (N_r, 6, ...) so the batched D1 GEMM contracts the
    # leading wall-normal axis transpose-free; the component axis is 1.
    all_vparity = jnp.stack([up_n, um_n, NLp_j, NLp_n, NLm_j, NLm_n], axis=1)
    dy_common = apply_y_matrix(flow_.D1_pos, all_vparity, component_axis=1)
    dy_ghost = apply_y_matrix(flow_.D1_ghost, all_vparity, component_axis=1)
    dy_all = dy_common.at[:g].add(parity_sign_v * dy_ghost)

    # Cylindrical divergence at time n.  ``dnsjax.analysis`` mirrors
    # this operator in physical components; changing it here means
    # changing ``snapshot_ops.divergence`` and the transcription in
    # ``tests/test_snapshot_export.py`` (``_solver_divergence``),
    # which pins the two together.
    div_n = (
        (dy_all[:, 0] + (m + 1) * inv_r * up_n) / 2
        + (dy_all[:, 1] + (1 - m) * inv_r * um_n) / 2
        + ikz * uz_n
    )

    # Divergence of nonlinear terms at times n and j.
    div_NLj = (
        (dy_all[:, 2] + (m + 1) * inv_r * NLp_j) / 2
        + (dy_all[:, 4] + (1 - m) * inv_r * NLm_j) / 2
        + ikz * NLz_j
    )
    div_NLn = (
        (dy_all[:, 3] + (m + 1) * inv_r * NLp_n) / 2
        + (dy_all[:, 5] + (1 - m) * inv_r * NLm_n) / 2
        + ikz * NLz_n
    )

    Lk_d = _lk_matvec(div_n, flow_, fourier_)

    f_hat = div_n / dt + c * div_NLj + (1 - c) * div_NLn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure.
    f_hat_P = f_hat.at[-1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: Helmholtz solves for each component.  The Hk construction
    # is built **y-leading** ``(N_r, C, ...)`` so the batched D1/D2 GEMMs
    # contract the leading wall-normal axis transpose-free (component axis
    # 1); the solve takes that layout directly (``component_axis=1``) and
    # we unstack.  ``inv_r``/``inv_r2`` get a trailing axis to broadcast
    # over the C axis; ``kz2``/``mean_mask`` are trailing-mode broadcasts
    # (layout-invariant).
    inv_r_y = inv_r[..., None]  # (N_r, 1, 1, 1) over the C axis
    vel_n_stack = jnp.stack([up_n, um_n, uz_n], axis=1)  # (N_r, 3, ...)
    pP_and_vel = jnp.concatenate([pP[:, None], vel_n_stack], axis=1)
    D1_batch = apply_y_matrix(flow_.D1_pos, pP_and_vel, component_axis=1)
    D1g_batch = apply_y_matrix(flow_.D1_ghost, pP_and_vel, component_axis=1)

    # pP pressure gradient (parity (-1)^m -> parity_sign_p).
    D1_pP = D1_batch[:, 0].at[:g].add(parity_sign_p * D1g_batch[:, 0])
    m_over_r = m * inv_r  # (1, Nm, 1) * (Nr, 1, 1) → (Nr, Nm, 1)

    grad_pP_plus = D1_pP - m_over_r * pP
    grad_pP_minus = D1_pP + m_over_r * pP
    grad_pP_z = ikz * pP

    # Batched `$H_k^-$` matvec for all three components (y-leading).
    D1_vel = D1_batch[:, 1:]
    D1g_vel = D1g_batch[:, 1:]
    D2_all = apply_y_matrix(flow_.D2_pos, vel_n_stack, component_axis=1)
    D2g_all = apply_y_matrix(flow_.D2_ghost, vel_n_stack, component_axis=1)
    common_hk = D2_all + inv_r_y * D1_vel
    ghost_hk = D2g_all + inv_r_y[:g] * D1g_vel
    parity_hk = jnp.stack(
        [parity_sign_v, parity_sign_v, parity_sign_p], axis=1
    )
    Abase_stack = common_hk.at[:g].add(parity_hk * ghost_hk)
    meff2_stack = jnp.stack([m_plus_1_sq, m_minus_1_sq, m_sq], axis=1)
    inv_r2 = flow_.inv_r2[:, None, None, None]  # (N_r, 1, 1, 1)
    lapl_stack = (
        Abase_stack - (meff2_stack * inv_r2 + fourier_.kz2) * vel_n_stack
    )
    Hk_minus_stack = (1.0 / dt) * vel_n_stack + (1.0 - c) * nu * lapl_stack
    Hk_minus_stack = Hk_minus_stack.at[-1].set(vel_n_stack[-1])

    R_stack = (
        Hk_minus_stack
        - jnp.stack([grad_pP_plus, grad_pP_minus, grad_pP_z], axis=1)
        + c * jnp.stack([NLp_j, NLm_j, NLz_j], axis=1)
        + (1 - c) * jnp.stack([NLp_n, NLm_n, NLz_n], axis=1)
    )

    # Zero wall BC (Dirichlet no-slip).
    R_stack = R_stack.at[-1].set(0.0)

    # Zero the u_r part of the +/- RHS at the mean mode so
    # the Helmholtz solves produce u_r = 0 there.  At m=0,
    # Hk_plus and Hk_minus are identical (m_eff^2 = 1, same
    # parity), so the antisymmetric RHS gives up = -um.
    Rr_corr = jnp.where(
        fourier_.mean_mask, (R_stack[:, 0] + R_stack[:, 1]) / 2, 0.0
    )
    R_stack = R_stack.at[:, 0].add(-Rr_corr)
    R_stack = R_stack.at[:, 1].add(-Rr_corr)

    # Batched Helmholtz solve (y-leading, component axis 1).
    arb_stack = flow_.Hk_op.solve(R_stack, component_axis=1)
    up_arb, um_arb, uz_arb = (
        arb_stack[:, 0],
        arb_stack[:, 1],
        arb_stack[:, 2],
    )

    # Stage 4: wall divergence residual.
    D1_wall_row = flow_.D1_wall.ravel()
    ur_arb = (up_arb + um_arb) / 2
    d_wall = jnp.einsum("j, jmz -> mz", D1_wall_row, ur_arb)

    # Mean mode: pressure is a gauge; zero the residual.
    d_wall = jnp.where(fourier_.mean_mask[0], 0.0, d_wall)

    if params.res.consistent_imm:
        # Stage 4b (closure): the second wall condition -- the boundary
        # correction sigma_hat equals the interior-form u_r-momentum
        # residual on the wall row the Dirichlet replacement discards.
        # At the wall u_+ = u_- = 0, so only A_base_v survives the
        # Helmholtz; A_base_v is the parity-(-1)^{m+1} u_r viscous
        # operator (even m -> A_base_odd, odd m -> A_base_even).  See
        # :meth:`_derive_imm_closure` and the Cartesian _imm_iteration.
        ur_n = (up_n + um_n) / 2
        nl_ur = (c * (NLp_j + NLm_j) + (1 - c) * (NLp_n + NLm_n)) / 2
        miv = 1.0 - fourier_.m_is_even[0, :, 0]  # (Nm,), 1 if m odd
        abv_wall = jnp.where(
            miv[:, None],
            flow_.A_base_even[-1][None, :],
            flow_.A_base_odd[-1][None, :],
        )  # (Nm, Nr)
        av_arb = jnp.einsum("mj, jmz -> mz", abv_wall, ur_arb)
        av_n = jnp.einsum("mj, jmz -> mz", abv_wall, ur_n)
        closure = (
            c * nu * av_arb + ur_n[-1] / dt + (1 - c) * nu * av_n + nl_ur[-1]
        )
        closure = jnp.where(fourier_.mean_mask[0], 0.0, closure)
        d_vec = jnp.stack([d_wall, closure], axis=-1)  # (Nm, Nkz, 2)
        # Stage 5: 2x2 influence-matrix algebra alpha = -M_inv @ d.
        alpha_vec = -jnp.einsum("mzab, mzb -> mza", flow_.M_inv, d_vec)
        alpha = alpha_vec[..., 0][None]  # Neumann column
        alpha2 = alpha_vec[..., 1][None]  # closure column
        # Stage 6: corrected velocity (rank-2 update).
        up_new = up_arb + alpha * flow_.v_plus_1 + alpha2 * flow_.v_plus_2
        um_new = um_arb + alpha * flow_.v_minus_1 + alpha2 * flow_.v_minus_2
        qz_corr = alpha * flow_.q_z_1 + alpha2 * flow_.q_z_2
    else:
        # Stage 5: influence matrix correction (scalar per mode).
        alpha = (-flow_.M_inv * d_wall)[None]  # (1, Nm, Nkz)
        # Stage 6: corrected velocity.
        up_new = up_arb + alpha * flow_.v_plus_1
        um_new = um_arb + alpha * flow_.v_minus_1
        qz_corr = alpha * flow_.q_z_1

    # Stage 7: zero mean-mode u_r, preserving u_theta.
    ur_corr = jnp.where(fourier_.mean_mask, (up_new + um_new) / 2, 0.0)
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
    # bulk correction fuse into a single expression.  The
    # write mask is ``mean_mask``: no other mode (padding
    # included) receives the correction.
    if params.phys.driving == "constant_bulk_velocity":
        mean_uz = extract_mean_mode(uz_arb[None])[0].real
        bulk_uz = 2 * jnp.dot(flow_.y_weights, mean_uz)
        uz_new = (
            uz_arb
            - ikz * qz_corr
            + jnp.where(
                fourier_.mean_mask,
                -bulk_uz
                * flow_.H_bulk_inv
                * flow_.h_bulk_response[:, None, None],
                0.0,
            )
        )
    else:
        uz_new = uz_arb - ikz * qz_corr

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
    r"""L2 convergence norm of a solver-basis correction.

    Corrections live in the decoupled `$(u_z, u_+, u_-)$` basis, so
    the 1/2 weight on the pair makes this the *physical* norm of the
    corresponding correction
    (`$|u_r|^2 + |u_\theta|^2 = (|u_+|^2 + |u_-|^2)/2$`) -- the same
    scalar :func:`get_norm2_cyl` reports for a physical-basis array.
    """
    pm2 = get_norm2(correction[1:], fourier_.k_metric, flow_.y_weights)
    uz2 = get_norm2(correction[:1], fourier_.k_metric, flow_.y_weights)
    return jnp.sqrt(uz2 + pm2 / 2)


# ── Stepper factory ─────────────────────────────────────────────


def build_cylindrical_stepper(
    flow: CylindricalFlow,
) -> tuple[
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array, Array, Array], tuple[Array, Array, Array]],
    Callable[[str | None], Array],
    Callable[[Array], tuple[Array, Array, Array]],
    Callable[[Array], tuple[Array, Array, Array, dict[str, Array]]],
    Callable[[Array, Array], tuple[Array, Array, Array, Array]],
    Callable[
        [Array, Array], tuple[Array, Array, Array, Array, dict[str, Array]]
    ],
    Callable[[float], None],
    Callable[[], None],
]:
    """Build time-stepping functions for a cylindrical flow.

    Returns ``(predict_and_correct, iterate_correction,
    init_state_bound, predict_and_fully_correct,
    predict_and_fully_correct_measured, step_cnab2,
    step_cnab2_measured, set_dt, reset_ab2_kappa)`` with the
    ``fourier`` and *flow* singletons already bound.  ``_l_bf`` (the
    FFT-free base-flow coupling) is passed so the CN/AB2 scheme
    treats it implicitly; ``_build_dt_leaves`` backs the adaptive-dt
    ``set_dt`` rebuild.  Every array crossing these steppers is in
    the decoupled `$(u_z, u_+, u_-)$` solver basis (the module
    docstring).
    """
    return build_wall_bounded_stepper(
        _get_rhs,
        _predict,
        _correct,
        _norm,
        fourier,
        flow,
        _get_rhs_measured,
        _l_bf,
        dt_leaves_fn=_build_dt_leaves,
    )
