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
    integrate_scalar,  # noqa: F401 — re-exported
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

    The wavenumber arrays are global multi-device arrays: host-side
    consumers recompute them from the JAX-free
    :mod:`dnsjax.harmonics` sequences (`$\times\,2\pi/L$`, azimuthal
    `$\times\,m_0$`), never ``np.asarray`` on these fields.
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

    Rejected alternative: an axis-regular fit in `$x = r^2$` (an
    axis-regular field is analytic in `$x$`), which gives
    `$D_{1,\mathrm{even}} = 2\,\mathrm{diag}(r) D_x$`,
    `$D_{1,\mathrm{odd}} = S + \mathrm{diag}(r) D_{1,\mathrm{even}} S$`
    with a matching direct `$D_2 = 2 D_x + 4x D_{xx}$`.  It buys a
    5-1000x *pointwise near-axis* accuracy gain but loses on every
    global measure: the refit trades away accuracy
    at `$r \approx 1$`, where the pipe's optimal-growth and wall-shear
    physics live.  On the Schmid & Henningson `$G_{\max} = 649$` anchor
    the mirrored fold errs by -4.1 / -0.6 / -0.06 / +0.01 % at
    `$N_r = 20/28/40/72$` against the fit's +357 / +37 / +3.5 / +0.25 %
    (unchanged with ``res.consistent_imm`` either way), and on a
    random-IC pipe run the fit cost ~17x the corrector iterations.  Its
    other job -- making the near-axis `$1/r$` commutator exact, which
    only the rejected composed-`$D_2$` ``consistent_imm`` route needed
    -- is moot: the reconstruction scheme
    (:func:`_imm_iteration_vw`) needs no operator identity at all.

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


def build_cylindrical_grid(
    ny: int,
    fd_order: int,
    wall_grid: str | None = None,
    grid_type: str | None = None,
    grid_stretch: float = 1.5,
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
        rs, fd_order
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


def _vw_recovery_parts(
    m2: Array, inv_r: Array, inv_r2: Array, kz2: Array, mean_mask: Array
) -> tuple[Array, Array]:
    r"""Per-mode pieces of the `$u_r$` recovery operator (vw scheme).

    Identical algebra to the annular twin
    (``annular._vw_recovery_parts``): the `$\Phi$` definition with the
    reconstruction's `$u_\theta(u_r, \omega_r)$` substituted in, so the
    recovery is exact per pass.

    .. math::
        L_{v,\mathrm{mod}} = A_{\mathrm{base}}^{(v)}
        - \Bigl(\frac{m^2+1}{r^2} + k_z^2\Bigr) I
        + \frac{2 m^2}{r^3\,\Delta}\,\Bigl(D_1^{(v)}
          + \frac{1}{r}\Bigr),
        \qquad \Delta = k_z^2 + \frac{m^2}{r^2}.

    The `$1/r^3$` coefficient is **odd**, and `$(D_1 + 1/r)$` maps the
    `$u_r$` parity class `$(-1)^{m+1}$` to `$(-1)^m$`, so the product
    lands back in the `$u_r$` class: the correction is parity-consistent
    and rides the same parity-reduced band as
    `$A_{\mathrm{base}}^{(v)}$`.  Zero at `$m = 0$` and masked at the
    mean (where `$\Delta = 0$`).
    """
    diag = -((m2 + 1.0) * inv_r2 + kz2)
    det = kz2 + m2 * inv_r2
    det_safe = jnp.where(mean_mask, 1.0, det)
    coeff = jnp.where(mean_mask, 0.0, 2.0 * m2 * inv_r2 * inv_r / det_safe)
    return diag, coeff


def _build_Lv_dir_band_gpu(
    D1_even: Array,
    D1_odd: Array,
    band_even: Array,
    band_odd: Array,
    m_is_even_vel: Array,
    m2: Array,
    inv_r: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
    p: int,
) -> Array:
    r"""Build the vw `$u_r$` recovery operator in banded storage.

    `$L_{v,\mathrm{mod}}$` of :func:`_vw_recovery_parts` on the
    velocity parity `$(-1)^{m+1}$`, with a Dirichlet identity row at the
    single wall `$r = 1$` (`$u_r|_{\mathrm{wall}} = 0$`); the axis is
    closed by the parity reduction, exactly as for `$H_{k,\pm}$`.
    ``dt``-free, like the legacy Neumann `$L_k$` it replaces, and no
    mean pin is needed -- `$m_{\mathrm{eff}}^2 = m^2 + 1 \ge 1$` keeps
    the operator regular at every mode including `$k^2 = 0$`.
    """
    Nr = band_even.shape[0]
    diag, coeff = _vw_recovery_parts(m2, inv_r, inv_r2, kz2, mean_mask)
    band_base = jnp.where(m_is_even_vel, band_even[None], band_odd[None])
    # Band the (D1 + 1/r) correction per parity, then select -- never
    # forming a per-mode (Nr, Nr).
    eye_Nr = jnp.eye(Nr, dtype=band_even.dtype)
    corr_even = _banded_from_dense(D1_even + inv_r[:, None] * eye_Nr, p)
    corr_odd = _banded_from_dense(D1_odd + inv_r[:, None] * eye_Nr, p)
    band_corr = jnp.where(m_is_even_vel, corr_even[None], corr_odd[None])
    band = band_base[:, None] + coeff[..., None] * band_corr[:, None]
    eN = _banded_diag_column(p, band_even.dtype)
    return _assemble_banded_operator(band, 1.0, diag, [(Nr - 1, eN)])


# ── Dense-backend operator builders ───────────────────────────────


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


def _build_Lv_dir_dense_gpu(
    D1_even: Array,
    D1_odd: Array,
    A_base_even: Array,
    A_base_odd: Array,
    m_is_even_vel: Array,
    m2: Array,
    inv_r: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
) -> Array:
    r"""Dense twin of :func:`_build_Lv_dir_band_gpu`."""
    Nr = A_base_even.shape[0]
    dtype = A_base_even.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)
    diag, coeff = _vw_recovery_parts(m2, inv_r, inv_r2, kz2, mean_mask)
    base = jnp.where(
        m_is_even_vel[..., None], A_base_even, A_base_odd
    )  # (Nm, 1, Nr, Nr)
    corr = jnp.where(
        m_is_even_vel[..., None],
        D1_even + inv_r[:, None] * eye_Nr,
        D1_odd + inv_r[:, None] * eye_Nr,
    )
    Lv = base + diag[..., None] * eye_Nr + coeff[..., None] * corr
    return Lv.at[..., -1, :].set(eye_Nr[-1, :])


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
    D2_wall: Array | None = field(init=False)
    A_base_even: Array = field(init=False)
    A_base_odd: Array = field(init=False)
    A_base_pos: Array = field(init=False)
    A_base_ghost: Array = field(init=False)
    Lk_op: _WallBoundedOp = field(init=False)
    Hk_op: _WallBoundedOp = field(init=False)
    # Primitive-scheme influence columns (``None``, and therefore
    # static pytree aux-data rather than traced leaves, under
    # ``res.consistent_imm``, which has no pressure).
    v_plus_1: Array | None = field(init=False)
    v_minus_1: Array | None = field(init=False)
    q_z_1: Array | None = field(init=False)
    # The vw scheme's homogeneous `$u_r$` response to a unit `$\Phi$`
    # wall value (``None`` on the legacy path).
    ur_1: Array | None = field(init=False)
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
        ) = build_parity_reduced_matrices(self.rs, params.res.fd_order)

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

        # Wall rows of D1/D2 (parity-independent: the ghost correction
        # touches only the first ``g_rows``, never the wall).  D2's is
        # read only by the default pass, to evaluate the quad's wall
        # data on the corrector iterate, so the legacy build leaves it
        # ``None`` -- static aux-data rather than a dead traced leaf.
        self.D1_wall = jax.device_put(D1_pos[-1:, :], sharding.no_shard)
        self.D2_wall = (
            jax.device_put(D2_pos[-1:, :], sharding.no_shard)
            if params.res.consistent_imm
            else None
        )

        # Base operators.
        self.A_base_even = _build_A_base(D1_even, D2_even, self.inv_r)
        self.A_base_odd = _build_A_base(D1_odd, D2_odd, self.inv_r)

        # The same `$A_{\mathrm{base}} = D_2 + (1/r) D_1$` in the
        # *parity-reduced* ``pos``/``ghost`` pair, so an explicit-half
        # matvec can apply it as **one** :func:`_parity_y_matvec`
        # instead of a `$D_2$` matvec, a `$D_1$` matvec, a field-sized
        # `$1/r$` multiply and an add.  Exact in real arithmetic (the
        # ghost correction only ever touches the first ``g_rows``,
        # which is where ``inv_r[:g_rows]`` applies), and it halves the
        # FD GEMMs of the quad-wide stage -- measured as the largest
        # non-solve stage of the default pass.  Built for **both**
        # schemes: the legacy primitive path's ``_a_base_matvec`` and
        # its `$H_k^-$` batch compute the same combination by hand.
        self.A_base_pos = jax.device_put(
            _build_A_base(D1_pos, D2_pos, self.inv_r), sharding.no_shard
        )
        self.A_base_ghost = jax.device_put(
            _build_A_base(
                D1_ghost_np[:g_rows],
                D2_ghost_np[:g_rows],
                self.inv_r[:g_rows],
            ),
            sharding.no_shard,
        )

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
        # every operator, so its own stencil need not fit.  Both
        # `$D_2$` fits are direct, so this is ``fd_order`` under either
        # flag.  Mirrors the Cartesian build; ``_hk_bands`` reads it
        # back from the factored ``Lk``.
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

        if params.solver.backend == "pallas":
            # Pallas backend: one-program-per-mode banded sweep.
            # Operators are assembled directly in banded storage (no
            # (Nr, Nr) per mode) and factored by the setup-checked
            # no-pivot banded LU (_build_pallas_operator).
            band_even = _banded_from_dense(self.A_base_even, p_band)
            band_odd = _banded_from_dense(self.A_base_odd, p_band)

            if params.res.consistent_imm:
                # vw scheme: the dt-free Dirichlet u_r recovery operator
                # lives in the Lk_op slot (there is no pressure),
                # preserving the _hk_bands band readback.
                Lk_band = _build_Lv_dir_band_gpu(
                    D1_even,
                    D1_odd,
                    band_even,
                    band_odd,
                    1.0 - m_is_even_s,
                    m_sq,
                    self.inv_r,
                    self.inv_r2,
                    kz2_s,
                    mean_s,
                    p_band,
                )
                self.Lk_op = _build_pallas_operator([Lk_band], "Lv_dir")
            else:
                from . import _cylindrical_primitive_imm as prim

                # Lk (meff = m, pressure parity: pressure / u_z use
                # (-1)^m -> m_is_even; the u_+/u_- masks live in
                # ``_hk_bands`` / ``_hk_dense_op``).
                Lk_band = prim._build_Lk_band_gpu(
                    self.D1_wall.ravel(),
                    band_even,
                    band_odd,
                    m_is_even_s,
                    m_sq,
                    self.inv_r2,
                    kz2_s,
                    mean_s,
                    p_band,
                )
                self.Lk_op = _build_pallas_operator([Lk_band], "Lk")
            del Lk_band

            # Hk group -- the default spin pair (L_{s+}, L_{s-}), or the
            # legacy (plus, minus, z) triple: stacked into one
            # homogeneous operator and stability-checked as a group.
            if params.res.consistent_imm:
                hk_bands_fn = _hk_vw_bands
            else:
                from . import _cylindrical_primitive_imm as prim

                hk_bands_fn = prim._hk_bands
            if params.step.adaptive:
                # Verify the no-pivot LU where the Helmholtz
                # diagonal is least dominant; adaptive rebuilds at
                # dt <= dt_max then skip the check
                # (solvers._factor_pallas_operator).
                _build_pallas_operator(
                    hk_bands_fn(params.step.dt_max, fourier, self),
                    "Hk(dt_max)",
                )
            self.Hk_op = _build_pallas_operator(
                hk_bands_fn(dt, fourier, self), "Hk"
            )

        else:
            # Dense backend: full matrices are built, LU-factored
            # (donated, so the factors reuse their buffers), then
            # dropped — only the factors are kept.
            if params.res.consistent_imm:
                Lk_dense = _build_Lv_dir_dense_gpu(
                    D1_even,
                    D1_odd,
                    self.A_base_even,
                    self.A_base_odd,
                    1.0 - m_is_even_s,
                    m_sq,
                    self.inv_r,
                    self.inv_r2,
                    kz2_s,
                    mean_s,
                )
            else:
                from . import _cylindrical_primitive_imm as prim

                # Pressure parity, as in the banded branch above.
                Lk_dense = prim._build_Lk_dense_gpu(
                    self.D1_wall,
                    self.A_base_even,
                    self.A_base_odd,
                    m_is_even_s,
                    m_sq,
                    self.inv_r2,
                    kz2_s,
                    mean_s,
                )
            self.Lk_op = DenseJAXSolver(Lk_dense)
            del Lk_dense

            # Combined Hk: the default spin pair (L_{s+}, L_{s-}), or
            # the legacy (plus, minus, z) triple.
            if params.res.consistent_imm:
                self.Hk_op = _hk_vw_dense_op(dt, fourier, self)
            else:
                from . import _cylindrical_primitive_imm as prim

                self.Hk_op = prim._hk_dense_op(dt, fourier, self)

        self._derive_imm_homogeneous_data(fourier, Nm, Nkz, Nr)
        self._precompute_bulk_response(fourier, Nm, Nkz, Nr)

    def _derive_imm_homogeneous_data(
        self, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
    ) -> None:
        r"""Fill the homogeneous responses and the `$1 \times 1$`
        ``M_inv`` on-device: dispatch on ``res.consistent_imm``.

        Both schemes carry the same scalar (one-wall) capacitance
        structure; only the chain the column solves differs.

        - **default** -- :meth:`_derive_vw_homogeneous_data`: the
          `$u_r$` response of the spin-quad scheme, with no pressure to
          carry (``v_plus_1``/``v_minus_1``/``q_z_1`` stay ``None``).
        - **legacy** (flag off) --
          :func:`._cylindrical_primitive_imm.derive_homogeneous_data`:
          the `$u_\pm$` responses to a unit wall pressure, plus the
          axial potential ``q_z_1`` (``ur_1`` stays ``None``).

        Both fill ``M_inv`` and are re-run at a changed ``dt`` by
        :func:`_build_dt_leaves`.
        """
        if params.res.consistent_imm:
            self._derive_vw_homogeneous_data(fourier_, Nm, Nkz, Nr)
            return

        from . import _cylindrical_primitive_imm as prim

        prim.derive_homogeneous_data(self, fourier_, Nm, Nkz, Nr)

    def _derive_vw_homogeneous_data(
        self, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
    ) -> None:
        r"""Homogeneous data of the `$u_r$`-`$\omega_r$` scheme
        (``res.consistent_imm``).

        The pipe's single wall gives one free `$\Phi$` wall value, so
        the influence matrix is `$1 \times 1$`.  `$\Phi$` is the *sum*
        part of the evolved spin pair, so a unit wall value is applied
        to **both** slots and the response averaged:

        .. math::
            \Phi_1 = \tfrac12\bigl(L_{s+}^{H,-1} + L_{s-}^{H,-1}\bigr)
                     e_{\mathrm{wall}} , \qquad
            u_{r,1} = L_{v,\mathrm{mod}}^{-1}\,(\Phi_1)_P ,

        (wall row zeroed before the recovery, so
        `$u_r|_{\mathrm{wall}} = 0$` exactly), and
        `$M = D_{1,\mathrm{wall}} \cdot u_{r,1}$` with
        `$\alpha = -M^{-1} d_{\mathrm{wall}}$` imposing
        `$(D_1 u_r)|_{\mathrm{wall}} = 0$`.  With
        `$\omega_r|_{\mathrm{wall}} = 0$` the per-point reconstruction
        then makes tangential no-slip *emerge*.  The `$\omega$` slots
        need no column.  ``M_inv`` and the column are zeroed at the
        mean mode (packed planes; no influence there).
        """
        e_wall = (
            jnp.zeros(
                (Nm, Nkz, Nr),
                dtype=sharding.float_type,
                out_sharding=sharding.spec_imm_corr_shard,
            )
            .at[..., -1]
            .set(1.0)
        )
        # One two-component batch against the stacked spin pair: the
        # same unit wall datum through L_{s+} and L_{s-}.
        stacked = jnp.stack([e_wall, e_wall])  # (2, Nm, Nkz, Nr)
        phi_pm = self.Hk_op.solve(stacked.transpose(0, 3, 1, 2)).transpose(
            0, 2, 3, 1
        )
        phi_1 = (phi_pm[0] + phi_pm[1]) / 2
        phi_1 = phi_1.at[..., -1].set(0.0)
        ur_1 = self.Lk_op.solve(phi_1.transpose(2, 0, 1)).transpose(1, 2, 0)

        is_mean = fourier_.mean_mask[0]  # (Nm, Nkz)
        ur_1 = jnp.where(is_mean[..., None], 0.0, ur_1)
        M = jnp.einsum("j, mzj -> mz", self.D1_wall.ravel(), ur_1)
        self.M_inv = jnp.where(is_mean, 0.0, 1.0 / jnp.where(is_mean, 1.0, M))

        # Field layout (Nr, Nm, Nkz); the pressure-scheme columns are
        # static aux-data by default.
        self.ur_1 = ur_1.transpose(2, 0, 1)
        self.v_plus_1 = self.v_minus_1 = self.q_z_1 = None

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

        # The mean-mode axial Helmholtz: by default it is the mean
        # plane of the minus slot; on the legacy path the z slot of the
        # (+, -, z) group
        # IS the same operator (spliced there by the packing, see
        # :func:`_vw_spin_groups`).
        zeros = jnp.zeros_like(rhs)
        if params.res.consistent_imm:
            stack, comp = [zeros, rhs], 1
        else:
            stack, comp = [zeros, zeros, rhs], 2
        h_full = self.Hk_op.solve(
            jnp.stack(stack).transpose(0, 3, 1, 2)
        ).transpose(0, 2, 3, 1)[comp]

        # ``reshard`` (not ``device_put``): this method also runs
        # inside the jitted ``set_dt`` rebuild, where placing a
        # traced value is expressed as a resharding.
        self.h_bulk_response = jax.sharding.reshard(
            extract_mean_mode(h_full.transpose(2, 0, 1)[None])[0],
            sharding.no_shard,
        )
        H_bulk = 2 * jnp.dot(self.y_weights, self.h_bulk_response)
        self.H_bulk_inv = 1.0 / H_bulk


def _vw_spin_groups(
    fourier_: Fourier,
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    r"""Per-slot `$(\text{parity}, m_{\mathrm{eff}}^2)$` of the spin
    pair `$(+, -)$` used by the vw scheme, with the mean-plane packing
    exception folded into the `$-$` slot.

    The vw quad `$(\Phi_\pm, \omega_\pm)$` rides the **existing**
    `$H_{k,\pm}$` families: `$m_{\mathrm{eff}}^2 = (m \pm 1)^2$` on the
    velocity parity `$(-1)^{m+1}$`.  On the packed `$k^2 = 0$` plane the
    `$\omega_+$` slot carries `$u_{\theta,00}$`, whose operator
    `$((m+1)^2 = 1$`, odd parity`$)$` is already right, while the
    `$\Phi_-$` slot carries `$u_{z,00}$` and needs the mean axial
    Helmholtz `$(m_{\mathrm{eff}}^2 = 0$`, *even* parity`$)$`.  The
    parity masks are `$(N_m, 1, 1)$` and cannot express a
    `$k_z$`-dependent flip, so the caller splices the **assembled**
    bands instead; this returns the two ingredient triples.
    """
    m_s = fourier_.m[0, ..., None]  # (Nm, 1, 1)
    m_is_even_s = fourier_.m_is_even[0, ..., None]  # (Nm, 1, 1)
    m_is_even_v = 1.0 - m_is_even_s  # (-1)^{m+1}
    return (
        (m_is_even_v, (m_s + 1) ** 2),
        (m_is_even_v, (m_s - 1) ** 2),
    )


def _hk_vw_bands(
    dt: float | Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> list[Array]:
    r"""Assemble the banded spin pair `$(L_{s+}, L_{s-})$` Helmholtz
    group at *dt* (``res.consistent_imm``; Pallas backend).

    Two families, not four: the vw quad solves `$(\Phi_+, \Phi_-)$` and
    `$(\omega_+, \omega_-)$` as two separate two-component batches
    against this **same** stacked pair (:func:`_imm_iteration_vw`), so
    the operator storage is 2 band families against the primitive
    scheme's 3 `$H_k$` + 1 `$L_k$`.

    The `$-$` slot's mean plane is spliced to the mean axial Helmholtz
    (:func:`_vw_spin_groups`), which is why `$u_{z,00}$` is packed
    there and `$u_{\theta,00}$` into `$\omega_+$`.  Splicing the
    assembled band costs one transient of the band's own shape and no
    persistent storage.
    """
    p_band = flow_.Lk_op.L.shape[1]
    kz2_s = fourier_.kz2[0, ..., None]
    mean_s = fourier_.mean_mask[0, ..., None]  # (Nm, Nkz, 1)
    m_s = fourier_.m[0, ..., None]
    m_is_even_s = fourier_.m_is_even[0, ..., None]
    band_even = _banded_from_dense(flow_.A_base_even, p_band)
    band_odd = _banded_from_dense(flow_.A_base_odd, p_band)

    def _band(parity: Array, meff2: Array) -> Array:
        return _build_Hk_band_gpu(
            band_even,
            band_odd,
            parity,
            meff2,
            flow_.inv_r2,
            kz2_s,
            dt,
            params.step.implicitness,
            derived_params.nu,
            p_band,
        )

    (par_p, meff2_p), (par_m, meff2_m) = _vw_spin_groups(fourier_)
    band_plus = _band(par_p, meff2_p)
    band_minus = jnp.where(
        mean_s[..., None], _band(m_is_even_s, m_s**2), _band(par_m, meff2_m)
    )
    return [band_plus, band_minus]


def _hk_vw_dense_mats(
    dt: float | Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> list[Array]:
    r"""Dense spin pair `$(L_{s+}, L_{s-})$` at *dt*, unfactored --
    the twin of :func:`_hk_vw_bands` (which the band-vs-dense parity
    test compares against)."""
    kz2_s = fourier_.kz2[0, ..., None]
    mean_s = fourier_.mean_mask[0, ..., None]
    m_s = fourier_.m[0, ..., None]
    m_is_even_s = fourier_.m_is_even[0, ..., None]

    def _dense(parity: Array, meff2: Array) -> Array:
        return _build_Hk_dense_gpu(
            flow_.A_base_even,
            flow_.A_base_odd,
            parity,
            meff2,
            flow_.inv_r2,
            kz2_s,
            dt,
            params.step.implicitness,
            derived_params.nu,
        )

    (par_p, meff2_p), (par_m, meff2_m) = _vw_spin_groups(fourier_)
    return [
        _dense(par_p, meff2_p),
        jnp.where(
            mean_s[..., None],
            _dense(m_is_even_s, m_s**2),
            _dense(par_m, meff2_m),
        ),
    ]


def _hk_vw_dense_op(
    dt: float | Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> DenseJAXSolver:
    r"""Factored dense spin pair `$(L_{s+}, L_{s-})$` at *dt* (dense
    backend)."""
    ops = [DenseJAXSolver(M) for M in _hk_vw_dense_mats(dt, fourier_, flow_)]
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
    if params.res.consistent_imm:
        hk_bands_fn, hk_dense_fn = _hk_vw_bands, _hk_vw_dense_op
    else:
        from . import _cylindrical_primitive_imm as prim

        hk_bands_fn, hk_dense_fn = prim._hk_bands, prim._hk_dense_op
    if params.solver.backend == "pallas":
        new.Hk_op = _factor_pallas_operator(hk_bands_fn(dt, fourier_, new))
    else:
        new.Hk_op = hk_dense_fn(dt, fourier_, new)
    new._derive_imm_homogeneous_data(
        fourier_, sharding.nz_spec, sharding.nx_spec, params.res.ny
    )
    new._precompute_bulk_response(
        fourier_, sharding.nz_spec, sharding.nx_spec, params.res.ny
    )
    leaves = {
        "dt": new.dt,
        "Hk_op": new.Hk_op,
        "M_inv": new.M_inv,
        "h_bulk_response": new.h_bulk_response,
        "H_bulk_inv": new.H_bulk_inv,
    }
    if params.res.consistent_imm:
        # The vw scheme's u_r column; the pressure-scheme columns are
        # None (static aux-data) and Lk_op (= the dt-free recovery) is
        # deliberately absent -- see test_adaptive's leaf dicts.
        leaves |= {"ur_1": new.ur_1}
    else:
        leaves |= {
            "v_plus_1": new.v_plus_1,
            "v_minus_1": new.v_minus_1,
            "q_z_1": new.q_z_1,
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


def _parity_y_matvec(
    M_pos: Array,
    M_ghost: Array,
    x: Array,
    parity_sign: Array,
    component_axis: int = 0,
) -> Array:
    r"""Apply one parity-reduced FD matrix to a (stacked) field.

    `$M^{(\sigma)} x = M_{\mathrm{pos}} x
    + (-1)^{m_{\mathrm{eff}}}\,\widetilde M_{\mathrm{ghost}} x$`, with
    the ghost GEMM restricted to its `$g$` nonzero near-axis rows.
    *parity_sign* broadcasts against the result, so a stacked *x* can
    carry a different parity per component (and, on the packed mean
    plane, per mode).

    The ghost scatter has to land on the **wall-normal** axis, whose
    position follows *component_axis*: leading for a 3-d *x* or the
    transpose-free ``component_axis=1`` stacking, but axis 1 when a 4-d
    *x* is component-leading.  Getting that wrong corrupts the first
    `$g$` *components* instead of the first `$g$` radial rows, silently
    and without a shape error, so the axis is derived here rather than
    left to each call site.
    """
    g = M_ghost.shape[0]
    out = apply_y_matrix(M_pos, x, component_axis=component_axis)
    ghost = apply_y_matrix(M_ghost, x, component_axis=component_axis)
    if x.ndim == 4 and component_axis == 0:
        return out.at[:, :g].add(parity_sign * ghost)
    return out.at[:g].add(parity_sign * ghost)


# ── IMM iteration (1x1) ─────────────────────────────────────────


def _imm_iteration_vw(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> tuple[Array, Array]:
    r"""`$u_r$`-`$\omega_r$` step via the spin quad
    (``res.consistent_imm``).

    The pipe's form of the reconstruction scheme whose derivation the
    Cartesian ``_imm_iteration_vw`` carries and whose cylindrical
    algebra ``annular._imm_iteration_vw`` sets out in full: advance the
    wall-normal velocity and vorticity, *reconstruct* the tangential
    pair, never form a pressure.  Everything downstream of the implicit
    solve is the annulus's, verbatim modulo parity -- the same
    conservative-curl sources, the same exact `$L_{v,\mathrm{mod}}$`
    recovery, the same per-point reconstruction (which is what makes
    the discrete divergence vanish at every row), the same
    `$(D_1 u_r)|_{\mathrm{wall}} = 0$` influence condition, here
    `$1 \times 1$` because there is one wall.  What differs is the
    **implicit half**.

    Why the pipe evolves four scalars, not two
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The pair `$(\Phi, \omega_r) = ((\Delta\mathbf{u})_r, \omega_r)$`
    does not close: each diffuses against its `$\theta$` partner
    through the `$-2im/r^2$` spin coupling.  On the annulus that
    coupling is lagged to the corrector iterate, contracting at
    `$\rho \le 0.02$` in its shipped configurations -- a corner, not a
    bound, and ``annular._imm_iteration_vw`` carries the caveat.  Near
    the pipe axis it **diverges** (measured
    worst `$\rho = 1.13$`, and `$19.1$` on the retired `$x = r^2$`
    fit, whose sharper near-axis stencils amplified the loop), so it
    cannot be iterated at all.

    The fix is to evolve the *spin combinations*, which diagonalise
    that coupling exactly -- the same trick `$u_\pm = u_r \pm i
    u_\theta$` already plays for the primitive scheme.  With
    `$\Phi_\pm := (\Delta\mathbf{u})_\pm$` and
    `$\omega_\pm := \omega_r \pm i\omega_\theta$`,

    .. math::
        (\Delta\mathbf{u})_\pm = L_{s\pm}\,u_\pm , \qquad
        L_{s\pm} = A_{\mathrm{base}}^{(v)}
                 - \frac{(m \pm 1)^2}{r^2} - k_z^2 ,

    which are **the operators the solver already builds** for
    `$H_{k,\pm}$`; the vorticity pair, being a vector's transverse
    pair too, rides the same two.  So the coupled Crank-Nicolson system
    is solved *exactly* by four scalar Helmholtz solves over two
    operator families, and the sums

    .. math::
        \Phi = \tfrac12(\Phi_+ + \Phi_-), \qquad
        \omega_r = \tfrac12(\omega_+ + \omega_-)

    feed the recovery.  Nothing in the **interior** is Picard-iterated
    -- the spin coupling the annulus lags is diagonalised exactly here,
    not lagged.  The one iterated quantity is the pair of free wall
    differences below, whose loop the corrector's own contraction
    bounds (and reports).  Cost: five per-mode banded solves
    against the primitive scheme's four, over **three** band families
    against its four (the quad shares two; the recovery is
    ``dt``-free), all at half-width ``fd_order``.

    That extra solve is why the pipe is the one geometry where this
    flag costs throughput: measured per step on an H100,
    ``res.consistent_imm`` is **-17 %** on plane-couette and **-12 %**
    on Taylor-Couette -- both of which go 4 solves to 3 -- against
    **+6 %** here.  Memory moves the other way for all three (four band
    families to three, and the pressure-response columns are replaced
    by the cheaper `$u_r$` ones).  The trade is forced, not chosen: the
    `$\mp 2im/r^2$` spin coupling is what the annulus lags and the axis
    forbids lagging, so exact diagonalisation -- and the doubling it
    brings -- is the only route here.

    Why this pass costs ~2x Cartesian, measured
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Those figures count *solves*, and the pipe's cost is not in them.
    Measured with ``scripts/pallas_solve_profile.py`` Parts B/A2 on
    CPU, one device, at matched resolution (`$128^3$`, ``fd_order 8``):

    ==============  =================  ==============
    geometry        ``_imm_iteration``  isolated Lk+Hk
    ==============  =================  ==============
    plane-couette   350 ms              278 ms
    taylor-couette  448 ms              274 ms
    pipe            749 ms              275 ms
    ==============  =================  ==============

    The solve cost is **geometry-independent to 1.5 %**, so the whole
    spread is non-solve.  Do *not* subtract the isolated solve from
    ``_imm_iteration`` to get "non-solve work": the isolated timing
    over-counts the fused one and the difference goes negative in the
    Cartesian row.

    The annulus is the control that attributes the rest, since it
    shares every curvilinear cost (`$u_\pm$` basis crossings, the
    `$1/r$` metric, the `$A_{\mathrm{base}}$` pair) but has neither the
    spin quad nor the parity reduction: curvilinear accounts for
    `$1.28\times$`, the quad and parity for a further `$1.67\times$`.
    Within this pass the two `$A_{\mathrm{base}}$` stages -- the
    quad-wide explicit CN half (18 % of the pass) and the stage-1 pair
    assembly (17 %) -- were together about equal to the solves, while
    the mechanisms the quad adds are individually small: parity costs
    only `$1.26\times$` a plain GEMM, quad assembly 0.9 %, the basis
    crossings 4.5 %, the metric multiplies 0.2 %.  So the excess is
    matvec **volume** (a 4-wide quad, each matvec parity-doubled), not
    the parity machinery -- which is what made fusing
    `$D_2 + (1/r) D_1$` into one operator the lever, worth ~10 % of
    this pass and ~11 % of the annulus's (interleaved A/B, both
    orderings).

    A related idea, measured and **rejected**: this pass is dense in
    real-coefficient products on complex fields (`$1/r$`, `$1/r^2$`,
    `$k_z^2$`, `$m_{\mathrm{eff}}^2$`, the parity signs), and each
    promotes its real operand to ``c128`` and runs a full complex
    multiply -- 4 real multiplies where 2 would do.  Hand-splitting
    them buys nothing: the products move ~24 bytes per element for 2-4
    flops, so they are memory-bound and the extra multiplies are free
    (three interleaved repeats straddle zero: +25 %, +4 %, -29 %).
    The promotion is also bit-identical to the split form, since
    `$(w + 0i)(a + bi)$` evaluates the zero cross-terms exactly.

    Boundary conditions, and the two iterated wall differences
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Split the quad's wall data into sums and differences.  The **sums**
    are the physical set: `$\omega_r|_{\mathrm{wall}} = 0$`, and
    `$\Phi|_{\mathrm{wall}}$` is the influence-matrix unknown (taken
    zero in the particular solve, corrected by `$\alpha$`).  The
    **differences** `$(\Delta\mathbf{u})_\theta|_{\mathrm{wall}}$` and
    `$\omega_\theta|_{\mathrm{wall}}$` have no boundary condition at
    all -- the latter is the wall shear -- so they are evaluated on the
    corrector **iterate**, which at the fixed point places them at
    `$t^{n+1}$`.  They cancel exactly out of both sums
    (`$(+d) + (-d) = 0$` in floating point), so `$\Phi_{arb}$` and
    `$\omega_r$` still vanish at the wall to the last bit and the
    downstream identities are untouched.  Cost: three wall-row
    contractions per pass against two wall-row vectors (``D1_wall``,
    ``D2_wall``) -- `$O(N_r)$` each, not GEMMs -- and no parity
    handling, since the ghost correction only ever touches the first
    `$g$` rows while the wall is the last.

    Having *four* wall values against *two* conditions is the price of
    the spin diagonalisation above, and it is unique to this geometry:
    Cartesian and annular evolve exactly as many scalars as they have
    conditions plus the influence unknown, so neither has a free wall
    value to source at all.

    Why the iterate and not `$t^n$` -- measured
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Lagging the two differences to `$t^n$`, which this scheme did until
    2026-08-01, is **unstable**, and invisible until it is fatal.  What
    the sums do not cancel is `$(L_{s+}^{-1} - L_{s-}^{-1})\,d$`: the
    two spin families differ by `$4m/r^2$` in `$m_{\mathrm{eff}}^2$`
    against a Helmholtz scale `$1/(c\nu\Delta t)$`, so the leftover is
    small only while `$\nu/\Delta r$` is.  Being computed from the
    state, it fed the next step's wall data, closing a growth loop
    **across time steps**, where nothing damped or observed it.
    Measured on ``pipe`` (`$32^2$` transverse modes, `$l_z = 5$`,
    `$\Delta t = 0.01$`, random IC of amplitude 0.1), lagged against
    iterated, with a legacy-path control clean in every row:

    - `$\mathrm{Re} = 1$` / `$n_r = 32$`: lagged non-finite at
      `$t = 0.37$`; iterated decays monotonically to 1.7e-4 over 100
      steps (legacy 1.6e-4).
    - `$\mathrm{Re} = 10$` / `$n_r = 32$`: lagged non-finite at
      `$t = 2.06$`; iterated clean, 3.0e-4 at `$t = 3$`.
    - `$\mathrm{Re} = 100$` / `$n_r = 64$`: lagged tracked the legacy
      path to
      **six significant figures for 600 steps** and then departed
      exponentially (0.65 against 1.5e-2 at `$t = 9$`); iterated tracks
      it throughout (1.733550e-2 against 1.733529e-2 at step 800).
    - `$\mathrm{Re} = 100$` / `$n_r = 128$`: lagged non-finite at
      `$t = 5.1$`; iterated 9.351495e-3 against the legacy path's
      9.351498e-3
      at step 999 -- seven significant figures.
    - `$\mathrm{Re} = 1800$` / `$n_r = 128$`, the shipped
      ``pipe-consistent-imm`` regime at a production wall-normal
      resolution: both forms clean and identical to seven significant
      figures (2.758548e-1 at step 1999).  **The repair is a no-op
      where the lag was already benign** -- and its price is nil:
      identical ``pipe-consistent-imm`` temporal self-convergence to
      four significant figures (1.130e-3 / 5.422e-4 / 2.357e-4, orders
      1.06 / 1.20) and a step time inside CPU noise.

    Two properties of the old failure say what a guard for this class
    of defect has to look like.  Its growth rate was proportional to
    `$\nu$` and **independent of `$\Delta t$`** (a 10x smaller step
    diverged at the same physical time, so no step reduction helped),
    and its boundary was crossed by **refinement** at fixed
    `$\mathrm{Re}$`.  A fixed-horizon, fixed-resolution smoke entry can
    see neither; what catches it is a default-vs-legacy comparison at
    the intended `$(\mathrm{Re}, n_r)$`, read digit by digit.  It also
    had nothing to do with the polymer, though it was first found and
    misattributed there: ``viscoelastic-pipe`` reproduced every row,
    including at `$\beta = 1$` where the polymer stress is decoupled
    from the velocity entirely, and raising `$\kappa$` 200x changed
    nothing.

    Zeroing the differences instead of lagging them -- formally as
    admissible, since only the sums are physical -- was also tried and
    is *worse* than the lag (`$t \approx 0.35$` against `$0.37$`): they
    are load-bearing, not arbitrary.  Record:
    ``investigate-consistent-imm-viscoelastic-pipe-axial-heron.md``.

    Parity
    ~~~~~~
    All four evolved scalars carry the velocity parity
    `$(-1)^{m+1}$`, like `$u_\pm$` themselves: `$L_{s\pm}$` preserves
    parity, and `$\omega_r = (im/r)u_z - ik_z u_\theta$`,
    `$\omega_\theta = ik_z u_r - D_1 u_z$` each flip `$u_z$`'s
    `$(-1)^m$` exactly once.  In the sources, `$C_z$` and its operands
    `$N_z$` and `$r N_\theta$` carry `$(-1)^m$` -- so the two inner
    `$D_1$` applications there are `$z$`-parity and batch with
    `$D_1 u_z$`.  The recovery's `$(D_1 + 1/r)$` correction carries an
    **odd** `$1/r^3$` coefficient, which is what keeps
    `$L_{v,\mathrm{mod}}$` inside the `$u_r$` parity class.

    Mean mode
    ~~~~~~~~~
    At `$k^2 = 0$` the reconstruction is singular and all four evolved
    scalars are structurally zero, so two of the slots carry the mean
    axial and azimuthal momentum instead.  Which two is fixed by the
    operators: `$\omega_+$` already *is* the mean `$u_\theta$` operator
    (`$(m+1)^2 = 1$`, odd parity at `$m = 0$`), while `$u_{z,00}$`
    needs `$(m_{\mathrm{eff}}^2 = 0$`, even parity`$)$` -- one splice,
    placed on the `$-$` family's mean plane (:func:`_vw_spin_groups`),
    which is why `$u_{z,00}$` rides `$\Phi_-$`.  `$\Phi_+$` and
    `$\omega_-$` are dead there.  Both packed updates then reproduce
    the primitive scheme's mean-mode update term for term (at `$m = 0$`
    the mean pressure gradient `$D_1 p$` is the same in both `$u_\pm$`
    rows, so it cancels out of `$u_\theta$` exactly as the mean-`$u_r$`
    projection removes it).  Padding modes need no special-casing.
    """
    c = params.step.implicitness
    dt = flow_.dt
    nu = derived_params.nu

    m = fourier_.m
    im = 1j * m
    ikz = 1j * fourier_.kz
    kz2 = fourier_.kz2
    inv_r = flow_.inv_r[:, None, None]
    inv_r2 = flow_.inv_r2[:, None, None]
    mean_mask = fourier_.mean_mask
    psp = fourier_.m_is_even * 2 - 1  # (-1)^m   (u_z, N_z, C_z)
    psv = -psp  # (-1)^{m+1} (u_r, u_theta, the quad)

    # Stage 0: cross into physical components.  No corrector iterate
    # enters the linear part -- the spin quad makes every linear term
    # implicit -- so only the nonlinear CN combination is needed.
    state_n = from_pm_basis(velocity_n)
    nonlin = from_pm_basis(c * nonlin_j + (1 - c) * nonlin_n)

    # Stage 1: one batched `$A_\mathrm{base}$` over the v-parity state
    # pair, and one batched D1 over the three z-parity fields
    # (D1_pos/D1_ghost are parity-independent; only the ghost sign
    # differs).  The pair needs `$D_2 + (1/r) D_1$` and nothing else
    # from `$D_1$`, so it takes the fused operator rather than riding
    # the D1 stack: the pair costs 2 GEMMs where a `$D_1$` and a
    # `$D_2$` were 4, taking the stage from 7 to 5, and the field-sized
    # `$1/r$` multiply-add over the pair goes with them.
    # GEMM counts here and below are **full-width `pos` field-GEMMs**
    # -- one `$N_r \times N_r$` matrix against one field.  The
    # `$g \times N_r$` ghost partner of each rides along at ~`$g/N_r$`
    # of that cost and is not counted.
    pair_n = jnp.stack([velocity_n[1], velocity_n[2]], axis=1)
    d1_in = jnp.stack(
        [
            state_n[0],  # u_z^n       (z) -> omega_theta
            nonlin[0],  # N_z          (z) -> C_theta
            flow_.rs[:, None, None] * nonlin[2],  # (z) -> C_z
        ],
        axis=1,
    )
    par_v2 = jnp.stack([psv, psv], axis=1)
    d1 = _parity_y_matvec(
        flow_.D1_pos,
        flow_.D1_ghost,
        d1_in,
        jnp.stack([psp, psp, psp], axis=1),
        component_axis=1,
    )
    inv_r2_y = inv_r2[..., None]  # (Nr, 1, 1, 1) over the C axis
    kz2_y = kz2[:, None]
    A_pair = _parity_y_matvec(
        flow_.A_base_pos,
        flow_.A_base_ghost,
        pair_n,
        par_v2,
        component_axis=1,
    )

    # Stage 2: the evolved quad, recomputed on FULL rows (wall
    # included) from the carried u_+/u_- state.
    meff2_pm = jnp.stack([(m + 1) ** 2, (m - 1) ** 2], axis=1)
    phi_pm = A_pair - (meff2_pm * inv_r2_y + kz2_y) * pair_n
    ur_n, ut_n = state_n[1], state_n[2]
    om_r_n = im * inv_r * state_n[0] - ikz * ut_n
    om_t_n = ikz * ur_n - d1[:, 0]  # D1 u_z^n

    def _pack(minus_slot: Array, plus_val: Array, minus_val: Array) -> Array:
        """Mean-plane packing of one spin pair (docstring)."""
        return jnp.stack(
            [
                jnp.where(mean_mask, plus_val, minus_slot[:, 0]),
                jnp.where(mean_mask, minus_val, minus_slot[:, 1]),
            ],
            axis=1,
        )

    zero = jnp.zeros_like(mean_mask, dtype=phi_pm.dtype)
    phi_pm = _pack(phi_pm, zero, state_n[0])  # Phi_- carries u_z00
    om_pm_n = _pack(
        jnp.stack([om_r_n + 1j * om_t_n, om_r_n - 1j * om_t_n], axis=1),
        ut_n,  # omega_+ carries u_theta00
        zero,
    )

    # Stage 3: the pressure-free sources -- the discrete double curl,
    # with the conservative C_z that annihilates a discrete gradient
    # exactly (the annular docstring).
    C_r = im * inv_r * nonlin[0] - ikz * nonlin[2]
    C_t = ikz * nonlin[1] - d1[:, 1]  # D1 N_z
    C_z = inv_r * (d1[:, 2] - im * nonlin[1])
    d1_Cz = _parity_y_matvec(flow_.D1_pos, flow_.D1_ghost, C_z, psp)
    cc_r = im * inv_r * C_z - ikz * C_t
    cc_t = ikz * C_r - d1_Cz
    S_phi = _pack(
        jnp.stack([-(cc_r + 1j * cc_t), -(cc_r - 1j * cc_t)], axis=1),
        zero,
        nonlin[0],
    )
    S_om = _pack(
        jnp.stack([C_r + 1j * C_t, C_r - 1j * C_t], axis=1),
        nonlin[2],
        zero,
    )

    # Stage 4: the explicit CN half of all four slots.  The minus
    # family's mean plane carries the spliced mean axial Helmholtz, so
    # its m_eff^2 and ghost sign take the same exception the band does.
    quad = jnp.concatenate([phi_pm, om_pm_n], axis=1)  # (Nr, 4, Nm, Nkz)
    # The parity signs and ``(m + 1)^2`` ride ``m``'s spec, which is
    # unsharded on the k_z (np1) axis, while their ``jnp.where``
    # siblings inherited the mean mask's full one -- so each broadcast
    # must be given the target sharding explicitly or the stacks below
    # are cross-spec operand mismatches under np1 > 1.
    psv_b = jnp.broadcast_to(
        psv, mean_mask.shape, out_sharding=sharding.spec_scalar_shard
    )
    psv_m = jnp.where(mean_mask, psp, psv)
    par_quad = jnp.stack([psv_b, psv_m, psv_b, psv_m], axis=1)
    meff2_m = jnp.where(mean_mask, m**2, (m - 1) ** 2)
    meff2_p = jnp.broadcast_to(
        (m + 1) ** 2,
        mean_mask.shape,
        out_sharding=sharding.spec_scalar_shard,
    )
    meff2_quad = jnp.stack([meff2_p, meff2_m, meff2_p, meff2_m], axis=1)
    # One fused `$A_\mathrm{base}$` matvec over the whole quad: 4 GEMMs
    # instead of 8, and the field-sized `$1/r$` multiply-add over four
    # components goes with them.  This stage is the pass's largest
    # non-solve cost, so it is where the fusion pays most.
    A_quad = _parity_y_matvec(
        flow_.A_base_pos,
        flow_.A_base_ghost,
        quad,
        par_quad,
        component_axis=1,
    )
    lapl_quad = A_quad - (meff2_quad * inv_r2_y + kz2_y) * quad
    R_quad = (
        quad / dt
        + (1 - c) * nu * lapl_quad
        + jnp.concatenate([S_phi, S_om], axis=1)
    )

    # Wall row: the sums take zero (omega_r's physical value, and
    # Phi's arbitrary particular choice); the differences are evaluated
    # on the corrector ITERATE, so the fixed point carries them at
    # t^{n+1} and no lag survives (docstring).  Two wall-row dot
    # products, not GEMMs -- and no parity handling, because the ghost
    # correction only ever touches the first g rows.
    pair_j = velocity_j[1:3]
    d1w_pair = jnp.einsum("j, cjmz -> cmz", flow_.D1_wall.ravel(), pair_j)
    d2w_pair = jnp.einsum("j, cjmz -> cmz", flow_.D2_wall.ravel(), pair_j)
    meff2_w = meff2_pm[0]  # (2, Nm, 1); wall-independent
    phi_w = (
        d2w_pair
        + inv_r[-1] * d1w_pair
        - (meff2_w * inv_r2[-1] + kz2) * pair_j[:, -1]
    )
    state_j_w = from_pm_basis(velocity_j[:, -1])
    om_t_w = ikz[0] * state_j_w[1] - jnp.einsum(
        "j, jmz -> mz", flow_.D1_wall.ravel(), velocity_j[0]
    )
    d_phi = (phi_w[0] - phi_w[1]) / 2
    d_om = 1j * om_t_w
    wall = jnp.where(
        mean_mask[0], 0.0, jnp.stack([d_phi, -d_phi, d_om, -d_om])
    )
    R_quad = R_quad.at[-1].set(wall)

    # Two two-component batches against the same stacked spin pair.
    phi_arb_pm = flow_.Hk_op.solve(R_quad[:, :2], component_axis=1)
    om_pm = flow_.Hk_op.solve(R_quad[:, 2:], component_axis=1)
    phi_arb = (phi_arb_pm[:, 0] + phi_arb_pm[:, 1]) / 2
    omega_new = (om_pm[:, 0] + om_pm[:, 1]) / 2

    # Stage 5: exact recovery of u_r.  Lk_op holds L_v,mod here,
    # with a Dirichlet identity wall row; phi_arb and omega_new both
    # vanish at the wall, so u_r|wall = 0 exactly.
    det = kz2 + fourier_.m2 * inv_r2
    inv_det = 1.0 / jnp.where(mean_mask, 1.0, det)
    om_shift = 2.0 * m * fourier_.kz * inv_r2 * inv_det
    ur_arb = flow_.Lk_op.solve(phi_arb - om_shift * omega_new)

    # Stage 6: influence matrix (1x1) -- the free Phi wall value that
    # makes (D1 u_r)|wall = 0.
    d_wall = jnp.einsum("j, jmz -> mz", flow_.D1_wall.ravel(), ur_arb)
    ur_new = ur_arb + (-flow_.M_inv * d_wall)[None] * flow_.ur_1

    # Stage 7: per-point reconstruction of (u_z, u_theta) from the
    # continuity row and the omega_r definition.
    d1_ur = _parity_y_matvec(flow_.D1_pos, flow_.D1_ghost, ur_new, psv)
    chi = -(d1_ur + inv_r * ur_new)
    b_th = im * inv_r
    uz_new = (-ikz * chi - b_th * omega_new) * inv_det
    ut_new = (-b_th * chi + ikz * omega_new) * inv_det

    # Stage 8: unpack the mean plane (which inv_det left at zero) and
    # zero the mean-mode u_r, which continuity forces.
    uz_new = jnp.where(mean_mask, phi_arb_pm[:, 1], uz_new)
    ut_new = jnp.where(mean_mask, om_pm[:, 0], ut_new)
    ur_new = jnp.where(mean_mask, 0.0, ur_new)

    if params.phys.driving == "constant_bulk_velocity":
        # Zero the mean-mode perturbation bulk axial velocity.  Like
        # every mean-plane write, this is confined to k^2 = 0, the one
        # plane the reconstruction never touches.
        mean_uz = extract_mean_mode(uz_new[None])[0].real
        bulk_uz = 2 * jnp.dot(flow_.y_weights, mean_uz)
        uz_new = uz_new + jnp.where(
            mean_mask,
            -bulk_uz * flow_.H_bulk_inv * flow_.h_bulk_response[:, None, None],
            0.0,
        )

    velocity_new = to_pm_basis(jnp.stack([uz_new, ur_new, ut_new]))
    correction = velocity_new - velocity_j

    return velocity_new, correction


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> tuple[Array, Array]:
    r"""One implicit cylindrical step: dispatch on
    ``res.consistent_imm``.

    Two formulations of the same second-order-in-time scheme, sharing
    the carried `$(u_z, u_+, u_-)$` state, the signature, the parity
    reduction and the `$1 \times 1$` shape of the influence matrix:

    - **on, the default** -- :func:`_imm_iteration_vw`, the
      `$u_r$`-`$\omega_r$` formulation via the spin quad: advance the
      radial velocity and vorticity, reconstruct `$(u_z, u_\theta)$`,
      never form a pressure.
    - **off, the legacy path** --
      :func:`._cylindrical_primitive_imm._imm_iteration_vp`, the
      primitive Kleiser-Schumann influence-matrix method: solve for
      `$(u_z, u_+, u_-)$` against a pressure Poisson solve, enforcing
      continuity at the wall.  Kept for reference and for reproducing
      older trajectories; not recommended.

    The branch is a Python ``if`` on a parameter fixed before this
    module is imported, so it costs nothing at trace time and the two
    bodies never mix.  The legacy body lives in a sibling module
    imported only here, so the default path never loads it.

    Why there are two, and why the second one is *this* one, is
    derived once for all three geometries in the Cartesian dispatcher
    :func:`~dnsjax.geometries.wall_bounded.cartesian._imm_iteration`.
    The pipe's amendment to that record: route 1
    (`$D_2 := D_1 D_1$` on an axis-regular
    `$x = r^2$` fit, plus a 1-wall boundary closure) reaches
    `$d \sim 6\times10^{-5}$` and can go no further -- the structural
    invariant `$\mathrm{diag}(\Theta) + \mathrm{diag}(\Phi) = 2/r^2$`
    forbids both radial parities' `$1/r$` commutators vanishing at
    once, so a stepped state always keeps the other parity's residual
    -- and, being built on a *composed* `$D_2$`, is not
    grid-scale-dissipative, so it needs a resolved initial condition.
    The reconstruction has neither limitation: it needs no operator
    identity at all, both `$D_2$` fits stay direct, and the residual is
    machine-eps and flat under refinement on any initial condition.
    That failure is also why the `$x = r^2$` fit has no remaining job
    (:func:`build_parity_reduced_matrices`).
    """
    if params.res.consistent_imm:
        return _imm_iteration_vw(
            velocity_n, velocity_j, nonlin_n, nonlin_j, fourier_, flow_
        )

    from . import _cylindrical_primitive_imm as prim

    return prim._imm_iteration_vp(
        velocity_n, velocity_j, nonlin_n, nonlin_j, fourier_, flow_
    )


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
    Callable[[], Array],
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

    Returns ``(init_state_bound, predict_and_fully_correct,
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
