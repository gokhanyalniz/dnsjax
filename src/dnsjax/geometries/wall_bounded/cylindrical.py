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
:mod:`dnsjax.__main__`).

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

What lives where
----------------
This module owns the *geometry*: the wavenumber class, the radial grid
and its parity-reduced FD matrices, the band families and per-mode
operators, the norms, and the ``CylindricalFlow`` dataclass.  The
**stepping** built on them -- the pseudo-spectral RHS, the FFT-free
base-flow coupling, the influence-matrix pass, the predictor /
corrector / norm and the stepper factory -- lives once in
:mod:`._cylindrical_stepping`, shared with the curved (toroidal) pipe
and parametrised by the flow dataclass; the names below re-exported
from it keep this module the single import site it has always been.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

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
from ...operators import (
    complex_harmonics,
    pad_harmonics,
    real_harmonics,
)
from ...parameters import derived_params, params
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
)
from ._cylindrical_stepping import (
    CARRIED_FIELDS,  # noqa: F401 — re-exported
    CFL_NAMES,  # noqa: F401 — re-exported
    DRIVING_KEY_Z,  # noqa: F401 — re-exported
    N_CARRIED,
    ModeColumns,  # noqa: F401 — re-exported
    _apply_bulk_correction,  # noqa: F401 — re-exported
    _correct,  # noqa: F401 — re-exported
    _curl_fn,  # noqa: F401 — re-exported
    _get_rhs,  # noqa: F401 — re-exported
    _get_rhs_measured,  # noqa: F401 — re-exported
    _imm_iteration,  # noqa: F401 — re-exported
    _imm_iteration_vw,  # noqa: F401 — re-exported
    _l_bf,  # noqa: F401 — re-exported
    _norm,  # noqa: F401 — re-exported
    _parity_y_matvec,  # noqa: F401 — re-exported
    _predict,  # noqa: F401 — re-exported
    build_stepper,
    column_differences,  # noqa: F401 — re-exported
    kinematic_differences,  # noqa: F401 — re-exported
    mean_driving,  # noqa: F401 — re-exported
    with_carried,  # noqa: F401 — re-exported
)

#: The solver -> physical half of the basis boundary, which the
#: flow modules re-export so :mod:`dnsjax.__main__` can take the
#: physical view of a state without knowing which geometry it drives
#: (Cartesian and triply-periodic simply have none).  It reads the
#: three velocity slots only, so it also drops the pass's carried
#: slots.  It runs in the hot loop, so it is exported jitted:
#: ``__main__`` never jits a flow function itself, because an outer
#: jit would trace the flow's global arrays in as constants, which a
#: multi-process run refuses.  The other half, ``to_solver_basis``,
#: appends the carried slots and so needs the flow's operators: each
#: flow module binds its own
#: (:func:`._cylindrical_stepping.with_carried`).
from_solver_basis = jax.jit(from_pm_basis)


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

    Applied as one matvec wherever a field needs
    `$D_2 x + (1/r) D_1 x$` and `$D_1 x$` has no other consumer (why it
    pays: :func:`._cylindrical_stepping._imm_iteration_vw`); where
    `$D_1 x$` is reused, the split form stays.  The fused product is
    not bit-identical to the split one, so a change to it is guarded by
    ``tests/test_imm_continuity.py``, the band-vs-dense parity tests
    and the two viscoelastic suites -- not by the laminar smoke, whose
    `$u' = 0$` makes every stage zero either way.

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
    D1_ghost:
        Ghost correction for `$D_1$`
        (`$D_{1,\mathrm{even}} - D_{1,\mathrm{pos}}$`).
        Nonzero only in the first
        `$g \sim (p+2)//2$` rows near `$r = 0$`, so only
        those rows are stored: shape ``(g, Nr)``.  Applied
        via ``out.at[:g].add(...)`` so the ghost GEMM cost
        is `$g/N_r$` of the pos part instead of doubling it.
    A_base_pos, A_base_ghost:
        The radial base operator
        `$A_{\mathrm{base}} = D_2 + (1/r) D_1$` in that same
        ``pos``/``ghost`` pair, shapes ``(Nr, Nr)`` / ``(g, Nr)``.
        Every runtime consumer of `$D_2$` needs exactly this
        combination and nothing else from it, so the combination is
        what is carried; `$D_{2,\mathrm{pos}}$` and its ghost stay
        build-time locals (they still form ``D2_wall`` and the
        even/odd pair in ``__post_init__``).
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

    # ── Metric adapter surface ──────────────────────────────────
    #
    # The straight pipe's half of the surface
    # :mod:`._cylindrical_stepping` is written against; the curved
    # (toroidal) pipe overrides it in :mod:`.cylindrical_curved`.
    # Every member here is the trivial one, and ``is_curved`` gates
    # each call site at **trace** time, so the straight pipe's
    # compiled program is exactly what it was before the surface
    # existed.  ``ClassVar``s and methods are not
    # ``dataclasses.fields``, so none of this enters the pytree.

    #: Whether the metric carries curvature.  A Python ``bool`` read
    #: while tracing, never a traced value.
    is_curved: ClassVar[bool] = False
    #: Per-direction CFL column labels, in physical-component order.
    cfl_names: ClassVar[tuple[str, str, str]] = CFL_NAMES
    #: ``stats.dat`` column name for the applied mean-mode driving.
    #: One invariant spans the corrector's *aux*, the flow's
    #: ``get_driving`` and ``__main__``'s buffer width, so the geometry
    #: and its flow must name the column identically.
    driving_key: ClassVar[str] = DRIVING_KEY_Z

    @property
    def n_carried(self) -> int:
        """Trailing carried slots of the solver-basis state.

        The default pass carries its two spin-quad difference halves
        (``_cylindrical_stepping._imm_iteration_vw``); the legacy
        primitive pass carries nothing.  A property, so no pytree leaf.
        """
        return N_CARRIED if params.res.consistent_imm else 0

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
    D1_ghost: Array = field(init=False)
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
    # wall value, and the two halves of that unit response across the
    # spin pair: the difference half is what the carried `$d_\Phi$`
    # slot receives with the influence correction, the sum half what a
    # carried slot receives when its wall datum is re-anchored on the
    # accepted velocity (all ``None`` on the legacy path).
    ur_1: Array | None = field(init=False)
    phi_1_diff: Array | None = field(init=False)
    phi_1_sum: Array | None = field(init=False)
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

        # Ghost correction matrices: the difference between the
        # parity-reduced and the common (pos) part.  Stencils cross
        # r = 0 only near the axis, so just the first g rows are
        # nonzero; only those rows are stored and applied (a full
        # (Nr, Nr) ghost GEMM would cost as much as its pos
        # counterpart, doubling every FD matvec).  ``g_rows`` is the
        # union over D1 and D2, so it bounds the ghost support of
        # ``A_base_ghost`` too.
        D1_ghost_np = np.asarray(D1_even - D1_pos)
        D2_ghost_np = np.asarray(D2_even - D2_pos)
        g_rows = _ghost_row_count(D1_ghost_np, D2_ghost_np)
        self.D1_ghost = jax.device_put(D1_ghost_np[:g_rows], sharding.no_shard)

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
        # Any metric the geometry carries is built here: the grid,
        # weights and FD matrices above are its inputs, and the bulk
        # response below reads it back through ``bulk_of_mean_profile``.
        self._build_metric()
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
        phi_1_sum = (phi_pm[0] + phi_pm[1]) / 2
        phi_1 = phi_1_sum.at[..., -1].set(0.0)
        ur_1 = self.Lk_op.solve(phi_1.transpose(2, 0, 1)).transpose(1, 2, 0)

        is_mean = fourier_.mean_mask[0]  # (Nm, Nkz)
        ur_1 = jnp.where(is_mean[..., None], 0.0, ur_1)
        # The same unit response's difference half: what the influence
        # correction adds to the carried `$d_\Phi$` (zero at the wall,
        # where both slots take the same unit value).
        phi_1_diff = (phi_pm[0] - phi_pm[1]) / 2
        phi_1_diff = jnp.where(is_mean[..., None], 0.0, phi_1_diff)
        # Its sum half, wall value 1 kept: the carried slots' response
        # to a re-anchored wall datum.
        phi_1_sum = jnp.where(is_mean[..., None], 0.0, phi_1_sum)
        M = jnp.einsum("j, mzj -> mz", self.D1_wall.ravel(), ur_1)
        self.M_inv = jnp.where(is_mean, 0.0, 1.0 / jnp.where(is_mean, 1.0, M))

        # Field layout (Nr, Nm, Nkz); the pressure-scheme columns are
        # static aux-data by default.
        self.ur_1 = ur_1.transpose(2, 0, 1)
        self.phi_1_diff = phi_1_diff.transpose(2, 0, 1)
        self.phi_1_sum = phi_1_sum.transpose(2, 0, 1)
        self.v_plus_1 = self.v_minus_1 = self.q_z_1 = None

    def _precompute_bulk_response(
        self, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
    ) -> None:
        r"""Precompute the Helmholtz response for constant-bulk-
        velocity enforcement.

        Solves `$H_{k,z}\,h = \mathbf{1}$` (unit uniform RHS,
        zero wall BC) at the mean mode `$(m, k_z) = (0, 0)$`.
        The response `$h(r)$` is the velocity profile produced
        by a unit uniform **body force** over one implicit time
        step (`$H_{k,z} = I/\Delta t - c\nu L$` carries
        accelerations on its RHS), so the scaling `$G$` below is
        `$-\partial p/\partial z = -\Pi_z$` -- the sign the
        ``-dPdz'`` diagnostic reports.  Its bulk
        `$H = 2 \int_0^1 h\,r\,dr$` gives the scaling needed to zero
        the perturbation bulk velocity:

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
        self.H_bulk_inv = 1.0 / self.bulk_of_mean_profile(self.h_bulk_response)

    # ── Metric adapter: the straight-pipe (trivial) members ─────

    def rhs_extra_spec_fn(
        self, fourier_: Fourier
    ) -> Callable[[Array], Array] | None:
        r"""``vorticity_spec -> extra fields`` for the RHS transform.

        ``None`` here: the straight pipe's nonlinear term needs the
        velocity and the vorticity and nothing else.
        """
        return None

    def to_physical(
        self, velocity_phys: Array, vorticity_phys: Array
    ) -> tuple[Array, Array]:
        r"""Carried physical-space components -> the physical triad.

        The identity here: the straight pipe carries
        `$(u_z, u_r, u_\theta)$` themselves.
        """
        return velocity_phys, vorticity_phys

    def metric_rhs(
        self, nonlin_phys: Array, vorticity_phys: Array, extra_phys: Array
    ) -> Array:
        """Weighting and curvature remainder on the physical RHS.

        The identity here: the straight pipe's momentum equation
        carries no metric weight and no curvature term.
        """
        return nonlin_phys

    def divergence_defect(self, state: Array, fourier_: Fourier) -> None:
        r"""The straight divergence `$\nabla_0\cdot$` of the carried
        state, which incompressibility forces to **zero** here, and to
        an `$O(\kappa)$` field on the curved pipe.  ``None`` means
        zero, and removes every dependent term at trace time.
        """
        return None

    def mean_radial(self, defect: Array) -> None:
        r"""The `$(m, k) = (0, 0)$` radial velocity.

        ``None`` (i.e. zero) here: with no defect, continuity forces
        the mean-mode `$u_r$` to vanish identically.
        """
        return None

    def _build_metric(self) -> None:
        """Build whatever metric the geometry carries.

        Nothing here: the straight pipe's metric is the identity.  The
        curved pipe overrides it (:mod:`.cylindrical_curved`), and the
        call site in ``__post_init__`` is placed so the grid it needs is
        already built and the bulk response that reads it is not yet.
        """

    def bulk_deficit(self, uz_src: Array) -> Array:
        r"""Bulk streamwise velocity **against its target**.

        What the bulk correction has to cancel.  The straight pipe
        evolves a perturbation whose bulk should be zero, so the
        deficit is the bulk itself, `$U_b = 2\int_0^1 \bar u_z\,
        r\,dr$` -- carried entirely by the `$(0, 0)$` mode, and equal
        to the mass flux over the cross-section area.  A total-field
        flow subtracts its target here instead of changing the call
        site.
        """
        mean_uz = extract_mean_mode(uz_src[None])[0].real
        return 2 * jnp.dot(self.y_weights, mean_uz)

    def bulk_of_mean_profile(self, profile: Array) -> Array:
        r"""Bulk of a `$(0, 0)$`-mode profile.

        Scales the bulk correction: the response of one implicit step
        to a unit uniform body force is a mean-mode profile, and this
        is the bulk it carries.  Separate from :meth:`bulk_deficit`
        because a metric-weighted flux reads a *field* over the whole
        `$k = 0$` plane but a *profile* through `$m = 0$` alone.
        """
        return 2 * jnp.dot(self.y_weights, profile)


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
        # The vw scheme's u_r column and its carried-slot partner; the
        # pressure-scheme columns are None (static aux-data) and Lk_op
        # (= the dt-free recovery) is deliberately absent -- see
        # test_adaptive's leaf dicts.
        leaves |= {
            "ur_1": new.ur_1,
            "phi_1_diff": new.phi_1_diff,
            "phi_1_sum": new.phi_1_sum,
        }
    else:
        leaves |= {
            "v_plus_1": new.v_plus_1,
            "v_minus_1": new.v_minus_1,
            "q_z_1": new.q_z_1,
        }
    return leaves


# ── Stepper factory ─────────────────────────────────


def build_cylindrical_stepper(
    flow: CylindricalFlow,
) -> tuple[
    Callable[[], Array],
    Callable[[Array], tuple[Array, Array, Array, dict[str, Array]]],
    Callable[
        [Array], tuple[Array, Array, Array, dict[str, Array], dict[str, Array]]
    ],
    Callable[
        [Array, Array], tuple[Array, Array, Array, Array, dict[str, Array]]
    ],
    Callable[
        [Array, Array],
        tuple[Array, Array, Array, Array, dict[str, Array], dict[str, Array]],
    ],
    Callable[[float], None],
    Callable[[], None],
]:
    """Build time-stepping functions for a straight-pipe flow.

    Binds this geometry's ``fourier`` singleton and ``_build_dt_leaves``
    to the shared ``_cylindrical_stepping.build_stepper``; everything
    else about the returned functions is documented there.
    """
    return build_stepper(flow, fourier, _build_dt_leaves)
