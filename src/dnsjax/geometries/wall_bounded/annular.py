r"""Annular geometry: Fourier class, norms, IMM, and solvers.

Provides all geometry-general infrastructure for wall-bounded flow in
the annulus between two concentric cylinders (the Taylor-Couette
geometry): the ``Fourier`` wavenumber class, the ``AnnularFlow`` base
dataclass (radial grid on `$[r_1, r_2]$`, finite-difference matrices,
influence-matrix operators), spectral solvers (influence-matrix method,
predictor-corrector time stepping), and diagnostic helpers (norms,
perturbation energy).

This geometry is a synthesis of the two existing wall-bounded families:

- From the **cylindrical** geometry (``cylindrical.py``): the decoupled
  `$u_\pm$` velocity formulation, the azimuthal `$m$` / axial `$k_z$`
  Fourier layout, the cylindrical divergence and pressure gradient (the
  `$(m \pm 1)/r$` terms), and the `$u_\pm$` norm with radial Jacobian.
- From the **Cartesian** geometry (``cartesian.py``): two no-slip walls
  with Dirichlet/Neumann boundary conditions and the `$2 \times 2$`
  influence (capacitance) matrix.

Unlike the pipe there is **no `$r = 0$` axis**: `$r_1 > 0$`, so the
parity-reduced FD matrices, ghost corrections, and ``m_is_even`` operator
selection of the cylindrical geometry are **not** needed.  The radial
operators use a single first/second-derivative matrix `$D_1$`, `$D_2$` on
the grid `$[r_1, r_2]$`, applied identically to all azimuthal modes.

Decoupled velocity formulation
------------------------------
The cylindrical Navier-Stokes vector Laplacian couples `$u_r$` and
`$u_\theta$` through `$1/r^2$` terms.  Following Openpipeflow
(Willis 2017), we decouple them via

.. math::
    u_+ = u_r + i\,u_\theta, \qquad
    u_- = u_r - i\,u_\theta,

reducing the vector problem to three scalar Helmholtz equations with
**effective azimuthal modes**:

.. math::
    m_{\mathrm{eff}} = m + 1 \;\text{for } u_+, \qquad
    m_{\mathrm{eff}} = m - 1 \;\text{for } u_-, \qquad
    m_{\mathrm{eff}} = m     \;\text{for } u_z.

The radial operator for each component is
`$\partial_r^2 + (1/r)\partial_r - m_{\mathrm{eff}}^2/r^2$`.  The
pressure Poisson operator uses `$m_{\mathrm{eff}} = m$`.

Influence-matrix method (`$2 \times 2$`)
----------------------------------------
The annulus has two physical walls, at `$r = r_1$` and `$r = r_2$`.
Enforcing continuity `$\nabla \cdot \mathbf{u} = 0$` at both walls gives
a `$2 \times 2$` influence matrix (one boundary degree of freedom per
wall) -- the same structure as the Cartesian Kleiser-Schumann method, but
with the cylindrical `$u_\pm$` divergence and pressure-gradient operators.
See :func:`_imm_iteration`.

Driving: shear-driven and force-driven flows
--------------------------------------------
The geometry supports two driving modes via the same infrastructure:

- **Shear-driven** (Taylor-Couette): the rotating walls set an azimuthal
  circular-Couette base flow `$U_\theta(r) = A_0 r + B_0/r$` and the
  perturbation `$\mathbf{u}'$` is integrated.  The base coupling enters
  only through the rotational-form nonlinear term (see ``rhs.py``) via
  ``base_flow`` and ``curl_base_flow`` -- no hand-coded coupling terms.
- **Force-driven** (Dean flow): both walls are stationary and the
  **total** velocity is integrated (``base_flow = curl_base_flow = 0``,
  so the rotational term computes the full `$(\nabla\times\mathbf{u})
  \times\mathbf{u}$`).  An azimuthally-/axially-uniform, radius-dependent
  azimuthal body force is supplied through ``AnnularFlow.pi_theta`` (a
  radial profile, zero by default) and added at the mean mode by
  :func:`_get_rhs_core`.  See ``flows.wall_bounded.dean`` and
  :func:`dean_laminar_u_theta`.

An optional ``block_mean_spanwise_velocity`` zeroes the mean **axial**
velocity (the undriven homogeneous direction); the azimuthal mean
evolves freely.

Flow-specific modules (e.g. ``flows.wall_bounded.taylor_couette``,
``flows.wall_bounded.dean``) subclass ``AnnularFlow``, set the base flow
and/or ``pi_theta``, then call ``build_annular_stepper`` to obtain
ready-to-use time-stepping functions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ...fd import (
    build_diff_matrices,
    build_integration_weights,
    clenshaw_curtis_weights,
    local_grid_spacing,
    tanh_two_sided_grid,
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
)
from ._base import (
    apply_y_matrix,
    base_flow_coupling,
    build_wall_bounded_stepper,
    extract_mean_mode,
    frozen_profile_flow,  # noqa: F401 — re-exported
    get_inprod,  # noqa: F401 — re-exported
    get_norm,  # noqa: F401 — re-exported
    get_norm2,
    init_state,  # noqa: F401 — re-exported
    integrate_scalar,
    pad_base_flow,  # noqa: F401 — re-exported
    phys_to_spec,  # noqa: F401 — re-exported
    spec_to_phys,  # noqa: F401 — re-exported
)


@register_dataclass_pytree
@dataclass
class Fourier:
    r"""Wavenumber grids for the annular geometry.

    Identical to the cylindrical ``Fourier`` class but **without**
    ``m_is_even``: the annulus needs no parity selection (`$r_1 > 0$`).
    Broadcasting shapes match the spectral layout
    ``(Nr, Nm, Nkz)`` = ``(ny, nz-1, nx//2)``:

    - ``kz``: shape ``(1, 1, nx//2)`` -- axial wavenumber (real FFT on
      the streamwise ``x`` parameter direction, period ``geo.lx``).
    - ``m``: shape ``(1, nz-1, 1)`` -- azimuthal mode number (complex
      FFT on the ``z`` parameter direction with `$l_z = 2\pi/m_0$`);
      the resolved modes are the integer multiples `$m = m_0 j$` of the
      wedge fundamental `$m_0$` (``geo.m0``; `$m_0 = 1$` full circle).

    The coordinate mapping is:

    =============  ===========  ============  =============
    Physical       Parameter    Transform     Wavenumber
    =============  ===========  ============  =============
    `$z_{axial}$`  ``x`` (rfft) real FFT      `$k_z$`
    `$\theta$`     ``z`` (cfft) complex FFT   `$m$` (int)
    `$r$`          ``y`` (FD)   none          grid points
    =============  ===========  ============  =============

    ``k_metric`` is 2 for `$k_z > 0$` and 1 for `$k_z = 0$`, accounting
    for the Hermitian symmetry of the real FFT.  ``mean_mask`` is True
    only at the mean mode `$(m, k_z) = (0, 0)$` -- the unique
    `$m^2 + k_z^2 = 0$` mode (padding slots carry nonzero placeholder
    wavenumbers; see ``pad_harmonics`` in :mod:`dnsjax.operators`).
    """

    kz: Array = field(init=False)
    m: Array = field(init=False)
    k_metric: Array = field(init=False)
    kz2: Array = field(init=False)
    m2: Array = field(init=False)
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
        # multiply is exact and keeps the padding placeholders nonzero.
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

        # One-hot at the mean mode (m, kz) = (0, 0): the true modes
        # precede the padding, so it is global index (0, 0).
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


# ── Annular-specific norms ──────────────────────────────────────


def get_norm2_annular(
    state: Array, k_metric: Array, y_weights: Array
) -> Array:
    r"""Annular squared L2 norm for `$(u_z, u_+, u_-)$`.

    The physical velocity magnitude satisfies
    `$|u_r|^2 + |u_\theta|^2 + |u_z|^2
    = (|u_+|^2 + |u_-|^2)/2 + |u_z|^2$`, so the `$u_\pm$` components
    carry a factor of 1/2 relative to `$u_z$`.  Identical to the
    cylindrical norm; the radial Jacobian `$r$` lives in *y_weights*.

    Parameters
    ----------
    state:
        Spectral velocity in `$(u_z, u_+, u_-)$` form,
        shape ``(3, Nr, Nm, Nkz)``.
    k_metric:
        Hermitian-symmetry weight for the real FFT axis.
    y_weights:
        Radial integration weights `$w_j r_j$`.
    """
    u_z, u_plus, u_minus = state[0], state[1], state[2]
    pm_norm2 = get_norm2(jnp.stack([u_plus, u_minus]), k_metric, y_weights)
    uz_norm2 = get_norm2(u_z[None], k_metric, y_weights)
    return pm_norm2 / 2 + uz_norm2


def get_enstrophy_annular(
    state: Array,
    D1: Array,
    inv_r: Array,
    m: Array,
    kz2: Array,
    k_metric: Array,
    y_weights: Array,
) -> Array:
    r"""Enstrophy `$\langle |\nabla \mathbf{u}|^2 \rangle$` of the
    given annular state.

    Geometry-general: the *state* may be a perturbation `$\mathbf{u}'$`
    (shear-driven Taylor-Couette) or the total field `$\mathbf{u}$`
    (force-driven Dean flow).  Uses the identity split into
    radial-derivative, azimuthal, and axial terms:

    .. math::
        \langle |\nabla \mathbf{u}|^2 \rangle
        = \langle |D_1 \mathbf{u}|^2 \rangle
        + \langle |m_{\mathrm{eff}}/r\;\mathbf{u}|^2 \rangle
        + \langle k_z^2\,|\mathbf{u}|^2 \rangle

    where `$m_{\mathrm{eff}} = m$` for `$u_z$`, `$m + 1$` for `$u_+$`,
    `$m - 1$` for `$u_-$`.  Unlike the cylindrical version the radial
    derivative uses a single `$D_1$` (no parity / ghost correction).

    Parameters
    ----------
    state:
        Spectral velocity in `$(u_z, u_+, u_-)$` form,
        shape ``(3, Nr, Nm, Nkz)``.
    D1:
        First-derivative FD matrix, shape ``(Nr, Nr)``.
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
    dy_state = apply_y_matrix(D1, state)
    enstrophy_D1 = get_norm2_annular(dy_state, k_metric, y_weights)

    inv_r_3d = inv_r[:, None, None]
    state_m = jnp.stack(
        [
            m * inv_r_3d * state[0],
            (m + 1) * inv_r_3d * state[1],
            (m - 1) * inv_r_3d * state[2],
        ]
    )
    enstrophy_m = get_norm2_annular(state_m, k_metric, y_weights)

    enstrophy_kz = get_norm2_annular(state, kz2 * k_metric, y_weights)

    return enstrophy_D1 + enstrophy_m + enstrophy_kz


# ── Radial grid and FD matrices ────────────────────────────────────


def build_annular_grid(
    ny: int,
    fd_order: int,
    r_inner: float,
    r_outer: float,
    wall_grid: str | None = None,
    grid_type: str | None = None,
    grid_stretch: float = 1.5,
) -> tuple[Array, np.ndarray, np.ndarray, Array, Array]:
    r"""Build the radial grid, FD matrices, weights, and `$1/r$`.

    Grid selection (precedence):

    1. *wall_grid*: load from file (wall-to-interior order; validated to
       span `$[r_1, r_2]$`).
    2. *grid_type*: ``"tanh"`` for two-sided tanh stretching, ``"cgl"``
       for default CGL.
    3. Default: CGL of `$[-1, 1]$` affinely mapped to `$[r_1, r_2]$`,
       clustering at **both** walls.

    Parameters
    ----------
    ny:
        Number of radial grid points (`$N_r$`).
    fd_order:
        Finite-difference stencil half-bandwidth.
    r_inner, r_outer:
        Non-dimensional inner / outer radii `$r_1$`, `$r_2$`.
    wall_grid:
        Optional path to a custom radial grid file.
    grid_type:
        Named grid type (``"cgl"`` or ``"tanh"``).
    grid_stretch:
        Stretching parameter for ``grid_type="tanh"``.

    Returns
    -------
    rs:
        Radial grid on `$[r_1, r_2]$`, shape ``(ny,)``.
    D1, D2:
        First/second-derivative FD matrices, shape ``(ny, ny)``.
    y_weights:
        Integration weights with radial Jacobian `$w_j r_j$`,
        satisfying `$\sum_j w_j r_j f_j \approx
        \int_{r_1}^{r_2} f\,r\,dr$` (the grid spans the full annulus,
        so no edge extension is needed).  For the default CGL grid
        these are affine-mapped **Clenshaw-Curtis** weights
        (spectral for smooth integrands, exact for the smooth
        base/mean profiles); custom / tanh grids fall back to the
        ``fd_order`` composite rule.
    inv_r:
        `$1/r$` on the grid, shape ``(ny,)``.
    """
    mid = 0.5 * (r_inner + r_outer)
    half = 0.5 * (r_outer - r_inner)
    if wall_grid is not None:
        grid_raw = np.loadtxt(wall_grid, dtype=np.float64)
        if len(grid_raw) != ny:
            raise ValueError(
                f"Wall grid file has {len(grid_raw)} points, expected ny={ny}"
            )
        grid = grid_raw[::-1].copy()
        if not (
            np.isclose(grid[0], r_inner) and np.isclose(grid[-1], r_outer)
        ):
            raise ValueError(
                "Annular wall grid must span "
                f"[{r_inner}, {r_outer}] (got [{grid[0]}, {grid[-1]}])"
            )
        rs = jnp.asarray(grid, dtype=sharding.float_type)
    elif grid_type == "tanh":
        xi = tanh_two_sided_grid(ny, grid_stretch)  # [-1, 1]
        rs = jnp.asarray(mid + half * xi, dtype=sharding.float_type)
        is_cgl = False
    else:
        xi = -np.cos(np.arange(ny) * np.pi / (ny - 1))  # CGL [-1, 1]
        rs = jnp.asarray(mid + half * xi, dtype=sharding.float_type)
        is_cgl = True

    inv_r = 1.0 / rs
    if is_cgl:
        # The default annular grid is a CGL grid affinely mapped to
        # [r1, r2], so Clenshaw-Curtis quadrature applies: with
        # dr = half * dxi,  int_{r1}^{r2} f r dr = half * int_{-1}^1
        # f(r(xi)) r(xi) dxi.  Fold the affine scale and the Jacobian
        # r_j into the CC weights.  Spectral for smooth integrands
        # (exact for the smooth base / mean profiles: flow rate,
        # bulk-velocity response) -- there is no axis, so no parity
        # subtlety.
        y_weights = (
            jnp.asarray(
                half * clenshaw_curtis_weights(ny), dtype=sharding.float_type
            )
            * rs
        )
    else:
        # Custom / tanh grids are not CGL: use the general fd_order
        # composite rule times the Jacobian.
        w = build_integration_weights(np.asarray(rs), fd_order)
        y_weights = jnp.asarray(w, dtype=sharding.float_type) * rs
    D1, D2 = build_diff_matrices(np.asarray(rs), fd_order)
    return rs, D1, D2, y_weights, inv_r


def annular_forced_laminar_u_theta(
    rs: Array, r1: float, r2: float, C: float
) -> Array:
    r"""Analytical laminar azimuthal profile for an annular body force.

    Closed-form steady solution of the azimuthal momentum balance
    `$(1/\mathrm{Re})\,(\nabla^2 \mathbf{U})_\theta + \Pi_\theta = 0$`
    for a radius-dependent azimuthal body force
    `$\Pi_\theta = C/(r\,\mathrm{Re})$` with no-slip walls
    `$U_\theta(r_1) = U_\theta(r_2) = 0$`.  The `$1/\mathrm{Re}$`
    cancels, so the profile is **Reynolds-independent**:

    .. math::
        U_\theta(r) = -\tfrac{C}{2}\,r\ln r + \alpha\,r
        + \frac{\beta}{r},

    where the wall conditions fix

    .. math::
        \alpha = \frac{C}{2}\,
        \frac{r_1^2 \ln r_1 - r_2^2 \ln r_2}{r_1^2 - r_2^2},
        \qquad
        \beta = \frac{C}{2}\,
        \frac{(r_1 r_2)^2 (\ln r_2 - \ln r_1)}{r_1^2 - r_2^2}.

    Shared by the two force-driven annular flows: Newtonian Dean
    (:func:`dean_laminar_u_theta`, `$C = 2(r_1 + r_2)$`) and the
    viscoelastic sPTT flow (`$C = r_1 + r_2$`, the reference
    normalisation).  Pure function (no flow construction), so it is
    importable both by the ``start_from_laminar`` state and by
    :mod:`dnsjax.random_field` (the total-field IC = laminar profile +
    perturbation).

    Parameters
    ----------
    rs:
        Radial grid on `$[r_1, r_2]$`, shape ``(Nr,)``.
    r1, r2:
        Inner / outer non-dim radii.
    C:
        Body-force coefficient (`$\Pi_\theta = C/(r\,\mathrm{Re})$`).
    """
    denom = r1**2 - r2**2
    alpha = (C / 2.0) * (r1**2 * np.log(r1) - r2**2 * np.log(r2)) / denom
    beta = (C / 2.0) * (r1 * r2) ** 2 * (np.log(r2) - np.log(r1)) / denom
    return -(C / 2.0) * rs * jnp.log(rs) + alpha * rs + beta / rs


def dean_laminar_u_theta(rs: Array, eta: float) -> Array:
    r"""Analytical laminar Dean-flow azimuthal profile `$U_\theta(r)$`.

    Thin wrapper over :func:`annular_forced_laminar_u_theta` for the
    Newtonian Dean body force
    `$\Pi_\theta = (2\eta + 2)/(r\,\mathrm{Re}\,(1 - \eta))$`, i.e.
    `$C = 2(\eta + 1)/(1 - \eta) = 2(r_1 + r_2)$` on the gap-1 radii
    `$r_1 = \eta/(1-\eta)$`, `$r_2 = 1/(1-\eta)$`.

    Parameters
    ----------
    rs:
        Radial grid on `$[r_1, r_2]$`, shape ``(Nr,)``.
    eta:
        Radius ratio `$\eta = r_1/r_2$`.
    """
    C = 2.0 * (eta + 1.0) / (1.0 - eta)
    r1 = eta / (1.0 - eta)
    r2 = 1.0 / (1.0 - eta)
    return annular_forced_laminar_u_theta(rs, r1, r2, C)


# ── Operator builders ──────────────────────────────────────────────


def _build_A_base(D1: Array, D2: Array, inv_r: Array) -> Array:
    r"""Build the radial base operator
    `$A_{\mathrm{base}} = D_2 + \mathrm{diag}(1/r)\,D_1$`."""
    return D2 + inv_r[:, None] * D1


# ── Pallas-backend banded operator builders ───────────────────────


def _build_Lk_band_gpu(
    D1: Array,
    A_base: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
    p: int,
) -> Array:
    r"""Build `$L_k$` in banded storage for the Pallas backend.

    Same operator as :func:`_build_Lk_dense_gpu`
    (`$L_k = A_{\mathrm{base}} - (m^2/r^2 + k_z^2) I$`), assembled
    directly in banded layout ``(Nm, Nkz, Nr, 2p+1)`` from the base
    band ``_banded_from_dense(A_base, p)``, with no ``(Nr, Nr)`` per
    mode.  The two-wall row-setting mirrors the Cartesian builder:
    Neumann `$D_1$` rows at the inner (row 0) and outer (row Nr-1)
    walls, with a mean-mode identity pin at the outer wall.  No parity
    selection (single `$A_{\mathrm{base}}$`).
    """
    Nr = A_base.shape[0]
    band_base = _banded_from_dense(A_base, p)  # (Nr, 2p+1)
    diag = -(m2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    inner = _banded_wall_row(D1[0], 0, p)  # Neumann, inner wall
    neumann_outer = _banded_wall_row(D1[-1], Nr - 1, p)  # Neumann, outer
    outer = jnp.where(
        mean_mask, _banded_diag_column(p, band_base.dtype), neumann_outer
    )  # (Nm, Nkz, 2p+1)
    return _assemble_banded_operator(
        band_base, 1.0, diag, [(0, inner), (Nr - 1, outer)]
    )


def _build_Hk_band_gpu(
    A_base: Array,
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
    ``(Nm, Nkz, Nr, 2p+1)``:
    `$H_k = (1/\Delta t) I + c\nu (m_{\mathrm{eff}}^2/r^2 + k_z^2) I
    - c\nu A_{\mathrm{base}}$` with Dirichlet no-slip identity rows at
    **both** walls.  The caller supplies `$m_{\mathrm{eff}}^2$` for the
    component.
    """
    Nr = A_base.shape[0]
    band_base = _banded_from_dense(A_base, p)
    diag = 1.0 / dt + c * nu * (meff2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    eN = _banded_diag_column(p, band_base.dtype)  # identity wall row
    return _assemble_banded_operator(
        band_base, -c * nu, diag, [(0, eN), (Nr - 1, eN)]
    )


def _build_Lk_dense_gpu(
    D1: Array,
    A_base: Array,
    m2: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
) -> Array:
    r"""Build dense `$L_k$` on GPU (dense backend only)."""
    Nr = A_base.shape[0]
    dtype = A_base.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)

    diag_shift = -(m2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    Lk = A_base[None, None] + diag_shift[..., None] * eye_Nr

    Lk = Lk.at[..., 0, :].set(D1[0, :])  # Neumann inner
    pin = eye_Nr[-1, :]
    rowN = jnp.where(mean_mask, pin, D1[-1, :])  # Neumann outer / pin mean
    Lk = Lk.at[..., -1, :].set(rowN)
    return Lk


def _build_Hk_dense_gpu(
    A_base: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    nu: float,
) -> Array:
    r"""Build dense `$H_k$` on GPU (dense backend only)."""
    Nr = A_base.shape[0]
    dtype = A_base.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)

    diag_coeff = 1.0 / dt + c * nu * (meff2 * inv_r2 + kz2)
    Hk = diag_coeff[..., None] * eye_Nr - c * nu * A_base

    e0 = jnp.zeros(Nr, dtype=dtype).at[0].set(1.0)
    eN = jnp.zeros(Nr, dtype=dtype).at[-1].set(1.0)
    Hk = Hk.at[..., 0, :].set(e0)
    Hk = Hk.at[..., -1, :].set(eN)
    return Hk


# ── AnnularFlow base dataclass ─────────────────────────────────────

_WallBoundedOp = DenseJAXSolver | PerModeBandedPallasOperator


@register_dataclass_pytree
@dataclass
class AnnularFlow:
    r"""Precomputed data for wall-bounded annular flows.

    Subclasses must set ``base_flow`` and ``curl_base_flow`` *after*
    calling ``super().__post_init__()``, which builds the radial grid on
    `$[r_1, r_2]$`, the FD matrices, and all per-mode IMM operators.

    The velocity state is stored in decoupled form `$(u_z, u_+, u_-)$`
    with `$u_\pm = u_r \pm i\,u_\theta$`.  Three Helmholtz operators are
    built (`$m_{\mathrm{eff}} = m+1, m-1, m$` for `$u_+, u_-, u_z$`) and
    one pressure Poisson operator (`$m_{\mathrm{eff}} = m$`).

    Attributes
    ----------
    rs, inv_r, inv_r2:
        Radial grid `$r_j$`, `$1/r_j$`, `$1/r_j^2$`.
    y_weights:
        Integration weights with radial Jacobian `$w_j r_j$`.
    pi_theta:
        Mean-mode azimuthal body force on the radial grid, shape
        ``(Nr,)``; zero for shear-driven flows, set by force-driven
        subclasses (Dean flow).
    D1, D2:
        First/second-derivative FD matrices, shape ``(Nr, Nr)``.
    D1_bnd:
        Wall rows of `$D_1$` (inner, outer), shape ``(2, Nr)``.
    A_base:
        `$D_2 + \mathrm{diag}(1/r)\,D_1$`, shape ``(Nr, Nr)``.
    Lk_op, Hk_op:
        Factored pressure Poisson and stacked Helmholtz operators.
    v_plus_i, v_minus_i, q_z_i:
        Homogeneous `$u_\pm$` velocity and axial potential responses to
        a unit pressure at wall ``i`` (``1`` = inner, ``2`` = outer).
    M_inv:
        Inverse `$2 \times 2$` influence matrix per mode.
    h_bulk_response, H_bulk_inv:
        Axial-bulk-blocking response (zero unless
        ``block_mean_spanwise_velocity``).
    """

    rs: Array = field(init=False)
    inv_r: Array = field(init=False)
    inv_r2: Array = field(init=False)
    y_weights: Array = field(init=False)
    cfl_inv_spacing: Array = field(init=False)
    pi_theta: Array = field(init=False)
    base_flow: Array = field(init=False)
    curl_base_flow: Array = field(init=False)
    base_flow_padded: Array = field(init=False)
    curl_base_flow_padded: Array = field(init=False)
    base_flow_adv_padded: Array = field(init=False)
    D1: Array = field(init=False)
    D2: Array = field(init=False)
    D1_bnd: Array = field(init=False)
    A_base: Array = field(init=False)
    Lk_op: _WallBoundedOp = field(init=False)
    Hk_op: _WallBoundedOp = field(init=False)
    v_plus_1: Array = field(init=False)
    v_minus_1: Array = field(init=False)
    q_z_1: Array = field(init=False)
    v_plus_2: Array = field(init=False)
    v_minus_2: Array = field(init=False)
    q_z_2: Array = field(init=False)
    M_inv: Array = field(init=False)
    h_bulk_response: Array = field(init=False)
    H_bulk_inv: Array = field(init=False)

    def __post_init__(self) -> None:
        r"""Build radial grid, FD matrices, and IMM operators."""
        Nr = params.res.ny
        self.rs, D1_np, D2_np, self.y_weights, self.inv_r = build_annular_grid(
            Nr,
            params.res.fd_order,
            derived_params.r_inner,
            derived_params.r_outer,
            params.geo.wall_grid,
            params.geo.grid_type,
            params.geo.grid_stretch,
        )
        self.inv_r2 = self.inv_r**2

        derived_params.wall_normal_grid = [
            float(v) for v in np.asarray(self.rs)
        ]

        # Inverse local advection length scales for the CFL diagnostic,
        # per physical component (u_z, u_r, u_theta), zero in the
        # ny_y_pad rows.  Axial uses the spectral spacing L/n; radial the
        # local grid spacing; azimuthal the arc length r*dtheta with
        # dtheta = lz/nz (theta period lz = 2*pi/m0 over the wedge).
        inv_sp = np.zeros(
            (3, Nr + sharding.ny_y_pad), dtype=sharding.float_type
        )
        inv_sp[0, :Nr] = params.res.nx / params.geo.lx
        inv_sp[1, :Nr] = 1.0 / local_grid_spacing(np.asarray(self.rs))
        inv_sp[2, :Nr] = np.asarray(self.inv_r) * params.res.nz / params.geo.lz
        self.cfl_inv_spacing = jax.device_put(
            inv_sp[:, :, None, None], sharding.no_shard
        )

        # FD matrices, wall rows, and base operator.
        self.D1 = jax.device_put(D1_np, sharding.no_shard)
        self.D2 = jax.device_put(D2_np, sharding.no_shard)
        self.D1_bnd = jax.device_put(
            np.stack([D1_np[0], D1_np[-1]]), sharding.no_shard
        )
        self.A_base = _build_A_base(self.D1, self.D2, self.inv_r)

        self.rs = jax.device_put(self.rs, sharding.no_shard)
        self.inv_r = jax.device_put(self.inv_r, sharding.no_shard)
        self.inv_r2 = jax.device_put(self.inv_r2, sharding.no_shard)
        self.y_weights = jax.device_put(self.y_weights, sharding.no_shard)

        # Mean-mode azimuthal body force on the radial grid, zero by
        # default.  Force-driven subclasses (Dean flow) overwrite it;
        # applied at the mean mode by ``_get_rhs_core``.
        self.pi_theta = jnp.zeros(
            Nr, dtype=sharding.float_type, out_sharding=sharding.no_shard
        )

        Nm = sharding.nz_spec
        Nkz = sharding.nx_spec

        fd_p = params.res.fd_order
        dt = params.step.dt
        c_impl = params.step.implicitness
        # Solvent viscosity: 1/re for Newtonian Taylor-Couette / Dean,
        # beta/re for the viscoelastic subclass (see
        # ``derived_params.nu`` in ``parameters.update_parameters``).
        nu = derived_params.nu

        # Solver-internal wavenumber arrays.
        m_s = fourier.m[0, ..., None]  # (Nm, 1, 1)
        kz2_s = fourier.kz2[0, ..., None]  # (1, Nkz, 1)
        mean_s = fourier.mean_mask[0, ..., None]  # (Nm, Nkz, 1)
        m_plus_1_sq = (m_s + 1) ** 2
        m_minus_1_sq = (m_s - 1) ** 2
        m_sq = m_s**2

        if params.solver.backend == "pallas":
            # Pallas backend: one-program-per-mode banded sweep.
            # Operators are assembled directly in banded storage (no
            # (Nr, Nr) per mode) and factored by the setup-checked
            # no-pivot banded LU (_build_pallas_operator).
            Lk_band = _build_Lk_band_gpu(
                self.D1,
                self.A_base,
                m_sq,
                self.inv_r2,
                kz2_s,
                mean_s,
                fd_p,
            )
            self.Lk_op = _build_pallas_operator([Lk_band], "Lk")
            del Lk_band

            # Hk group (plus, minus, z): stacked into one homogeneous
            # operator and stability-checked as a single group.
            Hp_band = _build_Hk_band_gpu(
                self.A_base,
                m_plus_1_sq,
                self.inv_r2,
                kz2_s,
                dt,
                c_impl,
                nu,
                fd_p,
            )
            Hm_band = _build_Hk_band_gpu(
                self.A_base,
                m_minus_1_sq,
                self.inv_r2,
                kz2_s,
                dt,
                c_impl,
                nu,
                fd_p,
            )
            Hz_band = _build_Hk_band_gpu(
                self.A_base,
                m_sq,
                self.inv_r2,
                kz2_s,
                dt,
                c_impl,
                nu,
                fd_p,
            )
            self.Hk_op = _build_pallas_operator(
                [Hp_band, Hm_band, Hz_band], "Hk"
            )
            del Hp_band, Hm_band, Hz_band
        else:
            Lk_dense = _build_Lk_dense_gpu(
                self.D1, self.A_base, m_sq, self.inv_r2, kz2_s, mean_s
            )
            self.Lk_op = DenseJAXSolver(Lk_dense)
            del Lk_dense

            Hp_dense = _build_Hk_dense_gpu(
                self.A_base, m_plus_1_sq, self.inv_r2, kz2_s, dt, c_impl, nu
            )
            Hk_plus_solver = DenseJAXSolver(Hp_dense)
            del Hp_dense

            Hm_dense = _build_Hk_dense_gpu(
                self.A_base, m_minus_1_sq, self.inv_r2, kz2_s, dt, c_impl, nu
            )
            Hk_minus_solver = DenseJAXSolver(Hm_dense)
            del Hm_dense

            Hz_dense = _build_Hk_dense_gpu(
                self.A_base, m_sq, self.inv_r2, kz2_s, dt, c_impl, nu
            )
            Hk_z_solver = DenseJAXSolver(Hz_dense)
            del Hz_dense

            self.Hk_op = DenseJAXSolver.from_factors(
                lu=jnp.stack(
                    [
                        Hk_plus_solver.lu,
                        Hk_minus_solver.lu,
                        Hk_z_solver.lu,
                    ]
                ),
                perm=jnp.stack(
                    [
                        Hk_plus_solver.perm,
                        Hk_minus_solver.perm,
                        Hk_z_solver.perm,
                    ]
                ),
            )
            del Hk_plus_solver, Hk_minus_solver, Hk_z_solver

        self._derive_imm_homogeneous_data(Nm, Nkz, Nr)
        self._precompute_bulk_response(Nm, Nkz, Nr)

    def _derive_imm_homogeneous_data(self, Nm: int, Nkz: int, Nr: int) -> None:
        r"""Fill the homogeneous responses and the `$2 \times 2$`
        ``M_inv`` on-device.

        Two unit-wall pressures (`$L_k p_i = e_i$`, `$e_1$` at the inner
        wall, `$e_2$` at the outer) give, via the pressure gradient and
        the Helmholtz solves, the `$u_\pm$` responses ``v_plus_i``,
        ``v_minus_i`` and the axial potentials ``q_z_i``.  The `$u_r$`
        part is zeroed at the mean mode (continuity forces `$u_r \equiv
        0$` there).  The influence matrix
        `$M_{ji} = D_{1,\mathrm{wall}_j} \cdot (v_{+,i} + v_{-,i})/2$`
        is `$2 \times 2$`; ``M_inv`` is its inverse, set to zero at the
        mean mode (where `$d_{\mathrm{wall}} = 0$`, so the correction
        vanishes regardless).
        """
        # This run-once setup stays in the mode-outer (Nm, Nkz, Nr)
        # layout: the influence-matrix einsums below operate on it and
        # the results are transposed to field layout (Nr, Nm, Nkz) at
        # the end.  ``.solve`` now takes a mode-inner field, so each
        # setup solve is wrapped (transpose in, transpose out) to keep
        # this layout.  FUTURE: rebuild this setup natively mode-inner to
        # drop the wrappers -- the hot path already is; here it only
        # relocates a one-time transpose, so it is deferred.
        e_inner = (
            jnp.zeros(
                (Nm, Nkz, Nr),
                dtype=sharding.float_type,
                out_sharding=sharding.spec_imm_corr_shard,
            )
            .at[..., 0]
            .set(1.0)
        )
        e_outer = (
            jnp.zeros(
                (Nm, Nkz, Nr),
                dtype=sharding.float_type,
                out_sharding=sharding.spec_imm_corr_shard,
            )
            .at[..., -1]
            .set(1.0)
        )
        p1_s = self.Lk_op.solve(e_inner.transpose(2, 0, 1)).transpose(1, 2, 0)
        p2_s = self.Lk_op.solve(e_outer.transpose(2, 0, 1)).transpose(1, 2, 0)

        m_s = fourier.m[0, ..., None]  # (Nm, 1, 1)
        m_over_r_s = m_s * self.inv_r  # (Nm, 1, Nr)
        mean_s = fourier.mean_mask[0, ..., None]  # (Nm, Nkz, 1)

        def _helm_responses(p_s: Array) -> tuple[Array, Array, Array]:
            D1_p = jnp.einsum("ij, mzj -> mzi", self.D1, p_s)
            rhs_v_plus = -(D1_p - m_over_r_s * p_s)
            rhs_v_minus = -(D1_p + m_over_r_s * p_s)
            rhs_v_plus = rhs_v_plus.at[..., 0].set(0.0).at[..., -1].set(0.0)
            rhs_v_minus = rhs_v_minus.at[..., 0].set(0.0).at[..., -1].set(0.0)
            q_rhs = p_s.at[..., 0].set(0.0).at[..., -1].set(0.0)
            stacked = jnp.stack([rhs_v_plus, rhs_v_minus, q_rhs])
            res = self.Hk_op.solve(stacked.transpose(0, 3, 1, 2)).transpose(
                0, 2, 3, 1
            )
            vp, vm = res[0], res[1]
            # Zero the u_r part at the mean mode, preserving u_theta.
            vr = jnp.where(mean_s, (vp + vm) / 2, 0.0)
            return vp - vr, vm - vr, res[2]

        vp1, vm1, qz1 = _helm_responses(p1_s)
        vp2, vm2, qz2 = _helm_responses(p2_s)

        # 2x2 influence matrix M[j, i] = D1_bnd[j] . u_r^(i).
        ur1 = (vp1 + vm1) / 2
        ur2 = (vp2 + vm2) / 2
        M00 = jnp.einsum("j, mzj -> mz", self.D1_bnd[0], ur1)
        M01 = jnp.einsum("j, mzj -> mz", self.D1_bnd[0], ur2)
        M10 = jnp.einsum("j, mzj -> mz", self.D1_bnd[1], ur1)
        M11 = jnp.einsum("j, mzj -> mz", self.D1_bnd[1], ur2)

        is_mean = fourier.mean_mask[0]  # (Nm, Nkz)
        det = M00 * M11 - M01 * M10
        safe_det = jnp.where(is_mean, 1.0, det)
        # Mean mode: u_r is zeroed and d_wall = 0 there, so the
        # correction vanishes; M_inv = 0 keeps it NaN-free.
        inv_00 = jnp.where(is_mean, 0.0, M11 / safe_det)
        inv_01 = jnp.where(is_mean, 0.0, -M01 / safe_det)
        inv_10 = jnp.where(is_mean, 0.0, -M10 / safe_det)
        inv_11 = jnp.where(is_mean, 0.0, M00 / safe_det)
        self.M_inv = jnp.stack(
            [
                jnp.stack([inv_00, inv_01], axis=-1),
                jnp.stack([inv_10, inv_11], axis=-1),
            ],
            axis=-2,
        )

        # Transpose to field layout (Nr, Nm, Nkz).
        self.v_plus_1 = vp1.transpose(2, 0, 1)
        self.v_minus_1 = vm1.transpose(2, 0, 1)
        self.q_z_1 = qz1.transpose(2, 0, 1)
        self.v_plus_2 = vp2.transpose(2, 0, 1)
        self.v_minus_2 = vm2.transpose(2, 0, 1)
        self.q_z_2 = qz2.transpose(2, 0, 1)

    def _precompute_bulk_response(self, Nm: int, Nkz: int, Nr: int) -> None:
        r"""Precompute the Helmholtz response for blocking the mean
        axial velocity.

        Solves `$H_{k,z}\,h = \mathbf{1}$` (unit uniform RHS, zero
        Dirichlet at both walls) at the mean mode.  Its bulk
        `$H = \int h\,r\,dr / \mathrm{volfac}$` gives the scaling that
        zeroes the perturbation bulk axial velocity:
        `$G = -U_{b,z}/H$`, `$\bar{u}'_z \leftarrow \bar{u}'_z + G\,h$`.
        Active only when ``block_mean_spanwise_velocity`` (the undriven
        homogeneous direction for Taylor-Couette is axial `$z$`).
        """
        if not params.phys.block_mean_spanwise_velocity:
            self.h_bulk_response = jnp.zeros(
                Nr, dtype=sharding.float_type, out_sharding=sharding.no_shard
            )
            self.H_bulk_inv = jnp.zeros((), dtype=sharding.float_type)
            return

        ones_vec = (
            jnp.ones(Nr, dtype=sharding.float_type)
            .at[0]
            .set(0.0)
            .at[-1]
            .set(0.0)
        )
        rhs = jnp.where(fourier.mean_mask[0, ..., None], ones_vec, 0.0)
        zeros = jnp.zeros_like(rhs)
        h_full = self.Hk_op.solve(
            jnp.stack([zeros, zeros, rhs]).transpose(0, 3, 1, 2)
        ).transpose(0, 2, 3, 1)[2]

        self.h_bulk_response = jax.device_put(
            extract_mean_mode(h_full.transpose(2, 0, 1)[None])[0],
            sharding.no_shard,
        )
        H_bulk = (
            jnp.dot(self.y_weights, self.h_bulk_response)
            / derived_params.volume_fac
        )
        self.H_bulk_inv = 1.0 / H_bulk


# ── Solver functions ─────────────────────────────────────────────


def _curl_fn(
    state: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> Array:
    r"""Spectral curl in cylindrical coordinates (single `$D_1$`).

    Input/output in `$(u_z, u_r, u_\theta)$` representation:

    .. math::
        \omega_r = \frac{im}{r}\,u_z - ik_z\,u_\theta, \quad
        \omega_\theta = ik_z\,u_r - D_1\,u_z, \quad
        \omega_z = D_1\,u_\theta + \frac{1}{r}\,u_\theta
                 - \frac{im}{r}\,u_r
    """
    uz, ur, utheta = state[0], state[1], state[2]

    im = 1j * fourier_.m
    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]

    # Stack y-leading (N_r, 2, ...) so the batched D1 GEMM contracts the
    # leading wall-normal axis transpose-free; unstack back to 3-d.
    dy_fields = apply_y_matrix(
        flow_.D1, jnp.stack([utheta, uz], axis=1), component_axis=1
    )
    dy_utheta, dy_uz = dy_fields[:, 0], dy_fields[:, 1]

    omega_r = im * inv_r * uz - ikz * utheta
    omega_theta = ikz * ur - dy_uz
    omega_z = dy_utheta + inv_r * utheta - im * inv_r * ur

    return jnp.array([omega_z, omega_r, omega_theta])


def _l_bf(
    state: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> Array:
    r"""Linear base-flow coupling (FFT-free), in `$(u_z, u_+, u_-)$`.

    As in cylindrical: convert to the `$(u_z, u_r, u_\theta)$` triad and
    evaluate only the two *linear* base-flow cross-product terms
    (:func:`base_flow_coupling`) with the spectral :func:`_curl_fn` --
    no Fourier transform.  For shear-driven Taylor-Couette (rotating
    walls) the azimuthal base flow makes `$U_\theta\,\partial_r u'$`
    stiff on the wall-clustered grid, so the CN/AB2 scheme advances this
    term implicitly while the self-advection `$\mathbf{u}' \times
    \boldsymbol{\omega}' = \text{get\_rhs} - \text{\_l\_bf}$` stays
    explicit (the constant Dean body force ``pi_theta`` is not part of
    this coupling; it rides in the explicit term).

    With ``params.step.implicit_mean_coupling`` (default on) the
    *instantaneous mean-flow* coupling is folded in by adding the
    `$m = k_z = 0$` mean profiles of the `$(u_z, u_r, u_\theta)$`
    state and of `$\boldsymbol{\omega}'$` onto the base-flow profiles
    -- FFT-free; see the Cartesian ``_l_bf``.  For force-driven Dean
    (total field, ``base_flow = 0``) this is the *only* coupling:
    without it ``_l_bf`` is identically zero in the default lab frame
    (a moving frame adds its diagonal term), i.e. the entire evolving
    mean profile -- laminar included -- would ride the explicit AB2
    term; with it the mean-flow advection is implicit, as ``L_bf`` is
    for the perturbation-form flows.  See ``step_cnab2`` in
    :mod:`dnsjax.timestep`.
    """
    u_z, u_plus, u_minus = state[0], state[1], state[2]
    ur = (u_plus + u_minus) / 2
    utheta = -1j * (u_plus - u_minus) / 2
    state_rthz = jnp.array([u_z, ur, utheta])

    omega = _curl_fn(state_rthz, fourier_, flow_)
    base = flow_.base_flow
    curl_base = flow_.curl_base_flow
    if params.step.implicit_mean_coupling:
        base = base + extract_mean_mode(state_rthz)[:, :, None, None]
        curl_base = curl_base + extract_mean_mode(omega)[:, :, None, None]
    l_z, l_r, l_theta = base_flow_coupling(state_rthz, omega, base, curl_base)
    l_bf = jnp.array([l_z, l_r + 1j * l_theta, l_r - 1j * l_theta])
    # Moving frame: the convective frame term (the same expression
    # ``_get_rhs_core`` adds, diagonal in the native basis) belongs
    # to the linear coupling, so CN/AB2 integrates it implicitly.
    u_grid = derived_params.u_grid
    if u_grid == 0:
        return l_bf
    return l_bf + (1j * u_grid) * fourier_.kz * state


# Per-direction CFL column names, matching the physical-space component
# order (u_z, u_r, u_theta).
CFL_NAMES: tuple[str, str, str] = ("CFL_z", "CFL_r", "CFL_th")


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
    measure_fn: Callable[[Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    r"""Evaluate the nonlinear RHS in `$(u_z, u_+, u_-)$` form.

    1. Convert `$(u_z, u_+, u_-) \to (u_z, u_r, u_\theta)$`.
    2. Compute the rotational-form nonlinear term via
       :func:`~dnsjax.rhs.get_nonlin` with the cylindrical curl (and the
       optional physical-space *measure_fn*).  The base coupling enters
       here through ``base_flow_padded`` / ``curl_base_flow_padded``.
       For force-driven flows (Dean) ``base_flow`` is zero, so this is
       the full `$(\nabla\times\mathbf{u})\times\mathbf{u}$` of the
       total field.
    3. Add the mean-mode azimuthal body force ``flow_.pi_theta`` to
       `$NL_\theta$` (zero for shear-driven Taylor-Couette).
    4. Convert `$(NL_z, NL_r, NL_\theta) \to (NL_z, NL_+, NL_-)$`.
    """
    u_z, u_plus, u_minus = state[0], state[1], state[2]
    ur = (u_plus + u_minus) / 2
    utheta = -1j * (u_plus - u_minus) / 2

    state_rthz = jnp.array([u_z, ur, utheta])

    nonlin_rthz = get_nonlin(
        state_rthz,
        flow_.base_flow_padded,
        flow_.curl_base_flow_padded,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier_, flow_),
        measure_fn,
    )
    if measure_fn is not None:
        nonlin_rthz, measurements = nonlin_rthz

    NL_z, NL_r, NL_theta = nonlin_rthz[0], nonlin_rthz[1], nonlin_rthz[2]
    # Mean-mode azimuthal body force (Dean flow), applied only at the
    # mean mode (m, k_z) = (0, 0); zero for shear-driven Taylor-Couette.
    # get_nonlin returns the +u x omega rotational term, so the force
    # enters the RHS with a + sign (and into NL_+/- via NL_theta below).
    NL_theta = NL_theta + jnp.where(
        fourier_.mean_mask, flow_.pi_theta[:, None, None], 0.0
    )
    NL_plus = NL_r + 1j * NL_theta
    NL_minus = NL_r - 1j * NL_theta

    rhs = jnp.array([NL_z, NL_plus, NL_minus])
    # Moving frame: convective-form frame term
    # `$+ i k_z U_{grid} \mathbf{u}'$` -- the axial derivative is
    # component-diagonal in the `$(u_z, u_+, u_-)$` basis, so it is
    # added on the native state (mode-diagonal, divergence-free; see
    # ``pad_base_flow``).
    u_grid = derived_params.u_grid
    if u_grid != 0:
        rhs = rhs + (1j * u_grid) * fourier_.kz * state
    if measure_fn is None:
        return rhs
    return rhs, measurements


def _get_rhs(
    state: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> Array:
    r"""Evaluate the nonlinear RHS in `$(u_z, u_+, u_-)$` form."""
    return _get_rhs_core(state, fourier_, flow_, None)


def _get_rhs_measured(
    state: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> tuple[Array, dict[str, Array]]:
    """Evaluate the nonlinear RHS + CFL measurements."""

    def _measure(u_phys: Array, omega_phys: Array) -> dict[str, Array]:
        return get_cfl(
            u_phys,
            flow_.base_flow_adv_padded,
            flow_.cfl_inv_spacing,
            CFL_NAMES,
        )

    return _get_rhs_core(state, fourier_, flow_, _measure)


# ── Matrix-free matvecs ──────────────────────────────────────────


def _abase_matvec(u: Array, flow_: AnnularFlow) -> Array:
    r"""Apply `$A_{\mathrm{base}} u = (D_2 + (1/r) D_1)\,u$`."""
    inv_r = flow_.inv_r[:, None, None]
    D2_u = apply_y_matrix(flow_.D2, u)
    D1_u = apply_y_matrix(flow_.D1, u)
    return D2_u + inv_r * D1_u


def _lk_matvec(
    u: Array,
    flow_: AnnularFlow,
    fourier_: Fourier,
) -> Array:
    r"""Apply `$L_k u = A_{\mathrm{base}} u - (m^2/r^2 + k_z^2) u$`.

    Neumann wall rows at both walls; the outer-wall row pins
    `$p_{N_r-1}$` at the mean mode (the only `$k^2 = 0$` system).
    """
    Abase_u = _abase_matvec(u, flow_)
    inv_r2 = flow_.inv_r2[:, None, None]
    out = Abase_u - (fourier_.m2 * inv_r2 + fourier_.kz2) * u

    inner = jnp.einsum("j, jmz -> mz", flow_.D1_bnd[0], u)
    outer_neumann = jnp.einsum("j, jmz -> mz", flow_.D1_bnd[1], u)
    outer = jnp.where(fourier_.mean_mask[0], u[-1], outer_neumann)
    return out.at[0].set(inner).at[-1].set(outer)


# ── IMM iteration (2x2) ─────────────────────────────────────────


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> tuple[Array, Array]:
    r"""Influence-matrix method for the annular geometry.

    Combines the cylindrical `$u_\pm$` divergence / pressure-gradient
    structure with the Cartesian two-wall `$2 \times 2$` influence
    matrix.  Stages (plus mean-mode projections):

    1. **Poisson RHS** from the cylindrical divergence
       `$(D_1 u_+ + (m{+}1)/r\,u_+)/2 + (D_1 u_- + (1{-}m)/r\,u_-)/2
       + ik_z u_z$`.
    2. **Particular pressure** `$L_k p_P = \hat f_P$` (both Neumann wall
       rows zeroed).
    3. **Helmholtz solves** for `$u_{+,-,z}$` against
       `$(\nabla p)_\pm = D_1 p \mp (m/r) p$`, `$(\nabla p)_z = ik_z p$`
       (both Dirichlet wall rows zeroed; mean-mode `$u_r$` removed).
    4. **Wall divergence residual** (2-vector) `$d_{\mathrm{wall}} =
       D_{1,\mathrm{bnd}} \cdot (u_{+,arb} + u_{-,arb})/2$`.
    5. **Influence matrix** `$\boldsymbol\alpha = -M^{-1}
       d_{\mathrm{wall}}$`.
    6. **Correction** `$u_\pm = u_{\pm,arb} + \alpha_1 v_{\pm,1}
       + \alpha_2 v_{\pm,2}$`, `$u_z = u_{z,arb} - ik_z(\alpha_1 q_{z,1}
       + \alpha_2 q_{z,2})$`.
    7. **Zero mean-mode** `$u_r$` (preserve `$u_\theta$`).
    8. *(optional)* If ``block_mean_spanwise_velocity``, zero the
       mean-mode perturbation bulk axial velocity `$u_z$`.
    """
    c = params.step.implicitness
    dt = params.step.dt
    nu = derived_params.nu  # solvent viscosity (see AnnularFlow.__post_init__)

    uz_n, up_n, um_n = velocity_n[0], velocity_n[1], velocity_n[2]
    NLz_n, NLp_n, NLm_n = nonlin_n[0], nonlin_n[1], nonlin_n[2]
    NLz_j, NLp_j, NLm_j = nonlin_j[0], nonlin_j[1], nonlin_j[2]

    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]
    m = fourier_.m
    mean_mask = fourier_.mean_mask

    m_plus_1_sq = (m + 1) ** 2
    m_minus_1_sq = (m - 1) ** 2
    m_sq = fourier_.m2

    # Batch the D1 derivatives for the divergence and the explicit
    # Hk^- matvec (u_z included) into a single GEMM; only the
    # just-solved pP needs a second D1 after the Poisson solve below.
    # Stack y-leading (N_r, 7, ...) so the batched D1 GEMM (shared by
    # the divergence and the Hk^- matvec below) contracts the leading
    # wall-normal axis transpose-free; the component axis is 1.
    all_v = jnp.stack([up_n, um_n, uz_n, NLp_j, NLp_n, NLm_j, NLm_n], axis=1)
    dy_all = apply_y_matrix(flow_.D1, all_v, component_axis=1)

    div_n = (
        (dy_all[:, 0] + (m + 1) * inv_r * up_n) / 2
        + (dy_all[:, 1] + (1 - m) * inv_r * um_n) / 2
        + ikz * uz_n
    )
    div_NLj = (
        (dy_all[:, 3] + (m + 1) * inv_r * NLp_j) / 2
        + (dy_all[:, 5] + (1 - m) * inv_r * NLm_j) / 2
        + ikz * NLz_j
    )
    div_NLn = (
        (dy_all[:, 4] + (m + 1) * inv_r * NLp_n) / 2
        + (dy_all[:, 6] + (1 - m) * inv_r * NLm_n) / 2
        + ikz * NLz_n
    )

    Lk_d = _lk_matvec(div_n, flow_, fourier_)
    f_hat = div_n / dt + c * div_NLj + (1 - c) * div_NLn + (1 - c) * nu * Lk_d

    # Stage 2: particular pressure (both Neumann wall rows zeroed).
    f_hat_P = f_hat.at[0].set(0.0).at[-1].set(0.0)
    pP = flow_.Lk_op.solve(f_hat_P)

    # Stage 3: pressure gradient and explicit Hk^- matvec.  D1 of the
    # velocity (u_+, u_-, u_z) was already formed above as dy_all[:3];
    # only the just-solved pP needs a fresh D1.
    # y-leading (N_r, 3, ...) Hk construction: the D2 GEMM and the
    # reused D1_vel stay transpose-free (component axis 1); the solve
    # takes this layout (component_axis=1) and we unstack.  inv_r/inv_r2
    # get a trailing axis to broadcast over C; kz2/mean_mask are
    # trailing-mode broadcasts (layout-invariant).
    vel_n_stack = jnp.stack([up_n, um_n, uz_n], axis=1)  # (N_r, 3, ...)
    D1_pP = apply_y_matrix(flow_.D1, pP)
    D1_vel = dy_all[:, :3]
    m_over_r = m * inv_r

    grad_pP_plus = D1_pP - m_over_r * pP
    grad_pP_minus = D1_pP + m_over_r * pP
    grad_pP_z = ikz * pP

    inv_r_y = inv_r[..., None]  # (N_r, 1, 1, 1) over the C axis
    D2_all = apply_y_matrix(flow_.D2, vel_n_stack, component_axis=1)
    Abase_stack = D2_all + inv_r_y * D1_vel
    meff2_stack = jnp.stack([m_plus_1_sq, m_minus_1_sq, m_sq], axis=1)
    inv_r2 = flow_.inv_r2[:, None, None, None]  # (N_r, 1, 1, 1)
    lapl_stack = (
        Abase_stack - (meff2_stack * inv_r2 + fourier_.kz2) * vel_n_stack
    )
    Hk_minus_stack = (1.0 / dt) * vel_n_stack + (1.0 - c) * nu * lapl_stack
    # Identity wall rows at both walls.
    Hk_minus_stack = Hk_minus_stack.at[0].set(vel_n_stack[0])
    Hk_minus_stack = Hk_minus_stack.at[-1].set(vel_n_stack[-1])

    R_stack = (
        Hk_minus_stack
        - jnp.stack([grad_pP_plus, grad_pP_minus, grad_pP_z], axis=1)
        + c * jnp.stack([NLp_j, NLm_j, NLz_j], axis=1)
        + (1 - c) * jnp.stack([NLp_n, NLm_n, NLz_n], axis=1)
    )
    # Zero Dirichlet wall rows (both walls).
    R_stack = R_stack.at[0].set(0.0).at[-1].set(0.0)

    # Mean mode: zero the u_r part of the +/- RHS so u_r = 0 there.
    Rr_corr = jnp.where(mean_mask, (R_stack[:, 0] + R_stack[:, 1]) / 2, 0.0)
    R_stack = R_stack.at[:, 0].add(-Rr_corr)
    R_stack = R_stack.at[:, 1].add(-Rr_corr)

    arb_stack = flow_.Hk_op.solve(R_stack, component_axis=1)
    up_arb, um_arb, uz_arb = (
        arb_stack[:, 0],
        arb_stack[:, 1],
        arb_stack[:, 2],
    )

    # Stage 4: wall divergence residual (inner, outer).
    ur_arb = (up_arb + um_arb) / 2
    d_wall = jnp.einsum("bj, jmz -> mzb", flow_.D1_bnd, ur_arb)  # (Nm, Nkz, 2)
    d_wall = jnp.where(mean_mask[0][..., None], 0.0, d_wall)

    # Stage 5: 2x2 influence-matrix algebra.
    alpha = -jnp.einsum("mzab, mzb -> mza", flow_.M_inv, d_wall)
    alpha1 = alpha[..., 0][None]  # (1, Nm, Nkz)
    alpha2 = alpha[..., 1][None]

    # Stage 6: corrected velocity.
    up_new = up_arb + alpha1 * flow_.v_plus_1 + alpha2 * flow_.v_plus_2
    um_new = um_arb + alpha1 * flow_.v_minus_1 + alpha2 * flow_.v_minus_2

    # Stage 7: zero mean-mode u_r, preserving u_theta.
    ur_corr = jnp.where(mean_mask, (up_new + um_new) / 2, 0.0)
    up_new = up_new - ur_corr
    um_new = um_new - ur_corr

    q_corr = alpha1 * flow_.q_z_1 + alpha2 * flow_.q_z_2
    if params.phys.block_mean_spanwise_velocity:
        # Zero the mean-mode perturbation bulk axial velocity.  At the
        # mean mode alpha = 0 and ikz = 0, so uz_arb already equals the
        # uncorrected uz there; reading the bulk from uz_arb fuses the
        # IMM and bulk corrections.
        mean_uz = extract_mean_mode(uz_arb[None])[0].real
        bulk_uz = (
            integrate_scalar(mean_uz, flow_.y_weights)
            / derived_params.volume_fac
        )
        uz_new = (
            uz_arb
            - ikz * q_corr
            + jnp.where(
                mean_mask,
                -bulk_uz
                * flow_.H_bulk_inv
                * flow_.h_bulk_response[:, None, None],
                0.0,
            )
        )
    else:
        uz_new = uz_arb - ikz * q_corr

    velocity_new = jnp.array([uz_new, up_new, um_new])
    correction = velocity_new - velocity_j

    return velocity_new, correction


def _predict(
    velocity_n: Array,
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> Array:
    """Euler predictor via the annular IMM."""
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
    flow_: AnnularFlow,
) -> tuple[Array, Array]:
    """Crank-Nicolson corrector via the annular IMM."""
    return _imm_iteration(
        state_prev, prediction_state, rhs_prev, rhs_next, fourier_, flow_
    )


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> Array:
    """L2 convergence norm with annular weighting."""
    return jnp.sqrt(
        get_norm2_annular(correction, fourier_.k_metric, flow_.y_weights)
    )


# ── Stepper factory ─────────────────────────────────────────────


def build_annular_stepper(
    flow: AnnularFlow,
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
]:
    """Build time-stepping functions for an annular flow.

    Returns ``(predict_and_correct, iterate_correction,
    init_state_bound, predict_and_fully_correct,
    predict_and_fully_correct_measured, step_cnab2,
    step_cnab2_measured)`` with the ``fourier`` and *flow*
    singletons already bound.  ``_l_bf`` (the FFT-free base-flow
    coupling) is passed so the CN/AB2 scheme treats it implicitly.
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
    )
