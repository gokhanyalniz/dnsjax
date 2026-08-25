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
  `$(m \pm 1)/r$` terms), and the radial Jacobian in the norm weights.
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

As in cylindrical, this is the solver's **working** basis -- the
carried state, the RHS, the corrector iterates and every operator
below -- while everything outside the time stepper (snapshots,
diagnostics, probes, initial conditions, the analysis package) works
in the physical triad `$(u_z, u_r, u_\theta)$`, a given state
crossing between the two at most once (``to_pm_basis`` /
``from_pm_basis`` in ``_base.py``).  :func:`_get_rhs_core` and
:func:`_l_bf` convert internally because the real FFT needs
individually Hermitian-symmetric components, which `$u_\pm$` are not;
see the ``cylindrical.py`` module docstring.

Component-order convention (the annular exception)
--------------------------------------------------
The physical triad is ordered `$(u_z, u_r, u_\theta)$` (axial,
radial, azimuthal), inherited unchanged from the pipe so the shared
right-handed `$(\hat e_z, \hat e_r, \hat e_\theta)$` curl, cross
product, and FD operators apply without a sign change.  Because the
annular main flow is *azimuthal*, this does **not** match the
``(streamwise, wall-normal, spanwise)`` component order that
triply-periodic, Cartesian, and the pipe follow: the streamwise
(azimuthal) velocity is component **2** and the spanwise (axial) one
is component **0**.  Reordering to `$(u_\theta, u_r, u_z)$` was
rejected -- that basis is left-handed, so it could not reuse the
shared right-handed machinery.

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

- **Shear-driven** (Taylor-Couette / quasi-Keplerian -- the same
  circular-Couette flow under two parameterizations): the rotating
  walls set an azimuthal base flow `$U_\theta(r) = A_0 r + B_0/r$` and
  the perturbation `$\mathbf{u}'$` is integrated.  The base coupling enters
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

The viscoelastic sPTT extension (``annular_viscoelastic.py``, bound by
``flows.wall_bounded.viscoelastic_dean``) is a third user of the same
infrastructure: it reuses the force-driven total-field mode with the
9-component velocity + conformation state.

An optional ``block_mean_spanwise_velocity`` zeroes the mean **axial**
velocity (the undriven homogeneous direction); the azimuthal mean
evolves freely.

Flow-specific modules (e.g. ``flows.wall_bounded.taylor_couette``,
``flows.wall_bounded.dean``) subclass ``AnnularFlow``, set the base flow
and/or ``pi_theta``, then call ``build_annular_stepper`` to obtain
ready-to-use time-stepping functions.
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
    build_diff_matrices,
    build_integration_weights,
    clenshaw_curtis_weights,
    local_grid_spacing,
    matrix_half_bandwidth,
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
    _build_pallas_operator,
    _factor_pallas_operator,
)
from ._base import (
    apply_y_matrix,
    base_flow_coupling,
    build_wall_bounded_stepper,
    extract_mean_mode,
    extract_mean_modes,
    from_pm_basis,
    frozen_profile_flow,  # noqa: F401 — re-exported
    get_inprod,  # noqa: F401 — re-exported
    get_norm,  # noqa: F401 — re-exported
    get_norm2,
    integrate_scalar,
    pad_base_flow,  # noqa: F401 — re-exported
    phys_to_spec,  # noqa: F401 — re-exported
    spec_to_phys,  # noqa: F401 — re-exported
    to_pm_basis,
)

#: Role aliases for the basis boundary (see ``cylindrical.py``).
to_solver_basis = to_pm_basis
from_solver_basis = from_pm_basis


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


# ── Annular-specific norms ──────────────────────────────────────


def get_norm2_annular(
    state: Array, k_metric: Array, y_weights: Array
) -> Array:
    r"""Annular squared L2 norm for `$(u_z, u_r, u_\theta)$`.

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
    radial-derivative, azimuthal, and axial terms; the azimuthal term
    is the covariant azimuthal gradient in `$(u_z, u_r, u_\theta)$`
    components,

    .. math::
        \frac{|im\,u_z|^2 + |im\,u_r - u_\theta|^2
        + |im\,u_\theta + u_r|^2}{r^2},

    pointwise equal to the `$m_{\mathrm{eff}}$`-diagonal form of the
    solver-interior decoupled basis (`$|m u_z|^2 +
    \tfrac{1}{2}|(m{+}1)u_+|^2 + \tfrac{1}{2}|(m{-}1)u_-|^2$`).
    Unlike the cylindrical version the radial derivative uses a
    single `$D_1$` (no parity / ghost correction).

    Parameters
    ----------
    state:
        Spectral velocity in `$(u_z, u_r, u_\theta)$` form,
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
    :mod:`dnsjax.ic.random_field` (the total-field IC = laminar profile +
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


def _vw_recovery_parts(
    m2: Array,
    inv_r: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
) -> tuple[Array, Array]:
    r"""Per-mode pieces of the `$u_r$` recovery operator (vw scheme).

    The recovery realises the `$\Phi$` definition with the
    reconstruction's `$u_\theta(u_r, \omega_r)$` substituted in, so it
    is exact per pass (no iterated operand):

    .. math::
        L_{v,\mathrm{mod}} = A_{\mathrm{base}}
        - \Bigl(\frac{m^2+1}{r^2} + k_z^2\Bigr) I
        + \frac{2 m^2}{r^3\,\Delta}\,\Bigl(D_1 + \frac{1}{r}\Bigr),
        \qquad \Delta = k_z^2 + \frac{m^2}{r^2}.

    Returns the diagonal shift ``(Nm, Nkz, Nr)`` and the per-mode
    coefficient of the `$(D_1 + 1/r)$` correction (odd in `$r$`, so
    the composed operator stays in the `$u_r$` parity class; zero at
    `$m = 0$` and masked at the mean, where `$\Delta = 0$`).
    """
    diag = -((m2 + 1.0) * inv_r2 + kz2)
    det = kz2 + m2 * inv_r2
    det_safe = jnp.where(mean_mask, 1.0, det)
    coeff = jnp.where(mean_mask, 0.0, 2.0 * m2 * inv_r2 * inv_r / det_safe)
    return diag, coeff


def _build_Lv_dir_band_gpu(
    D1: Array,
    A_base: Array,
    m2: Array,
    inv_r: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
    p: int,
) -> Array:
    r"""Build the vw `$u_r$` recovery operator in banded storage.

    `$L_{v,\mathrm{mod}}$` of :func:`_vw_recovery_parts` with Dirichlet
    identity rows at **both** walls (`$u_r|_{\mathrm{wall}} = 0$`).
    ``dt``-free, like the legacy Neumann `$L_k$` it replaces; no mean
    pin is needed -- `$m_{\mathrm{eff}}^2 = m^2 + 1 \ge 1$` keeps the
    operator regular at every mode including `$k^2 = 0$`.
    """
    Nr = A_base.shape[0]
    diag, coeff = _vw_recovery_parts(m2, inv_r, inv_r2, kz2, mean_mask)
    band_base = _banded_from_dense(A_base, p)  # (Nr, 2p+1)
    corr = D1 + inv_r[:, None] * jnp.eye(Nr, dtype=A_base.dtype)
    band_corr = _banded_from_dense(corr, p)
    band = band_base + coeff[..., None] * band_corr
    eN = _banded_diag_column(p, band_base.dtype)
    return _assemble_banded_operator(band, 1.0, diag, [(0, eN), (Nr - 1, eN)])


def _build_Lv_dir_dense_gpu(
    D1: Array,
    A_base: Array,
    m2: Array,
    inv_r: Array,
    inv_r2: Array,
    kz2: Array,
    mean_mask: Array,
) -> Array:
    r"""Dense twin of :func:`_build_Lv_dir_band_gpu`."""
    Nr = A_base.shape[0]
    dtype = A_base.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)
    diag, coeff = _vw_recovery_parts(m2, inv_r, inv_r2, kz2, mean_mask)
    corr = D1 + inv_r[:, None] * eye_Nr
    Lv = (
        A_base[None, None]
        + diag[..., None] * eye_Nr
        + coeff[..., None] * corr[None, None]
    )
    Lv = Lv.at[..., 0, :].set(eye_Nr[0, :])
    Lv = Lv.at[..., -1, :].set(eye_Nr[-1, :])
    return Lv


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

    The velocity state is carried through the solver in decoupled form
    `$(u_z, u_+, u_-)$` with `$u_\pm = u_r \pm i\,u_\theta$`, and in
    the physical triad everywhere outside it (the module docstring;
    ``to_pm_basis``/``from_pm_basis``).  Three Helmholtz operators are
    built (`$m_{\mathrm{eff}} = m+1, m-1, m$` for `$u_+, u_-, u_z$`)
    and one pressure Poisson operator (`$m_{\mathrm{eff}} = m$`).

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
        Factored per-mode operators.  Flag-off: the Neumann pressure
        Poisson `$L_k$` and the stacked `$(+,-,z)$` Helmholtz group.
        Under ``res.consistent_imm``: the Dirichlet `$u_r$` recovery
        `$L_{v,\mathrm{mod}}$` (:func:`_build_Lv_dir_band_gpu`,
        ``dt``-free) and the 2-slot `$(\Phi, \omega_r)$` Helmholtz
        pair (:func:`_hk_vw_bands`).
    v_plus_i, v_minus_i, q_z_i:
        Homogeneous `$u_\pm$` velocity and axial potential responses to
        a unit pressure at wall ``i`` (``1`` = inner, ``2`` = outer).
        ``None`` under ``res.consistent_imm`` (no pressure).
    ur_1, ur_2:
        The vw scheme's homogeneous `$u_r$` responses to a unit
        `$\Phi$` wall value (``None`` on the legacy path).
    M_inv:
        Inverse `$2 \times 2$` influence matrix per mode.
    h_bulk_response, H_bulk_inv:
        Axial-bulk-blocking response (zero unless
        ``block_mean_spanwise_velocity``).
    dt, ab2_kappa:
        Live time step and AB2 step ratio, 0-d array leaves (see
        ``CartesianFlow`` and the builder ``set_dt``).
    """

    dt: Array = field(init=False)
    ab2_kappa: Array = field(init=False)
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
    v_plus_1: Array | None = field(init=False)
    v_minus_1: Array | None = field(init=False)
    q_z_1: Array | None = field(init=False)
    v_plus_2: Array | None = field(init=False)
    v_minus_2: Array | None = field(init=False)
    q_z_2: Array | None = field(init=False)
    ur_1: Array | None = field(init=False)
    ur_2: Array | None = field(init=False)
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

        # Banded half-width: measured, not assumed (see the Cartesian
        # ``__post_init__`` note).  Both wall rows are overwritten with
        # BC rows, so their own stencil width need not fit.
        fd_p = matrix_half_bandwidth(np.asarray(self.A_base), (0, -1))
        dt = params.step.dt

        # Live-dt pytree leaves (class docstring; rebuilt by the
        # builder's ``set_dt`` with identical dtype/shape).
        self.dt = jnp.asarray(dt, dtype=sharding.float_type)
        self.ab2_kappa = jnp.ones((), dtype=sharding.float_type)

        # Solver-internal wavenumber arrays.
        m_s = fourier.m[0, ..., None]  # (Nm, 1, 1)
        kz2_s = fourier.kz2[0, ..., None]  # (1, Nkz, 1)
        mean_s = fourier.mean_mask[0, ..., None]  # (Nm, Nkz, 1)
        m_sq = m_s**2

        if params.solver.backend == "pallas":
            # Pallas backend: one-program-per-mode banded sweep.
            # Operators are assembled directly in banded storage (no
            # (Nr, Nr) per mode) and factored by the setup-checked
            # no-pivot banded LU (_build_pallas_operator).
            if params.res.consistent_imm:
                # vw scheme: the dt-free Dirichlet u_r recovery
                # operator lives in the Lk_op slot (there is no
                # pressure), preserving the _hk_bands band readback.
                Lk_band = _build_Lv_dir_band_gpu(
                    self.D1,
                    self.A_base,
                    m_sq,
                    self.inv_r,
                    self.inv_r2,
                    kz2_s,
                    mean_s,
                    fd_p,
                )
                self.Lk_op = _build_pallas_operator([Lk_band], "Lv_dir")
            else:
                from . import _annular_primitive_imm as prim

                Lk_band = prim._build_Lk_band_gpu(
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

            # Hk group -- the default 2-slot (Phi, omega_r) pair, or
            # the legacy (plus, minus, z) triple: stacked into one
            # homogeneous operator and stability-checked as a group.
            if params.res.consistent_imm:
                hk_bands_fn = _hk_vw_bands
            else:
                from . import _annular_primitive_imm as prim

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
            if params.res.consistent_imm:
                Lk_dense = _build_Lv_dir_dense_gpu(
                    self.D1,
                    self.A_base,
                    m_sq,
                    self.inv_r,
                    self.inv_r2,
                    kz2_s,
                    mean_s,
                )
            else:
                from . import _annular_primitive_imm as prim

                Lk_dense = prim._build_Lk_dense_gpu(
                    self.D1, self.A_base, m_sq, self.inv_r2, kz2_s, mean_s
                )
            self.Lk_op = DenseJAXSolver(Lk_dense)
            del Lk_dense

            # Combined Hk: the default (Phi, omega_r) pair, or the
            # legacy (plus, minus, z) triple.
            if params.res.consistent_imm:
                self.Hk_op = _hk_vw_dense_op(dt, fourier, self)
            else:
                from . import _annular_primitive_imm as prim

                self.Hk_op = prim._hk_dense_op(dt, fourier, self)

        self._derive_imm_homogeneous_data(fourier, Nm, Nkz, Nr)
        self._precompute_bulk_response(fourier, Nm, Nkz, Nr)

    def _derive_imm_homogeneous_data(
        self, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
    ) -> None:
        r"""Fill the homogeneous responses and the `$2 \times 2$`
        ``M_inv`` on-device: dispatch on ``res.consistent_imm``.

        Both schemes carry the same `$2 \times 2$` capacitance
        structure; only the chain the columns solve differs.

        - **default** -- :meth:`_derive_vw_homogeneous_data`: the
          `$u_r$` responses of the `$u_r$`-`$\omega_r$` scheme, with no
          pressure to carry (``v_plus_*``/``v_minus_*``/``q_z_*`` stay
          ``None``).
        - **legacy** (flag off) --
          :func:`._annular_primitive_imm.derive_homogeneous_data`: the
          `$u_\pm$` responses to two unit-wall pressures, plus the
          axial potentials ``q_z_i`` (``ur_1``/``ur_2`` stay ``None``).

        Both fill ``M_inv`` and are re-run at a changed ``dt`` by
        :func:`_build_dt_leaves`.
        """
        if params.res.consistent_imm:
            self._derive_vw_homogeneous_data(fourier_, Nm, Nkz, Nr)
            return

        from . import _annular_primitive_imm as prim

        prim.derive_homogeneous_data(self, fourier_, Nm, Nkz, Nr)

    def _derive_vw_homogeneous_data(
        self, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
    ) -> None:
        r"""Homogeneous data of the `$u_r$`-`$\omega_r$` scheme
        (``res.consistent_imm``).

        Mirrors the Cartesian ``_derive_vw_homogeneous_data``: a unit
        `$\Phi$` wall value at wall ``b`` gives
        `$\Phi_b = H_k^{-1} e_b$` (identity wall rows, so
        `$\Phi_b|_{\mathrm{wall}_b} = 1$`), the recovery
        `$u_{r,b} = L_{v,\mathrm{mod}}^{-1} (\Phi_b)_P$` (wall rows
        zeroed -- `$u_r|_{\mathrm{wall}} = 0$` exactly), and the
        `$2 \times 2$` influence matrix
        `$M_{jb} = D_{1,\mathrm{wall}_j} \cdot u_{r,b}$` whose
        `$\alpha = -M^{-1} d_{\mathrm{wall}}$` imposes
        `$(D_1 u_r)|_{\mathrm{wall}} = 0$`.  With
        `$\omega_r|_{\mathrm{wall}} = 0$` (Dirichlet) the per-point
        reconstruction then makes tangential no-slip *emerge*.  The
        `$\omega$` slot needs no columns.  Columns and ``M_inv`` are
        zeroed at the mean mode (packed planes; no influence there).

        The two wall columns are batched through the 2-slot ``Hk_op``
        as one stack: its per-slot bands differ only on the mean
        plane, and every mean-plane value is zeroed below, so the
        second column riding the `$\omega$` band is harmless.
        """
        e_cols = []
        for row in (0, Nr - 1):
            e_cols.append(
                jnp.zeros(
                    (Nm, Nkz, Nr),
                    dtype=sharding.float_type,
                    out_sharding=sharding.spec_imm_corr_shard,
                )
                .at[..., row]
                .set(1.0)
            )
        stacked = jnp.stack(e_cols)  # (2, Nm, Nkz, Nr)
        phi_b = self.Hk_op.solve(stacked.transpose(0, 3, 1, 2)).transpose(
            0, 2, 3, 1
        )
        phi_b = phi_b.at[..., 0].set(0.0).at[..., -1].set(0.0)
        ur_b = self.Lk_op.solve(phi_b.transpose(0, 3, 1, 2)).transpose(
            0, 2, 3, 1
        )

        is_mean = fourier_.mean_mask[0]  # (Nm, Nkz)
        ur_b = jnp.where(is_mean[..., None], 0.0, ur_b)
        ur1, ur2 = ur_b[0], ur_b[1]

        M00 = jnp.einsum("j, mzj -> mz", self.D1_bnd[0], ur1)
        M01 = jnp.einsum("j, mzj -> mz", self.D1_bnd[0], ur2)
        M10 = jnp.einsum("j, mzj -> mz", self.D1_bnd[1], ur1)
        M11 = jnp.einsum("j, mzj -> mz", self.D1_bnd[1], ur2)
        det = M00 * M11 - M01 * M10
        safe_det = jnp.where(is_mean, 1.0, det)
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

        # Field layout (Nr, Nm, Nkz); the pressure-scheme columns are
        # static aux-data by default.
        self.ur_1 = ur1.transpose(2, 0, 1)
        self.ur_2 = ur2.transpose(2, 0, 1)
        self.v_plus_1 = self.v_minus_1 = self.q_z_1 = None
        self.v_plus_2 = self.v_minus_2 = self.q_z_2 = None

    def _precompute_bulk_response(
        self, fourier_: Fourier, Nm: int, Nkz: int, Nr: int
    ) -> None:
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
        rhs = jnp.where(fourier_.mean_mask[0, ..., None], ones_vec, 0.0)
        zeros = jnp.zeros_like(rhs)
        # The mean-mode axial Helmholtz: by default it is the mean
        # plane of the Phi slot; on the legacy path the z slot of the
        # (+, -, z) group
        # IS the same operator (m_eff^2 = 0 there by the packing).
        if params.res.consistent_imm:
            stack, comp = [rhs, zeros], 0
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
        H_bulk = (
            jnp.dot(self.y_weights, self.h_bulk_response)
            / derived_params.volume_fac
        )
        self.H_bulk_inv = 1.0 / H_bulk


def _vw_meff2(fourier_: Fourier) -> tuple[Array, Array]:
    r"""Per-slot `$m_{\mathrm{eff}}^2$` of the `$(\Phi, \omega_r)$`
    pair, with the mean-plane packing exception.

    Both slots share the spin-diagonal `$m^2 + 1$`.  On the packed
    `$k^2 = 0$` plane the `$\Phi$` slot carries `$u_{z,00}$` and needs
    `$m_{\mathrm{eff}}^2 = 0$` (the mean axial Helmholtz), while the
    `$\omega$` slot carries `$u_{\theta,00}$`, whose operator is
    `$m_{\mathrm{eff}}^2 = 1$` -- exactly `$m^2 + 1$` at `$m = 0$`, so
    only `$\Phi$` needs the exception.
    """
    m_s = fourier_.m[0, ..., None]
    mean_s = fourier_.mean_mask[0, ..., None]
    pair = m_s**2 + 1.0
    return jnp.where(mean_s, 0.0, pair), pair


def _hk_vw_bands(
    dt: float | Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> list[Array]:
    r"""Assemble the banded `$(\Phi, \omega_r)$` Helmholtz pair at
    *dt* (``res.consistent_imm``; Pallas backend).

    Same shape contract as :func:`_hk_bands` (a per-slot stacked
    group), with two slots sharing the spin-diagonal
    `$m_{\mathrm{eff}}^2 = m^2 + 1$` (:func:`_vw_meff2`; the slots
    differ only on the packed mean plane).  Dirichlet identity rows at
    both walls: `$\omega_r|_{\mathrm{wall}} = 0$` is physical for
    every annular flow (perturbation form or stationary walls), and
    `$\Phi$`'s zero wall rows are the arbitrary particular choice the
    influence matrix corrects.
    """
    kz2_s = fourier_.kz2[0, ..., None]
    return [
        _build_Hk_band_gpu(
            flow_.A_base,
            meff2,
            flow_.inv_r2,
            kz2_s,
            dt,
            params.step.implicitness,
            derived_params.nu,
            flow_.Lk_op.L.shape[1],
        )
        for meff2 in _vw_meff2(fourier_)
    ]


def _hk_vw_dense_op(
    dt: float | Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> DenseJAXSolver:
    r"""Factored dense `$(\Phi, \omega_r)$` pair at *dt* (dense
    backend)."""
    kz2_s = fourier_.kz2[0, ..., None]
    ops = [
        DenseJAXSolver(
            _build_Hk_dense_gpu(
                flow_.A_base,
                meff2,
                flow_.inv_r2,
                kz2_s,
                dt,
                params.step.implicitness,
                derived_params.nu,
            )
        )
        for meff2 in _vw_meff2(fourier_)
    ]
    return DenseJAXSolver.from_factors(
        lu=jnp.stack([o.lu for o in ops]),
        perm=jnp.stack([o.perm for o in ops]),
    )


def _build_dt_leaves(
    dt: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
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
        from . import _annular_primitive_imm as prim

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
        # The vw scheme's u_r columns; the pressure-scheme columns are
        # None (static aux-data) and Lk_op (= the dt-free recovery) is
        # deliberately absent -- see test_adaptive's leaf dicts.
        leaves |= {"ur_1": new.ur_1, "ur_2": new.ur_2}
    else:
        leaves |= {
            "v_plus_1": new.v_plus_1,
            "v_minus_1": new.v_minus_1,
            "q_z_1": new.q_z_1,
            "v_plus_2": new.v_plus_2,
            "v_minus_2": new.v_minus_2,
            "q_z_2": new.q_z_2,
        }
    return leaves


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
    state_rthz = from_pm_basis(state)
    omega = _curl_fn(state_rthz, fourier_, flow_)
    base = flow_.base_flow
    curl_base = flow_.curl_base_flow
    if params.step.implicit_mean_coupling:
        # One collective for the pair: this runs once per corrector
        # iteration under cnab2 / the split corrector, and the psum is
        # latency-bound (:func:`extract_mean_modes`).
        mean_u, mean_om = extract_mean_modes(state_rthz, omega)
        base = base + mean_u[:, :, None, None]
        curl_base = curl_base + mean_om[:, :, None, None]
    l_bf = to_pm_basis(base_flow_coupling(state_rthz, omega, base, curl_base))
    # Moving frame: the convective frame term (the same expression
    # ``_get_rhs_core`` adds, diagonal in the solver basis) belongs
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

    1. Convert `$(u_z, u_+, u_-) \to (u_z, u_r, u_\theta)$` -- the real
       FFTs need individually Hermitian-symmetric components, which
       `$u_\pm$` are not (see the ``cylindrical.py`` docstring).
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
    nonlin = get_nonlin(
        from_pm_basis(state),
        flow_.base_flow_padded,
        flow_.curl_base_flow_padded,
        spec_to_phys_2d,
        phys_to_spec_2d,
        lambda s: _curl_fn(s, fourier_, flow_),
        measure_fn,
    )
    if measure_fn is not None:
        nonlin, measurements = nonlin

    # Mean-mode azimuthal body force (Dean flow), applied only at the
    # mean mode (m, k_z) = (0, 0); zero for shear-driven Taylor-Couette.
    # get_nonlin returns the +u x omega rotational term, so the force
    # enters the RHS with a + sign (and into NL_+/- via NL_theta).
    rhs = to_pm_basis(
        nonlin.at[2].add(
            jnp.where(fourier_.mean_mask, flow_.pi_theta[:, None, None], 0.0)
        )
    )
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
            flow_.dt,
        )

    return _get_rhs_core(state, fourier_, flow_, _measure)


# ── IMM iteration (2x2) ─────────────────────────────────────────


#: ``stats.dat`` column name for the mean-mode driving this geometry
#: applies (:func:`_apply_bulk_correction`).  The blocked direction is
#: **axial** here, not spanwise-in-the-Cartesian-sense, and the sign is
#: the applied **forcing** `$-\partial p'/\partial z$`, positive when it
#: accelerates the flow.
DRIVING_KEY_Z = "-dPdz'"


def mean_driving(state: Array, flow_: AnnularFlow) -> dict[str, Array]:
    r"""Wall-shear **inference** of the axial driving, from a state.

    Area-averaging the mean-mode axial momentum over the annulus, with
    `$\int_{r_1}^{r_2} r^{-1}(r\,\bar{u}_z')'\,r\,dr =
    [r\,\bar{u}_z']_{r_1}^{r_2}$` and ``volume_fac``
    `$= (r_2^2 - r_1^2)/2$`:

    .. math::
        \frac{d U_{b,z}'}{dt} = \Pi'_z
        + \nu\,\frac{r_2\tau_2 - r_1\tau_1}{\mathrm{volfac}},

    so blocking the bulk applies exactly the negative of that flux
    term, under the same key and sign as
    :func:`_apply_bulk_correction`.

    **Newtonian only.**  It uses the solvent viscosity alone, so it
    does not close the balance for the viscoelastic annular flow, whose
    polymer stress carries its own axial wall traction -- that flow
    therefore does not export it (its ``t = t0`` row is written as
    zero instead of a wrong number; see :mod:`dnsjax.__main__`).
    """
    if not params.phys.block_mean_spanwise_velocity:
        return {}
    mean_uz = extract_mean_mode(state)[0].real
    tau = flow_.D1_bnd @ mean_uz  # (2,) = [inner, outer]
    r1, r2 = flow_.rs[0], flow_.rs[-1]
    flux = (r2 * tau[1] - r1 * tau[0]) / derived_params.volume_fac
    return {DRIVING_KEY_Z: -flux / params.phys.re}


def _apply_bulk_correction(
    uz_new: Array,
    uz_src: Array,
    mean_mask: Array,
    flow_: AnnularFlow,
) -> tuple[Array, dict[str, Array]]:
    r"""Axial bulk blocking, shared by both IMM schemes.

    Under ``phys.block_mean_spanwise_velocity`` (the annulus' undriven
    homogeneous direction is the axial one) adds a uniform body force
    `$\Pi'_z$` to the mean-mode `$u_z$` Helmholtz RHS so the
    perturbation bulk axial velocity is zero, in the equivalent
    post-solve form `$u_z \mathrel{+}= \Pi'_z\,h$` with `$h$` from
    :meth:`AnnularFlow._precompute_bulk_response`.  Like every
    mean-plane write it is confined to `$k^2 = 0$`, the one plane the
    reconstruction never touches.

    *uz_src* is where the bulk is **read**, *uz_new* what the correction
    is **added to**; they differ only on the legacy primitive path,
    whose `$u_z$` carries an extra `$-ik_z q$` term that vanishes at the
    mean mode, so reading from the uncorrected ``uz_arb`` fuses the two
    corrections.

    Returns the corrected field and the applied `$\Pi'_z$` as the
    corrector's *aux* diagnostics -- the correction's own prefactor, so
    the reported number is by construction the applied one.  Empty, and
    a trace-time no-op, when the knob is off (the default).
    """
    if not params.phys.block_mean_spanwise_velocity:
        return uz_new, {}
    mean_uz = extract_mean_mode(uz_src[None])[0].real
    bulk_uz = (
        integrate_scalar(mean_uz, flow_.y_weights) / derived_params.volume_fac
    )
    pi_z = -bulk_uz * flow_.H_bulk_inv  # the applied body force
    return (
        uz_new
        + jnp.where(
            mean_mask, pi_z * flow_.h_bulk_response[:, None, None], 0.0
        ),
        {DRIVING_KEY_Z: pi_z},
    )


def _imm_iteration_vw(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> tuple[Array, Array, dict[str, Array]]:
    r"""`$u_r$`-`$\omega_r$` step (``res.consistent_imm``).

    The cylindrical transcription of the Cartesian `$v$`-`$\omega_y$`
    scheme (``cartesian._imm_iteration_vw``, which carries the shared
    derivation): advance the wall-normal
    velocity and the wall-normal vorticity, *reconstruct* the
    tangential pair from them, and never form a pressure.  Per mode
    `$(m, k_z)$`, with `$A_{\mathrm{base}} = D_2 + (1/r) D_1$` and

    .. math::
        L_v = A_{\mathrm{base}} - \frac{m^2+1}{r^2} - k_z^2 ,

    the evolved scalars are the discrete radial vector-Laplacian
    component and the radial vorticity,

    .. math::
        \Phi = (\Delta \mathbf{u})_r
             = L_v u_r - \frac{2im}{r^2} u_\theta , \qquad
        \omega_r = \frac{im}{r} u_z - i k_z u_\theta ,

    obeying the curl and double-curl of the momentum equation,

    .. math::
        \partial_t \Phi = \nu\Bigl[L_v \Phi
            - \tfrac{2im}{r^2} (\Delta\mathbf{u})_\theta\Bigr]
            + S_\Phi , \qquad
        \partial_t \omega_r = \nu\Bigl[L_v \omega_r
            - \tfrac{2im}{r^2}\omega_\theta\Bigr] + S_\omega .

    `$m^2+1$` is the diagonal of the spin block
    `$\bigl[\begin{smallmatrix} m^2+1 & -2im \\ 2im & m^2+1
    \end{smallmatrix}\bigr]/r^2$` whose eigenvalues are the primitive
    scheme's own `$m_{\mathrm{eff}}^2 = (m \pm 1)^2$`, so **both slots
    share one implicit operator** and go through a single two-slot
    banded solve.

    Why the discrete divergence vanishes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Per mode the code's own divergence (the `$u_\pm$` expression of
    :func:`_imm_iteration_vp`, mirrored by
    ``analysis.snapshot_ops.divergence``) is
    `$(D_1 + 1/r) u_r + (im/r) u_\theta + i k_z u_z$`.  Stage 7 solves
    the `$2 \times 2$` per-point system

    .. math::
        i k_z u_z + \frac{im}{r} u_\theta = -\Bigl(D_1
            + \frac{1}{r}\Bigr) u_r , \qquad
        \frac{im}{r} u_z - i k_z u_\theta = \omega_r

    of determinant `$k_z^2 + m^2/r^2$` -- nonzero at every mode but
    the single `$k^2 = 0$` plane -- so continuity holds **as algebra**
    at every row, walls included, for any `$D_1$`, any `$D_2$`, any
    grid or axis fit.  Nothing is written after the reconstruction
    except the `$k^2 = 0$` plane it never touches.

    Discrete pressure elimination: the conservative curl
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    `$S_\omega = C_r(N)$` and `$S_\Phi = -[\nabla\times C(N)]_r$` with
    the discrete curl

    .. math::
        C_r = \frac{im}{r} N_z - i k_z N_\theta , \quad
        C_\theta = i k_z N_r - D_1 N_z , \quad
        C_z = \frac{1}{r}\bigl[D_1 (r N_\theta) - i m N_r\bigr] .

    The **conservative** `$C_z$` is mandatory: only in that form does
    the curl annihilate a discrete gradient
    `$(ik_z q, D_1 q, (im/r) q)$` exactly (`$\le 2\times10^{-15}$`
    measured on arbitrary dense `$D_1$` and random `$r$`), because
    `$r \cdot (1/r) = 1$` and `$1/r^2 = (1/r)^2$` are diagonal
    identities and the metric commutator `$[D_1, 1/r]$` never appears;
    the direct form `$C_z = (D_1 + 1/r) N_\theta - (im/r) N_r$` fails
    at `$O(10)$`.  This is the cylindrical counterpart of the
    Cartesian scalar `$k^2$` commuting with `$D_1$`, and it is what
    makes the sources exactly pressure-free.  It is a private choice
    *inside the sources*: the divergence operator the solver and the
    analysis package share is unchanged.

    Exact recovery, and the one lagged coupling
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Stage 5 inverts the `$\Phi$` definition for `$u_r$`.  Naively that
    needs `$u_\theta$`, which is not yet known -- but the stage-7
    reconstruction *is* `$u_\theta(u_r, \omega_r)$`, and substituting
    it folds the coupling into the operator:

    .. math::
        L_{v,\mathrm{mod}} u_r = \Phi
            - \frac{2 m k_z}{r^2 \Delta}\,\omega_r , \qquad
        L_{v,\mathrm{mod}} = L_v + \frac{2m^2}{r^3 \Delta}
            \Bigl(D_1 + \frac{1}{r}\Bigr),

    `$\Delta = k_z^2 + m^2/r^2$` (:func:`_vw_recovery_parts`).  The
    correction is a real, `$dt$`-free, band-preserving addition, so the
    recovery costs exactly what a plain `$L_v$` solve would and is
    **exact per pass** -- no iterated operand, hence no `$O(\Delta t)$`
    bias bolted onto a definitional identity.

    What *is* lagged is the pair's own spin partners
    `$(\Delta\mathbf{u})_\theta$` and `$\omega_\theta$`, evaluated at
    the running corrector iterate (Crank-Nicolson-combined, so the
    converged fixed point is the fully coupled scheme).  They vanish
    identically at `$m = 0$`, cost no memory, and contract at measured
    `$\rho \le 0.02$` over the production corner -- against `$0.25$`
    with the naive recovery, which is why the exact one is not an
    optimization but the reason the loop is comfortable.  (The pipe's
    axis makes the same lag diverge; ``cylindrical.py`` evolves the
    spin quad instead.)

    **That `$0.02$` is a measurement over a corner, not a bound.**  The
    lagged partners enter multiplied by `$2m/r^2$` (``spin`` below)
    against a damping `$1/(c\nu\Delta t) + k^2$`, so the contraction
    degrades toward small inner radius, large `$\nu$`, large
    `$\Delta t$` and high `$m$` -- `$r \to 0$` is exactly what makes it
    diverge on the pipe.  Measured on ``taylor-couette``
    (`$\mathrm{re}_1 = 1$`, `$\mathrm{re}_2 = -0.5$`,
    `$n_r = n_\theta = 64$`, `$\Delta t = 0.01$`, corrector tolerance
    `$10^{-12}$` so the count is not floored), corrector passes per
    step against `$\eta$`:

    ===========  ==========  ==========
    `$\eta$`     default     legacy
    ===========  ==========  ==========
    0.5          5           3
    0.3          7           3
    0.1          10          3
    ===========  ==========  ==========

    -- so the legacy primitive scheme is `$\eta$`-independent while
    this one reaches ``max_corrector_iterations`` by `$\eta = 0.1$`.
    At the default tolerance `$10^{-5}$` every one of those converges
    in 0-1 passes, so no shipped configuration is affected.  This is
    **the one corner where selecting ``res.consistent_imm = False`` is
    a reasonable answer** rather than a compatibility choice (the
    ``Resolution.consistent_imm`` docs point here for exactly that).
    The important structural point is that the lag sits **inside** the
    corrector: degradation surfaces as an iteration count and
    ultimately as a *reported* non-convergence, never as the silent
    across-step growth the pipe's rejected wall-data lag produced.  A
    wide-gap run at low `$\mathrm{Re}$` should watch
    ``corrector.dat``.

    Retired route: decoupling the pair
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The `$u_\pm$` trick diagonalizes the spin block only *within* one
    vector field's transverse pair.  This pair mixes two fields --
    `$u_r$` (velocity, whose fourth-order pressure-eliminated dynamics
    is exactly what makes the scheme stable, see the Cartesian
    dispatcher's route 4) and `$\omega_r$` -- whose diffusion couples
    to `$u_\theta$` and `$\omega_\theta$`, outside every linear
    combination of the pair; closing it through the solenoidal
    constraint reimports composed `$D_1 \mathrm{diag} D_1$` operators,
    i.e. the band and truncation regression this rewrite removed.  The
    exactly-decoupled alternatives were evaluated and are all worse:
    `$(\omega_+, \omega_-)$` degenerates at `$k_z = 0$` to functions of
    `$u_z$` alone (leaving `$u_r$` undetermined on the streak plane),
    abandons the Kim-Moin-Moser structure, and needs a wide composed
    `$u_z$` BVP plus a `$4 \times 4$` influence matrix;
    `$((\Delta u)_+, (\Delta u)_-)$` has eight chain boundary
    conditions against six physical ones; mixed chiral pairs
    `$(\Phi_+, \omega_-)$` are chirality-asymmetric, so Hermitian mode
    pairs would evolve under different schemes and real fields would
    stop being real.  Ledger: this scheme runs **three** band families
    at half-width ``fd_order`` (the pair shares one, the recovery is
    ``dt``-free) against the primitive scheme's four and the retired
    composed-`$D_2$` route's four at half-width ~``fd_order + 4``.

    Boundary conditions
    ~~~~~~~~~~~~~~~~~~~
    `$\omega_r|_{\mathrm{wall}} = 0$` (physical for every annular flow
    -- perturbation form for Taylor-Couette / quasi-Keplerian,
    stationary walls for Dean and the viscoelastic mode) and
    `$u_r|_{\mathrm{wall}} = 0$` are identity wall rows of `$H_k$` and
    `$L_{v,\mathrm{mod}}$`; `$(D_1 u_r)|_{\mathrm{wall}} = 0$` is
    imposed by the `$2 \times 2$` influence matrix over the two free
    `$\Phi$` wall values (:meth:`AnnularFlow._derive_vw_homogeneous_data`).
    Tangential no-slip is then not imposed but *emergent*: at a wall
    `$(D_1 + 1/r) u_r = 0$` and `$\omega_r = 0$` make the
    reconstruction return `$u_z = u_\theta = 0$`, so their wall values
    are a live diagnostic of influence-matrix health.

    `$\Phi$` is **not** carried between steps: the state stays
    `$(u_z, u_+, u_-)$` -- the basis the RHS, the probes, the forcing
    and the viscoelastic extension all share -- and stage 2 recomputes
    `$\Phi^n$` from it on full rows.  The wall rows of that
    recomputation are therefore the discrete
    `$(\Delta\mathbf{u})_r$` rather than the influence matrix's own
    `$\alpha$` from the previous step, and they reach the answer only
    through the explicit half's near-wall `$D_2$` stencil (the solve's
    own wall rows are overwritten in stage 4).  All three geometries
    make that trade -- ``cartesian._imm_iteration_vw`` carries the
    shared argument and the measurement (one-step spectral radius
    `$\le 1$`) -- and in all three it is the only wall quantity taken
    from `$t^n$`: the lagged spin partners above and the pipe's
    surplus wall differences both ride the corrector iterate.
    Carrying `$\Phi$` would mean a third representation of the state
    for the sake of two rows, which was tried on Cartesian and
    reverted.

    Mean mode and padding
    ~~~~~~~~~~~~~~~~~~~~~
    At `$k^2 = 0$` the reconstruction is singular and both evolved
    scalars are structurally zero (`$u_{r,00} \equiv 0$`, and every
    term carries `$m$` or `$k_z$`), so the two slots carry the mean
    axial and azimuthal momentum instead: `$\Phi_{00} := u_{z,00}$`
    with `$m_{\mathrm{eff}}^2 = 0$` (:func:`_vw_meff2`) and
    `$\omega_{00} := u_{\theta,00}$`, whose `$m^2 + 1 = 1$` operator is
    already the primitive scheme's mean `$u_\pm$` one.  Both packed
    updates then reproduce the primitive mean update term for term (the
    mean pressure gradient is `$D_1 p$` in *both* `$u_\pm$` rows, so it
    cancels out of `$u_\theta$` exactly as the mean-`$u_r$` projection
    removes it), which is what lets Dean's body force and the bulk
    corrections ride unchanged.  Padding modes need no special-casing:
    their placeholder wavenumbers keep `$\Delta > 0$`, and zero fields
    stay zero through every stage.
    """
    c = params.step.implicitness
    dt = flow_.dt
    nu = derived_params.nu  # solvent viscosity (see __post_init__)

    im = 1j * fourier_.m
    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]
    inv_r2 = flow_.inv_r2[:, None, None]
    kz2 = fourier_.kz2
    mean_mask = fourier_.mean_mask

    # Spin-diagonal m_eff^2 of the pair, and the Phi slot's packed
    # mean-plane exception -- the explicit twin of _vw_meff2.
    pair2 = fourier_.m2 + 1.0
    phi2 = jnp.where(mean_mask, 0.0, pair2)
    # The 2 i m / r^2 spin coupling; identically zero at m = 0.
    spin = 2.0 * im * inv_r2

    # Stage 0: cross into physical components.  Both maps are diagonal
    # in the component axis, so the corrector's CN combination is
    # formed in u_+/u_- and converted once.
    state_n = from_pm_basis(velocity_n)
    state_cn = from_pm_basis(c * velocity_j + (1 - c) * velocity_n)
    nonlin = from_pm_basis(c * nonlin_j + (1 - c) * nonlin_n)

    # Stage 1: two batched FD matvecs.  The two fields entering a
    # vector Laplacian need exactly `$A_\mathrm{base} = D_2 + (1/r)
    # D_1$` and nothing else from `$D_1$`, so they take the fused
    # operator (one GEMM per field instead of two, and no field-sized
    # `$1/r$` multiply-add); the rest need `$D_1$` alone.  Both stacks
    # are y-leading, so both GEMMs stay transpose-free.
    A_in = jnp.stack([state_n[1], state_cn[2]], axis=1)
    A_pair_n = apply_y_matrix(flow_.A_base, A_in, component_axis=1)
    A_ur_n, A_ut_it = A_pair_n[:, 0], A_pair_n[:, 1]
    d1_in = jnp.stack(
        [
            state_cn[0],  # u_z^it               -> omega_theta
            nonlin[0],  # N_z                    -> C_theta
            flow_.rs[:, None, None] * nonlin[2],  # r N_theta -> C_z
        ],
        axis=1,
    )
    d1 = apply_y_matrix(flow_.D1, d1_in, component_axis=1)

    # Stage 2: the evolved scalars, recomputed from the carried
    # u_+/u_- state on FULL rows (walls included).  The mean plane
    # carries the packed mean momentum instead (docstring).
    phi_n = A_ur_n - (pair2 * inv_r2 + kz2) * state_n[1] - spin * state_n[2]
    omega_n = im * inv_r * state_n[0] - ikz * state_n[2]
    phi_n = jnp.where(mean_mask, state_n[0], phi_n)
    omega_n = jnp.where(mean_mask, state_n[2], omega_n)

    # Stage 3: the pressure-free sources -- the discrete double curl,
    # with the conservative C_z that annihilates a discrete gradient
    # exactly (docstring).
    C_r = im * inv_r * nonlin[0] - ikz * nonlin[2]
    C_theta = ikz * nonlin[1] - d1[:, 1]  # D1 N_z
    C_z = inv_r * (d1[:, 2] - im * nonlin[1])
    S_phi = jnp.where(
        mean_mask, nonlin[0], -(im * inv_r * C_z - ikz * C_theta)
    )
    S_omega = jnp.where(mean_mask, nonlin[2], C_r)

    # Stage 4: the explicit CN half of both slots (they share L_v),
    # plus the spin partners lagged to the running iterate.
    pair_n = jnp.stack([phi_n, omega_n], axis=1)  # (Nr, 2, Nm, Nkz)
    inv_r2_y = inv_r2[..., None]  # (Nr, 1, 1, 1) over the C axis
    # One fused `$A_\mathrm{base}$` matvec, as in stage 1.
    A_pair = apply_y_matrix(flow_.A_base, pair_n, component_axis=1)
    # ``pair2`` rides ``m2``'s spec, which is unsharded on the k_z
    # (np1) axis, while ``phi2`` inherited the mean mask's full one --
    # so the broadcast must be given the target sharding explicitly or
    # the stack below is a cross-spec operand mismatch under np1 > 1.
    meff2_pair = jnp.stack(
        [
            phi2,
            jnp.broadcast_to(
                pair2, phi2.shape, out_sharding=sharding.spec_scalar_shard
            ),
        ],
        axis=1,
    )
    lapl_pair = A_pair - (meff2_pair * inv_r2_y + kz2[:, None]) * pair_n
    partner = jnp.stack(
        [
            A_ut_it
            - (pair2 * inv_r2 + kz2) * state_cn[2]
            + spin * state_cn[1],  # (Delta u)_theta
            ikz * state_cn[1] - d1[:, 0],  # omega_theta (D1 u_z^it)
        ],
        axis=1,
    )
    R_stack = (
        pair_n / dt
        + (1 - c) * nu * lapl_pair
        - nu * spin[:, None] * partner
        + jnp.stack([S_phi, S_omega], axis=1)
    )
    # Dirichlet wall rows: zero is omega_r's physical value and Phi's
    # arbitrary particular choice (the influence matrix supplies the
    # rest in stage 6).
    R_stack = R_stack.at[0].set(0.0).at[-1].set(0.0)
    arb = flow_.Hk_op.solve(R_stack, component_axis=1)
    phi_arb, omega_new = arb[:, 0], arb[:, 1]

    # Stage 5: exact recovery of u_r.  Lk_op holds L_v,mod here,
    # with Dirichlet identity wall rows; phi_arb and omega_new both
    # vanish at the walls, so u_r|wall = 0 exactly.
    det = kz2 + fourier_.m2 * inv_r2
    inv_det = 1.0 / jnp.where(mean_mask, 1.0, det)
    om_shift = 2.0 * fourier_.m * fourier_.kz * inv_r2 * inv_det
    ur_arb = flow_.Lk_op.solve(phi_arb - om_shift * omega_new)

    # Stage 6: influence matrix -- the two free Phi wall values that
    # make (D1 u_r)|wall = 0.
    d_wall = jnp.einsum("bj, jmz -> mzb", flow_.D1_bnd, ur_arb)
    alpha = -jnp.einsum("mzab, mzb -> mza", flow_.M_inv, d_wall)
    ur_new = (
        ur_arb
        + alpha[..., 0][None] * flow_.ur_1
        + alpha[..., 1][None] * flow_.ur_2
    )

    # Stage 7: per-point reconstruction of (u_z, u_theta) from the
    # continuity row and the omega_r definition -- the stage that makes
    # the discrete divergence vanish at every row.
    chi = -(apply_y_matrix(flow_.D1, ur_new) + inv_r * ur_new)
    b_th = im * inv_r
    uz_new = (-ikz * chi - b_th * omega_new) * inv_det
    ut_new = (-b_th * chi + ikz * omega_new) * inv_det

    # Stage 8: unpack the mean plane (which inv_det left at zero) and
    # zero the mean-mode u_r, which continuity forces.
    uz_new = jnp.where(mean_mask, phi_arb, uz_new)
    ut_new = jnp.where(mean_mask, omega_new, ut_new)
    ur_new = jnp.where(mean_mask, 0.0, ur_new)

    uz_new, aux = _apply_bulk_correction(uz_new, uz_new, mean_mask, flow_)

    velocity_new = to_pm_basis(jnp.stack([uz_new, ur_new, ut_new]))
    correction = velocity_new - velocity_j

    return velocity_new, correction, aux


def _imm_iteration(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> tuple[Array, Array, dict[str, Array]]:
    r"""One implicit annular step: dispatch on ``res.consistent_imm``.

    Two formulations of the same second-order-in-time scheme, sharing
    the carried `$(u_z, u_+, u_-)$` state, the signature, the
    capacitance-matrix structure and the `$2 \times 2$` shape of its
    influence matrix:

    - **on, the default** -- :func:`_imm_iteration_vw`, the
      `$u_r$`-`$\omega_r$` formulation: advance the radial velocity and
      vorticity, reconstruct `$(u_z, u_\theta)$`, and never form a
      pressure.
    - **off, the legacy path** --
      :func:`._annular_primitive_imm._imm_iteration_vp`, the primitive
      Kleiser-Schumann influence-matrix method: solve for
      `$(u_z, u_+, u_-)$` against a pressure Poisson solve, enforcing
      continuity at the two walls.  Kept for reference and for
      reproducing older trajectories; not recommended.

    The branch is a Python ``if`` on a parameter fixed before this
    module is imported, so it costs nothing at trace time and the two
    bodies never mix.  The legacy body lives in a sibling module
    imported only here, so the default path never loads it.

    Why there are two, and why the second one is *this* one, is
    derived once for all three geometries in the Cartesian dispatcher
    :func:`~dnsjax.geometries.wall_bounded.cartesian._imm_iteration`:
    the primitive scheme's stepped state carries an `$O(1)$` relative
    discrete-divergence residual (a pressure-operator mismatch, a
    `$[D_1, D_2]$` commutator and a boundary term, the last two
    boundary-amplified by `$D_1[1,0] \sim N_y^2$`), and of the five
    repairs measured only the wall-normal velocity/vorticity
    reformulation both removes it by construction and survives
    nonlinear integration.  Here that record applies verbatim with one
    cylindrical amendment: route 1's operator-side identity
    (`$D_2 := D_1 D_1$` plus a `$4 \times 4$` Kleiser-Schumann
    closure) was this geometry's shipped mechanism until 2026-07-26 and
    reached `$d \sim 7\times10^{-5}$` -- five orders short of the
    Cartesian result, because `$D_2 := D_1 D_1$` cannot also make the
    *metric* commutator `$[D_1, 1/r]$` vanish.  The reconstruction has
    no such floor: it needs no operator identity at all, so the
    residual is machine-eps and flat under refinement.
    """
    if params.res.consistent_imm:
        return _imm_iteration_vw(
            velocity_n, velocity_j, nonlin_n, nonlin_j, fourier_, flow_
        )

    from . import _annular_primitive_imm as prim

    return prim._imm_iteration_vp(
        velocity_n, velocity_j, nonlin_n, nonlin_j, fourier_, flow_
    )


def _predict(
    velocity_n: Array,
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> Array:
    """Euler predictor via the annular IMM."""
    nonlin_n = rhs_no_lapl
    prediction_state, _, _ = _imm_iteration(
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
) -> tuple[Array, Array, dict[str, Array]]:
    """Crank-Nicolson corrector via the annular IMM.

    Third return: the corrector-side *aux* diagnostics, here the
    applied mean-mode driving (:func:`_apply_bulk_correction`).
    """
    return _imm_iteration(
        state_prev, prediction_state, rhs_prev, rhs_next, fourier_, flow_
    )


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: AnnularFlow,
) -> Array:
    r"""L2 convergence norm of a solver-basis correction.

    Corrections live in the decoupled `$(u_z, u_+, u_-)$` basis, so
    the 1/2 weight on the pair makes this the *physical* norm of the
    corresponding correction
    (`$|u_r|^2 + |u_\theta|^2 = (|u_+|^2 + |u_-|^2)/2$`) -- the same
    scalar :func:`get_norm2_annular` reports for a physical-basis
    array.
    """
    pm2 = get_norm2(correction[1:], fourier_.k_metric, flow_.y_weights)
    uz2 = get_norm2(correction[:1], fourier_.k_metric, flow_.y_weights)
    return jnp.sqrt(uz2 + pm2 / 2)


# ── Stepper factory ─────────────────────────────────────────────


def build_annular_stepper(
    flow: AnnularFlow,
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
    """Build time-stepping functions for an annular flow.

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
