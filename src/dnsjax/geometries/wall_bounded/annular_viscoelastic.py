r"""Viscoelastic (sPTT) extension of the annular geometry.

Adds a symmetric conformation tensor `$\mathbf{c}$` to the annular
(concentric-cylinder) geometry, coupling it to the velocity via the
polymer-stress divergence.  The time-integrated state grows from the 3
velocity components to **9**: 3 velocity + 6 independent symmetric-tensor
components, carried as a single stacked spectral array
``(9, Nr, Nm, Nkz)`` in the layout

.. math::
    [\,u_z,\; u_+,\; u_-,\;
      c_{zz},\; c_{z+},\; c_{z-},\; c_{+-},\; c_{++},\; c_{--}\,]

where the velocity uses the decoupled `$u_\pm = u_r \pm i u_\theta$`
formulation of :mod:`~dnsjax.geometries.wall_bounded.annular`, and the
tensor uses the analogous **spin** projections

.. math::
    c_{z\pm} = c_{rz} \pm i c_{\theta z}, \qquad
    c_{+-}   = c_{rr} + c_{\theta\theta}, \qquad
    c_{\pm\pm} = (c_{rr} - c_{\theta\theta}) \pm 2 i c_{r\theta}.

As in the annular geometry this is the solver's **working** basis.
Outside the time stepper -- snapshots, diagnostics, probes, initial
conditions, the analysis package -- the state is the physical
9-component layout

.. math::
    [\,u_z,\; u_r,\; u_\theta,\;
      c_{zz},\; c_{rz},\; c_{\theta z},\;
      c_{rr},\; c_{\theta\theta},\; c_{r\theta}\,],

and a given state crosses between the two at most once
(:func:`to_spin_basis` / :func:`from_spin_basis`, driven by
:mod:`dnsjax.__main__`).

Spin diagonalisation of the tensor Laplacian
--------------------------------------------
The cylindrical Laplacian couples the physical tensor components through
`$1/r^2$` terms, exactly as the vector Laplacian couples `$u_r, u_\theta$`.
Writing `$\partial_\theta \to im$` and collecting the basis-rotation
generator `$\mathcal R$` (the `$\partial_\theta$` action on the tensor
basis), the angular part of the Laplacian is
`$\tfrac{1}{r^2}(\mathcal R + im)^2$`.  Each spin projection is an
eigenvector of `$\mathcal R$` with eigenvalue `$is$` (spin weight
`$s$`), so `$(\mathcal R + im)^2 \to -(m + s)^2$` and the tensor
Laplacian **diagonalises**:

.. math::
    (\nabla^2 \mathbf{c})_{\text{spin }s} =
    \Bigl[\partial_r^2 + \tfrac{1}{r}\partial_r
    - \tfrac{(m+s)^2}{r^2} - k_z^2\Bigr]\,\mathbf{c}_{\text{spin }s},

with spin weights `$s = 0$` for `$c_{zz}, c_{+-}$`, `$s = \pm1$` for
`$c_{z\pm}$`, and `$s = \pm2$` for `$c_{\pm\pm}$` -- the same mechanism as
`$u_\pm$` (`$m_{\mathrm{eff}} = m \pm 1$`).  Each spin component therefore
diffuses through a scalar Helmholtz solve with its own
`$m_{\mathrm{eff}}$`, reusing the annular dense/Pallas machinery.

Governing equations (sPTT)
--------------------------
.. math::
    \partial_t \mathbf{u} + \mathbf{u}\cdot\nabla\mathbf{u}
      &= -\nabla p + \tfrac{\beta}{\mathrm{Re}}\nabla^2\mathbf{u}
      + \tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}\nabla\cdot\mathbf{c}
      + \boldsymbol{\Pi}, \\
    \partial_t \mathbf{c} + \mathbf{u}\cdot\nabla\mathbf{c}
      - (\nabla\mathbf{u})^{\!\top}\!\cdot\mathbf{c}
      - \mathbf{c}\cdot\nabla\mathbf{u}
      &= \kappa\nabla^2\mathbf{c}
      - \tfrac{\mathbf{c}-\mathbb{I}}{\mathrm{Wi}}
        (1 - 3\epsilon + \epsilon\,\mathrm{tr}\,\mathbf{c}),

with no-slip `$\mathbf{u}=0$` and `$\nabla^2\mathbf{c}=0$` at both walls,
and the azimuthal body force `$\Pi_\theta = (r_1+r_2)/(\mathrm{Re}\,r)$`
(see :mod:`~dnsjax.flows.wall_bounded.viscoelastic_dean`).  All products
are at most quadratic (`$\mathrm{tr}(\mathbf{c})\,\mathbf{c}$`), so the
existing 3/2-rule dealiasing is exact.

Time integration
----------------
Both ``iterative-cn`` (default) and ``cnab2`` schemes are supported.
``get_rhs`` returns the full 9-component nonlinear RHS -- velocity
(`$\mathbf{u}\times\boldsymbol\omega$` + `$\boldsymbol\Pi$` + FFT-free
polymer divergence
`$\tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}\nabla\cdot\mathbf{c}$`) and
conformation (`$-\mathbf{u}\cdot\nabla\mathbf{c} + (\nabla u)^{\!\top}c
+ c\nabla u - \tfrac{f(\mathrm{tr}\,c)}{\mathrm{Wi}}(c-\mathbb{I})$`) --
built from a single fused pseudo-spectral evaluation (one batched
inverse transform of ~36 fields, one batched forward transform of the 9
outputs; the vorticity is free from the velocity-gradient tensor).  The
predictor/corrector then solves the velocity via the annular 2x2 IMM
(:func:`~dnsjax.geometries.wall_bounded.annular._imm_iteration`,
solvent viscosity `$\nu = \beta/\mathrm{Re}$`) and the conformation via a
Crank-Nicolson Helmholtz solve per spin component (`$H_c = \tfrac1{\Delta
t}I - \theta\kappa\nabla^2$`, Laplacian BC wall rows).  With
`$\kappa = 0$` the transport is purely hyperbolic (no wall BC) and the
conformation update degenerates to the explicit CN combination.

The ``cnab2`` scheme (one FFT/step) makes the FFT-free linear/mean
coupling implicit via :func:`_l_bf` -- velocity mean-flow coupling +
polymer-stress divergence, conformation mean advection / mean-shear
stretching + linear relaxation (all gated / structured so the explicit
AB2 remainder is the pure fluctuation-fluctuation nonlinearity plus the
nonlinear relaxation) -- and advances that remainder explicitly.  It
reproduces ``iterative-cn`` to O(`$\Delta t^2$`) at ~1 FFT/step versus
~4 (the coupled tensor system inherits the wall-bounded velocity's
reduced projection-splitting order, shared by both schemes).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from ...fd import build_diff_matrices, fornberg_weights
from ...fft import chunked_transform
from ...measurements import get_cfl
from ...operators import phys_to_spec_2d, spec_to_phys_2d
from ...parameters import derived_params, params
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
    extract_mean_mode,
    get_norm2,
    integrate_scalar,  # noqa: F401 -- re-exported for the flow module
)
from .annular import (
    CFL_NAMES,
    AnnularFlow,
    Fourier,
    _build_A_base,
    _imm_iteration,
    fourier,
)
from .annular import (
    _build_dt_leaves as _annular_dt_leaves,
)
from .annular import (
    _l_bf as _annular_l_bf,
)

# ── State layout ────────────────────────────────────────────────────
#
# Solver basis: state[0:3] = velocity (u_z, u_+, u_-); state[3:9] =
# conformation spin components (c_zz, c_z+, c_z-, c_+-, c_++, c_--).
# Physical (everything outside the stepper): (u_z, u_r, u_theta) +
# (c_zz, c_rz, c_theta_z, c_rr, c_theta_theta, c_r_theta) -- the
# to_spin_basis / from_spin_basis pair below.
N_VE_COMPONENTS = 9
_N_TENSOR = 6

# Spin weight s per tensor spin component (solver slot order c_zz,
# c_z+, c_z-, c_+-, c_++, c_--); the Laplacian / Helmholtz uses
# m_eff = m + s.
_TENSOR_SPIN = np.array([0, 1, -1, 0, 2, -2])

# Frobenius weights of the *physical* tensor components (c_zz, c_rz,
# c_theta_z, c_rr, c_theta_theta, c_r_theta): off-diagonals count
# twice in ||c||_F^2 = sum_ij |c_ij|^2.  Used by the diagnostic norm.
_C_FROB_WEIGHT = np.array([1.0, 2.0, 2.0, 1.0, 1.0, 2.0])
# sqrt, precomputed (applied per component before the shared get_norm2).
_C_FROB_SQRT = np.sqrt(_C_FROB_WEIGHT)

# The same physical scalar expressed on the *spin* slots: ||c||_F^2 =
# |c_zz|^2 + |c_z+|^2 + |c_z-|^2 + |c_+-|^2/2 + (|c_++|^2+|c_--|^2)/4.
# Used by the corrector ``_norm``, whose arguments are solver-basis.
_C_FROB_SQRT_SPIN = np.sqrt(np.array([1.0, 1.0, 1.0, 0.5, 0.25, 0.25]))


# ── Spin <-> physical tensor conversions (linear, any space) ────────


def _spin_to_phys_combos(
    c_zz: Array,
    c_zp: Array,
    c_zm: Array,
    c_pm: Array,
    c_pp: Array,
    c_mm: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    r"""Spin components `$\to$` physical `$(c_{rr}, c_{\theta\theta},
    c_{r\theta}, c_{rz}, c_{\theta z}, c_{zz})$`."""
    d = (c_pp + c_mm) / 2  # = c_rr - c_theta_theta
    c_rr = c_pm / 2 + d / 2
    c_thth = c_pm / 2 - d / 2
    c_rth = -0.5j * (c_pp - c_mm) / 2  # (c_++ - c_--)/(4i)
    c_rz = (c_zp + c_zm) / 2
    c_thz = -0.5j * (c_zp - c_zm)  # (c_z+ - c_z-)/(2i)
    return c_rr, c_thth, c_rth, c_rz, c_thz, c_zz


def _phys_combos_to_spin(
    c_rr: Array,
    c_thth: Array,
    c_rth: Array,
    c_rz: Array,
    c_thz: Array,
    c_zz: Array,
) -> Array:
    r"""Physical tensor components `$\to$` stacked spin components,
    ``(6, ...)`` in the order `$(c_{zz}, c_{z+}, c_{z-}, c_{+-},
    c_{++}, c_{--})$`."""
    c_zp = c_rz + 1j * c_thz
    c_zm = c_rz - 1j * c_thz
    c_pm = c_rr + c_thth
    c_pp = (c_rr - c_thth) + 2j * c_rth
    c_mm = (c_rr - c_thth) - 2j * c_rth
    return jnp.array([c_zz, c_zp, c_zm, c_pm, c_pp, c_mm])


# ── Physical <-> solver basis (the 9-component boundary) ────────────


def to_spin_basis(state: Array) -> Array:
    r"""Physical 9-component state `$\to$` solver spin basis.

    Maps `$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` to
    `$(u_z, u_+, u_-, c_{zz}, c_{z+}, c_{z-}, c_{+-}, c_{++},
    c_{--})$` -- the per-mode-diagonal basis of the `$H_k$`/`$H_c$`
    solves in which the whole time stepper works (every entry a spin
    projection: `$u_\pm$` are the spin-`$\pm1$` components of the
    velocity vector).  The 9-component counterpart of
    :func:`~dnsjax.geometries.wall_bounded._base.to_pm_basis`, and
    like it the boundary crossed at most once per state
    (:mod:`dnsjax.__main__`); elementwise on the unsharded component
    axis.  Inverse: :func:`from_spin_basis`.
    """
    u_r, u_theta = state[1], state[2]
    vel = jnp.array([state[0], u_r + 1j * u_theta, u_r - 1j * u_theta])
    spin = _phys_combos_to_spin(
        state[6], state[7], state[8], state[4], state[5], state[3]
    )
    return jnp.concatenate([vel, spin])


def from_spin_basis(state: Array) -> Array:
    r"""Solver spin basis `$\to$` physical 9-component state.

    Inverse of :func:`to_spin_basis`.
    """
    u_plus, u_minus = state[1], state[2]
    c_rr, c_thth, c_rth, c_rz, c_thz, c_zz = _spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    return jnp.array(
        [
            state[0],
            (u_plus + u_minus) / 2,
            -1j * (u_plus - u_minus) / 2,
            c_zz,
            c_rz,
            c_thz,
            c_rr,
            c_thth,
            c_rth,
        ]
    )


#: Role aliases for the basis boundary (see ``cylindrical.py``).
to_solver_basis = to_spin_basis
from_solver_basis = from_spin_basis


# ── Analytical laminar profiles (JAX-free, build-time) ──────────────


def _solve_ptt_f(g: np.ndarray) -> np.ndarray:
    r"""Solve `$f^3 - f^2 = g$` for `$f \ge 1$` (Newton, `$g \ge 0$`).

    `$g = 2\epsilon(\mathrm{Wi}\,S)^2$` is the sPTT extensibility term;
    `$f = 1$` for `$\epsilon = 0$` (or zero shear).
    """
    f = np.ones_like(g)
    for _ in range(100):
        num = f**3 - f**2 - g
        den = 3.0 * f**2 - 2.0 * f
        f = f - num / den
    return f


def viscoelastic_laminar_profiles(
    rs: np.ndarray, D1: np.ndarray, r1: float, r2: float, wi: float, eps: float
) -> np.ndarray:
    r"""9-component laminar `$r$`-profiles for a force-driven annular
    sPTT flow (complex ``(9, Nr)``), in the **physical** state layout
    `$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` -- these feed initial conditions
    and the flow's laminar reference, both of which live outside the
    solver (:func:`to_spin_basis` converts when one enters it).

    Velocity: the azimuthal profile `$U_\theta(r)$` (body-force
    coefficient `$C = r_1 + r_2$`).  Conformation: the pointwise sPTT
    equilibrium on the **discrete** local shear `$S = D_1 U_\theta -
    U_\theta/r$` (see the
    :mod:`~dnsjax.flows.wall_bounded.viscoelastic_dean` module
    docstring): `$c_{rr} = c_{zz} = 1$`, `$c_{r\theta} =
    \mathrm{Wi}\,S/f$`, `$c_{\theta\theta} = 1 + 2 c_{r\theta}^2$`.
    Pure (NumPy, build-time); shared by the flow's laminar state and
    the viscoelastic random / rolls ICs.
    """
    from .annular import annular_forced_laminar_u_theta

    rs_np = np.asarray(rs)
    u_theta = np.asarray(
        annular_forced_laminar_u_theta(jnp.asarray(rs_np), r1, r2, r1 + r2)
    )
    shear = np.asarray(D1) @ u_theta - u_theta / rs_np
    wis = wi * shear
    f = _solve_ptt_f(2.0 * eps * wis**2)
    x = wis / f  # c_r_theta
    c_thth = 1.0 + 2.0 * x**2  # c_theta_theta
    zeros = np.zeros_like(rs_np, dtype=np.complex128)
    ones = np.ones_like(rs_np, dtype=np.complex128)
    return np.stack(
        [
            zeros,  # u_z
            zeros,  # u_r
            u_theta,  # u_theta
            ones,  # c_zz
            zeros,  # c_rz
            zeros,  # c_theta_z
            ones,  # c_rr
            c_thth,  # c_theta_theta
            x,  # c_r_theta
        ]
    ).astype(np.complex128)


# ── Narrow (banded-storage-fitting) Laplacian BC wall rows ──────────


def _narrow_abase_wall_rows(
    rs: np.ndarray, D1: np.ndarray, fd_order: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""Full `$A_{\mathrm{base}}$` wall rows using a **narrow** `$D_2$`.

    The regular `$D_2$` boundary row (:func:`dnsjax.fd.build_diff_matrices`)
    spans `$p+2$` points and does not fit banded storage (half-bandwidth
    `$p$`).  The `$\nabla^2 c = 0$` wall BC only needs the Laplacian
    *evaluated at the wall row*, so a `$(p+1)$`-point one-sided `$D_2$`
    stencil (accuracy `$p-1$`, acceptable for an artificial-diffusion BC)
    is used for the two wall rows only, giving a row that fits the band
    (columns `$0..p$` at the inner wall, `$N-p-1..N-1$` at the outer).
    The identical narrow row is used in every backend so all factor the
    same matrix.  `$D_1$` already fits (`$p+1$`-point), so only `$D_2$`
    is narrowed.  Returns the two full-length `$(N_r,)$` rows
    `$A_{\mathrm{base}} = D_2 + (1/r) D_1$`.
    """
    N = len(rs)
    p = fd_order
    row0 = np.zeros(N)
    rowN = np.zeros(N)
    # Inner wall (row 0): narrow one-sided D2 on nodes 0..p.
    w0 = fornberg_weights(rs[0], rs[0 : p + 1], 2)[:, 2]
    row0[0 : p + 1] = w0 + (1.0 / rs[0]) * D1[0, 0 : p + 1]
    # Outer wall (row N-1): narrow one-sided D2 on nodes N-p-1..N-1.
    wN = fornberg_weights(rs[-1], rs[N - p - 1 :], 2)[:, 2]
    rowN[N - p - 1 :] = wN + (1.0 / rs[-1]) * D1[-1, N - p - 1 :]
    return row0, rowN


# ── H_c Helmholtz operator builders (per spin component) ────────────


def _build_Hc_dense_gpu(
    A_base: Array,
    narrow0: Array,
    narrowN: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    kappa: float,
) -> Array:
    r"""Dense `$H_c = \tfrac1{\Delta t} I - c\kappa\nabla^2$` for one spin
    component (dense backend).  Interior rows carry the diagonal
    Helmholtz shift; both wall rows are the narrow Laplacian BC row
    `$A_{\mathrm{base}} - (m_{\mathrm{eff}}^2/r^2 + k_z^2) I$`."""
    Nr = A_base.shape[0]
    dtype = A_base.dtype
    eye_Nr = jnp.eye(Nr, dtype=dtype)
    diag_coeff = 1.0 / dt + c * kappa * (meff2 * inv_r2 + kz2)  # (Nm,Nkz,Nr)
    Hc = diag_coeff[..., None] * eye_Nr - c * kappa * A_base
    # Wall rows: narrow Laplacian BC (mode-dependent diagonal shift).
    shift0 = meff2 * inv_r2[0] + kz2  # (Nm, Nkz, 1)
    shiftN = meff2 * inv_r2[-1] + kz2
    row0 = narrow0[None, None] - shift0 * eye_Nr[0]  # (Nm, Nkz, Nr)
    rowN = narrowN[None, None] - shiftN * eye_Nr[-1]
    Hc = Hc.at[..., 0, :].set(row0)
    Hc = Hc.at[..., -1, :].set(rowN)
    return Hc


def _build_Hc_band_gpu(
    A_base: Array,
    narrow0: Array,
    narrowN: Array,
    meff2: Array,
    inv_r2: Array,
    kz2: Array,
    dt: float,
    c: float,
    kappa: float,
    p: int,
) -> Array:
    r"""Banded `$H_c$` for one spin component (Pallas backend), layout
    ``(Nm, Nkz, Nr, 2p+1)``; narrow Laplacian BC wall rows."""
    Nr = A_base.shape[0]
    band_base = _banded_from_dense(A_base, p)
    diag = 1.0 / dt + c * kappa * (meff2 * inv_r2 + kz2)  # (Nm, Nkz, Nr)
    e = _banded_diag_column(p, band_base.dtype)
    # Narrow BC bands (mode-constant) minus the mode-dependent shift.
    band0 = _banded_wall_row(narrow0, 0, p)  # (2p+1,)
    bandN = _banded_wall_row(narrowN, Nr - 1, p)
    shift0 = meff2 * inv_r2[0] + kz2  # (Nm, Nkz, 1)
    shiftN = meff2 * inv_r2[-1] + kz2
    row0 = band0 - shift0 * e  # (Nm, Nkz, 2p+1)
    rowN = bandN - shiftN * e
    return _assemble_banded_operator(
        band_base, -c * kappa, diag, [(0, row0), (Nr - 1, rowN)]
    )


# ── Spectral tensor operators (FFT-free) ────────────────────────────


def _tensor_laplacian_spin(
    c_spin: Array, fourier_: Fourier, flow_: ViscoelasticAnnularFlow
) -> Array:
    r"""Spin-diagonal tensor Laplacian, `$(6, N_r, N_m, N_{kz})$`.

    `$(\nabla^2 c)_{\text{spin }s} = A_{\mathrm{base}} c
    - (m_{\mathrm{eff}}^2/r^2 + k_z^2) c$` with
    `$m_{\mathrm{eff}} = m + s$` per spin component.
    """
    inv_r = flow_.inv_r[:, None, None]
    Abase_c = apply_y_matrix(flow_.D2, c_spin) + inv_r * apply_y_matrix(
        flow_.D1, c_spin
    )
    m = fourier_.m  # (1, Nm, 1)
    meff = m + flow_.tensor_spin[:, None, None, None]  # (6, 1, Nm, 1)
    meff2_over_r2 = (meff**2) * flow_.inv_r2[None, :, None, None]
    return Abase_c - (meff2_over_r2 + fourier_.kz2) * c_spin


def _div_c(
    c_rr: Array,
    c_thth: Array,
    c_rth: Array,
    c_rz: Array,
    c_thz: Array,
    c_zz: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> tuple[Array, Array, Array]:
    r"""Spectral divergence of the symmetric tensor, `$(\nabla\cdot
    c)_r, (\nabla\cdot c)_\theta, (\nabla\cdot c)_z$` (FFT-free):

    .. math::
        (\nabla\cdot c)_r &= \partial_r c_{rr}
            + \tfrac{im}{r}c_{r\theta} + ik_z c_{rz}
            + \tfrac{c_{rr}-c_{\theta\theta}}{r}, \\
        (\nabla\cdot c)_\theta &= \partial_r c_{r\theta}
            + \tfrac{im}{r}c_{\theta\theta} + ik_z c_{\theta z}
            + \tfrac{2 c_{r\theta}}{r}, \\
        (\nabla\cdot c)_z &= \partial_r c_{rz}
            + \tfrac{im}{r}c_{\theta z} + ik_z c_{zz}
            + \tfrac{c_{rz}}{r}.
    """
    im = 1j * fourier_.m
    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]
    dr = apply_y_matrix(
        flow_.D1, jnp.array([c_rr, c_rth, c_rz])
    )  # (3, Nr, Nm, Nkz)
    div_r = dr[0] + im * inv_r * c_rth + ikz * c_rz + inv_r * (c_rr - c_thth)
    div_th = dr[1] + im * inv_r * c_thth + ikz * c_thz + inv_r * 2 * c_rth
    div_z = dr[2] + im * inv_r * c_thz + ikz * c_zz + inv_r * c_rz
    return div_r, div_th, div_z


# ── Norms ───────────────────────────────────────────────────────────


def get_norm2_conformation(
    c_phys: Array, k_metric: Array, y_weights: Array
) -> Array:
    r"""Volume-averaged Frobenius norm `$\langle \|c\|_F^2 \rangle$` from
    the **physical** components `$(c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` (off-diagonal weights 2) -- the
    diagnostic form, applied outside the solver.  Its solver-basis
    counterpart is the spin weighting in :func:`_norm`."""
    w = jnp.asarray(_C_FROB_SQRT, dtype=c_phys.real.dtype).reshape(
        _N_TENSOR, 1, 1, 1
    )
    return get_norm2(c_phys * w, k_metric, y_weights)


# ── Viscoelastic annular flow dataclass ─────────────────────────────

_WallBoundedOp = DenseJAXSolver | PerModeBandedPallasOperator


@register_dataclass_pytree
@dataclass
class ViscoelasticAnnularFlow(AnnularFlow):
    r"""Precomputed data for viscoelastic (sPTT) annular flow.

    Extends :class:`~dnsjax.geometries.wall_bounded.annular.AnnularFlow`
    (velocity radial grid, FD matrices, 2x2 IMM operators -- built with
    solvent viscosity `$\nu = \beta/\mathrm{Re}$` via
    ``derived_params.nu``) with the conformation-tensor machinery: the
    stacked Crank-Nicolson Helmholtz operator ``Hc_op`` (6 spin
    components, `$m_{\mathrm{eff}} = m + s$`; two share
    `$m_{\mathrm{eff}} = m$`) and the physical `$1/r$` profile on the
    padded grid.  When `$\kappa = 0$` no Helmholtz operator is built
    (``Hc_op = None``): the conformation transport is hyperbolic and the
    update is the explicit CN combination.

    Subclasses (:class:`~dnsjax.flows.wall_bounded.viscoelastic_dean`)
    set ``pi_theta`` and zero the base flow (total-field integration).
    """

    tensor_spin: Array = field(init=False)
    inv_r_padded: Array = field(init=False)
    # `$A_{base}$` for the conformation operator.  The conformation
    # tensor is not solenoidal and never enters the IMM projection, so
    # it keeps the **direct-fit** `$D_2$` (more accurate, and a band
    # narrow enough that the six stacked `$H_c$` operators -- this
    # flow's largest allocation -- do not grow) even when
    # ``res.consistent_imm`` widens the velocity operators.  Aliases
    # ``A_base`` when the flag is off.
    A_base_c: Array = field(init=False)
    Hc_op: _WallBoundedOp | None = field(init=False, default=None)
    # Narrow Laplacian BC wall rows of Hc, stored as leaves so the
    # jitted adaptive-dt rebuild (``_build_dt_leaves``) can reuse
    # them (their NumPy build cannot run on tracers).  ``None`` (aux)
    # while kappa == 0, where no Hc exists.
    hc_narrow0: Array | None = field(init=False, default=None)
    hc_narrowN: Array | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Build velocity grid / FD matrices / 2x2 IMM (nu = beta/re).
        super().__post_init__()

        self.tensor_spin = jax.device_put(
            jnp.asarray(_TENSOR_SPIN, dtype=sharding.float_type),
            sharding.no_shard,
        )

        if params.res.consistent_imm:
            # Rebuild the direct-fit second derivative for `$H_c$`
            # (see the ``A_base_c`` field note).
            _, d2_np = build_diff_matrices(
                np.asarray(self.rs), params.res.fd_order
            )
            self.A_base_c = _build_A_base(
                self.D1,
                jax.device_put(d2_np, sharding.no_shard),
                self.inv_r,
            )
        else:
            self.A_base_c = self.A_base

        Nr = params.res.ny
        ny_phys = Nr + sharding.ny_y_pad
        inv_r_pad = np.zeros(ny_phys, dtype=sharding.float_type)
        inv_r_pad[:Nr] = np.asarray(self.inv_r)
        self.inv_r_padded = jax.device_put(
            inv_r_pad.reshape(ny_phys, 1, 1), sharding.no_shard
        )

        kappa = params.phys.kappa
        if kappa == 0:
            # Hyperbolic conformation transport: no diffusion, no wall BC.
            self.Hc_op = None
            return

        self._build_conformation_operator()

    def _build_conformation_operator(self) -> None:
        r"""Build the stacked 6-component `$H_c$` Crank-Nicolson operator.

        Stores the narrow Laplacian BC wall rows as the
        ``hc_narrow0`` / ``hc_narrowN`` leaves (their JAX-free NumPy
        build cannot run on tracers, so the jitted adaptive-``dt``
        rebuild reuses them), optionally pre-checks the no-pivot LU
        at ``dt_max`` (``step.adaptive``; the velocity `$H_k$`
        analogue), and delegates the assembly/factorization to
        :func:`_build_hc_operator`.
        """
        # Full narrow Laplacian BC wall rows (JAX-free build).
        row0_np, rowN_np = _narrow_abase_wall_rows(
            np.asarray(self.rs), np.asarray(self.D1), params.res.fd_order
        )
        self.hc_narrow0 = jax.device_put(row0_np, sharding.no_shard)
        self.hc_narrowN = jax.device_put(rowN_np, sharding.no_shard)

        if params.step.adaptive and params.solver.backend == "pallas":
            # Verify the no-pivot LU where the Helmholtz diagonal is
            # least dominant; adaptive rebuilds at dt <= dt_max then
            # skip the check (solvers._factor_pallas_operator).
            _build_hc_operator(
                params.step.dt_max, fourier, self, label="Hc(dt_max)"
            )
        self.Hc_op = _build_hc_operator(self.dt, fourier, self, label="Hc")


def _build_hc_operator(
    dt: float | Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
    *,
    label: str | None,
) -> _WallBoundedOp:
    r"""Factored 6-component `$H_c$` at *dt*.

    Five distinct `$m_{\mathrm{eff}}^2$` operators (`$m, m\pm1,
    m\pm2$`) are built and stacked into the 6-component order
    `$(c_{zz}, c_{z+}, c_{z-}, c_{+-}, c_{++}, c_{--})$`, so the
    `$m_{\mathrm{eff}} = m$` operator serves both `$c_{zz}$` and
    `$c_{+-}$`.

    The stacked storage **duplicates** that shared operator's
    factors (slot 0 and slot 3 hold the same data -- ~1/6 of the
    ``Hc_op`` memory), because the uniform stacked ``.solve``
    contract pairs component ``i`` of the RHS with operator ``i``.
    Deduplicating would need a nonuniform component-to-operator
    solve mapping (5 operators against 6 RHS components) in every
    backend -- deferred as not worth the contract complexity for a
    small, setup-persistent array (the velocity ``Hk_op`` stack and
    the per-step transform transients are far larger).

    *label* selects the pallas factorization path: a string runs the
    setup-checked :func:`solvers._build_pallas_operator` under that
    diagnostic label; ``None`` runs the unchecked, jittable
    :func:`solvers._factor_pallas_operator` (the ``set_dt``
    rebuild).  The dense backend is pivoted and ignores *label*.
    Wall rows come from the ``hc_narrow0``/``hc_narrowN`` leaves.
    """
    kappa = params.phys.kappa
    c_impl = params.step.implicitness
    fd_p = params.res.fd_order
    m_s = fourier_.m[0, ..., None]  # (Nm, 1, 1)
    kz2_s = fourier_.kz2[0, ..., None]  # (1, Nkz, 1)
    # m_eff^2 for spin s = 0, +1, -1, +2, -2 (the 5 distinct values).
    meff2 = {s: (m_s + s) ** 2 for s in (0, 1, -1, 2, -2)}

    if params.solver.backend == "pallas":
        # Six per-spin banded operators (slot 3 repeats s = 0),
        # stacked into one homogeneous operator.
        bands = [
            _build_Hc_band_gpu(
                flow_.A_base_c,
                flow_.hc_narrow0,
                flow_.hc_narrowN,
                meff2[s],
                flow_.inv_r2,
                kz2_s,
                dt,
                c_impl,
                kappa,
                fd_p,
            )
            for s in (0, 1, -1, 0, 2, -2)
        ]
        if label is not None:
            return _build_pallas_operator(bands, label)
        return _factor_pallas_operator(bands)

    def _dense(s: int) -> DenseJAXSolver:
        H = _build_Hc_dense_gpu(
            flow_.A_base_c,
            flow_.hc_narrow0,
            flow_.hc_narrowN,
            meff2[s],
            flow_.inv_r2,
            kz2_s,
            dt,
            c_impl,
            kappa,
        )
        return DenseJAXSolver(H)

    solvers_by_spin = {s: _dense(s) for s in (0, 1, -1, 2, -2)}
    order = [0, 1, -1, 0, 2, -2]
    return DenseJAXSolver.from_factors(
        lu=jnp.stack([solvers_by_spin[s].lu for s in order]),
        perm=jnp.stack([solvers_by_spin[s].perm for s in order]),
    )


def _build_dt_leaves(
    dt: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> dict[str, object]:
    r"""Rebuild every ``dt``-dependent flow leaf at the traced *dt*.

    The annular velocity set (`$H_k$` group + IMM leaves;
    ``annular._build_dt_leaves``, with the solvent
    `$\nu = \beta/\mathrm{Re}$` via ``derived_params.nu``) plus the
    conformation `$H_c$` (unchecked factorization,
    :func:`_build_hc_operator`) when diffusion is active.  At
    `$\kappa = 0$` ``Hc_op`` is ``None`` (static aux) and stays out
    of the rebuild -- the trace-time branch matches construction.
    """
    leaves = _annular_dt_leaves(dt, fourier_, flow_)
    if flow_.Hc_op is not None:
        leaves["Hc_op"] = _build_hc_operator(dt, fourier_, flow_, label=None)
    return leaves


# ── Fused pseudo-spectral RHS ───────────────────────────────────────


def _get_rhs_core(
    state: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
    measure_fn: Callable[[Array, Array], dict[str, Array]] | None,
) -> Array | tuple[Array, dict[str, Array]]:
    r"""Evaluate the full 9-component nonlinear RHS ``rhs_no_lapl``.

    One batched inverse transform of ~36 fields (velocity, velocity
    gradient `$L_{ij}$`, physical tensor, and its 18 advection
    derivatives), pointwise physical-space arithmetic, one batched
    forward transform of the 9 outputs.  The viscous / diffusive
    Laplacians are added implicitly by the predictor/corrector, so they
    are absent here.  See the module docstring.

    The two radial-derivative GEMMs (3 velocity + 6 conformation combos)
    are fused into one ``apply_y_matrix`` call, and at the default
    ``solver.rhs_transform_chunks = 1`` the inverse/forward transforms
    are each a **single batched** FFT over all fields (pinned by
    ``test_fused_rhs_transform_count``).

    **Memory vs throughput** (``solver.rhs_transform_chunks``): this
    36-field inverse transform dominates a viscoelastic step's peak
    memory -- not the held physical outputs themselves, but the
    transform's padded intermediate stage buffers (~2 complex copies
    of the whole batch at the dealiased size; see :mod:`dnsjax.fft`).
    The shared :func:`dnsjax.fft.chunked_transform` applies the knob:
    ``k`` balanced groups cut that transient by ~``k`` at the cost of
    ``k``x the FFT dispatches (and ``k`` smaller reshard rounds per
    stage on multi-device runs); the results are identical.

    **Deferred optimisation (interleaved transform/accumulate)**:
    chunking caps only the transform transient -- all 36 physical
    fields must still coexist as inputs of the single pointwise
    stage, so they plus the 9 outputs (~45 oversampled fields) are
    the floor the knob cannot cut.  That floor is decomposable
    because the pointwise stage has sparse field incidence: the 18
    advection derivatives are strictly per-component (only
    `$\mathrm{adv}(c_i)$` reads the
    `$(\partial_r, \partial_\theta, \partial_z) c_i$` triple),
    while the velocities, `$L_{ij}$`, and tensor combos are shared.
    Interleaving would hold the shared fields, then per component
    transform its derivative triple, multiply-accumulate its
    advection contribution into the output, and let the triple die
    before the next component's transform -- cutting the held floor
    to ~30 fields (further if the `$L_{ij}$` contributions are
    accumulated and freed first).  Deferred because it hard-codes
    chunking's throughput cost even when memory is not tight: the
    fused one-pass pointwise stage shatters into per-group kernels,
    the outputs are re-read/re-written once per group, the transform
    batches become permanently small, the schedule is specific to
    this RHS's term structure (unlike the flow-agnostic
    ``chunked_transform``), and the freeing relies on XLA liveness
    rather than construction.  The 9-output forward transform stays
    fused for the related reason that all outputs exist before it
    starts, so chunking it could shave only its own minor transient.
    """
    im = 1j * fourier_.m
    ikz = 1j * fourier_.kz
    inv_r = flow_.inv_r[:, None, None]

    # ── Spectral prep ──
    u_z, u_plus, u_minus = state[0], state[1], state[2]
    u_r = (u_plus + u_minus) / 2
    u_th = -0.5j * (u_plus - u_minus)

    # Conformation physical combos (still spectral here; cs_* denotes
    # the spectral tensor combos, distinct from the physical crr.. below).
    cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz = _spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    combos = jnp.array([cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz])

    # Single batched D1 GEMM over the 3 radial velocity derivatives
    # (velocity gradient L_ij = d_i u_j) and the 6 radial conformation
    # advection derivatives -- one GEMM instead of two (bit-identical;
    # the per-field matmul is batch-independent).
    dr_all = apply_y_matrix(
        flow_.D1,
        jnp.concatenate([jnp.array([u_r, u_th, u_z]), combos]),
    )
    Lrr, Lrth, Lrz = dr_all[0], dr_all[1], dr_all[2]
    dr_c = dr_all[3:9]  # (6, Nr, Nm, Nkz)
    Lthr = im * inv_r * u_r - inv_r * u_th
    Lthth = im * inv_r * u_th + inv_r * u_r
    Lthz = im * inv_r * u_z
    Lzr = ikz * u_r
    Lzth = ikz * u_th
    Lzz = ikz * u_z

    # Spectral advection derivatives of the conformation combos.
    dth_c = im * combos
    dz_c = ikz * combos

    # ── Batched inverse transform (36 fields) ──
    # One fused batch by default; ``solver.rhs_transform_chunks = k``
    # (trace-time, static) splits it into k balanced groups to cap the
    # transform-stage transient -- see the docstring.
    L_spec = jnp.array([Lrr, Lrth, Lrz, Lthr, Lthth, Lthz, Lzr, Lzth, Lzz])
    u_spec = jnp.array([u_z, u_r, u_th])
    stack = jnp.concatenate([u_spec, L_spec, combos, dr_c, dth_c, dz_c])
    phys = chunked_transform(spec_to_phys_2d, stack)

    uz_p, ur_p, uth_p = phys[0], phys[1], phys[2]
    Lrr_p, Lrth_p, Lrz_p = phys[3], phys[4], phys[5]
    Lthr_p, Lthth_p, Lthz_p = phys[6], phys[7], phys[8]
    Lzr_p, Lzth_p, Lzz_p = phys[9], phys[10], phys[11]
    crr, cthth, crth, crz, cthz, czz = (
        phys[12],
        phys[13],
        phys[14],
        phys[15],
        phys[16],
        phys[17],
    )
    drc = phys[18:24]
    dthc = phys[24:30]
    dzc = phys[30:36]

    inv_rp = flow_.inv_r_padded
    uth_over_r = uth_p * inv_rp

    # ── Velocity nonlinear term u x omega (rotational form) ──
    # omega is free from the antisymmetric part of L.
    om_r = Lthz_p - Lzth_p
    om_th = Lzr_p - Lrz_p
    om_z = Lrth_p - Lthr_p
    NLu_r = uth_p * om_z - uz_p * om_th
    NLu_th = uz_p * om_r - ur_p * om_z
    NLu_z = ur_p * om_th - uth_p * om_r

    # ── Conformation nonlinear term N_c (advection + stretching +
    # relaxation), physical, per component ──
    # advection scalar part adv(f) = u_r d_r f + (u_th/r) d_th f + u_z d_z f
    def _adv(k: int) -> Array:
        return ur_p * drc[k] + uth_over_r * dthc[k] + uz_p * dzc[k]

    adv_rr, adv_thth, adv_rth = _adv(0), _adv(1), _adv(2)
    adv_rz, adv_thz, adv_zz = _adv(3), _adv(4), _adv(5)
    ucgrad_rr = adv_rr - 2 * uth_over_r * crth
    ucgrad_thth = adv_thth + 2 * uth_over_r * crth
    ucgrad_rth = adv_rth + uth_over_r * (crr - cthth)
    ucgrad_rz = adv_rz - uth_over_r * cthz
    ucgrad_thz = adv_thz + uth_over_r * crz
    ucgrad_zz = adv_zz

    # Stretching S = L^T c + c L (symmetric); L_ij = d_i u_j.
    S_rr = 2 * (Lrr_p * crr + Lthr_p * crth + Lzr_p * crz)
    S_thth = 2 * (Lrth_p * crth + Lthth_p * cthth + Lzth_p * cthz)
    S_zz = 2 * (Lrz_p * crz + Lthz_p * cthz + Lzz_p * czz)
    S_rth = (
        Lrr_p * crth
        + Lthr_p * cthth
        + Lzr_p * cthz
        + crr * Lrth_p
        + crth * Lthth_p
        + crz * Lzth_p
    )
    S_rz = (
        Lrr_p * crz
        + Lthr_p * cthz
        + Lzr_p * czz
        + crr * Lrz_p
        + crth * Lthz_p
        + crz * Lzz_p
    )
    S_thz = (
        Lrth_p * crz
        + Lthth_p * cthz
        + Lzth_p * czz
        + crth * Lrz_p
        + cthth * Lthz_p
        + cthz * Lzz_p
    )

    # Relaxation -(c - I)(1 - 3eps + eps tr c)/Wi.
    wi = params.phys.wi
    eps = params.phys.epsilon
    trc = crr + cthth + czz
    fac = (1.0 - 3.0 * eps + eps * trc) / wi
    R_rr = -(crr - 1.0) * fac
    R_thth = -(cthth - 1.0) * fac
    R_zz = -(czz - 1.0) * fac
    R_rth = -crth * fac
    R_rz = -crz * fac
    R_thz = -cthz * fac

    Nc_rr = -ucgrad_rr + S_rr + R_rr
    Nc_thth = -ucgrad_thth + S_thth + R_thth
    Nc_rth = -ucgrad_rth + S_rth + R_rth
    Nc_rz = -ucgrad_rz + S_rz + R_rz
    Nc_thz = -ucgrad_thz + S_thz + R_thz
    Nc_zz = -ucgrad_zz + S_zz + R_zz

    # ── Single batched forward transform (9 outputs) ──
    out_phys = jnp.array(
        [
            NLu_z,
            NLu_r,
            NLu_th,
            Nc_rr,
            Nc_thth,
            Nc_rth,
            Nc_rz,
            Nc_thz,
            Nc_zz,
        ]
    )
    out_spec = phys_to_spec_2d(out_phys)
    NL_z, NL_r, NL_th = out_spec[0], out_spec[1], out_spec[2]

    # Azimuthal body force at the mean mode.
    NL_th = NL_th + jnp.where(
        fourier_.mean_mask, flow_.pi_theta[:, None, None], 0.0
    )
    # FFT-free polymer-stress divergence coef * div(c).
    coef = (1.0 - params.phys.beta) / (params.phys.re * wi)
    div_r, div_th, div_z = _div_c(
        cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz, fourier_, flow_
    )
    NL_z = NL_z + coef * div_z
    NL_r = NL_r + coef * div_r
    NL_th = NL_th + coef * div_th

    rhs_uz = NL_z
    rhs_up = NL_r + 1j * NL_th
    rhs_um = NL_r - 1j * NL_th

    # Conformation outputs -> spin components.
    Nc_spin = _phys_combos_to_spin(
        out_spec[3],
        out_spec[4],
        out_spec[5],
        out_spec[6],
        out_spec[7],
        out_spec[8],
    )

    rhs = jnp.concatenate([jnp.array([rhs_uz, rhs_up, rhs_um]), Nc_spin])

    # Moving-frame convective term (mode-diagonal on every component).
    u_grid = derived_params.u_grid
    if u_grid != 0:
        rhs = rhs + (1j * u_grid) * fourier_.kz * state

    if measure_fn is None:
        return rhs
    u_phys = jnp.array([uz_p, ur_p, uth_p])
    om_phys = jnp.array([om_z, om_r, om_th])
    measurements = measure_fn(u_phys, om_phys, trc)
    return rhs, measurements


def _get_rhs(
    state: Array, fourier_: Fourier, flow_: ViscoelasticAnnularFlow
) -> Array:
    """Evaluate the 9-component nonlinear RHS."""
    return _get_rhs_core(state, fourier_, flow_, None)


def _get_rhs_measured(
    state: Array, fourier_: Fourier, flow_: ViscoelasticAnnularFlow
) -> tuple[Array, dict[str, Array]]:
    """Evaluate the RHS + CFL / max-tr(c) measurements."""

    def _measure(
        u_phys: Array, om_phys: Array, trc: Array
    ) -> dict[str, Array]:
        meas = get_cfl(
            u_phys,
            flow_.base_flow_adv_padded,
            flow_.cfl_inv_spacing,
            CFL_NAMES,
            flow_.dt,
        )
        meas["TrC_max"] = jnp.max(trc)
        return meas

    return _get_rhs_core(state, fourier_, flow_, _measure)


# ── FFT-free linear / mean coupling (CN/AB2 scheme) ─────────────────


def _conformation_coupling(
    state: Array,
    combos: tuple[Array, Array, Array, Array, Array, Array],
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> Array:
    r"""FFT-free linear/mean conformation coupling, 6 spin components.

    The parts of the conformation RHS made implicit by the CN/AB2
    scheme:

    - the **linear relaxation** `$-(1-3\epsilon)(c-\mathbb{I})/
      \mathrm{Wi}$` and the moving-frame convective term -- **always**
      folded in (the linear reaction / frame terms, like the viscous
      Laplacian);
    - the instantaneous **mean-flow** coupling -- mean advection
      `$-\bar{\mathbf{u}}\cdot\nabla\mathbf{c}$` (the `$\bar u_\theta\,
      \partial_\theta$` / `$\bar u_z\,\partial_z$` parts with the
      curvilinear Christoffel corrections; `$\bar u_r \equiv 0$` at the
      mean mode) and mean-shear stretching `$\bar{L}^{\!\top}c + c\bar
      {L}$` with the mean velocity gradient `$\bar L$`
      (`$\bar L_{r\theta}=\partial_r\bar u_\theta$`,
      `$\bar L_{rz}=\partial_r\bar u_z$`,
      `$\bar L_{\theta r}=-\bar u_\theta/r$`; spin-mixing, fine for the
      Picard corrector) -- gated by ``params.step.implicit_mean_coupling``
      (default on), exactly as the velocity `$L_{mf}$` is (see the
      annular ``_l_bf``).

    Written on the physical tensor combos (identical algebra to
    :func:`_get_rhs_core`, restricted to `$\bar{\mathbf{u}}$`), so the
    explicit remainder `$\text{get\_rhs} - \text{\_l\_bf}$` is exactly
    the fluctuation-fluctuation advection / stretching and the nonlinear
    relaxation part.  No Fourier transform (mean profile x spectral
    field is a pointwise-in-`$r$` product).
    """
    cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz = combos

    # Linear relaxation -(1 - 3 eps)(c - I)/Wi (always implicit).  The
    # identity I is a constant field -- its spectral support is the mean
    # mode alone -- so subtract 1 from the diagonal c_rr / c_thth / c_zz
    # at the mean mode only (in physical space ``_get_rhs_core`` gets
    # this for free; here the arithmetic is spectral).
    faclin = (1.0 - 3.0 * params.phys.epsilon) / params.phys.wi
    ident = jnp.where(fourier_.mean_mask, 1.0, 0.0)
    Nc_rr = -(cs_rr - ident) * faclin
    Nc_thth = -(cs_thth - ident) * faclin
    Nc_zz = -(cs_zz - ident) * faclin
    Nc_rth = -cs_rth * faclin
    Nc_rz = -cs_rz * faclin
    Nc_thz = -cs_thz * faclin

    if params.step.implicit_mean_coupling:
        im = 1j * fourier_.m
        ikz = 1j * fourier_.kz
        inv_r = flow_.inv_r[:, None, None]

        # Instantaneous mean velocity profile (u_z, u_r, u_theta); the
        # mean u_r is structurally 0, so its d_r term vanishes.
        u_z, u_plus, u_minus = state[0], state[1], state[2]
        u_r = (u_plus + u_minus) / 2
        u_th = -0.5j * (u_plus - u_minus)
        mean_vel = extract_mean_mode(jnp.array([u_z, u_r, u_th]))  # (3, Nr)
        muz = mean_vel[0][:, None, None]
        uth_over_r = mean_vel[2][:, None, None] * inv_r

        # Mean velocity gradient profiles: D1 on the bare (N_r,) mean
        # profiles is a direct matmul (no Fourier axes here).
        Lbar_rth = (flow_.D1 @ mean_vel[2])[:, None, None]  # d_r u_theta
        Lbar_rz = (flow_.D1 @ mean_vel[0])[:, None, None]  # d_r u_z
        Lbar_thr = -uth_over_r  # -u_theta / r

        # Mean advection (u_r == 0): (u_theta/r) d_theta + u_z d_z, with
        # the same Christoffel corrections as ``_get_rhs_core``.
        def _madv(x: Array) -> Array:
            return uth_over_r * (im * x) + muz * (ikz * x)

        ucgrad_rr = _madv(cs_rr) - 2 * uth_over_r * cs_rth
        ucgrad_thth = _madv(cs_thth) + 2 * uth_over_r * cs_rth
        ucgrad_rth = _madv(cs_rth) + uth_over_r * (cs_rr - cs_thth)
        ucgrad_rz = _madv(cs_rz) - uth_over_r * cs_thz
        ucgrad_thz = _madv(cs_thz) + uth_over_r * cs_rz
        ucgrad_zz = _madv(cs_zz)

        # Mean-shear stretching L_bar^T c + c L_bar (spin-mixing).
        Nc_rr = Nc_rr - ucgrad_rr + 2 * Lbar_thr * cs_rth
        Nc_thth = Nc_thth - ucgrad_thth + 2 * Lbar_rth * cs_rth
        Nc_zz = Nc_zz - ucgrad_zz + 2 * Lbar_rz * cs_rz
        Nc_rth = Nc_rth - ucgrad_rth + Lbar_thr * cs_thth + cs_rr * Lbar_rth
        Nc_rz = Nc_rz - ucgrad_rz + Lbar_thr * cs_thz + cs_rr * Lbar_rz
        Nc_thz = Nc_thz - ucgrad_thz + Lbar_rth * cs_rz + cs_rth * Lbar_rz

    conf = _phys_combos_to_spin(Nc_rr, Nc_thth, Nc_rth, Nc_rz, Nc_thz, Nc_zz)

    u_grid = derived_params.u_grid
    if u_grid != 0:
        conf = conf + (1j * u_grid) * fourier_.kz * state[3:]
    return conf


def _l_bf(
    state: Array, fourier_: Fourier, flow_: ViscoelasticAnnularFlow
) -> Array:
    r"""FFT-free linear coupling for the CN/AB2 scheme, all 9 components.

    Velocity slice: the annular base/mean-flow coupling (:func:`~dnsjax.
    geometries.wall_bounded.annular._l_bf`, including the moving-frame
    term) plus the **polymer-stress divergence**
    `$\tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}\nabla\cdot\mathbf{c}$`
    (the elastic velocity`$\leftrightarrow$`conformation coupling, linear
    in `$c$` and FFT-free).  Conformation slice:
    :func:`_conformation_coupling`.

    ``step_cnab2`` advances the explicit remainder
    `$\text{get\_rhs} - \text{\_l\_bf}$` (pure fluctuation-fluctuation
    advection / stretching + nonlinear relaxation + the constant body
    force) with AB2 and makes this coupling implicit through the
    FFT-free corrector.  For total-field viscoelastic Dean the mean
    coupling (velocity *and* the large mean conformation profile) is the
    dominant stiffness, exactly as the mean-flow coupling is for
    Newtonian Dean.
    """
    vel_lbf = _annular_l_bf(state[:3], fourier_, flow_)

    cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz = _spin_to_phys_combos(
        state[3], state[4], state[5], state[6], state[7], state[8]
    )
    coef = (1.0 - params.phys.beta) / (params.phys.re * params.phys.wi)
    div_r, div_th, div_z = _div_c(
        cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz, fourier_, flow_
    )
    vel_lbf = vel_lbf + coef * jnp.array(
        [div_z, div_r + 1j * div_th, div_r - 1j * div_th]
    )

    conf_lbf = _conformation_coupling(
        state,
        (cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz),
        fourier_,
        flow_,
    )
    return jnp.concatenate([vel_lbf, conf_lbf])


# ── Conformation Crank-Nicolson update ──────────────────────────────


def _c_cn_update(
    c_n: Array,
    Nc_n: Array,
    Nc_j: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> Array:
    r"""Crank-Nicolson conformation update (6 spin components).

    Solves `$H_c c^{new} = \tfrac1{\Delta t} c^n + (1-\theta)\kappa
    \nabla^2 c^n + \theta N_c^j + (1-\theta) N_c^n$` with the wall-row
    RHS zeroed (the `$\nabla^2 c = 0$` BC).  With `$\kappa = 0$` there is
    no diffusion / wall BC and the update degenerates to
    `$c^{new} = c^n + \Delta t(\theta N_c^j + (1-\theta) N_c^n)$`.
    """
    dt = flow_.dt
    c_impl = params.step.implicitness
    nl = c_impl * Nc_j + (1.0 - c_impl) * Nc_n
    if flow_.Hc_op is None:  # kappa == 0 (trace-time branch)
        return c_n + dt * nl
    kappa = params.phys.kappa
    lap_cn = _tensor_laplacian_spin(c_n, fourier_, flow_)
    R = (1.0 / dt) * c_n + (1.0 - c_impl) * kappa * lap_cn + nl
    R = R.at[:, 0].set(0.0).at[:, -1].set(0.0)  # zero wall-row RHS
    return flow_.Hc_op.solve(R)


# ── Predictor / corrector / norm ────────────────────────────────────


def _correct(
    state_prev: Array,
    prediction: Array,
    rhs_prev: Array,
    rhs_next: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> tuple[Array, Array]:
    """Coupled velocity-IMM + conformation-CN corrector.

    Velocity: the annular 2x2 influence-matrix iteration.  Conformation:
    the Crank-Nicolson Helmholtz update.  The returned correction stacks
    both so the single convergence norm covers `$u$` and `$c$`.
    """
    vel_new, vel_corr = _imm_iteration(
        state_prev[:3],
        prediction[:3],
        rhs_prev[:3],
        rhs_next[:3],
        fourier_,
        flow_,
    )
    c_new = _c_cn_update(
        state_prev[3:], rhs_prev[3:], rhs_next[3:], fourier_, flow_
    )
    c_corr = c_new - prediction[3:]
    state_new = jnp.concatenate([vel_new, c_new])
    correction = jnp.concatenate([vel_corr, c_corr])
    return state_new, correction


def _predict(
    state_n: Array,
    rhs_no_lapl: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> Array:
    """Euler predictor (nonlinear at `$u^n$`, viscous/diffusive CN)."""
    prediction, _ = _correct(
        state_n, state_n, rhs_no_lapl, rhs_no_lapl, fourier_, flow_
    )
    return prediction


def _norm(
    correction: Array,
    fourier_: Fourier,
    flow_: ViscoelasticAnnularFlow,
) -> Array:
    r"""Combined L2 convergence norm, `$\sqrt{\|u\|^2 + \|c\|_F^2}$`.

    Corrections live in the solver spin basis, so the `$u_\pm$` pair
    carries the 1/2 weight and the tensor slots the spin Frobenius
    weights (``_C_FROB_SQRT_SPIN``) -- the same physical scalar the
    diagnostic norms report for a physical-basis array.
    """
    k_m, y_w = fourier_.k_metric, flow_.y_weights
    pm2 = get_norm2(correction[1:3], k_m, y_w)
    uz2 = get_norm2(correction[:1], k_m, y_w)
    w = jnp.asarray(_C_FROB_SQRT_SPIN, dtype=correction.real.dtype).reshape(
        _N_TENSOR, 1, 1, 1
    )
    c2 = get_norm2(correction[3:] * w, k_m, y_w)
    return jnp.sqrt(uz2 + pm2 / 2 + c2)


# ── Stepper factory ─────────────────────────────────────────────────


def build_viscoelastic_stepper(flow: ViscoelasticAnnularFlow):
    """Build time-stepping functions for a viscoelastic annular flow.

    Returns the same 9-tuple as
    :func:`~dnsjax.geometries.wall_bounded._base.build_wall_bounded_stepper`
    (incl. the adaptive-dt ``set_dt`` / ``reset_ab2_kappa``, backed
    by this module's ``_build_dt_leaves``).
    ``_l_bf`` (the FFT-free linear/mean coupling: velocity mean-flow
    coupling + polymer-stress divergence, conformation mean advection /
    stretching / linear relaxation) is passed so the CN/AB2 scheme treats
    it implicitly and the explicit AB2 remainder stays pure
    fluctuation-fluctuation nonlinearity.
    """
    from ._base import build_wall_bounded_stepper

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
