r"""Geometry-free machinery shared by the viscoelastic (sPTT) flows.

The sPTT extension exists in two wall-bounded geometries -- the annulus
(:mod:`~dnsjax.geometries.wall_bounded.annular_viscoelastic`) and the
pipe (:mod:`~dnsjax.geometries.wall_bounded.cylindrical_viscoelastic`).
Both write the same equations in the **same** coordinate system
(cylindrical `$(z, r, \theta)$`), so everything that is not a radial
operator or a boundary/axis condition is literally identical between
them and lives here: the spin `$\leftrightarrow$` physical tensor
algebra, the basis boundary, the sPTT scalar root, the Frobenius
weights, and the pointwise physical-space RHS kernel.

What is **not** here is exactly what the two geometries disagree on:
how a radial derivative is taken (a plain FD matrix on the annulus, a
parity-reduced pair on the pipe), how many walls carry a boundary row,
and the driving.  Each geometry module owns those.

State layout
------------
Solver (spin) basis, the stacked spectral array ``(9, N_r, N_m,
N_{kz})``:

.. math::
    [\,u_z,\; u_+,\; u_-,\;
      c_{zz},\; c_{z+},\; c_{z-},\; c_{+-},\; c_{++},\; c_{--}\,],

with `$u_\pm = u_r \pm i u_\theta$` and the tensor spin projections

.. math::
    c_{z\pm} = c_{rz} \pm i c_{\theta z}, \qquad
    c_{+-}   = c_{rr} + c_{\theta\theta}, \qquad
    c_{\pm\pm} = (c_{rr} - c_{\theta\theta}) \pm 2 i c_{r\theta}.

Physical basis, everything outside the time stepper (snapshots,
diagnostics, probes, initial conditions, the analysis package):

.. math::
    [\,u_z,\; u_r,\; u_\theta,\;
      c_{zz},\; c_{rz},\; c_{\theta z},\;
      c_{rr},\; c_{\theta\theta},\; c_{r\theta}\,].

A given state crosses between the two at most once
(:func:`to_spin_basis` / :func:`from_spin_basis`, driven by
:mod:`dnsjax.__main__`).

Spin weights
------------
Each spin projection is an eigenvector of the basis-rotation generator
`$\mathcal R$` (the `$\partial_\theta$` action on the tensor basis) with
eigenvalue `$is$`, so the angular part
`$\tfrac{1}{r^2}(\mathcal R + im)^2$` of the Laplacian becomes
`$-(m+s)^2/r^2$` and the tensor Laplacian **diagonalises** with
`$m_{\mathrm{eff}} = m + s$` (:data:`TENSOR_SPIN`).  The velocity is the
same mechanism at spin `$\pm1$` (`$u_\pm$`) and `$0$` (`$u_z$`).

In the pipe that spin weight also fixes the axis parity,
`$(-1)^{m+s}$` -- see the ``cylindrical_viscoelastic`` module
docstring.  The annulus has no axis and ignores it.
"""

import numpy as np
from jax import Array
from jax import numpy as jnp

from ...fd import fornberg_weights
from ._base import get_norm2

# ── Component counts and spin weights ───────────────────────────────

#: Stacked state components: 3 velocity + 6 symmetric-tensor.
N_VE_COMPONENTS = 9
#: Independent symmetric conformation-tensor components.
N_TENSOR = 6

#: Spin weight `$s$` per tensor spin component, in the solver slot
#: order `$(c_{zz}, c_{z+}, c_{z-}, c_{+-}, c_{++}, c_{--})$`; the
#: Laplacian / Helmholtz uses `$m_{\mathrm{eff}} = m + s$` (and the
#: pipe's axis parity is `$(-1)^{m+s}$`).
TENSOR_SPIN = np.array([0, 1, -1, 0, 2, -2])

#: Spin weight of every slot of the **physical** tensor combo tuple
#: `$(c_{rr}, c_{\theta\theta}, c_{r\theta}, c_{rz}, c_{\theta z},
#: c_{zz})$`, in the sense that only its parity `$s \bmod 2$` is
#: meaningful: a physical component's axis parity is set by how many
#: of its indices are in `$\{r, \theta\}$` (each flips sign under the
#: axis reflection), which is what these entries encode.
PHYS_COMBO_SPIN = np.array([2, 2, 2, 1, 1, 0])

# Frobenius weights of the *physical* tensor components (c_zz, c_rz,
# c_theta_z, c_rr, c_theta_theta, c_r_theta): off-diagonals count
# twice in ||c||_F^2 = sum_ij |c_ij|^2.  Used by the diagnostic norm.
_C_FROB_WEIGHT = np.array([1.0, 2.0, 2.0, 1.0, 1.0, 2.0])
#: sqrt, precomputed (applied per component before the shared get_norm2).
C_FROB_SQRT = np.sqrt(_C_FROB_WEIGHT)

#: The same physical scalar expressed on the *spin* slots:
#: `$\|c\|_F^2 = |c_{zz}|^2 + |c_{z+}|^2 + |c_{z-}|^2 + |c_{+-}|^2/2 +
#: (|c_{++}|^2+|c_{--}|^2)/4$`.  Used by the correctors' ``_norm``,
#: whose arguments are solver-basis.
C_FROB_SQRT_SPIN = np.sqrt(np.array([1.0, 1.0, 1.0, 0.5, 0.25, 0.25]))


# ── Spin <-> physical tensor conversions (linear, any space) ────────


def spin_to_phys_combos(
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


def phys_combos_to_spin(
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
    axis, so it needs no collective, applies unchanged to a single
    mode column ``(9, N_y)``, and is shared verbatim by both
    viscoelastic geometries.  Inverse: :func:`from_spin_basis`.
    """
    u_r, u_theta = state[1], state[2]
    vel = jnp.array([state[0], u_r + 1j * u_theta, u_r - 1j * u_theta])
    spin = phys_combos_to_spin(
        state[6], state[7], state[8], state[4], state[5], state[3]
    )
    return jnp.concatenate([vel, spin])


def from_spin_basis(state: Array) -> Array:
    r"""Solver spin basis `$\to$` physical 9-component state.

    Inverse of :func:`to_spin_basis`.
    """
    u_plus, u_minus = state[1], state[2]
    c_rr, c_thth, c_rth, c_rz, c_thz, c_zz = spin_to_phys_combos(
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


# ── sPTT scalar root (JAX-free, build-time) ─────────────────────────


def solve_ptt_f(g: np.ndarray) -> np.ndarray:
    r"""Solve `$f^3 - f^2 = g$` for `$f \ge 1$` (Newton, `$g \ge 0$`).

    `$g = 2\epsilon(\mathrm{Wi}\,S)^2$` is the sPTT extensibility term
    of the laminar (pointwise-equilibrium) conformation, with `$S$` the
    local laminar shear; `$f = 1$` for `$\epsilon = 0$` (or zero shear).
    """
    f = np.ones_like(g)
    for _ in range(100):
        num = f**3 - f**2 - g
        den = 3.0 * f**2 - 2.0 * f
        f = f - num / den
    return f


# ── Narrow (banded-storage-fitting) Laplacian BC wall row ───────────


def narrow_abase_wall_row(
    rs: np.ndarray, D1: np.ndarray, fd_order: int, *, inner: bool
) -> np.ndarray:
    r"""One full `$A_{\mathrm{base}}$` wall row using a **narrow** `$D_2$`.

    The regular `$D_2$` boundary row (:func:`dnsjax.fd.build_diff_matrices`)
    spans `$p+2$` points and does not fit banded storage (half-bandwidth
    `$p$`).  The `$\nabla^2 c = 0$` wall BC only needs the Laplacian
    *evaluated at the wall row*, so a `$(p+1)$`-point one-sided `$D_2$`
    stencil (accuracy `$p-1$`, acceptable for an artificial-diffusion BC)
    is used for the wall rows only, giving a row that fits the band
    (columns `$0..p$` at an inner wall, `$N-p-1..N-1$` at an outer one).
    The identical narrow row is used in every backend so all factor the
    same matrix.  `$D_1$` already fits (`$p+1$`-point), so only `$D_2$`
    is narrowed.

    Returns the full-length `$(N_r,)$` row
    `$A_{\mathrm{base}} = D_2 + (1/r) D_1$` at the inner wall
    (*inner*) or the outer one.  The annulus takes both; the pipe takes
    only the outer (its axis is closed by parity, not by a boundary
    row).  One-sided at a wall, the `$D_2$` stencil never crosses the
    pipe axis, so the row is parity-independent there.
    """
    N = len(rs)
    p = fd_order
    row = np.zeros(N)
    if inner:
        # Inner wall (row 0): narrow one-sided D2 on nodes 0..p.
        w = fornberg_weights(rs[0], rs[0 : p + 1], 2)[:, 2]
        row[0 : p + 1] = w + (1.0 / rs[0]) * D1[0, 0 : p + 1]
    else:
        # Outer wall (row N-1): narrow one-sided D2 on nodes N-p-1..N-1.
        w = fornberg_weights(rs[-1], rs[N - p - 1 :], 2)[:, 2]
        row[N - p - 1 :] = w + (1.0 / rs[-1]) * D1[-1, N - p - 1 :]
    return row


# ── Norms ───────────────────────────────────────────────────────────


def get_norm2_conformation(
    c_phys: Array, k_metric: Array, y_weights: Array
) -> Array:
    r"""Volume-averaged Frobenius norm `$\langle \|c\|_F^2 \rangle$` from
    the **physical** components `$(c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$` (off-diagonal weights 2) -- the
    diagnostic form, applied outside the solver.  Its solver-basis
    counterpart is the spin weighting :data:`C_FROB_SQRT_SPIN` that the
    correctors' ``_norm`` applies."""
    w = jnp.asarray(C_FROB_SQRT, dtype=c_phys.real.dtype).reshape(
        N_TENSOR, 1, 1, 1
    )
    return get_norm2(c_phys * w, k_metric, y_weights)


# ── Spectral tensor divergence (the curvature terms) ────────────────


def div_c_assemble(
    dr: Array,
    c_rr: Array,
    c_thth: Array,
    c_rth: Array,
    c_rz: Array,
    c_thz: Array,
    c_zz: Array,
    im: Array,
    ikz: Array,
    inv_r: Array,
) -> tuple[Array, Array, Array]:
    r"""Assemble `$\nabla\cdot c$` from its radial derivatives.

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

    *dr* holds `$(\partial_r c_{rr}, \partial_r c_{r\theta},
    \partial_r c_{rz})$` -- the only geometry-dependent input, since
    the annulus differentiates with a plain `$D_1$` and the pipe with
    the parity-reduced pair.  Everything else (the `$1/r$` curvature
    terms) is common, and is single-sourced here so the two geometries
    cannot drift.
    """
    div_r = dr[0] + im * inv_r * c_rth + ikz * c_rz + inv_r * (c_rr - c_thth)
    div_th = dr[1] + im * inv_r * c_thth + ikz * c_thz + inv_r * 2 * c_rth
    div_z = dr[2] + im * inv_r * c_thz + ikz * c_zz + inv_r * c_rz
    return div_r, div_th, div_z


# ── Pointwise physical-space RHS kernel ─────────────────────────────


def pointwise_rhs(
    phys: Array, inv_r_padded: Array, wi: float, eps: float
) -> tuple[Array, Array, Array]:
    r"""The whole pointwise (physical-space) stage of the sPTT RHS.

    Pure, coordinate-level arithmetic on the 36 transformed fields --
    identical in every cylindrical-coordinate geometry, so both
    viscoelastic flows call this one kernel.

    *phys* is the batch produced by the inverse transform, in the
    fixed order

    ``[u_z, u_r, u_theta]`` (3),
    ``[L_rr, L_rth, L_rz, L_thr, L_thth, L_thz, L_zr, L_zth, L_zz]``
    (9, the velocity gradient `$L_{ij} = \partial_i u_j$`),
    ``[c_rr, c_thth, c_rth, c_rz, c_thz, c_zz]`` (6),
    then that same tensor order repeated for
    `$\partial_r c$`, `$\partial_\theta c$`, `$\partial_z c$`
    (18) -- 36 fields.  *inv_r_padded* is `$1/r$` on the padded
    physical grid.

    Returns ``(out_phys, om_phys, trc)``:

    - *out_phys* ``(9, ...)``, the nonlinear terms in the physical
      order ``[NL_z, NL_r, NL_th, N_rr, N_thth, N_rth, N_rz, N_thz,
      N_zz]``: the rotational velocity term
      `$\mathbf{u}\times\boldsymbol\omega$`, and the conformation
      transport
      `$-\mathbf{u}\cdot\nabla\mathbf{c}
      + (\nabla u)^{\!\top}c + c\,\nabla u
      - \tfrac{f(\mathrm{tr}\,c)}{\mathrm{Wi}}(c-\mathbb{I})$` with
      `$f = 1 - 3\epsilon + \epsilon\,\mathrm{tr}\,c$`.  The tensor
      advection carries the `$u_\theta/r$` basis-rotation
      (Christoffel) corrections, which is the only place the
      curvilinear frame enters pointwise.
    - *om_phys* `$(\omega_z, \omega_r, \omega_\theta)$`, taken free
      from the antisymmetric part of `$L$` (no separate curl).
    - *trc* `$\mathrm{tr}\,c$`, for the ``TrC_max`` measurement.

    The viscous / diffusive Laplacians, the driving, and the
    polymer-stress divergence are **not** here: they are FFT-free and
    each geometry adds them spectrally around this call.
    """
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

    uth_over_r = uth_p * inv_r_padded

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
    om_phys = jnp.array([om_z, om_r, om_th])
    return out_phys, om_phys, trc


# ── FFT-free linear / mean conformation coupling (CN/AB2 scheme) ────


def conformation_coupling_core(
    combos: tuple[Array, Array, Array, Array, Array, Array],
    ident: Array,
    eps: float,
    wi: float,
    mean: tuple[Array, Array, Array, Array, Array, Array, Array] | None,
) -> Array:
    r"""The CN/AB2-implicit part of the conformation RHS, 6 spin
    components -- the spectral algebra, free of any radial operator.

    - the **linear relaxation** `$-(1-3\epsilon)(c-\mathbb{I})/
      \mathrm{Wi}$` -- **always** folded in (a linear reaction term,
      like the viscous Laplacian);
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
      (default on), exactly as the velocity `$L_{mf}$` is.

    Written on the physical tensor *combos* (identical algebra to
    :func:`pointwise_rhs`, restricted to `$\bar{\mathbf{u}}$`), so the
    explicit remainder ``get_rhs - _l_bf`` is exactly the
    fluctuation-fluctuation advection / stretching and the nonlinear
    relaxation part.  No Fourier transform (mean profile x spectral
    field is a pointwise-in-`$r$` product).

    *ident* is the spectral image of `$\mathbb{I}$` (1 at the mean
    mode, 0 elsewhere) -- in physical space :func:`pointwise_rhs` gets
    that for free, but here the arithmetic is spectral.

    *mean* is ``None`` when the mean coupling is off (the caller then
    skips building it at all), or the tuple ``(mean_uz, mean_uth,
    dr_mean_uz, dr_mean_uth, im, ikz, inv_r)`` with the profiles
    already broadcast to ``(N_r, 1, 1)``: taking `$\partial_r$` of a
    mean profile is the one geometry-dependent step, so the caller
    owns it.  The moving-frame term is likewise the caller's (it needs
    `$k_z$` against the solver-basis state).
    """
    cs_rr, cs_thth, cs_rth, cs_rz, cs_thz, cs_zz = combos

    faclin = (1.0 - 3.0 * eps) / wi
    Nc_rr = -(cs_rr - ident) * faclin
    Nc_thth = -(cs_thth - ident) * faclin
    Nc_zz = -(cs_zz - ident) * faclin
    Nc_rth = -cs_rth * faclin
    Nc_rz = -cs_rz * faclin
    Nc_thz = -cs_thz * faclin

    if mean is not None:
        mean_uz, mean_uth, dr_mean_uz, dr_mean_uth, im, ikz, inv_r = mean
        uth_over_r = mean_uth * inv_r
        Lbar_rth = dr_mean_uth  # d_r u_theta
        Lbar_rz = dr_mean_uz  # d_r u_z
        Lbar_thr = -uth_over_r  # -u_theta / r

        # Mean advection (u_r == 0): (u_theta/r) d_theta + u_z d_z, with
        # the same Christoffel corrections as ``pointwise_rhs``.
        def _madv(x: Array) -> Array:
            return uth_over_r * (im * x) + mean_uz * (ikz * x)

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

    return phys_combos_to_spin(Nc_rr, Nc_thth, Nc_rth, Nc_rz, Nc_thz, Nc_zz)


# ── Corrector convergence norm ──────────────────────────────────────


def combined_norm(
    correction: Array, k_metric: Array, y_weights: Array
) -> Array:
    r"""Combined L2 convergence norm, `$\sqrt{\|u\|^2 + \|c\|_F^2}$`.

    Corrections live in the solver spin basis, so the `$u_\pm$` pair
    carries the 1/2 weight and the tensor slots the spin Frobenius
    weights (:data:`C_FROB_SQRT_SPIN`) -- the same physical scalar the
    diagnostic norms report for a physical-basis array.
    """
    pm2 = get_norm2(correction[1:3], k_metric, y_weights)
    uz2 = get_norm2(correction[:1], k_metric, y_weights)
    w = jnp.asarray(C_FROB_SQRT_SPIN, dtype=correction.real.dtype).reshape(
        N_TENSOR, 1, 1, 1
    )
    c2 = get_norm2(correction[3:] * w, k_metric, y_weights)
    return jnp.sqrt(uz2 + pm2 / 2 + c2)
