r"""Time stepping shared by the cylindrical (pipe) geometries.

The straight pipe (:mod:`~dnsjax.geometries.wall_bounded.cylindrical`)
and the curved, zero-torsion pipe
(:mod:`~dnsjax.geometries.wall_bounded.cylindrical_curved`) share one
set of stepping functions: the pseudo-spectral RHS, the FFT-free
base-flow coupling, the `$u_r$`-`$\omega_r$` influence-matrix pass, the
predictor / corrector / norm and the stepper factory.  They are written
once here and parametrised by the flow dataclass, whose operators,
grids, parity classes and band families all stay in the geometry
modules.

This module imports **neither** geometry: the type names in the
signatures below exist only under ``TYPE_CHECKING`` (hence
``from __future__ import annotations`` -- ``jax.jit`` runs
``inspect.signature``, which would otherwise raise on them), and the
two singletons the factory needs, ``fourier`` and ``_build_dt_leaves``,
arrive as arguments from each geometry's own thin
``build_*_stepper(flow)``.

The derivations stay with the code they describe: the decoupled
`$u_\pm$` formulation, the parity reduction and the `$1 \times 1$`
influence matrix in the ``cylindrical`` module docstring; the
reconstruction scheme's shared record in
``cartesian._imm_iteration``; the cylindrical algebra in
``annular._imm_iteration_vw``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from jax import Array
from jax import numpy as jnp

from ...measurements import get_cfl
from ...operators import phys_to_spec_2d, spec_to_phys_2d
from ...parameters import derived_params, params
from ...rhs import get_nonlin
from ...sharding import sharding
from ._base import (
    apply_y_matrix,
    base_flow_coupling,
    build_wall_bounded_stepper,
    extract_mean_mode,
    extract_mean_modes,
    from_pm_basis,
    get_norm2,
    to_pm_basis,
)

if TYPE_CHECKING:
    from .cylindrical import CylindricalFlow, Fourier


# ── Mean-mode driving ─────────────────────────────────────


#: ``stats.dat`` column name for the mean-mode driving this geometry
#: applies (:func:`_apply_bulk_correction`).  The sign is the applied
#: **forcing** `$-\partial p'/\partial z$`, positive when it accelerates
#: the flow, carried in the name so it cannot be read as the pressure
#: gradient.
DRIVING_KEY_Z = "-dPdz'"


def mean_driving(state: Array, flow_: CylindricalFlow) -> dict[str, Array]:
    r"""Wall-shear **inference** of the driving, from a state alone.

    Area-averaging the mean-mode axial momentum over the disc, with
    `$\int_0^1 r^{-1}(r\,\bar{u}_z')'\,r\,dr = [r\,\bar{u}_z']_0^1$`
    and ``volume_fac`` `$= \int_0^1 r\,dr = 1/2$`:

    .. math::
        \frac{d U_{b,z}'}{dt} = -\Pi_z + 2\,\nu\,\tau_z ,

    with `$\Pi_z$` the `$(0,0)$` mode of `$\partial p/\partial z$`, so
    the applied force is `$-\Pi_z$` (the codebase convention;
    :mod:`dnsjax.ic.mean_mode` derives it).  Holding the bulk fixed
    therefore applies exactly `$-\Pi_z = -2\nu\tau_z$` -- the same
    number :func:`_apply_bulk_correction` applies, up to the time
    discretization, under the same key and sign.

    Used for the ``t = t0`` ``stats.dat`` row, which has no step behind
    it (:mod:`dnsjax.__main__`); every other row reports the value the
    corrector actually applied.
    """
    if params.phys.driving != "constant_bulk_velocity":
        return {}
    mean_uz = extract_mean_mode(state)[0].real
    tau_z = jnp.dot(flow_.D1_wall.ravel(), mean_uz)
    return {flow_.driving_key: -2 * tau_z / params.phys.re}


def _apply_bulk_correction(
    uz_new: Array,
    uz_src: Array,
    mean_mask: Array,
    flow_: CylindricalFlow,
) -> tuple[Array, dict[str, Array]]:
    r"""Constant-bulk-velocity enforcement, shared by both IMM schemes.

    Adds a uniform body force `$-\Pi_z$` to the mean-mode `$u_z$`
    Helmholtz RHS so the perturbation bulk axial velocity is zero,
    in its equivalent post-solve form `$u_z \mathrel{+}= -\Pi_z\,h$`
    with `$h$` the response of
    :meth:`~.cylindrical.CylindricalFlow._precompute_bulk_response` and
    `$-\Pi_z = -U_{b,\mathrm{pert}} / H_{\mathrm{bulk}}$` (``force_z``
    below holds that force, not `$\Pi_z$`).  Like every
    mean-plane write it is confined to `$k^2 = 0$`, the one plane the
    reconstruction never touches; ``mean_mask`` is the write mask, so
    no other mode (padding included) receives it.

    *uz_src* is where the bulk is **read** and *uz_new* what the
    correction is **added to**.  They differ only on the legacy
    primitive path, whose `$u_z$` carries an extra `$-ik_z q_z$` term
    that vanishes at the mean mode: reading the bulk from the
    uncorrected ``uz_arb`` there lets the IMM and bulk corrections fuse
    into one expression.

    Returns the corrected field and the applied `$-\Pi_z$` as the
    corrector's *aux* diagnostics -- the correction's own scalar
    prefactor, so what is reported cannot drift from what is applied.
    Empty, and a trace-time no-op, under any other driving.
    """
    if params.phys.driving != "constant_bulk_velocity":
        return uz_new, {}
    bulk_uz = flow_.bulk_deficit(uz_src)
    force_z = -bulk_uz * flow_.H_bulk_inv  # the applied ``-Pi_z``
    return (
        uz_new
        + jnp.where(
            mean_mask, force_z * flow_.h_bulk_response[:, None, None], 0.0
        ),
        {flow_.driving_key: force_z},
    )


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
        flow_.rhs_extra_spec_fn(fourier_),
        flow_.to_physical if flow_.is_curved else None,
        flow_.metric_rhs if flow_.is_curved else None,
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
            flow_.cfl_names,
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


def _straight_divergence(
    state_rthz: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> Array:
    r"""Straight cylindrical divergence of the carried state.

    .. math::
        \nabla_0\cdot\mathbf{w} = \frac{1}{r}\partial_r(r w_r)
        + \frac{im}{r} w_\theta + i k_s w_s

    Zero for an incompressible straight pipe; on the curved pipe it is
    the `$O(\kappa)$` field the metric leaves behind, and the only
    place curvature reaches the reconstruction
    (:meth:`~.cylindrical_curved.CurvedCylindricalFlow.divergence_defect`
    computes the same quantity from the *constraint*, which is what the
    reconstruction must be driven by).
    """
    psv = 1 - fourier_.m_is_even * 2  # (-1)^{m+1}: the u_r parity class
    inv_r = flow_.inv_r[:, None, None]
    d1_ur = _parity_y_matvec(flow_.D1_pos, flow_.D1_ghost, state_rthz[1], psv)
    return (
        d1_ur
        + inv_r * state_rthz[1]
        + 1j * fourier_.m * inv_r * state_rthz[2]
        + 1j * fourier_.kz * state_rthz[0]
    )


def _grad_pm(
    scalar: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> Array:
    r"""Transverse spin pair of a scalar gradient,
    `$(\nabla_0 g)_\pm = \partial_r g \mp (m/r)\,g$`, stacked on
    axis 1 like the quad's spin pairs.  The scalar carries the
    `$(-1)^m$` parity class.
    """
    psp = fourier_.m_is_even * 2 - 1
    d1_g = _parity_y_matvec(flow_.D1_pos, flow_.D1_ghost, scalar, psp)
    m_over_r = fourier_.m * flow_.inv_r[:, None, None]
    return jnp.stack(
        [d1_g - m_over_r * scalar, d1_g + m_over_r * scalar], axis=1
    )


# ── IMM iteration (1x1) ─────────────────────────────────────────


def _imm_iteration_vw(
    velocity_n: Array,
    velocity_j: Array,
    nonlin_n: Array,
    nonlin_j: Array,
    fourier_: Fourier,
    flow_: CylindricalFlow,
) -> tuple[Array, Array, dict[str, Array]]:
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
    placed on the `$-$` family's mean plane
    (:func:`~.cylindrical._vw_spin_groups`), which is why
    `$u_{z,00}$` rides `$\Phi_-$`.  `$\Phi_+$` and
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
    if flow_.is_curved:
        # The evolved `$\Phi = -\nabla_0\times\nabla_0\times w$` is
        # the *solenoidal* Laplacian, which is what makes its evolution
        # equation pressure-free for any explicit RHS.  The straight
        # vector Laplacian above overshoots it by
        # `$\nabla_0(\nabla_0\cdot w)$` -- identically zero on the
        # straight pipe, `$O(\kappa)$` here, and exactly evaluable at
        # `$t^n$` from the accepted state.
        phi_pm = phi_pm - _grad_pm(
            _straight_divergence(state_n, fourier_, flow_), fourier_, flow_
        )
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

    # The curved pipe's continuity rows below carry an O(kappa)
    # defect, evaluated on the corrector ITERATE so the fixed point
    # places it at t^{n+1} (the wall-data record above is why it may
    # never be lagged across a step).  ``None`` on the straight pipe
    # removes every dependent term at trace time.
    defect = flow_.divergence_defect(velocity_j, fourier_)

    # Stage 5: exact recovery of u_r.  Lk_op holds L_v,mod here,
    # with a Dirichlet identity wall row; phi_arb and omega_new both
    # vanish at the wall, so u_r|wall = 0 exactly.
    det = kz2 + fourier_.m2 * inv_r2
    inv_det = 1.0 / jnp.where(mean_mask, 1.0, det)
    om_shift = 2.0 * m * fourier_.kz * inv_r2 * inv_det
    lv_rhs = phi_arb - om_shift * omega_new
    if defect is not None:
        # L_v,mod was derived by eliminating u_theta through continuity
        # and Phi through the vector Laplacian, so a nonzero straight
        # divergence enters twice: once as grad(div w) (stage 2's
        # partner) and once through the reconstruction's chi.
        # Zeroed on the wall row, like every other source reaching
        # this solve: ``Lk_op`` carries a Dirichlet identity row there,
        # so whatever the RHS holds *is* `$u_r|_{\mathrm{wall}}$`.  The
        # defect vanishes at the wall at the fixed point (no-slip makes
        # the curved constraint read `$h_w(\partial_r w_r)|_w = 0$`),
        # but the iterate's does not, and leaving it in imposes a
        # nonzero wall velocity -- measured: the corrector stops
        # converging above `$\kappa \approx 0.02$`, and diverges
        # outright at 0.13.  Zeroing it costs nothing at the fixed
        # point and is what keeps `$u_r|_w = 0$` exact throughout.
        psp = fourier_.m_is_even * 2 - 1
        lv_rhs = lv_rhs + (
            _parity_y_matvec(flow_.D1_pos, flow_.D1_ghost, defect, psp)
            + (2.0 * fourier_.m2 * inv_r * inv_r2 * inv_det) * defect
        ).at[-1].set(0.0)
    ur_arb = flow_.Lk_op.solve(lv_rhs)

    # Stage 6: influence matrix (1x1) -- the free Phi wall value that
    # makes (D1 u_r)|wall = 0.
    d_wall = jnp.einsum("j, jmz -> mz", flow_.D1_wall.ravel(), ur_arb)
    ur_new = ur_arb + (-flow_.M_inv * d_wall)[None] * flow_.ur_1

    # Stage 7: per-point reconstruction of (u_z, u_theta) from the
    # continuity row and the omega_r definition.
    d1_ur = _parity_y_matvec(flow_.D1_pos, flow_.D1_ghost, ur_new, psv)
    chi = -(d1_ur + inv_r * ur_new)
    if defect is not None:
        chi = chi + defect
    b_th = im * inv_r
    uz_new = (-ikz * chi - b_th * omega_new) * inv_det
    ut_new = (-b_th * chi + ikz * omega_new) * inv_det

    # Stage 8: unpack the mean plane (which inv_det left at zero) and
    # zero the mean-mode u_r, which continuity forces.
    uz_new = jnp.where(mean_mask, phi_arb_pm[:, 1], uz_new)
    ut_new = jnp.where(mean_mask, om_pm[:, 0], ut_new)
    mean_ur = flow_.mean_radial(defect)
    ur_new = jnp.where(mean_mask, 0.0 if mean_ur is None else mean_ur, ur_new)

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
    flow_: CylindricalFlow,
) -> tuple[Array, Array, dict[str, Array]]:
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
    (:func:`~.cylindrical.build_parity_reduced_matrices`).
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
    flow_: CylindricalFlow,
) -> tuple[Array, Array, dict[str, Array]]:
    """Crank-Nicolson corrector via the cylindrical IMM.

    Third return: the corrector-side *aux* diagnostics, here the
    applied mean-mode driving (:func:`_apply_bulk_correction`).
    """
    prediction_state_new, correction, aux = _imm_iteration(
        state_prev,
        prediction_state,
        rhs_prev,
        rhs_next,
        fourier_,
        flow_,
    )
    return prediction_state_new, correction, aux


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
    scalar :func:`~.cylindrical.get_norm2_cyl` reports for a
    physical-basis array.
    """
    pm2 = get_norm2(correction[1:], fourier_.k_metric, flow_.y_weights)
    uz2 = get_norm2(correction[:1], fourier_.k_metric, flow_.y_weights)
    return jnp.sqrt(uz2 + pm2 / 2)


# ── Stepper factory ─────────────────────────────────────────────


def build_stepper(
    flow: CylindricalFlow,
    fourier_: Fourier,
    dt_leaves_fn: Callable[..., dict],
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
    """Build time-stepping functions for a cylindrical flow.

    Called through each geometry module's own thin
    ``build_*_stepper(flow)``, which binds that geometry's ``fourier``
    singleton and ``_build_dt_leaves``.

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
        fourier_,
        flow,
        _get_rhs_measured,
        _l_bf,
        dt_leaves_fn=dt_leaves_fn,
    )
