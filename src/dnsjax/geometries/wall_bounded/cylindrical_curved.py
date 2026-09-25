r"""Curved (toroidal, zero-torsion) pipe geometry.

A circular pipe of radius `$a$` whose centreline is a circle of radius
`$R_c$`: constant curvature `$\kappa = a/R_c$` (``geo.curvature``) and
**zero torsion**.  Everything here is the straight pipe's
:class:`~dnsjax.geometries.wall_bounded.cylindrical.CylindricalFlow`
plus a metric; the grids, the parity reduction, the band families, the
per-mode operators, the influence matrix and every stepping function
are inherited unchanged, and this module is the curved half of the
adapter surface they are written against.

Coordinates and orientation
---------------------------
Orthogonal coordinates `$(r, \theta, s)$` with scale factors
`$(1, r, h)$`,

.. math::
    h(r, \theta) = 1 + \kappa r \cos\theta ,

where `$r \in (0, 1]$` is the cross-section radius, `$\theta$` the
poloidal angle measured from the **outer** wall (away from the centre
of curvature), and `$s$` the arclength along the **centreline** (so
`$s = R_c \Theta$` for the toroidal angle `$\Theta$`).  The triad is
right-handed, `$\hat e_r \times \hat e_\theta = \hat e_s$`, which places
`$\hat e_\theta|_{\theta = 0}$` along `$-\hat e_Z$`; the flow is
invariant under `$\theta \to -\theta$`, so the choice only labels which
Dean vortex is which, but the curl below assumes it.

Every metric coefficient is **independent of `$s$`**, so `$\partial_s
\to i k_s$` stays exact and *any* streamwise period is geometrically
consistent -- a sub-torus box is as legitimate as a short straight-pipe
box, and the complete torus is `$L_s = 2\pi/\kappa$`.  The volume
element is `$r h\,dr\,d\theta\,ds$` (so the torus volume is `$\pi L_s$`,
the straight pipe's), while the cross-section area element `$r\,dr\,
d\theta$` carries no `$h$` -- the mass flux is metric-free even though
no volume integral is.  Curvature breaks the discrete `$\theta \to
\theta + 2\pi/m_0$` symmetry, so the azimuthal wedge is not available
here.

The carried state: `$w = (h\,u_s,\, u_r,\, u_\theta)$`
-------------------------------------------------------
The solver carries the **metric-weighted** streamwise component (then,
as always, in the decoupled `$u_\pm$` basis).  That one choice is what
keeps the whole influence-matrix pass the straight pipe's, through four
identities -- each verified numerically against independently coded
toroidal operators:

1. **The curl is already right.**  With `$\Omega \equiv \nabla_0 \times
   w$` the *straight* cylindrical curl of the carried state,

   .. math::
       \Omega = (h\,\omega_r,\; h\,\omega_\theta,\; \omega_s),
       \qquad \boldsymbol{\omega} = \nabla_c \times \mathbf{u},

   so ``_curl_fn`` is inherited verbatim and the physical vorticity is
   one pointwise division by `$h$`, done in physical space where it is
   free (:meth:`CurvedCylindricalFlow.to_physical`).

2. **The pressure is an exact straight gradient.**  `$(\nabla_c
   \Pi)_s = h^{-1}\partial_s \Pi$`, so multiplying the `$s$`-momentum
   equation by `$h$` -- which is exactly what carrying `$h u_s$` does --
   leaves `$\partial_t w = -\nabla_0\Pi + N$`.  Hence
   `$-\nabla_0\times\nabla_0\times$` annihilates the pressure exactly,
   and the two evolved quantities of the reconstruction scheme,
   `$\Phi = -\nabla_0\times\nabla_0\times w$` and `$\omega = \nabla_0
   \times w$`, satisfy **the straight pipe's own equations**

   .. math::
       \partial_t \Phi = \nu \Delta_0 \Phi
                       - \nabla_0\times\nabla_0\times N , \qquad
       \partial_t \omega = \nu \Delta_0 \omega + \nabla_0 \times N

   for *any* explicit `$N$` -- no solenoidality and no metric
   assumption enters, both following from vector identities alone.  So
   the four Helmholtz solves, their operators, the spin quad, the
   parity classes, the `$L_{v,\mathrm{mod}}$` recovery and the
   `$1\times1$` influence matrix are used unmodified, and **no operator
   couples azimuthal modes**.

3. **The viscous curvature term is pointwise.**

   .. math::
       M(-\nabla_c\times\boldsymbol{\omega}) = \Delta_0 w
       + \big[P\cdot\Omega + Q\cdot\partial_s\Omega\big]
       - \nabla_0(\nabla_0\cdot w) , \qquad M = \mathrm{diag}(1,1,h)

   with `$P$`, `$Q$` pointwise functions of `$(r,\theta)$`
   (:meth:`CurvedCylindricalFlow.metric_rhs`).  The remainder needs no
   new derivative operator at all: `$\Omega$` is already in physical
   space for the cross product, `$\partial_s\Omega$` rides the same
   batched transform, and the trailing gradient is a pure gradient,
   absorbed by the pressure and annihilated by the scheme's own
   conservative double curl.

4. **Incompressibility is polynomial in `$h$`.**  `$\nabla_c\cdot
   \mathbf{u} = 0$` reads

   .. math::
       h\,\partial_r(r h\,w_r) + h\,\partial_\theta(h\,w_\theta)
       + r\,\partial_s w_s = 0 ,

   so the straight divergence `$\nabla_0\cdot w$` -- zero on the
   straight pipe -- is here an `$O(\kappa)$` field
   (:meth:`CurvedCylindricalFlow.divergence_defect`) needing only
   multiplication by `$\chi = r\cos\theta$`, an `$m \pm 1$` shift.  It
   is the **only** place curvature reaches the implicit pass, entering
   three algebraic terms (the `$\Phi$` definition, the
   `$L_{v,\mathrm{mod}}$` recovery and the reconstruction's `$\chi$`)
   and the `$(0,0)$` radial velocity, always on the corrector
   **iterate** -- never across a time step, for the reason
   ``_cylindrical_stepping._imm_iteration_vw`` records.

That iteration is what pays for keeping every operator: continuity
holds at the corrector's **fixed point**, so its residual tracks
``step.corrector_tolerance`` rather than machine epsilon, and the
corrector needs more passes than the straight pipe's.  Both are
measured (``tests/test_curved_pipe.py``).  At `$\kappa = 0.037$`,
`$n_r = 24$`, the relative toroidal divergence of a stepped random
state is `$7\times10^{-7}$`, `$2.7\times10^{-9}$` and
`$9\times10^{-13}$` at tolerances `$10^{-6}$`, `$10^{-9}$` and
`$10^{-12}$`, reached in 1, 3 and 6 extra corrector passes -- a
contraction of roughly `$\kappa$` per pass, as the `$O(\kappa)$` size
of the lagged terms predicts.  At `$\kappa = 0.13$` the same decades
cost more than the default cap of 10, which is a statement about a
tolerance seven decades tighter than the default `$10^{-5}$`, not
about a production run.

Driving
-------
In a torus "constant streamwise pressure gradient" can only mean
constant `$dP/d\Theta$` -- the single-valued choice -- whose physical
body force is `$f_s = G/h$`.  In the carried variable that is
`$M(f)_s = G$`, a uniform constant, so the driving is the straight
pipe's `$-\Pi = 4/\mathrm{Re}$` and `$\kappa \to 0$` reproduces
``pipe`` exactly.  A *uniform* force on `$u_s$` would be a different,
non-potential driving.  Under ``constant_bulk_velocity`` the same
uniform force shape is scaled each step to hold the **mass flux**
`$Q = \int u_s\,r\,dr\,d\theta$`, which in the carried variable needs
the exact Fourier coefficients of `$1/h$`
(:meth:`CurvedCylindricalFlow.bulk_deficit`).

Metric-weighted integrals
-------------------------
`$h$` has exactly three azimuthal harmonics, so every `$r h$`-weighted
integral needs only `$m = 0, \pm 1$`:

.. math::
    \int f\,r h\,dr\,d\theta = 2\pi \int r
    \Big[\hat f_0 + \tfrac{\kappa r}{2}(\hat f_1 + \hat f_{-1})\Big] dr ,

exact, spectral, and parity-correct (`$\hat f_{\pm1}$` and the extra
`$r$` land back in the even class).

Reference: the toroidal equation set this reproduces is Webster &
Humphrey, *Phys. Fluids* **9**, 407 (1997), Eqs. (2)-(5); their
primitive-variable convective and viscous terms are the rotational
form's `$\nabla(|u|^2/2) - u\times\omega$` and `$\nabla(\nabla\cdot u)
- \nabla\times\nabla\times u$` in this metric, so the curvature terms
are never transcribed here -- they follow from the operators
(``tests/test_curved_pipe.py`` pins that equivalence).
"""

from dataclasses import dataclass, field
from typing import ClassVar

import jax
import numpy as np
from jax import Array, lax, shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ...harmonics import complex_harmonics, real_harmonics
from ...operators import pad_harmonics
from ...parameters import derived_params, padded_res, params
from ...sharding import register_dataclass_pytree, sharding
from ._base import extract_mean_mode, from_pm_basis, pad_base_flow
from ._cylindrical_stepping import _parity_y_matvec, build_stepper
from .cylindrical import CylindricalFlow, Fourier, _build_dt_leaves, fourier

#: Per-direction CFL column names: the streamwise direction is the
#: centreline arclength `$s$`, not a straight axis.
CFL_NAMES_S: tuple[str, str, str] = ("CFL_s", "CFL_r", "CFL_th")

#: ``stats.dat`` column name for the applied mean-mode driving, with
#: the same sign convention as every other geometry's: the applied
#: **forcing** `$-\partial p/\partial s$`, positive when accelerating.
DRIVING_KEY_S: str = "-dPds'"

#: Dimensionless curvature `$\kappa = a/R_c$`, captured at import like
#: every other resolved parameter the geometry is built from.
KAPPA: float = params.geo.curvature

#: Spectral azimuthal padding slot count, needed by the `$m \pm 1$`
#: shift below (a plain ``int``, not a field: it must be read while
#: tracing, and a field would be a pytree leaf).
_M_PAD: int = sharding.nz_spec_pad

#: Applied mean-mode forcing `$-\Pi$`, uniform in the carried
#: variable and equal to the straight pipe's `$4/\mathrm{Re}$` -- zero
#: under ``constant_bulk_velocity``, where the bulk correction supplies
#: the whole driving instead.  A module constant, not a field: it is
#: read *while tracing* (``if FORCE_S == 0.0``), and a float field on a
#: registered dataclass is a pytree **leaf**, hence a tracer.
FORCE_S: float = (
    0.0
    if params.phys.driving == "constant_bulk_velocity"
    else 4.0 / params.phys.re
)

#: Target area-averaged streamwise velocity under
#: ``constant_bulk_velocity``: the straight pipe's laminar
#: `$U_b = 2\int_0^1 (1 - r^2) r\,dr = 1/2$`, so the two flows are
#: driven to the same mass flux at the same ``phys.re``.
BULK_TARGET: float = 0.5


def _inv_h_harmonics(rs: np.ndarray, m_vals: np.ndarray) -> np.ndarray:
    r"""Exact Fourier coefficients `$\widehat{(1/h)}_m(r)$`.

    .. math::
        \frac{1}{1 + \epsilon\cos\theta}
        = \frac{1}{\sqrt{1-\epsilon^2}}
          \Big[1 + 2\sum_{n\ge1} q^n \cos n\theta\Big], \qquad
        q = \frac{-\epsilon}{1 + \sqrt{1-\epsilon^2}},

    with `$\epsilon = \kappa r$`, so the complex coefficients are
    `$c_m = q^{|m|}/\sqrt{1-\epsilon^2}$` -- real, even in `$m$`, and
    `$O(r^{|m|})$` at the axis, hence in the `$(-1)^m$` parity class
    like the field they weight.  Exact to machine precision (checked
    against an FFT of `$1/h$`), so the flux read below is exact rather
    than truncated.

    Returns shape ``(len(rs), len(m_vals))``.
    """
    eps = KAPPA * rs[:, None]
    root = np.sqrt(1.0 - eps**2)
    q = -eps / (1.0 + root)
    return q ** np.abs(m_vals)[None, :] / root


@register_dataclass_pytree
@dataclass
class CurvedCylindricalFlow(CylindricalFlow):
    r"""Straight-pipe flow data plus the toroidal metric.

    Everything the solver needs beyond
    :class:`~dnsjax.geometries.wall_bounded.cylindrical.CylindricalFlow`
    is precomputed here: the physical-space metric factors the RHS
    multiplies by, the `$m\pm1$` shift data behind
    `$\chi = r\cos\theta$`, the exact `$1/h$` harmonics the flux read
    contracts with, and the mean-plane radial operator.  All of them
    are `$\Delta t$`-independent, so the adaptive-`$dt$` rebuild does
    not touch them.
    """

    is_curved: ClassVar[bool] = True
    cfl_names: ClassVar[tuple[str, str, str]] = CFL_NAMES_S
    #: The streamwise direction is the centreline arclength.
    driving_key: ClassVar[str] = DRIVING_KEY_S

    #: Physical-space `$(1, n_y, n_\theta, 1)$` metric factors, on the
    #: **padded** grid the RHS transforms onto.
    h_phys: Array = field(init=False)
    inv_h_phys: Array = field(init=False)
    kap_sin_h: Array = field(init=False)  # `$\kappa\sin\theta/h$`
    kap_cos_h: Array = field(init=False)  # `$\kappa\cos\theta/h$`
    inv_h2_m1: Array = field(init=False)  # `$1/h^2 - 1$`
    #: `$r/2$`, the radial factor of one `$\chi$` multiply.
    half_rs: Array = field(init=False)
    #: `$m\pm1$` shift masks and the two wrap selectors (see
    #: :meth:`chi_mul`).
    m_up_mask: Array = field(init=False)
    m_dn_mask: Array = field(init=False)
    m_first: Array = field(init=False)
    m_last: Array = field(init=False)
    #: `$2\,w_j\,c_m(r_j)$`, the exact mass-flux quadrature.
    flux_weights: Array = field(init=False)
    flux_weights_mean: Array = field(init=False)
    #: One-hot on the `$k_s = 0$` plane the flux lives on.
    kz0_mask: Array = field(init=False)
    #: Inverse of `$(\partial_r + 1/r)$` with a Dirichlet wall row.
    mean_radial_inv: Array = field(init=False)

    def __post_init__(self) -> None:
        """Build the straight-pipe data (which builds the metric)."""
        super().__post_init__()
        # Total-field formulation: no base flow to subtract, and the
        # mean pressure gradient enters the carried variable as a
        # uniform `$-\Pi = 4/\mathrm{Re}$` (the module docstring).
        self.base_flow = jnp.zeros(
            (3, params.res.ny),
            dtype=sharding.float_type,
            out_sharding=sharding.no_shard,
        )[:, :, None, None]
        self.curl_base_flow = jnp.zeros_like(self.base_flow)
        pad_base_flow(self)

    def _build_metric(self) -> None:
        """Precompute every metric quantity the stepping reads."""
        ny, nth = params.res.ny, padded_res.nz_padded
        ny_p = ny + sharding.ny_y_pad
        rs = np.asarray(self.rs)

        # Physical (r, theta) metric.  The y-padding rows are zero
        # fields; giving them r = 0 keeps h = 1 there, so nothing ever
        # divides by a metric factor that is not a metric.
        r_p = np.zeros(ny_p)
        r_p[:ny] = rs
        theta = (params.geo.lz / nth) * np.arange(nth)
        h = 1.0 + KAPPA * r_p[:, None] * np.cos(theta)[None, :]

        def _phys(a: np.ndarray) -> Array:
            return jax.device_put(
                jnp.asarray(a[None, :, :, None], dtype=sharding.float_type),
                sharding.phys_vector_shard,
            )

        self.h_phys = _phys(h)
        self.inv_h_phys = _phys(1.0 / h)
        self.kap_sin_h = _phys(KAPPA * np.sin(theta)[None, :] / h)
        self.kap_cos_h = _phys(KAPPA * np.cos(theta)[None, :] / h)
        self.inv_h2_m1 = _phys(1.0 / h**2 - 1.0)

        # `$\chi = r\cos\theta$` multiply: the (m ± 1) shift.
        self.half_rs = jax.device_put(
            jnp.asarray(0.5 * rs, dtype=sharding.float_type)[:, None, None],
            sharding.no_shard,
        )
        m_vals = np.asarray(
            pad_harmonics(
                complex_harmonics(params.res.nz), params.res.nz, _M_PAD
            )
            * params.geo.m0
        )
        n_true = params.res.nz - 1
        top = params.res.nz // 2 - 1  # index of m = +M
        up = np.ones(m_vals.shape[0])
        dn = np.ones(m_vals.shape[0])
        up[n_true:] = 0.0  # padding slots stay zero
        dn[n_true:] = 0.0
        up[top + 1] = 0.0  # m = -M would alias in from m = +M
        dn[top] = 0.0  # m = +M would alias in from m = -M
        first = np.zeros(m_vals.shape[0])
        last = np.zeros(m_vals.shape[0])
        first[0] = 1.0
        last[n_true - 1] = 1.0

        def _mode(a: np.ndarray) -> Array:
            return jax.device_put(
                jnp.asarray(a[None, :, None], dtype=sharding.float_type),
                P(None, sharding.a0, None),
            )

        self.m_up_mask = _mode(up)
        self.m_dn_mask = _mode(dn)
        self.m_first = _mode(first)
        self.m_last = _mode(last)

        # Exact 1/h harmonics -> the mass-flux quadrature.  y_weights
        # already carry the radial Jacobian, so
        # `$Q/\pi = 2\sum_m \int r\,\hat w_{s,m} c_m\,dr$`.
        c_m = _inv_h_harmonics(rs, m_vals)
        c_m[:, n_true:] = 0.0
        yw = np.asarray(self.y_weights)
        self.flux_weights = jax.device_put(
            jnp.asarray(2.0 * yw[:, None] * c_m, dtype=sharding.float_type)[
                :, :, None
            ],
            P(None, sharding.a0, None),
        )
        self.flux_weights_mean = jax.device_put(
            jnp.asarray(2.0 * yw * c_m[:, 0], dtype=sharding.float_type),
            sharding.no_shard,
        )
        kz_vals = np.asarray(
            pad_harmonics(
                real_harmonics(params.res.nx),
                params.res.nx,
                sharding.nx_spec_pad,
            )
        )
        self.kz0_mask = jax.device_put(
            jnp.asarray(
                (kz_vals == 0).astype(float)[None, None, :],
                dtype=sharding.float_type,
            ),
            P(None, None, sharding.a1),
        )

        # Mean-plane radial operator: continuity at (m, k) = (0, 0) is
        # the first-order ODE `$(\partial_r + 1/r) w_r = g$` with
        # `$w_r(1) = 0$`, discretised with the *same* odd-parity D1 the
        # rest of the pass uses, so the discrete divergence it feeds is
        # the discrete divergence the reconstruction enforces.  Stored
        # inverted: it is applied to a single mode column per corrector
        # iteration, where a dense matvec is free.
        g_rows = self.D1_ghost.shape[0]
        d1_odd = np.asarray(self.D1_pos).copy()
        d1_odd[:g_rows] -= np.asarray(self.D1_ghost)
        A = d1_odd + np.diag(1.0 / rs)
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        self.mean_radial_inv = jax.device_put(
            jnp.asarray(np.linalg.inv(A), dtype=sharding.float_type),
            sharding.no_shard,
        )

    # ── Metric adapter: the curved members ──────────────────────

    def chi_mul(self, field_: Array) -> Array:
        r"""Multiply a spectral field by `$\chi = r\cos\theta$`.

        `$\chi$` is all of `$h - 1$` over `$\kappa$`, and the one
        genuinely new spectral primitive the curved pipe needs:

        .. math::
            \widehat{(\chi f)}_m = \frac{r}{2}
            \big(\hat f_{m-1} + \hat f_{m+1}\big) .

        Three things make it more than a roll.  It must **truncate**
        rather than wrap at `$|m| = M$` -- the outgoing coefficient
        leaves the resolved set and the incoming one does not exist --
        which the two masks do.  The FFT ordering
        `$[0 \ldots M, -M \ldots -1]$` puts the remaining wrap
        (`$m = -1 \to 0$`, and `$m = 0 \to -1$`) at the ends of the
        *true-mode* block, not of the array, so with spectral padding
        present a cyclic roll would carry a padding slot into
        `$m = 0$`; the shift here is therefore non-cyclic and the two
        one-hot planes supply those contributions instead.  And the
        `$m$` axis is the ``np0``-sharded one, where ``jnp.roll`` is
        refused outright and slicing it is the failure mode
        ``snapshot._to_io_layout_core`` documents -- so when it *is*
        sharded the shift runs inside ``shard_map`` with a one-plane
        halo (``ppermute``) and the wrap planes come from ``psum`` over
        masked one-hots.  No index arithmetic ever crosses a shard
        boundary, and the unsharded path computes the same thing
        without the collectives.

        Takes a `$(N_r, N_m, N_k)$` field or a component-leading stack
        of them.  Multiplication by `$\chi$` **preserves the parity
        class**: it shifts `$m$` by one and multiplies by `$r$`, and
        `$(-1)^{m\pm1}$` times the extra sign of `$r$` is `$(-1)^m$`
        again, so every caller keeps the parity it had.
        """
        lead = field_.ndim - 3  # 1 for a component-leading stack
        shape = (1,) * lead + (1, -1, 1)

        def _core(
            f: Array,
            up_mask: Array,
            dn_mask: Array,
            first: Array,
            last: Array,
            sharded: bool,
        ) -> Array:
            up_mask, dn_mask, first, last = (
                x.reshape(shape) for x in (up_mask, dn_mask, first, last)
            )
            if sharded:
                np0 = sharding.np0
                idx = lax.axis_index("np0")
                recv_up = lax.ppermute(
                    f[..., -1:, :],
                    "np0",
                    [(i, (i + 1) % np0) for i in range(np0)],
                )
                recv_dn = lax.ppermute(
                    f[..., :1, :],
                    "np0",
                    [(i, (i - 1) % np0) for i in range(np0)],
                )
                recv_up = jnp.where(idx == 0, 0.0, recv_up)
                recv_dn = jnp.where(idx == np0 - 1, 0.0, recv_dn)
            else:
                recv_up = jnp.zeros_like(f[..., :1, :])
                recv_dn = recv_up
            up = jnp.concatenate([recv_up, f[..., :-1, :]], axis=-2)
            dn = jnp.concatenate([f[..., 1:, :], recv_dn], axis=-2)
            plane_last = jnp.sum(f * last, axis=-2, keepdims=True)
            plane_first = jnp.sum(f * first, axis=-2, keepdims=True)
            if sharded:
                plane_last = lax.psum(plane_last, "np0")
                plane_first = lax.psum(plane_first, "np0")
            return up_mask * (up + first * plane_last) + dn_mask * (
                dn + last * plane_first
            )

        masks = (self.m_up_mask, self.m_dn_mask, self.m_first, self.m_last)
        if sharding.a0 is None:
            shifted = _core(field_, *masks, sharded=False)
        else:
            field_spec = P(*((None,) * (1 + lead)), sharding.a0, sharding.a1)
            mode_spec = P(None, sharding.a0, None)
            shifted = shard_map(
                lambda *a: _core(*a, sharded=True),
                mesh=sharding.mesh,
                in_specs=(field_spec,) + (mode_spec,) * 4,
                out_specs=field_spec,
            )(field_, *masks)
        return self.half_rs.reshape((1,) * lead + (-1, 1, 1)) * shifted

    def rhs_extra_spec_fn(self, fourier_: Fourier):
        r"""`$\partial_s\Omega_{r,\theta}$`, riding the RHS transform.

        The viscous curvature remainder needs the streamwise derivative
        of the transverse vorticity pair in physical space.  Sent
        through :func:`~dnsjax.rhs.get_nonlin`'s batch, it costs two
        more fields in an inverse transform that is already happening
        and no spectral operator beyond the diagonal `$ik_s$`.
        """
        ikz = 1j * fourier_.kz

        def _extra(vorticity_spec: Array) -> Array:
            return ikz * vorticity_spec[1:3]

        return _extra

    def to_physical(
        self, velocity_phys: Array, vorticity_phys: Array
    ) -> tuple[Array, Array]:
        r"""Carried components -> the physical triad, by `$1/h$`.

        `$u_s = w_s/h$` and, by identity 1 of the module docstring,
        `$(\omega_r, \omega_\theta) = (\Omega_r, \Omega_\theta)/h$`:
        the carried state's *straight* curl is already the `$h$`-weighted
        toroidal vorticity.  Both divisions are pointwise here and
        impossible in spectral space (`$1/h$` couples every azimuthal
        mode), which is why the RHS is where they happen.
        """
        return (
            velocity_phys.at[0].multiply(self.inv_h_phys[0]),
            vorticity_phys.at[1:].multiply(self.inv_h_phys),
        )

    def metric_rhs(
        self, nonlin_phys: Array, vorticity_phys: Array, extra_phys: Array
    ) -> Array:
        r"""Weight the RHS and add the viscous curvature remainder.

        Identity 3 of the module docstring, componentwise in the
        `$(s, r, \theta)$` slot order, with `$\Omega$` the **raw**
        (`$h$`-weighted) vorticity the transform produced:

        .. math::
            \nu\Big[(\kappa\cos\theta/h)\,\Omega_\theta
                   + (\kappa\sin\theta/h)\,\Omega_r\Big]_s , \quad
            \nu\Big[(\kappa\sin\theta/h)\,\Omega_s
                   + (h^{-2}-1)\,\partial_s\Omega_\theta\Big]_r , \quad
            \nu\Big[(\kappa\cos\theta/h)\,\Omega_s
                   - (h^{-2}-1)\,\partial_s\Omega_r\Big]_\theta .

        The `$s$` slot additionally carries `$M$`'s own `$h$` on the
        Lamb term and the uniform driving `$-\Pi$` (zero under
        ``constant_bulk_velocity``, where the bulk correction supplies
        it instead).  A constant added in physical space *is* the
        `$(0,0)$` mode after a ``norm="forward"`` transform, exactly.
        """
        om_s, om_r, om_t = (
            vorticity_phys[0],
            vorticity_phys[1],
            vorticity_phys[2],
        )
        ds_om_r, ds_om_t = extra_phys[0], extra_phys[1]
        sin_h, cos_h = self.kap_sin_h[0], self.kap_cos_h[0]
        gap = self.inv_h2_m1[0]
        remainder = jnp.stack(
            [
                cos_h * om_t + sin_h * om_r,
                sin_h * om_s + gap * ds_om_t,
                cos_h * om_s - gap * ds_om_r,
            ]
        )
        out = nonlin_phys.at[0].multiply(self.h_phys[0]) + (
            derived_params.nu * remainder
        )
        if FORCE_S == 0.0:
            return out
        return out.at[0].add(FORCE_S)

    def divergence_defect(self, state: Array, fourier_: Fourier) -> Array:
        r"""The straight divergence `$\nabla_0\cdot w$` the metric leaves.

        From identity 4, with `$\chi = r\cos\theta$` and `$h = 1 +
        \kappa\chi$`,

        .. math::
            r\,(\nabla_0\cdot w) = -\kappa\,\big[h\,A + B\big], \qquad
            A = \partial_r(r\chi w_r) + im\,(\chi w_\theta), \quad
            B = \chi\big[\partial_r(r w_r) + im\,w_\theta\big],

        exact in `$\kappa$` (the `$\kappa^2$` term is the `$\kappa\chi$`
        half of `$h A$`) and costing four `$\chi$` multiplies and two
        radial derivatives.  Read on the corrector **iterate**, so the
        fixed point places it at `$t^{n+1}$`.

        Parity: `$\chi$` preserves the class, `$r\,w_r$` crosses into
        the `$(-1)^m$` one and its derivative crosses back, so `$A$`,
        `$B$` and `$rg$` all sit in the `$u_r$` class and `$g$` itself
        in the scalar `$(-1)^m$` class the gradient consumer expects.
        """
        w = from_pm_basis(state)
        psv = 1 - fourier_.m_is_even * 2  # the u_r parity class
        im = 1j * fourier_.m
        r_col = self.rs[:, None, None]

        def _dr(x: Array) -> Array:
            r"""Discrete `$\partial_r(r x)$` in the **scheme's** form.

            The reconstruction builds its continuity row as
            `$\chi = -(D_1 w_r + w_r/r)$`, so the defect has to be
            `$r D_1 x + x$` and not `$D_1(r x)$`.  The two agree
            analytically and differ by the FD truncation error, and it
            is the *discrete* divergence the reconstruction annihilates
            exactly: measured, the other form leaves an
            `$O(\kappa)\times$` truncation continuity residual that
            refinement does not remove, where this one is machine
            epsilon.
            """
            return (
                r_col * _parity_y_matvec(self.D1_pos, self.D1_ghost, x, psv)
                + x
            )

        a_term = _dr(self.chi_mul(w[1])) + im * self.chi_mul(w[2])
        b_term = self.chi_mul(_dr(w[1]) + im * w[2])
        h_a = a_term + KAPPA * self.chi_mul(a_term)
        return (-KAPPA) * self.inv_r[:, None, None] * (h_a + b_term)

    def mean_radial(self, defect: Array) -> Array:
        r"""The `$(m, k_s) = (0, 0)$` radial velocity.

        The reconstruction is singular on the mean plane, so continuity
        fixes `$w_r$` there directly: at `$m = k_s = 0$` it is the
        first-order ODE `$(\partial_r + 1/r)w_r = g$` with
        `$w_r(1) = 0$`.  Unlike the straight pipe -- where `$g = 0$`
        forces `$w_r|_{00} = 0$` identically -- the curved mean radial
        velocity is a nonzero `$O(\kappa)$` profile: it is what keeps
        `$(h u_r)$` mean-free, as the cross-section mass balance
        requires.
        """
        g00 = extract_mean_mode(defect[None])[0]
        return (self.mean_radial_inv @ g00.at[-1].set(0.0))[:, None, None]

    def bulk_deficit(self, uz_src: Array) -> Array:
        r"""Mass-flux deficit against the straight-pipe laminar value.

        The conserved quantity is the flux `$Q = \int u_s\,r\,dr\,
        d\theta$` -- metric-free, because the cross-section area element
        is -- which in the carried variable contracts the whole
        `$k_s = 0$` plane against the exact `$1/h$` harmonics:

        .. math::
            \frac{Q}{\pi} = 2\sum_m \int r\,\hat w_{s,m}(r)\,c_m(r)\,dr .

        The sum runs over the resolved `$m$` only and is **exact** even
        so: the `$\theta$` integral keeps just the `$m = 0$` component
        of the product, and every term of it is resolved.
        """
        return (
            jnp.sum(self.flux_weights * self.kz0_mask * uz_src).real
            - BULK_TARGET
        )

    def bulk_of_mean_profile(self, profile: Array) -> Array:
        r"""Flux contribution of a mean-mode `$w_s$` profile.

        The bulk-correction response is a `$(0,0)$` profile, so only
        `$c_0$` of :func:`_inv_h_harmonics` weights it.
        """
        return jnp.dot(self.flux_weights_mean, profile)


def build_curved_cylindrical_stepper(flow: CurvedCylindricalFlow) -> tuple:
    r"""Build time-stepping functions for a curved-pipe flow.

    Binds the shared cylindrical ``fourier`` singleton and
    ``_build_dt_leaves`` -- both identical to the straight pipe's,
    since curvature changes neither the wavenumber grid nor any
    `$\Delta t$`-dependent operator -- to the shared
    ``_cylindrical_stepping.build_stepper``.
    """
    return build_stepper(flow, fourier, _build_dt_leaves)
