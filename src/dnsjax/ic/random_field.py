r"""Random divergence-free initial-condition generators.

Builds a random divergence-free perturbation of the base flow for any
implemented flow system, returned as a sharded spectral state ready to
time step.  This is the implementation behind ``dnsjax.__main__``'s
in-process random initial-condition start mode (``init.random_field``,
the default when no snapshot is given) -- there is no offline CLI.

Each mode is given the decaying **amplitude** envelope

.. math::
    A = (1 - s)^{|k_x| + |k_z| (+ |k_y|)},

so its energy is `$A^2$`.  `$s$` is the ``smoothness`` argument, and
the exponent carries the *physical* wavenumbers, so the field's
correlation length is a property of `$s$` alone and does not change
when the box is resized.

The wall-bounded families need a second factor, because a column draw
is ``standard_normal`` per grid point -- grid-white in the wall-normal
direction, flat to the wall-normal Nyquist -- which no wall window
repairs; for the pipe it also makes near-axis regularity unattainable,
since that is a statement about *derivatives*.
:func:`_wall_normal_filter` supplies it, weighting the **energy** of
wall-normal polynomial index `$j$` by `$(1 - s_w)^{j}$`, where `$s_w$`
is the separate ``wall_smoothness`` argument
(``init.random_wall_smoothness`` / ``twin.wall_smoothness``).

**Why the two are separate knobs.**  They are not the same law -- `$A$`
is an amplitude and this is an energy, `$j$` counts modes where `$|k|$`
carries units -- but that alone would not force two *numbers*.  What
does is that the moment either moves, they want values an order of
magnitude apart, and moving one is a thing users do: the shipped
`$s = 0.4$` puts 93 % of a KMM-box perturbation's energy inside
`$|k_z| \le 2$` and starts every mode above `$|k_z| \approx 45$` below
round-off, which a high-Reynolds-number twin campaign has good reason
to change (``scripts/random_ic_calibrate.py``).  The wall-normal law
must **not** follow it down: lowering `$s_w$` moves the wall-normal
profile *away* from the wall rather than towards it (the wall window,
not the filter, is what sets that distribution -- see
:func:`_scaled_wall_window`), and `$s_w$` is also what keeps
grid-white Nyquist content out of the legacy IMM boundary term
(``tests/test_imm_continuity.py``).  So `$s$` buys a physical
correlation length where there is a box length to measure it against,
and `$s_w$` a polynomial-degree cutoff where there is not.  The
triply-periodic family has neither a filter nor a wall window: it
carries `$|k_y|$` in `$A$` and takes `$s$` alone.

**What `$s$` is not calibrated to, and why it has not moved.**  Scored
on the `$(y, k)$` shape a turbulent difference field settles into, a
ten-member plane-Poiseuille twin ensemble at `$Re_\tau \approx 180$`
wants `$s \approx 0.04$` (overlap 0.31 -> 0.95) and an HKW minimal
plane-Couette box at `$Re_\tau \approx 34$` wants `$s \approx 0.15$`.
The disagreement is physical -- the envelope carries a *physical*
wavenumber, and the band that amplifies scales with `$Re_\tau$` --
and it is not the whole story: at the minimal box, where growth can
actually be measured, it runs the other way.  Over 60 advective units
the difference energy gains 2.07 decades at `$s = 0.4$`, 1.15 at
`$0.15$` and 0.09 at `$0.04$`, decaying outright for the first five
units at the last (two seeds, both).  The shape score is therefore a
statement about *shape* and not a growth predictor -- the attractor's
small-scale content is sustained by transfer from larger scales, and
seeding it directly just feeds `$k^2/Re$`.  `$s$` stays at 0.4 until a
growth measurement at a Reynolds number with real scale separation
says otherwise.

**The wall window is scale-dependent** (``random_wall_confinement``,
the `$a$` of :func:`_scaled_wall_window`): a mode's window peaks where
the base window equals `$1/(a|k|)$`, so modes below `$1/a$` still fill
the gap while smaller ones sit nearer the wall.  This is the one factor
that moves the perturbation's wall-normal distribution at all -- the
smoothness filter does not, measurably.  Unlike `$s$` it is safe to
default on: it improves the `$(y, k)$` overlap at both Reynolds
numbers above, and at the minimal box, where growth is measurable, it
is exactly neutral (2.07 and 1.65 decades over 60 units with it, the
same two numbers without).  `$a$` is dimensionless, so it carries no
Reynolds number and no box length; `$a = 0$` is the plain window; and
`$|k| = 0$` recovers it whatever `$a$` is, which is why the `$(0, 0)$`
column and :mod:`dnsjax.ic.mean_mode` see no change from any of this.

The field is then normalised so the volume-averaged L2 norm equals
``amplitude``.  Each wall-bounded mode is built by solving continuity for
one velocity component (a `$1/k$` factor that would **inflate** the
low-wavenumber spectrum and make it domain-dependent -- as the box grows
the lowest mode would dominate), so the finished divergence-free mode is
rescaled to the envelope energy (:func:`_normalize_mode`) rather than
scaling the raw draw.  The triply-periodic family instead uses a Leray
projection (:func:`_leray`) that never divides by `$k$` and needs no such
step.

**Total-field Dean flows** (``dean`` / ``viscoelastic-dean``) integrate
the *total* field: the divergence-free perturbation is added to the
analytical laminar profile -- ``add_dean_laminar`` for Dean, while the
viscoelastic builder forms its 9-component total state directly and
``add_viscoelastic_laminar`` serves velocity-only ICs (the rolls path).

**The mean mode** `$(k_x, k_z) = (0, 0)$` is dropped unless the
caller asks for it -- ``init.random_mean_flow`` for a solver run
(**off** by default), ``twin.mean_flow`` for the ``dnsjax-twin``
partner (**on** by default).  Only the Cartesian flows offer the knob
at all: every other flow defers it, because only there are the
mean-mode conservation laws established.  When it is on, the generator
keeps that column and conditions it on those laws
(:mod:`dnsjax.ic.mean_mode`): the perturbed field drives the flow at
the same mean pressure gradient (equivalently, stays compatible with
no-slip at both walls) and, under a held mean
(``phys.driving = "constant_bulk_velocity"`` /
``phys.block_mean_spanwise_velocity``), carries an unchanged bulk
velocity.  The wall-normal component's mean mode is identically zero by
continuity in every geometry.

**Per-device, non-JAX construction**: each device fills only its own
spectral modes -- keyed by the *global* mode index, so the field is
identical at any ``(np0, np1)`` -- with NumPy per-mode loops (the
`$D_1 \mathbf{v}$` continuity matvecs and the wall windows), because
Python-level looping in JAX would incur tracing overhead.  No full array
is ever materialised: the shards are assembled with
:func:`dnsjax.snapshot.assemble_local_shards`, and only the final
norm/scale runs in JAX.  The wall-normal velocity carries a *squared*
wall window so its value and first derivative vanish at the walls
analytically; the continuity-derived component's no-slip is then only
truncation-level (projected by the first corrector step).

**The seed** is the first element of every per-mode key, so the field
is a function of ``(seed, global mode index)`` alone -- identical at any
``(np0, np1)``, *provided every process holds the same seed*.  That is
the entry points' job: an unset ``init.random_seed`` / ``twin.seed`` is
drawn once and agreed across processes before any generator here is
called (:mod:`dnsjax.seeding`, ``bootstrap.resolve_seed``).  Per-rank
draws would assemble one field out of unrelated streams -- still
divergence-free, still correctly normalised, and reproducible from no
recorded seed at all.

**Import-order discipline**: only NumPy and the JAX-free
``harmonics`` / ``parameters`` leaves are imported at module top.  ``jax``,
``sharding``, and the geometry modules (which build the ``fourier``
singleton at import) are imported lazily inside each generator, so
importing this module is safe before JAX is configured and before the
flow system is selected.
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING

import numpy as np

# Wavenumber sequences come from the JAX-free ``harmonics`` leaf: the
# per-device generators must never fetch the ``fourier`` singleton's
# wavenumber arrays, which are global multi-device arrays (not
# addressable per process under ``mpirun``).
from ..flows.registry import (
    annular_systems,
    annular_viscoelastic_systems,
    cartesian_systems,
    cylindrical_systems,
    cylindrical_viscoelastic_systems,
    periodic_systems,
)
from ..harmonics import complex_harmonics, real_harmonics
from ..parameters import derived_params, params
from .mean_mode import build_cartesian_projector

if TYPE_CHECKING:
    # ``Array`` is used only in (stringised) annotations, so it never
    # needs importing at runtime -- keeping this module importable
    # before JAX is configured (see the module docstring).
    from jax import Array

# ── Hermitian-symmetry enforcement ───────────────────────────────

# The real-FFT axis (kx for Cartesian/periodic, kz for cylindrical)
# stores only non-negative wavenumbers.  On the complex-FFT axis
# at kx=0 (or kz=0 for cylindrical), the stored modes must satisfy
# conjugate symmetry for the physical field to be real.  The helper
# below is pure NumPy (no JAX) since it works on the host array.


def enforce_hermitian_slice(
    slc: np.ndarray,
    n_physical: int,
) -> None:
    """Enforce conjugate symmetry in-place on a 1-D slice.

    ``slc`` has leading shape ``(Nc, ...)`` where
    ``Nc = n_physical - 1`` (Nyquist omitted), indexed by
    ``complex_harmonics(n_physical)``:
    ``[0, 1, ..., n//2-1, -n//2+1, ..., -1]``.

    Parameters
    ----------
    slc:
        The complex-FFT axis slice to fix, with the
        complex-FFT axis as axis 0.
    n_physical:
        Physical-space size of this direction.
    """
    n_pos = n_physical // 2
    Nc = n_physical - 1

    # Index 0 (wavenumber 0) must be real.
    slc[0] = slc[0].real

    # Pair positive kz at index i with negative kz at Nc-i.
    for i in range(1, n_pos):
        slc[Nc - i] = np.conj(slc[i])

    # Odd n: unpaired negative mode (Nyquist partner omitted).
    if n_physical % 2 == 1:
        slc[n_pos] = 0.0


# ── Per-device mode generation (no full-array replication) ───────
#
# Each device fills only its own ``(axis2, axis3)`` modes, keyed by the
# *global* mode index so the field is identical at any ``(np0, np1)``
# device configuration.  Numpy has no random access into a single PRNG
# stream, so each mode draws from its own key; the divergence-free /
# no-slip / norm properties and device-count independence hold by
# construction.  Conjugate symmetry on the real-FFT-axis-0
# plane is enforced *by construction* (the negative partner is the
# conjugate of the same canonical draw), so no cross-device communication
# and no plane replication are needed.
#
# Future note (not applicable while this stays numpy): if these
# generators are ever vectorised into JAX (removing the per-mode Python
# loops), use ``jax_threefry_partitionable=True`` with a replicated key
# and draws under ``out_shardings`` for trivial partition-aware PRNG.


def _column_draw(
    seed: int, i2: int, i3: int, n: int, rows: int = 3, stream: tuple = ()
) -> np.ndarray:
    """Independent complex ``(rows, n)`` draw keyed by global
    ``(i2, i3)``; *stream* extends the PRNG key to keep e.g. the
    viscoelastic conformation draw distinct from the velocity draw."""
    rng = np.random.default_rng((seed, i2, i3, *stream))
    return rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))


def _hermitian_column(
    seed: int, i2: int, n2: int, n: int, rows: int = 3, stream: tuple = ()
) -> np.ndarray:
    r"""Conjugate-consistent ``(rows, n)`` draw for the real-FFT-axis-0
    plane.

    Mirrors :func:`enforce_hermitian_slice`'s pairing over the length
    ``n2 - 1`` complex axis (axis-2 index ``i2``): index 0 real,
    ``i <-> n2-1-i`` conjugate pairs (the negative member is the conjugate
    of the same canonical draw, so both owning devices agree without
    communication), and the unpaired mode (odd ``n2``) zeroed.

    Every component built here -- Cartesian `$(u_x, u_y, u_z)$`,
    cylindrical / annular `$(u_z, u_r, u_\theta)$`, and the physical
    conformation tensor -- is the transform of a real field, so each
    row is made Hermitian individually.  (ICs are built in physical
    components throughout; ``__main__`` converts the finished state
    into the solver basis once.)
    """
    n_pos = n2 // 2
    if i2 == 0:
        rng = np.random.default_rng((seed, 0, 0, *stream))
        return rng.standard_normal((rows, n)).astype(np.complex128)
    if n2 % 2 == 1 and i2 == n_pos:
        return np.zeros((rows, n), dtype=np.complex128)
    canonical = i2 if i2 < n_pos else (n2 - 1) - i2
    rng = np.random.default_rng((seed, canonical, 0, *stream))
    d = rng.standard_normal((rows, n)) + 1j * rng.standard_normal((rows, n))
    if i2 < n_pos:
        return d
    return np.conj(d)


def _leray(
    col: np.ndarray, kx: float, ky: np.ndarray, kz: float
) -> np.ndarray:
    r"""Leray-project a ``(3, Nky)`` column: ``u - k (k·u)/|k|^2``."""
    k2 = kx**2 + ky**2 + kz**2
    k2_safe = np.where(k2 > 0, k2, 1.0)
    k_dot_u = kx * col[0] + ky * col[1] + kz * col[2]
    proj = k_dot_u / k2_safe
    out = np.empty_like(col)
    out[0] = col[0] - kx * proj
    out[1] = col[1] - ky * proj
    out[2] = col[2] - kz * proj
    return out


def _wall_normal_filter(coord: np.ndarray, decay: float) -> np.ndarray:
    r"""Wall-normal factor of the smoothness envelope, as a real
    ``(N, N)`` operator applied to a raw column draw.

    The periodic directions get their `$(1-s)^{2|k|}$` energy envelope
    from :func:`_normalize_mode`'s per-mode target, but a raw draw is
    ``standard_normal`` **per grid point** -- grid-white in the
    wall-normal direction, i.e. flat all the way to the wall-normal
    Nyquist.  This applies the missing factor: expand the column in the
    orthonormal polynomial basis of *coord*, weight index `$j$` by
    `$\sqrt{\text{decay}}^{\,j}$` -- so its *energy* follows
    `$\text{decay}^{\,j}$` -- and transform back.  *decay* is
    `$1 - s_w$` from the **separate** ``wall_smoothness`` knob, not the
    periodic directions' `$1 - s$`: theirs is `$(1-s)^{2|k|}$` in
    energy, in a wavenumber that carries units where `$j$` counts
    modes, and the two calibrate an order of magnitude apart.  The
    module docstring has the argument, and
    :func:`_scaled_wall_window` has the reason this is *not* the knob
    that moves the field's wall-normal energy distribution.

    *coord* is the variable the field is smooth in, ascending: `$y$`
    (Cartesian), `$r$` (annular), and `$r^2$` for the pipe -- an
    axis-regular field is an analytic function of `$r^2$`, so the
    filtered draw is even in `$r$` and the `$r^{|m_{\mathrm{eff}}|}$`
    envelope applied afterwards supplies the parity.

    Returns the identity at ``smoothness = 0``.  The basis is built by
    QR of the Chebyshev Vandermonde on *coord* affinely mapped to
    `$[-1, 1]$`, which is well conditioned (measured `$\kappa \approx
    1.5$`, orthonormal to 2e-15, up to at least ``N = 385``) on CGL,
    tanh and `$r^2$` node sets alike.
    """
    n = len(coord)
    lo, hi = float(coord[0]), float(coord[-1])
    t = 2.0 * (np.asarray(coord, dtype=float) - lo) / (hi - lo) - 1.0
    basis, _ = np.linalg.qr(np.polynomial.chebyshev.chebvander(t, n - 1))
    return (basis * decay ** (np.arange(n) / 2.0)) @ basis.T


def _scaled_wall_window(
    base: np.ndarray, k: float, confinement: float
) -> np.ndarray:
    r"""The wall window of one mode: *base*, narrowed towards the wall
    for modes above `$1/a$`.

    *base* is the geometry's own wall window, already normalised to a
    peak of 1 -- `$1 - y^2$` (Cartesian), `$1 - r$` (pipe, peaking at
    the axis), `$(r - r_1)(r_2 - r)$` (annulus, peaking at mid-gap).
    With ``confinement = 0`` it is returned unchanged, which is the
    behaviour every mode had before this argument existed.  Otherwise

    .. math::
        w_k = \frac{b\,e^{-a |k| b}}{\max_y b\,e^{-a |k| b}},
        \qquad b = \text{base},\ a = \text{confinement},

    which peaks at `$b = 1/(a|k|)$` -- inside the range only once
    `$|k| > 1/a$`, so large scales keep the plain window and smaller
    ones are pulled towards the wall as `$1/|k|$`.  That is the
    attached-eddy ridge in one expression, and `$a$` is dimensionless,
    so it fixes a ratio rather than a wall distance and needs no
    Reynolds number.

    Why this rather than a smaller ``wall_smoothness``: the wall-normal
    energy distribution of a windowed draw is set by the window and
    essentially not by the filter.  Sweeping the filter's `$s_w$` over
    `$0.4 \to 0.01$` moves the energy-weighted median wall distance of
    the finished field by under 5 % (and the wrong way); the bare
    `$(1 - y^2)^2$` energy window alone reproduces the full generator's
    profile.  Only the window can move it, because `$1 - y^2$` is
    centre-peaked by construction for any `$s_w$`.

    The narrowing is inert structurally: it is a real, positive,
    smooth factor applied before the continuity closure, and
    :func:`_normalize_mode` rescales the column afterwards, so it
    changes a mode's `$y$` distribution and not its energy, leaves the
    value/derivative wall conditions of the squared window standing,
    and commutes with the conjugate pairing (a pair shares `$|k|$`).
    At `$|k| = 0$` the exponential is 1, so the mean-mode column and
    the projector built on *base* (:mod:`dnsjax.ic.mean_mode`) are
    untouched.
    """
    # The early return is the exact `$a = 0$` limit, not a sentinel:
    # it also spares the renormalisation, which would only reproduce
    # *base* up to whether the base's own peak is exactly 1.
    if confinement == 0.0:
        return base
    w = base * np.exp(-(confinement * k) * base)
    peak = float(np.max(w))
    return w / peak if peak > 0.0 else base


def _normalize_mode(
    col: np.ndarray, y_weights: np.ndarray, envelope: float
) -> np.ndarray:
    r"""Scale a ``(C, Ny)`` spectral mode to wall-normal energy
    `$= \text{envelope}^2$`.

    The wall-bounded generators build each divergence-free mode by solving
    continuity for one component (`$u_z = -\mathrm{div}/\mathrm{i}k_z$`,
    or the `$u_x$` analogue) -- a `$1/k$` factor that **inflates** the
    derived component at low wavenumber, so applying the spectral envelope
    to the *draw* would leave the low-`$k$` modes
    over-energetic and the resulting spectrum **domain-dependent** (as the
    box grows the lowest mode dominates).  Instead this rescales the
    finished divergence-free mode so its wall-normal energy follows the
    envelope directly, giving a domain-independent spectrum.  A uniform
    real scaling preserves divergence-freeness, the wall BCs, and
    conjugate symmetry (the scale is identical for a conjugate pair: same
    `$|k|$`, same energy).  The triply-periodic generator needs no such
    step (its :func:`_leray` projection never divides by `$k$`).
    """
    energy = float(np.sum(y_weights[None, :] * np.abs(col) ** 2))
    if energy > 0.0:
        col = col * (envelope / np.sqrt(energy))
    return col


def _periodic_hermitian_raw(
    seed: int, i2: int, n2: int, ny: int, ky_flip: np.ndarray
) -> np.ndarray:
    r"""Hermitian-consistent raw ``(3, ny-1)`` column for the ``kx=0``
    plane of a triply-periodic field (the joint ``(ky, kz)`` symmetry
    ``f(ky,kz,0)=conj(f(-ky,-kz,0))``).

    ``kz`` (axis 2) is paired as in :func:`_hermitian_column`; the
    negative-``kz`` partner is ``conj`` of the canonical column with its
    ``ky`` axis flipped (``ky_flip``).  The ``kz=0`` column carries the
    within-column ``ky`` symmetry (:func:`enforce_hermitian_slice`).
    Returns the raw draw; the caller applies the (symmetry-preserving)
    decay and Leray projection.
    """
    n_pos = n2 // 2
    nky = ny - 1
    if i2 == 0:
        rng = np.random.default_rng((seed, 0, 0))
        col = rng.standard_normal((3, nky)) + 1j * rng.standard_normal(
            (3, nky)
        )
        enforce_hermitian_slice(col.T, ny)
        return col
    if n2 % 2 == 1 and i2 == n_pos:
        return np.zeros((3, nky), dtype=np.complex128)
    if i2 < n_pos:
        rng = np.random.default_rng((seed, i2, 0))
        return rng.standard_normal((3, nky)) + 1j * rng.standard_normal(
            (3, nky)
        )
    rng = np.random.default_rng((seed, (n2 - 1) - i2, 0))
    canon = rng.standard_normal((3, nky)) + 1j * rng.standard_normal((3, nky))
    return np.conj(canon[:, ky_flip])


# ── Cartesian wall-bounded generation ────────────────────────────


def generate_cartesian(
    amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
    mean_flow: bool,
) -> Array:
    """Generate a random divergence-free Cartesian perturbation.

    Built per device (no full-array replication): each device fills only
    its own ``(k_z, k_x)`` modes, keyed by the global mode index.  The
    wall-normal velocity carries a squared no-slip window
    ``(1-y^2)^2`` -- narrowed towards the wall by the mode's own
    wavenumber (:func:`_scaled_wall_window`) -- so its value and first
    derivative vanish at the walls and the
    continuity-derived component inherits a truncation-level wall value
    (projected by the first corrector step) while the independent
    components keep exact wall zeros.  Returns the sharded spectral state
    of shape ``(3, Ny, Nkz, Nkx)``.

    With *mean_flow* the `$(k_x, k_z) = (0, 0)$` column is kept rather
    than zeroed, conditioned on the mean-mode conservation laws by
    :func:`dnsjax.ic.mean_mode.build_cartesian_projector` -- so a
    perturbed field is compatible with the no-slip boundary condition
    at both walls and, under a held mean, carries the same bulk
    velocity as the state it perturbs.  The mean-mode draw is real
    (:func:`_hermitian_column` at ``i2 = 0``) and the filter, window,
    tilt rotation and projector are all real, so its reality needs no
    separate enforcement.  Conditioning runs *before*
    :func:`_normalize_mode`, whose uniform real scaling preserves every
    (homogeneous) constraint.
    """
    from ..geometries.wall_bounded._base import get_norm
    from ..geometries.wall_bounded.cartesian import (
        build_cartesian_grid,
        fourier,
    )
    from ..snapshot import assemble_local_shards

    nx = params.res.nx
    ny = params.res.ny
    nz = params.res.nz

    ys, D1, D2, y_weights = build_cartesian_grid(
        ny,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(ys)]

    ys_np = np.asarray(ys)
    D1_np = np.asarray(D1)
    yw_np = np.asarray(y_weights)
    kx_np = real_harmonics(nx) * (2 * pi / params.geo.lx)  # (Nkx,)
    kz_np = complex_harmonics(nz) * (2 * pi / params.geo.lz)  # (Nkz,)

    decay = 1.0 - smoothness
    wn_filter = _wall_normal_filter(ys_np, 1.0 - wall_smoothness)
    # Base wall window (peak 1 at the centreline); each mode narrows it
    # towards the wall by its own wavenumber (``_scaled_wall_window``).
    window_base = 1.0 - ys_np**2

    # One factorization for the single (0, 0) column, hoisted out of
    # the mode loop (and skipped entirely when the mode is zeroed).
    # It takes the *base* window, which is what ``_scaled_wall_window``
    # returns at |k| = 0 -- the mean mode's own window, unchanged.
    project_mean = (
        build_cartesian_projector(
            D1_np, np.asarray(D2), yw_np, window_base, wn_filter
        )
        if mean_flow
        else None
    )

    def fill_local(buf, kz_start, nkz, kx_start, nkx):
        for li in range(nkz):
            g2 = kz_start + li  # global k_z index (axis 2)
            kz_val = kz_np[g2]
            for lj in range(nkx):
                g3 = kx_start + lj  # global k_x index (axis 3, real)
                kx_val = kx_np[g3]
                window_tang = _scaled_wall_window(
                    window_base, np.hypot(kx_val, kz_val), wall_confinement
                )
                window_wn = window_tang**2  # value + derivative zero
                if g3 == 0:
                    col = _hermitian_column(seed, g2, nz, ny)
                else:
                    col = _column_draw(seed, g2, g3, ny)
                col = col @ wn_filter.T
                col[0] *= window_tang
                col[1] *= window_wn
                col[2] *= window_tang
                # Divergence-free by construction (same D1).
                if kx_val == 0 and kz_val == 0:
                    # Mean mode: continuity with no-slip forces
                    # ``u_y = 0``; the tangential pair is either
                    # dropped or conditioned on the mean-mode
                    # conservation laws (``dnsjax.ic.mean_mode``).
                    col[1] = 0.0
                    if project_mean is None:
                        col[0] = 0.0
                        col[2] = 0.0
                    else:
                        col[0], col[2] = project_mean(col[0].real, col[2].real)
                elif kz_val != 0:
                    col[2] = -(1j * kx_val * col[0] + D1_np @ col[1]) / (
                        1j * kz_val
                    )
                else:
                    col[0] = -(D1_np @ col[1]) / (1j * kx_val)
                # Energy = envelope^2 (no continuity 1/k low-k inflation).
                col = _normalize_mode(
                    col, yw_np, decay ** (abs(kz_val) + abs(kx_val))
                )
                buf[:, :, li, lj] = col

    state = assemble_local_shards(fill_local)
    norm = get_norm(state, fourier.k_metric, y_weights)
    return state * (amplitude / norm)


# ── Cylindrical generation ───────────────────────────────────────


def generate_cylindrical(
    amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
) -> Array:
    r"""Generate a random perturbation for pipe flow.

    Built per device (no full-array replication): each device fills only
    its own ``(m, k_z)`` modes, keyed by the global mode index.  Returns
    the sharded spectral state of shape ``(3, Nr, Nm, Nkz)`` in
    `$(u_z, u_r, u_\theta)$` form.  `$u_r$` and `$u_\theta$` carry a
    squared wall window `$(1-r)^2$` (value and first derivative vanish
    at `$r = 1$`), narrowed towards the wall by the mode's own
    wavenumber (:func:`_scaled_wall_window`), so for
    `$k_z \neq 0$` the continuity-derived `$u_z$`
    inherits a truncation-level wall value (projected by the first
    corrector step).  The inner end `$r = 0$` is the axis, not a wall,
    and every column carries the **axis-regularity** envelope

    .. math::
        u_z \sim r^{|m|}, \qquad
        u_\pm = u_r \pm i\,u_\theta \sim r^{|m \pm 1|},

    applied in the `$u_\pm$` basis: the condition is a cancellation
    *between* `$u_r$` and `$u_\theta$`, so enveloping them separately
    reproduces only the slower `$r^{|m|-1}$` leading behaviour and
    leaves `$u_+$` two orders too large.  Parity is implied by it
    (`$r^{|m \pm 1|}$` carries `$(-1)^{m+1}$`, `$r^{|m|}$` carries
    `$(-1)^m$`), and a parity-only window is merely its
    `$|m_{\mathrm{eff}}| = 1$` case: that admits near-axis content the
    continuum forbids, measured to drive a stepped state's discrete
    divergence to `$O(1)$` at the innermost radial node and to grow as
    `$N_r^2$` under refinement.  The envelope also preserves the
    `$k_z = 0$` Hermitian pairing, since `$\hat u_+(-m) =
    \overline{\hat u_-(m)}$` and the two carry the same real factor.
    For `$k_z = 0$` the axial `$u_z$` drops out of continuity, so it
    is closed through `$u_\theta$` instead (its `$im/r$` coefficient is
    diagonal, so the solve is exact discretely); the `$m = 0$` mean has
    `$u_r = 0$` by no-slip.  The
    `$k_z = 0$` plane is drawn per-component Hermitian
    (:func:`_hermitian_column`) so every physical component --
    including the axial-mean swirl -- is real.
    """
    from ..geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
        fourier,
        get_norm2_cyl,
    )
    from ..snapshot import assemble_local_shards

    nx = params.res.nx
    Nr = params.res.ny
    nz = params.res.nz

    rs, D1_even, D1_odd, _, y_weights, _, inv_r = build_cylindrical_grid(
        Nr,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    rs_np = np.asarray(rs)
    inv_r_np = np.asarray(inv_r)
    D1_even_np = np.asarray(D1_even)
    D1_odd_np = np.asarray(D1_odd)
    yw_np = np.asarray(y_weights)
    kz_np = real_harmonics(nx) * (2 * pi / params.geo.lx)  # axial
    # Physical azimuthal wavenumbers m = m0 * harmonic over the wedge
    # (m0 = 1 full circle): the continuity relation and axis parity need
    # the physical m; the decay envelope then weights by |m| directly.
    m_np = params.geo.m0 * complex_harmonics(nz)

    decay = 1.0 - smoothness
    # Filter in x = r^2: an axis-regular field is analytic in it, so the
    # filtered draw is even in r and the r^|m_eff| envelope below
    # supplies the parity.
    wn_filter = _wall_normal_filter(rs_np**2, 1.0 - wall_smoothness)
    # Base wall window, peak 1 at the axis; narrowed per mode below.
    window_base = 1.0 - rs_np  # u_z: f(1) = 0

    def fill_local(buf, m_start, n_m, kz_start, n_kz):
        for li in range(n_m):
            g2 = m_start + li  # global m index (axis 2)
            m_val = int(m_np[g2])
            # u_r/u_theta parity (-1)^{m+1}: even D1 when m odd, else
            # odd D1.
            D1_v = D1_even_np if (m_val + 1) % 2 == 0 else D1_odd_np
            for lj in range(n_kz):
                g3 = kz_start + lj  # global k_z index (axis 3, real)
                kz_val = kz_np[g3]
                window_wall = _scaled_wall_window(
                    window_base, np.hypot(kz_val, m_val), wall_confinement
                )
                window_wn = window_wall**2  # value + derivative zero
                if g3 == 0:
                    col = _hermitian_column(seed, g2, nz, Nr)
                else:
                    col = _column_draw(seed, g2, g3, Nr)
                col = col @ wn_filter.T
                col[0] *= window_wall
                col[1] *= window_wn
                col[2] *= window_wn
                # Axis-regularity envelope at r = 0 (the docstring):
                # u_z ~ r^|m|, u_pm ~ r^|m +- 1|.  Applied in the u_pm
                # basis because the condition is a cancellation
                # *between* u_r and u_theta -- enveloping them
                # separately leaves u_+ two orders too large.  Parity
                # follows from it, so this replaces (not supplements)
                # a parity-only window.
                cp = (col[1] + 1j * col[2]) * rs_np ** abs(m_val + 1)
                cm = (col[1] - 1j * col[2]) * rs_np ** abs(m_val - 1)
                col[1] = (cp + cm) / 2
                col[2] = (cp - cm) / 2j
                col[0] *= rs_np ** abs(m_val)
                # Close continuity per mode against the cylindrical
                # divergence D1 u_r + u_r/r + (im/r) u_theta + i k_z u_z,
                # keeping u_r's filtered/windowed/enveloped draw.  Both
                # u_z (k_z != 0) and u_theta (k_z = 0, m != 0) enter with
                # an r-diagonal coefficient, so solving for one of them is
                # exact discretely ((1/r).r = I elementwise).
                if kz_val != 0:
                    div_perp = (
                        D1_v @ col[1]
                        + inv_r_np * col[1]
                        + 1j * m_val * inv_r_np * col[2]
                    )
                    col[0] = -div_perp / (1j * kz_val)
                elif m_val != 0:
                    # k_z = 0: u_z drops out; close through u_theta.
                    col[2] = (
                        1j
                        * rs_np
                        * (D1_v @ col[1] + inv_r_np * col[1])
                        / m_val
                    )
                else:
                    # k_z = 0, m = 0 mean mode: (1/r) d(r u_r)/dr = 0 with
                    # no-slip forces u_r = 0 (u_theta swirl / u_z axial
                    # stay).
                    col[1] = 0.0
                # Energy = envelope^2 (no continuity 1/k low-k inflation).
                col = _normalize_mode(
                    col,
                    yw_np,
                    decay ** (abs(kz_val) + abs(m_val)),
                )
                # Mean mode: the (0, 0) conservation laws are only
                # established for the Cartesian flows, so every other
                # flow defers ``init.random_mean_flow``
                # (``dnsjax.ic.mean_mode``, and the per-flow
                # ``DeferredSpec``s).
                if g2 == 0 and g3 == 0:
                    col[:] = 0.0
                buf[:, :, li, lj] = col

    state = assemble_local_shards(fill_local)
    norm2 = get_norm2_cyl(state, fourier.k_metric, y_weights)
    return state * (amplitude / norm2**0.5)


# ── Annular generation ───────────────────────────────────────────


def generate_annular(
    amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
) -> Array:
    r"""Generate a random perturbation for Taylor-Couette flow.

    Built per device (no full-array replication): each device fills only
    its own ``(m, k_z)`` modes, keyed by the global mode index.  Returns
    the sharded spectral state of shape ``(3, Nr, Nm, Nkz)`` in
    `$(u_z, u_r, u_\theta)$` form.  `$u_r$` and `$u_\theta$` carry a
    squared no-slip window `$((r-r_1)(r_2-r))^2$` (value and first
    derivative vanish at both walls), narrowed towards the walls by the
    mode's own wavenumber (:func:`_scaled_wall_window`), so for
    `$k_z \neq 0$` the
    continuity-derived `$u_z$` inherits a truncation-level wall value
    (projected by the first corrector step); the independent components
    keep exact wall zeros.  For `$k_z = 0$` the axial `$u_z$` drops out
    of continuity, so it is closed through `$u_\theta$` instead (exact
    discretely); the `$m = 0$` mean has `$u_r = 0$`.  The `$k_z = 0$`
    plane is drawn per-component Hermitian (:func:`_hermitian_column`)
    so every physical component -- including the axial-mean swirl --
    is real.
    """
    from ..geometries.wall_bounded.annular import (
        build_annular_grid,
        fourier,
        get_norm2_annular,
    )
    from ..snapshot import assemble_local_shards

    nx = params.res.nx
    Nr = params.res.ny
    nz = params.res.nz

    r1 = derived_params.r_inner
    r2 = derived_params.r_outer
    rs, D1, _, y_weights, inv_r = build_annular_grid(
        Nr,
        params.res.fd_order,
        r1,
        r2,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    rs_np = np.asarray(rs)
    inv_r_np = np.asarray(inv_r)
    D1_np = np.asarray(D1)
    yw_np = np.asarray(y_weights)
    kz_np = real_harmonics(nx) * (2 * pi / params.geo.lx)  # axial
    # Physical azimuthal wavenumbers m = m0 * harmonic over the wedge
    # (m0 = 1 full circle): the continuity relation needs the physical m;
    # the decay envelope then weights by |m| directly.
    m_np = params.geo.m0 * complex_harmonics(nz)

    decay = 1.0 - smoothness
    wn_filter = _wall_normal_filter(rs_np, 1.0 - wall_smoothness)
    # No-slip window: zero at both walls, peak 1 at mid-gap; narrowed
    # towards the walls per mode below (``_scaled_wall_window``).
    window_base = (rs_np - r1) * (r2 - rs_np)
    window_base = window_base / np.max(window_base)

    def fill_local(buf, m_start, n_m, kz_start, n_kz):
        for li in range(n_m):
            g2 = m_start + li  # global m index (axis 2)
            m_val = int(m_np[g2])
            for lj in range(n_kz):
                g3 = kz_start + lj  # global k_z index (axis 3, real)
                kz_val = kz_np[g3]
                window_lin = _scaled_wall_window(
                    window_base, np.hypot(kz_val, m_val), wall_confinement
                )
                window_wn = window_lin**2  # value + derivative zero
                if g3 == 0:
                    col = _hermitian_column(seed, g2, nz, Nr)
                else:
                    col = _column_draw(seed, g2, g3, Nr)
                col = col @ wn_filter.T
                col[0] *= window_lin
                col[1] *= window_wn
                col[2] *= window_wn
                # Close continuity per mode against the annular
                # divergence D1 u_r + u_r/r + (im/r) u_theta + i k_z u_z,
                # keeping u_r's filtered/windowed draw.  Both u_z
                # (k_z != 0) and u_theta (k_z = 0, m != 0) enter with an
                # r-diagonal coefficient, so solving for one of them is
                # exact discretely ((1/r).r = I elementwise).
                if kz_val != 0:
                    div_perp = (
                        D1_np @ col[1]
                        + inv_r_np * col[1]
                        + 1j * m_val * inv_r_np * col[2]
                    )
                    col[0] = -div_perp / (1j * kz_val)
                elif m_val != 0:
                    # k_z = 0: u_z drops out; close through u_theta.
                    col[2] = (
                        1j
                        * rs_np
                        * (D1_np @ col[1] + inv_r_np * col[1])
                        / m_val
                    )
                else:
                    # k_z = 0, m = 0 mean mode: (1/r) d(r u_r)/dr = 0 with
                    # no-slip forces u_r = 0 (u_theta swirl / u_z axial
                    # stay).
                    col[1] = 0.0
                # Energy = envelope^2 (no continuity 1/k low-k inflation).
                col = _normalize_mode(
                    col,
                    yw_np,
                    decay ** (abs(kz_val) + abs(m_val)),
                )
                # Mean mode: the (0, 0) conservation laws are only
                # established for the Cartesian flows, so every other
                # flow defers ``init.random_mean_flow``
                # (``dnsjax.ic.mean_mode``, and the per-flow
                # ``DeferredSpec``s).
                if g2 == 0 and g3 == 0:
                    col[:] = 0.0
                buf[:, :, li, lj] = col

    state = assemble_local_shards(fill_local)
    norm2 = get_norm2_annular(state, fourier.k_metric, y_weights)
    return state * (amplitude / norm2**0.5)


def add_dean_laminar(state: Array) -> Array:
    r"""Add the analytical laminar Dean profile to a perturbation.

    Dean flow integrates the **total** velocity, so a usable initial
    condition is the closed-form laminar azimuthal profile (placed at
    the mean mode) plus the divergence-free random perturbation from
    :func:`generate_annular`.  The laminar profile is axisymmetric and
    zero at both walls, so it preserves the perturbation's
    divergence-free and no-slip properties.  Returns the total spectral
    state in `$(u_z, u_r, u_\theta)$` form.
    """
    from jax import numpy as jnp

    from ..geometries.wall_bounded.annular import (
        build_annular_grid,
        dean_laminar_u_theta,
        fourier,
    )
    from ..sharding import sharding

    rs, *_ = build_annular_grid(
        params.res.ny,
        params.res.fd_order,
        derived_params.r_inner,
        derived_params.r_outer,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    u_theta = dean_laminar_u_theta(rs, params.geo.eta)  # (Nr,) real
    # Place U_theta at the mean mode as the u_theta component.
    u_spec = jnp.where(fourier.mean_mask, u_theta[:, None, None], 0.0)
    zeros = jnp.zeros_like(u_spec, dtype=sharding.complex_type)
    laminar = jnp.stack([zeros, zeros, u_spec.astype(sharding.complex_type)])
    return state + laminar


# Viscoelastic conformation noise: every physical tensor
# component (c_zz, c_rz, c_theta_z, c_rr, c_theta_theta, c_r_theta) is
# the transform of a real field, so its draws use the same
# per-component machinery as the velocity -- with a trailing ``1`` in
# the PRNG key (the ``stream`` argument) to keep the conformation
# stream distinct from the velocity draw at the same mode.


def add_viscoelastic_laminar(vel_state: Array) -> Array:
    r"""Turn a 3-component velocity perturbation into a 9-component
    viscoelastic total-field state.

    Adds the analytical laminar velocity profile to *vel_state* and
    appends the laminar sPTT-equilibrium conformation (both at the mean
    mode), giving the total-field IC in the physical layout
    `$(u_z, u_r, u_\theta, c_{zz}, c_{rz}, c_{\theta z}, c_{rr},
    c_{\theta\theta}, c_{r\theta})$`.
    Used by the localized-rolls IC (a velocity-only perturbation); the
    random IC builds its 9 components directly.
    """
    from jax import numpy as jnp

    from ..geometries.wall_bounded.annular import build_annular_grid, fourier
    from ..geometries.wall_bounded.annular_viscoelastic import (
        viscoelastic_laminar_profiles,
    )

    r1 = derived_params.r_inner
    r2 = derived_params.r_outer
    rs, D1, *_ = build_annular_grid(
        params.res.ny,
        params.res.fd_order,
        r1,
        r2,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    prof = viscoelastic_laminar_profiles(
        rs, D1, r1, r2, params.phys.wi, params.phys.epsilon
    )
    prof_jax = jnp.asarray(prof, dtype=vel_state.dtype)
    laminar = jnp.where(
        fourier.mean_mask[None], prof_jax[:, :, None, None], 0.0
    )
    total_vel = vel_state + laminar[:3]
    return jnp.concatenate([total_vel, laminar[3:]])


def generate_viscoelastic_dean(
    amplitude: float,
    conf_amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
) -> Array:
    r"""Random 9-component IC for viscoelastic (sPTT) Dean flow.

    Built per device (no full-array replication): the velocity part is
    the divergence-free annular draw of :func:`generate_annular` (rows
    ``0:3``); the conformation part (rows ``3:9``) is windowed,
    spectrally-decaying symmetric-tensor noise (the reference
    restart recipe).  Velocity and conformation noise are
    rescaled to *amplitude* / *conf_amplitude* separately, then the
    analytical laminar pair (velocity profile + sPTT-equilibrium
    conformation) is added at the mean mode (total-field IC).  The
    conformation noise vanishes at both walls and at the mean mode (so
    the laminar wall / mean values are preserved).
    """
    from jax import numpy as jnp

    from ..geometries.wall_bounded.annular import (
        build_annular_grid,
        fourier,
        get_norm2_annular,
    )
    from ..geometries.wall_bounded.annular_viscoelastic import (
        get_norm2_conformation,
        viscoelastic_laminar_profiles,
    )
    from ..snapshot import assemble_local_shards

    nx = params.res.nx
    Nr = params.res.ny
    nz = params.res.nz
    r1 = derived_params.r_inner
    r2 = derived_params.r_outer
    rs, D1, _, y_weights, inv_r = build_annular_grid(
        Nr,
        params.res.fd_order,
        r1,
        r2,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    rs_np = np.asarray(rs)
    inv_r_np = np.asarray(inv_r)
    D1_np = np.asarray(D1)
    yw_np = np.asarray(y_weights)
    kz_np = real_harmonics(nx) * (2 * pi / params.geo.lx)  # axial
    # Physical azimuthal wavenumbers m = m0 * harmonic over the wedge
    # (m0 = 1 full circle): the continuity relation needs the physical
    # m; the decay envelope then weights by |m| directly.
    m_np = params.geo.m0 * complex_harmonics(nz)

    decay = 1.0 - smoothness
    wn_filter = _wall_normal_filter(rs_np, 1.0 - wall_smoothness)
    window_base = (rs_np - r1) * (r2 - rs_np)
    window_base = window_base / np.max(window_base)

    def fill_local(buf, m_start, n_m, kz_start, n_kz):
        for li in range(n_m):
            g2 = m_start + li
            m_val = int(m_np[g2])
            for lj in range(n_kz):
                g3 = kz_start + lj
                kz_val = kz_np[g3]
                envelope = decay ** (abs(kz_val) + abs(m_val))
                window_lin = _scaled_wall_window(
                    window_base, np.hypot(kz_val, m_val), wall_confinement
                )
                window_wn = window_lin**2

                # Velocity (rows 0:3): divergence-free draw.
                if g3 == 0:
                    vcol = _hermitian_column(seed, g2, nz, Nr)
                else:
                    vcol = _column_draw(seed, g2, g3, Nr)
                vcol = vcol @ wn_filter.T
                vcol[0] *= window_lin
                vcol[1] *= window_wn
                vcol[2] *= window_wn
                # Close continuity per mode against the annular
                # divergence, exactly as :func:`generate_annular` does
                # (same operator, same r-diagonal coefficients).
                if kz_val != 0:
                    div_perp = (
                        D1_np @ vcol[1]
                        + inv_r_np * vcol[1]
                        + 1j * m_val * inv_r_np * vcol[2]
                    )
                    vcol[0] = -div_perp / (1j * kz_val)
                elif m_val != 0:
                    # k_z = 0: u_z drops out; close through u_theta.
                    vcol[2] = (
                        1j
                        * rs_np
                        * (D1_np @ vcol[1] + inv_r_np * vcol[1])
                        / m_val
                    )
                else:
                    # k_z = 0, m = 0 mean mode: (1/r) d(r u_r)/dr = 0
                    # with no-slip forces u_r = 0.
                    vcol[1] = 0.0
                vcol = _normalize_mode(vcol, yw_np, envelope)
                # Mean mode: the (0, 0) conservation laws are only
                # established for the Cartesian flows, so every other
                # flow defers ``init.random_mean_flow``
                # (``dnsjax.ic.mean_mode``, and the per-flow
                # ``DeferredSpec``s).
                if g2 == 0 and g3 == 0:
                    vcol[:] = 0.0
                buf[0:3, :, li, lj] = vcol

                # Conformation (rows 3:9): windowed, wall-vanishing noise;
                # zero at the mean mode (laminar added below).
                if g3 == 0:
                    ccol = _hermitian_column(
                        seed, g2, nz, Nr, rows=6, stream=(1,)
                    )
                else:
                    ccol = _column_draw(seed, g2, g3, Nr, rows=6, stream=(1,))
                ccol = (ccol @ wn_filter.T) * window_wn
                ccol = _normalize_mode(ccol, yw_np, envelope)
                if g2 == 0 and g3 == 0:
                    ccol[:] = 0.0
                buf[3:9, :, li, lj] = ccol

    state = assemble_local_shards(fill_local)

    # Rescale velocity and conformation noise separately.
    vel_norm2 = get_norm2_annular(state[:3], fourier.k_metric, y_weights)
    conf_norm2 = get_norm2_conformation(state[3:], fourier.k_metric, y_weights)
    scale_v = amplitude / vel_norm2**0.5
    scale_c = conf_amplitude / conf_norm2**0.5
    state = jnp.concatenate([state[:3] * scale_v, state[3:] * scale_c])

    # Add the laminar pair at the mean mode (total-field IC).
    prof = viscoelastic_laminar_profiles(
        rs, D1, r1, r2, params.phys.wi, params.phys.epsilon
    )
    prof_jax = jnp.asarray(prof, dtype=state.dtype)
    laminar = jnp.where(
        fourier.mean_mask[None], prof_jax[:, :, None, None], 0.0
    )
    return state + laminar


def add_viscoelastic_pipe_laminar(vel_state: Array) -> Array:
    r"""Pipe twin of :func:`add_viscoelastic_laminar`.

    Adds the Hagen-Poiseuille laminar velocity to *vel_state* and
    appends the laminar sPTT-equilibrium conformation (both at the mean
    mode), giving the 9-component total-field IC in the physical
    layout.  Used by the localized-rolls IC (a velocity-only
    perturbation); the random IC builds its 9 components directly.
    """
    from jax import numpy as jnp

    from ..geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
        fourier,
    )
    from ..geometries.wall_bounded.cylindrical_viscoelastic import (
        viscoelastic_laminar_profiles,
    )

    rs, D1_even, *_ = build_cylindrical_grid(
        params.res.ny,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    prof = viscoelastic_laminar_profiles(
        np.asarray(rs),
        np.asarray(D1_even),
        params.phys.wi,
        params.phys.epsilon,
    )
    prof_jax = jnp.asarray(prof, dtype=vel_state.dtype)
    laminar = jnp.where(
        fourier.mean_mask[None], prof_jax[:, :, None, None], 0.0
    )
    total_vel = vel_state + laminar[:3]
    return jnp.concatenate([total_vel, laminar[3:]])


def generate_viscoelastic_pipe(
    amplitude: float,
    conf_amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
) -> Array:
    r"""Random 9-component IC for viscoelastic (sPTT) pipe flow.

    Built per device (no full-array replication): the velocity part is
    the divergence-free pipe draw of :func:`generate_cylindrical` (rows
    ``0:3``); the conformation part (rows ``3:9``) is windowed,
    spectrally-decaying symmetric-tensor noise.  Velocity and
    conformation noise are rescaled to *amplitude* / *conf_amplitude*
    separately, then the analytical laminar pair (Hagen-Poiseuille
    velocity + sPTT-equilibrium conformation) is added at the mean mode
    (total-field IC).  The conformation noise vanishes at the wall and
    at the mean mode (so the laminar wall / mean values are preserved).

    Unlike the annular twin, the inner end is the pipe **axis**, so
    each tensor column also carries the axis-regularity envelope
    `$r^{|m+s|}$` of its spin weight `$s$` -- applied in the spin
    basis, for the same reason the velocity envelope is
    (:func:`generate_cylindrical`): the condition is a cancellation
    *between* physical components, so enveloping them separately would
    leave the high-spin combinations orders too large near the axis.
    Parity follows from the envelope.
    """
    from jax import numpy as jnp

    from ..geometries.wall_bounded._viscoelastic_common import (
        get_norm2_conformation,
    )
    from ..geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
        fourier,
        get_norm2_cyl,
    )
    from ..geometries.wall_bounded.cylindrical_viscoelastic import (
        viscoelastic_laminar_profiles,
    )
    from ..snapshot import assemble_local_shards

    nx = params.res.nx
    Nr = params.res.ny
    nz = params.res.nz

    rs, D1_even, D1_odd, _, y_weights, _, inv_r = build_cylindrical_grid(
        Nr,
        params.res.fd_order,
        params.geo.wall_grid,
        params.geo.grid_type,
        params.geo.grid_stretch,
    )
    derived_params.wall_normal_grid = [float(v) for v in np.asarray(rs)]

    rs_np = np.asarray(rs)
    inv_r_np = np.asarray(inv_r)
    D1_even_np = np.asarray(D1_even)
    D1_odd_np = np.asarray(D1_odd)
    yw_np = np.asarray(y_weights)
    kz_np = real_harmonics(nx) * (2 * pi / params.geo.lx)  # axial
    # Physical azimuthal wavenumbers m = m0 * harmonic over the wedge.
    m_np = params.geo.m0 * complex_harmonics(nz)

    decay = 1.0 - smoothness
    # Filter in x = r^2 (an axis-regular field is analytic in it), then
    # a wall window; the r^|m+s| envelopes below supply the axis
    # behaviour and the parity, for the velocity and the tensor alike.
    wn_filter = _wall_normal_filter(rs_np**2, 1.0 - wall_smoothness)
    window_base = 1.0 - rs_np

    def fill_local(buf, m_start, n_m, kz_start, n_kz):
        for li in range(n_m):
            g2 = m_start + li
            m_val = int(m_np[g2])
            D1_v = D1_even_np if (m_val + 1) % 2 == 0 else D1_odd_np
            # Axis-regularity envelope per spin weight s (u_z and the
            # spin-0 tensor slots at s = 0, u_pm / c_z+- at s = +-1,
            # c_+-+- at s = +-2).
            env = {s: rs_np ** abs(m_val + s) for s in (0, 1, -1, 2, -2)}
            for lj in range(n_kz):
                g3 = kz_start + lj
                kz_val = kz_np[g3]
                envelope = decay ** (abs(kz_val) + abs(m_val))
                window_wall = _scaled_wall_window(
                    window_base, np.hypot(kz_val, m_val), wall_confinement
                )
                window_wn = window_wall**2

                # ── Velocity (rows 0:3): the divergence-free pipe draw
                # of ``generate_cylindrical`` (same windows, envelope
                # and per-mode continuity closure). ──
                if g3 == 0:
                    col = _hermitian_column(seed, g2, nz, Nr)
                else:
                    col = _column_draw(seed, g2, g3, Nr)
                col = col @ wn_filter.T
                col[0] *= window_wall
                col[1] *= window_wn
                col[2] *= window_wn
                cp = (col[1] + 1j * col[2]) * env[1]
                cm = (col[1] - 1j * col[2]) * env[-1]
                col[1] = (cp + cm) / 2
                col[2] = (cp - cm) / 2j
                col[0] *= env[0]
                if kz_val != 0:
                    div_perp = (
                        D1_v @ col[1]
                        + inv_r_np * col[1]
                        + 1j * m_val * inv_r_np * col[2]
                    )
                    col[0] = -div_perp / (1j * kz_val)
                elif m_val != 0:
                    col[2] = (
                        1j
                        * rs_np
                        * (D1_v @ col[1] + inv_r_np * col[1])
                        / m_val
                    )
                else:
                    col[1] = 0.0
                col = _normalize_mode(col, yw_np, envelope)
                # Mean mode: the (0, 0) conservation laws are only
                # established for the Cartesian flows, so every other
                # flow defers ``init.random_mean_flow``
                # (``dnsjax.ic.mean_mode``, and the per-flow
                # ``DeferredSpec``s).
                if g2 == 0 and g3 == 0:
                    col[:] = 0.0
                buf[0:3, :, li, lj] = col

                # ── Conformation (rows 3:9): windowed, wall-vanishing
                # noise, axis-enveloped in the spin basis; zero at the
                # mean mode (laminar added below). ──
                if g3 == 0:
                    ccol = _hermitian_column(
                        seed, g2, nz, Nr, rows=6, stream=(1,)
                    )
                else:
                    ccol = _column_draw(seed, g2, g3, Nr, rows=6, stream=(1,))
                ccol = (ccol @ wn_filter.T) * window_wn
                # Stored physical order (c_zz, c_rz, c_thz, c_rr,
                # c_thth, c_rth) -> spin combos (the definitions of
                # ``_viscoelastic_common.phys_combos_to_spin``, inlined
                # in NumPy for this host-side per-mode loop) ->
                # envelope by spin weight -> back.
                c_zz, c_rz, c_thz = ccol[0], ccol[1], ccol[2]
                c_rr, c_thth, c_rth = ccol[3], ccol[4], ccol[5]
                c_zp = (c_rz + 1j * c_thz) * env[1]
                c_zm = (c_rz - 1j * c_thz) * env[-1]
                c_pm = (c_rr + c_thth) * env[0]
                c_pp = ((c_rr - c_thth) + 2j * c_rth) * env[2]
                c_mm = ((c_rr - c_thth) - 2j * c_rth) * env[-2]
                d = (c_pp + c_mm) / 2  # = c_rr - c_theta_theta
                ccol = np.stack(
                    [
                        c_zz * env[0],
                        (c_zp + c_zm) / 2,
                        -0.5j * (c_zp - c_zm),
                        c_pm / 2 + d / 2,
                        c_pm / 2 - d / 2,
                        -0.5j * (c_pp - c_mm) / 2,
                    ]
                )
                ccol = _normalize_mode(ccol, yw_np, envelope)
                if g2 == 0 and g3 == 0:
                    ccol[:] = 0.0
                buf[3:9, :, li, lj] = ccol

    state = assemble_local_shards(fill_local)

    # Rescale velocity and conformation noise separately.
    vel_norm2 = get_norm2_cyl(state[:3], fourier.k_metric, y_weights)
    conf_norm2 = get_norm2_conformation(state[3:], fourier.k_metric, y_weights)
    state = jnp.concatenate(
        [
            state[:3] * (amplitude / vel_norm2**0.5),
            state[3:] * (conf_amplitude / conf_norm2**0.5),
        ]
    )

    # Add the laminar pair at the mean mode (total-field IC).
    prof = viscoelastic_laminar_profiles(
        rs_np, D1_even_np, params.phys.wi, params.phys.epsilon
    )
    prof_jax = jnp.asarray(prof, dtype=state.dtype)
    laminar = jnp.where(
        fourier.mean_mask[None], prof_jax[:, :, None, None], 0.0
    )
    return state + laminar


# ── Triply-periodic generation ───────────────────────────────────


def generate_triply_periodic(
    amplitude: float,
    smoothness: float,
    seed: int,
) -> Array:
    """Generate a random divergence-free periodic perturbation.

    Built per device (no full-array replication): each device fills only
    its own ``(k_z, k_x)`` columns over the full (unsharded) ``k_y`` axis,
    keyed by the global mode index, and Leray-projects each column to
    enforce incompressibility.  Conjugate symmetry on the ``k_x = 0``
    plane (the joint ``f(ky,kz,0) = conj(f(-ky,-kz,0))``) holds by
    construction.  Returns the sharded spectral state of shape
    ``(3, Nky, Nkz, Nkx)``.
    """
    from ..geometries.triply_periodic.triply_periodic import (
        fourier,
        get_norm,
        ly,
    )
    from ..snapshot import assemble_local_shards

    nx = params.res.nx
    ny = params.res.ny
    nz = params.res.nz
    Nky = ny - 1

    kx_np = real_harmonics(nx) * (2 * pi / params.geo.lx)  # (Nkx,)
    kz_np = complex_harmonics(nz) * (2 * pi / params.geo.lz)  # (Nkz,)
    ky_np = complex_harmonics(ny) * (2 * pi / ly)  # (Nky,)

    decay = 1.0 - smoothness
    # k_y conjugate-partner permutation (index i <-> ny-1-i, 0 -> 0).
    ky_flip = np.array([0] + [(ny - 1) - i for i in range(1, Nky)], dtype=int)

    def fill_local(buf, kz_start, nkz, kx_start, nkx):
        for li in range(nkz):
            g2 = kz_start + li  # global k_z index (axis 2)
            kz_val = kz_np[g2]
            for lj in range(nkx):
                g3 = kx_start + lj  # global k_x index (axis 3, real)
                kx_val = kx_np[g3]
                if g3 == 0:
                    col = _periodic_hermitian_raw(seed, g2, nz, ny, ky_flip)
                else:
                    col = _column_draw(seed, g2, g3, Nky)
                col = col * decay ** (
                    np.abs(ky_np) + abs(kz_val) + abs(kx_val)
                )
                col = _leray(col, kx_val, ky_np, kz_val)
                # Mean mode: the (0, 0) conservation laws are only
                # established for the Cartesian flows, so every other
                # flow defers ``init.random_mean_flow``
                # (``dnsjax.ic.mean_mode``, and the per-flow
                # ``DeferredSpec``s).
                if g2 == 0 and g3 == 0:
                    col[:, 0] = 0.0  # mean mode (ky=kz=kx=0)
                buf[:, :, li, lj] = col

    state = assemble_local_shards(fill_local)
    norm = get_norm(state, fourier.k_metric)
    return state * (amplitude / norm)


# ── Dispatch ─────────────────────────────────────────────────────


def generate_random_state(
    amplitude: float,
    smoothness: float,
    wall_smoothness: float,
    wall_confinement: float | None,
    seed: int,
    mean_flow: bool = False,
) -> Array:
    """Generate a random initial state for the configured flow system.

    Dispatches to the geometry-specific generator for
    ``params.phys.system`` and returns the sharded spectral state (on
    ``sharding.spec_vector_shard``), ready to time step -- the same
    object type that ``init_state`` / ``load_snapshot`` return.  For the
    total-field Dean flow the analytical laminar profile is added to the
    perturbation; every other system returns the perturbation directly.

    *mean_flow* (``init.random_mean_flow`` for the solver,
    ``twin.mean_flow`` for the twin partner) reaches only the Cartesian
    generator: every other flow defers the knob, so its mean mode is
    zeroed unconditionally (module docstring).

    *wall_smoothness* and *wall_confinement* reach only the wall-bounded
    generators, and their surfaces are the only ones offering them
    (``wall_fields()``): the triply-periodic family has neither a
    wall-normal filter nor a wall window, so it takes *smoothness*
    alone and both arguments are ignored for it.

    Requires JAX to be configured and the parameter singletons set (the
    geometry ``fourier`` singleton is built lazily by the dispatched
    generator's import).
    """
    system = params.phys.system
    if system not in periodic_systems:
        assert 0 < smoothness < 1, (
            "0 < smoothness < 1 required for wall-bounded random states"
        )
        assert 0 < wall_smoothness < 1, (
            "0 < wall_smoothness < 1 required for wall-bounded random states"
        )
        assert wall_confinement >= 0, "wall_confinement must be >= 0"
    if system in cartesian_systems:
        return generate_cartesian(
            amplitude,
            smoothness,
            wall_smoothness,
            wall_confinement,
            seed,
            mean_flow,
        )
    # Rheology before geometry: the viscoelastic systems are members of
    # their geometry's list too, and need the 9-component builder (see
    # ``flows.registry``).
    if system in annular_viscoelastic_systems:
        return generate_viscoelastic_dean(
            amplitude,
            params.init.random_conformation_amplitude,
            smoothness,
            wall_smoothness,
            wall_confinement,
            seed,
        )
    if system in cylindrical_viscoelastic_systems:
        return generate_viscoelastic_pipe(
            amplitude,
            params.init.random_conformation_amplitude,
            smoothness,
            wall_smoothness,
            wall_confinement,
            seed,
        )
    if system in cylindrical_systems:
        return generate_cylindrical(
            amplitude, smoothness, wall_smoothness, wall_confinement, seed
        )
    if system in annular_systems:
        state = generate_annular(
            amplitude, smoothness, wall_smoothness, wall_confinement, seed
        )
        if system == "dean":
            state = add_dean_laminar(state)
        return state
    if system in periodic_systems:
        return generate_triply_periodic(amplitude, smoothness, seed)
    raise ValueError(f"Unknown system: {system}")
