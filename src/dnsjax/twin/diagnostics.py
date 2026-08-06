r"""Twin-run difference-field diagnostics (Cartesian wall-bounded).

Online diagnostics of the difference field
`$\Delta\mathbf{u} = \mathbf{u}^{(2)} - \mathbf{u}^{(1)}$` between the
two DNS states the ``dnsjax-twin`` driver (:mod:`dnsjax.twin.driver`)
steps
in lockstep, following the methodology of Egerique-de-la-Concha &
Hwang, *J. Fluid Mech.* **1036**, A52 (2026).  Both states store the
spectral perturbation about the *same* laminar base flow, so
``state2 - state1`` is exactly the spectral `$\Delta\mathbf{u}$` --
the base flow cancels identically and every solver-measure norm
helper (:func:`~dnsjax.geometries.wall_bounded._base.get_norm2`)
applies to it unchanged.

Component decomposition
-----------------------
Fields are split by their wall-parallel Fourier support into the
three components of the reference (mean / streak / streamwise-varying
triad):

- mean `$\Delta U$`: the `$(k_z, k_x) = (0, 0)$` mode
  (``fourier.mean_mask``, one-hot by the padding-slot invariant --
  see "Mean mode and padding modes" in
  ``geometries/wall_bounded/CLAUDE.md``);
- streaks `$\Delta u_1$`: `$k_x = 0$`, `$k_z \ne 0$` (the
  streamwise-averaged fluctuation);
- streamwise-varying `$\Delta u_2$`: `$k_x \ne 0$`.

The three masks partition the whole mode grid: spectral padding
slots carry nonzero placeholder wavenumbers
(:func:`dnsjax.operators.pad_harmonics`), so `$k_x$` padding lands in
the `$\Delta u_2$` mask and `$k_z$` padding (at `$k_x = 0$`) in the
`$\Delta u_1$` mask -- both weight identically-zero state entries and
are inert.  Consequently

.. math::
    E_{\Delta U} + E_{\Delta u_1} + E_{\Delta u_2} = E_\Delta

holds to rounding (a guard in ``tests/test_twin_unit.py``), where
each energy is the volume-averaged
`$E_X = \|X\|^2 / 2$` in the solver measure (Parseval over the
wall-parallel modes with the real-FFT weight ``k_metric``, quadrature
``y_weights`` wall-normally, divided by ``derived_params.volume_fac``
-- identical to every flow's ``get_perturbation_energy``).
`$E_{\Delta u_1}$` is additionally split per velocity component
(``E_du1_x`` / ``E_du1_y`` / ``E_du1_z``): the streamwise dominance
of the streak difference field is the lift-up signature (fig. 11 of
the paper).

Budget terms
------------
:func:`twin_budget` evaluates the volume-averaged energy budget of
the three components (eqs. 2.7-2.17 of the paper): 12 production and
12 transport terms of the form

.. math::
    -\langle \mathbf{a} \cdot (\mathbf{b} \cdot \nabla)
    \mathbf{c} \rangle,

with `$(\mathbf{a}, \mathbf{b}, \mathbf{c})$` triples over the six
decomposed fields -- `$\Delta U, \Delta u_1, \Delta u_2$` (``dU`` /
``du1`` / ``du2``) and the reference's `$U^{(1)}, u_1^{(1)},
u_2^{(1)}$` (``rU`` / ``ru1`` / ``ru2``; `$U^{(1)}$` *includes the
laminar base profile*, so the terms are those of the total field) --
plus the three dissipations
`$\epsilon_{\Delta X} = -\langle \Delta X \cdot \nabla^2 \Delta X
\rangle / Re$` (the paper's eq. 2.17; see "Dissipation form" below)
and the consistency sums ``P_tot`` / ``T_tot`` / ``eps_tot``.  Column
names encode the triple, e.g. ``P_du1(du1,rU)`` is
`$-\langle \Delta u_1 \cdot (\Delta u_1 \cdot \nabla) U^{(1)}
\rangle$`.  The transport terms cancel pairwise by parts (each
advector appears symmetrically), so ``T_tot`` vanishes up to spatial
truncation; per component,
`$\partial_t E_X = P_X + T_X - \epsilon_X$` closes up to the
(pressure-projection + quadrature/FD-adjointness) truncation error
and the `$O(\Delta t^2)$` stepping error -- the guard in
``tests/test_twin_driver.py``.

Four evaluation classes, the first three FFT-free:

- **c mean** (`$\mathbf{c} \in \{U^{(1)}, \Delta U\}$`, 7 terms):
  `$(\mathbf{b}\cdot\nabla)\mathbf{c} = b_y\, \partial_y c_i(y)$`,
  so the term is a per-`$y$` Parseval cross-mean of `$(a_i, b_y)$`
  against `$\partial_y c_i$` -- no transform.
- **b mean** (`$\mathbf{b} = \Delta U$`, 2 terms): advection by a
  `$y$`-profile is diagonal in `$(k_z, k_x)$`:
  `$(i k_x b_x + i k_z b_z)\hat{c} + b_y \partial_y \hat{c}$`.
- **a mean** (`$\mathbf{a} = \Delta U$`, 6 terms): the `$(0,0)$`
  projection of the quadratic `$(\mathbf{b}\cdot\nabla)\mathbf{c}$`
  is a Parseval cross of `$\nabla\mathbf{c}$` with `$\mathbf{b}$`.
- **triple-fluctuating** (9 terms):
  `$\mathbf{q} = (\mathbf{b}\cdot\nabla)\mathbf{c}$` is formed on
  the padded physical grid (alias-free for a quadratic product),
  transformed back, and paired spectrally with
  `$\hat{\mathbf{a}}$` -- fully alias-controlled, never a third
  physical field.  Pairs are grouped by `$\mathbf{c}$` (one gradient
  set each) with the three advector fields' physical forms cached:
  69 single-field transforms per sample, ~the FFT cost of 5-10
  steps.  ``solver.rhs_transform_chunks`` bounds the transform
  transient via :func:`dnsjax.fft.chunked_transform`.

`$U^{(1)}$` and `$\Delta U$` are needed only as `$(3, N_y)$`
profiles in the a/b/c mean slots (no advecting-`$U^{(1)}$` term
exists: self-advection contributes no energy and is excluded from
the paper's lists), except `$\Delta U$`'s full masked field, which
`$\epsilon_{\Delta U}$` needs anyway.

Dissipation form
----------------
`$\epsilon_{\Delta X}$` is evaluated in the discrete-Laplacian
(operator) form `$-\langle \Delta X \cdot (\nabla_h^2 + D_2) \Delta
X\rangle / Re$` -- the operator the solver's implicit viscous update
actually applies -- rather than the positive-definite quadratic form
`$\langle |\nabla \Delta X|^2 \rangle / Re$` of
:func:`~dnsjax.geometries.wall_bounded._base.get_pert_enstrophy`
(which ``get_stats`` keeps).  Continuously the two coincide; the
discrete pair is not summation-by-parts in the quadrature inner
product, so they differ by
`$\Delta X^{T}(D_1^{T} W D_1 + W D_2)\,\Delta X$`.  That defect is a
truncation error *of the resolved part only*: at ``fd_order = 8`` it
is `$<10^{-4}$` for a decaying wall-normal spectrum but `$\sim 40\,\%$`
for content at half the grid scale -- and a difference field is the
adverse case, since it re-populates the grid scale as the grid
refines (measured flat at `$\sim 3\,\%$` from `$N_y = 17$` to
`$257$`, against `$6\times10^{-3} \to 5\times10^{-10}$` for a fixed
smooth field).  Only the operator form -- the one the implicit
viscous update actually applies -- therefore closes the discrete
budget `$\partial_t E_X = P_X + T_X - \epsilon_X$` against the
stepped states.  The price is positivity: unlike the quadratic form
this one is not positive-definite (the symmetric part of `$-W D_2$`
has genuinely negative eigenvalues), which is why ``get_stats``
keeps the other.

Frame invariance
----------------
A moving frame (``phys.u_grid``, e.g. the plane-Poiseuille default
`$2/3$`) shifts the streamwise mean of *both* states by the same
constant, which cancels in `$\Delta\mathbf{u}$`; the only remaining
carrier is the `$U^{(1)}$` profile, which enters every budget term
through `$\partial_y$` alone.  All quantities here are therefore
frame-invariant and need no ``u_grid`` handling.

Sharding
--------
The masks derive from ``fourier.kx`` (spec ``P(None, None, a1)``) and
``fourier.mean_mask`` (``P(None, a0, a1)``) through *binary* ops
only, which infer the combined partition spec -- the
``jnp.broadcast_to``-keeps-the-source-spec trap (see the precedent
note in ``cylindrical.py``, ``_imm_iteration_vw``) cannot arise
because no mask is ever materialised standalone at full shape.
Reductions are plain ``get_norm2`` sums over the sharded axes;
outputs are replicated scalars.
"""

import importlib

from jax import Array, jit, lax, shard_map
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P

from ..fft import chunked_transform
from ..flows.registry import cartesian_systems, spec_for
from ..geometries.wall_bounded._base import (
    apply_y_matrix,
    extract_mean_mode,
    get_inprod,
    get_norm2,
    integrate_scalar,
    phys_to_spec,
    spec_to_phys,
)
from ..geometries.wall_bounded.cartesian import Fourier, fourier
from ..parameters import derived_params, params
from ..sharding import sharding

if params.phys.system not in cartesian_systems:  # pragma: no cover
    raise RuntimeError(
        "dnsjax.twin.diagnostics supports the Cartesian wall-bounded "
        f"flows only (system {params.phys.system!r}); the [twin] "
        "surface should have rejected this configuration."
    )

#: The selected flow's module (shared singletons with the driver via
#: the import cache); its ``flow`` instance carries the grid
#: quadrature ``y_weights`` (and, for the budget terms, ``D1`` and
#: the laminar base profile).
_flow_mod = importlib.import_module(spec_for(params.phys.system).flow_module)
flow = _flow_mod.flow


def component_masks(fourier_: Fourier) -> tuple[Array, Array, Array]:
    r"""The mean / streak / streamwise-varying mode masks.

    Returns ``(m_mean, m_u1, m_u2)`` boolean masks broadcastable
    against the spectral state's trailing ``(N_y, N_{k_z}, N_{k_x})``
    axes (shapes ``(1, N_{k_z}, N_{k_x})``, ditto, and
    ``(1, 1, N_{k_x})``).  Built from ``fourier_`` fields through
    binary ops only (see the module docstring's sharding note); cheap
    enough to rebuild inside every jitted diagnostic, keeping the
    jaxprs free of captured device-array constants.
    """
    m_mean = fourier_.mean_mask
    m_u1 = (fourier_.kx == 0) & ~m_mean
    m_u2 = fourier_.kx != 0
    return m_mean, m_u1, m_u2


@jit
def _twin_energies_jit(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> dict[str, Array]:
    r"""Difference-field component energies (see the module docstring).

    - ``E_d``: total `$E_\Delta = \|\Delta\mathbf{u}\|^2/2$`.
    - ``E_dU`` / ``E_du1`` / ``E_du2``: the mean / streak /
      streamwise-varying components (computed independently; their
      sum equals ``E_d`` to rounding).
    - ``E_du1_x`` / ``E_du1_y`` / ``E_du1_z``: `$E_{\Delta u_1}$`
      per velocity component.
    - ``E_ref``: the reference state's own `$E'$` (context for
      saturation levels and the laminarization read).

    Keys are chosen so their *sorted* order (the ``twin.dat`` column
    order -- dicts returned through ``jit`` are canonicalised, see
    :mod:`dnsjax.measurements`) groups the components readably.
    """
    k_metric = fourier_.k_metric
    w = flow_.y_weights
    delta = state2 - state1
    m_mean, m_u1, m_u2 = component_masks(fourier_)
    du1 = delta * m_u1
    return {
        "E_d": get_norm2(delta, k_metric, w) / 2,
        "E_dU": get_norm2(delta * m_mean, k_metric, w) / 2,
        "E_du1": get_norm2(du1, k_metric, w) / 2,
        "E_du1_x": get_norm2(du1[0:1], k_metric, w) / 2,
        "E_du1_y": get_norm2(du1[1:2], k_metric, w) / 2,
        "E_du1_z": get_norm2(du1[2:3], k_metric, w) / 2,
        "E_du2": get_norm2(delta * m_u2, k_metric, w) / 2,
        "E_ref": get_norm2(state1, k_metric, w) / 2,
    }


def twin_energies(state1: Array, state2: Array) -> dict[str, Array]:
    """Wrapper around ``_twin_energies_jit`` binding the singletons."""
    return _twin_energies_jit(state1, state2, fourier, flow)


# ── Budget terms (see the module docstring's "Budget terms") ─────────

#: The paper's production triples ``(a, b, c)`` of
#: `$-\langle a \cdot (b \cdot \nabla) c \rangle$` (eqs. 2.11-2.13),
#: with ``d*`` the difference components and ``r*`` the reference's.
_PRODUCTION: tuple[tuple[str, str, str], ...] = (
    ("dU", "dU", "rU"),
    ("dU", "du1", "ru1"),
    ("dU", "du2", "ru2"),
    ("du1", "du1", "rU"),
    ("du1", "dU", "ru1"),
    ("du1", "du1", "ru1"),
    ("du1", "du2", "ru2"),
    ("du2", "dU", "ru2"),
    ("du2", "du1", "ru2"),
    ("du2", "du2", "rU"),
    ("du2", "du2", "ru1"),
    ("du2", "du2", "ru2"),
)

#: The transport triples (eqs. 2.14-2.16).  Each advector ``b`` acts
#: symmetrically on an ``(a, c)`` pair across two rows, so the twelve
#: terms cancel pairwise by parts (``T_tot`` ~ 0).
_TRANSPORT: tuple[tuple[str, str, str], ...] = (
    ("dU", "ru1", "du1"),
    ("dU", "du1", "du1"),
    ("dU", "ru2", "du2"),
    ("dU", "du2", "du2"),
    ("du1", "ru1", "dU"),
    ("du1", "du1", "dU"),
    ("du1", "ru2", "du2"),
    ("du1", "du2", "du2"),
    ("du2", "ru2", "dU"),
    ("du2", "du2", "dU"),
    ("du2", "ru2", "du1"),
    ("du2", "du2", "du1"),
)

#: Fields that are pure mean profiles (see the module docstring).
_MEANS: frozenset[str] = frozenset({"dU", "rU"})


def budget_names() -> list[str]:
    """The ``twin_budget`` keys (unsorted; JIT sorts the columns)."""
    names = [
        f"{kind}_{a}({b},{c})"
        for kind, table in (("P", _PRODUCTION), ("T", _TRANSPORT))
        for a, b, c in table
    ]
    names += ["eps_dU", "eps_du1", "eps_du2", "P_tot", "T_tot", "eps_tot"]
    return names


@jit
def _twin_budget_jit(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> dict[str, Array]:
    r"""The 24 advective terms, 3 dissipations, and consistency sums.

    Term math, evaluation classes, and naming: the module docstring.
    All Python control flow below is trace-time (static tables).
    """
    k_metric = fourier_.k_metric
    kx = fourier_.kx
    kz = fourier_.kz
    D1 = flow_.D1
    w = flow_.y_weights
    vf = derived_params.volume_fac
    re = params.phys.re

    delta = state2 - state1
    m_mean, m_u1, m_u2 = component_masks(fourier_)
    full = {
        "dU": delta * m_mean,
        "du1": delta * m_u1,
        "du2": delta * m_u2,
        "ru1": state1 * m_u1,
        "ru2": state1 * m_u2,
    }
    prof = {
        "dU": extract_mean_mode(delta).real,
        "rU": extract_mean_mode(state1).real + flow_.base_flow[:, :, 0, 0],
    }

    def d_dy_prof(p: Array) -> Array:
        """FD wall-normal derivative of a ``(3, Ny)`` profile."""
        return jnp.einsum("ij,cj->ci", D1, p)

    def xz_mean_cross(f: Array, g: Array) -> Array:
        r"""``(C, Ny)`` profile of the `$xz$`-mean of the product of
        the real fields with spectral coefficients *f* (``(C,...)``)
        and *g* (``(1,...)``, broadcast) -- Parseval with the
        real-FFT weight."""
        return jnp.sum(k_metric * (f * jnp.conj(g)).real, axis=(2, 3))

    def grad_spec(c: Array) -> tuple[Array, Array, Array]:
        return 1j * kx * c, apply_y_matrix(D1, c), 1j * kz * c

    def term_c_mean(a: str, b: str, c: str) -> Array:
        dyc = d_dy_prof(prof[c])
        cross = xz_mean_cross(full[a], full[b][1:2])
        return -integrate_scalar(jnp.sum(dyc * cross, axis=0), w) / vf

    def term_b_mean(a: str, b: str, c: str) -> Array:
        bx = prof[b][0][:, None, None]
        by = prof[b][1][:, None, None]
        bz = prof[b][2][:, None, None]
        adv = 1j * (bx * kx + bz * kz) * full[c] + by * apply_y_matrix(
            D1, full[c]
        )
        return -get_inprod(full[a], adv, k_metric, w)

    def term_a_mean(a: str, b: str, c: str) -> Array:
        dxc, dyc, dzc = grad_spec(full[c])
        mean_prof = (
            xz_mean_cross(dxc, full[b][0:1])
            + xz_mean_cross(dyc, full[b][1:2])
            + xz_mean_cross(dzc, full[b][2:3])
        )
        return -integrate_scalar(jnp.sum(prof[a] * mean_prof, axis=0), w) / vf

    out: dict[str, Array] = {}

    # Triple-fluctuating terms: grouped by c (one gradient set each),
    # the advector physical forms cached across the pass.
    fluct = [
        (kind, a, b, c)
        for kind, table in (("P", _PRODUCTION), ("T", _TRANSPORT))
        for a, b, c in table
        if not ({a, b, c} & _MEANS)
    ]
    b_names = tuple(dict.fromkeys(t[2] for t in fluct))
    b_stack = jnp.concatenate([full[n] for n in b_names], axis=0)
    b_phys_all = chunked_transform(spec_to_phys, b_stack)
    b_phys = {
        n: b_phys_all[3 * i : 3 * (i + 1)] for i, n in enumerate(b_names)
    }
    for c in dict.fromkeys(t[3] for t in fluct):
        grad_stack = jnp.concatenate(grad_spec(full[c]), axis=0)
        # Rows [j * 3 + i] = the j-derivative of component i.
        grad_phys = chunked_transform(spec_to_phys, grad_stack)
        for b in dict.fromkeys(t[2] for t in fluct if t[3] == c):
            q_phys = jnp.stack(
                [
                    sum(b_phys[b][j] * grad_phys[3 * j + i] for j in range(3))
                    for i in range(3)
                ]
            )
            q_spec = chunked_transform(phys_to_spec, q_phys)
            for kind, a, b_t, c_t in fluct:
                if (b_t, c_t) == (b, c):
                    out[f"{kind}_{a}({b},{c})"] = -get_inprod(
                        full[a], q_spec, k_metric, w
                    )

    # Mean-slot terms (FFT-free classes; priority c > b > a keeps the
    # dispatch unambiguous for multi-mean triples).
    for kind, table in (("P", _PRODUCTION), ("T", _TRANSPORT)):
        for a, b, c in table:
            name = f"{kind}_{a}({b},{c})"
            if name in out:
                continue
            if c in _MEANS:
                out[name] = term_c_mean(a, b, c)
            elif b in _MEANS:
                out[name] = term_b_mean(a, b, c)
            else:
                out[name] = term_a_mean(a, b, c)

    # Dissipations (the discrete-Laplacian form -- the module
    # docstring's "Dissipation form") and the consistency sums.
    for x in ("dU", "du1", "du2"):
        lap = -fourier_.k2 * full[x] + apply_y_matrix(flow_.D2, full[x])
        out[f"eps_{x}"] = -get_inprod(full[x], lap, k_metric, w) / re
    out["P_tot"] = sum(out[f"P_{a}({b},{c})"] for a, b, c in _PRODUCTION)
    out["T_tot"] = sum(out[f"T_{a}({b},{c})"] for a, b, c in _TRANSPORT)
    out["eps_tot"] = out["eps_dU"] + out["eps_du1"] + out["eps_du2"]
    return out


def twin_budget(state1: Array, state2: Array) -> dict[str, Array]:
    """Wrapper around ``_twin_budget_jit`` binding the singletons."""
    return _twin_budget_jit(state1, state2, fourier, flow)


# ── (kz, kx) energy spectra ──────────────────────────────────────────


def _mode_energy_replicated(field: Array, w: Array, k_metric: Array) -> Array:
    r"""Per-mode energy of a spectral field, replicated across devices.

    `$E(k_z, k_x) = \tfrac{1}{2}\,\mathrm{metric}\,
    \int |\hat{u}|^2\, w\, \mathrm{d}y \,/\, V$` summed over the
    velocity components -- so summing the returned array over the
    true modes reproduces the total energy exactly (the twin.dat
    convention).  Each device reduces its own ``(k_z, k_x)`` tile and
    scatters it into a zero global-shape array at its mesh position;
    a ``psum`` over both mesh axes assembles the **replicated**
    global spectrum (the disjoint-tile analogue of
    ``extract_mean_mode``) -- required because the writer's rank-0
    host transfer needs a fully-addressable array under multi-process
    launches.  Shape ``(N_{k_z}, N_{k_x})`` *padded* sizes; the
    padding rows/columns weight zero data and are stripped by the
    caller.
    """
    nz_spec, nx_spec = sharding.spec_shape[1], sharding.spec_shape[2]
    vf = derived_params.volume_fac

    def _local(shard: Array, w_loc: Array, k_metric_loc: Array) -> Array:
        e_loc = (
            jnp.einsum("j,cjkl->kl", w_loc, jnp.abs(shard) ** 2)
            * k_metric_loc[0]
        )
        row0 = lax.axis_index("np0") * e_loc.shape[0]
        col0 = lax.axis_index("np1") * e_loc.shape[1]
        full = jnp.zeros((nz_spec, nx_spec), dtype=e_loc.dtype)
        full = lax.dynamic_update_slice(full, e_loc, (row0, col0))
        return lax.psum(full, ("np0", "np1"))

    gathered = shard_map(
        _local,
        mesh=sharding.mesh,
        in_specs=(
            sharding.spec_vector_shard,
            P(None),
            P(None, None, sharding.a1),
        ),
        out_specs=P(None, None),
    )(field, w, k_metric)
    return gathered / (2.0 * vf)


@jit
def _twin_spectra_jit(
    state1: Array, state2: Array, fourier_: Fourier, flow_: object
) -> dict[str, Array]:
    r"""``(k_z, k_x)`` energy spectra of the difference and reference.

    ``e_delta`` is the per-mode `$E_\Delta(k_z, k_x)$` and ``e_ref``
    the reference state's own spectrum (their ratio
    `$E_\Delta / 2 E^{(1)}$` is the offline decorrelation measure:
    fully decorrelated independent fields give 1).  True modes only
    (padding stripped); summing ``e_delta`` reproduces ``twin.dat``'s
    ``E_d`` to rounding (a ``tests/test_twin_unit.py`` guard).
    """
    n2 = params.res.nz - 1
    n3 = params.res.nx // 2
    w = flow_.y_weights
    k_metric = fourier_.k_metric
    delta = state2 - state1
    return {
        "e_delta": _mode_energy_replicated(delta, w, k_metric)[:n2, :n3],
        "e_ref": _mode_energy_replicated(state1, w, k_metric)[:n2, :n3],
    }


def twin_spectra_2d(state1: Array, state2: Array) -> dict[str, Array]:
    """Wrapper around ``_twin_spectra_jit`` binding the singletons."""
    return _twin_spectra_jit(state1, state2, fourier, flow)
