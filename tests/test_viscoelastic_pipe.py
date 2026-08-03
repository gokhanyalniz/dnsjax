r"""Unit tests for the viscoelastic (sPTT) cylindrical (pipe) geometry.

Covers what the port from the annular sPTT geometry actually changed
(see :mod:`dnsjax.geometries.wall_bounded.cylindrical_viscoelastic`);
the geometry-free algebra it shares with the annulus -- the spin
`$\leftrightarrow$` physical maps, the Frobenius weights, the pointwise
RHS kernel -- is guarded once, in ``test_viscoelastic.py``.

1. **Axis parity.**  Every radial derivative the viscoelastic pipe
   takes carries a per-slot ghost sign `$(-1)^{m+s}$`.  Two tests pin
   it: the per-slot signs of the fused RHS batch and of
   `$\nabla\cdot c$` reproduce, mode by mode, the explicitly assembled
   even/odd parity-reduced `$D_1$` (which also catches a ghost
   scatter-add landing on the component axis instead of the radial
   one -- a silent corruption of the first `$g$` slots), and the
   physical-basis parity classes are shown to be the ones the mirrored
   continuum field obeys.
2. **Spin-diagonal tensor Laplacian** equals an independently coded
   coupled reference built from the same parity-reduced matrices, with
   the `$\tfrac1{r^2}(\mathcal R + im)^2$` angular part carried by the
   6x6 basis-rotation generator.
3. **Laminar fixed point**, and exactly how far it reaches.  The
   conformation slice of the RHS vanishes at every `$\epsilon$`; a full
   predictor/corrector step reproduces the pair at `$\epsilon = 0$`;
   and its diagnostics satisfy `$I = D_s - W_p$` with the analytically
   known values `$2/\mathrm{Re}$`, `$2\beta/\mathrm{Re}$`,
   `$-2(1-\beta)/\mathrm{Re}$`.  The profile itself is re-derived
   independently from the sPTT equilibrium equations.  The axial
   *momentum* balance is pinned separately, and closes **only** at
   `$\epsilon = 0$`: `$W = 1 - r^2$` neglects the polymer's shear
   thinning, so at `$\epsilon > 0$` the pair is not a steady state.
4. `$H_c$` **band-vs-dense parity** including the single narrow
   Laplacian BC wall row, per spin slot, plus an independent pin of the
   flow's `$(-1)^{m+s}$` parity-band selector.
5. The **adapter surface** the shared stepper
   (:mod:`dnsjax.geometries.wall_bounded._viscoelastic_stepping`)
   dispatches on: the pipe zeroes exactly one `$H_c$` wall row (the
   annulus two), and importing this geometry must not build the
   annular one.
6. The **probe stream**'s 9-component labels and column conversion for
   the pipe.
7. **Fused-RHS transform-count** guard.

Run as a script via ``uv run python tests/test_viscoelastic_pipe.py``.
"""

from __future__ import annotations

import subprocess
import sys

# Select the JAX backend from --dist.platform (default cpu) before the
# geometry import below builds sharding.  --dist.platform cuda runs the
# Pallas Hc parity on a GPU.
from dnsjax.bootstrap import (  # noqa: E402
    configure_jax_platform,
    platform_from_argv,
)
from dnsjax.parameters import (  # noqa: E402
    Parameters,
    padded_res,
    params,
    update_parameters,
)

sys.stdout.reconfigure(line_buffering=True)

configure_jax_platform(platform_from_argv())

# Module config: viscoelastic-pipe with epsilon = kappa = 0 (the exact
# discrete laminar fixed point, group 3) and a modest Weissenberg
# number.  el = wi keeps the derived Re = wi / el = 1.  The H_c
# operator tests build their own kappa > 0 operators, independent of
# the flow's (kappa = 0, so ``flow.Hc_op is None``).
update_parameters(
    Parameters(
        phys={
            "system": "viscoelastic-pipe",
            "el": 5.0,
            "wi": 5.0,
            "beta": 0.8,
            "epsilon": 0.0,
            "kappa": 0.0,
        },
        res={
            "nx": 8,
            "ny": 25,
            "nz": 8,
            "fd_order": 4,
            "double_precision": True,
        },
    )
)
padded_res.set_padded_resolution(params)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.flows.wall_bounded.viscoelastic_pipe import (  # noqa: E402
    _laminar_state,
    flow,
    get_stats,
    predict_and_fully_correct,
)
from dnsjax.geometries.wall_bounded._viscoelastic_common import (  # noqa: E402
    PHYS_COMBO_SPIN,
    TENSOR_SPIN,
    from_spin_basis,
    solve_ptt_f,
    to_spin_basis,
)
from dnsjax.geometries.wall_bounded.cylindrical import (  # noqa: E402
    _parity_y_matvec,
    build_parity_reduced_matrices,
)
from dnsjax.geometries.wall_bounded.cylindrical_viscoelastic import (  # noqa: E402
    _DIV_C_SPIN,
    _DR_BATCH_SPIN,
    _build_Hc_band_gpu,
    _build_Hc_dense_gpu,
    _div_c,
    _get_rhs,
    _parity_signs,
    _tensor_laplacian_spin,
    fourier,
    viscoelastic_laminar_profiles,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import _banded_from_dense  # noqa: E402

# 6x6 basis-rotation generator R in the physical tensor-component order
# (c_rr, c_thth, c_rth, c_rz, c_thz, c_zz), i.e. the theta-derivative
# action on the orthonormal tensor basis.  Its eigenvalues are the spin
# weights i*s, s in {0, 0, +-1, +-2}.
_R_GEN = np.array(
    [
        [0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


# ── helpers ──────────────────────────────────────────────────────────


def _random_tensor(rng, shape):
    """Random complex tensor of *shape* (real + imaginary parts)."""
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _count_fft_prims(jaxpr) -> int:
    """Total ``fft`` primitives in *jaxpr* (recurses into nested jaxprs)."""
    n = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "fft":
            n += 1
        for val in eqn.params.values():
            vals = val if isinstance(val, (tuple, list)) else (val,)
            for v in vals:
                sub = getattr(v, "jaxpr", None) or (
                    v if hasattr(v, "eqns") else None
                )
                if sub is not None:
                    n += _count_fft_prims(sub)
    return n


def _parity_matrices():
    """Assembled even / odd parity-reduced ``D1`` on the flow's grid."""
    d1_even, _, d1_odd, _, _, _ = build_parity_reduced_matrices(
        np.asarray(flow.rs), params.res.fd_order
    )
    return np.asarray(d1_even), np.asarray(d1_odd)


def _reference_dr(field: np.ndarray, spins: np.ndarray) -> np.ndarray:
    r"""Per-slot, per-mode radial derivative from explicit matrices.

    Chooses `$D_{1,\mathrm{even}}$` or `$D_{1,\mathrm{odd}}$` for slot
    ``c`` and azimuthal mode ``m`` by the parity of ``m + spins[c]``.
    Deliberately independent of the ghost-correction machinery the
    solver uses.
    """
    d1_even, d1_odd = _parity_matrices()
    m_vals = np.asarray(fourier.m).ravel()
    out = np.empty_like(field)
    for c, s in enumerate(spins):
        for mi, m in enumerate(m_vals):
            d1 = d1_even if (int(m) + int(s)) % 2 == 0 else d1_odd
            out[c, :, mi, :] = d1 @ field[c, :, mi, :]
    return out


# ── tests ────────────────────────────────────────────────────────────

# Group A: axis parity


def test_parity_signs_match_the_reduced_matrices() -> None:
    r"""Every per-slot ghost sign selects the operator its class needs.

    The two sign tables the RHS uses -- the fused 9-slot batch
    (velocity triad + physical tensor combos) and the 3-slot
    `$\nabla\cdot c$` batch -- are pushed through the *solver's*
    :func:`_parity_y_matvec` and compared against explicitly assembled
    `$D_{1,\mathrm{even}}$` / `$D_{1,\mathrm{odd}}$` matmuls chosen per
    mode.  A wrong entry picks the wrong ghost sign near the axis; and
    because the ghost correction touches only the first `$g$` rows, a
    scatter-add landing on the *component* axis instead of the radial
    one would corrupt exactly the first `$g$` slots -- silently, with
    no shape error.  Both failures show up here.
    """
    Nr = params.res.ny
    Nm, Nkz = sharding.nz_spec, sharding.nx_spec
    rng = np.random.default_rng(11)

    for spins in (_DR_BATCH_SPIN, _DIV_C_SPIN):
        n = len(spins)
        host = _random_tensor(rng, (n, Nr, Nm, Nkz))
        # y-leading, exactly as the RHS stacks it.
        field = jnp.stack([jnp.asarray(host[c]) for c in range(n)], axis=1)
        got = _parity_y_matvec(
            flow.D1_pos,
            flow.D1_ghost,
            field,
            _parity_signs(spins, fourier),
            component_axis=1,
        )
        got = np.asarray(jnp.swapaxes(got, 0, 1))
        assert_allclose(
            got,
            _reference_dr(host, spins),
            atol=1e-12,
            err_msg=f"per-slot parity D1 mismatch for spins {spins}",
        )
        # The two classes must actually differ on this data, or the
        # comparison above would pass for any sign table.
        wrong = _reference_dr(host, spins + 1)
        assert not np.allclose(got, wrong, atol=1e-8)


def test_spin_combos_carry_the_parity_their_operator_assumes() -> None:
    r"""The spin combos of a correctly-mirrored physical tensor field
    carry the parity `$(-1)^{m+s}$` that :data:`TENSOR_SPIN` claims.

    A rank-2 component picks up one sign flip per index in
    `$\{r, \theta\}$` when the axis is crossed (`$\hat e_r,
    \hat e_\theta \to -\hat e_r, -\hat e_\theta$`), on top of the single
    `$(-1)^m$` of the Fourier mode -- that index count is what
    :data:`PHYS_COMBO_SPIN` encodes.  Building each physical component
    on the mirrored grid with exactly that symmetry, this checks that
    the six spin projections come out with the parity their `$H_c$`
    band and their ghost sign are chosen for.  A continuum statement:
    independent of any particular ``D1``.
    """
    rs = np.asarray(flow.rs)

    def _mirror(vals, sign):
        return np.concatenate([sign * vals[::-1], vals])

    prof = np.exp(-2 * rs**2) * (1 + rs)
    # Distinct profiles so an accidental cancellation cannot hide a
    # wrong class (e.g. c_rr - c_thth vanishing identically).
    profs = [prof * (1.0 + 0.3 * k) ** 2 for k in range(6)]

    for m in (1, 2, 3):
        pm = (-1.0) ** m
        # Physical combo order (c_rr, c_thth, c_rth, c_rz, c_thz, c_zz);
        # parity = (-1)^m x (-1)^(index count in {r, theta}).
        comps = [
            _mirror(p, pm * (-1.0) ** (int(s) % 2))
            for p, s in zip(profs, PHYS_COMBO_SPIN, strict=True)
        ]
        c_rr, c_thth, c_rth, c_rz, c_thz, c_zz = comps
        # Spin order (c_zz, c_z+, c_z-, c_+-, c_++, c_--).
        spin = [
            c_zz,
            c_rz + 1j * c_thz,
            c_rz - 1j * c_thz,
            c_rr + c_thth,
            (c_rr - c_thth) + 2j * c_rth,
            (c_rr - c_thth) - 2j * c_rth,
        ]
        for slot, (comp, s) in enumerate(zip(spin, TENSOR_SPIN, strict=True)):
            want = pm * (-1.0) ** (int(s) % 2)
            assert np.allclose(comp[::-1], want * comp), (m, slot)
            assert not np.allclose(comp[::-1], -want * comp), (m, slot)


# Group B: tensor Laplacian


def test_tensor_laplacian_diagonalization() -> None:
    r"""Spin-diagonal Laplacian == coupled physical-basis reference.

    The reference builds the angular part from the 6x6 generator
    `$\mathcal R$` rather than the analytic eigenvalue `$-(m+s)^2$`,
    and takes its radial derivatives with explicitly assembled
    parity-reduced matrices selected per `$(m + s)$` -- so it shares no
    machinery with the implementation beyond the grid.
    """
    Nr = params.res.ny
    m_vals = np.asarray(fourier.m).ravel()
    kz_vals = np.asarray(fourier.kz).ravel()
    Nm, Nkz = len(m_vals), len(kz_vals)

    d1_even, d1_odd = _parity_matrices()
    d2_even, d2_odd = (
        np.asarray(a)
        for a in (
            build_parity_reduced_matrices(
                np.asarray(flow.rs), params.res.fd_order
            )[1],
            build_parity_reduced_matrices(
                np.asarray(flow.rs), params.res.fd_order
            )[3],
        )
    )
    inv_r = np.asarray(flow.inv_r)
    inv_r2 = np.asarray(flow.inv_r2)

    rng = np.random.default_rng(2)
    spin_np = _random_tensor(rng, (6, Nr, Nm, Nkz))
    spin = jax.device_put(jnp.asarray(spin_np), sharding.spec_vector_shard)
    got = np.asarray(_tensor_laplacian_spin(spin, fourier, flow))

    # Reference: convert to the physical combo basis, apply the coupled
    # Laplacian there, convert back.
    c_zz, c_zp, c_zm, c_pm, c_pp, c_mm = spin_np
    d = (c_pp + c_mm) / 2
    cphys = np.stack(
        [
            c_pm / 2 + d / 2,  # c_rr
            c_pm / 2 - d / 2,  # c_thth
            -0.5j * (c_pp - c_mm) / 2,  # c_rth
            (c_zp + c_zm) / 2,  # c_rz
            -0.5j * (c_zp - c_zm),  # c_thz
            c_zz,
        ]
    )
    ref = np.zeros_like(cphys)
    eye6 = np.eye(6)
    for mi, m in enumerate(m_vals):
        for ki, kz in enumerate(kz_vals):
            for comp in range(6):
                # Radial part on the component's own parity class.
                s = int(PHYS_COMBO_SPIN[comp])
                even = (int(m) + s) % 2 == 0
                D1 = d1_even if even else d1_odd
                D2 = d2_even if even else d2_odd
                vec = cphys[comp, :, mi, ki]
                ref[comp, :, mi, ki] = (
                    D2 @ vec + inv_r * (D1 @ vec) - kz**2 * vec
                )
            gen2 = _R_GEN @ _R_GEN + 2j * m * _R_GEN - m**2 * eye6
            ref[:, :, mi, ki] += (gen2 @ cphys[:, :, mi, ki]) * inv_r2[None, :]

    r_rr, r_thth, r_rth, r_rz, r_thz, r_zz = ref
    ref_spin = np.stack(
        [
            r_zz,
            r_rz + 1j * r_thz,
            r_rz - 1j * r_thz,
            r_rr + r_thth,
            (r_rr - r_thth) + 2j * r_rth,
            (r_rr - r_thth) - 2j * r_rth,
        ]
    )
    scale = np.abs(ref_spin).max()
    err = np.abs(got - ref_spin).max() / scale
    assert err < 1e-10, f"spin-diagonal tensor Laplacian mismatch {err:.2e}"


# Group C: laminar fixed point


def test_laminar_profiles_solve_the_ptt_equilibrium() -> None:
    r"""The analytical laminar pair satisfies the sPTT equilibrium.

    Re-derives `$c$` from the equilibrium equations (independently of
    the builder's Newton solve) at a nonzero `$\epsilon$`, and checks
    the velocity is Hagen-Poiseuille with the discrete shear the
    builder claims.
    """
    rs = np.asarray(flow.rs)
    d1_even, _ = _parity_matrices()
    wi, eps = 5.0, 0.01
    prof = viscoelastic_laminar_profiles(rs, d1_even, wi, eps)
    u_z, c_zz, c_rz, c_rr, c_thth = (
        prof[0].real,
        prof[3].real,
        prof[4].real,
        prof[6].real,
        prof[7].real,
    )
    assert_allclose(u_z, 1.0 - rs**2, atol=1e-14)
    assert_allclose(prof[1], 0.0, atol=0)  # u_r
    assert_allclose(prof[2], 0.0, atol=0)  # u_theta
    assert_allclose(prof[5], 0.0, atol=0)  # c_theta_z
    assert_allclose(prof[8], 0.0, atol=0)  # c_r_theta
    assert_allclose(c_rr, 1.0, atol=1e-14)
    assert_allclose(c_thth, 1.0, atol=1e-14)

    # Equilibrium: f = 1 - 3 eps + eps tr c, c_rz = Wi S / f,
    # c_zz = 1 + 2 (Wi S)^2 / f^2, with the *discrete* shear.
    shear = d1_even @ (1.0 - rs**2)
    f = 1.0 - 3.0 * eps + eps * (c_rr + c_thth + c_zz)
    assert_allclose(c_rz, wi * shear / f, atol=1e-13)
    assert_allclose(c_zz, 1.0 + 2.0 * (wi * shear) ** 2 / f**2, atol=1e-13)
    # ... and the cubic the builder actually solves.
    assert_allclose(f**3 - f**2, 2.0 * eps * (wi * shear) ** 2, atol=1e-13)
    # eps = 0 collapses to f = 1 (the Oldroyd-B limit).
    assert_allclose(solve_ptt_f(np.zeros(4)), 1.0, atol=1e-15)


def test_laminar_conformation_rhs_vanishes() -> None:
    r"""At `$\kappa = 0$` the conformation RHS vanishes at the laminar
    pair, for every `$\epsilon$` -- the flow is unidirectional, so the
    advection and all but one stretching term drop out algebraically
    and the relaxation cancels the survivor."""
    state = to_spin_basis(_laminar_state)
    rhs = np.asarray(_get_rhs(state, fourier, flow))
    scale = float(np.abs(np.asarray(_laminar_state[3:])).max())
    err = np.abs(rhs[3:]).max() / scale
    assert err < 1e-12, f"conformation RHS at laminar = {err:.2e}"


def test_laminar_velocity_balance_closes_only_at_zero_epsilon() -> None:
    r"""The laminar pair balances *momentum* only at `$\epsilon = 0$`.

    The builder pairs the sPTT-equilibrium conformation with the
    **Newtonian** `$W = 1 - r^2$`, so at `$\epsilon > 0$` the polymer's
    shear thinning (`$f > 1$`) is unaccounted for and the axial balance
    `$\nu A_{\mathrm{base}} W + \tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}
    (\nabla\cdot c)_z + \Pi_z = 0$` carries a real residual -- the
    approximation the module docstring records.  Pinned here in both
    directions so the `$\epsilon = 0$` exactness cannot rot and a future
    exact profile turns the second bound into a tight one rather than a
    silent improvement.  NumPy only: `$\epsilon$` is an *argument* of
    the profile builder, so no reconfiguration of the module is needed.
    """
    rs = np.asarray(flow.rs)
    d1_even, d1_odd = _parity_matrices()
    a_even = np.asarray(flow.A_base_even)
    beta, re, wi = params.phys.beta, params.phys.re, params.phys.wi
    nu = beta / re
    coef = (1.0 - beta) / (re * wi)
    pi_z = 4.0 / re

    def _residual(eps: float) -> float:
        prof = viscoelastic_laminar_profiles(rs, d1_even, wi, eps)
        u_z, c_rz = prof[0].real, prof[4].real
        # m = k_z = 0, so (div c)_z is the radial part alone; c_rz is
        # the odd-parity class at m = 0.
        div_z = d1_odd @ c_rz + c_rz / rs
        resid = nu * (a_even @ u_z) + coef * div_z + pi_z
        return float(np.abs(resid).max() / pi_z)

    # At epsilon = 0 only FD truncation survives (~1e-12 here), which is
    # still six orders below the epsilon > 0 residual below.
    exact = _residual(0.0)
    assert exact < 1e-9, f"epsilon = 0 must balance exactly, got {exact:.2e}"
    # At the shipped default epsilon the residual is percent-level; the
    # loose window is what makes this a measurement rather than a
    # tautology (a correct profile would drop it to ~1e-12).
    got = _residual(1e-3)
    assert 1e-3 < got < 1.0, f"epsilon = 1e-3 residual {got:.2e}"
    print(f"  laminar axial-momentum residual / Pi_z: {got:.2e} at eps=1e-3")


def test_laminar_full_step_fixed_point() -> None:
    r"""A full predictor/corrector step reproduces the laminar state.

    Stronger than the RHS check: it also closes the *velocity* balance
    (the polymer-stress divergence against the solvent Laplacian and
    the body force `$\Pi_z = 4/\mathrm{Re}$`) through the influence
    matrix, at `$\epsilon = 0$` where `$W = 1 - r^2$` is the exact
    profile.
    """
    state = to_spin_basis(_laminar_state)
    stepped, err, _ = predict_and_fully_correct(jnp.copy(state))
    drift = float(jnp.abs(stepped - state).max())
    assert drift < 1e-12, f"laminar step drift {drift:.2e}"
    assert float(err) < 1e-12


def test_laminar_energy_balance() -> None:
    r"""The laminar diagnostics hit their analytical values.

    For `$W = 1 - r^2$` on `$\int_0^1 r\,dr = 1/2$`:
    `$I = \langle W\Pi_z\rangle = 2/\mathrm{Re}$`,
    `$D_s = \nu\langle|W'|^2\rangle = 2\beta/\mathrm{Re}$`, and
    `$W_p = -2(1-\beta)/\mathrm{Re}$` -- so the steady balance
    `$I = D_s - W_p$` closes, which is what the laminar smoke test
    asserts at runtime.
    """
    st = get_stats(_laminar_state)
    Re, beta = params.phys.re, params.phys.beta
    assert_allclose(float(st["I"]), 2.0 / Re, rtol=1e-12)
    assert_allclose(float(st["D_s"]), 2.0 * beta / Re, rtol=1e-10)
    assert_allclose(float(st["W_p"]), -2.0 * (1.0 - beta) / Re, rtol=1e-10)
    imbalance = abs(
        float(st["I"]) - (float(st["D_s"]) - float(st["W_p"]))
    ) / abs(float(st["I"]))
    assert imbalance < 1e-12, f"I != D_s - W_p ({imbalance:.2e})"
    assert float(st["E'"]) < 1e-28
    assert_allclose(float(st["Ub_z"]), 0.5, rtol=1e-12)  # laminar bulk


# Group D: H_c operator


def test_Hc_band_vs_dense() -> None:
    r"""Banded `$H_c$` == banded-from-dense, per spin slot.

    Covers the parity band selection and the single narrow Laplacian BC
    wall row (the pipe carries one, against the annulus's two; the axis
    is closed by the parity reduction, not a row).

    The per-spin base operators come from the flow's own
    ``hc_spin_bases`` adapter -- the pipe half of the surface the
    shared stepper dispatches on -- so the parity selection under test
    is the shipped one; a separate assertion below pins it against
    `$(-1)^{m+s}$` independently.
    """
    from dnsjax.geometries.wall_bounded._viscoelastic_common import (
        narrow_abase_wall_row,
    )

    d1_even, _ = _parity_matrices()
    narrowN = jnp.asarray(
        narrow_abase_wall_row(
            np.asarray(flow.rs), d1_even, params.res.fd_order, inner=False
        )
    )
    p = params.res.fd_order
    kappa, dt, c_impl = 5.0e-5, 0.01, 0.5
    m_s = fourier.m[0, ..., None]
    kz2_s = fourier.kz2[0, ..., None]
    # The one wall, as ``flow.hc_wall_rows()`` returns it (it cannot be
    # called here: the module kappa is 0, so the leaf is unset).
    walls = ((params.res.ny - 1, narrowN),)

    spins = (0, 1, -1, 2, -2)
    band_bases = flow.hc_spin_bases(fourier, spins, banded=True, p=p)
    dense_bases = flow.hc_spin_bases(fourier, spins, banded=False, p=p)

    # The selector obeys parity (-1)^{m+s}: a spin-s slot rides the
    # even-parity base exactly at the modes where m + s is even.
    m_even = np.asarray(fourier.m_is_even[0, :, 0]) > 0.5
    for s, dense_base in zip(spins, dense_bases, strict=True):
        want_even = m_even if s % 2 == 0 else ~m_even
        ref_base = np.where(
            want_even[:, None, None],
            np.asarray(flow.A_base_even),
            np.asarray(flow.A_base_odd),
        )
        assert_allclose(
            np.asarray(dense_base)[:, 0],
            ref_base,
            atol=0.0,
            err_msg=f"parity band selection, spin {s}",
        )

    for s, band_base, dense_base in zip(
        spins, band_bases, dense_bases, strict=True
    ):
        meff2 = (m_s + s) ** 2
        band = np.asarray(
            _build_Hc_band_gpu(
                band_base,
                walls,
                meff2,
                flow.inv_r2,
                kz2_s,
                dt,
                c_impl,
                kappa,
                p,
            )
        )
        dense = np.asarray(
            _build_Hc_dense_gpu(
                dense_base,
                walls,
                meff2,
                flow.inv_r2,
                kz2_s,
                dt,
                c_impl,
                kappa,
            )
        )
        ref = np.stack(
            [
                np.stack(
                    [
                        np.asarray(_banded_from_dense(dense[i, j], p))
                        for j in range(dense.shape[1])
                    ]
                )
                for i in range(dense.shape[0])
            ]
        )
        assert_allclose(band, ref, atol=1e-12, err_msg=f"H_c spin {s}")

    # The BC row is genuinely the narrow Laplacian, not an identity.
    assert np.count_nonzero(np.asarray(narrowN)) > 1


def test_hc_wall_rows_single_wall() -> None:
    r"""The pipe zeroes exactly **one** `$H_c$` RHS row, at `$r = 1$`.

    ``zero_hc_wall_rows`` is the adapter the shared ``_c_cn_update``
    applies for the `$\nabla^2 c = 0$` BC, and the wall count is the
    one thing the two geometries disagree on there (the annulus zeroes
    two).  Zeroing the axis row as well would impose a boundary
    condition the pipe does not have.  Every ``test_laminar_smoke``
    viscoelastic entry runs at `$\kappa = 0$`, where that branch is not
    reached at all, so pin it directly.
    """
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(11)
    R = jnp.asarray(_random_tensor(rng, (6, Nr, Nm, Nkz)))
    out = np.asarray(flow.zero_hc_wall_rows(R))

    zero_rows = [i for i in range(Nr) if not np.any(out[:, i])]
    assert zero_rows == [Nr - 1], f"zeroed radial rows {zero_rows}"
    assert_allclose(out[:, : Nr - 1], np.asarray(R)[:, : Nr - 1], atol=0.0)


def test_no_cross_geometry_import() -> None:
    """Importing the pipe sPTT geometry must not build the annulus.

    Each geometry's ``Fourier`` singleton (and its radial grid / FD
    matrices) is constructed at **import**, so a stray import across
    the two families would build a grid the flow never uses on every
    viscoelastic run.  The shared stepper
    (``_viscoelastic_stepping``) is written to depend on neither
    geometry, which is what keeps that true; this is the guard, in a
    fresh subprocess since the check is about import-time side effects.
    """
    src = (
        "import sys\n"
        "from dnsjax.bootstrap import configure_jax_platform\n"
        "from dnsjax.parameters import (\n"
        "    Parameters, padded_res, params, update_parameters)\n"
        "configure_jax_platform('cpu')\n"
        "update_parameters(Parameters(\n"
        "    phys={'system': 'viscoelastic-pipe'},\n"
        "    res={'nx': 8, 'ny': 13, 'nz': 8}))\n"
        "padded_res.set_padded_resolution(params)\n"
        "import dnsjax.geometries.wall_bounded.cylindrical_viscoelastic\n"
        "bad = sorted(m for m in sys.modules if 'annular' in m)\n"
        "print(' '.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True
    )
    assert r.returncode == 0, (
        f"annular modules imported: {r.stdout.strip()}\n{r.stderr[-800:]}"
    )


# Group E: probe stream and transform count


def test_probe_stream_component_basis() -> None:
    r"""The pipe's probe gather crosses the 9-component boundary and
    advertises the labels of the components it returns (checked against
    the analysis package's stored-component schema, so the two
    9-component surfaces cannot drift apart)."""
    from dnsjax.analysis._core import geometry_info
    from dnsjax.probes import _component_labels, build_mode_extractor

    info = geometry_info(params)
    assert info.family == "cylindrical"
    assert _component_labels(9) == list(info.components)

    rng = np.random.default_rng(5)
    shape = (9, params.res.ny, sharding.nz_spec, sharding.nx_spec)
    host = _random_tensor(rng, shape)
    state = jax.device_put(jnp.asarray(host), sharding.spec_vector_shard)
    modes = [(0, 0), (2, 1)]
    got = np.asarray(build_mode_extractor(modes)(state))
    for k, (i2, i3) in enumerate(modes):
        col = host[:, :, i2, i3]
        assert_allclose(
            got[k],
            np.asarray(from_spin_basis(jnp.asarray(col))),
            atol=1e-13,
            err_msg=f"mode ({i2},{i3})",
        )
        assert not np.allclose(got[k], col)  # not the identity


def test_fused_rhs_transform_count() -> None:
    r"""The 9-component RHS keeps a *bounded* (batched) FFT count.

    The fused evaluation does one batched inverse transform of ~36
    fields and one batched forward transform of the 9 outputs -- a
    small constant, **independent** of the field count.
    """
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(6)
    state = jax.device_put(
        jnp.asarray(_random_tensor(rng, (9, Nr, Nm, Nkz))),
        sharding.spec_vector_shard,
    )
    jaxpr = jax.make_jaxpr(lambda s: _get_rhs(s, fourier, flow))(state).jaxpr
    n_fft = _count_fft_prims(jaxpr)
    assert 0 < n_fft <= 8, (
        f"fused RHS FFT count {n_fft} not bounded -- per-field "
        "transforms (regression from the single batched pair)?"
    )


def test_tensor_spin_tables_agree() -> None:
    """The spin tables the operators and the RHS batches read agree.

    ``TENSOR_SPIN`` orders the *solver* slots and ``PHYS_COMBO_SPIN``
    the physical combos; only their parities are comparable, and the
    two orderings must describe the same six components.
    """
    # Solver order (c_zz, c_z+, c_z-, c_+-, c_++, c_--) vs physical
    # combo order (c_rr, c_thth, c_rth, c_rz, c_thz, c_zz).
    assert sorted(np.abs(TENSOR_SPIN) % 2) == sorted(
        np.abs(PHYS_COMBO_SPIN) % 2
    )
    # The fused batch is the velocity triad (u_r, u_th: odd; u_z: even)
    # followed by the physical combos.
    assert list(_DR_BATCH_SPIN[:3] % 2) == [1, 1, 0]
    assert list(_DR_BATCH_SPIN[3:]) == list(PHYS_COMBO_SPIN)
    # div(c) differentiates (c_rr, c_rth, c_rz).
    assert list(_DIV_C_SPIN % 2) == [0, 0, 1]


def test_consistent_imm_is_accepted() -> None:
    r"""The flow accepts ``res.consistent_imm`` and keeps the pipe's
    own grid/scheme check.

    The flag was rejected here for one commit, on a nonlinear blow-up
    measured at `$\mathrm{Re} \approx 1$`.  That blow-up was real but
    belonged to the cylindrical flag-on pass's lagged wall data, not to
    this flow -- it reproduces at `$\beta = 1$` and in the Newtonian
    pipe -- so rejecting it here while ``pipe`` offered the same scheme
    was asymmetric.  It is fixed in
    ``cylindrical._imm_iteration_vw``.  This asserts the surface, not
    the stability: no *validate* hook and no linear gate can see that
    class of defect; the nonlinear guard is the
    ``viscoelastic-pipe-consistent-imm`` entry in
    ``tests/test_random_smoke.py``.

    Called on a stand-in parameter object rather than the live
    singletons: this module configures those once at import, and a
    re-``update_parameters`` would mutate state the other tests share.
    """
    from types import SimpleNamespace

    from dnsjax.flows.registry import spec_for

    validate = spec_for("viscoelastic-pipe").validate
    assert validate is not None
    stand_in = SimpleNamespace(
        res=SimpleNamespace(consistent_imm=False),
        geo=SimpleNamespace(grid_type="half-cgl"),
        step=SimpleNamespace(scheme="iterative-cn"),
    )
    validate(stand_in, None)  # flag-off passes
    stand_in.res.consistent_imm = True
    validate(stand_in, None)  # and so does flag-on

    # The pipe's own half-CGL / scheme check still applies.
    stand_in.res.consistent_imm = False
    stand_in.step.scheme = "cnab2"
    try:
        validate(stand_in, None)
    except ValueError as exc:
        assert "half-cgl" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("the inherited grid check was dropped")


def test_div_c_matches_a_direct_reference() -> None:
    r"""`$\nabla\cdot c$` equals its component formulas evaluated with
    explicitly assembled parity-reduced matrices."""
    Nr = params.res.ny
    Nm, Nkz = sharding.nz_spec, sharding.nx_spec
    rng = np.random.default_rng(13)
    host = _random_tensor(rng, (6, Nr, Nm, Nkz))
    c_rr, c_thth, c_rth, c_rz, c_thz, c_zz = (
        jax.device_put(jnp.asarray(a), sharding.spec_scalar_shard)
        for a in host
    )
    got = [
        np.asarray(a)
        for a in _div_c(c_rr, c_thth, c_rth, c_rz, c_thz, c_zz, fourier, flow)
    ]

    dr = _reference_dr(np.stack([host[0], host[2], host[3]]), _DIV_C_SPIN)
    m = np.asarray(fourier.m)
    kz = np.asarray(fourier.kz)
    inv_r = np.asarray(flow.inv_r)[:, None, None]
    ref = [
        dr[0]
        + 1j * m * inv_r * host[2]
        + 1j * kz * host[3]
        + inv_r * (host[0] - host[1]),
        dr[1]
        + 1j * m * inv_r * host[1]
        + 1j * kz * host[4]
        + inv_r * 2 * host[2],
        dr[2] + 1j * m * inv_r * host[4] + 1j * kz * host[5] + inv_r * host[3],
    ]
    for name, g, r in zip(("r", "theta", "z"), got, ref, strict=True):
        assert_allclose(g, r, atol=1e-11, err_msg=f"div(c)_{name}")


if __name__ == "__main__":
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
        else:
            print(f"  PASS  {name}")
    print()
    if failures:
        print(f"{len(tests) - len(failures)} passed, {len(failures)} failed.")
        for name, exc in failures:
            print(f"  {name}: {exc}")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
