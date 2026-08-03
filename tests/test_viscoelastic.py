r"""Unit tests for the viscoelastic (sPTT) annular geometry.

Tests cover the conformation-tensor machinery added on top of the
annular geometry (see
:mod:`dnsjax.geometries.wall_bounded.annular_viscoelastic`):

1. Spin `$\leftrightarrow$` physical tensor conversions are mutual
   inverses (both directions), as are the 9-component
   ``to_spin_basis``/``from_spin_basis`` maps built on them -- the
   component-basis boundary crossed once per state (``__main__``),
   where an inversion error would be silent.  The runtime probe
   stream crosses the same boundary per *column*: its gather and the
   component labels it advertises (checked against the analysis
   package's stored-component schema) are covered here too.
2. The spin-diagonal tensor Laplacian equals an independently coded
   coupled cylindrical tensor Laplacian (radial/axial scalar part plus
   the `$\tfrac1{r^2}(\mathcal R + im)^2$` angular part built from the
   6x6 basis-rotation generator `$\mathcal R$`) on random spectral data.
3. Laminar fixed point at `$\epsilon = 0, \kappa = 0$`: the conformation
   slice of the nonlinear RHS vanishes at the analytical laminar pair
   (advection Christoffels + stretching + relaxation cancel), and a full
   predictor/corrector step reproduces the laminar state (the velocity
   polymer-divergence balance closes too).  The azimuthal *momentum*
   balance is pinned separately, and closes **only** at
   `$\epsilon = 0$`: the builder's Newtonian `$U_\theta$` neglects the
   polymer's shear thinning, so at `$\epsilon > 0$` the pair is not a
   steady state.
4. `$H_c$` band-vs-dense parity including the narrow Laplacian BC
   wall rows (mirrors ``test_annular``'s operator parity).
5. The adapter surface the shared stepper
   (:mod:`dnsjax.geometries.wall_bounded._viscoelastic_stepping`)
   dispatches on: the annulus zeroes both `$H_c$` wall rows (the pipe
   one), and importing this geometry must not build the cylindrical
   one.
6. ``get_norm2_conformation`` reproduces the tensor Frobenius norm.
7. Fused-RHS transform-count guard: the 9-component nonlinear RHS
   keeps a bounded, batched FFT count (fused evaluation, not one
   transform per field).

Run as a script via ``uv run python tests/test_viscoelastic.py``.
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

# Module config: viscoelastic-dean with epsilon = kappa = 0 (the exact
# discrete laminar fixed point, test 3) and a modest Weissenberg number.
# el = wi keeps the derived Re = wi / el = 1.  The H_c operator tests
# build their own kappa > 0 operators, independent of the flow's
# (kappa = 0, so ``flow.Hc_op is None``).
update_parameters(
    Parameters(
        phys={
            "system": "viscoelastic-dean",
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

from dnsjax.flows.wall_bounded.viscoelastic_dean import (  # noqa: E402
    _laminar_state,
    flow,
    predict_and_fully_correct,
)
from dnsjax.geometries.wall_bounded import get_norm2  # noqa: E402
from dnsjax.geometries.wall_bounded._viscoelastic_common import (  # noqa: E402
    from_spin_basis,
    get_norm2_conformation,
    phys_combos_to_spin,
    spin_to_phys_combos,
    to_spin_basis,
)
from dnsjax.geometries.wall_bounded.annular_viscoelastic import (  # noqa: E402
    _build_Hc_band_gpu,
    _build_Hc_dense_gpu,
    _get_rhs,
    _narrow_abase_wall_rows,
    _tensor_laplacian_spin,
    fourier,
    viscoelastic_laminar_profiles,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _banded_factor,
    _banded_from_dense,
)

# 6x6 basis-rotation generator R in the physical tensor-component order
# (c_rr, c_thth, c_rth, c_rz, c_thz, c_zz), i.e. the theta-derivative
# action on the orthonormal tensor basis:
#   R(c)_rr = -2 c_rth,  R(c)_thth = +2 c_rth,  R(c)_rth = c_rr - c_thth,
#   R(c)_rz = -c_thz,    R(c)_thz  =  c_rz,     R(c)_zz  = 0.
# Its eigenvalues are the spin weights i*s, s in {0, 0, +-1, +-2}.
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


def _tensor_laplacian_physical_reference(
    cphys: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    inv_r: np.ndarray,
    inv_r2: np.ndarray,
    m_vals: np.ndarray,
    kz_vals: np.ndarray,
) -> np.ndarray:
    r"""Coupled cylindrical tensor Laplacian in the physical basis.

    `$(\nabla^2 c) = [\partial_r^2 + \tfrac1r\partial_r - k_z^2]\,c
    + \tfrac1{r^2}(\mathcal R + im)^2 c$` applied per mode, with the
    angular part carried by the 6x6 generator `$\mathcal R$`
    (:data:`_R_GEN`).  Independent of the spin-diagonal implementation
    (which instead uses
    the analytic eigenvalue `$-(m+s)^2$`).  ``cphys`` is ordered
    ``(c_rr, c_thth, c_rth, c_rz, c_thz, c_zz)``.
    """
    Nr, Nm, Nkz = cphys.shape[1:]
    out = np.zeros_like(cphys)
    eye6 = np.eye(6)
    for mi in range(Nm):
        for ki in range(Nkz):
            m, kz = m_vals[mi], kz_vals[ki]
            # Radial + axial scalar part, per component.
            for comp in range(6):
                vec = cphys[comp, :, mi, ki]
                out[comp, :, mi, ki] = (
                    D2 @ vec + inv_r * (D1 @ vec) - kz**2 * vec
                )
            # Angular part (1/r^2) (R + i m)^2 acting on the 6-vector.
            gen2 = _R_GEN @ _R_GEN + 2j * m * _R_GEN - m**2 * eye6
            block = cphys[:, :, mi, ki]  # (6, Nr)
            out[:, :, mi, ki] += (gen2 @ block) * inv_r2[None, :]
    return out


# ── tests ────────────────────────────────────────────────────────────

# Group A: spin <-> physical conversions


def test_spin_physical_round_trip() -> None:
    """``spin_to_phys_combos`` and ``phys_combos_to_spin`` invert."""
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(1)

    # spin -> physical -> spin.
    spin = _random_tensor(rng, (6, Nr, Nm, Nkz))
    phys = spin_to_phys_combos(*[jnp.asarray(spin[i]) for i in range(6)])
    spin_rt = np.asarray(phys_combos_to_spin(*phys))
    assert_allclose(spin_rt, spin, atol=1e-13, err_msg="spin->phys->spin")

    # physical -> spin -> physical.
    phys0 = [jnp.asarray(_random_tensor(rng, (Nr, Nm, Nkz))) for _ in range(6)]
    spin_from = phys_combos_to_spin(*phys0)
    phys_rt = spin_to_phys_combos(*[spin_from[i] for i in range(6)])
    for got, ref, name in zip(
        phys_rt, phys0, ("rr", "thth", "rth", "rz", "thz", "zz"), strict=True
    ):
        assert_allclose(
            np.asarray(got),
            np.asarray(ref),
            atol=1e-13,
            err_msg=f"phys->spin->phys c_{name}",
        )

    # The 9-component boundary maps built on them: the pair crossed
    # once per state by ``__main__``, so a silent inversion error here
    # would corrupt every snapshot and diagnostic.
    state = jnp.asarray(_random_tensor(rng, (9, Nr, Nm, Nkz)))
    assert_allclose(
        np.asarray(from_spin_basis(to_spin_basis(state))),
        np.asarray(state),
        atol=1e-13,
        err_msg="phys->spin->phys 9-component state",
    )
    assert_allclose(
        np.asarray(to_spin_basis(from_spin_basis(state))),
        np.asarray(state),
        atol=1e-13,
        err_msg="spin->phys->spin 9-component state",
    )


def test_probe_stream_component_basis() -> None:
    r"""The probe gather crosses the 9-component boundary, and the
    labels it advertises name the components it returns.

    The probe stream is the only consumer that converts *columns*
    rather than whole states, and it is written once and read by
    positional consumers (``response.lim`` / ``response.ssi``), so a
    wrong map or a mislabelled slot is silent.  The label list is
    checked against the analysis package's stored-component schema --
    the snapshot's -- so the two 9-component surfaces cannot drift
    apart.
    """
    from dnsjax.analysis._core import geometry_info
    from dnsjax.probes import _component_labels, build_mode_extractor

    assert _component_labels(9) == list(geometry_info(params).components)

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


# Group B: tensor Laplacian


def test_tensor_laplacian_diagonalization() -> None:
    r"""Spin-diagonal Laplacian == coupled physical-basis reference."""
    Nr = params.res.ny
    m_vals = np.asarray(fourier.m).ravel()
    kz_vals = np.asarray(fourier.kz).ravel()
    Nm, Nkz = len(m_vals), len(kz_vals)

    D1 = np.asarray(flow.D1)
    D2 = np.asarray(flow.D2)
    inv_r = np.asarray(flow.inv_r)
    inv_r2 = np.asarray(flow.inv_r2)

    rng = np.random.default_rng(2)
    spin_np = _random_tensor(rng, (6, Nr, Nm, Nkz))
    spin = jax.device_put(jnp.asarray(spin_np), sharding.spec_vector_shard)

    # Implementation: spin-diagonal, analytic -(m + s)^2 eigenvalue.
    got = np.asarray(_tensor_laplacian_spin(spin, fourier, flow))

    # Reference: convert to physical, apply the coupled Laplacian with
    # the 6x6 generator, convert back to spin.
    cphys = np.stack([np.asarray(x) for x in spin_to_phys_combos(*spin_np)])
    ref_phys = _tensor_laplacian_physical_reference(
        cphys, D1, D2, inv_r, inv_r2, m_vals, kz_vals
    )
    ref = np.asarray(phys_combos_to_spin(*ref_phys))

    assert_allclose(got, ref, atol=1e-10, rtol=1e-10, err_msg="tensor Lap")


# Group C: laminar fixed point (epsilon = kappa = 0)


def test_laminar_conformation_rhs_vanishes() -> None:
    r"""Conformation slice of the RHS vanishes at the laminar pair.

    At the analytical laminar state the curvilinear advection, stretching
    and relaxation cancel to (discrete) machine precision, so the
    conformation slice of ``_get_rhs`` (which excludes the diffusion
    Laplacian) is ~0 relative to the conformation magnitude.
    """
    # ``_laminar_state`` is physical (the flow hands it to ``__main__``,
    # which converts once); the RHS is a solver-basis function.
    state = to_spin_basis(_laminar_state)
    rhs = np.asarray(_get_rhs(state, fourier, flow))
    conf_rhs = rhs[3:]
    conf_scale = float(np.max(np.abs(np.asarray(state[3:]))))
    residual = float(np.max(np.abs(conf_rhs)))
    assert residual < 1e-9 * conf_scale, (
        f"conformation RHS {residual:.3e} not << scale {conf_scale:.3e}"
    )


def test_laminar_velocity_balance_closes_only_at_zero_epsilon() -> None:
    r"""The laminar pair balances *momentum* only at `$\epsilon = 0$`.

    The builder pairs the sPTT-equilibrium conformation with the
    **Newtonian** `$U_\theta$` (``annular_forced_laminar_u_theta``,
    which never receives `$\beta$` or `$\mathrm{Re}$`), so at
    `$\epsilon > 0$` the polymer's shear thinning is unaccounted for
    and the azimuthal balance
    `$\nu(A_{\mathrm{base}} - 1/r^2)U_\theta +
    \tfrac{1-\beta}{\mathrm{Re}\,\mathrm{Wi}}(\nabla\cdot c)_\theta
    + \Pi_\theta = 0$` carries a real residual -- the approximation the
    flow module records.  Pinned in both directions (the twin of
    ``test_viscoelastic_pipe``'s check) so the `$\epsilon = 0$`
    exactness cannot rot and a future exact profile turns the second
    bound into a tight one.  NumPy only: `$\epsilon$` is an *argument*
    of the profile builder, so the module needs no reconfiguration.
    """
    from dnsjax.parameters import derived_params

    rs = np.asarray(flow.rs)
    d1 = np.asarray(flow.D1)
    inv_r = 1.0 / rs
    a_base = np.asarray(flow.D2) + np.diag(inv_r) @ d1
    beta, re, wi = params.phys.beta, params.phys.re, params.phys.wi
    nu = beta / re
    coef = (1.0 - beta) / (re * wi)
    r1, r2 = derived_params.r_inner, derived_params.r_outer
    pi_theta = (r1 + r2) * inv_r / re

    def _residual(eps: float) -> float:
        prof = viscoelastic_laminar_profiles(rs, d1, r1, r2, wi, eps)
        u_th, c_rth = prof[2].real, prof[8].real
        # m = k_z = 0, so (div c)_theta is d_r c_rth + 2 c_rth / r.
        div_th = d1 @ c_rth + 2.0 * c_rth * inv_r
        resid = (
            nu * (a_base @ u_th - inv_r**2 * u_th) + coef * div_th + pi_theta
        )
        # Interior only: the wall rows carry the no-slip BC, not the
        # momentum balance.
        return float(np.abs(resid[1:-1]).max() / np.abs(pi_theta).max())

    # Unlike the pipe's polynomial ``1 - r^2``, the annular profile is
    # not exactly representable by the FD operators, so the epsilon = 0
    # residual is FD truncation (~1e-7 here), not round-off.  Compare
    # the two against each other rather than against an absolute bound:
    # that is the claim, and it is grid- and order-independent.
    exact, got = _residual(0.0), _residual(1e-3)
    assert exact < 1e-5, (
        f"epsilon = 0 must be truncation-level, got {exact:.2e}"
    )
    assert got > 1e3 * exact, (
        f"epsilon = 1e-3 residual {got:.2e} not >> the "
        f"epsilon = 0 truncation floor {exact:.2e}"
    )
    print(
        f"  laminar azimuthal-momentum residual / Pi_theta: "
        f"{got:.2e} at eps=1e-3 vs {exact:.2e} at eps=0"
    )


def test_laminar_full_step_fixed_point() -> None:
    r"""A full predictor/corrector step reproduces the laminar state.

    At `$\epsilon = 0, \kappa = 0$` the analytical laminar pair is the
    exact discrete steady state of both the conformation transport and
    the momentum balance (the polymer-stress divergence reconstructs the
    missing `$(1-\beta)$` viscous stress), so one step returns the same
    state to FD truncation: velocity deviation energy ~0 and the
    conformation unchanged.
    """
    # Solver basis on both sides of the step (``_laminar_state`` is
    # physical; ``__main__`` performs this same single conversion).
    state0 = np.asarray(to_spin_basis(_laminar_state))
    state_new, error, num_c = predict_and_fully_correct(
        to_spin_basis(_laminar_state)
    )
    state_new = np.asarray(state_new)

    # Velocity deviation energy E' (solver basis: the 1/2 weight on the
    # u_pm pair makes this the physical energy).
    dvel = jnp.asarray(state_new[:3] - state0[:3])
    e_prime = (
        float(
            get_norm2(dvel[:1], fourier.k_metric, flow.y_weights)
            + get_norm2(dvel[1:], fourier.k_metric, flow.y_weights) / 2
        )
        / 2
    )
    assert e_prime < 1e-10, f"velocity E' after step {e_prime:.3e}"

    # Conformation drift, relative to its magnitude.
    conf_scale = float(np.max(np.abs(state0[3:])))
    conf_drift = float(np.max(np.abs(state_new[3:] - state0[3:])))
    assert conf_drift < 1e-8 * conf_scale, (
        f"conformation drift {conf_drift:.3e} vs scale {conf_scale:.3e}"
    )
    assert np.isfinite(float(error)), "non-finite corrector error"


# Group D: H_c operator parity


def test_Hc_band_vs_dense() -> None:
    r"""``H_c`` banded/Pallas matches the dense operator.

    Builds the conformation Crank-Nicolson Helmholtz operator with a
    finite `$\kappa$` for representative spin components (the five
    distinct `$m_{\mathrm{eff}} = m + s$`), and checks: the banded
    assembly equals ``banded(dense)`` including the narrow Laplacian BC
    wall rows, and the no-pivot banded (Pallas CPU) solve reproduces
    the dense solve.

    The per-spin base operators come from the flow's own
    ``hc_spin_bases`` adapter (the annular half of the surface the
    shared stepper dispatches on), so the geometry binding is under
    test here and not re-derived by the test.
    """
    Nr = params.res.ny
    p = params.res.fd_order

    dt = params.step.dt
    c = params.step.implicitness
    kappa = 0.05  # finite diffusion (module kappa is 0)

    inv_r2 = flow.inv_r2
    m_s = fourier.m[0, ..., None]  # (Nm, 1, 1)
    kz2_s = fourier.kz2[0, ..., None]  # (1, Nkz, 1)

    row0_np, rowN_np = _narrow_abase_wall_rows(
        np.asarray(flow.rs), np.asarray(flow.D1), p
    )
    # Both walls, as ``flow.hc_wall_rows()`` returns them (it cannot be
    # called here: the module kappa is 0, so the narrow-row leaves are
    # unset).
    walls = ((0, jnp.asarray(row0_np)), (Nr - 1, jnp.asarray(rowN_np)))

    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(4)
    b = _random_tensor(rng, (Nr, Nm, Nkz))
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_scalar_shard)

    to_band = jax.vmap(jax.vmap(lambda A: _banded_from_dense(A, p)))

    spins = (0, 1, -1, 2, -2)
    dense_bases = flow.hc_spin_bases(fourier, spins, banded=False, p=p)
    band_bases = flow.hc_spin_bases(fourier, spins, banded=True, p=p)

    for s, dense_base, band_base in zip(
        spins, dense_bases, band_bases, strict=True
    ):
        meff2 = (m_s + s) ** 2
        dense = _build_Hc_dense_gpu(
            dense_base, walls, meff2, inv_r2, kz2_s, dt, c, kappa
        )
        band = _build_Hc_band_gpu(
            band_base, walls, meff2, inv_r2, kz2_s, dt, c, kappa, p
        )

        assert_allclose(
            np.asarray(band),
            np.asarray(to_band(dense)),
            atol=1e-12,
            err_msg=f"s={s}: band assembly",
        )

        dense_solver = DenseJAXSolver(dense)
        pallas = PerModeBandedPallasOperator.from_banded_factors(
            *_banded_factor(band)
        )

        ref = np.asarray(dense_solver.solve(rhs))
        assert_allclose(
            np.asarray(pallas.solve(rhs)),
            ref,
            atol=1e-9,
            rtol=1e-9,
            err_msg=f"s={s}: pallas solve",
        )


def test_hc_wall_rows_both_walls() -> None:
    r"""The annulus zeroes **two** `$H_c$` RHS rows, at both walls.

    ``zero_hc_wall_rows`` is the adapter the shared ``_c_cn_update``
    applies for the `$\nabla^2 c = 0$` BC, and the wall count is the
    one thing the two geometries disagree on there (the pipe zeroes
    one; its axis carries no row).  Every ``test_laminar_smoke``
    viscoelastic entry runs at `$\kappa = 0$`, where that branch is not
    reached at all, so pin it directly.
    """
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(6)
    R = jnp.asarray(_random_tensor(rng, (6, Nr, Nm, Nkz)))
    out = np.asarray(flow.zero_hc_wall_rows(R))

    zero_rows = [i for i in range(Nr) if not np.any(out[:, i])]
    assert zero_rows == [0, Nr - 1], f"zeroed radial rows {zero_rows}"
    # Everything else survives untouched.
    keep = np.asarray(R)[:, 1 : Nr - 1]
    assert_allclose(out[:, 1 : Nr - 1], keep, atol=0.0)


def test_no_cross_geometry_import() -> None:
    """Importing the annular sPTT geometry must not build the pipe.

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
        "    phys={'system': 'viscoelastic-dean'},\n"
        "    res={'nx': 8, 'ny': 13, 'nz': 8}))\n"
        "padded_res.set_padded_resolution(params)\n"
        "import dnsjax.geometries.wall_bounded.annular_viscoelastic\n"
        "bad = sorted(m for m in sys.modules if 'cylindrical' in m)\n"
        "print(' '.join(bad))\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True
    )
    assert r.returncode == 0, (
        f"cylindrical modules imported: {r.stdout.strip()}\n{r.stderr[-800:]}"
    )


# Group E: norms


def test_conformation_frobenius_norm() -> None:
    r"""``get_norm2_conformation`` == the tensor Frobenius norm.

    `$\langle\|c\|_F^2\rangle = \langle c_{rr}^2 + c_{\theta\theta}^2 +
    c_{zz}^2 + 2c_{r\theta}^2 + 2c_{rz}^2 + 2c_{\theta z}^2\rangle$`
    computed on the native components equals the explicit off-diagonal
    x2 reference, and the spin-weighted norm of the spin image gives
    the same scalar (the metric identity behind the corrector
    ``_norm``).
    """
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(5)

    phys = [jnp.asarray(_random_tensor(rng, (Nr, Nm, Nkz))) for _ in range(6)]
    c_rr, c_thth, c_rth, c_rz, c_thz, c_zz = phys

    # Native layout (c_zz, c_rz, c_theta_z, c_rr, c_theta_theta,
    # c_r_theta).
    native = jnp.stack([c_zz, c_rz, c_thz, c_rr, c_thth, c_rth])
    got = float(
        get_norm2_conformation(native, fourier.k_metric, flow.y_weights)
    )

    # Physical Frobenius: off-diagonal components carry a sqrt(2) so
    # their squared contribution is 2 c^2 (symmetric tensor).
    root2 = np.sqrt(2.0)
    phys_stack = jnp.stack(
        [c_rr, c_thth, c_zz, root2 * c_rth, root2 * c_rz, root2 * c_thz]
    )
    ref = float(get_norm2(phys_stack, fourier.k_metric, flow.y_weights))
    assert_allclose(got, ref, rtol=1e-12, err_msg="Frobenius norm")

    # Spin-image identity: weights (1, 1, 1, 1/2, 1/4, 1/4).
    spin = phys_combos_to_spin(c_rr, c_thth, c_rth, c_rz, c_thz, c_zz)
    w = jnp.asarray(np.sqrt([1.0, 1.0, 1.0, 0.5, 0.25, 0.25]))[
        :, None, None, None
    ]
    spin_ref = float(get_norm2(spin * w, fourier.k_metric, flow.y_weights))
    assert_allclose(got, spin_ref, rtol=1e-12, err_msg="spin identity")


# Group F: fused transform


def test_fused_rhs_transform_count() -> None:
    r"""The 9-component RHS uses a *bounded* (batched) transform count.

    The fused evaluation does one batched inverse transform of ~36
    fields and one batched forward transform of the 9 outputs -- a small
    constant number of ``fft`` primitives, **independent** of the field
    count.  A regression to per-field transforms would be ``O(36 + 9)``
    times larger; guard that the count stays bounded.
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


def test_rhs_transform_chunks_parity() -> None:
    r"""``solver.rhs_transform_chunks`` does not change the RHS.

    The chunked inverse transform (a memory knob: it splits the
    36-field batch into balanced groups to cap the transform-stage
    transient) must reproduce the fused evaluation -- the per-field
    transforms are independent, so any difference is a chunking bug.
    Checked at ``k = 3`` (uneven 12/12/12 over the concatenated
    9+27-field stack) and ``k = 5`` (uneven group sizes).  The module
    config (``rhs_transform_chunks = 1``) is restored on exit.
    """
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(7)
    state = jax.device_put(
        jnp.asarray(_random_tensor(rng, (9, Nr, Nm, Nkz))),
        sharding.spec_vector_shard,
    )
    fused = np.asarray(_get_rhs(state, fourier, flow))
    try:
        for k in (3, 5):
            params.solver.rhs_transform_chunks = k
            chunked = np.asarray(_get_rhs(state, fourier, flow))
            assert_allclose(
                chunked,
                fused,
                rtol=1e-14,
                atol=1e-14 * np.max(np.abs(fused)),
                err_msg=f"chunked (k={k}) RHS != fused RHS",
            )
    finally:
        params.solver.rhs_transform_chunks = 1


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
