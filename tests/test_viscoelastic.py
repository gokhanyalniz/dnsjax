r"""Unit tests for the viscoelastic (sPTT) annular geometry.

Tests cover the conformation-tensor machinery added on top of the
annular geometry (see
:mod:`dnsjax.geometries.wall_bounded.annular_viscoelastic`):

1. Spin `$\leftrightarrow$` physical tensor conversions are mutual
   inverses (both directions).
2. The spin-diagonal tensor Laplacian equals an independently coded
   coupled cylindrical tensor Laplacian (radial/axial scalar part plus
   the `$\tfrac1{r^2}(\mathcal R + im)^2$` angular part built from the
   6x6 basis-rotation generator `$\mathcal R$`) on random spectral data.
3. Laminar fixed point at `$\epsilon = 0, \kappa = 0$`: the conformation
   slice of the nonlinear RHS vanishes at the analytical laminar pair
   (advection Christoffels + stretching + relaxation cancel), and a full
   predictor/corrector step reproduces the laminar state (the velocity
   polymer-divergence balance closes too).
4. `$H_c$` band-vs-dense-vs-SPIKE parity including the narrow Laplacian
   BC wall rows (mirrors ``test_annular``'s operator parity).
5. ``get_norm2_conformation`` reproduces the tensor Frobenius norm.

Run as a script via ``uv run python tests/test_viscoelastic.py``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from dnsjax.parameters import (  # noqa: E402
    Parameters,
    padded_res,
    params,
    update_parameters,
)

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

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax.flows.wall_bounded.viscoelastic_dean import (  # noqa: E402
    _laminar_state,
    flow,
    predict_and_fully_correct,
)
from dnsjax.geometries.wall_bounded import get_norm2  # noqa: E402
from dnsjax.geometries.wall_bounded.annular_viscoelastic import (  # noqa: E402
    _build_Hc_band_gpu,
    _build_Hc_blocks_gpu,
    _build_Hc_dense_gpu,
    _get_rhs,
    _narrow_abase_wall_rows,
    _phys_combos_to_spin,
    _spin_to_phys_combos,
    _tensor_laplacian_spin,
    fourier,
    get_norm2_annular,
    get_norm2_conformation,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _banded_factor,
    _banded_from_dense,
    _spike_factor,
    validate_spike_partition,
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
    """``_spin_to_phys_combos`` and ``_phys_combos_to_spin`` invert."""
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(1)

    # spin -> physical -> spin.
    spin = _random_tensor(rng, (6, Nr, Nm, Nkz))
    phys = _spin_to_phys_combos(*[jnp.asarray(spin[i]) for i in range(6)])
    spin_rt = np.asarray(_phys_combos_to_spin(*phys))
    assert_allclose(spin_rt, spin, atol=1e-13, err_msg="spin->phys->spin")

    # physical -> spin -> physical.
    phys0 = [jnp.asarray(_random_tensor(rng, (Nr, Nm, Nkz))) for _ in range(6)]
    spin_from = _phys_combos_to_spin(*phys0)
    phys_rt = _spin_to_phys_combos(*[spin_from[i] for i in range(6)])
    for got, ref, name in zip(
        phys_rt, phys0, ("rr", "thth", "rth", "rz", "thz", "zz"), strict=True
    ):
        assert_allclose(
            np.asarray(got),
            np.asarray(ref),
            atol=1e-13,
            err_msg=f"phys->spin->phys c_{name}",
        )


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
    cphys = np.stack([np.asarray(x) for x in _spin_to_phys_combos(*spin_np)])
    ref_phys = _tensor_laplacian_physical_reference(
        cphys, D1, D2, inv_r, inv_r2, m_vals, kz_vals
    )
    ref = np.asarray(_phys_combos_to_spin(*ref_phys))

    assert_allclose(got, ref, atol=1e-10, rtol=1e-10, err_msg="tensor Lap")


# Group C: laminar fixed point (epsilon = kappa = 0)


def test_laminar_conformation_rhs_vanishes() -> None:
    r"""Conformation slice of the RHS vanishes at the laminar pair.

    At the analytical laminar state the curvilinear advection, stretching
    and relaxation cancel to (discrete) machine precision, so the
    conformation slice of ``_get_rhs`` (which excludes the diffusion
    Laplacian) is ~0 relative to the conformation magnitude.
    """
    state = jnp.copy(_laminar_state)
    rhs = np.asarray(_get_rhs(state, fourier, flow))
    conf_rhs = rhs[3:]
    conf_scale = float(np.max(np.abs(np.asarray(state[3:]))))
    residual = float(np.max(np.abs(conf_rhs)))
    assert residual < 1e-9 * conf_scale, (
        f"conformation RHS {residual:.3e} not << scale {conf_scale:.3e}"
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
    state0 = np.asarray(_laminar_state)
    state_new, error, num_c = predict_and_fully_correct(
        jnp.copy(_laminar_state)
    )
    state_new = np.asarray(state_new)

    # Velocity deviation energy E'.
    e_prime = (
        float(
            get_norm2_annular(
                jnp.asarray(state_new[:3] - state0[:3]),
                fourier.k_metric,
                flow.y_weights,
            )
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


def test_Hc_band_vs_dense_vs_spike() -> None:
    r"""``H_c`` band / Pallas / SPIKE match the dense operator.

    Builds the conformation Crank-Nicolson Helmholtz operator with a
    finite `$\kappa$` for representative spin components (the five
    distinct `$m_{\mathrm{eff}} = m + s$`), and checks: the banded
    assembly equals ``banded(dense)`` including the narrow Laplacian BC
    wall rows, the no-pivot banded (Pallas CPU) solve reproduces the
    dense solve, and the SPIKE block solve does too.
    """
    Nr = params.res.ny
    p = params.res.fd_order
    P_blk, m_blk = validate_spike_partition(Nr, p, "Nr")

    dt = params.step.dt
    c = params.step.implicitness
    kappa = 0.05  # finite diffusion (module kappa is 0)

    A_base = flow.A_base
    inv_r2 = flow.inv_r2
    m_s = fourier.m[0, ..., None]  # (Nm, 1, 1)
    kz2_s = fourier.kz2[0, ..., None]  # (1, Nkz, 1)

    row0_np, rowN_np = _narrow_abase_wall_rows(
        np.asarray(flow.rs), np.asarray(flow.D1), p
    )
    narrow0 = jnp.asarray(row0_np)
    narrowN = jnp.asarray(rowN_np)

    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(4)
    b = _random_tensor(rng, (Nr, Nm, Nkz))
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_scalar_shard)

    to_band = jax.vmap(jax.vmap(lambda A: _banded_from_dense(A, p)))

    for s in (0, 1, -1, 2, -2):
        meff2 = (m_s + s) ** 2
        dense = _build_Hc_dense_gpu(
            A_base, narrow0, narrowN, meff2, inv_r2, kz2_s, dt, c, kappa
        )
        band = _build_Hc_band_gpu(
            A_base, narrow0, narrowN, meff2, inv_r2, kz2_s, dt, c, kappa, p
        )
        blocks = _build_Hc_blocks_gpu(
            A_base,
            narrow0,
            narrowN,
            meff2,
            inv_r2,
            kz2_s,
            dt,
            c,
            kappa,
            p,
            P_blk,
            m_blk,
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
        spike = _spike_factor(*blocks)

        ref = np.asarray(dense_solver.solve(rhs))
        assert_allclose(
            np.asarray(pallas.solve(rhs)),
            ref,
            atol=1e-9,
            rtol=1e-9,
            err_msg=f"s={s}: pallas solve",
        )
        assert_allclose(
            np.asarray(spike.solve(rhs)),
            ref,
            atol=1e-9,
            rtol=1e-9,
            err_msg=f"s={s}: spike solve",
        )


# Group E: norms


def test_conformation_frobenius_norm() -> None:
    r"""``get_norm2_conformation`` == the tensor Frobenius norm.

    `$\langle\|c\|_F^2\rangle = \langle c_{rr}^2 + c_{\theta\theta}^2 +
    c_{zz}^2 + 2c_{r\theta}^2 + 2c_{rz}^2 + 2c_{\theta z}^2\rangle$`
    computed from the physical components equals the spin-weighted norm.
    """
    Nr = params.res.ny
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(5)

    # Random physical symmetric tensor -> spin components.
    phys = [jnp.asarray(_random_tensor(rng, (Nr, Nm, Nkz))) for _ in range(6)]
    c_rr, c_thth, c_rth, c_rz, c_thz, c_zz = phys
    spin = _phys_combos_to_spin(c_rr, c_thth, c_rth, c_rz, c_thz, c_zz)

    got = float(get_norm2_conformation(spin, fourier.k_metric, flow.y_weights))

    # Physical Frobenius: off-diagonal components carry a sqrt(2) so
    # their squared contribution is 2 c^2 (symmetric tensor).
    root2 = np.sqrt(2.0)
    phys_stack = jnp.stack(
        [c_rr, c_thth, c_zz, root2 * c_rth, root2 * c_rz, root2 * c_thz]
    )
    ref = float(get_norm2(phys_stack, fourier.k_metric, flow.y_weights))

    assert_allclose(got, ref, rtol=1e-12, err_msg="Frobenius norm")


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


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
