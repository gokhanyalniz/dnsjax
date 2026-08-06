"""Unit tests for the cylindrical geometry operators.

Tests cover:

1. Radial CGL grid: rigged-CGL (g=1, default) vs half-CGL (g=0)
   innermost-point formula, ~2x r_0 ratio, and the g=0 closed form.
2. Parity-reduced FD matrices vs full auxiliary grid reference.
3. `$A_{\\mathrm{base}}$` dense operator vs NumPy reference.
4. ``_abase_matvec`` matrix-free vs dense reference.
5. ``_lk_matvec`` vs per-mode NumPy reference.
6. Pallas band-vs-dense parity (banded storage == ``banded(dense)``,
   no-pivot banded solve == dense solve) for ``Lk``/``Hk`` and the two
   ``res.consistent_imm`` operators -- also the regression guard for
   the parity-reduced builders' refactor onto the shared
   ``solvers._assemble_banded_operator`` helpers.
6b. The pipe's ``res.consistent_imm`` structure: the spin pair really
    diagonalises the vector Laplacian onto the existing
    `$H_{k,\\pm}$` operators, every vw quantity lands in the parity
    class its operator assumes, and the packed mean plane reuses the
    primitive scheme's mean operators bit-exactly (including the
    band splice).  The geometry-independent vw algebra (sources,
    reconstruction) is checked once, in ``test_annular.py``.
6c. ``dnsjax.analysis`` rebuilds the same parity-reduced radial ``D1``
    the solver uses -- the ingredient behind every exported operator,
    where a mismatch fails silently.
7. ``get_norm2_cyl`` correctness, and the metric identity linking it
   to the solver basis' 1/2-weighted norm.
8. ``to_solver_basis``/``from_solver_basis`` round-trip: the
   physical/solver component-basis boundary crossed once per state
   (``__main__``), where an inversion error would be silent.
9. Composite integration weights on the radial CGL grid.
10. ``interpolate_to_axis``: polynomial exactness, parity paths,
    multi-dimensional/complex inputs.
11. Centreline mean axial velocity under time stepping (a small
    random perturbation keeps the interpolated ``r = 0`` mean
    axial velocity near the laminar centreline value 1).

Run as a script via ``uv run python tests/test_cylindrical.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

# Select the JAX backend from --dist.platform (default cpu) before the
# geometry import below builds sharding.  --dist.platform cuda runs the
# Pallas band-vs-dense parity on a GPU.
from dnsjax.bootstrap import (
    configure_jax_platform,
    platform_from_argv,
)
from dnsjax.parameters import (
    Parameters,
    padded_res,
    params,
    update_parameters,
)

sys.stdout.reconfigure(line_buffering=True)

configure_jax_platform(platform_from_argv())

update_parameters(
    Parameters(
        phys={"system": "pipe"},
        res={
            "nx": 4,
            "ny": 16,
            "nz": 4,
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

from dnsjax.fd import (  # noqa: E402
    build_diff_matrices,
    build_integration_weights,
)
from dnsjax.flows.wall_bounded.pipe import flow as pipe_flow  # noqa: E402
from dnsjax.geometries.wall_bounded import get_norm2  # noqa: E402
from dnsjax.geometries.wall_bounded.cylindrical import (  # noqa: E402
    _abase_matvec,
    _build_A_base,
    _build_Hk_band_gpu,
    _build_Hk_dense_gpu,
    _build_Lk_band_gpu,
    _build_Lk_dense_gpu,
    _build_Lv_dir_band_gpu,
    _build_Lv_dir_dense_gpu,
    _ghost_row_count,
    _hk_vw_bands,
    _hk_vw_dense_mats,
    _lk_matvec,
    build_parity_reduced_matrices,
    build_radial_cgl_grid,
    extract_mean_mode,
    fourier,
    from_solver_basis,
    get_norm2_cyl,
    interpolate_to_axis,
    to_solver_basis,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.solvers import (  # noqa: E402
    DenseJAXSolver,
    PerModeBandedPallasOperator,
    _banded_factor,
    _banded_from_dense,
)

# ── helpers ──────────────────────────────────────────────────────────


def _build_Lk_reference_cyl(
    m_val: float,
    kz_val: float,
    A_base: np.ndarray,
    D1_wall_row: np.ndarray,
    inv_r2: np.ndarray,
) -> np.ndarray:
    r"""Build single-mode cylindrical `$L_k$` in NumPy.

    `$L_k = A_{\mathrm{base}} - (m^2/r^2 + k_z^2)\,I$` with
    Neumann wall row (last row), or pin for mean mode.
    """
    diag_shift = m_val**2 * inv_r2 + kz_val**2
    Lk = A_base.copy() - np.diag(diag_shift)
    is_mean = m_val == 0.0 and kz_val == 0.0
    if is_mean:
        Lk[-1, :] = 0.0
        Lk[-1, -1] = 1.0
    else:
        Lk[-1, :] = D1_wall_row
    return Lk


# ── tests ────────────────────────────────────────────────────────────

# Group A: Grid and FD matrices


def test_radial_cgl_grid_rigged_vs_half() -> None:
    """Radial CGL grid: rigged (g=1, default) vs half-CGL (g=0)."""
    for Nr in [8, 16, 32]:
        # g = 0 reproduces the direct half-CGL expression bit-exactly
        # (identical formula: even auxiliary total 2 Nr).
        direct = -jnp.cos(
            jnp.arange(2 * Nr, dtype=jnp.float64) * jnp.pi / (2 * Nr - 1)
        )[Nr:]
        assert np.array_equal(
            np.asarray(build_radial_cgl_grid(Nr, axis_gap=0)),
            np.asarray(direct),
        ), f"Nr={Nr}: g=0 != direct half-CGL expression"

        for g in (0, 1):
            rs = np.asarray(build_radial_cgl_grid(Nr, g))
            assert rs.shape == (Nr,), f"g={g}: wrong shape {rs.shape}"
            assert np.all(rs > 0), f"Nr={Nr}, g={g}: non-positive"
            assert np.all(np.diff(rs) > 0), f"g={g}: not increasing"
            assert_allclose(rs[-1], 1.0, atol=1e-14)
            # Innermost point r_0 = sin(pi (g+1) / (2 (2 Nr + g - 1))).
            r0 = np.sin(np.pi * (g + 1) / (2 * (2 * Nr + g - 1)))
            assert_allclose(rs[0], r0, atol=1e-14, err_msg=f"g={g} r0")

        # Rigged r0 is ~2x the half-CGL r0 (the cnab2 CFL relief);
        # the ratio approaches 2 from below as Nr grows.
        r0_half = np.asarray(build_radial_cgl_grid(Nr, 0))[0]
        r0_rig = np.asarray(build_radial_cgl_grid(Nr, 1))[0]
        assert 1.8 < r0_rig / r0_half < 2.0, f"Nr={Nr}: ratio"

    # The default axis_gap is 1 (rigged-CGL).
    assert np.array_equal(
        np.asarray(build_radial_cgl_grid(16)),
        np.asarray(build_radial_cgl_grid(16, axis_gap=1)),
    )


def test_parity_reduced_matrices_vs_full_grid() -> None:
    """Parity-reduced matrices match the full auxiliary grid."""
    Nr = params.res.ny
    p = params.res.fd_order
    rs = build_radial_cgl_grid(Nr)

    D1_even, D2_even, D1_odd, D2_odd, D1_pos, D2_pos = (
        build_parity_reduced_matrices(rs, p)
    )

    # Reference: build full auxiliary grid explicitly.
    aux_grid = jnp.concatenate([-rs[::-1], rs])
    D1_full, D2_full = build_diff_matrices(aux_grid, p)

    D1_full_np = np.asarray(D1_full)
    D2_full_np = np.asarray(D2_full)

    D1_pos_ref = D1_full_np[Nr:, Nr:]
    D1_ghost_flipped = D1_full_np[Nr:, :Nr][:, ::-1]
    D2_pos_ref = D2_full_np[Nr:, Nr:]
    D2_ghost_flipped = D2_full_np[Nr:, :Nr][:, ::-1]

    assert_allclose(
        np.asarray(D1_pos),
        D1_pos_ref,
        atol=1e-14,
        err_msg="D1_pos mismatch",
    )
    assert_allclose(
        np.asarray(D2_pos),
        D2_pos_ref,
        atol=1e-14,
        err_msg="D2_pos mismatch",
    )
    assert_allclose(
        np.asarray(D1_even),
        D1_pos_ref + D1_ghost_flipped,
        atol=1e-14,
        err_msg="D1_even mismatch",
    )
    assert_allclose(
        np.asarray(D1_odd),
        D1_pos_ref - D1_ghost_flipped,
        atol=1e-14,
        err_msg="D1_odd mismatch",
    )
    assert_allclose(
        np.asarray(D2_even),
        D2_pos_ref + D2_ghost_flipped,
        atol=1e-14,
        err_msg="D2_even mismatch",
    )
    assert_allclose(
        np.asarray(D2_odd),
        D2_pos_ref - D2_ghost_flipped,
        atol=1e-14,
        err_msg="D2_odd mismatch",
    )


# Group B: Operators


def test_A_base_matches_reference() -> None:
    r"""``_build_A_base`` matches `$D_2 + \mathrm{diag}(1/r) D_1$`."""
    Nr = params.res.ny
    p = params.res.fd_order
    rs = build_radial_cgl_grid(Nr)
    inv_r = 1.0 / rs

    D1_even, D2_even, D1_odd, D2_odd, _, _ = build_parity_reduced_matrices(
        rs, p
    )

    for label, D1, D2 in [
        ("even", D1_even, D2_even),
        ("odd", D1_odd, D2_odd),
    ]:
        A_base = np.asarray(_build_A_base(D1, D2, inv_r))
        ref = np.asarray(D2) + np.diag(np.asarray(inv_r)) @ np.asarray(D1)
        assert_allclose(
            A_base,
            ref,
            atol=1e-12,
            err_msg=f"A_base ({label} parity)",
        )


def test_abase_matvec_matches_dense() -> None:
    """``_abase_matvec`` matches dense ``A_base @ u``."""
    Nr = params.res.ny
    p = params.res.fd_order
    rs = build_radial_cgl_grid(Nr)
    inv_r = 1.0 / rs

    D1_even, D2_even, D1_odd, D2_odd, D1_pos, D2_pos = (
        build_parity_reduced_matrices(rs, p)
    )
    # Row-sliced ghost storage, as in CylindricalFlow.__post_init__.
    D1_ghost_full = D1_even - D1_pos
    D2_ghost_full = D2_even - D2_pos
    g_rows = _ghost_row_count(
        np.asarray(D1_ghost_full), np.asarray(D2_ghost_full)
    )
    assert g_rows < Nr, "ghost matrices unexpectedly full"
    D1_ghost = D1_ghost_full[:g_rows]
    D2_ghost = D2_ghost_full[:g_rows]
    A_even = np.asarray(_build_A_base(D1_even, D2_even, inv_r))
    A_odd = np.asarray(_build_A_base(D1_odd, D2_odd, inv_r))

    flow_ = SimpleNamespace(
        D1_pos=D1_pos,
        D2_pos=D2_pos,
        D1_ghost=D1_ghost,
        D2_ghost=D2_ghost,
        inv_r=inv_r,
    )

    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(10)
    u_np = rng.standard_normal((Nr, Nm, Nkz)) + 1j * rng.standard_normal(
        (Nr, Nm, Nkz)
    )
    u = jnp.asarray(u_np)

    for label, parity_val, A_ref in [
        ("even", 1.0, A_even),
        ("odd", -1.0, A_odd),
    ]:
        parity_sign = jnp.full((1, Nm, 1), parity_val)
        got = np.asarray(_abase_matvec(u, flow_, parity_sign))
        ref = np.einsum("ij, jmz -> imz", A_ref, u_np)
        assert_allclose(
            got,
            ref,
            atol=1e-10,
            rtol=1e-10,
            err_msg=f"parity={label}",
        )


def test_lk_matvec_matches_reference() -> None:
    r"""``_lk_matvec`` matches per-mode NumPy reference."""
    Nr = params.res.ny
    inv_r2 = np.asarray(pipe_flow.inv_r2)
    D1_wall_row = np.asarray(pipe_flow.D1_wall).ravel()
    A_even = np.asarray(pipe_flow.A_base_even)
    A_odd = np.asarray(pipe_flow.A_base_odd)

    m_vals = np.asarray(fourier.m).ravel()
    kz_vals = np.asarray(fourier.kz).ravel()
    m_is_even_np = np.asarray(fourier.m_is_even).ravel()

    Nm = len(m_vals)
    Nkz = len(kz_vals)
    rng = np.random.default_rng(60)
    u_np = rng.standard_normal((Nr, Nm, Nkz)) + 1j * rng.standard_normal(
        (Nr, Nm, Nkz)
    )

    # Reference: build Lk per mode and apply.
    ref = np.zeros_like(u_np)
    for mi in range(Nm):
        A_base = A_even if m_is_even_np[mi] else A_odd
        for ki in range(Nkz):
            Lk = _build_Lk_reference_cyl(
                m_vals[mi],
                kz_vals[ki],
                A_base,
                D1_wall_row,
                inv_r2,
            )
            ref[:, mi, ki] = Lk @ u_np[:, mi, ki]

    u = jax.device_put(
        jnp.asarray(u_np),
        sharding.spec_scalar_shard,
    )
    got = np.asarray(_lk_matvec(u, pipe_flow, fourier))
    assert_allclose(got, ref, atol=1e-10, rtol=1e-10)


def test_pallas_vs_dense_on_cylindrical_operators() -> None:
    r"""``PerModeBandedPallasOperator`` matches ``DenseJAXSolver`` on
    cylindrical Lk/Hk_plus/Hk_minus/Hk_z.

    Guards the parity-reduced Pallas band assembly
    (``_build_{Lk,Hk}_band_gpu``, refactored onto the shared
    ``solvers._assemble_banded_operator``): the banded operator equals
    ``banded(dense)`` exactly and the no-pivot banded sweep reproduces
    the dense solve on a complex RHS.
    """
    Nr = params.res.ny
    p = params.res.fd_order

    m_s = fourier.m[0, ..., None]
    kz2 = fourier.kz2[0, ..., None]
    mean_mask = fourier.mean_mask[0, ..., None]
    m_is_even_s = fourier.m_is_even[0, ..., None]
    m_is_even_p = m_is_even_s
    m_is_even_v = 1.0 - m_is_even_s

    m_sq = m_s**2
    m_plus_1_sq = (m_s + 1) ** 2
    m_minus_1_sq = (m_s - 1) ** 2

    dt = params.step.dt
    c = params.step.implicitness
    nu = 1.0 / params.phys.re

    A_even = pipe_flow.A_base_even
    A_odd = pipe_flow.A_base_odd
    inv_r2 = pipe_flow.inv_r2
    D1_wall = pipe_flow.D1_wall.ravel()
    band_even = _banded_from_dense(A_even, p)
    band_odd = _banded_from_dense(A_odd, p)

    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    rng = np.random.default_rng(71)
    b = rng.standard_normal((Nr, Nm, Nkz)) + 1j * rng.standard_normal(
        (Nr, Nm, Nkz)
    )
    rhs = jax.device_put(jnp.asarray(b), sharding.spec_scalar_shard)

    to_band = jax.vmap(jax.vmap(lambda A: _banded_from_dense(A, p)))

    def _check(label: str, band: jax.Array, full: jax.Array) -> None:
        assert_allclose(
            np.asarray(band),
            np.asarray(to_band(full)),
            atol=1e-12,
            err_msg=f"{label} assembly",
        )
        pallas = PerModeBandedPallasOperator.from_banded_factors(
            *_banded_factor(band)
        )
        dense = DenseJAXSolver(full)
        assert_allclose(
            np.asarray(pallas.solve(rhs)),
            np.asarray(dense.solve(rhs)),
            atol=1e-9,
            rtol=1e-9,
            err_msg=label,
        )

    _check(
        "Lk",
        _build_Lk_band_gpu(
            D1_wall,
            band_even,
            band_odd,
            m_is_even_p,
            m_sq,
            inv_r2,
            kz2,
            mean_mask,
            p,
        ),
        _build_Lk_dense_gpu(
            D1_wall,
            A_even,
            A_odd,
            m_is_even_p,
            m_sq,
            inv_r2,
            kz2,
            mean_mask,
        ),
    )
    for label, meff2, parity in [
        ("Hk_plus", m_plus_1_sq, m_is_even_v),
        ("Hk_minus", m_minus_1_sq, m_is_even_v),
        ("Hk_z", m_sq, m_is_even_p),
    ]:
        _check(
            label,
            _build_Hk_band_gpu(
                band_even,
                band_odd,
                parity,
                meff2,
                inv_r2,
                kz2,
                dt,
                c,
                nu,
                p,
            ),
            _build_Hk_dense_gpu(
                A_even,
                A_odd,
                parity,
                meff2,
                inv_r2,
                kz2,
                dt,
                c,
                nu,
            ),
        )

    # ``res.consistent_imm``: the dt-free Dirichlet recovery
    # L_v,mod on the velocity parity.  Its (D1 + 1/r) correction has to
    # come out of the *same* band as A_base -- the dense twin carries
    # the untruncated operator, so a band that lost an entry shows up
    # in the solve comparison.
    D1_even, _, D1_odd, _, _, _ = build_parity_reduced_matrices(
        pipe_flow.rs, p
    )
    _check(
        "Lv_dir",
        _build_Lv_dir_band_gpu(
            D1_even,
            D1_odd,
            band_even,
            band_odd,
            m_is_even_v,
            m_sq,
            pipe_flow.inv_r,
            inv_r2,
            kz2,
            mean_mask,
            p,
        ),
        _build_Lv_dir_dense_gpu(
            D1_even,
            D1_odd,
            A_even,
            A_odd,
            m_is_even_v,
            m_sq,
            pipe_flow.inv_r,
            inv_r2,
            kz2,
            mean_mask,
        ),
    )
    # The two-family spin pair, including the minus family's spliced
    # mean plane (the one structurally new banded-data pattern).
    for slot, (band, full) in enumerate(
        zip(
            _hk_vw_bands(dt, fourier, pipe_flow),
            _hk_vw_dense_mats(dt, fourier, pipe_flow),
            strict=True,
        )
    ):
        _check(f"Hk_vw[{slot}]", band, full)


def test_vw_spin_pair_diagonalises_the_vector_laplacian() -> None:
    r"""`$(\Delta\mathbf{u})_\pm = L_{s\pm} u_\pm$` on the *existing*
    `$H_{k,\pm}$` base operators.

    The load-bearing identity of the pipe's ``res.consistent_imm``
    scheme (``cylindrical._imm_iteration_vw``): the `$-2im/r^2$` spin
    coupling that ties `$(\Delta\mathbf{u})_r$` to `$u_\theta$` is
    diagonalised by the same `$u_\pm$` combination the primitive
    scheme already uses, with eigenvalues `$(m \pm 1)^2/r^2$` -- which
    is why the quad needs no new operator family.  Checked against the
    physical-component vector Laplacian on random radial profiles, per
    parity-reduced operator.
    """
    Nr = params.res.ny
    inv_r = np.asarray(pipe_flow.inv_r)
    inv_r2 = inv_r**2
    rng = np.random.default_rng(17)

    for m in (0, 1, 2, 5):
        even = m % 2 == 0
        # u_r / u_theta carry parity (-1)^{m+1}: the opposite class.
        A_v = np.asarray(
            pipe_flow.A_base_odd if even else pipe_flow.A_base_even
        )
        for kz in (0.0, 1.5):
            ur = rng.standard_normal(Nr) + 1j * rng.standard_normal(Nr)
            ut = rng.standard_normal(Nr) + 1j * rng.standard_normal(Nr)
            lv = A_v - np.diag((m**2 + 1) * inv_r2 + kz**2)
            lap_r = lv @ ur - 2j * m * inv_r2 * ut
            lap_t = lv @ ut + 2j * m * inv_r2 * ur
            pairs = ((1.0, lap_r + 1j * lap_t), (-1.0, lap_r - 1j * lap_t))
            for sign, lap in pairs:
                ls = A_v - np.diag((m + sign) ** 2 * inv_r2 + kz**2)
                got = ls @ (ur + sign * 1j * ut)
                assert_allclose(
                    got, lap, rtol=1e-11, atol=1e-11, err_msg=f"m={m} s={sign}"
                )


def test_vw_parity_classes() -> None:
    r"""Every vw quantity lands in the parity class its operator
    assumes.

    The pipe's parity reduction is not a convenience: the wrong class
    silently uses the wrong ghost sign near the axis.  Reconstructing
    the mirrored field explicitly, this checks that

    - `$\Phi_\pm = L_{s\pm} u_\pm$` and
      `$\omega_\pm = \omega_r \pm i\omega_\theta$` carry the velocity
      parity `$(-1)^{m+1}$`;
    - the source chain's `$C_z$` -- and the `$r N_\theta$` and `$N_z$`
      it differentiates -- carry `$(-1)^m$`, so both inner `$D_1$`
      applications are `$z$`-parity;
    - the recovery correction `$(2m^2/(r^3\Delta))(D_1 + 1/r)$` maps
      `$(-1)^{m+1}$` back to itself (odd coefficient x parity flip).

    Checked as a symmetry of the *continuum* fields on a mirrored
    grid, which is what the reduction encodes, so it is independent of
    any particular ``D1``.
    """
    rs = np.asarray(build_radial_cgl_grid(24))
    aux = np.concatenate([-rs[::-1], rs])

    def _mirror(vals: np.ndarray, sign: float) -> np.ndarray:
        """Extend a positive-r profile to the mirrored grid."""
        return np.concatenate([sign * vals[::-1], vals])

    for m in (1, 2, 3):
        pz = (-1.0) ** m  # u_z, N_z, C_z
        pv = -pz  # u_r, u_theta, the quad
        prof = np.exp(-2 * rs**2) * (1 + rs)
        # A field with the declared parity, on the mirrored grid.
        f_v, f_z = _mirror(prof, pv), _mirror(prof, pz)
        assert np.allclose(f_v[::-1], pv * f_v)
        assert np.allclose(f_z[::-1], pz * f_z)
        # r N_theta: r is odd, N_theta is (-1)^{m+1} -> (-1)^m.
        r_ntheta = aux * f_v
        assert np.allclose(r_ntheta[::-1], pz * r_ntheta)
        # (1/r^3) x (D1 + 1/r) applied to a (-1)^{m+1} field: D1 and
        # 1/r each flip once, 1/r^3 flips once more -> back to
        # (-1)^{m+1}.
        d1_f = np.gradient(f_v, aux)
        corr = (d1_f + f_v / aux) / aux**3
        assert np.allclose(corr[::-1], pv * corr, atol=1e-8), m


def test_vw_mean_plane_packing_reuses_the_primitive_operators() -> None:
    r"""The packed `$k^2 = 0$` plane carries the primitive scheme's own
    mean-mode Helmholtz operators, **bit-exactly**.

    Flag-on, the four evolved slots are structurally zero at the mean
    mode, so two of them carry `$u_{z,00}$` and `$u_{\theta,00}$`
    instead (``cylindrical._imm_iteration_vw``).  The `$\omega_+$` slot
    rides `$H_{k,+}$`, which at `$m = 0$` already *is* the mean
    `$u_\theta$` operator; the `$\Phi_-$` slot needs the mean
    `$H_{k,z}$` (`$m_{\mathrm{eff}}^2 = 0$`, **even** parity) and gets
    it from the band splice.  Asserted as equality on the mean column
    and *inequality* off it, so a builder that dropped the splice fails
    rather than passing vacuously.
    """
    p = params.res.fd_order
    m_s = fourier.m[0, ..., None]
    kz2 = fourier.kz2[0, ..., None]
    m_is_even_s = fourier.m_is_even[0, ..., None]
    dt, c = params.step.dt, 1.0 / params.phys.re
    band_even = _banded_from_dense(pipe_flow.A_base_even, p)
    band_odd = _banded_from_dense(pipe_flow.A_base_odd, p)

    def _band(parity, meff2):
        return np.asarray(
            _build_Hk_band_gpu(
                band_even,
                band_odd,
                parity,
                meff2,
                pipe_flow.inv_r2,
                kz2,
                dt,
                params.step.implicitness,
                c,
                p,
            )
        )

    band_plus, band_minus = (
        np.asarray(b) for b in _hk_vw_bands(dt, fourier, pipe_flow)
    )
    want_z = _band(m_is_even_s, m_s**2)
    want_plus = _band(1.0 - m_is_even_s, (m_s + 1) ** 2)
    is_mean = np.asarray(fourier.mean_mask[0])
    assert is_mean.sum() == 1, "expected exactly one k^2 = 0 mode"
    for label, got, want in (
        ("Phi_- slot == mean Hk_z", band_minus, want_z),
        ("omega_+ slot == mean Hk_+", band_plus, want_plus),
    ):
        assert_allclose(
            got[is_mean], want[is_mean], atol=0.0, rtol=0.0, err_msg=label
        )
    # Off the mean plane the minus family must *not* be Hk_z, or the
    # splice above is invisible.
    assert not np.allclose(band_minus[~is_mean], want_z[~is_mean])


def test_analysis_radial_d1_matches_the_solver() -> None:
    r"""``dnsjax.analysis`` rebuilds the *same* radial `$D_1$` pair the
    solver uses.

    The analysis package reconstructs the pipe's parity-reduced radial
    derivative from a snapshot's recorded parameters
    (``_core.parity_radial_d1``), and every exported operator --
    divergence, curl, gradient, enstrophy -- is built on it.  A
    reconstruction that disagrees with the solver is a *silent*
    failure: an exactly-solenoidal stepped state reads back as O(1)
    divergent and nothing raises.  There is one radial construction on
    each side, which is exactly what this pins.

    (``tests/test_snapshot_export.py`` pins the assembled divergence
    node-for-node against the solver's own; this pins the ingredient.)
    """
    from dnsjax.analysis._core import parity_radial_d1

    rs = np.asarray(build_radial_cgl_grid(24))
    for p in (4, 8):
        solver = build_parity_reduced_matrices(rs, p)
        mine = parity_radial_d1(rs, p)
        for got, want, label in (
            (mine[0], solver[0], "even"),
            (mine[1], solver[2], "odd"),
        ):
            assert_allclose(
                np.asarray(got),
                np.asarray(want),
                atol=1e-12,
                err_msg=f"D1_{label} fd_order={p}",
            )


# Group C: Norms


def test_get_norm2_cyl() -> None:
    r"""``get_norm2_cyl`` is the plain norm of `$(u_z, u_r, u_\theta)$`.

    The native state carries physical components, so the geometry norm
    is the unweighted component sum -- and equals the 1/2-weighted norm
    of the solver-basis image `$(u_z, u_+, u_-)$` (the metric identity
    `$|u_r|^2 + |u_\theta|^2 = (|u_+|^2 + |u_-|^2)/2$` behind the
    corrector ``_norm``).
    """
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    Nr = params.res.ny

    rng = np.random.default_rng(80)
    s_shape = (3, Nr, Nm, Nkz)
    state_np = rng.standard_normal(s_shape) + 1j * rng.standard_normal(s_shape)
    state = jnp.asarray(state_np)

    k_m = fourier.k_metric
    y_w = pipe_flow.y_weights
    got = float(get_norm2_cyl(state, k_m, y_w))

    # Reference: plain per-component norms.
    ref = float(get_norm2(state[1:], k_m, y_w)) + float(
        get_norm2(state[0:1], k_m, y_w)
    )
    assert_allclose(got, ref, atol=1e-12, err_msg="get_norm2_cyl")

    # Metric identity vs the solver-basis image.
    pm = jnp.stack(
        [state[0], state[1] + 1j * state[2], state[1] - 1j * state[2]]
    )
    pm_ref = float(get_norm2(pm[1:], k_m, y_w)) / 2 + float(
        get_norm2(pm[0:1], k_m, y_w)
    )
    assert_allclose(got, pm_ref, rtol=1e-12, err_msg="pm metric identity")


def test_pm_basis_round_trip() -> None:
    r"""``to_solver_basis`` / ``from_solver_basis`` invert exactly.

    The pair is crossed once per state at the physical/solver boundary
    (``__main__``), so a silent inversion error would corrupt every
    snapshot, diagnostic and initial condition without failing loudly.
    """
    Nm = params.res.nz - 1
    Nkz = params.res.nx // 2
    Nr = params.res.ny
    rng = np.random.default_rng(11)
    state = jnp.asarray(
        rng.standard_normal((3, Nr, Nm, Nkz))
        + 1j * rng.standard_normal((3, Nr, Nm, Nkz))
    )
    assert_allclose(
        np.asarray(from_solver_basis(to_solver_basis(state))),
        np.asarray(state),
        atol=1e-14,
        err_msg="phys->pm->phys",
    )
    assert_allclose(
        np.asarray(to_solver_basis(from_solver_basis(state))),
        np.asarray(state),
        atol=1e-14,
        err_msg="pm->phys->pm",
    )
    # u_z is untouched by the mixing.
    assert_allclose(
        np.asarray(to_solver_basis(state)[0]),
        np.asarray(state[0]),
        atol=0.0,
        err_msg="u_z must pass through",
    )


# Group D: Integration


def test_cylindrical_integration_weights() -> None:
    """Composite weights on the radial CGL grid: sum + exactness."""
    p = params.res.fd_order
    for Nr in [8, 16, 32]:
        rs = build_radial_cgl_grid(Nr)
        rs_np = np.asarray(rs)
        w = build_integration_weights(rs, p=p)
        w_np = np.asarray(w)

        r_min = rs_np[0]
        interval_length = 1.0 - r_min
        assert_allclose(
            np.sum(w_np),
            interval_length,
            atol=1e-12,
            err_msg=f"Nr={Nr}: weight sum",
        )

        for d in range(p + 1):
            computed = float(np.dot(w_np, rs_np**d))
            exact = (1.0 - r_min ** (d + 1)) / (d + 1)
            assert_allclose(
                computed,
                exact,
                atol=1e-10,
                err_msg=f"Nr={Nr}, degree={d}",
            )

    # Full-disc radial quadrature: the parity-specific spectral
    # Clenshaw-Curtis weights (rigged / half-CGL).  y_weights (even)
    # and y_weights_odd (odd) are BOTH strictly positive (definite
    # energy norm), and each is *spectral* for its parity -- exact for
    # the polynomial moments int r^d * r dr = 1/(d+2): even d via
    # y_weights, odd d via y_weights_odd (odd integrands, e.g. the
    # mean u_theta, need the odd exactness), and machine
    # precision on a smooth integrand.
    from dnsjax.geometries.wall_bounded.cylindrical import (
        build_cylindrical_grid,
    )

    for gt, g in (("half-cgl", 0), (None, 1)):
        _, _, _, _, yw, yw_odd, _ = build_cylindrical_grid(16, p, grid_type=gt)
        rs_g = np.asarray(build_radial_cgl_grid(16, g))
        we = np.asarray(yw)
        wo = np.asarray(yw_odd)
        assert np.all(we > 0), f"grid_type={gt}: negative y_weights"
        assert np.all(wo > 0), f"grid_type={gt}: negative y_weights_odd"
        assert_allclose(float(we.sum()), 0.5, atol=1e-12)  # f=1 (even)
        assert_allclose(float(we @ rs_g**2), 0.25, atol=1e-12)  # r^2
        assert_allclose(float(wo @ rs_g), 1.0 / 3.0, atol=1e-12)  # r odd
        assert_allclose(
            float(wo @ rs_g**3),
            0.2,
            atol=1e-12,  # r^3 (odd)
        )
        # Spectral on a smooth integrand (vs fd_order for a composite).
        fine = np.linspace(0.0, 1.0, 2_000_001)
        ref = float(np.trapezoid(np.cos(2.0 * fine) * fine, fine))
        assert abs(float(we @ np.cos(2.0 * rs_g)) - ref) < 1e-10


# Group E: Centreline (r = 0) interpolation


def test_interpolate_to_axis_polynomials() -> None:
    """Axis (r = 0) evaluation: polynomial exactness (one-sided
    Fornberg; spectral parity-constrained even path)."""
    Nr = 16
    rs = np.asarray(build_radial_cgl_grid(Nr))
    order = params.res.fd_order

    # One-sided: exact for degree <= order.
    for d in range(order + 1):
        val = float(interpolate_to_axis(jnp.asarray(rs**d), rs))
        exact = 1.0 if d == 0 else 0.0
        assert_allclose(val, exact, atol=1e-11, err_msg=f"deg {d}")

    # The pipe base-flow profile 1 - r^2 has centreline value 1
    # (quadratic: exact one-sided and via the even path).
    prof = jnp.asarray(1.0 - rs**2)
    assert_allclose(float(interpolate_to_axis(prof, rs)), 1.0, atol=1e-12)
    assert_allclose(
        float(interpolate_to_axis(prof, rs, parity="even")),
        1.0,
        atol=1e-12,
    )

    # Even path on a detected CGL grid: the exact parity-constrained
    # fit in x = r^2 -- exact for even polynomials up to the full
    # degree 2 * (Nr - 1), far past the local rule's 2 * order.
    for d2 in (2 * order, 2 * (Nr - 1)):
        val = float(
            interpolate_to_axis(jnp.asarray(rs**d2), rs, parity="even")
        )
        assert_allclose(val, 0.0, atol=1e-11, err_msg=f"even r^{d2}")

    # Odd parity: identically zero on the axis.
    v = interpolate_to_axis(jnp.asarray(rs**3), rs, parity="odd")
    assert float(v) == 0.0

    try:
        interpolate_to_axis(prof, rs, parity="both")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown parity accepted")


def test_interpolate_to_axis_even_superconvergence() -> None:
    """Even-parity path (spectral in r^2 on CGL) beats one-sided."""
    Nr = 16
    rs = np.asarray(build_radial_cgl_grid(Nr))
    f = jnp.asarray(np.exp(-(rs**2)))  # even, f(0) = 1
    err_even = abs(float(interpolate_to_axis(f, rs, parity="even")) - 1.0)
    err_side = abs(float(interpolate_to_axis(f, rs)) - 1.0)
    assert err_even < 1e-12, f"even-path error {err_even:.3e}"
    assert err_side < 5e-3, f"one-sided error {err_side:.3e}"
    assert err_even < err_side


def test_interpolate_to_axis_multidim() -> None:
    """Any radial-axis position; complex data; matches manual dot."""
    from dnsjax.fd import fornberg_weights

    Nr = 16
    rs = np.asarray(build_radial_cgl_grid(Nr))
    rng = np.random.default_rng(7)
    a = rng.standard_normal((Nr, 3, 5)) + 1j * rng.standard_normal((Nr, 3, 5))
    v0 = np.asarray(interpolate_to_axis(jnp.asarray(a), rs, axis=0))
    assert v0.shape == (3, 5)
    v1 = np.asarray(
        interpolate_to_axis(jnp.asarray(np.moveaxis(a, 0, 1)), rs, axis=1)
    )
    assert_allclose(v1, v0, atol=1e-13)

    # Manual reference: interpolation weights on the innermost
    # order + 1 points.
    n = params.res.fd_order + 1
    w = fornberg_weights(0.0, rs[:n], 0)[:, 0]
    ref = np.tensordot(w, a[:n], axes=(0, 0))
    assert_allclose(v0, ref, atol=1e-13)


def test_axis_extrapolation_weights_shared_leaf() -> None:
    """``interpolate_to_axis`` and the JAX-free
    ``fd.axis_extrapolation_weights`` leaf (the same spectral even
    weights behind the rigged completion) agree on the even-parity
    axis value; the spectral path is gated to detected CGL grids."""
    from dnsjax.fd import axis_extrapolation_weights, tanh_one_sided_grid

    Nr = 16
    rs = np.asarray(build_radial_cgl_grid(Nr))
    order = params.res.fd_order
    f = np.cos(0.7 * rs**2) + 0.2 * rs**2  # even in r
    via_leaf = float(axis_extrapolation_weights(rs, order, "even") @ f)
    via_interp = float(interpolate_to_axis(jnp.asarray(f), rs, parity="even"))
    assert_allclose(via_leaf, via_interp, atol=1e-13)
    # Odd parity: the leaf returns all-zero weights (f(0) = 0).
    assert np.all(axis_extrapolation_weights(rs, order, "odd") == 0.0)
    # CGL even path is the full-grid spectral fit (all weights live);
    # on a non-CGL (tanh) grid it stays on the local order + 1
    # stencil (a full-order global fit is ill-conditioned there).
    w_cgl = axis_extrapolation_weights(rs, order, "even")
    assert np.count_nonzero(w_cgl) == Nr
    rt = np.asarray(tanh_one_sided_grid(Nr, 2.0))
    wt = axis_extrapolation_weights(rt, order, "even")
    assert np.count_nonzero(wt) <= order + 1


def test_parity_dispatch_interpolation() -> None:
    """The ``__main__`` spectral parity-tuple resume dispatch: applying
    ``T_even``/``T_odd`` per azimuthal mode (u_z parity ``(-1)^m``,
    u_r/u_theta parity ``(-1)^{m+1}``) to a parity-consistent state
    recovers the field on the new radial grid to machine precision.
    Mirrors ``_interpolate_if_needed``'s tuple branch and layout
    (3, r, m, kz)."""
    from dnsjax.fd import cgl_parity_interpolation_matrices
    from dnsjax.operators import complex_harmonics

    ny_old, ny_new, nz, nkz = 16, 24, 8, 3
    ro = np.asarray(build_radial_cgl_grid(ny_old, 1))
    rn = np.asarray(build_radial_cgl_grid(ny_new, 1))
    m = np.asarray(complex_harmonics(nz))
    nm = len(m)  # azimuthal-mode axis length (state layout)

    def field(rr, sigma):  # smooth, parity sigma in r
        return np.cos(0.5 * np.pi * rr) if sigma > 0 else rr * np.cos(rr)

    def build(grid):
        s = np.zeros((3, len(grid), nm, nkz), dtype=np.complex128)
        for j, mm in enumerate(m):
            sig_z = 1.0 if mm % 2 == 0 else -1.0
            for k in range(nkz):
                s[0, :, j, k] = field(grid, sig_z) * (1 + 0.1 * k)
                s[1, :, j, k] = field(grid, -sig_z) * (1 + 0.1 * k)
                s[2, :, j, k] = field(grid, -sig_z) * (0.5 + 0.1 * k)
        return s

    src = jnp.asarray(build(ro))
    t_even, t_odd = cgl_parity_interpolation_matrices(ny_old, ny_new, 1, 1)
    m_is_even = m % 2 == 0
    t_p = np.where(m_is_even[:, None, None], t_even, t_odd)  # u_z
    t_v = np.where(m_is_even[:, None, None], t_odd, t_even)  # u_r/u_th
    t_p_j = jnp.asarray(t_p, dtype=src.dtype)
    t_v_j = jnp.asarray(t_v, dtype=src.dtype)
    s0 = jnp.einsum("mij, jmk -> imk", t_p_j, src[0])
    s1 = jnp.einsum("mij, jmk -> imk", t_v_j, src[1])
    s2 = jnp.einsum("mij, jmk -> imk", t_v_j, src[2])
    out = np.asarray(jnp.stack([s0, s1, s2]))
    assert_allclose(out, build(rn), atol=1e-8)


def test_centerline_mean_axial_velocity() -> None:
    """Centreline mean axial velocity stays laminar while stepping.

    Steps a small random perturbation (iterative-cn) and
    interpolates the axially+azimuthally averaged axial velocity
    (base flow + mean-mode ``u_z`` perturbation, an even profile)
    to the missing ``r = 0`` point: it must stay near the laminar
    centreline value ``U_z(0) = 1`` while the perturbation is
    small.
    """
    from dnsjax.flows.wall_bounded.pipe import (
        predict_and_fully_correct,
        to_solver_basis,
    )
    from dnsjax.ic.random_field import generate_random_state

    amp = 1e-3
    # ICs are built in physical components; the stepper works in the
    # solver basis (``__main__`` performs this same single crossing).
    state = to_solver_basis(generate_random_state(amp, 0.4, 3))
    rs = np.asarray(pipe_flow.rs)

    def centreline(s) -> float:
        mean_uz = jnp.real(extract_mean_mode(s)[0])
        profile = pipe_flow.base_flow[0, :, 0, 0] + mean_uz
        return float(interpolate_to_axis(profile, rs, parity="even"))

    tol = 10.0 * amp
    v0 = centreline(state)
    assert abs(v0 - 1.0) < tol, f"t=0 centreline {v0}"
    for _ in range(20):
        # predict_and_fully_correct donates its argument; the loop
        # rebinds state, so nothing reuses the donated buffer.
        state, err, _ = predict_and_fully_correct(state)
        assert float(err) < params.step.corrector_tolerance, (
            f"corrector not converged (err {float(err):.3e})"
        )
    v1 = centreline(state)
    assert abs(v1 - 1.0) < tol, f"t=0.2 centreline {v1}"


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
