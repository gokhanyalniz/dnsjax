r"""Unit tests for the twin-run diagnostics (``dnsjax.twin.diagnostics``).

Offline, on 4 forced host CPU devices with a ``(np0, np1) = (2, 2)``
Explicit mesh so **both** mode axes are genuinely sharded (the
``broadcast_to``-keeps-the-source-spec class of sharding bug only
shows at ``np0 > 1`` *and* ``np1 > 1``):

1. The mean / streak / streamwise-varying masks partition the whole
   padded mode grid (every slot in exactly one mask), so
   ``E_dU + E_du1 + E_du2 == E_d`` and
   ``E_du1_x + E_du1_y + E_du1_z == E_du1`` to rounding on a
   deterministic full-spectrum state pair.
2. Every ``twin_energies`` output equals an independent host-NumPy
   Parseval + quadrature reference built from the harmonic index
   layout alone (true-mode counts, real-FFT weight, quadrature
   weights) -- no ``fourier`` fields reused.
3. The ``e0`` energy convention: a random perturbation generated at
   amplitude `$\sqrt{2 e_0}$` has solver-measure `$E' = e_0$`, and
   the driver's ``state1 + delta`` construction reproduces
   ``E_d == e0`` through ``twin_energies`` to the float cancellation
   floor.
4. Every ``twin_budget`` output -- the 24 advective terms across all
   four evaluation classes, the 3 dissipations, and the consistency
   sums -- equals an independent host-NumPy reference that evaluates
   every term one way (no class split): fine-grid (4x zero-padded)
   ``np.fft`` physical fields, pointwise products (exact for the
   cubic integrand below the fine-grid Nyquist), grid-mean over
   `$xz$`, and the code's own quadrature weights / ``D1``.  The
   states are Hermitian-consistent (real physical fields), as any
   real pair of solver states is.
5. Budget frame invariance: adding one constant to the streamwise
   mean column of *both* states (the ``phys.u_grid`` Galilean shift)
   leaves every budget column unchanged.
6. The ``[twin]`` extension validate hook, dispatched through the
   production paths: stray knobs without ``e0``, ``step.adaptive``
   rejection (both via ``validate_parameters``), and the
   unsupported-system rejection (via ``validate_extensions``).

Run as a script via ``uv run python tests/test_twin_unit.py``.
"""

from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(line_buffering=True)

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

# Configure JAX and the parameter singletons before importing sharding
# or any geometry module (the bootstrap contract); the twin module
# config is the plane-Couette minimal flow unit at a tiny resolution.
from dnsjax.bootstrap import configure_jax_platform  # noqa: E402

configure_jax_platform("cpu")

from dnsjax.parameters import (  # noqa: E402
    Parameters,
    padded_res,
    params,
    update_parameters,
)

update_parameters(
    Parameters(
        phys={"system": "plane-couette", "re": 400.0},
        geo={"lx": 5.497787143782138, "lz": 3.7699111843077517},
        res={
            "nx": 8,
            "ny": 9,
            "nz": 8,
            "fd_order": 4,
            "double_precision": True,
        },
        dist={"np0": 2, "np1": 2},
    )
)
padded_res.set_padded_resolution(params)

import math  # noqa: E402

import numpy as np  # noqa: E402
from numpy.testing import (  # noqa: E402
    assert_allclose,
    assert_array_equal,
)

from dnsjax.extensions import validate_extensions  # noqa: E402
from dnsjax.flows.wall_bounded.plane_couette import (  # noqa: E402
    get_perturbation_energy,
)
from dnsjax.geometries.wall_bounded.cartesian import fourier  # noqa: E402
from dnsjax.harmonics import complex_harmonics  # noqa: E402
from dnsjax.parameters import (  # noqa: E402
    derived_params,
    validate_parameters,
)
from dnsjax.sharding import sharding  # noqa: E402
from dnsjax.snapshot import assemble_local_shards  # noqa: E402
from dnsjax.twin import diagnostics as td  # noqa: E402
from dnsjax.twin.driver import twin_params  # noqa: E402

NY = params.res.ny
N2_TRUE = params.res.nz - 1  # true complex (kz) modes
N3_TRUE = params.res.nx // 2  # true real-FFT (kx) modes
NY_SPEC, N2_SPEC, N3_SPEC = sharding.spec_shape


def _mode_column(i2: int, i3: int, salt: float) -> np.ndarray:
    """Deterministic dense ``(3, NY)`` complex column for mode (i2, i3)."""
    c = np.arange(3)[:, None]
    j = np.arange(NY)[None, :]
    phase = 0.1 * c + 0.2 * j + 0.3 * i2 + 0.4 * i3 + salt
    return (1.0 + 0.05 * c + 0.01 * j + 0.02 * i2 + 0.03 * i3) * np.exp(
        1j * phase
    )


def _make_state(salt: float):
    """Sharded spectral state with every *true* mode filled."""

    def fill_local(buf, kz_start, nkz, kx_start, nkx):
        for li in range(nkz):
            for lj in range(nkx):
                g2, g3 = kz_start + li, kx_start + lj
                buf[:, :, li, lj] = _mode_column(g2, g3, salt)

    return assemble_local_shards(fill_local)


def _host_state(salt: float) -> np.ndarray:
    """The same state as ``_make_state`` on the host, padding zeroed."""
    full = np.zeros((3, NY, N2_SPEC, N3_SPEC), dtype=np.complex128)
    for i2 in range(N2_TRUE):
        for i3 in range(N3_TRUE):
            full[:, :, i2, i3] = _mode_column(i2, i3, salt)
    return full


def _host_energies(delta: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Host-NumPy reference of every ``twin_energies`` output.

    Built from the harmonic index layout alone: true modes occupy the
    leading ``N2_TRUE`` / ``N3_TRUE`` slots, the real-FFT weight is 1
    on the ``kx``-index-0 column and 2 elsewhere, and the masks are
    index tests -- independent of ``fourier``'s device arrays.
    """
    w = np.asarray(td.flow.y_weights)
    vf = derived_params.volume_fac
    k_metric = np.full(N3_SPEC, 2.0)
    k_metric[0] = 1.0

    def energy(field: np.ndarray) -> float:
        mode_sum = np.sum(
            (np.abs(field) ** 2) * k_metric[None, None, None, :],
            axis=(0, 2, 3),
        )
        return float(w @ mode_sum) / vf / 2.0

    m_mean = np.zeros((N2_SPEC, N3_SPEC), dtype=bool)
    m_mean[0, 0] = True
    m_u1 = np.zeros_like(m_mean)
    m_u1[1:, 0] = True  # kx index 0, kz index != 0 (padding included)
    m_u2 = np.zeros_like(m_mean)
    m_u2[:, 1:] = True  # kx index != 0 (padding included)
    du1 = delta * m_u1[None, None]
    return {
        "E_d": energy(delta),
        "E_dU": energy(delta * m_mean[None, None]),
        "E_du1": energy(du1),
        "E_du1_x": energy(du1[0:1]),
        "E_du1_y": energy(du1[1:2]),
        "E_du1_z": energy(du1[2:3]),
        "E_du2": energy(delta * m_u2[None, None]),
        "E_ref": energy(ref),
    }


# ── Masks and energy partition ───────────────────────────────────────


def test_masks_partition() -> None:
    """The three masks cover every padded mode slot exactly once."""
    m_mean, m_u1, m_u2 = (np.asarray(m) for m in td.component_masks(fourier))
    total = (
        m_mean.astype(int)
        + m_u1.astype(int)
        + np.broadcast_to(m_u2, m_mean.shape).astype(int)
    )
    assert total.shape == (1, N2_SPEC, N3_SPEC)
    assert (total == 1).all(), "masks do not partition the mode grid"
    assert m_mean.sum() == 1 and m_mean[0, 0, 0]
    print("masks partition the padded mode grid: OK")


def test_energy_partition() -> None:
    """Component energies sum to the total to rounding."""
    state1 = _make_state(salt=0.0)
    state2 = _make_state(salt=1.0)
    tvals = {
        k: float(v)
        for k, v in td.twin_energies(state1, state2, bins=True).items()
    }
    assert_allclose(
        tvals["E_dU"] + tvals["E_du1"] + tvals["E_du2"],
        tvals["E_d"],
        rtol=1e-13,
        err_msg="component energies do not partition E_d",
    )
    assert_allclose(
        tvals["E_du1_x"] + tvals["E_du1_y"] + tvals["E_du1_z"],
        tvals["E_du1"],
        rtol=1e-13,
        err_msg="velocity components do not partition E_du1",
    )
    assert all(v > 0 for v in tvals.values())
    print("energy partition identities: OK")


def test_energies_vs_numpy() -> None:
    """Every output matches the independent host reference."""
    state1 = _make_state(salt=0.0)
    state2 = _make_state(salt=1.0)
    tvals = td.twin_energies(state1, state2, bins=True)
    ref1 = _host_state(salt=0.0)
    ref2 = _host_state(salt=1.0)
    expected = _host_energies(ref2 - ref1, ref1)
    assert set(tvals) == set(expected)
    for name, value in expected.items():
        assert_allclose(float(tvals[name]), value, rtol=1e-12, err_msg=name)
    print("twin_energies vs host NumPy reference: OK")


# ── The e0 energy convention ─────────────────────────────────────────


def test_e0_convention() -> None:
    r"""``amplitude = sqrt(2 e0)`` gives solver-measure `$E' = e_0$`,
    and the driver's additive construction reproduces it through
    ``twin_energies`` to the float cancellation floor."""
    from dnsjax.ic.random_field import generate_random_state

    e0 = 1e-6
    delta = generate_random_state(math.sqrt(2.0 * e0), 0.4, seed=7)
    e_delta = float(get_perturbation_energy(delta))
    assert_allclose(e_delta, e0, rtol=1e-12)

    # The driver's exact-rescale guard is a no-op up to rounding.
    factor = math.sqrt(e0 / e_delta)
    assert abs(factor - 1.0) < 1e-12

    state1 = generate_random_state(0.05, 0.4, seed=11)
    tvals = td.twin_energies(state1, state1 + delta * factor, bins=True)
    # (state1 + delta) - state1 cancels state1 to eps * |state1|,
    # which is eps * (|state1|/|delta|) relative to delta.
    assert_allclose(float(tvals["E_d"]), e0, rtol=1e-10)
    print("e0 energy convention: OK")


# ── Budget terms vs an independent fine-grid reference ───────────────

FINE = 32  # 4x the mode counts: exact xz quadrature for the cubic
KZ_HARM = [int(m) for m in complex_harmonics(params.res.nz)]


def _hermitian_spec(seed: int, amp: float = 0.1) -> np.ndarray:
    r"""Random host spectral state with *real* physical fields.

    The ``kx = 0`` plane is made Hermitian (``complex_harmonics``
    pairs index ``i2`` with ``N2_TRUE - i2``) and the mean column
    real, as for any pair of genuine solver states.

    Deliberately **not** divergence-free and **not** no-slip (that is
    :func:`_solenoidal_pair`): the point of these states is to drive
    every index-layout path with generic data.  The one structural
    property they do carry is the one the diagnostics are entitled to
    assume -- `$\hat v_{00} \equiv 0$`, forced by continuity at
    `$k^2 = 0$` plus no-slip and exact at every point a state can
    enter the driver (:mod:`dnsjax.twin.diagnostics`, "State
    preconditions").  Without it these states would exercise
    ``term_b_mean``'s `$b_y\,\partial_y$` branch, which no state the
    solver produces ever reaches, and pin a term that is identically
    zero in production.  The reference in :func:`_ref_budget` is
    untouched by this and still evaluates every term the generic way,
    so the cross-check is unweakened: it agrees on the omitted branch
    because that branch is genuinely zero here, not because it was
    told to.
    """
    rng = np.random.default_rng(seed)
    spec = np.zeros((3, NY, N2_SPEC, N3_SPEC), dtype=np.complex128)
    spec[:, :, :N2_TRUE, :N3_TRUE] = amp * (
        rng.standard_normal((3, NY, N2_TRUE, N3_TRUE))
        + 1j * rng.standard_normal((3, NY, N2_TRUE, N3_TRUE))
    )
    spec[:, :, 0, 0] = spec[:, :, 0, 0].real
    spec[1, :, 0, 0] = 0.0  # continuity + no-slip at k^2 = 0
    for i2 in range(1, N2_TRUE):
        j2 = N2_TRUE - i2
        if i2 < j2:
            spec[:, :, j2, 0] = np.conj(spec[:, :, i2, 0])
    return spec


def _device_state(spec: np.ndarray):
    """The sharded device state holding *spec*'s true modes."""

    def fill_local(buf, kz_start, nkz, kx_start, nkx):
        for li in range(nkz):
            for lj in range(nkx):
                buf[:, :, li, lj] = spec[:, :, kz_start + li, kx_start + lj]

    return assemble_local_shards(fill_local)


def _phys_fine(spec: np.ndarray) -> np.ndarray:
    """Real physical field on the (FINE, FINE) grid (exact modes)."""
    fullspec = np.zeros((3, NY, FINE, FINE), dtype=np.complex128)
    for i2, m in enumerate(KZ_HARM):
        for i3 in range(N3_TRUE):
            coeff = spec[:, :, i2, i3]
            fullspec[:, :, m % FINE, i3] = fullspec[:, :, m % FINE, i3] + coeff
            if i3 > 0:
                fullspec[:, :, (-m) % FINE, FINE - i3] = fullspec[
                    :, :, (-m) % FINE, FINE - i3
                ] + np.conj(coeff)
    f = np.fft.ifft2(fullspec, axes=(2, 3), norm="forward")
    assert np.abs(f.imag).max() < 1e-12 * max(1.0, np.abs(f.real).max())
    return f.real


def _ref_budget(spec1: np.ndarray, spec2: np.ndarray) -> dict[str, float]:
    """Independent reference of every ``twin_budget`` output.

    Every advective term is evaluated the same single way -- physical
    fields on the fine grid, pointwise products, xz grid mean, the
    code's quadrature -- with no FFT-free/mean-slot specialisation,
    so it genuinely cross-checks all four evaluation classes.
    """
    kx_mult = np.zeros(N3_SPEC, dtype=complex)
    kx_mult[:N3_TRUE] = 1j * (2 * np.pi / params.geo.lx) * np.arange(N3_TRUE)
    kz_mult = np.zeros(N2_SPEC, dtype=complex)
    kz_mult[:N2_TRUE] = (
        1j * (2 * np.pi / params.geo.lz) * np.asarray(KZ_HARM, dtype=float)
    )
    D1_np = np.asarray(td.flow.D1)
    w = np.asarray(td.flow.y_weights)
    vf = derived_params.volume_fac
    re = params.phys.re

    def dx(spec):
        return kx_mult[None, None, None, :] * spec

    def dz(spec):
        return kz_mult[None, None, :, None] * spec

    def dy(spec):
        return np.einsum("ij,cjkl->cikl", D1_np, spec)

    m_mean = np.zeros((N2_SPEC, N3_SPEC), dtype=bool)
    m_mean[0, 0] = True
    m_u1 = np.zeros_like(m_mean)
    m_u1[1:, 0] = True
    m_u2 = np.zeros_like(m_mean)
    m_u2[:, 1:] = True

    delta = spec2 - spec1
    fields = {
        "dU": delta * m_mean[None, None],
        "du1": delta * m_u1[None, None],
        "du2": delta * m_u2[None, None],
        "ru1": spec1 * m_u1[None, None],
        "ru2": spec1 * m_u2[None, None],
    }
    rU = spec1 * m_mean[None, None]
    rU[:, :, 0, 0] += np.asarray(td.flow.base_flow[:, :, 0, 0])
    fields["rU"] = rU

    def term(a: str, b: str, c: str) -> float:
        ap = _phys_fine(fields[a])
        bp = _phys_fine(fields[b])
        gx = _phys_fine(dx(fields[c]))
        gy = _phys_fine(dy(fields[c]))
        gz = _phys_fine(dz(fields[c]))
        integrand = np.sum(ap * (bp[0] * gx + bp[1] * gy + bp[2] * gz), axis=0)
        return -float(w @ integrand.mean(axis=(1, 2))) / vf

    expected: dict[str, float] = {}
    for kind, table in (("P", td._PRODUCTION), ("T", td._TRANSPORT)):
        for a, b, c in table:
            expected[f"{kind}_{a}({b},{c})"] = term(a, b, c)
    D2_np = np.asarray(td.flow.D2)
    for x in ("dU", "du1", "du2"):
        # The discrete-Laplacian (operator) form, matching the code
        # (the twin/diagnostics.py "Dissipation form" note): horizontal
        # parts spectral, wall-normal via the same D2.
        lap = (
            dx(dx(fields[x]))
            + dz(dz(fields[x]))
            + np.einsum("ij,cjkl->cikl", D2_np, fields[x])
        )
        integrand = -np.sum(_phys_fine(fields[x]) * _phys_fine(lap), axis=0)
        expected[f"eps_{x}"] = float(w @ integrand.mean(axis=(1, 2))) / vf / re
    expected["P_tot"] = sum(
        expected[f"P_{a}({b},{c})"] for a, b, c in td._PRODUCTION
    )
    expected["T_tot"] = sum(
        expected[f"T_{a}({b},{c})"] for a, b, c in td._TRANSPORT
    )
    expected["eps_tot"] = sum(
        expected[f"eps_{x}"] for x in ("dU", "du1", "du2")
    )
    return expected


def test_budget_vs_numpy() -> None:
    r"""All 30 budget outputs match the independent reference.

    ``P_dU(dU,rU)`` is an identical zero on both sides, here and in
    production: it is `$-\langle \Delta U_i\,\Delta V\,
    \partial_y U^{(1)}_i \rangle$` and `$\Delta V \equiv 0$` at the
    mean mode (:func:`_hermitian_spec`).  The paper lists the triad,
    so the column stays; this test cannot discriminate it.
    """
    spec1 = _hermitian_spec(21)
    spec2 = spec1 + _hermitian_spec(22, amp=0.03)
    got = td.twin_budget(_device_state(spec1), _device_state(spec2))
    assert set(got) == set(td.budget_names())
    expected = _ref_budget(spec1, spec2)
    scale = max(abs(v) for v in expected.values())
    for name, value in expected.items():
        assert_allclose(
            float(got[name]),
            value,
            rtol=1e-10,
            atol=1e-13 * scale,
            err_msg=name,
        )
    print("twin_budget vs fine-grid NumPy reference: OK")


def test_budget_frame_invariance() -> None:
    """A Galilean shift of both states changes no budget column."""
    spec1 = _hermitian_spec(31)
    spec2 = spec1 + _hermitian_spec(32, amp=0.05)
    base = td.twin_budget(_device_state(spec1), _device_state(spec2))
    shift = 0.37
    spec1s = spec1.copy()
    spec2s = spec2.copy()
    spec1s[0, :, 0, 0] += shift
    spec2s[0, :, 0, 0] += shift
    shifted = td.twin_budget(_device_state(spec1s), _device_state(spec2s))
    scale = max(abs(float(v)) for v in base.values())
    for name in base:
        assert_allclose(
            float(shifted[name]),
            float(base[name]),
            rtol=1e-9,
            atol=1e-12 * scale,
            err_msg=f"{name} not frame-invariant",
        )
    print("budget frame invariance: OK")


# ── (kz, kx) spectra ─────────────────────────────────────────────────


def test_spectra_sum_identity() -> None:
    """The per-mode spectrum partitions exactly into the energies.

    Total sum == ``E_d``, the ``(0,0)`` entry == ``E_dU``, the rest
    of the ``kx = 0`` column == ``E_du1`` (hence the remainder is
    ``E_du2``), and the reference spectrum sums to ``E_ref`` -- all
    on the (2, 2) mesh, so the scatter-psum gather is exercised on
    both axes.
    """
    spec1 = _hermitian_spec(41)
    spec2 = spec1 + _hermitian_spec(42, amp=0.05)
    s1, s2 = _device_state(spec1), _device_state(spec2)
    sp = {k: np.asarray(v) for k, v in td.twin_spectra_2d(s1, s2).items()}
    tvals = {
        k: float(v) for k, v in td.twin_energies(s1, s2, bins=True).items()
    }
    assert sp["e_delta"].shape == (N2_TRUE, N3_TRUE)
    assert sp["e_ref"].shape == (N2_TRUE, N3_TRUE)
    assert_allclose(sp["e_delta"].sum(), tvals["E_d"], rtol=1e-12)
    assert_allclose(sp["e_ref"].sum(), tvals["E_ref"], rtol=1e-12)
    assert_allclose(sp["e_delta"][0, 0], tvals["E_dU"], rtol=1e-12)
    assert_allclose(sp["e_delta"][1:, 0].sum(), tvals["E_du1"], rtol=1e-12)
    print("spectra sum identities: OK")


# ── [twin] validate hook (production dispatch paths) ────────────────


def _expect_value_error(fragment: str, dispatch) -> None:
    try:
        dispatch()
    except ValueError as exc:
        assert fragment in str(exc), f"{fragment!r} not in {exc}"
        return
    raise AssertionError(f"expected ValueError({fragment!r})")


def test_validate_hook() -> None:
    """Stray knobs, adaptive-dt, and unsupported-system rejections."""
    # Stray secondary knob without e0 (via the full production
    # validate_parameters dispatch).
    twin_params.seed = 5
    _expect_value_error("without twin.e0", validate_parameters)
    # ``None``, not ``1``: unset is what the stray check compares
    # against, and an unset seed is drawn later (:mod:`dnsjax.seeding`).
    twin_params.seed = None

    # The stream-shaping flags are on the same list: left on without
    # ``e0`` they configure nothing, and must say so rather than be
    # silently ignored.
    twin_params.rotational_ybudget = True
    _expect_value_error("rotational_ybudget", validate_parameters)
    twin_params.rotational_ybudget = False

    # Adaptive dt rejected when configured.
    twin_params.e0 = 1e-6
    params.step.adaptive = True
    params.step.dt_max = 0.1
    _expect_value_error("fixed time step", validate_parameters)
    params.step.adaptive = False
    params.step.dt_max = None

    # Unsupported system (validate_extensions is the registry layer
    # validate_parameters dispatches; the core parameter checks of
    # the foreign system do not apply to this config).
    system = params.phys.system
    params.phys.system = "pipe"
    _expect_value_error("Cartesian", lambda: validate_extensions(params))
    params.phys.system = system

    twin_params.e0 = None
    validate_parameters()  # restored config is valid again
    print("[twin] validate hook: OK")


# ── Wall-normal-resolved spectra and budget ──────────────────────────


def _fold(a: np.ndarray) -> np.ndarray:
    """Host-side ``_fold_kz`` on a stripped ``k_z`` axis."""
    npos = params.res.nz // 2
    out = a[..., :npos].copy()
    out[..., 1:] += a[..., npos:][..., ::-1]
    return out


def test_marginal_bins_need_even_nz() -> None:
    r"""``marginal_bin_counts`` refuses an odd ``res.nz``.

    :func:`~dnsjax.twin.diagnostics._fold_kz` pairs stored `$k_z$`
    entry `$j$` with `$-j$`, and at odd `$n_z$` the outermost negative
    mode has no positive partner to fold onto.  Both CLIs reject that
    earlier with a fuller message, so this guard is unreachable
    through either and has to be called directly -- which is the point
    of keeping it with the function whose contract states it.

    ``res.nz`` is read at call time, so poking it rebuilds nothing;
    restored in ``finally`` because the singleton is shared with every
    other test in this module.
    """
    nz = params.res.nz
    try:
        params.res.nz = nz + 1
        _expect_value_error("even res.nz", td.marginal_bin_counts)
    finally:
        params.res.nz = nz
    assert td.marginal_bin_counts() == (nz // 2, params.res.nx // 2)
    print("marginal_bin_counts refuses an odd nz: OK")


def test_yspectra_vs_numpy() -> None:
    """Every stored marginal matches a host index-layout reference."""
    state1, state2 = _make_state(salt=0.0), _make_state(salt=1.0)
    out = {
        k: np.asarray(v)
        for k, v in td.twin_yspectra(state1, state2, x0=True).items()
    }
    npos = params.res.nz // 2
    assert out["e_x"].shape == (3, NY, npos)
    assert out["e_z"].shape == (3, NY, N3_TRUE)
    assert out["e_x0"].shape == (3, NY, npos)
    assert out["e_xz00"].shape == (3, NY)

    delta = _host_state(1.0) - _host_state(0.0)
    k_metric = np.full(N3_SPEC, 2.0)
    k_metric[0] = 1.0
    dens = (
        (np.abs(delta) ** 2)
        * k_metric[None, None, None, :]
        / (2.0 * derived_params.volume_fac)
    )
    assert_allclose(
        out["e_x"], _fold(dens.sum(axis=3)[:, :, :N2_TRUE]), rtol=1e-13, atol=0
    )
    assert_allclose(
        out["e_z"], dens.sum(axis=2)[:, :, :N3_TRUE], rtol=1e-13, atol=0
    )
    assert_allclose(
        out["e_x0"], _fold(dens[:, :, :N2_TRUE, 0]), rtol=1e-13, atol=0
    )
    assert_allclose(out["e_xz00"], dens[:, :, 0, 0], rtol=1e-13, atol=0)
    print("y-resolved marginals vs NumPy: OK")


def test_xz00_is_the_plane_column_exactly() -> None:
    r"""``_xz00`` *is* ``_x0[..., 0]``, and ``x0`` costs nothing else.

    Two claims in one, both of which the gate depends on.  The
    `$(0, 0)$` column is scattered by one device and summed with exact
    zeros from the rest, and the fold leaves index 0 alone, so the two
    routes to that mode are **bit-identical** -- not merely close.
    And with ``x0`` off the dicts carry no `$k_x = 0$` field at all,
    which is what "never traced" looks like from outside
    (:func:`diagnostics._marginals_replicated` builds the block only
    under the flag).
    """
    state1, state2 = _make_state(salt=0.0), _make_state(salt=1.0)
    full = {
        k: np.asarray(v)
        for k, v in td.twin_yspectra(state1, state2, x0=True).items()
    }
    lean = {
        k: np.asarray(v) for k, v in td.twin_yspectra(state1, state2).items()
    }
    for prefix in ("e", "r"):
        assert_array_equal(
            full[f"{prefix}_xz00"], full[f"{prefix}_x0"][:, :, 0]
        )
    assert set(lean) == {f"{p}_{s}" for p in "er" for s in ("x", "z", "xz00")}
    for name, value in lean.items():
        assert_array_equal(value, full[name])

    from dnsjax.twin.pressure import DifferencePressure

    pressure = DifferencePressure(td.flow, fourier)
    yb_full = td.twin_ybudget(state1, state2, pressure, x0=True)
    yb_lean = td.twin_ybudget(state1, state2, pressure)
    terms = td.ybudget_terms(False)
    for name in terms:
        assert_array_equal(
            np.asarray(yb_full[f"{name}_xz00"]),
            np.asarray(yb_full[f"{name}_x0"])[:, 0],
        )
    assert set(yb_lean) == {
        f"{t}_{s}" for t in terms for s in ("x", "z", "xz00")
    }
    print("xz00 == x0[..., 0] bit-for-bit; x0 off drops the plane: OK")


def test_yspectra_fold_is_two_sided() -> None:
    r"""The folded `$k_x$` marginal is the true two-sided spectrum.

    The load-bearing check on :func:`diagnostics._fold_kz`: the stored
    half-plane's ``k_metric`` weight makes an entry the energy of the
    conjugate *pair*, whose partner sits at `$-k_z$`, so the `$k_x$`
    marginal is right only after the `$\pm k_z$` fold.  Built on a
    Hermitian-consistent state (a real physical field), where the full
    two-sided plane can be reconstructed by reflection.  The test also
    asserts the *unfolded* marginal is visibly wrong, so it cannot
    pass by both sides being the same thing.
    """
    kz_h = complex_harmonics(params.res.nz)
    rng = np.random.default_rng(7)
    half = rng.standard_normal(
        (3, NY, N2_TRUE, N3_TRUE)
    ) + 1j * rng.standard_normal((3, NY, N2_TRUE, N3_TRUE))
    for i2, kz in enumerate(kz_h):
        if kz < 0:
            j2 = int(np.nonzero(kz_h == -kz)[0][0])
            half[:, :, i2, 0] = np.conj(half[:, :, j2, 0])
    half[:, :, int(np.nonzero(kz_h == 0)[0][0]), 0].imag = 0.0

    def fill_local(buf, kz_start, nkz, kx_start, nkx):
        for li in range(nkz):
            for lj in range(nkx):
                g2, g3 = kz_start + li, kx_start + lj
                if g2 < N2_TRUE and g3 < N3_TRUE:
                    buf[:, :, li, lj] = half[:, :, g2, g3]

    state = assemble_local_shards(fill_local)
    zero = assemble_local_shards(lambda buf, *a: None)
    out = {k: np.asarray(v) for k, v in td.twin_yspectra(zero, state).items()}

    npos = params.res.nz // 2
    vf = derived_params.volume_fac
    ref_x = np.zeros((3, NY, npos))
    ref_z = np.zeros((3, NY, N3_TRUE))
    for i2, kz in enumerate(kz_h):
        for i3 in range(N3_TRUE):
            for kx, col in (
                (i3, half[:, :, i2, i3]),
                *(((-i3, np.conj(half[:, :, i2, i3])),) if i3 else ()),
            ):
                e = 0.5 * np.abs(col) ** 2 / vf
                kz_eff = kz if kx >= 0 else -kz
                if abs(kz_eff) < npos:
                    ref_x[:, :, abs(kz_eff)] += e
                if abs(kx) < N3_TRUE:
                    ref_z[:, :, abs(kx)] += e
    assert_allclose(out["e_x"], ref_x, rtol=1e-12, atol=0)
    assert_allclose(out["e_z"], ref_z, rtol=1e-12, atol=0)

    km = np.full(N3_TRUE, 2.0)
    km[0] = 1.0
    unfolded = (0.5 * np.abs(half) ** 2 * km[None, None, None, :] / vf).sum(
        axis=3
    )
    naive = _fold(unfolded) * 0 + unfolded[..., :npos]
    rel = np.abs(naive - ref_x).max() / ref_x.max()
    assert rel > 0.1, f"the unfolded marginal is not visibly wrong ({rel:.1e})"
    print(f"fold == two-sided marginal (unfolded is {rel:.0%} off): OK")


def test_yspectra_partition() -> None:
    """The stored marginals recover the three-bin energies exactly."""
    state1, state2 = _make_state(salt=0.0), _make_state(salt=1.0)
    out = {
        k: np.asarray(v)
        for k, v in td.twin_yspectra(state1, state2, x0=True).items()
    }
    tvals = {
        k: float(v)
        for k, v in td.twin_energies(state1, state2, bins=True).items()
    }
    w = np.asarray(td.flow.y_weights)
    got = {
        "E_dU": float(np.einsum("j,cj->", w, out["e_xz00"])),
        "E_du1": float(np.einsum("j,cjk->", w, out["e_x0"][:, :, 1:])),
        "E_du2": float(np.einsum("j,cjk->", w, out["e_x"] - out["e_x0"])),
    }
    for name, value in got.items():
        assert_allclose(value, tvals[name], rtol=1e-13)
    for marg in ("e_x", "e_z"):
        assert_allclose(
            float(np.einsum("j,cjk->", w, out[marg])),
            tvals["E_d"],
            rtol=1e-13,
        )
    print("bin energies and E_d recovered from the marginals: OK")


def _solenoidal_pair(amp: float = 0.01):
    """A divergence-free, no-slip state pair (a real solver state's
    two structural properties, which the index-layout states above
    deliberately lack)."""
    from dnsjax.ic.random_field import generate_random_state

    s1 = generate_random_state(0.05, 0.4, 11, False)
    return s1, s1 + generate_random_state(amp, 0.4, 23, False)


#: Relative floor for a sum that is exactly zero by pointwise
#: algebra: float cancellation over the mode set only.
TRANSFER_TOL = 1e-11


def test_ybudget_sums() -> None:
    r"""`$\sum_k \int$` of the budget densities against ``twin_budget``.

    In the **default convective form** production and the viscous
    term are *algebraic* identities -- the same Parseval sum,
    regrouped -- so they hold to rounding at any resolution; the
    transport terms agree only up to the discrete
    integration-by-parts residual that makes ``T_tot`` nonzero in the
    first place, and are checked on the wall-normal ladder of
    ``tests/test_twin_budget.py`` instead.

    Under ``twin.rotational_ybudget`` the production identity moves:
    `$\sum_k\int(P_U + P_r)$` is ``P_tot + T_tot`` plus the work of a
    gradient, both truncation-limited, while the *classical* lift-up
    density is carried unchanged as ``P_lift`` and still reproduces
    the three ``P_*(*,rU)`` columns exactly.  The viscous identity is
    form-independent, ``V`` being the same array either way.
    """
    from dnsjax.twin.pressure import DifferencePressure

    s1, s2 = _solenoidal_pair()
    pressure = DifferencePressure(td.flow, fourier)
    bud = {k: float(v) for k, v in td.twin_budget(s1, s2).items()}
    w = np.asarray(td.flow.y_weights)
    rows = [f"P_{b}({b},rU)" for b in ("dU", "du1", "du2")]

    for rot in (False, True):
        yb = {
            k: np.asarray(v)
            for k, v in td.twin_ybudget(
                s1, s2, pressure, rotational=rot, x0=True
            ).items()
        }

        def total(name: str, marg: str, yb=yb) -> float:
            return float(np.einsum("j,jk->", w, yb[f"{name}_{marg}"]))

        def bins(name: str, yb=yb) -> np.ndarray:
            x0 = np.einsum("j,jk->k", w, yb[f"{name}_x0"])
            x = np.einsum("j,jk->k", w, yb[f"{name}_x"])
            return np.array([x0[0], x0[1:].sum(), (x - x0).sum()])

        # Form-independent: V is literally the same array.
        for marg in ("x", "z"):
            assert_allclose(
                -total("V", marg),
                bud["eps_tot"],
                rtol=1e-12,
                err_msg=f"viscous term over the {marg} marginal (rot={rot})",
            )
        assert_allclose(
            -bins("V"),
            [bud[f"eps_{b}"] for b in ("dU", "du1", "du2")],
            rtol=1e-11,
            atol=1e-30,
        )

        # Production: the whole of it convectively, the mean-gradient
        # part of it (as P_lift) rotationally.
        if rot:
            for marg in ("x", "z"):
                assert_allclose(
                    total("P_lift", marg),
                    sum(bud[r] for r in rows),
                    rtol=1e-12,
                    err_msg=f"P_lift over the {marg} marginal",
                )
            assert_allclose(
                bins("P_lift"), [bud[r] for r in rows], rtol=1e-11, atol=1e-30
            )
        else:
            for marg in ("x", "z"):
                assert_allclose(
                    total("P_U", marg) + total("P_r", marg),
                    bud["P_tot"],
                    rtol=1e-12,
                    err_msg=f"P over the {marg} marginal",
                )
            assert_allclose(
                bins("P_U") + bins("P_r"),
                [
                    sum(v for k, v in bud.items() if k.startswith(f"P_{b}("))
                    for b in ("dU", "du1", "du2")
                ],
                rtol=1e-11,
                atol=1e-30,
            )
    print("budget k-sums and k-set bins vs twin_budget, both forms: OK")


def test_ybudget_transfer_split() -> None:
    r"""What each form's transfer terms do to `$\sum_k T(y)$`.

    The two forms differ here **physically**, not by an error, and
    the test pins both sides of that so neither can drift:

    - **rotational**: `$\sum_k T(y) = 0$` at every `$y$`, exactly.
      Both terms are `$\Delta\hat{\mathbf{u}}^*\cdot(\Delta\mathbf{u}
      \times\mathbf{b})$` and `$\mathbf{a}\cdot(\mathbf{a}\times
      \mathbf{b}) = 0$` pointwise, so the `$k$`-sum is the `$xz$`-mean
      of a pointwise-zero field (Parseval, exact because the 3/2 rule
      represents the quadratic product).  It needs no structural
      property of the states -- not `$\nabla\cdot\Delta\mathbf{u} =
      0$`, not no-slip, not `$\Delta\boldsymbol{\omega} = \nabla\times
      \Delta\mathbf{u}$` -- so the discriminating axis is the state,
      not `$N_y$`, and two very different pairs are used.
    - **convective**: `$\sum_k T(y) = -\partial_y\langle v^{(2)}
      |\Delta\mathbf{u}|^2/2\rangle_{xz}$` -- the turbulent transport
      of difference energy, carried by the **perturbed** member: the
      two terms advect `$\Delta\mathbf{u}$` by `$\mathbf{u}'^{(1)}$`
      and by `$\Delta\mathbf{u}'$`, and the mean advectors they drop
      contribute exactly zero, so the flux sums to
      `$\mathbf{u}^{(1)} + \Delta\mathbf{u}$`.  A genuine wall-normal
      flux, zero only after integrating in `$y$`.  Asserted
      **nonzero** here: a convective build that produced zero would
      have lost that transport.

    The two forms do not differ by one block.  ``Wp`` rotationally
    absorbs `$-\partial_y\langle\phi\,\Delta v\rangle$` with
    `$\phi = \mathbf{u}^{(1)}\!\cdot\Delta\mathbf{u}
    + |\Delta\mathbf{u}|^2/2$`, whose `$|\Delta\mathbf{u}|^2/2$` half
    is exactly the `$\Delta v$`-mediated half of the convective flux;
    the `$\mathbf{u}^{(1)}$`-mediated half goes into the rotational
    production instead.
    """
    from dnsjax.twin.pressure import DifferencePressure

    pressure = DifferencePressure(td.flow, fourier)
    base = _hermitian_spec(51)
    pairs = {
        "generic": (
            _device_state(base),
            _device_state(base + _hermitian_spec(52, amp=0.03)),
        ),
        "solenoidal": _solenoidal_pair(),
    }

    def net(yb, name):
        """``max_y |sum_k T(y)|`` relative to ``max_y sum_k |T|``."""
        marg = yb[f"{name}_x"]  # (ny, n_kz), already summed over k_x
        return float(
            np.abs(marg.sum(axis=-1)).max()
            / max(np.abs(marg).sum(axis=-1).max(), 1e-300)
        )

    worst = 0.0
    for label, (s1, s2) in pairs.items():
        rot = {
            k: np.asarray(v)
            for k, v in td.twin_ybudget(
                s1, s2, pressure, rotational=True
            ).items()
        }
        for name in ("T_vort", "T_self"):
            rel = net(rot, name)
            worst = max(worst, rel)
            assert rel < TRANSFER_TOL, (
                f"{label}: max_y |sum_k {name}(y)| is {rel:.2e} of its "
                f"own magnitude -- not a redistribution"
            )
        con = {
            k: np.asarray(v)
            for k, v in td.twin_ybudget(
                s1, s2, pressure, rotational=False
            ).items()
        }
        flux = max(net(con, "T_ref"), net(con, "T_self"))
        assert flux > 1e-3, (
            f"{label}: the convective transfer terms sum to "
            f"{flux:.2e} of their magnitude over k -- the turbulent "
            f"transport of difference energy has gone missing"
        )
    print(
        f"transfer split: rotational redistributes exactly ({worst:.1e}), "
        "convective carries the transport flux: OK"
    )


def test_nonlinear_matches_solver() -> None:
    r"""The rotational ``n_hat`` **is** the solver's own term.

    `$\mathbf{u}^{(1)}\times\Delta\boldsymbol{\omega} +
    \Delta\mathbf{u}\times\boldsymbol{\omega}^{(1)} +
    \Delta\mathbf{u}\times\Delta\boldsymbol{\omega}$` is exactly what
    ``cartesian._get_rhs`` forms on each state, so under
    ``twin.rotational_ybudget`` the diagnostic's Poisson source can be
    pinned against the solver rather than argued.  It pins the sign,
    all five terms of the mean/fluctuation split, and the
    moving-frame contribution at once.  The convective form has no
    such counterpart -- the solver never assembles that operator --
    which is one of the two things it trades away.

    The two mean-profile halves are evaluated spectrally here and in
    padded physical space there; they agree *exactly* because a
    `$k = 0$` profile times a fluctuation cannot alias.  The partner
    is deliberately large, so the difference of the two RHS
    evaluations is not cancellation-dominated.
    """
    from dnsjax.geometries.wall_bounded.cartesian import _get_rhs
    from dnsjax.twin.pressure import DifferencePressure

    s1, s2 = _solenoidal_pair(amp=0.5)
    pressure = DifferencePressure(td.flow, fourier)
    got = np.asarray(
        td.twin_pressure_check(s1, s2, pressure, rotational=True)["n_hat"]
    )
    r1 = np.asarray(_get_rhs(s1, fourier, td.flow))
    r2 = np.asarray(_get_rhs(s2, fourier, td.flow))
    scale = max(np.abs(r1).max(), np.abs(r2).max())
    err = np.abs(got - (r2 - r1)).max() / scale
    assert err < 1e-12, (
        f"n_hat departs from the solver's own RHS difference by "
        f"{err:.2e} relative"
    )
    print(f"rotational n_hat == solver RHS difference ({err:.1e}): OK")


def test_pressure_solve() -> None:
    r"""The difference pressure solves what it claims to solve.

    Three independent identities, none a restatement of the solve:
    the interior Poisson equation; the wall closure that was actually
    imposed (`$D_1(\partial_t\Delta\hat v)|_w = 0$`, machine-exact
    at every mode but `$(0,0)$`, where the influence matrix is
    structurally singular and `$\Delta\hat v \equiv 0$` makes the
    condition vacuous); and the fact that the analytic Neumann
    condition is *not* zero -- the IMM closure declines it, and a
    test that found it satisfied would mean the wrong closure ran.
    """
    from dnsjax.twin.pressure import DifferencePressure

    s1, s2 = _solenoidal_pair()
    pressure = DifferencePressure(td.flow, fourier)
    out = {
        k: np.asarray(v)
        for k, v in td.twin_pressure_check(s1, s2, pressure).items()
    }

    rhs = np.abs(out["div_n"][1:-1]).max()
    assert np.abs(out["poisson"]).max() < 1e-11 * rhs, (
        f"interior Poisson residual {np.abs(out['poisson']).max():.2e} "
        f"against a right-hand side of {rhs:.2e}"
    )

    scale = np.abs(out["dy_dtv"]).max()
    closure = out["closure"].reshape(2, -1)
    # Column 0 of the flattened (kz, kx) plane is the mean mode.
    off_mean = np.abs(np.delete(closure, 0, axis=1))
    assert off_mean.max() < 1e-11 * scale, (
        f"the imposed wall closure is not met: {off_mean.max():.2e} "
        f"against a scale of {scale:.2e}"
    )

    neumann = np.abs(out["neumann"]).max()
    assert neumann > 1e-8 * scale, (
        "the analytic Neumann condition came out satisfied; the IMM "
        "closure cannot have been the one imposed"
    )
    print(
        "pressure: Poisson + imposed wall closure exact "
        f"({off_mean.max() / scale:.1e} relative), Neumann residual "
        f"{neumann / scale:.1e} as expected: OK"
    )


if __name__ == "__main__":
    test_masks_partition()
    test_energy_partition()
    test_energies_vs_numpy()
    test_e0_convention()
    test_budget_vs_numpy()
    test_budget_frame_invariance()
    test_spectra_sum_identity()
    test_marginal_bins_need_even_nz()
    test_yspectra_vs_numpy()
    test_xz00_is_the_plane_column_exactly()
    test_yspectra_fold_is_two_sided()
    test_yspectra_partition()
    test_ybudget_sums()
    test_ybudget_transfer_split()
    test_nonlinear_matches_solver()
    test_pressure_solve()
    test_validate_hook()
    print("All twin unit tests passed.")
