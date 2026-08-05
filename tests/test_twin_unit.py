r"""Unit tests for the twin-run diagnostics (``dnsjax.twin_diagnostics``).

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
from numpy.testing import assert_allclose  # noqa: E402

from dnsjax import twin_diagnostics as td  # noqa: E402
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
from dnsjax.twin import twin_params  # noqa: E402

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
    tvals = {k: float(v) for k, v in td.twin_energies(state1, state2).items()}
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
    tvals = td.twin_energies(state1, state2)
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
    from dnsjax.random_field import generate_random_state

    e0 = 1e-6
    delta = generate_random_state(math.sqrt(2.0 * e0), 0.4, seed=7)
    e_delta = float(get_perturbation_energy(delta))
    assert_allclose(e_delta, e0, rtol=1e-12)

    # The driver's exact-rescale guard is a no-op up to rounding.
    factor = math.sqrt(e0 / e_delta)
    assert abs(factor - 1.0) < 1e-12

    state1 = generate_random_state(0.05, 0.4, seed=11)
    tvals = td.twin_energies(state1, state1 + delta * factor)
    # (state1 + delta) - state1 cancels state1 to eps * |state1|,
    # which is eps * (|state1|/|delta|) relative to delta.
    assert_allclose(float(tvals["E_d"]), e0, rtol=1e-10)
    print("e0 energy convention: OK")


# ── Budget terms vs an independent fine-grid reference ───────────────

FINE = 32  # 4x the mode counts: exact xz quadrature for the cubic
KZ_HARM = [int(m) for m in complex_harmonics(params.res.nz)]


def _hermitian_spec(seed: int, amp: float = 0.1) -> np.ndarray:
    """Random host spectral state with *real* physical fields.

    The ``kx = 0`` plane is made Hermitian (``complex_harmonics``
    pairs index ``i2`` with ``N2_TRUE - i2``) and the mean column
    real, as for any pair of genuine solver states.
    """
    rng = np.random.default_rng(seed)
    spec = np.zeros((3, NY, N2_SPEC, N3_SPEC), dtype=np.complex128)
    spec[:, :, :N2_TRUE, :N3_TRUE] = amp * (
        rng.standard_normal((3, NY, N2_TRUE, N3_TRUE))
        + 1j * rng.standard_normal((3, NY, N2_TRUE, N3_TRUE))
    )
    spec[:, :, 0, 0] = spec[:, :, 0, 0].real
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
        # (the twin_diagnostics "Dissipation form" note): horizontal
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
    """All 30 budget outputs match the independent reference."""
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
    tvals = {k: float(v) for k, v in td.twin_energies(s1, s2).items()}
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
    twin_params.seed = 1

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


if __name__ == "__main__":
    test_masks_partition()
    test_energy_partition()
    test_energies_vs_numpy()
    test_e0_convention()
    test_budget_vs_numpy()
    test_budget_frame_invariance()
    test_spectra_sum_identity()
    test_validate_hook()
    print("All twin unit tests passed.")
