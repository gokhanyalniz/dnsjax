r"""First-principles checks of the Kolmogorov ``get_stats`` diagnostics.

Pins every stats column of the triply-periodic monochromatic flow
against an independent full-spectrum NumPy evaluation on a random
divergence-free band-limited state:

- ``E'``: Parseval `$\sum_k \|\hat{u}'_k\|^2 / 2$` over the full
  (Hermitian-completed) mode set;
- ``E``: the same sum for `$\mathbf{u}' + \mathbf{U}$` with the
  Kolmogorov base flow added spectrally;
- ``I``: `$\langle \mathbf{F}\cdot\mathbf{u}\rangle$` with the forcing
  `$\mathbf{F} = F\,\sin(2\pi y/L_y)\,\hat{\mathbf{x}}$` placed at its
  `$\pm q_f$` coefficients;
- ``D``: `$\nu\,\langle|\nabla \mathbf{u}|^2\rangle
  = \sum_k k^2 \|\hat{u}_k\|^2 / \mathrm{Re}$` -- the guard for the
  Hermitian real-FFT weight (2 for `$k_x > 0$`) in ``get_enstrophy``:
  dropping it under-counts a generic field's dissipation by ~2x.

All references sum the *stored* modes with the ``k_metric`` weight,
which the module docstring's earlier NumPy study verified equals both
the full-spectrum Parseval sum and the physical-space
`$\langle|\nabla u|^2\rangle$`.  The laminar limit (zero perturbation)
is pinned too: `$E = E_{\mathrm{lam}}$`, `$I = D = I_{\mathrm{lam}}$`,
`$E' = 0$`.

Offline, single CPU device, in-process singletons (configured once at
module top via ``update_parameters`` -- the tilt factors and the flow
derive hook only run there, and ``unit_force`` reads
``derived_params.cos_tilt``).

Run as a script::

    uv run python tests/test_monochromatic.py
"""

import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from dnsjax.bootstrap import (  # noqa: E402
    configure_jax_platform,
    platform_from_argv,
)
from dnsjax.parameters import (  # noqa: E402
    Geometry,
    Parameters,
    Physics,
    Resolution,
    padded_res,
    params,
    update_parameters,
)

# ── Configure singletons (before importing sharding/geometry) ────────
NX, NY, NZ = 16, 16, 16
RE = 40.0
update_parameters(
    Parameters(
        phys=Physics(system="kolmogorov", re=RE),
        geo=Geometry(lx=4.0, lz=4.0),
        res=Resolution(nx=NX, ny=NY, nz=NZ, double_precision=True),
    )
)
padded_res.set_padded_resolution(params)

configure_jax_platform(platform_from_argv(), double_precision=True)

from dnsjax.flows.triply_periodic.monochromatic import (  # noqa: E402
    flow,
    get_stats,
)
from dnsjax.ic.random_field import generate_random_state  # noqa: E402

LY = 4.0  # fixed shear-direction box length (triply_periodic.ly)


def _harmonics(n: int) -> np.ndarray:
    q = (np.arange(n) + n // 2) % n - n // 2
    return np.concatenate([q[: n // 2], q[n // 2 + 1 :]])


def _reference_stats(u: np.ndarray) -> dict[str, float]:
    """Full-spectrum reference E', E, I, D for a stored-layout state."""
    ky = _harmonics(NY)[:, None, None] * (2 * np.pi / LY)
    kz = _harmonics(NZ)[None, :, None] * (2 * np.pi / params.geo.lz)
    kx = np.arange(NX // 2)[None, None, :] * (2 * np.pi / params.geo.lx)
    k2 = kx**2 + ky**2 + kz**2
    k_metric = np.where(kx == 0, 1.0, 2.0)

    # Base flow U_x = sin(2 pi y / L_y): -+0.5j at ky-index of +-1,
    # mean kz = kx = 0 slot.  Forcing F = nu k_f^2 U at the same slots.
    u_total = u.copy()
    u_total[0, 1, 0, 0] += -0.5j
    u_total[0, NY - 2, 0, 0] += 0.5j
    f_hat = np.zeros_like(u)
    famp = np.pi**2 / (4 * RE)
    f_hat[0, 1, 0, 0] = -0.5j * famp
    f_hat[0, NY - 2, 0, 0] = 0.5j * famp

    def _parseval(a: np.ndarray, w: np.ndarray) -> float:
        return float(np.sum(w * np.abs(a) ** 2).real)

    e_pert = _parseval(u, k_metric) / 2
    e_tot = _parseval(u_total, k_metric) / 2
    energy_input = float(np.sum(k_metric * np.conj(f_hat) * u_total).real)
    dissipation = _parseval(u_total, k_metric * k2) / RE
    return {"E'": e_pert, "E": e_tot, "I": energy_input, "D": dissipation}


def main() -> None:
    failed = 0

    # Random divergence-free band-limited state (single device: the
    # stored plane carries no mesh padding at this resolution).
    # (kolmogorov defers ``init.random_mean_flow``: the periodic mean
    # mode is a passive Galilean shift the solver re-zeroes anyway, so
    # the generator drops it unconditionally.)
    state = generate_random_state(1.0, 0.4, 0.4, 0.14, seed=7)
    u = np.asarray(state)
    assert u.shape == (3, NY - 1, NZ - 1, NX // 2), u.shape

    stats = {k: float(v) for k, v in get_stats(state).items()}
    ref = _reference_stats(u)
    for name in ("E'", "E", "I", "D"):
        rel = abs(stats[name] - ref[name]) / max(abs(ref[name]), 1e-300)
        ok = rel < 1e-12
        print(
            f"  {'PASS' if ok else 'FAIL'}  random-state {name}: "
            f"stats {stats[name]:+.12e}  ref {ref[name]:+.12e}  "
            f"rel {rel:.2e}"
        )
        failed += 0 if ok else 1

    # Laminar limit: E = E_lam, I = D = I_lam, E' = 0.
    import jax.numpy as jnp

    zeros = jnp.zeros_like(state)
    lam = {k: float(v) for k, v in get_stats(zeros).items()}
    i_lam = float(np.pi**2 / (8 * RE))
    checks = {
        "E": (lam["E"], float(flow.ekin_lam)),
        "I": (lam["I"], i_lam),
        "D": (lam["D"], i_lam),
        "E'": (lam["E'"], 0.0),
    }
    for name, (got, want) in checks.items():
        ok = abs(got - want) < 1e-14
        print(
            f"  {'PASS' if ok else 'FAIL'}  laminar {name}: "
            f"got {got:+.12e}  want {want:+.12e}"
        )
        failed += 0 if ok else 1

    if failed:
        print(f"\n{failed} check(s) failed.")
        sys.exit(1)
    print("\nAll monochromatic stats checks passed.")


if __name__ == "__main__":
    main()
