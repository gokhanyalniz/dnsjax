#!/usr/bin/env python3
r"""Generate a random divergence-free perturbation and save as a
single-file (tar-wrapped zarr3) snapshot.

This is a thin CLI wrapper over :mod:`dnsjax.random_field`, which holds
the actual generators (shared with the in-process ``init.random_field``
start mode of ``dnsjax.__main__``).  See that module's docstring for the
generation algorithm, array shapes, and the divergence-free / boundary
enforcement details.

The output snapshot is immediately usable as initial condition::

    uv run python scripts/random_field.py \
        --system plane-couette \
        --nx 128 --ny 65 --nz 128 \
        --amplitude 0.1 --smoothness 0.4 --seed 1 \
        --output random_ic.tar

    uv run python -m dnsjax \
        --init.snapshot random_ic.tar ...

A provided snapshot takes precedence over every in-process init mode,
so no other init flag is needed.

The wavenumber-dependent amplitude of each Fourier mode
`$(k_x, k_z)$` decays as

.. math::
    A(k_x, k_z) = (1 - s)^{|k_x| + |k_z|}

where `$s$` is the ``--smoothness`` parameter and `$k_x, k_z$`
are the physical wavenumbers.  The field is then normalised
so that the volume-averaged L2 norm equals ``--amplitude``.

**Dean flow** (``--system dean``) integrates the *total* field, so the
generated divergence-free perturbation is added to the analytical laminar
Dean profile to form the total-field IC; ``--amplitude`` still sets the
perturbation norm.

Run ``--test`` for self-verification: it checks the configured system's
generator (divergence-free, truncation-level wall BCs, norm, mean-mode,
Hermitian symmetry, seed determinism, and for Dean the total-field wall
BCs) and exits with a pass/fail status, writing no snapshot.  The
wall-normal velocity uses a squared no-slip window, so the
continuity-derived component's wall BC is truncation-level rather than
exact (the first corrector step projects it out).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ── CLI ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a random divergence-free perturbation "
            "and save as a zarr3 snapshot."
        ),
    )
    # Simulation parameters
    ap.add_argument(
        "--system",
        required=True,
        choices=[
            "plane-couette",
            "plane-poiseuille",
            "pipe",
            "taylor-couette",
            "dean",
            "kolmogorov",
            "waleffe",
            "decaying-box",
        ],
    )
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--lx", type=float, default=4.0)
    ap.add_argument("--lz", type=float, default=4.0)
    ap.add_argument("--re", type=float, default=1000.0)
    # Taylor-Couette control parameters.
    ap.add_argument("--re1", type=float, default=None)
    ap.add_argument("--re2", type=float, default=None)
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--fd-order", type=int, default=4)
    ap.add_argument("--tilt-degree", type=float, default=0.0)
    ap.add_argument("--wall-grid", type=str, default=None)
    ap.add_argument("--single-precision", action="store_true")
    ap.add_argument(
        "--driving",
        choices=["constant_pressure_gradient", "constant_bulk_velocity"],
        default="constant_pressure_gradient",
    )
    ap.add_argument(
        "--block-mean-spanwise-velocity",
        action="store_true",
    )
    ap.add_argument(
        "--parameters-toml",
        type=str,
        default=None,
        help="Load simulation parameters from a TOML file.",
    )

    # Random-field parameters
    ap.add_argument(
        "--amplitude",
        type=float,
        default=0.1,
        help="Target L2 norm of the perturbation.",
    )
    ap.add_argument(
        "--smoothness",
        type=float,
        default=0.4,
        help="Spectral decay rate (0 < s < 1). Higher = smoother.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (NumPy PRNG, device-count independent).",
    )
    ap.add_argument(
        "--mean-flow",
        action="store_true",
        help="Also perturb the mean mode (kx=kz=0).",
    )

    # Output
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the snapshot tar file (e.g. "
        "random_ic.tar). Required unless --test is given.",
    )

    # Self-test
    ap.add_argument(
        "--test",
        action="store_true",
        help="Run self-contained verification tests and exit "
        "(no snapshot is written).",
    )
    args = ap.parse_args()
    if not args.test and args.output is None:
        ap.error("--output is required unless --test is given")
    return args


# ── JAX + singleton setup ────────────────────────────────────────


def _setup_jax_and_params(args: argparse.Namespace) -> None:
    """Configure JAX for CPU and set the global parameter
    singletons exactly as ``__main__.py`` does."""
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    os.environ["NPROC"] = "1"

    import jax

    double = not args.single_precision
    jax.config.update("jax_enable_x64", double)
    jax.config.update("jax_platforms", "cpu")

    from dnsjax.parameters import (
        Parameters,
        padded_res,
        params,
        read_parameters,
        update_parameters,
    )

    if args.parameters_toml is not None:
        params_in = read_parameters(Path(args.parameters_toml))
        update_parameters(params_in)

    cli_params = Parameters(
        dist={"np": 1, "platform": "cpu"},
        phys={
            "system": args.system,
            "re": args.re,
            "re1": args.re1,
            "re2": args.re2,
            "driving": args.driving,
            "block_mean_spanwise_velocity": (
                args.block_mean_spanwise_velocity
            ),
        },
        geo={
            "lx": args.lx,
            "lz": args.lz,
            "tilt_degree": args.tilt_degree,
            "eta": args.eta,
            "wall_grid": args.wall_grid,
        },
        res={
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "fd_order": args.fd_order,
            "double_precision": double,
        },
        outs={},
    )
    update_parameters(cli_params)
    padded_res.set_padded_resolution(params)


# ── Self-test ────────────────────────────────────────────────────


def _run_tests() -> None:
    """Run self-contained verification for the configured geometry.

    The generation singletons (Fourier / sharding) are built from
    ``params`` at import time, so the self-test runs the block for the
    configured ``--system`` rather than switching systems mid-process.
    Generators come from :mod:`dnsjax.random_field`; grids/operators
    used by the checks are rebuilt here via the ``build_*_grid``
    helpers.
    """
    from jax import numpy as jnp

    from dnsjax.parameters import (
        annular_systems,
        cartesian_systems,
        cylindrical_systems,
        params,
    )
    from dnsjax.random_field import (
        add_dean_laminar,
        generate_annular,
        generate_cartesian,
        generate_cylindrical,
    )

    passed = 0
    failed = 0

    def _check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  PASS: {name}")
        else:
            failed += 1
            print(f"  FAIL: {name}  {detail}")

    system = params.phys.system

    if system in cartesian_systems:
        print(f"Cartesian ({system}):")
        from dnsjax.geometries.wall_bounded._base import get_norm
        from dnsjax.geometries.wall_bounded.cartesian import (
            build_cartesian_grid,
            fourier,
        )

        state = generate_cartesian(
            args.amplitude, args.smoothness, args.seed, args.mean_flow
        )
        _, D1, _, y_weights = build_cartesian_grid(
            params.res.ny, params.res.fd_order, params.geo.wall_grid
        )
        D1_np = np.asarray(D1)

        Nkz = params.res.nz - 1
        Nkx = params.res.nx // 2
        state_np = np.asarray(state)
        kx_np = np.asarray(fourier.kx).ravel()
        kz_np = np.asarray(fourier.kz).ravel()

        # Divergence-free.
        max_div = 0.0
        for iz in range(Nkz):
            for ix in range(Nkx):
                kx_v = kx_np[ix]
                kz_v = kz_np[iz]
                dv_dy = D1_np @ state_np[1, :, iz, ix]
                div = (
                    1j * kx_v * state_np[0, :, iz, ix]
                    + dv_dy
                    + 1j * kz_v * state_np[2, :, iz, ix]
                )
                max_div = max(max_div, float(np.max(np.abs(div))))
        _check(
            "divergence-free", max_div < 1e-10, f"max |div| = {max_div:.2e}"
        )

        # Wall BCs.  The wall-normal velocity carries a squared no-slip
        # window so the independent components vanish exactly at the walls;
        # the continuity-derived component is only truncation-level (it is
        # projected onto the divergence-free space by the first corrector
        # step), so the tolerance is a small fraction of the amplitude
        # rather than roundoff.
        bc_err = max(
            float(np.max(np.abs(state_np[:, 0]))),
            float(np.max(np.abs(state_np[:, -1]))),
        )
        bc_tol = 0.1 * args.amplitude
        _check("wall BCs", bc_err < bc_tol, f"max |BC| = {bc_err:.2e}")

        # Norm.
        norm = float(get_norm(state, fourier.k_metric, y_weights))
        norm_err = abs(norm - args.amplitude) / args.amplitude
        _check(
            "norm matches target",
            norm_err < 1e-12,
            f"|norm - target| / target = {norm_err:.2e}",
        )

        # Mean mode zero.
        mean_err = float(np.max(np.abs(state_np[:, :, 0, 0])))
        _check(
            "mean mode zero", mean_err < 1e-30, f"max |mean| = {mean_err:.2e}"
        )

        # Hermitian symmetry at kx=0.
        n_pos = params.res.nz // 2
        sym_err = 0.0
        for i in range(1, n_pos):
            j = Nkz - i
            diff = state_np[:, :, j, 0] - np.conj(state_np[:, :, i, 0])
            sym_err = max(sym_err, float(np.max(np.abs(diff))))
        kz0_imag = float(np.max(np.abs(state_np[:, :, 0, 0].imag)))
        sym_err = max(sym_err, kz0_imag)
        _check(
            "Hermitian symmetry at kx=0",
            sym_err < 1e-14,
            f"max sym error = {sym_err:.2e}",
        )

        # Seed determinism.
        state2 = generate_cartesian(
            args.amplitude, args.smoothness, args.seed, args.mean_flow
        )
        det_err = float(jnp.max(jnp.abs(state - state2)))
        _check("seed determinism", det_err == 0.0, f"max diff = {det_err:.2e}")

    elif system in annular_systems:
        print(f"Annular ({system}):")
        from dnsjax.geometries.wall_bounded.annular import (
            build_annular_grid,
            fourier,
            get_norm2_annular,
        )
        from dnsjax.parameters import derived_params

        state = generate_annular(
            args.amplitude, args.smoothness, args.seed, args.mean_flow
        )
        state_np = np.asarray(state)

        Nm = params.res.nz - 1
        Nkz = params.res.nx // 2
        rs, D1, _, y_weights, inv_r = build_annular_grid(
            params.res.ny,
            params.res.fd_order,
            derived_params.r_inner,
            derived_params.r_outer,
            params.geo.wall_grid,
        )
        D1_np = np.asarray(D1)
        inv_r_np = np.asarray(inv_r)
        kz_np = np.asarray(fourier.kz).ravel()
        m_np = np.asarray(fourier.m).ravel()

        # Divergence-free for kz != 0 (kz = 0 modes carry a residual
        # divergence projected out by the first corrector step).
        max_div = 0.0
        for im in range(Nm):
            m_val = int(m_np[im])
            for ik in range(Nkz):
                kz_v = kz_np[ik]
                if kz_v == 0:
                    continue
                up = state_np[1, :, im, ik]
                um = state_np[2, :, im, ik]
                div_r = (
                    D1_np @ up
                    + (m_val + 1) * inv_r_np * up
                    + D1_np @ um
                    + (1 - m_val) * inv_r_np * um
                ) / 2.0
                div = 1j * kz_v * state_np[0, :, im, ik] + div_r
                max_div = max(max_div, float(np.max(np.abs(div))))
        _check(
            "divergence-free (kz!=0)",
            max_div < 1e-10,
            f"max |div| = {max_div:.2e}",
        )

        # Wall BCs (both walls); truncation-level for the
        # continuity-derived component (corrector-projected) -- see the
        # Cartesian branch.
        bc_err = max(
            float(np.max(np.abs(state_np[:, 0]))),
            float(np.max(np.abs(state_np[:, -1]))),
        )
        bc_tol = 0.1 * args.amplitude
        _check("wall BCs", bc_err < bc_tol, f"max |BC| = {bc_err:.2e}")

        # Norm.
        norm = float(
            jnp.sqrt(get_norm2_annular(state, fourier.k_metric, y_weights))
        )
        norm_err = abs(norm - args.amplitude) / args.amplitude
        _check(
            "norm matches target",
            norm_err < 1e-12,
            f"|norm - target| / target = {norm_err:.2e}",
        )

        # Mean mode zero.
        mean_err = float(np.max(np.abs(state_np[:, :, 0, 0])))
        _check(
            "mean mode zero", mean_err < 1e-30, f"max |mean| = {mean_err:.2e}"
        )

        # Hermitian symmetry at kz=0 (over the azimuthal m axis).
        n_pos = params.res.nz // 2
        sym_err = 0.0
        for i in range(1, n_pos):
            j = Nm - i
            diff = state_np[:, :, j, 0] - np.conj(state_np[:, :, i, 0])
            sym_err = max(sym_err, float(np.max(np.abs(diff))))
        kz0_imag = float(np.max(np.abs(state_np[:, :, 0, 0].imag)))
        sym_err = max(sym_err, kz0_imag)
        _check(
            "Hermitian symmetry at kz=0",
            sym_err < 1e-14,
            f"max sym error = {sym_err:.2e}",
        )

        # Seed determinism.
        state2 = generate_annular(
            args.amplitude, args.smoothness, args.seed, args.mean_flow
        )
        det_err = float(jnp.max(jnp.abs(state - state2)))
        _check("seed determinism", det_err == 0.0, f"max diff = {det_err:.2e}")

        # Dean total-field IC: the laminar profile (added to the
        # perturbation) is axisymmetric and zero at both walls, so the
        # total field must still satisfy no-slip.
        if system == "dean":
            total_np = np.asarray(add_dean_laminar(state))
            bc_err = max(
                float(np.max(np.abs(total_np[:, 0]))),
                float(np.max(np.abs(total_np[:, -1]))),
            )
            # The laminar profile vanishes at both walls analytically, so
            # the total wall BC is the perturbation's truncation-level
            # value (corrector-projected).
            _check(
                "dean total-field wall BCs",
                bc_err < 0.1 * args.amplitude,
                f"max |BC| = {bc_err:.2e}",
            )

    elif system in cylindrical_systems:
        print(f"Cylindrical ({system}):")
        from dnsjax.geometries.wall_bounded.cylindrical import (
            build_cylindrical_grid,
            fourier,
            get_norm2_cyl,
        )

        state = generate_cylindrical(
            args.amplitude, args.smoothness, args.seed, args.mean_flow
        )
        state_np = np.asarray(state)

        Nm = params.res.nz - 1
        Nkz = params.res.nx // 2
        rs, D1_even, D1_odd, _, y_weights, inv_r = build_cylindrical_grid(
            params.res.ny, params.res.fd_order, params.geo.wall_grid
        )
        D1_even_np = np.asarray(D1_even)
        D1_odd_np = np.asarray(D1_odd)
        inv_r_np = np.asarray(inv_r)
        kz_np = np.asarray(fourier.kz).ravel()
        m_np = np.asarray(fourier.m).ravel()

        # Divergence-free for kz != 0 (kz = 0 modes carry a residual
        # divergence projected out by the first corrector step).  The
        # radial-derivative operator is parity-selected: u_pm have
        # parity (-1)^{m+1}.
        max_div = 0.0
        for im in range(Nm):
            m_val = int(m_np[im])
            D1_pm = D1_even_np if (m_val + 1) % 2 == 0 else D1_odd_np
            for ik in range(Nkz):
                kz_v = kz_np[ik]
                if kz_v == 0:
                    continue
                up = state_np[1, :, im, ik]
                um = state_np[2, :, im, ik]
                div_r = (
                    D1_pm @ up
                    + (m_val + 1) * inv_r_np * up
                    + D1_pm @ um
                    + (1 - m_val) * inv_r_np * um
                ) / 2.0
                div = 1j * kz_v * state_np[0, :, im, ik] + div_r
                max_div = max(max_div, float(np.max(np.abs(div))))
        _check(
            "divergence-free (kz!=0)",
            max_div < 1e-10,
            f"max |div| = {max_div:.2e}",
        )

        # Wall BC (outer wall r=1 only; the inner end r=0 is the axis,
        # governed by parity/regularity, not a Dirichlet BC).
        # Truncation-level for the continuity-derived component
        # (corrector-projected) -- see the Cartesian branch.
        bc_err = float(np.max(np.abs(state_np[:, -1])))
        bc_tol = 0.1 * args.amplitude
        _check("wall BC (r=1)", bc_err < bc_tol, f"max |BC| = {bc_err:.2e}")

        # Norm.
        norm = float(
            jnp.sqrt(get_norm2_cyl(state, fourier.k_metric, y_weights))
        )
        norm_err = abs(norm - args.amplitude) / args.amplitude
        _check(
            "norm matches target",
            norm_err < 1e-12,
            f"|norm - target| / target = {norm_err:.2e}",
        )

        # Mean mode zero.
        mean_err = float(np.max(np.abs(state_np[:, :, 0, 0])))
        _check(
            "mean mode zero", mean_err < 1e-30, f"max |mean| = {mean_err:.2e}"
        )

        # Hermitian symmetry at kz=0 (over the azimuthal m axis).
        n_pos = params.res.nz // 2
        sym_err = 0.0
        for i in range(1, n_pos):
            j = Nm - i
            diff = state_np[:, :, j, 0] - np.conj(state_np[:, :, i, 0])
            sym_err = max(sym_err, float(np.max(np.abs(diff))))
        kz0_imag = float(np.max(np.abs(state_np[:, :, 0, 0].imag)))
        sym_err = max(sym_err, kz0_imag)
        _check(
            "Hermitian symmetry at kz=0",
            sym_err < 1e-14,
            f"max sym error = {sym_err:.2e}",
        )

        # Seed determinism.
        state2 = generate_cylindrical(
            args.amplitude, args.smoothness, args.seed, args.mean_flow
        )
        det_err = float(jnp.max(jnp.abs(state - state2)))
        _check("seed determinism", det_err == 0.0, f"max diff = {det_err:.2e}")

    else:
        print(
            f"  (self-test implemented for the wall-bounded systems; "
            f"'{system}' skipped)"
        )

    print(f"\n{passed} passed, {failed} failed.")
    sys.exit(1 if failed > 0 else 0)


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    global args
    args = _parse_args()
    _setup_jax_and_params(args)

    from dnsjax.parameters import params

    if args.test:
        _run_tests()
        return

    from dnsjax.random_field import generate_random_state
    from dnsjax.sharding import sharding
    from dnsjax.snapshot import save_snapshot

    system = params.phys.system
    state = generate_random_state(
        args.amplitude, args.smoothness, args.seed, args.mean_flow
    )

    save_snapshot(state, t=0.0, it=0, path=args.output)
    label = (
        "Dean total-field IC (laminar + perturbation)"
        if system == "dean"
        else "random perturbation"
    )
    sharding.print(
        f"Saved {label} to {args.output}\n"
        f"  system={system}, "
        f"resolution=({args.nx}, {args.ny}, {args.nz}), "
        f"amplitude={args.amplitude}, "
        f"smoothness={args.smoothness}, "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()
