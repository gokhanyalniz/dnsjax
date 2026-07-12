r"""Inject a single-mode perturbation into an existing dnsjax snapshot.

Offline CLI + library (the ``snapshot_import.py`` sibling for
*existing* dnsjax snapshots): loads a snapshot, adds a scaled
``(C, Ny)`` complex wall-normal profile at one global spectral mode
``(i2, i3)`` -- with the real-FFT conjugate partner handled by
:func:`dnsjax.analysis.transient_growth.single_mode_state` -- and
writes a new snapshot.  The parent's ``t``/``it`` are kept, so a run
resumed from the output continues the parent trajectory's clock with
the perturbation applied.  The primary use is seeding perturbed
ensemble members from harvested turbulent snapshots
(``scripts/ensemble_setup.py``).

Runs single-device (``np0 = np1 = 1``) on the snapshot's own
parameters and stored precision, so every untouched mode round-trips
**bit-identically** (the injection changes exactly the target column
and, for ``i3 = 0``, its conjugate partner).  Supported systems: the
transient-growth set (plane-couette, plane-poiseuille, pipe,
taylor-couette).

Perturbation sources (exactly one):

- ``--tg-npz FILE --which input|response``: the mode's optimal
  perturbation from a transient-growth bundle (``<stem>_tg.npz``);
  the profile is Fornberg-regridded when the bundle's wall-normal
  grid differs from the snapshot's (with a note; ``--interp-order``).
- ``--modes-npz FILE --index J``: column ``J`` of the mode's
  controllability-mode array (``cont_modes_{i2}_{i3}``) from an
  ``operator_tools.save_modes_npz`` bundle (same regrid rule).
- ``--npy FILE``: a raw ``(C, Ny)`` complex ``.npy`` profile on the
  snapshot's grid, in the stored component basis (Cartesian
  ``(u_x, u_y, u_z)``; cyl/annular ``(u_z, u_+, u_-)``).

Amplitude (exactly one):

- ``--amplitude-energy E0``: scale the injected single-mode field
  (conjugate partner included) so its solver-measure perturbation
  energy is ``E0`` -- the same convention as the transient-growth
  ``--export-amplitude`` seed, evaluated with the solver's own
  ``get_norm2*`` (:func:`~dnsjax.analysis.transient_growth.
  mode_state_energy`), so the injected ``E'`` matches the solver's
  diagnostic exactly.
- ``--amplitude-scale S``: a raw multiplier.

Choosing ``E0``: small enough for a *linear* response (the check:
halving ``E0`` must leave the amplitude-normalised ensemble response
unchanged), large enough that the response stands above the residual
ensemble noise at the chosen member count.  Antithetic pairing
cancels the even-order nonlinear contributions, which widens the
usable window considerably.

``--negate`` flips the sign after scaling (antithetic ``+/-`` pairs
for ensemble variance cancellation).

The injected profile need not be discretely divergence-free: the
influence-matrix pressure solve projects any residual divergence out
on the first corrector step of the resumed run.  Transient-growth
optimals and controllability modes are divergence-free by
construction.  The mean mode ``(0, 0)`` is a valid target only with a
real profile (the mean of a real field), and is rejected under
``constant_bulk_velocity`` driving, whose bulk constraint makes the
mean mode affine.

Usage::

    uv run python scripts/snapshot_perturb.py \
        --snapshot state00042.tar --out member/seed.tar \
        --mode 3,0 --tg-npz U_mean_tg.npz --which input \
        --amplitude-energy 1e-6 [--negate] [--dist.platform cpu]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "configure_from_snapshot",
    "load_profile_tg",
    "load_profile_modes",
    "load_profile_npy",
    "perturb_state",
    "main",
]


# ── Setup ────────────────────────────────────────────────────────


def configure_from_snapshot(
    snapshot: str | Path, platform: str = "cpu"
) -> tuple[Any, Any, str]:
    """Configure the dnsjax singletons from the snapshot's parameters.

    Once per process, before any other dnsjax work: layers the
    snapshot's embedded parameters (single device), matches the
    snapshot's **stored precision** (so untouched modes round-trip
    bit-identically; the resume path is precision-agnostic, an
    offline injection must not be), then imports the flow / geometry
    modules.  Returns ``(fmod, gmod, family)``.
    """
    from dnsjax.bootstrap import configure_jax_platform
    from dnsjax.parameters import (
        padded_res,
        params,
        read_snapshot_params,
        update_parameters,
        validate_parameters,
    )
    from dnsjax.snapshot_meta import read_snapshot_meta

    snapshot = Path(snapshot)
    snap_params = read_snapshot_params(snapshot)
    if snap_params is None:
        raise SystemExit(f"{snapshot} is not a dnsjax snapshot")
    stored_dp = bool(
        read_snapshot_meta(snapshot)["params"]["res"]["double_precision"]
    )
    configure_jax_platform(platform, double_precision=stored_dp)
    params.res.double_precision = stored_dp
    update_parameters(snap_params)
    validate_parameters()
    padded_res.set_padded_resolution(params)
    if params.dist.np0 * params.dist.np1 != 1:
        raise SystemExit("snapshot_perturb is single-device (np0*np1 = 1)")

    from dnsjax.analysis.transient_growth import _SYSTEMS, _dispatch

    system = params.phys.system
    if system not in _SYSTEMS:
        raise SystemExit(
            f"system {system!r} is not supported (one of "
            f"{', '.join(_SYSTEMS)}; the force-driven/viscoelastic "
            "total-field systems have no single-mode perturbation "
            "convention here)"
        )
    return _dispatch(system)


# ── Profile sources ──────────────────────────────────────────────


def _regrid_columns(
    vec: np.ndarray, y_from: np.ndarray, y_to: np.ndarray, order: int
) -> np.ndarray:
    """Fornberg-regrid a ``(C, Ny_from)`` complex profile if needed."""
    from dnsjax.fd import local_interpolation_matrix

    if len(y_from) == len(y_to) and np.max(np.abs(y_from - y_to)) < 1e-12:
        return vec
    print(
        f"[perturb] regridding the profile from {len(y_from)} to "
        f"{len(y_to)} wall-normal points (order {order})."
    )
    mat = local_interpolation_matrix(
        np.asarray(y_from, dtype=float), np.asarray(y_to, dtype=float), order
    )
    return vec @ mat.T  # real matrix acts on each complex row


def _check_npz_system(npz: Any, path: Path) -> None:
    from dnsjax.parameters import params

    npz_system = str(np.asarray(npz["system"]))
    if npz_system != params.phys.system:
        raise SystemExit(
            f"{path} was computed for system {npz_system!r}; the "
            f"snapshot is {params.phys.system!r}"
        )


def load_profile_tg(
    npz_path: str | Path,
    i2: int,
    i3: int,
    which: str = "input",
    interp_order: int = 8,
) -> np.ndarray:
    """Mode ``(i2, i3)``'s optimal perturbation from a TG bundle.

    Reads ``opt_input`` / ``opt_response`` from a transient-growth
    ``<stem>_tg.npz``, matched on the ``mode_i2``/``mode_i3`` row;
    regridded onto the snapshot's grid when the bundle's ``code_grid``
    differs.
    """
    from dnsjax.parameters import derived_params

    npz_path = Path(npz_path)
    with np.load(npz_path) as npz:
        _check_npz_system(npz, npz_path)
        rows = np.nonzero((npz["mode_i2"] == i2) & (npz["mode_i3"] == i3))[0]
        if rows.size == 0:
            raise SystemExit(
                f"mode ({i2},{i3}) is not in {npz_path} (modes: "
                f"{np.stack([npz['mode_i2'], npz['mode_i3']], 1).tolist()})"
            )
        key = {"input": "opt_input", "response": "opt_response"}[which]
        vec = np.asarray(npz[key][rows[0]])
        y_from = np.asarray(npz["code_grid"], dtype=float)
    y_to = np.asarray(derived_params.wall_normal_grid, dtype=float)
    return _regrid_columns(vec, y_from, y_to, interp_order)


def load_profile_modes(
    npz_path: str | Path,
    i2: int,
    i3: int,
    index: int,
    interp_order: int = 8,
) -> np.ndarray:
    """Controllability-mode column ``index`` for mode ``(i2, i3)``.

    Reads ``cont_modes_{i2}_{i3}`` (shape ``(m, C, Ny)``) from an
    ``operator_tools.save_modes_npz`` bundle (same system / regrid
    rules as :func:`load_profile_tg`).
    """
    from dnsjax.parameters import derived_params

    npz_path = Path(npz_path)
    with np.load(npz_path) as npz:
        _check_npz_system(npz, npz_path)
        key = f"cont_modes_{i2}_{i3}"
        if key not in npz:
            raise SystemExit(
                f"{npz_path} has no {key!r} (available: "
                f"{[k for k in npz.files if k.startswith('cont_modes')]})"
            )
        arr = np.asarray(npz[key])
        if not (0 <= index < arr.shape[0]):
            raise SystemExit(
                f"--index {index} out of range ({key} holds "
                f"{arr.shape[0]} modes)"
            )
        vec = arr[index]
        y_from = np.asarray(npz["code_grid"], dtype=float)
    y_to = np.asarray(derived_params.wall_normal_grid, dtype=float)
    return _regrid_columns(vec, y_from, y_to, interp_order)


def load_profile_npy(path: str | Path) -> np.ndarray:
    """A raw ``(C, Ny)`` complex profile on the snapshot's grid."""
    from dnsjax.parameters import params

    vec = np.asarray(np.load(path))
    if vec.ndim != 2 or vec.shape[1] != params.res.ny:
        raise SystemExit(
            f"{path}: expected a (C, ny={params.res.ny}) profile, got "
            f"shape {vec.shape}"
        )
    return vec


# ── Injection ────────────────────────────────────────────────────


def perturb_state(
    state: Any, vec: np.ndarray, i2: int, i3: int, family: str, scale: float
) -> Any:
    """``state + scale * single_mode_state(vec, i2, i3)``."""
    from dnsjax.analysis.transient_growth import single_mode_state

    return state + scale * single_mode_state(vec, i2, i3, family)


# ── CLI ──────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python scripts/snapshot_perturb.py",
        description="Inject a single-mode perturbation into a dnsjax "
        "snapshot (see the module docstring).",
        allow_abbrev=False,
    )
    p.add_argument("--snapshot", required=True, help="input snapshot tar")
    p.add_argument("--out", required=True, help="output snapshot tar")
    p.add_argument(
        "--mode",
        required=True,
        help='"i2,i3" global spectral index of the injected mode',
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--tg-npz", default=None, help="transient-growth <stem>_tg.npz"
    )
    src.add_argument(
        "--modes-npz",
        default=None,
        help="controllability-modes npz (operator_tools.save_modes_npz)",
    )
    src.add_argument(
        "--npy", default=None, help="raw (C, Ny) complex .npy profile"
    )
    p.add_argument(
        "--which",
        default="input",
        choices=("input", "response"),
        help="which TG optimal to inject (--tg-npz only)",
    )
    p.add_argument(
        "--index",
        type=int,
        default=0,
        help="controllability-mode column (--modes-npz only)",
    )
    amp = p.add_mutually_exclusive_group(required=True)
    amp.add_argument(
        "--amplitude-energy",
        type=float,
        default=None,
        help="perturbation energy E' of the injected field",
    )
    amp.add_argument(
        "--amplitude-scale",
        type=float,
        default=None,
        help="raw multiplier on the profile",
    )
    p.add_argument(
        "--negate",
        action="store_true",
        help="flip the sign after scaling (antithetic pairs)",
    )
    p.add_argument(
        "--interp-order",
        type=int,
        default=8,
        help="Fornberg order for profile regridding (only used when "
        "the npz grid differs from the snapshot's; rarely worth "
        "changing)",
    )
    p.add_argument(
        "--dist.platform",
        dest="platform",
        default="cpu",
        choices=("cpu", "cuda", "rocm", "tpu"),
        help="JAX backend (single device)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fmod, gmod, family = configure_from_snapshot(args.snapshot, args.platform)

    import jax

    from dnsjax.analysis.transient_growth import (
        mode_state_energy,
        single_mode_state,
    )
    from dnsjax.harmonics import parse_mode_pairs
    from dnsjax.parameters import params
    from dnsjax.snapshot import load_snapshot, save_snapshot

    pairs = parse_mode_pairs(args.mode)
    if len(pairs) != 1:
        raise SystemExit("--mode takes exactly one 'i2,i3' pair")
    i2, i3 = pairs[0]
    n2, n3 = params.res.nz - 1, params.res.nx // 2
    if i2 >= n2 or i3 >= n3:
        raise SystemExit(
            f"mode ({i2},{i3}) out of range (0..{n2 - 1}, 0..{n3 - 1})"
        )

    if args.tg_npz is not None:
        vec = load_profile_tg(
            args.tg_npz, i2, i3, args.which, args.interp_order
        )
    elif args.modes_npz is not None:
        vec = load_profile_modes(
            args.modes_npz, i2, i3, args.index, args.interp_order
        )
    else:
        vec = load_profile_npy(args.npy)

    if (i2, i3) == (0, 0):
        # The mean of a real field is real; and under constant-bulk-
        # velocity driving the mean mode is constrained (affine), so
        # an injected mean would be partially projected away.
        if np.max(np.abs(vec.imag)) > 1e-13 * max(np.max(np.abs(vec)), 1.0):
            raise SystemExit(
                "a (0,0) mean-mode profile must be real "
                "(the mean of a real field)"
            )
        if params.phys.driving == "constant_bulk_velocity":
            raise SystemExit(
                "injecting the (0,0) mean mode under "
                "constant_bulk_velocity driving is rejected (the bulk "
                "constraint makes the mean mode affine)"
            )

    mode_state = single_mode_state(vec, i2, i3, family)
    if args.amplitude_energy is not None:
        energy = mode_state_energy(mode_state, family, gmod, fmod.flow)
        if energy <= 0.0:
            raise SystemExit("the injected profile has zero energy")
        scale = float(np.sqrt(args.amplitude_energy / energy))
    else:
        scale = float(args.amplitude_scale)
    if args.negate:
        scale = -scale

    state, t, it = load_snapshot(args.snapshot)
    state = state + scale * mode_state
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_snapshot(jax.block_until_ready(state), t, it, out, isnap=0)
    print(
        f"[perturb] wrote {out}: mode ({i2},{i3}) "
        f"scale {scale:+.6e} onto {args.snapshot} (t = {t:g}, "
        f"it = {it})."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
