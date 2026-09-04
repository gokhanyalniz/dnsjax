r"""Inject a single-mode perturbation into an existing dnsjax snapshot.

Offline CLI + library (the
:mod:`dnsjax.analysis.snapshot_import` counterpart for *existing*
dnsjax snapshots): loads a snapshot, adds a scaled
``(C, Ny)`` complex wall-normal profile at one global spectral mode
``(i2, i3)`` -- with the real-FFT conjugate partner handled by
:func:`dnsjax.analysis.transient_growth.single_mode_state` -- and
writes a new snapshot.  The parent's ``t``/``it`` are kept, so a run
resumed from the output continues the parent trajectory's clock with
the perturbation applied -- provided the resuming run's own
trajectory-defining parameters match the ones the parent recorded.
``res.consistent_imm`` is one of them, so a parent written before it
defaulted on resumes as a **new** trajectory unless that run passes
``--res.consistent_imm False`` (or ``--init.force_resume``).  The
primary use is seeding perturbed ensemble members from harvested
turbulent snapshots (``scripts/ensemble_setup.py``).

Runs single-device (``np0 = np1 = 1``) on the snapshot's own
parameters and stored precision, so every untouched mode round-trips
**bit-identically** (the injection changes exactly the target column
and, for ``i3 = 0``, its conjugate partner).  Supported systems: the
transient-growth set (plane-couette, plane-poiseuille, pipe,
taylor-couette, quasi-keplerian).

The CLI rides the shared per-flow surface
(:func:`dnsjax.bootstrap.resolve_parameters`) with the script's own
knobs as the ``[perturb]`` extension section (``--perturb.<field>``;
:class:`PerturbParams`): the snapshot is ``--init.snapshot`` and the
backend ``--dist.platform``, exactly as on a solver run.  The
configuration is snapshot + CLI only -- no ``parameters.toml`` is
read (``toml_path=False``), so a production TOML in the working
directory cannot re-layer the snapshot's parameters and break the
bit-identical round-trip.

Perturbation sources (exactly one):

- ``--perturb.tg_npz FILE --perturb.which input|response``: the
  mode's optimal perturbation from a transient-growth bundle
  (``<stem>_tg.npz``); the profile is Fornberg-regridded when the
  bundle's wall-normal grid differs from the snapshot's (with a
  note; ``--perturb.interp_order``).
- ``--perturb.modes_npz FILE --perturb.index J``: column ``J`` of
  the mode's profile array (``profiles_{i2}_{i3}``) from an
  ``operator_tools.save_modes_npz`` bundle (e.g. controllability
  modes; same regrid rule).
- ``--perturb.npy FILE``: a raw ``(C, Ny)`` complex ``.npy`` profile
  on the snapshot's grid, in the stored component basis (Cartesian
  ``(u_x, u_y, u_z)``; cyl/annular ``(u_z, u_r, u_theta)``).

Amplitude (exactly one):

- ``--perturb.amplitude_energy E0``: scale the injected single-mode
  field (conjugate partner included) so its solver-measure
  perturbation energy is ``E0`` -- the same convention as the
  transient-growth ``--tg.export_amplitude`` seed, evaluated with
  the solver's own ``get_norm2*``
  (:func:`~dnsjax.analysis.transient_growth.mode_state_energy`), so
  the injected ``E'`` matches the solver's diagnostic exactly.
- ``--perturb.amplitude_scale S``: a raw multiplier.

Choosing ``E0``: small enough for a *linear* response (the check:
halving ``E0`` must leave the amplitude-normalised ensemble response
unchanged), large enough that the response stands above the residual
ensemble noise at the chosen member count.  Antithetic pairing
cancels the even-order nonlinear contributions, which widens the
usable window considerably.

``--perturb.negate True`` flips the sign after scaling (antithetic
``+/-`` pairs for ensemble variance cancellation).

The injected profile need not be discretely divergence-free: the
influence-matrix pressure solve projects any residual divergence out
on the first corrector step of the resumed run.  Transient-growth
optimals and controllability modes are divergence-free by
construction.  The mean mode ``(0, 0)`` is a valid target on the
Cartesian flows only, and the profile is **checked, not reshaped** --
this script's caller owns what it injects, so an incompatible profile
is refused with its measured residuals rather than silently projected.
It must be real (the mean of a real field), have no wall-normal
component (continuity with no-slip forces ``<v> = 0``), satisfy
no-slip, and satisfy the mean-mode conservation laws of each tilted
direction (:mod:`dnsjax.ic.mean_mode`, which also documents the
tolerance) -- including an unchanged bulk velocity in a direction
whose mean the driving holds.

Usage::

    uv run python scripts/snapshot_perturb.py \
        --init.snapshot state00042.tar --perturb.out member/seed.tar \
        --perturb.mode 3,0 --perturb.tg_npz U_mean_tg.npz \
        --perturb.which input --perturb.amplitude_energy 1e-6 \
        [--perturb.negate True] [--dist.platform cpu]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dnsjax.analysis.transient_growth import WALL_BOUNDED_TG_SYSTEMS
from dnsjax.extensions import ParamExtension, register_extension
from dnsjax.flows.registry import cartesian_systems
from dnsjax.harmonics import parse_mode_pairs
from dnsjax.ic.mean_mode import check_cartesian_mean_profile

__all__ = [
    "PerturbParams",
    "PERTURB_EXTENSION",
    "configure_from_snapshot",
    "load_profile_tg",
    "load_profile_modes",
    "load_profile_npy",
    "perturb_state",
    "main",
]

_PROG = "python scripts/snapshot_perturb.py"


# ── The [perturb] extension section ──────────────────────────────


class PerturbParams(BaseModel):
    r"""Injection knobs: the ``[perturb]`` extension section.

    Parsed as ``--perturb.<field>`` CLI flags on the shared per-flow
    surface (:func:`dnsjax.bootstrap.resolve_parameters`; snapshot +
    CLI only, no TOML).  Source and amplitude selections are
    exactly-one-of groups (:func:`_validate_perturb`); knob guidance:
    the module docstring.
    """

    model_config = ConfigDict(extra="forbid")

    out: str | None = Field(
        default=None, description="Output snapshot tar.  Required."
    )
    mode: str | None = Field(
        default=None,
        description=(
            "'i2,i3' global spectral index of the injected mode.  Required."
        ),
    )
    tg_npz: str | None = Field(
        default=None,
        description=("Source: transient-growth <stem>_tg.npz (see 'which')."),
    )
    modes_npz: str | None = Field(
        default=None,
        description=(
            "Source: controllability-modes npz "
            "(operator_tools.save_modes_npz; see 'index')."
        ),
    )
    npy: str | None = Field(
        default=None,
        description="Source: raw (C, Ny) complex .npy profile.",
    )
    which: Literal["input", "response"] = Field(
        default="input",
        description="Which TG optimal to inject (tg_npz only).",
    )
    index: int = Field(
        default=0,
        ge=0,
        description="Controllability-mode column (modes_npz only).",
    )
    amplitude_energy: float | None = Field(
        default=None,
        gt=0,
        description="Perturbation energy E' of the injected field.",
    )
    amplitude_scale: float | None = Field(
        default=None,
        description="Raw multiplier on the profile.",
    )
    negate: bool = Field(
        default=False,
        description="Flip the sign after scaling (antithetic pairs).",
    )
    interp_order: int = Field(
        default=8,
        ge=1,
        description=(
            "Fornberg order for profile regridding (only used when "
            "the npz grid differs from the snapshot's)."
        ),
    )


def _validate_perturb(values: PerturbParams, params) -> None:
    # Structural checks on the configured section: exactly one source,
    # exactly one amplitude, a supported system, and a single in-range
    # mode.  File-content checks stay with the profile loaders.
    probed = (
        values.out,
        values.mode,
        values.tg_npz,
        values.modes_npz,
        values.npy,
        values.amplitude_energy,
        values.amplitude_scale,
    )
    if all(v is None for v in probed):
        return
    sources = [
        k
        for k, v in (
            ("tg_npz", values.tg_npz),
            ("modes_npz", values.modes_npz),
            ("npy", values.npy),
        )
        if v is not None
    ]
    if len(sources) != 1:
        raise ValueError(
            "perturb: exactly one profile source (perturb.tg_npz / "
            f"modes_npz / npy) must be set; got {sources or 'none'}."
        )
    amplitudes = [
        k
        for k, v in (
            ("amplitude_energy", values.amplitude_energy),
            ("amplitude_scale", values.amplitude_scale),
        )
        if v is not None
    ]
    if len(amplitudes) != 1:
        raise ValueError(
            "perturb: exactly one of perturb.amplitude_energy / "
            f"amplitude_scale must be set; got {amplitudes or 'none'}."
        )
    if params.phys.system not in WALL_BOUNDED_TG_SYSTEMS:
        raise ValueError(
            f"perturb: system {params.phys.system!r} is not supported "
            f"(one of {', '.join(WALL_BOUNDED_TG_SYSTEMS)}; the "
            "force-driven/viscoelastic total-field systems have no "
            "single-mode perturbation convention here)."
        )
    if values.mode is None:
        raise ValueError("perturb.mode ('i2,i3') is required.")
    pairs = parse_mode_pairs(values.mode)
    if len(pairs) != 1:
        raise ValueError("perturb.mode takes exactly one 'i2,i3' pair.")
    i2, i3 = pairs[0]
    n2, n3 = params.res.nz - 1, params.res.nx // 2
    if i2 >= n2 or i3 >= n3:
        raise ValueError(
            f"perturb.mode ({i2},{i3}) out of range "
            f"(0..{n2 - 1}, 0..{n3 - 1})."
        )
    if (i2, i3) == (0, 0) and params.phys.system not in cartesian_systems:
        raise ValueError(
            "perturb: injecting the (0,0) mean mode is implemented "
            "only for the Cartesian flows, whose mean-mode "
            "conservation laws the profile is checked against; "
            f"system {params.phys.system!r} defers it (see "
            "dnsjax.ic.mean_mode)."
        )


PERTURB_EXTENSION = register_extension(
    ParamExtension(
        name="perturb",
        model=PerturbParams,
        relevant=lambda system: system in WALL_BOUNDED_TG_SYSTEMS,
        summary="Single-mode snapshot injection (this script's knobs).",
        validate=_validate_perturb,
        # Injection-run config, not trajectory state: the written
        # snapshot must not carry it.
        record_in_metadata=False,
    )
)

#: Live ``[perturb]`` values (resolved by ``resolve_parameters`` in
#: :func:`main`).
perturb_params: PerturbParams = PERTURB_EXTENSION.values


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
    snap = read_snapshot_params(snapshot)
    if snap is None:
        raise SystemExit(f"{snapshot} is not a dnsjax snapshot")
    snap_params, _ = snap  # extension overlays are irrelevant offline
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

    return _dispatch_supported(params.phys.system)


def _dispatch_supported(system: str) -> tuple[Any, Any, str]:
    """``transient_growth._dispatch`` gated on the supported set."""
    from dnsjax.analysis.transient_growth import _dispatch

    if system not in WALL_BOUNDED_TG_SYSTEMS:
        raise SystemExit(
            f"system {system!r} is not supported (one of "
            f"{', '.join(WALL_BOUNDED_TG_SYSTEMS)}; the force-driven/"
            "viscoelastic total-field systems have no single-mode "
            "perturbation convention here)"
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
    """Profile column ``index`` for mode ``(i2, i3)``.

    Reads ``profiles_{i2}_{i3}`` (shape ``(m, C, Ny)``) from an
    ``operator_tools.save_modes_npz`` bundle (e.g. controllability
    modes; same system / regrid rules as :func:`load_profile_tg`).
    """
    from dnsjax.parameters import derived_params

    npz_path = Path(npz_path)
    with np.load(npz_path) as npz:
        _check_npz_system(npz, npz_path)
        key = f"profiles_{i2}_{i3}"
        if key not in npz:
            raise SystemExit(
                f"{npz_path} has no {key!r} (available: "
                f"{[k for k in npz.files if k.startswith('profiles')]})"
            )
        arr = np.asarray(npz[key])
        if not (0 <= index < arr.shape[0]):
            raise SystemExit(
                f"perturb.index {index} out of range ({key} holds "
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

    return state + scale * single_mode_state(vec, i2, i3)


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    from dnsjax.bootstrap import configure_jax_platform, resolve_parameters
    from dnsjax.parameters import params
    from dnsjax.snapshot_meta import read_snapshot_meta

    # Snapshot + CLI layers only (toml_path=False): the [perturb]
    # section and the shared surface (--init.snapshot,
    # --dist.platform) parse together; validate_parameters runs the
    # structural checks (_validate_perturb).
    resolve_parameters(
        argv,
        toml_path=False,
        extensions=(PERTURB_EXTENSION,),
        prog=_PROG,
    )
    p = perturb_params
    snapshot = params.init.snapshot
    if snapshot is None:
        raise SystemExit(f"{_PROG}: error: --init.snapshot is required")
    if p.mode is None or p.out is None:
        raise SystemExit(
            f"{_PROG}: error: --perturb.mode and --perturb.out are required"
        )
    if params.dist.np0 * params.dist.np1 != 1:
        raise SystemExit("snapshot_perturb is single-device (np0*np1 = 1)")

    # Match the snapshot's stored precision before JAX initializes
    # arrays (bit-identical round-trip of the untouched modes); the
    # written snapshot's metadata records it via params.
    stored_dp = bool(
        read_snapshot_meta(snapshot)["params"]["res"]["double_precision"]
    )
    params.res.double_precision = stored_dp
    configure_jax_platform(params.dist.platform, double_precision=stored_dp)
    fmod, gmod, family = _dispatch_supported(params.phys.system)

    import jax

    from dnsjax.analysis.transient_growth import (
        mode_state_energy,
        single_mode_state,
    )
    from dnsjax.snapshot import load_snapshot, save_snapshot

    ((i2, i3),) = parse_mode_pairs(p.mode)

    if p.tg_npz is not None:
        vec = load_profile_tg(p.tg_npz, i2, i3, p.which, p.interp_order)
    elif p.modes_npz is not None:
        vec = load_profile_modes(p.modes_npz, i2, i3, p.index, p.interp_order)
    else:
        vec = load_profile_npy(p.npy)

    # A mean-mode profile is not free: reality, no wall-normal
    # component, no-slip, and each tilted direction's conservation
    # laws (module docstring).  Refused, not reshaped -- the caller
    # owns the profile it injects.  (_validate_perturb has already
    # restricted (0,0) to the Cartesian flows.)
    if (i2, i3) == (0, 0):
        bad = check_cartesian_mean_profile(
            vec,
            np.asarray(fmod.flow.D1),
            np.asarray(fmod.flow.D2),
            np.asarray(fmod.flow.y_weights),
        )
        if bad:
            raise SystemExit(
                "the (0,0) mean-mode profile is not a legal "
                "perturbation:\n  - " + "\n  - ".join(bad)
            )

    mode_state = single_mode_state(vec, i2, i3)
    if p.amplitude_energy is not None:
        energy = mode_state_energy(mode_state, family, gmod, fmod.flow)
        if energy <= 0.0:
            raise SystemExit("the injected profile has zero energy")
        scale = float(np.sqrt(p.amplitude_energy / energy))
    else:
        scale = float(p.amplitude_scale)
    if p.negate:
        scale = -scale

    state, t, it = load_snapshot(snapshot)
    state = state + scale * mode_state
    out = Path(p.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_snapshot(jax.block_until_ready(state), t, it, out, isnap=0)
    print(
        f"[perturb] wrote {out}: mode ({i2},{i3}) "
        f"scale {scale:+.6e} onto {snapshot} (t = {t:g}, "
        f"it = {it})."
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
