r"""Shared pieces of the transient-growth tests, core and lab.

The CLI driver (:func:`run_tg`, :func:`load`, :func:`close`), the
laminar profile files (:func:`write_profile` and the per-flow writers),
the per-system in-process :func:`worker` (the frozen-profile hook
against the builtin coupling, and block-diagonality) and the per-system
:func:`cli_smoke`.  Each script lists its own systems and published
values.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from _live import run_live

REPO = Path(__file__).resolve().parent.parent
MODULE = "dnsjax.analysis.transient_growth"

#: Appended by ``--legacy-imm`` to every CLI run of :func:`run_tg`, the
#: published values among them: the primitive `$(v, p)$` scheme
#: (``res.consistent_imm = False``) in *every* wall-bounded geometry.
#: By default the published-value runs use the default reconstruction
#: scheme and check it against the published digits -- the strongest
#: available statement that the reconstruction leaves the linear physics
#: unchanged (the eigenvalue content of the operator these runs measure
#: is exactly what an Orr-Sommerfeld/Squire check would test).  Measured
#: agreement between the two propagators: 4-6 significant figures on
#: every case, so ``--legacy-imm`` must hit the same published digits
#: too.
EXTRA_ARGS: list[str] = []


# ── laminar profile files (top-wall-first / descending) ──────────


def ccf(re1: float, re2: float, eta: float) -> tuple[float, float]:
    """Circular-Couette ``A0``, ``B0`` (``U_theta = A0 r + B0/r``)."""
    re_ref = re1 if re1 > 0 else re2
    a0 = (re2 - eta * re1) / ((1 + eta) * re_ref)
    b0 = eta * (re1 - eta * re2) / ((1 + eta) * (1 - eta) ** 2 * re_ref)
    return a0, b0


def write_profile(path: Path, x: np.ndarray, u: np.ndarray) -> None:
    """A two-column profile file, top wall first."""
    np.savetxt(path, np.column_stack([x, u]))


def write_ccf_laminar(path: Path, re1: float, re2: float, eta: float) -> None:
    """The circular-Couette laminar profile, outer wall first."""
    a0, b0 = ccf(re1, re2, eta)
    r = np.linspace(1.0 / (1 - eta), eta / (1 - eta), 401)  # r2 -> r1
    write_profile(path, r, a0 * r + b0 / r)


def ccf_profile(flow, derived) -> np.ndarray:
    """The circular-Couette laminar profile on a flow's code grid."""
    rs = np.asarray(flow.rs)
    return derived.ccf_A * rs + derived.ccf_B / rs


# ── CLI driver ───────────────────────────────────────────────────


def run_tg(profile: Path, out_dir: Path, args: list[str]) -> str:
    """Invoke the CLI; raise on failure.

    Runs in the profile's directory (always a temp dir here), not the
    repo root: the CLI loads a ``./parameters.toml`` when present, and
    a stray one must not leak into these runs.
    """
    cmd = [
        sys.executable,
        "-m",
        MODULE,
        "--tg.profile",
        str(profile),
        "--tg.out_dir",
        str(out_dir),
        *args,
        *EXTRA_ARGS,
    ]
    res = run_live(cmd, cwd=profile.parent)
    if res.returncode != 0 or "FAILED" in res.stdout:
        raise SystemExit(f"tg run failed: {' '.join(args)}")
    return res.stdout


def load(out_dir: Path, stem: str) -> dict:
    return dict(np.load(out_dir / f"{stem}_tg.npz", allow_pickle=True))


def close(got: float, want: float, rtol: float, name: str) -> None:
    if abs(got - want) > rtol * abs(want):
        raise SystemExit(
            f"{name}: got {got:.6g}, want {want:.6g} (rtol {rtol})"
        )
    print(f"    {name}: {got:.6g} (want ~{want:.6g})  OK")


# ── in-process worker: hooks + block-diagonality ─────────────────


def worker(
    system: str, phys: dict, geo: dict, profile, curl_tol: float
) -> None:
    """Per-system: hook == builtin coupling, and block-diagonality.

    *system* at its *phys* / *geo* parameters; *profile* ``(flow,
    derived_params) -> array`` is its laminar total profile on the code
    grid, and *curl_tol* the bound on the hook's curl against the
    builtin (FD-exact polynomial profiles meet ``1e-12``).
    """
    from dnsjax.bootstrap import configure_jax_platform, platform_from_argv
    from dnsjax.parameters import (
        Parameters,
        derived_params,
        padded_res,
        params,
        update_parameters,
        validate_parameters,
    )

    update_parameters(
        Parameters(
            phys={"system": system, **phys},
            geo=geo,
            res={
                "nx": 8,
                "ny": 17,
                "nz": 8,
                "fd_order": 4,
                "double_precision": True,
            },
            step={
                "dt": 0.01,
                "implicitness": 1.0,
                "implicit_mean_coupling": False,
                "corrector_tolerance": 1e-12,
                "max_corrector_iterations": 60,
            },
        )
    )
    validate_parameters()
    padded_res.set_padded_resolution(params)
    configure_jax_platform(platform_from_argv())

    import importlib

    import jax.numpy as jnp

    from dnsjax.flows.registry import spec_for
    from dnsjax.sharding import sharding
    from dnsjax.timestep import make_stepper

    spec = spec_for(system)
    fmod = importlib.import_module(spec.flow_module)
    gmod = importlib.import_module(
        f"dnsjax.geometries.wall_bounded.{spec.geometry}"
    )
    flow = fmod.flow

    # Laminar total profile on the code grid (relevant component).
    prof = profile(flow, derived_params)

    frozen = fmod.frozen_profile_flow(
        jnp.asarray(prof, dtype=sharding.float_type)
    )
    db = float(np.max(np.abs(np.asarray(frozen.base_flow - flow.base_flow))))
    dc = float(
        np.max(np.abs(np.asarray(frozen.curl_base_flow - flow.curl_base_flow)))
    )
    assert db < 1e-12, f"{system}: base_flow hook != builtin ({db:.2e})"
    assert dc < curl_tol, f"{system}: curl hook != builtin ({dc:.2e})"

    # Block-diagonality: a basis vector at all modes vs one mode gives
    # the identical column at that mode.  Everything here stays in the
    # geometry's solver basis (the property is about modes, not
    # components, and both runs use the same stepper); the physical
    # wrapper lives in ``transient_growth._linear_step``.
    raw = make_stepper(
        gmod._l_bf, gmod._predict, gmod._correct, gmod._norm, None, None
    )
    pfc = raw[0]
    ny = params.res.ny
    n2, n3 = sharding.spec_shape[1], sharding.spec_shape[2]
    i2, i3 = 1, 0

    # The raw stepper's own state shape: the velocity, plus the slots a
    # pipe's pass carries across steps (``flow.n_carried``).
    n_slots = 3 + getattr(flow, "n_carried", 0)

    def _state(all_modes: bool):
        st = jnp.zeros(
            (n_slots, ny, n2, n3),
            dtype=sharding.complex_type,
            out_sharding=sharding.spec_vector_shard,
        )
        if all_modes:
            st = st.at[0, ny // 2].set(1.0).at[:, :, 0, 0].set(0.0)
        else:
            st = st.at[0, ny // 2, i2, i3].set(1.0)
        return st

    col_all = np.asarray(pfc(_state(True), gmod.fourier, flow)[0])[
        :, :, i2, i3
    ]
    col_one = np.asarray(pfc(_state(False), gmod.fourier, flow)[0])[
        :, :, i2, i3
    ]
    bd = float(np.max(np.abs(col_all - col_one)))
    assert bd < 1e-9, f"{system}: not block-diagonal ({bd:.2e})"
    print(
        f"  {system}: hook==builtin (base {db:.1e}, curl {dc:.1e}); "
        f"block-diagonal ({bd:.1e})  OK"
    )


# ── CLI smoke ────────────────────────────────────────────────────


def cli_smoke(system: str, write_laminar, phys_args: list[str]) -> None:
    """A per-system tiny run: G(0)=1, stable, forced overrides applied.

    *write_laminar(path)* writes the system's laminar profile file and
    *phys_args* are its parameter flags.
    """
    from dnsjax.flows.registry import spec_for

    cyl_annular = spec_for(system).geometry in ("cylindrical", "annular")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        prof = p / "lam.txt"
        res_flags = (
            ["--res.nr", "20", "--res.nz", "8", "--res.ntheta", "8"]
            if cyl_annular
            else ["--res.ny", "20", "--res.nx", "8", "--res.nz", "8"]
        )
        extra = [
            "--phys.system",
            system,
            *res_flags,
            "--tg.t_max",
            "40",
            "--tg.nt",
            "12",
            "--tg.modes",
            "1,1",
        ]
        write_laminar(prof)
        extra += phys_args
        run_tg(prof, p, extra)
        z = load(p, "lam")
        assert abs(z["G"][0, 0] - 1.0) < 1e-9, "G(0) != 1"
        assert np.all(np.isfinite(z["G"])), "non-finite G"
        assert float(z["extraction_residual"][0]) < 1e-9, "eig residual"
        assert float(z["G_max"][0]) >= 1.0 - 1e-9, "G_max < 1"
        pj = json.loads(str(z["params_json"]))
        assert pj["phys"]["u_grid"] == 0.0, "u_grid not zeroed"
        assert pj["step"]["implicitness"] == 1.0, "theta != 1"
    print(f"  {system}: CLI smoke (G(0)=1, u_grid=0, theta=1)  OK")
