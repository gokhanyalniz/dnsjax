r"""Set up ensemble response experiments (harvest + member run trees).

JAX-free orchestration (stdlib + the JAX-free ``dnsjax`` leaves
``snapshot_meta``/``harmonics``) for ensemble-averaged response
experiments: run a perturbed copy of many statistically independent
snapshots, record the perturbed mode with the probe stream, and
average -- the ensemble mean isolates the coherent (impulse) response
while the incoherent turbulence cancels.  Residual noise in the mean
decays as `$1/\sqrt{N}$` in the member count, so ``--n`` trades
compute for a cleaner response (antithetic pairing lowers the
prefactor, not the rate).  Aggregation / identification of the
resulting probe streams: :mod:`dnsjax.analysis.response.ensemble`.

Two subcommands:

``harvest``
    Select statistically independent snapshots from a completed run
    directory: all ``*.tar`` dnsjax snapshots with ``t >= --t-min``
    (cut the transient), thinned to a minimum spacing ``--spacing``
    (in simulation time units; pick several eddy-turnover times), at
    most ``--n`` of them.  Writes a JSON manifest consumed by
    ``build``.

``build``
    Materialise the member run tree from a manifest: per parent
    snapshot, one directory per ensemble member containing a
    perturbed seed snapshot (via ``scripts/snapshot_perturb.py``, one
    subprocess each -- sources and the ``--amplitude-energy``
    convention are its own) and a generated ``parameters.toml`` (the
    seed inherits the physics; the TOML layer sets only the stop
    horizon and the probe stream).  Pairing (``--pairing``):

    - ``antithetic`` (default): members ``mNNNN_p`` / ``mNNNN_m``
      seeded with ``+eps`` / ``-eps`` from the *same* parent.  The
      combination ``(u_+ - u_-)/2`` cancels the common turbulent
      evolution **and** all even-order nonlinear contributions at the
      same cost as a baseline pair (dnsjax runs are deterministic for
      a fixed configuration and device layout, so the shared
      background cancels to the linear-response accuracy).
    - ``baseline``: ``mNNNN_p`` (perturbed) / ``mNNNN_b`` (the
      unperturbed parent, resumed as-is); combination ``u_p - u_b``.
    - ``none``: perturbed members only; the background turbulence
      only cancels in the plain ensemble mean (slowest convergence).

    Emits ``run_commands.txt`` (one scheduler-agnostic launch line
    per member; submit them however the site likes) and
    ``members.json`` (the aggregation index).  ``--dry-run`` prints
    the planned tree and every command without writing anything.
    This script never executes solver runs.

The injected mode is auto-appended to ``--probe-modes`` when missing
(the response could otherwise never be recorded), and injecting the
``(0,0)`` mean mode is rejected (under constant-bulk-velocity driving
it is constrained/affine, and its ensemble response is not what this
machinery measures).  Pick ``--horizon`` to cover the response
feature to be measured or fitted -- e.g. past the transient-growth
peak `$t_\mathrm{opt}$` of the injected mode (the TG summary) -- and
no longer: member cost is linear in it, and identification horizons
must lie inside it.  It should also be a whole number of probe
intervals (``it_probes * dt``) so every member records the
cadence-aligned final sample; ``build`` checks this against the
parent snapshot's ``step.dt`` and warns otherwise.

Usage::

    uv run python scripts/ensemble_setup.py harvest \
        --run-dir prod/ --t-min 200 --spacing 5 --n 300 \
        --out manifest.json

    uv run python scripts/ensemble_setup.py build \
        --manifest manifest.json --tree members/ --mode 3,0 \
        --tg-npz U_mean_tg.npz --which input --amplitude-energy 1e-6 \
        --horizon 30 --probe-modes "3,0" --it-probes 10 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dnsjax.harmonics import parse_mode_pairs
from dnsjax.snapshot_meta import (
    git_hash,
    is_snapshot_file,
    read_snapshot_meta,
)

_REPO = Path(__file__).resolve().parent.parent

_PERTURB = _REPO / "scripts" / "snapshot_perturb.py"


# ── harvest ──────────────────────────────────────────────────────


def harvest(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    candidates = []
    for path in sorted(run_dir.glob("*.tar")):
        if not is_snapshot_file(path):
            continue
        meta = read_snapshot_meta(path)
        candidates.append(
            {
                "path": str(path.resolve()),
                "t": float(meta["t"]),
                "it": int(meta["it"]),
                "isnap": int(meta.get("isnap", -1)),
            }
        )
    candidates.sort(key=lambda c: c["t"])

    picked: list[dict] = []
    for cand in candidates:
        if cand["t"] < args.t_min:
            continue
        if picked and cand["t"] - picked[-1]["t"] < args.spacing:
            continue
        picked.append(cand)
        if len(picked) == args.n:
            break
    if not picked:
        raise SystemExit(
            f"no snapshots with t >= {args.t_min} in {run_dir} "
            f"({len(candidates)} snapshots scanned)"
        )
    if len(picked) < args.n:
        print(
            f"[harvest] only {len(picked)} of the requested "
            f"{args.n} snapshots satisfy the spacing; extend the run "
            "or reduce --spacing."
        )

    manifest = {
        "run_dir": str(run_dir.resolve()),
        "t_min": args.t_min,
        "spacing": args.spacing,
        "created": datetime.now(UTC).isoformat(),
        "git_hash": git_hash(),
        "snapshots": picked,
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"[harvest] {len(picked)} snapshots "
        f"(t = {picked[0]['t']:g} .. {picked[-1]['t']:g}) -> {args.out}"
    )
    return 0


# ── build ────────────────────────────────────────────────────────


def _source_args(args: argparse.Namespace) -> tuple[list[str], dict]:
    """The snapshot_perturb source arguments + the members.json record."""
    if args.tg_npz is not None:
        return (
            [
                "--perturb.tg_npz",
                str(Path(args.tg_npz).resolve()),
                "--perturb.which",
                args.which,
            ],
            {
                "kind": "tg-npz",
                "path": str(Path(args.tg_npz).resolve()),
                "which": args.which,
            },
        )
    if args.modes_npz is not None:
        return (
            [
                "--perturb.modes_npz",
                str(Path(args.modes_npz).resolve()),
                "--perturb.index",
                str(args.index),
            ],
            {
                "kind": "modes-npz",
                "path": str(Path(args.modes_npz).resolve()),
                "index": args.index,
            },
        )
    return (
        ["--perturb.npy", str(Path(args.npy).resolve())],
        {"kind": "npy", "path": str(Path(args.npy).resolve())},
    )


def _member_plan(
    args: argparse.Namespace, snapshots: list[dict]
) -> list[dict]:
    """One record per member: directory, sign, parent."""
    signs = {
        "antithetic": (("p", +1), ("m", -1)),
        "baseline": (("p", +1), ("b", 0)),
        "none": (("p", +1),),
    }[args.pairing]
    members = []
    for k, snap in enumerate(snapshots):
        for tag, sign in signs:
            members.append(
                {
                    "dir": f"m{k:04d}_{tag}",
                    "sign": sign,
                    "parent": snap["path"],
                    "parent_t": snap["t"],
                }
            )
    return members


def _member_toml(
    seed_path: str, t_end: float, probe_modes: str, it_probes: int
) -> str:
    return (
        "# generated by scripts/ensemble_setup.py build\n"
        "[init]\n"
        f'snapshot = "{seed_path}"\n'
        "\n"
        "[stop]\n"
        f"max_sim_time = {t_end!r}\n"
        "check_laminarization = false\n"
        "\n"
        "[outs]\n"
        "snapshot_save_initial = false\n"
        "snapshot_save_final = false\n"
        "\n"
        "[probes]\n"
        f'modes = "{probe_modes}"\n'
        f"it_probes = {it_probes}\n"
    )


def build(args: argparse.Namespace) -> int:
    with open(args.manifest) as f:
        manifest = json.load(f)
    snapshots = manifest["snapshots"]

    pairs = parse_mode_pairs(args.mode)
    if len(pairs) != 1:
        raise SystemExit("--mode takes exactly one 'i2,i3' pair")
    i2, i3 = pairs[0]
    if (i2, i3) == (0, 0):
        raise SystemExit(
            "injecting the (0,0) mean mode is not supported (see the "
            "module docstring)"
        )

    probe_pairs = parse_mode_pairs(args.probe_modes)
    if (i2, i3) not in probe_pairs:
        print(
            f"[build] note: adding the injected mode ({i2},{i3}) to "
            "the probe list (its response must be recorded)."
        )
        probe_pairs.append((i2, i3))
    probe_modes = ";".join(f"{a},{b}" for a, b in probe_pairs)

    # Final-sample alignment check against the parents' step.dt.
    meta0 = read_snapshot_meta(Path(snapshots[0]["path"]))
    dt = float(meta0["params"]["step"]["dt"])
    interval = dt * args.it_probes
    n_intervals = args.horizon / interval
    if abs(n_intervals - round(n_intervals)) > 1e-9:
        print(
            f"[build] note: --horizon {args.horizon:g} is not a whole "
            f"number of probe intervals (it_probes * dt = "
            f"{interval:g}); the members' final samples will not be "
            "cadence-aligned."
        )

    dnsjax_bin = args.dnsjax_bin
    if dnsjax_bin is None:
        sibling = Path(sys.executable).with_name("dnsjax")
        dnsjax_bin = str(sibling) if sibling.exists() else "dnsjax"

    tree = Path(args.tree)
    src_args, src_record = _source_args(args)
    members = _member_plan(args, snapshots)

    seed_cmds: list[tuple[dict, list[str]]] = []
    run_lines: list[str] = []
    for mem in members:
        mdir = tree / mem["dir"]
        if mem["sign"] == 0:
            seed = mem["parent"]  # baseline: resume the parent as-is
        else:
            seed = str((mdir / "seed.tar").resolve())
            cmd = [
                sys.executable,
                str(_PERTURB),
                "--init.snapshot",
                mem["parent"],
                "--perturb.out",
                seed,
                "--perturb.mode",
                f"{i2},{i3}",
                *src_args,
                "--perturb.amplitude_energy",
                str(args.amplitude_energy),
            ]
            if mem["sign"] < 0:
                cmd += ["--perturb.negate", "True"]
            seed_cmds.append((mem, cmd))
        mem["seed"] = seed
        mem["t_end"] = mem["parent_t"] + args.horizon
        # ``--dist.np1`` as well as ``mpirun -np``: the mesh size is a
        # *parameter*, and a run whose visible device count does not
        # equal ``np0 * np1`` exits 1 at startup (``sharding.py``).
        # Emitting only ``-np N`` therefore produced a launch line that
        # could never run for any N > 1.  np1 is the spanwise / k_x
        # axis, the 1-D split the solver's own launch recipes use.
        dist = f" --dist.np1 {args.np}" if args.np > 1 else ""
        run_lines.append(
            f"cd {mdir.resolve()} && mpirun -np {args.np} {dnsjax_bin}{dist}"
        )

    if args.dry_run:
        print(f"[build] DRY RUN: {len(members)} members under {tree}/")
        for mem in members:
            print(f"  {mem['dir']}: seed {mem['seed']}")
        for _, cmd in seed_cmds:
            print("  seed-cmd:", " ".join(cmd))
        for line in run_lines:
            print("  run-cmd:", line)
        return 0

    tree.mkdir(parents=True, exist_ok=True)
    for mem, cmd in seed_cmds:
        (tree / mem["dir"]).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(
                f"seed generation failed for {mem['dir']}:\n"
                + result.stdout[-2000:]
                + result.stderr[-2000:]
            )
    for mem in members:
        mdir = tree / mem["dir"]
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "parameters.toml").write_text(
            _member_toml(
                mem["seed"], mem["t_end"], probe_modes, args.it_probes
            )
        )

    (tree / "run_commands.txt").write_text("\n".join(run_lines) + "\n")
    with open(tree / "members.json", "w") as f:
        json.dump(
            {
                "manifest": str(Path(args.manifest).resolve()),
                "mode": [i2, i3],
                "amplitude_energy": args.amplitude_energy,
                "pairing": args.pairing,
                "horizon": args.horizon,
                "probe_modes": probe_modes,
                "it_probes": args.it_probes,
                "source": src_record,
                "created": datetime.now(UTC).isoformat(),
                "git_hash": git_hash(),
                "members": members,
            },
            f,
            indent=2,
        )
    print(
        f"[build] {len(members)} members under {tree}/ "
        f"({len(seed_cmds)} seeds written); launch with the lines in "
        f"{tree / 'run_commands.txt'}, then aggregate with "
        "python -m dnsjax.analysis.response.ensemble."
    )
    return 0


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python scripts/ensemble_setup.py",
        description="Harvest snapshots / build ensemble member trees "
        "(see the module docstring).",
        allow_abbrev=False,
    )
    sub = p.add_subparsers(dest="command", required=True)

    ph = sub.add_parser("harvest", help="select independent snapshots")
    ph.add_argument("--run-dir", required=True)
    ph.add_argument(
        "--t-min",
        type=float,
        default=0.0,
        help="discard the initial transient before this time",
    )
    ph.add_argument(
        "--spacing",
        type=float,
        required=True,
        help="minimum time separation between selected snapshots",
    )
    ph.add_argument(
        "--n",
        type=int,
        required=True,
        help="ensemble size target (mean noise decays as 1/sqrt(n))",
    )
    ph.add_argument("--out", required=True, help="manifest JSON path")
    ph.set_defaults(func=harvest)

    pb = sub.add_parser("build", help="materialise the member tree")
    pb.add_argument("--manifest", required=True)
    pb.add_argument("--tree", required=True, help="member tree root")
    pb.add_argument("--mode", required=True, help='"i2,i3" injected mode')
    src = pb.add_mutually_exclusive_group(required=True)
    src.add_argument("--tg-npz", default=None)
    src.add_argument("--modes-npz", default=None)
    src.add_argument("--npy", default=None)
    pb.add_argument("--which", default="input", choices=("input", "response"))
    pb.add_argument("--index", type=int, default=0)
    pb.add_argument(
        "--amplitude-energy",
        type=float,
        required=True,
        help="injected E' per member (linearity guidance: the "
        "snapshot_perturb.py docstring)",
    )
    pb.add_argument(
        "--pairing",
        default="antithetic",
        choices=("antithetic", "baseline", "none"),
    )
    pb.add_argument(
        "--horizon",
        type=float,
        required=True,
        help="member run length past the parent snapshot time "
        "(cover the response of interest; see the docstring)",
    )
    pb.add_argument("--probe-modes", required=True)
    pb.add_argument("--it-probes", type=int, required=True)
    pb.add_argument(
        "--np",
        type=int,
        default=1,
        help="devices per member run (emitted as mpirun -np N plus "
        "--dist.np1 N, the spanwise split, for N > 1)",
    )
    pb.add_argument(
        "--dnsjax-bin",
        default=None,
        help="solver binary in run_commands.txt (default: the "
        "installed console script next to this interpreter)",
    )
    pb.add_argument("--dry-run", action="store_true")
    pb.set_defaults(func=build)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
