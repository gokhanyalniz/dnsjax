#!/usr/bin/env python3
r"""Behavioural fingerprint of a dnsjax tree, to gate a refactor.

Usage::

    capture.py TREE OUT [--only REGEX] [--jobs N] [--no-hlo] [--merge]

Drives only the public command lines of *TREE* through its own
``.venv`` (``dnsjax``, ``dnsjax-twin``, the transient-growth CLI), so
one capture runs unchanged on every commit of a refactor and two
captures compare with :mod:`compare` (``verify.py`` does both for a
commit).  Writes into *OUT*:

- ``texts/``: ``--help`` / ``--sample-toml`` renders of every flow
  (``NO_COLOR``, fixed ``COLUMNS``);
- ``runs/<name>/``: short fixed-seed runs -- every flow x scheme x
  start mode, variants (driving, wedge, ``[lowres]``, legacy IMM, the
  GEMM path, adaptive ``dt``, the embedded carry), regridding and
  same-grid resumes, the probe and forcing streams, a twin member and
  the transient-growth CLI -- with their snapshots, streams and
  sidecars, an XLA HLO dump (``hlo/``) of the random-start runs, and
  the JAX-free reader's view of the final snapshot;
- ``manifest.json``: every command, its exit code and its duration,
  the HLO fingerprints and the analysis views.

The runs are CPU-only and small (about 12 minutes at ``--jobs 4``).
``--only REGEX`` keeps the texts and runs whose names it matches; a
resume, stream or twin run reads its parent run's final snapshot, so
select the parent too, or ``--merge`` into a capture that holds it.
Three exit nonzero by design: the curved pipe refuses ``cnab2`` (both
its runs), and its regridding resume stops on a corrector failure at
this resolution; a refactor keeps those exits as they are.  The run
list names the lab's flows and streams, so it needs a tree where the
lab is installed (``uv sync``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SYSTEMS: dict[str, list[str]] = {
    "kolmogorov": ["--phys.re", "620", "--geo.lx", "5", "--geo.lz", "5"],
    "plane-couette": ["--phys.re", "330", "--geo.lx", "5", "--geo.lz", "5"],
    "plane-poiseuille": [
        "--phys.re",
        "2000",
        "--geo.lx",
        "5",
        "--geo.lz",
        "5",
    ],
    "pipe": ["--phys.re", "1800", "--geo.lz", "5"],
    "curved-pipe": [
        "--phys.re",
        "1800",
        "--geo.curvature",
        "0.3",
        "--geo.lz",
        "5",
    ],
    "viscoelastic-pipe": [
        "--phys.wi",
        "20",
        "--phys.el",
        "0.02",
        "--init.random_conformation_amplitude",
        "10",
        "--geo.lz",
        "5",
    ],
    "taylor-couette": [
        "--phys.re1",
        "400",
        "--phys.re2",
        "-400",
        "--geo.eta",
        "0.5",
        "--geo.lz",
        "5",
    ],
    "quasi-keplerian": [
        "--phys.re1",
        "100",
        "--phys.r_omega",
        "-1.2",
        "--geo.eta",
        "0.71",
        "--geo.lz",
        "5",
    ],
    "dean": ["--phys.re", "1000", "--geo.eta", "0.5", "--geo.lz", "5"],
    "viscoelastic-dean": [
        "--phys.wi",
        "20",
        "--phys.el",
        "20",
        "--init.random_conformation_amplitude",
        "10",
        "--geo.lz",
        "5",
    ],
}
CYL_ANN = {
    "pipe",
    "curved-pipe",
    "viscoelastic-pipe",
    "taylor-couette",
    "quasi-keplerian",
    "dean",
    "viscoelastic-dean",
}


def res_flags(system: str, ny: int = 17) -> list[str]:
    if system == "kolmogorov":
        return ["--res.nx", "8", "--res.ny", "8", "--res.nz", "8"]
    if system in CYL_ANN:
        return ["--res.nz", "8", "--res.nr", str(ny), "--res.ntheta", "8"]
    return ["--res.nx", "8", "--res.ny", str(ny), "--res.nz", "8"]


COMMON = [
    "--step.dt",
    "0.005",
    "--stop.max_sim_time",
    "0.02",
    "--outs.it_stats",
    "1",
    "--outs.it_steps",
    "1",
    "--outs.snapshot_save_initial",
    "True",
    "--stop.check_laminarization",
    "False",
]
RANDOM = [
    "--init.random_field",
    "True",
    "--init.random_seed",
    "7",
    "--init.random_amplitude",
    "0.1",
]
ROLLS = ["--init.localized_rolls", "True"]


def solver_runs() -> list[dict]:
    """Wave 1: every system x scheme x IC mode, plus variants."""
    runs = []
    for system, sargs in SYSTEMS.items():
        for scheme in ("iterative-cn", "cnab2"):
            for ic, icargs in (("random", RANDOM), ("rolls", ROLLS)):
                corr = (
                    ["--outs.it_corrector", "1", "--outs.it_error_check", "1"]
                    if scheme == "iterative-cn" and system != "kolmogorov"
                    else []
                )
                runs.append(
                    {
                        "name": f"{system}.{ic}.{scheme}",
                        "cmd": [
                            "dnsjax",
                            "--phys.system",
                            system,
                            *sargs,
                            *res_flags(system),
                            "--step.scheme",
                            scheme,
                            *icargs,
                            *COMMON,
                            *corr,
                        ],
                        "hlo": ic == "random",
                    }
                )
    variants = {
        "plane-poiseuille-cbv": (
            "plane-poiseuille",
            ["--phys.driving", "constant_bulk_velocity"],
        ),
        "curved-pipe-cbv": (
            "curved-pipe",
            ["--phys.driving", "constant_bulk_velocity"],
        ),
        "pipe-wedge": ("pipe", ["--geo.m0", "2"]),
        "taylor-couette-wedge": ("taylor-couette", ["--geo.m0", "2"]),
        "plane-couette-lowres": (
            "plane-couette",
            [
                "--lowres.it_lowres",
                "2",
                "--lowres.nx",
                "4",
                "--lowres.ny",
                "9",
                "--lowres.nz",
                "4",
                "--lowres.pressure",
                "True",
            ],
        ),
        "pipe-legacy-imm": ("pipe", ["--res.consistent_imm", "False"]),
        "plane-couette-gemm": (
            "plane-couette",
            ["--solver.wall_normal_matvec", "dense"],
        ),
        "plane-couette-adaptive": (
            "plane-couette",
            [
                "--step.adaptive",
                "True",
                "--step.dt_max",
                "0.01",
                "--step.cfl_target",
                "0.5",
                "--step.cfl_cadence",
                "2",
            ],
        ),
        "pipe-embed-carry": (
            "pipe",
            ["--outs.snapshot_embed_carry", "True"],
        ),
    }
    for name, (system, extra) in variants.items():
        runs.append(
            {
                "name": f"{name}.random.iterative-cn",
                "cmd": [
                    "dnsjax",
                    "--phys.system",
                    system,
                    *SYSTEMS[system],
                    *res_flags(system),
                    *RANDOM,
                    *COMMON,
                    *extra,
                ],
                "hlo": True,
            }
        )
    return runs


def wave2_runs(out: Path) -> list[dict]:
    """Runs that read wave-1 snapshots: resumes, streams, twin."""
    runs = []

    def final(name: str) -> str:
        return str(out / "runs" / name / "state00001.tar")

    for system in (
        "pipe",
        "viscoelastic-pipe",
        "taylor-couette",
        "plane-couette",
        "curved-pipe",
        "dean",
    ):
        runs.append(
            {
                "name": f"resume-regrid.{system}",
                "cmd": [
                    "dnsjax",
                    "--phys.system",
                    system,
                    *SYSTEMS[system],
                    *res_flags(system, ny=21),
                    "--init.snapshot",
                    final(f"{system}.random.iterative-cn"),
                    *COMMON,
                ],
                "hlo": False,
            }
        )
    runs.append(
        {
            "name": "resume-same.plane-couette",
            "cmd": [
                "dnsjax",
                "--phys.system",
                "plane-couette",
                *SYSTEMS["plane-couette"],
                *res_flags("plane-couette"),
                "--init.snapshot",
                final("plane-couette.random.iterative-cn"),
                *COMMON,
            ],
            "hlo": False,
        }
    )
    runs.append(
        {
            "name": "streams.plane-couette",
            "cmd": [
                "dnsjax",
                "--phys.system",
                "plane-couette",
                *SYSTEMS["plane-couette"],
                *res_flags("plane-couette"),
                *RANDOM,
                *COMMON,
                "--probes.modes",
                "1,1;0,0",
                "--probes.it_probes",
                "1",
                "--force.modes",
                "1,1",
                "--force.profiles",
                "profiles.npz",
                "--force.amplitude",
                "0.001",
                "--force.it_force",
                "2",
                "--force.seed",
                "5",
                "--force.n_channels",
                "2",
            ],
            "hlo": True,
            "profiles_from": final("plane-couette.random.iterative-cn"),
        }
    )
    runs.append(
        {
            "name": "streams.pipe-probes",
            "cmd": [
                "dnsjax",
                "--phys.system",
                "pipe",
                *SYSTEMS["pipe"],
                *res_flags("pipe"),
                *RANDOM,
                *COMMON,
                "--probes.modes",
                "1,1;0,0",
                "--probes.it_probes",
                "1",
            ],
            "hlo": False,
        }
    )
    runs.append(
        {
            "name": "twin.plane-couette",
            "cmd": [
                "dnsjax-twin",
                "--init.snapshot",
                final("plane-couette.random.iterative-cn"),
                "--twin.e0",
                "1e-6",
                "--twin.seed",
                "3",
                "--stop.max_sim_time",
                "0.02",
                "--step.dt",
                "0.005",
                "--outs.it_stats",
                "1",
            ],
            "hlo": True,
        }
    )
    return runs


def _ccf(re1: float, re2: float, eta: float) -> tuple[float, float]:
    """Circular-Couette A0, B0 (copied from tests/test_transient_growth)."""
    re_ref = re1 if re1 > 0 else re2
    a0 = (re2 - eta * re1) / ((1 + eta) * re_ref)
    b0 = eta * (re1 - eta * re2) / ((1 + eta) * (1 - eta) ** 2 * re_ref)
    return a0, b0


def tg_runs() -> list[dict]:
    runs = []
    common = ["--tg.t_max", "10", "--tg.nt", "4", "--tg.modes", "1,1"]
    y = np.linspace(1.0, -1.0, 401)
    profiles = {
        "plane-couette": np.column_stack([y, y]),
        "pipe": np.column_stack(
            [
                np.linspace(1.0, 1e-3, 401),
                1.0 - np.linspace(1.0, 1e-3, 401) ** 2,
            ]
        ),
    }
    eta, re1, rom = 0.71, 100.0, -1.2
    re2 = re1 * (1 - eta + rom) / (eta * rom - (1 - eta))
    a0, b0 = _ccf(re1, re2, eta)
    r = np.linspace(1.0 / (1 - eta), eta / (1 - eta), 401)
    profiles["quasi-keplerian"] = np.column_stack([r, a0 * r + b0 / r])
    a0, b0 = _ccf(400.0, -400.0, 0.5)
    r = np.linspace(1.0 / 0.5, 0.5 / 0.5, 401)
    profiles["taylor-couette"] = np.column_stack([r, a0 * r + b0 / r])
    for system, prof in profiles.items():
        runs.append(
            {
                "name": f"tg.{system}",
                "cmd": [
                    "python",
                    "-m",
                    "dnsjax.analysis.transient_growth",
                    "--phys.system",
                    system,
                    *SYSTEMS[system],
                    *res_flags(system, ny=21),
                    "--tg.profile",
                    "profile.txt",
                    "--tg.out_dir",
                    "out",
                    *common,
                ],
                "hlo": False,
                "profile": prof,
            }
        )
    return runs


TEXTS: list[tuple[str, list[str]]] = [
    ("help", ["dnsjax", "--help"]),
    ("twin-help", ["dnsjax-twin", "--help"]),
    (
        "tg-help",
        ["python", "-m", "dnsjax.analysis.transient_growth", "--help"],
    ),
    *[(f"help.{s}", ["dnsjax", "--help", s]) for s in SYSTEMS],
    *[(f"sample-toml.{s}", ["dnsjax", "--sample-toml", s]) for s in SYSTEMS],
    *[
        (f"twin-help.{s}", ["dnsjax-twin", "--help", s])
        for s in ("plane-couette", "plane-poiseuille")
    ],
]


def _env(hlo_dir: Path | None) -> dict:
    env = dict(os.environ)
    env.pop("FORCE_COLOR", None)
    env.update(
        NO_COLOR="1", COLUMNS="100", PYTHONHASHSEED="0", JAX_PLATFORMS="cpu"
    )
    if hlo_dir is not None:
        env["XLA_FLAGS"] = (
            f"--xla_dump_to={hlo_dir} --xla_dump_hlo_as_text "
            + env.get("XLA_FLAGS", "")
        ).strip()
    return env


def _exe(tree: Path, cmd: list[str]) -> list[str]:
    head = cmd[0]
    path = tree / ".venv" / "bin" / head
    return [str(path), *cmd[1:]]


def run_one(tree: Path, out: Path, spec: dict, hlo: bool) -> dict:
    rdir = out / "runs" / spec["name"]
    rdir.mkdir(parents=True, exist_ok=True)
    if "profiles_from" in spec:
        _write_profiles(Path(spec["profiles_from"]), rdir / "profiles.npz")
    if "profile" in spec:
        np.savetxt(rdir / "profile.txt", spec["profile"])
    hlo_dir = rdir / "hlo" if (hlo and spec.get("hlo")) else None
    t0 = time.time()
    with open(rdir / "out.log", "w") as log:
        proc = subprocess.run(
            ["nice", "-n", "10", *_exe(tree, spec["cmd"])],
            cwd=rdir,
            env=_env(hlo_dir),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    info = {
        "cmd": spec["cmd"],
        "exit": proc.returncode,
        "seconds": round(time.time() - t0, 1),
    }
    if hlo_dir is not None:
        info["hlo"] = hlo_fingerprint(hlo_dir)
    final = sorted(rdir.glob("state*.tar"))
    if final:
        info["analysis"] = analysis_view(tree, final[-1], rdir)
    print(f"[{info['exit']}] {spec['name']} ({info['seconds']} s)", flush=True)
    return info


def _write_profiles(snapshot: Path, path: Path) -> None:
    """Deterministic channel profiles on *snapshot*'s wall grid."""
    with tarfile.open(snapshot) as tf:
        meta = json.load(tf.extractfile("_dnsjax_meta.json"))
    grid = np.asarray(meta["wall_normal_grid"])
    rng = np.random.default_rng(11)
    arr = rng.standard_normal((3, 3, grid.size)) + 1j * rng.standard_normal(
        (3, 3, grid.size)
    )
    np.savez(
        path,
        system="plane-couette",
        code_grid=grid,
        component_labels=np.asarray(["u_x", "u_y", "u_z"]),
        profiles_1_1=arr,
    )


_META = re.compile(r",?\s*metadata=\{[^}]*\}")
_ATTR = re.compile(
    r",?\s*(source_file|source_line|source_end_line|"
    r"source_column|source_end_column)=[^,}\s]+"
)


_FRAME_TABLES = ("FileNames", "FunctionNames", "FileLocations", "StackFrames")


def _strip_frames(text: str) -> str:
    """Drop the stack-frame tables (file paths, line numbers)."""
    out, skipping = [], False
    for line in text.splitlines():
        if line.strip() in _FRAME_TABLES:
            skipping = True
            continue
        if skipping and (line.startswith("%") or line.startswith("ENTRY")):
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def hlo_fingerprint(hlo_dir: Path) -> dict:
    """{module name: [sha of normalised optimised HLO, ...]}."""
    import hashlib

    out: dict[str, list[str]] = {}
    for f in sorted(hlo_dir.glob("*after_optimizations.txt")):
        name = re.sub(r"^module_\d+\.", "", f.name)
        text = _ATTR.sub("", _META.sub("", _strip_frames(f.read_text())))
        out.setdefault(name, []).append(
            hashlib.sha256(text.encode()).hexdigest()[:16]
        )
    return {k: sorted(v) for k, v in sorted(out.items())}


_ANALYSIS = r"""
import json, sys, hashlib
import numpy as np
from dnsjax.analysis import read_state, geometry_info
st = read_state(sys.argv[1], return_spectral=True)
def h(arrs):
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode()); m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]
gi = geometry_info(st.params)
print(json.dumps({
    "geometry_info": repr(gi),
    "physical": h(st.physical), "physical_coords": h(st.physical_coords),
    "spectral": h(st.spectral), "spectral_coords": h(st.spectral_coords),
}))
"""


def analysis_view(tree: Path, snapshot: Path, rdir: Path) -> dict | str:
    proc = subprocess.run(
        [
            str(tree / ".venv" / "bin" / "python"),
            "-c",
            _ANALYSIS,
            str(snapshot),
        ],
        capture_output=True,
        text=True,
        env=_env(None),
        cwd=rdir,
    )
    if proc.returncode != 0:
        return "ERROR: " + proc.stderr.strip().splitlines()[-1]
    return json.loads(proc.stdout)


def capture_texts(tree: Path, out: Path, only: re.Pattern | None) -> dict:
    tdir = out / "texts"
    tdir.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, cmd in TEXTS:
        if only and not only.search(name):
            continue
        proc = subprocess.run(
            _exe(tree, cmd),
            capture_output=True,
            text=True,
            env=_env(None),
            cwd=tdir,
        )
        (tdir / f"{name}.txt").write_text(proc.stdout)
        (tdir / f"{name}.err").write_text(proc.stderr)
        res[name] = {"cmd": cmd, "exit": proc.returncode}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-hlo", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    tree, out = a.tree.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    only = re.compile(a.only) if a.only else None
    manifest: dict = {"tree": str(tree), "texts": {}, "runs": {}}
    if a.merge and (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
    manifest["git"] = subprocess.run(
        ["git", "-C", str(tree), "log", "-1", "--format=%h %s"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest["texts"].update(capture_texts(tree, out, only))

    def wave(specs: list[dict]) -> None:
        specs = [s for s in specs if not only or only.search(s["name"])]
        with ThreadPoolExecutor(a.jobs) as pool:
            for spec, info in zip(
                specs,
                pool.map(lambda s: run_one(tree, out, s, not a.no_hlo), specs),
                strict=True,
            ):
                manifest["runs"][spec["name"]] = info

    wave(solver_runs())
    wave(wave2_runs(out) + tg_runs())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    bad = [n for n, i in manifest["runs"].items() if i["exit"] != 0]
    bad += [
        f"text:{n}" for n, i in manifest["texts"].items() if i["exit"] != 0
    ]
    print(
        f"done: {len(manifest['runs'])} runs, "
        f"{len(manifest['texts'])} texts; nonzero exits: {bad}"
    )


if __name__ == "__main__":
    sys.exit(main())
