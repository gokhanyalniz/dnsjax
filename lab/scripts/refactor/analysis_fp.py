#!/usr/bin/env python3
"""Fingerprint the JAX-free analysis API on a capture's snapshots.

Usage::

    analysis_fp.py run CAPTURE PREFIX
    analysis_fp.py compare PREFIX_A PREFIX_B

``run`` reads the last snapshot of every run in *CAPTURE* (a
``capture.py`` output) through :mod:`dnsjax.analysis` and saves what
the API computes from it -- divergence, curl, gradients and
derivatives of every stored component, volume and partial integrals --
to ``PREFIX.npz``, with each snapshot's :func:`geometry_info` schema in
``PREFIX.json``.  Run it with the Python of the tree under test
(``<tree>/.venv/bin/python``), whose ``dnsjax`` is the one
fingerprinted; a lab flow's snapshot needs the lab installed there.

``compare`` checks two fingerprints for bitwise equality, array by
array, and the schemas for equality; exit status 1 on any difference.
A refactor that keeps the operators keeps the fingerprint bitwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def fingerprint(capture: Path) -> tuple[dict, dict]:
    """``(arrays, schemas)`` of every run's last snapshot in *capture*."""
    from dnsjax.analysis import (
        curl,
        derivative,
        divergence,
        geometry_info,
        gradient,
        integrate,
        read_state,
    )

    arrays: dict[str, np.ndarray] = {}
    schemas: dict[str, dict] = {}
    for run in sorted((capture / "runs").iterdir()):
        tars = sorted(run.glob("state*.tar"))
        if not tars:
            continue
        tar, key = tars[-1], run.name
        st = read_state(tar, return_spectral=True)
        info = geometry_info(st.params)
        if len(info.components) > len(st.spectral):
            st = read_state(
                tar,
                return_spectral=True,
                components=tuple(range(len(info.components))),
            )
        schemas[key] = {
            "family": info.family,
            "name": list(info.name),
            "components": list(info.components),
            "n": list(info.n),
            "m0": info.azimuthal_m0,
            "metric": getattr(info, "metric", None) is not None,
        }
        sp, coords = st.spectral, st.spectral_coords
        vel = tuple(sp[:3])
        arrays[f"{key}|div"] = divergence(vel, st.params, coords)
        for i, w in enumerate(curl(vel, st.params, coords)):
            arrays[f"{key}|curl{i}"] = w
        phys, pcoords = st.physical, st.physical_coords
        energy = sum(np.abs(u) ** 2 for u in phys[:3])
        arrays[f"{key}|energy"] = np.asarray(
            integrate(energy, st.params, pcoords)
        )
        arrays[f"{key}|prof"] = np.asarray(
            integrate(phys[0], st.params, pcoords, directions=info.name[1:])
        )
        arrays[f"{key}|axial"] = np.asarray(
            integrate(phys[0], st.params, pcoords, directions=info.name[2:3])
        )
        cylindrical = info.family == "cylindrical"
        for c, label in enumerate(info.components):
            parity = {"cylindrical_parity": label} if cylindrical else {}
            grads = gradient(sp[c], st.params, coords, **parity)
            for a, g in enumerate(grads):
                arrays[f"{key}|grad{c}.{a}"] = g
            arrays[f"{key}|d{c}"] = derivative(
                sp[c], info.name[0], st.params, coords, **parity
            )
    return arrays, schemas


def compare(prefix_a: str, prefix_b: str) -> list[str]:
    """The differences between two saved fingerprints."""
    za, zb = np.load(f"{prefix_a}.npz"), np.load(f"{prefix_b}.npz")
    out = [
        f"array {k}: only in one"
        for k in sorted(set(za.files) ^ set(zb.files))
    ]
    out += [
        f"array {k}: differs"
        for k in sorted(set(za.files) & set(zb.files))
        if not np.array_equal(za[k], zb[k])
    ]
    sa = json.loads(Path(f"{prefix_a}.json").read_text())
    sb = json.loads(Path(f"{prefix_b}.json").read_text())
    out += [
        f"schema {k}: differs"
        for k in sorted(set(sa) | set(sb))
        if sa.get(k) != sb.get(k)
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("capture", type=Path)
    run.add_argument("prefix")
    cmp = sub.add_parser("compare")
    cmp.add_argument("prefix_a")
    cmp.add_argument("prefix_b")
    a = ap.parse_args()
    if a.cmd == "run":
        arrays, schemas = fingerprint(a.capture)
        np.savez(f"{a.prefix}.npz", **arrays)
        Path(f"{a.prefix}.json").write_text(
            json.dumps(schemas, indent=1, sort_keys=True)
        )
        print(f"{len(arrays)} arrays from {len(schemas)} snapshots")
        return 0
    diffs = compare(a.prefix_a, a.prefix_b)
    for line in diffs:
        print(line)
    print(f"{len(diffs)} difference(s)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
