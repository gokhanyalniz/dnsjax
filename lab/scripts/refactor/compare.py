#!/usr/bin/env python3
"""Diff two captures written by ``capture.py``.

Usage::

    compare.py REF NEW [--allow REGEX ...] [--reorder REGEX ...]
                       [--only REGEX]

What counts as equal:

- texts: byte-identical stdout, the same exit code;
- snapshot tars: every member byte-identical, except
  ``_dnsjax_meta.json``, compared as JSON without its volatile keys
  (``git_hash``, timestamps, host names) and with the capture roots
  replaced by a placeholder;
- ``.dat`` / ``.bin`` / ``.txt`` / ``.npy`` / ``.csv``: byte-identical;
  ``.npz``: array by array;
- other ``.json`` sidecars: as JSON, without the volatile keys;
- the status files: their ``reason`` / ``exit`` / ``snapshot`` lines;
- the HLO fingerprints and the analysis views in ``manifest.json``.

``--allow REGEX`` names differences already reviewed (matched against
the reported line): they are listed, but do not fail the comparison.
``--reorder REGEX`` accepts a text whose lines are the same lines in
another order, for the texts it names (a flow list that changed order,
say).  ``--only REGEX`` restricts the report to the runs and texts it
names.  Exit status 1 on any unallowed difference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

VOLATILE = {
    "git_hash",
    "created",
    "started",
    "finished",
    "timestamp",
    "hostname",
    "host",
    "wall_time",
    "wall_seconds",
    "elapsed",
    "date",
    "time_created",
    "dnsjax_version",
}


ROOTS: list[str] = []


def _strip(obj):
    if isinstance(obj, str):
        for root in ROOTS:
            obj = obj.replace(root, "<CAPTURE>")
        return obj
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _json_diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a:
                out.append(f"{path}.{k}: added {str(b[k])[:80]}")
            elif k not in b:
                out.append(f"{path}.{k}: removed {str(a[k])[:80]}")
            else:
                out += _json_diff(a[k], b[k], f"{path}.{k}")
        return out
    if a != b:
        return [f"{path}: {str(a)[:80]} -> {str(b)[:80]}"]
    return []


def cmp_tar(a: Path, b: Path) -> list[str]:
    out = []
    with tarfile.open(a) as ta, tarfile.open(b) as tb:
        ma = {m.name: m for m in ta.getmembers() if m.isfile()}
        mb = {m.name: m for m in tb.getmembers() if m.isfile()}
        for name in sorted(set(ma) | set(mb)):
            if name not in ma or name not in mb:
                out.append(
                    f"member {name} only in {'new' if name in mb else 'ref'}"
                )
                continue
            da = ta.extractfile(ma[name]).read()
            db = tb.extractfile(mb[name]).read()
            if name.endswith("_dnsjax_meta.json"):
                out += [
                    f"meta{d}"
                    for d in _json_diff(
                        _strip(json.loads(da)), _strip(json.loads(db))
                    )
                ]
            elif da != db:
                out.append(f"member {name} differs")
    return out


def cmp_npz(a: Path, b: Path) -> list[str]:
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    out = []
    for k in sorted(set(za.files) | set(zb.files)):
        if k not in za.files or k not in zb.files:
            out.append(f"array {k} only in one")
        elif not np.array_equal(za[k], zb[k]):
            out.append(f"array {k} differs")
    return out


def cmp_status(a: Path, b: Path) -> list[str]:
    def keep(p):
        return [
            ln
            for ln in p.read_text().splitlines()
            if ln.split(":")[0] in ("reason", "exit", "snapshot")
        ]

    return [] if keep(a) == keep(b) else ["status lines differ"]


def cmp_file(a: Path, b: Path) -> list[str]:
    if a.name in ("FINISHED", "STOPPED", "TERMINATED", "RUNNING"):
        return cmp_status(a, b)
    if a.suffix == ".tar":
        return cmp_tar(a, b)
    if a.suffix == ".npz":
        return cmp_npz(a, b)
    if a.suffix == ".json":
        return _json_diff(
            _strip(json.loads(a.read_text())),
            _strip(json.loads(b.read_text())),
        )
    if a.read_bytes() != b.read_bytes():
        return ["bytes differ"]
    return []


REORDER: list[re.Pattern] = []
SKIP_DIRS = {"hlo"}
SKIP_FILES = {"out.log", "profile.txt"}


def compare(ref: Path, new: Path) -> list[str]:
    diffs: list[str] = []
    mr = json.loads((ref / "manifest.json").read_text())
    mn = json.loads((new / "manifest.json").read_text())
    for name, info in mr["texts"].items():
        if name not in mn["texts"]:
            diffs.append(f"text {name}: missing")
            continue
        if info["exit"] != mn["texts"][name]["exit"]:
            diffs.append(
                f"text {name}: exit {info['exit']} -> "
                f"{mn['texts'][name]['exit']}"
            )
        a = (ref / "texts" / f"{name}.txt").read_bytes()
        b = (new / "texts" / f"{name}.txt").read_bytes()
        if a != b:
            same_lines = sorted(a.splitlines()) == sorted(b.splitlines())
            if same_lines and any(p.search(name) for p in REORDER):
                diffs.append(f"text {name}: lines reordered (allowed)")
            else:
                diffs.append(f"text {name}: stdout differs")
    for name, info in mr["runs"].items():
        if name not in mn["runs"]:
            diffs.append(f"run {name}: missing")
            continue
        ni = mn["runs"][name]
        if info["exit"] != ni["exit"]:
            diffs.append(f"run {name}: exit {info['exit']} -> {ni['exit']}")
        if info.get("hlo") != ni.get("hlo"):
            ha, hb = info.get("hlo") or {}, ni.get("hlo") or {}
            for mod in sorted(set(ha) | set(hb)):
                if ha.get(mod) != hb.get(mod):
                    diffs.append(f"run {name}: hlo {mod} differs")
        if info.get("analysis") != ni.get("analysis"):
            aa, ab = info.get("analysis"), ni.get("analysis")
            if isinstance(aa, dict) and isinstance(ab, dict):
                for k in sorted(set(aa) | set(ab)):
                    if aa.get(k) != ab.get(k):
                        diffs.append(
                            f"run {name}: analysis {k} differs: "
                            f"{str(aa.get(k))[:160]} -> "
                            f"{str(ab.get(k))[:160]}"
                        )
            else:
                diffs.append(f"run {name}: analysis {aa} -> {ab}")
        rdir, ndir = ref / "runs" / name, new / "runs" / name
        for f in sorted(rdir.rglob("*")):
            rel = f.relative_to(rdir)
            if f.is_dir() or rel.parts[0] in SKIP_DIRS or f.name in SKIP_FILES:
                continue
            g = ndir / rel
            if not g.exists():
                diffs.append(f"run {name}: {rel} missing")
                continue
            diffs += [f"run {name}: {rel}: {d}" for d in cmp_file(f, g)]
        for g in sorted(ndir.rglob("*")):
            rel = g.relative_to(ndir)
            if g.is_dir() or rel.parts[0] in SKIP_DIRS or g.name in SKIP_FILES:
                continue
            if not (rdir / rel).exists():
                diffs.append(f"run {name}: {rel} new file")
    for name in mn["runs"]:
        if name not in mr["runs"]:
            diffs.append(f"run {name}: new run")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--allow", action="append", default=[])
    ap.add_argument("--reorder", action="append", default=[])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    REORDER.extend(re.compile(p) for p in a.reorder)
    allow_reordered = re.compile(r"lines reordered \(allowed\)")
    allow = [re.compile(p) for p in a.allow]
    ROOTS.extend([str(a.ref.resolve()), str(a.new.resolve())])
    diffs = compare(a.ref, a.new)
    if a.only:
        only = re.compile(a.only)
        diffs = [
            d
            for d in diffs
            if (m := re.match(r"(?:run|text) ([^:]+):", d))
            and only.search(m.group(1))
        ]
    allow.append(allow_reordered)
    bad = [d for d in diffs if not any(p.search(d) for p in allow)]
    ok = [d for d in diffs if d not in bad]
    for d in ok:
        print(f"ALLOWED  {d}")
    for d in bad:
        print(f"DIFF     {d}")
    print(f"{len(bad)} unallowed difference(s), {len(ok)} allowed.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
