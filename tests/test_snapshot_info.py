r"""``dnsjax-info``, a snapshot's metadata without its state (no JAX).

:mod:`dnsjax.tools.snapshot_info` prints the JSON members of dnsjax
tars.  This script pins its views on archives built here with the
standard library in the writer's member order and PAX headers
(:func:`dnsjax.snapshot.write_archive`): two state snapshots with
embedded stats, a ``lowres`` file (``isnap`` null, nested entries, no
stats), a format-5 file and one with an unparsable stats member.  Each
case runs the module in a subprocess, as a user would, in a few seconds
in all:

- ``imports``: the module imports neither NumPy, JAX nor pydantic.
- ``entry``: the installed ``dnsjax-info`` console script runs.
- ``readable``: the default view's rows, the summarized wall-normal
  grid, the TOML-shaped parameters (an unset field commented), the
  ``stats`` row and the ``[stats]`` block, the per-file headers.
- ``raw``: one file alone prints the metadata member byte for byte, a
  member written in a layout other than the writer's included (so it is
  the stored text, not a re-serialization); ``--stats`` wraps it.
- ``raw-select``: ``--time`` and ``--provenance`` keep exactly their
  records, each at its stored path, in stored order.
- ``json-lines``: several files give one object per line, in argument
  order, with a ``path`` key.
- ``table``: the ``--time`` columns, ``-`` for a null or missing cell,
  right-aligned numbers, the stats columns.
- ``errors``: a missing path, a file that is not a tar, a format below
  :data:`~dnsjax.snapshot_meta.MIN_FORMAT_VERSION` and a damaged stats
  member are each one stderr line, the readable files still print, and
  the exit status is 1.
- ``pipe``: a reader closing the pipe early ends the run with status 1
  and no traceback.
- ``members``: only the two JSON members are extracted (in process,
  counting ``TarFile.extractfile``).

:func:`~dnsjax.snapshot_meta.read_snapshot_json` on an archive the real
writer made: ``tests/test_snapshot.py``'s ``isnap + embedded stats``
case.

Run as a script::

    uv run python tests/test_snapshot_info.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _live import report  # noqa: E402

MODULE = "dnsjax.tools.snapshot_info"
GRID = [-1.0 + j / 8 for j in range(17)]
STATS_A = {"D": 1.5, "E": 0.5, "E'": 0.25}
STATS_B = {"D": 1.25, "E": 0.75, "E'": 0.125}


def state_meta(t: float, it: int, isnap: int | None, **extra) -> dict:
    """Metadata in the writer's key order (``snapshot._metadata``)."""
    return {
        "format_version": 6,
        "git_hash": "0123456789ab",
        "t": t,
        "it": it,
        "isnap": isnap,
        "geometry": "wall_bounded",
        "system": "plane-couette",
        "native_shape": [3, 17, 7, 4],
        "dtype": "complex128",
        "n_devices": 2,
        "wall_normal_grid": GRID,
        "params": {
            "dist": {"np0": 2, "np1": 1, "platform": "cpu"},
            "phys": {"re": 400.0, "system": "plane-couette"},
            "init": {
                "snapshot": "parent.tar",
                "t0": 100.0,
                "it0": 10000,
                "isnap0": 0,
                "force_resume": False,
            },
            "step": {"dt": 0.01, "adaptive": True, "dt_max": None},
        },
        **extra,
    }


def write_tar(path: Path, meta: bytes, stats: bytes | None = None) -> None:
    """A dnsjax tar of *meta* (and *stats*) plus three state chunks."""
    members = [("_dnsjax_meta.json", meta)]
    if stats is not None:
        members.append(("_dnsjax_stats.json", stats))
    members.append(("state/zarr.json", b"{}"))
    members += [(f"state/c/{c}/0/0/0", bytes(4096)) for c in range(3)]
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def dumps(obj) -> bytes:
    """The writer's serialization: indented, no trailing newline."""
    return json.dumps(obj, indent=2).encode()


META_A = state_meta(0.1 + 0.2, 30, 3)
META_B = state_meta(1.25, 125, 12)
META_LOW = state_meta(
    0.5,
    50,
    None,
    lowres={"field": "state", "source_res": {"nx": 32, "ny": 33}},
)


def build(d: Path) -> None:
    write_tar(d / "a.tar", dumps(META_A), dumps(STATS_A))
    write_tar(d / "b.tar", dumps(META_B), dumps(STATS_B))
    write_tar(d / "low.tar", dumps(META_LOW))
    compact = json.dumps(META_A, separators=(",", ":")).encode()
    write_tar(d / "compact.tar", compact)
    write_tar(d / "old.tar", dumps({**META_A, "format_version": 5}))
    write_tar(d / "badstats.tar", dumps(META_A), b"{not json")
    (d / "notatar.txt").write_text("plain text\n")


def run(d: Path, *args: str, exe: list[str] | None = None):
    """Run the module on *args* in *d*; ``(status, stdout bytes, stderr)``."""
    env = {**os.environ, "NO_COLOR": "1"}
    env.pop("FORCE_COLOR", None)
    cmd = exe or [sys.executable, "-m", MODULE]
    r = subprocess.run(
        [*cmd, *args], cwd=d, capture_output=True, env=env, check=False
    )
    return r.returncode, r.stdout, r.stderr.decode()


def ok(d: Path, *args: str) -> str:
    """stdout of a run that must exit 0 with an empty stderr."""
    status, out, err = run(d, *args)
    assert status == 0 and not err, (args, status, err)
    return out.decode()


def case_imports(d: Path) -> None:
    code = (
        f"import sys, {MODULE}; "
        "print(sorted({'numpy', 'jax', 'pydantic'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    ).stdout
    assert out.strip() == "[]", out


def case_entry(d: Path) -> None:
    exe = Path(sys.executable).with_name("dnsjax-info")
    assert exe.is_file(), f"{exe} is missing: is the package installed?"
    status, out, err = run(d, "--time", "a.tar", exe=[str(exe)])
    assert status == 0 and not err, (status, err)
    assert out.decode().split()[:2] == ["file", "t"], out


def rows(text: str) -> dict[str, str]:
    """``{key: value}`` of the aligned ``key  value`` lines of *text*."""
    out = {}
    for line in text.splitlines():
        key, sep, value = line.partition("  ")
        if sep and not key.startswith(" "):
            out[key] = value.strip()
    return out


def case_readable(d: Path) -> None:
    text = ok(d, "a.tar")
    got = rows(text)
    assert got["t"] == "0.30000000000000004", got["t"]
    assert got["isnap"] == "3" and got["system"] == "plane-couette", got
    assert got["native_shape"] == "[3, 17, 7, 4]", got
    assert got["wall_normal_grid"] == "17 values in [-1.0, 1.0]", got
    assert got["stats"] == "embedded (--stats)", got
    lines = text.splitlines()
    for want in (
        "[step]",
        "dt = 0.01",
        "adaptive = true",
        "#dt_max =",
        'snapshot = "parent.tar"',
    ):
        assert want in lines, (want, lines)
    assert "[stats]" not in lines and "==> a.tar <==" not in lines

    lines = ok(d, "--stats", "a.tar").splitlines()
    block = lines[lines.index("[stats]") + 1 :]
    assert rows("\n".join(block)) == {"D": "1.5", "E": "0.5", "E'": "0.25"}
    assert "stats" not in rows("\n".join(lines))

    got = rows(ok(d, "low.tar"))
    assert got["isnap"] == "null" and got["stats"] == "not embedded", got
    assert got["lowres.field"] == "state", got
    assert got["lowres.source_res.ny"] == "33", got
    lines = ok(d, "-s", "low.tar").splitlines()
    assert lines[lines.index("[stats]") + 1] == "(none embedded)", lines

    lines = ok(d, "a.tar", "low.tar").splitlines()
    assert lines[0] == "==> a.tar <==", lines[0]
    i = lines.index("==> low.tar <==")
    assert lines[i - 1] == "", lines[i - 1]


def case_raw(d: Path) -> None:
    out = run(d, "--raw", "a.tar")[1]
    assert out == dumps(META_A) + b"\n", out[:200]
    compact = json.dumps(META_A, separators=(",", ":")).encode()
    out = run(d, "--raw", "compact.tar")[1]
    assert out == compact + b"\n", out[:200]
    doc = json.loads(ok(d, "--raw", "--stats", "a.tar"))
    assert doc == {"meta": META_A, "stats": STATS_A}, doc
    doc = json.loads(ok(d, "-rs", "low.tar"))
    assert doc == {"meta": META_LOW, "stats": None}, doc


def case_raw_select(d: Path) -> None:
    doc = json.loads(ok(d, "-rt", "a.tar"))
    want = {"t": META_A["t"], "it": 30, "isnap": 3}
    want["params"] = {"step": {"dt": 0.01}}
    assert doc == want and list(doc) == list(want), doc

    doc = json.loads(ok(d, "-rp", "a.tar"))
    params = META_A["params"]
    want = {
        "git_hash": "0123456789ab",
        "t": META_A["t"],
        "it": 30,
        "isnap": 3,
        "n_devices": 2,
        "params": {
            "dist": params["dist"],
            "init": params["init"],
            "step": {"dt": 0.01},
        },
    }
    assert doc == want and list(doc) == list(want), doc
    assert list(doc["params"]) == ["dist", "init", "step"], doc

    doc = json.loads(ok(d, "-rts", "a.tar"))
    assert doc["stats"] == STATS_A and doc["meta"]["it"] == 30, doc


def case_json_lines(d: Path) -> None:
    lines = ok(d, "-r", "b.tar", "a.tar").splitlines()
    docs = [json.loads(line) for line in lines]
    assert [x["path"] for x in docs] == ["b.tar", "a.tar"], docs
    assert docs[1]["meta"] == META_A and "stats" not in docs[1], docs[1]
    docs = [
        json.loads(x)
        for x in ok(d, "-rts", "a.tar", "low.tar").split("\n")
        if x
    ]
    assert docs[0]["meta"]["t"] == META_A["t"], docs[0]
    assert docs[0]["stats"] == STATS_A and docs[1]["stats"] is None, docs


def case_table(d: Path) -> None:
    lines = ok(d, "--time", "a.tar", "low.tar").splitlines()
    assert lines[0].split() == ["file", "t", "it", "isnap", "step.dt"]
    assert lines[1].split() == [
        "a.tar",
        "0.30000000000000004",
        "30",
        "3",
        "0.01",
    ], lines[1]
    assert lines[2].split() == ["low.tar", "0.5", "50", "-", "0.01"]
    # Every column is numeric bar the first, so each line is padded to
    # the same width.
    assert len({len(line) for line in lines}) == 1, lines

    lines = ok(d, "-ts", "a.tar", "low.tar", "b.tar").splitlines()
    assert lines[0].split()[5:] == ["D", "E", "E'"], lines[0]
    assert lines[2].split()[5:] == ["-", "-", "-"], lines[2]
    assert lines[3].split()[5:] == ["1.25", "0.75", "0.125"], lines[3]

    lines = ok(d, "--provenance", "a.tar").splitlines()
    head, row = lines[0].split(), lines[1].split()
    want = {
        "git_hash": "0123456789ab",
        "n_devices": "2",
        "dist.np0": "2",
        "dist.platform": "cpu",
        "init.snapshot": "parent.tar",
        "init.force_resume": "false",
        "init.t0": "100.0",
        "init.it0": "10000",
        "init.isnap0": "0",
    }
    got = dict(zip(head, row, strict=True))
    assert head[:5] == ["file", "t", "it", "isnap", "step.dt"], head
    assert {k: got.get(k) for k in want} == want, got


def case_errors(d: Path) -> None:
    status, out, err = run(
        d,
        "-t",
        "missing.tar",
        "a.tar",
        "notatar.txt",
        "old.tar",
        "badstats.tar",
    )
    assert status == 1, status
    lines = out.decode().splitlines()
    assert [x.split()[0] for x in lines] == ["file", "a.tar"], lines
    err_lines = err.splitlines()
    assert len(err_lines) == 4, err
    for line, want in zip(
        err_lines,
        ("missing.tar", "notatar.txt", "format_version 5", "badstats.tar"),
        strict=True,
    ):
        assert line.startswith("dnsjax-info: ") and want in line, line
    assert "does not parse" in err_lines[3], err_lines[3]
    assert "Traceback" not in err, err


def case_pipe(d: Path) -> None:
    # Enough JSON Lines to fill a pipe's buffer several times over.
    env = {**os.environ, "NO_COLOR": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", MODULE, "-r", *["a.tar"] * 400],
        cwd=d,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    first = proc.stdout.readline()
    proc.stdout.close()
    err = proc.stderr.read().decode()
    status = proc.wait(timeout=60)
    assert json.loads(first)["path"] == "a.tar", first[:100]
    assert status == 1 and "Traceback" not in err, (status, err)


def case_members(d: Path) -> None:
    from dnsjax.tools import snapshot_info

    seen = []
    original = tarfile.TarFile.extractfile

    def spy(self, member):
        seen.append(getattr(member, "name", member))
        return original(self, member)

    tarfile.TarFile.extractfile = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            status = snapshot_info.main(["--stats", str(d / "a.tar")])
    finally:
        tarfile.TarFile.extractfile = original
    assert status == 0, status
    assert set(seen) == {"_dnsjax_meta.json", "_dnsjax_stats.json"}, seen


CASES = {
    "imports": case_imports,
    "entry": case_entry,
    "readable": case_readable,
    "raw": case_raw,
    "raw-select": case_raw_select,
    "json-lines": case_json_lines,
    "table": case_table,
    "errors": case_errors,
    "pipe": case_pipe,
    "members": case_members,
}


def main() -> int:
    passed, failures = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        build(d)
        for name, case in CASES.items():
            try:
                case(d)
            except AssertionError as exc:
                print(f"  FAIL  {name}: {exc}")
                failures.append((name, str(exc)))
            else:
                print(f"  PASS  {name}")
                passed += 1
    return report(passed, failures)


if __name__ == "__main__":
    sys.exit(main())
