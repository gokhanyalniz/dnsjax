"""Print the metadata and embedded stats of snapshots: ``dnsjax-info``.

Reads the ``_dnsjax_meta.json`` member of each dnsjax tar named (a
state snapshot, a reduced-resolution ``lowres`` file, a twin cube: the
format is in :mod:`dnsjax.snapshot`) and the optional
``_dnsjax_stats.json`` beside it, through
:func:`dnsjax.snapshot_meta.read_snapshot_json`.  Nothing is extracted
and no component chunk is read: the tar headers and those two members
are the whole read, the same for a state of any size.  The module
imports the standard library and :mod:`dnsjax.snapshot_meta` alone (no
NumPy, JAX or pydantic).

Usage::

    dnsjax-info state00016.tar              # every record
    dnsjax-info --time state*.tar           # t, it, isnap, step.dt
    dnsjax-info --time --provenance state*.tar
    dnsjax-info --time --stats state*.tar   # with the stats columns
    dnsjax-info --raw state00016.tar        # the stored JSON

In a checkout, prefix ``uv run``; ``python -m
dnsjax.tools.snapshot_info`` is the same entry point.

Views
-----
- **Default**: per file, the top-level records as aligned ``key value``
  rows (a nested entry such as ``lowres`` or ``twin`` as dotted keys, a
  numeric list longer than eight, the wall-normal grid among them, as
  its count and range), a ``stats`` row saying whether diagnostics are
  embedded, then the embedded parameters TOML-shaped, as the solver
  prints them at startup (an unset field is a commented ``#name =``).
  Several files print one block each under a ``==> path <==`` header.
- ``--time``: a table, one row per file, of the records that change
  from one snapshot to the next within a run (:data:`CLOCK`): ``t``,
  ``it``, ``isnap`` and ``step.dt``, the live time step, which
  ``step.adaptive`` changes during a run.  ``--provenance`` (implying
  ``--time``) adds the records that change from one run segment, a
  resume, to the next (:data:`PROVENANCE`).  A record a file lacks or
  holds as ``null`` (the ``isnap`` of a ``lowres`` file) prints ``-``.
- ``--stats``: the embedded diagnostics (the ``get_stats`` values of
  the stored state), as a ``[stats]`` block after the parameters or as
  further table columns.
- ``--raw``: JSON, for programs and agents.  One file and no other flag
  prints the metadata member exactly as stored.  ``--time`` cuts the
  stored object down to the selected records, each at its stored path
  (``.params.step.dt`` reads alike in both); ``--stats`` wraps the
  output in one object, ``{"meta": ..., "stats": ...}``, with ``stats``
  ``null`` when none is embedded.  Several files print JSON Lines, one
  object per file in argument order: ``{"path": ..., "meta": ...}``,
  plus ``"stats"`` under ``--stats``.

Floats print in full, in Python's shortest round-trip form.  A run
accumulates ``t`` one ``dt`` at a time, so it carries the round-off of
that sum: 160000 steps of ``0.01`` from zero give
``1599.9999999986899``.  ``it`` is the exact step count.

A path that cannot be read (missing, not a dnsjax tar, damaged, or of
a format below :data:`dnsjax.snapshot_meta.MIN_FORMAT_VERSION`) is
reported on stderr and skipped, and the exit status is 1, as it is when
the reader of a pipe closes it early (``| head``).
"""

import argparse
import json
import math
import os
import sys

from dnsjax.snapshot_meta import read_snapshot_json

#: The records that change from one snapshot to the next within a run,
#: as paths into the metadata.
CLOCK: tuple[tuple[str, ...], ...] = (
    ("t",),
    ("it",),
    ("isnap",),
    ("params", "step", "dt"),
)

#: The records that change from one run segment (a resume) to the next:
#: the code revision, the device count and layout, and how the segment
#: started (``init.t0`` / ``init.it0`` are the resumed snapshot's ``t``
#: / ``it``; ``init.snapshot`` and ``init.force_resume`` are the
#: lineage fields a resume never inherits).  A path naming a section
#: selects all of it.
PROVENANCE: tuple[tuple[str, ...], ...] = (
    ("git_hash",),
    ("n_devices",),
    ("params", "dist"),
    ("params", "init", "snapshot"),
    ("params", "init", "force_resume"),
    ("params", "init", "t0"),
    ("params", "init", "it0"),
    ("params", "init", "isnap0"),
)

#: A numeric list longer than this prints as its count and range.
_LIST_SUMMARY: int = 8

_PROG: str = "dnsjax-info"


def _project(meta: dict, paths) -> dict:
    """*meta* cut down to the records at *paths*, in stored order.

    Each kept record stays at its stored path, so a reader looks it up
    as in the full object; a path naming a dict keeps all of it, and a
    path *meta* lacks is skipped.
    """
    out = {}
    for key, value in meta.items():
        rest = [path[1:] for path in paths if path[0] == key]
        if not rest:
            continue
        if not all(rest):
            out[key] = value
        elif isinstance(value, dict) and (sub := _project(value, rest)):
            out[key] = sub
    return out


def _columns(meta: dict, paths) -> dict:
    """``{label: value}`` of the records at *paths*, in path order.

    The label is the dotted path, ``params.`` dropped (``step.dt``, the
    public name a TOML file and the CLI use); a section expands to one
    column per field, and a path *meta* lacks is skipped.
    """
    cols = {}
    for path in paths:
        node = meta
        for key in path:
            if not isinstance(node, dict) or key not in node:
                break
            node = node[key]
        else:
            label = ".".join(path[1:] if path[0] == "params" else path)
            if isinstance(node, dict):
                for key, value in node.items():
                    cols[f"{label}.{key}"] = value
            else:
                cols[label] = node
    return cols


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _plain(value) -> str:
    """A value in the readable views: a string bare, the rest as JSON."""
    return value if isinstance(value, str) else json.dumps(value)


def _summary(value) -> str:
    """:func:`_plain`, with a long numeric list as its count and range."""
    if (
        isinstance(value, list)
        and len(value) > _LIST_SUMMARY
        and all(_is_number(v) for v in value)
    ):
        lo, hi = _plain(min(value)), _plain(max(value))
        return f"{len(value)} values in [{lo}, {hi}]"
    return _plain(value)


def _toml(value) -> str:
    """A TOML-shaped value: its JSON, but for the non-finite floats."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)  # TOML's nan / inf / -inf, not JSON's NaN
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    return json.dumps(value)


def _flat(meta: dict, prefix: str = "") -> list[tuple[str, str]]:
    """``(dotted key, value)`` rows of *meta*, nested dicts flattened."""
    rows = []
    for key, value in meta.items():
        if isinstance(value, dict) and value:
            rows += _flat(value, f"{prefix}{key}.")
        else:
            rows.append((f"{prefix}{key}", _summary(value)))
    return rows


def _aligned(rows: list[tuple[str, str]]) -> list[str]:
    width = max((len(key) for key, _ in rows), default=0)
    return [f"{key:<{width}}  {value}" for key, value in rows]


def _readable(meta: dict, stats: dict | None, show_stats: bool) -> list[str]:
    """The default view of one file (module docstring, "Views")."""
    rows = _flat({k: v for k, v in meta.items() if k != "params"})
    if not show_stats:
        state = "not embedded" if stats is None else "embedded (--stats)"
        rows.append(("stats", state))
    lines = _aligned(rows)
    for section, fields in (meta.get("params") or {}).items():
        lines += ["", f"[{section}]"]
        for name, value in fields.items():
            if value is None:
                lines.append(f"#{name} =")
            else:
                lines.append(f"{name} = {_toml(value)}")
    if show_stats:
        lines += ["", "[stats]"]
        if stats is None:
            lines.append("(none embedded)")
        else:
            lines += _aligned([(k, _plain(v)) for k, v in stats.items()])
    return lines


def _table(rows: list[tuple[str, dict]]) -> list[str]:
    """Aligned lines of ``(path, {label: value})`` rows, a header first.

    The columns are every label in first-seen order; a missing or
    ``null`` cell prints ``-``, and a column of numbers is right-aligned.
    """
    labels = list(dict.fromkeys(k for _, cols in rows for k in cols))
    right = [False] + [
        all(_is_number(c[k]) for _, c in rows if c.get(k) is not None)
        for k in labels
    ]
    cells = [["file", *labels]] + [
        [path, *("-" if c.get(k) is None else _plain(c[k]) for k in labels)]
        for path, c in rows
    ]
    widths = [max(len(row[j]) for row in cells) for j in range(len(right))]
    return [
        "  ".join(
            cell.rjust(w) if r else cell.ljust(w)
            for cell, w, r in zip(row, widths, right, strict=True)
        ).rstrip()
        for row in cells
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Print the metadata of dnsjax snapshots, and the stats "
            "embedded in them, reading their two JSON members alone "
            "(never the state).  The views and the JSON shapes: the "
            "dnsjax.tools.snapshot_info module docstring."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="SNAPSHOT",
        help="a dnsjax tar: a state snapshot, a lowres file or a cube",
    )
    parser.add_argument(
        "-t",
        "--time",
        action="store_true",
        help=(
            "only the records that change between snapshots of a run "
            "(t, it, isnap, step.dt), as a table with one row per file"
        ),
    )
    parser.add_argument(
        "-p",
        "--provenance",
        action="store_true",
        help=(
            "--time, adding the records that change between run "
            "segments (git_hash, n_devices, the [dist] section, "
            "init.snapshot, init.force_resume, init.t0, init.it0, "
            "init.isnap0)"
        ),
    )
    parser.add_argument(
        "-s",
        "--stats",
        action="store_true",
        help="also the embedded stats (the get_stats values)",
    )
    parser.add_argument(
        "-r",
        "--raw",
        action="store_true",
        help=(
            "print JSON: for one file alone, the metadata member as "
            "stored; with --time, the stored object cut down to those "
            'records; with --stats, {"meta": ..., "stats": ...}; for '
            "several files, one object per line, with a path key"
        ),
    )
    return parser


def _emit(args, snaps: list[tuple[str, dict, bytes, dict | None]]) -> None:
    """Print the view *args* selects of the files read (*snaps*)."""
    if args.provenance:
        select = CLOCK + PROVENANCE
    elif args.time:
        select = CLOCK
    else:
        select = None
    single = len(args.paths) == 1
    if args.raw:
        for path, meta, raw, stats in snaps:
            doc = meta if select is None else _project(meta, select)
            if single and select is None and not args.stats:
                sys.stdout.buffer.write(raw.rstrip(b"\n") + b"\n")
            elif single:
                if args.stats:
                    doc = {"meta": doc, "stats": stats}
                print(json.dumps(doc, indent=2))
            else:
                line = {"path": path, "meta": doc}
                if args.stats:
                    line["stats"] = stats
                print(json.dumps(line))
    elif select is not None:
        rows = []
        for path, meta, _, stats in snaps:
            cols = _columns(meta, select)
            if args.stats:
                cols |= stats or {}
            rows.append((path, cols))
        print("\n".join(_table(rows)))
    else:
        for i, (path, meta, _, stats) in enumerate(snaps):
            if i:
                print()
            if not single:
                print(f"==> {path} <==")
            print("\n".join(_readable(meta, stats, args.stats)))


def main(argv: list[str] | None = None) -> int:
    """Run ``dnsjax-info`` on *argv*; return the exit status."""
    args = _parser().parse_args(argv)
    snaps = []
    failed = False
    for path in args.paths:
        try:
            meta, raw, stats = read_snapshot_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(
                f"{_PROG}: {path}: a JSON member does not parse: {exc}",
                file=sys.stderr,
            )
            failed = True
        except (OSError, ValueError) as exc:
            print(f"{_PROG}: {exc}", file=sys.stderr)
            failed = True
        else:
            snaps.append((path, meta, raw, stats))
    try:
        _emit(args, snaps)
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader went away (``| head``): stop without a traceback,
        # pointing stdout at /dev/null so the flush at exit cannot raise
        # again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
