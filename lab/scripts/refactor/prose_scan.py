#!/usr/bin/env python3
"""List the lab's vocabulary in the core's prose.

Usage::

    prose_scan.py [ROOT ...] [--count] [--pattern REGEX]

``tests/test_layering.py`` keeps the core's *code* from naming the lab;
this lists where its *prose* does -- docstrings, comments and other
strings in every ``.py`` file under each *ROOT* (default: the core's
``src/dnsjax``, ``tests`` and ``scripts``), one line per hit, grouped by
file.  The core's prose describes the seam a lab feature plugs into
and names the lab only where a measurement was taken on a lab flow or
a lab module consumes a core contract, so a review reads every hit.
The default vocabulary (:data:`PATTERN`) matches the lab's flows, its
rheology, the twin runs, the probe and forcing streams and the
response analysis; it also matches words with a core meaning (the
transient-growth "probe" step, a "dense twin" of a function, the
Kolmogorov or Adams-Bashforth "forcing"), which a review skips.
``--count`` prints the per-file counts only.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

#: The repository holding this script (``lab/scripts/refactor/``).
REPO = Path(__file__).resolve().parents[3]

#: The lab's vocabulary, case-insensitive.
PATTERN = (
    r"curved|toroid|viscoelast|sptt|\bptt\b|conformation|polymer|rheolog|"
    r"weissenberg|keplerian|r_omega|\bdean\b|twin|\bprobes?\b|"
    r"\[force\]|force\.|``force``|force_params|forcing\b|dnsjax_lab|"
    r"analysis\.response|"
    r"response/|\bkick|\blab\b"
)


def prose_hits(path: Path, pattern: re.Pattern) -> list[tuple[int, str]]:
    """``(line, text)`` of every prose line in *path* matching."""
    hits = []
    source = io.StringIO(path.read_text()).readline
    for tok in tokenize.generate_tokens(source):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING) and (
            tok.type != getattr(tokenize, "FSTRING_START", -1)
        ):
            continue
        for k, line in enumerate(tok.string.splitlines()):
            if pattern.search(line):
                hits.append((tok.start[0] + k, line.strip()))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("roots", type=Path, nargs="*")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--pattern", default=PATTERN)
    a = ap.parse_args()
    roots = a.roots or [
        REPO / "src" / "dnsjax",
        REPO / "tests",
        REPO / "scripts",
    ]
    pattern = re.compile(a.pattern, re.IGNORECASE)
    total = 0
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            hits = prose_hits(path, pattern)
            if not hits:
                continue
            total += len(hits)
            print(f"=== {path.relative_to(root.parent)} ({len(hits)})")
            if not a.count:
                for line, text in hits:
                    print(f"  L{line}: {text[:110]}")
    print("total", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
