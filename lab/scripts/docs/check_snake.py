"""List backticked snake_case tokens that name nothing in the tree.

A ``snake_case`` word in backticks reads as an identifier; one that
matches no top-level name, no class member and no parameter field in
the tree is a stale name or a misspelling.  Scans the docstrings and
comments of every tracked ``.py`` file and every tracked ``.md`` file
but the agent notes and this directory, through the index
:mod:`check_refs` builds at import, and prints
``file:line: token (xN)`` with each token's count.
Review by hand: a local variable or a shell/TOML key is a legitimate
hit.

Usage::

    uv run python lab/scripts/docs/check_snake.py
"""

import re
import sys
from collections import Counter

from check_refs import (
    ROOT,
    SELF,
    TICK,
    all_names,
    docstrings_and_comments,
    fields,
    mdfiles,
    pyfiles,
    top,
)

SNAKE = re.compile(r"^_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


def main() -> int:
    known_fields = set().union(*fields.values())
    found = []

    def scan(text, here, lineno):
        for m in TICK.finditer(text):
            tok = m.group(1).strip().rstrip("()")
            if (
                SNAKE.match(tok)
                and tok not in all_names
                and tok not in known_fields
                and tok not in top.get(here, set())
            ):
                found.append((here, lineno, tok))

    for f in pyfiles:
        if f.startswith(SELF):
            continue
        for text, lineno in docstrings_and_comments(f):
            scan(text, f, lineno)
    for f in mdfiles:
        if f.startswith(SELF):
            continue
        for i, line in enumerate((ROOT / f).read_text().splitlines(), 1):
            scan(line, f, i)
    counts = Counter(tok for _, _, tok in found)
    for here, lineno, tok in found:
        print(f"{here}:{lineno}: {tok}  (x{counts[tok]})")
    print(len(found), "hits,", len(counts), "distinct", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
