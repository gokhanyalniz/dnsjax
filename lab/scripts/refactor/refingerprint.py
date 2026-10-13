#!/usr/bin/env python3
"""Recompute a capture's HLO fingerprints from the dumps it kept.

Usage::

    refingerprint.py CAPTURE

For a capture taken before a change to :func:`capture.hlo_fingerprint`
(how the optimized HLO is normalized before hashing), so that an older
reference and a new capture are compared under one rule.  Rewrites the
capture's ``manifest.json`` in place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import hlo_fingerprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("capture", type=Path)
    capture = ap.parse_args().capture
    manifest = json.loads((capture / "manifest.json").read_text())
    for name, info in manifest["runs"].items():
        hlo_dir = capture / "runs" / name / "hlo"
        if "hlo" in info and hlo_dir.is_dir():
            info["hlo"] = hlo_fingerprint(hlo_dir)
    (capture / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("refingerprinted", capture)
    return 0


if __name__ == "__main__":
    sys.exit(main())
