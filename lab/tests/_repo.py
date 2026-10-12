"""Repository paths for the lab's test scripts, and the core's helpers.

A lab test script runs from the repository root, like a core one, with
its own directory first on ``sys.path``.  Importing this module also
puts the core suite's directory on ``sys.path``, after it, so a lab
script imports the core's shared helpers (``_live``, ``_fake_host``,
the runners the split scripts share) by the same names the core
scripts use.  Those helper modules are named with a leading underscore,
so none can be shadowed by a lab test script.
"""

import sys
from pathlib import Path

#: The lab's test directory, the lab and the repository root.
LAB_TESTS = Path(__file__).resolve().parent
LAB = LAB_TESTS.parent
REPO = LAB.parent
#: The core suite's directory.
CORE_TESTS = REPO / "tests"

if str(CORE_TESTS) not in sys.path:
    sys.path.append(str(CORE_TESTS))
