"""Parse-validate one documented command (JAX-free), one per process.

Usage: python parse_cmds.py <kind> <args...>
  kind = solver | twin | toml:<path>
"""

import sys
from pathlib import Path

kind = sys.argv[1]
argv = sys.argv[2:]
from dnsjax import bootstrap  # noqa: E402

try:
    if kind == "solver":
        bootstrap.resolve_parameters(argv, toml_path=False)
    elif kind == "twin":
        from dnsjax_lab.twin.driver import TWIN_EXTENSION

        bootstrap.resolve_parameters(
            argv,
            toml_path=False,
            extensions=(TWIN_EXTENSION,),
            prog="dnsjax-twin",
        )
    elif kind.startswith("toml:"):
        bootstrap.resolve_parameters(argv, toml_path=Path(kind[5:]))
    print("OK")
except SystemExit as e:
    print("SYSTEMEXIT", e.code)
except Exception as e:  # noqa: BLE001
    print("ERROR", type(e).__name__, str(e)[:400])
