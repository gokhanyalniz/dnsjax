"""Parse-validate one documented command (JAX-free), one per process.

Usage: python parse_cmds.py <kind> <args...>
  kind = solver | twin | toml:<path> | twin-toml:<path>

``twin-toml:<path>`` resolves a ``dnsjax-twin`` ``parameters.toml`` as
the driver does: ``[twin]`` is the entry point's own section and every
extension relevant to the flow rides beside it.  The snapshot layer
reads ``init.snapshot`` only where that path exists, so run it from a
directory without the parent snapshot to check the file alone (and to
leave a multi-GB snapshot unopened).
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
    elif kind.startswith("twin-toml:"):
        from dnsjax_lab.twin.driver import TWIN_EXTENSION

        bootstrap.resolve_parameters(
            argv,
            toml_path=Path(kind[len("twin-toml:") :]),
            own_extensions=(TWIN_EXTENSION,),
            prog="dnsjax-twin",
        )
    print("OK")
except SystemExit as e:
    print("SYSTEMEXIT", e.code)
except Exception as e:  # noqa: BLE001
    print("ERROR", type(e).__name__, str(e)[:400])
