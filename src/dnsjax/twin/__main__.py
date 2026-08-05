"""``python -m dnsjax.twin`` entry point.

Process entry only: hands off to :func:`dnsjax.twin.driver.main`
(also the ``dnsjax-twin`` console-script target), so the driver is
always imported exactly once, as :mod:`dnsjax.twin.driver`.
"""

import sys

from .driver import main

if __name__ == "__main__":
    sys.exit(main())
