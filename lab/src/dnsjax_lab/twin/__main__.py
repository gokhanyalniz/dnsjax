"""``python -m dnsjax_lab.twin`` entry point.

Process entry only: hands off to :func:`dnsjax_lab.twin.driver.main`
(also the ``dnsjax-twin`` console-script target), so the driver is
always imported exactly once, as :mod:`dnsjax_lab.twin.driver`.
"""

import sys

from dnsjax_lab.twin.driver import main

if __name__ == "__main__":
    sys.exit(main())
