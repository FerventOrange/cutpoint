"""Allow ``python -m cutpoint``."""

import sys

from cutpoint.cli import main

if __name__ == "__main__":
    sys.exit(main())
