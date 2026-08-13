"""Entry point for ``python -m gensrt`` and for the packaged executable.

The ``if __name__`` guard and the ``sys.exit`` are both load-bearing.

Without the guard, *importing* this module runs the CLI — which meant that
``gensrt --self-check``, whose whole job is to import every module in the
package, re-entered the CLI and ran itself twice.  Anything that walks the
package (a test collector, a documentation tool, PyInstaller's analysis)
would do the same.

Without ``sys.exit``, ``main()``'s return value was discarded and the process
always exited 0.  A build script checking ``$LASTEXITCODE`` would see success
no matter what happened.
"""

import sys

from gensrt.cli import main

if __name__ == "__main__":
    sys.exit(main())
