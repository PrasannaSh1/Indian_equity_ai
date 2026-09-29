"""IMPORTANT (Windows): torch must be imported before pandas/pyarrow anywhere
in a process, or its native DLL load can fail (see src/news/sentiment.py for
the full explanation). pytest runs every test module in one process, so this
constraint is process-wide, not per-file: relying on individual test files to
each import torch first only works if pytest happens to collect them before
any other pandas-importing test file, which is fragile (alphabetical luck,
broken by adding any earlier-sorting test file). Importing torch here, in the
root conftest.py, guarantees it happens before pytest collects any test
module, regardless of file naming or collection order.
"""

import torch  # noqa: F401
