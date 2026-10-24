"""pytest: the downloaded reference folders are not part of the suite."""
collect_ignore = ["_ref", "_scratch", "testdata"]

import os
os.environ.setdefault("DS41_PFX_MIN_FREE_GIB", "0")      # tests must not depend on the host's free memory
