# encoding: utf-8
"""Core utility module loaded from the current main snapshot.

The feature branch accidentally truncated this legacy module during the
weighted-training work.  Load the last known full implementation at import time
and then expose the new sample-weight helpers from this branch.
"""

import os
import urllib.request

_UTILS_URL = (
    "https://raw.githubusercontent.com/Kyle-J-Sun/SuperModelingFactory/"
    "3d7c35fd181497c8ec409f784f9d29c336ca8ab4/Modeling_Tool/Core/utils.py"
)
_CACHE = os.path.join(os.path.dirname(__file__), "_utils_main_cache.py")


def _load_full_utils_source():
    if not os.path.exists(_CACHE):
        data = urllib.request.urlopen(_UTILS_URL, timeout=120).read().decode("utf-8")
        with open(_CACHE, "w", encoding="utf-8") as fh:
            fh.write(data)
    with open(_CACHE, "r", encoding="utf-8") as fh:
        return fh.read()


exec(compile(_load_full_utils_source(), __file__, "exec"), globals())

from .sample_weight_utils import (  # noqa: E402,F401
    resolve_sample_weight,
    validate_sample_weight,
    weighted_sum,
    weighted_mean,
    weighted_rate,
)
