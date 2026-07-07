"""Sentinel constants shared across SMF modules."""

from __future__ import annotations

import numpy as np


SMF_MISSING_BIN = np.float64(np.finfo(np.float64).min)

__all__ = ["SMF_MISSING_BIN"]
