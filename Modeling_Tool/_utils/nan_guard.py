# encoding: utf-8
"""Shared NaN warning helpers."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def warn_if_nan_ratio_exceeds(values, threshold: float = 0.05, context: str = "") -> dict:
    """Warn when the NaN/Inf share in ``values`` exceeds ``threshold``.

    Returns a small stats dict so callers can surface the same information in
    reports without recomputing it.
    """
    arr = pd.to_numeric(pd.Series(values), errors="coerce")
    n_total = int(len(arr))
    n_nan = int((~np.isfinite(arr.to_numpy(dtype=float))).sum()) if n_total else 0
    ratio = n_nan / n_total if n_total else 0.0
    if n_total and ratio > float(threshold):
        label = f"{context}: " if context else ""
        warnings.warn(
            f"{label}{n_nan}/{n_total} ({ratio:.1%}) values are NaN/Inf.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {"n_total": n_total, "n_nan": n_nan, "nan_ratio": ratio}


__all__ = ["warn_if_nan_ratio_exceeds"]
