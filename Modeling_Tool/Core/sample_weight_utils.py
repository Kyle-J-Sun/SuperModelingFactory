# encoding: utf-8
"""Sample-weight resolution and weighted aggregation helpers."""
from __future__ import annotations

import numpy as np


def resolve_sample_weight(
    data=None,
    weight_col=None,
    sample_weight=None,
    expected_len=None,
    wgt=None,
    wgt_col=None,
):
    """Resolve sample weights from a DataFrame column or array.

    Accepts ``weight_col`` / ``wgt_col`` and ``sample_weight`` / ``wgt`` aliases.
    Returns ``None`` when no weight source is provided.
    """
    if weight_col is None:
        weight_col = wgt_col
    if sample_weight is None:
        sample_weight = wgt

    if sample_weight is not None and weight_col is not None:
        raise ValueError("Provide either weight_col or sample_weight, not both.")

    if weight_col is not None:
        if data is None:
            raise ValueError("data is required when weight_col is provided.")
        if weight_col not in data.columns:
            raise KeyError("weight column '{0}' not found in data".format(weight_col))
        sample_weight = data[weight_col].values

    if sample_weight is None:
        return None

    return validate_sample_weight(sample_weight, expected_len=expected_len)


def validate_sample_weight(weights, expected_len=None):
    """Validate and return a 1-D float numpy weight vector."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("sample_weight must be 1-dimensional.")
    if expected_len is not None and len(w) != expected_len:
        raise ValueError(
            "sample_weight length {0} != expected {1}".format(len(w), expected_len)
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must be finite (no NaN/inf).")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative.")
    return w


def weighted_sum(values, weights):
    """Weighted sum of values."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.sum(v * w))


def weighted_mean(values, weights):
    """Weighted mean; returns NaN when total weight is zero."""
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total == 0:
        return np.nan
    return weighted_sum(values, weights) / total


def weighted_rate(mask, weights):
    """Weighted rate of True / 1 values in mask."""
    m = np.asarray(mask, dtype=float)
    return weighted_mean(m, weights)
