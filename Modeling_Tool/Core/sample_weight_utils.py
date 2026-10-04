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

    Parameters
    ----------
    data : pandas.DataFrame or None, default None
        Frame that holds the weight column; required when ``weight_col`` (or ``wgt_col``) is given.
    weight_col : str or None, default None
        Name of the weight column in ``data``.
    sample_weight : array-like or None, default None
        Weights given directly, one per row.
    expected_len : int or None, default None
        Required length of the weight vector; None skips the length check.
    wgt : array-like or None, default None
        Alias of ``sample_weight``, used only when ``sample_weight`` is None.
    wgt_col : str or None, default None
        Alias of ``weight_col``, used only when ``weight_col`` is None.

    Returns
    -------
    numpy.ndarray or None
        The validated 1-D float weight vector (see ``validate_sample_weight``), or None when neither a column nor an array
        was given.

    Raises
    ------
    ValueError
        If both a column and an array are given, if a column is given without ``data``, or if the weights are invalid.
    KeyError
        If the weight column is not in ``data``.
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
    """Validate and return a 1-D float numpy weight vector.

    Parameters
    ----------
    weights : array-like
        Candidate weights, converted with ``numpy.asarray(weights, dtype=float)``.
    expected_len : int or None, default None
        Required length; None skips the length check.

    Returns
    -------
    numpy.ndarray
        The weights as a 1-D float array.

    Raises
    ------
    ValueError
        If the array is not 1-dimensional, has a length other than ``expected_len``, contains NaN or infinity or a
        negative value, or sums to zero or less.
    """
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
    total = float(np.sum(w))
    if total <= 0:
        raise ValueError(
            "sample_weight sum must be > 0; got sum={0}. "
            "All-zero weights make weighted metrics undefined.".format(total)
        )
    return w


def weighted_sum(values, weights):
    """Weighted sum of values.

    Parameters
    ----------
    values : array-like
        Values to add up.
    weights : array-like
        Weights, one per value; no validation is done.

    Returns
    -------
    float
        ``sum(values * weights)``.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.sum(v * w))


def weighted_mean(values, weights):
    """Weighted mean; returns NaN when total weight is zero.

    Parameters
    ----------
    values : array-like
        Values to average.
    weights : array-like
        Weights, one per value; no validation is done.

    Returns
    -------
    float
        ``sum(values * weights) / sum(weights)``, or NaN when the weights sum to zero.
    """
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total == 0:
        return np.nan
    return weighted_sum(values, weights) / total


def weighted_rate(mask, weights):
    """Weighted rate of True / 1 values in mask.

    Parameters
    ----------
    mask : array-like
        Boolean (or 0/1) flags, one per row.
    weights : array-like
        Weights, one per row; no validation is done.

    Returns
    -------
    float
        The share of the total weight that falls on flagged rows; NaN when the weights sum to zero.
    """
    m = np.asarray(mask, dtype=float)
    return weighted_mean(m, weights)
