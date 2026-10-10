# encoding: utf-8
"""Native weighted evaluation helpers.

Shared implementation for public Eval APIs when ``sample_weight`` or
``weight_col`` is supplied. Unweighted callers keep using the historical
implementations in ``evaluate_model.py`` and ``Model_Eval_Tool.py``.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight
from Modeling_Tool._utils.robust import smf_logger


def resolve_weights(data=None, weight_col=None, sample_weight=None, expected_len=None, wgt=None, wgt_col=None):
    """Resolve sample weights from a DataFrame column or an array.

    Thin wrapper around ``Modeling_Tool.Core.sample_weight_utils.resolve_sample_weight`` that first maps the aliases
    ``wgt_col`` to ``weight_col`` and ``wgt`` to ``sample_weight``.

    Parameters
    ----------
    data : pandas.DataFrame or None, default None
        Frame that holds the weight column; required when ``weight_col`` (or ``wgt_col``) is given.
    weight_col : str or None, default None
        Name of the weight column in ``data``.
    sample_weight : array-like or None, default None
        Weights given directly, one per row. It cannot be combined with ``weight_col``.
    expected_len : int or None, default None
        Required length of the weight vector; None skips the length check.
    wgt : array-like or None, default None
        Alias of ``sample_weight``, used only when ``sample_weight`` is None.
    wgt_col : str or None, default None
        Alias of ``weight_col``, used only when ``weight_col`` is None.

    Returns
    -------
    numpy.ndarray or None
        The validated 1-D float weight vector, or None when neither a column nor an array was given.

    Raises
    ------
    ValueError
        If both a column and an array are given, if a column is given without ``data``, or if the weights are not
        1-dimensional, have a length other than ``expected_len``, contain NaN, infinity or a negative value, or sum to
        zero or less.
    KeyError
        If the weight column is not in ``data``.
    """
    if weight_col is None:
        weight_col = wgt_col
    if sample_weight is None:
        sample_weight = wgt
    return resolve_sample_weight(
        data=data,
        weight_col=weight_col,
        sample_weight=sample_weight,
        expected_len=expected_len,
    )


def safe_weighted_average(values, weights=None):
    """Weighted mean that returns NaN instead of failing on empty or zero-weight input.

    Parameters
    ----------
    values : array-like
        Values to average, converted with ``numpy.asarray(values, dtype=float)``. NaN values are not skipped: they make
        the result NaN.
    weights : array-like or None, default None
        Weights aligned with ``values``. None gives the plain unweighted mean.

    Returns
    -------
    float
        The weighted mean. NaN when ``weights`` is None and ``values`` is empty, or when the weights sum to zero.

    Notes
    -----
    The weights are not validated: negative weights are used as given, and a ``weights`` array whose length differs from
    ``values`` makes ``numpy.average`` raise ``TypeError``.
    """
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(values)) if len(values) else np.nan
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total == 0:
        return np.nan
    return float(np.average(values, weights=weights))


def safe_auc(y_true, y_score, sample_weight=None):
    """Area under the ROC curve, or NaN when it cannot be computed.

    Parameters
    ----------
    y_true : array-like
        Actual binary labels (0/1).
    y_score : array-like
        Predicted scores or probabilities.
    sample_weight : array-like or None, default None
        Per-sample weights passed to ``sklearn.metrics.roc_auc_score``; None means equal weights.

    Returns
    -------
    float
        The (weighted) AUC. NaN when scikit-learn raises ``ValueError`` or ``ZeroDivisionError``, for example when
        ``y_true`` holds a single class or ``y_score`` contains NaN. The error is recorded with
        ``smf_logger.record_and_continue`` (stage ``weighted_eval.safe_auc``) and not re-raised.
    """
    try:
        return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))
    except (ValueError, ZeroDivisionError) as exc:
        smf_logger.record_and_continue("auc", exc, stage="weighted_eval.safe_auc")
        return np.nan


def calc_roc(y_true, y_score, sample_weight=None):
    """Compute the statistics of the ROC curve, optionally with sample weights.

    Based on ``sklearn.metrics.roc_curve``. The rows where ``y_true``, ``y_score`` or ``sample_weight`` is NaN or infinite
    are dropped first.

    Parameters
    ----------
    y_true : array-like
        Actual binary labels (0/1), converted to float.
    y_score : array-like
        Predicted scores or probabilities, converted to float.
    sample_weight : array-like or None, default None
        Per-sample weights aligned with ``y_true``, used for the curve and for ``thresholds_percentile``. They are not
        validated by this function. None means equal weights.

    Returns
    -------
    pandas.DataFrame
        One row per ROC threshold with the columns ``fpr``, ``tpr``, ``thresholds``, ``thresholds_percentile`` (percentage of
        the rows, or of the total weight, whose score is at or below the threshold), ``FPR`` and ``TPR`` (copies of ``fpr``
        and ``tpr``) and ``KS`` (``abs(tpr - fpr)``). An empty DataFrame with the same columns when no row is left after
        the filtering.

    Raises
    ------
    ValueError
        If ``y_true`` is not binary (raised by scikit-learn).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    if weight is not None:
        mask = mask & np.isfinite(weight)
        weight = weight[mask]
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0:
        return pd.DataFrame(columns=["fpr", "tpr", "thresholds", "thresholds_percentile", "FPR", "TPR", "KS"])

    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=weight)
    out = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
    if weight is None:
        out["thresholds_percentile"] = [100 * np.mean(y_score <= x) for x in thresholds]
    else:
        total_weight = float(weight.sum()) or 1.0
        out["thresholds_percentile"] = [
            100 * float(weight[y_score <= x].sum()) / total_weight for x in thresholds
        ]
    out["FPR"] = out["fpr"]
    out["TPR"] = out["tpr"]
    out["KS"] = (out["tpr"] - out["fpr"]).abs()
    return out


def calc_pr(y_true, y_score, sample_weight=None):
    """Compute the statistics of the P-R curve, optionally with sample weights.

    Based on ``sklearn.metrics.precision_recall_curve``.

    Parameters
    ----------
    y_true : array-like
        Actual binary labels (0/1).
    y_score : array-like
        Predicted scores or probabilities.
    sample_weight : array-like or None, default None
        Per-sample weights aligned with ``y_true``; None means equal weights. They are not validated by this function.

    Returns
    -------
    pandas.DataFrame
        One row per threshold with the columns ``precision``, ``recall`` and ``thresholds``. The last row (precision 1,
        recall 0) has no threshold (NaN). Unlike the unweighted ``evaluate_model.calc_pr`` there is no
        ``thresholds_percentile`` column.

    Notes
    -----
    NaN or infinite values are not filtered out here, so scikit-learn raises ``ValueError`` when ``y_true`` or ``y_score``
    contains them.
    """
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_score,
        sample_weight=sample_weight,
    )
    thresholds = np.r_[thresholds, np.nan]
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }
    )


def rank_bins(score, weight, nbins, ascending=False):
    """Weighted equal-frequency rank bins.

    ``ascending=False`` (default, legacy) sorts scores descending — bin 1 is
    the highest-score bucket. ``ascending=True`` sorts scores ascending so
    bin 1 is the lowest-score bucket, matching the unweighted
    ``gains_ascending=True`` convention.

    Parameters
    ----------
    score : array-like
        Scores, converted to float. A NaN score is ranked after every finite score, whatever ``ascending`` is, so it falls
        in the highest-numbered bin(s).
    weight : array-like or None
        Weights aligned with ``score`` and used as given (not validated); None gives every row the weight 1. There is no
        default: the argument must be passed.
    nbins : int
        Number of bins (converted with ``int``).
    ascending : bool, default False
        False ranks the highest score first (bin 1 holds the highest scores); True ranks the lowest score first.

    Returns
    -------
    numpy.ndarray
        Integer bin labels from 1 to ``nbins``, in the order of the input rows. A row gets the label
        ``ceil(cumulative weight / total weight * nbins)`` (clipped to ``[1, nbins]``) computed in rank order, so every bin
        holds about ``1 / nbins`` of the total weight.

    Notes
    -----
    Rows with the same score are ordered by their position in the input, so they can be split across two adjacent bins.
    """
    score = np.asarray(score, dtype=float)
    weight = np.ones(len(score), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    sort_key = score if ascending else -score
    order = np.lexsort((np.arange(len(score)), sort_key))
    bins = np.empty(len(score), dtype=int)
    total = float(weight.sum()) or float(len(score)) or 1.0
    cum_weight = np.cumsum(weight[order])
    labels = np.ceil(cum_weight / total * int(nbins)).astype(int)
    labels = np.clip(labels, 1, int(nbins))
    bins[order] = labels
    return bins


def _split_special_scores(df, score, spec_values):
    """Split a frame into (normal, special) rows by exact score membership.

    Special score values (e.g. the all-missing override ``-1``) are business
    sentinels, not model outputs: they must not participate in quantile
    binning or ranking metrics.
    """
    if not spec_values:
        return df, None
    spec_mask = df[score].isin(list(spec_values))
    if not bool(spec_mask.any()):
        return df, None
    return df[~spec_mask], df[spec_mask]


def get_gains_table(data, dep, score, nbins=10, weight_col=None, weighted_binning=None,
                    ascending=False, spec_values=None, include_missing=False, **kwargs):
    """Sample-weight-aware Gains table: equal-weight score bins with their target statistics.

    The rows are ranked by score (see ``rank_bins``) and cut into ``nbins`` bins that each hold about ``1 / nbins`` of the
    total weight. ``data`` is not modified.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset that holds the target, the score and, when given, the weight column.
    dep : str
        Name of the target column; numeric, 1 for bad and 0 for good.
    score : str
        Name of the score column; converted to float.
    nbins : int, default 10
        Number of bins.
    weight_col : str or None, default None
        Name of the sample weight column in ``data``. None gives every row the weight 1, so ``N`` is a row count.
    weighted_binning : bool or None, default None
        Accepted for compatibility but without any effect: the bins are always equal-weight bins.
    ascending : bool, default False
        False puts the highest scores in bin 1; True puts the lowest scores in bin 1.
    spec_values : list or None, default None
        Special score values (business sentinels such as ``-1``). The rows holding one of them are taken out before the
        binning and reported in rows of their own (see Notes). None or an empty list means no special values.
    include_missing : bool, default False
        Rows whose score is missing never enter the bins. False leaves them out of the table; True reports them in a
        row of their own (see Notes).
    **kwargs
        Accepted and ignored.

    Returns
    -------
    pandas.DataFrame
        One row per bin, indexed by ``_bin_num`` and ``_bin_range`` (a MultiIndex whose two levels both hold the bin number
        1 to ``nbins``), with the columns:

        - ``MIN``, ``MAX``: lowest and highest score in the bin.
        - ``N``: sum of the weights in the bin; ``N_RAW``: number of rows; ``PERF_CNT``: sum of the weights of the rows
          whose target is observed (rows with a missing target count in ``N`` only, as in the unweighted table).
        - ``N_BAD``, ``N_GOOD``: weighted counts of bad (``weight * target``) and good (``weight * (1 - target)``) rows
          among the rows with an observed target.
        - ``AVG_SCORE``: weighted mean score of the bin (NaN when the bin has NaN scores); ``UNIQUE_SCORE``: number of
          distinct scores.
        - ``PROP``: ``N`` divided by the total weight of the rows in the table, special (and reported missing) scores
          included, so ``PROP`` adds up to 1; ``AVG_BAD`` and ``AVG_GOOD``: ``N_BAD`` and ``N_GOOD`` divided by
          ``PERF_CNT``.
        - ``BAD_PCT_IN_EACH_BIN``, ``GOOD_PCT_IN_EACH_BIN``: share of all bad and all good weight that falls in the bin.
        - ``N_CUM_BAD``, ``N_CUM_GOOD``, ``CUM_BAD_PCT``, ``CUM_GOOD_PCT``: cumulative sums, from bin 1 down, of ``N_BAD``,
          ``N_GOOD``, ``BAD_PCT_IN_EACH_BIN`` and ``GOOD_PCT_IN_EACH_BIN``.
        - ``KS_PER_BIN``: ``abs(CUM_BAD_PCT - CUM_GOOD_PCT)``; ``KS`` is a copy of it.
        - ``LIFT``: ``AVG_BAD`` divided by the overall bad rate (``N_BAD`` over ``PERF_CNT`` of all bins).
        - ``TRUE_BAD_SHIFT``: relative change of ``AVG_BAD`` from the previous bin, ``previous / current - 1`` when
          ``ascending`` is False and ``current / previous - 1`` when it is True (NaN in bin 1).
        - ``RANK_ORDER_BUMP``: 1 when ``TRUE_BAD_SHIFT`` is negative (the bad rate is not monotonic), else 0.
        - ``WOE``: ``ln(BAD_PCT_IN_EACH_BIN / GOOD_PCT_IN_EACH_BIN)``, with 0 where it is infinite or undefined; ``IV``:
          ``(BAD_PCT_IN_EACH_BIN - GOOD_PCT_IN_EACH_BIN) * WOE``.
        - ``AUC``: weighted AUC of the score on the non-special rows with an observed target (the same value on every
          row), NaN when it cannot be computed.

    Raises
    ------
    KeyError
        If ``dep``, ``score`` or ``weight_col`` is not a column of ``data``.
    ValueError
        If the weights contain NaN, infinity or a negative value, or sum to zero or less.

    Notes
    -----
    The statistics are computed on the rows that do not hold a special score. Each special score value gets one more row
    after the bins, indexed by ``"special:<value>"`` in both index levels (in ascending order of the value), with the
    columns ``MIN``, ``MAX`` (both the value), ``N``, ``N_RAW``, ``PERF_CNT``, ``N_BAD``, ``N_GOOD``, ``AVG_SCORE``,
    ``UNIQUE_SCORE`` (1), ``PROP``, ``AVG_BAD``, ``AVG_GOOD`` and ``AUC`` (the AUC of the regular rows); every other column
    is NaN. ``PROP`` is the share of the weight of all rows in the regular bins and in the special rows alike, so the
    column adds up to 1. The other shares (``BAD_PCT_IN_EACH_BIN`` and ``GOOD_PCT_IN_EACH_BIN``), ``LIFT`` and the overall
    bad rate still refer to the rows without a special score.

    Rows with a missing score carry no ranking information and never enter the bins. With ``include_missing=True`` they
    get one row after the bins (and after the special rows), indexed by ``"Missing"`` in both index levels, with ``N``,
    ``N_RAW``, ``PERF_CNT``, ``N_BAD``, ``N_GOOD``, ``UNIQUE_SCORE`` (0), ``PROP``, ``AVG_BAD``, ``AVG_GOOD`` and ``AUC``;
    every other column is NaN. Rows with the same score can be split across two adjacent bins.
    """
    cols = [dep, score]
    if weight_col is not None and weight_col in data.columns:
        cols.append(weight_col)
    df = data[cols].copy()
    # A missing score has no rank: it would be sorted to the end of the ranking and fill the last bins
    score_missing = df[score].isna()
    missing_df = df[score_missing] if bool(score_missing.any()) else None
    df = df[~score_missing]
    df, special_df = _split_special_scores(df, score, spec_values)
    weight = resolve_weights(df, weight_col=weight_col, expected_len=len(df))
    if weight is None:
        weight = np.ones(len(df), dtype=float)

    y = df[dep].astype(float).to_numpy()
    labelled = ~np.isnan(y)
    y_obs = np.where(labelled, y, 0.0)
    s = df[score].astype(float).to_numpy()
    df["_bin_num"] = rank_bins(s, weight, nbins, ascending=ascending)
    df["_bin_range"] = df["_bin_num"]
    df["_w"] = weight
    # Rows with a missing target are in N but in neither the bad nor the good counts
    df["_perf_w"] = weight * labelled
    df["_bad_w"] = weight * y_obs
    df["_good_w"] = weight * labelled * (1.0 - y_obs)
    df["_score_w"] = weight * s
    grouped = df.groupby(["_bin_num", "_bin_range"], sort=True, dropna=False)
    out = grouped.agg(
        MIN=(score, "min"),
        MAX=(score, "max"),
        N=("_w", "sum"),
        N_RAW=(dep, "size"),
        PERF_CNT=("_perf_w", "sum"),
        N_BAD=("_bad_w", "sum"),
        N_GOOD=("_good_w", "sum"),
        SCORE_W=("_score_w", "sum"),
        SCORE_COUNT=(score, "count"),
        UNIQUE_SCORE=(score, "nunique"),
    )
    out["AVG_SCORE"] = out["SCORE_W"] / out["N"].replace(0, np.nan)
    out.loc[out["SCORE_COUNT"].ne(out["N_RAW"]), "AVG_SCORE"] = np.nan
    out = out.drop(columns=["SCORE_W", "SCORE_COUNT"])
    out = out[
        [
            "MIN", "MAX", "N", "N_RAW", "PERF_CNT", "N_BAD",
            "N_GOOD", "AVG_SCORE", "UNIQUE_SCORE",
        ]
    ]

    total_perf = float(out["PERF_CNT"].sum())
    total_bad = float(out["N_BAD"].sum()) or 1.0
    total_good = float(out["N_GOOD"].sum()) or 1.0
    overall_bad_rate = float(out["N_BAD"].sum()) / total_perf if total_perf else np.nan

    # PROP is the share of the weight of ALL rows in the table, special (and reported missing) ones included,
    # so the bins and the extra rows add up to 1.
    grand_total = float(out["N"].sum())
    if special_df is not None and len(special_df):
        spec_weight = resolve_weights(special_df, weight_col=weight_col, expected_len=len(special_df))
        if spec_weight is None:
            spec_weight = np.ones(len(special_df), dtype=float)
        grand_total += float(np.sum(spec_weight))
    report_missing = include_missing and missing_df is not None
    if report_missing:
        miss_weight = resolve_weights(missing_df, weight_col=weight_col, expected_len=len(missing_df))
        if miss_weight is None:
            miss_weight = np.ones(len(missing_df), dtype=float)
        grand_total += float(np.sum(miss_weight))
    out["PROP"] = out["N"] / grand_total if grand_total else np.nan
    out["AVG_BAD"] = out["N_BAD"] / out["PERF_CNT"].replace(0, np.nan)
    out["AVG_GOOD"] = out["N_GOOD"] / out["PERF_CNT"].replace(0, np.nan)
    out["BAD_PCT_IN_EACH_BIN"] = out["N_BAD"] / total_bad
    out["GOOD_PCT_IN_EACH_BIN"] = out["N_GOOD"] / total_good
    out["N_CUM_BAD"] = out["N_BAD"].cumsum()
    out["N_CUM_GOOD"] = out["N_GOOD"].cumsum()
    out["CUM_BAD_PCT"] = out["BAD_PCT_IN_EACH_BIN"].cumsum()
    out["CUM_GOOD_PCT"] = out["GOOD_PCT_IN_EACH_BIN"].cumsum()
    out["KS_PER_BIN"] = (out["CUM_BAD_PCT"] - out["CUM_GOOD_PCT"]).abs()
    out["KS"] = out["KS_PER_BIN"]
    out["LIFT"] = out["AVG_BAD"] / overall_bad_rate if overall_bad_rate else np.nan
    out["TRUE_BAD_SHIFT"] = (
        (out["AVG_BAD"].shift(1) / out["AVG_BAD"] - 1)
        if not ascending
        else (out["AVG_BAD"] / out["AVG_BAD"].shift(1) - 1)
    )
    out["RANK_ORDER_BUMP"] = out["TRUE_BAD_SHIFT"].lt(0).astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["WOE"] = np.log(out["BAD_PCT_IN_EACH_BIN"] / out["GOOD_PCT_IN_EACH_BIN"])
    out["WOE"] = out["WOE"].replace([np.inf, -np.inf], 0).fillna(0)
    out["IV"] = (out["BAD_PCT_IN_EACH_BIN"] - out["GOOD_PCT_IN_EACH_BIN"]) * out["WOE"]
    out["AUC"] = safe_auc(y[labelled], s[labelled], sample_weight=weight[labelled])

    def _extra_row(label, part, part_w, value):
        part_y = part[dep].astype(float).to_numpy()
        part_obs = ~np.isnan(part_y)
        n_w = float(np.sum(part_w))
        perf_w = float(np.sum(part_w[part_obs]))
        n_bad = float(np.sum(part_w[part_obs] * part_y[part_obs]))
        return pd.DataFrame(
            {
                "MIN": [value],
                "MAX": [value],
                "N": [n_w],
                "N_RAW": [int(len(part))],
                "PERF_CNT": [perf_w],
                "N_BAD": [n_bad],
                "N_GOOD": [perf_w - n_bad],
                "AVG_SCORE": [float(value)],
                "UNIQUE_SCORE": [1 if pd.notna(value) else 0],
                "PROP": [n_w / grand_total if grand_total else np.nan],
                "AVG_BAD": [n_bad / perf_w if perf_w else np.nan],
                "AVG_GOOD": [(perf_w - n_bad) / perf_w if perf_w else np.nan],
            },
            index=pd.MultiIndex.from_tuples([(label, label)], names=out.index.names),
        )

    extra_rows = []
    if special_df is not None and len(special_df):
        # Special sentinel scores (e.g. -1 for all-missing rows) get their own
        # descriptive rows: never part of quantile edges, cumulative columns,
        # or ranking metrics (those stay NaN by construction).
        for value, part in special_df.groupby(score, sort=True):
            part_w = resolve_weights(part, weight_col=weight_col, expected_len=len(part))
            if part_w is None:
                part_w = np.ones(len(part), dtype=float)
            extra_rows.append(_extra_row(f"special:{value}", part, part_w, value))
    if report_missing:
        extra_rows.append(_extra_row("Missing", missing_df, miss_weight, np.nan))
    if extra_rows:
        out = pd.concat([out] + extra_rows)
        out["AUC"] = out["AUC"].iloc[0]
    return out


def calc_lift_apt(y_true, y_score, start=1.0, stop=3.0, step=0.1, sample_weight=None):
    """Closest available Lift values for a grid of target Lifts, from a weighted 100-bin Gains table.

    Parameters
    ----------
    y_true : array-like
        Actual binary labels (0/1), converted to float.
    y_score : array-like
        Predicted scores or probabilities, converted to float.
    start : float, default 1.0
        First target Lift.
    stop : float, default 3.0
        Last target Lift (included up to rounding of the grid).
    step : float, default 0.1
        Distance between two target Lifts.
    sample_weight : array-like or None, default None
        Per-sample weights aligned with ``y_true``; None means equal weights. They are validated like the weights of
        ``get_gains_table``.

    Returns
    -------
    numpy.ndarray
        One-dimensional float array with one value per target Lift of ``np.arange(start, stop + step / 2, step)``: the
        ``LIFT`` of the bin of ``get_gains_table(..., nbins=100)`` (highest scores first) that is closest to that target.

    Raises
    ------
    ValueError
        If the weights are invalid (see ``get_gains_table``).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    weight = np.ones(len(y_true), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    gains = get_gains_table(
        pd.DataFrame({"y": y_true, "s": y_score, "w": weight}),
        "y",
        "s",
        nbins=100,
        weight_col="w",
    )
    vals = []
    for target in np.arange(start, stop + step / 2.0, step):
        idx = (gains["LIFT"] - target).abs().idxmin()
        vals.append(float(gains.loc[idx, "LIFT"]))
    return np.asarray(vals)


def calc_equid_dist(y_true, y_score, bins=10, sample_weight=None, **kwargs):
    """Weighted Gains table over ``bins`` equal-weight score bins.

    Despite its name the bins are not equal-width: the weighted implementation cuts the ranked scores into bins that each
    hold about ``1 / bins`` of the total weight. It is the weighted counterpart of ``evaluate_model.calc_equid_dist``.

    Parameters
    ----------
    y_true : array-like
        Actual binary labels (0/1).
    y_score : array-like
        Predicted scores or probabilities, same length as ``y_true``.
    bins : int, default 10
        Number of bins.
    sample_weight : array-like or None, default None
        Per-sample weights aligned with ``y_true``; None gives every row the weight 1. They are validated like the weights
        of ``get_gains_table``.
    **kwargs
        Forwarded to ``get_gains_table``: ``ascending``, ``spec_values`` and ``include_missing`` take effect, other names are ignored.
        ``nbins`` and ``weight_col`` must not be passed (``TypeError``: they are already set).

    Returns
    -------
    pandas.DataFrame
        The Gains table of ``get_gains_table``: ``bins`` rows indexed by ``_bin_num`` and ``_bin_range`` (see
        ``get_gains_table`` for the columns).

    Raises
    ------
    ValueError
        If the weights are invalid (see ``get_gains_table``).
    """
    weight = np.ones(len(y_true), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return get_gains_table(
        pd.DataFrame({"y": y_true, "s": y_score, "w": weight}),
        "y",
        "s",
        nbins=bins,
        weight_col="w",
        **kwargs,
    )


def calc_equid_pct(y_true, y_score, bins=10, sample_weight=None, **kwargs):
    """Weighted Gains table over ``bins`` equal-weight score bins.

    Same as ``calc_equid_dist``, to which it delegates: the weighted counterpart of ``evaluate_model.calc_equid_pct``.

    Parameters
    ----------
    y_true : array-like
        Actual binary labels (0/1).
    y_score : array-like
        Predicted scores or probabilities, same length as ``y_true``.
    bins : int, default 10
        Number of bins.
    sample_weight : array-like or None, default None
        Per-sample weights aligned with ``y_true``; None gives every row the weight 1.
    **kwargs
        Forwarded to ``calc_equid_dist`` and then to ``get_gains_table``: ``ascending``, ``spec_values`` and
        ``include_missing`` take effect, other names are ignored.

    Returns
    -------
    pandas.DataFrame
        The Gains table of ``get_gains_table``: ``bins`` rows indexed by ``_bin_num`` and ``_bin_range`` (see
        ``get_gains_table`` for the columns).

    Raises
    ------
    ValueError
        If the weights are invalid (see ``get_gains_table``).
    """
    return calc_equid_dist(y_true, y_score, bins=bins, sample_weight=sample_weight, **kwargs)


def calc_fixed_pct(y_true, y_score, sample_weight=None, **kwargs):
    """Weighted Gains table over equal-weight score bins; the weighted counterpart of ``evaluate_model.calc_fixed_pct``.

    Fixed bin edges are not supported here: the call delegates to ``calc_equid_dist``, so the table has 10 equal-weight
    bins unless ``bins`` is passed in ``kwargs``.

    Parameters
    ----------
    y_true : array-like
        Actual binary labels (0/1).
    y_score : array-like
        Predicted scores or probabilities, same length as ``y_true``.
    sample_weight : array-like or None, default None
        Per-sample weights aligned with ``y_true``; None gives every row the weight 1.
    **kwargs
        Forwarded to ``calc_equid_dist``: ``bins`` (number of bins, default 10), ``ascending`` and ``spec_values`` take
        effect; other names (such as ``bin_edges``) are ignored.

    Returns
    -------
    pandas.DataFrame
        The Gains table of ``get_gains_table``: ``bins`` rows indexed by ``_bin_num`` and ``_bin_range`` (see
        ``get_gains_table`` for the columns).

    Raises
    ------
    ValueError
        If the weights are invalid (see ``get_gains_table``).
    """
    return calc_equid_dist(y_true, y_score, sample_weight=sample_weight, **kwargs)


def dataset_summary(name, data, tgt_name, scr_name, weight_col=None, nbins=10,
                    ascending=False, spec_values=None):
    """Performance summary of one dataset: AUC, KS, Lift, IV and the average target and score.

    Parameters
    ----------
    name : str
        Label of the dataset; stored in the ``index``, ``dataset`` and ``DATASET`` entries.
    data : pandas.DataFrame
        Dataset that holds the target, the score and, when given, the weight column.
    tgt_name : str
        Name of the target column (binary, 1 for bad).
    scr_name : str
        Name of the score column.
    weight_col : str or None, default None
        Name of the sample weight column in ``data``; None gives every row the weight 1.
    nbins : int, default 10
        Number of bins of the Gains table behind ``LIFT`` and ``IV``.
    ascending : bool, default False
        Bin order of that Gains table (see ``get_gains_table``).
    spec_values : list or None, default None
        Special score values (such as ``-1``) that are left out of the ranking statistics and reported separately.

    Returns
    -------
    dict
        The entries ``index``, ``dataset`` and ``DATASET`` (all equal to ``name``), ``AUC``, ``KS`` (maximum of
        ``abs(tpr - fpr)`` on the ROC curve), ``LIFT`` (largest bin Lift), ``IV`` (sum over the bins), ``N`` (sum of the
        weights of all rows, or the number of rows without weights), ``N_RAW`` (number of rows), ``avgTrue`` (weighted mean
        of the target over all rows) and ``avgScore`` (weighted mean of the score over the non-special rows). ``N_SPECIAL``
        (sum of the weights) and ``N_SPECIAL_RAW`` (number of rows) are added when at least one row holds a special score.

    Raises
    ------
    KeyError
        If ``tgt_name``, ``scr_name`` or ``weight_col`` is not a column of ``data``.
    ValueError
        If the weights contain NaN, infinity or a negative value, or sum to zero or less.

    Notes
    -----
    ``AUC``, ``KS``, ``LIFT``, ``IV`` and ``avgScore`` use the non-special rows only, because sentinel scores carry no
    ordering information; ``avgTrue``, ``N`` and ``N_RAW`` cover all rows. ``AUC`` and ``KS`` are NaN when they cannot be
    computed (for example with a single class).
    """
    weight = resolve_weights(data, weight_col=weight_col, expected_len=len(data))
    y_true = data[tgt_name].to_numpy()
    y_score = data[scr_name].to_numpy()
    # Ranking metrics (AUC/KS/avgScore) are computed on non-special rows only:
    # sentinel scores like -1 carry no ordering information.
    rank_data, special_part = _split_special_scores(data, scr_name, spec_values)
    rank_weight = resolve_weights(rank_data, weight_col=weight_col, expected_len=len(rank_data))
    rank_true = rank_data[tgt_name].to_numpy()
    rank_score = rank_data[scr_name].to_numpy()
    roc_df = calc_roc(rank_true, rank_score, sample_weight=rank_weight)
    gains = get_gains_table(
        data, tgt_name, scr_name, nbins=nbins, weight_col=weight_col,
        ascending=ascending, spec_values=spec_values,
    )
    summary = {
        "index": name,
        "dataset": name,
        "DATASET": name,
        "AUC": safe_auc(rank_true, rank_score, sample_weight=rank_weight),
        "KS": float(roc_df["KS"].max()) if "KS" in roc_df else np.nan,
        "LIFT": float(gains["LIFT"].max()) if "LIFT" in gains else np.nan,
        "IV": float(gains["IV"].sum()) if "IV" in gains else np.nan,
        "N": float(np.sum(weight)) if weight is not None else float(len(data)),
        "N_RAW": int(len(data)),
        "avgTrue": safe_weighted_average(y_true, weight),
        "avgScore": safe_weighted_average(rank_score, rank_weight),
    }
    if special_part is not None:
        spec_weight = resolve_weights(special_part, weight_col=weight_col, expected_len=len(special_part))
        summary["N_SPECIAL"] = (
            float(np.sum(spec_weight)) if spec_weight is not None else float(len(special_part))
        )
        summary["N_SPECIAL_RAW"] = int(len(special_part))
    return summary


def get_perf_summary(train=None, validation=None, oot=None, tgt_name=None, scr_name=None, weight_col=None, nbins=10,
                     ascending=False, spec_values=None, **kwargs):
    """Performance summary of the train, validation and OOT datasets, one row each.

    Parameters
    ----------
    train : pandas.DataFrame or None, default None
        Training dataset, reported in the row ``ins``; skipped when None.
    validation : pandas.DataFrame or None, default None
        Validation dataset, reported in the row ``oos``; skipped when None.
    oot : pandas.DataFrame or None, default None
        Out-of-time dataset, reported in the row ``oot``; skipped when None.
    tgt_name : str or None, default None
        Name of the target column (binary, 1 for bad); required as soon as a dataset is given.
    scr_name : str or None, default None
        Name of the score column; required as soon as a dataset is given.
    weight_col : str or None, default None
        Name of the sample weight column, which must exist in every given dataset; None gives every row the weight 1.
    nbins : int, default 10
        Number of bins of the Gains table behind ``LIFT`` and ``IV``.
    ascending : bool, default False
        Bin order of that Gains table (see ``get_gains_table``).
    spec_values : list or None, default None
        Special score values (such as ``-1``) that are left out of the ranking statistics (see ``dataset_summary``).
    **kwargs
        Accepted and ignored.

    Returns
    -------
    pandas.DataFrame
        One row per given dataset, in the order ``ins``, ``oos``, ``oot``, with the entries of ``dataset_summary`` as
        columns (``index``, ``dataset``, ``DATASET``, ``AUC``, ``KS``, ``LIFT``, ``IV``, ``N``, ``N_RAW``, ``avgTrue``,
        ``avgScore`` and, when special scores occur, ``N_SPECIAL`` and ``N_SPECIAL_RAW``). A DataFrame without rows or
        columns when no dataset is given.

    Raises
    ------
    KeyError
        If a target, score or weight column is missing from a given dataset.
    ValueError
        If the weights are invalid (see ``dataset_summary``).
    """
    rows = []
    for name, data in (("ins", train), ("oos", validation), ("oot", oot)):
        if data is not None:
            rows.append(dataset_summary(
                name, data, tgt_name, scr_name, weight_col=weight_col, nbins=nbins,
                ascending=ascending, spec_values=spec_values,
            ))
    return pd.DataFrame(rows)


def evaluate_performance(datasets=None, tgt_name=None, scr_name=None, sample_weight=None, nbins=10, **kwargs):
    """Performance summary of several named datasets given as DataFrames or as label/score arrays.

    Parameters
    ----------
    datasets : dict or None, default None
        Mapping of dataset name to a payload dict. A payload is either ``{"data": DataFrame}`` (the frame must hold the
        ``tgt_name`` and ``scr_name`` columns; a payload whose ``data`` is None is skipped) or
        ``{"y_true": array-like, "y_score": array-like}``. Either form may carry a ``"sample_weight"`` entry that
        overrides the ``sample_weight`` argument for that dataset. None gives an empty result.
    tgt_name : str or None, default None
        Name of the target column (binary, 1 for bad); for array payloads, the name given to the target column of the frame
        built from them.
    scr_name : str or None, default None
        Name of the score column; for array payloads, the name given to the score column of the frame built from them.
    sample_weight : array-like or None, default None
        Default weights, one per row, for the datasets whose payload has no ``"sample_weight"`` entry. None gives every
        row the weight 1.
    nbins : int, default 10
        Number of bins of the Gains table behind ``LIFT`` and ``IV``.
    **kwargs
        Accepted and ignored.

    Returns
    -------
    pandas.DataFrame
        One row per dataset, in the order of ``datasets``, with the entries of ``dataset_summary`` as columns (``index``,
        ``dataset``, ``DATASET``, ``AUC``, ``KS``, ``LIFT``, ``IV``, ``N``, ``N_RAW``, ``avgTrue``, ``avgScore``). A
        DataFrame without rows or columns when ``datasets`` is None or empty.

    Raises
    ------
    KeyError
        If an array payload lacks ``"y_true"`` or ``"y_score"``, or a frame lacks the target or score column.
    ValueError
        If the weights are invalid: wrong length, NaN, infinity, negative values or a sum that is not positive.

    Notes
    -----
    Each frame is copied and given the weight column ``_w`` (overwriting any column of that name). ``ascending`` and
    ``spec_values`` are not available here: the Gains table behind ``LIFT`` and ``IV`` uses ``ascending=False`` and no
    special values.
    """
    rows = []
    if datasets is None:
        return pd.DataFrame(rows)
    for name, payload in datasets.items():
        if isinstance(payload, dict) and "data" in payload:
            data = payload.get("data")
            weight = payload.get("sample_weight", sample_weight)
        else:
            data = pd.DataFrame(
                {
                    tgt_name: payload["y_true"],
                    scr_name: payload["y_score"],
                }
            )
            weight = payload.get("sample_weight", sample_weight)
        if data is None:
            continue
        tmp = data.copy()
        tmp["_w"] = np.ones(len(tmp), dtype=float) if weight is None else weight
        rows.append(dataset_summary(name, tmp, tgt_name, scr_name, weight_col="_w", nbins=nbins))
    return pd.DataFrame(rows)


class GainsTableCalculator:
    """Gains-table calculator bound to one dataset.

    Stores the arguments of ``get_gains_table`` and computes the table with ``calculate``.

    Parameters
    ----------
    data : pandas.DataFrame or None, default None
        Dataset that holds the target, the score and, when given, the weight column. It must be set before ``calculate`` is
        called.
    dep : str or None, default None
        Name of the target column (binary, 1 for bad).
    score : str or None, default None
        Name of the score column.
    nbins : int, default 10
        Number of bins.
    weight_col : str or None, default None
        Name of the sample weight column in ``data``; None gives every row the weight 1.
    weighted_binning : bool or None, default None
        Accepted for compatibility but without any effect: the bins are always equal-weight bins.
    **kwargs
        Extra keyword arguments of ``get_gains_table`` (for example ``ascending`` or ``spec_values``), stored and passed on
        by ``calculate``. Nothing is validated before ``calculate`` is called.

    Attributes
    ----------
    data, dep, score, nbins, weight_col, weighted_binning : object
        The constructor arguments, as given.
    kwargs : dict
        The extra keyword arguments.
    """
    def __init__(self, data=None, dep=None, score=None, nbins=10, weight_col=None, weighted_binning=None, **kwargs):
        self.data = data
        self.dep = dep
        self.score = score
        self.nbins = nbins
        self.weight_col = weight_col
        self.weighted_binning = weighted_binning
        self.kwargs = kwargs

    def calculate(self, weight_col=None, **kwargs):
        """Compute the Gains table of the stored dataset with ``get_gains_table``.

        Parameters
        ----------
        weight_col : str or None, default None
            Weight column for this call; None keeps the ``weight_col`` given to the constructor.
        **kwargs
            Extra keyword arguments of ``get_gains_table`` for this call; they take precedence over the ones given to the
            constructor. ``nbins`` and ``weighted_binning`` must not be passed (``TypeError``: they are already set).

        Returns
        -------
        pandas.DataFrame
            The Gains table (see ``get_gains_table`` for the columns).

        Raises
        ------
        KeyError
            If the target, score or weight column is not a column of ``data``.
        ValueError
            If the weights are invalid (see ``get_gains_table``).
        """
        return get_gains_table(
            self.data,
            self.dep,
            self.score,
            nbins=self.nbins,
            weight_col=self.weight_col if weight_col is None else weight_col,
            weighted_binning=self.weighted_binning,
            **{**self.kwargs, **kwargs},
        )


class PerformanceEvaluator:
    """Collect named datasets and summarize their performance with ``dataset_summary``.

    Parameters
    ----------
    tgt_name : str or None, default None
        Name of the target column (binary, 1 for bad) in every dataset.
    scr_name : str or None, default None
        Name of the score column in every dataset.
    weight_col : str or None, default None
        Default sample weight column, used for the datasets added without one.
    nbins : int, default 10
        Number of bins of the Gains table behind ``LIFT`` and ``IV``.
    **kwargs
        Stored in ``self.kwargs`` but not used.

    Attributes
    ----------
    tgt_name, scr_name, weight_col, nbins : object
        The constructor arguments, as given.
    kwargs : dict
        The extra keyword arguments.
    datasets : collections.OrderedDict
        Added datasets in insertion order: name mapped to the tuple ``(data, weight_col)``.
    """
    def __init__(self, tgt_name=None, scr_name=None, weight_col=None, nbins=10, **kwargs):
        self.tgt_name = tgt_name
        self.scr_name = scr_name
        self.weight_col = weight_col
        self.nbins = nbins
        self.kwargs = kwargs
        self.datasets = OrderedDict()

    def add_dataset(self, name, data, weight_col=None, **kwargs):
        """Register a dataset to be summarized by ``evaluate``.

        Parameters
        ----------
        name : str
            Label of the dataset. Adding a name that already exists replaces that dataset and keeps its position.
        data : pandas.DataFrame
            Dataset that holds the target and score columns (and the weight column, when used).
        weight_col : str or None, default None
            Sample weight column of this dataset; None falls back to the ``weight_col`` of ``evaluate`` and then to the
            one of the constructor.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        PerformanceEvaluator
            The evaluator itself, so that calls can be chained.
        """
        self.datasets[name] = (data, weight_col)
        return self

    def evaluate(self, weight_col=None, to_show=False, display=False, **kwargs):
        """Summarize every added dataset.

        Parameters
        ----------
        weight_col : str or None, default None
            Weight column for the datasets added without their own; None falls back to the ``weight_col`` of the
            constructor (and to equal weights when that is None too).
        to_show : bool, default False
            Accepted for API compatibility; it has no effect.
        display : bool, default False
            Accepted for API compatibility; it has no effect.
        **kwargs
            Accepted and ignored.

        Returns
        -------
        pandas.DataFrame
            One row per added dataset, in insertion order, with the entries of ``dataset_summary`` as columns (``index``,
            ``dataset``, ``DATASET``, ``AUC``, ``KS``, ``LIFT``, ``IV``, ``N``, ``N_RAW``, ``avgTrue``, ``avgScore``). A
            DataFrame without rows or columns when no dataset was added.

        Raises
        ------
        KeyError
            If a target, score or weight column is missing from a dataset.
        ValueError
            If the weights of a dataset are invalid (see ``dataset_summary``).

        Notes
        -----
        The Gains table behind ``LIFT`` and ``IV`` uses ``nbins`` of the constructor, ``ascending=False`` and no special
        score values.
        """
        rows = []
        for name, (data, ds_weight_col) in self.datasets.items():
            wc = ds_weight_col or weight_col or self.weight_col
            rows.append(dataset_summary(name, data, self.tgt_name, self.scr_name, weight_col=wc, nbins=self.nbins))
        return pd.DataFrame(rows)


def cross_risk_weighted_mean(data, agg_col, sample_weight, score_list, margin_name="Total_Avg_Risk"):
    """Weighted-mean cross-risk table after bin columns are assigned.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset that already holds the bin columns ``_bin_num1``, ``_bin_range1`` (first score) and ``_bin_num2``,
        ``_bin_range2`` (second score), and the column ``agg_col``.
    agg_col : str
        Name of the column to average (for example the target or a risk measure); converted with
        ``pandas.to_numeric(errors="coerce")``, so non-numeric entries become NaN.
    sample_weight : array-like
        Weights, one per row of ``data`` and aligned with it by position; converted to float and not validated.
    score_list : list of str
        Names of the two scores, ``[first, second]``; used as the names of the row index levels (``first``) and of the
        column index levels (``second``).
    margin_name : str, default "Total_Avg_Risk"
        Label of the margin row and column, which hold the weighted means over a whole row, a whole column and the whole
        table.

    Returns
    -------
    pandas.DataFrame
        For every cell, the sum of ``weight * agg_col`` divided by the sum of the weights: the rows are indexed by
        (``_bin_num1``, ``_bin_range1``) and the columns by (``_bin_num2``, ``_bin_range2``), each with two levels named
        after ``score_list``. Cells without rows, whose weights sum to zero, or whose ``agg_col`` values are all NaN, are NaN.

    Notes
    -----
    A row whose ``agg_col`` value is NaN is skipped: it adds nothing to the numerator and its weight is left out of the
    denominator, so the result is the weighted mean of the values that exist.
    """
    weight = np.asarray(sample_weight, dtype=float)
    values = pd.to_numeric(data[agg_col], errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(values)
    frame = data[["_bin_num1", "_bin_range1", "_bin_num2", "_bin_range2"]].copy()
    frame["_w"] = np.where(valid, weight, 0.0)
    frame["_wv"] = np.where(valid, values * weight, 0.0)

    numerator = pd.crosstab(
        [frame["_bin_num1"], frame["_bin_range1"]],
        [frame["_bin_num2"], frame["_bin_range2"]],
        values=frame["_wv"],
        aggfunc="sum",
        margins=True,
        margins_name=margin_name,
        rownames=[score_list[0], score_list[0]],
        colnames=[score_list[1], score_list[1]],
    )
    denominator = pd.crosstab(
        [frame["_bin_num1"], frame["_bin_range1"]],
        [frame["_bin_num2"], frame["_bin_range2"]],
        values=frame["_w"],
        aggfunc="sum",
        margins=True,
        margins_name=margin_name,
        rownames=[score_list[0], score_list[0]],
        colnames=[score_list[1], score_list[1]],
    )
    return numerator / denominator.replace(0, np.nan)
