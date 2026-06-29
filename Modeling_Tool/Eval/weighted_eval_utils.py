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


def resolve_weights(data=None, weight_col=None, sample_weight=None, expected_len=None, wgt=None, wgt_col=None):
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
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(values)) if len(values) else np.nan
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total == 0:
        return np.nan
    return float(np.average(values, weights=weights))


def safe_auc(y_true, y_score, sample_weight=None):
    try:
        return float(roc_auc_score(y_true, y_score, sample_weight=sample_weight))
    except Exception:
        return np.nan


def calc_roc(y_true, y_score, sample_weight=None):
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


def rank_bins(score, weight, nbins):
    score = np.asarray(score, dtype=float)
    weight = np.ones(len(score), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    order = np.lexsort((np.arange(len(score)), -score))
    bins = np.empty(len(score), dtype=int)
    total = float(weight.sum()) or float(len(score)) or 1.0
    cum_weight = np.cumsum(weight[order])
    labels = np.ceil(cum_weight / total * int(nbins)).astype(int)
    labels = np.clip(labels, 1, int(nbins))
    bins[order] = labels
    return bins


def get_gains_table(data, dep, score, nbins=10, weight_col=None, weighted_binning=None, **kwargs):
    cols = [dep, score]
    if weight_col is not None and weight_col in data.columns:
        cols.append(weight_col)
    df = data[cols].copy()
    weight = resolve_weights(df, weight_col=weight_col, expected_len=len(df))
    if weight is None:
        weight = np.ones(len(df), dtype=float)

    y = df[dep].astype(float).to_numpy()
    s = df[score].astype(float).to_numpy()
    df["_bin_num"] = rank_bins(s, weight, nbins)
    df["_bin_range"] = df["_bin_num"]
    df["_w"] = weight
    df["_bad_w"] = weight * y
    df["_good_w"] = weight * (1.0 - y)
    grouped = df.groupby(["_bin_num", "_bin_range"], sort=True, dropna=False)
    out = grouped.agg(
        MIN=(score, "min"),
        MAX=(score, "max"),
        N=("_w", "sum"),
        N_RAW=(dep, "size"),
        PERF_CNT=("_w", "sum"),
        N_BAD=("_bad_w", "sum"),
        N_GOOD=("_good_w", "sum"),
        AVG_SCORE=(score, lambda x: safe_weighted_average(x, df.loc[x.index, "_w"])),
        UNIQUE_SCORE=(score, "nunique"),
    )

    total_weight = float(out["N"].sum()) or 1.0
    total_bad = float(out["N_BAD"].sum()) or 1.0
    total_good = float(out["N_GOOD"].sum()) or 1.0
    overall_bad_rate = float(out["N_BAD"].sum()) / total_weight if total_weight else np.nan

    out["PROP"] = out["N"] / total_weight
    out["AVG_BAD"] = out["N_BAD"] / out["N"].replace(0, np.nan)
    out["AVG_GOOD"] = out["N_GOOD"] / out["N"].replace(0, np.nan)
    out["BAD_PCT_IN_EACH_BIN"] = out["N_BAD"] / total_bad
    out["GOOD_PCT_IN_EACH_BIN"] = out["N_GOOD"] / total_good
    out["N_CUM_BAD"] = out["N_BAD"].cumsum()
    out["N_CUM_GOOD"] = out["N_GOOD"].cumsum()
    out["CUM_BAD_PCT"] = out["BAD_PCT_IN_EACH_BIN"].cumsum()
    out["CUM_GOOD_PCT"] = out["GOOD_PCT_IN_EACH_BIN"].cumsum()
    out["KS_PER_BIN"] = (out["CUM_BAD_PCT"] - out["CUM_GOOD_PCT"]).abs()
    out["KS"] = out["KS_PER_BIN"]
    out["LIFT"] = out["AVG_BAD"] / overall_bad_rate if overall_bad_rate else np.nan
    out["TRUE_BAD_SHIFT"] = out["AVG_BAD"].shift(1) / out["AVG_BAD"] - 1
    out["RANK_ORDER_BUMP"] = out["TRUE_BAD_SHIFT"].apply(lambda x: 1 if x < 0 else 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["WOE"] = np.log(out["BAD_PCT_IN_EACH_BIN"] / out["GOOD_PCT_IN_EACH_BIN"])
    out["WOE"] = out["WOE"].replace([np.inf, -np.inf], 0).fillna(0)
    out["IV"] = (out["BAD_PCT_IN_EACH_BIN"] - out["GOOD_PCT_IN_EACH_BIN"]) * out["WOE"]
    out["AUC"] = safe_auc(y, s, sample_weight=weight)
    return out


def calc_lift_apt(y_true, y_score, start=1.0, stop=3.0, step=0.1, sample_weight=None):
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
    weight = np.ones(len(y_true), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return get_gains_table(
        pd.DataFrame({"y": y_true, "s": y_score, "w": weight}),
        "y",
        "s",
        nbins=bins,
        weight_col="w",
    )


def calc_equid_pct(y_true, y_score, bins=10, sample_weight=None, **kwargs):
    return calc_equid_dist(y_true, y_score, bins=bins, sample_weight=sample_weight, **kwargs)


def calc_fixed_pct(y_true, y_score, sample_weight=None, **kwargs):
    return calc_equid_dist(y_true, y_score, sample_weight=sample_weight, **kwargs)


def dataset_summary(name, data, tgt_name, scr_name, weight_col=None, nbins=10):
    weight = resolve_weights(data, weight_col=weight_col, expected_len=len(data))
    y_true = data[tgt_name].to_numpy()
    y_score = data[scr_name].to_numpy()
    roc_df = calc_roc(y_true, y_score, sample_weight=weight)
    gains = get_gains_table(data, tgt_name, scr_name, nbins=nbins, weight_col=weight_col)
    return {
        "index": name,
        "dataset": name,
        "DATASET": name,
        "AUC": safe_auc(y_true, y_score, sample_weight=weight),
        "KS": float(roc_df["KS"].max()) if "KS" in roc_df else np.nan,
        "LIFT": float(gains["LIFT"].max()) if "LIFT" in gains else np.nan,
        "IV": float(gains["IV"].sum()) if "IV" in gains else np.nan,
        "N": float(np.sum(weight)) if weight is not None else float(len(data)),
        "N_RAW": int(len(data)),
        "avgTrue": safe_weighted_average(y_true, weight),
        "avgScore": safe_weighted_average(y_score, weight),
    }


def get_perf_summary(train=None, validation=None, oot=None, tgt_name=None, scr_name=None, weight_col=None, nbins=10, **kwargs):
    rows = []
    for name, data in (("ins", train), ("oos", validation), ("oot", oot)):
        if data is not None:
            rows.append(dataset_summary(name, data, tgt_name, scr_name, weight_col=weight_col, nbins=nbins))
    return pd.DataFrame(rows)


def evaluate_performance(datasets=None, tgt_name=None, scr_name=None, sample_weight=None, nbins=10, **kwargs):
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
    def __init__(self, data=None, dep=None, score=None, nbins=10, weight_col=None, weighted_binning=None, **kwargs):
        self.data = data
        self.dep = dep
        self.score = score
        self.nbins = nbins
        self.weight_col = weight_col
        self.weighted_binning = weighted_binning
        self.kwargs = kwargs

    def calculate(self, weight_col=None, **kwargs):
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
    def __init__(self, tgt_name=None, scr_name=None, weight_col=None, nbins=10, **kwargs):
        self.tgt_name = tgt_name
        self.scr_name = scr_name
        self.weight_col = weight_col
        self.nbins = nbins
        self.kwargs = kwargs
        self.datasets = OrderedDict()

    def add_dataset(self, name, data, weight_col=None, **kwargs):
        self.datasets[name] = (data, weight_col)
        return self

    def evaluate(self, weight_col=None, to_show=False, display=False, **kwargs):
        rows = []
        for name, (data, ds_weight_col) in self.datasets.items():
            wc = ds_weight_col or weight_col or self.weight_col
            rows.append(dataset_summary(name, data, self.tgt_name, self.scr_name, weight_col=wc, nbins=self.nbins))
        return pd.DataFrame(rows)


def cross_risk_weighted_mean(data, agg_col, sample_weight, score_list, margin_name="Total_Avg_Risk"):
    """Weighted-mean cross-risk table after bin columns are assigned."""
    weight = np.asarray(sample_weight, dtype=float)
    values = pd.to_numeric(data[agg_col], errors="coerce").to_numpy(dtype=float)
    frame = data[["_bin_num1", "_bin_range1", "_bin_num2", "_bin_range2"]].copy()
    frame["_w"] = weight
    frame["_wv"] = values * weight

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
