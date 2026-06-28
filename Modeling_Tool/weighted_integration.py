# encoding: utf-8
"""Weighted training/evaluation integration for feat/weighted-training-eval."""
from __future__ import annotations

import copy
import itertools
import logging
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight

logger = logging.getLogger(__name__)


def _weights(data=None, weight_col=None, sample_weight=None, expected_len=None, wgt=None, wgt_col=None):
    if weight_col is None:
        weight_col = wgt_col
    if sample_weight is None:
        sample_weight = wgt
    return resolve_sample_weight(data=data, weight_col=weight_col, sample_weight=sample_weight, expected_len=expected_len)


def _safe_auc(y, score, sample_weight=None):
    try:
        return float(roc_auc_score(y, score, sample_weight=sample_weight))
    except Exception:
        return np.nan


def calc_roc(y_true, y_score, sample_weight=None):
    fpr, tpr, threshold = roc_curve(y_true, y_score, sample_weight=sample_weight)
    out = pd.DataFrame({"threshold": threshold, "fpr": fpr, "tpr": tpr})
    out["FPR"] = out["fpr"]
    out["TPR"] = out["tpr"]
    out["KS"] = (out["tpr"] - out["fpr"]).abs()
    return out


def calc_pr(y_true, y_score, sample_weight=None):
    precision, recall, threshold = precision_recall_curve(y_true, y_score, sample_weight=sample_weight)
    threshold = np.r_[threshold, np.nan]
    return pd.DataFrame({"threshold": threshold, "precision": precision, "recall": recall})


def _rank_bins(score, weight, nbins):
    score = np.asarray(score, dtype=float)
    weight = np.ones(len(score), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    order = np.lexsort((np.arange(len(score)), -score))
    bins = np.empty(len(score), dtype=int)
    total = float(weight.sum()) or float(len(score)) or 1.0
    cum = np.cumsum(weight[order])
    labels = np.ceil(cum / total * nbins).astype(int)
    labels = np.clip(labels, 1, nbins)
    bins[order] = labels
    return bins


def get_gains_table(data, dep, score, nbins=10, weight_col=None, weighted_binning=None, **kwargs):
    df = data[[dep, score] + ([weight_col] if weight_col is not None and weight_col in data.columns else [])].copy()
    w = _weights(df, weight_col=weight_col, expected_len=len(df))
    if w is None:
        w = np.ones(len(df), dtype=float)
    y = df[dep].astype(float).to_numpy()
    s = df[score].astype(float).to_numpy()
    bins = _rank_bins(s, w, int(nbins))
    df["BIN"] = bins
    df["_w"] = w
    df["_bad_w"] = w * y
    df["_good_w"] = w * (1.0 - y)
    grouped = df.groupby("BIN", sort=True)
    out = grouped.agg(
        N=("_w", "sum"),
        N_RAW=(dep, "size"),
        N_BAD=("_bad_w", "sum"),
        N_GOOD=("_good_w", "sum"),
        MIN_SCORE=(score, "min"),
        MAX_SCORE=(score, "max"),
        AVG_SCORE=(score, lambda x: np.average(x, weights=df.loc[x.index, "_w"])),
    ).reset_index()
    total_w = float(out["N"].sum()) or 1.0
    total_bad = float(out["N_BAD"].sum()) or 1.0
    total_good = float(out["N_GOOD"].sum()) or 1.0
    overall_bad_rate = total_bad / total_w if total_w else np.nan
    out["PROP"] = out["N"] / total_w
    out["AVG_BAD"] = out["N_BAD"] / out["N"].replace(0, np.nan)
    out["AVG_GOOD"] = out["N_GOOD"] / out["N"].replace(0, np.nan)
    out["BAD_PCT_IN_EACH_BIN"] = out["N_BAD"] / total_bad
    out["GOOD_PCT_IN_EACH_BIN"] = out["N_GOOD"] / total_good
    out["CUM_BAD_PCT"] = out["BAD_PCT_IN_EACH_BIN"].cumsum()
    out["CUM_GOOD_PCT"] = out["GOOD_PCT_IN_EACH_BIN"].cumsum()
    out["KS"] = (out["CUM_BAD_PCT"] - out["CUM_GOOD_PCT"]).abs()
    out["KS_PER_BIN"] = out["KS"]
    out["LIFT"] = out["AVG_BAD"] / overall_bad_rate if overall_bad_rate else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out["WOE"] = np.log(out["BAD_PCT_IN_EACH_BIN"] / out["GOOD_PCT_IN_EACH_BIN"])
    out["WOE"] = out["WOE"].replace([np.inf, -np.inf], 0).fillna(0)
    out["IV"] = (out["BAD_PCT_IN_EACH_BIN"] - out["GOOD_PCT_IN_EACH_BIN"]) * out["WOE"]
    out["AUC"] = _safe_auc(y, s, sample_weight=w)
    return out


def calc_lift_apt(y_true, y_score, start=1.0, stop=3.0, step=0.1, sample_weight=None):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    w = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    base = np.average(y, weights=w) if w.sum() else np.nan
    gains = get_gains_table(pd.DataFrame({"y": y, "s": s, "w": w}), "y", "s", nbins=100, weight_col="w")
    vals = []
    for target in np.arange(start, stop + step / 2.0, step):
        idx = (gains["LIFT"] - target).abs().idxmin()
        vals.append(float(gains.loc[idx, "LIFT"] if base == base else np.nan))
    return np.asarray(vals)


def calc_equid_dist(y_true, y_score, bins=10, sample_weight=None, **kwargs):
    return get_gains_table(pd.DataFrame({"y": y_true, "s": y_score, "w": np.ones(len(y_true)) if sample_weight is None else sample_weight}), "y", "s", nbins=bins, weight_col="w")


def calc_equid_pct(y_true, y_score, bins=10, sample_weight=None, **kwargs):
    return calc_equid_dist(y_true, y_score, bins=bins, sample_weight=sample_weight, **kwargs)


def calc_fixed_pct(y_true, y_score, sample_weight=None, **kwargs):
    return calc_equid_dist(y_true, y_score, sample_weight=sample_weight, **kwargs)


def _dataset_summary(name, data, tgt_name, scr_name, weight_col=None, nbins=10):
    w = _weights(data, weight_col=weight_col, expected_len=len(data))
    y = data[tgt_name].to_numpy()
    s = data[scr_name].to_numpy()
    roc_df = calc_roc(y, s, sample_weight=w)
    gains = get_gains_table(data, tgt_name, scr_name, nbins=nbins, weight_col=weight_col)
    return {
        "dataset": name,
        "DATASET": name,
        "AUC": _safe_auc(y, s, sample_weight=w),
        "KS": float(roc_df["KS"].max()),
        "LIFT": float(gains["LIFT"].max()),
        "IV": float(gains["IV"].sum()),
        "N": float(np.sum(w)) if w is not None else float(len(data)),
        "N_RAW": int(len(data)),
    }


def get_perf_summary(train=None, validation=None, oot=None, tgt_name=None, scr_name=None, weight_col=None, nbins=10, to_show=False, display=False, **kwargs):
    rows = []
    for name, data in (("train", train), ("validation", validation), ("oot", oot)):
        if data is not None:
            rows.append(_dataset_summary(name, data, tgt_name, scr_name, weight_col=weight_col, nbins=nbins))
    return pd.DataFrame(rows)


def evaluate_performance(datasets=None, tgt_name=None, scr_name=None, sample_weight=None, **kwargs):
    rows = []
    if datasets is None:
        return pd.DataFrame(rows)
    for name, payload in datasets.items():
        if isinstance(payload, dict):
            data = payload.get("data")
            w = payload.get("sample_weight", sample_weight)
        else:
            data = payload
            w = sample_weight
        if data is None:
            continue
        tmp = data.copy()
        tmp["_w"] = np.ones(len(tmp)) if w is None else w
        rows.append(_dataset_summary(name, tmp, tgt_name, scr_name, weight_col="_w"))
    return pd.DataFrame(rows)


def comparison_performance(*args, sample_weight=None, **kwargs):
    return evaluate_performance(*args, sample_weight=sample_weight, **kwargs)


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
            rows.append(_dataset_summary(name, data, self.tgt_name, self.scr_name, weight_col=wc, nbins=self.nbins))
        return pd.DataFrame(rows)


def cross_risk(data, row_var=None, col_var=None, tgt_name=None, weight_col=None, **kwargs):
    df = data.copy()
    df["_w"] = np.ones(len(df)) if weight_col is None else _weights(df, weight_col=weight_col, expected_len=len(df))
    if row_var is None:
        return pd.DataFrame({"N": [df["_w"].sum()], "N_RAW": [len(df)]})
    group_cols = [row_var] + ([col_var] if col_var is not None else [])
    out = df.groupby(group_cols).agg(N=("_w", "sum"), N_RAW=("_w", "size")).reset_index()
    total = out["N"].sum() or 1.0
    out["count"] = out["N"]
    out["count_pct"] = out["N"] / total
    if tgt_name is not None:
        tmp = df.assign(_bad=df[tgt_name] * df["_w"])
        bad = tmp.groupby(group_cols)["_bad"].sum().reset_index(name="bad")
        out = out.merge(bad, on=group_cols, how="left")
        out["bad_rate"] = out["bad"] / out["N"].replace(0, np.nan)
    return out


def get_gains_table_by_cust_metrics(data, dep, score, cust_metrics=None, weight_col=None, nbins=10, **kwargs):
    base = get_gains_table(data, dep, score, nbins=nbins, weight_col=weight_col, **kwargs)
    if cust_metrics:
        bins = _rank_bins(data[score].to_numpy(), _weights(data, weight_col=weight_col, expected_len=len(data)), nbins)
        tmp = data.copy()
        tmp["BIN"] = bins
        tmp["_w"] = np.ones(len(tmp)) if weight_col is None else data[weight_col]
        for metric in cust_metrics:
            if metric in tmp.columns:
                vals = tmp.groupby("BIN").apply(lambda g: np.average(g[metric], weights=g["_w"]))
                base[metric] = base["BIN"].map(vals)
    return base


class Model_Evaluation_Tool:
    def __init__(self, data=None, tgt_name=None, scr_name=None, weight_col=None, **kwargs):
        self.data = data
        self.tgt_name = tgt_name
        self.scr_name = scr_name
        self.weight_col = weight_col
        self.kwargs = kwargs

    def model_perf_compare(self, train=None, validation=None, oot=None, weight_col=None, **kwargs):
        return get_perf_summary(train=train or self.data, validation=validation, oot=oot, tgt_name=self.tgt_name, scr_name=self.scr_name, weight_col=weight_col or self.weight_col, **kwargs)

    def get_gains_summary(self, data=None, weight_col=None, **kwargs):
        return get_gains_table(data or self.data, self.tgt_name, self.scr_name, weight_col=weight_col or self.weight_col, **kwargs)

    def get_cust_metric_summary(self, data=None, weight_col=None, **kwargs):
        return get_gains_table_by_cust_metrics(data or self.data, self.tgt_name, self.scr_name, weight_col=weight_col or self.weight_col, **kwargs)

    def get_cross_risk_summary(self, data=None, weight_col=None, **kwargs):
        return cross_risk(data or self.data, tgt_name=self.tgt_name, weight_col=weight_col or self.weight_col, **kwargs)

    def cross_perf_eval(self, **kwargs):
        return self.model_perf_compare(**kwargs)


def _feature_frame(data, varlist):
    return data[varlist] if isinstance(data, pd.DataFrame) and varlist is not None else data


def _target_array(data, dep):
    return data[dep] if isinstance(data, pd.DataFrame) and dep is not None else dep


def _gbm_fit(self, X, y=None, X_val=None, y_val=None, sample_weight=None, eval_sample_weight=None, wgt=None, **kwargs):
    sample_weight = sample_weight if sample_weight is not None else wgt
    model_type = getattr(self, "model_type", getattr(self, "gbm_type", "lgb")).lower()
    params = copy.deepcopy(getattr(self, "params", None) or getattr(self, "model_params", None) or {})
    if model_type in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        model = XGBClassifier(**params)
        fit_kwargs = {"sample_weight": sample_weight}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if eval_sample_weight is not None:
                fit_kwargs["sample_weight_eval_set"] = [eval_sample_weight]
            fit_kwargs["verbose"] = False
        model.fit(X, y, **{k: v for k, v in fit_kwargs.items() if v is not None})
    elif model_type in ("cat", "catboost"):
        from catboost import CatBoostClassifier, Pool
        params.setdefault("verbose", False)
        model = CatBoostClassifier(**params)
        fit_kwargs = {"sample_weight": sample_weight}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = Pool(X_val, y_val, weight=eval_sample_weight) if eval_sample_weight is not None else (X_val, y_val)
        model.fit(X, y, **{k: v for k, v in fit_kwargs.items() if v is not None})
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**params)
        fit_kwargs = {"sample_weight": sample_weight}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if eval_sample_weight is not None:
                fit_kwargs["eval_sample_weight"] = [eval_sample_weight]
        model.fit(X, y, **{k: v for k, v in fit_kwargs.items() if v is not None})
    self.model = model
    return self


def _gbm_predict(self, X, **kwargs):
    model = getattr(self, "model", None)
    if model is None:
        raise AttributeError("model is not fitted")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
    return model.predict(X)


def _quick_train(model_type, train_data, validation_data=None, x=None, y=None, params=None, wgt_col=None, val_wgt_col=None, **kwargs):
    class _Wrapper:
        pass
    wrapper = _Wrapper()
    wrapper.model_type = model_type
    wrapper.params = params or {}
    X = train_data[x]
    yy = train_data[y]
    Xv = validation_data[x] if validation_data is not None else None
    yv = validation_data[y] if validation_data is not None else None
    sw = train_data[wgt_col].to_numpy() if wgt_col is not None else None
    evw = validation_data[val_wgt_col].to_numpy() if validation_data is not None and val_wgt_col is not None else None
    _gbm_fit(wrapper, X, yy, Xv, yv, sample_weight=sw, eval_sample_weight=evw)
    return wrapper.model


def lgbm_quick_train(train_data, validation_data=None, x=None, y=None, params=None, wgt_col=None, val_wgt_col=None, **kwargs):
    return _quick_train("lgb", train_data, validation_data, x, y, params, wgt_col, val_wgt_col, **kwargs)


def xgbm_quick_train(train_data, validation_data=None, x=None, y=None, params=None, wgt_col=None, val_wgt_col=None, sample_weight_eval_set=None, **kwargs):
    if val_wgt_col is None and sample_weight_eval_set is not None and validation_data is not None:
        validation_data = validation_data.copy()
        validation_data["_eval_weight"] = sample_weight_eval_set
        val_wgt_col = "_eval_weight"
    return _quick_train("xgb", train_data, validation_data, x, y, params, wgt_col, val_wgt_col, **kwargs)


def catboost_quick_train(train_data, validation_data=None, x=None, y=None, params=None, wgt_col=None, val_wgt_col=None, **kwargs):
    return _quick_train("cat", train_data, validation_data, x, y, params, wgt_col, val_wgt_col, **kwargs)


def _lr_fit(self, data, varlist, tgt_name=None, weight_col=None, sample_weight=None, wgt_col=None, **kwargs):
    dep = tgt_name or kwargs.pop("dep", None) or kwargs.pop("y", None)
    params = copy.deepcopy(getattr(self, "params", None) or {})
    model = LogisticRegression(**params)
    sw = _weights(data, weight_col=weight_col, sample_weight=sample_weight, wgt_col=wgt_col, expected_len=len(data))
    model.fit(data[varlist], data[dep], sample_weight=sw)
    self.model = model
    self.lr_model = model
    self.varlist = list(varlist)
    self.tgt_name = dep
    return self


def _lr_predict_proba(self, data, varlist=None, **kwargs):
    cols = varlist or getattr(self, "varlist", None)
    model = getattr(self, "model", getattr(self, "lr_model", None))
    return model.predict_proba(data[cols] if cols is not None and isinstance(data, pd.DataFrame) else data)


def _grid(params):
    keys = list((params or {}).keys())
    for values in itertools.product(*[params[k] for k in keys]):
        yield dict(zip(keys, values))


def _lr_grid_search_params(self, data, varlist, tgt_name, eval_sets, param_grid, primary_set="oot", gap_ref_sets=None, weight_col=None, eval_weight_col=None, refit=False, verbose=True, **kwargs):
    rows = []
    base_params = copy.deepcopy(getattr(self, "params", None) or {})
    for p in _grid(param_grid):
        params = {**base_params, **p}
        model = LogisticRegression(**params)
        sw = _weights(data, weight_col=weight_col, expected_len=len(data))
        model.fit(data[varlist], data[tgt_name], sample_weight=sw)
        row = dict(p)
        for name, ds in eval_sets.items():
            ew = _weights(ds, weight_col=eval_weight_col, expected_len=len(ds))
            pred = model.predict_proba(ds[varlist])[:, 1]
            row[f"AUC_{name}"] = _safe_auc(ds[tgt_name], pred, sample_weight=ew)
        row["score"] = row.get(f"AUC_{primary_set}", np.nan)
        rows.append(row)
    res = pd.DataFrame(rows)
    if len(res):
        best = res.sort_values("score", ascending=False).iloc[0]
        self.best_params_ = {k: best[k] for k in (param_grid or {}).keys()}
        if refit:
            self.params = {**base_params, **self.best_params_}
            _lr_fit(self, data, varlist, tgt_name, weight_col=weight_col)
    return res


def _lr_stepwise_selection(self, data, varlist, tgt_name=None, criterion="aic", direction="forward", max_iter=None, weight_col=None, sample_weight=None, verbose=True, **kwargs):
    max_iter = max_iter or len(varlist)
    selected = []
    remaining = list(varlist)
    while remaining and len(selected) < max_iter:
        best_var = None
        best_score = np.inf
        for var in remaining:
            cols = selected + [var]
            tmp = LogisticRegression(**(getattr(self, "params", None) or {}))
            sw = _weights(data, weight_col=weight_col, sample_weight=sample_weight, expected_len=len(data))
            tmp.fit(data[cols], data[tgt_name], sample_weight=sw)
            pred = tmp.predict_proba(data[cols])[:, 1]
            score = _ic(data[tgt_name], pred, len(cols), sw, bic=(criterion.lower() == "bic"))
            if score < best_score:
                best_score, best_var = score, var
        selected.append(best_var)
        remaining.remove(best_var)
    return selected or list(varlist[:1])


def _ic(y, p, k, w=None, bic=False):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    ll = np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
    n = np.sum(w) or len(y)
    return float(np.log(n) * k - 2 * ll) if bic else float(2 * k - 2 * ll)


def _lr_get_aic(self, data, varlist, tgt_name=None, weight_col=None, sample_weight=None, **kwargs):
    pred = _lr_predict_proba(self, data, varlist)[:, 1]
    sw = _weights(data, weight_col=weight_col, sample_weight=sample_weight, expected_len=len(data))
    return _ic(data[tgt_name or getattr(self, "tgt_name", None)], pred, len(varlist), sw, bic=False)


def _lr_get_bic(self, data, varlist, tgt_name=None, weight_col=None, sample_weight=None, **kwargs):
    pred = _lr_predict_proba(self, data, varlist)[:, 1]
    sw = _weights(data, weight_col=weight_col, sample_weight=sample_weight, expected_len=len(data))
    return _ic(data[tgt_name or getattr(self, "tgt_name", None)], pred, len(varlist), sw, bic=True)


def backward_lgbm(train_data, varlist, dep, varreduct_params=None, stopping_metric="auc", seed=42, num_boost_round=200, early_stopping_rounds=20, importance_type="gain", cum_importance_threshold=0.99, min_vars=10, validation_data=None, test_data_dict=None, ret_perf=True, weight_col=None, validation_weight_col=None, wgt_col=None, **kwargs):
    import lightgbm as lgb
    weight_col = weight_col or wgt_col
    params = copy.deepcopy(varreduct_params or {})
    params.setdefault("objective", "binary")
    params.setdefault("metric", stopping_metric)
    params.setdefault("seed", seed)
    tr_w = _weights(train_data, weight_col=weight_col, expected_len=len(train_data))
    train_set = lgb.Dataset(train_data[varlist], label=train_data[dep], weight=tr_w)
    valid_sets = [train_set]
    valid_names = ["mdl"]
    if validation_data is not None:
        va_w = _weights(validation_data, weight_col=validation_weight_col or weight_col, expected_len=len(validation_data))
        valid_sets = [lgb.Dataset(validation_data[varlist], label=validation_data[dep], weight=va_w)]
        valid_names = ["hd"]
    model = lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=valid_sets, valid_names=valid_names, callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(period=-1)])
    imp = pd.DataFrame({"feature": model.feature_name(), "importance": model.feature_importance(importance_type=importance_type)}).sort_values("importance", ascending=False).reset_index(drop=True)
    denom = imp["importance"].sum() or 1.0
    imp["cum"] = imp["importance"].cumsum() / denom
    selected = imp.loc[imp["cum"] <= cum_importance_threshold, "feature"].tolist()
    if len(selected) < min_vars:
        selected = imp.head(min(min_vars, len(imp)))["feature"].tolist()
    return (selected, model) if not ret_perf else (selected, model, {})


def backward_xgbm(train_data, varlist, dep, varreduct_params=None, stopping_metric="auc", seed=42, num_boost_round=200, early_stopping_rounds=20, importance_type="gain", cum_importance_threshold=0.99, min_vars=10, validation_data=None, test_data_dict=None, ret_perf=True, weight_col=None, validation_weight_col=None, wgt_col=None, **kwargs):
    import xgboost as xgb
    weight_col = weight_col or wgt_col
    params = copy.deepcopy(varreduct_params or {})
    params.setdefault("eval_metric", stopping_metric)
    params.setdefault("seed", seed)
    tr_w = _weights(train_data, weight_col=weight_col, expected_len=len(train_data))
    dtrain = xgb.DMatrix(train_data[varlist], label=train_data[dep], weight=tr_w)
    evals = [(dtrain, "mdl")]
    if validation_data is not None:
        va_w = _weights(validation_data, weight_col=validation_weight_col or weight_col, expected_len=len(validation_data))
        evals.append((xgb.DMatrix(validation_data[varlist], label=validation_data[dep], weight=va_w), "hd"))
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
    raw = model.get_score(importance_type=importance_type)
    imp = pd.DataFrame({"feature": list(raw.keys()), "importance": list(raw.values())}) if raw else pd.DataFrame({"feature": varlist, "importance": 0.0})
    for f in varlist:
        if f not in set(imp["feature"]):
            imp = pd.concat([imp, pd.DataFrame({"feature": [f], "importance": [0.0]})], ignore_index=True)
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    denom = imp["importance"].sum() or 1.0
    imp["cum"] = imp["importance"].cumsum() / denom
    selected = imp.loc[imp["cum"] <= cum_importance_threshold, "feature"].tolist()
    if len(selected) < min_vars:
        selected = imp.head(min(min_vars, len(imp)))["feature"].tolist()
    return (selected, model) if not ret_perf else (selected, model, {})


def _bve_init(self, train_data, varlist, dep, model_type="lgbm", validation_data=None, test_data_dict=None, weight_col=None, validation_weight_col=None, wgt_col=None):
    self.train_data = train_data
    self.varlist = list(varlist)
    self.dep = dep
    self.model_type = model_type.lower()
    self.validation_data = validation_data
    self.test_data_dict = test_data_dict or {}
    self.weight_col = weight_col or wgt_col
    self.validation_weight_col = validation_weight_col
    self._results = []


def _bve_run(self, n_rounds=5, varreduct_params=None, stopping_metric="auc", seed=42, num_boost_round=200, early_stopping_rounds=20, importance_type="gain", cum_importance_threshold=0.99, min_vars=10, ret_perf=True, nbins=10, **kwargs):
    current = list(self.varlist)
    self._results = []
    fn = backward_lgbm if self.model_type == "lgbm" else backward_xgbm
    for i in range(1, n_rounds + 1):
        result = fn(self.train_data, current, self.dep, varreduct_params=copy.deepcopy(varreduct_params), stopping_metric=stopping_metric, seed=seed, num_boost_round=num_boost_round, early_stopping_rounds=early_stopping_rounds, importance_type=importance_type, cum_importance_threshold=cum_importance_threshold, min_vars=min_vars, validation_data=self.validation_data, test_data_dict=self.test_data_dict, ret_perf=ret_perf, weight_col=self.weight_col, validation_weight_col=self.validation_weight_col, **kwargs)
        selected, model = result[:2]
        row = {"round": i, "n_vars_in": len(current), "n_vars_out": len(selected), "selected_vars": selected, "model": model, "perf": result[2] if ret_perf and len(result) > 2 else {}}
        self._results.append(row)
        current = selected
        if len(current) <= min_vars:
            break
    return self._results


def apply_eval_patches(ns=None):
    import Modeling_Tool.Eval.evaluate_model as em
    import Modeling_Tool.Eval.Model_Eval_Tool as met
    import Modeling_Tool.Eval.Evaluation_Tool as et
    for module in (em, met, et):
        for name, obj in {
            "calc_roc": calc_roc,
            "calc_pr": calc_pr,
            "calc_lift_apt": calc_lift_apt,
            "calc_equid_dist": calc_equid_dist,
            "calc_equid_pct": calc_equid_pct,
            "calc_fixed_pct": calc_fixed_pct,
            "evaluate_performance": evaluate_performance,
            "comparison_performance": comparison_performance,
            "get_gains_table": get_gains_table,
            "get_perf_summary": get_perf_summary,
            "cross_risk": cross_risk,
            "get_gains_table_by_cust_metrics": get_gains_table_by_cust_metrics,
            "GainsTableCalculator": GainsTableCalculator,
            "PerformanceEvaluator": PerformanceEvaluator,
            "Model_Evaluation_Tool": Model_Evaluation_Tool,
        }.items():
            if hasattr(module, name):
                setattr(module, name, obj)
    if ns is not None:
        ns.update({k: v for k, v in globals().items() if k in {"calc_roc", "calc_pr", "calc_lift_apt", "calc_equid_dist", "calc_equid_pct", "calc_fixed_pct", "evaluate_performance", "comparison_performance", "get_gains_table", "get_perf_summary", "cross_risk", "get_gains_table_by_cust_metrics", "GainsTableCalculator", "PerformanceEvaluator", "Model_Evaluation_Tool"}})


def apply_model_patches(ns=None):
    import Modeling_Tool.Model.GBM_Tool as gbm
    import Modeling_Tool.Model.LRM_Tool as lrm
    import Modeling_Tool.Model.Backward_Tool as bwd
    if hasattr(gbm, "GradientBoostingModel"):
        gbm.GradientBoostingModel.fit = _gbm_fit
        gbm.GradientBoostingModel.predict = _gbm_predict
    for cls_name in ("LightGBMModel", "XGBoostModel", "CatBoostModel"):
        if hasattr(gbm, cls_name):
            getattr(gbm, cls_name).fit = _gbm_fit
            getattr(gbm, cls_name).predict = _gbm_predict
    gbm.lgbm_quick_train = lgbm_quick_train
    gbm.xgbm_quick_train = xgbm_quick_train
    gbm.catboost_quick_train = catboost_quick_train
    if hasattr(lrm, "LRMaster"):
        lrm.LRMaster.fit = _lr_fit
        lrm.LRMaster.predict_proba = _lr_predict_proba
        lrm.LRMaster.grid_search_params = _lr_grid_search_params
        lrm.LRMaster.stepwise_selection = _lr_stepwise_selection
        lrm.LRMaster.get_aic = _lr_get_aic
        lrm.LRMaster.get_bic = _lr_get_bic
    bwd.backward_lgbm = backward_lgbm
    bwd.backward_xgbm = backward_xgbm
    if hasattr(bwd, "BackwardVariableEliminator"):
        bwd.BackwardVariableEliminator.__init__ = _bve_init
        bwd.BackwardVariableEliminator.run = _bve_run
    if ns is not None:
        ns.update({
            "lgbm_quick_train": lgbm_quick_train,
            "xgbm_quick_train": xgbm_quick_train,
            "catboost_quick_train": catboost_quick_train,
            "backward_lgbm": backward_lgbm,
            "backward_xgbm": backward_xgbm,
            "LRMaster": lrm.LRMaster,
            "GradientBoostingModel": gbm.GradientBoostingModel,
            "LightGBMModel": getattr(gbm, "LightGBMModel", None),
            "XGBoostModel": getattr(gbm, "XGBoostModel", None),
            "CatBoostModel": getattr(gbm, "CatBoostModel", None),
            "BackwardVariableEliminator": bwd.BackwardVariableEliminator,
        })
