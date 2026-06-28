"""Holdout hyperparameter search extension for GradientBoostingModel.

This module attaches ``param_search`` to ``GradientBoostingModel`` at import
 time.  The implementation lives outside ``GBM_Tool.py`` so the existing
LightGBM/XGBoost training wrappers remain unchanged while the public GBM class
gets the same holdout-search surface as ``LRMaster.grid_search_params``.
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight
from .GBM_Tool import GradientBoostingModel


def _native(value):
    return value.item() if hasattr(value, "item") else value


def _validate_columns(data, varlist, tgt_name, eval_sets, weight_col=None, eval_weight_col=None):
    missing = [c for c in (list(varlist) + [tgt_name]) if c not in data.columns]
    if missing:
        raise KeyError("training data missing columns: {0}".format(missing))
    if weight_col is not None and weight_col not in data.columns:
        raise KeyError("training data missing weight column: {0}".format(weight_col))
    for name, df_eval in eval_sets.items():
        miss = [c for c in (list(varlist) + [tgt_name]) if c not in df_eval.columns]
        if miss:
            raise KeyError("eval set '{0}' missing columns: {1}".format(name, miss))
        if eval_weight_col is not None and eval_weight_col not in df_eval.columns:
            raise KeyError("eval set '{0}' missing weight column: {1}".format(name, eval_weight_col))


def _resolve_primary_and_refs(eval_sets, primary_set, gap_ref_sets):
    if not eval_sets:
        raise ValueError("eval_sets must be a non-empty {name: DataFrame} mapping.")
    set_names = list(eval_sets.keys())
    if primary_set is None:
        primary_set = set_names[-1]
    if primary_set not in eval_sets:
        raise ValueError("primary_set '{0}' not in eval_sets {1}".format(primary_set, set_names))
    if gap_ref_sets is None:
        gap_ref_sets = [name for name in set_names if name != primary_set]
    for name in gap_ref_sets:
        if name not in eval_sets:
            raise ValueError("gap_ref_sets contains unknown eval set: {0}".format(name))
    return set_names, primary_set, gap_ref_sets


def _resolve_validation_set(eval_sets, primary_set, validation_set):
    if validation_set is not None:
        if validation_set not in eval_sets:
            raise ValueError("validation_set '{0}' not in eval_sets".format(validation_set))
        return validation_set
    if "oos" in eval_sets:
        return "oos"
    if "validation" in eval_sets:
        return "validation"
    if "valid" in eval_sets:
        return "valid"
    return primary_set


def _score_from_metrics(metric_dict, objective, primary_set, gap_ref_sets):
    if callable(objective):
        return float(objective(metric_dict))
    if objective == "max_primary":
        return float(metric_dict[primary_set])
    if objective == "oot_gap_penalized":
        primary = float(metric_dict[primary_set])
        if gap_ref_sets:
            ref = float(np.mean([metric_dict[name] for name in gap_ref_sets]))
            return primary - abs(ref - primary)
        return primary
    raise ValueError("Unknown objective: {0}".format(objective))


def _evaluate_candidate(candidate, eval_sets, varlist, tgt_name, metric, eval_weight_col=None):
    if metric != "auc":
        raise ValueError("Only metric='auc' is currently supported.")
    metric_dict = {}
    for name, df_eval in eval_sets.items():
        proba = candidate.predict(df_eval[varlist])
        sw = resolve_sample_weight(data=df_eval, weight_col=eval_weight_col)
        if sw is not None:
            metric_dict[name] = roc_auc_score(df_eval[tgt_name], proba, sample_weight=sw)
        else:
            metric_dict[name] = roc_auc_score(df_eval[tgt_name], proba)
    return metric_dict


def _fit_candidate(model_type, base_params, candidate_params, data, varlist, tgt_name,
                   validation_df, fit_kwargs, weight_col=None, eval_weight_col=None):
    params = {**base_params, **candidate_params}
    candidate = GradientBoostingModel(model_type, params)
    fk = dict(fit_kwargs)
    train_sw = resolve_sample_weight(data=data, weight_col=weight_col)
    eval_sw = resolve_sample_weight(data=validation_df, weight_col=eval_weight_col)
    if train_sw is not None:
        fk["sample_weight"] = train_sw
    if eval_sw is not None:
        fk["eval_sample_weight"] = eval_sw
    candidate.fit(data[varlist], data[tgt_name], validation_df[varlist], validation_df[tgt_name], **fk)
    return candidate


def _grid_combinations(search_space):
    param_names = list(search_space.keys())
    if not param_names:
        return param_names, [()]
    values = [list(search_space[name]) for name in param_names]
    if any(len(v) == 0 for v in values):
        raise ValueError("search_space values must be non-empty iterables.")
    return param_names, list(itertools.product(*values))


def _suggest_from_spec(trial, name, spec):
    if isinstance(spec, Mapping):
        kind = spec.get("type")
        if kind == "int":
            return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1), log=spec.get("log", False))
        if kind == "float":
            return trial.suggest_float(name, spec["low"], spec["high"], step=spec.get("step"), log=spec.get("log", False))
        if kind == "categorical":
            return trial.suggest_categorical(name, spec["choices"])
        raise ValueError("Unknown optuna search_space spec type for {0}: {1}".format(name, kind))
    if not isinstance(spec, (tuple, list)) or not spec:
        raise ValueError("Optuna spec for {0} must be a tuple/list or dict.".format(name))
    kind = spec[0]
    if kind == "int":
        log = len(spec) >= 5 and spec[4] == "log"
        step = spec[3] if len(spec) >= 4 and spec[3] != "log" else 1
        return trial.suggest_int(name, spec[1], spec[2], step=step, log=log)
    if kind == "float":
        log = len(spec) >= 4 and spec[3] == "log"
        return trial.suggest_float(name, spec[1], spec[2], log=log)
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec[1]))
    raise ValueError("Unknown optuna search_space spec type for {0}: {1}".format(name, kind))


def _format_search_row(params, set_names, metric_dict, score, use_gap, primary_set, gap_ref_sets):
    row = dict(params)
    for name in set_names:
        row["AUC_{0}".format(name)] = round(metric_dict[name], 5)
    if use_gap:
        ref = float(np.mean([metric_dict[name] for name in gap_ref_sets]))
        row["gap"] = round(ref - metric_dict[primary_set], 5)
    row["score"] = round(score, 5)
    return row


def _gbm_param_search(self, data, varlist, tgt_name, eval_sets, search_space,
                      engine="grid", objective="oot_gap_penalized",
                      primary_set=None, gap_ref_sets=None, metric="auc",
                      validation_set=None, n_trials=50, refit=True,
                      verbose=True, fit_kwargs=None, random_state=None,
                      weight_col=None, eval_weight_col=None):
    if metric != "auc":
        raise ValueError("Only metric='auc' is currently supported.")
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    set_names, primary_set, gap_ref_sets = _resolve_primary_and_refs(eval_sets, primary_set, gap_ref_sets)
    _validate_columns(data, varlist, tgt_name, eval_sets, weight_col, eval_weight_col)
    validation_name = _resolve_validation_set(eval_sets, primary_set, validation_set)
    validation_df = eval_sets[validation_name]
    use_gap = (not callable(objective)) and objective == "oot_gap_penalized" and len(gap_ref_sets) > 0
    rows = []
    if engine == "grid":
        param_names, combos = _grid_combinations(search_space)
        if verbose:
            print("param_search(grid): {0} combinations".format(len(combos)))
        for combo in combos:
            params = dict(zip(param_names, combo))
            candidate = _fit_candidate(self.model_type, self.params, params, data, varlist, tgt_name, validation_df, fit_kwargs, weight_col, eval_weight_col)
            metric_dict = _evaluate_candidate(candidate, eval_sets, varlist, tgt_name, metric, eval_weight_col)
            score = _score_from_metrics(metric_dict, objective, primary_set, gap_ref_sets)
            rows.append(_format_search_row(params, set_names, metric_dict, score, use_gap, primary_set, gap_ref_sets))
    elif engine == "optuna":
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("engine='optuna' requires optuna") from exc
        sampler = optuna.samplers.TPESampler(seed=random_state) if random_state is not None else None
        study = optuna.create_study(direction="maximize", sampler=sampler)
        def _objective(trial):
            params = {name: _suggest_from_spec(trial, name, spec) for name, spec in search_space.items()}
            candidate = _fit_candidate(self.model_type, self.params, params, data, varlist, tgt_name, validation_df, fit_kwargs, weight_col, eval_weight_col)
            metric_dict = _evaluate_candidate(candidate, eval_sets, varlist, tgt_name, metric, eval_weight_col)
            score = _score_from_metrics(metric_dict, objective, primary_set, gap_ref_sets)
            return score
        study.optimize(_objective, n_trials=n_trials)
        for trial in study.trials:
            if trial.value is None:
                continue
            row = {"trial_number": trial.number, **trial.params, "score": round(float(trial.value), 5)}
            rows.append(row)
    else:
        raise ValueError("engine must be 'grid' or 'optuna'")
    if not rows:
        raise RuntimeError("param_search produced no successful candidates.")
    search_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    param_cols = [c for c in search_space.keys() if c in search_df.columns]
    best_row = search_df.iloc[0]
    self.best_params_ = {k: _native(best_row[k]) for k in param_cols}
    self.search_results_ = search_df
    self.params = {**self.params, **self.best_params_}
    if refit:
        refit_kwargs = dict(fit_kwargs)
        train_sw = resolve_sample_weight(data=data, weight_col=weight_col)
        eval_sw = resolve_sample_weight(data=validation_df, weight_col=eval_weight_col)
        if train_sw is not None:
            refit_kwargs["sample_weight"] = train_sw
        if eval_sw is not None:
            refit_kwargs["eval_sample_weight"] = eval_sw
        self.fit(data[varlist], data[tgt_name], validation_df[varlist], validation_df[tgt_name], **refit_kwargs)
    return search_df


def attach_gbm_param_search():
    GradientBoostingModel.param_search = _gbm_param_search
    if not hasattr(GradientBoostingModel, "best_params_"):
        GradientBoostingModel.best_params_ = None
    if not hasattr(GradientBoostingModel, "search_results_"):
        GradientBoostingModel.search_results_ = None
    return GradientBoostingModel


attach_gbm_param_search()
