"""
Backward Variable Elimination Toolkit (Unified Version)
=======================================================

This module provides backward variable elimination based on LightGBM and XGBoost.
Variables are screened with a cumulative feature-importance threshold, and
post-training performance analysis is supported.

Functions
---------
backward_lgbm
    Run backward variable elimination with a LightGBM model
backward_xgbm
    Run backward variable elimination with an XGBoost model

Classes
-------
BackwardVariableEliminator
    Backward variable eliminator supporting LightGBM and XGBoost
BackwardEliminationAnalyzer
    Analyzer for backward elimination results
"""

import os
import sys
import copy
import logging
import functools
from collections import OrderedDict
from typing import Optional, List, Dict, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight


def _resolve_backward_perf_weight_col(
    split_name: str,
    df: pd.DataFrame,
    weight_col: Optional[str],
    validation_weight_col: Optional[str],
) -> Optional[str]:
    """Pick the weight column present on a scored backward-evaluation frame."""
    if split_name == "hd" and validation_weight_col and validation_weight_col in df.columns:
        return validation_weight_col
    if weight_col and weight_col in df.columns:
        return weight_col
    return None


def _backward_perf_summary(
    df_score: pd.DataFrame,
    *,
    split_name: str,
    dep: str,
    score_col: str,
    weight_col: Optional[str] = None,
    validation_weight_col: Optional[str] = None,
    nbins: int = 10,
    precision: int = 5,
    min_bin_prop: float = 0.05,
    include_missing: bool = True,
    equal_freq: bool = True,
):
    """Build a performance summary for one backward-evaluation dataset split."""
    from Modeling_Tool.Eval.Model_Eval_Tool import get_perf_summary

    perf_weight_col = _resolve_backward_perf_weight_col(
        split_name,
        df_score,
        weight_col=weight_col,
        validation_weight_col=validation_weight_col,
    )
    kwargs = {
        "train": df_score if split_name == "mdl" else None,
        "validation": df_score if split_name == "hd" else None,
        "oot": df_score if split_name not in ("mdl", "hd") else None,
        "tgt_name": dep,
        "scr_name": score_col,
        "pct_bins": nbins,
        "precision": precision,
        "min_bin_prop": min_bin_prop,
        "include_missing": include_missing,
        "equal_freq": equal_freq,
        "display": False,
        "to_show": False,
    }
    if perf_weight_col is not None:
        kwargs["weight_col"] = perf_weight_col
    return get_perf_summary(**kwargs)


def backward_lgbm(
    train_data: pd.DataFrame,
    varlist: List[str],
    dep: str,
    varreduct_params: Optional[Dict] = None,
    stopping_metric: str = "auc",
    seed: int = 42,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
    importance_type: str = "gain",
    cum_importance_threshold: float = 0.99,
    min_vars: int = 10,
    validation_data: Optional[pd.DataFrame] = None,
    test_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ret_perf: bool = True,
    nbins: int = 10,
    precision: int = 5,
    min_bin_prop: float = 0.05,
    include_missing: bool = True,
    equal_freq: bool = True,
    ascending: bool = True,
    fillna: Optional[float] = None,
    spec_values: Optional[List] = None,
    weight_col: Optional[str] = None,
    validation_weight_col: Optional[str] = None,
    wgt_col: Optional[str] = None,
) -> Tuple:
    """
    Run backward variable elimination with a LightGBM model.

    Train a LightGBM model and screen the variables by a cumulative
    feature-importance threshold (backward variable elimination).

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset. Must contain the ``dep`` column and all feature columns in ``varlist``.
    varlist : list of str
        List of feature variables used for modeling.
    dep : str
        Name of the target column (binary 0/1 variable).
    varreduct_params : dict, optional
        LightGBM hyperparameter dictionary. Required parameters that are not specified use preset values.
    stopping_metric : str, default "auc"
        Evaluation metric for early stopping, e.g. "auc" or "binary_logloss".
    seed : int, default 42
        Random seed for reproducibility.
    num_boost_round : int, default 200
        Maximum number of boosting rounds.
    early_stopping_rounds : int, default 20
        Number of early-stopping rounds; training stops when the validation metric has not improved for this many consecutive rounds.
    importance_type : str, default "gain"
        Feature importance type, "gain" or "split".
    cum_importance_threshold : float, default 0.99
        Cumulative feature-importance threshold; select the fewest features that cover this proportion of the total importance.
    min_vars : int, default 10
        Minimum number of variables to keep.
    validation_data : pd.DataFrame, optional
        Validation dataset, used for early stopping.
    test_data_dict : dict, optional
        Dictionary of test datasets, in the form ``{name: DataFrame}``.
    ret_perf : bool, default True
        Whether to return model performance metrics.
    nbins : int, default 10
        Number of bins in the Gains table.
    precision : int, default 5
        Numeric precision.
    min_bin_prop : float, default 0.05
        Minimum bin proportion.
    include_missing : bool, default True
        Whether to include a missing-value bin.
    equal_freq : bool, default True
        Whether to use equal-frequency binning.
    ascending : bool, default True
        Whether to sort the Gains table in ascending order.
    fillna : float, optional
        Fill value for missing values.
    spec_values : list, optional
        List of special values.

    Returns
    -------
    tuple
        ``(selected_vars, model, perf_dict)``, or ``(selected_vars, model)`` when ``ret_perf`` is False.

    Raises
    ------
    TypeError
        If the input data is not a ``pandas.DataFrame``.
    ImportError
        If lightgbm is not installed.

    Examples
    --------
    >>> selected_vars, model, perf = backward_lgbm(
    ...     train_data=train_df,
    ...     varlist=feature_cols,
    ...     dep='target',
    ...     validation_data=val_df
    ... )
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Please install lightgbm: pip install lightgbm")

    if varreduct_params is None:
        varreduct_params = {}

    if test_data_dict is None:
        test_data_dict = {}

    if spec_values is None:
        spec_values = []

    datain_all = OrderedDict()
    datain_all["mdl"] = train_data
    if validation_data is not None:
        datain_all["hd"] = validation_data
    datain_all.update(test_data_dict)

    # Check data format consistency
    try:
        for k, v in datain_all.items():
            assert isinstance(v, pd.DataFrame)
    except AssertionError:
        logging.warning("Please provide data in pandas.DataFrame format")
        raise TypeError("Please provide data in pandas.DataFrame format")

    # Preset parameters (to keep the model reproducible)
    hyperparams_preset = {
        "metric": stopping_metric,
        "seed": seed,
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_threads": 8
    }

    # Fill in any missing required parameters
    lacked_params = [k for k in list(hyperparams_preset.keys()) if k not in list(varreduct_params.keys())]
    for param in lacked_params:
        varreduct_params[param] = hyperparams_preset[param]

    weight_col = weight_col or wgt_col

    # Build the LightGBM datasets
    train_weight = resolve_sample_weight(data=train_data, weight_col=weight_col, expected_len=len(train_data))
    lgb_train = lgb.Dataset(train_data[varlist], label=train_data[dep], weight=train_weight)

    if validation_data is not None:
        valid_weight = resolve_sample_weight(
            data=validation_data,
            weight_col=validation_weight_col or weight_col,
            expected_len=len(validation_data),
        )
        lgb_valid = lgb.Dataset(validation_data[varlist], label=validation_data[dep], weight=valid_weight)
        valid_sets = [lgb_valid]
        valid_names = ["hd"]
    else:
        valid_sets = [lgb_train]
        valid_names = ["mdl"]

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1)
    ]

    # Train the model
    model = lgb.train(
        params=varreduct_params,
        train_set=lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )

    # Get the feature importance and screen the variables
    importance_df = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type=importance_type)
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df["cum_importance"] = importance_df["importance"].cumsum() / importance_df["importance"].sum()

    # Select the variables within the cumulative-importance threshold
    selected_idx = importance_df[importance_df["cum_importance"] <= cum_importance_threshold].index.tolist()

    # Make sure at least min_vars variables are kept
    if len(selected_idx) < min_vars:
        selected_idx = list(range(min(min_vars, len(importance_df))))

    selected_vars = importance_df.loc[selected_idx, "feature"].tolist()

    if not ret_perf:
        return selected_vars, model

    # Compute performance
    score_col = "_lgbm_score"
    perf_dict = {}

    for name, df in datain_all.items():
        df_score = df.copy()
        df_score[score_col] = model.predict(df_score[varlist])
        perf_dict[name] = _backward_perf_summary(
            df_score,
            split_name=name,
            dep=dep,
            score_col=score_col,
            weight_col=weight_col,
            validation_weight_col=validation_weight_col,
            nbins=nbins,
            precision=precision,
            min_bin_prop=min_bin_prop,
            include_missing=include_missing,
            equal_freq=equal_freq,
        )

    return selected_vars, model, perf_dict


def backward_xgbm(
    train_data: pd.DataFrame,
    varlist: List[str],
    dep: str,
    varreduct_params: Optional[Dict] = None,
    stopping_metric: str = "auc",
    seed: int = 42,
    num_boost_round: int = 200,
    early_stopping_rounds: int = 20,
    importance_type: str = "gain",
    cum_importance_threshold: float = 0.99,
    min_vars: int = 10,
    validation_data: Optional[pd.DataFrame] = None,
    test_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    ret_perf: bool = True,
    nbins: int = 10,
    precision: int = 5,
    min_bin_prop: float = 0.05,
    include_missing: bool = True,
    equal_freq: bool = True,
    ascending: bool = True,
    fillna: Optional[float] = None,
    spec_values: Optional[List] = None,
    monotone_constraints: Optional[Dict[str, int]] = None,
    weight_col: Optional[str] = None,
    validation_weight_col: Optional[str] = None,
    wgt_col: Optional[str] = None,
) -> Tuple:
    """
    Run backward variable elimination with an XGBoost model.

    Train an XGBoost model and screen the variables by a cumulative
    feature-importance threshold (backward variable elimination).

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset. Must contain the ``dep`` column and all feature columns in ``varlist``.
    varlist : list of str
        List of feature variables used for modeling.
    dep : str
        Name of the target column (binary 0/1 variable).
    varreduct_params : dict, optional
        XGBoost hyperparameter dictionary. Required parameters that are not specified use preset values.
    stopping_metric : str, default "auc"
        Evaluation metric for early stopping.
    seed : int, default 42
        Random seed.
    num_boost_round : int, default 200
        Maximum number of boosting rounds.
    early_stopping_rounds : int, default 20
        Number of early-stopping rounds.
    importance_type : str, default "gain"
        Feature importance type.
    cum_importance_threshold : float, default 0.99
        Cumulative feature-importance threshold.
    min_vars : int, default 10
        Minimum number of variables to keep.
    validation_data : pd.DataFrame, optional
        Validation dataset.
    test_data_dict : dict, optional
        Dictionary of test datasets.
    ret_perf : bool, default True
        Whether to return performance metrics.
    nbins : int, default 10
        Number of bins in the Gains table.
    precision : int, default 5
        Numeric precision.
    min_bin_prop : float, default 0.05
        Minimum bin proportion.
    include_missing : bool, default True
        Whether to include a missing-value bin.
    equal_freq : bool, default True
        Whether to use equal-frequency binning.
    ascending : bool, default True
        Whether to sort the Gains table in ascending order.
    fillna : float, optional
        Fill value for missing values.
    spec_values : list, optional
        List of special values.
    monotone_constraints : dict, optional
        Dictionary of monotone constraints, in the form ``{feature_name: 1 or -1}``.

    Returns
    -------
    tuple
        ``(selected_vars, model, perf_dict)``, or ``(selected_vars, model)`` when ``ret_perf`` is False.

    Raises
    ------
    TypeError
        If the input data is not a ``pandas.DataFrame``.
    ImportError
        If xgboost is not installed.

    Examples
    --------
    >>> selected_vars, model, perf = backward_xgbm(
    ...     train_data=train_df,
    ...     varlist=feature_cols,
    ...     dep='target',
    ...     validation_data=val_df
    ... )
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("Please install xgboost: pip install xgboost")

    if varreduct_params is None:
        varreduct_params = {}

    if test_data_dict is None:
        test_data_dict = {}

    if spec_values is None:
        spec_values = []

    if monotone_constraints is None:
        monotone_constraints = {}

    # Build the monotone constraint mapping for all variables
    mc_dict = {var: monotone_constraints.get(var, 0) for var in varlist}

    datain_all = OrderedDict()
    datain_all["mdl"] = train_data
    if validation_data is not None:
        datain_all["hd"] = validation_data
    datain_all.update(test_data_dict)

    # Check data format consistency
    try:
        for k, v in datain_all.items():
            assert isinstance(v, pd.DataFrame)
    except AssertionError:
        logging.warning("Please provide data in pandas.DataFrame format")
        raise TypeError("Please provide data in pandas.DataFrame format")

    # Preset parameters (to keep the model reproducible)
    hyperparams_preset = {
        'eval_metric': stopping_metric,
        'tree_method': 'exact',
        'booster': 'gbtree',
        'seed': seed,
        'monotone_constraints': mc_dict
    }

    # Fill in any missing required parameters
    lacked_params = [k for k in list(hyperparams_preset.keys()) if k not in list(varreduct_params.keys())]
    for param in lacked_params:
        varreduct_params[param] = hyperparams_preset[param]

    weight_col = weight_col or wgt_col

    # Build the XGBoost datasets
    train_weight = resolve_sample_weight(data=train_data, weight_col=weight_col, expected_len=len(train_data))
    xgb_train = xgb.DMatrix(train_data[varlist], label=train_data[dep], weight=train_weight)

    evals = [(xgb_train, "mdl")]
    if validation_data is not None:
        valid_weight = resolve_sample_weight(
            data=validation_data,
            weight_col=validation_weight_col or weight_col,
            expected_len=len(validation_data),
        )
        xgb_valid = xgb.DMatrix(validation_data[varlist], label=validation_data[dep], weight=valid_weight)
        evals.append((xgb_valid, "hd"))

    # Train the model
    evals_result = {}
    model = xgb.train(
        params=varreduct_params,
        dtrain=xgb_train,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=False
    )

    # Get the feature importance and screen the variables
    importance_raw = model.get_score(importance_type=importance_type)
    importance_df = pd.DataFrame(
        list(importance_raw.items()), columns=["feature", "importance"]
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    # Add features with zero importance
    missing_feats = [f for f in varlist if f not in importance_df["feature"].values]
    if missing_feats:
        zero_df = pd.DataFrame({"feature": missing_feats, "importance": 0.0})
        importance_df = pd.concat([importance_df, zero_df], ignore_index=True)

    importance_df["cum_importance"] = importance_df["importance"].cumsum() / (importance_df["importance"].sum() or 1)

    selected_idx = importance_df[importance_df["cum_importance"] <= cum_importance_threshold].index.tolist()

    if len(selected_idx) < min_vars:
        selected_idx = list(range(min(min_vars, len(importance_df))))

    selected_vars = importance_df.loc[selected_idx, "feature"].tolist()

    if not ret_perf:
        return selected_vars, model

    # Compute performance
    score_col = "_xgbm_score"
    perf_dict = {}

    for name, df in datain_all.items():
        df_score = df.copy()
        xgb_dmat = xgb.DMatrix(df_score[varlist])
        df_score[score_col] = model.predict(xgb_dmat)
        perf_dict[name] = _backward_perf_summary(
            df_score,
            split_name=name,
            dep=dep,
            score_col=score_col,
            weight_col=weight_col,
            validation_weight_col=validation_weight_col,
            nbins=nbins,
            precision=precision,
            min_bin_prop=min_bin_prop,
            include_missing=include_missing,
            equal_freq=equal_freq,
        )

    return selected_vars, model, perf_dict


class BackwardVariableEliminator:
    """
    Backward variable eliminator for LightGBM and XGBoost.

    Wrap the LightGBM/XGBoost backward variable elimination workflow,
    with support for multiple elimination rounds and result summaries.

    Parameters
    ----------
    train_data : pd.DataFrame
        Training dataset.
    varlist : list of str
        Initial list of feature variables.
    dep : str
        Name of the target column.
    model_type : str, default "lgbm"
        Model type, either "lgbm" or "xgbm".
    validation_data : pd.DataFrame, optional
        Validation dataset.
    test_data_dict : dict, optional
        Dictionary of test datasets.

    Examples
    --------
    >>> eliminator = BackwardVariableEliminator(
    ...     train_data=train_df,
    ...     varlist=feature_cols,
    ...     dep='target',
    ...     model_type='lgbm',
    ...     validation_data=val_df
    ... )
    >>> results = eliminator.run(n_rounds=5)
    """

    def __init__(
        self,
        train_data: pd.DataFrame,
        varlist: List[str],
        dep: str,
        model_type: str = "lgbm",
        validation_data: Optional[pd.DataFrame] = None,
        test_data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        weight_col: Optional[str] = None,
        validation_weight_col: Optional[str] = None,
        wgt_col: Optional[str] = None,
    ):
        self.train_data = train_data
        self.varlist = varlist
        self.dep = dep
        self.model_type = model_type.lower()
        self.validation_data = validation_data
        self.test_data_dict = test_data_dict or {}
        self.weight_col = weight_col or wgt_col
        self.validation_weight_col = validation_weight_col
        self._results = []

    def run(
        self,
        n_rounds: int = 5,
        varreduct_params: Optional[Dict] = None,
        stopping_metric: str = "auc",
        seed: int = 42,
        num_boost_round: int = 200,
        early_stopping_rounds: int = 20,
        importance_type: str = "gain",
        cum_importance_threshold: float = 0.99,
        min_vars: int = 10,
        ret_perf: bool = True,
        nbins: int = 10,
        **kwargs,
    ) -> List[Dict]:
        """
        Run multiple rounds of backward variable elimination.

        Parameters
        ----------
        n_rounds : int, default 5
            Number of elimination rounds.
        varreduct_params : dict, optional
            Model hyperparameters.
        stopping_metric : str, default "auc"
            Early-stopping metric.
        seed : int, default 42
            Random seed.
        num_boost_round : int, default 200
            Maximum number of boosting rounds.
        early_stopping_rounds : int, default 20
            Number of early-stopping rounds.
        importance_type : str, default "gain"
            Feature importance type.
        cum_importance_threshold : float, default 0.99
            Cumulative importance threshold.
        min_vars : int, default 10
            Minimum number of variables to keep.
        ret_perf : bool, default True
            Whether to return performance metrics.
        nbins : int, default 10
            Number of bins.

        Returns
        -------
        list of dict
            Results of each elimination round.
        """
        current_vars = self.varlist.copy()
        self._results = []

        backward_fn = backward_lgbm if self.model_type == "lgbm" else backward_xgbm

        for round_idx in range(1, n_rounds + 1):
            logging.info(f"[BackwardVariableEliminator] Round {round_idx}/{n_rounds}, vars={len(current_vars)}")

            result = backward_fn(
                train_data=self.train_data,
                varlist=current_vars,
                dep=self.dep,
                varreduct_params=copy.deepcopy(varreduct_params),
                stopping_metric=stopping_metric,
                seed=seed,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                importance_type=importance_type,
                cum_importance_threshold=cum_importance_threshold,
                min_vars=min_vars,
                validation_data=self.validation_data,
                test_data_dict=self.test_data_dict,
                ret_perf=ret_perf,
                nbins=nbins,
                weight_col=self.weight_col,
                validation_weight_col=self.validation_weight_col,
                **kwargs,
            )

            if ret_perf:
                selected_vars, model, perf_dict = result
            else:
                selected_vars, model = result
                perf_dict = {}

            round_result = {
                "round": round_idx,
                "n_vars_in": len(current_vars),
                "n_vars_out": len(selected_vars),
                "selected_vars": selected_vars,
                "model": model,
                "perf": perf_dict,
            }
            self._results.append(round_result)
            current_vars = selected_vars

            if len(current_vars) <= min_vars:
                logging.info(f"[BackwardVariableEliminator] Reached min_vars={min_vars}, stopping early.")
                break

        return self._results

    def get_final_vars(self) -> List[str]:
        """Return the final list of variables after elimination."""
        if not self._results:
            return self.varlist
        return self._results[-1]["selected_vars"]

    def get_summary(self) -> pd.DataFrame:
        """Return the summary table of each elimination round."""
        rows = []
        for r in self._results:
            rows.append({
                "round": r["round"],
                "n_vars_in": r["n_vars_in"],
                "n_vars_out": r["n_vars_out"],
                "vars_removed": r["n_vars_in"] - r["n_vars_out"],
            })
        return pd.DataFrame(rows)


class BackwardEliminationAnalyzer:
    """
    Analyzer for backward elimination results.

    Analyze and visualize the results of a ``BackwardVariableEliminator`` run.

    Parameters
    ----------
    results : list of dict
        Return value of ``BackwardVariableEliminator.run()``.

    Examples
    --------
    >>> analyzer = BackwardEliminationAnalyzer(results)
    >>> analyzer.plot_var_reduction()
    >>> final_vars = analyzer.get_stable_vars(top_n=20)
    """

    def __init__(self, results: List[Dict]):
        self.results = results

    def get_stable_vars(self, top_n: Optional[int] = None) -> List[str]:
        """
        Return the stable variables that are kept in every round.

        Parameters
        ----------
        top_n : int, optional
            Number of stable variables to return (the first N); None returns all of them.

        Returns
        -------
        list of str
        """
        if not self.results:
            return []

        stable = set(self.results[0]["selected_vars"])
        for r in self.results[1:]:
            stable &= set(r["selected_vars"])

        stable_list = sorted(stable)
        if top_n is not None:
            stable_list = stable_list[:top_n]
        return stable_list

    def plot_var_reduction(
        self,
        figsize: Tuple[int, int] = (8, 4),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot a line chart of the number of variables against the elimination round.

        Parameters
        ----------
        figsize : tuple, default (8, 4)
            Figure size.
        save_path : str, optional
            File path for saving the figure; if None, the figure is displayed directly.
        """
        rounds = [r["round"] for r in self.results]
        n_vars = [r["n_vars_out"] for r in self.results]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rounds, n_vars, marker="o", linewidth=2, color="#4C72B0")
        ax.set_xlabel("Elimination Round")
        ax.set_ylabel("Number of Variables Kept")
        ax.set_title("Backward Variable Elimination: Change in Variable Count")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def get_perf_trend(self, dataset: str = "mdl", metric: str = "IV") -> pd.DataFrame:
        """
        Return the trend of a performance metric over the rounds for a given dataset.

        Parameters
        ----------
        dataset : str, default "mdl"
            Dataset name, e.g. "mdl", "hd", "oot".
        metric : str, default "IV"
            Name of the performance metric column.

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for r in self.results:
            perf = r.get("perf", {})
            if dataset in perf and perf[dataset] is not None:
                summary = perf[dataset]
                val = None
                if isinstance(summary, dict):
                    val = summary.get(metric, None)
                elif isinstance(summary, pd.DataFrame) and metric in summary.columns and len(summary):
                    # BackwardVariableEliminator stores each split's get_perf_summary() frame
                    val = summary[metric].iloc[0]
                rows.append({"round": r["round"], metric: val})
        return pd.DataFrame(rows)
