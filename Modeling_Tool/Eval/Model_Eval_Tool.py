import logging
import os
import warnings
import numpy as np
import pandas as pd
from Modeling_Tool.Core.Binning_Tool import get_bin_range_list, super_binning
from Modeling_Tool.Core.utils import load_model, calc_iv, calc_woe
from Modeling_Tool._utils.frames import concat_non_empty
from .evaluate_model import evaluate_performance
from . import weighted_eval_utils as _weighted_eval

###################################################### Private Functions #############################################################

def _get_gains_table_scr(data, score, dep, nbins = 10, precision = 5, 
                         min_bin_prop = 0.05, include_missing = True, equal_freq = True, 
                         chi2_method = False, chi2_p = 0.95, init_equi_bins = 2000, 
                         fillna = -999999, spec_values = [], retSummary = False, 
                         tree_binning = False, random_state=42, ascending = False,
                         withSummary = False, add_func = None):
    """
    Compute the Gains table for the given score column.
    
    After binning the data, compute the statistics of each bin, including the number of
    samples, the bad rate, the cumulative number of good/bad samples, WOE, IV, etc.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    score : str
        Name of the score column.
    dep : str
        Name of the target variable (binary label, 0 and 1).
    nbins : int, default 10
        Number of bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default True
        Whether to include missing values.
    equal_freq : bool, default True
        True for equal-frequency binning, False for equal-width binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    chi2_p : float, default 0.95
        Significance level of the chi-square test.
    init_equi_bins : int, default 2000
        Initial number of equal-frequency bins.
    fillna : any, default -999999
        Fill value for missing values.
    spec_values : list, default []
        List of special values.
    retSummary : bool, default False
        Whether to return only the summary metrics.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    ascending : bool, default False
        Whether the bin order is ascending.
    withSummary : bool, default False
        Whether to include an overall summary row.
    add_func : callable, optional
        Custom statistics function.
    
    Returns
    -------
    pandas.DataFrame
        Gains table containing the statistics of each bin.
    """
    
    res, edges = super_binning(data = data, 
                               score = score, 
                               dep = dep, 
                               nbins = nbins, 
                               precision = precision, 
                               min_bin_prop = min_bin_prop, 
                               include_missing = include_missing, 
                               equal_freq = equal_freq, 
                               chi2_method = chi2_method, 
                               chi2_p = chi2_p, 
                               init_equi_bins = init_equi_bins, 
                               fillna = fillna, 
                               spec_values = spec_values, 
                               tree_binning = tree_binning, 
                               random_state = random_state, 
                               return_edges = True, 
                               bin_colnames = ("_bin_num", "_bin_range"),
                               ascending = ascending)
    
    res = res.copy()
    res["_smf_bad_ind"] = res[dep].eq(1).astype(np.int64)
    res["_smf_good_ind"] = res[dep].eq(0).astype(np.int64)
    grouped = res.groupby(["_bin_num", "_bin_range"], dropna=False)
    gains_table = grouped.agg(
        MIN=(score, "min"),
        MAX=(score, "max"),
        N=(dep, "size"),
        PERF_CNT=(dep, "count"),
        AVG_SCORE=(score, "mean"),
        UNIQUE_SCORE=(score, "nunique"),
        N_BAD=("_smf_bad_ind", "sum"),
        N_GOOD=("_smf_good_ind", "sum"),
    )
    perf_denom = gains_table["PERF_CNT"].replace(0, np.nan)
    gains_table["PROP"] = gains_table["N"] / max(len(res), 1)
    gains_table["AVG_BAD"] = (gains_table["N_BAD"] / perf_denom).fillna(0.0)
    gains_table["AVG_GOOD"] = (gains_table["N_GOOD"] / perf_denom).fillna(0.0)
    gains_table["LIFT"] = gains_table["AVG_BAD"] / res[dep].mean()
    gains_table = gains_table[
        [
            "MIN", "MAX", "N", "PROP", "PERF_CNT", "AVG_SCORE",
            "UNIQUE_SCORE", "AVG_BAD", "AVG_GOOD", "N_BAD", "N_GOOD", "LIFT",
        ]
    ]

    gains_table["BAD_PCT_IN_EACH_BIN"] = gains_table["N_BAD"] / gains_table["N_BAD"].sum()
    gains_table["GOOD_PCT_IN_EACH_BIN"] = gains_table["N_GOOD"] / gains_table["N_GOOD"].sum()
    
    gains_table["N_CUM_BAD"] = gains_table["N_BAD"].cumsum()
    gains_table["N_CUM_GOOD"] = gains_table["N_GOOD"].cumsum()

    gains_table["CUM_BAD_PCT"] = gains_table["N_CUM_BAD"]/gains_table["N_BAD"].sum()
    gains_table["CUM_GOOD_PCT"] = gains_table["N_CUM_GOOD"]/gains_table["N_GOOD"].sum()
    gains_table["KS_PER_BIN"] = np.abs((gains_table["CUM_BAD_PCT"] - gains_table["CUM_GOOD_PCT"]))


    gains_table["TRUE_BAD_SHIFT"] = (gains_table['AVG_BAD'].shift(1) / gains_table['AVG_BAD'] - 1) if not ascending else (gains_table['AVG_BAD'] / gains_table['AVG_BAD'].shift(1) - 1) 
    gains_table["RANK_ORDER_BUMP"] = gains_table["TRUE_BAD_SHIFT"].lt(0).astype(int)
    
    gains_table["WOE"] = calc_woe(data = gains_table, bad_pct = "BAD_PCT_IN_EACH_BIN", good_pct = "GOOD_PCT_IN_EACH_BIN")
    gains_table["IV"] = calc_iv(data = gains_table, bad_pct = "BAD_PCT_IN_EACH_BIN", good_pct = "GOOD_PCT_IN_EACH_BIN")
    

    if add_func is not None:
        # Explicitly select all columns (including the grouping columns): add_func still sees _bin_num / _bin_range,
        # and the pandas deprecation warning about apply including the grouping columns by default is not triggered
        gains_table_add = res.groupby(["_bin_num", "_bin_range"], dropna=False)[res.columns.unique().tolist()].apply(add_func)
        gains_table = gains_table.merge(gains_table_add, right_index = True, left_index = True, how = 'left')
    
    if retSummary:
        summ_metrics = ["N_BUMP", "MIN_RISK_DEP", "MAX_RISK_DEP", "KS_IN_GAINS", "LIFT_IN_GAINS", "IV", "N_BINS"]
        res_summary = {
            "N_BUMP": gains_table["RANK_ORDER_BUMP"].sum(),
            "MIN_RISK_DEP": gains_table["TRUE_BAD_SHIFT"].round(4).min(),
            "MAX_RISK_DEP": gains_table["TRUE_BAD_SHIFT"].round(4).max(),
            "KS_IN_GAINS": gains_table["KS_PER_BIN"].round(4).max(),
            "LIFT_IN_GAINS": gains_table["LIFT"].round(4).max(),
            "IV": gains_table["IV"].replace([np.inf, -np.inf], 0).sum(),
            "N_BINS": gains_table.shape[0]
        }

        res_summary = pd.DataFrame(res_summary, index = [0])
        res_summary["LABEL_NAME"] = dep
        res_summary["SCR_NAME"] = score
        return res_summary[summ_metrics]
    
    if withSummary:
        grand_total = {"MIN": res[score].min(),
                       "MAX": res[score].max(),
                       "N": res.shape[0],
                       "AVG_SCORE": res[score].mean(),
                       "UNIQUE_SCORE": res[score].nunique(),
                       "AVG_BAD": res[dep].mean(),
                       "AVG_GOOD": 1 - res[dep].mean(),
                       "N_BAD": res[dep].sum(),
                       "N_GOOD": (res[dep] == 0).sum(),
                       "BAD_PCT_IN_EACH_BIN": gains_table["BAD_PCT_IN_EACH_BIN"].sum(),
                       "GOOD_PCT_IN_EACH_BIN": gains_table["GOOD_PCT_IN_EACH_BIN"].sum(),
                       "N_CUM_BAD": res[dep].sum(),
                       "N_CUM_GOOD": (res[dep] == 0).sum(),
                       "CUM_BAD_PCT": 1,
                       "CUM_GOOD_PCT": 1,
                       "KS_PER_BIN": gains_table["KS_PER_BIN"].max(),
                       "LIFT": 1,
                       "TRUE_BAD_SHIFT": 1,
                       "RANK_ORDER_BUMP": gains_table["RANK_ORDER_BUMP"].sum(),
                       "WOE": gains_table["WOE"].mean(),
                       "IV": gains_table["IV"].sum(),
                       "PROP": res.shape[0] / res.shape[0],
                       "PERF_CNT": gains_table["PERF_CNT"].sum()}
        grand_total = pd.DataFrame(grand_total, index = [("Grand Summary", "")])
        gains_table = pd.concat([gains_table, grand_total])
        
        gains_table.index.names = ("_bin_num", "_bin_range")
    
    return gains_table


def _get_gains_table_single(data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, 
                            score = None, model = None, varlist = None, equal_freq = True, chi2_method = False,
                            chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], retSummary = False,
                            tree_binning = False, random_state=42, ascending = False,
                            withSummary = False, add_func = None):
    """
    Compute the Gains table of a single model.
    
    Compute the Gains table from the given score column or from the model predictions.
    The given score column takes precedence; if none is given, the predicted probabilities
    of the model are used.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    dep : str
        Name of the target variable.
    nbins : int, default 10
        Number of bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default True
        Whether to include missing values.
    score : str, optional
        Name of the score column.
    model : sklearn-like model, optional
        Machine learning model (used when score is None).
    varlist : list, optional
        List of model features.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    chi2_p : float, default 0.95
        Significance level of the chi-square test.
    init_equi_bins : int, default 100
        Initial number of equal-frequency bins.
    fillna : any, default -999999
        Fill value for missing values.
    spec_values : list, default []
        List of special values.
    retSummary : bool, default False
        Whether to return only the summary metrics.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    ascending : bool, default False
        Whether the bin order is ascending.
    withSummary : bool, default False
        Whether to include an overall summary row.
    add_func : callable, optional
        Custom statistics function.
    
    Returns
    -------
    pandas.DataFrame or int
        Gains table; -1/-2/-3 is returned if required parameters are missing.
    """
    
    if score is None and model is None and varlist is None:
        return -1
    
    if score is None and model is None:
        return -2
    
    if score is None and varlist is None:
        return -3
    
    if score is None:
        data = data.copy()  # the scored column is ours; do not leave it on the caller's frame
        data['_mdl_scr'] = model.predict_proba(data.loc[:, varlist])[:, 1]
        score = '_mdl_scr'
        
    res = _get_gains_table_scr(data = data, 
                              score = score, 
                              dep = dep, 
                              nbins = nbins, 
                              precision = precision, 
                              min_bin_prop=min_bin_prop, 
                              include_missing = include_missing, 
                              equal_freq = equal_freq, 
                              chi2_method = chi2_method, 
                              chi2_p = chi2_p, 
                              init_equi_bins = init_equi_bins, 
                              fillna = fillna, 
                              spec_values = spec_values, 
                              retSummary = retSummary, 
                              tree_binning = tree_binning,
                              random_state = random_state,
                              ascending = ascending,
                              withSummary = withSummary,
                              add_func = add_func)
    return res


def _get_perf_summary_single(train, 
                             validation, 
                             oot, 
                             tgt_name, 
                             scr_name = None,
                             model = None, 
                             feature_cols = None, 
                             fig_save_path = None, 
                             rpt_save_path = None,
                             to_show = False, 
                             display = True,
                             dist_bins = 20, 
                             pct_bins = 10,
                             precision = 5, 
                             min_bin_prop = 0.05,
                             include_missing = False, 
                             equal_freq = True,
                             chi2_method = False, 
                             init_equi_bins = 1000, 
                             chi2_p = 0.9, 
                             tree_binning = False, 
                             random_state = 42, 
                             gains_table = False):
    """
    Compute the performance evaluation summary of a single model.
    
    Evaluate the model performance on the training, validation and OOT samples, including
    metrics such as AUC, KS and Lift, and optionally generate the Gains table.
    
    Parameters
    ----------
    train : pandas.DataFrame, optional
        Training dataset.
    validation : pandas.DataFrame, optional
        Validation dataset.
    oot : pandas.DataFrame, optional
        Out-of-time (OOT) dataset.
    tgt_name : str
        Name of the target variable.
    scr_name : str, optional
        Name of the score column.
    model : sklearn-like model, optional
        Machine learning model.
    feature_cols : list, optional
        List of model features.
    fig_save_path : str, optional
        Path to save the figure.
    rpt_save_path : str, optional
        Path to save the report.
    to_show : bool, default False
        Whether to display the figures.
    display : bool, default True
        Whether to print the results.
    dist_bins : int, default 20
        Number of bins for the score distribution.
    pct_bins : int, default 10
        Number of percentile bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default False
        Whether to include missing values.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    init_equi_bins : int, default 1000
        Initial number of equal-frequency bins.
    chi2_p : float, default 0.9
        Significance level of the chi-square test.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    gains_table : bool, default True
        Whether to compute the Gains table.
    
    Returns
    -------
    pandas.DataFrame or int
        Performance evaluation summary table; -1/-2/-3 is returned if required parameters are missing.
    """
    
    if scr_name is None and model is None and feature_cols is None:
        return -1
    
    if scr_name is None and model is None:
        return -2
    
    if scr_name is None and feature_cols is None:
        return -3
    
    if model is not None and feature_cols is not None:
        ins_prob = model.predict_proba(train.loc[:, feature_cols])[:, 1] if train is not None else None
        oos_prob = model.predict_proba(validation.loc[:, feature_cols])[:, 1] if validation is not None else None
        oot_prob = model.predict_proba(oot.loc[:, feature_cols])[:, 1] if oot is not None else None
    
    if scr_name is not None:
        ins_prob = train[scr_name] if train is not None else None
        oos_prob = validation[scr_name] if validation is not None else None
        oot_prob = oot[scr_name] if oot is not None else None
    
    datasets = {}
    if ins_prob is not None:
        datasets['ins'] = {"y_true": train[tgt_name],       "y_score": ins_prob}
    if oos_prob is not None:
        datasets['oos'] = {"y_true": validation[tgt_name],  "y_score": oos_prob}
    if oot_prob is not None:
        datasets['oot'] = {"y_true": oot[tgt_name],         "y_score": oot_prob}
        
    model_eval_result_df = evaluate_performance(
        datasets=datasets, 
        dist_bins=dist_bins, 
        pct_bins=pct_bins, 
        square_figsize=5,
        to_show=to_show, 
        save_path = fig_save_path,
        gains_table = gains_table,
        equal_freq = equal_freq
    )
    
#     print(model_eval_result_df)
#     if model_eval_result_df.shape[0] == 0:
#         return model_eval_result_df
#     print(model_eval_result_df.columns)

#     print(pct_bins)
    quantile = np.ceil((10 / pct_bins) * 10) if (np.ceil((10 / pct_bins) * 10) - ((10 / pct_bins) * 10)) < 0.5 else np.floor((10 / pct_bins) * 10)
#     print(quantile)
    quantile = int(quantile)
#     display(model_eval_result_df)

    import re
#     btm_str = [x for x in model_eval_result_df.columns if x.startswith("Btm")][0]
#     top_str = [x for x in model_eval_result_df.columns if x.startswith("Top")][0]
    btm_cols = [x for x in model_eval_result_df.columns if x.startswith("Btm")]
    top_cols = [x for x in model_eval_result_df.columns if x.startswith("Top")]
    
    if btm_cols and top_cols:
        btm_str = btm_cols[0]
        top_str = top_cols[0]
        # Extract the number
        numbers = re.findall(r'\d+', btm_str)
        if numbers:
            quantile = int(numbers[0])
        else:
            # Default value, e.g. the reciprocal of pct_bins?
            quantile = pct_bins  # or another reasonable default
    else:
        # If these columns are missing, the data is insufficient or they were not generated: skip the remaining computation or assign a default value
        # Here we may skip the computation of the Lift columns and return model_eval_result_df directly
        return model_eval_result_df
            
#     quantile = int(re.findall(r'\d+', btm_str)[0])
    
    model_eval_result_df[f"Btm{quantile}%_Lift"] = model_eval_result_df[btm_str]/model_eval_result_df["avgTrue"]
    model_eval_result_df[f"Top{quantile}%_Lift"] = model_eval_result_df[top_str]/model_eval_result_df["avgTrue"]
    
    model_eval_result_df["AUC_Shift"] = model_eval_result_df["AUC"].shift(1)/model_eval_result_df["AUC"] - 1
    model_eval_result_df["KS_Shift"] = model_eval_result_df["KS"].shift(1)/model_eval_result_df["KS"] - 1
    
    ### Gains Table Summary
    gains_table_cols = ['N_BUMP', 'MIN_RISK_DEP', 'MAX_RISK_DEP', 'KS_IN_GAINS', 'LIFT_IN_GAINS', 'IV', 'N_BINS']
    if train is not None:
        ins_gains = get_gains_table(data = train, 
                                    dep = tgt_name, 
                                    nbins=pct_bins, 
                                    precision=precision, 
                                    min_bin_prop=min_bin_prop, 
                                    include_missing=include_missing, 
                                    score=scr_name, 
                                    equal_freq=equal_freq, 
                                    chi2_method=chi2_method, 
                                    model = model, 
                                    varlist = feature_cols,
                                    init_equi_bins = init_equi_bins, 
                                    chi2_p = chi2_p, 
                                    retSummary=True, 
                                    tree_binning = tree_binning, 
                                    random_state = random_state)
    else:
        ins_gains = pd.DataFrame([], columns = gains_table_cols)
    
    if validation is not None:
        oos_gains = get_gains_table(data = validation, 
                                    dep = tgt_name, 
                                    nbins = pct_bins, 
                                    precision=precision, 
                                    min_bin_prop=min_bin_prop, 
                                    include_missing=include_missing, 
                                    score=scr_name, 
                                    equal_freq=equal_freq, 
                                    chi2_method=chi2_method, 
                                    model = model, 
                                    varlist = feature_cols,
                                    init_equi_bins=init_equi_bins, 
                                    chi2_p=chi2_p, 
                                    retSummary=True, 
                                    tree_binning = tree_binning, 
                                    random_state = random_state)
    else:
        oos_gains = pd.DataFrame([], columns = gains_table_cols)

    if oot is not None:
        oot_gains = get_gains_table(data = oot, 
                                    dep = tgt_name, 
                                    nbins=pct_bins, 
                                    precision=precision, 
                                    min_bin_prop=min_bin_prop, 
                                    include_missing=include_missing, 
                                    score=scr_name, 
                                    equal_freq=equal_freq, 
                                    chi2_method=chi2_method, 
                                    model = model, 
                                    varlist = feature_cols,
                                    init_equi_bins=init_equi_bins, 
                                    chi2_p=chi2_p, 
                                    retSummary=True, 
                                    tree_binning = tree_binning, 
                                    random_state = random_state)
    else:
        oot_gains = pd.DataFrame([], columns = gains_table_cols)
        
    ins_gains['index'] = 'ins'
    oos_gains['index'] = 'oos'
    oot_gains['index'] = 'oot'
    
    # Empty placeholder tables for missing sample sets are excluded from the concatenation, otherwise they turn the integer columns (N_BUMP / N_BINS) into object dtype
    gains_summ = concat_non_empty([ins_gains, oos_gains, oot_gains])
    
    model_eval_result_df = model_eval_result_df.merge(gains_summ, on = ['index'], how = 'left')

    if display:
        from IPython.display import display
        display(model_eval_result_df)
        
    if rpt_save_path:
        model_eval_result_df.to_csv(rpt_save_path, index=False)
        
    return model_eval_result_df


def _get_gains_by_custom_metrics_scr(data, score, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, equal_freq = True, 
                                    chi2_method = False, chi2_p = 0.95, init_equi_bins = 2000, fillna = -999999, spec_values = [],
                                    tree_binning = False, random_state=42,
                                    eval_metrics = ["age", "monthly_income", "education"], metric_agg_func = "mean", 
                                    ascending = False, withSummary = False):
    """
    Compute the Gains table for the given score column, including aggregated statistics of custom metrics.
    
    After binning the data, compute the basic statistics of each bin as well as
    the aggregated values of the custom metrics (e.g. the mean).
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    score : str
        Name of the score column.
    dep : str
        Name of the target variable.
    nbins : int, default 10
        Number of bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default True
        Whether to include missing values.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    chi2_p : float, default 0.95
        Significance level of the chi-square test.
    init_equi_bins : int, default 2000
        Initial number of equal-frequency bins.
    fillna : any, default -999999
        Fill value for missing values.
    spec_values : list, default []
        List of special values.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    eval_metrics : list, default ["age", "monthly_income", "education"]
        List of custom metrics to compute.
    metric_agg_func : str or callable, default "mean"
        Aggregation function for the custom metrics.
    ascending : bool, default False
        Whether the bin order is ascending.
    withSummary : bool, default False
        Whether to include an overall summary row.
    
    Returns
    -------
    pandas.DataFrame
        Gains table containing the basic statistics and the aggregated values of the custom metrics.
    """
    
    res, edges = super_binning(data = data, 
                               score = score, 
                               dep = dep, 
                               nbins = nbins, 
                               precision = precision, 
                               min_bin_prop = min_bin_prop, 
                               include_missing = include_missing, 
                               equal_freq = equal_freq, 
                               chi2_method = chi2_method, 
                               chi2_p = chi2_p, 
                               init_equi_bins = init_equi_bins, 
                               fillna = fillna, 
                               spec_values = spec_values, 
                               tree_binning = tree_binning, 
                               random_state = random_state, 
                               return_edges = True, 
                               bin_colnames = ("_bin_num", "_bin_range"),
                               ascending = ascending)


    # Compute the statistics of each bin
    gains_table_info = res.groupby(["_bin_num", "_bin_range"], dropna = False)\
                     .agg(MIN = (score, "min"),
                          MAX = (score, "max"),
                          N = ("_bin_num", "count"),
                          AVG_SCORE = (score, "mean"),
                          AVG_BAD = (dep, "mean"),
                          N_BAD = (dep, "sum"),
                          N_GOOD = (dep, lambda x: (x==0).sum()))

    gains_table_metric = res.groupby(["_bin_num", "_bin_range"], dropna = False).agg({metric: metric_agg_func for metric in eval_metrics})

    fnl_res = gains_table_info.merge(gains_table_metric, left_index = True, right_index = True)
    
    if withSummary:
        grand_total = {"MIN": res[score].min(),
                        "MAX": res[score].max(),
                        "N": res.shape[0],
                        "AVG_SCORE": res[score].mean(),
                        "AVG_BAD": res[dep].mean(),
                        "N_BAD": res[dep].sum(),
                        "N_GOOD": (res[dep] == 0).sum()}
        grand_total.update({x: res[x].mean() for x in eval_metrics})
        grand_total = pd.DataFrame(grand_total, index = [("Grand Summary", "")])
        fnl_res = pd.concat([fnl_res, grand_total])

        fnl_res.index.names = ("_bin_num", "_bin_range")

    return fnl_res


def _get_cust_gains_table_single(data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, 
                                   score = None, model = None, varlist = None, equal_freq = True, chi2_method = False,
                                   chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], 
                                   tree_binning = False, random_state=42, ascending = True,
                                   eval_metrics = ["age", "monthly_income", "education"], metric_agg_func = "mean",
                                  withSummary = False):
    """
    Compute the Gains table with custom metrics of a single model.
    
    Compute the Gains table, including the aggregated statistics of the custom metrics, from the given
    score column or from the model predictions.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    dep : str
        Name of the target variable.
    nbins : int, default 10
        Number of bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default True
        Whether to include missing values.
    score : str, optional
        Name of the score column.
    model : sklearn-like model, optional
        Machine learning model.
    varlist : list, optional
        List of model features.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    chi2_p : float, default 0.95
        Significance level of the chi-square test.
    init_equi_bins : int, default 100
        Initial number of equal-frequency bins.
    fillna : any, default -999999
        Fill value for missing values.
    spec_values : list, default []
        List of special values.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    ascending : bool, default True
        Whether the bin order is ascending.
    eval_metrics : list, default ["age", "monthly_income", "education"]
        List of custom metrics to compute.
    metric_agg_func : str or callable, default "mean"
        Aggregation function for the custom metrics.
    withSummary : bool, default False
        Whether to include an overall summary row.
    
    Returns
    -------
    pandas.DataFrame or int
        Gains table; -1/-2/-3 is returned if required parameters are missing.
    """
    
    if score is None and model is None and varlist is None:
        return -1
    
    if score is None and model is None:
        return -2
    
    if score is None and varlist is None:
        return -3
    
    if score is None:
        data = data.copy()  # the scored column is ours; do not leave it on the caller's frame
        data['_mdl_scr'] = model.predict_proba(data.loc[:, varlist])[:, 1]
        score = '_mdl_scr'
        
    res = _get_gains_by_custom_metrics_scr(data = data, 
                                          score = score, 
                                          dep = dep, 
                                          nbins = nbins, 
                                          precision = precision, 
                                          min_bin_prop=min_bin_prop, 
                                          include_missing = include_missing, 
                                          equal_freq = equal_freq, 
                                          chi2_method = chi2_method, 
                                          chi2_p = chi2_p, 
                                          init_equi_bins = init_equi_bins, 
                                          fillna = fillna, 
                                          spec_values = spec_values, 
                                          eval_metrics = eval_metrics,
                                          tree_binning = tree_binning,
                                          random_state = random_state,
                                          metric_agg_func = metric_agg_func,
                                          ascending = ascending,
                                          withSummary = withSummary)
    return res

###################################################### Private Functions (End) #############################################################


def get_gains_table(data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, 
                    score = None, model = None, varlist = None, equal_freq = True, chi2_method = False,
                    grp_name = None, min_data_size = 100, grp_colname = None, sync_range = True,
                    chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], retSummary = False,
                    tree_binning = False, random_state=42, ascending = False, withSummary = False, wholeGroup = False, 
                    add_func = None, weight_col = None, weighted_binning = None):
    """
    Compute the grouped Gains table.
    
    Split the data into groups by the grouping column and compute the Gains table of each group separately.
    If no grouping column is specified, compute the overall Gains table.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    dep : str
        Name of the target variable.
    nbins : int, default 10
        Number of bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default True
        Whether to include missing values.
    score : str, optional
        Name of the score column.
    model : sklearn-like model, optional
        Machine learning model.
    varlist : list, optional
        List of model features.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    grp_name : str, optional
        Name of the grouping column.
    min_data_size : int, default 100
        Minimum number of samples per group.
    grp_colname : str, optional
        Name of the group column in the output.
    sync_range : bool, default True
        Whether to synchronize the bin boundaries across groups.
    chi2_p : float, default 0.95
        Significance level of the chi-square test.
    init_equi_bins : int, default 100
        Initial number of equal-frequency bins.
    fillna : any, default -999999
        Fill value for missing values.
    spec_values : list, default []
        List of special values.
    retSummary : bool, default False
        Whether to return only the summary metrics.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    ascending : bool, default False
        Whether the bin order is ascending.
    withSummary : bool, default False
        Whether to include an overall summary row.
    wholeGroup : bool, default False
        Whether to use all the data for binning.
    add_func : callable, optional
        Custom statistics function.
    weight_col : str, optional
        Name of the sample weight column; when provided and ``grp_name`` is not given, the Gains table is
        aggregated by weight (``N`` in the output is the sum of the weights and ``N_RAW`` is the number of rows).
    weighted_binning : bool, optional
        If True, use equal-frequency binning by cumulative weight; if False, bin by the number of rows (default).

    Returns
    -------
    pandas.DataFrame
        Grouped Gains table.
    """
    
    if weight_col is not None and grp_name is None:
        work_data = data.copy()
        work_score = score
        if work_score is None:
            if model is None or varlist is None:
                return _get_gains_table_single(
                    data=data,
                    dep=dep,
                    nbins=nbins,
                    precision=precision,
                    min_bin_prop=min_bin_prop,
                    include_missing=include_missing,
                    score=score,
                    model=model,
                    varlist=varlist,
                    equal_freq=equal_freq,
                    chi2_method=chi2_method,
                    chi2_p=chi2_p,
                    init_equi_bins=init_equi_bins,
                    fillna=fillna,
                    spec_values=spec_values,
                    retSummary=retSummary,
                    tree_binning=tree_binning,
                    random_state=random_state,
                    ascending=ascending,
                    withSummary=withSummary,
                    add_func=add_func,
                )
            work_score = "_mdl_scr"
            work_data[work_score] = model.predict_proba(work_data.loc[:, varlist])[:, 1]
        gains = _weighted_eval.get_gains_table(
            work_data,
            dep,
            work_score,
            nbins=nbins,
            weight_col=weight_col,
            weighted_binning=weighted_binning,
            ascending=ascending,
        )
        if retSummary:
            return pd.DataFrame({
                "N_BUMP": [gains["RANK_ORDER_BUMP"].sum()],
                "MIN_RISK_DEP": [gains["TRUE_BAD_SHIFT"].round(4).min()],
                "MAX_RISK_DEP": [gains["TRUE_BAD_SHIFT"].round(4).max()],
                "KS_IN_GAINS": [gains["KS_PER_BIN"].round(4).max()],
                "LIFT_IN_GAINS": [gains["LIFT"].round(4).max()],
                "IV": [gains["IV"].replace([np.inf, -np.inf], 0).sum()],
                "N_BINS": [gains.shape[0]],
            })
        return gains

    if grp_colname is None:
        grp_colname = grp_name
        
    if grp_name is None:
        fnl_df = _get_gains_table_single(data = data, 
                                         dep = dep, 
                                         nbins = nbins, 
                                         precision = precision, 
                                         min_bin_prop = min_bin_prop, 
                                         include_missing = include_missing, 
                                         score = score, 
                                         model = model, 
                                         varlist = varlist, 
                                         equal_freq = equal_freq, 
                                         chi2_method = chi2_method,
                                         chi2_p = chi2_p, 
                                         init_equi_bins = init_equi_bins, 
                                         fillna = fillna, 
                                         spec_values = spec_values,
                                         retSummary = retSummary, 
                                         tree_binning = tree_binning, 
                                         random_state = random_state,
                                         ascending = ascending,
                                         withSummary = withSummary,
                                         add_func = add_func)
        return fnl_df
    
    withSummary = False
    grp_list = data[grp_name].sort_values(ascending = True).unique().tolist()
    valid_grp_list = [x for x in grp_list if data[data[grp_name].isin([x])].shape[0] >= min_data_size]
    
    if len(valid_grp_list) == 0:
        fnl_df = _get_gains_table_single(data = data, 
                                         dep = dep, 
                                         nbins = nbins, 
                                         precision = precision, 
                                         min_bin_prop = min_bin_prop, 
                                         include_missing = include_missing, 
                                         score = score, 
                                         model = model, 
                                         varlist = varlist, 
                                         equal_freq = equal_freq, 
                                         chi2_method = chi2_method,
                                         chi2_p = chi2_p, 
                                         init_equi_bins = init_equi_bins, 
                                         fillna = fillna, 
                                         spec_values = spec_values,
                                         retSummary = retSummary, 
                                         tree_binning = tree_binning, 
                                         random_state = random_state,
                                         ascending = ascending,
                                         withSummary = withSummary,
                                         add_func = add_func)
        fnl_df[grp_colname] = np.nan
        col_index = pd.MultiIndex.from_tuples([], names=['_bin_num', '_bin_range'])
        fnl_df = pd.DataFrame([], columns = fnl_df.columns, index = col_index)
        return fnl_df
    
    first_grp = valid_grp_list[0]
    data_grp = data[data[grp_name].isin([first_grp])]
    
    grp_for_binning = data_grp if not wholeGroup else data
    
    if sync_range and retSummary:
        
        full_gains = _get_gains_table_single(data = grp_for_binning, 
                                            dep = dep, 
                                            nbins = nbins, 
                                            precision = precision, 
                                            min_bin_prop = min_bin_prop, 
                                            include_missing = include_missing, 
                                            score = score, 
                                            model = model, 
                                            varlist = varlist, 
                                            equal_freq = equal_freq, 
                                            chi2_method = chi2_method,
                                            chi2_p = chi2_p, 
                                            init_equi_bins = init_equi_bins, 
                                            fillna = fillna, 
                                            spec_values = spec_values,
                                            retSummary = False, 
                                            tree_binning = tree_binning, 
                                            random_state = random_state,
                                            ascending = ascending,
                                            withSummary = withSummary,
                                            add_func = add_func)
        
        bin_range = get_bin_range_list(full_gains.reset_index())
        bin_range = bin_range + [-np.inf, np.inf]
        bin_range = sorted(list(set(bin_range)))
    
        fnl_df = _get_gains_table_single(data = data_grp, 
                                         dep = dep, 
                                         nbins = nbins, 
                                         precision = precision, 
                                         min_bin_prop = min_bin_prop, 
                                         include_missing = include_missing, 
                                         score = score, 
                                         model = model, 
                                         varlist = varlist, 
                                         equal_freq = equal_freq, 
                                         chi2_method = chi2_method,
                                         chi2_p = chi2_p, 
                                         init_equi_bins = init_equi_bins, 
                                         fillna = fillna, 
                                         spec_values = spec_values,
                                         retSummary = retSummary, 
                                         tree_binning = tree_binning, 
                                         random_state = random_state,
                                         ascending = ascending,
                                         withSummary = withSummary,
                                         add_func = add_func)

        fnl_df[grp_colname] = first_grp

        nbins = bin_range
        
        equal_freq = False
        chi2_method = False
        
    if sync_range and not retSummary:
        
        fnl_df = _get_gains_table_single(data = data_grp, 
                                        dep = dep, 
                                        nbins = nbins, 
                                        precision = precision, 
                                        min_bin_prop = min_bin_prop, 
                                        include_missing = include_missing, 
                                        score = score, 
                                        model = model, 
                                        varlist = varlist, 
                                        equal_freq = equal_freq, 
                                        chi2_method = chi2_method,
                                        chi2_p = chi2_p, 
                                        init_equi_bins = init_equi_bins, 
                                        fillna = fillna, 
                                        spec_values = spec_values,
                                        retSummary = retSummary, 
                                        tree_binning = tree_binning, 
                                        random_state = random_state,
                                        ascending = ascending,
                                        withSummary = withSummary,
                                        add_func = add_func)

        fnl_df[grp_colname] = first_grp

        nbins = get_bin_range_list(fnl_df.reset_index())
        nbins = nbins + [-np.inf, np.inf]
        nbins = sorted(list(set(nbins)))
        
        equal_freq = False
        chi2_method = False
        
    if (not sync_range and not retSummary) or (not sync_range and retSummary):
        
        fnl_df = _get_gains_table_single(data = data_grp, 
                                         dep = dep, 
                                         nbins = nbins, 
                                         precision = precision, 
                                         min_bin_prop = min_bin_prop, 
                                         include_missing = include_missing, 
                                         score = score, 
                                         model = model, 
                                         varlist = varlist, 
                                         equal_freq = equal_freq, 
                                         chi2_method = chi2_method,
                                         chi2_p = chi2_p, 
                                         init_equi_bins = init_equi_bins, 
                                         fillna = fillna, 
                                         spec_values = spec_values,
                                         retSummary = retSummary, 
                                         tree_binning = tree_binning, 
                                         random_state = random_state,
                                         ascending = ascending,
                                         withSummary = withSummary,
                                         add_func = add_func)

        fnl_df[grp_colname] = first_grp
    
    i = 1
    while i < len(valid_grp_list):
        
        grp = valid_grp_list[i]
        data_grp = data[data[grp_name].isin([grp])]

        perf_res = _get_gains_table_single(data = data_grp, 
                                           dep = dep, 
                                           nbins = nbins, 
                                           precision = precision, 
                                           min_bin_prop = min_bin_prop, 
                                           include_missing = include_missing, 
                                           score = score, 
                                           model = model, 
                                           varlist = varlist, 
                                           equal_freq = equal_freq, 
                                           chi2_method = chi2_method,
                                           chi2_p = chi2_p, 
                                           init_equi_bins = init_equi_bins, 
                                           fillna = fillna, 
                                           spec_values = spec_values,
                                           retSummary = retSummary, 
                                           tree_binning = tree_binning, 
                                           random_state = random_state,
                                           ascending = ascending,
                                           withSummary = withSummary,
                                           add_func = add_func)
    
        perf_res[grp_colname] = grp

        fnl_df = pd.concat([fnl_df, perf_res])
        i += 1
            
    return fnl_df


def get_perf_summary(train, validation, oot, tgt_name, 
                     scr_name = None,
                     model = None, 
                     feature_cols = None, 
                     fig_save_path = None, 
                     rpt_save_path = None,
                     to_show = False, 
                     display = True,
                     dist_bins = 20, 
                     pct_bins = 10, 
                     precision = 5, 
                     min_bin_prop = 0.05,
                     include_missing = False, 
                     equal_freq = True,
                     chi2_method = False, 
                     init_equi_bins = 1000, 
                     chi2_p = 0.9,
                     oot_grp_name = None, 
                     min_data_size = 100, 
                     grp_colname = None,
                     tree_binning = False, 
                     random_state = 42,
                     gains_table = False,
                     weight_col = None):
    """
    Compute the grouped performance evaluation summary.
    
    Evaluate the performance on the training, validation and OOT samples by group: the data are
    split by the ``oot_grp_name`` column and the performance metrics of each group are computed separately.
    
    Parameters
    ----------
    train : pandas.DataFrame, optional
        Training dataset.
    validation : pandas.DataFrame, optional
        Validation dataset.
    oot : pandas.DataFrame, optional
        Out-of-time (OOT) dataset.
    tgt_name : str
        Name of the target variable.
    scr_name : str, optional
        Name of the score column.
    model : sklearn-like model, optional
        Machine learning model.
    feature_cols : list, optional
        List of model features.
    fig_save_path : str, optional
        Path to save the figure.
    rpt_save_path : str, optional
        Path to save the report.
    to_show : bool, default False
        Whether to display the figures.
    display : bool, default True
        Whether to print the results.
    dist_bins : int, default 20
        Number of bins for the score distribution.
    pct_bins : int, default 10
        Number of percentile bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default False
        Whether to include missing values.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    init_equi_bins : int, default 1000
        Initial number of equal-frequency bins.
    chi2_p : float, default 0.9
        Significance level of the chi-square test.
    oot_grp_name : str, optional
        Name of the grouping column for the OOT data.
    min_data_size : int, default 100
        Minimum number of samples per group.
    grp_colname : str, optional
        Name of the group column in the output.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    gains_table : bool, default True
        Whether to compute the Gains table.
    weight_col : str, optional
        Sample weight column in each dataset DataFrame; used for weighted metrics such as AUC/KS when there is no grouping.

    Returns
    -------
    pandas.DataFrame
        Grouped performance evaluation summary table.
    """
    
    if weight_col is not None and oot_grp_name is None:
        return _weighted_eval.get_perf_summary(
            train=train,
            validation=validation,
            oot=oot,
            tgt_name=tgt_name,
            scr_name=scr_name,
            weight_col=weight_col,
            nbins=pct_bins,
        )

    if grp_colname is None:
        grp_colname = oot_grp_name
        
    if oot_grp_name is None:
        fnl_df = _get_perf_summary_single(train = train, 
                                          validation = validation, 
                                          oot = oot, 
                                          scr_name = scr_name,
                                          model = model, 
                                          tgt_name = tgt_name, 
                                          display=display,
                                          feature_cols=feature_cols, 
                                          to_show=to_show,
                                          fig_save_path=fig_save_path,
                                          rpt_save_path=rpt_save_path,
                                          dist_bins=dist_bins, 
                                          pct_bins=pct_bins,
                                          precision = precision, 
                                          min_bin_prop = min_bin_prop,
                                          include_missing = include_missing, 
                                          equal_freq = equal_freq,
                                          chi2_method = chi2_method, 
                                          init_equi_bins = init_equi_bins, 
                                          chi2_p = chi2_p, 
                                          tree_binning = tree_binning,
                                          random_state = random_state,
                                          gains_table = gains_table)
        return fnl_df
        
    grp_list = oot[oot_grp_name].sort_values(ascending = True).unique().tolist()
    valid_grp_list = [x for x in grp_list if oot[oot[oot_grp_name].isin([x])].shape[0] >= min_data_size]
    
    if len(valid_grp_list) == 0:
        return pd.DataFrame([])
    
    first_grp = valid_grp_list[0]
    oot_grp = oot[oot[oot_grp_name].isin([first_grp])]
    
    fnl_df = _get_perf_summary_single(train = train, 
                                      validation = validation, 
                                      oot = oot_grp, 
                                      scr_name = scr_name,
                                      model = model, 
                                      tgt_name = tgt_name, 
                                      display=display,
                                      feature_cols=feature_cols, 
                                      to_show=to_show,
                                      fig_save_path=fig_save_path,
                                      rpt_save_path=rpt_save_path,
                                      dist_bins=dist_bins, 
                                      pct_bins=pct_bins,
                                      precision = precision, 
                                      min_bin_prop = min_bin_prop,
                                      include_missing = include_missing, 
                                      equal_freq = equal_freq,
                                      chi2_method = chi2_method, 
                                      init_equi_bins = init_equi_bins, 
                                      chi2_p = chi2_p,
                                      tree_binning = tree_binning,
                                      random_state = random_state,
                                      gains_table = gains_table)
    fnl_df[grp_colname] = first_grp
    
    i = 1
    while i < len(valid_grp_list):
        
        grp = valid_grp_list[i]
        oot_grp = oot[oot[oot_grp_name].isin([grp])]
        
#         print(grp)

        perf_res = _get_perf_summary_single(train = train, 
                                           validation = validation, 
                                           oot = oot_grp, 
                                           scr_name = scr_name,
                                           model = model, 
                                           tgt_name = tgt_name, 
                                           display=display,
                                           feature_cols=feature_cols, 
                                           to_show=to_show,
                                           fig_save_path=fig_save_path,
                                           rpt_save_path=rpt_save_path,
                                           dist_bins=dist_bins, 
                                           pct_bins=pct_bins,
                                           precision = precision, 
                                           min_bin_prop = min_bin_prop,
                                           include_missing = include_missing, 
                                           equal_freq = equal_freq,
                                           chi2_method = chi2_method, 
                                           init_equi_bins = init_equi_bins, 
                                           chi2_p = chi2_p,
                                           tree_binning = tree_binning,
                                           random_state = random_state,
                                           gains_table = gains_table)
        perf_res[grp_colname] = grp

        fnl_df = pd.concat([fnl_df, perf_res])
        i += 1
            
    return fnl_df


def cross_risk(data, score_list, dep, nbins, agg_col = None, precision = 5, min_bin_prop = 0.05, include_missing = False, 
               equal_freq = True, binning_numeric = [True, True], agg_func = 'mean', chi2_method = False,
               chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], 
               tree_binning = False, random_state = 42, weight_col = None, sample_weight = None, wgt_col = None):
    """
    Create a cross-risk table.
    
    After binning the two score columns, compute the aggregated risk value of each cross group.
    Numeric variables can be binned automatically.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    score_list : list
        List of score columns, of length 2.
    dep : str
        Name of the target variable.
    nbins : int or list
        Number of bins (an integer or a list of length 2).
    agg_col : str, optional
        Name of the column to aggregate; defaults to ``dep``.
    precision : int or list, default 5
        Precision of the bin boundary values.
    min_bin_prop : float or list, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default False
        Whether to include missing values.
    equal_freq : bool, default True
        True for equal-frequency binning.
    binning_numeric : list, default [True, True]
        Whether to bin numeric columns.
    agg_func : str, callable, tuple or dict, default 'mean'
        Aggregation function.
        
        Regular usage is the same as the ``aggfunc`` of ``pandas.crosstab``, e.g. ``'mean'``,
        ``'sum'``, ``'count'`` or a custom function.
        
        The ratio of two aggregated columns can also be computed directly:
        
        1. Shorthand form:
           ``agg_col=(numerator_col, denominator_col), agg_func='ratio'``
        2. Tuple form:
           ``agg_func=('ratio', numerator_col, denominator_col)``
        3. Dict form:
           ``agg_func={
               'func': 'ratio',
               'numerator': numerator_col,
               'denominator': denominator_col,
               'numerator_agg': 'sum',
               'denominator_agg': 'sum',
               'return_count': True,
               'return_count_pct': True,
               'count_name': 'N',
               'count_pct_name': 'N_Pct',
               'ratio_name': 'ratio',
               'valid_only': True
           }``
        
        When ``return_count=True``, the columns of the result get an additional level of metric names,
        containing the ratio matrix and the N matrix; when ``return_count_pct=True``,
        the matrix of the share of N in each cell is also returned, computed as:
        count of the current cell / Total count in the bottom-right corner.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    chi2_p : float, default 0.95
        Significance level of the chi-square test.
    init_equi_bins : int, default 100
        Initial number of equal-frequency bins.
    fillna : any, default -999999
        Fill value for missing values.
    spec_values : list, default []
        List of special values.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    weight_col : str, optional
        Column in ``data`` with per-row sample weights (alias: ``wgt_col``).
    sample_weight : array-like, optional
        Explicit per-row weights; must align with ``len(data)``.
    wgt_col : str, optional
        Alias for ``weight_col``.
    
    Returns
    -------
    pandas.DataFrame
        Cross-risk table.
    
    Examples
    --------
    >>> cross_risk(data, score_list=['score1', 'score2'], dep='target', nbins=10)
    """
    
    from pandas.api.types import is_numeric_dtype
    
    if agg_col is None:
        agg_col = dep

    resolved_weight = None
    if weight_col is not None or sample_weight is not None or wgt_col is not None:
        resolved_weight = _weighted_eval.resolve_weights(
            data=data,
            weight_col=weight_col or wgt_col,
            sample_weight=sample_weight,
            expected_len=len(data),
        )

    def _parse_ratio_agg(agg_col, agg_func):
        """
        Parse ratio aggregation syntax while keeping backward compatibility
        with the original cross_risk agg_func behavior.
        """
        ratio_cfg = None

        if isinstance(agg_func, str) and agg_func.lower() in ["ratio", "rate", "divide"]:
            if not isinstance(agg_col, (list, tuple)) or len(agg_col) != 2:
                raise ValueError(
                    "When agg_func='ratio', `agg_col` must be a two-element "
                    "list/tuple: (numerator_col, denominator_col)."
                )
            ratio_cfg = {
                "numerator": agg_col[0],
                "denominator": agg_col[1],
            }

        elif isinstance(agg_func, (list, tuple)) and len(agg_func) >= 3:
            if str(agg_func[0]).lower() in ["ratio", "rate", "divide"]:
                ratio_cfg = {
                    "numerator": agg_func[1],
                    "denominator": agg_func[2],
                }
                if len(agg_func) >= 4:
                    ratio_cfg["return_count"] = bool(agg_func[3])

        elif isinstance(agg_func, dict):
            func_name = str(
                agg_func.get("func", agg_func.get("type", agg_func.get("agg_func", "")))
            ).lower()
            if func_name in ["ratio", "rate", "divide"]:
                ratio_cfg = dict(agg_func)
                ratio_cfg["numerator"] = ratio_cfg.get(
                    "numerator", ratio_cfg.get("num", ratio_cfg.get("numerator_col"))
                )
                ratio_cfg["denominator"] = ratio_cfg.get(
                    "denominator", ratio_cfg.get("denom", ratio_cfg.get("denominator_col"))
                )

        if ratio_cfg is None:
            return None

        if ratio_cfg.get("numerator") is None or ratio_cfg.get("denominator") is None:
            raise ValueError(
                "Ratio aggregation requires both numerator and denominator columns. "
                "Use agg_func={'func': 'ratio', 'numerator': ..., 'denominator': ...}."
            )

        ratio_cfg.setdefault("numerator_agg", "sum")
        ratio_cfg.setdefault("denominator_agg", "sum")
        ratio_cfg.setdefault("return_count", False)
        ratio_cfg.setdefault("return_count_pct", False)
        ratio_cfg.setdefault("count_name", "N")
        ratio_cfg.setdefault("count_pct_name", "N_Pct")
        ratio_cfg.setdefault("ratio_name", f"{ratio_cfg['numerator']}/{ratio_cfg['denominator']}")
        ratio_cfg.setdefault("valid_only", True)
        ratio_cfg.setdefault("zero_division", np.nan)
        return ratio_cfg

    def _parse_regular_agg(agg_func):
        """
        Parse the extended regular aggregation syntax:
        agg_func={
            'func': ['mean', 'count'],
            'return_count_pct': True,
            'count_func_name': 'count',
            'count_pct_name': 'N_Pct'
        }
        """
        if not isinstance(agg_func, dict):
            return None

        func_name = str(
            agg_func.get("func", agg_func.get("type", agg_func.get("agg_func", "")))
        ).lower()
        if func_name in ["ratio", "rate", "divide"]:
            return None

        if "func" not in agg_func and "agg_func" not in agg_func:
            return None

        regular_cfg = dict(agg_func)
        regular_cfg["func"] = regular_cfg.get("func", regular_cfg.get("agg_func"))
        regular_cfg.setdefault("return_count_pct", False)
        regular_cfg.setdefault("count_func_name", "count")
        regular_cfg.setdefault("count_pct_name", "N_Pct")
        regular_cfg.setdefault("margins_name", "Total_Avg_Risk")
        return regular_cfg

    ratio_cfg = _parse_ratio_agg(agg_col, agg_func)
    regular_cfg = _parse_regular_agg(agg_func) if ratio_cfg is None else None

    if is_numeric_dtype(data[score_list[0]]) and binning_numeric[0]:
        
        data, edges1 = super_binning(data = data, 
                                     score = score_list[0], 
                                     dep = dep, 
                                     nbins = nbins[0] if isinstance(nbins, list) else nbins, 
                                     precision = precision[0] if isinstance(precision, list) else precision, 
                                     min_bin_prop = min_bin_prop[0] if isinstance(min_bin_prop, list) else min_bin_prop, 
                                     include_missing = include_missing, 
                                     equal_freq = equal_freq, 
                                     chi2_method = chi2_method, 
                                     chi2_p = chi2_p, 
                                     init_equi_bins = init_equi_bins, 
                                     fillna = fillna, 
                                     spec_values = spec_values, 
                                     tree_binning = tree_binning, 
                                     random_state = random_state, 
                                     return_edges = True, 
                                     bin_colnames = ("_bin_num1", "_bin_range1"),
                                     ascending = True)

    else:
        data["_bin_num1"] = data[score_list[0]]
        data["_bin_range1"] = data[score_list[0]]

    if is_numeric_dtype(data[score_list[1]]) and binning_numeric[1]:
        
        data, edges2 = super_binning(data = data, 
                                     score = score_list[1], 
                                     dep = dep, 
                                     nbins = nbins[1] if isinstance(nbins, list) else nbins, 
                                     precision = precision[1] if isinstance(precision, list) else precision, 
                                     min_bin_prop = min_bin_prop[1] if isinstance(min_bin_prop, list) else min_bin_prop, 
                                     include_missing = include_missing, 
                                     equal_freq = equal_freq, 
                                     chi2_method = chi2_method, 
                                     chi2_p = chi2_p, 
                                     init_equi_bins = init_equi_bins, 
                                     fillna = fillna, 
                                     spec_values = spec_values, 
                                     tree_binning = tree_binning, 
                                     random_state = random_state, 
                                     return_edges = True, 
                                     bin_colnames = ("_bin_num2", "_bin_range2"),
                                     ascending = True)

    else:
        
        data["_bin_num2"] = data[score_list[1]]
        data["_bin_range2"] = data[score_list[1]]


    if ratio_cfg is not None:
        numerator = ratio_cfg["numerator"]
        denominator = ratio_cfg["denominator"]
        numerator_agg = ratio_cfg["numerator_agg"]
        denominator_agg = ratio_cfg["denominator_agg"]
        return_count = ratio_cfg["return_count"]
        return_count_pct = ratio_cfg["return_count_pct"]
        count_name = ratio_cfg["count_name"]
        count_pct_name = ratio_cfg["count_pct_name"]
        ratio_name = ratio_cfg["ratio_name"]
        valid_only = ratio_cfg["valid_only"]
        zero_division = ratio_cfg["zero_division"]
        margin_name = ratio_cfg.get("margins_name", "Total_Avg_Risk")

        missing_cols = [c for c in [numerator, denominator] if c not in data.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"Columns not found in data for ratio aggregation: {missing_cols}")

        agg_data = data.copy()
        if valid_only:
            agg_data = agg_data.loc[
                agg_data[numerator].notna()
                & agg_data[denominator].notna()
                & (agg_data[denominator] != 0)
            ].copy()

        numerator_res = pd.crosstab([agg_data['_bin_num1'], agg_data['_bin_range1']],
                                    [agg_data['_bin_num2'], agg_data['_bin_range2']],
                                    rownames=[score_list[0], score_list[0]],
                                    colnames=[score_list[1], score_list[1]],
                                    values=agg_data[numerator],
                                    aggfunc=numerator_agg,
                                    margins=True,
                                    margins_name=margin_name)

        denominator_res = pd.crosstab([agg_data['_bin_num1'], agg_data['_bin_range1']],
                                      [agg_data['_bin_num2'], agg_data['_bin_range2']],
                                      rownames=[score_list[0], score_list[0]],
                                      colnames=[score_list[1], score_list[1]],
                                      values=agg_data[denominator],
                                      aggfunc=denominator_agg,
                                      margins=True,
                                      margins_name=margin_name)

        res = numerator_res / denominator_res.replace(0, np.nan)
        if not pd.isna(zero_division):
            res = res.fillna(zero_division)

        if return_count or return_count_pct:
            count_col = ratio_cfg.get("count_col", numerator)
            count_res = pd.crosstab([agg_data['_bin_num1'], agg_data['_bin_range1']],
                                    [agg_data['_bin_num2'], agg_data['_bin_range2']],
                                    rownames=[score_list[0], score_list[0]],
                                    colnames=[score_list[1], score_list[1]],
                                    values=agg_data[count_col],
                                    aggfunc="count",
                                    margins=True,
                                    margins_name=margin_name)
            output_dict = {ratio_name: res}

            if return_count:
                output_dict[count_name] = count_res

            if return_count_pct:
                total_count = count_res.iloc[-1, -1] if count_res.shape[0] > 0 and count_res.shape[1] > 0 else 0
                if total_count == 0 or pd.isna(total_count):
                    count_pct_res = count_res * np.nan
                else:
                    count_pct_res = count_res / total_count
                output_dict[count_pct_name] = count_pct_res

            res = pd.concat(output_dict, axis=1)

        return res

    actual_agg_func = regular_cfg["func"] if regular_cfg is not None else agg_func
    margin_name = regular_cfg.get("margins_name", "Total_Avg_Risk") if regular_cfg is not None else "Total_Avg_Risk"

    if resolved_weight is not None:
        if ratio_cfg is not None or regular_cfg is not None:
            raise ValueError(
                "weighted cross_risk currently supports only simple mean aggregation "
                "(agg_func='mean' without extended ratio/regular dict syntax)."
            )
        agg_name = getattr(actual_agg_func, "__name__", str(actual_agg_func)).lower()
        if agg_name != "mean":
            raise ValueError(
                "weighted cross_risk currently supports only agg_func='mean'; got {0!r}".format(actual_agg_func)
            )
        return _weighted_eval.cross_risk_weighted_mean(
            data=data,
            agg_col=agg_col,
            sample_weight=resolved_weight,
            score_list=score_list,
            margin_name=margin_name,
        )

    res = pd.crosstab([data['_bin_num1'], data['_bin_range1']], 
                        [data['_bin_num2'], data['_bin_range2']], 
                        rownames=[score_list[0], score_list[0]], 
                        colnames=[score_list[1], score_list[1]], 
                        values = data[agg_col], aggfunc = actual_agg_func, 
                        margins = True, margins_name=margin_name)

    if regular_cfg is not None and regular_cfg.get("return_count_pct", False):
        count_func_name = regular_cfg.get("count_func_name", "count")
        count_pct_name = regular_cfg.get("count_pct_name", "N_Pct")

        count_res = None
        if isinstance(res.columns, pd.MultiIndex) and count_func_name in res.columns.get_level_values(0):
            count_res = res[count_func_name]

        if count_res is None:
            count_res = pd.crosstab([data['_bin_num1'], data['_bin_range1']],
                                    [data['_bin_num2'], data['_bin_range2']],
                                    rownames=[score_list[0], score_list[0]],
                                    colnames=[score_list[1], score_list[1]],
                                    values=data[agg_col],
                                    aggfunc="count",
                                    margins=True,
                                    margins_name=margin_name)

        total_count = count_res.iloc[-1, -1] if count_res.shape[0] > 0 and count_res.shape[1] > 0 else 0
        if total_count == 0 or pd.isna(total_count):
            count_pct_res = count_res * np.nan
        else:
            count_pct_res = count_res / total_count

        if isinstance(res.columns, pd.MultiIndex):
            res = pd.concat([res, pd.concat({count_pct_name: count_pct_res}, axis=1)], axis=1)
        else:
            res = pd.concat({regular_cfg.get("func_name", str(actual_agg_func)): res,
                             count_pct_name: count_pct_res}, axis=1)
    
    return res


def get_gains_table_by_cust_metrics(data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, 
                                    score = None, model = None, varlist = None, equal_freq = True, chi2_method = False,
                                    grp_name = None, min_data_size = 100, grp_colname = None, sync_range = True,
                                    chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], 
                                    tree_binning = False, random_state=42, ascending = True,
                                    eval_metrics = ["age", "monthly_income", "education"], metric_agg_func = "mean", withSummary = False):
    """
    Compute the grouped Gains table with custom metrics.
    
    Split the data into groups by the grouping column and compute the custom-metric Gains table of each group separately.
    The Gains table contains the basic statistics as well as the aggregated values of the custom metrics.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    dep : str
        Name of the target variable.
    nbins : int, default 10
        Number of bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default True
        Whether to include missing values.
    score : str, optional
        Name of the score column.
    model : sklearn-like model, optional
        Machine learning model.
    varlist : list, optional
        List of model features.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    grp_name : str, optional
        Name of the grouping column.
    min_data_size : int, default 100
        Minimum number of samples per group.
    grp_colname : str, optional
        Name of the group column in the output.
    sync_range : bool, default True
        Whether to synchronize the bin boundaries across groups.
    chi2_p : float, default 0.95
        Significance level of the chi-square test.
    init_equi_bins : int, default 100
        Initial number of equal-frequency bins.
    fillna : any, default -999999
        Fill value for missing values.
    spec_values : list, default []
        List of special values.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    ascending : bool, default True
        Whether the bin order is ascending.
    eval_metrics : list, default ["age", "monthly_income", "education"]
        List of custom metrics to compute.
    metric_agg_func : str or callable, default "mean"
        Aggregation function for the custom metrics.
    withSummary : bool, default False
        Whether to include an overall summary row.
    
    Returns
    -------
    pandas.DataFrame
        Grouped Gains table with custom metrics.
    """
    
    if grp_colname is None:
        grp_colname = grp_name
        
    if grp_name is None:
        fnl_df = _get_cust_gains_table_single(data = data, 
                                             dep = dep, 
                                             nbins = nbins, 
                                             precision = precision, 
                                             min_bin_prop = min_bin_prop, 
                                             include_missing = include_missing, 
                                             score = score, 
                                             model = model, 
                                             varlist = varlist, 
                                             equal_freq = equal_freq, 
                                             chi2_method = chi2_method,
                                             chi2_p = chi2_p, 
                                             init_equi_bins = init_equi_bins, 
                                             fillna = fillna, 
                                             spec_values = spec_values,
                                             eval_metrics = eval_metrics, 
                                             tree_binning = tree_binning, 
                                             random_state = random_state, 
                                             metric_agg_func = metric_agg_func,
                                             ascending = ascending,
                                             withSummary = withSummary)
        return fnl_df
    
    grp_list = data[grp_name].sort_values(ascending = True).unique().tolist()
    valid_grp_list = [x for x in grp_list if data[data[grp_name].isin([x])].shape[0] >= min_data_size]
    
    first_grp = valid_grp_list[0]
    data_grp = data[data[grp_name].isin([first_grp])]
        
    if sync_range:
        
        fnl_df = _get_cust_gains_table_single(data = data_grp, 
                                             dep = dep, 
                                             nbins = nbins, 
                                             precision = precision, 
                                             min_bin_prop = min_bin_prop, 
                                             include_missing = include_missing, 
                                             score = score, 
                                             model = model, 
                                             varlist = varlist, 
                                             equal_freq = equal_freq, 
                                             chi2_method = chi2_method,
                                             chi2_p = chi2_p, 
                                             init_equi_bins = init_equi_bins, 
                                             fillna = fillna, 
                                             spec_values = spec_values,
                                             eval_metrics = eval_metrics, 
                                             tree_binning = tree_binning, 
                                             random_state = random_state, 
                                             metric_agg_func = metric_agg_func,
                                             ascending = ascending,
                                             withSummary = withSummary)

        fnl_df[grp_colname] = first_grp

        nbins = get_bin_range_list(fnl_df.reset_index())
        equal_freq = False
        chi2_method = False
        
    else:

        fnl_df = _get_cust_gains_table_single(data = data_grp, 
                                             dep = dep, 
                                             nbins = nbins, 
                                             precision = precision, 
                                             min_bin_prop = min_bin_prop, 
                                             include_missing = include_missing, 
                                             score = score, 
                                             model = model, 
                                             varlist = varlist, 
                                             equal_freq = equal_freq, 
                                             chi2_method = chi2_method,
                                             chi2_p = chi2_p, 
                                             init_equi_bins = init_equi_bins, 
                                             fillna = fillna, 
                                             spec_values = spec_values,
                                             eval_metrics = eval_metrics, 
                                             tree_binning = tree_binning, 
                                             random_state = random_state, 
                                             metric_agg_func = metric_agg_func,
                                             ascending = ascending,
                                             withSummary = withSummary)

        fnl_df[grp_colname] = first_grp
    
    i = 1
    while i < len(valid_grp_list):
        
        grp = valid_grp_list[i]
        data_grp = data[data[grp_name].isin([grp])]

        perf_res = _get_cust_gains_table_single(data = data_grp, 
                                               dep = dep, 
                                               nbins = nbins, 
                                               precision = precision, 
                                               min_bin_prop = min_bin_prop, 
                                               include_missing = include_missing, 
                                               score = score, 
                                               model = model, 
                                               varlist = varlist, 
                                               equal_freq = equal_freq, 
                                               chi2_method = chi2_method,
                                               chi2_p = chi2_p, 
                                               init_equi_bins = init_equi_bins, 
                                               fillna = fillna, 
                                               spec_values = spec_values, 
                                               eval_metrics = eval_metrics, 
                                               tree_binning = tree_binning, 
                                               random_state = random_state, 
                                               metric_agg_func = metric_agg_func,
                                               ascending = ascending,
                                               withSummary = withSummary)
        perf_res[grp_colname] = grp

        fnl_df = pd.concat([fnl_df, perf_res])
        i += 1
            
    return fnl_df


def tie_score_rate(data, score):
    """
    Compute the score tie rate.
    
    Compute the proportion of non-unique score values, i.e. the share of samples whose score is duplicated.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    score : str
        Name of the score column.
    
    Returns
    -------
    float
        Score tie rate (between 0 and 1).
    
    Examples
    --------
    >>> tie_score_rate(data, 'score')
    0.15  # means that 15% of the samples have a duplicated score
    """
    
    n_unique_scr = len(data[score].unique())
    unique_scr_prop = n_unique_scr / data.shape[0]
    return (1 - unique_scr_prop)


def score_unique_rate(data, score):
    """
    Compute the score unique rate.
    
    Compute the proportion of unique score values in the total number of samples.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table (this parameter is retained but not used).
    score : array-like
        Score column or array.
    
    Returns
    -------
    float
        Score unique rate (between 0 and 1).
    
    Examples
    --------
    >>> score_unique_rate(data, data['score'])
    0.85  # means that the scores of 85% of the samples are unique
    """
    
    return len(np.unique(score)) / len(score)


class GainsTableCalculator:
    """
    Gains table calculator.
    
    Integrates the various Gains table functions and supports both the basic Gains table and the
    Gains table with custom metrics. Provides an object-oriented interface for computing grouped Gains tables.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table.
    dep : str
        Name of the target variable.
    nbins : int, default 10
        Number of bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default True
        Whether to include missing values.
    score : str, optional
        Name of the score column.
    model : sklearn-like model, optional
        Machine learning model.
    varlist : list, optional
        List of model features.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    chi2_p : float, default 0.95
        Significance level of the chi-square test.
    init_equi_bins : int, default 100
        Initial number of equal-frequency bins.
    fillna : any, default -999999
        Fill value for missing values.
    spec_values : list, default []
        List of special values.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    ascending : bool, default False
        Whether the bin order is ascending.
    
    Examples
    --------
    >>> calc = GainsTableCalculator(data, dep='target', score='score', nbins=10)
    >>> result = calc.calculate(grp_name='region')
    """
    
    def __init__(self, data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05,
                 include_missing = True, score = None, model = None, varlist = None,
                 equal_freq = True, chi2_method = False, chi2_p = 0.95, 
                 init_equi_bins = 100, fillna = -999999, spec_values = [],
                 tree_binning = False, random_state = 42, ascending = False,
                 weight_col = None, weighted_binning = None):
        """
        Initialize the Gains table calculator.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input data table.
        dep : str
            Name of the target variable.
        nbins : int, default 10
            Number of bins.
        precision : int, default 5
            Precision of the bin boundary values.
        min_bin_prop : float, default 0.05
            Minimum proportion of samples per bin.
        include_missing : bool, default True
            Whether to include missing values.
        score : str, optional
            Name of the score column.
        model : sklearn-like model, optional
            Machine learning model.
        varlist : list, optional
            List of model features.
        equal_freq : bool, default True
            True for equal-frequency binning.
        chi2_method : bool, default False
            Whether to use chi-square binning.
        chi2_p : float, default 0.95
            Significance level of the chi-square test.
        init_equi_bins : int, default 100
            Initial number of equal-frequency bins.
        fillna : any, default -999999
            Fill value for missing values.
        spec_values : list, default []
            List of special values.
        tree_binning : bool, default False
            Whether to use decision-tree binning.
        random_state : int, default 42
            Random seed.
        ascending : bool, default False
            Whether the bin order is ascending.
        weight_col : str, optional
            Name of the sample weight column (must be in ``data``).
        weighted_binning : bool, optional
            If True, use equal-frequency binning by cumulative weight.
        """
        self.data = data
        self.dep = dep
        self.nbins = nbins
        self.precision = precision
        self.min_bin_prop = min_bin_prop
        self.include_missing = include_missing
        self.score = score
        self.model = model
        self.varlist = varlist
        self.equal_freq = equal_freq
        self.chi2_method = chi2_method
        self.chi2_p = chi2_p
        self.init_equi_bins = init_equi_bins
        self.fillna = fillna
        self.spec_values = spec_values
        self.tree_binning = tree_binning
        self.random_state = random_state
        self.ascending = ascending
        self.weight_col = weight_col
        self.weighted_binning = weighted_binning
    
    def calculate(self, grp_name = None, min_data_size = 100, grp_colname = None,
                  sync_range = True, retSummary = False, withSummary = False,
                  wholeGroup = False, add_func = None, weight_col = None):
        """
        Compute the Gains table.
        
        Parameters
        ----------
        grp_name : str, optional
            Name of the grouping column.
        min_data_size : int, default 100
            Minimum number of samples per group.
        grp_colname : str, optional
            Name of the group column in the output.
        sync_range : bool, default True
            Whether to synchronize the bin boundaries across groups.
        retSummary : bool, default False
            Whether to return only the summary metrics.
        withSummary : bool, default False
            Whether to include an overall summary row.
        wholeGroup : bool, default False
            Whether to use all the data for binning.
        add_func : callable, optional
            Custom statistics function.
        
        Returns
        -------
        pandas.DataFrame
            Gains table.
        """
        return get_gains_table(
            data = self.data,
            dep = self.dep,
            nbins = self.nbins,
            precision = self.precision,
            min_bin_prop = self.min_bin_prop,
            include_missing = self.include_missing,
            score = self.score,
            model = self.model,
            varlist = self.varlist,
            equal_freq = self.equal_freq,
            chi2_method = self.chi2_method,
            grp_name = grp_name,
            min_data_size = min_data_size,
            grp_colname = grp_colname,
            sync_range = sync_range,
            chi2_p = self.chi2_p,
            init_equi_bins = self.init_equi_bins,
            fillna = self.fillna,
            spec_values = self.spec_values,
            retSummary = retSummary,
            tree_binning = self.tree_binning,
            random_state = self.random_state,
            ascending = self.ascending,
            withSummary = withSummary,
            wholeGroup = wholeGroup,
            add_func = add_func,
            weight_col = self.weight_col if weight_col is None else weight_col,
            weighted_binning = self.weighted_binning,
        )


class PerformanceEvaluator:
    """
    Performance evaluator.
    
    Integrates the various model performance evaluation functions and supports evaluation across
    multiple datasets and multiple groups. Provides an object-oriented interface for computing and
    summarizing performance metrics.
    
    Parameters
    ----------
    tgt_name : str or list of str
        Name of the target variable. A list/tuple of several y labels can be passed: each label is evaluated
        separately and the results are stacked vertically (the output gains a ``tgt_name`` column),
        and a separate figure is drawn for each label.
    scr_name : str, optional
        Name of the score column.
    model : sklearn-like model, optional
        Machine learning model.
    feature_cols : list, optional
        List of model features.
    dist_bins : int, default 20
        Number of bins for the score distribution.
    pct_bins : int, default 10
        Number of percentile bins.
    precision : int, default 5
        Precision of the bin boundary values.
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin.
    include_missing : bool, default False
        Whether to include missing values.
    equal_freq : bool, default True
        True for equal-frequency binning.
    chi2_method : bool, default False
        Whether to use chi-square binning.
    init_equi_bins : int, default 1000
        Initial number of equal-frequency bins.
    chi2_p : float, default 0.9
        Significance level of the chi-square test.
    tree_binning : bool, default False
        Whether to use decision-tree binning.
    random_state : int, default 42
        Random seed.
    weight_col : str, optional
        Default weight column; each ``add_dataset`` call can also specify its own.

    Examples
    --------
    >>> evaluator = PerformanceEvaluator(tgt_name='target', model=model, feature_cols=features)
    >>> evaluator.add_dataset('train', train_df)
    >>> evaluator.add_dataset('validation', val_df)
    >>> evaluator.add_dataset('oot', oot_df)
    >>> result = evaluator.evaluate()
    >>>
    >>> # Multiple y labels: pass a list as tgt_name -> a vertically stacked table with a tgt_name column, one figure per label
    >>> evaluator = PerformanceEvaluator(tgt_name=['bad_dpd7', 'bad_dpd30'], model=model, feature_cols=features)
    >>> evaluator.add_dataset('train', train_df).add_dataset('oot', oot_df)
    >>> result = evaluator.evaluate(to_show=True)
    """
    
    def __init__(self, tgt_name, scr_name = None, model = None, feature_cols = None,
                 dist_bins = 20, pct_bins = 10, precision = 5, min_bin_prop = 0.05,
                 include_missing = False, equal_freq = True, chi2_method = False,
                 init_equi_bins = 1000, chi2_p = 0.9, tree_binning = False, random_state = 42,
                 weight_col = None, spec_values = None, ascending = None):
        """
        Initialize the performance evaluator.
        
        Parameters
        ----------
        tgt_name : str or list of str
            Name of the target variable. A list/tuple of several y labels can be passed: each label is evaluated
            separately and the results are stacked vertically (the output gains a ``tgt_name`` column),
            and a separate figure is drawn for each label.
        scr_name : str, optional
            Name of the score column.
        model : sklearn-like model, optional
            Machine learning model.
        feature_cols : list, optional
            List of model features.
        dist_bins : int, default 20
            Number of bins for the score distribution.
        pct_bins : int, default 10
            Number of percentile bins.
        precision : int, default 5
            Precision of the bin boundary values.
        min_bin_prop : float, default 0.05
            Minimum proportion of samples per bin.
        include_missing : bool, default False
            Whether to include missing values.
        equal_freq : bool, default True
            True for equal-frequency binning.
        chi2_method : bool, default False
            Whether to use chi-square binning.
        init_equi_bins : int, default 1000
            Initial number of equal-frequency bins.
        chi2_p : float, default 0.9
            Significance level of the chi-square test.
        tree_binning : bool, default False
            Whether to use decision-tree binning.
        random_state : int, default 42
            Random seed.
        weight_col : str, optional
            Default weight column; each ``add_dataset`` call can also specify its own.
        """
        self.tgt_name = tgt_name
        self.scr_name = scr_name
        self.model = model
        self.feature_cols = feature_cols
        self.dist_bins = dist_bins
        self.pct_bins = pct_bins
        self.precision = precision
        self.min_bin_prop = min_bin_prop
        self.include_missing = include_missing
        self.equal_freq = equal_freq
        self.chi2_method = chi2_method
        self.init_equi_bins = init_equi_bins
        self.chi2_p = chi2_p
        self.tree_binning = tree_binning
        self.random_state = random_state
        self.weight_col = weight_col
        # spec_values: sentinel scores (e.g. -1 for all-missing overrides)
        # that get their own evaluation bin and are excluded from quantile
        # edges and ranking metrics. ascending: None keeps every underlying
        # function's legacy direction; an explicit bool is threaded through
        # summary, gains, and figures (including weighted paths) uniformly.
        self.spec_values = list(spec_values) if spec_values else []
        self.ascending = ascending
        self.datasets = {}
        self.dataset_weight_cols = {}
        self.evaluate_status = None

    def _governed_eval_kwargs(self):
        """Kwargs to thread spec_values/ascending into underlying eval calls.

        Nothing is passed when unset, so every legacy default stays intact
        (unweighted summary False, comparison paths True, weighted False).
        """
        kwargs = {}
        if self.spec_values:
            kwargs["spec_values"] = list(self.spec_values)
        if self.ascending is not None:
            kwargs["ascending"] = bool(self.ascending)
        return kwargs
    
    def add_dataset(self, name, data, weight_col = None, overwrite = False):
        """
        Add a dataset.
        
        Parameters
        ----------
        name : str
            Name of the dataset (e.g. 'train', 'validation', 'oot').
        data : pandas.DataFrame
            Dataset.
        
        Returns
        -------
        self
            The evaluator itself, to allow method chaining.
        """
        if name in self.datasets and not overwrite:
            raise KeyError(
                f"Dataset name {name!r} already exists. Use overwrite=True to replace it, "
                "or pass a different dataset name."
            )
        if name in self.datasets and overwrite:
            warnings.warn(
                f"PerformanceEvaluator.add_dataset overwriting existing dataset {name!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.datasets[name] = data
        self.dataset_weight_cols[name] = weight_col
        return self
    
    def evaluate(self, oot_grp_name = None, min_data_size = 100, grp_colname = None,
                 fig_save_path = None, rpt_save_path = None, to_show = False, 
                 display = True, gains_table = False, benchmark_dataset = None,
                 weight_col = None):
        """
        Run the performance evaluation.
        
        Parameters
        ----------
        oot_grp_name : str, optional
            Name of the grouping column for the OOT data.
        min_data_size : int, default 100
            Minimum number of samples per group.
        grp_colname : str, optional
            Name of the group column in the output.
        fig_save_path : str, optional
            Path to save the figure.
        rpt_save_path : str, optional
            Path to save the report.
        to_show : bool, default False
            Whether to display the figures.
        display : bool, default True
            Whether to print the results.
        gains_table : bool, default True
            Whether to compute the Gains table.
        benchmark_dataset : str or pandas.DataFrame, optional
            Benchmark dataset used to fix the bin boundaries. If a str is passed, the dataset is looked up by name
            among the datasets added with ``add_dataset``; if a DataFrame is passed, it is used directly.
            The default None means that each dataset is binned independently.
        
        Returns
        -------
        pandas.DataFrame
            Performance evaluation summary table. If ``tgt_name`` of the instance is a list/tuple of several labels,
            each label is evaluated separately and the results are stacked vertically after a new ``tgt_name`` column
            is added; when ``to_show=True``, one figure is drawn for each label, and
            ``fig_save_path`` automatically gets a label suffix (e.g. ``perf.png`` -> ``perf_<label>.png``).
        """
        if len(self.datasets) == 0:
            self.evaluate_status = "no_datasets"
            return pd.DataFrame([])

        if self.scr_name is None and self.model is None and self.feature_cols is None:
            self.evaluate_status = "compute_failed"
            warnings.warn(
                "PerformanceEvaluator.evaluate() now returns an empty DataFrame instead of -1; "
                "inspect .evaluate_status for the failure reason.",
                DeprecationWarning,
                stacklevel=2,
            )
            return pd.DataFrame([])
        
        if self.scr_name is None and self.model is None:
            self.evaluate_status = "compute_failed"
            warnings.warn(
                "PerformanceEvaluator.evaluate() now returns an empty DataFrame instead of -2; "
                "inspect .evaluate_status for the failure reason.",
                DeprecationWarning,
                stacklevel=2,
            )
            return pd.DataFrame([])
        
        if self.scr_name is None and self.feature_cols is None:
            self.evaluate_status = "compute_failed"
            warnings.warn(
                "PerformanceEvaluator.evaluate() now returns an empty DataFrame instead of -3; "
                "inspect .evaluate_status for the failure reason.",
                DeprecationWarning,
                stacklevel=2,
            )
            return pd.DataFrame([])

        active_weight_col = weight_col or self.weight_col
        has_dataset_weight = any(v is not None for v in self.dataset_weight_cols.values())
        if (active_weight_col is not None or has_dataset_weight) and oot_grp_name is None and benchmark_dataset is None and not isinstance(self.tgt_name, (list, tuple)):
            rows = []
            for name, data in self.datasets.items():
                if data is None:
                    continue
                wc = self.dataset_weight_cols.get(name) or active_weight_col
                work_data = data.copy()
                scr_name = self.scr_name
                if scr_name is None:
                    scr_name = "_mdl_scr"
                    work_data[scr_name] = self.model.predict_proba(work_data.loc[:, self.feature_cols])[:, 1]
                rows.append(
                    _weighted_eval.dataset_summary(
                        name,
                        work_data,
                        self.tgt_name,
                        scr_name,
                        weight_col=wc,
                        nbins=self.pct_bins,
                        ascending=bool(self.ascending) if self.ascending is not None else False,
                        spec_values=self.spec_values or None,
                    )
                )
            fnl_df = pd.DataFrame(rows)
            if display:
                from IPython.display import display as _ipy_display
                _ipy_display(fnl_df)
            if fig_save_path:
                eval_datasets = {}
                for name, data in self.datasets.items():
                    if data is None:
                        continue
                    wc = self.dataset_weight_cols.get(name) or active_weight_col
                    work_data = data.copy()
                    scr_name = self.scr_name
                    if scr_name is None:
                        scr_name = "_mdl_scr"
                        work_data[scr_name] = self.model.predict_proba(work_data.loc[:, self.feature_cols])[:, 1]
                    resolved_weight = _weighted_eval.resolve_weights(
                        work_data,
                        weight_col=wc,
                        expected_len=len(work_data),
                    )
                    y_true = work_data[self.tgt_name].to_numpy()
                    y_score = work_data[scr_name].to_numpy()
                    if self.spec_values:
                        keep = ~pd.Series(y_score).isin(self.spec_values).to_numpy()
                        y_true = y_true[keep]
                        y_score = y_score[keep]
                        if resolved_weight is not None:
                            resolved_weight = np.asarray(resolved_weight)[keep]
                    payload = {
                        "y_true": y_true,
                        "y_score": y_score,
                    }
                    if resolved_weight is not None:
                        payload["sample_weight"] = resolved_weight
                    eval_datasets[name] = payload
                if eval_datasets:
                    evaluate_performance(
                        datasets=eval_datasets,
                        dist_bins=self.dist_bins,
                        pct_bins=self.pct_bins,
                        square_figsize=5,
                        to_show=to_show,
                        save_path=fig_save_path,
                        gains_table=gains_table,
                        equal_freq=self.equal_freq,
                        ascending=self.ascending,
                    )
            if rpt_save_path:
                fnl_df.to_csv(rpt_save_path, index=False)
            self.evaluate_status = "ok" if not fnl_df.empty else "empty_input"
            return fnl_df

        # ── Multiple y labels: when tgt_name is a list/tuple, evaluate label by label, then stack the results vertically ──
        #    Table:   the result of each label gets a new tgt_name column, then they are stacked vertically;
        #    Figures: each label gets its own figure (shown in a loop when to_show=True; fig_save_path automatically gets a label suffix).
        if isinstance(self.tgt_name, (list, tuple)):
            import os as _os

            def _suffix_path(p, lab):
                if not p:
                    return None
                root, ext = _os.path.splitext(str(p))
                return "{0}_{1}{2}".format(root, lab, ext)

            _orig_tgt = self.tgt_name
            multi_results = []
            try:
                for _t in list(self.tgt_name):
                    self.tgt_name = _t
                    sub_df = self.evaluate(
                        oot_grp_name = oot_grp_name,
                        min_data_size = min_data_size,
                        grp_colname = grp_colname,
                        fig_save_path = _suffix_path(fig_save_path, _t),
                        rpt_save_path = None,        # saved once after stacking (see below)
                        to_show = to_show,           # each label gets its own figure
                        display = False,             # displayed once after stacking
                        gains_table = gains_table,
                        benchmark_dataset = benchmark_dataset,
                    )
                    if isinstance(sub_df, pd.DataFrame) and sub_df.shape[0] > 0:
                        sub_df.insert(0, "tgt_name", _t)
                        multi_results.append(sub_df)
            finally:
                self.tgt_name = _orig_tgt

            fnl_df = pd.concat(multi_results, ignore_index = True) if multi_results else pd.DataFrame([])

            if display:
                from IPython.display import display as _ipy_display
                _ipy_display(fnl_df)
            if rpt_save_path:
                fnl_df.to_csv(rpt_save_path, index = False)
            self.evaluate_status = "ok" if not fnl_df.empty else "empty_input"
            return fnl_df

        def _get_score(data):
            if self.scr_name is not None:
                return data[self.scr_name]
            return self.model.predict_proba(data.loc[:, self.feature_cols])[:, 1]

        def _rank_payload(data):
            """(y_true, y_score) with special sentinel scores removed.

            Sentinels like the all-missing override -1 carry no ordering
            information: AUC/KS/distribution figures are computed on the
            non-special rows; their bin-level accounting lives in the gains
            table via spec_values.
            """
            y_true = data[self.tgt_name]
            y_score = _get_score(data)
            if not self.spec_values:
                return y_true, y_score
            keep = ~pd.Series(np.asarray(y_score)).isin(self.spec_values).to_numpy()
            return np.asarray(y_true)[keep], np.asarray(y_score)[keep]

        def _get_benchmark_bin_edges():
            if benchmark_dataset is None:
                return None

            if isinstance(benchmark_dataset, str):
                if benchmark_dataset not in self.datasets:
                    raise ValueError("benchmark_dataset must be one of added datasets: {0}".format(list(self.datasets.keys())))
                benchmark_data = self.datasets[benchmark_dataset]
            else:
                benchmark_data = benchmark_dataset

            benchmark_score = "_benchmark_score"
            benchmark_df = pd.DataFrame({
                self.tgt_name: benchmark_data[self.tgt_name],
                benchmark_score: _get_score(benchmark_data)
            })

            _, benchmark_bin_edges = super_binning(
                data = benchmark_df,
                score = benchmark_score,
                dep = self.tgt_name,
                nbins = self.pct_bins,
                precision = self.precision,
                min_bin_prop = self.min_bin_prop,
                include_missing = self.include_missing,
                equal_freq = self.equal_freq,
                chi2_method = self.chi2_method,
                chi2_p = self.chi2_p,
                init_equi_bins = self.init_equi_bins,
                tree_binning = self.tree_binning,
                random_state = self.random_state,
                return_edges = True,
                ascending = True if self.ascending is None else bool(self.ascending)
            )

            if benchmark_bin_edges is None or len(benchmark_bin_edges) < 2:
                raise ValueError("Cannot generate benchmark bin edges from benchmark_dataset.")

            return list(benchmark_bin_edges)

        benchmark_bin_edges = _get_benchmark_bin_edges()

        def _evaluate_dataset_dict(dataset_dict, save_path = None):
            eval_datasets = {}
            for name, data in dataset_dict.items():
                if data is None:
                    continue
                y_true, y_score = _rank_payload(data)
                eval_datasets[name] = {
                    "y_true": y_true,
                    "y_score": y_score
                }

            if len(eval_datasets) == 0:
                self.evaluate_status = "empty_input"
                return pd.DataFrame([])

            model_eval_result_df = evaluate_performance(
                datasets = eval_datasets,
                dist_bins = self.dist_bins,
                pct_bins = self.pct_bins,
                square_figsize = 5,
                to_show = to_show,
                save_path = save_path,
                gains_table = gains_table,
                equal_freq = self.equal_freq,
                pct_bin_edges = benchmark_bin_edges,
                ascending = self.ascending,
            )

            import re
            btm_cols = [x for x in model_eval_result_df.columns if x.startswith("Btm")]
            top_cols = [x for x in model_eval_result_df.columns if x.startswith("Top")]

            if btm_cols and top_cols:
                btm_str = btm_cols[0]
                top_str = top_cols[0]
                numbers = re.findall(r'\d+', btm_str)
                if numbers:
                    quantile = int(numbers[0])
                else:
                    quantile = self.pct_bins

                model_eval_result_df[f"Btm{quantile}%_Lift"] = model_eval_result_df[btm_str] / model_eval_result_df["avgTrue"]
                model_eval_result_df[f"Top{quantile}%_Lift"] = model_eval_result_df[top_str] / model_eval_result_df["avgTrue"]
                model_eval_result_df["AUC_Shift"] = model_eval_result_df["AUC"].shift(1) / model_eval_result_df["AUC"] - 1
                model_eval_result_df["KS_Shift"] = model_eval_result_df["KS"].shift(1) / model_eval_result_df["KS"] - 1

            gains_table_cols = ['N_BUMP', 'MIN_RISK_DEP', 'MAX_RISK_DEP', 'KS_IN_GAINS', 'LIFT_IN_GAINS', 'IV', 'N_BINS']
            gains_summ_list = []
            for name, data in dataset_dict.items():
                if data is None:
                    gains_res = pd.DataFrame([], columns = gains_table_cols)
                else:
                    gains_res = get_gains_table(
                        data = data,
                        dep = self.tgt_name,
                        nbins = benchmark_bin_edges if benchmark_bin_edges is not None else self.pct_bins,
                        precision = self.precision,
                        min_bin_prop = self.min_bin_prop,
                        include_missing = self.include_missing,
                        score = self.scr_name,
                        equal_freq = self.equal_freq,
                        chi2_method = False if benchmark_bin_edges is not None else self.chi2_method,
                        model = self.model,
                        varlist = self.feature_cols,
                        init_equi_bins = self.init_equi_bins,
                        chi2_p = self.chi2_p,
                        retSummary = True,
                        tree_binning = False if benchmark_bin_edges is not None else self.tree_binning,
                        random_state = self.random_state,
                        **self._governed_eval_kwargs(),
                    )
                gains_res['index'] = name
                gains_summ_list.append(gains_res)

            if gains_summ_list:
                gains_summ = concat_non_empty(gains_summ_list)
                model_eval_result_df = model_eval_result_df.merge(gains_summ, on = ['index'], how = 'left')

            return model_eval_result_df

        if oot_grp_name is None:
            fnl_df = _evaluate_dataset_dict(self.datasets, save_path = fig_save_path)
        else:
            if grp_colname is None:
                grp_colname = oot_grp_name

            grouped_results = []
            for dataset_name, data in self.datasets.items():
                if data is None or oot_grp_name not in data.columns:
                    continue

                grp_list = data[oot_grp_name].sort_values(ascending = True).unique().tolist()
                valid_grp_list = [x for x in grp_list if data[data[oot_grp_name].isin([x])].shape[0] >= min_data_size]

                for grp in valid_grp_list:
                    data_grp = data[data[oot_grp_name].isin([grp])]
                    perf_res = _evaluate_dataset_dict({dataset_name: data_grp}, save_path = None)
                    if perf_res.shape[0] > 0:
                        perf_res[grp_colname] = grp
                        grouped_results.append(perf_res)

            if len(grouped_results) == 0:
                fnl_df = pd.DataFrame([])
            else:
                fnl_df = pd.concat(grouped_results, ignore_index = True)

        if display:
            from IPython.display import display
            display(fnl_df)

        if rpt_save_path:
            fnl_df.to_csv(rpt_save_path, index = False)

        self.evaluate_status = "ok" if isinstance(fnl_df, pd.DataFrame) and not fnl_df.empty else "empty_input"
        return fnl_df
