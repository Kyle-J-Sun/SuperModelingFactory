import logging
import os
import numpy as np
import pandas as pd
from Modeling_Tool.Core.Binning_Tool import get_bin_range_list, super_binning
from Modeling_Tool.Core.utils import load_model, calc_iv, calc_woe
from .evaluate_model import evaluate_performance

###################################################### Private Functions #############################################################

def _get_gains_table_scr(data, score, dep, nbins = 10, precision = 5, 
                         min_bin_prop = 0.05, include_missing = True, equal_freq = True, 
                         chi2_method = False, chi2_p = 0.95, init_equi_bins = 2000, 
                         fillna = -999999, spec_values = [], retSummary = False, 
                         tree_binning = False, random_state=42, ascending = False,
                         withSummary = False, add_func = None):
    """
    计算指定分数字段的收益表（Gains Table）。
    
    对数据进行分箱处理后，计算每个分箱的统计指标，包括样本数、坏样本率、
    累计好/坏样本数、WOE、IV等。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    score : str
        分数字段名
    dep : str
        目标变量名（二分类标签，0和1）
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱，False为等距分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 2000
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    retSummary : bool, default False
        是否只返回汇总指标
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    ascending : bool, default False
        分箱顺序是否升序
    withSummary : bool, default False
        是否包含总体汇总行
    add_func : callable, optional
        自定义统计函数
    
    Returns
    -------
    pandas.DataFrame
        收益表，包含各分箱统计指标
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
    
    def _compute_gains_tmp(group):
        # 计算各项指标
        min_val = group[score].min()
        max_val = group[score].max()
        # NOTE: pandas 2.3+ 起 groupby(...).apply(func) 传入 func 的 group
        # DataFrame 不再包含分组列,不能再用 group['_bin_num']。
        # len(group) 与原 .count() 数值等价(_bin_num 为分组主键,不为 NaN)。
        n = len(group)
        avg_score = group[score].mean()
        unique_score = group[score].nunique()

        # dep相关
        dep_vals = group[dep]
        
        grand_total = res.shape[0]
        grand_perf_cnt = res[dep].count()
        grand_total_bad = res[dep].sum()
        grand_total_good = (grand_perf_cnt - grand_total_bad)
        
        perf_cnt = dep_vals.count()
        n_bad = (dep_vals == 1).sum()
        n_good = (dep_vals == 0).sum()
        avg_bad = n_bad / dep_vals.count() if dep_vals.count() > 0 else 0  # 避免除以0
        avg_good = n_good / dep_vals.count() if dep_vals.count() > 0 else 0
        
        lift = avg_bad / res[dep].mean()
        prop = n / grand_total

        # 返回Series
        return pd.Series({
            'MIN': min_val,
            'MAX': max_val,
            'N': n,
            'PROP': prop,
            'PERF_CNT': perf_cnt,
            'AVG_SCORE': avg_score,
            'UNIQUE_SCORE': unique_score,
            'AVG_BAD': avg_bad,
            'AVG_GOOD': avg_good,
            'N_BAD': n_bad,
            'N_GOOD': n_good,
            'LIFT': lift
        })
    
    gains_table = res.groupby(["_bin_num", "_bin_range"], dropna=False).apply(_compute_gains_tmp)

    gains_table["BAD_PCT_IN_EACH_BIN"] = gains_table["N_BAD"] / gains_table["N_BAD"].sum()
    gains_table["GOOD_PCT_IN_EACH_BIN"] = gains_table["N_GOOD"] / gains_table["N_GOOD"].sum()
    
    gains_table["N_CUM_BAD"] = gains_table["N_BAD"].cumsum()
    gains_table["N_CUM_GOOD"] = gains_table["N_GOOD"].cumsum()

    gains_table["CUM_BAD_PCT"] = gains_table["N_CUM_BAD"]/gains_table["N_BAD"].sum()
    gains_table["CUM_GOOD_PCT"] = gains_table["N_CUM_GOOD"]/gains_table["N_GOOD"].sum()
    gains_table["KS_PER_BIN"] = np.abs((gains_table["CUM_BAD_PCT"] - gains_table["CUM_GOOD_PCT"]))


    gains_table["TRUE_BAD_SHIFT"] = (gains_table['AVG_BAD'].shift(1) / gains_table['AVG_BAD'] - 1) if not ascending else (gains_table['AVG_BAD'] / gains_table['AVG_BAD'].shift(1) - 1) 
    gains_table["RANK_ORDER_BUMP"] = gains_table["TRUE_BAD_SHIFT"].apply(lambda x: 1 if x < 0 else 0)
    
    gains_table["WOE"] = calc_woe(data = gains_table, bad_pct = "BAD_PCT_IN_EACH_BIN", good_pct = "GOOD_PCT_IN_EACH_BIN")
    gains_table["IV"] = calc_iv(data = gains_table, bad_pct = "BAD_PCT_IN_EACH_BIN", good_pct = "GOOD_PCT_IN_EACH_BIN")
    

    if add_func is not None:
        gains_table_add = res.groupby(["_bin_num", "_bin_range"], dropna=False).apply(add_func)
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
    计算单个模型的收益表。
    
    根据传入的score字段或模型预测结果，计算收益表。
    优先使用传入的score字段，若无则使用模型预测概率。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    dep : str
        目标变量名
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    score : str, optional
        分数字段名
    model : sklearn-like model, optional
        机器学习模型（当score为None时使用）
    varlist : list, optional
        模型特征列表
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 100
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    retSummary : bool, default False
        是否只返回汇总指标
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    ascending : bool, default False
        分箱顺序是否升序
    withSummary : bool, default False
        是否包含总体汇总行
    add_func : callable, optional
        自定义统计函数
    
    Returns
    -------
    pandas.DataFrame or int
        收益表；若缺少必要参数返回-1/-2/-3
    """
    
    if score is None and model is None and varlist is None:
        return -1
    
    if score is None and model is None:
        return -2
    
    if score is None and varlist is None:
        return -3
    
    if score is None:
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
    计算单个模型的性能评估汇总。
    
    对训练集、验证集和oot样本进行模型性能评估，包括AUC、KS、
    Lift等指标，并可选择生成收益表。
    
    Parameters
    ----------
    train : pandas.DataFrame, optional
        训练数据集
    validation : pandas.DataFrame, optional
        验证数据集
    oot : pandas.DataFrame, optional
        oot（Out-of-Time）数据集
    tgt_name : str
        目标变量名
    scr_name : str, optional
        分数字段名
    model : sklearn-like model, optional
        机器学习模型
    feature_cols : list, optional
        模型特征列表
    fig_save_path : str, optional
        图片保存路径
    rpt_save_path : str, optional
        报告保存路径
    to_show : bool, default False
        是否显示图形
    display : bool, default True
        是否打印结果
    dist_bins : int, default 20
        分布分箱数
    pct_bins : int, default 10
        百分比分箱数
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default False
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    init_equi_bins : int, default 1000
        初始等频分箱数量
    chi2_p : float, default 0.9
        卡方检验显著性水平
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    gains_table : bool, default True
        是否计算收益表
    
    Returns
    -------
    pandas.DataFrame or int
        性能评估汇总表；若缺少必要参数返回-1/-2/-3
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
        # 提取数字
        numbers = re.findall(r'\d+', btm_str)
        if numbers:
            quantile = int(numbers[0])
        else:
            # 默认值，例如设为 pct_bins 的倒数？
            quantile = pct_bins  # 或其他合理默认
    else:
        # 如果没有这些列，说明数据不足或未生成，跳过后续计算或赋予默认值
        # 这里可以选择跳过 Lift 列的计算，直接返回 model_eval_result_df
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
    
    gains_summ = pd.concat([ins_gains, oos_gains, oot_gains])
    
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
    计算指定分数字段的收益表，并包含自定义指标的聚合统计。
    
    对数据进行分箱处理后，计算每个分箱的基础统计指标以及
    自定义指标的聚合值（如均值等）。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    score : str
        分数字段名
    dep : str
        目标变量名
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 2000
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    eval_metrics : list, default ["age", "monthly_income", "education"]
        需要统计的自定义指标列表
    metric_agg_func : str or callable, default "mean"
        自定义指标的聚合函数
    ascending : bool, default False
        分箱顺序是否升序
    withSummary : bool, default False
        是否包含总体汇总行
    
    Returns
    -------
    pandas.DataFrame
        收益表，包含基础统计和自定义指标聚合值
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


    # 计算每个分箱的统计信息
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
    计算单个模型的自定义指标收益表。
    
    根据传入的score字段或模型预测结果，计算包含自定义指标聚合统计的收益表。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    dep : str
        目标变量名
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    score : str, optional
        分数字段名
    model : sklearn-like model, optional
        机器学习模型
    varlist : list, optional
        模型特征列表
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 100
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    ascending : bool, default True
        分箱顺序是否升序
    eval_metrics : list, default ["age", "monthly_income", "education"]
        需要统计的自定义指标列表
    metric_agg_func : str or callable, default "mean"
        自定义指标的聚合函数
    withSummary : bool, default False
        是否包含总体汇总行
    
    Returns
    -------
    pandas.DataFrame or int
        收益表；若缺少必要参数返回-1/-2/-3
    """
    
    if score is None and model is None and varlist is None:
        return -1
    
    if score is None and model is None:
        return -2
    
    if score is None and varlist is None:
        return -3
    
    if score is None:
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
                    add_func = None):
    """
    计算分组收益表。
    
    根据分组字段对数据进行分组，分别计算每个分组的收益表。
    若未指定分组字段，则计算整体收益表。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    dep : str
        目标变量名
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    score : str, optional
        分数字段名
    model : sklearn-like model, optional
        机器学习模型
    varlist : list, optional
        模型特征列表
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    grp_name : str, optional
        分组字段名
    min_data_size : int, default 100
        每组最小样本数
    grp_colname : str, optional
        分组结果列名
    sync_range : bool, default True
        是否同步分箱边界
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 100
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    retSummary : bool, default False
        是否只返回汇总指标
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    ascending : bool, default False
        分箱顺序是否升序
    withSummary : bool, default False
        是否包含总体汇总行
    wholeGroup : bool, default False
        是否使用全部数据分组
    add_func : callable, optional
        自定义统计函数
    
    Returns
    -------
    pandas.DataFrame
        分组收益表
    """
    
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
        
        grp = grp_list[i]
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
                     gains_table = False):
    """
    计算分组性能评估汇总。
    
    对训练集、验证集和oot样本进行分组性能评估，根据oot_grp_name
    字段分组后分别计算各组的性能指标。
    
    Parameters
    ----------
    train : pandas.DataFrame, optional
        训练数据集
    validation : pandas.DataFrame, optional
        验证数据集
    oot : pandas.DataFrame, optional
        oot数据集
    tgt_name : str
        目标变量名
    scr_name : str, optional
        分数字段名
    model : sklearn-like model, optional
        机器学习模型
    feature_cols : list, optional
        模型特征列表
    fig_save_path : str, optional
        图片保存路径
    rpt_save_path : str, optional
        报告保存路径
    to_show : bool, default False
        是否显示图形
    display : bool, default True
        是否打印结果
    dist_bins : int, default 20
        分布分箱数
    pct_bins : int, default 10
        百分比分箱数
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default False
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    init_equi_bins : int, default 1000
        初始等频分箱数量
    chi2_p : float, default 0.9
        卡方检验显著性水平
    oot_grp_name : str, optional
        oot分组字段名
    min_data_size : int, default 100
        每组最小样本数
    grp_colname : str, optional
        分组结果列名
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    gains_table : bool, default True
        是否计算收益表
    
    Returns
    -------
    pandas.DataFrame
        分组性能评估汇总表
    """
    
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
        
        grp = grp_list[i]
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
               tree_binning = False, random_state = 42):
    """
    创建交叉风险表。
    
    对两个分数字段进行分箱后，计算交叉分组的风险聚合值。
    支持对数值型变量进行自动分箱。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    score_list : list
        分数字段列表，长度为2
    dep : str
        目标变量名
    nbins : int or list
        分箱数量（整数或长度为2的列表）
    agg_col : str, optional
        聚合列名，默认为dep
    precision : int or list, default 5
        边界值精度
    min_bin_prop : float or list, default 0.05
        每箱最小样本占比
    include_missing : bool, default False
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱
    binning_numeric : list, default [True, True]
        是否对数值型字段分箱
    agg_func : str, callable, tuple or dict, default 'mean'
        聚合函数。
        
        常规用法与 ``pandas.crosstab`` 的 ``aggfunc`` 一致，例如 ``'mean'``、
        ``'sum'``、``'count'`` 或自定义函数。
        
        也支持直接计算两个字段聚合后的比值：
        
        1. 简写形式：
           ``agg_col=(numerator_col, denominator_col), agg_func='ratio'``
        2. tuple形式：
           ``agg_func=('ratio', numerator_col, denominator_col)``
        3. dict形式：
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
        
        当 ``return_count=True`` 时，返回结果的 columns 会增加一层指标名，
        包含 ratio 矩阵和 N 矩阵；当 ``return_count_pct=True`` 时，
        额外返回每个格子的 N 占比矩阵，计算方式为：
        当前格子 count / 右下角 Total count。
    chi2_method : bool, default False
        是否使用卡方分箱
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 100
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    
    Returns
    -------
    pandas.DataFrame
        交叉风险表
    
    Examples
    --------
    >>> cross_risk(data, score_list=['score1', 'score2'], dep='target', nbins=10)
    """
    
    from pandas.api.types import is_numeric_dtype
    
    if agg_col is None:
        agg_col = dep

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
    计算分组自定义指标收益表。
    
    根据分组字段对数据进行分组，分别计算每个分组的自定义指标收益表。
    收益表包含基础统计指标以及自定义指标的聚合值。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    dep : str
        目标变量名
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    score : str, optional
        分数字段名
    model : sklearn-like model, optional
        机器学习模型
    varlist : list, optional
        模型特征列表
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    grp_name : str, optional
        分组字段名
    min_data_size : int, default 100
        每组最小样本数
    grp_colname : str, optional
        分组结果列名
    sync_range : bool, default True
        是否同步分箱边界
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 100
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    ascending : bool, default True
        分箱顺序是否升序
    eval_metrics : list, default ["age", "monthly_income", "education"]
        需要统计的自定义指标列表
    metric_agg_func : str or callable, default "mean"
        自定义指标的聚合函数
    withSummary : bool, default False
        是否包含总体汇总行
    
    Returns
    -------
    pandas.DataFrame
        分组自定义指标收益表
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
        
        grp = grp_list[i]
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
    计算分数重复率。
    
    计算分数中非唯一值的比例，即存在重复的样本占比。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    score : str
        分数字段名
    
    Returns
    -------
    float
        分数重复率（0到1之间）
    
    Examples
    --------
    >>> tie_score_rate(data, 'score')
    0.15  # 表示15%的样本存在分数重复
    """
    
    n_unique_scr = len(data[score].unique())
    unique_scr_prop = n_unique_scr / data.shape[0]
    return (1 - unique_scr_prop)


def score_unique_rate(data, score):
    """
    计算分数唯一率。
    
    计算分数中唯一值占总样本数的比例。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表（此参数保留但未使用）
    score : array-like
        分数字段或数组
    
    Returns
    -------
    float
        分数唯一率（0到1之间）
    
    Examples
    --------
    >>> score_unique_rate(data, data['score'])
    0.85  # 表示85%的样本分数是唯一的
    """
    
    return len(np.unique(score)) / len(score)


class GainsTableCalculator:
    """
    收益表计算器。
    
    整合了收益表计算的多种功能，支持基础收益表和自定义指标收益表。
    提供面向对象的接口进行分组收益表计算。
    
    Parameters
    ----------
    data : pandas.DataFrame
        输入数据表
    dep : str
        目标变量名
    nbins : int, default 10
        分箱数量
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default True
        是否包含缺失值
    score : str, optional
        分数字段名
    model : sklearn-like model, optional
        机器学习模型
    varlist : list, optional
        模型特征列表
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    chi2_p : float, default 0.95
        卡方检验显著性水平
    init_equi_bins : int, default 100
        初始等频分箱数量
    fillna : any, default -999999
        缺失值填充值
    spec_values : list, default []
        特殊值列表
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    ascending : bool, default False
        分箱顺序是否升序
    
    Examples
    --------
    >>> calc = GainsTableCalculator(data, dep='target', score='score', nbins=10)
    >>> result = calc.calculate(grp_name='region')
    """
    
    def __init__(self, data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05,
                 include_missing = True, score = None, model = None, varlist = None,
                 equal_freq = True, chi2_method = False, chi2_p = 0.95, 
                 init_equi_bins = 100, fillna = -999999, spec_values = [],
                 tree_binning = False, random_state = 42, ascending = False):
        """
        初始化收益表计算器。
        
        Parameters
        ----------
        data : pandas.DataFrame
            输入数据表
        dep : str
            目标变量名
        nbins : int, default 10
            分箱数量
        precision : int, default 5
            边界值精度
        min_bin_prop : float, default 0.05
            每箱最小样本占比
        include_missing : bool, default True
            是否包含缺失值
        score : str, optional
            分数字段名
        model : sklearn-like model, optional
            机器学习模型
        varlist : list, optional
            模型特征列表
        equal_freq : bool, default True
            True为等频分箱
        chi2_method : bool, default False
            是否使用卡方分箱
        chi2_p : float, default 0.95
            卡方检验显著性水平
        init_equi_bins : int, default 100
            初始等频分箱数量
        fillna : any, default -999999
            缺失值填充值
        spec_values : list, default []
            特殊值列表
        tree_binning : bool, default False
            是否使用决策树分箱
        random_state : int, default 42
            随机种子
        ascending : bool, default False
            分箱顺序是否升序
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
    
    def calculate(self, grp_name = None, min_data_size = 100, grp_colname = None,
                  sync_range = True, retSummary = False, withSummary = False,
                  wholeGroup = False, add_func = None):
        """
        计算收益表。
        
        Parameters
        ----------
        grp_name : str, optional
            分组字段名
        min_data_size : int, default 100
            每组最小样本数
        grp_colname : str, optional
            分组结果列名
        sync_range : bool, default True
            是否同步分箱边界
        retSummary : bool, default False
            是否只返回汇总指标
        withSummary : bool, default False
            是否包含总体汇总行
        wholeGroup : bool, default False
            是否使用全部数据分组
        add_func : callable, optional
            自定义统计函数
        
        Returns
        -------
        pandas.DataFrame
            收益表
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
            add_func = add_func
        )


class PerformanceEvaluator:
    """
    性能评估器。
    
    整合了模型性能评估的多种功能，支持多数据集、多分组的性能评估。
    提供面向对象的接口进行性能指标计算和汇总。
    
    Parameters
    ----------
    tgt_name : str or list of str
        目标变量名。可传入多个 y 标签的 list/tuple → 逐标签评估后纵向拼接 (输出新增 tgt_name 列),
        并为每个标签各自出图。
    scr_name : str, optional
        分数字段名
    model : sklearn-like model, optional
        机器学习模型
    feature_cols : list, optional
        模型特征列表
    dist_bins : int, default 20
        分布分箱数
    pct_bins : int, default 10
        百分比分箱数
    precision : int, default 5
        边界值精度
    min_bin_prop : float, default 0.05
        每箱最小样本占比
    include_missing : bool, default False
        是否包含缺失值
    equal_freq : bool, default True
        True为等频分箱
    chi2_method : bool, default False
        是否使用卡方分箱
    init_equi_bins : int, default 1000
        初始等频分箱数量
    chi2_p : float, default 0.9
        卡方检验显著性水平
    tree_binning : bool, default False
        是否使用决策树分箱
    random_state : int, default 42
        随机种子
    
    Examples
    --------
    >>> evaluator = PerformanceEvaluator(tgt_name='target', model=model, feature_cols=features)
    >>> evaluator.add_dataset('train', train_df)
    >>> evaluator.add_dataset('validation', val_df)
    >>> evaluator.add_dataset('oot', oot_df)
    >>> result = evaluator.evaluate()
    >>>
    >>> # 多 y 标签: tgt_name 传 list → 输出含 tgt_name 列的纵向拼接表, 每个标签各自出图
    >>> evaluator = PerformanceEvaluator(tgt_name=['bad_dpd7', 'bad_dpd30'], model=model, feature_cols=features)
    >>> evaluator.add_dataset('train', train_df).add_dataset('oot', oot_df)
    >>> result = evaluator.evaluate(to_show=True)
    """
    
    def __init__(self, tgt_name, scr_name = None, model = None, feature_cols = None,
                 dist_bins = 20, pct_bins = 10, precision = 5, min_bin_prop = 0.05,
                 include_missing = False, equal_freq = True, chi2_method = False,
                 init_equi_bins = 1000, chi2_p = 0.9, tree_binning = False, random_state = 42):
        """
        初始化性能评估器。
        
        Parameters
        ----------
        tgt_name : str or list of str
            目标变量名。可传入多个 y 标签的 list/tuple → 逐标签评估后纵向拼接 (输出新增 tgt_name 列),
            并为每个标签各自出图。
        scr_name : str, optional
            分数字段名
        model : sklearn-like model, optional
            机器学习模型
        feature_cols : list, optional
            模型特征列表
        dist_bins : int, default 20
            分布分箱数
        pct_bins : int, default 10
            百分比分箱数
        precision : int, default 5
            边界值精度
        min_bin_prop : float, default 0.05
            每箱最小样本占比
        include_missing : bool, default False
            是否包含缺失值
        equal_freq : bool, default True
            True为等频分箱
        chi2_method : bool, default False
            是否使用卡方分箱
        init_equi_bins : int, default 1000
            初始等频分箱数量
        chi2_p : float, default 0.9
            卡方检验显著性水平
        tree_binning : bool, default False
            是否使用决策树分箱
        random_state : int, default 42
            随机种子
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
        self.datasets = {}
    
    def add_dataset(self, name, data):
        """
        添加数据集。
        
        Parameters
        ----------
        name : str
            数据集名称（如'train'、'validation'、'oot'）
        data : pandas.DataFrame
            数据集
        
        Returns
        -------
        self
            返回自身以便链式调用
        """
        self.datasets[name] = data
        return self
    
    def evaluate(self, oot_grp_name = None, min_data_size = 100, grp_colname = None,
                 fig_save_path = None, rpt_save_path = None, to_show = False, 
                 display = True, gains_table = False, benchmark_dataset = None):
        """
        执行性能评估。
        
        Parameters
        ----------
        oot_grp_name : str, optional
            oot分组字段名
        min_data_size : int, default 100
            每组最小样本数
        grp_colname : str, optional
            分组结果列名
        fig_save_path : str, optional
            图片保存路径
        rpt_save_path : str, optional
            报告保存路径
        to_show : bool, default False
            是否显示图形
        display : bool, default True
            是否打印结果
        gains_table : bool, default True
            是否计算收益表
        benchmark_dataset : str or pandas.DataFrame, optional
            固定分箱边界的基准数据集。若传入str, 则从add_dataset添加的数据集中按名称获取；
            若传入DataFrame, 则直接使用该DataFrame。默认None表示各数据集独立分箱。
        
        Returns
        -------
        pandas.DataFrame
            性能评估汇总表。若实例的 ``tgt_name`` 为多标签 list/tuple, 则对每个标签分别评估,
            结果新增 ``tgt_name`` 列后纵向拼接; ``to_show=True`` 时每个标签各出一张图,
            ``fig_save_path`` 自动按标签加后缀 (如 ``perf.png`` → ``perf_<label>.png``)。
        """
        if len(self.datasets) == 0:
            return pd.DataFrame([])

        if self.scr_name is None and self.model is None and self.feature_cols is None:
            return -1
        
        if self.scr_name is None and self.model is None:
            return -2
        
        if self.scr_name is None and self.feature_cols is None:
            return -3

        # ── 多 y 标签支持: tgt_name 为 list/tuple 时, 逐标签评估后纵向拼接 ──
        #    表格: 每个标签结果新增 tgt_name 列后纵向拼接;
        #    图片: 每个标签各自出图 (to_show=True 时循环显示, fig_save_path 自动按标签加后缀)。
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
                        rpt_save_path = None,        # 拼接后统一保存 (见下)
                        to_show = to_show,           # 每个标签各自出图
                        display = False,             # 拼接后统一展示
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
            return fnl_df

        def _get_score(data):
            if self.scr_name is not None:
                return data[self.scr_name]
            return self.model.predict_proba(data.loc[:, self.feature_cols])[:, 1]

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
                ascending = True
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
                eval_datasets[name] = {
                    "y_true": data[self.tgt_name],
                    "y_score": _get_score(data)
                }

            if len(eval_datasets) == 0:
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
                pct_bin_edges = benchmark_bin_edges
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
                        random_state = self.random_state
                    )
                gains_res['index'] = name
                gains_summ_list.append(gains_res)

            if gains_summ_list:
                gains_summ = pd.concat(gains_summ_list)
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

        return fnl_df
