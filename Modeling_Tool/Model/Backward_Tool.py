"""
向后变量消除工具包（统一版）
=============================

本模块提供基于LightGBM和XGBoost的向后变量消除（Backward Variable Elimination）功能，
通过累计特征重要性阈值进行变量筛选，并支持训练后的性能分析。

Functions
---------
backward_lgbm
    使用LightGBM模型进行向后变量消除。
backward_xgbm
    使用XGBoost模型进行向后变量消除。
get_backward_perf
    获取后退消元性能报告（整体分析）。
get_backward_summary
    获取后退消元性能报告（分组分析）。

Classes
-------
BackwardVariableEliminator
    向后变量消除统一封装类，支持训练和分析一体化操作。

Examples
--------
# 方式一：训练 + 分析一体化
>>> eliminator = BackwardVariableEliminator(
...     model_type='lgb',
...     train_data=df_train,
...     validation_data=df_val,
...     oot_data=df_oot,
...     params={'n_estimators': 100, 'learning_rate': 0.1},
...     y='target'
... )
>>> eliminator.fit(
...     x=['var1', 'var2', 'var3'],
...     results_output_dir='./results/',
...     modelsave_dir='./models/'
... ).analyze()

# 方式二：仅训练
>>> eliminator = BackwardVariableEliminator('lgb', df_train, df_val, params, 'target')
>>> eliminator.fit(x=['var1', 'var2'], results_output_dir='./', modelsave_dir='./')

# 方式三：仅分析（使用已保存的模型）
>>> eliminator = BackwardVariableEliminator(
...     train_data=df_train,
...     validation_data=df_val,
...     oot_data=df_oot,
...     y='target',
...     modelsave_dir='./models/'
... )
>>> result = eliminator.analyze()
"""

import os
import time
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from collections import OrderedDict

from .GBM_Tool import lgbm_quick_train, lgb_varimp, xgbm_quick_train, xgb_varimp
from Modeling_Tool.Core.utils import save_model, load_model
from Modeling_Tool.Eval.Model_Eval_Tool import get_perf_summary


# ============================================================================
# 原函数（保持独立）- backward_lgbm
# ============================================================================

def backward_lgbm(train_data, validation_data,
                  x, y, varreduct_params, results_output_dir,
                  varreduct_perf_filename, modelsave_dir,
                  cat_x=None, wgt_col=None,
                  test_data_dict=None,
                  stopping_metric="AUC",
                  seed=12345,
                  varreduct_modelid_prefix="varreduct_test0",
                  varreduct_max_nmodels=50, varreduct_minvar=30, cum_varimp=0.98,
                  backward_continue_from_nmodel=0,
                  scrutiny_range=None,
                  scrutiny_step=None):
    """使用LightGBM模型进行向后变量消除。

    基于累计特征重要性阈值，对LightGBM模型进行向后变量筛选。
    支持断点续跑、最小变量数限制、最大迭代次数限制等功能。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集
    validation_data : pd.DataFrame
        验证数据集
    x : list
        待筛选的特征变量名列表
    y : str
        目标变量列名
    varreduct_params : dict
        LightGBM模型参数字典，必须包含模型训练所需参数
    results_output_dir : str
        结果输出目录路径（需以斜杠结尾）
    varreduct_perf_filename : str
        性能结果文件名
    modelsave_dir : str
        模型保存目录路径（需以斜杠结尾）
    cat_x : list, optional
        类别型特征列表，默认为None
    wgt_col : str, optional
        样本权重列名，默认为None
    test_data_dict : dict, optional
        测试数据集字典，默认为空字典
    stopping_metric : str, optional
        早停评估指标，默认为"AUC"
    seed : int, optional
        随机种子，默认为12345
    varreduct_modelid_prefix : str, optional
        模型ID前缀，默认为"varreduct_test0"
    varreduct_max_nmodels : int, optional
        最大迭代模型数量，默认为50
    varreduct_minvar : int, optional
        变量筛选最小数量阈值，默认为30
    cum_varimp : float, optional
        累计特征重要性阈值，默认为0.98
    backward_continue_from_nmodel : int, optional
        断点续跑参数：
        - 0: 从头开始
        - 正整数: 从指定模型号继续
        - 负整数: 从最后一个模型继续
    scrutiny_range : tuple/list, optional
        精确筛选的范围，元组形式(下限, 上限)
    scrutiny_step : int, optional
        精确筛选每次减少的变量数

    Returns
    -------
    int
        返回0表示执行完成

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'learning_rate': 0.1,
    ...     'early_stopping_rounds': 10
    ... }
    >>> backward_lgbm(
    ...     train_data=df_train,
    ...     validation_data=df_val,
    ...     x=['var1', 'var2', 'var3'],
    ...     y='target',
    ...     varreduct_params=params,
    ...     results_output_dir='./results/',
    ...     varreduct_perf_filename='perf.csv',
    ...     modelsave_dir='./models/'
    ... )
    """
    if test_data_dict is None:
        test_data_dict = {}

    start = time.time()
    modelsave_dir = os.path.abspath(modelsave_dir)
    results_output_dir = os.path.abspath(results_output_dir)
    os.makedirs(modelsave_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    ret_model_dict = {}

    datain_all = OrderedDict()
    datain_all["mdl"] = train_data
    if validation_data is not None:
        datain_all["hd"] = validation_data
    datain_all.update(test_data_dict)

    # 检查数据格式一致性
    try:
        for k, v in datain_all.items():
            assert isinstance(v, pd.DataFrame)
    except AssertionError:
        logging.warning("请提供pandas.DataFrame格式数据")
        exit(1)

    # 预设参数（保证模型可复现性）
    hyperparams_preset = {
        "metric": stopping_metric,
        "seed": seed,
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_threads": 8
    }

    # 补充缺失的必需参数
    lacked_params = [k for k in list(hyperparams_preset.keys()) if k not in list(varreduct_params.keys())]
    for param in lacked_params:
        logging.info(f"updating params {param}")
        varreduct_params[param] = hyperparams_preset[param]

    varreduct_params_new = varreduct_params
    varreduct_perf = pd.DataFrame()
    varreduct_process_x = pd.DataFrame()

    # 初始化迭代计数器
    i = 1

    # 处理断点续跑
    if backward_continue_from_nmodel != 0:
        last_backward_process_perf_dir = os.path.abspath(os.path.join(results_output_dir, varreduct_perf_filename))
        last_backward_process_x_dir = os.path.abspath(os.path.join(results_output_dir, varreduct_modelid_prefix + "_varlist.csv"))

        if backward_continue_from_nmodel > 0:
            last_model_path = os.path.abspath(os.path.join(modelsave_dir, varreduct_modelid_prefix + f"{backward_continue_from_nmodel}.pkl"))
        if backward_continue_from_nmodel < 0 and os.path.exists(last_backward_process_perf_dir):
            last_model_id = pd.read_csv(last_backward_process_perf_dir)["Model_id"].tolist()[backward_continue_from_nmodel]
            last_model_path = os.path.abspath(os.path.join(modelsave_dir, f"{last_model_id}.pkl"))
            tmp_backward_continue_from_nmodel = len(pd.read_csv(last_backward_process_perf_dir)["Model_id"].tolist()) + backward_continue_from_nmodel + 1
            backward_continue_from_nmodel = tmp_backward_continue_from_nmodel

        if os.path.exists(last_backward_process_perf_dir) and os.path.exists(last_backward_process_x_dir):
            varreduct_perf = pd.read_csv(last_backward_process_perf_dir)[:backward_continue_from_nmodel]
            varreduct_process_x = pd.read_csv(last_backward_process_x_dir).iloc[:, 0:2 * backward_continue_from_nmodel]
        else:
            logging.warning("无法找到之前的变量消除结果，将从头开始")

        if os.path.exists(last_model_path):
            last_model = load_model(last_model_path)
            var_imp = lgb_varimp(last_model).sort_values(by='Gain_Rank')
            last_model_xgb_varimp = var_imp.rename(columns={"Imp_Gain_Wight": "percentage", "Var_Name": "variable"})
            last_model_xgb_varimp["cum_pctg"] = np.cumsum(last_model_xgb_varimp.percentage)

            last_model_len_varimp = last_model_xgb_varimp.shape[0]
            if scrutiny_range is not None and scrutiny_step is not None and last_model_len_varimp > scrutiny_range[0] and last_model_len_varimp <= scrutiny_range[1]:
                nindex = last_model_len_varimp - scrutiny_step
            else:
                nindex = last_model_xgb_varimp.cum_pctg[last_model_xgb_varimp.cum_pctg > cum_varimp].index.to_list()[0]

            last_model_x_raw1 = list(last_model_xgb_varimp.variable[:(nindex), ])
            last_model_x_raw2 = [x.split(".")[0] for x in last_model_x_raw1]
            x = list(OrderedDict.fromkeys(last_model_x_raw2))

            var_ndiff = len(set([x.split('.')[0] for x in last_model_xgb_varimp.variable.tolist()])) - len(x)
            if var_ndiff == 0:
                logging.warning("本次迭代变量与上次相同，自动移除最不重要的变量以避免循环")
                x = x[:-1]

            i = backward_continue_from_nmodel + 1
            logging.info(f"从Model #{backward_continue_from_nmodel}继续，包含{last_model_len_varimp}个变量")

    logging.info("向后变量选择过程开始，共%s个变量", len(x))

    if wgt_col is None:
        train_data["wgt"] = 1
        validation_data["wgt"] = 1
        wgt_col = "wgt"

    while(len(x) > varreduct_minvar & i <= varreduct_max_nmodels):

        cat_x_train = [var for var in cat_x if var in x] if cat_x and len(cat_x) > 0 else None
        xgbm_quick = lgbm_quick_train(train_data, validation_data, x, y, varreduct_params_new, wgt_col, cat_x_train)

        if save_model:
            save_model(xgbm_quick, os.path.join(modelsave_dir, f"{varreduct_modelid_prefix + str(i)}.pkl"))

        xgb_varimp = lgb_varimp(xgbm_quick.booster_).sort_values(by='rank')
        ret_model_dict[varreduct_modelid_prefix + str(i)] = xgbm_quick
        xgb_varimp["cum_pctg"] = np.cumsum(xgb_varimp.percentage)

        len_varimp = xgb_varimp.shape[0]
        if scrutiny_range is not None and scrutiny_step is not None and len_varimp > scrutiny_range[0] and len_varimp <= scrutiny_range[1]:
            nindex = len_varimp - scrutiny_step
        else:
            xgb_varimp["cum_pctg"] = np.cumsum(xgb_varimp.percentage)
            nindex = xgb_varimp.cum_pctg[xgb_varimp.cum_pctg > cum_varimp].index.to_list()[0]

        allxlist = xgb_varimp[["variable", "percentage"]]
        allxlist.columns = ['variable_' + str(i), 'percentage_' + str(i)]
        varreduct_process_x = pd.concat([varreduct_process_x, allxlist], axis=1)

        varreduct_process_x.to_csv(os.path.join(results_output_dir, varreduct_modelid_prefix + "_varlist.csv"), index=False)

        nvars_varimp_gt0 = sum(xgb_varimp["percentage"] > 0)
        nvars = len(xgb_varimp["percentage"])
        nvars_input_unique = len(x)
        logging.info(f"开始训练模型，使用{nvars_input_unique}个变量")

        len_keep = nindex + 1
        if scrutiny_range is None and scrutiny_step is None and len_keep == len_varimp:
            logging.info("变量选择因cum_varimp=%s完成于第%s轮", cum_varimp, str(i))
            opt_final = i
            break
        elif len_keep < varreduct_minvar:
            logging.info("变量选择因varreduct_minvar=%s完成于第%s轮", varreduct_minvar, str(i))
            opt_final = i
            break
        elif i == varreduct_max_nmodels:
            logging.info("变量选择因max_iter=%s完成于第%s轮", varreduct_max_nmodels, str(i))
            opt_final = i
            break
        else:
            x_raw1 = list(xgb_varimp.variable[:(nindex), ])
            x_raw2 = [var.split(".")[0] for var in x_raw1]
            x = list(OrderedDict.fromkeys(x_raw2))

            var_ndiff = len(set([var.split('.')[0] for var in xgb_varimp.variable.tolist()])) - len(x)
            if var_ndiff == 0:
                logging.warning("本次迭代变量与上次相同，自动移除最不重要的变量")
                x = x[:-1]
            i = i + 1

    return 0


# ============================================================================
# 原函数（保持独立）- backward_xgbm
# ============================================================================

def backward_xgbm(train_data, validation_data,
                  x, y, varreduct_params, results_output_dir,
                  varreduct_perf_filename, modelsave_dir,
                  mc_dict=None, cat_x=None, wgt_col=None,
                  test_data_dict=None,
                  stopping_metric="AUC",
                  seed=12345,
                  varreduct_modelid_prefix="varreduct_test0",
                  varreduct_max_nmodels=50, varreduct_minvar=30, cum_varimp=0.98,
                  backward_continue_from_nmodel=0,
                  scrutiny_range=None,
                  scrutiny_step=None):
    """使用XGBoost模型进行向后变量消除。

    基于累计特征重要性阈值，对XGBoost模型进行向后变量筛选。
    支持单调性约束、断点续跑、最小变量数限制等功能。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集
    validation_data : pd.DataFrame
        验证数据集
    x : list
        待筛选的特征变量名列表
    y : str
        目标变量列名
    varreduct_params : dict
        XGBoost模型参数字典，必须包含模型训练所需参数
    results_output_dir : str
        结果输出目录路径（需以斜杠结尾）
    varreduct_perf_filename : str
        性能结果文件名
    modelsave_dir : str
        模型保存目录路径（需以斜杠结尾）
    mc_dict : dict, optional
        单调性约束字典，默认为None
    cat_x : list, optional
        类别型特征列表，默认为None
    wgt_col : str, optional
        样本权重列名，默认为None
    test_data_dict : dict, optional
        测试数据集字典，默认为空字典
    stopping_metric : str, optional
        早停评估指标，默认为"AUC"
    seed : int, optional
        随机种子，默认为12345
    varreduct_modelid_prefix : str, optional
        模型ID前缀，默认为"varreduct_test0"
    varreduct_max_nmodels : int, optional
        最大迭代模型数量，默认为50
    varreduct_minvar : int, optional
        变量筛选最小数量阈值，默认为30
    cum_varimp : float, optional
        累计特征重要性阈值，默认为0.98
    backward_continue_from_nmodel : int, optional
        断点续跑参数：
        - 0: 从头开始
        - 正整数: 从指定模型号继续
        - 负整数: 从最后一个模型继续
    scrutiny_range : tuple/list, optional
        精确筛选的范围，元组形式(下限, 上限)
    scrutiny_step : int, optional
        精确筛选每次减少的变量数

    Returns
    -------
    int
        返回0表示执行完成

    Examples
    --------
    >>> params = {
    ...     'n_estimators': 100,
    ...     'learning_rate': 0.1,
    ...     'max_depth': 5
    ... }
    >>> backward_xgbm(
    ...     train_data=df_train,
    ...     validation_data=df_val,
    ...     x=['var1', 'var2', 'var3'],
    ...     y='target',
    ...     varreduct_params=params,
    ...     results_output_dir='./results/',
    ...     varreduct_perf_filename='perf.csv',
    ...     modelsave_dir='./models/',
    ...     mc_dict={'var1': 1}  # 单调性约束
    ... )
    """
    if mc_dict is None:
        mc_dict = {}
    if cat_x is None:
        cat_x = []
    if test_data_dict is None:
        test_data_dict = {}

    start = time.time()
    modelsave_dir = os.path.abspath(modelsave_dir)
    results_output_dir = os.path.abspath(results_output_dir)
    os.makedirs(modelsave_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    ret_model_dict = {}

    datain_all = OrderedDict()
    datain_all["mdl"] = train_data
    if validation_data is not None:
        datain_all["hd"] = validation_data
    datain_all.update(test_data_dict)

    # 检查数据格式一致性
    try:
        for k, v in datain_all.items():
            assert isinstance(v, pd.DataFrame)
    except AssertionError:
        logging.warning("请提供pandas.DataFrame格式数据")
        exit(1)

    # 预设参数（保证模型可复现性）
    hyperparams_preset = {
        'eval_metric': stopping_metric,
        'tree_method': 'exact',
        'booster': 'gbtree',
        'seed': seed,
        'monotone_constraints': mc_dict
    }

    # 补充缺失的必需参数
    lacked_params = [k for k in list(hyperparams_preset.keys()) if k not in list(varreduct_params.keys())]
    for param in lacked_params:
        logging.info(f"updating params {param}")
        varreduct_params[param] = hyperparams_preset[param]

    varreduct_params_new = varreduct_params
    varreduct_perf = pd.DataFrame()
    varreduct_process_x = pd.DataFrame()

    # 初始化迭代计数器
    i = 1

    # 处理断点续跑
    if backward_continue_from_nmodel != 0:
        last_backward_process_perf_dir = os.path.abspath(os.path.join(results_output_dir, varreduct_perf_filename))
        last_backward_process_x_dir = os.path.abspath(os.path.join(results_output_dir, varreduct_modelid_prefix + "_varlist.csv"))

        if backward_continue_from_nmodel > 0:
            last_model_path = os.path.abspath(os.path.join(modelsave_dir, varreduct_modelid_prefix + f"{backward_continue_from_nmodel}.pkl"))
        if backward_continue_from_nmodel < 0 and os.path.exists(last_backward_process_perf_dir):
            last_model_id = pd.read_csv(last_backward_process_perf_dir)["Model_id"].tolist()[backward_continue_from_nmodel]
            last_model_path = os.path.abspath(os.path.join(modelsave_dir, f"{last_model_id}.pkl"))
            tmp_backward_continue_from_nmodel = len(pd.read_csv(last_backward_process_perf_dir)["Model_id"].tolist()) + backward_continue_from_nmodel + 1
            backward_continue_from_nmodel = tmp_backward_continue_from_nmodel

        if os.path.exists(last_backward_process_perf_dir) and os.path.exists(last_backward_process_x_dir):
            varreduct_perf = pd.read_csv(last_backward_process_perf_dir)[:backward_continue_from_nmodel]
            varreduct_process_x = pd.read_csv(last_backward_process_x_dir).iloc[:, 0:2 * backward_continue_from_nmodel]
        else:
            logging.warning("无法找到之前的变量消除结果，将从头开始")

        if os.path.exists(last_model_path):
            last_model = load_model(last_model_path)
            var_imp = xgb_varimp(last_model)
            xgb_varimp_res = var_imp
            last_model_xgb_varimp = xgb_varimp_res.copy()
            last_model_xgb_varimp["cum_pctg"] = np.cumsum(last_model_xgb_varimp.percentage)

            last_model_len_varimp = last_model_xgb_varimp.shape[0]
            if scrutiny_range is not None and scrutiny_step is not None and last_model_len_varimp > scrutiny_range[0] and last_model_len_varimp <= scrutiny_range[1]:
                nindex = last_model_len_varimp - scrutiny_step
            else:
                nindex = last_model_xgb_varimp.cum_pctg[last_model_xgb_varimp.cum_pctg > cum_varimp].index.to_list()[0]

            last_model_x_raw1 = list(last_model_xgb_varimp.variable[:(nindex), ])
            last_model_x_raw2 = [var.split(".")[0] for var in last_model_x_raw1]
            x = list(OrderedDict.fromkeys(last_model_x_raw2))

            var_ndiff = len(set([var.split('.')[0] for var in last_model_xgb_varimp.variable.tolist()])) - len(x)
            if var_ndiff == 0:
                logging.warning("本次迭代变量与上次相同，自动移除最不重要的变量以避免循环")
                x = x[:-1]

            i = backward_continue_from_nmodel + 1
            logging.info(f"从Model #{backward_continue_from_nmodel}继续，包含{last_model_len_varimp}个变量")

    logging.info("向后变量选择过程开始，共%s个变量", len(x))

    if wgt_col is None:
        train_data["wgt"] = 1
        validation_data["wgt"] = 1
        wgt_col = "wgt"

    while(len(x) > varreduct_minvar & i <= varreduct_max_nmodels):

        cat_x_train = [var for var in cat_x if var in x] if len(cat_x) > 0 else None
        varreduct_params_new['monotone_constraints'] = mc_dict
        xgbm_quick = xgbm_quick_train(train_data, validation_data, x, y, wgt_col, varreduct_params_new, cat_x_train)

        if save_model:
            save_model(xgbm_quick, os.path.join(modelsave_dir, f"{varreduct_modelid_prefix + str(i)}.pkl"))

        xgb_varimp_res = xgb_varimp(xgbm_quick).sort_values(by='rank')
        ret_model_dict[varreduct_modelid_prefix + str(i)] = xgbm_quick
        xgb_varimp_res["cum_pctg"] = np.cumsum(xgb_varimp_res.percentage)

        len_varimp = xgb_varimp_res.shape[0]
        if scrutiny_range is not None and scrutiny_step is not None and len_varimp > scrutiny_range[0] and len_varimp <= scrutiny_range[1]:
            nindex = len_varimp - scrutiny_step
        else:
            xgb_varimp_res["cum_pctg"] = np.cumsum(xgb_varimp_res.percentage)
            nindex = xgb_varimp_res.cum_pctg[xgb_varimp_res.cum_pctg > cum_varimp].index.to_list()[0]

        allxlist = xgb_varimp_res[["variable", "percentage"]]
        allxlist.columns = ['variable_' + str(i), 'percentage_' + str(i)]
        varreduct_process_x = pd.concat([varreduct_process_x, allxlist], axis=1)

        varreduct_process_x.to_csv(os.path.join(results_output_dir, varreduct_modelid_prefix + "_varlist.csv"), index=False)

        nvars_varimp_gt0 = sum(xgb_varimp_res["percentage"] > 0)
        nvars = len(xgb_varimp_res["percentage"])
        nvars_input_unique = len(x)
        logging.info(f"开始训练模型，使用{nvars_input_unique}个变量")

        len_keep = nindex + 1
        if scrutiny_range is None and scrutiny_step is None and len_keep == len_varimp:
            logging.info("变量选择因cum_varimp=%s完成于第%s轮", cum_varimp, str(i))
            opt_final = i
            break
        elif len_keep < varreduct_minvar:
            logging.info("变量选择因varreduct_minvar=%s完成于第%s轮", varreduct_minvar, str(i))
            opt_final = i
            break
        elif i == varreduct_max_nmodels:
            logging.info("变量选择因max_iter=%s完成于第%s轮", varreduct_max_nmodels, str(i))
            opt_final = i
            break
        else:
            x_raw1 = list(xgb_varimp_res.variable[:(nindex), ])
            x_raw2 = [var.split(".")[0] for var in x_raw1]
            x = list(OrderedDict.fromkeys(x_raw2))

            if mc_dict:
                mc_dict = {k: v for k, v in mc_dict.items() if k in x}

            var_ndiff = len(set([var.split('.')[0] for var in xgb_varimp_res.variable.tolist()])) - len(x)
            if var_ndiff == 0:
                logging.warning("本次迭代变量与上次相同，自动移除最不重要的变量")
                x = x[:-1]
            i = i + 1

    return 0


# ============================================================================
# 统一封装类：BackwardVariableEliminator
# ============================================================================

class BackwardVariableEliminator:
    """向后变量消除统一封装类。

    整合训练和分析功能，支持LightGBM和XGBoost两种模型，
    提供从变量消除训练到性能分析的一站式服务。

    Parameters
    ----------
    model_type : str
        模型类型，可选值：
        - 'lgb' 或 'lightgbm': 使用LightGBM
        - 'xgb' 或 'xgboost': 使用XGBoost
    train_data : pd.DataFrame
        训练数据集
    validation_data : pd.DataFrame
        验证数据集
    params : dict
        模型参数字典
    y : str
        目标变量列名
    oot_data : pd.DataFrame, optional
        OOT测试数据集，默认为None（用于后续分析）
    wgt_col : str, optional
        样本权重列名，默认为None
    cat_x : list, optional
        类别型特征列表，默认为None
    mc_dict : dict, optional
        单调性约束字典，仅XGBoost支持，默认为None
    stopping_metric : str, optional
        早停评估指标，默认为"AUC"
    seed : int, optional
        随机种子，默认为12345
    modelsave_dir : str, optional
        模型保存目录（用于后续分析），默认为None
    algorithm : str, optional
        算法类型，'xgb'或'lgb'，默认为'lgb'

    Attributes
    ----------
    model_type : str
        当前使用的模型类型
    train_data : pd.DataFrame
        训练数据集
    validation_data : pd.DataFrame
        验证数据集
    oot_data : pd.DataFrame
        OOT测试数据集
    params : dict
        模型参数字典
    y : str
        目标变量列名
    modelsave_dir : str
        模型保存目录
    algorithm : str
        算法类型
    models_ : dict
        存储迭代过程中的模型字典
    varimp_history_ : pd.DataFrame
        变量重要性历史记录
    result_ : pd.DataFrame
        分析结果（调用analyze后填充）

    Examples
    --------
    # 方式一：训练 + 分析一体化
    >>> eliminator = BackwardVariableEliminator(
    ...     model_type='lgb',
    ...     train_data=df_train,
    ...     validation_data=df_val,
    ...     oot_data=df_oot,
    ...     params={'n_estimators': 100, 'learning_rate': 0.1},
    ...     y='target',
    ...     modelsave_dir='./models/'
    ... )
    >>> eliminator.fit(
    ...     x=['var1', 'var2', 'var3'],
    ...     results_output_dir='./results/',
    ...     modelsave_dir='./models/'
    ... ).analyze()

    # 方式二：仅训练
    >>> eliminator = BackwardVariableEliminator(
    ...     model_type='lgb',
    ...     train_data=df_train,
    ...     validation_data=df_val,
    ...     params={'n_estimators': 100},
    ...     y='target'
    ... )
    >>> eliminator.fit(x=['var1', 'var2'], results_output_dir='./', modelsave_dir='./')

    # 方式三：仅分析（使用已保存的模型）
    >>> eliminator = BackwardVariableEliminator(
    ...     train_data=df_train,
    ...     validation_data=df_val,
    ...     oot_data=df_oot,
    ...     y='target',
    ...     modelsave_dir='./models/'
    ... )
    >>> result = eliminator.analyze()
    """

    SUPPORTED_MODELS = ['lgb', 'lightgbm', 'xgb', 'xgboost']

    def __init__(self, model_type=None, train_data=None, validation_data=None, params=None, y=None,
                 oot_data=None, wgt_col=None, cat_x=None, mc_dict=None,
                 stopping_metric="AUC", seed=12345,
                 modelsave_dir=None, algorithm='lgb',
                 results_output_dir=None, varreduct_perf_filename='perf.csv', varreduct_modelid_prefix="varreduct"):
        """初始化向后变量消除器。

        Parameters
        ----------
        model_type : str, optional
            模型类型（训练时必需）
        train_data : pd.DataFrame, optional
            训练数据集（训练时必需）
        validation_data : pd.DataFrame, optional
            验证数据集（训练时必需）
        params : dict, optional
            模型参数字典（训练时必需）
        y : str
            目标变量列名（分析时必需）
        oot_data : pd.DataFrame, optional
            OOT测试数据集（分析时使用）
        wgt_col : str, optional
            样本权重列名
        cat_x : list, optional
            类别型特征列表
        mc_dict : dict, optional
            单调性约束字典
        stopping_metric : str, optional
            早停评估指标
        seed : int, optional
            随机种子
        modelsave_dir : str, optional
            模型保存目录
        algorithm : str, optional
            算法类型，默认为'lgb'
        """
        # 训练相关参数
        self.model_type = model_type.lower() if model_type else None
        self.train_data = train_data
        self.validation_data = validation_data
        self.params = params
        self.y = y
        self.wgt_col = wgt_col
        self.cat_x = cat_x if cat_x else []
        self.mc_dict = mc_dict if mc_dict else {}
        self.stopping_metric = stopping_metric
        self.seed = seed

        # 分析相关参数
        self.oot_data = oot_data
        self.modelsave_dir = modelsave_dir
        self.algorithm = algorithm

        # 结果存储
        self.models_ = {}
        self.varimp_history_ = None
        self.result_ = None

        # 分组分析参数
        self.grp_name = None
        self.subgroup_value = None
        self.overfitting_cut = [np.inf, np.inf]
        self.min_data_size = 10
        
        # 文件存储参数
        self.results_output_dir = results_output_dir
        self.varreduct_perf_filename = varreduct_perf_filename
        self.varreduct_modelid_prefix = varreduct_modelid_prefix

    def _validate_train_params(self):
        """验证训练参数是否完整。

        Raises
        ------
        ValueError
            当缺少必需的训练参数时抛出
        """
        missing = []
        if self.model_type is None:
            missing.append("model_type")
        if self.train_data is None:
            missing.append("train_data")
        if self.validation_data is None:
            missing.append("validation_data")
        if self.params is None:
            missing.append("params")
        if self.y is None:
            missing.append("y")
        if missing:
            raise ValueError(f"训练参数不完整，缺少: {', '.join(missing)}")

    def _validate_analyze_params(self):
        """验证分析参数是否完整。

        Raises
        ------
        ValueError
            当缺少必需的分析参数时抛出
        """
        missing = []
        if self.y is None:
            missing.append("y (dep)")
        if self.modelsave_dir is None:
            missing.append("modelsave_dir")
        if missing:
            raise ValueError(f"分析参数不完整，缺少: {', '.join(missing)}")

    def _get_hyperparams_preset(self):
        """获取预设的超参数。

        Returns
        -------
        dict
            预设的超参数字典
        """
        if self.model_type in ['lgb', 'lightgbm']:
            return {
                "metric": self.stopping_metric,
                "seed": self.seed,
                "objective": "binary",
                "boosting_type": "gbdt",
                "num_threads": 8
            }
        else:
            return {
                'eval_metric': self.stopping_metric,
                'tree_method': 'exact',
                'booster': 'gbtree',
                'seed': self.seed,
                'monotone_constraints': self.mc_dict
            }

    def _merge_params(self):
        """合并用户参数和预设参数。

        Returns
        -------
        dict
            合并后的参数字典
        """
        merged = self.params.copy()
        preset = self._get_hyperparams_preset()
        for key, value in preset.items():
            if key not in merged:
                merged[key] = value
        if self.model_type in ['xgb', 'xgboost'] and self.mc_dict:
            merged['monotone_constraints'] = self.mc_dict
        return merged

    def fit(self, x, 
            varreduct_max_nmodels=50, 
            varreduct_minvar=30, 
            cum_varimp=0.98,
            backward_continue_from_nmodel=0,
            scrutiny_range=None, 
            scrutiny_step=None,
            test_data_dict=None):
        """执行向后变量消除训练。

        Parameters
        ----------
        x : list
            待筛选的特征变量名列表
        varreduct_max_nmodels : int, optional
            最大迭代模型数量，默认为50
        varreduct_minvar : int, optional
            变量筛选最小数量阈值，默认为30
        cum_varimp : float, optional
            累计特征重要性阈值，默认为0.98
        backward_continue_from_nmodel : int, optional
            断点续跑参数
        scrutiny_range : tuple/list, optional
            精确筛选的范围
        scrutiny_step : int, optional
            精确筛选每次减少的变量数
        test_data_dict : dict, optional
            测试数据集字典

        Returns
        -------
        self
            返回自身实例，支持链式调用

        Examples
        --------
        >>> eliminator = BackwardVariableEliminator('lgb', df_train, df_val, params, 'target')
        >>> eliminator.fit(
        ...     x=['var1', 'var2', 'var3'],
        ...     results_output_dir='./results/',
        ...     modelsave_dir='./models/'
        ... )
        """
        self._validate_train_params()

        if test_data_dict is None:
            test_data_dict = {}

#         # 同步modelsave_dir
#         if modelsave_dir:
#             self.modelsave_dir = modelsave_dir

        merged_params = self._merge_params()

        # 根据模型类型选择对应的函数
        if self.model_type in ['lgb', 'lightgbm']:
            result = backward_lgbm(
                train_data=self.train_data,
                validation_data=self.validation_data,
                x=x,
                y=self.y,
                varreduct_params=merged_params,
                results_output_dir=self.results_output_dir or self.modelsave_dir or './',
                varreduct_perf_filename=self.varreduct_perf_filename,
                modelsave_dir=self.modelsave_dir or self.modelsave_dir or './',
                cat_x=self.cat_x,
                wgt_col=self.wgt_col,
                test_data_dict=test_data_dict,
                stopping_metric=self.stopping_metric,
                seed=self.seed,
                varreduct_modelid_prefix=self.varreduct_modelid_prefix,
                varreduct_max_nmodels=varreduct_max_nmodels,
                varreduct_minvar=varreduct_minvar,
                cum_varimp=cum_varimp,
                backward_continue_from_nmodel=backward_continue_from_nmodel,
                scrutiny_range=scrutiny_range,
                scrutiny_step=scrutiny_step
            )
        else:
            result = backward_xgbm(
                train_data=self.train_data,
                validation_data=self.validation_data,
                x=x,
                y=self.y,
                varreduct_params=merged_params,
                results_output_dir=self.results_output_dir or self.modelsave_dir or './',
                varreduct_perf_filename=self.varreduct_perf_filename,
                modelsave_dir=self.modelsave_dir or self.modelsave_dir or './',
                mc_dict=self.mc_dict,
                cat_x=self.cat_x,
                wgt_col=self.wgt_col,
                test_data_dict=test_data_dict,
                stopping_metric=self.stopping_metric,
                seed=self.seed,
                varreduct_modelid_prefix=self.varreduct_modelid_prefix,
                varreduct_max_nmodels=varreduct_max_nmodels,
                varreduct_minvar=varreduct_minvar,
                cum_varimp=cum_varimp,
                backward_continue_from_nmodel=backward_continue_from_nmodel,
                scrutiny_range=scrutiny_range,
                scrutiny_step=scrutiny_step
            )

        return self
    
    def get_backward_perf(self, overfitting_cut = [np.inf, np.inf]):
        """
        生成后退消元模型的性能汇总。

        遍历指定目录下的所有模型文件，计算每个模型的性能指标，
        并按AUC降序排列。可根据过拟合阈值过滤结果。

        Parameters
        ----------
        overfitting_cut : list, default [np.inf, np.inf]
            过拟合阈值[AUC_Shift, KS_Shift]

        Returns
        -------
        pandas.DataFrame
            各模型性能汇总表，包含AUC_Shift、KS_Shift等指标

        Examples
        --------
        >>> perf = eliminator.get_backward_perf(overfitting_cut=[0.05, 0.05])
        """
        from tqdm import tqdm # type: ignore

        n_models = len([name for name in os.listdir(self.modelsave_dir) if name.endswith(".pkl")])
        logging.info(f"{n_models} models in var reduct folder.")

        col_names = ['model_id', "num_of_vars", 'index', 'N', 'KS', 'AUC', 'Btm10%_Lift', "Top10%_Lift", "AUC_Shift", "KS_Shift"]

        ins_auc = pd.DataFrame(columns = col_names)
        oos_auc = pd.DataFrame(columns = col_names)
        oot_auc = pd.DataFrame(columns = col_names)

        for i in tqdm(range(1, n_models + 1)):
            m = load_model(f"{self.modelsave_dir}/{self.varreduct_modelid_prefix}{str(i)}.pkl")

            if self.model_type in ['xgb', 'xgboost']:
                varlist = m.feature_names_in_
            else:
                varlist = m.booster_.feature_name()

            perf_res = get_perf_summary(train = self.train_data, 
                                        validation = self.validation_data, 
                                        oot = self.oot_data, 
                                        model = m, 
                                        tgt_name = self.y, 
                                        display = False,
                                        feature_cols=varlist, 
                                        to_show=False,
                                        fig_save_path = f"{self.results_output_dir}/{self.varreduct_perf_filename}{str(i)}.jpg",
                                        rpt_save_path = f"{self.results_output_dir}/{self.varreduct_perf_filename}{str(i)}.csv")

            perf_res["model_id"] = f"{self.varreduct_modelid_prefix}{str(i)}.pkl"
            perf_res["num_of_vars"] = len(varlist)

            ins_auc = pd.concat([perf_res[perf_res["index"] == "ins"], ins_auc])
            oos_auc = pd.concat([perf_res[perf_res["index"] == "oos"], oos_auc])    
            oot_auc = pd.concat([perf_res[perf_res["index"] == "oot"], oot_auc])

        ins_vr_res = ins_auc.reset_index(drop = True)
        oos_vr_res = oos_auc.reset_index(drop = True)
        oot_vr_res = oot_auc.reset_index(drop = True)

        combined_res = pd.concat([ins_vr_res, oos_vr_res, oot_vr_res])
        combined_res = combined_res.rename(columns={"index":"sample_ind"})
        combined_res = combined_res.pivot(columns = ['sample_ind'], index = ['model_id', 'num_of_vars'])
        combined_res = combined_res.loc[(combined_res[('AUC_Shift', 'oos')] <= overfitting_cut[0]) & (combined_res[('KS_Shift', 'oos')] <= overfitting_cut[1])]
        combined_res = combined_res.sort_values([('AUC','oot')], ascending = False)

        combined_res.columns = combined_res.columns.map('_'.join)
        return combined_res
    
    def get_backward_summary(self, grp_name = None, subgroup_value = None, min_data_size = 10):
        """
        生成后退消元的分组性能汇总。

        对整体和各个子组分别计算后退消元模型的性能汇总。

        Parameters
        ----------
        grp_name : str, optional
            分组字段名
        subgroup_value : list, optional
            子组值列表（若为None则自动获取唯一值）
        min_data_size : int, default 10
            每组最小样本数

        Returns
        -------
        pandas.DataFrame
            各子组后退消元性能汇总表

        Examples
        --------
        >>> summary = get_backward_summary(train_df, val_df, oot_df, 'target',
        ...                                 'models/', grp_name='region', algorithm='xgb')
        """
        
        ins_df = self.train_data.copy()
        oos_df = self.validation_data.copy()
        oot_df = self.oot_data.copy()
        dep = self.y
        dir_path = self.modelsave_dir
        algorithm = self.model_type

        if grp_name is not None and subgroup_value is None:
            subgroup_value = pd.concat([ins_df, oos_df, oot_df])[grp_name].unique()
            logger.info(subgroup_value)

        vr_res_dict = {}
        vr_res_dict['overall'] = self.get_backward_perf()

        for g in subgroup_value:
            ins_df_ = ins_df.query(f"{grp_name} == '{g}'")
            oos_df_ = oos_df.query(f"{grp_name} == '{g}'")
            oot_df_ = oot_df.query(f"{grp_name} == '{g}'")

            if (ins_df_.shape[0] > min_data_size) and (oos_df_.shape[0] > min_data_size) and (oot_df_.shape[0] > min_data_size):
                vr_res_dict[g] = self.get_backward_perf()

        vr_fnl_perf = []
        for name, res in vr_res_dict.items():

            res[grp_name] = name
            vr_fnl_perf.append(res)

        fnl_vr_res = pd.concat(vr_fnl_perf)

        return fnl_vr_res

    def add_group(self, grp_name, subgroup_value=None):
        """添加分组字段（用于分组分析）。

        Parameters
        ----------
        grp_name : str
            分组字段名
        subgroup_value : list, optional
            子组值列表

        Returns
        -------
        self
            返回自身以便链式调用

        Examples
        --------
        >>> eliminator = BackwardVariableEliminator(...)
        >>> eliminator.add_group('region').analyze()
        """
        self.grp_name = grp_name
        self.subgroup_value = subgroup_value
        return self

    def set_overfitting_cut(self, auc_shift=np.inf, ks_shift=np.inf):
        """设置过拟合阈值。

        Parameters
        ----------
        auc_shift : float, optional
            AUC偏移阈值，默认为inf
        ks_shift : float, optional
            KS偏移阈值，默认为inf

        Returns
        -------
        self
            返回自身以便链式调用

        Examples
        --------
        >>> eliminator = BackwardVariableEliminator(...)
        >>> eliminator.set_overfitting_cut(auc_shift=0.02, ks_shift=0.05).analyze()
        """
        self.overfitting_cut = [auc_shift, ks_shift]
        return self

    def analyze(self, result_dir=None):
        """执行后退消元结果分析。

        Parameters
        ----------
        result_dir : str, optional
            结果保存目录路径，默认为None（使用modelsave_dir）

        Returns
        -------
        pd.DataFrame
            各模型/子组性能汇总表

        Examples
        --------
        # 整体分析
        >>> eliminator = BackwardVariableEliminator(
        ...     train_data=df_train,
        ...     validation_data=df_val,
        ...     oot_data=df_oot,
        ...     y='target',
        ...     modelsave_dir='./models/'
        ... )
        >>> result = eliminator.analyze()

        # 分组分析
        >>> result = eliminator.add_group('region').analyze()

        # 链式调用（训练+分析）
        >>> eliminator = BackwardVariableEliminator(...)
        >>> result = eliminator.fit(x=['var1', 'var2'], ...).analyze()
        """
        self._validate_analyze_params()

        if result_dir is None:
            result_dir = self.modelsave_dir

        if self.grp_name is None:
            # 整体分析
            self.result_ = self.get_backward_perf(overfitting_cut=self.overfitting_cut)
        else:
            # 分组分析
            self.result_ = self.get_backward_summary(
                grp_name=self.grp_name,
                subgroup_value=self.subgroup_value,
                min_data_size=self.min_data_size
            )

        return self.result_

    def fit_and_analyze(self, x, results_output_dir=None, modelsave_dir=None,
                        varreduct_perf_filename='perf.csv',
                        varreduct_modelid_prefix="varreduct",
                        varreduct_max_nmodels=50, varreduct_minvar=30, cum_varimp=0.98,
                        scrutiny_range=None, scrutiny_step=None):
        """训练 + 分析一体化。

        执行向后变量消除训练后，自动进行性能分析。

        Parameters
        ----------
        x : list
            待筛选的特征变量名列表
        results_output_dir : str, optional
            结果输出目录路径
        modelsave_dir : str, optional
            模型保存目录路径
        varreduct_perf_filename : str, optional
            性能结果文件名
        varreduct_modelid_prefix : str, optional
            模型ID前缀
        varreduct_max_nmodels : int, optional
            最大迭代模型数量
        varreduct_minvar : int, optional
            变量筛选最小数量阈值
        cum_varimp : float, optional
            累计特征重要性阈值
        scrutiny_range : tuple/list, optional
            精确筛选的范围
        scrutiny_step : int, optional
            精确筛选每次减少的变量数

        Returns
        -------
        pd.DataFrame
            各模型性能汇总表

        Examples
        --------
        >>> eliminator = BackwardVariableEliminator(
        ...     model_type='lgb',
        ...     train_data=df_train,
        ...     validation_data=df_val,
        ...     oot_data=df_oot,
        ...     params={'n_estimators': 100},
        ...     y='target'
        ... )
        >>> result = eliminator.fit_and_analyze(
        ...     x=['var1', 'var2', 'var3'],
        ...     results_output_dir='./results/',
        ...     modelsave_dir='./models/'
        ... )
        """
        self.fit(
            x=x,
            results_output_dir=results_output_dir,
            modelsave_dir=modelsave_dir,
            varreduct_perf_filename=varreduct_perf_filename,
            varreduct_modelid_prefix=varreduct_modelid_prefix,
            varreduct_max_nmodels=varreduct_max_nmodels,
            varreduct_minvar=varreduct_minvar,
            cum_varimp=cum_varimp,
            scrutiny_range=scrutiny_range,
            scrutiny_step=scrutiny_step
        )
        return self.analyze(result_dir=results_output_dir)

    def switch_model(self, new_model_type):
        """切换模型类型。

        Parameters
        ----------
        new_model_type : str
            新的模型类型，可选值：'lgb'、'lightgbm'、'xgb'、'xgboost'

        Returns
        -------
        self
            返回自身实例

        Examples
        --------
        >>> eliminator = BackwardVariableEliminator('lgb', df_train, df_val, params, 'target')
        >>> eliminator.switch_model('xgb')
        """
        if new_model_type.lower() not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {new_model_type}. "
                f"Supported types: {self.SUPPORTED_MODELS}"
            )
        self.model_type = new_model_type.lower()
        return self

    def get_result(self):
        """获取分析结果。

        Returns
        -------
        pd.DataFrame or None
            分析结果DataFrame，如果尚未执行分析则返回None
        """
        return self.result_


# ============================================================================
# 保留旧类名作为别名（向后兼容）
# ============================================================================

class BackwardEliminationAnalyzer(BackwardVariableEliminator):
    """后退消元分析器（向后兼容别名）。

    此类是 BackwardVariableEliminator 的别名，
    保留旧名称以确保向后兼容性。

    .. deprecated::
        请使用 BackwardVariableEliminator 替代此类

    Examples
    --------
    >>> analyzer = BackwardEliminationAnalyzer(
    ...     ins_df=df_train,
    ...     oos_df=df_val,
    ...     oot_df=df_oot,
    ...     dep='target',
    ...     dir_path='./models/'
    ... )
    >>> result = analyzer.analyze()
    """

    def __init__(self, ins_df=None, oos_df=None, oot_df=None, dep=None,
                 dir_path=None, algorithm='xgb', min_data_size=10):
        """初始化后退消元分析器。

        Parameters
        ----------
        ins_df : pd.DataFrame
            训练数据集
        oos_df : pd.DataFrame
            验证数据集
        oot_df : pd.DataFrame
            OOT测试数据集
        dep : str
            目标变量列名
        dir_path : str
            模型文件目录路径
        algorithm : str, optional
            算法类型，默认为'xgb'
        min_data_size : int, optional
            每组最小样本数，默认为10
        """
        super().__init__(
            train_data=ins_df,
            validation_data=oos_df,
            oot_data=oot_df,
            y=dep,
            modelsave_dir=dir_path,
            algorithm=algorithm
        )
        self.min_data_size = min_data_size

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# ============================================================================
# # 分析相关函数（保持独立）
# # ============================================================================

# def get_backward_perf(ins_df, oos_df, oot_df, dep, vr_model_dir, result_dir=None,
#                        overfitting_cut=[np.inf, np.inf], algorithm='xgb'):
#     """获取后退消元性能报告（整体分析）。

#     对后退消元过程中保存的所有模型进行性能评估，
#     输出整体性能对比报告。

#     Parameters
#     ----------
#     ins_df : pd.DataFrame
#         训练数据集
#     oos_df : pd.DataFrame
#         验证数据集
#     oot_df : pd.DataFrame
#         OOT测试数据集
#     dep : str
#         目标变量列名
#     vr_model_dir : str
#         向后消元模型保存目录
#     result_dir : str, optional
#         结果输出目录，默认为None（使用vr_model_dir）
#     overfitting_cut : list, optional
#         过拟合阈值，[auc_shift, ks_shift]，默认为[inf, inf]
#     algorithm : str, optional
#         算法类型，'xgb'或'lgb'，默认为'xgb'

#     Returns
#     -------
#     pd.DataFrame
#         各模型性能汇总表

#     Examples
#     --------
#     >>> perf = get_backward_perf(
#     ...     ins_df, oos_df, oot_df,
#     ...     dep='target',
#     ...     vr_model_dir='./models/'
#     ... )
#     """

#     return get_backward_perf(
#         ins_df=ins_df,
#         oos_df=oos_df,
#         oot_df=oot_df,
#         dep=dep,
#         vr_model_dir=vr_model_dir,
#         result_dir=result_dir,
#         overfitting_cut=overfitting_cut,
#         algorithm=algorithm
#     )


# def get_backward_summary(ins_df, oos_df, oot_df, dep, dir_path, grp_name,
#                           subgroup_value=None, algorithm='xgb', min_data_size=10):
#     """获取后退消元性能报告（分组分析）。

#     对后退消元过程中保存的所有模型进行分组性能评估，
#     输出各子组的性能对比报告。

#     Parameters
#     ----------
#     ins_df : pd.DataFrame
#         训练数据集
#     oos_df : pd.DataFrame
#         验证数据集
#     oot_df : pd.DataFrame
#         OOT测试数据集
#     dep : str
#         目标变量列名
#     dir_path : str
#         模型和结果保存目录
#     grp_name : str
#         分组字段名
#     subgroup_value : list, optional
#         子组值列表，默认为None（使用所有子组）
#     algorithm : str, optional
#         算法类型，'xgb'或'lgb'，默认为'xgb'
#     min_data_size : int, optional
#         每组最小样本数，默认为10

#     Returns
#     -------
#     pd.DataFrame
#         各子组性能汇总表

#     Examples
#     --------
#     >>> summary = get_backward_summary(
#     ...     ins_df, oos_df, oot_df,
#     ...     dep='target',
#     ...     dir_path='./results/',
#     ...     grp_name='region'
#     ... )
#     """
#     from Modeling_Tool.Eval.Model_Eval_Tool import get_backward_summary as _get_backward_summary
#     return _get_backward_summary(
#         ins_df=ins_df,
#         oos_df=oos_df,
#         oot_df=oot_df,
#         dep=dep,
#         dir_path=dir_path,
#         grp_name=grp_name,
#         subgroup_value=subgroup_value,
#         algorithm=algorithm,
#         min_data_size=min_data_size
#     )
