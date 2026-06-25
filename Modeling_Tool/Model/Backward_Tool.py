"""
向后变量消除工具包（统一版）
=============================

本模块提供基于LightGBM和XGBoost的向后变量消除（Backward Variable Elimination）功能，
通过累计特征重要性阈值进行变量筛选，并支持训练后的性能分析。

Functions
---------
backward_lgbm
    使用LightGBM模型进行向后变量消除
backward_xgbm
    使用XGBoost模型进行向后变量消除

Classes
-------
BackwardVariableEliminator
    向后变量消除器，支持LightGBM和XGBoost
BackwardEliminationAnalyzer
    向后消除结果分析器
"""

import os
import sys
import copy
import logging
import warnings
import functools
from collections import OrderedDict
from typing import Optional, List, Dict, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


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
) -> Tuple:
    """
    使用LightGBM模型进行向后变量消除。

    通过训练LightGBM模型并根据特征重要性累计阈值筛选变量，
    实现向后变量消除（Backward Variable Elimination）。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集，必须包含dep列和varlist中的所有特征列
    varlist : list of str
        参与建模的特征变量列表
    dep : str
        目标变量列名（0/1二元变量）
    varreduct_params : dict, optional
        LightGBM超参数字典，未指定的必需参数将使用预设值
    stopping_metric : str, default "auc"
        早停评估指标，可选 "auc", "binary_logloss" 等
    seed : int, default 42
        随机种子，保证可复现性
    num_boost_round : int, default 200
        最大迭代轮数
    early_stopping_rounds : int, default 20
        早停轮数，验证集指标连续N轮无提升则停止
    importance_type : str, default "gain"
        特征重要性类型，可选 "gain", "split"
    cum_importance_threshold : float, default 0.99
        累计特征重要性阈值，筛选覆盖该比例重要性的最少特征
    min_vars : int, default 10
        保留的最小变量数量
    validation_data : pd.DataFrame, optional
        验证数据集，用于早停
    test_data_dict : dict, optional
        测试数据集字典，格式为 {名称: DataFrame}
    ret_perf : bool, default True
        是否返回模型性能指标
    nbins : int, default 10
        增益表分箱数
    precision : int, default 5
        数值精度
    min_bin_prop : float, default 0.05
        最小分箱比例
    include_missing : bool, default True
        是否包含缺失值分箱
    equal_freq : bool, default True
        是否使用等频分箱
    ascending : bool, default True
        增益表是否升序排列
    fillna : float, optional
        缺失值填充值
    spec_values : list, optional
        特殊值列表

    Returns
    -------
    tuple
        (selected_vars, model, perf_dict) 或 (selected_vars, model)

    Raises
    ------
    TypeError
        当输入数据不是pandas.DataFrame格式时
    ImportError
        当lightgbm未安装时

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
        raise ImportError("请安装lightgbm: pip install lightgbm")

    from Modeling_Tool.Eval.Model_Eval_Tool import get_perf_summary

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

    # 检查数据格式一致性
    try:
        for k, v in datain_all.items():
            assert isinstance(v, pd.DataFrame)
    except AssertionError:
        logging.warning("请提供pandas.DataFrame格式数据")
        raise TypeError("请提供pandas.DataFrame格式数据")

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
        varreduct_params[param] = hyperparams_preset[param]

    # 构建 LightGBM 数据集
    lgb_train = lgb.Dataset(train_data[varlist], label=train_data[dep])

    if validation_data is not None:
        lgb_valid = lgb.Dataset(validation_data[varlist], label=validation_data[dep])
        valid_sets = [lgb_valid]
        valid_names = ["hd"]
    else:
        valid_sets = [lgb_train]
        valid_names = ["mdl"]

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1)
    ]

    # 训练模型
    model = lgb.train(
        params=varreduct_params,
        train_set=lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks
    )

    # 获取特征重要性并筛选变量
    importance_df = pd.DataFrame({
        "feature": model.feature_name(),
        "importance": model.feature_importance(importance_type=importance_type)
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df["cum_importance"] = importance_df["importance"].cumsum() / importance_df["importance"].sum()

    # 筛选达到阈值的变量
    selected_idx = importance_df[importance_df["cum_importance"] <= cum_importance_threshold].index.tolist()

    # 确保至少保留 min_vars 个变量
    if len(selected_idx) < min_vars:
        selected_idx = list(range(min(min_vars, len(importance_df))))

    selected_vars = importance_df.loc[selected_idx, "feature"].tolist()

    if not ret_perf:
        return selected_vars, model

    # 计算性能
    score_col = "_lgbm_score"
    perf_dict = {}

    for name, df in datain_all.items():
        df_score = df.copy()
        df_score[score_col] = model.predict(df_score[varlist])
        perf = get_perf_summary(
            train=df_score if name == "mdl" else None,
            validation=df_score if name == "hd" else None,
            oot=df_score if name not in ("mdl", "hd") else None,
            tgt_name=dep,
            scr_name=score_col,
            nbins=nbins,
            precision=precision,
            min_bin_prop=min_bin_prop,
            include_missing=include_missing,
            equal_freq=equal_freq,
            ascending=ascending,
            fillna=fillna,
            spec_values=spec_values
        )
        perf_dict[name] = perf

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
) -> Tuple:
    """
    使用XGBoost模型进行向后变量消除。

    通过训练XGBoost模型并根据特征重要性累计阈值筛选变量，
    实现向后变量消除（Backward Variable Elimination）。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集，必须包含dep列和varlist中的所有特征列
    varlist : list of str
        参与建模的特征变量列表
    dep : str
        目标变量列名（0/1二元变量）
    varreduct_params : dict, optional
        XGBoost超参数字典，未指定的必需参数将使用预设值
    stopping_metric : str, default "auc"
        早停评估指标
    seed : int, default 42
        随机种子
    num_boost_round : int, default 200
        最大迭代轮数
    early_stopping_rounds : int, default 20
        早停轮数
    importance_type : str, default "gain"
        特征重要性类型
    cum_importance_threshold : float, default 0.99
        累计特征重要性阈值
    min_vars : int, default 10
        保留的最小变量数量
    validation_data : pd.DataFrame, optional
        验证数据集
    test_data_dict : dict, optional
        测试数据集字典
    ret_perf : bool, default True
        是否返回性能指标
    nbins : int, default 10
        增益表分箱数
    precision : int, default 5
        数值精度
    min_bin_prop : float, default 0.05
        最小分箱比例
    include_missing : bool, default True
        是否包含缺失值分箱
    equal_freq : bool, default True
        是否使用等频分箱
    ascending : bool, default True
        增益表是否升序排列
    fillna : float, optional
        缺失值填充值
    spec_values : list, optional
        特殊值列表
    monotone_constraints : dict, optional
        单调约束字典，格式为 {特征名: 1/-1}

    Returns
    -------
    tuple
        (selected_vars, model, perf_dict) 或 (selected_vars, model)

    Raises
    ------
    TypeError
        当输入数据不是pandas.DataFrame格式时
    ImportError
        当xgboost未安装时

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
        raise ImportError("请安装xgboost: pip install xgboost")

    from Modeling_Tool.Eval.Model_Eval_Tool import get_perf_summary

    if varreduct_params is None:
        varreduct_params = {}

    if test_data_dict is None:
        test_data_dict = {}

    if spec_values is None:
        spec_values = []

    if monotone_constraints is None:
        monotone_constraints = {}

    # 构建单调约束向量
    mc_dict = {var: monotone_constraints.get(var, 0) for var in varlist}

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
        raise TypeError("请提供pandas.DataFrame格式数据")

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
        varreduct_params[param] = hyperparams_preset[param]

    # 构建 XGBoost 数据集
    xgb_train = xgb.DMatrix(train_data[varlist], label=train_data[dep])

    evals = [(xgb_train, "mdl")]
    if validation_data is not None:
        xgb_valid = xgb.DMatrix(validation_data[varlist], label=validation_data[dep])
        evals.append((xgb_valid, "hd"))

    # 训练模型
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

    # 获取特征重要性并筛选变量
    importance_raw = model.get_score(importance_type=importance_type)
    importance_df = pd.DataFrame(
        list(importance_raw.items()), columns=["feature", "importance"]
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    # 补充重要性为0的特征
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

    # 计算性能
    score_col = "_xgbm_score"
    perf_dict = {}

    for name, df in datain_all.items():
        df_score = df.copy()
        xgb_dmat = xgb.DMatrix(df_score[varlist])
        df_score[score_col] = model.predict(xgb_dmat)
        perf = get_perf_summary(
            train=df_score if name == "mdl" else None,
            validation=df_score if name == "hd" else None,
            oot=df_score if name not in ("mdl", "hd") else None,
            tgt_name=dep,
            scr_name=score_col,
            nbins=nbins,
            precision=precision,
            min_bin_prop=min_bin_prop,
            include_missing=include_missing,
            equal_freq=equal_freq,
            ascending=ascending,
            fillna=fillna,
            spec_values=spec_values
        )
        perf_dict[name] = perf

    return selected_vars, model, perf_dict


class BackwardVariableEliminator:
    """
    向后变量消除器。

    封装LightGBM/XGBoost向后变量消除流程，
    支持多轮消除和结果汇总。

    Parameters
    ----------
    train_data : pd.DataFrame
        训练数据集
    varlist : list of str
        初始特征变量列表
    dep : str
        目标变量列名
    model_type : str, default "lgbm"
        模型类型，可选 "lgbm" 或 "xgbm"
    validation_data : pd.DataFrame, optional
        验证数据集
    test_data_dict : dict, optional
        测试数据集字典

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
    ):
        self.train_data = train_data
        self.varlist = varlist
        self.dep = dep
        self.model_type = model_type.lower()
        self.validation_data = validation_data
        self.test_data_dict = test_data_dict or {}
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
        运行多轮向后变量消除。

        Parameters
        ----------
        n_rounds : int, default 5
            消除轮数
        varreduct_params : dict, optional
            模型超参数
        stopping_metric : str, default "auc"
            早停指标
        seed : int, default 42
            随机种子
        num_boost_round : int, default 200
            最大迭代轮数
        early_stopping_rounds : int, default 20
            早停轮数
        importance_type : str, default "gain"
            特征重要性类型
        cum_importance_threshold : float, default 0.99
            累计重要性阈值
        min_vars : int, default 10
            最小保留变量数
        ret_perf : bool, default True
            是否返回性能指标
        nbins : int, default 10
            分箱数

        Returns
        -------
        list of dict
            每轮消除结果列表
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
        """获取最终筛选后的变量列表。"""
        if not self._results:
            return self.varlist
        return self._results[-1]["selected_vars"]

    def get_summary(self) -> pd.DataFrame:
        """获取每轮消除汇总表。"""
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
    向后消除结果分析器。

    对 BackwardVariableEliminator 的运行结果进行分析和可视化。

    Parameters
    ----------
    results : list of dict
        BackwardVariableEliminator.run() 的返回值

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
        获取在所有轮次中均被保留的稳定变量。

        Parameters
        ----------
        top_n : int, optional
            返回前N个稳定变量，None表示返回全部

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
        绘制变量数量随消除轮次变化的折线图。

        Parameters
        ----------
        figsize : tuple, default (8, 4)
            图形尺寸
        save_path : str, optional
            图片保存路径，None表示直接显示
        """
        rounds = [r["round"] for r in self.results]
        n_vars = [r["n_vars_out"] for r in self.results]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(rounds, n_vars, marker="o", linewidth=2, color="#4C72B0")
        ax.set_xlabel("消除轮次")
        ax.set_ylabel("保留变量数")
        ax.set_title("向后变量消除：变量数量变化")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def get_perf_trend(self, dataset: str = "mdl", metric: str = "IV") -> pd.DataFrame:
        """
        获取指定数据集上性能指标随轮次变化趋势。

        Parameters
        ----------
        dataset : str, default "mdl"
            数据集名称，如 "mdl", "hd", "oot"
        metric : str, default "IV"
            性能指标列名

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for r in self.results:
            perf = r.get("perf", {})
            if dataset in perf and perf[dataset] is not None:
                val = perf[dataset].get(metric, None) if isinstance(perf[dataset], dict) else None
                rows.append({"round": r["round"], metric: val})
        return pd.DataFrame(rows)
