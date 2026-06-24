"""
Model Evaluation Tool - Optimized Version

This module provides classes and functions for model evaluation, performance comparison,
and gains analysis. It includes utilities for calculating various metrics and generating
cross-risk summaries.

Author: Matrix Agent
"""

import pandas as pd
import numpy as np
import inspect
from typing import List, Dict, Optional, Callable, Any, Union
from .Model_Eval_Tool import get_perf_summary, get_gains_table, get_gains_table_by_cust_metrics, cross_risk
import logging
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """
    链式调用的流水线对象，支持 .group_by() 和 .subset_by() 链式添加条件，
    最后通过 .apply(func) 执行。
    支持 func 返回 pandas.DataFrame 或 dict（值为 DataFrame）。
    """
    def __init__(self, m_eval, data=None):
        self.m_eval = m_eval
        self.original_data = m_eval.data if data is None else data
        self.steps = []
        self._expected_columns = None   # 用于 DataFrame 返回时的列名缓存
        self._expected_dict_keys = None # 用于 dict 返回时的键缓存

    def group_by(self, group_name, min_size=100, group_var_name=None):
        if group_var_name is None:
            group_var_name = group_name
        self.steps.append({
            'type': 'group',
            'group_name': group_name,
            'group_var_name': group_var_name,
            'min_size': min_size
        })
        return self

    def subset_by(self, condition_dict, name='eval_subset', min_size=100):
        self.steps.append({
            'type': 'subset',
            'condition_dict': condition_dict,
            'subset_var_name': name,
            'min_size': min_size
        })
        return self

    def apply(self, func, **kwargs):
        
        import inspect
        sig = inspect.signature(func)
        
        def _execute(step_idx, current_data, current_meta):
            if step_idx >= len(self.steps):
                old_data = self.m_eval.data
                self.m_eval.data = current_data
                try:
                    # 自动注入当前数据（如果函数接受 'current_data' 参数）
                    if 'current_data' in sig.parameters:
                        kwargs['current_data'] = current_data
                    res = func(**kwargs)
                except Exception as e:
                    logger.info(f"Error in group {current_meta}: {e}")
                    if self._expected_dict_keys is not None:
                        res = {k: pd.DataFrame() for k in self._expected_dict_keys}
                    else:
                        res = pd.DataFrame()
                finally:
                    self.m_eval.data = old_data

                # （处理 dict 和 DataFrame 的逻辑）
                if isinstance(res, dict):
                    if self._expected_dict_keys is None and res:
                        self._expected_dict_keys = list(res.keys())
                    for key, df in res.items():
                        if df is not None and not df.empty:
                            for col_name, col_val in current_meta.items():
                                df[col_name] = col_val
                        else:
                            empty_df = pd.DataFrame()
                            for col_name, col_val in current_meta.items():
                                empty_df[col_name] = [col_val]
                            res[key] = empty_df
                    return res
                else:
                    if self._expected_columns is None and res is not None and not res.empty:
                        self._expected_columns = res.columns.tolist()
                    if res is not None and not res.empty:
                        res = res.copy()
                        for key, val in current_meta.items():
                            res[key] = val
                    else:
                        if self._expected_columns is not None:
                            res = pd.DataFrame(columns=self._expected_columns)
                        else:
                            res = pd.DataFrame()
                    return res

            step = self.steps[step_idx]
            if step['type'] == 'group':
                group_name = step['group_name']
                group_var_name = step['group_var_name']
                min_size = step['min_size']
                results = []
                for gval in current_data[group_name].unique():
                    sub_data = current_data[current_data[group_name] == gval]
                    if len(sub_data) >= min_size:
                        new_meta = current_meta.copy()
                        new_meta[group_var_name] = gval
                        sub_res = _execute(step_idx + 1, sub_data, new_meta)
                        if sub_res is not None:
                            results.append(sub_res)
                if not results:
                    if self._expected_dict_keys is not None:
                        return {k: pd.DataFrame() for k in self._expected_dict_keys}
                    else:
                        return pd.DataFrame()
                # 合并结果
                if isinstance(results[0], dict):
                    merged = {k: pd.concat([r[k] for r in results], ignore_index=True) for k in results[0].keys()}
                    return merged
                else:
                    return pd.concat(results, ignore_index=True)

            elif step['type'] == 'subset':
                condition_dict = step['condition_dict']
                subset_var_name = step['subset_var_name']
                min_size = step['min_size']
                results = []
                for label, query in condition_dict.items():
                    if query == "":
                        sub_data = current_data.copy()
                    else:
                        sub_data = current_data.query(query)
                    if len(sub_data) >= min_size:
                        new_meta = current_meta.copy()
                        new_meta[subset_var_name] = label
                        sub_res = _execute(step_idx + 1, sub_data, new_meta)
                        if sub_res is not None:
                            results.append(sub_res)
                if not results:
                    if self._expected_dict_keys is not None:
                        return {k: pd.DataFrame() for k in self._expected_dict_keys}
                    else:
                        return pd.DataFrame()
                if isinstance(results[0], dict):
                    merged = {k: pd.concat([r[k] for r in results], ignore_index=True) for k in results[0].keys()}
                    return merged
                else:
                    return pd.concat(results, ignore_index=True)

        return _execute(0, self.original_data, {})


class Utility_Functions:
    """Utility functions for data processing and calculations."""
    
    @staticmethod
    def valid_average(x):
        """
        Calculate the average of positive values in a Series.
        
        This function filters out non-positive values (including zeros and negatives)
        before calculating the mean, useful for financial or count data where
        zero or negative values should not contribute to the average.
        
        Parameters
        ----------
        x : pandas.Series
            Input series of numeric values.
            
        Returns
        -------
        float
            Rounded average of positive values, or 0 if no valid values exist.
            
        Examples
        --------
        >>> import pandas as pd
        >>> Utility_Functions.valid_average(pd.Series([1, 2, -1, 0, 3]))
        2.0
        """
        valid_mask = x > 0
        valid_count = valid_mask.sum()
        valid_values = x[valid_mask]
        value_sum = valid_values.sum()
        
        if valid_count > 0:
            return round(value_sum / valid_count, 2)
        return 0


class Model_Evaluation_Tool:
    """
    A comprehensive tool for model evaluation, performance comparison, and gains analysis.
    
    This class provides methods for:
    - Base score calculation and correlation analysis
    - Model performance comparison across multiple metrics
    - Gains table generation and analysis
    - Custom metric evaluation
    - Cross-risk analysis
    - Multi-dimensional evaluation with subsets and groupings
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing features and target variables.
    dep : str
        Name of the dependent/target variable.
    comp_scrlist : list
        List of comparison score column names.
    model : object, optional
        Trained model with predict_proba method. Default is None.
    base_score : str, optional
        Name of the base score column. Default is None.
    min_data_size : int, optional
        Minimum data size threshold. Default is 500.
    equal_freq : bool, optional
        Whether to use equal frequency binning. Default is True.
    precision : int, optional
        Decimal precision for rounding. Default is 5.
    chi2_method : bool, optional
        Whether to use chi-square method. Default is False.
    chi2_p : float, optional
        Chi-square p-value threshold. Default is 0.999.
    init_equi_bins : int, optional
        Initial number of equal frequency bins. Default is 500.
    tree_binning : bool, optional
        Whether to use tree-based binning. Default is False.
    random_seed : int, optional
        Random seed for reproducibility. Default is 3407.
    nbins : int, optional
        Number of bins for analysis. Default is 10.
    min_bin_prop : float, optional
        Minimum proportion for each bin. Default is 0.05.
    include_missing : bool, optional
        Whether to include missing values. Default is True.
    missing_rate_ref : int or float, optional
        Reference value for missing rate. Default is -99999999.
    excel_path : str, optional
        Path for Excel output. Default is None.
        
    Attributes
    ----------
    data : pandas.DataFrame
        The input data (may be modified during processing).
    dep : str
        Target variable name.
    base_score : str or None
        Name of the base score column.
    comp_scrlist : list
        List of comparison score names.
    eval_metrics : list
        List of evaluation metrics.
    eval_ylabels : list
        Labels for y-axis evaluation.
    gains_display_metric_list : list
        Metrics to display in gains summary.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        dep: str,
        comp_scrlist: List[str],
        model: Any = None,
        base_score: Optional[str] = None,
        min_data_size: int = 500,
        equal_freq: bool = True,
        precision: int = 5,
        chi2_method: bool = False,
        chi2_p: float = 0.999,
        init_equi_bins: int = 500,
        tree_binning: bool = False,
        random_seed: int = 3407,
        nbins: int = 10,
        min_bin_prop: float = 0.05,
        include_missing: bool = True,
        missing_rate_ref: Union[int, float] = -99999999,
        excel_path: Optional[str] = None,
        fillna = -999999,
        spec_values = []
    ):
        """
        Initialize the Model_Evaluation_Tool with configuration parameters.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input dataset for evaluation.
        dep : str
            Name of the target/dependent variable.
        comp_scrlist : list
            List of comparison score column names.
        model : object, optional
            Trained model with predict_proba method.
        base_score : str, optional
            Base score column name.
        min_data_size : int, optional
            Minimum data size threshold.
        equal_freq : bool, optional
            Use equal frequency binning if True.
        precision : int, optional
            Decimal precision for calculations.
        chi2_method : bool, optional
            Use chi-square method if True.
        chi2_p : float, optional
            Chi-square p-value threshold.
        init_equi_bins : int, optional
            Initial equal bins count.
        tree_binning : bool, optional
            Use tree-based binning if True.
        random_seed : int, optional
            Random seed value.
        nbins : int, optional
            Number of bins.
        min_bin_prop : float, optional
            Minimum bin proportion.
        include_missing : bool, optional
            Include missing values if True.
        missing_rate_ref : int or float, optional
            Missing rate reference value.
        excel_path : str, optional
            Path for Excel output file.
        """
        self.data = data
        self.dep = dep
        self.model = model
        self.comp_scrlist = comp_scrlist
        self.base_score = base_score
        self.min_data_size = min_data_size
        self.equal_freq = equal_freq
        self.precision = precision
        self.nbins = nbins
        self.min_bin_prop = min_bin_prop
        self.include_missing = include_missing
        self.tree_binning = tree_binning
        self.seed = random_seed
        self.excel_path = excel_path
        self.missing_rate_ref = missing_rate_ref
        self.fillna = fillna
        self.spec_values = spec_values
        
        self.chi2_method = chi2_method
        self.chi2_p = chi2_p
        self.init_equi_bins = init_equi_bins
        
        self.eval_metrics = [
            'age', 
            'monthlyincome', 
            'education', 
            'credit_limit'
        ]
        
        self.udf = Utility_Functions()
        self.cross_agg_dict = {
            'is_dpd7':                          ['count', lambda x: (x.sum() / x.count()).round(4)],
            'credit_limit':                     ['count', lambda x: self.udf.valid_average(x)],
            'monthlyincome':                    ['count', lambda x: self.udf.valid_average(x)],
            'education':                        ['count', lambda x: self.udf.valid_average(x)],
        }
        
        self.subset_condition_dict = {
            "Overall": "", 
            "AE": "aprvvrsn_2 in ('AE')",
            "NON_AE": "aprvvrsn_2 not in ('AE')",
            "FT": "aprvvrsn_2 in ('FT2.0')",
            "NON_FT": "aprvvrsn_2 not in ('FT2.0')",
            "NON_FT_AE": "aprvvrsn_2 not in ('FT2.0') and aprvvrsn_2 not in ('AE')"
        }
        
        self.eval_ylabels = ['is_dpd7']
        self.grp_namelist = ['sample_ind_fnl', 'aprvvrsn_2', 'week_start_date']
        self.gains_display_metric_list = ['MIN', 'MAX', 'N', 'PROP', 'AVG_SCORE', 'AVG_BAD', 'CUM_BAD_PCT', 'KS_PER_BIN', 'LIFT', 'RANK_ORDER_BUMP']
    
    def __init_excel_master(self):
        """
        Initialize Excel Master for report generation.
        
        Returns
        -------
        ExcelMaster
            Initialized ExcelMaster instance.
            
        Notes
        -----
        Requires ExcelMaster package to be installed.
        """
        from ExcelMaster.ExcelMaster import ExcelMaster
        
        em = ExcelMaster(filepath=self.excel_path, verbose=False)
        return em
    
    def get_base_score(self, scorename: str = '_base_model_score_', disp: bool = False):
        """
        Calculate base model scores using the provided model.
        
        This method uses the model's predict_proba method to generate scores
        for the input data and optionally displays correlation analysis.
        
        Parameters
        ----------
        scorename : str, optional
            Name for the score column. Default is '_base_model_score_'.
        disp : bool, optional
            Whether to display intermediate results. Default is False.
            
        Returns
        -------
        Model_Evaluation_Tool
            Returns self for method chaining.
            
        Raises
        ------
        AttributeError
            If model is None or lacks required attributes.
        """
        if self.model is None:
            raise AttributeError("Model must be provided to calculate base score.")
        
        data = self.data.copy()
        model = self.model
        
        inputs = model.feature_names_in_.tolist()
        data[scorename] = model.predict_proba(data[inputs])[:, 1]
        self.data = data
        self.base_score = scorename
        
        scrlist = [scorename] + self.comp_scrlist
        if disp:
            from IPython.display import display
            from Modeling_Tool.Feature.Distribution_Tool import proc_means_by_grp
            display(proc_means_by_grp(data, scrlist))
        
        tmp_data = data[data[scrlist] > 0]
        
        if disp:
            from IPython.display import display
            display(tmp_data[scrlist].corr())
        
        return self
    
    def get_score_correlation(
        self,
        score_list: Optional[List[str]] = None,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between score columns.
        
        Parameters
        ----------
        score_list : list, optional
            List of score column names. If None, uses base_score and comp_scrlist.
        method : str, optional
            Correlation method ('pearson', 'spearman', 'kendall'). Default is 'pearson'.
            
        Returns
        -------
        pandas.DataFrame
            Correlation summary in long format with columns: 'base', 'compare', 'corr'.
        """
        if score_list is None:
            score_list = [self.base_score] + self.comp_scrlist
        
        if self.base_score is None:
            raise ValueError("Base score must be set before calculating correlation.")
        
        data = self.data.copy()
        
        score_query = " and ".join([s + " > 0" for s in score_list])
        data = data.query(score_query)
        
        corr_summary = data[score_list]\
            .corr(method=method)\
            .reset_index(drop=False)\
            .melt(id_vars=['index'], var_name='compare', value_name='corr')\
            .rename(columns={"index": "base"})
        
        return corr_summary
    
    def model_perf_compare(
        self,
        data: Optional[pd.DataFrame] = None,
        grp_name: Optional[str] = None,
        dist_bins: int = 100,
        pct_bins: int = 10,
        min_data_size: int = 50,
        sync_data_size: bool = True
    ) -> pd.DataFrame:
        """
        Compare model performance across different scores.
        
        This method calculates performance metrics (AUC, KS, etc.) for the base score
        and all comparison scores, optionally filtering by group.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to use. If None, uses self.data.
        grp_name : str, optional
            Group name column for stratification.
        dist_bins : int, optional
            Number of distribution bins. Default is 100.
        pct_bins : int, optional
            Number of percentile bins. Default is 10.
        min_data_size : int, optional
            Minimum data size per bin. Default is 50.
        sync_data_size : bool, optional
            Whether to filter out zero/negative scores. Default is True.
            
        Returns
        -------
        pandas.DataFrame
            Performance comparison results sorted by score order.
        """
        
        data = self.data.copy() if data is None else data.copy()
        score_list = self.comp_scrlist
        dep = self.dep
        base_score = self.base_score
        precision = self.precision
        
        # 过滤掉无效分数和目标值
        valid_mask = (
            data[dep].notna() & 
            data[base_score].notna() &
            np.isfinite(data[base_score])
        )
        clean_data = data[valid_mask]
        
        if len(clean_data) == 0:
            return pd.DataFrame()  # 返回空结果
        # 后续使用 clean_data 进行计算
        
        oot_df = clean_data.copy()
        sample_name = 'oot'
        oot_df_ = clean_data.copy()
        
        if sync_data_size:
            if len(score_list) > 0:
                score_query = " and ".join([s + " > 0" for s in score_list])
                oot_df_ = oot_df.query(score_query) if oot_df is not None else None
        
        perf_comp_dict = {}
        
        if base_score is not None:
            base_perf = get_perf_summary(
                train=None, 
                validation=None, 
                oot=oot_df_.query(f"{base_score} > 0"), 
                tgt_name=dep,
                scr_name=base_score,
                to_show=False,
                display=False,
                dist_bins=dist_bins,
                pct_bins=pct_bins,
                precision=precision,
                min_bin_prop=0.05,
                include_missing=False,
                equal_freq=True,
                oot_grp_name=grp_name,
                min_data_size=min_data_size
            )
            
            if not base_perf.empty and 'index' in base_perf.columns:
                base_perf = base_perf.query(f"index == '{sample_name}'")
            
            perf_comp_dict[base_score] = base_perf
        
        if len(score_list) > 0:
            for score in score_list:
                """Loop Score List."""
                perf_comp_dict[score] = get_perf_summary(
                    train=None, 
                    validation=None, 
                    oot=oot_df_.query(f"{score} > 0"), 
                    scr_name=score,
                    tgt_name=dep,
                    to_show=False,
                    display=False,
                    dist_bins=dist_bins,
                    pct_bins=pct_bins,
                    precision=precision,
                    min_bin_prop=0.05,
                    include_missing=False,
                    equal_freq=True,
                    oot_grp_name=grp_name,
                    min_data_size=min_data_size
                ).query(f"index == '{sample_name}'")
        
        if not perf_comp_dict:
            return pd.DataFrame()
        
        perf_comp_res = []
        for score_name, perf in perf_comp_dict.items():
            perf = perf.copy()
            perf['score_name'] = score_name
            perf_comp_res.append(perf)
        
        perf_comp_res = pd.concat(perf_comp_res)
        
        drop_cols = ['AUC_Shift', 'KS_Shift']
        for col in drop_cols:
            if col in perf_comp_res.columns:
                perf_comp_res = perf_comp_res.drop(columns=[col])
        
        custom_order = [base_score] + score_list if base_score else score_list
        order_map = {val: i for i, val in enumerate(custom_order) if val in perf_comp_dict}
        perf_comp_res['sort_key'] = perf_comp_res['score_name'].map(order_map)
        perf_comp_res_sorted = perf_comp_res.sort_values('sort_key').drop('sort_key', axis=1)
        
        return perf_comp_res_sorted
    
    def get_gains_summary(
        self,
        data: Optional[pd.DataFrame] = None,
        grp_name: Optional[str] = None,
        disp: bool = True,
        grp_disp_metric: List[str] = None,
        grp_nbins: int = 5,
        withSummary: bool = True,
        add_func: Optional[Callable] = None,
        sync_range: bool = True,
        spec_values = [],
        include_missing = False,
        fillna = None
    ) -> pd.DataFrame:
        """
        Generate gains summary table for score analysis.
        
        This method creates gains tables showing the distribution of targets
        across score bins, with optional group stratification.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to use. If None, uses self.data.
        grp_name : str, optional
            Group name for stratified analysis.
        disp : bool, optional
            Whether to display results. Default is True.
        grp_disp_metric : list, optional
            Metrics to display for grouped analysis.
        grp_nbins : int, optional
            Number of bins for grouped analysis. Default is 5.
        withSummary : bool, optional
            Include summary row. Default is True.
        add_func : callable, optional
            Additional function to apply to gains table.
        sync_range : bool, optional
            Synchronize bin ranges. Default is True.
            
        Returns
        -------
        pandas.DataFrame
            Gains summary results with score names.
        """
        if grp_disp_metric is None:
            grp_disp_metric = ['N', 'PROP', 'AVG_BAD', 'LIFT']
            
        if fillna is None:
            fillna = self.fillna
        
        new_oot_df = self.data.copy() if data is None else data.copy()
        
        dep = self.dep
        tree_binning = self.tree_binning
        digit = self.precision
        base_score = self.base_score
        equal_freq = self.equal_freq
        min_data_size = self.min_data_size
        nbins = self.nbins if grp_name is None else grp_nbins
        min_bin_prop = self.min_bin_prop
        
        score_list = [base_score] + self.comp_scrlist
        display_metric_list = self.gains_display_metric_list if add_func is None else None
        
        if self.data is None or len(self.data) == 0:
            empty_df = pd.DataFrame()
            empty_df.index = pd.MultiIndex.from_tuples([], names=['_bin_num', '_bin_range'])
            return empty_df  # 或返回空结果
        
        gains_table_dict = {}
        for score in score_list:
            if grp_name is None:
                ### Overall Gains
                gains_table_dict[score] = get_gains_table(
#                     data=new_oot_df.query(f"{score} > 0"), 
                    data=new_oot_df, 
                    dep=dep, 
                    ascending=True,
                    tree_binning=tree_binning,
                    nbins=nbins, 
                    precision=digit, 
                    min_bin_prop=min_bin_prop, 
                    include_missing=False, 
                    score=score, 
                    fillna = fillna,
                    equal_freq=equal_freq,
                    withSummary=withSummary,
                    add_func=add_func,
                    spec_values=spec_values
                )
                if display_metric_list is not None:
                    gains_table_dict[score] = gains_table_dict[score][display_metric_list]
                gains_table_dict[score] = gains_table_dict[score].reset_index(drop=False)
                
            else:
                if new_oot_df.query(f"{score} > 0").shape[0] > 10:
                    oot_by_group = get_gains_table(
                        data=new_oot_df, 
                        dep=dep, 
                        ascending=True,
                        tree_binning=tree_binning,
                        nbins=grp_nbins, 
                        precision=digit, 
                        min_bin_prop=min_bin_prop, 
                        include_missing=False, 
                        score=score,
                        fillna = fillna,
                        equal_freq=equal_freq, 
                        sync_range=sync_range, 
                        retSummary=False,
                        grp_name=grp_name, 
                        min_data_size=min_data_size,
                        withSummary=withSummary,
                        spec_values=spec_values,
                        add_func=add_func
                    )
                    if display_metric_list is not None:
                        oot_by_group = oot_by_group[[*display_metric_list, grp_name]]
                    oot_by_group = oot_by_group.reset_index(drop=False)
                    
                    grp_metric_dict = {}
                    for value in [x for x in grp_disp_metric if x not in ['PROP']]:
                        grp_metric_dict[value] = oot_by_group\
                            .reset_index(drop=False)\
                            .pivot_table(index=['_bin_num', '_bin_range'], columns=[grp_name], values=value, margins=True, margins_name="Grand_Total")
                    
                    if 'PROP' in grp_disp_metric:
                        tot_cnt = oot_by_group\
                            .reset_index(drop=False)\
                            .pivot_table(index=['_bin_num', '_bin_range'], columns=[grp_name], values=['N'], margins=True, margins_name="Grand_Total", aggfunc='sum')
                        
                        if tot_cnt.shape[0] > 0:
                            grp_metric_dict['PROP'] = tot_cnt.iloc[0:tot_cnt.shape[0]:] / np.array(tot_cnt.iloc[-1:].T['Grand_Total'].tolist())
                            grp_metric_dict['PROP'].columns = grp_metric_dict['PROP'].rename(columns={'N': ''}).columns.map("".join)
                    
                    fnl_grp_gains_res = []
                    for colname, gains in grp_metric_dict.items():
                        gains = gains.copy()
                        gains['variable'] = colname
#                         gains = gains.reset_index(drop=False)
                        if disp:
                            from IPython.display import display
                            display(gains)
                        fnl_grp_gains_res.append(gains)
                    fnl_grp_gains_res = pd.concat(fnl_grp_gains_res)
                    
                    gains_table_dict[score] = oot_by_group
        
        fnl_gains_res = []
        for score_name, gains_res in gains_table_dict.items():
            gains_res = gains_res.copy()
            gains_res['score_name'] = score_name
            fnl_gains_res.append(gains_res)
        
        fnl_gains_res = pd.concat(fnl_gains_res)
        
        return fnl_gains_res
    
    def get_cust_metric_summary(
        self,
        eval_metrics: Optional[List[str]] = None,
        metric_agg_func: Union[str, Callable] = 'mean',
        withSummary: bool = True,
        spec_values = []
    ) -> pd.DataFrame:
        """
        Generate summary for custom evaluation metrics.
        
        This method calculates gains tables for custom metrics defined in
        self.eval_metrics, allowing analysis of various feature distributions
        across score bins.
        
        Parameters
        ----------
        eval_metrics : list, optional
            List of metrics to evaluate. If None, uses self.eval_metrics.
        metric_agg_func : str or callable, optional
            Aggregation function for metrics. Default is 'mean'.
        withSummary : bool, optional
            Include summary row. Default is True.
            
        Returns
        -------
        pandas.DataFrame
            Custom metric summary with score names.
        """
        data = self.data.copy()
        tgt_name = self.dep
        nbins = self.nbins
        precision = self.precision
        min_bin_prop = self.min_bin_prop
        include_missing = self.include_missing
        equal_freq = self.equal_freq
        
        score_list = [self.base_score] + self.comp_scrlist
        
        if eval_metrics is None:
            eval_metrics = self.eval_metrics
        
        cust_metric_res = {}
        for score in score_list:
            cust_metric_res[score] = get_gains_table_by_cust_metrics(
                data=data,
                dep=tgt_name,
                nbins=nbins,
                precision=precision,
                min_bin_prop=min_bin_prop,
                include_missing=include_missing,
                score=score,
                tree_binning=self.tree_binning,
                equal_freq=equal_freq,
                fillna=-999999,
                spec_values=spec_values,
                ascending=True,
                eval_metrics=eval_metrics,
                metric_agg_func=metric_agg_func,
                withSummary=withSummary
            )
        
        cust_gains_fnl_res = []
        for score_name, res in cust_metric_res.items():
            res = res.copy()
            res['score_name'] = score_name
            cust_gains_fnl_res.append(res)
        
        cust_gains_fnl_res = pd.concat(cust_gains_fnl_res)
        
        return cust_gains_fnl_res
    
    def get_cross_risk_summary(
        self,
        cross_agg_dict: Optional[Dict] = None,
        nbins: int = 5,
        equal_freq: Optional[bool] = None,
        disp: bool = True,
        spec_values = []
    ) -> pd.DataFrame:
        """
        Generate cross-risk analysis summary between base and comparison scores.
        
        This method creates a comprehensive cross-tabulation showing risk metrics
        across different score ranges for both base and comparison scores.
        
        Parameters
        ----------
        cross_agg_dict : dict, optional
            Dictionary of column names and aggregation functions.
        nbins : int, optional
            Number of bins. Default is 5.
        equal_freq : bool, optional
            Use equal frequency binning. If None, uses self.equal_freq.
        disp : bool, optional
            Whether to display results. Default is True.
            
        Returns
        -------
        pandas.DataFrame
            Cross-risk summary with base score range, comparison score range, and metrics.
        """
        data = self.data.copy()
        dep = self.dep
        min_bin_prop = self.min_bin_prop
        precision = self.precision
        tree_binning = self.tree_binning
        equal_freq = equal_freq if equal_freq is not None else self.equal_freq
        fillna = self.missing_rate_ref
        include_missing = self.include_missing
        
        base_score = self.base_score
        compare_scrlist = self.comp_scrlist
        
        if cross_agg_dict is None:
            cross_agg_dict = self.cross_agg_dict.copy()
        
        multi_scr_res = {}
        for compare_scr in compare_scrlist:
            logger.info(compare_scr)
            score_list = [base_score, compare_scr]
            score_query = " and ".join([s + " > 0" for s in score_list])
            
            prepared_data = data.query(score_query)
            
            if prepared_data.empty:
                return pd.DataFrame()  # 或跳过该组
            
            tot_cnt = prepared_data[score_list].shape[0]
            cross_agg_dict_copy = cross_agg_dict.copy()
            cross_agg_dict_copy.update({'flow_id': ['count', lambda x: x.count() / tot_cnt]})
            
            cross_res = {}
            for colname, aggfunc in cross_agg_dict_copy.items():
                if data.shape[0] > nbins:
                    cross_res[colname] = cross_risk(
                        data=prepared_data,
                        score_list=[base_score, compare_scr],
                        dep=dep,
                        nbins=nbins,
                        agg_col=colname,
                        precision=precision,
                        min_bin_prop=min_bin_prop,
                        include_missing=include_missing,
                        equal_freq=equal_freq,
                        binning_numeric=[True, True],
                        tree_binning=tree_binning,
                        agg_func=aggfunc,
                        fillna=fillna,
                        spec_values=spec_values
                    )
            
            fnl_res = []
            for colname, res in cross_res.items():
                res = res.copy()
                res[('', 'eval_metric', '')] = colname
                res = res.rename(columns={"<lambda>": "", "count": "n"})
                if disp:
                    from IPython.display import display
                    display(res)
                fnl_res.append(res)
            
            fnl_res = pd.concat(fnl_res)
            
            fnl_res[('', 'score_name', '')] = compare_scr
            multi_scr_res[compare_scr] = fnl_res
        
        combined_cross_res = []
        for score_name, cross_data in multi_scr_res.items():
            cross_data = cross_data.copy()
            cross_data.columns = cross_data.columns.map(
                lambda x: f"{x[0]}_{x[1] if len(str(x[1])) >= 2 else ('0' + str(x[1]))}_{x[2]}".strip("_")
            )
            cross_data.index = cross_data.index.map(
                lambda x: f"{x[0] if len(str(x[0])) >= 2 else ('0' + str(x[0]))}_{x[1]}".strip("_")
            )
            cross_data = cross_data.reset_index(drop=False).melt(
                id_vars=['index', 'eval_metric', 'score_name'],
                var_name='comp_scr_range'
            ).rename(columns={"index": "base_scr_range"})
            combined_cross_res.append(cross_data)
        combined_cross_res = pd.concat(combined_cross_res)
        
        return combined_cross_res
    
    def multi_subset_wrapper(
        self,
        condition_dict: Optional[Dict[str, str]] = None,
        subset_var_name: str = 'eval_subset',
        func: Optional[Callable] = None,
        min_subset_size = 10,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply evaluation function across multiple data subsets.
        
        This method iterates over predefined subset conditions, filters the data
        accordingly, and applies the evaluation function to each subset.
        
        Parameters
        ----------
        condition_dict : dict, optional
            Dictionary mapping subset names to query conditions.
            If None, uses self.subset_condition_dict.
        subset_var_name : str, optional
            Column name for subset identifier in results. Default is 'eval_subset'.
        func : callable, optional
            Function to apply to each subset. If None, uses model_perf_compare.
        **kwargs
            Additional keyword arguments passed to the evaluation function.
            
        Returns
        -------
        pandas.DataFrame
            Combined results from all subsets.
        """
        if condition_dict is None:
            condition_dict = self.subset_condition_dict
        
        original_data = self.data.copy()
        if func is None:
            func = self.model_perf_compare
        
        fnl_subset_results = []
        
        if condition_dict:
            for group_value, query in condition_dict.items():
                logger.info(f"INFO:: Multi_Subset_Eval: Running Subset {group_value}")
                
                subset_data = self.data.query(query).copy() if query != "" else self.data.copy()
                subset_data_size = subset_data.shape[0]
                
                if subset_data_size > min_subset_size:
                    self.data = subset_data
                    
                    subset_result = func(**kwargs)
                    if subset_result is not None and not subset_result.empty:
                        subset_result = subset_result.copy()
                        subset_result[subset_var_name] = group_value
                        fnl_subset_results.append(subset_result)
                    else:
#                         fnl_subset_results.append(pd.DataFrame({subset_var_name: [group_value]}))
                        self.data = original_data
                        continue
                
                    self.data = original_data
                    
        
        if fnl_subset_results:
            return pd.concat(fnl_subset_results)
        
        return pd.DataFrame({subset_var_name: [group_value]})
    
    def multi_ylabel_wrapper(
        self,
        ylabels: Optional[List[str]] = None,
        ylabel_var_name: str = 'eval_ylabel',
        eval_func: Optional[Callable] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply evaluation function across multiple target labels.
        
        This method iterates over different target/label columns, temporarily
        updating the dependent variable for each iteration.
        
        Parameters
        ----------
        ylabels : list, optional
            List of target column names. If None, uses self.eval_ylabels.
        ylabel_var_name : str, optional
            Column name for label identifier in results. Default is 'eval_ylabel'.
        eval_func : callable, optional
            Function to apply for each label. If None, uses model_perf_compare.
        **kwargs
            Additional keyword arguments passed to the evaluation function.
            
        Returns
        -------
        pandas.DataFrame
            Combined results from all labels.
        """
        original_dep = self.dep
        original_data = self.data.copy()
        if eval_func is None:
            eval_func = self.model_perf_compare
        
        fnl_ylabel_results = []
        
        if ylabels is None:
            ylabels = self.eval_ylabels
        
        for current_dep in ylabels:
            logger.info(f"INFO:: Multi_yLabel_Eval: Running Label {current_dep}")
            
            input_data = self.data.query(f"{current_dep} == {current_dep}").copy()
            subset_data_size = input_data.shape[0]
            
            if subset_data_size > 0:
                self.dep = current_dep
                self.data = input_data
                
                subset_result = eval_func(**kwargs)
                subset_result = subset_result.copy()
                subset_result[ylabel_var_name] = current_dep
                fnl_ylabel_results.append(subset_result)
        
                self.dep = original_dep
                self.data = original_data
        
        if fnl_ylabel_results:
            return pd.concat(fnl_ylabel_results)
        return pd.DataFrame()
    
    def multi_group_wrapper(
        self,
        group_name: Optional[str] = None,
        group_var_name: str = 'group_name',
        group_eval_func: Optional[Callable] = None,
        min_subset_size: int = 10,
        **kwargs
    ) -> pd.DataFrame:
        """
        Apply evaluation function across multiple groups.
        
        This method splits data by a grouping column and applies the evaluation
        function to each group separately.
        
        Parameters
        ----------
        group_name : str, optional
            Column name to group by. If None, processes all data together.
        group_var_name : str, optional
            Column name for group identifier in results. Default is 'group_name'.
        group_eval_func : callable, optional
            Function to apply to each group. If None, uses model_perf_compare.
        min_subset_size : int, optional
            Minimum data size required to process a group. Default is 10.
        **kwargs
            Additional keyword arguments passed to the evaluation function.
            
        Returns
        -------
        pandas.DataFrame
            Combined results from all groups.
        """
        original_data = self.data.copy()
        
        if group_name is None:
            return group_eval_func(**kwargs) if group_eval_func else pd.DataFrame()
        
        group_value_list = list(set(original_data[group_name].unique().tolist()))
        
        if group_eval_func is None:
            group_eval_func = self.model_perf_compare
        
        fnl_group_results = []
        for group_value in group_value_list:
            logger.info(f"INFO:: Multi_Group_Eval: Running Group {group_value}")
            
            input_data = original_data.query(f"{group_name} == '{group_value}'").copy()
            subset_data_size = input_data.shape[0]
            
            if subset_data_size > min_subset_size:
                self.data = input_data
                
                subset_result = group_eval_func(**kwargs)
                subset_result = subset_result.copy()
                subset_result[group_var_name] = group_value
                fnl_group_results.append(subset_result)
        
                self.data = original_data
        
        if fnl_group_results:
            return pd.concat(fnl_group_results)
        return pd.DataFrame()
    
    def multi_dim_eval(
        self,
        condition_dict: Optional[Dict[str, str]] = None,
        eval_ylabels: Optional[List[str]] = None,
        grp_namelist: Optional[List[str]] = None,
        eval_func: Optional[Callable] = None,
        subset_var_name: str = 'eval_subset',
        ylabel_var_name: str = 'eval_ylabel',
        var_name: str = 'group_name',
        value_name: str = 'group_value',
        **kwargs
    ) -> pd.DataFrame:
        """
        Perform multi-dimensional evaluation across subsets, labels, and groups.
        
        This method combines multi_subset_wrapper and multi_ylabel_wrapper to
        evaluate model performance across different dimensions.
        
        Parameters
        ----------
        condition_dict : dict, optional
            Subset conditions. If None, uses self.subset_condition_dict.
        eval_ylabels : list, optional
            Target labels. If None, uses self.eval_ylabels.
        grp_namelist : list, optional
            Group columns. If None, uses self.grp_namelist.
        eval_func : callable, optional
            Evaluation function. If None, uses model_perf_compare.
        subset_var_name : str, optional
            Column name for subset identifier.
        ylabel_var_name : str, optional
            Column name for label identifier.
        var_name : str, optional
            Column name for group name in results.
        value_name : str, optional
            Column name for group value in results.
        **kwargs
            Additional keyword arguments.
            
        Returns
        -------
        pandas.DataFrame
            Multi-dimensional evaluation results.
        """
        if condition_dict is None:
            condition_dict = self.subset_condition_dict
            
        if eval_ylabels is None:
            eval_ylabels = self.eval_ylabels
            
        if grp_namelist is None:
            grp_namelist = self.grp_namelist
            
        if eval_func is None:
            eval_func = self.model_perf_compare
        
        fnl_perf_res = []
        for group_name in grp_namelist:
            logger.info(f"INFO:: Multi_Dim_Eval: Running Group {group_name}")
            
            sig = inspect.signature(eval_func)
            parameters = sig.parameters
            
            if 'grp_name' in parameters:
                sub_perf_res = self.multi_subset_wrapper(
                    condition_dict=condition_dict,
                    subset_var_name=subset_var_name,
                    func=self.multi_ylabel_wrapper, 
                    ylabels=eval_ylabels, 
                    ylabel_var_name=ylabel_var_name,
                    eval_func=eval_func, 
                    grp_name=group_name,
                    **kwargs
                )
            else:
                sub_perf_res = self.multi_subset_wrapper(
                    condition_dict=condition_dict,
                    subset_var_name=subset_var_name,
                    func=self.multi_ylabel_wrapper, 
                    ylabels=eval_ylabels, 
                    ylabel_var_name=ylabel_var_name,
                    eval_func=eval_func, 
                    **kwargs
                )
            
            sub_perf_res = sub_perf_res.copy()
            sub_perf_res[var_name] = group_name
            fnl_perf_res.append(sub_perf_res)
        
        fnl_perf_res = pd.concat(fnl_perf_res)
        
        if 'grp_name' in parameters:
            if all(x in fnl_perf_res.columns for x in grp_namelist):
                fnl_perf_res[value_name] = fnl_perf_res[grp_namelist].bfill(axis=1).iloc[:, 0]
                fnl_perf_res = fnl_perf_res.drop(columns=grp_namelist)
        
        return fnl_perf_res
    
    def cross_perf_eval(
        self,
        bad_list: List[str],
        scr_list: List[str],
        data: Optional[pd.DataFrame] = None,
        eval_metric: List[str] = None,
        melt: bool = True
    ) -> pd.DataFrame:
        """
        Cross-evaluate model performance across multiple bad flags and scores.
        
        This method calculates performance metrics for all combinations of
        bad flags and scores, useful for comparing model behavior across
        different definitions of "bad" outcomes.
        
        Parameters
        ----------
        bad_list : list
            List of bad/negative outcome column names.
        scr_list : list
            List of score column names.
        data : pandas.DataFrame, optional
            Data to use. If None, uses self.data.
        eval_metric : list, optional
            Metrics to extract. If None, uses ['AUC', 'KS', 'N'].
        melt : bool, optional
            Whether to melt the pivot table. Default is True.
            
        Returns
        -------
        pandas.DataFrame
            Cross performance evaluation results.
        """
        if eval_metric is None:
            eval_metric = ['AUC', 'KS', 'N']
        
        if data is None:
            data = self.data.copy()
        
        score_query = " and ".join([s + " > 0" for s in scr_list])
        bad_query = " and ".join([s + " == " + s for s in bad_list])
        
        data = data.query(bad_query)
        data = data.query(score_query)
        
        if data.shape[0] > self.nbins:
            fnl_res = []
            for bad in bad_list:
                for scr in scr_list:
                    perf = get_perf_summary(
                        None,
                        None,
                        data,
                        tgt_name=bad,
                        scr_name=scr,
                        to_show=False,
                        display=False,
                        dist_bins=self.nbins,
                        pct_bins=self.nbins,
                        precision=self.precision,
                        min_bin_prop=self.min_bin_prop,
                        equal_freq=True
                    )
                    perf = perf.copy()
                    perf['eval_ylabel'] = bad
                    perf['score_name'] = scr
                    fnl_res.append(perf)
            
            fnl_res = pd.concat(fnl_res)
            
            summary = fnl_res.pivot_table(index="score_name", columns='eval_ylabel', values=eval_metric)
            summary.columns = summary.columns.map(lambda x: f"{x[0]}_{x[1]}")
            
            if melt:
                summary = summary.reset_index(drop=False).melt(id_vars=['score_name'])
            
            return summary
        
        return pd.DataFrame(columns=['eval_ylabel', 'score_name'])
    
    def run_variable_analysis_summary(
        self,
        varlist: List[str],
        data: Optional[pd.DataFrame] = None,
        dep: Optional[str] = None,
        nbins: Optional[int] = None,
        equal_freq: Optional[bool] = None,
        min_bin_prop: Optional[float] = None,
        precision: Optional[int] = None,
        chi2_method: Optional[bool] = None,
        chi2_p: Optional[float] = None,
        init_equi_bins: Optional[int] = None,
        tree_binning: Optional[bool] = None,
        include_missing: Optional[bool] = None,
        missing_rate_ref: Optional[Union[int, float]] = None,
        seed: Optional[int] = None,
        spec_values = []
    ) -> pd.DataFrame:
        """
        Run comprehensive variable analysis and generate summary report.
        
        This method performs binning analysis on specified variables,
        calculating metrics like IV (Information Value) and chi-square statistics.
        
        Parameters
        ----------
        varlist : list
            List of variable names to analyze.
        data : pandas.DataFrame, optional
            Data to use. If None, uses self.data.
        dep : str, optional
            Target variable name. If None, uses self.dep.
        nbins : int, optional
            Number of bins. If None, uses self.nbins.
        equal_freq : bool, optional
            Use equal frequency binning. If None, uses self.equal_freq.
        min_bin_prop : float, optional
            Minimum bin proportion. If None, uses self.min_bin_prop.
        precision : int, optional
            Decimal precision. If None, uses self.precision.
        chi2_method : bool, optional
            Use chi-square method. If None, uses self.chi2_method.
        chi2_p : float, optional
            Chi-square p-value. If None, uses self.chi2_p.
        init_equi_bins : int, optional
            Initial equal bins. If None, uses self.init_equi_bins.
        tree_binning : bool, optional
            Use tree binning. If None, uses self.tree_binning.
        include_missing : bool, optional
            Include missing values. If None, uses self.include_missing.
        missing_rate_ref : int or float, optional
            Missing reference. If None, uses self.missing_rate_ref.
        seed : int, optional
            Random seed. If None, uses self.seed.
            
        Returns
        -------
        pandas.DataFrame
            Variable analysis summary with IV and other metrics.
        """
        if data is None:
            data = self.data.copy()
        
        if dep is None:
            dep = self.dep
        
        nbins = nbins if nbins is not None else self.nbins
        equal_freq = equal_freq if equal_freq is not None else self.equal_freq
        min_bin_prop = min_bin_prop if min_bin_prop is not None else self.min_bin_prop
        precision = precision if precision is not None else self.precision
        chi2_method = chi2_method if chi2_method is not None else self.chi2_method
        chi2_p = chi2_p if chi2_p is not None else self.chi2_p
        init_equi_bins = init_equi_bins if init_equi_bins is not None else self.init_equi_bins
        tree_binning = tree_binning if tree_binning is not None else self.tree_binning
        include_missing = include_missing if include_missing is not None else self.include_missing
        missing_rate_ref = missing_rate_ref if missing_rate_ref is not None else self.missing_rate_ref
        seed = seed if seed is not None else self.seed
        
        from Modeling_Tool.Feature.Feature_Insights import VarExtractionInsights

        varInsights = VarExtractionInsights(
            data=data,
            dep=dep,
            plot_path=None,
            nbins=nbins,
            equal_freq=equal_freq,
            min_bin_prop=min_bin_prop,
            precision=precision,
            chi2_method=chi2_method,
            chi2_p=chi2_p,
            init_equi_bins=init_equi_bins,
            tree_binning=tree_binning,
            include_missing=include_missing,
            seed=seed,
            missing_rate_ref=missing_rate_ref,
            spec_values=spec_values
        )
        
        data = data.copy()
        for var in varlist:
            data[var] = data[var].fillna(missing_rate_ref)
        
        var_summary = varInsights.get_var_analysis_report(data=data, varlist=varlist, dep=dep, iv_cut=0)
        
        return var_summary
    
    
    def pipe(self, data=None):
            """
            返回一个 EvaluationPipeline 对象，用于链式分组和子集评估。

            Parameters
            ----------
            data : pd.DataFrame, optional
                指定数据，默认使用 self.data

            Returns
            -------
            EvaluationPipeline
            """
            return EvaluationPipeline(self, data)